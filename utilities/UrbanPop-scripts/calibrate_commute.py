#!/usr/bin/env -S python -u
"""Fit a version-3 bundle's commute tables against CTPP (commute.r and commute.kern).

S3 corrects LODES toward how far people actually commute (popgen/commute.py). The two fitted
tables are properties of the data, not of a run, so they are fitted once here and stored:

  commute.kern  the time-band kernel P(band | d) / P(band): a lognormal straight-line speed model
                d = v t, t each worker's car-equivalent reported minutes (PUMS JWMNP by mode),
                v lognormal(v50, sigma), with (v50, sigma) chosen on a grid so the model's
                distance profile best matches CTPP's (binned KL).
  commute.r     reliability per distance band: starts at CTPP's commuter distance profile over
                LODES' job profile, then iterates r <- r * CTPP / realised, each time running the
                real S3 (popgen/workers.py) on one generated population, until the realised
                commuter profile matches CTPP's.

CTPP is the 2012-2016 tract-to-tract flow table (utilities/download_ctpp_flows.py). Its universe
includes people who work at home, counted in their own tract; those are removed from the
same-tract flows in proportion to the population's work-from-home share, so both sides count
commuters. Distances are chord lengths between tract internal points, as at runtime.

    calibrate_commute.py --bundle IN.upb --ctpp_flows ca_ctpp2016_tract_flows.csv \\
        --tract_shapefiles tl_2010_06_tract10.shp --load-alloc oracle_ca_s1.alloc --seed 1 \\
        --out OUT.upb
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_precompute as BP  # noqa: E402
import generate_exaepi as G  # noqa: E402
from popgen import bundle, cbp, commute, persons, placement, workers  # noqa: E402

FIT_EDGES = np.array([0, 2, 5, 10, 20, 35, 50, 75, 100, 150, 250, 400, np.inf])


def ctpp_commuters(path, points, wfh_share):
    """CTPP tract flows with chord km and commuter weights (same-tract flows less WFH)."""
    c = pd.read_csv(path)
    xs, xd = commute.tract_xyz(points, c.src.values), commute.tract_xyz(points, c.dst.values)
    c["km"] = np.sqrt(commute.chord2(xs, xd))
    same = c.flow[c.km == 0].sum() / c.flow.sum()
    c["w"] = c.flow.astype(float) * np.where(c.km == 0, max(0.0, (same - wfh_share) / same), 1.0)
    return c


def profile(b, km, w):
    e = b["commute.dist_edges"]
    return np.bincount(commute.bin_of(e, km), weights=w, minlength=len(e) - 1) / np.sum(w)


def fit_kernel(teff, ctpp):
    q = np.bincount(np.searchsorted(FIT_EDGES, ctpp.km.values, side="right") - 1, weights=ctpp.w.values,
                    minlength=len(FIT_EDGES) - 1)
    q /= q.sum()
    best = None
    for v50 in np.arange(20, 71, 2.5):
        for sg in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            k = commute.SpeedKernel(teff, v50, sg)
            p = k.bin_probs(FIT_EDGES)
            kl = float(np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(p, 1e-12))))
            if best is None or kl < best[0]:
                best = (kl, k)
    print(f"  kernel: v50 {best[1].v50} km/h, sigma {best[1].sigma}, KL to CTPP {best[0]:.4f}")
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ctpp_flows", required=True)
    ap.add_argument("--tract_shapefiles", nargs="+", required=True)
    ap.add_argument("--load-alloc", default=None, help="saved allocations (generate_exaepi.py --save-alloc) "
                                                       "instead of solving")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.003, help="stop at this profile TVD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    b = bundle.read(a.bundle)
    placements, _ = G.generate(b, a.seed, 0, verbose=False,
                               allocs_in=G.read_allocations(a.load_alloc) if a.load_alloc else None)
    P = persons.build(b, placement.expand(b, placements), a.seed, 0)
    tables = cbp.SizeTables(b)
    emp = P["employed"]
    com = np.flatnonzero(emp & (P["travel"] != commute.TRAVEL_WFH))
    wfh = float((P["travel"][emp] == commute.TRAVEL_WFH).mean())
    print(f"population: {len(P['bg'])} persons, {emp.sum()} employed, {wfh:.1%} work from home "
          f"({time.perf_counter() - t0:.0f} s)")

    points = commute.tract_points(a.tract_shapefiles)
    ctpp = ctpp_commuters(a.ctpp_flows, points, wfh)
    target = profile(b, ctpp.km.values, ctpp.w.values)

    kernel, kl = fit_kernel(commute.car_minutes(P["travel"][com], P["jwmnp"][com]), ctpp)
    b["commute.kern"] = kernel.table(b["commute.kern_edges"])

    lh = np.repeat(b["lodes.home_geoid"], np.diff(b["lodes.indptr"]))
    ld = b["lodes.dest_geoid"][b["lodes.indices"]]
    lodes_km = np.sqrt(commute.chord2(commute.xyz_of(b, lh), commute.xyz_of(b, ld)))
    lp = profile(b, lodes_km, b["lodes.data"].astype(float))
    r = np.where(lp > 0, target / np.maximum(lp, 1e-12), 1.0)
    far = commute.bin_of(b["commute.dist_edges"], np.array([100.5]))[0]
    log = []
    xh = commute.xyz_of(b, P["bg"][com])
    for it in range(a.iters):
        t = time.perf_counter()
        b["commute.r"] = r
        work, st = workers.allocate(b, P, tables, a.seed, 0)
        got = profile(b, np.sqrt(commute.chord2(xh, commute.xyz_of(b, work[com]))), np.ones(len(com)))
        tvd = float(0.5 * np.abs(got - target).sum())
        log.append(dict(it=it, tvd=tvd, far100=float(got[far:].sum()), r=r.tolist()))
        print(f"  iteration {it}: profile TVD {tvd:.4f}, > 100 km {got[far:].sum():.4f} (CTPP "
              f"{target[far:].sum():.4f}), demand moved {st['demand_moved'] / max(st['demanded'], 1):.3f} "
              f"({time.perf_counter() - t:.0f} s)")
        if tvd < a.tol:
            break
        r = r * np.where(got > 0, target / np.maximum(got, 1e-12), 1.0)

    raw = bundle.read_raw(a.bundle)
    meta = json.loads(raw.pop("meta").tobytes().decode())
    meta["commute"] = {"calibrated": True, "ctpp_flows": os.path.basename(a.ctpp_flows), "seed": a.seed,
                       "v50_kmh": kernel.v50, "sigma": kernel.sigma, "kernel_kl": kl,
                       "iterations": len(log), "profile_tvd": log[-1]["tvd"], "far100": log[-1]["far100"],
                       "ctpp_far100": float(target[far:].sum()), "wfh_share": wfh}
    bw = BP.BundleWriter()
    for name, arr in raw.items():
        bw.add(name, b["commute.r"] if name == "commute.r" else b["commute.kern"] if name == "commute.kern" else arr)
    bw.add_json("meta", meta)
    bw.write(a.out)
    print(f"wrote {a.out}: {json.dumps(meta['commute'])} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
