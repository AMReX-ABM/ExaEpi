#!/usr/bin/env python
"""The bundle's Laplace replicates, scored with resolve_spread.py's measures.

resolve_spread.py reports block-group population spread, fit to published counts in SE units, and
household turnover between draws for re-solved populations. This computes the same numbers for the
20 ridge-regularised Laplace replicates already in nm_reps20_fixed.upb, so the two ways of varying
the population sit on one scale. Population per donor is est_ind's `population` column, as in
resolve_spread.py; the allocation is the expected household count per cell, before integerization.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import read_bundle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composition  # noqa: E402
from gpu_solve import load  # noqa: E402


def strings(b, name):
    blob, off = b[name + ".blob"].tobytes(), b[name + ".offsets"]
    return [blob[int(a):int(e)].decode() for a, e in zip(off[:-1], off[1:])]


def expected(b, p, rep):
    bg_off, don_off = b["almat.puma_bg_offset"], b["almat.puma_donor_offset"]
    b0, b1 = int(bg_off[p]), int(bg_off[p + 1])
    d0, d1 = int(don_off[p]), int(don_off[p + 1])
    n_bg, n_don = b1 - b0, d1 - d0
    # Cells are stored donor-major, PUMA after PUMA.
    start = int(sum((don_off[i + 1] - don_off[i]) * (bg_off[i + 1] - bg_off[i]) for i in range(p)))
    q = b["almat.values"][rep, start:start + n_don * n_bg].reshape(n_don, n_bg)
    return q.astype(np.float64) / 65535.0 * b["almat.scale"][rep, d0:d1][:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500804"])
    ap.add_argument("--bundle", default="./nm_reps20_fixed.upb")
    ap.add_argument("--point", default="./nm_full.upb")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    b, bp = read_bundle(args.bundle), read_bundle(args.point)
    names = strings(b, "almat.puma")
    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    for fips in args.pumas:
        p = names.index(fips)
        lp = load(fips, key, "./llcache_minimal")
        pup = lp["pup"]
        persons = pup.est_ind["population"].to_numpy(float)
        bgs = pup.est_g2.index.astype(str)
        pub = np.asarray(pup.est_g2, float)[:, 0]
        pse = lp["se2"][:, 0]
        acs_cv = (moe["se"] / moe["acs"]).reindex(bgs).to_numpy()
        als = [expected(b, p, r) for r in range(b["almat.values"].shape[0])]
        assert als[0].shape == (len(persons), len(bgs)), "donor/block-group order mismatch"
        al0 = expected(bp, strings(bp, "almat.puma").index(fips), 0)
        pops = np.array([a.T @ persons for a in als])
        m = pops.mean(0)
        cv = pops.std(0, ddof=1) / np.where(m > 0, m, np.nan)
        ok = np.isfinite(cv) & np.isfinite(acs_cv)
        z = (pops - pub) / np.where(pse > 0, pse, np.nan)
        zm = (m - pub) / np.where(pse > 0, pse, np.nan)
        turn = np.mean([0.5 * np.abs(x - y).sum() / y.sum() for x, y in zip(als[1:], als[:-1])])
        shift = np.mean([0.5 * np.abs(x - al0).sum() / al0.sum() for x in als])
        print(f"PUMA {fips}, {len(als)} Laplace replicates: CV {np.median(cv[ok]):.4f}, ratio "
              f"{np.median(cv[ok]) / np.median(acs_cv[ok]):.3f}, z rms "
              f"{np.sqrt(np.nanmean(z ** 2)):.3f}, >MOE {100 * np.nanmean(np.abs(z) > 1.645):.1f}%,"
              f" |bias| z {np.nanmedian(np.abs(zm)):.3f}, turnover {100 * turn:.2f}%, shift from "
              f"point estimate {100 * shift:.2f}%")
        print(composition.header())
        print("  " + composition.line(composition.measure(
            als, lp["C"], lp["cols"], lp["A1"], lp["est1"], lp["est2"], lp["se1"], lp["se2"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
