#!/usr/bin/env -S python -u
"""Generate a fresh ExaEpi population from a version-2 precompute bundle (reference generator).

This is the Python reference ("oracle") for ExaEpi's in-process population generation: the same
bundle, the same stages and the same keyed random draws the C++ port implements, so every C++
stage can be checked against it. It is also usable on its own -- its output is a `.bin` that
today's ExaEpi runs unchanged -- which gives per-run fresh populations before any C++ exists.

Stages implemented so far (popgen/):

    perturb     targets redrawn within their ACS standard errors (block groups first, tracts
                rebuilt from them) and the PUMS prior weights by a Bayesian bootstrap
    solve       P-MEDM re-solve per PUMA (float32 incremental L-BFGS, popgen/solver.py)
    place       TRS integerization per block group and expansion to persons (bg, h, p)

Every draw is keyed on (seed, rep, stage, global identifiers) through KR64, so the result depends
only on the bundle, the seed and the replicate number -- not on processing order.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from popgen import bundle, perturb, placement, solver  # noqa: E402
from popgen.problem import PumaProblem, puma_count  # noqa: E402


def generate(b, seed, rep=0, pumas=None, verbose=True):
    """Placements for every (or the named) PUMA: (bg_geoid, donor_global, count) plus stats."""
    names = bundle.strings(b, "solve.puma")
    pop_col = bundle.strings(b, "solve.constraints").index("population")
    parts, stats = [], []
    for p in range(puma_count(b)):
        if pumas and names[p] not in pumas:
            continue
        t0 = time.perf_counter()
        prob = PumaProblem(b, p)
        Y, y2 = perturb.perturbed_targets(prob, seed, rep)
        logq = perturb.log_prior(perturb.perturbed_prior(prob, seed, rep), prob.G)
        al, iters, gnorm = solver.solve(prob, Y, logq)
        bg, row, ct = placement.place_puma(b, prob, al, y2[:, pop_col], seed, rep)
        parts.append((bg, prob.donor_index[row].astype(np.int64), ct))
        stats.append(dict(puma=names[p], donors=prob.D, bgs=prob.G, iters=iters,
                          grad=gnorm, households=int(ct.sum()), seconds=time.perf_counter() - t0))
        if verbose:
            s = stats[-1]
            print(f"  PUMA {s['puma']}: {s['donors']} donors x {s['bgs']} bgs, {s['iters']} "
                  f"iterations, {s['households']} households, {s['seconds']:.2f} s", flush=True)
    placements = tuple(np.concatenate([x[i] for x in parts]) for i in range(3))
    return placements, stats


def compare(pers, feathers):
    """Totals, per-block-group population correlation and mean age against delivered feathers."""
    import polars as pl

    ref = pl.concat([pl.read_ipc(p, columns=["geoid", "pr_age"]) for p in feathers])
    rc = ref.group_by("geoid").agg(pl.len().alias("n"))
    ref_n = dict(zip(rc["geoid"].cast(pl.Int64).to_list(), rc["n"].to_list()))
    g, n = np.unique(pers["bg"], return_counts=True)
    keys = sorted(set(ref_n) | set(g.tolist()))
    gen_n = dict(zip(g.tolist(), n.tolist()))
    a = np.array([gen_n.get(k, 0) for k in keys])
    r = np.array([ref_n.get(k, 0) for k in keys])
    return dict(persons=int(a.sum()), delivered=int(r.sum()),
                corr=float(np.corrcoef(a, r)[0, 1]), ref_age=float(ref["pr_age"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--pumas", nargs="*", default=None, help="restrict to these PUMAs")
    ap.add_argument("--compare", default=None, help="glob of delivered feathers to score against")
    ap.add_argument("--out", default=None, help="write the person arrays to this .npz")
    args = ap.parse_args()

    t0 = time.perf_counter()
    b = bundle.read(args.bundle)
    print(f"bundle {args.bundle}: {puma_count(b)} PUMAs, {len(b['bg.geoid'])} block groups")
    placements, stats = generate(b, args.seed, args.rep, args.pumas)
    pers = placement.expand(b, placements)
    age = b["donors.age"][pers["src"]]
    print(f"\n{len(pers['bg'])} persons in {pers['n_households']} households, "
          f"{len(np.unique(pers['bg']))} block groups, mean age {age.mean():.2f}; "
          f"solve+place {sum(s['seconds'] for s in stats):.1f} s, total "
          f"{time.perf_counter() - t0:.1f} s")
    if args.compare:
        c = compare(pers, sorted(glob.glob(args.compare)))
        print(f"vs delivered: {c['persons']} vs {c['delivered']} persons "
              f"({100 * c['persons'] / c['delivered'] - 100:+.2f}%), per-block-group correlation "
              f"{c['corr']:.4f}, delivered mean age {c['ref_age']:.2f}")
    if args.out:
        np.savez_compressed(args.out, **{k: v for k, v in pers.items() if k != "n_households"})
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
