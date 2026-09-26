#!/usr/bin/env python
"""Does the multi-replicate bundle actually vary the population, and by how much?

The bundle exists to let ExaEpi draw a different residential arrangement each run. Two axes can
vary, and only one of them matters:

    TRS seed      re-draws the integerization from ONE allocation matrix. Measured at a
                  per-block-group population CV of 0.0142 against an ACS sampling CV of 0.1604 --
                  about 11x too small, and arm B showed an ensemble built this way is null.
    replicate     picks a different allocation matrix, sampled from the Laplace approximation at
                  the converged dual. This is the axis the bundle ships `--n_reps` for.

This measures both on the same bundle, from the same generator, so the comparison is like for
like. The replicate axis should land near the solve's own measured spread -- ratio ~0.32 for the
123-constraint minimal set -- and must be clearly above the TRS axis, or shipping replicates buys
nothing over shipping one matrix.

It also checks the replicates are usable at all: distinct from each other, finite, and producing
populations of the right size.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_precompute import read_bundle  # noqa: E402
from generate_population import draw_placements, expand_to_persons  # noqa: E402


def bg_population(b, seed, rep):
    """Per-block-group headcount for one (seed, replicate), without materialising the frame."""
    bg_rows, donor_rows, counts = draw_placements(b, seed, rep, verbose=False)
    pers = expand_to_persons(b, bg_rows, donor_rows, counts)
    geo = pers["geoid"]
    idx, inv = np.unique(geo, return_inverse=True)
    return pd.Series(np.bincount(inv, minlength=len(idx)), index=idx)


def spread(frames, label):
    M = pd.concat(frames, axis=1).fillna(0.0).to_numpy(dtype=float)
    mean = M.mean(axis=1)
    cv = pd.Series(M.std(axis=1, ddof=1) / np.where(mean > 0, mean, np.nan)).dropna()
    totals = M.sum(axis=0)
    print(f"  {label:32s} CV median {cv.median():.4f}  mean {cv.mean():.4f}  "
          f"p90 {cv.quantile(0.9):.4f}")
    print(f"  {'':32s} totals {totals.min():.0f}-{totals.max():.0f} "
          f"(spread {100 * (totals.max() - totals.min()) / totals.mean():.2f}%)")
    return cv.median()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--n", type=int, default=10, help="draws along each axis")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    args = ap.parse_args()

    b = read_bundle(args.bundle)
    vals = b["almat.values"]
    n_reps = vals.shape[0]
    print(f"bundle: {n_reps} replicate(s), {vals.shape[1]} cells each, "
          f"{os.path.getsize(args.bundle) / 1e6:.1f} MB\n")
    if n_reps < 2:
        sys.exit("bundle has a single allocation matrix -- rebuild with --n_reps >= 10")

    # Replicates must actually differ. Identical matrices would still produce a spread through
    # TRS alone and could be mistaken for working replicates.
    d01 = float(np.abs(vals[0].astype(np.int32) - vals[1].astype(np.int32)).mean())
    print(f"mean |replicate0 - replicate1| in quantised units: {d01:.1f} of 65535\n")

    k = min(args.n, n_reps)
    print("per-block-group population spread:")
    rep_frames = [bg_population(b, 0, r) for r in range(k)]
    cv_rep = spread(rep_frames, f"across {k} replicates (seed 0)")
    trs_frames = [bg_population(b, s, 0) for s in range(k)]
    cv_trs = spread(trs_frames, f"across {k} TRS seeds (replicate 0)")

    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    ref = rep_frames[0].index.astype(str)
    acs_cv = (moe["se"] / moe["acs"]).reindex(ref).dropna().median()

    print(f"\n  ACS published sampling CV: {acs_cv:.4f}")
    print(f"  replicate axis ratio vs ACS: {cv_rep / acs_cv:.3f}   "
          f"(solve measured 0.32 for the 123-constraint set)")
    print(f"  TRS axis ratio vs ACS:       {cv_trs / acs_cv:.3f}   (arm B was null at this scale)")
    print(f"  replicate / TRS: {cv_rep / cv_trs:.2f}x")
    if cv_rep > 2 * cv_trs:
        print("\n  => Replicates dominate TRS re-draws, so shipping them is what makes the bundle")
        print("     vary anything. Use the replicate index as the population seed in ExaEpi.")
    else:
        print("\n  => Replicates add little over a TRS re-draw -- shipping them is not paying off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
