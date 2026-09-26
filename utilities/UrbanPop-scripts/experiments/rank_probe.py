#!/usr/bin/env python
"""How many principal components capture the replicate allocation matrices' variation?

Option A ships N discrete replicate allocation matrices; option B ships their mean plus the top-k
principal components, and samples new matrices at runtime as a weighted sum. A component in
allocation space has exactly the same dimensions as an allocation matrix, so B is only smaller
than A when k is much less than N -- and whether it is depends entirely on the effective rank of
the replicate variation, which is an empirical property of the problem.

Generating replicates is free once the Hessian exists (n_reps=100 costs the same as n_reps=2), so
this draws a large set and reports the explained-variance spectrum.
"""

import argparse
import os

import numpy as np

EXAEPI_MINIMAL = {
    "universe": True,
    # hhtype_hhsize is required by homesim.synthesize, which integerises at the
    # household level and needs household sizes to expand households to persons.
    "demographic": ["sex_age", "hhtype_hhsize"],
    "social": ["race"],
    "worker": ["sexnaics"],
    "student": ["grade"],
    "mobility": ["travel", "veh_occ"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=100)
    ap.add_argument("--cache_folder", default="./cache_minimal")
    # Measured: New Mexico's block-diagonal allocation matrix is 3,533,563 cells across 18 PUMAs.
    ap.add_argument("--nm_cells", type=int, default=3_533_563)
    args = ap.parse_args()

    from livelike import acs, config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    pup = acs.puma(
        args.fips,
        constraints_selection=EXAEPI_MINIMAL,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder=args.cache_folder, censusapikey=key,
    )
    pmd = PMEDM(
        pup.year, pup.est_ind.index, pup.wt,
        pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
        n_reps=args.n_reps, random_state=1,
    )
    pmd.solve()

    reps = np.asarray(pmd.almat_reps)          # (n_reps, donors, block groups)
    n, d, b = reps.shape
    flat = reps.reshape(n, d * b)
    print(f"replicates {n}   allocation matrix {d} x {b} = {d * b} cells")

    centred = flat - flat.mean(axis=0, keepdims=True)
    # Singular values of the centred replicate stack; squares are the variance per component.
    sv = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
    var = sv**2
    cum = np.cumsum(var) / var.sum()

    print("\ncumulative explained variance:")
    for k in [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, n - 1]:
        if k <= len(cum):
            print(f"  k={k:3d}: {cum[k - 1]:.4f}")

    ks = {t: int(np.searchsorted(cum, t) + 1) for t in (0.90, 0.95, 0.99)}
    print("\ncomponents needed:")
    for t, k in ks.items():
        print(f"  {t:.0%} of variance: k = {k}")

    # Size comparison for a whole New Mexico bundle, f32.
    one = args.nm_cells * 4 / 1e6
    print(f"\nNM allocation matrix, f32: {one:.1f} MB each")
    print(f"{'':22s}{'option A':>12s}{'option B':>12s}")
    for N in (20, 50, 100, 1000):
        for t, k in ks.items():
            if t != 0.95:
                continue
            a = N * one
            bsz = (1 + k) * one
            print(f"  N={N:<5d} (k={k:3d})   {a:9.0f} MB {bsz:9.0f} MB"
                  f"   {'B saves ' + format(a / bsz, '.1f') + 'x' if bsz < a else 'A smaller'}")
    print("\n(option B is unbounded in realizations; option A gives exactly N)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
