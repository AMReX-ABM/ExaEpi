#!/usr/bin/env python
"""Does the 123-constraint minimal set shrink the replicate spread the bundle can express?

The constraint reduction was justified on two grounds, both measured: the Hessian is (dual x dual)
with dual proportional to the constraint count, so 123 against up_expanded's 298 turns ~42 GB into
~8.8 GB; and the minimal set fits the variables ExaEpi actually reads slightly BETTER
(in_MOE 1.0000 vs 0.9996, median RAE 0.1421 vs 0.1466).

Neither of those is the property the bundle depends on. What matters for varying the population
between runs is how much SPREAD the replicates carry, and a smaller constraint set means a smaller
dual, a tighter posterior, and therefore less of it. The record says replicates reach CV 0.1425
against an ACS CV of 0.1604 (ratio 0.889), but that was measured on up_expanded; the minimal set
came in at 0.0501 on PUMA 3500804.

If that gap is real it is a genuine tension rather than a bug: the reduction that made the solve
affordable also throws away most of the uncertainty the bundle exists to reproduce. Same PUMA,
same replicate count, same seed -- only the constraint selection differs.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

EXAEPI_MINIMAL = {
    "universe": True,
    "demographic": ["sex_age", "hhtype_hhsize"],
    "social": ["race"],
    "worker": ["sexnaics"],
    "student": ["grade"],
    "mobility": ["travel", "veh_occ"],
}


def run(fips, selection, cache_folder, n_reps, year, key):
    from livelike import acs, config
    from pymedm import PMEDM

    t0 = time.time()
    pup = acs.puma(
        fips, constraints_selection=selection,
        constraints_theme_order=config.up_constraints_theme_order,
        year=year, target_zone="bg", cache=True,
        cache_folder=cache_folder, censusapikey=key,
    )
    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
                n_reps=n_reps, random_state=1)
    pmd.solve()
    reps = np.asarray(pmd.almat_reps)
    hh = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(1).to_numpy()
    total = float(np.asarray(pup.est_g2)[:, 0].sum())
    pops = np.vstack([reps[i].T @ hh * total for i in range(reps.shape[0])])
    mean = pops.mean(axis=0)
    cv = pd.Series(pops.std(axis=0, ddof=1) / np.where(mean > 0, mean, np.nan),
                   index=pup.est_g2.index).dropna()
    n_con = pup.est_ind.shape[1]
    dual = n_con * (pup.est_g2.shape[0] + pup.est_g1.shape[0] + 1)
    return cv, n_con, dual, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--min_cache", default="./llcache_minimal")
    ap.add_argument("--full_cache", default="./llcache_expanded")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    args = ap.parse_args()

    from livelike import config

    key = os.environ.get("CENSUS_API_KEY") or None
    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")

    print(f"PUMA {args.fips}, {args.n_reps} replicates, random_state=1\n")
    print(f"{'selection':14s}{'cons':>6s}{'dual':>8s}{'CV med':>9s}{'ratio':>8s}{'time':>8s}")
    out = {}
    for name, sel, cache in [("minimal", EXAEPI_MINIMAL, args.min_cache),
                             ("up_expanded", config.up_expanded_constraints_selection,
                              args.full_cache)]:
        cv, n_con, dual, dt = run(args.fips, sel, cache, args.n_reps, 2019, key)
        acs_cv = (moe["se"] / moe["acs"]).reindex(cv.index.astype(str)).dropna()
        ratio = cv.median() / acs_cv.median()
        out[name] = (cv.median(), ratio)
        print(f"{name:14s}{n_con:6d}{dual:8d}{cv.median():9.4f}{ratio:8.3f}{dt:7.1f}s")
        print(f"{'':14s}{'':6s}{'':8s}  ACS CV on the same block groups: {acs_cv.median():.4f}")

    a, b = out["minimal"], out["up_expanded"]
    print(f"\nminimal carries {a[0] / b[0]:.2f}x the spread of up_expanded "
          f"({a[0]:.4f} vs {b[0]:.4f})")
    print("\nArm D established that a CV of 0.05 still moves epidemic outcomes (ICC 0.488,")
    print("p=0.006), so a reduced spread is not fatal -- but it is a smaller claim about")
    print("uncertainty than the 0.889 ratio on record, which was measured on up_expanded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
