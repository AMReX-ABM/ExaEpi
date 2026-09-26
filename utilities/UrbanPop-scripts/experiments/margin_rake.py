#!/usr/bin/env python
"""Can ExaEpi get unlimited residential realizations from margins alone, instead of N matrices?

The plan's goal is "complete randomization per ExaEpi run with no pre-generated realizations", but
the measurements so far do not deliver it. A single allocation matrix plus runtime TRS moves
per-block-group population by a CV of 0.0044 against an ACS sampling CV of 0.1604 -- 36x too small,
and the ensemble built that way was null. Real variation needs replicate allocation matrices, and
those are discrete artifacts: ship N, get N. At ~5 MB each quantised, N=20 is a 100 MB section.

Storing the Laplace parameters instead, so the runtime could draw its own, needs H^-1 at
(dual x dual) -- ~676 MB per PUMA at 123 constraints. A low-rank factor will not rescue it either:
the replicate spectrum is nearly flat (k=88 of 99 components for 95% of variance).

This tests a third option that would be both smaller and unbounded. P-MEDM's replicates express
uncertainty that ultimately comes from the ACS margins' own sampling error, and the bundle can
carry those margins and their standard errors directly:

    123 constraints x 1445 block groups x (estimate, standard error) x 4 bytes = 1.4 MB

At runtime ExaEpi would draw perturbed margins ~ N(est, se) and rake the PUMS donor weights onto
them -- unlimited realizations, and a raking kernel rather than an LBFGS solve to port, which is
the same kernel the worker allocator already needs.

The thing that could break it is correlation structure. P-MEDM samples the dual's posterior, which
borrows strength across block group, tract and PUMA, so its replicates come out TIGHTER than the
raw margins: measured CV 0.1425 against ACS 0.1604, a ratio of 0.889. Perturbing each margin
independently ignores that hierarchy and should over-disperse. This measures by how much, and
whether raking still lands inside the ACS margins of error at all.
"""

import argparse
import os

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


def rake_bg(X, w0, target, n_iter=80, tol=1e-8):
    """Rake donor weights onto one block group's constraint margins.

    Multiplicative updates, one constraint at a time: scale the donors contributing to constraint k
    so that constraint's weighted total hits its target, then move to the next. This is IPF
    generalised to a non-square design matrix, and it is what P-MEDM reduces to as the penalties
    on the soft constraints go to zero.

    Returns (weights, max relative margin error). The error matters because perturbed margins are
    not guaranteed mutually consistent -- a draw can ask for more 5-year-olds than household sizes
    allow -- and raking then oscillates instead of converging. Reporting it keeps that visible
    rather than silently returning whatever the last sweep produced.
    """
    w = w0.astype(np.float64).copy()
    nz = [np.flatnonzero(X[:, k] > 0) for k in range(X.shape[1])]
    for _ in range(n_iter):
        worst = 0.0
        for k in range(X.shape[1]):
            idx = nz[k]
            if len(idx) == 0 or target[k] <= 0:
                continue
            cur = float(w[idx] @ X[idx, k])
            if cur <= 0:
                continue
            f = target[k] / cur
            worst = max(worst, abs(f - 1.0))
            w[idx] *= f
        if worst < tol:
            break
    err = 0.0
    for k in range(X.shape[1]):
        if target[k] > 0:
            cur = float(w[nz[k]] @ X[nz[k], k])
            err = max(err, abs(cur - target[k]) / target[k])
    return w, err


def bg_pop(w_by_bg, hh_people):
    """Expected people per block group: each donor household's weight times its own size."""
    return np.array([float(w @ hh_people) for w in w_by_bg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--cache_folder", default="./llcache_minimal")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--max_bg", type=int, default=0, help="limit block groups, 0 = all")
    args = ap.parse_args()

    from livelike import acs, config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    pup = acs.puma(
        args.fips, constraints_selection=EXAEPI_MINIMAL,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder=args.cache_folder, censusapikey=key,
    )
    X = np.asarray(pup.est_ind, dtype=np.float64)
    est = np.asarray(pup.est_g2, dtype=np.float64)
    se = np.asarray(pup.se_g2, dtype=np.float64)
    w0 = np.asarray(pup.wt, dtype=np.float64)
    hh_people = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(1).to_numpy()
    n_bg, n_con = est.shape
    print(f"PUMA {args.fips}: {X.shape[0]} donors, {n_con} constraints, {n_bg} block groups")

    sel = np.arange(n_bg if args.max_bg <= 0 else min(args.max_bg, n_bg))

    # --- reference: P-MEDM replicates -------------------------------------------------------
    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
                n_reps=args.n_reps, random_state=1)
    pmd.solve()
    reps = np.asarray(pmd.almat_reps)
    total_pop = float(est[:, 0].sum())
    pops_pmedm = np.vstack([reps[i].T @ hh_people * total_pop for i in range(reps.shape[0])])
    cv_pmedm = pd.Series(
        pops_pmedm.std(axis=0, ddof=1) / np.where(pops_pmedm.mean(axis=0) > 0,
                                                  pops_pmedm.mean(axis=0), np.nan),
        index=pup.est_g2.index).dropna()

    # --- candidate: perturb the published margins, rake onto them ---------------------------
    rng = np.random.default_rng(1)
    pops_rake, errs = [], []
    for r in range(args.n_reps):
        ws, bad = [], 0.0
        for g in sel:
            # Truncate at zero: a negative count is not a margin, and the ACS's own published
            # bounds are non-negative.
            t = np.maximum(rng.normal(est[g], se[g]), 0.0)
            w, e = rake_bg(X, w0, t)
            ws.append(w)
            bad = max(bad, e)
        pops_rake.append(bg_pop(ws, hh_people))
        errs.append(bad)
        print(f"  rake replicate {r + 1}/{args.n_reps}  worst margin error {bad:.2e}", flush=True)
    pops_rake = np.vstack(pops_rake)
    mean_r = pops_rake.mean(axis=0)
    cv_rake = pd.Series(pops_rake.std(axis=0, ddof=1) / np.where(mean_r > 0, mean_r, np.nan),
                        index=pup.est_g2.index[sel]).dropna()

    # --- the published ACS sampling scale, the target both are measured against -------------
    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    acs_cv = (moe["se"] / moe["acs"]).reindex(cv_rake.index.astype(str)).dropna()

    print(f"\n{'source':28s}{'CV median':>11s}{'ratio vs ACS':>14s}")
    print(f"{'ACS published sampling':28s}{acs_cv.median():11.4f}{1.0:14.3f}")
    print(f"{'P-MEDM replicates':28s}{cv_pmedm.reindex(cv_rake.index).dropna().median():11.4f}"
          f"{cv_pmedm.reindex(cv_rake.index).dropna().median() / acs_cv.median():14.3f}")
    print(f"{'perturbed margins + rake':28s}{cv_rake.median():11.4f}"
          f"{cv_rake.median() / acs_cv.median():14.3f}")
    print(f"\nworst raking margin error across replicates: {max(errs):.2e}")
    print(f"block groups scored: {len(cv_rake)} of {n_bg}")
    print("\nA ratio near 1.0 means the method reproduces raw ACS sampling scale. P-MEDM comes in")
    print("BELOW 1.0 because it borrows strength across block group, tract and PUMA; independent")
    print("margin perturbation has no hierarchy to borrow from and should sit higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
