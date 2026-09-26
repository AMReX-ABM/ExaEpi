#!/usr/bin/env python
"""Step 1 of the plan: benchmark one P-MEDM solve, and measure TRS spread.

Two questions, both cheap once a PUMA is solved:

1. What does a P-MEDM solve cost? Nothing upstream publishes per-PUMA timings, and the whole
   cost estimate for regenerating UrbanPop rests on it.

2. Is a TRS re-draw from one allocation matrix a large enough perturbation to matter? The 70-run
   ensemble found residential variation at ACS scale (per-block-group CV median 0.160 for NM)
   moves epidemic outcomes decisively, while assignment-level variation does not. If TRS spread
   sits far below the ACS scale, TRS-only randomisation is the weak axis and calibrated replicate
   allocation matrices are needed instead.
"""

import argparse
import os
import resource
import sys
import time

import numpy as np
import pandas as pd


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Phase:
    """Time a phase and report peak RSS after it."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"--- {self.name} ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"--- {self.name}: {self.dt:.1f} s   peak RSS {rss_gb():.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", required=True, help="7-digit PUMA FIPS, e.g. 3500804")
    ap.add_argument("--nsim", type=int, default=20, help="TRS realizations for the spread test")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--cache_folder", default="./livelike_acs_cache")
    args = ap.parse_args()

    from livelike import acs, config, homesim
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"PUMA {args.fips}   census api key: {'set' if key else 'NOT set (500 queries/day cap)'}")

    with Phase("acs.puma (Census download + constraint build)") as p_dl:
        pup = acs.puma(
            args.fips,
            constraints_selection=config.up_expanded_constraints_selection,
            constraints_theme_order=config.up_constraints_theme_order,
            year=2019,
            target_zone="bg",
            cache=True,
            cache_folder=args.cache_folder,
            censusapikey=key,
        )

    n_donors, n_cons = pup.est_ind.shape
    n_bg = pup.est_g2.shape[0]
    n_trt = pup.est_g1.shape[0]
    print(f"    donors {n_donors}   constraints {n_cons}   block groups {n_bg}   tracts {n_trt}")
    print(f"    dual dimension ~= {n_cons * (n_bg + n_trt + 1)}")

    with Phase("PMEDM build") as p_build:
        pmd = PMEDM(
            pup.year, pup.est_ind.index, pup.wt,
            pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
        )

    with Phase("PMEDM solve (jaxopt LBFGS)") as p_solve:
        pmd.solve()

    almat = np.asarray(pmd.almat)
    print(f"    almat {almat.shape}  {almat.nbytes / 1e6:.1f} MB f64 / {almat.nbytes / 2e6:.1f} MB f32")
    res = getattr(pmd, "res", None) or getattr(pmd, "solver_state", None)
    if res is not None:
        for attr in ("iter_num", "num_fun_eval", "error"):
            if hasattr(res, attr):
                print(f"    solver {attr}: {getattr(res, attr)}")

    # --- TRS spread -------------------------------------------------------------------------
    with Phase(f"homesim.synthesize nsim={args.nsim}") as p_trs:
        sims = homesim.synthesize(
            almat, pup.est_ind, pup.est_g2, pup.sporder,
            nsim=args.nsim, random_state=0, longform=True,
        )

    # longform: index h_id, columns sim / geoid / count. Household counts -> people needs the
    # household size, which is the number of PUMS person records sharing that household id.
    hh_size = pup.sporder.groupby(level=0).size() if pup.sporder.index.nlevels else None
    sims = sims.reset_index()
    if hh_size is not None:
        sims["people"] = sims["count"] * sims[sims.columns[0]].map(hh_size).fillna(1).values
    else:
        sims["people"] = sims["count"]

    pop = sims.groupby(["sim", "geoid"])["people"].sum().unstack("sim")
    cv = (pop.std(axis=1, ddof=1) / pop.mean(axis=1)).dropna()

    print()
    print("=== TRS per-block-group population spread ===")
    print(f"  block groups: {len(cv)}   realizations: {args.nsim}")
    print(f"  CV  median {cv.median():.4f}   mean {cv.mean():.4f}   p90 {cv.quantile(.9):.4f}   max {cv.max():.4f}")

    if os.path.exists(args.moe_csv):
        moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
        moe = moe[moe.acs > 0].set_index("geoid")
        acs_cv = (moe["se"] / moe["acs"]).reindex(cv.index).dropna()
        if len(acs_cv):
            print(f"  ACS CV on the same block groups: median {acs_cv.median():.4f}")
            print(f"  ratio TRS/ACS (median): {cv.median() / acs_cv.median():.3f}")
            print()
            if cv.median() < 0.5 * acs_cv.median():
                print("  => TRS spread is well below ACS sampling scale. TRS-only randomisation is")
                print("     the weak axis; use replicate allocation matrices (PMEDM n_reps) instead.")
            else:
                print("  => TRS spread is comparable to ACS sampling scale. TRS-only randomisation")
                print("     from one allocation matrix is sufficient.")

    print()
    print("=== summary ===")
    print(f"  download {p_dl.dt:.1f} s | build {p_build.dt:.1f} s | solve {p_solve.dt:.1f} s | "
          f"trs {p_trs.dt:.1f} s | peak RSS {rss_gb():.2f} GB")
    solve_total = p_build.dt + p_solve.dt
    print(f"  extrapolated to 18 NM PUMAs, serial: {18 * solve_total / 60:.1f} min of solve")
    if os.path.isdir(args.cache_folder):
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(args.cache_folder) for f in fs)
        print(f"  ACS cache: {sz / 1e6:.1f} MB for 1 PUMA -> ~{18 * sz / 1e6:.0f} MB for NM")


if __name__ == "__main__":
    sys.exit(main())
