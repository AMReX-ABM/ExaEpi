#!/usr/bin/env python
"""Measure what replicate allocation matrices cost, and whether they reach ACS scale.

Step 1 established that a TRS re-draw from one allocation matrix moves per-block-group population
by a CV of 0.0044, against an ACS sampling CV of 0.1604 -- 36x too small to be the randomisation
axis that matters. `PMEDM(n_reps=N)` is the alternative: it samples N replicate allocation
matrices from a Laplace approximation at the converged dual, so each replicate reflects the
uncertainty in the allocation itself rather than just its integerisation.

Two things decide whether that is the path:

1. Cost. The replicate workflow needs the Hessian at the solution, which is (dual x dual) --
   32k^2 would be 8.1 GB dense, and the PMEDM build already peaks at 8.1 GB on its own. The
   upstream code notes it uses scipy sparse here specifically to avoid a JAX OOM.

2. Whether it actually helps. If replicate spread is also far below ACS scale, then neither cheap
   path works and only full replicate-weight re-solves would.
"""

import argparse
import os
import resource
import time

import numpy as np
import pandas as pd


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"--- {self.name} ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"--- {self.name}: {self.dt:.1f} s   peak RSS {rss_gb():.2f} GB", flush=True)


def bg_pop_from_almat(almat, hh_people, total_pop):
    """Expected people per block group implied by an allocation matrix.

    almat holds occurrence probabilities per (donor household, block group); weighting each
    household by its own size and scaling by the area total gives expected population.
    """
    return np.asarray(almat).T @ hh_people * total_pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", required=True)
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--cache_folder", default="./livelike_acs_cache")
    ap.add_argument("--minimal", action="store_true",
                    help="constrain only on what ExaEpi actually reads, rather than up_expanded")
    args = ap.parse_args()

    from livelike import acs, config
    from pymedm import PMEDM

    # The Hessian the replicate workflow needs is (dual x dual), and dual is
    # n_constraints x (block groups + tracts + 1) -- so its memory falls with the SQUARE of the
    # constraint count. ExaEpi reads only age, sex, race, NAICS, travel mode, vehicle occupancy
    # and school grade, so every other theme is paying quadratically for a field nothing consumes.
    #
    # Constraints govern only how the population is calibrated in SPACE. The attribute values
    # themselves come from the matched PUMS donor, so dropping a theme does not coarsen any field
    # ExaEpi reads -- exact age and detailed NAICS still arrive from the donor record either way.
    # What it can do is weaken the spatial fit of the themes that remain, since households carry
    # all attributes jointly. That is measurable, and is what the fit comparison below reports.
    EXAEPI_MINIMAL = {
        "universe": True,                    # population/housing guardrails, paper Table 1
        # hhtype_hhsize is required by homesim.synthesize, which integerises at the
    # household level and needs household sizes to expand households to persons.
    "demographic": ["sex_age", "hhtype_hhsize"],          # pr_age, pr_sex
        "social": ["race"],                  # pr_race
        "worker": ["sexnaics"],              # pr_naics
        "student": ["grade"],                # pr_grade
        "mobility": ["travel", "veh_occ"],   # pr_travel, pr_veh_occ
    }
    selection = EXAEPI_MINIMAL if args.minimal else config.up_expanded_constraints_selection

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"PUMA {args.fips}   n_reps {args.n_reps}   "
          f"selection {'EXAEPI_MINIMAL' if args.minimal else 'up_expanded'}   "
          f"key: {'set' if key else 'NOT set'}")

    with Phase("acs.puma (cached)"):
        pup = acs.puma(
            args.fips,
            constraints_selection=selection,
            constraints_theme_order=config.up_constraints_theme_order,
            year=2019, target_zone="bg", cache=True,
            cache_folder=args.cache_folder, censusapikey=key,
        )

    n_donors, n_cons = pup.est_ind.shape
    n_bg, n_trt = pup.est_g2.shape[0], pup.est_g1.shape[0]
    dual = n_cons * (n_bg + n_trt + 1)
    print(f"    donors {n_donors}  constraints {n_cons}  bgs {n_bg}  tracts {n_trt}  dual ~{dual}")
    print(f"    dense Hessian at this dual would be {dual**2 * 8 / 1e9:.1f} GB")

    with Phase("PMEDM build") as p_build:
        pmd = PMEDM(
            pup.year, pup.est_ind.index, pup.wt,
            pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
            n_reps=args.n_reps, random_state=1,
        )

    with Phase(f"solve + {args.n_reps} replicates (Hessian, invert, sample)") as p_solve:
        pmd.solve()

    reps = pmd.almat_reps
    if reps is None:
        print("!! almat_reps is None -- replicate workflow did not run")
        return 1
    reps = np.asarray(reps)
    print(f"    almat_reps {reps.shape}  {reps.nbytes / 1e6:.1f} MB f64 / {reps.nbytes / 2e6:.1f} MB f32")

    # Per-block-group expected population under each replicate.
    hh_people = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(1).to_numpy()
    total_pop = float(pup.est_g2.iloc[:, 0].sum()) if pup.est_g2.shape[1] else 1.0

    mats = [reps[i] for i in range(reps.shape[0])] if reps.ndim == 3 else list(reps)
    pops = np.vstack([bg_pop_from_almat(m, hh_people, total_pop) for m in mats])
    mean = pops.mean(axis=0)
    cv = pd.Series(pops.std(axis=0, ddof=1) / np.where(mean > 0, mean, np.nan),
                   index=pup.est_g2.index).dropna()

    print()
    print("=== replicate per-block-group population spread ===")
    print(f"  block groups {len(cv)}   replicates {len(mats)}")
    print(f"  CV  median {cv.median():.4f}  mean {cv.mean():.4f}  p90 {cv.quantile(.9):.4f}  max {cv.max():.4f}")

    if os.path.exists(args.moe_csv):
        moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
        moe = moe[moe.acs > 0].set_index("geoid")
        acs_cv = (moe["se"] / moe["acs"]).reindex(cv.index.astype(str)).dropna()
        if len(acs_cv):
            ratio = cv.median() / acs_cv.median()
            print(f"  ACS CV, same block groups: median {acs_cv.median():.4f}")
            print(f"  ratio replicate/ACS: {ratio:.3f}   (TRS managed 0.028)")
            print()
            if ratio >= 0.5:
                print("  => Replicates reach ACS scale. This is the randomisation axis to use.")
            elif ratio >= 0.15:
                print("  => Partway there: bigger than TRS but below ACS scale. Usable, but the")
                print("     ensemble would understate real uncertainty.")
            else:
                print("  => Still far below ACS scale. Neither cheap path works; only full")
                print("     replicate-weight re-solves (make_replicate_pumas) would.")

    print()
    print(f"=== summary ===")
    print(f"  build {p_build.dt:.1f} s | solve+replicates {p_solve.dt:.1f} s | peak RSS {rss_gb():.2f} GB")
    print(f"  extrapolated to 18 NM PUMAs, serial: {18 * (p_build.dt + p_solve.dt) / 60:.1f} min")
    print(f"  storage for {len(mats)} replicates x 18 PUMAs: "
          f"{18 * reps.nbytes / 2e6:.0f} MB f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
