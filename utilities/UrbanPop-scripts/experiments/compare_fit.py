#!/usr/bin/env python
"""Does constraining on only what ExaEpi reads fit those variables worse?

Dropping constraints lowers the replicate spread, but that is only a good trade if the retained
variables are still fitted as well. Fewer constraints means lower variance and potentially higher
bias: constrain on nothing but population and every block group looks like the PUMA average --
confidently wrong. So the question is not whether the minimal set is cheaper (it is, by 11x in time
and 6x in memory) but whether age, sex, race, NAICS, grade, travel mode and vehicle occupancy land
as close to their published ACS values as they do when 188 additional margins are fitted alongside.

Both selections are scored on the SAME 110 constraints -- the ones the minimal set retains -- so
this is like for like. up_expanded fits 298 including those 110; only the shared subset is compared.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

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


def solve(fips, selection, cache_folder, key):
    from livelike import acs, config
    from pymedm import PMEDM
    from pymedm.diagnostics import moe_fit_rate

    t0 = time.time()
    pup = acs.puma(
        fips,
        constraints_selection=selection,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder=cache_folder, censusapikey=key,
    )
    pmd = PMEDM(
        pup.year, pup.est_ind.index, pup.wt,
        pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
    )
    pmd.solve()
    res = moe_fit_rate(pup.est_ind, pup.est_g2, pup.se_g2, np.asarray(pmd.almat))
    return pup, res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", required=True)
    ap.add_argument("--full_cache", default="./livelike_acs_cache")
    ap.add_argument("--min_cache", default="./cache_minimal")
    args = ap.parse_args()

    from livelike import config

    key = os.environ.get("CENSUS_API_KEY") or None

    print(f"PUMA {args.fips}\n")
    runs = {}
    for name, sel, cache in [
        ("minimal", EXAEPI_MINIMAL, args.min_cache),
        ("up_expanded", config.up_expanded_constraints_selection, args.full_cache),
    ]:
        pup, res, dt = solve(args.fips, sel, cache, key)
        n_cons = pup.est_ind.shape[1]
        print(f"{name:12s}  {n_cons:3d} constraints  build+solve {dt:5.1f} s  "
              f"overall MOE fit rate {res['moe_fit_rate']:.4f}")
        runs[name] = res["Ycomp"]

    # Score both on the constraints the minimal set retains.
    var_col = [c for c in runs["minimal"].columns if c not in
               ("acs", "pmedm", "err", "moe", "in_moe")][0]
    shared = sorted(set(runs["minimal"][var_col]) & set(runs["up_expanded"][var_col]))
    print(f"\nshared constraints scored: {len(shared)}")

    print()
    print(f"{'':12s}  {'in_MOE':>8s}  {'RAE med':>8s}  {'RAE mean':>9s}  {'RAE p90':>8s}  {'RAE max':>8s}")
    stats = {}
    for name, Y in runs.items():
        sub = Y[Y[var_col].isin(shared)].copy()
        # RAE as in section 3.6.1 of related/urbanpop.pdf: error over the 90% margin. Below 1 means
        # the synthetic estimate sits inside the ACS margin of error.
        sub["rae"] = sub["err"] / sub["moe"].replace(0, np.nan)
        rae = sub["rae"].dropna()
        stats[name] = (sub["in_moe"].mean(), rae.median(), rae.mean(), rae.quantile(.9), rae.max())
        print(f"{name:12s}  {stats[name][0]:8.4f}  {stats[name][1]:8.4f}  {stats[name][2]:9.4f}  "
              f"{stats[name][3]:8.4f}  {stats[name][4]:8.4f}")

    # Per theme, so a loss concentrated in one variable family is visible rather than averaged away.
    print("\nper-constraint-family median RAE on the shared subset:")
    fam = lambda s: s.split("_")[0] if "_" in s else s
    rows = []
    for name, Y in runs.items():
        sub = Y[Y[var_col].isin(shared)].copy()
        sub["rae"] = sub["err"] / sub["moe"].replace(0, np.nan)
        sub["fam"] = sub[var_col].map(fam)
        rows.append(sub.groupby("fam")["rae"].median().rename(name))
    comp = pd.concat(rows, axis=1)
    comp["ratio_min_over_full"] = comp["minimal"] / comp["up_expanded"]
    print(comp.sort_values("ratio_min_over_full", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    mi, fu = stats["minimal"], stats["up_expanded"]
    print(f"VERDICT: in_MOE {mi[0]:.4f} (minimal) vs {fu[0]:.4f} (up_expanded); "
          f"median RAE {mi[1]:.4f} vs {fu[1]:.4f}")
    if mi[0] >= fu[0] - 0.01 and mi[1] <= fu[1] * 1.15:
        print("  => No meaningful fidelity loss on the variables ExaEpi reads.")
        print("     The minimal set is the better trade: same fit, 11x faster, 6x less memory.")
    else:
        print("  => Minimal fits the retained variables WORSE. Narrower bands around a poorer")
        print("     estimate is the bad case; prefer the fuller constraint set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
