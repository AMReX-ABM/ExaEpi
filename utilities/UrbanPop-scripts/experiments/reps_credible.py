#!/usr/bin/env python
"""Is up_expanded's replicate spread real uncertainty, or a numerical artifact?

The theme curve produced a result that does not look statistically possible. Adding each dropped
theme to the minimal set ALONE moves the replicate spread almost not at all -- the largest single
contribution is +0.013 on a ratio of 0.323, and several are negative -- yet all of them together
jump it to 0.899. In variance terms the individual contributions sum to roughly a thirteenth of
the gap. Sampling error from independent noisy margins should accumulate roughly additively; seven
themes each contributing nothing cannot combine into a 2.8x increase by any ordinary mechanism.

So the jump needs explaining before anyone builds a bundle on it, and pymedm's own code says why
it might not be real:

    "The Hessian Matrix functionality here is highly experimental and not mature. Use with
     caution."

Replicates are drawn as lam ~ N(lam_hat, H^-1) at the converged dual. If H is near-singular at
dual 31,886 -- it is 298 x (block groups + tracts + 1) -- then H^-1 has enormous eigenvalues along
the ill-conditioned directions and the draws blow up. That would produce exactly this signature:
nothing until the dual gets large, then a sudden jump.

The test is whether the replicates remain CREDIBLE. A replicate is a plausible alternative
population only if it still respects the ACS margins it was fitted to. So compute the margin-of-
error fit rate for each replicate and compare it with the point estimate's:

    point estimate in_MOE ~ 1.0, replicates in_MOE ~ 1.0   spread is genuine posterior uncertainty
    point estimate in_MOE ~ 1.0, replicates in_MOE << 1.0  replicates fall outside the published
                                                           margins -- over-dispersed, not credible
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402


def run(name, sel, fips, n_reps, cache_root, key):
    from livelike import acs, config
    from pymedm import PMEDM
    from pymedm.diagnostics import moe_fit_rate

    cache = os.path.join(cache_root, name)
    os.makedirs(cache, exist_ok=True)
    pup = acs.puma(
        fips, constraints_selection=sel,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True, cache_folder=cache, censusapikey=key,
    )
    se1, _ = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, dtype=float), "t")
    se2, _ = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, dtype=float), "b")
    se1 = pd.DataFrame(se1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    se2 = pd.DataFrame(se2, index=pup.se_g2.index, columns=pup.se_g2.columns)

    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, se1, se2,
                n_reps=n_reps, random_state=1)
    pmd.solve()

    point = moe_fit_rate(pup.est_ind, pup.est_g2, se2, np.asarray(pmd.almat))["moe_fit_rate"]
    reps = np.asarray(pmd.almat_reps)
    rates = [moe_fit_rate(pup.est_ind, pup.est_g2, se2, reps[i])["moe_fit_rate"]
             for i in range(reps.shape[0])]

    # Total allocated mass: a credible replicate reallocates people, it does not invent or lose
    # them. A drifting total is a second, independent sign of a bad draw.
    tot = np.array([reps[i].sum() for i in range(reps.shape[0])])
    point_tot = float(np.asarray(pmd.almat).sum())

    n_con = pup.est_ind.shape[1]
    dual = n_con * (pup.est_g2.shape[0] + pup.est_g1.shape[0] + 1)
    return {
        "name": name, "constraints": n_con, "dual": dual,
        "point_in_moe": float(point),
        "rep_in_moe_mean": float(np.mean(rates)), "rep_in_moe_min": float(np.min(rates)),
        "point_total": point_tot,
        "rep_total_mean": float(tot.mean()), "rep_total_cv": float(tot.std(ddof=1) / tot.mean()),
        "rep_total_min": float(tot.min()), "rep_total_max": float(tot.max()),
        "neg_frac": float((reps < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--cache_root", default="./theme_cache")
    args = ap.parse_args()

    from livelike import config

    key = os.environ.get("CENSUS_API_KEY") or None
    out = []
    for name, sel in [("minimal", EXAEPI_MINIMAL),
                      ("up_expanded", config.up_expanded_constraints_selection)]:
        r = run(name, sel, args.fips, args.n_reps, args.cache_root, key)
        out.append(r)
        print(f"--- {name}: {r['constraints']} constraints, dual {r['dual']}")
        print(f"    point estimate in_MOE      {r['point_in_moe']:.4f}")
        print(f"    replicate in_MOE  mean     {r['rep_in_moe_mean']:.4f}   min {r['rep_in_moe_min']:.4f}")
        print(f"    allocated total   point    {r['point_total']:.1f}")
        print(f"                      reps     mean {r['rep_total_mean']:.1f}  "
              f"CV {r['rep_total_cv']:.4f}  [{r['rep_total_min']:.1f}, {r['rep_total_max']:.1f}]")
        print(f"    negative cells in replicates {r['neg_frac']:.6f}")
        print(flush=True)

    a, b = out
    print("=" * 74)
    drop_a = a["point_in_moe"] - a["rep_in_moe_mean"]
    drop_b = b["point_in_moe"] - b["rep_in_moe_mean"]
    print(f"in_MOE lost by replication:  minimal {drop_a:+.4f}   up_expanded {drop_b:+.4f}")
    if drop_b > 0.05 and drop_b > 3 * max(drop_a, 1e-6):
        print("\n  => up_expanded's replicates fall OUTSIDE the ACS margins the point estimate")
        print("     satisfies. Its larger spread is not credible posterior uncertainty, and the")
        print("     0.899 ratio should not be treated as the calibration target.")
    elif drop_b < 0.02:
        print("\n  => Replicates stay inside the ACS margins. The spread is genuine, and the")
        print("     nonlinearity across themes is a real property of the posterior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
