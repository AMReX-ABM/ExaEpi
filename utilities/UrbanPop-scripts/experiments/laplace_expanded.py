#!/usr/bin/env python
"""pymedm's own Laplace replicates on the full up_expanded set, scored like resolve_spread.py.

reps_by_constraints.py measured a block-group spread ratio of 0.889 for up_expanded replicates on
PUMA 3500804 but no fit or composition measures. This draws the same replicates (20, seed 1, the
controlled-SE repair applied) and scores them with composition.py, so they can be set against a
re-solve on the same constraint set. Peak memory is ~46.5 GB: run nothing else alongside it.
Replicates are normalised to the PUMA's household total, as the re-solve allocations are.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composition  # noqa: E402
from gpu_solve import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--cache", default="./llcache_expanded")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    args = ap.parse_args()
    from livelike import config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    p = load(args.fips, key, args.cache, config.up_expanded_constraints_selection)
    pup = p["pup"]
    out = f"./laplace_expanded_{args.fips}.npy"
    if os.path.exists(out):
        reps = np.load(out)
    else:
        s1 = pd.DataFrame(p["se1"], index=pup.se_g1.index, columns=pup.se_g1.columns)
        s2 = pd.DataFrame(p["se2"], index=pup.se_g2.index, columns=pup.se_g2.columns)
        pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt, pup.est_ind, pup.est_g1, pup.est_g2,
                    s1, s2, n_reps=args.n_reps, random_state=1)
        pmd.solve()
        reps = np.asarray(pmd.almat_reps, dtype=np.float64)
        np.save(out, reps)
    als = [r / r.sum() * p["N"] for r in reps]
    conc = max(a.max() / a.sum() for a in als)

    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    acs_cv = (moe["se"] / moe["acs"]).reindex(pup.est_g2.index.astype(str)).to_numpy()
    pops = np.array([a.T @ p["C"][:, 0] for a in als])
    m = pops.mean(0)
    cv = pops.std(0, ddof=1) / np.where(m > 0, m, np.nan)
    ok = np.isfinite(cv) & np.isfinite(acs_cv)
    pub, pse = p["est2"][:, 0], p["se2"][:, 0]
    z = (pops - pub) / np.where(pse > 0, pse, np.nan)
    turn = np.mean([0.5 * np.abs(x - y).sum() / y.sum() for x, y in zip(als[1:], als[:-1])])
    print(f"PUMA {args.fips}, up_expanded, {len(als)} Laplace replicates (largest cell "
          f"{100 * conc:.2f}% of mass): CV {np.median(cv[ok]):.4f}, ratio "
          f"{np.median(cv[ok]) / np.median(acs_cv[ok]):.3f}, z rms "
          f"{np.sqrt(np.nanmean(z ** 2)):.3f}, >MOE {100 * np.nanmean(np.abs(z) > 1.645):.1f}%, "
          f"turnover {100 * turn:.2f}%")
    print(composition.header())
    print("  " + composition.line(composition.measure(
        als, p["C"], p["cols"], p["A1"], p["est1"], p["est2"], p["se1"], p["se2"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
