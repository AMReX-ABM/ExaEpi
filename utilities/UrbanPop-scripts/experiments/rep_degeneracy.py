#!/usr/bin/env python
"""Why do some PUMAs' replicate allocation matrices collapse into a single cell?

The 20-replicate NM bundle has four bad PUMAs. Two of them, 3500300 and 3500400, put their ENTIRE
allocation -- 71,083 and 56,824 households -- into one (donor, block group) cell. The point
estimates for the same PUMAs are fine, so this is the Laplace replicate draw, not the solve.

Two suspects:

    the SE repair   Controlled ACS estimates have no published margin of error, and
                    `repair_controlled_se` substitutes the TIGHTEST sigma the same constraint shows
                    elsewhere. P-MEDM weights constraints by 1/sigma^2, so the tightest sigma is
                    also the largest weight in the problem. The point solve tolerates that; the
                    Hessian it induces may not, and `simulate_allocation_matrix` inverts that
                    Hessian. Both conc=1.0 PUMAs are repaired ones.
    conditioning    But 3501100 and 3500200 are also degenerate and needed no repair at all, so
                    ill-conditioning is not exclusively the repair's doing.

This compares fill strategies for the controlled sigma -- tightest, median, and a fixed relative
CV -- on the affected PUMAs, scoring each by how concentrated the replicates come out. A healthy
replicate looks like the point estimate: no cell holding a meaningful share of the total.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_precompute import EXAEPI_MINIMAL  # noqa: E402


def repair(se, est, how):
    """Fill non-finite standard errors by one of several strategies."""
    se = np.asarray(se, dtype=np.float64).copy()
    est = np.asarray(est, dtype=np.float64)
    bad = ~np.isfinite(se)
    if not bad.any():
        return se, 0
    for c in range(se.shape[1]):
        m = ~np.isfinite(se[:, c])
        if not m.any():
            continue
        good = se[:, c][np.isfinite(se[:, c]) & (se[:, c] > 0)]
        if how == "min":
            fill = good.min() if len(good) else 1.0
        elif how == "median":
            fill = np.median(good) if len(good) else 1.0
        elif how == "cv10":
            # Treat a controlled estimate as if it carried a typical 10% relative error: it is
            # not tight enough to dominate the objective, and not so loose as to be ignored.
            fill = np.maximum(0.10 * np.abs(est[m, c]), 1.0)
        else:
            raise ValueError(how)
        se[m, c] = fill
    return se, int(bad.sum())


def run(fips, how, n_reps, cache, key):
    from livelike import acs, config
    from pymedm import PMEDM

    pup = acs.puma(fips, constraints_selection=EXAEPI_MINIMAL,
                   constraints_theme_order=config.up_constraints_theme_order,
                   year=2019, target_zone="bg", cache=True,
                   cache_folder=cache, censusapikey=key)
    s1, n1 = repair(pup.se_g1, np.asarray(pup.est_g1, float), how)
    s2, n2 = repair(pup.se_g2, np.asarray(pup.est_g2, float), how)
    s1 = pd.DataFrame(s1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    s2 = pd.DataFrame(s2, index=pup.se_g2.index, columns=pup.se_g2.columns)
    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, s1, s2,
                n_reps=n_reps, random_state=1)
    pmd.solve()
    a = np.asarray(pmd.almat)
    R = np.asarray(pmd.almat_reps)
    conc = np.array([R[i].max() / R[i].sum() for i in range(R.shape[0])])
    return {
        "repaired": n1 + n2,
        "point_conc": float(a.max() / a.sum()),
        "rep_conc_max": float(conc.max()),
        "rep_conc_median": float(np.median(conc)),
        "bad_reps": int((conc > 0.01).sum()),
        "n_reps": int(R.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+",
                    default=["3500300", "3500200", "3500804"])
    ap.add_argument("--hows", nargs="+", default=["min", "median", "cv10"])
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--cache", default="./llcache_minimal")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"{'puma':10s}{'fill':9s}{'repaired':>9s}{'point':>9s}{'rep med':>9s}"
          f"{'rep max':>9s}{'bad/20':>8s}")
    for fips in args.pumas:
        for how in args.hows:
            r = run(fips, how, args.n_reps, args.cache, key)
            print(f"{fips:10s}{how:9s}{r['repaired']:9d}{r['point_conc']:9.4f}"
                  f"{r['rep_conc_median']:9.4f}{r['rep_conc_max']:9.4f}"
                  f"{r['bad_reps']:5d}/{r['n_reps']}", flush=True)
    print("\nconcentration = largest single cell / total allocated mass.")
    print("A healthy matrix is ~1e-3 or below; 1.0 means every household went to one cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
