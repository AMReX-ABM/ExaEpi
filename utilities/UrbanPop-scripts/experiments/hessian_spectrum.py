#!/usr/bin/env python
"""Is the replicate degeneracy explained by flat directions in the Hessian?

The proposed mechanism: P-MEDM is solved in the dual, over one lambda per (constraint, zone). The
objective's Hessian H is the matrix of second derivatives at the optimum -- the curvature of the
peak. The Laplace approximation treats the solution as Gaussian with covariance H^-1 / N, so a
direction's variance is the INVERSE of its curvature. A direction the data pins down sharply has
large curvature and small variance; a flat direction has near-zero curvature and enormous variance.

If some directions are exactly flat, H is singular, H^-1 is unbounded along them, and lambda draws
run away there. Since the allocation is recovered as q * exp(X @ lambda), an enormous lambda
component exponentiates into one cell taking the entire mass -- the observed concentration of 1.0.

Exact flatness is not hypothetical here: the constraint set contains linear dependencies by
construction. `universe` includes Total Population, and the sex_age categories sum to exactly that
same total. Shifting lambda along one and compensating along the other leaves the objective
unchanged -- a perfectly flat valley. The point-estimate solve does not care, because LBFGS simply
stops somewhere in the valley; the Laplace approximation does, because it reads flatness as
infinite uncertainty.

This compares the extreme eigenvalues of H for a PUMA whose replicates are healthy (3500804) and
one whose replicates all collapsed (3500300), and shows what the ridge does to them.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402


def hessian_for(fips, cache, key):
    from livelike import acs, config
    from pymedm import PMEDM
    from pymedm.pmedm import compute_hessian_matrix

    pup = acs.puma(fips, constraints_selection=EXAEPI_MINIMAL,
                   constraints_theme_order=config.up_constraints_theme_order,
                   year=2019, target_zone="bg", cache=True,
                   cache_folder=cache, censusapikey=key)
    s1, _ = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, float), "t")
    s2, _ = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, float), "b")
    s1 = pd.DataFrame(s1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    s2 = pd.DataFrame(s2, index=pup.se_g2.index, columns=pup.se_g2.columns)
    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, s1, s2,
                n_reps=0, random_state=1)
    pmd.solve()
    return np.asarray(compute_hessian_matrix(pmd), dtype=np.float64), pmd


def extremes(H, k=6):
    """Largest and smallest eigenvalues, via a shift-invert-free dense-free route.

    H is symmetric, so eigsh gives both ends cheaply compared with a full decomposition of a
    13,000-square matrix.
    """
    from scipy.sparse.linalg import eigsh

    hi = eigsh(H, k=k, which="LA", return_eigenvectors=False)
    lo = eigsh(H, k=k, which="SA", return_eigenvectors=False)
    return np.sort(lo), np.sort(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500804", "3500300"])
    ap.add_argument("--cache", default="./llcache_minimal")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    for fips in args.pumas:
        H, pmd = hessian_for(fips, args.cache, key)
        n = H.shape[0]
        diag = float(np.mean(np.abs(np.diag(H))))
        lo, hi = extremes(H, args.k)
        print(f"\n=== PUMA {fips}: Hessian is {n} x {n}, mean |diag| = {diag:.4g}")
        print(f"  smallest eigenvalues: {np.array2string(lo, precision=3)}")
        print(f"  largest  eigenvalues: {np.array2string(hi, precision=3)}")
        cond = hi[-1] / max(abs(lo[0]), 1e-300)
        print(f"  condition number ~ {cond:.3e}")
        # The variance the Laplace draw assigns to the flattest direction, before and after a
        # ridge. This is the number that decides whether a draw runs away.
        for eps in (0.0, 0.01, 0.1):
            shift = eps * diag
            worst_var = 1.0 / (abs(lo[0]) + shift) / pmd.N
            print(f"  ridge {eps:<5g} -> flattest-direction variance {worst_var:.4g}")
    print("\nVariance along a direction is 1/curvature, so a flat direction (eigenvalue ~ 0)")
    print("gets an enormous variance. The ridge adds eps to every eigenvalue, which caps that")
    print("at 1/(eps*scale) while leaving sharp directions -- large eigenvalues -- untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
