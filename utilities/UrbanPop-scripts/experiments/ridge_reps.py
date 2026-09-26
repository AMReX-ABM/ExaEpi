#!/usr/bin/env python
"""Can a ridge on the Hessian rescue the degenerate replicate draws?

Nine of NM's eighteen PUMAs produce at least one degenerate replicate allocation matrix, and four
are severe -- 3500300 puts all 71,083 of its households into a single (donor, block group) cell in
every one of 20 draws, at every random seed tried. The point estimates are fine; only the
replicates fail. Neither the controlled-standard-error fill strategy nor reseeding changes it, so
it is structural per PUMA.

The mechanism is visible in `simulate_allocation_matrix`:

    H      = compute_hessian_matrix(pmd)
    inv_H  = linalg.inv(H)            # no regularisation
    cov    = inv_H / N
    lam_r ~ MVN(lam_hat, cov)
    almat  = compute_allocation(..., lam=lam_r)   # exponential form

If H is near-singular, inv_H has enormous eigenvalues along its near-null directions, lam_r picks
up extreme components there, and the exponential in `compute_allocation` sends one cell to the
entire mass. That is exactly the observed signature.

The standard remedy is Tikhonov regularisation: invert (H + eps*I) instead, which caps the
covariance eigenvalues at 1/eps and corresponds to a weakly-informative prior on lam. eps is set
relative to H's own scale so it transfers across PUMAs.

Sweeping eps answers whether the replicate path is salvageable at all:

    concentration back to ~1e-3 at small eps   the draws are recoverable, the bundle can ship
    degenerate until spread is crushed         regularisation only trades one failure for
                                               another, and this path is dead
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402


def draw(pmd, n, seed, eps):
    """Replicate allocation matrices with a ridge on the Hessian before inversion."""
    from pymedm.pmedm import compute_allocation, compute_hessian_matrix

    H = np.asarray(compute_hessian_matrix(pmd), dtype=np.float64)
    if eps > 0:
        # Scale the ridge to the Hessian's own diagonal so one eps works across PUMAs.
        scale = float(np.mean(np.abs(np.diag(H))))
        H = H + eps * scale * np.eye(H.shape[0])
    cov = linalg.inv(H) / pmd.N
    rng = np.random.default_rng(seed)
    lam = rng.multivariate_normal(mean=np.asarray(pmd.lam), cov=cov, size=n,
                                  tol=1e-3, method="cholesky")
    out = []
    for lr in lam:
        out.append(np.asarray(compute_allocation(
            q=pmd.q, X=pmd.X, lam=lr, prob=True, counts=True, reshape=True,
            N=pmd.N, n_obs=pmd.n, n_geo=pmd.n_topo)))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500300", "3501100", "3500804"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 1e-6, 1e-4, 1e-2, 1e-1])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cache", default="./llcache_minimal")
    args = ap.parse_args()

    from livelike import acs, config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"{'puma':10s}{'eps':>9s}{'bad/n':>8s}{'med conc':>10s}{'max conc':>10s}"
          f"{'bg pop CV':>11s}")
    for fips in args.pumas:
        pup = acs.puma(fips, constraints_selection=EXAEPI_MINIMAL,
                       constraints_theme_order=config.up_constraints_theme_order,
                       year=2019, target_zone="bg", cache=True,
                       cache_folder=args.cache, censusapikey=key)
        s1, _ = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, float), "t")
        s2, _ = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, float), "b")
        s1 = pd.DataFrame(s1, index=pup.se_g1.index, columns=pup.se_g1.columns)
        s2 = pd.DataFrame(s2, index=pup.se_g2.index, columns=pup.se_g2.columns)
        pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                    pup.est_ind, pup.est_g1, pup.est_g2, s1, s2,
                    n_reps=0, random_state=1)
        pmd.solve()
        hh = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(0).to_numpy()

        for eps in args.eps:
            R = draw(pmd, args.n, 1, eps)
            conc = np.array([R[i].max() / R[i].sum() for i in range(R.shape[0])])
            pops = np.vstack([R[i].T @ hh for i in range(R.shape[0])])
            m = pops.mean(axis=0)
            cv = np.nanmedian(pops.std(axis=0, ddof=1) / np.where(m > 0, m, np.nan))
            print(f"{fips:10s}{eps:9.0e}{int((conc > 0.01).sum()):5d}/{args.n}"
                  f"{np.median(conc):10.5f}{conc.max():10.5f}{cv:11.4f}", flush=True)
        print()
    print("concentration = largest cell / total mass; healthy is ~1e-3.")
    print("bg pop CV is the spread the bundle would ship -- it must survive the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
