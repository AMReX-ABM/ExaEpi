"""Per-run perturbation of one PUMA's P-MEDM inputs.

Each run re-solves P-MEDM on inputs redrawn within their sampling error, which is what varies who
lives where (experiments/resolve_spread.py: block-group spread 0.30-0.33 of ACS, <=0.2% of block
groups outside their published margins of error). Two independent sources:

    targets   block-group estimates redrawn from their standard errors; tract targets rebuilt as
              published tract + (sum of the tract's perturbed block groups - sum of published),
              so tracts and block groups stay consistent. Perturbing them independently makes the
              two levels contradict each other and the solve averages the contradiction away
              (measured: ratio 0.27 independent vs 0.31 nested).
    prior     PUMS survey weights times Exp(1): a Bayesian bootstrap over the donor sample,
              standing in for the sampling error of the weights themselves. It measured almost the
              same as the Census replicate weights (turnover 44.7% vs 44.1% on PUMA 3500804) and
              needs no extra data in the bundle.

Draws are keyed through KR64 on (seed, rep, stage, puma, constraint, block group) and (seed, rep,
stage, puma, donor row). They use log and cos (Box-Muller), so the C++ port need not match them
bit for bit -- the solve that consumes them is not bitwise-reproducible across implementations
anyway. They must still be pure functions of their keys.
"""

import numpy as np

from . import kr64, stages
from .problem import targets


def normal(k):
    """Standard normal deviates from uint64 keys (Box-Muller on draws 0 and 1)."""
    u1 = kr64.u01(kr64.draw(k, 0))
    u2 = kr64.u01(kr64.draw(k, 1))
    return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


def exponential(k):
    """Exp(1) deviates from uint64 keys."""
    return -np.log1p(-kr64.u01(kr64.draw(k, 0)))


def perturbed_targets(prob, seed, rep):
    """(Y, est_bg perturbed) for one PUMA: nested block-group-first perturbation."""
    K = prob.K
    k = kr64.key(seed, rep, stages.PERTURB_TARGET, int(prob.fips),
                 np.arange(K, dtype=np.int64)[None, :], prob.bg_geoid[:, None])
    y2 = np.maximum(prob.est_bg + prob.se_bg * normal(k), 0.0)
    y1 = np.maximum(prob.est_tract + prob.A1 @ (y2 - prob.est_bg), 0.0)
    return targets(y2, y1, prob.N), y2


def perturbed_prior(prob, seed, rep):
    """Prior weights times Exp(1), keyed per donor row."""
    k = kr64.key(seed, rep, stages.PERTURB_PRIOR, int(prob.fips), np.arange(prob.D, dtype=np.int64))
    return prob.wt * exponential(k)


def log_prior(w, G):
    """pymedm's prior: each donor's weight spread evenly over the PUMA's block groups (log)."""
    return np.broadcast_to(np.log(w / w.sum() / G)[:, None], (len(w), G))
