"""CBP reference tables: workgroup-size targets and establishment-size distributions.

Workgroup targets come straight from the bundle (cbp.wg_*), defaulting to
cbp.default_workgroup_target for a (state, NAICS) with no row.

Establishment sizes replace upop_to_exaepi.load_establishment_size_dists, which expanded each CBP
band into 256 log-uniform draws from ONE numpy default_rng(0) stream run across the whole national
file in order -- reproducible only by re-reading that exact file, never from per-state bundle data,
and not at all in C++. This is the deterministic equivalent: per (state, NAICS, band) with n
establishments and e employees,

    hi_eff  = band upper bound; for the open 1000+ band max(lo + 1, (4 e) // n)
    r       = ((hi_eff + 1) / lo) ** (1/256)   computed as 8 successive square roots
    q_i     = lo * sqrt(r) * r**i,  i = 0..255 (midpoints of 256 equal log-strata, by repeated
              multiplication)
    size_i  = max(1, rint(q_i * (e / n) / mean(q)))   with mean(q) a sequential sum / 256

Same data and same per-band means as before, without the random extremes. Only +, *, / and sqrt
are used, all correctly rounded in IEEE, so C++ reproduces the table exactly. Sampling one size:
band = int_cdf(n_b, draw 0), then i = index(draw 1, 256).
"""

import numpy as np

from . import kr64

POINTS = 256


def band_sizes(lo, hi, n, e):
    """The 256 sizes for one band (int64), or None when the band is empty."""
    if n <= 0 or e <= 0:
        return None
    hi_eff = hi if hi >= 0 else max(lo + 1, (4 * e) // n)
    r = (hi_eff + 1) / lo
    for _ in range(8):
        r = np.sqrt(r)
    q = np.empty(POINTS)
    q[0] = lo * np.sqrt(r)
    for i in range(1, POINTS):
        q[i] = q[i - 1] * r
    mean_q = np.cumsum(q)[-1] / POINTS
    f = (e / n) / mean_q
    return np.maximum(1, np.rint(q * f)).astype(np.int64)


class SizeTables:
    """Lookup of workgroup targets and establishment-size tables by (state, naics index)."""

    def __init__(self, b):
        self.default_target = int(b["cbp.default_workgroup_target"][0])
        self.targets = {(int(s), int(n)): int(z) for s, n, z in
                        zip(b["cbp.wg_state"], b["cbp.wg_naics"], b["cbp.wg_size"])}
        lo, hi = b["cbp.est_band_lo"], b["cbp.est_band_hi"]
        self.est = {}
        for s, n, row in zip(b["cbp.est_state"], b["cbp.est_naics"], b["cbp.est_bands"]):
            counts, tables = [], []
            for k in range(len(lo)):
                t = band_sizes(int(lo[k]), int(hi[k]), int(row[2 * k]), int(row[2 * k + 1]))
                if t is not None:
                    counts.append(int(row[2 * k]))
                    tables.append(t)
            if counts:
                self.est[(int(s), int(n))] = (np.array(counts, dtype=np.int64), np.stack(tables))

    def target(self, state, naics):
        return self.targets.get((int(state), int(naics)), self.default_target)

    def has_est(self, state, naics):
        return (int(state), int(naics)) in self.est

    def sample(self, state, naics, k):
        """One establishment size for key k (uint64 key array allowed): band then stratum."""
        counts, tables = self.est[(int(state), int(naics))]
        band = kr64.int_cdf(counts, kr64.draw(k, 0))
        i = kr64.index(kr64.draw(k, 1), POINTS)
        return tables[band, i]

    def mean(self, state, naics):
        """Exact mean establishment size: sum_b n_b * sum_i size_bi / (256 * sum_b n_b)."""
        counts, tables = self.est[(int(state), int(naics))]
        return float((counts * tables.sum(axis=1)).sum()) / (POINTS * float(counts.sum()))
