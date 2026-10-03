"""Commute geometry for stage S3: distances, the corrected LODES prior, and the tables behind it.

S3 places workers with LODES as the prior, but LODES links jobs to residences from administrative
records, not daily commutes: its long-distance tail is ~10x CTPP's (NM 11.7% of jobs > 100 km vs
0.8% of commuters), almost all of it 1-2-job block pairs. So the prior is corrected, with tables
fitted once per bundle against CTPP (calibrate_commute.py) and stored in it:

  commute.dist_edges, commute.r   a reliability weight per distance band, applied to every LODES
                                  pair (and to the background below) wherever LODES enters S3
  commute.kern_edges, commute.kern
                                  P(time band | distance) / P(time band), from a lognormal
                                  straight-line speed model, per band on a grid of distances; a
                                  worker's fill row is (home, time band), and the band's row of
                                  this table weights its destinations
  commute.band                    time band per (travel mode, reported minutes JWMNP); the last
                                  band is neutral (no time reported), with weight 1 everywhere
  commute.decay                   background decay on the same distance grid
  commute.params                  [background mass alpha, radius km, nearest-k, distance floor km]

Background: LODES block-group pairs are sparse (NM: 11% of pairs), and a home whose LODES row
reaches no destination demanding its workers' industry nearby would be sent far. Each home h
also reaches its k nearest destinations within the radius, with weight
alpha * jobs_from(h) * jobs_at(d) * decay(d) / (the same summed over those destinations).

Distances are chord lengths (km) between the block groups' points in geo.xyz (Earth-centred, km;
each block group's tract internal point; geo.geoid lists every block group the bundle names), so
the runtime needs only +, -, * and a correctly rounded sqrt -- the same double in Python and
C++. Same-tract pairs are exactly 0. Every table
lookup is a comparison against stored edges. The corrected weights are quantised to integers
(W_SCALE per job), so S3's destination totals and commute-shed supplies stay exact integers.
"""

import numpy as np

TRAVEL_WFH = 7
W_SCALE = 1 << 16          # quantised prior weight per job
MAX_MINUTES = 200          # commute.band covers reported minutes 0 .. MAX_MINUTES (clamped)
EARTH_KM = 6371.0088
BIG = 1e30                 # the open top edge of a table (sections must be finite)

# Build-time defaults (calibrate_commute.py); the runtime reads everything from the bundle.
DIST_EDGES = np.array([0, 1e-9, 2, 5, 10, 20, 35, 50, 75, 100, 150, 250, 400, BIG])
# Car-equivalent speed ratio per travel code: car/truck/van, public transport, bicycle, walked,
# motorcycle, taxicab, other; wfh has no commute.
MODE_RATIO = np.array([1.0, 0.55, 0.30, 0.12, 1.0, 1.0, 0.7, 0.0])
# Time bands on car-equivalent minutes (upper edges fall between reported-minute heaps).
TIME_EDGES = np.array([0, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 40.5, 45.5, 60.5, 90.5, np.inf])
N_BANDS = len(TIME_EDGES) - 1          # time bands; band N_BANDS is neutral
D_FLOOR = 1.5                          # km: kernel distance sqrt(d^2 + D_FLOOR^2)
KERN_EDGES = np.r_[0.0, np.geomspace(D_FLOOR, 2500.0, 96), BIG]
BG_ALPHA, BG_RADIUS, BG_K, BG_DECAY_KM = 0.2, 50.0, 300, 10.0
# Tract codes renumbered after the 2010 TIGER files (LA County, 2012): the code PUMS-era products
# use -> the 2010 tract whose internal point stands in for it.
TRACT_ALIASES = {6037137000: 6037930401}


def bin_of(edges, x):
    """Index i with edges[i] <= x < edges[i + 1] (edges ascending, edges[0] <= x)."""
    return np.searchsorted(edges, x, side="right") - 1


def chord2(a, b):
    """Squared chord distance (km^2) between rows of two (n, 3) coordinate arrays, summed x, y, z
    in that order."""
    d = a - b
    return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]


def xyz_of(b, geoids):
    """geo.xyz rows for block-group geoids (geo.geoid is sorted)."""
    g = b["geo.geoid"]
    i = np.searchsorted(g, geoids)
    if np.any(i >= len(g)) or np.any(g[np.minimum(i, len(g) - 1)] != geoids):
        raise ValueError("block group without coordinates in geo.xyz")
    return b["geo.xyz"][i]


def row_bands(b):
    """Fill-row bands, the neutral one last: the kernel table's row count."""
    return int(b["commute.kern"].shape[0])


def floor2(b):
    """The kernel distance floor squared (by multiplication: pow is not portable bit for bit)."""
    f = float(b["commute.params"][3])
    return f * f


def bands(b, travel, minutes):
    """Time band per worker from the bundle's (mode, minutes) table; no mode is neutral."""
    tab = b["commute.band"]
    t = np.clip(travel.astype(np.int64), 0, tab.shape[0] - 1)
    m = np.clip(minutes.astype(np.int64), 0, tab.shape[1] - 1)
    out = tab[t, m].astype(np.int64)
    return np.where(travel < 0, row_bands(b) - 1, out)


def background(b, homes, dests, pair_h, pair_d, pair_c):
    """(home index, dest index, weight) background pairs; pairs sorted by (home, dest).

    For each home: destinations within the radius, the nearest k by (squared chord, dest index),
    weight alpha * H_h * J_d * decay[kernel bin] / sum over them, as (alpha * H_h) * g / S with
    g = J_d * decay and S the sequential sum of g in distance order."""
    alpha, radius, k, _ = b["commute.params"]
    k = int(k)
    J = np.bincount(pair_d, weights=pair_c, minlength=len(dests))
    H = np.bincount(pair_h, weights=pair_c, minlength=len(homes))
    xh, xd = xyz_of(b, homes), xyz_of(b, dests)
    kern_edges, decay = b["commute.kern_edges"], b["commute.decay"]
    f2 = floor2(b)
    r2 = float(radius) * float(radius)
    oh, od, ow = [], [], []
    for h in range(len(homes)):
        d2 = chord2(np.broadcast_to(xh[h], xd.shape), xd)
        cand = np.flatnonzero(d2 <= r2)
        cand = cand[np.argsort(d2[cand], kind="stable")][:k]
        if len(cand) == 0 or H[h] == 0:
            continue
        g = J[cand] * decay[bin_of(kern_edges, np.sqrt(d2[cand] + f2))]
        s = np.cumsum(g)[-1]
        if s <= 0:
            continue
        w = (alpha * H[h]) * g / s
        o = np.argsort(cand, kind="stable")
        oh.append(np.full(len(cand), h, dtype=np.int64))
        od.append(cand[o])
        ow.append(w[o])
    if not oh:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(oh), np.concatenate(od), np.concatenate(ow)


def prior(b, homes, dests, pair_h, pair_d, pair_c):
    """The corrected, quantised S3 prior over (home index, dest index) pairs, sorted by (home,
    dest): LODES plus background, times r(distance), rescaled to the total before r, times
    W_SCALE, rounded half up; pairs rounding to 0 dropped. Also each pair's kernel-distance bin.

    Returns (h, d, weight int64, kernel bin)."""
    bh, bd, bw = background(b, homes, dests, pair_h, pair_d, pair_c)
    key_l = pair_h * len(dests) + pair_d
    key_b = bh * len(dests) + bd
    keys = np.union1d(key_l, key_b)
    c = np.zeros(len(keys))
    c[np.searchsorted(keys, key_l)] = pair_c.astype(np.float64)
    c[np.searchsorted(keys, key_b)] += bw
    h, d = keys // len(dests), keys % len(dests)
    d2 = chord2(xyz_of(b, homes[h]), xyz_of(b, dests[d]))
    w = c * b["commute.r"][bin_of(b["commute.dist_edges"], np.sqrt(d2))]
    f = np.cumsum(c)[-1] / np.cumsum(w)[-1]
    q = np.floor((w * f) * W_SCALE + 0.5).astype(np.int64)
    kb = bin_of(b["commute.kern_edges"], np.sqrt(d2 + floor2(b)))
    keep = q > 0
    return h[keep], d[keep], q[keep], kb[keep]


# --- build time --------------------------------------------------------------------------------
def tract_points(shapefiles):
    """Tract internal points (lat, lon) indexed by int64 GEOID10, from 2010 TIGER tract files."""
    import geopandas as gp
    import pandas as pd
    t = pd.concat([gp.read_file(f, ignore_geometry=True)[["GEOID10", "INTPTLAT10", "INTPTLON10"]]
                   for f in shapefiles])
    return pd.DataFrame({"lat": t.INTPTLAT10.astype(float).values, "lon": t.INTPTLON10.astype(float).values},
                        index=t.GEOID10.astype(np.int64).values)


def tract_xyz(points, tracts):
    """Earth-centred coordinates (km) of tracts' internal points; tracts renumbered after 2010 go
    through TRACT_ALIASES. Raises on a tract with no point."""
    t = np.array([TRACT_ALIASES.get(int(x), int(x)) for x in tracts], dtype=np.int64)
    p = points.reindex(t)
    bad = np.isnan(p.lat.values)
    if bad.any():
        raise ValueError(f"{bad.sum()} tracts without internal points, e.g. {t[bad][:5].tolist()}")
    la, lo = np.radians(p.lat.values), np.radians(p.lon.values)
    return EARTH_KM * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def default_tables():
    """Uncalibrated commute sections: r = 1, neutral kernel -- LODES plus the background.
    calibrate_commute.py replaces commute.r and commute.kern."""
    nk = len(KERN_EDGES) - 1
    return {
        "commute.band": band_table(),
        "commute.dist_edges": DIST_EDGES.astype(np.float64),
        "commute.r": np.ones(len(DIST_EDGES) - 1),
        "commute.kern_edges": KERN_EDGES.astype(np.float64),
        "commute.kern": np.ones((N_BANDS + 1, nk)),
        "commute.decay": decay_table(KERN_EDGES, BG_DECAY_KM),
        "commute.params": np.array([BG_ALPHA, BG_RADIUS, BG_K, D_FLOOR], dtype=np.float64),
    }


def band_table():
    """int8 (8 modes, MAX_MINUTES + 1): time band from car-equivalent minutes; minutes 0 (no time
    reported) and wfh are neutral."""
    m = np.arange(MAX_MINUTES + 1, dtype=np.float64)
    tab = np.full((len(MODE_RATIO), MAX_MINUTES + 1), N_BANDS, dtype=np.int8)
    for t, ratio in enumerate(MODE_RATIO):
        if ratio == 0:
            continue
        teff = m * ratio
        band = np.searchsorted(TIME_EDGES, teff, side="right") - 1
        tab[t] = np.where(m > 0, band, N_BANDS)
    return tab


def car_minutes(travel, minutes):
    """Car-equivalent minutes (0 = no time) per worker, for fitting the kernel."""
    t = np.clip(travel.astype(np.int64), 0, len(MODE_RATIO) - 1)
    return np.where((minutes > 0) & (travel >= 0), minutes * MODE_RATIO[t], 0.0)


class SpeedKernel:
    """P(band | d) / P(band) under d = v t, v lognormal(ln v50, sigma), t the band's own mix of
    car-equivalent minutes (from the workers)."""

    def __init__(self, teff, v50, sigma):
        self.v50, self.sigma = v50, sigma
        band = np.where(teff > 0, np.searchsorted(TIME_EDGES, teff, side="right") - 1, N_BANDS)
        self.t, self.tw, self.pi = [], [], np.zeros(N_BANDS)
        for k in range(N_BANDS):
            tk = teff[band == k]
            u, c = np.unique(np.round(tk, 2), return_counts=True)
            self.t.append(u)
            self.tw.append(c / c.sum() if len(c) else c)
            self.pi[k] = len(tk)
        self.pi /= self.pi.sum()

    def _log_k(self, d):
        from scipy.special import logsumexp
        ld = np.log(d)[None, :]
        out = np.full((N_BANDS, len(d)), -np.inf)
        for k in range(N_BANDS):
            if len(self.t[k]):
                mu = np.log(self.v50 * self.t[k] / 60.0)[:, None]
                out[k] = logsumexp(-0.5 * ((ld - mu) / self.sigma) ** 2 + np.log(self.tw[k])[:, None], axis=0)
        return out

    def table(self, edges):
        """(N_BANDS + 1, bins) ratio at each bin's geometric-mean distance; last row neutral."""
        from scipy.special import logsumexp
        rep = bin_distance(edges)
        lk = self._log_k(rep)
        lm = logsumexp(lk + np.log(np.maximum(self.pi, 1e-300))[:, None], axis=0)
        r = np.exp(np.maximum(lk - lm[None, :], np.log(1e-6)))
        return np.vstack([r, np.ones((1, len(rep)))])

    def bin_probs(self, edges):
        """Model straight-line distance profile over edges."""
        from scipy.special import ndtr
        out = np.zeros(len(edges) - 1)
        le = np.log(np.maximum(edges, 1e-12))
        for k in range(N_BANDS):
            if len(self.t[k]):
                mu = np.log(self.v50 * self.t[k] / 60.0)[:, None]
                cdf = ndtr((le[None, :] - mu) / self.sigma)
                cdf[:, 0], cdf[:, -1] = 0.0, 1.0
                out += self.pi[k] * (self.tw[k][:, None] * np.diff(cdf, axis=1)).sum(axis=0)
        return out


def bin_distance(edges):
    """A kernel-grid bin's distance: the geometric mean of its edges (floored at D_FLOOR), or its
    lower edge for the open top bin."""
    lo, hi = np.maximum(edges[:-1], D_FLOOR), edges[1:]
    return np.where(hi >= BIG, lo, np.sqrt(lo * np.minimum(hi, BIG)))


def decay_table(edges, decay_km):
    return np.exp(-bin_distance(edges) / decay_km)
