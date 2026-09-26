"""One PUMA's P-MEDM problem, rebuilt from a bundle's `solve.*` sections.

The bundle stores only published and survey quantities -- the donor constraint matrix, prior
weights, and block-group and tract estimates with (repaired) standard errors. Everything the
solver needs is derived from them here exactly as pymedm derives it (pymedm/pmedm.py,
PMEDM.__init__), so the runtime needs neither pymedm nor livelike:

    N  = sum of prior weights          n = number of donors
    tract of a block group = its geoid with the last digit dropped; tracts sorted ascending
    Y  = [ Y0 ; Y1 ; Y2 ] / N          V = [ V0 ; V1 ; V2 ] * n / N^2
         Y0 = column sums of the block-group estimates (the PUMA level)
         Y1, Y2 = tract and block-group estimates flattened constraint-major ("F" order)
         V0 = (0.1 * sqrt(sum over block groups of se^2))^2   -- pymedm tightens the PUMA level
         V1, V2 = tract and block-group se^2, same order

build_precompute.py checks this reconstruction against pymedm itself for every PUMA it writes.
The C++ port (src/PmedmSolver) implements the same derivation.
"""

import numpy as np

CG0_SE_FACTOR = 0.1


class PumaProblem:
    """Dense arrays for one PUMA: C (donors x K), A1 (tracts x bgs), Y, V, wt, N."""

    def __init__(self, bundle, p):
        b = bundle
        self.fips = _strings(b, "solve.puma")[p]
        g0, g1 = (int(x) for x in b["solve.bg_offset"][p:p + 2])
        t0, t1 = (int(x) for x in b["solve.tract_offset"][p:p + 2])
        d0, d1 = (int(x) for x in b["solve.donor_offset"][p:p + 2])
        self.bg_geoid = b["solve.bg_geoid"][g0:g1]
        self.tract_geoid = b["solve.tract_geoid"][t0:t1]
        self.donor_index = b["solve.donor_index"][d0:d1]
        self.donor_hh_size = b["solve.donor_hh_size"][d0:d1]
        self.wt = b["solve.prior_weight"][d0:d1].astype(np.float64)
        self.est_bg = b["solve.est_bg"][g0:g1].astype(np.float64)
        self.se_bg = b["solve.se_bg"][g0:g1].astype(np.float64)
        self.est_tract = b["solve.est_tract"][t0:t1].astype(np.float64)
        self.se_tract = b["solve.se_tract"][t0:t1].astype(np.float64)
        K = self.est_bg.shape[1]
        G, T, D = g1 - g0, t1 - t0, d1 - d0

        # Donor constraint matrix from the global CSR rows d0..d1.
        ip = b["solve.c_indptr"]
        lo, hi = int(ip[d0]), int(ip[d1])
        self.C = np.zeros((D, K))
        rows = np.repeat(np.arange(D), np.diff(ip[d0:d1 + 1]))
        self.C[rows, b["solve.c_indices"][lo:hi].astype(np.int64)] = b["solve.c_values"][lo:hi]

        tract_of_bg = b["solve.bg_tract"][g0:g1].astype(np.int64)
        self.A1 = np.zeros((T, G))
        self.A1[tract_of_bg, np.arange(G)] = 1.0

        self.N = float(self.wt.sum())
        self.n = D
        self.Y = targets(self.est_bg, self.est_tract, self.N)
        sg0 = CG0_SE_FACTOR * np.sqrt(np.square(self.se_bg).sum(axis=0))
        self.V = np.concatenate([np.square(sg0), np.square(self.se_tract).flatten("F"),
                                 np.square(self.se_bg).flatten("F")]) * (self.n / self.N ** 2)
        self.K, self.T, self.G, self.D = K, T, G, D


def targets(est_bg, est_tract, N):
    """pymedm's Y vector from block-group and tract estimates (which may be perturbed)."""
    return np.concatenate([est_bg.sum(axis=0), est_tract.flatten("F"),
                           est_bg.flatten("F")]) / N


def _strings(b, name):
    blob, off = b[name + ".blob"].tobytes(), b[name + ".offsets"]
    return [blob[int(a):int(e)].decode() for a, e in zip(off[:-1], off[1:])]


def puma_count(bundle):
    return len(bundle["solve.bg_offset"]) - 1
