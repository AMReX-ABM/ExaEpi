#!/usr/bin/env python
"""What does the replicate allocation matrix actually cost on disk?

The 283-707 MB figures for option A are raw f32. Dense float data compresses badly -- a synthetic
test gave 1.05-1.08x with deflate -- but that was synthetic, and quantisation is the real lever
here rather than entropy coding. TRS only needs w * N accurate enough to split a cell into its
whole and fractional parts, so full float precision is not obviously required.

Measured on real replicates, with the reconstruction error each representation incurs.
"""

import os
import zlib

import numpy as np

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
NM_CELLS = 3_533_563          # measured: NM block-diagonal allocation matrix, all 18 PUMAs


def quantise_rows(a, dtype):
    """Per-donor-row scaled integers. Each row is scaled by its own max, so a row's relative
    precision is 1/max(dtype) of that row rather than of the global maximum."""
    info = np.iinfo(dtype)
    scale = a.max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    q = np.rint(a / scale * info.max).astype(dtype)
    back = q.astype(np.float64) / info.max * scale
    return q, back, scale


def main():
    from livelike import acs, config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    pup = acs.puma(
        "3500804", constraints_selection=EXAEPI_MINIMAL,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder="./cache_minimal", censusapikey=key,
    )
    pmd = PMEDM(
        pup.year, pup.est_ind.index, pup.wt,
        pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
        n_reps=20, random_state=1,
    )
    pmd.solve()
    reps = np.asarray(pmd.almat_reps)
    n, d, b = reps.shape
    print(f"{n} replicates, {d} x {b} cells each\n")

    # Scale this PUMA's per-cell cost up to a whole New Mexico bundle.
    per_cell = lambda nbytes: nbytes / (n * d * b)
    nm = lambda nbytes, N: per_cell(nbytes) * NM_CELLS * N / 1e6

    total_pop = float(pup.est_g2.iloc[:, 0].sum()) if pup.est_g2.shape[1] else 1.0
    print(f"{'representation':<26s}{'bytes/cell':>11s}{'NM N=20':>10s}{'NM N=50':>10s}"
          f"{'max cell err':>14s}")

    rows = []
    f64 = reps.astype(np.float64)

    for name, arr in [("f32 raw", reps.astype(np.float32)),
                      ("f16 raw", reps.astype(np.float16))]:
        raw = arr.nbytes
        err = np.abs(arr.astype(np.float64) - f64).max() * total_pop
        rows.append((name, raw, err))
        z = len(zlib.compress(arr.tobytes(), 6))
        rows.append((f"{name} + deflate", z, err))

    for name, dt in [("uint16 per-row", np.uint16), ("uint8 per-row", np.uint8)]:
        q, back, _ = quantise_rows(f64.reshape(n * d, b), dt)
        err = np.abs(back.reshape(n, d, b) - f64).max() * total_pop
        rows.append((name, q.nbytes, err))
        rows.append((f"{name} + deflate", len(zlib.compress(q.tobytes(), 6)), err))

    for name, nbytes, err in rows:
        print(f"{name:<26s}{per_cell(nbytes):11.2f}{nm(nbytes, 20):9.0f}M{nm(nbytes, 50):9.0f}M"
              f"{err:14.4f}")

    print("\nmax cell err is in PEOPLE: the largest absolute error in a cell's expected headcount")
    print("after round-tripping through that representation. TRS splits a cell into whole and")
    print("fractional parts, so an error well below 1 person changes nothing it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
