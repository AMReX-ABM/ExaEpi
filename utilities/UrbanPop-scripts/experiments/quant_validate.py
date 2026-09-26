#!/usr/bin/env python
"""Does quantising the allocation matrix change the population TRS produces?

Reconstruction error is the wrong test. TRS splits each cell into a whole and a fractional part,
takes the whole parts, and samples the remainder in proportion to the fractions -- so it only cares
whether those parts survive the round trip, which is far weaker than exact reconstruction.

The right yardstick is a perturbation already known to be harmless. A TRS re-draw from the SAME
allocation matrix moves per-block-group population by a CV of 0.0044, and arm B established that
variation at that level does not move epidemic outcomes at all. So if quantising perturbs the
population by less than a TRS re-draw does, it is beneath the noise the model already ignores.

Compares, on identical seeds:
    f32 vs uint16-quantised   -- the cost of quantising
    f32 seed 0 vs seed 1      -- the cost of simply re-drawing, as the reference
"""

import os

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


def quantise_rows(a, dtype):
    """Per-donor-row scaled integers, round-tripped back to float."""
    info = np.iinfo(dtype)
    scale = a.max(axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    q = np.rint(a / scale * info.max).astype(dtype)
    return q.astype(np.float64) / info.max * scale


def bg_pop(sims, pup, sim_id):
    """Per-block-group population for one simulation of a longform synthesize() result."""
    hh = pup.sporder.groupby(level=0).size()
    s = sims[sims["sim"] == sim_id].reset_index()
    idcol = s.columns[0]
    s = s.assign(people=s["count"] * s[idcol].map(hh).fillna(1).values)
    return s.groupby("geoid")["people"].sum()


def compare(a, b, label):
    common = a.index.intersection(b.index)
    x, y = a.reindex(common).to_numpy(), b.reindex(common).to_numpy()
    denom = np.where(x > 0, x, np.nan)
    rel = np.abs(y - x) / denom
    ident = int((x == y).sum())
    print(f"  {label:<34s} identical {ident:4d}/{len(common)}  "
          f"mean|rel| {np.nanmean(rel):.6f}  max|rel| {np.nanmax(rel):.6f}  "
          f"CV-equivalent {np.nanstd(rel):.6f}")
    return np.nanmean(rel)


def main():
    from livelike import acs, config, homesim
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    pup = acs.puma(
        "3500804", constraints_selection=EXAEPI_MINIMAL,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder="./cache_min2", censusapikey=key,
    )
    pmd = PMEDM(
        pup.year, pup.est_ind.index, pup.wt,
        pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
    )
    pmd.solve()
    a64 = np.asarray(pmd.almat, dtype=np.float64)
    print(f"allocation matrix {a64.shape}   value range [{a64.min():.3e}, {a64.max():.3e}]\n")

    variants = {
        "f32": a64.astype(np.float32).astype(np.float64),
        "uint16": quantise_rows(a64, np.uint16),
        "uint8": quantise_rows(a64, np.uint8),
    }

    synth = lambda m, seed: homesim.synthesize(
        m, pup.est_ind, pup.est_g2, pup.sporder, nsim=1, random_state=seed, longform=True)

    base = bg_pop(synth(variants["f32"], 0), pup, 0)
    ref = bg_pop(synth(variants["f32"], 1), pup, 0)

    print("against f32 at the same seed:")
    q16 = compare(base, bg_pop(synth(variants["uint16"], 0), pup, 0), "uint16 quantised")
    q8 = compare(base, bg_pop(synth(variants["uint8"], 0), pup, 0), "uint8 quantised")
    print("\nreference -- the same f32 matrix, different TRS seed:")
    redraw = compare(base, ref, "f32 seed 0 vs seed 1")

    print()
    for name, val in (("uint16", q16), ("uint8", q8)):
        ratio = val / redraw if redraw else float("inf")
        verdict = ("beneath TRS re-draw noise -- safe" if ratio < 1
                   else "LARGER than a TRS re-draw -- not safe")
        print(f"  {name}: {ratio:.3f}x the effect of a TRS re-draw  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
