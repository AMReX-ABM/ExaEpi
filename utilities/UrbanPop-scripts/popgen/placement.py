"""From a solved allocation to whole households and persons (TRS integerization and expansion).

The allocation holds EXPECTED households per (donor, block group), vacant units included. For each
block group:

  1. Only occupied donors can place people, so vacant units (true size 0) are dropped here. They
     still did their job in the solve, satisfying the housing-unit constraints.
  2. The number of occupied households to place is this draw's perturbed block-group population
     divided by the mean true household size of the occupied allocation, rounded half to even.
     Targeting through TRUE sizes matters: livelike's est_ind `population` is fractional (it
     rescales by the PUMS person/household weight ratio), so taking the allocation's household
     count at face value under-delivers people by ~9% statewide.
  3. TRS (truncate, replicate, sample): keep each cell's whole part; fill the shortfall by drawing
     cells with probability proportional to their fractional parts, with replacement; or, if the
     whole parts already exceed the target, remove one household from each of the first cells of a
     keyed shuffle of the nonzero cells.

Keys: TRS (bg, k) for the k-th shortfall draw, TRS_TRIM (bg, donor_row) for the trim order. Every
draw is a pure function of the block group and donor identities, so block groups can be processed
in any order, on any rank.

Expansion: a block group's households are ordered by donor row, repeated by count, and numbered
h = 0, 1, ... densely; each household copy contributes its donor's persons in SPORDER order,
numbered p = 0, 1, .... (bg, h, p) is the person identity every later stage keys on.
"""

import numpy as np

from . import kr64, stages


def true_sizes(b, prob):
    """Persons in each of the PUMA's donor households (0 for vacant or unmatched rows)."""
    off = b["donors.hh_offset"]
    di = prob.donor_index.astype(np.int64)
    ok = di >= 0
    size = np.zeros(prob.D, dtype=np.int64)
    size[ok] = off[di[ok] + 1] - off[di[ok]]
    return size


def trs(vals, target, seed, rep, bg, rows):
    """Integer household counts for one block group; rows are the cells' donor-row identities."""
    whole = np.floor(vals).astype(np.int64)
    short = int(target) - int(whole.sum())
    if short > 0:
        frac = vals - whole
        if np.cumsum(frac)[-1] <= 0:
            return whole
        xs = kr64.draw(kr64.key(seed, rep, stages.TRS, bg, np.arange(short, dtype=np.int64)), 0)
        whole += np.bincount(kr64.float_cdf(frac, xs), minlength=len(vals))
    elif short < 0:
        nz = np.flatnonzero(whole > 0)
        if len(nz):
            ids = rows[nz]
            order = kr64.shuffle_order(kr64.key(seed, rep, stages.TRS_TRIM, bg, ids), ids)
            whole[nz[order[:min(-short, len(nz))]]] -= 1
    return whole


def place_puma(b, prob, al, bg_pop, seed, rep):
    """(bg_geoid, donor_row, count) placements for one PUMA; bg_pop is the perturbed population."""
    size = true_sizes(b, prob)
    occ = np.flatnonzero(size > 0)
    out_bg, out_row, out_ct = [], [], []
    for j in range(prob.G):
        vals = al[occ, j]
        mass = np.cumsum(vals)[-1] if len(vals) else 0.0
        if mass <= 0:
            continue
        mean_size = np.cumsum(vals * size[occ])[-1] / mass
        target = int(np.rint(bg_pop[j] / mean_size))
        counts = trs(vals, target, seed, rep, int(prob.bg_geoid[j]), occ)
        keep = counts > 0
        out_bg.append(np.full(int(keep.sum()), prob.bg_geoid[j], dtype=np.int64))
        out_row.append(occ[keep])
        out_ct.append(counts[keep])
    return (np.concatenate(out_bg), np.concatenate(out_row), np.concatenate(out_ct))


def expand(b, placements):
    """Persons from placements: dict of arrays bg, h, p (identity) and src (donor person row).

    placements: (bg_geoid, donor_global, count), each block group's rows contiguous and sorted by
    donor row, as place_puma emits them (with rows mapped to global donor indices).
    """
    bg, donor, count = placements
    off = b["donors.hh_offset"]
    hh_bg = np.repeat(bg, count)
    hh_donor = np.repeat(donor, count)
    # Dense household index within each block group, in placement order (sorted by donor row).
    starts = np.flatnonzero(np.r_[True, hh_bg[1:] != hh_bg[:-1]])
    first = np.repeat(starts, np.diff(np.r_[starts, len(hh_bg)]))
    h = np.arange(len(hh_bg)) - first
    sizes = off[hh_donor + 1] - off[hh_donor]
    person_hh = np.repeat(np.arange(len(hh_bg)), sizes)
    p = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return {
        "bg": hh_bg[person_hh],
        "h": h[person_hh].astype(np.int32),
        "p": p.astype(np.int16),
        "src": off[hh_donor[person_hh]] + p,
        "n_households": len(hh_bg),
    }
