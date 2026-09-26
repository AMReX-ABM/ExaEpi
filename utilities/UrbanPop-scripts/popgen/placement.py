"""From a solved allocation to whole households and persons, following livelike.homesim.synthesize.

The allocation holds EXPECTED households per (donor, block group), vacant units included. This
follows UrbanPop's own synthesis step (livelike/homesim.py, synthesize), with keyed draws:

  1. Reweight each donor household: almat_adj = almat * est_ind.population / household size.
     livelike counts each person in est_ind as (person weight / householder's person weight), so
     the solve fits the ACS in those units while an agent is one whole person. The reweighting is
     the household-level part of the conversion -- a family whose members carry high person
     weights gets proportionally more copies -- and it is what keeps UrbanPop's age mix near the
     ACS (NM: 65+ +3.7%, under 18 -2.0%; skipping it gave +6.9% and -1.2% expected, +9% and -4.6%
     realized). Vacant units (size 0) keep weight 1. Household size is livelike's own member count
     (solve.donor_hh_size, from sporder).
  2. Fractional counts per (category, block group), categories being household type x size, group
     quarters and vacant: tosamp = res^T almat_adj, summed sequentially over donor rows.
  3. One TRS over the whole PUMA's (category x block group) matrix, as livelike does: whole parts,
     then round(sum of fractional parts) extra cells drawn without replacement in proportion to
     the fractional parts -- sequential keyed draws over the category-major flattened matrix,
     removing each chosen cell (HH_TYPE_TRS (puma, k)).
  4. For each (block group, category), that many donor households drawn with replacement in
     proportion to almat_adj[:, g] * res[:, category] (HH_DRAW (bg, category, k)).

Vacant households are drawn (they are part of the category scheme) but contribute nobody, so they
are dropped before expansion. Every draw is keyed on PUMA, block group and category identities,
so PUMAs can be processed in any order, on any rank.

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


def categories(prob, cols):
    """(D x V) residential category indicators: household type x size, group quarters, vacant."""
    hht = [j for j, c in enumerate(cols) if c.startswith("hht") and "hhsize" in c]
    if not hht:
        raise ValueError("constraints must include household type by household size")
    gq, occ = cols.index("group_quarters_pop"), cols.index("occhu")
    vacant = ((prob.C[:, occ] == 0) & (prob.C[:, gq] == 0)).astype(np.float64)
    return np.column_stack([prob.C[:, hht], prob.C[:, gq], vacant])


def _seq_colsum(M):
    """Column sums accumulated sequentially down the rows (reproducible in C++)."""
    return np.cumsum(M, axis=0)[-1] if len(M) else np.zeros(M.shape[1])


def place_puma(b, prob, al, cols, seed, rep):
    """(bg_geoid, donor_row, count) placements of occupied households for one PUMA."""
    pop = prob.C[:, cols.index("population")]
    nm = prob.donor_hh_size.astype(np.float64)
    adj = np.where(nm > 0, pop / np.where(nm > 0, nm, 1.0), 1.0)
    A = al * adj[:, None]                                          # D x G
    R = categories(prob, cols)                                     # D x V
    V, G = R.shape[1], prob.G
    tosamp = np.stack([_seq_colsum(R[:, v][:, None] * A) for v in range(V)])   # V x G

    # 3. TRS over the whole PUMA matrix, category-major (livelike flattens (V, G) row-major).
    whole = np.floor(tosamp).astype(np.int64).ravel()
    frac = (tosamp - np.floor(tosamp)).ravel()
    extra = int(np.rint(np.cumsum(frac)[-1])) if len(frac) else 0
    f = frac.copy()
    fips = int(prob.fips)
    for k in range(extra):
        if np.cumsum(f)[-1] <= 0:
            break
        i = int(kr64.float_cdf(f, kr64.draw(kr64.key(seed, rep, stages.HH_TYPE_TRS, fips, k), 0)))
        whole[i] += 1
        f[i] = 0.0
    counts_vg = whole.reshape(V, G)

    # 4. Donors within each (block group, category), with replacement.
    size = true_sizes(b, prob)
    per_bg = []
    for g in range(G):
        n_d = np.zeros(prob.D, dtype=np.int64)
        bg = int(prob.bg_geoid[g])
        for v in np.flatnonzero(counts_vg[:, g]):
            w = A[:, g] * R[:, v]
            if np.cumsum(w)[-1] <= 0:
                continue
            n = int(counts_vg[v, g])
            xs = kr64.draw(kr64.key(seed, rep, stages.HH_DRAW, bg, int(v),
                                    np.arange(n, dtype=np.int64)), 0)
            n_d += np.bincount(kr64.float_cdf(w, xs), minlength=prob.D)
        keep = np.flatnonzero((n_d > 0) & (size > 0) & (prob.donor_index >= 0))
        per_bg.append((np.full(len(keep), bg, dtype=np.int64), keep, n_d[keep]))
    return tuple(np.concatenate([x[i] for x in per_bg]) for i in range(3))


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
