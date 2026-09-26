"""Stage S4: student school assignment (replaces upop_to_exaepi.alloc_students).

Same rules as the converter, with keyed draws in place of the shared numpy stream:

  * Levels P, E, M, H, U, C are independent: each takes the schools whose level string contains
    its letter, with capacity ceil(students / number of levels the school serves), and assigns the
    students whose grade falls in that level's range.
  * Scales run from fine to coarse -- block group (12-digit geoid prefix), tract (11), county
    subdivision (10), place (7), county (5); childcare stops at 10 -- with each school's remaining
    capacity carried from one scale to the next. Regions within a scale are disjoint, so they are
    independent.
  * In a region, schools are filled in a keyed random order (STU_SCHOOL_PERM on level, scale,
    region and the school's (geoid, ordinal)) until demand is met; students are taken in canonical
    (bg, h, p) order, so the students left over are the tail of that order. At the last scale
    (except for university) the shortfall is spread over the region's schools in proportion to
    capacity, one keyed draw per extra student (STU_SPILL).
  * Remaining capacity is clipped to at least 1 after each round, as the converter does -- a leak
    that lets a school take a few students more than its capacity, kept for fidelity.
  * University gets a final pass over each home county plus its neighbours, counties visited in
    ascending FIPS with capacity updated after each (neighbourhoods overlap).
  * A student still unplaced stays home and leaves school (grade -1).
"""

import numpy as np

from . import kr64, stages, units

LEVELS = ["P", "E", "M", "H", "U", "C"]
LEVEL_GRADES = {"C": (3, 3), "P": (4, 4), "E": (5, 10), "M": (11, 13), "H": (14, 17), "U": (18, 19)}
SCALES = [12, 11, 10, 7, 5]
CHILDCARE_SCALES = [12, 11, 10]


def prefix(geoid, s):
    return geoid // 10 ** (12 - s)


def allocate(b, P, work, seed, rep):
    """Assign students; sets work[i] to the school's geoid and returns school row per person
    (-1 if none). Updates P['grade'] to -1 for students left unplaced."""
    level_names = _strings(b, "schools.level_names")
    sg = b["schools.geoid"].astype(np.int64)
    so = b["schools.ord"].astype(np.int64)
    s_students = b["schools.students"].astype(np.int64)
    s_level = b["schools.level"].astype(np.int64)
    county_of = b["adjacency.county"].astype(np.int64)
    adj_ip, adj_ix = b["adjacency.indptr"], b["adjacency.indices"]
    school = np.full(len(P["id"]), -1, dtype=np.int64)
    stats = {}
    # The converter keeps only schools located in a populated block group.
    in_pop = np.isin(sg, np.unique(P["bg"]))

    for li, L in units.each(list(enumerate(LEVELS))):
        lo, hi = LEVEL_GRADES[L]
        stud = np.flatnonzero(P["student"] & (P["grade"] >= lo) & (P["grade"] <= hi))
        if len(stud) == 0:
            continue
        # canonical order (bg, h, p) -- P is already in it, so index order is canonical order
        rows = np.array([i for i, lv in enumerate(s_level)
                         if L in level_names[lv] and in_pop[i]], dtype=np.int64)
        nlev = np.array([len(level_names[lv]) for lv in s_level[rows]], dtype=np.int64)
        remaining = (s_students[rows] + nlev - 1) // nlev
        scales = CHILDCARE_SCALES if L == "C" else SCALES
        placed = np.zeros(len(stud), dtype=bool)
        for scale in scales:
            alloc_all = L != "U" and scale == scales[-1]
            todo = np.flatnonzero(~placed)
            if len(todo) == 0:
                break
            reg_s = prefix(P["bg"][stud[todo]], scale)
            reg_c = prefix(sg[rows], scale)
            taken = np.zeros(len(rows), dtype=np.int64)
            for region in units.each(np.unique(reg_s)):
                who = todo[reg_s == region]
                cand = np.flatnonzero(reg_c == region)
                if len(cand) == 0:
                    continue
                w_i, c_i = _fill(who, cand, remaining, rows, sg, so, li, scale, int(region),
                                 alloc_all, seed, rep)
                school[stud[w_i]] = rows[c_i]
                placed[w_i] = True
                np.add.at(taken, c_i, 1)
            remaining = np.maximum(remaining - taken, 1)
        if L == "U":
            todo = np.flatnonzero(~placed)
            home_cty = prefix(P["bg"][stud[todo]], 5)
            sch_cty = prefix(sg[rows], 5)
            for cty in np.unique(home_cty):
                who = todo[home_cty == cty]
                k = np.searchsorted(county_of, cty)
                nb = {int(cty)}
                if k < len(county_of) and county_of[k] == cty:
                    nb |= {int(county_of[j]) for j in adj_ix[adj_ip[k]:adj_ip[k + 1]]}
                cand = np.flatnonzero(np.isin(sch_cty, sorted(nb)))
                if len(cand) == 0:
                    continue
                w_i, c_i = _fill(who, cand, remaining, rows, sg, so, li, 0, int(cty), True, seed,
                                 rep)
                taken = np.zeros(len(rows), dtype=np.int64)
                school[stud[w_i]] = rows[c_i]
                placed[w_i] = True
                np.add.at(taken, c_i, 1)
                remaining = np.maximum(remaining - taken, 1)
        unplaced = stud[~placed]
        P["grade"][unplaced] = -1
        stats[L] = (len(stud), int((~placed).sum()))

    has = school >= 0
    work[has] = sg[school[has]]
    return school, stats


def _fill(who, cand, remaining, rows, sg, so, li, scale, region, alloc_all, seed, rep):
    """One region: (student indices, row-local school indices). who is in canonical order."""
    need = len(who)
    kk = kr64.key(seed, rep, stages.STU_SCHOOL_PERM, li, scale, region, sg[rows[cand]], so[rows[cand]])
    order = cand[np.lexsort((so[rows[cand]], sg[rows[cand]], kr64.draw(kk, 0)))]
    caps = remaining[order]
    cum = np.cumsum(caps)
    total = int(cum[-1])
    if need <= total:
        n_used = min(int(np.searchsorted(cum, need, side="left")) + 1, len(caps))
        counts = caps[:n_used].copy()
        counts[-1] -= int(cum[n_used - 1] - need)
        order = order[:n_used]
    else:
        counts = caps.copy()
        if alloc_all:
            short = need - total
            w = caps if caps.sum() > 0 else np.ones_like(caps)
            # Spill weights in school-identity order, not the shuffled fill order.
            ident = np.lexsort((so[rows[order]], sg[rows[order]]))
            xs = kr64.draw(kr64.key(seed, rep, stages.STU_SPILL, li, scale, region,
                                    np.arange(short, dtype=np.int64)), 0)
            extra = np.bincount(kr64.int_cdf(w[ident], xs), minlength=len(ident))
            counts[ident] += extra
    slot = np.repeat(order, counts)
    return who[:len(slot)], slot[:len(who)]


def _strings(b, name):
    blob, off = b[name + ".blob"].tobytes(), b[name + ".offsets"]
    return [blob[int(a):int(e)].decode() for a, e in zip(off[:-1], off[1:])]
