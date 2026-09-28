"""Stage S4: student school assignment (replaces upop_to_exaepi.alloc_students).

Same rules as the converter, with keyed draws in place of the shared numpy stream:

  * Levels P, E, M, H, U, C are independent: each takes the schools whose level string contains
    its letter and that have places for it -- for P, E, M and H the school's own places for that
    level (schools.level_places, from NCES's per-grade counts or the grade span), for U and C its
    `students` -- and assigns the students whose grade falls in that level's range.
  * Scales run from fine to coarse -- block group (12-digit geoid prefix), tract (11), county
    subdivision (10), place (7), county (5); childcare stops at 10 -- with each school's remaining
    places carried from one scale to the next. Regions within a scale are disjoint, so they are
    independent.
  * In a region, schools are filled in a keyed random order (STU_SCHOOL_PERM on level, scale,
    region and the school's (geoid, ordinal)) until demand is met; students are taken in canonical
    (bg, h, p) order, so the students left over are the tail of that order.
  * Only childcare overfills: at its last scale the shortfall is spread over the region's schools
    in proportion to their places, one keyed draw per extra student (STU_SPILL). University
    overfills in its neighbour pass instead. A preschool or K-12 student with no place in reach
    stays home, standing in for online and home schooling -- squeezing them in anyway put 1,476 CA
    schools at over 5x their listed enrollment.
  * Remaining places are clipped at 0 after each round: a school squeezed past capacity has
    nothing left.
  * University and K-12 get a final pass over each home county plus its neighbours, counties
    visited in ascending FIPS with places updated after each (neighbourhoods overlap). K-12 places
    don't follow county lines (Espanola straddles Rio Arriba / Santa Fe).
  * Preschoolers still unplaced then take the childcare places left over (level index
    PRESCHOOL_IN_CHILDCARE, childcare scales, no overfill), and become childcare children
    (grade 3) there. NCES reports no preschool counts for some states, CA among them.
  * A student still unplaced stays home and leaves school (grade -1).
"""

import numpy as np

from . import kr64, stages, units

LEVELS = ["P", "E", "M", "H", "U", "C"]
LEVEL_GRADES = {"C": (3, 3), "P": (4, 4), "E": (5, 10), "M": (11, 13), "H": (14, 17), "U": (18, 19)}
SCALES = [12, 11, 10, 7, 5]
CHILDCARE_SCALES = [12, 11, 10]
# columns of schools.level_places
PLACE_LEVELS = "PEMH"
# levels whose last tier places every remaining student past capacity
OVERFLOW = ("C", "U")
# levels with a final pass over each home county and its neighbours
NEIGHBOUR_PASS = ("E", "M", "H", "U")
# the level word keying the pass placing unplaced preschoolers in childcare
PRESCHOOL_IN_CHILDCARE = len(LEVELS)


def prefix(geoid, s):
    return geoid // 10 ** (12 - s)


def allocate(b, P, work, seed, rep):
    """Assign students; sets work[i] to the school's geoid and returns school row per person
    (-1 if none). Updates P['grade'] to -1 for students left unplaced, and to 3 for preschoolers
    placed in childcare."""
    level_names = _strings(b, "schools.level_names")
    sg = b["schools.geoid"].astype(np.int64)
    so = b["schools.ord"].astype(np.int64)
    s_students = b["schools.students"].astype(np.int64)
    s_places = b["schools.level_places"].astype(np.int64).reshape(len(sg), len(PLACE_LEVELS))
    s_level = b["schools.level"].astype(np.int64)
    county_of = b["adjacency.county"].astype(np.int64)
    adj_ip, adj_ix = b["adjacency.indptr"], b["adjacency.indices"]
    school = np.full(len(P["id"]), -1, dtype=np.int64)
    ctx = (P, school, sg, so, county_of, adj_ip, adj_ix, seed, rep)
    # The converter keeps only schools located in a populated block group.
    in_pop = np.isin(sg, np.unique(P["bg"]))
    done = {}

    for li, L in units.each(list(enumerate(LEVELS))):
        lo, hi = LEVEL_GRADES[L]
        stud = np.flatnonzero(P["student"] & (P["grade"] >= lo) & (P["grade"] <= hi))
        if len(stud) == 0:
            continue
        # canonical order (bg, h, p) -- P is already in it, so index order is canonical order
        cap = s_places[:, PLACE_LEVELS.index(L)] if L in PLACE_LEVELS else s_students
        rows = np.array([i for i, lv in enumerate(s_level)
                         if L in level_names[lv] and in_pop[i] and cap[i] > 0], dtype=np.int64)
        remaining = cap[rows].copy()
        scales = CHILDCARE_SCALES if L == "C" else SCALES
        placed = _place(ctx, stud, rows, remaining, li, scales, L in OVERFLOW and L != "U",
                        L in NEIGHBOUR_PASS, L == "U")
        done[L] = (stud, placed, rows, remaining)

    if "P" in done and "C" in done:
        stud, placed, _, _ = done["P"]
        _, _, rows, remaining = done["C"]
        todo = np.flatnonzero(~placed)
        if len(todo):
            got = _place(ctx, stud[todo], rows, remaining, PRESCHOOL_IN_CHILDCARE,
                         CHILDCARE_SCALES, False, False, False)
            P["grade"][stud[todo[got]]] = LEVEL_GRADES["C"][0]
            placed[todo[got]] = True

    stats = {}
    for L in LEVELS:
        if L in done:
            stud, placed = done[L][:2]
            P["grade"][stud[~placed]] = -1
            stats[L] = (len(stud), int((~placed).sum()))
    has = school >= 0
    work[has] = sg[school[has]]
    return school, stats


def _place(ctx, stud, rows, remaining, li, scales, spill, neighbours, neighbour_overflow):
    """Place students stud (canonical order) into schools rows, whose remaining places are updated
    in place; sets school[] and returns which of stud were placed. spill: overfill at the last
    scale; neighbours: then a county-neighbourhood pass, overfilling if neighbour_overflow."""
    P, school, sg, so, county_of, adj_ip, adj_ix, seed, rep = ctx
    placed = np.zeros(len(stud), dtype=bool)
    for scale in scales:
        alloc_all = spill and scale == scales[-1]
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
        remaining[:] = np.maximum(remaining - taken, 0)
    if neighbours:
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
            w_i, c_i = _fill(who, cand, remaining, rows, sg, so, li, 0, int(cty),
                             neighbour_overflow, seed, rep)
            taken = np.zeros(len(rows), dtype=np.int64)
            school[stud[w_i]] = rows[c_i]
            placed[w_i] = True
            np.add.at(taken, c_i, 1)
            remaining[:] = np.maximum(remaining - taken, 0)
    return placed


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
