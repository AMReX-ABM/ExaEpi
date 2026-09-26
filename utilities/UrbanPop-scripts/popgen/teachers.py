"""Stage S5: teacher assignment (replaces upop_to_exaepi.allocate_teachers).

Same rules as the converter, with keyed draws:

  * Three school types, independent of one another (disjoint teacher pools and schools):
        childcare   NAICS 6244, schools of level "C", students with grade <= 3
        secondary   NAICS 6111, schools of level P/E/M/H and their combinations, grades 4-17
        university  NAICS 611,  schools of level "U", grades 18-19
  * A school needs ceil(teachers * placed students / nominal students) teachers, and only schools
    that received students take part. A school listing 0 students needs none (the converter would
    divide by zero there).
  * Only schools located in a worker's home block group are used, as in the converter.
  * Scales fine to coarse (block group ... county) on the TEACHER's work location from S3 and the
    school's location. In each region with demand, schools are filled in keyed order
    (TCH_SCHOOL_PERM) and the teachers are the first of a keyed shuffle of the region's unassigned
    teachers of that type (TCH_PICK) -- replacing polars sample(seed=...), which used one fixed seed
    everywhere and so took the same positions in every region. The i-th picked teacher takes the
    i-th slot. Regions within a scale are disjoint, so they are independent.
  * Then a pass over school counties in ascending FIPS, each drawing on unassigned teachers working
    in the county or its neighbours, with teachers and capacity updated after each county.
  * A teacher's grade: uniform in the school level's grade range (TCH_GRADE draw 0), and a
    graduate-level (19) teacher becomes undergraduate (18) with probability 1/2 (draw 1). The
    teacher's work location becomes the school's.
"""

import numpy as np

from . import kr64, stages, units
from .students import SCALES, _strings, prefix

TYPES = [
    ("childcare", "6244", {"C"}, (0, 3)),
    ("secondary", "6111", {"E", "EM", "EMH", "M", "MH", "H", "P", "PE", "PEM", "PEMH"}, (4, 17)),
    ("university", "611", {"U"}, (18, 19)),
]
LEVEL_RANGE = {"C": (3, 3), "P": (4, 4), "E": (5, 10), "M": (11, 13), "H": (14, 17), "U": (18, 19),
               "PE": (4, 10), "PEM": (4, 13), "PEMH": (4, 17), "EM": (5, 13), "EMH": (5, 17),
               "MH": (11, 17)}


def allocate(b, P, work, school, seed, rep):
    """Assign teachers in place: school row, work geoid and grade for the chosen workers."""
    naics_codes = _strings(b, "naics.codes")
    level_names = _strings(b, "schools.level_names")
    sg = b["schools.geoid"].astype(np.int64)
    so = b["schools.ord"].astype(np.int64)
    s_students = b["schools.students"].astype(np.int64)
    s_teachers = b["schools.teachers"].astype(np.int64)
    s_level = [level_names[v] for v in b["schools.level"]]
    county_of = b["adjacency.county"].astype(np.int64)
    adj_ip, adj_ix = b["adjacency.indptr"], b["adjacency.indices"]
    worker_homes = np.unique(P["bg"][P["employed"]])
    stats = {}

    for ti, (name, code, levels, (glo, ghi)) in units.each(list(enumerate(TYPES))):
        n_code = naics_codes.index(code)
        rows = np.array([i for i in range(len(sg)) if s_level[i] in levels], dtype=np.int64)
        rows = rows[np.isin(sg[rows], worker_homes)]
        # students actually placed at each school, among this type's grade range
        stud = (school >= 0) & P["student"] & (P["grade"] >= glo) & (P["grade"] <= ghi)
        placed = np.bincount(school[stud], minlength=len(sg))[rows]
        keep = placed > 0
        rows, placed = rows[keep], placed[keep]
        nom = s_students[rows]
        need = np.where(nom > 0, (s_teachers[rows] * placed + nom - 1) // np.maximum(nom, 1), 0)
        teach = np.flatnonzero(P["employed"] & (P["naics"] == n_code) & (school < 0))
        free = np.ones(len(teach), dtype=bool)
        required = int(need.sum())

        for scale in SCALES + [0]:
            if scale:
                t_reg = prefix(work[teach], scale)
                s_reg = prefix(sg[rows], scale)
                regions = np.unique(s_reg[need > 0])
            else:
                t_reg = prefix(work[teach], 5)
                s_reg = prefix(sg[rows], 5)
                regions = np.unique(s_reg[need > 0])
            # regions of a scale are disjoint; the county pass (scale 0) overlaps, so it is ordered
            for region in (units.each(regions) if scale else regions):
                cand = np.flatnonzero((s_reg == region) & (need > 0))
                if scale:
                    pool = np.flatnonzero(free & (t_reg == region))
                else:
                    k = np.searchsorted(county_of, region)
                    nb = {int(region)}
                    if k < len(county_of) and county_of[k] == region:
                        nb |= {int(county_of[j]) for j in adj_ix[adj_ip[k]:adj_ip[k + 1]]}
                    pool = np.flatnonzero(free & np.isin(t_reg, sorted(nb)))
                if len(pool) == 0 or len(cand) == 0:
                    continue
                num = min(int(need[cand].sum()), len(pool))
                sk = kr64.key(seed, rep, stages.TCH_SCHOOL_PERM, ti, scale, int(region),
                              sg[rows[cand]], so[rows[cand]])
                order = cand[np.lexsort((so[rows[cand]], sg[rows[cand]], kr64.draw(sk, 0)))]
                cum = np.cumsum(need[order])
                n_used = min(int(np.searchsorted(cum, num, side="left")) + 1, len(order))
                counts = need[order[:n_used]].copy()
                counts[-1] -= int(cum[n_used - 1] - num)
                slots = np.repeat(order[:n_used], counts)
                ti_ = teach[pool]
                tk = kr64.key(seed, rep, stages.TCH_PICK, ti, scale, int(region), P["bg"][ti_],
                              P["h"][ti_], P["p"][ti_])
                pick = pool[np.lexsort((P["p"][ti_], P["h"][ti_], P["bg"][ti_],
                                        kr64.draw(tk, 0)))][:num]
                who = teach[pick]
                school[who] = rows[slots]
                work[who] = sg[rows[slots]]
                P["grade"][who] = _grades(P, who, [s_level[r] for r in rows[slots]], seed, rep)
                free[pick] = False
                np.subtract.at(need, slots, 1)
        stats[name] = (required, int((~free).sum()))
    return stats


def _grades(P, who, levels, seed, rep):
    k = kr64.key(seed, rep, stages.TCH_GRADE, P["bg"][who], P["h"][who], P["p"][who])
    lo = np.array([LEVEL_RANGE[lv][0] for lv in levels], dtype=np.int64)
    hi = np.array([LEVEL_RANGE[lv][1] for lv in levels], dtype=np.int64)
    g = lo + kr64.index(kr64.draw(k, 0), hi - lo + 1)
    g = np.where((g == 19) & (kr64.u01(kr64.draw(k, 1)) < 0.5), 18, g)
    return g.astype(np.int16)
