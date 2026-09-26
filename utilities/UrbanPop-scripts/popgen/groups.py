"""Stages S6-S10: dense ids and group structure (replaces adjust_indexes and group_assignment.py).

S6  school_id: 1 + dense rank of the schools actually used at each work block group, by school
    identity (geoid, ordinal); 0 for anyone not at a school. (The converter's rank also counted the
    "no school" bucket, so school ids shifted by one wherever non-school agents shared a work block
    group; ExaEpi sizes arrays by the largest id, so gap-free ids only help.) household_id = h.
S7  Home groups (assign_home_groups): nborhood uniform over round-half-up(home_pop / 500)
    neighbourhoods, one keyed draw per household (HOME_NB (bg, h)), so households are never split;
    hh_cluster = h mod ceil(households / 4).
S8  Work groups (assign_work_groups) for agents who go to a workplace -- employed, not an educator,
    not working from home. Per (work block group, NAICS): members in keyed order (WG_ORDER),
    establishment count round-half-up(members / mean size) clamped to [1, members], sizes drawn
    from the CBP table keyed per establishment (WG_EST_SIZE) -- replacing the converter's pools
    shared across block groups, which made results depend on visiting order -- members apportioned
    among them by integer largest remainder, each establishment split into teams at the industry's
    target size. work_group = the team's position in a scan over groups sorted by (geoid, NAICS).
S9  School groups (assign_school_groups): per (work block group, school, grade) raw group, classes
    by teacher count clamped to 5-50 students each, excess teachers in admin groups of 20, students
    round-robin up to the floor and then smeared with a keyed draw (CLASS_SMEAR) instead of a hash
    of the global group index. school_class_group = scan over raw groups.
S10 Day neighbourhoods (assign_day_neighborhoods): atoms (a work group, a whole school, a class of a
    school bigger than a neighbourhood, a household that stays home) packed per daytime block group
    into round-half-up(total / 500) bins by the midpoint rule on an integer grid, atoms ordered by a
    keyed draw (DAY_NB) with oversized atoms first.

Every count, rounding and grid cut here is integer arithmetic, and every sort ends in an identity
key, so the C++ port reproduces these stages bit for bit given the same inputs.
"""

import numpy as np

from . import kr64, stages

NBORHOOD_SIZE = 500
WORKGROUP_SIZE = 20
CLASS_SIZE, CLASS_MIN, CLASS_MAX = 20, 5, 50
COLLEGE_INSTRUCTIONAL_FRACTION = 0.1
TRAVEL_WFH = 7


def _runs(sorted_keys):
    """Start indices of runs of equal rows in a lexicographically sorted list of key arrays."""
    n = len(sorted_keys[0])
    brk = np.zeros(n, dtype=bool)
    if n:
        brk[0] = True
    for k in sorted_keys:
        brk[1:] |= k[1:] != k[:-1]
    return np.flatnonzero(brk), n


def school_ids(b, P, work, school):
    """S6: block-group-local dense school id (0 = none)."""
    sg = b["schools.geoid"].astype(np.int64)
    so = b["schools.ord"].astype(np.int64)
    sid = np.zeros(len(P["id"]), dtype=np.int64)
    at = np.flatnonzero((school >= 0) & (P["grade"] != -1))
    if len(at):
        g, o = sg[school[at]], so[school[at]]
        # distinct schools in use, sorted by (geoid, ord); each numbered 1.. within its geoid
        pairs, inv = np.unique(np.stack([g, o]), axis=1, return_inverse=True)
        starts, npairs = _runs([pairs[0]])
        first = np.repeat(starts, np.diff(np.r_[starts, npairs]))
        local = np.arange(npairs) - first + 1
        sid[at] = local[inv.ravel()]
    return sid


def home_groups(P, seed, rep):
    """S7: nborhood and hh_cluster."""
    bg, h = P["bg"], P["h"]
    ubg, inv, pop = np.unique(bg, return_inverse=True, return_counts=True)
    n_hh = np.zeros(len(ubg), dtype=np.int64)
    np.maximum.at(n_hh, inv, h + 1)
    max_nb = np.maximum(1, (2 * pop + NBORHOOD_SIZE) // (2 * NBORHOOD_SIZE))
    nborhood = kr64.index(kr64.draw(kr64.key(seed, rep, stages.HOME_NB, bg, h), 0), max_nb[inv])
    hh_cluster = h % np.maximum(1, (n_hh + 3) // 4)[inv]
    return nborhood.astype(np.int64), hh_cluster.astype(np.int64)


def _apportion(w, total):
    """Split total among len(w) >= 1 groups in proportion to integer weights w, each >= 1."""
    k = len(w)
    spare = total - k
    W = int(w.sum())
    q = w * spare
    base = q // W + 1
    rem = total - int(base.sum())
    if rem > 0:
        base[np.lexsort((np.arange(k), -(q % W)))[:rem]] += 1
    return base


def work_groups(P, work, sid, tables, seed, rep):
    """S8: workgroup (1-based team within (work bg, NAICS); 0 = none) and dense work_group."""
    n = len(P["id"])
    workgroup = np.zeros(n, dtype=np.int64)
    work_group = np.full(n, -1, dtype=np.int64)
    el = np.flatnonzero((P["naics"] != -1) & (sid == 0) & (P["travel"] != TRAVEL_WFH))
    if len(el) == 0:
        return workgroup, work_group
    g_geo, g_naics = work[el], P["naics"][el].astype(np.int64)
    ok = kr64.draw(kr64.key(seed, rep, stages.WG_ORDER, g_geo, g_naics, P["bg"][el], P["h"][el],
                            P["p"][el]), 0)
    order = np.lexsort((P["p"][el], P["h"][el], P["bg"][el], ok, g_naics, g_geo))
    s = el[order]
    s_geo, s_naics = work[s], P["naics"][s].astype(np.int64)
    starts, ns = _runs([s_geo, s_naics])
    ends = np.r_[starts[1:], ns]
    next_id = 0
    for lo, hi in zip(starts, ends):
        pop = int(hi - lo)
        geo, nai = int(s_geo[lo]), int(s_naics[lo])
        state = geo // 10**10
        target = max(1, tables.target(state, nai))
        if tables.has_est(state, nai):
            k = min(max(1, int(np.floor(pop / tables.mean(state, nai) + 0.5))), pop)
            sizes = tables.sample(state, nai, kr64.key(seed, rep, stages.WG_EST_SIZE, geo, nai,
                                                       np.arange(k, dtype=np.int64)))
        else:
            k = min(max(1, int(np.floor(pop / target + 0.5))), pop)
            sizes = np.full(k, target, dtype=np.int64)
        est = _apportion(np.asarray(sizes, dtype=np.int64), pop)
        est_of = np.repeat(np.arange(k), est)
        pos = np.arange(pop) - (np.cumsum(est) - est)[est_of]
        n_teams = np.maximum(1, (2 * est + target) // (2 * target))
        team_base = np.cumsum(n_teams) - n_teams
        wg = team_base[est_of] + pos % n_teams[est_of] + 1
        workgroup[s[lo:hi]] = wg
        work_group[s[lo:hi]] = next_id + wg - 1
        next_id += int(n_teams.sum())
    return workgroup, work_group


def school_groups(b, P, work, school, sid, seed, rep):
    """S9: school_class and dense school_class_group (0 / -1 for anyone not at a school)."""
    n = len(P["id"])
    school_class = np.zeros(n, dtype=np.int64)
    scg = np.full(n, -1, dtype=np.int64)
    en = np.flatnonzero(sid > 0)
    if len(en) == 0:
        return school_class, scg
    sg = b["schools.geoid"].astype(np.int64)
    so = b["schools.ord"].astype(np.int64)
    grade = P["grade"].astype(np.int64)
    order = np.lexsort((P["id"][en], grade[en], sid[en], work[en]))
    s = en[order]
    starts, ns = _runs([work[s], sid[s], grade[s]])
    ends = np.r_[starts[1:], ns]
    next_id = 0
    for lo, hi in zip(starts, ends):
        members = s[lo:hi]
        is_st = P["naics"][members] == -1
        n_st = int(is_st.sum())
        n_te = len(members) - n_st
        n_classes = 0
        if n_st > 0:
            college = grade[members[0]] > 17
            eff = n_te * COLLEGE_INSTRUCTIONAL_FRACTION if college else float(n_te)
            raw = max(1, int(eff)) if eff > 0.0 else max(1, -(-n_st // CLASS_SIZE))
            n_classes = max(-(-n_st // CLASS_MAX), min(raw, max(1, n_st // CLASS_MIN)))
        excess = n_te - n_classes
        n_admin = -(-excess // WORKGROUP_SIZE) if excess > 0 else 0
        base = next_id
        next_id += n_classes + n_admin
        # rank within the raw group by agent id, separately for students and teachers
        cls = np.empty(len(members), dtype=np.int64)
        t_rank = np.arange(n_te)
        t_cls = np.where(t_rank < n_classes, t_rank, 0)
        if n_admin > 0:
            t_cls = np.where(t_rank < n_classes, t_rank, -2 - (t_rank - n_classes) % n_admin)
        cls[~is_st] = t_cls
        if n_st > 0:
            r = np.arange(n_st)
            m0 = members[0]
            smear = kr64.index(kr64.draw(kr64.key(seed, rep, stages.CLASS_SMEAR, int(work[m0]),
                                                  int(sg[school[m0]]), int(so[school[m0]]),
                                                  int(grade[m0]), r), 0), n_classes)
            cls[is_st] = np.where(r < CLASS_MIN * n_classes, r % n_classes, smear)
        school_class[members] = cls
        scg[members] = np.where(cls >= 0, base + cls, base + n_classes + (-2 - cls))
    return school_class, scg


def day_neighborhoods(b, P, work, school, sid, workgroup, school_class, scg, seed, rep):
    """S10: work_nborhood, a dense id per daytime block group."""
    n = len(P["id"])
    so = b["schools.ord"].astype(np.int64)
    sg = b["schools.geoid"].astype(np.int64)
    at_school = sid != 0
    at_work = workgroup > 0
    stay = ~at_school & ~at_work
    day = np.where(stay, P["bg"], work)
    # a school too big for one neighbourhood goes in class by class
    s_key = np.where(at_school, day * 100000 + sid, -1)
    uk, inv, cnt = np.unique(s_key[at_school], return_inverse=True, return_counts=True)
    big = np.zeros(n, dtype=bool)
    big[at_school] = cnt[inv] > NBORHOOD_SIZE
    kind = np.full(n, -1, dtype=np.int64)
    a = np.zeros(n, dtype=np.int64)
    bb = np.zeros(n, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    kind[at_work] = 0
    a[at_work], bb[at_work] = P["naics"][at_work], workgroup[at_work]
    whole = at_school & ~big
    kind[whole] = 1
    a[whole], bb[whole] = sg[school[whole]], so[school[whole]]
    kind[big] = 2
    a[big], bb[big], c[big] = so[school[big]], P["grade"][big], school_class[big]
    kind[stay] = 3
    a[stay] = P["h"][stay]
    # atoms = distinct (day, kind, a, b, c)
    atoms, ainv, asize = np.unique(np.stack([day, kind, a, bb, c]), axis=1, return_inverse=True,
                                   return_counts=True)
    ainv = ainv.ravel()
    A_day, A_kind, A_a, A_b, A_c = atoms
    over = asize > NBORHOOD_SIZE
    dk = kr64.draw(kr64.key(seed, rep, stages.DAY_NB, A_day, A_kind, A_a, A_b, A_c), 0)
    order = np.lexsort((A_c, A_b, A_a, A_kind, dk, ~over, A_day))
    d_s, sz, ov = A_day[order], asize[order].astype(np.int64), over[order]
    starts, na = _runs([d_s])
    run = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, na]))
    pos = np.arange(na) - starts[run]
    n_over = np.bincount(run, weights=ov, minlength=len(starts)).astype(np.int64)
    packed = np.where(ov, 0, sz)
    ex = np.cumsum(packed) - packed
    ex = ex - ex[starts][run]
    total = np.bincount(run, weights=packed, minlength=len(starts)).astype(np.int64)
    n_bins = np.maximum(1, (2 * total + NBORHOOD_SIZE) // (2 * NBORHOOD_SIZE))
    T = np.maximum(total, 1)[run]
    grid = np.minimum(((2 * ex + packed) * n_bins[run]) // (2 * T), n_bins[run] - 1)
    bin_s = np.where(ov, pos, n_over[run] + grid)
    new = np.ones(na, dtype=bool)
    new[1:] = (run[1:] != run[:-1]) | (bin_s[1:] != bin_s[:-1])
    dense = np.cumsum(new) - 1
    dense = dense - dense[starts][run]
    atom_bin = np.empty(na, dtype=np.int64)
    atom_bin[order] = dense
    return atom_bin[ainv]
