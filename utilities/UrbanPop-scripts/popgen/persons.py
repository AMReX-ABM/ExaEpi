"""Stages S0-S2: the person table, childcare, and the worker/student split.

S0  Person attributes from the donor records, sorted into canonical order (bg, h, p) -- block
    group, dense household index, person index within the household -- whatever order the rows
    arrive in (placement.expand emits PUMA by PUMA, and NM's PUMAs interleave in geoid order).
    Agent id is the position in that order; S4 fills schools and S9 ranks classes by it. Grade
    uses ExaEpi's coding: the bundle's donors.grade is pr_grade's category index (preschool 1 ...
    grad 16, from PUMS SCHG), and process_upop shifts it by 3, so childcare = 3, preschool = 4,
    kindergarten = 5, 1st = 6 ... 12th = 17, undergraduate = 18, graduate = 19.
S1  Childcare (upop_to_exaepi.set_childcare): children under 5 not already in school go to
    center-based care until each age's share in care, preschoolers included, matches NCES's 2019
    rate (14.1% at age 0, 26.5% at 1-2, 62.5% at 3-4; kindergartners are left out of the count).
    A child in a household where every adult (18+) has an industry is twice as likely to be
    picked. The converter finds its scale by bisection; with weights of only 1 and 2 it has a
    closed form (_care_probs). One keyed Bernoulli per person, key CHILDCARE (bg, h, p).
S2  Split (upop_to_exaepi.generate_nt_dt): employed = has an industry and is either not in school
    or older than 26 (as coded; the original comment says 25). Students are the rest of those
    with a grade, and lose their industry. Everyone else stays home.
"""

import numpy as np

from . import kr64, stages

CHILDCARE_GRADE = 3
GRADE_SHIFT = 3
PRESCHOOL_GRADE = 4
KINDERGARTEN_GRADE = 5
# share of all children of each age (0-4) in center-based care: NCES Digest table 202.30, 2019
CENTER_CARE_RATE = np.array([0.141, 0.265, 0.265, 0.625, 0.625])
# a child in a household where every adult works is this many times as likely to be picked
WORKING_HOUSEHOLD_CARE_RATIO = 2


def build(b, pers, seed, rep):
    """Person table (dict of arrays) for expanded persons, with S1 and S2 applied."""
    o = np.lexsort((pers["p"], pers["h"], pers["bg"]))
    pers = {k: pers[k][o] for k in ("bg", "h", "p", "src")}
    src = pers["src"]
    grade = b["donors.grade"][src].astype(np.int16)
    grade = np.where(grade >= 0, grade + GRADE_SHIFT, -1).astype(np.int16)
    P = {
        "bg": pers["bg"].astype(np.int64),
        "h": pers["h"].astype(np.int64),
        "p": pers["p"].astype(np.int64),
        "age": b["donors.age"][src].astype(np.int16),
        "sex": b["donors.sex"][src].astype(np.int8),
        "race": b["donors.race"][src].astype(np.int8),
        "naics": b["donors.naics"][src].astype(np.int16),
        "travel": b["donors.travel"][src].astype(np.int8),
        "jwmnp": b["donors.jwmnp"][src].astype(np.int16),
        "veh_occ": b["donors.veh_occ"][src].astype(np.int8),
        "grade": grade,
    }
    P["id"] = np.arange(len(src), dtype=np.int64)
    childcare(P, seed, rep)
    split(P)
    return P


def _care_probs(wanted, n1, n2):
    """Probabilities (p1, p2) for children of weight 1 and 2, p_w = min(1, k * w), with k such that
    n1 * p1 + n2 * p2 = wanted (everyone, if wanted is more than there are)."""
    if wanted <= 0:
        return 0.0, 0.0
    if wanted >= n1 + n2:
        return 1.0, 1.0
    if 2.0 * wanted <= n1 + 2 * n2:  # k <= 1/2: nobody capped
        k = wanted / (n1 + 2 * n2)
        return k, 2.0 * k
    return (wanted - n2) / n1, 1.0  # weight-2 children all in care


def childcare(P, seed, rep):
    age, grade = P["age"], P["grade"]
    # households where every adult has an industry; rows are sorted by (bg, h, p)
    new = np.ones(len(age), dtype=bool)
    new[1:] = (P["bg"][1:] != P["bg"][:-1]) | (P["h"][1:] != P["h"][:-1])
    hh = np.cumsum(new) - 1
    idle = np.zeros(int(hh[-1]) + 1 if len(hh) else 0, dtype=bool)
    idle[hh[(age >= 18) & (P["naics"] == -1)]] = True
    all_work = ~idle[hh]
    prob = np.zeros(len(age))
    for a in range(len(CENTER_CARE_RATE)):
        at_age = (age == a) & (grade < KINDERGARTEN_GRADE)
        wanted = CENTER_CARE_RATE[a] * int(at_age.sum()) - int((at_age & (grade == PRESCHOOL_GRADE)).sum())
        elig = (age == a) & (grade == -1)
        n2 = int((elig & all_work).sum())
        p1, p2 = _care_probs(wanted, int(elig.sum()) - n2, n2)
        prob[elig] = np.where(all_work[elig], p2, p1)
    idx = np.flatnonzero((grade == -1) & (age >= 0) & (age < len(CENTER_CARE_RATE)))
    u = kr64.u01(kr64.draw(kr64.key(seed, rep, stages.CHILDCARE, P["bg"][idx], P["h"][idx],
                                    P["p"][idx]), 0))
    P["grade"][idx[u < prob[idx]]] = CHILDCARE_GRADE


def split(P):
    in_school = P["grade"] != -1
    employed = (P["naics"] != -1) & (~in_school | (P["age"] > 26))
    student = in_school & ~employed
    P["employed"] = employed
    P["student"] = student
    P["naics"] = np.where(student, -1, P["naics"]).astype(np.int16)
    # Workers leave school (alloc_workers sets grade -1 for everyone it places).
    P["grade"] = np.where(employed, -1, P["grade"]).astype(np.int16)
