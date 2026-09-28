"""Stages S0-S2: the person table, childcare, and the worker/student split.

S0  Person attributes from the donor records, sorted into canonical order (bg, h, p) -- block
    group, dense household index, person index within the household -- whatever order the rows
    arrive in (placement.expand emits PUMA by PUMA, and NM's PUMAs interleave in geoid order).
    Agent id is the position in that order; S4 fills schools and S9 ranks classes by it. Grade
    uses ExaEpi's coding: the bundle's donors.grade is pr_grade's category index (preschool 1 ...
    grad 16, from PUMS SCHG), and process_upop shifts it by 3, so childcare = 3, preschool = 4,
    kindergarten = 5, 1st = 6 ... 12th = 17, undergraduate = 18, graduate = 19.
S1  Childcare (upop_to_exaepi.set_childcare): children under 5 not already in school go to
    center-based care with probability 0.32 at age 0, 0.47 at 1-2, 0.83 at 3-4 (NCES). One keyed
    Bernoulli per person, key CHILDCARE (bg, h, p).
S2  Split (upop_to_exaepi.generate_nt_dt): employed = has an industry and is either not in school
    or older than 26 (as coded; the original comment says 25). Students are the rest of those
    with a grade, and lose their industry. Everyone else stays home.
"""

import numpy as np

from . import kr64, stages

CHILDCARE_GRADE = 3
GRADE_SHIFT = 3
CHILDCARE_PROB = np.array([0.32, 0.47, 0.47, 0.83, 0.83])


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


def childcare(P, seed, rep):
    age = P["age"]
    elig = (P["grade"] == -1) & (age >= 0) & (age < 5)
    idx = np.flatnonzero(elig)
    u = kr64.u01(kr64.draw(kr64.key(seed, rep, stages.CHILDCARE, P["bg"][idx], P["h"][idx],
                                    P["p"][idx]), 0))
    P["grade"][idx[u < CHILDCARE_PROB[age[idx]]]] = CHILDCARE_GRADE


def split(P):
    in_school = P["grade"] != -1
    employed = (P["naics"] != -1) & (~in_school | (P["age"] > 26))
    student = in_school & ~employed
    P["employed"] = employed
    P["student"] = student
    P["naics"] = np.where(student, -1, P["naics"]).astype(np.int16)
    # Workers leave school (alloc_workers sets grade -1 for everyone it places).
    P["grade"] = np.where(employed, -1, P["grade"]).astype(np.int16)
