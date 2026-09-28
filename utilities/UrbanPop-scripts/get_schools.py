#!/usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import configparser
import glob
import re
import time
import psutil
import os
import censusgeocode as cg
import sys
import shapely.geometry
from colorama import Fore
from typing import cast


def timer(func):
    # @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        mem_after = process.memory_info().rss
        mem_used = float(mem_after - mem_before) / 1024 / 1024 / 1024
        print(f"{Fore.BLUE}Elapsed time for {func.__name__}: {elapsed_time:0.2f} seconds, memory {mem_used:0.2f} G{Fore.RESET}")
        return value

    return wrapper_timer


@timer
def fetch_census_geographies(school_df):
    start_t = time.time()
    # fetch census geographies corresponding to addresses - unfortunately, about 15% of address don't get a match
    school_df.rename(columns={"NCES ID": "id", "Address": "street", "City": "city", "State": "state", "Zip": "zip"}, inplace=True)
    school_df.to_csv("school_df.csv", index=False)
    addresses = school_df[["id", "street", "city", "state", "zip"]].to_dict("records")

    print("Fetching census geographies...")
    cg2010 = cg.CensusGeocode(benchmark="Public_AR_Current", vintage="Census2010_Current")
    num_schools = len(school_df.index)
    batch_size = 2000
    dfs = []
    for batch in np.arange(0, num_schools, step=batch_size):
        print("Fetching from", batch, "out of", num_schools, end=": ", flush=True)
        t = time.time()
        df = pd.DataFrame(cg2010.addressbatch(addresses[batch : batch + batch_size], returntype="geographies"))
        df.to_csv("batch." + str(batch) + ".csv", index=False)
        dfs.append(df)
        print(len(dfs[-1].index), "records in %.3f s" % (time.time() - t), flush=True)

    geographies = pd.concat(dfs, ignore_index=True)
    num_addresses = len(geographies.index)
    not_found = geographies[(geographies.match == False)]
    not_found.to_csv("unmatched_address_schools.csv", index=False)
    geographies = geographies[(geographies.match == True)]
    print("Found", len(geographies.index), "address matches out of", num_addresses)
    geoids_df = pd.DataFrame()
    geoids_df["id"] = geographies.id
    # only need down to the census tract
    geoids_df["GEOID"] = (geographies.statefp + geographies.countyfp + geographies.tract).astype("int64")
    schools_geoids_df = school_df[["id", "Enrollment", "Start Grade", "End Grade", "Full Time Teachers"]].merge(
        geoids_df, on="id"
    )
    schools_geoids_df.to_csv("schools_geoids.csv", index=False)

    print("Processed", len(schools_geoids_df.index), "records in %.3f s" % (time.time() - start_t))


@timer
def get_census_bgs(args):
    print("Reading Census bg files")
    t = time.time()
    census_bgs_df = pd.DataFrame()
    for fname in args.census_bg_files:
        census_bgs = gpd.read_file(fname)
        census_bgs_df = pd.concat([census_bgs_df, census_bgs])
    print("Read", len(census_bgs_df.index), "census bgs in %.3f s" % (time.time() - t))
    # census_bgs_df.to_csv("census_bgs.csv", index=False)
    return census_bgs_df


PREK_SCHOOL_AGES = [0, 4]
ELEM_SCHOOL_AGES = [5, 10]
MID_SCHOOL_AGES = [11, 13]
HIGH_SCHOOL_AGES = [14, 18]
LEVEL_KEYS = {0: "P", 1: "E", 2: "M", 3: "H"}


def get_level_from_age(start_age, end_age):
    levels = ""
    try:
        start_age = int(start_age)
        end_age = int(end_age)
    except:
        # if the age ranges are messed up, just use full range
        start_age = ELEM_SCHOOL_AGES[0]
        end_age = HIGH_SCHOOL_AGES[1]
    for i, (low_age, high_age) in enumerate([PREK_SCHOOL_AGES, ELEM_SCHOOL_AGES, MID_SCHOOL_AGES, HIGH_SCHOOL_AGES]):
        if start_age >= low_age and start_age <= high_age:
            levels += LEVEL_KEYS[i]
        elif end_age >= low_age and end_age <= high_age:
            levels += LEVEL_KEYS[i]
        elif start_age < low_age and end_age > high_age:
            levels += LEVEL_KEYS[i]
    # for those rare cases when no start and end grades are present, just assume the schools handles all levels
    if levels == "":
        levels = "PEMH"
    return levels


# How a school's enrollment is shared among the levels it teaches, one students_<level> column
# each, so upop_to_exaepi.py can fill each level to its own capacity. Splitting evenly by the
# number of levels instead gives K-5's six grades the same places as 6-8's three and hands a
# PK-8 school a third of its enrollment as preschool places -- on CA that left elementary with
# ~576k fewer places than students, middle school ~507k more, and dumped the shortfall into a
# few schools -- 1,476 at over 5x their listed enrollment.
LEVEL_CAPACITY_COLS = {level: f"students_{level}" for level in LEVEL_KEYS.values()}
# Ages each level's grades span, for sharing out a total that has no per-grade breakdown: the
# level tags' own ranges, except preschool, which counts only ages 3-4 -- the ages UrbanPop
# enrolls in preschool -- so a school taking infants from age 1 doesn't get most of its
# enrollment as preschool places. High school stops at 17, the age of 12th grade.
LEVEL_WEIGHT_AGES = {"P": (3, 4), "E": tuple(ELEM_SCHOOL_AGES), "M": tuple(MID_SCHOOL_AGES), "H": (14, 17)}
# NCES per-grade enrollment columns making up each level (G13 is the rare 13th grade)
NCES_LEVEL_GRADE_COLS = {
    "P": ["PK"],
    "E": ["KG", "G01", "G02", "G03", "G04", "G05"],
    "M": ["G06", "G07", "G08"],
    "H": ["G09", "G10", "G11", "G12", "G13"],
}


def apportion(totals, weights):
    """Split each totals[i] into integer parts proportional to weights[i] (largest remainder), so
    each row's parts sum to exactly its total. Rows whose weights sum to zero get all zeros."""
    weights = np.asarray(weights, dtype=float)
    totals = np.asarray(totals, dtype=np.int64)
    row_sums = weights.sum(axis=1, keepdims=True)
    shares = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0) * totals[:, None]
    parts = np.floor(shares).astype(np.int64)
    short = totals - parts.sum(axis=1)
    short[row_sums[:, 0] == 0] = 0
    # hand the remaining units to the largest fractional remainders, one each
    order = np.argsort(-(shares - parts), axis=1)
    for i in np.flatnonzero(short > 0):
        parts[i, order[i, : short[i]]] += 1
    return parts


def add_level_capacities(df, level_weights=None, start_age_col=None, end_age_col=None):
    """Add a students_<level> column per level (see LEVEL_CAPACITY_COLS), sharing out each school's
    `students` among the levels in its `level` tag.

    level_weights, if given, is an (n, 4) array of per-level enrollment (P, E, M, H) -- NCES's
    per-grade counts. Otherwise, or for a row where those are all zero, the weights are how many
    of the level's ages (LEVEL_WEIGHT_AGES) the school's [start_age_col, end_age_col] range
    covers; failing that, the levels in the tag share equally. Levels outside the tag always get
    nothing, since upop_to_exaepi.py only offers a school to the levels it is tagged with.
    """
    levels = list(LEVEL_CAPACITY_COLS)
    in_tag = np.array([[lv in tag for lv in levels] for tag in df["level"].astype(str)], dtype=float)
    weights = np.zeros((len(df), len(levels)))
    if level_weights is not None:
        weights = np.clip(np.nan_to_num(np.asarray(level_weights, dtype=float)), 0, None) * in_tag
    if start_age_col is not None:
        start = pd.to_numeric(df[start_age_col], errors="coerce").to_numpy(dtype=float)
        end = pd.to_numeric(df[end_age_col], errors="coerce").to_numpy(dtype=float)
        span = np.stack(
            [np.minimum(end, hi) - np.maximum(start, lo) + 1 for lo, hi in LEVEL_WEIGHT_AGES.values()], axis=1
        )
        span = np.clip(np.nan_to_num(span), 0, None) * in_tag
        missing = weights.sum(axis=1) == 0
        weights[missing] = span[missing]
    missing = weights.sum(axis=1) == 0
    weights[missing] = in_tag[missing]
    parts = apportion(df["students"].fillna(0).clip(lower=0).to_numpy(), weights)
    for j, level in enumerate(levels):
        df[LEVEL_CAPACITY_COLS[level]] = parts[:, j]
    return df


def get_age_from_grade(grade_str):
    try:
        return int(grade_str) + 5
    except:
        if grade_str == "KG":
            return 5
        if grade_str == "PK":
            return 4
    return -1


# Best-effort name match for schools/colleges that don't correspond to a single physical
# location (statewide virtual charter schools, online-only university branches, etc). These
# aren't data errors -- their enrollment figures are real -- but treating them as one physical
# in-person mixing group would manufacture a huge fake contact hub. There's no HIFLD field that
# flags this directly (NCES has an authoritative "VIRTUAL" column instead, used separately in
# get_nces_public_schools), so this is necessarily incomplete: it catches the most obviously
# named cases but misses e.g. large online-only universities with no "online"/"virtual" in
# their name (Western Governors University, Southern New Hampshire University).
VIRTUAL_NAME_PATTERN = re.compile(
    r"VIRTUAL|CYBER|CONNECTIONS ACADEMY|\bONLINE\b|DIGITAL ACAD|E-SCHOOL|DISTANCE (?:ED|LEARN)|HOME LEARNING|\bK12\b|\bSTRIDE\b",
    re.IGNORECASE,
)


def drop_virtual_schools(df, name_col):
    is_virtual = df[name_col].str.contains(VIRTUAL_NAME_PATTERN, na=False)
    n_virtual = int(is_virtual.sum())
    if n_virtual:
        print(f"Dropping {n_virtual} likely virtual/online schools by name match")
    return df[~is_virtual].drop(columns=[name_col])


def invalidate_bad_teacher_ratios(schools_with_geoids, min_ratio=1, max_ratio=100):
    """Treat an implausible students/teachers ratio as a missing teacher count.

    Real schools run roughly 4-36 students per teacher; some HIFLD records report a raw
    teacher count wildly outside that (e.g. "1 full time teacher" for a school of nearly
    2000 students), which get_complete()'s dropna/to_keep filters would otherwise accept
    as-is since both fields are individually positive. Zeroing the teacher count here
    routes these rows into get_complete()'s existing average-ratio backfill instead.
    """
    valid = (schools_with_geoids.teachers > 0) & (schools_with_geoids.students > 0)
    ratio = schools_with_geoids.students / schools_with_geoids.teachers.replace(0, np.nan)
    bad_ratio = valid & ((ratio < min_ratio) | (ratio > max_ratio))
    schools_with_geoids.loc[bad_ratio, "teachers"] = 0
    return schools_with_geoids


def get_complete(schools_with_geoids):
    num_schools = len(schools_with_geoids)
    # we could have schools without geoids - missing lng/lat?
    schools_with_geoids.dropna(inplace=True)
    schools_complete = schools_with_geoids[(schools_with_geoids.teachers > 0) & (schools_with_geoids.students > 0)][
        ["teachers", "students"]
    ]
    sum_students = schools_complete.students.sum()
    sum_teachers = schools_complete.teachers.sum()
    avg_school_size = sum_students / len(schools_complete)
    avg_teacher_ratio = sum_students / sum_teachers
    print("Found", sum_students, "students and", sum_teachers, "teachers in", len(schools_complete), "schools")
    print("Avg school size %.0f and avg student/teacher ratio %.2f" % (avg_school_size, avg_teacher_ratio))
    # Only keep the schools with complete records, or with only student counts, if those are above a certain level
    to_fix = (schools_with_geoids.students >= 10) & (schools_with_geoids.teachers <= 0)
    schools_with_geoids.loc[to_fix, "teachers"] = np.int32(np.ceil(schools_with_geoids[to_fix].students / avg_teacher_ratio))
    to_keep = (schools_with_geoids.teachers > 0) & (schools_with_geoids.students > 0)
    schools_with_geoids = schools_with_geoids[to_keep]
    print("Dropped", num_schools - len(schools_with_geoids), "incomplete records")
    return schools_with_geoids


@timer
def get_hifld_public_schools(args, census_bgs_df):
    school_df = pd.DataFrame()
    cols_to_read = ["NCES ID", "Name", "Latitude", "Longitude", "Enrollment", "Start Grade", "End Grade", "Full Time Teachers"]
    for fname in args.public_school_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[cols_to_read]
        df["Start Grade"] = list(map(get_age_from_grade, df["Start Grade"]))
        df["End Grade"] = list(map(get_age_from_grade, df["End Grade"]))
        df["level"] = list(map(get_level_from_age, df["Start Grade"], df["End Grade"]))
        school_df = pd.concat([school_df, df], ignore_index=True)
        print(len(df.index), "records in % .3f s" % (time.time() - t))

    geometry = [shapely.geometry.Point(xy) for xy in zip(school_df.Longitude, school_df.Latitude)]
    school_gdf = gpd.GeoDataFrame(school_df, crs="EPSG:4269", geometry=geometry)
    schools_with_geoids = pd.DataFrame(gpd.sjoin(school_gdf, census_bgs_df, how="left", predicate="within"))
    schools_with_geoids = schools_with_geoids.rename(
        columns={
            "NCES ID": "id",
            "Enrollment": "students",
            "Full Time Teachers": "teachers",
            "GEOID10": "geoid",
        }
    )[["id", "students", "teachers", "level", "geoid", "Name", "Start Grade", "End Grade"]]
    # HIFLD has only a total, so it is shared out by grade span (the grades are ages by now)
    schools_with_geoids = add_level_capacities(schools_with_geoids, start_age_col="Start Grade", end_age_col="End Grade")
    schools_with_geoids = schools_with_geoids.drop(columns=["Start Grade", "End Grade"])
    schools_with_geoids = drop_virtual_schools(schools_with_geoids, "Name")
    schools_with_geoids = invalidate_bad_teacher_ratios(schools_with_geoids)
    schools_with_geoids = get_complete(schools_with_geoids)
    schools_with_geoids.to_csv("non_college_schools_with_geoids.csv", index=False)
    print("Wrote", len(schools_with_geoids), "schools to non_college_schools_with_geoids.csv")
    return schools_with_geoids


@timer
def get_hifld_private_schools(args, census_bgs_df):
    school_df = pd.DataFrame()
    cols_to_read = ["NCES ID", "Name", "Latitude", "Longitude", "Enrollment", "Start Grade", "End Grade", "Full Time Teachers"]
    for fname in args.private_school_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[cols_to_read]
        # the grades are actually ages for these private schools
        df["level"] = list(map(get_level_from_age, df["Start Grade"], df["End Grade"]))
        school_df = pd.concat([school_df, df], ignore_index=True)
        print(len(df.index), "records in % .3f s" % (time.time() - t))

    geometry = [shapely.geometry.Point(xy) for xy in zip(school_df.Longitude, school_df.Latitude)]
    school_gdf = gpd.GeoDataFrame(school_df, crs="EPSG:4269", geometry=geometry)
    schools_with_geoids = pd.DataFrame(gpd.sjoin(school_gdf, census_bgs_df, how="left", predicate="within"))
    schools_with_geoids = schools_with_geoids.rename(
        columns={
            "NCES ID": "id",
            "Enrollment": "students",
            "Full Time Teachers": "teachers",
            "GEOID10": "geoid",
        }
    )[["id", "students", "teachers", "level", "geoid", "Name", "Start Grade", "End Grade"]]
    # only a total here too, so shared out by grade span, as for HIFLD public schools
    schools_with_geoids = add_level_capacities(schools_with_geoids, start_age_col="Start Grade", end_age_col="End Grade")
    schools_with_geoids = schools_with_geoids.drop(columns=["Start Grade", "End Grade"])
    schools_with_geoids = drop_virtual_schools(schools_with_geoids, "Name")
    schools_with_geoids = invalidate_bad_teacher_ratios(schools_with_geoids)
    schools_with_geoids = get_complete(schools_with_geoids)
    schools_with_geoids.to_csv("non_college_schools_with_geoids.csv", index=False)
    print("Wrote", len(schools_with_geoids), "schools to non_college_schools_with_geoids.csv")
    return schools_with_geoids


@timer
def get_hifld_childcare(args, census_bgs_df):
    childcare_df = pd.DataFrame()
    for fname in args.childcare_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[["ID", "LATITUDE", "LONGITUDE", "POPULATION"]]
        childcare_df = pd.concat([childcare_df, df], ignore_index=True)
        print(len(df.index), "records in % .3f s" % (time.time() - t))

    geometry = [shapely.geometry.Point(xy) for xy in zip(childcare_df.LONGITUDE, childcare_df.LATITUDE)]
    childcare_gdf = gpd.GeoDataFrame(childcare_df, crs="EPSG:4269", geometry=geometry)
    childcare_with_geoids = pd.DataFrame(gpd.sjoin(childcare_gdf, census_bgs_df, how="left", predicate="within"))
    childcare_with_geoids = childcare_with_geoids.rename(
        columns={
            "ID": "id",
            "POPULATION": "students",
            "GEOID10": "geoid",
        }
    )[["id", "students", "geoid"]]
    num_childcare = len(childcare_with_geoids)
    # we could have childcare without geoids - missing lng/lat?
    childcare_with_geoids.dropna(inplace=True)
    # only keep childcare with complete records
    childcare_complete = childcare_with_geoids[childcare_with_geoids.students > 0]
    print("Avg childcare size", childcare_complete.students.mean())
    # HIFLD marks unknown population as -999 (or occasionally 0). Rather than filling every
    # such record with the single national-average value -- which, since -999 covers 100% of
    # some states' records, would make every childcare center in that state identical -- sample
    # each one from the empirical distribution of known sizes, so the average is preserved but
    # individual centers still vary realistically.
    missing = childcare_with_geoids.students <= 0
    num_missing = int(missing.sum())
    if num_missing > 0:
        childcare_with_geoids.loc[missing, "students"] = np.random.choice(
            childcare_complete.students.to_numpy(), size=num_missing, replace=True
        )
    # childcare_with_geoids = childcare_with_geoids[childcare_with_geoids.students > 0]
    # assume 5 children per adult
    childcare_with_geoids.insert(cast(int, childcare_with_geoids.columns.get_loc("students")) + 1, "teachers", int(0))
    childcare_with_geoids.insert(cast(int, childcare_with_geoids.columns.get_loc("teachers")) + 1, "level", "C")
    childcare_with_geoids["teachers"] = np.int32(np.ceil(childcare_with_geoids.students / 7))
    sum_children = childcare_with_geoids.students.sum()
    childcare_with_geoids.to_csv("childcare_with_geoids.csv", index=False)
    print("Wrote", len(childcare_with_geoids), "childcare records to childcare_with_geoids.csv")
    print("Total children:", sum_children)
    print("Dropped", num_childcare - len(childcare_with_geoids), "incomplete records")
    return childcare_with_geoids


@timer
def get_hifld_colleges(args):
    start_t = time.time()
    # we don't have lng/lat for colleges so we have to fetch with addresses
    colleges_df = pd.DataFrame()
    for fname in args.college_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[["UNIQUEID", "NAME", "ADDRESS", "CITY", "STATE", "ZIP", "TOT_ENROLL", "TOT_EMP"]]
        colleges_df = pd.concat([colleges_df, df], ignore_index=True)
        print(len(df.index), "records in % .3f s" % (time.time() - t))

    colleges_df.rename(
        columns={
            "UNIQUEID": "id",
            "ADDRESS": "street",
            "CITY": "city",
            "STATE": "state",
            "ZIP": "zip",
            "TOT_ENROLL": "students",
            "TOT_EMP": "teachers",
        },
        inplace=True,
    )
    colleges_df.to_csv("colleges_df.csv", index=False)
    addresses = colleges_df[["id", "street", "city", "state", "zip"]].to_dict("records")
    print("Fetching census geographies for college addresses...")
    cg2010 = cg.CensusGeocode(benchmark="Public_AR_Current", vintage="Census2010_Current")
    num_colleges = len(colleges_df)
    batch_size = 1000
    geo_df = pd.DataFrame()
    for batch in np.arange(0, num_colleges, step=batch_size):
        t = time.time()
        print("Fetching from", batch, "out of", num_colleges, end=": ", flush=True)
        batch_fname = "batch." + str(batch) + ".csv"
        try:
            # check to see if batch already exists
            df = pd.read_csv(batch_fname, dtype={"statefp": str, "countyfp": str, "tract": str, "block": str})
        except FileNotFoundError:
            df = pd.DataFrame(cg2010.addressbatch(addresses[batch : batch + batch_size], returntype="geographies"))
            # backup for resuming
            df.to_csv(batch_fname, index=False)
        geo_df = pd.concat([geo_df, df], ignore_index=True)
        print(len(df), "records in %.3f s" % (time.time() - t), flush=True)

    num_addresses = len(geo_df.index)
    not_found = geo_df[(geo_df.match == False)]
    not_found.to_csv("unmatched_address_colleges.csv", index=False)
    geo_df = geo_df[(geo_df.match == True)]
    print("Found", len(geo_df), "address matches out of", num_addresses)
    geoids_df = pd.DataFrame()
    geoids_df["id"] = geo_df.id.astype("int64")
    # only need down to the census tract
    geoids_df["geoid"] = (geo_df.statefp + geo_df.countyfp + geo_df.tract + geo_df.block).str[:12]
    # add a college level indicator to fit with schools data
    colleges_df["level"] = "U"
    colleges_with_geoids = colleges_df[["id", "students", "teachers", "level", "NAME"]].merge(geoids_df, on="id")
    colleges_with_geoids = drop_virtual_schools(colleges_with_geoids, "NAME")
    # unlike K-12 "Full Time Teachers", a college/university's TOT_EMP legitimately includes
    # non-teaching staff (hospitals and medical centers for research universities, etc.), so a
    # students/employees ratio far outside the K-12 range isn't necessarily an error here -- e.g.
    # Ohio State's ~35,000 TOT_EMP includes its medical center, and University of the People's
    # ~1:1682 ratio reflects its real, almost entirely volunteer-faculty model. Skip the ratio
    # sanity check that applies to get_hifld_public_schools/get_hifld_private_schools.
    colleges_with_geoids = get_complete(colleges_with_geoids)
    colleges_with_geoids.to_csv("colleges_with_geoids.csv", index=False)
    print("Wrote", len(colleges_with_geoids), "colleges to colleges_with_geoids.csv")
    return colleges_with_geoids


@timer
def get_nces_public_schools(args, census_bgs_df):
    print(f"Reading from {args.public_nces_school_file}: ", end="")
    t = time.time()
    grade_cols = [col for cols in NCES_LEVEL_GRADE_COLS.values() for col in cols]
    df = pd.read_csv(args.public_nces_school_file)[
        ["NCESSCH", "TOTAL", "STUTERATIO", "LATCOD", "LONCOD", "GSLO", "GSHI", "VIRTUAL"] + grade_cols
    ]
    print(len(df.index), "records in % .3f s" % (time.time() - t))
    # exclude statewide virtual/cyber schools -- their enrollment is real, but they don't
    # correspond to a single physical location, so treating them as one in-person mixing group
    # would manufacture a huge fake contact hub. NCES's VIRTUAL field is authoritative here,
    # unlike the HIFLD sources (see VIRTUAL_NAME_PATTERN/drop_virtual_schools), which have no
    # such flag and have to fall back to a name match.
    is_virtual = df["VIRTUAL"].isin(["Full Virtual", "Virtual with face to face options"])
    print(f"Dropping {int(is_virtual.sum())} virtual schools (NCES VIRTUAL field)")
    df = df[~is_virtual].drop(columns=["VIRTUAL"])
    # drop schools without grade information or for adults - N = not available, UG = ungraded, AE = adult education, M = missing
    no_grades = ["N ", "UG", "AE", "M "]
    for no_grade in no_grades:
        df = df.drop(df[df["GSLO"] == no_grade].index)

    geometry = [shapely.geometry.Point(xy) for xy in zip(df.LONCOD, df.LATCOD)]
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4269", geometry=geometry)
    geoids_df = pd.DataFrame(gpd.sjoin(gdf, census_bgs_df, how="left", predicate="within"))
    geoids_df["teachers"] = np.ceil(geoids_df.TOTAL / geoids_df.STUTERATIO)
    # avoid infinities, and treat implausible reported ratios (real schools run roughly 4-100
    # students per teacher) the same as a missing ratio -- some NCES records have STUTERATIO
    # far outside any plausible range (e.g. 7950), which would otherwise produce a nonsense
    # teacher count. get_complete() below already backfills teachers <= 0 from the dataset's
    # average ratio, so routing bad ratios through that same path recovers a sane value.
    bad_ratio = (geoids_df["STUTERATIO"] < 1) | (geoids_df["STUTERATIO"] > 100)
    geoids_df.loc[bad_ratio, "teachers"] = 0
    # only fill the columns about to be cast to int below -- filling the whole frame would also
    # touch the string columns brought in by the spatial join (e.g. GEOID10), which (a) errors
    # under pandas' Arrow-backed string dtype and (b) would wrongly turn an unmatched geoid into
    # "0" instead of leaving it NaN to be dropped by get_complete()'s dropna.
    fill_cols = ["TOTAL", "teachers", "GSLO", "GSHI"]
    geoids_df[fill_cols] = geoids_df[fill_cols].fillna(0)

    grade_descr_to_num = {"PK": "-1", "KG": "0"}
    for grade_descr, grade_num in grade_descr_to_num.items():
        geoids_df.loc[geoids_df["GSLO"] == grade_descr, "GSLO"] = grade_num
        geoids_df.loc[geoids_df["GSHI"] == grade_descr, "GSHI"] = grade_num
    geoids_df = geoids_df.astype({"TOTAL": "int", "teachers": "int", "GSLO": "int", "GSHI": "int"})

    geoids_df["GSLO"] = list(map(get_age_from_grade, geoids_df["GSLO"]))
    geoids_df["GSHI"] = list(map(get_age_from_grade, geoids_df["GSHI"]))
    geoids_df["level"] = list(map(get_level_from_age, geoids_df["GSLO"], geoids_df["GSHI"]))
    # NCES counts every grade, so each level's places come from its own grades' enrollment (NCES
    # codes a missing count as negative). Ungraded students have no level and are shared out with
    # the rest. Some states report no PK counts at all -- CA among them, whose TOTAL is exactly its
    # K-12 grades -- so a PK-tagged school there gets no preschool places.
    level_weights = np.stack(
        [
            geoids_df[cols].apply(pd.to_numeric, errors="coerce").clip(lower=0).fillna(0).sum(axis=1)
            for cols in NCES_LEVEL_GRADE_COLS.values()
        ],
        axis=1,
    )

    geoids_df = geoids_df.rename(
        columns={
            "NCESSCH": "id",
            "TOTAL": "students",
            "GEOID10": "geoid",
        }
    )
    geoids_df = add_level_capacities(geoids_df, level_weights=level_weights, start_age_col="GSLO", end_age_col="GSHI")
    geoids_df = geoids_df[["id", "students", "teachers", "level", "geoid"] + list(LEVEL_CAPACITY_COLS.values())]
    geoids_df = get_complete(geoids_df)
    geoids_df.to_csv("non_college_schools_with_geoids.csv", index=False)
    print("Wrote", len(geoids_df), "schools to non_college_schools_with_geoids.csv")
    geoids_df.to_csv("schools_geoids_nces.csv", index=False)
    return geoids_df


@timer
def main():
    cfg_parser = argparse.ArgumentParser(
        description="Generate school list with Census Block Group GEOID, using Census bg shapefiles and HIFLD data",
        add_help=False,
    )
    cfg_parser.add_argument("-c", "--config", help="Config file", metavar="FILE")
    args, remaining_argv = cfg_parser.parse_known_args()
    main_args: dict[str, str | list[str]] = {
        "private_school_files": "",
        "public_school_files": "",
        "public_nces_school_files": "",
        "college_files": "",
        "childcare_files": "",
        "census_bg_files": "",
    }
    glob_keys = set(main_args.keys())
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read([args.config])
        main_args.update(dict(cfg.items("main")))
        # only the file-list options above get glob-expanded -- other config keys (rseed,
        # public_nces_school_file) are scalars and must pass through untouched, or argparse's
        # own type conversion/defaulting for them breaks (e.g. rseed=29 silently becoming None)
        for key in glob_keys:
            files = str(main_args[key]).split()
            file_list = []
            for f in files:
                file_list.extend(glob.glob(f))
            main_args[key] = file_list
    parser = argparse.ArgumentParser(parents=[cfg_parser])
    parser.set_defaults(**main_args)
    parser.add_argument("--private_school_files", "-p", nargs="+", help="HIFLD Private school CSV files")
    parser.add_argument("--public_school_files", "-s", nargs="+", help="HIFLD Public school CSV files")
    # NCES data is only available for public schools, but is recommended over HIFLD's public school
    # data: its 2018-19 vintage matches the ~2019 vintage already used elsewhere in this pipeline
    # (see the LODES note in README.md), and its authoritative VIRTUAL field lets
    # get_nces_public_schools() reliably drop statewide virtual/cyber schools, which HIFLD has no
    # equivalent flag for (see VIRTUAL_NAME_PATTERN/drop_virtual_schools). See
    # data/EducationData/README.md for the full comparison.
    parser.add_argument("--public_nces_school_file", help="NCES Public school CSV file - recommended over HIFLD")
    parser.add_argument("--college_files", "-u", nargs="+", help="HIFLD College/University CSV files")
    parser.add_argument("--childcare_files", "-a", nargs="+", help="HIFLD Childcare CSV files")
    parser.add_argument("--census_bg_files", "-b", nargs="+", help="Census Block Group (bg) shape files")
    parser.add_argument(
        # no default= here: an explicit default would clobber the value set_defaults(**main_args)
        # already applied above from the config file's rseed= entry, since set_defaults() runs
        # before this add_argument() call.
        "--rseed", type=int, help="Random seed, for reproducible childcare size estimates"
    )
    args = parser.parse_args(remaining_argv)
    print(Fore.CYAN, "Options:", sep="")
    for arg, value in args.__dict__.items():
        print(f"  {arg:20s} {value}")
    print(Fore.RESET, end="")

    np.random.seed(args.rseed)
    start_t = time.time()
    census_bgs_df = get_census_bgs(args)
    childcare_geoids_df = get_hifld_childcare(args, census_bgs_df)
    colleges_geoids_df = get_hifld_colleges(args)
    private_schools_geoids_df = get_hifld_private_schools(args, census_bgs_df)
    if args.public_nces_school_file:
        public_schools_geoids_df = get_nces_public_schools(args, census_bgs_df)
    else:
        public_schools_geoids_df = get_hifld_public_schools(args, census_bgs_df)
    schools_geoids_df = pd.concat(
        [public_schools_geoids_df, private_schools_geoids_df, colleges_geoids_df, childcare_geoids_df], ignore_index=True
    )
    # colleges and childcare have a single level, so all their places are in `students`
    capacity_cols = list(LEVEL_CAPACITY_COLS.values())
    schools_geoids_df[capacity_cols] = schools_geoids_df[capacity_cols].fillna(0).astype(int)
    schools_geoids_df.to_csv("schools_with_geoids.csv", index=False)
    print("Wrote", len(schools_geoids_df), "schools to schools_with_geoids.csv")

    print("Finished in %.3f s" % (time.time() - start_t))


if __name__ == "__main__":
    main()
