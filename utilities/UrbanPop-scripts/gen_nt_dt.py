#!/usr/bin/env python

# This code is for generating nighttime/daytime flows that are missing from UrbanPop

import functools
import time
import pandas as pd
import pickle
import os
import sys
import argparse
import numpy as np
import random
from pandas.api.types import CategoricalDtype
from colorama import Fore


grade_categs = CategoricalDtype(
    categories=[
        "childcare",
        "preschl",
        "kind",
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "undergrad",
        "grad",
    ]
)

naics_to_description = {
    "11": "agr_ffh",  # Agriculture, forestry, fishing and hunting
    "21": "ext",  # Mining, quarrying, and oil and gas extraction
    "22": "utl",  # Utilities
    "23": "con",  # Construction
    "31": "mfg",  # Manufacturing
    "32": "mfg",  # Manufacturing
    "33": "mfg",  # Manufacturing
    "3M": "mfg",  # Manufacturing, not specified
    "42": "whl",  # Wholesale trade
    "44": "ret",  # Retail trade
    "45": "ret",  # Retail trade
    "4M": "ret",  # Retail trade
    "48": "trn_whs",  # Transportation and warehousing
    "49": "trn_whs",  # Transportation and warehousing
    "51": "inf",  # Information
    "52": "fin_ins",  # Finance and insurance
    "53": "rrl",  # Real estate rental and leasing
    "54": "prf",  # Professional, scientific and technical services
    "55": "mgt",  # Management of companies and enterprises
    "56": "adm_wmr",  # Administrative and support and waste management and remediating services
    "61": "edu",  # Educational services
    "62": "med_sca",  # Health care and social services
    "71": "ent",  # Arts, entertainment and recreation
    "72": "afs",  # Accomodation and food services
    "81": "srv",  # Other services (except public administration)
    "92": "pad",  # Public administration
    "10": "wfh",  # Work from home
}


region_scales = {12: "blockgroup", 11: "tract", 10: "county_subdiv", 7: "census_places", 5: "county", 2: "state"}

BLOCKGROUP_SCALE = 12
TRACT_SCALE = 11
COUNTY_SUBDIV_SCALE = 10
CENSUS_PLACES_SCALE = 7
COUNTY_SCALE = 5
STATE_SCALE = 2


def timer(func):
    # @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"{Fore.BLUE}Elapsed time for {func.__name__}: {elapsed_time:0.2f} seconds {Fore.RESET}")
        return value

    return wrapper_timer


def perc_str(n, m):
    return "%d out of %d (%.2f%%)" % (n, m, 100.0 * float(n) / m)


DUMP_INTERMEDIATES = False


@timer
def load_urbanpop_files(fnames):
    print(Fore.GREEN + "Loading UrbanPop files" + Fore.RESET)
    # each urbanpop file contains the following:
    # p_id,pums_id,h_id,geoid,hh_size,hh_type,hh_living_arrangement,hh_age,hh_has_kids,hh_income,hh_nb_wrks,hh_nb_non_wrks,
    # hh_nb_adult_wrks,hh_nb_adult_non_wrks,hh_dwg,hh_tenure,hh_vehicles,pr_age,pr_sex,pr_race,pr_hsplat,pr_ipr,pr_naics,
    # pr_emp_stat,pr_travel,pr_veh_occ,pr_commute,pr_grade
    upop_df = pd.DataFrame()
    start_t = time.time()
    for fname in fnames:
        # print("Reading data from", fname, end=": ")
        t = time.time()
        df_read = pd.read_feather(fname)[["p_id", "geoid", "pr_age", "pr_naics", "pr_emp_stat", "pr_grade"]]
        upop_df = pd.concat([upop_df, df_read], ignore_index=True)
        # print(len(df_read.index), "records in %.3f s" % (time.time() - t))

    upop_df.geoid = upop_df.geoid.astype(str)
    upop_df.sort_values(by=["p_id"], inplace=True)
    print("Processed", len(upop_df.index), "records in %.3f s" % (time.time() - start_t))
    return upop_df


@timer
def get_lodes_groups(lodes_fname):
    # Origin-Destination (OD) File Structure (LODES flows)
    # Pos Variable Type Explanation
    # 1 w_geocode Char15 Workplace Census Block Code
    # 2 h_geocode Char15 Residence Census Block Code
    # 3 S000 Num Total number of jobs
    # 4 SA01 Num Number of jobs of workers age 29 or younger
    # 5 SA02 Num Number of jobs for workers age 30 to 54
    # 6 SA03 Num Number of jobs for workers age 55 or older
    # 7 SE01 Num Number of jobs with earnings $1250/month or less
    # 8 SE02 Num Number of jobs with earnings $1251/month to $3333/month
    # 9 SE03 Num Number of jobs with earnings greater than $3333/month
    # 10 SI01 Num Number of jobs in Goods Producing industry sectors
    # 11 SI02 Num Number of jobs in Trade, Transportation, and Utilities industry sectors
    # 12 SI03 Num Number of jobs in All Other Services industry sectors
    # 13 createdate Char Date on which data was created, formatted as YYYYMMDD
    print("Loading", lodes_fname)
    lodes_df = pd.read_csv(lodes_fname, dtype={"w_geocode": str, "h_geocode": str})[["w_geocode", "h_geocode", "S000"]]
    print("Loaded", len(lodes_df), "entries")
    # truncate geoids to first 12, i.e. just census block groups
    lodes_df.h_geocode = lodes_df.h_geocode.str[:12]
    lodes_df.w_geocode = lodes_df.w_geocode.str[:12]
    # lodes_df = lodes_df.groupby(lodes_df.columns.tolist()).size().reset_index().rename(columns={0: "count"})
    lodes_df = lodes_df.groupby(["w_geocode", "h_geocode"]).S000.sum().reset_index().rename(columns={0: "count"})
    if DUMP_INTERMEDIATES:
        lodes_df.to_csv(lodes_fname + "-short.csv", index=False)
    return lodes_df


@timer
def alloc_workers(args, workers_df):
    print(Fore.GREEN + "Allocating workers" + Fore.RESET)
    # need to alloc files with the following columns:
    # p_id,role,orig_geoid,dest_geoid,lodes_segment,naics,grade,school_id
    # we can skip the lodes_segment, since we don't use it in later processing
    worker_groups = workers_df.groupby(["geoid"])
    lodes_df = get_lodes_groups(args.lodes_file)
    lodes_groups = lodes_df.groupby(["h_geocode"])
    nt_dt_df = pd.DataFrame()
    for name, worker_group in worker_groups:
        num_workers = len(worker_group)
        # print("Worker group for", name, "of size", num_workers)
        lodes_group = lodes_groups.get_group(name)
        if len(lodes_group) == 0:
            print("ERROR: Could not find GEOID", name, "in LODES data")
            sys.exit(1)
        sum_flows = lodes_group["S000"].sum()
        flow_probs = lodes_group["S000"] / sum_flows
        # print("Found lodes group of size", len(lodes_group), "with", sum_flows, "flows")
        rnd_sample = lodes_group.sample(n=num_workers, weights=flow_probs, replace=True)
        nt_dt_group = worker_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=rnd_sample["w_geocode"].tolist())
        nt_dt_df = pd.concat([nt_dt_df, nt_dt_group], ignore_index=True)

    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    nt_dt_df["role"] = "worker"
    # workers have no grade since that indicates that the agent is in school
    nt_dt_df["grade"] = ""
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade"]]
    nt_dt_df.naics = nt_dt_df.naics.str[:2].replace(naics_to_description)
    print("Added destinations for", len(nt_dt_df), "workers")
    if DUMP_INTERMEDIATES:
        nt_dt_df.to_csv("workers_nt_dt.csv", index=False)
    return nt_dt_df


@timer
def get_schools(fname):
    print("Loading schools from", fname)
    schools_df = pd.read_csv(fname, low_memory=False, dtype={"geoid": str})
    print("Loaded", len(schools_df), "entries:", schools_df.students.sum(), "students,", schools_df.teachers.sum(), "teachers")
    if DUMP_INTERMEDIATES:
        schools_df.to_csv("schools.csv", sep="\t", index=False)
    nm_schools = schools_df[schools_df.geoid.str.startswith("35")]
    if DUMP_INTERMEDIATES:
        nm_schools.to_csv("nm_schools.csv", sep="\t", index=False)
    print("NM schools:", len(nm_schools), "students", nm_schools.students.sum())
    return schools_df


# @timer
def alloc_students_region(students_df, schools_df, geoid_scaling, alloc_all):
    schools_df["region"] = schools_df.geoid.str[:geoid_scaling]
    # group by region
    # students_df = students_df[(students_df.allocated == False)]
    students_df.loc[:, "region"] = students_df.geoid.str[:geoid_scaling]
    student_groups = students_df.groupby(["region"])
    # print("      Found", len(student_groups), "regions for student groups")
    school_groups = schools_df.groupby(["region"])
    # print("      Found", len(schools_df), "regions for school groups")
    nt_dt_df = pd.DataFrame()
    missing_regions = 0
    for group_name, student_group in student_groups:
        num_students_reqd = len(student_group)
        if num_students_reqd == 0:
            continue
        # defaults to empty in case we can't find a key
        # schools_selected = pd.DataFrame({"geoid": [""] * num_students_reqd, "id": ["-1"] * num_students_reqd})
        try:
            school_group = school_groups.get_group(group_name)[["geoid", "id", "remaining_students"]]
        except KeyError:
            missing_regions += 1
            continue
        num_students_avail = school_group.remaining_students.sum()
        if alloc_all:
            num_students_avail = num_students_reqd
            norm_factor = school_group.remaining_students.sum()
        else:
            missing_students = max(num_students_reqd - num_students_avail, 0)
            if missing_students > 0:
                # add a dummy empty school for missing prob
                school_group.loc[len(school_group)] = ["", "-1", missing_students]
            norm_factor = num_students_avail + missing_students
        num_students = num_students_reqd
        school_probs = school_group.remaining_students / norm_factor
        schools_selected = school_group.sample(n=num_students, weights=school_probs, replace=True)
        schools_selected.index.rename("index", inplace=True)

        nt_dt_group = student_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=schools_selected.geoid.tolist(), school_id=schools_selected.id.tolist())
        nt_dt_df = pd.concat([nt_dt_df, nt_dt_group], ignore_index=True)

        continue

        # reduce the available students at schools count according to how many have been allocated
        schools_selected_groups = schools_selected.groupby(["index"])
        selected_school_indexes = list(schools_selected_groups.groups.keys())
        selected_school_counts = schools_selected_groups.id.count().tolist()
        schools_df.loc[schools_df.index.isin(selected_school_indexes), "remaining_students"] -= np.int32(selected_school_counts)
        # always leave at least 1 student so we can always allocate to these schools, even at a low probability
        schools_df.loc[schools_df.remaining_students < 1, "remaining_students"] = 1

    print("      Found", len(student_groups), "student regions,", missing_regions, "without schools")
    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    nt_dt_df["role"] = "student"
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]]
    nt_dt_df.naics = ""
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    # students_df.loc[:, "allocated"] = np.bool(nt_dt_df.school_id != "-1")
    # students_df.loc[:, "allocated"] = np.bool(nt_dt_df.school_id != np.nan)
    # only subset if this is not the last round
    if not alloc_all:
        nt_dt_df = nt_dt_df[(nt_dt_df.school_id != "-1")]
    return nt_dt_df, students_df


@timer
def alloc_students_level(schools_df, students_df, level):
    age_levels = {"C": [3, 3], "P": [4, 4], "E": [5, 10], "M": [11, 13], "H": [14, 17], "U": [18, 19]}
    start_age = age_levels[level][0]
    end_age = age_levels[level][1]
    students_df = students_df[(students_df.pr_grade >= start_age) & (students_df.pr_grade <= end_age)].copy()
    students_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    if DUMP_INTERMEDIATES:
        students_df[["p_id", "pr_grade"]].to_csv("students-" + level + ".csv", sep="\t", index=False)
    # students_df["allocated"] = False
    # Get all schools with the required level
    schools_df = schools_df[(schools_df.level.str.find(level) != -1)].copy()
    # Some schools have multiple levels and so students should be allocated proportionately. For simplicity, we
    # just allocate uniformly among the levels, e.g. a EMH school would have 1/3 students each in elem, middle and high
    # divide by the number of levels to get the proportion
    schools_df.loc[:, "students"] = np.int32(np.ceil(schools_df.students / schools_df.level.str.len()))
    schools_df["remaining_students"] = schools_df.students
    nt_dt_df = pd.DataFrame()
    # for childcare, we assume that it is all close to home
    scales = list(region_scales.keys()) if level != "C" else list(region_scales.keys())[:3]
    # pick schools for students from decreasing resolution; the goal is to allocate students as close to home as possible
    for scale in scales:
        # FIXME: need to reduce student counts after each allocation round so that random weighting is correct in the next round
        print("    Region", region_scales[scale])
        alloc_all = False
        # make sure to allocate all at the final region scale
        alloc_all = True if scale == scales[-1] else False
        students_nt_dt_df, students_df = alloc_students_region(students_df, schools_df, scale, alloc_all)
        # num_unalloc_students = len(students_df[(students_df.allocated == False)])
        num_unalloc_students = len(students_df[(students_df.school_id == "-1")])
        print("      Set destinations for", len(students_nt_dt_df), "students,", num_unalloc_students, "still unallocated")
        nt_dt_df = pd.concat([nt_dt_df, students_nt_dt_df], ignore_index=True)
        if num_unalloc_students == 0:
            break

    return nt_dt_df


@timer
def alloc_students(args, students_df):
    print(Fore.GREEN + "Allocating students" + Fore.RESET)
    schools_df = get_schools(args.schools_file)
    # convert grades to ages
    grade_categs_found = list(students_df["pr_grade"].unique())
    grade_categs_expected = list(grade_categs.categories)
    missing_grades = [x for x in grade_categs_found if x not in grade_categs_expected and x != "" and x is not None]
    if missing_grades:
        print("WARNING: Found missing categories for pr_grade:", missing_grades)
    students_df["pr_grade"] = students_df["pr_grade"].astype(grade_categs).cat.codes + 3
    students_df["school_id"] = "-1"

    # allocate students from each level
    students_nt_dt_df = pd.DataFrame()
    for level in ["C", "P", "E", "M", "H", "U"]:
        print("  Allocating students for level", level)
        df = alloc_students_level(schools_df, students_df, level=level)
        students_nt_dt_df = pd.concat([students_nt_dt_df, df], ignore_index=True)

    # set all unallocated students
    unalloc = students_nt_dt_df.school_id == "-1"
    students_nt_dt_df.loc[unalloc, "dest_geoid"] = students_nt_dt_df.orig_geoid
    students_nt_dt_df.loc[unalloc, "grade"] = None
    students_nt_dt_df.loc[unalloc, "role"] = "nope"
    students_nt_dt_df.loc[unalloc, "school_id"] = None

    if DUMP_INTERMEDIATES:
        students_nt_dt_df.to_csv("students_nt_dt.csv", sep="\t", index=False)

    return students_nt_dt_df


def get_unemp(unemp_df):
    print(Fore.GREEN + "Adding unemployed" + Fore.RESET)
    unemp_df = unemp_df[["p_id", "geoid", "pr_naics", "pr_grade"]].copy()
    unemp_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    unemp_df["role"] = "nope"
    # unemployed have no grade since that indicates the agent is in school
    unemp_df["grade"] = ""
    unemp_df["dest_geoid"] = unemp_df.orig_geoid
    return unemp_df


@timer
def set_childcare(upop_df):
    print(Fore.GREEN + "Setting childcare" + Fore.RESET)
    # UrbanPop doesn't have allocations to childcare outside of PK in schools, so we randomly assign agents under age 5 not in PK
    # to childcare. According to NCES, 32% of under 1 year, 47% of 1-2 yrs and 83 of 3-5 yrs are in center-based care
    PROBS = {0: 0.32, 1: 0.47, 2: 0.47, 3: 0.83, 4: 0.83}
    df = upop_df.copy()
    if DUMP_INTERMEDIATES:
        df.to_csv("upop.csv", sep="\t")
    for age in np.arange(0, 5):
        is_candidate = (df.pr_grade == "") & (df.pr_age == age)
        num_candidates = int(len(df[is_candidate]) * PROBS[age])
        to_set = df.sample(n=num_candidates, replace=False)
        print("  Age", age, "set", len(to_set), "out of", len(df[is_candidate]))
        df.loc[to_set.index, "pr_grade"] = "childcare"

    print("Set", len(df[(df.pr_grade.str.startswith("childcare"))]), "agents to childcare")
    if DUMP_INTERMEDIATES:
        df.to_csv("childcare_set.csv", sep="\t")
    return df


def alloc_teachers_region(teachers_df, schools_df, geoid_scaling):
    schools_df["region"] = schools_df.geoid.str[:geoid_scaling]
    schools_df.to_csv("schools.csv", sep="\t")
    teachers_df = teachers_df[(teachers_df.grade == "")].copy()
    teachers_df.loc[:, "region"] = teachers_df.orig_geoid.str[:geoid_scaling]
    teacher_groups = teachers_df.groupby(["region"])
    school_groups = schools_df.groupby(["region"])
    # print("    Found", len(teacher_groups), "teacher groups")
    missing_regions = 0
    tot_reqd_teachers = 0
    for group_name, school_group in school_groups:
        num_teachers_reqd = school_group.adj_teachers.sum()
        if num_teachers_reqd == 0:
            continue
        tot_reqd_teachers += num_teachers_reqd
        try:
            teacher_group = teacher_groups.get_group(group_name)
        except KeyError:
            missing_regions += 1
            continue
        # print("    Group", group_name, "num_teachers_reqd", num_teachers_reqd, "num_teachers_avail", teacher_group.p_id.count())
        num_teachers_reqd = min(num_teachers_reqd, teacher_group.p_id.count())
        teachers_selected = teacher_group.sample(n=num_teachers_reqd, replace=False)
        probs = school_group.adj_teachers / num_teachers_reqd
        schools_selected = school_group.sample(n=num_teachers_reqd, weights=probs, replace=True)
        schools_selected.index.rename("index", inplace=True)
        teachers_df.loc[teachers_selected.index, "school_id"] = list(schools_selected.id)
        # FIXME: need to convert the school level to an age for the grade to match with the students
        teachers_df.loc[teachers_selected.index, "grade"] = list(schools_selected.level)
        teachers_df.loc[teachers_selected.index, "dest_geoid"] = list(schools_selected.geoid)
        # reduce the reqd teachers count according to how many have been allocated
        schools_selected_groups = schools_selected.groupby(["index"])
        selected_school_indexes = list(schools_selected_groups.groups.keys())
        selected_school_counts = schools_selected_groups.id.count().tolist()
        schools_df.loc[schools_df.index.isin(selected_school_indexes), "adj_teachers"] -= np.int32(selected_school_counts)
        schools_df.loc[schools_df.adj_teachers < 0, "adj_teachers"] = 0
        schools_df.loc[schools_df.index.isin(selected_school_indexes), "alloc_teachers"] -= np.int32(selected_school_counts)
        # print("      Region:", group_name, ":", len(teachers_selected), "teachers,", len(schools_selected_groups), "schools")

    teachers_df.drop("region", axis=1, inplace=True)
    print("    Found", len(school_groups), "school regions,", missing_regions, "without teachers")
    print("    Allocated", len(teachers_df[(teachers_df.grade != "")]), "teachers out of", tot_reqd_teachers)
    return teachers_df


@timer
def alloc_teachers(args, workers_nt_dt_df, students_nt_dt_df):
    print(Fore.GREEN + "Allocating teachers" + Fore.RESET)
    # The number of students at each school will not exactly match the original schools data, so we use the actual number
    # of students we have allocated, and the original student/teacher ratio to set the desired teacher counts for each school
    schools_df = get_schools(args.schools_file)
    schools_df.to_csv("schools.csv", sep="\t")
    alloc_schools_df = (
        students_nt_dt_df.groupby(["school_id"])["p_id"]
        .count()
        .reset_index()
        .rename(columns={"p_id": "alloc_students", "school_id": "id"})
    )
    schools_df = schools_df.merge(alloc_schools_df, on="id")
    schools_df["ratio"] = schools_df.alloc_students / schools_df.students
    schools_df["adj_teachers"] = np.int32(np.ceil(schools_df.teachers * schools_df.ratio))
    schools_df.to_csv("selected_schools.csv", sep="\t", float_format="%.1f", index=False)
    schools_df["alloc_teachers"] = schools_df.adj_teachers

    # workers_nt_dt_df.to_csv("workers_nt_dt.csv", sep="\t")
    teachers_df = workers_nt_dt_df[(workers_nt_dt_df.naics == "edu")].copy()
    teachers_df["school_id"] = None
    if DUMP_INTERMEDIATES:
        teachers_df.to_csv("teachers.csv", sep="\t")
    num_reqd_teachers = schools_df.adj_teachers.sum()
    print("Found", len(teachers_df), "edu workers for", num_reqd_teachers, "required teachers")
    scales = list(region_scales.keys())
    # pick schools for teachers from decreasing resolution; the goal is to allocate teachers as close to home as possible
    for scale in scales:
        print("  Region", region_scales[scale])
        teachers_df = alloc_teachers_region(teachers_df, schools_df, scale)
        workers_nt_dt_df.loc[teachers_df.index, "school_id"] = teachers_df.school_id
        workers_nt_dt_df.loc[teachers_df.index, "dest_geoid"] = teachers_df.dest_geoid
        workers_nt_dt_df.loc[teachers_df.index, "grade"] = teachers_df.grade
    schools_df.to_csv("adjusted_schools.csv", sep="\t")

    return workers_nt_dt_df


def main():
    t = time.time()
    parser = argparse.ArgumentParser(description="Generate nighttime/daytime worker/student populations for UrbanPop from LODES")
    parser.add_argument("--urbanpop_files", "-f", required=True, nargs="+", help="UrbanPop feather files")
    parser.add_argument("--lodes_file", "-l", required=True, help="LODES7 origin-destination (OD) file in csv format")
    parser.add_argument("--output_file", "-o", required=True, help="Output file (will be written in feather format)")
    parser.add_argument("--schools_file", "-s", required=True, help="File containing schools data in CSV")
    parser.add_argument("--rseed", "-r", help="Random seed", default=29, type=int)
    args = parser.parse_args()

    np.random.seed(args.rseed)

    upop_df = load_urbanpop_files(args.urbanpop_files)
    # randomly allocate some young agents to childcare
    upop_df = set_childcare(upop_df)
    # now split into students, workers and unemployed
    in_school = upop_df.pr_grade != ""
    students_df = upop_df[in_school].copy()
    is_employed = (upop_df.pr_emp_stat == "employed") | (upop_df.pr_emp_stat == "mil")
    workers_df = upop_df[is_employed & ~in_school]
    unemp_df = upop_df[~is_employed & ~in_school]
    tot = len(workers_df) + len(students_df) + len(unemp_df)
    print("Counts:")
    print("  workers:   ", len(workers_df))
    print("  students:  ", len(students_df))
    print("  unemployed:", len(unemp_df))
    print("  total:     ", tot)
    if tot != len(upop_df):
        print("ERROR: total agents mismatch, allocated", tot, "but found", len(upop_df), "in UrbanPop feather files")

    # get students
    students_nt_dt_df = alloc_students(args, students_df)
    # now get workers
    workers_nt_dt_df = alloc_workers(args, workers_df)
    # allocate teachers
    workers_nt_dt_df = alloc_teachers(args, workers_nt_dt_df, students_nt_dt_df)
    # get non-workers
    unemp_df = get_unemp(unemp_df)

    nt_dt_df = pd.concat([workers_nt_dt_df, unemp_df, students_nt_dt_df], ignore_index=True)
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    nt_dt_df.to_csv(args.output_file + "_nt_dt.csv", index=False)
    print("Wrote nt/dt data to", args.output_file + "_nt_dt.csv")

    print("Completed in %.2f s" % (time.time() - t))


if __name__ == "__main__":
    main()
