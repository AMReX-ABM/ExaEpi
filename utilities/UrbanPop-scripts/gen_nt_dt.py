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
        print("Reading data from", fname, end=": ")
        t = time.time()
        df_read = pd.read_feather(fname)[["p_id", "geoid", "pr_age", "pr_naics", "pr_emp_stat", "pr_grade"]]
        upop_df = pd.concat([upop_df, df_read])
        print(len(df_read.index), "records in %.3f s" % (time.time() - t))

    upop_df.geoid = upop_df.geoid.astype("int64")
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
    lodes_df = pd.read_csv(lodes_fname)[["w_geocode", "h_geocode", "S000"]]
    print("Loaded", len(lodes_df), "entries")
    # truncate geoids to first 12, i.e. just census block groups
    lodes_df.h_geocode = np.floor(lodes_df.h_geocode / 1000)
    lodes_df.w_geocode = np.floor(lodes_df.w_geocode / 1000)
    lodes_df.h_geocode = lodes_df.h_geocode.astype("int64")
    lodes_df.w_geocode = lodes_df.w_geocode.astype("int64")
    # lodes_df = lodes_df.groupby(lodes_df.columns.tolist()).size().reset_index().rename(columns={0: "count"})
    lodes_df = lodes_df.groupby(["w_geocode", "h_geocode"]).S000.sum().reset_index().rename(columns={0: "count"})
    lodes_df.to_csv(lodes_fname + "-short.csv")
    return lodes_df


@timer
def alloc_nt_dt_workers(args, workers_df):
    print(Fore.GREEN + "Allocating workers" + Fore.RESET)
    # need to alloc files with the following columns:
    # p_id,role,orig_geoid,dest_geoid,lodes_segment,naics,grade,school_id
    # we can skip the lodes_segment, since we don't use it in later processing
    worker_groups = workers_df.groupby(["geoid"])
    lodes_df = get_lodes_groups(args.lodes_file)
    lodes_groups = lodes_df.groupby(["h_geocode"])
    rgen = np.random.default_rng(seed=29)
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
        dests = rgen.choice(a=lodes_group["w_geocode"], size=num_workers, p=flow_probs, replace=True)
        # print("Got", len(dests), "destinations")
        nt_dt_group = worker_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=dests)
        nt_dt_df = pd.concat([nt_dt_df, nt_dt_group])

    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    nt_dt_df["role"] = "worker"
    # workers have no grade since that indicates that the agent is in school
    nt_dt_df["grade"] = ""
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade"]]
    nt_dt_df.naics = nt_dt_df.naics.str[:2].replace(naics_to_description)
    print("Added destinations for", len(nt_dt_df), "workers")
    nt_dt_df.to_csv("workers_nt_dt.csv")
    return nt_dt_df


# @timer
def get_schools(fname):
    print("Loading", fname)
    schools_df = pd.read_csv(fname, low_memory=False)
    print("Loaded", len(schools_df), "entries:", schools_df.students.sum(), "students,", schools_df.teachers.sum(), "teachers")
    schools_df.geoid = schools_df.geoid.astype("str")
    schools_df.to_csv("schools.csv", sep="\t")
    nm_schools = schools_df[schools_df.geoid.str.startswith("35")]
    nm_schools.to_csv("nm_schools.csv", sep="\t")
    print("NM schools:", len(nm_schools), "students", nm_schools.students.sum())
    schools_df.geoid = schools_df.geoid.astype("int64")
    return schools_df


# @timer
def alloc_nt_dt_school_students_region(students_df, schools_df, geoid_scaling, rgen):
    schools_df["region"] = np.floor(schools_df.geoid / geoid_scaling)
    schools_df.region = schools_df.region.astype("int64")
    # group by region
    students_df = students_df[(students_df.allocated == False)]
    students_df.loc[:, "region"] = np.floor(students_df.geoid / geoid_scaling)
    student_groups = students_df.groupby(["region"])
    print("Found", len(student_groups), "regions for student groups")
    school_groups = schools_df.groupby(["region"])
    print("Found", len(schools_df), "regions for school groups")
    nt_dt_df = pd.DataFrame()
    missing_keys = 0
    for name, student_group in student_groups:
        num_students_reqd = len(student_group)
        # defaults to empty in case we can't find a key
        dests = pd.DataFrame({0: [""] * num_students_reqd, 1: ["-1"] * num_students_reqd})
        try:
            school_group = school_groups.get_group(name)
            school_group = school_group[["geoid", "id", "students"]]
            num_students_avail = school_group.students.sum()
            num_students = num_students_reqd
            missing_students = max(num_students_reqd - num_students_avail, 0)
            if missing_students > 0:
                # add a dummy empty school for missing prob
                school_group.loc[len(school_group)] = ["", "-1", missing_students]
            school_probs = school_group.students / (num_students_avail + missing_students)
            dests = pd.DataFrame(rgen.choice(a=school_group[["geoid", "id"]], size=num_students, p=school_probs, replace=True))
            # print("Group of", len(school_group), "schools,", num_students_reqd, "reqd,", num_students_avail, "available,", end="")
            # print(" ratio %.2f" % (num_students_reqd / num_students_avail))
        except KeyError:
            missing_keys += 1
        nt_dt_group = student_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=list(dests.iloc[:, 0]), school_id=list(dests.loc[:, 1]))
        nt_dt_df = pd.concat([nt_dt_df, nt_dt_group])

    print("Missing keys", missing_keys, "out of", len(student_groups))
    print("Unallocated students", len(nt_dt_df[(nt_dt_df.school_id == "-1")]))
    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    nt_dt_df["role"] = "student"
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]]
    nt_dt_df.naics = ""
    print("Added destinations for", len(nt_dt_df), "students")
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    students_df.loc[:, "allocated"] = np.bool(nt_dt_df.school_id != "-1")
    nt_dt_df = nt_dt_df[(nt_dt_df.school_id != "-1")]
    return nt_dt_df, students_df


@timer
def alloc_nt_dt_school_students(students_df, level):
    BLOCKGROUP_SCALE = 1
    TRACT_SCALE = 10
    COUNTY_SUBDIV_SCALE = 100
    COUNTY_SCALE = 10000000
    CENSUS_PLACES_SCALE = 100000

    age_levels = {"P": [0, 4], "E": [5, 10], "M": [11, 13], "H": [14, 18]}
    start_age = age_levels[level][0]
    end_age = age_levels[level][1]
    students_df = students_df[(students_df.pr_grade >= start_age) & (students_df.pr_grade <= end_age)].copy()
    students_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    students_df["allocated"] = False
    rgen = np.random.default_rng(seed=29)
    schools_df = get_schools(args.schools_file)
    # Get all schools with the required level
    schools_df = schools_df[(schools_df.level.str.find(level) != -1)]
    # Some schools have multiple levels and so students should be allocated proportionately. For simplicity, we
    # just allocate uniformly among the levels, e.g. a EMH school would have 1/3 students each in elem, middle and high
    # divide by the number of levels to get the proportion
    schools_df.loc[:, "students"] = np.int32(np.ceil(schools_df.students / schools_df.level.str.len()))
    nt_dt_df = pd.DataFrame()
    # pick schools for students from decreasing resolution; the goal is to allocate students as close to home as possible
    for scale in [TRACT_SCALE, COUNTY_SUBDIV_SCALE, COUNTY_SCALE]:
        students_places_df, students_df = alloc_nt_dt_school_students_region(students_df, schools_df, scale, rgen)
        print("students places", len(students_places_df))
        students_df.to_csv("upop_students-" + str(scale) + ".csv", sep="\t")
        nt_dt_df = pd.concat([nt_dt_df, students_places_df])

    nt_dt_df.to_csv("students_nt_dt-" + level + ".csv", sep="\t")
    return nt_dt_df


@timer
def alloc_nt_dt_students(args, school_students_df):
    print(Fore.GREEN + "Allocating students" + Fore.RESET)
    # convert grades to ages
    grade_categs_found = list(school_students_df["pr_grade"].unique())
    grade_categs_expected = list(grade_categs.categories)
    missing_grades = [x for x in grade_categs_found if x not in grade_categs_expected and x != "" and x is not None]
    if missing_grades:
        print("WARNING: Found missing categories for pr_grade:", missing_grades)
    school_students_df["pr_grade"] = school_students_df["pr_grade"].astype(grade_categs).cat.codes + 3

    # allocate students from each level
    students_nt_dt_df = pd.DataFrame()
    for level in ["P", "E", "M", "H"]:
        print("Allocating students for level", level)
        elem_students_nt_dt_df = alloc_nt_dt_school_students(school_students_df, level=level)
        students_nt_dt_df = pd.concat([students_nt_dt_df, elem_students_nt_dt_df])

    students_nt_dt_df.to_csv("students_nt_dt.csv", sep="\t")

    # the lines below for checking
    alloc_schools_df = (
        students_nt_dt_df.groupby(["school_id"])["p_id"].count().reset_index().rename(columns={"p_id": "alloc_students"})
    )
    schools_df = get_schools(args.schools_file)
    actual_schools_df = schools_df[["id", "students"]]
    actual_schools_df = actual_schools_df.rename(columns={"id": "school_id"})
    merged_df = actual_schools_df.merge(alloc_schools_df, on="school_id")
    merged_df["ratio"] = merged_df.alloc_students / merged_df.students
    merged_df.to_csv("merged_schools.csv", sep="\t", float_format="%.1f")

    return students_nt_dt_df


def get_nt_dt_unemp(unemp_df):
    print(Fore.GREEN + "Adding unemployed" + Fore.RESET)
    unemp_df = unemp_df[["p_id", "geoid", "pr_naics", "pr_grade"]].copy()
    unemp_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    unemp_df["role"] = "nope"
    # unemployed have no grade since that indicates the agent is in school
    unemp_df["grade"] = ""
    unemp_df["dest_geoid"] = unemp_df.orig_geoid
    return unemp_df


if __name__ == "__main__":
    t = time.time()
    np.random.seed(29)
    parser = argparse.ArgumentParser(description="Generate nighttime/daytime worker/student populations for UrbanPop from LODES")
    parser.add_argument("--urbanpop_files", "-f", required=True, nargs="+", help="UrbanPop feather files")
    parser.add_argument("--lodes_file", "-l", required=True, help="LODES7 origin-destination (OD) file in csv format")
    parser.add_argument("--output_file", "-o", required=True, help="Output file (will be written in feather format)")
    parser.add_argument("--schools_file", "-s", required=True, help="File containing schools data in CSV")
    args = parser.parse_args()

    upop_df = load_urbanpop_files(args.urbanpop_files)
    # now split into students, workers and unemployed
    # students includes all children with assigned grades and adults with grad or undergrad grades
    # all other children are homeschooled or not in daycare
    in_college = (upop_df.pr_grade == "grad") | (upop_df.pr_grade == "undergrad")  # & (upop_df.pr_age > 17)
    college_students_df = upop_df[in_college]
    in_school = (upop_df.pr_grade != "") & ~in_college
    school_students_df = upop_df[in_school].copy()
    is_employed = (upop_df.pr_emp_stat == "employed") | (upop_df.pr_emp_stat == "mil")
    workers_df = upop_df[is_employed & ~in_school & ~in_college]
    unemp_df = upop_df[~is_employed & ~in_school & ~in_college]
    tot = len(workers_df) + len(school_students_df) + len(college_students_df) + len(unemp_df)
    print("Counts:")
    print("  workers:         ", len(workers_df))
    print("  school students: ", len(school_students_df))
    print("  college students:", len(college_students_df))
    print("  unemployed:      ", len(unemp_df))
    print("  total:           ", tot)
    if tot != len(upop_df):
        print("ERROR: total agents mismatch, allocated", tot, "but found", len(upop_df), "in UrbanPop feather files")
    students_nt_dt_df = alloc_nt_dt_students(args, school_students_df)
    # now get workers
    workers_nt_dt_df = alloc_nt_dt_workers(args, workers_df)
    # append non-workers
    unemp_df = get_nt_dt_unemp(unemp_df)

    nt_dt_df = pd.concat([workers_nt_dt_df, unemp_df, students_nt_dt_df])
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    nt_dt_df.to_csv(args.output_file + "_nt_dt.csv")
    print("Wrote nt/dt data to", args.output_file + "_nt_dt.csv")

    print("Completed in %.2f s" % (time.time() - t))
