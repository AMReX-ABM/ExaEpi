#!/usr/bin/env python

# This code is for generating nighttime/daytime flows that are missing from UrbanPop

import time
import pandas as pd
import sys
import argparse
import configparser
import glob
import numpy as np
from pandas.api.types import CategoricalDtype
from colorama import Fore
import get_schools
from get_schools import timer

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


region_scales = {12: "blockgroup", 11: "tract", 10: "county_subdiv", 7: "census_places", 5: "county", 2: "state"}

BLOCKGROUP_SCALE = 12
TRACT_SCALE = 11
COUNTY_SUBDIV_SCALE = 10
CENSUS_PLACES_SCALE = 7
COUNTY_SCALE = 5
STATE_SCALE = 2

age_levels = {
    "C": [3, 3],
    "P": [4, 4],
    "E": [5, 10],
    "M": [11, 13],
    "H": [14, 17],
    "U": [18, 19],
    "PE": [4, 10],
    "PEM": [4, 13],
    "PEMH": [4, 17],
    "EM": [5, 13],
    "EMH": [5, 17],
    "MH": [11, 17],
}


def perc_str(n, m):
    return "%d out of %d (%.2f%%)" % (n, m, 100.0 * float(n) / m)


DUMP_INTERMEDIATES = False

CORR_CHECK_LEVEL = 0.8


@timer
def load_urbanpop_files(fnames):
    print(f"{Fore.GREEN}Loading {len(fnames)} UrbanPop files{Fore.RESET}")
    # each urbanpop file contains the following:
    # p_id,pums_id,h_id,geoid,hh_size,hh_type,hh_living_arrangement,hh_age,hh_has_kids,hh_income,hh_nb_wrks,hh_nb_non_wrks,
    # hh_nb_adult_wrks,hh_nb_adult_non_wrks,hh_dwg,hh_tenure,hh_vehicles,pr_age,pr_sex,pr_race,pr_hsplat,pr_ipr,pr_naics,
    # pr_emp_stat,pr_travel,pr_veh_occ,pr_commute,pr_grade
    upop_df = pd.DataFrame()
    upop_dfs = []
    for i, fname in enumerate(fnames):
        print(f"  {i} {fname}")
        df_read = pd.read_feather(fname)[["p_id", "geoid", "pr_age", "pr_naics", "pr_emp_stat", "pr_grade"]]
        # strip letters from NAICS code
        df_read.pr_naics = df_read.pr_naics.str.extract(r"(\d+)").astype("float").fillna(-1).astype("int")
        upop_dfs.append(df_read)
    upop_df = pd.concat(upop_dfs, ignore_index=True)
    # upop_df.geoid = upop_df.geoid.astype(str)
    upop_df.sort_values(by=["p_id"], inplace=True)
    print(f"Processed {len(upop_df.index)} records from {len(fnames)} files")
    return upop_df


@timer
def get_lodes_groups(lodes_fnames):
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
    df = pd.DataFrame()
    for lodes_fname in lodes_fnames:
        print("Loading", lodes_fname)
        lodes_df = pd.read_csv(lodes_fname, dtype={"w_geocode": str, "h_geocode": str, "S000": int})[
            ["w_geocode", "h_geocode", "S000"]
        ]
        # print("Loaded", len(lodes_df), "entries")
        # truncate geoids to first 12, i.e. just census block groups
        lodes_df.h_geocode = lodes_df.h_geocode.str[:12]
        lodes_df.w_geocode = lodes_df.w_geocode.str[:12]
        # lodes_df = lodes_df.groupby(lodes_df.columns.tolist()).size().reset_index().rename(columns={0: "count"})
        lodes_df = lodes_df.groupby(["w_geocode", "h_geocode"]).S000.sum().reset_index().rename(columns={0: "count"})
        if DUMP_INTERMEDIATES:
            lodes_df.to_csv(lodes_fname + "-short.csv", index=False)
        df = pd.concat([df, lodes_df], ignore_index=True)
    return df


@timer
def alloc_workers(args, workers_df):
    print(Fore.GREEN + "Allocating workers" + Fore.RESET)
    # need to alloc files with the following columns:
    # p_id,role,orig_geoid,dest_geoid,lodes_segment,naics,grade,school_id
    # we can skip the lodes_segment, since we don't use it in later processing
    worker_groups = workers_df.groupby(["geoid"])
    lodes_df = get_lodes_groups(args.lodes_files)
    lodes_groups = lodes_df.groupby(["h_geocode"])
    nt_dt_df = pd.DataFrame()
    print(f"Assigning workers in {len(worker_groups)} GEOIDS")
    i = 0
    nt_dts = [nt_dt_df]
    for name, worker_group in worker_groups:
        i += 1
        if i % 100 == 0:
            print(f"  {i} {name[0]}")
        num_workers = len(worker_group)
        try:
            lodes_group = lodes_groups.get_group(name)
        except KeyError as err:
            # the home geoid derived from the upop workers is not found in the home (origin) geoid in the LODES data, so we have
            # no flows from that geoid
            print(f"{Fore.RED}WARNING: Could not find origin GEOID {name} in LODES data for {num_workers} workers{Fore.RESET}")
            lodes_group = pd.DataFrame()
            lodes_group["w_geocode"] = name
            lodes_group["h_geocode"] = name
            lodes_group["S000"] = num_workers
            # raise err
        sum_flows = lodes_group["S000"].sum()
        flow_probs = lodes_group["S000"] / sum_flows
        rnd_sample = lodes_group.sample(n=num_workers, weights=flow_probs, replace=True)
        nt_dt_group = worker_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=rnd_sample["w_geocode"].tolist())
        nt_dts.append(nt_dt_group)

    nt_dt_df = pd.concat(nt_dts, ignore_index=True)

    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    nt_dt_df["role"] = "worker"
    # workers have no grade since that indicates that the agent is in school
    nt_dt_df["grade"] = ""
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade"]]
    num_without_naics = len(nt_dt_df.loc[nt_dt_df["naics"] == "", "naics"])
    if num_without_naics > 0:
        print(f"{Fore.RED}WARNING: There are {num_without_naics} workers without NAICS classification{Fore.RESET}")
    print("Added destinations for", len(nt_dt_df), "workers")
    if DUMP_INTERMEDIATES:
        nt_dt_df.to_csv("workers_nt_dt.csv", index=False)
    # check that the actual allocated workers follow the lodes flows closely
    generated_df = nt_dt_df.groupby(["orig_geoid", "dest_geoid"]).size().reset_index(name="total")
    generated_df["key"] = generated_df["orig_geoid"].astype(str) + "-" + generated_df["dest_geoid"].astype(str)
    lodes_df["key"] = lodes_df["h_geocode"].astype(str) + "-" + lodes_df["w_geocode"].astype(str)
    merged_df = generated_df.merge(lodes_df, on="key", how="outer")[["key", "total", "S000"]]
    merged_df = merged_df.fillna(0).astype({"total": "int", "S000": "int"})
    if DUMP_INTERMEDIATES:
        merged_df.to_csv("worker_check_ods.csv")
    corr = merged_df.total.corr(merged_df.S000)
    print(f"LODES flows correlation: {corr:.3f} (generated count {merged_df.total.sum()}, actual count {merged_df.S000.sum()})")
    if corr < CORR_CHECK_LEVEL:
        print(f"{Fore.RED}WARNING: low correlation!{Fore.RESET}")

    return nt_dt_df


@timer
def load_schools(fname):
    print("Loading schools from", fname)
    schools_df = pd.read_csv(fname, low_memory=False, dtype={"geoid": str})
    print("Loaded", len(schools_df), "entries:", schools_df.students.sum(), "students,", schools_df.teachers.sum(), "teachers")
    if DUMP_INTERMEDIATES:
        schools_df.to_csv("schools.csv", sep="\t", index=False)
    return schools_df


# @timer
def alloc_students_region(students_df, schools_df, geoid_scaling, alloc_all):
    schools_df["region"] = schools_df.geoid.str[:geoid_scaling]
    # group by region
    students_df = students_df[(students_df.school_id == "")]
    students_df.loc[:, "region"] = students_df.geoid.str[:geoid_scaling]
    student_groups = students_df.groupby(["region"])
    school_groups = schools_df.groupby(["region"])
    nt_dt_df = pd.DataFrame()
    nt_dt_dfs = []
    missing_regions = 0
    for group_name, student_group in student_groups:
        num_students_reqd = len(student_group)
        if num_students_reqd == 0:
            print(f"{Fore.RED}WARNING: no students requested for group {group_name}")
            continue
        # defaults to empty in case we can't find a key
        try:
            school_group = school_groups.get_group(group_name)[["geoid", "id", "remaining_student_places"]]
        except KeyError:
            missing_regions += 1
            continue
        sum_remaining_student_places = school_group.remaining_student_places.sum()
        if not alloc_all and num_students_reqd > sum_remaining_student_places:
            # add an extra dummy row so that all students have a chance of being allocated to available slots
            school_group.loc[len(school_group)] = ["", "", num_students_reqd - sum_remaining_student_places]
            # calculate probs with sum of newly remaining places (incl dummy spots)
            school_probs = school_group.remaining_student_places / school_group.remaining_student_places.sum()
        else:
            school_probs = school_group.remaining_student_places / sum_remaining_student_places
        schools_selected = school_group.sample(n=num_students_reqd, weights=school_probs, replace=True)
        schools_selected.index.rename("index", inplace=True)

        nt_dt_group = student_group[["p_id", "geoid", "pr_naics", "pr_grade"]]
        nt_dt_group = nt_dt_group.assign(dest_geoid=schools_selected.geoid.tolist(), school_id=schools_selected.id.tolist())
        nt_dt_dfs.append(nt_dt_group)
        # clear out dummy schools
        schools_selected = schools_selected[(schools_selected.id != "")]
        # reduce the available students at schools count according to how many have been allocated
        schools_selected_groups = schools_selected.groupby(["index"])
        selected_school_indexes = list(schools_selected_groups.groups.keys())
        selected_school_counts = schools_selected_groups.id.count().tolist()
        indexes = schools_df.index.isin(selected_school_indexes)
        schools_df.loc[indexes, "remaining_student_places"] -= np.int32(selected_school_counts)
        # always set to 1 to enable a slight chance of allocating to this school
        schools_df.loc[schools_df.remaining_student_places < 1, "remaining_student_places"] = 1

    if len(nt_dt_dfs) == 0:
        return nt_dt_df, students_df[(students_df.school_id == "")]

    nt_dt_df = pd.concat(nt_dt_dfs, ignore_index=True)
    # print("    Found", len(student_groups), "student regions,", missing_regions, "without schools")
    if len(nt_dt_df) > 0:
        nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
        nt_dt_df["role"] = "student"
        # reorder the columns
        nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]]
    else:
        nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]] = pd.DataFrame(
            [["", "", "", "", 0, 0, ""]], index=nt_dt_df.index
        )

    nt_dt_df.naics = ""
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)

    idx_to = students_df["p_id"].isin(nt_dt_df["p_id"])
    idx_from = nt_dt_df["p_id"].isin(students_df["p_id"])
    students_df.loc[idx_to, "school_id"] = nt_dt_df.loc[idx_from, "school_id"].values

    # only subset if this is not the last round
    if not alloc_all:
        nt_dt_df = nt_dt_df[(nt_dt_df.school_id != "")]
    return nt_dt_df, students_df[(students_df.school_id == "")]


@timer
def alloc_students_level(schools_df, students_df, level):
    start_age = age_levels[level][0]
    end_age = age_levels[level][1]
    students_df = students_df[(students_df.pr_grade >= start_age) & (students_df.pr_grade <= end_age)].copy()
    print(f"Allocating {len(students_df)} students for level {level}")
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
    schools_df["remaining_student_places"] = schools_df.students
    nt_dt_df = pd.DataFrame()
    # for childcare, we assume that it is all close to home
    scales = list(region_scales.keys()) if level != "C" else list(region_scales.keys())[:3]
    nt_dt_dfs = []
    # pick schools for students from decreasing resolution; the goal is to allocate students as close to home as possible
    for scale in scales:
        # print("  Region", region_scales[scale])
        # make sure to allocate all at the final region scale
        alloc_all = True if scale == scales[-1] else False
        students_nt_dt_df, students_df = alloc_students_region(students_df, schools_df, scale, alloc_all)
        num_unalloc_students = len(students_df[(students_df.school_id == "")])
        # print("    Set destinations for", len(students_nt_dt_df), "students,", num_unalloc_students, "unallocated")
        nt_dt_dfs.append(students_nt_dt_df)
        if alloc_all:
            unalloc_students_df = students_df[(students_df.school_id == "")].copy()
            unalloc_students_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
            unalloc_students_df["role"] = "student"
            unalloc_students_df["dest_geoid"] = unalloc_students_df["orig_geoid"]
            unalloc_students_df = unalloc_students_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]]
            nt_dt_dfs.append(unalloc_students_df)
        if num_unalloc_students == 0:
            break

    nt_dt_df = pd.concat(nt_dt_dfs, ignore_index=True)

    return nt_dt_df


@timer
def alloc_students(args, students_df):
    print(Fore.GREEN + "Allocating students" + Fore.RESET)
    schools_df = load_schools(args.schools_file)
    # convert grades to ages
    grade_categs_found = list(students_df["pr_grade"].unique())
    grade_categs_expected = list(grade_categs.categories)
    missing_grades = [x for x in grade_categs_found if x not in grade_categs_expected and x != "" and x is not None]
    if missing_grades:
        print("WARNING: Found missing categories for pr_grade:", missing_grades)
    students_df["pr_grade"] = students_df["pr_grade"].astype(grade_categs).cat.codes + 3
    students_df["school_id"] = ""

    # allocate students from each level
    students_nt_dt_df = pd.DataFrame()
    students_nt_dt_dfs = []
    for level in ["C", "P", "E", "M", "H", "U"]:
        df = alloc_students_level(schools_df, students_df, level=level)
        students_nt_dt_dfs.append(df)

    students_nt_dt_df = pd.concat(students_nt_dt_dfs, ignore_index=True)

    # set all unallocated students
    unalloc = students_nt_dt_df.school_id == ""
    print("Unallocated students:", len(students_nt_dt_df.loc[unalloc]))
    students_nt_dt_df.loc[unalloc, "dest_geoid"] = students_nt_dt_df.orig_geoid
    students_nt_dt_df.loc[unalloc, "grade"] = -1
    students_nt_dt_df.loc[unalloc, "role"] = "nope"

    if DUMP_INTERMEDIATES:
        students_nt_dt_df.to_csv("students_nt_dt.csv", sep="\t", index=False)

    gen_schools_df = (
        students_nt_dt_df.loc[students_nt_dt_df["role"] == "student"].groupby(["school_id"]).size().reset_index(name="students")
    )
    schools_check_df = schools_df.copy()
    gen_schools_df["key"] = gen_schools_df["school_id"].astype(str)
    schools_check_df["key"] = schools_check_df["id"].astype(str)
    # only merge in the matching schools from the actual data, since that consists of potentially many more schools than we are
    # using for this subset (e.g. NM)
    merged_df = gen_schools_df.merge(schools_check_df, on="key", how="left", suffixes=["_gen", "_orig"])[
        ["key", "students_gen", "students_orig"]
    ]
    merged_df = merged_df.fillna(0).astype({"students_gen": "int", "students_orig": "int"})
    if DUMP_INTERMEDIATES:
        merged_df.to_csv("schools_check.csv")
    corr = merged_df.students_gen.corr(merged_df.students_orig)
    print(
        f"Student school counts correlation: {corr:.3f} "
        + f"(generated count {merged_df.students_gen.sum()}, actual count {merged_df.students_orig.sum()})"
    )
    if corr < CORR_CHECK_LEVEL:
        print(f"{Fore.RED}WARNING: low correlation!{Fore.RESET}")

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
        df.to_csv("upop.csv", sep="\t", index=False)
    for age in np.arange(0, 5):
        is_candidate = (df.pr_grade == "") & (df.pr_age == age)
        num_candidates = int(len(df[is_candidate]) * PROBS[age])
        to_set = df[is_candidate].sample(n=num_candidates, replace=False)
        print("  Age", age, "set", len(to_set), "out of", len(df[is_candidate]))
        df.loc[to_set.index, "pr_grade"] = "childcare"

    print("Set", len(df[(df.pr_grade.str.startswith("childcare"))]), "agents to childcare")
    if DUMP_INTERMEDIATES:
        df.to_csv("childcare_set.csv", sep="\t", index=False)
    return df


def alloc_teachers_region(teachers_df, schools_df, geoid_scaling):
    schools_df["region"] = schools_df.geoid.str[:geoid_scaling]
    teachers_df = teachers_df[teachers_df.grade == ""].copy()
    teachers_df["region"] = teachers_df.orig_geoid.str[:geoid_scaling]
    teacher_groups = teachers_df.groupby(["region"])
    school_groups = schools_df.groupby(["region"])
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
        num_teachers_reqd = min(num_teachers_reqd, teacher_group.p_id.count())
        teachers_selected = teacher_group.sample(n=num_teachers_reqd, replace=False)
        probs = school_group.adj_teachers / num_teachers_reqd
        schools_selected = school_group.sample(n=num_teachers_reqd, weights=probs, replace=True)
        schools_selected.index.rename("index", inplace=True)
        schools_selected["grade"] = (
            schools_selected["level"].map(age_levels).apply(lambda x: np.random.randint(x[0], x[1] + 1)).astype("str")
        )
        # the university teachers will be uniformly split betw undergrad and grad. Readjust this to a 80:20 split
        # FIXME: at college level, "teachers" will likely interact with both grad and undergrads
        schools_selected["rnd"] = np.random.uniform(size=len(schools_selected))
        schools_selected.loc[(schools_selected.grade == "19") & (schools_selected.rnd < 0.5), "grade"] = "18"
        teachers_df.loc[teachers_selected.index, "school_id"] = list(schools_selected.id)
        teachers_df.loc[teachers_selected.index, "grade"] = list(schools_selected.grade)
        teachers_df.loc[teachers_selected.index, "dest_geoid"] = list(schools_selected.geoid)
        # reduce the reqd teachers count according to how many have been allocated
        schools_selected_groups = schools_selected.groupby(["index"])
        selected_school_indexes = list(schools_selected_groups.groups.keys())
        selected_school_counts = schools_selected_groups.id.count().tolist()
        schools_df.loc[schools_df.index.isin(selected_school_indexes), "adj_teachers"] -= np.int32(selected_school_counts)
        schools_df.loc[schools_df.adj_teachers < 0, "adj_teachers"] = 0
        schools_df.loc[schools_df.index.isin(selected_school_indexes), "alloc_teachers"] -= np.int32(selected_school_counts)

    teachers_df.drop("region", axis=1, inplace=True)
    num_allocated = len(teachers_df[(teachers_df.grade != "")])
    print(f"  For {region_scales[geoid_scaling]}, allocated {num_allocated} teachers out of {tot_reqd_teachers}")
    return teachers_df


@timer
def alloc_teachers(workers_nt_dt_df, students_nt_dt_df, schools_df, school_type):
    print(f"{Fore.GREEN}Allocating teachers for {school_type}{Fore.RESET}")
    # The number of students at each school will not exactly match the original schools data, so we use the actual number
    # of students we have allocated, and the original student/teacher ratio to set the desired teacher counts for each school
    if DUMP_INTERMEDIATES:
        schools_df.to_csv("schools.csv", sep="\t", index=False)

    if school_type == "childcare":
        naics_code = 6244
        schools_df = schools_df[schools_df.level == "C"]
        students_nt_dt_df = students_nt_dt_df[students_nt_dt_df.grade <= 3]
    elif school_type == "secondary":
        naics_code = 6111
        schools_df = schools_df[
            (schools_df.level == "E")
            | (schools_df.level == "EM")
            | (schools_df.level == "EMH")
            | (schools_df.level == "M")
            | (schools_df.level == "MH")
            | (schools_df.level == "H")
            | (schools_df.level == "P")
            | (schools_df.level == "PE")
            | (schools_df.level == "PEM")
            | (schools_df.level == "PEMH")
        ]
        students_nt_dt_df = students_nt_dt_df[(students_nt_dt_df.grade > 3) & (students_nt_dt_df.grade <= 17)]
    elif school_type == "university":
        naics_code = 611
        schools_df = schools_df[schools_df.level == "U"]
        students_nt_dt_df = students_nt_dt_df[(students_nt_dt_df.grade >= 18) & (students_nt_dt_df.grade <= 19)]
    else:
        err_str = f"{Fore.RED}ERROR: school_type {school_type} not valid{Fore.RESET}"
        raise RuntimeError(err_str)

    print(f"Number of schools to use {len(schools_df)}, number of students {len(students_nt_dt_df)}")

    alloc_schools_df = (
        students_nt_dt_df.groupby(["school_id"])["p_id"]
        .count()
        .reset_index()
        .rename(columns={"p_id": "alloc_students", "school_id": "id"})
    )
    schools_df = schools_df.merge(alloc_schools_df, on="id")
    schools_df["ratio"] = schools_df.alloc_students / schools_df.students
    schools_df["adj_teachers"] = np.int32(np.ceil(schools_df.teachers * schools_df.ratio))
    if DUMP_INTERMEDIATES:
        schools_df.to_csv("selected_schools.csv", sep="\t", float_format="%.2f", index=False)
    schools_df["alloc_teachers"] = schools_df.adj_teachers
    teachers_df = workers_nt_dt_df[workers_nt_dt_df.naics == naics_code].copy()
    teachers_df["school_id"] = None
    teachers_df["grade"] = ""
    num_reqd_teachers = schools_df.adj_teachers.sum()
    print("Found", len(teachers_df), "edu workers for", num_reqd_teachers, "required teachers")
    scales = list(region_scales.keys())
    # pick schools for teachers from decreasing resolution; the goal is to allocate teachers as close to home as possible
    for scale in scales:
        # print(f"  Region {region_scales[scale]}")
        teachers_df = alloc_teachers_region(teachers_df, schools_df, scale)
        workers_nt_dt_df.loc[teachers_df.index, "school_id"] = teachers_df.school_id
        workers_nt_dt_df.loc[teachers_df.index, "dest_geoid"] = teachers_df.dest_geoid
        workers_nt_dt_df.loc[teachers_df.index, "grade"] = teachers_df.grade
        if DUMP_INTERMEDIATES:
            teachers_df.to_csv("teachers-" + region_scales[scale] + ".csv", sep="\t", index=False)

    return workers_nt_dt_df


def check_teacher_corr(workers_nt_dt_df, schools_df):
    gen_schools_df = (
        workers_nt_dt_df.loc[workers_nt_dt_df["school_id"] != ""].groupby(["school_id"]).size().reset_index(name="teachers")
    )
    schools_check_df = schools_df.copy()
    gen_schools_df["key"] = gen_schools_df["school_id"].astype(str)
    schools_check_df["key"] = schools_check_df["id"].astype(str)
    # only merge in the matching schools from the actual data, since that consists of potentially many more schools than we are
    # using for this subset (e.g. NM)
    merged_df = gen_schools_df.merge(schools_check_df, on="key", how="left", suffixes=["_gen", "_orig"])[
        ["key", "teachers_gen", "teachers_orig"]
    ]
    merged_df = merged_df.fillna(0).astype({"teachers_gen": "int", "teachers_orig": "int"})
    if DUMP_INTERMEDIATES:
        merged_df.to_csv("schools_check.csv")
    corr = merged_df.teachers_gen.corr(merged_df.teachers_orig)
    print(
        f"Teacher school counts correlation: {corr:.3f} "
        + f"(generated count {merged_df.teachers_gen.sum()}, actual count {merged_df.teachers_orig.sum()})"
    )
    if corr < CORR_CHECK_LEVEL:
        print(f"{Fore.RED}WARNING: low correlation!{Fore.RESET}")


def get_level_from_age(min_age, max_age):
    if min_age >= 18:
        return "U"
    if max_age == 3:
        return "C"
    return get_schools.get_level_from_age(min_age, max_age)


@timer
def get_from_up_nt_dt(args, upop_df):
    df = pd.DataFrame()
    print(f"{Fore.GREEN}Loading UrbanPop nighttime/daytime files{Fore.RESET}")
    for fname in args.up_nt_dt_files:
        region_df = pd.read_feather(fname)[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade", "school_id"]].astype(
            {"orig_geoid": "str", "dest_geoid": "str", "grade": "str"}
        )
        df = pd.concat([df, region_df], ignore_index=True)
    print(f"Loaded {len(df)} entries from {len(args.up_nt_dt_files)} files")
    workers_df = df[df.role == "worker"].reset_index(drop=True)
    workers_df["grade"] = ""
    workers_df["naics"] = workers_df.merge(upop_df, on="p_id", how="left")["pr_naics"]

    unemp_df = df[df.role == "nope"].reset_index(drop=True)
    unemp_df["grade"] = ""
    students_df = df[df.role == "student"].reset_index(drop=True)
    idx = students_df["grade"].isin(["childcare", "undergrad_male", "undergrad_female", "grad_male", "grad_female"])
    students_df.loc[idx, "grade"] = students_df.loc[idx, "grade"].str.split("_", n=1, expand=True).iloc[:, 0]
    students_df.loc[~idx, "grade"] = students_df.loc[~idx, "grade"].str.split("_", n=1, expand=True).iloc[:, 1]
    students_df["grade"] = students_df["grade"].astype(grade_categs).cat.codes + 3
    students_df.to_csv("students_from_up.csv", index=False)
    schools_df = students_df.groupby(["school_id", "dest_geoid"]).size().reset_index(name="students")
    schools_df.rename(columns={"school_id": "id", "dest_geoid": "geoid"}, inplace=True)
    # we don't have information for these schools, so we just use an average student:teacher ratio of 12:1
    schools_df["teachers"] = np.ceil(schools_df.students / 12).astype(int)
    schools_df["min_age"] = students_df.groupby(["school_id", "dest_geoid"], as_index=False)["grade"].min()["grade"]
    schools_df["max_age"] = students_df.groupby(["school_id", "dest_geoid"], as_index=False)["grade"].max()["grade"]
    schools_df["level"] = list(map(get_level_from_age, schools_df["min_age"], schools_df["max_age"]))
    schools_df = schools_df[["id", "students", "teachers", "level", "geoid"]]
    schools_df.to_csv("schools_with_geoids_up.csv", index=False)
    return workers_df, students_df, schools_df, unemp_df


@timer
def main():
    t = time.time()
    cfg_parser = argparse.ArgumentParser(
        description="Generate nighttime/daytime worker/student populations for UrbanPop from LODES", add_help=False
    )
    cfg_parser.add_argument("-c", "--config", help="Config file", metavar="FILE")
    args, remaining_argv = cfg_parser.parse_known_args()
    # set defaults
    main_args = {
        "urbanpop_files": "",
        "lodes_files": "",
        "schools_file": "",
        "output_file": "",
        "up_nt_dt_files": "",
        "rseed": 11,
    }
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read([args.config])
        main_args.update(dict(cfg.items("main")))
        for files_label in ["urbanpop_files", "lodes_files", "up_nt_dt_files"]:
            file_list = []
            for f in main_args[files_label].split():
                file_list.extend(glob.glob(f))
            main_args[files_label] = file_list
    parser = argparse.ArgumentParser(parents=[cfg_parser])
    parser.set_defaults(**main_args)
    parser.add_argument("--urbanpop_files", "-f", nargs="+", help="UrbanPop feather files")
    parser.add_argument(
        "--lodes_files",
        "-l",
        nargs="+",
        help="LODES7 origin-destination (OD) files in CSV format. Not needed if UrbanPop nighttime/daytime files are used",
    )
    parser.add_argument("--output_file", "-o", help="Output file prefix (will be written in feather format)")
    parser.add_argument(
        "--schools_file",
        "-s",
        help="File containing schools data in CSV format. Not needed if UrbanPop nighttime/daytime files are used",
    )
    parser.add_argument(
        "--up_nt_dt_files",
        "-n",
        nargs="+",
        help="Files containing nighttime/daytime data from UrbanPop in feather format. Used instead of LODES files",
    )
    parser.add_argument("--rseed", "-r", help="Random seed", default=29, type=int)
    args = parser.parse_args(remaining_argv)
    print(Fore.CYAN, "Options:", sep="")
    for arg, value in args.__dict__.items():
        print(f"  {arg:20s} {value}")
    print(Fore.RESET, end="")

    np.random.seed(args.rseed)

    upop_df = load_urbanpop_files(args.urbanpop_files)
    # randomly allocate some young agents to childcare
    upop_df = set_childcare(upop_df)
    # now split into students, workers and unemployed.
    # Note that the UrbanPop nt/dt data classifies mil as unemployed (nope); because they don't commute?
    # UrbanPop nt/dt sometimes classifies employed people with undergrad/grad grade as students, not workers. Not sure when/why
    # it does this. So not sure what to do. Taking all employed as workers only and not students give a better match to the
    # schools data
    # is_employed = (upop_df.pr_emp_stat == "employed") & (upop_df.pr_grade != "undergrad") & (upop_df.pr_grade != "grad")
    is_employed = upop_df.pr_emp_stat == "employed"
    in_school = (upop_df.pr_grade != "") & ~is_employed
    students_df = upop_df[in_school].copy()
    workers_df = upop_df[is_employed]
    unemp_df = upop_df[~is_employed & ~in_school]
    tot = len(workers_df) + len(students_df) + len(unemp_df)
    print("Counts:")
    print("  workers:   ", len(workers_df))
    print("  students:  ", len(students_df))
    print("  unemployed:", len(unemp_df))
    print("  total:     ", tot)
    if tot != len(upop_df):
        print("ERROR: total agents mismatch, allocated", tot, "but found", len(upop_df), "in UrbanPop feather files")

    if args.up_nt_dt_files:
        workers_nt_dt_df, students_nt_dt_df, schools_df, unemp_df = get_from_up_nt_dt(args, upop_df)
    else:
        workers_nt_dt_df = alloc_workers(args, workers_df)
        students_nt_dt_df = alloc_students(args, students_df)
        schools_df = load_schools(args.schools_file)
        unemp_df = get_unemp(unemp_df)

    workers_nt_dt_df = alloc_teachers(workers_nt_dt_df, students_nt_dt_df, schools_df, "childcare")
    workers_nt_dt_df = alloc_teachers(workers_nt_dt_df, students_nt_dt_df, schools_df, "secondary")
    workers_nt_dt_df = alloc_teachers(workers_nt_dt_df, students_nt_dt_df, schools_df, "university")
    check_teacher_corr(workers_nt_dt_df, schools_df)

    nt_dt_df = pd.concat([workers_nt_dt_df, unemp_df, students_nt_dt_df], ignore_index=True)
    # these astype calls are needed to satisfy the to_feather call
    nt_dt_df.grade = nt_dt_df.grade.astype(str)
    nt_dt_df.loc[nt_dt_df.grade == "-1", "grade"] = ""
    nt_dt_df.naics = nt_dt_df.naics.astype(str)
    nt_dt_df.loc[(nt_dt_df.naics == "") | (nt_dt_df.naics == "None"), "naics"] = "-1"
    # ensure dest_geoid is set to origin if blank
    nt_dt_df.loc[nt_dt_df.dest_geoid == "", "dest_geoid"] = nt_dt_df.orig_geoid
    nt_dt_df.sort_values(by=["p_id"], inplace=True, ignore_index=True)
    nt_dt_df.to_csv(args.output_file + "_nt_dt.csv", index=False)
    nt_dt_df.to_feather(args.output_file + "_nt_dt.feather")
    print("Wrote nt/dt data to", args.output_file + "_nt_dt.csv")

    print("Completed in %.2f s" % (time.time() - t))


if __name__ == "__main__":
    main()
