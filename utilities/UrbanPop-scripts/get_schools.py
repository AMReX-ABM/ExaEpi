#!/usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import time
import censusgeocode as cg
import sys
import shapely.geometry


def fetch_census_geographies(school_df):
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


def get_age_from_grade(grade_str):
    try:
        return int(grade_str) + 5
    except:
        if grade_str == "KG":
            return 5
        if grade_str == "PK":
            return 4
    return -1


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


def get_schools(args, census_bgs_df):
    school_df = pd.DataFrame()
    cols_to_read = ["NCES ID", "Latitude", "Longitude", "Enrollment", "Start Grade", "End Grade", "Full Time Teachers"]
    for fname in args.private_school_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[cols_to_read]
        # the grades are actually ages for these private schools
        df["level"] = list(map(get_level_from_age, df["Start Grade"], df["End Grade"]))
        school_df = pd.concat([school_df, df], ignore_index=True)
        print(len(df.index), "records in % .3f s" % (time.time() - t))

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
    )[["id", "students", "teachers", "level", "geoid"]]
    schools_with_geoids = get_complete(schools_with_geoids)
    schools_with_geoids.to_csv("non_college_schools_with_geoids.csv", index=False)
    print("Wrote", len(schools_with_geoids), "schools to non_college_schools_with_geoids.csv")
    return schools_with_geoids


def get_childcare(args, census_bgs_df):
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
    avg_childcare = int(childcare_complete.students.sum() / len(childcare_complete))
    print("Avg childcare size", avg_childcare)
    childcare_with_geoids.loc[(childcare_with_geoids.students <= 0), "students"] = avg_childcare
    # childcare_with_geoids = childcare_with_geoids[childcare_with_geoids.students > 0]
    # assume 5 children per adult
    childcare_with_geoids.insert(childcare_with_geoids.columns.get_loc("students") + 1, "teachers", int(0))
    childcare_with_geoids.insert(childcare_with_geoids.columns.get_loc("teachers") + 1, "level", "C")
    childcare_with_geoids.teachers = np.int32(np.ceil(childcare_with_geoids.students / 7))
    sum_children = childcare_with_geoids.students.sum()
    childcare_with_geoids.to_csv("childcare_with_geoids.csv", index=False)
    print("Wrote", len(childcare_with_geoids), "childcare records to childcare_with_geoids.csv")
    print("Total children:", sum_children)
    print("Dropped", num_childcare - len(childcare_with_geoids), "incomplete records")
    return childcare_with_geoids


def get_colleges(args):
    start_t = time.time()
    # we don't have lng/lat for colleges so we have to fetch with addresses
    colleges_df = pd.DataFrame()
    for fname in args.college_files:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df = pd.read_csv(fname, low_memory=False)[["UNIQUEID", "ADDRESS", "CITY", "STATE", "ZIP", "TOT_ENROLL", "TOT_EMP"]]
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
    colleges_with_geoids = colleges_df[["id", "students", "teachers", "level"]].merge(geoids_df, on="id")
    colleges_with_geoids = get_complete(colleges_with_geoids)
    colleges_with_geoids.to_csv("colleges_with_geoids.csv", index=False)
    print("Wrote", len(colleges_with_geoids), "colleges to colleges_with_geoids.csv")
    return colleges_with_geoids


parser = argparse.ArgumentParser(
    description="Generate school list with Census Block Group GEOID, using Census bg shapefiles and HIFLD data"
)
parser.add_argument("--private_school_files", "-p", required=True, nargs="+", help="Private school CSV files")
parser.add_argument("--public_school_files", "-s", required=True, nargs="+", help="Public school CSV files")
parser.add_argument("--census_bg_files", "-c", required=True, nargs="+", help="Census Block Group (bg) shape files")
parser.add_argument("--college_files", "-u", required=True, nargs="+", help="College/University CSV files")
parser.add_argument("--childcare_files", "-a", required=True, nargs="+", help="Childcare CSV files")
args = parser.parse_args()

start_t = time.time()
census_bgs_df = get_census_bgs(args)
childcare_geoids_df = get_childcare(args, census_bgs_df)
colleges_geoids_df = get_colleges(args)
schools_geoids_df = get_schools(args, census_bgs_df)
schools_geoids_df = pd.concat([schools_geoids_df, colleges_geoids_df, childcare_geoids_df], ignore_index=True)
schools_geoids_df.to_csv("schools_with_geoids.csv", index=False)
print("Finished in %.3f s" % (time.time() - start_t))
