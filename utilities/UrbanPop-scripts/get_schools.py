#!/usr/bin/env python

import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import time
import censusgeocode as cg
import sys
import shapely.geometry

parser = argparse.ArgumentParser(description="Generate school list with Census Block Group GEOID")
parser.add_argument("--school_files", "-s", required=True, nargs="+", help="School CSV files")
parser.add_argument("--census_bg_files", "-c", required=True, nargs="+", help="Census Block Group (bg) shape files")
parser.add_argument("--college_file", help="College CSV file")
parser.add_argument("--child_care_file", help="Childcare CSV file")
args = parser.parse_args()


def fetch_census_geographies(school_df):
    # fetch census geographies corresponding to addresses - unfortunately, about 15% of address don't get a match
    school_df.rename(columns={"NCES ID": "id", "Address": "street", "City": "city", "State": "state", "Zip": "zip"}, inplace=True)
    school_df.to_csv("school_df.csv")
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
        df.to_csv("batch." + str(batch) + ".csv")
        dfs.append(df)
        print(len(dfs[-1].index), "records in %.3f s" % (time.time() - t), flush=True)

    geographies = pd.concat(dfs)
    num_addresses = len(geographies.index)
    not_found = geographies[(geographies.match == False)]
    not_found.to_csv("unmatched_address_schools.csv")
    geographies = geographies[(geographies.match == True)]
    print("Found", len(geographies.index), "address matches out of", num_addresses)
    geoids_df = pd.DataFrame()
    geoids_df["id"] = geographies.id
    # only need down to the census tract
    geoids_df["GEOID"] = (geographies.statefp + geographies.countyfp + geographies.tract).astype("int64")
    schools_geoids_df = school_df[["id", "Enrollment", "Start Grade", "End Grade", "Full Time Teachers"]].merge(
        geoids_df, on="id"
    )
    schools_geoids_df.to_csv("schools_geoids.csv")

    print("Processed", len(schools_geoids_df.index), "records in %.3f s" % (time.time() - start_t))


start_t = time.time()
school_df = pd.DataFrame()
for fname in args.school_files:
    print("Reading data from", fname, end=": ")
    t = time.time()
    df = pd.read_csv(fname, low_memory=False)[
        [
            "NCES ID",
            "Address",
            "City",
            "State",
            "Zip",
            "Latitude",
            "Longitude",
            "Enrollment",
            "Start Grade",
            "End Grade",
            "Full Time Teachers",
        ]
    ]
    school_df = pd.concat([school_df, df])
    print(len(df.index), "records in %.3f s" % (time.time() - t))

print("Reading Census bg files")
t = time.time()
census_bgs_df = pd.DataFrame()
for fname in args.census_bg_files:
    census_bgs = gpd.read_file(fname)
    census_bgs_df = pd.concat([census_bgs_df, census_bgs])
print("Read", len(census_bgs_df.index), "census bgs in %.3f s" % (time.time() - t))
# census_bgs_df.to_csv("census_bgs.csv")

geometry = [shapely.geometry.Point(xy) for xy in zip(school_df.Longitude, school_df.Latitude)]
school_gdf = gpd.GeoDataFrame(school_df, crs="EPSG:4269", geometry=geometry)
schools_with_geoids = pd.DataFrame(gpd.sjoin(school_gdf, census_bgs_df, how="left", predicate="within"))
schools_with_geoids = schools_with_geoids.rename(
    columns={
        "NCES ID": "id",
        "Enrollment": "students",
        "Full Time Teachers": "teachers",
        "Start Grade": "start_grade",
        "End Grade": "end_grade",
        "GEOID10": "geoid",
    }
)[["id", "students", "teachers", "start_grade", "end_grade", "geoid"]]
num_schools = len(schools_with_geoids)
# we could have schools without geoids - missing lng/lat?
schools_with_geoids.dropna(inplace=True)
# we could have schools without enrollment and/or staff data
schools_with_geoids = schools_with_geoids[(schools_with_geoids.students != -999) & (schools_with_geoids.teachers != -999)]
schools_with_geoids.to_csv("schools_with_geoids.csv")
students = schools_with_geoids.students.sum()
teachers = schools_with_geoids.teachers.sum()
print(
    "Wrote",
    len(schools_with_geoids),
    "school records with",
    students,
    "students and",
    teachers,
    "teachers to",
    "schools_with_geoids.csv",
)
print("Dropped", num_schools - len(schools_with_geoids), "incomplete records")
