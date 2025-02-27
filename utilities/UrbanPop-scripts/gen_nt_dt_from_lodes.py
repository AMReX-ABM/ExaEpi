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


naics_to_description = {
    11: "agr_ffh",  # Agriculture, forestry, fishing and hunting
    21: "ext",  # Mining, quarrying, and oil and gas extraction
    22: "utl",  # Utilities
    23: "con",  # Construction
    31: "mfg",  # Manufacturing
    32: "mfg",  # Manufacturing
    33: "mfg",  # Manufacturing
    42: "whl",  # Wholesale trade
    44: "ret",  # Retail trade
    45: "ret",  # Retail trade
    48: "trn_whs",  # Transportation and warehousing
    49: "trn_whs",  # Transportation and warehousing
    51: "inf",  # Information
    52: "fin_ins",  # Finance and insurance
    53: "rrl",  # Real estate rental and leasing
    54: "prf",  # Professional, scientific and technical services
    55: "mgt",  # Management of companies and enterprises
    56: "adm_wmr",  # Administrative and support and waste management and remediating services
    61: "edu",  # Educational services
    62: "med_sca",  # Health care and social services
    71: "ent",  # Arts, entertainment and recreation
    72: "afs",  # Accomodation and food services
    81: "srv",  # Other services (except public administration)
    92: "pad",  # Public administration
    100: "wfh",  # Work from home
}


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time for {func.__name__}: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


def perc_str(n, m):
    return "%d out of %d (%.2f%%)" % (n, m, 100.0 * float(n) / m)


def load_urbanpop_files(fnames):
    # each urbanpop file contains the following:
    # p_id,pums_id,h_id,geoid,hh_size,hh_type,hh_living_arrangement,hh_age,hh_has_kids,hh_income,hh_nb_wrks,hh_nb_non_wrks,
    # hh_nb_adult_wrks,hh_nb_adult_non_wrks,hh_dwg,hh_tenure,hh_vehicles,pr_age,pr_sex,pr_race,pr_hsplat,pr_ipr,pr_naics,
    # pr_emp_stat,pr_travel,pr_veh_occ,pr_commute,pr_grade
    dfs = []
    start_t = time.time()
    for fname in fnames:
        print("Reading data from", fname, end=": ")
        t = time.time()
        df_read = pd.read_feather(fname)[["p_id", "geoid", "pr_age", "pr_naics", "pr_emp_stat", "pr_grade"]]
        dfs.append(df_read)
        print(len(dfs[-1].index), "records in %.3f s" % (time.time() - t))

    df = pd.concat(dfs)
    df.geoid = df.geoid.astype("int64")
    df.sort_values(by=["p_id"], inplace=True)
    print("Processed", len(df.index), "records in %.3f s" % (time.time() - start_t))
    return df


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
def generate_nt_dt_feather_files(urbanpop_df, lodes_df):
    # need to generate files with the following columns:
    # p_id,role,orig_geoid,dest_geoid,lodes_segment,naics,grade,school_id
    # we can blank out the lodes_segment
    workers = urbanpop_df[
        ((urbanpop_df.pr_emp_stat == "employed") | (urbanpop_df.pr_emp_stat == "mil")) & (urbanpop_df.pr_grade == "")
    ]
    print("Found", len(workers), "workers")
    worker_groups = workers.groupby(["geoid"])
    lodes_groups = lodes_df.groupby(["h_geocode"])
    rgen = np.random.default_rng(seed=29)
    nt_dt_groups = []
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
        # nt_dt_group["role"] = "worker"
        # nt_dt_group.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
        # nt_dt_group = nt_dt_group[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade"]]
        # convert NAICS code to string used in nt_dt files
        nt_dt_groups.append(nt_dt_group)

    nt_dt_df = pd.concat(nt_dt_groups)
    nt_dt_df["role"] = "worker"
    nt_dt_df.rename(columns={"geoid": "orig_geoid", "pr_naics": "naics", "pr_grade": "grade"}, inplace=True)
    # reorder the columns
    nt_dt_df = nt_dt_df[["p_id", "role", "orig_geoid", "dest_geoid", "naics", "grade"]]
    # nt_dt_df.naics = nt_dt_df.naics.map(naics_to_description)
    print("Added destinations for", len(nt_dt_df), "workers")
    nt_dt_df.to_csv("workers_nt_dt.csv")
    # append non-workers
    non_workers = urbanpop_df[(urbanpop_df.pr_emp_stat != "employed") & (urbanpop_df.pr_emp_stat != "mil")]
    print("Found", len(non_workers), "unemployed")
    # nt_dt_df = pd.concat([nt_dt_df, non_workers])
    # nt_dt_df.to_csv("nt_dt.csv")


if __name__ == "__main__":
    t = time.time()
    np.random.seed(29)
    parser = argparse.ArgumentParser(description="Generate nighttime/daytime worker/student populations for UrbanPop from LODES")
    parser.add_argument("--urbanpop_files", "-f", required=True, nargs="+", help="UrbanPop feather files")
    parser.add_argument("--lodes_file", "-l", required=True, help="LODES7 origin-destination (OD) file in csv format")
    args = parser.parse_args()

    urbanpop_df = load_urbanpop_files(args.urbanpop_files)
    lodes_groups_df = get_lodes_groups(args.lodes_file)
    generate_nt_dt_feather_files(urbanpop_df, lodes_groups_df)

    print("Completed in %.2f s" % (time.time() - t))
