#!/usr/bin/env python

import pandas as pd
import argparse
import configparser
import glob
from colorama import Fore


cfg_parser = argparse.ArgumentParser(
    description="Check generated daytime/nighttime flows and schools data against UrbanPop nt/dt", add_help=False
)
cfg_parser.add_argument("-c", "--config", help="Config file", metavar="FILE")
args, remaining_argv = cfg_parser.parse_known_args()
main_args = {"generated_nt_dt_file": "", "urbanpop_nt_dt_file": "", "schools_file": ""}
if args.config:
    cfg = configparser.ConfigParser()
    cfg.read([args.config])
    main_args.update(dict(cfg.items("main")))
parser = argparse.ArgumentParser(parents=[cfg_parser])
parser.set_defaults(**main_args)
parser.add_argument("--generated_nt_dt_file", "-f", help="Generated nightime/daytime file")
parser.add_argument("--urbanpop_nt_dt_file", "-u", help="UrbanPop nightime/daytime file")
parser.add_argument("--schools_file", "-s", help="File containing schools data in CSV")
args = parser.parse_args(remaining_argv)
print(Fore.CYAN, "Options:", sep="")
for arg, value in args.__dict__.items():
    print(f"  {arg:20s} {value}")
print(Fore.RESET, end="")

nt_dt_df = pd.read_csv(args.generated_nt_dt_file)
up_nt_dt_df = pd.read_csv(args.urbanpop_nt_dt_file)

print("Correlations for worker flows (generated vs UrbanPop):")
for role in ["worker", "nope", "student", "all"]:
    if role == "all":
        gen_df = nt_dt_df.groupby(["orig_geoid", "dest_geoid"]).size().reset_index(name="count")
        up_df = up_nt_dt_df.groupby(["orig_geoid", "dest_geoid"]).size().reset_index(name="count")
    else:
        gen_df = nt_dt_df.loc[nt_dt_df["role"] == role].groupby(["orig_geoid", "dest_geoid"]).size().reset_index(name="count")
        up_df = (
            up_nt_dt_df.loc[up_nt_dt_df["role"] == role].groupby(["orig_geoid", "dest_geoid"]).size().reset_index(name="count")
        )
    gen_df["key"] = gen_df["orig_geoid"].astype(str) + "-" + gen_df["dest_geoid"].astype(str)
    up_df["key"] = up_df["orig_geoid"].astype(str) + "-" + up_df["dest_geoid"].astype(str)

    merged_df = gen_df.merge(up_df, on="key", how="outer", suffixes=["_gen", "_up"])[["key", "count_gen", "count_up"]]
    merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_up": "int"})
    merged_df.to_csv("nm_od_" + role + ".csv")
    corr = merged_df.count_gen.corr(merged_df.count_up)
    print(f"  {role}: {corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_up.sum()}")

gen_school_geoids_df = nt_dt_df.loc[nt_dt_df["role"] == "student"].groupby(["dest_geoid"]).size().reset_index(name="count")
up_school_geoids_df = up_nt_dt_df.loc[up_nt_dt_df["role"] == "student"].groupby(["dest_geoid"]).size().reset_index(name="count")
gen_school_geoids_df["key"] = gen_school_geoids_df["dest_geoid"].astype(str)
up_school_geoids_df["key"] = up_school_geoids_df["dest_geoid"].astype(str)

merged_df = gen_school_geoids_df.merge(up_school_geoids_df, on="key", how="outer", suffixes=["_gen", "_up"])[
    ["key", "count_gen", "count_up"]
]
merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_up": "int"})
corr = merged_df.count_gen.corr(merged_df.count_up)
print(
    f"Correlation of school populations per GEOID (generated vs UrbanPop):",
    f"{corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_up.sum()}",
)

ed_schools_df = pd.read_csv(args.schools_file, dtype={"id": "str"})
ed_school_geoids_df = ed_schools_df.groupby(["geoid"]).students.sum().reset_index(name="count")
ed_school_geoids_df["key"] = ed_school_geoids_df["geoid"].astype(str)
print("Correlations with school populations for GEOIDS:")
for compare_df in [(gen_school_geoids_df, "generated"), (up_school_geoids_df, "UrbanPop")]:
    merged_df = compare_df[0].merge(ed_school_geoids_df, on="key", how="inner", suffixes=["_gen", "_ed"])[
        ["key", "count_gen", "count_ed"]
    ]
    merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_ed": "int"})
    corr = merged_df.count_gen.corr(merged_df.count_ed)
    print(f"  {compare_df[1]}: {corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_ed.sum()}")
