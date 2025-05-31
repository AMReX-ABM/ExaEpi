#!/usr/bin/env python

import pandas as pd

nt_dt_df = pd.read_csv("nm_nt_dt.csv")
up_nt_dt_df = pd.read_csv("up_nm_nt_dt.csv")

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

ed_schools_df = pd.read_csv("../../EducationData/schools_with_geoids.hifld.csv", dtype={"id": "str"})
ed_school_geoids_df = ed_schools_df.groupby(["geoid"]).students.sum().reset_index(name="count")
ed_school_geoids_df["key"] = ed_school_geoids_df["geoid"].astype(str)
print("Correlations with HIFLD school populations for GEOIDS:")
for compare_df in [(gen_school_geoids_df, "generated"), (up_school_geoids_df, "UrbanPop")]:
    merged_df = compare_df[0].merge(ed_school_geoids_df, on="key", how="inner", suffixes=["_gen", "_ed"])[
        ["key", "count_gen", "count_ed"]
    ]
    merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_ed": "int"})
    corr = merged_df.count_gen.corr(merged_df.count_ed)
    print(f"  {compare_df[1]}: {corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_ed.sum()}")

ed_schools_df = pd.read_csv("../../EducationData/schools_with_geoids.nces.csv", dtype={"id": "str"})
ed_school_geoids_df = ed_schools_df.groupby(["geoid"]).students.sum().reset_index(name="count")
ed_school_geoids_df["key"] = ed_school_geoids_df["geoid"].astype(str)
print("Correlations with NCES school populations for GEOIDS:")
for compare_df in [(gen_school_geoids_df, "generated"), (up_school_geoids_df, "UrbanPop")]:
    merged_df = compare_df[0].merge(ed_school_geoids_df, on="key", how="inner", suffixes=["_gen", "_ed"])[
        ["key", "count_gen", "count_ed"]
    ]
    merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_ed": "int"})
    corr = merged_df.count_gen.corr(merged_df.count_ed)
    print(f"  {compare_df[1]}: {corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_ed.sum()}")
