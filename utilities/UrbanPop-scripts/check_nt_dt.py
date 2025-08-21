#!/usr/bin/env -S python -u

from os.path import basename, splitext
import pandas as pd
import argparse
import configparser
from colorama import Fore
import matplotlib.pyplot as plt
import glob

import upop_to_exaepi


def compare_nt_dt_flows(gen_df, up_df):
    print("Correlations for night/day flows (generated vs UrbanPop):")
    for role in ["worker", "student", "all"]:
        if role == "all":
            role_gen_df = gen_df
            role_up_df = up_df
        elif role == "worker":
            role_gen_df = gen_df.loc[gen_df.naics != -1]
            role_up_df = up_df.loc[up_df.naics != -1]
        elif role == "student":
            role_gen_df = gen_df.loc[(gen_df.naics == -1) & (gen_df.school_id != 0)]
            role_up_df = up_df.loc[(up_df.naics == -1) & (up_df.school_id != 0)]
        else:
            raise RuntimeError("no option")
        role_gen_df = (
            role_gen_df.groupby(["home_geoid", "work_geoid"]).size().reset_index(name="count")
        )
        role_up_df = (
            role_up_df.groupby(["home_geoid", "work_geoid"]).size().reset_index(name="count")
        )

        role_gen_df["key"] = (
            role_gen_df["home_geoid"].astype(str) + "-" + role_gen_df["work_geoid"].astype(str)
        )
        role_up_df["key"] = (
            role_up_df["home_geoid"].astype(str) + "-" + role_up_df["work_geoid"].astype(str)
        )

        merged_df = role_gen_df.merge(role_up_df, on="key", how="outer", suffixes=["_gen", "_up"])[
            ["key", "count_gen", "count_up"]
        ]
        merged_df = merged_df.fillna(0)

        corr = merged_df.count_gen.corr(merged_df.count_up)
        print(f"  {role}: correlation {corr:.3f}")
        gen_sum = merged_df.count_gen.sum()
        up_sum = merged_df.count_up.sum()
        print(f"    Total flows - Generated: {gen_sum:.0f}" f", UrbanPop: {up_sum:.0f}")

        plt.figure(figsize=(10, 10))
        plt.scatter(merged_df.count_gen, merged_df.count_up, alpha=0.5)
        plt.xlabel("Generated Flow Counts")
        # plt.xscale("log")
        # plt.yscale("log")
        max_val = max(merged_df.count_gen.max(), merged_df.count_up.max())
        min_val = max(
            1e-6,
            min(
                merged_df.count_gen[merged_df.count_gen > 0].min(),
                merged_df.count_up[merged_df.count_up > 0].min(),
            ),
        )

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.xlabel("Generated Flow Counts")
        plt.ylabel("UrbanPop Flow Counts")
        plt.title(f"Flow Counts Comparison for {role}")
        # Add diagonal line for perfect correlation
        plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5)
        # Add correlation coefficient to plot
        plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f"Total Generated: {gen_sum:.0f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.85, f"Total UrbanPop: {up_sum:.0f}", transform=plt.gca().transAxes)
        plt.tight_layout()
        plot_file = f"flow_comparison_{role}.png"
        plt.savefig(plot_file)
        print(f"    Plot saved to {plot_file}")
        plt.close()


def compare_educational_flows(args, gen_df, up_df):
    gen_school_geoids_df = (
        gen_df.loc[(gen_df.school_id != 0) & (gen_df.naics == -1)]
        .groupby(["work_geoid"])
        .size()
        .reset_index(name="count")
    )
    up_schools_geoids_df = (
        up_df.loc[(gen_df.school_id != 0) & (gen_df.naics == -1)]
        .groupby(["work_geoid"])
        .size()
        .reset_index(name="count")
    )
    gen_school_geoids_df["key"] = gen_school_geoids_df["work_geoid"].astype(str)
    up_schools_geoids_df["key"] = up_schools_geoids_df["work_geoid"].astype(str)

    merged_df = gen_school_geoids_df.merge(
        up_schools_geoids_df, on="key", how="outer", suffixes=["_gen", "_up"]
    )[["key", "count_gen", "count_up"]]
    merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_up": "int"})
    corr = merged_df.count_gen.corr(merged_df.count_up)
    print(
        f"Correlation of school populations per GEOID (generated vs UrbanPop):",
        f"{corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_up.sum()}",
    )

    ed_schools_df = pd.read_csv(args.schools_file, dtype={"id": "str"})
    ed_school_geoids_df = ed_schools_df.groupby(["geoid"]).students.sum().reset_index(name="count")
    ed_school_geoids_df["key"] = ed_school_geoids_df["geoid"].astype(str)
    print(f'Correlations with school populations for GEOIDS using "{args.schools_file}":')
    for compare_df in [(gen_school_geoids_df, "generated"), (up_schools_geoids_df, "UrbanPop")]:
        merged_df = compare_df[0].merge(
            ed_school_geoids_df, on="key", how="left", suffixes=["_gen", "_ed"]
        )[["key", "count_gen", "count_ed"]]
        merged_df = merged_df.fillna(0).astype({"count_gen": "int", "count_ed": "int"})
        total_gen = merged_df["count_gen"].sum()
        total_ed = merged_df["count_ed"].sum()
        merged_df["count_gen_normalized"] = merged_df["count_gen"] / total_gen
        merged_df["count_ed_normalized"] = merged_df["count_ed"] / total_ed

        corr = merged_df.count_gen.corr(merged_df.count_ed)
        print(
            f"  {compare_df[1]}: {corr:.3f} {merged_df.count_gen.sum()} {merged_df.count_ed.sum()}"
        )

        plt.figure(figsize=(10, 10))
        plt.scatter(merged_df.count_gen_normalized, merged_df.count_ed_normalized, alpha=0.5)
        plt.xlabel(f"Normalized {compare_df[1]} School Population")
        plt.ylabel("Normalized Educational Dataset School Population")
        plt.title(f"School Population Comparison\n{compare_df[1]} vs Educational Dataset")
        # Add diagonal line for perfect correlation
        max_val = max(merged_df.count_gen_normalized.max(), merged_df.count_ed_normalized.max())
        min_val = max(
            1e-6,
            min(
                merged_df.count_gen_normalized[merged_df.count_gen_normalized > 0].min(),
                merged_df.count_ed_normalized[merged_df.count_ed_normalized > 0].min(),
            ),
        )
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        # Add correlation coefficient to plot
        plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes)
        plt.tight_layout()
        plot_file = f"school_population_comparison_{compare_df[1].lower()}.png"
        plt.savefig(plot_file)
        print(f"  Plot saved to {plot_file}")
        plt.close()


def compare_lodes_files(args, gen_df, up_df):
    lodes_df = upop_to_exaepi.get_lodes_groups(args.lodes_files)
    workers_gen_df = gen_df.loc[gen_df.naics != -1]
    workers_up_df = up_df.loc[up_df.naics != -1]
    for compare_df in [(workers_gen_df, "generated"), (workers_up_df, "UrbanPop")]:
        print(f"Checking flows for {compare_df[1]} workers")
        # Group and normalize worker flows
        flows_df = (
            compare_df[0].groupby(["home_geoid", "work_geoid"]).size().reset_index(name="count")
        )
        flows_df["key"] = (
            flows_df["home_geoid"].astype(str) + "-" + flows_df["work_geoid"].astype(str)
        )
        total_flows = flows_df["count"].sum()
        flows_df["count_normalized"] = flows_df["count"] / total_flows
        # Normalize LODES data
        lodes_comp_df = lodes_df.copy()
        lodes_comp_df["key"] = (
            lodes_comp_df["h_geocode"].astype(str) + "-" + lodes_comp_df["w_geocode"].astype(str)
        )
        total_lodes = lodes_comp_df["S000"].sum()
        lodes_comp_df["S000_normalized"] = lodes_comp_df["S000"] / total_lodes
        # Merge and compare normalized flows
        merged_df = flows_df.merge(
            lodes_comp_df[["key", "S000", "S000_normalized"]], on="key", how="outer"
        )
        merged_df = merged_df.fillna(0)
        # Calculate correlations for both raw and normalized counts
        corr_raw = merged_df["count"].corr(merged_df["S000"])
        corr_norm = merged_df["count_normalized"].corr(merged_df["S000_normalized"])
        print(
            f"  Correlation: {corr_raw:.3f} (flows: {merged_df['count'].sum():.0f}, "
            f"LODES: {merged_df['S000'].sum():.0f})"
        )
        # Create correlation plot with normalized values
        plt.figure(figsize=(10, 10))
        plt.scatter(merged_df["count_normalized"], merged_df["S000_normalized"], alpha=0.5)
        # Set log scales and limits
        # plt.xscale("log")
        # plt.yscale("log")
        max_val = max(merged_df["count_normalized"].max(), merged_df["S000_normalized"].max())
        min_val = min(merged_df["count_normalized"].min(), merged_df["S000_normalized"].min())
        min_val = max(min_val, 1e-6)  # Set minimum value to avoid log(0)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        # Labels and title
        plt.xlabel(f"Normalized {compare_df[1]} Worker Flow (%)")
        plt.ylabel("Normalized LODES Flow (%)")
        plt.title(f"Worker Flow Comparison\n{compare_df[1]} vs LODES")
        # Add diagonal line for perfect correlation
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
        # Add correlation coefficients to plot
        plt.text(0.05, 0.88, f"Correlation: {corr_norm:.3f}", transform=plt.gca().transAxes)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plot_file = f"worker_flows_comparison_{compare_df[1].lower()}_vs_lodes.png"
        plt.savefig(plot_file)
        print(f"  Plot saved to {plot_file}")
        plt.close()


def main():
    cfg_parser = argparse.ArgumentParser(
        description="Check generated daytime/nighttime flows and schools data against UrbanPop nt/dt",
        add_help=False,
    )
    cfg_parser.add_argument("-c", "--config", help="Config file", metavar="FILE")
    args, remaining_argv = cfg_parser.parse_known_args()
    main_args = {
        "gen_file": "",
        "up_file": "",
        "schools_file": "",
        "lodes_files": "",
        "upop_files": "",
    }
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read([args.config])
        main_args.update(dict(cfg.items("main")))
        for files_label in ["lodes_files", "upop_files"]:
            file_list = []
            for f in main_args[files_label].split():
                file_list.extend(glob.glob(f))
            main_args[files_label] = file_list  # type: ignore
    parser = argparse.ArgumentParser(parents=[cfg_parser])
    parser.set_defaults(**main_args)
    parser.add_argument("--gen_file", "-g", help="UrbanPop file with generated day/night")
    parser.add_argument("--up_file", "-u", help="UrbanPop file with UrbanPop day/night")
    parser.add_argument("--schools_file", "-s", help="File containing schools data in CSV")
    parser.add_argument(
        "--lodes_files", "-l", nargs="+", help="LODES7 origin-destination (OD) files in CSV format"
    )
    parser.add_argument("--upop_files", "-f", nargs="+", help="UrbanPop files in Feather format")
    args = parser.parse_args(remaining_argv)
    print(Fore.CYAN, "Options:", sep="")
    for arg, value in args.__dict__.items():
        print(f"  {arg:20s} {value}")
    print(Fore.RESET, end="")

    gen_df = pd.read_csv(args.gen_file)
    up_df = pd.read_csv(args.up_file)

    common_pids = set(gen_df["id"]).intersection(set(up_df["id"]))
    gen_df = gen_df[gen_df["id"].isin(common_pids)]
    up_df = up_df[up_df["id"].isin(common_pids)]
    print(f"Found {len(common_pids)} common ids between datasets")
    if args.upop_files != "" and args.upop_files != []:
        upop_df = upop_to_exaepi.load_upop_feather_files(args.upop_files)
        upop_df = upop_df[upop_df["p_id"].isin(common_pids)]
        upop_file = "urbanpop.common"
        upop_df.to_feather(upop_file + ".feather")
        upop_df.to_csv(upop_file + ".csv", index=False)
        print(f"Common UrbanPop data saved to {upop_file}")

    compare_nt_dt_flows(gen_df, up_df)
    compare_educational_flows(args, gen_df, up_df)
    compare_lodes_files(args, gen_df, up_df)


if __name__ == "__main__":
    main()
