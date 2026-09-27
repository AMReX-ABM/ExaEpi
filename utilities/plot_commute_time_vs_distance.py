#!/usr/bin/env python

"""Plot ExaEpi's assigned home-to-work distance against each commuter's own reported commute time,
to check whether the workplaces the UrbanPop pipeline assigns are ones the agents could plausibly
reach every day.

Every UrbanPop person carries pr_commute, the one-way travel time to work in minutes from the ACS
PUMS record it was synthesized from (the ACS asks about the commute the person actually made the
previous week), but the workplace ExaEpi uses is assigned separately, from LODES flows, and that
assignment doesn't see pr_commute. This plots, for each reported-time bin, the median and 10-90%
range of the assigned straight-line distance between the home and work block groups' internal
points, with a reference line for the distance reachable at a given straight-line speed.

pr_commute is not carried into the UrbanPop .bin, so it is read from the pipeline's
<name>.upop.intermediate.csv and joined on id: upop_to_exaepi.py's adjust_indexes() numbers agents
0..N-1 in sorted p_id order, so agent id i is the i-th row in p_id order. The join is checked
against the home block group and age of every agent, and refused if either doesn't match (e.g. the
intermediate file is from a different run of the pipeline than the .bin).

Commuters are chosen by the same rule plot_commute_distance.py uses (see exaepi_commuters());
those with no reported time (pr_commute <= 0) are left out.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import polars as pl
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402
from plot_commute_distance import (  # noqa: E402
    read_urbanpop_columns,
    exaepi_commuters,
    load_internal_points,
    haversine_km,
)

# Reported one-way commute time bins (minutes, right-closed). ACS travel times are mostly reported
# in multiples of 5 minutes, so 5-minute bins up to an hour, then wider ones for the sparse tail.
DEFAULT_TIME_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 90, 120, 160]

# Floor for the log distance axis: same-block-group commutes have distance 0.
MIN_DISTANCE_KM = 0.1


def load_commuter_times_and_distances(urbanpop_bin, intermediate_csv, blockgroup_files):
    """Return a DataFrame (minutes, km) with one row per ExaEpi commuter who has a reported
    commute time: minutes is pr_commute, km the straight-line home-to-work block group distance."""
    cols = read_urbanpop_columns(
        urbanpop_bin, ["id", "home_geoid", "work_geoid", "age", "naics", "school_id", "travel"])
    order = np.argsort(cols["id"])
    cols = {name: col[order] for name, col in cols.items()}
    if not np.array_equal(cols["id"], np.arange(len(order))):
        sys.exit(f"error: {urbanpop_bin}'s agent ids are not 0..N-1, so they can't be joined on p_id order")

    print("Reading reported commute times from", intermediate_csv)
    upop = (pl.read_csv(intermediate_csv, columns=["p_id", "geoid", "pr_age", "pr_commute"],
                        schema_overrides={"p_id": pl.Utf8, "geoid": pl.Int64})
            .sort("p_id"))
    if len(upop) != len(order):
        sys.exit(f"error: {intermediate_csv} has {len(upop):,} people but {urbanpop_bin} has "
                 f"{len(order):,} agents -- they are not from the same pipeline run")
    home_ok = upop["geoid"].to_numpy() == cols["home_geoid"]
    age_ok = upop["pr_age"].round().cast(pl.Int64).to_numpy() == cols["age"]
    if not (home_ok.all() and age_ok.all()):
        sys.exit(f"error: joining on p_id order matches home block group for {100 * home_ok.mean():.2f}% "
                 f"and age for {100 * age_ok.mean():.2f}% of agents -- {intermediate_csv} is not from "
                 f"the same pipeline run as {urbanpop_bin}")
    minutes = upop["pr_commute"].to_numpy()

    commuter = exaepi_commuters(cols)
    timed = commuter & (minutes > 0)
    print(f"{int(timed.sum()):,} commuters have a reported commute time "
          f"({int((commuter & ~timed).sum()):,} don't and are left out)")

    points = load_internal_points(blockgroup_files)
    home = points.reindex(cols["home_geoid"][timed])
    work = points.reindex(cols["work_geoid"][timed])
    km = haversine_km(home["lat"].to_numpy(), home["lon"].to_numpy(), work["lat"].to_numpy(), work["lon"].to_numpy())
    km[cols["home_geoid"][timed] == cols["work_geoid"][timed]] = 0.0
    df = pd.DataFrame({"minutes": minutes[timed], "km": km})
    missing = df["km"].isna()
    if missing.any():
        print(f"WARNING: dropped {int(missing.sum()):,} commuters whose home or work block group has no internal point")
    return df[~missing]


def summarize(df, time_edges, ref_speed):
    """Per reported-time bin: count, 10th/50th/90th percentile of assigned distance, and the
    fraction whose assigned distance would need a straight-line speed above ref_speed."""
    df = df.assign(bin=pd.cut(df["minutes"], time_edges),
                   too_fast=df["km"] > ref_speed * df["minutes"] / 60)
    g = df.groupby("bin", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "p10": g["km"].quantile(0.1),
        "median": g["km"].median(),
        "p90": g["km"].quantile(0.9),
        "too_fast": g["too_fast"].mean(),
    })
    out["mid"] = [(b.left + b.right) / 2 for b in out.index]
    return out


def main():
    apply_style()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urbanpop", "-u", required=True,
                        help="The UrbanPop .bin the ExaEpi run reads, e.g. data/UrbanPop/urbanpop_nm.bin")
    parser.add_argument("--intermediate", "-i", required=True,
                        help="The same pipeline run's <name>.upop.intermediate.csv, which still has pr_commute, "
                        "e.g. data/UrbanPop/urbanpop_nm.upop.intermediate.csv")
    parser.add_argument("--blockgroups", "-g", required=True, nargs="+",
                        help="2010 Census BLOCK GROUP shapefile(s) for the state, e.g. "
                        "data/US_2010_Census_BlockGroups/tl_2010_35_bg10.shp")
    parser.add_argument("--ref_speed", type=float, default=60,
                        help="Straight-line speed (km/h) for the reference line, and for the printed "
                        "fraction of commuters who would have to beat it. Default: 60")
    parser.add_argument("--title", default="Assigned distance vs. reported time", help="Plot title ('' for none)")
    parser.add_argument("--output", "-o", default="commute_time_vs_distance.pdf", help="Output file name for plot")
    args = parser.parse_args()

    df = load_commuter_times_and_distances(args.urbanpop, args.intermediate, args.blockgroups)
    summary = summarize(df, DEFAULT_TIME_EDGES, args.ref_speed)

    too_fast = (df["km"] > args.ref_speed * df["minutes"] / 60).mean()
    print(f"Overall: median time {df['minutes'].median():.0f} min, median distance {df['km'].median():.1f} km; "
          f"{100 * too_fast:.1f}% would need a straight-line speed above {args.ref_speed:g} km/h, "
          f"{100 * (df['km'] > 100).mean():.1f}% are assigned over 100 km")
    print(summary[["n", "p10", "median", "p90", "too_fast"]].round(2).to_string())

    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    x = summary["mid"].to_numpy()
    ax.fill_between(x, np.maximum(summary["p10"], MIN_DISTANCE_KM), summary["p90"], color="red", alpha=0.2,
                    linewidth=0, label="ExaEpi, 10–90%")
    ax.plot(x, np.maximum(summary["median"], MIN_DISTANCE_KM), color="red", linewidth=1, marker="o", markersize=3,
            label="ExaEpi, median")
    t = np.array([DEFAULT_TIME_EDGES[0], DEFAULT_TIME_EDGES[-1]], dtype=float)
    ax.plot(t, np.maximum(args.ref_speed * t / 60, MIN_DISTANCE_KM), color="black", linestyle="--", linewidth=0.7,
            label=f"{args.ref_speed:g} km/h straight line")
    ax.set_yscale("log")
    ax.set_xlim(DEFAULT_TIME_EDGES[0], DEFAULT_TIME_EDGES[-1])
    ax.set_xlabel("Reported one-way commute time (min)")
    ax.set_ylabel("Assigned distance (km)")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5)

    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
