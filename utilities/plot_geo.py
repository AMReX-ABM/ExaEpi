#!/usr/bin/env python

import pylab as plt
import re
import sys
import os
import numpy as np
import pandas as pd
import argparse
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mp
import yt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_agg_utils import aggregate_to_county  # noqa: E402


def _parse_day_from_plot_dir(plot_dir):
    """Extract the trailing step/day number from an ExaEpi plotfile directory name, e.g.
    'plt00050' or 'plt00050/' -> 50. Returns None if no trailing digits are found.
    """
    m = re.search(r"(\d+)$", plot_dir.rstrip("/"))
    return int(m.group(1)) if m else None


def load_exaepi_grid_stats(plot_dir, tract_level=False, county_level=False):
    """Read an ExaEpi AMReX plotfile directory and return a per-community DataFrame with columns:
    GEOID10, pop, never_infected, infected, immune -- aggregated up to the Census tract level if
    tract_level is set, or further to the county level if county_level is set (which takes
    precedence over tract_level if both are set).
    """
    print("Reading ExaEpi data from directory", plot_dir)
    ds = yt.load(plot_dir)  # type: ignore
    ad = ds.all_data()
    print(ds._field_list)
    grid_stats_df = pd.DataFrame(
        {
            "FIPS": ad["FIPS"],
            "Tract": ad["Tract"],
            "pop": ad["total"],
            "never_infected": ad["never_infected"],
            "infected": ad["infected"],
            "immune": ad["immune"],
            # "dead": ad["dead"],
        }
    )
    # FIPS is the first 5 digits of the block group code, and tract is the last 7. These need to be
    # combined to give the geoids found in the urbanpop file mapping geoids to lng/lat
    grid_stats_df = grid_stats_df[grid_stats_df.FIPS != -1].reset_index(drop=True)
    grid_stats_df["FIPS"] = grid_stats_df["FIPS"].astype("int").astype(str).str.zfill(5)
    grid_stats_df["Tract"] = grid_stats_df["Tract"].astype("int").astype(str).str.zfill(7)
    grid_stats_df["GEOID10"] = grid_stats_df["FIPS"] + grid_stats_df["Tract"]
    grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"].astype("int64")

    if county_level:
        grid_stats_df = aggregate_to_county(grid_stats_df)
    elif tract_level:
        # Drop the last digit (the block group number) to get the 11-digit Census tract GEOID,
        # then sum every block group that shares a tract into one row before merging with a
        # tract-level shapefile (otherwise each block group would re-attach to the same tract
        # geometry and inflate the per-tract counts).
        grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"] // 10
        grid_stats_df = grid_stats_df.groupby("GEOID10", as_index=False)[
            ["pop", "never_infected", "infected", "immune"]
        ].sum()
    return grid_stats_df


def main():
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 24

    parser = argparse.ArgumentParser(description="Plot UrbanPop ExaEpi outputs")
    # parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument(
        "--plot_dir",
        "-p",
        required=True,
        nargs="+",
        help="One or more plot directories (e.g. plt00000 plt00010 plt00020). A single directory "
        "plots one choropleth as before; multiple directories are plotted as a horizontal "
        "sequence, one panel per directory, each labeled by the day parsed from its name.",
    )
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census block group shape files (.shp), or Census tract shape files if --tract_level "
        "is passed. Available from\n"
        + "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups",
    )
    parser.add_argument(
        "--states_file",
        "-e",
        required=True,
        help="Shape file for US states",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="geo.pdf",
        help="Output file name for plot",
    )
    parser.add_argument(
        "--coord_bounds",
        "-b",
        default=[-170, -66.6, 18.5, 71.5],
        nargs="+",
        help="Range for longitude: min,max",
    )
    parser.add_argument(
        "--tract_level",
        "-t",
        action="store_true",
        default=False,
        help="Aggregate and plot at the Census tract level (11-digit GEOID, summed across all "
        "block groups in each tract) instead of the default block group level (12-digit GEOID10). "
        "Pass a Census tract shapefile (not a block group one) via --shape_files when using this. "
        "Ignored if --county_level is also passed.",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Aggregate and plot at the Census county level (5-digit GEOID, summed across all "
        "block groups in each county) instead of the default block group level. Pass a Census "
        "county shapefile via --shape_files when using this. Takes precedence over --tract_level.",
    )

    args = parser.parse_args()

    geo_unit = "county" if args.county_level else ("tract" if args.tract_level else "block group")
    geo_unit_pl = "counties" if args.county_level else ("tracts" if args.tract_level else "block groups")

    shp_dfs = []
    state_codes = []
    for fname in args.shape_files:
        if not fname.endswith(".shp"):
            print(
                "WARNING: file",
                fname,
                "passed with --shape_files does not appear to be a shapefile with .shp extension",
            )
            continue
        print("Reading data from", fname)
        shp_dfs.append(gp.read_file(fname))
        state_code = os.path.basename(fname).split("_")[2]
        state_codes.append(state_code)

    shp_data = pd.concat(shp_dfs)
    shp_data.GEOID10 = shp_data.GEOID10.astype("int64")
    print("Read in", len(shp_data), f"Census {geo_unit_pl}")

    states = gp.read_file(args.states_file)
    states = states[states.STATE.isin(state_codes)]
    max_count = 30000  # never_infected_agents["count"].max()

    example = {"county": "tl_2010_35_county10.shp", "tract": "tl_2010_35_tract10.shp", "block group": "tl_2010_35_bg10.shp"}[geo_unit]

    # Load and merge each plot_dir independently, so a sequence of several directories becomes a
    # horizontal row of panels (a single directory is just the N=1 case of the same loop).
    panels = []
    for plot_dir in args.plot_dir:
        grid_stats_df = load_exaepi_grid_stats(plot_dir, tract_level=args.tract_level, county_level=args.county_level)
        df = pd.merge(shp_data, grid_stats_df, on=["GEOID10"], how="inner")
        if df.empty:
            raise SystemExit(
                f"No rows matched after merging {plot_dir}: 0 of {len(grid_stats_df)} ExaEpi "
                f"{geo_unit} GEOIDs were found among the {len(shp_data)} shapefile rows. This "
                f"almost always means --shape_files is at the wrong granularity for the current "
                f"--tract_level/--county_level setting -- pass a Census {geo_unit} shapefile "
                f"(e.g. {example}) matching that setting."
            )
        day = _parse_day_from_plot_dir(plot_dir)
        label = f"Day {day}" if day is not None else os.path.basename(plot_dir.rstrip("/"))
        panels.append((df, label))
    # df.to_csv("merged.csv")
    # df[["GEOID10", "pop", "never_infected", "infected", "immune", "dead"]].to_csv("merged.csv")
    # df[["GEOID10", "pop", "never_infected", "infected", "immune"]].to_csv("merged.csv")

    # Bounds are the union across every panel's data, so the whole sequence shares one consistent
    # geographic extent instead of each panel framing itself differently.
    all_lon = pd.concat([df.INTPTLON10.astype("float") for df, _ in panels])
    all_lat = pd.concat([df.INTPTLAT10.astype("float") for df, _ in panels])
    xmin = max(float(args.coord_bounds[0]), float(all_lon.min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(all_lon.max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(all_lat.min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(all_lat.max()) + 0.5)
    yrange = ymax - ymin

    n = len(panels)
    panel_width = 12.0
    fig_x = panel_width * n
    fig_y = panel_width * yrange / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, axes = plt.subplots(1, n, figsize=(fig_x, fig_y), squeeze=False)
    axes = axes[0]

    for ax, (df, label) in zip(axes, panels):
        states.boundary.plot(ax=ax, lw=1, color="black")
        # Some decent colormaps: RdPu OrRd Greys
        df.plot(
            ax=ax,
            column="infected",
            cmap="OrRd",
            legend=True,
            norm=mp.colors.LogNorm(vmin=1.0, vmax=max_count),  # type: ignore
        )
        ax.set_title(label, fontsize=48)
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        ax.set_frame_on(False)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    axes_all = fig.get_axes()
    for cb in axes_all[n:-1]:
        cb.remove()
    axes_all[-1].set_box_aspect(50)
    # cb.set_frame_on(False)
    plt.tight_layout()
    print("Plotting results to", args.output)
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
