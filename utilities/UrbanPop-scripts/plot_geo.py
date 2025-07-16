#!/usr/bin/env python

import pylab as plt
import sys
import os
import numpy as np
import pandas as pd
import argparse
from yt.frontends.boxlib.api import AMReXDataset
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mp


def main():
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 24

    parser = argparse.ArgumentParser(description="Plot UrbanPop ExaEpi outputs")
    # parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--plot_dir", "-p", required=True, help="Plot directory")
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Shape files census block group shape files (.shp). Available from\n"
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
    parser.add_argument("--coord_bounds", "-b", default=[-170, -66.6, 18.5, 71.5], nargs="+", help="Range for longitude: min,max")

    args = parser.parse_args()

    print("Reading ExaEpi data from directory", args.plot_dir)
    ds = AMReXDataset(args.plot_dir)
    # print(ds.field_list)
    ad = ds.all_data()
    grid_stats_df = pd.DataFrame(
        {
            "FIPS": ad["FIPS"],
            "Tract": ad["Tract"],
            "pop": ad["total"],
            "never_infected": ad["never_infected"],
            "infected": ad["infected"],
            "immune": ad["immune"],
            "dead": ad["dead"],
        }
    )
    # FIPS is the first 5 digits of the block group code, and tract is the last 7. These need to be combined to give the geoids
    # found in the urbanpop file mapping geoids to lng/lat
    grid_stats_df = grid_stats_df[grid_stats_df.FIPS != -1].reset_index(drop=True)
    grid_stats_df.FIPS = grid_stats_df.FIPS.astype("int").astype(str).str.zfill(5)
    grid_stats_df.Tract = grid_stats_df.Tract.astype("int").astype(str).str.zfill(7)
    grid_stats_df["GEOID10"] = grid_stats_df.FIPS + grid_stats_df.Tract
    grid_stats_df.GEOID10 = grid_stats_df.GEOID10.astype("int64")
    grid_stats_df.to_csv("grid_stats.csv")

    shp_dfs = []
    state_codes = []
    for fname in args.shape_files:
        if not fname.endswith(".shp"):
            print("WARNING: file", fname, "passed with --shape_files does not appear to be a shapefile with .shp extension")
            continue
        print("Reading data from", fname)
        shp_dfs.append(gp.read_file(fname))
        state_code = os.path.basename(fname).split("_")[2]
        state_codes.append(state_code)

    shp_data = pd.concat(shp_dfs)
    shp_data.GEOID10 = shp_data.GEOID10.astype("int64")
    print("Read in", len(shp_data), "Census block groups")

    states = gp.read_file(args.states_file)
    states = states[states.STATE.isin(state_codes)]
    max_count = 30000  # never_infected_agents["count"].max()

    df = pd.merge(shp_data, grid_stats_df, on=["GEOID10"], how="inner")
    df.to_csv("merged.csv")
    df[["GEOID10", "pop", "never_infected", "infected", "immune", "dead"]].to_csv("merged.csv")
    xmin = max(float(args.coord_bounds[0]), float(df.INTPTLON10.astype("float").min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(df.INTPTLON10.astype("float").max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(df.INTPTLAT10.astype("float").min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(df.INTPTLAT10.astype("float").max()) + 0.5)
    yrange = ymax - ymin

    fig_x = 32.0
    fig_y = float(fig_x) * yrange / 1.8 / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    # _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(32, 32))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_x, fig_y))

    status_list = {
        # 0: ["never_infected", ax1, "Blues"],
        1: ["infected", ax1, "OrRd"],
        # 2: ["immune", ax3, "Greens"],
        # 3: ["susceptible", ax4, "OrRd"],
        4: ["dead", ax2, "OrRd"],
    }

    for _, status in status_list.items():
        ax = status[1]
        states.boundary.plot(ax=ax, lw=1, color="black")
        # Some decent colormaps: RdPu OrRd Greys
        df.plot(ax=ax, column=status[0], cmap=status[2], legend=True, norm=mp.colors.LogNorm(vmin=1.0, vmax=max_count))
        # ax.set_xlim([ds.domain_left_edge[0], ds.domain_right_edge[0]])
        # ax.set_ylim([ds.domain_left_edge[1], ds.domain_right_edge[1]])
        ax.set_title(status[0].upper())
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        ax.set_frame_on(False)
        # ax.set_xlim([max(-170.0, xmin), min(-57.0, xmax)])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    axes = fig.get_axes()
    for cb in axes[len(status_list) : -1]:
        cb.remove()
    cb = axes[-1]
    cb.set_box_aspect(50)
    # cb.set_frame_on(False)
    plt.tight_layout()
    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
