#!/usr/bin/env python

"""Plot ExaEpi community size (population per community) against population density (people /
km^2) for one ExaEpi plotfile directory.

Reuses load_exaepi_grid_stats (plot_geo.py) at its default, finest granularity to get each
community's population ("pop" -- a grid cell in the AMReX plotfile, capped at CensusData's
2000-person COMMUNITY_SIZE except for the last, partial community in each Census unit). Density is
computed from the same Census block group shapefiles plot_geo.py takes via --shape_files
(GEOID10 + ALAND10 land area): when a Census unit's population exceeds 2000, ExaEpi splits it into
several communities that all share the same GEOID10 and therefore the same land area, so this is a
many-communities-to-one-area join, not one-to-one.

Only Census-initialized runs (ic_type=Census) are supported, same as plot_geo.py -- the plotfile's
FIPS/Tract fields this depends on aren't populated for UrbanPop runs. This is a single run's static
community layout, not a time series: population and land area don't change day to day, so any one
plot directory from a run gives the same answer as any other.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_grid_stats  # noqa: E402

SQ_M_PER_SQ_KM = 1_000_000.0
SERIES_COLOR = "#2a78d6"
TREND_COLOR = "#eb6834"
MIN_BIN_COUNT = 5  # minimum communities in a density bin before its median is plotted


def load_community_density(plot_dir, shape_files):
    """Return a DataFrame with one row per ExaEpi community: GEOID10, pop (community size),
    area_km2, and density (pop / area_km2)."""
    grid_stats_df = load_exaepi_grid_stats(plot_dir)

    shp_dfs = []
    for fname in shape_files:
        if not fname.endswith(".shp"):
            print(f"WARNING: {fname} passed with --shape_files does not appear to be a shapefile with .shp extension")
            continue
        print("Reading data from", fname)
        shp_dfs.append(gp.read_file(fname))
    shp_data = pd.concat(shp_dfs, ignore_index=True)
    shp_data["GEOID10"] = shp_data["GEOID10"].astype("int64")
    shp_data["area_km2"] = shp_data["ALAND10"].astype("float64") / SQ_M_PER_SQ_KM

    df = pd.merge(grid_stats_df, shp_data[["GEOID10", "area_km2"]], on="GEOID10", how="inner")
    if df.empty:
        raise SystemExit(
            f"No rows matched after merging: 0 of {len(grid_stats_df)} ExaEpi community GEOIDs "
            f"were found among the {len(shp_data)} shapefile rows. Pass the Census block group "
            "shapefile(s) (tl_2010_NN_bg10.shp) matching this run's state(s)."
        )
    # A handful of block groups are entirely water (airports, reservoirs) with recorded land area
    # of zero; density there is undefined, not zero, so they must be dropped rather than
    # divide-by-zero.
    df = df[df["area_km2"] > 0].copy()
    df["density"] = df["pop"] / df["area_km2"]
    return df


def plot_community_size_vs_density(df, output):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["density"], df["pop"], s=10, alpha=0.35, color=SERIES_COLOR, linewidths=0, label="Community")

    # Median community size within log-spaced density bins -- shows the trend through the
    # scatter's heavy overplotting rather than relying on the eye to average it.
    log_density = np.log10(df["density"])
    bins = np.linspace(log_density.min(), log_density.max(), 25)
    bin_idx = np.digitize(log_density, bins)
    bin_centers, bin_medians = [], []
    for i in range(1, len(bins)):
        sel = df["pop"][bin_idx == i]
        if len(sel) >= MIN_BIN_COUNT:
            bin_centers.append(10 ** ((bins[i - 1] + bins[i]) / 2))
            bin_medians.append(sel.median())
    ax.plot(bin_centers, bin_medians, color=TREND_COLOR, lw=2, label="Median (binned)")

    ax.set_xscale("log")
    ax.set_xlabel("Population density (people / km²)")
    ax.set_ylabel("Community size (population)")
    ax.set_title("ExaEpi community size vs. population density")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    correlation = np.corrcoef(log_density, df["pop"])[0, 1]
    print(f"{len(df)} communities plotted")
    print(f"Pearson correlation (log10 density, community size): {correlation:.3f}")

    plt.tight_layout()
    print("Plotting results to", output)
    plt.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot_dir", "-p", required=True, help="One ExaEpi plotfile directory (e.g. plt00000)")
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census block group shape files (.shp), same as plot_geo.py's --shape_files. Available from "
        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups",
    )
    parser.add_argument("--output", "-o", default="community_size_vs_density.png", help="Output plot file name")
    args = parser.parse_args()

    df = load_community_density(args.plot_dir, args.shape_files)
    plot_community_size_vs_density(df, args.output)


if __name__ == "__main__":
    main()
