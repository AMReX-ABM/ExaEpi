#!/usr/bin/env python

"""Plot ExaEpi community size (population per community) against population density (people /
km^2) for one run's day/night population CSV, as two overlaid series: nighttime (home-based) and
daytime (work-based) population.

Each ExaEpi community is one AMReX grid cell. Nighttime population counts agents by home cell;
daytime population counts the same agents by work/school cell instead -- agents with no separate
work/school location (retirees, preschoolers, ...) contribute equally to both. This is the same
home/work split plot_geo_daynight.py uses; see its docstring for why it's an exact, unbiased
census of every agent rather than a partial sample.

Density is computed from the same Census block group shapefiles (GEOID10 + ALAND10 land area,
via --shape_files): when a Census unit's population exceeds 2000, ExaEpi splits it into several
communities (grid cells) that all share the same GEOID10 and therefore the same land area, so
this is a many-communities-to-one-area join, not one-to-one.

Population comes from the static, once-per-run <prefix>_day_night_population.csv ExaEpi writes
when --aggregated_diag_int is enabled (see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp,
and plot_geo.load_exaepi_day_night_population) -- already per-community, so no plotfile/agent data
is needed at all.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_day_night_population  # noqa: E402
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402

SQ_M_PER_SQ_KM = 1_000_000.0
MIN_BIN_COUNT = 5  # minimum communities in a density bin before its median is plotted

# Per series: scatter color, trend-line color, and label. Labels are kept short -- the
# "(home)"/"(work)" clarification belongs in the caption -- since this figure is only ~3.1in wide
# in the paper, leaving no room for a longer legend at PLOS's 8-12pt font floor.
SERIES_STYLE = {
    "night": {"scatter": "#2a78d6", "trend": "#0b3d78", "label": "Nighttime"},
    "day": {"scatter": "#eb9134", "trend": "#a83214", "label": "Daytime"},
}


def load_community_density(day_night_csv, shape_files):
    """Return a DataFrame with one row per ExaEpi community: GEOID10, night_pop, day_pop,
    area_km2, density_night, density_day."""
    df = load_exaepi_day_night_population(day_night_csv)[["GEOID10", "night_pop", "day_pop"]]
    n_communities = len(df)

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

    df = pd.merge(df, shp_data[["GEOID10", "area_km2"]], on="GEOID10", how="inner")
    if df.empty:
        raise SystemExit(
            f"No rows matched after merging: 0 of {n_communities} ExaEpi community "
            f"GEOIDs were found among the {len(shp_data)} shapefile rows. Pass the Census block group "
            "shapefile(s) (tl_2010_NN_bg10.shp) matching this run's state(s)."
        )
    # A handful of block groups are entirely water (airports, reservoirs) with recorded land area
    # of zero; density there is undefined, not zero, so they must be dropped rather than
    # divide-by-zero.
    df = df[df["area_km2"] > 0].copy()
    df["density_night"] = df["night_pop"] / df["area_km2"]
    df["density_day"] = df["day_pop"] / df["area_km2"]
    return df


def _add_series(ax, df, pop_col, density_col, style, log):
    """Scatter one (population, density) series plus its own binned-median trend line and fitted
    line, and return (scatter_handle, line_handle, fit_handle, corr, n, (slope, intercept, r)) --
    corr/n are the Pearson correlation (log10 density, population) and community count, and
    slope/intercept/r describe the fitted line, all for the printed summary. The caller lays the
    three handles out as a three-column legend (marker+name, line+"median", dashes+"fit") so each
    series is one row: see plot_community_size_vs_density."""
    sub = df[df[pop_col] > 0]
    scatter_handle = ax.scatter(sub[pop_col], sub[density_col], s=12, alpha=0.25, color=style["scatter"],
                                 marker="+", linewidths=0.8, label=style["label"])

    # Median community size within density bins -- shows the trend through the scatter's heavy
    # overplotting rather than relying on the eye to average it. Bin edges are spaced in
    # whichever space matches the displayed axis: log10(density) for --log (equal *ratio* bins,
    # matching a log-scaled axis), raw density otherwise (equal-width bins, matching a linear
    # one) -- the wrong choice would bunch most bins at one end of the axis actually shown.
    density = np.log10(sub[density_col]) if log else sub[density_col]
    bins = np.linspace(density.min(), density.max(), 25)
    bin_idx = np.digitize(density, bins)
    bin_centers, bin_medians = [], []
    for i in range(1, len(bins)):
        sel = sub[pop_col][bin_idx == i]
        if len(sel) >= MIN_BIN_COUNT:
            center = (bins[i - 1] + bins[i]) / 2
            bin_centers.append(10 ** center if log else center)
            bin_medians.append(sel.median())
    line_handle, = ax.plot(bin_medians, bin_centers, color=style["trend"], lw=1.5, label="median")

    # Least-squares line fit (density on community size) in the displayed axis space, so it plots
    # as a straight line: a power law density = 10^intercept * size^slope for --log, a plain
    # linear fit otherwise. r is the Pearson correlation in that same space.
    x = np.log10(sub[pop_col]) if log else sub[pop_col]
    y = np.log10(sub[density_col]) if log else sub[density_col]
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fit_x = np.linspace(x.min(), x.max(), 100)
    fit_y = intercept + slope * fit_x
    fit_handle, = ax.plot(10 ** fit_x if log else fit_x, 10 ** fit_y if log else fit_y, color=style["trend"],
                          lw=1.5, ls="--", label="fit")

    log_density = np.log10(sub[density_col])
    corr = np.corrcoef(log_density, sub[pop_col])[0, 1]
    return scatter_handle, line_handle, fit_handle, corr, len(sub), (slope, intercept, r)


def plot_community_size_vs_density(df, output, log=False):
    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")

    scatter_handles, line_handles, fit_handles = [], [], []
    for series, pop_col, density_col in (("night", "night_pop", "density_night"),
                                          ("day", "day_pop", "density_day")):
        style = SERIES_STYLE[series]
        scatter_handle, line_handle, fit_handle, corr, n, (slope, intercept, r) = _add_series(
            ax, df, pop_col, density_col, style, log)
        scatter_handles.append(scatter_handle)
        line_handles.append(line_handle)
        fit_handles.append(fit_handle)
        print(f"{n} communities plotted ({series})")
        print(f"Pearson correlation (log10 density, community size), {series}: {corr:.3f}")
        if log:
            print(f"Fit, {series}: log10(density) = {intercept:.4f} + {slope:.4f} * log10(size)  "
                  f"[density = {10 ** intercept:.4g} * size^{slope:.4f}], r = {r:.3f}, r^2 = {r * r:.3f}")
        else:
            print(f"Fit, {series}: density = {intercept:.4g} + {slope:.4g} * size, r = {r:.3f}, r^2 = {r * r:.3f}")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Community size (population)")
    ax.set_ylabel("Population density (people / km²)")
    # Three-column legend, one row per series: marker + series name in column 1, solid line +
    # "median" in column 2, dashed line + "fit" in column 3 (matplotlib fills a multi-column
    # legend's handles column-major, so listing all scatter handles, then all median handles, then
    # all fit handles lines up each series' entries on the same row).
    ax.legend(handles=scatter_handles + line_handles + fit_handles, ncol=3, columnspacing=0.8, handletextpad=0.5,
              frameon=False, loc="upper left")
    ax.set_ylim(0.01, 1000000)
    print("Plotting results to", output)
    plt.savefig(output)


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--day_night_csv", "-n", required=True,
        help="ExaEpi's <prefix>_day_night_population.csv (written when --aggregated_diag_int is "
        "enabled -- see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp)",
    )
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census block group shape files (.shp), same as plot_geo.py's --shape_files. Available from "
        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups",
    )
    parser.add_argument("--output", "-o", default="community_size_vs_density.png", help="Output plot file name")
    parser.add_argument(
        "--log", action="store_true", help="Use log-log axes instead of the default linear-linear"
    )
    args = parser.parse_args()

    df = load_community_density(args.day_night_csv, args.shape_files)
    plot_community_size_vs_density(df, args.output, log=args.log)


if __name__ == "__main__":
    main()
