#!/usr/bin/env python

"""Plot a 3-panel ExaEpi vs Epicast infection comparison for one matched day: two absolute
"infected" choropleths (same OrRd/log-scale style as plot_geo.py and plot_geo_epicast.py) side by
side, plus a third panel showing the log2 ratio between them -- so the same figure shows both
current infection levels and how the two models diverge at that point in the epidemic.

ExaEpi and Epicast are offset in time (see compare_to_epicast.py's -s auto), so the day encoded in
--plot_dir (e.g. plt00050 -> ExaEpi day 50) and the day passed via --epicast_day are independent
parameters -- the caller picks a matching pair, e.g. from a prior compare_to_epicast.py -s auto run.
Both inputs are compared at Census tract granularity by default (ExaEpi's block groups are summed
up to their tract, same as plot_geo.py's --tract_level, and Epicast is natively tract-level), since
that's the finest resolution Epicast's data supports. Pass --county_level to aggregate both sides
further, up to the county (with a county shapefile via --shape_files instead of a tract one).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_grid_stats  # noqa: E402
from plot_geo_epicast import load_epicast_grid_stats  # noqa: E402
from geo_agg_utils import aggregate_to_county  # noqa: E402


def main():
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 24

    parser = argparse.ArgumentParser(
        description="Plot a 3-panel ExaEpi vs Epicast infection comparison (absolute, absolute, divergence)"
    )
    parser.add_argument("--plot_dir", "-p", required=True, help="ExaEpi plotfile directory (e.g. plt00050)")
    parser.add_argument("--events_file", "-c", required=True, help="Epicast run.events.bin file")
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census TRACT shape files (.shp) by default, or Census COUNTY shape files if "
        "--county_level is passed. Available from\n"
        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Census+Tracts",
    )
    parser.add_argument("--states_file", "-e", required=True, help="Shape file for US states")
    parser.add_argument("--output", "-o", default="geo_compare.pdf", help="Output file name for plot")
    parser.add_argument(
        "--coord_bounds",
        "-b",
        default=[-170, -66.6, 18.5, 71.5],
        nargs="+",
        help="Range for longitude: min,max",
    )
    parser.add_argument(
        "--epicast_day",
        "-d",
        type=int,
        default=None,
        help="0-based Epicast day to compare against --plot_dir's ExaEpi snapshot (default: the "
        "last day in --events_file). Pick this to match --plot_dir's day using whatever "
        "ExaEpi/Epicast shift applies (see compare_to_epicast.py -s auto).",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Aggregate and plot at the Census county level (5-digit GEOID, summed across all "
        "tracts in each county) instead of the default tract level. Pass a Census county shapefile "
        "(not a tract one) via --shape_files when using this.",
    )

    args = parser.parse_args()

    exaepi_df = load_exaepi_grid_stats(args.plot_dir, tract_level=True)
    epicast_df, resolved_epicast_day = load_epicast_grid_stats(args.events_file, day=args.epicast_day)

    geo_unit = "county" if args.county_level else "tract"
    geo_unit_pl = "counties" if args.county_level else "tracts"
    if args.county_level:
        exaepi_df = aggregate_to_county(exaepi_df)
        epicast_df = aggregate_to_county(epicast_df)

    df = pd.merge(exaepi_df, epicast_df, on="GEOID10", how="inner", suffixes=("_exaepi", "_epicast"))
    if df.empty:
        raise SystemExit(
            f"No rows matched merging ExaEpi ({len(exaepi_df)} {geo_unit_pl}) with Epicast "
            f"({len(epicast_df)} {geo_unit_pl}) on GEOID10 -- check both cover the same state/region."
        )
    print(f"Matched {len(df)} {geo_unit_pl} between ExaEpi and Epicast (of {len(exaepi_df)}/{len(epicast_df)})")

    # log2 ratio of currently-infected counts (+1 pseudocount to avoid log(0)): 0 = perfect
    # agreement, positive = ExaEpi over-predicts that tract, negative = Epicast over-predicts.
    df["log2_ratio"] = np.log2((df["infected_exaepi"] + 1.0) / (df["infected_epicast"] + 1.0))

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

    geo_df = pd.merge(shp_data, df, on=["GEOID10"], how="inner")
    if geo_df.empty:
        raise SystemExit(
            f"No rows matched after merging: 0 of {len(df)} matched ExaEpi/Epicast {geo_unit} "
            f"GEOIDs were found among the {len(shp_data)} shapefile rows. Check --shape_files is a "
            f"Census {geo_unit.upper()} shapefile (matching --county_level) covering the same state."
        )

    xmin = max(float(args.coord_bounds[0]), float(geo_df.INTPTLON10.astype("float").min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(geo_df.INTPTLON10.astype("float").max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(geo_df.INTPTLAT10.astype("float").min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(geo_df.INTPTLAT10.astype("float").max()) + 0.5)
    yrange = ymax - ymin

    panel_width = 12.0
    fig_x = panel_width * 3
    fig_y = panel_width * yrange / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_x, fig_y))

    max_count = 30000
    infected_norm = mp.colors.LogNorm(vmin=1.0, vmax=max_count)

    # Bound the diverging color scale at the 99th percentile of |log2 ratio| rather than the raw
    # max, so a handful of near-zero-count tracts (where any ratio is essentially noise) don't
    # wash out the color scale for every other tract; matplotlib clips anything beyond the bound
    # to the extreme color rather than dropping it.
    abs_ratio = geo_df["log2_ratio"].abs()
    ratio_bound = max(1.0, float(np.percentile(abs_ratio, 99)))
    n_clipped = int((abs_ratio > ratio_bound).sum())
    if n_clipped:
        print(
            f"NOTE: {n_clipped} {geo_unit}(s) exceed the log2-ratio color scale (+/-{ratio_bound:.2f}) "
            "and are clipped to the extreme color -- likely tiny/near-zero infected counts on one side."
        )
    ratio_norm = mp.colors.Normalize(vmin=-ratio_bound, vmax=ratio_bound)

    # Fade the divergence panel toward transparent wherever NEITHER model predicts many cases, so a
    # 5-vs-1 mismatch doesn't read as visually loud as a 5000-vs-1000 one. Magnitude = whichever of
    # the two models is larger at that tract ("did either model predict a lot of cases here"),
    # log-scaled onto an alpha floor so no tract fully disappears.
    magnitude = np.maximum(geo_df["infected_exaepi"], geo_df["infected_epicast"])
    alpha_floor = 0.15
    magnitude_norm = mp.colors.LogNorm(vmin=1.0, vmax=max_count)
    ratio_alpha = alpha_floor + (1.0 - alpha_floor) * np.clip(magnitude_norm(magnitude.values), 0.0, 1.0)
    ratio_rgba = mp.colormaps["RdBu_r"](ratio_norm(geo_df["log2_ratio"].values))
    ratio_rgba[:, 3] = ratio_alpha

    panels = [
        (ax1, "infected_exaepi", "OrRd", infected_norm, "ExaEpi infected", None),
        (ax2, "infected_epicast", "OrRd", infected_norm, "Epicast infected", None),
        (ax3, "log2_ratio", "RdBu_r", ratio_norm, "log2(ExaEpi / Epicast)", ratio_rgba),
    ]
    for ax, column, cmap, norm, title, explicit_color in panels:
        states.boundary.plot(ax=ax, lw=1, color="black")
        if explicit_color is not None:
            geo_df.plot(ax=ax, color=explicit_color, legend=False)
        else:
            geo_df.plot(ax=ax, column=column, cmap=cmap, legend=False, norm=norm)
        sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_box_aspect(50)
        ax.set_title(title)
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        ax.set_frame_on(False)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    plt.tight_layout()
    print("Plotting results to", args.output)
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
