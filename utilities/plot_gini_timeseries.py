#!/usr/bin/env python

"""Plot how geographically widespread infections are over time, via either the Gini coefficient or
global Moran's I of per-tract (or per-county) infected counts each day, for ExaEpi and/or Epicast.

Gini (--metric gini, the default) measures CONCENTRATION only, independent of geography: 0 means
infections are spread perfectly evenly across every tract/county, ->1 means they're concentrated in
very few -- it doesn't care whether the affected units are next to each other or scattered across
the map. It needs no shapefile, only each unit's infected count (from
load_exaepi_grid_stats/reconstruct_epicast_snapshot's GEOID10-keyed DataFrame).

Global Moran's I (--metric moran) measures spatial AUTOCORRELATION instead: ~+1 means neighboring
tracts/counties tend to have similar infection levels (a contiguous outbreak region), ~0 means the
pattern is spatially random, ~-1 means neighbors tend to differ sharply (a checkerboard pattern).
This needs a shapefile (--shape_files) to build a queen-contiguity adjacency matrix (two units are
neighbors if their boundaries touch), built once per data source and reused across every day, since
the geographic units and their adjacency don't change -- only each day's infected values do.

Both sides are always compared at Census tract granularity at minimum (ExaEpi's block groups are
summed up to their tract, same as plot_geo.py's --tract_level; Epicast is natively tract-level)
since that's the finest resolution common to both, and the whole point is a fair comparison. Pass
--county_level to aggregate both further, up to the county.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_grid_stats, _parse_day_from_plot_dir  # noqa: E402
from plot_geo_epicast import reconstruct_epicast_snapshot  # noqa: E402
from read_epicast_events import read_events_bin  # noqa: E402

import geopandas as gp  # noqa: E402


def gini(x):
    """Gini coefficient of a non-negative array x (0 = perfectly even, ->1 = maximally concentrated
    in one element). Zero-valued entries (e.g. a tract with no infections at all) must be included,
    not filtered out -- they're exactly what pulls Gini toward 1 when infections concentrate
    elsewhere, so len(x) should be every tract/county in the region, not just the affected ones.

    Returns NaN if every entry is zero (e.g. no active infections anywhere, typically once an
    epidemic has burned out): that's a degenerate "nothing to distribute" state, not the same thing
    as "perfectly evenly spread," so it must not silently read as Gini=0 on a plot.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / total) / n)


def build_queen_weights(gdf):
    """Build a binary queen-contiguity adjacency matrix for the given GeoDataFrame (rows assumed
    already in the final desired order), using its spatial index for efficiency. Returns an (n, n)
    array W where W[i, j] = 1 if geometries i and j share any boundary point, else 0 (diagonal is
    always 0 -- a unit is not its own neighbor).
    """
    n = len(gdf)
    W = np.zeros((n, n))
    sindex = gdf.sindex
    for i, geom in enumerate(gdf.geometry.values):
        for j in sindex.query(geom, predicate="touches"):
            if j != i:
                W[i, j] = 1.0
    return W


def morans_i(x, W):
    """Global Moran's I for values x (length n) given an (n, n) adjacency matrix W in the same
    order. ~+1 = neighboring units have similar values (clustered), ~0 = spatially random,
    ~-1 = neighboring units differ sharply (checkerboard). Returns NaN if x has no variance (e.g.
    all zero, typically once an epidemic has burned out) or W has no edges at all.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    dx = x - x.mean()
    S0 = W.sum()
    denom = np.sum(dx ** 2)
    if S0 == 0 or denom == 0:
        return float("nan")
    numer = np.sum(W * np.outer(dx, dx))
    return float((n / S0) * (numer / denom))


def _prepare_geo_weights(shp_data, sample_grid_stats_df, geo_unit):
    """Intersect the shapefile with a sample day's GEOID10 set to get a canonical, geometry-ordered
    list of units, and build the queen-contiguity weights matrix for that fixed order (reused for
    every subsequent day from the same data source, since only the infected values change by day,
    not the set of units or their adjacency). Returns (geoid_order, W).
    """
    merged = pd.merge(shp_data, sample_grid_stats_df[["GEOID10"]], on="GEOID10", how="inner")
    merged = merged.sort_values("GEOID10").reset_index(drop=True)
    if merged.empty:
        raise SystemExit(
            "No rows matched merging --shape_files with the data's GEOID10s -- check --shape_files "
            "is at the same granularity (tract vs county) as --county_level."
        )
    W = build_queen_weights(merged)
    print(f"Built queen-contiguity adjacency for {len(merged)} {geo_unit}s ({int(W.sum())} directed edges)")
    return merged["GEOID10"].values, W


def _reindexed_infected(grid_stats_df, geoid_order):
    """Return grid_stats_df's 'infected' column reindexed to match geoid_order exactly (the fixed
    row order the weights matrix W was built in), filling any missing GEOID10 with 0.
    """
    s = grid_stats_df.set_index("GEOID10")["infected"]
    return s.reindex(geoid_order).fillna(0).values


def main():
    parser = argparse.ArgumentParser(
        description="Plot the Gini coefficient or global Moran's I of infection spread (ExaEpi and/or Epicast) over time"
    )
    parser.add_argument(
        "--plot_dirs", "-p", nargs="+", default=None,
        help="ExaEpi plotfile directories, one per day (e.g. plt00000 plt00010 ... or a shell glob "
        "like plt000*)",
    )
    parser.add_argument("--events_file", "-c", default=None, help="Epicast run.events.bin file")
    parser.add_argument(
        "--days", type=int, nargs="+", default=None,
        help="Days to sample from --events_file (default: every day from 0 to the last)",
    )
    parser.add_argument(
        "--county_level", action="store_true", default=False,
        help="Aggregate to the county level instead of the (default) Census tract level, for both "
        "ExaEpi and Epicast.",
    )
    parser.add_argument(
        "--metric", choices=["gini", "moran"], default="gini",
        help="Which spread metric to compute per day: 'gini' (concentration, no shapefile needed) "
        "or 'moran' (global Moran's I spatial autocorrelation, needs --shape_files) (default: gini)",
    )
    parser.add_argument(
        "--shape_files", "-s", nargs="+", default=None,
        help="Census tract shapefiles (or county shapefiles, with --county_level) -- required for "
        "--metric moran, used to build the queen-contiguity adjacency matrix. Not used for "
        "--metric gini.",
    )
    parser.add_argument("--output", "-o", default="spread_timeseries.png", help="Output plot file")
    parser.add_argument("--fontsize", type=int, default=18, help="Base font size for the plot (default: 18)")
    args = parser.parse_args()

    if not args.plot_dirs and not args.events_file:
        parser.error("At least one of --plot_dirs or --events_file must be given")
    if args.metric == "moran" and not args.shape_files:
        parser.error("--metric moran requires --shape_files (to build the spatial adjacency matrix)")

    plt.rcParams.update({"font.size": args.fontsize})
    geo_unit = "county" if args.county_level else "tract"

    shp_data = None
    if args.metric == "moran":
        shp_dfs = []
        for fname in args.shape_files:
            if not fname.endswith(".shp"):
                print("WARNING: file", fname, "does not appear to be a shapefile with .shp extension")
                continue
            print("Reading data from", fname)
            shp_dfs.append(gp.read_file(fname))
        shp_data = pd.concat(shp_dfs)
        shp_data["GEOID10"] = shp_data["GEOID10"].astype("int64")

    fig, ax = plt.subplots(figsize=(12, 7))

    if args.plot_dirs:
        days, values = [], []
        geoid_order, W = None, None
        for plot_dir in args.plot_dirs:
            grid_stats_df = load_exaepi_grid_stats(plot_dir, tract_level=True, county_level=args.county_level)
            day = _parse_day_from_plot_dir(plot_dir)
            if args.metric == "moran":
                if geoid_order is None:
                    geoid_order, W = _prepare_geo_weights(shp_data, grid_stats_df, geo_unit)
                v = morans_i(_reindexed_infected(grid_stats_df, geoid_order), W)
            else:
                v = gini(grid_stats_df["infected"].values)
            days.append(day)
            values.append(v)
            print(f"ExaEpi day {day}: {args.metric} = {v:.4f} (n={len(grid_stats_df)} {geo_unit}s)")
        order = np.argsort(days)
        days = np.array(days)[order]
        values = np.array(values)[order]
        ax.plot(days, values, "-", label="ExaEpi", color="tab:red", lw=5)

    if args.events_file:
        print("Reading Epicast data from", args.events_file)
        events_df, demog_df = read_events_bin(args.events_file)
        print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")
        max_day = int(events_df.timestep.max() // 2)
        days = args.days if args.days is not None else list(range(0, max_day + 1))
        values = []
        geoid_order, W = None, None
        for day in days:
            grid_stats_df, resolved_day = reconstruct_epicast_snapshot(
                events_df, demog_df, day=day, county_level=args.county_level
            )
            if args.metric == "moran":
                if geoid_order is None:
                    geoid_order, W = _prepare_geo_weights(shp_data, grid_stats_df, geo_unit)
                v = morans_i(_reindexed_infected(grid_stats_df, geoid_order), W)
            else:
                v = gini(grid_stats_df["infected"].values)
            values.append(v)
            print(f"Epicast day {resolved_day}: {args.metric} = {v:.4f} (n={len(grid_stats_df)} {geo_unit}s)")
        ax.plot(days, values, "-", label="Epicast", color="tab:blue", lw=5)

    ax.set_xlabel("Day")
    if args.metric == "moran":
        #ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Spatial randomness (I=0)")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel(f"Global Moran's I of infected count (per {geo_unit})")
        #ax.set_title("Spatial autocorrelation of infections over time")
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Gini coefficient of infected count (per {geo_unit})")
        #ax.set_title("Geographic spread of infections over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print("Wrote plot to", args.output)


if __name__ == "__main__":
    main()
