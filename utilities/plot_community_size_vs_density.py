#!/usr/bin/env python

"""Plot ExaEpi community size (population per community) against population density (people /
km^2) for one ExaEpi plotfile directory, as two overlaid series: nighttime (home-based) and
daytime (work-based) population.

Each ExaEpi community is one AMReX grid cell. Nighttime population counts agents by home cell
(home_i, home_j); daytime population counts the same agents by work/school cell (work_i,
work_j) instead -- agents with no separate work/school location (retirees, preschoolers, ...)
have work_i/work_j == home_i/home_j, so they contribute equally to both. This is the same
home/work reconstruction plot_geo_daynight.py uses; see its docstring for why it's an exact,
unbiased census of every agent rather than a partial sample.

Density is computed from the same Census block group shapefiles (GEOID10 + ALAND10 land area,
via --shape_files): when a Census unit's population exceeds 2000, ExaEpi splits it into several
communities (grid cells) that all share the same GEOID10 and therefore the same land area, so
this is a many-communities-to-one-area join, not one-to-one. Land area doesn't depend on which
plotfile you pass, but population does (nighttime vs daytime) -- unlike land area, so is *not*
static across a run's other plotfiles for a residential-vs-workplace split like this one.

Needs the plotfile directory AT STEP 0 specifically (e.g. plt00000): home_i/home_j/work_i/work_j
are static per agent and only written into the "agents" particle plotfile at that step (see
read_exaepi_agents.py). Only Census-initialized runs (ic_type=Census) are supported, same as
plot_geo.py/plot_geo_daynight.py -- the plotfile's FIPS/Tract mesh fields this depends on aren't
populated for UrbanPop runs.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import yt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_exaepi_agents import read_agent_fields  # noqa: E402

SQ_M_PER_SQ_KM = 1_000_000.0
MIN_BIN_COUNT = 5  # minimum communities in a density bin before its median is plotted

# Per series: scatter color, trend-line color, and label.
SERIES_STYLE = {
    "night": {"scatter": "#2a78d6", "trend": "#0b3d78", "label": "Nighttime (home)"},
    "day": {"scatter": "#eb9134", "trend": "#a83214", "label": "Daytime (work)"},
}


def _build_geoid_grid(ds):
    """Return a 2D int64 array geoid_grid[i, j] = 12-digit block-group GEOID10 for that grid cell
    (-1 for inactive/off-domain cells), aligned with the same (i, j) indexing used by agents'
    home_i/home_j/work_i/work_j -- see plot_geo_daynight.py, which this is copied from."""
    dims = ds.domain_dimensions
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    fips = cg["boxlib", "FIPS"][:, :, 0].astype("int64")
    tract = cg["boxlib", "Tract"][:, :, 0].astype("int64")
    return np.where(fips >= 0, fips * 10_000_000 + tract, -1)


def load_community_density(plot_dir, shape_files):
    """Return a DataFrame with one row per ExaEpi community (grid cell): GEOID10, night_pop,
    day_pop, area_km2, density_night, density_day."""
    print("Reading ExaEpi mesh data from", plot_dir)
    ds = yt.load(plot_dir)  # type: ignore
    geoid_grid = _build_geoid_grid(ds)
    ni, nj = geoid_grid.shape

    print("Reading agent home/work assignments from", plot_dir)
    agents = read_agent_fields(plot_dir, ["home_i", "home_j", "work_i", "work_j"])
    print(f"Read {len(agents['home_i']):,} agents")

    def cell_counts(i_arr, j_arr):
        flat = i_arr.astype(np.int64) * nj + j_arr.astype(np.int64)
        return np.bincount(flat, minlength=ni * nj).reshape(ni, nj)

    night_grid = cell_counts(agents["home_i"], agents["home_j"])
    day_grid = cell_counts(agents["work_i"], agents["work_j"])

    df = pd.DataFrame(
        {
            "GEOID10": geoid_grid.ravel(),
            "night_pop": night_grid.ravel(),
            "day_pop": day_grid.ravel(),
        }
    )
    # Inactive/off-domain cells (GEOID10 == -1) and cells nobody lives or works in either.
    df = df[(df["GEOID10"] >= 0) & ((df["night_pop"] > 0) | (df["day_pop"] > 0))].reset_index(drop=True)

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
            f"No rows matched after merging: 0 of {geoid_grid[geoid_grid >= 0].size} ExaEpi community "
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


def _add_series(ax, df, pop_col, density_col, style, log, label_suffix=""):
    """Scatter one (population, density) series plus its own binned-median trend line, and
    return the Pearson correlation (log10 density, population) for the printed summary."""
    sub = df[df[pop_col] > 0]
    ax.scatter(sub[pop_col], sub[density_col], s=100, alpha=0.35, color=style["scatter"],
               linewidths=0, label=style["label"] + label_suffix)

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
    ax.plot(bin_medians, bin_centers, color=style["trend"], lw=5,
            label=f"Median {style['label'].lower()} size (binned)")

    log_density = np.log10(sub[density_col])
    return np.corrcoef(log_density, sub[pop_col])[0, 1], len(sub)


def plot_community_size_vs_density(df, output, log=False):
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(10, 8))

    for series, pop_col, density_col in (("night", "night_pop", "density_night"),
                                          ("day", "day_pop", "density_day")):
        corr, n = _add_series(ax, df, pop_col, density_col, SERIES_STYLE[series], log)
        print(f"{n} communities plotted ({series})")
        print(f"Pearson correlation (log10 density, community size), {series}: {corr:.3f}")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Community size (population)")
    ax.set_ylabel("Population density (people / km²)")
    ax.legend(frameon=False, fontsize=18, loc="upper left")
    ax.set_ylim(0.01, 1000000)
    plt.tight_layout()
    print("Plotting results to", output)
    plt.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plot_dir", "-p", required=True,
        help="ExaEpi plotfile directory AT STEP 0 (e.g. plt00000) -- home/work assignments are "
        "only written there (see read_exaepi_agents.py)",
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

    df = load_community_density(args.plot_dir, args.shape_files)
    plot_community_size_vs_density(df, args.output, log=args.log)


if __name__ == "__main__":
    main()
