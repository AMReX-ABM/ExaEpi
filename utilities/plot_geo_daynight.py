#!/usr/bin/env python

"""Plot an ExaEpi choropleth of daytime-minus-nighttime population per Census tract (or county).

Nighttime population = agents counted by home tract (this exactly reproduces the standard
plotfile's per-community "total" field -- see the sanity check in main()). Daytime population =
the same agents counted by work/school tract instead. Agents with no separate work/school location
assigned (retirees, preschoolers, ...) have work_i/work_j == home_i/home_j, so they contribute
equally to both and don't affect the difference; agents who work/attend school locally (same grid
cell as home) likewise net out to zero even though they do have a job/school assigned.

Positive (red) tracts gain population during the day (workplace/school destinations); negative
(blue) tracts lose population (bedroom communities). This is the direct ExaEpi analogue of the
Epicast day/night investigation -- unlike Epicast's run.events.bin (which only records a location
for an agent's single exposure event, so household- and work/school-exposed agents are mutually
exclusive and can never be paired into a home+work commute), ExaEpi's agents plotfile has an
explicit, static home_i/home_j/work_i/work_j pair for every agent, so this is an exact
reconstruction, not a partial/biased sample -- see read_exaepi_agents.py's docstring.

Needs the plotfile directory AT STEP 0 specifically (e.g. plt00000): home_i/home_j/work_i/work_j
are static per agent and only written into the "agents" particle plotfile at that step (see
read_exaepi_agents.py).

Pass --population workers or --population students to restrict the whole analysis (both the
night/home and day/work side) to just that subset of agents, instead of everyone -- e.g.
--population workers isolates commuting to workplaces from the (usually larger, more local)
school-run pattern.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_exaepi_agents import read_agent_fields  # noqa: E402

# Same worker/student definition used by check_nt_dt.py: naics != -1 identifies a worker
# (regardless of school_id); among the rest, school_id != 0 identifies a student. The two are
# mutually exclusive by construction.
_POPULATION_FILTERS = {
    "all": None,
    "workers": lambda naics, school_id: naics != -1,
    "students": lambda naics, school_id: (naics == -1) & (school_id != 0),
}


def _build_geoid_grid(ds):
    """Return a 2D int64 array geoid_grid[i, j] = 12-digit block-group GEOID10 for that grid cell
    (-1 for inactive/off-domain cells), aligned with the same (i, j) indexing used by agents'
    home_i/home_j/work_i/work_j.
    """
    dims = ds.domain_dimensions
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    fips = cg["boxlib", "FIPS"][:, :, 0].astype("int64")
    tract = cg["boxlib", "Tract"][:, :, 0].astype("int64")
    return np.where(fips >= 0, fips * 10_000_000 + tract, -1)


def compute_day_night_pop(plot_dir, county_level=False, population="all"):
    """Return a DataFrame (GEOID10, night_pop, day_pop, diff) at the tract level (or county level
    if county_level is set), built from every agent's home/work grid cell.

    population : {"all", "workers", "students"}
        Restrict to just the agents with a job (naics != -1) or just those enrolled in school
        (school_id != 0 among the non-workers) before counting -- see _POPULATION_FILTERS. With
        "all" (default), every agent is counted, matching the plotfile's own census exactly.
    """
    print("Reading ExaEpi mesh data from", plot_dir)
    ds = yt.load(plot_dir)  # type: ignore
    geoid_grid = _build_geoid_grid(ds)

    fields = ["home_i", "home_j", "work_i", "work_j"]
    if population != "all":
        fields += ["naics", "school_id"]
    print("Reading agent home/work assignments from", plot_dir)
    agents = read_agent_fields(plot_dir, fields)
    home_geoid = geoid_grid[agents["home_i"], agents["home_j"]]
    work_geoid = geoid_grid[agents["work_i"], agents["work_j"]]
    print(f"Read {len(home_geoid):,} agents")

    if population != "all":
        mask = _POPULATION_FILTERS[population](agents["naics"], agents["school_id"])
        home_geoid = home_geoid[mask]
        work_geoid = work_geoid[mask]
        print(f"Restricted to {population}: {mask.sum():,} of {len(mask):,} agents")

    night = pd.Series(home_geoid).value_counts().rename("night_pop")
    day = pd.Series(work_geoid).value_counts().rename("day_pop")
    df = pd.concat([night, day], axis=1).fillna(0).astype("int64")
    df.index.name = "GEOID10"
    df = df.reset_index()
    df = df[df.GEOID10 >= 0].reset_index(drop=True)

    if population == "all":
        # Sanity check: night_pop (agents by home cell) must exactly reproduce the standard
        # plotfile's per-community "total" field, since both are simply a census of agents by
        # home cell. Only valid for the full population -- a worker/student subset is expected to
        # undercount "total" by construction.
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        total_grid = cg["boxlib", "total"][:, :, 0].astype("int64")
        mesh_total = (
            pd.DataFrame({"GEOID10": geoid_grid.ravel(), "mesh_total": total_grid.ravel()})
            .loc[lambda d: d.GEOID10 >= 0]
            .groupby("GEOID10", as_index=False)
            .mesh_total.sum()
        )
        check = df.merge(mesh_total, on="GEOID10", how="outer").fillna(0)
        n_mismatch = int((check.night_pop != check.mesh_total).sum())
        if n_mismatch:
            print(
                f"WARNING: {n_mismatch} of {len(check)} block groups have a night_pop that doesn't "
                "match the plotfile's own 'total' field -- home_i/home_j to GEOID10 mapping may be off."
            )
        else:
            print(f"Sanity check passed: night_pop matches the plotfile's 'total' field for all {len(check)} block groups")

    geo_unit = "county" if county_level else "tract"
    divisor = 10 ** 7 if county_level else 10  # block group (12-digit) -> county (5) or tract (11)
    df["GEOID10"] = df["GEOID10"] // divisor
    df = df.groupby("GEOID10", as_index=False)[["night_pop", "day_pop"]].sum()
    df["diff"] = df["day_pop"] - df["night_pop"]
    print(f"Aggregated to {len(df)} {geo_unit}s")
    return df


def main():
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 24

    parser = argparse.ArgumentParser(
        description="Plot an ExaEpi choropleth of daytime-minus-nighttime population per tract/county"
    )
    parser.add_argument(
        "--plot_dir", "-p", required=True,
        help="ExaEpi plotfile directory AT STEP 0 (e.g. plt00000) -- home/work assignments are "
        "only written there (see read_exaepi_agents.py)",
    )
    parser.add_argument(
        "--shape_files", "-s", required=True, nargs="+",
        help="Census TRACT shape files (.shp) by default, or Census COUNTY shape files if "
        "--county_level is passed. Available from\n"
        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Census+Tracts",
    )
    parser.add_argument("--states_file", "-e", required=True, help="Shape file for US states")
    parser.add_argument("--output", "-o", default="geo_daynight.pdf", help="Output file name for plot")
    parser.add_argument(
        "--coord_bounds", "-b", default=[-170, -66.6, 18.5, 71.5], nargs="+",
        help="Range for longitude: min,max",
    )
    parser.add_argument(
        "--county_level", action="store_true", default=False,
        help="Aggregate and plot at the Census county level instead of the default tract level. "
        "Pass a Census county shapefile (not a tract one) via --shape_files when using this.",
    )
    parser.add_argument(
        "--population", choices=list(_POPULATION_FILTERS), default="all",
        help="Restrict to just the day/night movement of workers (naics != -1) or students "
        "(enrolled in school, among the non-workers) instead of the whole population "
        "(default: all).",
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Minimum value for the color scale; tracts/counties below this are clipped to the "
        "same color as this value (default: -99th percentile of |diff|, i.e. symmetric around 0).",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Maximum value for the color scale; tracts/counties above this are clipped to the "
        "same color as this value (default: 99th percentile of |diff|, i.e. symmetric around 0).",
    )
    args = parser.parse_args()

    geo_unit = "county" if args.county_level else "tract"
    geo_unit_pl = "counties" if args.county_level else "tracts"

    df = compute_day_night_pop(args.plot_dir, county_level=args.county_level, population=args.population)

    shp_dfs = []
    state_codes = []
    for fname in args.shape_files:
        if not fname.endswith(".shp"):
            print(
                "WARNING: file", fname,
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

    geo_df = pd.merge(shp_data, df, on="GEOID10", how="inner")
    if geo_df.empty:
        example = "tl_2010_35_county10.shp" if args.county_level else "tl_2010_35_tract10.shp"
        raise SystemExit(
            f"No rows matched after merging: 0 of {len(df)} ExaEpi {geo_unit} GEOIDs were found "
            f"among the {len(shp_data)} shapefile rows. Check --shape_files is a Census "
            f"{geo_unit.upper()} shapefile (e.g. {example}) covering the same state as --plot_dir."
        )
    print(f"Matched {len(geo_df)} of {len(df)} {geo_unit_pl}")

    print(f"Total population moving day<->night: {geo_df['diff'].abs().sum() // 2:,} agents "
          f"(of {geo_df['night_pop'].sum():,} total)")
    top_gain = geo_df.nlargest(5, "diff")[["GEOID10", "night_pop", "day_pop", "diff"]]
    top_loss = geo_df.nsmallest(5, "diff")[["GEOID10", "night_pop", "day_pop", "diff"]]
    print("Top 5 daytime-population-gaining", geo_unit_pl + ":\n", top_gain.to_string(index=False))
    print("Top 5 daytime-population-losing", geo_unit_pl + ":\n", top_loss.to_string(index=False))

    xmin = max(float(args.coord_bounds[0]), float(geo_df.INTPTLON10.astype("float").min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(geo_df.INTPTLON10.astype("float").max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(geo_df.INTPTLAT10.astype("float").min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(geo_df.INTPTLAT10.astype("float").max()) + 0.5)
    yrange = ymax - ymin

    panel_width = 16.0
    fig_x = panel_width
    fig_y = panel_width * yrange / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, ax = plt.subplots(figsize=(fig_x, fig_y))

    # Diverging colormap bounded by the 99th percentile of |diff| by default, so a handful of
    # extreme tracts (usually tiny-population ones where any change looks huge in relative terms)
    # don't wash out the color scale for every other tract. --vmin/--vmax override either side
    # independently. TwoSlopeNorm keeps 0 pinned to the middle (white) of the colormap regardless
    # of how asymmetric vmin/vmax are -- a plain Normalize would instead place 0 wherever it falls
    # proportionally between vmin and vmax, off-center whenever they're not symmetric. Values
    # outside [vmin, vmax] are clipped to the exact endpoint color automatically.
    auto_bound = max(1.0, float(np.percentile(geo_df["diff"].abs(), 99)))
    vmin = args.vmin if args.vmin is not None else -auto_bound
    vmax = args.vmax if args.vmax is not None else auto_bound
    if vmin >= 0 or vmax <= 0:
        raise SystemExit(
            f"--vmin ({vmin}) must be negative and --vmax ({vmax}) must be positive, so that 0 "
            "sits at the center of the color scale."
        )
    n_clipped = int(((geo_df["diff"] < vmin) | (geo_df["diff"] > vmax)).sum())
    if n_clipped:
        print(
            f"NOTE: {n_clipped} {geo_unit}(s) fall outside the color scale [{vmin:.0f}, "
            f"{vmax:.0f}] and are clipped to the extreme color."
        )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    title = "ExaEpi daytime − nighttime population"
    if args.population != "all":
        title += f" ({args.population})"

    states.boundary.plot(ax=ax, lw=1, color="black")
    geo_df.plot(ax=ax, column="diff", cmap="bwr", legend=True, norm=norm)
    # geopandas appends the colorbar as a new axes on the same figure; grab it to double its tick
    # label size (it otherwise inherits the much smaller default rcParams ytick.labelsize).
    fig.axes[-1].tick_params(labelsize=24)
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
