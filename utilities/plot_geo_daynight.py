#!/usr/bin/env python

"""Plot an ExaEpi choropleth of daytime-minus-nighttime population per Census tract (or county).

Nighttime population = agents counted by home cell; daytime population = the same agents counted
by work/school cell instead. Agents with no separate work/school location assigned (retirees,
preschoolers, ...) have work_i/work_j == home_i/home_j, so they contribute equally to both and
don't affect the difference; agents who work/attend school locally (same grid cell as home)
likewise net out to zero even though they do have a job/school assigned.

Positive (red) tracts gain population during the day (workplace/school destinations); negative
(blue) tracts lose population (bedroom communities). This is the direct ExaEpi analogue of the
Epicast day/night investigation -- unlike Epicast's run.events.bin (which only records a location
for an agent's single exposure event, so household- and work/school-exposed agents are mutually
exclusive and can never be paired into a home+work commute), ExaEpi's static, once-per-run
day/night population CSV (see below) is built from every agent's explicit home_i/home_j/work_i/
work_j pair, so this is an exact reconstruction, not a partial/biased sample.

Population comes from the static, once-per-run <prefix>_day_night_population.csv ExaEpi writes
when --aggregated_diag_int is enabled (see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp,
and plot_geo.load_exaepi_day_night_population) -- already broken out by everyone/workers/students,
so no plotfile/agent data is needed at all.

Pass --population workers or --population students to restrict the whole analysis (both the
night/home and day/work side) to just that subset of agents, instead of everyone -- e.g.
--population workers isolates commuting to workplaces from the (usually larger, more local)
school-run pattern. This is purely a choice of which pre-computed CSV columns to use, not a
recomputation -- the CSV always carries all three breakdowns.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import load_exaepi_day_night_population  # noqa: E402
from plos_compbio_style import (  # noqa: E402
    apply_style,
    HALF_PAGE_WIDTH_IN,
    FULL_PAGE_WIDTH_IN,
    AXES_LINEWIDTH,
    FONT_TICK,
    FONT_TITLE,
)

# Which pair of load_exaepi_day_night_population's columns each --population choice selects --
# "workers"/"students" match ExaEpi::IO::writeStaticAggregatedData's own naics!=-1 /
# (naics==-1 & school_id!=0) split. ("all" uses night_pop/day_pop, the loader's renamed form of
# the CSV's night_total/day_total columns.)
_POPULATION_COLUMNS = {
    "all": ("night_pop", "day_pop"),
    "workers": ("night_workers", "day_workers"),
    "students": ("night_students", "day_students"),
}


def compute_day_night_pop(day_night_csv, county_level=False, population="all"):
    """Return a DataFrame (GEOID10, night_pop, day_pop, diff) at the tract level (or county level
    if county_level is set), built from ExaEpi's static day/night population CSV (see
    plot_geo.load_exaepi_day_night_population).

    population : {"all", "workers", "students"}
        Which of the CSV's pre-computed everyone/workers/students breakdowns to use -- see
        _POPULATION_COLUMNS. This is purely a column choice: the CSV always carries all three, so
        nothing is recomputed here.
    """
    night_col, day_col = _POPULATION_COLUMNS[population]
    df = load_exaepi_day_night_population(day_night_csv)[["GEOID10", night_col, day_col]]
    df = df.rename(columns={night_col: "night_pop", day_col: "day_pop"})

    geo_unit = "county" if county_level else "tract"
    divisor = 10 ** 7 if county_level else 10  # block group (12-digit) -> county (5) or tract (11)
    df["GEOID10"] = df["GEOID10"] // divisor
    df = df.groupby("GEOID10", as_index=False)[["night_pop", "day_pop"]].sum()
    df["diff"] = df["day_pop"] - df["night_pop"]
    print(f"Aggregated to {len(df)} {geo_unit}s")
    return df


def compute_day_night_pop_epicast(lodes_file, county_level=False):
    """Return a DataFrame (GEOID10, night_pop, day_pop, diff) at the tract level (or
    county level if county_level is set), built from Epicast's LODES-derived home->work
    commute-flow CSV (columns: src, dst, flow -- 11-digit Census tract GEOIDs, already at
    tract granularity so no block-group-to-tract rollup is needed the way ExaEpi's raw
    agent data requires).

    This is WORKER commute flow only (LODES has no student data), unlike ExaEpi's
    agent-based count which can include the whole population -- see --population in
    main(), which defaults to "workers" whenever an Epicast comparison is requested so
    the two sides share the same population basis.
    """
    print("Reading Epicast LODES commute flows from", lodes_file)
    flows = pd.read_csv(lodes_file)
    night = flows.groupby("src")["flow"].sum().rename("night_pop")
    day = flows.groupby("dst")["flow"].sum().rename("day_pop")
    df = pd.concat([night, day], axis=1).fillna(0).astype("int64")
    df.index.name = "GEOID10"
    df = df.reset_index()
    print(f"Read {len(flows):,} commute flows, {len(df)} tracts")

    if county_level:
        df["GEOID10"] = df["GEOID10"] // 10 ** 6  # 11-digit tract -> 5-digit county
        df = df.groupby("GEOID10", as_index=False)[["night_pop", "day_pop"]].sum()
    df["diff"] = df["day_pop"] - df["night_pop"]
    geo_unit = "county" if county_level else "tract"
    print(f"Aggregated to {len(df)} {geo_unit}s")
    return df


def _prepare_geo_df(df, shp_data, geo_unit, geo_unit_pl, source_label):
    """Merge a day/night pop df (GEOID10, night_pop, day_pop, diff) with the shapefile data,
    print that source's top gainers/losers and total moving population, and return the merged
    geo_df. Raises SystemExit if nothing matched.
    """
    geo_df = pd.merge(shp_data, df, on="GEOID10", how="inner")
    if geo_df.empty:
        example = "tl_2010_35_county10.shp" if geo_unit == "county" else "tl_2010_35_tract10.shp"
        raise SystemExit(
            f"No rows matched after merging: 0 of {len(df)} {source_label} {geo_unit} GEOIDs were "
            f"found among the {len(shp_data)} shapefile rows. Check --shape_files is a Census "
            f"{geo_unit.upper()} shapefile (e.g. {example}) covering the same state as the data."
        )
    print(f"{source_label}: matched {len(geo_df)} of {len(df)} {geo_unit_pl}")

    print(f"{source_label}: total population moving day<->night: {geo_df['diff'].abs().sum() // 2:,} "
          f"agents (of {geo_df['night_pop'].sum():,} total)")
    top_gain = geo_df.nlargest(5, "diff")[["GEOID10", "night_pop", "day_pop", "diff"]]
    top_loss = geo_df.nsmallest(5, "diff")[["GEOID10", "night_pop", "day_pop", "diff"]]
    print(f"{source_label} top 5 daytime-population-gaining", geo_unit_pl + ":\n", top_gain.to_string(index=False))
    print(f"{source_label} top 5 daytime-population-losing", geo_unit_pl + ":\n", top_loss.to_string(index=False))
    return geo_df


def _plot_panel(ax, states, geo_df, norm, legend):
    """Draw one day/night choropleth panel (state boundaries + diff choropleth) onto ax, with the
    shared formatting (no ticks/frame) common to every panel. legend=True lets geopandas append
    its own colorbar (single-panel mode); pass False and draw a shared colorbar separately instead
    for a multi-panel figure (see plot_geo.py's identical convention).
    """
    states.boundary.plot(ax=ax, lw=AXES_LINEWIDTH, color="black")
    geo_df.plot(ax=ax, column="diff", cmap="bwr", legend=legend, norm=norm)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    ax.set_frame_on(False)


def main():
    apply_style()

    parser = argparse.ArgumentParser(
        description="Plot an ExaEpi choropleth of daytime-minus-nighttime population per "
        "tract/county, optionally alongside a matching Epicast panel for side-by-side comparison"
    )
    parser.add_argument(
        "--day_night_csv", "-n", required=True,
        help="ExaEpi's <prefix>_day_night_population.csv (written when --aggregated_diag_int is "
        "enabled -- see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp)",
    )
    parser.add_argument(
        "--epicast_file", "-c", default=None,
        help="Epicast LODES-derived home->work commute-flow CSV (columns: src, dst, flow -- "
        "11-digit Census tract GEOIDs), e.g. "
        "data/results/emerge-paper/epicast/2019-lodes7-thr0_us.csv. If given, plots a second "
        "'Epicast' panel next to the 'ExaEpi' one, sharing the same color scale. This data is "
        "WORKER commute flow only (LODES has no student data) -- see --population.",
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
        "--population", choices=list(_POPULATION_COLUMNS), default=None,
        help="Restrict ExaEpi to just the day/night movement of workers (naics != -1) or students "
        "(enrolled in school, among the non-workers) instead of the whole population. Default: "
        "'all' -- unless --epicast_file is given, in which case it defaults to 'workers' instead, "
        "since Epicast's LODES commute data is worker-only and this keeps both panels on the same "
        "population basis unless explicitly overridden.",
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Minimum value for the color scale (shared across both panels if --epicast_file is "
        "given); tracts/counties below this are clipped to the same color as this value (default: "
        "-99th percentile of |diff|, i.e. symmetric around 0).",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Maximum value for the color scale (shared across both panels if --epicast_file is "
        "given); tracts/counties above this are clipped to the same color as this value (default: "
        "99th percentile of |diff|, i.e. symmetric around 0).",
    )
    args = parser.parse_args()

    population = args.population if args.population is not None else ("workers" if args.epicast_file else "all")

    geo_unit = "county" if args.county_level else "tract"
    geo_unit_pl = "counties" if args.county_level else "tracts"

    exaepi_df = compute_day_night_pop(args.day_night_csv, county_level=args.county_level, population=population)
    epicast_df = (
        compute_day_night_pop_epicast(args.epicast_file, county_level=args.county_level)
        if args.epicast_file else None
    )

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

    panels = [(_prepare_geo_df(exaepi_df, shp_data, geo_unit, geo_unit_pl, "ExaEpi"), "ExaEpi")]
    if epicast_df is not None:
        panels.append((_prepare_geo_df(epicast_df, shp_data, geo_unit, geo_unit_pl, "Epicast"), "Epicast"))

    # Bounds are the union across every panel, so a two-panel figure shares one consistent
    # geographic extent instead of each panel framing itself differently (see plot_geo.py's
    # identical multi-panel bounds handling).
    # Bounds come from the polygons' own geometry, not the INTPTLON10/INTPTLAT10 internal points --
    # see plot_geo.py for why (a centroid sits an arbitrary distance inside its own polygon's edge,
    # which crops the map's edges outright at county granularity).
    bounds = np.array([geo_df.total_bounds for geo_df, _ in panels])  # minx, miny, maxx, maxy
    pad = 0.1
    xmin = max(float(args.coord_bounds[0]), bounds[:, 0].min() - pad)
    xmax = min(float(args.coord_bounds[1]), bounds[:, 2].max() + pad)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), bounds[:, 1].min() - pad)
    ymax = min(float(args.coord_bounds[3]), bounds[:, 3].max() + pad)
    yrange = ymax - ymin

    # Diverging colormap bounded by the 99th percentile of |diff| by default -- taken over the
    # union of every panel's diff values, so a two-panel figure's shared scale is set fairly rather
    # than by whichever panel happens to be considered first -- so a handful of extreme tracts
    # (usually tiny-population ones where any change looks huge in relative terms) don't wash out
    # the color scale for every other tract. --vmin/--vmax override either side independently.
    # TwoSlopeNorm keeps 0 pinned to the middle (white) of the colormap regardless of how asymmetric
    # vmin/vmax are -- a plain Normalize would instead place 0 wherever it falls proportionally
    # between vmin and vmax, off-center whenever they're not symmetric. Values outside [vmin, vmax]
    # are clipped to the exact endpoint color automatically.
    all_diff = pd.concat([geo_df["diff"] for geo_df, _ in panels])
    auto_bound = max(1.0, float(np.percentile(all_diff.abs(), 99)))
    vmin = args.vmin if args.vmin is not None else -auto_bound
    vmax = args.vmax if args.vmax is not None else auto_bound
    if vmin >= 0 or vmax <= 0:
        raise SystemExit(
            f"--vmin ({vmin}) must be negative and --vmax ({vmax}) must be positive, so that 0 "
            "sits at the center of the color scale."
        )
    for geo_df, label in panels:
        n_clipped = int(((geo_df["diff"] < vmin) | (geo_df["diff"] > vmax)).sum())
        if n_clipped:
            print(
                f"NOTE: {label}: {n_clipped} {geo_unit}(s) fall outside the color scale "
                f"[{vmin:.0f}, {vmax:.0f}] and are clipped to the extreme color."
            )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # "ExaEpi" is dropped from the single-panel title (kept for the caption instead) -- this figure
    # is only ~3.1in wide in the paper, and the full "ExaEpi daytime - nighttime population" doesn't
    # fit at PLOS's 8-12pt font floor. Not used in two-panel mode, where each subplot is titled with
    # its own model name instead (see below) and there's no overall figure title.
    title = "Day − night population"
    if population != "all":
        title += f" ({population})"

    if epicast_df is None:
        fig_x = HALF_PAGE_WIDTH_IN
        fig_y = fig_x * yrange / xrange
        print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

        fig, ax = plt.subplots(figsize=(fig_x, fig_y), layout="constrained")
        _plot_panel(ax, states, panels[0][0], norm, legend=True)
        # geopandas appends the colorbar as a new axes on the same figure; grab it to match the
        # shared tick label size (it otherwise inherits matplotlib's own default, not our rcParams
        # override).
        fig.axes[-1].tick_params(labelsize=FONT_TICK)
        ax.set_title(title)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    else:
        # Two panels (ExaEpi, Epicast) side by side, sharing the color scale computed above.
        # Axes are placed by hand (fig.add_axes with explicit rects) rather than via
        # plt.subplots + constrained_layout: geopandas' choropleths get a fixed DATA aspect
        # ratio, and constrained_layout reserves title space by shrinking each axes' allocated
        # grid cell, not the aspect-locked box itself -- the box ends up smaller than its cell
        # with the leftover space auto-anchored below the title, which reads as a big gap above
        # each map no matter how small a title pad is requested. Placing every axes at an exact,
        # precomputed rect sidesteps that entirely: each map's box is sized to need exactly the
        # space it's given, so there's no leftover for the layout engine to insert.
        n = len(panels)
        fig_x = FULL_PAGE_WIDTH_IN
        # FONT_TITLE because that's what set_title() draws at, inherited from apply_style()'s
        # axes.titlesize rather than named at the call site below.
        top_margin_in = (FONT_TITLE * 1.3 + 4) / 72  # one panel-title line + its pad
        # The colorbar's own bottom-most tick label is vertically CENTERED on its tick, so roughly
        # half that label's text height would sit below y=0 (off the bottom of the figure) if the
        # colorbar's axis started flush at the bottom edge like the maps do (which have no tick
        # labels, so they're fine flush). Reserve a small margin so that undershoot lands inside
        # the figure instead of overflowing/getting clipped, and apply it to every axes so the
        # maps/colorbar all still line up on the same bottom edge.
        bottom_margin_in = FONT_TICK / 2 / 72
        cbar_w_in = 0.12
        cbar_gap_in = 0.08
        cbar_label_w_in = 0.65  # room for the colorbar's own tick labels (e.g. "-10000")
        panel_gap_in = -1.5
        panel_w_in = (fig_x - cbar_w_in - cbar_gap_in - cbar_label_w_in - (n - 1) * panel_gap_in) / n
        map_h_in = panel_w_in * yrange / xrange
        fig_y = top_margin_in + map_h_in + bottom_margin_in
        print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

        fig = plt.figure(figsize=(fig_x, fig_y))
        bottom = bottom_margin_in / fig_y
        height = map_h_in / fig_y
        for i, (geo_df, label) in enumerate(panels):
            left_in = i * (panel_w_in + panel_gap_in)
            ax = fig.add_axes((left_in / fig_x, bottom, panel_w_in / fig_x, height))
            _plot_panel(ax, states, geo_df, norm, legend=False)
            ax.set_title(label, pad=3)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

        cbar_left_in = n * panel_w_in + (n - 1) * panel_gap_in + cbar_gap_in
        cax = fig.add_axes((cbar_left_in / fig_x, bottom, cbar_w_in / fig_x, height))
        sm = mcm.ScalarMappable(norm=norm, cmap="bwr")
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_TICK)

    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
