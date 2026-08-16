#!/usr/bin/env python

"""Plot an Epicast infection-spread choropleth, matching plot_geo.py's look and formatting.

Epicast's run.events.bin file has no per-timestep snapshot the way an ExaEpi plotfile does --
it's a log of AgentTransition events (one row per disease_state change). To get a "snapshot as
of day D" comparable to plot_geo.py's per-community pop/never_infected/infected/immune columns,
this script reconstructs each agent's most recent disease_state (and the tract they were in when
that transition happened) among all their events with timestep <= day D, then buckets agents by
that tract:
    immune         = last state is "recovered"
    infected       = last state is exposed/presymptomatic/symptomatic/asymptomatic (still active)
    never_infected = tract population (from the file's demographics) minus the above two
Agents with zero events by day D never appear in the reconstruction and are implicitly counted as
never_infected via that subtraction.

Epicast's finest geographic unit is the Census TRACT (11-digit GEOID), not the block group ExaEpi
communities use -- so this plots at tract level and needs a tract shapefile, not a block group one.
Pass --county_level to aggregate further, up to the county (with a county shapefile instead).
"""

import os
import sys
import argparse
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from read_epicast_events import read_events_bin  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_agg_utils import aggregate_to_county  # noqa: E402

_ACTIVE_STATES = {"exposed", "presymptomatic", "symptomatic", "asymptomatic"}


def reconstruct_epicast_snapshot(events_df, demog_df, day=None, county_level=False):
    """Given already-loaded Epicast events/demographics (see read_events_bin), reconstruct a
    snapshot DataFrame (columns GEOID10, pop, never_infected, infected, immune) as of the given
    0-based day (default: the last day in the data; clamped if it exceeds that), aggregated up to
    the county level if county_level is set (Epicast's native granularity is the tract). Returns
    (grid_stats_df, day). Separated from load_epicast_grid_stats so a sequence of several days from
    the SAME events file can reuse one read_events_bin call instead of re-reading the file per day.
    See the module docstring for how the snapshot is reconstructed from the underlying transition log.
    """
    max_day = int(events_df.timestep.max() // 2)
    day = max_day if day is None else day
    if day > max_day:
        print(f"WARNING: requested day {day} exceeds the last available day ({max_day}); using {max_day} instead")
        day = max_day
    cutoff_timestep = 2 * day + 1
    print(f"Reconstructing snapshot at day {day} (timestep <= {cutoff_timestep})")

    # Reconstruct each agent's most recent disease_state (and the tract of that transition) among
    # events at or before the cutoff -- see module docstring for why this, rather than a simple
    # per-column aggregate, is needed to get a snapshot-like view out of a transition log.
    sub = events_df[events_df.timestep <= cutoff_timestep]
    last_idx = sub.groupby("true_agent_id")["timestep"].idxmax()
    last_events = sub.loc[last_idx]

    immune = last_events[last_events.disease_state == "recovered"].groupby("tract_fips").size()
    infected = last_events[last_events.disease_state.isin(_ACTIVE_STATES)].groupby("tract_fips").size()

    grid_stats_df = demog_df.rename(columns={"fips": "tract_fips", "total": "pop"})[["tract_fips", "pop"]].copy()
    grid_stats_df = grid_stats_df.set_index("tract_fips")
    grid_stats_df["immune"] = immune
    grid_stats_df["infected"] = infected
    grid_stats_df = grid_stats_df.fillna(0)
    # A small number of tracts can end up with pop < immune+infected (an agent's last event before
    # the cutoff landed in a different tract than earlier events for that same agent -- Epicast's
    # location_id records where each transition happened, not a fixed home tract). Clip rather than
    # let those tracts go negative.
    grid_stats_df["never_infected"] = (grid_stats_df["pop"] - grid_stats_df["immune"] - grid_stats_df["infected"]).clip(lower=0)
    grid_stats_df = grid_stats_df.reset_index()

    grid_stats_df["GEOID10"] = grid_stats_df["tract_fips"].astype("int64")
    grid_stats_df = grid_stats_df[["GEOID10", "pop", "never_infected", "infected", "immune"]]
    if county_level:
        grid_stats_df = aggregate_to_county(grid_stats_df)
    return grid_stats_df, day


def load_epicast_grid_stats(events_file, day=None, county_level=False):
    """Read an Epicast run.events.bin file and return (grid_stats_df, day) for a single day -- a
    convenience wrapper around read_events_bin + reconstruct_epicast_snapshot for callers that only
    need one day from one file. See reconstruct_epicast_snapshot for details.
    """
    print("Reading Epicast data from", events_file)
    events_df, demog_df = read_events_bin(events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")
    return reconstruct_epicast_snapshot(events_df, demog_df, day=day, county_level=county_level)


def main():
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 24

    parser = argparse.ArgumentParser(description="Plot Epicast infection-spread choropleth")
    parser.add_argument("--events_file", "-p", required=True, help="Epicast run.events.bin file")
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census TRACT shape files (.shp) by default -- Epicast's finest geographic unit is "
        "the tract, not the block group -- or Census COUNTY shape files if --county_level is "
        "passed. Available from\n"
        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Census+Tracts",
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
        "--day",
        "-d",
        type=int,
        nargs="+",
        default=[None],
        help="One or more 0-based days to plot (default: the last day in the events file). A "
        "single day plots one choropleth as before; multiple days are plotted as a horizontal "
        "sequence, one panel per day, all reconstructed from the SAME events file (read once).",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Aggregate and plot at the Census county level (5-digit GEOID, summed across all "
        "tracts in each county) instead of the default tract level. Pass a Census county "
        "shapefile (not a tract one) via --shape_files when using this.",
    )

    args = parser.parse_args()

    geo_unit = "county" if args.county_level else "tract"
    geo_unit_pl = "counties" if args.county_level else "tracts"

    print("Reading Epicast data from", args.events_file)
    events_df, demog_df = read_events_bin(args.events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

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
    max_count = 30000

    example = "tl_2010_35_county10.shp" if args.county_level else "tl_2010_35_tract10.shp"

    # Reconstruct and merge each requested day independently (from the SAME already-loaded events
    # file), so a sequence of several days becomes a horizontal row of panels (a single day is just
    # the N=1 case of the same loop).
    panels = []
    for day in args.day:
        grid_stats_df, resolved_day = reconstruct_epicast_snapshot(
            events_df, demog_df, day=day, county_level=args.county_level
        )
        df = pd.merge(shp_data, grid_stats_df, on=["GEOID10"], how="inner")
        if df.empty:
            raise SystemExit(
                f"No rows matched after merging day {resolved_day}: 0 of {len(grid_stats_df)} "
                f"Epicast {geo_unit} GEOIDs were found among the {len(shp_data)} shapefile rows. "
                f"Check that --shape_files is a Census {geo_unit.upper()} shapefile (e.g. {example}) "
                f"covering the same state as the events file."
            )
        panels.append((df, f"Day {resolved_day}"))

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
    plt.tight_layout()
    print("Plotting results to", args.output)
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
