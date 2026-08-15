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

_ACTIVE_STATES = {"exposed", "presymptomatic", "symptomatic", "asymptomatic"}


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
        help="Census TRACT shape files (.shp) -- Epicast's finest geographic unit is the tract, "
        "not the block group. Available from\n"
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
        default=None,
        help="0-based day to plot the snapshot for (default: the last day in the events file)",
    )

    args = parser.parse_args()

    print("Reading Epicast data from", args.events_file)
    events_df, demog_df = read_events_bin(args.events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

    max_day = int(events_df.timestep.max() // 2)
    day = max_day if args.day is None else args.day
    if day > max_day:
        print(f"WARNING: requested day {day} exceeds the last available day ({max_day}); using {max_day} instead")
        day = max_day
    cutoff_timestep = 2 * day + 1
    print(f"Plotting snapshot at day {day} (timestep <= {cutoff_timestep})")

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
    # grid_stats_df.to_csv("grid_stats.csv")

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
    print("Read in", len(shp_data), "Census tracts")

    states = gp.read_file(args.states_file)
    states = states[states.STATE.isin(state_codes)]
    max_count = 30000

    df = pd.merge(shp_data, grid_stats_df, on=["GEOID10"], how="inner")
    if df.empty:
        raise SystemExit(
            f"No rows matched after merging: 0 of {len(grid_stats_df)} Epicast tract GEOIDs were "
            f"found among the {len(shp_data)} shapefile rows. Check that --shape_files is a Census "
            f"TRACT shapefile (e.g. tl_2010_35_tract10.shp) covering the same state as the events file."
        )

    xmin = max(float(args.coord_bounds[0]), float(df.INTPTLON10.astype("float").min()) - 0.5)
    xmax = min(float(args.coord_bounds[1]), float(df.INTPTLON10.astype("float").max()) + 0.5)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), float(df.INTPTLAT10.astype("float").min()) - 0.5)
    ymax = min(float(args.coord_bounds[3]), float(df.INTPTLAT10.astype("float").max()) + 0.5)
    yrange = ymax - ymin

    fig_x = 12.0
    fig_y = float(fig_x) * yrange / xrange
    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, (ax1) = plt.subplots(1, 1, figsize=(fig_x, fig_y))

    status_list = {
        1: ["infected", ax1, "OrRd"],
    }

    for _, status in status_list.items():
        ax = status[1]
        states.boundary.plot(ax=ax, lw=1, color="black")
        # Some decent colormaps: RdPu OrRd Greys
        df.plot(
            ax=ax,
            column=status[0],
            cmap=status[2],
            legend=True,
            norm=mp.colors.LogNorm(vmin=1.0, vmax=max_count),  # type: ignore
        )
        ax.set_title(status[0].upper())
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        ax.set_frame_on(False)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    axes = fig.get_axes()
    for cb in axes[len(status_list):-1]:
        cb.remove()
    cb = axes[-1]
    cb.set_box_aspect(50)
    plt.tight_layout()
    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
