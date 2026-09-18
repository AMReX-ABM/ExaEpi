#!/usr/bin/env python

"""Plot ExaEpi and/or Epicast infection-spread choropleths over a sequence of days.

Pass --exaepi_files alone to plot only ExaEpi (one row), --events_file alone to plot only Epicast
(one row), or both together to plot them stacked in two rows (Epicast on top, ExaEpi below) with
each day's column additionally labeled with that day's log-scale Pearson r and RMSLE of raw infected
COUNT (see compare_day() for how these are computed and what they mean, and why they're shown here
instead of the rate-based rho/r/RMSE compare_day() also returns -- those can look deceptively good
once an epidemic has mostly burned out) -- a per-community comparison that plot_geo_compare.py also
uses (importing the loaders and compare_day from here) to plot the rate-based quantities as a
day-scalar time series instead.

Days are given explicitly as a list (--day), since a single day value has to resolve independently
to an Epicast snapshot (reconstructed from the events log) and/or a matching ExaEpi per-day file
(looked up by the day parsed from its name) -- there's no single natural sequence of "days" shared
by both data sources the way there is for ExaEpi's own per-day files alone.

Epicast's finest geographic unit is the Census tract, not the block group ExaEpi communities use --
so ExaEpi is aggregated up to the tract by default (or further to the county, with
--county_level), and a tract (or county) shapefile is required via --shape_files, not a block group
one, whenever Epicast is involved.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib
from scipy.stats import spearmanr

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_agg_utils import aggregate_to_county  # noqa: E402
from read_epicast_events import read_events_bin  # noqa: E402
from plos_compbio_style import (  # noqa: E402
    apply_style,
    FONT_TICK,
    FONT_TITLE,
    AXES_LINEWIDTH,
    FULL_PAGE_WIDTH_IN,
)


def _parse_day_from_filename(fname):
    """Extract the trailing step/day number from an ExaEpi per-day file/dir name, e.g.
    'cases00050' or 'cases00050/' -> 50.
    """
    m = re.search(r"(\d+)$", fname.rstrip("/"))
    if not m:
        raise SystemExit(f"Could not parse a trailing day/step number from: {fname}")
    return int(m.group(1))


def load_exaepi_grid_stats(csv_path, tract_level=False, county_level=False):
    """Read one of ExaEpi's lightweight aggregated-diagnostics CSV files (written directly by the
    simulation via --aggregated_diag_int, see ExaEpi::IO::writeAggregatedData in src/IO.cpp) and
    return a per-community DataFrame with columns: GEOID10, pop, never_infected, infected, immune
    -- aggregated up to the Census tract level if tract_level is set, or further to the county
    level if county_level is set (which takes precedence over tract_level if both are set).
    """
    print("Reading ExaEpi aggregated diagnostic data from", csv_path)
    grid_stats_df = pd.read_csv(csv_path)
    grid_stats_df = grid_stats_df.rename(columns={"GEOID": "GEOID10", "total": "pop"})
    grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"].astype("int64")

    if county_level:
        grid_stats_df = aggregate_to_county(grid_stats_df, input_level="block_group")
    elif tract_level:
        # Drop the last digit (the block group number) to get the 11-digit Census tract GEOID,
        # then sum every block group that shares a tract into one row before merging with a
        # tract-level shapefile (otherwise each block group would re-attach to the same tract
        # geometry and inflate the per-tract counts).
        grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"] // 10
        grid_stats_df = grid_stats_df.groupby("GEOID10", as_index=False)[
            ["pop", "never_infected", "infected", "immune"]
        ].sum()
    return grid_stats_df


def load_exaepi_day_night_population(csv_path):
    """Read the static, once-per-run <prefix>_day_night_population.csv ExaEpi writes when
    --aggregated_diag_int is enabled (see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp) and
    return a per-community DataFrame with columns: GEOID10, night_pop, night_workers,
    night_students, day_pop, day_workers, day_students -- everyone/workers/students counted by
    home cell (night) and by work/school cell (day). Unlike load_exaepi_grid_stats, this is not
    aggregated up to tract/county level here -- callers needing that should roll it up themselves
    the same way (see geo_agg_utils.aggregate_to_county and load_exaepi_grid_stats's tract-level
    groupby for the pattern), since which columns to sum depends on which of the 6 they actually
    need.
    """
    print("Reading ExaEpi day/night population data from", csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "GEOID": "GEOID10",
            "night_total": "night_pop",
            "day_total": "day_pop",
        }
    )
    df["GEOID10"] = df["GEOID10"].astype("int64")
    return df


_ACTIVE_STATES = {"exposed", "presymptomatic", "symptomatic", "asymptomatic"}


def reconstruct_epicast_snapshot(events_df, demog_df, day=None, county_level=False):
    """Given already-loaded Epicast events/demographics (see read_events_bin), reconstruct a
    snapshot DataFrame (columns GEOID10, pop, never_infected, infected, immune) as of the START of
    the given 0-based day (default: the last day in the data; clamped if it exceeds that),
    aggregated up to the county level if county_level is set (Epicast's native granularity is the
    tract). Returns (grid_stats_df, day).

    Epicast's run.events.bin file has no per-timestep snapshot the way ExaEpi's own per-day output
    does -- it's a log of AgentTransition events (one row per disease_state change), two timesteps
    (a "day" half-step and a "night" half-step) per calendar day. To get a "snapshot as of the
    start of day D" comparable to ExaEpi's own day-D data (written before day D's own dynamics run
    -- day 0 is the raw seed state, zero elapsed transmission), this reconstructs each agent's most
    recent disease_state (and the tract they were in when that transition happened) among all their
    events with timestep <= cutoff, then buckets agents by that tract:
        immune         = last state is "recovered"
        infected       = last state is exposed/presymptomatic/symptomatic/asymptomatic (still active)
        never_infected = tract population (from the file's demographics) minus the above two
    Agents with zero events by the cutoff never appear in the reconstruction and are implicitly
    counted as never_infected via that subtraction.

    The cutoff is day D's day-half timestep (2*D) minus one -- i.e. everything through day D-1's
    night-half -- EXCEPT day 0, which has no "day -1" to stop after: day 0's day-half (timestep 0)
    IS the initial seeding itself (the same agents/tracts as ExaEpi's plt00000), so day 0 stops
    right there instead, at timestep 0. Without this exception (i.e. the plain 2*day-1 formula
    extended to day 0), day 0 would already include day 0's own night-half dynamics -- which lets
    already-seeded agents' disease-state progression get logged from wherever they physically are
    at that later timestep (e.g. a commuter's workplace tract), so tracts/counties with no seeded
    infections of their own can appear "infected" at what's supposed to be the starting snapshot.
    """
    max_day = (int(events_df.timestep.max()) + 1) // 2
    day = max_day if day is None else day
    if day > max_day:
        print(f"WARNING: requested day {day} exceeds the last available day ({max_day}); using {max_day} instead")
        day = max_day
    cutoff_timestep = 0 if day == 0 else 2 * day - 1
    print(f"Reconstructing snapshot at day {day} (timestep <= {cutoff_timestep})")

    # Reconstruct each agent's most recent disease_state (and the tract of that transition) among
    # events at or before the cutoff -- see the docstring above for why this, rather than a simple
    # per-column aggregate, is needed to get a snapshot-like view out of a transition log.
    sub = events_df[events_df.timestep <= cutoff_timestep]
    last_idx = sub.groupby("true_agent_id")["timestep"].idxmax()
    last_events = sub.loc[last_idx]

    immune = last_events[last_events.disease_state == "recovered"].groupby("tract_fips").size()
    infected = last_events[last_events.disease_state.isin(_ACTIVE_STATES)].groupby("tract_fips").size()

    grid_stats_df = _grid_stats_from_counts(demog_df, immune, infected, county_level)
    return grid_stats_df, day


def _grid_stats_from_counts(demog_df, immune, infected, county_level):
    """Build the (GEOID10, pop, never_infected, infected, immune) snapshot DataFrame given per-tract
    immune/infected counts (Series indexed by tract_fips) -- shared by reconstruct_epicast_snapshot
    and iter_epicast_snapshots so both apply identical never_infected/clipping/aggregation logic.
    """
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
        grid_stats_df = aggregate_to_county(grid_stats_df, input_level="tract")
    return grid_stats_df


def iter_epicast_snapshots(events_df, demog_df, days, county_level=False):
    """Like reconstruct_epicast_snapshot, but reconstructs a whole sequence of daily snapshots in
    one pass, reusing state across days instead of restarting from scratch each time.

    reconstruct_epicast_snapshot always re-derives every agent's last known disease_state from
    timestep 0, so calling it once per day in a loop (as plot_gini_timeseries.py does) redoes all
    of the earlier days' work again on every later call -- the per-call cost grows with the day
    being reconstructed, so such a loop is quadratic in the number of days overall. This instead
    sorts the event log by timestep once, then walks it a single time, only re-deriving the last
    state for agents whose events fall in the (small) slice between the previous and current
    cutoff, and maintains running per-tract immune/infected counts incrementally rather than
    re-aggregating from every event seen so far -- so the whole sequence costs O(total events)
    once, not O(total events) per requested day.

    `days` may be given in any order and may repeat; each is independently clamped to the last
    available day exactly as reconstruct_epicast_snapshot does. Yields (day, grid_stats_df) pairs,
    one per entry of `days`, in the same order as `days` itself.
    """
    max_day = (int(events_df.timestep.max()) + 1) // 2
    resolved = []
    for requested_day in days:
        day = requested_day
        if day > max_day:
            print(f"WARNING: requested day {day} exceeds the last available day ({max_day}); using {max_day} instead")
            day = max_day
        resolved.append(day)

    resolved_cutoffs = [0 if d == 0 else 2 * d - 1 for d in resolved]
    cutoffs = sorted(set(resolved_cutoffs))
    day_by_cutoff = {(0 if d == 0 else 2 * d - 1): d for d in resolved}

    sorted_events = events_df.sort_values("timestep", kind="stable")
    timesteps = sorted_events["timestep"].to_numpy()
    agent_ids = sorted_events["true_agent_id"].to_numpy()
    disease_states = sorted_events["disease_state"].to_numpy()
    tract_fips = sorted_events["tract_fips"].to_numpy()
    n = len(sorted_events)

    # Per-agent last-known state as of the current cutoff, held as dense arrays indexed directly by
    # true_agent_id (Epicast assigns these as small dense integers, not sparse/hashed IDs) so that
    # both looking up an agent's previous state and recording its new one are vectorized numpy
    # gather/scatter operations over just this chunk's agents, not a Python-level loop -- with a
    # dict-based version of this same state, those two loops dominated the total run time (their
    # per-iteration cost is tiny, but there can be millions of iterations in a single chunk during
    # an outbreak's peak, and the interpreter overhead of a Python loop adds up fast at that scale).
    n_agents = int(agent_ids.max()) + 1
    state_tract = np.full(n_agents, -1, dtype=np.int64)  # -1 = agent not seen yet
    state_category = np.zeros(n_agents, dtype=np.int8)  # meaningful only where state_tract != -1; 1=immune, 2=infected
    immune_counts = pd.Series(dtype=float)
    infected_counts = pd.Series(dtype=float)

    snapshots = {}
    pos = 0
    next_to_yield = 0
    for cutoff in cutoffs:
        end = pos
        while end < n and timesteps[end] <= cutoff:
            end += 1
        if end > pos:
            # Resolve each agent's last state within just this slice of new events (mirroring the
            # groupby("true_agent_id")["timestep"].idxmax() in reconstruct_epicast_snapshot, but
            # over the much smaller slice instead of the whole cumulative event set).
            chunk_df = pd.DataFrame({
                "true_agent_id": agent_ids[pos:end],
                "timestep": timesteps[pos:end],
                "disease_state": disease_states[pos:end],
                "tract_fips": tract_fips[pos:end],
            })
            chunk_last = chunk_df.loc[chunk_df.groupby("true_agent_id")["timestep"].idxmax()].set_index("true_agent_id")

            target_ids = chunk_last.index.to_numpy()
            new_tract = chunk_last["tract_fips"].to_numpy()
            new_category = np.where(
                chunk_last["disease_state"].to_numpy() == "recovered",
                1,
                np.where(chunk_last["disease_state"].isin(_ACTIVE_STATES).to_numpy(), 2, 0),
            ).astype(np.int8)

            # Gather this chunk's agents' PREVIOUS state (a vectorized numpy gather, not a Python
            # loop over potentially millions of agents) so its contribution to the running counts
            # can be undone before the new state below is applied.
            old_tract = state_tract[target_ids]
            old_category = state_category[target_ids]
            has_old = old_tract != -1
            if has_old.any():
                old_tract_seen = old_tract[has_old]
                old_category_seen = old_category[has_old]
                immune_counts = immune_counts.subtract(
                    pd.Series(old_tract_seen[old_category_seen == 1]).value_counts(), fill_value=0
                )
                infected_counts = infected_counts.subtract(
                    pd.Series(old_tract_seen[old_category_seen == 2]).value_counts(), fill_value=0
                )

            immune_counts = immune_counts.add(pd.Series(new_tract[new_category == 1]).value_counts(), fill_value=0)
            infected_counts = infected_counts.add(
                pd.Series(new_tract[new_category == 2]).value_counts(), fill_value=0
            )

            # Record the new state -- a vectorized scatter; target_ids has no duplicates here since
            # chunk_last already holds exactly one (last-by-timestep) row per agent.
            state_tract[target_ids] = new_tract
            state_category[target_ids] = new_category

            pos = end

        day = day_by_cutoff[cutoff]
        print(f"Reconstructing snapshot at day {day} (timestep <= {cutoff})")
        snapshots[cutoff] = _grid_stats_from_counts(demog_df, immune_counts, infected_counts, county_level)

        # Yield any requested days that are now ready, in their original request order -- for the
        # common case of non-decreasing requested days this means each day is handed back right
        # after its own (incremental) computation, instead of only after the entire sequence of
        # cutoffs has been processed.
        while next_to_yield < len(resolved) and resolved_cutoffs[next_to_yield] in snapshots:
            yield resolved[next_to_yield], snapshots[resolved_cutoffs[next_to_yield]]
            next_to_yield += 1

    while next_to_yield < len(resolved):
        yield resolved[next_to_yield], snapshots[resolved_cutoffs[next_to_yield]]
        next_to_yield += 1


def _is_aggregated_file(path):
    """An ExaEpi aggregated-diagnostics file (see ExaEpi::IO::writeAggregatedData / --
    load_exaepi_grid_stats) is a plain file whose first line is the fixed CSV header this reader
    expects -- check that, rather than just the filename, to tell it apart from an unrelated file
    a glob/parent-directory listing might also pick up.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            return f.readline().rstrip("\n") == "GEOID,total,never_infected,infected,immune"
    except UnicodeDecodeError:
        # Not a text file at all (e.g. a compressed or binary file living alongside the CSVs in
        # the same directory) -- not an aggregated-diagnostics file, but not an error either.
        return False


def expand_aggregated_files(paths):
    """Expand each of `paths` into the individual ExaEpi aggregated-diagnostics CSV files it refers
    to (see load_exaepi_grid_stats), so callers can point at a whole run's worth of output without
    listing every cases* file by hand. Each entry in `paths` may be: a single CSV file (e.g.
    cases00050), a parent directory containing many such files (e.g. a run's output directory), or
    a glob pattern (e.g. "results/cases*"). Returns the resulting paths deduplicated and sorted by
    the day parsed from their name.
    """
    expanded = []
    for path in paths:
        path = path.rstrip("/")
        if _is_aggregated_file(path):
            expanded.append(path)
        elif os.path.isdir(path):
            children = sorted(
                os.path.join(path, name) for name in os.listdir(path) if _is_aggregated_file(os.path.join(path, name))
            )
            if not children:
                raise SystemExit(f"No aggregated-diagnostics CSV files found under {path}")
            expanded.extend(children)
        else:
            matches = sorted(p for p in glob.glob(path) if _is_aggregated_file(p))
            if not matches:
                raise SystemExit(f"No aggregated-diagnostics CSV files matched: {path}")
            expanded.extend(matches)

    seen = set()
    unique = [d for d in expanded if not (d in seen or seen.add(d))]
    unique.sort(key=_parse_day_from_filename)
    return unique


def weighted_pearsonr(x, y, w):
    """Weighted Pearson correlation coefficient between x and y, weighted by w. Unlike plain
    Pearson r, a community's contribution to the correlation scales with its weight -- so a handful
    of low-weight communities disagreeing doesn't move r as much as a handful of high-weight ones
    would.
    """
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    wsum = w.sum()
    xbar = (w * x).sum() / wsum
    ybar = (w * y).sum() / wsum
    cov_xy = (w * (x - xbar) * (y - ybar)).sum()
    var_x = (w * (x - xbar) ** 2).sum()
    var_y = (w * (y - ybar) ** 2).sum()
    return cov_xy / np.sqrt(var_x * var_y)


def weighted_rmse(x, y, w):
    """Weighted root-mean-square of (x - y), weighted by w -- a direct magnitude-of-disagreement
    metric (not a correlation), so it isn't fooled by two similar values swapping relative order
    and isn't blind to a systematic offset between x and y the way a correlation coefficient is.
    """
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    return np.sqrt((w * (x - y) ** 2).sum() / w.sum())


def compare_day(exaepi_df, epicast_df):
    """Merge one day's ExaEpi and Epicast per-community DataFrames on GEOID10 and return
    (rho, pval, r, rmse, r_log, rmse_log, n, merged_df).

    rho/r/rmse are the Spearman rank correlation, infection-weighted Pearson correlation, and
    infection-weighted RMSE of infection RATE (infected / pop) between the two. Rate rather than raw
    infected count is compared so that communities of very different population size are compared on
    a like-for-like basis. r and rmse are weighted by each community's average infected count
    (across the two simulators) rather than its population, so that a community currently at or near
    zero infection doesn't get outsized influence just because it has a large population -- a rate
    difference there is mostly noise, whereas the same difference in a heavily-infected community
    reflects a real, larger-magnitude disagreement. The corollary is that once an epidemic has mostly
    burned out and every community's rate is near zero, rmse necessarily shrinks toward zero right
    along with it (two numbers close to zero can't differ by much in absolute terms) even if the two
    simulators agree poorly on which of the few remaining cases are where -- rmse alone can look
    deceptively good late in a run.

    r_log/rmse_log are a log-scale counterpart computed on raw infected COUNT (not rate), unweighted
    (every community counted equally), matching what the choropleth itself actually shows: it colors
    every community by log(infected count) with no population weighting, so a small community's count
    going from 2 to 20 is exactly as visible on the map as a large community's count going from 200 to
    2000 -- a difference the rate-based rmse above washes out once both counts are small relative to
    population. rmse_log is the root-mean-square log error (RMSLE), on log1p(count) so a count of 0 is
    still defined; unlike rmse, it stays sensitive to disagreement in the residual tail of a mostly-
    resolved epidemic, which is exactly where rmse's near-zero-by-construction floor is least
    informative. rho isn't given a log counterpart since Spearman rank correlation is unchanged by any
    monotonic transform (log included) of the values it ranks.

    merged_df carries a rate_exaepi/rate_epicast column per matched community (GEOID10) -- the
    community-by-community comparison underlying the summary scalars, for callers that want to look
    beyond them. Returns (None, None, None, None, None, None, n, merged_df) if fewer than two
    communities match, since none of these are meaningful below that.
    """
    df = pd.merge(exaepi_df, epicast_df, on="GEOID10", suffixes=("_exaepi", "_epicast"))
    df = df[(df.pop_exaepi > 0) & (df.pop_epicast > 0)].copy()
    df["rate_exaepi"] = df.infected_exaepi / df.pop_exaepi
    df["rate_epicast"] = df.infected_epicast / df.pop_epicast
    if len(df) < 2:
        return None, None, None, None, None, None, len(df), df
    rho, pval = spearmanr(df.rate_exaepi, df.rate_epicast)
    weight = (df.infected_exaepi + df.infected_epicast) / 2.0
    r = weighted_pearsonr(df.rate_exaepi, df.rate_epicast, weight)
    rmse = weighted_rmse(df.rate_exaepi, df.rate_epicast, weight)

    log_exaepi = np.log1p(df.infected_exaepi)
    log_epicast = np.log1p(df.infected_epicast)
    equal_weight = np.ones(len(df))
    r_log = weighted_pearsonr(log_exaepi, log_epicast, equal_weight)
    rmse_log = weighted_rmse(log_exaepi, log_epicast, equal_weight)

    return rho, pval, r, rmse, r_log, rmse_log, len(df), df


def main():
    apply_style()

    parser = argparse.ArgumentParser(
        description="Plot ExaEpi and/or Epicast choropleths, one column per day. With both given, "
        "rows are stacked (Epicast on top, ExaEpi below) and each column is labeled with that "
        "day's log-scale Pearson r and RMSLE (count-based); with only one given, that single row "
        "is plotted with no stats."
    )
    parser.add_argument(
        "--exaepi_files",
        "-g",
        nargs="+",
        default=None,
        help="Where to find ExaEpi's aggregated-diagnostics CSV files (e.g. cases00050, written "
        "via --aggregated_diag_int -- see load_exaepi_grid_stats). Pass a parent directory "
        "containing many such files, individual files, or a glob pattern. Which specific day(s) "
        "get plotted is chosen by --day, not by which paths match here -- this just needs to cover "
        "them. At least one of --exaepi_files/--events_file is required.",
    )
    parser.add_argument(
        "--events_file",
        "-f",
        default=None,
        help="Epicast run.events.bin file. At least one of --exaepi_files/--events_file is required.",
    )
    parser.add_argument(
        "--day",
        "-d",
        type=int,
        nargs="+",
        default=[None],
        help="One or more 0-based days to plot, one column each (default: the last day available). "
        "Each day is used to reconstruct the Epicast snapshot and/or look up the matching ExaEpi "
        "CSV file (by the day parsed from its filename), whichever apply.",
    )
    parser.add_argument(
        "--shape_files",
        "-s",
        required=True,
        nargs="+",
        help="Census shape files (.shp) at the granularity being plotted -- tract by default, or "
        "county if --county_level is passed. Block group shapefiles only work in ExaEpi-only mode "
        "without --tract_level/--county_level.",
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
        help="Range for longitude/latitude: lon_min lon_max lat_min lat_max",
    )
    parser.add_argument(
        "--tract_level",
        "-t",
        action="store_true",
        default=False,
        help="Aggregate ExaEpi up to the Census tract level (ignored -- always on -- whenever "
        "--events_file is given, since Epicast is natively tract-level). Only meaningful in "
        "ExaEpi-only mode, where the default is the finer Census block group level.",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Aggregate/plot at the Census county level instead of the default (Census tract, or "
        "block group in ExaEpi-only mode without --tract_level). Takes precedence over "
        "--tract_level. Pass a matching county shapefile via --shape_files.",
    )
    args = parser.parse_args()

    if not args.exaepi_files and not args.events_file:
        parser.error("At least one of --exaepi_files/--events_file must be given")

    both = bool(args.exaepi_files) and bool(args.events_file)
    uses_epicast = bool(args.events_file)
    tract_level = (not args.county_level) if (uses_epicast or args.tract_level) else False
    geo_unit = "county" if args.county_level else ("tract" if tract_level else "block group")
    geo_unit_pl = "counties" if args.county_level else ("tracts" if tract_level else "block groups")
    example = {"county": "tl_2010_35_county10.shp", "tract": "tl_2010_35_tract10.shp", "block group": "tl_2010_35_bg10.shp"}[
        geo_unit
    ]

    events_df = demog_df = None
    if args.events_file:
        print("Reading Epicast data from", args.events_file)
        events_df, demog_df = read_events_bin(args.events_file)
        print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

    day_to_file = {}
    if args.exaepi_files:
        exaepi_files = expand_aggregated_files(args.exaepi_files)
        day_to_file = {_parse_day_from_filename(f): f for f in exaepi_files}
        print(f"Found {len(day_to_file)} ExaEpi days:", sorted(day_to_file))

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

    # rows_spec fixes the row order (Epicast above ExaEpi when both are present) and which data
    # source feeds each row; only one row is used when only one data source was given.
    rows_spec = []
    if args.events_file:
        rows_spec.append(("Epicast", "epicast"))
    if args.exaepi_files:
        rows_spec.append(("ExaEpi", "exaepi"))

    # For each requested day, reconstruct/load whichever data source(s) were given, compute the
    # rho/r/RMSE comparison (only when both are present), and merge onto the shapefile geometry.
    # panels holds one (exaepi_geo_df_or_None, epicast_geo_df_or_None, label) tuple per column.
    panels = []
    for day in args.day:
        epicast_df = None
        if args.events_file:
            epicast_df, resolved_day = reconstruct_epicast_snapshot(
                events_df, demog_df, day=day, county_level=args.county_level
            )
            if day is not None and resolved_day != day:
                print(f"WARNING: requested day {day}, but Epicast clamped it to day {resolved_day}")
        else:
            resolved_day = day if day is not None else max(day_to_file)

        exaepi_df = None
        if args.exaepi_files:
            if resolved_day not in day_to_file:
                available = ", ".join(str(d) for d in sorted(day_to_file))
                raise SystemExit(
                    f"No ExaEpi data found for day {resolved_day} among --exaepi_files. Available "
                    f"days: {available}"
                )
            csv_path = day_to_file[resolved_day]
            exaepi_df = load_exaepi_grid_stats(csv_path, tract_level=tract_level, county_level=args.county_level)

        if both:
            _, _, _, _, r_log, rmse_log, n, _ = compare_day(exaepi_df, epicast_df)
            if r_log is None:
                stats_str = f"(only {n} matched {geo_unit_pl})"
            else:
                stats_str = f"log r={r_log:.2f}\nRMSLE={rmse_log:.2f}"
        else:
            stats_str = None
        label = (f"Day {resolved_day}", stats_str)

        exaepi_geo_df = pd.merge(shp_data, exaepi_df, on=["GEOID10"], how="inner") if exaepi_df is not None else None
        epicast_geo_df = (
            pd.merge(shp_data, epicast_df, on=["GEOID10"], how="inner") if epicast_df is not None else None
        )
        for geo_df in (exaepi_geo_df, epicast_geo_df):
            if geo_df is not None and geo_df.empty:
                raise SystemExit(
                    f"No rows matched after merging day {resolved_day}: check --shape_files is a "
                    f"Census {geo_unit.upper()} shapefile (e.g. {example}) covering the same state "
                    f"as the data."
                )
        panels.append((exaepi_geo_df, epicast_geo_df, label))

    # Bounds are the union across every panel's data (every row), so the whole grid shares one
    # consistent geographic extent instead of each panel framing itself differently.
    all_geo_dfs = [df for pair in panels for df in pair[:2] if df is not None]
    # Bounds come from the polygons' own geometry, NOT from the INTPTLON10/INTPTLAT10 columns:
    # those are each polygon's internal point (roughly its centroid), which sits an arbitrary
    # distance inside its own edge. At county level the huge desert counties' centroids are so far
    # from the state line that even a generous pad still cropped California's whole eastern edge
    # off the plot; tract centroids happened to land close enough to the edge to hide the bug.
    bounds = np.array([df.total_bounds for df in all_geo_dfs])  # minx, miny, maxx, maxy per panel
    pad = 0.1
    xmin = max(float(args.coord_bounds[0]), bounds[:, 0].min() - pad)
    xmax = min(float(args.coord_bounds[1]), bounds[:, 2].max() + pad)
    xrange = xmax - xmin
    ymin = max(float(args.coord_bounds[2]), bounds[:, 1].min() - pad)
    ymax = min(float(args.coord_bounds[3]), bounds[:, 3].max() + pad)
    yrange = ymax - ymin

    n = len(panels)
    num_rows = len(rows_spec)
    # Total figure width is fixed at the paper's full-page width regardless of how many day
    # columns there are -- each column just gets narrower as more days are added, rather than the
    # whole figure growing past the page (see paper_style.py).
    panel_width = FULL_PAGE_WIDTH_IN / n
    fig_x = FULL_PAGE_WIDTH_IN
    map_height = num_rows * panel_width * yrange / xrange

    # The title (1-3 lines, depending on whether stats are shown) sits in a margin ABOVE the map
    # grid, not inside it -- so map_height above is exactly the maps' own height only if that
    # margin is added on top of it. Without this, the fixed total figure height would force
    # constrained_layout to steal room from the maps themselves to fit the title, shrinking them
    # (they'd stay letterboxed to the right aspect ratio, just smaller, with dead space around
    # them) -- exactly the "too small" problem being fixed here.
    max_title_lines = max((label[0] + "\n" + (label[1] or "")).count("\n") + 1 for _, _, label in panels)
    # FONT_TITLE because that's what the titles below are drawn at -- set_title() inherits it from
    # apply_style()'s axes.titlesize rather than naming it, so this is the one place it's spelled out.
    title_pt = FONT_TITLE * 1.4 * max_title_lines + 6  # +6pt is matplotlib's own default title pad
    fig_y = map_height + title_pt / 72

    print(f"Plot dimensions: lng/lat {xmin}, {xmax}, {ymin}, {ymax}, figure size: {fig_x}, {fig_y}")

    fig, axes = plt.subplots(num_rows, n, figsize=(fig_x, fig_y), squeeze=False, layout="constrained")
    # Shrink constrained_layout's own default padding to a small margin -- its defaults leave more
    # breathing room than wanted here, at the direct expense of the maps' own size. wspace/hspace
    # are kept just above 0 rather than exactly 0: constrained_layout's solver doesn't handle
    # aspect-locked axes (geopandas plots are equal-aspect) cleanly at exactly zero spacing, and can
    # let an axes overflow past its cell -- clipping a map at the figure edge -- instead of shrinking
    # it to fit.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.002, hspace=0.002)

    norm = mp.colors.LogNorm(vmin=1.0, vmax=max_count)
    for j, (exaepi_geo_df, epicast_geo_df, label) in enumerate(panels):
        for row, (row_name, which) in enumerate(rows_spec):
            geo_df = epicast_geo_df if which == "epicast" else exaepi_geo_df
            ax = axes[row][j]
            states.boundary.plot(ax=ax, lw=AXES_LINEWIDTH, color="black")
            geo_df.plot(ax=ax, column="infected", cmap="OrRd", legend=False, norm=norm)  # type: ignore
            ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
            ax.set_frame_on(False)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            if j == 0:
                ax.set_ylabel(row_name)
        day_str, stats_str = label
        # A single multi-line title (matplotlib spaces embedded newlines correctly on its own)
        # rather than a separately-positioned second text object -- that manual positioning was
        # tuned for a much larger font scale and stopped fitting once these panels shrank to their
        # PLOS print size.
        title = f"{day_str}\n{stats_str}" if stats_str else day_str
        axes[0][j].set_title(title, linespacing=1.4)

    # A single colorbar spanning every row, rather than one per panel -- built from an explicit
    # ScalarMappable (since legend=False above) and handed every axes in the grid so matplotlib
    # sizes/positions it to span the full height instead of attaching to just one panel.
    sm = mp.cm.ScalarMappable(norm=norm, cmap="OrRd")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
