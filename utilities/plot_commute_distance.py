#!/usr/bin/env python

"""Plot the commute-distance distribution -- a histogram of the fraction of workers by home-to-work
distance, or with --cdf the cumulative fraction whose distance is at most d -- for ExaEpi, Epicast
and the raw LODES data both are built from.

Distances are great-circle distances between Census tract internal points (INTPTLAT10/INTPTLON10,
roughly each tract's centroid). Every source is compared at tract level, the finest unit Epicast
resolves, so the curves are directly comparable:
  - ExaEpi: every agent's home_geoid/work_geoid block-group pair in the UrbanPop .bin the run
    reads, rolled up to tracts. A commuter is chosen by the same rule ExaEpi uses to place agents
    (UrbanPopData.cpp's agent initialization): anyone with a NAICS code except a declared
    work-from-home non-educator, who is kept at home for the day. Educators always go to their
    school. This is the same population plot_geo_daynight.py's --population workers maps, and
    rebuilding that script's day/night worker counts from these pairs matches ExaEpi's
    <prefix>_day_night_population.csv exactly.
  - Epicast: the home->work pairs Epicast actually generated, for the workers who were infected at
    work in one or more runs. Each ctx_work infection event records the
    workplace (tract + community); the home tract comes from the agent id, since Epicast numbers
    agents consecutively tract by tract in the same order as the events file's per-tract
    population table (checked against every household infection, whose location is the home).
    This is a sample, not the whole workforce -- use a high-attack-rate run for the most pairs --
    and it is slightly biased towards workplaces with more transmission and away from same-tract
    commuters, who are more likely to be infected at home first. (Epicast samples workplaces from
    its LODES-derived commute-flow table, and on CA and NM this sample's distances match that
    table's to within a few percent.)
  - LODES: the raw block-level OD data, rolled up to tracts. LODES counts jobs (JT00 = all jobs),
    not workers, so a worker with two jobs counts twice. It links a job to a residence from
    administrative records, so a pair is not necessarily a commute anyone makes daily.
  - CTPP (optional): tract-to-tract flows from the ACS question about where each worker worked the
    previous week (see download_ctpp_flows.py), so closer to the daily commutes the models
    simulate. Its universe includes people who worked at home, who add to the same-tract share.

A commute that starts and ends in the same tract has distance 0. On the default linear x axis it
falls in the first histogram bin, or is the CDF's step at x = 0. A log x axis (--logx) cannot show
it, but those commutes are still counted in every series' denominator: the histogram's bins then sum
to less than 1, and each CDF starts at its own same-tract fraction at the left edge rather than at 0.
The fractions are also printed with the other summary statistics.

All sources are in-state only (LODES "main" files; the "aux" files with cross-state commutes are
not used by either model).
"""

import os
import sys
import argparse
import struct
import zlib
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402
from read_epicast_events import read_events_records  # noqa: E402
from seirhd_params import AGENT_FIELD_WIDTHS  # noqa: E402

# Mean Earth radius (IUGG), for the haversine distance.
EARTH_RADIUS_KM = 6371.0088

# TRAVEL::_wfh in UrbanPopAgentStruct.H.
TRAVEL_WFH = 7

# Raw Epicast event context codes (see read_epicast_events._CONTEXT_MAP).
CTX_HOUSEHOLD = 0x00
CTX_WORK = 0x04

# numpy dtype for each field width in AGENT_FIELD_WIDTHS (all fields are signed).
_WIDTH_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}


def read_urbanpop_columns(path, fields):
    """Return {field: array over every agent} for the named agent fields of a UrbanPop .bin.

    Same layout read_urbanpop_age_fractions (seirhd_params.py) reads: a 40-byte header, an index
    with one entry per GEOID, then one frame per home GEOID with its agents stored column by
    column, so each field is one contiguous slice of the frame.
    """
    header_struct = struct.Struct("<2I 2I Q I I Q")
    widths = dict(AGENT_FIELD_WIDTHS)
    offsets = {}
    offset = 0
    for name, width in AGENT_FIELD_WIDTHS:
        offsets[name] = offset
        offset += width
    columns = {name: [] for name in fields}
    print("Reading ExaEpi agents from", path)
    with open(path, "rb") as f:
        (magic, _version, num_naics, _num_geoids, num_agents, record_size, codec,
         index_end) = header_struct.unpack(f.read(header_struct.size))
        if magic != 0x55504F50:
            sys.exit(f"error: {path} is not a UrbanPop .bin (bad magic number)")
        if codec not in (0, 1):
            sys.exit(f"error: {path} uses unsupported codec {codec}")
        if record_size != offset:
            sys.exit(f"error: {path} has {record_size}-byte agent records, expected {offset} -- "
                     f"AGENT_FIELD_WIDTHS in seirhd_params.py is out of date")
        index_struct = struct.Struct(f"<QQ III {num_naics}I")
        index = f.read(index_end - header_struct.size)
        for i in range(len(index) // index_struct.size):
            _geoid, frame_offset, nbytes, pop = index_struct.unpack_from(index, i * index_struct.size)[:4]
            if pop == 0:  # a work-only GEOID: nobody lives there, no frame
                continue
            f.seek(frame_offset)
            blob = f.read(nbytes)
            frame = zlib.decompress(blob) if codec == 1 else blob
            for name in fields:
                start = offsets[name] * pop
                columns[name].append(
                    np.frombuffer(frame[start:start + widths[name] * pop], dtype=_WIDTH_DTYPES[widths[name]]))
    columns = {name: np.concatenate(parts) for name, parts in columns.items()}
    print(f"Read {num_agents:,} agents")
    return columns


def exaepi_commuters(cols):
    """Return the boolean mask of agents ExaEpi sends to a workplace each day, from the naics,
    school_id and travel columns of read_urbanpop_columns: anyone with a NAICS code except a
    declared work-from-home non-educator (UrbanPopData.cpp's agent initialization)."""
    employed = cols["naics"] != -1
    wfh = employed & (cols["travel"] == TRAVEL_WFH) & (cols["school_id"] == 0)
    commuter = employed & ~wfh
    print(f"ExaEpi: {int(employed.sum()):,} employed, {int(wfh.sum()):,} work from home, "
          f"{int(commuter.sum()):,} commuters")
    return commuter


def load_exaepi_pairs(urbanpop_bin):
    """Return a DataFrame (home, work, count) of ExaEpi commuters per home/work tract pair. See the
    module docstring for who counts as a commuter."""
    cols = read_urbanpop_columns(urbanpop_bin, ["home_geoid", "work_geoid", "naics", "school_id", "travel"])
    commuter = exaepi_commuters(cols)
    # 12-digit block group -> 11-digit tract
    df = pd.DataFrame({"home": cols["home_geoid"][commuter] // 10, "work": cols["work_geoid"][commuter] // 10})
    return df.groupby(["home", "work"]).size().rename("count").reset_index()


def load_epicast_event_pairs(events_files):
    """Return a DataFrame (home, work, count) of the home/work tract pairs Epicast generated for
    the workers infected at work, pooled over events_files. See the module docstring for how the
    home tract is recovered; that mapping is checked against each file's household infections and
    the file is refused if it doesn't hold."""
    id_mask = ~(np.uint64(0b111111) << np.uint64(58))  # strip the home_state bits

    frames = []
    for path in events_files:
        print("Reading Epicast events from", path)
        records, demog = read_events_records(path)
        if "total" not in demog.columns:
            sys.exit(f"error: {path} has no per-tract 'total' population column to map agent ids to "
                     f"home tracts (demographic columns: {list(demog.columns[1:])})")
        ends = np.cumsum(demog["total"].to_numpy().astype(np.int64))
        fips = demog["fips"].to_numpy().astype(np.int64)

        def home_tract(agent_id):
            idx = np.searchsorted(ends, (agent_id & id_mask).astype(np.int64), side="right")
            return np.where(idx < len(fips), fips[np.minimum(idx, len(fips) - 1)], -1)

        context = records["context"]
        household = records[context == CTX_HOUSEHOLD]
        match = np.mean(home_tract(household["agent_id"]) == (household["location_id"] >> np.uint64(8)).astype(np.int64))
        if match < 0.999:
            sys.exit(f"error: {path}: agent id -> home tract mapping only matches {100 * match:.2f}% of "
                     f"household infections, so it can't be used to recover home tracts")

        work = records[context == CTX_WORK]
        # One work infection per agent is expected; drop any repeats (reinfections) so each agent's
        # commute counts once per run.
        _, first = np.unique(work["agent_id"], return_index=True)
        work = work[first]
        frames.append(pd.DataFrame({"home": home_tract(work["agent_id"]),
                                    "work": (work["location_id"] >> np.uint64(8)).astype(np.int64)}))
        print(f"  {len(records):,} events, {len(work):,} workers infected at work "
              f"(home tract mapping matches {100 * match:.3f}% of household infections)")
    df = pd.concat(frames).groupby(["home", "work"]).size().rename("count").reset_index()
    print(f"Epicast events: {int(df['count'].sum()):,} work-infected workers over {len(df):,} tract pairs")
    return df


def load_lodes_pairs(lodes_file):
    """Return a DataFrame (home, work, count) of LODES jobs per home/work tract pair."""
    print("Reading LODES OD data from", lodes_file)
    df = pd.read_csv(lodes_file, usecols=["h_geocode", "w_geocode", "S000"],
                     dtype={"h_geocode": np.int64, "w_geocode": np.int64, "S000": np.int64})
    # 15-digit block -> 11-digit tract
    df = pd.DataFrame({"home": df["h_geocode"] // 10000, "work": df["w_geocode"] // 10000, "count": df["S000"]})
    df = df.groupby(["home", "work"], as_index=False)["count"].sum()
    print(f"LODES: {int(df['count'].sum()):,} jobs over {len(df):,} tract pairs")
    return df


def load_ctpp_pairs(ctpp_file):
    """Return a DataFrame (home, work, count) of CTPP workers per home/work tract pair, from
    download_ctpp_flows.py's src, dst, flow CSV."""
    print("Reading CTPP flows from", ctpp_file)
    df = pd.read_csv(ctpp_file, usecols=["src", "dst", "flow"]).rename(
        columns={"src": "home", "dst": "work", "flow": "count"})
    df = df.groupby(["home", "work"], as_index=False)["count"].sum()
    print(f"CTPP: {int(df['count'].sum()):,} workers over {len(df):,} tract pairs")
    return df


def load_internal_points(shape_files):
    """Return a DataFrame (lat, lon) indexed by int64 GEOID10, from the internal-point attribute
    columns of Census shapefiles (the geometry itself is not needed, so it is not read)."""
    frames = []
    for fname in shape_files:
        print("Reading internal points from", fname)
        frames.append(gp.read_file(fname, ignore_geometry=True)[["GEOID10", "INTPTLAT10", "INTPTLON10"]])
    df = pd.concat(frames)
    return pd.DataFrame(
        {"lat": df["INTPTLAT10"].astype(float).to_numpy(), "lon": df["INTPTLON10"].astype(float).to_numpy()},
        index=df["GEOID10"].astype(np.int64).to_numpy(),
    )


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def pair_distances(pairs, points, label):
    """Return (distance_km, count) arrays for the pairs whose home and work tracts both have an
    internal point. Pairs that don't are dropped and reported -- a nonzero drop usually means the
    shapefile is the wrong vintage for these GEOIDs."""
    home = points.reindex(pairs["home"].to_numpy())
    work = points.reindex(pairs["work"].to_numpy())
    ok = home["lat"].notna().to_numpy() & work["lat"].notna().to_numpy()
    counts = pairs["count"].to_numpy()
    dropped = counts[~ok].sum()
    if dropped:
        print(f"WARNING: {label}: dropped {int(dropped):,} of {int(counts.sum()):,} commuters "
              f"({100 * dropped / counts.sum():.3f}%) whose home or work tract has no internal point")
    dist = haversine_km(home["lat"].to_numpy()[ok], home["lon"].to_numpy()[ok],
                        work["lat"].to_numpy()[ok], work["lon"].to_numpy()[ok])
    # Same-tract pairs get exactly 0, not a rounding residue, so the zero fraction is exact.
    dist[pairs["home"].to_numpy()[ok] == pairs["work"].to_numpy()[ok]] = 0.0
    return dist, counts[ok]


def weighted_quantiles(dist, weights, qs):
    order = np.argsort(dist)
    cum = np.cumsum(weights[order]) / weights.sum()
    return [dist[order][np.searchsorted(cum, q)] for q in qs]


def print_stats(label, dist, weights):
    total = weights.sum()
    zero = weights[dist == 0].sum() / total
    q25, q50, q75, q90 = weighted_quantiles(dist, weights, [0.25, 0.5, 0.75, 0.9])
    mean = (dist * weights).sum() / total
    print(f"{label}: n={int(total):,}  same tract={100 * zero:.2f}%  mean={mean:.1f} km  "
          f"p25={q25:.1f}  median={q50:.1f}  p75={q75:.1f}  p90={q90:.1f} km  "
          f"<=10km={100 * weights[dist <= 10].sum() / total:.1f}%  "
          f">100km={100 * weights[dist > 100].sum() / total:.1f}%")


def histogram_edges(series, n_bins, log_x, xlim):
    """Bin edges shared by every series: n_bins equal-width bins on a linear axis, or equal-log-width
    bins on a log one, spanning xlim if given, else all the (nonzero, for log) distances."""
    if log_x:
        lo = xlim[0] if xlim else min(dist[dist > 0].min() for _, dist, _, _, _ in series)
        hi = xlim[1] if xlim else max(dist.max() for _, dist, _, _, _ in series)
        return np.geomspace(lo, hi, n_bins + 1)
    lo, hi = xlim if xlim else (0.0, max(dist.max() for _, dist, _, _, _ in series))
    return np.linspace(lo, hi, n_bins + 1)


def plot_histogram(ax, dist, weights, label, edges, color, reference):
    """Plot the fraction of all workers in each bin. The fractions are of every worker, including
    those outside the bins (same-tract commuters on a log axis, or anything beyond --xlim), so all
    series share one scale. The reference (LODES) is an outline drawn on top; the models are filled."""
    frac = weights / weights.sum()
    if reference:
        ax.hist(dist, bins=edges, weights=frac, histtype="step", color=color, linewidth=1, zorder=3,
                label=label)
    else:
        ax.hist(dist, bins=edges, weights=frac, histtype="stepfilled", color=color, alpha=0.5, label=label)


def plot_cdf(ax, dist, weights, label, log_x, **style):
    """Step-plot the worker-weighted CDF. On a log x axis only the nonzero distances are drawn, but
    the denominator still includes the same-tract commuters, so the curve starts at that fraction
    rather than at 0; on a linear axis the curve starts from 0 at x = 0 and those commuters are
    the step there."""
    order = np.argsort(dist)
    dist, weights = dist[order], weights[order]
    cum = np.cumsum(weights) / weights.sum()
    if log_x:
        keep = dist > 0
        dist, cum = dist[keep], cum[keep]
    else:
        dist, cum = np.concatenate([[0.0], dist]), np.concatenate([[0.0], cum])
    style = {"linewidth": 1, **style}
    ax.step(dist, cum, where="post", label=label, **style)


def main():
    apply_style()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--urbanpop", "-u", required=True,
        help="The UrbanPop .bin the ExaEpi run reads (its agent.urbanpop_filename), e.g. "
        "data/UrbanPop/urbanpop_ca.bin",
    )
    parser.add_argument(
        "--epicast_events", "-e", default=None, nargs="+",
        help="Epicast run.events.bin file(s) to take the generated home->work pairs of workers "
        "infected at work from, pooled over all files given. A high-attack-rate run gives the most "
        "pairs, e.g. data/results/emerge-paper/epicast/ca/ca-p02-r0-run_000.events.bin",
    )
    parser.add_argument(
        "--lodes", "-l", default=None,
        help="Raw LODES OD 'main' file for the state, e.g. data/LODES7/ca_od_main_JT00_2019.csv.gz",
    )
    parser.add_argument(
        "--ctpp", default=None,
        help="CTPP tract-to-tract flow CSV written by download_ctpp_flows.py (2012-2016 data, on 2010 "
        "tracts), e.g. data/CTPP/nm_ctpp2016_tract_flows.csv",
    )
    parser.add_argument(
        "--tracts", "-t", required=True, nargs="+",
        help="2010 Census TRACT shapefile(s) for the state, e.g. "
        "data/US_2010_Census_Tracts/tl_2010_06_tract10.shp",
    )
    parser.add_argument(
        "--logx", action="store_true", default=False,
        help="Use a log x axis instead of the default linear one, to see the whole range of "
        "distances at once. On the linear axis most commutes are bunched at the left, so pair it "
        "with --xlim to see the bulk of the distribution (e.g. --xlim 0 100).",
    )
    parser.add_argument(
        "--cdf", action="store_true", default=False,
        help="Plot the cumulative fraction of workers instead of the default histogram",
    )
    parser.add_argument(
        "--bins", type=int, default=50,
        help="Number of histogram bins over the x range (equal-log-width with --logx). Default: 50",
    )
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="x-axis range in km")
    parser.add_argument("--output", "-o", default="commute_distance.pdf", help="Output file name for plot")
    args = parser.parse_args()

    points = load_internal_points(args.tracts)

    # (label, pairs, color, is a reference), in legend order: the references (LODES, which both
    # models are built from, and CTPP) are drawn on top of the models so they stay visible where
    # they overlap.
    series = []
    if args.lodes:
        series.append(("LODES", load_lodes_pairs(args.lodes), "black", True))
    if args.ctpp:
        series.append(("CTPP", load_ctpp_pairs(args.ctpp), "green", True))
    if args.epicast_events:
        series.append(("Epicast", load_epicast_event_pairs(args.epicast_events), "blue", False))
    series.append(("ExaEpi", load_exaepi_pairs(args.urbanpop), "red", False))

    series = [(label, *pair_distances(pairs, points, label), color, reference)
              for label, pairs, color, reference in series]
    for label, dist, weights, _, _ in series:
        print_stats(label, dist, weights)

    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    if args.cdf:
        for label, dist, weights, color, reference in series:
            # The reference is dashed and thinner, since Epicast's curve follows it closely.
            style = (dict(color=color, linestyle="--", linewidth=0.7, zorder=3) if reference
                     else dict(color=color, linestyle="-", alpha=0.7))
            plot_cdf(ax, dist, weights, label, log_x=args.logx, **style)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cumulative fraction of workers")
    else:
        edges = histogram_edges(series, args.bins, args.logx, args.xlim)
        for label, dist, weights, color, reference in series:
            plot_histogram(ax, dist, weights, label, edges, color, reference)
        ax.set_ylim(0, None)
        ax.set_ylabel("Fraction of workers")
    if args.logx:
        ax.set_xscale("log")
        if args.xlim:
            ax.set_xlim(args.xlim)
    else:
        ax.set_xlim(args.xlim if args.xlim else (0, None))
    ax.set_xlabel("Home-to-work tract distance (km)")
    ax.grid(True, which="major", alpha=0.3)
    # CDF: on a linear axis the curves rise steeply at the left, leaving lower right empty; on a log
    # one upper left is empty instead (every curve is still below ~0.6 out to ~15 km). Histogram:
    # the mass is at short distances, so upper right is empty.
    if args.cdf:
        loc = "upper left" if args.logx else "lower right"
    else:
        loc = "upper right"
    ax.legend(loc=loc, frameon=False, handlelength=1.5)

    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
