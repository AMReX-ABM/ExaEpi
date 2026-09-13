#!/usr/bin/env python

"""Compare an ExaEpi run against an Epicast run over a sequence of days, at the level of individual
communities (see plot_geo.py for the single-choropleth-per-day view this builds on, and for the
loaders and compare_day() this reuses). Two outputs are produced:
  - a day-scalar summary plot of infection-weighted Pearson's r: correlation of the raw infection
    rates, weighted by how much infection is actually present in each community (see compare_day
    for why weighting this way rather than by population). Plotted as two series, tract-level and
    county-level r computed independently from the same underlying data, so the plot also shows
    whether the two simulators' agreement holds up across geographic granularity or is an artifact
    of one particular level.
  - a community-by-community scatter plot, one panel per day: ExaEpi infection rate vs Epicast
    infection rate, one point per matched community, with a y=x reference line -- the full
    per-community comparison underlying that day's scalars, for spotting which specific communities
    the two simulators disagree on. This one is still at a single geographic level, chosen by
    --county_level (tract by default), since a scatter plot with both levels overlaid would just
    superimpose two point clouds at different resolutions.

Both simulators' loaders already return a per-community DataFrame keyed by GEOID10 (see
load_exaepi_grid_stats / reconstruct_epicast_snapshot), so the two are simply merged on GEOID10 per
day. Epicast's finest geographic unit is the Census tract, so tract level is the finest granularity
at which the two can be compared; county level aggregates further.

If the two runs' start dates aren't aligned (e.g. one simulator was seeded a few days later in
epidemic progression than the other), pass --exaepi_day_shift to re-time the ExaEpi day parsed from
each file before it's matched against Epicast -- everything downstream (the Epicast day looked up,
the r values, the reported/plotted day) is computed from that shifted day.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_geo import (  # noqa: E402
    load_exaepi_grid_stats,
    _parse_day_from_filename,
    iter_epicast_snapshots,
    expand_aggregated_files,
    compare_day,
)
from read_epicast_events import read_events_bin  # noqa: E402
from geo_agg_utils import aggregate_to_county  # noqa: E402
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402


def _fmt(r):
    """Format a Pearson r for the console, or a placeholder if that level had too few matching
    communities to compute one (compare_day returns None in that case)."""
    return f"{r:.4f}" if r is not None else "n/a"


def main():
    apply_style()

    parser = argparse.ArgumentParser(
        description="Compare ExaEpi and Epicast per-community infection rates across days: a "
        "Pearson's-r-vs-day summary plot (tract and county level), plus a community-by-community "
        "scatter plot per day"
    )
    parser.add_argument(
        "--exaepi_files",
        "-g",
        required=True,
        nargs="+",
        help="ExaEpi aggregated-diagnostics CSV files to compare, one per day (e.g. cases00050, "
        "written via --aggregated_diag_int). Each entry may be a single CSV file, a parent "
        "directory containing many such files (e.g. a run's whole output directory), or a glob "
        "pattern (e.g. 'results/cases*') -- so a whole run's output can be pointed at directly "
        "instead of listing every file by hand. The day for each is parsed from its trailing "
        "digits.",
    )
    parser.add_argument(
        "--events_file",
        "-f",
        required=True,
        help="Epicast run.events.bin file to compare against",
    )
    parser.add_argument(
        "--county_level",
        action="store_true",
        default=False,
        help="Use the Census county level (5-digit GEOID) instead of the default Census tract "
        "level (11-digit GEOID) for the community-by-community scatter plot and the "
        "console/--csv diagnostics (rho, rmse, etc.). The main Pearson's-r-vs-day summary plot "
        "always shows both tract and county level regardless of this flag.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="geo_compare.pdf",
        help="Output file name for the Pearson's-r-vs-day plot",
    )
    parser.add_argument(
        "--exaepi_day_shift",
        type=int,
        default=0,
        help="Shift the ExaEpi day parsed from each file by this many days before "
        "matching it up against Epicast (e.g. 5 treats a cases00020 file as day 25). Use this "
        "to correct for a real start-date misalignment between the two runs (e.g. one simulator "
        "seeded a few days later than the other) -- the day reported in the output, the day used "
        "to look up the Epicast snapshot, and the r values are all computed from this shifted day, "
        "not the original one.",
    )
    parser.add_argument(
        "--epicast_day_offset",
        type=int,
        default=0,
        help="Shift the Epicast day used for comparison by this many days relative to the "
        "(possibly already --exaepi_day_shift-ed) ExaEpi day (e.g. 20 compares ExaEpi day D "
        "against Epicast day D+20). Diagnostic option for checking that the comparison metrics "
        "are actually sensitive to a temporal misalignment between the two runs, rather than e.g. "
        "being dominated by shared population geography -- unlike --exaepi_day_shift, this does "
        "not change the day reported in the output.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional file to also write the day,pearson_r_tract,n_tract,pearson_r_county,"
        "n_county,rho,pval,pearson_r,rmse,rmse_people,r_log,rmse_log,n table to as CSV (the "
        "un-suffixed rho/pearson_r/rmse/etc. columns are the --county_level-selected level, "
        "matching the console output and scatter plot)",
    )
    parser.add_argument(
        "--scatter_output",
        "-s",
        default=None,
        help="Output file name for the community-by-community scatter plot (one panel per day: "
        "ExaEpi infection rate vs Epicast infection rate, one point per matched community). If "
        "omitted, this plot is skipped and only the Pearson's-r-vs-day plot is produced.",
    )
    args = parser.parse_args()

    print("Reading Epicast data from", args.events_file)
    events_df, demog_df = read_events_bin(args.events_file)
    print(f"Read {len(events_df):,} events, {len(demog_df)} Census tracts")

    exaepi_files = expand_aggregated_files(args.exaepi_files)
    print(f"Comparing {len(exaepi_files)} days:", ", ".join(os.path.basename(f) for f in exaepi_files))

    day_infos = []
    for csv_path in exaepi_files:
        parsed_day = _parse_day_from_filename(csv_path)
        if parsed_day is None:
            raise SystemExit(f"Could not parse a day number from file name: {csv_path}")
        day = parsed_day + args.exaepi_day_shift
        epicast_day = day + args.epicast_day_offset
        day_infos.append((csv_path, parsed_day, day, epicast_day))

    # Reconstruct every requested Epicast day's TRACT-level snapshot in one incremental pass over
    # the shared event log, instead of the loop below re-deriving each day's whole disease-state
    # history from scratch (which is what calling reconstruct_epicast_snapshot once per day here
    # used to do -- see iter_epicast_snapshots for why that's quadratic in the number of days).
    # Always reconstructed at tract level, since the summary plot's county-level series can be had
    # far more cheaply below via aggregate_to_county() on this same result, rather than by running a
    # second full incremental pass at county level.
    epicast_tract_snapshots = list(
        iter_epicast_snapshots(events_df, demog_df, [info[3] for info in day_infos], county_level=False)
    )

    rows = []
    scatter_panels = []
    for (csv_path, parsed_day, day, epicast_day), (resolved_day, epicast_tract_df) in zip(
        day_infos, epicast_tract_snapshots
    ):
        if resolved_day != epicast_day:
            print(
                f"WARNING: {csv_path} (ExaEpi day {parsed_day}, shifted to {day}) requested Epicast "
                f"day {epicast_day} (shifted ExaEpi day {day} + offset {args.epicast_day_offset}), "
                f"but Epicast clamped it to day {resolved_day}"
            )

        exaepi_tract_df = load_exaepi_grid_stats(csv_path, tract_level=True)
        exaepi_county_df = load_exaepi_grid_stats(csv_path, county_level=True)
        epicast_county_df = aggregate_to_county(epicast_tract_df, input_level="tract")

        if args.county_level:
            exaepi_df, epicast_df = exaepi_county_df, epicast_county_df
            other_exaepi_df, other_epicast_df = exaepi_tract_df, epicast_tract_df
        else:
            exaepi_df, epicast_df = exaepi_tract_df, epicast_tract_df
            other_exaepi_df, other_epicast_df = exaepi_county_df, epicast_county_df

        rho, pval, r, rmse, r_log, rmse_log, n, merged_df = compare_day(exaepi_df, epicast_df)
        if rho is None:
            print(f"WARNING: day {day}: fewer than 2 matching communities ({n}), skipping")
            continue

        # The summary plot always shows Pearson's r at both geographic levels, regardless of
        # --county_level (which only picks the level used for the diagnostics below and the
        # scatter plot) -- other_exaepi_df/other_epicast_df (the level not selected above) supply
        # that second series.
        _, _, other_r, _, _, _, other_n, _ = compare_day(other_exaepi_df, other_epicast_df)
        r_tract, n_tract = (r, n) if not args.county_level else (other_r, other_n)
        r_county, n_county = (r, n) if args.county_level else (other_r, other_n)

        # Purely a display convenience: express the (unitless) RMSE of infection RATE as a
        # people-equivalent, by scaling it up by the total matched population -- "if this RMSE
        # applied uniformly across everyone being compared, that's about how many people it'd be."
        # The RMSE itself is still computed on rates, weighted by infection level (see compare_day)
        # -- this is just a more human-readable way to report that same number, not a different
        # metric. r_log/rmse_log (see compare_day) have no such people-equivalent: they're computed
        # on log1p(count), not rate, so a "people" scaling wouldn't be meaningful.
        total_pop = ((merged_df.pop_exaepi + merged_df.pop_epicast) / 2.0).sum()
        rmse_people = rmse * total_pop
        print(
            f"Day {day}: tract Pearson r = {_fmt(r_tract)} (n = {n_tract}), "
            f"county Pearson r = {_fmt(r_county)} (n = {n_county}); "
            f"Spearman rho = {rho:.4f}, weighted RMSE = {rmse:.4f} (~{rmse_people:,.0f} people) "
            f"(p = {pval:.3g}), log Pearson r = {r_log:.4f}, RMSLE = {rmse_log:.4f}"
        )
        rows.append(
            {
                "day": day,
                "pearson_r_tract": r_tract,
                "n_tract": n_tract,
                "pearson_r_county": r_county,
                "n_county": n_county,
                "rho": rho,
                "pval": pval,
                "pearson_r": r,
                "rmse": rmse,
                "rmse_people": rmse_people,
                "r_log": r_log,
                "rmse_log": rmse_log,
                "n": n,
            }
        )
        scatter_panels.append(
            (merged_df, f"Day {day}\nρ = {rho:.2f}, r = {r:.2f}, RMSE = {rmse:.3f} (~{rmse_people:,.0f} people)")
        )

    if not rows:
        raise SystemExit("No day had enough matching communities to compute a correlation")

    result_df = pd.DataFrame(rows).sort_values("day")
    if args.csv:
        result_df.to_csv(args.csv, index=False)
        print("Wrote table to", args.csv)

    # Labels are kept short (the "infection-weighted" detail belongs in the figure caption, not the
    # plot itself) since this whole figure is only ~3.1in wide in the paper -- a long label/legend
    # string simply has no room to fit at PLOS's 8-12pt font floor, regardless of layout engine.
    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    ax.plot(result_df.day, result_df.pearson_r_tract, label="Tract", lw=1, color="#eb6834")
    ax.plot(result_df.day, result_df.pearson_r_county, label="County", lw=1, color="#4a3aa7")
    ax.set_xlabel("Day")
    ax.set_ylabel("Pearson's r")
    ax.set_xlim(left=0)
    ymin = min(result_df.pearson_r_tract.min(skipna=True), result_df.pearson_r_county.min(skipna=True))
    ax.set_ylim(ymin - 0.05, 1.05)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    ax.legend()
    print("Plotting results to", args.output)
    fig.savefig(args.output)

    if not args.scatter_output:
        return

    # Community-by-community scatter: one panel per day, ExaEpi rate vs Epicast rate, one point per
    # matched community. All panels share the same axis range so panels are visually comparable
    # across days, and a y=x reference line marks perfect agreement. Total width is fixed at the
    # paper's half-page width regardless of how many days there are (see paper_style.py) -- each
    # panel just gets narrower as more days are added.
    axis_max = max(max(df.rate_exaepi.max(), df.rate_epicast.max()) for df, _ in scatter_panels)
    axis_max = axis_max * 1.05 if axis_max > 0 else 1.0

    n = len(scatter_panels)
    panel_width = HALF_PAGE_WIDTH_IN / n
    fig2, axes2 = plt.subplots(1, n, figsize=(HALF_PAGE_WIDTH_IN, panel_width), squeeze=False, layout="constrained")
    axes2 = axes2[0]
    for ax, (df, label) in zip(axes2, scatter_panels):
        ax.plot([0, axis_max], [0, axis_max], color="gray", lw=0.5, ls="--", zorder=1)
        ax.scatter(df.rate_exaepi, df.rate_epicast, s=6, alpha=0.5, linewidths=0, zorder=2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal")
        ax.set_xlabel("ExaEpi infection rate")
        ax.set_ylabel("Epicast infection rate")
        ax.set_title(label)
    print("Plotting community-by-community scatter to", args.scatter_output)
    fig2.savefig(args.scatter_output)


if __name__ == "__main__":
    main()
