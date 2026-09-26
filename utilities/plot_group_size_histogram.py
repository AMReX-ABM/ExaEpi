#!/usr/bin/env python

"""Plot histograms of ExaEpi community/neighborhood sizes, or of neighborhoods per community,
optionally overlaid with the same distribution for an Epicast run -- one plot for the nighttime
and one for the daytime population, written as <output>_night.<ext> and <output>_day.<ext>. Both
come from a single read of the inputs, which for a large Epicast events.bin is most of the time.

Community size is either the residential/nighttime population per community, or the daytime
population (every agent has a work location: a real job/school site for workers/students, or
straight back home for everyone else, so the daytime count is a true headcount, not just
employed workers).

Neighborhood size is likewise either the nighttime (home) neighborhoods, or the daytime ones --
wherever an agent actually spends the day, which for a commuter, a student or a teacher is not
where they live. Both are the number of agents in a given (community, neighborhood) pair;
neighborhood IDs are only unique within a community, so both parts are needed to identify one.

Neighborhoods per community is the number of nonempty neighborhoods in each community. Each
community is a single block group, split into round(home_population / nborhood_size)
neighborhoods when the UrbanPop .bin is built, with whole households dealt across them
(UrbanPop-scripts/group_assignment.py).

The ExaEpi side comes from the <prefix>_day_night_population.csv, <prefix>_nborhood_sizes.txt /
_work_nborhood_sizes.txt and _nborhoods_per_community.txt / _work_nborhoods_per_community.txt
files ExaEpi itself writes when --aggregated_diag_int is enabled (see
ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp) -- the same small text files the other
comparison scripts read, not a multi-gigabyte plotfile.

The Epicast side (--epicast) comes from a run's events.bin, which only records agents who get
infected, so its community populations are estimates scaled up to the exact tract populations in
the file's header (see read_epicast_events.estimate_community_populations for how, and what that
assumes). Epicast has no neighborhood data in its output at all: every community is split into
exactly 4 neighborhoods with households dealt across them in ~equal number (Epicast 2.0 paper), so
its neighborhood sizes are its community sizes / 4, and its neighborhoods per community is always
4. For daytime neighborhoods that even split is only an approximation -- workgroups, not
households, are what Epicast deals across a community's neighborhoods by day.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN  # noqa: E402
from read_epicast_events import (  # noqa: E402
    EPICAST_NEIGHBORHOODS_PER_COMMUNITY,
    estimate_community_populations,
    read_events_bin,
)

# The Epicast/ExaEpi colors every comparison script in this repo uses (e.g.
# compare_group_sizes_to_epicast.py), in the order they're drawn: Epicast first, so ExaEpi's
# bars blend over it.
SERIES_COLORS = {"Epicast": "blue", "ExaEpi": "red"}


PERIODS = ["night", "day"]


def exaepi_path(prefix, field, period):
    """The ExaEpi aggregated-diagnostics file holding field for period ("night" or "day")."""
    if field == "community":
        return f"{prefix}_day_night_population.csv"
    work = "work_" if period == "day" else ""
    suffix = "nborhood_sizes" if field == "neighborhood" else "nborhoods_per_community"
    return f"{prefix}_{work}{suffix}.txt"


def exaepi_sizes(prefix, field, period):
    """ExaEpi's per-community population, per-neighborhood size, or neighborhoods per community,
    for the given period ("night" or "day").

    Communities with no agents at all in that period are dropped: they are not communities in any
    meaningful sense here, and a zero would break --logx.
    """
    path = exaepi_path(prefix, field, period)
    if field == "community":
        pop = pd.read_csv(path)[f"{period}_total"]
        return pop[pop > 0].to_numpy()
    return np.loadtxt(path, dtype=int)


def epicast_populations(events_path):
    """Every Epicast community's estimated night_pop and day_pop, from one read of events_path
    (see read_epicast_events.estimate_community_populations)."""
    if not os.path.exists(events_path):
        sys.exit(f"No such file: {events_path}")
    print(f"Reading Epicast events {events_path}")
    events_df, demog_df = read_events_bin(events_path)
    return estimate_community_populations(events_df, demog_df)


def epicast_sizes(pops, field, period):
    """The same quantity as exaepi_sizes, from epicast_populations' output (see the module
    docstring for how neighborhoods are derived, since Epicast's output has none)."""
    pop = pops[f"{period}_pop"].to_numpy()
    pop = pop[pop > 0]
    if field == "community":
        return np.rint(pop).astype(int)
    if field == "neighborhood":
        return np.repeat(np.rint(pop / EPICAST_NEIGHBORHOODS_PER_COMMUNITY).astype(int),
                         EPICAST_NEIGHBORHOODS_PER_COMMUNITY)
    return np.full(len(pop), EPICAST_NEIGHBORHOODS_PER_COMMUNITY)


# Widest spread that still gets one histogram bin per integer by default. Beyond this the
# bars get too thin to read and a fixed bin count is the better default.
MAX_INTEGER_BINS = 200

# Default number of bins across a series' spread (see series_bins) when it's too wide for one bin
# per integer.
DEFAULT_BINS = 50

# Percentiles that define a series' spread for bin sizing: its extreme 0.1% at either end can be
# outliers (e.g. a single-worker block group in an otherwise unpopulated area, or one enormous
# job-center community) that would otherwise stretch the width to a few giant bins.
SPREAD_PERCENTILES = (0.1, 99.9)


def nice_width(raw_width):
    """Round a bin width up to a "nice" value (1/2/5 x a power of 10).

    matplotlib's default tick locator also picks its step from that same 1/2/5 x 10^n family, so
    whichever step it lands on is essentially always an integer multiple of this bin width --
    meaning bin *centers* fall exactly on the ticks it draws, the same way one-bin-per-integer
    bins naturally do (every integer tick is trivially some bin's center when width=1). An
    arbitrary width (e.g. span/50) has no such relationship to the ticks, so they end up looking
    like they're aligned to bin edges in some spots and nothing in particular elsewhere.
    """
    raw_width = max(raw_width, 1e-9)
    magnitude = 10 ** np.floor(np.log10(raw_width))
    return next((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_width), 10 * magnitude)


def series_bins(sizes, view_min, view_max, n_bins, logx):
    """Bin edges for one series, sized from that series' own spread within the view.

    Sizing every series' bins from their combined range gives a tight series (e.g. Epicast's
    communities, all ~2000) only a handful of bars when another series (e.g. ExaEpi's, with a tail
    past 100,000) sets the range. So each series gets its own width, from the spread
    (SPREAD_PERCENTILES) of its values inside [view_min, view_max] -- in view, so zooming with
    --xlim re-sizes the bins for the zoomed range -- and bins of that width then cover all of its
    in-view data, outliers included, so no visible bar is an odd-sized catch-all. Density
    normalizes by bin width, so series with different widths still overlay on a fair scale.

    These are integer counts, so a spread of at most MAX_INTEGER_BINS (with no explicit n_bins)
    gets one bin per integer, centered on it: a width that isn't a whole number of integers
    aliases badly -- adjacent bars then cover 1 or 2 integers depending on where the edges land,
    which looks like structure in the data but isn't.

    With logx, bins are log-spaced instead -- equal-width bins would render as ever-narrower,
    unreadable slivers once most of them get squeezed into the rightmost decade -- with the ratio
    capped so no bin is narrower than 1 unit: for integer data, sub-unit bins near the low end
    look like huge density spikes purely from their tiny bin_width denominator (count /
    bin_width), not from the data.

    Data outside the view gets one catch-all bin at each end, invisible once the view is clipped
    to it, so the density still reflects ALL of the series.
    """
    in_view = sizes[(sizes >= view_min) & (sizes <= view_max)]
    if len(in_view) == 0:
        in_view = sizes  # nothing of this series is visible; any bins will do
    lo, hi = float(in_view.min()), float(in_view.max())
    spread_lo, spread_hi = np.percentile(in_view, SPREAD_PERCENTILES)
    if logx:
        # Log10 step: spread / n_bins, but no finer than 1 unit at the lowest edge (lo).
        step = max(np.log10(max(spread_hi, spread_lo + 1) / spread_lo) / (n_bins or DEFAULT_BINS),
                   np.log10(1 + 1.0 / lo))
        first = np.floor(np.log10(lo) / step) * step
        edges = 10 ** np.arange(first, np.log10(hi) + step, step)
    elif n_bins is None and spread_hi - spread_lo <= MAX_INTEGER_BINS:
        edges = np.arange(np.floor(lo) - 0.5, hi + 1.5, 1.0)
    else:
        width = nice_width((spread_hi - spread_lo) / (n_bins or DEFAULT_BINS))
        first_center = np.floor(lo / width) * width
        edges = np.arange(first_center - width / 2, hi + width, width)
    if sizes.min() < edges[0]:
        edges = np.concatenate(([sizes.min()], edges))
    if sizes.max() > edges[-1]:
        edges = np.append(edges, sizes.max())
    return edges.tolist()


# Per --field presentation: what one sample is (used for the "Found N ..." message), the x-axis
# label, the noun for the quantity being summarized, and the default output basename.
FIELD_INFO = {
    "community": ("communities", "Community size (number of agents)", "size", "community_sizes"),
    "neighborhood": ("neighborhoods", "Neighborhood size (number of agents)", "size", "neighborhood_sizes"),
    "nborhoods_per_community": ("communities", "Neighborhoods per community", "count", "nborhoods_per_community"),
}


def print_stats(name, sizes):
    # Weighted by size (each group of size s stands in for s members who experience that group
    # size) rather than one point per group -- a plain per-group view makes the many small groups
    # look dominant even when most members are actually in a big one, so both the plot and its
    # summary stats are member-weighted throughout. Printed rather than shown in the legend -- this
    # figure is only ~3.1in wide in the paper, with no room for it at PLOS's 8-12pt font floor, and
    # the legend should just name the series.
    weighted_mean = np.average(sizes, weights=sizes)
    sorted_sizes = np.sort(sizes)
    cum_members = np.cumsum(sorted_sizes)
    weighted_median = sorted_sizes[np.searchsorted(cum_members, cum_members[-1] / 2)]
    print(
        f"{name}: n={len(sizes):,}, mean={weighted_mean:.1f}, median={weighted_median:.1f}, "
        f"max={sizes.max():,}"
    )


def plot_distribution(series_list, args, xlabel, stat_noun, output):
    """Draw one histogram (or CDF, with --cdf) of the (name, sizes) series in series_list and save
    it to output."""
    all_sizes = np.concatenate([sizes for _, sizes in series_list])
    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    # Without --xlim, anchor the view at the 0.1th percentile rather than the true minimum, in both
    # log and linear mode: the smallest handful of communities/groups can be extreme outliers (e.g.
    # a single-worker block group in an otherwise unpopulated area) that would otherwise stretch the
    # axis across a long, nearly-empty stretch for very little data. The right edge is left to
    # autoscale.
    if args.xlim is not None:
        view_min, view_max = args.xlim
    else:
        view_min, view_max = float(np.percentile(all_sizes, 0.1)), float(all_sizes.max())
    colors = [SERIES_COLORS[name] for name, _ in series_list]

    if args.cdf:
        for (name, sizes), color in zip(series_list, colors):
            print_stats(name, sizes)
            # Weighted cumulative fraction: cumsum(sorted_sizes) at position i is exactly "how
            # many members are in a group of size <= sorted_sizes[i]" (each group's own size is
            # both its x-value and its member-count contribution), divided by the total member
            # count.
            sorted_sizes = np.sort(sizes)
            cumulative_frac = np.cumsum(sorted_sizes) / sorted_sizes.sum()
            ax.step(sorted_sizes, cumulative_frac, where="post", color=color, alpha=0.7, label=name)
        ax.set_ylabel("Cumulative fraction of members")
    else:
        for (name, sizes), color in zip(series_list, colors):
            print_stats(name, sizes)
            bins = series_bins(sizes, view_min, view_max, args.bins, args.logx)
            ax.hist(sizes, bins=bins, weights=sizes, density=len(series_list) > 1, color=color,
                    alpha=0.5 if len(series_list) > 1 else 0.7, label=name)
        ax.set_ylabel(f"Density ({stat_noun}-weighted)" if len(series_list) > 1 else f"Frequency ({stat_noun}s)")
    if args.logx:
        ax.set_xscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (when --xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=view_min, right=args.xlim[1] if args.xlim is not None else None)
    if not args.logx:
        # Default tick count/spacing can pack 5-6 digit values (e.g. community size, up to the
        # tens of thousands) too tightly for the figure width, running labels into each other.
        # Fewer ticks plus thousands separators keeps them legible; skipped for --logx, which
        # already gets its own (multiplicative) tick locator suited to a log axis.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if len(series_list) > 1:
        ax.legend()

    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"{'CDF' if args.cdf else 'Histogram'} saved to {output}")


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prefix", "-p", required=True,
        help="ExaEpi's --aggregated_diag_prefix (e.g. 'cases'), matching the run's "
        "<prefix>_day_night_population.csv / _nborhood_sizes.txt / _work_nborhood_sizes.txt / "
        "_nborhoods_per_community.txt / _work_nborhoods_per_community.txt "
        "files -- see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp",
    )
    parser.add_argument(
        "--epicast", "-e", default=None,
        help="An Epicast run's events.bin to overlay (e.g. "
        "data/results/emerge-paper/epicast/nm/nm-p01-r0-run_000.events.bin). The higher the run's "
        "attack rate, the more agents its community population estimates are based on. "
        "Default: plot ExaEpi only.",
    )
    parser.add_argument(
        "--field",
        "-f",
        choices=list(FIELD_INFO),
        default="community",
        help="Which quantity to histogram (default: community)",
    )
    parser.add_argument(
        "--bins",
        "-b",
        type=int,
        default=None,
        help="Number of histogram bins across each series' own spread (ignored with --cdf). "
        f"Default: one bin per distinct integer value for a series whose spread is at most "
        f"{MAX_INTEGER_BINS}, else {DEFAULT_BINS}.",
    )
    parser.add_argument(
        "--cdf", action="store_true", help="Plot the empirical cumulative distribution instead of a histogram"
    )
    parser.add_argument("--logx", action="store_true", help="Use a logarithmic x axis")
    parser.add_argument(
        "--xlim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
        help="x-axis range to display; bins are sized for this range",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output image file; the period is inserted before the extension, so e.g. out.png "
        "gives out_night.png and out_day.png (default: <field>_<histogram|cdf>.png)",
    )
    args = parser.parse_args()
    if args.xlim is not None:
        if args.xlim[0] >= args.xlim[1]:
            sys.exit(f"--xlim MIN must be less than MAX, got {args.xlim[0]:g} {args.xlim[1]:g}")
        if args.logx and args.xlim[0] <= 0:
            sys.exit(f"--logx requires a positive --xlim MIN, got {args.xlim[0]:g}")

    plural, xlabel, stat_noun, basename = FIELD_INFO[args.field]
    root, ext = os.path.splitext(args.output or f"{basename}_{'cdf' if args.cdf else 'histogram'}.png")

    # Check ExaEpi's small files before the much slower events.bin read, so a missing one is
    # reported right away. A missing period is skipped rather than fatal: e.g. runs from before
    # ExaEpi wrote daytime neighborhood files still get their nighttime plot.
    periods = []
    for period in PERIODS:
        path = exaepi_path(args.prefix, args.field, period)
        if os.path.exists(path):
            periods.append(period)
        else:
            print(f"No such file: {path} -- skipping the {period}time plot. The static aggregated "
                  "diagnostics are only written when ExaEpi runs with agent.aggregated_diag_int "
                  "enabled, and only on a fresh start (never on restart); daytime neighborhood "
                  "files are also missing from runs made before ExaEpi started writing them.")
    if not periods:
        sys.exit("No ExaEpi aggregated diagnostics to plot.")

    pops = epicast_populations(args.epicast) if args.epicast is not None else None
    plots = []
    for period in periods:
        print(f"Reading ExaEpi aggregated diagnostics {exaepi_path(args.prefix, args.field, period)}")
        # Epicast first in the list, to be drawn underneath.
        series_list = [("ExaEpi", exaepi_sizes(args.prefix, args.field, period))]
        if pops is not None:
            series_list.insert(0, ("Epicast", epicast_sizes(pops, args.field, period)))
        for name, sizes in series_list:
            print(f"Found {len(sizes)} {period}time {name} {plural}")
            print(pd.Series(sizes, name=stat_noun).describe())
            if args.logx and sizes.min() <= 0:
                sys.exit(f"--logx requires strictly positive {stat_noun}s, but the {period}time "
                         f"{name} minimum is {sizes.min()}")
        plots.append((period, series_list))

    for period, series_list in plots:
        print(f"--- {period}time")
        plot_distribution(series_list, args, xlabel, stat_noun, f"{root}_{period}{ext}")


if __name__ == "__main__":
    main()
