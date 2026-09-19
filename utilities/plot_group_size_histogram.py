#!/usr/bin/env python

"""Plot a histogram of ExaEpi community/neighborhood sizes, or of neighborhoods per
community, from the static aggregated diagnostics a run writes.

Community size is plotted as two overlaid series: the residential/nighttime population per
community, and the daytime population (every agent has a work location: a real job/school
site for workers/students, or straight back home for everyone else, so the daytime count is
a true headcount, not just employed workers).

Neighborhood size is the number of agents in a given (community, neighborhood) pair --
neighborhood IDs are only unique within a community, so both are needed to identify one.

Neighborhoods per community is the number of nonempty neighborhoods in each community. Each
community is a single block group, split into round(home_population / nborhood_size)
neighborhoods when the UrbanPop .bin is built, with whole households dealt across them
(UrbanPop-scripts/group_assignment.py).

All of this comes from the <prefix>_day_night_population.csv and <prefix>_nborhood_sizes.txt
/ _nborhoods_per_community.txt files ExaEpi itself writes when --aggregated_diag_int is
enabled (see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp) -- the same small text
files the other comparison scripts read, not a multi-gigabyte plotfile.
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


def _read_file(path, read):
    if not os.path.exists(path):
        sys.exit(
            f"No such file: {path} -- the static aggregated diagnostics are only written when "
            "ExaEpi runs with agent.aggregated_diag_int enabled, and only on a fresh start "
            "(never on restart)."
        )
    return read(path)


def community_sizes(prefix):
    """Nighttime and daytime population per community, from <prefix>_day_night_population.csv.

    Communities with no agents at all are dropped: they are not communities in any meaningful
    sense here, and a zero would break --logx.
    """
    df = _read_file(f"{prefix}_day_night_population.csv", pd.read_csv)
    return [
        ("Nighttime", df.night_total[df.night_total > 0].to_numpy()),
        ("Daytime", df.day_total[df.day_total > 0].to_numpy()),
    ]


def counts_from_txt(prefix, suffix):
    """One integer per line -- the same plain format as the workgroup/class/school size files
    compare_group_sizes_to_epicast.py reads, and as Epicast's own reference files."""
    return _read_file(f"{prefix}_{suffix}.txt", lambda p: np.loadtxt(p, dtype=int))


# Widest data span that still gets one histogram bin per integer by default. Beyond this the
# bars get too thin to read and a fixed bin count is the better default.
MAX_INTEGER_BINS = 200


def log_spaced_integer_bins(vmin, vmax, max_bins=50):
    """Log-spaced bin edges from vmin to vmax, capped so no bin is narrower than 1 unit.

    Log-spaced bins are the right choice for a log-x histogram since equal *ratio* renders as
    equal visual width -- but for integer count data, too many bins makes the ones near vmin
    narrower than a single unit, which then look like huge density spikes purely from having a
    tiny bin_width denominator (count / bin_width), not from the data. Rather than patch that by
    flooring individual bin widths after the fact (which breaks the constant-ratio property and
    makes bars render at visibly different widths), this picks a small enough bin count up front
    that every bin -- including the narrowest, at vmin -- stays >= 1 unit wide on its own.
    """
    if vmin <= 0 or vmax <= vmin:
        return max_bins
    max_n_for_resolution = int(np.floor(np.log(vmax / vmin) / np.log(1 + 1.0 / vmin)))
    n_bins = max(1, min(max_bins, max_n_for_resolution))
    return np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)


def nice_linear_bins(vmin, vmax, target_bins=50):
    """Linear bin edges from ~vmin to vmax, with a "nice" width (1/2/5 x a power of 10) instead
    of vmax-vmin split into an arbitrary number of equal pieces.

    matplotlib's default tick locator also picks its step from that same 1/2/5 x 10^n family, so
    whichever step it lands on is essentially always an integer multiple of this bin width --
    meaning bin *centers* fall exactly on the ticks it draws, the same way one-bin-per-integer
    bins naturally do (every integer tick is trivially some bin's center when width=1). An
    arbitrary width (e.g. span/50) has no such relationship to the ticks, so they end up looking
    like they're aligned to bin edges in some spots and nothing in particular elsewhere.
    """
    span = max(vmax - vmin, 1e-9)
    raw_width = span / target_bins
    magnitude = 10 ** np.floor(np.log10(raw_width))
    width = next((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_width), 10 * magnitude)
    first_center = np.floor(vmin / width) * width
    return np.arange(first_center - width / 2, vmax + width, width)


# Per --field presentation: what one sample is (used for the count line in the stats box and
# the "Found N ..." message), the x-axis label, the noun for the quantity being summarized,
# and the default output basename.
FIELD_INFO = {
    "community": ("communities", "Community size (number of agents)", "size", "community_sizes"),
    "neighborhood": ("neighborhoods", "Neighborhood size (number of agents)", "size", "neighborhood_sizes"),
    "nborhoods_per_community": ("communities", "Neighborhoods per community", "count", "nborhoods_per_community"),
}


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prefix", "-p", required=True,
        help="ExaEpi's --aggregated_diag_prefix (e.g. 'cases'), matching the run's "
        "<prefix>_day_night_population.csv / _nborhood_sizes.txt / _nborhoods_per_community.txt "
        "files -- see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp",
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
        help="Number of histogram bins (ignored with --cdf). Default: one bin per distinct "
        f"integer value when the data span at most {MAX_INTEGER_BINS}, else 50.",
    )
    parser.add_argument(
        "--cdf", action="store_true", help="Plot the empirical cumulative distribution instead of a histogram"
    )
    parser.add_argument("--logx", action="store_true", help="Use a logarithmic x axis")
    parser.add_argument("--xlim", type=float, default=None, help="Maximum x-axis value to display")
    parser.add_argument(
        "--output", "-o", default=None, help="Output image file (default: <field>_sizes_<histogram|cdf>.png)"
    )
    args = parser.parse_args()

    plural, xlabel, stat_noun, basename = FIELD_INFO[args.field]
    output = args.output or f"{basename}_{'cdf' if args.cdf else 'histogram'}.png"

    print(f"Reading ExaEpi aggregated diagnostics {args.prefix}_*")
    if args.field == "community":
        # Two overlaid series -- residential (nighttime) vs daytime population per community --
        # rather than the single series every other --field produces.
        series_list = community_sizes(args.prefix)
    elif args.field == "neighborhood":
        series_list = [(plural.capitalize(), counts_from_txt(args.prefix, "nborhood_sizes"))]
    else:
        series_list = [(plural.capitalize(), counts_from_txt(args.prefix, "nborhoods_per_community"))]

    for name, sizes in series_list:
        found_what = f"{name.lower()} {plural}" if args.field == "community" else plural
        print(f"Found {len(sizes)} {found_what}")
        print(pd.Series(sizes, name=stat_noun).describe())

    all_sizes = np.concatenate([sizes for _, sizes in series_list])
    if args.logx and all_sizes.min() <= 0:
        sys.exit(f"--logx requires strictly positive {stat_noun}s, but the minimum is {all_sizes.min()}")

    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    # Anchor the view at the 0.1th percentile rather than the true minimum, in both log and linear
    # mode: the smallest handful of communities/groups can be extreme outliers (e.g. a
    # single-worker block group in an otherwise unpopulated area) that would otherwise stretch the
    # axis across a long, nearly-empty stretch for very little data.
    left_edge = float(np.percentile(all_sizes, 0.1))
    # Avoid tab:blue/tab:red here -- every other script in this repo uses that pair specifically
    # for Epicast/ExaEpi, and reusing it for an unrelated distinction (nighttime vs daytime
    # population) would misleadingly suggest this is also a simulator comparison.
    colors = ["tab:purple", "tab:orange", "tab:green", "tab:brown"]

    def print_stats(name, sizes):
        # Weighted by size (each group of size s stands in for s members who experience that
        # group size) rather than one point per group -- a plain per-group view makes the many
        # small groups look dominant even when most members are actually in a big one, so both
        # the plot and its summary stats are member-weighted throughout. Printed rather than shown
        # in the legend -- this figure is only ~3.1in wide in the paper, with no room for it at
        # PLOS's 8-12pt font floor, and the legend should just name the series.
        weighted_mean = np.average(sizes, weights=sizes)
        sorted_sizes = np.sort(sizes)
        cum_members = np.cumsum(sorted_sizes)
        weighted_median = sorted_sizes[np.searchsorted(cum_members, cum_members[-1] / 2)]
        print(
            f"{name}: n={len(sizes):,}, mean={weighted_mean:.1f}, median={weighted_median:.1f}, "
            f"max={sizes.max():,}"
        )

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
        # These are all integer counts, so by default give each distinct value its own bin
        # centered on it. A fixed bin count whose width isn't a whole number of integers
        # aliases badly -- adjacent bars then cover 1 or 2 integers depending on where the
        # edges land, which looks like structure in the data but isn't. An explicit --bins
        # still wins, and wide-spanning data (community sizes) falls back to 50.
        # When zooming with --xlim, size the bins for the zoomed-in range rather than the full
        # data range -- otherwise a few extreme outliers can set a bin width so wide that only
        # one or two giant bins are even visible in the zoomed view. A final catch-all bin
        # (invisible once xlim clips the view) keeps all the data in the density/frequency
        # normalization. Bins are shared across all series (sized from their combined range) so
        # multiple series overlay on a fair, common set of bars.
        overall_min = all_sizes.min()
        data_max = all_sizes.max()
        bin_max = min(data_max, args.xlim) if args.xlim is not None else data_max
        # Bins are sized from bin_min (defaulting to left_edge, the 0.1th-percentile anchor
        # chosen above, not overall_min) in both log and linear mode -- otherwise the excluded
        # bottom 0.1% would still stretch the bins across a long, nearly-empty range even though
        # the view itself doesn't show it (the same reasoning --xlim already applies to bin_max
        # on the right).
        bin_min = overall_min
        span = int(bin_max - left_edge)
        if args.logx:
            # Linear (equal-width) bins would render as ever-narrower, unreadable slivers
            # once the x axis is log-scaled, since most of them get squeezed into the
            # rightmost decade. Log-spaced bins keep them visually even instead.
            max_bins = args.bins if args.bins is not None else 50
            bin_min = left_edge
            bins = log_spaced_integer_bins(bin_min, bin_max, max_bins=max_bins)
        elif args.bins is not None:
            bins = args.bins
        elif span <= MAX_INTEGER_BINS:
            # Round to the nearest integer so bin edges stay half-integer (X.5), keeping every
            # bin centered on a whole size value the way the un-skipped range already was.
            bin_min = round(left_edge)
            bins = np.arange(bin_min - 0.5, bin_max + 1.5, 1.0).tolist()
        else:
            bin_min = left_edge
            bins = nice_linear_bins(bin_min, bin_max).tolist()
        if not isinstance(bins, int):
            # Check the bins actually produced, not bin_min itself: nice_linear_bins() can (via
            # its "nice" rounding) undershoot its own first edge below overall_min on its own --
            # skipping bin_min's whole percentile range in that case -- in which case there's
            # already a bin covering everything down to the true min and no gap to fill.
            if bins[0] > overall_min:
                # The excluded bottom 0.1% still needs a bin of its own, so the histogram/density
                # normalization reflects ALL the data -- it's just outside the view once
                # ax.set_xlim(left=...) clips below left_edge, exactly like the catch-all bin
                # below handles the >bin_max side for --xlim.
                bins = np.concatenate(([overall_min], bins))
            bins = np.asarray(bins).tolist()
        if bin_max < data_max and not isinstance(bins, int):
            # nice_linear_bins() (and the integer scheme) can both overshoot bin_max by up to
            # one bin width, which -- for an xlim close enough to the true max -- can already
            # exceed data_max. Drop any such edges before appending it, or the result isn't
            # monotonically increasing and numpy.histogram rejects it outright.
            bins_arr = np.asarray(bins)
            bins = np.append(bins_arr[bins_arr < data_max], data_max).tolist()
        for (name, sizes), color in zip(series_list, colors):
            print_stats(name, sizes)
            ax.hist(sizes, bins=bins, weights=sizes, density=len(series_list) > 1, color=color,
                    alpha=0.5 if len(series_list) > 1 else 0.7, label=name)
        ax.set_ylabel(f"Density ({stat_noun}-weighted)" if len(series_list) > 1 else f"Frequency ({stat_noun}s)")
    if args.logx:
        ax.set_xscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (the default, when --xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=left_edge, right=args.xlim)
    if not args.logx:
        # Default tick count/spacing can pack 5-6 digit values (e.g. community size, up to the
        # tens of thousands) too tightly for the figure width, running labels into each other.
        # Fewer ticks plus thousands separators keeps them legible; skipped for --logx, which
        # already gets its own (multiplicative) tick locator suited to a log axis.
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(xlabel)
    #ax.set_title(f"Histogram of ExaEpi {args.field} sizes")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if len(series_list) > 1:
        ax.legend()

    plt.savefig(output, dpi=300)
    print(f"{'CDF' if args.cdf else 'Histogram'} saved to {output}")


if __name__ == "__main__":
    main()
