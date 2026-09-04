#!/usr/bin/env python

"""Plot a histogram of ExaEpi community/neighborhood sizes, or of neighborhoods per
community, from a run's plotfile.

Community size is plotted as two overlaid series: agents grouped by home (home_i, home_j)
-- the residential/nighttime population per community -- and by work (work_i, work_j) --
the daytime population (every agent has a work_i/work_j: a real job/school site for
workers/students, or straight to home_i/home_j for everyone else).

Neighborhood size is the number of agents in a given (community, neighborhood) pair --
neighborhood IDs are only unique within a community, so agents are grouped by the
(home_i, home_j, nborhood) triple.

Neighborhoods per community is the number of distinct neighborhood IDs among the agents
living in each community.

All three read per-agent fields (home_i/j, work_i/j, nborhood) that are only written to
the first plotfile of a run (typically plt00000), so this script requires that file.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt
import yt.frontends.amrex.api  # noqa: F401  (registers the AMReX frontend's IO handlers)
from yt.frontends.amrex.data_structures import AMReXDataset


def _agent_df(ds, *fields):
    """Per-agent particle fields, as a DataFrame. These (home_i/j, work_i/j, nborhood, ...) are
    static per-agent attributes only written to the first plotfile of a run (e.g. plt00000)."""
    ptype = "agents"
    particle_fields = {f[1] for f in ds.field_list if f[0] == ptype}
    required = {f"particle_{f}" for f in fields}
    missing = required - particle_fields
    if missing:
        sys.exit(
            f"Plotfile is missing per-agent field(s) {sorted(missing)}; these are only written "
            "to the first plotfile of a run (e.g. plt00000)."
        )
    ad = ds.all_data()
    return pd.DataFrame(
        {f: np.rint(np.asarray(ad[(ptype, f"particle_{f}")])).astype(int) for f in fields}
    )


def home_community_sizes(ds):
    """Residential (nighttime) population per community -- agents grouped by home (home_i,
    home_j). Community size straight from the "comm"/"total" mesh fields would be equivalent and
    doesn't require plt00000, but computing it the same way as work_community_sizes (per-agent
    fields) keeps the two directly comparable -- both counts then come from exactly the same
    agent set and grouping logic, just on a different pair of columns."""
    df = _agent_df(ds, "home_i", "home_j")
    return df.groupby(["home_i", "home_j"]).size().to_numpy()


def work_community_sizes(ds):
    """Daytime population per community -- agents grouped by work (work_i, work_j). Every agent
    has a work_i/work_j (UrbanPopData.cpp sets it to a real job/school site for workers/students,
    or straight to home_i/home_j for everyone else -- work-from-home, unemployed, retired, etc.),
    so this is the true daytime headcount, not just employed workers."""
    df = _agent_df(ds, "work_i", "work_j")
    return df.groupby(["work_i", "work_j"]).size().to_numpy()


def _home_nborhood_df(ds):
    return _agent_df(ds, "home_i", "home_j", "nborhood")


def neighborhood_sizes(ds):
    df = _home_nborhood_df(ds)
    return df.groupby(["home_i", "home_j", "nborhood"]).size().to_numpy()


def nborhoods_per_community(ds):
    """Number of distinct neighborhood IDs among the agents living in each community.

    Each community is a single block group, which ExaEpi gives
    get_max_nborhood(nborhood_size, home_population) == round(home_population /
    nborhood_size) neighborhoods, drawing each agent's ID uniformly over that range
    (UrbanPopData.cpp). Counting distinct IDs therefore recovers that allocation exactly
    so long as every neighborhood drew at least one agent -- which at realistic
    nborhood_size values it reliably does, since a neighborhood is hundreds of agents.
    """
    df = _home_nborhood_df(ds)
    return df.groupby(["home_i", "home_j"])["nborhood"].nunique().to_numpy()


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot_dir", "-p", required=True, help="ExaEpi plotfile directory, e.g. plt00000")
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

    print(f"Reading ExaEpi plotfile {args.plot_dir}")
    # yt.load() auto-detects ExaEpi's plotfiles as the plain BoxlibDataset, which does not look
    # for the "agents" particle subdirectory -- that auto-detection only happens in the
    # AMReXDataset subclass, so it is instantiated directly here.
    ds = AMReXDataset(args.plot_dir)

    if args.field == "community":
        # Two overlaid series -- residential (home) vs daytime (work) population per community
        # -- rather than the single series every other --field produces.
        series_list = [("Home", home_community_sizes(ds)), ("Work", work_community_sizes(ds))]
    elif args.field == "neighborhood":
        series_list = [(plural.capitalize(), neighborhood_sizes(ds))]
    else:
        series_list = [(plural.capitalize(), nborhoods_per_community(ds))]

    for name, sizes in series_list:
        found_what = f"{name.lower()} {plural}" if args.field == "community" else plural
        print(f"Found {len(sizes)} {found_what}")
        print(pd.Series(sizes, name=stat_noun).describe())

    all_sizes = np.concatenate([sizes for _, sizes in series_list])
    if args.logx and all_sizes.min() <= 0:
        sys.exit(f"--logx requires strictly positive {stat_noun}s, but the minimum is {all_sizes.min()}")

    fig, ax = plt.subplots(figsize=(5, 4))
    left_edge = float(all_sizes.min()) if args.logx else 0
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    def label_with_stats(name, sizes):
        # Weighted by size (each group of size s stands in for s members who experience that
        # group size) rather than one point per group -- a plain per-group view makes the many
        # small groups look dominant even when most members are actually in a big one, so both
        # the plot and its summary stats are member-weighted throughout. Split across two lines
        # -- matplotlib legends render "\n" fine, and this label is long enough on one line to
        # push the legend box wider than the whole figure.
        n_members = sizes.sum()
        weighted_mean = np.average(sizes, weights=sizes)
        sorted_sizes = np.sort(sizes)
        cum_members = np.cumsum(sorted_sizes)
        weighted_median = sorted_sizes[np.searchsorted(cum_members, cum_members[-1] / 2)]
        return (
            f"{name}: n={len(sizes):,} ({n_members:,} {stat_noun}s)\n"
            f"mean={weighted_mean:.2f}, median={weighted_median:.2f}, max={sizes.max():,}"
        )

    if args.cdf:
        for (name, sizes), color in zip(series_list, colors):
            # Weighted cumulative fraction: cumsum(sorted_sizes) at position i is exactly "how
            # many members are in a group of size <= sorted_sizes[i]" (each group's own size is
            # both its x-value and its member-count contribution), divided by the total member
            # count.
            sorted_sizes = np.sort(sizes)
            cumulative_frac = np.cumsum(sorted_sizes) / sorted_sizes.sum()
            ax.step(sorted_sizes, cumulative_frac, where="post", color=color, alpha=0.7,
                    label=label_with_stats(name, sizes))
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
        span = int(bin_max - overall_min)
        if args.logx:
            # Linear (equal-width) bins would render as ever-narrower, unreadable slivers
            # once the x axis is log-scaled, since most of them get squeezed into the
            # rightmost decade. Log-spaced bins keep them visually even instead.
            max_bins = args.bins if args.bins is not None else 50
            bins = log_spaced_integer_bins(overall_min, bin_max, max_bins=max_bins)
        elif args.bins is not None:
            bins = args.bins
        elif span <= MAX_INTEGER_BINS:
            bins = np.arange(overall_min - 0.5, bin_max + 1.5, 1.0).tolist()
        else:
            bins = nice_linear_bins(overall_min, bin_max).tolist()
        if not isinstance(bins, int):
            # nice_linear_bins() (and the integer scheme) anchor bin centers/edges relative to
            # the data, but nice_linear_bins can still put a bin's left edge below 0 even
            # though the data's own minimum is well above it. Forcing the view to start exactly
            # at 0 would then clip that bin in half; starting it at the bin's own left edge
            # instead always shows the full first bar.
            left_edge = bins[0]
        if bin_max < data_max and not isinstance(bins, int):
            # nice_linear_bins() (and the integer scheme) can both overshoot bin_max by up to
            # one bin width, which -- for an xlim close enough to the true max -- can already
            # exceed data_max. Drop any such edges before appending it, or the result isn't
            # monotonically increasing and numpy.histogram rejects it outright.
            bins_arr = np.asarray(bins)
            bins = np.append(bins_arr[bins_arr < data_max], data_max).tolist()
        for (name, sizes), color in zip(series_list, colors):
            ax.hist(sizes, bins=bins, weights=sizes, density=len(series_list) > 1, color=color,
                    alpha=0.5 if len(series_list) > 1 else 0.7, edgecolor="black",
                    label=label_with_stats(name, sizes))
        ax.set_ylabel(f"Density ({stat_noun}-weighted)" if len(series_list) > 1 else f"Frequency ({stat_noun}s)")
    if args.logx:
        ax.set_xscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (the default, when --xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=left_edge, right=args.xlim)
    ax.set_xlabel(xlabel)
    #ax.set_title(f"Histogram of ExaEpi {args.field} sizes")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"{'CDF' if args.cdf else 'Histogram'} saved to {output}")


if __name__ == "__main__":
    main()
