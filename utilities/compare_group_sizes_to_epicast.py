#!/usr/bin/env python

"""Compare Epicast vs. ExaEpi distributions of workgroup size, school-class size, and
school size for the NM run used in the emerge paper.

Epicast side: data/results/emerge-paper/epicast/epicast_nm_{workgroup,schoolgroup,school}_sizes.txt
-- plain text, one integer (group size) per line. "workgroup" is a workplace peer group,
"schoolgroup" is a classroom-level cohort, "school" is a whole school.

ExaEpi side: reads <prefix>_workgroup_sizes.txt / _class_sizes.txt / _school_sizes.txt, the
static, once-per-run group-size distributions ExaEpi writes when --aggregated_diag_int is
enabled (see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp), in the same plain
one-integer-per-line format as the Epicast files below -- computed by ExaEpi itself from
agents' work_i/work_j/naics/workgroup/school_id/school_class_group attributes:

  - Workgroup size: agents with workgroup > 0 (0 means not assigned to a workgroup --
    not working, or working from home), grouped by (work_i, work_j, naics, workgroup).
    Workgroup IDs are only unique within a (work community, naics) pair -- see
    InteractionModWork.H's max_workgroup * max_naics sizing -- so all three keys are
    needed to recover each actual workgroup.

  - School class size: agents with naics == -1 (a student, not an employee) and
    school_class_group >= 0 (enrolled in a real classroom, not the -1 sentinel for
    unenrolled agents), grouped by school_class_group alone -- that field is already a
    globally unique, densely-packed ID for one (community, school_id, grade, class)
    mixing bucket (see UrbanPop-scripts/group_assignment.py / InteractionModSchool.H).
    Restricting to naics == -1 excludes each class's own homeroom teacher and any
    non-classroom "admin" pools of surplus teachers (which contain no naics == -1
    agents at all), matching the student headcount ExaEpi's own
    "School class size" log histogram reports.

  - School size: agents with school_id > 0 (both students and staff), grouped by
    (work_i, work_j, school_id) -- school_id is only unique within a community, like
    workgroup.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPICAST_DIR = os.path.join(REPO_ROOT, "data", "results", "emerge-paper", "epicast")

# Per --group-name: default Epicast sizes file, the ExaEpi field(s) needed, and axis/label text.
GROUP_INFO = {
    "workgroup": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_workgroup_sizes.txt"),
        "xlabel": "Workgroup size (number of agents)",
        "title": "Workgroup size",
        "basename": "workgroup_sizes",
        "weight_noun": "worker",
    },
    "class": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_schoolgroup_sizes.txt"),
        "xlabel": "Class size (number of students)",
        "title": "School class size",
        "basename": "class_sizes",
        "weight_noun": "student",
    },
    "school": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_school_sizes.txt"),
        "xlabel": "School size (number of agents)",
        "title": "School size",
        "basename": "school_sizes",
        # school size includes both students and staff (see exaepi_school_sizes' docstring
        # note above), but students dominate the headcount, so "student-weighted" is the more
        # intuitive label here even though it's not literally students-only.
        "weight_noun": "student",
    },
}


def load_epicast_sizes(fname):
    sizes = np.loadtxt(fname, dtype=int)
    print(f"Read {len(sizes):,} sizes from Epicast file {fname}")
    return sizes


def exaepi_sizes(prefix, group_name):
    """Read <prefix>_<basename>.txt (see GROUP_INFO), the group-size distribution ExaEpi itself
    computed and wrote -- same plain one-integer-per-line format load_epicast_sizes reads."""
    fname = f"{prefix}_{GROUP_INFO[group_name]['basename']}.txt"
    sizes = np.loadtxt(fname, dtype=int)
    print(f"Read {len(sizes):,} ExaEpi {group_name}s from {fname}")
    return sizes


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


# How many bins the narrowest distribution on a panel has to be resolved into, whatever the
# combined range is (see nice_linear_bins). Deliberately well below the whole panel's target: the
# narrow series only needs enough bins to show its shape, not the same resolution as the wide one.
MIN_BINS_FOR_NARROWEST = 25

# ...and the floor that stops that from overshooting: average samples per bin, for whichever series
# has the fewest in view. Finer bins resolve a tight distribution but also expose counting noise in
# a sparse one, and past a point the panel is showing sampling scatter rather than shape -- New
# Mexico has 1,982 ExaEpi schools in view against California's 24,358, so the same width that reads
# as a smooth curve for one is visibly spiky for the other.
MIN_SAMPLES_PER_BIN = 25


def core_span(sizes, lo_pct=1, hi_pct=99):
    """The x-range a distribution actually occupies, ignoring its tails.

    Percentiles rather than min/max because a single outlier -- one 70,000-agent university among
    California's 25,612 schools -- says nothing about how much room the distribution needs.
    """
    return float(np.percentile(sizes, hi_pct) - np.percentile(sizes, lo_pct))


def nice_linear_bins(vmin, vmax, target_bins=50, resolve_span=None, sparsest_count=None):
    """Linear bin edges from ~vmin to vmax, with a "nice" width (1/2/5 x a power of 10) instead
    of vmax-vmin split into an arbitrary number of equal pieces.

    matplotlib's default tick locator also picks its step from that same 1/2/5 x 10^n family, so
    whichever step it lands on is essentially always an integer multiple of this bin width --
    meaning bin *centers* fall exactly on the ticks it draws, the same way one-bin-per-integer
    bins naturally do (every integer tick is trivially some bin's center when width=1). An
    arbitrary width (e.g. span/50) has no such relationship to the ticks, so they end up looking
    like they're aligned to bin edges in some spots and nothing in particular elsewhere.

    `resolve_span`, if given, is a second, narrower range that also has to come out with enough
    bins to read; the width is then whichever of the two demands is finer. Two overlaid
    distributions share one set of bins (they have to, to be compared as densities), so a width
    sized only by the combined range starves whichever of them is tightly clustered: California's
    school sizes run to 1,200 on the ExaEpi side while 99.8% of Epicast's sit below 250, which at
    the 50-wide bins that range implies left the entire Epicast distribution drawn as five bars.

    `sparsest_count`, if given, is how many values the thinnest-sampled series has in view, and
    caps how far that refinement can go (MIN_SAMPLES_PER_BIN).
    """
    span = max(vmax - vmin, 1e-9)
    raw_width = span / target_bins
    if resolve_span:
        raw_width = min(raw_width, resolve_span / MIN_BINS_FOR_NARROWEST)
    if sparsest_count:
        raw_width = max(raw_width, span / max(1.0, sparsest_count / MIN_SAMPLES_PER_BIN))
    magnitude = 10 ** np.floor(np.log10(raw_width))
    width = next((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_width), 10 * magnitude)
    first_center = np.floor(vmin / width) * width
    return np.arange(first_center - width / 2, vmax + width, width)


def load_cbp_establishment_sizes(fname, state_fips, rng_seed=0):
    """Real establishment sizes for one state, straight from the CBP derived cache
    (data/UrbanPop/cbp19st_derived.csv, written by compute_workgroup_sizes.py).

    Returns (sizes, est_counts): each entry is a representative establishment size and how many
    real establishments it stands for, so weighting by size * est_counts gives the same
    worker-weighted density as the model series. Each CBP employment-size band holds n
    establishments totalling e employees, so sizes are spread log-uniformly across the band and
    rescaled to that band's own mean e/n -- preserving both the establishment count and the
    employment CBP reports.

    Only 2-digit NAICS sectors are summed. CBP reports every level of the NAICS hierarchy as its
    own row (11, 111, 1111, ...), so adding up all of them would count the same establishment
    once per level; the 2-digit sectors partition the state's establishments exactly once.

    NOTE this is the WORKPLACE tier, not the work-group tier: a work-group is a co-worker team
    inside an establishment (Epicast 2.0 sec. 2.4.2), so a 2451-person hospital is one CBP
    establishment but many work-groups. It belongs on this plot as a reference for what real
    workplaces look like, not as a target the work-group distribution should match.
    """
    import csv

    bands = [("<5", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49), ("50_99", 50, 99),
             ("100_249", 100, 249), ("250_499", 250, 499), ("500_999", 500, 999), ("1000", 1000, None)]
    rng = np.random.default_rng(rng_seed)
    sizes, counts = [], []
    with open(fname) as f:
        reader = csv.DictReader(f)
        if "n<5" not in (reader.fieldnames or []):
            sys.exit(f"{fname} has no establishment-size-band columns -- regenerate it with "
                     "compute_workgroup_sizes.py --refresh-cbp-cache")
        for row in reader:
            if int(row["fipstate"]) != state_fips or len(row["naics"]) != 2:
                continue
            for name, lo, hi in bands:
                n, e = int(row["n" + name] or 0), int(row["e" + name] or 0)
                if n <= 0 or e <= 0:
                    continue
                mean = e / n
                hi_eff = hi if hi is not None else max(lo + 1, int(mean * 4))
                draw = np.exp(rng.uniform(np.log(lo), np.log(hi_eff + 1), size=64))
                draw *= mean / draw.mean()
                sizes.append(np.clip(np.rint(draw), 1, None))
                counts.append(np.full(len(draw), n / len(draw)))
    if not sizes:
        sys.exit(f"No CBP rows for state FIPS {state_fips} in {fname}")
    return np.concatenate(sizes), np.concatenate(counts)


def plot_comparison(ax, epicast_sizes, exaepi_sizes, xlabel, title, cdf, weight_noun="member",
                     logx=False, logy=False, max_integer_bins=200, xlim=None, cbp=None):
    epicast_sizes = np.asarray(epicast_sizes)
    exaepi_sizes = np.asarray(exaepi_sizes)
    overall_min = min(epicast_sizes.min(), exaepi_sizes.min())
    left_edge = overall_min if logx else 0

    if logx and (epicast_sizes.min() <= 0 or exaepi_sizes.min() <= 0):
        sys.exit(f"--logx requires strictly positive sizes, but {title} has a minimum of "
                 f"{min(epicast_sizes.min(), exaepi_sizes.min())}")

    def print_stats(name, sizes, counts=None):
        # Weighted by size (each group of size s stands in for s members who experience that
        # group size) rather than one point per group -- a plain per-group histogram makes the
        # many small groups look dominant even when most members are actually in a big one, so
        # both the plot and its summary stats are weighted throughout by weight_noun. Printed
        # rather than shown in the legend -- this figure is only ~3.1in wide in the paper, with no
        # room for it at PLOS's 8-12pt font floor, and the legend should just name the series.
        # counts lets one entry stand for many real groups (the CBP bands are stored that way);
        # without it every entry is one group, which is how both model series are stored.
        n_groups = len(sizes) if counts is None else counts.sum()
        member_w = sizes if counts is None else sizes * counts
        weighted_mean = np.average(sizes, weights=member_w)
        order = np.argsort(sizes)
        sorted_sizes, cum_members = sizes[order], np.cumsum(member_w[order])
        weighted_median = sorted_sizes[np.searchsorted(cum_members, cum_members[-1] / 2)]
        print(
            f"{title} -- {name}: n={n_groups:,.0f}, mean={weighted_mean:.1f}, "
            f"median={weighted_median:.1f}, max={sizes.max():,.0f}"
        )

    if cdf:
        for sizes, color, label in (
            (epicast_sizes, "blue", "Epicast"),
            (exaepi_sizes, "red", "ExaEpi"),
        ):
            print_stats(label, sizes)
            sorted_sizes = np.sort(sizes)
            # Weighted cumulative fraction: cumsum(sorted_sizes) at position i is exactly "how
            # many workers are in a group of size <= sorted_sizes[i]" (each group's own size is
            # both its x-value and its worker-count contribution), divided by the total worker
            # count to normalize to a fraction.
            cumulative_frac = np.cumsum(sorted_sizes) / sorted_sizes.sum()
            # alpha<1, like the histogram's fill, so an overlapping segment blends to a visibly
            # distinct color instead of the later-drawn line fully hiding the other
            ax.step(sorted_sizes, cumulative_frac, where="post", color=color, linewidth=1,
                    alpha=0.7, label=label)
        if cbp is not None:
            order = np.argsort(cbp[0])
            s, w = cbp[0][order], (cbp[0] * cbp[1])[order]
            ax.step(s, np.cumsum(w) / w.sum(), where="post", color="black", linewidth=1,
                    label="CBP establishments")
        ax.set_ylabel(f"Cumulative fraction of {weight_noun}s")
    else:
        # Shared, density-normalized bins (the two models produce very different group counts,
        # so only a density comparison is fair) -- one bin per integer when the combined span is
        # small enough to stay readable, else a fixed bin count, matching
        # plot_group_size_histogram.py's convention.
        combined_max = max(epicast_sizes.max(), exaepi_sizes.max())
        combined_min = overall_min
        # When zooming with xlim, size the bins for the zoomed-in range rather than the full
        # data range -- otherwise a few extreme outliers (e.g. a university) set a bin width
        # so wide that only one or two giant bins are even visible in the zoomed view. A final
        # catch-all bin (invisible once xlim clips the view) keeps all the data -- including
        # the outliers -- in the density normalization.
        bin_max = min(combined_max, xlim) if xlim is not None else combined_max
        if logx:
            # Linear (equal-width) bins would render as ever-narrower, unreadable slivers
            # once the x axis is log-scaled, since most of them get squeezed into the
            # rightmost decade. Log-spaced bins keep them visually even instead.
            bins = log_spaced_integer_bins(combined_min, bin_max)
        else:
            span = int(bin_max - combined_min)
            bins = (
                np.arange(combined_min - 0.5, bin_max + 1.5, 1.0)
                if span <= max_integer_bins
                # Sized so the tighter of the two distributions is resolved too, not just the
                # combined range, and no finer than the sparser one can support -- see
                # nice_linear_bins.
                else nice_linear_bins(
                    combined_min, bin_max,
                    resolve_span=min(core_span(epicast_sizes), core_span(exaepi_sizes)),
                    sparsest_count=min((epicast_sizes <= bin_max).sum(), (exaepi_sizes <= bin_max).sum()),
                )
            )
            # nice_linear_bins() anchors bin centers to global multiples of the bin width (so
            # they land on the same "nice" values matplotlib's tick locator picks -- see its
            # docstring), which can put a bin's center at/near 0 even though the data's own
            # minimum is well above it. Forcing the view to start exactly at 0 would then clip
            # that bin in half; starting it at the bin's own left edge instead always shows the
            # full first bar, at the cost of a little empty margin left of 0 when this happens.
            left_edge = bins[0]
        if bin_max < combined_max and not isinstance(bins, int):
            # nice_linear_bins() (and the integer scheme) can both overshoot bin_max by up to
            # one bin width, which -- for an xlim close enough to the true max -- can already
            # exceed combined_max. Drop any such edges before appending it, or the result isn't
            # monotonically increasing and numpy.histogram rejects it outright.
            bins = np.append(bins[bins < combined_max], combined_max)
        print_stats("Epicast", epicast_sizes)
        print_stats("ExaEpi", exaepi_sizes)
        if cbp is not None:
            print_stats("CBP establishments", cbp[0], cbp[1])
        ax.hist(epicast_sizes, bins=bins, weights=epicast_sizes, density=True, color="blue",
                alpha=0.5, label="Epicast")
        ax.hist(exaepi_sizes, bins=bins, weights=exaepi_sizes, density=True, color="red",
                alpha=0.5, label="ExaEpi")
        if cbp is not None:
            # Outline rather than a third filled patch -- this is a reference curve for what real
            # workplaces look like, not a third model, and two translucent fills are already
            # overlapping here. Establishments larger than the last bin fall outside `bins` and
            # so are dropped by numpy.histogram, which renormalizes this curve over the plotted
            # range; that is the intended comparison (the models' own groups are capped far
            # below CBP's tail) but it does mean the curve is conditional on that range.
            ax.hist(cbp[0], bins=bins, weights=cbp[0] * cbp[1], density=True, histtype="step",
                    color="black", linewidth=1, label="CBP establishments")
        ax.set_ylabel(f"Density ({weight_noun}-weighted)")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (the default, when xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=left_edge, right=xlim)
    # Headroom above the tallest bar/curve for the legend to sit in -- "best" placement
    # sometimes has nowhere left to go but on top of a peak, especially with a two-line label
    # per series.
    #ax.set_ylim(top=ax.get_ylim()[1] * 1.25)

    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend()


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prefix", "-p", required=True,
        help="ExaEpi's --aggregated_diag_prefix (matching the run's <prefix>_workgroup_sizes.txt / "
        "_class_sizes.txt / _school_sizes.txt files, written when --aggregated_diag_int is enabled "
        "-- see ExaEpi::IO::writeStaticAggregatedData in src/IO.cpp)",
    )
    parser.add_argument(
        "--groups", "-g", nargs="+", choices=list(GROUP_INFO), default=list(GROUP_INFO),
        help="Which group-size distributions to plot (default: all three)",
    )
    parser.add_argument(
        "--epicast_workgroup", default=GROUP_INFO["workgroup"]["epicast_file"],
        help="Epicast workgroup sizes file (one size per line)",
    )
    parser.add_argument(
        "--epicast_class", default=GROUP_INFO["class"]["epicast_file"],
        help="Epicast school-class (schoolgroup) sizes file (one size per line)",
    )
    parser.add_argument(
        "--epicast_school", default=GROUP_INFO["school"]["epicast_file"],
        help="Epicast school sizes file (one size per line)",
    )
    parser.add_argument(
        "--histogram", action="store_true",
        help="Plot density histograms instead of the default CDF. Group sizes here span "
        "several orders of magnitude, and a log-x histogram's bins can't be both uniform "
        "width and finer than the ~1.5x-per-bin resolution set by the smallest sizes -- the "
        "CDF has no such trade-off, so it's the default.",
    )
    parser.add_argument(
        "--logx", action="store_true", help="Use a logarithmic x-axis",
    )
    parser.add_argument(
        "--logy", action="store_true", help="Use a logarithmic y-axis",
    )
    parser.add_argument(
        "--xlim", type=float, default=None, help="Maximum x-axis value to display",
    )
    parser.add_argument(
        "--output", "-o", default="group_size_comparison.png", help="Output image file",
    )
    parser.add_argument(
        "--cbp_state", type=int, default=None, metavar="FIPS",
        help="Overlay the real CBP establishment-size distribution for this state FIPS (6 = CA, "
        "35 = NM) on the workgroup panel. This is the WORKPLACE tier, not the workgroup tier -- "
        "a workgroup is a co-worker team inside an establishment -- so it is a reference for what "
        "real workplaces look like, not a target the workgroup sizes should match.",
    )
    parser.add_argument(
        "--cbp_sizes_file",
        default=os.path.join(REPO_ROOT, "data", "UrbanPop", "cbp19st_derived.csv"),
        help="CBP derived cache with establishment-size bands (see compute_workgroup_sizes.py)",
    )
    args = parser.parse_args()
    cdf = not args.histogram
    cbp = None
    if args.cbp_state is not None:
        cbp = load_cbp_establishment_sizes(args.cbp_sizes_file, args.cbp_state)

    epicast_files = {
        "workgroup": args.epicast_workgroup,
        "class": args.epicast_class,
        "school": args.epicast_school,
    }

    # Total width fixed at the paper's half-page width regardless of how many groups are plotted
    # side by side -- each panel just gets narrower as more are added (see plos_compbio_style.py).
    # Height stays the shared standard regardless of how many panels there are.
    fig, axes = plt.subplots(1, len(args.groups), figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    if len(args.groups) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, args.groups):
        info = GROUP_INFO[group_name]
        epicast_data = load_epicast_sizes(epicast_files[group_name])
        exaepi_data = exaepi_sizes(args.prefix, group_name)
        plot_comparison(ax, epicast_data, exaepi_data, info["xlabel"], info["title"], cdf,
                         weight_noun=info["weight_noun"], logx=args.logx, logy=args.logy,
                         xlim=args.xlim, cbp=cbp if group_name == "workgroup" else None)

    plt.savefig(args.output, dpi=300)
    print(f"{'CDF' if cdf else 'Histogram'} comparison saved to {args.output}")


if __name__ == "__main__":
    main()
