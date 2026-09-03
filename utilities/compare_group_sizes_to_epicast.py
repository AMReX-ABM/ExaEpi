#!/usr/bin/env python

"""Compare Epicast vs. ExaEpi distributions of workgroup size, school-class size, and
school size for the NM run used in the emerge paper.

Epicast side: data/results/emerge-paper/epicast/epicast_nm_{workgroup,schoolgroup,school}_sizes.txt
-- plain text, one integer (group size) per line. "workgroup" is a workplace peer group,
"schoolgroup" is a classroom-level cohort, "school" is a whole school.

ExaEpi side: there is no direct per-group-size output, so this reads step-0 per-agent
fields (work_i, work_j, naics, workgroup, school_id, school_class_group -- see
read_exaepi_agents.py) from an ExaEpi plotfile and reconstructs the same three
distributions by grouping agents:

  - Workgroup size: agents with workgroup > 0 (0 means not assigned to a workgroup --
    not working, or working from home), grouped by (work_i, work_j, naics, workgroup).
    Workgroup IDs are only unique within a (work community, naics) pair -- see
    InteractionModWork.H's max_workgroup * max_naics sizing -- so all three keys are
    needed to recover each actual workgroup.

  - School class size: agents with naics == -1 (a student, not an employee) and
    school_class_group >= 0 (enrolled in a real classroom, not the -1 sentinel for
    unenrolled agents), grouped by school_class_group alone -- that field is already a
    globally unique, densely-packed ID for one (community, school_id, grade, class)
    mixing bucket (see AgentContainer::assignSchoolClasses / InteractionModSchool.H).
    Restricting to naics == -1 excludes each class's own homeroom teacher and any
    non-classroom "admin" pools of surplus teachers (which contain no naics == -1
    agents at all), matching the student headcount ExaEpi's own
    "School class size" log histogram reports.

  - School size: agents with school_id > 0 (both students and staff), grouped by
    (work_i, work_j, school_id) -- school_id is only unique within a community, like
    workgroup.

All three ExaEpi fields above are static per agent, so ExaEpi only writes them to the
first plotfile of a run (typically plt00000); see read_exaepi_agents.py's module
docstring.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from read_exaepi_agents import read_agent_fields

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPICAST_DIR = os.path.join(REPO_ROOT, "data", "results", "emerge-paper", "epicast")

# Per --group-name: default Epicast sizes file, the ExaEpi field(s) needed, and axis/label text.
GROUP_INFO = {
    "workgroup": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_workgroup_sizes.txt"),
        "xlabel": "Workgroup size (number of agents)",
        "title": "Workgroup size",
        "basename": "workgroup_sizes",
    },
    "class": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_schoolgroup_sizes.txt"),
        "xlabel": "Class size (number of students)",
        "title": "School class size",
        "basename": "class_sizes",
    },
    "school": {
        "epicast_file": os.path.join(EPICAST_DIR, "epicast_nm_school_sizes.txt"),
        "xlabel": "School size (number of agents)",
        "title": "School size",
        "basename": "school_sizes",
    },
}


def load_epicast_sizes(fname):
    sizes = np.loadtxt(fname, dtype=int)
    print(f"Read {len(sizes):,} sizes from Epicast file {fname}")
    return sizes


def exaepi_workgroup_sizes(fields):
    df = pd.DataFrame(fields)
    df = df[df["workgroup"] > 0]
    return df.groupby(["work_i", "work_j", "naics", "workgroup"]).size().to_numpy()


def exaepi_class_sizes(fields):
    df = pd.DataFrame(fields)
    df = df[(df["naics"] == -1) & (df["school_class_group"] >= 0)]
    return df.groupby("school_class_group").size().to_numpy()


def exaepi_school_sizes(fields):
    df = pd.DataFrame(fields)
    df = df[df["school_id"] > 0]
    return df.groupby(["work_i", "work_j", "school_id"]).size().to_numpy()


EXAEPI_FIELDS = {
    "workgroup": ("work_i", "work_j", "naics", "workgroup"),
    "class": ("naics", "school_class_group"),
    "school": ("work_i", "work_j", "school_id"),
}

EXAEPI_COMPUTE = {
    "workgroup": exaepi_workgroup_sizes,
    "class": exaepi_class_sizes,
    "school": exaepi_school_sizes,
}


def exaepi_sizes(plot_dir, group_name):
    fields = read_agent_fields(plot_dir, EXAEPI_FIELDS[group_name])
    sizes = EXAEPI_COMPUTE[group_name](fields)
    print(f"Found {len(sizes):,} ExaEpi {group_name}s from {plot_dir}")
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


def plot_comparison(ax, epicast_sizes, exaepi_sizes, xlabel, title, cdf, logx=False, logy=False,
                     max_integer_bins=200, xlim=None):
    epicast_sizes = np.asarray(epicast_sizes)
    exaepi_sizes = np.asarray(exaepi_sizes)
    overall_min = min(epicast_sizes.min(), exaepi_sizes.min())
    left_edge = overall_min if logx else 0

    if logx and (epicast_sizes.min() <= 0 or exaepi_sizes.min() <= 0):
        sys.exit(f"--logx requires strictly positive sizes, but {title} has a minimum of "
                 f"{min(epicast_sizes.min(), exaepi_sizes.min())}")

    if cdf:
        for sizes, color, label in (
            (epicast_sizes, "blue", "Epicast"),
            (exaepi_sizes, "red", "ExaEpi"),
        ):
            sorted_sizes = np.sort(sizes)
            cumulative_frac = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
            # alpha<1, like the histogram's fill, so an overlapping segment blends to a visibly
            # distinct color instead of the later-drawn line fully hiding the other
            ax.step(sorted_sizes, cumulative_frac, where="post", color=color, linewidth=2,
                    alpha=0.7, label=label)
        ax.set_ylabel("Cumulative fraction")
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
                else nice_linear_bins(combined_min, bin_max)
            )
            # nice_linear_bins() anchors bin centers to global multiples of the bin width (so
            # they land on the same "nice" values matplotlib's tick locator picks -- see its
            # docstring), which can put a bin's center at/near 0 even though the data's own
            # minimum is well above it. Forcing the view to start exactly at 0 would then clip
            # that bin in half; starting it at the bin's own left edge instead always shows the
            # full first bar, at the cost of a little empty margin left of 0 when this happens.
            left_edge = bins[0]
        if bin_max < combined_max:
            # nice_linear_bins() (and the integer scheme) can both overshoot bin_max by up to
            # one bin width, which -- for an xlim close enough to the true max -- can already
            # exceed combined_max. Drop any such edges before appending it, or the result isn't
            # monotonically increasing and numpy.histogram rejects it outright.
            bins = np.append(bins[bins < combined_max], combined_max)
        ax.hist(epicast_sizes, bins=bins, density=True, color="blue", alpha=0.5,
                edgecolor="black", label="Epicast")
        ax.hist(exaepi_sizes, bins=bins, density=True, color="red", alpha=0.5,
                edgecolor="black", label="ExaEpi")
        ax.set_ylabel("Density")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (the default, when xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=left_edge, right=xlim)

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    stats_text = (
        f"Epicast: n={len(epicast_sizes):,}, mean={epicast_sizes.mean():.1f}, "
        f"median={np.median(epicast_sizes):.1f}, max={epicast_sizes.max():,}\n"
        f"ExaEpi:  n={len(exaepi_sizes):,}, mean={exaepi_sizes.mean():.1f}, "
        f"median={np.median(exaepi_sizes):.1f}, max={exaepi_sizes.max():,}"
    )
    text_y, valign = (0.35, "top") if cdf else (0.97, "top")
    ax.text(
        0.98, text_y, stats_text, transform=ax.transAxes, fontsize=8,
        verticalalignment=valign, horizontalalignment="right", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--plot_dir", "-p", required=True,
        help="ExaEpi step-0 plotfile directory (e.g. plt00000) to read agent fields from",
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
    args = parser.parse_args()
    cdf = not args.histogram

    epicast_files = {
        "workgroup": args.epicast_workgroup,
        "class": args.epicast_class,
        "school": args.epicast_school,
    }

    fig, axes = plt.subplots(1, len(args.groups), figsize=(6 * len(args.groups), 5))
    if len(args.groups) == 1:
        axes = [axes]

    for ax, group_name in zip(axes, args.groups):
        info = GROUP_INFO[group_name]
        epicast_data = load_epicast_sizes(epicast_files[group_name])
        exaepi_data = exaepi_sizes(args.plot_dir, group_name)
        plot_comparison(ax, epicast_data, exaepi_data, info["xlabel"], info["title"], cdf,
                         logx=args.logx, logy=args.logy, xlim=args.xlim)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"{'CDF' if cdf else 'Histogram'} comparison saved to {args.output}")


if __name__ == "__main__":
    main()
