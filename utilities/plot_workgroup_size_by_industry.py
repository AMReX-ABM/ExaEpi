#!/usr/bin/env python

"""Plot the mean workgroup size in each 2-digit NAICS industry sector for ExaEpi, Epicast and the
2019 County Business Patterns (CBP) data both models size their workgroups from.

The mean is per group -- workers divided by groups -- which is the same definition as CBP's
employees per establishment, so the three series are directly comparable. (The worker-weighted
mean compare_group_sizes_to_epicast.py prints is a different, larger number.)

  - ExaEpi: every workgroup in the UrbanPop .bin the run reads. work_group is a globally dense id
    for one (work block group, NAICS, workgroup) team (see UrbanPop-scripts/group_assignment.py),
    and ExaEpi takes it straight from the file, so grouping agents by it reproduces the run's own
    <prefix>_workgroup_sizes.txt exactly. Only agents at a workplace have one: educators (who are
    in school classes instead) and declared work-from-home agents are excluded.
  - Epicast: Epicast reads the same UrbanPop population, but its workgroup-sizes dump
    (<state>_workgroup_sizes.txt, one size per line) does not say which industry each group is
    in, so on its own it only gives the all-industries mean. Pass --epicast_by_industry, a dump
    with a NAICS code next to each group's size, to plot Epicast per sector as well --
    estimate_epicast_workgroups.py writes an estimate of one from an events file (label it with
    --epicast_label). The per-industry target Epicast sizes groups from (Epicast 2.0 sec. 2.8.3)
    is no substitute on its own: its groups average 5.95 workers on CA, against 14.8 for the
    target, since every industry in every community is split separately.
  - CBP: employment / establishments for the state's 2-digit sector rows in the CBP derived cache
    (data/UrbanPop/cbp19st_derived.csv, written by compute_workgroup_sizes.py). These are
    WORKPLACES, not workgroups -- a hospital is one establishment but many co-worker teams -- and
    both models also cap each detailed industry's target at 86 (compute_workgroup_sizes.py), so
    CBP is the reference the targets start from, not a value the models should reproduce.
    CBP does not cover public administration (NAICS 92), so that sector has no CBP point.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

# This script only ever saves figures to a file, never displays them -- force the non-interactive
# Agg backend so rendering never touches an X server. Must happen before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UrbanPop-scripts"))
from compute_workgroup_sizes import load_cbp_cache, parse_naics_descriptions  # noqa: E402
from plos_compbio_style import apply_style, HALF_PAGE_HEIGHT_IN, HALF_PAGE_WIDTH_IN  # noqa: E402
from plot_commute_distance import read_urbanpop_columns  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2-digit NAICS sectors, keyed by the code CBP reports each one under (the multi-code sectors
# 31-33, 44-45 and 48-49 appear in CBP under their first code only).
SECTOR_NAMES = {
    "11": "Agriculture & forestry",
    "21": "Mining, oil & gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "42": "Wholesale trade",
    "44": "Retail trade",
    "48": "Transport & warehousing",
    "51": "Information",
    "52": "Finance & insurance",
    "53": "Real estate",
    "54": "Professional & technical",
    "55": "Management of companies",
    "56": "Admin. & waste services",
    "61": "Educational services",
    "62": "Health care & social asst.",
    "71": "Arts & recreation",
    "72": "Accommodation & food",
    "81": "Other services",
    "92": "Public administration",
}

# Second and third codes of the multi-code sectors, plus UrbanPop's single-digit placeholders for
# Manufacturing and Retail Trade (see ALIASES in compute_workgroup_sizes.py).
SECTOR_ALIASES = {"32": "31", "33": "31", "45": "44", "49": "48", "3": "31", "4": "44"}

ALL_INDUSTRIES = "All industries"

# Font size of the sector names on the y axis, in points. Smaller than the shared style's 8pt
# tick labels so ~21 rows fit in a standard half-page figure's height.
SECTOR_LABEL_FONT = 6

# Two-letter postal abbreviation -> state FIPS code, as CBP's fipstate column uses.
STATE_FIPS = {
    "al": 1, "ak": 2, "az": 4, "ar": 5, "ca": 6, "co": 8, "ct": 9, "de": 10, "dc": 11, "fl": 12,
    "ga": 13, "hi": 15, "id": 16, "il": 17, "in": 18, "ia": 19, "ks": 20, "ky": 21, "la": 22,
    "me": 23, "md": 24, "ma": 25, "mi": 26, "mn": 27, "ms": 28, "mo": 29, "mt": 30, "ne": 31,
    "nv": 32, "nh": 33, "nj": 34, "nm": 35, "ny": 36, "nc": 37, "nd": 38, "oh": 39, "ok": 40,
    "or": 41, "pa": 42, "ri": 44, "sc": 45, "sd": 46, "tn": 47, "tx": 48, "ut": 49, "vt": 50,
    "va": 51, "wa": 53, "wv": 54, "wi": 55, "wy": 56,
}


def sector_of(naics_code):
    """The SECTOR_NAMES key for a NAICS code at any level (3, 31, 311, 3113, ...)."""
    code = str(naics_code).strip()
    if code in SECTOR_ALIASES:
        return SECTOR_ALIASES[code]
    return SECTOR_ALIASES.get(code[:2], code[:2])


def group_means(sectors, sizes):
    """{sector: workers / groups} for one group per entry, plus the ALL_INDUSTRIES mean."""
    sectors, sizes = np.asarray(sectors), np.asarray(sizes)
    means = {s: sizes[sectors == s].mean() for s in np.unique(sectors)}
    means[ALL_INDUSTRIES] = sizes.mean()
    return means


def exaepi_group_means(urbanpop_bin, naics_header):
    """Per-sector mean workgroup size for every workgroup in the UrbanPop .bin -- see the module
    docstring for how the groups are identified and who is in one."""
    codes = np.array(parse_naics_descriptions(Path(naics_header)))
    cols = read_urbanpop_columns(urbanpop_bin, ["work_group", "naics"])
    member = cols["work_group"] >= 0
    groups, first, sizes = np.unique(cols["work_group"][member], return_index=True, return_counts=True)
    # Every member of a group shares its NAICS code (a group is one work block group, NAICS and
    # workgroup), so any one member's code is the group's.
    naics = cols["naics"][member][first]
    print(f"ExaEpi: {len(groups):,} workgroups, {int(sizes.sum()):,} workers")
    sector_by_code = np.array([sector_of(c) for c in codes])
    return group_means(sector_by_code[naics], sizes)


def load_epicast_sizes(fname):
    """The all-industries mean from an Epicast sizes file with one group size per line."""
    sizes = np.loadtxt(fname, dtype=int)
    print(f"Epicast: {len(sizes):,} workgroups, {int(sizes.sum()):,} workers, from {fname}")
    return {ALL_INDUSTRIES: sizes.mean()}


def load_epicast_by_industry(fname):
    """Per-sector means from an Epicast dump with one "naics size" pair (whitespace or comma
    separated) per workgroup. Lines that do not start with a number, such as a header, are
    skipped."""
    sectors, sizes = [], []
    with open(fname) as f:
        for line in f:
            fields = line.replace(",", " ").split()
            if len(fields) < 2 or not fields[0].isdigit():
                continue
            sectors.append(sector_of(fields[0]))
            sizes.append(int(fields[1]))
    if not sizes:
        sys.exit(f"error: no 'naics size' rows in {fname}")
    print(f"Epicast: {len(sizes):,} workgroups, {sum(sizes):,} workers, by industry from {fname}")
    return group_means(sectors, sizes)


def cbp_means(cbp_file, state_fips):
    """Per-sector employees per establishment for one state, plus the all-industries mean. Only
    2-digit rows are used: CBP reports every level of the NAICS hierarchy as its own row, and the
    2-digit sectors partition the state's establishments exactly once."""
    emp_est = {sector_of(naics): v for (fips, naics), v in load_cbp_cache(Path(cbp_file)).items()
               if int(fips) == state_fips and len(naics) == 2}
    if not emp_est:
        sys.exit(f"error: no CBP rows for state FIPS {state_fips} in {cbp_file}")
    means = {s: emp / est for s, (emp, est) in emp_est.items() if s in SECTOR_NAMES and est > 0}
    means[ALL_INDUSTRIES] = sum(e for e, _ in emp_est.values()) / sum(n for _, n in emp_est.values())
    return means


def print_table(rows, series):
    widths = [max(10, len(label) + 2) for label, _means, _style in series]
    print(f"\n{'Sector':<30}" + "".join(f"{label:>{w}}" for (label, _m, _s), w in zip(series, widths)))
    for key, name in rows:
        vals = "".join(f"{means[key]:>{w}.1f}" if key in means else f"{'-':>{w}}"
                       for (_l, means, _s), w in zip(series, widths))
        print(f"{name:<30}{vals}")


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", "-s", type=str.lower, choices=sorted(STATE_FIPS), default="ca",
                        help="Two-letter state abbreviation; selects the state's CBP rows and the default "
                        "--urbanpop and --epicast_sizes files")
    parser.add_argument("--urbanpop", "-u", default=None,
                        help="UrbanPop .bin the ExaEpi run read (agent.urbanpop_filename; default "
                        "data/UrbanPop/urbanpop_<state>.bin)")
    parser.add_argument("--naics_header", default=os.path.join(REPO_ROOT, "src", "UrbanPopAgentStruct.H"),
                        help="Generated header whose naics_descriptions maps the .bin's NAICS indices to codes")
    parser.add_argument("--epicast_sizes", "-e", default=None,
                        help="Epicast workgroup sizes, one per line, for the all-industries mean (default "
                        "data/results/emerge-paper/epicast/<state>/<state>_workgroup_sizes.txt)")
    parser.add_argument("--epicast_by_industry", default=None,
                        help="Epicast workgroups as 'naics size' pairs, one per line, to plot Epicast per sector "
                        "(replaces --epicast_sizes), e.g. as written by estimate_epicast_workgroups.py")
    parser.add_argument("--epicast_label", default="Epicast",
                        help="Legend label for the Epicast series, e.g. 'Epicast (est.)' for an estimate")
    parser.add_argument("--cbp_file", default=os.path.join(REPO_ROOT, "data", "UrbanPop", "cbp19st_derived.csv"),
                        help="CBP derived cache (see compute_workgroup_sizes.py)")
    parser.add_argument("--output", "-o", default="workgroup_size_by_industry.png", help="Output image file")
    args = parser.parse_args()
    urbanpop = args.urbanpop or os.path.join(REPO_ROOT, "data", "UrbanPop", f"urbanpop_{args.state}.bin")
    epicast_sizes = args.epicast_sizes or os.path.join(REPO_ROOT, "data", "results", "emerge-paper", "epicast",
                                                       args.state, f"{args.state}_workgroup_sizes.txt")

    exaepi = exaepi_group_means(urbanpop, args.naics_header)
    epicast = (load_epicast_by_industry(args.epicast_by_industry) if args.epicast_by_industry
               else load_epicast_sizes(epicast_sizes))
    cbp = cbp_means(args.cbp_file, STATE_FIPS[args.state])

    # Same colors as compare_group_sizes_to_epicast.py (Epicast blue, ExaEpi red, CBP black), and a
    # different marker for each so the series are not told apart by color alone.
    series = [
        ("ExaEpi", exaepi, dict(color="red", marker="o")),
        (args.epicast_label, epicast, dict(color="blue", marker="s")),
        ("CBP", cbp, dict(color="black", marker="D", markerfacecolor="none")),
    ]

    # Largest CBP establishments at the top; sectors with no CBP row (public administration) last.
    sectors = sorted((s for s in SECTOR_NAMES if s in exaepi or s in epicast),
                     key=lambda s: (s not in cbp, -cbp.get(s, 0)))
    rows = [(ALL_INDUSTRIES, ALL_INDUSTRIES)] + [(s, SECTOR_NAMES[s]) for s in sectors]
    print_table(rows, series)

    # Same height as the other half-page figures, which leaves ~6.5pt per sector row -- hence sector
    # labels and markers below the shared style's sizes.
    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    y = {key: i for i, (key, _name) in enumerate(rows)}
    for label, means, style in series:
        keys = [key for key, _name in rows if key in means]
        ax.plot([means[k] for k in keys], [y[k] for k in keys], linestyle="none", markersize=3,
                markeredgewidth=0.6, alpha=0.8, label=label, **style)
    ax.set_yticks(range(len(rows)), [name for _key, name in rows], fontsize=SECTOR_LABEL_FONT)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.axhline(0.5, color="0.5", linewidth=0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    # Room past the largest mean so its marker isn't cut in half by the right spine.
    ax.set_xlim(0, 1.08 * max(max(means.values()) for _label, means, _style in series))
    ax.set_xlabel("Mean workgroup size")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")

    plt.savefig(args.output)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
