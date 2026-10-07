#!/usr/bin/env python

"""Compare K-12 class size across states for the school data, ExaEpi and Epicast, as grouped bars
(one group per source, one bar per state), with each source's ratio of the first state to the
last (CA/NM by default) written above its first state's bar. It shows at a glance whether a model's
classes follow the data from one state to another.

Every bar is the class size the average student is in -- the member-weighted mean,
sum(n^2) / sum(n) over classes of n students -- since each classmate is a source of infection, so
that is what per-classmate transmission responds to. Students only, K-12 with preschool, which is
what Epicast's schools cover, and counted exactly as the class panels of
compare_group_sizes_to_epicast.py count them (the same numbers as its printed means):

  - Data: each K-12 school's student-teacher ratio, standing for `teachers` classes of that size
    (compare_group_sizes_to_epicast.load_school_class_sizes), from the schools CSV ExaEpi's
    population is built from (get_schools.py).
  - ExaEpi: the K-12 classes in its UrbanPop .bin, by their students
    (compare_group_sizes_to_epicast.exaepi_sizes), leaving out childcare and colleges.
  - Epicast: students per class, from its schoolgroup sizes file, which counts every member:
    leaving out groups of 1-2 members, which appear to be the administrative groups of excess
    teachers, and taking the one teacher out of each class
    (compare_group_sizes_to_epicast.epicast_class_students).

Pass one --urbanpop and one --epicast file per state, in the order of --states. ExaEpi also gets a
second group of bars, over all of its classes, childcare and colleges included: its school
transmission includes them, so that is the measure its overall school transmission responds to.
The data have no counterpart to it, since their childcare and college teacher counts aren't a
faculty headcount (see load_school_class_sizes).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_group_sizes_to_epicast import epicast_class_students, exaepi_sizes, load_school_class_sizes  # noqa: E402
from plos_compbio_style import FONT_LEGEND, HALF_PAGE_HEIGHT_IN, HALF_PAGE_WIDTH_IN, apply_style  # noqa: E402

STATE_FIPS = {"ca": 6, "nm": 35}
STATE_NAMES = {"ca": "California", "nm": "New Mexico"}


def member_weighted(sizes, counts=None):
    """sum(n^2) / sum(n) over classes, each size standing for `counts` classes (default 1)."""
    sizes = np.asarray(sizes, dtype=float)
    counts = np.ones_like(sizes) if counts is None else np.asarray(counts, dtype=float)
    return (counts * sizes * sizes).sum() / (counts * sizes).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", nargs="+", default=["ca", "nm"], choices=sorted(STATE_FIPS))
    parser.add_argument("--schools", required=True, help="Schools CSV (id, students, teachers, level, geoid, ...)")
    parser.add_argument("--urbanpop", nargs="+", required=True, help="ExaEpi UrbanPop .bin per state")
    parser.add_argument("--epicast", nargs="+", required=True, help="Epicast schoolgroup sizes file per state")
    parser.add_argument("--output", "-o", default="class-size-by-state.png")
    args = parser.parse_args()
    if not len(args.states) == len(args.urbanpop) == len(args.epicast):
        parser.error("give one --urbanpop and one --epicast file per state")

    apply_style()
    # group label -> one value per state; the labels are the x tick labels, so two lines each
    values = {"Data": [], "ExaEpi\nK-12": [], "ExaEpi\nall schools": [], "Epicast": []}
    for st, upb, epi in zip(args.states, args.urbanpop, args.epicast):
        values["Data"].append(member_weighted(*load_school_class_sizes(args.schools, [STATE_FIPS[st]])))
        classes = exaepi_sizes(upb, ["class"])[0]["class"]
        values["ExaEpi\nK-12"].append(member_weighted(classes["k12"]))
        values["ExaEpi\nall schools"].append(member_weighted(np.concatenate(list(classes.values()))))
        values["Epicast"].append(member_weighted(epicast_class_students(np.loadtxt(epi, dtype=int))))
    first, last = args.states[0], args.states[-1]
    for src, vals in values.items():
        print(f"{src.replace(chr(10), ' '):<20}" + "  ".join(f"{st} {v:.1f}" for st, v in zip(args.states, vals))
              + f"  ratio {first}/{last} {vals[0] / vals[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    n = len(args.states)
    width = 0.8 / n
    colors = ["0.35", "0.75", "0.55", "0.9"]
    x = np.arange(len(values))
    for i, st in enumerate(args.states):
        ax.bar(x + (i - (n - 1) / 2) * width, [v[i] for v in values.values()], width,
               label=STATE_NAMES[st], color=colors[i], edgecolor="black", linewidth=0.5)
    top = max(max(v) for v in values.values())
    # the ratio of the first state to the last, above the first state's bar -- but clear of the
    # group's tallest bar, which the label is wider than its own bar would otherwise run into
    for j, vals in enumerate(values.values()):
        ax.text(x[j] - (n - 1) / 2 * width, max(vals) + 0.03 * top, f"{vals[0] / vals[-1]:.2f}$\\times$", ha="center",
                va="bottom", fontsize=FONT_LEGEND)
    ax.set_xticks(x, list(values))
    ax.set_ylabel("Class size (students)")
    ax.set_ylim(0, top * 1.38)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND, frameon=False, ncol=n)
    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
