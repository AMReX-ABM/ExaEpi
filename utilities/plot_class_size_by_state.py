#!/usr/bin/env python

"""Compare mean class size across states for the school data, ExaEpi and Epicast, as grouped bars
(one group per source, one bar per state), with each source's ratio between the last state and
the first above its group. It shows at a glance whether a model's classes follow the data from one
state to another.

  - Data: students per teacher over the K-12 schools of each state (school levels made of P, E, M,
    H only; schools with fewer than 3 or more than 60 students per teacher are dropped as reporting
    errors), from the schools CSV ExaEpi's population is built from (get_schools.py).
  - ExaEpi: mean class size, from the <prefix>_class_sizes.txt a run writes (one size per line).
  - Epicast: mean school-group size, from its schoolgroup sizes file, leaving out groups of 1-2
    members, which appear to be the administrative groups of excess teachers.

Pass one --exaepi and one --epicast file per state, in the order of --states.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import FONT_LEGEND, HALF_PAGE_HEIGHT_IN, HALF_PAGE_WIDTH_IN, apply_style  # noqa: E402

STATE_FIPS = {"ca": "06", "nm": "35"}
STATE_NAMES = {"ca": "California", "nm": "New Mexico"}


def sizes(path):
    v = np.loadtxt(path, usecols=0)
    return v[v > 0]


def data_ratio(schools, fips):
    s = schools[(schools["geoid"].str[:2] == fips) & schools["level"].str.fullmatch(r"[PEMH]+")
                & (schools["students"] > 0) & (schools["teachers"] > 0)]
    r = s["students"] / s["teachers"]
    s = s[(r >= 3) & (r <= 60)]
    return s["students"].sum() / s["teachers"].sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", nargs="+", default=["ca", "nm"], choices=sorted(STATE_FIPS))
    parser.add_argument("--schools", required=True, help="Schools CSV (id, students, teachers, level, geoid, ...)")
    parser.add_argument("--exaepi", nargs="+", required=True, help="ExaEpi class sizes file per state")
    parser.add_argument("--epicast", nargs="+", required=True, help="Epicast schoolgroup sizes file per state")
    parser.add_argument("--output", "-o", default="class-size-by-state.png")
    args = parser.parse_args()
    if not len(args.states) == len(args.exaepi) == len(args.epicast):
        parser.error("give one --exaepi and one --epicast file per state")

    apply_style()
    schools = pd.read_csv(args.schools, dtype={"geoid": str, "id": str}, low_memory=False)
    values = {
        "Data": [data_ratio(schools, STATE_FIPS[st]) for st in args.states],
        "ExaEpi": [sizes(f).mean() for f in args.exaepi],
        "Epicast": [(lambda v: v[v > 2].mean())(sizes(f)) for f in args.epicast],
    }
    for src, vals in values.items():
        print(f"{src:<8}" + "  ".join(f"{st} {v:.1f}" for st, v in zip(args.states, vals))
              + f"  ratio {vals[-1] / vals[0]:.2f}")

    fig, ax = plt.subplots(figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    n = len(args.states)
    width = 0.8 / n
    colors = ["0.35", "0.75", "0.55", "0.9"]
    x = np.arange(len(values))
    for i, st in enumerate(args.states):
        ax.bar(x + (i - (n - 1) / 2) * width, [v[i] for v in values.values()], width,
               label=STATE_NAMES[st], color=colors[i], edgecolor="black", linewidth=0.5)
    top = max(max(v) for v in values.values())
    for j, vals in enumerate(values.values()):
        ax.text(x[j], max(vals) + 0.03 * top, f"{vals[-1] / vals[0]:.2f}$\\times$", ha="center", va="bottom",
                fontsize=FONT_LEGEND)
    ax.set_xticks(x, list(values))
    ax.set_ylabel("Mean class size (students)")
    ax.set_ylim(0, top * 1.38)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND, frameon=False, ncol=n)
    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
