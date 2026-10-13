#!/usr/bin/env python

"""Plot each county's peak day against its distance from the seed county, for groups of ExaEpi
runs and an Epicast run, to show how fast the epidemic spreads out from where it was seeded.

A county's peak day is the day of the maximum of its residents' 7-day centered mean daily new
infections. For an ExaEpi group, it is the mean over the group's runs (each run directory holding
the run's per-day aggregated files, cases00000, cases00001, ...); for Epicast, that of the single
run in its events file. A county's distance is between population-weighted centroids of its block
groups' internal points (residents from ExaEpi's per-day files), and the seed county is plotted at
distance 0. Each group's county peak lag (the population-weighted mean peak day of the other
counties minus the seed county's) is printed.

Example:
    plot_county_peaks.py -x "nm-flowswap/ctpp-s1/r*:ExaEpi" -x "nm-flowswap/lodes-s1/r*:ExaEpi, LODES flows" \\
        -e ../epicast/nm/nm-p01-r3-run_000.events.bin:Epicast \\
        -b ../../../US_2010_Census_BlockGroups/tl_2010_35_bg10.shp.csv --seed_county 35001 -o nm-county-peaks.png
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import PAGE_WIDTHS_IN, apply_style  # noqa: E402

ECOLS = ["EHH", "ENC", "ENbhD", "ECommD", "ENbhN", "ECommN", "EWork", "ESchool"]
# the default colors of the -x groups; --colors overrides them
EXAEPI_COLORS = ["red", "darkorange", "purple", "saddlebrown"]
EXAEPI_MARKERS = ["o", "s", "^", "D"]


def smooth(df):
    return df.rolling(7, center=True, min_periods=1).mean()


def exaepi_run(rundir):
    """Per-county peak day, residents per county and per-block-group residents of one run."""
    rows, bg_pop = {}, None
    for f in sorted(glob.glob(os.path.join(rundir, "cases0*"))):
        d = pd.read_csv(f, usecols=["GEOID", "total"] + ECOLS)
        rows[int(f[-5:])] = d[ECOLS].sum(axis=1).groupby(d.GEOID // 10**7).sum()
        if bg_pop is None:
            bg_pop = d.set_index("GEOID")["total"]
    return smooth(pd.DataFrame(rows).T.sort_index().fillna(0)).idxmax(), bg_pop


def epicast_run(events_file):
    from plot_geo import epicast_home_tracts
    from read_epicast_events import read_events_bin
    ev, dem = read_events_bin(events_file)
    inf = ev[ev.disease_state == "exposed"]
    daily = pd.crosstab(inf.timestep.to_numpy() // 2, epicast_home_tracts(inf, dem) // 10**6)
    return smooth(daily).idxmax()


def county_centroids(points_file, bg_pop):
    pts = pd.read_csv(points_file, dtype={"GEOID10": np.int64}).set_index("GEOID10")
    pts = pts.join(bg_pop.rename("pop"), how="inner")
    pts["county"] = pts.index // 10**7
    w = pts["pop"].clip(lower=1e-9)
    lat = (pts.INTPTLAT10 * w).groupby(pts.county).sum() / w.groupby(pts.county).sum()
    lon = (pts.INTPTLON10 * w).groupby(pts.county).sum() / w.groupby(pts.county).sum()
    return lat, lon, pts["pop"].groupby(pts.county).sum()


def km_from(lat, lon, seed):
    la1, lo1, la2, lo2 = map(np.radians, (lat[seed], lon[seed], lat, lon))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def lag(peak, pop, seed):
    rest = peak.index[peak.index != seed]
    return np.average(peak[rest], weights=pop.reindex(rest).fillna(0)) - peak[seed]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exaepi", "-x", action="append", default=[], metavar="RUNDIR_GLOB:LABEL",
                        help="Glob of ExaEpi run directories (per-day files inside), with a label. Can be repeated.")
    parser.add_argument("--epicast", "-e", metavar="EVENTS_FILE:LABEL", help="Epicast events file, with a label")
    parser.add_argument("--blockgroups", "-b", required=True,
                        help="Block-group internal points CSV (GEOID10, INTPTLAT10, INTPTLON10)")
    parser.add_argument("--seed_county", type=int, required=True, help="FIPS of the seed county, e.g. 35001")
    parser.add_argument("--colors", nargs="+", default=EXAEPI_COLORS, metavar="COLOR",
                        help="Colors of the -x groups, in order")
    parser.add_argument("--width", choices=list(PAGE_WIDTHS_IN), default="half")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    apply_style()

    groups, bg_pop = [], None
    for spec in args.exaepi:
        pattern, label = spec.rsplit(":", 1)
        runs = [exaepi_run(d) for d in sorted(glob.glob(pattern))]
        if not runs:
            sys.exit(f"no run directories match {pattern}")
        bg_pop = runs[0][1] if bg_pop is None else bg_pop
        groups.append((f"{label}", pd.concat([r[0] for r in runs], axis=1).mean(axis=1), len(runs)))
    lat, lon, pop = county_centroids(args.blockgroups, bg_pop)
    dist = km_from(lat, lon, args.seed_county)

    fig, ax = plt.subplots(figsize=PAGE_WIDTHS_IN[args.width])
    if args.epicast:
        events_file, label = args.epicast.rsplit(":", 1)
        peak = epicast_run(events_file)
        c = peak.index.intersection(dist.index)
        ax.scatter(dist[c], peak[c], s=12, marker="o", facecolors="none", edgecolors="blue", linewidths=0.8,
                   label=label, zorder=3)
        print(f"{label}: county peak lag {lag(peak, pop, args.seed_county):.1f} days")
    for i, (label, peak, n) in enumerate(groups):
        c = peak.index.intersection(dist.index)
        ax.scatter(dist[c], peak[c], s=10, marker=EXAEPI_MARKERS[i % 4], color=args.colors[i % len(args.colors)], alpha=0.8,
                   linewidths=0, label=label, zorder=2)
        print(f"{label} ({n} runs): county peak lag {lag(peak, pop, args.seed_county):.1f} days")
    ax.set_xlabel("Distance from the seed county (km)")
    ax.set_ylabel("County peak day")
    # the near counties peak early in every model, so the top left is clear for the legend
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.55 * (hi - lo))
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
