#!/usr/bin/env python

"""Map, by county (or tract), the share of the jobs located there that are held by workers living
more than --threshold km away, for Epicast and ExaEpi side by side on a shared color scale.

This is where long-distance commuting lands, which a net-flow map (plot_geo_daynight.py) can't
show: LODES' spurious long links run in both directions -- rural residents recorded as working in
the cities and city residents recorded as working in rural counties -- so they largely cancel in
the inflow minus outflow, but they make up a large share of the gross inflow of rural counties.
Those are the links that couple distant regional epidemics.

Epicast's commutes are its tract-to-tract worker flow file (<state>_workerflow.csv: src, dst,
flow), which is LODES; ExaEpi's are the home/work tracts of the commuters in its UrbanPop .bin (see
plot_commute_distance.exaepi_commuters). Distances are between tract internal points, as in
plot_commute_distance.py. The per-area table is also printed, largest Epicast share first.
"""

import argparse
import os
import sys

import geopandas as gp
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import FONT_TICK, FONT_TITLE, FULL_PAGE_WIDTH_IN, apply_style  # noqa: E402
from plot_commute_distance import load_exaepi_pairs, load_internal_points, pair_distances  # noqa: E402


def far_shares(pairs, points, threshold_km, label, county_level):
    """Per work area (county or tract GEOID): jobs, jobs held by workers living more than
    threshold_km away, and their share in percent."""
    keep = points.index
    pairs = pairs[pairs["home"].isin(keep) & pairs["work"].isin(keep)]
    dist, counts = pair_distances(pairs, points, label)
    work = pairs["work"].to_numpy()
    if county_level:
        work = work // 10**6  # 11-digit tract -> 5-digit county
    df = pd.DataFrame({"work": work, "jobs": counts, "far": np.where(dist > threshold_km, counts, 0)})
    df = df.groupby("work").sum()
    df["share"] = 100.0 * df["far"] / df["jobs"]
    total = 100.0 * df["far"].sum() / df["jobs"].sum()
    print(f"{label}: {int(df['jobs'].sum()):,} commuters, {total:.1f}% of them over {threshold_km:g} km")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urbanpop", "-u", required=True, help="ExaEpi UrbanPop .bin")
    parser.add_argument("--epicast_flows", "-f", required=True,
                        help="Epicast tract-to-tract worker flows (<state>_workerflow.csv: src, dst, flow)")
    parser.add_argument("--tracts", "-t", required=True, help="2010 Census tract shapefile (internal points)")
    parser.add_argument("--counties", "-c", required=True, help="2010 Census county shapefile, for the map")
    parser.add_argument("--tract_level", action="store_true",
                        help="Map tracts (drawn from --tracts) instead of counties")
    parser.add_argument("--threshold", type=float, default=100.0, help="Distance threshold in km (default 100)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Top of the color scale in percent (default: the largest share in either panel)")
    parser.add_argument("--output", "-o", default="long-commutes.png", help="Output file")
    args = parser.parse_args()

    apply_style()
    points = load_internal_points([args.tracts])
    flows = pd.read_csv(args.epicast_flows).rename(columns={"src": "home", "dst": "work", "flow": "count"})
    county_level = not args.tract_level
    shares = {
        "Epicast": far_shares(flows, points, args.threshold, "Epicast (LODES)", county_level),
        "ExaEpi": far_shares(load_exaepi_pairs(args.urbanpop), points, args.threshold, "ExaEpi", county_level),
    }

    if county_level:
        geo = gp.read_file(args.counties)
        names = dict(zip(geo["GEOID10"].astype(np.int64), geo["NAME10"]))
    else:
        geo = gp.read_file(args.tracts)
        names = {}
    geo["GEOID"] = geo["GEOID10"].astype(np.int64)
    counties = gp.read_file(args.counties)

    table = pd.concat({k: v[["jobs", "share"]] for k, v in shares.items()}, axis=1).fillna(0)
    table = table.sort_values(("Epicast", "share"), ascending=False)
    table.index = [names.get(i, i) for i in table.index]
    print(f"\nShare of jobs held by workers living over {args.threshold:g} km away (%):")
    print(table.round(1).head(25).to_string())

    vmax = args.vmax or max(v["share"].max() for v in shares.values())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = "YlOrRd"

    # Two maps side by side with a shared colorbar, placed by hand as in plot_geo_daynight.py:
    # geopandas fixes each map's data aspect, which constrained_layout handles badly.
    xmin, ymin, xmax, ymax = counties.total_bounds
    xrange, yrange = xmax - xmin, ymax - ymin
    fig_x = FULL_PAGE_WIDTH_IN
    top_margin_in = (FONT_TITLE * 1.3 + 4) / 72
    bottom_margin_in = FONT_TICK / 2 / 72
    cbar_w_in, cbar_gap_in, cbar_label_w_in = 0.12, 0.08, 0.75
    panel_gap_in = 0.1
    panel_w_in = (fig_x - cbar_w_in - cbar_gap_in - cbar_label_w_in - panel_gap_in) / 2
    map_h_in = panel_w_in * yrange / xrange
    fig_y = top_margin_in + map_h_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_x, fig_y))
    bottom, height = bottom_margin_in / fig_y, map_h_in / fig_y
    for i, (label, df) in enumerate(shares.items()):
        ax = fig.add_axes((i * (panel_w_in + panel_gap_in) / fig_x, bottom, panel_w_in / fig_x, height))
        merged = geo.merge(df[["share"]], left_on="GEOID", right_index=True, how="left")
        merged.plot(column="share", ax=ax, cmap=cmap, norm=norm, linewidth=0,
                    missing_kwds={"color": "lightgrey"})
        counties.boundary.plot(ax=ax, color="black", linewidth=0.2)
        ax.set_title(label, pad=3)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_axis_off()
    cax = fig.add_axes(((2 * panel_w_in + panel_gap_in + cbar_gap_in) / fig_x, bottom, cbar_w_in / fig_x, height))
    cbar = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.set_label(f"Jobs held by workers living\nover {args.threshold:g} km away (%)", fontsize=FONT_TICK)

    print("Plotting results to", args.output)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
