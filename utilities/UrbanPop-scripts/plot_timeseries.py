#!/usr/bin/env python

import sys
import time
import argparse
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import functools


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser(description="Plot time series")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="Time series csv files")
    parser.add_argument("--labels", "-l", nargs="+", help="Labels for each file")
    parser.add_argument(
        "--output",
        "-o",
        default="timeseries.pdf",
        help="Output file for timeseries plots (choose extension for format, e.g. pdf, png, etc",
    )
    args = parser.parse_args()

    plt.rc("font", size=16)
    px = 1.0 / plt.rcParams["figure.dpi"]
    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(1870 * px, 2000 * px))
    # _, (ax1, ax6) = plt.subplots(1, 2, figsize=(1000*px, 600*px))
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Infected")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Asymptomatic")
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Hospitalized")
    ax4.set_xlabel("Days")
    ax4.set_ylabel("ICU")
    ax5.set_xlabel("Days")
    ax5.set_ylabel("Ventilated")
    ax6.set_xlabel("Days")
    ax6.set_ylabel("Deaths")

    for i, fname in enumerate(args.files):
        df = pd.read_csv(fname, delimiter=r"\s+")
        print("Read", len(df), "records in %.3f s" % (time.time() - t))
        if args.labels != None and i < len(args.labels):
            label = args.labels[i]
        else:
            label = fname
        # ls = ""
        # marker = "+"
        # alpha = 0.3
        # if "gen" in fname:
        #    c = "green"
        # else:
        #    c = "blue"
        ax1.plot(
            list(df.index),
            list(df.Infected),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )
        ax2.plot(
            list(df.index),
            list(df.Asymptomatic),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )
        ax3.plot(
            list(df.index),
            list(df.Hospitalized),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )
        ax4.plot(
            list(df.index),
            list(df.ICU),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )
        ax5.plot(
            list(df.index),
            list(df.Ventilated),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )
        ax6.plot(
            list(df.index),
            list(df.Deaths.diff()),
            label=label,
            lw=3,
            # ls=ls,
            # alpha=alpha,
            # marker=marker,
            # color=c,
        )

    # ax1.text(
    #    0.02,
    #    0.98,
    #    "Generated",
    #    color="green",
    #    transform=ax1.transAxes,
    #    verticalalignment="top",
    #    fontsize=14,
    # )
    # ax1.text(
    #    0.02,
    #    0.93,
    #    "UrbanPop NT/DT",
    #    color="blue",
    #    transform=ax1.transAxes,
    #    verticalalignment="top",
    #    fontsize=14,
    # )
    # ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax6.grid()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to '{args.output}'")
