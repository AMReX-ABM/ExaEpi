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
    parser.add_argument(
        "--cloud", "-c", action="store_true", help="Plot results as a cloud of markers"
    )
    args = parser.parse_args()

    plt.rc("font", size=16)
    px = 1.0 / plt.rcParams["figure.dpi"]
    _, axes = plt.subplots(3, 1, figsize=(1100 * px, 1600 * px))
    state_maps = {
        "Infected": ["PS/PI", "S/PI", "S/I", "PS/I", "A/PI", "A/I", "H/I"],
        "Hospitalized": ["H/NI", "H/I", "ICU", "V"],
        "Dead": ["D"],
    }
    for i, state in enumerate(state_maps.keys()):
        axes[i].set_xlabel("Days")
        axes[i].set_ylabel(state)

    for i, fname in enumerate(args.files):
        df = pd.read_csv(fname, delimiter=r"\s+")
        print("Read", len(df), "records in %.3f s" % (time.time() - t))
        for state, columns in state_maps.items():
            df[state] = df[columns].sum(axis=1)
            # if state == "Dead":
            #    df.Deaths = df.Deaths.diff().fillna(0)

        label = args.labels[i] if args.labels != None and i < len(args.labels) else fname
        c = "green" if "gen" in fname else "blue"
        if args.cloud:
            kwargs = {"label": label, "lw": 3, "ls": "", "alpha": 0.3, "marker": "+", "color": c}
        else:
            kwargs = {"label": label, "lw": 3}
        for ax_i, state in enumerate(state_maps.keys()):
            axes[ax_i].plot(list(df.index), list(df[state]), **kwargs)

    if args.cloud:
        axes[0].text(
            0.02,
            0.98,
            "Generated",
            color="green",
            transform=axes[0].transAxes,
            verticalalignment="top",
            fontsize=14,
        )
        axes[0].text(
            0.02,
            0.93,
            "UrbanPop NT/DT",
            color="blue",
            transform=axes[0].transAxes,
            verticalalignment="top",
            fontsize=14,
        )
    else:
        axes[0].legend()

    for ax in axes:
        ax.grid()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to '{args.output}'")
