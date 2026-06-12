#!/usr/bin/env -S python -u

import sys
import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from read_epicast_events import read_events_bin, aggregate_events


def load_epicast(fname):
    print(f"Reading binary Epicast file {fname} ...")
    events_df, _ = read_events_bin(fname)
    print(f"Read {len(events_df):,} events from {fname}")

    agg_df = aggregate_events(events_df)
    print(f"Aggregated into {len(agg_df)} timesteps")

    # aggregate_events groups by timestep; each row is one timestep.
    # disease_state columns: exposed, recovered, symptomatic, asymptomatic, presymptomatic
    # context columns: ctx_removed, ctx_symptomatic, ctx_asymptomatic, ctx_presymptomatic,
    #                  ctx_icu, ctx_ventilated, ctx_hospitalized, ...

    def _col(name):
        return agg_df[name] if name in agg_df.columns else pd.Series(0, index=agg_df.index)

    converted_df = pd.DataFrame()
    converted_df["exposed"] = _col("exposed").values
    converted_df["symptomatic"] = _col("ctx_symptomatic").values
    converted_df["asymptomatic"] = _col("ctx_asymptomatic").values
    converted_df["presymptomatic"] = _col("ctx_presymptomatic").values
    converted_df["hospitalized"] = (
        _col("ctx_hospitalized") + _col("ctx_icu") + _col("ctx_ventilated")
    ).values
    converted_df["dead"] = _col("ctx_removed").values
    converted_df["recovered"] = _col("recovered").values

    days = len(converted_df)
    print(f"Epicast has {days} days")

    # converted_df.to_csv(fname + "-converted.csv")

    converted_df["cumulative_exposed"] = converted_df.exposed.cumsum()

    tot_exposed = converted_df.exposed.sum()
    tot_symp = float(converted_df.symptomatic.sum())
    tot_hosp = float(converted_df.hospitalized.sum())
    print(f"Epicast total infected/exposed {tot_exposed}")
    print(f"Epicast total symptomatic {tot_symp} {(tot_symp / tot_exposed):.2f}")
    print(f"Epicast total hospitalized {tot_hosp} {(tot_hosp / tot_symp):.2f}")

    return converted_df


def load_exaepi(fname):
    df = pd.read_csv(fname, sep="\\s+")
    print(f"Read {len(df)} lines from the ExaEpi file {fname}")

    df["in_hospital"] = df[["H/NI", "H/I"]].sum(axis=1)

    days = len(df)
    delta_dead = [0] * days
    delta_recovered = [0] * days
    for i in range(1, days):
        delta_dead[i] = df.loc[i, "D"] - df.loc[i - 1, "D"]  # type: ignore
        delta_recovered[i] = df.loc[i, "R"] - df.loc[i - 1, "R"]  # type: ignore
    df["delta_dead"] = delta_dead
    df["delta_recovered"] = delta_recovered
    df["cum_exposed"] = df.NewI.cumsum()

    print(f"ExaEpi total infected/exposed {df.NewI.sum()}")

    print(f"ExaEpi hospitalized by age:")
    ages = ["U5", "5to17", "18to29", "30to49", "50to64", "O64"]
    for i in range(len(ages)):
        num_symp = float(df["Symp" + ages[i]].sum())
        num_hosp = float(df["Hosp" + ages[i]].sum())
        frac_hosp = num_hosp / num_symp
        print(f"  {ages[i]:8s}   {num_hosp:8.0f} {frac_hosp:.3f}")

    tot_symp = float(df.NewS.sum())
    tot_hosp = float(df.NewH.sum())
    print(f"ExaEpi total symptomatic {tot_symp} {(tot_symp / df.NewI.sum()):.2f}")
    print(f"ExaEpi total hospitalized {tot_hosp} {(tot_hosp / tot_symp):.2f}")

    if not fname.startswith("adjusted"):
        transformed_df = df.copy()
        transformed_df["Day"] += 4
        for col in transformed_df.columns:
            if col != "Day":
                transformed_df[col] *= 1
        # transformed_df.to_csv("adjusted-" + fname, index=False, sep=" ")

    return df


def parse_file_with_label(file_spec):
    """Parse a file specification like 'path/to/file.csv:MyLabel'

    Returns:
        tuple: (pattern, explicit_label_or_None)
            explicit_label_or_None is None when no ':label' was given.
    """
    if ":" in file_spec:
        parts = file_spec.split(":", 1)
        return parts[0], parts[1]
    else:
        return file_spec, None


def expand_file_spec(file_spec):
    """Expand a file specification (possibly containing wildcards) into a list of
    (filename, legend_label, is_wildcard) tuples.

    legend_label is the string to show in the legend, or None if no legend entry
    should be created for this file.

    Rules:
    - No ':label' suffix → legend_label is None for every matched file (no legend entry).
    - ':label' suffix, single file (or no wildcard) → legend_label = label.
    - ':label' suffix, wildcard matching N>1 files → first file gets legend_label = label,
      the rest get legend_label = None (label shown only once).
    - Wildcard matching multiple files → is_wildcard=True (faint lines).
    - Single file (explicit or single wildcard match) → is_wildcard=False.

    Returns:
        list of (filename, legend_label, is_wildcard)
    """
    pattern, explicit_label = parse_file_with_label(file_spec)
    has_wildcard = any(c in pattern for c in ("*", "?", "["))

    if has_wildcard:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"Warning: no files matched pattern '{pattern}'", file=sys.stderr)
            return []
        if len(matched) == 1:
            # Single match – treat as explicit (not faint)
            return [(matched[0], explicit_label, False)]
        # Multiple matches – faint lines; label shown at most once
        results = []
        for idx, fpath in enumerate(matched):
            legend_label = explicit_label if (idx == 0 and explicit_label is not None) else None
            results.append((fpath, legend_label, True))
        return results
    else:
        return [(pattern, explicit_label, False)]


def plot_series(ax, epicast_data, exaepi_data, label):
    """Plot time series data from multiple files.

    Args:
        epicast_data: dict mapping filenames to (dataframe, label, is_wildcard) tuples
        exaepi_data: dict mapping filenames to (dataframe, label, is_wildcard) tuples
        label: the data series to plot (e.g., 'exposed', 'symptomatic')
    """
    # Mapping from plot labels to ExaEpi column names
    col_mapping = {
        "exposed": "NewI",
        "symptomatic": "NewS",
        "presymptomatic": "NewP",
        "asymptomatic": "NewA",
        "hospitalized": "NewH",
        "dead": "delta_dead",
        "recovered": "delta_recovered",
        "cumulative_exposed": "cum_exposed",
    }
    # Define colors for multiple series
    epicast_colors = ["blue", "green", "purple", "orange", "brown", "pink"]
    exaepi_colors = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]

    col_name = label.lower().replace(" ", "_")
    exaepi_col = col_mapping.get(col_name, col_name)

    auc_lines = []

    # Plot each Epicast file
    for i, (fname, (df, legend_label, is_wildcard)) in enumerate(epicast_data.items()):
        # Wildcard series all use the same base blue; explicit series cycle through colors
        color = "blue" if is_wildcard else epicast_colors[i % len(epicast_colors)]
        y_vals = df[col_name][: args.xlimit]
        auc = np.sum(y_vals)
        auc_lines.append((legend_label, auc, color, is_wildcard))

        # Use the explicit label if provided; otherwise suppress from legend
        plot_label = legend_label if legend_label is not None else "_nolegend_"

        if is_wildcard:
            ax.plot(
                df[col_name],
                label=plot_label,
                color=color,
                linewidth=1,
                linestyle="-",
                alpha=0.3,
            )
        else:
            ax.plot(
                df[col_name],
                label=plot_label,
                color=color,
                linewidth=2,
                linestyle="--",
            )

    # Plot each ExaEpi file
    for i, (fname, (df, legend_label, is_wildcard)) in enumerate(exaepi_data.items()):
        # Wildcard series all use the same base red; explicit series cycle through colors
        color = "red" if is_wildcard else exaepi_colors[i % len(exaepi_colors)]
        x_vals = df["Day"] + args.shift
        y_vals = df[exaepi_col]
        auc = np.sum(y_vals)
        auc_lines.append((legend_label, auc, color, is_wildcard))

        # Use the explicit label if provided; otherwise suppress from legend
        plot_label = legend_label if legend_label is not None else "_nolegend_"

        if is_wildcard:
            ax.plot(
                x_vals,
                y_vals,
                label=plot_label,
                color=color,
                linewidth=1,
                linestyle="-",
                alpha=0.3,
            )
        else:
            ax.plot(
                x_vals,
                y_vals,
                label=plot_label,
                color=color,
                linewidth=2,
            )

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    # Calculate ylim from all series
    max_vals = [df[col_name][: args.xlimit].max() for df, _, _wc in epicast_data.values()]
    max_vals.extend([df[exaepi_col][: args.xlimit].max() for df, _, _wc in exaepi_data.values()])
    if max_vals:
        ylim_top = 1.1 * max(max_vals)
        ax.set_ylim([0, ylim_top])

    ax.set_title(label)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    # Annotate AUC (or max value for cumulative) for each series in the upper-right corner
    # Only annotate series that have an explicit label (legend_label is not None).
    if col_name == "cumulative_exposed":
        # For cumulative plot: show max value per labelled series
        row = 0
        for i, (fname, (df, legend_label, is_wildcard)) in enumerate(epicast_data.items()):
            if legend_label is None:
                continue
            color = epicast_colors[i % len(epicast_colors)]
            max_val = df[col_name][: args.xlimit].max()
            ax.text(
                0.98,
                0.97 - row * 0.10,
                f"Max {legend_label}: {max_val:,.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=color,
            )
            row += 1
        for i, (fname, (df, legend_label, is_wildcard)) in enumerate(exaepi_data.items()):
            if legend_label is None:
                continue
            color = exaepi_colors[i % len(exaepi_colors)]
            max_val = df[exaepi_col][: args.xlimit].max()
            ax.text(
                0.98,
                0.97 - row * 0.10,
                f"Max {legend_label}: {max_val:,.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=color,
            )
            row += 1
    else:
        print(f"{col_name}")
        row = 0
        for lbl, auc, color, is_wildcard in auc_lines:
            if lbl is None:
                print(f"  (unlabelled): {auc:.0f}")
                continue
            ax.text(
                0.98,
                0.97 - row * 0.10,
                f"AUC {lbl}: {auc:,.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=color,
            )
            print(f"  {lbl}: {auc:.0f}")
            row += 1


parser = argparse.ArgumentParser(
    description="Compare ExaEpi and Epicast simulation outputs",
    epilog=(
        "File specifications can include optional labels using the format: 'filename.csv:Label'. "
        "Both -e and -x can be repeated multiple times to plot multiple series, "
        "e.g.: -e file1.bin -e file2.bin:Label2 -x run1.csv -x run2.csv:Label2"
    ),
)
parser.add_argument(
    "--epicast_file",
    "-e",
    action="append",
    default=[],
    metavar="FILE[:LABEL]",
    help="Epicast binary file, optionally with a label (e.g., 'file.bin:MyLabel'). Can be repeated.",
)
parser.add_argument(
    "--exaepi_file",
    "-x",
    action="append",
    default=[],
    metavar="FILE[:LABEL]",
    help="ExaEpi csv file, optionally with a label (e.g., 'file.csv:MyLabel'). Can be repeated.",
)
parser.add_argument(
    "--xlimit", "-l", type=int, default=250, help="X-axis limit for plotting (default: 250)"
)
parser.add_argument(
    "--shift",
    "-s",
    type=int,
    default=0,
    help="Shift the ExaEpi curve along the x-axis in days (positive = right, negative = left, default: 0)",
)
parser.add_argument(
    "--output", "-o", required=True, help="Output file name for the plot (e.g., comparison.png)"
)
args = parser.parse_args()

if not args.epicast_file and not args.exaepi_file:
    parser.error("At least one -e/--epicast_file or -x/--exaepi_file must be specified.")

epicast_data = {}
for file_spec in args.epicast_file:
    for fname, label, is_wildcard in expand_file_spec(file_spec):
        df = load_epicast(fname)
        epicast_data[fname] = (df, label, is_wildcard)

exaepi_data = {}
for file_spec in args.exaepi_file:
    for fname, label, is_wildcard in expand_file_spec(file_spec):
        print(f"{fname}")
        df = load_exaepi(fname)
        exaepi_data[fname] = (df, label, is_wildcard)


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 11))

plot_series(ax1, epicast_data, exaepi_data, "Exposed")
plot_series(ax2, epicast_data, exaepi_data, "Symptomatic")
plot_series(ax3, epicast_data, exaepi_data, "Presymptomatic")
plot_series(ax4, epicast_data, exaepi_data, "Asymptomatic")
plot_series(ax5, epicast_data, exaepi_data, "Hospitalized")
plot_series(ax6, epicast_data, exaepi_data, "Dead")
plot_series(ax7, epicast_data, exaepi_data, "Recovered")
plot_series(ax8, epicast_data, exaepi_data, "Cumulative Exposed")

# plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight")
plt.show()
