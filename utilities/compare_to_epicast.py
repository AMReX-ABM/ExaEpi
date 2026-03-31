#!/usr/bin/env -S python -u

import sys
import os
import pandas as pd
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

    converted_df.to_csv(fname + "-converted.csv")

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
        transformed_df.to_csv("adjusted-" + fname, index=False, sep=" ")

    return df


def parse_file_with_label(file_spec):
    """Parse a file specification like 'path/to/file.csv:MyLabel'

    Returns:
        tuple: (filename, label) where label is None if not specified
    """
    if ":" in file_spec:
        parts = file_spec.split(":", 1)
        return parts[0], parts[1]
    else:
        # Use short filename without extension as default label
        short_name = file_spec.split("/")[-1].split(".")[0]
        return file_spec, short_name


def plot_series(ax, epicast_data, exaepi_data, label):
    """Plot time series data from multiple files.

    Args:
        epicast_data: dict mapping filenames to (dataframe, label) tuples
        exaepi_data: dict mapping filenames to (dataframe, label) tuples
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

    # Plot each Epicast file
    for i, (fname, (df, file_label)) in enumerate(epicast_data.items()):
        ax.plot(
            df[col_name],
            label=f"{file_label}",
            color=epicast_colors[i % len(epicast_colors)],
            linewidth=2,
            linestyle="--",
        )

    # Plot each ExaEpi file
    for i, (fname, (df, file_label)) in enumerate(exaepi_data.items()):
        # x_vals = range(len(df[exaepi_col]))
        x_vals = df["Day"]
        y_vals = df[exaepi_col]
        # x_vals = range(len(df[exaepi_col]))[4:]
        # y_vals = df[exaepi_col] * 1.33
        # y_vals = y_vals[:-4]
        ax.plot(
            x_vals,
            y_vals,
            label=f"{file_label}",
            color=exaepi_colors[i % len(exaepi_colors)],
            linewidth=2,
        )

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    # Calculate ylim from all series
    max_vals = [df[col_name][: args.xlimit].max() for df, _ in epicast_data.values()]
    max_vals.extend([df[exaepi_col][: args.xlimit].max() for df, _ in exaepi_data.values()])
    ax.set_ylim([0, 1.1 * max(max_vals)])

    ax.set_title(label)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()


parser = argparse.ArgumentParser(
    description="Compare ExaEpi and Epicast simulation outputs",
    epilog="File specifications can include optional labels using the format: 'filename.csv:Label'",
)
parser.add_argument(
    "--epicast_file",
    "-e",
    nargs="+",
    required=True,
    help="One or more Epicast csv files, optionally with labels (e.g., 'file.csv:MyLabel')",
)
parser.add_argument(
    "--exaepi_file",
    "-x",
    nargs="+",
    required=True,
    help="One or more ExaEpi csv files, optionally with labels (e.g., 'file.csv:MyLabel')",
)
parser.add_argument(
    "--xlimit", "-l", type=int, default=250, help="X-axis limit for plotting (default: 250)"
)
args = parser.parse_args()

epicast_data = {}
for file_spec in args.epicast_file:
    fname, label = parse_file_with_label(file_spec)
    df = load_epicast(fname)
    epicast_data[fname] = (df, label)

exaepi_data = {}
for file_spec in args.exaepi_file:
    fname, label = parse_file_with_label(file_spec)
    print(f"{fname}")
    df = load_exaepi(fname)

    exaepi_data[fname] = (df, label)


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 11))

plot_series(ax1, epicast_data, exaepi_data, "Exposed")
plot_series(ax2, epicast_data, exaepi_data, "Symptomatic")
plot_series(ax3, epicast_data, exaepi_data, "Presymptomatic")
plot_series(ax4, epicast_data, exaepi_data, "Asymptomatic")
plot_series(ax5, epicast_data, exaepi_data, "Hospitalized")
plot_series(ax6, epicast_data, exaepi_data, "Dead")
plot_series(ax7, epicast_data, exaepi_data, "Recovered")
plot_series(ax8, epicast_data, exaepi_data, "Cumulative Exposed")
ax8.legend(loc="lower right")

plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig("exaepi_v_epicast_comparison.png", bbox_inches="tight")
plt.show()
