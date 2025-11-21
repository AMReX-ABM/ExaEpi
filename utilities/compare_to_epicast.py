#!/usr/bin/env -S python -u

import sys
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def load_epicast(fname):
    df = pd.read_csv(fname)
    print(f"Read {len(df)} lines from the Epicast file {fname}")
    # print("Epicast columns:")
    # print(df.dtypes)

    exposed = df.groupby("day")["exposed"].sum().to_list()
    recovered = df.groupby("day")["recovered"].sum().to_list()
    dead = df.groupby("day")["ctx_removed"].sum().to_list()
    symptomatic = df.groupby("day")["ctx_symptomatic"].sum().to_list()
    asymptomatic = df.groupby("day")["ctx_asymptomatic"].sum().to_list()
    presymptomatic = df.groupby("day")["ctx_presymptomatic"].sum().to_list()
    icu = df.groupby("day")["ctx_icu"].sum()
    vent = df.groupby("day")["ctx_ventilated"].sum()
    hospitalized = df.groupby("day")["ctx_hospitalized"].sum() + icu + vent
    hospitalized = hospitalized.to_list()
    # treatment_recovered = df.groupby("day")["ctx_treatment_recovered"].sum().to_list()

    days = len(exposed)
    print(f"Epicast has {days} days")

    converted_df = pd.DataFrame()
    converted_df["exposed"] = exposed
    converted_df["symptomatic"] = symptomatic
    converted_df["asymptomatic"] = asymptomatic
    converted_df["presymptomatic"] = presymptomatic
    converted_df["hospitalized"] = hospitalized
    converted_df["dead"] = dead
    converted_df.to_csv(fname + "-converted.csv")

    return converted_df


def plot_series(ax, epicast_dfs, exaepi_vals, label):
    # Define colors for multiple Epicast series
    colors = ["blue", "green", "purple", "orange", "brown", "pink"]

    # Extract column name from label
    col_map = {
        "Newly Infected": "exposed",
        "Newly Symptomatic": "symptomatic",
        "Newly Presymptomatic": "presymptomatic",
        "Newly Asymptomatic": "asymptomatic",
        "Newly Hospitalized": "hospitalized",
        "Newly Dead": "dead",
    }
    col_name = col_map.get(label, label.lower())

    # Plot each Epicast file
    for i, (fname, df) in enumerate(epicast_dfs.items()):
        # Get short filename for legend
        short_name = fname.split("/")[-1].split(".")[0]
        ax.plot(
            df[col_name],
            label=f"Epicast-{short_name}",
            color=colors[i % len(colors)],
            linewidth=2,
            linestyle="--",
        )

    # Plot ExaEpi
    ax.plot(exaepi_vals, label="ExaEpi", color="red", linewidth=2)
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    # Calculate ylim from all series
    max_vals = [df[col_name].max() for df in epicast_dfs.values()]
    max_vals.append(exaepi_vals.max())
    ax.set_ylim([0, max(max_vals)])

    ax.set_title(label)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--epicast_file", "-e", nargs="+", required=True, help="One or more Epicast csv files"
)
parser.add_argument("--exaepi_file", "-x", required=True, help="ExaEpi csv file")
parser.add_argument("--xlimit", "-l", type=int, default=250, help="X-axis limit for plotting")
args = parser.parse_args()

epicast_dfs = {}
for fname in args.epicast_file:
    epicast_dfs[fname] = load_epicast(fname)

exaepi_df = pd.read_csv(args.exaepi_file, sep="\\s+")
print(f"Read {len(exaepi_df)} lines from the ExaEpi file {args.exaepi_file}")

exaepi_df["in_hospital"] = exaepi_df[["H/NI", "H/I"]].sum(axis=1)

days = len(exaepi_df)
delta_dead = [0] * days
for i in range(1, days):
    delta_dead[i] = exaepi_df.loc[i, "D"] - exaepi_df.loc[i - 1, "D"]  # type: ignore
exaepi_df["delta_dead"] = delta_dead

exaepi_df.to_csv("exaepi.csv")

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 11))

plot_series(ax1, epicast_dfs, exaepi_df.NewI, "Newly Infected")
plot_series(ax2, epicast_dfs, exaepi_df.NewS, "Newly Symptomatic")
plot_series(ax3, epicast_dfs, exaepi_df.NewP, "Newly Presymptomatic")
plot_series(ax4, epicast_dfs, exaepi_df.NewA, "Newly Asymptomatic")
plot_series(ax5, epicast_dfs, exaepi_df.NewH, "Newly Hospitalized")
plot_series(ax6, epicast_dfs, exaepi_df.delta_dead, "Newly Dead")
ax1.legend()

plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig("exaepi_v_epicast_comparison.png", bbox_inches="tight")
plt.show()
