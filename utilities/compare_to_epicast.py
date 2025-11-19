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
    symptomatic = df.groupby("day")["symptomatic"].sum().to_list()
    asymptomatic = df.groupby("day")["asymptomatic"].sum().to_list()
    presymptomatic = df.groupby("day")["presymptomatic"].sum().to_list()
    hospitalized = df.groupby("day")["ctx_hospitalized"].sum().to_list()
    icu = df.groupby("day")["ctx_icu"].sum().to_list()
    vent = df.groupby("day")["ctx_ventilated"].sum().to_list()
    treatment_recovered = df.groupby("day")["ctx_treatment_recovered"].sum().to_list()

    days = len(exposed)
    print(f"Epicast has {days} days")

    infected = [0] * days
    in_hospital = [0] * days
    infected[0] = exposed[0] - recovered[0]
    for i in range(1, days):
        infected[i] = infected[i - 1] + exposed[i] - recovered[i]
        in_hospital[i] = (
            in_hospital[i - 1]
            + hospitalized[i]
            + icu[i]
            + vent[i]
            - treatment_recovered[i]
            - dead[i]
        )

    converted_df = pd.DataFrame()
    converted_df["exposed"] = exposed
    converted_df["symptomatic"] = symptomatic
    converted_df["asymptomatic"] = asymptomatic
    converted_df["presymptomatic"] = presymptomatic
    converted_df["hospitalized"] = hospitalized
    converted_df["infected"] = infected
    converted_df["in_hospital"] = in_hospital
    converted_df["dead"] = dead
    converted_df.to_csv(fname + "-converted.csv")
    return converted_df


parser = argparse.ArgumentParser()
parser.add_argument("--epicast_file", "-e", required=True, help="Epicast csv file")
parser.add_argument("--exaepi_file", "-x", required=True, help="ExaEpi csv file")
parser.add_argument("--xlimit", "-l", type=int, default=250, help="X-axis limit for plotting")
args = parser.parse_args()


epicast_df = load_epicast(args.epicast_file)
exaepi_df = pd.read_csv(args.exaepi_file, sep="\\s+")
print(f"Read {len(exaepi_df)} lines from the ExaEpi file {args.exaepi_file}")

# print(exaepi_df.dtypes)
# exaepi_df["infected"] = exaepi_df[
#    ["PS/PI", "S/PI", "S/I", "PS/I", "A/PI", "A/I", "H/I"]
# ].sum(axis=1)
exaepi_df["in_hospital"] = exaepi_df[["H/NI", "H/I"]].sum(axis=1)
# exaepi_df["NewI"] = exaepi_df["NewI"] * 2.0

days = len(exaepi_df)
delta_dead = [0] * days
for i in range(1, days):
    delta_dead[i] = exaepi_df.loc[i, "D"] - exaepi_df.loc[i - 1, "D"]  # type: ignore
exaepi_df["delta_dead"] = delta_dead

exaepi_df.to_csv("exaepi.csv")

maxy = max(max(epicast_df.exposed), max(exaepi_df.NewI))
# fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(12, 11))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11))
# Plot infected cases
# ax1.plot(epicast_df.infected, label="Epicast", color="blue", linewidth=2)
# ax1.plot(exaepi_df.infected, label="ExaEpi", color="red", linewidth=2)
ax1.plot(epicast_df.exposed, label="Epicast", color="blue", linewidth=2)
ax1.plot(epicast_df.exposed, label="Epicast", color="blue", linewidth=2)
ax1.plot(epicast_df.exposed, label="Epicast", color="blue", linewidth=2)
ax1.plot(exaepi_df.NewI, label="ExaEpi", color="red", linewidth=2)
ax1.set_xlabel("Days")
ax1.set_ylabel("Number of Newly Infected")
ax1.set_xlim([0, args.xlimit])
ax1.set_title("Newly Infected Cases")
ax1.legend()
ax1.grid(True, which="major")
ax1.grid(True, which="minor", alpha=0.3)
ax1.set_ylim([0, maxy])
ax1.minorticks_on()

# Plot hospitalized
# ax2.plot(epicast_df.in_hospital, label="Epicast", color="blue", linewidth=2)
# ax2.plot(exaepi_df.in_hospital, label="ExaEpi", color="red", linewidth=2)
ax2.plot(epicast_df.symptomatic, label="Epicast Symp", color="blue", linewidth=2)
# ax2.plot(epicast_df.hospitalized, label="Epicast", color="blue", linewidth=2)
# ax2.plot(exaepi_df.NewH, label="ExaEpi", color="red", linewidth=2)
ax2.plot(exaepi_df.NewS, label="ExaEpi", color="red", linewidth=2)
ax2.set_xlabel("Days")
# ax2.set_ylabel("Number of Newly Hospitalized")
ax2.set_ylabel("Number of Newly Symptomatic")
ax2.set_xlim([0, args.xlimit])
ax2.set_ylim([0, maxy])
# ax2.set_title("Comparison of Hospitalized")
ax2.set_title("Newly Symptomatic")
ax2.grid(True, which="major")
ax2.grid(True, which="minor", alpha=0.3)
ax2.minorticks_on()

# Plot daily deaths
ax3.plot(epicast_df.presymptomatic, label="Epicast Presymp", color="blue", linewidth=2)
# ax3.plot(epicast_df.dead, label="Epicast", color="blue", linewidth=2)
# ax3.plot(exaepi_df.delta_dead, label="ExaEpi", color="red", linewidth=2)
ax3.plot(exaepi_df.NewP, label="ExaEpi", color="red", linewidth=2)
ax3.set_xlabel("Days")
# ax3.set_ylabel("Number of Newly Dead")
ax3.set_ylabel("Number of Newly Presymptomatic")
ax3.set_xlim([0, args.xlimit])
ax3.set_ylim([0, maxy])
# ax3.set_title("Comparison of Deaths")
ax3.set_title("Newly Presymptomatic")
ax3.grid(True, which="major")
ax3.grid(True, which="minor", alpha=0.3)
ax3.minorticks_on()

# Plot daily deaths
ax4.plot(epicast_df.asymptomatic, label="Epicast Asymp", color="blue", linewidth=2)
# ax3.plot(epicast_df.dead, label="Epicast", color="blue", linewidth=2)
# ax3.plot(exaepi_df.delta_dead, label="ExaEpi", color="red", linewidth=2)
ax4.plot(exaepi_df.NewA, label="ExaEpi", color="red", linewidth=2)
ax4.set_xlabel("Days")
# ax3.set_ylabel("Number of Newly Dead")
ax4.set_ylabel("Number of Newly Asymptomatic")
ax4.set_xlim([0, args.xlimit])
ax4.set_ylim([0, maxy])
# ax3.set_title("Comparison of Deaths")
ax4.set_title("Newly Asymptomatic")
ax4.grid(True, which="major")
ax4.grid(True, which="minor", alpha=0.3)
ax4.minorticks_on()

plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig("exaepi_v_epicast_comparison.png", bbox_inches="tight")
plt.show()
