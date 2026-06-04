#!/usr/bin/env -S python -u

import sys
import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

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

    converted_df["cumulative_exposed"] = converted_df.exposed.cumsum()

    tot_exposed = converted_df.exposed.sum()
    tot_symp = float(converted_df.symptomatic.sum())
    tot_hosp = float(converted_df.hospitalized.sum())
    print(f"Epicast total infected/exposed {tot_exposed}")
    print(f"Epicast total symptomatic {tot_symp} {(tot_symp / tot_exposed):.2f}")
    print(f"Epicast total hospitalized {tot_hosp} {(tot_hosp / tot_symp):.2f}")

    # Add a "day" column (0-based) for clarity in the CSV
    converted_df.insert(0, "day", range(days))

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


def run_seir(beta, sigma, gamma, N, seed, days):
    """Run a standard SEIR model and return a DataFrame with daily new counts.

    Compartments:
        S  – Susceptible
        E  – Exposed (latent, not yet infectious)
        I  – Infectious
        R  – Removed (recovered + dead)

    Parameters
    ----------
    beta  : transmission rate (contacts * probability of transmission per day)
    sigma : rate of progression from E→I  (1/sigma = mean latent period)
    gamma : recovery rate (1/gamma = mean infectious period)
    N     : total population
    seed  : initial number of infectious individuals
    days  : number of days to simulate
    """

    def seir_odes(t, y):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    S0 = N - seed
    E0 = 0.0
    I0 = float(seed)
    R0 = 0.0
    y0 = [S0, E0, I0, R0]

    t_eval = np.arange(0, days + 1, 1, dtype=float)
    sol = solve_ivp(seir_odes, [0, days], y0, t_eval=t_eval, method="RK45", max_step=0.1)

    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]

    # Compute daily *new* exposures (flux S→E) as difference in S between days
    new_exposed = np.maximum(0, -np.diff(S))   # length = days
    new_recovered = np.maximum(0, np.diff(R))  # length = days

    df = pd.DataFrame()
    df["day"] = np.arange(days)
    # "exposed" in the plot corresponds to new infections per day
    df["exposed"] = new_exposed
    # SEIR has a single infectious compartment; map it to "symptomatic"
    df["symptomatic"] = new_exposed  # same as new_exposed (all exposed become symptomatic)
    df["presymptomatic"] = new_exposed  # not separately modelled – use new_exposed as proxy
    df["asymptomatic"] = np.zeros(days)
    df["hospitalized"] = np.zeros(days)
    df["dead"] = np.zeros(days)
    df["recovered"] = new_recovered
    df["cumulative_exposed"] = df["exposed"].cumsum()

    print(f"SEIR total infected/exposed: {new_exposed.sum():.0f}")
    print(f"SEIR R0 = {beta / gamma:.2f}")

    return df


def fit_seir(target_y, N, seed, days, fixed=None):
    """Fit SEIR parameters to a target new-exposures time series.

    Parameters
    ----------
    target_y : array-like, daily new exposures (length >= days)
    N        : total population (used as initial value; fixed if 'N' in fixed)
    seed     : initial guess for infectious seed count
    days     : number of days to simulate
    fixed    : set of parameter names to hold fixed, e.g. {'beta', 'gamma'}.
               Any parameter not in this set is free to be optimised.
               Valid names: 'beta', 'sigma', 'gamma', 'N', 'seed'.

    Returns
    -------
    (beta, sigma, gamma, seed, fitted_df)  – best-fit parameters and the resulting SEIR DataFrame
    """
    if fixed is None:
        fixed = set()

    target = np.array(target_y[:days], dtype=float)

    # Build list of free parameter names and their initial values / bounds
    all_params = [
        ("beta",  args.beta,       (1e-4, 5.0)),
        ("sigma", args.sigma,      (1e-4, 5.0)),
        ("gamma", args.gamma,      (1e-4, 5.0)),
        ("seed",  float(seed),     (1.0, float(N))),
    ]
    free_params  = [(name, val, bnd) for name, val, bnd in all_params if name not in fixed]
    fixed_values = {name: val for name, val, _   in all_params if name in fixed}
    # N is always fixed (population is not a dynamical parameter)
    fixed_values.setdefault("N", float(N))

    def _unpack(free_vals):
        """Reconstruct full parameter set from free values."""
        it = iter(free_vals)
        vals = {}
        for name, _, _ in all_params:
            vals[name] = next(it) if name not in fixed else fixed_values[name]
        vals["N"] = fixed_values["N"]
        return vals

    def objective(free_vals):
        p = _unpack(free_vals)
        if any(p[k] <= 0 for k in ("beta", "sigma", "gamma", "seed")):
            return 1e18
        df = run_seir(p["beta"], p["sigma"], p["gamma"], int(round(p["N"])),
                      int(round(p["seed"])), days)
        pred = df["exposed"].values
        weights = 1.0 + target / (target.max() + 1e-9)
        return float(np.sum(weights * (pred - target) ** 2))

    x0     = [val for _, val, _   in free_params]
    bounds = [bnd for _, _,   bnd in free_params]

    if x0:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        p = _unpack(result.x)
        converged = result.success
    else:
        # Everything fixed – just evaluate
        p = _unpack([])
        converged = True

    beta_fit  = p["beta"]
    sigma_fit = p["sigma"]
    gamma_fit = p["gamma"]
    seed_fit  = int(round(p["seed"]))
    N_fit     = int(round(p["N"]))

    fitted_df = run_seir(beta_fit, sigma_fit, gamma_fit, N_fit, seed_fit, days)
    fixed_str = f" [fixed: {', '.join(sorted(fixed))}]" if fixed else ""
    print(f"  Fit converged={converged}  β={beta_fit:.4f}  σ={sigma_fit:.4f}  γ={gamma_fit:.4f}  seed={seed_fit}  R0={beta_fit/gamma_fit:.2f}{fixed_str}")
    return beta_fit, sigma_fit, gamma_fit, seed_fit, fitted_df


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


def plot_series(ax, epicast_data, exaepi_data, label, seir_df=None, fit_results=None):
    """Plot time series data from multiple files.

    Args:
        epicast_data: dict mapping filenames to (dataframe, label, is_wildcard) tuples
        exaepi_data: dict mapping filenames to (dataframe, label, is_wildcard) tuples
        label: the data series to plot (e.g., 'exposed', 'symptomatic')
        seir_df: optional DataFrame from run_seir(); plotted only on the 'exposed' subplot
        fit_results: optional list of (series_label, color, beta, sigma, gamma, fitted_df)
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

    # Plot fitted SEIR curves first (under experimental lines) on exposed/cumulative subplots
    seir_col = "cumulative_exposed" if col_name == "cumulative_exposed" else "exposed"
    if fit_results and col_name in ("exposed", "cumulative_exposed"):
        for (series_lbl, color, beta_f, sigma_f, gamma_f, seed_f, fdf) in fit_results:
            fit_y = fdf[seir_col].values[: args.xlimit]
            ax.plot(
                np.arange(len(fit_y)),
                fit_y,
                color="green",
                linewidth=3,
                linestyle="-",
                zorder=1,
            )
            auc = np.sum(fit_y)
            fit_lbl = f"SEIR fit ({series_lbl})" if series_lbl else "SEIR fit"
            auc_lines.append((fit_lbl, auc, "green", False))

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
                zorder=2,
            )
        else:
            ax.plot(
                df[col_name],
                label=plot_label,
                color=color,
                linewidth=2,
                linestyle="--",
                zorder=2,
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
                zorder=2,
            )
        else:
            ax.plot(
                x_vals,
                y_vals,
                label=plot_label,
                color=color,
                linewidth=2,
                zorder=2,
            )

    # Plot manual SEIR curve on the exposed and cumulative_exposed subplots
    if seir_df is not None and col_name in ("exposed", "cumulative_exposed"):
        seir_y = seir_df[seir_col].values[: args.xlimit]
        ax.plot(
            np.arange(len(seir_y)),
            seir_y,
            label=f"SEIR (β={args.beta}, σ={args.sigma}, γ={args.gamma})",
            color="green",
            linewidth=2,
            linestyle="-.",
        )
        auc = np.sum(seir_y)
        auc_lines.append((f"SEIR", auc, "green", False))

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    # Calculate ylim from all series
    max_vals = [df[col_name][: args.xlimit].max() for df, _, _wc in epicast_data.values()]
    max_vals.extend([df[exaepi_col][: args.xlimit].max() for df, _, _wc in exaepi_data.values()])
    if seir_df is not None and col_name in ("exposed", "cumulative_exposed"):
        max_vals.append(seir_df[seir_col].values[: args.xlimit].max())
    if fit_results and col_name in ("exposed", "cumulative_exposed"):
        for (_, _c, _b, _s, _g, _sd, fdf) in fit_results:
            max_vals.append(fdf[seir_col].values[: args.xlimit].max())
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
        # Add parameter label for the manual SEIR curve
        if seir_df is not None and col_name in ("exposed", "cumulative_exposed"):
            ax.text(
                0.98,
                0.97 - row * 0.10,
                f"β={args.beta}\nσ={args.sigma}, γ={args.gamma}\nN={args.N:,}, seed={args.seed}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="green",
            )
            row += 3  # three lines of text
        # Add parameter labels for each fitted SEIR curve
        if fit_results and col_name in ("exposed", "cumulative_exposed"):
            for (series_lbl, color, beta_f, sigma_f, gamma_f, seed_f, fdf) in fit_results:
                ax.text(
                    0.98,
                    0.97 - row * 0.10,
                    f"fit ({series_lbl})\nβ={beta_f:.3f}, σ={sigma_f:.3f}, γ={gamma_f:.3f}\nR0={beta_f/gamma_f:.2f}, seed={seed_f:,}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color='green',
                )
                row += 3


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
parser.add_argument(
    "--seir",
    action="store_true",
    default=False,
    help="Overlay an SEIR model curve using the specified parameters",
)
parser.add_argument(
    "--fit",
    action="store_true",
    default=False,
    help="Fit an SEIR model to each experimental series and overlay the fitted curves",
)
parser.add_argument(
    "--beta", type=float, default=0.44, help="SEIR transmission rate (default: 0.44)"
)
parser.add_argument(
    "--sigma", type=float, default=0.3, help="SEIR E→I progression rate (default: 0.3)"
)
parser.add_argument(
    "--gamma", type=float, default=0.173, help="SEIR I→R recovery rate (default: 0.17)"
)
parser.add_argument(
    "--N", type=int, default=1_800_000, help="SEIR total population (default: 1800000)"
)
parser.add_argument(
    "--seed", type=int, default=1000, help="SEIR initial infectious count (default: 1000)"
)
args = parser.parse_args()

# Track which SEIR parameters were explicitly provided on the command line.
# Parameters not provided are free to be fitted; provided ones are fixed.
_seir_params = {"beta", "sigma", "gamma", "N", "seed"}
_argv_flags = set()
for _tok in sys.argv[1:]:
    if _tok.startswith("--"):
        _argv_flags.add(_tok.lstrip("-").split("=")[0])
fit_fixed = {p for p in _seir_params if p in _argv_flags}

if not args.epicast_file and not args.exaepi_file:
    parser.error("At least one -e/--epicast_file or -x/--exaepi_file must be specified.")

epicast_data = {}
for file_spec in args.epicast_file:
    for fname, label, is_wildcard in expand_file_spec(file_spec):
        df = load_epicast(fname)
        epicast_data[fname] = (df, label, is_wildcard)
        # Write per-day CSV alongside the source file
        csv_out = fname + "-plot_values.csv"
        df.to_csv(csv_out, index=False)
        print(f"Wrote plot values to {csv_out}")

exaepi_data = {}
for file_spec in args.exaepi_file:
    for fname, label, is_wildcard in expand_file_spec(file_spec):
        print(f"{fname}")
        df = load_exaepi(fname)
        exaepi_data[fname] = (df, label, is_wildcard)
        # Write per-day CSV in the same format as the Epicast output
        converted_exaepi_df = pd.DataFrame()
        converted_exaepi_df["day"] = df["Day"] + args.shift
        converted_exaepi_df["exposed"] = df["NewI"].values
        converted_exaepi_df["symptomatic"] = df["NewS"].values
        converted_exaepi_df["asymptomatic"] = df["NewA"].values
        converted_exaepi_df["presymptomatic"] = df["NewP"].values
        converted_exaepi_df["hospitalized"] = df["NewH"].values
        converted_exaepi_df["dead"] = df["delta_dead"].values
        converted_exaepi_df["recovered"] = df["delta_recovered"].values
        converted_exaepi_df["cumulative_exposed"] = df["cum_exposed"].values
        csv_out = fname + "-plot_values.csv"
        converted_exaepi_df.to_csv(csv_out, index=False)
        print(f"Wrote plot values to {csv_out}")


seir_df = None
if args.seir:
    seir_df = run_seir(args.beta, args.sigma, args.gamma, args.N, args.seed, args.xlimit)

# fit_results: list of (label, color, beta, sigma, gamma, fitted_df)
fit_results = []
if args.fit:
    epicast_colors = ["blue", "green", "purple", "orange", "brown", "pink"]
    exaepi_colors = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]
    for i, (fname, (df, legend_label, is_wildcard)) in enumerate(epicast_data.items()):
        target_y = df["exposed"].values
        lbl = legend_label or f"Epicast {i+1}"
        color = "blue" if is_wildcard else epicast_colors[i % len(epicast_colors)]
        print(f"Fitting SEIR to {lbl} ...")
        b, s, g, sd, fdf = fit_seir(target_y, args.N, args.seed, args.xlimit, fixed=fit_fixed)
        fit_results.append((lbl, color, b, s, g, sd, fdf))
    for i, (fname, (df, legend_label, is_wildcard)) in enumerate(exaepi_data.items()):
        target_y = df["NewI"].values
        lbl = legend_label or f"ExaEpi {i+1}"
        color = "red" if is_wildcard else exaepi_colors[i % len(exaepi_colors)]
        print(f"Fitting SEIR to {lbl} ...")
        b, s, g, sd, fdf = fit_seir(target_y, args.N, args.seed, args.xlimit, fixed=fit_fixed)
        fit_results.append((lbl, color, b, s, g, sd, fdf))

if args.seir or fit_results:
    fig, (ax1, ax8) = plt.subplots(1, 2, figsize=(14, 5))
    plot_series(ax1, epicast_data, exaepi_data, "Exposed", seir_df=seir_df, fit_results=fit_results)
    plot_series(ax8, epicast_data, exaepi_data, "Cumulative Exposed", seir_df=seir_df, fit_results=fit_results)
else:
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
