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
        num_symp = float(df["Symp" + ages[i]].to_numpy().sum())
        num_hosp = float(df["Hosp" + ages[i]].to_numpy().sum())
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


def run_seir(beta, sigma, gamma, h, gamma_h, mu, N, seed, days):
    """Run a SEIRHD model and return a DataFrame with daily new counts.

    Compartments:
        S  – Susceptible
        E  – Exposed (latent, not yet infectious)
        I  – Infectious (community)
        H  – Hospitalised
        R  – Recovered
        D  – Dead (only from H)

    Flow:
        S → E  (rate β·I/N)
        E → I  (rate σ)
        I → H  (rate h  – hospitalisation)
        I → R  (rate γ  – direct community recovery)
        H → R  (rate γ_h – hospital recovery)
        H → D  (rate μ  – death only from hospital)

    Parameters
    ----------
    beta    : transmission rate (per day)
    sigma   : E→I progression rate  (1/sigma = mean latent period)
    gamma   : I→R direct recovery rate
    h       : I→H hospitalisation rate
    gamma_h : H→R hospital recovery rate
    mu      : H→D death rate  (HFR = mu / (gamma_h + mu))
    N       : total population
    seed    : initial number of infectious individuals
    days    : number of days to simulate
    """

    def seirhd_odes(t, y):
        S, E, I, H, R, D = y
        inf  = beta * S * I / N
        dS   = -inf
        dE   =  inf - sigma * E
        dI   =  sigma * E - (gamma + h) * I
        dH   =  h * I - (gamma_h + mu) * H
        dR   =  gamma * I + gamma_h * H
        dD   =  mu * H
        return [dS, dE, dI, dH, dR, dD]

    y0 = [float(N - seed), 0.0, float(seed), 0.0, 0.0, 0.0]

    t_eval = np.arange(0, days + 1, 1, dtype=float)
    sol = solve_ivp(seirhd_odes, [0, days], y0, t_eval=t_eval, method="RK45", max_step=0.1)

    S = sol.y[0]
    H = sol.y[3]
    R = sol.y[4]
    D = sol.y[5]

    new_exposed      = np.maximum(0, -np.diff(S))
    new_hospitalized = np.maximum(0,  np.diff(H + R + D)) - new_exposed  # flux into H
    # simpler: new hospitalized = h * I  integrated per day
    new_hospitalized = np.maximum(0, np.diff(H) + np.maximum(0, np.diff(R + D)))
    # most direct: use the H inflow = diff in cumulative (S drop - direct recoveries)
    # Just track H compartment entries via finite differences of D+R+H
    new_hospitalized = np.maximum(0, np.diff(sol.y[3] + sol.y[4] + sol.y[5])
                                   - np.maximum(0, -np.diff(sol.y[0]))
                                   + np.maximum(0, -np.diff(sol.y[0])))
    # Cleanest: new_hosp = h*I averaged per interval; use midpoint of sol
    I_mid = 0.5 * (sol.y[2][:-1] + sol.y[2][1:])
    new_hospitalized = np.maximum(0, h * I_mid)

    new_recovered = np.maximum(0, np.diff(R))
    new_dead      = np.maximum(0, np.diff(D))

    df = pd.DataFrame()
    df["day"]                = np.arange(days)
    df["exposed"]            = new_exposed
    df["symptomatic"]        = new_exposed
    df["presymptomatic"]     = new_exposed
    df["asymptomatic"]       = np.zeros(days)
    df["hospitalized"]       = new_hospitalized
    df["dead"]               = new_dead
    df["recovered"]          = new_recovered
    df["cumulative_exposed"] = new_exposed.cumsum()

    r0  = beta / (gamma + h)
    hfr = mu / (gamma_h + mu)
    ifr = (h / (gamma + h)) * hfr
    print(f"SEIRHD  exposed={new_exposed.sum():.0f}  hosp={new_hospitalized.sum():.0f}  dead={new_dead.sum():.0f}")
    print(f"SEIRHD  R0={r0:.2f}  hosp_rate={h/(gamma+h):.4f}  HFR={hfr:.4f}  IFR={ifr:.4f}")

    return df


def fit_seir(target_exposed, target_dead, target_hosp, N, seed, days, fixed=None):
    """Fit SEIRHD parameters to target time series.

    Parameters
    ----------
    target_exposed : array-like, daily new exposures  (length >= days)
    target_dead    : array-like, daily new deaths      (length >= days); may be all zeros
    target_hosp    : array-like, daily new hospitalized(length >= days); may be all zeros
    N              : total population (fixed)
    seed           : initial guess for infectious seed count
    days           : number of days to simulate
    fixed          : set of parameter names to hold fixed during optimisation.
                     Valid names: 'beta', 'sigma', 'gamma', 'hosp_rate', 'gamma_h', 'mu', 'seed'.

    Returns
    -------
    (beta, sigma, gamma, hosp_rate, gamma_h, mu, seed, fitted_df)
    """
    if fixed is None:
        fixed = set()

    exp_arr  = np.array(target_exposed[:days], dtype=float)
    dead_arr = np.array(target_dead[:days],    dtype=float)
    hosp_arr = np.array(target_hosp[:days],    dtype=float)

    exp_scale  = exp_arr.max()  + 1e-9
    dead_scale = dead_arr.max() + 1e-9
    hosp_scale = hosp_arr.max() + 1e-9
    has_deaths = dead_arr.max() > 0
    has_hosp   = hosp_arr.max() > 0

    all_params = [
        ("beta",      args.beta,       (1e-4, 5.0)),
        ("sigma",     args.sigma,      (1e-4, 5.0)),
        ("gamma",     args.gamma,      (1e-4, 5.0)),
        ("hosp_rate", args.hosp_rate,  (1e-6, 2.0)),
        ("gamma_h",   args.gamma_h,    (1e-4, 5.0)),
        ("mu",        args.mu,         (1e-6, 1.0)),
        ("seed",      float(seed),     (1.0, float(N))),
    ]
    free_params  = [(name, val, bnd) for name, val, bnd in all_params if name not in fixed]
    fixed_values = {name: val for name, val, _ in all_params if name in fixed}
    fixed_values.setdefault("N", float(N))

    def _unpack(free_vals):
        it = iter(free_vals)
        vals = {}
        for name, _, _ in all_params:
            vals[name] = next(it) if name not in fixed else fixed_values[name]
        vals["N"] = fixed_values["N"]
        return vals

    def objective(free_vals):
        p = _unpack(free_vals)
        if any(p[k] <= 0 for k in ("beta", "sigma", "gamma", "hosp_rate", "gamma_h", "mu", "seed")):
            return 1e18
        df = run_seir(p["beta"], p["sigma"], p["gamma"], p["hosp_rate"], p["gamma_h"], p["mu"],
                      int(round(p["N"])), int(round(p["seed"])), days)
        w_exp = 1.0 + exp_arr / exp_scale
        res = np.sum(w_exp * ((df["exposed"].values - exp_arr) / exp_scale) ** 2)
        if has_deaths:
            w_dead = 1.0 + dead_arr / dead_scale
            res += np.sum(w_dead * ((df["dead"].values - dead_arr) / dead_scale) ** 2)
        if has_hosp:
            w_hosp = 1.0 + hosp_arr / hosp_scale
            res += np.sum(w_hosp * ((df["hospitalized"].values - hosp_arr) / hosp_scale) ** 2)
        return float(res)

    x0     = [val for _, val, _   in free_params]
    bounds = [bnd for _, _,   bnd in free_params]

    if x0:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        p = _unpack(result.x)
        converged = result.success
    else:
        p = _unpack([])
        converged = True

    beta_fit      = p["beta"]
    sigma_fit     = p["sigma"]
    gamma_fit     = p["gamma"]
    hosp_rate_fit = p["hosp_rate"]
    gamma_h_fit   = p["gamma_h"]
    mu_fit        = p["mu"]
    seed_fit      = int(round(p["seed"]))
    N_fit         = int(round(p["N"]))

    fitted_df = run_seir(beta_fit, sigma_fit, gamma_fit, hosp_rate_fit, gamma_h_fit,
                         mu_fit, N_fit, seed_fit, days)
    r0  = beta_fit / (gamma_fit + hosp_rate_fit)
    hfr = mu_fit / (gamma_h_fit + mu_fit)
    ifr = (hosp_rate_fit / (gamma_fit + hosp_rate_fit)) * hfr
    fixed_str = f" [fixed: {', '.join(sorted(fixed))}]" if fixed else ""
    print(f"  Fit converged={converged}  β={beta_fit:.4f}  σ={sigma_fit:.4f}  "
          f"γ={gamma_fit:.4f}  h={hosp_rate_fit:.4f}  γ_h={gamma_h_fit:.4f}  μ={mu_fit:.5f}  "
          f"seed={seed_fit}  R0={r0:.2f}  HFR={hfr:.4f}  IFR={ifr:.4f}{fixed_str}")
    return beta_fit, sigma_fit, gamma_fit, hosp_rate_fit, gamma_h_fit, mu_fit, seed_fit, fitted_df


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
    - Wildcard matching multiple files → is_wildcard=True.
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
            return [(matched[0], explicit_label, False)]
        results = []
        for idx, fpath in enumerate(matched):
            legend_label = explicit_label if (idx == 0 and explicit_label is not None) else None
            results.append((fpath, legend_label, True))
        return results
    else:
        return [(pattern, explicit_label, False)]


def _align_arrays(dfs, col, xlimit):
    """Stack column values from a list of DataFrames, truncated to xlimit.

    Returns a 2-D array of shape (n_files, n_days).
    """
    max_len = min(min(len(df) for df in dfs), xlimit)
    return np.vstack([df[col].values[:max_len] for df in dfs])


def _auto_shift(epicast_data, exaepi_data, xlimit, shift_range=60):
    """Find the integer day-shift (same convention as --shift: positive delays the ExaEpi curve)
    that minimizes RMSE between the first Epicast group's 'exposed' curve and the first ExaEpi
    group's 'NewI' curve. Wildcard groups (multiple files matched by one -e/-x pattern) are
    averaged first, same as elsewhere in this script. Only the first -e/-x pair is used even if
    both were repeated -- shift is a single global x-axis offset applied to every ExaEpi curve.
    """
    if not epicast_data or not exaepi_data:
        return 0.0

    e_entry = epicast_data[0]
    x_entry = exaepi_data[0]

    if e_entry["is_wildcard"] and len(e_entry["dfs"]) > 1:
        e = _align_arrays(e_entry["dfs"], "exposed", xlimit).mean(axis=0)
    else:
        e = e_entry["dfs"][0]["exposed"].values[:xlimit]

    if x_entry["is_wildcard"] and len(x_entry["dfs"]) > 1:
        x = _align_arrays(x_entry["dfs"], "NewI", xlimit).mean(axis=0)
    else:
        x = x_entry["dfs"][0]["NewI"].values[:xlimit]

    n = min(len(e), xlimit)
    e = e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))
    x = np.asarray(x, dtype=float)

    best_shift, best_rmse = 0, float("inf")
    for s in range(-shift_range, shift_range + 1):
        d = np.arange(n)
        src = d - s
        x_shifted = np.where((src >= 0) & (src < len(x)), x[np.clip(src, 0, len(x) - 1)], 0.0)
        rmse = np.sqrt(np.mean((e - x_shifted) ** 2))
        if rmse < best_rmse:
            best_rmse, best_shift = rmse, s
    return float(best_shift)


_EPICAST_EXTENT_COLS = ["exposed", "symptomatic", "asymptomatic", "presymptomatic", "hospitalized", "dead", "recovered"]
_EXAEPI_EXTENT_COLS  = ["NewI", "NewS", "NewA", "NewP", "NewH", "delta_dead", "delta_recovered"]


def _furthest_nonzero_day(df, cols, threshold=10):
    """Last day index (0-based) at which any of the given columns is still >= threshold, i.e. this
    curve's day-numbering-native extent before it drops into stray-case noise. A >= threshold test
    (rather than != 0) keeps a single lingering case reported long after the real tail from forcing
    the whole plot to keep a long, mostly-empty trailing window. Returns the last row index if
    every column stays at/above threshold throughout (nothing to trim), or 0 if none ever reaches it.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return len(df) - 1 if len(df) else 0
    above = np.flatnonzero((df[cols].to_numpy() >= threshold).any(axis=1))
    return int(above[-1]) if len(above) else 0


def _auto_xlimit(epicast_data, exaepi_data, shift, margin=5):
    """Size the x-axis to the minimum, across every -e/-x input file, of that file's furthest
    above-threshold day (see _furthest_nonzero_day) -- i.e. trim to whichever curve runs out of
    signal first, so no plot is padded with a long trailing flat tail from a shorter-lived curve.
    ExaEpi files are measured in their OWN day-numbering then offset by `shift` (already resolved
    to a number by the time this runs), since that's where they actually land once plotted.
    """
    extents = []
    for entry in epicast_data:
        for df in entry["dfs"]:
            extents.append(_furthest_nonzero_day(df, _EPICAST_EXTENT_COLS))
    for entry in exaepi_data:
        for df in entry["dfs"]:
            extents.append(_furthest_nonzero_day(df, _EXAEPI_EXTENT_COLS) + shift)
    if not extents:
        return 250
    return max(1, int(np.ceil(min(extents))) + margin)


_CONTEXT_COLS = {
    "EWork":   ("Work",                "tab:blue"),
    "EHosp":   ("Hospital",            "tab:cyan"),
    "ESchool": ("School",              "tab:orange"),
    "ENbhD":   ("Neighborhood (day)",  "tab:green"),
    "ECommD":  ("Community (day)",     "tab:olive"),
    "EHH":     ("Household",           "tab:red"),
    "ENC":     ("NC cluster",          "tab:brown"),
    "ENbhN":   ("Neighborhood (night)","tab:purple"),
    "ECommN":  ("Community (night)",   "tab:pink"),
}


def plot_context(ax, exaepi_data):
    """Plot per-context expected infections from ExaEpi diagnostic columns."""
    ax.set_title("Expected infections by context (ExaEpi)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Expected new infections")
    ax.set_xlim([0, args.xlimit])
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    for entry in exaepi_data:
        for df in entry["dfs"]:
            x = (df["Day"] + args.shift).values[:args.xlimit]
            for col, (label_str, color) in _CONTEXT_COLS.items():
                if col in df.columns:
                    y = df[col].values[:args.xlimit]
                    ax.plot(x, y, label=label_str, color=color, linewidth=1)

    ax.legend(fontsize=7)


def plot_series(ax, epicast_data, exaepi_data, label, seir_df=None, fit_results=None):
    """Plot time series data from multiple files.

    Both epicast_data and exaepi_data are lists of group dicts:
        {'label': str|None, 'is_wildcard': bool, 'dfs': [df, ...], 'fnames': [str, ...]}
    A wildcard group with N>1 files is rendered as an average line with a
    semi-transparent band between the per-day min and max.

    Args:
        label: the data series to plot (e.g., 'exposed', 'symptomatic')
        seir_df: optional DataFrame from run_seir()
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
    epicast_colors = ["blue", "green", "purple", "orange", "brown", "pink"]
    exaepi_colors  = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]

    col_name   = label.lower().replace(" ", "_")
    exaepi_col = col_mapping.get(col_name, col_name)

    auc_lines = []

    _seird_cols = {"exposed", "cumulative_exposed", "hospitalized", "dead", "recovered"}
    seir_col = col_name if col_name in _seird_cols else None

    # Plot fitted SEIRHD curves first (under experimental lines)
    if fit_results and seir_col is not None:
        for (series_lbl, _, _, _, _, _, _, _, _, fdf) in fit_results:
            fit_y = fdf[seir_col].values[: args.xlimit]
            ax.plot(np.arange(len(fit_y)), fit_y, color="green", linewidth=3, linestyle="-", zorder=1)
            auc = np.sum(fit_y)
            fit_lbl = f"SEIRHD fit ({series_lbl})" if series_lbl else "SEIRHD fit"
            auc_lines.append((fit_lbl, auc, "green", False))

    def _plot_group(entry, i, base_colors, col, x_col=None, x_shift=0):
        """Plot one group entry; return (legend_label, auc, color)."""
        legend_label = entry["label"]
        is_wildcard  = entry["is_wildcard"]
        color        = base_colors[i % len(base_colors)]
        plot_label   = legend_label if legend_label is not None else "_nolegend_"

        if is_wildcard and len(entry["dfs"]) > 1:
            y_mat  = _align_arrays(entry["dfs"], col, args.xlimit)
            y_mean = y_mat.mean(axis=0)
            y_min  = y_mat.min(axis=0)
            y_max  = y_mat.max(axis=0)
            n      = y_mat.shape[1]
            x_vals = (entry["dfs"][0][x_col].values[:n] + x_shift) if x_col else np.arange(n)

            ax.fill_between(x_vals, y_min, y_max, alpha=0.25, color=color,
                            zorder=1, label="_nolegend_")
            ax.plot(x_vals, y_mean, label=plot_label, color=color, linewidth=1, zorder=2)
            auc = float(np.sum(y_mean))
        else:
            df     = entry["dfs"][0]
            x_vals = (df[x_col] + x_shift) if x_col else np.arange(len(df[col]))
            y_vals = df[col]
            auc    = float(np.sum(y_vals[: args.xlimit]))
            lsargs = {"linestyle": "--"} if base_colors[0] == "blue" else {}
            ax.plot(x_vals, y_vals, label=plot_label, color=color, linewidth=1, zorder=2, **lsargs)

        return legend_label, auc, color, is_wildcard

    # Plot each Epicast group
    for i, entry in enumerate(epicast_data):
        lbl, auc, color, is_wc = _plot_group(entry, i, epicast_colors, col_name)
        auc_lines.append((lbl, auc, color, is_wc))

    # Plot each ExaEpi group
    for i, entry in enumerate(exaepi_data):
        lbl, auc, color, is_wc = _plot_group(entry, i, exaepi_colors, exaepi_col,
                                              x_col="Day", x_shift=args.shift)
        auc_lines.append((lbl, auc, color, is_wc))

    # Plot manual SEIRD curve
    if seir_df is not None and seir_col is not None:
        seir_y = seir_df[seir_col].values[: args.xlimit]
        ax.plot(
            np.arange(len(seir_y)), seir_y,
            label=f"SEIRHD (β={args.beta}, h={args.hosp_rate}, μ={args.mu})",
            color="green", linewidth=2, linestyle="-.",
        )
        auc_lines.append(("SEIRHD", np.sum(seir_y), "green", False))

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    # ylim from all series (use max of the band ceiling for wildcard groups)
    max_vals = []
    for entry in epicast_data:
        if entry["is_wildcard"] and len(entry["dfs"]) > 1:
            max_vals.append(float(_align_arrays(entry["dfs"], col_name, args.xlimit).max()))
        else:
            max_vals.append(float(entry["dfs"][0][col_name][: args.xlimit].max()))
    for entry in exaepi_data:
        if entry["is_wildcard"] and len(entry["dfs"]) > 1:
            max_vals.append(float(_align_arrays(entry["dfs"], exaepi_col, args.xlimit).max()))
        else:
            max_vals.append(float(entry["dfs"][0][exaepi_col][: args.xlimit].max()))
    if seir_df is not None and seir_col is not None:
        max_vals.append(seir_df[seir_col].values[: args.xlimit].max())
    if fit_results and seir_col is not None:
        for (_, _c, _b, _s, _g, _h, _gh, _d, _sd, fdf) in fit_results:
            max_vals.append(fdf[seir_col].values[: args.xlimit].max())
    if args.ylimit is not None:
        ax.set_ylim([0, args.ylimit])
    elif max_vals:
        ax.set_ylim([0, 1.1 * max(max_vals)])

    ax.set_title(label)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    # Annotate per-series summary values in the upper-right corner
    print(f"{col_name}")
    if col_name == "cumulative_exposed":
        row = 0
        for i, entry in enumerate(epicast_data):
            legend_label = entry["label"]
            color = epicast_colors[i % len(epicast_colors)]
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                max_val = float(_align_arrays(entry["dfs"], col_name, args.xlimit).mean(axis=0).max())
            else:
                max_val = float(entry["dfs"][0][col_name][: args.xlimit].max())
            lbl_str = legend_label if legend_label is not None else "(unlabelled)"
            print(f"  Max {lbl_str}: {max_val:,.0f}")
            if legend_label is not None:
                ax.text(0.98, 0.97 - row * 0.10, f"Max {legend_label}: {max_val:,.0f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=color)
                row += 1
        for i, entry in enumerate(exaepi_data):
            legend_label = entry["label"]
            color = exaepi_colors[i % len(exaepi_colors)]
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                max_val = float(_align_arrays(entry["dfs"], exaepi_col, args.xlimit).mean(axis=0).max())
            else:
                max_val = float(entry["dfs"][0][exaepi_col][: args.xlimit].max())
            lbl_str = legend_label if legend_label is not None else "(unlabelled)"
            print(f"  Max {lbl_str}: {max_val:,.0f}")
            if legend_label is not None:
                ax.text(0.98, 0.97 - row * 0.10, f"Max {legend_label}: {max_val:,.0f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=color)
                row += 1
    else:
        row = 0
        for lbl, auc, color, is_wildcard in auc_lines:
            lbl_str = lbl if lbl is not None else "(unlabelled)"
            print(f"  AUC {lbl_str}: {auc:,.0f}")
            if lbl is not None:
                ax.text(0.98, 0.97 - row * 0.10, f"AUC {lbl}: {auc:,.0f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=color)
                row += 1


parser = argparse.ArgumentParser(
    description="Compare ExaEpi and Epicast simulation outputs",
    epilog=(
        "File specifications can include optional labels using the format: 'filename:Label'. "
        "Both -e and -x can be repeated and accept glob patterns, "
        "e.g.: -e 'runs/*.bin:Epicast' -x 'runs/*.csv:ExaEpi'. "
        "When a pattern matches multiple files, their average is plotted with a "
        "semi-transparent min/max band in the same color as the line."
    ),
)
parser.add_argument(
    "--epicast_file", "-e",
    action="append", default=[], metavar="FILE[:LABEL]",
    help="Epicast binary file or glob pattern, optionally with a label. Can be repeated.",
)
parser.add_argument(
    "--exaepi_file", "-x",
    action="append", default=[], metavar="FILE[:LABEL]",
    help=(
        "ExaEpi csv file or glob pattern, optionally with a label. Can be repeated. "
        "Multiple matched files are averaged with a min/max band."
    ),
)
def _xlimit_type(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return int(value)


parser.add_argument(
    "--xlimit", "-l", type=_xlimit_type, default=250,
    help="X-axis limit for plotting, or 'auto' to size it to the minimum, across all -e/-x "
         "inputs, of each input's furthest day with an exposed/symptomatic/asymptomatic/"
         "presymptomatic/hospitalized/dead/recovered value >= 10 (default: 250)",
)
parser.add_argument(
    "--ylimit", "-y", type=float, default=None, help="Y-axis maximum for all plots (default: auto)"
)
def _shift_type(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return float(value)


parser.add_argument(
    "--shift", "-s", type=_shift_type, default=0.0,
    help="Shift the ExaEpi curve along the x-axis in days, or 'auto' to pick the shift "
         "(searched over +/-60 days) that minimizes RMSE between the first -e and -x curves' "
         "'exposed'/NewI series (default: 0)",
)
parser.add_argument(
    "--output", "-o", required=True, help="Output file name for the plot (e.g., comparison.png)"
)
parser.add_argument(
    "--seir", action="store_true", default=False,
    help="Overlay an SEIR model curve using the specified parameters",
)
parser.add_argument(
    "--fit", action="store_true", default=False,
    help="Fit an SEIR model to each experimental series and overlay the fitted curves",
)
parser.add_argument("--beta",      type=float, default=0.48,       help="SEIR transmission rate (default: 0.44)")
parser.add_argument("--sigma",     type=float, default=0.263,        help="SEIR E→I progression rate (default: 0.3)")
parser.add_argument("--gamma",     type=float, default=0.152,      help="SEIR I→R recovery rate (default: 0.17)")
parser.add_argument("--hosp_rate", type=float, default=0.042,       help="SEIRHD I→H hospitalisation rate (default: 0.01)")
parser.add_argument("--gamma_h",   type=float, default=0.162,        help="SEIRHD H→R hospital recovery rate (default: 0.1)")
parser.add_argument("--mu",        type=float, default=0.017,      help="SEIRHD H→D death rate (default: 0.005)")
parser.add_argument("--N",         type=int,   default=2_100_000,  help="SEIRHD total population (default: 1800000)")
parser.add_argument("--seed",      type=int,   default=12000,       help="SEIRHD initial infectious count (default: 1000)")
parser.add_argument(
    "--plots", "-p",
    nargs="+", metavar="PLOT", default=None,
    help=(
        "Which plots to show, in the order given. Rendered in 2-column layout. "
        "Valid names (case-insensitive): Exposed, Symptomatic, Presymptomatic, "
        "Asymptomatic, Hospitalized, Dead, Recovered, 'Cumulative Exposed'. "
        "Default: all 8 (or Exposed/Recovered/Cumulative Exposed when --seir/--fit is used)."
    ),
)
args = parser.parse_args()

# Track which SEIR parameters were explicitly set so fitting can hold them fixed.
_seir_params = {"beta", "sigma", "gamma", "hosp_rate", "gamma_h", "mu", "N", "seed"}
_argv_flags = set()
for _tok in sys.argv[1:]:
    if _tok.startswith("--"):
        _argv_flags.add(_tok.lstrip("-").split("=")[0])
fit_fixed = {p for p in _seir_params if p in _argv_flags}

ALL_PLOTS = [
    "Exposed", "Symptomatic", "Presymptomatic", "Asymptomatic",
    "Hospitalized", "Dead", "Recovered", "Cumulative Exposed", "Context",
]
_plot_map = {p.lower(): p for p in ALL_PLOTS}

if args.plots is not None:
    resolved = []
    for name in args.plots:
        canonical = _plot_map.get(name.lower())
        if canonical is None:
            parser.error(f"Unknown plot '{name}'. Valid names: {', '.join(ALL_PLOTS)}")
        resolved.append(canonical)
    args.plots = resolved

if not args.epicast_file and not args.exaepi_file:
    parser.error("At least one -e/--epicast_file or -x/--exaepi_file must be specified.")


def _load_grouped(file_specs, load_fn, extra_csv_fn=None):
    """Expand each file spec into a group dict, loading DataFrames with load_fn.

    Returns a list of {'label', 'is_wildcard', 'dfs', 'fnames'} dicts.
    """
    groups = []
    for file_spec in file_specs:
        expanded = expand_file_spec(file_spec)
        if not expanded:
            continue
        group_label = expanded[0][1]
        is_wc = len(expanded) > 1
        entry = {"label": group_label, "is_wildcard": is_wc, "dfs": [], "fnames": []}
        for fname, _, _ in expanded:
            print(f"{fname}")
            df = load_fn(fname)
            entry["dfs"].append(df)
            entry["fnames"].append(fname)
            if extra_csv_fn:
                extra_csv_fn(fname, df)
        groups.append(entry)
    return groups


def _write_epicast_csv(fname, df):
    csv_out = fname + "-plot_values.csv"
    df.to_csv(csv_out, index=False)
    print(f"Wrote plot values to {csv_out}")


def _write_exaepi_csv(fname, df):
    out = pd.DataFrame({
        "day":               df["Day"] + args.shift,
        "exposed":           df["NewI"].values,
        "symptomatic":       df["NewS"].values,
        "asymptomatic":      df["NewA"].values,
        "presymptomatic":    df["NewP"].values,
        "hospitalized":      df["NewH"].values,
        "dead":              df["delta_dead"].values,
        "recovered":         df["delta_recovered"].values,
        "cumulative_exposed":df["cum_exposed"].values,
    })
    csv_out = fname + "-plot_values.csv"
    out.to_csv(csv_out, index=False)
    print(f"Wrote plot values to {csv_out}")


epicast_data = _load_grouped(args.epicast_file, load_epicast, _write_epicast_csv)
# CSV-writing is deferred until after args.shift is resolved below (it embeds args.shift into the
# "day" column, which isn't known yet if --shift auto was requested).
exaepi_data  = _load_grouped(args.exaepi_file,  load_exaepi,  None)

if args.shift == "auto":
    # args.xlimit may itself still be "auto" here -- it only bounds how much of the curve the
    # shift search looks at, so fall back to the parser's own default (250) rather than depending
    # on the xlimit auto-sizing below (which in turn depends on args.shift already being resolved).
    _shift_window = args.xlimit if isinstance(args.xlimit, (int, float)) else 250
    args.shift = _auto_shift(epicast_data, exaepi_data, _shift_window)
    print(f"Auto-detected shift: {args.shift:+.0f} days (minimizes RMSE of the 'exposed'/NewI curve)")

if args.xlimit == "auto":
    args.xlimit = _auto_xlimit(epicast_data, exaepi_data, args.shift)
    print(f"Auto-detected xlimit: {args.xlimit} days (minimum furthest >=10 extent across all inputs)")

for _entry in exaepi_data:
    for _fname, _df in zip(_entry["fnames"], _entry["dfs"]):
        _write_exaepi_csv(_fname, _df)

seir_df = None
if args.seir:
    seir_df = run_seir(args.beta, args.sigma, args.gamma, args.hosp_rate, args.gamma_h,
                       args.mu, args.N, args.seed, args.xlimit)

# fit_results: list of (label, color, beta, sigma, gamma, h, gamma_h, mu, seed, fitted_df)
fit_results = []
if args.fit:
    epicast_colors = ["blue", "green", "purple", "orange", "brown", "pink"]
    exaepi_colors  = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]

    for i, entry in enumerate(epicast_data):
        lbl   = entry["label"] or f"Epicast {i+1}"
        color = epicast_colors[i % len(epicast_colors)]
        print(f"Fitting SEIRHD to {lbl} ...")
        if entry["is_wildcard"] and len(entry["dfs"]) > 1:
            exp  = _align_arrays(entry["dfs"], "exposed",      args.xlimit).mean(axis=0)
            dead = _align_arrays(entry["dfs"], "dead",         args.xlimit).mean(axis=0)
            hosp = _align_arrays(entry["dfs"], "hospitalized", args.xlimit).mean(axis=0)
        else:
            df   = entry["dfs"][0]
            exp, dead, hosp = df["exposed"].values, df["dead"].values, df["hospitalized"].values
        b, s, g, h, gh, d, sd, fdf = fit_seir(exp, dead, hosp, args.N, args.seed,
                                               args.xlimit, fixed=fit_fixed)
        fit_results.append((lbl, color, b, s, g, h, gh, d, sd, fdf))

    for i, entry in enumerate(exaepi_data):
        lbl   = entry["label"] or f"ExaEpi {i+1}"
        color = exaepi_colors[i % len(exaepi_colors)]
        print(f"Fitting SEIRHD to {lbl} ...")
        if entry["is_wildcard"] and len(entry["dfs"]) > 1:
            new_i = _align_arrays(entry["dfs"], "NewI",       args.xlimit).mean(axis=0)
            ddead = _align_arrays(entry["dfs"], "delta_dead", args.xlimit).mean(axis=0)
            new_h = _align_arrays(entry["dfs"], "NewH",       args.xlimit).mean(axis=0)
        else:
            df    = entry["dfs"][0]
            new_i, ddead, new_h = df["NewI"].values, df["delta_dead"].values, df["NewH"].values
        b, s, g, h, gh, d, sd, fdf = fit_seir(new_i, ddead, new_h, args.N, args.seed,
                                               args.xlimit, fixed=fit_fixed)
        fit_results.append((lbl, color, b, s, g, h, gh, d, sd, fdf))

if args.plots is not None:
    selected_plots = args.plots
elif args.seir or fit_results:
    show_hosp = args.hosp_rate != 0
    if show_hosp:
        selected_plots = ["Exposed", "Cumulative Exposed", "Hospitalized", "Dead", "Recovered"]
    else:
        selected_plots = ["Exposed", "Cumulative Exposed", "Recovered"]
else:
    selected_plots = [p for p in ALL_PLOTS if p != "Context"]

n = len(selected_plots)
ncols = 1 if n == 1 else 2
nrows = (n + ncols - 1) // ncols
fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, nrows * 3.5), squeeze=False)
axes = axes_grid.flatten()

for i, plot_name in enumerate(selected_plots):
    if plot_name == "Context":
        plot_context(axes[i], exaepi_data)
    else:
        plot_series(axes[i], epicast_data, exaepi_data, plot_name,
                    seir_df=seir_df, fit_results=fit_results)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

# plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight")
#plt.show()
