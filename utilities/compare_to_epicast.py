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
from read_epicast_events import (
    read_events_bin,
    aggregate_events,
    aggregate_infections_by_source,
    SOURCE_CATEGORIES,
)


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

    # Empirical share of new infections attributable to each interaction context (see
    # aggregate_infections_by_source): the realized-count analog of ExaEpi's analytic
    # E<source>/sum(E<source>) shares, joined in as "<source>_frac" columns.
    src_df = aggregate_infections_by_source(events_df)
    frac_cols = [c + "_frac" for c in SOURCE_CATEGORIES]
    converted_df = converted_df.merge(src_df[["day"] + frac_cols], on="day", how="left")
    converted_df[frac_cols] = converted_df[frac_cols].fillna(0.0)

    return converted_df


# Groups ExaEpi's per-phase context_diag columns into the same buckets Epicast's
# aggregate_infections_by_source uses. ENbhD/ECommD/ENbhN/ECommN are summed into one
# "neighborhood_community" bucket because Epicast records all four under a single context with
# no day/night split (see the comment above _CONTEXT_TO_SOURCE in read_epicast_events.py).
_EXAEPI_SOURCE_MAPPING = {
    "household":              ["EHH"],
    "cluster":                ["ENC"],
    "neighborhood_community": ["ENbhD", "ECommD", "ENbhN", "ECommN"],
    "work":                   ["EWork"],
    "school":                 ["ESchool"],
    "hospital":               ["EHosp"],
}


def _add_exaepi_source_fractions(df):
    """Add "<source>_frac" columns to df: each source bucket's expected-infection contribution
    divided by the day's total across all context_diag columns. No-op (columns simply absent
    downstream) if the run wasn't started with context_diag=true.
    """
    needed_cols = [c for cols in _EXAEPI_SOURCE_MAPPING.values() for c in cols]
    if not all(c in df.columns for c in needed_cols):
        return df
    total = sum(df[c] for c in needed_cols)
    for source, cols in _EXAEPI_SOURCE_MAPPING.items():
        bucket_sum = sum(df[c] for c in cols)
        df[source + "_frac"] = (bucket_sum / total.replace(0, np.nan)).fillna(0.0)
    return df


def load_exaepi(fname):
    df = pd.read_csv(fname, sep="\\s+")
    print(f"Read {len(df)} lines from the ExaEpi file {fname}")
    df = _add_exaepi_source_fractions(df)

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


def _get_group_y(entry, col, xlimit):
    """Return the (possibly averaged, for wildcard groups) y-array for a group entry."""
    if entry["is_wildcard"] and len(entry["dfs"]) > 1:
        return _align_arrays(entry["dfs"], col, xlimit).mean(axis=0)
    return entry["dfs"][0][col].values[:xlimit]


def _shift_array(y, shift, n):
    """Shift array y by `shift` days (same convention as --shift: positive delays it) and
    pad/truncate the result to length n, so it lines up day-for-day with an unshifted reference.
    """
    y = np.asarray(y, dtype=float)
    s = int(round(shift))
    d = np.arange(n)
    src = d - s
    return np.where((src >= 0) & (src < len(y)), y[np.clip(src, 0, len(y) - 1)], 0.0)


def _goodness_of_fit(ref_y, y):
    """Return (R^2, NRMSE) of curve y against reference curve ref_y, comparing over their
    common length. NRMSE is RMSE normalized by the reference curve's mean absolute value.
    Either value is NaN if it isn't computable (e.g. a constant-zero reference).
    """
    n = min(len(ref_y), len(y))
    if n == 0:
        return None
    r = np.asarray(ref_y[:n], dtype=float)
    v = np.asarray(y[:n], dtype=float)
    ss_res = np.sum((r - v) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    ref_mean = np.mean(np.abs(r))
    nrmse = (np.sqrt(ss_res / n) / ref_mean) if ref_mean > 0 else float("nan")
    return r2, nrmse


def _peak_day(y, smooth_window=5):
    """Day-index of y's peak, located on a short moving-average of y rather than the raw series
    so a single noisy day-to-day spike isn't mistaken for the true peak.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return 0
    w = min(smooth_window, len(y))
    smoothed = np.convolve(y, np.ones(w) / w, mode="same")
    return int(np.argmax(smoothed))


# Mapping from plot/series labels to ExaEpi column names, shared by plot_series below.
COL_MAPPING = {
    "exposed": "NewI",
    "symptomatic": "NewS",
    "presymptomatic": "NewP",
    "asymptomatic": "NewA",
    "hospitalized": "NewH",
    "dead": "delta_dead",
    "recovered": "delta_recovered",
    "cumulative_exposed": "cum_exposed",
}


def _auto_shift_per_exaepi_group(epicast_data, exaepi_data, xlimit, shift_range=60):
    """For each ExaEpi input group (i.e. each -x file or wildcard pattern), find the integer
    day-shift (same convention as --shift: positive delays the ExaEpi curve) that lines up the
    peak of THAT group's own 'NewI' curve with the peak of the first Epicast group's 'exposed'
    curve. Every series (Symptomatic, Hospitalized, Dead, ...) plotted for a given ExaEpi group
    reuses that one group-level shift -- shifting is a per-input-file thing, driven by the
    exposed/NewI peak, not something recomputed per series. Peak-matching (rather than minimizing
    RMSE over the whole curve) is used because RMSE picks a poor alignment whenever the two
    curves' overall shapes disagree, even though the peaks themselves are well separated and
    matching them is what actually gives a sensible alignment. Wildcard groups (multiple files
    matched by one -e/-x pattern) are averaged first, same as elsewhere in this script.

    Returns a list parallel to exaepi_data (one shift per group), or a list of 0.0s if there's no
    Epicast reference to align against.
    """
    if not epicast_data or not exaepi_data:
        return [0.0] * len(exaepi_data)

    e_entry = epicast_data[0]
    if e_entry["is_wildcard"] and len(e_entry["dfs"]) > 1:
        e = _align_arrays(e_entry["dfs"], "exposed", xlimit).mean(axis=0)
    else:
        e = e_entry["dfs"][0]["exposed"].values[:xlimit]
    e_peak = _peak_day(e)

    shifts = []
    for x_entry in exaepi_data:
        if x_entry["is_wildcard"] and len(x_entry["dfs"]) > 1:
            x = _align_arrays(x_entry["dfs"], "NewI", xlimit).mean(axis=0)
        else:
            x = x_entry["dfs"][0]["NewI"].values[:xlimit]
        shift = e_peak - _peak_day(x)
        shifts.append(float(np.clip(shift, -shift_range, shift_range)))
    return shifts


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


def _auto_xlimit(epicast_data, exaepi_data, shift_by_group, epicast_shift=0.0, margin=5):
    """Size the x-axis to the minimum, across every -e/-x input file, of that file's furthest
    above-threshold day (see _furthest_nonzero_day) -- i.e. trim to whichever curve runs out of
    signal first, so no plot is padded with a long trailing flat tail from a shorter-lived curve.
    ExaEpi files are measured in their OWN day-numbering then offset by their own group's shift
    (shift_by_group, parallel to exaepi_data, already resolved to numbers by the time this runs),
    since that's where they actually land once plotted. Epicast files are likewise offset by
    `epicast_shift` (nonzero only when an auto shift came out negative and got swapped onto
    Epicast instead -- see the --shift "auto" handling).
    """
    extents = []
    for entry in epicast_data:
        for df in entry["dfs"]:
            extents.append(_furthest_nonzero_day(df, _EPICAST_EXTENT_COLS) + epicast_shift)
    for entry, shift in zip(exaepi_data, shift_by_group):
        for df in entry["dfs"]:
            extents.append(_furthest_nonzero_day(df, _EXAEPI_EXTENT_COLS) + shift)
    if not extents:
        return 250
    return max(1, int(np.ceil(min(extents))) + margin)


def _mark_day_zero(ax, x, label, color):
    """Draw a vertical marker + label at x-position `x`, marking where some curve's own day 0
    lands after a shift is applied, so a shifted curve's origin stays visible instead of implicit.
    """
    ax.axvline(x, color=color, linestyle=":", linewidth=1, zorder=0, alpha=0.7)
    ax.annotate(
        label, xy=(x, 0.98), xycoords=("data", "axes fraction"),
        rotation=90, va="top", ha="right", fontsize=6, color=color, alpha=0.8,
    )


def _mark_exaepi_start(ax, shifts, colors=None):
    """Draw a marker for each distinct ExaEpi group shift (see `_mark_day_zero`). `shifts` is a
    list (one per ExaEpi group); when groups differ, each distinct shift gets its own marker,
    colored to match that group's plotted curve if `colors` (parallel to `shifts`) is given.
    """
    colors = colors if colors is not None else ["#aa0000"] * len(shifts)
    seen = set()
    for shift, color in zip(shifts, colors):
        if shift in seen:
            continue  # groups sharing a shift don't need a duplicate line/label
        seen.add(shift)
        _mark_day_zero(ax, shift, "ExaEpi day 0", color)


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

    for entry, group_shift in zip(exaepi_data, shift_by_group):
        for df in entry["dfs"]:
            x = (df["Day"] + group_shift).values[:args.xlimit]
            for col, (label_str, color) in _CONTEXT_COLS.items():
                if col in df.columns:
                    y = df[col].values[:args.xlimit]
                    ax.plot(x, y, label=label_str, color=color, linewidth=1)

    if exaepi_data:
        _mark_exaepi_start(ax, shift_by_group)

    ax.legend(fontsize=7)


_SOURCE_LABELS = {
    "household":              "Household",
    "cluster":                "Nbhd/HH cluster",
    "neighborhood_community": "Neighborhood+Comm",
    "work":                   "Work",
    "school":                 "School",
    "hospital":               "Hospital",
}

# One plot name per context pair, e.g. "Source: Household" -> source key "household". Keeping
# them as separate ALL_PLOTS entries (rather than one combined panel) means selecting all six
# via -p lands in the existing ncols=2 grid layout as 2 columns x 3 rows automatically.
SOURCE_PLOT_NAMES = [f"Source: {label}" for label in _SOURCE_LABELS.values()]
_SOURCE_PLOT_TO_KEY = {f"Source: {label}": key for key, label in _SOURCE_LABELS.items()}


def _source_frac_max(epicast_data, exaepi_data, source_keys, xlimit):
    """Largest "<source>_frac" value across the given source_keys, over the first -e and first
    -x group only (matching what plot_single_source actually draws). Used to give every "Source:
    ..." subplot in a run the same y-axis peak, sized to whichever of them needs the most room.
    Returns None if no group has any of the requested frac columns.
    """
    vals = []
    for key in source_keys:
        col = key + "_frac"
        if epicast_data:
            df0 = epicast_data[0]["dfs"][0]
            if col in df0.columns:
                vals.append(float(_get_group_y(epicast_data[0], col, xlimit).max()))
        if exaepi_data:
            df0 = exaepi_data[0]["dfs"][0]
            if col in df0.columns:
                vals.append(float(_get_group_y(exaepi_data[0], col, xlimit).max()))
    return max(vals) if vals else None


def plot_single_source(ax, epicast_data, exaepi_data, source_key, title, ylimit):
    """Compare one interaction context's share of new infections between the two models.

    Epicast's share (solid) is an empirical fraction from its realized per-agent context
    attribution (read_epicast_events.aggregate_infections_by_source). ExaEpi's share (dashed)
    is the analytic E<source>/sum(E<source>) share from its context_diag columns, grouped so
    neighborhood/community day+night match Epicast's single merged context (see
    _EXAEPI_SOURCE_MAPPING). Only the first -e and first -x group are shown (averaged if a
    wildcard group).

    ylimit sets the shared y-axis peak across all "Source: ..." subplots in this run (see
    _source_frac_max) so they're visually comparable rather than each auto-scaling to its own
    fraction's range.
    """
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("Fraction of new infections")
    ax.set_xlim([0, args.xlimit])
    ax.set_ylim([0, ylimit])
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.minorticks_on()

    col = source_key + "_frac"

    if epicast_data:
        entry = epicast_data[0]
        df0 = entry["dfs"][0]
        if col in df0.columns:
            x = (df0["day"] + epicast_shift).values[: args.xlimit]
            y = _get_group_y(entry, col, args.xlimit)
            ax.plot(x[: len(y)], y, color="blue", linewidth=1.5, linestyle="-", label="Epicast")

    if exaepi_data:
        entry = exaepi_data[0]
        shift = shift_by_group[0]
        df0 = entry["dfs"][0]
        if col in df0.columns:
            x = (df0["Day"] + shift).values[: args.xlimit]
            y = _get_group_y(entry, col, args.xlimit)
            ax.plot(x[: len(y)], y, color="red", linewidth=1.5, linestyle="--", label="ExaEpi")

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
    epicast_colors = ["blue", "green", "purple", "orange", "brown", "pink"]
    exaepi_colors  = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]

    col_name   = label.lower().replace(" ", "_")
    exaepi_col = COL_MAPPING.get(col_name, col_name)

    auc_lines = []

    _seird_cols = {"exposed", "cumulative_exposed", "hospitalized", "dead", "recovered"}
    seir_col = col_name if col_name in _seird_cols else None

    # Reference curve for goodness-of-fit: the first Epicast group's curve for this series,
    # shifted by epicast_shift so it lines up with the shifted curves it's compared against
    # (epicast_shift is nonzero only when an auto shift came out negative -- see --shift "auto").
    reference_y = (
        _shift_array(_get_group_y(epicast_data[0], col_name, args.xlimit), epicast_shift, args.xlimit)
        if epicast_data else None
    )

    # Plot fitted SEIRHD curves first (under experimental lines)
    if fit_results and seir_col is not None:
        for (series_lbl, _, _, _, _, _, _, _, _, fdf) in fit_results:
            fit_y = fdf[seir_col].values[: args.xlimit]
            ax.plot(np.arange(len(fit_y)), fit_y, color="green", linewidth=3, linestyle="-", zorder=1)
            auc = np.sum(fit_y)
            fit_lbl = f"SEIRHD fit ({series_lbl})" if series_lbl else "SEIRHD fit"
            auc_lines.append((fit_lbl, auc, "green", False, _shift_array(fit_y, 0, args.xlimit), False))

    def _plot_group(entry, i, base_colors, col, x_col=None, x_shift=0.0):
        """Plot one group entry; return (legend_label, auc, color, is_wildcard, y_for_gof)."""
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
            y_for_gof = _shift_array(y_mean, x_shift, args.xlimit)
        else:
            df     = entry["dfs"][0]
            x_vals = (df[x_col] + x_shift) if x_col else np.arange(len(df[col]))
            y_vals = df[col]
            auc    = float(np.sum(y_vals[: args.xlimit]))
            lsargs = {"linestyle": "--"} if base_colors[0] == "blue" else {}
            ax.plot(x_vals, y_vals, label=plot_label, color=color, linewidth=1, zorder=2, **lsargs)
            y_for_gof = _shift_array(y_vals.values, x_shift, args.xlimit)

        return legend_label, auc, color, is_wildcard, y_for_gof

    # Plot each Epicast group, shifted right by epicast_shift (nonzero only when an auto shift
    # came out negative and got swapped onto Epicast instead of ExaEpi -- see --shift "auto").
    for i, entry in enumerate(epicast_data):
        lbl, auc, color, is_wc, y_for_gof = _plot_group(entry, i, epicast_colors, col_name,
                                              x_col="day", x_shift=epicast_shift)
        auc_lines.append((lbl, auc, color, is_wc, y_for_gof, i == 0))

    # Plot each ExaEpi group, each shifted by its OWN group-level shift (from matching that
    # group's exposed/NewI peak -- see _auto_shift_per_exaepi_group).
    for i, entry in enumerate(exaepi_data):
        lbl, auc, color, is_wc, y_for_gof = _plot_group(entry, i, exaepi_colors, exaepi_col,
                                              x_col="Day", x_shift=shift_by_group[i])
        auc_lines.append((lbl, auc, color, is_wc, y_for_gof, False))

    # Plot manual SEIRD curve
    if seir_df is not None and seir_col is not None:
        seir_y = seir_df[seir_col].values[: args.xlimit]
        ax.plot(
            np.arange(len(seir_y)), seir_y,
            label=f"SEIRHD (β={args.beta}, h={args.hosp_rate}, μ={args.mu})",
            color="green", linewidth=2, linestyle="-.",
        )
        auc_lines.append(("SEIRHD", np.sum(seir_y), "green", False, _shift_array(seir_y, 0, args.xlimit), False))

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of " + label)
    ax.set_xlim([0, args.xlimit])

    if epicast_shift:
        # The auto shift came out negative and got swapped onto Epicast (see --shift "auto"), so
        # ExaEpi sits unshifted at day 0 and the interesting origin to call out is Epicast's.
        _mark_day_zero(ax, epicast_shift, "Epicast day 0", epicast_colors[0])
    elif exaepi_data:
        _mark_exaepi_start(ax, shift_by_group, [exaepi_colors[i % len(exaepi_colors)] for i in range(len(exaepi_data))])

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
        for lbl, auc, color, is_wildcard, y_for_gof, is_reference in auc_lines:
            lbl_str = lbl if lbl is not None else "(unlabelled)"
            gof_str = ""
            if not is_reference and reference_y is not None:
                gof = _goodness_of_fit(reference_y, y_for_gof)
                if gof is not None:
                    r2, nrmse = gof
                    r2_str    = f"{r2:.3f}"    if np.isfinite(r2)    else "N/A"
                    nrmse_str = f"{nrmse:.3f}" if np.isfinite(nrmse) else "N/A"
                    gof_str = f"  R²={r2_str}  NRMSE={nrmse_str}"
            print(f"  AUC {lbl_str}: {auc:,.0f}{gof_str}")
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
    help="Shift the ExaEpi curve(s) along the x-axis in days, or 'auto' to pick a shift "
         "independently per -x input file/group (clamped to +/-60 days each) that lines up that "
         "group's own 'NewI' (exposed) peak with the first -e input's 'exposed' peak. Every "
         "series (Symptomatic, Hospitalized, Dead, ...) plotted for a given ExaEpi group reuses "
         "that same group-level shift -- only different -x inputs can end up with different "
         "shifts, not different series of the same input (default: 0)",
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
    "Hospitalized", "Dead", "Recovered", "Cumulative Exposed", "Context", *SOURCE_PLOT_NAMES,
]
_plot_map = {p.lower(): p for p in ALL_PLOTS}
_plot_map["source fractions"] = SOURCE_PLOT_NAMES  # legacy alias: expands to all 6 context plots

if args.plots is not None:
    resolved = []
    for name in args.plots:
        canonical = _plot_map.get(name.lower())
        if canonical is None:
            parser.error(f"Unknown plot '{name}'. Valid names: {', '.join(ALL_PLOTS)}")
        elif isinstance(canonical, list):
            resolved.extend(canonical)
        else:
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


def _write_epicast_csv(fname, df, shift):
    out = df.copy()
    out["day"] = out["day"] + shift
    csv_out = fname + "-plot_values.csv"
    out.to_csv(csv_out, index=False)
    print(f"Wrote plot values to {csv_out}")


def _write_exaepi_csv(fname, df, shift):
    out = pd.DataFrame({
        "day":               df["Day"] + shift,
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


epicast_data = _load_grouped(args.epicast_file, load_epicast, None)
# CSV-writing is deferred until after shift_by_group/epicast_shift are resolved below (they embed
# the group's own shift into the "day" column, which isn't known yet if --shift auto was requested).
exaepi_data  = _load_grouped(args.exaepi_file,  load_exaepi,  None)

epicast_shift = 0.0
if args.shift == "auto":
    # args.xlimit may itself still be "auto" here -- it only bounds how much of the curve the
    # shift search looks at, so fall back to the parser's own default (250) rather than depending
    # on the xlimit auto-sizing below (which in turn depends on shift_by_group already being
    # resolved).
    _shift_window = args.xlimit if isinstance(args.xlimit, (int, float)) else 250
    shift_by_group = _auto_shift_per_exaepi_group(epicast_data, exaepi_data, _shift_window)
    _min_shift = min(shift_by_group) if shift_by_group else 0.0
    if _min_shift < 0:
        # A negative shift would push ExaEpi left of day 0, off the start of the plot. Since only
        # the RELATIVE offset between the curves matters, add the same constant to every ExaEpi
        # shift and to Epicast instead: this pins the most-negative ExaEpi group at day 0 and
        # shifts Epicast right by the same amount, preserving every peak alignment unchanged.
        epicast_shift = -_min_shift
        shift_by_group = [s + epicast_shift for s in shift_by_group]
    for _i, _s in enumerate(shift_by_group):
        _lbl = exaepi_data[_i]["label"] or f"ExaEpi input {_i + 1}"
        print(f"Auto-detected shift for {_lbl}: {_s:+.0f} days (aligns that file's exposed/NewI peak)")
    if epicast_shift:
        print(f"Auto-detected shift was negative; keeping ExaEpi at day 0 and shifting "
              f"Epicast +{epicast_shift:.0f} days instead")
    # args.shift keeps a single scalar for callers with no per-group concept (just the xlimit
    # fallback default below): the first group's shift.
    args.shift = shift_by_group[0] if shift_by_group else 0.0
else:
    shift_by_group = [args.shift] * len(exaepi_data)

if args.xlimit == "auto":
    args.xlimit = _auto_xlimit(epicast_data, exaepi_data, shift_by_group, epicast_shift)
    print(f"Auto-detected xlimit: {args.xlimit} days (minimum furthest >=10 extent across all inputs)")

for _entry in epicast_data:
    for _fname, _df in zip(_entry["fnames"], _entry["dfs"]):
        _write_epicast_csv(_fname, _df, epicast_shift)

for _entry, _shift in zip(exaepi_data, shift_by_group):
    for _fname, _df in zip(_entry["fnames"], _entry["dfs"]):
        _write_exaepi_csv(_fname, _df, _shift)

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
    selected_plots = [p for p in ALL_PLOTS if p != "Context" and p not in SOURCE_PLOT_NAMES]

n = len(selected_plots)
ncols = 1 if n == 1 else 2
nrows = (n + ncols - 1) // ncols
scale = 2 if n <= 2 else 1
fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols * scale, nrows * 3.5 * scale), squeeze=False)
axes = axes_grid.flatten()

_selected_source_keys = [_SOURCE_PLOT_TO_KEY[p] for p in selected_plots if p in _SOURCE_PLOT_TO_KEY]
if args.ylimit is not None:
    source_ylimit = args.ylimit
else:
    _max_frac = _source_frac_max(epicast_data, exaepi_data, _selected_source_keys, args.xlimit)
    source_ylimit = 1.1 * _max_frac if _max_frac is not None else 1.0

for i, plot_name in enumerate(selected_plots):
    if plot_name == "Context":
        plot_context(axes[i], exaepi_data)
    elif plot_name in _SOURCE_PLOT_TO_KEY:
        plot_single_source(axes[i], epicast_data, exaepi_data, _SOURCE_PLOT_TO_KEY[plot_name], plot_name,
                            source_ylimit)
    else:
        plot_series(axes[i], epicast_data, exaepi_data, plot_name,
                    seir_df=seir_df, fit_results=fit_results)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

# plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight")
#plt.show()
