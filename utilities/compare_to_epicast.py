#!/usr/bin/env -S python -u

import sys
import os
import glob
import io
import contextlib
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(__file__))
from read_epicast_events import read_epicast_summary, EPICAST_SUMMARY_SUFFIX
from plos_compbio_style import apply_style, FULL_PAGE_WIDTH_IN, FONT_TICK, AXES_LINEWIDTH
import seirhd_params

apply_style()


def load_epicast(fname):
    if fname.endswith(EPICAST_SUMMARY_SUFFIX):
        # Pre-extracted by extract_epicast_data.py: already the small per-day summary, so skip
        # the (often multi-GB, tens-of-seconds) raw events.bin parse entirely.
        print(f"Reading pre-extracted Epicast summary {fname} ...")
        df = pd.read_csv(fname)
        print(f"Epicast has {len(df)} days (from pre-extracted summary)")
        return df
    return read_epicast_summary(fname)


# Groups ExaEpi's per-phase context_diag columns into the same buckets Epicast's
# aggregate_infections_by_source uses. ENbhD/ECommD/ENbhN/ECommN are summed into one
# "neighborhood_community" bucket because Epicast records all four under a single context with
# no day/night split (see the comment above _CONTEXT_TO_SOURCE in read_epicast_events.py). EHosp
# is deliberately omitted: it's always ~0 here, matching Epicast's ctx_hospitalized never being an
# infection source in any checked run (see the comment above _CONTEXT_TO_SOURCE).
_EXAEPI_SOURCE_MAPPING = {
    "household":              ["EHH"],
    "cluster":                ["ENC"],
    "neighborhood_community": ["ENbhD", "ECommD", "ENbhN", "ECommN"],
    "work":                   ["EWork"],
    "school":                 ["ESchool"],
}


def _add_exaepi_source_fractions(df):
    """Add "<source>_frac" columns to df: each source bucket's expected-infection contribution
    divided by the run's grand total across all days and all context_diag columns (NOT that
    day's own total) -- see the matching normalization in aggregate_infections_by_source, which
    this must match for the two models' curves to be comparable. No-op (columns simply absent
    downstream) if the run wasn't started with context_diag=true.
    """
    needed_cols = [c for cols in _EXAEPI_SOURCE_MAPPING.values() for c in cols]
    if not all(c in df.columns for c in needed_cols):
        return df
    grand_total = df[needed_cols].to_numpy().sum()
    for source, cols in _EXAEPI_SOURCE_MAPPING.items():
        bucket_sum = df[cols].sum(axis=1)
        df[source + "_frac"] = (bucket_sum / grand_total) if grand_total > 0 else 0.0
    return df


# The compartments an agent is in exactly one of, so their sum is the run's total population.
# ICU and V are deliberately absent: both are subsets of the hospitalized columns (H/NI, H/I) --
# main.cpp fills them from a separate reduction -- so counting them would overstate the population
# (by 190,384 on the CA run this was checked against).
_EXAEPI_COMPARTMENTS = ["Su", "PS/PI", "S/PI/NH", "S/PI/H", "PS/I", "S/I/NH", "S/I/H",
                        "A/PI", "A/I", "H/NI", "H/I", "R", "D"]


def _exaepi_population(df):
    """Total agents in an ExaEpi run, for turning a cumulative count into an attack rate.

    Read off row 0 rather than summed some other way, and no agent ever enters or leaves, so this
    is exactly constant over a run (verified on CA: 39,247,867 on every one of 250 days).

    Epicast's per-day summary carries no population, so the same figure is used for both models.
    That is not an approximation here: the demographics block in an Epicast events.bin totals
    39,247,867 for CA as well -- the two models are handed the same population -- but --population
    overrides it for a dataset where they are not.
    """
    if not all(c in df.columns for c in _EXAEPI_COMPARTMENTS):
        return None
    return float(df[_EXAEPI_COMPARTMENTS].iloc[0].sum())


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
        frac_hosp = num_hosp / num_symp if num_symp > 0 else float("nan")
        print(f"  {ages[i]:8s}   {num_hosp:8.0f} {frac_hosp:.3f}")

    tot_symp = float(df.NewS.sum())
    tot_hosp = float(df.NewH.sum())
    tot_exposed = float(df.NewI.sum())
    frac_symp = tot_symp / tot_exposed if tot_exposed > 0 else float("nan")
    frac_hosp = tot_hosp / tot_symp if tot_symp > 0 else float("nan")
    print(f"ExaEpi total symptomatic {tot_symp} {frac_symp:.2f}")
    print(f"ExaEpi total hospitalized {tot_hosp} {frac_hosp:.2f}")

    if not fname.startswith("adjusted"):
        transformed_df = df.copy()
        transformed_df["Day"] += 4
        for col in transformed_df.columns:
            if col != "Day":
                transformed_df[col] *= 1
        # transformed_df.to_csv("adjusted-" + fname, index=False, sep=" ")

    return df


def run_seirhd_erlang(beta, sigma, gamma, h, gamma_h, mu, N, seed, days, kE=3, kI=4, kD=1):
    """Run a SEIRHD model with Erlang (gamma-distributed) compartment sojourn times.

    Compartments are S -> E -> I -> {R, H} and H -> {R, D}, with

        beta     S -> E   transmission rate
        sigma    E -> I   1/sigma        = mean latent period
        gamma    I -> R   1/(gamma + h)  = mean infectious period
        h        I -> H   h/(gamma + h)  = P(hospitalised | infected)
        gamma_h  H -> R   1/(gamma_h+mu) = mean hospital stay
        mu       H -> D   mu/(gamma_h+mu)= P(dead | hospitalised)

    A plain one-compartment stage gives its sojourn time an exponential
    distribution -- CV = 1, mode at zero -- which is a poor match for a
    latent or infectious period. So each of E, I and H is instead a chain of
    k sequential exponential sub-stages run at k times the rate, making the
    *total* time through the chain Gamma(shape=k, scale=1/(k*rate)): the same
    mean, but CV = 1/sqrt(k). kE, kI and kD choose those shapes, and kD=1
    leaves H as the single exponential stage.

    The I chain's two exits compete only at its last sub-stage, and likewise
    the recovery/death split (frac_hr/frac_hd) happens only at the end of the
    H chain, so both hospital outcomes share one dwell-time distribution --
    matching ExaEpi, which draws a single hospital-stay length per agent
    (checkHospitalization in DiseaseParm.H) regardless of whether that agent
    goes on to recover or die.

    See seirhd_params.py, which derives every one of these from an ExaEpi
    .ini -- the rates from each compartment's mean dwell time and the shapes
    from its CV.
    """

    rate_E = kE * sigma
    rate_I = kI * (gamma + h)
    rate_H = kD * (gamma_h + mu)
    frac_h = h / (gamma + h)       # I -> hospitalized vs. I -> recovered directly
    frac_r = gamma / (gamma + h)
    frac_hr = gamma_h / (gamma_h + mu)  # hospitalized -> recovered vs. -> dead, at discharge
    frac_hd = mu / (gamma_h + mu)

    n_state = 1 + kE + kI + kD + 2
    i_E, i_I, i_H = 1, 1 + kE, 1 + kE + kI
    i_R           = i_H + kD
    i_D           = i_R + 1

    def seirhd_erlang_odes(t, y):
        S = y[0]
        E = y[i_E:i_I]
        I = y[i_I:i_H]
        H = y[i_H:i_R]
        R, D = y[i_R], y[i_D]

        I_total = I.sum()
        inf = beta * S * I_total / N

        dy = np.empty(n_state)
        dy[0] = -inf

        dy[i_E]        = inf - rate_E * E[0]
        dy[i_E + 1:i_I] = rate_E * E[:-1] - rate_E * E[1:]

        dy[i_I]        = rate_E * E[-1] - rate_I * I[0]
        dy[i_I + 1:i_H] = rate_I * I[:-1] - rate_I * I[1:]

        I_last = I[-1]
        hosp_in = rate_I * frac_h * I_last

        dy[i_H]        = hosp_in - rate_H * H[0]
        dy[i_H + 1:i_R] = rate_H * H[:-1] - rate_H * H[1:]

        H_last = H[-1]
        dy[i_R] = rate_I * frac_r * I_last + frac_hr * rate_H * H_last
        dy[i_D] = frac_hd * rate_H * H_last
        return dy

    y0 = np.zeros(n_state)
    y0[0]    = float(N - seed)
    y0[i_I]  = float(seed)  # seed the first infectious sub-stage, matching run_seir

    t_eval = np.arange(0, days + 1, 1, dtype=float)
    sol = solve_ivp(seirhd_erlang_odes, [0, days], y0, t_eval=t_eval, method="RK45", max_step=0.1)

    S = sol.y[0]
    I_last = sol.y[i_H - 1]  # last I sub-stage: only stage that feeds H/R
    R = sol.y[i_R]
    D = sol.y[i_D]

    new_exposed = np.maximum(0, -np.diff(S))

    I_last_mid = 0.5 * (I_last[:-1] + I_last[1:])
    new_hospitalized = np.maximum(0, kI * h * I_last_mid)

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
    print(f"SEIRHD-Erlang(kE={kE}, kI={kI}, kD={kD})  exposed={new_exposed.sum():.0f}  "
          f"hosp={new_hospitalized.sum():.0f}  dead={new_dead.sum():.0f}")
    print(f"SEIRHD-Erlang  R0={r0:.2f}  hosp_rate={h/(gamma+h):.4f}  HFR={hfr:.4f}  IFR={ifr:.4f}  "
          f"E_CV={1/np.sqrt(kE):.3f}  I_CV={1/np.sqrt(kI):.3f}  D_CV={1/np.sqrt(kD):.3f}")

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


def _medoid_index(y_mat):
    """Index of the row in y_mat (files x days) with the smallest total (Euclidean) distance to
    every other row -- the file whose full day-by-day curve is most representative of the group
    as a whole.

    Used to pick an actual constituent file as a wildcard group's representative curve, rather
    than a pointwise day-by-day mean/median across files (which would blend pre-peak and
    post-peak values from files with different outbreak timing, flattening and widening the
    result relative to any single real run) or a single scalar summary like AUC (which, for a
    flow/rate column such as daily new exposures, collapses to that file's total size and says
    nothing about *when* those exposures happened -- so ranking by it can pick a file with an
    unremarkable total but wildly atypical timing). Comparing full day-by-day curves accounts
    for magnitude and timing together, for any column type.
    """
    diffs = y_mat[:, None, :] - y_mat[None, :, :]  # (n_files, n_files, n_days)
    dist  = np.sqrt(np.sum(diffs ** 2, axis=2))    # (n_files, n_files) pairwise distance
    return int(np.argmin(dist.sum(axis=1)))


def _get_group_y(entry, col, xlimit):
    """Return the y-array for a group entry: a single file's values directly, or -- for a
    wildcard group matching multiple files -- the actual file closest to the group's medoid
    (see _medoid_index)."""
    if entry["is_wildcard"] and len(entry["dfs"]) > 1:
        y_mat = _align_arrays(entry["dfs"], col, xlimit)
        return y_mat[_medoid_index(y_mat)]
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


_ENVELOPE_SMOOTH_WINDOW = 9

# Central percentage of runs a wildcard group's shaded band covers; 100 is the pointwise
# min/max. Overridden by --band, which takes any number of coverages (see _draw_spread_bands).
_DEFAULT_BAND_COVERAGE = 100.0

# Fill opacity for a spread band. Drawn with no edge at all, not merely a matching one: passing
# fill_between a single `color=` sets the face AND the edge, and at alpha the two compound wherever
# they overlap, ringing every band in a darker outline that reads as a plotted boundary rather than
# as the edge of a shaded region -- particularly misleading here, where the band's edges are
# percentiles across runs and not any run's own curve.
_BAND_ALPHA = 0.12


def _draw_spread_bands(ax, x, y_mat, color, label=None):
    """Shade each --band coverage for one group, and say whether anything was drawn.

    Several coverages nest into a fan: the widest is laid down first and the narrower ones over
    it, so the alpha of the overlap does the work and the core -- where the runs actually are --
    comes out darker than the extremes without a single boundary being drawn anywhere. That reads
    as a density, which one band on its own cannot: a min/max band in particular is an extent, and
    on the CA replicates half the runs sit inside a fifth of its width while 5 runs of 30 account
    for every edge of it. `--band 50 90` is the useful pairing -- the bulk, and how far the tails
    reach.

    Only the first fill carries `label`, so a group is one legend entry rather than one per band.
    """
    drawn = False
    for i, coverage in enumerate(args.band):
        band_lo, band_hi = _smoothed_band(y_mat, coverage)
        ax.fill_between(x[: y_mat.shape[1]], band_lo, band_hi, alpha=_BAND_ALPHA, facecolor=color,
                        edgecolor="none", zorder=1,
                        label=label if (i == 0 and label is not None) else "_nolegend_")
        drawn = True
    return drawn


def _smoothed_band(y_mat, coverage=_DEFAULT_BAND_COVERAGE, window=_ENVELOPE_SMOOTH_WINDOW):
    """Pointwise central-`coverage`-percent interval across the rows in y_mat (files x days),
    lightly smoothed (a `window`-day moving average).

    `coverage` is the percentage of runs the band is meant to contain: the edges are the
    (100-coverage)/2 and (100+coverage)/2 percentiles, so the default 100 is exactly the
    pointwise min/max and e.g. 90 is the 5th-95th percentile. Taking a percentile rather than
    always the extremes matters once the spread is skewed, because a min/max edge is by
    construction owned by a single run at every day. On the emerge-paper CA p01 replicates a
    single late run held the entire upper edge over the whole falling limb, with the 95th
    percentile well below it (how far below depends on the run set), so the band showed that
    one run's timing as the model's extent. Below 100 the band shows where the bulk of the runs
    are and the medoid line sits centered in it; at 100 it shows worst-case extent. Percentile
    edges are interpolated between runs, so they are not in general any single run's curve
    (unlike min/max, which always is).

    The smoothing removes the day-to-day jaggedness inherent in taking an extreme statistic over
    few samples: at any single day the max is whichever one run happens to be highest *that day*,
    and which run "wins" can flip from day to day as different runs pass through their own
    rise/peak/fall, producing a spurious multi-humped trace even though every run is a clean
    single-peaked curve (confirmed against the emerge-paper CA replicate runs,
    output_ca_epicast_p01-c35-r*.dat: the raw pointwise max had 5 local extrema in a 70-day window
    spanning the peak; a 9-day moving average brings that down to 1, while changing the peak
    height by under 1% and every far-field value by only a few percent). Interior percentiles are
    smoother to begin with, but they get the same treatment so that the only thing --band changes
    is which statistic is drawn.

    Two approaches that were tried and rejected in favor of this one (see git history):
    - Shifting/aligning each run's curve to a common peak day before taking mean +/- std: this
      assumes every run is a simple time-translate of one shape, which can silently hide or
      distort a run whose shape genuinely differs (steeper rise, different width) rather than
      just being phase-shifted -- and a follow-on band built by sliding the aligned/medoid curve
      to represent the timing spread reintroduced the same multi-humped artifact this function
      fixes, only worse (built from just 2-3 widely-spaced translated copies instead of blending
      all N runs).
    - Plotting every run as its own translucent line: faithful to the data, but alpha-blending
      means whatever region happens to have the most overlapping runs (typically right around the
      shared peak) renders visibly more solid/opaque than the sparser flanks, which can compete
      with an overlaid reference curve (e.g. Epicast) for visual attention. A single smoothed fill
      has uniform opacity everywhere regardless of how many runs agree in a given region.

    Returns (lo, hi), both smoothed, same length as y_mat's columns.
    """
    kernel = np.ones(window) / window
    pad = window // 2

    def _smooth(y):
        # Edge-pad before convolving (then keep only the "valid", fully-overlapping positions)
        # instead of plain np.convolve(y, kernel, mode="same"): "same" mode implicitly zero-pads
        # past the array's ends, so for the last/first `pad` days the average silently mixes in
        # phantom zeros instead of real data. That's invisible on a series whose true value is
        # already near zero out there (an epidemic curve's tail), but for a monotonically
        # increasing series like cumulative exposed -- genuinely near its plateau maximum at the
        # tail, not near zero -- it drags the last few smoothed days down toward zero for no
        # reason (confirmed directly: ca-p01.png's Cumulative Exposed band plunged from ~24.29M to
        # ~13.47M over the last 4 days purely from this, even though every underlying run's actual
        # value there was still ~24.29M). Edge-padding holds the boundary value instead of
        # inventing zeros, so the smoothed boundary tracks the real data.
        return np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")

    lo_pct = (100.0 - coverage) / 2.0
    lo = _smooth(np.percentile(y_mat, lo_pct, axis=0))
    hi = _smooth(np.percentile(y_mat, 100.0 - lo_pct, axis=0))
    return lo, hi


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
    matched by one -e/-x pattern) use their medoid file's curve (via _get_group_y), the same
    representative curve plotted as that group's line elsewhere in this script, so the shift
    lines up with what's drawn.

    Returns a list parallel to exaepi_data (one shift per group), or a list of 0.0s if there's no
    Epicast reference to align against.
    """
    if not epicast_data or not exaepi_data:
        return [0.0] * len(exaepi_data)

    e_entry = epicast_data[0]
    e_peak = _peak_day(_get_group_y(e_entry, "exposed", xlimit))

    shifts = []
    for x_entry in exaepi_data:
        x = _get_group_y(x_entry, "NewI", xlimit)
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


def _format_shift(shift):
    """A shift in days as a signed label, e.g. '+12 d', '-5 d', '+3.5 d'. The sign is always
    shown: it carries the --shift convention (positive delays the curve), which an unsigned
    number would leave ambiguous.
    """
    s = float(shift)
    num = f"{s:+.0f}" if float(s).is_integer() else f"{s:+.1f}"
    return f"{num} d"


# Points below the axes bottom at which the shift value is annotated: far enough down to clear
# the tick labels, but still inside the band constrained_layout already reserves for the x-label,
# so the text isn't clipped at the figure edge (it is not itself part of the layout).
_SHIFT_LABEL_OFFSET_PT = -15


def _mark_day_zero(ax, x, label, color, sublabel=None, label_top=0.98):
    """Draw a vertical marker + label at x-position `x`, marking where some curve's own day 0
    lands after a shift is applied, so a shifted curve's origin stays visible instead of implicit.

    `sublabel`, if given, is written below the x-axis at the same x (see
    _SHIFT_LABEL_OFFSET_PT) -- the marker shows *where* the curve was moved to, the sublabel
    says by *how much*, which otherwise has to be read off the axis by eye.

    `label_top` is the axes-fraction height the (rotated, top-aligned) label hangs down from;
    lower it on a panel whose top strip is spoken for, e.g. the legend headroom the stacked-source
    panels reserve (see _STACK_LEGEND_HEADROOM), where the default runs the text behind the legend.

    Call this after the axes' x-limits are set: a marker for a negative shift lands left of the
    plotted range, where the vertical line is clipped away, and the sublabel is suppressed to
    match rather than left floating under the axis annotating a line that isn't drawn.
    """
    ax.axvline(x, color=color, linestyle=":", linewidth=1, zorder=0, alpha=0.7)
    ax.annotate(
        label, xy=(x, label_top), xycoords=("data", "axes fraction"),
        rotation=90, va="top", ha="right", fontsize=FONT_TICK, color=color, alpha=0.8,
    )
    xlo, xhi = ax.get_xlim()
    if sublabel is not None and xlo <= x <= xhi:
        ax.annotate(
            sublabel, xy=(x, 0), xycoords=("data", "axes fraction"),
            xytext=(0, _SHIFT_LABEL_OFFSET_PT), textcoords="offset points",
            ha="center", va="top", fontsize=FONT_TICK, color=color, alpha=0.8,
            annotation_clip=False,
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
        if shift:
            _mark_day_zero(ax, shift, "ExaEpi day 0", color, _format_shift(shift))


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
    # Shortened from "Expected infections by context (ExaEpi)" -- doesn't fit at PLOS's 8-12pt
    # font floor on this panel's now-much-smaller width.
    ax.set_title("Infections by context (ExaEpi)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Expected new infections")
    ax.set_xlim([0, args.xlimit])
    ax.grid(True, which="major", linewidth=AXES_LINEWIDTH)
    ax.grid(True, which="minor", alpha=0.3, linewidth=AXES_LINEWIDTH)
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

    ax.legend()


_SOURCE_LABELS = {
    "household":              "Household",
    "cluster":                "Nbhd/HH cluster",
    "neighborhood_community": "Neighborhood+Comm",
    "work":                   "Work",
    "school":                 "School",
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
    """Compare one interaction context's daily contribution to total infections between the
    two models, each expressed as a fraction of that model's own run-wide total exposed count
    (not that day's total -- see aggregate_infections_by_source / _add_exaepi_source_fractions).
    That keeps each curve shaped like its raw daily-count time series, so the comparison stays
    informative during the epidemic's peak rather than being dominated by noisy day-to-day
    ratios where daily counts are small (e.g. at the start/tail of the epidemic).

    Epicast's curve (solid) is an empirical fraction from its realized per-agent context
    attribution. ExaEpi's curve (dashed) is the analytic E<source>/(run-wide total E) share from
    its context_diag columns, grouped so neighborhood/community day+night match Epicast's single
    merged context (see _EXAEPI_SOURCE_MAPPING). Only the first -e and first -x group are shown
    (as the medoid curve, with a smoothed shaded band over the central --band percent of the
    files, if a wildcard group matches multiple files -- see _smoothed_band).

    ylimit sets the shared y-axis peak across all "Source: ..." subplots in this run (see
    _source_frac_max) so they're visually comparable rather than each auto-scaling to its own
    fraction's range.
    """
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("Fraction of total exposed")
    ax.set_xlim([0, args.xlimit])
    ax.set_ylim([0, ylimit])
    ax.grid(True, which="major", linewidth=AXES_LINEWIDTH)
    ax.grid(True, which="minor", alpha=0.3, linewidth=AXES_LINEWIDTH)
    ax.minorticks_on()

    col = source_key + "_frac"
    row = 0

    print(title)

    # Reference curve for goodness-of-fit: Epicast's (day-aligned, post-shift) curve, compared
    # against ExaEpi's below -- same convention as plot_series's reference_y.
    reference_y = None

    if epicast_data:
        entry = epicast_data[0]
        legend_label = entry["label"]
        df0 = entry["dfs"][0]
        if col in df0.columns:
            x = (df0["day"] + epicast_shift).values[: args.xlimit]
            y = _get_group_y(entry, col, args.xlimit)
            band_drawn = False
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                y_mat = _align_arrays(entry["dfs"], col, args.xlimit)
                medoid_idx = _medoid_index(y_mat)
                print(f"  Medoid file (Epicast, {col}): {entry['fnames'][medoid_idx]}")
                peak_days = [_peak_day(row) for row in y_mat]
                print(f"  Peak-day range (Epicast, {col}): "
                      f"[{min(peak_days)}, {max(peak_days)}]  std={np.std(peak_days):.1f}d")
                band_drawn = _draw_spread_bands(
                    ax, x, y_mat, "blue", "Epicast" if args.band_only else None)
            # See _plot_group in plot_series: --band_only leaves the band to speak for the group,
            # but only where there is one, so a group is never silently left off the plot.
            if not (args.band_only and band_drawn):
                ax.plot(x[: len(y)], y, color="blue", linewidth=1, linestyle="-", label="Epicast")
            auc = float(np.sum(y))
            print(f"  Epicast AUC: {auc:.3f}")
            if legend_label is not None:
                text = f"{legend_label} AUC: {auc:.3f}" if args.show_auc else legend_label
                ax.text(0.98, 0.95 - row * 0.09, text,
                        transform=ax.transAxes, ha="right", va="top", fontsize=FONT_TICK, color="blue")
                row += 1
            reference_y = _shift_array(y, epicast_shift, args.xlimit)

    if exaepi_data:
        entry = exaepi_data[0]
        legend_label = entry["label"]
        shift = shift_by_group[0]
        df0 = entry["dfs"][0]
        if col in df0.columns:
            x = (df0["Day"] + shift).values[: args.xlimit]
            y = _get_group_y(entry, col, args.xlimit)
            band_drawn = False
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                y_mat = _align_arrays(entry["dfs"], col, args.xlimit)
                medoid_idx = _medoid_index(y_mat)
                print(f"  Medoid file (ExaEpi, {col}): {entry['fnames'][medoid_idx]}")
                peak_days = [_peak_day(row) for row in y_mat]
                print(f"  Peak-day range (ExaEpi, {col}): "
                      f"[{min(peak_days)}, {max(peak_days)}]  std={np.std(peak_days):.1f}d")
                band_drawn = _draw_spread_bands(
                    ax, x, y_mat, "red", "ExaEpi" if args.band_only else None)
            # See _plot_group in plot_series: --band_only leaves the band to speak for the group,
            # but only where there is one, so a group is never silently left off the plot.
            if not (args.band_only and band_drawn):
                ax.plot(x[: len(y)], y, color="red", linewidth=1, linestyle="-", label="ExaEpi")
            auc = float(np.sum(y))
            gof_str = ""
            if reference_y is not None:
                gof = _goodness_of_fit(reference_y, _shift_array(y, shift, args.xlimit))
                if gof is not None:
                    r2, nrmse = gof
                    r2_str    = f"{r2:.3f}"    if np.isfinite(r2)    else "N/A"
                    nrmse_str = f"{nrmse:.3f}" if np.isfinite(nrmse) else "N/A"
                    gof_str = f"  R²={r2_str}  NRMSE={nrmse_str}"
            print(f"  ExaEpi AUC: {auc:.3f}{gof_str}")
            if legend_label is not None:
                text = f"{legend_label} AUC: {auc:.3f}" if args.show_auc else legend_label
                ax.text(0.98, 0.95 - row * 0.09, text,
                        transform=ax.transAxes, ha="right", va="top", fontsize=FONT_TICK, color="red")
                row += 1


# Bottom-to-top stacking order for the stacked-composition panels, and one color per source.
# "other" (Epicast's ctx_customer/ctx_bar_social bucket -- see _CONTEXT_TO_SOURCE) has no ExaEpi
# counterpart in _EXAEPI_SOURCE_MAPPING, so it simply doesn't appear on the ExaEpi panel.
_SOURCE_STACK_ORDER = ["household", "cluster", "neighborhood_community", "work", "school", "other"]

# Shorter than _SOURCE_LABELS: these go in one legend row spanning the figure, and the full names
# ("Neighborhood+Comm") make it wider than the figure itself at PLOS's 8pt legend font.
_SOURCE_STACK_LABELS = {
    "household":              "Household",
    "cluster":                "HH Cluster",
    "neighborhood_community": "Nbhd+Comm",
    "work":                   "Work",
    "school":                 "School",
    "other":                  "Other",
}

# Muted versions of the tab10 hues the Context panel uses (_CONTEXT_COLS), so a source still reads
# as the same color there, just softer -- solid bands covering the whole axes at full tab10
# saturation are much louder than the thin lines those hues were picked for.
#
# Chosen against the categorical-palette checks rather than by eye, in the stacking order below,
# since that is the order the bands actually touch in. Two results worth recording:
#
#   * The pair that decides the palette is Household/Cluster, and the tab10 pair it replaces
#     (#d62728/#8c564b) FAILED outright -- 14.6 normal-vision dE, below the 15 floor, and 4.6 under
#     protanopia. Muting alone makes that worse, because chroma is what carries the separation; what
#     fixes it is stepping LIGHTNESS along the stack (dark red, light tan, mid green, dark blue,
#     light orange), which survives color-vision deficiency where hue alone does not. This palette
#     comes out at 16.9 normal-vision and 6.8 protan, so it is both softer and better separated.
#   * 6.8 is inside the band that needs a secondary encoding to be legible, which is what the white
#     boundary line between bands in plot_source_stack provides (it is also just a good idea on a
#     stacked chart). The low-chroma warning is inherent to the ask: muted means less chroma.
#
# "other" stays neutral gray, the usual convention for a catch-all bucket, and never appears
# alongside the rest in any checked run anyway.
_SOURCE_STACK_COLORS = {
    "household":              "#a85252",
    "cluster":                "#d6a677",
    "neighborhood_community": "#5f9e6e",
    "work":                   "#4c72a8",
    "school":                 "#e39c4e",
    "other":                  "#9a9a9a",
}

# One panel per model: a stacked composition is a single model's breakdown, so unlike the
# "Source: ..." line panels the two models can't share one axes. The -p name has to say which
# panel it selects; the panel's own title doesn't, since the two sit side by side and the figure
# is only ever about these two models -- so it's just the model name.
SOURCE_STACK_PLOT_NAMES = ["Source Stack (Epicast)", "Source Stack (ExaEpi)"]
_SOURCE_STACK_TITLES = {"epicast": "Epicast", "exaepi": "ExaEpi"}

# Filled by plot_source_stack, {label: bar container}, in stacking order. Both panels draw the same
# sources, so one legend serves them both; it's added to the FIGURE once every panel is drawn (see
# after the plotting loop) rather than per-axes, which would repeat it and eat into the plot area.
SOURCE_STACK_HANDLES = {}

# Blank fraction of the axes left above the bars, so a day-0 marker's rotated label has somewhere
# to go: the bars fill 0..1, and the label would otherwise sit on top of whichever band is at the
# top. Reserved on every stacked panel, including those with no marker to place, because the panels
# are meant to be read against each other and would otherwise be drawn at different y-scales. The
# legend needs no room here -- it is a single figure-level one along the bottom (see the
# SOURCE_STACK_HANDLES block after the plotting loop).
_STACK_LABEL_HEADROOM = 0.15
_SOURCE_STACK_TO_MODEL = {
    "Source Stack (Epicast)": "epicast",
    "Source Stack (ExaEpi)":  "exaepi",
}


def _daily_source_composition(entry, xlimit, window=1):
    """Each day's infections split into per-source fractions that sum to 1 -- i.e. the mix of
    contexts driving transmission on that day, independent of how large that day's outbreak is.

    This is a different normalization from the one the "Source: ..." line panels use: there each
    source is divided by the run's grand total across all days (see _add_exaepi_source_fractions /
    aggregate_infections_by_source), so the curves keep the shape of the raw daily counts. Here
    each day is divided by its OWN total instead, which is what makes the bars all reach 1 and
    turns the panel into a picture of composition over time rather than of magnitude.

    Both models' "<source>_frac" columns share one run-wide divisor, so that constant cancels in
    the per-day ratio and the two models' compositions are directly comparable even though their
    underlying quantities aren't (Epicast's are realized event counts, ExaEpi's are analytic
    expected infections).

    For a wildcard group matching several files, the sources are pooled across files before the
    per-day ratio is taken, rather than picking one representative file as the line panels do
    (_get_group_y/_medoid_index): a composition is a ratio, and pooling numerators and denominators
    across replicates averages out the day-to-day sampling noise that a single run's ratio has --
    which matters most exactly where each run's own counts are smallest (the start and the tail).
    Because every file's fracs are normalized by that file's own run-wide total, pooling weights
    each file equally regardless of its absolute outbreak size.

    `window`, if > 1, first applies a centered `window`-day moving average to each source's series
    (edge-padded, as in _smoothed_band) before ratioing, so the early/late days -- where a handful
    of infections can make the mix jump between 0 and 100% from one day to the next -- read as a
    trend rather than as noise. The ratio is taken after smoothing, so the bars still sum to 1.

    Days with no infections at all get all-zero fractions (an empty column in the plot), since
    there's no mix to report.

    Returns (keys, frac) where keys are the sources actually present and nonzero, in stacking
    order, and frac is a (len(keys) x n_days) array -- or None if this group has no source columns
    at all (i.e. an ExaEpi run without context_diag=true).
    """
    dfs = entry["dfs"]
    keys = [k for k in _SOURCE_STACK_ORDER if (k + "_frac") in dfs[0].columns]
    if not keys:
        return None

    n = min(min(len(df) for df in dfs), xlimit)
    totals = np.zeros((len(keys), n))
    for df in dfs:
        for i, key in enumerate(keys):
            totals[i] += df[key + "_frac"].values[:n]

    if window > 1:
        w = min(window, n)
        kernel = np.ones(w) / w
        pad = w // 2
        totals = np.vstack([
            np.convolve(np.pad(row, pad, mode="edge"), kernel, mode="valid")[:n] for row in totals
        ])

    day_total = totals.sum(axis=0)
    frac = np.where(day_total > 0, totals / np.where(day_total > 0, day_total, 1.0), 0.0)

    nonzero = [i for i in range(len(keys)) if frac[i].max() > 0]
    return [keys[i] for i in nonzero], frac[nonzero]


def plot_source_stack(ax, epicast_data, exaepi_data, model, title):
    """Stacked bar chart of each interaction context's share of that day's new infections, for one
    model (see _daily_source_composition). Every bar reaches 1; what changes over time is how it's
    subdivided, so the panel shows when transmission shifts between e.g. school/work and household.

    Only the first -e (model="epicast") or first -x (model="exaepi") group is shown, matching
    plot_single_source, and it's drawn at that group's own shift so it lines up day-for-day with
    the other panels.
    """
    ax.set_title(_SOURCE_STACK_TITLES[model])
    ax.set_xlabel("Days")
    # Shorter than "Fraction of day's infections": a rotated label is bounded by the axes HEIGHT,
    # and the figure legend along the bottom takes enough of it that the longer text is clipped.
    # "day's" is what the x-axis already says.
    ax.set_ylabel("Fraction of infections")
    ax.set_xlim([0, args.xlimit])
    if model == "epicast":
        data, day_col, shift = epicast_data, "day", epicast_shift
    else:
        data, day_col, shift = exaepi_data, "Day", (shift_by_group[0] if shift_by_group else 0.0)

    # The ticks stay at 0..1 so the reserved strip doesn't read as part of the scale.
    ax.set_ylim([0, 1 + _STACK_LABEL_HEADROOM])
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    print(title)

    composition = _daily_source_composition(data[0], args.xlimit, args.stack_window) if data else None
    if composition is None:
        # No -e/-x input for this model, or an ExaEpi run without context_diag=true: say so on the
        # panel rather than leaving a blank axes that looks like a plotting bug.
        ax.text(0.5, 0.5, "no per-source data", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_TICK, color="gray")
        print("  No per-source data")
        return

    entry = data[0]
    keys, frac = composition
    n = frac.shape[1]
    x = (entry["dfs"][0][day_col].values[:n] + shift)

    bottom = np.zeros(n)
    for key, y in zip(keys, frac):
        bar = ax.bar(x, y, bottom=bottom, width=1.0, linewidth=0, color=_SOURCE_STACK_COLORS[key],
                     label=_SOURCE_STACK_LABELS[key], zorder=2)
        bottom += y
        if key is not keys[-1]:
            # Thin surface-colored line along each band's top edge. Two jobs: it separates bands
            # that a reader with a color-vision deficiency would otherwise have to tell apart by
            # hue alone (the Cluster/Nbhd+Comm pair is close enough under protanopia to need it --
            # see _SOURCE_STACK_COLORS), and it makes the boundary itself, which is the thing the
            # panel is actually about, legible against the grid.
            ax.step(x, bottom, where="mid", color="white", linewidth=0.6, zorder=3)
        SOURCE_STACK_HANDLES.setdefault(_SOURCE_STACK_LABELS[key], bar)
        # The run-wide share of each source, i.e. what its slice would be if the whole run were a
        # single bar -- the one number the per-day picture doesn't show directly.
        overall = float(np.mean([df[key + "_frac"].values[:n].sum() for df in entry["dfs"]]))
        print(f"  {_SOURCE_STACK_LABELS[key]:20s} run-wide share: {overall:.3f}")

    # Same day-0 marker the other panels draw, but with its label held in the strip reserved above
    # the bars (_mark_exaepi_start isn't used here for exactly that reason -- it has no label_top).
    if shift:
        mark_label, mark_color = (("ExaEpi day 0", "red") if model == "exaepi"
                                  else ("Epicast day 0", "blue"))
        _mark_day_zero(ax, shift, mark_label, mark_color, _format_shift(shift),
                       label_top=1.0 / (1.0 + _STACK_LABEL_HEADROOM) - 0.02)

    # Same major+minor grid as every other panel here, with two differences forced by the bars:
    # it's drawn over them (set_axisbelow(False)) and in white rather than the default gray, since
    # under a solid stack it would be invisible and a gray line on saturated fills reads as dirt.
    # Without it the y-value of a boundary between two sources can't be read off the panel at all.
    ax.set_axisbelow(False)
    ax.grid(True, which="major", color="white", alpha=0.6, linewidth=AXES_LINEWIDTH)
    ax.grid(True, which="minor", color="white", alpha=0.2, linewidth=AXES_LINEWIDTH)
    ax.minorticks_on()




def plot_series(ax, epicast_data, exaepi_data, label, seir_dfs=None):
    """Plot time series data from multiple files.

    Both epicast_data and exaepi_data are lists of group dicts:
        {'label': str|None, 'is_wildcard': bool, 'dfs': [df, ...], 'fnames': [str, ...]}
    A wildcard group with N>1 files is rendered as its medoid file's curve with a
    semi-transparent band showing the smoothed per-day spread across files -- by default their
    min/max, or the central --band percent of them (see _smoothed_band).

    Args:
        label: the data series to plot (e.g., 'exposed', 'symptomatic')
        seir_dfs: optional list of (curve_index, resolved_params, df) from run_seir()/
            run_seirhd_erlang(), one per --seir_from_ini curve (see _resolve_seir_params)
    """
    epicast_colors = ["blue", "darkblue", "royalblue", "steelblue", "navy", "cornflowerblue"]
    exaepi_colors  = ["red", "darkred", "crimson", "firebrick", "maroon", "indianred"]
    # One shade of green per SEIRHD curve, sampled from a sequential colormap so any number of
    # curves stay visually distinguishable (a short fixed color list ran out of contrast for
    # more than a couple of curves).
#    seir_colors = ([plt.cm.Greens(x) for x in np.linspace(0.35, 0.85, len(seir_dfs))]
    #seir_colors = ([plt.cm.Greens(x) for x in np.linspace(0.55, 0.75, len(seir_dfs))]
    seir_colors = ([plt.cm.Greens(x) for x in np.linspace(0.55, 1.0, len(seir_dfs))]
                   if seir_dfs else [])

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

    # Second reference for SEIRHD curves specifically: the first ExaEpi group's curve, shifted
    # by that group's own shift, so SEIRHD overlays can be scored against both models.
    exaepi_reference_y = (
        _shift_array(_get_group_y(exaepi_data[0], exaepi_col, args.xlimit), shift_by_group[0], args.xlimit)
        if exaepi_data else None
    )

    def _plot_group(entry, i, base_colors, col, x_col=None, x_shift=0.0):
        """Plot one group entry; return (legend_label, auc, color, is_wildcard, y_for_gof)."""
        legend_label = entry["label"]
        is_wildcard  = entry["is_wildcard"]
        color        = base_colors[i % len(base_colors)]
        plot_label   = legend_label if legend_label is not None else "_nolegend_"

        if is_wildcard and len(entry["dfs"]) > 1:
            y_mat       = _align_arrays(entry["dfs"], col, args.xlimit)
            medoid_idx  = _medoid_index(y_mat)
            y_medoid    = y_mat[medoid_idx]
            medoid_lbl  = legend_label if legend_label is not None else f"group {i}"
            print(f"  Medoid file ({medoid_lbl}, {col}): {entry['fnames'][medoid_idx]}")
            peak_days = [_peak_day(row) for row in y_mat]
            print(f"  Peak-day range ({medoid_lbl}, {col}): "
                  f"[{min(peak_days)}, {max(peak_days)}]  std={np.std(peak_days):.1f}d")
            n        = y_mat.shape[1]
            x_vals = (entry["dfs"][0][x_col].values[:n] + x_shift) if x_col else np.arange(n)

            band_drawn = _draw_spread_bands(
                ax, x_vals, y_mat, color, plot_label if args.band_only else None)
            # --band_only leaves the band to speak for the group; with no band to draw (--band 0)
            # the line goes in regardless, so a group is never silently left off the plot.
            if not (args.band_only and band_drawn):
                ax.plot(x_vals, y_medoid, label=plot_label, color=color, linewidth=1, zorder=2)
            auc = float(np.sum(y_medoid))
            y_for_gof = _shift_array(y_medoid, x_shift, args.xlimit)
        else:
            df     = entry["dfs"][0]
            x_vals = (df[x_col] + x_shift) if x_col else np.arange(len(df[col]))
            y_vals = df[col]
            auc    = float(np.sum(y_vals[: args.xlimit]))
            ax.plot(x_vals, y_vals, label=plot_label, color=color, linewidth=1, zorder=2)
            y_for_gof = _shift_array(y_vals.values, x_shift, args.xlimit)

        return legend_label, auc, color, is_wildcard, y_for_gof

    print(f"{col_name}")

    # Plot each Epicast group, shifted right by epicast_shift (nonzero only when an auto shift
    # came out negative and got swapped onto Epicast instead of ExaEpi -- see --shift "auto").
    for i, entry in enumerate(epicast_data):
        lbl, auc, color, is_wc, y_for_gof = _plot_group(entry, i, epicast_colors, col_name,
                                              x_col="day", x_shift=epicast_shift)
        auc_lines.append((lbl, auc, color, is_wc, y_for_gof, i == 0, False))

    # Plot each ExaEpi group, each shifted by its OWN group-level shift (from matching that
    # group's exposed/NewI peak -- see _auto_shift_per_exaepi_group).
    for i, entry in enumerate(exaepi_data):
        lbl, auc, color, is_wc, y_for_gof = _plot_group(entry, i, exaepi_colors, exaepi_col,
                                              x_col="Day", x_shift=shift_by_group[i])
        auc_lines.append((lbl, auc, color, is_wc, y_for_gof, False, False))

    # Plot each SEIRHD curve (one per --seir_from_ini curve; see _resolve_seir_params)
    if seir_dfs and seir_col is not None:
        multi = len(seir_dfs) > 1
        for i, (idx, p, seir_df) in enumerate(seir_dfs):
            seir_y = seir_df[seir_col].values[: args.xlimit]
            color = seir_colors[i % len(seir_colors)]
            short_lbl = f"SEIRHD {idx}" if multi else "SEIRHD"
            ax.plot(
                np.arange(len(seir_y)), seir_y,
                label=f"{short_lbl} (β={p['beta']:.4g}, h={p['hosp_rate']:.4g}, "
                      f"μ={p['mu']:.4g})",
                color=color, linewidth=1.5, linestyle="-",
            )
            auc_lines.append((short_lbl, np.sum(seir_y), color, False,
                               _shift_array(seir_y, 0, args.xlimit), False, True))

    # Shortened from "Number of " + label (e.g. "Number of Cumulative Exposed") -- doesn't fit at
    # PLOS's 8-12pt font floor on this panel's now-much-smaller width; the plot's own title
    # already names the series, so the axis label doesn't need to restate it in full.
    ax.set_xlabel("Days")
    ax.set_ylabel(label)
    ax.set_xlim([0, args.xlimit])

    if epicast_shift:
        # The auto shift came out negative and got swapped onto Epicast (see --shift "auto"), so
        # ExaEpi sits unshifted at day 0 and the interesting origin to call out is Epicast's.
        _mark_day_zero(ax, epicast_shift, "Epicast day 0", epicast_colors[0],
                       _format_shift(epicast_shift))
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
    if seir_dfs and seir_col is not None:
        for _, _p, seir_df in seir_dfs:
            max_vals.append(seir_df[seir_col].values[: args.xlimit].max())
    if args.ylimit is not None:
        ax.set_ylim([0, args.ylimit])
    elif max_vals:
        ax.set_ylim([0, 1.1 * max(max_vals)])

    ax.set_title(label)
    ax.grid(True, which="major", linewidth=AXES_LINEWIDTH)
    ax.grid(True, which="minor", alpha=0.3, linewidth=AXES_LINEWIDTH)
    ax.minorticks_on()

    # Annotate per-series summary values in the upper-right corner (or lower-right for the
    # cumulative curve, since it rises into the upper-right area)
    if col_name == "cumulative_exposed":
        # Collect (label, text, color) in the same top-to-bottom order used by the other plots'
        # AUC block below (Epicast, then ExaEpi, then the SEIRHD curves). Unlabelled entries are
        # still printed (with a "(unlabelled)" fallback) but skip the on-plot text, matching the
        # other plots' behavior.
        text_entries = []

        def summary(max_val, pop):
            """What this curve's final cumulative count says, as the share of the population it
            reached, as (on-plot text, console text). A count on its own means nothing without the
            population behind it, and the two models' counts are not even on the same axis as a
            SEIRHD curve's, which is run at the population its own .ini names. Falls back to the raw count
            where there is no population to divide by (see --population).

            The console keeps the count alongside the rate; the panel does not, since it is a
            summary line on an already-busy plot and the curve itself shows the count."""
            if not pop:
                text = f"Max {max_val:,.0f}"
                return text, text
            rate = f"attack rate {100.0 * max_val / pop:.1f}%"
            return rate, f"{rate} ({max_val:,.0f})"

        for i, entry in enumerate(epicast_data):
            legend_label = entry["label"]
            color = epicast_colors[i % len(epicast_colors)]
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                y_mat = _align_arrays(entry["dfs"], col_name, args.xlimit)
                max_val = float(y_mat[_medoid_index(y_mat)].max())
            else:
                max_val = float(entry["dfs"][0][col_name][: args.xlimit].max())
            lbl_str = legend_label if legend_label is not None else "(unlabelled)"
            text, logged = summary(max_val, population)
            print(f"  {lbl_str}: {logged}")
            if legend_label is not None:
                text_entries.append((legend_label, text, color))
        for i, entry in enumerate(exaepi_data):
            legend_label = entry["label"]
            color = exaepi_colors[i % len(exaepi_colors)]
            if entry["is_wildcard"] and len(entry["dfs"]) > 1:
                y_mat = _align_arrays(entry["dfs"], exaepi_col, args.xlimit)
                max_val = float(y_mat[_medoid_index(y_mat)].max())
            else:
                max_val = float(entry["dfs"][0][exaepi_col][: args.xlimit].max())
            lbl_str = legend_label if legend_label is not None else "(unlabelled)"
            text, logged = summary(max_val, population)
            print(f"  {lbl_str}: {logged}")
            if legend_label is not None:
                text_entries.append((legend_label, text, color))
        if seir_dfs and seir_col is not None:
            multi = len(seir_dfs) > 1
            for i, (idx, p, seir_df) in enumerate(seir_dfs):
                max_val = float(seir_df[seir_col].values[: args.xlimit].max())
                short_lbl = f"SEIRHD {idx}" if multi else "SEIRHD"
                # Each curve is run at the N derived from its own .ini.
                text, logged = summary(max_val, float(p["N"]))
                print(f"  {short_lbl}: {logged}")
                text_entries.append((short_lbl, text, seir_colors[i % len(seir_colors)]))

        # This block anchors text to the bottom (va="bottom") and grows upward, so the first
        # entry placed ends up at the bottom -- draw in reverse to keep the on-plot top-to-bottom
        # reading order matching the other plots' top-anchored (va="top") AUC block above.
        for row, (lbl_str, text, color) in enumerate(reversed(text_entries)):
            ax.text(0.98, 0.03 + row * 0.09, f"{lbl_str} {text}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_TICK, color=color)
    else:
        row = 0
        for lbl, auc, color, is_wildcard, y_for_gof, is_reference, is_seir in auc_lines:
            lbl_str = lbl if lbl is not None else "(unlabelled)"
            gof_str = ""
            if not is_reference and reference_y is not None:
                gof = _goodness_of_fit(reference_y, y_for_gof)
                if gof is not None:
                    r2, nrmse = gof
                    r2_str    = f"{r2:.3f}"    if np.isfinite(r2)    else "N/A"
                    nrmse_str = f"{nrmse:.3f}" if np.isfinite(nrmse) else "N/A"
                    gof_str = f"  R²={r2_str}  NRMSE={nrmse_str}"
            print(f"  {lbl_str} AUC: {auc:,.0f}{gof_str}")
            if is_seir and exaepi_reference_y is not None:
                gof_x = _goodness_of_fit(exaepi_reference_y, y_for_gof)
                if gof_x is not None:
                    r2_x, nrmse_x = gof_x
                    r2_x_str    = f"{r2_x:.3f}"    if np.isfinite(r2_x)    else "N/A"
                    nrmse_x_str = f"{nrmse_x:.3f}" if np.isfinite(nrmse_x) else "N/A"
                    print(f"    vs ExaEpi:  R²={r2_x_str}  NRMSE={nrmse_x_str}")
            if lbl is not None:
                text = f"{lbl} AUC: {auc:,.0f}" if args.show_auc else lbl
                ax.text(0.98, 0.95 - row * 0.09, text,
                        transform=ax.transAxes, ha="right", va="top", fontsize=FONT_TICK, color=color)
                row += 1


parser = argparse.ArgumentParser(
    description="Compare ExaEpi and Epicast simulation outputs",
    epilog=(
        "File specifications can include optional labels using the format: 'filename:Label'. "
        "Both -e and -x can be repeated and accept glob patterns, "
        "e.g.: -e 'runs/*.bin:Epicast' -x 'runs/*.csv:ExaEpi'. "
        "When a pattern matches multiple files, the medoid file's curve is plotted with a "
        "semi-transparent smoothed spread band (min/max, or --band) in the same color as the line."
    ),
)
parser.add_argument(
    "--epicast_file", "-e",
    action="append", default=[], metavar="FILE[:LABEL]",
    help=(
        "Epicast binary events file or glob pattern, optionally with a label. Can be repeated. "
        f"A file ending in '{EPICAST_SUMMARY_SUFFIX}' is read as an already-extracted per-day "
        "summary (see extract_epicast_data.py) instead of a raw binary file."
    ),
)
parser.add_argument(
    "--exaepi_file", "-x",
    action="append", default=[], metavar="FILE[:LABEL]",
    help=(
        "ExaEpi csv file or glob pattern, optionally with a label. Can be repeated. "
        "Multiple matched files are shown as the medoid file's curve with a smoothed "
        "spread band (min/max, or --band)."
    ),
)
def _band_type(value):
    pct = float(value)
    if not 0.0 <= pct <= 100.0:
        raise argparse.ArgumentTypeError(f"--band must be between 0 and 100, got {value}")
    return pct


parser.add_argument(
    "--band_only", action="store_true", default=False,
    help="For a -e/-x pattern matching several files, draw only the spread band, leaving out the "
         "medoid line normally drawn through it. The band alone shows where the runs are without "
         "asserting that any one of them is the group's answer, which is the honest picture when "
         "the point being made is about the spread rather than about a representative run. The "
         "medoid is still what the printed AUC, goodness-of-fit and attack rate are computed "
         "from, and a group matching a single file is still drawn as its own line -- it has no "
         "band to stand in for it.",
)
parser.add_argument(
    "--band", type=_band_type, nargs="+", default=[_DEFAULT_BAND_COVERAGE], metavar="PCT",
    help="Percentage of runs each shaded band around a multi-file group covers, as a central "
         "percentile interval: 90 draws the 5th-95th percentile, 50 the quartiles, 0 no band at "
         "all. The default, 100, is the pointwise min/max, which shows worst-case extent but has "
         "each edge owned by a single run at every day. Give several to nest them into a fan, "
         "widest behind narrowest -- '--band 50 90' shows where the bulk of the runs are and how "
         "far the tails reach at once, which one band cannot (default: 100)",
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
    "--population", type=float, default=None, metavar="N",
    help="Total population, used to report the Cumulative Exposed panel's attack rate. Taken from "
         "the first -x input's own compartment counts when not given, which is exact for ExaEpi "
         "and, for every dataset checked, for the Epicast run beside it too; pass this where the "
         "two models were given different populations, or where there is no -x input to read one "
         "from (the panel falls back to reporting the cumulative count itself).",
)
parser.add_argument(
    "--show_auc", action="store_true", default=False,
    help="Show each series' AUC value in its on-plot label (e.g. 'Epicast AUC: 1,177,264'). Off "
         "by default, showing just the series name; AUC values are still printed to the console "
         "either way.",
)
MAX_SEIR_CURVES = 9
# Every flag belonging to the SEIRHD overlay carries this prefix, so --help groups them and
# nothing here can be confused with a flag about the plotted data. The table below names them
# without it -- the prefix goes on at registration, and _seir_flag puts it back when reading.
SEIR_FLAG_PREFIX = "seir_"
# SEIRHD curves are only ever plotted from an .ini: --seir_from_ini both turns the overlay on
# and says where its parameters come from, so there is no way to get a curve built out of
# stale defaults. Everything the .ini determines -- sigma, gamma, hosp_rate, gamma_h, mu, kE,
# kI and N -- is derived and has no flag. What is left here is the three things no disease
# parameter fixes, plus kD.
#
# Each is registered bare (the shared value, documented in --help) and with numbered siblings
# --seir_<name>1 .. --seir_<name>{MAX_SEIR_CURVES} (hidden from --help) that override it for
# one curve, so several curves can be compared in a single plot -- see _resolve_seir_params.
_SEIR_PARAM_ARGS = [
    ("from_ini",   str,   None,
     "plot a SEIRHD curve whose parameters are derived from this ExaEpi .ini: sigma, gamma, "
     "hosp_rate, gamma_h, mu, kE, kI and N, obtained by replaying ExaEpi's own per-agent "
     "disease lifecycle over the age composition of the UrbanPop population the .ini names "
     "(see seirhd_params.py). No simulation output is read. Without this flag no SEIRHD curve "
     "is plotted. To plot several, number the flags: --seir_from_ini2 for a second .ini, or "
     "e.g. --seir_rate_scale2 to vary one knob against the same .ini"),
    ("r0",         float, None,
     "R0 for the curve, which sets beta = R0 * (gamma + hosp_rate). beta is the one SEIRHD "
     "parameter that no disease parameter determines -- it depends on the contact network -- "
     "so it has to be fitted, and going through R0 keeps it consistent with the derived rates "
     "instead of silently changing R0 whenever they change. Required with --seir_from_ini"),
    ("seed",       int,   None,
     "initial infectious count. Not derivable either: it is a phase offset, since an ODE "
     "started from the .ini's handful of index cases cannot line up with a stochastic "
     "take-off. Required with --seir_from_ini"),
    ("rate_scale", float, 1.0,
     "multiply beta/sigma/gamma/hosp_rate/gamma_h/mu by this factor. R0 and every branch "
     "fraction are invariant under it, so it rescales time alone: >1 gives a taller, narrower "
     "peak at an earlier day, <1 a flatter, broader one, with the same final attack rate"),
    ("kD",         int,   1,
     "sub-stages in the hospitalisation-to-discharge chain (see run_seirhd_erlang). Left out "
     "of the derivation deliberately: it is the one shape that barely reaches the plotted "
     "curves, since admissions come from the I chain and carry no kD at all. 1 makes H a "
     "single exponential compartment"),
]
_SEIR_PARAM_DEFAULTS = {_name: _default for _name, _, _default, _ in _SEIR_PARAM_ARGS}
# Registered with default=None rather than the real default so _resolve_seir_params can tell
# "not given" from "given a value that happens to equal the default". The documented default
# is generated from the table instead of being restated in each help string, which is how the
# two came to disagree before, back when the parameters were all stated by hand.
for _name, _typ, _default, _help in _SEIR_PARAM_ARGS:
    _flag = f"--{SEIR_FLAG_PREFIX}{_name}"
    _suffix = "" if _default is None else f" (default: {_default})"
    parser.add_argument(_flag, type=_typ, default=None, help=_help + _suffix)
    for _i in range(1, MAX_SEIR_CURVES + 1):
        parser.add_argument(f"{_flag}{_i}", type=_typ, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--stack_window", type=int, default=1, metavar="DAYS",
    help="Smooth the 'Source Stack ...' panels with a centered moving average of this many days "
         "before taking each day's per-source shares. Only affects those panels. Use it when the "
         "start/tail of the run, where a handful of infections can swing the mix between 0 and "
         "100%% from one day to the next, drowns out the trend (default: 1, i.e. no smoothing)",
)
parser.add_argument(
    "--plots", "-p",
    nargs="+", metavar="PLOT", default=None,
    help=(
        "Which plots to show, in the order given. Rendered in 2-column layout. "
        "Valid names (case-insensitive): Exposed, Symptomatic, Presymptomatic, "
        "Asymptomatic, Hospitalized, Dead, Recovered, 'Cumulative Exposed', Context, "
        + ", ".join(f"'{n}'" for n in SOURCE_PLOT_NAMES + SOURCE_STACK_PLOT_NAMES) + ". "
        "'Source Fractions' is a legacy alias that expands to all of the per-context "
        "'Source: ...' plots, and 'Source Stack' expands to both 'Source Stack (...)' plots. "
        "Default: all 8 (or Exposed/Recovered/Cumulative Exposed when --seir_from_ini is used)."
    ),
)
args = parser.parse_args()

# The flags actually given, which is how the numbered per-curve overrides are spotted below.
_argv_flags = set()
for _tok in sys.argv[1:]:
    if _tok.startswith("--"):
        _argv_flags.add(_tok.lstrip("-").split("=")[0])

# Which SEIRHD curves to plot, and with what. A curve exists only where an .ini resolves for
# it -- curve 1 from the bare --seir_from_ini, curve N from --seir_from_ini{N} or from the bare
# one when some other numbered flag (say --seir_rate_scale3) references N. With no .ini
# anywhere, there are no curves.
_SEIR_PARAM_NAMES = [name for name, *_ in _SEIR_PARAM_ARGS]
_seir_curve_indices = {1}
for _flag in _argv_flags:
    for _name in _SEIR_PARAM_NAMES:
        _prefixed = SEIR_FLAG_PREFIX + _name
        if _flag.startswith(_prefixed) and _flag[len(_prefixed):].isdigit():
            _seir_curve_indices.add(int(_flag[len(_prefixed):]))

# Deriving costs a Monte Carlo over the agent lifecycle and, the first time, a pass over the
# UrbanPop .bin, so several curves sharing an .ini should only pay for it once.
_derived_seir_cache = {}


def _derive_seir_params(ini):
    """SEIRHD rates implied by an ExaEpi .ini (see seirhd_params.params_from_ini)."""
    if ini not in _derived_seir_cache:
        derivation = seirhd_params.params_from_ini(ini)
        print(f"Derived SEIRHD rates from {ini}")
        print(f"  age composition: {derivation.age_source}")
        print("  " + "  ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in derivation.rates.items()))
        _derived_seir_cache[ini] = derivation.rates
    return _derived_seir_cache[ini]


# beta and the rates it is scaled alongside. kE/kI/kD are shapes and N/seed are counts, none of
# which a --seir_rate_scale should touch -- scaling exactly these six is what leaves R0 and
# every branch fraction invariant.
_SEIR_RATE_NAMES = ("beta", "sigma", "gamma", "hosp_rate", "gamma_h", "mu")


def _seir_flag(idx, name):
    """The value of --seir_<name>{idx} if given, else the shared bare --seir_<name>. `name` is
    the table's unprefixed spelling, which is also the key the model uses for it."""
    prefixed = SEIR_FLAG_PREFIX + name
    numbered = getattr(args, f"{prefixed}{idx}")
    return numbered if numbered is not None else getattr(args, prefixed)


def _resolve_seir_params(idx):
    """Everything run_seirhd_erlang needs for curve `idx`: the rates and shapes derived from
    that curve's .ini, with --seir_r0 supplying beta and --seir_rate_scale applied to the
    result."""
    ini = _seir_flag(idx, "from_ini")
    resolved = dict(_derive_seir_params(ini))
    for name in ("seed", "kD", "rate_scale", "r0"):
        value = _seir_flag(idx, name)
        resolved[name] = _SEIR_PARAM_DEFAULTS[name] if value is None else value

    resolved["beta"] = resolved["r0"] * (resolved["gamma"] + resolved["hosp_rate"])
    for name in _SEIR_RATE_NAMES:
        resolved[name] *= resolved["rate_scale"]
    return resolved


_seir_curve_indices = sorted(i for i in _seir_curve_indices
                             if _seir_flag(i, "from_ini") is not None)
for _idx in _seir_curve_indices:
    for _required in ("r0", "seed"):
        if _seir_flag(_idx, _required) is None:
            parser.error(f"--seir_from_ini needs --{SEIR_FLAG_PREFIX}{_required} as well "
                         f"(curve {_idx}): it is not derivable from disease parameters, see "
                         f"its --help entry")
    # Every one of these has to be strictly positive for the model to mean anything, and
    # without the check a bad value surfaces far from its cause -- a --seir_rate_scale of 0
    # in particular zeroes every rate and only shows up as a ZeroDivisionError inside the ODE.
    for _positive in ("r0", "seed", "rate_scale", "kD"):
        _value = _seir_flag(_idx, _positive)
        if _value is not None and _value <= 0:
            parser.error(f"--{SEIR_FLAG_PREFIX}{_positive} must be greater than zero "
                         f"(curve {_idx}), got {_value}")

ALL_PLOTS = [
    "Exposed", "Symptomatic", "Presymptomatic", "Asymptomatic",
    "Hospitalized", "Dead", "Recovered", "Cumulative Exposed", "Context", *SOURCE_PLOT_NAMES,
    *SOURCE_STACK_PLOT_NAMES,
]
_plot_map: dict[str, str | list[str]] = {p.lower(): p for p in ALL_PLOTS}
_plot_map["source fractions"] = SOURCE_PLOT_NAMES  # legacy alias: expands to all context plots
_plot_map["source stack"] = SOURCE_STACK_PLOT_NAMES  # both models' stacked-composition panels

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

# Widest first so the narrower ones are drawn over them (see _draw_spread_bands), and a 0
# ("no band") among several is just nothing to draw rather than a contradiction.
args.band = sorted({b for b in args.band if b > 0}, reverse=True)

if args.band_only and not args.band:
    # Not an error: the two options are individually meaningful and the combination has an
    # obvious reading (draw neither), it is just not one worth producing an empty panel for.
    print("Note: --band_only with --band 0 would leave nothing to draw, so the medoid lines are "
          "kept. Raise --band to get bands instead of lines.")


def _load_and_capture(load_fn, fname):
    """Run load_fn(fname) with its stdout captured, returning (result, log) instead of printing
    directly. Used when load_fn runs in a worker process (see _load_grouped): each worker writes
    to the same underlying stdout fd, and since this script runs unbuffered (-u), concurrent
    writes from different processes interleave mid-line rather than one line at a time, garbling
    load_epicast/load_exaepi's diagnostic prints. Capturing lets the parent process print each
    file's log as a whole, in file order, once loading completes.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = load_fn(fname)
    return result, buf.getvalue()


_PEAK_RSS_PER_FILE_SIZE = 7.5  # see _safe_max_workers


def _safe_max_workers(fnames):
    """Cap the process pool's worker count so loading a wildcard match in parallel can't
    overcommit available RAM.

    A single load_epicast call (one worker, one file) peaks at roughly 4.4x a file's on-disk
    size in RSS in isolation (measured directly against the actual code path: a 2.47 GB CA
    events.bin file peaked at 10.8 GB while its minimal-columns events_df -- see
    read_events_bin's `full` parameter -- aggregate_events, and aggregate_infections_by_source
    were all live at once). But that isolated number understates real concurrent demand: a live
    run with 4 such workers (a 5.0x multiplier, i.e. only ~15% headroom over the 4.4x measurement)
    drove free system memory to near zero and pushed ~1.7 GB into swap before finishing -- it
    survived only because swap was there to absorb the gap, not because 4 workers actually fit
    comfortably. _PEAK_RSS_PER_FILE_SIZE=7.5 leaves real headroom instead of relying on swap as a
    safety net (swap isn't guaranteed to be configured, and even when it is, running from swap is
    much slower than the point of parallelizing in the first place).

    Running N workers needs roughly N times one worker's estimated peak, so with e.g. 10 such
    files and ~57 GB available, unrestricted parallelism (one worker per file, or per CPU)
    reliably OOMs; this sizes the pool down to however many can run at once within that estimate.

    load_exaepi's already-aggregated per-day CSVs are tiny, so this same on-disk-size-based
    estimate naturally comes out generous for them too -- no separate case is needed.
    """
    largest_file = max(os.path.getsize(f) for f in fnames)
    available = psutil.virtual_memory().available
    mem_limited = max(1, int(available // (largest_file * _PEAK_RSS_PER_FILE_SIZE)))
    return max(1, min(len(fnames), mp.cpu_count(), mem_limited))


def _load_grouped(file_specs, load_fn, extra_csv_fn=None):
    """Expand each file spec into a group dict, loading DataFrames with load_fn.

    A wildcard match (multiple files) loads them in parallel across processes: each file's
    read+aggregate is independent, and for Epicast's raw per-event binary files (millions of
    rows, multiple pandas groupbys per file in load_epicast) that's expensive enough that
    loading a large wildcard match sequentially can take minutes. ExaEpi's already-aggregated
    per-day CSVs are cheap enough that the parallelism is close to free either way. A single
    explicit file loads directly, with no process-pool startup overhead. The worker count is
    capped by _safe_max_workers so this parallelism can't overcommit available RAM (see its
    docstring) -- with large enough files/matches that can mean falling back to well below one
    worker per file, trading speed for not OOMing.

    Returns a list of {'label', 'is_wildcard', 'dfs', 'fnames'} dicts.
    """
    groups = []
    for file_spec in file_specs:
        expanded = expand_file_spec(file_spec)
        if not expanded:
            continue
        group_label = expanded[0][1]
        fnames = [fname for fname, _, _ in expanded]
        is_wc = len(fnames) > 1

        for fname in fnames:
            print(f"{fname}")
        if is_wc:
            max_workers = _safe_max_workers(fnames)
            if max_workers < len(fnames):
                print(f"  Loading with {max_workers} parallel worker(s) (capped to fit "
                      f"available memory; {len(fnames)} files matched)")
            with ProcessPoolExecutor(max_workers=max_workers,
                                      mp_context=mp.get_context("fork")) as executor:
                results = list(executor.map(functools.partial(_load_and_capture, load_fn), fnames))
            dfs = []
            for df, log in results:
                if log:
                    sys.stdout.write(log)
                dfs.append(df)
        else:
            dfs = [load_fn(fname) for fname in fnames]

        entry = {"label": group_label, "is_wildcard": is_wc, "dfs": dfs, "fnames": fnames}
        if extra_csv_fn:
            for fname, df in zip(fnames, dfs):
                extra_csv_fn(fname, df)
        groups.append(entry)
    return groups


def _write_epicast_csv(fname, df, shift):
    out = df.copy()
    out["day"] = out["day"] + shift
    csv_out = fname + "-plot_values.csv"
    out.to_csv(csv_out, index=False)


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

# Population behind the Cumulative Exposed panel's attack rate. SEIRHD curves carry their own N,
# derived from their .ini, so they are converted with that instead, not with this.
population = args.population
if population is None and exaepi_data:
    population = _exaepi_population(exaepi_data[0]["dfs"][0])

if args.xlimit == "auto":
    args.xlimit = _auto_xlimit(epicast_data, exaepi_data, shift_by_group, epicast_shift)
    print(f"Auto-detected xlimit: {args.xlimit} days (minimum furthest >=10 extent across all inputs)")

for _entry in epicast_data:
    for _fname, _df in zip(_entry["fnames"], _entry["dfs"]):
        _write_epicast_csv(_fname, _df, epicast_shift)

for _entry, _shift in zip(exaepi_data, shift_by_group):
    for _fname, _df in zip(_entry["fnames"], _entry["dfs"]):
        _write_exaepi_csv(_fname, _df, _shift)

# seir_dfs: list of (curve_index, resolved_params, df), one per --seir_from_ini curve
# (curve 1 plus any curve N referenced by a --<param>N override -- see _seir_curve_indices).
seir_dfs = []
for _idx in _seir_curve_indices:
    _p = _resolve_seir_params(_idx)
    if len(_seir_curve_indices) > 1:
        print(f"SEIRHD curve {_idx}:")
    _df = run_seirhd_erlang(_p["beta"], _p["sigma"], _p["gamma"], _p["hosp_rate"], _p["gamma_h"],
                             _p["mu"], _p["N"], _p["seed"], args.xlimit,
                             kE=_p["kE"], kI=_p["kI"], kD=_p["kD"])
    seir_dfs.append((_idx, _p, _df))

if args.plots is not None:
    selected_plots = args.plots
elif seir_dfs:
    selected_plots = ["Exposed", "Cumulative Exposed", "Hospitalized", "Dead", "Recovered"]
else:
    selected_plots = [p for p in ALL_PLOTS
                      if p != "Context" and p not in SOURCE_PLOT_NAMES
                      and p not in SOURCE_STACK_PLOT_NAMES]

n = len(selected_plots)
ncols = 1 if n == 1 else 2
nrows = (n + ncols - 1) // ncols
# Every individual panel keeps the same aspect ratio and footprint regardless of how many end up
# in the grid (unlike e.g. plot_geo.py, which shrinks each panel as more are added) -- a 2-column
# grid (the common case) spans the paper's full page width, a 1-column grid spans half; adding
# more panels only grows the number of ROWS, not each panel's own size. panel_height preserves
# this script's original 6x3.5 panel aspect ratio, just at the new PLOS scale.
panel_width = FULL_PAGE_WIDTH_IN / 2
panel_height = panel_width * (3.5 / 6)
fig, axes_grid = plt.subplots(
    nrows, ncols, figsize=(ncols * panel_width, nrows * panel_height), squeeze=False, layout="constrained"
)
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
    elif plot_name in _SOURCE_STACK_TO_MODEL:
        plot_source_stack(axes[i], epicast_data, exaepi_data, _SOURCE_STACK_TO_MODEL[plot_name],
                           plot_name)
    else:
        plot_series(axes[i], epicast_data, exaepi_data, plot_name, seir_dfs=seir_dfs)

for i in range(n, len(axes)):
    axes[i].set_visible(False)

if SOURCE_STACK_HANDLES:
    # One legend for every stacked panel, in one row under the whole figure. "outside lower center"
    # makes constrained_layout reserve space for it instead of overlaying the axes, so it costs the
    # panels nothing and stays put however many of them there are.
    fig.legend(SOURCE_STACK_HANDLES.values(), SOURCE_STACK_HANDLES.keys(),
               loc="outside lower center", ncols=len(SOURCE_STACK_HANDLES), frameon=False,
               handlelength=1.0, handletextpad=0.5, columnspacing=1.5)

# plt.suptitle("ExaEpi vs Epicast Comparison", y=1.05)
plt.savefig(args.output, dpi=300)
print(f"Wrote {args.output}")
#plt.show()
