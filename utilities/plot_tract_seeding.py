#!/usr/bin/env -S python -u

"""Show whether one model's faster statewide growth comes from faster growth between tracts
(seeding new ones) or within them, for ExaEpi vs Epicast. Writes two half-page figures with the
same x axis, statewide cumulative infections, so that they read side by side and the day shift
between the models drops out:

  <prefix>-tract-seeding.png       growth between tracts: the number of tracts with an infected
                                   resident, whose first infections can only have come from outside
  <prefix>-tract-local-growth.png  growth within tracts: R_t of the tracts with >= 10 residents
                                   infected so far, estimated as in compare_to_epicast.py's
                                   reproduction-number plot (renewal equation with the SEIRHD
                                   generation interval from --ini), from their infections in home
                                   settings only (see below), with a bootstrap band over the tracts

Epicast is drawn as a median line and range band over the given runs, ExaEpi as one line per run.

The within-tract R_t counts only infections in home settings -- household, household cluster and
the nighttime neighborhood and community -- which happen in the infected agent's home community,
and so can't have come from an infector living in another tract. Infections at work, at school and
in the daytime neighborhood and community can, so leaving them in would let faster spread between
tracts show up as faster growth within them. Epicast records every infection's context; ExaEpi
writes its per-block-group context columns (EHH, ENC, ENbhN, ECommN, ...) to the case files only
when run with agent.context_diag = 1, and they are expected rather than counted infections. Pass
--local_infections all to count every infection instead, as for runs without those columns.

Both models' infections are placed at the infected agent's HOME block group, summed to tracts:
ExaEpi's per-block-group case files (cases00000, cases00001, ... one per day) already count
residents, and Epicast agents are mapped to their UrbanPop home block group with
urbanpop_agent_index (Epicast's own event location is the agent's location at that timestep, which
is the daytime community for daytime infections). The UrbanPop .bin must be the one both runs used.
"""

import argparse
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import apply_style, PAGE_WIDTHS_IN, FONT_TICK  # noqa: E402
from read_epicast_events import read_events_records, urbanpop_agent_index  # noqa: E402
from plot_commute_distance import read_urbanpop_columns  # noqa: E402
import seirhd_params  # noqa: E402

EXPOSED = 0x01  # Epicast disease_state code for an infection event
# Epicast contexts (read_epicast_events._CONTEXT_MAP codes) of home-setting infections: household
# and household cluster, plus neighborhood/community at night (even timesteps), when everyone is
# in their home community. Epicast doesn't split neighborhood/community into day and night itself.
EPICAST_HOUSEHOLD, EPICAST_CLUSTER, EPICAST_NBHD_COMM = 0x00, 0x06, 0x0A
# ExaEpi's per-block-group case-file columns of the same settings (agent.context_diag)
EXAEPI_HOME_COLUMNS = ["EHH", "ENC", "ENbhN", "ECommN"]
BAND_ALPHA = 0.25
# Days of the centered moving average before estimating R_t: compare_to_epicast.py's default
# --share_window, so the tract R_t matches the statewide one in its reproduction-number plot.
RT_WINDOW = 7
# Statewide cumulative infections at which the console reports both plots' values.
PRINT_COUNTS = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6)
# The Epicast/ExaEpi colors every comparison script in this repo uses.
SERIES_COLORS = {"Epicast": "blue", "ExaEpi": "red"}


def tract_index(home_geoid):
    """Sorted unique tracts (11-digit GEOIDs) and their resident counts, from home block groups."""
    return np.unique(np.asarray(home_geoid, dtype=np.int64) // 10, return_counts=True)


def daily_by_tract(tract_of_event, day, tracts, ndays):
    """(len(tracts), ndays) matrix of new infections per tract per day."""
    ti = pd.Index(tracts).get_indexer(tract_of_event)
    ok = (ti >= 0) & (day >= 0) & (day < ndays)
    m = np.zeros((len(tracts), ndays))
    np.add.at(m, (ti[ok], day[ok]), 1)
    return m


def load_epicast(fname, home_geoid, tracts, ndays):
    """New infections per tract per day, all of them and those in home settings."""
    print("Reading Epicast events from", fname)
    rec, demog = read_events_records(fname)
    ex = rec[rec["disease_state"] == EXPOSED]
    agent = urbanpop_agent_index(ex["agent_id"], demog, home_geoid)
    tract = home_geoid[agent] // 10
    timestep = ex["timestep"].astype(np.int64)
    day = timestep // 2  # two 12-hour timesteps per day
    ctx = ex["context"]
    home = ((ctx == EPICAST_HOUSEHOLD) | (ctx == EPICAST_CLUSTER) | ((ctx == EPICAST_NBHD_COMM) & (timestep % 2 == 0)))
    return daily_by_tract(tract, day, tracts, ndays), daily_by_tract(tract[home], day[home], tracts, ndays)


def _day_from_filename(path):
    m = re.search(r"(\d+)$", os.path.basename(path))
    if not m:
        sys.exit(f"error: can't get a day number from ExaEpi case file name {path}")
    return int(m.group(1))


def load_exaepi(case_files, tracts, ndays):
    """New infections per tract per day from one run's cumulative per-block-group case files, and
    the expected new infections in home settings from their context columns, or None if the run
    didn't write them (agent.context_diag off). A file's context columns are the infections of the
    previous day's interactions, the same ones its counts are the first to include, so they line up
    with the differences of the counts."""
    print(f"Reading {len(case_files)} ExaEpi case files, {case_files[0]} ...")
    cum = np.zeros((len(tracts), ndays))
    home = np.zeros((len(tracts), ndays))
    has_home = True
    ti = pd.Index(tracts)
    for f in case_files:
        day = _day_from_filename(f)
        if day >= ndays:
            continue
        df = pd.read_csv(f)
        tract = df.GEOID.values // 10
        by_tract = (df.total - df.never_infected).groupby(tract).sum()
        cum[ti.get_indexer(by_tract.index), day] = by_tract.values
        if has_home and set(EXAEPI_HOME_COLUMNS) <= set(df.columns):
            by_tract = df[EXAEPI_HOME_COLUMNS].sum(axis=1).groupby(tract).sum()
            home[ti.get_indexer(by_tract.index), day] = by_tract.values
        else:
            has_home = False
    # carry the last count forward past a run's final file
    last = max(_day_from_filename(f) for f in case_files)
    cum[:, last + 1:] = cum[:, [min(last, ndays - 1)]]
    return np.diff(cum, axis=1, prepend=0), (home if has_home else None)


def tracts_reached(m, grid, min_infected):
    """Tracts with >= min_infected infected residents, on the first day the statewide cumulative
    count reaches each value in grid."""
    cum_t = np.cumsum(m, axis=1)
    cum_s = cum_t.sum(axis=0)
    reached = (cum_t >= min_infected).sum(axis=0)
    days = np.searchsorted(cum_s, grid)
    out = np.full(len(grid), np.nan)
    ok = days < len(cum_s)
    out[ok] = reached[days[ok]]
    return out


def smooth(m, window=RT_WINDOW):
    """Centered `window`-day moving average of each row, padding with the end values, as
    compare_to_epicast.py's _reproduction_number does."""
    pad = window // 2
    c = np.cumsum(np.pad(m, ((0, 0), (pad, pad)), mode="edge"), axis=1)
    c = np.concatenate([np.zeros((len(m), 1)), c], axis=1)
    return (c[:, window:] - c[:, :-window]) / window


def established_rt(m, tpop, g, grid, args, rng=None, local=None):
    """R_t of the tracts with an established outbreak (>= --established_infections residents infected
    so far), on the first day the statewide cumulative count reaches each value in grid.

    It is compare_to_epicast.py's renewal-equation estimate, R_t = i_t / sum_k g_k i_{t-k}, applied
    to those tracts together: their new infections over their infectious pressure, which is the
    pressure-weighted mean of the tracts' own R_t. That is steadier than a plain mean or median of
    per-tract ratios of a few infections a day, and over all tracts it is exactly the statewide R_t.
    Days with fewer than --min_tracts such tracts are NaN.

    Given local, a matrix like m of only the infections that can't have come from outside the
    tract (those in home settings), i_t is those: the infectious pressure, and which tracts are
    established, still come from all of m, since infected residents infect in every setting.

    Given rng, also returns --bootstrap resamples of it, (resamples, len(grid)), over the tracts:
    each draws every tract a Poisson(1) number of times (the Poisson bootstrap, which resamples all
    days at once instead of redrawing the established tracts day by day). Their spread is the
    uncertainty that comes from how few tracts are established, which is large early on."""
    x = smooth(m)
    pressure = np.zeros_like(x)
    for k in range(1, min(len(g), x.shape[1])):
        pressure[:, k:] += g[k] * x[:, :-k]
    cum = np.cumsum(m, axis=1)
    est = (cum >= args.established_infections) & (tpop >= args.min_tract_pop)[:, None]
    x_est = np.where(est, x if local is None else smooth(local), 0)
    p_est = np.where(est, pressure, 0)
    enough = est.sum(axis=0) >= args.min_tracts

    def ratio(new, pres):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(enough & (pres > 0), new / pres, np.nan)

    days = np.searchsorted(cum.sum(axis=0), grid)
    ok = days < len(enough)

    def on_grid(rt):
        out = np.full(rt.shape[:-1] + (len(grid),), np.nan)
        out[..., ok] = rt[..., days[ok]]
        return out

    rt = on_grid(ratio(x_est.sum(axis=0), p_est.sum(axis=0)))
    if rng is None:
        return rt
    w = rng.poisson(1.0, (args.bootstrap, len(m))).astype(float)
    return rt, on_grid(ratio(w @ x_est, w @ p_est))


def style_axes(ax):
    """Same grid and full axes box as plot_group_size_histogram.py's figures."""
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def statewide_grid(tpop, args):
    """The statewide cumulative-infection grid both plots share, so they line up side by side."""
    if args.xlim:
        xmin, xmax = args.xlim
    else:
        # from 1,000 up to the decade above a quarter of the population, well past the growth phase
        xmin, xmax = 1e3, 10 ** np.ceil(np.log10(0.25 * tpop.sum()))
    return np.logspace(np.log10(xmin), np.log10(xmax), 200)


def plot_vs_statewide(ax, grid, epi, exa, tpop, args, bands=None):
    """Epicast's median and range band over its runs, and one line per ExaEpi run, against
    statewide cumulative infections; epi and exa are (runs, len(grid)) arrays. bands, if given,
    maps a model name to the (lo, hi) band to draw for it instead of Epicast's range over runs."""
    pop = tpop.sum()
    style_axes(ax)
    if args.fit_window:
        lo, hi = args.fit_window
        ax.axvspan(lo * pop, hi * pop, color="0.93", zorder=0, linewidth=0)
        ax.text(np.sqrt(lo * hi) * pop, 0.97, "growth-rate\nfit window", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FONT_TICK, color="0.4")
    for name, (lo, hi) in (bands or {}).items():
        ax.fill_between(grid, lo, hi, facecolor=SERIES_COLORS[name], alpha=BAND_ALPHA, edgecolor="none", zorder=1)
    if len(epi):
        with warnings.catch_warnings():
            # grid points past a run's last day are NaN in every run
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(epi) > 1 and bands is None:
                ax.fill_between(grid, np.nanmin(epi, axis=0), np.nanmax(epi, axis=0),
                                facecolor=SERIES_COLORS["Epicast"], alpha=BAND_ALPHA, edgecolor="none", zorder=1)
            ax.plot(grid, np.nanmedian(epi, axis=0), color=SERIES_COLORS["Epicast"], linewidth=1, label="Epicast",
                    zorder=2)
    for i, e in enumerate(exa):
        ax.plot(grid, e, color=SERIES_COLORS["ExaEpi"], linewidth=1, label="ExaEpi" if i == 0 else None, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel("Statewide cumulative infections")


def print_at(label, fn, epi_runs, exa_runs, fmt):
    """fn(run, counts) at a few statewide cumulative counts, exactly rather than off the plotting
    grid (and so independent of --xlim)."""
    print(f"  {label}:")
    for c in PRINT_COUNTS:
        msg = f"    {c:9,.0f} statewide infections:"
        if epi_runs:
            e = np.array([fn(m, [c])[0] for m in epi_runs])
            if np.isfinite(e).any():
                msg += (f" Epicast median {np.nanmedian(e):{fmt}} [{np.nanmin(e):{fmt}}, "
                        f"{np.nanmax(e):{fmt}}]")
        for m in exa_runs:
            msg += f" ExaEpi {fn(m, [c])[0]:{fmt}}"
        print(msg)


def plot_seeding(epi_runs, exa_runs, tpop, args, out):
    """Tracts reached, against statewide cumulative infections: growth BETWEEN tracts. A tract's
    first infections can only come from outside it (apart from the initial seeds), so with the
    default threshold of one infected resident this counts tracts seeded from elsewhere, before
    any growth within them."""
    grid = statewide_grid(tpop, args)
    n = args.reached_infections
    epi = np.array([tracts_reached(m, grid, n) for m in epi_runs])
    exa = np.array([tracts_reached(m, grid, n) for m in exa_runs])
    fig, ax = plt.subplots(figsize=PAGE_WIDTHS_IN[args.width], layout="constrained")
    plot_vs_statewide(ax, grid, epi, exa, tpop, args)
    if args.logy:
        ax.set_yscale("log")
        # start just below the smallest nonzero count in view (0 tracts can't be drawn on a log axis)
        vals = np.concatenate([epi.ravel(), exa.ravel()])
        vals = vals[vals > 0]
        ax.set_ylim(max(1.0, 0.7 * vals.min()) if len(vals) else 1.0, None)
        # the counts span under two decades, so label 1-2-5 steps as plain numbers rather than
        # leaving matplotlib to label every minor tick (2x10^2, 3x10^2, ...) or none
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, None)
    ax.set_ylabel("Tracts with an infected resident" if n == 1 else f"Tracts with $\\geq${n} infected residents")
    # the curves rise from the lower left and flatten at the top, leaving the lower right empty
    ax.legend(loc="lower right")
    fig.savefig(out)
    print("Wrote", out)
    print_at("tracts reached", lambda m, c: tracts_reached(m, c, n), epi_runs, exa_runs, ".0f")


def plot_local_rt(epi_runs, exa_runs, epi_local, exa_local, tpop, g, args, out):
    """R_t of tracts with an established outbreak, against statewide cumulative infections: growth
    WITHIN tracts. epi_local and exa_local hold each run's home-setting infections (see
    established_rt), or Nones to count all of a tract's infections -- which include those from
    infectors living elsewhere (at work, say), so that spread into established tracts from outside
    raises it too."""
    grid = statewide_grid(tpop, args)
    home = args.local_infections == "home"
    label = f"{'home-setting ' if home else ''}R_t of tracts >= {args.established_infections} infected"
    epi_pairs, exa_pairs = list(zip(epi_runs, epi_local)), list(zip(exa_runs, exa_local))
    if not args.bootstrap:
        epi = np.array([established_rt(m, tpop, g, grid, args, local=h) for m, h in epi_pairs])
        exa = np.array([established_rt(m, tpop, g, grid, args, local=h) for m, h in exa_pairs])
        bands = None
    else:
        # The band is the central --band percent of the bootstrap resamples, pooled over each model's
        # runs (so with several runs it takes in the run-to-run spread as well), drawn for both models.
        rng = np.random.default_rng(0)
        q = [50 - args.band / 2, 50 + args.band / 2]
        # the plotting grid and then the console's counts, evaluated together
        both = np.concatenate([grid, PRINT_COUNTS])
        lines, bands, at = {}, {}, {}
        for name, pairs in (("Epicast", epi_pairs), ("ExaEpi", exa_pairs)):
            if not pairs:
                continue
            res = [established_rt(m, tpop, g, both, args, rng, local=h) for m, h in pairs]
            rt = np.array([r for r, _ in res])
            with warnings.catch_warnings():
                # points without enough established tracts are NaN in every run and resample
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanpercentile(np.concatenate([b for _, b in res]), q, axis=0)
                mid = np.nanmedian(rt, axis=0)
            lines[name], bands[name] = rt[:, :len(grid)], (lo[:len(grid)], hi[:len(grid)])
            at[name] = list(zip(mid[len(grid):], lo[len(grid):], hi[len(grid):]))
        epi, exa = lines.get("Epicast", np.empty((0, len(grid)))), lines.get("ExaEpi", np.empty((0, len(grid))))
    fig, ax = plt.subplots(figsize=PAGE_WIDTHS_IN[args.width], layout="constrained")
    plot_vs_statewide(ax, grid, epi, exa, tpop, args, bands)
    ax.set_ylabel(f"{'Home-setting ' if home else ''}$R_t$, tracts $\\geq${args.established_infections} infected")
    # below the hump and clear of the wide early bands at the left
    ax.legend(loc="lower center")
    fig.savefig(out)
    print("Wrote", out)
    if not args.bootstrap:
        # print_at passes the run's all-infection matrix; find its local one by identity
        local_of = {id(m): h for m, h in epi_pairs + exa_pairs}
        print_at(label, lambda m, c: established_rt(m, tpop, g, c, args, local=local_of[id(m)]), epi_runs, exa_runs,
                 ".2f")
        return
    print(f"  {label} (median over runs, central {args.band:g}% of {args.bootstrap} bootstrap resamples):")
    for i, c in enumerate(PRINT_COUNTS):
        print(f"    {c:9,.0f} statewide infections:"
              + "".join(f" {name} {v[i][0]:.2f} [{v[i][1]:.2f}, {v[i][2]:.2f}]" for name, v in at.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urbanpop", "-u", required=True,
                        help="UrbanPop .bin both runs were built from (gives every agent's home block group)")
    parser.add_argument("--events_files", "-e", nargs="+", default=[],
                        help="Epicast run.events.bin files, one per run")
    parser.add_argument("--exaepi_files", "-g", nargs="+", default=[],
                        help="ExaEpi per-block-group case files (cases00000 ...) of ONE run; for "
                        "several runs use --exaepi_run_dirs")
    parser.add_argument("--exaepi_run_dirs", nargs="+", default=[],
                        help="Directories each holding one ExaEpi run's cases????? files, for several runs")
    parser.add_argument("--days", type=int, default=250, help="Days to read from each run (default: 250)")
    parser.add_argument("--ini", required=True,
                        help="ExaEpi .ini the runs used, for the SEIRHD generation interval of the R_t "
                        "estimate (as compare_to_epicast.py's --seir_from_ini)")
    parser.add_argument("--reached_infections", type=int, default=1,
                        help="On the tract-seeding plot, a tract counts as reached once this many of its "
                        "residents have been infected (default: 1, so that only infections from "
                        "outside the tract can have reached it)")
    parser.add_argument("--established_infections", type=int, default=10,
                        help="On the local-growth plot, R_t is of the tracts with at least this many "
                        "residents infected so far (default: 10, enough for about 20 tracts to qualify "
                        "by 1,000 statewide infections, where the tract-seeding plot starts; a "
                        "fraction such as 1%% of residents leaves too few until ten times that)")
    parser.add_argument("--min_tract_pop", type=int, default=1000,
                        help="Tracts with fewer residents are left out of the local-growth plot "
                        "(default: 1000)")
    parser.add_argument("--local_infections", choices=["home", "all"], default="home",
                        help="Infections the local-growth plot's R_t counts: those in home settings, "
                        "which can't have come from another tract (needs ExaEpi runs made with "
                        "agent.context_diag = 1), or all of them (default: home)")
    parser.add_argument("--min_tracts", type=int, default=20,
                        help="Days with fewer established tracts are left off the local-growth plot, "
                        "since their R_t is noise (default: 20)")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Resamples of the tracts for the uncertainty band on the local-growth plot; "
                        "0 for no band (default: 200)")
    parser.add_argument("--band", type=float, default=90,
                        help="Central percentage of the bootstrap resamples the band covers (default: 90)")
    parser.add_argument("--fit_window", type=float, nargs=2, default=[0.0005, 0.01], metavar=("LO", "HI"),
                        help="Shade the statewide cumulative-infection window (as fractions of the "
                        "population) used to fit early growth rates; pass 0 0 to omit (default: 0.0005 0.01)")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Statewide cumulative-infection range of both plots (default: 1000 up to "
                        "the decade above a quarter of the population)")
    parser.add_argument("--logy", action="store_true",
                        help="Log y axis (tracts reached) on the tract-seeding plot")
    parser.add_argument("--width", choices=list(PAGE_WIDTHS_IN), default="half",
                        help="Figure size for the slot it is placed in in the paper (see plos_compbio_style.py)")
    parser.add_argument("--output", "-o", default="tract",
                        help="Output prefix: writes <prefix>-tract-seeding.png and "
                        "<prefix>-tract-local-growth.png (default: tract)")
    args = parser.parse_args()
    if not args.events_files and not args.exaepi_files and not args.exaepi_run_dirs:
        parser.error("give at least one of --events_files, --exaepi_files, --exaepi_run_dirs")
    if args.fit_window == [0.0, 0.0]:
        args.fit_window = None
    if args.xlim and not 0 < args.xlim[0] < args.xlim[1]:
        parser.error("--xlim needs 0 < MIN < MAX (the x axis is logarithmic)")

    apply_style()

    home = read_urbanpop_columns(args.urbanpop, ["home_geoid"])["home_geoid"].astype(np.int64)
    tracts, tpop = tract_index(home)
    tpop = tpop.astype(float)
    epi_runs, epi_home = map(list, zip(*[load_epicast(f, home, tracts, args.days) for f in args.events_files])) \
        if args.events_files else ([], [])
    exa_sets = [sorted(args.exaepi_files)] if args.exaepi_files else []
    for d in args.exaepi_run_dirs:
        files = sorted(f for f in os.listdir(d) if re.fullmatch(r"cases\d+", f))
        exa_sets.append([os.path.join(d, f) for f in files])
    exa_runs, exa_home = map(list, zip(*[load_exaepi(files, tracts, args.days) for files in exa_sets])) \
        if exa_sets else ([], [])
    print(f"{len(epi_runs)} Epicast run(s), {len(exa_runs)} ExaEpi run(s)")
    if args.local_infections == "home":
        # Epicast's events always carry their context; only ExaEpi's case files can lack it
        missing = [str(i) for i, h in enumerate(exa_home) if h is None]
        if missing:
            sys.exit(f"error: ExaEpi run(s) {', '.join(missing)} have no per-context columns in their case files, "
                     "so no home-setting infection counts: rerun with agent.context_diag = 1, or pass "
                     "--local_infections all")
    else:
        epi_home, exa_home = [None] * len(epi_runs), [None] * len(exa_runs)
    g = seirhd_params.generation_interval(seirhd_params.params_from_ini(args.ini, urbanpop=args.urbanpop).rates)
    print(f"Generation interval mean {np.dot(np.arange(len(g)), g):.2f} days")

    plot_seeding(epi_runs, exa_runs, tpop, args, f"{args.output}-tract-seeding.png")
    plot_local_rt(epi_runs, exa_runs, epi_home, exa_home, tpop, g, args, f"{args.output}-tract-local-growth.png")


if __name__ == "__main__":
    main()
