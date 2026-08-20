#!/usr/bin/env -S python -u

"""Generate a metapopulation (multi-patch) SEIRHD model.

Extends the single-population SEIRHD model (see run_seir() in
compare_to_epicast.py) to n coupled patches. Disease parameters (beta, sigma,
gamma, hosp_rate, gamma_h, mu) are shared across all patches; heterogeneity
comes only from each patch's population, initial seeding, and the inter-patch
coupling strength. Patches mix via a row-stochastic matrix built from a single
--coupling parameter: each patch keeps (1 - coupling) of its contacts local
and spreads the remainder across other patches, according to --topology:
    all_to_all (default): spread evenly across every other patch.
    grid: patches are laid out on a --grid_shape ROWS COLS grid and spread
          only across their (up/down/left/right) grid neighbors.

Unlike compare_to_epicast.py's run_seir(), the output only carries the
distinct disease columns (exposed, hospitalized, dead, recovered,
cumulative_exposed) -- there's no separate symptomatic/presymptomatic/
asymptomatic split here, since those were pure aliases/zeros kept only to
match Epicast's column vocabulary for comparison plots.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ALL_PLOTS = ["Exposed", "Hospitalized", "Dead", "Recovered", "Cumulative Exposed"]
_plot_map = {p.lower(): p for p in ALL_PLOTS}


def build_all_to_all_coupling_matrix(n_patches, coupling):
    """Build a row-stochastic n x n all-to-all inter-patch mixing matrix.

    Diagonal entries are (1 - coupling); off-diagonal entries split the
    remaining `coupling` mass evenly across the other n_patches - 1 patches.
    With a single patch, coupling is meaningless and M is simply [[1.0]].
    """
    if n_patches == 1:
        return np.array([[1.0]])
    M = np.full((n_patches, n_patches), coupling / (n_patches - 1))
    np.fill_diagonal(M, 1.0 - coupling)
    return M


def build_grid_coupling_matrix(nrows, ncols, coupling):
    """Build a row-stochastic n x n grid (nearest-neighbor) mixing matrix.

    Patches are laid out on a nrows x ncols grid in row-major order (patch i
    is at row i // ncols, column i % ncols). Each patch keeps (1 - coupling)
    of its contacts local and splits the remaining `coupling` mass evenly
    across its up/down/left/right grid neighbors only (no wraparound, so
    edge and corner patches have fewer neighbors than interior ones). A
    patch with no neighbors (a 1x1 grid) keeps all its weight local.
    """
    n = nrows * ncols
    M = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, ncols)
        neighbors = []
        if r > 0:
            neighbors.append(i - ncols)
        if r < nrows - 1:
            neighbors.append(i + ncols)
        if c > 0:
            neighbors.append(i - 1)
        if c < ncols - 1:
            neighbors.append(i + 1)
        if neighbors:
            M[i, i] = 1.0 - coupling
            share = coupling / len(neighbors)
            for j in neighbors:
                M[i, j] = share
        else:
            M[i, i] = 1.0
    return M


def build_coupling_matrix(n_patches, coupling, topology="all_to_all", grid_shape=None):
    """Build a row-stochastic n x n inter-patch mixing matrix for the given topology.

    topology="all_to_all": see build_all_to_all_coupling_matrix().
    topology="grid": see build_grid_coupling_matrix(); grid_shape=(nrows, ncols)
                      is required and nrows * ncols must equal n_patches.
    """
    if topology == "grid":
        if grid_shape is None:
            raise ValueError("grid_shape is required when topology='grid'")
        nrows, ncols = grid_shape
        return build_grid_coupling_matrix(nrows, ncols, coupling)
    return build_all_to_all_coupling_matrix(n_patches, coupling)


def run_seirhd_metapop(beta, sigma, gamma, h, gamma_h, mu, pop, seed, coupling, days, patch_names,
                       topology="all_to_all", grid_shape=None):
    """Integrate the metapopulation SEIRHD ODEs and derive daily new-count series.

    Compartments per patch: S, E, I, H, R, D, with the same flows as the
    single-population SEIRHD model (S->E rate beta*I/N, E->I rate sigma,
    I->H rate h, I->R rate gamma, H->R rate gamma_h, H->D rate mu). Patches
    are coupled through the force of infection: patch i's exposure rate is
    beta * S_i * sum_j(M[i,j] * I_j / N_j), where M is the mixing matrix from
    build_coupling_matrix() and N_j is patch j's fixed initial population.

    Parameters
    ----------
    beta, sigma, gamma, h, gamma_h, mu : disease parameters, shared across patches
    pop        : array-like, length n, total population per patch
    seed       : array-like, length n, initial infectious count per patch
    coupling   : inter-patch mixing strength in [0, 1]
    days       : number of days to simulate
    patch_names: list of n patch labels
    topology   : "all_to_all" or "grid" -- see build_coupling_matrix()
    grid_shape : (nrows, ncols), required when topology="grid"

    Returns
    -------
    (patch_dfs, aggregate_df) : (list[pd.DataFrame], pd.DataFrame)
    """
    n = len(pop)
    pop = np.asarray(pop, dtype=float)
    seed = np.asarray(seed, dtype=float)
    M = build_coupling_matrix(n, coupling, topology=topology, grid_shape=grid_shape)

    def seirhd_metapop_odes(t, y):
        S, E, I, H, R, D = (y[k * n:(k + 1) * n] for k in range(6))
        force = beta * S * (M @ (I / pop))
        dS = -force
        dE = force - sigma * E
        dI = sigma * E - (gamma + h) * I
        dH = h * I - (gamma_h + mu) * H
        dR = gamma * I + gamma_h * H
        dD = mu * H
        return np.concatenate([dS, dE, dI, dH, dR, dD])

    y0 = np.zeros(6 * n)
    y0[0 * n:1 * n] = pop - seed
    y0[2 * n:3 * n] = seed

    t_eval = np.arange(0, days + 1, 1, dtype=float)
    sol = solve_ivp(seirhd_metapop_odes, [0, days], y0, t_eval=t_eval, method="RK45", max_step=0.1)

    patch_dfs = []
    for i in range(n):
        S = sol.y[0 * n + i]
        I = sol.y[2 * n + i]
        H = sol.y[3 * n + i]
        R = sol.y[4 * n + i]
        D = sol.y[5 * n + i]

        new_exposed = np.maximum(0, -np.diff(S))
        I_mid = 0.5 * (I[:-1] + I[1:])
        new_hospitalized = np.maximum(0, h * I_mid)
        new_recovered = np.maximum(0, np.diff(R))
        new_dead = np.maximum(0, np.diff(D))

        df = pd.DataFrame()
        df["day"] = np.arange(days)
        df["patch"] = patch_names[i]
        df["exposed"] = new_exposed
        df["hospitalized"] = new_hospitalized
        df["dead"] = new_dead
        df["recovered"] = new_recovered
        df["cumulative_exposed"] = new_exposed.cumsum()
        patch_dfs.append(df)

        peak_day = int(np.argmax(new_exposed))
        peak_val = new_exposed[peak_day]
        print(f"[{patch_names[i]}] SEIRHD  exposed={new_exposed.sum():.0f}  "
              f"hosp={new_hospitalized.sum():.0f}  dead={new_dead.sum():.0f}  "
              f"peak_exposed_day={peak_day}  peak_exposed={peak_val:.0f}")

    r0 = beta / (gamma + h)
    hfr = mu / (gamma_h + mu)
    ifr = (h / (gamma + h)) * hfr
    print(f"SEIRHD  R0={r0:.2f}  hosp_rate={h/(gamma+h):.4f}  HFR={hfr:.4f}  IFR={ifr:.4f}")

    aggregate_df = pd.DataFrame()
    aggregate_df["day"] = np.arange(days)
    for col in ("exposed", "hospitalized", "dead", "recovered"):
        aggregate_df[col] = sum(df[col].values for df in patch_dfs)
    aggregate_df["cumulative_exposed"] = aggregate_df["exposed"].cumsum()

    if n > 1:
        agg_exposed = aggregate_df["exposed"].to_numpy(dtype=float)
        agg_peak_day = int(np.argmax(agg_exposed))
        agg_peak_val = agg_exposed[agg_peak_day]
        print(f"[Aggregate]  exposed={aggregate_df['exposed'].sum():.0f}  "
              f"hosp={aggregate_df['hospitalized'].sum():.0f}  dead={aggregate_df['dead'].sum():.0f}  "
              f"peak_exposed_day={agg_peak_day}  peak_exposed={agg_peak_val:.0f}")

    return patch_dfs, aggregate_df


def plot_results(runs, patch_names, plots, per_patch, output, single_patch_df=None):
    """Render the selected series in a 2-column matplotlib grid and save to `output`.

    runs: list of (coupling_value, patch_dfs, aggregate_df) tuples, one per
    simulated --coupling value. With a single run, its aggregate is drawn in
    bold black (labeled "Aggregate" only if a legend is otherwise needed); with
    multiple runs, each aggregate is drawn in its own color and labeled by its
    coupling value so the runs can be compared directly.

    single_patch_df, if given, is the equivalent single well-mixed population
    (same total N and total seed as the metapopulation, coupling-independent)
    and is overlaid on every axis as a fixed black dotted reference line.
    """
    n_plots = len(plots)
    ncols = 1 if n_plots == 1 else 2
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, nrows * 3.5), squeeze=False)
    axes = axes_grid.flatten()

    cmap = plt.get_cmap("tab10")
    multi_coupling = len(runs) > 1
    show_legend = per_patch or single_patch_df is not None or multi_coupling

    for i, plot_name in enumerate(plots):
        ax = axes[i]
        col = plot_name.lower().replace(" ", "_")

        if multi_coupling:
            for j, (coupling, patch_dfs, aggregate_df) in enumerate(runs):
                color = cmap(j % 10)
                if per_patch:
                    for df in patch_dfs:
                        ax.plot(df["day"], df[col], color=color, linewidth=1,
                                linestyle="--", alpha=0.4, label="_nolegend_")
                ax.plot(aggregate_df["day"], aggregate_df[col], color=color,
                        linewidth=2.5, zorder=5, label=f"coupling={coupling:g}")
        else:
            _, patch_dfs, aggregate_df = runs[0]
            if per_patch:
                for j, df in enumerate(patch_dfs):
                    ax.plot(df["day"], df[col], color=cmap(j % 10), linewidth=1,
                            linestyle="--", alpha=0.7, label=patch_names[j])
            agg_label = "Aggregate" if show_legend else None
            ax.plot(aggregate_df["day"], aggregate_df[col], color="black",
                    linewidth=2.5, zorder=5, label=agg_label)

        if single_patch_df is not None:
            ax.plot(single_patch_df["day"], single_patch_df[col], color="black", linewidth=2,
                    linestyle=":", zorder=4, label="Single population (equiv.)")

        if show_legend:
            ax.legend(fontsize=7)

        ax.set_title(plot_name)
        ax.set_xlabel("Days")
        ax.set_ylabel("Number of " + plot_name)
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.3)
        ax.minorticks_on()

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")


def _resolve_per_patch_values(values, n, name, default, parser: argparse.ArgumentParser):
    """Resolve a --pop style argument: broadcast a single value, accept
    exactly n values, or fall back to `default` (a length-n list) if omitted."""
    if values is None:
        return list(default)
    if len(values) == 1:
        return list(values) * n
    if len(values) == n:
        return list(values)
    parser.error(f"--{name} must be given as 1 value (broadcast) or exactly --patches ({n}) values, "
                 f"got {len(values)}")


def _resolve_seed_spec(tokens, n, default, parser: argparse.ArgumentParser):
    """Resolve --seed tokens of the form 'PATCH:VALUE' or 'all:VALUE'.

    Tokens are applied in order, so 'all:VALUE' sets every patch's seed and a
    later 'PATCH:VALUE' can override individual patches (or vice versa).
    Falls back to `default` (a length-n list) if no tokens are given.
    """
    if tokens is None:
        return list(default)
    seed = [0] * n
    for tok in tokens:
        if ":" not in tok:
            parser.error(f"--seed values must be given as 'PATCH:VALUE' or 'all:VALUE', got '{tok}'")
        key, _, val_str = tok.partition(":")
        try:
            val = int(val_str)
        except ValueError:
            parser.error(f"--seed value for '{key}' must be an integer, got '{val_str}'")
        if key == "all":
            seed = [val] * n
        else:
            try:
                idx = int(key)
            except ValueError:
                parser.error(f"--seed patch key must be an integer patch index or 'all', got '{key}'")
            if not 0 <= idx < n:
                parser.error(f"--seed patch index {idx} out of range for --patches ({n})")
            seed[idx] = val
    return seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate a metapopulation (multi-patch) SEIRHD model",
        epilog=(
            "Disease parameters (--beta, --sigma, --gamma, --hosp_rate, --gamma_h, --mu) are shared "
            "across all patches; heterogeneity comes only from --pop, --seed, and --coupling. "
            "--pop accepts either a single value (broadcast to every patch) or exactly --patches values. "
            "--seed accepts one or more 'PATCH:VALUE' tokens (0-indexed) and/or a single 'all:VALUE' "
            "token, applied in order, e.g. '--seed all:100 0:5000' seeds every patch with 100 except "
            "patch 0, which gets 5000."
        ),
    )
    parser.add_argument("--patches", "-k", type=int, default=3, help="number of patches (default: 3)")
    parser.add_argument("--pop", type=int, nargs="+", default=None, metavar="POP",
                        help="total population per patch: 1 value (broadcast) or --patches values "
                             "(default: 2100000 split evenly across patches)")
    parser.add_argument("--seed", type=str, nargs="+", default=None, metavar="PATCH:VALUE",
                        help="initial infectious count per patch, given as one or more 'PATCH:VALUE' "
                             "tokens (0-indexed) and/or a single 'all:VALUE' token to seed every patch "
                             "the same, applied in order so later tokens override earlier ones for the "
                             "same patch, e.g. '--seed all:100 0:5000' (default: 12000 in patch 0 only, "
                             "0 elsewhere)")
    parser.add_argument("--patch_names", type=str, nargs="+", default=None, metavar="NAME",
                        help="labels for each patch, must be exactly --patches names "
                             "(default: patch_0, patch_1, ...)")
    parser.add_argument("--beta", type=float, default=0.54, help="SEIR transmission rate (default: 0.48)")
    parser.add_argument("--sigma", type=float, default=0.263, help="SEIR E->I progression rate (default: 0.263)")
    parser.add_argument("--gamma", type=float, default=0.152, help="SEIR I->R recovery rate (default: 0.152)")
    parser.add_argument("--hosp_rate", type=float, default=0.042, help="SEIRHD I->H hospitalisation rate (default: 0.042)")
    parser.add_argument("--gamma_h", type=float, default=0.162, help="SEIRHD H->R hospital recovery rate (default: 0.162)")
    parser.add_argument("--mu", type=float, default=0.017, help="SEIRHD H->D death rate (default: 0.017)")
    parser.add_argument("--coupling", type=float, nargs="+", default=[0.05], metavar="COUPLING",
                        help="inter-patch mixing strength(s) in [0, 1]: fraction of each patch's "
                             "contacts spread across its neighbors under --topology. Give multiple "
                             "values to simulate each one and plot them as separate lines (default: 0.05)")
    parser.add_argument("--topology", choices=["all_to_all", "grid"], default="all_to_all",
                        help="inter-patch coupling structure: 'all_to_all' spreads coupling evenly "
                             "across every other patch; 'grid' spreads it only across up/down/left/right "
                             "neighbors on a --grid_shape grid (default: all_to_all)")
    parser.add_argument("--grid_shape", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"),
                        help="grid dimensions when --topology grid; ROWS*COLS must equal --patches "
                             "(required when --topology grid)")
    parser.add_argument("--days", "-l", type=int, default=200, help="number of days to simulate (default: 200)")
    parser.add_argument("--output", "-o", required=True, help="output file name for the plot (e.g., seirhd_metapop.png)")
    parser.add_argument("--output_csv", default=None, metavar="PREFIX",
                        help="if given, write '{PREFIX}_{patch_name}.csv' per patch plus '{PREFIX}_aggregate.csv'")
    parser.add_argument("--plots", "-p", nargs="+", metavar="PLOT", default=None,
                        help=f"which plots to show, in the order given. Valid names (case-insensitive): "
                             f"{', '.join(ALL_PLOTS)}. Default: all {len(ALL_PLOTS)}")
    parser.add_argument("--per_patch", action="store_true", default=False,
                        help="overlay each patch's curve alongside the aggregate curve")
    args = parser.parse_args()

    if args.patches < 1:
        parser.error("--patches must be at least 1")
    for c in args.coupling:
        if not 0.0 <= c <= 1.0:
            parser.error(f"--coupling values must be in [0, 1], got {c}")

    n = args.patches

    grid_shape = None
    if args.topology == "grid":
        if args.grid_shape is None:
            parser.error("--grid_shape ROWS COLS is required when --topology grid")
        nrows, ncols = args.grid_shape
        if nrows * ncols != n:
            parser.error(f"--grid_shape rows*cols ({nrows}*{ncols}={nrows * ncols}) must equal "
                         f"--patches ({n})")
        grid_shape = (nrows, ncols)
    elif args.grid_shape is not None:
        parser.error("--grid_shape is only used with --topology grid")

    default_pop = [2_100_000 // n] * n
    pop = _resolve_per_patch_values(args.pop, n, "pop", default_pop, parser)

    default_seed = [12000] + [0] * (n - 1)
    seed = _resolve_seed_spec(args.seed, n, default_seed, parser)

    if args.patch_names is None:
        if grid_shape is not None:
            patch_names = [f"r{i // grid_shape[1]}c{i % grid_shape[1]}" for i in range(n)]
        else:
            patch_names = [f"patch_{i}" for i in range(n)]
    elif len(args.patch_names) == n:
        patch_names = args.patch_names
    else:
        parser.error(f"--patch_names must give exactly --patches ({n}) names, got {len(args.patch_names)}")

    if args.plots is not None:
        resolved = []
        for name in args.plots:
            canonical = _plot_map.get(name.lower())
            if canonical is None:
                parser.error(f"Unknown plot '{name}'. Valid names: {', '.join(ALL_PLOTS)}")
            resolved.append(canonical)
        plots = resolved
    else:
        plots = ALL_PLOTS

    multi_coupling = len(args.coupling) > 1
    runs = []
    for c in args.coupling:
        if multi_coupling:
            print()
            print(f"=== coupling={c:g} ===")
        patch_dfs, aggregate_df = run_seirhd_metapop(
            args.beta, args.sigma, args.gamma, args.hosp_rate, args.gamma_h, args.mu,
            pop, seed, c, args.days, patch_names,
            topology=args.topology, grid_shape=grid_shape,
        )
        runs.append((c, patch_dfs, aggregate_df))

    single_patch_df = None
    if n > 1:
        print()
        print("Equivalent single well-mixed population (same total N and total seed):")
        _, single_patch_df = run_seirhd_metapop(
            args.beta, args.sigma, args.gamma, args.hosp_rate, args.gamma_h, args.mu,
            [sum(pop)], [sum(seed)], 0.0, args.days, ["single_population"],
        )

    if args.output_csv:
        for c, patch_dfs, aggregate_df in runs:
            prefix = f"{args.output_csv}_coupling{c:g}" if multi_coupling else args.output_csv
            for df in patch_dfs:
                df.to_csv(f"{prefix}_{df['patch'].iloc[0]}.csv", index=False)
            aggregate_df.to_csv(f"{prefix}_aggregate.csv", index=False)
        if single_patch_df is not None:
            single_patch_df.to_csv(f"{args.output_csv}_single_population.csv", index=False)

    plot_results(runs, patch_names, plots, args.per_patch, args.output,
                 single_patch_df=single_patch_df)


if __name__ == "__main__":
    main()
