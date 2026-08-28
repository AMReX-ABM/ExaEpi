#!/usr/bin/env -S python -u

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, pearsonr
from scipy.optimize import minimize, differential_evolution
import sys

plt.rcParams.update({"font.size": 20})

# latent period
exposed_to_presymp = [0.0] * 1
exposed_to_presymp.extend([0.1, 0.25, 0.55, 0.66, 0.78, 0.85, 1.0])
# incubation period
incubation = [0.0] * 1
incubation.extend(exposed_to_presymp)
# infectious period - in epicast runs from 3 to 9 days, excluding the presymp infectious period, so this needs to start at 4
infectious = [0.0] * 4
infectious.extend([0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])

transitions = [
    #(exposed_to_presymp, 2.82, 1.36, 0.0, "Exposed to Presymptomatic"),
    (exposed_to_presymp, 2.77, 1.5, 0.0, "Exposed to Presymptomatic"),
    #(exposed_to_presymp, 1.95, 2.53, 0.0, "Exposed to Presymptomatic"),
    #(exposed_to_presymp, 6, 0.73, 0, "Exposed to Presymptomatic"),
    #(exposed_to_presymp, 2.55, 1.34, 0.0, "Exposed to Presymptomatic"),
    #(exposed_to_presymp, 1.5, 3.0, 0.0, "Exposed to Presymptomatic"),
    
    #(infectious, 1.5, 3, 3.0, "Infectious Period"),
    #(infectious, 5.221, 0.946, 2.24, "Infectious Period"),
    
    #(infectious, 3.54, 1.22, 2.75, "Infectious Period"),
    #(infectious, 2.5, 1.3, 3.0, "Infectious Period"),
    
    #(infectious, 11.95, 0.51, 1.0, "Infectious Period"),
    
    #(incubation, 2.82, 1.36, 1.0, "Incubation Period"),
    #(incubation, 2.55, 1.34, 1.0, "Incubation Period"),
]


def cumulative_to_pmf(cum_probs):
    """Convert a cumulative probability array to a PMF (probability mass function).

    The cumulative array gives P(transition by day t). The PMF gives
    P(transition on exactly day t), accounting for the fact that an agent
    can only transition once (survival analysis / hazard model).
    """
    days = len(cum_probs)
    pmf = np.zeros(days)
    for t in range(days):
        if t == 0:
            pmf[t] = cum_probs[t]
        else:
            # P(transition on day t) = P(not yet transitioned by t-1) * P(transition by t | not yet)
            # = (1 - cum_probs[t-1]) * hazard[t]
            # But more directly: pmf[t] = cum_probs[t] - cum_probs[t-1]
            pmf[t] = cum_probs[t] - cum_probs[t - 1]
    return pmf


def fit_gamma_to_pmf(days, pmf, title=""):
    """Fit a gamma distribution to an empirical PMF using least-squares optimization.

    Uses differential evolution for a global search followed by local refinement,
    minimizing the sum of squared differences between the gamma PDF and the empirical PMF.

    Returns (shape, loc, scale) that best fit the PMF.
    """
    # Normalize PMF (in case it doesn't sum to exactly 1)
    pmf = np.array(pmf, dtype=float)
    pmf_sum = pmf.sum()
    if pmf_sum > 0:
        pmf = pmf / pmf_sum

    x = np.arange(len(days))

    def objective(params):
        shape, loc, scale = params
        if shape <= 0 or scale <= 0:
            return 1e10
        # Evaluate gamma PDF at each integer day
        pdf_vals = gamma.pdf(x, a=shape, loc=loc, scale=scale)
        # Normalize so it sums to 1 over the support
        pdf_sum = pdf_vals.sum()
        if pdf_sum <= 0:
            return 1e10
        pdf_vals = pdf_vals / pdf_sum
        # Least-squares residual
        return np.sum((pdf_vals - pmf) ** 2)

    # Global search with differential evolution
    bounds = [(0.5, 50.0), (-2.0, 5.0), (0.1, 50.0)]
    result_global = differential_evolution(
        objective,
        bounds,
        rng=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1,
    )

    # Local refinement from the global best
    result_local = minimize(
        objective,
        result_global.x,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50000},
    )

    best = result_local.x if result_local.fun < result_global.fun else result_global.x
    shape_fit, loc_fit, scale_fit = best

    if title:
        print(f"  Optimized gamma fit (least-squares on PMF):")
        print(f"    shape (α): {shape_fit:.4f}")
        print(f"    loc:       {loc_fit:.4f}")
        print(f"    scale (β): {scale_fit:.4f}")
        print(f"    residual:  {min(result_local.fun, result_global.fun):.6e}")

    return shape_fit, loc_fit, scale_fit


def discrete_gamma_cdf(day_indices, shape, scale, loc):
    """Cumulative probability at each whole day for a Gamma(shape, scale)+loc period that is
    rounded to the nearest day.

    Rounding sends a drawn period X to day t whenever t - 0.5 <= X < t + 0.5, so the
    cumulative probability of having transitioned by day t is P(X < t + 0.5). This is the
    quantity comparable to Epicast's per-day table (and to what the model sees, since the
    periods drawn in setInfected() are rounded to whole days downstream) -- evaluating the
    continuous CDF at t instead shifts every point by half a day's worth of probability.
    """
    return gamma.cdf(np.asarray(day_indices, dtype=float) + 0.5, a=shape, loc=loc, scale=scale)


def group_transitions(trans_list):
    """Group entries sharing the same cumulative-probability array onto one subplot.

    Keyed by the array's *contents* rather than its identity, so two entries referring to
    the same list (or to equal lists) end up as separate series on a single plot. Groups
    come back in order of first appearance.
    """
    groups = {}
    for entry in trans_list:
        groups.setdefault(tuple(entry[0]), []).append(entry)
    return list(groups.values())


parser = argparse.ArgumentParser(
    description="Compare manually-tuned gamma distributions against Epicast's per-day "
                "transition probability tables.",
)
parser.add_argument(
    "--fit", "-f",
    action="store_true", default=False,
    help="Also derive gamma parameters automatically by least-squares fitting the PMF, and "
         "overlay the result. Off by default because the global differential-evolution "
         "search it runs per group is slow.",
)
args = parser.parse_args()

# Colors cycled through the series within a single group, so each manually-tuned gamma
# plotted against the same Epicast curve is distinguishable.
SERIES_COLORS = ["red", "green", "darkorange", "purple", "brown", "magenta", "olive"]

groups = group_transitions(transitions)

figsize = (10, 8) if len(groups) == 1 else (10 * len(groups), 8)

fig, axes = plt.subplots(1, len(groups), figsize=figsize, squeeze=False)

for idx, group in enumerate(groups):
    ax = axes[0][idx]

    # Every entry in a group shares the same cumulative array by construction.
    trans_probs = group[0][0]
    days = len(trans_probs)
    day_indices = list(range(days))

    # Titles within a group are normally identical; if they differ, the subplot takes the
    # first and each series carries its own title in the legend instead.
    titles = [entry[4] for entry in group]
    group_title = titles[0]
    label_with_title = len(set(titles)) > 1

    print(f"{group_title}")

    # --- Plot Epicast cumulative probabilities as bars ---
    bars = ax.bar(
        range(days),
        trans_probs,
        color="blue",
        alpha=0.2,
        label="Epicast (cumulative)",
    )
    for i, (bar, prob) in enumerate(zip(bars, trans_probs)):
        height = bar.get_height()
        if prob > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{prob:.2f}",
                ha="center",
                va="bottom",
                fontsize=20,
            )

    # --- Plot each manually-tuned gamma in the group as its own series ---
    # Each is evaluated only at whole days (see discrete_gamma_cdf), so it lines up one-for-one
    # with the Epicast bars and the correlation below is a direct day-for-day comparison rather
    # than a resampling of a continuous curve.
    corr_by_series = []
    for series_idx, (_, shape, scale, loc, title) in enumerate(group):
        gamma_manual = discrete_gamma_cdf(day_indices, shape, scale, loc)
        corr_manual = pearsonr(gamma_manual, trans_probs)[0]
        params_str = f"α={shape:.2f}, β={scale:.2f}, loc={loc:.2f} r={corr_manual:.3f}"
        label = f"{title}: Gamma ({params_str})" if label_with_title else f"Gamma ({params_str})"
        corr_by_series.append(("Manual", params_str, corr_manual))
        ax.plot(
            day_indices,
            gamma_manual,
            color=SERIES_COLORS[series_idx % len(SERIES_COLORS)],
            linestyle="-",
            # The rounded period is constant across [t - 0.5, t + 0.5), so the CDF is a step
            # function whose treads are centered on the integer days -- steps-mid, not the
            # steps-post/pre that would imply the jump happens at the day boundary.
            drawstyle="steps-mid",
            marker="o",
            markersize=8,
            lw=2,
            label=label,
        )

    # The fit is derived and drawn only under --fit: the global search in fit_gamma_to_pmf is
    # by far the slowest part of this script, and with the flag off nothing uses its result.
    # Kept in one block (rather than fitting earlier and plotting here) so the fitted
    # parameters can't be read on a path that never assigned them.
    if args.fit:
        # --- Convert cumulative probs to PMF ---
        pmf = cumulative_to_pmf(trans_probs)

        # --- Fit gamma to the PMF via optimization ---
        shape_opt, loc_opt, scale_opt = fit_gamma_to_pmf(day_indices, pmf, title=group_title)

        # --- Plot optimized gamma (green) ---
        gamma_opt = discrete_gamma_cdf(day_indices, shape_opt, scale_opt, loc_opt)
        corr_opt = pearsonr(gamma_opt, trans_probs)[0]
        corr_by_series.append(
            ("Optimized", f"α={shape_opt:.2f}, β={scale_opt:.2f}, loc={loc_opt:.2f}", corr_opt)
        )
        ax.plot(
            day_indices,
            gamma_opt,
            color="green",
            lw=2,
            linestyle="-",
            drawstyle="steps-mid",
            marker="^",
            markersize=8,
            label=f"Optimized gamma (α={shape_opt:.2f}, β={scale_opt:.2f}, loc={loc_opt:.2f} , r={corr_opt:.3f})",
        )

    print(f"  Pearson r vs Epicast cumulative probs:")
    for name, params_str, corr in corr_by_series:
        print(f"    {name} ({params_str}): {corr:.4f}")
    print()

    ax.set_xlim(0, days - 1)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(group_title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, 1.0))

plt.tight_layout()
plt.savefig("epicast_transitions_comparison.png", bbox_inches="tight", dpi=300)
plt.show()
