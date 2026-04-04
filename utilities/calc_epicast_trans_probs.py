#!/usr/bin/env -S python -u

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, pearsonr
from scipy.optimize import minimize, differential_evolution
import sys

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
    #(exposed_to_presymp, 6.0, 0.73, 0.0, "Exposed to Presymptomatic"),
    #(incubation, 6.0, 0.73, 1.0, "Incubation Period"),
    #(infectious, 27.0, 0.25, 0.0, "Infectious Period"),
    (exposed_to_presymp, 1.82, 2.36, 0.0, "Exposed to Presymptomatic"),
    (incubation, 1.82, 2.36, 1.0, "Incubation Period"),
    (infectious, 3.56, 1.22, 2.5, "Infectious Period"),
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


fig, axes = plt.subplots(3, 1, figsize=(10, 14))

for idx, (trans_probs, shape, scale, loc, title) in enumerate(transitions):
    ax = axes[idx]

    days = len(trans_probs)
    day_indices = list(range(days))

    print(f"{title}")

    # --- Convert cumulative probs to PMF ---
    pmf = cumulative_to_pmf(trans_probs)

    # --- Fit gamma to the PMF via optimization ---
    shape_opt, loc_opt, scale_opt = fit_gamma_to_pmf(day_indices, pmf, title=title)

    # --- Also do MLE fit on samples for comparison ---
    num_agents = 100000
    day_counts = [0] * days
    agent_days = []
    for i in range(num_agents):
        for t in range(days):
            if np.random.uniform() <= trans_probs[t]:
                day_counts[t] += 1
                agent_days.append(t)
                break

    fshape_mle, loc_mle, fscale_mle = gamma.fit(agent_days, floc=0)
    print(f"  MLE gamma fit (from simulated samples):")
    print(f"    shape (α): {fshape_mle:.4f}")
    print(f"    loc:       {loc_mle:.4f}")
    print(f"    scale (β): {fscale_mle:.4f}")

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
                fontsize=10,
            )

    x_cont = np.linspace(0, days - 1, 10000)

    # --- Plot original manually-tuned gamma (red) ---
    gamma_manual = gamma.cdf(x_cont, a=shape, loc=loc, scale=scale)
    corr_manual = pearsonr(
        np.interp(np.linspace(0, 1, days), np.linspace(0, 1, len(gamma_manual)), gamma_manual),
        trans_probs,
    )[0]
    ax.plot(
        x_cont,
        gamma_manual,
        "r--",
        lw=2,
        label=f"Manual gamma (α={shape:.2f}, β={scale:.2f}, r={corr_manual:.3f})",
    )

    # --- Plot MLE-fitted gamma (orange) ---
    gamma_mle = gamma.cdf(x_cont, a=fshape_mle, loc=loc_mle, scale=fscale_mle)
    corr_mle = pearsonr(
        np.interp(np.linspace(0, 1, days), np.linspace(0, 1, len(gamma_mle)), gamma_mle),
        trans_probs,
    )[0]
    ax.plot(
        x_cont,
        gamma_mle,
        color="orange",
        lw=2,
        linestyle="-.",
        label=f"MLE gamma (α={fshape_mle:.2f}, β={fscale_mle:.2f}, r={corr_mle:.3f})",
    )

    # --- Plot optimized gamma (green) ---
    gamma_opt = gamma.cdf(x_cont, a=shape_opt, loc=loc_opt, scale=scale_opt)
    corr_opt = pearsonr(
        np.interp(np.linspace(0, 1, days), np.linspace(0, 1, len(gamma_opt)), gamma_opt),
        trans_probs,
    )[0]
    ax.plot(
        x_cont,
        gamma_opt,
        "g-",
        lw=2,
        label=f"Optimized gamma (α={shape_opt:.2f}, loc={loc_opt:.2f}, β={scale_opt:.2f}, r={corr_opt:.3f})",
    )

    print(f"  Pearson r vs Epicast cumulative probs:")
    print(f"    Manual:    {corr_manual:.4f}")
    print(f"    MLE:       {corr_mle:.4f}")
    print(f"    Optimized: {corr_opt:.4f}")
    print()

    ax.set_xlim(0, days - 1)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("epicast_transitions_comparison.png", bbox_inches="tight", dpi=300)
plt.show()
