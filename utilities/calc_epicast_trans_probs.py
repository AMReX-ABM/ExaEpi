#!/usr/bin/env -S python -u

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, pearsonr, norm
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
    # (exposed_to_presymp, 5.2, 0.75, "Exposed to Presymptomatic"),
    # (incubation, 7.5, 0.65, "Incubation Period"),
    # (infectious, 26.2, 0.23, "Infectious Period"),
    (exposed_to_presymp, 6.0, 0.73, "Exposed to Presymptomatic"),
    (incubation, 8.0, 0.65, "Incubation Period"),
    (infectious, 27.0, 0.25, "Infectious Period"),
]

fig, axes = plt.subplots(3, 1, figsize=(10, 14))

for idx, (trans_probs, shape, scale, title) in enumerate(transitions):
    ax = axes[idx]

    num_agents = 100000
    days = len(trans_probs)
    day_counts = [0] * days
    agent_days = []

    print(f"{title}")

    for i in range(num_agents):
        for t in range(days):
            if np.random.uniform() <= trans_probs[t]:
                day_counts[t] += 1
                agent_days.append(t)
                break

    # plt.bar(range(days), day_counts, color="blue", alpha=0.6, label="Observed")
    bars = ax.bar(
        range(days),
        # [p * num_agents for p in trans_probs],
        trans_probs,
        color="blue",
        alpha=0.2,
        label="Epicast",
    )
    # Add value labels on top of each bar
    for i, (bar, prob) in enumerate(zip(bars, trans_probs)):
        height = bar.get_height()
        if prob > 0:  # Only show non-zero values
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{prob:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # x = np.linspace(0, days - 1, num_agents)
    x = np.linspace(0, 10, num_agents)

    fshape, loc, fscale = gamma.fit(agent_days, floc=0)
    print(f"  Gamma fit:\n  shape {fshape:.3f}\n  loc {loc:.3f}\n  scale {fscale:.3f}")

    # for alpha, beta, color in ((fshape, fscale, "r-"), (shape, scale, "g-")):
    for alpha, beta, color in [(shape, scale, "r-")]:
        fitted_gamma = gamma.cdf(x, a=alpha, loc=loc, scale=beta)  # * num_agents
        corr = pearsonr(sorted(fitted_gamma), sorted(agent_days))[0]
        print(f"  Correlation: {corr:.3f}")
        print(f"  Gamma Distribution Parameters:")
        print(f"    Shape (α): {alpha:.2f}")
        print(f"    Rate (β): {beta:.2f}")
        print(f"    Mean: {fitted_gamma.mean():.2f}")

        # Plot fitted gamma distribution
        ax.plot(
            x,
            fitted_gamma,
            color,
            lw=2,
            label=f"Gamma fit (α={alpha:.2f}, β={beta:.2f}, $R^2={corr:.2f}$)",
        )
        ax.set_xlim(0, 10)

    ax.set_xlabel("Days")
    ax.set_ylabel("Number of Transitions")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("epicast_transitions_comparison.png", bbox_inches="tight", dpi=300)
# plt.show()
