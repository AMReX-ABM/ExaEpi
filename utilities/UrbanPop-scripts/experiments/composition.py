"""Block-group composition measures shared by resolve_spread.py and laplace_ref.py.

Household turnover (the share of households placed differently between two draws) counts swapping
a donor for a near-identical one as a change, so it overstates how different two populations are
for an epidemic model. These measure composition directly, from each donor's constraint counts
(est_ind), as expected counts per block group under an allocation:

    constrained    shares ExaEpi reads that the ACS pins per block group -- under 18, 65 and over,
                   workers, K-12 students, working from home. Reported as the CV across draws, and
                   against the ACS CV of the same aggregate (SE combined as sqrt(sum se^2), which
                   ignores covariance between categories)
    joint          household-level combinations no constraint sees, which only the donors and
                   their prior weights determine: children living with someone 65+, people 65+ in
                   households of three or more, K-12 students living with a worker, and people 65+
                   living with a health/education/social-care worker. Reported as the CV across
                   draws; there is no published reference

Also `in_moe`: the share of every block-group and tract constraint cell whose fitted count lies
within the published 90% margin of error, pymedm's own fit diagnostic, averaged over draws.

Column names are those of the minimal set, all of which also appear in up_expanded.
"""

import numpy as np

AGE_U18 = ("a05u", "a05_09", "a10_14", "a15_17")
AGE_65O = ("a65_66", "a67_69", "a70_74", "a75_79", "a80_84", "a85o")
K12 = ("kind", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th",
       "12th")


def _idx(cols, names):
    return [cols.index(n) for n in names]


def groups(cols):
    """Column indices of each constrained aggregate."""
    sexes = ("male", "female")
    return {
        "under18": _idx(cols, [f"{s}_{a}" for s in sexes for a in AGE_U18]),
        "65plus": _idx(cols, [f"{s}_{a}" for s in sexes for a in AGE_65O]),
        "workers": [i for i, c in enumerate(cols) if c.startswith("sexnaics_")],
        "k12": _idx(cols, [f"grade_{g}" for g in K12]),
        "wfh": _idx(cols, ["travel_wfh"]),
    }


def joint_features(C, cols):
    """Per-donor counts of household-level combinations no constraint targets."""
    g = groups(cols)
    u18, o65 = C[:, g["under18"]].sum(1), C[:, g["65plus"]].sum(1)
    workers, k12 = C[:, g["workers"]].sum(1), C[:, g["k12"]].sum(1)
    size = C[:, cols.index("population")]
    care = C[:, _idx(cols, ["sexnaics_male_edu_med_sca", "sexnaics_female_edu_med_sca"])].sum(1)
    return {
        "kids_with_65plus": np.where(o65 > 0, u18, 0.0),
        "65plus_in_hh3plus": np.where(size >= 3, o65, 0.0),
        "k12_with_worker": np.where(workers > 0, k12, 0.0),
        "65plus_with_care_worker": np.where(care > 0, o65, 0.0),
    }


def _cv(x):
    m = x.mean(0)
    return x.std(0, ddof=1) / np.where(m > 0, m, np.nan)


def measure(als, C, cols, A1, est1, est2, se1, se2):
    """Composition and fit measures over a list of (donors x block groups) allocations."""
    g, feats = groups(cols), joint_features(C, cols)
    bgK = np.array([a.T @ C for a in als])                      # draws x G x K
    out = {}
    for name, ix in g.items():
        cnt = bgK[:, :, ix].sum(2)
        pub = est2[:, ix].sum(1)
        acs_cv = np.sqrt((se2[:, ix] ** 2).sum(1)) / np.where(pub > 0, pub, np.nan)
        cv = _cv(cnt)
        ok = np.isfinite(cv) & np.isfinite(acs_cv) & (pub >= 20)
        out[f"cv_{name}"] = float(np.median(cv[ok]))
        out[f"ratio_{name}"] = float(np.median(cv[ok]) / np.median(acs_cv[ok]))
    for name, f in feats.items():
        cnt = np.array([a.T @ f for a in als])
        cv = _cv(cnt)
        ok = np.isfinite(cv) & (cnt.mean(0) >= 5)
        out[f"cv_{name}"] = float(np.median(cv[ok]))
    trK = np.einsum("tg,dgk->dtk", A1, bgK)
    inside = [np.abs(bgK - est2) <= 1.645 * se2 + 1e-9, np.abs(trK - est1) <= 1.645 * se1 + 1e-9]
    out["in_moe"] = float(np.concatenate([x.reshape(len(als), -1) for x in inside], 1).mean())
    return out


CONSTRAINED = ("under18", "65plus", "workers", "k12", "wfh")
JOINT = ("kids_with_65plus", "65plus_in_hh3plus", "k12_with_worker", "65plus_with_care_worker")


def header():
    return ("  constrained CV (ratio to ACS): " + ", ".join(CONSTRAINED)
            + "\n  joint CV: " + ", ".join(JOINT))


def line(m):
    c = " ".join(f"{m['cv_' + n]:.4f}({m['ratio_' + n]:.2f})" for n in CONSTRAINED)
    j = " ".join(f"{m['cv_' + n]:.4f}" for n in JOINT)
    return f"in_moe {m['in_moe']:.4f} | {c} | {j}"
