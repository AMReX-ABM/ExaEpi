#!/usr/bin/env -S python -u

"""Decompose ExaEpi output variance into disease stochasticity and population realization.

ExaEpi normally runs against a single UrbanPop realization, so its reported spread reflects only
the disease model's own randomness. This script compares runs grouped into arms that add one
source of population variation at a time:

    A   one population, many disease seeds      -- the noise floor
    B   populations differing by --rseed        -- adds work/school destinations and mixing groups
    C   populations additionally perturbed      -- adds ACS-scale variation in who lives where
        within ACS sampling error

The arms are nested, so the interesting comparisons are B against A (does re-drawing assignment
matter?) and C against B (does re-drawing the residential population matter on top of that?). An
arm whose spread is indistinguishable from A is telling you that source of variation does not
reach the model's outputs.

Two families of metric are reported, because they can disagree and the disagreement is the point.
Aggregate curve metrics come from the per-day diagnostic table; a state-wide curve can absorb a
lot of spatial rearrangement. The spatial metric is the per-block-group attack rate, compared by
correlation rather than by variance, since it is a field rather than a scalar.

Usage:
    compare_realizations.py --runs <dir> [--curve output_nm.dat] [--spatial cases00070]
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np


def parse_curve(fname: str) -> dict[str, np.ndarray]:
    """Read ExaEpi's whitespace-aligned per-day diagnostic table into column arrays."""
    with open(fname) as f:
        header = f.readline().split()
        rows = [line.split() for line in f if line.strip()]
    if not rows:
        raise RuntimeError(f"{fname} has no data rows")
    data = np.array(rows, dtype=np.float64)
    return {name: data[:, i] for i, name in enumerate(header)}


def curve_metrics(cols: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce one run's epidemic curve to scalars.

    Infectious prevalence is the sum of every compartment carrying an active infection:
    pre-symptomatic and symptomatic, hospitalised or not, plus asymptomatic. `Su` is the
    susceptible count, so the drop in it over the run is the cumulative infection total, which
    avoids depending on how the incidence columns treat reinfection.
    """
    infectious_cols = ["PS/PI", "S/PI/NH", "S/PI/H", "PS/I", "S/I/NH", "S/I/H", "A/PI", "A/I"]
    present = [c for c in infectious_cols if c in cols]
    if not present:
        raise RuntimeError(f"no infectious columns found; have {sorted(cols)}")
    infectious = np.sum([cols[c] for c in present], axis=0)

    peak_idx = int(np.argmax(infectious))
    metrics = {
        "peak_infectious": float(infectious[peak_idx]),
        "peak_day": float(cols["Day"][peak_idx]),
        "final_infectious": float(infectious[-1]),
    }
    if "Su" in cols:
        metrics["cumulative_infected"] = float(cols["Su"][0] - cols["Su"][-1])
        metrics["attack_rate"] = float((cols["Su"][0] - cols["Su"][-1]) / cols["Su"][0])
    if "D" in cols:
        metrics["deaths"] = float(cols["D"][-1])
    return metrics


def parse_spatial(fname: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a per-block-group aggregated diagnostic. Returns (geoids, attack_rate)."""
    geoids, total, never = [], [], []
    with open(fname) as f:
        header = f.readline().strip().split(",")
        try:
            i_total = header.index("total")
            i_never = header.index("never_infected")
        except ValueError as exc:
            raise RuntimeError(f"{fname} lacks total/never_infected: {header}") from exc
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(i_total, i_never):
                continue
            geoids.append(parts[0])
            total.append(float(parts[i_total]))
            never.append(float(parts[i_never]))
    t = np.array(total)
    n = np.array(never)
    with np.errstate(divide="ignore", invalid="ignore"):
        attack = np.where(t > 0, 1.0 - n / t, np.nan)
    return np.array(geoids), attack


def arm_of(run_name: str) -> str | None:
    """Arm label from a run directory name like 'B_r3_s1' or 'A_base_s7'."""
    m = re.match(r"^([ABC])_", run_name)
    return m.group(1) if m else None


def collect(runs_dir: str, curve_name: str, spatial_name: str):
    """Gather per-run metrics, excluding runs that did not finish.

    A run still in progress leaves a truncated diagnostic table behind, and reducing one of those
    yields a plausible-looking but wrong result: a mid-epidemic snapshot reads as a much lower
    attack rate and puts the 'peak' at whatever day the file happens to stop. Nothing about the
    numbers flags it, so incomplete runs are detected here by row count -- any run short of the
    longest curve seen is dropped and named, rather than quietly averaged in.
    """
    loaded: list[tuple[str, str, dict[str, np.ndarray], str]] = []
    missing = []
    for name in sorted(os.listdir(runs_dir)):
        arm = arm_of(name)
        if arm is None:
            continue
        d = os.path.join(runs_dir, name)
        cfile = os.path.join(d, curve_name)
        if not os.path.exists(cfile):
            missing.append(name)
            continue
        loaded.append((name, arm, parse_curve(cfile), os.path.join(d, spatial_name)))

    curves: dict[str, list[dict[str, float]]] = defaultdict(list)
    spatial: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    incomplete = []
    if not loaded:
        return curves, spatial, missing, incomplete

    full_len = max(len(cols["Day"]) for _, _, cols, _ in loaded)
    for name, arm, cols, sfile in loaded:
        if len(cols["Day"]) < full_len:
            incomplete.append((name, len(cols["Day"])))
            continue
        metrics = curve_metrics(cols)
        metrics["_run"] = name
        curves[arm].append(metrics)
        if os.path.exists(sfile):
            geoids, attack = parse_spatial(sfile)
            spatial[arm].append((name, _keyed(geoids, attack)))
    return curves, spatial, missing, incomplete


def population_of(run_name: str) -> str:
    """Population id from a run name: 'B_r3_s1' -> 'B_r3', 'A_base_s7' -> 'A_base'."""
    return re.sub(r"_s\d+$", "", run_name)


def nested_variance(records: list[dict[str, float]], metric: str) -> dict[str, float] | None:
    """Split an arm's spread into between-population and within-population (disease seed) parts.

    An arm built as several populations times several seeds carries both sources at once, and
    pooling every run into one standard deviation mixes them. This is the one-way decomposition
    that separates them, which is what the question actually asks: how much of the spread comes
    from which population was used, rather than from the disease model's own randomness.

    Returns None unless there are at least two populations with at least two runs each.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if metric in rec and "_run" in rec:
            groups[population_of(str(rec["_run"]))].append(rec[metric])
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(usable) < 2:
        return None

    k = len(usable)
    n_total = sum(len(v) for v in usable.values())
    grand = sum(sum(v) for v in usable.values()) / n_total
    means = {g: sum(v) / len(v) for g, v in usable.items()}

    ss_between = sum(len(v) * (means[g] - grand) ** 2 for g, v in usable.items())
    ss_within = sum((x - means[g]) ** 2 for g, v in usable.items() for x in v)
    df_between, df_within = k - 1, n_total - k
    if df_within <= 0:
        return None
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Expected mean squares for a balanced design: MS_between estimates
    # sigma^2_within + n * sigma^2_between. Clamp at zero -- a negative estimate just means the
    # between-population component is not resolvable above the seed noise.
    n_per = n_total / k
    var_between = max(0.0, (ms_between - ms_within) / n_per)
    var_within = ms_within
    total = var_between + var_within

    from scipy import stats as _stats

    return {
        "populations": k,
        "runs": n_total,
        "sd_between": math.sqrt(var_between),
        "sd_within": math.sqrt(var_within),
        "icc": var_between / total if total > 0 else float("nan"),
        "p": float(_stats.f.sf(ms_between / ms_within, df_between, df_within)) if ms_within > 0 else 1.0,
    }


def _keyed(geoids: np.ndarray, attack: np.ndarray) -> dict[str, float]:
    return {g: a for g, a in zip(geoids, attack) if not math.isnan(a)}


def f_test_greater(var_num: float, n_num: int, var_den: float, n_den: int) -> float:
    """One-sided p for var_num > var_den, i.e. is this arm wider than the noise floor.

    Returns 1.0 when the floor has no spread at all, since an arm cannot then be shown wider by
    this test.
    """
    if var_den <= 0 or n_num < 2 or n_den < 2:
        return 1.0
    from scipy import stats as _stats

    return float(_stats.f.sf(var_num / var_den, n_num - 1, n_den - 1))


def report_curves(curves: dict[str, list[dict[str, float]]]) -> None:
    arms = [a for a in ("A", "B", "C") if a in curves]
    if not arms:
        return
    names = sorted({k for arm in arms for m in curves[arm] for k in m if not k.startswith("_")})

    print("\n=== Aggregate curve metrics ===")
    print("  A = disease seed only (noise floor)")
    print("  B = + assignment/group structure (--rseed)")
    print("  C = + residential population within ACS sampling error\n")

    for metric in names:
        print(f"{metric}:")
        stats = {}
        for arm in arms:
            vals = np.array([m[metric] for m in curves[arm] if metric in m])
            if len(vals) < 2:
                continue
            stats[arm] = (vals.mean(), vals.std(ddof=1), len(vals))
            print(
                f"  {arm}: n={len(vals):>3}  mean={vals.mean():>14.4f}  sd={vals.std(ddof=1):>12.4f}"
                f"  cv={vals.std(ddof=1) / abs(vals.mean()) if vals.mean() else float('nan'):>8.4f}"
            )
        # Variance above the floor, in the nested sense: how much of arm X's spread is not
        # explained by disease stochasticity. The sd ratio alone is not evidence -- with ten runs
        # in arm A its sd is itself uncertain to roughly a quarter of its value, so ratios well
        # above 1 arise routinely by chance. The F test on the variance ratio says whether the
        # arm is genuinely wider; without it a 1.3x ratio reads as a finding when it is noise.
        if "A" in stats:
            var_a = stats["A"][1] ** 2
            for arm in ("B", "C"):
                if arm not in stats:
                    continue
                var = stats[arm][1] ** 2
                ratio = stats[arm][1] / stats["A"][1] if stats["A"][1] > 0 else float("inf")
                share = (var - var_a) / var if var > 0 else float("nan")
                n_arm, n_a = stats[arm][2], stats["A"][2]
                p = f_test_greater(var, n_arm, var_a, n_a)
                verdict = "wider than floor" if p < 0.05 else "indistinguishable from floor"
                print(
                    f"    {arm} vs A: sd ratio {ratio:>6.2f}x   "
                    f"excess variance share {share:>7.1%}   F p={p:.3f}  [{verdict}]"
                )
        # The within-arm decomposition, which does not depend on arm A at all.
        for arm in ("B", "C"):
            if arm not in curves:
                continue
            nv = nested_variance(curves[arm], metric)
            if nv is None:
                continue
            verdict = "population matters" if nv["p"] < 0.05 else "no resolvable population effect"
            print(
                f"    {arm} nested: {nv['populations']} populations x seeds   "
                f"sd_between={nv['sd_between']:.4g}  sd_within={nv['sd_within']:.4g}  "
                f"ICC={nv['icc']:.3f}  F p={nv['p']:.3f}  [{verdict}]"
            )
        print()


def mean_pairwise_corr(members: list[tuple[str, dict[str, float]]]) -> tuple[float, int]:
    """Mean Pearson correlation between every pair of per-block-group attack-rate fields."""
    if len(members) < 2:
        return float("nan"), 0
    corrs = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i][1], members[j][1]
            shared = sorted(set(a) & set(b))
            if len(shared) < 3:
                continue
            va = np.array([a[g] for g in shared])
            vb = np.array([b[g] for g in shared])
            if va.std() == 0 or vb.std() == 0:
                continue
            corrs.append(float(np.corrcoef(va, vb)[0, 1]))
    if not corrs:
        return float("nan"), 0
    return float(np.mean(corrs)), len(corrs)


def report_spatial(spatial: dict[str, list[tuple[str, dict[str, float]]]]) -> None:
    arms = [a for a in ("A", "B", "C") if a in spatial and len(spatial[a]) >= 2]
    if not arms:
        print("=== Per-block-group attack rate: no spatial output found ===")
        return

    print("=== Per-block-group attack rate ===")
    print("Mean pairwise correlation between runs within each arm. Arm A is the ceiling: two runs")
    print("of the same population differ only by disease seed. A lower value in B or C means that")
    print("source of population variation is moving the epidemic's spatial pattern.\n")
    for arm in arms:
        corr, npairs = mean_pairwise_corr(spatial[arm])
        n_bg = len(spatial[arm][0][1])
        print(f"  {arm}: runs={len(spatial[arm]):>3}  pairs={npairs:>4}  "
              f"block groups={n_bg:>5}  mean pairwise r={corr:.4f}")
    print()


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", "-r", required=True, help="directory holding the per-run subdirectories")
    p.add_argument("--curve", default="output_nm.dat", help="per-day diagnostic filename within each run")
    p.add_argument("--spatial", default="cases00070", help="per-block-group diagnostic filename within each run")
    return p.parse_args()


def main():
    args = get_args()
    curves, spatial, missing, incomplete = collect(args.runs, args.curve, args.spatial)
    total = sum(len(v) for v in curves.values())
    print(f"Collected {total} complete runs from {args.runs}")
    for arm in ("A", "B", "C"):
        if arm in curves:
            print(f"  arm {arm}: {len(curves[arm])} runs")
    if missing:
        print(f"  WARNING: {len(missing)} run dirs had no {args.curve}: {missing[:5]}")
    if incomplete:
        print(f"  WARNING: {len(incomplete)} runs excluded as incomplete (short curve): "
              f"{[f'{n} ({d} days)' for n, d in incomplete[:5]]}")
    if total == 0:
        sys.exit("no runs collected")
    report_curves(curves)
    report_spatial(spatial)


if __name__ == "__main__":
    main()
