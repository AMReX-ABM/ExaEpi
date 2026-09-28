#!/usr/bin/env python

"""Estimate Epicast's workgroups, with the industry of each, from an events file and the UrbanPop
population Epicast shares with ExaEpi -- a rough stand-in for the per-industry dump Epicast doesn't
write. The output has one "naics size" line per estimated workgroup, the format
plot_workgroup_size_by_industry.py's --epicast_by_industry reads.

Epicast splits every industry (3-digit NAICS) in every daytime community into its own workgroups,
sized from a per-industry CBP target (Epicast 2.0 sec. 2.4.2, 2.8.3). So what sets the group sizes
is how many workers of each industry each community holds, and this rebuilds those counts:

  - Workers: every UrbanPop agent with a NAICS code outside educational services (61), which
    Epicast sends to schools instead -- none of its work-context infections are in sector 61.
    Epicast agent ids index the .bin directly (read_epicast_events.urbanpop_agent_index), which is
    where each worker's industry comes from.
  - Daytime community: from the worker's ctx_presymptomatic event, which every infected agent has,
    always on a (week)day timestep, so it's the regular workplace -- see
    read_epicast_events.estimate_community_populations. A worker who was never infected is given the
    daytime community of a randomly drawn infected worker from the same home tract (Epicast draws
    workplaces from tract-level commute flows, with no dependence on industry). A high-attack-rate
    run leaves little to fill in this way: CA p02 observes 94% of workers directly.
  - Working students: Epicast also puts ~1.07M CA students in workgroups, whose NAICS codes the
    UrbanPop converter drops (upop_to_exaepi.py treats an employed agent of 26 or under with a school
    grade as a student). Those infected at work are known to be among them, with a known workplace;
    each is replicated to make up the difference between Epicast's actual workgroup membership and
    the worker count above, and given the industry of a randomly drawn worker aged 26 or under from
    the same daytime community. This is the roughest part of the estimate, but only ~6% of members.
  - Groups: each (community, industry) cell of W workers gets ceil(W / T) groups, and every worker
    joins one at random. T is the state's CBP employment per establishment for that industry,
    capped at 86, times a single scale factor (--scale). Industries CBP doesn't cover (public
    administration) get 20, as in compute_workgroup_sizes.py.

With the scale at 1, which is the rule as the paper describes it, CA comes out with 15% fewer groups
than Epicast actually has (2.44M vs 2.86M), so Epicast's real targets are evidently smaller than
CBP's mean establishment size. --scale auto (the default) fits the one factor so the expected number
of groups matches the actual sizes file; on CA p02 that is ~0.74, and it also brings the size
distribution closer (KS distance 0.063 -> 0.041) though it isn't fitted to it. Nothing
distinguishes one industry's target from another's in that fit, so an industry whose real target is
off by more than the average is off by that much here too. Each run prints a summary of how
the estimate compares with the actual sizes.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UrbanPop-scripts"))
from compute_workgroup_sizes import (  # noqa: E402
    DEFAULT_CAP, DEFAULT_SIZE, load_cbp_cache, parse_naics_descriptions, resolve_naics_code)
from plot_commute_distance import read_urbanpop_columns  # noqa: E402
from plot_workgroup_size_by_industry import STATE_FIPS  # noqa: E402
from read_epicast_events import read_events_records, urbanpop_agent_index  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw Epicast event context codes (see read_epicast_events._CONTEXT_MAP).
CTX_WORK = 0x04
CTX_PRESYMPTOMATIC = 0x0C

# upop_to_exaepi.py counts an employed agent with a school grade as a student at this age or under.
MAX_STUDENT_WORKER_AGE = 26


def draw_by_key(pool_keys, pool_values, keys, rng):
    """For each of keys, a value drawn uniformly from the pool entries with the same key, or from the
    whole pool where there are none. Returns (values, number of keys that fell back to the pool)."""
    order = np.argsort(pool_keys, kind="stable")
    pool_keys, pool_values = pool_keys[order], pool_values[order]
    lo = np.searchsorted(pool_keys, keys, "left")
    hi = np.searchsorted(pool_keys, keys, "right")
    none = hi == lo
    lo[none], hi[none] = 0, len(pool_keys)
    return pool_values[lo + (rng.random(len(keys)) * (hi - lo)).astype(np.int64)], int(none.sum())


def first_event_per_agent(records, context):
    """(agent_id, location_id) of each agent's first event in the given raw context."""
    selected = records[records["context"] == context]
    agent_ids, first = np.unique(selected["agent_id"], return_index=True)
    return agent_ids, selected["location_id"][first]


def workgroup_members(urbanpop, events, naics_header, total_members, rng):
    """(daytime community, 3-digit NAICS key) of every estimated Epicast workgroup member -- see the
    module docstring for who they are and where the values come from."""
    codes = parse_naics_descriptions(Path(naics_header))
    key_of_index = np.array([c[:3] for c in codes])
    cols = read_urbanpop_columns(urbanpop, ["home_geoid", "naics", "age"])
    naics = cols["naics"]
    print("Reading Epicast events from", events)
    records, demog = read_events_records(events)

    day = np.zeros(len(naics), dtype=np.uint64)
    seen = np.zeros(len(naics), dtype=bool)
    agent_ids, location_ids = first_event_per_agent(records, CTX_PRESYMPTOMATIC)
    infected = urbanpop_agent_index(agent_ids, demog, cols["home_geoid"])
    day[infected], seen[infected] = location_ids, True

    education = np.array([c.startswith("61") for c in codes])
    workers = np.flatnonzero((naics != -1) & ~education[np.maximum(naics, 0)])
    unseen = workers[~seen[workers]]
    home_tract = cols["home_geoid"] // 10
    observed = workers[seen[workers]]
    day[unseen], n_fallback = draw_by_key(home_tract[observed], day[observed], home_tract[unseen], rng)
    print(f"Workers: {len(workers):,}, daytime community observed for {len(observed):,} "
          f"({len(observed) / len(workers):.1%}); {len(unseen):,} drawn from their home tract"
          + (f" ({n_fallback:,} from anywhere, no observed worker in their tract)" if n_fallback else ""))
    worker_day, worker_key = day[workers], key_of_index[naics[workers]]

    work_ids, _ = first_event_per_agent(records, CTX_WORK)
    work_infected = urbanpop_agent_index(work_ids, demog, cols["home_geoid"])
    students = work_infected[naics[work_infected] == -1]
    n_needed = total_members - len(workers)
    if len(students) == 0 or n_needed <= 0:
        print(f"Working students: none added ({len(students):,} infected at work, {n_needed:,} members short)")
        return worker_day, worker_key
    factor = n_needed / len(students)
    copies = np.floor(factor).astype(np.int64) + (rng.random(len(students)) < factor % 1)
    student_day = np.repeat(day[students], copies)
    young = cols["age"][workers] <= MAX_STUDENT_WORKER_AGE
    student_key, n_fallback = draw_by_key(worker_day[young], worker_key[young], student_day, rng)
    print(f"Working students: {len(students):,} infected at work, each standing for {factor:.2f} to make up "
          f"the {n_needed:,} members Epicast has beyond the workers above"
          + (f" ({n_fallback:,} given a statewide industry, no young worker in their community)" if n_fallback else ""))
    return np.concatenate([worker_day, student_day]), np.concatenate([worker_key, student_key])


def cbp_targets(keys, cbp_file, state_fips):
    """{NAICS key: CBP employment per establishment, capped at DEFAULT_CAP}, resolved up the NAICS
    hierarchy like compute_workgroup_sizes.py; DEFAULT_SIZE where CBP has no coverage."""
    cbp = load_cbp_cache(Path(cbp_file))
    fips = f"{state_fips:02d}"
    targets = {}
    for key in keys:
        code, _note = resolve_naics_code(fips, key, cbp)
        if code is None:
            targets[key] = float(DEFAULT_SIZE)
        else:
            emp, est = cbp[(fips, code)]
            targets[key] = max(1.0, min(emp / est, DEFAULT_CAP))
    return targets


def n_groups(cell_workers, cell_targets, scale):
    """Groups per cell: ceil(W / T) at the scaled target, which can't go below 1."""
    return np.ceil(cell_workers / np.maximum(1.0, scale * cell_targets)).astype(np.int64)


def expected_nonempty(cell_workers, groups):
    """Expected number of groups left non-empty when each cell's workers join its groups at random."""
    # A single-group cell has log1p(-1) = -inf, which correctly gives it exactly one non-empty group.
    with np.errstate(divide="ignore"):
        return float(np.sum(groups * -np.expm1(cell_workers * np.log1p(-1.0 / groups))))


def fit_scale(cell_workers, cell_targets, n_actual):
    """The target scale factor whose expected non-empty group count matches n_actual, by bisection
    (the count only falls as the scale rises)."""
    lo, hi = 0.05, 20.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if expected_nonempty(cell_workers, n_groups(cell_workers, cell_targets, mid)) > n_actual:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def summarize(label, sizes, reference=None):
    line = (f"{label:<10}{len(sizes):>12,}{int(sizes.sum()):>13,}{sizes.mean():>8.2f}{np.mean(sizes == 1):>9.1%}"
            f"{np.percentile(sizes, 90):>6.0f}{np.percentile(sizes, 99):>6.0f}{sizes.max():>6d}")
    if reference is not None:
        top = max(sizes.max(), reference.max()) + 1
        cdf = np.cumsum(np.bincount(sizes, minlength=top)) / len(sizes)
        ref_cdf = np.cumsum(np.bincount(reference, minlength=top)) / len(reference)
        line += f"{np.abs(cdf - ref_cdf).max():>8.3f}"
    print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", "-s", type=str.lower, choices=sorted(STATE_FIPS), default="ca",
                        help="Two-letter state abbreviation; selects the state's CBP rows and the default input files")
    parser.add_argument("--events", "-e", required=True,
                        help="Epicast events.bin -- the higher the attack rate, the more workplaces are observed")
    parser.add_argument("--urbanpop", "-u", default=None,
                        help="UrbanPop .bin the Epicast run was built from (default data/UrbanPop/urbanpop_<state>.bin)")
    parser.add_argument("--epicast_sizes", default=None,
                        help="Epicast's actual workgroup sizes, one per line, for the membership total, the scale fit "
                        "and the comparison (default data/results/emerge-paper/epicast/<state>/<state>_workgroup_sizes.txt)")
    parser.add_argument("--naics_header", default=os.path.join(REPO_ROOT, "src", "UrbanPopAgentStruct.H"),
                        help="Generated header whose naics_descriptions maps the .bin's NAICS indices to codes")
    parser.add_argument("--cbp_file", default=os.path.join(REPO_ROOT, "data", "UrbanPop", "cbp19st_derived.csv"),
                        help="CBP derived cache (see compute_workgroup_sizes.py)")
    parser.add_argument("--scale", default="auto",
                        help="Factor on every CBP target, or 'auto' to fit it to the actual number of groups")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the draws")
    parser.add_argument("--output", "-o", required=True, help="Output file of 'naics size' lines, one per workgroup")
    args = parser.parse_args()
    urbanpop = args.urbanpop or os.path.join(REPO_ROOT, "data", "UrbanPop", f"urbanpop_{args.state}.bin")
    epicast_sizes = args.epicast_sizes or os.path.join(REPO_ROOT, "data", "results", "emerge-paper", "epicast",
                                                       args.state, f"{args.state}_workgroup_sizes.txt")
    rng = np.random.default_rng(args.seed)

    actual = np.loadtxt(epicast_sizes, dtype=np.int64)
    print(f"Epicast actual: {len(actual):,} workgroups, {int(actual.sum()):,} members, from {epicast_sizes}")
    day, key = workgroup_members(urbanpop, args.events, args.naics_header, int(actual.sum()), rng)

    cells = pd.DataFrame({"day": day, "key": key}).groupby(["day", "key"]).size()
    cell_keys = cells.index.get_level_values("key").to_numpy()
    cell_workers = cells.to_numpy()
    targets = cbp_targets(np.unique(cell_keys), args.cbp_file, STATE_FIPS[args.state])
    cell_targets = np.array([targets[k] for k in cell_keys])
    scale = fit_scale(cell_workers, cell_targets, len(actual)) if args.scale == "auto" else float(args.scale)
    groups = n_groups(cell_workers, cell_targets, scale)
    print(f"{len(cells):,} (community, industry) cells; target scale {scale:.3f}"
          + (" (fitted to the actual group count)" if args.scale == "auto" else ""))

    # Every worker joins one of its cell's groups at random; groups nobody joined are dropped.
    first_group = np.cumsum(groups) - groups
    group_of_worker = (np.repeat(first_group, cell_workers)
                       + (rng.random(int(cell_workers.sum())) * np.repeat(groups, cell_workers)).astype(np.int64))
    sizes = np.bincount(group_of_worker, minlength=int(groups.sum()))
    group_keys = np.repeat(cell_keys, groups)
    keep = sizes > 0
    sizes, group_keys = sizes[keep], group_keys[keep]

    print(f"\n{'':<10}{'groups':>12}{'members':>13}{'mean':>8}{'size 1':>9}{'p90':>6}{'p99':>6}{'max':>6}{'KS':>8}")
    summarize("actual", actual)
    summarize("estimate", sizes, actual)

    with open(args.output, "w") as f:
        f.write(f"# Estimated Epicast workgroups (estimate_epicast_workgroups.py) from {args.events}, "
                f"target scale {scale:.3f}\n# naics size\n")
        for k, s in zip(group_keys, sizes):
            f.write(f"{k} {s}\n")
    print(f"\nWrote {len(sizes):,} workgroups to {args.output}")


if __name__ == "__main__":
    main()
