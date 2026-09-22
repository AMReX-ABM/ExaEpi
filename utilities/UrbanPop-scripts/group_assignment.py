"""Assign every agent its structural group memberships, for writing into the UrbanPop .bin.

These five attributes -- home neighborhood, household cluster, work neighborhood, work-group and
school class -- used to be drawn inside ExaEpi's C++ init (UrbanPopData::initAgents and
AgentContainer::assignSchoolClasses). They are properties of the synthetic population, not of a
particular simulation, so they belong here: computing them once, up front, means a .bin fully
determines its own population structure. Concretely that buys:

  * Rank invariance. The C++ versions drew from per-rank RNG streams and, for school classes, a
    rank-local atomic counter, so the same population came out differently on a different number
    of MPI ranks. Reading the values from the file cannot.

  * Work-groups sized like real workplaces. workgroup used to be a uniform draw over a count
    derived from block-group employment, so team sizes carried no information about the workplaces
    they sit in. Here establishments are drawn from the real CBP establishment-size distribution
    and then split into teams, so a five-person employer contributes a team of five rather than a
    fragment of a notional twenty-person one. The establishment is a device for that sizing, not a
    group anything mixes in -- ExaEpi has no workplace context, and none is written to the .bin.

  * Households intact by construction. The C++ drew a neighborhood per *agent* and then ran a
    second GPU kernel that forced each family onto its last member's draw by scanning forward --
    correct only while agents stay contiguous and ordered by (home cell, family), an invariant
    nothing checked. Dealing whole households removes both the kernel and the invariant.

Everything here is seeded from the caller's RNG, so a given (input, seed) pair always produces the
same .bin.
"""

import sys
from dataclasses import dataclass

import numpy as np
import polars as pl

# TRAVEL category index for "work from home" -- must match the TRAVEL enum in the generated
# UrbanPopAgentStruct.H, which is itself generated from the same categorical ordering.
TRAVEL_WFH = 7

# Agents not enrolled in a school get these, matching what assignSchoolClasses used to write for
# school_id == 0 (school_class 0, school_class_group -1 as the "no mixing bucket" sentinel).
NO_SCHOOL_CLASS = 0
NO_SCHOOL_CLASS_GROUP = -1

# Agents who do not go to a workplace get this for `work_group`, mirroring NO_SCHOOL_CLASS_GROUP.
# InteractionModWork.H never indexes on it: WorkCandidate already excludes everyone whose
# `workgroup` is 0, which is exactly this set.
NO_WORK_GROUP = -1


@dataclass
class GroupParams:
    """Targets that used to live in ExaEpi's TestParams (Utils.H). The defaults match the C++ ones
    they replace, except nborhood_size -- see below."""

    # Epicast's own neighborhood size, and not a free parameter as long as xmit_hood is Epicast's
    # too. The neighborhood interaction model is density-dependent by design -- it multiplies a
    # susceptible's escape probability once per infectious neighbor, with no size correction (see
    # InteractionModNborhood.H, which explains why it deliberately skips the one the community
    # model uses) -- so one infectious agent's expected neighborhood infections are xmit_hood *
    # (size - 1). Size is therefore a multiplier on this venue's contribution to R0, and xmit_hood
    # only means what Epicast measured it to mean at Epicast's size. The community model is the
    # opposite case: computeCommunitySizeScale divides its force of infection by community
    # population, making it frequency-dependent, so community size genuinely is free.
    #
    # This used to be 360, chosen so that a median block group (~1500 residents) split into 4
    # neighborhoods, matching the 4 that Epicast divides each of its fixed 2000-agent communities
    # into. But the count is not what the model reads -- nothing anywhere consumes "neighborhoods
    # per community", only the (community, neighborhood) grouping and hence its size -- and the
    # ratio that copying the count preserves has no mechanical meaning either, since the community
    # is frequency-dependent. Copying the count while inheriting xmit_hood ran this venue at
    # 359/499 = 0.72 of Epicast's, which a calibrated xmit_comm_scale would then quietly absorb
    # into the community.
    nborhood_size: int = 500
    workgroup_size: int = 20
    school_class_size: int = 20
    school_class_size_min: int = 5
    school_class_size_max: int = 50
    college_instructional_fraction: float = 0.1


def _round_half_up(x):
    """C++ amrex::Math::round semantics (half away from zero), not numpy/Python banker's rounding.

    Only ever called on non-negative sizes here, so half-up is enough.
    """
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def _check_fits_int16(name: str, values: np.ndarray):
    """The .bin stores these as int16 to keep the per-agent record small. Nothing in the data
    should come close to the limit, but a silent wraparound here would be an extremely confusing
    bug to chase down in the C++, so check rather than assume."""
    hi = int(values.max()) if len(values) else 0
    if hi > np.iinfo(np.int16).max:
        raise SystemExit(
            f"error: {name} reached {hi}, which does not fit in the int16 field used for it in "
            f"the UrbanPop .bin -- widen the column in upop_to_exaepi.py and regenerate "
            f"UrbanPopAgentStruct.H"
        )


# --------------------------------------------------------------------------------------------
# home neighborhood and household cluster
# --------------------------------------------------------------------------------------------


def assign_home_groups(df: pl.DataFrame, params: GroupParams, rng: np.random.Generator) -> pl.DataFrame:
    """Add `nborhood` and `hh_cluster`.

    nborhood splits a block group's residents into round(home_pop / nborhood_size) neighborhoods
    (at least one), drawn uniformly -- but per household, not per agent, so a household is never
    split across neighborhoods.

    hh_cluster is a straight port: household_id % ceil(n_households / 4). The modulo is strided on
    purpose rather than a consecutive division (household_id / 4): UrbanPop numbers households
    within a block group in an order that groups similar households together, so consecutive
    division would pile the large households into the same clusters, while striding spreads them.
    """
    n_households = pl.col("household_id").max().over("home_geoid") + 1
    num_clusters = ((n_households + 3) // 4).clip(lower_bound=1)

    home_pop = pl.len().over("home_geoid")
    max_nborhood = (home_pop / params.nborhood_size + 0.5).floor().cast(pl.Int64).clip(lower_bound=1)

    # one uniform draw per row, then collapsed to the household's first row -- gives every
    # household a single independent uniform draw without materializing a per-household frame
    df = df.with_columns(pl.Series("_u", rng.random(len(df))))
    df = df.with_columns(
        (pl.col("_u").first().over(["home_geoid", "household_id"]) * max_nborhood)
        .floor()
        .cast(pl.Int16)
        .alias("nborhood"),
        (pl.col("household_id") % num_clusters).cast(pl.Int16).alias("hh_cluster"),
    ).drop("_u")

    _check_fits_int16("nborhood", df["nborhood"].to_numpy())
    _check_fits_int16("hh_cluster", df["hh_cluster"].to_numpy())
    return df


# --------------------------------------------------------------------------------------------
# work neighborhood and work-group
# --------------------------------------------------------------------------------------------


def _apportion(weights: np.ndarray, total: int) -> np.ndarray:
    """Split `total` indivisible people across len(weights) groups in proportion to `weights`,
    giving every group at least one and summing to exactly `total`.

    One person per group is reserved up front, so no group can round away to nothing; the rest is
    largest-remainder, which keeps the totals exact without biasing which groups get the leftovers.
    """
    k = len(weights)
    spare = total - k
    share = weights / weights.sum()
    base = np.floor(share * spare).astype(np.int64) + 1
    rem = total - int(base.sum())
    if rem > 0:
        frac = share * spare - np.floor(share * spare)
        base[np.argsort(-frac)[:rem]] += 1
    return base


class _EstablishmentSampler:
    """Draws establishment sizes from the per-(state, NAICS) CBP size distribution.

    Batched, because the alternative -- one np.random.choice call per (work block group, NAICS)
    pair -- is hundreds of thousands of calls at state scale and dominates the runtime. Each
    (state, NAICS) key keeps a pool that is refilled in bulk and handed out in order; that is
    distributionally identical to drawing one at a time, since the draws are i.i.d.
    """

    _POOL_REFILL = 4096

    def __init__(self, dists, fallback_size_for, rng):
        self._dists = dists
        self._fallback_size_for = fallback_size_for
        self._rng = rng
        self._pools = {}
        self._cursors = {}
        self._means = {}
        self.missing_keys = set()

    def _refill(self, key, at_least):
        dist = self._dists.get(key)
        if dist is None:
            # No CBP size bands for this (state, NAICS): fall back to establishments of exactly
            # the industry's target work-group size. Recorded and reported by the caller rather
            # than silently absorbed -- it means this industry's workplaces have no real size
            # spread at all.
            self.missing_keys.add(key)
            sizes = np.array([self._fallback_size_for(key)], dtype=np.float64)
            probs = np.array([1.0])
        else:
            sizes, probs = dist
        n = max(self._POOL_REFILL, int(at_least))
        self._pools[key] = self._rng.choice(sizes, size=n, p=probs).astype(np.int64)
        self._cursors[key] = 0
        self._means[key] = float(np.dot(sizes, probs))

    def _take(self, key, n):
        if key not in self._pools or self._cursors[key] + n > len(self._pools[key]):
            self._refill(key, n)
        c = self._cursors[key]
        self._cursors[key] = c + n
        return self._pools[key][c : c + n]

    def _mean(self, key):
        if key not in self._means:
            self._refill(key, 1)
        return self._means[key]

    def partition(self, key, population: int) -> np.ndarray:
        """Split `population` workers into establishment sizes summing to exactly that.

        Picks the establishment count first -- round(population / mean size), at least one and at
        most one per worker -- then draws that many sizes and apportions the workers among them in
        proportion. Deliberately *not* "draw sizes until the running total covers the population,
        then truncate the last one to fit": that is a renewal process, and it always needs one
        extra draw to cross the finish line, so it produces about one spurious establishment per
        (block group, NAICS) group. Measured on NM that was a 52% overcount (101,870 rather than
        67,200 establishments) and it halved the mean establishment size, because the spurious
        ones are all small.

        Note the floor of one establishment per group is not a rounding convenience: a block group
        holding three workers in some industry really does hold one small workplace, not a
        fragment of a large one somewhere else. It is also why the establishment count here
        exceeds the CBP count for the same employment -- block-group granularity fragments what
        CBP counts as single large employers.
        """
        mean = self._mean(key)
        k = max(1, int(_round_half_up(population / mean)))
        k = min(k, population)
        sizes = self._take(key, k).astype(np.float64)
        return _apportion(sizes, population)


def assign_work_groups(
    df: pl.DataFrame,
    params: GroupParams,
    naics_codes: list,
    workgroup_targets: dict,
    est_size_dists: dict,
    default_target: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Add `workgroup` and `work_group`.

    `workgroup` is the team id within one (work block group, NAICS) pair; `work_group` is a
    globally dense id for the same team, numbered across the whole population. The two carry
    identical information -- work_group is a bijection with the (work block group, NAICS,
    workgroup) triple -- but only work_group is a single small integer.

    That matters because the (block group, NAICS, workgroup) triple is what transmission is keyed
    on, and InteractionModWork.H used to tally it into an array indexed by that triple directly:
    max_communities * max_workgroup * max_naics entries, the last two being global maxima. At CA
    scale that is 43M slots per box holding at most ~133k workers -- 0.3% occupied, ~173 MB, zeroed
    every timestep -- because every community gets room for the busiest (community, NAICS) pair in
    the country across all 251 NAICS codes. Indexing on work_group instead makes the array one
    entry per team that actually exists. This is the same reason school_class_group exists
    alongside school_class (see assign_school_groups and InteractionModSchool.H).

    Only agents who physically go to a workplace get a real work-group: employed (naics != -1),
    not an educator (school_id == 0) and not declared work-from-home. The two exclusions are
    carried over from the C++ verbatim, and both matter:

      * Educators already mix through school_class_group, which covers every school_id > 0 agent
        in properly class-sized buckets. Routing them through the work channels as well
        double-counts their contacts, and for a large school (a university with thousands of
        staff) collapses them into one giant undifferentiated pool.

      * Work-from-home agents have an assigned work_geoid they never actually visit, so its
        population is not a meaningful size for a group they never physically join.

    Both get workgroup 0 ("not working" to the work interaction model), as do the unemployed.

    The establishments drawn here are not themselves a transmission context and do not survive
    into the .bin: nothing in ExaEpi groups agents by workplace. What they determine is how many
    teams a given body of workers is split into, and how big those teams are -- a five-worker
    establishment yields one team of five, not a fragment of a twenty-person one -- which is why
    they are drawn from the real CBP size distribution rather than assumed.
    """
    n = len(df)
    naics = df["naics"].to_numpy()
    school_id = df["school_id"].to_numpy()
    travel = df["travel"].to_numpy()
    work_geoid = df["work_geoid"].to_numpy()

    # default: not at a workplace -- no work-group
    workgroup = np.zeros(n, dtype=np.int64)
    work_group = np.full(n, NO_WORK_GROUP, dtype=np.int64)

    eligible = np.flatnonzero((naics != -1) & (school_id == 0) & (travel != TRAVEL_WFH))
    if len(eligible) == 0:
        return df.with_columns(
            pl.Series("workgroup", workgroup, dtype=pl.Int16),
            pl.Series("work_group", work_group, dtype=pl.Int32),
        )

    # Shuffle before grouping, then rely on lexsort being stable: workers end up in a random
    # order within each (work block group, NAICS) run, so slicing them into establishments below
    # does not accidentally group co-workers by home block group or by agent id.
    eligible = rng.permutation(eligible)
    order = np.lexsort((naics[eligible], work_geoid[eligible]))
    sidx = eligible[order]
    s_geoid = work_geoid[sidx]
    s_naics = naics[sidx]
    n_elig = len(sidx)

    # leading 2 digits of the 12-digit GEOID are the state FIPS
    s_state = (s_geoid // 10_000_000_000).astype(np.int64)

    def target_for(state_fips, naics_idx):
        if not (0 <= naics_idx < len(naics_codes)):
            return default_target
        return workgroup_targets.get((int(state_fips), naics_codes[naics_idx]), default_target)

    sampler = _EstablishmentSampler(
        est_size_dists, lambda key: max(1, workgroup_targets.get(key, default_target)), rng
    )

    # contiguous runs of (work block group, NAICS): each sets its own establishments and the
    # teams inside them
    grp_break = np.empty(n_elig, dtype=bool)
    grp_break[0] = True
    grp_break[1:] = (s_geoid[1:] != s_geoid[:-1]) | (s_naics[1:] != s_naics[:-1])
    grp_starts = np.flatnonzero(grp_break)
    grp_ends = np.append(grp_starts[1:], n_elig)

    out_workgroup = np.empty(n_elig, dtype=np.int64)
    out_work_group = np.empty(n_elig, dtype=np.int64)

    # running count of teams emitted so far, which is what makes work_group dense and global:
    # each (block group, NAICS) group's 1-based team ids are simply shifted past every team
    # already numbered
    n_work_groups = 0

    n_establishments = 0
    for g in range(len(grp_starts)):
        lo, hi = grp_starts[g], grp_ends[g]
        pop = hi - lo
        state = s_state[lo]
        naics_idx = s_naics[lo]
        key = (int(state), naics_codes[naics_idx]) if 0 <= naics_idx < len(naics_codes) else (int(state), "")

        est_sizes = sampler.partition(key, pop)
        n_establishments += len(est_sizes)
        est_of_worker = np.repeat(np.arange(len(est_sizes)), est_sizes)
        est_start = np.cumsum(est_sizes) - est_sizes
        pos_in_est = np.arange(pop) - est_start[est_of_worker]

        # split each establishment into teams at the industry's target size, balanced to within
        # one worker of each other; team ids are numbered densely across the whole (block group,
        # NAICS) group because that is the key the work interaction model indexes on
        target = max(1, target_for(state, naics_idx))
        n_teams = np.maximum(1, _round_half_up(est_sizes / target)).astype(np.int64)
        team_base = np.cumsum(n_teams) - n_teams
        # workgroup 0 means "not working", so real work-groups are 1-based
        out_workgroup[lo:hi] = team_base[est_of_worker] + (pos_in_est % n_teams[est_of_worker]) + 1

        # The same teams, numbered globally instead of per group. Every team id in 1..n_teams.sum()
        # is used -- n_teams is never larger than its establishment's size, so `pos_in_est %
        # n_teams` reaches every team -- which is what keeps this dense rather than merely unique.
        out_work_group[lo:hi] = n_work_groups + out_workgroup[lo:hi] - 1
        n_work_groups += int(n_teams.sum())

    workgroup[sidx] = out_workgroup
    work_group[sidx] = out_work_group

    if sampler.missing_keys:
        print(
            f"  warning: {len(sampler.missing_keys)} (state, NAICS) pairs had no establishment-size "
            f"bands; their workplaces were all sized at the target work-group size",
            file=sys.stderr,
        )
    print(f"  {len(eligible)} workers in {n_establishments} establishments, {n_work_groups} work-groups")

    _check_fits_int16("workgroup", workgroup)
    if n_work_groups > np.iinfo(np.int32).max:
        raise SystemExit("error: work_group overflowed the int32 field used for it in the .bin")
    return df.with_columns(
        pl.Series("workgroup", workgroup, dtype=pl.Int16),
        pl.Series("work_group", work_group, dtype=pl.Int32),
    )


# --------------------------------------------------------------------------------------------
# daytime neighborhood
# --------------------------------------------------------------------------------------------


def _dense_ids(offset: int, *cols: np.ndarray):
    """Dense ids `offset, offset+1, ...` for the distinct rows of the given columns, as
    (ids, count).

    Columns are folded in one at a time, re-densifying at every step, rather than packed into one
    integer by shifting each by a fixed width. Packing needs a width per column that is wider than
    that column's real range, and getting one of those widths wrong does not fail -- it silently
    merges rows that differ only in the overflowed column, which here would mean two groups in
    different block groups quietly becoming one. Folding needs no such assumption: after each step
    the running id is already dense, so it is bounded by the row count no matter what the columns
    contain.
    """
    ids = np.zeros(len(cols[0]), dtype=np.int64)
    n = 1
    for col in cols:
        col = np.asarray(col, dtype=np.int64)
        _, ids = np.unique(ids * (int(col.max()) + 1 if len(col) else 1) + col, return_inverse=True)
        ids = ids.astype(np.int64)
        n = int(ids.max()) + 1 if len(ids) else 0
    return ids + offset, n


def _pack_atoms_into_bins(atom_geo, atom_size, capacity, rng):
    """Fill atoms into bins of `capacity` members, bin ids numbered from 0 within each block
    group. An atom is a group that must not be split across bins, and is never split here; one
    bigger than `capacity` on its own gets a bin to itself.

    The remaining atoms are shuffled and then cut on an evenly spaced grid of running totals, each
    atom going to the bin its own MIDPOINT falls in. Cutting on the midpoint rather than on the
    atom's leading edge is what keeps bin sizes centered on the target: a leading-edge cut can
    only ever overshoot (the atom that straddles a boundary always lands in the lower bin), so
    every bin comes out at the spacing plus part of one atom, whereas the midpoint rule sends a
    straddling atom whichever way it leans and leaves the error symmetric -- spacing +/- half an
    atom instead of spacing + a whole one.

    The grid is spaced at total/round(total/capacity) within each block group, not at `capacity`
    flat. Both give bins of about `capacity`, but a flat grid divides a block group into a whole
    number of full bins plus whatever is left over, so every block group ends up with one
    undersized remainder bin (averaging around half the target). Dividing the total into
    round(total/capacity) even parts spreads that shortfall across all of them instead, which is
    also exactly how assign_home_groups splits a block group's residents at night, so day and
    night neighborhoods come out of the same rule.

    A grid is used rather than a true bin-packing heuristic (first-fit-decreasing and friends)
    because the error is already small at these sizes -- atoms are mostly work-groups and
    households, an order of magnitude below capacity -- and because the grid is a handful of
    vectorized numpy operations over every atom in the country at once. First-fit needs a
    per-atom search over open bins, which at tens of millions of atoms is not something to run in
    a Python loop.

    Shuffling first matters: the input arrives grouped by industry and by household, so cutting it
    in place would fill each bin from a single industry (and put neighbors in the same bin purely
    because their household ids are adjacent), which is exactly the structure a neighborhood is
    not supposed to have.
    """
    n = len(atom_size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    oversized = atom_size > capacity
    # Sort by block group; within one, oversized atoms first (they take the leading bins), then
    # the rest in random order.
    order = np.lexsort((rng.random(n), ~oversized, atom_geo))
    geo_s = atom_geo[order]
    size_s = atom_size[order]
    over_s = oversized[order]

    is_run_start = np.empty(n, dtype=bool)
    is_run_start[0] = True
    is_run_start[1:] = geo_s[1:] != geo_s[:-1]
    run_start = np.flatnonzero(is_run_start)
    run_id = np.cumsum(is_run_start) - 1
    pos_in_run = np.arange(n) - run_start[run_id]

    n_over = np.bincount(run_id, weights=over_s, minlength=len(run_start)).astype(np.int64)

    # Running total within the block group, counting only the atoms being packed onto the grid
    # (an oversized atom has its own bin and must not push the others' totals along).
    packed_size = np.where(over_s, 0, size_s)
    excl_prefix = np.cumsum(packed_size) - packed_size
    run_base = np.zeros(len(run_start), dtype=excl_prefix.dtype)
    run_base[1:] = excl_prefix[run_start[1:]]
    midpoint = (excl_prefix - run_base[run_id]) + packed_size / 2.0

    packed_total = np.bincount(run_id, weights=packed_size, minlength=len(run_start))
    n_bins = np.maximum(1, _round_half_up(packed_total / capacity)).astype(np.int64)
    spacing = np.where(packed_total > 0, packed_total / n_bins, 1.0)
    # Clamped: the very last atom's midpoint can round up to n_bins itself when its half-size
    # pushes it past the final edge.
    on_grid = np.minimum(np.floor(midpoint / spacing[run_id]).astype(np.int64), n_bins[run_id] - 1)

    bin_s = np.where(over_s, pos_in_run, n_over[run_id] + on_grid)     # oversized: one bin each

    # bin_s is nondecreasing within a run by construction, so renumbering it densely is just
    # counting the distinct values seen so far -- no second sort. Dense ids matter because the
    # C++ sizes its per-community neighborhood array from the largest id in use (getMaxGroup in
    # InteractionModNborhood.H), so a gap would cost memory in every community.
    is_new_bin = np.empty(n, dtype=bool)
    is_new_bin[0] = True
    is_new_bin[1:] = (run_id[1:] != run_id[:-1]) | (bin_s[1:] != bin_s[:-1])
    dense = np.cumsum(is_new_bin) - 1
    dense -= dense[run_start][run_id]

    out = np.empty(n, dtype=np.int64)
    out[order] = dense
    return out


def assign_day_neighborhoods(df: pl.DataFrame, params: GroupParams, rng: np.random.Generator) -> pl.DataFrame:
    """Add `work_nborhood`: the neighborhood an agent mixes in during the DAY, at wherever it is
    they spend it.

    Sized to the same nborhood_size target as the home neighborhood (assign_home_groups), so a
    neighborhood means the same thing by day as by night. That is not what falls out of the
    daytime population on its own: a block group's daytime headcount is its commuters, its
    students and their teachers, and whichever of its own residents stayed home, and none of those
    are related to the count of home neighborhoods it was divided into. Splitting daytime
    population on a home-derived count leaves neighborhood sizes ranging over orders of magnitude
    -- an employment centre draws in far more people by day than live there, a bedroom block group
    far fewer -- and since a neighborhood's force of infection scales with how many infectious
    people are in it, that spread is not a neutral bookkeeping detail: it makes daytime
    neighborhood transmission concentrate in the largest daytime neighborhoods, which are
    precisely the ones an epidemic reaches first.

    Three kinds of group are kept whole, because being in one already means spending the day
    together:

      * a school -- every student and educator at it, in one neighborhood
      * a work-group -- the whole team, in one neighborhood
      * a household of people who stayed home, matching how assign_home_groups deals whole
        households into home neighborhoods

    A school with more members than nborhood_size cannot be both intact and within the target --
    a university has tens of thousands -- so it goes in by class group instead (school_class_group:
    a classroom, or one of the pools of excess teachers, and what InteractionModSchool.H actually
    mixes within). Its classes each stay whole and land in a neighborhood of the ordinary size;
    what is given up is only that the campus as a whole is no longer one neighborhood, which at
    that size it could never have been without abandoning the target entirely. Keeping it intact
    instead would leave the average agent in a daytime neighborhood several times the target,
    concentrated exactly on the school-age population -- the opposite of what sizing them is for.

    That leaves one case where the target genuinely cannot be met: a single class group (or
    work-group, if one were ever configured above the target) bigger than a whole neighborhood.
    Those are reported at the end.

    The ids are a fresh dense space per block group, unrelated to `nborhood`. They used to be the
    home neighborhood's own ids -- agents who stayed home kept theirs, and workers drew from a
    range sized by the employed population -- which quietly merged unrelated groups whenever the
    numbers collided, and made the count of daytime neighborhoods a function of the wrong
    population. Day and night are separate attributes on the agent (`work_nborhood` vs
    `nborhood`, read by InteractionModNborhood.H's day and night instances), so they have no
    reason to share a numbering.
    """
    capacity = params.nborhood_size
    n = len(df)
    school_id = df["school_id"].to_numpy()
    naics = df["naics"].to_numpy()
    home_geoid = df["home_geoid"].to_numpy()
    work_geoid = df["work_geoid"].to_numpy()
    household_id = df["household_id"].to_numpy().astype(np.int64)
    workgroup = df["workgroup"].to_numpy()
    school_class_group = df["school_class_group"].to_numpy()

    at_school = school_id != 0
    at_work = workgroup > 0  # workgroup 0 is assign_work_groups' "not at a workplace"
    stay_home = ~at_school & ~at_work

    # Where the day is actually spent. Everyone commutes to work_geoid (AgentContainer.cpp's
    # moveAgentsToWork), including students and educators, whose work_geoid is their school's
    # block group -- except the work-from-home and the unemployed, whom UrbanPopData.cpp keeps at
    # home_geoid. (For the unemployed the two are equal anyway; it is asserted on load.)
    day_geoid = np.where(stay_home, home_geoid, work_geoid)
    geo_idx, n_geo = _dense_ids(0, day_geoid)

    # A school too big for a neighborhood goes in class group by class group instead of whole.
    school_of, n_schools = _dense_ids(0, geo_idx[at_school], school_id[at_school])
    big_school = np.bincount(school_of, minlength=n_schools) > capacity
    by_class = np.zeros(n, dtype=bool)
    by_class[at_school] = big_school[school_of]
    whole_school = at_school & ~by_class

    # One id per atom, numbered so that no two kinds share one. Each is keyed exactly the way the
    # interaction model that owns it indexes one -- (community, workgroup, naics) for a work-group
    # (InteractionModWork.H), school_class_group for a class (InteractionModSchool.H, where it is
    # already a globally dense id) -- so an atom is one whole transmission group, never part of one.
    atom = np.full(n, -1, dtype=np.int64)
    n_atoms = 0
    atom[at_work], k = _dense_ids(n_atoms, geo_idx[at_work], naics[at_work], workgroup[at_work])
    n_atoms += k
    atom[whole_school], k = _dense_ids(n_atoms, geo_idx[whole_school], school_id[whole_school])
    n_atoms += k
    atom[by_class], k = _dense_ids(n_atoms, school_class_group[by_class])
    n_atoms += k
    atom[stay_home], k = _dense_ids(n_atoms, geo_idx[stay_home], household_id[stay_home])
    n_atoms += k

    atom_size = np.bincount(atom, minlength=n_atoms)
    atom_geo = np.zeros(n_atoms, dtype=geo_idx.dtype)
    atom_geo[atom] = geo_idx  # every member of an atom shares its block group, so any wins

    atom_bin = _pack_atoms_into_bins(atom_geo, atom_size, capacity, rng)
    work_nborhood = atom_bin[atom]

    # Per-neighborhood headcounts, and which neighborhoods are a single group too big to fit.
    # Those are the only ones that miss the target for a structural reason -- every other
    # neighborhood is the target give or take part of one group, in both directions, so simply
    # counting the ones above `capacity` would report roughly half of them as exceptions.
    key = geo_idx.astype(np.int64) * (work_nborhood.max() + 1) + work_nborhood
    sizes = np.bincount(key)
    indivisible = np.bincount(
        atom_geo.astype(np.int64) * (work_nborhood.max() + 1) + atom_bin,
        weights=atom_size > capacity,
        minlength=len(sizes),
    )[: len(sizes)]
    nonempty = sizes > 0
    sizes, indivisible = sizes[nonempty], indivisible[nonempty] > 0
    ordinary = sizes[~indivisible]
    print(
        f"  {n_atoms} daytime groups packed into {len(sizes)} neighborhoods across {n_geo} block "
        f"groups: mean {ordinary.mean():.1f}, 5th-95th pct {np.percentile(ordinary, 5):.0f}-"
        f"{np.percentile(ordinary, 95):.0f} (target {capacity})"
    )
    n_big_school = int(big_school.sum())
    if n_big_school:
        print(f"  {n_big_school} schools larger than {capacity} went in by class group")
    if indivisible.any():
        over = sizes[indivisible]
        print(
            f"  {indivisible.sum()} ({100 * indivisible.sum() / len(sizes):.2f}%) hold one group too "
            f"big to fit: median {np.median(over):.0f}, max {over.max()}"
        )

    _check_fits_int16("work_nborhood", work_nborhood)
    return df.with_columns(pl.Series("work_nborhood", work_nborhood, dtype=pl.Int16))


# --------------------------------------------------------------------------------------------
# school classes
# --------------------------------------------------------------------------------------------

_U64 = np.uint64


def _hash64(a, b):
    """Port of AgentContainer.cpp's hash64, used to spread students above the guaranteed class
    floor. Deterministic, so classes are reproducible; not a simple function of rank, so class
    sizes are not all forced to the same value (see assign_school_groups)."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    h = a * _U64(2654435761) + b * _U64(0x9E3779B97F4A7C15) + _U64(0x9E3779B97F4A7C15)
    h ^= h >> _U64(33)
    h *= _U64(0xFF51AFD7ED558CCD)
    h ^= h >> _U64(33)
    h *= _U64(0xC4CEB9FE1A85EC53)
    h ^= h >> _U64(33)
    return h


def assign_school_groups(df: pl.DataFrame, params: GroupParams) -> pl.DataFrame:
    """Add `school_class` and `school_class_group` for every school_id > 0 agent.

    A "raw group" is one (work block group, school_id, grade) combination -- school_id is only
    unique within a block group, which is why the location is part of the key. Each raw group gets:

      * One class per teacher actually present, at most, clamped so no class averages fewer than
        school_class_size_min or more than school_class_size_max students. College raw groups
        first scale their headcount by college_instructional_fraction, because that headcount
        comes from total college employment rather than a faculty-specific count. A raw group with
        students but no identified teachers falls back to ceil(students / school_class_size).

      * Teachers beyond one per class pooled into "admin" groups sized like a regular work-group,
        marked school_class -2, -3, -4, ... -- one sentinel per admin group. The school
        interaction model recognizes an admin-group member from that value alone.

    Students are placed in two phases. The first school_class_size_min * n_classes of them go
    round-robin, which guarantees every class its floor; the rest are hash-smeared on top. Neither
    half alone works: pure round-robin balances classes to within one student, which makes every
    raw group with a similar teacher:student ratio land on almost exactly the same class size
    (observed as colleges piling up at 40-41 students and childcare at 6-7), while pure random
    assignment has enough multinomial variance to push a real fraction of classes below the floor.

    school_class_group is a globally dense id for the actual mixing bucket, and is what the
    interaction model indexes on. Assigning it here rather than in the C++ makes it independent of
    the simulation's grid and rank count.
    """
    n = len(df)
    school_id = df["school_id"].to_numpy()
    naics = df["naics"].to_numpy()
    grade = df["grade"].to_numpy().astype(np.int64)
    work_geoid = df["work_geoid"].to_numpy()

    school_class = np.full(n, NO_SCHOOL_CLASS, dtype=np.int64)
    school_class_group = np.full(n, NO_SCHOOL_CLASS_GROUP, dtype=np.int64)

    enrolled = np.flatnonzero(school_id > 0)
    if len(enrolled) == 0:
        return df.with_columns(
            pl.Series("school_class", school_class, dtype=pl.Int16),
            pl.Series("school_class_group", school_class_group, dtype=pl.Int32),
        )

    order = np.lexsort((grade[enrolled], school_id[enrolled], work_geoid[enrolled]))
    sidx = enrolled[order]
    s_geoid = work_geoid[sidx]
    s_school = school_id[sidx]
    s_grade = grade[sidx]
    s_is_student = naics[sidx] == -1
    n_enrolled = len(sidx)

    brk = np.empty(n_enrolled, dtype=bool)
    brk[0] = True
    brk[1:] = (s_geoid[1:] != s_geoid[:-1]) | (s_school[1:] != s_school[:-1]) | (s_grade[1:] != s_grade[:-1])
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], n_enrolled)

    out_class = np.empty(n_enrolled, dtype=np.int64)
    out_group = np.empty(n_enrolled, dtype=np.int64)

    next_group_id = 0
    n_admin_total = 0
    for g in range(len(starts)):
        lo, hi = starts[g], ends[g]
        is_student = s_is_student[lo:hi]
        student_count = int(is_student.sum())
        teacher_count = (hi - lo) - student_count

        n_classes = 0
        if student_count > 0:
            # grade is part of the raw-group key, so it is constant across the group
            is_college = s_grade[lo] > 17
            eff_teachers = teacher_count * params.college_instructional_fraction if is_college else float(teacher_count)
            if eff_teachers > 0.0:
                raw_n_classes = max(1, int(eff_teachers))
            else:
                raw_n_classes = max(1, -(-student_count // params.school_class_size))
            lower = -(-student_count // params.school_class_size_max)
            upper = max(1, student_count // params.school_class_size_min)
            n_classes = max(lower, min(raw_n_classes, upper))
        # else: no students means no real classes -- any educators recorded here (a school_id
        # shared by non-teaching staff, say) go entirely to the admin groups below rather than
        # into a fake class with a homeroom teacher and nobody to teach

        excess_teachers = teacher_count - n_classes
        n_admin = -(-excess_teachers // params.workgroup_size) if excess_teachers > 0 else 0
        n_admin_total += n_admin
        base = next_group_id
        next_group_id += n_classes + n_admin

        # rank within the raw group, separately for students and teachers
        rank = np.empty(hi - lo, dtype=np.int64)
        rank[is_student] = np.arange(student_count)
        rank[~is_student] = np.arange(teacher_count)

        cls = np.empty(hi - lo, dtype=np.int64)

        # teachers: the first n_classes get a class each; the rest fill the admin groups
        # round-robin so those come out balanced rather than one taking the whole remainder
        t_rank = rank[~is_student]
        t_cls = np.empty(teacher_count, dtype=np.int64)
        homeroom = t_rank < n_classes
        t_cls[homeroom] = t_rank[homeroom]
        if n_admin > 0:
            admin_group = (t_rank[~homeroom] - n_classes) % n_admin
            t_cls[~homeroom] = -2 - admin_group
        cls[~is_student] = t_cls

        if student_count > 0:
            s_rank = rank[is_student]
            base_total = params.school_class_size_min * n_classes
            smeared = (_hash64(np.full(student_count, g, dtype=np.uint64), s_rank.astype(np.uint64)) % _U64(n_classes)).astype(
                np.int64
            )
            cls[is_student] = np.where(s_rank < base_total, s_rank % n_classes, smeared)

        out_class[lo:hi] = cls
        # real classes are numbered from the group's base; admin groups follow them, recovered
        # from the -2, -3, ... sentinel
        out_group[lo:hi] = np.where(cls >= 0, base + cls, base + n_classes + (-2 - cls))

    school_class[sidx] = out_class
    school_class_group[sidx] = out_group

    # the core invariant: a real class never gets two homeroom teachers
    teacher_groups = out_group[(~s_is_student) & (out_class >= 0)]
    if len(teacher_groups):
        _, counts = np.unique(teacher_groups, return_counts=True)
        if counts.max() > 1:
            raise SystemExit("error: a real school class ended up with more than one homeroom teacher")

    print(f"  {len(enrolled)} enrolled agents in {next_group_id} class groups ({n_admin_total} admin)")

    if next_group_id > np.iinfo(np.int32).max:
        raise SystemExit("error: school_class_group overflowed the int32 field used for it in the .bin")
    _check_fits_int16("school_class", np.abs(school_class))
    return df.with_columns(
        pl.Series("school_class", school_class, dtype=pl.Int16),
        pl.Series("school_class_group", school_class_group, dtype=pl.Int32),
    )
