"""Assign every agent its structural group memberships, for writing into the UrbanPop .bin.

These five attributes -- home neighborhood, household cluster, work neighborhood, work-group and
school class -- used to be drawn inside ExaEpi's C++ init (UrbanPopData::initAgents and
AgentContainer::assignSchoolClasses). They are properties of the synthetic population, not of a
particular simulation, so they belong here: computing them once, up front, means a .bin fully
determines its own population structure. Concretely that buys:

  * Rank invariance. The C++ versions drew from per-rank RNG streams and, for school classes, a
    rank-local atomic counter, so the same population came out differently on a different number
    of MPI ranks. Reading the values from the file cannot.

  * Coherent work tiers. workgroup and work_nborhood were independent uniform draws, so a
    worker's team and the workplace containing it were unrelated. Here they come from one pass:
    establishments are drawn from the real CBP establishment-size distribution, each establishment
    lands in one work neighborhood, and each is then split into teams. That is the two-tier
    structure (work-groups bundled into workplaces) the model is supposed to have.

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


@dataclass
class GroupParams:
    """Targets that used to live in ExaEpi's TestParams (Utils.H). Defaults match the C++ ones
    they replace, so a .bin built without overrides reproduces the old targets."""

    nborhood_size: int = 360
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
    """Add `workgroup` and `work_nborhood`, assigned together from one establishment draw.

    Only agents who physically go to a workplace get a real work-group: employed (naics != -1),
    not an educator (school_id == 0) and not declared work-from-home. The two exclusions are
    carried over from the C++ verbatim, and both matter:

      * Educators already mix through school_class_group, which covers every school_id > 0 agent
        in properly class-sized buckets. Routing them through the work channels as well
        double-counts their contacts, and for a large school (a university with thousands of
        staff) collapses them into one giant undifferentiated pool.

      * Work-from-home agents have an assigned work_geoid they never actually visit, so its
        population is not a meaningful size for a group they never physically join.

    Both get workgroup 0 ("not working" to the work interaction model) and their home neighborhood
    as their work neighborhood, as do the unemployed -- everyone who is not at a workplace mixes
    in their home neighborhood during the day.

    The number of work neighborhoods a block group is split into is deliberately still taken from
    its whole employed population -- every agent whose assigned work_geoid is this block group,
    educators and work-from-home included -- rather than from the commuters who actually show up.
    Counting only the latter is arguably more correct, but work_nborhood shares an id space per
    community with the home neighborhoods of everyone who stays home during the day, so shrinking
    the count packs workers into ids that more residents also occupy. Measured on NM that lifted
    daytime neighborhood transmission by 39% on its own. That is a separate modelling question
    from work-group formation, so it is left alone here: what changes is only *which* work
    neighborhood a worker lands in, now the same one as the rest of their establishment.
    """
    n = len(df)
    naics = df["naics"].to_numpy()
    school_id = df["school_id"].to_numpy()
    travel = df["travel"].to_numpy()
    work_geoid = df["work_geoid"].to_numpy()
    nborhood = df["nborhood"].to_numpy().astype(np.int64)

    # default: not at a workplace -- no work-group, and daytime mixing in the home neighborhood
    workgroup = np.zeros(n, dtype=np.int64)
    work_nborhood = nborhood.copy()

    # how many work neighborhoods each work block group is split into, over its whole employed
    # population (see the docstring for why not just the commuters who show up)
    employed_geoid, employed_count = np.unique(work_geoid[naics != -1], return_counts=True)
    geoid_n_work_nborhoods = dict(
        zip(employed_geoid.tolist(), np.maximum(1, _round_half_up(employed_count / params.nborhood_size)).astype(np.int64))
    )

    eligible = np.flatnonzero((naics != -1) & (school_id == 0) & (travel != TRAVEL_WFH))
    if len(eligible) == 0:
        return df.with_columns(
            pl.Series("workgroup", workgroup, dtype=pl.Int16),
            pl.Series("work_nborhood", work_nborhood, dtype=pl.Int16),
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
    out_work_nborhood = np.empty(n_elig, dtype=np.int64)

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

        # every establishment sits in exactly one work neighborhood -- that is what makes a
        # work-group and the workplace containing it consistent with each other
        max_wn = geoid_n_work_nborhoods[int(s_geoid[lo])]
        out_work_nborhood[lo:hi] = rng.integers(0, max_wn, size=len(est_sizes))[est_of_worker]

        # split each establishment into teams at the industry's target size, balanced to within
        # one worker of each other; team ids are numbered densely across the whole (block group,
        # NAICS) group because that is the key the work interaction model indexes on
        target = max(1, target_for(state, naics_idx))
        n_teams = np.maximum(1, _round_half_up(est_sizes / target)).astype(np.int64)
        team_base = np.cumsum(n_teams) - n_teams
        # workgroup 0 means "not working", so real work-groups are 1-based
        out_workgroup[lo:hi] = team_base[est_of_worker] + (pos_in_est % n_teams[est_of_worker]) + 1

    workgroup[sidx] = out_workgroup
    work_nborhood[sidx] = out_work_nborhood

    if sampler.missing_keys:
        print(
            f"  warning: {len(sampler.missing_keys)} (state, NAICS) pairs had no establishment-size "
            f"bands; their workplaces were all sized at the target work-group size",
            file=sys.stderr,
        )
    print(f"  {len(eligible)} workers in {n_establishments} establishments")

    _check_fits_int16("workgroup", workgroup)
    _check_fits_int16("work_nborhood", work_nborhood)
    return df.with_columns(
        pl.Series("workgroup", workgroup, dtype=pl.Int16),
        pl.Series("work_nborhood", work_nborhood, dtype=pl.Int16),
    )


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
