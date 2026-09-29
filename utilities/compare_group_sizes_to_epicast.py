#!/usr/bin/env python

"""Compare Epicast vs. ExaEpi distributions of workgroup size, school-class size, and
school size for the CA run used in the emerge paper.

Epicast side: data/results/emerge-paper/epicast/ca/ca_{workgroup,schoolgroup,school}_sizes.txt
-- plain text, one integer (group size) per line. "workgroup" is a workplace peer group,
"schoolgroup" is a classroom-level cohort, "school" is a whole school. Epicast's schools run
from preschool (UrbanPop's own preschoolers) to 12th grade, with no childcare or colleges
(Epicast 2.0 sec. 2.4.1). Its school-groups appear to count staff as well as students: every
school member is in exactly one of them (both files total 8.51M on CA, 461k on NM), about 1.3M more
than UrbanPop's preschool-12 students on CA -- roughly its NAICS 611 workers, whom Epicast makes
the teachers, one per school-group, pooling the rest into an administrative school-group per
school. ExaEpi's class sizes count students only (see below).

ExaEpi side: computed straight from the UrbanPop .bin ExaEpi reads its agents from (default
data/UrbanPop/urbanpop_ca.bin), so no ExaEpi run -- and so no cases file -- is needed. The
group structure is all in the file (see UrbanPopData::initAgents in src/UrbanPopData.cpp), and
this reproduces the tally ExaEpi itself does in AgentContainer::computeGroupSizeDistributions
(the <prefix>_workgroup_sizes.txt / _class_sizes.txt / _school_sizes.txt it writes when
--aggregated_diag_int is enabled). An agent's work community is the block group of its
work_geoid, except for a declared work-from-home non-educator, who stays in its home block group:

  - Workgroup size: agents with workgroup > 0 (0 means not assigned to a workgroup --
    not working, or working from home), grouped by (work community, naics, workgroup).
    Workgroup IDs are only unique within a (work community, naics) pair -- see
    InteractionModWork.H's max_workgroup * max_naics sizing -- so all three keys are
    needed to recover each actual workgroup.

  - School class size: agents with naics == -1 (a student, not an employee) and
    school_class_group >= 0 (enrolled in a real classroom, not the -1 sentinel for
    unenrolled agents), grouped by school_class_group alone -- that field is already a
    globally unique, densely-packed ID for one (community, school_id, grade, class)
    mixing bucket (see UrbanPop-scripts/group_assignment.py / InteractionModSchool.H).
    Restricting to naics == -1 excludes each class's own homeroom teacher and any
    non-classroom "admin" pools of surplus teachers (which contain no naics == -1
    agents at all), matching the student headcount ExaEpi's own
    "School class size" log histogram reports.

  - School size: agents with school_id > 0 (both students and staff), grouped by
    (work community, school_id) -- school_id is only unique within a community, like
    workgroup.

The class and school panels show K-12 (with preschool) by default, matching Epicast, which has no
childcare or colleges; --childcare and --colleges add each as a series of its own. They also get a
reference curve straight from the schools data UrbanPop's allocation reads: each K-12 school's
student-teacher ratio (load_school_class_sizes), and each school's listed students plus staff
(load_school_sizes), for childcare and colleges too when shown. The workgroup panel can get real
establishment sizes from CBP (--cbp_state).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plos_compbio_style import apply_style, HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN
from plot_commute_distance import read_urbanpop_columns, TRAVEL_WFH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Highest pre-college grade code in the .bin: grades there run 3..19, with 18 and 19 undergrad and
# grad -- the same is_college test UrbanPop-scripts/group_assignment.py sizes classes with.
COLLEGE_GRADE_MAX = 17
# Grade code of childcare children in the .bin (and of their staff, who carry the grade they
# teach). upop_to_exaepi.py's set_childcare imputes them from the under-5s UrbanPop has not in
# school, and preschoolers placed at a childcare center are given it too; Epicast has no
# equivalent, only preschools for UrbanPop's own preschoolers.
CHILDCARE_GRADE = 3
EPICAST_DIR = os.path.join(REPO_ROOT, "data", "results", "emerge-paper", "epicast", "ca")

# Per --group-name: default Epicast sizes file, the ExaEpi field(s) needed, and axis/label text.
GROUP_INFO = {
    "workgroup": {
        "epicast_file": os.path.join(EPICAST_DIR, "ca_workgroup_sizes.txt"),
        "xlabel": "Workgroup size (number of agents)",
        "title": "Workgroup size",
        "weight_noun": "worker",
    },
    "class": {
        "epicast_file": os.path.join(EPICAST_DIR, "ca_schoolgroup_sizes.txt"),
        "xlabel": "Class size (number of students)",
        "title": "School class size",
        "weight_noun": "student",
    },
    "school": {
        "epicast_file": os.path.join(EPICAST_DIR, "ca_school_sizes.txt"),
        "xlabel": "School size (number of agents)",
        "title": "School size",
        # school size includes both students and staff (see exaepi_school_sizes' docstring
        # note above), but students dominate the headcount, so "student-weighted" is the more
        # intuitive label here even though it's not literally students-only.
        "weight_noun": "student",
    },
}


def load_epicast_sizes(fname):
    sizes = np.loadtxt(fname, dtype=int)
    print(f"Read {len(sizes):,} sizes from Epicast file {fname}")
    return sizes


def group_counts(keys):
    """Size of each distinct group: how many agents share each value of keys."""
    return np.unique(keys, return_counts=True)[1]


# The kinds of school the class and school panels tell apart. K-12 (with preschool) is always
# shown; childcare and colleges only on request (--childcare, --colleges), since Epicast has
# neither.
SCHOOL_CATEGORIES = ("k12", "childcare", "college")
# (ExaEpi series, schools-data series) label and color for each category; the workgroup panel's
# ExaEpi series uses the K-12 one
SERIES_STYLE = {
    "k12": ({"label": "ExaEpi", "color": "red"}, {"label": "Schools data", "color": "black"}),
    "childcare": ({"label": "ExaEpi childcare", "color": "tab:green"},
                  {"label": "Schools data childcare", "color": "darkgreen"}),
    "college": ({"label": "ExaEpi colleges", "color": "tab:orange"},
                {"label": "Schools data colleges", "color": "saddlebrown"}),
}


def sizes_by_category(keys, counted, grade):
    """{category: group sizes} for the groups identified by keys, each group sized by its members
    with `counted` set and put in a category (SCHOOL_CATEGORIES) by its members' grades -- all of
    them, so a class's homeroom teacher counts too, since educators carry the grade they teach.
    A group with a college member is a college one, else one with a childcare member is
    childcare; in practice no school or class mixes them with other grades."""
    is_college = np.isin(keys, np.unique(keys[grade > COLLEGE_GRADE_MAX]))
    is_childcare = np.isin(keys, np.unique(keys[grade == CHILDCARE_GRADE])) & ~is_college
    members = {"k12": ~is_college & ~is_childcare, "childcare": is_childcare, "college": is_college}
    return {category: group_counts(keys[counted & members[category]]) for category in SCHOOL_CATEGORIES}


def exaepi_sizes(urbanpop_file, group_names):
    """({group_name: sizes} for each of group_names, the state FIPS codes the population covers),
    tallied from the UrbanPop .bin the way AgentContainer::computeGroupSizeDistributions does (see
    the module docstring). Workgroup sizes are one array; class and school sizes are split by
    category ({category: array}, see sizes_by_category).

    Each grouping is packed into one int64 key rather than grouped on several columns: a
    12-digit GEOID fits in 37 bits, naics (< NAICS_COUNT = 251) in 8 and the int16 workgroup /
    school_id in 15, so (geoid, naics, workgroup) takes 60 bits and nothing collides.
    """
    cols = read_urbanpop_columns(urbanpop_file, ["home_geoid", "work_geoid", "naics", "school_id",
                                                 "workgroup", "school_class_group", "travel", "grade"])
    grade = cols["grade"]
    naics = cols["naics"].astype(np.int64)
    school_id = cols["school_id"].astype(np.int64)
    # Declared work-from-home non-educators spend the day at home (UrbanPopData::initAgents);
    # everyone else is at their work_geoid's block group, which is one ExaEpi community.
    wfh = (naics != -1) & (school_id == 0) & (cols["travel"] == TRAVEL_WFH)
    community = np.where(wfh, cols["home_geoid"], cols["work_geoid"])
    # a 12-digit block-group GEOID starts with the 2-digit state FIPS
    states = sorted(int(s) for s in np.unique(cols["home_geoid"] // 10**10))
    del cols["home_geoid"], cols["work_geoid"]

    sizes = {}
    if "workgroup" in group_names:
        workgroup = cols["workgroup"].astype(np.int64)
        in_wg = workgroup > 0
        sizes["workgroup"] = group_counts((community[in_wg] << 23) | (naics[in_wg] << 15) | workgroup[in_wg])
    if "class" in group_names:
        scg = cols["school_class_group"]
        in_class = scg >= 0
        # classes are sized by their students only
        sizes["class"] = sizes_by_category(scg[in_class], (naics == -1)[in_class], grade[in_class])
    if "school" in group_names:
        in_school = school_id > 0
        school_keys = (community[in_school] << 15) | school_id[in_school]
        sizes["school"] = sizes_by_category(school_keys, np.ones(len(school_keys), dtype=bool), grade[in_school])
    for name, s in sizes.items():
        if isinstance(s, dict):
            print(f"Found ExaEpi {name} groups in {urbanpop_file}: "
                  + ", ".join(f"{len(v):,} {category}" for category, v in s.items()))
        else:
            print(f"Found {len(s):,} ExaEpi {name} groups in {urbanpop_file}")
    return sizes, states


def load_school_class_sizes(fname, states):
    """Class sizes implied by the schools file UrbanPop's allocation reads (get_schools.py's
    schools_with_geoids.csv), for the K-12 schools in the given states.

    Returns (sizes, class_counts) in the same form as load_cbp_establishment_sizes: each school is
    one entry, of size students / teachers and standing for `teachers` classes, so weighting by
    size * count counts each of its students once. The file has already dropped virtual and
    online schools and records with no usable enrollment or teacher count.

    K-12 only: childcare and college teacher counts aren't a faculty headcount (childcare's is
    imputed at 7 children per adult, a college's is its total employment), so their ratios say
    nothing about class size.

    NOTE this is a student-teacher ratio, which runs somewhat below the average class: the
    teacher counts include teachers without a homeroom class of their own (specialists, resource
    teachers), so there are fewer classes than teachers.
    """
    df = read_schools(fname, states, "k12")
    return (df.students / df.teachers).to_numpy(), df.teachers.to_numpy(dtype=float)


def load_school_sizes(fname, states, category):
    """School sizes for one category (SCHOOL_CATEGORIES) from the same schools file as
    load_school_class_sizes: each school's listed students plus teachers, since ExaEpi's school
    size counts staff too. K-12 is public and private schools. Returns (sizes, counts) with every
    count 1 -- each entry is one school.

    Childcare sizes are HIFLD's licensed capacity, and their staff is imputed at 7 children per
    adult. HIFLD reports no capacity at all for some states' centers (every NM one, for
    instance), and get_schools.py then samples each from the national distribution, so a
    childcare curve for such a state is that distribution rather than data about the state.
    A college's staff is its total employment, hospitals and all.
    """
    df = read_schools(fname, states, category)
    sizes = (df.students + df.teachers).to_numpy(dtype=float)
    return sizes, np.ones(len(sizes))


# get_schools.py's level tags for each SCHOOL_CATEGORIES entry other than K-12, which is the rest
CATEGORY_LEVELS = {"childcare": ["C"], "college": ["U"]}


def read_schools(fname, states, category):
    """The schools in `states` and one category (SCHOOL_CATEGORIES) from a get_schools.py schools
    file, keeping only records with both students and teachers."""
    import pandas as pd

    df = pd.read_csv(fname, dtype={"geoid": str, "id": str})
    if category == "k12":
        at_level = ~df.level.isin([lv for levels in CATEGORY_LEVELS.values() for lv in levels])
    else:
        at_level = df.level.isin(CATEGORY_LEVELS[category])
    df = df[df.geoid.str[:2].astype(int).isin(states) & at_level]
    df = df[(df.students > 0) & (df.teachers > 0)]
    if df.empty:
        sys.exit(f"No {category} schools for state FIPS {states} in {fname}")
    print(f"Read {len(df):,} {category} schools from {fname}")
    return df


def log_spaced_integer_bins(vmin, vmax, max_bins=50):
    """Log-spaced bin edges covering vmin to vmax, snapped to half-integers.

    Log-spaced bins are the right choice for a log-x histogram since equal *ratio* renders as
    equal visual width -- but for integer count data, bins near vmin come out narrower than a
    single unit, and one holding no integer at all is empty next to a neighbor with the whole
    count, which reads as a spike. Snapping every edge to a half-integer (and dropping the
    duplicates that makes) gives the small sizes one bin per integer and leaves the larger ones
    at full log resolution. Capping the bin count instead, so the narrowest bin stays a unit wide,
    left sizes from 1 to 6,560 with 12 bins.
    """
    lo, hi = max(vmin - 0.5, 0.5), vmax + 0.5
    if hi <= lo:
        return max_bins
    edges = np.floor(np.logspace(np.log10(lo), np.log10(hi), max_bins + 1)) + 0.5
    return np.unique(np.concatenate([[lo], edges[(edges > lo) & (edges < hi)], [hi]]))


# How many bins the narrowest distribution on a panel has to be resolved into, whatever the
# combined range is (see nice_linear_bins). Deliberately well below the whole panel's target: the
# narrow series only needs enough bins to show its shape, not the same resolution as the wide one.
MIN_BINS_FOR_NARROWEST = 25

# ...and the floor that stops that from overshooting: average samples per bin, for whichever series
# has the fewest in view. Finer bins resolve a tight distribution but also expose counting noise in
# a sparse one, and past a point the panel is showing sampling scatter rather than shape -- New
# Mexico has 1,982 ExaEpi schools in view against California's 24,358, so the same width that reads
# as a smooth curve for one is visibly spiky for the other.
MIN_SAMPLES_PER_BIN = 25


def core_span(sizes, lo_pct=1, hi_pct=99):
    """The x-range a distribution actually occupies, ignoring its tails.

    Percentiles rather than min/max because a single outlier -- one 70,000-agent university among
    California's 25,612 schools -- says nothing about how much room the distribution needs.
    """
    return float(np.percentile(sizes, hi_pct) - np.percentile(sizes, lo_pct))


def nice_linear_bins(vmin, vmax, target_bins=50, resolve_span=None, sparsest_count=None):
    """Linear bin edges from ~vmin to vmax, with a "nice" width (1/2/5 x a power of 10) instead
    of vmax-vmin split into an arbitrary number of equal pieces.

    matplotlib's default tick locator also picks its step from that same 1/2/5 x 10^n family, so
    whichever step it lands on is essentially always an integer multiple of this bin width --
    meaning bin *centers* fall exactly on the ticks it draws, the same way one-bin-per-integer
    bins naturally do (every integer tick is trivially some bin's center when width=1). An
    arbitrary width (e.g. span/50) has no such relationship to the ticks, so they end up looking
    like they're aligned to bin edges in some spots and nothing in particular elsewhere.

    `resolve_span`, if given, is a second, narrower range that also has to come out with enough
    bins to read; the width is then whichever of the two demands is finer. Two overlaid
    distributions share one set of bins (they have to, to be compared as densities), so a width
    sized only by the combined range starves whichever of them is tightly clustered: California's
    school sizes run to 1,200 on the ExaEpi side while 99.8% of Epicast's sit below 250, which at
    the 50-wide bins that range implies left the entire Epicast distribution drawn as five bars.

    `sparsest_count`, if given, is how many values the thinnest-sampled series has in view, and
    caps how far that refinement can go (MIN_SAMPLES_PER_BIN).
    """
    span = max(vmax - vmin, 1e-9)
    raw_width = span / target_bins
    if resolve_span:
        raw_width = min(raw_width, resolve_span / MIN_BINS_FOR_NARROWEST)
    if sparsest_count:
        raw_width = max(raw_width, span / max(1.0, sparsest_count / MIN_SAMPLES_PER_BIN))
    magnitude = 10 ** np.floor(np.log10(raw_width))
    width = next((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_width), 10 * magnitude)
    first_center = np.floor(vmin / width) * width
    return np.arange(first_center - width / 2, vmax + width, width)


def load_cbp_establishment_sizes(fname, state_fips, rng_seed=0):
    """Real establishment sizes for one state, straight from the CBP derived cache
    (data/UrbanPop/cbp19st_derived.csv, written by compute_workgroup_sizes.py).

    Returns (sizes, est_counts): each entry is a representative establishment size and how many
    real establishments it stands for, so weighting by size * est_counts gives the same
    worker-weighted density as the model series. Each CBP employment-size band holds n
    establishments totalling e employees, so sizes are spread log-uniformly across the band and
    rescaled to that band's own mean e/n -- preserving both the establishment count and the
    employment CBP reports.

    Only 2-digit NAICS sectors are summed. CBP reports every level of the NAICS hierarchy as its
    own row (11, 111, 1111, ...), so adding up all of them would count the same establishment
    once per level; the 2-digit sectors partition the state's establishments exactly once.

    NOTE this is the WORKPLACE tier, not the work-group tier: a work-group is a co-worker team
    inside an establishment (Epicast 2.0 sec. 2.4.2), so a 2451-person hospital is one CBP
    establishment but many work-groups. It belongs on this plot as a reference for what real
    workplaces look like, not as a target the work-group distribution should match.
    """
    import csv

    bands = [("<5", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49), ("50_99", 50, 99),
             ("100_249", 100, 249), ("250_499", 250, 499), ("500_999", 500, 999), ("1000", 1000, None)]
    rng = np.random.default_rng(rng_seed)
    sizes, counts = [], []
    with open(fname) as f:
        reader = csv.DictReader(f)
        if "n<5" not in (reader.fieldnames or []):
            sys.exit(f"{fname} has no establishment-size-band columns -- regenerate it with "
                     "compute_workgroup_sizes.py --refresh-cbp-cache")
        for row in reader:
            if int(row["fipstate"]) != state_fips or len(row["naics"]) != 2:
                continue
            for name, lo, hi in bands:
                n, e = int(row["n" + name] or 0), int(row["e" + name] or 0)
                if n <= 0 or e <= 0:
                    continue
                mean = e / n
                hi_eff = hi if hi is not None else max(lo + 1, int(mean * 4))
                draw = np.exp(rng.uniform(np.log(lo), np.log(hi_eff + 1), size=64))
                draw *= mean / draw.mean()
                sizes.append(np.clip(np.rint(draw), 1, None))
                counts.append(np.full(len(draw), n / len(draw)))
    if not sizes:
        sys.exit(f"No CBP rows for state FIPS {state_fips} in {fname}")
    return np.concatenate(sizes), np.concatenate(counts)


def histogram_bins(models, logx, xlim, max_integer_bins):
    """(bin edges, left edge for the view) shared by the given series: one bin per integer when the
    combined span is small enough to stay readable, else "nice" linear or log-spaced bins,
    matching plot_group_size_histogram.py's convention."""
    combined_max = max(s["sizes"].max() for s in models)
    combined_min = min(s["sizes"].min() for s in models)
    left_edge = combined_min if logx else 0
    # When zooming with xlim, size the bins for the zoomed-in range rather than the full
    # data range -- otherwise a few extreme outliers (e.g. a university) set a bin width
    # so wide that only one or two giant bins are even visible in the zoomed view. A final
    # catch-all bin (invisible once xlim clips the view) keeps all the data -- including
    # the outliers -- in the density normalization.
    bin_max = min(combined_max, xlim) if xlim is not None else combined_max
    if logx:
        # Linear (equal-width) bins would render as ever-narrower, unreadable slivers
        # once the x axis is log-scaled, since most of them get squeezed into the
        # rightmost decade. Log-spaced bins keep them visually even instead.
        bins = log_spaced_integer_bins(combined_min, bin_max)
    else:
        span = int(bin_max - combined_min)
        bins = (
            np.arange(combined_min - 0.5, bin_max + 1.5, 1.0)
            if span <= max_integer_bins
            # Sized so the tightest of the distributions is resolved too, not just the
            # combined range, and no finer than the sparsest one can support -- see
            # nice_linear_bins.
            else nice_linear_bins(
                combined_min, bin_max,
                resolve_span=min(core_span(s["sizes"]) for s in models),
                sparsest_count=min((s["sizes"] <= bin_max).sum() for s in models),
            )
        )
        # nice_linear_bins() anchors bin centers to global multiples of the bin width (so
        # they land on the same "nice" values matplotlib's tick locator picks -- see its
        # docstring), which can put a bin's center at/near 0 even though the data's own
        # minimum is well above it. Forcing the view to start exactly at 0 would then clip
        # that bin in half; starting it at the bin's own left edge instead always shows the
        # full first bar, at the cost of a little empty margin left of 0 when this happens.
        left_edge = bins[0]
    if bin_max < combined_max and not isinstance(bins, int):
        # nice_linear_bins() (and the integer scheme) can both overshoot bin_max by up to
        # one bin width, which -- for an xlim close enough to the true max -- can already
        # exceed combined_max. Drop any such edges before appending it, or the result isn't
        # monotonically increasing and numpy.histogram rejects it outright.
        bins = np.append(bins[bins < combined_max], combined_max)
    return bins, left_edge


def plot_comparison(ax, series, xlabel, title, cdf, weight_noun="member",
                     logx=False, logy=False, max_integer_bins=200, xlim=None):
    """Draw each of `series` on ax, as density histograms or (cdf) cumulative curves, all weighted
    by weight_noun.

    Each series is a dict: sizes, label, color, `reference` (True for real data, drawn as an
    outline over the model fills rather than as another fill) and optionally counts, letting one
    entry stand for many real groups (see load_cbp_establishment_sizes / load_school_class_sizes),
    and bin_group: series with the same bin_group share histogram bins, sized from that group's
    model series.
    """
    series = [s for s in series if len(s["sizes"])]
    for s in series:
        s["sizes"] = np.asarray(s["sizes"], dtype=float)
        s["weights"] = s["sizes"] * s["counts"] if s.get("counts") is not None else s["sizes"]
    models = [s for s in series if not s.get("reference")]
    overall_min = min(s["sizes"].min() for s in models)
    left_edge = overall_min if logx else 0

    if logx and overall_min <= 0:
        sys.exit(f"--logx requires strictly positive sizes, but {title} has a minimum of {overall_min}")

    def print_stats(s):
        # Weighted by size (each group of size s stands in for s members who experience that
        # group size) rather than one point per group -- a plain per-group histogram makes the
        # many small groups look dominant even when most members are actually in a big one, so
        # both the plot and its summary stats are weighted throughout by weight_noun. Printed
        # rather than shown in the legend -- this figure is only ~3.1in wide in the paper, with no
        # room for it at PLOS's 8-12pt font floor, and the legend should just name the series.
        sizes, member_w = s["sizes"], s["weights"]
        n_groups = len(sizes) if s.get("counts") is None else s["counts"].sum()
        weighted_mean = np.average(sizes, weights=member_w)
        order = np.argsort(sizes)
        sorted_sizes, cum_members = sizes[order], np.cumsum(member_w[order])
        weighted_median = sorted_sizes[np.searchsorted(cum_members, cum_members[-1] / 2)]
        print(
            f"{title} -- {s['label']}: n={n_groups:,.0f}, mean={weighted_mean:.1f}, "
            f"median={weighted_median:.1f}, max={sizes.max():,.0f}"
        )

    for s in series:
        print_stats(s)
    if cdf:
        for s in series:
            # Weighted cumulative fraction: the cumulative weight at position i is exactly "how
            # many members are in a group of size <= sorted_sizes[i]", divided by the total to
            # normalize to a fraction.
            order = np.argsort(s["sizes"])
            w = s["weights"][order]
            # the models at alpha<1, like the histogram's fill, so an overlapping segment blends
            # to a visibly distinct color instead of the later-drawn line fully hiding the other
            ax.step(s["sizes"][order], np.cumsum(w) / w.sum(), where="post", color=s["color"], linewidth=1,
                    alpha=1.0 if s.get("reference") else 0.7, label=s["label"])
        ax.set_ylabel(f"Cumulative fraction of {weight_noun}s")
    else:
        # Density-normalized (the models produce very different group counts, so only a density
        # comparison is fair), with bins shared within each bin_group -- a model series and its
        # schools-data outline, say -- but sized separately for each group. A single set of bins
        # across a school panel with colleges on it took its width from the colleges: 369 of them
        # spread over 0-70,000 agents allowed nothing finer than ~5,000-agent bins
        # (MIN_SAMPLES_PER_BIN), which left K-12 and childcare as one or two bars each.
        groups = list(dict.fromkeys(s.get("bin_group") for s in series))
        left_edges = []
        for group in groups:
            members = [s for s in series if s.get("bin_group") == group]
            group_models = [s for s in members if not s.get("reference")] or members
            bins, group_left = histogram_bins(group_models, logx, xlim, max_integer_bins)
            left_edges.append(group_left)
            # Drawn from np.histogram so that on a log axis the height can be density per decade
            # rather than per agent: per agent, the wide bins at large sizes come out so low that a
            # college's is invisible next to a classroom's. Each series is normalized over the
            # values inside its bins, as density=True would; entries past the last bin are
            # dropped, which only happens to a reference's tail beyond its models' range.
            widths = np.diff(np.log10(bins)) if logx else np.diff(bins)
            for s in sorted(members, key=lambda s: bool(s.get("reference"))):
                weights, _ = np.histogram(s["sizes"], bins=bins, weights=s["weights"])
                density = weights / weights.sum() / widths
                if s.get("reference"):
                    # Outline rather than another filled patch -- a reference curve for what the
                    # real groups look like, not another model, and the translucent fills already
                    # overlap here.
                    ax.stairs(density, bins, color=s["color"], linewidth=1, label=s["label"])
                else:
                    ax.stairs(density, bins, fill=True, color=s["color"], alpha=0.5, label=s["label"])
        left_edge = min(left_edges)
        ax.set_ylabel(f"Density per decade\n({weight_noun}-weighted)" if logx else f"Density ({weight_noun}-weighted)")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    # Anchor the left edge explicitly rather than leaving it to matplotlib's default ~5%
    # margin (which otherwise leaves a visible gap before 0, or goes negative once xlim pulls
    # the right edge in far enough that the margin becomes a large fraction of the range).
    # right=None (the default, when xlim isn't given) leaves the right edge autoscaled.
    ax.set_xlim(left=left_edge, right=xlim)
    # Headroom above the tallest bar/curve for the legend to sit in -- "best" placement
    # sometimes has nowhere left to go but on top of a peak, especially with a two-line label
    # per series.
    #ax.set_ylim(top=ax.get_ylim()[1] * 1.25)

    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if len(series) > 4:
        # with childcare and colleges on, there are too many entries to fit over the data, and
        # beside it they squeeze the axes at the paper's half-page width
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize="small")
    else:
        ax.legend()


def main():
    apply_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--urbanpop_file", "-u", default=os.path.join(REPO_ROOT, "data", "UrbanPop", "urbanpop_ca.bin"),
        help="UrbanPop .bin ExaEpi reads its agents from; the ExaEpi group sizes are computed from it "
        "directly (see the module docstring)",
    )
    parser.add_argument(
        "--childcare", action="store_true",
        help="Add ExaEpi's childcare centers to the class and school panels as their own series, "
        "with the schools data's on the school panel. Epicast has no childcare (only preschools, for "
        "the children UrbanPop has in preschool), so by default these panels show K-12 only.",
    )
    parser.add_argument(
        "--colleges", action="store_true",
        help="Likewise for colleges and universities (any school or class with undergrad or grad "
        "members), which Epicast has none of either.",
    )
    parser.add_argument(
        "--groups", "-g", nargs="+", choices=list(GROUP_INFO), default=list(GROUP_INFO),
        help="Which group-size distributions to plot (default: all three)",
    )
    parser.add_argument(
        "--epicast_workgroup", default=GROUP_INFO["workgroup"]["epicast_file"],
        help="Epicast workgroup sizes file (one size per line)",
    )
    parser.add_argument(
        "--epicast_class", default=GROUP_INFO["class"]["epicast_file"],
        help="Epicast school-class (schoolgroup) sizes file (one size per line)",
    )
    parser.add_argument(
        "--epicast_school", default=GROUP_INFO["school"]["epicast_file"],
        help="Epicast school sizes file (one size per line)",
    )
    parser.add_argument(
        "--histogram", action="store_true",
        help="Plot density histograms instead of the default CDF. Group sizes here span "
        "several orders of magnitude, and a log-x histogram's bins can't be both uniform "
        "width and finer than the ~1.5x-per-bin resolution set by the smallest sizes -- the "
        "CDF has no such trade-off, so it's the default.",
    )
    parser.add_argument(
        "--logx", action="store_true", help="Use a logarithmic x-axis",
    )
    parser.add_argument(
        "--logy", action="store_true", help="Use a logarithmic y-axis",
    )
    parser.add_argument(
        "--xlim", type=float, default=None, help="Maximum x-axis value to display",
    )
    parser.add_argument(
        "--output", "-o", default="group_size_comparison.png", help="Output image file",
    )
    parser.add_argument(
        "--cbp_state", type=int, default=None, metavar="FIPS",
        help="Overlay the real CBP establishment-size distribution for this state FIPS (6 = CA, "
        "35 = NM) on the workgroup panel. This is the WORKPLACE tier, not the workgroup tier -- "
        "a workgroup is a co-worker team inside an establishment -- so it is a reference for what "
        "real workplaces look like, not a target the workgroup sizes should match.",
    )
    parser.add_argument(
        "--cbp_sizes_file",
        default=os.path.join(REPO_ROOT, "data", "UrbanPop", "cbp19st_derived.csv"),
        help="CBP derived cache with establishment-size bands (see compute_workgroup_sizes.py)",
    )
    parser.add_argument(
        "--schools_file",
        default=os.path.join(REPO_ROOT, "data", "EducationData", "schools_with_geoids.csv"),
        help="Schools file UrbanPop's allocation reads (get_schools.py), for the reference curves "
        "on the class and school panels, in the states the UrbanPop file covers: each K-12 school's "
        "student-teacher ratio (load_school_class_sizes), and each school's listed students plus "
        "staff -- public and private K-12, and childcare and colleges with --childcare/--colleges "
        "(load_school_sizes). An empty string leaves the curves out.",
    )
    args = parser.parse_args()
    cdf = not args.histogram
    categories = ["k12"] + (["childcare"] if args.childcare else []) + (["college"] if args.colleges else [])

    epicast_files = {
        "workgroup": args.epicast_workgroup,
        "class": args.epicast_class,
        "school": args.epicast_school,
    }

    # Total width fixed at the paper's half-page width regardless of how many groups are plotted
    # side by side -- each panel just gets narrower as more are added (see plos_compbio_style.py).
    # Height stays the shared standard regardless of how many panels there are.
    fig, axes = plt.subplots(1, len(args.groups), figsize=(HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN), layout="constrained")
    if len(args.groups) == 1:
        axes = [axes]

    exaepi_data_by_group, states = exaepi_sizes(args.urbanpop_file, args.groups)
    for ax, group_name in zip(axes, args.groups):
        info = GROUP_INFO[group_name]
        # Epicast is K-12 only, so it shares the K-12 series' bins (see plot_comparison)
        series = [{"sizes": load_epicast_sizes(epicast_files[group_name]), "label": "Epicast", "color": "blue",
                   "bin_group": "k12"}]
        if group_name == "workgroup":
            series.append({"sizes": exaepi_data_by_group[group_name], **SERIES_STYLE["k12"][0], "bin_group": "k12"})
            if args.cbp_state is not None:
                sizes, counts = load_cbp_establishment_sizes(args.cbp_sizes_file, args.cbp_state)
                series.append({"sizes": sizes, "counts": counts, "label": "CBP establishments", "color": "black",
                               "reference": True, "bin_group": "k12"})
        else:
            for category in categories:
                exaepi_style, data_style = SERIES_STYLE[category]
                series.append({"sizes": exaepi_data_by_group[group_name][category], **exaepi_style, "bin_group": category})
                if not args.schools_file:
                    continue
                # the schools data's class sizes are only meaningful for K-12 (see
                # load_school_class_sizes)
                if group_name == "class" and category == "k12":
                    sizes, counts = load_school_class_sizes(args.schools_file, states)
                elif group_name == "school":
                    sizes, counts = load_school_sizes(args.schools_file, states, category)
                else:
                    continue
                series.append({"sizes": sizes, "counts": counts, **data_style, "reference": True, "bin_group": category})
        plot_comparison(ax, series, info["xlabel"], info["title"], cdf, weight_noun=info["weight_noun"],
                        logx=args.logx, logy=args.logy, xlim=args.xlim)

    plt.savefig(args.output, dpi=300)
    print(f"{'CDF' if cdf else 'Histogram'} comparison saved to {args.output}")


if __name__ == "__main__":
    main()
