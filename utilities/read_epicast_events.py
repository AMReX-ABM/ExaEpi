#!/usr/bin/env -S python -u

"""
Read a run.events.bin file produced by Epicast into a pandas DataFrame.

Binary format (little-endian):
  Header:
    nrow        : UInt64  - number of FIPS tracts
    ncol        : UInt64  - number of event columns (unused for events file)
    n_pt        : UInt64  - number of timepoints
    ncol_demog  : UInt64  - number of demographic columns
    demo_len    : UInt64  - byte length of null-separated demographic column names
    col_len     : UInt64  - byte length of null-separated event column names
    demo_names  : demo_len bytes, null-separated strings
    col_names   : col_len  bytes, null-separated strings  (ignored for events)
    fips        : nrow × UInt64  - FIPS code for each tract
    demographics: nrow × ncol_demog × UInt32

  Event records (AgentTransition, 24 bytes each, little-endian):
    agent_id    : UInt64  (bits 63-58 encode home_state; bits 57-0 encode agent id)
    location_id : UInt64  (bits 63-8 encode tract FIPS; bits 7-0 encode community)
    timestep    : UInt16
    context     : UInt8
    disease_state: UInt8
    variant     : UInt8
    _pad        : 3 bytes  (Julia struct alignment padding)

Derived columns added to the events DataFrame:
    home_state      - top 6 bits of agent_id  (agent_id >> 58)
    true_agent_id   - agent_id with home_state bits masked out
    tract_fips      - location_id >> 8
    tract_community - location_id & 0xff
"""

import struct
import numpy as np
import pandas as pd


# Mapping from context integer codes to human-readable labels
_CONTEXT_MAP = {
    0x00: "ctx_household",
    0x01: "ctx_playgroup",
    0x02: "ctx_daycare",
    0x03: "ctx_school",
    0x04: "ctx_work",
    0x05: "ctx_teachers",
    0x06: "ctx_household_cluster",
    0x07: "ctx_bar_social",
    0x08: "ctx_student_teacher",
    0x09: "ctx_teacher_student",
    0x0A: "ctx_neighborhood_community",
    0x0B: "ctx_customer",
    0x0C: "ctx_presymptomatic",
    0x0D: "ctx_asymptomatic_recovered",
    0x0E: "ctx_symptomatic_recovered",
    0x0F: "ctx_symptomatic",
    0x10: "ctx_asymptomatic",
    0x11: "ctx_hospitalized",
    0x12: "ctx_icu",
    0x13: "ctx_ventilated",
    0x14: "ctx_treatment_recovered",
    0x15: "ctx_removed",
    0xFF: "ctx_index_case",
}
_CONTEXT_DTYPE = pd.CategoricalDtype(
    categories=list(_CONTEXT_MAP.values()),
    ordered=False,
)

# Mapping from disease_state integer codes to human-readable labels
DISEASE_STATE_CATEGORIES = [
    "recovered",  # 0x00
    "exposed",  # 0x01
    "symptomatic",  # 0x02
    "asymptomatic",  # 0x03
    None,  # 0x04 (unused)
    None,  # 0x05 (unused)
    None,  # 0x06 (unused)
    "presymptomatic",  # 0x07
]
_DISEASE_STATE_MAP = {i: v for i, v in enumerate(DISEASE_STATE_CATEGORIES) if v is not None}
_DISEASE_STATE_DTYPE = pd.CategoricalDtype(
    categories=["recovered", "exposed", "symptomatic", "asymptomatic", "presymptomatic"],
    ordered=False,
)

# AgentTransition struct layout (24 bytes total, little-endian)
# Q  = UInt64 (8 bytes)
# H  = UInt16 (2 bytes)
# B  = UInt8  (1 byte)
# 3x = 3 padding bytes
_AGENT_TRANSITION_FMT = "<QQHBBBxxx"
_AGENT_TRANSITION_SIZE = struct.calcsize(_AGENT_TRANSITION_FMT)  # should be 24
assert _AGENT_TRANSITION_SIZE == 24, f"Unexpected struct size: {_AGENT_TRANSITION_SIZE}"


def read_events_bin(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a run.events.bin file.

    Parameters
    ----------
    path : str
        Path to the .events.bin file.

    Returns
    -------
    events_df : pd.DataFrame
        One row per AgentTransition event with columns:
          agent_id, location_id, timestep, context, disease_state, variant,
          home_state, true_agent_id, tract_fips, tract_community
    demog_df : pd.DataFrame
        One row per FIPS tract with columns: fips, <demographic column names>
    """
    with open(path, "rb") as f:
        # ------------------------------------------------------------------ #
        # Header
        # ------------------------------------------------------------------ #
        nrow = struct.unpack("<Q", f.read(8))[0]
        ncol = struct.unpack("<Q", f.read(8))[0]  # noqa: F841
        n_pt = struct.unpack("<Q", f.read(8))[0]  # noqa: F841
        ncol_demog = struct.unpack("<Q", f.read(8))[0]
        demo_len = struct.unpack("<Q", f.read(8))[0]
        col_len = struct.unpack("<Q", f.read(8))[0]  # noqa: F841

        demo_names = [s for s in f.read(demo_len).decode("utf-8").split("\x00") if s]
        _col_names = [s for s in f.read(col_len).decode("utf-8").split("\x00") if s]  # noqa: F841

        # FIPS codes (one per tract row)
        fips = np.frombuffer(f.read(nrow * 8), dtype="<u8").copy()

        # Demographics matrix: nrow × ncol_demog, stored as UInt32
        demog_raw = (
            np.frombuffer(f.read(nrow * ncol_demog * 4), dtype="<u4")
            .reshape(nrow, ncol_demog)
            .copy()
        )

        # ------------------------------------------------------------------ #
        # AgentTransition event records
        # ------------------------------------------------------------------ #
        event_bytes = f.read()

    n_events = len(event_bytes) // _AGENT_TRANSITION_SIZE
    remainder = len(event_bytes) % _AGENT_TRANSITION_SIZE
    if remainder != 0:
        import warnings

        warnings.warn(
            f"Event data contains {remainder} trailing bytes that do not form "
            "a complete AgentTransition record and will be ignored."
        )

    # Parse all records at once using numpy structured array for speed
    dtype = np.dtype(
        [
            ("agent_id", "<u8"),
            ("location_id", "<u8"),
            ("timestep", "<u2"),
            ("context", "u1"),
            ("disease_state", "u1"),
            ("variant", "u1"),
            ("_pad", "V3"),  # 3 padding bytes
        ]
    )
    assert dtype.itemsize == 24

    records = np.frombuffer(event_bytes[: n_events * _AGENT_TRANSITION_SIZE], dtype=dtype)

    agent_id = records["agent_id"].astype(np.uint64)
    location_id = records["location_id"].astype(np.uint64)

    # Derived columns (matching Julia helper functions)
    home_state = (agent_id >> np.uint64(58)).astype(np.uint8)
    mask = ~(np.uint64(0b0111111) << np.uint64(58))
    true_agent_id = agent_id & mask
    tract_fips = location_id >> np.uint64(8)
    tract_community = (location_id & np.uint64(0xFF)).astype(np.uint8)

    raw_disease_state = pd.Series(records["disease_state"], dtype="uint8")
    unmapped_mask = ~raw_disease_state.isin(_DISEASE_STATE_MAP.keys())
    n_unmapped = unmapped_mask.sum()
    if n_unmapped > 0:
        import warnings

        unmapped_codes = raw_disease_state[unmapped_mask].unique().tolist()
        warnings.warn(
            f"{n_unmapped} event(s) have unmapped disease_state code(s) {unmapped_codes} "
            "(expected codes: 0x00–0x03, 0x07). These events will have NaN disease_state "
            "and will be excluded from disease_state counts and 'total' in aggregate_events()."
        )

    events_df = pd.DataFrame(
        {
            "agent_id": agent_id,
            "location_id": location_id,
            "timestep": records["timestep"],
            "context": pd.Series(records["context"], dtype="uint8")
            .map(_CONTEXT_MAP)
            .astype(_CONTEXT_DTYPE),
            "disease_state": raw_disease_state.map(_DISEASE_STATE_MAP).astype(_DISEASE_STATE_DTYPE),
            "variant": records["variant"],
            "home_state": home_state,
            "true_agent_id": true_agent_id,
            "tract_fips": tract_fips,
            "tract_community": tract_community,
        }
    )

    # ------------------------------------------------------------------ #
    # Demographics DataFrame
    # ------------------------------------------------------------------ #
    demog_df = pd.DataFrame({"fips": fips})
    for i, name in enumerate(demo_names):
        demog_df[name] = demog_raw[:, i]

    return events_df, demog_df


def aggregate_events(events_df: pd.DataFrame, split_day_night: bool = False) -> pd.DataFrame:
    """
    Aggregate event counts by day (pairs of timesteps), disease_state, and context.

    Timesteps are grouped into consecutive pairs: timesteps 0 and 1 form day 0,
    timesteps 2 and 3 form day 1, etc.

    When *split_day_night* is ``False`` (default) the two 12-hour periods within
    each calendar day are merged and the result has one row per day.

    When *split_day_night* is ``True`` the periods are kept separate and the
    result has two rows per calendar day: one for the night period (even timestep)
    and one for the day period (odd timestep).  An extra ``period`` column
    contains ``"day"`` or ``"night"``.

    In both cases each row contains one column per disease_state category, one
    column per context category, and a ``total`` column with the total number of
    events in that period/day.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of :func:`read_events_bin`.
    split_day_night : bool, optional
        If ``True``, keep day and night periods as separate rows.
        Default is ``False`` (aggregate both periods into a single daily row).

    Returns
    -------
    agg_df : pd.DataFrame
        When split_day_night is False:
            Columns: day, <disease_state categories…>, <context categories…>, total
        When split_day_night is True:
            Columns: day, period, <disease_state categories…>, <context categories…>, total
    """
    df = events_df.copy()
    # day = which 24-hour calendar day (timestep // 2)
    df["day"] = (df["timestep"] // 2).astype(int)
    # period: even timestep → "night" half (household/household_cluster context events occur
    # exclusively here), odd timestep → "day" half (school/work/teacher context events occur
    # exclusively here) -- confirmed empirically via context vs. timestep-parity cross-tabulation.
    df["period"] = np.where(df["timestep"] % 2 == 0, "night", "day")

    group_keys = ["day", "period"] if split_day_night else ["day"]

    def _pivot(col, dtype):
        piv = df.groupby(group_keys + [col], observed=True).size().unstack(fill_value=0)
        # Flatten CategoricalIndex to plain string Index so concat/join works
        piv.columns = piv.columns.astype(str)
        for cat in dtype.categories:
            if cat not in piv.columns:
                piv[cat] = 0
        return piv[list(dtype.categories)]

    ds_piv = _pivot("disease_state", _DISEASE_STATE_DTYPE)
    ctx_piv = _pivot("context", _CONTEXT_DTYPE)

    # Count all events per group (including those with unmapped/NaN disease_state)
    total_per_group = df.groupby(group_keys).size().rename("total")

    agg_df = pd.concat([ds_piv, ctx_piv], axis=1).fillna(0).astype(int)
    agg_df = agg_df.join(total_per_group, how="left").fillna(0)
    agg_df["total"] = agg_df["total"].astype(int)
    agg_df = agg_df.reset_index()
    return agg_df


# Mapping from Epicast's fine-grained transmission contexts to the coarser buckets ExaEpi's
# context_diag output uses. ExaEpi tracks neighborhood/community separately by day and night
# (ENbhD, ECommD, ENbhN, ECommN); Epicast records both under the single "neighborhood_community"
# context with no day/night split, so all four ExaEpi columns must be summed to compare against
# it (see aggregate_source_fractions in compare_to_epicast.py). "Household cluster" (Epicast) and
# "neighborhood cluster" (ExaEpi's ENC / InteractionModNC) are the same concept under different
# names. Contexts absent from this mapping (disease-progression contexts like ctx_symptomatic,
# and ctx_index_case for initial seed infections) never represent a transmission source and are
# dropped by aggregate_infections_by_source rather than mapped to "other".
SOURCE_CATEGORIES = ["household", "cluster", "neighborhood_community", "work", "school", "hospital", "other"]

_CONTEXT_TO_SOURCE = {
    "ctx_household":            "household",
    "ctx_household_cluster":    "cluster",
    "ctx_neighborhood_community": "neighborhood_community",
    "ctx_work":                 "work",
    "ctx_hospitalized":         "hospital",
    "ctx_school":               "school",
    "ctx_teachers":             "school",
    "ctx_teacher_student":      "school",
    "ctx_student_teacher":      "school",
    "ctx_playgroup":            "school",
    "ctx_daycare":              "school",
    "ctx_customer":             "other",
    "ctx_bar_social":           "other",
}


def aggregate_infections_by_source(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate new-infection ("exposed") events by day and infection-source bucket, giving an
    empirical estimate -- from Epicast's realized, per-agent context attribution -- of the same
    quantity ExaEpi's context_diag columns estimate analytically (see AgentContainer::
    sumContextInfections): the probability/share of new infections attributable to each
    interaction context.

    ctx_index_case events (initial seed infections, not caused by any modeled interaction) are
    excluded from both the counts and the fractions, matching what ExaEpi's context_diag would
    report for the same run (it has no equivalent "seed" bucket).

    Returns
    -------
    pd.DataFrame with columns: day, <SOURCE_CATEGORIES counts>, total, <SOURCE_CATEGORIES>_frac
        The *_frac columns are each day's count divided by that day's total (0 when total is 0),
        directly comparable to ExaEpi's E<source>/sum(E<source>) shares.
    """
    exposed = events_df.loc[events_df["disease_state"] == "exposed"].copy()
    exposed = exposed.loc[exposed["context"] != "ctx_index_case"]

    source = exposed["context"].astype(str).map(_CONTEXT_TO_SOURCE)
    unmapped_mask = source.isna()
    if unmapped_mask.any():
        import warnings

        unmapped_ctx = exposed.loc[unmapped_mask, "context"].unique().tolist()
        warnings.warn(
            f"{unmapped_mask.sum()} exposed event(s) have context(s) {unmapped_ctx} not in "
            "_CONTEXT_TO_SOURCE; they will be excluded from aggregate_infections_by_source."
        )
    exposed = exposed.loc[~unmapped_mask].copy()
    exposed["source"] = source[~unmapped_mask]
    exposed["day"] = (exposed["timestep"] // 2).astype(int)

    counts = (
        exposed.groupby(["day", "source"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for cat in SOURCE_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[SOURCE_CATEGORIES]
    counts["total"] = counts.sum(axis=1)

    fracs = counts[SOURCE_CATEGORIES].div(counts["total"].replace(0, np.nan), axis=0).fillna(0)
    fracs.columns = [c + "_frac" for c in SOURCE_CATEGORIES]

    out = counts.join(fracs).reset_index()
    return out


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="Read an Epicast run.events.bin file and write an aggregated CSV."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="run.events.bin",
        help="Path to the .events.bin file (default: run.events.bin)",
    )
    parser.add_argument(
        "--split-day-night",
        action="store_true",
        default=False,
        help=(
            "Keep day and night periods as separate rows in the output. "
            "When set, the output CSV contains two rows per calendar day "
            "(one for the day period, one for the night period) and a "
            "'period' column with values 'day' or 'night'. "
            "The output file is named <base>.agg.split.csv instead of <base>.agg.csv."
        ),
    )
    args = parser.parse_args()

    print(f"Reading {args.path} ...")
    events_df, demog_df = read_events_bin(args.path)

    print(f"\n=== Events DataFrame ({len(events_df):,} rows) ===")
    print(events_df.dtypes)
    print(events_df.head(10))

    print(f"\n=== Demographics DataFrame ({len(demog_df):,} rows) ===")
    print(demog_df.dtypes)
    print(demog_df.head(10))

    # Write events DataFrame to CSV
    base = os.path.splitext(args.path)[0]
    # csv_path = base + ".csv"
    # print(f"\nWriting events CSV to {csv_path} ...")
    # events_df.to_csv(csv_path, index=False)
    # print(f"Done. {len(events_df):,} rows written.")

    # Aggregate and write CSV
    agg_df = aggregate_events(events_df, split_day_night=args.split_day_night)
    if args.split_day_night:
        agg_path = base + ".agg.split.csv"
        period_desc = "day/night rows"
    else:
        agg_path = base + ".agg.csv"
        period_desc = "days"
    print(f"\n=== Aggregated DataFrame ({len(agg_df):,} {period_desc}) ===")
    print(agg_df.head(10))
    print(f"\nWriting aggregated CSV to {agg_path} ...")
    agg_df.to_csv(agg_path, index=False)
    print(f"Done. {len(agg_df):,} rows written.")
