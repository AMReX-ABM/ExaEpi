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

    events_df = pd.DataFrame(
        {
            "agent_id": agent_id,
            "location_id": location_id,
            "timestep": records["timestep"],
            "context": pd.Series(records["context"], dtype="uint8")
            .map(_CONTEXT_MAP)
            .astype(_CONTEXT_DTYPE),
            "disease_state": pd.Series(records["disease_state"], dtype="uint8")
            .map(_DISEASE_STATE_MAP)
            .astype(_DISEASE_STATE_DTYPE),
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


def aggregate_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event counts by day (pairs of timesteps), disease_state, and context.

    Timesteps are grouped into consecutive pairs: timesteps 0 and 1 form day 0,
    timesteps 2 and 3 form day 1, etc.  Returns a DataFrame with one row per day
    and one column per disease_state category and one column per context category,
    containing the count of events in each category on that day.  An additional
    ``total`` column holds the total number of events on that day.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of :func:`read_events_bin`.

    Returns
    -------
    agg_df : pd.DataFrame
        Columns: day, <disease_state categories…>, <context categories…>, total
    """
    # Assign each timestep to a day: day = timestep // 2
    df = events_df.copy()
    df["day"] = (df["timestep"] // 2).astype(int)

    def _pivot(col, dtype):
        piv = df.groupby(["day", col], observed=True).size().unstack(fill_value=0)
        # Flatten CategoricalIndex to plain string Index so concat/join works
        piv.columns = piv.columns.astype(str)
        for cat in dtype.categories:
            if cat not in piv.columns:
                piv[cat] = 0
        return piv[list(dtype.categories)]

    ds_piv = _pivot("disease_state", _DISEASE_STATE_DTYPE)
    ctx_piv = _pivot("context", _CONTEXT_DTYPE)

    agg_df = pd.concat([ds_piv, ctx_piv], axis=1).fillna(0).astype(int)
    agg_df["total"] = ds_piv.sum(axis=1)
    agg_df = agg_df.reset_index()
    return agg_df


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import os

    path = sys.argv[1] if len(sys.argv) > 1 else "run.events.bin"
    print(f"Reading {path} ...")

    events_df, demog_df = read_events_bin(path)

    print(f"\n=== Events DataFrame ({len(events_df):,} rows) ===")
    print(events_df.dtypes)
    print(events_df.head(10))

    print(f"\n=== Demographics DataFrame ({len(demog_df):,} rows) ===")
    print(demog_df.dtypes)
    print(demog_df.head(10))

    # Write events DataFrame to CSV
    base = os.path.splitext(path)[0]
    # csv_path = base + ".csv"
    # print(f"\nWriting events CSV to {csv_path} ...")
    # events_df.to_csv(csv_path, index=False)
    # print(f"Done. {len(events_df):,} rows written.")

    # Aggregate and write .agg.csv
    agg_df = aggregate_events(events_df)
    agg_path = base + ".agg.csv"
    print(f"\n=== Aggregated DataFrame ({len(agg_df):,} days) ===")
    print(agg_df.head(10))
    print(f"\nWriting aggregated CSV to {agg_path} ...")
    agg_df.to_csv(agg_path, index=False)
    print(f"Done. {len(agg_df):,} rows written.")
