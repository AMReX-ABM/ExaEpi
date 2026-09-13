"""Shared, dependency-light geographic-aggregation helper for the plot_geo* scripts.

Deliberately has no imports beyond the standard library -- plot_geo.py needs both yt (for ExaEpi
plotfiles) and read_epicast_events (for Epicast), but neither of those should be required just to
reuse this one function, so it lives on its own rather than in that module.
"""


def aggregate_to_county(grid_stats_df, input_level):
    """Collapse a per-community DataFrame (GEOID10 at block group [12-digit] or Census tract
    [11-digit] granularity, per input_level: "block_group" or "tract") down to the county level
    (5-digit GEOID10: state+county only), summing pop/never_infected/infected/immune across every
    unit in each county.

    The divisor is fixed by input_level rather than auto-detected from the input GEOID's digit
    count: as an int64, a state FIPS code starting with "0" (e.g. California, 06) loses its
    leading zero, which would shift the apparent digit count down by one and silently corrupt the
    computed county GEOID for that state alone.
    """
    tract_digits_dropped = {"tract": 6, "block_group": 7}[input_level]
    grid_stats_df = grid_stats_df.copy()
    grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"] // 10**tract_digits_dropped
    return grid_stats_df.groupby("GEOID10", as_index=False)[
        ["pop", "never_infected", "infected", "immune"]
    ].sum()
