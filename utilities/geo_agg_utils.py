"""Shared, dependency-light geographic-aggregation helper for the plot_geo* scripts.

Deliberately has no imports beyond the standard library -- plot_geo.py needs yt and
plot_geo_epicast.py needs read_epicast_events, but neither of those should be required just to
reuse this one function, so it lives on its own rather than in either of those modules.
"""


def aggregate_to_county(grid_stats_df):
    """Collapse a per-community DataFrame (GEOID10 at block group [12-digit] or Census tract
    [11-digit] granularity) down to the county level (5-digit GEOID10: state+county only), summing
    pop/never_infected/infected/immune across every unit in each county. Works from either starting
    granularity -- the digit count of the input GEOID is auto-detected.
    """
    grid_stats_df = grid_stats_df.copy()
    digits = len(str(int(grid_stats_df["GEOID10"].iloc[0])))
    divisor = 10 ** (digits - 5)
    grid_stats_df["GEOID10"] = grid_stats_df["GEOID10"] // divisor
    return grid_stats_df.groupby("GEOID10", as_index=False)[
        ["pop", "never_infected", "infected", "immune"]
    ].sum()
