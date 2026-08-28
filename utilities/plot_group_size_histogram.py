#!/usr/bin/env python

"""Plot a histogram of ExaEpi community/neighborhood sizes, or of neighborhoods per
community, from a run's plotfile.

Community size is the number of agents whose home is in a given community (one AMReX
grid cell == one community in ExaEpi). This is read directly from the "comm" and
"total" mesh fields written to every plotfile, so it works for a plotfile from any day.

Neighborhood size is the number of agents in a given (community, neighborhood) pair --
neighborhood IDs are only unique within a community, so agents are grouped by the
(home_i, home_j, nborhood) triple.

Neighborhoods per community is the number of distinct neighborhood IDs among the agents
living in each community.

Both of the latter two read the per-agent "nborhood" field, which is only written to the
first plotfile of a run (typically plt00000), so --field neighborhood and
--field nborhoods_per_community require that file.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yt
import yt.frontends.amrex.api  # noqa: F401  (registers the AMReX frontend's IO handlers)
from yt.frontends.amrex.data_structures import AMReXDataset


def _find_mesh_field(ds, name):
    """Resolve a mesh field name, allowing for the per-disease prefix ExaEpi uses when a
    run has more than one disease (e.g. "total" -> "covid_total")."""
    field_names = {f[1] for f in ds.field_list}
    if name in field_names:
        return name
    candidates = sorted(f for f in field_names if f.endswith("_" + name))
    if not candidates:
        sys.exit(f"Field '{name}' not found in plotfile")
    if len(candidates) > 1:
        print(f"Multiple '{name}' fields found for multi-disease run {candidates}; using {candidates[0]}")
    return candidates[0]


def community_sizes(ds):
    ad = ds.all_data()
    comm = np.rint(np.asarray(ad[_find_mesh_field(ds, "comm")])).astype(int)
    total = np.rint(np.asarray(ad[_find_mesh_field(ds, "total")])).astype(int)
    return total[comm != -1]


def _home_nborhood_df(ds):
    """Per-agent home community (home_i, home_j) and neighborhood ID, as a DataFrame."""
    ptype = "agents"
    particle_fields = {f[1] for f in ds.field_list if f[0] == ptype}
    required = {"particle_home_i", "particle_home_j", "particle_nborhood"}
    missing = required - particle_fields
    if missing:
        sys.exit(
            f"Plotfile is missing per-agent field(s) {sorted(missing)} needed for neighborhood "
            "data; these are only written to the first plotfile of a run (e.g. plt00000)."
        )
    ad = ds.all_data()
    return pd.DataFrame(
        {
            "home_i": np.rint(np.asarray(ad[(ptype, "particle_home_i")])).astype(int),
            "home_j": np.rint(np.asarray(ad[(ptype, "particle_home_j")])).astype(int),
            "nborhood": np.rint(np.asarray(ad[(ptype, "particle_nborhood")])).astype(int),
        }
    )


def neighborhood_sizes(ds):
    df = _home_nborhood_df(ds)
    return df.groupby(["home_i", "home_j", "nborhood"]).size().to_numpy()


def nborhoods_per_community(ds):
    """Number of distinct neighborhood IDs among the agents living in each community.

    Each community is a single block group, which ExaEpi gives
    get_max_nborhood(nborhood_size, home_population) == round(home_population /
    nborhood_size) neighborhoods, drawing each agent's ID uniformly over that range
    (UrbanPopData.cpp). Counting distinct IDs therefore recovers that allocation exactly
    so long as every neighborhood drew at least one agent -- which at realistic
    nborhood_size values it reliably does, since a neighborhood is hundreds of agents.
    """
    df = _home_nborhood_df(ds)
    return df.groupby(["home_i", "home_j"])["nborhood"].nunique().to_numpy()


# Widest data span that still gets one histogram bin per integer by default. Beyond this the
# bars get too thin to read and a fixed bin count is the better default.
MAX_INTEGER_BINS = 200

# Per --field presentation: what one sample is (used for the count line in the stats box and
# the "Found N ..." message), the x-axis label, the noun for the quantity being summarized,
# and the default output basename.
FIELD_INFO = {
    "community": ("communities", "Community size (number of agents)", "size", "community_sizes"),
    "neighborhood": ("neighborhoods", "Neighborhood size (number of agents)", "size", "neighborhood_sizes"),
    "nborhoods_per_community": ("communities", "Neighborhoods per community", "count", "nborhoods_per_community"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot_dir", "-p", required=True, help="ExaEpi plotfile directory, e.g. plt00000")
    parser.add_argument(
        "--field",
        "-f",
        choices=list(FIELD_INFO),
        default="community",
        help="Which quantity to histogram (default: community)",
    )
    parser.add_argument(
        "--bins",
        "-b",
        type=int,
        default=None,
        help="Number of histogram bins (ignored with --cdf). Default: one bin per distinct "
        f"integer value when the data span at most {MAX_INTEGER_BINS}, else 50.",
    )
    parser.add_argument(
        "--cdf", action="store_true", help="Plot the empirical cumulative distribution instead of a histogram"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output image file (default: <field>_sizes_<histogram|cdf>.png)"
    )
    args = parser.parse_args()

    plural, xlabel, stat_noun, basename = FIELD_INFO[args.field]
    output = args.output or f"{basename}_{'cdf' if args.cdf else 'histogram'}.png"

    print(f"Reading ExaEpi plotfile {args.plot_dir}")
    # yt.load() auto-detects ExaEpi's plotfiles as the plain BoxlibDataset, which does not look
    # for the "agents" particle subdirectory -- that auto-detection only happens in the
    # AMReXDataset subclass, so it is instantiated directly here.
    ds = AMReXDataset(args.plot_dir)

    if args.field == "community":
        sizes = community_sizes(ds)
    elif args.field == "neighborhood":
        sizes = neighborhood_sizes(ds)
    else:
        sizes = nborhoods_per_community(ds)
    sizes = pd.Series(sizes, name=stat_noun)

    print(f"Found {len(sizes)} {plural}")
    print(sizes.describe())

    fig, ax = plt.subplots(figsize=(5, 4))
    if args.cdf:
        sorted_sizes = np.sort(sizes.to_numpy())
        cumulative_frac = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        ax.step(sorted_sizes, cumulative_frac, where="post", color="blue")
        ax.set_ylabel("Cumulative fraction")
    else:
        # These are all integer counts, so by default give each distinct value its own bin
        # centered on it. A fixed bin count whose width isn't a whole number of integers
        # aliases badly -- adjacent bars then cover 1 or 2 integers depending on where the
        # edges land, which looks like structure in the data but isn't. An explicit --bins
        # still wins, and wide-spanning data (community sizes) falls back to 50.
        span = int(sizes.max() - sizes.min())
        if args.bins is not None:
            bins = args.bins
        elif span <= MAX_INTEGER_BINS:
            bins = np.arange(sizes.min() - 0.5, sizes.max() + 1.5, 1.0).tolist()
        else:
            bins = 50
        ax.hist(sizes, bins=bins, color="blue", alpha=0.7, edgecolor="black")
        ax.set_ylabel("Frequency")
    ax.set_xlabel(xlabel)
    #ax.set_title(f"Histogram of ExaEpi {args.field} sizes")
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"{plural.capitalize()}: {len(sizes)}\n"
        f"Mean {stat_noun}: {sizes.mean():.2f}\n"
        f"Median {stat_noun}: {sizes.median():.2f}\n"
        f"Max {stat_noun}: {sizes.max()}"
    )
    # A CDF is flat at 1.0 across the top, so the stats box goes in the empty bottom-right
    # corner instead of top-right (which is where the histogram's empty space usually is).
    text_y, valign = (0.35, "top") if args.cdf else (0.97, "top")
    ax.text(
        0.98,
        text_y,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment=valign,
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"{'CDF' if args.cdf else 'Histogram'} saved to {output}")


if __name__ == "__main__":
    main()
