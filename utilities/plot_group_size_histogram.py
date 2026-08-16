#!/usr/bin/env python

"""Plot a histogram of ExaEpi community or neighborhood sizes from a run's plotfile.

Community size is the number of agents whose home is in a given community (one AMReX
grid cell == one community in ExaEpi). This is read directly from the "comm" and
"total" mesh fields written to every plotfile, so it works for a plotfile from any day.

Neighborhood size is the number of agents in a given (community, neighborhood) pair --
neighborhood IDs are only unique within a community, so agents are grouped by the
(home_i, home_j, nborhood) triple. The per-agent "nborhood" field is only written to the
first plotfile of a run (typically plt00000), so --field neighborhood requires that file.
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


def neighborhood_sizes(ds):
    ptype = "agents"
    particle_fields = {f[1] for f in ds.field_list if f[0] == ptype}
    required = {"particle_home_i", "particle_home_j", "particle_nborhood"}
    missing = required - particle_fields
    if missing:
        sys.exit(
            f"Plotfile is missing per-agent field(s) {sorted(missing)} needed for neighborhood "
            "sizes; these are only written to the first plotfile of a run (e.g. plt00000)."
        )
    ad = ds.all_data()
    df = pd.DataFrame(
        {
            "home_i": np.rint(np.asarray(ad[(ptype, "particle_home_i")])).astype(int),
            "home_j": np.rint(np.asarray(ad[(ptype, "particle_home_j")])).astype(int),
            "nborhood": np.rint(np.asarray(ad[(ptype, "particle_nborhood")])).astype(int),
        }
    )
    return df.groupby(["home_i", "home_j", "nborhood"]).size().to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot_dir", "-p", required=True, help="ExaEpi plotfile directory, e.g. plt00000")
    parser.add_argument(
        "--field",
        "-f",
        choices=["community", "neighborhood"],
        default="community",
        help="Which grouping to histogram (default: community)",
    )
    parser.add_argument("--bins", "-b", type=int, default=50, help="Number of histogram bins (ignored with --cdf)")
    parser.add_argument(
        "--cdf", action="store_true", help="Plot the empirical cumulative distribution instead of a histogram"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output image file (default: <field>_sizes_<histogram|cdf>.png)"
    )
    args = parser.parse_args()

    output = args.output or f"{args.field}_sizes_{'cdf' if args.cdf else 'histogram'}.png"

    print(f"Reading ExaEpi plotfile {args.plot_dir}")
    # yt.load() auto-detects ExaEpi's plotfiles as the plain BoxlibDataset, which does not look
    # for the "agents" particle subdirectory -- that auto-detection only happens in the
    # AMReXDataset subclass, so it is instantiated directly here.
    ds = AMReXDataset(args.plot_dir)

    if args.field == "community":
        sizes = community_sizes(ds)
    else:
        sizes = neighborhood_sizes(ds)
    sizes = pd.Series(sizes, name="size")

    print(f"Found {len(sizes)} {args.field} groups")
    print(sizes.describe())

    label = args.field.capitalize()
    plural = "Communities" if args.field == "community" else "Neighborhoods"

    fig, ax = plt.subplots(figsize=(5, 3))
    if args.cdf:
        sorted_sizes = np.sort(sizes.to_numpy())
        cumulative_frac = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        ax.step(sorted_sizes, cumulative_frac, where="post", color="blue")
        ax.set_ylabel("Cumulative fraction")
    else:
        ax.hist(sizes, bins=args.bins, color="blue", alpha=0.7, edgecolor="black")
        ax.set_ylabel("Frequency")
    ax.set_xlabel(f"{label} size (number of agents)")
    #ax.set_title(f"Histogram of ExaEpi {args.field} sizes")
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"{plural}: {len(sizes)}\n"
        f"Mean size: {sizes.mean():.2f}\n"
        f"Median size: {sizes.median():.2f}\n"
        f"Max size: {sizes.max()}"
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
