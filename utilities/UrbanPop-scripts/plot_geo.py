import pylab as plt
import sys
import numpy as np
import pandas as pd
import argparse
from yt.frontends.boxlib.api import AMReXDataset
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mp


def plot_lng_lat(fname, xdim, ydim):
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(float(xdim) / 500, float(ydim) / 500))

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    titles = ["Never infected", "Infected", "Immune", "Dead"]
    agents_to_plt = [
        never_infected_agents,
        infected_agents,
        immune_agents,
        dead_agents,
    ]
    axes = [ax1, ax2, ax3, ax4]

    for i in range(len(axes)):
        print(titles[i], agents_to_plt[i]["count"].sum())
        axes[i].set_title(titles[i])
        counts = np.log(agents_to_plt[i]["count"])
        sc = axes[i].scatter(
            agents_to_plt[i]["x"],
            agents_to_plt[i]["y"],
            c=counts,
            s=2,
            cmap=plt.colormaps["RdPu"],
            ec="none",
            vmin=0,
            vmax=max_count,
        )
        plt.colorbar(sc, ax=axes[i])

    plt.tight_layout()
    plt.savefig("plot-" + fname + ".pdf")


parser = argparse.ArgumentParser(description="Plot UrbanPop ExaEpi outputs")
# parser.add_argument("--output", "-o", required=True, help="Output file")
parser.add_argument("--plot_dir", "-p", required=True, help="Plot directory")
parser.add_argument(
    "--shape_files",
    "-s",
    required=True,
    nargs="+",
    help="Shape files census block group shape files (.shp). Available from\n"
    + "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups",
)
parser.add_argument(
    "--geoid_file",
    "-g",
    required=True,
    help="File containing GEOIDs and lng/lat. This is produced by an ExaEpi run and has extension .geoids.csv",
)
parser.add_argument(
    "--output-format",
    "-o",
    default="pdf",
    help="Output format for plot files, can be any extension supported by matplotlib, e.g. png, pdf",
)
args = parser.parse_args()

print("Reading ExaEpi data from directory", args.plot_dir)
ds = AMReXDataset(args.plot_dir)
# print(ds.field_list)

ad = ds.all_data()

agents = pd.DataFrame({"x": ad["particle_position_x"], "y": ad["particle_position_y"], "status": ad["particle_status"]})
agents.to_csv("agents.csv")

aggr_agents = agents.value_counts().reset_index()

shp_dfs = []
for fname in args.shape_files:
    if not fname.endswith(".shp"):
        print("WARNING: file", fname, "passed with --shape_files does not appear to be a shapefile with .shp extension")
        continue
    print("Reading data from", fname)
    shp_dfs.append(gp.read_file(fname))

shp_data = pd.concat(shp_dfs)
shp_data.GEOID10 = shp_data.GEOID10.astype("int64")
print("Read in", len(shp_data), "Census block groups")

print("Reading GEOIDs from", args.geoid_file)
geoids_df = pd.read_csv(args.geoid_file)
geoids_df = geoids_df.rename(columns={"GEOID": "GEOID10", "lng": "x", "lat": "y"})
# ensure all datasets have the same lat/lon resolution
decimals = 3
aggr_agents.x = np.round(aggr_agents.x, decimals)
aggr_agents.y = np.round(aggr_agents.y, decimals)
geoids_df.x = np.round(geoids_df.x, decimals)
geoids_df.y = np.round(geoids_df.y, decimals)


max_count = 30000  # never_infected_agents["count"].max()

xdim, ydim = ds.domain_dimensions[0:2]
_, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(float(xdim) / 300, float(ydim) / 300))

status_list = {
    0: ["never_infected", ax1, "Blues"],
    1: ["infected", ax2, "OrRd"],
    2: ["immune", ax3, "Greens"],
    4: ["dead", ax4, "OrRd"],
}

for i, status in status_list.items():
    df = aggr_agents[aggr_agents.status == i]
    df = pd.merge(df, geoids_df, on=["x", "y"], how="outer")
    df.fillna(0, inplace=True)
    df.to_csv(status[0] + ".csv")
    df = pd.merge(shp_data, df, on=["GEOID10"], how="inner")
    df[["x", "y", "GEOID10", "count"]].to_csv(status[0] + "-merged.csv")
    print("Block groups with " + status[0], len(df), "count", df["count"].sum())
    # df.boundary.plot(ax=axes[i], lw=0.1)
    ax = status[1]
    # Some decent colormaps: RdPu OrRd Greys
    df.plot(ax=ax, column="count", cmap=status[2], legend=True, norm=mp.colors.LogNorm(vmin=1.0, vmax=max_count))
    ax.set_xlim([ds.domain_left_edge[0], ds.domain_right_edge[0]])
    ax.set_ylim([ds.domain_left_edge[1], ds.domain_right_edge[1]])
    ax.set_title(status[0])
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["font.size"] = 16
plt.tight_layout()
plt_fname = "plot-" + args.plot_dir + "." + args.output_format
print("Plotting results to", plt_fname)
plt.savefig(plt_fname)
