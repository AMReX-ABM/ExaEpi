import pylab as plt
import sys
import numpy as np
import pandas as pd
import argparse
from yt.frontends.boxlib.api import AMReXDataset


fn = sys.argv[1]

ds = AMReXDataset(fn)

print(ds.field_list)

ad = ds.all_data()

xdim, ydim = ds.domain_dimensions[0:2]
print("dimensions", xdim, ydim)


agents = pd.DataFrame({"x": ad["particle_position_x"], "y": ad["particle_position_y"], "status": ad["particle_status"]})
agents.to_csv("agents.csv")

aggr_agents = agents.value_counts().reset_index()
aggr_agents.to_csv("aggr_agents.csv")

# max_count = np.log(aggr_agents["count"].max())
# hard-coded for NM
max_count = np.log(30000)

never_infected_agents = aggr_agents[aggr_agents["status"] == 0]
infected_agents = aggr_agents[aggr_agents["status"] == 1]
immune_agents = aggr_agents[aggr_agents["status"] == 2]
susceptible_agents = aggr_agents[aggr_agents["status"] == 3]
dead_agents = aggr_agents[aggr_agents["status"] == 4]

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
        cmap=plt.colormaps["plasma"],
        ec="none",
        vmin=0,
        vmax=max_count,
    )
    plt.colorbar(sc, ax=axes[i])


plt.tight_layout()
plt.savefig("plot-" + fn + ".pdf")
