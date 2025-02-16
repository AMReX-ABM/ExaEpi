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
print(ad["total"].sum())
print(ad["infected"].sum())
print(ad["immune"].sum())

# for col in ["total", "infected", "immune", "susceptible", "dead"]:
#    pd.DataFrame(ad[col]).to_csv(col + ".csv")

xdim, ydim = ds.domain_dimensions[0:2]
print("dimensions", xdim, ydim)

# pos_x = ad["particle_position_x"]
# pos_y = ad["particle_position_y"]
# home_i = ad["particle_home_i"]
# home_j = ad["particle_home_j"]
# pd.DataFrame({"x": pos_x, "y": pos_y, "home_i": home_i, "home_j": home_j}).to_csv("home.csv", sep=" ")

# z = ad["total"].reshape(ydim, xdim)
# xdim, ydim = z.shape
# z = np.random.rand(xdim, ydim)
#
# high = np.where(ad["total"] > 8000)
# pd.DataFrame(high).to_csv("total-high-flat.csv", sep="\t")

agents = pd.DataFrame({"x": ad["particle_position_x"], "y": ad["particle_position_y"], "status": ad["particle_status"]})
agents.to_csv("agents.csv")

aggr_agents = agents.value_counts().reset_index()
aggr_agents.to_csv("aggr_agents.csv")

infected_agents = aggr_agents[aggr_agents["status"] == 1]
immune_agents = aggr_agents[aggr_agents["status"] == 2]
susceptible_agents = aggr_agents[aggr_agents["status"] == 3]
dead_agents = aggr_agents[aggr_agents["status"] == 4]

# tot = ad["total"].reshape(xdim, ydim)
# x, y = np.where(tot > 0)
# z = tot[x, y]

plt.figure(figsize=[float(xdim) / 500, float(ydim) / 500])

agents_to_plt = infected_agents
counts = agents_to_plt["count"]
max_count = counts.max()
print("max count", max_count)
counts = np.log(counts) / np.log(max_count)
sc = plt.scatter(
    agents_to_plt["x"], agents_to_plt["y"], c=counts, s=counts * 50, alpha=1.0, cmap=plt.colormaps["plasma"], ec="none"
)
plt.colorbar(sc)
plt.savefig("test.pdf")
