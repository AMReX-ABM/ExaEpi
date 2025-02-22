import yt
from yt.frontends import boxlib
from yt.frontends.boxlib.api import AMReXDataset

import pylab as plt
import sys

fn = sys.argv[1]

ds = AMReXDataset(fn)

print(ds.field_list)

ad = ds.all_data()
print(ad["total"].sum())
print(ad["infected"].sum())
print(ad["immune"].sum())
print(ad["previously_infected"].sum())

plt.pcolormesh(ad["total"].reshape(ds.domain_dimensions[0:2]))
plt.savefig("test")
