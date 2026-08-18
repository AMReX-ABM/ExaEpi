import numpy as np
import pandas as pd
import h5py
import sys
import os

filename = sys.argv[1]

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): 
                yield (path, item)
            elif isinstance(item, h5py.Group): 
                yield from h5py_dataset_iterator(item, path)

            if key == "dataframe":
                group = hdf_file["/"+key]
                data_dict = {}
                # Iterate through datasets within the group and convert to numpy arrays
                for k in group.keys():
                    if isinstance(group[k], h5py.Dataset):
                        data_dict[k] = group[k][()]
                df = pd.DataFrame.from_dict(data_dict)
                df = df[['geometry_id', 'STATEFP', 'COUNTYFP', 'hurs', 'huss', 'model', 'n_observations', 'pr', 'rlds', 'rsds', 'scenario', 'sfcWind', 'tas', 'tasmax', 'tasmin', 'week', 'week_start']]
                df= df.sort_values(by=['STATEFP', 'COUNTYFP'])
                name, ext = os.path.splitext(filename)
                df.to_csv(name+'.csv', index=False)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path

with h5py.File(filename, "r") as f:
    for dset in traverse_datasets(f):
        data = f[dset][()]
        #print(data)
