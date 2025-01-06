import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
from matplotlib import cm
import os
import fiona
import geopandas as geopd
import pysal as ps
import shapefile
import pandas as pd
from shapely.geometry import shape
import numpy as np
import os
import sys
import re


def genWorkerflow(dirPath: str, state: str):
        sgDir= dirPath + "/"
        gdf = geopd.GeoDataFrame(columns=["geoid_o", "geoid_d", "lng_o", "lat_o", "lng_d", "lat_d", "date", "visitor_flows", "pop_flows", "stateFIPS_o", "stateFIPS_d", "countyFIPS_o", "countyFIPS_d"])
        stateFIPS= {'AL': 1, 'AK': 2, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8, 'CT':9 , 'DE': 10, 'DC': 11, 'FL': 12, 'GA': 13, 'HI': 15, 'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19, 'KS': 20, 'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24, 'MA': 25, 'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30, 'NE': 31, 'NV': 32, 'NH': 33, 'NJ': 34, 'NM': 35, 'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39, 'OK': 40, 'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45, 'SD': 46, 'TN': 47, 'TX': 48, 'UT': 49, 'VT': 50, 'VA': 51, 'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56}
        wfFiles= [os.path.join(sgDir, f) for f in os.listdir(sgDir) if f.endswith(".csv")]
        for wfFile in wfFiles:
                 df000 = pd.read_csv(wfFile)
                 df000["stateFIPS_o"]= (df000["geoid_o"]/1e9).astype(int)
                 df000["stateFIPS_d"]= (df000["geoid_d"]/1e9).astype(int)
                 df000= df000.loc[df000['stateFIPS_o'] == stateFIPS[state]]
                 df000= df000.loc[df000['stateFIPS_d'] == stateFIPS[state]]
                 df000["countyFIPS_o"] = (df000["geoid_o"]/1e6).astype(int)
                 df000["countyFIPS_d"] = (df000["geoid_d"]/1e6).astype(int)
                 df000["flows"]= df000["visitor_flows"] + df000["pop_flows"]
                 gdf= gdf._append(df000)
        gdf= gdf[["countyFIPS_o", "countyFIPS_d", "flows"]]
        gdf= gdf.groupby(["countyFIPS_o", "countyFIPS_d"]).sum()
        data = gdf.to_records()
        output_file = open(state+"-sg-wf.bin", 'wb')
        data.tofile(output_file)

if __name__ == "__main__":
        argList= sys.argv[1:]
        if len(argList) < 2: print("usage: python parseSafeGraph.py /path/to/SafeGraphDir state")
        sgFile  = str(argList[0])
        outF    = str(argList[1])
        genWorkerflow(sgFile, outF)
