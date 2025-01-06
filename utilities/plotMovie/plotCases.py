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
import yt
from yt.frontends import boxlib
from yt.frontends.boxlib.data_structures import AMReXDataset
from collections import defaultdict
import os
import sys
import re


def write_rawCases(fn: str):
    yt.set_log_level(50)
    ds = AMReXDataset(fn)
    ad = ds.all_data()

    inf = ad['infected'].d
    tot = ad['total'].d
    fips = ad['FIPS'].d
    tract = ad['Tract'].d

    d_inf = defaultdict(int)
    for i, to, f, tr in zip(inf, tot, fips, tract):
       code = 1000000*int(f) + int(tr)
       if code < 0:
           continue
       d_inf[code] += i

    output_file = open(fn+".csv", 'w')
    stdout =sys.stdout
    sys.stdout = output_file
    for k, v in d_inf.items():
        print("{:011d}".format(k), ",", v)
    sys.stdout= stdout
    return output_file

argList= sys.argv[1:]
zoom=""
group=""
BayAreaDataFormat=""
if(len(argList)<5): #5 is the minimum mandatory arguments
    print("Some mandatory argument is missing")
    exit
if len(argList)>= 1: infCasesDir= argList[0]+"/"
if len(argList)>= 2: CensusDir  = argList[1]+"/"
if len(argList)>= 3: outputMetric  = argList[2]
if len(argList)>= 4: datanorm  = argList[3]
if len(argList)>= 5: group  = argList[4]
if len(argList)>= 6: zoom  = argList[5] #format: "startTime, stopTime, target x, target y, zoomFactor"
if len(argList)>= 7: BayAreaDataFormat= argList[6]

gdf = geopd.GeoDataFrame(columns=["STATEFP00", "COUNTYFP00", "TRACTCE00", "CTIDFP00", "NAME00", "NAMELSAD00", "MTFCC00", "FUNCSTAT00", "ALAND00", "AWATER00", "INTPTLAT00", "INTPTLON00", "geometry"])
#for Bay Area dataset, use the following format:
if(BayAreaDataFormat=="YES"): gdf = geopd.GeoDataFrame(columns=["geoid", "name_", "namelsad", "mtfcc", "funcstat", "aland", "awater", "intptlat", "intptlon", "SHAPE_Leng", "SHAPE_Area"])

for f in os.listdir(infCasesDir):
    if f.startswith("plt")==True and f.endswith(".csv")==False:
        write_rawCases(infCasesDir+f)

infCasesFiles= [os.path.join(infCasesDir, f) for f in os.listdir(infCasesDir) if f.endswith(".csv")]

for file in os.listdir(CensusDir):
    filename = CensusDir + file + "/" + file
    myshp = open(filename+".shp", "rb")
    mydbf = open(filename+".dbf", "rb")
    myprj = open(filename+".prj", "rb")
    r = shapefile.Reader(shp=myshp, dbf=mydbf, prj=myprj)
    attributes, geometry = [], []
    field_names = [field[0] for field in r.fields[1:]]
    for row in r.shapeRecords():
        geometry.append(shape(row.shape.__geo_interface__))
        attributes.append(dict(zip(field_names, row.record)))
    df = geopd.GeoDataFrame(data = attributes, geometry = geometry)
    gdf= gdf._append(df)
    r.close()
print(gdf)

if(BayAreaDataFormat=="NO"): gdf['fips'] = gdf["CTIDFP00"].astype(int)
else: gdf['fips'] = gdf["geoid"].astype(int)
gdf["county"]= gdf["fips"]//1000000
gdf0= gdf.copy()
gdf1= gdf.copy()
gdf0['per'] = 0.0
gdf1['per'] = 0.0
gdf1.set_index('fips',inplace=True)


sorted_infCasesFiles= sorted(infCasesFiles, key=lambda x: int( re.findall('\d+', x)[0]))
for caseFile in sorted_infCasesFiles:
    idx = int(re.findall('\d+', caseFile)[0])
    print("processing file", caseFile)
    df000 = pd.read_csv(caseFile, header=None, names=["fips", "per"], dtype={"fips":int, "per":float})
    #df000.set_index("fips",inplace=True)
    gdf_join = pd.merge(gdf,df000, left_on="fips", right_on="fips", how='inner')
    if(group=="COUNTY"):
        if(datanorm=="RAW"):
            df000["county"]= df000["fips"]//1000000
            df000.drop("fips", inplace=True, axis=1)
            df000= df000.groupby("county").sum()
            gdf_join = pd.merge(gdf,df000, left_on='county', right_on='county', how='inner')
    #gdf_join.set_index('fips',inplace=True)
    columns= gdf1.columns.values
    columns= columns[:-1].tolist() #exclude per
    if(outputMetric=="DAILY"):
        gdf1= pd.concat([gdf0, gdf_join]).groupby(columns).sum().reset_index()
    if(outputMetric=="CUML"):
        gdf1= pd.concat([gdf0, gdf_join]).groupby(columns).sum().reset_index()

    fig, ax = plt.subplots(figsize=(10,10))

    gdfplot= geopd.GeoDataFrame(gdf1)
    #use cmap with a norm to convert a number to RGB
    #two slope norm
    #set center to 0, adjust the negative vmin to control how fast color changes from 0 to vmax
    if(datanorm=="RAW"):
        vmin = -1.0
        vmax = 25
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.000, vmax=vmax)
        cmap = cm.cool
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        gdfplot["per"] = gdfplot["per"] + 1 #avoid log(0)
        gdfplot["per"] = np.log2(gdfplot["per"])
        plt.colorbar(sm, ax=ax, ticks=np.linspace(0, 25, 6), label="Log2 Cumulative Cases", orientation="vertical")
    else:
        if(datanorm=="POP"):
            vmin = -10.0
            vmax = 1.0
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.000, vmax=vmax)
            cmap = cm.cool
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, ax=ax, ticks=np.linspace(0, 1, 11), label="Cumulative Cases per Population", orientation="vertical")
        else:
            print("INPUT METRIC NOT PROVIDED", norm)
            exit
    ax.xaxis.set_ticks_position('top')
    ax.set(aspect='equal', xticks=[], yticks=[])

    gdfplot.plot(column= 'per', ax = ax, norm=norm, cmap=cmap, lw=3)#, edgecolor='black',)#, legend=False)

    #adjust the position and zoom in
    if(zoom!="NO"):
        startTime, stopTime, targetX, targetY, zoomFactor= zoom.split(',')
        startTime= int(startTime)
        stopTime= int(stopTime)
        shiftTime=10 #we need 10 days to smoothly shift to the target position
        targetX= float(targetX)
        targetY= float(targetY)
        zoomFactor= float(zoomFactor)
        xrate= (0.5-targetX)/10
        yrate= (0.5-targetY)/10
        zoomrate= (1.0-1.0/zoomFactor)/(stopTime-shiftTime-startTime)

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        if(idx>=startTime and idx<(startTime+shiftTime)):
            #gradually shift to the interested local area
            ax.set_xlim((xlim[0]-(xlim[1]-xlim[0])*xrate*(idx-startTime), xlim[1]-(xlim[1]-xlim[0])*xrate*(idx-startTime)))
            ax.set_ylim((ylim[0]-(ylim[1]-ylim[0])*yrate*(idx-startTime), ylim[1]-(ylim[1]-ylim[0])*yrate*(idx-startTime)))
        if(idx>=(startTime+shiftTime) and idx<stopTime):
            #now all figures start looking at the interested local area
            ax.set_xlim((xlim[0]-(xlim[1]-xlim[0])*xrate*shiftTime, xlim[1]-(xlim[1]-xlim[0])*xrate*shiftTime) )
            ax.set_ylim((ylim[0]-(ylim[1]-ylim[0])*yrate*shiftTime, ylim[1]-(ylim[1]-ylim[0])*yrate*shiftTime) )
            #we want to zoom in a little more
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.set_xlim((xlim[0]+(xlim[1]-xlim[0])*zoomrate*(idx-startTime-shiftTime)/2, xlim[1]-(xlim[1]-xlim[0])*zoomrate*(idx-startTime-shiftTime)/2) )
            ax.set_ylim((ylim[0]+(ylim[1]-ylim[0])*zoomrate*(idx-startTime-shiftTime)/2, ylim[1]-(ylim[1]-ylim[0])*zoomrate*(idx-startTime-shiftTime)/2) )
        if(idx>=stopTime):
            #stay exactly where last figure was. No more shifting, no more zooming
            ax.set_xlim((xlim[0]-(xlim[1]-xlim[0])*xrate*shiftTime, xlim[1]-(xlim[1]-xlim[0])*xrate*shiftTime) )
            ax.set_ylim((ylim[0]-(ylim[1]-ylim[0])*yrate*shiftTime, ylim[1]-(ylim[1]-ylim[0])*yrate*shiftTime) )
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.set_xlim((xlim[0]+(xlim[1]-xlim[0])*zoomrate*(stopTime-shiftTime-startTime)/2, xlim[1]-(xlim[1]-xlim[0])*zoomrate*(stopTime-shiftTime-startTime)/2 ))
            ax.set_ylim((ylim[0]+(ylim[1]-ylim[0])*zoomrate*(stopTime-shiftTime-startTime)/2, ylim[1]-(ylim[1]-ylim[0])*zoomrate*(stopTime-shiftTime-startTime)/2 ))
    output= "casefile"+str(idx+1).zfill(5)+".png"
    plt.savefig(output)
    plt.close()
