import numpy as np
import pandas as pd
import math
import os
import sys
import re

def scaleCensus(filename: str, scaleF: float):
        df= pd.read_csv(filename, sep='\s+', names=["unit", "pop", "worker", "FIP", "tract", "N5", "N17", "N29", "N64", "N65+", "H1", "H2", "H3", "H4", "H5", "H6", "H7"])
        df= df[1:]
        df["unit"]= df["unit"].astype(int)
        df["pop"]= (df["pop"]*scaleF).astype(int)
        df["worker"]= (df["worker"]*scaleF).astype(int)
        df["FIP"]= df["FIP"].astype(int)
        df["tract"]= df["tract"].astype(int)
        df["N5"]= (df["N5"]*scaleF).astype(int)
        df["N17"]= (df["N17"]*scaleF).astype(int)
        df["N29"]= (df["N29"]*scaleF).astype(int)
        df["N64"]= (df["N64"]*scaleF).astype(int)
        df["N65+"]= (df["N65+"]*scaleF).astype(int)
        df["H1"]= (df["H1"]*scaleF).astype(int)
        df["H2"]= (df["H2"]*scaleF).astype(int)
        df["H3"]= (df["H3"]*scaleF).astype(int)
        df["H4"]= (df["H4"]*scaleF).astype(int)
        df["H5"]= (df["H5"]*scaleF).astype(int)
        df["H6"]= (df["H6"]*scaleF).astype(int)
        df["H7"]= (df["H7"]*scaleF).astype(int)
        #write to a new file
        name=os.path.splitext(filename)[0]
        ext=os.path.splitext(filename)[1]
        newfilename= name+"_scaled"+ext
        numRows= len(df.index)
        with open(newfilename, 'w') as file:
                file.write(str(numRows)+'\n')
        df.to_csv(newfilename, mode='a', index=False, header=False, sep='\t')

def scaleWorkerflow(filename: str, scaleF: float):
        name=os.path.splitext(filename)[0]
        ext=os.path.splitext(filename)[1]
        newfilename= name+"_scaled"+ext
        if(ext=='.bin'):
                dt = np.dtype([('s', 'i4'), ('d', 'i4'), ('w', 'i4')])
                data = np.fromfile(filename, dtype=dt)
                data["w"]= np.rint(data["w"]*scaleF)
                output_file = open(newfilename, 'wb')
                data.tofile(output_file)
        else: #non binary formats, e.g. .dat
                df= pd.read_csv(filename, sep='\s+', names=["s",  "d", "w"])
                df["s"]= df["s"].astype(int)
                df["d"]= df["d"].astype(int)
                df["w"]= (df["w"]*scaleF).astype(int)
                df.to_csv(newfilename, mode='w', index=False, header=False, sep=' ')

def main(censusFileName: str, workerFileName: str, scaleF: float):
        scaleCensus(censusFileName, scaleF)
        scaleWorkerflow(workerFileName, scaleF)

if __name__ == "__main__":
        argList= sys.argv[1:]
        if len(argList) < 3: print("usage: python scaleCensus.py /path/to/CensusFile /path/to/WorkerFlowFile scaleFactor")
        censusFile  = str(argList[0])
        workerFile  = str(argList[1])
        scaleFactor  = float(argList[2])
        main(censusFile, workerFile, scaleFactor)
