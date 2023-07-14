#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def check_isfile(fname):
    if os.path.isfile(fname):
        print("File {0} already exists. Overwriting...".format(fname))
        os.remove(fname)
    file_out = open(fname, mode="xt", encoding="utf-8", newline="\n")
    return file_out

# user specified paths and data
gwls = [0.856, 1.362, 2.422, 3.381]
path_oeks15 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/oeks15_anomalies/"
outf = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP6_OEKS15.csv"

# create summary csv file for CMIP5 OEKS15 GWLs
outfile = check_isfile(outf)
outfile.write("OEKS15 ensemble member;Mean year per GWL;;;;Period per GWL\n")
outfile.write(";1.5°C;2.0°C;3.0°C;4.0°C;1.5°C;2.0°C;3.0°C;4.0°C\n")
for rcp in ["rcp26", "rcp45", "rcp85"]:
    infiles = sorted(glob.glob(path_oeks15+"tas_*"+rcp+"*.csv"))
    for file in infiles:
        # open file and load in DataArray for processing
        f1 = open(file, mode="rt")
        f1 = [x.replace("\n","") for x in f1]
        data = [x.split(";")[1] for x in f1[1:]]
        years = [x.split(";")[0] for x in f1[1:]]
        time = np.array(years, np.datetime64)
        anomalies = xr.DataArray(np.array(data, dtype=np.float32), coords={"time":time})

        # calculate  20-year rolling mean
        anomalies_smooth = anomalies.rolling(time=20, center=True, min_periods=20).mean(skipna=True)

        gwl_list = []
        mean_years = []
        for gwl in gwls:
            try:
                timeind = (anomalies_smooth.values >= gwl).nonzero()[0][0]
                mean_year = anomalies_smooth[timeind].time.dt.year.values
                period = "{0}-{1}".format(mean_year-10, mean_year+9)
                mean_years.append(str(mean_year))
                gwl_list.append(period)
            except IndexError:
                mean_years.append("n/a")
                gwl_list.append("n/a")

        modelname = file.split("/")[-1].replace("tas_","").replace("_annual_anomalies_1991-2020.csv","")
        
        # create data to write to list
        outfile.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(modelname, *mean_years, *gwl_list))

outfile.close()
print("Writing file {0} complete!".format(outf))