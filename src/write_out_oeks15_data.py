#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import glob
import numpy as np
import xarray as xr

rcps = ["rcp26", "rcp45", "rcp85"]
for rcp in rcps:
    print(rcp)
    infiles = sorted(glob.glob("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/oeks15_anomalies/pr_SDM*"+rcp+"*.csv"))
    for f in infiles:
        f1 = open(f, mode="rt")
        data_list = [l.replace("\n","") for l in f1]
        yrs = [x.split(";")[0] for x in data_list]
        data = [x.split(";")[1] for x in data_list]
        yrs = yrs[1:]
        data = data[1:]
        time = np.array(yrs, dtype=np.datetime64)
        data_array = xr.DataArray(np.array(data, dtype = np.float32), coords={"time": time})
        data_2050 = data_array.sel(time=slice("2081","2097")).mean(skipna=True)
        print(data_2050.values)