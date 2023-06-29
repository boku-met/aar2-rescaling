#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_to_indicators = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/"
searchterm_indicators = "very_heavy_precipitation_days_20mm"

# for defining the point in the time period
quantile = 0.5

infiles = sorted(glob.glob(path_to_indicators+searchterm_indicators+"*.nc"))

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    time_sample = f1[searchterm_indicators].mean(skipna=True, dim="time")
    ensemble_mean = time_sample.quantile(0.5, dim="ens", skipna=True)
    return ensemble_mean.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles)
f1 = xr.open_dataset(infiles[0])
vis_data = xr.DataArray(par_results, coords={"GWL":["1.5°C","2.0°C", "3.0°C","4.0°C"], "y": f1.y, "x": f1.x})

lats = xr.DataArray(f1.lat[:,0].values, coords={"y":f1.y})
lons = xr.DataArray(f1.lon[0,:].values, coords = {"x": f1.x})
vis_data.coords["lat"] = lats
vis_data.coords["lon"] = lons


g = vis_data.plot(x = "lon", y = "lat", col="GWL", col_wrap=2, aspect = 1.5, size = 3, 
                  cmap="YlOrBr", cbar_kwargs={"label": "No. of days in >= 5-day dry periods"}, levels = 7)
plt.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_cdd_gwl.png", dpi=300)