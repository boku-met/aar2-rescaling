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


path_to_indicators_cm5 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
path_to_indicators_cm6 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP6/"
searchterm_indicators = "kysely_periods_noofdays"
searchterm_refperiod = "kysely_periods_noofdays_reference_period_1991_2020"


xmin = 614384
xmax = 640853
ymin = 476883
ymax = 493485

# for defining the point in the time period
quantile = 0.5

infiles_cm5 = sorted(glob.glob(path_to_indicators_cm5+searchterm_indicators+"*.nc"))
infiles_cm6 = sorted(glob.glob(path_to_indicators_cm6+searchterm_indicators+"*.nc"))


def paralell_loop(f):
    f1 = xr.open_dataset(f)
    #anomalies = f1[searchterm_indicators]
    #ref_period = f1[searchterm_refperiod]
    #rel_an = (anomalies / ref_period) * 100
    time_sample = f1[searchterm_indicators].mean(dim="time", skipna=True)
    area_sample = time_sample.sel(y=slice(ymin, ymax), x=slice(xmin, xmax)).mean(dim=("y","x"), skipna=True)
    return area_sample.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles_cm5)
f1 = xr.open_dataset(infiles_cm5[0])
vis_data_cm5 = [x.flatten() for x in par_results]

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    #anomalies = f1[searchterm_indicators]
    #ref_period = f1[searchterm_refperiod]
    #rel_an = (anomalies / ref_period) * 100
    time_sample = f1[searchterm_indicators].mean(dim="time", skipna=True)
    area_sample = time_sample.sel(y=slice(ymin, ymax), x=slice(xmin, xmax)).mean(dim=("y","x"), skipna=True)
    return area_sample.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles_cm6)
f1 = xr.open_dataset(infiles_cm6[0])
vis_data_cm6 = [x.flatten() for x in par_results]

fig, axs = plt.subplots(figsize = (5, 4))
axs.set_ylabel("Change of annual precipitation sum (%)")
axs.boxplot(vis_data, positions=[2,4,6,8], widths=1.1,patch_artist=True,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "green", "linewidth": 1.0}, 
            labels=["1.5°C", "2.0°C", "3.0°C", "4.0°C"])

axs.set_xlabel("GWLs")
fig.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/pr_vienna_bp.png", dpi=300)


# comparison plots
fig, axs = plt.subplots(figsize = (12, 6))
axs.set_ylabel("Number of days in Kysely periods (Vienna, AT)")
axs.boxplot(vis_data, positions=[1,2, 3.5, 4.5, 6,7,8.5,9.5], widths=0.8,patch_artist=True,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "green", "linewidth": 1.0}, 
            labels=["1.5°C(CM5)", "1.5°C(CM6)", "2.0°C(CM5)", "2.0°C(CM6)", "3.0°C(CM5)", "3.0°C(CM6)", "4.0°C(CM5)", "4.0°C(CM6)"])

axs.set_xlabel("GWLs")
fig.savefig("/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/kysely_vienna_cm5_cm6_bp.png", dpi=300)