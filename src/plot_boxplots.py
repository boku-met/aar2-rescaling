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

path_to_indicators = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/"
searchterm_indicators = "pr_"
varname1 = "pr_anomalies"
varname2 = "pr_reference_period_1991_2020"

xmin = 614384
xmax = 640853
ymin = 476883
ymax = 493485

# for defining the point in the time period
quantile = 0.5

infiles = sorted(glob.glob(path_to_indicators+searchterm_indicators+"*.nc"))

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    anomalies = f1[varname1]
    ref_period = f1[varname2]
    rel_an = (anomalies / ref_period) * 100
    area_sample = rel_an.sel(y=slice(ymin, ymax), x=slice(xmin, xmax)).mean(dim=("y","x"), skipna=True)
    return area_sample.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles)
vis_data = [x.flatten() for x in par_results]

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