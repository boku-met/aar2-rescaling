#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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

if __name__ == '__main__':
    # create and configure the process pool
    with Pool(14) as pool:
        # execute tasks, block until all completed
        par_results = pool.map(paralell_loop, infiles_cm5)
    # process pool is closed automatically

f1 = xr.open_dataset(infiles_cm5[0])
vis_data_cm5 = [x.flatten() for x in par_results]

if __name__ == '__main__':
    # create and configure the process pool
    with Pool(14) as pool:
        # execute tasks, block until all completed
        par_results = pool.map(paralell_loop, infiles_cm6)
    # process pool is closed automatically

f1 = xr.open_dataset(infiles_cm6[0])
vis_data_cm6 = [x.flatten() for x in par_results]

vis_data = []
for cm5, oeks in zip(vis_data_cm5, vis_data_cm6):
    vis_data.append(oeks)
    vis_data.append(cm5)

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
fig, axs = plt.subplots(figsize = (11, 6))
axs.boxplot(vis_data, positions=[1,2, 3.5, 4.5, 6,7,8.5,9.5], widths=0.8,patch_artist=True,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "green", "linewidth": 1.0})

axs.tick_params(labelsize=14)
axs.set_ylabel("Change in number of days in Kysely periods (Vienna, AT)", fontsize = 14, labelpad=10)
axs.set_xlabel("Ensembles grouped after GWL", fontsize = 14, labelpad=15)
axs.set_xticklabels(["1.5°C(CMIP5)", "1.5°C(CMIP6)", "2.0°C(CMIP5)", "2.0°C(CMIP6)",
                     "3.0°C(CMIP5)", "3.0°C(CMIP6)", "4.0°C(CMIP5)", "4.0°C(CMIP6)"],
                    rotation=45, fontsize=12)
axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
#axs.set(axisbelow=True)
plt.title("Annual mean temperature anomalies in Austria relative to 1991-2020\nfor OEKS15 and CMIP5 background models at four GWLs", fontsize = 16, pad = 15)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/kysely_vienna_cm5_cm6_bp.png"
plt.savefig(outpath,dpi=300, bbox_inches ="tight")