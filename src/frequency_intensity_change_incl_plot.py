import os
import glob
import copy
from multiprocessing.pool import Pool
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

gwls = [1.5, 2.0, 3.0, 4.0]

# infiles paths
path_gwl_ind = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
path_obs = "/sto0/data/Results/Indicators/prmax_SPARTACUS_annual_1961-2021.nc"
path_mask_spart = "/sto0/data/Input/Gridded/SPARTACUS/V2.1/TN/TN_19610101_20221031.nc"

# variable names
indicator_searchterm = "prmax_CMIP5_GWL_"
varname = "prmax"
varname_spart = "prmax"

ds_mask_spart = xr.open_dataset(path_mask_spart)
mask_spart = xr.where(ds_mask_spart.mask == 1, 1, np.nan)

infiles_gwl = sorted(glob.glob(path_gwl_ind+indicator_searchterm+"*.nc"))

# calculate values for refperiod
f_obs = xr.open_dataset(path_obs)
#f_obs = f_obs.resample(time="A", skipna=True).sum()
ds_obs_6190 = f_obs[varname_spart].sel(time=slice("1961","1990")) * mask_spart
ds_obs_9120 = f_obs[varname_spart].sel(time=slice("1991","2020")) * mask_spart

mean_obs_refperiod = ds_obs_9120.mean(dim="time", skipna=True)

areamean_obs_6190 = ds_obs_6190.mean(dim=("y","x"), skipna=True)
areamean_obs_9120 = ds_obs_9120.mean(dim=("y","x"), skipna=True)

mean_event_obs_6190 = areamean_obs_6190.mean(dim="time", skipna=True)
p90_event_obs_6190 = areamean_obs_6190.quantile(0.9, dim="time", skipna=True)

mean_event_obs_9120 = areamean_obs_9120.mean(dim="time", skipna=True)
p90_event_obs_9120 = areamean_obs_9120.quantile(0.9, dim="time", skipna=True)

print("Mean event 1961-1990;{0}".format(mean_event_obs_6190.values))
print("Mean event 1991-2020;{0}".format(mean_event_obs_9120.values))
print("10-year event 1961-1990;{0}".format(p90_event_obs_6190.values))
print("10-year event 1991-2020;{0}".format(p90_event_obs_9120.values))

#calculate values for gwl period

for file, gwl in zip(infiles_gwl, gwls):
    f_gwl = xr.open_dataset(file)
    ds_gwl = f_gwl[varname+"_anomalies"]

    abs_gwl_period = mean_obs_refperiod + ds_gwl
    areamean_abs_gwl_period = abs_gwl_period.mean(dim=("y","x"), skipna=True)
    mean_event_gwl = areamean_abs_gwl_period.mean(dim=("time"), skipna=True)
    p90_event_gwl = areamean_abs_gwl_period.quantile(0.9, dim="time", skipna=True)

    print("Mean event at GWL{0} (Lower);{1}".format(gwl, mean_event_gwl.quantile(0.1, dim="ens", skipna=True).values))
    print("Mean event at GWL{0} (Median);{1}".format(gwl, mean_event_gwl.quantile(0.5, dim="ens", skipna=True).values))
    print("Mean event at GWL{0} (Upper);{1}".format(gwl, mean_event_gwl.quantile(0.9, dim="ens", skipna=True).values))

    print("10-year event at GWL{0} (Lower);{1}".format(gwl, p90_event_gwl.quantile(0.1, dim="ens", skipna=True).values))
    print("10-year event at GWL{0} (Median);{1}".format(gwl, p90_event_gwl.quantile(0.5, dim="ens", skipna=True).values))
    print("10-year event at GWL{0} (Upper);{1}".format(gwl, p90_event_gwl.quantile(0.9, dim="ens", skipna=True).values))
