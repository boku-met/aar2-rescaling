
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

path_obs = "/sto0/data/Results/Indicators/SPEI-3months_annual_min_AMJJAS_pyet_fao56_1961-2023_1km_ref1991-2020.nc"

ds_mask_spart = xr.open_dataset(path_obs)
mask_spart = xr.where(ds_mask_spart.SPEI[4,:,:] > -999, 1, np.nan)

# variable names
indicator_searchterm = "SPEI_CMIP5_GWL_"
ref_searchterm = "SPEI_single_years_refperiod_CMIP5_GWL_"
varname = "SPEI"
varname_spart = "SPEI"

infiles_gwl = sorted(glob.glob(path_gwl_ind+indicator_searchterm+"*.nc"))
infile_ref = sorted(glob.glob(path_gwl_ind+ref_searchterm+"*.nc"))

ds_mask_gwl = xr.open_dataset(infiles_gwl[2])
mask_gwl = xr.where(ds_mask_gwl.SPEI[0,4,:,:] > -999, 1, np.nan)

f_obs = xr.open_dataset(path_obs)
#f_obs = f_obs.resample(time="A", skipna=True).sum()
ds_obs_6190 = f_obs[varname_spart].sel(time=slice("1961","1990"))
ds_obs_9120 = f_obs[varname_spart].sel(time=slice("1991","2020"))

p90_event_obs_9120 = ds_obs_9120.quantile(0.1, dim="time", skipna=True)

#areamean_obs_9120 = p90_event_obs_9120.mean(dim=("y","x"), skipna=True)
exceeds_9120 = (xr.where(ds_obs_9120 < p90_event_obs_9120, 1, 0).mean(dim=("time"), skipna = True)) * mask_spart
mean_exceeds_9120 = exceeds_9120.mean(dim=("y","x"), skipna=True)
exceeds_6190 = (xr.where(ds_obs_6190 < p90_event_obs_9120, 1, 0).mean(dim=("time"), skipna = True)) * mask_spart
mean_exceeds_6190 = exceeds_6190.mean(dim=("y","x"), skipna=True)

print("10-year event 1961-1990;{0}".format(mean_exceeds_6190.values))
print("10-year event 1991-2020;{0}".format(mean_exceeds_9120.values))

for file, file_ref, gwl in zip(infiles_gwl, infile_ref, gwls):
    f_gwl = xr.open_dataset(file)
    f_ref = xr.open_dataset(file_ref)
    ds_gwl = f_gwl[varname]
    ds_ref= f_ref[varname+"_reference_period_1991_2020"]

    p90_event_ref = ds_ref.quantile(0.1, dim="time", skipna=True)

    exceeds_ref = xr.where(ds_ref < p90_event_ref, 1, 0).mean(dim=("time"), skipna = True) * mask_gwl
    exceeds_ref_mean = exceeds_ref.mean(dim=("y","x"), skipna=True)

    exceeds_gwl = xr.where(ds_gwl < p90_event_ref, 1, 0).mean(dim=("time"), skipna = True) * mask_gwl
    exceeds_gwl_mean = exceeds_gwl.mean(dim=("y","x"), skipna=True)    

    relation = exceeds_gwl_mean / exceeds_ref_mean
    modelmean_relation = relation.mean(dim="ens", skipna=True)
    
    print("10-year event at GWL{0} (Lower);{1}".format(gwl, relation.quantile(0.1, dim="ens", skipna=True).values))
    print("10-year event at GWL{0} (Median);{1}".format(gwl, relation.quantile(0.5, dim="ens", skipna=True).values))
    print("10-year event at GWL{0} (Upper);{1}".format(gwl, relation.quantile(0.9, dim="ens", skipna=True).values))
    print("10-year event at GWL{0} (Mean);{1}".format(gwl, modelmean_relation.values))