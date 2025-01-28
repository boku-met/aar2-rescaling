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

infiles_gwl_ind = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"

gwls = [1.5, 2.0, 3.0, 4.0]
indicator_searchterm = "tas_CMIP5"
varname = "tas"
infiles_gwls = sorted(glob.glob(infiles_gwl_ind+indicator_searchterm+"*.nc"))

f1 = xr.open_dataset(infiles_gwls[1])

mask = xr.where(f1.tas[2,:,:,:].mean(dim="time", skipna=True) >= -999, 1, np.nan)

for f in infiles_gwls[1:]:
    f1 = xr.open_dataset(f)  
    f_ref = xr.open_dataset(infiles_gwls[0])  

    area_refperiod = (f_ref[varname] * mask).mean(dim=("y","x"), skipna=True)
    refperiod = area_refperiod.mean(dim="time", skipna=True).compute()
    area_sample = (f1[varname] * mask).mean(dim=("y","x"), skipna=True).compute()

    anomalies = area_sample - refperiod
    print("### GWL: {0} #####".format(f))
    mdnm = ""
    for en in anomalies.ens:
        mdnm += "{0};".format(en.values)
    print(mdnm)
    for i in range(anomalies.time.size):
        mdln = ""
        for en in anomalies.ens:
            mdln += "{0};".format(anomalies.sel(ens=en)[i].values)
        print(mdln)