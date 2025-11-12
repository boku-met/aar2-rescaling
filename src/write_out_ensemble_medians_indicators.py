#!/hpx/Bennib/miniconda3/envs/Pytools/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on Thu March 6th, 2025

@author: bennib
"""
import os
import glob
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

infiles = sorted(glob.glob("/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/*.nc"))
infiles = [x for x  in infiles if not "DJF" in x and not "JJA" in  x and not "MAM" in x and not "SON" in x]
infiles = [x for x in infiles if not "SPEI" in x]
infiles = [x for x in infiles if not "gmuend" in x]
infiles = [x for x in infiles if not "last_day" in x]

path_out = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/Ensemble_medians/"

for file in infiles:
    ds_in = xr.open_dataset(file)

    climatology = ds_in.mean(dim="time", skipna=True)
    ensemble_median = climatology.median(dim="ens", skipna=True)
    ensemble_median = ensemble_median.drop_vars(("lat","lon"))

    encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
    ensemble_median.encoding.update(encoding_dict)
    ensemble_median.attrs = ds_in.attrs
    ensemble_median.attrs["title"] = ensemble_median.attrs["title"].replace("Ensemble", "Ensemble median")

    filename = file.split("/")[-1].replace(".nc","_ensemble_median.nc")
    outf = path_out + filename
    if os.path.isfile(outf):
        print("file {} already exists. Overwriting.".format(outf))
        os.remove(outf)
    ensemble_median.to_netcdf(outf)
    print ("writing ensemble median for file {0} complete!".format(file))