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

infiles = sorted(glob.glob("/sto0/data/Results/Indicators/*SPARTAC*"))
infiles = [x for x in infiles if not "DJF" in x and not "JJA" in x and not "SON" in x and not "MAM" in x]
infiles = [x for x in infiles if not "monthly" in x]
infiles = [x for x in infiles if not "waterbalance" in x]
infiles = [x for x in infiles if not "gmuend" in x and not "huglin" in x]

if2 = sorted(glob.glob("/sto1/home/bennib/Bennib/Fluvial_processes/data/indicators/*aggre*q10*spart*"))
for x in if2:
    infiles.append(x)
    
if3 = sorted(glob.glob("/sto0/data/Results/Indicators/*FORSIT*"))
for x in if3[0:2]:
    infiles.append(x)

path_out = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/OBS/Climatologies/"

periods = [1961, 1991, "GWL10"]

for file in infiles:
    ds_in = xr.open_dataset(file)
    for sy in periods:
        flag = None
        try:
            ey = sy + 29
        except(TypeError):
            sy = 2001
            ey = 2020
            flag = 1
        
        climatology = ds_in.sel(time=slice(str(sy), str(ey))).mean(dim="time", skipna=True)
        climatology = climatology.drop_vars(("lat","lon"))

        encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                            'complevel': 1, 'fletcher32': False, 
                            'contiguous': False}
        climatology.encoding.update(encoding_dict)
        climatology.attrs = ds_in.attrs

        filename = file.split("/")[-1]
        filename = "_".join(filename.split("_")[:-2])
        if flag:
            filename += "_climatology_GWL10.nc".format(sy, ey)
        else:
            filename += "_climatology_{0}-{1}.nc".format(sy, ey)
        outf = path_out + filename

        if os.path.isfile(outf):
            print("file {} already exists. Overwriting.".format(outf))
            os.remove(outf)
        climatology.to_netcdf(outf)
        print ("writing climatology {0} complete!".format(file))
print("finished processing all files!")




# Special case monthly indicator:

infiles = sorted(glob.glob("/sto0/data/Results/Indicators/HeavyPrecipitationDays_SPARTACUS_month*"))

for file in infiles:
    ds_in = xr.open_dataset(file)
    mask = xr.where(ds_in.very_heavy_precipitation_days_20mm[0:29,:,:].mean(dim="time", skipna=True) >= -999, 1, np.nan).astype(np.float32)
    ds_in = ds_in.resample(time="YE", skipna=True).sum()
    for sy in periods:
        flag = None
        try:
            ey = sy + 29
        except(TypeError):
            sy = 2001
            ey = 2020
            flag = 1
        
        climatology = ds_in.sel(time=slice(str(sy), str(ey))).mean(dim="time", skipna=True)
        encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                            'complevel': 1, 'fletcher32': False, 
                            'contiguous': False}

        climatology["heavy_precipitation_days_10mm"] = climatology.heavy_precipitation_days_10mm * mask
        climatology["very_heavy_precipitation_days_20mm"] = climatology.very_heavy_precipitation_days_20mm * mask
        climatology.heavy_precipitation_days_10mm.encoding.update(encoding_dict)
        climatology.very_heavy_precipitation_days_20mm.encoding.update(encoding_dict)
        climatology = climatology.drop_vars(("lat","lon"))
        climatology.encoding.update(encoding_dict)
        climatology.attrs = ds_in.attrs
        climatology.attrs["comment"] = climatology.attrs["comment"].replace("Monthly", "Annual")

        filename = file.split("/")[-1]
        filename = "_".join(filename.split("_")[:-2])
        if flag:
            filename += "_climatology_GWL10.nc".format(sy, ey)
        else:
            filename += "_climatology_{0}-{1}.nc".format(sy, ey)
        outf = path_out + filename

        if os.path.isfile(outf):
            print("file {} already exists. Overwriting.".format(outf))
            os.remove(outf)
        climatology.to_netcdf(outf)
        print ("writing climatology {0} complete!".format(file))
print("finished processing all files!")