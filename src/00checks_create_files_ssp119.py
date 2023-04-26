#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""

import os
import glob
import numpy as np
import xarray as xr

infiles = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/tas_ssp119.txt", mode="rt")
files = [x.replace("\n","") for x in infiles]

temp_files = [x for x in files if "Amon" in x]
temp_files = [x.replace("ImranN/", "/hpx/ImranN/") for x in temp_files]
temp_files.sort()

unique_files = [x[:-16] for x in temp_files]
unique_files = np.array(unique_files)
unique_files = np.unique(unique_files)
unique_files = [x for x in unique_files]

for file in unique_files:
    f_in = sorted(glob.glob(file+"*.nc"))
    if len(f_in) > 1:
        f1 = xr.open_mfdataset(f_in, concat_dim="time", combine="nested", data_vars='minimal', coords='minimal', compat='override')
    elif len(f_in) == 1:
        print("Only single file found. Opening without concatting.")
        f1 = xr.open_dataset(f_in[0])
    else:
        print("file {0} not found, continuing with next file".format(f_in))
    if f1.time.dtype == "object":
        syr = str(f1.time[0].dt.year.values)
        eyr = str(f1.time[-1].dt.year.values)
        smn = str(f1.time[0].dt.month.values)
        emn = str(f1.time[-1].dt.month.values)
        if len(smn) < 2:
            smn = "0"+smn
        if len(emn) < 2:
            emn = "0"+emn
        stmn = syr + smn
        enmn = eyr + emn
    else:    
        stmn = np.datetime_as_string(f1.time[0].values)[:7].replace("-","")
        enmn = np.datetime_as_string(f1.time[-1].values)[:7].replace("-","")
    outf = "/hpx/Bennib/CMIP6_data_temp/Projection/" + file.split("/")[-1] + stmn + "-" + enmn + ".nc"
    if os.path.isfile(outf):
        print("file {0} already exists. Deleting...".format(outf))
        os.remove(outf)
    f1.to_netcdf(outf, unlimited_dims="time")

print("All ssp119 files successfully concatted and copied")