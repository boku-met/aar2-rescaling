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

infiles_rcp = sorted(glob.glob("/hpx/Bennib/CMIP5_data_temp/OEKS15_models/*.nc"))
infiles_hist = sorted(glob.glob("/hpx/Bennib/CMIP5_data_temp/OEKS15_historical/*.nc"))

outn_rcp = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/model_diagnostics/CMIP5_model_diagnostics_projections.csv"
outn_hist = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/model_diagnostics/CMIP5_model_diagnostics_historical.csv"

if os.path.isfile(outn_hist):
    print("file {0} already exists. Deleting.".format(outn_hist))
    os.remove(outn_hist)
if os.path.isfile(outn_rcp):
    print("file {0} already exists. Deleting.".format(outn_rcp))
    os.remove(outn_rcp)

outf_rcp = open(file=outn_rcp, mode="xt", encoding="utf-8", newline="\n")
outf_hist = open(file=outn_hist, mode="xt", encoding="utf-8", newline="\n")

outf_hist.write("Model name   Start date   End date   Unit   Dimensions   Grid   Resolution   Data type\n")
outf_rcp.write("Model name   Start date   End date   Unit   Dimensions   Grid   Resolution   Data type\n")

for f in infiles_hist:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
    except Exception:
        print("Some errors occured in file "+f)
        continue

    modelname = f.split("/")[-1].replace(".nc","")

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

    try:
        outf_hist.write("{0}   {1}   {2}   {3}   {4}   {5}   {6}   {7}\n".format(modelname, stmn, enmn, f1.tas.attrs["units"], f1.dims, f1.attrs["grid"], f1.attrs["nominal_resolution"], f1.tas.encoding["dtype"]))
    except KeyError:
        print("Some attributes missing in file "+f)
        outf_hist.write("{0}   {1}   {2}   {3}   {4}   {5}   {6}   {7}\n".format(modelname, stmn, enmn, f1.tas.attrs["units"], f1.dims, "X", "X", f1.tas.encoding["dtype"]))
        

for f in infiles_rcp:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
    except Exception:
        print("Some errors occured in file "+f)
        continue

    modelname = f.split("/")[-1].replace(".nc","")

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

    try:
        outf_rcp.write("{0}   {1}   {2}   {3}   {4}   {5}   {6}   {7}\n".format(modelname, stmn, enmn, f1.tas.attrs["units"], f1.dims, f1.attrs["grid"], f1.attrs["nominal_resolution"], f1.tas.encoding["dtype"]))
    except KeyError:
        print("Some attributes missing in file "+f)
        outf_rcp.write("{0}   {1}   {2}   {3}   {4}   {5}   {6}   {7}\n".format(modelname, stmn, enmn, f1.tas.attrs["units"], f1.dims, "X", "X", f1.tas.encoding["dtype"]))

outf_hist.close()
outf_rcp.close()

print("Opening and checking CMIP5 data completed.")