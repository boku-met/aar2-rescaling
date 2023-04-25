#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
import xarray as xr

infiles_ssp = sorted(glob.glob("/hpx/Bennib/CMIP6_data_temp/Projection/*.nc"))
infiles_hist = sorted(glob.glob("/hpx/Bennib/CMIP6_data_temp/Historic/*.nc"))

outn_ssp = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/model_diagnostics_projections.csv"
outn_hist = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/model_diagnostics_historical.csv"

if os.path.isfile(outn_hist):
    print("file {0} already exists. Deleting.".format(outn_hist))
    os.remove(outn_hist)
if os.path.isfile(outn_ssp):
    print("file {0} already exists. Deleting.".format(outn_ssp))
    os.remove(outn_ssp)

outf_ssp = open(file=outn_ssp, mode="xt", encoding="utf-8", newline="\n")
outf_hist = open(file=outn_hist, mode="xt", encoding="utf-8", newline="\n")

outf_hist.write("Model name;Start date;End date;Unit;Dimensions;Grid;Resolution;Data type\n")
outf_ssp.write("Model name;Start date;End date;Unit;Dimensions;Grid;Resolution;Data type\n")

for f in infiles_hist:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
        modelname = f.split("/")[-1].replace(".nc","")
        edate = f1.time[-1].values
        sdate = f1.time[0].values
        outf_hist.write("{0};{1};{2};{3};{4};{5};{6};{7}\n".format(modelname, sdate, edate, f1.tas.attrs["units"], f1.dims, f1.attrs["grid"], f1.attrs["nominal_resolution"], f1.tas.encoding["dtype"]))
    except(Exception):
        print("Some errors occured in file "+f)

for f in infiles_ssp:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
        modelname = f.split("/")[-1].replace(".nc","")
        edate = f1.time[-1].values
        sdate = f1.time[0].values
        outf_ssp.write("{0};{1};{2};{3};{4};{5};{6};{7}\n".format(modelname, sdate, edate, f1.tas.attrs["units"], f1.dims, f1.attrs["grid"], f1.attrs["nominal_resolution"], f1.tas.encoding["dtype"]))
    except(Exception):
        print("Some errors occured in file "+f)

outf_hist.close()
outf_ssp.close()

print("Opening and checking CMIP6 data completed.")