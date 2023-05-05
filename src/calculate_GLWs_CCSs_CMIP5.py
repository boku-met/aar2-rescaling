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

#latmin=46.5
#latmax=49
#lonmin=9.5
#lonmax=17

# user specified paths and data
gwls = [1.5, 2.0, 3.0, 4.0]
path_cmip5_models = "/hpx/Bennib/CMIP5_data_temp/OEKS15_models/"
path_cmip5_hist = "/hpx/Bennib/CMIP5_data_temp/OEKS15_historical/"
path_oeks15 = "/nas5/Projects/OEK15/tas_daily/"
outf = "/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP5.csv"
if os.path.isfile(outf):
    print("File {0} already exists. Overwriting...".format(outf))
    os.remove(outf)
outfile = open(outf, mode="xt", encoding="utf-8", newline="\n")
outfile.write("Linked models;;Mean year per GWL;;;;Period per GWL\n")
outfile.write("GCM (CMIP5);OEKS15 ensemble member;1.5°C;2.0°C;3.0°C;4.0°C;1.5°C;2.0°C;3.0°C;4.0°C\n")

for rcp in  ["rcp26", "rcp45", "rcp85"]:
    # create filelist for each rcp
    infiles = sorted(glob.glob(path_cmip5_models+"tas_*"+rcp+"*.nc"))
    for file in infiles:
        # search for associated historical/oeks15 files
        search_term = file.split("/")[-1][:-17]
        search_hist = search_term.replace(rcp, "historical")
        search_oeks15 = search_term.replace("tas_Amon_", "")
        file_oeks15 = sorted(glob.glob(path_oeks15+"*"+search_oeks15+"*.nc"))
        file_hist = glob.glob(path_cmip5_hist+search_hist+"*.nc")
        assert(len(file_hist) == 1)
        # open files and determine time period
        f1_proj = xr.open_dataset(file)
        f1_hist = xr.open_dataset(file_hist[0])
        min_yr = f1_hist.time[0].dt.year.values
        max_yr = f1_proj.time[-1].dt.year.values
        # concat files for complete timeseries
        f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2005")), f1_proj.sel(time=slice("2006", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')
        # calculate global annual mean temperature timeseries
        weights = np.cos(np.deg2rad(f1.lat))
        tas_weighted = f1.tas.weighted(weights)
        series_global = tas_weighted.mean(dim=('lat', 'lon'), skipna=True).compute()
        series_global = series_global.resample(time="A", skipna = True).mean()
        # calculate anomalies and smooth timeseries
        ref_gmt = series_global.sel(time=slice(str(min_yr),"1900")).mean(skipna=True)
        anomalies = series_global - ref_gmt
        anomalies_smooth = anomalies.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
        # create data to write to list
        gwl_list = []
        mean_years = []
        for gwl in gwls:
            try:
                timeind = (anomalies_smooth.values >= gwl).nonzero()[0][0]
                mean_year = anomalies_smooth[timeind].time.dt.year.values
                period = "{0}-{1}".format(mean_year-9, mean_year+10)
                mean_years.append(str(mean_year))
                gwl_list.append(period)
            except IndexError:
                mean_years.append("n/a")
                gwl_list.append("n/a")
        # write data to file
        for f in file_oeks15:
            modelname = f.split("/")[-1].replace("tas_","").replace(".nc","")
            outfile.write("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}\n".format(search_oeks15, modelname, *mean_years, *gwl_list))

outfile.close()
print("Writing file {0} complete!".format(outf))