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

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def check_isfile(fname):
    if os.path.isfile(fname):
        print("File {0} already exists. Overwriting...".format(fname))
        os.remove(fname)
    file_out = open(fname, mode="xt", encoding="utf-8", newline="\n")
    return file_out

latmin=46.5
latmax=49
lonmin=9.5
lonmax=17

# user specified paths and data
gwls = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
path_cmip6_models = "/hpx/Bennib/CMIP6_data_temp/Projection/"
path_cmip6_hist = "/hpx/Bennib/CMIP6_data_temp/Historic/"
outf = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP6_all_models.csv"

# create summary csv file for CMIP5 OEKS15 GWLs
outfile = check_isfile(outf)
outfile.write("Model;Mean year per GWL;;;;;;Period per GWL;;;;;;AUT GCM CCS 1991-2020\n")
outfile.write("GCM (CMIP6);1.0°C;1.5°C;2.0°C;3.0°C;4.0°C;5.0°C;1.0°C;1.5°C;2.0°C;3.0°C;4.0°C;5.0°C;1.0°C;1.5°C;2.0°C;3.0°C;4.0°C;5.0°C\n")
    
for rcp in  ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]:
    # create filelist for each ssp
    infiles = sorted(glob.glob(path_cmip6_models+"tas_*"+rcp+"*.nc"))
    for file in infiles:
        # search for associated historical/oeks15 files
        search_term = file.split("/")[-1][:-17]
        search_hist = search_term.replace(rcp, "historical")
        file_hist = glob.glob(path_cmip6_hist+search_hist+"*.nc")
        assert(len(file_hist) == 1)
        # open files and determine time period
        f1_proj = xr.open_dataset(file)
        f1_hist = xr.open_dataset(file_hist[0])
        min_yr = f1_hist.time[0].dt.year.values
        max_yr = f1_proj.time[-1].dt.year.values
        # concat files for complete timeseries
        try:
            f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2014")), f1_proj.sel(time=slice("2015", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')
        except Exception:
            if "time_bnds" in f1_hist.data_vars:
                f1_hist = f1_hist.drop_vars("time_bnds")
            if "time_bnds" in f1_proj.data_vars:
                f1_proj = f1_proj.drop_vars("time_bnds")
            f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2014")), f1_proj.sel(time=slice("2015", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')
            print("Dropped time_bnds variable from datasets, concatting successful!")
        # calculate global annual mean temperature timeseries
        weights = np.cos(np.deg2rad(f1.lat))
        tas_weighted = f1.tas.weighted(weights)
        series_global = tas_weighted.mean(dim=('lat', 'lon'), skipna=True).compute()
        series_global = series_global.resample(time="A", skipna = True).mean()
        # calculate AUT annual mean temperature timeseries
        tas_aut = f1.tas.sel(lat = slice(latmin, latmax), lon = slice(lonmin, lonmax))
        weights_aut = weights.sel(lat = slice(latmin, latmax))
        tas_weighted_aut = tas_aut.weighted(weights_aut)
        series_aut = tas_weighted_aut.mean(dim=("lat","lon"), skipna=True).compute()
        series_aut = series_aut.resample(time="A", skipna=True).mean()
        # calculate anomalies and smooth timeseries
        ref_gmt = series_global.sel(time=slice(str(min_yr),"1900")).mean(skipna=True)
        ref_amt = series_aut.sel(time = slice("1991","2020")).mean(skipna = True)
        anomalies = series_global - ref_gmt
        anomalies_aut = series_aut - ref_amt
        anomalies_smooth = anomalies.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
        anomalies_aut_smooth = anomalies_aut.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
        # create data to write to list
        gwl_list = []
        mean_years = []
        ccs_aut_gcm = []
        for gwl in gwls:
            try:
                timeind = (anomalies_smooth.values >= gwl).nonzero()[0][0]
                mean_year = anomalies_smooth[timeind].time.dt.year.values
                period = "{0}-{1}".format(mean_year-10, mean_year+9)
                ccs_aut_gcm1 = anomalies_aut_smooth[timeind].values
                # add data to lists
                mean_years.append(str(mean_year))
                gwl_list.append(period)
                ccs_aut_gcm.append(str(ccs_aut_gcm1))
            except IndexError:
                mean_years.append("n/a")
                gwl_list.append("n/a")
                ccs_aut_gcm.append("n/a")
        
        # write data to files
        modelname = file.split("/")[-1].replace("tas_Amon_","").replace(".nc","")
        outfile.write("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15};{16};{17};{18}\n".format(modelname, *mean_years, *gwl_list, *ccs_aut_gcm))
        print("Writing data for model {0} finished!".format(file))
outfile.close()
print("Writing file {0} complete!".format(outf))