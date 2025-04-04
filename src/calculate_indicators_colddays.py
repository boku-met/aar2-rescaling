#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:04:42 2022

@author: benedikt.becsi<at>boku.ac.at
"""
import os
import glob
from multiprocessing.pool import Pool
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass


path_tn = "/nas/nas5/Projects/OEK15/tn_daily/" 
path_tx = "/nas/nas5/Projects/OEK15/tx_daily/" 
path_out = "/sto0/data/Results/Indicators/"
            
infiles_tx = sorted(glob.glob(path_tx+"tx_SDM_*.nc"))
infiles_tn = sorted(glob.glob(path_tn+"tn_SDM_*.nc"))
        
for file_tn in infiles_tn:
    mname = file_tn.replace(".nc","")
    modelname = "_".join(mname.split("/")[-1].split("_")[2:6])
    file_tx = [x for x in infiles_tx if modelname in x]

    ds_in_tn = xr.open_dataset(file_tn)
    ds_in_tx = xr.open_dataset(file_tx[0])
    for dvar in ds_in_tn.data_vars:
        if "lambert" in dvar:
            crsvar = dvar
    check_endyear = (ds_in_tn.time.dt.month == 12) & (ds_in_tn.time.dt.day == 30)
    time_fullyear = ds_in_tn.time[check_endyear]
    years = np.unique(time_fullyear.dt.year)
    ds_in_tn = ds_in_tn.sel(time=slice(str(min(years)), str(max(years))))
    ds_in_tx = ds_in_tx.sel(time=slice(str(min(years)), str(max(years))))

    years_tx = np.unique(ds_in_tx.time.dt.year)
    if years_tx.size != years.size:
        print('Faulty file detected. TN has time variable of {0} and '
        'TX has time variable of {1}. Skipping file {2}'.format(years.size,
        years_tx.size,file_tn))
        continue

    mask = xr.where(ds_in_tx.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                                skipna=True) 
                    >= -990, 1, np.nan).compute()
    print("*** Loading dataset {0} complete. Mask created.".format(file_tn))
    
    # Calculate indicator with parallel processing
    def parallel_loop(y):
        cur_tx = ds_in_tx.tasmax[ds_in_tx.time.dt.year == y].load()
        cur_tn = ds_in_tn.tasmin[ds_in_tn.time.dt.year == y].load()

        frostdays = xr.where(cur_tn <= 0, 1, 0).sum(dim="time", skipna=True)
        icedays =  xr.where(cur_tx <= 0, 1, 0).sum(dim="time", skipna=True)
        
        return frostdays, icedays

    if __name__ == '__main__':
    # create and configure the process pool
        with Pool(24) as pool:
        # execute tasks, block until all completed
            parallel_results = pool.map(parallel_loop, years)

    frostdays_year = xr.concat([x[0] for x in parallel_results], dim="time")
    icedays_year = xr.concat([x[1] for x in parallel_results], dim="time")

    frostdays_year = (frostdays_year * mask).compute()
    icedays_year = (icedays_year * mask).compute()

    print("--> Calculation of indicators for dataset {0} complete".format(file_tn))
            
    # Add CF-conformal metadata
    
    # Attributes for the indicator variables:
    attr_dict = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "number_of_days_with_air_temperature_below_threshold", 
                    "units": "1"}
    frostdays_year.attrs = attr_dict
    icedays_year.attrs = attr_dict
    
    frostdays_year.attrs.update({"cell_methods":"time: sum over days "
                    "(days with tmin below 0°C)",
                    "long_name": "Days with daily minimum temperature below 0°C" })
    icedays_year.attrs.update({"cell_methods":"time: sum over days "
                    "(days with tmax below 0°C)", 
                    "long_name": "Days with daily maximum temperature below 0°C"})
    

    time_resampled = ds_in_tn.time.resample(time="YE")
    start_inds = np.array([x.start for x in time_resampled.groups.values()])
    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
    end_inds[-1] = ds_in_tn.time.size
    end_inds -= 1
    start_inds = start_inds.astype(np.int32)
    end_inds = end_inds.astype(np.int32)
    
    frostdays_year.coords["time"] = ds_in_tn.time[end_inds]
    icedays_year.coords["time"] = ds_in_tn.time[end_inds]
                                            
    frostdays_year.time.attrs.update({"climatology":"climatology_bounds"})
    icedays_year.time.attrs.update({"climatology":"climatology_bounds"})
       
    # Encoding and compression
    encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
    
    frostdays_year.encoding = encoding_dict
    icedays_year.encoding = encoding_dict
                                
    # Climatology variable
    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
    climatology = xr.DataArray(np.stack((ds_in_tn.time[start_inds],
                                            ds_in_tn.time[end_inds]), 
                                        axis=1), 
                                coords={"time": frostdays_year.time, 
                                        "nv": np.arange(2, dtype=np.int16)},
                                dims = ["time","nv"], 
                                attrs=climatology_attrs)
        
    climatology.encoding.update({"dtype":np.float64,'units': ds_in_tn.time.encoding['units'],
                                    'calendar': ds_in_tn.time.encoding['calendar']})
    
    crs = xr.DataArray(np.nan, attrs=ds_in_tn[crsvar].attrs)
    
    file_attrs = {'title': 'Cold Days',
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': 'File containing two different cold indicators: Frost days (Tmin <= 0°C), Ice days (Tmax <= 0°C)',
        'Conventions': 'CF-1.8'}
    
    ds_out = xr.Dataset(data_vars={"frostdays": frostdays_year,
                                    "icedays": icedays_year,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_tn.lat,
                                    "lon": ds_in_tn.lon}, 
                        coords={"time":frostdays_year.time, "y": ds_in_tn.y,
                                "x":ds_in_tn.x},
                        attrs=file_attrs)
    
    if path_out.endswith("/"):
        None
    else:
        path_out += "/"
    outf = path_out + "Colddays_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf):
        print("File {0} already exists. Removing...".format(outf))
        os.remove(outf)
    
    # Write final file to disk
    ds_out.to_netcdf(outf, unlimited_dims="time")
    print("Writing file {0} completed!".format(outf))
    ds_in_tn.close()
    ds_in_tx.close()
    ds_out.close()
print("Successfully processed all input files!")