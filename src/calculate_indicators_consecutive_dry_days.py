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
import helper_functions as hf

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires precipitation data.
    path_to_data = "/nas/nas5/Projects/OEK15/pr_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/sto1/home/bennib/Bennib/HotSpotKlim/HotSpotKlim_Salzburg/Data/Indicators"
             
    return path_to_data, output_path
    
(path_in, path_out) = user_data()

if path_in.endswith("/"):
    None
else:
    path_in += "/"
infiles_pr = sorted(glob.glob(path_in+"pr_SDM_*.nc"))
        
for file in infiles_pr:
    ds_in_pr = xr.open_dataset(file)
    for dvar in ds_in_pr.data_vars:
        if "lambert" in dvar:
            crsvar = dvar
    check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
    time_fullyear = ds_in_pr.time[check_endyear]
    years = np.unique(time_fullyear.dt.year)
    ds_in_pr = ds_in_pr.sel(time=slice(str(min(years)), str(max(years))))
    mask = xr.where(ds_in_pr.pr.isel(time=slice(0,60)).mean(dim="time", 
                                                                skipna=True) 
                    >= -990, 1, np.nan).compute()
    print("*** Loading dataset {0} complete. Mask created.".format(file))
    
    # Calculate indicator with parallel processing
    def parallel_loop(y):
        cur_pr = ds_in_pr.pr.sel(time=slice(str(y)+"-04",str(y)+"-09")).load() #select all days from April-September
        (loesch, temp_ccd_maxlen, temp_ccd_nrofdays) = hf.consecutive_3d(cur_pr, 1.0, 5, lt=True)
        return temp_ccd_maxlen, temp_ccd_nrofdays
    if __name__ == '__main__':
            # create and configure the process pool
            with Pool(12) as pool:
                # execute tasks, block until all completed
                parallel_results = pool.map(parallel_loop, years)
            # process pool is closed automatically
    
    ccd_maxlen = xr.DataArray([x[0] for x in parallel_results], coords={"time": years, "y": ds_in_pr.y, "x": ds_in_pr.x})
    ccd_nrofdays = xr.DataArray([x[1] for x in parallel_results], coords={"time": years, "y": ds_in_pr.y, "x": ds_in_pr.x})

    ccd_maxlen = (ccd_maxlen * mask).compute()  
    ccd_nrofdays = (ccd_nrofdays * mask).compute()


    print("--> Calculation of indicators for dataset {0} complete".format(file))
            
    # Add CF-conformal metadata
    
    # Attributes for the indicator variables:
    attr_dict = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "number_of_days_with_precipitation_above_thresholds", 
                    "units": "1"}
    ccd_maxlen.attrs = attr_dict
    ccd_nrofdays.attrs = attr_dict

    ccd_maxlen.attrs.update({"cell_methods":"time: sum within days "
                    "(consecutive days with less than 1 mm of precipitation)",
                    "long_name": "maximum length of periods with 5 or more consecutive "
        "dry days (<1mm daily precipitation) during the summer half year "
        "(April-September)" })
    ccd_nrofdays.attrs.update({"cell_methods":"time: sum within days "
                    "(consecutive days with less than 1 mm of precipitation)", 
                    "long_name": "number of days within periods with "
        "5 or more consecutive dry days (<1mm daily precipitation) during "
        "the summer half year (April-September)"})

    time_resampled = ds_in_pr.time.resample(time="A")
    start_inds = np.array([x.start for x in time_resampled.groups.values()])
    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
    end_inds[-1] = ds_in_pr.time.size
    end_inds -= 1
    start_inds = start_inds.astype(np.int32)
    end_inds = end_inds.astype(np.int32)
    
    ccd_maxlen.coords["time"] = ds_in_pr.time[end_inds]
    ccd_nrofdays.coords["time"] = ds_in_pr.time[end_inds]
                                            
    ccd_maxlen.time.attrs.update({"climatology":"climatology_bounds"})
    ccd_nrofdays.time.attrs.update({"climatology":"climatology_bounds"})
    
    # Encoding and compression
    encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
    
    ccd_maxlen.encoding = encoding_dict
    ccd_nrofdays.encoding = encoding_dict
                            
    # Climatology variable
    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
    climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                            ds_in_pr.time[end_inds]), 
                                        axis=1), 
                                coords={"time": ccd_maxlen.time, 
                                        "nv": np.arange(2, dtype=np.int16)},
                                dims = ["time","nv"], 
                                attrs=climatology_attrs)
        
    climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                    'calendar': ds_in_pr.time.encoding['calendar']})
    
    crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)
    mname = file.replace(".nc","")
    modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

    file_attrs = {'title': '<Consecutive Dry Days>: Indicators calculated from bias-corrected model data',
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': "The file contains the maximum length of and number of days "
                    "in consecutive periods that are 5 days or longer during April-September",
        'Conventions': 'CF-1.8'}
    
    ds_out = xr.Dataset(data_vars={"consecutive_dry_days_maxlength": ccd_maxlen,
                                    "consecutive_dry_days_noofdays": ccd_nrofdays,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_pr.lat,
                                    "lon": ds_in_pr.lon}, 
                        coords={"time":ccd_maxlen.time, "y": ds_in_pr.y,
                                "x":ds_in_pr.x},
                        attrs=file_attrs)
    
    if path_out.endswith("/"):
        None
    else:
        path_out += "/"
    outf = path_out + "consecutive_dry_days_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf):
        print("File {0} already exists. Removing...".format(outf))
        os.remove(outf)
    
    # Write final file to disk
    ds_out.to_netcdf(outf, unlimited_dims="time")
    print("Writing file {0} completed!".format(outf))
    ds_in_pr.close()
    ds_out.close()
print("Successfully processed all input files!")