#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:04:42 2022

@author: benedikt.becsi<at>boku.ac.at
Careful, here the winter season is not consecutive. It does not matter much, 
but for more detail winter should be defined correctly.
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

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires precipitation data.
    path_to_data = "/nas/nas5/Projects/OEK15/pr_daily/" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/sto0/data/Results/Indicators/"
             
    return path_to_data, output_path
         
(path_in, path_out) = user_data()

if path_in.endswith("/"):
    None
else:
    path_in += "/"
infiles_pr = sorted(glob.glob(path_in+"pr_*.nc"))
        
for file_pr in infiles_pr:
    ds_in_pr = xr.open_dataset(file_pr)
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
    print("*** Loading dataset {0} complete. Mask created.".format(file_pr))
    
    # Calculate indicator with parallel processing
    min_year = min(years)
    max_year = max(years)
    if max_year - min_year < 29:
        print("Dataset {0} needs at least 30 years of data. Exiting.".format(file_pr))
        exit()
    startyrs = np.arange(min_year, max_year-28)
    sns = np.unique(ds_in_pr.time.dt.season.values)
    for sn in sns: 
        def parallel_loop(sy):
            curind = (ds_in_pr.time.dt.year >= sy) & (ds_in_pr.time.dt.year <= sy+29) & (ds_in_pr.time.dt.season == sn)
            cur_pr = ds_in_pr.pr[curind,:,:].load()
            ex_pr = cur_pr.quantile(0.999, method = "linear", dim="time", skipna=True)
            start_times = cur_pr.time[0]
            end_times = cur_pr.time[-1]
            return ex_pr, start_times, end_times
        if __name__ == '__main__':
            # create and configure the process pool
            with Pool(56) as pool:
                # execute tasks, block until all completed
                parallel_results = pool.map(parallel_loop, startyrs)
            # process pool is closed automatically

        pr_extreme = xr.combine_nested([x[0] for x in parallel_results], concat_dim="time", coords='minimal')
        timestart = xr.combine_nested([x[1] for x in parallel_results], concat_dim="time", coords='minimal')
        timeend = xr.combine_nested([x[2] for x in parallel_results], concat_dim="time", coords='minimal')

        pr_extreme = (pr_extreme * mask).compute()

        print("--> Calculation of indicators for dataset {0} complete".format(file_pr))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"coordinates": "time y x", 
                    "grid_mapping": "crs", 
                    "standard_name": "precipitation_amount", 
                        "units": "kg m-2"}
        pr_extreme.attrs = attr_dict

        pr_extreme.attrs.update({"cell_methods":"time: quantile over years"
                    "(99.9 percentile of seasonal precipitation sums over 30 year periods)",
                    "long_name": "99.9 percentile of daily precipitation sums over 30 year perdiods within season {0}".format(sn)})
    
        pr_extreme.coords["time"] = timeend
                                                
        pr_extreme.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
        
        pr_extreme.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        climatology = xr.DataArray(np.stack((timestart,timeend), axis=1), 
                                    coords={"time": pr_extreme.time, 
                                            "nv": np.arange(2, dtype=np.int16)},
                                    dims = ["time","nv"], 
                                    attrs=climatology_attrs)
            
        climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                    'calendar': ds_in_pr.time.encoding['calendar']})
        
        crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)
        mname_pr = file_pr.replace(".nc","")
        modelname_pr = "_".join(mname_pr.split("/")[-1].split("_")[2:6])

        file_attrs = {'title': 'Extreme precipitation for season {0}'.format(sn),
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname_pr,
        'comment': '99.9 percentile of daily precipitation sums over 30-year periods within season {0}'.format(sn),
        'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"extreme_precipitation": pr_extreme,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_pr.lat,
                                    "lon": ds_in_pr.lon}, 
                            coords={"time":pr_extreme.time, "y": ds_in_pr.y,
                                    "x":ds_in_pr.x},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + "extreme_precipitation_" + modelname_pr + "_seasonal_{0}_{1}-{2}.nc".format(sn, min(years), max(years))
        if os.path.isfile(outf):
            print("File {0} already exists. Removing...".format(outf))
            os.remove(outf)
        
        # Write final file to disk
        ds_out.to_netcdf(outf, unlimited_dims="time")
        print("Writing file {0} completed!".format(outf))
        ds_in_pr.close()
        ds_out.close()
print("Successfully processed all input files!")
