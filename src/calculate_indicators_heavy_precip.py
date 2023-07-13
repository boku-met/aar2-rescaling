#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:04:42 2022

@author: benedikt.becsi<at>boku.ac.at
"""
import os
import glob
from joblib import Parallel, delayed
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires precipitation data.
    path_to_data = "/nas/nas5/Projects/OEK15/pr_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/hp8/Projekte_Benni/Temp_Data/Indicators"
             
    return path_to_data, output_path
        
def main():
    
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
            cur_pr = ds_in_pr.pr[ds_in_pr.time.dt.year == y].load()
            cond_10mm = xr.where(cur_pr >= 10, 1, 0).resample(time="M", skipna=True).sum()
            cond_20mm = xr.where(cur_pr >= 20, 1, 0).resample(time="M", skipna=True).sum()
            return cond_10mm, cond_20mm

        parallel_results = Parallel(n_jobs=8, prefer="threads")(delayed(parallel_loop)(y) for y in years)
        pr_10mm = [x[0] for x in parallel_results]
        pr_20mm = [x[1] for x in parallel_results]
        pr_10mm = xr.combine_nested(pr_10mm, concat_dim="time", coords='minimal')
        pr_20mm = xr.combine_nested(pr_20mm, concat_dim="time", coords='minimal')

        pr_10mm = (pr_10mm * mask).compute()  
        pr_20mm = (pr_20mm * mask).compute()


        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"coordinates": "time lat lon", 
                     "grid_mapping": "crs", 
                     "standard_name": "number_of_days_with_precipitation_above_thresholds", 
                        "units": "1"}
        pr_10mm.attrs = attr_dict
        pr_20mm.attrs = attr_dict

        pr_10mm.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                     "(days exceeding 10 mm of precipitation)",
                      "long_name": "Days exceeding 10 mm of daily precipitation" })
        pr_20mm.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                     "(days exceeding 20mm of precipitation)", 
                     "long_name": "Days exceeding 20 mm of daily precipitation"})

        time_resampled = ds_in_pr.time.resample(time="M")
        start_inds = np.array([x.start for x in time_resampled.groups.values()])
        end_inds = np.array([x.stop for x in time_resampled.groups.values()])
        end_inds[-1] = ds_in_pr.time.size
        end_inds -= 1
        start_inds = start_inds.astype(np.int32)
        end_inds = end_inds.astype(np.int32)
        
        pr_10mm.coords["time"] = ds_in_pr.time[end_inds]
        pr_20mm.coords["time"] = ds_in_pr.time[end_inds]
                                                
        pr_10mm.time.attrs.update({"climatology":"climatology_bounds"})
        pr_20mm.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'complevel': 1, 'fletcher32': False, 
                         'contiguous': False}
        
        pr_10mm.encoding = encoding_dict
        pr_20mm.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                                ds_in_pr.time[end_inds]), 
                                            axis=1), 
                                    coords={"time": pr_10mm.time, 
                                            "nv": np.arange(2, dtype=np.int16)},
                                    dims = ["time","nv"], 
                                    attrs=climatology_attrs)
            
        climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                     'calendar': ds_in_pr.time.encoding['calendar']})
        
        crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)
        mname = file.replace(".nc","")
        modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

        file_attrs = {'title': 'Heavy Precipitation Days',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'comment': 'Monthly sum of days exceeding 10 mm or 20 mm of daily precipitation sums',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"heavy_precipitation_days_10mm": pr_10mm,
                                       "very_heavy_precipitation_days_20mm": pr_20mm,
                                       "climatology_bounds": climatology, 
                                       "crs": crs,
                                       "lat": ds_in_pr.lat,
                                       "lon": ds_in_pr.lon}, 
                            coords={"time":pr_10mm.time, "y": ds_in_pr.y,
                                    "x":ds_in_pr.x},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + "HeavyPrecipitationDays_" + modelname + "_monthly_{0}-{1}.nc".format(min(years), max(years))
        if os.path.isfile(outf):
            print("File {0} already exists. Removing...".format(outf))
            os.remove(outf)
        
        # Write final file to disk
        ds_out.to_netcdf(outf, unlimited_dims="time")
        print("Writing file {0} completed!".format(outf))
        ds_in_pr.close()
        ds_out.close()
    print("Successfully processed all input files!")
main()