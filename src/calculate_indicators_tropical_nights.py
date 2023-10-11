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
    path_to_data = "/nas/nas5/Projects/OEK15/tn_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/hp8/Projekte_Benni/Temp_Data/Indicators"
             
    return path_to_data, output_path
        
def main():
    
    (path_in, path_out) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_pr = sorted(glob.glob(path_in+"tn_SDM_*.nc"))
          
    for file in infiles_pr:
        ds_in_pr = xr.open_dataset(file)
        for dvar in ds_in_pr.data_vars:
            if "lambert" in dvar:
                crsvar = dvar
        check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
        time_fullyear = ds_in_pr.time[check_endyear]
        years = np.unique(time_fullyear.dt.year)
        ds_in_pr = ds_in_pr.sel(time=slice(str(min(years)), str(max(years))))
        mask = xr.where(ds_in_pr.tasmin.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        
        # Calculate indicator with parallel processing
        def parallel_loop(y):
            cur_pr = ds_in_pr.tasmin[ds_in_pr.time.dt.year == y].load()
            cond_20 = xr.where(cur_pr >= 20, 1, 0).sum(dim="time", skipna=True)
            return cond_20

        parallel_results = Parallel(n_jobs=8, prefer="threads")(delayed(parallel_loop)(y) for y in years)
        tn_20 = xr.combine_nested(parallel_results, concat_dim="time", coords='minimal')

        tn_20 = (tn_20 * mask).compute()  
        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"coordinates": "time lat lon", 
                     "grid_mapping": "crs", 
                     "standard_name": "number_of_days_with_air_temperature_above_threshold", 
                        "units": "1"}
        tn_20.attrs = attr_dict
        
        tn_20.attrs.update({"cell_methods":"time: sum over days "
                     "(days exceeding tmin of 20°C)",
                      "long_name": "Days exceeding daily minimum temperature of 20°C" })
        
        time_resampled = ds_in_pr.time.resample(time="A")
        start_inds = np.array([x.start for x in time_resampled.groups.values()])
        end_inds = np.array([x.stop for x in time_resampled.groups.values()])
        end_inds[-1] = ds_in_pr.time.size
        end_inds -= 1
        start_inds = start_inds.astype(np.int32)
        end_inds = end_inds.astype(np.int32)
        
        tn_20.coords["time"] = ds_in_pr.time[end_inds]
        
        tn_20.time.attrs.update({"climatology":"climatology_bounds"})
                
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'complevel': 1, 'fletcher32': False, 
                         'contiguous': False}
        
        tn_20.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                                ds_in_pr.time[end_inds]), 
                                            axis=1), 
                                    coords={"time": tn_20.time, 
                                            "nv": np.arange(2, dtype=np.int16)},
                                    dims = ["time","nv"], 
                                    attrs=climatology_attrs)
            
        climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                     'calendar': ds_in_pr.time.encoding['calendar']})
        
        crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)
        mname = file.replace(".nc","")
        modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

        file_attrs = {'title': 'Tropical Nights',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'comment': 'Annual number of days with daily minimum temperature above 20 degC',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"tropical_nights_20": tn_20,
                                       "climatology_bounds": climatology, 
                                       "crs": crs,
                                       "lat": ds_in_pr.lat,
                                       "lon": ds_in_pr.lon}, 
                            coords={"time":tn_20.time, "y": ds_in_pr.y,
                                    "x":ds_in_pr.x},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + "Tropicalnights_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
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