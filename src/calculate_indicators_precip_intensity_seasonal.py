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
    path_to_data = "/hp8/Projekte_Benni/Temp_Data/Indicators" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/hp8/Projekte_Benni/Temp_Data/Indicators"
             
    return path_to_data, output_path
        
def main():
    
    (path_in, path_out) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_pr = sorted(glob.glob(path_in+"pr_*monthly*.nc"))
    infiles_wd = sorted(glob.glob(path_in+"Wetdays_*monthly*.nc"))
          
    for file_pr in infiles_pr:
        mname_pr = file_pr.replace(".nc","")
        modelname_pr = "_".join(mname_pr.split("/")[-1].split("_")[1:5])
        for file_wd in infiles_wd:
            mname_wd = file_wd.replace(".nc","")
            modelname_wd = "_".join(mname_wd.split("/")[-1].split("_")[1:5])
            if modelname_pr == modelname_wd:
                ds_in_pr = xr.open_dataset(file_pr)
                ds_in_wd = xr.open_dataset(file_wd)
                years = np.unique(ds_in_pr.time.dt.year)
                mask = xr.where(ds_in_pr.pr.isel(time=slice(0,60)).mean(dim="time", 
                                                                        skipna=True) 
                                >= -990, 1, np.nan).compute()
                print("*** Loading dataset {0}, {1} complete. Mask created.".format(file_pr, file_wd))
                
                # Calculate indicator with parallel processing
                assert(len(ds_in_pr.time) == len(ds_in_wd.time))
                sns = np.unique(ds_in_pr.time.dt.season.values)
                for sn in sns: 
                    def parallel_loop(y):
                        curind = (ds_in_pr.time.dt.year == y) & (ds_in_pr.time.dt.season == sn)
                        cur_pr = ds_in_pr.pr[curind,:,:].load()
                        cur_wd = ds_in_wd.wet_days_1mm[curind,:,:].load()
                        start_times = cur_pr.time[0]
                        end_times = cur_pr.time[-1]
                        pr_sum = cur_pr.sum(dim="time", skipna = True)
                        wd_sum = cur_wd.sum(dim="time", skipna = True)
                        pr_intensity = pr_sum / wd_sum
                        return pr_intensity, start_times, end_times

                    parallel_results = Parallel(n_jobs=8, prefer="threads")(delayed(parallel_loop)(y) for y in years)
                    pr_annual = xr.combine_nested([x[0] for x in parallel_results], concat_dim="time", coords='minimal')
                    timestart = xr.combine_nested([x[1] for x in parallel_results], concat_dim="time", coords='minimal')
                    timeend = xr.combine_nested([x[2] for x in parallel_results], concat_dim="time", coords='minimal')

                    pr_annual = (pr_annual * mask).compute()

                    print("--> Calculation of indicators for dataset {0} complete".format(modelname_pr))
                            
                    # Add CF-conformal metadata
                    
                    # Attributes for the indicator variables:
                    attr_dict = {"coordinates": "time y x", 
                                "grid_mapping": "crs", 
                                "standard_name": "precipitation_amount", 
                                    "units": "kg m-2"}
                    pr_annual.attrs = attr_dict

                    pr_annual.attrs.update({"cell_methods":"time: sum over months"
                                "(sum of monthly precipitation sums)",
                                "long_name": "Mean seasonal precipitation intensity for {0}".format(sn)})
                
                    pr_annual.coords["time"] = timeend
                                                            
                    pr_annual.time.attrs.update({"climatology":"climatology_bounds"})
                    
                    # Encoding and compression
                    encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                                    'complevel': 1, 'fletcher32': False, 
                                    'contiguous': False}
                    
                    pr_annual.encoding = encoding_dict
                                            
                    # Climatology variable
                    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
                    climatology = xr.DataArray(np.stack((timestart,timeend), axis=1), 
                                                coords={"time": pr_annual.time, 
                                                        "nv": np.arange(2, dtype=np.int16)},
                                                dims = ["time","nv"], 
                                                attrs=climatology_attrs)
                        
                    climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                                'calendar': ds_in_pr.time.encoding['calendar']})
                    
                    crs = xr.DataArray(np.nan, attrs=ds_in_pr.crs.attrs)

                    file_attrs = {'title': 'Mean seasonal precipitation intensity for {0}'.format(sn),
                    'institution': 'Institute of Meteorology and Climatology, University of '
                    'Natural Resources and Life Sciences, Vienna, Austria',
                    'source': modelname_pr,
                    'comment': 'Precipitation sum divided by total number of wet days per season ({0})'.format(sn),
                    'Conventions': 'CF-1.8'}
                    
                    ds_out = xr.Dataset(data_vars={"precipitation_intensity": pr_annual,
                                                "climatology_bounds": climatology, 
                                                "crs": crs,
                                                "lat": ds_in_pr.lat,
                                                "lon": ds_in_pr.lon}, 
                                        coords={"time":pr_annual.time, "y": ds_in_pr.y,
                                                "x":ds_in_pr.x},
                                        attrs=file_attrs)
                    
                    if path_out.endswith("/"):
                        None
                    else:
                        path_out += "/"
                    outf = path_out + "precipitation_intensity_" + modelname_pr + "_seasonal_{0}_{1}-{2}.nc".format(sn, min(years), max(years))
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