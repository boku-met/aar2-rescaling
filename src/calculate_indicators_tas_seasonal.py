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
    path_to_data = "/nas/nas5/Projects/OEK15/tas_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/hp8/Projekte_Benni/Temp_Data/Indicators"
             
    return path_to_data, output_path
        
def main():
    
    (path_in, path_out) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_pr = sorted(glob.glob(path_in+"tas_SDM_*.nc"))
          
    for file in infiles_pr:
        ds_in_pr = xr.open_dataset(file)
        for dvar in ds_in_pr.data_vars:
            if "lambert" in dvar:
                crsvar = dvar
        check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
        time_fullyear = ds_in_pr.time[check_endyear]
        years = np.unique(time_fullyear.dt.year)
        ds_in_pr = ds_in_pr.sel(time=slice(str(min(years)), str(max(years))))
        mask = xr.where(ds_in_pr.tas.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        
        # Calculate indicator with parallel processing
        sns = np.unique(ds_in_pr.time.dt.season.values)
        for sn in sns: 
            def parallel_loop(y):
                curind = (ds_in_pr.time.dt.year == y) & (ds_in_pr.time.dt.season == sn)
                cur_pr = ds_in_pr.tas[curind,:,:].load()
                start_times = cur_pr.time[0]
                end_times = cur_pr.time[-1]
                pr_sum = cur_pr.mean(dim="time", skipna = True)
                return pr_sum, start_times, end_times

            parallel_results = Parallel(n_jobs=8, prefer="threads")(delayed(parallel_loop)(y) for y in years)
            pr_annual = xr.combine_nested([x[0] for x in parallel_results], concat_dim="time", coords='minimal')
            timestart = xr.combine_nested([x[1] for x in parallel_results], concat_dim="time", coords='minimal')
            timeend = xr.combine_nested([x[2] for x in parallel_results], concat_dim="time", coords='minimal')

            pr_annual = (pr_annual * mask).compute()

            print("--> Calculation of indicators for dataset {0} complete".format(file))
                    
            # Add CF-conformal metadata
            
            # Attributes for the indicator variables:
            attr_dict = {"coordinates": "time y x", 
                        "grid_mapping": "crs", 
                        "standard_name": "near_surface_air_temperature", 
                            "units": "degC"}
            pr_annual.attrs = attr_dict

            pr_annual.attrs.update({"cell_methods":"time: mean over days"
                        "(mean of daily mean temperature)",
                        "long_name": "Seasonal mean temperature for {0}".format(sn)})
         
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
            
            crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)
            mname = file.replace(".nc","")
            modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

            file_attrs = {'title': 'Seasonal Mean Temperature for {0}'.format(sn),
            'institution': 'Institute of Meteorology and Climatology, University of '
            'Natural Resources and Life Sciences, Vienna, Austria',
            'source': modelname,
            'comment': 'Seasonal mean ({0}) of daily mean temperatures'.format(sn),
            'Conventions': 'CF-1.8'}
            
            ds_out = xr.Dataset(data_vars={"tas": pr_annual,
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
            outf = path_out + "tas_{0}".format(sn) + modelname + "_seasonal_{0}-{1}.nc".format(min(years), max(years))
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