#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:10:54 2019

@author: fabianl
"""

import glob
import numpy as np
from multiprocessing.pool import Pool
# import bottleneck as bn # faster that numpyimport matplotlib.pyplot as plt
import xarray as xr # besser als netcdf4
import os 

try: 
    os.nice(3-os.nice(0)) # set current nice level to 3, if it is lower 
except: # nice level already above 3
    pass
# from functions import plot_stmk_borders, plot_stmk_rivers, moving_Window_Regression_uncorrected
# import nclcmaps

### WBIL_during_veg_period': Sum of water budget (RR-ET) during vegetation period
### GSL: days above threshold temperature according to different definitions/methods

###  ----------------------------------------------------------------------------  ####



''' Growing season length 
Duration of the longest continuous period with a mean temperature of at least 5°C. Earlier and later periods are included if they are longer than the sum of days below 5°C inbetween.
'''

path_tas = "/nas/nas5/Projects/OEK15/tas_daily" 
path_out = "/sto0/data/Results/Indicators/"

if path_tas.endswith("/"):
    None
else:
    path_tas += "/"
infiles_tas = sorted(glob.glob(path_tas+"tas_SDM_*.nc"))

for file_tas in infiles_tas:
    ds_in_tas = xr.open_dataset(file_tas)
    for dvar in ds_in_tas.data_vars:
            if "lambert" in dvar:
                crsvar = dvar
    check_endyear = (ds_in_tas.time.dt.month == 12) & (ds_in_tas.time.dt.day == 30)
    time_fullyear = ds_in_tas.time[check_endyear]
    years = np.unique(time_fullyear.dt.year)
    ds_in_tas = ds_in_tas.sel(time=slice(str(min(years)), str(max(years))))
    mask = xr.where(ds_in_tas.tas.isel(time=slice(0,60)).mean(dim="time", 
                                                            skipna=True) 
                    >= -990, 1, np.nan).compute()
    print("*** Loading datasets {0} complete. Mask created.".format(file_tas))

    threshold_temp=5
    threshold_days=5  ## number of days (threshold for start of vegetation period)

    length_y=len(ds_in_tas.y)
    length_x=len(ds_in_tas.x)         

    def calculate_vtp(year):
        curind = (ds_in_tas.time.dt.year == year)
        data_year = ds_in_tas.tas[curind,:,:].load()
        '''  change here if necessary '''     
        
        #begin of vegetation period 
        days_warm_period= np.zeros([len(data_year),length_y, length_x], dtype='f2')
        days_cold_period= np.zeros([len(data_year), length_y, length_x], dtype='f2')
        days_longest_warm_period = np.zeros([length_y, length_x], dtype='f2')

        # last day of year:             
        days_warm_period[len(data_year)-1] = (data_year[len(data_year)-1].values>=threshold_temp)*np.float16(1)
        days_cold_period[len(data_year)-1] = (data_year[len(data_year)-1].values<threshold_temp)*np.float16(1)
        
        # reverse loop over year to find start of vegetation period:
        for dayofyear in range(len(data_year)-2, -1, -1):
            # cumulative sum, if already in a warm period
            days_warm_period[dayofyear] = (data_year[dayofyear].values>=threshold_temp)*np.float16(1) + days_warm_period[dayofyear+1]
            
            # set warm period to zero, where temperature is below threshold 
            days_warm_period[dayofyear][data_year[dayofyear].values<threshold_temp] = 0
            
            # save length of longest warm period: 
            days_longest_warm_period = np.maximum(days_longest_warm_period, days_warm_period[dayofyear])
            
            # cumulative sum of cold days
            days_cold_period[dayofyear] = (data_year[dayofyear].values<threshold_temp)*np.float16(1) + days_cold_period[dayofyear+1]
            
            # set sum of cold days to zero, where the cumulative sum of warm days is larger (or equal?)
            days_cold_period[dayofyear][days_cold_period[dayofyear]<days_warm_period[dayofyear]]=0
            
            # set sum of cold days to zero, when the current warm period is the longest so far
            days_cold_period[dayofyear][days_longest_warm_period<=days_warm_period[dayofyear]] = 0
        
        # day of year of vegetation start, where days_warm_period > days_cold_period for the first time (starting on 1st of Jan)
        veg_begin_day = np.argmax((days_warm_period-days_cold_period)>0, axis=0) +1
        
        #end of vegetation period 
        days_warm_period= np.zeros([len(data_year),length_y, length_x], dtype='f2')
        days_cold_period= np.zeros([len(data_year), length_y, length_x], dtype='f2')
        days_longest_warm_period = np.zeros([length_y, length_x], dtype='f2')

        # first day of year:             
        days_warm_period[0] = (data_year[0].values>=threshold_temp)*np.float16(1)
        days_cold_period[0] = (data_year[0].values<threshold_temp)*np.float16(1)
        
        # loop over year to find start of vegetation period:
        for dayofyear in range(1, len(data_year)):
            # cumulative sum, if already in a warm period
            days_warm_period[dayofyear] = (data_year[dayofyear].values>=threshold_temp)*np.float16(1) + days_warm_period[dayofyear-1]
            
            # set warm period to zero, where temperature is below threshold 
            days_warm_period[dayofyear][data_year[dayofyear].values<threshold_temp] = 0
            
            # save length of longest warm period: 
            days_longest_warm_period = np.maximum(days_longest_warm_period, days_warm_period[dayofyear])
            
            # cumulative sum of cold days
            days_cold_period[dayofyear] = (data_year[dayofyear].values<threshold_temp)*np.float16(1) + days_cold_period[dayofyear-1]
            
            # set sum of cold days to zero, where the cumulative sum of warm days is larger (or equal?)
            days_cold_period[dayofyear][days_cold_period[dayofyear]<days_warm_period[dayofyear]]=0
            
            # set sum of cold days to zero, when the current warm period is the longest so far
            days_cold_period[dayofyear][days_longest_warm_period<=days_warm_period[dayofyear]] = 0
            
            
        # vegetation end, where days_warm_period > days_cold_period for the last(!) time
        reverse_diff = (days_warm_period-days_cold_period)[::-1,:,:] # reverse time axis in matrix
        veg_end_day = len(data_year) - np.argmax((reverse_diff)>0, axis=0)
            
        # number of days in vegetation period between start and end of period:     
        number_of_veg_days = veg_end_day - veg_begin_day +1
        
        number_of_veg_days[days_longest_warm_period<threshold_days] = 0 # zero days do not work otherwise
        veg_begin_day.astype(np.float32)[days_longest_warm_period<threshold_days] = np.nan #
        veg_end_day.astype(np.float32)[days_longest_warm_period<threshold_days] = np.nan # 
        return number_of_veg_days, veg_begin_day, veg_end_day
    
    if __name__ == '__main__':
    # create and configure the process pool
        with Pool(16) as pool:
        # execute tasks, block until all completed
            parallel_results = pool.map(calculate_vtp, years)
    
    temp_number_of_veg_days = xr.DataArray([x[0] for x in parallel_results], coords={"time": years, "y": ds_in_tas.y, "x": ds_in_tas.x})
    temp_veg_begin_day = xr.DataArray([x[1] for x in parallel_results], coords={"time": years, "y": ds_in_tas.y, "x": ds_in_tas.x})
    temp_veg_end_day = xr.DataArray([x[2] for x in parallel_results], coords={"time": years, "y": ds_in_tas.y, "x": ds_in_tas.x})

    temp_number_of_veg_days = (temp_number_of_veg_days * mask).compute()
    temp_veg_begin_day = (temp_veg_begin_day * mask).compute()
    temp_veg_end_day = (temp_veg_end_day * mask).compute()

    print("--> Calculation of indicators for dataset {0} complete".format(file_tas))

    # Add CF-conformal metadata
    # Attributes for the indicator variables:
    attr_dict = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "number_of_days_with_air_temperature_above_threshold", 
                    "units": "1"}
    temp_number_of_veg_days.attrs = attr_dict
    temp_veg_begin_day.attrs = attr_dict
    temp_veg_end_day.attrs = attr_dict

    temp_number_of_veg_days.attrs.update({"cell_methods":"time: sum over days "
                    "(days within vegetation period)",
                    "long_name": "number of days per year that fall within the "
                 "vegetation period" })
    temp_veg_begin_day.attrs.update({"cell_methods":"time: point over days "
                    "(first day of the vegetation period)", 
                    "long_name": "first day of year that falls within the vegetation period"})
    
    temp_veg_end_day.attrs.update({"cell_methods":"time: point over days "
                    "(last day of the vegetation period)", 
                    "long_name": "last day of year that falls within the vegetation period"})

    time_resampled = ds_in_tas.time.resample(time="A")
    start_inds = np.array([x.start for x in time_resampled.groups.values()])
    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
    end_inds[-1] = ds_in_tas.time.size
    end_inds -= 1
    start_inds = start_inds.astype(np.int32)
    end_inds = end_inds.astype(np.int32)
    
    temp_number_of_veg_days.coords["time"] = ds_in_tas.time[end_inds]
    temp_veg_begin_day.coords["time"] = ds_in_tas.time[end_inds]
    temp_veg_end_day.coords["time"] = ds_in_tas.time[end_inds]
                                            
    temp_number_of_veg_days.time.attrs.update({"climatology":"climatology_bounds"})
    temp_veg_begin_day.time.attrs.update({"climatology":"climatology_bounds"})
    temp_veg_end_day.time.attrs.update({"climatology":"climatology_bounds"})
    
    # Encoding and compression
    encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
    
    temp_number_of_veg_days.encoding.update(encoding_dict)
    temp_veg_begin_day.encoding.update(encoding_dict)
    temp_veg_end_day.encoding.update(encoding_dict)
             
    # Climatology variable
    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
    climatology = xr.DataArray(np.stack((ds_in_tas.time[start_inds],
                                            ds_in_tas.time[end_inds]), 
                                        axis=1), 
                                coords={"time": temp_number_of_veg_days.time, 
                                        "nv": np.arange(2, dtype=np.int16)},
                                dims = ["time","nv"], 
                                attrs=climatology_attrs)
        
    climatology.encoding.update({"dtype":np.float64,'units': ds_in_tas.time.encoding['units'],
                                    'calendar': ds_in_tas.time.encoding['calendar']})
    
    crs = xr.DataArray(np.nan, attrs=ds_in_tas[crsvar].attrs)
    mname = file_tas.replace(".nc","")
    modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

    file_attrs = {'title': "<Vegetation Period>: Indicators calculated"
                    " from downscaled model data",
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': "The file contains the annual number of days within the vegetation period"
                    "and the first and last day of the year within the vegetation period."
                    " Definition: Duration of the longest continuous period with a mean temperature"
                    " of at least 5°C. Earlier and later periods are included if they are longer"
                    " than the sum of days below 5°C inbetween.",
                     'Conventions': 'CF-1.8'}
    
    ds_out = xr.Dataset(data_vars={"vegetation_period_number_of_days": temp_number_of_veg_days,
                                   "vegetation_period_first_day": temp_veg_begin_day,
                                   "vegetation_period_last_day": temp_veg_end_day,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_tas.lat,
                                    "lon": ds_in_tas.lon}, 
                        coords={"time":temp_number_of_veg_days.time, "y": ds_in_tas.y,
                                "x":ds_in_tas.x},
                        attrs=file_attrs)
    
    if path_out.endswith("/"):
        None
    else:
        path_out += "/"
    outf = path_out + "vegetation_period_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf):
        print("File {0} already exists. Removing...".format(outf))
        os.remove(outf)
    
    # Write final file to disk
    ds_out.to_netcdf(outf, unlimited_dims="time")
    print("Writing file {0} completed!".format(outf))
    ds_in_tas.close()
    ds_out.close()
print("Successfully processed all input files!")