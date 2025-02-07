#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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


path_tx = "/nas/nas5/Projects/OEK15/tx_daily" 
path_out = ""

if path_tx.endswith("/"):
    None
else:
    path_tx += "/"
infiles_tx = sorted(glob.glob(path_tx+"tx_SDM_*.nc"))

for file_tx in infiles_tx:
    ds_in_tx = xr.open_dataset(file_tx)
    for dvar in ds_in_tx.data_vars:
            if "lambert" in dvar:
                crsvar = dvar
    check_endyear = (ds_in_tx.time.dt.month == 12) & (ds_in_tx.time.dt.day == 30)
    time_fullyear = ds_in_tx.time[check_endyear]
    years = np.unique(time_fullyear.dt.year)
    ds_in_tx = ds_in_tx.sel(time=slice(str(min(years)), str(max(years))))
    mask = xr.where(ds_in_tx.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                            skipna=True) 
                    >= -990, 1, np.nan).compute()
    print("*** Loading datasets {0} complete. Mask created.".format(file_tx))
    
    # Calculate indicator with parallel processing
    def parallel_loop(y):
        curind = (ds_in_tx.time.dt.year == y)
        tmax = ds_in_tx.tasmax[curind,:,:].values

        start_day_counter = np.zeros(tmax.shape, dtype = np.int16)
        ntim = tmax.shape[0]
        start_day_counter[0,:,:] = np.where(tmax[0,:,:] >= 30.0, 1, 0)
        t = 1
        while t < ntim:
            pre_counter = start_day_counter[t-1,:,:]
            cur_counter = np.where(tmax[t,:,:] >= 30.0, 1, 0)
            cur_counter = cur_counter + pre_counter
            start_day_counter[t,:,:] = np.where(cur_counter > pre_counter, cur_counter, 0)
            t += 1
        # Und jetzt die Tage, die potentiell Teil einer Kysely-Periode sein können
        participate_days = np.where(tmax >= 25.0, 1, 0)
        # Die Berechungsfunktion ist im helper_function Skript
        (kysely_binary, maxlength) = hf.kysely_periods(start_day_counter, 3, participate_days,
                                                tmax, 30.0, 3)
        kysely_noofdays = kysely_binary.sum(axis=0)
        return kysely_noofdays, maxlength
    
    if __name__ == '__main__':
    # create and configure the process pool
        with Pool(20) as pool:
        # execute tasks, block until all completed
            parallel_results = pool.map(parallel_loop, years)

    temp_kysely_binary = xr.DataArray([x[0] for x in parallel_results], coords={"time": years, "y": ds_in_tx.y, "x": ds_in_tx.x})
    temp_kysely_maxlen = xr.DataArray([x[1] for x in parallel_results], coords={"time": years, "y": ds_in_tx.y, "x": ds_in_tx.x})

    temp_kysely_binary = (temp_kysely_binary * mask).compute()
    temp_kysely_maxlen = (temp_kysely_maxlen * mask).compute()

    print("--> Calculation of indicators for dataset {0} complete".format(file_tx))

    # Add CF-conformal metadata
    # Attributes for the indicator variables:
    attr_dict = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "number_of_days_with_air_temperature_above_threshold", 
                    "units": "1"}
    temp_kysely_binary.attrs = attr_dict
    temp_kysely_maxlen.attrs = attr_dict

    temp_kysely_binary.attrs.update({"cell_methods":"time: sum over days "
                    "(days within Kysely periods)",
                    "long_name": "number of days per year that fall within a "
                 "kysely period" })
    temp_kysely_maxlen.attrs.update({"cell_methods":"time: sum over days "
                    "(maximum duration of annual kysely periods)", 
                    "long_name": "annual maximum length of a kysely period"})

    time_resampled = ds_in_tx.time.resample(time="A")
    start_inds = np.array([x.start for x in time_resampled.groups.values()])
    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
    end_inds[-1] = ds_in_tx.time.size
    end_inds -= 1
    start_inds = start_inds.astype(np.int32)
    end_inds = end_inds.astype(np.int32)
    
    temp_kysely_binary.coords["time"] = ds_in_tx.time[end_inds]
    temp_kysely_maxlen.coords["time"] = ds_in_tx.time[end_inds]
                                            
    temp_kysely_binary.time.attrs.update({"climatology":"climatology_bounds"})
    temp_kysely_maxlen.time.attrs.update({"climatology":"climatology_bounds"})
    
    # Encoding and compression
    encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}
    
    temp_kysely_binary.encoding.update(encoding_dict)
    temp_kysely_maxlen.encoding.update(encoding_dict)
             
    # Climatology variable
    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
    climatology = xr.DataArray(np.stack((ds_in_tx.time[start_inds],
                                            ds_in_tx.time[end_inds]), 
                                        axis=1), 
                                coords={"time": temp_kysely_binary.time, 
                                        "nv": np.arange(2, dtype=np.int16)},
                                dims = ["time","nv"], 
                                attrs=climatology_attrs)
        
    climatology.encoding.update({"dtype":np.float64,'units': ds_in_tx.time.encoding['units'],
                                    'calendar': ds_in_tx.time.encoding['calendar']})
    
    crs = xr.DataArray(np.nan, attrs=ds_in_tx[crsvar].attrs)
    mname = file_tx.replace(".nc","")
    modelname = "_".join(mname.split("/")[-1].split("_")[2:6])

    file_attrs = {'title': "<Kysely Periods>: Indicators calculated"
                    " from downscaled model data",
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': "The file contains the annual number of days in a Kysely "
                    "heat wave period and the maximum length of a Kysely heat "
                    "wave period. Definition: All days are counted as a heat "
                    "wave that, starting afer a period of three consecutive days "
                    "with Tmax >= 30°C, don't fall below Tmax >= 25°C. In addition, "
                    "the average Tmax of the whole period so far does not fall "
                    "below 30°C.",
        'Conventions': 'CF-1.8'}
    
    ds_out = xr.Dataset(data_vars={"kysely_periods_noofdays": temp_kysely_binary,
                                    "kysely_periods_maxlength": temp_kysely_maxlen,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_tx.lat,
                                    "lon": ds_in_tx.lon}, 
                        coords={"time":temp_kysely_binary.time, "y": ds_in_tx.y,
                                "x":ds_in_tx.x},
                        attrs=file_attrs)
    
    if path_out.endswith("/"):
        None
    else:
        path_out += "/"
    outf = path_out + "kysely_periods_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf):
        print("File {0} already exists. Removing...".format(outf))
        os.remove(outf)
    
    # Write final file to disk
    ds_out.to_netcdf(outf, unlimited_dims="time")
    print("Writing file {0} completed!".format(outf))
    ds_in_tx.close()
    ds_out.close()
print("Successfully processed all input files!")