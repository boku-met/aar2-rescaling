#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benedikt.becsi<at>boku.ac.at
"""
import os
import glob
from joblib import Parallel, delayed
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
    path_to_tx = "/nas/nas5/Projects/OEK15/tx_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = ""
             
    return path_to_tx, output_path
        
def main():
    
    (path_tx, path_out) = user_data()
    
    if path_tx.endswith("/"):
        None
    else:
        path_tx += "/"
    infiles_tx = sorted(glob.glob(path_tx+"tx_SDM_*.nc"))

    for file_tx in infiles_tx:
        ds_in_tx = xr.open_dataset(file_tx)
        check_endyear = (ds_in_tx.time.dt.month == 12) & (ds_in_tx.time.dt.day == 30)
        time_fullyear = ds_in_tx.time[check_endyear]
        years = np.unique(time_fullyear.dt.year)
        ds_in_tx = ds_in_tx.sel(time=slice(str(min(years)), str(max(years))))
        mask = xr.where(ds_in_tx.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                                skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading datasets {0}, {1} complete. Mask created.".format(file_tx, file_tas))
        
        # Calculate indicator with parallel processing
        start_inds = []
        end_inds = []
        def parallel_loop(y):
            curind = (ds_in_tx.time.dt.year == y)
            start_inds.append(np.nonzero(curind.values)[0][0])
            end_inds.append(np.nonzero(curind.values)[0][-1])

            tmax = ds_in_tx.tasmax[curind,:,:].values

            start_day_counter = np.zeros(tmax.shape, dtype = np.int16)
            ntim = tmax.shape[0]
            start_day_counter[0,:,:] = np.where(tmax[0,:,:] >= 30.0, 1, 0)
            t = 1
            while t < ntim:
                pre_counter = start_day_counter[t-1,:,:]
                cur_counter = np.where(tmax[t,:,:] <= 30.0, 1, 0)
                cur_counter = cur_counter + pre_counter
                start_day_counter[t,:,:] = np.where(cur_counter > pre_counter, cur_counter, 0)
                t += 1
            # Und jetzt die Tage, die potentiell Teil einer Kysely-Periode sein können
            participate_days = np.where(tmax >= 25.0, 1, 0)
            # Die Berechungsfunktion ist im helper_function Skript
            (kysely_binary, maxlength) = hf.kysely_periods(start_day_counter, 3, participate_days,
                                                    tmax, 30.0, 3)
            return kysely_binary, maxlength

        parallel_results = Parallel(n_jobs=16, prefer="threads")(delayed(parallel_loop)(y) for y in years)
        temp_kysely_binary = xr.DataArray([x[0] for x in parallel_results], coords={"time": ds_in_tx.time, "y": ds_in_tx.y, "x": ds_in_tx.x})
        temp_kysely_maxlen = xr.DataArray([x[1] for x in parallel_results], coords={"time": years, "y": ds_in_tx.y, "x": ds_in_tx.x})

        temp_kysely_binary = (temp_kysely_binary * mask).compute()
        temp_kysely_maxlen = (temp_kysely_maxlen * mask).compute()

        print("--> Calculation of indicators for dataset {0} complete".format(file_tx))
        
main()