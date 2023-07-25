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
import helper_functions as hf

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires precipitation data.
    path_to_tx = "/nas/nas5/Projects/OEK15/tx_daily" 
    path_to_tas = "/nas/nas5/Projects/OEK15/tas_daily" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/hp8/Projekte_Benni/Temp_Data/Indicators"
             
    return path_to_tx, path_to_tas, output_path
        
def main():
    
    (path_tx, path_tas, path_out) = user_data()
    
    if path_tx.endswith("/"):
        None
    else:
        path_tx += "/"
    infiles_tx = sorted(glob.glob(path_tx+"tx_SDM_*.nc"))

    if path_tas.endswith("/"):
        None
    else:
        path_tas += "/"
    infiles_tas = sorted(glob.glob(path_tas+"tas_SDM_*.nc"))

    for file_tx in infiles_tx:
        mname_tx = file_tx.replace(".nc","")
        modelname_tx = "_".join(mname_tx.split("/")[-1].split("_")[2:6])
        for file_tas in infiles_tas:
            mname_tas = file_tas.replace(".nc","")
            modelname_tas = "_".join(mname_tas.split("/")[-1].split("_")[2:6])
            if modelname_tx == modelname_tas:
                ds_in_tx = xr.open_dataset(file_tx)
                ds_in_tas = xr.open_dataset(file_tas)

                for dvar in ds_in_tx.data_vars:
                    if "lambert" in dvar:
                        crsvar = dvar
                check_endyear = (ds_in_tx.time.dt.month == 3) & (ds_in_tx.time.dt.day == 30)
                time_fullyear = ds_in_tx.time[check_endyear]
                years = np.unique(time_fullyear.dt.year)
                ds_in_tx = ds_in_tx.sel(time=slice(str(min(years)), str(max(years))))
                ds_in_tas = ds_in_tas.sel(time=slice(str(min(years)), str(max(years))))
                mask = xr.where(ds_in_tx.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                                        skipna=True) 
                                >= -990, 1, np.nan).compute()
                print("*** Loading datasets {0}, {1} complete. Mask created.".format(file_tx, file_tas))
                
                # Calculate indicator with parallel processing
                start_inds = []
                end_inds = []
                def parallel_loop(y):
                    curind = (ds_in_tx.time.dt.year == y) & (ds_in_tx.time.dt.month <= 3)
                    start_inds.append(np.nonzero(curind.values)[0][0])
                    end_inds.append(np.nonzero(curind.values)[0][-1])

                    tmax = ds_in_tx.tasmax[curind,:,:].values
                    tmean = ds_in_tas.tas[curind,:,:].values

                    start_day_counter = np.zeros(tmax.shape, dtype = np.int16)
                    ntim = tmax.shape[0]
                    start_day_counter[0,:,:] = np.where(tmax[0,:,:] <= 2, 1, 0)
                    t = 1
                    while t < ntim:
                        pre_counter = start_day_counter[t-1,:,:]
                        cur_counter = np.where(tmax[t,:,:] <= 2, 1, 0)
                        cur_counter = cur_counter + pre_counter
                        start_day_counter[t,:,:] = np.where(cur_counter > pre_counter, cur_counter, 0)
                        t += 1
                    # Und jetzt die Tage, die potentiell Teil einer Kysely-Periode sein können
                    participate_days = np.where(tmax <= 2, 1, 0)
                    # Die Berechungsfunktion ist im helper_function Skript
                    kyselydays, maxlength = hf.kysely_periods(start_day_counter, 1, participate_days,
                                                            tmean, -0.001, 10, lt=True)
                    
                    return np.sum(kyselydays, axis=0)

                parallel_results = Parallel(n_jobs=16, prefer="threads")(delayed(parallel_loop)(y) for y in years)
                pr_annual = xr.DataArray(parallel_results, coords={"time": years, "y": ds_in_tx.y, "x": ds_in_tx.x})

                pr_annual = (pr_annual * mask).compute()

                print("--> Calculation of indicators for dataset {0} complete".format(file_tx))
                        
                # Add CF-conformal metadata
                
                # Attributes for the indicator variables:
                attr_dict = {"coordinates": "time y x", 
                            "grid_mapping": "crs", 
                            "standard_name": "number_of_days_with_air_temperature_below_threshold", 
                                "units": "1"}
                pr_annual.attrs = attr_dict

                pr_annual.attrs.update({"cell_methods":"time: sum over days",
                            "long_name": "Number of days in Jan-Mar within cold spells" })
                start_inds.sort()
                end_inds.sort()
                
                pr_annual.coords["time"] = ds_in_tx.time[end_inds]
                                                        
                pr_annual.time.attrs.update({"climatology":"climatology_bounds"})
                
                # Encoding and compression
                encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                                'complevel': 1, 'fletcher32': False, 
                                'contiguous': False}
                
                pr_annual.encoding = encoding_dict
                                        
                # Climatology variable
                climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
                climatology = xr.DataArray(np.stack((ds_in_tx.time[start_inds],
                                                        ds_in_tx.time[end_inds]), 
                                                    axis=1), 
                                            coords={"time": pr_annual.time, 
                                                    "nv": np.arange(2, dtype=np.int16)},
                                            dims = ["time","nv"], 
                                            attrs=climatology_attrs)
                    
                climatology.encoding.update({"dtype":np.float64,'units': ds_in_tx.time.encoding['units'],
                                            'calendar': ds_in_tx.time.encoding['calendar']})
                
                crs = xr.DataArray(np.nan, attrs=ds_in_tx[crsvar].attrs)

                file_attrs = {'title': 'Days in cold spells (Jan-Mar)',
                'institution': 'Institute of Meteorology and Climatology, University of '
                'Natural Resources and Life Sciences, Vienna, Austria',
                'source': modelname_tx,
                'comment': "Indicator definition: Number of days during January-March that"
                " fall within a cold spell. Definition: A cold spell is an at least 10-day-period "
                "where each day has a tmax <= 2 degC and the average tmean of the period is below 0 degC.",
                'Conventions': 'CF-1.8'}
                
                ds_out = xr.Dataset(data_vars={"cold_spells_noofdays": pr_annual,
                                            "climatology_bounds": climatology, 
                                            "crs": crs,
                                            "lat": ds_in_tx.lat,
                                            "lon": ds_in_tx.lon}, 
                                    coords={"time":pr_annual.time, "y": ds_in_tx.y,
                                            "x":ds_in_tx.x},
                                    attrs=file_attrs)
                
                if path_out.endswith("/"):
                    None
                else:
                    path_out += "/"
                outf = path_out + "coldspells_" + modelname_tx + "_annual_{0}-{1}.nc".format(min(years), max(years))
                if os.path.isfile(outf):
                    print("File {0} already exists. Removing...".format(outf))
                    os.remove(outf)
                
                # Write final file to disk
                ds_out.to_netcdf(outf, unlimited_dims="time")
                print("Writing file {0} completed!".format(outf))
                ds_in_tx.close()
                ds_in_tas
                ds_out.close()
    print("Successfully processed all input files!")
main()