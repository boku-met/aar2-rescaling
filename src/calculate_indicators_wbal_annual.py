#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:10:54 2019

@author: bennib
"""

import glob
import numpy as np
from multiprocessing.pool import Pool
import xarray as xr 
import os 

try: 
    os.nice(3-os.nice(0)) # set current nice level to 3, if it is lower 
except: # nice level already above 3
    pass


path_pr = "/nas/nas5/Projects/OEK15/pr_daily/" 
path_pet = "/nas/nas5/Projects/OEK15/PET_daily/" 
path_out = "/nas/nas5/Projects/OEK15/WBAL_annual/"

infiles_pr = sorted(glob.glob(path_pr+"pr_SDM_*.nc"))
infiles_pet = sorted(glob.glob(path_pet+"PET_*.nc"))

for file_pet in infiles_pet:
    mname = file_pet.replace(".nc","")
    modelname = "_".join(mname.split("/")[-1].split("_")[1:6])
    file_pr = [x for x in infiles_pr if modelname in x]
    
    ds_in_pr = xr.open_dataset(file_pr[0])
    ds_in_pet = xr.open_dataset(file_pet)

    for dvar in ds_in_pr.data_vars:
            if "lambert" in dvar:
                crsvar = dvar
    check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
    time_fullyear = ds_in_pr.time[check_endyear]
    years = np.unique(time_fullyear.dt.year)
    ds_in_pr = ds_in_pr.sel(time=slice(str(min(years)), str(max(years))))
    ds_in_pet = ds_in_pet.sel(time=slice(str(min(years)), str(max(years))))

    years_pet = np.unique(ds_in_pet.time.dt.year)
    if years_pet.size != years.size:
        print('Faulty file detected. Pr has time variable of {0} and '
        'PET has time variable of {1}. Skipping file {2}'.format(years.size,
        years_pet.size,file_pet))
        continue

    mask = xr.where(ds_in_pet.PET.isel(time=slice(0,60)).mean(dim="time", 
                                                            skipna=True) 
                    >= -990, 1, np.nan).compute()
    print("*** Loading datasets {0} complete. Mask created.".format(file_pet))      

    def calculate_vtp(y):
        curind_pr = (ds_in_pr.time.dt.year == y)
        curind_pet = (ds_in_pet.time.dt.year == y)
        pet_year = ds_in_pet.PET[curind_pet,:,:].load()
        pr_year = ds_in_pr.pr[curind_pr,:,:].load()

        pet_sum = pet_year.resample(time="YE", skipna=True).sum()
        pr_sum = pr_year.resample(time="YE", skipna = True).sum()

        wbal = (pr_sum - pet_sum).astype('f4')
        
        return pet_sum, wbal

    if __name__ == '__main__':
    # create and configure the process pool
        with Pool(24) as pool:
        # execute tasks, block until all completed
            parallel_results = pool.map(calculate_vtp, years)
    
    pet_mon = xr.concat([x[0] for x in parallel_results], dim="time")
    wbal_mon = xr.concat([x[1] for x in parallel_results], dim="time")
    pet_mon = (pet_mon * mask).compute()
    wbal_mon = (wbal_mon * mask).compute()

    print("--> Calculation of indicators for dataset {0} complete".format(file_pet))

    attr_dict_wb = {"coordinates": "time lat lon", 
                        "grid_mapping": "crs", 
                        "standard_name": "climatological_waterbalance", 
                        "units": "kg m-2"}   
    attr_dict_pet = {"coordinates": "time lat lon", 
    "grid_mapping": "crs", 
    "standard_name": "reference_evapotranspiration", 
    "units": "kg m-2"}                  
    pet_mon.attrs = attr_dict_pet
    wbal_mon.attrs = attr_dict_wb

    pet_mon.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                    "(sum of daily reference evapotranspiration)",
                    "long_name": "Annual reference evapotranspiration" })
    wbal_mon.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                    "(balance of precipitation - reference evapotranspiration)",
                    "long_name": "Annual climatological water balance" })

    time_resampled = ds_in_pr.time.resample(time="YE")
    start_inds = np.array([x.start for x in time_resampled.groups.values()])
    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
    end_inds[-1] = ds_in_pr.time.size
    end_inds -= 1
    start_inds = start_inds.astype(np.int32)
    end_inds = end_inds.astype(np.int32)

    pet_mon.coords["time"] = ds_in_pr.time[end_inds]
    wbal_mon.coords["time"] = ds_in_pr.time[end_inds]
                                                    
    pet_mon.time.attrs.update({"climatology":"climatology_bounds"})
    wbal_mon.time.attrs.update({"climatology":"climatology_bounds"})
            
    # Encoding and compression
    encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                        'complevel': 1, 'fletcher32': False, 
                        'contiguous': False}

    pet_mon.encoding = encoding_dict
    wbal_mon.encoding = encoding_dict
                                    
    # Climatology variable
    climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
    climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                            ds_in_pr.time[end_inds]), 
                                        axis=1), 
                                coords={"time": pet_mon.time, 
                                        "nv": np.arange(2, dtype=np.int16)},
                                dims = ["time","nv"], 
                                attrs=climatology_attrs)
        
    climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                    'calendar': ds_in_pr.time.encoding['calendar']})

    crs = xr.DataArray(np.nan, attrs=ds_in_pr[crsvar].attrs)


    file_attrs_wb = {'title': 'Annual Climatological Waterbalance',
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': 'Annual precipitation sum - Annual reference evapotranspiration sum',
        'Conventions': 'CF-1.8'}

    file_attrs_pet = {'title': 'Annual Reference Evapotranspiration',
        'institution': 'Institute of Meteorology and Climatology, University of '
        'Natural Resources and Life Sciences, Vienna, Austria',
        'source': modelname,
        'comment': 'Annual reference evapotranspiration sum',
        'Conventions': 'CF-1.8'}

    ds_out_wb = xr.Dataset(data_vars={"WBAL": wbal_mon,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_pet.lat,
                                    "lon": ds_in_pet.lon}, 
                        coords={"time":wbal_mon.time, "y": ds_in_pet.y,
                                "x":ds_in_pet.x},
                        attrs=file_attrs_wb)

    ds_out_pet = xr.Dataset(data_vars={"PET": pet_mon,
                                    "climatology_bounds": climatology, 
                                    "crs": crs,
                                    "lat": ds_in_pet.lat,
                                    "lon": ds_in_pet.lon}, 
                        coords={"time":pet_mon.time, "y": ds_in_pet.y,
                                "x":ds_in_pet.x},
                        attrs=file_attrs_pet)

    if path_out.endswith("/"):
        None
    else:
        path_out += "/"

    outf_wb = path_out + "WBAL_" + modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf_wb):
        print("File {0} already exists. Removing...".format(outf_wb))
        os.remove(outf_wb)

    outf_pet = path_out + "PET_"+ modelname + "_annual_{0}-{1}.nc".format(min(years), max(years))
    if os.path.isfile(outf_pet):
        print("File {0} already exists. Removing...".format(outf_pet))
        os.remove(outf_pet)

    # Write final file to disk
    ds_out_pet.to_netcdf(outf_pet, unlimited_dims="time")
    ds_out_wb.to_netcdf(outf_wb, unlimited_dims="time")

    print("Writing file {0} completed!".format(outf_pet))
    ds_in_pr.close()
    ds_in_pet.close()
    ds_out_pet.close()
    ds_out_wb.close()
print("Successfully processed all input files!")