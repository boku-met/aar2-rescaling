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
    path_to_data = "/sto0/data/Intermediate/Forsite2/hist_daily/PET_AAR2/" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = "/sto0/data/Results/Indicators/"
             
    return path_to_data, output_path
        
(path_in, path_out) = user_data()

if path_in.endswith("/"):
    None
else:
    path_in += "/"
infiles_pr = sorted(glob.glob(path_in+"PET_Austria_*.nc"))

ds_in_pr = xr.open_mfdataset(infiles_pr)
for dvar in ds_in_pr.coords:
    if "spatial_r" in dvar:
        crsvar = dvar
check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
time_fullyear = ds_in_pr.time[check_endyear]
years = np.unique(time_fullyear.dt.year)

mask = xr.where(ds_in_pr.PET.isel(time=slice(0,60)).mean(dim="time", 
                                                            skipna=True) 
                >= -990, 1, np.nan).compute()
print("*** Loading dataset complete. Mask created.")


lat = xr.open_dataset('/sto1/projects/forsite2/data/DGM/dgm_oe_250m_EPSG31287_latlon.nc')['lat'].astype('f4').load()
lat = lat/180*np.pi
y_vector = lat.y[0::4]
x_vector = lat.x[0::4]

def parallel_loop(y):
    ds_in_pet = xr.open_dataset("/sto0/data/Intermediate/Forsite2/hist_daily/PET_AAR2/PET_Austria_{0}_pyet_fao56_1km.nc".format(y))
    ds_in_pr = xr.open_dataset("/sto0/data/Intermediate/Forsite2/hist_daily/pr/pr_Austria_{0}_250m_v2.1.nc".format(y))
    pet_year = ds_in_pet.PET.load()
    pr_year = ds_in_pr.pr.sel(y=y_vector, x=x_vector).load()
    pet_sum = pet_year.resample(time="YE", skipna=True).sum()
    pr_sum = pr_year.resample(time="YE", skipna=True).sum()

    wbal = (pr_sum - pet_sum).astype('f4')
        
    return pet_sum, wbal, pr_sum
  
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(18) as pool:
        # execute tasks, block until all completed
        parallel_results = pool.map(parallel_loop, years)
    # process pool is closed automatically

pet_mon = xr.concat([x[0] for x in parallel_results], dim="time")
wbal_mon = xr.concat([x[1] for x in parallel_results], dim="time")
pr_mon = xr.concat([x[2] for x in parallel_results], dim="time")

pet_mon = (pet_mon * mask).compute()
pr_mon  = (pr_mon * mask).compute()
wbal_mon = (wbal_mon * mask).compute()

print("--> Calculation of indicators for dataset complete")

attr_dict_pr = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "precipitation_amount", 
                    "units": "kg m-2"}
attr_dict_wb = {"coordinates": "time lat lon", 
                    "grid_mapping": "crs", 
                    "standard_name": "climatological_waterbalance", 
                    "units": "kg m-2"}   
attr_dict_pet = {"coordinates": "time lat lon", 
"grid_mapping": "crs", 
"standard_name": "reference_evapotranspiration", 
"units": "kg m-2"}                  
pet_mon.attrs = attr_dict_pet
pr_mon.attrs = attr_dict_pr
wbal_mon.attrs = attr_dict_wb

pr_mon.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                "(sum of daily precipitation sums)",
                "long_name": "Annual precipitation sum" })
pet_mon.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                "(sum of daily reference evapotranspiration)",
                "long_name": "Annual reference evapotranspiration" })
wbal_mon.attrs.update({"cell_methods":"time: sum within days time: sum over days "
                "(balance of precipitation - reference evapotranspiration)",
                "long_name": "Annual climatological water balance" })

time_resampled = ds_in_pr.time.resample(time="A")
start_inds = np.array([x.start for x in time_resampled.groups.values()])
end_inds = np.array([x.stop for x in time_resampled.groups.values()])
end_inds[-1] = ds_in_pr.time.size
end_inds -= 1
start_inds = start_inds.astype(np.int32)
end_inds = end_inds.astype(np.int32)

pet_mon.coords["time"] = ds_in_pr.time[end_inds]
wbal_mon.coords["time"] = ds_in_pr.time[end_inds]
pr_mon.coords["time"] = ds_in_pr.time[end_inds]
                                                
pet_mon.time.attrs.update({"climatology":"climatology_bounds"})
wbal_mon.time.attrs.update({"climatology":"climatology_bounds"})
pr_mon.time.attrs.update({"climatology":"climatology_bounds"})
        
# Encoding and compression
encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                    'complevel': 1, 'fletcher32': False, 
                    'contiguous': False}

pet_mon.encoding = encoding_dict
wbal_mon.encoding = encoding_dict
pr_mon.encoding = encoding_dict
                                
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
    'source': "FORSITE2",
    'comment': 'Annual precipitation sum - Annual reference evapotranspiration sum',
    'Conventions': 'CF-1.8'}

file_attrs_pr = {'title': 'Annual Precipitation Sum',
'institution': 'Institute of Meteorology and Climatology, University of '
'Natural Resources and Life Sciences, Vienna, Austria',
'source': "FORSITE2",
'comment': 'Annual sum of daily precipitation sums',
'Conventions': 'CF-1.8'}

file_attrs_pet = {'title': 'Annual Reference Evapotranspiration',
    'institution': 'Institute of Meteorology and Climatology, University of '
    'Natural Resources and Life Sciences, Vienna, Austria',
    'source': "FORSITE2",
    'comment': 'Annual reference evapotranspiration sum',
    'Conventions': 'CF-1.8'}

ds_out_pr = xr.Dataset(data_vars={"pr": pr_mon,
                                "climatology_bounds": climatology, 
                                "crs": crs,
                                "lat": ds_in_pr.lat,
                                "lon": ds_in_pr.lon}, 
                    coords={"time":pr_mon.time, "y": ds_in_pr.y,
                            "x":ds_in_pr.x},
                    attrs=file_attrs_pr)

ds_out_wb = xr.Dataset(data_vars={"WBAL": wbal_mon,
                                "climatology_bounds": climatology, 
                                "crs": crs,
                                "lat": ds_in_pr.lat,
                                "lon": ds_in_pr.lon}, 
                    coords={"time":wbal_mon.time, "y": ds_in_pr.y,
                            "x":ds_in_pr.x},
                    attrs=file_attrs_wb)

ds_out_pet = xr.Dataset(data_vars={"PET": pet_mon,
                                "climatology_bounds": climatology, 
                                "crs": crs,
                                "lat": ds_in_pr.lat,
                                "lon": ds_in_pr.lon}, 
                    coords={"time":pet_mon.time, "y": ds_in_pr.y,
                            "x":ds_in_pr.x},
                    attrs=file_attrs_pet)

if path_out.endswith("/"):
    None
else:
    path_out += "/"

outf_pr = path_out + "pr_FORSITE2_annual_{0}-{1}.nc".format(min(years), max(years))
if os.path.isfile(outf_pr):
    print("File {0} already exists. Removing...".format(outf_pr))
    os.remove(outf_pr)

outf_wb = path_out + "WBAL_FORSITE2_annual_{0}-{1}.nc".format(min(years), max(years))
if os.path.isfile(outf_wb):
    print("File {0} already exists. Removing...".format(outf_wb))
    os.remove(outf_wb)

outf_pet = path_out + "PET_FORSITE2_annual_{0}-{1}.nc".format(min(years), max(years))
if os.path.isfile(outf_pet):
    print("File {0} already exists. Removing...".format(outf_pet))
    os.remove(outf_pet)

# Write final file to disk
ds_out_pet.to_netcdf(outf_pet, unlimited_dims="time")
ds_out_wb.to_netcdf(outf_wb, unlimited_dims="time")
ds_out_pr.to_netcdf(outf_pr, unlimited_dims="time")

print("Successfully processed all input files!")