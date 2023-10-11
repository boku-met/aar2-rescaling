
import os
import glob
import numpy as np
import xarray as xr
try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def check_isfile(fname):
    if os.path.isfile(fname):
        print("File {0} already exists. Overwriting...".format(fname))
        os.remove(fname)
    file_out = open(fname, mode="xt", encoding="utf-8", newline="\n")
    return file_out

path_cmip5_models = "/hpx/Bennib/CMIP3_data_temp/Scenarios/"
path_cmip5_hist = "/hpx/Bennib/CMIP3_data_temp/Historical/"

gwls = [1.5, 2.0, 3.0, 4.0]

for rcp in  ["B1", "A1B", "A2"]:
    # create filelist for each rcp
    infiles = sorted(glob.glob(path_cmip5_models+"tas_*"+rcp+"*.nc"))
    for file in infiles:
        search_term = file.split("/")[-1][:-12]
        search_hist = search_term.replace(rcp, "historical")
        file_hist = glob.glob(path_cmip5_hist+search_hist+"*.nc")
        assert(len(file_hist) == 1)

        f1_proj = xr.open_dataset(file)
        f1_hist = xr.open_dataset(file_hist[0])

        min_yr = f1_hist.time[0].dt.year.values
        max_yr = f1_proj.time[-1].dt.year.values

        f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2000")), f1_proj.sel(time=slice("2001", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')

        weights = np.cos(np.deg2rad(f1.lat))
        tas_weighted = f1.tas.weighted(weights)
        series_global = tas_weighted.mean(dim=('lat', 'lon'), skipna=True).compute()
        series_global = series_global.resample(time="A", skipna = True).mean()

        ref_gmt = series_global.sel(time=slice(str(min_yr),"1900")).mean(skipna=True)
        anomalies = series_global - ref_gmt
        anomalies_smooth = anomalies.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
        print("Done!")
        mean_years = []
        gwl_list = []
        for gwl in gwls:
            try:
                timeind = (anomalies_smooth.values >= gwl).nonzero()[0][0]
                mean_year = anomalies_smooth[timeind].time.dt.year.values
                period = "{0}-{1}".format(mean_year-10, mean_year+9)
                # add data to lists
                mean_years.append(str(mean_year))
                gwl_list.append(period)
            except IndexError:
                mean_years.append("n/a")    
                gwl_list.append("n/a")    
        modelname = file.split("/")[-1].replace("tas_","").replace(".nc","")
        print("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(modelname, *mean_years, *gwl_list))



