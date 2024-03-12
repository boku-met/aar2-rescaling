
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

path_cmip5_models = "/hpx/Bennib/CMIP5_data_temp/CMIP5_all_models/"
path_cmip5_hist = "/hpx/Bennib/CMIP5_data_temp/CMIP5_all_hist/"

gwls = [2035, 2050, 2088]

for rcp in  ["rcp26", "rcp45", "rcp60", "rcp85"]:
    # create filelist for each rcp
    infiles = sorted(glob.glob(path_cmip5_models+"tas_*"+rcp+"*.nc"))
    for file in infiles:
        search_term = file.split("/")[-1][:-17]
        search_hist = search_term.replace(rcp, "historical")
        file_hist = glob.glob(path_cmip5_hist+search_hist+"*.nc")
        assert(len(file_hist) == 1)

        f1_proj = xr.open_dataset(file)
        f1_hist = xr.open_dataset(file_hist[0])

        min_yr = f1_hist.time[0].dt.year.values
        max_yr = f1_proj.time[-1].dt.year.values

        f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2005")), f1_proj.sel(time=slice("2006", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')
        try:
            timetest = f1.time.sel(time="2005-12")
        except Exception:
            print("Found ya! "+file_hist[0])
            print(file)
            f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2005-11")), 
                            f1_proj.sel(time=slice("2005-12", str(max_yr)))],
                            dim="time", data_vars='minimal', coords='minimal', 
                            compat='override')
        weights = np.cos(np.deg2rad(f1.lat))
        tas_weighted = f1.tas.weighted(weights)
        series_global = tas_weighted.mean(dim=('lat', 'lon'), skipna=True).compute()
        series_global = series_global.resample(time="A", skipna = True).mean()

        ref_gmt = series_global.sel(time=slice(str(min_yr),"1900")).mean(skipna=True)
        anomalies = series_global - ref_gmt
        anomalies_smooth = anomalies.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
        print("Done!")
        mean_years = []
        for gwl in gwls:
            try:
                timeind = anomalies_smooth.time.dt.year == gwl
                mean_year = anomalies_smooth[timeind].values
                # add data to lists
                mean_years.append(str(mean_year))
            except IndexError:
                mean_years.append("n/a")        
        modelname = file.split("/")[-1].replace("tas_","").replace(".nc","")
        print("{0};{1};{2};{3};{4}\n".format(modelname,rcp, *mean_years))



