
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_cmip5_models = "/hpx/Bennib/CMIP5_data_temp/OEKS15_models/"
path_cmip5_hist = "/hpx/Bennib/CMIP5_data_temp/OEKS15_historical/"

rcp = "rcp45"
infiles = sorted(glob.glob(path_cmip5_models+"tas_*"+rcp+"*.nc"))
file = infiles[3]
search_term = file.split("/")[-1][:-17]
search_hist = search_term.replace(rcp, "historical")
file_hist = glob.glob(path_cmip5_hist+search_hist+"*.nc")
assert(len(file_hist) == 1)
f1_proj = xr.open_dataset(file)
f1_hist = xr.open_dataset(file_hist[0])
min_yr = f1_hist.time[0].dt.year.values
max_yr = f1_proj.time[-1].dt.year.values
# concat files for complete timeseries

f1 = xr.concat([f1_hist.sel(time=slice(str(min_yr), "2005")), f1_proj.sel(time=slice("2006", str(max_yr)))], dim="time", data_vars='minimal', coords='minimal', compat='override')
# calculate global annual mean temperature timeseries

weights = np.cos(np.deg2rad(f1.lat))
tas_weighted = f1.tas.weighted(weights)
series_global = tas_weighted.mean(dim=('lat', 'lon'), skipna=True).compute()
series_global = series_global.resample(time="A", skipna = True).mean()
ref_gmt = series_global.sel(time=slice(str(min_yr),"1900")).mean(skipna=True)
anomalies = series_global - ref_gmt
anomalies_smooth = anomalies.rolling(time = 20, center = True, min_periods = 20).mean(skipna = True).compute()
anomalies_smooth.name = "ΔTg (°C)"

yrs = anomalies_smooth.time.dt.year.astype(str)
time = np.array(yrs, dtype=np.datetime64)
anomalies_smooth["time"] = time

#plotting routine
fig, axs = plt.subplots(figsize = (7, 4))
anomalies_smooth2.sel(time=slice("2000", "2099")).plot(ax=axs, label = "rcp45")
anomalies_smooth.sel(time=slice("2000", yrs[-1])).plot(ax=axs, label = "rcp85")
axs.yaxis.set_major_locator(MultipleLocator(0.5))
#axs.tick_params(labelrotation = 0)
plt.title("Global mean temperature change relative to 1850-1900")
axs.legend()
axs.set_xlabel("Year")
plt.tight_layout()
fig.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/gwl_method.png", dpi=300)


