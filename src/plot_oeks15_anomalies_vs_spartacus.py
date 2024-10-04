import os
import glob
import copy
from multiprocessing.pool import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass


infiles_spart_tn = sorted(glob.glob("/sto0/data/Input/Gridded/SPARTACUS/V2.1/TN/*.nc"))
infiles_spart_tx = sorted(glob.glob("/sto0/data/Input/Gridded/SPARTACUS/V2.1/TX/*.nc"))
infiles_oeks15_rcp26 = sorted(glob.glob("/nas/nas5/Projects/OEK15/tas_daily/*rcp26*.nc"))
infiles_oeks15_rcp45 = sorted(glob.glob("/nas/nas5/Projects/OEK15/tas_daily/*rcp45*.nc"))
infiles_oeks15_rcp85 = sorted(glob.glob("/nas/nas5/Projects/OEK15/tas_daily/*rcp85*.nc"))
path_lookup_table = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP5_OEKS15_lookup_table.csv"

lookup_table = open(path_lookup_table, mode="rt")
lookup_table = [x.replace("\n","") for x in lookup_table]

mask_ds = xr.open_dataset(infiles_oeks15_rcp45[0])
mask = xr.where(mask_ds.tas[20:70,:,:].mean(dim="time", skipna=True) >= -999, 1, np.nan)

def prepare_oeks15(file):
    # get gwl periods
    modelname = "_".join(file.split("/")[-1].split("_")[2:6]).replace(".nc","")
    for l in lookup_table:
        if modelname in l:
            ll = l.split(";")
            if ll[1]:
                p_start = ll[1].split("-")[0]
                p_end = ll[1].split("-")[1]
    f1 = xr.open_dataset(file).sel(time=slice("1971","2097"))
    tmean_annual = f1.tas.resample(time="YE", skipna=True).mean()
    tmean_anomalies = tmean_annual - tmean_annual.sel(time=slice("2001", "2020")).mean(dim="time", skipna=True)
    tmean_anomalies = tmean_anomalies * mask
    tmean_anomalies_areamean = tmean_anomalies.mean(dim=("y", "x"), skipna=True)
    return tmean_anomalies_areamean

if __name__ == '__main__':
    # create and configure the process pool
    with Pool(24) as pool:
        # execute tasks, block until all completed
        results_rcp26 = pool.map(prepare_oeks15, infiles_oeks15_rcp26)
    # process pool is closed automatically

if __name__ == '__main__':
    # create and configure the process pool
    with Pool(24) as pool:
        # execute tasks, block until all completed
        results_rcp45 = pool.map(prepare_oeks15, infiles_oeks15_rcp45)
    # process pool is closed automatically


if __name__ == '__main__':
    # create and configure the process pool
    with Pool(24) as pool:
        # execute tasks, block until all completed
        results_rcp85 = pool.map(prepare_oeks15, infiles_oeks15_rcp85)
    # process pool is closed automatically


#alternative using already finished anomalies (run x3 for each RCP):
anomalies_rcp85 = []
for file in infiles_oeks15_rcp85:
    # open file and load in DataArray for processing
    f1 = open(file, mode="rt")
    f1 = [x.replace("\n","") for x in f1]
    data = [x.split(";")[1] for x in f1[1:]]
    years = [x.split(";")[0] for x in f1[1:]]
    time = np.array(years, np.datetime64)
    time=xr.DataArray(time, coords={"time":time})
    anomalies = xr.DataArray(np.array(data, dtype=np.float32), coords={"time":time}).sel(time=slice("1971","2097"))
    anomalies_rcp85.append(anomalies)


newtime = np.arange("1971","2098", dtype=np.datetime64)
newtime = xr.DataArray(newtime, coords={"time":newtime})

mname_rcp26 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp26]
mname_rcp45 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp45]
mname_rcp85 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp85]

ds_rcp26 = xr.DataArray(results_rcp26, coords={"ens": mname_rcp26, "time":newtime})
ds_rcp45 = xr.DataArray(results_rcp45, coords={"ens": mname_rcp45, "time":newtime})
ds_rcp85 = xr.DataArray(results_rcp85, coords={"ens": mname_rcp85, "time":newtime})

# calcualte spartacus timeseries
ds_spart_tn = xr.open_mfdataset(infiles_spart_tn, concat_dim="time", combine="nested")
ds_spart_tx = xr.open_mfdataset(infiles_spart_tx, concat_dim="time", combine="nested")

spart_tn = ds_spart_tn.TN.sel(time=slice("1971", "2023"))
spart_tx = ds_spart_tx.TX.sel(time=slice("1971", "2023"))
spart_tas = (spart_tn + spart_tx)/2

spart_tas_annual = spart_tas.resample(time="YE", skipna=True).mean()
anomalies_spart = spart_tas_annual - spart_tas_annual.sel(time=slice("1971","2000")).mean(dim="time", skipna=True)
anomalies_areamean = anomalies_spart.mean(dim=("y","x"), skipna=True).compute()

# filter the data
time_tmean = np.unique(ds_rcp26.time.dt.year)
time_spart = np.unique(anomalies_areamean.time.dt.year)

rcp26_lowess = copy.deepcopy(ds_rcp26)
rcp45_lowess = copy.deepcopy(ds_rcp45)
rcp85_lowess = copy.deepcopy(ds_rcp85)
for e in rcp26_lowess.ens:
    cur_tmean = ds_rcp26.sel(ens = e)
    rcp26_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (30 / cur_tmean.size), it = 2)
    
for e in rcp45_lowess.ens:
    cur_tmean = ds_rcp45.sel(ens = e)
    rcp45_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (30 / cur_tmean.size), it = 2)
    
for e in rcp85_lowess.ens:
    cur_tmean = ds_rcp85.sel(ens = e)
    rcp85_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (30 / cur_tmean.size), it = 2)

spart_lowess = copy.deepcopy(anomalies_areamean)
spart_lowess.values = lowess(anomalies_areamean.values, time_spart, return_sorted=False, 
                             frac = (30 / anomalies_areamean.size), it = 2)


median_rcp26 = rcp26_lowess.median(dim="ens", skipna=True)
median_rcp45 = rcp45_lowess.median(dim="ens", skipna=True)
median_rcp85 = rcp85_lowess.median(dim="ens", skipna=True)
q10_rcp26 = rcp26_lowess.quantile(0.1, dim="ens", skipna=True)
q90_rcp26 = rcp26_lowess.quantile(0.9, dim="ens", skipna=True)
q10_rcp45 = rcp45_lowess.quantile(0.1, dim="ens", skipna=True)
q90_rcp45 = rcp45_lowess.quantile(0.9, dim="ens", skipna=True)
q10_rcp85 = rcp85_lowess.quantile(0.1, dim="ens", skipna=True)
q90_rcp85 = rcp85_lowess.quantile(0.9, dim="ens", skipna=True)



# quicklook
spart_lowess.plot(label = "SPARTACUS")
rcp26_lowess.plot(label = "rcp26")
rcp45_lowess.plot(label = "rcp45")
rcp85_lowess.plot(label = "rcp85")
plt.legend()



# plotting routine tmean
fig, axs = plt.subplots(figsize=(9,6.75))
#fig.patch.set_facecolor("lightgrey")
#axs.set_facecolor('#d8d8d8')

# single years as points
axs.scatter(time_spart, anomalies_areamean.values, s = 4.9, color="gray", marker = "p", label = "SPARTACUS single years")
axs.plot(time_spart, spart_lowess.values, lw = 1.4, color="black", label = "SPARTACUS 30-year LOWESS filter")

axs.plot(time_tmean, median_rcp26.values, lw = 1.4, color="royalblue", label = "RCP2.6 Ensemble Median")
axs.plot(time_tmean, median_rcp45.values, lw = 1.4, color="orange", label = "RCP4.5 Ensemble Median")
axs.plot(time_tmean, median_rcp85.values, lw = 1.4, color="magenta", label = "RCP8.5 Ensemble Median")

axs.fill_between(time_tmean, q10_rcp26, q90_rcp26, color="royalblue", alpha = 0.3, label = "RCP2.6 p10 - p90")
axs.fill_between(time_tmean, q10_rcp45, q90_rcp45, color="orange", alpha = 0.3, label = "RCP4.5 p10 - p90")
axs.fill_between(time_tmean, q10_rcp85, q90_rcp85, color="magenta", alpha = 0.3, label = "RCP8.5 p10 - p90")


axs.plot(time_spart[(time_spart >= 1991) & (time_spart <= 2020)], np.zeros(30), lw = 2, color = "red", alpha = 0.9,
         label = "Average of the period 1991-2020")

crossing_v_rcp85 = median_rcp85[median_rcp85 >= spart_lowess.max()][0].time.dt.year.values
crossing_v_rcp45 = median_rcp45[median_rcp45 >= spart_lowess.max()][0].time.dt.year.values
crossing_v_rcp26 = median_rcp26[median_rcp26 >= spart_lowess.max()][0].time.dt.year.values



plt.axvline(crossing_v_rcp85, ymax=0.433, color="magenta", lw = 1.1, ls = "--", label = str(crossing_v_rcp85))
plt.axvline(crossing_v_rcp45, ymax=0.433, color="orange", lw = 1.1, ls = "--", label = str(crossing_v_rcp45))
plt.axvline(crossing_v_rcp26, ymax=0.433, color="royalblue", lw = 1.1, ls = "--", label = str(crossing_v_rcp26))

axs.yaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_minor_locator(MultipleLocator(0.2)) # set minor tick spacing
axs.xaxis.set_major_locator(MultipleLocator(20))
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
#plt.minorticks_on()
plt.grid(True, which="major", axis = "y", linestyle = "-")
#plt.grid(True, which="major", axis = "x", linestyle = "-")
plt.grid(True, which="minor", axis = "y", linestyle = "--", alpha = 0.5)

axs.spines['left'].set_linewidth(1.5) # Set the left spine to be thicker
axs.spines['bottom'].set_linewidth(1.5) # Set the bottom spine to be thicker
#axs.spines['bottom'].set_zorder(5)
# Hide the right and top spines if desired
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')
axs.spines['bottom'].set_bounds(1971, 2097)
axs.spines['left'].set_bounds(anomalies_areamean.min().values, q90_rcp85.max().values)

plt.legend(loc = "upper left", fontsize = 11)
#plt.ylim(-20, 30)
plt.xlabel("Year", fontsize = 11, labelpad=12)
plt.ylabel("Temperature anomaly (°C)", fontsize = 12, labelpad=10)
plt.title("Annual mean temperature anomalies relative to 1991-2020\n for SPARTACUS and ÖKS15 in Austria", fontsize = 14, pad = 18)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/oeks15_vs_spartacus_lehner_special_edition.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")