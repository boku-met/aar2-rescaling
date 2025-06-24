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
spart_mask_path = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_AT_spartacus_v2.nc"

infiles_oeks15_rcp26 = sorted(glob.glob("/nas/nas5/Projects/OEK15/Timeseries_full_ensemble/tas/*rcp26*.nc"))
infiles_oeks15_rcp45 = sorted(glob.glob("/nas/nas5/Projects/OEK15/Timeseries_full_ensemble/tas/*rcp45*.nc"))
infiles_oeks15_rcp85 = sorted(glob.glob("/nas/nas5/Projects/OEK15/Timeseries_full_ensemble/tas/*rcp85*.nc"))
path_lookup_table = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP5_OEKS15_lookup_table.csv"

lookup_table = open(path_lookup_table, mode="rt")
lookup_table = [x.replace("\n","") for x in lookup_table]

mask_ds = xr.open_dataset(infiles_oeks15_rcp45[0])
mask = xr.where(mask_ds.tas[20:70,:,:].mean(dim="time", skipna=True) >= -999, 1, np.nan)

spart_mask_ds = xr.open_dataset(spart_mask_path)
mask_spart = xr.where(spart_mask_ds.Band1 == 1, 1, np.nan)

sy = "2001"
ey = "2020"

# dont need to do that because timeseries are available!
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
    tmean_anomalies = tmean_annual - tmean_annual.sel(time=slice("1971", "2000")).mean(dim="time", skipna=True)
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
    f1 = xr.open_dataset(file)
    modelname = "_".join(file.split("/")[-1].split("_")[2:6]).replace(".nc","")
    for l in lookup_table:
        if modelname in l:
            ll = l.split(";")
            if ll[1]:
                p_start = ll[1].split("-")[0]
                p_end = ll[1].split("-")[1]
    tmean = f1.tas.resample(time="YE", skipna=True).mean()
    refperiod = tmean.sel(time=slice(p_start, p_end)).mean(dim="time", skipna=True) #except for GWL1.0
    anomalies = tmean.sel(time=slice("1971","2097")) - refperiod
    anomalies_rcp85.append(anomalies)


newtime = np.arange("1971","2098", dtype=np.datetime64)
newtime = xr.DataArray(newtime, coords={"time":newtime})

mname_rcp26 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp26]
mname_rcp45 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp45]
mname_rcp85 = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15_rcp85]

ds_rcp26 = xr.DataArray(anomalies_rcp26, coords={"ens": mname_rcp26, "time":newtime})
ds_rcp45 = xr.DataArray(anomalies_rcp45, coords={"ens": mname_rcp45, "time":newtime})
ds_rcp85 = xr.DataArray(anomalies_rcp85, coords={"ens": mname_rcp85, "time":newtime})


# calcualte spartacus timeseries
def prepare_spart(tn, tx):  
    ds_tn = xr.open_dataset(tn)
    ds_tx = xr.open_dataset(tx)
    tmean = ((ds_tn.TN + ds_tx.TX) / 2)
    indicator_masked = tmean * mask_spart
    indicator_areamean = indicator_masked.mean(dim=("y", "x"), skipna=True)
    indicator_annual = indicator_areamean.resample(time="YE", skipna=True).mean()
    return indicator_annual
    
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(24) as pool:
        # execute tasks, block until all completed
        results_spart = pool.starmap(prepare_spart, zip(infiles_spart_tn, infiles_spart_tx))

ds_spart = xr.concat(results_spart, dim="time")

anomalies_areamean = ds_spart - ds_spart.sel(time=slice(sy, ey)).mean(dim="time", skipna=True)
anomalies_areamean = anomalies_areamean.sel(time=slice("1971","2024"))

# filter the data
time_tmean = np.unique(ds_rcp26.time.dt.year)
time_spart = np.unique(anomalies_areamean.time.dt.year)

rcp26_lowess = copy.deepcopy(ds_rcp26)
rcp45_lowess = copy.deepcopy(ds_rcp45)
rcp85_lowess = copy.deepcopy(ds_rcp85)
for e in rcp26_lowess.ens:
    cur_tmean = ds_rcp26.sel(ens = e)
    rcp26_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (41 / cur_tmean.size), it = 2)
    
for e in rcp45_lowess.ens:
    cur_tmean = ds_rcp45.sel(ens = e)
    rcp45_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (41 / cur_tmean.size), it = 2)
    
for e in rcp85_lowess.ens:
    cur_tmean = ds_rcp85.sel(ens = e)
    rcp85_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (41 / cur_tmean.size), it = 2)

spart_lowess = copy.deepcopy(anomalies_areamean)
spart_lowess.values = lowess(anomalies_areamean.values, time_spart, return_sorted=False, 
                             frac = (41 / anomalies_areamean.size), it = 2)

# or use min/max
median_rcp26 = rcp26_lowess.median(dim="ens", skipna=True)
median_rcp45 = rcp45_lowess.median(dim="ens", skipna=True)
median_rcp85 = rcp85_lowess.median(dim="ens", skipna=True)
q10_rcp26 = rcp26_lowess.min(dim="ens", skipna=True)
q90_rcp26 = rcp26_lowess.max(dim="ens", skipna=True)
q10_rcp45 = rcp45_lowess.min(dim="ens", skipna=True)
q90_rcp45 = rcp45_lowess.max(dim="ens", skipna=True)
q10_rcp85 = rcp85_lowess.min(dim="ens", skipna=True)
q90_rcp85 = rcp85_lowess.max(dim="ens", skipna=True)

# quicklook
spart_lowess.plot.line(label = "SPARTACUS", color = "black")
rcp26_lowess.plot.line(x="time", label = "rcp26")
rcp45_lowess.plot.line(x="time", label = "rcp45")
rcp85_lowess.plot.line(x="time", label = "rcp85")

plt.legend()

# plotting routine tmean
fig, axs = plt.subplots(figsize=(6.88,5.2))
#fig.patch.set_facecolor("lightgrey")
#axs.set_facecolor('#d8d8d8')

# single years as points
axs.scatter(time_spart, anomalies_areamean.values, s = 4.5, color="gray", marker = "p", label = "SPARTACUS single years")
axs.plot(time_spart, spart_lowess.values, lw = 1.2, color="black", label = "SPARTACUS LOESS-filtered (41 years)")

axs.plot(time_tmean, median_rcp26.values, lw = 1.2, color="#648FFF")
axs.plot(time_tmean, median_rcp45.values, lw = 1.2, color="#FFB000")
axs.plot(time_tmean, median_rcp85.values, lw = 1.2, color="#DC267F")

axs.fill_between(time_tmean, q10_rcp26, q90_rcp26, color="#648FFF", alpha = 0.3, label = "RCP2.6")
axs.fill_between(time_tmean, q10_rcp45, q90_rcp45, color="#FFB000", alpha = 0.3, label = "RCP4.5")
axs.fill_between(time_tmean, q10_rcp85, q90_rcp85, color="#DC267F", alpha = 0.3, label = "RCP8.5")


axs.plot(time_spart[(time_spart >= 1971) & (time_spart <= 2000)], np.zeros(30), lw = 1.4,ls="--", color = "#882255", 
         label = "Average of the period 1971-2000")

crossing_v_rcp85 = median_rcp85[median_rcp85 >= spart_lowess.max()][0].time.dt.year.values
crossing_v_rcp45 = median_rcp45[median_rcp45 >= spart_lowess.max()][0].time.dt.year.values
#crossing_v_rcp26 = median_rcp26[median_rcp26 >= spart_lowess.max()][0].time.dt.year.values

plt.axvline(crossing_v_rcp85, ymax=0.44, color="#DC267F", lw = 1.2, ls = "--", label = str(crossing_v_rcp85))
plt.axvline(crossing_v_rcp45, ymax=0.44, color="#FFB000", lw = 1.2, ls = "--", label = str(crossing_v_rcp45))
#plt.axvline(crossing_v_rcp26, ymax=0.433, color="#648FFF", lw = 1.1, ls = "--", label = str(crossing_v_rcp26))

axs.yaxis.set_major_locator(MultipleLocator(1))
axs.yaxis.set_minor_locator(MultipleLocator(0.2)) # set minor tick spacing
axs.xaxis.set_major_locator(MultipleLocator(20))
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
#axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
#plt.minorticks_on()
plt.grid(True, which="major", axis = "y", linestyle = "-")
#plt.grid(True, which="major", axis = "x", linestyle = "-")
plt.grid(True, which="minor", axis = "y", linestyle = "--", alpha = 0.5)

axs.spines['left'].set_linewidth(1.2) # Set the left spine to be thicker
axs.spines['bottom'].set_linewidth(1.2) # Set the bottom spine to be thicker
#axs.spines['bottom'].set_zorder(5)
# Hide the right and top spines if desired
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')
axs.spines['bottom'].set_bounds(1971, 2097)
axs.spines['left'].set_bounds(anomalies_areamean.min().values, q90_rcp85.max().values)

plt.legend(loc = "upper left")
#plt.ylim(-20, 30)
plt.xlabel("Year", fontsize = 11, labelpad=12)
plt.ylabel("Temperature anomaly (°C)", fontsize = 12, labelpad=11)
plt.title("Annual mean temperature anomalies relative to 1971-2000\n for SPARTACUS and ÖKS15 in Austria",
          fontsize = 12, pad = 18)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/oeks15_vs_spartacus_1971-2000.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/oeks15_vs_spartacus_1971-2000.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")


# plot for demonstrating the GWL principle. You should adapt the shown data ranges of the variables
# used above.

fig, axs = plt.subplots(figsize=(6.88,3.9))

axs.plot(time_tmean, median_rcp45.values, lw = 1.2, color="#648FFF")
axs.plot(time_tmean, median_rcp85.values, lw = 1.2, color="#DC267F")

plt.axhline(2.0,  color="grey", lw = 1.2)
plt.axhline(3.0,  color="grey", lw = 1.2)

crossing_v_rcp85_gwl2 = int(median_rcp85[median_rcp85 <= 2.0][-1].time.dt.year.values)
crossing_v_rcp85_gwl3 = int(median_rcp85[median_rcp85 <= 3.0][-1].time.dt.year.values)
crossing_v_rcp45_gwl2 = int(median_rcp45[median_rcp45 <= 2.0][-1].time.dt.year.values)

axs.annotate("", xy=(crossing_v_rcp85_gwl2,0), xytext=(int(crossing_v_rcp85_gwl2),2), arrowprops={"arrowstyle":"->", "color" : "#DC267F", "lw":1.2})
axs.annotate("", xy=(crossing_v_rcp85_gwl3,0), xytext=(int(crossing_v_rcp85_gwl3),3), arrowprops={"arrowstyle":"->", "color" : "#DC267F", "lw":1.2})
axs.annotate("", xy=(crossing_v_rcp45_gwl2,0), xytext=(int(crossing_v_rcp45_gwl2),2), arrowprops={"arrowstyle":"->", "color" : "#648FFF", "lw":1.2})

yfill1 = (0,2)
yfill2 = (0,3)

axs.fill_betweenx(yfill1, crossing_v_rcp45_gwl2 - 10, crossing_v_rcp45_gwl2 + 10, color = "#648FFF", alpha = 0.3)
axs.fill_betweenx(yfill1, crossing_v_rcp85_gwl2 - 10, crossing_v_rcp85_gwl2 + 10, color = "#DC267F", alpha = 0.3)
axs.fill_betweenx(yfill2, crossing_v_rcp85_gwl3 - 10, crossing_v_rcp85_gwl3 + 10, color = "#DC267F", alpha = 0.3)

axs.yaxis.set_major_locator(MultipleLocator(0.5))
axs.yaxis.set_minor_locator(MultipleLocator(0.2)) # set minor tick spacing
axs.xaxis.set_major_locator(MultipleLocator(10))
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
axs.tick_params(axis="x", which = "major", rotation = 45)
#axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
#plt.minorticks_on()

plt.ylim(0, 5)

annotate = []
annotate.append(mpl.lines.Line2D([],[], lw = 1.2, label="Model under RCP4.5", color = "#648FFF"))
annotate.append(mpl.lines.Line2D([],[],lw = 1.2, label="Model under RCP8.5", color = "#DC267F"))

annotate.append(mpl.patches.Patch(label = "GWL period for RCP4.5 model", color = "#648FFF", alpha = 0.3))
annotate.append(mpl.patches.Patch(label = "GWL periods for RCP8.5 model", color = "#DC267F", alpha = 0.3))

annotate.append(mpl.lines.Line2D([],[], lw = 0.9, label="Center years of GWL periods", color = "grey", marker=7, markersize=8))

plt.legend(loc = "upper left", handles = annotate)

plt.xlabel("Year", fontsize = 11, labelpad=12)
plt.ylabel("ΔGMST (°C)", fontsize = 12, labelpad=11)
plt.title("Global mean temperature change relative to 1850-1900",
          fontsize = 12, pad = 18)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/gwl_concept.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/gwl_concept.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")




# Writing out gap data for plots
infiles_all_oeks = sorted(glob.glob("/nas/nas5/Projects/OEK15/Timeseries_full_ensemble/tas/*.nc"))
anomalies_full = []
for file in infiles_all_oeks:
    # open file and load in DataArray for processing
    f1 = xr.open_dataset(file)
    modelname = "_".join(file.split("/")[-1].split("_")[2:6]).replace(".nc","")
    for l in lookup_table:
        if modelname in l:
            ll = l.split(";")
            if ll[1]:
                p_start = ll[1].split("-")[0]
                p_end = ll[1].split("-")[1]
    tmean = f1.tas.resample(time="YE", skipna=True).mean()
    refperiod = tmean.sel(time=slice(p_start, p_end)).mean(dim="time", skipna=True) #except for GWL1.0
    anomalies = tmean.sel(time=slice("1971","2097")) - refperiod
    anomalies_full.append(anomalies)

mname_all = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_all_oeks]
ds_oeks = xr.DataArray(anomalies_full, coords={"ens": mname_all, "time":newtime})

ds_spart = xr.concat(results_spart, dim="time")
anomalies_areamean = ds_spart - ds_spart.sel(time=slice(sy, ey)).mean(dim="time", skipna=True)
anomalies_areamean = anomalies_areamean.sel(time=slice("1971","2024"))

time_tmean = np.unique(ds_oeks.time.dt.year)
time_spart = np.unique(anomalies_areamean.time.dt.year)

oeks15_lowess = copy.deepcopy(ds_oeks)
for e in oeks15_lowess.ens:
    cur_tmean = ds_oeks.sel(ens = e)
    oeks15_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                             frac = (41 / cur_tmean.size), it = 2)
    
spart_lowess = copy.deepcopy(anomalies_areamean)
spart_lowess.values = lowess(anomalies_areamean.values, time_spart, return_sorted=False, 
                             frac = (41 / anomalies_areamean.size), it = 2)


for e in oeks15_lowess.ens:
    try:
        year = oeks15_lowess.time.dt.year.values[oeks15_lowess.sel(ens=e).values >= spart_lowess.sel(time="2024").values][0]
    except(IndexError):
        year = 2120
    print("{0};{1}".format(e.values, year))
    
