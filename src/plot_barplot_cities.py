import os
import glob
import copy
from multiprocessing.pool import Pool
import matplotlib.patches as mpatches
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass


gwls = [1.5, 2.0, 3.0, 4.0]

infiles_gwl_ind = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
infiles_mask = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_hauptstaedte_oeks15.nc"
spart_mask = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_hauptstaedte_spartacus_v2.nc"

region_names = {1:"Eisenstadt", 2:"Klagenfurt",3:"St. Pölten", 4:"Linz", 5:"Salzburg",6:"Graz", 7:"Innsbruck", 8:"Bregenz",9:"Vienna"}

seasons = ["DJF", "MAM", "JJA", "SON"]

indicator_searchterm = "heatdays_30_CMIP5_GWL_"
varname = "heatdays_30"

spart_file = "/sto0/data/Results/Indicators/Heatdays_SPARTACUS_annual_1961-2021.nc"
varname_spart = "heatdays_30"

infiles_gwls = sorted(glob.glob(infiles_gwl_ind+indicator_searchterm+"*.nc"))

mask = xr.open_dataset(infiles_mask)
mask_obs = xr.open_dataset(spart_mask)


gwls_results = []
for f in infiles_gwls[1:]:
    f1 = xr.open_dataset(f)  
    f_ref = xr.open_dataset(infiles_gwls[0])  
    par_results = []
    for rn in region_names.keys():
        mask_region = xr.where(mask.Band1 == rn, 1, np.nan)

        area_refperiod = (f_ref[varname] * mask_region).mean(dim=("y","x"), skipna=True)
        refperiod = area_refperiod.mean(dim="time", skipna=True).compute()
        
        area_sample = (f1[varname] * mask_region).mean(dim=("y","x"), skipna=True)
        area_sample = area_sample.mean(dim="time", skipna=True).compute()        

        anomalies = area_sample - refperiod
        anomalies = anomalies.median(dim="ens", skipna=True)

        par_results.append(anomalies)
    gwls_results.append(par_results)

vis_data_gwl = xr.DataArray([x for x in gwls_results], 
                            coords={"gwls": [str(x) for x in gwls], 
                                    "cities": [x for x in region_names.values()]})


f2 = xr.open_dataset(spart_file)
tstart = ["1961", "1991", "2001"]
tend = ["1990","2020","2020"]
periods_results = []
for ts, te in zip(tstart, tend):
    ds_hist = f2[varname_spart].sel(time=slice(ts, te))
    par_results = []
    for rn in region_names.keys():
        mask_region = xr.where(mask_obs.Band1 == rn, 1, np.nan)

        area_refperiod = (ds_hist * mask_region).mean(dim=("y","x"), skipna=True).compute()
        mean_refperiod = area_refperiod.mean(dim="time", skipna = True).compute()

        par_results.append(mean_refperiod)
    periods_results.append(par_results)

vis_data_obs = xr.DataArray([x for x in periods_results], 
                            coords={"gwls": [str(x) for x in tstart], 
                                    "cities": [x for x in region_names.values()]})

gwls_abs = vis_data_obs[-1,:] + vis_data_gwl
vis_data_final = xr.concat((vis_data_obs, gwls_abs), dim="gwls")
vis_data_final = vis_data_final.drop_sel(gwls=["2001","1.5", "3.0"])


fig, axs = plt.subplots(figsize = (6.88, 6.88/2))
# set basecolours for boxes
basecolours = ["#fec472", "#ff7461", "#f04d52", "#8e3e73"]
periods = ["1961-1990","1991-2020","GWL 2.0 °C","GWL 4.0 °C"]

axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=1, zorder = 0)
#axs.yaxis.grid(True, linestyle='--', which='minor', color='lightgrey', alpha=0.5)

x = np.arange(len(vis_data_final.cities))  # the label locations
width = 1/5  # the width of the bars
multiplier = 0

for period, cols, pnames in zip(vis_data_final.gwls, basecolours, periods):
    data = vis_data_final.sel(gwls = period).values
    offset = width * multiplier
    rects = axs.bar(x + offset, data, width=width, label=pnames, color = cols, zorder = 2)
    multiplier += 1


plt.title("Hot days: Past and possible future scenarios", fontsize = 10, weight = "bold", pad = 30, x = 0.6)
fig.text(y= 0.93, x=0.4, s="Average number of hot days per year", fontsize = 10)

axs.set_xticks(x + width+0.1, vis_data_final.cities.values, rotation=45, fontsize = 9, weight = "bold")
plt.yticks(fontweight="bold")

#axs.yaxis.set_minor_locator(MultipleLocator(0.5)) # set minor tick spacing
axs.yaxis.set_major_locator(MultipleLocator(10)) # set major tick spacing
#axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks

#plt.ylim(-20, 30)
#axs.set(axisbelow=True)

axs.legend(loc = (1.03, 0.15), ncols = 1, fontsize = 9, frameon = False)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/barplot_hotdays_cities_gwls_changes.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/barplot_hotdays_cities_gwls_changes.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")

