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
infiles_mask = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_regions_AT_oeks15.nc"
spart_mask = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_regions_AT_spartacus_v2.nc"
region_names = {1:"West", 2:"North",3:"South"}

indicator_searchterm = "extreme_precipitation_"
varname = "extreme_precipitation_anomalies"
v_ref = "extreme_precipitation_reference_period_1991_2020"

spart_file = "/sto0/data/Results/Indicators/extreme_precipitation_SPARTACUS_seasonal_*"
varname_spart = "extreme_precipitation"

infiles_gwls = sorted(glob.glob(infiles_gwl_ind+indicator_searchterm+"*.nc"))

mask = xr.open_dataset(infiles_mask)
mask_obs = xr.open_dataset(spart_mask)

############# ANNUAL PLOTS ##################
par_results = []
for f in infiles_gwls:
    f1 = xr.open_dataset(f)    
    for rn in region_names.keys():
        mask_region = xr.where(mask.Band1 == rn, 1, np.nan)
        #area_refperiod = (f1[v_ref] * mask_region).mean(dim=("y","x"), skipna=True).compute()
        area_sample = (f1[varname] * mask_region).mean(dim=("y","x"), skipna=True).compute()
        #area_sample = (area_sample / area_refperiod) * 100
        ars = area_sample.values.flatten()
        ars = ars[~np.isnan(ars)]
        par_results.append(ars)
            
vis_data_cm5 = [x for x in par_results]

f2 = xr.open_dataset(spart_file).sel(time=slice("1991","2020"))
vis_data_obs = []
for rn in region_names.keys():
    mask_region = xr.where(mask_obs.Band1 == rn, 1, np.nan)
    area_refperiod = (f2[varname_spart] * mask_region).mean(dim=("y","x"), skipna=True).compute()
    #area_refperiod = area_refperiod.resample(time="A").sum(dim="time", skipna=True)
    mean_refperiod = area_refperiod.mean(dim="time", skipna = True)
    anomalies_refperiod = area_refperiod - mean_refperiod
    anomalies_refperiod = anomalies_refperiod[~np.isnan(anomalies_refperiod)]
    #area_sample = (anomalies_refperiod / mean_refperiod) * 100
    vis_data_obs.append(anomalies_refperiod)


#plot for annual indicators
fig, axs = plt.subplots(figsize = (12, 6))
labels = "GWL1.5", "GWL2.0","GWL3.0","GWL4.0"
# set basecolours for boxes
basecolours = ["#3288bd", "#99d594", "#fee08b", "#fc8d59"]
obscolour = "#353535"
plt.axhline(0, color="gray")
axs.boxplot(vis_data_obs, positions=[1,2,3], widths=0.65,patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.0},
            boxprops={"facecolor": obscolour, "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": obscolour, "linewidth": 1.5},
            capprops={"color": obscolour, "linewidth": 1.5},
            flierprops={"markeredgecolor": obscolour, "linewidth": 1.0})
bp = axs.boxplot(vis_data_cm5, positions=[5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], widths=0.65,patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.0},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "C0", "linewidth": 1.0})

# set colour list for boxes and fliers
groupcolors = []
for col in basecolours:
    for _ in range(3):
        groupcolors.append(col)

# set colour list for whiskers and caps
whiskercolors = []
for col in basecolours:
    for _ in range(6):
        whiskercolors.append(col)

items = ["whiskers", "caps", "boxes", "fliers"]

#create colors for box elements
for it in items:
    if ("whisk" in it or "cap" in it):
        for bps, colors in zip( bp[it], whiskercolors):
            plt.setp(bps, color=colors)
    elif ("box" in it):
        for bps, colors in zip( bp[it], groupcolors):
            plt.setp(bps, color=colors)
    else:
        for bps, colors in zip( bp[it], groupcolors):
            plt.setp(bps, markeredgecolor=colors)


axs.tick_params(labelsize=14)
axs.set_ylabel("Change in number of hot days\ncompared to 1991-2020", fontsize = 14, labelpad=10)
axs.set_xticklabels(["West","North","South","West","North","South","West","North","South","West","North","South","West","North","South"],
                    rotation=45, fontsize=12)

axs.yaxis.set_minor_locator(MultipleLocator(5)) # set minor tick spacing
axs.yaxis.set_major_locator(MultipleLocator(10)) # set major tick spacing
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks

axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
axs.yaxis.grid(True, linestyle='--', which='minor', color='lightgrey', alpha=0.5)

#plt.ylim(-20, 30)
plt.title("Change in the annual number of hot days (>= 30 °C) in Austria\nfor different regions and global warming levels", fontsize = 16, pad = 15)
#create legend entries
pt = []
pt.append(mpatches.Patch(color=obscolour, label="Observations"))
for la, bc in zip(labels, basecolours):
    pt.append(mpatches.Patch(color=bc, label=la))
axs.legend(handles = pt, loc = "upper left", fontsize = 12)

#axs.set(axisbelow=True)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/boxplot_heatdays_regions_annual.png"
plt.savefig(outpath,dpi=300, bbox_inches ="tight")




################ SEASONAL PLOTS ##################
vis_data_cm5 = {}.fromkeys(["MAM","JJA","SON","DJF"],[])
vis_data_obs = {}.fromkeys(["MAM","JJA","SON","DJF"],[])

infiles_obs = sorted(glob.glob(spart_file))

sns = [x for x in vis_data_cm5]
for sn in sns:
    infiles = [x for x in infiles_gwls if (sn in x)]
    f_obs = infiles_obs[0]
    #f_obs = [x for x in infiles_obs if (sn in x)][0]###
    f2 = xr.open_dataset(f_obs).sel(time=slice("1991","2020"))
    pr_cur = f2[varname_spart][f2.time.dt.season == sn]
    pr_cur = pr_cur.resample(time="A", skipna=True).sum(dim="time", skipna=True)
    rgns_obs = []
    for rn in region_names.keys():
        mask_region = xr.where(mask_obs.Band1 == rn, 1, np.nan)
        area_refperiod = (pr_cur * mask_region).mean(dim=("y","x"), skipna=True).compute()
        mean_refperiod = area_refperiod.mean(dim="time", skipna = True)
        anomalies_refperiod = area_refperiod - mean_refperiod
        area_sample = (anomalies_refperiod / mean_refperiod) * 100
        area_sample = area_sample[~np.isnan(area_sample)]
        rgns_obs.append(area_sample.values)
    vis_data_obs[sn] = copy.deepcopy(rgns_obs)

    def parallel_loop(f):
        f1 = xr.open_dataset(f) 
        rgns = []   
        for rn in region_names.keys():
            mask_region = xr.where(mask.Band1 == rn, 1, np.nan)
            area_refperiod = (f1[v_ref] * mask_region).mean(dim=("y","x"), skipna=True).compute()
            area_sample = (f1[varname] * mask_region).mean(dim=("y","x"), skipna=True).compute()
            area_sample = (area_sample / area_refperiod) * 100
            ars = area_sample.values.flatten()
            ars = ars[~np.isnan(ars)]
            rgns.append(ars)
        return rgns
    if __name__ == '__main__':
        # create and configure the process pool
        with Pool(18) as pool:
            # execute tasks, block until all completed
            paralel_results = pool.map(parallel_loop, infiles)
        # process pool is closed automatically
    temp_list = []
    for tl in paralel_results:
        for i in range(3):
            temp_list.append(tl[i])
    vis_data_cm5[sn] = copy.deepcopy(temp_list)

#plot for seasonal indicators
fig, axs = plt.subplots(2,2, figsize = (12, 10), layout = 'constrained')
labels = "GWL1.5","GWL2.0","GWL3.0","GWL4.0"
# set basecolours for boxes
basecolours = ["#3288bd", "#99d594", "#fee08b", "#fc8d59"]
obscolour = "#353535"

for sn, ax in zip(sns, axs.flat):
    ax.axhline(0, color="gray")
    ax.boxplot(vis_data_obs[sn], positions=[1,2,3], widths=0.65,patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.0},
            boxprops={"facecolor": obscolour, "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": obscolour, "linewidth": 1.5},
            capprops={"color": obscolour, "linewidth": 1.5},
            flierprops={"markeredgecolor": obscolour, "linewidth": 1.0})
    bp = ax.boxplot(vis_data_cm5[sn], positions=[5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], widths=0.65,patch_artist=True,
                medianprops={"color": "white", "linewidth": 1.0},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5},
                flierprops={"color": "C0", "linewidth": 1.0})
    ax.set(title = sn)
    # set colour list for boxes and fliers
    groupcolors = []
    for col in basecolours:
        for _ in range(3):
            groupcolors.append(col)

    # set colour list for whiskers and caps
    whiskercolors = []
    for col in basecolours:
        for _ in range(6):
            whiskercolors.append(col)

    items = ["whiskers", "caps", "boxes", "fliers"]

    #create colors for box elements
    for it in items:
        if ("whisk" in it or "cap" in it):
            for bps, colors in zip( bp[it], whiskercolors):
                plt.setp(bps, color=colors)
        elif ("box" in it):
            for bps, colors in zip( bp[it], groupcolors):
                plt.setp(bps, color=colors)
        else:
            for bps, colors in zip( bp[it], groupcolors):
                plt.setp(bps, markeredgecolor=colors)


    ax.tick_params(labelsize=14)
    ax.set_xticklabels(["West","North","South","West","North","South","West","North","South","West","North","South","West","North","South"],
                        rotation=45, fontsize=12)

    ax.yaxis.set_minor_locator(MultipleLocator(5)) # set minor tick spacing
    ax.yaxis.set_major_locator(MultipleLocator(10)) # set major tick spacing
    ax.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks

    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', which='minor', color='lightgrey', alpha=0.5)
    #ax.set(axisbelow=True)
    ax.set_ylim(-20,50)
fig.supylabel("Change in daily precipitation extremes (%) compared to 1991-2020", fontsize = 14)
fig.suptitle("Change in seasonal extremes in daily precipitation in Austria\nfor different regions and global warming levels", fontsize = 16, x = 0.5)
#create legend entries
fig.get_layout_engine().set(h_pad = 0.11, w_pad = 0.11)

pt = []
pt.append(mpatches.Patch(color=obscolour, label="Observations"))
for la, bc in zip(labels, basecolours):
    pt.append(mpatches.Patch(color=bc, label=la))
ax.legend(handles = pt, loc = "upper left", fontsize = 12)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/boxplot_extreme_precip_regions_seasonal.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")