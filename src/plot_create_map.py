#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
from multiprocessing.pool import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_to_indicators_cm5 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
#path_to_indicators_cm6 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP6/"
path_mask_spartacus = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/masks/mask_AT_spartacus_v2.nc"

path_to_obs = "/sto0/data/Results/Indicators/extreme_precipitation_SPARTACUS_seasonal_*"
lookup_name = "extreme_precipitation_*"
varname_proj = "extreme_precipitation"
varname_obs = "extreme_precipitation"

infiles_cm5 = sorted(glob.glob(path_to_indicators_cm5+lookup_name+"CMIP5*.nc"))
#infiles_ref_cm5 = sorted(glob.glob(path_to_indicators_cm5+lookup_name+"*single*.nc"))
#infiles_cm6 = sorted(glob.glob(path_to_indicators_cm6+lookup_name+"*.nc"))

# if needed, load reference period
f_mask_spart = xr.open_dataset(path_mask_spartacus)
mask_spart = xr.where(f_mask_spart.Band1 ==1, 1, np.nan)

# check if season is calculated correctly
f_ref_period = xr.open_dataset(path_to_obs).sel(time=slice("2001","2020"))
#f_ref_period = f_ref_period.resample(time="A", skipna=True).sum(dim="time", skipna=True)
vis_data_refperiod = f_ref_period[varname_obs].mean(dim="time", skipna=True)
# maybe you need other quantile than mean
vis_data_refperiod = (vis_data_refperiod * mask_spart).compute()

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    f_ref = xr.open_dataset(infiles_cm5[0])

    time_sample = f1[varname_proj].mean(skipna=True, dim="time")
    ref_period = f_ref[varname_proj].mean(skipna=True, dim="time")

    #percent = ((time_sample - ref_period) / ref_period) * 100
    diff = time_sample - ref_period
    ensemble_mean = diff.median(dim="ens", skipna=True)
    return ensemble_mean.values
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(18) as pool:
        # execute tasks, block until all completed
        par_results = pool.map(paralell_loop, infiles_cm5[1:]) # or use starmap and zip(arg1, arg2) to allow for multiple arguments 
    # process pool is closed automatically

f1 = xr.open_dataset(infiles_cm5[0])
vis_data_cm5 = xr.DataArray(par_results, coords={"GWL":["1.5","2.0", "3.0","4.0"], "y": f1.y, "x": f1.x})

#calculating values for annotation
gwls = vis_data_cm5.coords["GWL"].values
values_cmip5 = {}.fromkeys(gwls, [])
values_cmip6 = {}.fromkeys(gwls, [])

values_refperiod = (vis_data_refperiod.min().values.round(1), 
                    vis_data_refperiod.mean(skipna=True).values.round(1), 
                    vis_data_refperiod.max().values.round(1))


values_refperiod = [str(x) for x in values_refperiod]

for gwl in gwls: 
    values_cmip5.update({gwl:(str(vis_data_cm5.sel(GWL=gwl).min().values.round(1)), 
                    str(vis_data_cm5.sel(GWL=gwl).mean(skipna=True).values.round(1)), 
                    str(vis_data_cm5.sel(GWL=gwl).max().values.round(1)))})
    # values_cmip6.append((vis_data_cm6.sel(GWL=lv).min().values.round(1), 
    #                 vis_data_cm6.sel(GWL=lv).mean(skipna=True).values.round(1), 
    #                 vis_data_cm6.sel(GWL=lv).max().values.round(1)))



#simple quicklook
g = vis_data_cm5.plot(x = "x", y = "y", col="GWL", col_wrap=2, aspect = 1.9, size = 3, 
                  cmap="Oranges", cbar_kwargs={"label": "Cooling degree days"}, levels = 9)
#plt.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_cdd_gwl.png", dpi=300)

glolv = "4.0"

vis_data_cm5.sel(GWL=glolv).plot.hist(label = "4.0")
vis_data_cm5.sel(GWL="1.5").plot.hist(label = "1.5")
plt.legend()
vis_data_refperiod.plot.hist()

# create colors, ticks and labels for colorbar
lvls = 22 # check to define colors and levels for outside the value range
cmap = plt.cm.RdBu_r.resampled(lvls)
lst = cmap(np.linspace(0,1,lvls))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
custom_clrs = custom_clrs[11:]

level = [x for x in np.arange(0.0, 5.5, 0.5)]
#custom_ticks = [-2, 0, 2, 4, 6, 8, 10, 12, 14]
#tick_labels = ["-2", "0", "2", "4", "6", "8", "10", "12", "14"]

# optional when creating several plots with the same colour range
ccl_below = custom_clrs[:-1]
ccl_above = custom_clrs[1:]
ccl_wihtin = custom_clrs[1:-1]
ccl_both = custom_clrs

# create colors for reference colorbar
lvls_refcbar = 5
cmap_refcbar = plt.cm.YlOrRd.resampled(lvls_refcbar)
lst = cmap_refcbar(np.linspace(0,1,lvls_refcbar))
lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs_refcbar = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
#custom_clrs_refcbar = custom_clrs_refcbar[:7]

level_refcbar = [x for x in np.arange(-0.6, 0.3, 0.1)]
#custom_ticks_refcbar = [500, 750, 1000, 1250, 1500, 1750, 2000]
#tick_labels_refcbar = ["-2", "0", "2", "4", "6", "8", "10", "12", "14"]

# setting projections
proj_obs = ccrs.epsg(31287)
proj_oeks = ccrs.epsg(3416)
gridcrs = ccrs.Geodetic()

# start plotting the figure
fig, axs = plt.subplots(ncols=2, nrows=2, 
                            subplot_kw=dict(projection=proj_oeks), 
                            figsize = (6.88, 6), 
                            layout = 'constrained')
for gwl, axis in zip(vis_data_cm5.coords["GWL"].values, axs.flat):
    pltdata = vis_data_cm5.sel(GWL=gwl)
    # setcustom color bar for each individual axis
    if pltdata.min().values >= min(level):
        if pltdata.max().values  < max(level):
            custom_clrs = ccl_wihtin
        else:
            custom_clrs = ccl_above
    elif pltdata.min().values < min(level):
        if pltdata.max().values  < max(level):
            custom_clrs = ccl_below
        else:
            custom_clrs = ccl_both

    im = pltdata.plot.imshow(ax = axis, add_colorbar = False, 
                                               colors = custom_clrs, levels = level)
    gl = axis.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                        xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
                        #xlabel_style = {"fontsize":9}, ylabel_style = {"fontsize":9})
    gl.top_labels = False
    gl.right_labels = False
    axis.set(ylabel = "lat", xlabel = "lon")
    axis.axes.set_title("GWL"+gwl, {"size":10.8})
    axis.add_feature(cfeature.BORDERS)

    axis.text(145000,455000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_cmip5[gwl]), 
                 style='italic', bbox={'facecolor': 'white'})#, size = 9)
    
    
fig.suptitle("Change in SPEI of dryest summer month (Apr-Sep)\nrelative to GWL1.0", size = 11)#, x = 0.54)

#cax, kwar = mpl.colorbar.make_axes(parents=axs, location="right", fraction=0.17, pad = 0.1)
#fig.delaxes(cax)
cbar = fig.colorbar(ax= axs, mappable=im, spacing = 'uniform', orientation = "horizontal", fraction = 0.05)
cbar.ax.set_xlabel("Change in SPEI (sd)", labelpad = 10)#, size=10)
#cbar.set_ticks(ticks=custom_ticks, labels=tick_labels)
fig.get_layout_engine().set(h_pad = 0.12, w_pad = 0.12)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_SPEI_gwls_changes.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_SPEI_gwls_changes.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")



# Single plots 
fig, axs = plt.subplots(subplot_kw=dict(projection=proj_obs), layout = 'constrained', 
                        figsize = (6.88/2, 6.88))
im = vis_data_refperiod.plot.imshow(colors = custom_clrs_refcbar, levels = level_refcbar, 
                                    add_colorbar = False)
gl = axs.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                   xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
gl.top_labels = False
gl.right_labels = False
axs.set(ylabel = "lat", xlabel = "lon")
axs.add_feature(cfeature.BORDERS)

axs.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_refperiod), 
                 style='italic', bbox={'facecolor': 'white'})

cbar = fig.colorbar(im, spacing =  'uniform', extend ='both', fraction = 0.4, 
                    orientation = "horizontal")
cbar.ax.set_xlabel("SPEI (sd)")#, size=10, labelpad = 10)
#cbar.set_ticks(ticks=custom_ticks_refcbar, labels=tick_labels_refcbar)
plt.title("SPEI of dryest summer month (Apr-Sep)\nfrom observational data at GWL1.0", size = 11, pad = 16)#, x = 0.54)

#fig.get_layout_engine().set(h_pad = 4.8, w_pad = 0.1)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_SPEI_obs_gwl10.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_SPEI_obs_gwl10.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")



# Single plots, seasonal from monthly data

# check if season is calculated correctly
seasons = ["DJF", "JJA", "SON", "MAM"]

# create colors for reference colorbar
lvls_refcbar = 10
cmap_refcbar = plt.cm.Blues.resampled(lvls_refcbar)
lst = cmap_refcbar(np.linspace(0,1,lvls_refcbar))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs_refcbar = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]

ccl_below = custom_clrs_refcbar[:-1]
ccl_above = custom_clrs_refcbar[1:]
ccl_wihtin = custom_clrs_refcbar[1:-1]
ccl_both = custom_clrs_refcbar

level_refcbar = [x for x in np.arange(15, 60, 5)]
#custom_ticks_refcbar = [500, 750, 1000, 1250, 1500, 1750, 2000]
#tick_labels_refcbar = ["-2", "0", "2", "4", "6", "8", "10", "12", "14"]


# for single obs files
#f_ref_period = xr.open_dataset(path_to_obs).sel(time=slice("2001","2020"))
for sn in seasons:
    # for seasonally split obs files
    infiles_obs = sorted(glob.glob(path_to_obs+sn+"*.nc"))
    f_ref_period = xr.open_dataset(infiles_obs[0]).sel(time=slice("2001","2020"))

    #curind = f_ref_period.time.dt.season == sn

    cur_pr = f_ref_period[varname_obs]#[curind,:,:].resample(time="A", skipna = True).sum(dim="time", skipna = True)
    vis_data_refperiod = cur_pr.mean(dim="time", skipna=True)
    vis_data_refperiod = (vis_data_refperiod * mask_spart).compute()
    # only use in special cases where the border pixels create problems!
    #vis_data_refperiod = xr.where(vis_data_refperiod == 0, np.nan, vis_data_refperiod)
    
    values_refperiod = (vis_data_refperiod.min().values.round(1), 
                        vis_data_refperiod.mean(skipna=True).values.round(1), 
                        vis_data_refperiod.max().values.round(1))
    values_refperiod = [str(x) for x in values_refperiod]
    
    if vis_data_refperiod.min().values >= min(level_refcbar):
        if vis_data_refperiod.max().values  < max(level_refcbar):
            custom_clrs_refcbar = ccl_wihtin
        else:
            custom_clrs_refcbar = ccl_above
    elif vis_data_refperiod.min().values < min(level_refcbar):
        if vis_data_refperiod.max().values  < max(level_refcbar):
            custom_clrs_refcbar = ccl_below
        else:
            custom_clrs_refcbar = ccl_both
    
        

    fig, axs = plt.subplots(subplot_kw=dict(projection=proj_obs), layout = 'constrained', 
                            figsize = (6.88/2, 6.88))
    im = vis_data_refperiod.plot.imshow(colors = custom_clrs_refcbar, levels = level_refcbar, 
                                        add_colorbar = False)
    gl = axs.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                        xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
    gl.top_labels = False
    gl.right_labels = False
    axs.set(ylabel = "lat", xlabel = "lon")
    axs.add_feature(cfeature.BORDERS)

    axs.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_refperiod), 
                 style='italic', bbox={'facecolor': 'white'})

    cbar = fig.colorbar(im, spacing =  'uniform', fraction = 0.4,  
                        orientation = "horizontal")
    cbar.ax.set_xlabel("Temperature (°C)")#, size=10, labelpad = 10)
    #cbar.set_ticks(ticks=custom_ticks_refcbar, labels=tick_labels_refcbar)

    plt.title("Seasonal mean temperature ({0})\nfrom observational data at GWL1.0".format(sn), size = 11, pad = 16)#, x = 0.54)
    
    #fig.get_layout_engine().set(h_pad = 0.11, w_pad = 0.11)


    outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_tas_obs_seasonal_{0}_gwl10.jpg".format(sn)
    outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/maps_tas_obs_seasonal_{0}_gwl10.eps".format(sn)
    plt.savefig(outpath2, bbox_inches ="tight")
    plt.savefig(outpath,dpi=600, bbox_inches ="tight")
    

    # note: all plots with single subplots (maps and all boxplots) should be half page size (6.88/2)
    # for each map plot there is a single box plot with the same size
    # what to do with full-page width plots (4 maps) and single box plots? How to combine best? Maybe 2 next to each other?