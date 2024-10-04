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
path_to_indicators_cm6 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP6/"
path_mask_spartacus = "/sto0/data/Input/Gridded/SPARTACUS/V2.1/TN/TN_19610101_20221031.nc"

path_to_obs = "/sto0/data/Results/Indicators/extreme_precipitation_SPARTACUS_seasonal_*"
lookup_name = "extreme_precipitation_*"
searchterm_indicators = "extreme_precipitation"
searchterm_refperiod = "extreme_precipitation"

# for defining the point in the time period
quantile = 0.5

infiles_cm5 = sorted(glob.glob(path_to_indicators_cm5+lookup_name+"*.nc"))
infiles_cm6 = sorted(glob.glob(path_to_indicators_cm6+lookup_name+"*.nc"))

# if needed, load reference period
f_mask_spart = xr.open_dataset(path_mask_spartacus)
mask_spart = xr.where(f_mask_spart.mask == 1, 1, np.nan)

# check if season is calculated correctly
f_ref_period = xr.open_dataset(path_to_obs).sel(time=slice("1991","2020"))
#f_ref_period = f_ref_period.resample(time="A", skipna=True).sum(dim="time", skipna=True)
vis_data_refperiod = f_ref_period[searchterm_refperiod].mean(dim="time", skipna=True)
vis_data_refperiod = (vis_data_refperiod * mask_spart).compute()

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    time_sample = f1[searchterm_indicators+"_anomalies"].mean(skipna=True, dim="time")
    ref_period = f1[searchterm_indicators+"_reference_period_1991_2020"]
    percent = (time_sample / ref_period) * 100
    ensemble_mean = percent.quantile(0.5, dim="ens", skipna=True)
    return ensemble_mean.values
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(18) as pool:
        # execute tasks, block until all completed
        par_results = pool.map(paralell_loop, infiles_cm5)
    # process pool is closed automatically

f1 = xr.open_dataset(infiles_cm5[0])
vis_data_cm5 = xr.DataArray(par_results, coords={"GWL":["1.5","2.0", "3.0","4.0"], "y": f1.y, "x": f1.x})

#simple quicklook
g = vis_data_cm5.plot(x = "x", y = "y", col="GWL", col_wrap=2, aspect = 1.9, size = 3, 
                  cmap="Oranges", cbar_kwargs={"label": "Cooling degree days"}, levels = 9)
#plt.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_cdd_gwl.png", dpi=300)

glolv = "4.0"

vis_data_cm5.sel(GWL=glolv).plot.hist(label = "4.0")
vis_data_cm5.sel(GWL="1.5").plot.hist(label = "1.5")
plt.legend()
vis_data_refperiod.plot.hist()

# create colors for colorbar
lvls = 12
cmap = plt.cm.BrBG.resampled(lvls)
lst = cmap(np.linspace(0,1,lvls))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
custom_clrs = custom_clrs[3:]

# create colors for reference colorbar
lvls_refcbar = 8
cmap_refcbar = plt.cm.YlGnBu.resampled(lvls_refcbar)
lst = cmap_refcbar(np.linspace(0,1,lvls_refcbar))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs_refcbar = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
#custom_clrs_refcbar = custom_clrs_refcbar[3:]

#setting colorbar ticks and labels
level = [-6, -2, 0, 2, 4, 6, 8, 10, 12, 14]
custom_ticks = [-2, 0, 2, 4, 6, 8, 10, 12, 14]
tick_labels = ["-2", "0", "2", "4", "6", "8", "10", "12", "14"]

level_refcbar = [200, 500, 750, 1000, 1250, 1500, 1750, 2000]
custom_ticks_refcbar = [500, 750, 1000, 1250, 1500, 1750, 2000]
tick_labels_refcbar = ["-2", "0", "2", "4", "6", "8", "10", "12", "14"]

# setting projections
proj = ccrs.epsg(31287)
gridcrs = ccrs.Geodetic()

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

# start plotting the figure
fig, axs = plt.subplots(ncols=2, nrows=2, 
                            subplot_kw=dict(projection=proj), 
                            figsize = (12,7), 
                            layout = 'constrained')
for gwl, axis in zip(vis_data_cm5.coords["GWL"].values, axs.flat):
    im = vis_data_cm5.sel(GWL=gwl).plot.imshow(ax = axis, add_colorbar = False, colors = custom_clrs, levels = level)
    gl = axis.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                        xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
    gl.top_labels = False
    gl.right_labels = False
    axis.set(ylabel = "lat", xlabel = "lon", title = "GWL"+gwl)
    axis.add_feature(cfeature.BORDERS)
    axis.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_cmip5[gwl]), 
                 style='italic', bbox={'facecolor': 'white'})
    
    
fig.suptitle("Change in annual precipitation sums\nin Austria, compared to 1991-2020", size = 16, x = 0.5)
cax, kwar = mpl.colorbar.make_axes(parents=axs, location="right", fraction=0.17, pad = 0.1)
fig.delaxes(cax)
cbar = fig.colorbar(mappable=im, ax=cax, spacing = 'uniform', **kwar)
cbar.ax.set_ylabel("Precipitation change (%)")#, size=12)
cbar.set_ticks(ticks=custom_ticks, labels=tick_labels)
fig.get_layout_engine().set(h_pad = 0.05, w_pad = 0.11)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/maps_pr_gwls_changes.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")


# Single plots 
fig, axs = plt.subplots(subplot_kw=dict(projection=proj), layout = 'constrained', figsize = (8,4))
im = vis_data_refperiod.plot.imshow(colors = custom_clrs_refcbar, levels = level_refcbar, add_colorbar = False)
cbar = fig.colorbar(im, spacing =  'uniform')
cbar.ax.set_ylabel("SPEI (sd)")#, size=12)
cbar.set_ticks(ticks=custom_ticks_refcbar, labels=tick_labels_refcbar)

gl = axs.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                        xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
gl.top_labels = False
gl.right_labels = False
axs.set(ylabel = "lat", xlabel = "lon")
axs.add_feature(cfeature.BORDERS)
axs.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_refperiod), 
                 style='italic', bbox={'facecolor': 'white'})

plt.title("Annual precipitation sums\nin Austria for the reference period 1991-2020", size = 13, x = 0.54)
fig.get_layout_engine().set(h_pad = 0.11, w_pad = 0.11)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/maps_pr_obs_reference_period.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")


# Single plots, seasonal from monthly data

f_mask_spart = xr.open_dataset(path_mask_spartacus)
mask_spart = xr.where(f_mask_spart.mask == 1, 1, np.nan)

# check if season is calculated correctly
seasons = ["DJF", "JJA", "SON", "MAM"]

# create colors for reference colorbar
lvls_refcbar = 7
cmap_refcbar = plt.cm.PuBuGn.resampled(lvls_refcbar)
lst = cmap_refcbar(np.linspace(0,1,lvls_refcbar))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs_refcbar = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
#custom_clrs_refcbar = custom_clrs_refcbar[3:]

#setting colorbar ticks and labels
level_refcbar = [-20, 20, 40, 60, 80, 100, 120]
custom_ticks_refcbar = [20, 40, 60, 80, 100, 120]


# setting map x and y labels
xtickspacing = 100000
ytickspacing = 100000
xoffset = 12500
yoffset = 58500 
xticks = np.arange(f_ref_period.x.min()-xoffset+xtickspacing, f_ref_period.x.max(), xtickspacing)
yticks = np.arange(f_ref_period.y.min()-yoffset+ytickspacing, f_ref_period.y.max(), ytickspacing)

# lat and long ticks
#lats = np.arange(46, 50, 0.5)
#lons = np.arange(9, 18, 0.5)

# setting projection
proj = ccrs.epsg(31287)

# for single obs files
#f_ref_period = xr.open_dataset(path_to_obs).sel(time=slice("1991","2020"))
for sn in seasons:
    # for seasonally split obs files
    infiles_obs = sorted(glob.glob(path_to_obs+sn+"*.nc"))
    f_ref_period = xr.open_dataset(infiles_obs[0]).sel(time=slice("1991","2020"))

    #curind = f_ref_period.time.dt.season == sn

    cur_pr = f_ref_period[searchterm_refperiod]#.resample(time="A", skipna = True).sum(dim="time", skipna = True)
    vis_data_refperiod = cur_pr.mean(dim="time", skipna=True)
    vis_data_refperiod = (vis_data_refperiod * mask_spart).compute()

    values_refperiod = (vis_data_refperiod.min().values.round(1), 
                        vis_data_refperiod.mean(skipna=True).values.round(1), 
                        vis_data_refperiod.max().values.round(1))
    values_refperiod = [str(x) for x in values_refperiod]


    fig, axs = plt.subplots(subplot_kw=dict(projection=proj), layout = 'constrained', figsize = (8,4))
    im = vis_data_refperiod.plot.imshow(colors = custom_clrs_refcbar, levels = level_refcbar, add_colorbar = False)
    cbar = fig.colorbar(im, spacing =  'uniform')
    cbar.ax.set_ylabel("SPEI (sd)")#, size=12)
    cbar.set_ticks(ticks=custom_ticks_refcbar, labels=tick_labels_refcbar)
    gl = axis.gridlines(transform = gridcrs, draw_labels=True, dms=False, 
                        xlocs = MultipleLocator(2), ylocs = MultipleLocator(1))
    gl.top_labels = False
    gl.right_labels = False
    axs.add_feature(cfeature.BORDERS)
    axs.set(ylabel = "lat", xlabel = "lon")
    axs.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_refperiod), 
                 style='italic', bbox={'facecolor': 'white'})

    plt.title("Mean seasonal precipitation intensities ({0})\nin Austria for the reference period 1991-2020".format(sn), size = 13, x = 0.54)
    fig.get_layout_engine().set(h_pad = 0.11, w_pad = 0.11)

    outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/maps_precip_intensity_obs_seasonal_{0}_reference_period.png".format(sn)
    plt.savefig(outpath,dpi=600, bbox_inches ="tight")