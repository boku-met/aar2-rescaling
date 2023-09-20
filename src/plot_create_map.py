#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""
import os
import glob
import copy
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import helper_functions as hf

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_to_indicators_cm5 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
lookup_name = "tas_CMI"
searchterm_indicators = "tas"
searchterm_refperiod = "tas_reference_period_1991_2020"

# for defining the point in the time period
quantile = 0.5

infiles_cm5 = sorted(glob.glob(path_to_indicators_cm5+lookup_name+"*.nc"))

# if needed, load reference period
f_ref_period = xr.open_dataset(infiles_cm5[0])
vis_data_refperiod = f_ref_period[searchterm_refperiod].quantile(0.5, dim="ens", skipna=True)

def paralell_loop(f):
    f1 = xr.open_dataset(f)
    time_sample = f1[searchterm_indicators].mean(skipna=True, dim="time")
    ensemble_mean = time_sample.quantile(0.5, dim="ens", skipna=True)
    return ensemble_mean.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles_cm5)
f1 = xr.open_dataset(infiles_cm5[0])
vis_data_cm5 = xr.DataArray(par_results, coords={"GWL":["1.5°C","2.0°C", "3.0°C","4.0°C"], "y": f1.y, "x": f1.x})

#lats = xr.DataArray(f1.lat[:,0].values, coords={"y":f1.y})
#lons = xr.DataArray(f1.lon[0,:].values, coords = {"x": f1.x})
#vis_data.coords["lat"] = lats
#vis_data.coords["lon"] = lons

#simple quicklook
g = vis_data_cm5.plot(x = "x", y = "y", col="GWL", col_wrap=2, aspect = 1.9, size = 3, 
                  cmap="Oranges", cbar_kwargs={"label": "Cooling degree days"}, levels = 9)
plt.savefig("/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_cdd_gwl.png", dpi=300)

vis_data_cm5.sel(GWL="4.0°C").plot.hist()

lvls = 9
cmap = plt.cm.YlGnBu.resampled(lvls)
lst = cmap(np.linspace(0,1,lvls))
lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
#setting colorbar ticks and labels

level = [0, 10, 15, 20, 25, 30, 35, 40, 45]
custom_ticks = [10, 15, 20, 25, 30, 35, 40, 45]
tick_labels = ["10", "15", "20", "25", "30", "35", "40", "45"]

# setting map x and y labels
xtickspacing = 100000
ytickspacing = 100000
xoffset = 11500
yoffset = 8500+67000 
xticks = np.arange(vis_data_cm5.x.min()-xoffset+xtickspacing, vis_data_cm5.x.max(), xtickspacing)
yticks = np.arange(vis_data_cm5.y.min()-yoffset+ytickspacing, vis_data_cm5.y.max(), ytickspacing)

# setting projection
proj = ccrs.epsg(31287)

fig, axs = plt.subplots(ncols=2, nrows=2, 
                            subplot_kw=dict(projection=proj), 
                            figsize = (12,7), 
                            layout = 'constrained')
for gwl, axis in zip(vis_data_cm5.coords["GWL"], axs.flat):
    im = vis_data_cm5.sel(GWL=gwl).plot.imshow(ax = axis, add_colorbar = False, colors = custom_clrs, levels = level)
    axis.gridlines(crs = proj, xlocs = xticks, ylocs = yticks)
    axis.set(yticks = yticks, ylabel = "y", xticks = xticks, xlabel = "x", title = gwl.values)
fig.suptitle("Heat days", size = 16)
cax, kwar = mpl.colorbar.make_axes(parents=axs, location="right", fraction=0.17, pad = 0.1)
fig.delaxes(cax)
cbar = fig.colorbar(mappable=im, ax=cax, label = "Number of days with tmax >= 30°C", spacing = 'uniform', **kwar)
cbar.set_ticks(ticks=custom_ticks, labels=tick_labels)
fig.get_layout_engine().set(h_pad = 0.15, w_pad = 0.15)
#outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_heatwave_intensity_gwl.png"
#plt.savefig(outpath,dpi=300, bbox_inches ="tight")


# Single plots
fig, axs = plt.subplots(subplot_kw=dict(projection=proj), layout = 'constrained', figsize = (8,4))
im = vis_data_refperiod.plot.imshow(colors = custom_clrs, levels = level, cbar_kwargs={"label": "Number of heat days", "spacing": 'uniform', "ticks": custom_ticks})
axs.gridlines(crs = proj, xlocs = xticks, ylocs = yticks)
axs.set(yticks = yticks, ylabel = "y", xticks = xticks, xlabel = "x", title = "Reference period 1991-2020")
fig.suptitle("Mean annual number of heat days (tmax >= 30°C)", size = 16)
fig.get_layout_engine().set(h_pad = 0.15, w_pad = 0.15)
outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/maps_heatdays_gwl_reference_period.png"
plt.savefig(outpath,dpi=300, bbox_inches ="tight")