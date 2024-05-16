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
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_to_indicators_cm5 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"
path_to_indicators_cm6 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP6/"
path_mask_spartacus = "/hp8/Projekte_Benni/Fluvial_processes/data/masks/mask_AT_spartacus_v2.nc"

path_to_obs = "/hp8/Projekte_Benni/Temp_Data/Indicators/pr_SPARTACUS_annual_1961-2021.nc"
lookup_name = "pr_CMIP5_GWL_"
searchterm_indicators = "pr"
searchterm_refperiod = "pr"

# for defining the point in the time period
quantile = 0.5

infiles_cm5 = sorted(glob.glob(path_to_indicators_cm5+lookup_name+"*.nc"))
infiles_cm6 = sorted(glob.glob(path_to_indicators_cm6+lookup_name+"*.nc"))

# if needed, load reference period
f_mask_spart = xr.open_dataset(path_mask_spartacus)
mask_spart = xr.where(f_mask_spart.Band1 == 1, 1, np.nan)

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

#lats = xr.DataArray(f1.lat[:,0].values, coords={"y":f1.y})
#lons = xr.DataArray(f1.lon[0,:].values, coords = {"x": f1.x})
#vis_data.coords["lat"] = lats
#vis_data.coords["lon"] = lons

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
lvls = 16
cmap = plt.cm.BrBG.resampled(lvls)
lst = cmap(np.linspace(0,1,lvls))
#lst = np.insert(lst, 0,[0.8,0.8,0.8,1], axis = 0) # insert grey at position 0
custom_clrs = [mpl.colors.to_hex(x, keep_alpha=True) for x in lst]
custom_clrs = custom_clrs[6:]

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


# setting map x and y labels
xtickspacing = 100000
ytickspacing = 100000
xoffset = 11500
yoffset = 8500+67000 
xticks = np.arange(vis_data_cm5.x.min()-xoffset+xtickspacing, vis_data_cm5.x.max(), xtickspacing)
yticks = np.arange(vis_data_cm5.y.min()-yoffset+ytickspacing, vis_data_cm5.y.max(), ytickspacing)

# setting projection
proj = ccrs.epsg(31287)

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

fig, axs = plt.subplots(ncols=2, nrows=2, 
                            subplot_kw=dict(projection=proj), 
                            figsize = (12,7), 
                            layout = 'constrained')
for gwl, axis in zip(vis_data_cm5.coords["GWL"].values, axs.flat):
    im = vis_data_cm5.sel(GWL=gwl).plot.imshow(ax = axis, add_colorbar = False, colors = custom_clrs, levels = level)
    axis.gridlines(crs = proj, xlocs = xticks, ylocs = yticks)
    axis.set(yticks = yticks, ylabel = "y", xticks = xticks, xlabel = "x", title = "GWL"+gwl)
    axis.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_cmip5[gwl]), 
                 style='italic', bbox={'facecolor': 'white'})
    
fig.suptitle("Change in annual precipitation sums\nin Austria, compared to 1991-2020", size = 16, x = 0.5)
cax, kwar = mpl.colorbar.make_axes(parents=axs, location="right", fraction=0.17, pad = 0.1)
fig.delaxes(cax)
cbar = fig.colorbar(mappable=im, ax=cax, label = "Precipitation change (%)", spacing = 'uniform', **kwar)
cbar.set_ticks(ticks=custom_ticks, labels=tick_labels)
fig.get_layout_engine().set(h_pad = 0.05, w_pad = 0.11)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/maps_pr_gwls_changes.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")


# Single plots 
fig, axs = plt.subplots(subplot_kw=dict(projection=proj), layout = 'constrained', figsize = (8,4))
im = vis_data_refperiod.plot.imshow(colors = custom_clrs_refcbar, levels = level_refcbar, 
                                    cbar_kwargs={"label": "Precipitation (mm)", "spacing": 'uniform', "ticks": custom_ticks_refcbar})
axs.gridlines(crs = proj, xlocs = xticks, ylocs = yticks)
axs.set(yticks = yticks, ylabel = "y", xticks = xticks, xlabel = "x")
axs.text(180000,450000, "Min: {0}\nMean: {1}\nMax: {2}".format(*values_refperiod), 
                 style='italic', bbox={'facecolor': 'white'})

fig.suptitle("Annual precipitation sums\nin Austria for the reference period 1991-2020", size = 13, x = 0.54)
fig.get_layout_engine().set(h_pad = 0.11, w_pad = 0.11)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/SOD_plots/maps_pr_obs_reference_period.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")