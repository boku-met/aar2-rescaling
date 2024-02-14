import os
import glob
import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

xmin = 614384
xmax = 640853
ymin = 476883
ymax = 493485

gwls = [1.5, 2.0, 3.0, 4.0]
rcps = ["rcp26", "rcp45", "rcp85"]
timeslices_start = [2021, 2041, 2079]

infiles_original_ind = "/hp8/Projekte_Benni/Temp_Data/Indicators/"
infiles_gwl_ind = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/CMIP5/"

key_ids = []
for ts in timeslices_start:    
    for rcp in rcps:
        key_ids.append(rcp+"_"+str(ts+9))

data_heatdays_rcp = dict.fromkeys(key_ids, [])

for rcp in rcps:
    infiles_hd = sorted(glob.glob(infiles_original_ind+"Heatdays_*"+rcp+"*.nc"))
    for file in infiles_hd:
        f1 = xr.open_dataset(file)
        hd_sel = f1.heatdays_30.sel(x=slice(str(xmin),str(xmax)), y=slice(str(ymin), str(ymax))).mean(dim=("y","x"), skipna=True)
        for ts in timeslices_start:
            hd_ts = hd_sel.sel(time=slice(str(ts), str(ts+19)))
            key_id = rcp+"_"+str(ts+9)
            templist_hd = copy.deepcopy(data_heatdays_rcp[key_id])
            templist_hd.append([x for x in hd_ts.values])
            data_heatdays_rcp[key_id] = copy.deepcopy(templist_hd)

vis_data_rcp = [np.array(data_heatdays_rcp[x]).flatten() for x in data_heatdays_rcp]

infiles_gwls = sorted(glob.glob(infiles_gwl_ind+"heatdays_30*.nc"))
def paralell_loop(f):
    f1 = xr.open_dataset(f)    
    area_sample = f1.heatdays_30.sel(y=slice(ymin, ymax), x=slice(xmin, xmax)).mean(dim=("y","x"), skipna=True)
    return area_sample.values
par_results = Parallel(n_jobs=12)(delayed(paralell_loop)(f) for f in infiles_gwls)
vis_data_cm5 = [x.flatten() for x in par_results]

fig, axs = plt.subplots(figsize = (11, 6))
axs.boxplot(vis_data_rcp, positions=[1,2,3, 4.5,5.5,6.5, 8,9,10], widths=0.65,patch_artist=True,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "green", "linewidth": 1.0})

axs.tick_params(labelsize=14)
axs.set_ylabel("Anzahl der Hitzetage im Jahr", fontsize = 14, labelpad=10)
axs.set_xlabel("Gruppiert nach Perioden: 2030, 2050 und 2090", fontsize = 14, labelpad=15)
axs.set_xticklabels(["RCP2.6", "RCP4.5", "RCP8.5", "RCP2.6", "RCP4.5", "RCP8.5", "RCP2.6", "RCP4.5", "RCP8.5"],
                    rotation=45, fontsize=12)
axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
#axs.set(axisbelow=True)
plt.title("Anzahl der jährlichen Hitzetage in Wien, gegliedert nach Periode und RCP", fontsize = 16, pad = 15)
outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/heatdays_vienna_rcps_variance_boxplot.png"
plt.savefig(outpath,dpi=300, bbox_inches ="tight")

fig, axs = plt.subplots(figsize = (6, 6))
axs.boxplot(vis_data_cm5, positions=[1,2,3,4], widths=0.65,patch_artist=True,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
            whiskerprops={"color": "C0", "linewidth": 1.5},
            capprops={"color": "C0", "linewidth": 1.5},
            flierprops={"color": "green", "linewidth": 1.0})

axs.tick_params(labelsize=14)
axs.set_ylabel("Anzahl der Hitzetage im Jahr", fontsize = 14, labelpad=10)
axs.set_xticklabels(["GWL 1.5 °C", "GWL 2.0 °C", "GWL 3.0 °C", "GWL 4.0 °C"],
                    rotation=45, fontsize=12)
axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
#axs.set(axisbelow=True)
plt.title("Anzahl der jährlichen Hitzetage in Wien,\n gegliedert nach global warming level", fontsize = 16, pad = 15)
outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/heatdays_vienna_gwls_variance_boxplot.png"
plt.savefig(outpath,dpi=300, bbox_inches ="tight")