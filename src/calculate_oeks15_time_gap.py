
import os
import glob
import copy
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

infiles_spart_tn = sorted(glob.glob("/sto0/data/Input/Gridded/SPARTACUS/V2.1/TN/*.nc"))
infiles_spart_tx = sorted(glob.glob("/sto0/data/Input/Gridded/SPARTACUS/V2.1/TX/*.nc"))
infiles_oeks15 = sorted(glob.glob("/nas/nas5/Projects/OEK15/tas_daily/*.nc"))

path_lookup_table = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP5_OEKS15_lookup_table.csv"

lookup_table = open(path_lookup_table, mode="rt")
lookup_table = [x.replace("\n","") for x in lookup_table]

mask_ds = xr.open_dataset(infiles_oeks15[0])
mask = xr.where(mask_ds.tas[20:70,:,:].mean(dim="time", skipna=True) >= -999, 1, np.nan)

startyears = ["1971", "1981", "1991"]
endyears = ["2000", "2010", "2020"]

for sy, ey in zip(startyears, endyears):

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
        tmean_anomalies = tmean_annual - tmean_annual.sel(time=slice(sy, ey)).mean(dim="time", skipna=True)
        tmean_anomalies = tmean_anomalies * mask
        tmean_anomalies_areamean = tmean_anomalies.mean(dim=("y", "x"), skipna=True)
        return tmean_anomalies_areamean


    if __name__ == '__main__':
        # create and configure the process pool
        with Pool(24) as pool:
            # execute tasks, block until all completed
            results_oeks = pool.map(prepare_oeks15, infiles_oeks15)


    newtime = np.arange("1971","2098", dtype=np.datetime64)
    newtime = xr.DataArray(newtime, coords={"time":newtime})

    mname_oeks = [("_").join(x.split("/")[-1].split("_")[2:6]).replace(".nc","") for x in infiles_oeks15]
    ds_oeks = xr.DataArray(results_oeks, coords={"ens": mname_oeks, "time":newtime})

    ds_spart_tn = xr.open_mfdataset(infiles_spart_tn, concat_dim="time", combine="nested")
    ds_spart_tx = xr.open_mfdataset(infiles_spart_tx, concat_dim="time", combine="nested")

    spart_tn = ds_spart_tn.TN.sel(time=slice("1971", "2023"))
    spart_tx = ds_spart_tx.TX.sel(time=slice("1971", "2023"))
    spart_tas = (spart_tn + spart_tx)/2

    spart_tas_annual = spart_tas.resample(time="YE", skipna=True).mean()
    anomalies_spart = spart_tas_annual - spart_tas_annual.sel(time=slice(sy, ey)).mean(dim="time", skipna=True)
    anomalies_areamean = anomalies_spart.mean(dim=("y","x"), skipna=True).compute()

    time_tmean = np.unique(ds_oeks.time.dt.year)
    time_spart = np.unique(anomalies_areamean.time.dt.year)

    oeks_lowess = copy.deepcopy(ds_oeks)
    for e in oeks_lowess.ens:
        cur_tmean = ds_oeks.sel(ens = e)
        oeks_lowess.loc[e,:] = lowess(cur_tmean.values, time_tmean, return_sorted=False, 
                                frac = (30 / cur_tmean.size), it = 2)
    spart_lowess = copy.deepcopy(anomalies_areamean)
    spart_lowess.values = lowess(anomalies_areamean.values, time_spart, return_sorted=False, 
                                frac = (30 / anomalies_areamean.size), it = 2)

    anomaly_current = spart_lowess.sel(time="2023").values

    overshoot = {}
    for e in oeks_lowess.ens.values:
        try:
            overshoot.update({e: oeks_lowess.time.dt.year.values[oeks_lowess.sel(ens=e).values >=  anomaly_current][0]})
        except IndexError:
            overshoot.update({e: 2120})

    print("Tmean anomalies based on {0}-{1}".format(sy, ey))
    for k in overshoot.keys():
        print(k+";"+str(overshoot[k]))

    # plotting routine tmean
    fig, axs = plt.subplots(figsize=(6.88, 6.88))
    #fig.patch.set_facecolor("lightgrey")
    #axs.set_facecolor('#d8d8d8')
    dk = [x for x in overshoot.keys()]
    dk.reverse()
    j = dk[0]
    axs.stem(j, overshoot[j], orientation="horizontal", label = "Model overshoot", ls = "")
    for k in dk[1:]:
        axs.stem(k, overshoot[k], orientation="horizontal")
    plt.axvline(2023, color="black", lw = 1.1, ls = "--", label = "2023 temperature anomaly (obs)")


    plt.xlim(2020, 2100)
    #axs.yaxis.set_major_locator(MultipleLocator(1))
    #axs.yaxis.set_minor_locator(MultipleLocator(0.2)) # set minor tick spacing
    axs.xaxis.set_major_locator(MultipleLocator(10))
    axs.xaxis.set_minor_locator(MultipleLocator(5))
    axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
    axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
    #plt.minorticks_on()
    plt.grid(True, which="major", axis = "x", linestyle = "-")
    #plt.grid(True, which="major", axis = "x", linestyle = "-")
    plt.grid(True, which="minor", axis = "x", linestyle = "--", alpha = 0.5)


    plt.legend(loc = (0.41, 0.25), fontsize = 11)

    plt.xlabel("Year", fontsize = 11, labelpad=12)
    #plt.ylabel("Temperature anomaly (°C)", fontsize = 12, labelpad=10)
    plt.title("Time gap of ÖKS15 models compared to 2023 observations\nbased on {0}-{1} mean temperature anomalies".format(sy, ey), 
    fontsize = 14, pad = 18,  x = 0.1)

    outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/model_overshoot_2023_{0}-{1}.png".format(sy, ey)
    plt.savefig(outpath,dpi=600, bbox_inches ="tight")
    print("Figure for period {0}-{1} successfully saved!".format(sy, ey))

print("Success! All period-based plots complete!")
