import os
import glob
import copy
import random
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

path_lookup_table = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/Gaps_OEKS15_lookup_table.csv"
lookup_table = open(path_lookup_table, mode="rt", encoding="utf-8-sig")
lookup_table = [x.replace("\n","") for x in lookup_table]

oeks_gaps = np.ndarray((len(lookup_table), 4))

for i in range(len(lookup_table)):
    for j in range(oeks_gaps.shape[1]):
        oeks_gaps[i, j] = lookup_table[i].split(";")[j+1]

oeks_gaps = oeks_gaps.astype(np.int16)

mnames = [x.split(";")[0] for x in lookup_table]
markernames = []
for rcp in ("rcp26_", "rcp45_", "rcp85_"):
    for i, l in enumerate(mnames):
        if rcp in l:
            markernames.append(l.replace(rcp, ""))
markernames = np.unique(markernames)

rdm_markers = [x for x in Line2D.filled_markers]
random.shuffle(rdm_markers)
assert(len(markernames) == len(rdm_markers))

fig, axs = plt.subplots(figsize=(9, 6.88))
#fig.patch.set_facecolor("lightgrey")
axs.set_facecolor('lightgrey')
for j, xlabel in enumerate(["1971-2000", "1981-2010", "1991-2020", "GWL1.0"]):
    for i, mname in enumerate(mnames):
        rcpn = mname.split("_")[1]
        modn = mname.replace(rcpn+"_", "")
        for coln, rcp in zip(["#003466", "#709fcc", "#980002"],["26", "45", "85"]):
            if rcp in rcpn:
                color = coln
        for k, markern in enumerate(markernames):
            if modn in markern:
                marker = rdm_markers[k]
        axs.plot(xlabel, oeks_gaps[i, j], 
                 markersize=10, marker = marker, markerfacecolor = color, 
                 markeredgecolor = "white",mew = 0.8,ls = "",
                 label = (mname if j==0 else None))

plt.axhline(2023, color="black", lw = 1.1, ls = "--")#, label = "2023 temperature anomaly (obs)")
plt.ylim(2020, 2100)

axs.yaxis.set_major_locator(MultipleLocator(10))
axs.yaxis.set_minor_locator(MultipleLocator(5)) # set minor tick spacing
#axs.xaxis.set_major_locator(MultipleLocator(10))
#axs.xaxis.set_minor_locator(MultipleLocator(5))
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
#plt.minorticks_on()
plt.grid(True, which="major", axis = "y", linestyle = "-", color = "white")
#plt.grid(True, which="major", axis = "x", linestyle = "-")
plt.grid(True, which="minor", axis = "y", linestyle = "--", color = "white", alpha = 0.5)

plt.xlabel("Baseline period", fontsize = 12, labelpad=12)
plt.ylabel("Year", fontsize = 12, labelpad=12)

plt.title("Time gap of ÖKS15 temperature anomalies vs. 2023 observations\ngrouped after baseline period", 
fontsize = 14, pad = 24)

plt.legend(loc = (-0.26,-0.98), fontsize = 11, ncols = 2)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/oeks15_timegap_summary.png"
plt.savefig(outpath,dpi=600, bbox_inches ="tight")

