import os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from matplotlib.lines import Line2D
import numpy as np

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

table_rcp26 = []
table_rcp45 = []
table_rcp85 = []
for i in range(1, 5):
    tabline_rcp26 = []
    tabline_rcp45 = []
    tabline_rcp85 = []
    for l in lookup_table:
        if "rcp26" in l:
            tabline_rcp26.append(int(l.split(";")[i]))
        elif "rcp45" in l:
            tabline_rcp45.append(int(l.split(";")[i]))
        elif "rcp85" in l:
            tabline_rcp85.append(int(l.split(";")[i]))
    table_rcp26.append(tabline_rcp26)
    table_rcp45.append(tabline_rcp45)
    table_rcp85.append(tabline_rcp85)


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

rcpcols = ["#648FFF", "#FFB000", "#DC267F"]
xticklabs = ["1971-2000", "1981-2010", "1991-2020", "GWL1.0"]
positions=[0, 1, 2, 3]
offset = 0.18

fig, axs = plt.subplots(figsize=(6.88/1.9, 6.88*1.5))
#fig.patch.set_facecolor("lightgrey")
axs.set_facecolor('lightgrey')

plt.axhline(2024, color="black", lw = 1.1, ls = "--", label = "Observed level (2024)")

k=0
for data, coln in zip([table_rcp26, table_rcp45, table_rcp85], rcpcols):
    axs.hlines([np.median(x) for x in data], xmin=np.array(positions)-offset, 
               xmax=np.array(positions)+offset, colors=coln, label = "Ensemble median RCP{0}".format(["2.6","4.5","8.5"][k]))
    k += 1

for j, xlabel in enumerate(xticklabs):
    for i, mname in enumerate(mnames):
        rcpn = mname.split("_")[1]
        modn = mname.replace(rcpn+"_", "")
        for coln, rcp in zip(rcpcols,["26", "45", "85"]):
            if rcp in rcpn:
                color = coln
        for k, markern in enumerate(markernames):
            if modn in markern:
                marker = rdm_markers[k]
        axs.plot(xlabel, oeks_gaps[i, j], 
                 marker = marker, markerfacecolor = color, 
                 markeredgecolor = "white",markersize = 7, mew = 0.8,ls = "",
                 label = (mname if j==0 else None))  
        


plt.ylim(2015, 2100)

axs.yaxis.set_major_locator(MultipleLocator(10))
axs.yaxis.set_minor_locator(MultipleLocator(5)) # set minor tick spacing
#axs.xaxis.set_major_locator(MultipleLocator(10))
#axs.xaxis.set_minor_locator(MultipleLocator(5))
axs.tick_params(axis="y", which = "minor", length = 0) #hide minor ticks
#axs.tick_params(size = 7, width = 1.5, labelsize=11, pad = 10)
#plt.minorticks_on()
plt.grid(True, which="major", axis = "y", linestyle = "-", color = "white")
#plt.grid(True, which="major", axis = "x", linestyle = "-")
plt.grid(True, which="minor", axis = "y", linestyle = "--", color = "white", alpha = 0.5)
axs.set_xticklabels(xticklabs)
plt.xlabel("Baseline period", fontsize = 10, labelpad=12)
plt.ylabel("Year",  fontsize = 11, labelpad=12)

plt.title("Time lag of ÖKS15 temperature anomalies vs. 2024 observations\ngrouped after baseline period", 
fontsize = 11, pad = 24, x=1.4)

plt.legend(loc = (1.15,-0.012), ncols = 1, fontsize = 10)

outpath = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/oeks15_timegap_summary.jpg"
outpath2 = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/plots/FD_plots/oeks15_timegap_summary.eps"
plt.savefig(outpath2, bbox_inches ="tight")
plt.savefig(outpath,dpi=600, bbox_inches ="tight")
