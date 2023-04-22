#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""

# open and clean up lists for CMIP6 file import
infiles = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/dat/tas_models.txt", mode="rt")
files = []
for line in infiles:
    files.append(line)
files = [x.replace("\n","") for x in files]

for i, l in enumerate(files):
    if "ssp119" in l:
        print("found ya!" +l)
        del(files[i])

hist_files = []
for i, l in enumerate (files):
    if "historical" in l:
        print("found historical file. Add to other list: "+l)
        hist_files.append(l)
        del(files[i])

for i, l in enumerate(files):
    if not "Monthly" in l:
        print("Error in line "+l)
        del(files[i])

for j, l in enumerate(hist_files):
    if not "Monthly" in l:
        print("Error in line "+l)
        del(hist_files[j])

files.sort()
hist_files.sort()

outf  = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/dat/cmip6_historic.txt", mode="xt", encoding="utf-8", newline="\n")
for l in hist_files:
    outf.write(l+"\n")
outf.close()

outf  = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/dat/cmip6_models.txt", mode="xt", encoding="utf-8", newline="\n")
for l in files:
    outf.write(l+"\n")
outf.close()

#for f in hist_files:
#    try:
#        f1 = xr.open_dataset(f)
#        print("successfully opened file: "+f)
#    except(Exception):
#        print("error opening file "+f)

#for f in files:
#    try:
#        f1 = xr.open_dataset(f)
#        print("successfully opened file: "+f)
#    except(Exception):
#        print("error opening file "+f)

print("all checks completed.")