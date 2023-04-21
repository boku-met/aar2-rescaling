#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""

import os
import glob
import numpy as np
import xarray as xr

# open and clean up lists for file import
infiles = open(file="/metstor_nfs/projects/VTreasures/CMIP6_Data/All_tas_metstor.txt", mode="rt")
files = []
for line in infiles:
    files.append(line)
files = [x.replace("./", "/metstor_nfs/projects/VTreasures/CMIP6_Data/") for x in files]
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

for f in hist_files:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
    except(Exception):
        print("error opening file "+f)

for f in files:
    try:
        f1 = xr.open_dataset(f)
        print("successfully opened file: "+f)
    except(Exception):
        print("error opening file "+f)

print("all checks completed.")