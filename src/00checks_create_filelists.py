#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""

# open and clean up lists for CMIP6 file import
infiles = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/all_tas_models.txt", mode="rt")

files = [x.replace("\n","") for x in infiles]
tmp_files = [x for x in files if not "ssp119" in x and "Monthly" in x]

hist_files = sorted([x for x in tmp_files if "historical" in x])
model_files = sorted([x for x in tmp_files if "ssp" in x])

outf  = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/tas_cmip6_historic.txt", mode="xt", encoding="utf-8", newline="\n")
for l in hist_files:
    outf.write(l+"\n")
outf.close()

outf  = open(file="/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/tas_cmip6_models.txt", mode="xt", encoding="utf-8", newline="\n")
for l in files:
    outf.write(l+"\n")
outf.close()