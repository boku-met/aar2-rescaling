#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:56:57 2023

@author: bbecsi
"""

import glob
import numpy as np
import xarray as xr

infiles = sorted(glob.glob("/hpx/"))

# start with concatting model data, maybe even in own script before this