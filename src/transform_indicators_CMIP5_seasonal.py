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

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

# user specified paths and data
gwls = [1.0, 1.5, 2.0, 3.0, 4.0]
path_indicator = "/sto0/data/Results/Indicators/"
path_lookup_table = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/gwl_lists/GWLs_CMIP5_OEKS15_lookup_table.csv"
path_outfile = "/nas/nas5/Projects/AAR2_rescaling/aar2-rescaling/data/indicators_gwl/"

searchterm_indicator = "extreme_precipitation_"
varname_indicator = "extreme_precipitation"
aggregate_method = "" # pick "mean" or "sum" to determine the method of aggregation to annual values

# please choose mask according to dataset the indicator is based on
f_mask = xr.open_dataset("/nas/nas5/Projects/OEK15/tas_daily/tas_SDM_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_CNRM-ALADIN53.nc")
mask = xr.where(f_mask.tas[0:30,:,:].mean(dim = "time", skipna = True) > -999, 1, np.nan)

#infiles = sorted(glob.glob(path_indicator+"*"+searchterm_indicator+"*.nc")) ## check correct file names
lookup_table = open(path_lookup_table, mode="rt")
lookup_table = [x.replace("\n","") for x in lookup_table]

sns = ['DJF', 'JJA', 'MAM', 'SON']
for sn in sns:
    infiles = sorted(glob.glob(path_indicator+"*"+searchterm_indicator+"*"+"seasonal_"+sn+"*.nc"))
    for i, gwl in enumerate(gwls):
        ensemble_info = []
        indicator_data = []
        ref_period = []
        anomalies = []
        for l in lookup_table:
            ll = l.split(";")
            if ll[i+1]:
                modelname = "_".join(ll[0].split("_")[1:5])
                p_start = ll[i+1].split("-")[0]
                p_end = ll[i+1].split("-")[1]
                for f in infiles:
                    if modelname in f:
                        f1 = xr.open_dataset(f)
                        if "year" in f1.dims:
                            try:
                                f1 = f1.drop_dims("time")
                                f1 = f1.rename({"year":"time"})
                                print("time variable exchanged in file {0}".format(f))
                            except ValueError:
                                None
                        curind = ((f1.time.dt.year >= int(p_start)) & (f1.time.dt.year <= int(p_end))) #& (f1.time.dt.season == sn)
                        curind_refperiod = ((f1.time.dt.year >= 1991) & (f1.time.dt.year <= 2020)) #& (f1.time.dt.season == sn)
                        ind_slice = f1[varname_indicator][curind,:,:]#.resample(time="A", skipna=True).sum()
                        ind_ref_period = f1[varname_indicator][curind_refperiod,:,:]#.resample(time="A", skipna=True).sum()
                        ind_ref_period_mean = ind_ref_period.mean(dim="time", skipna=True)
                        if ind_slice.time.size > 20:
                            if aggregate_method == "mean":
                                ind_slice = ind_slice.resample(time="A", skipna=True).mean()
                            if aggregate_method == "sum":
                                ind_slice = ind_slice.resample(time="A", skipna=True).sum()
                                ind_ref_period_annual = ind_ref_period.resample(time="A", skipna=True).sum()
                                ind_ref_period_mean = ind_ref_period_annual.mean(dim="time", skipna=True)
                            else:
                                print("""Method of aggregation for variables with higher than annual resolution not found. Please choose 'mean' or 'sum'""")
                        ind_anomaly = ind_slice - ind_ref_period_mean
                        indicator_data.append(ind_slice)
                        ref_period.append(ind_ref_period_mean)
                        anomalies.append(ind_anomaly)
                        ensemble_info.append(modelname)
        gwl_ensemble = xr.combine_nested(indicator_data, concat_dim="ens", join="override", coords='minimal')
        ref_period_ensemble = xr.combine_nested(ref_period, concat_dim="ens", join="override", coords='minimal')
        anomalies_ensemble = xr.combine_nested(anomalies, concat_dim="ens", join="override", coords='minimal')
        gwl_ensemble["ens"], ref_period_ensemble["ens"], anomalies_ensemble["ens"] = ensemble_info, ensemble_info, ensemble_info
        gwl_ensemble = gwl_ensemble * mask
        ref_period_ensemble = ref_period_ensemble * mask
        anomalies_ensemble = anomalies_ensemble * mask
        # set metadata
        gwl_ensemble = gwl_ensemble.astype(np.float32)
        ref_period_ensemble = ref_period_ensemble.astype(np.float32)
        anomalies_ensemble = anomalies_ensemble.astype(np.float32)

        gwl_ensemble.attrs, ref_period_ensemble.attrs, anomalies_ensemble.attrs = f1[varname_indicator].attrs, f1[varname_indicator].attrs, f1[varname_indicator].attrs
        encoding_dict = {'zlib': True, 'complevel': 1,'dtype': np.float32, '_FillValue': 9.96921e+36}
        gwl_ensemble.encoding, ref_period_ensemble.encoding, anomalies_ensemble.encoding = encoding_dict, encoding_dict, encoding_dict

        file_attrs = {'title': 'Ensemble for indicator <{0} ({1})>, conforming to the global warming level of {2}°C'.format(varname_indicator, sn, gwl),
                    'institution': 'Institute of Meteorology and Climatology, University of Natural Resources and Life Sciences, Vienna, Austria',
                    'source': 'ÖKS15 Austrian climate scenarios',
                    'comment': "Global warming levels are calculated from CMIP5 models. Time resolution: Annual. The time coordinate is randomly taken from the input ensemble, but is not relevant for all ensemble members",
                    'Conventions': 'CF-1.8'}
        
        fout = xr.Dataset({varname_indicator: gwl_ensemble, 
                        "{0}_reference_period_1991_2020".format(varname_indicator):ref_period_ensemble,
                            "{0}_anomalies".format(varname_indicator) : anomalies_ensemble,
                        'crs': f1.crs, # careful to check correct name!
                        "lat": f1.lat, "lon": f1.lon}, 
                        coords={"ens": ensemble_info, "time": gwl_ensemble.time, 
                                "y": gwl_ensemble.y, "x": gwl_ensemble.x},
                        attrs= file_attrs)
        outf = path_outfile+"{0}_{1}_CMIP5_GWL_{2}degC.nc".format(varname_indicator, sn, str(gwl).replace(".",""))
        if os.path.isfile(outf):
            print("File {0} already exists. Overwriting...".format(outf))
            os.remove(outf)

        fout.to_netcdf(outf)
        print("File {0} written successfully!".format(outf))
print("Writing out all files complete!")