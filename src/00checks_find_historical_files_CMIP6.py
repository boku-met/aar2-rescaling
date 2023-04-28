
import glob
import xarray as xr

ssps = ["ssp126", "ssp245", "ssp370", "ssp585", "ssp119"]
for ssp in ssps:
    infiles = sorted(glob.glob("/hpx/Bennib/CMIP6_data_temp/Projection/tas_Amon_*"+ssp+"*.nc"))
    for f in infiles:
        file_hist = f.split("/")[-1][:-16]
        file_hist = file_hist.replace(ssp, "historical")
        try:
            fin = glob.glob("/hpx/Bennib/CMIP6_data_temp/Historic/"+file_hist+"*.nc")
            f1 = xr.open_dataset(fin[0])
            print("success! Historical file {0} found".format(file_hist))
        except Exception:
            print("file {0} for model file {1} not found.".format(file_hist, f))