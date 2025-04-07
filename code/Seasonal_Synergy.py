import pandas as pd
import xarray as xr
import math
import numpy as np
import xarray as xr
import gc
import ctypes
from dask.distributed import Client
import multiprocessing
from tqdm import tqdm
def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def setup_dask_client():
    # Starting up parallel processing
    client = Client(n_workers=16, threads_per_worker=8, memory_limit='128GB')
    # client = Client()
    return client

if __name__ == '__main__':
    multiprocessing.freeze_support()  # This line can be omitted if not freezing the script
    
    # Garbage Cleanup
    client = setup_dask_client()
    client.run(gc.collect)
    
    # Memory trimming
    client.run(trim_memory)
    ##Loading dataset
    print("Loading Data")                                                                                                                                                                                                               mds_wind=xr.open_mfdataset('/scratch/vishald/jrf2_monsoonlab/shiv/download_data/Wind_*.nc',combine='nested',concat_dim = 'time', parallel=True, chunks={'longitude': 200, 'latitude': 200})
    #mds_solar=xr.open_mfdataset('../../yearly_datadir/year*2019*.nc',preprocess=preprocess_solar,combine='nested', concat_dim='time',parallel=True,chunks={'x': 20, 'y': 20})
    mds_solar = xr.open_mfdataset('/scratch/vishald/jrf2_monsoonlab/shiv/download_data/Solar_*.nc',combine = 'nested',concat_dim = 'time', parallel=True, chunks = {'latitude':200, 'longitude':200})
    #preprocess for wind power density
    mds_solar = mds_solar['ssrd']
    mds_solar = mds_solar/3600
    day_threshold = 10
    u = mds_wind['u100']
    v = mds_wind['v100']
    speed = (u*u + v*v)**0.5
    wpd_all = 0.5*1.2*(speed**3)
    #mds_solar=xr.open_mfdataset('../../yearly_datadir/year*2019*.nc',preprocess=preprocess_solar,combine='nested', concat_dim='time',parallel=True,chunks={'x': 20, 'y': 20})
    wpd_day = wpd_all.where(mds_solar >= day_threshold )
    mds_solar = mds_solar.where( mds_solar >= day_threshold )
    mds_solar_all = mds_solar
    mds_wind_all = wpd_day
    from tqdm import tqdm
    for start_month, end_month in tqdm([[6,9],[10,11],[12,2],[3,5]]):

        
        if start_month == 12:
            mds_solar = mds_solar_all.sel(time=np.logical_or(mds_solar_all.time.dt.month >= start_month, mds_solar_all.time.dt.month <= end_month))
        else:
            mds_solar = mds_solar_all.sel(time=np.logical_and(mds_solar_all.time.dt.month >= start_month, mds_solar_all.time.dt.month <= end_month))
        if start_month == 12:
            mds_wind = mds_wind_all.sel(time=np.logical_or(mds_wind_all.time.dt.month >= start_month, mds_wind_all.time.dt.month <= end_month))
        else:
            mds_wind = mds_wind_all.sel(time=np.logical_and(mds_wind_all.time.dt.month >= start_month, mds_wind_all.time.dt.month <= end_month))   

        solar_th = 170
        wind_th = 240
        day_time_hours = mds_solar.count(dim='time').load()
        wss_bool = ((mds_solar > solar_th) & (mds_wind <= wind_th)) | ((mds_solar <= solar_th) & (mds_wind > wind_th))
        wss = (mds_solar.where(wss_bool)).count(dim='time').load()
        wss = (wss/day_time_hours)*100
        wss.to_netcdf(f'/scratch/deleted_monthly/shiv/datastats/{start_month}-{end_month} WSS.nc')
