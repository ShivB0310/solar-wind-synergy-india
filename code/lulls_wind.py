# ----------------------------
# Import Libraries
# ----------------------------
import pandas as pd  # Not used in current implementation
import xarray as xr  # Core library for handling NetCDF data
import math  # Not used in current implementation
import numpy as np  # Numerical operations
import gc  # Garbage collection
import ctypes  # Low-level memory management
from dask.distributed import Client  # Parallel processing framework
import multiprocessing  # Process management

# ----------------------------
# Memory Management Functions
# ----------------------------
def trim_memory() -> int:
    """Releases unused memory using Linux's malloc_trim system call"""
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def setup_dask_client():
    """Configures Dask cluster with 64 workers and 128GB memory limit"""
    return Client(n_workers=64, threads_per_worker=2, memory_limit='128GB')

# ----------------------------
# Main Processing Block
# ----------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows compatibility safeguard
    
    # Initialize parallel processing cluster
    client = setup_dask_client()
    
    # Memory optimization steps
    client.run(gc.collect)  # Force garbage collection on all workers
    client.run(trim_memory)  # Release unused memory

    # ----------------------------
    # Data Loading & Processing
    # ----------------------------
    print("Loading Data")
    # Load wind data with spatial chunking for parallel processing
    mds_wind = xr.open_mfdataset(
        '/scratch/vishald/jrf2_monsoonlab/shiv/download_data/Wind_*.nc',
        combine='nested',
        concat_dim='time',
        parallel=True,
        chunks={'longitude': 100, 'latitude': 100}
    )
    print("Dataset Loaded")

    # Calculate wind power density (WPD)
    u = mds_wind['u100']  # Eastward wind component at 100m height
    v = mds_wind['v100']  # Northward wind component at 100m height
    speed = (u*u + v*v)**0.5  # Wind speed magnitude
    wpd_all = 0.5*1.2*(speed**3)  # Wind power density formula (1.2 = air density)
    print("Data Processed")

    # Select 20-year period (1980-1999)
    selected_data = wpd_all.sel(time=slice('1980-01-01', '1999-12-31'))

    # ----------------------------
    # Lull Detection Logic
    # ----------------------------
    WIND_LULL_THRESHOLD = 240  # W/m² threshold for wind power lulls

    def lulls_wind(mds_wind_array):
        """
        Calculates mean duration of low-wind periods (lulls) below 240 W/m²
        Input: Time series array of wind power density values
        Output: Mean lull duration in hours (NaN if no lulls)
        """
        # Track transitions between lull/non-lull states
        groups = accumulate([0] + [(a >= 240) != (b >= 240) 
                                 for a,b in zip(mds_wind_array, mds_wind_array[1:])])
        
        # Count durations of each state
        counts = sorted(Counter(groups).items())
        
        # Extract lull durations (periods below threshold)
        below = [c for n,c in counts if (n%2 == 0) != (mds_wind_array[0] >= 240)]
        
        return np.nanmean(below) if below else np.nan

    # ----------------------------
    # Parallel Computation
    # ----------------------------
    # Apply calculation across all grid points
    plot_lull_wind = xr.apply_ufunc(
        lulls_wind,
        selected_data,
        input_core_dims=[['time']],  # Process along time dimension
        output_core_dims=[[]],  # Scalar output per location
        vectorize=True,  # Enable NumPy vectorization
        output_dtypes=[float],
        dask='parallelized',  # Distribute computation across Dask cluster
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    # ----------------------------
    # Save Results
    # ----------------------------
    print("Calculation Started")
    plot_lull_wind = plot_lull_wind.compute()  # Trigger actual computation
    plot_lull_wind.to_netcdf(
        '/scratch/vishald/jrf2_monsoonlab/shiv/datastats/Final_Twenty_1980_1999_Wind_Lulls.nc'
    )
    print("Lull Complete")
