# ----------------------------
# Import Libraries
# ----------------------------
import pandas as pd  # Not used in this script
import xarray as xr  # For handling NetCDF data
import math  # Not used in this script
import numpy as np  # Numerical operations
import gc  # Garbage collection
import ctypes  # Low-level memory management
from dask.distributed import Client  # Parallel processing
import multiprocessing  # Process management

# ----------------------------
# Memory Management Functions
# ----------------------------
def trim_memory() -> int:
    """Release unused memory using Linux's malloc_trim"""
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def setup_dask_client():
    """Initialize Dask cluster with 64 workers and 128GB memory limit"""
    return Client(n_workers=64, threads_per_worker=2, memory_limit='128GB')

# ----------------------------
# Main Processing Block
# ----------------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows compatibility
    
    # Initialize Dask client and clean memory
    client = setup_dask_client()
    client.run(gc.collect)  # Worker garbage collection
    client.run(trim_memory)  # Memory optimization

    # ----------------------------
    # Data Loading & Preprocessing
    # ----------------------------
    print("Loading Data")
    # Load solar radiation data with spatial chunking
    mds_solar = xr.open_mfdataset(
        '/scratch/vishald/jrf2_monsoonlab/shiv/download_data/Solar_*.nc',
        combine='nested',
        concat_dim='time',
        parallel=True,
        chunks={'latitude':200, 'longitude':200}
    )
    
    # Extract surface solar radiation (ssrd) and convert J/m² to W/m²
    mds_solar = mds_solar['ssrd'] / 3600
    
    # Filter out nighttime/low-radiation values (<10 W/m²)
    day_threshold = 10
    mds_solar = mds_solar.where(mds_solar >= day_threshold)
    
    # Select 1980-1999 timeframe
    selected_data = mds_solar.sel(time=slice('1980-01-01', '1999-12-31'))

    # ----------------------------
    # Lull Period Calculation
    # ----------------------------
    solar_threshold = 170  # W/m² threshold for lull detection
    
    def lulls_solar(mds_solar_array):
        """
        Calculate mean duration of solar radiation lulls below 170 W/m²
        Input: Time series array of solar radiation values
        Output: Mean lull duration (hours)
        """
        arr = mds_solar_array.copy()
        
        # 1. Mask nighttime values (already <10 W/m² from preprocessing)
        arr[arr < 10] = 0
        
        # 2. Separate daytime/nighttime values
        daytime_values = [num for num in arr if num != 0]
        nighttime_values = [num for num in arr if num == 0]
        arr = np.array(daytime_values + nighttime_values)  # Concatenate
        
        # 3. Identify transitions between lull/non-lull periods
        groups = accumulate([0] + [(a >= 170) != (b >= 170) 
                                 for a, b in zip(arr[:-1], arr[1:])])
        
        # 4. Count lull durations
        counts = sorted(Counter(groups).items())
        below = [c for n, c in counts if (n % 2 == 0) != (arr[0] >= 170)]
        
        return np.nanmean(below) if below else np.nan

    # ----------------------------
    # Parallel Execution
    # ----------------------------
    # Apply function across all grid cells
    plot_lull_solar = xr.apply_ufunc(
        lulls_solar,
        selected_data,
        input_core_dims=[['time']],  # Process time dimension
        output_core_dims=[[]],  # Scalar output per location
        vectorize=True,
        output_dtypes=[float],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    # ----------------------------
    # Save Results
    # ----------------------------
    print("Calculation Started")
    plot_lull_solar = plot_lull_solar.compute()  # Trigger actual computation
    plot_lull_solar.to_netcdf(
        '/scratch/vishald/jrf2_monsoonlab/shiv/datastats/Final_Twenty_1980_1999_Solar_Lulls.nc'
    )
    print("Lull Complete")
