## ----------------------
## Parallel Processing Setup
## ----------------------
import xarray as xr
import numpy as np
import gc
import ctypes
from dask.distributed import Client
import multiprocessing

def trim_memory() -> int:
    """Release unused memory using libc's malloc_trim"""
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def setup_dask_client():
    """Initialize Dask cluster with optimized parameters"""
    return Client(n_workers=64, threads_per_worker=2, memory_limit='128GB')

## ----------------------
## Main Processing Pipeline
## ----------------------
if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows compatibility
    
    # Initialize parallel processing
    client = setup_dask_client()
    
    # Memory management
    client.run(gc.collect)
    client.run(trim_memory)

    ## ----------------------
    ## Data Loading & Preparation
    ## ----------------------
    print("Loading Data")
    # Load and chunk solar radiation data
    mds_solar = xr.open_mfdataset(
        '/scratch/Solar_*.nc',
        combine='nested',
        concat_dim='time',
        parallel=True,
        chunks={'latitude':200, 'longitude':200}
    )
    
    # Process radiation data
    solar_data = mds_solar['ssrd'] / 3600  # Convert J/m² to W/m²
    solar_data = solar_data.where(solar_data >= 10)  # Filter low values
    
    # Select temporal subset
    selected_data = solar_data.sel(time=slice('1980-01-01', '1999-12-31'))

    ## ----------------------
    ## Episode Length Calculation
    ## ----------------------
    def calculate_solar_episodes(data_array):
        """Calculate mean duration of solar radiation episodes above threshold"""
        arr = data_array.copy()
        arr[arr < 170] = 0  # Set values below episode threshold to 0
        
        # Identify consecutive periods above threshold
        groups = np.cumsum([0] + [(a >= 170) != (b >= 170) 
                                for a, b in zip(arr[:-1], arr[1:])])
        
        # Calculate episode durations
        counts = sorted(Counter(groups).items())
        above = [c for n, c in counts if (n % 2 == 0) == (arr[0] >= 170)]
        
        return np.nanmean(above) if above else np.nan

    ## ----------------------
    ## Parallel Computation
    ## ----------------------
    print("Starting Calculation")
    result = xr.apply_ufunc(
        calculate_solar_episodes,
        selected_data,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    
    ## ----------------------
    ## Save Results
    ## ----------------------
    final_output = result.compute()
    final_output.to_netcdf('/scratch/Final_Twenty_1980_1999_Solar_Episode_length.nc')
    print("Calculation Completed")
