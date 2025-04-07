# Importing required libraries
import pandas as pd  # Unused in this script but commonly used for data manipulation
import xarray as xr  # For working with multi-dimensional arrays (e.g., NetCDF files)
import math  # Unused in this script
import numpy as np  # For numerical operations
import gc  # For garbage collection to manage memory
import ctypes  # For low-level memory management
from dask.distributed import Client  # For parallel processing with Dask
import multiprocessing  # To handle multiprocessing compatibility

# Function to release unused memory
def trim_memory() -> int:
    """
    Releases unused memory using libc's malloc_trim function.
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

# Function to set up a Dask client for parallel processing
def setup_dask_client():
    """
    Initializes a Dask distributed client with specified parameters.
    """
    client = Client(n_workers=64, threads_per_worker=2, memory_limit='128GB')  # Configures Dask workers
    return client

# Main script execution block
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Ensures compatibility when using multiprocessing on Windows
    
    # Initialize the Dask client and perform garbage collection
    client = setup_dask_client()
    client.run(gc.collect)  # Runs garbage collection on all workers
    
    # Trim unused memory across all workers
    client.run(trim_memory)

    ## ----------------------
    ## Loading Dataset
    ## ----------------------
    print("Loading Data")
    
    # Load wind data from multiple NetCDF files using xarray's open_mfdataset
    mds_wind = xr.open_mfdataset(
        '/scratch/vishald/jrf2_monsoonlab/shiv/download_data/Wind_*.nc', 
        combine='nested', 
        concat_dim='time', 
        parallel=True, 
        chunks={'longitude': 100, 'latitude': 100}  # Chunking for parallel processing
    )
    
    print("Dataset Loaded")
    
    ## ----------------------
    ## Data Preparation
    ## ----------------------
    
    # Extract wind components at 100m height and calculate wind speed and power density
    u = mds_wind['u100']  # Eastward wind component at 100m height
    v = mds_wind['v100']  # Northward wind component at 100m height
    
    # Calculate wind speed using the Pythagorean theorem (speed = sqrt(u^2 + v^2))
    speed = (u*u + v*v)**0.5
    
    # Calculate Wind Power Density (WPD) using the formula: WPD = 0.5 * air_density * speed^3
    wpd_all = 0.5 * 1.2 * (speed**3)  # Air density is assumed to be 1.2 kg/m³
    
    print("Data Processed")
    
    # Select data within the specified time range (1980-1999)
    start_year = 1980
    end_year = 1999
    selected_data = wpd_all.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

## ----------------------
## Defining Functions for Episode Length Calculation
## ----------------------

# Import additional libraries needed for episode length calculation
import random  # Unused in this script but imported here
from itertools import accumulate  # For cumulative sums (used in episode detection)
from collections import Counter  # For counting occurrences of elements in an iterable

def episode_length_wind(mds_wind_array):
    """
    Calculates the mean length of wind power density episodes above a threshold.
    
    Parameters:
        mds_wind_array: A NumPy array of wind power density values over time.
    
    Returns:
        The mean duration of episodes where wind power density exceeds the threshold.
        Returns NaN if no episodes are found.
    """
    
    # Identify groups of consecutive values above or below the threshold (240 W/m²)
    groups = accumulate([0] + [(a >= 240) != (b >= 240) for a, b in zip(mds_wind_array, mds_wind_array[1:])])
    
    # Count the number of occurrences in each group
    counts = sorted(Counter(groups).items())
    
    # Filter groups where values are above the threshold and calculate their lengths
    above = [c for n, c in counts if (n % 2 == 0) == (mds_wind_array[0] >= 240)]
    
    if len(above) > 0:
        max_value = np.nanmean(np.array(above))  # Calculate the mean duration of episodes above the threshold
    else:
        max_value = np.nan  # Return NaN if no episodes are found
    
    return max_value

# Apply the episode length function to the dataset using xarray's apply_ufunc for parallelized computation
plot_ep_wind = xr.apply_ufunc(
               episode_length_wind,
               selected_data,  
               input_core_dims=[['time']],          # Specify that input operates along the 'time' dimension
               output_core_dims=[[]],               # Output is a scalar per grid point (no dimensions)
               vectorize=True,                      # Enable vectorization for efficiency
               output_dtypes=[float],               # Output data type is float
               dask='parallelized',                 # Enable Dask parallelization
               dask_gufunc_kwargs={'allow_rechunk': True}  # Allow rechunking during computation if necessary
              )

## ----------------------
## Saving Results to File
## ----------------------

print("Calculation Started")

# Compute the result and save it to a NetCDF file
plot_ep_wind = plot_ep_wind.compute()
plot_ep_wind.to_netcdf(f'/scratch/vishald/jrf2_monsoonlab/shiv/datastats/Final_Twenty_1980_1999_Wind_Episode_length.nc')

print("EP Complete")
