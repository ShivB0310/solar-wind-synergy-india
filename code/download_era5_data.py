import cdsapi

# Initialize the CDS API client
c = cdsapi.Client()

# Define the parameters for data retrieval
params = {
    'product_type': ['ensemble_mean', 'ensemble_members', 'ensemble_spread', 'reanalysis'],
    'format': 'netcdf',
    'variable': 'land_sea_mask',  # Change this for other variables like 'ssrd' or 'wind_speed'
    'year': ['2021'],  # Change this for other years or ranges
    'month': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
    ],
    'day': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
    ],
    'time': [
        '00:00', '01:00', '02:00',
        '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00',
        '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00',
        '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00',
        '21:00', '22:00', '23:00',
    ],
    'area': [
        38.95, 65.55, 3.75,
        97.35,
    ],
    'grid': [0.1, 0.1], 
}

# Output file path
output_file = 'land_sea_mask.nc' #insert your file path here

# Retrieve data from ERA5
c.retrieve(
    'reanalysis-era5-single-levels',
    params,
    output_file
)
