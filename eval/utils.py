"""
Author: Yujie Zhang
Article: "Physics-based models outperform AI weather forecasts of record-breaking extremes
Synopsis: This script contains utility functions: changing of kelvin to celsius,
unifying the dimension names (lon and lat), and adjusting longitude coordinates from 0-360 to -180--180.

Edits: only some comments
"""

import xarray as xr
import numpy as np

#transform temperature unit from Kelvin to Celsius
def Kelvin2Celsius(data: xr.DataArray) -> xr.DataArray:
    return data-273.15

#unify dimension names (longitude and latitude, and lon and lat)
def rename_latlon(data: xr.DataArray) -> xr.DataArray:
    return data.rename({'longitude': 'lon', 'latitude': 'lat'})

#change longitude from 0-360 to -180--180 and sort from south to north if necessary
def change_lonlat(data: xr.DataArray) -> xr.DataArray:
    if 'lon' not in data.coords:
        data = rename_latlon(data)
    if data.lon.max() > 180:
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon')
    if any(np.diff(data.lat) < 0):
        data = data.sortby('lat') # sort latitude from south to north
    return data
