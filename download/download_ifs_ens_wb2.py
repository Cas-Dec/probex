"""
Download one single-lead IFS ENS chunk from WeatherBench 2.
    variable: 2m_temperature or 10m_wind_speed
    dims: time, number, prediction_timedelta, latitude, longitude
"""

from __future__ import annotations
import argparse
from pathlib import Path
import xarray as xr

IFS_ZARR = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr"

parser = argparse.ArgumentParser(description="Download a single-lead IFS ENS chunk from WB2.")
parser.add_argument("--zarr-bucket", type=str, default=IFS_ZARR, help="weatherbench2 Zarr bucket to read from.")
parser.add_argument("--year", type=int, default=2020, choices=[2018, 2020], help="Forecast year.")
parser.add_argument("--variable", type=str, default="2m_temperature", choices=["2m_temperature", "10m_wind_speed"],
help="WeatherBench 2 variable name.")
parser.add_argument("--lead-hour", type=int, default=24, help="Lead time in hours.")
parser.add_argument("--out-root", type=str, default="data/IFS_ENS", help="Output directory.")
args = parser.parse_args()

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)
year = args.year
variable = args.variable
lead_hour = args.lead_hour
region = "no_Antarctic"
lat_min = -10 #-60
lat_max = 10 # 90
n_members = 1#50
if variable == "2m_temperature": var = "t2m"
elif variable == "10m_wind_speed": var = "w10"
out_file = (out_root
    / f"IFS_ENS_{year}_{region}_{var}_"
        f"m{n_members}_0p25_lead_{lead_hour:03d}h.zarr")

print("----------------------------------------------")
print("Download a single lead of IFS ENS from WeatherBench 2.")
print("----------------------------------------------")
print(f"year: {year}")
print(f"variable: {variable}")
print(f"lead hour: {lead_hour}")
print(f"latitude: {lat_min} to {lat_max}")
print(f"members: {n_members}")
print(f"output: {out_file}")
print("\nOpening Zarr store....")

# Force prediction_timedelta to remain integer hours.
ds = xr.open_zarr(args.zarr_bucket, storage_options={"token": "anon"}, chunks={}, decode_timedelta=False)
x = ds[variable]
x = x.assign_coords(longitude=(((x.longitude + 180) % 360) - 180)).sortby("longitude") # from 0,360 to -180,180
x = x.sel(prediction_timedelta=[lead_hour], latitude=slice(lat_min, lat_max), longitude=slice(-180, 180))
x = x.sel(time=slice(f"{year - 1}-12-22", f"{year}-12-31T12:00:00"))
x = x.isel(number=slice(0, n_members))

# since these are huge arrays, we store them in chunks on disk
x = x.chunk({"time": 8, "number": min(5, x.sizes["number"]), "prediction_timedelta": 1, "latitude": 180, "longitude": 360})
ds_out = x.to_dataset(name=variable)

# Remove inherited encoding.
for name in ds_out.variables: ds_out[name].encoding = {}
encoding_chunks = tuple(x.chunksizes[dim][0] for dim in x.dims)

print("\nWriting output chunks...")
print(dict(zip(x.dims, encoding_chunks)))

ds_out.to_zarr(out_file, mode="w", encoding={variable: {"chunks": encoding_chunks}}, consolidated=True)
print("\nDone.")
print(f"Saved: {out_file}")