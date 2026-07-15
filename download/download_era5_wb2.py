"""
Download an ERA5 verification chunk from WeatherBench 2.
    variable: 2m_temperature or 10m_wind_speed
    dims: time, latitude, longitude
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import xarray as xr

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"


def _load_dotenv(path: Path = Path(".env")) -> None:
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


_load_dotenv()
default_out_root = os.environ.get(
    "ERA5_DOWNLOAD_OUT_ROOT",
    os.environ.get("FORECAST_ROOT", "data/forecasts") + "/ERA5",
)

parser = argparse.ArgumentParser(description="Download ERA5 verification data from WB2.")
parser.add_argument("--zarr-bucket", type=str, default=ERA5_ZARR, help="weatherbench2 Zarr bucket to read from.")
parser.add_argument("--year", type=int, default=2022, help="Verification year.")
parser.add_argument("--variable", type=str, default="2m_temperature", choices=["2m_temperature", "10m_wind_speed"],
help="WeatherBench 2 variable name.")
parser.add_argument("--out-root", type=str, default=default_out_root, help="Output directory.")
args = parser.parse_args()

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)
year = args.year
variable = args.variable
region = "no_Antarctic"
lat_min = -60
lat_max = 90
if variable == "2m_temperature": var = "t2m"
elif variable == "10m_wind_speed": var = "w10"
out_file = out_root / f"{year}_{region}_{var}_0p25.zarr"

print("----------------------------------------------")
print("Download ERA5 verification data from WeatherBench 2.")
print("----------------------------------------------")
print(f"year: {year}")
print(f"variable: {variable}")
print(f"latitude: {lat_min} to {lat_max}")
print(f"output: {out_file}")
print("\nOpening Zarr store....")

ds = xr.open_zarr(args.zarr_bucket, storage_options={"token": "anon"}, chunks={})
x = ds[variable]
x = x.assign_coords(longitude=(((x.longitude + 180) % 360) - 180)).sortby("longitude") # from 0,360 to -180,180
x = x.sortby("latitude") # ERA5 latitude seems to be ordered from north to south ...
x = x.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"),
          latitude=slice(lat_min, lat_max),
          longitude=slice(-180, 180))
# store in chunks so we never have to load the full year into memory
x = x.chunk({"time": 24, "latitude": 180, "longitude": 360})
ds_out = x.to_dataset(name=variable)

for name in ds_out.variables: ds_out[name].encoding = {}
encoding_chunks = tuple(x.chunksizes[dim][0] for dim in x.dims)

print("\nWriting output chunks...")
ds_out.to_zarr(out_file, mode="w", encoding={variable: {"chunks": encoding_chunks}}, consolidated=True)
print("\nDone.")
print(f"Saved: {out_file}")