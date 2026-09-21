"""
Download HRES-fc0 verification data from WeatherBench 2.
    variable: 2m_temperature or 10m_wind_speed
    dims: time, prediction_timedelta, latitude, longitude
    resolution: 1.5° (240x121 equiangular)
"""

from __future__ import annotations
import argparse
import os

# Force gRPC's native DNS resolver: the default c-ares resolver tries IPv6 first
# and times out on this cluster's IPv4-only network, breaking gs:// access.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from pathlib import Path
import xarray as xr

HRES_ZARR = "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"


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
    "HRES_FC0_OUT_ROOT",
    os.environ.get("FORECAST_ROOT", "data/forecasts") + "/HRES_fc0",
)

parser = argparse.ArgumentParser(description="Download HRES-fc0 data from WB2.")
parser.add_argument("--zarr-bucket", type=str, default=HRES_ZARR, help="weatherbench2 Zarr bucket to read from.")
parser.add_argument("--year", type=int, default=2022, choices=[2018, 2020, 2022], help="Verification year.")
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
out_file = out_root / f"HRES_fc0_{year}_{region}_{var}_1p50.zarr"

print("----------------------------------------------")
print("Download HRES-fc0 verification data from WeatherBench 2.")
print("----------------------------------------------")
print(f"year: {year}")
print(f"variable: {variable}")
print(f"latitude: {lat_min} to {lat_max}")
print(f"output: {out_file}")
print("\nOpening Zarr store....")

# Force prediction_timedelta to remain integer hours.
ds = xr.open_zarr(args.zarr_bucket, storage_options={"token": "anon"}, chunks={}, decode_timedelta=False)
x = ds[variable]
x = x.assign_coords(longitude=(((x.longitude + 180) % 360) - 180)).sortby("longitude") # from 0,360 to -180,180
x = x.sel(prediction_timedelta=0, drop=True) # HRES analysis / fc0
x = x.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"),
          latitude=slice(lat_min, lat_max),
          longitude=slice(-180, 180))

x = x.where(x.time.dt.hour.isin([0, 12]), drop=True)

# store in chunks so we never have to load the full year into memory
x = x.chunk({"time": 24, "latitude": 180, "longitude": 360})
ds_out = x.to_dataset(name=variable)

# Remove inherited encoding.
for name in ds_out.variables: ds_out[name].encoding = {}
encoding_chunks = tuple(x.chunksizes[dim][0] for dim in x.dims)

print("\nWriting output chunks...")
ds_out.to_zarr(out_file, mode="w", encoding={variable: {"chunks": encoding_chunks}}, consolidated=True)
print("\nDone.")
print(f"Saved: {out_file}")