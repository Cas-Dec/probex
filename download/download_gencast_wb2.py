"""
Download a Gencast chunk (single lead, for initializations 00 and 12,all ensemble members) from WeatherBench 2.
    variable: 2m_temperature or 10m_wind_speed
    dims: time, number, prediction_timedelta, latitude, longitude
"""

from __future__ import annotations
import argparse
import os

# Force gRPC's native DNS resolver: the default c-ares resolver tries IPv6 first
# and times out on this cluster's IPv4-only network, breaking gs:// access.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from pathlib import Path
import xarray as xr


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
    "GENCAST_OUT_ROOT",
    os.environ.get("FORECAST_ROOT", "data/forecasts") + "/GenCast",
)

parser = argparse.ArgumentParser(description="Download a single-lead GenCast chunk from WB2.")
parser.add_argument(
    "--zarr-bucket",type=str,default="gs://weatherbench2/datasets/gencast/2020-1440x721.zarr",
    help="weatherbench2 Zarr bucket to read from.")
parser.add_argument("--year",type=int,default=2020, help="Forecast year.")
parser.add_argument("--variable",type=str,default="2m_temperature",choices=["2m_temperature", "10m_wind_speed"],
help="WeatherBench 2 variable name.")
parser.add_argument("--lead-hour",type=int,default=24,help="Lead time in hours.")
parser.add_argument("--out-root",type=str,default=default_out_root,help="Output directory.",)
args = parser.parse_args()

out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)
year = args.year
variable = args.variable
lead_hour = args.lead_hour
region = "no_Antarctic"
lat_min = -10 # -60
lat_max = 10 # 90
n_members = 1 # 50
if variable == "2m_temperature": var = "t2m"
elif variable == "10m_wind_speed": var = "w10"
out_file = (out_root
    / f"GenCast_{year}_{region}_{var}_"
        f"m{n_members}_0p25_lead_{lead_hour:03d}h.zarr")

print("----------------------------------------------")
print("Download a single lead of GenCast from WeatherBench 2.")
print("----------------------------------------------")
print(f"year: {year}")
print(f"variable: {variable}")
print(f"lead hour: {lead_hour}")
print(f"latitude: {lat_min} to {lat_max}")
print(f"members: {n_members}")
print(f"output: {out_file}")
print("\nOpening Zarr store....")

 # for some reason the cluster read timedelta as f*king nanoseconds, so let's force it to read it as int64 (hours)
ds = xr.open_zarr(args.zarr_bucket,storage_options={"token": "anon"}, chunks={}, decode_timedelta=False)
x = ds[variable]
x = x.assign_coords(lon=(((x.lon + 180) % 360) - 180)).sortby("lon") # from 0,360 to -180,180
x = x.rename({"lon": "longitude", "lat": "latitude", "sample": "number"})
x = x.sel(prediction_timedelta=[lead_hour],latitude=slice(lat_min, lat_max),longitude=slice(-180, 180)) # exclude antarctica
x = x.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"))
x = x.where(x.time.dt.hour.isin([0, 12]), drop=True)
x = x.isel(number=slice(0, n_members))
# since these are huge arrays (approx 732 x 50 x 1 x 601 x 1440), we can store it in little
# pieces (chunks) on disk  so we never have to load 100s of GB into the memory at once (like we would do with netcdf)
x = x.chunk({"time": 8,"number": min(5, x.sizes["number"]),"prediction_timedelta": 1,"latitude": 180,"longitude": 360})
ds_out = x.to_dataset(name=variable)
# Remove inherited encoding. every variable can have an encoding dictionary (and we don't want it to interfere)
for name in ds_out.variables: ds_out[name].encoding = {}
encoding_chunks = tuple(x.chunksizes[dim][0] for dim in x.dims) # extract chunk sizes.
print("\nWriting output chunks...")
ds_out.to_zarr(out_file,mode="w", encoding={variable: {"chunks": encoding_chunks}},consolidated=True) # one metadata file on the chunks (consolidated)
print("\nDone.")
print(f"Saved: {out_file}")
