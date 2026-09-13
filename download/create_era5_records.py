"""
Create ERA5 monthly record climatology files used by evaluation scripts.

Outputs (written into --out-root):
    ERA5_record_max_month_<record-tag>_t2m.nc
    ERA5_record_min_month_<record-tag>_t2m.nc
    ERA5_record_max_month_<record-tag>_w10.nc
    ERA5_lsm_sfc.nc

By default, records are computed from ERA5 years 1979-2021 and only
00/12 UTC timestamps to match forecast verification times.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = {
    "t2m": "2m_temperature",
    "w10": "10m_wind_speed",
}


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


def _normalize_lon_lat(da: xr.DataArray) -> xr.DataArray:
    out = da
    if "longitude" in out.coords:
        out = out.assign_coords(
            longitude=(((out.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    if "latitude" in out.coords:
        out = out.sortby("latitude")
    return out


def _parse_hours(hours_csv: str) -> list[int]:
    return [int(h.strip()) for h in hours_csv.split(",") if h.strip()]


_load_dotenv()

heat_data = Path(os.environ.get("HEAT_DATA", "data"))
era5_root_default = Path(os.environ.get("ERA5_ROOT", str(heat_data / "wb2")))
era5_truth_default = Path(
    os.environ.get(
        "ERA5_TRUTH_ZARR",
        str(era5_root_default / "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
    )
)

parser = argparse.ArgumentParser(description="Create ERA5 monthly record climatology files.")
parser.add_argument("--era5-zarr", type=str, default=str(era5_truth_default),
                    help="Path to ERA5 Zarr store.")
parser.add_argument("--out-root", type=str, default=str(era5_root_default),
                    help="Directory where record files are written.")
parser.add_argument("--record-tag", type=str, default="1979_2021",
                    help="Tag used in output filenames.")
parser.add_argument("--start-year", type=int, default=1979,
                    help="First year in climatology window.")
parser.add_argument("--end-year", type=int, default=2021,
                    help="Last year in climatology window.")
parser.add_argument("--hours", type=str, default="0,12",
                    help="Comma-separated UTC hours used for climatology, e.g. '0,12'.")
args = parser.parse_args()

era5_zarr = Path(args.era5_zarr)
out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

hours = _parse_hours(args.hours)
time_start = f"{args.start_year}-01-01"
time_end = f"{args.end_year}-12-31T23:59:59"

print("----------------------------------------------")
print("Create ERA5 monthly record climatology")
print("----------------------------------------------")
print(f"ERA5 zarr   : {era5_zarr}")
print(f"Output root : {out_root}")
print(f"Record tag  : {args.record_tag}")
print(f"Year window : {args.start_year}-{args.end_year}")
print(f"Hours (UTC) : {hours}")

print("\nOpening ERA5 store...")
ds = xr.open_zarr(era5_zarr, chunks={})

for var_short, var_long in VARIABLES.items():
    print(f"\nProcessing {var_short} ({var_long})...")

    da = ds[var_long]
    da = _normalize_lon_lat(da)
    da = da.sel(time=slice(time_start, time_end))

    if hours:
        da = da.where(da.time.dt.hour.isin(hours), drop=True)

    rec_max = da.groupby("time.month").max("time", skipna=True)
    rec_max = rec_max.rename({"latitude": "lat", "longitude": "lon"})
    rec_max_file = out_root / f"ERA5_record_max_month_{args.record_tag}_{var_short}.nc"
    rec_max.name = f"ERA5_record_max_month_{args.record_tag}_{var_short}"
    rec_max.to_netcdf(rec_max_file)
    print(f"  Wrote {rec_max_file}")

    if var_short == "t2m":
        rec_min = da.groupby("time.month").min("time", skipna=True)
        rec_min = rec_min.rename({"latitude": "lat", "longitude": "lon"})
        rec_min_file = out_root / f"ERA5_record_min_month_{args.record_tag}_{var_short}.nc"
        rec_min.name = f"ERA5_record_min_month_{args.record_tag}_{var_short}"
        rec_min.to_netcdf(rec_min_file)
        print(f"  Wrote {rec_min_file}")

print("\nPreparing land-sea mask...")
if "land_sea_mask" in ds:
    lsm = ds["land_sea_mask"]
elif "lsm" in ds:
    lsm = ds["lsm"]
else:
    raise KeyError("No land-sea mask variable found in ERA5 Zarr (expected 'land_sea_mask' or 'lsm').")

lsm = _normalize_lon_lat(lsm)

if "time" in lsm.dims:
    lsm = lsm.isel(time=0, drop=True)

# Eval scripts expect a dummy time dimension and variable name 'lsm'.
lsm = lsm.expand_dims(time=[pd.Timestamp(f"{args.start_year}-01-01")])
lsm_ds = lsm.to_dataset(name="lsm")
lsm_file = out_root / "ERA5_lsm_sfc.nc"
lsm_ds.to_netcdf(lsm_file)
print(f"  Wrote {lsm_file}")

print("\nDone.")
