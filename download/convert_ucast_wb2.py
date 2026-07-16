"""
Convert the uCast ensemble Zarr to per-lead, per-variable Zarr files
compatible with the probex evaluation framework.

Run from the project root:
    python download/convert_ucast_wb2.py

Writes to data/uCast/:
    uCast_{year}_no_Antarctic_t2m_m{N}_1p50_lead_{LLL}h.zarr
    uCast_{year}_no_Antarctic_w10_m{N}_1p50_lead_{LLL}h.zarr

Transformations applied:
    - Check and patch missing coordinate data (lead_time, init_time) if the
      zarr ships positional indices instead of real values (see
      _check_and_fix_ucast_coords); no-op once the upstream zarr is fixed
    - Fix NaN coordinate metadata at index 0 for longitude (0.0°) and
      ensemble_member (0) — both axes carry real data; only the label is wrong
    - Rename: init_time -> time, ensemble_member -> number, lead_time -> prediction_timedelta
    - Convert lead_time to integer hours
    - Re-index number to 0-based int32 (source values 0-63)
    - Longitude 0-360 -> -180-180, sorted
    - Clip to no_Antarctic domain (-60 to 90); latitude is already ascending
    - Transpose to (time, number, prediction_timedelta, latitude, longitude)
    - Strip _predicted suffix from variable names; compute 10m_wind_speed = sqrt(u^2 + v^2)
    - Split by variable and lead; one Zarr file per combination
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Known uCast lead times in hours — used only as fallback when the zarr
# coordinate arrays are missing (positional-index bug).
_KNOWN_LEAD_HOURS = [12, 48, 120, 168, 240]


def _check_and_fix_ucast_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Detect and patch missing coordinate data in the uCast zarr.

    Some exports store lead_time and init_time as bare positional integer
    indices rather than actual timedelta/datetime values (the chunk files for
    those coordinates are absent from the zarr).  When that is detected the
    function patches in the known-correct values and emits a warning so the
    caller is informed.

    Once the upstream zarr is fixed this function becomes a no-op: xarray
    will decode the coordinates to timedelta64/datetime64 dtypes, which
    causes both guards to pass without any patching.
    """
    # --- lead_time ---
    lt = ds.lead_time.values
    if np.issubdtype(lt.dtype, np.timedelta64):
        # Properly decoded by xarray — nothing to do.
        pass
    elif lt.max() < 3600:
        # Positional indices (0-4): patch with known lead hours expressed as
        # nanoseconds so the existing conversion formula (// 3_600_000_000_000)
        # continues to work unchanged.
        warnings.warn(
            f"uCast lead_time looks like positional indices (max={lt.max()}); "
            f"patching with known leads {_KNOWN_LEAD_HOURS} h.",
            stacklevel=2,
        )
        ns_per_hour = 3_600_000_000_000
        ds = ds.assign_coords(
            lead_time=np.array(
                [h * ns_per_hour for h in _KNOWN_LEAD_HOURS], dtype="int64"
            )
        )
    else:
        warnings.warn(
            f"uCast lead_time has unexpected integer values (max={lt.max()}); "
            "treating as raw seconds and converting to nanoseconds.",
            stacklevel=2,
        )
        ds = ds.assign_coords(lead_time=(lt * 1_000_000_000).astype("int64"))

    # --- init_time ---
    it = ds.init_time.values
    if np.issubdtype(it.dtype, np.datetime64):
        # Properly decoded by xarray — nothing to do.
        pass
    elif it.max() < 1_000_000:
        # Positional indices (0-729): reconstruct as 00Z/12Z timestamps for
        # the forecast year.  n is used instead of a hardcoded 730 so the
        # function generalises to other year lengths.
        n = ds.sizes["init_time"]
        warnings.warn(
            f"uCast init_time looks like positional indices (max={it.max()}); "
            f"patching with {n} steps at 12-hour frequency starting 2022-01-01.",
            stacklevel=2,
        )
        ds = ds.assign_coords(
            init_time=pd.date_range("2022-01-01", periods=n, freq="12h")
        )
    else:
        # Raw Unix seconds — decode directly.
        warnings.warn(
            f"uCast init_time has unexpected integer values (max={it.max()}); "
            "treating as Unix seconds.",
            stacklevel=2,
        )
        ds = ds.assign_coords(
            init_time=pd.to_datetime(it, unit="s", utc=True).tz_localize(None)
        )

    return ds


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_dotenv()

forecast_root = os.environ.get("FORECAST_ROOT", "data/forecasts")

DEFAULT_IN_ZARR = os.environ.get(
    "UCAST_FORECASTS_2022",
    str(Path(forecast_root) / "ucast_forecasts_2022.zarr"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--in-zarr", default=DEFAULT_IN_ZARR,
                    help="Path to the source uCast Zarr store.")
parser.add_argument("--out-root", default="data",
                    help="Root output directory (files go into <out-root>/uCast/).")
parser.add_argument("--year", type=int, default=2022,
                    help="Forecast year encoded in the output filenames.")
args = parser.parse_args()

in_zarr = Path(args.in_zarr)
out_root = Path(args.out_root)
year = args.year
model = "uCast"

print("----------------------------------------------")
print("Convert uCast Zarr for probex eval")
print("----------------------------------------------")
print(f"in_zarr  : {in_zarr}")
print(f"out_root : {out_root}")
print(f"year     : {year}")
print()

print("Opening source Zarr...")
ds = xr.open_zarr(in_zarr, chunks={}, consolidated=False)
ds = _check_and_fix_ucast_coords(ds)

# Source file has NaN as the coordinate label at index 0 for both longitude
# and ensemble_member; the underlying data is valid.  Patch the labels in
# place: longitude step is 1.5° so index 0 is 0.0°; members are 0-based.
lon_vals = ds.longitude.values.copy()
lon_vals[0] = 0.0
ds = ds.assign_coords(longitude=lon_vals)

mem_vals = ds.ensemble_member.values.copy()
mem_vals[0] = 0.0
ds = ds.assign_coords(ensemble_member=mem_vals)

# Convert lead_time from timedelta64[ns] to integer hours.
lead_hours = (ds.lead_time.values.astype("int64") // 3_600_000_000_000).astype("int32")

# Some exports still decode to duplicated/all-zero lead labels. Patch once here
# so subsequent .sel(prediction_timedelta=[lead]) calls use a unique index.
if len(np.unique(lead_hours)) != lead_hours.size:
    if ds.sizes["lead_time"] == len(_KNOWN_LEAD_HOURS):
        warnings.warn(
            "uCast lead_time contains duplicate values after decode; "
            f"overriding with known leads {_KNOWN_LEAD_HOURS} h.",
            stacklevel=2,
        )
        lead_hours = np.array(_KNOWN_LEAD_HOURS, dtype="int32")
    else:
        raise ValueError(
            "uCast lead_time contains duplicate labels and cannot be patched "
            f"safely (size={ds.sizes['lead_time']})."
        )

ds = ds.assign_coords(lead_time=lead_hours)

# Rename dimensions to match framework conventions.
ds = ds.rename({
    "init_time": "time",
    "ensemble_member": "number",
    "lead_time": "prediction_timedelta",
})

# Re-index members to 0-based int32 (source values are 0.0-63.0 as float).
n_members = ds.sizes["number"]
ds = ds.assign_coords(number=np.arange(n_members, dtype="int32"))

# Longitude 0-360 -> -180-180.
ds = ds.assign_coords(
    longitude=(((ds.longitude + 180) % 360) - 180)
).sortby("longitude")

# Clip to no_Antarctic domain; latitude is already ascending.
ds = ds.sel(latitude=slice(-60, 90))

leads = ds.prediction_timedelta.values.tolist()

print(f"n_members  : {n_members}")
print(f"leads (h)  : {leads}")
print(f"lat range  : {float(ds.latitude.min()):.1f} to {float(ds.latitude.max()):.1f}  "
      f"({ds.sizes['latitude']} points)")
print(f"lon range  : {float(ds.longitude.min()):.1f} to {float(ds.longitude.max()):.1f}  "
      f"({ds.sizes['longitude']} points)")
print()

VARIABLES = {
    "t2m": "2m_temperature",
    "w10": "10m_wind_speed",
}

for var_short, var_long in VARIABLES.items():
    for lead_hour in leads:

        out_dir = out_root / model
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / (
            f"{model}_{year}_no_Antarctic_{var_short}"
            f"_m{n_members}_1p50_lead_{lead_hour:03d}h.zarr"
        )

        if out_file.exists():
            print(f"Skip (exists): {out_file.name}")
            continue

        print(f"Writing {out_file.name} ...")

        ds_lead = ds.sel(prediction_timedelta=[lead_hour])

        if var_long == "10m_wind_speed":
            da = np.sqrt(
                ds_lead["10m_u_component_of_wind_predicted"] ** 2
                + ds_lead["10m_v_component_of_wind_predicted"] ** 2
            ).astype("float32")
            da.name = "10m_wind_speed"
        else:
            da = ds_lead[f"{var_long}_predicted"].rename(var_long)

        # Transpose to standard order: (time, number, prediction_timedelta, latitude, longitude).
        da = da.transpose("time", "number", "prediction_timedelta", "latitude", "longitude")

        da = da.chunk({
            "time": 8,
            "number": min(5, n_members),
            "prediction_timedelta": 1,
            "latitude": 50,
            "longitude": 120,
        })

        # Drop source encoding metadata (especially chunks) before writing,
        # otherwise xarray may reject dask/zarr chunk alignment.
        da.encoding = {}
        da.to_dataset(name=da.name).to_zarr(out_file, mode="w")
        print(f"  -> {out_file}")

print("\nDone.")
