"""
Convert the ArchesWeatherGen ensemble Zarr to per-lead, per-variable Zarr files
compatible with the probex evaluation framework.

Run from the project root:
    python download/convert_arches_wb2.py

Writes to data/ArchesWeatherGen/:
    ArchesWeatherGen_{year}_no_Antarctic_t2m_m{N}_1p50_lead_{LLL}h.zarr
    ArchesWeatherGen_{year}_no_Antarctic_w10_m{N}_1p50_lead_{LLL}h.zarr

Transformations applied:
    - Rename: init_time -> time, member -> number, lead_time -> prediction_timedelta
    - Longitude 0-360 -> -180-180, sorted
    - Latitude sorted south to north, clipped to no_Antarctic domain (-60 to 90)
    - 10m_wind_speed = sqrt(u^2 + v^2)
    - Split by variable and lead; one Zarr file per combination

Note: ArchesWeatherGen runs on 24 h steps, so only leads [48, 120, 168, 240] h are
available (a subset of the standard 20-lead framework set). The eval scripts must be
run with leads restricted to this list and with model="ArchesWeatherGen".
The output files use the resolution tag "1p50" instead of "0p25" to reflect the
native 1.5 degree grid; update the path in the eval scripts accordingly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


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
    "ARCHESWEATHERGEN_FORECASTS_2022",
    str(Path(forecast_root) / "archesweathergen_forecasts_2022.zarr"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--in-zarr", default=DEFAULT_IN_ZARR,
                    help="Path to the source ArchesWeatherGen Zarr store.")
parser.add_argument("--out-root", default="data",
                    help="Root output directory (files go into <out-root>/ArchesWeatherGen/).")
parser.add_argument("--year", type=int, default=2022,
                    help="Forecast year encoded in the output filenames.")
args = parser.parse_args()

in_zarr = Path(args.in_zarr)
out_root = Path(args.out_root)
year = args.year
model = "ArchesWeatherGen"

print("----------------------------------------------")
print("Convert ArchesWeatherGen Zarr for probex eval")
print("----------------------------------------------")
print(f"in_zarr  : {in_zarr}")
print(f"out_root : {out_root}")
print(f"year     : {year}")
print()

print("Opening source Zarr...")
ds = xr.open_zarr(in_zarr, chunks={})

# Rename dimensions to match framework conventions.
ds = ds.rename({
    "init_time": "time",
    "member": "number",
    "lead_time": "prediction_timedelta",
})

# Longitude 0-360 -> -180-180.
ds = ds.assign_coords(
    longitude=(((ds.longitude + 180) % 360) - 180)
).sortby("longitude")

# Latitude: sort south to north, then clip to no_Antarctic domain.
ds = ds.sortby("latitude").sel(latitude=slice(-60, 90))

n_members = ds.sizes["number"]
leads = ds.prediction_timedelta.values.tolist()   # int32 hours: [48, 120, 168, 240]

print(f"n_members  : {n_members}")
print(f"leads (h)  : {leads}")
print(f"lat range  : {float(ds.latitude.min()):.1f} to {float(ds.latitude.max()):.1f}  "
      f"({ds.sizes['latitude']} points)")
print(f"lon range  : {float(ds.longitude.min()):.1f} to {float(ds.longitude.max()):.1f}  "
      f"({ds.sizes['longitude']} points)")
print()

# Variables to extract.  wind speed is derived from the u/v components.
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

        # Select this single lead time.
        ds_lead = ds.sel(prediction_timedelta=[lead_hour])

        if var_long == "10m_wind_speed":
            da = np.sqrt(
                ds_lead["10m_u_component_of_wind"] ** 2
                + ds_lead["10m_v_component_of_wind"] ** 2
            ).astype("float32")
            da.name = "10m_wind_speed"
        else:
            da = ds_lead[var_long]

        # Chunk sizes scaled for the 1.5 degree grid (101 lat x 240 lon after clipping).
        da = da.chunk({
            "time": 8,
            "number": min(5, n_members),
            "prediction_timedelta": 1,
            "latitude": 50,
            "longitude": 120,
        })

        da.to_dataset().to_zarr(out_file, mode="w")
        print(f"  -> {out_file}")

print("\nDone.")
