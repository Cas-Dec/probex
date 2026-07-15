
from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime
from pathlib import Path

import xarray as xr

import functions_occurrence

MODEL_CONFIG = {
    "GenCast": {
        "leads": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120,
                  132, 144, 156, 168, 180, 192, 204, 216, 228, 240],
        "n_members": 50,
        "resolution_tag": "0p25",
    },
    "IFS_ENS": {
        "leads": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120,
                  132, 144, 156, 168, 180, 192, 204, 216, 228, 240],
        "n_members": 50,
        "resolution_tag": "1p50",
    },
    "ArchesWeatherGen": {
        "leads": [48, 120, 168, 240],
        "n_members": 64,
        "resolution_tag": "1p50",
    },
    "uCast": {
        "leads": [12, 48, 120, 168, 240],
        "n_members": 64,
        "resolution_tag": "1p50",
    },
    "FGN": {
        "leads": [12, 48, 120, 168, 240],
        "n_members": 64,
        "resolution_tag": "1p50",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2020, choices=[2018, 2020, 2022])
parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "w10"])
parser.add_argument("--model", type=str, default="GenCast", choices=["GenCast", "IFS_ENS", "ArchesWeatherGen", "uCast", "FGN"])
parser.add_argument("--truth", type=str, default="ERA5", choices=["ERA5", "HRES_fc0"])
parser.add_argument("--record-tag", type=str, default="1979_2018")
args = parser.parse_args()


def _load_dotenv(path: Path) -> None:
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


root = Path.cwd()
_load_dotenv(root / ".env")

heat_data_root = Path(os.environ.get("HEAT_DATA", "data"))

forecast_root = Path(
    os.environ.get("FORECAST_ROOT", str(heat_data_root / "forecasts"))
)
era5_root = Path(
    os.environ.get("ERA5_ROOT", str(heat_data_root / "wb2"))
)
era5_truth_file = Path(
    os.environ.get(
        "ERA5_TRUTH_ZARR",
        str(era5_root / "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
    )
)
records_root = Path(os.environ.get("RECORDS_ROOT", str(root / "data" / "era5_records")))

year = args.year
var = args.var
model = args.model
truth_name = args.truth
record_tag = args.record_tag

config = MODEL_CONFIG[model]
leads = config["leads"]
n_members = config["n_members"]
resolution_tag = config["resolution_tag"]

if var == "t2m":
    variable = "2m_temperature"
elif var == "w10":
    variable = "10m_wind_speed"

eval_start = datetime(year, 1, 1, 0)
eval_end = datetime(year, 12, 31, 18)

no_Antarctic = dict(latitude=slice(-60, 90), longitude=slice(-180, 180))

cases = ["max"]

if var == "t2m":
    cases.append("min")

out_dir = root / "results" / model / "Occurrence"
out_dir.mkdir(parents=True, exist_ok=True)

if truth_name == "ERA5":
    truth_file = era5_truth_file
else:
    truth_file = forecast_root / "HRES_fc0" / f"HRES_fc0_{year}_no_Antarctic_{var}_1p50.zarr"

record_max_file = records_root / f"ERA5_record_max_month_{record_tag}_{var}.nc"
record_min_file = records_root / f"ERA5_record_min_month_{record_tag}_{var}.nc"

lsm_file = records_root / "ERA5_lsm_sfc.nc"


print("--------------------------")
print("Occurrence evaluation")
print("--------------------------")
print(f"year    : {year}")
print(f"var     : {var}")
print(f"model   : {model}")
print(f"truth   : {truth_name}")
print(f"records : {record_tag}")
print(f"out_dir : {out_dir}")

print("\nLoading truth...")
truth_da = xr.open_zarr(truth_file, chunks={})[variable]

truth_da = truth_da.sel(time=slice(eval_start, eval_end))
truth_da = truth_da.sel(no_Antarctic)

# Forecast valid times are 00/12 UTC because the lead times are multiples of 12.
truth_da = truth_da.where(truth_da["time"].dt.hour.isin([0, 12]), drop=True)


print("Loading land-sea mask...")
lsm = xr.open_dataset(lsm_file)["lsm"]

lsm = lsm.isel(time=0, drop=True)

lsm = lsm.assign_coords(
    longitude=(((lsm.longitude + 180) % 360) - 180)
).sortby("longitude")

lsm = lsm.sortby("latitude")
lsm = lsm.sel(no_Antarctic)


print("Loading records...")
record_max = xr.open_dataarray(record_max_file)

record_max = record_max.rename({"lat": "latitude", "lon": "longitude"})

record_max = record_max.assign_coords(
    longitude=(((record_max.longitude + 180) % 360) - 180)
).sortby("longitude")

record_max = record_max.sortby("latitude")
record_max = record_max.sel(no_Antarctic)
records = {"max": record_max}

if var == "t2m":

    record_min = xr.open_dataarray(record_min_file)
    record_min = record_min.rename({"lat": "latitude", "lon": "longitude"})
    record_min = record_min.assign_coords(
        longitude=(((record_min.longitude + 180) % 360) - 180)
    ).sortby("longitude")

    record_min = record_min.sortby("latitude")
    record_min = record_min.sel(no_Antarctic)

    records["min"] = record_min

outputs_by_case = {case: [] for case in cases}

for i, lead_hour in enumerate(leads, start=1):

    forecast_file = (
        forecast_root / model
        / f"{model}_{year}_no_Antarctic_{var}_m{n_members}_{resolution_tag}_lead_{lead_hour:03d}h.zarr"
    )

    print("\n" + "=" * 80)
    print(f"{model} {var}: lead {lead_hour:03d}h ({i}/{len(leads)})")
    print(forecast_file)
    print("=" * 80)

    forecast = xr.open_zarr(forecast_file, chunks={})[variable]
    forecast = forecast.sel(no_Antarctic)

    for case in cases:

        out = functions_occurrence.evaluate_occurrence_one_lead(
            forecast=forecast,
            truth=truth_da,
            record=records[case],
            land_sea_mask=lsm,
            tail=case,
            eval_start=eval_start,
            eval_end=eval_end,
        )

        outputs_by_case[case].append(out)

    del forecast
    gc.collect()

print("\nCombining and saving outputs...")

for case in cases:

    ds = xr.concat(outputs_by_case[case], dim="leadtime").sortby("leadtime")

    ds.attrs["verification_target"] = truth_name
    ds.attrs["record_tag"] = record_tag
    ds.attrs["domain"] = "no_Antarctic"

    if var == "t2m" and case == "max":
        ds.attrs["event"] = "heat"
    elif var == "t2m" and case == "min":
        ds.attrs["event"] = "cold"
    elif var == "w10" and case == "max":
        ds.attrs["event"] = "wind"

    out_file = out_dir / (
        f"{model}_{case}_occurrence_{year}_{var}_"
        f"no_Antarctic_records_{record_tag}_cond_{truth_name}.nc"
    )

    ds.to_netcdf(out_file)

    print(f"Saved {case}:")
    print(f"  {out_file}")

print("\nDone.")