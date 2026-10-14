"""
Point forecast evaluation.

1 job = 1 year, 1 variable, 1 model.

Evaluates ensemble-mean and ensemble-median point forecasts against ERA5 or HRES_fc0.

Cases:
    max : observed record-breaking high events
    all : all valid data
    min : observed record-breaking low events, only for t2m

Scores:
    RMSE of ensemble mean
    bias of ensemble mean
    RMSE of ensemble median
    bias of ensemble median
    n = number of valid grid-time points used

Only evaluates the no_Antarctic domain.
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path

import xarray as xr

import functions


parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2020, choices=[2018, 2020, 2022])
parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "w10"])
parser.add_argument("--model", type=str, default="GenCast", choices=list(functions.MODEL_CONFIG))
parser.add_argument("--truth", type=str, default="ERA5", choices=["ERA5", "HRES_fc0"])
parser.add_argument("--record-tag", type=str, default="1979_2021")
args = parser.parse_args()


root = Path.cwd()
functions.load_dotenv(root / ".env")

year = args.year
var = args.var
model = args.model
truth = args.truth
record_tag = args.record_tag

config = functions.MODEL_CONFIG[model]
leads = config["leads"]
n_members = config["n_members"]
resolution_tag = config["resolution_tag"]

variable = functions.VARIABLES[var]

eval_start = datetime(year, 1, 1, 0)
eval_end = datetime(year, 12, 31, 18)


cases = ["max", "all"]

if var == "t2m":
    cases.append("min")

out_dir = root / "results" / model / "Point"
out_dir.mkdir(parents=True, exist_ok=True)

forecast_root, truth_file, record_files, lsm_file = functions.eval_paths(root, year, var, truth, record_tag)


print("--------------------------")
print("Point forecast evaluation")
print("--------------------------")
print(f"year    : {year}")
print(f"var     : {var}")
print(f"model   : {model}")
print(f"truth   : {truth}")
print(f"records : {record_tag}")
print(f"out_dir : {out_dir}")

print("\nLoading truth...")
truth_da = xr.open_zarr(truth_file, chunks={})[variable]
truth_da = truth_da.assign_coords(longitude=(((truth_da.longitude + 180) % 360) - 180)).sortby("longitude")
truth_da = truth_da.sel(time=slice(eval_start, eval_end))
truth_da = truth_da.sel(functions.NO_ANTARCTIC)


print("Loading land-sea mask...")
lsm = xr.open_dataset(lsm_file)["lsm"]

lsm = lsm.isel(time=0, drop=True)

lsm = lsm.assign_coords(
    longitude=(((lsm.longitude + 180) % 360) - 180)
).sortby("longitude")

lsm = lsm.sortby("latitude")
lsm = lsm.sel(functions.NO_ANTARCTIC)


print("Loading records...")
record_max = functions.load_record(record_files["max"])
record_min = functions.load_record(record_files["min"]) if var == "t2m" else None

print("Transforming verification time...")
truth_da = functions.transform_ERA5_time(truth_da).sel(hour=[0, 12])

print("Creating masks...")
mask_max = truth_da > record_max
mask_all = xr.ones_like(mask_max, dtype=bool)

masks = {
    "max": mask_max,
    "all": mask_all,
}

if var == "t2m":
    mask_min = truth_da < record_min
    masks["min"] = mask_min


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
    forecast = forecast.sel(functions.NO_ANTARCTIC)

    out = functions.evaluate_point_one_lead(
        forecast,
        truth_da,
        masks,
        lsm,
        eval_start,
        eval_end,
    )

    for case in cases:
        outputs_by_case[case].append(out[case])

    del forecast
    gc.collect()

print("\nCombining and saving outputs...")

for case in cases:

    ds = xr.concat(outputs_by_case[case], dim="leadtime").sortby("leadtime")

    ds.attrs["verification_target"] = truth
    ds.attrs["record_tag"] = record_tag
    ds.attrs["domain"] = "no_Antarctic"

    out_file = out_dir / (
        f"{model}_{case}_point_{year}_{var}_"
        f"no_Antarctic_records_{record_tag}_cond_{truth}.nc"
    )

    ds.to_netcdf(out_file)

    print(f"Saved {case}:")
    print(f"  {out_file}")

print("\nDone.")
