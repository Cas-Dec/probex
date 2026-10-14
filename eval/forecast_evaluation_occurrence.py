#####################################
# Calculate Brier score for records
#####################################
from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
import xarray as xr

import functions
import functions_occurrence


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
truth_name = args.truth
record_tag = args.record_tag

config = functions.MODEL_CONFIG[model]
leads = config["leads"]
n_members = config["n_members"]
resolution_tag = config["resolution_tag"]

variable = functions.VARIABLES[var]

eval_start = datetime(year, 1, 1, 0)
eval_end = datetime(year, 12, 31, 18)


cases = ["max"]

if var == "t2m":
    cases.append("min")

out_dir = root / "results" / model / "Occurrence"
out_dir.mkdir(parents=True, exist_ok=True)

forecast_root, truth_file, record_files, lsm_file = functions.eval_paths(root, year, var, truth_name, record_tag)


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
truth_da = truth_da.assign_coords(longitude=(((truth_da.longitude + 180) % 360) - 180)).sortby("longitude")
truth_da = truth_da.sel(time=slice(eval_start, eval_end))
truth_da = truth_da.sel(functions.NO_ANTARCTIC)

# Forecast valid times are 00/12 UTC because the lead times are multiples of 12.
truth_da = truth_da.where(truth_da["time"].dt.hour.isin([0, 12]), drop=True)


print("Loading land-sea mask...")
lsm = xr.open_dataset(lsm_file)["lsm"]

lsm = lsm.isel(time=0, drop=True)

lsm = lsm.assign_coords(
    longitude=(((lsm.longitude + 180) % 360) - 180)
).sortby("longitude")

lsm = lsm.sortby("latitude")
lsm = lsm.sel(functions.NO_ANTARCTIC)


print("Loading records...")
records = {case: functions.load_record(record_files[case]) for case in cases}

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
    ds.attrs["brier_estimator"] = "fair ensemble Brier score"
    ds.attrs["brier_reference"] = "Ferro (2014), doi:10.1002/qj.2270"

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
