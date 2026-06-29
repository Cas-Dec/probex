"""
CRPS forecast evaluation. 1 job=1 year (2020), 1 variable (t2m, w10), 1 model (GC, IFS ENS) vs its respective truths.
For the no_Antarctic domain, but can be shrunk to smaller domains easily.
"""

from __future__ import annotations
import argparse
import gc # garbage collection, delete the forecasts after each lead to free memory(this can be a problem)
from datetime import datetime
from pathlib import Path

import xarray as xr

import functions # from zhang, time reshaping
import functions_crps # adapted from zhang, CRPS evaluation

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
        "resolution_tag": "0p25",
    },
    "ArchesWeatherGen": {
        "leads": [48, 120, 168, 240],
        "n_members": 64,
        "resolution_tag": "1p50",
    },
    "uCast": {
        "leads": [12, 48, 120, 168, 240],
        "n_members": 63,
        "resolution_tag": "1p50",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2020, choices=[2018, 2020, 2022])
parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "w10"])
parser.add_argument("--model", type=str, default="GenCast", choices=["GenCast", "IFS_ENS", "ArchesWeatherGen", "uCast"])
parser.add_argument("--truth", type=str, default="ERA5", choices=["ERA5", "HRES_fc0"])
parser.add_argument("--record-tag", type=str, default="1979_2018")
args = parser.parse_args()

root = Path.cwd()
data_dir = root / "data"

year = args.year
var = args.var
model = args.model
truth = args.truth
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
cases = ["max", "all"]
if var == "t2m":
    cases.append("min")

out_dir = root / "results" / model / "CRPS"
out_dir.mkdir(parents=True, exist_ok=True)

truth_file = data_dir / f"{truth}" / f"{truth}_{year}_no_Antarctic_{var}_0p25.zarr"

record_max_file = data_dir / "era5_records" / f"ERA5_record_max_month_{record_tag}_{var}.nc"
record_min_file = data_dir / "era5_records" / f"ERA5_record_min_month_{record_tag}_{var}.nc"
lsm_file = data_dir / "era5_records" / "ERA5_lsm_sfc.nc" #land sea mark (0 and 1s; from zhang)

print("--------------------------")
print("CRPS forecast evaluation")
print("--------------------------")
print(f"year    : {year}")
print(f"var     : {var}")
print(f"model   : {model}")
print(f"truth   : {truth}")
print(f"records : {record_tag}")
print(f"out_dir : {out_dir}")

print("\nLoading truth...")
truth_da = xr.open_zarr(truth_file, chunks={})[variable]
truth_da = truth_da.sel(time=slice(eval_start, eval_end))
truth_da = truth_da.sel(no_Antarctic)

print("Loading land-sea mask...")
lsm = xr.open_dataset(lsm_file)["lsm"]
lsm = lsm.isel(time=0, drop=True) # time dimension is just a dummy dimension, we can drop it 
lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")
lsm = lsm.sortby("latitude")
lsm = lsm.sel(no_Antarctic)

print("Loading records...")
record_max = xr.open_dataarray(record_max_file)
record_max = record_max.rename({"lat": "latitude", "lon": "longitude"})
record_max = record_max.sel(no_Antarctic)

if var == "t2m":
    record_min = xr.open_dataarray(record_min_file)
    record_min = record_min.rename({"lat": "latitude", "lon": "longitude"})
    record_min = record_min.sel(no_Antarctic)
else: record_min = None # for wind speeds this is irrelevant

print("Transforming verification time...") # from Zhang2026; time idx to year, month, day, hour idx
# unstack the original time dimension into month and hour, then select only 00 and 12 UTC hours
truth_da = functions.transform_ERA5_time(truth_da).sel(hour=[0, 12])

print("Creating masks...")
# for every valid time, compare truth value to historical record for same calendar month and grid cell.
mask_max = truth_da > record_max
mask_all = xr.ones_like(mask_max, dtype=bool)

if var == "t2m":
    mask_min = truth_da < record_min

crps_by_case = {case: [] for case in cases}
stat_by_case = {case: [] for case in cases}

# until here everything works perfectly.
for i, lead_hour in enumerate(leads, start=1):
    forecast_file = (data_dir / model / f"{model}_{year}_no_Antarctic_{var}_m{n_members}_{resolution_tag}_lead_{lead_hour:03d}h.zarr")

    print("\n" + "=" * 80)
    print(f"{model} {var}: lead {lead_hour:03d}h ({i}/{len(leads)})")
    print(forecast_file)
    print("=" * 80)

    forecast = xr.open_zarr(forecast_file, chunks={})[variable]
    forecast = forecast.sel(no_Antarctic)
    
    print("START max records") # compute crps for current lead time, only for mask_max=1
    crps, stat = functions_crps.ENS_CRPS_landseasep(
        forecast, 
        truth_da, 
        mask_max,
        lsm, # separate scores for land and sea and all
        eval_start,
        eval_end,
    ) # what are crps and stat?

    crps_by_case["max"].append(crps) # sel by leadtime, season, landsea
    stat_by_case["max"].append(stat) # size (# valid grid-time events), mean crps, variance of the crps (before averaging)
    print("DONE max records")
    
    print("START all data")
    crps, stat = functions_crps.ENS_CRPS_landseasep(
        forecast,
        truth_da,
        mask_all,
        lsm,
        eval_start,
        eval_end,
    )

    crps_by_case["all"].append(crps)
    stat_by_case["all"].append(stat)
    print("DONE all data")

    if var == "t2m":
        print("START min records")
        crps, stat = functions_crps.ENS_CRPS_landseasep(
            forecast,
            truth_da,
            mask_min,
            lsm,
            eval_start,
            eval_end,
        )

        crps_by_case["min"].append(crps)
        stat_by_case["min"].append(stat)
        print("DONE min records")

    del forecast # get it out of memory asap (before the next lead time is loaded)
    gc.collect() # free memory

print("\nCombining and saving outputs...")
for case in cases: # max all and min (only for t2m)
    crps = xr.concat(crps_by_case[case], dim="leadtime").sortby("leadtime") # crps: leadtime, season, landsea
    stat = xr.concat(stat_by_case[case], dim="leadtime").sortby("leadtime")

    #crps = crps.rename("CRPS")
    # add metadata
    crps.attrs["verification_target"] = truth
    crps.attrs["record_tag"] = record_tag
    crps.attrs["domain"] = "no_Antarctic"
    stat.attrs["verification_target"] = truth
    stat.attrs["record_tag"] = record_tag
    stat.attrs["domain"] = "no_Antarctic"

    crps_file = out_dir / (f"{model}_{case}_CRPS_{year}_{var}_"
        f"no_Antarctic_records_{record_tag}_cond_{truth}.nc")
    stat_file = out_dir / (f"{model}_{case}_stat_{year}_{var}_"
        f"no_Antarctic_records_{record_tag}_cond_{truth}.nc")
    crps.to_netcdf(crps_file)
    stat.to_netcdf(stat_file)

    print(f"Saved {case}:")
    print(f"  {crps_file}")
    print(f"  {stat_file}")

print("\nDone.")