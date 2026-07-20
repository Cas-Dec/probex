from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime
from pathlib import Path

import dask
import numpy as np
import xarray as xr

import functions
import functions_crps
import functions_occurrence


dask.config.set({"array.slicing.split_large_chunks": True})

MODEL_CONFIG = {
    "GenCast": {
        "leads": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120,
                  132, 144, 156, 168, 180, 192, 204, 216, 228, 240],
        "n_members": 50,
        "resolution_tag": "0p25",
    },
    "IFS_ENS": {
        "leads": [12, 48, 120, 168, 240],
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
parser.add_argument("--margin-max", type=float, default=6.0)
parser.add_argument("--margin-step", type=float, default=0.25)
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

margins = np.round(
    np.arange(0.0, args.margin_max + 0.5 * args.margin_step, args.margin_step),
    10,
)

eval_start = datetime(year, 1, 1, 0)
eval_end = datetime(year, 12, 31, 18)

no_Antarctic = dict(latitude=slice(-60, 90), longitude=slice(-180, 180))

landsea_names = ["all", "land", "sea"]

cases = ["max"]
if var == "t2m":
    cases.append("min")

out_dir = root / "results" / model / "Margin"
out_dir.mkdir(parents=True, exist_ok=True)

if truth_name == "ERA5":
    truth_file = era5_truth_file
else:
    truth_file = forecast_root / "HRES_fc0" / f"HRES_fc0_{year}_no_Antarctic_{var}_1p50.zarr"

record_max_file = records_root / f"ERA5_record_max_month_{record_tag}_{var}.nc"
record_min_file = records_root / f"ERA5_record_min_month_{record_tag}_{var}.nc"

lsm_file = records_root / "ERA5_lsm_sfc.nc"

print("--------------------------")
print("Margin evaluation")
print("--------------------------")
print(f"year        : {year}")
print(f"var         : {var}")
print(f"model       : {model}")
print(f"truth       : {truth_name}")
print(f"records     : {record_tag}")
print(f"margin max  : {args.margin_max}")
print(f"margin step : {args.margin_step}")
print(f"out_dir     : {out_dir}")

print("\nLoading truth...")
truth_da = xr.open_zarr(truth_file, chunks={})[variable]
truth_da = truth_da.assign_coords(longitude=(((truth_da.longitude + 180) % 360) - 180)).sortby("longitude")
truth_da = truth_da.sel(time=slice(eval_start, eval_end))
truth_da = truth_da.sel(no_Antarctic)

# Forecast valid times are 00/12 UTC because all lead times are multiples of 12.
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
    forecast = functions.normalize_spatial_coords(forecast)

    lead_delta = np.timedelta64(lead_hour, "h")

    # The downloaded file is single-lead.
    forecast = forecast.isel(prediction_timedelta=0, drop=True)

    # Convert initialization time to valid time.
    valid_time = forecast["time"].values.astype("datetime64[ns]") + lead_delta
    forecast = forecast.assign_coords(time=valid_time)
    forecast = forecast.sel(time=slice(eval_start, eval_end))

    if forecast.sizes["time"] == 0:
        print("No valid forecast times. Skipping lead.")
        continue

    truth_sub = truth_da.sel(time=forecast["time"])
    truth_sub = functions.normalize_spatial_coords(truth_sub)

    truth_sub = truth_sub.reindex(
        latitude=forecast["latitude"],
        longitude=forecast["longitude"],
        method="nearest",
    )

    forecast, truth_sub = xr.align(
        forecast,
        truth_sub,
        join="inner",
    )

    forecast = forecast.chunk(
        {
            "time": 31,
            "number": -1,
            "latitude": 30,
            "longitude": 60,
        }
    )

    truth_sub = truth_sub.chunk(
        {
            "time": 31,
            "latitude": 30,
            "longitude": 60,
        }
    )

    lsm_sub = lsm.reindex(
        latitude=forecast["latitude"],
        longitude=forecast["longitude"],
        method="nearest",
    )
    lsm_sub = functions.normalize_spatial_coords(lsm_sub)

    weights = functions.latitude_weights(forecast)

    # This is independent of the record case.
    crps = functions_crps.ensemble_crps_sorted(
        forecast,
        truth_sub,
    )

    for case in cases:

        print(f"    Case: {case}")
        record = records[case]

        # Select monthly records at the forecast valid times.
        record_sub = functions_occurrence.record_by_time(
            record,
            forecast["time"].values,
        )

        record_sub = record_sub.reindex(
            latitude=forecast["latitude"],
            longitude=forecast["longitude"],
            method="nearest",
        )

        forecast_case, truth_case, record_sub, crps_case = xr.align(
            forecast,
            truth_sub,
            record_sub,
            crps,
            join="inner",
        )

        if case == "max":

            margin = truth_case - record_sub
            record_probability = functions_occurrence.forecast_members_exceed(
                forecast_case,
                record_sub,
                tail="max",
            )

            out_of_ensemble = truth_case > forecast_case.max("number", skipna=True)

        elif case == "min":

            margin = record_sub - truth_case
            record_probability = functions_occurrence.forecast_members_exceed(
                forecast_case,
                record_sub,
                tail="min",
            )

            out_of_ensemble = truth_case < forecast_case.min("number", skipna=True)

        ens_mean = forecast_case.mean("number", skipna=True)
        ens_median = forecast_case.median("number", skipna=True)

        err_mean = ens_mean - truth_case
        err_median = ens_median - truth_case

        q05 = forecast_case.quantile(0.05, dim="number", skipna=True)
        q95 = forecast_case.quantile(0.95, dim="number", skipna=True)

        coverage_90 = (truth_case >= q05) & (truth_case <= q95)

        finite = (
            margin.notnull()
            & truth_case.notnull()
            & record_sub.notnull()
            & crps_case.notnull()
            & record_probability.notnull()
            & err_mean.notnull()
            & err_median.notnull()
        )

        coords = {
            "leadtime": [lead_hour],
            "landsea": landsea_names,
            "margin": margins,
        }

        dims_out = ("leadtime", "landsea", "margin")
        shape_out = (1, len(landsea_names), len(margins))

        ds = xr.Dataset(
            data_vars={
                "n": (dims_out, np.full(shape_out, np.nan)),
                "weight": (dims_out, np.full(shape_out, np.nan)),
                "mean_margin": (dims_out, np.full(shape_out, np.nan)),
                "crps": (dims_out, np.full(shape_out, np.nan)),
                "rmse_mean": (dims_out, np.full(shape_out, np.nan)),
                "bias_mean": (dims_out, np.full(shape_out, np.nan)),
                "rmse_median": (dims_out, np.full(shape_out, np.nan)),
                "bias_median": (dims_out, np.full(shape_out, np.nan)),
                "record_probability": (dims_out, np.full(shape_out, np.nan)),
                "out_of_ensemble_rate": (dims_out, np.full(shape_out, np.nan)),
                "coverage_90": (dims_out, np.full(shape_out, np.nan)),
            },
            coords=coords,
        )

        tasks = []
        keys = []

        # This is for all months. If we later want seasonal diagnostics, this is where we would first
        # subset margin, CRPS, errors and masks to DJF/MAM/JJA/SON.
        for k, landsea in enumerate(landsea_names):

            valid_landsea = finite

            if landsea == "land":
                valid_landsea = valid_landsea & (lsm_sub > 0.5)

            if landsea == "sea":
                valid_landsea = valid_landsea & (lsm_sub <= 0.5)

            for u in margins:

                valid = valid_landsea & (margin > u)
                dims = list(valid.dims)
                n = valid.sum(dim=dims)
                weight = weights.where(valid).sum(dim=dims)

                mean_margin = (
                    (margin * weights).where(valid).sum(dim=dims)
                    / weight
                )

                crps_mean = (
                    (crps_case * weights).where(valid).sum(dim=dims)
                    / weight
                )

                mse_mean = (
                    ((err_mean ** 2) * weights).where(valid).sum(dim=dims)
                    / weight
                )

                bias_mean = (
                    (err_mean * weights).where(valid).sum(dim=dims)
                    / weight
                )

                mse_median = (
                    ((err_median ** 2) * weights).where(valid).sum(dim=dims)
                    / weight
                )

                bias_median = (
                    (err_median * weights).where(valid).sum(dim=dims)
                    / weight
                )

                prob_mean = (
                    (record_probability * weights).where(valid).sum(dim=dims)
                    / weight
                )

                outens_rate = (
                    (out_of_ensemble.astype("float32") * weights)
                    .where(valid)
                    .sum(dim=dims)
                    / weight
                )

                coverage_rate = (
                    (coverage_90.astype("float32") * weights)
                    .where(valid)
                    .sum(dim=dims)
                    / weight
                )

                tasks.extend(
                    [
                        n,
                        weight,
                        mean_margin,
                        crps_mean,
                        mse_mean,
                        bias_mean,
                        mse_median,
                        bias_median,
                        prob_mean,
                        outens_rate,
                        coverage_rate,
                    ]
                )

                keys.append((k, u))

        computed = dask.compute(*tasks)

        pos = 0

        for k, u in keys:
            n_val = functions.scalar(computed[pos])
            weight_val = functions.scalar(computed[pos + 1])
            mean_margin_val = functions.scalar(computed[pos + 2])
            crps_val = functions.scalar(computed[pos + 3])
            mse_mean_val = functions.scalar(computed[pos + 4])
            bias_mean_val = functions.scalar(computed[pos + 5])
            mse_median_val = functions.scalar(computed[pos + 6])
            bias_median_val = functions.scalar(computed[pos + 7])
            prob_val = functions.scalar(computed[pos + 8])
            outens_val = functions.scalar(computed[pos + 9])
            cover_val = functions.scalar(computed[pos + 10])

            pos += 11

            if np.isfinite(mse_mean_val):
                rmse_mean_val = np.sqrt(mse_mean_val)
            else:
                rmse_mean_val = np.nan

            if np.isfinite(mse_median_val):
                rmse_median_val = np.sqrt(mse_median_val)
            else:
                rmse_median_val = np.nan

            loc = dict(
                leadtime=lead_hour,
                landsea=landsea_names[k],
                margin=u,
            )

            ds["n"].loc[loc] = n_val
            ds["weight"].loc[loc] = weight_val
            ds["mean_margin"].loc[loc] = mean_margin_val
            ds["crps"].loc[loc] = crps_val
            ds["rmse_mean"].loc[loc] = rmse_mean_val
            ds["bias_mean"].loc[loc] = bias_mean_val
            ds["rmse_median"].loc[loc] = rmse_median_val
            ds["bias_median"].loc[loc] = bias_median_val
            ds["record_probability"].loc[loc] = prob_val
            ds["out_of_ensemble_rate"].loc[loc] = outens_val
            ds["coverage_90"].loc[loc] = cover_val

        outputs_by_case[case].append(ds)

    del forecast
    gc.collect()

print("\nCombining and saving outputs...")
for case in cases:

    ds = xr.concat(outputs_by_case[case], dim="leadtime").sortby("leadtime")

    ds.attrs["verification_target"] = truth_name
    ds.attrs["record_tag"] = record_tag
    ds.attrs["domain"] = "no_Antarctic"
    ds.attrs["margin_max"] = args.margin_max
    ds.attrs["margin_step"] = args.margin_step
    ds.attrs["season"] = "all"

    if var == "t2m" and case == "max":
        ds.attrs["event"] = "heat"
    elif var == "t2m" and case == "min":
        ds.attrs["event"] = "cold"
    elif var == "w10" and case == "max":
        ds.attrs["event"] = "wind"

    out_file = out_dir / (
        f"{model}_{case}_margin_{year}_{var}_"
        f"no_Antarctic_records_{record_tag}_cond_{truth_name}.nc"
    )

    ds.to_netcdf(out_file)

    print(f"Saved {case}:")
    print(f"  {out_file}")

print("\nDone.")