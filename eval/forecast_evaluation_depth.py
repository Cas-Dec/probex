"""Evaluate one model/variable at R +/- k*s. Reads forecasts, verification data,
historical records and scales and calculates scores at increasingly extreme thresholds. 

Fair rtwCRPS and Brier use all cases. Record-conditioned CRPS/RMSE are
secondary diagnostics. Save the final summary as NetCDF and CSV.

API:  e.g.
python eval/forecast_evaluation_depth.py \
    --model FGN \
    --var t2m \
    --truth ERA5 \
    --compare-models IFS_ENS FGN \
    --leads 12 48 120 240 \
    --depths 0 0.5 1 2 # can go even further but be aware that this depletes your sample size!
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

import functions
import functions_crps
import functions_occurrence

MODEL_CONFIG = {
    "GenCast": {"leads": list(range(12, 241, 12)), "n_members": 50, "resolution_tag": "0p25"}, # for now at least...
    "IFS_ENS": {"leads": [12, 48, 120, 168, 240], "n_members": 50, "resolution_tag": "1p50"},
    "ArchesWeatherGen": {"leads": [48, 120, 168, 240], "n_members": 64, "resolution_tag": "1p50"},
    "uCast": {"leads": [12, 48, 120, 168, 240], "n_members": 64, "resolution_tag": "1p50"},
    "FGN": {"leads": [12, 48, 120, 168, 240], "n_members": 64, "resolution_tag": "1p50"},
}
VARIABLES = {"t2m": "2m_temperature", "w10": "10m_wind_speed"}
SPATIAL = ["latitude", "longitude"]
REDUCE_DIMS = ["time", *SPATIAL]
ERA5_SOURCE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# rename coordinates and convert to -180..180 longitude, sort lat/lon, check for duplicates and nonfinite values
def normalize_grid(data):
    data = data.rename({a: b for a, b in [("lat", "latitude"), ("lon", "longitude")] if a in data.dims})
    data = data.sortby("latitude").sel(latitude=slice(-60, 90))
    data = data.assign_coords(longitude=(data.longitude + 180) % 360 - 180)
    for dim in SPATIAL:
        values = np.round(data[dim].values.astype(float), 6)
        if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f"Invalid or duplicate {dim} coordinates.")
        data = data.assign_coords({dim: values}).sortby(dim)
    return data


# temperature in K and WS in m/s
def check_units(data, var, required=False):
    units = str(data.attrs.get("units") or "").lower().replace(" ", "").replace("**", "^")
    allowed = {"k", "kelvin"} if var == "t2m" else {"ms-1", "ms^-1", "m/s"}
    if (required and not units) or (units and units not in allowed):
        raise ValueError(f"{data.name}: unexpected/missing units {data.attrs.get('units')!r}.")

# make sure if we initiate at 1/1 0h a lead time 48h, we compare with 3/1 0h
def forecast_at_valid_time(dataset, variable, lead, members):
    forecast = normalize_grid(dataset[variable])
    if forecast.sizes.get("prediction_timedelta") != 1 or forecast.sizes.get("number") != members:
        raise ValueError("Unexpected lead dimension or ensemble size.")
    stored = forecast.prediction_timedelta.values[0]
    hours = stored / np.timedelta64(1, "h") if isinstance(stored, np.timedelta64) else stored
    if float(hours) != lead:
        raise ValueError(f"Stored lead {hours} does not match {lead} h.")
    forecast = forecast.isel(prediction_timedelta=0, drop=True)
    if set(forecast.dims) != {"time", "number", *SPATIAL}:
        raise ValueError(f"Unexpected forecast dimensions: {forecast.dims}")
    times = forecast.time.values.astype("datetime64[ns]") + np.timedelta64(lead, "h")
    if np.isnat(times).any() or len(np.unique(times)) != len(times):
        raise ValueError("Missing or duplicate forecast times.")
    return forecast.assign_coords(time=times).sortby("time")


def depth_scores(forecast, truth, history, surfaces, depths, case, base):
    """ This is the core of the depth evaluation, ie for every cell and verification time, we compute the threshold R +/- k*s
    where k is threshold depth, R is the historical record and s is the historical scale.
    Then we compute the scores for all cases as well as for record-conditioned cases. """
    chunks = {"time": 31, "latitude": 30, "longitude": 60}
    record = functions_occurrence.record_by_time(history.record, truth.time.values).chunk(chunks)
    scale = functions_occurrence.record_by_time(history.scale, truth.time.values).chunk(chunks)
    sign = 1 if case == "max" else -1
    k = xr.DataArray(depths, dims="depth", coords={"depth": depths}) # threshold depth
    threshold = record + sign * k * scale
    # select cases where sign * (Y_i-R_i)/s_i > k
    event = sign * (truth - threshold) > 0
    eligible = surfaces & (scale > 0)
    weights = np.maximum(np.cos(np.deg2rad(truth.latitude)), 0)
    weight = xr.where(eligible, weights, 0)
    event_weight = xr.where(event, weight, 0)

    result = xr.Dataset({
        "valid_count": eligible.sum(REDUCE_DIMS),
        "valid_weight": weight.sum(REDUCE_DIMS),
        "record_count": (eligible & event).sum(REDUCE_DIMS),
        "record_weight": event_weight.sum(REDUCE_DIMS),
        "excluded_scale_count": (surfaces & (scale <= 0)).sum(REDUCE_DIMS),
        "invalid_count": (eligible & (~np.isfinite(truth) | ~np.isfinite(forecast).all("number"))).sum(REDUCE_DIMS),
    })
    all_weight = result.valid_weight.where(result.valid_weight > 0)
    record_weight = result.record_weight.where(result.record_weight > 0)

    # Average ordinary scores over all cases and over observed exceedances.
    for name, field in base.items():
        for subset, mask, denominator in [("all", eligible, all_weight),
                                          ("record", eligible & event, record_weight)]:
            value = xr.where(mask, field * weight, 0).sum(REDUCE_DIMS, skipna=False) / denominator
            score = "rmse" if name == "squared_error" else name
            result[f"{score}_{subset}"] = np.sqrt(value) if name == "squared_error" else value

    x, y = forecast - threshold, truth - threshold
    tail_crps = functions_crps.record_threshold_weighted_crps_sorted(x, y, xr.zeros_like(threshold), case)
    p = (sign * x > 0).mean("number")
    m = forecast.sizes["number"]
    brier = (p - event) ** 2
    first = abs(xr.where(sign * x > 0, sign * x, 0) - xr.where(sign * y > 0, sign * y, 0)).mean("number")
    tail = {"rtwcrps": tail_crps, "rtwcrps_empirical": (m - 1) / m * tail_crps + first / m,
            "brier": brier - p * (1 - p) / (m - 1), "brier_empirical": brier, "probability": p}

    # Tail scores use all  cases, including non-exceedances.
    for name, field in tail.items():
        result[name] = xr.where(eligible, field * weight, 0).sum(REDUCE_DIMS, skipna=False) / all_weight
    result["observed_probability"] = result.record_weight / all_weight
    return result


def main(argv=None):
    root = Path.cwd()
    if (root / ".env").is_file():
        for line in (root / ".env").read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--var", choices=list(VARIABLES), default="t2m")
    parser.add_argument("--model", choices=list(MODEL_CONFIG), default="FGN")
    parser.add_argument("--truth", choices=["ERA5", "HRES_fc0"], default="ERA5")
    parser.add_argument("--record-tag", default="1979_2021")
    parser.add_argument("--depths", nargs="+", type=float, default=[0, 0.5, 1, 2])
    parser.add_argument("--leads", nargs="+", type=int, help="Default: the model's existing lead list.")
    parser.add_argument("--compare-models", nargs="+", choices=list(MODEL_CONFIG), help="Match valid dates with these models before scoring this model.")
    parser.add_argument("--resolution-tag", choices=["0p25", "1p50"], help="Filename tag for prepared inputs; this does not regrid data.")
    parser.add_argument("--forecast-root", type=Path, default=os.environ.get("FORECAST_ROOT", "data/forecasts"))
    parser.add_argument("--records-root", type=Path, default=os.environ.get("RECORDS_ROOT", "data/era5_records"))
    parser.add_argument("--truth-file", type=Path, help="Override the configured verification store.")
    parser.add_argument("--out-root", type=Path, default=root / "results")
    args = parser.parse_args(argv)
    leads = sorted(set(args.leads or MODEL_CONFIG[args.model]["leads"]))
    depths = sorted(set(args.depths))
    peers = list(dict.fromkeys([args.model] + (args.compare_models or [])))
    start_year, end_year = map(int, args.record_tag.split("_"))
    if not start_year < end_year < args.year or min(leads) <= 0:
        parser.error("Require a historical baseline of at least two years and positive leads.")
    if min(depths) < 0 or not np.isfinite(depths).all():
        parser.error("Depths must be finite and nonnegative.")

    def forecast_path(model, lead):
        config = MODEL_CONFIG[model]
        resolution = args.resolution_tag or config["resolution_tag"]
        return args.forecast_root / model / f"{model}_{args.year}_no_Antarctic_{args.var}_m{config['n_members']}_{resolution}_lead_{lead:03d}h.zarr"

    variable = VARIABLES[args.var]
    era5_default = Path(os.environ.get("ERA5_ROOT", "data/wb2")) / ERA5_SOURCE.rsplit("/", 1)[-1]
    truth_file = args.truth_file or (Path(os.environ.get("ERA5_TRUTH_ZARR", str(era5_default))) if args.truth == "ERA5"
                                   else args.forecast_root / "HRES_fc0" / f"HRES_fc0_{args.year}_no_Antarctic_{args.var}_1p50.zarr")
    out_dir = args.out_root / args.model / "Depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Depth evaluation: {args.model}, {args.var}, {args.year}, truth={args.truth}\nLeads: {leads}; depths: {depths}", flush=True)

    with xr.open_zarr(truth_file, chunks={}) as truth_store:
        truth = normalize_grid(truth_store[variable]).sel(time=slice(f"{args.year}-01-01", f"{args.year}-12-31T23:59:59"))
        truth = truth.where(truth.time.dt.hour.isin([0, 12]), drop=True)
        times = truth.time.values.astype("datetime64[ns]")
        if np.isnat(times).any() or len(np.unique(times)) != len(times) or np.any(times != times.astype("datetime64[h]")):
            raise ValueError("Missing, duplicate or off-hour verification timestamps.")
        if not truth.attrs.get("units"):
            truth.attrs["units"] = "K" if args.var == "t2m" else "m s-1"
        check_units(truth, args.var, required=True)

        # Metadata-only pass: use the same dates at every requested lead/model.
        # Grids need not match exactly: xr.align with join="inner" will reconcile them.
        for model in peers:
            for lead in leads:
                with xr.open_zarr(forecast_path(model, lead), chunks={}) as source:
                    field = forecast_at_valid_time(source, variable, lead, MODEL_CONFIG[model]["n_members"])
                    check_units(field, args.var)
                    times = np.intersect1d(times, field.time.values)
        if not len(times):
            raise ValueError("No common valid dates.")
        truth = truth.sel(time=times).astype("float64").chunk({"time": 31, "latitude": 30, "longitude": 60})
        print(f"Using {len(times)} common valid times: {times[0]} to {times[-1]}", flush=True)

        with xr.open_dataset(args.records_root / "ERA5_lsm_sfc.nc") as source:
            lsm = normalize_grid(source.lsm.squeeze("time", drop=True) if "time" in source.lsm.dims else source.lsm).load()
        lsm, truth = xr.align(lsm, truth, join="inner")
        if not np.isfinite(lsm).all():
            raise ValueError("Nonfinite land-sea mask.")
        surfaces = xr.concat([xr.ones_like(lsm, dtype=bool), lsm > 0.5, lsm <= 0.5], dim=xr.IndexVariable("landsea", functions.LANDSEA_NAMES))
        histories = {}
        for case in (["max", "min"] if args.var == "t2m" else ["max"]):
            path = args.records_root / f"ERA5_record_{case}_month_{args.record_tag}_{args.var}.nc"
            with xr.open_dataset(path) as source:
                hist = normalize_grid(source).load()
            hist, truth = xr.align(hist, truth, join="inner")
            expected = {"record_tag": args.record_tag, "record_start_year": start_year, "record_end_year": end_year,
                        "record_hours_utc": "0,12", "extreme": case, "variable": args.var,
                        "scale_ddof": 1, "n_years": end_year - start_year + 1, "scale_detrended": "false"}
            if any(str(hist.attrs.get(key)) != str(value) for key, value in expected.items()) or not np.array_equal(hist.month, np.arange(1, 13)):
                raise ValueError(f"Historical metadata or months do not match: {path}")
            for name in ["record", "scale"]:
                check_units(hist[name], args.var, required=True)
                if set(hist[name].dims) != {"month", *SPATIAL} or not np.isfinite(hist[name]).all():
                    raise ValueError(f"Invalid dimensions or nonfinite historical {name}.")
            if (hist.scale < 0).any():
                raise ValueError("Negative historical scale.")
            histories[case] = hist

        outputs = {case: [] for case in histories}
        for index, lead in enumerate(leads, start=1):
            print(f"[{index}/{len(leads)}] {args.model} {args.var}, {lead} h", flush=True)
            with xr.open_zarr(forecast_path(args.model, lead), chunks={}) as source:
                forecast = forecast_at_valid_time(source, variable, lead, MODEL_CONFIG[args.model]["n_members"])
                forecast, truth = xr.align(forecast, truth, join="inner")
                forecast = forecast.sel(time=times).astype("float64").chunk({"time": 31, "number": -1, "latitude": 30, "longitude": 60})
                # compute ordinary scores for all cases again
                error = forecast.mean("number") - truth # bias; square, average and take the square root for RMSE
                crps = functions_crps.ensemble_crps_sorted(forecast - truth, xr.zeros_like(truth)) # this is a numerical convenience, since CRPS (X,Y) = CRPS (X-Y, 0)
                m = forecast.sizes["number"]
                base = {"crps": crps, "crps_empirical": (m - 1) / m * crps + abs(forecast - truth).mean("number") / m,
                        "bias": error, "squared_error": error ** 2}
                tasks = [depth_scores(forecast, truth, hist, surfaces, depths, case, base) for case, hist in histories.items()]
                with dask.config.set(scheduler="threads", num_workers=4), ProgressBar(dt=1, minimum=3):
                    computed = dask.compute(*tasks)
            for case, scores in zip(histories, computed):
                if scores.invalid_count.sum():
                    raise ValueError("Nonfinite forecast/truth at eligible cases; no results saved.")
                if float(scores.valid_weight.sel(landsea="all").sum()) <= 0:
                    raise ValueError("No positive-weight cases with positive historical scale.")
                outputs[case].append(scores.drop_vars("invalid_count").expand_dims(leadtime=[lead]))

        for case, parts in outputs.items():
            summary = xr.concat(parts, dim="leadtime", join="exact")
            event = "cold" if case == "min" else ("heat" if args.var == "t2m" else "wind")
            hist = histories[case]
            reference = hashlib.sha256()
            for field in [hist.record, hist.scale, lsm, hist.latitude, hist.longitude]:
                reference.update(np.ascontiguousarray(field.values, dtype="float64").tobytes())
            attrs = {"model": args.model, "event": event, "variable": args.var, "year": args.year,
                     "verification_target": args.truth, "truth_file": str(truth_file.resolve()),
                     "record_tag": args.record_tag, "reference_fingerprint": reference.hexdigest(),
                     "valid_time_count": len(times), "common_dates_across_leads": "true",
                     "comparison_models": ",".join(peers), "units": "K" if args.var == "t2m" else "m s-1"}
            stem = f"{args.model}_{case}_depth_{args.year}_{args.var}_no_Antarctic_records_{args.record_tag}_{args.truth}"
            summary.to_dataframe().reset_index().to_csv(out_dir / f"{stem}.csv", index=False)
            summary = summary.assign_coords(verification_time=times)
            summary.attrs = attrs
            output = out_dir / f"{stem}.nc"
            partial = output.with_suffix(".nc.partial")
            summary.to_netcdf(partial, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4} for name in summary.data_vars})
            os.replace(partial, output)
            print(f"Saved {output}")


if __name__ == "__main__":
    main()