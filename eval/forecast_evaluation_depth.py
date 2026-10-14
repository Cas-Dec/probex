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
import json
import os
import socket
from datetime import datetime
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

import functions
import functions_crps
import functions_occurrence

MODEL_CONFIG = functions.MODEL_CONFIG
VARIABLES = functions.VARIABLES
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
    hours = functions.lead_hours(forecast)
    if hours != lead:
        raise ValueError(f"Stored lead {hours} does not match {lead} h.")
    forecast = forecast.isel(prediction_timedelta=0, drop=True)
    if set(forecast.dims) != {"time", "number", *SPATIAL}:
        raise ValueError(f"Unexpected forecast dimensions: {forecast.dims}")
    times = forecast.time.values.astype("datetime64[ns]") + np.timedelta64(lead, "h")
    if np.isnat(times).any() or len(np.unique(times)) != len(times):
        raise ValueError("Missing or duplicate forecast times.")
    return forecast.assign_coords(time=times).sortby("time")


def validate_forecast_consistency(forecast, model, variable, lead):
    """
    Strictly validate the forecast grid and time stamps. Dimensions, member count,
    finite coordinates and unique times are already checked by normalize_grid and
    forecast_at_valid_time. AWG/uCast may have an extra -60°S latitude row.
    """
    issues = []

    # Check latitude range
    lat = forecast.latitude.values
    lat_min, lat_max = lat.min(), lat.max()

    # Allowed: -60 to 90 (AWG/uCast) OR -58.5 to 90 (others)
    if model in ["ArchesWeatherGen", "uCast"]:
        if not (np.isclose(lat_min, -60.0, atol=0.1) and np.isclose(lat_max, 90.0, atol=0.1)):
            issues.append(
                f"Unexpected latitude range for {model}: [{lat_min:.2f}, {lat_max:.2f}]\n"
                f"  Expected: [-60.0, 90.0]"
            )
    else:
        if not (np.isclose(lat_min, -58.5, atol=0.1) and np.isclose(lat_max, 90.0, atol=0.1)):
            issues.append(
                f"Unexpected latitude range for {model}: [{lat_min:.2f}, {lat_max:.2f}]\n"
                f"  Expected: [-58.5, 90.0]"
            )

    # Check longitude range (all models should be -180 to 178.5)
    lon = forecast.longitude.values
    lon_min, lon_max = lon.min(), lon.max()
    if not (np.isclose(lon_min, -180.0, atol=0.1) and np.isclose(lon_max, 178.5, atol=0.5)):
        issues.append(
            f"Unexpected longitude range: [{lon_min:.2f}, {lon_max:.2f}]\n"
            f"  Expected: [-180.0, 178.5]"
        )

    # Check grid spacing (should be 1.5° for all at this stage)
    lat_spacing = np.diff(lat)
    lon_spacing = np.diff(lon)
    expected_spacing = 1.5

    if not np.allclose(lat_spacing, expected_spacing, atol=0.01):
        issues.append(
            f"Non-uniform latitude spacing: {np.unique(np.round(lat_spacing, 3))}\n"
            f"  Expected uniform: {expected_spacing}°"
        )

    if not np.allclose(lon_spacing, expected_spacing, atol=0.01):
        issues.append(
            f"Non-uniform longitude spacing: {np.unique(np.round(lon_spacing, 3))}\n"
            f"  Expected uniform: {expected_spacing}°"
        )

    # Check that times are on-hour (no minutes/seconds)
    times = forecast.time.values
    times_as_hour = times.astype('datetime64[h]')
    if not (times == times_as_hour).all():
        issues.append("Time coordinates not on exact hours")

    if issues:
        raise ValueError(
            f"Forecast consistency validation FAILED for {model}, {variable}, lead {lead}h:\n" +
            "\n".join(f"  ❌ {issue}" for issue in issues) +
            f"\n\nThese are RED FLAGS indicating data preprocessing problems."
        )

    print(f"    ✓ Forecast consistency validated: {model} {variable} {lead}h")


def select_viable_depths(truth, history, case, candidate_depths, min_events=30):
    """
    Select depth levels that have at least min_events exceedances.

    Returns:
        viable_depths: list of depth values meeting sample size criterion
        sample_counts: dict mapping depth -> number of events
    """
    # Transform records to time-indexed
    record = functions_occurrence.record_by_time(history.record, truth.time.values)
    scale = functions_occurrence.record_by_time(history.scale, truth.time.values)

    sign = 1 if case == "max" else -1
    k = xr.DataArray(candidate_depths, dims="depth", coords={"depth": candidate_depths})
    event = sign * (truth - (record + sign * k * scale)) > 0
    # One pass over truth for all depths; count valid events (where scale > 0)
    counts = (event & (scale > 0) & truth.notnull()).sum(REDUCE_DIMS).compute()
    sample_counts = dict(zip(candidate_depths, counts.values.tolist()))
    viable_depths = [k for k, n in sample_counts.items() if n >= min_events]
    for k, n in sample_counts.items():
        if n < min_events:
            print(f"  Excluding depth {k:.2f}σ: only {n} events (< {min_events} required)")

    if not viable_depths:
        raise ValueError(
            f"No depth levels have sufficient sample size (>= {min_events} events)!\n"
            f"Sample counts: {sample_counts}"
        )

    print(f"  Selected depths: {viable_depths}")
    print(f"  Sample counts: {', '.join([f'{k:.2f}σ: {sample_counts[k]}' for k in viable_depths])}")

    return viable_depths, sample_counts


def validate_scores(scores: xr.Dataset, case: str):
    """
    Validate computed scores are within expected ranges.

    Raises ValueError if any validation fails.
    """
    issues = []

    # CRPS must be non-negative
    for crps_var in ['crps_all', 'crps_record', 'crps_empirical_all', 'crps_empirical_record']:
        if crps_var in scores.data_vars and (scores[crps_var] < 0).any():
            n_neg = int((scores[crps_var] < 0).sum().item())
            issues.append(f"{crps_var}: {n_neg} negative values")

    # RMSE must be non-negative
    for rmse_var in ['rmse_all', 'rmse_record']:
        if rmse_var in scores.data_vars and (scores[rmse_var] < 0).any():
            n_neg = int((scores[rmse_var] < 0).sum().item())
            issues.append(f"{rmse_var}: {n_neg} negative values")

    # Brier score must be in [0, 1]
    for brier_var in ['brier', 'brier_empirical']:
        if brier_var in scores.data_vars:
            if (scores[brier_var] < 0).any() or (scores[brier_var] > 1).any():
                n_invalid = int(((scores[brier_var] < 0) | (scores[brier_var] > 1)).sum().item())
                issues.append(f"{brier_var}: {n_invalid} values outside [0, 1]")

    # rtwCRPS must be non-negative
    for rtwcrps_var in ['rtwcrps', 'rtwcrps_empirical']:
        if rtwcrps_var in scores.data_vars and (scores[rtwcrps_var] < 0).any():
            n_neg = int((scores[rtwcrps_var] < 0).sum().item())
            issues.append(f"{rtwcrps_var}: {n_neg} negative values")

    # Probability must be in [0, 1]
    for prob_var in ['probability', 'observed_probability']:
        if prob_var in scores.data_vars:
            if (scores[prob_var] < 0).any() or (scores[prob_var] > 1).any():
                n_invalid = int(((scores[prob_var] < 0) | (scores[prob_var] > 1)).sum().item())
                issues.append(f"{prob_var}: {n_invalid} values outside [0, 1]")

    # Counts must be non-negative
    for count_var in ['valid_count', 'record_count', 'excluded_scale_count']:
        if count_var in scores.data_vars and (scores[count_var] < 0).any():
            n_neg = int((scores[count_var] < 0).sum().item())
            issues.append(f"{count_var}: {n_neg} negative values")

    # Weights must be non-negative
    for weight_var in ['valid_weight', 'record_weight']:
        if weight_var in scores.data_vars and (scores[weight_var] < 0).any():
            n_neg = int((scores[weight_var] < 0).sum().item())
            issues.append(f"{weight_var}: {n_neg} negative values")

    # Check conservation: observed_probability should match record_weight / valid_weight
    if 'observed_probability' in scores.data_vars and 'record_weight' in scores.data_vars and 'valid_weight' in scores.data_vars:
        computed_obs_prob = scores.record_weight / scores.valid_weight.where(scores.valid_weight > 0)
        diff = abs(scores.observed_probability - computed_obs_prob)
        if (diff > 1e-6).any():
            n_mismatch = int((diff > 1e-6).sum().item())
            issues.append(f"observed_probability: {n_mismatch} values don't match record_weight/valid_weight")

    if issues:
        raise ValueError(
            f"Score validation failed for case '{case}':\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )


FAIR_CRPS_REF = "Ferro (2014), doi:10.1002/qj.2270"
RTW_CRPS_REF = "Gneiting and Ranjan (2011), doi:10.1198/jbes.2010.08110"


def variable_attrs(name, units, case):
    """Explicit per-variable metadata. Replaces whatever attrs leaked in from the
    land-sea mask, the records or the truth field during the computation."""
    tail = f"beyond record {'+' if case == 'max' else '-'} depth * scale"
    table = {
        "valid_count": ("1", "number of eligible (time, cell) cases (scale > 0)", None),
        "valid_weight": ("1", "sum of cos(latitude) weights over eligible cases", None),
        "record_count": ("1", f"number of eligible cases with truth {tail}", None),
        "record_weight": ("1", f"sum of cos(latitude) weights over cases with truth {tail}", None),
        "excluded_scale_count": ("1", "number of cases dropped because scale <= 0", None),
        "crps_all": (units, "CRPS over all eligible cases", "fair ensemble CRPS"),
        "crps_record": (units, f"CRPS over cases with truth {tail}", "fair ensemble CRPS"),
        "crps_empirical_all": (units, "CRPS over all eligible cases", "empirical (non-fair) ensemble CRPS"),
        "crps_empirical_record": (units, f"CRPS over cases with truth {tail}", "empirical (non-fair) ensemble CRPS"),
        "bias_all": (units, "ensemble-mean bias (forecast - truth) over all eligible cases", None),
        "bias_record": (units, f"ensemble-mean bias (forecast - truth) over cases with truth {tail}", None),
        "rmse_all": (units, "ensemble-mean RMSE over all eligible cases", None),
        "rmse_record": (units, f"ensemble-mean RMSE over cases with truth {tail}", None),
        "rtwcrps": (units, "record-threshold weighted CRPS over all eligible cases", "fair rtwCRPS"),
        "rtwcrps_empirical": (units, "record-threshold weighted CRPS over all eligible cases", "empirical (non-fair) rtwCRPS"),
        "brier": ("1", f"Brier score for truth {tail}, over all eligible cases", "fair Brier score"),
        "brier_empirical": ("1", f"Brier score for truth {tail}, over all eligible cases", "empirical (non-fair) Brier score"),
        "rtwcrps_record_contribution": (units, "Observed-exceedance contribution to all-case rtwCRPS", "fair rtwCRPS"),
        "rtwcrps_nonrecord_contribution": (units, "Non-exceedance contribution to all-case rtwCRPS", "fair rtwCRPS"),
        "brier_record_contribution": ("1", "Observed-exceedance contribution to all-case Brier score", "fair Brier score"),
        "brier_nonrecord_contribution": ("1", "Non-exceedance contribution to all-case Brier score", "fair Brier score"),
        "probability": ("1", f"mean forecast probability of truth {tail}", None),
        "observed_probability": ("1", f"observed frequency of truth {tail} (record_weight / valid_weight)", None),
    }
    units, long_name, estimator = table[name]
    attrs = {"units": units, "long_name": long_name}
    if estimator:
        attrs["estimator"] = estimator
    if "crps" in name:
        attrs["reference"] = f"{RTW_CRPS_REF}; {FAIR_CRPS_REF}" if "rtw" in name else FAIR_CRPS_REF
    if "rtwcrps" in name:
        attrs["tail"] = case
    return attrs


def load_fingerprint_registry(registry_path):
    """Load existing fingerprint registry or create empty one."""
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {}


def update_fingerprint_registry(registry, key, fingerprint, metadata):
    """Update registry with new or verified fingerprint."""
    if key in registry:
        entry = registry[key]
        if entry["fingerprint"] != fingerprint:
            raise ValueError(
                f"Fingerprint mismatch for {key}!\n"
                f"  Expected: {entry['fingerprint']}\n"
                f"  Got:      {fingerprint}\n"
                f"  Previous: {entry['date']} by {entry.get('model', 'unknown')}\n"
                f"Historical records may have changed. Verify before comparing results!"
            )
        print(f"  ✓ Fingerprint verified for {key}")
    else:
        registry[key] = {
            "fingerprint": fingerprint,
            "date": datetime.now().isoformat(),
            **metadata
        }
        print(f"  ✓ Fingerprint registered for {key}")

    return registry


def sync_fingerprints(registry_path, fingerprints, metadata, write=False):
    """Check fingerprints against the shared registry; with write=True also store new ones.
    Array tasks share the file, so re-read it right before writing and replace it atomically."""
    registry = load_fingerprint_registry(registry_path)
    for key, fingerprint in fingerprints.items():
        update_fingerprint_registry(registry, key, fingerprint, metadata)
    if write:
        # shortcut: no file lock, two tasks writing in the same instant can drop one new entry; it is re-registered on the next run.
        partial = registry_path.with_name(f"{registry_path.name}.{socket.gethostname()}.{os.getpid()}.partial")
        partial.write_text(json.dumps(registry, indent=2))
        os.replace(partial, registry_path)


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
        if name in {"rtwcrps", "brier"}:
            # Keep the ALL-case denominator: the two contributions sum to result[name].
            for subset, mask in [("record", event), ("nonrecord", ~event)]:
                result[f"{name}_{subset}_contribution"] = (
                    xr.where(eligible & mask, field * weight, 0).sum(REDUCE_DIMS, skipna=False) / all_weight
                )
    result["observed_probability"] = result.record_weight / all_weight
    return result


def main(argv=None):
    root = Path.cwd()
    functions.load_dotenv(root / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--var", choices=list(VARIABLES), default="t2m")
    parser.add_argument("--model", choices=list(MODEL_CONFIG), default="FGN")
    parser.add_argument("--truth", choices=["ERA5", "HRES_fc0"], default="ERA5")
    parser.add_argument("--record-tag", default="1979_2021")
    parser.add_argument("--depths", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3])
    parser.add_argument("--min-sample-size", type=int, default=30, help="Minimum number of events required to include a depth level (default: 30)")
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
        check_units(truth, args.var)

        # Metadata-only pass: use the same dates at every requested lead/model.
        # Grids need not match exactly: xr.align with join="inner" will reconcile them.
        for model in peers:
            for lead in leads:
                with xr.open_zarr(forecast_path(model, lead), chunks={}) as source:
                    field = forecast_at_valid_time(source, variable, lead, MODEL_CONFIG[model]["n_members"])
                    validate_forecast_consistency(field, model, variable, lead)
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
            # create_era5_records.py already refuses scale <= 0; this guards against older files.
            if (hist.scale <= 0).any():
                raise ValueError(f"Historical scale <= 0 in {path}; rebuild it with download/create_era5_records.py")
            histories[case] = hist

        # Check the record fingerprints now, so a mismatch fails before hours of compute.
        fingerprints = {}
        for case, hist in histories.items():
            reference = hashlib.sha256()
            for field in [hist.record, hist.scale, lsm, hist.latitude, hist.longitude]:
                reference.update(np.ascontiguousarray(field.values, dtype="float64").tobytes())
            fingerprints[case] = reference.hexdigest()
        fingerprint_registry_path = args.out_root / "depth_fingerprints.json"
        registry_entries = {f"{args.record_tag}_{args.var}_{case}": fp for case, fp in fingerprints.items()}
        fingerprint_metadata = {"model": args.model, "truth": args.truth, "year": args.year}
        sync_fingerprints(fingerprint_registry_path, registry_entries, fingerprint_metadata)

        # Adaptive depth selection per case
        print("\nSelecting viable depths based on sample size criterion:")
        candidate_depths = sorted(set(args.depths))
        depths_by_case = {}
        for case, hist in histories.items():
            print(f"  Case: {case}")
            viable_depths, sample_counts = select_viable_depths(
                truth, hist, case, candidate_depths, min_events=args.min_sample_size
            )
            depths_by_case[case] = viable_depths

        outputs = {case: [] for case in histories}
        for index, lead in enumerate(leads, start=1):
            print(f"[{index}/{len(leads)}] {args.model} {args.var}, {lead} h", flush=True)
            with xr.open_zarr(forecast_path(args.model, lead), chunks={}) as source:
                forecast = forecast_at_valid_time(source, variable, lead, MODEL_CONFIG[args.model]["n_members"])
                validate_forecast_consistency(forecast, args.model, variable, lead)
                forecast, truth = xr.align(forecast, truth, join="inner")
                forecast = forecast.sel(time=times).astype("float64").chunk({"time": 31, "number": -1, "latitude": 30, "longitude": 60})
                # compute ordinary scores for all cases again
                error = forecast.mean("number") - truth # bias; square, average and take the square root for RMSE
                crps = functions_crps.ensemble_crps_sorted(forecast - truth, xr.zeros_like(truth)) # this is a numerical convenience, since CRPS (X,Y) = CRPS (X-Y, 0)
                m = forecast.sizes["number"]
                base = {"crps": crps, "crps_empirical": (m - 1) / m * crps + abs(forecast - truth).mean("number") / m,
                        "bias": error, "squared_error": error ** 2}
                # Use case-specific viable depths
                tasks = [depth_scores(forecast, truth, hist, surfaces, depths_by_case[case], case, base) for case, hist in histories.items()]
                with dask.config.set(scheduler="threads", num_workers=4), ProgressBar(dt=1, minimum=3):
                    computed = dask.compute(*tasks)
            for case, scores in zip(histories, computed):
                if scores.invalid_count.sum():
                    raise ValueError("Nonfinite forecast/truth at eligible cases; no results saved.")
                if float(scores.valid_weight.sel(landsea="all").sum()) <= 0:
                    raise ValueError("No positive-weight cases with positive historical scale.")
                # Validate score ranges
                validate_scores(scores, case)
                outputs[case].append(scores.drop_vars("invalid_count").expand_dims(leadtime=[lead]))

        for case, parts in outputs.items():
            summary = xr.concat(parts, dim="leadtime", join="exact")
            summary = summary.transpose("leadtime", "landsea", "depth")
            event = "cold" if case == "min" else ("heat" if args.var == "t2m" else "wind")
            attrs = {"model": args.model, "event": event, "variable": args.var, "year": args.year,
                     "verification_target": args.truth, "truth_file": str(truth_file.resolve()),
                     "record_tag": args.record_tag, "reference_fingerprint": fingerprints[case],
                     "valid_time_count": len(times), "common_dates_across_leads": "true",
                     "comparison_models": ",".join(peers), "units": "K" if args.var == "t2m" else "m s-1",
                     "min_sample_size": args.min_sample_size,
                     "selected_depths": ",".join([f"{d:.2f}" for d in depths_by_case[case]])}
            stem = f"{args.model}_{case}_depth_{args.year}_{args.var}_no_Antarctic_records_{args.record_tag}_{args.truth}"
            summary.to_dataframe().reset_index().to_csv(out_dir / f"{stem}.csv", index=False)
            summary = summary.assign_coords(verification_time=times)
            summary.attrs = attrs
            for name in summary.data_vars:
                summary[name].attrs = variable_attrs(name, attrs["units"], case)
            output = out_dir / f"{stem}.nc"
            partial = output.with_suffix(".nc.partial")
            summary.to_netcdf(partial, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4} for name in summary.data_vars})
            os.replace(partial, output)
            print(f"Saved {output}")

        sync_fingerprints(fingerprint_registry_path, registry_entries, fingerprint_metadata, write=True)
        print(f"\n✓ Fingerprint registry updated: {fingerprint_registry_path}")


if __name__ == "__main__":
    main()