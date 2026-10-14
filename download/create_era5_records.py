"""
Create ERA5 monthly record climatology files used by evaluation scripts.

Outputs (written into --out-root):
    ERA5_record_max_month_<record-tag>_t2m.nc
    ERA5_record_min_month_<record-tag>_t2m.nc
    ERA5_record_max_month_<record-tag>_w10.nc
    ERA5_lsm_sfc.nc

By default, records are computed from ERA5 years 1979-2021 and only
00/12 UTC timestamps to match forecast verification times.

Each record file contains record and scale on (month, lat, lon). The scale is
the sample SD (ddof=1) across yearly monthly maxima/minima, not across weather
values or monthly means. Remote ERA5 is reduced lazily; raw fields are never
saved locally.

An existing record file is kept only if it matches the output format of this
script exactly (variables, dims, coordinates, dtypes and attributes, which
encode the year window, hours, source and latitude bounds). Otherwise it is
recomputed and overwritten.
"""

from __future__ import annotations

import argparse
import os

# Match the existing downloaders on IPv4-only cluster networks.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from pathlib import Path
import tempfile

import dask
from dask.diagnostics import ProgressBar
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
    """Normalize coordinates to [-180, 180] longitude and sorted latitude."""
    out = da
    if "longitude" in out.coords:
        # Check for consistent longitude order before normalizing
        lon_diff = np.diff(out.longitude.values)
        if not (np.all(lon_diff > 0) or np.all(lon_diff < 0)):
            raise ValueError(
                "Longitude coordinates are not monotonic. "
                f"Found non-uniform spacing: {lon_diff[np.abs(lon_diff) > 2][:5]}"
            )
        out = out.assign_coords(
            longitude=(((out.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    if "latitude" in out.coords:
        # Check for consistent latitude order before sorting
        lat_diff = np.diff(out.latitude.values)
        if not (np.all(lat_diff > 0) or np.all(lat_diff < 0)):
            raise ValueError(
                "Latitude coordinates are not monotonic. "
                f"Found non-uniform spacing: {lat_diff[np.abs(lat_diff) > 2][:5]}"
            )
        out = out.sortby("latitude")
    return out


def _parse_hours(hours_csv: str) -> list[int]:
    return [int(h.strip()) for h in hours_csv.split(",") if h.strip()]


def _open_zarr(source: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    return xr.open_zarr(source, storage_options=storage_options, chunks={})


def _check_complete_times(
    times: xr.DataArray, start_year: int, end_year: int, hours: list[int]
) -> None:
    """Require every requested observation time before comparing yearly blocks."""
    days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    expected = pd.DatetimeIndex(
        (days.values[:, None] + np.asarray(hours, dtype="timedelta64[h]")).ravel()
    ).sort_values()
    actual = pd.DatetimeIndex(times.values)
    if not actual.equals(expected):
        missing = expected.difference(actual)
        raise ValueError(
            "Incomplete or non-unique historical timestamps for records/scales: "
            f"expected {len(expected)}, found {len(actual)}. "
            f"First missing times: {list(missing[:3])}."
        )


def _validate_grid(da: xr.DataArray, expected_spacing: float = 1.5) -> None:
    """Validate grid spacing and coordinate uniqueness."""
    if len(da.latitude) < 2 or len(da.longitude) < 2:
        raise ValueError(
            f"Insufficient grid points: lat={len(da.latitude)}, lon={len(da.longitude)}"
        )

    lat_spacing = np.abs(np.diff(da.latitude.values))
    lon_spacing = np.abs(np.diff(da.longitude.values))

    if not np.allclose(lat_spacing, expected_spacing, atol=0.01):
        raise ValueError(
            f"ERA5 latitude spacing is not {expected_spacing}°. "
            f"Found spacings: {lat_spacing[:5]}"
        )
    if not np.allclose(lon_spacing, expected_spacing, atol=0.01):
        raise ValueError(
            f"ERA5 longitude spacing is not {expected_spacing}°. "
            f"Found spacings: {lon_spacing[:5]}"
        )

    if len(np.unique(da.latitude)) != len(da.latitude):
        raise ValueError("Duplicate latitudes found in ERA5 source")
    if len(np.unique(da.longitude)) != len(da.longitude):
        raise ValueError("Duplicate longitudes found in ERA5 source")


def _validate_scale(scale: xr.DataArray, var_short: str, extreme: str, units: str) -> None:
    """Strictly validate that all scale values are finite and positive."""
    # Check for NaN or Inf
    if not np.isfinite(scale.values).all():
        n_invalid = int((~np.isfinite(scale.values)).sum())
        total = scale.size
        pct = 100 * n_invalid / total
        invalid_locs = np.where(~np.isfinite(scale.values))
        sample_info = []
        for i in range(min(5, len(invalid_locs[0]))):
            month_idx = invalid_locs[0][i]
            lat_idx = invalid_locs[1][i]
            lon_idx = invalid_locs[2][i]
            sample_info.append(
                f"    month={int(scale.month[month_idx])}, "
                f"lat={float(scale.lat[lat_idx]):.1f}, "
                f"lon={float(scale.lon[lon_idx]):.1f}, "
                f"scale={float(scale.values[month_idx, lat_idx, lon_idx])}"
            )
        raise ValueError(
            f"CRITICAL: Found {n_invalid}/{total} points ({pct:.2f}%) with non-finite scale!\n"
            f"Variable: {var_short} {extreme}\n"
            f"Zero tolerance for missing data in record scale computation.\n"
            f"First few invalid locations:\n" + "\n".join(sample_info) + "\n"
            f"Possible causes:\n"
            f"  - Incomplete historical data (missing timestamps)\n"
            f"  - Insufficient years for reliable statistics (need at least 2 years for ddof=1)\n"
            f"Check ERA5 source completeness for {var_short}"
        )

    # Check for scale <= 0
    if (scale <= 0).any():
        n_invalid = int((scale <= 0).sum())
        total = scale.size
        pct = 100 * n_invalid / total
        invalid_locs = np.where(scale <= 0)
        sample_info = []
        for i in range(min(5, len(invalid_locs[0]))):
            month_idx = invalid_locs[0][i]
            lat_idx = invalid_locs[1][i]
            lon_idx = invalid_locs[2][i]
            sample_info.append(
                f"    month={int(scale.month[month_idx])}, "
                f"lat={float(scale.lat[lat_idx]):.1f}, "
                f"lon={float(scale.lon[lon_idx]):.1f}, "
                f"scale={float(scale.values[month_idx, lat_idx, lon_idx]):.6f} {units}"
            )
        raise ValueError(
            f"CRITICAL: Found {n_invalid}/{total} points ({pct:.2f}%) with scale <= 0!\n"
            f"Variable: {var_short} {extreme}\n"
            f"Zero tolerance for invalid scale values.\n"
            f"First few invalid locations:\n" + "\n".join(sample_info) + "\n"
            f"Possible causes:\n"
            f"  - Constant climate (all yearly extremes identical)\n"
            f"  - Insufficient interannual variability\n"
            f"  - Data quality issues\n"
            f"Check ERA5 source for {var_short} monthly {extreme}ima variability"
        )


def _matches_output_format(
    path: Path, attrs: dict, var_specs: dict[str, tuple[np.dtype, dict]], coords: dict
) -> bool:
    """True if path holds exactly the record+scale Dataset this script would write."""
    try:
        with xr.open_dataset(path) as existing:
            if set(existing.data_vars) != set(var_specs) or set(existing.coords) != set(coords):
                return False
            if any(not np.array_equal(existing[name].values, values) for name, values in coords.items()):
                return False
            for name, (dtype, var_attrs) in var_specs.items():
                var = existing[name]
                if var.dims != ("month", "lat", "lon") or var.dtype != dtype:
                    return False
                if any(var.attrs.get(key) != value for key, value in var_attrs.items()):
                    return False
            return existing.attrs == attrs
    except Exception:
        # Unreadable or not a NetCDF Dataset: treat as stale and overwrite.
        return False


_load_dotenv()

heat_data = Path(os.environ.get("HEAT_DATA", "data"))
era5_root_default = Path(os.environ.get("ERA5_ROOT", str(heat_data / "wb2")))
era5_truth_default = os.environ.get(
    "ERA5_RECORD_SOURCE",
    os.environ.get(
        "ERA5_WB2_SOURCE",
        os.environ.get(
            "ERA5_TRUTH_ZARR",
            str(
                era5_root_default
                / "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
            ),
        ),
    ),
)

parser = argparse.ArgumentParser(description="Create ERA5 monthly record climatology files.")
parser.add_argument("--era5-zarr", type=str, default=era5_truth_default,
                    help="Local path or gs:// URI of the ERA5 Zarr store.")
parser.add_argument("--out-root", type=str, default=os.environ.get("RECORDS_ROOT", str(era5_root_default)),
                    help="Directory where record files are written.")
parser.add_argument("--record-tag", type=str, default="1979_2021",
                    help="Tag used in output filenames.")
parser.add_argument("--start-year", type=int, default=1979,
                    help="First year in climatology window.")
parser.add_argument("--end-year", type=int, default=2021,
                    help="Last year in climatology window.")
parser.add_argument("--hours", type=str, default="0,12",
                    help="Comma-separated UTC hours used for climatology, e.g. '0,12'.")
parser.add_argument("--lat-min", type=float, default=-60.0,
                    help="Southern latitude bound retained in record files.")
parser.add_argument("--lat-max", type=float, default=90.0,
                    help="Northern latitude bound retained in record files.")
parser.add_argument("--workers", type=int, default=4,
                    help="Maximum Dask worker threads (keep modest on shared hosts).")
args = parser.parse_args()

if args.start_year > args.end_year:
    raise ValueError("start-year must not be later than end-year.")
if args.lat_min >= args.lat_max:
    raise ValueError("lat-min must be smaller than lat-max.")
if args.workers < 1:
    raise ValueError("workers must be at least 1.")
if args.end_year <= args.start_year:
    raise ValueError("At least two historical years are needed for a sample SD.")

expected_record_tag = f"{args.start_year}_{args.end_year}"
if args.record_tag != expected_record_tag:
    raise ValueError(
        f"record-tag is '{args.record_tag}', but the requested years imply "
        f"'{expected_record_tag}'. Use matching values to avoid mislabelled records."
    )

era5_zarr = args.era5_zarr
out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

hours = _parse_hours(args.hours)
if not hours:
    raise ValueError("At least one verification hour must be supplied.")
if any(hour < 0 or hour > 23 for hour in hours):
    raise ValueError("Verification hours must lie between 0 and 23 UTC.")
if len(hours) != len(set(hours)):
    raise ValueError("Verification hours must be unique.")
hours = sorted(hours)
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
print(f"Latitude    : {args.lat_min} to {args.lat_max}")
print(f"Workers     : {args.workers}")
print("Scale       : SD across yearly monthly extremes (ddof=1; no detrending)")

print("\nOpening ERA5 store...")
ds = _open_zarr(era5_zarr)

available_years = np.unique(ds["time"].dt.year.values).astype(int)
missing_years = np.setdiff1d(
    np.arange(args.start_year, args.end_year + 1),
    available_years,
)
if missing_years.size:
    raise ValueError(
        "ERA5 source does not contain all requested record years. Missing: "
        + ", ".join(str(year) for year in missing_years)
    )

available_hours = np.unique(ds["time"].dt.hour.values).astype(int)
missing_hours = np.setdiff1d(np.asarray(hours), available_hours)
if missing_hours.size:
    raise ValueError(
        "ERA5 source does not contain all requested UTC hours. Missing: "
        + ", ".join(str(hour) for hour in missing_hours)
    )

record_attrs = {
    "record_tag": args.record_tag,
    "record_start_year": args.start_year,
    "record_end_year": args.end_year,
    "record_hours_utc": ",".join(str(hour) for hour in hours),
    "source": str(era5_zarr),
    "latitude_bounds": f"{args.lat_min},{args.lat_max}",
}

def _output_attrs(extreme: str, var_short: str) -> dict:
    attrs = dict(record_attrs)
    attrs.update({
        "extreme": extreme,
        "variable": var_short,
        "n_years": args.end_year - args.start_year + 1,
        "scale_definition": "Interannual sample SD of yearly monthly " + extreme + "ima",
        "scale_ddof": 1,
        "scale_detrended": "false",
        "missing_data_policy": "Require complete timestamps and finite monthly blocks",
    })
    return attrs


def _var_specs(extreme: str, var_short: str) -> dict[str, tuple[np.dtype, dict]]:
    var_long = VARIABLES[var_short]
    units = ds[var_long].attrs.get("units", "K" if var_short == "t2m" else "m s-1")
    return {
        "record": (ds[var_long].dtype, {
            **record_attrs, "long_name": "Historical monthly record", "units": units,
        }),
        "scale": (np.dtype("float32"), {
            "long_name": "Sample SD across yearly monthly " + extreme + "ima",
            "units": units,
            "ddof": 1,
        }),
    }


def _record_file(extreme: str, var_short: str) -> Path:
    return out_root / f"ERA5_record_{extreme}_month_{args.record_tag}_{var_short}.nc"


record_tasks: list[tuple[str, str, xr.DataArray]] = []

for var_short, var_long in VARIABLES.items():
    if var_long not in ds:
        raise KeyError(f"ERA5 source does not contain required variable '{var_long}'.")

    print(f"\nPreparing {var_short} ({var_long})...")
    da = _normalize_lon_lat(ds[var_long])

    # Validate domain selection before processing
    da = da.sel(
        time=slice(time_start, time_end),
        latitude=slice(args.lat_min, args.lat_max),
        longitude=slice(-180, 180),
    )

    if da.sizes["latitude"] == 0 or da.sizes["longitude"] == 0:
        raise ValueError(
            f"Requested domain ({args.lat_min} to {args.lat_max}, -180 to 180) "
            f"not found in ERA5 source. Available: "
            f"lat=[{float(ds.latitude.min()):.1f}, {float(ds.latitude.max()):.1f}], "
            f"lon=[{float(ds.longitude.min()):.1f}, {float(ds.longitude.max()):.1f}]"
        )

    # Validate grid before any heavy processing
    _validate_grid(da, expected_spacing=1.5)

    # Skip extremes whose existing file already matches the output format.
    coords = {
        "month": np.arange(1, 13),
        "lat": da.latitude.values,
        "lon": da.longitude.values,
    }
    extremes = []
    for extreme in (["max", "min"] if var_short == "t2m" else ["max"]):
        record_file = _record_file(extreme, var_short)
        if record_file.exists() and _matches_output_format(
            record_file, _output_attrs(extreme, var_short), _var_specs(extreme, var_short), coords
        ):
            print(f"  Keeping existing {record_file}")
        else:
            if record_file.exists():
                print(f"  {record_file} does not match the output format; will overwrite")
            extremes.append(extreme)
    if not extremes:
        continue

    # Filter to requested hours
    da = da.where(da.time.dt.hour.isin(hours), drop=True)

    # Check completeness BEFORE chunking and computation
    _check_complete_times(da.time, args.start_year, args.end_year, hours)

    # Now chunk for processing
    da = da.chunk({"time": 256, "latitude": -1, "longitude": 120})
    # Preserve one extreme per year and calendar month. Both the record
    # and its scale will be derived from these same small block summaries.
    # A missing value makes its monthly block invalid, rather than silently
    # comparing maxima/minima based on different numbers of observations.
    for extreme in extremes:
        block_extremes = getattr(da.resample(time="MS"), extreme)("time", skipna=False)
        record_tasks.append((extreme, var_short, block_extremes))

if record_tasks:
    print("\nReducing remote ERA5 to yearly monthly extremes...")
    # One Dataset means one shared task graph: temperature maximum and minimum
    # reuse the same source reads. Do not compute or write each reduction separately.
    tasks = xr.Dataset({
        f"{extreme}_{var_short}": task
        for extreme, var_short, task in record_tasks
    })
    with dask.config.set(scheduler="threads", num_workers=args.workers), ProgressBar(dt=5.0):
        computed = tasks.compute()
else:
    print("\nAll record files already match the output format; nothing to compute.")

for extreme, var_short, _ in record_tasks:
    values = computed[f"{extreme}_{var_short}"]
    if not np.isfinite(values.values).all():
        raise ValueError(
            f"Incomplete or non-finite monthly blocks for {var_short} {extreme}. "
            "Record scales require complete historical blocks at every retained cell."
        )
    by_month = values.groupby("time.month")
    record = getattr(by_month, extreme)("time", skipna=False)
    # Float64 accumulation avoids cancellation for temperatures in Kelvin.
    scale = values.astype("float64").groupby("time.month").std(
        "time", skipna=False, ddof=1
    ).astype("float32")
    record = record.rename({"latitude": "lat", "longitude": "lon"})
    scale = scale.rename({"latitude": "lat", "longitude": "lon"})
    record_file = _record_file(extreme, var_short)
    var_specs = _var_specs(extreme, var_short)
    units = var_specs["record"][1]["units"]

    # Validate scale strictly (zero tolerance for NaN or scale <= 0)
    _validate_scale(scale, var_short, extreme, units)

    # Log scale statistics
    print(f"  Scale statistics ({extreme} {var_short}):")
    print(f"    min   = {float(scale.min()):.6f} {units}")
    print(f"    max   = {float(scale.max()):.6f} {units}")
    print(f"    mean  = {float(scale.mean()):.6f} {units}")
    print(f"    median= {float(scale.median()):.6f} {units}")
    pair = xr.Dataset({
        "record": record.transpose("month", "lat", "lon"),
        "scale": scale.transpose("month", "lat", "lon"),
    })
    pair.attrs = _output_attrs(extreme, var_short)
    for name, (_, var_attrs) in var_specs.items():
        pair[name].attrs.update(var_attrs)
    # Use process-specific temporary file to avoid race conditions
    with tempfile.NamedTemporaryFile(
        mode='wb', dir=out_root, suffix='.nc.partial', delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pair.to_netcdf(
            tmp_path,
            engine="netcdf4",
            encoding={name: {"zlib": True, "complevel": 4, "shuffle": True} for name in pair.data_vars},
        )
        os.replace(tmp_path, record_file)
        print(f"  Wrote {record_file} (record+scale; {record_file.stat().st_size / 1e6:.2f} MB)")
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise

print("\nPreparing land-sea mask...")
if "land_sea_mask" in ds:
    lsm = ds["land_sea_mask"]
elif "lsm" in ds:
    lsm = ds["lsm"]
else:
    raise KeyError("No land-sea mask variable found in ERA5 Zarr (expected 'land_sea_mask' or 'lsm').")

lsm = _normalize_lon_lat(lsm)
lsm = lsm.sel(
    latitude=slice(args.lat_min, args.lat_max),
    longitude=slice(-180, 180),
)

if "time" in lsm.dims:
    lsm = lsm.isel(time=0, drop=True)

# Eval scripts expect a dummy time dimension and variable name 'lsm'.
lsm = lsm.expand_dims(time=[pd.Timestamp(f"{args.start_year}-01-01")])
lsm_ds = lsm.to_dataset(name="lsm")
lsm_file = out_root / "ERA5_lsm_sfc.nc"
if lsm_file.exists():
    print(f"  Keeping existing {lsm_file}")
else:
    lsm_ds.to_netcdf(lsm_file)
    print(f"  Wrote {lsm_file}")

print("\nDone.")
