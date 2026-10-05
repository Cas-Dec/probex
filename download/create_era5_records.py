"""
Create ERA5 monthly record climatology files used by evaluation scripts.

Outputs (written into --out-root):
    ERA5_record_max_month_<record-tag>_t2m.nc
    ERA5_record_min_month_<record-tag>_t2m.nc
    ERA5_record_max_month_<record-tag>_w10.nc
    ERA5_lsm_sfc.nc

By default, records are computed from ERA5 years 1979-2021 and only
00/12 UTC timestamps to match forecast verification times.

With --with-scale, also write ERA5_record_scale_<max|min>_month_<tag>_<var>.nc.
Each file contains record and scale on (month, lat, lon). The scale is the
sample SD (ddof=1) across yearly monthly maxima/minima, not across weather
values or monthly means. Remote ERA5 is reduced lazily; raw fields are never
saved locally.
"""

from __future__ import annotations

import argparse
import os

# Match the existing downloaders on IPv4-only cluster networks.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from pathlib import Path

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
    out = da
    if "longitude" in out.coords:
        out = out.assign_coords(
            longitude=(((out.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    if "latitude" in out.coords:
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
parser.add_argument("--with-scale", action="store_true",
                    help="Also save record/scale pairs; scale is the sample SD of yearly monthly extremes.")
args = parser.parse_args()

if args.start_year > args.end_year:
    raise ValueError("start-year must not be later than end-year.")
if args.lat_min >= args.lat_max:
    raise ValueError("lat-min must be smaller than lat-max.")
if args.workers < 1:
    raise ValueError("workers must be at least 1.")
if args.with_scale and args.end_year <= args.start_year:
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
if args.with_scale:
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

record_tasks: list[tuple[str, str, xr.DataArray]] = []

for var_short, var_long in VARIABLES.items():
    if var_long not in ds:
        raise KeyError(f"ERA5 source does not contain required variable '{var_long}'.")

    print(f"\nPreparing {var_short} ({var_long})...")
    da = _normalize_lon_lat(ds[var_long])
    da = da.sel(
        time=slice(time_start, time_end),
        latitude=slice(args.lat_min, args.lat_max),
        longitude=slice(-180, 180),
    )
    da = da.where(da.time.dt.hour.isin(hours), drop=True)
    da = da.chunk({"time": 256, "latitude": -1, "longitude": 120})

    _check_complete_times(da.time, args.start_year, args.end_year, hours)
    if args.with_scale:
        # Preserve one extreme per year and calendar month. Both the record
        # and its scale will be derived from these same small block summaries.
        # A missing value makes its monthly block invalid, rather than silently
        # comparing maxima/minima based on different numbers of observations.
        rec_max = da.resample(time="MS").max("time", skipna=False)
    else:
        rec_max = da.groupby("time.month").max("time", skipna=True)
    record_tasks.append(("max", var_short, rec_max))

    if var_short == "t2m":
        if args.with_scale:
            rec_min = da.resample(time="MS").min("time", skipna=False)
        else:
            rec_min = da.groupby("time.month").min("time", skipna=True)
        record_tasks.append(("min", var_short, rec_min))

if args.with_scale:
    print("\nReducing remote ERA5 to yearly monthly extremes...")
else:
    print("\nComputing records from the source store...")
# One Dataset means one shared task graph: temperature maximum and minimum
# reuse the same source reads. Do not compute or write each reduction separately.
tasks = xr.Dataset({
    f"{extreme}_{var_short}": task
    for extreme, var_short, task in record_tasks
})
with dask.config.set(scheduler="threads", num_workers=args.workers), ProgressBar(dt=5.0):
    computed = tasks.compute()

computed_records = []
computed_scales = []
for extreme, var_short, _ in record_tasks:
    values = computed[f"{extreme}_{var_short}"]
    if args.with_scale:
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
        computed_scales.append(scale)
        computed_records.append(record)
    else:
        computed_records.append(values)

for index, ((extreme, var_short, _), record) in enumerate(zip(record_tasks, computed_records)):
    record = record.rename({"latitude": "lat", "longitude": "lon"})
    record.name = f"ERA5_record_{extreme}_month_{args.record_tag}_{var_short}"
    record.attrs.update(record_attrs)
    record_file = (
        out_root
        / f"ERA5_record_{extreme}_month_{args.record_tag}_{var_short}.nc"
    )

    if args.with_scale:
        # When computing scales, verify existing record-only file matches if present
        if record_file.exists():
            try:
                with xr.open_dataarray(record_file) as existing:
                    xr.testing.assert_equal(record.transpose(*existing.dims), existing)
                print(f"  Verified existing {record_file}")
            except ValueError:
                # File exists but is a Dataset (record+scale), try loading record variable
                try:
                    with xr.open_dataset(record_file) as ds_existing:
                        if "record" in ds_existing:
                            xr.testing.assert_equal(record.transpose(*ds_existing["record"].dims), ds_existing["record"])
                            print(f"  Verified existing {record_file} (record variable)")
                        else:
                            raise ValueError(f"Existing {record_file} does not contain 'record' variable")
                except AssertionError as error:
                    raise ValueError(
                        f"Recomputed record differs from {record_file}. "
                        "Check the source, baseline, hours and grid before using its scale."
                    ) from error

        # Write record+scale as a single Dataset with same filename
        scale = computed_scales[index].rename({"latitude": "lat", "longitude": "lon"})
        units = ds[VARIABLES[var_short]].attrs.get("units", "K" if var_short == "t2m" else "m s-1")
        pair = xr.Dataset({
            "record": record.transpose("month", "lat", "lon"),
            "scale": scale.transpose("month", "lat", "lon"),
        })
        pair.attrs.update(record_attrs)
        pair.attrs.update({
            "extreme": extreme,
            "variable": var_short,
            "n_years": args.end_year - args.start_year + 1,
            "scale_definition": "Interannual sample SD of yearly monthly " + extreme + "ima",
            "scale_ddof": 1,
            "scale_detrended": "false",
            "missing_data_policy": "Require complete timestamps and finite monthly blocks",
        })
        pair["record"].attrs.update({"long_name": "Historical monthly record", "units": units})
        pair["scale"].attrs.update({
            "long_name": "Sample SD across yearly monthly " + extreme + "ima",
            "units": units,
            "ddof": 1,
        })
        partial = record_file.with_suffix(".nc.partial")
        pair.to_netcdf(
            partial,
            engine="netcdf4",
            encoding={name: {"zlib": True, "complevel": 4, "shuffle": True} for name in pair.data_vars},
        )
        os.replace(partial, record_file)
        print(f"  Wrote {record_file} (record+scale; {record_file.stat().st_size / 1e6:.2f} MB)")
    else:
        # Without scale, write record-only as DataArray
        record.to_netcdf(record_file)
        print(f"  Wrote {record_file}")

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
if args.with_scale and lsm_file.exists():
    print(f"  Keeping existing {lsm_file}")
else:
    lsm_ds.to_netcdf(lsm_file)
    print(f"  Wrote {lsm_file}")

print("\nDone.")
