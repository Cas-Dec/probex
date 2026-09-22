"""Download the compact ERA5 truth store used by the ProbEx evaluations.

The output is one 1.5-degree Zarr store containing 2 m temperature, 10 m wind
speed, and the static land-sea mask. By default only 00/12 UTC data for 2022
and latitudes north of 60 S are retained.
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
import xarray as xr


ERA5_WB2_SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
DYNAMIC_VARIABLES = ("2m_temperature", "10m_wind_speed")
STATIC_VARIABLE = "land_sea_mask"


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


def _parse_hours(hours_csv: str) -> list[int]:
    hours = [int(hour.strip()) for hour in hours_csv.split(",") if hour.strip()]
    if not hours:
        raise ValueError("At least one UTC hour must be supplied.")
    if any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError("UTC hours must lie between 0 and 23.")
    return sorted(set(hours))


def _normalize_lon_lat(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    out = obj
    if "longitude" in out.coords:
        out = out.assign_coords(
            longitude=(((out.longitude + 180) % 360) - 180)
        ).sortby("longitude")
    if "latitude" in out.coords:
        out = out.sortby("latitude")
    return out


def _open_zarr(source: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    return xr.open_zarr(source, storage_options=storage_options, chunks={})


def _validate_source(ds: xr.Dataset, year: int, hours: list[int]) -> None:
    required = set(DYNAMIC_VARIABLES + (STATIC_VARIABLE,))
    missing_variables = sorted(required.difference(ds.data_vars))
    if missing_variables:
        raise KeyError(
            "ERA5 source is missing required variables: "
            + ", ".join(missing_variables)
        )

    years = np.unique(ds["time"].dt.year.values).astype(int)
    if year not in years:
        raise ValueError(f"ERA5 source does not contain requested year {year}.")

    available_hours = np.unique(ds["time"].dt.hour.values).astype(int)
    missing_hours = np.setdiff1d(np.asarray(hours), available_hours)
    if missing_hours.size:
        raise ValueError(
            "ERA5 source is missing requested UTC hours: "
            + ", ".join(str(hour) for hour in missing_hours)
        )


def _build_subset(
    ds: xr.Dataset,
    source: str,
    year: int,
    hours: list[int],
    lat_min: float,
    lat_max: float,
) -> xr.Dataset:
    dynamic = _normalize_lon_lat(ds[list(DYNAMIC_VARIABLES)])
    dynamic = dynamic.sel(
        time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"),
        latitude=slice(lat_min, lat_max),
        longitude=slice(-180, 180),
    )
    dynamic = dynamic.where(dynamic.time.dt.hour.isin(hours), drop=True)
    dynamic = dynamic.chunk({"time": 32, "latitude": -1, "longitude": 120})

    lsm = _normalize_lon_lat(ds[STATIC_VARIABLE])
    lsm = lsm.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(-180, 180),
    ).chunk({"latitude": -1, "longitude": 120})

    out = dynamic.assign({STATIC_VARIABLE: lsm}).transpose("time", "latitude", "longitude", missing_dims="ignore").drop_encoding()
    out.attrs.update(
        {
            "source": source,
            "verification_year": year,
            "verification_hours_utc": ",".join(str(hour) for hour in hours),
            "latitude_bounds": f"{lat_min},{lat_max}",
            "probex_purpose": "ERA5 verification truth",
        }
    )
    return out


def _validate_output(
    output: Path,
    year: int,
    hours: list[int],
) -> None:
    check = xr.open_zarr(output, chunks=None)
    try:
        required = set(DYNAMIC_VARIABLES + (STATIC_VARIABLE,))
        missing = sorted(required.difference(check.data_vars))
        if missing:
            raise RuntimeError("Written ERA5 store is missing: " + ", ".join(missing))

        output_years = np.unique(check["time"].dt.year.values).astype(int).tolist()
        output_hours = np.unique(check["time"].dt.hour.values).astype(int).tolist()
        if output_years != [year]:
            raise RuntimeError(f"Unexpected output years: {output_years}")
        if output_hours != hours:
            raise RuntimeError(f"Unexpected output hours: {output_hours}")
        if check[STATIC_VARIABLE].dims != ("latitude", "longitude"):
            raise RuntimeError("land_sea_mask must remain a static 2-D field.")
    finally:
        check.close()


def main() -> None:
    _load_dotenv()

    default_source = os.environ.get("ERA5_WB2_SOURCE", ERA5_WB2_SOURCE)
    default_output = os.environ.get(
        "ERA5_TRUTH_ZARR",
        "data/era5/ERA5_2022_0012_no_Antarctic_1p50.zarr",
    )

    parser = argparse.ArgumentParser(
        description="Download a compact ERA5 verification store from WeatherBench 2."
    )
    parser.add_argument("--source", default=default_source)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--hours", default="0,12")
    parser.add_argument("--lat-min", type=float, default=-60.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output store without downloading data.",
    )
    args = parser.parse_args()

    if args.lat_min >= args.lat_max:
        raise ValueError("lat-min must be smaller than lat-max.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")

    hours = _parse_hours(args.hours)
    output = Path(args.output).expanduser().resolve()
    partial = Path(str(output) + ".partial")

    if args.validate_only:
        if not output.exists():
            raise FileNotFoundError(f"ERA5 store does not exist: {output}")
        _validate_output(output, args.year, hours)
        print(f"ERA5 store is valid: {output}")
        return

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ERA5 store: {output}"
        )
    if partial.exists():
        raise FileExistsError(
            f"Partial store already exists: {partial}. Inspect or remove it before retrying."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    print("----------------------------------------------")
    print("Download compact ERA5 truth from WeatherBench 2")
    print("----------------------------------------------")
    print(f"source      : {args.source}")
    print(f"output      : {output}")
    print(f"year        : {args.year}")
    print(f"hours (UTC) : {hours}")
    print(f"latitude    : {args.lat_min} to {args.lat_max}")
    print(f"workers     : {args.workers}")

    print("\nOpening source store...")
    ds = _open_zarr(args.source)
    try:
        _validate_source(ds, args.year, hours)
        subset = _build_subset(
            ds,
            args.source,
            args.year,
            hours,
            args.lat_min,
            args.lat_max,
        )

        print("\nWriting temporary output store...")
        with dask.config.set(scheduler="threads", num_workers=args.workers), ProgressBar(dt=5.0):
            subset.to_zarr(partial, mode="w", consolidated=True, zarr_format=2)
    finally:
        ds.close()

    print("Validating output...")
    _validate_output(partial, args.year, hours)
    partial.rename(output)

    print("\nDone.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
