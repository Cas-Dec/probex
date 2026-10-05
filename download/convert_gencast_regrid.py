"""
Convert GenCast forecasts from 1.0° to 1.5° resolution and restructure into
per-lead, per-variable Zarr files compatible with the probex evaluation framework.

This script:
1. Loads GenCast forecasts at 1.0° resolution (360×151 grid)
2. Applies conservative regridding to 1.5° resolution (240×100 grid, no_Antarctic)
3. Splits by variable and lead time into separate Zarr stores

Run from the project root:
    python download/convert_gencast_regrid.py --model gencast
    python download/convert_gencast_regrid.py --model gencast-mini

Writes to /data/.../forecasts/GenCast/ or /data/.../forecasts/GenCast-mini/:
    GenCast_2022_no_Antarctic_t2m_m32_1p50_lead_012h.zarr
    GenCast_2022_no_Antarctic_w10_m32_1p50_lead_012h.zarr
    ...

Input format (from HEAT-S2S-Benchmarking):
    - Resolution: 1.0° (360 lon × 151 lat)
    - Domain: -180 to 179°E, -60 to 90°N
    - Dimensions: (time, number, prediction_timedelta, latitude, longitude)

Output format (probex standard):
    - Resolution: 1.5° (240 lon × 100 lat)
    - Domain: -180 to 178.5°E, -58.5 to 90°N (no_Antarctic)
    - Dimensions: (time, number, prediction_timedelta, latitude, longitude)
    - One Zarr per (variable, lead_time) combination
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

# Import the conservative regridder
from regrid_conservative import ConservativeRegridder


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


_load_dotenv()

heat_data_root = Path(os.environ.get("HEAT_DATA", "/data/gent/vo/000/gvo00090/HEAT/Data"))
forecast_root = heat_data_root / "forecasts"


def create_target_grid_no_antarctic() -> tuple[np.ndarray, np.ndarray]:
    """
    Create the target 1.5° grid matching the probex no_Antarctic domain.

    Returns:
        (longitude, latitude) arrays in degrees
    """
    # Longitude: -180 to 178.5, step 1.5 (240 points)
    lon = np.arange(-180.0, 180.0, 1.5)

    # Latitude: -58.5 to 90, step 1.5 (100 points)
    # This matches ERA5 records domain (excluding Antarctic)
    lat = np.arange(-58.5, 90.0 + 0.75, 1.5)

    assert len(lon) == 240, f"Expected 240 lon points, got {len(lon)}"
    assert len(lat) == 100, f"Expected 100 lat points, got {len(lat)}"

    return lon, lat


def prepare_source_coordinates(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and validate source grid coordinates.

    Args:
        ds: GenCast dataset at 1.0° resolution

    Returns:
        (longitude, latitude) arrays in degrees, sorted ascending
    """
    lon = ds.longitude.values
    lat = ds.latitude.values

    # Ensure coordinates are sorted ascending
    if not np.all(np.diff(lon) > 0):
        raise ValueError("Source longitude must be strictly increasing")

    if not np.all(np.diff(lat) > 0):
        # Flip if descending
        if np.all(np.diff(lat) < 0):
            print("  Source latitude is descending, will flip during regridding")
            lat = lat[::-1]
        else:
            raise ValueError("Source latitude must be monotonic")

    return lon, lat


def regrid_and_save_one_lead(
    source_zarr: Path,
    output_dir: Path,
    model_name: str,
    year: int,
    lead_hour: int,
    var_short: str,
    var_long: str,
    regridder: ConservativeRegridder,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    n_members: int,
) -> None:
    """
    Regrid and save a single (variable, lead_time) combination.

    Supports incremental updates: if output exists, only processes new times.

    Args:
        source_zarr: Path to source GenCast Zarr (1.0° resolution)
        output_dir: Directory to write output Zarr
        model_name: Model name for filename (e.g., "GenCast")
        year: Forecast year
        lead_hour: Lead time in hours
        var_short: Short variable name (t2m, w10)
        var_long: Long variable name (2m_temperature, 10m_wind_speed)
        regridder: ConservativeRegridder instance
        target_lon: Target longitude coordinates
        target_lat: Target latitude coordinates
        n_members: Number of ensemble members
    """
    out_file = output_dir / (
        f"{model_name}_{year}_no_Antarctic_{var_short}"
        f"_m{n_members}_1p50_lead_{lead_hour:03d}h.zarr"
    )

    # Load source data for this lead time and variable
    print(f"  Processing {var_short} lead {lead_hour:03d}h...")
    print(f"    Loading from source...")
    ds_source = xr.open_zarr(source_zarr, chunks={})
    ds_lead = ds_source.sel(prediction_timedelta=[lead_hour])
    da_source = ds_lead[var_long]

    n_times = len(da_source.time)
    n_members = len(da_source.number)

    # Check if file exists and is valid for incremental update
    file_exists = out_file.exists()
    mode = 'r+' if file_exists else 'w'

    if file_exists:
        try:
            existing = xr.open_zarr(out_file, chunks={})
            if var_long not in existing.data_vars:
                print(f"    Rewriting (missing variable): {out_file.name}")
                shutil.rmtree(out_file, ignore_errors=True)
                file_exists = False
                mode = 'w'
            elif existing.sizes['time'] != n_times:
                print(f"    Rewriting (wrong time dimension): {out_file.name}")
                shutil.rmtree(out_file, ignore_errors=True)
                file_exists = False
                mode = 'w'
            else:
                print(f"    Updating existing file: {out_file.name}")
        except Exception as e:
            print(f"    Rewriting (unreadable): {out_file.name} ({e})")
            shutil.rmtree(out_file, ignore_errors=True)
            file_exists = False
            mode = 'w'

    # Initialize output Zarr store if needed
    if not file_exists:
        print(f"    Initializing new Zarr store...")
        ds_template = xr.Dataset(
            {
                var_long: xr.DataArray(
                    np.full((n_times, n_members, 1, len(target_lat), len(target_lon)), np.nan, dtype=np.float32),
                    dims=['time', 'number', 'prediction_timedelta', 'latitude', 'longitude'],
                    coords={
                        'time': da_source.time,
                        'number': da_source.number,
                        'prediction_timedelta': da_source.prediction_timedelta,
                        'latitude': target_lat,
                        'longitude': target_lon,
                    }
                )
            }
        )

        # Clear encoding
        for var_name in ds_template.variables:
            ds_template[var_name].encoding = {}

        # Create Zarr with template structure
        ds_template.to_zarr(out_file, mode='w', zarr_format=2)
        del ds_template

    print(f"    Writing regridded data incrementally...")

    # Open zarr for reading/writing
    z = zarr.open(str(out_file), mode='r+')

    # Regrid and write each time slice
    n_written = 0
    n_skipped_empty = 0
    n_skipped_existing = 0

    for t_idx in range(n_times):
        # Check if target already has valid data (for incremental updates)
        if file_exists:
            existing_sample = z[var_long][t_idx, 0, 0, 50, 120]
            if np.isfinite(existing_sample):
                # Already processed
                n_skipped_existing += 1
                continue

        # Check if source has any valid data (sample first member)
        sample = da_source.isel(time=t_idx, number=0, prediction_timedelta=0, latitude=50, longitude=120).compute().values

        if not np.isfinite(sample):
            # Skip this time - source has no data
            n_skipped_empty += 1
            continue

        # Regrid all members for this time
        time_data = np.full((n_members, 1, len(target_lat), len(target_lon)), np.nan, dtype=np.float32)

        for m_idx in range(n_members):
            # Extract single time/member slice
            slice_data = da_source.isel(time=t_idx, number=m_idx, prediction_timedelta=0).compute().values

            # Regrid
            regridded_slice = regridder.regrid_array(slice_data)
            time_data[m_idx, 0, :, :] = regridded_slice

        # Write this time slice to Zarr directly
        z[var_long][t_idx, :, :, :, :] = time_data
        del time_data

        n_written += 1

        if n_written % 50 == 0:
            print(f"      Written {n_written} new times (skipped {n_skipped_empty} empty, {n_skipped_existing} already done)...")

    print(f"    ✓ Saved: {out_file.name}")
    print(f"    Summary: {n_written} times written, {n_skipped_empty} empty in source, {n_skipped_existing} already processed")

    # Cleanup
    del da_source, ds_lead, ds_source, z
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Regrid GenCast forecasts from 1.0° to 1.5° resolution"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gencast", "gencast-mini"],
        help="Which GenCast variant to process"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Forecast year (default: 2022)"
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=[12, 48, 120, 168, 240],
        help="Lead times in hours to process (default: evaluation leads [12, 48, 120, 168, 240])"
    )
    args = parser.parse_args()

    # Determine paths
    if args.model == "gencast":
        source_zarr = forecast_root / "gencast_forecasts_2022.zarr"
        model_name = "GenCast"
        output_dir = forecast_root / "GenCast"
    elif args.model == "gencast-mini":
        source_zarr = forecast_root / "gencast_mini_forecasts_2022.zarr"
        model_name = "GenCast-mini"
        output_dir = forecast_root / "GenCast-mini"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("=" * 80)
    print("GenCast Conservative Regridding: 1.0° → 1.5°")
    print("=" * 80)
    print(f"Model:       {model_name}")
    print(f"Source:      {source_zarr}")
    print(f"Output dir:  {output_dir}")
    print(f"Year:        {args.year}")
    print(f"Lead times:  {args.leads}")
    print()

    # Check source exists
    if not source_zarr.exists():
        raise FileNotFoundError(
            f"Source Zarr not found: {source_zarr}\n\n"
            f"GenCast forecasts should be created by:\n"
            f"  HEAT-S2S-Benchmarking/submit_gencast_2022.sh"
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source metadata
    print("Loading source dataset metadata...")
    ds = xr.open_zarr(source_zarr, chunks={})

    # Validate format
    expected_dims = {"time", "number", "prediction_timedelta", "latitude", "longitude"}
    if set(ds.dims.keys()) != expected_dims:
        raise ValueError(
            f"Source has unexpected dimensions: {set(ds.dims.keys())}\n"
            f"Expected: {expected_dims}"
        )

    expected_vars = {"2m_temperature", "10m_wind_speed"}
    if not expected_vars.issubset(set(ds.data_vars.keys())):
        raise ValueError(
            f"Source missing required variables. "
            f"Has: {set(ds.data_vars.keys())}, needs: {expected_vars}"
        )

    n_members = ds.sizes["number"]
    available_leads = sorted([int(x) for x in ds.prediction_timedelta.values])

    print(f"  Members:     {n_members}")
    print(f"  Lead times:  {available_leads}")
    print(f"  Source grid: {ds.sizes['longitude']} lon × {ds.sizes['latitude']} lat")
    print(f"  Lon range:   {float(ds.longitude.min()):.1f}° to {float(ds.longitude.max()):.1f}°")
    print(f"  Lat range:   {float(ds.latitude.min()):.1f}° to {float(ds.latitude.max()):.1f}°")
    print()

    # Prepare grids
    print("Preparing regridding grids...")
    source_lon, source_lat = prepare_source_coordinates(ds)
    target_lon, target_lat = create_target_grid_no_antarctic()

    print(f"  Source: {len(source_lon)} lon × {len(source_lat)} lat (1.0° resolution)")
    print(f"  Target: {len(target_lon)} lon × {len(target_lat)} lat (1.5° resolution)")
    print()

    # Initialize regridder
    regridder = ConservativeRegridder(source_lon, source_lat, target_lon, target_lat)
    print()

    # Variables to process
    VARIABLES = {
        "t2m": "2m_temperature",
        "w10": "10m_wind_speed",
    }

    # Process each variable and lead time
    print("Starting regridding and conversion...")
    print()

    for var_short, var_long in VARIABLES.items():
        print(f"Variable: {var_short} ({var_long})")

        for lead_hour in args.leads:
            if lead_hour not in available_leads:
                print(f"  WARNING: Lead {lead_hour}h not in source data, skipping")
                continue

            regrid_and_save_one_lead(
                source_zarr=source_zarr,
                output_dir=output_dir,
                model_name=model_name,
                year=args.year,
                lead_hour=lead_hour,
                var_short=var_short,
                var_long=var_long,
                regridder=regridder,
                target_lon=target_lon,
                target_lat=target_lat,
                n_members=n_members,
            )

        print()

    print("=" * 80)
    print("✓ Conversion complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
