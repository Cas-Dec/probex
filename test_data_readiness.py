#!/usr/bin/env python
"""
Test script to verify all required data is present and loadable before starting evaluation.

Checks:
1. Truth data (ERA5, HRES_fc0) exists and has correct structure
2. ERA5 record files exist
3. Forecast data exists for all models, variables, and lead times
4. Data can be opened without errors
5. Coordinates match expected conventions
6. Dimensions are compatible

Usage:
    python test_data_readiness.py [--year YEAR] [--quick]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import xarray as xr

# Model configuration (same as eval scripts)
MODEL_CONFIG = {
    "IFS_ENS": {
        "leads": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120,
                  132, 144, 156, 168, 180, 192, 204, 216, 228, 240],
        "n_members": 50,
        "resolution_tag": "1p50",
    },
    "ArchesWeatherGen": {
        "leads": [48, 120, 168, 240],
        "n_members": 64, # should we subsample to 50 for fair comparison?
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

VARIABLES = ["t2m", "w10"]
VARIABLE_MAP = {
    "t2m": "2m_temperature",
    "w10": "10m_wind_speed",
}

# Required coordinates for forecasts
REQUIRED_FORECAST_COORDS = ["time", "number", "prediction_timedelta", "latitude", "longitude"]

# Data paths
FORECAST_ROOT = Path("/data/gent/vo/000/gvo00090/HEAT/Data/forecasts")
ERA5_ZARR = Path("/data/gent/vo/000/gvo00090/HEAT/Data/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
RECORDS_ROOT = Path("/kyukon/scratch/gent/vo/000/gvo00090/vsc46240/PhD/probex/data/era5_records")


def test_truth_data(year: int) -> bool:
    """Test that ERA5 and HRES_fc0 truth data exist and are loadable."""
    print("\n" + "="*60)
    print("TESTING TRUTH DATA")
    print("="*60)

    all_ok = True

    # Test ERA5 (shared store)
    print(f"\n✓ Checking shared ERA5 store...")
    print(f"  Path: {ERA5_ZARR}")

    if not ERA5_ZARR.exists():
        print(f"  ✗ MISSING: {ERA5_ZARR}")
        all_ok = False
    else:
        try:
            ds = xr.open_zarr(ERA5_ZARR, chunks={})

            for var in VARIABLES:
                variable = VARIABLE_MAP[var]

                if variable not in ds:
                    print(f"  ✗ Variable '{variable}' not found in dataset")
                    all_ok = False
                    continue

                # Check year coverage
                time_slice = ds['time'].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                if len(time_slice) == 0:
                    print(f"  ✗ Year {year} not found in ERA5 store")
                    all_ok = False
                    continue

                print(f"  ✓ {var}: {len(time_slice)} time steps for {year}")

            print(f"  ✓ ERA5 store loaded successfully")
            print(f"    Resolution: {len(ds.longitude)}x{len(ds.latitude)} (1.5° equiangular)")

        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            all_ok = False

    # Test HRES_fc0
    for var in VARIABLES:
        hres_file = FORECAST_ROOT / "HRES_fc0" / f"HRES_fc0_{year}_no_Antarctic_{var}_1p50.zarr"
        print(f"\n✓ Checking HRES_fc0 {var} for {year}...")
        print(f"  Path: {hres_file}")

        if not hres_file.exists():
            print(f"  ⚠ NOT FOUND (optional truth): {hres_file}")
            continue

        try:
            ds = xr.open_zarr(hres_file, chunks={})
            variable = VARIABLE_MAP[var]

            if variable not in ds:
                print(f"  ✗ Variable '{variable}' not found in dataset")
                all_ok = False
                continue

            data = ds[variable]
            print(f"  ✓ Loaded successfully")
            print(f"    Dimensions: {dict(data.dims)}")

        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            all_ok = False

    return all_ok


def test_era5_records(record_tag: str = "1979_2018") -> bool:
    """Test that ERA5 record files exist."""
    print("\n" + "="*60)
    print("TESTING ERA5 RECORD FILES")
    print("="*60)

    all_ok = True

    for var in VARIABLES:
        for extreme in ["max", "min"]:
            if extreme == "min" and var == "w10":
                continue  # w10 only has max records

            record_file = RECORDS_ROOT / f"ERA5_record_{extreme}_month_{record_tag}_{var}.nc"
            print(f"\n✓ Checking {extreme} records for {var}...")
            print(f"  Path: {record_file}")

            if not record_file.exists():
                print(f"  ✗ MISSING: {record_file}")
                all_ok = False
                continue

            try:
                ds = xr.open_dataset(record_file)
                print(f"  ✓ Loaded successfully")
                print(f"    Dimensions: {dict(ds.dims)}")

            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
                all_ok = False

    # Check land-sea mask
    lsm_file = RECORDS_ROOT / "ERA5_lsm_sfc.nc"
    print(f"\n✓ Checking land-sea mask...")
    print(f"  Path: {lsm_file}")

    if not lsm_file.exists():
        print(f"  ✗ MISSING: {lsm_file}")
        all_ok = False
    else:
        try:
            ds = xr.open_dataset(lsm_file)
            print(f"  ✓ Loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            all_ok = False

    return all_ok


def test_forecast_data(year: int, quick: bool = False) -> bool:
    """Test that forecast data exists for all models, variables, and leads."""
    print("\n" + "="*60)
    print("TESTING FORECAST DATA")
    print("="*60)

    all_ok = True

    for model, config in MODEL_CONFIG.items():
        print(f"\n{'─'*60}")
        print(f"MODEL: {model}")
        print(f"{'─'*60}")

        leads = config["leads"]
        n_members = config["n_members"]
        resolution_tag = config["resolution_tag"]

        # In quick mode, only test first lead
        test_leads = [leads[0]] if quick else leads

        for var in VARIABLES:
            variable = VARIABLE_MAP[var]

            print(f"\n  Variable: {var} ({variable})")
            print(f"  Expected leads: {len(leads)} total")
            print(f"  Testing: {len(test_leads)} leads {'(quick mode)' if quick else ''}")

            missing_leads = []
            error_leads = []
            coord_errors = []

            for lead_hour in test_leads:
                forecast_file = (FORECAST_ROOT / model /
                               f"{model}_{year}_no_Antarctic_{var}_m{n_members}_{resolution_tag}_lead_{lead_hour:03d}h.zarr")

                if not forecast_file.exists():
                    missing_leads.append(lead_hour)
                    continue

                try:
                    ds = xr.open_zarr(forecast_file, chunks={})

                    if variable not in ds:
                        error_leads.append((lead_hour, f"Variable '{variable}' not found"))
                        continue

                    data = ds[variable]

                    # Check required coordinates
                    missing_coords = [c for c in REQUIRED_FORECAST_COORDS if c not in data.coords]
                    if missing_coords:
                        coord_errors.append((lead_hour, f"Missing coords: {missing_coords}"))

                    # Check number dimension
                    if "number" in data.dims:
                        actual_members = data.sizes["number"]
                        if actual_members != n_members:
                            coord_errors.append((lead_hour, f"Expected {n_members} members, got {actual_members}"))

                except Exception as e:
                    error_leads.append((lead_hour, str(e)))

            # Report results for this model/var
            tested_count = len(test_leads) - len(missing_leads) - len(error_leads)

            if not missing_leads and not error_leads and not coord_errors:
                print(f"  ✓ All {tested_count}/{len(test_leads)} leads OK")
            else:
                all_ok = False

                if missing_leads:
                    print(f"  ✗ Missing {len(missing_leads)} leads: {missing_leads}")

                if error_leads:
                    print(f"  ✗ Load errors for {len(error_leads)} leads:")
                    for lead, error in error_leads:
                        print(f"    - Lead {lead}h: {error}")

                if coord_errors:
                    print(f"  ✗ Coordinate errors for {len(coord_errors)} leads:")
                    for lead, error in coord_errors:
                        print(f"    - Lead {lead}h: {error}")

    return all_ok


def test_single_forecast_load(year: int) -> bool:
    """Test loading a single forecast file in detail (mimics eval script behavior)."""
    print("\n" + "="*60)
    print("TESTING DETAILED FORECAST LOAD (IFS_ENS, t2m, lead 12h)")
    print("="*60)

    model = "IFS_ENS"
    var = "t2m"
    variable = VARIABLE_MAP[var]
    lead_hour = 12

    config = MODEL_CONFIG[model]
    forecast_file = (FORECAST_ROOT / model /
                    f"{model}_{year}_no_Antarctic_{var}_m{config['n_members']}_{config['resolution_tag']}_lead_{lead_hour:03d}h.zarr")

    print(f"\nFile: {forecast_file}")

    if not forecast_file.exists():
        print(f"✗ File does not exist!")
        return False

    try:
        # Mimic eval script loading pattern
        ds = xr.open_zarr(forecast_file, chunks={})
        forecast = ds[variable]

        print(f"\n✓ Loaded successfully")
        print(f"\nDimensions: {dict(forecast.dims)}")
        print(f"Coordinates: {list(forecast.coords)}")
        print(f"Shape: {forecast.shape}")
        print(f"Chunks: {forecast.chunks}")

        # Test coordinate access (as eval scripts do)
        print(f"\nCoordinate details:")
        print(f"  time: {forecast['time'].values[0]} ... {forecast['time'].values[-1]}")
        print(f"  number: {forecast['number'].values.min()} to {forecast['number'].values.max()} ({len(forecast['number'])} members)")
        print(f"  prediction_timedelta: {forecast['prediction_timedelta'].values}")
        print(f"  latitude: {forecast['latitude'].values.min():.1f} to {forecast['latitude'].values.max():.1f}")
        print(f"  longitude: {forecast['longitude'].values.min():.1f} to {forecast['longitude'].values.max():.1f}")

        print(f"\n✓ All coordinate access successful")
        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test data readiness for evaluation")
    parser.add_argument("--year", type=int, default=2022, help="Year to test")
    parser.add_argument("--record-tag", type=str, default="1979_2018", help="ERA5 record tag")
    parser.add_argument("--quick", action="store_true", help="Quick mode: only test first lead of each model")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"DATA READINESS TEST - YEAR {args.year}")
    print("="*60)

    results = {}

    # Run tests
    results["truth"] = test_truth_data(args.year)
    results["records"] = test_era5_records(args.record_tag)
    results["forecasts"] = test_forecast_data(args.year, quick=args.quick)
    results["detailed"] = test_single_forecast_load(args.year)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for evaluation!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Fix issues before evaluation")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
