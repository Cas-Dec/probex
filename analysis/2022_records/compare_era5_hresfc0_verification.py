#!/usr/bin/env python
"""
Compare 2022 ERA5 vs HRES-fc0 as verification datasets.

This script checks if ERA5 and HRES-fc0 agree on:
1. Record occurrence (confusion matrix)
2. Record intensity (margin comparison)
3. Overall distribution (bias, RMSE, correlation, QQ plots)

Both datasets are used as verification ("truth") in forecast evaluations,
so understanding their agreement is critical for interpreting skill scores.

Usage:
    python analysis/era5_vs_hresfc0/compare_era5_hresfc0_verification.py --var t2m --record-tag 1979_2021
    python analysis/era5_vs_hresfc0/compare_era5_hresfc0_verification.py --var w10 --record-tag 1979_2021
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from scipy import stats
import dask

# Add eval directory to path for utility functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
import functions
import utils


def load_era5_2022(var: str, wb2_path: Path) -> xr.DataArray:
    """Load 2022 ERA5 data from WeatherBench2 store."""
    print(f"Loading ERA5 2022 data for {var}...")

    # Variable name mapping
    var_map = {
        "t2m": "2m_temperature",
        "w10": "10m_wind_speed",
    }

    ds = xr.open_zarr(wb2_path, chunks={})
    data = ds[var_map[var]].sel(time=slice("2022-01-01", "2022-12-31"))

    # Convert longitude from 0-360 to -180-180 to match HRES-fc0
    data = utils.change_lonlat(data)

    # Rename back to longitude/latitude (change_lonlat uses lon/lat)
    data = data.rename({'lon': 'longitude', 'lat': 'latitude'})

    # Convert to standard coordinates
    data = functions.normalize_spatial_coords(data)

    # Slice to no_Antarctic domain (-60 to 90) to match HRES-fc0 and all other data in repo
    data = data.sel(latitude=slice(-60, 90))

    # Convert temperature to Celsius (ERA5 and HRES-fc0 are stored in Kelvin)
    if var == "t2m":
        data = utils.Kelvin2Celsius(data)

    print(f"  Shape: {data.shape}")
    print(f"  Time range: {data.time.values[0]} to {data.time.values[-1]}")

    return data


def load_hresfc0_2022(var: str, forecast_root: Path) -> xr.DataArray:
    """Load 2022 HRES-fc0 data."""
    print(f"Loading HRES-fc0 2022 data for {var}...")

    hresfc0_file = forecast_root / "HRES_fc0" / f"HRES_fc0_2022_no_Antarctic_{var}_1p50.zarr"

    # Variable name mapping
    var_map = {
        "t2m": "2m_temperature",
        "w10": "10m_wind_speed",
    }

    ds = xr.open_zarr(hresfc0_file, chunks={})
    data = ds[var_map[var]]

    # HRES-fc0 has dims (time, prediction_timedelta, latitude, longitude)
    # We only want fc0, so squeeze out prediction_timedelta
    if "prediction_timedelta" in data.dims:
        data = data.isel(prediction_timedelta=0)

    # Convert to standard coordinates
    data = functions.normalize_spatial_coords(data)

    # Convert temperature to Celsius (ERA5 and HRES-fc0 are stored in Kelvin)
    if var == "t2m":
        data = utils.Kelvin2Celsius(data)

    print(f"  Shape: {data.shape}")
    print(f"  Time range: {data.time.values[0]} to {data.time.values[-1]}")

    return data


def load_era5_records(var: str, record_tag: str, records_dir: Path) -> tuple:
    """Load ERA5 monthly record thresholds and scales."""
    print(f"Loading ERA5 records ({record_tag}) for {var}...")

    # Load max records (heat for t2m, high wind for w10)
    record_max_file = records_dir / f"ERA5_record_max_month_{record_tag}_{var}.nc"
    # Variable name in the NetCDF file is just 'record'
    ds_max = xr.open_dataset(record_max_file)
    record_max = ds_max['record']
    scale_max = ds_max['scale'] if 'scale' in ds_max else None

    # Rename lat/lon to latitude/longitude for consistency
    if "lat" in record_max.dims:
        record_max = record_max.rename({"lat": "latitude", "lon": "longitude"})

    # Convert longitude from 0-360 to -180-180
    if record_max.longitude.max() > 180:
        record_max = record_max.assign_coords(
            longitude=(((record_max.longitude + 180) % 360) - 180)
        ).sortby('longitude')

    record_max = functions.normalize_spatial_coords(record_max)

    # Slice to no_Antarctic domain (-60 to 90) to match HRES-fc0 and all other data in repo
    record_max = record_max.sel(latitude=slice(-60, 90))

    # Normalize scale_max if it exists
    if scale_max is not None and "lat" in scale_max.dims:
        scale_max = scale_max.rename({"lat": "latitude", "lon": "longitude"})
    if scale_max is not None:
        if scale_max.longitude.max() > 180:
            scale_max = scale_max.assign_coords(
                longitude=(((scale_max.longitude + 180) % 360) - 180)
            ).sortby('longitude')
        scale_max = functions.normalize_spatial_coords(scale_max)
        scale_max = scale_max.sel(latitude=slice(-60, 90))

    # Convert temperature records from Kelvin to Celsius
    # IMPORTANT: Also convert scale (it has same units as the data)
    if var == "t2m":
        # Check if records are in Kelvin (values > 100)
        if float(record_max.max().values) > 100:
            record_max = utils.Kelvin2Celsius(record_max)
            # Scale is a standard deviation, so it also needs the same conversion
            # K to °C is just a shift by -273.15, but for std dev we don't shift,
            # the conversion is the same (1 K = 1 °C for differences)
            # So no conversion needed for scale! It's already in the right units.

    # Load min records (cold for t2m only)
    record_min = None
    scale_min = None
    if var == "t2m":
        record_min_file = records_dir / f"ERA5_record_min_month_{record_tag}_{var}.nc"
        # Variable name in the NetCDF file is just 'record'
        ds_min = xr.open_dataset(record_min_file)
        record_min = ds_min['record']
        scale_min = ds_min['scale'] if 'scale' in ds_min else None

        # Rename lat/lon to latitude/longitude for consistency
        if "lat" in record_min.dims:
            record_min = record_min.rename({"lat": "latitude", "lon": "longitude"})

        # Convert longitude from 0-360 to -180-180
        if record_min.longitude.max() > 180:
            record_min = record_min.assign_coords(
                longitude=(((record_min.longitude + 180) % 360) - 180)
            ).sortby('longitude')

        record_min = functions.normalize_spatial_coords(record_min)

        # Slice to no_Antarctic domain (-60 to 90)
        record_min = record_min.sel(latitude=slice(-60, 90))

        # Convert temperature records from Kelvin to Celsius
        if float(record_min.max().values) > 100:
            record_min = utils.Kelvin2Celsius(record_min)

        # Normalize scale_min if it exists
        if scale_min is not None and "lat" in scale_min.dims:
            scale_min = scale_min.rename({"lat": "latitude", "lon": "longitude"})
        if scale_min is not None:
            if scale_min.longitude.max() > 180:
                scale_min = scale_min.assign_coords(
                    longitude=(((scale_min.longitude + 180) % 360) - 180)
                ).sortby('longitude')
            scale_min = functions.normalize_spatial_coords(scale_min)
            scale_min = scale_min.sel(latitude=slice(-60, 90))

    return (record_max, scale_max), (record_min, scale_min)


def verify_grid_match(era5: xr.DataArray, hresfc0: xr.DataArray):
    """Verify that ERA5 and HRES-fc0 have matching spatial grids."""
    print("Verifying spatial grids match...")

    lat_match = (len(era5.latitude) == len(hresfc0.latitude) and
                 np.allclose(era5.latitude.values, hresfc0.latitude.values, atol=0.01))
    lon_match = (len(era5.longitude) == len(hresfc0.longitude) and
                 np.allclose(era5.longitude.values, hresfc0.longitude.values, atol=0.01))

    if not (lat_match and lon_match):
        raise ValueError(
            f"Grid mismatch! ERA5: lat={era5.latitude.shape} lon={era5.longitude.shape}, "
            f"HRES-fc0: lat={hresfc0.latitude.shape} lon={hresfc0.longitude.shape}. "
            f"Both should be on the same no_Antarctic grid (-60 to 90, 1.5° resolution)."
        )

    print(f"  ✓ Grids match: latitude {era5.latitude.shape}, longitude {era5.longitude.shape}")


def align_times(era5: xr.DataArray, hresfc0: xr.DataArray) -> tuple:
    """Align ERA5 and HRES-fc0 to common times."""
    print("Aligning to common times...")

    common_times = np.intersect1d(era5.time.values, hresfc0.time.values)
    print(f"  Common times: {len(common_times)} (ERA5: {len(era5.time)}, HRES-fc0: {len(hresfc0.time)})")

    return era5.sel(time=common_times), hresfc0.sel(time=common_times)


def compute_record_masks(
    data: xr.DataArray,
    record_max: xr.DataArray,
    record_min: xr.DataArray = None,
    var: str = "t2m"
) -> dict:
    """
    Compute boolean masks for record-breaking events.

    Returns
    -------
    dict with keys:
        'max': boolean mask where data exceeds record_max
        'min': boolean mask where data falls below record_min (t2m only)
    """
    print("Computing record masks...")

    # Add month coordinate
    data = data.assign_coords(month=data.time.dt.month)

    masks = {}

    # Max records - keep as dask array, only compute statistics
    record_max_at_time = record_max.sel(month=data.month)
    masks['max'] = data > record_max_at_time
    n_max = int(masks['max'].sum().compute().values)
    print(f"  Max records: {n_max} ({n_max / masks['max'].size * 100:.3f}%)")

    # Min records (t2m only)
    if var == "t2m" and record_min is not None:
        record_min_at_time = record_min.sel(month=data.month)
        masks['min'] = data < record_min_at_time
        n_min = int(masks['min'].sum().compute().values)
        print(f"  Min records: {n_min} ({n_min / masks['min'].size * 100:.3f}%)")

    return masks


def compute_depths(
    data: xr.DataArray,
    record_max: xr.DataArray,
    scale_max: xr.DataArray = None,
    record_min: xr.DataArray = None,
    scale_min: xr.DataArray = None,
    var: str = "t2m"
) -> dict:
    """
    Compute record depths (standardized distance from record threshold).

    For max records: depth = (data - record_max) / scale_max (positive when record broken)
    For min records: depth = (record_min - data) / scale_min (positive when record broken)

    If scale is not available, falls back to raw margins.
    """
    print("Computing record depths...")

    # Add month coordinate
    data = data.assign_coords(month=data.time.dt.month)

    depths = {}

    # Max depth - keep as dask array
    record_max_at_time = record_max.sel(month=data.month)
    if scale_max is not None:
        scale_max_at_time = scale_max.sel(month=data.month)
        depths['max'] = (data - record_max_at_time) / scale_max_at_time
        print("  Using scaled depths for max records")
    else:
        depths['max'] = data - record_max_at_time
        print("  WARNING: Scale not available for max records, using raw margins")

    # Min depth (t2m only)
    if var == "t2m" and record_min is not None:
        record_min_at_time = record_min.sel(month=data.month)
        if scale_min is not None:
            scale_min_at_time = scale_min.sel(month=data.month)
            depths['min'] = (data - record_min_at_time) / scale_min_at_time
            print("  Using scaled depths for min records")
        else:
            depths['min'] = data - record_min_at_time
            print("  WARNING: Scale not available for min records, using raw margins")

    return depths


def compare_occurrence(
    masks_era5: dict,
    masks_hresfc0: dict,
    var: str,
    case_output_dirs: dict
):
    """
    Compare record occurrence between ERA5 and HRES-fc0.

    Creates confusion matrices for max (and min if t2m).
    Excludes mutual no-record cases to focus on informative disagreements.
    """
    print("\n" + "=" * 80)
    print("COMPARING RECORD OCCURRENCE")
    print("=" * 80)

    results = {}

    for case in ['max', 'min']:
        if case not in masks_era5:
            continue

        print(f"\n{case.upper()} records:")

        # Compute masks together for efficiency
        era5_mask, hresfc0_mask = dask.compute(masks_era5[case], masks_hresfc0[case])
        era5_mask = era5_mask.values.flatten()
        hresfc0_mask = hresfc0_mask.values.flatten()

        # Confusion matrix
        # True Positive: both flag record
        # True Negative: neither flags record
        # False Positive: HRES-fc0 flags, ERA5 doesn't
        # False Negative: ERA5 flags, HRES-fc0 doesn't

        tp = np.sum(era5_mask & hresfc0_mask)
        tn = np.sum(~era5_mask & ~hresfc0_mask)
        fp = np.sum(~era5_mask & hresfc0_mask)
        fn = np.sum(era5_mask & ~hresfc0_mask)

        total = len(era5_mask)

        print(f"  True Positive (both flag):     {tp:8d} ({tp/total*100:5.2f}%)")
        print(f"  True Negative (neither flags): {tn:8d} ({tn/total*100:5.2f}%)")
        print(f"  False Positive (fc0 only):     {fp:8d} ({fp/total*100:5.2f}%)")
        print(f"  False Negative (ERA5 only):    {fn:8d} ({fn/total*100:5.2f}%)")

        # Metrics
        accuracy = (tp + tn) / total
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = np.nan
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = np.nan
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        results[case] = {
            'confusion_matrix': np.array([[tp, fn], [fp, tn]]),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        # Plot confusion matrix (excluding mutual no-record cases)
        # This focuses on the informative cases where at least one dataset flags a record
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Confusion matrix excluding mutual no-record (TN)
        # Focus on cases where at least one dataset flags a record
        total_without_tn = tp + fp + fn
        if total_without_tn > 0:
            # Matrix layout: [FN        TP]    <- ERA5 Yes
            #                [Excluded  FP]    <- ERA5 No
            conf_matrix_filtered = np.array([[fn, tp], [0, fp]])
            conf_matrix_filtered_norm = conf_matrix_filtered / total_without_tn * 100

            im = ax.imshow(conf_matrix_filtered_norm, cmap='Blues', aspect='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('% of record-flagged grid-cell-times')

            for i in range(2):
                for j in range(2):
                    if not (i == 1 and j == 0):  # Skip bottom-left (excluded TN)
                        ax.text(j, i, f'{conf_matrix_filtered_norm[i, j]:.2f}%\n(n={conf_matrix_filtered[i, j]})',
                               ha="center", va="center", color="black", fontsize=11)
                    else:
                        ax.text(j, i, 'Excluded\n(TN)',
                               ha="center", va="center", color="gray", fontsize=10, style='italic')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['HRES-fc0 No', 'HRES-fc0 Yes'])
            ax.set_yticklabels(['ERA5 Yes', 'ERA5 No'])
            ax.set_title(f'{var} {case} records: Record-Only Confusion Matrix\n({total_without_tn} events; {total_without_tn/total*100:.2f}% of total; TN excluded)')
            ax.set_ylabel('ERA5')
            ax.set_xlabel('HRES-fc0')
        else:
            ax.text(0.5, 0.5, 'No records flagged\nby either dataset',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        output_file = case_output_dirs[case] / f"confusion_matrix_{var}_{case}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    return results


def compare_intensity(
    depths_era5: dict,
    depths_hresfc0: dict,
    masks_era5: dict,
    masks_hresfc0: dict,
    var: str,
    case_output_dirs: dict
):
    """
    Compare record intensity (depths) between ERA5 and HRES-fc0.

    At grid-cell-times where at least one source flags a record,
    scatter plot and correlate the record depths.
    Record depth = (value - threshold) / scale (standardized margin).
    """
    print("\n" + "=" * 80)
    print("COMPARING RECORD INTENSITY (DEPTHS)")
    print("=" * 80)

    results = {}

    for case in ['max', 'min']:
        if case not in depths_era5:
            continue

        print(f"\n{case.upper()} records:")

        # Get depths where at least one source flags a record
        # Compute all arrays together for efficiency
        union_mask_arr, both_mask_arr, era5_depths_arr, hresfc0_depths_arr = dask.compute(
            masks_era5[case] | masks_hresfc0[case],
            masks_era5[case] & masks_hresfc0[case],
            depths_era5[case],
            depths_hresfc0[case]
        )

        union_mask = union_mask_arr.values.flatten()
        both_mask = both_mask_arr.values.flatten()
        era5_depths_flat = era5_depths_arr.values.flatten()
        hresfc0_depths_flat = hresfc0_depths_arr.values.flatten()

        era5_depths = era5_depths_flat[union_mask]
        hresfc0_depths = hresfc0_depths_flat[union_mask]

        # Also get depths where BOTH flag a record
        era5_depths_both = era5_depths_flat[both_mask]
        hresfc0_depths_both = hresfc0_depths_flat[both_mask]

        print(f"  Grid-cell-times with records (union): {len(era5_depths)}")
        print(f"  Grid-cell-times where both flag record: {len(era5_depths_both)}")

        # Remove NaNs if present (union)
        era5_nan_count = np.isnan(era5_depths).sum()
        hresfc0_nan_count = np.isnan(hresfc0_depths).sum()
        if era5_nan_count > 0 or hresfc0_nan_count > 0:
            print(f"  WARNING: NaNs detected (union) - ERA5: {era5_nan_count}, HRES-fc0: {hresfc0_nan_count}")
            valid_mask = ~(np.isnan(era5_depths) | np.isnan(hresfc0_depths))
            era5_depths = era5_depths[valid_mask]
            hresfc0_depths = hresfc0_depths[valid_mask]
            print(f"  After removing NaNs: {len(era5_depths)} points remain")

        # Remove NaNs if present (both)
        era5_nan_count_both = np.isnan(era5_depths_both).sum()
        hresfc0_nan_count_both = np.isnan(hresfc0_depths_both).sum()
        if era5_nan_count_both > 0 or hresfc0_nan_count_both > 0:
            print(f"  WARNING: NaNs detected (both) - ERA5: {era5_nan_count_both}, HRES-fc0: {hresfc0_nan_count_both}")
            valid_mask_both = ~(np.isnan(era5_depths_both) | np.isnan(hresfc0_depths_both))
            era5_depths_both = era5_depths_both[valid_mask_both]
            hresfc0_depths_both = hresfc0_depths_both[valid_mask_both]
            print(f"  After removing NaNs (both): {len(era5_depths_both)} points remain")

        # Statistics (union)
        if len(era5_depths) > 1 and np.std(era5_depths) > 0 and np.std(hresfc0_depths) > 0:
            correlation = np.corrcoef(era5_depths, hresfc0_depths)[0, 1]
            bias = np.mean(hresfc0_depths - era5_depths)
            rmse = np.sqrt(np.mean((hresfc0_depths - era5_depths)**2))
        else:
            print(f"  WARNING: Insufficient variance for correlation (std ERA5={np.std(era5_depths):.6f}, HRES-fc0={np.std(hresfc0_depths):.6f})")
            correlation = np.nan
            bias = np.nan
            rmse = np.nan

        print(f"  Correlation (union): {correlation:.4f}")
        print(f"  Bias (fc0 - ERA5, union): {bias:.4f} σ")
        print(f"  RMSE (union): {rmse:.4f} σ")

        # Statistics (both)
        if len(era5_depths_both) > 1 and np.std(era5_depths_both) > 0 and np.std(hresfc0_depths_both) > 0:
            correlation_both = np.corrcoef(era5_depths_both, hresfc0_depths_both)[0, 1]
            bias_both = np.mean(hresfc0_depths_both - era5_depths_both)
            rmse_both = np.sqrt(np.mean((hresfc0_depths_both - era5_depths_both)**2))
        else:
            print(f"  WARNING: Insufficient variance for correlation (both) (std ERA5={np.std(era5_depths_both):.6f}, HRES-fc0={np.std(hresfc0_depths_both):.6f})")
            correlation_both = np.nan
            bias_both = np.nan
            rmse_both = np.nan

        print(f"  Correlation (both): {correlation_both:.4f}")
        print(f"  Bias (fc0 - ERA5, both): {bias_both:.4f} σ")
        print(f"  RMSE (both): {rmse_both:.4f} σ")

        results[case] = {
            'correlation': correlation,
            'bias': bias,
            'rmse': rmse,
            'n_points': len(era5_depths),
            'correlation_both': correlation_both,
            'bias_both': bias_both,
            'rmse_both': rmse_both,
            'n_points_both': len(era5_depths_both),
        }

        # Scatter plot (union - at least one flags)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: scatter
        ax = axes[0]
        ax.hexbin(era5_depths, hresfc0_depths, gridsize=50, cmap='viridis', mincnt=1)
        ax.plot([era5_depths.min(), era5_depths.max()],
                [era5_depths.min(), era5_depths.max()],
                'r--', label='1:1 line')
        ax.set_xlabel(f'ERA5 record depth (σ)')
        ax.set_ylabel(f'HRES-fc0 record depth (σ)')
        ax.set_title(f'{var} {case} record depths (union)\n(r={correlation:.3f}, bias={bias:.3f}σ, RMSE={rmse:.3f}σ)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: difference histogram
        ax = axes[1]
        diff = hresfc0_depths - era5_depths
        ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax.axvline(bias, color='blue', linestyle='-', linewidth=2, label=f'Mean bias = {bias:.3f}σ')
        ax.set_xlabel(f'HRES-fc0 - ERA5 depth (σ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Record depth difference distribution (union)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = case_output_dirs[case] / f"depth_scatter_{var}_{case}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

        # NEW: Scatter plot (both flag record)
        if len(era5_depths_both) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: scatter
            ax = axes[0]
            ax.hexbin(era5_depths_both, hresfc0_depths_both, gridsize=50, cmap='viridis', mincnt=1)
            ax.plot([era5_depths_both.min(), era5_depths_both.max()],
                    [era5_depths_both.min(), era5_depths_both.max()],
                    'r--', label='1:1 line')
            ax.set_xlabel(f'ERA5 record depth (σ)')
            ax.set_ylabel(f'HRES-fc0 record depth (σ)')
            ax.set_title(f'{var} {case} record depths (both flag)\n(r={correlation_both:.3f}, bias={bias_both:.3f}σ, RMSE={rmse_both:.3f}σ)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Right: difference histogram
            ax = axes[1]
            diff_both = hresfc0_depths_both - era5_depths_both
            ax.hist(diff_both, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
            ax.axvline(bias_both, color='blue', linestyle='-', linewidth=2, label=f'Mean bias = {bias_both:.3f}σ')
            ax.set_xlabel(f'HRES-fc0 - ERA5 depth (σ)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Record depth difference distribution (both flag)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            output_file = case_output_dirs[case] / f"depth_scatter_both_{var}_{case}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_file}")
        else:
            print(f"  Skipping 'both flag' scatter plot (insufficient points: {len(era5_depths_both)})")

    return results


def compare_distributions(
    era5: xr.DataArray,
    hresfc0: xr.DataArray,
    lsm: xr.DataArray,
    var: str,
    base_output_dir: Path
):
    """
    Compare overall distributions of ERA5 vs HRES-fc0 across all 2022.

    Creates:
    - Bias/RMSE/correlation maps
    - Land/sea split statistics
    - Seasonal split statistics
    - Histograms and QQ plots

    Note: This produces variable-level plots (not case-specific),
    so they're saved to the base output directory.
    """
    print("\n" + "=" * 80)
    print("COMPARING OVERALL DISTRIBUTIONS")
    print("=" * 80)

    # Compute difference
    diff = (hresfc0 - era5).compute()

    # Overall statistics
    print("\nOverall statistics:")
    bias = float(diff.mean().values)
    rmse = float(np.sqrt((diff**2).mean()).values)
    correlation = float(xr.corr(era5, hresfc0).values)

    print(f"  Bias (fc0 - ERA5): {bias:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {correlation:.4f}")

    # Spatial maps
    print("\nCreating spatial maps...")
    bias_map = diff.mean(dim='time')
    rmse_map = np.sqrt((diff**2).mean(dim='time'))
    corr_map = xr.corr(era5, hresfc0, dim='time')

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Bias map (transpose to match pcolormesh expectations: Y, X)
    im = axes[0].pcolormesh(
        bias_map.longitude, bias_map.latitude, bias_map.T,
        cmap='RdBu_r', vmin=-np.abs(bias_map).max(), vmax=np.abs(bias_map).max(),
        shading='auto'
    )
    axes[0].set_title(f'{var} Bias (HRES-fc0 - ERA5) - mean: {bias:.3f}')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0], label=var)

    # RMSE map
    im = axes[1].pcolormesh(
        rmse_map.longitude, rmse_map.latitude, rmse_map.T,
        cmap='Reds',
        shading='auto'
    )
    axes[1].set_title(f'{var} RMSE - mean: {rmse:.3f}')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[1], label=var)

    # Correlation map
    im = axes[2].pcolormesh(
        corr_map.longitude, corr_map.latitude, corr_map.T,
        cmap='RdYlGn', vmin=0.9, vmax=1.0,
        shading='auto'
    )
    axes[2].set_title(f'{var} Correlation - mean: {correlation:.3f}')
    axes[2].set_ylabel('Latitude')
    axes[2].set_xlabel('Longitude')
    plt.colorbar(im, ax=axes[2], label='Correlation')

    output_file = base_output_dir / f"spatial_comparison_{var}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    # Land/sea split
    print("\nLand/sea statistics:")

    # Prepare land/sea mask (squeeze out time dimension if present)
    lsm_2d = lsm.squeeze() if 'time' in lsm.dims else lsm
    lsm_regrid = lsm_2d.interp(latitude=era5.latitude, longitude=era5.longitude, method='nearest')
    land_mask = lsm_regrid > 0.5

    for region_name, mask in [("Land", land_mask), ("Sea", ~land_mask)]:
        # Use skipna=True to handle masked values
        diff_region = diff.where(mask)
        era5_region = era5.where(mask)
        hresfc0_region = hresfc0.where(mask)

        bias_region = float(diff_region.mean(skipna=True).values)
        rmse_region = float(np.sqrt((diff_region**2).mean(skipna=True)).values)

        # Check if we have valid data before computing correlation
        if era5_region.count() > 0 and hresfc0_region.count() > 0:
            corr_region = float(xr.corr(era5_region, hresfc0_region, dim=['time', 'latitude', 'longitude']).values)
        else:
            corr_region = np.nan

        print(f"  {region_name}:")
        print(f"    Bias: {bias_region:.4f}")
        print(f"    RMSE: {rmse_region:.4f}")
        print(f"    Correlation: {corr_region:.4f}")

    # Seasonal split
    print("\nSeasonal statistics:")

    era5_with_season = era5.assign_coords(season=era5.time.dt.season)
    hresfc0_with_season = hresfc0.assign_coords(season=hresfc0.time.dt.season)
    diff_with_season = diff.assign_coords(season=diff.time.dt.season)

    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        diff_season = diff_with_season.where(diff_with_season.season == season, drop=True)
        era5_season = era5_with_season.where(era5_with_season.season == season, drop=True)
        hresfc0_season = hresfc0_with_season.where(hresfc0_with_season.season == season, drop=True)

        if len(diff_season.time) == 0:
            continue

        bias_season = float(diff_season.mean().values)
        rmse_season = float(np.sqrt((diff_season**2).mean()).values)
        corr_season = float(xr.corr(era5_season, hresfc0_season).values)

        print(f"  {season}:")
        print(f"    Bias: {bias_season:.4f}")
        print(f"    RMSE: {rmse_season:.4f}")
        print(f"    Correlation: {corr_season:.4f}")

    # Variability comparison
    print("\nVariability comparison:")
    era5_std = float(era5.std().values)
    hresfc0_std = float(hresfc0.std().values)
    print(f"  ERA5 std dev: {era5_std:.4f}")
    print(f"  HRES-fc0 std dev: {hresfc0_std:.4f}")
    print(f"  Ratio (fc0/ERA5): {hresfc0_std/era5_std:.4f}")

    # Spatial variability comparison
    print("\nCreating variability comparison plot...")
    era5_std_map = era5.std(dim='time')
    hresfc0_std_map = hresfc0.std(dim='time')
    std_ratio_map = hresfc0_std_map / era5_std_map

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # ERA5 std map
    im = axes[0].pcolormesh(
        era5_std_map.longitude, era5_std_map.latitude, era5_std_map.T,
        cmap='viridis',
        shading='auto'
    )
    axes[0].set_title(f'{var} Standard Deviation - ERA5 (mean: {era5_std:.3f})')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0], label=var)

    # HRES-fc0 std map
    im = axes[1].pcolormesh(
        hresfc0_std_map.longitude, hresfc0_std_map.latitude, hresfc0_std_map.T,
        cmap='viridis',
        shading='auto'
    )
    axes[1].set_title(f'{var} Standard Deviation - HRES-fc0 (mean: {hresfc0_std:.3f})')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[1], label=var)

    # Ratio map
    im = axes[2].pcolormesh(
        std_ratio_map.longitude, std_ratio_map.latitude, std_ratio_map.T,
        cmap='RdBu_r', vmin=0.8, vmax=1.2,
        shading='auto'
    )
    axes[2].set_title(f'{var} Variability Ratio (HRES-fc0 / ERA5) - mean: {hresfc0_std/era5_std:.3f}')
    axes[2].set_ylabel('Latitude')
    axes[2].set_xlabel('Longitude')
    plt.colorbar(im, ax=axes[2], label='Ratio')

    output_file = base_output_dir / f"variability_comparison_{var}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    return {
        'overall': {'bias': bias, 'rmse': rmse, 'correlation': correlation},
        'spatial_maps': {'bias': bias_map, 'rmse': rmse_map, 'correlation': corr_map},
        'variability': {'era5_std': era5_std, 'hresfc0_std': hresfc0_std, 'ratio': hresfc0_std/era5_std},
    }


def compare_peak_depth_snapshot(
    depths_era5: dict,
    depths_hresfc0: dict,
    masks_era5: dict,
    masks_hresfc0: dict,
    lsm: xr.DataArray,
    var: str,
    case_output_dirs: dict
):
    """
    Find datetime with highest global absolute depth-area and visualize spatial patterns.

    For each case (max/min), find the time with maximum global |depth| * area_weights,
    then create side-by-side maps of ERA5 and HRES-fc0 depths at that time.
    """
    print("\n" + "=" * 80)
    print("PEAK DEPTH SNAPSHOT COMPARISON")
    print("=" * 80)

    # Compute latitude weights (cosine of latitude)
    lats = depths_era5['max'].latitude.values
    lat_weights = np.cos(np.deg2rad(lats))
    lat_weights = lat_weights / lat_weights.sum()  # Normalize

    for case in ['max', 'min']:
        if case not in depths_era5:
            continue

        print(f"\n{case.upper()} records:")

        # Mask depths to only where records occur, then take absolute value
        depths_masked_era5 = depths_era5[case].where(masks_era5[case])
        depths_masked_hresfc0 = depths_hresfc0[case].where(masks_hresfc0[case])

        depths_abs_era5 = np.abs(depths_masked_era5)
        depths_abs_hresfc0 = np.abs(depths_masked_hresfc0)

        # Weight by latitude (broadcast across longitude)
        weighted_depth_era5 = depths_abs_era5 * xr.DataArray(
            lat_weights, dims=['latitude'], coords={'latitude': lats}
        )
        weighted_depth_hresfc0 = depths_abs_hresfc0 * xr.DataArray(
            lat_weights, dims=['latitude'], coords={'latitude': lats}
        )

        # Sum over space to get time series of global weighted depth-area (skipna=True for masked values)
        global_depth_era5 = weighted_depth_era5.sum(dim=['latitude', 'longitude'], skipna=True)
        global_depth_hresfc0 = weighted_depth_hresfc0.sum(dim=['latitude', 'longitude'], skipna=True)

        # Compute and find peak time for both datasets
        global_depth_era5_values = global_depth_era5.compute()
        global_depth_hresfc0_values = global_depth_hresfc0.compute()

        peak_time_era5 = global_depth_era5_values.idxmax(dim='time').values
        peak_time_hresfc0 = global_depth_hresfc0_values.idxmax(dim='time').values

        peak_value_era5 = float(global_depth_era5_values.sel(time=peak_time_era5).values)
        peak_value_hresfc0 = float(global_depth_hresfc0_values.sel(time=peak_time_hresfc0).values)

        # Use HRES-fc0 peak time (since we're comparing to HRES-fc0 as verification)
        peak_time = peak_time_hresfc0

        print(f"  Peak global weighted depth-area:")
        print(f"    ERA5:     {pd.Timestamp(peak_time_era5).strftime('%Y-%m-%d %H:%M')} (value: {peak_value_era5:.2f})")
        print(f"    HRES-fc0: {pd.Timestamp(peak_time_hresfc0).strftime('%Y-%m-%d %H:%M')} (value: {peak_value_hresfc0:.2f})")
        print(f"  Using HRES-fc0 peak time for comparison: {pd.Timestamp(peak_time).strftime('%Y-%m-%d %H:%M')}")

        # Extract depths and masks at peak time
        depth_era5_snap = depths_era5[case].sel(time=peak_time).compute()
        depth_hresfc0_snap = depths_hresfc0[case].sel(time=peak_time).compute()
        mask_era5_snap = masks_era5[case].sel(time=peak_time).compute()
        mask_hresfc0_snap = masks_hresfc0[case].sel(time=peak_time).compute()

        # Mask depths to only show where records occur
        depth_era5_snap_masked = depth_era5_snap.where(mask_era5_snap)
        depth_hresfc0_snap_masked = depth_hresfc0_snap.where(mask_hresfc0_snap)

        # Create side-by-side maps
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Determine color scale limits and colormap based on variable and case
        # Clip colormap range to avoid white at zero while keeping colorbar sensible
        from matplotlib.colors import LinearSegmentedColormap

        if var == 't2m' and case == 'max':
            # Clip Reds colormap to start at 30% (skip white colors)
            base_cmap = plt.cm.Reds
            cmap = LinearSegmentedColormap.from_list(
                'Reds_clipped', base_cmap(np.linspace(0.3, 1.0, 256))
            )
            vmin = 0
            vmax = max(
                np.nanmax(depth_era5_snap_masked.values),
                np.nanmax(depth_hresfc0_snap_masked.values)
            )
        elif var == 't2m' and case == 'min':
            # Clip Blues_r colormap to start at 30% (skip white colors)
            base_cmap = plt.cm.Blues_r
            cmap = LinearSegmentedColormap.from_list(
                'Blues_r_clipped', base_cmap(np.linspace(0.3, 1.0, 256))
            )
            vmax = 0
            vmin = min(
                np.nanmin(depth_era5_snap_masked.values),
                np.nanmin(depth_hresfc0_snap_masked.values)
            )
        else:  # w10 max
            # Clip YlOrBr colormap to start at 20% (skip very light yellows)
            base_cmap = plt.cm.YlOrBr
            cmap = LinearSegmentedColormap.from_list(
                'YlOrBr_clipped', base_cmap(np.linspace(0.2, 1.0, 256))
            )
            vmin = 0
            vmax = max(
                np.nanmax(depth_era5_snap_masked.values),
                np.nanmax(depth_hresfc0_snap_masked.values)
            )

        # Prepare land-sea mask for contouring (squeeze time dimension if present)
        lsm_2d = lsm.squeeze() if 'time' in lsm.dims else lsm

        # ERA5 map (masked to only show records)
        im = axes[0].pcolormesh(
            depth_era5_snap_masked.longitude, depth_era5_snap_masked.latitude, depth_era5_snap_masked.T,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading='auto'
        )
        axes[0].contour(
            lsm_2d.longitude, lsm_2d.latitude, lsm_2d.T,
            levels=[0.5], colors='black', linewidths=0.5, alpha=0.6
        )
        axes[0].set_title(f'ERA5 - {pd.Timestamp(peak_time).strftime("%Y-%m-%d %H:%M UTC")}')
        axes[0].set_ylabel('Latitude')
        axes[0].set_xlabel('Longitude')
        plt.colorbar(im, ax=axes[0], label=f'Record depth (σ)')

        # HRES-fc0 map (masked to only show records)
        im = axes[1].pcolormesh(
            depth_hresfc0_snap_masked.longitude, depth_hresfc0_snap_masked.latitude, depth_hresfc0_snap_masked.T,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading='auto'
        )
        axes[1].contour(
            lsm_2d.longitude, lsm_2d.latitude, lsm_2d.T,
            levels=[0.5], colors='black', linewidths=0.5, alpha=0.6
        )
        axes[1].set_title(f'HRES-fc0 - {pd.Timestamp(peak_time).strftime("%Y-%m-%d %H:%M UTC")}')
        axes[1].set_ylabel('Latitude')
        axes[1].set_xlabel('Longitude')
        plt.colorbar(im, ax=axes[1], label=f'Record depth (σ)')

        fig.suptitle(f'{var.upper()} {case.upper()} record depths - Peak global depth-area snapshot',
                     fontsize=14, y=1.00)

        output_file = case_output_dirs[case] / f"peak_snapshot_{var}_{case}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ERA5 vs HRES-fc0 as 2022 verification datasets"
    )
    parser.add_argument(
        "--var",
        required=True,
        choices=["t2m", "w10"],
        help="Variable to analyze"
    )
    parser.add_argument(
        "--record-tag",
        default="1979_2021",
        help="ERA5 record tag (default: 1979_2021)"
    )
    parser.add_argument(
        "--wb2-path",
        type=Path,
        default=Path("/data/gent/vo/000/gvo00090/HEAT/Data/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"),
        help="Path to WeatherBench2 ERA5 data"
    )
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("/data/gent/vo/000/gvo00090/HEAT/Data/forecasts"),
        help="Root directory for forecast data"
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path("/data/gent/vo/000/gvo00090/HEAT/Data/wb2"),
        help="Directory containing ERA5 record files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/2022_records/results"),
        help="Base output directory for results (will create t2m_min/, t2m_max/, w10_max/ subdirs)"
    )

    args = parser.parse_args()

    # Create base output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create case-specific output directories based on variable
    case_output_dirs = {}
    if args.var == "t2m":
        case_output_dirs["max"] = args.output_dir / "t2m_max"
        case_output_dirs["min"] = args.output_dir / "t2m_min"
    else:  # w10
        case_output_dirs["max"] = args.output_dir / "w10_max"

    # Create the case directories
    for case_dir in case_output_dirs.values():
        case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ERA5 vs HRES-fc0 Verification Comparison")
    print("=" * 80)
    print(f"Variable: {args.var}")
    print(f"Record tag: {args.record_tag}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    era5 = load_era5_2022(args.var, args.wb2_path)
    hresfc0 = load_hresfc0_2022(args.var, args.forecast_root)
    (record_max, scale_max), (record_min, scale_min) = load_era5_records(args.var, args.record_tag, args.records_dir)

    # Load land-sea mask
    lsm_file = args.records_dir / "ERA5_lsm_sfc.nc"
    lsm = xr.open_dataset(lsm_file)["lsm"]
    lsm = functions.normalize_spatial_coords(lsm)

    # Verify grids match and align times
    verify_grid_match(era5, hresfc0)
    era5_aligned, hresfc0_aligned = align_times(era5, hresfc0)

    # Compute record masks and depths
    masks_era5 = compute_record_masks(era5_aligned, record_max, record_min, args.var)
    masks_hresfc0 = compute_record_masks(hresfc0_aligned, record_max, record_min, args.var)

    depths_era5 = compute_depths(era5_aligned, record_max, scale_max, record_min, scale_min, args.var)
    depths_hresfc0 = compute_depths(hresfc0_aligned, record_max, scale_max, record_min, scale_min, args.var)

    # Ensure depths have same dimension order as masks
    # Masks come from data which has dims (time, longitude, latitude)
    # But depths might have (time, latitude, longitude) due to record file dimension order
    for key in depths_era5.keys():
        if depths_era5[key].dims != masks_era5[key].dims:
            depths_era5[key] = depths_era5[key].transpose(*masks_era5[key].dims)
    for key in depths_hresfc0.keys():
        if depths_hresfc0[key].dims != masks_hresfc0[key].dims:
            depths_hresfc0[key] = depths_hresfc0[key].transpose(*masks_hresfc0[key].dims)

    # Run comparisons
    occurrence_results = compare_occurrence(masks_era5, masks_hresfc0, args.var, case_output_dirs)
    intensity_results = compare_intensity(depths_era5, depths_hresfc0, masks_era5, masks_hresfc0, args.var, case_output_dirs)
    distribution_results = compare_distributions(era5_aligned, hresfc0_aligned, lsm, args.var, args.output_dir)
    compare_peak_depth_snapshot(depths_era5, depths_hresfc0, masks_era5, masks_hresfc0, lsm, args.var, case_output_dirs)

    # Save summary (one per case)
    for case in case_output_dirs.keys():
        if case not in occurrence_results:
            continue

        summary_file = case_output_dirs[case] / f"summary_{args.var}_{case}.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ERA5 vs HRES-fc0 Comparison Summary - {args.var} {case.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Record tag: {args.record_tag}\n\n")

            f.write("OCCURRENCE COMPARISON\n")
            f.write("-" * 80 + "\n")
            results = occurrence_results[case]
            f.write(f"{case.upper()} records:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1 Score:  {results['f1']:.4f}\n\n")

            f.write("INTENSITY COMPARISON (RECORD DEPTHS)\n")
            f.write("-" * 80 + "\n")
            results = intensity_results[case]
            f.write(f"{case.upper()} records:\n")
            f.write(f"  Correlation (union): {results['correlation']:.4f}\n")
            f.write(f"  Bias (σ, union):     {results['bias']:.4f}\n")
            f.write(f"  RMSE (σ, union):     {results['rmse']:.4f}\n")
            f.write(f"  N points (union):    {results['n_points']}\n\n")
            f.write(f"  Correlation (both):  {results['correlation_both']:.4f}\n")
            f.write(f"  Bias (σ, both):      {results['bias_both']:.4f}\n")
            f.write(f"  RMSE (σ, both):      {results['rmse_both']:.4f}\n")
            f.write(f"  N points (both):     {results['n_points_both']}\n\n")

            f.write("DISTRIBUTION COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall:\n")
            f.write(f"  Bias:        {distribution_results['overall']['bias']:.4f}\n")
            f.write(f"  RMSE:        {distribution_results['overall']['rmse']:.4f}\n")
            f.write(f"  Correlation: {distribution_results['overall']['correlation']:.4f}\n\n")

            f.write("VARIABILITY COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"ERA5 std dev:        {distribution_results['variability']['era5_std']:.4f}\n")
            f.write(f"HRES-fc0 std dev:    {distribution_results['variability']['hresfc0_std']:.4f}\n")
            f.write(f"Ratio (fc0/ERA5):    {distribution_results['variability']['ratio']:.4f}\n")

        print(f"\nSummary saved to: {summary_file}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
