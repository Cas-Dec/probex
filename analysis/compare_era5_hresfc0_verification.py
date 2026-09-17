#!/usr/bin/env python
"""
Compare 2022 ERA5 vs HRES-fc0 as verification datasets.

This script checks if ERA5 and HRES-fc0 agree on:
1. Record occurrence (confusion matrix)
2. Record intensity (margin comparison)
3. Overall distribution (bias, RMSE, correlation, QQ plots)

Motivated by discrepancies in "Expected - observed records count" where IFS
showed too many heat records and too few cold/wind records, suggesting possible
distribution shifts between ERA5 and HRES-fc0.

Usage:
    python analysis/compare_era5_hresfc0_verification.py --var t2m --record-tag 1979_2021
    python analysis/compare_era5_hresfc0_verification.py --var w10 --record-tag 1979_2021
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
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
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
    """Load ERA5 monthly record thresholds."""
    print(f"Loading ERA5 records ({record_tag}) for {var}...")

    # Load max records (heat for t2m, high wind for w10)
    record_max_file = records_dir / f"ERA5_record_max_month_{record_tag}_{var}.nc"
    # Variable name is the filename without .nc extension
    var_name_max = f"ERA5_record_max_month_{record_tag}_{var}"
    record_max = xr.open_dataset(record_max_file)[var_name_max]

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

    # Convert temperature records from Kelvin to Celsius
    if var == "t2m":
        # Check if records are in Kelvin (values > 100)
        if float(record_max.max().values) > 100:
            record_max = utils.Kelvin2Celsius(record_max)

    # Load min records (cold for t2m only)
    record_min = None
    if var == "t2m":
        record_min_file = records_dir / f"ERA5_record_min_month_{record_tag}_{var}.nc"
        var_name_min = f"ERA5_record_min_month_{record_tag}_{var}"
        record_min = xr.open_dataset(record_min_file)[var_name_min]

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

    return record_max, record_min


def align_datasets(era5: xr.DataArray, hresfc0: xr.DataArray) -> tuple:
    """Align ERA5 and HRES-fc0 to common times and verify spatial grids match."""
    print("Aligning datasets...")

    # Find common times
    common_times = np.intersect1d(era5.time.values, hresfc0.time.values)
    print(f"  Common times: {len(common_times)}")

    era5_aligned = era5.sel(time=common_times)
    hresfc0_aligned = hresfc0.sel(time=common_times)

    # Verify spatial grids match (both should be no_Antarctic: -60 to 90, 1.5° resolution)
    lat_match = (len(era5_aligned.latitude) == len(hresfc0_aligned.latitude) and
                 np.allclose(era5_aligned.latitude.values, hresfc0_aligned.latitude.values, atol=0.01))
    lon_match = (len(era5_aligned.longitude) == len(hresfc0_aligned.longitude) and
                 np.allclose(era5_aligned.longitude.values, hresfc0_aligned.longitude.values, atol=0.01))

    if not (lat_match and lon_match):
        raise ValueError(
            f"Grid mismatch! ERA5: {era5_aligned.latitude.shape}, HRES-fc0: {hresfc0_aligned.latitude.shape}. "
            f"Both should be on the same no_Antarctic grid (-60 to 90, 1.5° resolution)."
        )

    print(f"  Grids match: {era5_aligned.latitude.shape}")

    return era5_aligned, hresfc0_aligned


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


def compute_margins(
    data: xr.DataArray,
    record_max: xr.DataArray,
    record_min: xr.DataArray = None,
    var: str = "t2m"
) -> dict:
    """
    Compute margins (distance from record threshold).

    For max records: margin = data - record_max (positive when record broken)
    For min records: margin = record_min - data (positive when record broken)
    """
    print("Computing margins...")

    # Add month coordinate
    data = data.assign_coords(month=data.time.dt.month)

    margins = {}

    # Max margin - keep as dask array
    record_max_at_time = record_max.sel(month=data.month)
    margins['max'] = data - record_max_at_time

    # Min margin (t2m only)
    if var == "t2m" and record_min is not None:
        record_min_at_time = record_min.sel(month=data.month)
        margins['min'] = record_min_at_time - data

    return margins


def compare_occurrence(
    masks_era5: dict,
    masks_hresfc0: dict,
    var: str,
    output_dir: Path
):
    """
    Compare record occurrence between ERA5 and HRES-fc0.

    Creates confusion matrices for max (and min if t2m).
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

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        conf_matrix_norm = np.array([[tp, fn], [fp, tn]]) / total * 100

        # Plot heatmap with matplotlib
        im = ax.imshow(conf_matrix_norm, cmap='Blues', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage of total grid-cell-times')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{conf_matrix_norm[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=14)

        # Set ticks and labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['HRES-fc0 No', 'HRES-fc0 Yes'])
        ax.set_yticklabels(['ERA5 No', 'ERA5 Yes'])
        ax.set_title(f'{var} {case} records: ERA5 vs HRES-fc0 Confusion Matrix')
        ax.set_ylabel('ERA5')
        ax.set_xlabel('HRES-fc0')

        output_file = output_dir / f"confusion_matrix_{var}_{case}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    return results


def compare_intensity(
    margins_era5: dict,
    margins_hresfc0: dict,
    masks_era5: dict,
    masks_hresfc0: dict,
    var: str,
    output_dir: Path
):
    """
    Compare record intensity (margins) between ERA5 and HRES-fc0.

    At grid-cell-times where at least one source flags a record,
    scatter plot and correlate the margins.
    """
    print("\n" + "=" * 80)
    print("COMPARING RECORD INTENSITY (MARGINS)")
    print("=" * 80)

    results = {}

    for case in ['max', 'min']:
        if case not in margins_era5:
            continue

        print(f"\n{case.upper()} records:")

        # Get margins where at least one source flags a record
        # Compute all arrays together for efficiency
        union_mask_arr, era5_margins_arr, hresfc0_margins_arr = dask.compute(
            masks_era5[case] | masks_hresfc0[case],
            margins_era5[case],
            margins_hresfc0[case]
        )

        union_mask = union_mask_arr.values.flatten()
        era5_margins = era5_margins_arr.values.flatten()[union_mask]
        hresfc0_margins = hresfc0_margins_arr.values.flatten()[union_mask]

        print(f"  Grid-cell-times with records: {len(era5_margins)}")

        # Debug: Check for NaNs
        era5_nan_count = np.isnan(era5_margins).sum()
        hresfc0_nan_count = np.isnan(hresfc0_margins).sum()
        if era5_nan_count > 0 or hresfc0_nan_count > 0:
            print(f"  WARNING: NaNs detected - ERA5: {era5_nan_count}, HRES-fc0: {hresfc0_nan_count}")
            # Remove NaNs
            valid_mask = ~(np.isnan(era5_margins) | np.isnan(hresfc0_margins))
            era5_margins = era5_margins[valid_mask]
            hresfc0_margins = hresfc0_margins[valid_mask]
            print(f"  After removing NaNs: {len(era5_margins)} points remain")

        # Statistics
        if len(era5_margins) > 1 and np.std(era5_margins) > 0 and np.std(hresfc0_margins) > 0:
            correlation = np.corrcoef(era5_margins, hresfc0_margins)[0, 1]
            bias = np.mean(hresfc0_margins - era5_margins)
            rmse = np.sqrt(np.mean((hresfc0_margins - era5_margins)**2))
        else:
            print(f"  WARNING: Insufficient variance for correlation (std ERA5={np.std(era5_margins):.6f}, HRES-fc0={np.std(hresfc0_margins):.6f})")
            correlation = np.nan
            bias = np.nan
            rmse = np.nan

        print(f"  Correlation: {correlation:.4f}")
        print(f"  Bias (fc0 - ERA5): {bias:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        results[case] = {
            'correlation': correlation,
            'bias': bias,
            'rmse': rmse,
            'n_points': len(era5_margins),
        }

        # Scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: scatter
        ax = axes[0]
        ax.hexbin(era5_margins, hresfc0_margins, gridsize=50, cmap='viridis', mincnt=1)
        ax.plot([era5_margins.min(), era5_margins.max()],
                [era5_margins.min(), era5_margins.max()],
                'r--', label='1:1 line')
        ax.set_xlabel(f'ERA5 margin ({var})')
        ax.set_ylabel(f'HRES-fc0 margin ({var})')
        ax.set_title(f'{var} {case} record margins\n(r={correlation:.3f}, bias={bias:.3f}, RMSE={rmse:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: difference histogram
        ax = axes[1]
        diff = hresfc0_margins - era5_margins
        ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax.axvline(bias, color='blue', linestyle='-', linewidth=2, label=f'Mean bias = {bias:.3f}')
        ax.set_xlabel(f'HRES-fc0 - ERA5 margin ({var})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Margin difference distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = output_dir / f"margin_scatter_{var}_{case}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")

    return results


def compare_distributions(
    era5: xr.DataArray,
    hresfc0: xr.DataArray,
    lsm: xr.DataArray,
    var: str,
    output_dir: Path
):
    """
    Compare overall distributions of ERA5 vs HRES-fc0 across all 2022.

    Creates:
    - Bias/RMSE/correlation maps
    - Land/sea split statistics
    - Seasonal split statistics
    - Histograms and QQ plots
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

    output_file = output_dir / f"spatial_comparison_{var}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    # Land/sea split
    print("\nLand/sea statistics:")

    # Prepare land/sea mask
    lsm_regrid = lsm.interp(latitude=era5.latitude, longitude=era5.longitude, method='nearest')
    land_mask = lsm_regrid > 0.5

    for region_name, mask in [("Land", land_mask), ("Sea", ~land_mask)]:
        diff_region = diff.where(mask)
        bias_region = float(diff_region.mean().values)
        rmse_region = float(np.sqrt((diff_region**2).mean()).values)
        corr_region = float(xr.corr(era5.where(mask), hresfc0.where(mask)).values)

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

    # Histogram comparison
    print("\nCreating histogram and QQ plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax = axes[0]
    era5_flat = era5.values.flatten()
    hresfc0_flat = hresfc0.values.flatten()

    ax.hist(era5_flat, bins=100, alpha=0.5, label='ERA5', density=True)
    ax.hist(hresfc0_flat, bins=100, alpha=0.5, label='HRES-fc0', density=True)
    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.set_title(f'{var} Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # QQ plot
    ax = axes[1]
    quantiles = np.linspace(0, 1, 1000)
    era5_quantiles = np.quantile(era5_flat, quantiles)
    hresfc0_quantiles = np.quantile(hresfc0_flat, quantiles)

    ax.scatter(era5_quantiles, hresfc0_quantiles, s=1, alpha=0.5)
    ax.plot([era5_quantiles.min(), era5_quantiles.max()],
            [era5_quantiles.min(), era5_quantiles.max()],
            'r--', label='1:1 line')
    ax.set_xlabel(f'ERA5 quantiles ({var})')
    ax.set_ylabel(f'HRES-fc0 quantiles ({var})')
    ax.set_title('QQ Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = output_dir / f"distribution_comparison_{var}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")

    return {
        'overall': {'bias': bias, 'rmse': rmse, 'correlation': correlation},
        'spatial_maps': {'bias': bias_map, 'rmse': rmse_map, 'correlation': corr_map},
    }


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
        default=Path("data/era5_records"),
        help="Directory containing ERA5 record files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/era5_vs_hresfc0"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ERA5 vs HRES-fc0 Verification Comparison")
    print("=" * 80)
    print(f"Variable: {args.var}")
    print(f"Record tag: {args.record_tag}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    era5 = load_era5_2022(args.var, args.wb2_path)
    hresfc0 = load_hresfc0_2022(args.var, args.forecast_root)
    record_max, record_min = load_era5_records(args.var, args.record_tag, args.records_dir)

    # Load land-sea mask
    lsm_file = args.records_dir / "ERA5_lsm_sfc.nc"
    lsm = xr.open_dataset(lsm_file)["lsm"]
    lsm = functions.normalize_spatial_coords(lsm)

    # Align datasets
    era5_aligned, hresfc0_aligned = align_datasets(era5, hresfc0)

    # Compute record masks and margins
    masks_era5 = compute_record_masks(era5_aligned, record_max, record_min, args.var)
    masks_hresfc0 = compute_record_masks(hresfc0_aligned, record_max, record_min, args.var)

    margins_era5 = compute_margins(era5_aligned, record_max, record_min, args.var)
    margins_hresfc0 = compute_margins(hresfc0_aligned, record_max, record_min, args.var)

    # Run comparisons
    occurrence_results = compare_occurrence(masks_era5, masks_hresfc0, args.var, args.output_dir)
    intensity_results = compare_intensity(margins_era5, margins_hresfc0, masks_era5, masks_hresfc0, args.var, args.output_dir)
    distribution_results = compare_distributions(era5_aligned, hresfc0_aligned, lsm, args.var, args.output_dir)

    # Save summary
    summary_file = args.output_dir / f"summary_{args.var}.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ERA5 vs HRES-fc0 Comparison Summary - {args.var}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Record tag: {args.record_tag}\n\n")

        f.write("OCCURRENCE COMPARISON\n")
        f.write("-" * 80 + "\n")
        for case, results in occurrence_results.items():
            f.write(f"{case.upper()} records:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1 Score:  {results['f1']:.4f}\n\n")

        f.write("INTENSITY COMPARISON\n")
        f.write("-" * 80 + "\n")
        for case, results in intensity_results.items():
            f.write(f"{case.upper()} records:\n")
            f.write(f"  Correlation: {results['correlation']:.4f}\n")
            f.write(f"  Bias:        {results['bias']:.4f}\n")
            f.write(f"  RMSE:        {results['rmse']:.4f}\n")
            f.write(f"  N points:    {results['n_points']}\n\n")

        f.write("DISTRIBUTION COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall:\n")
        f.write(f"  Bias:        {distribution_results['overall']['bias']:.4f}\n")
        f.write(f"  RMSE:        {distribution_results['overall']['rmse']:.4f}\n")
        f.write(f"  Correlation: {distribution_results['overall']['correlation']:.4f}\n")

    print(f"\nSummary saved to: {summary_file}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
