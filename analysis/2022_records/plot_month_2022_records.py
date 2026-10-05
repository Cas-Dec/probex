#!/usr/bin/env python
"""
Plot 2022 monthly record analysis with 4 panels:
- Top left: Historical 1979-2021 monthly maximum/minimum records
- Top right: 2022 record exceedances (K or m/s)
- Bottom left: Scales (SD of yearly monthly extremes)
- Bottom right: 2022 record depth (sigma units)

Usage:
    python analysis/2022_records/plot_month_2022_records.py --month 7 --var t2m --truth ERA5 --case max
    python analysis/2022_records/plot_month_2022_records.py --month 1 --var w10 --truth HRES_fc0 --case max
    python analysis/2022_records/plot_month_2022_records.py --month 12 --var t2m --truth ERA5 --case min
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add eval directory for utility functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
import utils


# Month names
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# Month abbreviations
MONTH_ABBREV = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


def load_records(var: str, case: str, record_tag: str = "1979_2021") -> tuple:
    """Load ERA5 monthly record thresholds and scales."""
    records_dir = Path("/data/gent/vo/000/gvo00090/HEAT/Data/wb2")
    record_file = records_dir / f"ERA5_record_{case}_month_{record_tag}_{var}.nc"

    print(f"Loading ERA5 {case} records for {var}...")
    ds = xr.open_dataset(record_file)
    record = ds["record"]
    scale = ds["scale"] if "scale" in ds else None

    return record, scale


def load_truth_2022(var: str, truth: str, month: int) -> xr.DataArray:
    """Load 2022 observations from ERA5 or HRES-fc0."""
    var_map = {"t2m": "2m_temperature", "w10": "10m_wind_speed"}

    if truth == "ERA5":
        print(f"Loading {truth} 2022 data for {var}...")
        era5_zarr = "/data/gent/vo/000/gvo00090/HEAT/Data/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
        ds = xr.open_zarr(era5_zarr, chunks={})
        data = ds[var_map[var]].sel(time=slice(f"2022-{month:02d}", f"2022-{month:02d}"))

        # Filter to 00/12 UTC (matching record computation)
        data = data.where(data.time.dt.hour.isin([0, 12]), drop=True)

        # Normalize coordinates
        data = utils.change_lonlat(data)
        data = data.rename({"lon": "longitude", "lat": "latitude"})

    elif truth == "HRES_fc0":
        print(f"Loading {truth} 2022 data for {var}...")
        hresfc0_file = f"/data/gent/vo/000/gvo00090/HEAT/Data/forecasts/HRES_fc0/HRES_fc0_2022_no_Antarctic_{var}_1p50.zarr"
        ds = xr.open_zarr(hresfc0_file, chunks={})
        data = ds[var_map[var]]

        # Remove prediction_timedelta dimension (fc0 only)
        if "prediction_timedelta" in data.dims:
            data = data.isel(prediction_timedelta=0)

        # Filter to target month
        data = data.sel(time=slice(f"2022-{month:02d}", f"2022-{month:02d}"))

    else:
        raise ValueError(f"Unknown truth dataset: {truth}")

    # Slice to no_Antarctic domain
    data = data.sel(latitude=slice(-60, 90))

    print(f"  Shape: {data.shape}")
    print(f"  Time range: {data.time.values[0]} to {data.time.values[-1]}")

    return data


def load_lsm() -> xr.DataArray:
    """Load land-sea mask."""
    lsm_file = "/data/gent/vo/000/gvo00090/HEAT/Data/wb2/ERA5_lsm_sfc.nc"
    ds = xr.open_dataset(lsm_file)
    lsm = ds["lsm"].isel(time=0)
    return lsm


def compute_monthly_extreme(data: xr.DataArray, case: str) -> xr.DataArray:
    """Compute monthly maximum or minimum."""
    if case == "max":
        return data.max(dim="time", skipna=True)
    elif case == "min":
        return data.min(dim="time", skipna=True)
    else:
        raise ValueError(f"Unknown case: {case}")


def normalize_coords(da: xr.DataArray) -> xr.DataArray:
    """Normalize coordinates to lat/lon."""
    if "latitude" in da.coords:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da


def main():
    parser = argparse.ArgumentParser(description="Plot 2022 monthly record analysis")
    parser.add_argument("--month", type=int, required=True, help="Month to analyze (1-12)")
    parser.add_argument("--var", type=str, required=True, choices=["t2m", "w10"], help="Variable")
    parser.add_argument("--truth", type=str, required=True, choices=["ERA5", "HRES_fc0"], help="Truth dataset")
    parser.add_argument("--case", type=str, required=True, choices=["max", "min"], help="Record type")
    parser.add_argument("--record-tag", type=str, default="1979_2021", help="Record tag")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/2022_records"), help="Output directory")

    args = parser.parse_args()

    # Validate inputs
    if args.var == "w10" and args.case == "min":
        raise ValueError("Wind speed (w10) does not have minimum records")

    month_name = MONTH_NAMES[args.month]
    month_abbrev = MONTH_ABBREV[args.month]

    print("=" * 80)
    print(f"2022 {month_name} {args.var.upper()} {args.case.upper()} Records")
    print("=" * 80)
    print(f"Truth dataset: {args.truth}")
    print(f"Record tag: {args.record_tag}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    record, scale = load_records(args.var, args.case, args.record_tag)
    truth_2022 = load_truth_2022(args.var, args.truth, args.month)
    lsm = load_lsm()

    # Select month from records
    month_record = record.sel(month=args.month)
    month_scale = scale.sel(month=args.month) if scale is not None else None

    # Normalize coordinates
    month_record = normalize_coords(month_record)
    if month_scale is not None:
        month_scale = normalize_coords(month_scale)
    lsm = normalize_coords(lsm)
    truth_2022 = normalize_coords(truth_2022)

    # Compute monthly extreme
    print(f"Computing {month_name} 2022 monthly {args.case}...")
    month_2022_extreme = compute_monthly_extreme(truth_2022, args.case)

    # Compute exceedances and depths
    if args.case == "max":
        exceedance = month_2022_extreme - month_record
        exceedance_masked = exceedance.where(exceedance > 0)
    else:  # min
        exceedance = month_record - month_2022_extreme
        exceedance_masked = exceedance.where(exceedance < 0)

    # Compute depth (sigma units) if scale available
    depth = exceedance_masked / month_scale

    # Convert units for display
    if args.var == "t2m":
        # Convert K to °C for temperature
        month_record_display = month_record - 273.15
        units_label = "°C"
        scale_units_label = "°C"
    else:  # w10
        month_record_display = month_record
        units_label = "m s⁻¹"
        scale_units_label = "m s⁻¹"

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Determine colormaps based on variable and case
    if args.var == "t2m" and args.case == "max":
        record_cmap = "RdYlBu_r"
        exceedance_cmap = "YlOrRd"
        depth_cmap = "Reds"
    elif args.var == "t2m" and args.case == "min":
        record_cmap = "RdYlBu_r"
        exceedance_cmap = "Blues"
        depth_cmap = "Blues_r"
    else:  # w10 max
        record_cmap = "YlGnBu"
        exceedance_cmap = "YlOrRd"
        depth_cmap = "Oranges"

    # Panel (a): Historical records
    ax = axes[0, 0]
    month_record_display.plot(
        ax=ax, cmap=record_cmap,
        cbar_kwargs={"label": f"Record {args.case} ({units_label if args.var == 'w10' else '°C'})", "shrink": 0.8}
    )
    ax.contour(lsm.lon, lsm.lat, lsm.T, levels=[0.5], colors='k', linewidths=0.5, alpha=0.4)
    ax.set_title(f"(a) Historical {month_name} {args.case.capitalize()} Records (1979-2021)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Panel (b): 2022 exceedances
    ax = axes[0, 1]
    exceedance_masked.plot(
        x = "lon", y = "lat",
        ax=ax, cmap=exceedance_cmap,
        cbar_kwargs={"label": f"Exceedance ({units_label})", "shrink": 0.8}
    )
    ax.contour(lsm.lon, lsm.lat, lsm.T, levels=[0.5], colors='k', linewidths=0.5)

    ax.set_title(f"(b) {month_name} 2022 Record Exceedances ({args.truth})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Panel (c): Scales
    ax = axes[1, 0]
    if month_scale is not None:
        month_scale.plot(
            ax=ax, cmap="viridis",
            cbar_kwargs={"label": f"Scale σ ({scale_units_label})", "shrink": 0.8}
        )
        ax.contour(lsm.lon, lsm.lat, lsm.T, levels=[0.5], colors='k', linewidths=0.5, alpha=0.4)
        ax.set_title(f"(c) {month_name} Scales (SD of yearly monthly {args.case}ima)", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Scale data not available", ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f"(c) {month_name} Scales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Panel (d): Record depth
    ax = axes[1, 1]
    depth.plot(
        x="lon", y="lat",
        ax=ax, cmap=depth_cmap,
        cbar_kwargs={"label": "Record depth (σ)", "shrink": 0.8}
    )
    ax.contour(lsm.lon, lsm.lat, lsm.T, levels=[0.5], colors='k', linewidths=0.5)
    ax.set_title(f"(d) {month_name} 2022 Record Depth (exceedance/σ)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Main title
    var_name = "T2M" if args.var == "t2m" else "10m Wind Speed"
    fig.suptitle(f"{var_name} {args.case.capitalize()} Records — {month_name} 2022 (Truth: {args.truth})",
                 fontsize=18, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure to organized subdirectory
    case_dir = args.output_dir / "results" / f"{args.var}_{args.case}"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_file = case_dir / f"{month_abbrev}_{args.truth}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_file}")

    plt.close()


if __name__ == "__main__":
    main()
