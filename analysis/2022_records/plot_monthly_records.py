"""
Analyze and visualize 2022 records (T2M max/min, W10 max) against ERA5 historical climatology.

This script produces various visualizations of record-breaking events in 2022:
- Monthly 4-panel figures (historical records, exceedances, scales, depths)
- Threshold-based depth maps showing spatial patterns
- Annual summaries with latitudinal profiles
- Land-ocean partitioning of record events

Usage:
    python analysis/2022_records/plot_monthly_records.py --variable t2m_max --month 12
    python analysis/2022_records/plot_monthly_records.py --variable t2m_min --analysis annual-sum
    python analysis/2022_records/plot_monthly_records.py --variable w10_max --analysis thresholds --month 1
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# Month names for titles
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# Variable configuration
VARIABLE_CONFIG = {
    "t2m_max": {
        "era5_var": "2m_temperature",
        "records_file": "ERA5_record_max_month_1979_2021_t2m.nc",
        "title": "T2M Maximum",
        "units": "°C",
        "unit_conversion": lambda x: x - 273.15,  # K to C
        "cmap": "Reds",  # Hot colors for temperature maxima
    },
    "t2m_min": {
        "era5_var": "2m_temperature",
        "records_file": "ERA5_record_min_month_1979_2021_t2m.nc",
        "title": "T2M Minimum",
        "units": "°C",
        "unit_conversion": lambda x: x - 273.15,  # K to C
        "cmap": "Blues",  # Cold colors for temperature minima
    },
    "w10_max": {
        "era5_var": "10m_wind_speed",
        "records_file": "ERA5_record_max_month_1979_2021_w10.nc",
        "title": "10m Wind Speed Maximum",
        "units": "m/s",
        "unit_conversion": lambda x: x,  # Already in m/s
        "cmap": "YlGnBu",  # Wind-appropriate colormap (yellow-green-blue)
    },
}

# Data paths
ERA5_ZARR = "/data/gent/vo/000/gvo00090/HEAT/Data/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
RECORDS_ROOT = Path("/data/gent/vo/000/gvo00090/HEAT/Data/wb2")
LSM_FILE = "/data/gent/vo/000/gvo00090/HEAT/Data/wb2/ERA5_lsm_sfc.nc"

# Base output directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_common_data(variable):
    """Load datasets used across multiple analyses."""
    config = VARIABLE_CONFIG[variable]

    print(f"Loading ERA5 records and scales for {config['title']}...")
    records_file = RECORDS_ROOT / config["records_file"]
    ds_records = xr.open_dataset(records_file)

    print("Loading land-sea mask...")
    ds_lsm = xr.open_dataset(LSM_FILE)
    lsm = ds_lsm["lsm"].isel(time=0)
    lsm = lsm.rename({"latitude": "lat", "longitude": "lon"})

    print("Loading ERA5 2022 data...")
    ds_era5 = xr.open_zarr(ERA5_ZARR, chunks={})

    return ds_records, lsm, ds_era5, config


def normalize_coordinates(da):
    """Normalize coordinates to match record files (-180 to 180, sorted)."""
    da = da.assign_coords(
        longitude=(((da.longitude + 180) % 360) - 180)
    ).sortby("longitude").sortby("latitude")
    return da.rename({"latitude": "lat", "longitude": "lon"})


def compute_monthly_stats(ds_era5, month, ds_records, config):
    """
    Compute 2022 monthly extreme (max or min) and record exceedance statistics.

    Returns:
        Tuple of (month_2022_extreme, exceedance, exceedance_depth, month_record, month_scale)
    """
    month_name = MONTH_NAMES[month]
    era5_var = config["era5_var"]
    is_minimum = "min" in config["records_file"]

    print(f"Loading {month_name} 2022 ERA5 observations...")

    # Select month at 00 and 12 UTC (matching record computation)
    era5_month_2022 = ds_era5[era5_var].sel(
        time=slice(f"2022-{month:02d}", f"2022-{month:02d}")
    )
    era5_month_2022 = era5_month_2022.where(
        era5_month_2022.time.dt.hour.isin([0, 12]), drop=True
    )

    # Normalize coordinates
    era5_month_2022 = normalize_coordinates(era5_month_2022)

    # Compute monthly extreme (max or min)
    if is_minimum:
        print(f"Computing {month_name} 2022 monthly minimum...")
        month_2022_extreme = era5_month_2022.min(dim="time", skipna=True)
    else:
        print(f"Computing {month_name} 2022 monthly maximum...")
        month_2022_extreme = era5_month_2022.max(dim="time", skipna=True)

    # Get historical record and scale for this month
    month_record = ds_records["record"].sel(month=month)
    month_scale = ds_records["scale"].sel(month=month)

    # Compute exceedance and depth
    if is_minimum:
        # For minimum records, exceedance is negative (below record)
        exceedance = month_record - month_2022_extreme
        exceedance_depth = exceedance / month_scale
        # Mask to only show where records were broken (obs < record)
        exceedance = exceedance.where(exceedance > 0)
        exceedance_depth = exceedance_depth.where(exceedance > 0)
    else:
        # For maximum records, exceedance is positive (above record)
        exceedance = month_2022_extreme - month_record
        exceedance_depth = exceedance / month_scale
        # Mask to only show where records were exceeded (obs > record)
        exceedance = exceedance.where(exceedance > 0)
        exceedance_depth = exceedance_depth.where(exceedance > 0)

    return month_2022_extreme, exceedance, exceedance_depth, month_record, month_scale


def plot_monthly_4panel(month, ds_records, lsm, ds_era5, config, variable):
    """
    Create 4-panel figure for a specific month:
    - Top left: Historical records
    - Top right: 2022 exceedances
    - Bottom left: Scales
    - Bottom right: Record depth
    """
    month_name = MONTH_NAMES[month]
    print(f"\nCreating 4-panel figure for {month_name}...")

    month_2022_extreme, exceedance, exceedance_depth, month_record, month_scale = \
        compute_monthly_stats(ds_era5, month, ds_records, config)

    # Convert units for display
    month_record_display = config["unit_conversion"](month_record)
    units = config["units"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))

    # Panel (a): Historical records
    ax1 = axes[0, 0]
    month_record_display.plot(
        ax=ax1, cmap="RdBu_r",
        cbar_kwargs={"label": f"1979-2021 Record ({units})"}
    )
    ax1.set_title(
        f"Historical {month_name} {config['title']} Records (1979-2021)",
        fontsize=20
    )

    # Panel (b): 2022 exceedances
    ax2 = axes[0, 1]
    exceedance.plot(
        ax=ax2, x='lon', y='lat', cmap="YlOrRd",
        cbar_kwargs={"label": f"Exceedance ({units})"}
    )
    ax2.contour(
        lsm.lon, lsm.lat, lsm.values.transpose(),
        levels=[0.5], colors='k', linewidths=0.5
    )
    ax2.set_title(f"{month_name} 2022 Record Exceedances ({units})", fontsize=20)

    # Panel (c): Scales
    ax3 = axes[1, 0]
    month_scale.plot(
        ax=ax3, cmap="viridis",
        cbar_kwargs={"label": f"1σ Scale ({units})"}
    )
    extreme_type = "maxima" if "max" in config['records_file'] else "minima"
    ax3.set_title(
        f"{month_name} Scales (SD of yearly monthly {extreme_type})",
        fontsize=20
    )

    # Panel (d): Record depth
    ax4 = axes[1, 1]
    exceedance_depth.plot(
        ax=ax4, x='lon', y='lat', cmap="plasma",
        cbar_kwargs={"label": "Record Depth (σ)"}
    )
    ax4.contour(
        lsm.lon, lsm.lat, lsm.values.transpose(),
        levels=[0.5], colors='k', linewidths=0.5
    )
    ax4.set_title(f"{month_name} 2022 Record Depth (σ)", fontsize=20)

    # Format all axes
    for ax in axes.flatten():
        ax.set_xlabel("Longitude", fontsize=16)
        ax.set_ylabel("Latitude", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Main title
    fig.suptitle(
        f"{config['title']} Records — {month_name} 2022",
        fontsize=24, fontweight="bold", y=0.97
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_dir = RESULTS_DIR / variable
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"records_4panel_{month_name.lower()}_2022.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_threshold_maps(month, ds_records, lsm, ds_era5, config, variable, thresholds=[0, 1, 2, 3, 4]):
    """
    Create threshold-based depth maps showing spatial patterns at different sigma levels.
    """
    month_name = MONTH_NAMES[month]
    print(f"\nCreating threshold maps for {month_name}...")

    _, _, exceedance_depth, _, _ = compute_monthly_stats(ds_era5, month, ds_records, config)

    # Shared color scale
    vmax = float(np.ceil(exceedance_depth.max(skipna=True)))

    # Area weights (cos(lat))
    weights = np.cos(np.deg2rad(exceedance_depth.lat)).broadcast_like(exceedance_depth)
    total_area = weights.sum()

    fig, axes = plt.subplots(
        len(thresholds), 1,
        figsize=(14, 6 * len(thresholds)),
        layout="constrained"
    )

    if len(thresholds) == 1:
        axes = [axes]

    for ax, thr in zip(axes, thresholds):
        masked = exceedance_depth.where(exceedance_depth > thr)

        im = masked.plot(
            ax=ax, x="lon", y="lat",
            cmap="plasma", vmin=0, vmax=vmax,
            add_colorbar=False,
        )
        ax.contour(
            lsm.lon, lsm.lat, lsm.values.transpose(),
            levels=[0.5], colors="k", linewidths=0.5
        )

        n_points = int(masked.notnull().sum())
        area_frac = float(weights.where(masked.notnull()).sum() / total_area) * 100

        ax.set_title(
            f"Record depth > {thr}σ  ({n_points} grid points, "
            f"{area_frac:.2f}% of global area)",
            fontsize=18
        )
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)

    fig.colorbar(
        im, ax=axes, orientation="vertical", shrink=0.6,
        label="Record Depth (σ)", pad=0.02
    )

    fig.suptitle(
        f"{config['title']} Record Depth by Threshold — {month_name} 2022",
        fontsize=22, fontweight="bold"
    )

    output_dir = RESULTS_DIR / variable
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"records_thresholds_{month_name.lower()}_2022.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_annual_summary(ds_records, lsm, ds_era5, config, variable, aggregation='sum'):
    """
    Create annual summary with map and latitudinal profile.

    Args:
        aggregation: 'sum' or 'max' - how to aggregate across months
    """
    print(f"\nCreating annual {aggregation} summary...")

    era5_var = config["era5_var"]
    is_minimum = "min" in config["records_file"]

    # Load all months of 2022 at 00/12 UTC
    era5_2022 = ds_era5[era5_var].sel(time="2022")
    era5_2022 = era5_2022.where(era5_2022.time.dt.hour.isin([0, 12]), drop=True)
    era5_2022 = normalize_coordinates(era5_2022)

    # Compute monthly extremes (max or min)
    if is_minimum:
        print("Computing 2022 monthly minima...")
        monthly_extreme_2022 = era5_2022.groupby("time.month").min("time", skipna=True).compute()
        # For minima, exceedance is when obs < record
        exceed_all = ds_records["record"] - monthly_extreme_2022
    else:
        print("Computing 2022 monthly maxima...")
        monthly_extreme_2022 = era5_2022.groupby("time.month").max("time", skipna=True).compute()
        # For maxima, exceedance is when obs > record
        exceed_all = monthly_extreme_2022 - ds_records["record"]

    depth_all = (exceed_all / ds_records["scale"]).where(exceed_all > 0)

    # Aggregate across months
    if aggregation == 'sum':
        depth_map = depth_all.sum("month", min_count=1)
        lat_profile = depth_all.sum(["month", "lon"])
        title_agg = "Sum"
        cbar_label = "Summed record depth (σ)"
    else:  # max
        depth_map = depth_all.max("month", skipna=True)
        lat_profile = depth_all.max(["month", "lon"])
        title_agg = "Maximum"
        cbar_label = "Maximum record depth (σ)"

    # Create figure
    fig, (ax_map, ax_prof) = plt.subplots(
        1, 2, figsize=(20, 9),
        gridspec_kw={"width_ratios": [4, 1]},
        layout="constrained",
    )

    # Map
    depth_map.plot(
        ax=ax_map, x="lon", y="lat", cmap=config["cmap"],
        cbar_kwargs={"label": cbar_label, "pad": 0.01},
    )
    ax_map.contour(
        lsm.lon, lsm.lat, lsm.values.transpose(),
        levels=[0.5], colors="k", linewidths=1.5
    )
    ax_map.set_title(
        f"{title_agg} of record depths across all months",
        fontsize=22
    )
    ax_map.set_xlabel("Longitude", fontsize=14)
    ax_map.set_ylabel("Latitude", fontsize=14)

    # Latitudinal profile
    ax_prof.fill_betweenx(
        lat_profile.lat, 0, lat_profile,
        color="tab:purple", alpha=0.4
    )
    ax_prof.plot(lat_profile, lat_profile.lat, color="tab:purple", lw=1.5)
    ax_prof.set_title(
        f"{title_agg} over months\n& longitude",
        fontsize=22
    )
    ax_prof.set_xlabel(cbar_label, fontsize=14)
    ax_prof.set_ylabel("")
    ax_prof.set_xlim(left=0)
    ax_prof.grid(alpha=0.3)

    for ax in (ax_map, ax_prof):
        ax.set_ylim(-60, 90)
        ax.tick_params(axis="both", which="major", labelsize=20)

    fig.suptitle(
        f"{config['title']} Record Depth — 2022",
        fontsize=30, fontweight="bold"
    )

    output_dir = RESULTS_DIR / variable
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"records_annual_{aggregation}_2022.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Compute land-ocean statistics
    print(f"\nLand-ocean statistics ({aggregation}):")
    compute_land_ocean_stats(depth_map, lsm)

    plt.close()


def plot_annual_month_of_max(ds_records, lsm, ds_era5, config, variable):
    """
    Create a map showing the month when the maximum record depth occurred.

    The map uses a categorical colormap where each color represents a different month.
    Only grid cells with positive record exceedance are shown.
    """
    print(f"\nCreating month-of-max visualization...")

    era5_var = config["era5_var"]
    is_minimum = "min" in config["records_file"]

    # Load all months of 2022 at 00/12 UTC
    era5_2022 = ds_era5[era5_var].sel(time="2022")
    era5_2022 = era5_2022.where(era5_2022.time.dt.hour.isin([0, 12]), drop=True)
    era5_2022 = normalize_coordinates(era5_2022)

    # Compute monthly extremes (max or min)
    if is_minimum:
        print("Computing 2022 monthly minima...")
        monthly_extreme_2022 = era5_2022.groupby("time.month").min("time", skipna=True).compute()
        # For minima, exceedance is when obs < record
        exceed_all = ds_records["record"] - monthly_extreme_2022
    else:
        print("Computing 2022 monthly maxima...")
        monthly_extreme_2022 = era5_2022.groupby("time.month").max("time", skipna=True).compute()
        # For maxima, exceedance is when obs > record
        exceed_all = monthly_extreme_2022 - ds_records["record"]

    depth_all = (exceed_all / ds_records["scale"]).where(exceed_all > 0)

    # Find month with maximum depth for each grid cell
    month_of_max = depth_all.idxmax(dim="month", skipna=True)
    max_depth = depth_all.max(dim="month", skipna=True)

    # Mask to only show where records were exceeded
    month_of_max = month_of_max.where(max_depth > 0)

    # Create figure with two subplots
    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(22, 9),
        gridspec_kw={"width_ratios": [3, 1]},
        layout="constrained",
    )

    # Define colormap for months (12 distinct colors)
    import matplotlib.colors as mcolors
    # Use Set3 colormap which has 12 distinct colors
    month_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    cmap_months = mcolors.ListedColormap(month_colors)
    bounds = np.arange(0.5, 13.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_months.N)

    # Map
    im = month_of_max.plot(
        ax=ax_map, x="lon", y="lat",
        cmap=cmap_months, norm=norm,
        add_colorbar=False,
    )
    ax_map.contour(
        lsm.lon, lsm.lat, lsm.values.transpose(),
        levels=[0.5], colors="k", linewidths=1.0
    )

    extreme_type = "Maximum" if "max" in config['records_file'] else "Minimum"
    ax_map.set_title(
        f"Month of {extreme_type} Record Depth in 2022",
        fontsize=22
    )
    ax_map.set_xlabel("Longitude", fontsize=14)
    ax_map.set_ylabel("Latitude", fontsize=14)
    ax_map.set_ylim(-60, 90)
    ax_map.tick_params(axis="both", which="major", labelsize=12)

    # Add colorbar with month names
    cbar = fig.colorbar(
        im, ax=ax_map, orientation="horizontal",
        fraction=0.046, pad=0.08, aspect=40
    )
    cbar.set_ticks(range(1, 13))
    cbar.set_ticklabels([MONTH_NAMES[i][:3] for i in range(1, 13)])
    cbar.set_label("Month", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    # Histogram showing frequency of each month
    month_counts = []
    for m in range(1, 13):
        count = int((month_of_max == m).sum())
        month_counts.append(count)

    bars = ax_hist.barh(
        range(1, 13), month_counts,
        color=month_colors, edgecolor='black', linewidth=0.8
    )
    ax_hist.set_yticks(range(1, 13))
    ax_hist.set_yticklabels([MONTH_NAMES[i][:3] for i in range(1, 13)])
    ax_hist.set_xlabel("Number of Grid Cells", fontsize=14)
    ax_hist.set_ylabel("Month", fontsize=14)
    ax_hist.set_title("Record Frequency by Month", fontsize=18)
    ax_hist.tick_params(axis="both", which="major", labelsize=12)
    ax_hist.grid(axis='x', alpha=0.3)
    ax_hist.invert_yaxis()

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, month_counts)):
        if count > 0:
            ax_hist.text(
                count + max(month_counts) * 0.01, i + 1,
                str(count), va='center', ha='left', fontsize=10
            )

    fig.suptitle(
        f"{config['title']} — Month of Maximum Record Depth (2022)",
        fontsize=24, fontweight="bold"
    )

    output_dir = RESULTS_DIR / variable
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"records_month_of_max_2022.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def compute_land_ocean_stats(depth_map, lsm):
    """Compute and print land-ocean partitioning statistics."""
    # Align land-sea mask with depth map
    lsm_aligned = lsm.assign_coords(
        lon=(((lsm.lon + 180) % 360) - 180)
    ).sortby("lon").sortby("lat")

    depth_map, lsm_aligned = xr.align(
        depth_map.fillna(0), lsm_aligned, join="exact"
    )

    # Area weights
    area_w = np.cos(np.deg2rad(depth_map.lat)).broadcast_like(depth_map)

    def land_ocean_share(field, mask):
        land = float((field * mask).sum())
        ocean = float((field * (1 - mask)).sum())
        total = land + ocean
        if total == 0:
            return 0.0, 0.0
        return land / total, ocean / total

    land_u, ocean_u = land_ocean_share(depth_map, lsm_aligned)
    land_w, ocean_w = land_ocean_share(depth_map * area_w, lsm_aligned)
    land_area_frac = float((lsm_aligned * area_w).sum() / area_w.sum())

    print(f"  Unweighted:    land {land_u:.1%} | ocean {ocean_u:.1%}")
    print(f"  Area-weighted: land {land_w:.1%} | ocean {ocean_w:.1%}")
    print(f"  Baseline land area fraction: {land_area_frac:.1%}")
    if land_area_frac > 0:
        print(f"  Land over-representation (weighted): {land_w / land_area_frac:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize 2022 records (T2M max/min, W10 max)"
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="t2m_max",
        choices=["t2m_max", "t2m_min", "w10_max"],
        help="Variable to analyze"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="monthly",
        choices=["monthly", "thresholds", "annual-sum", "annual-max", "month-of-max", "all"],
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Month to analyze (1-12, required for monthly/thresholds)"
    )

    args = parser.parse_args()

    # Create output directory for this variable
    output_dir = RESULTS_DIR / args.variable
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load common datasets
    ds_records, lsm, ds_era5, config = load_common_data(args.variable)

    # Run requested analysis
    if args.analysis == "monthly":
        if args.month is None:
            parser.error("--month required for monthly analysis")
        plot_monthly_4panel(args.month, ds_records, lsm, ds_era5, config, args.variable)

    elif args.analysis == "thresholds":
        if args.month is None:
            parser.error("--month required for threshold analysis")
        plot_threshold_maps(args.month, ds_records, lsm, ds_era5, config, args.variable)

    elif args.analysis == "annual-sum":
        plot_annual_summary(ds_records, lsm, ds_era5, config, args.variable, aggregation='sum')

    elif args.analysis == "annual-max":
        plot_annual_summary(ds_records, lsm, ds_era5, config, args.variable, aggregation='max')

    elif args.analysis == "month-of-max":
        plot_annual_month_of_max(ds_records, lsm, ds_era5, config, args.variable)

    elif args.analysis == "all":
        # Run all analyses
        print("\n" + "="*80)
        print("Running all analyses for all months...")
        print("="*80)

        for month in range(1, 13):
            plot_monthly_4panel(month, ds_records, lsm, ds_era5, config, args.variable)
            plot_threshold_maps(month, ds_records, lsm, ds_era5, config, args.variable)

        plot_annual_summary(ds_records, lsm, ds_era5, config, args.variable, aggregation='sum')
        plot_annual_summary(ds_records, lsm, ds_era5, config, args.variable, aggregation='max')
        plot_annual_month_of_max(ds_records, lsm, ds_era5, config, args.variable)

        print("\n" + "="*80)
        print("All analyses complete!")
        print("="*80)

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
