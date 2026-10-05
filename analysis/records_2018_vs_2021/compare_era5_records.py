"""
Compare ERA5 monthly record climatologies across different time periods.

This script compares ERA5 records (e.g., 1979-2018 vs 1979-2022) to identify
which grid cells experienced new monthly extremes in the later period.

Usage:
    python analysis/compare_era5_records.py --tag1 1979_2018 --tag2 1979_2022 \
                                             --output-dir results/record_comparison

Outputs:
    - Summary statistics (printed to console and saved as text)
    - Spatial maps showing where new records occurred
    - Bar charts showing monthly distribution of new records
    - Histograms of record magnitude changes
"""

import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Parse arguments
parser = argparse.ArgumentParser(description="Compare ERA5 record climatologies")
parser.add_argument("--tag1", type=str, default="1979_2018",
                    help="First record period tag (baseline)")
parser.add_argument("--tag2", type=str, default="1979_2022",
                    help="Second record period tag (comparison)")
parser.add_argument("--records-dir", type=str,
                    default="/data/gent/vo/000/gvo00090/HEAT/Data/wb2",
                    help="Directory containing ERA5 record files")
parser.add_argument("--output-dir", type=str,
                    default="results/record_comparison",
                    help="Directory for output figures and text")
parser.add_argument("--dpi", type=int, default=150,
                    help="Figure resolution (DPI)")
args = parser.parse_args()

records_dir = Path(args.records_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"Comparing ERA5 Records: {args.tag1} vs {args.tag2}")
print("=" * 80)
print(f"Records dir: {records_dir}")
print(f"Output dir:  {output_dir}")

# Store results for summary report
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append(f"ERA5 Record Comparison: {args.tag1} vs {args.tag2}")
summary_lines.append("=" * 80)
summary_lines.append("")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Process each variable
for var in ["t2m", "w10"]:
    print(f"\n{'='*80}")
    print(f"VARIABLE: {var.upper()}")
    print(f"{'='*80}")

    summary_lines.append(f"\n{'='*80}")
    summary_lines.append(f"VARIABLE: {var.upper()}")
    summary_lines.append(f"{'='*80}")

    for record_type in ["max"] + (["min"] if var == "t2m" else []):
        print(f"\n--- {record_type.upper()} Records ---")
        summary_lines.append(f"\n--- {record_type.upper()} Records ---")

        # Load data
        file1 = records_dir / f"ERA5_record_{record_type}_month_{args.tag1}_{var}.nc"
        file2 = records_dir / f"ERA5_record_{record_type}_month_{args.tag2}_{var}.nc"

        if not file1.exists():
            print(f"  WARNING: {file1} not found, skipping...")
            continue
        if not file2.exists():
            print(f"  WARNING: {file2} not found, skipping...")
            continue

        ds1 = xr.open_dataset(file1)
        ds2 = xr.open_dataset(file2)

        var_name1 = list(ds1.data_vars)[0]
        var_name2 = list(ds2.data_vars)[0]

        rec1 = ds1[var_name1]
        rec2 = ds2[var_name2]

        # Grid info
        lat_coord = 'lat' if 'lat' in rec1.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in rec1.coords else 'longitude'

        total_cells = rec1.sizes[lat_coord] * rec1.sizes[lon_coord]
        total_grid_months = total_cells * 12

        print(f"\nGrid: {rec1.sizes[lon_coord]} lon × {rec1.sizes[lat_coord]} lat = {total_cells} cells")
        summary_lines.append(f"\nGrid: {rec1.sizes[lon_coord]} lon × {rec1.sizes[lat_coord]} lat = {total_cells} cells")

        # Compute changes
        if record_type == "max":
            new_records = rec2 > rec1
        else:
            new_records = rec2 < rec1

        n_new_per_month = new_records.sum(dim=[lat_coord, lon_coord]).values

        delta = rec2 - rec1
        delta_where_new = delta.where(new_records)

        # Print monthly statistics
        print(f"\nNew {record_type} records by month:")
        summary_lines.append(f"\nNew {record_type} records by month:")
        for month, count in zip(months, n_new_per_month):
            pct = 100 * count / total_cells
            line = f"  {month}: {int(count):5d} cells ({pct:5.2f}%)"
            print(line)
            summary_lines.append(line)

        total_new = new_records.sum().values
        total_pct = 100 * total_new / total_grid_months
        line = f"\nTotal new records: {int(total_new)} / {total_grid_months} ({total_pct:.2f}%)"
        print(line)
        summary_lines.append(line)

        # Magnitude statistics
        mean_change = float(delta_where_new.mean(skipna=True))
        median_change = float(delta_where_new.median(skipna=True))
        max_change = float(delta_where_new.max(skipna=True))
        min_change = float(delta_where_new.min(skipna=True))

        print(f"\nMagnitude of changes (where new records set):")
        summary_lines.append(f"\nMagnitude of changes (where new records set):")
        for label, val in [("Mean", mean_change), ("Median", median_change),
                           ("Max", max_change), ("Min", min_change)]:
            line = f"  {label:8s}: {val:8.4f}"
            print(line)
            summary_lines.append(line)

        # Create figure with multiple panels
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Spatial map: Total new records across all months
        ax1 = fig.add_subplot(gs[0, :2])
        total_new_per_cell = new_records.sum(dim='month')

        # Create custom colormap with white for zero
        cmap = plt.cm.YlOrRd
        cmap.set_under('white')

        # Ensure correct dimension order for pcolormesh: (lat, lon)
        if total_new_per_cell.dims[0] != lat_coord:
            total_new_per_cell = total_new_per_cell.transpose(lat_coord, lon_coord)

        im1 = ax1.pcolormesh(
            rec1[lon_coord], rec1[lat_coord], total_new_per_cell,
            cmap=cmap, vmin=0.5, vmax=12, shading='nearest'
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Number of New {record_type.capitalize()} Records per Grid Cell\n'
                     f'{var.upper()} ({args.tag1} → {args.tag2})')
        ax1.set_aspect('auto')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Number of months with new records (0-12)')

        # 2. Monthly bar chart
        ax2 = fig.add_subplot(gs[0, 2])
        pct_per_month = 100 * n_new_per_month / total_cells
        colors = plt.cm.YlOrRd(pct_per_month / pct_per_month.max())
        ax2.bar(range(12), pct_per_month, color=colors)
        ax2.set_xticks(range(12))
        ax2.set_xticklabels(months, rotation=45, ha='right')
        ax2.set_ylabel('Percentage of grid cells (%)')
        ax2.set_title(f'New {record_type.capitalize()} Records by Month')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Histogram of magnitude changes (where new records occurred)
        ax3 = fig.add_subplot(gs[1, 0])
        delta_flat = delta_where_new.values.flatten()
        delta_flat = delta_flat[~np.isnan(delta_flat)]

        if len(delta_flat) > 0:
            ax3.hist(delta_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.axvline(mean_change, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_change:.2f}')
            ax3.axvline(median_change, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_change:.2f}')
            ax3.set_xlabel('Change in record value')
            ax3.set_ylabel('Number of occurrences')
            ax3.set_title(f'Distribution of {record_type.capitalize()} Record Changes')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

        # 4. Spatial map: Mean magnitude of change (where new records occurred)
        ax4 = fig.add_subplot(gs[1, 1:])
        mean_delta_per_cell = delta_where_new.mean(dim='month')

        # Ensure correct dimension order for pcolormesh: (lat, lon)
        if mean_delta_per_cell.dims[0] != lat_coord:
            mean_delta_per_cell = mean_delta_per_cell.transpose(lat_coord, lon_coord)

        # Use symmetric colormap for min records, sequential for max
        if record_type == "min":
            # For min records, changes are negative - use only the negative half
            vmin_data = float(mean_delta_per_cell.min())
            im4 = ax4.pcolormesh(
                rec1[lon_coord], rec1[lat_coord], mean_delta_per_cell,
                cmap='Blues_r', vmin=vmin_data, vmax=0, shading='nearest'
            )
        else:
            im4 = ax4.pcolormesh(
                rec1[lon_coord], rec1[lat_coord], mean_delta_per_cell,
                cmap='YlOrRd', vmin=0, shading='nearest'
            )

        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title(f'Mean Magnitude of Change (where new {record_type} records set)\n'
                     f'{var.upper()} ({args.tag1} → {args.tag2})')
        ax4.set_aspect('auto')
        cbar4 = plt.colorbar(im4, ax=ax4, label='Mean change in record value')

        plt.suptitle(f'{var.upper()} {record_type.upper()} Records Comparison',
                    fontsize=16, fontweight='bold')

        # Save figure
        fig_name = f"record_comparison_{var}_{record_type}_{args.tag1}_vs_{args.tag2}.png"
        fig_path = output_dir / fig_name
        plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\nSaved figure: {fig_path}")
        plt.close()

# Save text summary
summary_file = output_dir / f"record_comparison_{args.tag1}_vs_{args.tag2}_summary.txt"
with open(summary_file, 'w') as f:
    f.write('\n'.join(summary_lines))

print(f"\n{'='*80}")
print(f"Summary saved to: {summary_file}")
print(f"Figures saved to: {output_dir}")
print(f"{'='*80}")
