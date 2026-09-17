"""
Compare forecast evaluation results using 2018 vs 2022 records.

This script compares how model skill metrics change when evaluated against
different ERA5 record climatologies (1979-2018 vs 1979-2022).

Usage:
    python analysis/compare_eval_results.py \
        --old-results old_results \
        --new-results results \
        --output-dir results/eval_comparison \
        --models IFS_ENS FGN

Outputs:
    - Comparison figures showing metric changes
    - Summary statistics tables
    - Analysis of which metrics are most affected by the record threshold change
"""

import argparse
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

parser = argparse.ArgumentParser(description="Compare evaluation results with 2018 vs 2022 records")
parser.add_argument("--old-results", type=str, default="old_results",
                    help="Directory with 2018 records results")
parser.add_argument("--new-results", type=str, default="results",
                    help="Directory with 2022 records results")
parser.add_argument("--output-dir", type=str, default="results/eval_comparison",
                    help="Output directory for comparison figures")
parser.add_argument("--models", nargs="+", default=["IFS_ENS", "FGN"],
                    help="Models to compare")
parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
args = parser.parse_args()

old_dir = Path(args.old_results)
new_dir = Path(args.new_results)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Comparing Evaluation Results: 2018 vs 2022 Records")
print("=" * 80)
print(f"Old results (2018 records): {old_dir}")
print(f"New results (2022 records): {new_dir}")
print(f"Output directory: {output_dir}")
print(f"Models: {args.models}")
print()

summary_data = []

for model in args.models:
    print(f"\n{'='*80}")
    print(f"MODEL: {model}")
    print(f"{'='*80}")

    # Determine truth dataset
    if model == "IFS_ENS":
        truth = "HRES_fc0"
    else:
        truth = "ERA5"

    for var in ["t2m", "w10"]:
        print(f"\n--- Variable: {var.upper()} ---")

        # Compare CRPS results
        for case in ["all", "max"] + (["min"] if var == "t2m" else []):

            # Load old and new CRPS files
            old_crps_file = old_dir / model / "CRPS" / f"{model}_{case}_CRPS_2022_{var}_no_Antarctic_records_1979_2018_cond_{truth}.nc"
            new_crps_file = new_dir / model / "CRPS" / f"{model}_{case}_CRPS_2022_{var}_no_Antarctic_records_1979_2022_cond_{truth}.nc"

            if not old_crps_file.exists() or not new_crps_file.exists():
                print(f"  Skipping {case} CRPS (files not found)")
                continue

            ds_old = xr.open_dataset(old_crps_file)
            ds_new = xr.open_dataset(new_crps_file)

            # Compare CRPS values
            crps_old = ds_old['CRPS']
            crps_new = ds_new['CRPS']

            # Compute mean across seasons and land/sea
            crps_old_mean = crps_old.sel(season='all', landsea='all').values
            crps_new_mean = crps_new.sel(season='all', landsea='all').values

            # Get leadtimes
            leadtimes = crps_old.leadtime.values

            print(f"\n  Case: {case}")
            print(f"    Old CRPS mean: {np.nanmean(crps_old_mean):.4f}")
            print(f"    New CRPS mean: {np.nanmean(crps_new_mean):.4f}")
            print(f"    Change: {np.nanmean(crps_new_mean - crps_old_mean):.4f}")

            # Store for summary
            summary_data.append({
                'Model': model,
                'Variable': var,
                'Case': case,
                'Metric': 'CRPS',
                'Old_Mean': np.nanmean(crps_old_mean),
                'New_Mean': np.nanmean(crps_new_mean),
                'Change': np.nanmean(crps_new_mean - crps_old_mean),
                'Pct_Change': 100 * (np.nanmean(crps_new_mean) / np.nanmean(crps_old_mean) - 1)
            })

            # Create comparison figure for this case
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{model} {var.upper()} - {case.upper()} Case: 2018 vs 2022 Records',
                        fontsize=14, fontweight='bold')

            # 1. CRPS comparison by lead time
            ax = axes[0, 0]
            ax.plot(leadtimes, crps_old_mean, 'o-', label='2018 records', linewidth=2, markersize=6)
            ax.plot(leadtimes, crps_new_mean, 's-', label='2022 records', linewidth=2, markersize=6)
            ax.set_xlabel('Lead time (hours)')
            ax.set_ylabel('CRPS')
            ax.set_title('CRPS by Lead Time')
            ax.legend()
            ax.grid(alpha=0.3)

            # 2. CRPS difference
            ax = axes[0, 1]
            diff = crps_new_mean - crps_old_mean
            ax.plot(leadtimes, diff, 'o-', color='darkred', linewidth=2, markersize=6)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Lead time (hours)')
            ax.set_ylabel('CRPS Change (2022 - 2018)')
            ax.set_title('CRPS Difference')
            ax.grid(alpha=0.3)

            # 3. CRPS by season (averaged over lead times)
            ax = axes[1, 0]
            seasons = ['DJF', 'MAM', 'JJA', 'SON']
            crps_old_seasonal = [crps_old.sel(season=s, landsea='all').mean().values for s in seasons]
            crps_new_seasonal = [crps_new.sel(season=s, landsea='all').mean().values for s in seasons]

            x = np.arange(len(seasons))
            width = 0.35
            ax.bar(x - width/2, crps_old_seasonal, width, label='2018 records', alpha=0.8)
            ax.bar(x + width/2, crps_new_seasonal, width, label='2022 records', alpha=0.8)
            ax.set_xlabel('Season')
            ax.set_ylabel('Mean CRPS')
            ax.set_title('CRPS by Season')
            ax.set_xticks(x)
            ax.set_xticklabels(seasons)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            # 4. CRPS by land/sea (averaged over lead times)
            ax = axes[1, 1]
            landsea_types = ['land', 'sea']
            crps_old_landsea = [crps_old.sel(season='all', landsea=ls).mean().values for ls in landsea_types]
            crps_new_landsea = [crps_new.sel(season='all', landsea=ls).mean().values for ls in landsea_types]

            x = np.arange(len(landsea_types))
            ax.bar(x - width/2, crps_old_landsea, width, label='2018 records', alpha=0.8)
            ax.bar(x + width/2, crps_new_landsea, width, label='2022 records', alpha=0.8)
            ax.set_xlabel('Surface Type')
            ax.set_ylabel('Mean CRPS')
            ax.set_title('CRPS by Land/Sea')
            ax.set_xticks(x)
            ax.set_xticklabels(['Land', 'Sea'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            # Save figure
            fig_name = f"eval_comparison_{model}_{var}_{case}_2018vs2022.png"
            fig_path = output_dir / fig_name
            plt.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
            print(f"    Saved: {fig_path.name}")
            plt.close()

# Create summary table
print(f"\n{'='*80}")
print("Creating summary table...")
print(f"{'='*80}")

df = pd.DataFrame(summary_data)

# Save as CSV
summary_csv = output_dir / "eval_comparison_summary.csv"
df.to_csv(summary_csv, index=False)
print(f"\nSaved summary table: {summary_csv}")

# Print formatted table
print("\n" + "="*80)
print("SUMMARY: CRPS Changes (2018 → 2022 Records)")
print("="*80)
for model in args.models:
    print(f"\n{model}:")
    model_df = df[df['Model'] == model]
    for var in ['t2m', 'w10']:
        var_df = model_df[model_df['Variable'] == var]
        if len(var_df) > 0:
            print(f"\n  {var.upper()}:")
            for _, row in var_df.iterrows():
                print(f"    {row['Case']:4s}: {row['Old_Mean']:.4f} → {row['New_Mean']:.4f} "
                      f"(Δ={row['Change']:+.4f}, {row['Pct_Change']:+.2f}%)")

print(f"\n{'='*80}")
print("Comparison complete!")
print(f"{'='*80}")
print(f"\nOutputs saved to: {output_dir}")
print(f"  - Individual comparison figures for each model/variable/case")
print(f"  - Summary CSV: {summary_csv.name}")
