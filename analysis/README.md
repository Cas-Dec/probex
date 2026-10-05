# Analysis

This directory contains analysis scripts for the `probex` project, organized by analysis type.

## Directory Structure

```
analysis/
├── 2022_records/             # Visualization and verification comparison for 2022 extremes
│   ├── README.md
│   ├── plot_month_2022_records.py
│   ├── plot_monthly_records.py
│   ├── compare_era5_hresfc0_verification.py
│   └── results/              # Output figures organized by variable/case
│
├── records_2018_vs_2021/     # ERA5 record climatology comparisons
│   ├── README.md
│   ├── compare_era5_records.py
│   └── results/              # Comparison figures and summaries
│
├── gencast/                  # GenCast vs GenCast-mini comparison (not part of main body, gitignored)
│   ├── README.md
│   ├── compare_gencast_mini.py
│   └── results/
│
├── perturbed_truth/          # Verification dataset sensitivity analysis (not part of main body, gitignored)
│   ├── README.md
│   ├── run_all_era5.sh
│   ├── compare_pipelines.py
│   ├── create_comparison_plots.py
│   └── results/
```

## Analysis Categories

### 1. 2022 Records Analysis ([2022_records/](2022_records/))

**Purpose**: Visualize and compare 2022 record-breaking extremes using ERA5 or HRES-fc0 as verification.

**Key capabilities**:
- Monthly record visualization (4-panel: historical records, 2022 exceedances, variability, depth)
- ERA5 vs HRES-fc0 verification comparison (confusion matrix, scatter, spatial bias)

**Key finding**: HRES-fc0 flags ~60% more record-breaking events than ERA5 in 2022.

See [2022_records/README.md](2022_records/README.md) for details.

### 2. ERA5 Records Comparison ([records_2018_vs_2021/](records_2018_vs_2021/))

**Purpose**: Compare ERA5 monthly record climatologies across different time periods (e.g., 1979-2018 vs 1979-2021) to identify new record-breaking activity.

**Key findings** (1979-2018 vs 1979-2021):
- Heat records occurred ~6.8× more frequently than cold records (16.15% vs 2.38% of grid-months)
- Wind records: 8.05% of grid-months
- Peak months: August for heat, March for cold, May for wind

See [records_2018_vs_2021/README.md](records_2018_vs_2021/README.md) for details.

## Quick Start

### ERA5 vs HRES-fc0 comparison
```bash
# Local run
bash bash_scripts/run_compare_verification.sh t2m 1979_2021

# HPC submission
qsub -v VAR=t2m job_scripts/submit_compare_verification.pbs
```

### Record climatology comparison
```bash
python analysis/records_2018_vs_2021/compare_era5_records.py \
    --tag1 1979_2018 \
    --tag2 1979_2021 \
    --records-dir /data/gent/vo/000/gvo00090/HEAT/Data/wb2 \
    --output-dir analysis/records_2018_vs_2021/record_comparison_2019_2021
```