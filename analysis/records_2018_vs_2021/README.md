# ERA5 Records Comparison: 2019-2021 vs Baseline

This directory analyzes how many ERA5 monthly records (1979-2018 baseline) were broken during 2019-2021.

## Purpose

Quantify record-breaking activity during the 2019-2021 period to contextualize extreme weather forecasting challenges. Shows where and when monthly temperature and wind speed extremes exceeded historical (1979-2018) thresholds.

## Script

### `compare_era5_records.py`

Compares ERA5 monthly records between two time periods and generates summary statistics plus visualizations.

**Usage:**
```bash
python analysis/records_2018_vs_2021/compare_era5_records.py \
    --tag1 1979_2018 \
    --tag2 1979_2021 \
    --records-dir /data/gent/vo/000/gvo00090/HEAT/Data/wb2 \
    --output-dir analysis/records_2018_vs_2021/results
```

**Arguments:**
- `--tag1`: Baseline period tag (default: `1979_2018`)
- `--tag2`: Comparison period tag (default: `1979_2021`)
- `--records-dir`: Directory containing ERA5 record NetCDF files
- `--output-dir`: Output directory for results
- `--dpi`: Figure resolution (default: 150)

**Outputs:**

For each variable (t2m, w10) and record type (max, min):

1. **Multi-panel figure** (`record_comparison_{var}_{extreme}_{tag1}_vs_{tag2}.png`):
   - Spatial map: Number of new records per grid cell (0-12 months)
   - Bar chart: Percentage of cells with new records by month
   - Histogram: Distribution of magnitude changes where new records occurred
   - Spatial map: Mean magnitude of change per grid cell

2. **Summary text file** (`record_comparison_{tag1}_vs_{tag2}_summary.txt`):
   - Total new records by variable and extreme type
   - Monthly breakdown
   - Statistics on magnitude changes

## Data Requirements

ERA5 record files in format:
```
ERA5_record_{max|min}_month_{tag}_{var}.nc
```

Where:
- `{max|min}`: Record type (maximum or minimum)
- `{tag}`: Time period tag (e.g., `1979_2018`, `1979_2021`)
- `{var}`: Variable name (`t2m` or `w10`)

All data at **1.5° resolution** (240×121 equiangular grid, no_Antarctic domain: 90°N to 60°S).

## Key Findings (1979-2018 vs 1979-2021)

### 2m Temperature (t2m)
- **Heat records (max):** 16.15% of grid-months broke previous records
  - Peak: August (18.71% of cells)
  - Mean increase: +0.53°C where records broken

- **Cold records (min):** 2.38% of grid-months broke previous records
  - Peak: March (3.32%)
  - Mean decrease: -0.67°C where records broken

### 10m Wind Speed (w10)
- **Wind records (max):** 8.05% of grid-months broke previous records
  - Peak: May (9.32%)
  - Mean increase: +0.87 m/s where records broken

**Notable:** Heat records occurred ~6.8× more frequently than cold records during 2019-2021, consistent with global warming trends.

## Directory Structure

```
records_2018_vs_2021/
├── README.md                    # This file
├── compare_era5_records.py      # Main analysis script
└── results/                     # Output figures and summary statistics
    ├── record_comparison_1979_2018_vs_1979_2021_summary.txt
    ├── record_comparison_t2m_max_1979_2018_vs_1979_2021.png
    ├── record_comparison_t2m_min_1979_2018_vs_1979_2021.png
    └── record_comparison_w10_max_1979_2018_vs_1979_2021.png
```

## Notes

- Comparison is based on 00 UTC and 12 UTC observations
- Records are monthly climatological extremes (e.g., "warmest July at this location over 1979-2018")
- Analysis focuses on identifying grid cells where 2019-2021 exceeded any monthly record from the baseline period
