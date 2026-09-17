# ERA5 Records Analysis

This directory contains scripts for analyzing ERA5 monthly record climatologies.

## Scripts

### `compare_era5_records.py`

Compares ERA5 records between two time periods (e.g., 1979-2018 vs 1979-2022) to identify where and when new monthly extremes occurred.

**Usage:**
```bash
python analysis/compare_era5_records.py \
    --tag1 1979_2018 \
    --tag2 1979_2022 \
    --records-dir /data/gent/vo/000/gvo00090/HEAT/Data/wb2 \
    --output-dir results/record_comparison \
    --dpi 150
```

**Arguments:**
- `--tag1`: Baseline period tag (default: `1979_2018`)
- `--tag2`: Comparison period tag (default: `1979_2022`)
- `--records-dir`: Directory containing ERA5 record NetCDF files (default: HEAT data location)
- `--output-dir`: Output directory for figures and summary (default: `results/record_comparison`)
- `--dpi`: Figure resolution (default: 150)

**Outputs:**

For each variable (t2m, w10) and record type (max, min):

1. **Multi-panel figure** showing:
   - Spatial map: Number of new records per grid cell (0-12 months)
   - Bar chart: Percentage of cells with new records by month
   - Histogram: Distribution of magnitude changes where new records occurred
   - Spatial map: Mean magnitude of change per grid cell

2. **Text summary** with statistics for all variables and record types

**Example output:**
```
results/record_comparison/
├── record_comparison_1979_2018_vs_1979_2022_summary.txt
├── record_comparison_t2m_max_1979_2018_vs_1979_2022.png
├── record_comparison_t2m_min_1979_2018_vs_1979_2022.png
└── record_comparison_w10_max_1979_2018_vs_1979_2022.png
```

## Data Requirements

The script expects ERA5 record files in the format:
```
ERA5_record_{max|min}_month_{tag}_{var}.nc
```

Where:
- `{max|min}`: Record type
- `{tag}`: Time period tag (e.g., `1979_2018`, `1979_2022`)
- `{var}`: Variable name (`t2m` for 2m temperature, `w10` for 10m wind speed)

Files should contain monthly climatological extremes on a regular lat/lon grid.

## Key Findings (1979-2018 vs 1979-2022)

### 2m Temperature (t2m)
- **Heat records (max):** 18.23% of grid-months experienced new records (63,543 total)
  - Peak: July & November (20.65% of cells each)
  - Mean increase: +0.62°C where records broken
  - Largest increase: +18.12°C

- **Cold records (min):** 3.88% of grid-months experienced new records (13,519 total)
  - Peak: June (4.62%)
  - Mean decrease: -0.69°C where records broken
  - Largest decrease: -11.03°C

### 10m Wind Speed (w10)
- **Wind records (max):** 10.59% of grid-months experienced new records (36,894 total)
  - Peak: December (12.29%)
  - Mean increase: +0.92 m/s where records broken
  - Largest increase: +12.74 m/s

**Notable:** Heat records occurred ~5× more frequently than cold records during 2019-2022, consistent with global warming trends.
