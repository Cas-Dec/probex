# 2022 Records Analysis

Visualize and compare 2022 record-breaking extremes using ERA5 or HRES-fc0 as verification.

## Quick Start

```bash
# Plot monthly records (4-panel visualization)
python analysis/2022_records/plot_month_2022_records.py --month 7 --var t2m --truth ERA5 --case max

# Advanced monthly analysis with thresholds and land-ocean partitioning
python analysis/2022_records/plot_monthly_records.py --variable t2m_max --month 12
python analysis/2022_records/plot_monthly_records.py --variable t2m_min --analysis annual-sum

# Compare ERA5 vs HRES-fc0 verification
python analysis/2022_records/compare_era5_hresfc0_verification.py --var t2m
```

## Scripts

### `plot_month_2022_records.py`

Generate 4-panel figures for any month, variable, and truth dataset:
- Historical records (1979-2021)
- 2022 exceedances
- Interannual variability (scales)
- Record depth (sigma units)

**Arguments:**
- `--month`: 1-12
- `--var`: `t2m` or `w10`
- `--truth`: `ERA5` or `HRES_fc0`
- `--case`: `max` or `min` (w10 only supports max)

### `plot_monthly_records.py`

Advanced visualization script for 2022 records with multiple analysis modes:
- **Monthly 4-panel figures**: Same as plot_month_2022_records.py but with enhanced depth thresholding
- **Threshold-based depth maps**: Spatial patterns of record depths exceeding specific σ thresholds
- **Annual summaries**: Yearly aggregation with latitudinal profiles
- **Land-ocean partitioning**: Record events split by surface type

**Arguments:**
- `--variable`: `t2m_max`, `t2m_min`, or `w10_max`
- `--month`: 1-12 (for monthly analysis)
- `--analysis`: `monthly` (default), `thresholds`, or `annual-sum`

**Examples:**
```bash
# Monthly 4-panel for December heat records
python analysis/2022_records/plot_monthly_records.py --variable t2m_max --month 12

# Threshold analysis showing depths > 1σ, 2σ, 3σ
python analysis/2022_records/plot_monthly_records.py --variable t2m_max --analysis thresholds --month 1

# Annual summary with latitudinal profile
python analysis/2022_records/plot_monthly_records.py --variable w10_max --analysis annual-sum
```

### `compare_era5_hresfc0_verification.py`

Compare ERA5 vs HRES-fc0 as verification datasets:
- Record occurrence agreement (confusion matrix)
- Record intensity correlation (depth scatter)
- Spatial bias/RMSE/correlation
- Variability comparison

**Key finding:** HRES-fc0 flags ~60% more record-breaking events than ERA5.

## Output Structure

```
results/
├── t2m_max/                      # Temperature maximum records
│   ├── Jan_ERA5.png              # January heat extremes (ERA5 truth)
│   ├── Jan_HRES_fc0.png          # January heat extremes (HRES-fc0 truth)
│   ├── Jul_ERA5.png              # July heat extremes (ERA5 truth)
│   ├── Jul_HRES_fc0.png          # July heat extremes (HRES-fc0 truth)
│   └── ...                       # Other months
│
├── t2m_min/                      # Temperature minimum records (cold)
│   ├── Dec_ERA5.png
│   ├── Jan_ERA5.png
│   └── ...
│
├── w10_max/                      # Wind speed maximum records
│   ├── Jan_ERA5.png
│   ├── Jan_HRES_fc0.png
│   └── ...
│
└── hres-era5-agreement/          # ERA5 vs HRES-fc0 comparison
    ├── confusion_matrix_t2m_max.png
    ├── depth_scatter_t2m_max.png
    ├── spatial_comparison_t2m.png
    ├── summary_t2m.txt
    └── ...
```

**Naming convention:**
- Monthly records: `{Month}_{Truth}.png`
- Verification comparison: `{metric}_{var}_{case?}.png`

## Record Depth

**Record depth** = (observation - historical_record) / scale

Where **scale** is the sample SD (ddof=1) of yearly monthly extremes:
1. Compute monthly max/min for each year-month
2. Group by calendar month
3. Compute SD across years

A depth of 2σ means the event exceeded the record by 2 standard deviations of interannual variability.

## Notes

- Data: 1.5° resolution (240×100 grid), -58.5°S to 90°N
- Historical baseline: 1979-2021 ERA5 (00/12 UTC)
- Evaluation year: 2022
