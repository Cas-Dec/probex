# Project Status: ERA5 Records and Forecast Evaluation

**Last Updated:** 2026-09-17

## Summary

This project evaluates probabilistic ensemble weather forecasts (IFS_ENS, FGN, ArchesWeatherGen, uCast, GenCast) against verification data (ERA5, HRES-fc0) with a focus on record-breaking extremes for 2m temperature (t2m) and 10m wind speed (w10).

**Key Achievements:**
- ✅ Created ERA5 records (1979-2021) excluding evaluation year
- ✅ Fixed circular evaluation error (records now exclude 2022)
- ✅ Implemented adaptive forecast loader for different data formats
- ✅ Identified critical HRES-fc0 data quality issues (11.2% missing data)
- ✅ Discovered temperature-specific warm tail bias in HRES-fc0 (3.2× heat record inflation)
- ✅ GenCast forecasts integrated into evaluation pipeline

## What Was Completed

### 1. ERA5 Records Extension ✅

**Created:** ERA5 records for 1979-2021 period (correctly excluding evaluation year 2022)

**Location:**
- `/data/gent/vo/000/gvo00090/HEAT/Data/wb2/ERA5_record_*_1979_2021_*.nc`
- `data/era5_records/ERA5_record_*_1979_2021_*.nc` (local copies)

**Files:**
- `ERA5_record_max_month_1979_2021_t2m.nc`
- `ERA5_record_min_month_1979_2021_t2m.nc`
- `ERA5_record_max_month_1979_2021_w10.nc`

**Grid:** 240 lon × 121 lat (1.5° resolution, global)

### 2. Record Comparison Analysis ✅

**Script:** `analysis/compare_era5_records.py`

**Key Findings (2019-2021 vs 1979-2018):**
- **Heat records (t2m max):** 14.63% of grid-months experienced new records
- **Cold records (t2m min):** 2.67% of grid-months experienced new records
- **Wind records (w10 max):** 7.88% of grid-months experienced new records

**Output:** Figures and summary in `results/record_comparison/`

### 3. Evaluation Infrastructure ✅

**Job Scripts Created/Updated:**
- `job_scripts/submit_eval_crps_all_2022records.pbs`
- `job_scripts/submit_eval_point_all_2022records.pbs`
- `job_scripts/submit_eval_occurrence_all_2022records.pbs`
- `job_scripts/submit_eval_margin_all_2022records.pbs`
- `job_scripts/submit_all_eval_2022records.sh` (master script)

**Configuration:**
- Evaluates: **2022 forecasts**
- Against: **1979-2021 records** (correct - excludes evaluation year)
- Models: IFS_ENS, ArchesWeatherGen, uCast, FGN, GenCast
- Variables: t2m, w10
- Verification: ERA5 (primary), HRES-fc0 (IFS_ENS only, for comparison)

### 4. Analysis Tools ✅

**Scripts:**
- `analysis/compare_era5_records.py` - Compare record periods
- `analysis/compare_eval_results.py` - Compare evaluation metrics
- `analysis/compare_era5_hresfc0_verification.py` - **NEW:** Compare ERA5 vs HRES-fc0 as verification datasets
- `download/create_era5_records.py` - Generate ERA5 records
- `eval/forecast_loader.py` - **NEW:** Adaptive loader for different forecast formats

**Documentation:**
- `SUMMARY_2022_RECORDS_UPDATE.md` - Comprehensive project summary (outdated - referred to 2022)
- `QUICK_START_2022_EVAL.md` - Quick reference (outdated)
- `job_scripts/README_2022_EVALUATION.md` - Evaluation guide (outdated)
- `analysis/README.md` - Analysis documentation
- `analysis/era5_vs_hresfc0/FINDINGS.md` - **NEW:** ERA5 vs HRES-fc0 comparison results
- `analysis/era5_vs_hresfc0/HRES-FC0_DATA_GAPS.md` - **NEW:** HRES-fc0 data quality analysis

## What Remains To Do

### 1. ~~ArchesWeatherGen and uCast Evaluation~~ ✅ **RESOLVED**

**Status:** Implemented adaptive forecast loader (`eval/forecast_loader.py`) that automatically detects and handles:
1. Per-lead files (IFS_ENS, FGN)
2. Combined Zarr with lead selection (ArchesWeatherGen, uCast, GenCast)

All evaluation scripts now use this loader and work with all model formats.

### 2. Complete Full Model Evaluation

**Status:** Evaluation infrastructure ready for all models (IFS_ENS, FGN, ArchesWeatherGen, uCast, GenCast)

**To run:**
```bash
bash job_scripts/submit_all_eval_2022records.sh
```

**Note:** Some evaluations may be in progress or completed. Check `results/` directory and `logs/` for status.

### 3. Address HRES-fc0 Data Quality Issues

**Discovered Issues:**
- 11.2% of 2022 data missing (82/730 timesteps)
- Missing periods: Jan 1-24, May 1-12, Dec 27-31
- Winter (DJF) only 67.8% complete

**Recommendations:**
1. Use ERA5 as primary verification dataset (100% coverage)
2. Use HRES-fc0 only for sensitivity analysis
3. Document data gaps in any HRES-fc0-verified results

See `analysis/era5_vs_hresfc0/HRES-FC0_DATA_GAPS.md` for details.

### 4. Investigate Temperature-Specific Verification Bias

**Discovered:** HRES-fc0 flags 3.2× more heat records than ERA5 in 2022, but wind records agree well (F1=0.56)

**Next Steps:**
1. Re-evaluate IFS t2m results with ERA5 as truth (currently use HRES-fc0)
2. Investigate physical cause of warm tail bias
3. Check ECMWF documentation for known HRES-fc0 issues
4. Consider sensitivity tests with different record baselines (1991-2020)

See `analysis/era5_vs_hresfc0/FINDINGS.md` for full analysis.

## Key Insights Discovered

### 1. Resolution Mismatch (Resolved)

Initially discovered that old results in `data/era5_records/` used 0.25° resolution, but actual evaluation uses 1.5° records from HEAT data location. Confirmed evaluation is using correct 1.5° data.

### 2. Circular Evaluation Error (Fixed)

Originally created 1979-2022 records and attempted to evaluate 2022 forecasts against them. This is fundamentally wrong - you cannot include the evaluation year in the climatology. Fixed by using 1979-2021 records.

### 3. Data Structure Differences (Resolved)

Models use different data storage:
- **IFS_ENS, FGN:** Per-lead-time Zarr files
- **ArchesWeatherGen, uCast, GenCast:** Combined Zarr files

Resolved by implementing adaptive `forecast_loader.py` that handles both formats automatically.

### 4. HRES-fc0 Data Quality Issues (NEW - Critical Finding)

**Data Gaps:** 11.2% of 2022 missing, concentrated in winter (32% of DJF)
- Missing: Jan 1-24, May 1-12, Dec 27-31
- Impact: Seasonal bias toward warm months

**Temperature-Specific Warm Tail Bias:**
- HRES-fc0 flags **3.2× more heat records** than ERA5 (90,849 vs 28,367 events)
- Poor precision (20%) and F1 score (0.31) for heat records
- Wind records show much better agreement (F1=0.56)

**Implication:** Using HRES-fc0 as verification for temperature extremes gives artificially inflated observed record counts, making forecast skill appear worse. **Recommendation: Use ERA5 as primary verification.**

See `analysis/era5_vs_hresfc0/` for detailed analysis.

### 5. Verification Dataset Choice Matters (for Temperature)

The choice between ERA5 and HRES-fc0 as "truth" has major implications:
- **For t2m extremes:** 3.2× difference in expected record counts → incomparable skill scores
- **For w10 extremes:** Minimal difference → verification choice less critical
- **For overall distribution:** Both excellent (r>0.99) → verification choice doesn't matter

**Best Practice:** Use ERA5 as primary verification, report HRES-fc0 results separately if needed for comparison.

## Files and Locations

### Data
```
/data/gent/vo/000/gvo00090/HEAT/Data/wb2/
├── ERA5_record_max_month_1979_2018_t2m.nc    (existing, 1.5°)
├── ERA5_record_min_month_1979_2018_t2m.nc    (existing, 1.5°)
├── ERA5_record_max_month_1979_2018_w10.nc    (existing, 1.5°)
├── ERA5_record_max_month_1979_2021_t2m.nc    (NEW, 1.5°)
├── ERA5_record_min_month_1979_2021_t2m.nc    (NEW, 1.5°)
└── ERA5_record_max_month_1979_2021_w10.nc    (NEW, 1.5°)

data/era5_records/  (local copies of 2021 records)

results/record_comparison/  (2018 vs 2021 comparison)

old_results/  (evaluation results using 2018 records)

analysis/era5_vs_hresfc0/  (NEW - verification dataset comparison)
├── FINDINGS.md
├── HRES-FC0_DATA_GAPS.md
├── summary_t2m.txt
├── summary_w10.txt
└── *.png (comparison plots)
```

### Scripts Ready to Use
- Record generation: `job_scripts/submit_create_era5_records.pbs`
- Record comparison: `analysis/compare_era5_records.py`
- **Verification comparison:** `analysis/compare_era5_hresfc0_verification.py` (NEW)
- **Forecast evaluation (all models):** `job_scripts/submit_eval_*_all_2022records.pbs`
- Result comparison: `analysis/compare_eval_results.py`
- **Adaptive forecast loader:** `eval/forecast_loader.py` (NEW - handles all model formats)

## Usage

### Generate Records for New Period
```bash
qsub -v RECORD_TAG=1979_YYYY,START_YEAR=1979,END_YEAR=YYYY \
     job_scripts/submit_create_era5_records.pbs
```

### Compare Record Periods
```bash
conda activate probex
python analysis/compare_era5_records.py \
    --tag1 1979_2018 \
    --tag2 1979_2021 \
    --output-dir results/record_comparison
```

### Compare Verification Datasets (ERA5 vs HRES-fc0)
```bash
# Submit as PBS job
qsub -v VAR=t2m job_scripts/submit_compare_verification.pbs
qsub -v VAR=w10 job_scripts/submit_compare_verification.pbs

# Or run locally
conda activate probex
python analysis/compare_era5_hresfc0_verification.py --var t2m --record-tag 1979_2021
python analysis/compare_era5_hresfc0_verification.py --var w10 --record-tag 1979_2021
```

### Run Evaluation (All Models)
```bash
# All evaluation types for all models
bash job_scripts/submit_all_eval_2022records.sh

# Single evaluation type
qsub job_scripts/submit_eval_crps_all_2022records.pbs
qsub job_scripts/submit_eval_point_all_2022records.pbs
qsub job_scripts/submit_eval_occurrence_all_2022records.pbs
qsub job_scripts/submit_eval_margin_all_2022records.pbs
```

### Compare Evaluation Results
```bash
conda activate probex
python analysis/compare_eval_results.py \
    --old-results old_results \
    --new-results results \
    --models IFS_ENS FGN ArchesWeatherGen uCast GenCast \
    --output-dir results/eval_comparison
```

## Notes

- **Debug flags:** Evaluation scripts contain hardcoded debug overrides (reduced domain, fewer leads/members). Remove before production runs.
- **Record tag:** Current default is `1979_2018` in evaluation scripts. Override with `RECORD_TAG=1979_2021` in job submissions (already set in 2022records PBS files).
- **Verification dataset:**
  - **t2m:** Use ERA5 as primary truth (HRES-fc0 has 3.2× warm tail bias)
  - **w10:** Either ERA5 or HRES-fc0 acceptable (good agreement)
  - IFS_ENS currently evaluated against HRES-fc0 - consider re-running with ERA5
- **Data completeness:** HRES-fc0 missing 11.2% of 2022 data (see `analysis/era5_vs_hresfc0/HRES-FC0_DATA_GAPS.md`)

## Contact

For questions or to continue this work:
1. Check this status document
2. Review `CLAUDE.md` for project structure
3. Consult `job_scripts/README_2022_EVALUATION.md` for evaluation details (note: may reference outdated 2022 records)
4. Check evaluation script comments for implementation details
