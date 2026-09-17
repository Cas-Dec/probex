# HRES-fc0 2022 Data Gaps Analysis

**Generated:** 2026-09-17
**Dataset:** WeatherBench2 HRES-fc0 (2022, 1.5° resolution, no_Antarctic domain)
**Variables:** 2m_temperature (t2m), 10m_wind_speed (w10)

---

## Executive Summary

The HRES-fc0 2022 verification dataset from WeatherBench2 has **significant data gaps**:

- **82 out of 730 timesteps** are completely missing (all NaN)
- **11.2% of the year** has no data
- **Both t2m and w10** have identical missing periods
- **Completeness: 88.77%**

---

## Missing Data Periods

### Period 1: Early January (Start of Year)
- **Dates:** January 1-24, 2022
- **Duration:** 24 days (48 timesteps at 12-hourly)
- **Impact:** 65.8% of January missing

### Period 2: Early May (Spring)
- **Dates:** May 1-12, 2022
- **Duration:** 12 days (24 timesteps at 12-hourly)
- **Impact:** 38.7% of May missing

### Period 3: End of December (End of Year)
- **Dates:** December 27-31, 2022
- **Duration:** 5 days (10 timesteps at 12-hourly)
- **Impact:** 16.1% of December missing

---

## Impact on Seasonal Coverage

| Season | Total Days | Missing Days | Available Days | Completeness |
|--------|-----------|--------------|----------------|--------------|
| **DJF** (Dec-Feb) | 90 | 29 | 61 | 67.8% |
| **MAM** (Mar-May) | 92 | 12 | 80 | 87.0% |
| **JJA** (Jun-Aug) | 92 | 0 | 92 | **100%** |
| **SON** (Sep-Nov) | 91 | 0 | 91 | **100%** |

**Key Finding:** Winter (DJF) has only **67.8% data coverage**, with 32% of the season missing.

---

## Impact on Monthly Coverage

| Month | Days | Available | Missing | Completeness |
|-------|------|-----------|---------|--------------|
| Jan | 31 | 10 | 24 | 32.3% |
| Feb | 28 | 28 | 0 | 100% |
| Mar | 31 | 31 | 0 | 100% |
| Apr | 30 | 30 | 0 | 100% |
| May | 31 | 19 | 12 | 61.3% |
| Jun | 30 | 30 | 0 | 100% |
| Jul | 31 | 31 | 0 | 100% |
| Aug | 31 | 31 | 0 | 100% |
| Sep | 30 | 30 | 0 | 100% |
| Oct | 31 | 31 | 0 | 100% |
| Nov | 30 | 30 | 0 | 100% |
| Dec | 31 | 26 | 5 | 83.9% |

**Worst months:** January (32%), May (61%), December (84%)

---

## Implications for Forecast Verification

### 1. Record-Breaking Event Counts
- **Missing 11.2% of potential record events**
- Winter extremes particularly undersampled (32% of DJF missing)
- May heat events undersampled (39% of May missing)

### 2. Seasonal Skill Scores
- **DJF scores based on only 67.8% of season** → biased estimates
- JJA and SON have full coverage → reliable seasonal statistics
- MAM moderately affected (87% coverage)

### 3. Annual Aggregations
- Mean/median statistics: slightly biased toward summer/autumn
- Extreme value statistics: underestimate winter extremes
- Comparison to ERA5: ERA5 has 100% coverage

### 4. Expected vs Observed Record Counts
The "too many heat records" finding when verifying against HRES-fc0 may be partially explained by:
- **Selection bias**: Missing data concentrated in winter (cold extremes)
- **Remaining data skewed toward warm seasons** (summer/autumn complete)
- Expected record probability calculations assume uniform sampling

---

## Comparison: ERA5 vs HRES-fc0 Coverage

| Dataset | 2022 Coverage | Missing Periods | Notes |
|---------|---------------|-----------------|-------|
| **ERA5** | 100% (1,460 timesteps at 6h) | None | Complete 6-hourly data |
| **HRES-fc0** | 88.77% (648/730 timesteps at 12h) | 3 gaps, 82 timesteps | WeatherBench2 archive incomplete |

**Impact on ERA5 vs HRES-fc0 comparison:**
- Common timesteps: Only where HRES-fc0 has data (648 out of 730)
- Comparison excludes 11.2% of year
- Winter comparison particularly affected (32% of DJF excluded)

---

## Root Cause

The data gaps originate from the **WeatherBench2 HRES archive**:
- Source: `gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr`
- These are the raw ECMWF HRES analysis (fc0) fields archived by Google
- Gaps likely due to:
  1. **Operational issues** at ECMWF during these periods
  2. **Archival delays** in ingesting data to WeatherBench2
  3. **Known data quality issues** flagged by ECMWF (data withheld)

---

## Recommendations

### Immediate Actions

1. ✅ **Document the data gaps** in all publications using HRES-fc0 verification
   - Report actual sample size (648 timesteps, not 730)
   - Note seasonal imbalance (only 68% of winter data)

2. ✅ **Filter NaN timesteps** before computing verification scores
   - Current eval scripts handle this correctly via xarray NaN semantics
   - Margin evaluation has explicit `.notnull()` filtering

3. ⚠️ **Adjust expected record probabilities** for seasonal imbalance
   - Current calculation assumes uniform sampling across year
   - Should weight by actual available days per season

### Long-term Solutions

1. **Use alternative verification years** with better coverage:
   ```bash
   # Check 2020 and 2018 data completeness
   python download/download_hres_fc0_wb2.py --year 2020
   python download/download_hres_fc0_wb2.py --year 2018
   ```

2. **Prefer ERA5 as primary verification** for consistency:
   - 100% temporal coverage
   - Longer record for establishing baselines (1959-present)
   - No seasonal gaps

3. **Use HRES-fc0 for sensitivity analysis only**:
   - Document as "incomplete verification dataset"
   - Compare ERA5 vs HRES-fc0 results to quantify verification uncertainty

### For Current Analysis

**ERA5 vs HRES-fc0 comparison script:**
- ✅ Already handles gaps correctly (alignment finds common times)
- ✅ NaN filtering implemented
- ⚠️ Results summary should note 88.77% coverage

**Forecast evaluation results:**
- Document: "Verification based on 648 available timesteps (Jan 25 - Apr 30, May 13 - Dec 26, 2022)"
- Add caveat: "DJF statistics based on 68% seasonal coverage"

---

## Data Quality Assessment

| Quality Metric | Status | Notes |
|----------------|--------|-------|
| **Spatial coverage** | ✅ Complete | No missing grid cells in valid timesteps |
| **Temporal coverage** | ⚠️ 88.77% | 3 gaps totaling 82 timesteps |
| **Seasonal balance** | ❌ Biased | Winter undersampled (32% missing) |
| **Variable consistency** | ✅ Good | t2m and w10 have identical gaps |
| **Forecast lead times** | ✅ Complete | fc0 (analysis) only, no lead time gaps |

---

## Verification Dataset Recommendations

**For Publication:**

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|--------|
| **Primary verification** | ERA5 | 100% coverage, longer record |
| **Sensitivity test** | HRES-fc0 | Compare verification uncertainty |
| **Seasonal statistics** | ERA5 (DJF, MAM), either (JJA, SON) | HRES-fc0 missing 32% of winter |
| **Annual aggregates** | ERA5 | Avoid seasonal bias |
| **Extreme events** | ERA5 | More complete sampling |

**Bottom Line:** Use **ERA5 as truth dataset** for main results, cite HRES-fc0 comparison in supplementary material with appropriate caveats about data gaps.
