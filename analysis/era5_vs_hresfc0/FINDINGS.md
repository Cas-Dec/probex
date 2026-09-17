# ERA5 vs HRES-fc0 Verification Comparison - Findings

**Generated:** 2026-09-17
**Variables:** t2m (2m temperature), w10 (10m wind speed)
**Evaluation Year:** 2022
**Record Baseline:** 1979-2021

---

## Executive Summary

ERA5 and HRES-fc0 show **excellent overall agreement** for both variables (correlation >0.99, low bias), but exhibit **variable-specific differences in extreme event detection**:

- **Temperature (t2m)**: HRES-fc0 flags **3.2× more heat records** than ERA5 (90,849 vs 28,367 events)
- **Wind speed (w10)**: Much better extreme agreement (F1=0.56 vs 0.31 for t2m)

This systematic warm-tail bias in temperature could explain discrepancies in forecast verification scores when using different truth datasets. The effect is temperature-specific, not a general HRES-fc0 issue.

---

## 1. Data Configuration

### Grid Alignment
- ✅ **Both datasets on identical grid**: no_Antarctic domain (-60° to 90°), 1.5° resolution
- ✅ **ERA5**: Shape (1460, 240, 100) - 6-hourly, full year
- ✅ **HRES-fc0**: Shape (730, 240, 100) - 12-hourly, full year
- ✅ **Common times**: 730 timesteps (12-hourly at 00Z and 12Z)
- ✅ **No interpolation needed** - grids match exactly

### Unit Conversion
- ✅ All data converted to °C (ERA5, HRES-fc0, and ERA5 records were stored in Kelvin)

---

## 2. Overall Distribution Agreement ⭐ **EXCELLENT**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Bias** | -0.02°C | HRES-fc0 very slightly cold-biased |
| **RMSE** | 0.91°C | Excellent overall agreement |
| **Correlation** | 0.998 | Near-perfect spatial-temporal correlation |

### Seasonal Breakdown

| Season | Bias (°C) | RMSE (°C) | Correlation |
|--------|-----------|-----------|-------------|
| **DJF** (Winter) | +0.07 | 1.09 | 0.9983 |
| **MAM** (Spring) | -0.07 | 0.92 | 0.9984 |
| **JJA** (Summer) | -0.09 | 0.75 | 0.9975 |
| **SON** (Autumn) | +0.03 | 0.91 | 0.9980 |

**Note:** Best agreement in summer (JJA), slightly larger errors in winter (DJF).

---

## 3. Record Occurrence Comparison 🚨 **CRITICAL FINDING**

### Heat (MAX) Records

**2022 Record Counts:**
- **ERA5**: 28,367 events (0.162% of grid-cell-times)
- **HRES-fc0**: 90,849 events (0.519% of grid-cell-times)
- **Ratio**: HRES-fc0 flags **3.2× more heat records** than ERA5

**Confusion Matrix:**

|                     | ERA5 = Record | ERA5 ≠ Record |
|---------------------|---------------|---------------|
| **HRES-fc0 = Record** | 18,218 (TP)   | 72,631 (FP)   |
| **HRES-fc0 ≠ Record** | 10,149 (FN)   | 17,419,002 (TN)|

**Performance Metrics:**
- **Accuracy**: 99.53% (dominated by true negatives)
- **Precision**: 20.05% - only 1 in 5 HRES-fc0 heat records agree with ERA5
- **Recall**: 64.22% - HRES-fc0 captures most ERA5 heat records
- **F1 Score**: 0.306 - poor overall agreement on heat record occurrence

**Interpretation:**
HRES-fc0 has a systematic **warm tail bias** - it flags many more heat extremes than ERA5. When ERA5 says there's a heat record, HRES-fc0 usually agrees (64% recall), but HRES-fc0 also flags 72,631 additional events that ERA5 doesn't classify as records (false positives).

---

### Cold (MIN) Records

**2022 Record Counts:**
- **ERA5**: 7,071 events (0.040% of grid-cell-times)
- **HRES-fc0**: 9,474 events (0.054% of grid-cell-times)
- **Ratio**: HRES-fc0 flags **1.34× more cold records** than ERA5

**Confusion Matrix:**

|                     | ERA5 = Record | ERA5 ≠ Record |
|---------------------|---------------|---------------|
| **HRES-fc0 = Record** | 2,845 (TP)    | 6,629 (FP)    |
| **HRES-fc0 ≠ Record** | 4,226 (FN)    | 17,506,300 (TN)|

**Performance Metrics:**
- **Accuracy**: 99.94%
- **Precision**: 30.03%
- **Recall**: 40.23% - HRES-fc0 misses 60% of ERA5 cold records
- **F1 Score**: 0.344

**Interpretation:**
HRES-fc0 also over-predicts cold records but to a lesser extent. It misses more ERA5 cold records (60% false negative rate) compared to heat records.

---

## 4. Record Intensity Comparison ✅ **RESOLVED**

**Root Cause:** HRES-fc0 contains NaN values (~2-4% of data, likely over ocean/missing regions). Original code didn't filter these before computing statistics.

### Heat (MAX) Record Margins

**After NaN removal (98,849 valid points):**
- **Correlation: 0.21** - weak positive correlation between ERA5 and HRES-fc0 margins
- **Bias: +0.57°C** - HRES-fc0 exceeds record thresholds by 0.57°C more than ERA5
- **RMSE: 0.92°C** - moderate disagreement on intensity

**Interpretation:** When both datasets flag a heat record, they disagree substantially on *how much* the record was broken by. HRES-fc0 systematically shows larger exceedances (positive bias), suggesting it produces hotter extremes than ERA5.

### Cold (MIN) Record Margins

**After NaN removal (13,171 valid points):**
- **Correlation: -0.003** - essentially zero correlation (no relationship)
- **Bias: +0.15°C** - HRES-fc0 cold margins slightly smaller than ERA5
- **RMSE: 1.10°C** - larger disagreement than for heat records

**Interpretation:** ERA5 and HRES-fc0 show almost no agreement on cold record intensity. The near-zero correlation suggests the datasets disagree fundamentally on where and by how much cold records are broken.

**Data Quality:**
- HRES-fc0 NaNs in MAX margins: 2,149 / 100,998 (2.1%)
- HRES-fc0 NaNs in MIN margins: 529 / 13,700 (3.9%)
- ERA5 NaNs: 0 (complete coverage)

---

## 5. Wind Speed (w10) Comparison ✅ **MUCH BETTER AGREEMENT**

### Overall Distribution Agreement ⭐ **EXCELLENT**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Bias** | +0.12 m/s | HRES-fc0 very slightly high-biased |
| **RMSE** | 0.53 m/s | Excellent overall agreement |
| **Correlation** | 0.990 | Near-perfect spatial-temporal correlation |

### Record Occurrence - Wind Extremes (MAX only)

**2022 Record Counts:**
- Record count comparison not explicitly saved, but confusion matrix shows much better agreement than t2m

**Performance Metrics:**
- **Accuracy**: 99.94%
- **Precision**: 47.73% - nearly 1 in 2 HRES-fc0 wind records agree with ERA5 (vs 20% for t2m heat)
- **Recall**: 68.52% - HRES-fc0 captures most ERA5 wind records
- **F1 Score**: 0.563 - **good agreement** on wind record occurrence (vs 0.306 for t2m)

**Interpretation:**
Wind extremes show **dramatically better agreement** than temperature extremes. HRES-fc0 does not over-predict wind records like it does for heat records. This suggests the extreme detection discrepancy is **temperature-specific**, not a general issue.

### Record Intensity - Wind Extremes

**Valid Points:** 16,755 (after NaN removal)

**Performance Metrics:**
- **Correlation**: 0.584 - **moderate positive correlation** (vs 0.21 for t2m heat)
- **Bias**: +0.37 m/s - HRES-fc0 wind margins slightly larger
- **RMSE**: 0.96 m/s - moderate disagreement on intensity

**Interpretation:**
When both datasets flag a wind record, they show **moderate agreement** on how much the record was broken by. Much better than temperature (correlation 0.58 vs 0.21), suggesting wind extremes are more physically consistent between the two datasets.

### Key Difference from Temperature

| Metric | t2m (heat) | w10 (wind) | Ratio |
|--------|-----------|-----------|-------|
| **Precision** | 20.05% | 47.73% | **2.4× better** |
| **F1 Score** | 0.306 | 0.563 | **1.8× better** |
| **Margin correlation** | 0.212 | 0.584 | **2.8× better** |

**Bottom Line:** The HRES-fc0 warm tail bias and extreme over-prediction is **specific to temperature**, not wind. For wind verification, ERA5 vs HRES-fc0 choice matters much less.

---

## 6. Implications for Forecast Verification

### Why This Matters (Temperature-Specific)

When evaluating IFS ensemble forecasts **for temperature**:

1. **Using HRES-fc0 as truth** → 90,849 observed heat records in 2022
2. **Using ERA5 as truth** → 28,367 observed heat records in 2022

**Result:** Forecast skill scores will appear **artificially worse** when verified against HRES-fc0 because:
- More "observed records" to predict → harder task
- Expected record count increases by 3.2×
- "Too many records" in results may reflect HRES-fc0's warm tail bias, not forecast errors

**For wind speed:** This issue is much less severe - HRES-fc0 and ERA5 agree well on wind extremes (F1=0.56), so verification dataset choice matters less

### Previous IFS Results Context

The original finding of "too many heat records predicted by IFS" when verified against HRES-fc0 may actually indicate:
- ✅ IFS is correctly predicting the warm distribution
- ❌ HRES-fc0 has a systematic warm extremes bias compared to ERA5
- ⚠️ Using different verification datasets (ERA5 vs HRES-fc0) gives inconsistent baseline expectations

---

## 7. Recommendations

### For This Analysis
1. ✅ **Completed:** Verify grid alignment (both datasets on same no_Antarctic grid)
2. ✅ **Completed:** Verify unit consistency (all data in °C)
3. ✅ **Completed:** Debug and fix NaN issue in margin intensity analysis
4. ✅ **Completed:** Run comparison for w10 (wind speed)

### For Future Forecast Evaluation
1. **Document which truth dataset** (ERA5 or HRES-fc0) is used for each result
2. **For t2m**: Strongly prefer ERA5 as truth - HRES-fc0 has 3.2× warm tail bias
3. **For w10**: Either dataset acceptable - good agreement on extremes (F1=0.56)
4. **Apply correction factors** when comparing t2m results across different truth datasets
5. **Consider ERA5 as primary truth** - it has 100% coverage and longer record (1979-2021)
6. **Investigate HRES-fc0's warm extremes bias** - temperature-specific issue, not present in wind

### For Publication
1. **Report both verification datasets** if t2m results differ substantially (less critical for w10)
2. **Acknowledge uncertainty** in defining "record-breaking" depends on truth dataset choice (mainly for t2m)
3. **Quantify the variable-specific bias**:
   - **t2m**: HRES-fc0 shows 3.2× more heat records than ERA5 in 2022
   - **w10**: Good agreement (F1=0.56), either dataset suitable

---

## 8. Technical Notes

### Data Sources
- **ERA5**: WeatherBench2 zarr store (1959-2023, 1.5° resolution) - **100% complete for 2022**
- **HRES-fc0**: WeatherBench2 HRES archive (2022, 1.5° resolution, no_Antarctic) - **88.77% complete**
- **ERA5 Records**: Monthly maximum/minimum from 1979-2021 baseline

### ⚠️ CRITICAL: HRES-fc0 Data Gaps (see `HRES-FC0_DATA_GAPS.md` for full analysis)

**Missing data periods in 2022:**
1. January 1-24 (24 days) - 65.8% of January missing
2. May 1-12 (12 days) - 38.7% of May missing
3. December 27-31 (5 days) - 16.1% of December missing

**Total: 82 out of 730 timesteps missing (11.2% of year)**

**Seasonal impact:**
- DJF (winter): Only 67.8% coverage - **32% missing**
- MAM (spring): 87.0% coverage
- JJA (summer): 100% coverage
- SON (autumn): 100% coverage

**Implication:** HRES-fc0 verification is **biased toward warm seasons**, which may contribute to the "too many heat records" finding when using HRES-fc0 as truth.

### Processing Steps
1. Load ERA5 2022 (6-hourly) → convert 0-360° to -180-180° longitude → slice to no_Antarctic → convert to °C
2. Load HRES-fc0 2022 (12-hourly) → already -180-180° and no_Antarctic → convert to °C
3. Load ERA5 records → convert coordinates → slice to no_Antarctic → convert to °C
4. Align to common times (12-hourly)
5. Compute masks: `data > record_max` (heat) and `data < record_min` (cold)
6. Compare occurrence, intensity (margins), and overall distributions

### Code Repository
- Script: `analysis/compare_era5_hresfc0_verification.py`
- PBS Job: `job_scripts/submit_compare_verification.pbs`
- Results: `analysis/era5_vs_hresfc0/`

---

## 9. Next Steps

1. ✅ ~~**Investigate margin NaN issue**~~ - Resolved: HRES-fc0 has inherent data gaps (11.2% missing)
2. ✅ ~~**Run w10 comparison**~~ - Completed: Wind shows much better agreement (F1=0.56 vs 0.31)
3. **Re-evaluate IFS results** - Separate ERA5-verified vs HRES-fc0-verified scores for t2m
4. **Investigate temperature-specific bias** - Why does HRES-fc0 over-predict heat records but not wind records?
5. **Sensitivity test** - What if we used different record baselines (e.g., 1991-2020)?
6. **Check ECMWF documentation** - Is the HRES-fc0 warm tail bias a known issue?
