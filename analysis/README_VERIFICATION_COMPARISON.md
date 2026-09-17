# ERA5 vs HRES-fc0 Verification Comparison

## Motivation

This analysis checks if ERA5 and HRES-fc0 agree on 2022 record occurrence and intensity, independent of any forecast model.

**Background**: We observed alarming discrepancies in "Expected - observed records count" where IFS showed:
- Too many heat records
- Too few cold records (t2m)
- Too few wind records (w10)

This raised suspicions of a **possible distribution shift between 2022 ERA5 and HRES-fc0**, particularly for extremes.

## Analysis Components

The script `analysis/compare_era5_hresfc0_verification.py` performs three types of comparisons:

### 1. Record Occurrence Agreement
**Question**: Do ERA5 and HRES-fc0 flag the same grid-cell-times as record-breaking?

- **Method**: Per-pixel confusion matrix (leadtime-independent)
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Output**: Confusion matrix heatmaps for max (and min if t2m) records

### 2. Record Intensity Agreement
**Question**: When records occur, do ERA5 and HRES-fc0 agree on how extreme they are?

- **Method**: At grid-cell-times where at least one source flags a record, compare margins (distance from record threshold)
- **Metrics**: Correlation, Bias, RMSE
- **Output**: Scatter plots and difference histograms

### 3. Overall Distribution Match
**Question**: Do ERA5 and HRES-fc0 show systematic differences across all 2022 data?

- **Method**: Compare full distributions (not just extremes)
- **Metrics**: Spatial bias/RMSE/correlation maps, land/sea split, seasonal split
- **Output**:
  - Spatial maps showing where differences are largest
  - Histograms comparing full distributions
  - QQ plots to identify distribution shifts

## Usage

### Run locally (for testing with limited data):

```bash
bash bash_scripts/run_compare_verification.sh t2m 1979_2021
bash bash_scripts/run_compare_verification.sh w10 1979_2021
```

### Run on HPC (recommended for full analysis):

```bash
# Submit both variables
qsub -v VAR=t2m job_scripts/submit_compare_verification.pbs
qsub -v VAR=w10 job_scripts/submit_compare_verification.pbs

# Monitor
qstat -u $USER
```

### Direct Python invocation:

```bash
conda activate probex
python analysis/compare_era5_hresfc0_verification.py --var t2m --record-tag 1979_2021
python analysis/compare_era5_hresfc0_verification.py --var w10 --record-tag 1979_2021
```

## Outputs

All outputs are saved to `analysis/era5_vs_hresfc0/`:

### Figures
- `confusion_matrix_{var}_{max|min}.png` - Record occurrence agreement
- `margin_scatter_{var}_{max|min}.png` - Record intensity agreement (scatter + difference histogram)
- `spatial_comparison_{var}.png` - Bias/RMSE/correlation maps
- `distribution_comparison_{var}.png` - Histograms and QQ plots

### Summary
- `summary_{var}.txt` - Text summary of all metrics

## Key Differences from Forecast Evaluation

- **No lead time dimension**: Compares verification datasets directly, not forecasts
- **No ensemble dimension**: Both ERA5 and HRES-fc0 are deterministic
- **Same record thresholds**: Uses ERA5 records for both (records are R, evaluation determines E and T)
- **Focus on agreement**: Not evaluating forecast skill, but consistency between truth datasets

## Next Steps (Proposed Workflow)

Based on colleague feedback:

1. **This analysis first** - Check ERA5 vs HRES-fc0 agreement independently
2. **If discrepancies found** - Consider full evaluation swaps:
   - Currently: IFS uses HRES-fc0 as T, ML models use ERA5 as T
   - Proposal: Also run IFS against ERA5, ML models against HRES-fc0
   - Check if model rankings are robust to truth dataset choice

3. **If differences are large** - Investigate all E/T permutations to pinpoint source of discrepancy

## Notes

- **Leadtime-independent**: This compares the raw verification data, not forecasts at different lead times
- **Record definitions**: All records use ERA5 thresholds (this is fixed, R is always ERA5)
- **2022 only**: Focuses on the evaluation year where discrepancies were observed
- **Spatial resolution**: All data at 1.5° (240x121 equiangular grid)

## Questions to Answer

1. **Do ERA5 and HRES-fc0 disagree on which events are records?**
   - Look at confusion matrix metrics
   - High False Positive rate → HRES-fc0 flags more records than ERA5
   - High False Negative rate → ERA5 flags more records than HRES-fc0

2. **When they agree a record occurred, do they agree on intensity?**
   - Look at margin scatter correlation
   - Systematic bias → one dataset consistently more/less extreme

3. **Are there spatial patterns in disagreement?**
   - Look at bias/RMSE maps
   - Check if land/sea or seasonal splits show different patterns

4. **Do the tails of the distribution differ?**
   - QQ plot should be linear if distributions match
   - Departures from 1:1 line in extremes → distribution shift
