# ProbEx

Evaluation of probabilistic weather forecasts for record-breaking temperature
and 10 m wind. The 2022 comparison uses monthly ERA5 records from
1979–2021, sampled at 00/12 UTC, on a common 1.5° grid north of 60°S.

## Outline of our code

| Content | Where to find it |
| --- | --- |
| Monthly heat, cold and wind local records + scales | `download/create_era5_records.py` |
| Fair CRPS, all cases and observed records; fair rtwCRPS at the record | `eval/forecast_evaluation_crps.py` |
| Fair Brier score, reliability and precision/recall | `eval/forecast_evaluation_occurrence.py` |
| Ensemble-mean/median RMSE and bias | `eval/forecast_evaluation_point.py` |
| Diagnostics grouped by  record margin | `eval/forecast_evaluation_margin.py` |
| Diagnostics grouped by threshold depths | `eval/forecast_evaluation_depth.py` |
| Lead-time figures | `plot_land_scores.py` |
| Depth vs skill figures | `plot_record_depth.py` |
| ERA5/HRES-fc0 reference-data comparison | `analysis/compare_era5_hresfc0_verification.py` |


## Setup and data paths

Run commands from the repository root in the project environment:

```bash
conda env create -f environment.yaml
conda activate probex
```

Scripts read `.env`; existing exported environment variables take precedence
in the Python entry points. Configure local paths there, or export them:

```bash
export FORECAST_ROOT=/path/to/probex-data/forecasts
export ERA5_TRUTH_ZARR=/path/to/probex-data/era5/ERA5_2022_0012_no_Antarctic_1p50.zarr
export RECORDS_ROOT=/path/to/probex-data/era5_records
export ERA5_RECORD_SOURCE=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
```


### ERA5 truth and records/scales

To download ERA5

```bash
python -u download/download_era5_truth_wb2.py --year 2022
```

Records and their local scales are always computed together:

```bash
python -u download/create_era5_records.py
```

An existing record file is kept if it exactly matches the script's output format
(variables, grid, dtypes and attributes such as years, hours and source); otherwise it is
recomputed and overwritten.

The source may be local or `gs://`. No
full historical ERA5 archive is saved locally during the calculation. Progress counts completed Dask
tasks.

Outputs to be found in `RECORDS_ROOT`:

- `ERA5_record_{max|min}_month_1979_2021_t2m.nc`
- `ERA5_record_max_month_1979_2021_w10.nc`
- `ERA5_lsm_sfc.nc`

Each record file contains `record` and `scale` on `(month, lat, lon)`.

For each grid cell and calendar month, the scale is the sample standard
deviation (`ddof=1`) across the 43 yearly monthly maxima or minima -  no detrending is applied.

The shell wrapper calculates both records and scales. It uses four positional
arguments: record tag, first year, last year and UTC hours.

```bash
bash bash_scripts/run_create_era5_records.sh 1979_2021 1979 2021 0,12
```

If you want a different latitude range (eg for visualization), use the Python command directly with
`--lat-min -60`, as described above.

### Forecast inputs

WB2 downloaders and conversion scripts to be found in `download/`.
For example, download one IFS/FGN lead:

```bash
python download/download_ifs_ens_wb2.py --year 2022 --variable 2m_temperature --lead-hour 48
python download/download_fgn_wb2.py --year 2022 --variable 2m_temperature --lead-hour 48
```

Repeat for the needed leads and `10m_wind_speed`, or use the shell/PBS
wrappers for HPC work. Main's IFS source and remote-access fixes are retained.

The per-lead file names are:

```text
FORECAST_ROOT/<model>/<model>_2022_no_Antarctic_<t2m|w10>_m<50|64>_1p50_lead_<HHH>h.zarr
Dimensions: time, number, prediction_timedelta, latitude, longitude
```

`time` denotes initialization; verification uses initialization plus lead.
 `eval/forecast_loader.py` and its companion
README are retained as optional loading utilities (?). 

## Evaluate and plot

Both plotting scripts run from top to bottom: read and check the summaries,
calculate comparisons, draw the figures, and save PNG/PDF files.

For one model/variable against a common ERA5 reference:

```bash
python eval/forecast_evaluation_crps.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
python eval/forecast_evaluation_occurrence.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
python eval/forecast_evaluation_point.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
```
and similarly for the AI models and W10. 
Then,

```bash
python plot_land_scores.py --models IFS_ENS FGN --surface land --truth ERA5
```

For depth evaluation, 

```bash
python -u eval/forecast_evaluation_depth.py \
  --year 2022 --var t2m --model FGN --truth ERA5 \
  --leads 12 48 120 240 --depths 0 0.5 1 2
```

Each run evaluates one model and variable, with heat/cold cases for T2m or
upper records for W10. All/land/sea results are saved together. Outputs go to
`results/<model>/Depth/`: one summary NetCDF, CSV and daily-sum NetCDF per
record case. 

For a model comparison (make sure to only pick leads that are common)

```bash
for model in IFS_ENS FGN; do
  for var in t2m w10; do
    python -u eval/forecast_evaluation_depth.py \
      --year 2022 --var "$var" --model "$model" --truth ERA5 \
      --compare-models IFS_ENS FGN --leads 12 48 120 240 \
      --depths 0 0.5 1 2
  done
done
python plot_record_depth.py --models IFS_ENS FGN --surface land
```

At depth `k`, the threshold is `R + k*s` for heat/wind and `R - k*s` for cold.
`k=0` is the historical record boundary. Fair rtwCRPS and Brier scores use all cases.
CRPS/RMSE conditioned on observations beyond that threshold
remain but are deemed secondary and prone to forecaster's dilemma. 

The plotter reads the per-model summaries and checks dates, reference data,
depths and event counts before computing relative skill. It saves ordinary
score curves and `100 * (1 - model_score / reference_score)` with IFS_ENS as the
default reference. Figures go to `figures/record_depth/`.

## Interpretation and remaining work

- Fair corrections target finite-ensemble sampling effects under appropriate
  independent-sampling assumptions. 
- CRPS/RMSE conditioned on observed records are descriptive. Primary tail
  comparisons use all-case rtwCRPS and Brier scores. 
- For 2022, records must end in 2021. 
- All evaluators, including depth, allow `--truth HRES_fc0`. Some `job_scripts/submit_eval_*_all.pbs` default to HRES_fc0
  for IFS and ERA5 for AI models. Set `TRUTH=ERA5` explicitly for a common-ERA5
  comparison (for PBS, pass it with `qsub -v`).
- Reference-data sensitivity and HRES-fc0 coverage diagnostics remain in
  [`analysis/README_VERIFICATION_COMPARISON.md`](analysis/README_VERIFICATION_COMPARISON.md).
  The baseline comparison and saved-result comparison scripts are documented
  in [`analysis/README.md`](analysis/README.md).
