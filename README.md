# ProbEx

Evaluation of probabilistic weather forecasts for record-breaking temperature
and 10 m wind. The current 2022 comparison uses monthly ERA5 records from
1979–2021, sampled at 00/12 UTC, on a common 1.5° grid north of 60°S.

## What is implemented

| Component | Entry point |
| --- | --- |
| Monthly heat, cold and wind records; optional local scales | `download/create_era5_records.py` |
| Fair CRPS, all cases and observed records; fair rtwCRPS at the record | `eval/forecast_evaluation_crps.py` |
| Fair Brier score, reliability and precision/recall | `eval/forecast_evaluation_occurrence.py` |
| Ensemble-mean/median RMSE and bias | `eval/forecast_evaluation_point.py` |
| Diagnostics grouped by observed physical record margin | `eval/forecast_evaluation_margin.py` |
| Paired scores across standardized threshold depths | `eval/forecast_evaluation_depth.py` |
| Lead-time figures from existing summaries | `plot_land_scores.py` |
| Depth curves and relative skill figures | `plot_record_depth.py` |
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

On Biobot, the project data directory can be
`/data/nvme1/bastiaan/probex-data`; no script requires that filesystem.
`ERA5_TRUTH_ZARR` is the local verification store. `ERA5_RECORD_SOURCE` is the
full historical source and must contain every requested historical timestamp;
a 2022-only truth store cannot produce 1979–2021 records.

### ERA5 truth and records/scales

If the local 2022 truth store is missing:

```bash
python -u download/download_era5_truth_wb2.py --year 2022
```

This downloads only T2m, W10 and the static land–sea mask at 1.5°, retaining
00/12 UTC and the evaluation latitude range. It refuses to overwrite an
existing store. Use `--validate-only` for its basic schema/year/hour check.
The older `download_era5_wb2.py` is retained for its separate download workflow;
it is not the compact two-variable truth-store builder.

For monthly thresholds alone:

```bash
python -u download/create_era5_records.py
```

For thresholds **and local scales**, in one shared reduction:

```bash
python -u download/create_era5_records.py --with-scale
```

The defaults are 1979–2021, 00/12 UTC and the full global record grid, preserving
the original main-branch record domain. If adding scales to records previously
created on Biobot with a southern bound of −60°, use `--lat-min -60` again.
Existing records are checked before a corresponding scale file is written;
a different grid, source, baseline or sampling requires resolving the mismatch.
Do not delete existing records merely to bypass this check.

The source may be local or `gs://`. Historical fields are reduced lazily; no
full historical ERA5 archive is saved locally. Progress counts completed Dask
tasks, not bytes or a reliable remaining-time estimate.

Outputs in `RECORDS_ROOT`:

- `ERA5_record_{max|min}_month_1979_2021_t2m.nc`
- `ERA5_record_max_month_1979_2021_w10.nc`
- `ERA5_lsm_sfc.nc`
- With `--with-scale`: matching `ERA5_record_scale_...nc` files containing
  `record` and `scale` on `(month, lat, lon)`.

For each grid cell and calendar month, the scale is the sample standard
deviation (`ddof=1`) across the **43 yearly monthly maxima or minima**.
It is not the SD of monthly means or within-month weather values; no detrending
is applied. Complete timestamps are required. Scale generation also rejects
non-finite historical monthly blocks. Metadata records the source, years,
hours and scale definition.

The shell wrapper calculates both records and scales. It uses four positional
arguments: record tag, first year, last year and UTC hours.

```bash
bash bash_scripts/run_create_era5_records.sh 1979_2021 1979 2021 0,12
```

For a different latitude range, use the Python command directly with
`--with-scale --lat-min -60`, as described above.

### Forecast inputs

Existing WB2 downloaders and conversion scripts remain in `download/`.
For example, download one IFS/FGN lead:

```bash
python download/download_ifs_ens_wb2.py --year 2022 --variable 2m_temperature --lead-hour 48
python download/download_fgn_wb2.py --year 2022 --variable 2m_temperature --lead-hour 48
```

Repeat for the needed leads and `10m_wind_speed`, or use the existing shell/PBS
wrappers. Main's IFS source and remote-access fixes are retained. FGN member
coordinates are unique positional IDs; the original source labels are kept
as `source_sample`, without inferring parameter-seed identities.

The per-lead contract is:

```text
FORECAST_ROOT/<model>/<model>_2022_no_Antarctic_<t2m|w10>_m<50|64>_1p50_lead_<HHH>h.zarr
Dimensions: time, number, prediction_timedelta, latitude, longitude
```

`time` denotes initialization; verification uses initialization plus lead.
Do not assume every model includes December 2021 initializations needed for
all early-January 2022 valid times. `eval/forecast_loader.py` and its companion
README are retained as optional loading utilities; existing evaluators still
use the per-lead files directly.

## Evaluate and plot

Both plotting scripts run from top to bottom: read and check the summaries,
calculate comparisons, draw the figures, and save PNG/PDF files.

For one model/variable against a common ERA5 reference:

```bash
python eval/forecast_evaluation_crps.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
python eval/forecast_evaluation_occurrence.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
python eval/forecast_evaluation_point.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
```

Repeat for FGN and W10. These use model-specific lead lists already in the
scripts (IFS/FGN: 12, 48, 120, 168, 240 h) and save to `results/<model>/`.
After these summaries exist:

```bash
python plot_land_scores.py --models IFS_ENS FGN --surface land --truth ERA5
```

Missing point-score files leave labelled empty RMSE panels. This plotter reads
saved summaries; it cannot repair unmatched verification dates in older runs.

For depth evaluation, use the same interface as the margin script:

```bash
python -u eval/forecast_evaluation_depth.py \
  --year 2022 --var t2m --model FGN --truth ERA5 \
  --leads 12 48 120 240 --depths 0 0.5 1 2
```

Each run evaluates one model and variable, with heat/cold cases for T2m or
upper records for W10. All/land/sea results are saved together. Outputs go to
`results/<model>/Depth/`: one summary NetCDF, CSV and daily-sum NetCDF per
record case. No shard management or resume system is involved; rerunning
recomputes and replaces these outputs.

For a model comparison, give each run the **same** comparison models and leads
so its valid dates are selected from their common overlap:

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

To include AWG or U-Cast, add their names to `--compare-models` and run them
with `--model ArchesWeatherGen` or `--model uCast`. All comparison inputs must
be present. With AWG, select common leads such as `48 120 240`.
`--compare-models` checks the other models' metadata and dates; it scores only
the model named by `--model`.

At depth `k`, the threshold is `R + k*s` for heat/wind and `R - k*s` for cold.
`k=0` is the historical record boundary. Fair rtwCRPS and Brier scores use all
eligible cases. CRPS/RMSE conditioned on observations beyond that threshold
remain secondary diagnostics. The code reuses the existing CRPS/clipping
helpers and centers temperatures when computing small tail scores.

Inputs require exact matching grids, complete ensembles and consistent
historical metadata. Non-finite eligible forecast/truth values cause an error;
zero historical scales are excluded. No nearest-neighbour regridding is done.

The plotter reads the per-model summaries and checks dates, reference data,
depths and event counts before computing relative skill. It saves ordinary
score curves and `100 * (1 - model_score / reference_score)` with IFS_ENS as the
default reference. Negative skill is allowed. Ratios are not drawn where the
reference score is nonpositive or there are no observed exceedances; raw
scores remain available. Figures go to `figures/record_depth/`.

To plot earlier combined Biobot depth outputs, use
`python plot_record_depth.py --results results/Depth`.

## Interpretation and remaining work

- Fair corrections target finite-ensemble sampling effects under appropriate
  independent-sampling assumptions. The empirical CRPS is a valid score for
  the issued empirical distribution; it was not an invalid formula. The depth
  output retains empirical companions for comparison, including for pooled
  model-seed ensembles whose sampling assumptions need checking.
- CRPS/RMSE conditioned on observed records are descriptive. Primary tail
  comparisons use all-case rtwCRPS and Brier scores. Lower raw tail scores at
  deeper thresholds do not by themselves mean increasing skill.
- Daily paired contributions are saved, but **bootstrap confidence intervals
  are not implemented**. Counts are grid-cell-times, not independent events.
- For 2022, records must end in 2021. Older evaluation years need a strictly
  earlier record baseline. Training/checkpoint provenance is a separate check;
  these scripts do not establish a strict out-of-training evaluation.
- All evaluators, including depth, allow `--truth HRES_fc0`. Some `job_scripts/submit_eval_*_all.pbs` default to HRES_fc0
  for IFS and ERA5 for AI models. Set `TRUTH=ERA5` explicitly for a common-ERA5
  comparison (for PBS, pass it with `qsub -v`).
- Reference-data sensitivity and HRES-fc0 coverage diagnostics remain in
  [`analysis/README_VERIFICATION_COMPARISON.md`](analysis/README_VERIFICATION_COMPARISON.md).
  The baseline comparison and saved-result comparison scripts are documented
  in [`analysis/README.md`](analysis/README.md).
