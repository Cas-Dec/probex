# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`probex` evaluates probabilistic ensemble weather forecasts (GenCast, IFS ENS) against verification data (ERA5, HRES-fc0) with a focus on record-breaking extremes. Variables are 2m temperature (`t2m`) and 10m wind speed (`w10`). All scripts must be run from the project root.

## Environment

Activate the Python environment before running any eval or download scripts:

```bash
source ~/venvs/weatherbench/bin/activate
```

If scripts were edited on Windows, fix line endings first:

```bash
dos2unix bash_scripts/*.sh
dos2unix job_scripts/*.pbs
```

## Running scripts

**On HPC (PBS scheduler) — full runs:**

```bash
# Download WeatherBench 2 data
qsub job_scripts/submit_download_era5_wb2.pbs
qsub job_scripts/submit_download_hres_fc0_wb2.pbs
qsub job_scripts/submit_download_gencast_wb2.pbs   # array job: 2 vars × 20 leads
qsub job_scripts/submit_download_ifs_ens_wb2.pbs   # array job: 2 vars × 20 leads

# Evaluation (array job over 4 tasks: GenCast t2m/w10, IFS_ENS t2m/w10)
qsub job_scripts/submit_eval_crps.pbs
qsub job_scripts/submit_eval_point.pbs
qsub job_scripts/submit_eval_occurrence.pbs
qsub job_scripts/submit_eval_margin.pbs
```

**Overriding defaults at submission:**

```bash
qsub -v YEAR=2018,TRUTH=HRES_fc0 job_scripts/submit_eval_point.pbs
qsub -v MARGIN_MAX=10,MARGIN_STEP=0.5 job_scripts/submit_eval_margin.pbs
qsub -t 1 job_scripts/submit_eval_crps.pbs   # run only task 1 (GenCast t2m)
```

**Running locally (via bash scripts):**

```bash
bash bash_scripts/run_eval_crps.sh GenCast t2m 2020 ERA5 1979_2018
bash bash_scripts/run_eval_point.sh GenCast t2m 2020 ERA5 1979_2018
bash bash_scripts/run_eval_occurrence.sh IFS_ENS w10 2020 ERA5 1979_2018
bash bash_scripts/run_eval_margin.sh GenCast t2m 2020 ERA5 1979_2018 6.0 0.25
```

**Running eval scripts directly (for debugging):**

```bash
cd <project_root>
python -u eval/forecast_evaluation_crps.py --year 2020 --var t2m --model GenCast --truth ERA5
python -u eval/forecast_evaluation_point.py --year 2020 --var t2m --model GenCast --truth ERA5
python -u eval/forecast_evaluation_occurrence.py --year 2020 --var t2m --model GenCast --truth ERA5
python -u eval/forecast_evaluation_margin.py --year 2020 --var t2m --model GenCast --truth ERA5 --margin-max 6.0 --margin-step 0.25
```

## Critical: debug flags

The evaluation and download scripts contain hardcoded debug overrides that restrict the domain, lead times, and ensemble members. **Remove these before running full jobs:**

```python
# In eval scripts — remove these debug lines:
leads = [24]          # should be the full list of 20 lead times
n_members = 1         # should be 50
no_Antarctic = dict(latitude=slice(-10, 10), longitude=slice(-10, 10))  # should be slice(-60, 90)

# In download scripts — similar debug overrides on lat_min/lat_max and n_members
```

## Data layout

```
data/
  ERA5/          <year>_no_Antarctic_<var>_0p25.zarr
  HRES_fc0/      HRES_fc0_<year>_no_Antarctic_<var>_0p25.zarr
  GenCast/       GenCast_<year>_no_Antarctic_<var>_m<N>_0p25_lead_<LLL>h.zarr
  IFS_ENS/       IFS_ENS_<year>_no_Antarctic_<var>_m<N>_0p25_lead_<LLL>h.zarr
  era5_records/  ERA5_record_{max,min}_month_<tag>_<var>.nc
                 ERA5_lsm_sfc.nc
results/
  <model>/
    CRPS/        <model>_<case>_CRPS_<year>_<var>_no_Antarctic_records_<tag>_cond_<truth>.nc
    Point/       <model>_<case>_point_...nc
    Occurrence/  <model>_<case>_occurrence_...nc
    Margin/      <model>_<case>_margin_...nc
logs/            one log per PBS array task
```

One Zarr file per year, variable, and lead time. Forecast dimensions: `time, number, prediction_timedelta, latitude, longitude` (with `valid_time = time + prediction_timedelta`). If downloads land in `Data/` (capital D), the bash scripts create a lowercase symlink automatically.

## Architecture

### Evaluation pipeline

Each `eval/forecast_evaluation_<type>.py` script follows the same pattern:
1. Parse CLI args (`--year`, `--var`, `--model`, `--truth`, `--record-tag`)
2. Load truth (ERA5 or HRES-fc0 Zarr) and ERA5 monthly records (NetCDF)
3. Transform ERA5 time into a `(year, month, day, hour)` MultiIndex via `functions.transform_ERA5_time()` — this is how observations are aligned to forecast valid times
4. Loop over lead times (12–240 h, step 12), loading one Zarr per lead, converting init time → valid time
5. Apply event masks (observed record-breaking events) and land/sea mask
6. Compute scores via dask, collect scalars, save NetCDF

### Eval modules

| File | Purpose |
|------|---------|
| `eval/functions.py` | Time transforms, latitude-weighted RMSE/bias, seasonal/land-sea selection |
| `eval/functions_crps.py` | Ensemble CRPS (sorted-member formula), weighted CRPS summaries |
| `eval/functions_occurrence.py` | Brier score, reliability diagram, precision-recall; `forecast_members_exceed()` converts ensemble to record-occurrence probability |
| `eval/utils.py` | Unit conversion (K→°C), coordinate normalisation (0–360→−180–180, lat sort) |
| `eval/regions.py` | Named lat/lon slices (Global, Europe, Tropics, etc.) |

### Evaluation cases

| Case | Meaning |
|------|---------|
| `max` | Observed grid-cell exceeds monthly record maximum |
| `min` | Observed grid-cell falls below monthly record minimum (t2m only) |
| `all` | All valid data (no record filter; used in CRPS and Point) |

### Score dimensions

All output NetCDFs share `leadtime` (hours), `season` (DJF/MAM/JJA/SON/all), and `landsea` (all/land/sea) coordinates. Margin adds a `margin` axis (how many σ/K above the record threshold). Occurrence adds `reliability_bin` and `threshold` axes.

### Dask strategy

All large arrays are opened with `chunks={}` (let Zarr choose) and re-chunked to `{time: 31, number: -1, latitude: 30, longitude: 60}` before compute. All dask tasks for a given lead are batched into a single `dask.compute(*tasks)` call to avoid scheduler overhead. Forecasts are `del`-ed and `gc.collect()`-ed after each lead.
