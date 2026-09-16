
# probex: Probabilistic Ensemble Forecast Evaluation On Extremes

Evaluates probabilistic ensemble weather forecasts (IFS_ENS, FGN, ArchesWeatherGen and UCast, GenCast) against verification data (ERA5, HRES-fc0) with a focus on record-breaking extremes. Computes fair CRPS, fair record-threshold weighted CRPS, fair Brier scores, probabilistic occurrence/reliability metrics, deterministic point scores, and margin-conditioned scores at 1.5° resolution.

## Installation & Environment Setup

### 1. Clone the repository and navigate to the project root
```bash
cd /kyukon/data/gent/vo/000/gvo00090/vsc42145/Github/probex
```

### 2. Create the Conda environment
```bash
conda env create -f environment.yaml
conda activate probex
```

### 3. Configure data paths (`.env` file)
All data paths are centralized in the `.env` file. Verify the paths match your system:

```bash
cat .env
```

Key paths:
- `FORECAST_ROOT=/data/gent/vo/000/gvo00090/HEAT/Data/forecasts` — Forecast model outputs
- `ERA5_ROOT=/data/gent/vo/000/gvo00090/HEAT/Data/wb2` — ERA5 and HRES-fc0 truth data
- `RECORDS_ROOT=/data/gent/vo/000/gvo00090/HEAT/Data/wb2` — Monthly climatological records and land-sea mask
- `ERA5_TRUTH_ZARR=/data/.../1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr` — ERA5 Zarr store

### 4. Fix line endings (if scripts were edited on Windows)
```bash
dos2unix bash_scripts/*.sh
dos2unix job_scripts/*.pbs
```

### 5. Portable Environment Discovery (for Collaborators)

All bash scripts in `bash_scripts/` are portable: they automatically discover the probex environment and work on any system where:
1. Conda is installed
2. The `probex` environment exists (created via `conda env create -f environment.yaml`)

**How it works:** Each script sources `find_probex_env.sh` which searches standard conda locations and exports:
- `PROBEX_PYTHON` — Path to probex Python executable

Scripts use this variable instead of hardcoded paths, ensuring they work across different conda installations and user environments.

**For different conda/miniconda setups:**  
The helper script searches these standard locations:
```
~/.conda/envs/probex
~/.local/miniconda/envs/probex
~/.local/miniconda3/envs/probex
~/miniconda/envs/probex
~/miniconda3/envs/probex
~/mambaforge/envs/probex
```

If your conda installation uses a different location, manually edit the search paths in `find_probex_env.sh`.

## Quick Start: Running a 2022 Analysis

This workflow downloads and evaluates 4 models (IFS_ENS, FGN, uCast, ArchesWeatherGen) for year 2022 with lead times {12, 48, 120, 168, 240}h and variables t2m (min/max) and w10 (max).

### Step 1: Generate ERA5 Monthly Records
One-time setup: creates monthly climatological extrema from ERA5 (1979–2021), using 00 and 12 UTC verification times.

```bash
qsub job_scripts/submit_create_era5_records.pbs
```

### Step 2: Download WeatherBench 2 Data
Download forecast and truth data for 2022.

```bash
# ERA5 and HRES GT data
qsub job_scripts/submit_download_era5_wb2.pbs
qsub job_scripts/submit_download_hres_fc0_wb2.pbs

# UCast and ArchesWeatherGen forecasts for 2022
qsub job_scripts/submit_download_ucast.pbs # TODO create
qsub job_scripts/submit_download_archesweathergen.pbs # TODO create

# IFS_ENS ensemble forecasts (array job: 2 vars × 5 leads)
qsub job_scripts/submit_download_ifs_ens_wb2.pbs

# FGN ensemble forecasts (array job: 2 vars × 5 leads)
qsub job_scripts/submit_download_fgn_wb2.pbs
```

### Step 3: Convert Combined-Source Forecasts
uCast and ArchesWeatherGen are stored as combined Zarrs; split into per-lead files for evaluation:

```bash
bash bash_scripts/run_convert_ucast_wb2.sh
bash bash_scripts/run_convert_arches_wb2.sh
```

Or via PBS for batch processing:
```bash
qsub job_scripts/submit_convert_ucast_wb2.pbs
qsub job_scripts/submit_convert_arches_wb2.pbs
```

**Lead-time coverage:**
- **uCast**: [12, 48, 120, 168, 240]h ✓
- **ArchesWeatherGen**: [48, 120, 168, 240]h (12h not available; runs on 24h steps)

### Step 4: Run Evaluations
Evaluate all 4 models across 4 metrics: CRPS, point, occurrence (probabilistic), margin.

```bash
# CRPS evaluation (array job: 4 models × 2 vars = 8 tasks)
qsub job_scripts/submit_eval_crps_all.pbs

# Deterministic point scores
qsub job_scripts/submit_eval_point_all.pbs

# Probabilistic occurrence/reliability
qsub job_scripts/submit_eval_occurrence_all.pbs

# Margin-conditioned scores (e.g., how many σ above record)
qsub job_scripts/submit_eval_margin_all.pbs
```

Results saved to `results/{model}/{METRIC}/`.

For backward compatibility, the occurrence output variable remains named
`brier_score`, but it contains Ferro's fair finite-ensemble estimator.

## Running Scripts Locally (for Debugging & Quick Tests)

All scripts in `bash_scripts/` can be run directly without PBS for development, testing, and debugging:

```bash
# Generate ERA5 records
bash bash_scripts/run_create_era5_records.sh [record_tag] [start_year] [end_year] [hours]
bash bash_scripts/run_create_era5_records.sh 1979_2021 1979 2021 0,12  # default

# Download data
bash bash_scripts/run_download_era5_wb2.sh [variable]
bash bash_scripts/run_download_hres_fc0_wb2.sh [variable]
bash bash_scripts/run_download_ifs_ens_wb2.sh [variable] [lead_hour]

# Convert model-specific forecasts
bash bash_scripts/run_convert_ucast_wb2.sh [year]
bash bash_scripts/run_convert_arches_wb2.sh [year]

# Run evaluations
bash bash_scripts/run_eval_crps.sh [model] [variable] [year] [truth] [record_tag]
bash bash_scripts/run_eval_point.sh [model] [variable] [year] [truth] [record_tag]
bash bash_scripts/run_eval_occurrence.sh [model] [variable] [year] [truth] [record_tag]
bash bash_scripts/run_eval_margin.sh [model] [variable] [year] [truth] [record_tag] [margin_max] [margin_step]
```

**Examples:**
```bash
# Create records for full 1979-2021 baseline
bash bash_scripts/run_create_era5_records.sh 1979_2021 1979 2021 0,12

# Download t2m data
bash bash_scripts/run_download_era5_wb2.sh 2m_temperature
bash bash_scripts/run_download_ifs_ens_wb2.sh 2m_temperature 12

# Evaluate IFS_ENS t2m CRPS for 2022 vs ERA5
bash bash_scripts/run_eval_crps.sh IFS_ENS t2m 2022 ERA5 1979_2021

# Test FGN w10 point scores
bash bash_scripts/run_eval_point.sh FGN w10 2022 ERA5 1979_2021
```

Scripts automatically log to `logs/` and save results to `results/{model}/{metric}/`.

## Advanced: Custom Runs

### Override variables at submission
```bash
# Specific year and truth data
qsub -v YEAR=2018,TRUTH=HRES_fc0 job_scripts/submit_eval_point.pbs

# Custom margin thresholds
qsub -v MARGIN_MAX=10,MARGIN_STEP=0.5 job_scripts/submit_eval_margin.pbs
```

### Run single task
```bash
# Task IDs: 1=IFS_ENS/t2m, 2=IFS_ENS/w10, 3=FGN/t2m, 4=FGN/w10, 5=uCast/t2m, 6=uCast/w10, 7=Arches/t2m, 8=Arches/w10
qsub -t 1 job_scripts/submit_eval_crps_all.pbs    # Run only IFS_ENS t2m
```

### Run evaluation scripts locally (debugging)
```bash
# Must activate conda environment first
conda activate probex
python -u eval/forecast_evaluation_crps.py --year 2022 --var t2m --model IFS_ENS --truth ERA5
python -u eval/forecast_evaluation_point.py --year 2022 --var t2m --model uCast --truth ERA5
python -u eval/forecast_evaluation_occurrence.py --year 2022 --var t2m --model ArchesWeatherGen --truth ERA5
python -u eval/forecast_evaluation_margin.py --year 2022 --var t2m --model FGN --truth ERA5
```

## Data Structure

**Grid:** 1.5° equiangular (240 × 121 points, -60° to 90° latitude, -180° to 180° longitude)

```
/data/gent/vo/000/gvo00090/HEAT/Data/
├── wb2/
│   ├── 1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr  (ERA5 truth)
│   ├── ERA5_record_max_month_1979_2021_{t2m,w10}.nc  (Monthly max extrema)
│   ├── ERA5_record_min_month_1979_2021_t2m.nc  (Monthly min extrema, t2m only)
│   └── ERA5_lsm_sfc.nc  (Land-sea mask)
│
├── forecasts/
│   ├── HRES_fc0/
│   │   └── HRES_fc0_{year}_no_Antarctic_{var}_1p50.zarr  (Deterministic reference)
│   ├── IFS_ENS/
│   │   └── IFS_ENS_{year}_no_Antarctic_{var}_m50_1p50_lead_{LLL}h.zarr
│   ├── FGN/
│   │   └── FGN_{year}_no_Antarctic_{var}_m64_1p50_lead_{LLL}h.zarr
│   ├── uCast/
│   │   └── uCast_{year}_no_Antarctic_{var}_m64_1p50_lead_{LLL}h.zarr  (Converted)
│   ├── ArchesWeatherGen/
│   │   └── ArchesWeatherGen_{year}_no_Antarctic_{var}_m64_1p50_lead_{LLL}h.zarr  (Converted)
│   ├── ucast_forecasts_{year}.zarr  (Combined source)
│   └── archesweathergen_forecasts_{year}.zarr  (Combined source)
│
results/
├── {model}/
│   ├── CRPS/
│   ├── Point/
│   ├── Occurrence/
│   └── Margin/
```

## Important Notes

- **Debug flags**: All evaluation scripts contain hardcoded debug restrictions (e.g., `leads = [24]`, `n_members = 1`). Remove these before production runs.
- **Evaluation cases**: "max" (record-breaking highs), "min" (record-breaking lows, t2m only), "all" (no record filter).
- **Coordinate conventions**: Longitude normalized to ±180°, latitude sorted south-to-north.
- **Memory management**: Forecasts are deleted after each lead time to free memory for large dask operations.

## Troubleshooting

### Environment not found
If you see `ERROR: probex conda environment not found`, ensure:
1. The environment was created: `conda env create -f environment.yaml`
2. You're running from the project root directory
3. Your conda installation is on the standard search path (checked by `find_probex_env.sh`)

**Custom conda location:** If your conda environment is in a non-standard location, edit `find_probex_env.sh` to add your path to the search list.

### Data alignment error (xarray)
If evaluation scripts fail with `ValueError: cannot reindex or align along dimension 'longitude' because the (pandas) index has duplicate values`:
- Issue: Forecast or truth data has duplicate longitude values in the coordinate
- Likely cause: Coordinate normalization (±180°, south-to-north latitude) not applied correctly
- Solution: Check [functions.py](eval/functions.py) for coordinate validation and normalization logic



### Running out of memory during large evaluations
Solutions:
1. Reduce dask chunk size in evaluation scripts (default: `{time: 31, number: -1, latitude: 30, longitude: 60}`)
2. Increase available memory per PBS job (edit `*_all.pbs` files)
3. Run evaluation tasks sequentially instead of in parallel arrays

### Script not executable / Permission denied
```bash
chmod +x bash_scripts/*.sh
chmod +x find_probex_env.sh
```
