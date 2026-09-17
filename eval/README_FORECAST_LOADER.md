# Forecast Loader Update

## Overview

The evaluation scripts have been updated to handle both per-lead-time Zarr files (IFS_ENS, FGN) and combined Zarr stores (ArchesWeatherGen, uCast).

## What Was Done

### 1. Created Helper Module ✅

**File:** `eval/forecast_loader.py`

This module provides a unified interface for loading forecast data regardless of storage format:
- `load_forecast_for_lead()` - Main function to load forecast for a specific lead time
- Automatically detects model type and uses appropriate loading method
- Handles dimension renaming and standardization
- Caches combined Zarr datasets for efficiency

### 2. Updated CRPS Evaluation ✅

**File:** `eval/forecast_evaluation_crps.py`

- Added `import forecast_loader`
- Replaced manual forecast loading logic with `forecast_loader.load_forecast_for_lead()`
- Now works with all model types

### 3. Added Imports to Other Scripts ✅

Added necessary imports to:
- `eval/forecast_evaluation_point.py`
- `eval/forecast_evaluation_occurrence.py`
- `eval/forecast_evaluation_margin.py`

### 4. Remaining Work ⚠️

The other three scripts (point, occurrence, margin) still need their forecast loading logic replaced with the helper function.

## How to Complete the Update

For each of `eval/forecast_evaluation_{point,occurrence,margin}.py`:

### Find the forecast loading section

Look for code like:
```python
for i, lead_hour in enumerate(leads, start=1):
    forecast_file = (
        forecast_root / model
        / f"{model}_{year}_no_Antarctic_{var}_m{n_members}_{resolution_tag}_lead_{lead_hour:03d}h.zarr"
    )
    forecast = xr.open_zarr(forecast_file, chunks={})[variable]
    forecast = forecast.sel(no_Antarctic)
```

### Replace with:

```python
for i, lead_hour in enumerate(leads, start=1):
    print("\n" + "=" * 80)
    print(f"{model} {var}: lead {lead_hour:03d}h ({i}/{len(leads)})")

    # Load forecast data (handles both per-lead and combined Zarr formats)
    forecast = forecast_loader.load_forecast_for_lead(
        model=model,
        var=var,
        year=year,
        lead_hour=lead_hour,
        forecast_root=forecast_root,
        n_members=n_members,
        resolution_tag=resolution_tag,
        no_Antarctic=no_Antarctic,
        variable=variable,
    )

    print("=" * 80)
```

## Testing

Once all scripts are updated, test with:

```bash
# Test IFS_ENS (per-lead format)
qsub -t 1 job_scripts/submit_eval_crps_all_2022records.pbs

# Test ArchesWeatherGen (combined format)
qsub -t 3 job_scripts/submit_eval_crps_all_2022records.pbs

# Test uCast (combined format)
qsub -t 5 job_scripts/submit_eval_crps_all_2022records.pbs
```

Check logs for any errors related to dimension names or data loading.

## Key Differences Between Models

### Per-Lead Files (IFS_ENS, FGN)
- One Zarr file per lead time
- File pattern: `{model}_{year}_no_Antarctic_{var}_m{n}_1p50_lead_{LLL}h.zarr`
- Already in correct format

### Combined Zarr (ArchesWeatherGen, uCast)
- One Zarr file with all lead times
- File: `{lowercase_model}_forecasts_{year}.zarr`
- Need to:
  - Select specific lead time
  - Rename dimensions (`member`/`ensemble_member` → `number`, `init_time` → `time`)
  - Add `prediction_timedelta` dimension
  - Handle variable name differences (`2m_temperature` vs `2m_temperature_predicted`)

## Variable Name Mappings

### ArchesWeatherGen
- `t2m` → `'2m_temperature'`
- `w10` → `'10m_wind_speed'`

### uCast
- `t2m` → `'2m_temperature_predicted'`
- `w10` → `'10m_wind_speed_predicted'`

(Computed from u and v components: `sqrt(u^2 + v^2)`)

## Notes

- The forecast_loader module caches combined Zarr datasets to avoid repeatedly loading the same file
- uCast uses `timedelta64[s]` for lead_time coordinate, which is automatically converted to hours
- All models are converted to standard dimensions expected by evaluation functions
