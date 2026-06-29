

# Line endings issues
Run on the terminal for all scripts:

```bash
dos2unix bash_scripts/*.sh
dos2unix job_scripts/*.pbs
```

This might be necessary to avoid issues with line endings when scripts were edited or downloaded on Windows.

# Download scripts

The download scripts download WeatherBench 2 data and save it under `Data/`.

```bash
qsub job_scripts/submit_download_era5_wb2.pbs
qsub job_scripts/submit_download_hres_fc0_wb2.pbs
qsub job_scripts/submit_download_gencast_wb2.pbs
qsub job_scripts/submit_download_ifs_ens_wb2.pbs
```

What they do:

ERA5      : verification data for t2m and w10
HRES-fc0  : deterministic verification/reference forecast at lead 0
GenCast   : ensemble forecasts for t2m and w10, all lead times
IFS ENS   : ensemble forecasts for t2m and w10, all lead times


The GenCast and IFS ENS jobs are array jobs over:

2 variables x 20 lead times = 40 tasks

with lead times:

12, 24, 36, ..., 240 hours

So, if you want to provide new forecast files, make sure we have one Zarr file per year, variable, and lead time, with the same structure as the WB2 ensemble forecasts (dimensions:
time, number, prediction_timedelta, latitude, longitude)
so valid time = time + prediction_timedelta, (where time=init_time) and number is the ensemble member index
# Evaluation scripts

The evaluation scripts run the four evaluation types:

```bash
qsub job_scripts/submit_eval_crps.pbs
qsub job_scripts/submit_eval_margin.pbs
qsub job_scripts/submit_eval_occurrence.pbs
qsub job_scripts/submit_eval_point.pbs
```

Each evaluation job runs over:

1. GenCast, t2m
2. GenCast, w10
3. IFS_ENS, t2m
4. IFS_ENS, w10


You can always override settings at submission time, for example:

```bash
qsub -v YEAR=2018,TRUTH=HRES_fc0 job_scripts/submit_eval_point.pbs
qsub -v MARGIN_MAX=10,MARGIN_STEP=0.5 job_scripts/submit_eval_margin.pbs
```

or if you want to only run certain tasks (e.g., only GenCast t2m):
```bash
qsub -t 1 job_scripts/submit_eval_point.pbs
qsub -t 1 job_scripts/submit_eval_occurrence.pbs
qsub -t 1 job_scripts/submit_eval_crps.pbs
qsub -t 1 job_scripts/submit_eval_margin.pbs
```
# Important

Before running the full download or evaluation, remove any debug restrictions in the Python files, such as:

```python
leads = [24]
n_members = 1
latitude=slice(-10, 10)
longitude=slice(-10, 10)
```

Otherwise the jobs will only run a small debug version.
