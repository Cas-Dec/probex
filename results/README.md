# results/

The main evaluation framework is **`<model>/Depth/`**. It shows how forecast skill
changes as observed events go further past the historical record. The other
subdirectories (`CRPS/`, `Point/`, `Occurrence/`, `Margin/`) come from older,
secondary evaluations and are not described here.

```
results/
├── README.md                  ← this file
├── depth_fingerprints.json    ← hash of records + scale + land-sea mask per (record_tag, var, case)
├── ArchesWeatherGen/Depth/
├── FGN/Depth/
├── GenCast-mini/Depth/
├── IFS_ENS/Depth/
└── uCast/Depth/
```

Produced by `eval/forecast_evaluation_depth.py`, submitted via
`job_scripts/submit_eval_depth_all.pbs` → `bash_scripts/run_eval_depth.sh`.

---

## 1. What "depth" means

For every grid cell and verification time, the record of the matching calendar
month is pushed out by k "scale" units:

```
 threshold(k) = R ± k · s          + for max (heat, wind),  − for min (cold)
 event(k)     = truth beyond threshold(k)
              ⇔ depth := ±(truth − R) / s  >  k

 R = ERA5 monthly record (1979–2021, 00/12 UTC)      s = sample SD (ddof=1) of the yearly
                                                         monthly extremes over the same period

        cold tail (min)                                     heat / wind tail (max)
 ◄──────────┼────────┼────────┼──── ··· ────┼────────┼────────┼──────────►  value
        R − 2s   R − 1s      R             R      R + 1s   R + 2s
        k = 2    k = 1     k = 0         k = 0    k = 1    k = 2
```

`depth = 0` is a plain record break. Every larger k selects a stricter, smaller
subset of those events.

## 2. Files

```
 <model> _ <case> _ depth _ <year> _ <var> _ no_Antarctic_records_ <record_tag> _ <truth> .nc / .csv

            t2m max (heat)   t2m min (cold)   w10 max (wind)
 ERA5            ✓                ✓                ✓          3 cases × 2 truths
 HRES_fc0        ✓                ✓                ✓          = 6 .nc + 6 .csv per model
```

* `.nc` is the main file: labelled variables, global metadata and the list of verification times.
* `.csv` is the same table flattened to one row per (leadtime, landsea, depth). Depth-independent
  columns are repeated on every depth row. It has no metadata and no verification times.

Current contents (all for year 2022, record tag `1979_2021`):

| model            | members | lead times (h)        |
|------------------|---------|-----------------------|
| IFS_ENS          | 50      | 48, 120, 168, 240     |
| ArchesWeatherGen | 64      | 48, 120, 168, 240     |
| FGN              | 64      | 48, 120, 168, 240     |
| uCast            | 64      | 48, 120, 168, 240     |
| GenCast-mini     | 32      | 12, 48, 120, 168, 240 |

## 3. How one file is computed

```
 forecast (time, members, lat, lon)    truth: ERA5 or HRES_fc0     ERA5 records: R, s per month
            │                          (00 / 12 UTC only)                    │
            └─────────────┬──────────────┘                                   │
                          ▼                                                  │
     keep only valid times present for EVERY model and EVERY lead            │
     → 710 times, 2022-01-11 00Z … 2022-12-31 12Z (identical across files)   │
                          │                                                  │
                          ▼                                                  ▼
            for each k in depths:   threshold = R ± k·s,   event_k = truth beyond it
                          │
     ┌────────────────────┼─────────────────────────────┬───────────────────────────┐
     ▼                    ▼                             ▼                           ▼
 "_all" scores       "_record" scores              tail scores                 bookkeeping
 crps, bias, rmse    same, but averaged only       rtwcrps, brier,             counts and
 over every point    where event_k happened        probability, averaged       cos(lat) weights
 (same for all k)    (conditional on the outcome)  over every point
     │                    │                             │                           │
     └────────────────────┴──── cos(lat)-weighted mean over (time, lat, lon) ───────┘
                                  separately for landsea = all / land (lsm > 0.5) / sea
```

* **Depth selection:** candidate depths are 0, 0.25, …, 2, 2.5, 3σ. A depth is kept only if it
  has at least `min_sample_size` (= 30) observed events over the whole year. As a result:
  - t2m max vs ERA5 stops at **2.5σ** (10 depths).
  - The other five combinations go to **3.0σ** (11 depths).
  - The list used is in the global attribute `selected_depths`.
* **Eligible points:** cells with scale > 0. `excluded_scale_count` counts the rest; it is 0 everywhere at present.
* **No season dimension:** all scores are for the whole year.

## 4. Variables in a `.nc` file

Dimensions: `leadtime`, `landsea` (all / land / sea), `depth` (σ units), `verification_time`
(coordinate only). Every data variable is ordered `(leadtime, landsea[, depth])`.

```
 ┌─ (leadtime, landsea) ── same at every depth ──────────────────────────────────────────┐
 │ valid_count            number of eligible (time, cell) cases                          │
 │ valid_weight           sum of cos(lat) weights over those cases                       │
 │ excluded_scale_count   cases dropped because scale ≤ 0                                │
 │ crps_all               fair CRPS                       ┐                              │
 │ crps_empirical_all     empirical (non-fair) CRPS       │ over every eligible case     │
 │ bias_all, rmse_all     of the ensemble mean            ┘                              │
 └───────────────────────────────────────────────────────────────────────────────────────┘
 ┌─ (leadtime, landsea, depth) ── CONDITIONAL: averaged only where event_k occurred ─────┐
 │ record_count, record_weight                  size of the event sample                 │
 │ crps_record, crps_empirical_record           CRPS on observed events                  │
 │ bias_record, rmse_record                     ensemble-mean bias / RMSE on events      │
 └───────────────────────────────────────────────────────────────────────────────────────┘
 ┌─ (leadtime, landsea, depth) ── UNCONDITIONAL: averaged over every eligible case ──────┐
 │ rtwcrps / rtwcrps_empirical   record-threshold weighted CRPS: only the part of the    │
 │                               forecast and truth beyond threshold(k) counts           │
 │ brier / brier_empirical       Brier score for the event "truth beyond threshold(k)"   │
 │ probability                   mean forecast probability p (fraction of members beyond)│
 │ observed_probability          observed event frequency = record_weight / valid_weight │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

**Fair vs empirical.** Names without a suffix are the **fair** (finite-ensemble corrected)
estimators. `*_empirical` are the plain ensemble formulas. With m members:

```
 CRPS       fair      = mean|x_i − y| − Σ_{i<j}|x_i − x_j| / (m(m−1))      (Ferro 2014)
            empirical = mean|x_i − y| − Σ_{i<j}|x_i − x_j| / m²
 Brier      fair      = (p − o)² − p(1 − p)/(m − 1)
            empirical = (p − o)²
```

The fair scores remove the advantage a larger ensemble gets just from having more members, so
use them to compare models with different member counts (32 / 50 / 64). The script computes the
fair CRPS first and derives the empirical one from it:
`crps_empirical = (m−1)/m · crps + mean|x−y| / m`.

**Which score to use for what:**

* `rtwcrps` and `brier` are **proper**: they are averaged over every point, not only where an
  event occurred. Use them to **rank models** at each depth.
* The `*_record` scores condition on the observed outcome. That makes them improper (they reward
  forecasts that always predict extremes), but they are useful as **diagnostics**: how large are
  the errors and the bias once an event actually happens?
* `probability` vs `observed_probability` is a quick reliability check: does the model forecast
  record-breaking as often as it is observed?

**Units:** CRPS, rtwCRPS, bias and RMSE are in K (t2m) or m s-1 (w10). Brier, the probabilities,
the counts and the weights are dimensionless (`units: "1"`). Every variable has `long_name`,
`units` and, where relevant, `estimator` / `reference` / `tail` attributes.

**Global attributes:** `model`, `event` (heat / cold / wind), `variable`, `year`,
`verification_target`, `truth_file`, `record_tag`, `units`, `selected_depths`, `min_sample_size`,
`valid_time_count`, `comparison_models`, `common_dates_across_leads`, `reference_fingerprint`.

## 5. Example: ArchesWeatherGen, t2m max vs ERA5, landsea = all

```
 lead depth  events  crps_record  bias_record  rtwcrps   brier    probability  observed_prob
  48h  0σ    27,759     0.37         −0.45     3.5e-4   1.0e-3     1.34e-3       1.76e-3
       1σ     2,102     0.48         −0.61     2.7e-5   8.5e-5     8.8e-5        1.25e-4
       2σ       136     0.55         −0.69     1.6e-6   6.2e-6     5.4e-6        8.3e-6
 240h  0σ    27,759     1.60         −2.19     5.7e-4   1.6e-3     1.06e-3       1.76e-3
       1σ     2,102     2.31         −3.01     3.8e-5   1.2e-4     6.7e-5        1.25e-4
       2σ       136     2.87         −3.57     2.1e-6   8.2e-6     3.8e-6        8.3e-6
```

`record_count` does not depend on lead time, because all leads use the same verification times.
Tail scores shrink with depth only because the events get rarer. To compare across depths,
normalise them, for example as a skill score against a reference model at the same depth.

## 6. Loading

```python
import xarray as xr
ds = xr.open_dataset("results/IFS_ENS/Depth/IFS_ENS_max_depth_2022_t2m_no_Antarctic_records_1979_2021_ERA5.nc")
ds.rtwcrps.sel(landsea="land", leadtime=120)          # rtwCRPS vs depth

# all models in one array; outer join because the depth lists can differ
import glob
files = sorted(glob.glob("results/*/Depth/*_max_depth_2022_t2m_*_ERA5.nc"))
allm = xr.concat([xr.open_dataset(f).expand_dims(model=[xr.open_dataset(f).attrs["model"]])
                  for f in files], dim="model", join="outer")
```

## 7. Caveats

* **Different time sample from `CRPS/`.** Depth files use the 710 valid times shared by all models,
  while `CRPS/` uses about 733. Their `crps_all` values therefore differ slightly. Don't mix the two.
* **Lead times differ.** GenCast-mini has a 12 h lead; the others do not. ArchesWeatherGen has no 12 h forecasts.
* **`comparison_models` is incomplete for GenCast-mini.** It only lists GenCast-mini itself, but its
  710 verification times are identical to those of the other models.
* **Check the fingerprints.** `reference_fingerprint` should be the same for a given (var, case) across
  all models. The central copy is in `depth_fingerprints.json`, and the script refuses to write if the
  reference data changed.
* **Relabelled on 2026-10-14.** The `.nc` files written before this date had wrong variable attributes
  (for example, `crps_empirical_*` was labelled "fair") and an inconsistent dimension order for
  `record_weight` / `observed_probability`. Both were fixed in place without changing any values.
  The script now writes the correct labels directly.
* **GenCast-mini repaired on 2026-10-14.** Its depths were filled in afterwards
  (with a one-off fill script, since removed), and the merge stacked the old and new runs along a
  spurious `index` dimension, leaving each depth in only one of two layers (int fill
  −9.2e18 in the counts). The layers were collapsed and the result checked against the
  pre-merge files, which have since been deleted. The main CSVs were regenerated, because the merged
  ones were missing the 3.0σ rows.
