"""
Loading helpers for probex results/ NetCDF files.

Filenames encode model, case (all/max/min), score type, year, variable,
record baseline and verification target, e.g.:
    results/FGN/Point/FGN_max_point_2022_t2m_no_Antarctic_records_1979_2018_cond_ERA5.nc

IFS_ENS is verified against HRES_fc0; every other model against ERA5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

SCORE_DIRS = {
    "point": "Point",
    "CRPS": "CRPS",
    "occurrence": "Occurrence",
    "margin": "Margin",
}


def truth_for_model(model: str) -> str:
    return "HRES_fc0" if model == "IFS_ENS" else "ERA5"


def find_result_file(
    model: str,
    score_type: str,
    case: str,
    var: str,
    results_root: Path,
    year: int | None = None,
    record_tag: str = "*",
) -> Path | None:
    """Locate a single results file, or None if no run exists yet for this combo."""
    folder = results_root / model / SCORE_DIRS[score_type]
    if not folder.exists():
        return None

    truth = truth_for_model(model)
    year_token = str(year) if year is not None else "*"
    pattern = (
        f"{model}_{case}_{score_type}_{year_token}_{var}"
        f"_no_Antarctic_records_{record_tag}_cond_{truth}.nc"
    )
    matches = sorted(folder.glob(pattern))
    return matches[-1] if matches else None


def leadtime_hours(leadtime: xr.DataArray) -> np.ndarray:
    """Normalize the leadtime coordinate to hours (some files store timedelta64[ns], others raw hours)."""
    values = np.asarray(leadtime.values)
    if values.dtype.kind == "m":
        return values / np.timedelta64(1, "h")
    if values.dtype.kind in "iu" and values.max() > 1e6:
        return values / 3.6e12
    return values.astype(float)


def load_scalar_series(
    model: str,
    score_type: str,
    case: str,
    var: str,
    metric: str,
    results_root: Path,
    season: str = "all",
    landsea: str = "all",
    year: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Return (leadtime_hours, values) for one model/case/var/metric from a
    results file, or (None, None) if that model has no such run yet.
    """
    path = find_result_file(model, score_type, case, var, results_root, year=year)
    if path is None:
        return None, None

    ds = xr.open_dataset(path)
    hours = leadtime_hours(ds["leadtime"])
    values = ds[metric].sel(season=season, landsea=landsea).values
    return hours, values


def load_point_series(
    model: str,
    case: str,
    var: str,
    metric: str,
    results_root: Path,
    season: str = "all",
    landsea: str = "all",
    year: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (leadtime_hours, values) for one model/case/var/metric from a Point results file."""
    return load_scalar_series(
        model, "point", case, var, metric, results_root, season, landsea, year
    )


def load_occurrence_diff(
    model: str,
    case: str,
    var: str,
    results_root: Path,
    season: str = "all",
    landsea: str = "all",
    year: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Return (leadtime_hours, expected_count - observed_count) from an Occurrence
    results file. observed_count is the number of observed record-breaking
    grid-time points in this case/var; expected_count is the ensemble's
    predicted-probability mass summed over those same points. Zero means the
    ensemble's predicted record probability was, on average, calibrated to
    what actually occurred.
    """
    path = find_result_file(model, "occurrence", case, var, results_root, year=year)
    if path is None:
        return None, None

    ds = xr.open_dataset(path)
    hours = leadtime_hours(ds["leadtime"])
    diff = (
        (ds["expected_count"] - ds["observed_count"])
        .sel(season=season, landsea=landsea)
        .values
    )
    return hours, diff
