"""
CRPS utilities for probabilistic ensemble forecasts.

Expected forecast dimensions:
    time, number, prediction_timedelta, latitude, longitude

Expected truth dimensions after functions.transform_ERA5_time(..., time="month"):
    year, month, day, hour, latitude, longitude

Expected event-mask dimensions:
    year, month, day, hour, latitude, longitude

Expected land-sea-mask dimensions:
    latitude, longitude

Important:
    We do not transform the ensemble forecast with transform_ERA5_time().
    Instead, we convert forecast initialization time to valid time and select
    matching truth/mask values from the already-transformed ERA5 arrays.
"""

from __future__ import annotations

from datetime import datetime

import dask
import numpy as np
import xarray as xr

import functions


dask.config.set({"array.slicing.split_large_chunks": True})

# =============================================================================
# Fair ensemble CRPS
# =============================================================================

def ensemble_crps_sorted(
    forecast: xr.DataArray,
    truth: xr.DataArray,
) -> xr.DataArray:
    """
    FAIR ensemble CRPS using the sorted-member formula.

    For ensemble members x_1,...,x_M and observation y,
        fair CRPS = 1/M sum_m |x_m - y|- 1/[M(M-1)] sum_m (2m - M - 1) x_(m),
    where x_(m) are the sorted ensemble members. See Ferro (2014)
    """

    forecast, truth = xr.align(forecast, truth, join="inner")

    if forecast.sizes["number"] < 2:
        raise ValueError("Fair CRPS requires at least two ensemble members.")

    # Keep all ensemble members in one chunk because we sort over this dimension.
    forecast = forecast.chunk(
        {
            "time": 31,
            "number": -1,
            "latitude": 30,
            "longitude": 60,
        }
    )

    truth = truth.chunk(
        {
            "time": 31,
            "latitude": 30,
            "longitude": 60,
        }
    )

    def crps_core(x, y):
        # x shape: (..., number)
        # y shape: (...)
        x = np.sort(x, axis=-1)
        m = x.shape[-1]

        term1 = np.mean(np.abs(x - np.expand_dims(y, axis=-1)), axis=-1)

        coeff = (
            2 * np.arange(1, m + 1, dtype=x.dtype) - m - 1
        ) / (m * (m - 1))
        term2 = np.sum(x * coeff, axis=-1)

        return (term1 - term2).astype(np.float32)

    crps = xr.apply_ufunc(
        crps_core,
        forecast,
        truth,
        input_core_dims=[["number"], []],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=False,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    crps.name = "CRPS"
    crps.attrs["estimator"] = "fair ensemble CRPS"
    crps.attrs["reference"] = "Ferro (2014), doi:10.1002/qj.2270"

    return crps

# =====================================
# Record-threshold weighted CRPS
# =====================================

def record_threshold_weighted_crps_sorted(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    record: xr.DataArray,
    tail: str,
) -> xr.DataArray:
    """
    (fair) Threshold-weighted CRPS for a record threshold.
    For upper records (heat and wind),v_r(z) = max(z, r).
    whereas for lower records (cold), use v_r(z) = min(z, r).
    """

    if tail not in {"max", "min"}:
        raise ValueError("tail must be 'max' or 'min'.")

    forecast, truth, record = xr.align(
        forecast,
        truth,
        record,
        join="inner",
    )

    if tail == "max":
        forecast_transformed = xr.where(forecast > record, forecast, record)
        truth_transformed = xr.where(truth > record, truth, record)
    else:
        forecast_transformed = xr.where(forecast < record, forecast, record)
        truth_transformed = xr.where(truth < record, truth, record)

    # xr.where would otherwise replace missing values by the threshold.
    forecast_transformed = forecast_transformed.where(
        forecast.notnull() & record.notnull()
    )
    truth_transformed = truth_transformed.where(
        truth.notnull() & record.notnull()
    )

    rtwcrps = ensemble_crps_sorted(
        forecast_transformed,
        truth_transformed,
    )
    rtwcrps.name = "rtwCRPS"
    rtwcrps.attrs["estimator"] = "fair record-threshold weighted CRPS"
    rtwcrps.attrs["tail"] = tail
    rtwcrps.attrs["reference"] = (
        "Gneiting and Ranjan (2011), doi:10.1198/jbes.2010.08110; "
        "fair correction: Ferro (2014), doi:10.1002/qj.2270"
    )

    return rtwcrps

# ================================
# Weighted CRPS summaries
# ================================

def weighted_stats(
    field: xr.DataArray,
    event_mask: xr.DataArray,
    land_sea_mask: xr.DataArray,
    landsea: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute weighted size, mean and variance of a field.

    field is usually the CRPS field.
    event_mask selects the relevant verification points.

    landsea is one of:
        all, land, sea
    """

    field = field.astype("float32")
    event_mask = event_mask.astype(bool)

    land_sea_mask = land_sea_mask.reindex(
        latitude=field["latitude"],
        longitude=field["longitude"],
        method="nearest",
    )

    valid = event_mask & field.notnull()

    if landsea == "land":
        valid = valid & (land_sea_mask > 0.5)

    if landsea == "sea":
        valid = valid & (land_sea_mask <= 0.5)

    weights = functions.latitude_weights(field).astype("float32")

    dims = list(field.dims)

    size = valid.sum(dim=dims)

    denom = weights.where(valid).sum(dim=dims)

    mean = (field * weights).where(valid).sum(dim=dims) / denom

    variance = (((field - mean) ** 2) * weights).where(valid).sum(dim=dims) / denom

    return size, mean, variance

# =============================================================================
# Main CRPS function
# =============================================================================

def ENS_CRPS_landseasep(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    event_mask: xr.DataArray,
    land_sea_mask: xr.DataArray,
    eval_start: datetime,
    eval_end: datetime,
    record_threshold: xr.DataArray | None = None,
    tail: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute fair CRPS for one single-lead forecast file.

    If record_threshold and tail are supplied, compute fair rtwCRPS instead.
    Existing calls without those arguments are unchanged.

    Returns:
        CRPS with dimensions
            leadtime, season, landsea

        stat with dimensions
            leadtime, season, landsea, stat
    """

    stat_names = ["size", "mean", "variance"]

    # Each downloaded file contains one lead time.
    # Handle both int64 (IFS_ENS) and timedelta64 (FGN) types
    td = forecast["prediction_timedelta"].values[0]
    if isinstance(td, np.timedelta64):
        lead_hour = int(td / np.timedelta64(1, "h"))
    else:
        lead_hour = int(td)
    lead_delta = np.timedelta64(lead_hour, "h")

    if record_threshold is None:
        score_name = "CRPS"
        print(f"    fair CRPS lead {lead_hour:03d}h")
    else:
        if tail not in {"max", "min"}:
            raise ValueError(
                "tail must be 'max' or 'min' when record_threshold is supplied."
            )
        score_name = "rtwCRPS"
        print(f"    fair rtwCRPS ({tail}) lead {lead_hour:03d}h")

    CRPS = xr.DataArray(
        np.full(
            (1, len(functions.SEASON_NAMES), len(functions.LANDSEA_NAMES)),
            np.nan,
            dtype=np.float32,
        ),
        coords=dict(
            leadtime=[lead_hour],
            season=functions.SEASON_NAMES,
            landsea=functions.LANDSEA_NAMES,
        ),
        dims=["leadtime", "season", "landsea"],
        name=score_name,
    )

    stat = xr.DataArray(
        np.full(
            (
                1,
                len(functions.SEASON_NAMES),
                len(functions.LANDSEA_NAMES),
                len(stat_names),
            ),
            np.nan,
            dtype=np.float64,
        ),
        coords=dict(
            leadtime=[lead_hour],
            season=functions.SEASON_NAMES,
            landsea=functions.LANDSEA_NAMES,
            stat=stat_names,
        ),
        dims=["leadtime", "season", "landsea", "stat"],
        name="stat",
    )

    # Select the single lead.
    forecast_sub = forecast.isel(prediction_timedelta=0, drop=True)
    forecast_sub = functions.normalize_spatial_coords(forecast_sub)

    # Convert initialization time to valid time.
    valid_time = forecast_sub["time"].values.astype("datetime64[ns]") + lead_delta

    forecast_sub = forecast_sub.assign_coords(time=valid_time)
    forecast_sub = forecast_sub.sel(time=slice(eval_start, eval_end))

    if forecast_sub.sizes["time"] == 0:
        return CRPS, stat

    # Select matching truth and event-mask values by valid time.
    truth_sub = functions.select_transformed_by_time(
        truth,
        forecast_sub["time"].values,
    )
    truth_sub = functions.normalize_spatial_coords(truth_sub)

    mask_sub = functions.select_transformed_by_time(
        event_mask,
        forecast_sub["time"].values,
    )
    mask_sub = functions.normalize_spatial_coords(mask_sub)

    # Spatial alignment.
    truth_sub = truth_sub.reindex(
        latitude=forecast_sub["latitude"],
        longitude=forecast_sub["longitude"],
        method="nearest",
    )

    mask_sub = mask_sub.reindex(
        latitude=forecast_sub["latitude"],
        longitude=forecast_sub["longitude"],
        method="nearest",
    )

    if record_threshold is None:
        forecast_sub, truth_sub, mask_sub = xr.align(
            forecast_sub,
            truth_sub,
            mask_sub,
            join="inner",
        )

        # Compute fair CRPS field lazily.
        crps_field = ensemble_crps_sorted(
            forecast_sub,
            truth_sub,
        )
    else:
        month_indexer = xr.DataArray(
            forecast_sub["time"].dt.month.values,
            dims="time",
            coords={"time": forecast_sub["time"].values},
        )
        record_sub = record_threshold.sel(month=month_indexer)
        if "month" in record_sub.coords:
            record_sub = record_sub.drop_vars("month")
        record_sub = functions.normalize_spatial_coords(record_sub)
        record_sub = record_sub.reindex(
            latitude=forecast_sub["latitude"],
            longitude=forecast_sub["longitude"],
            method="nearest",
        )

        forecast_sub, truth_sub, mask_sub, record_sub = xr.align(
            forecast_sub,
            truth_sub,
            mask_sub,
            record_sub,
            join="inner",
        )

        # Compute fair rtwCRPS field lazily.
        crps_field = record_threshold_weighted_crps_sorted(
            forecast_sub,
            truth_sub,
            record_sub,
            tail=tail,
        )

    # Reduce the CRPS field to small scalar summaries.
    scalar_tasks = []
    scalar_keys = []

    for j, months in enumerate(functions.SEASON_MONTHS):

        crps_season = functions.select_months(crps_field, months)
        mask_season = functions.select_months(mask_sub, months)

        for k, landsea in enumerate(functions.LANDSEA_NAMES):

            size, mean, variance = weighted_stats(
                crps_season,
                mask_season,
                land_sea_mask,
                landsea,
            )

            scalar_tasks.extend([size, mean, variance])
            scalar_keys.append((0, j, k))

    computed = dask.compute(*scalar_tasks)

    pos = 0

    for i, j, k in scalar_keys:

        size_val = functions.scalar(computed[pos])
        mean_val = functions.scalar(computed[pos + 1])
        var_val = functions.scalar(computed[pos + 2])

        pos += 3

        CRPS[i, j, k] = mean_val
        stat[i, j, k, 0] = size_val
        stat[i, j, k, 1] = mean_val
        stat[i, j, k, 2] = var_val

    CRPS.attrs.update(crps_field.attrs)

    return CRPS, stat
