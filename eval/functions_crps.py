"""
CRPS utilities for probabilistic ensemble forecasts.

This simplified version assumes the file structure used in our current workflow.

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
# Ensemble CRPS
# =============================================================================

def ensemble_crps_sorted(
    forecast: xr.DataArray,
    truth: xr.DataArray,
) -> xr.DataArray:
    """
    Empirical ensemble CRPS using the sorted-member formula.

    For ensemble members x_1,...,x_K and observation y,

        CRPS = 1/K sum_k |x_k - y|
               - 1/K^2 sum_k (2k - K - 1) x_(k),

    where x_(k) are the sorted ensemble members.
    """

    forecast, truth = xr.align(forecast, truth, join="inner")

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

        coeff = (2 * np.arange(1, m + 1, dtype=x.dtype) - m - 1) / (m**2)
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

    return crps


# =============================================================================
# Weighted CRPS summaries
# =============================================================================

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

    land_sea_mask = land_sea_mask.sel(
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
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute ensemble CRPS for one single-lead forecast file.

    Returns:
        CRPS with dimensions
            leadtime, season, landsea

        stat with dimensions
            leadtime, season, landsea, stat
    """

    stat_names = ["size", "mean", "variance"]

    # Each downloaded file contains one lead time.
    lead_hour = int(forecast["prediction_timedelta"].values[0])
    lead_delta = np.timedelta64(lead_hour, "h")

    print(f"    CRPS lead {lead_hour:03d}h")

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
        name="CRPS",
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

    mask_sub = functions.select_transformed_by_time(
        event_mask,
        forecast_sub["time"].values,
    )

    # Spatial alignment.
    truth_sub = truth_sub.sel(
        latitude=forecast_sub["latitude"],
        longitude=forecast_sub["longitude"],
        method="nearest",
    )

    mask_sub = mask_sub.sel(
        latitude=forecast_sub["latitude"],
        longitude=forecast_sub["longitude"],
        method="nearest",
    )

    forecast_sub, truth_sub, mask_sub = xr.align(
        forecast_sub,
        truth_sub,
        mask_sub,
        join="inner",
    )

    # Compute CRPS field lazily.
    crps_field = ensemble_crps_sorted(
        forecast_sub,
        truth_sub,
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

    return CRPS, stat