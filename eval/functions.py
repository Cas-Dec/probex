from __future__ import annotations

from datetime import datetime

import dask
import numpy as np
import pandas as pd
import xarray as xr


dask.config.set({"array.slicing.split_large_chunks": True})

SEASON_NAMES = ["DJF", "MAM", "JJA", "SON", "all"]

SEASON_MONTHS = [
    [12, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
]

LANDSEA_NAMES = ["all", "land", "sea"]


def _dedupe_dim(obj: xr.DataArray | xr.Dataset, dim: str):
    """
    Ensure a dimension has unique coordinate labels.

    If duplicates exist, keep the first occurrence for each coordinate value.
    """

    if dim not in obj.dims:
        return obj

    coord = np.asarray(obj[dim].values)

    if coord.ndim != 1:
        return obj

    _, first_idx = np.unique(coord, return_index=True)

    if len(first_idx) != coord.size:
        first_idx = np.sort(first_idx)
        obj = obj.isel({dim: first_idx})

    return obj


def normalize_spatial_coords(obj: xr.DataArray | xr.Dataset):
    """
    Normalize latitude/longitude dimensions for robust alignment.

    - deduplicate latitude/longitude coordinate labels
    - sort latitude and longitude in ascending order
    """

    for dim in ["latitude", "longitude"]:
        obj = _dedupe_dim(obj, dim)
        if dim in obj.dims:
            obj = obj.sortby(dim)

    return obj

def transform_ERA5_time(data: xr.DataArray) -> xr.DataArray:
    """
    Transform 
    time, latitude, longitude 
        into
    year, month, day, hour, latitude, longitude
    """
    year = data.time.dt.year
    month = data.time.dt.month
    day = data.time.dt.day
    hour = data.time.dt.hour

    data = data.assign_coords(
        coords={
            "year": ("time", year.values),
            "month": ("time", month.values),
            "day": ("time", day.values),
            "hour": ("time", hour.values),
        }
    )

    data = data.set_index(time=("year", "month", "day", "hour")).unstack("time")

    return data

def select_transformed_by_time(
    data: xr.DataArray,
    valid_times,
) -> xr.DataArray:
    """
    Select values from 
        year, month, day, hour, latitude, longitude
    array 
    and return as
        time, latitude, longitude
    where time is valid forecast time.
    """

    times = pd.DatetimeIndex(np.asarray(valid_times).astype("datetime64[ns]"))

    indexers = {
        "year": xr.DataArray(times.year, dims="time", coords={"time": times.values}),
        "month": xr.DataArray(times.month, dims="time", coords={"time": times.values}),
        "day": xr.DataArray(times.day, dims="time", coords={"time": times.values}),
        "hour": xr.DataArray(times.hour, dims="time", coords={"time": times.values}),
    }

    out = data.sel(indexers)
    out = out.assign_coords(time=("time", times.values))

    # Drop auxiliary coordinates created by vectorized indexing.
    for coord in ["year", "month", "day", "hour"]:
        if coord in out.coords:
            out = out.drop_vars(coord)

    return out

def latitude_weights(field: xr.DataArray) -> xr.DataArray:
    """
    Latitude weights proportional to grid-cell area.
    """

    weights = np.cos(np.deg2rad(field["latitude"]))

    weights = xr.DataArray(
        weights,
        coords={"latitude": field["latitude"]},
        dims=["latitude"],
    )

    return weights.astype("float32")


def select_months(field: xr.DataArray, months: list[int]) -> xr.DataArray:
    """
    Select calendar months from a field with a normal time coordinate.
    """

    return field.where(field["time"].dt.month.isin(months), drop=True)


def scalar(x) -> float:
    """
    Convert a scalar xarray/dask result to a Python float.
    """

    return float(np.asarray(x))

def weighted_rmse_bias(
    point_forecast: xr.DataArray,
    truth: xr.DataArray,
    event_mask: xr.DataArray,
    land_sea_mask: xr.DataArray,
    landsea: str,
):
    """
    Weighted count, RMSE and bias for a deterministic point forecast.

    landsea is one of:
        all, land, sea
    """

    point_forecast, truth, event_mask = xr.align(
        point_forecast,
        truth,
        event_mask,
        join="inner",
    )

    error = (point_forecast - truth).astype("float32")
    event_mask = event_mask.astype(bool)

    land_sea_mask = land_sea_mask.reindex(
        latitude=error["latitude"],
        longitude=error["longitude"],
        method="nearest",
    )

    valid = event_mask & point_forecast.notnull() & truth.notnull()

    if landsea == "land":
        valid = valid & (land_sea_mask > 0.5)

    if landsea == "sea":
        valid = valid & (land_sea_mask <= 0.5)

    weights = latitude_weights(error)

    dims = list(error.dims)

    n = valid.sum(dim=dims)

    denom = weights.where(valid).sum(dim=dims)

    bias = (error * weights).where(valid).sum(dim=dims) / denom

    mse = ((error ** 2) * weights).where(valid).sum(dim=dims) / denom
    rmse = np.sqrt(mse)

    return n, rmse, bias


def empty_point_result(lead_hour: int) -> xr.Dataset:
    """
    Empty point-forecast result dataset for one lead time.
    """
    coords = {
        "leadtime": [lead_hour],
        "season": SEASON_NAMES,
        "landsea": LANDSEA_NAMES,
    }

    dims = ("leadtime", "season", "landsea")
    shape = (1, len(SEASON_NAMES), len(LANDSEA_NAMES))

    data_vars = {
        "rmse_mean": xr.DataArray(np.full(shape, np.nan), coords=coords, dims=dims),
        "bias_mean": xr.DataArray(np.full(shape, np.nan), coords=coords, dims=dims),
        "rmse_median": xr.DataArray(np.full(shape, np.nan), coords=coords, dims=dims),
        "bias_median": xr.DataArray(np.full(shape, np.nan), coords=coords, dims=dims),
        "n": xr.DataArray(np.full(shape, np.nan), coords=coords, dims=dims),
    }

    return xr.Dataset(data_vars)


def evaluate_point_one_lead(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    masks: dict[str, xr.DataArray],
    land_sea_mask: xr.DataArray,
    eval_start: datetime,
    eval_end: datetime,
) -> dict[str, xr.Dataset]:
    """
    Evaluate a single-lead ensemble forecast file.

    Returns one dataset per case:
        max, all, and possibly min.
    The output datasets contain rmse_mean, bias_mean,rmse_median,bias_median.
    """

    # Each downloaded file contains exactly one lead.
    # Handle both int64 (IFS_ENS) and timedelta64 (FGN) types
    td = forecast["prediction_timedelta"].values[0]
    if isinstance(td, np.timedelta64):
        lead_hour = int(td / np.timedelta64(1, "h"))
    else:
        lead_hour = int(td)
    lead_delta = np.timedelta64(lead_hour, "h")

    print(f"    Point scores lead {lead_hour:03d}h")

    # Select the single lead.
    fc = forecast.isel(prediction_timedelta=0, drop=True)
    fc = normalize_spatial_coords(fc)

    # Convert initialization time to valid time.
    valid_time = fc["time"].values.astype("datetime64[ns]") + lead_delta
    fc = fc.assign_coords(time=valid_time)

    # Keep only valid times in the verification year.
    fc = fc.sel(time=slice(eval_start, eval_end))

    if fc.sizes["time"] == 0:
        return {case: empty_point_result(lead_hour) for case in masks}

    # Select matching truth.
    truth_sub = select_transformed_by_time(truth, fc["time"].values)
    truth_sub = normalize_spatial_coords(truth_sub)

    truth_sub = truth_sub.reindex(
        latitude=fc["latitude"],
        longitude=fc["longitude"],
        method="nearest",
    )

    # Select matching masks.
    mask_sub = {}

    for case, mask in masks.items():
        tmp = select_transformed_by_time(mask, fc["time"].values)
        tmp = normalize_spatial_coords(tmp)

        tmp = tmp.reindex(
            latitude=fc["latitude"],
            longitude=fc["longitude"],
            method="nearest",
        )

        mask_sub[case] = tmp

    fc, truth_sub = xr.align(fc, truth_sub, join="inner")

    for case in mask_sub:
        mask_sub[case] = mask_sub[case].sel(time=fc["time"])
    fc = fc.chunk(
        {
            "time": 31,
            "number": -1,
            "latitude": 30,
            "longitude": 60,
        }
    )

    truth_sub = truth_sub.chunk(
        {
            "time": 31,
            "latitude": 30,
            "longitude": 60,
        }
    )

    # Ensemble mean and median point forecasts.
    mean_fc = fc.mean("number", skipna=True)
    median_fc = fc.median("number", skipna=True)

    outputs = {case: empty_point_result(lead_hour) for case in masks}
    scalar_tasks = []
    scalar_keys = []

    for case, event_mask in mask_sub.items():
        for season_name, months in zip(SEASON_NAMES, SEASON_MONTHS):

            mean_season = select_months(mean_fc, months)
            median_season = select_months(median_fc, months)
            truth_season = select_months(truth_sub, months)
            mask_season = select_months(event_mask, months)

            for landsea in LANDSEA_NAMES:

                n, rmse_mean, bias_mean = weighted_rmse_bias(
                    mean_season,
                    truth_season,
                    mask_season,
                    land_sea_mask,
                    landsea,
                )

                _, rmse_median, bias_median = weighted_rmse_bias(
                    median_season,
                    truth_season,
                    mask_season,
                    land_sea_mask,
                    landsea,
                )

                scalar_tasks.extend([n, rmse_mean, bias_mean, rmse_median, bias_median])
                scalar_keys.append((case, season_name, landsea))

    # Share forecast reads and ensemble reductions across cases, seasons and
    # surfaces. Computing inside the loops rereads the same forecast chunks.
    computed = dask.compute(*scalar_tasks)
    names = ["n", "rmse_mean", "bias_mean", "rmse_median", "bias_median"]
    for index, (case, season_name, landsea) in enumerate(scalar_keys):
        loc = dict(leadtime=lead_hour, season=season_name, landsea=landsea)
        for offset, name in enumerate(names):
            outputs[case][name].loc[loc] = scalar(computed[5 * index + offset])

    return outputs
