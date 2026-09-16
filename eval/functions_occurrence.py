"""
Occurrence-evaluation helper functions.

This file contains the reusable logic for probabilistic record occurrence:

    y = observed record indicator
    p = ensemble probability of record occurrence

Expected coordinates:
    latitude, longitude

Expected forecast dimensions:
    time, number, prediction_timedelta, latitude, longitude

Expected truth dimensions:
    time, latitude, longitude

Expected record dimensions:
    month, latitude, longitude
"""

from __future__ import annotations

from datetime import datetime

import dask
import numpy as np
import pandas as pd
import xarray as xr

import functions


dask.config.set({"array.slicing.split_large_chunks": True})


# =============================================================================
# Settings for reliability and precision-recall
# =============================================================================

RELIABILITY_BINS = np.array(
    [0.0, 0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.000001]
)

RELIABILITY_BIN_CENTERS = 0.5 * (RELIABILITY_BINS[:-1] + RELIABILITY_BINS[1:])

PR_THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.02), 4)


# =============================================================================
# Monthly records at valid times
# =============================================================================

def record_by_time(record: xr.DataArray, valid_times) -> xr.DataArray:
    """
    Select monthly record values matching valid forecast times.

    Input:
        record with dimensions
            month, latitude, longitude

    Output:
        record with dimensions
            time, latitude, longitude
    """

    times = pd.DatetimeIndex(np.asarray(valid_times).astype("datetime64[ns]"))

    month_indexer = xr.DataArray(
        times.month,
        dims="time",
        coords={"time": times.values},
    )

    out = record.sel(month=month_indexer)
    out = out.assign_coords(time=("time", times.values))

    if "month" in out.coords:
        out = out.drop_vars("month")

    return out


# =============================================================================
# Scalar occurrence summaries
# =============================================================================

def weighted_occurrence_stats(
    p: xr.DataArray,
    y: xr.DataArray,
    member_count: xr.DataArray,
    land_sea_mask: xr.DataArray,
    landsea: str,
):
    """
    Weighted occurrence summaries for one season and one land/sea class.

    Returns:
        observed_count
        expected_count
        brier_score (fair finite-ensemble estimator)
    """

    p, y, member_count = xr.align(p, y, member_count, join="inner")

    p = p.astype("float32")
    y = y.astype("float32")
    member_count = member_count.astype("float32")

    land_sea_mask = land_sea_mask.reindex(
        latitude=p["latitude"],
        longitude=p["longitude"],
        method="nearest",
    )

    valid = p.notnull() & y.notnull() & (member_count >= 2)

    if landsea == "land":
        valid = valid & (land_sea_mask > 0.5)

    if landsea == "sea":
        valid = valid & (land_sea_mask <= 0.5)

    weights = functions.latitude_weights(p)

    dims = list(p.dims)

    observed_count = y.where(valid).sum(dim=dims)
    expected_count = p.where(valid).sum(dim=dims)

    denom = weights.where(valid).sum(dim=dims)

    # Ferro's finite-ensemble correction. The output name is deliberately
    # unchanged so existing plotting code continues to work.
    brier_field = (
        (p - y) ** 2
        - p * (1.0 - p) / (member_count - 1.0)
    )
    brier_score = (brier_field * weights).where(valid).sum(dim=dims) / denom

    return observed_count, expected_count, brier_score


# =============================================================================
# Reliability diagram
# =============================================================================

def reliability_stats(
    p: xr.DataArray,
    y: xr.DataArray,
    land_sea_mask: xr.DataArray,
    landsea: str,
):
    """
    Reliability diagram values.

    Returns:
        reliability_count
        reliability_predicted
        reliability_observed
    """

    p, y = xr.align(p, y, join="inner")

    p = p.astype("float32")
    y = y.astype("float32")

    land_sea_mask = land_sea_mask.reindex(
        latitude=p["latitude"],
        longitude=p["longitude"],
        method="nearest",
    )

    valid = p.notnull() & y.notnull()

    if landsea == "land":
        valid = valid & (land_sea_mask > 0.5)

    if landsea == "sea":
        valid = valid & (land_sea_mask <= 0.5)

    weights = functions.latitude_weights(p)
    dims = list(p.dims)

    counts = []
    predicted = []
    observed = []

    for left, right in zip(RELIABILITY_BINS[:-1], RELIABILITY_BINS[1:]):

        in_bin = valid & (p >= left) & (p < right)

        count = in_bin.sum(dim=dims)

        wsum = weights.where(in_bin).sum(dim=dims)

        pmean = (p * weights).where(in_bin).sum(dim=dims) / wsum
        ymean = (y * weights).where(in_bin).sum(dim=dims) / wsum

        counts.append(count)
        predicted.append(pmean)
        observed.append(ymean)

    count = xr.concat(counts, dim="reliability_bin")
    pred = xr.concat(predicted, dim="reliability_bin")
    obs = xr.concat(observed, dim="reliability_bin")

    reliability_bin = np.arange(len(RELIABILITY_BIN_CENTERS))

    count = count.assign_coords(reliability_bin=reliability_bin)
    pred = pred.assign_coords(reliability_bin=reliability_bin)
    obs = obs.assign_coords(reliability_bin=reliability_bin)

    return count, pred, obs


# =============================================================================
# Precision-recall
# =============================================================================

def precision_recall_stats(
    p: xr.DataArray,
    y: xr.DataArray,
    land_sea_mask: xr.DataArray,
    landsea: str,
):
    """
    Precision-recall curve for probability thresholds.
    """

    p, y = xr.align(p, y, join="inner")

    p = p.astype("float32")

    land_sea_mask = land_sea_mask.reindex(
        latitude=p["latitude"],
        longitude=p["longitude"],
        method="nearest",
    )

    valid = p.notnull() & y.notnull()
    y = y.astype(bool)

    if landsea == "land":
        valid = valid & (land_sea_mask > 0.5)

    if landsea == "sea":
        valid = valid & (land_sea_mask <= 0.5)

    dims = list(p.dims)

    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in PR_THRESHOLDS:

        pred = p > threshold

        tp = (valid & pred & y).sum(dim=dims)
        fp = (valid & pred & (~y)).sum(dim=dims)
        fn = (valid & (~pred) & y).sum(dim=dims)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2.0 * precision * recall / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = xr.concat(precision_list, dim="threshold")
    recall = xr.concat(recall_list, dim="threshold")
    f1 = xr.concat(f1_list, dim="threshold")

    precision = precision.assign_coords(threshold=PR_THRESHOLDS)
    recall = recall.assign_coords(threshold=PR_THRESHOLDS)
    f1 = f1.assign_coords(threshold=PR_THRESHOLDS)

    return precision, recall, f1


# =============================================================================
# Empty output
# =============================================================================

def empty_occurrence_result(lead_hour: int) -> xr.Dataset:
    """
    Empty result dataset for one lead time.
    """

    coords = {
        "leadtime": [lead_hour],
        "season": functions.SEASON_NAMES,
        "landsea": functions.LANDSEA_NAMES,
        "reliability_bin": np.arange(len(RELIABILITY_BIN_CENTERS)),
        "threshold": PR_THRESHOLDS,
    }

    scalar_shape = (
        1,
        len(functions.SEASON_NAMES),
        len(functions.LANDSEA_NAMES),
    )

    rel_shape = scalar_shape + (len(RELIABILITY_BIN_CENTERS),)
    pr_shape = scalar_shape + (len(PR_THRESHOLDS),)

    ds = xr.Dataset(
        data_vars={
            "observed_count": (
                ("leadtime", "season", "landsea"),
                np.full(scalar_shape, np.nan),
            ),
            "expected_count": (
                ("leadtime", "season", "landsea"),
                np.full(scalar_shape, np.nan),
            ),
            "count_ratio": (
                ("leadtime", "season", "landsea"),
                np.full(scalar_shape, np.nan),
            ),
            "brier_score": (
                ("leadtime", "season", "landsea"),
                np.full(scalar_shape, np.nan),
            ),
            "reliability_count": (
                ("leadtime", "season", "landsea", "reliability_bin"),
                np.full(rel_shape, np.nan),
            ),
            "reliability_predicted": (
                ("leadtime", "season", "landsea", "reliability_bin"),
                np.full(rel_shape, np.nan),
            ),
            "reliability_observed": (
                ("leadtime", "season", "landsea", "reliability_bin"),
                np.full(rel_shape, np.nan),
            ),
            "precision": (
                ("leadtime", "season", "landsea", "threshold"),
                np.full(pr_shape, np.nan),
            ),
            "recall": (
                ("leadtime", "season", "landsea", "threshold"),
                np.full(pr_shape, np.nan),
            ),
            "f1": (
                ("leadtime", "season", "landsea", "threshold"),
                np.full(pr_shape, np.nan),
            ),
        },
        coords=coords,
    )

    ds = ds.assign_coords(
        reliability_bin_center=("reliability_bin", RELIABILITY_BIN_CENTERS),
        reliability_bin_left=("reliability_bin", RELIABILITY_BINS[:-1]),
        reliability_bin_right=("reliability_bin", RELIABILITY_BINS[1:]),
    )

    ds["brier_score"].attrs["estimator"] = "fair ensemble Brier score"
    ds["brier_score"].attrs["reference"] = (
        "Ferro (2014), doi:10.1002/qj.2270"
    )

    return ds


# =============================================================================
# Main one-lead occurrence evaluation
# =============================================================================

def evaluate_occurrence_one_lead(
    forecast: xr.DataArray,
    truth: xr.DataArray,
    record: xr.DataArray,
    land_sea_mask: xr.DataArray,
    tail: str,
    eval_start: datetime,
    eval_end: datetime,
) -> xr.Dataset:
    """
    Evaluate probabilistic record occurrence for one single-lead forecast file.

    tail:
        max = high records
        min = low records
    """

    # Handle both int64 (IFS_ENS) and timedelta64 (FGN) types
    td = forecast["prediction_timedelta"].values[0]
    if isinstance(td, np.timedelta64):
        lead_hour = int(td / np.timedelta64(1, "h"))
    else:
        lead_hour = int(td)
    lead_delta = np.timedelta64(lead_hour, "h")

    print(f"    Occurrence lead {lead_hour:03d}h")

    ds = empty_occurrence_result(lead_hour)

    # Select single lead.
    fc = forecast.isel(prediction_timedelta=0, drop=True)
    fc = functions.normalize_spatial_coords(fc)

    # Convert initialization time to valid time.
    valid_time = fc["time"].values.astype("datetime64[ns]") + lead_delta
    fc = fc.assign_coords(time=valid_time)

    # Keep only valid times in verification year.
    fc = fc.sel(time=slice(eval_start, eval_end))

    if fc.sizes["time"] == 0:
        return ds

    # Select truth and monthly records at these valid times.
    truth_sub = truth.sel(time=fc["time"])
    record_sub = record_by_time(record, fc["time"].values)
    truth_sub = functions.normalize_spatial_coords(truth_sub)
    record_sub = functions.normalize_spatial_coords(record_sub)

    truth_sub = truth_sub.reindex(
        latitude=fc["latitude"],
        longitude=fc["longitude"],
        method="nearest",
    )

    record_sub = record_sub.reindex(
        latitude=fc["latitude"],
        longitude=fc["longitude"],
        method="nearest",
    )

    fc, truth_sub, record_sub = xr.align(
        fc,
        truth_sub,
        record_sub,
        join="inner",
    )

    # Conservative chunks.
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

    record_sub = record_sub.chunk(
        {
            "time": 31,
            "latitude": 30,
            "longitude": 60,
        }
    )

    valid_record = record_sub.notnull()
    member_count = fc.notnull().sum("number")

    if tail == "max":
        y = (truth_sub > record_sub).where(valid_record & truth_sub.notnull())
        p = (forecast_members_exceed(fc, record_sub, "max")).where(valid_record)

    elif tail == "min":
        y = (truth_sub < record_sub).where(valid_record & truth_sub.notnull())
        p = (forecast_members_exceed(fc, record_sub, "min")).where(valid_record)

    else:
        raise ValueError("tail must be 'max' or 'min'.")

    # Loop over seasons and land/sea classes.
    for j, months in enumerate(functions.SEASON_MONTHS):

        p_season = functions.select_months(p, months)
        y_season = functions.select_months(y, months)
        member_count_season = functions.select_months(member_count, months)

        for k, landsea in enumerate(functions.LANDSEA_NAMES):

            observed_count, expected_count, brier_score = weighted_occurrence_stats(
                p_season,
                y_season,
                member_count_season,
                land_sea_mask,
                landsea,
            )

            rel_count, rel_pred, rel_obs = reliability_stats(
                p_season,
                y_season,
                land_sea_mask,
                landsea,
            )

            precision, recall, f1 = precision_recall_stats(
                p_season,
                y_season,
                land_sea_mask,
                landsea,
            )

            computed = dask.compute(
                observed_count,
                expected_count,
                brier_score,
                rel_count,
                rel_pred,
                rel_obs,
                precision,
                recall,
                f1,
            )

            obs_val = functions.scalar(computed[0])
            exp_val = functions.scalar(computed[1])
            bs_val = functions.scalar(computed[2])

            if obs_val > 0:
                ratio_val = exp_val / obs_val
            else:
                ratio_val = np.nan

            loc = dict(
                leadtime=lead_hour,
                season=functions.SEASON_NAMES[j],
                landsea=landsea,
            )

            ds["observed_count"].loc[loc] = obs_val
            ds["expected_count"].loc[loc] = exp_val
            ds["count_ratio"].loc[loc] = ratio_val
            ds["brier_score"].loc[loc] = bs_val

            ds["reliability_count"].loc[loc] = computed[3].values
            ds["reliability_predicted"].loc[loc] = computed[4].values
            ds["reliability_observed"].loc[loc] = computed[5].values

            ds["precision"].loc[loc] = computed[6].values
            ds["recall"].loc[loc] = computed[7].values
            ds["f1"].loc[loc] = computed[8].values

    return ds


def forecast_members_exceed(
    forecast: xr.DataArray,
    record: xr.DataArray,
    tail: str,
) -> xr.DataArray:
    """
    Convert ensemble forecasts to probabilities of record occurrence.

    Output:
        time, latitude, longitude
    """

    finite_members = forecast.notnull()

    if tail == "max":
        events = (forecast > record) & finite_members

    elif tail == "min":
        events = (forecast < record) & finite_members

    else:
        raise ValueError("tail must be 'max' or 'min'.")

    numerator = events.sum("number")
    denominator = finite_members.sum("number")

    p = numerator / denominator
    p = p.where(denominator > 0)

    return p
