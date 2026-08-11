"""Shared constants and result-loading helpers for manuscript figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


MODEL_ORDER = ["IFS_ENS", "uCast", "FGN", "ArchesWeatherGen"]

MODEL_LABELS = {
    "IFS_ENS": "IFS-ENS",
    "uCast": "U-Cast",
    "FGN": "FGN",
    "ArchesWeatherGen": "ArchesWeatherGen",
}

MODEL_COLORS = {
    "IFS_ENS": "#2a78d6",
    "uCast": "#eb6834",
    "FGN": "#1baf7a",
    "ArchesWeatherGen": "#eda100",
}

SCORE_DIRS = {
    "point": "Point",
    "CRPS": "CRPS",
    "occurrence": "Occurrence",
    "margin": "Margin",
}

EVENTS = (
    {"title": "Heat records", "case": "max", "var": "t2m", "unit": "deg C"},
    {"title": "Cold records", "case": "min", "var": "t2m", "unit": "deg C"},
    {"title": "Wind records", "case": "max", "var": "w10", "unit": "m/s"},
)
EVENT_DEFINITIONS = EVENTS


def find_result_file(
    results_root: Path,
    model: str,
    metric: str,
    case: str,
    var: str,
    year: int,
    record_tag: str = "1979_2018",
) -> Path | None:
    """Return the matching evaluation result, if it exists."""
    truth = "HRES_fc0" if model == "IFS_ENS" else "ERA5"
    folder = results_root / model / SCORE_DIRS[metric]
    pattern = (
        f"{model}_{case}_{metric}_{year}_{var}_no_Antarctic_records_{record_tag}_cond_{truth}.nc"
    )
    matches = sorted(folder.glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous result files matching {pattern}: {matches}")
    return matches[0] if matches else None


def leadtime_hours(values) -> np.ndarray:
    """Return lead times as floating-point hours."""
    values = np.asarray(values)
    if values.dtype.kind == "m":
        return (values / np.timedelta64(1, "h")).astype(float)
    if values.dtype.kind in "iu" and values.size and values.max() > 1e6:
        return values.astype(float) / 3.6e12
    return values.astype(float)


def _series(
    results_root: Path,
    model: str,
    metric: str,
    event: dict,
    variable: str,
    year: int,
    season: str,
    landsea: str,
    record_tag: str = "1979_2018",
) -> tuple[np.ndarray, np.ndarray] | None:
    path = find_result_file(
        results_root, model, metric, event["case"], event["var"], year, record_tag
    )
    if path is None:
        return None

    try:
        with xr.open_dataset(path) as dataset:
            if metric == "occurrence" and variable == "count_difference":
                if "expected_count" not in dataset or "observed_count" not in dataset:
                    return None
                values = dataset["expected_count"] - dataset["observed_count"]
            elif variable in dataset:
                values = dataset[variable]
            else:
                return None
            selected = values.sel(season=season, landsea=landsea).values
            leadtimes = leadtime_hours(dataset["leadtime"].values)
    except (IndexError, KeyError, OSError, TypeError, ValueError):
        return None
    selected = np.asarray(selected, dtype=float)
    if selected.ndim != 1 or not np.any(np.isfinite(selected)):
        return None
    return leadtimes, selected


def _plot_series(
    ax, results_root, metric, event, variable, year, season, landsea, linestyle="-", record_tag="1979_2018"
) -> int:
    series_count = 0
    for model in MODEL_ORDER:
        series = _series(
            results_root, model, metric, event, variable, year, season, landsea, record_tag
        )
        if series is None:
            continue
        hours, values = series
        ax.plot(
            hours,
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            linestyle=linestyle,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        series_count += 1
    return series_count


def _shared_legend(fig, axes) -> None:
    handles_by_label = {}
    for axis in np.ravel(axes):
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        fig.legend(
            handles_by_label.values(),
            handles_by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=len(handles_by_label),
            frameon=False,
        )


def _save(fig, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _require_series(fig, series_count: int, year: int, season: str, landsea: str) -> None:
    if series_count == 0:
        plt.close(fig)
        raise FileNotFoundError(
            f"No usable series for year={year}, season={season}, landsea={landsea}."
        )


def _occurrence_curve(
    results_root, model, event, year, lead, season, landsea, variables, record_tag="1979_2018"
) -> tuple[np.ndarray, ...] | None:
    try:
        path = find_result_file(
            results_root, model, "occurrence", event["case"], event["var"], year, record_tag
        )
        if path is None:
            return None
        with xr.open_dataset(path) as dataset:
            if "leadtime" not in dataset or any(variable not in dataset for variable in variables):
                return None
            matches = np.flatnonzero(np.isclose(leadtime_hours(dataset["leadtime"].values), lead))
            if len(matches) != 1:
                return None
            selected = [
                np.asarray(
                    dataset[variable]
                    .sel(season=season, landsea=landsea)
                    .isel(leadtime=matches[0])
                    .values,
                    dtype=float,
                )
                for variable in variables
            ]
    except (IndexError, KeyError, OSError, TypeError, ValueError):
        return None
    if any(values.ndim != 1 for values in selected) or len({values.shape for values in selected}) != 1:
        return None
    return tuple(selected)


def _margin_curve(
    results_root, model, event, variable, year, lead, landsea, record_tag="1979_2018"
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        path = find_result_file(
            results_root, model, "margin", event["case"], event["var"], year, record_tag
        )
        if path is None:
            return None
        with xr.open_dataset(path) as dataset:
            if "leadtime" not in dataset or "margin" not in dataset or variable not in dataset:
                return None
            matches = np.flatnonzero(np.isclose(leadtime_hours(dataset["leadtime"].values), lead))
            if len(matches) != 1:
                return None
            selected = dataset[variable].sel(landsea=landsea).isel(leadtime=matches[0])
            if selected.dims != ("margin",):
                return None
            margins = np.asarray(dataset["margin"].values, dtype=float)
            values = np.asarray(selected.values, dtype=float)
    except (IndexError, KeyError, OSError, TypeError, ValueError):
        return None
    if (
        margins.ndim != 1
        or values.ndim != 1
        or margins.shape != values.shape
        or not np.any(np.isfinite(values))
    ):
        return None
    return margins, values


def _plot_margin_metrics(results_root, output, year, lead, landsea, metrics, record_tag="1979_2018") -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False)
    series_count = 0
    for column, event in enumerate(EVENTS):
        unit = "K" if event["var"] == "t2m" else "m s^-1"
        for row, (variable, label, zero, physical_unit) in enumerate(metrics):
            axis = axes[row, column]
            for model in MODEL_ORDER:
                curve = _margin_curve(
                    results_root, model, event, variable, year, lead, landsea, record_tag
                )
                if curve is None:
                    continue
                margins, values = curve
                axis.plot(
                    margins,
                    values,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
                series_count += 1
            if zero:
                axis.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=0)
            if row == 0:
                axis.set_title(event["title"])
            axis.set_xlabel(f"Margin ({unit})")
            axis.set_ylabel(f"{label} ({unit})" if physical_unit else label)
    _require_series(fig, series_count, year, "all", landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)


def plot_margin_intensity(results_root, output, year, lead, landsea, record_tag="1979_2018") -> None:
    """Write margin-conditioned intensity diagnostics at one lead time."""
    _plot_margin_metrics(
        results_root,
        output,
        year,
        lead,
        landsea,
        (
            ("crps", "CRPS", False, True),
            ("rmse_mean", "Ensemble-mean RMSE", False, True),
            ("bias_mean", "Ensemble-mean bias", True, True),
        ),
        record_tag,
    )


def plot_margin_probabilistic(results_root, output, year, lead, landsea, record_tag="1979_2018") -> None:
    """Write margin-conditioned probabilistic diagnostics at one lead time."""
    _plot_margin_metrics(
        results_root,
        output,
        year,
        lead,
        landsea,
        (
            ("record_probability", "Record probability", False, False),
            ("out_of_ensemble_rate", "Out-of-ensemble rate", False, False),
            ("coverage_90", "90% coverage", False, False),
        ),
        record_tag,
    )


def plot_precision_recall(results_root, output, year, lead, season, landsea, record_tag="1979_2018") -> None:
    """Write precision-recall curves for record occurrence at one lead time."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    series_count = 0
    for axis, event in zip(axes, EVENTS):
        for model in MODEL_ORDER:
            curve = _occurrence_curve(
                results_root, model, event, year, lead, season, landsea, ("recall", "precision"), record_tag
            )
            if curve is None:
                continue
            recall, precision = curve
            valid = np.isfinite(recall) & np.isfinite(precision)
            if not np.any(valid):
                continue
            axis.plot(
                recall[valid], precision[valid], marker="o", markersize=4, linewidth=1.8,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            )
            series_count += 1
        axis.set_title(event["title"])
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    _require_series(fig, series_count, year, season, landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)


def plot_reliability(results_root, output, year, lead, season, landsea, record_tag="1979_2018") -> None:
    """Write reliability diagrams for record occurrence at one lead time."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    series_count = 0
    for axis, event in zip(axes, EVENTS):
        axis.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=0)
        for model in MODEL_ORDER:
            curve = _occurrence_curve(
                results_root,
                model,
                event,
                year,
                lead,
                season,
                landsea,
                ("reliability_predicted", "reliability_observed", "reliability_count"),
                record_tag,
            )
            if curve is None:
                continue
            predicted, observed, count = curve
            valid = (count > 0) & np.isfinite(predicted) & np.isfinite(observed)
            if not np.any(valid):
                continue
            axis.plot(
                predicted[valid], observed[valid], marker="o", markersize=4, linewidth=1.8,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            )
            series_count += 1
        axis.set_title(event["title"])
        axis.set_xlabel("Forecast probability")
        axis.set_ylabel("Observed frequency")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    _require_series(fig, series_count, year, season, landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)


def plot_point(results_root, output, year, season, landsea, record_tag="1979_2018") -> None:
    """Write a 2x3 ensemble-mean RMSE and bias figure at native lead times."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False)
    series_count = 0
    for column, event in enumerate(EVENTS):
        for row, (variable, label) in enumerate(
            (("rmse_mean", "Ensemble-mean RMSE"), ("bias_mean", "Ensemble-mean bias"))
        ):
            axis = axes[row, column]
            series_count += _plot_series(
                axis, results_root, "point", event, variable, year, season, landsea, record_tag=record_tag
            )
            if row == 1:
                axis.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=0)
            axis.set_xlabel("Lead time (h)")
            axis.set_ylabel(f"{label} ({event['unit']})")
            if row == 0:
                axis.set_title(event["title"])
    _require_series(fig, series_count, year, season, landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)


def _plot_event_metric(results_root, output, year, season, landsea, metric, variable, ylabel, zero=False, record_tag="1979_2018") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    series_count = 0
    for axis, event in zip(axes, EVENTS):
        series_count += _plot_series(
            axis, results_root, metric, event, variable, year, season, landsea, record_tag=record_tag
        )
        if zero:
            axis.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=0)
        axis.set_title(event["title"])
        axis.set_xlabel("Lead time (h)")
        axis.set_ylabel(ylabel)
    _require_series(fig, series_count, year, season, landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)


def plot_counts(results_root, output, year, season, landsea, record_tag="1979_2018") -> None:
    """Write expected-minus-observed record counts versus native lead time."""
    _plot_event_metric(
        results_root, output, year, season, landsea, "occurrence", "count_difference",
        "Expected - observed records", zero=True, record_tag=record_tag,
    )


def plot_brier(results_root, output, year, season, landsea, record_tag="1979_2018") -> None:
    """Write record-occurrence Brier scores versus native lead time."""
    _plot_event_metric(
        results_root, output, year, season, landsea, "occurrence", "brier_score", "Brier score", record_tag=record_tag
    )


def plot_crps(results_root, output, year, season, landsea, record_tag="1979_2018") -> None:
    """Write solid record-conditioned and dashed all-weather CRPS curves."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    series_count = 0
    for axis, event in zip(axes, EVENTS):
        series_count += _plot_series(
            axis, results_root, "CRPS", event, "CRPS", year, season, landsea, record_tag=record_tag
        )
        series_count += _plot_series(
            axis,
            results_root,
            "CRPS",
            {**event, "case": "all"},
            "CRPS",
            year,
            season,
            landsea,
            linestyle="--",
            record_tag=record_tag,
        )
        axis.set_title(event["title"])
        axis.set_xlabel("Lead time (h)")
        axis.set_ylabel("CRPS")
    _require_series(fig, series_count, year, season, landsea)
    _shared_legend(fig, axes)
    fig.tight_layout()
    _save(fig, output)
