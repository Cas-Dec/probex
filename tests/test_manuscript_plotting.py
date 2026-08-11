from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from manuscript.scripts.plotting import (
    MODEL_ORDER,
    find_result_file,
    leadtime_hours,
    plot_brier,
    plot_counts,
    plot_crps,
    plot_margin_intensity,
    plot_margin_probabilistic,
    plot_point,
    plot_precision_recall,
    plot_reliability,
)


def write_point_result(
    results_root: Path, record_tag: str = "1979_2018", suffix: str = ""
) -> Path:
    path = (
        results_root
        / "IFS_ENS"
        / "Point"
        / f"IFS_ENS_max_point_2022_t2m_no_Antarctic_records_{record_tag}{suffix}_cond_HRES_fc0.nc"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(coords={"leadtime": np.array([np.timedelta64(12, "h"), np.timedelta64(48, "h")])}).to_netcdf(path)
    return path


def test_find_result_file_and_normalize_timedelta_leads(tmp_path: Path) -> None:
    expected_path = write_point_result(tmp_path)
    alternate_path = write_point_result(tmp_path, "1979_2017")

    assert find_result_file(tmp_path, "IFS_ENS", "point", "max", "t2m", 2022) == expected_path
    assert (
        find_result_file(
            tmp_path, "IFS_ENS", "point", "max", "t2m", 2022, record_tag="1979_2017"
        )
        == alternate_path
    )
    write_point_result(tmp_path, "1979_2016", "_run_a")
    write_point_result(tmp_path, "1979_2016", "_run_b")
    with pytest.raises(ValueError, match="Ambiguous result files"):
        find_result_file(
            tmp_path, "IFS_ENS", "point", "max", "t2m", 2022, record_tag="1979_2016*"
        )
    np.testing.assert_array_equal(
        leadtime_hours(np.array([np.timedelta64(12, "h"), np.timedelta64(48, "h")])),
        [12.0, 48.0],
    )
    np.testing.assert_array_equal(leadtime_hours(np.array([12, 48])), [12.0, 48.0])
    np.testing.assert_array_equal(
        leadtime_hours(np.array([43_200_000_000_000, 172_800_000_000_000])), [12.0, 48.0]
    )


def write_synthetic_results(results_root: Path) -> None:
    leads = np.array([np.timedelta64(12, "h"), np.timedelta64(48, "h")])
    coords = {
        "leadtime": leads,
        "season": ["all", "JJA"],
        "landsea": ["all", "land"],
        "threshold": [0.1, 0.5, 0.9],
        "reliability_bin": [0, 1, 2],
    }
    shape = (len(leads), 2, 2)
    curve_shape = (*shape, 3)
    margin_coords = {
        "leadtime": leads,
        "landsea": ["all", "land"],
        "margin": [0.0, 0.25, 1.25],
    }
    margin_shape = (len(leads), 2, 3)

    for model_index, model in enumerate(MODEL_ORDER):
        truth = "HRES_fc0" if model == "IFS_ENS" else "ERA5"
        for case, var in (("max", "t2m"), ("min", "t2m"), ("max", "w10")):
            precision = np.full(curve_shape, np.nan)
            recall = np.full(curve_shape, np.nan)
            predicted = np.full(curve_shape, np.nan)
            observed = np.full(curve_shape, np.nan)
            count = np.zeros(curve_shape)
            precision[0, 1, 1] = [0.9, np.nan, 0.3]
            recall[0, 1, 1] = [0.1, 0.5, np.inf]
            predicted[0, 1, 1] = [0.1, 0.5, 0.9]
            observed[0, 1, 1] = [0.2, np.nan, 0.8]
            count[0, 1, 1] = [3, 0, 2]
            point = xr.Dataset(
                {
                    "rmse_mean": (("leadtime", "season", "landsea"), np.full(shape, model_index + 1.0)),
                    "bias_mean": (("leadtime", "season", "landsea"), np.full(shape, model_index - 1.0)),
                },
                coords=coords,
            )
            occurrence = xr.Dataset(
                {
                    "expected_count": (("leadtime", "season", "landsea"), np.full(shape, model_index + 2.0)),
                    "observed_count": (("leadtime", "season", "landsea"), np.full(shape, 1.0)),
                    "brier_score": (("leadtime", "season", "landsea"), np.full(shape, 0.1 * (model_index + 1))),
                    "precision": (("leadtime", "season", "landsea", "threshold"), precision),
                    "recall": (("leadtime", "season", "landsea", "threshold"), recall),
                    "reliability_predicted": (("leadtime", "season", "landsea", "reliability_bin"), predicted),
                    "reliability_observed": (("leadtime", "season", "landsea", "reliability_bin"), observed),
                    "reliability_count": (("leadtime", "season", "landsea", "reliability_bin"), count),
                },
                coords=coords,
            )
            crps = xr.Dataset(
                {"CRPS": (("leadtime", "season", "landsea"), np.full(shape, 0.2 * (model_index + 1)))},
                coords=coords,
            )
            for metric, dataset in (("point", point), ("occurrence", occurrence), ("CRPS", crps)):
                path = (
                    results_root
                    / model
                    / {"point": "Point", "occurrence": "Occurrence", "CRPS": "CRPS"}[metric]
                    / f"{model}_{case}_{metric}_2022_{var}_no_Antarctic_records_1979_2018_cond_{truth}.nc"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                dataset.to_netcdf(path)
        for var in ("t2m", "w10"):
            all_weather = xr.Dataset(
                {"CRPS": (("leadtime", "season", "landsea"), np.full(shape, 9.0 + model_index))},
                coords=coords,
            )
            path = (
                results_root
                / model
                / "CRPS"
                / f"{model}_all_CRPS_2022_{var}_no_Antarctic_records_1979_2018_cond_{truth}.nc"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            all_weather.to_netcdf(path)
        for case, var in (("max", "t2m"), ("min", "t2m"), ("max", "w10")):
            margin = xr.Dataset(
                {
                    "crps": (("leadtime", "landsea", "margin"), np.full(margin_shape, model_index + 0.1)),
                    "rmse_mean": (("leadtime", "landsea", "margin"), np.full(margin_shape, model_index + 0.2)),
                    "bias_mean": (("leadtime", "landsea", "margin"), np.full(margin_shape, model_index - 0.3)),
                    "record_probability": (("leadtime", "landsea", "margin"), np.full(margin_shape, 0.1 * (model_index + 1))),
                    "out_of_ensemble_rate": (("leadtime", "landsea", "margin"), np.full(margin_shape, 0.2 * (model_index + 1))),
                    "coverage_90": (("leadtime", "landsea", "margin"), np.full(margin_shape, 0.9 - 0.1 * model_index)),
                },
                coords=margin_coords,
            )
            margin["crps"][0, 1] = [1.0 + model_index, np.nan, 3.0 + model_index]
            margin["record_probability"][0, 1] = [1.0 + model_index, np.nan, 3.0 + model_index]
            path = (
                results_root
                / model
                / "Margin"
                / f"{model}_{case}_margin_2022_{var}_no_Antarctic_records_1979_2018_cond_{truth}.nc"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            margin.to_netcdf(path)


@pytest.mark.parametrize(
    ("constructor", "name"),
    [(plot_point, "point"), (plot_counts, "counts"), (plot_brier, "brier"), (plot_crps, "crps")],
)
def test_plot_constructors_write_png(
    tmp_path: Path, constructor, name: str
) -> None:
    write_synthetic_results(tmp_path)
    output = tmp_path / f"{name}.png"

    constructor(tmp_path, output, 2022, "JJA", "land")

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_counts_uses_expected_minus_observed_values(tmp_path: Path, monkeypatch) -> None:
    write_synthetic_results(tmp_path)
    plotted_values = []
    original_plot = plt.Axes.plot

    def record_plot(axis, x, y, *args, **kwargs):
        plotted_values.append(np.asarray(y))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", record_plot)

    plot_counts(tmp_path, tmp_path / "counts.png", 2022, "JJA", "land")

    assert any(np.array_equal(values, [1.0, 1.0]) for values in plotted_values)


def test_plot_crps_draws_solid_record_and_dashed_all_weather_series(
    tmp_path: Path, monkeypatch
) -> None:
    write_synthetic_results(tmp_path)
    plotted = []
    original_plot = plt.Axes.plot

    def record_plot(axis, x, y, *args, **kwargs):
        plotted.append((np.asarray(y), kwargs.get("linestyle")))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", record_plot)

    plot_crps(tmp_path, tmp_path / "crps.png", 2022, "JJA", "land")

    assert sum(style == "-" for _, style in plotted) == 12
    assert sum(style == "--" for _, style in plotted) == 12
    assert any(np.array_equal(values, [9.0, 9.0]) and style == "--" for values, style in plotted)


@pytest.mark.parametrize("constructor", [plot_point, plot_counts, plot_brier, plot_crps])
def test_plot_constructors_reject_missing_data(tmp_path: Path, constructor) -> None:
    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, "JJA", "land")


@pytest.mark.parametrize(
    ("constructor", "pattern", "variables"),
    [
        (plot_point, "*/Point/*.nc", ("rmse_mean", "bias_mean")),
        (plot_counts, "*/Occurrence/*.nc", ("expected_count", "observed_count")),
        (plot_brier, "*/Occurrence/*.nc", ("brier_score",)),
        (plot_crps, "*/CRPS/*.nc", ("CRPS",)),
    ],
)
def test_scalar_plots_reject_all_nan_series_and_close_figure(
    tmp_path: Path, constructor, pattern: str, variables: tuple[str, ...]
) -> None:
    write_synthetic_results(tmp_path)
    for path in tmp_path.glob(pattern):
        def invalidate(dataset):
            for variable in variables:
                dataset[variable] = xr.full_like(dataset[variable], np.nan)
            return dataset

        rewrite_dataset(path, invalidate)
    figures_before = set(plt.get_fignums())

    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, "JJA", "land")

    assert set(plt.get_fignums()) == figures_before


@pytest.mark.parametrize(
    ("constructor", "name"),
    [(plot_precision_recall, "precision_recall"), (plot_reliability, "reliability")],
)
def test_occurrence_curve_plots_write_png(tmp_path: Path, constructor, name: str) -> None:
    write_synthetic_results(tmp_path)
    output = tmp_path / f"{name}.png"

    constructor(tmp_path, output, 2022, 12, "JJA", "land")

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("constructor", [plot_precision_recall, plot_reliability])
def test_occurrence_curve_plots_reject_missing_data(tmp_path: Path, constructor) -> None:
    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, 12, "JJA", "land")


def test_occurrence_curve_plots_filter_selected_values_and_style_models(
    tmp_path: Path, monkeypatch
) -> None:
    write_synthetic_results(tmp_path)
    plotted = []
    original_plot = plt.Axes.plot
    original_close = plt.close

    def record_plot(axis, x, y, *args, **kwargs):
        plotted.append((np.asarray(x), np.asarray(y), kwargs))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", record_plot)
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plot_precision_recall(tmp_path, tmp_path / "pr.png", 2022, 12, "JJA", "land")
    pr_figure = plt.gcf()
    pr_lines = [line for line in plotted if "label" in line[2]]
    assert len(pr_figure.axes) == 3
    assert len(pr_lines) == 12
    assert all(np.array_equal(x, [0.1]) and np.array_equal(y, [0.9]) for x, y, _ in pr_lines)
    assert {line[2]["label"] for line in pr_lines} == {"IFS-ENS", "U-Cast", "FGN", "ArchesWeatherGen"}
    assert all(line[2]["marker"] == "o" and line[2]["linewidth"] == 1.8 for line in pr_lines)
    original_close(pr_figure)

    plotted.clear()
    plot_reliability(tmp_path, tmp_path / "reliability.png", 2022, 12, "JJA", "land")
    reliability_figure = plt.gcf()
    reliability_lines = [line for line in plotted if "label" in line[2]]
    reference_lines = [line for line in plotted if "label" not in line[2]]
    assert len(reliability_figure.axes) == 3
    assert len(reliability_lines) == 12
    assert all(
        np.array_equal(x, [0.1, 0.9]) and np.array_equal(y, [0.2, 0.8])
        for x, y, _ in reliability_lines
    )
    assert len(reference_lines) == 3
    assert all(np.array_equal(x, [0, 1]) and np.array_equal(y, [0, 1]) for x, y, _ in reference_lines)
    assert all(axis.get_xlabel() == "Forecast probability" for axis in reliability_figure.axes)
    original_close(reliability_figure)


def rewrite_dataset(path: Path, transform) -> None:
    with xr.open_dataset(path) as dataset:
        transformed = transform(dataset).load()
    transformed.to_netcdf(path, mode="w")


@pytest.mark.parametrize(
    "transform",
    [
        lambda dataset: dataset.drop_vars(
            [
                "precision",
                "recall",
                "reliability_predicted",
                "reliability_observed",
                "reliability_count",
            ]
        ),
        lambda dataset: dataset.drop_vars(["season", "landsea"]),
        lambda dataset: dataset.drop_vars("leadtime"),
        lambda dataset: dataset.assign_coords(leadtime=[24, 48]),
        lambda dataset: dataset.assign_coords(leadtime=[12, 12]),
        lambda dataset: dataset.assign_coords(leadtime=["invalid", "lead"]),
        lambda dataset: dataset.drop_vars(["precision", "reliability_predicted"]).assign(
            precision=dataset["precision"].expand_dims(extra=[0]),
            reliability_predicted=dataset["reliability_predicted"].expand_dims(extra=[0]),
        ),
    ],
    ids=[
        "variables",
        "coordinates",
        "missing_lead",
        "unmatched_lead",
        "duplicate_lead",
        "incompatible_lead",
        "incompatible_curve",
    ],
)
@pytest.mark.parametrize("constructor", [plot_precision_recall, plot_reliability])
def test_occurrence_curve_plots_skip_malformed_model_event_result(
    tmp_path: Path, transform, constructor
) -> None:
    write_synthetic_results(tmp_path)
    path = (
        tmp_path
        / "IFS_ENS"
        / "Occurrence"
        / "IFS_ENS_max_occurrence_2022_t2m_no_Antarctic_records_1979_2018_cond_HRES_fc0.nc"
    )
    rewrite_dataset(path, transform)
    output = tmp_path / "curve.png"

    constructor(tmp_path, output, 2022, 12, "JJA", "land")

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("constructor", [plot_precision_recall, plot_reliability])
def test_occurrence_curve_plots_reject_all_invalid_values_and_close_figure(
    tmp_path: Path, constructor
) -> None:
    write_synthetic_results(tmp_path)
    variables = (
        ("precision", "recall")
        if constructor is plot_precision_recall
        else ("reliability_predicted", "reliability_observed", "reliability_count")
    )
    for path in tmp_path.glob("*/Occurrence/*.nc"):
        def invalidate(dataset):
            for variable in variables:
                fill_value = 0 if variable == "reliability_count" else np.nan
                dataset[variable] = xr.full_like(dataset[variable], fill_value)
            return dataset

        rewrite_dataset(path, invalidate)
    figures_before = set(plt.get_fignums())

    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, 12, "JJA", "land")

    assert set(plt.get_fignums()) == figures_before


@pytest.mark.parametrize(
    ("constructor", "name", "ylabels"),
    [
        (
            plot_margin_intensity,
            "margin_intensity",
            [
                "CRPS (K)", "CRPS (K)", "CRPS (m s^-1)",
                "Ensemble-mean RMSE (K)", "Ensemble-mean RMSE (K)", "Ensemble-mean RMSE (m s^-1)",
                "Ensemble-mean bias (K)", "Ensemble-mean bias (K)", "Ensemble-mean bias (m s^-1)",
            ],
        ),
        (
            plot_margin_probabilistic,
            "margin_probabilistic",
            [
                "Record probability", "Record probability", "Record probability",
                "Out-of-ensemble rate", "Out-of-ensemble rate", "Out-of-ensemble rate",
                "90% coverage", "90% coverage", "90% coverage",
            ],
        ),
    ],
)
def test_margin_plots_select_lead_landsea_and_preserve_sparse_margins(
    tmp_path: Path, monkeypatch, constructor, name: str, ylabels: list[str]
) -> None:
    write_synthetic_results(tmp_path)
    plotted = []
    original_plot = plt.Axes.plot
    original_close = plt.close

    def record_plot(axis, x, y, *args, **kwargs):
        plotted.append((np.asarray(x), np.asarray(y), kwargs))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", record_plot)
    monkeypatch.setattr(plt, "close", lambda figure: None)

    constructor(tmp_path, tmp_path / f"{name}.png", 2022, 12, "land")

    figure = plt.gcf()
    data_lines = [line for line in plotted if "label" in line[2]]
    assert len(figure.axes) == 9
    assert len(data_lines) == 36
    assert any(
        np.array_equal(x, [0.0, 0.25, 1.25]) and np.array_equal(y, [1.0, np.nan, 3.0], equal_nan=True)
        for x, y, _ in data_lines
    )
    assert all(axis.get_title() == title for axis, title in zip(figure.axes[:3], ["Heat records", "Cold records", "Wind records"]))
    assert [axis.get_ylabel() for axis in figure.axes] == ylabels
    assert [axis.get_xlabel() for axis in figure.axes[6:]] == ["Margin (K)", "Margin (K)", "Margin (m s^-1)"]
    assert {line[2]["label"] for line in data_lines} == {"IFS-ENS", "U-Cast", "FGN", "ArchesWeatherGen"}
    zero_lines = [
        line for axis in figure.axes for line in axis.lines
        if np.array_equal(line.get_ydata(), [0, 0])
    ]
    assert len(zero_lines) == (3 if constructor is plot_margin_intensity else 0)
    original_close(figure)


@pytest.mark.parametrize("constructor", [plot_margin_intensity, plot_margin_probabilistic])
def test_margin_plots_skip_malformed_results_and_reject_no_usable_data(
    tmp_path: Path, constructor
) -> None:
    write_synthetic_results(tmp_path)
    malformed = next(tmp_path.glob("IFS_ENS/Margin/*_max_margin_2022_t2m_*.nc"))
    rewrite_dataset(malformed, lambda dataset: dataset.drop_vars("margin"))

    output = tmp_path / "margin.png"
    constructor(tmp_path, output, 2022, 12, "land")
    assert output.is_file()

    for path in tmp_path.glob("*/Margin/*.nc"):
        rewrite_dataset(path, lambda dataset: dataset.drop_vars("margin", errors="ignore"))
    figures_before = set(plt.get_fignums())
    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, 12, "land")
    assert set(plt.get_fignums()) == figures_before


@pytest.mark.parametrize(
    ("constructor", "variables"),
    [
        (plot_margin_intensity, ("crps", "rmse_mean", "bias_mean")),
        (plot_margin_probabilistic, ("record_probability", "out_of_ensemble_rate", "coverage_90")),
    ],
)
def test_margin_plots_reject_all_nan_selected_curves_and_close_figure(
    tmp_path: Path, constructor, variables: tuple[str, ...]
) -> None:
    write_synthetic_results(tmp_path)
    for path in tmp_path.glob("*/Margin/*.nc"):
        def invalidate(dataset):
            for variable in variables:
                dataset[variable] = xr.full_like(dataset[variable], np.nan)
            return dataset

        rewrite_dataset(path, invalidate)
    figures_before = set(plt.get_fignums())

    with pytest.raises(FileNotFoundError, match="No usable series"):
        constructor(tmp_path, tmp_path / "empty.png", 2022, 12, "land")

    assert set(plt.get_fignums()) == figures_before


def test_margin_plot_save_uses_220dpi_and_tight_bounding_box(tmp_path: Path, monkeypatch) -> None:
    write_synthetic_results(tmp_path)
    save_calls = []
    original_savefig = Figure.savefig

    def record_savefig(figure, *args, **kwargs):
        save_calls.append((args, kwargs))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_savefig)

    plot_margin_intensity(tmp_path, tmp_path / "margin.png", 2022, 12, "land")

    assert len(save_calls) == 1
    assert save_calls[0][1]["dpi"] == 220
    assert save_calls[0][1]["bbox_inches"] == "tight"
