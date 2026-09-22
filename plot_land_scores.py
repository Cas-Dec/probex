#!/usr/bin/env python3
"""Plot saved ProbEx summaries; no forecast downloads or rescoring.

Run from the project root: python plot_land_scores.py
Missing point-score files leave labelled empty RMSE panels; other scores plot
immediately. Run eval/forecast_evaluation_point.py and repeat to fill those panels.

Figures use hours, a regular grid, and independent y-axis ranges per panel.
The title describes the ERA5 record baseline. Verification year, target and
coverage information remain in the output filenames, CSVs and console output.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr

EVENTS = [("heat", "t2m", "max", "K"), ("cold", "t2m", "min", "K"),
          ("wind", "w10", "max", "m/s")]
PANELS = [
    ("rmse_all", "RMSE (all cases)"),
    ("rmse_record", "RMSE (records only)"),
    ("crps_all", "Fair CRPS (all cases)"),
    ("crps_record", "Fair CRPS (records only)"),
    ("rtwcrps", "Fair rtwCRPS (all cases)"),
    ("brier", "Fair Brier score (all cases)"),
]
COLORS = {"IFS_ENS": "tab:orange", "FGN": "tab:blue", "uCast": "tab:green",
          "ArchesWeatherGen": "tab:purple", "GenCast": "tab:red"}

# 1. Options

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("results"))
parser.add_argument("--out-dir", type=Path, default=Path("figures/land_scores"))
parser.add_argument("--models", nargs="+", default=["IFS_ENS", "FGN"])
parser.add_argument("--reference", default="IFS_ENS")
parser.add_argument("--leads", nargs="+", type=int, default=[12, 48, 120, 168, 240])
parser.add_argument("--x-step", type=float, default=48,
                    help="Spacing of the regular x-axis grid, in hours (default: 48)")
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--record-tag", default="1979_2021")
parser.add_argument("--truth", choices=["ERA5", "HRES_fc0"], default="ERA5")
parser.add_argument("--surface", choices=["land", "sea", "all"], default="land")
parser.add_argument("--season", choices=["all", "DJF", "MAM", "JJA", "SON"], default="all")
parser.add_argument("--require-rmse", action="store_true")
args = parser.parse_args()
if len(set(args.models)) != len(args.models) or args.reference not in args.models:
    parser.error("Models must be unique and include --reference.")
if args.leads != sorted(set(args.leads)) or min(args.leads) <= 0:
    parser.error("Lead hours must be positive, sorted and unique.")
if not np.isfinite(args.x_step) or args.x_step <= 0:
    parser.error("--x-step must be finite and positive.")

# 2. Read the saved scores and check their metadata

rows, missing = [], []
for event, var, tail, unit in EVENTS:
    suffix = f"{args.year}_{var}_no_Antarctic_records_{args.record_tag}"
    for model in args.models:
        base = args.root / model
        definitions = [
            ("rmse_all", "Point", f"{model}_all_point_{suffix}_cond_{args.truth}.nc", "rmse_mean", False),
            ("rmse_record", "Point", f"{model}_{tail}_point_{suffix}_cond_{args.truth}.nc", "rmse_mean", False),
            ("crps_all", "CRPS", f"{model}_all_CRPS_{suffix}_cond_{args.truth}.nc", "CRPS", True),
            ("crps_record", "CRPS", f"{model}_{tail}_CRPS_{suffix}_cond_{args.truth}.nc", "CRPS", True),
            ("rtwcrps", "CRPS", f"{model}_{tail}_rtwCRPS_{suffix}_{args.truth}.nc", "rtwCRPS", True),
            ("brier", "Occurrence", f"{model}_{tail}_occurrence_{suffix}_cond_{args.truth}.nc", "brier_score", True),
        ]
        for metric, folder, filename, field, fair in definitions:
            path = base / folder / filename
            if not path.is_file():
                if folder != "Point" or args.require_rmse:
                    raise FileNotFoundError(path)
                missing.append(str(path))
                continue
            with xr.open_dataset(path) as ds:
                data = ds[field]
                attrs = {**ds.attrs, **data.attrs}
                for key, expected in [("verification_target", args.truth), ("record_tag", args.record_tag)]:
                    actual = attrs.get(key)
                    if actual is not None and actual != expected:
                        raise ValueError(f"{path}: {key}={actual}, expected {expected}")
                estimator = attrs.get("brier_estimator", attrs.get("estimator", "unspecified"))
                if fair and "fair" not in str(estimator).lower():
                    raise ValueError(f"{path}: fair estimator is not documented ({estimator}). "
                                     "Rerun with the fair-score evaluator before labelling this figure.")
                data = data.sel(season=args.season, landsea=args.surface)
                if set(data.dims) != {"leadtime"}:
                    raise ValueError(f"{path}: unexpected dimensions {data.dims}")
                hours = np.asarray(data.leadtime.values)
                if np.issubdtype(hours.dtype, np.timedelta64):
                    hours = np.asarray(hours / np.timedelta64(1, "h"), dtype=float)
                else:
                    hours = hours.astype(float)
                if len(np.unique(hours)) != len(hours) or not np.isfinite(hours).all():
                    raise ValueError(f"{path}: duplicated or invalid lead times")
                series = pd.Series(data.values.astype(float), index=hours).sort_index()
                absent = sorted(set(args.leads) - set(series.index))
                if absent:
                    raise ValueError(f"{path}: missing lead hours {absent}")
                series = series.reindex(args.leads)
                if np.isinf(series.values).any():
                    raise ValueError(f"{path}: infinite values in {field}")
            if not np.isfinite(series.values).all():
                warnings.warn(f"{path.name}: missing values are shown as gaps, not interpolated.")
            for lead, value in series.items():
                rows.append(dict(event=event, variable=var, tail=tail, model=model,
                                 leadtime_hours=lead, metric=metric, value=value,
                                 units="1" if metric == "brier" else unit,
                                 season=args.season, surface=args.surface,
                                 verification_target=args.truth, record_tag=args.record_tag,
                                 estimator=estimator, source_file=str(path),
                                 coverage="existing summaries; dates not checked by plotter"))

table = pd.DataFrame(rows)
missing = sorted(set(missing))

# 3. Save the scores and their differences from the reference model

args.out_dir.mkdir(parents=True, exist_ok=True)
label = f"{args.surface}_{args.year}_{args.season}_{args.truth}_{args.record_tag}"
table.to_csv(args.out_dir / f"scores_{label}.csv", index=False)

keys = ["event", "metric", "leadtime_hours", "season", "surface"]
base = table.loc[table.model == args.reference, keys + ["value"]].rename(columns={"value": "reference_value"})
comparisons = table.loc[table.model != args.reference].merge(base, on=keys, how="left", validate="many_to_one")
denominator = comparisons.reference_value.where(comparisons.reference_value > 0)
comparisons["relative_score_reduction_percent"] = 100 * (1 - comparisons.value / denominator)
comparisons["reference_model"] = args.reference
comparisons.to_csv(args.out_dir / f"relative_scores_{label}.csv", index=False)

# 4. Draw and save one figure per record type

plt.style.use("default")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42})

for event, _, _, unit in EVENTS:
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), sharex=True)
    hours = np.asarray(args.leads, dtype=float)
    for letter, axis, (metric, title) in zip("abcdef", axes.flat, PANELS):
        axis.set_title(f"({letter}) {title}", loc="left", fontsize=11)
        for index, model in enumerate(args.models):
            selected = table.loc[(table.event == event) & (table.metric == metric) & (table.model == model)]
            if not selected.empty:
                selected = selected.set_index("leadtime_hours").reindex(args.leads)
                axis.plot(hours, selected.value, marker="o", markersize=4, linewidth=1.6,
                          color=COLORS.get(model, f"C{index}"), label=model)
        if metric.startswith("rmse"):
            missing_rmse_models = [m for m in args.models if table.loc[
                (table.event == event) & (table.metric == metric) & (table.model == m)].empty]
            if missing_rmse_models:
                axis.text(.04, .94, "RMSE not yet saved: " + ", ".join(missing_rmse_models),
                          transform=axis.transAxes, va="top", fontsize=9, color="#666666", wrap=True)
        axis.set_xlim(0, hours.max() * 1.02)
        axis.xaxis.set_major_locator(MultipleLocator(args.x_step))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_ylabel("Brier score" if metric == "brier" else unit)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                              useOffset=False, useMathText=True)
        # Each panel fits its own data. This keeps all-case curves legible
        # without imposing the much larger range of record-conditioned errors.
        axis.margins(y=.12)
        low, high = axis.get_ylim()
        panel_values = table.loc[(table.event == event) & (table.metric == metric), "value"]
        finite = panel_values[np.isfinite(panel_values)]
        if len(finite) and finite.min() >= 0 and low < 0:
            axis.set_ylim(0, high)
        axis.set_axisbelow(True)
        # Axes.grid is the subplot equivalent of plt.grid().
        axis.grid(True, which="major", axis="both", color="0.85", linewidth=.7)
    for axis in axes[-1]:
        axis.set_xlabel("Lead time (hours)")
    title = {"heat": "Heat records", "cold": "Cold records", "wind": "Wind records"}[event]
    surface = {"land": "land only", "sea": "ocean only", "all": "land and ocean"}[args.surface]
    season = "" if args.season == "all" else f", {args.season}"
    fig.suptitle(f"{title} (ERA5 {args.record_tag.replace('_', '-')}, {surface}{season})",
                 fontsize=14, y=.985)
    handles = [Line2D([], [], color=COLORS.get(m, f"C{i}"), marker="o", label=m)
               for i, m in enumerate(args.models)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
               ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .90), h_pad=2, w_pad=2.5)
    stem = f"{event}_{args.surface}_{args.year}_{args.season}_{args.truth}_{args.record_tag}"
    for extension in ["pdf", "png"]:
        path = args.out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=180)
        print(f"Saved: {path}")
    plt.close(fig)

if missing:
    print("\nMissing point-score summaries (run the point evaluator, then repeat this command):")
    for path in missing:
        print("  " + path)
print("\nRelative-score CSV: positive = lower score than the reference; comparisons are not yet date-paired.")
print(f"Verification: {args.truth}, {args.year}; {args.surface}; season {args.season}; Antarctica excluded.")
print("Y-axis ranges are independent between panels. RMSE uses the ensemble mean.")
print("Record-conditioned panels are descriptive; no paired uncertainty intervals are shown.")
print("Compare models within a metric. CRPS and rtwCRPS have different targets; their sizes are not interchangeable.")
print("The fair correction assumes appropriate ensemble sampling; it does not remove outcome-selection bias.")
