"""
2x3 figure: RMSE (top) / Bias (bottom) x Heat / Cold / Wind records.

Each panel shows ensemble-mean point-forecast skill vs. lead time for all
four models (IFS-ENS, U-Cast, FGN, ArchesWeatherGen), read from the
per-model results/<model>/Point/ NetCDFs (case=max for heat/wind records,
case=min for cold records).

Run from the project root:
    python -u viz/plot_rmse_bias_by_record_type.py
    python -u viz/plot_rmse_bias_by_record_type.py --year 2022 --season JJA --landsea land
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data import load_point_series
from style import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, apply_style

COLUMNS = [
    {"title": "Heat records", "var": "t2m", "case": "max", "unit": "°C"},
    {"title": "Cold records", "var": "t2m", "case": "min", "unit": "°C"},
    {"title": "Wind records", "var": "w10", "case": "max", "unit": "m/s"},
]

ROWS = [
    {"title": "RMSE", "prefix": "rmse"},
    {"title": "Bias", "prefix": "bias"},
]


def build_figure(
    results_root: Path,
    year: int | None,
    season: str,
    landsea: str,
    metric: str,
):
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex="col")

    for col_idx, col in enumerate(COLUMNS):
        for row_idx, row in enumerate(ROWS):
            ax = axes[row_idx, col_idx]
            metric_name = f"{row['prefix']}_{metric}"

            if row["prefix"] == "bias":
                ax.axhline(0, color="#c3c2b7", linewidth=1, linestyle="--", zorder=0)

            for model in MODEL_ORDER:
                hours, values = load_point_series(
                    model,
                    col["case"],
                    col["var"],
                    metric_name,
                    results_root,
                    season=season,
                    landsea=landsea,
                    year=year,
                )
                if hours is None:
                    continue
                ax.plot(
                    hours,
                    values,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )

            ax.set_ylabel(f"{row['title']} ({col['unit']})")
            if row_idx == 0:
                ax.set_title(col["title"])
            if row_idx == len(ROWS) - 1:
                ax.set_xlabel("Lead time (h)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

    season_tag = "" if season == "all" else f", {season}"
    landsea_tag = "" if landsea == "all" else f", {landsea}"
    fig.suptitle(
        f"Record-breaking event skill — {year}{season_tag}{landsea_tag}",
        y=1.10,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022, choices=[2018, 2020, 2022])
    parser.add_argument(
        "--season", type=str, default="all", choices=["DJF", "MAM", "JJA", "SON", "all"]
    )
    parser.add_argument("--landsea", type=str, default="all", choices=["all", "land", "sea"])
    parser.add_argument("--metric", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--out", type=Path, default=Path("figures/rmse_bias_by_record_type.png")
    )
    args = parser.parse_args()

    fig = build_figure(
        args.results_root, args.year, args.season, args.landsea, args.metric
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
