"""
1x3 figure: expected minus observed record count, for Heat / Cold / Wind
records, all four models overlaid per panel.

expected_count is the ensemble's predicted record probability summed over
the observed record-breaking grid-time points; observed_count is the number
of those points. The difference is zero when the ensemble's predicted record
probability is calibrated to what actually occurred; positive means the
ensemble over-predicted records, negative means it under-predicted them.

Read from results/<model>/Occurrence/ (case=max for heat/wind, case=min for
cold).

Run from the project root:
    python -u viz/plot_expected_minus_observed_records.py
    python -u viz/plot_expected_minus_observed_records.py --year 2022 --season JJA --landsea land
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from data import load_occurrence_diff
from style import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, apply_style

COLUMNS = [
    {"title": "Heat records", "var": "t2m", "case": "max"},
    {"title": "Cold records", "var": "t2m", "case": "min"},
    {"title": "Wind records", "var": "w10", "case": "max"},
]


def build_figure(results_root: Path, year: int | None, season: str, landsea: str):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for col_idx, col in enumerate(COLUMNS):
        ax = axes[col_idx]
        ax.axhline(0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=0)

        for model in MODEL_ORDER:
            hours, diff = load_occurrence_diff(
                model,
                col["case"],
                col["var"],
                results_root,
                season=season,
                landsea=landsea,
                year=year,
            )
            if hours is None:
                continue
            ax.plot(
                hours,
                diff,
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )

        ax.set_title(col["title"])
        ax.set_xlabel("Lead time (h)")
        if col_idx == 0:
            ax.set_ylabel("Expected − observed records")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(MODEL_ORDER),
        frameon=False,
    )

    season_tag = "" if season == "all" else f", {season}"
    landsea_tag = "" if landsea == "all" else f", {landsea}"
    fig.suptitle(
        f"Record-count calibration — {year}{season_tag}{landsea_tag}",
        y=1.16,
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
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--out", type=Path, default=Path("figures/expected_minus_observed_records.png")
    )
    args = parser.parse_args()

    fig = build_figure(args.results_root, args.year, args.season, args.landsea)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
