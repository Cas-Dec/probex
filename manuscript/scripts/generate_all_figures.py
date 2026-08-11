"""Generate the complete manuscript figure inventory from evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manuscript.scripts.plotting import (
    plot_brier,
    plot_counts,
    plot_crps,
    plot_margin_intensity,
    plot_margin_probabilistic,
    plot_point,
    plot_precision_recall,
    plot_reliability,
)


SEASONS = ("all", "DJF", "MAM", "JJA", "SON")
LANDSEA = ("all", "land", "sea")
LEADS = (48, 120, 168, 240)


def generate_all(
    results_root, output_dir, year, record_tag: str = "1979_2018"
) -> tuple[int, int]:
    """Generate every manuscript figure, returning generated and skipped counts."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = skipped = 0

    jobs = []
    for family, constructor in (
        ("point", plot_point),
        ("counts", plot_counts),
        ("brier", plot_brier),
        ("crps", plot_crps),
    ):
        for season in SEASONS:
            for landsea in LANDSEA:
                jobs.append((f"{family}_{season}_{landsea}.png", constructor, (season, landsea)))
    for family, constructor in (
        ("precision_recall", plot_precision_recall),
        ("reliability", plot_reliability),
    ):
        for lead in LEADS:
            for season in SEASONS:
                for landsea in LANDSEA:
                    jobs.append(
                        (f"{family}_{lead}h_{season}_{landsea}.png", constructor, (lead, season, landsea))
                    )
    for family, constructor in (
        ("margin_intensity", plot_margin_intensity),
        ("margin_probabilistic", plot_margin_probabilistic),
    ):
        for lead in LEADS:
            for landsea in LANDSEA:
                jobs.append((f"{family}_{lead}h_{landsea}.png", constructor, (lead, landsea)))

    for name, constructor, arguments in jobs:
        try:
            constructor(results_root, output_dir / name, year, *arguments, record_tag=record_tag)
        except (FileNotFoundError, ValueError) as error:
            skipped += 1
            print(f"Skipped {name}: {error}")
        else:
            generated += 1

    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="manuscript/fig")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--record-tag", default="1979_2018")
    args = parser.parse_args()
    generate_all(args.results_root, args.output_dir, args.year, args.record_tag)


if __name__ == "__main__":
    main()
