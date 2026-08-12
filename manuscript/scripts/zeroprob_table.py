"""Generate a LaTex table of zero-probability record misses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import xarray as xr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manuscript.scripts.plotting import EVENTS, MODEL_LABELS, MODEL_ORDER, find_result_file, leadtime_hours


LEADS = (12, 48, 120)


def zero_probability_miss_rates(results_root: Path, model: str, event: dict, year: int, record_tag: str) -> dict[int, float]:
    """Return 1 - recall at the zero-probability threshold for available leads."""
    path = find_result_file(results_root, model, "occurrence", event["case"], event["var"], year, record_tag)
    if path is None:
        return {}
    with xr.open_dataset(path) as dataset:
        recall = dataset["recall"].sel(season="all", landsea="all", threshold=0.0)
        hours = leadtime_hours(dataset["leadtime"].values)
        return {int(hour): 1.0 - float(value) for hour, value in zip(hours, recall.values)}


def render_table(results_root: Path, year: int, record_tag: str = "1979_2018") -> str:
    """Render the supplementary zero-probability miss-rate table as LaTex."""
    lines = [
        "\\begin{table}[p]",
        "\\centering",
        "\\caption{Zero-probability miss rate, $\\Pr(\\hat{p}^{\\tau}=0 \\mid y^{\\tau}=1)$, for all-domain, all-season observed record events. The rate is the unweighted fraction of observed record grid-time pairs for which no ensemble member crossed the local historical record; lower values are better.}",
        "\\label{tab:supp-zeromiss}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Record type & Model & 12 h & 48 h & 120 h \\\\",
        "\\midrule",
    ]
    for event_index, event in enumerate(EVENTS):
        rates_by_model = {
            model: zero_probability_miss_rates(results_root, model, event, year, record_tag)
            for model in MODEL_ORDER
        }
        best_rates = {
            lead: min(rates[lead] for rates in rates_by_model.values() if lead in rates and np.isfinite(rates[lead]))
            for lead in LEADS
            if any(lead in rates and np.isfinite(rates[lead]) for rates in rates_by_model.values())
        }
        for model in MODEL_ORDER:
            rates = rates_by_model[model]
            cells = []
            for lead in LEADS:
                if lead not in rates or not np.isfinite(rates[lead]):
                    cells.append("--")
                else:
                    value = f"{rates[lead]:.3f}"
                    cells.append(f"\\textbf{{{value}}}" if np.isclose(rates[lead], best_rates[lead]) else value)
            lines.append(f"{event['title'].removesuffix(' records')} & {MODEL_LABELS[model]} & {' & '.join(cells)} \\\\")
        if event_index < len(EVENTS) - 1:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--record-tag", default="1979_2018")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    table = render_table(args.results_root, args.year, args.record_tag)
    if args.output:
        args.output.write_text(table + "\n")
    else:
        print(table)


if __name__ == "__main__":
    main()
