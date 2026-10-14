#!/usr/bin/env python3
"""Simple Matplotlib record-depth plots from the small paired summary files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

COLORS = {"IFS_ENS": "tab:orange", "FGN": "tab:blue", "uCast": "tab:green",
          "ArchesWeatherGen": "tab:purple", "GenCast": "tab:red"}

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=root / "results")
parser.add_argument("--models", nargs="+", default=["IFS_ENS", "FGN"])
parser.add_argument("--truth", choices=["ERA5", "HRES_fc0"], default="ERA5")
parser.add_argument("--output", type=Path, default=root / "figures/record_depth")
parser.add_argument("--year", type=int, default=2022)
parser.add_argument("--record-tag", default="1979_2021")
parser.add_argument("--surface", default="land", choices=["land", "sea", "all"])
parser.add_argument("--leads", nargs="+", type=int, help="Optional subset of already evaluated leads")
parser.add_argument("--reference", default="IFS_ENS")
parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

found = 0
for event in ["heat", "cold", "wind"]:
    stem = f"{event}_depth_{args.year}_{args.truth}_{args.record_tag}_{args.surface}"

    var = "w10" if event == "wind" else "t2m"
    case = "min" if event == "cold" else "max"
    paths = [args.results / model / "Depth" /
             f"{model}_{case}_depth_{args.year}_{var}_no_Antarctic_records_{args.record_tag}_{args.truth}.nc"
             for model in args.models]
    if not any(path.is_file() for path in paths):
        # Read earlier Biobot combined summaries when that folder is explicitly supplied.
        legacy = args.results / (stem + ".nc")
        if not legacy.is_file():
            continue
        with xr.open_dataset(legacy) as source:
            data = source.sel(model=args.models).load()
        ds = data.sel(leadtime=args.leads) if args.leads else data
    else:
        parts = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing model result: {path}")
            with xr.open_dataset(path) as source:
                parts.append(source.sel(landsea=args.surface, drop=True).load())
        leads = args.leads or sorted(set.intersection(*(set(part.leadtime.values) for part in parts)))
        if not leads:
            raise ValueError("No common leads in the selected model results.")
        parts = [part.sel(leadtime=leads) for part in parts]
        first = parts[0]
        for part in parts[1:]:
            for key in ["verification_target", "record_tag", "truth_file", "reference_fingerprint"]:
                if part.attrs.get(key) != first.attrs.get(key):
                    raise ValueError(f"Different {key}: evaluate the models against the same reference data.")
            if not np.array_equal(part.verification_time, first.verification_time):
                raise ValueError("Valid dates differ. Rerun each model with the same --compare-models and --leads.")
            for name in ["depth", "record_count", "valid_count", "record_weight", "valid_weight"]:
                xr.testing.assert_equal(part[name], first[name])
        data = xr.concat([part.drop_vars("verification_time").expand_dims(model=[model])
                          for model, part in zip(args.models, parts)], dim="model", join="exact")
        # Event support is a property of the reference.
        for name in ["record_count", "valid_count", "record_weight", "valid_weight", "excluded_scale_count",
                     "observed_probability"]:
            data[name] = first[name]
        data.attrs = first.attrs
        ds = data

    if ds.attrs.get("common_dates_across_leads") != "true":
        raise ValueError("Results do not establish common verification dates across leads.")
    found += 1
    surface = "all surfaces" if args.surface == "all" else f"{args.surface} only"
    title = f"{event.capitalize()} records (ERA5 {args.record_tag.replace('_', '-')}, {surface})"
    if args.truth != "ERA5":
        title += f"; verification: {args.truth}"

    for lead in ds.leadtime.values:
        data = ds.sel(leadtime=lead)
        depths = data.depth.values
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        fig.suptitle(f"{title} — {int(lead)} h", fontsize=14)
        panels = [("rmse_record", "RMSE", "rmse_all"), ("crps_record", "Fair CRPS", "crps_all"),
                  ("rtwcrps", "Fair rtwCRPS (all cases)", None), ("brier", "Fair Brier score (all cases)", None)]
        units = ds.attrs["units"].replace("m s-1", "m/s")
        for ax, (name, label, all_name) in zip(axes.flat, panels):
            for model in data.model.values:
                values = data[name].sel(model=model).values
                ax.plot(depths, values, "o-", color=COLORS.get(str(model)), label=str(model))
                if all_name:
                    ax.axhline(float(data[all_name].sel(model=model)), color=COLORS.get(str(model)),
                               ls="--", alpha=0.8)
            ax.set_title(label + (": records (solid), all (dashed)" if all_name else ""), fontsize=10)
            ax.set_xlabel("Record depth k (historical SD)")
            ax.set_ylabel("Score" if name == "brier" else f"Score ({units})")
            ax.set_xticks(depths)
            ax.grid(True, which="major", axis="both", alpha=0.35)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ax.set_ylim(bottom=0)
        axes[0, 0].legend(frameon=False)
        path = args.output / f"{stem}_{int(lead):03d}h"
        for fmt in args.formats:
            fig.savefig(path.with_suffix("." + fmt), dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path} ({', '.join(args.formats)})")

    print(f"{event}: {ds.attrs['valid_time_count']} paired valid times.")

    # Calculate relative skill, then draw it alongside the event counts

    if args.reference not in ds.model.values or ds.sizes["model"] < 2:
        print("Skipping relative skill figure: reference and a second model are required.")
        continue
    depths = ds.depth.values
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    fig.suptitle(title, fontsize=14)
    colors = plt.get_cmap("tab10")
    leads = ds.leadtime.values
    for ax, metric, name in zip(axes[:2], ["rtwcrps", "brier"], ["rtwCRPS", "Brier"]):
        models = [model for model in ds.model.values if model != args.reference]
        for model_index, model in enumerate(models):
            for j, lead in enumerate(leads):
                denom = ds[metric].sel(model=args.reference, leadtime=lead)
                support = ds.record_count.sel(leadtime=lead) > 0
                skill = 100 * (1 - ds[metric].sel(model=model, leadtime=lead) / denom.where(denom > 0))
                skill = skill.where(support)
                ax.plot(depths, skill, marker="o", ls=["-", "--", ":", "-."][model_index % 4],
                        color=colors(j % 10), label=f"{model}, {int(lead)} h")
        ax.axhline(0, color="0.3", lw=1, ls="--")
        ax.set_title(f"{name} skill versus {args.reference}", fontsize=11)
        ax.set_xlabel("Record depth k (historical SD)")
        ax.set_ylabel("Score reduction (%)")
        ax.set_xticks(depths)
        ax.grid(True, which="major", axis="both", alpha=0.35)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(4, len(labels)), frameon=False, fontsize=9)
    # Cases are paired across leads, so support must be identical.
    counts = ds.record_count.transpose("leadtime", "depth").values
    if not np.array_equal(counts, np.broadcast_to(counts[0], counts.shape)):
        raise ValueError("Record counts differ across leads; check common-case pairing.")
    axes[2].plot(depths, counts[0], "o-", color="0.25")
    axes[2].set_title("Observed exceedances", fontsize=11)
    axes[2].set_xlabel("Record depth k (historical SD)")
    axes[2].set_ylabel("Grid-cell-times")
    axes[2].set_xticks(depths)
    axes[2].grid(True, which="major", axis="both", alpha=0.35)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    axes[2].set_ylim(bottom=0)
    path = args.output / f"{stem}_skill"
    for fmt in args.formats:
        fig.savefig(path.with_suffix("." + fmt), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path} ({', '.join(args.formats)})")
