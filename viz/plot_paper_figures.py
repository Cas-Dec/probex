
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

root = Path(__file__).resolve().parents[1]
results_root = root / "results"

year = 2022
record_tag = "1979_2021"
surface = "land"                    # "land", "sea" or "all"
truths = ["ERA5", "HRES_fc0"]
primary_truth = "ERA5"
models = ["ArchesWeatherGen", "uCast", "FGN", "GenCast-mini"]
reference = "IFS_ENS"
leads = [48, 120, 168, 240]          
depths = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5]  
main_depth_max = 2                  
example_lead = 48                   # for truth-switch/ selection figures

formats = ["png", "pdf"]
dpi = 180
out = root / "figures" / f"paper_{year}_{record_tag}_{surface}_with_gencast"
out.mkdir(parents=True, exist_ok=True)

names = {
    "IFS_ENS": "IFS-ENS", "ArchesWeatherGen": "AWG",
    "uCast": "U-Cast", "FGN": "FGN", "GenCast": "GenCast", "GenCast-mini": "GenCast-mini",
}
colors = {
    "IFS_ENS": "#202d3a", "ArchesWeatherGen": "#078270",
    "uCast": "#8b55a1", "FGN": "#087eae", "GenCast": "#b3443c", "GenCast-mini": "#b3443c",
}
events = ["heat", "cold", "wind"]
variables = {"heat": "t2m", "cold": "t2m", "wind": "w10"}
cases = {"heat": "max", "cold": "min", "wind": "max"}
all_models = [reference] + models

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})

if primary_truth not in truths or example_lead not in leads or (depths is not None and 0 not in depths):
    raise ValueError("Include primary_truth, example_lead and depth 0 in the settings.")
if reference in models or len(set(all_models)) != len(all_models):
    raise ValueError("models should list each AI model once; IFS_ENS is added separately.")


# Read the depth results 
parts = []
for truth in truths:
    for event in events:
        comparison_depths = None
        for model in all_models:
            filename = (
                f"{model}_{cases[event]}_depth_{year}_{variables[event]}"
                f"_no_Antarctic_records_{record_tag}_{truth}.csv"
            )
            path = results_root / model / "Depth" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing result: {path}. Check the settings above.")
            table = pd.read_csv(path)
            table = table[
                (table.landsea == surface)
                & table.leadtime.isin(leads)
            ].copy()
            if depths is not None:
                table = table[table.depth.isin(depths)].copy()
            selected_depths = sorted(table.depth.unique().tolist()) if depths is None else sorted(depths)
            if 0 not in selected_depths:
                raise ValueError(f"No record-boundary (depth 0) result in {path}")
            expected = pd.MultiIndex.from_product([leads, selected_depths])
            actual = pd.MultiIndex.from_frame(table[["leadtime", "depth"]])
            if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
                raise ValueError(f"Incomplete or duplicated lead/depth rows in {path}")

            if comparison_depths is None:
                comparison_depths = selected_depths
                print(f"{truth}, {event}: depths {selected_depths}")
            elif selected_depths != comparison_depths:
                raise ValueError(f"Models have different depth grids for {truth}, {event}.")
            table["model"] = model
            table["truth"] = truth
            table["event"] = event
            parts.append(table)

data = pd.concat(parts, ignore_index=True)
depth_ticks = np.arange(0, data.depth.max() + 0.01, 0.5)  # readable half-step labels
keys = ["truth", "event", "leadtime", "depth"]

# Relative skill: divide averaged model scores by averaged IFS scores
metrics = ["crps_all", "crps_record", "rtwcrps", "brier"]
ref = data[data.model == reference][keys + metrics].rename(
    columns={metric: "ifs_" + metric for metric in metrics}
)
data = data.merge(ref, on=keys, validate="many_to_one")
for metric in metrics:
    denominator = data["ifs_" + metric].where(data["ifs_" + metric] > 0)
    data[metric + "_skill"] = 100 * (1 - data[metric] / denominator)
data["probability_ratio"] = data.probability / data.observed_probability.where(data.observed_probability > 0)
full_depth_data = data.copy()
full_depth_data.to_csv(out / "native_skill_all_depths.csv", index=False)
data = data[data.depth <= main_depth_max].copy()
depth_ticks = np.arange(0, main_depth_max + .01, .5)
data.to_csv(out / "native_skill.csv", index=False)

print(f"Read {len(parts)} depth files; figures will be saved in {out}")


# Record skill versus lead time (paper Figure 2)
fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.1), sharex=True)
fig.subplots_adjust(left=.075, right=.99, top=.90, bottom=.11, hspace=.40, wspace=.29)
for row, metric in enumerate(['rtwcrps', 'brier']):
    for col, event in enumerate(events):
        ax = axes[row, col]
        d = data[(data.truth == primary_truth) & (data.depth == 0) & (data.event == event)]
        for model in models:
            s = d[d.model == model].sort_values('leadtime')
            ax.plot(s.leadtime, s[metric + '_skill'], 'o-', color=colors[model], lw=2, ms=4, label=names[model])
        ax.axhline(0, color='.35', ls='--', lw=1)
        ax.grid(True, alpha=.25)
        ax.set_title(f'({chr(97+row*3+col)}) {event.capitalize()}')
        ax.set_xticks(leads)
        limits = d[d.model != reference][metric + '_skill']
        ax.set_ylim(min(0, limits.min()) - 4, max(0, limits.max()) + 4)
        if row == 1: ax.set_xlabel('Lead time (h)')
        if col == 0: ax.set_ylabel(('rtwCRPS' if row == 0 else 'Brier') + ' skill (%)')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=min(4, len(models)), frameon=False, bbox_to_anchor=(.54, 1.005))
for ext in formats:
    fig.savefig(out / ('record_skill.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=.06)
plt.close(fig)
print('Saved record_skill')

# Skill versus threshold depth

for truth in truths:
    for metric in ['rtwcrps', 'brier']:
        fig, axes = plt.subplots(3, len(leads), figsize=(max(6.4, 2.3 * len(leads)), 8.1), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.07, right=.99, top=.92, bottom=.09, hspace=.39, wspace=.22)
        for row, event in enumerate(events):
            subset = data[(data.truth == truth) & (data.event == event) & (data.model != reference)]
            lower = 10 * np.floor(min(0, subset[metric + '_skill'].min()) / 10) - 3
            upper = 10 * np.ceil(subset[metric + '_skill'].max() / 10) + 3
            for col, lead in enumerate(leads):
                ax = axes[row, col]
                for model in models:
                    s = subset[(subset.model == model) & (subset.leadtime == lead)].sort_values('depth')
                    ax.plot(s.depth, s[metric + '_skill'], 'o-', color=colors[model], lw=1.9, ms=4, label=names[model])
                ax.axhline(0, color='.35', ls='--', lw=1)
                ax.grid(True, alpha=.25)
                ax.set_xticks(depth_ticks)
                ax.set_ylim(lower, upper)
                ax.set_title(f'({chr(97+row*len(leads)+col)}) {event.capitalize()}, {lead} h', fontsize=11)
                if row == 2: ax.set_xlabel('Threshold depth k')
                if col == 0: ax.set_ylabel('Skill versus IFS-ENS (%)')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=min(4, len(models)), frameon=False, bbox_to_anchor=(.54, 1.005))
        for ext in formats:
            fig.savefig(out / f'{metric}_depth_{truth}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.06)
        plt.close(fig)
        print(f'Saved {metric}_depth_{truth}')

#  Conditioning, occurrence rates and score contributions

if {'FGN', 'ArchesWeatherGen'}.issubset(models):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    fig.subplots_adjust(left=.09, right=.98, top=.94, bottom=.10, hspace=.43, wspace=.27)
    for ax, event, model, letter in [(axes[0, 0], 'heat', 'FGN', 'a'), (axes[0, 1], 'wind', 'ArchesWeatherGen', 'b')]:
        s = data[(data.truth == primary_truth) & (data.event == event) & (data.leadtime == example_lead) & (data.model == model)].sort_values('depth')
        for metric, label, color, marker in [('crps_record', 'CRPS | observed exceedance', '#9f4b3d', 's'), ('rtwcrps', 'rtwCRPS | all cases', '#087eae', 'o'), ('brier', 'Brier | all cases', '#078270', '^')]:
            ax.plot(s.depth, s[metric + '_skill'], marker+'-', lw=2, ms=4, color=color, label=label)
        ax.axhline(0, color='.35', ls='--', lw=1)
        ax.set(title=f'({letter}) {names[model]}: {event}, {example_lead} h', xlabel='Depth k', ylabel='Skill versus IFS-ENS (%)', xticks=depth_ticks)
        ax.grid(True, alpha=.25)
    axes[0, 0].legend(frameon=True, facecolor='white', framealpha=.95, edgecolor='white', fontsize=8.5, loc='upper left')
    ax = axes[1, 0]
    for model in all_models:
        s = data[(data.truth == primary_truth) & (data.event == 'heat') & (data.leadtime == example_lead) & (data.model == model)].sort_values('depth')
        ax.plot(s.depth, s.probability_ratio, 'o-', color=colors[model], lw=2, ms=4, label=names[model])
    ax.axhline(1, color='.3', ls='--', lw=1)
    ax.set(title='(c) Heat occurrence ratio', xlabel='Depth k', ylabel='Ratio Q', xticks=depth_ticks)
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc='upper left')
    ax = axes[1, 1]
    c = data[
        (data.truth == primary_truth) & (data.event == 'heat')
        & (data.leadtime == example_lead) & (data.depth == 0)
    ].set_index('model')
    contribution_fields = ['rtwcrps_record_contribution', 'rtwcrps_nonrecord_contribution']
    if not set(contribution_fields).issubset(c.columns):

        path = results_root / f'heat_score_contributions_{year}_{surface}_{primary_truth}_{example_lead:03d}h.csv'
        saved = pd.read_csv(path)
        if saved.model.duplicated().any() or not set(all_models).issubset(saved.model):
            raise ValueError(f'Missing or duplicated models in {path}')
        saved = saved.set_index('model').loc[all_models]
        for column, expected in [('truth', primary_truth), ('year', year), ('record_tag', record_tag),
                                 ('event', 'heat'), ('leadtime', example_lead), ('landsea', surface), ('depth', 0)]:
            if not (saved[column] == expected).all():
                raise ValueError(f'Contribution results differ in {column}: {path}')
        for field in support + ['rtwcrps']:
            if not np.allclose(saved[field], c.loc[all_models, field], rtol=1e-10, atol=1e-12):
                raise ValueError(f'Contribution results differ from depth results in {field}.')
        c[contribution_fields] = saved[contribution_fields]
    if not np.isfinite(c[contribution_fields]).all().all() or not np.allclose(
        c.rtwcrps, c[contribution_fields].sum(axis=1), rtol=1e-10, atol=1e-12,
    ):
        raise ValueError('Exceedance/non-exceedance contributions must sum to all-case rtwCRPS.')
    for j, model in enumerate(models):
        for side, field, color in [(-.18, 'rtwcrps_record_contribution', '#9f4b3d'), (.18, 'rtwcrps_nonrecord_contribution', '#087eae')]:
            v = 1e4 * (c.loc[reference, field] - c.loc[model, field])
            ax.bar(j + side, v, width=.32, color=color,
                   label=('Observed records' if side < 0 else 'Other cases') if j == 0 else None)
    ax.axhline(0, color='.3', lw=1)
    ax.set(title='(d) Heat score contributions', ylabel='Score reduction (10⁻⁴ K)', xticks=range(len(models)), xticklabels=[names[x] for x in models])
    ax.margins(y=.25)
    ax.grid(True, axis='y', alpha=.25)
    ax.legend(frameon=False, fontsize=9)
    for ext in formats:
        fig.savefig(out / ('selection_and_calibration.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    print('Saved selection_and_calibration')
else:
    print('Skipping selection_and_calibration: requires FGN and AWG.')


#Change the common reference
if {'ERA5', 'HRES_fc0'}.issubset(truths):
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.8))
    fig.subplots_adjust(left=.08, right=.99, top=.91, bottom=.09, hspace=.35, wspace=.25)
    for row, metric in enumerate(['crps_all', 'rtwcrps']):
        for col, event in enumerate(events):
            ax = axes[row, col]
            sub = data[(data.event == event) & (data.leadtime == example_lead) & (data.depth == 0)]
            for model in models:
                s = sub[sub.model == model].set_index('truth').reindex(['ERA5', 'HRES_fc0'])
                ax.plot([0, 1], s[metric + '_skill'], 'o-', color=colors[model], lw=2, ms=6, label=names[model])
            ax.axhline(0, color='.35', lw=1, ls='--')
            ax.set_xticks([0, 1], ['ERA5', 'HRES-fc0'])
            ax.set_xlim(-.16, 1.16)
            ax.grid(True, alpha=.25)
            ax.set_title(f'({chr(97+row*3+col)}) {event.capitalize()}, {example_lead} h')
            if col == 0: ax.set_ylabel(('Overall CRPS' if row == 0 else 'Record rtwCRPS') + ' skill (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=min(4, len(models)), frameon=False, bbox_to_anchor=(.54, 1.005))
    for ext in formats:
        fig.savefig(out / ('truth_switch.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    print('Saved truth_switch')
else:
    print('Skipping truth_switch: requires ERA5 and HRES_fc0 results.')

# Ordinary CRPS skill
fig, axes = plt.subplots(len(truths), 3, figsize=(9.2, 3.3 * len(truths)), squeeze=False)
fig.subplots_adjust(left=.09, right=.99, top=.91, bottom=.10, hspace=.35, wspace=.25)
for row, truth in enumerate(truths):
    for col, event in enumerate(events):
        ax = axes[row, col]
        for model in models:
            s = data[(data.truth == truth) & (data.event == event) & (data.depth == 0) & (data.model == model)].sort_values('leadtime')
            ax.plot(s.leadtime, s.crps_all_skill, 'o-', color=colors[model], lw=2, ms=4, label=names[model])
        ax.axhline(0, color='.35', ls='--', lw=1)
        ax.set(title=f'{event.capitalize()} | {truth.replace("_", "-")}', xticks=leads)
        ax.grid(True, alpha=.25)
        if row == len(truths) - 1: ax.set_xlabel('Lead time (h)')
        if col == 0: ax.set_ylabel('Overall CRPS skill (%)')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=min(4, len(models)), frameon=False, bbox_to_anchor=(.54, 1.005))
for ext in formats:
    fig.savefig(out / ('overall_skill.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=.06)
plt.close(fig)
print('Saved overall_skill')

# Occurrence ratios
for truth in truths:
    metric = 'probability_ratio'
    fig, axes = plt.subplots(3, len(leads), figsize=(max(6.4, 2.3 * len(leads)), 8.1), sharex=True, squeeze=False)
    fig.subplots_adjust(left=.07, right=.99, top=.92, bottom=.09, hspace=.39, wspace=.25)
    for row, event in enumerate(events):
        for col, lead in enumerate(leads):
            ax = axes[row, col]
            for model in all_models:
                s = data[(data.truth == truth) & (data.event == event) & (data.leadtime == lead) & (data.model == model)].sort_values('depth')
                y = s[metric]
                ax.plot(s.depth, y, 'o-', color=colors[model], lw=1.8, ms=3.5, label=names[model])
            ax.axhline(1, color='.35', lw=1, ls='--')
            ax.grid(True, alpha=.25)
            ax.set_title(f'{event.capitalize()}, {lead} h', fontsize=11)
            ax.set_xticks(depth_ticks)

            ratios = data[(data.truth == truth) & (data.event == event)].probability_ratio
            if (ratios == 0).any():
                ax.set_yscale('symlog', linthresh=0.01)
                ax.set_ylim(0, max(1, ratios.max()) * 1.15)
            else:
                ax.set_yscale('log')
                ax.set_ylim(min(1, ratios.min()) / 1.15, max(1, ratios.max()) * 1.15)
            ax.yaxis.set_major_locator(FixedLocator([0,.01,.1,.2,.5,1,2,5,10,20,50,100]))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if row == 2: ax.set_xlabel('Depth k')
            if col == 0: ax.set_ylabel('Probability ratio Q')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=min(4, len(all_models)), frameon=False, bbox_to_anchor=(.54, 1.005))
    for ext in formats:
        fig.savefig(out / f'{metric}_{truth}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    print(f'Saved {metric}_{truth}')

#Save the counts and a small score summary 
counts = data[(data.model == reference) & (data.leadtime == leads[0])][['truth', 'event', 'depth', 'record_count', 'observed_probability', 'valid_count']]
counts.to_csv(out / 'exceedance_counts.csv', index=False)
ai = data[(data.model != reference) & data.depth.isin([0, .5, 1, 2])]
audit = {}
for metric in ['rtwcrps', 'brier']:
    audit[metric] = {'positive_comparisons': int((ai[metric+'_skill'] > 0).sum()),
        'total_comparisons': len(ai)}
(out / 'score_audit.json').write_text(json.dumps(audit, indent=2))
print('Created native figures and skill/count tables in', out)
print(json.dumps(audit, indent=2))


# absolute probabilistic scores from the same depth results
panels = [
    ("crps_all", "(a) Fair CRPS: all cases"),
    ("crps_record", "(b) Fair CRPS: observed records"),
    ("rtwcrps", "(c) Fair rtwCRPS: all cases"),
    ("brier", "(d) Fair Brier: all cases"),
]
for truth in truths:
    for event in events:
        selected = data[(data.truth == truth) & (data.event == event) & (data.depth == 0)]
        unit = "m/s" if event == "wind" else "K"
        surface_label = "all surfaces" if surface == "all" else f"{surface} only"
        title = f"{event.capitalize()} records (ERA5 {record_tag.replace('_', '-')}, {surface_label})"
        if truth != "ERA5":
            title += "\nVerification: HRES-fc0"
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), layout="constrained")
        fig.suptitle(title, fontsize=15)

        for ax, (metric, label) in zip(axes.flat, panels):
            for model in all_models:
                scores = selected[selected.model == model].sort_values("leadtime")
                ax.plot(scores.leadtime, scores[metric], "o-",
                        color=colors[model], label=names[model], linewidth=2, markersize=4)
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Lead time (h)")
            ax.set_ylabel("Brier score" if metric == "brier" else f"Score ({unit})")
            ax.set_xticks(leads)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)

            ax.set_ylim(bottom=min(0, selected[metric].min() * 1.05))
        axes[0, 0].legend(frameon=False, fontsize=10)
        for ext in formats:
            fig.savefig(out / f"{event}_scores_{truth}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {event}_scores_{truth}")

# Extended depths: supplementary
for truth in truths:
    fig, axes = plt.subplots(3, len(leads), figsize=(max(6.4, 2.3 * len(leads)), 8.1), squeeze=False)
    fig.subplots_adjust(left=.075, right=.99, top=.92, bottom=.09, hspace=.39, wspace=.22)
    for row, event in enumerate(events):
        subset = full_depth_data[(full_depth_data.truth == truth) & (full_depth_data.event == event) & (full_depth_data.model != reference)]
        lower = 10 * np.floor(min(0, subset.rtwcrps_skill.min()) / 10) - 3
        upper = 10 * np.ceil(max(0, subset.rtwcrps_skill.max()) / 10) + 3
        for col, lead in enumerate(leads):
            ax = axes[row, col]
            for model in models:
                s = subset[(subset.model == model) & (subset.leadtime == lead)].sort_values('depth')
                ax.plot(s.depth, s.rtwcrps_skill, 'o-', lw=1.8, ms=3.5, color=colors[model], label=names[model])
            ax.axhline(0, color='.35', ls='--', lw=1)
            ax.grid(True, alpha=.25)
            ax.set_ylim(lower, upper)
            ax.set_xticks(np.arange(0, subset.depth.max()+.01, 1.0))
            ax.set_title(f'{event.capitalize()}, {lead} h', fontsize=11)
            if row == 2: ax.set_xlabel('Threshold depth k')
            if col == 0: ax.set_ylabel('rtwCRPS skill (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=len(models), frameon=False)
    for ext in formats:
        fig.savefig(out / f'extended_rtwcrps_{truth}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)

full_depth_data[(full_depth_data.model == reference) & (full_depth_data.leadtime == leads[0])][
    ['truth', 'event', 'depth', 'record_count', 'observed_probability', 'valid_count']
].to_csv(out / 'exceedance_counts_all_depths.csv', index=False)
print('Done: main figures to k=2, extended-depth figures, and reproducible tables.')
