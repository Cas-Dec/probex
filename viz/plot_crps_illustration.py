"""Illustrate CRPS and record-threshold weighted CRPS with Matplotlib.

Place this script in probex/viz/ and run:
    python viz/plot_crps_illustration.py

Forecast values are denoted by x-hat; verifying values by x.
The scores use the continuous Gaussian distribution, not a finite ensemble.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


mean = 41.0
sd = 1.0
record = 40.0
observed_record = 43.0
observed_nonrecord = 38.0

root = Path(__file__).resolve().parents[1]
out = root / "figures"
out.mkdir(parents=True, exist_ok=True)

blue = "#087eae"
gray = "#aeb8c1"
ink = "#202d3a"
record_color = "#6a737c"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 11, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

assert sd > 0 and observed_nonrecord < record < observed_record
forecast = norm(loc=mean, scale=sd)
left = min(mean - 4 * sd, observed_nonrecord - 2)
right = max(mean + 4 * sd, observed_record + 2)

jumps = np.array([record, observed_record, observed_nonrecord])
z = np.unique(np.r_[np.linspace(left, right, 6000), jumps, jumps - 1e-8])
F = forecast.cdf(z)
H_record = (z >= observed_record).astype(float)
F_clipped = np.where(z < record, 0.0, F)
probability_below = forecast.cdf(record)
probability_above = forecast.sf(record)


scores = []
for observation in [observed_record, observed_nonrecord]:
    breaks = sorted(set([-np.inf, record, observation, np.inf]))
    whole = 0.0
    tail = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        area = quad(
            lambda value: (forecast.cdf(value) - float(observation <= value)) ** 2,
            a, b, epsabs=1e-10,
        )[0]
        whole += area
        if a >= record:
            tail += area

    # Check against the analytic Gaussian CRPS.
    standardized = (observation - mean) / sd
    analytic = sd * (
        standardized * (2 * norm.cdf(standardized) - 1)
        + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi)
    )
    assert np.isclose(whole, analytic, atol=1e-9)

    # Check the equivalence with CRPS after clipping BOTH forecasts and verification.
    clipped_observation = max(observation, record)
    clipped_score = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        clipped_score += quad(
            lambda value: (
                (0.0 if value < record else forecast.cdf(value))
                - float(clipped_observation <= value)
            ) ** 2,
            a, b, epsabs=1e-10,
        )[0]
    assert np.isclose(tail, clipped_score, atol=1e-9)
    scores.append((whole, tail))

assert scores[1][1] > 0


fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.15,
                    top=0.94, wspace=0.24, hspace=0.49)

for ax in axes.flat:
    ax.set_xlim(left, right)
    ax.set_ylim(-0.025, 1.06)
    ax.set_xticks(np.arange(np.ceil(left), right + 0.1, 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(True, alpha=0.25)
    ax.axvline(record, color=record_color, linestyle=":", linewidth=1.6)
    ax.text(record, 1.025, "$R$", color=record_color,
            ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Temperature (°C)")

ax = axes[0, 0]
ax.set_title("(a) Original forecast", pad=14)
ax.plot(z, F, color=blue, linewidth=2.4)
ax.plot(z, H_record, color=ink, linestyle="--", linewidth=1.5)
ax.set_ylabel("Cumulative probability")
ax.legend(handles=[
    Line2D([0], [0], color=blue, linewidth=2.4, label="Forecast CDF"),
    Line2D([0], [0], color=ink, linestyle="--", label="Verification"),
], loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.98)
ax.text(
    0.04, 0.40,
    rf"$\hat{{x}}\sim\mathcal{{N}}({mean:g}, {sd**2:g})$"
    + f"\n$R={record:g}$°C\n$x={observed_record:g}$°C",
    transform=ax.transAxes, fontsize=11, color=ink, linespacing=1.6,
)

ax = axes[0, 1]
ax.set_title("(b) Clipping at the record", pad=14)
ax.plot(z, F_clipped, color=blue, linewidth=2.4)
ax.plot(z, H_record, color=ink, linestyle="--", linewidth=1.5)
ax.set_ylabel("Cumulative probability")
ax.annotate(
    f"{probability_below:.0%} moved\nto the record",
    xy=(record, probability_below * 0.55), xytext=(left + 0.3, 0.42),
    fontsize=11, color=ink, linespacing=1.5,
    arrowprops=dict(arrowstyle="->", color=record_color, lw=1.0),
)
ax.text(0.98, 0.19, "Tail\nunchanged", transform=ax.transAxes,
        ha="right", color=blue, fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", pad=2))

for ax, observation, score, letter, title in zip(
    axes[1], [observed_record, observed_nonrecord], scores, ["c", "d"],
    ["Record exceeded", "Record not exceeded"],
):
    error = (F - (z >= observation).astype(float)) ** 2
    below = z <= record
    above = z >= record
    ax.fill_between(z[below], 0, error[below], color=gray, alpha=0.67)
    ax.fill_between(z[above], 0, error[above], color=blue, alpha=0.35)
    ax.plot(z[below], error[below], color="#7b8793", linewidth=1.7)
    ax.plot(z[above], error[above], color=blue, linewidth=2.0)
    ax.set_title(f"({letter}) {title}: $x={observation:g}$°C", pad=14)
    ax.set_ylabel("Squared CDF error")
    # Put the non-record scores on the right so the text does not hide its peak.
    text_x = 0.04 if letter == "c" else 0.96
    text_align = "left" if letter == "c" else "right"
    score_text = (
        f"CRPS = {score[0]:.3f} K\n"
        f"rtwCRPS = {score[1]:.3f} K"
    )
    ax.text(
        text_x, 0.92, score_text,
        transform=ax.transAxes, va="top", ha=text_align, color=ink, fontsize=11,
        linespacing=1.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.93, pad=3),
    )

axes[1, 1].annotate(
    "Forecast tail still\nincurs a penalty",
    xy=(record + 0.7, forecast.sf(record + 0.7) ** 2),
    xytext=(record + 1.3, 0.44), fontsize=11, color=blue,
    linespacing=1.5,
    arrowprops=dict(arrowstyle="->", color=blue, lw=1.1),
)

fig.legend(handles=[
    Patch(facecolor=gray, alpha=0.67, label="Below record: CRPS only"),
    Patch(facecolor=blue, alpha=0.35, label="Beyond record: CRPS and rtwCRPS"),
], loc="lower center", bbox_to_anchor=(0.535, 0.035),
   ncol=2, frameon=False, columnspacing=2.0)

for extension in ["pdf", "png", "svg"]:
    path = out / f"crps_rtwcrps_schematic.{extension}"
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")
plt.close(fig)

print(f"Forecast probability above the record: {probability_above:.3f}")
for observation, (whole, tail) in zip([observed_record, observed_nonrecord], scores):
    print(f"x={observation:g}°C: CRPS={whole:.6f} K; rtwCRPS={tail:.6f} K")
