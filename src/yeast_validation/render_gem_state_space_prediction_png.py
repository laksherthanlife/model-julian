#!/usr/bin/env python3
"""Render the GEM state-space prediction comparison directly to PNG with Matplotlib."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SEED = 11
MODELS = [
    ("polynomial_environment_pca", "Best direct env-to-PCA", "#b65d3b"),
    ("mechanistic_rate_state_space", "Best neural SSM", "#2f6f9f"),
    ("product_only_state_space", "Product-only SSM", "#558b2f"),
]


def clean_culture_id(cid: str) -> str:
    return (
        cid.replace("dynamic_congestion_feedback_", "")
        .replace("_", ", ")
        .replace("pH", "pH ")
        .replace("DO", "DO ")
    )


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    predictions = pd.read_csv(DATA / "gem_state_space_model_predictions.csv")
    metrics = pd.read_csv(DATA / "gem_state_space_metrics.csv")

    held = predictions[(predictions["split"].eq("heldout_combination")) & (predictions["model_seed"].eq(SEED))]
    culture_ids = list(held["culture_id"].drop_duplicates().head(4))
    held_metrics = (
        metrics[metrics["split"].eq("heldout_combination")]
        .groupby("model")["trajectory_normalized_rmse"]
        .mean()
        .to_dict()
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=220)
    axes = axes.ravel()

    for ax, cid in zip(axes, culture_ids):
        base = held[(held["culture_id"].eq(cid)) & (held["model"].eq(MODELS[0][0]))].sort_values("time")
        ax.plot(base["time"], base["B_true"], color="#111111", lw=2.4, label="True Yeast9-generated titer")
        for model, label, color in MODELS:
            g = held[(held["culture_id"].eq(cid)) & (held["model"].eq(model))].sort_values("time")
            ax.plot(g["time"], g["B_pred"], color=color, lw=2.0, label=f"{label} ({held_metrics[model]:.4f})")
        ax.set_title(clean_culture_id(cid), fontsize=10, pad=8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Accumulated beta-carotene, B(t)")
        ax.grid(True, color="#d9dee4", linewidth=0.7)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=False,
        fontsize=10,
        title="Heldout-combination trajectory-normalized RMSE in parentheses",
        title_fontsize=10,
    )
    fig.suptitle("Held-Out Beta-Carotene Trajectory Predictions", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.948,
        "Expanded 125-environment real-Yeast9 grid; deployment input restricted to fixed [T, pH, DO]",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.5,
        0.925,
        "Conclusion: the neural state-space model follows the broad shape but does not beat the direct polynomial env-to-PCA baseline.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#444444",
    )
    fig.text(
        0.012,
        0.012,
        "Source: data/gem_state_space_model_predictions.csv and data/gem_state_space_metrics.csv",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.84], h_pad=2.2, w_pad=2.0)
    out = FIGURES / "gem_state_space_true_vs_predicted_labeled_matplotlib.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
