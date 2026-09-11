#!/usr/bin/env python3
"""Create a labeled supervisor-facing prediction figure for the GEM SSM diagnostic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_fixed_environment_validation as fixed
from run_gem_state_space_validation import svg_polyline


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SEED = 11
MODELS = [
    ("polynomial_environment_pca", "Best direct env-to-PCA", "#b65d3b"),
    ("mechanistic_rate_state_space", "Best neural SSM", "#2f6f9f"),
    ("product_only_state_space", "Product-only SSM", "#558b2f"),
]
W, H = 1280, 820


def axis(x0: int, y0: int, w: int, h: int, xmax: float, ymax: float) -> list[str]:
    body = [
        f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#333" stroke-width="1.2"/>',
    ]
    for frac in [0.25, 0.50, 0.75]:
        y = y0 + h - frac * h
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="#d9dee4" stroke-width="0.8"/>')
        body.append(fixed.svg_text(x0 - 8, y + 4, f"{frac*ymax:.3f}", 9, anchor="end", color="#555"))
    for frac in [0.0, 0.5, 1.0]:
        x = x0 + frac * w
        body.append(f'<line x1="{x:.1f}" y1="{y0+h}" x2="{x:.1f}" y2="{y0+h+5}" stroke="#333" stroke-width="1"/>')
        body.append(fixed.svg_text(x, y0 + h + 18, f"{frac*xmax:.1f}", 9, color="#555"))
    return body


def clean_culture_id(cid: str) -> str:
    return cid.replace("dynamic_congestion_feedback_", "").replace("pH", "pH ")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    pred = pd.read_csv(DATA / "gem_state_space_model_predictions.csv")
    metrics = pd.read_csv(DATA / "gem_state_space_metrics.csv")
    held = pred[(pred["split"].eq("heldout_combination")) & (pred["model_seed"].eq(SEED))]
    culture_ids = list(held["culture_id"].drop_duplicates().head(4))
    summary = (
        metrics[metrics["split"].eq("heldout_combination")]
        .groupby("model")["trajectory_normalized_rmse"]
        .mean()
        .to_dict()
    )
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        fixed.svg_text(W / 2, 34, "Held-Out Beta-Carotene Trajectory Predictions", 22, "bold"),
        fixed.svg_text(
            W / 2,
            60,
            "Expanded 125-environment real-Yeast9 grid; deployment input restricted to fixed [T, pH, DO]",
            12,
            color="#444",
        ),
        fixed.svg_text(
            W / 2,
            82,
            "Conclusion: neural state-space matches the broad shape but does not beat the direct polynomial env-to-PCA baseline.",
            12,
            "bold",
            color="#444",
        ),
    ]
    lx, ly = 78, 106
    body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+42}" y2="{ly}" stroke="#111" stroke-width="2"/>')
    body.append(fixed.svg_text(lx + 52, ly + 4, "True Yeast9-generated titer", 11, anchor="start"))
    for i, (model, label, color) in enumerate(MODELS):
        y = ly + 22 * (i + 1)
        body.append(f'<line x1="{lx}" y1="{y}" x2="{lx+42}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        rmse = summary.get(model, float("nan"))
        body.append(fixed.svg_text(lx + 52, y + 4, f"{label}: heldout normalized RMSE {rmse:.4f}", 11, anchor="start"))

    for i, cid in enumerate(culture_ids):
        x0 = 82 + (i % 2) * 595
        y0 = 205 + (i // 2) * 285
        plot_w, plot_h = 470, 185
        base = held[(held["culture_id"].eq(cid)) & (held["model"].eq(MODELS[0][0]))].sort_values("time")
        model_rows = held[(held["culture_id"].eq(cid)) & (held["model"].isin([m[0] for m in MODELS]))]
        ymax = max(float(base["B_true"].max()), float(model_rows["B_pred"].max())) * 1.08
        xmax = float(base["time"].max())
        body.append(f'<rect x="{x0-18}" y="{y0-34}" width="{plot_w+72}" height="{plot_h+92}" rx="4" fill="#fbfbfb" stroke="#c8cdd2"/>')
        body.append(fixed.svg_text(x0 + plot_w / 2, y0 - 14, clean_culture_id(cid), 12, "bold"))
        body.extend(axis(x0, y0, plot_w, plot_h, xmax, ymax))
        body.append(svg_polyline(base["time"], base["B_true"], x0, y0, plot_w, plot_h, (0, xmax), (0, ymax), "#111111", 2.2))
        for model, _label, color in MODELS:
            g = held[(held["culture_id"].eq(cid)) & (held["model"].eq(model))].sort_values("time")
            body.append(svg_polyline(g["time"], g["B_pred"], x0, y0, plot_w, plot_h, (0, xmax), (0, ymax), color, 1.8))
        body.append(fixed.svg_text(x0 + plot_w / 2, y0 + plot_h + 42, "time", 10, color="#555"))
        body.append(fixed.svg_text(x0 - 48, y0 + plot_h / 2, "B(t)", 10, color="#555"))

    body.append(fixed.svg_text(80, H - 30, "Files: data/gem_state_space_model_predictions.csv and data/gem_state_space_metrics.csv", 10, anchor="start", color="#666"))
    body.append("</svg>")
    out = FIGURES / "gem_state_space_true_vs_predicted_labeled.svg"
    out.write_text("\n".join(body), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
