#!/usr/bin/env python3
"""Create separate true-versus-predicted curve figures for Experiments 3A-3F."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

EXPERIMENTS = {
    "3A": ("static_metabolic_control", "3A Static Metabolic Control"),
    "3B": ("state_machine_dfba", "3B State-Machine-Regulated dFBA"),
    "3C": ("useful_reporter_supervision", "3C Useful Reporter Supervision"),
    "3D": ("partial_reporter_coverage", "3D Incomplete Reporter Coverage"),
    "3E": ("irrelevant_er_reporter", "3E Causally Irrelevant ER Reporter"),
    "3F": ("model_baseline_comparison", "3F State-Space versus Direct Curve Prediction"),
}

DISPLAY_MODELS = {
    "direct_multi_output_mlp": "direct MLP",
    "coordinate_conditioned_mlp": "coordinate MLP",
    "basis_curve_decoder": "basis decoder",
    "product_only_state_space": "product-only state space",
    "oxidative_reporter_state_space": "oxidative reporter state space",
    "atp_reporter_state_space": "ATP reporter state space",
    "ox_atp_reporter_state_space": "oxidative+ATP reporter state space",
    "ox_atp_er_state_space": "useful+ER state space",
    "adaptive_ox_atp_er_state_space": "adaptive useful+ER state space",
    "er_only_state_space": "ER-only state space",
    "shuffled_reporter_control": "shuffled reporter control",
    "shuffled_er_control": "shuffled ER control",
    "random_smooth_aux_control": "random smooth aux control",
    "oracle_causal_burden_state_space": "causal-burden oracle",
    "oracle_all_burdens_state_space": "all-burden oracle",
}

SPLITS = ("interpolation", "heldout_combination", "extrapolation")
SPLIT_LABELS = {
    "interpolation": "interpolation",
    "heldout_combination": "held-out combination",
    "extrapolation": "numerical extrapolation",
}


def model_label(model: str) -> str:
    return DISPLAY_MODELS.get(model, model.replace("_", " "))


def selected_model(metrics: pd.DataFrame, exp: str) -> str:
    sub = metrics[
        (metrics["experiment"] == exp)
        & (metrics["split"] == "heldout_combination")
        & (~metrics["model"].astype(str).str.contains("oracle"))
    ]
    return str(sub.groupby("model")["trajectory_rmse"].mean().sort_values().index[0])


def choose_representative_culture(predictions: pd.DataFrame, exp: str, model: str, split: str) -> str | None:
    sub = predictions[
        (predictions["experiment"] == exp)
        & (predictions["model"] == model)
        & (predictions["split"] == split)
    ]
    if sub.empty:
        return None
    by_curve = (
        sub.groupby("culture_id", as_index=False)
        .agg(trajectory_rmse=("trajectory_rmse", "first"))
        .sort_values("trajectory_rmse")
        .reset_index(drop=True)
    )
    return str(by_curve.iloc[len(by_curve) // 2]["culture_id"])


def add_curve_panel(
    body: list[str],
    *,
    predictions: pd.DataFrame,
    exp: str,
    model: str,
    split: str,
    x0: float,
    y0: float,
    w: float,
    h: float,
) -> None:
    cid = choose_representative_culture(predictions, exp, model, split)
    body.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#fbfbfb" stroke="#ccc"/>')
    body.append(fixed.svg_text(x0 + w / 2, y0 - 9, SPLIT_LABELS[split], 12, "bold"))
    if cid is None:
        body.append(fixed.svg_text(x0 + w / 2, y0 + h / 2, "no saved prediction", 11))
        return
    g = predictions[
        (predictions["experiment"] == exp)
        & (predictions["model"] == model)
        & (predictions["split"] == split)
        & (predictions["culture_id"] == cid)
    ].sort_values("time")
    xlim = (float(g["time"].min()), float(g["time"].max()))
    ymax = float(max(g["B_total_true"].max(), g["B_total_pred"].max()) * 1.10)
    ylim = (0.0, max(ymax, 1e-8))
    plot_x = x0 + 42
    plot_y = y0 + 27
    plot_w = w - 78
    plot_h = h - 74
    for frac in (0.25, 0.50, 0.75):
        gy = plot_y + plot_h * frac
        body.append(f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{gy:.1f}" stroke="#e2e2e2" stroke-width="1"/>')
    body.append(fixed.plot_polyline_bounds(g["time"], g["B_total_true"], plot_x, plot_y, plot_w, plot_h, xlim, ylim, "#111", 2.4))
    body.append(fixed.plot_polyline_bounds(g["time"], g["B_total_pred"], plot_x, plot_y, plot_w, plot_h, xlim, ylim, "#2f6f9e", 2.0, "7 4"))
    env = g[["temperature", "pH", "DO"]].iloc[0]
    rmse = float(g["trajectory_rmse"].iloc[0])
    body.append(fixed.svg_text(x0 + 13, y0 + h - 35, f"T={env.temperature:.0f}, pH={env.pH:.1f}, DO={env.DO:.0f}", 9, anchor="start"))
    body.append(fixed.svg_text(x0 + 13, y0 + h - 18, f"trajectory RMSE {rmse:.4f}", 9, anchor="start"))
    body.append(fixed.svg_text(x0 + w / 2, y0 + h - 3, "time", 9))
    body.append(fixed.svg_text(x0 + 17, y0 + h / 2, "B(t)", 9))


def make_prediction_figure(exp: str, metrics: pd.DataFrame, predictions: pd.DataFrame) -> Path:
    slug, title = EXPERIMENTS[exp]
    model = selected_model(metrics, exp)
    body = [
        fixed.svg_text(540, 30, f"{title}: predicted versus true", 18, "bold"),
        fixed.svg_text(540, 53, f"Selected learned model: {model_label(model)}", 11),
        fixed.svg_text(878, 53, "solid=true   dashed=predicted", 10, anchor="end"),
    ]
    panel_w = 315
    panel_h = 225
    for idx, split in enumerate(SPLITS):
        add_curve_panel(
            body,
            predictions=predictions,
            exp=exp,
            model=model,
            split=split,
            x0=54 + idx * 337,
            y0=95,
            w=panel_w,
            h=panel_h,
        )
    path = FIGURES / f"dfba_pred_vs_true_{exp.lower()}_{slug}.svg"
    fixed.save_svg(path, 1080, 365, "\n".join(body))
    return path


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    metrics = pd.read_csv(DATA / "dfba_canonical_metrics.csv")
    predictions = pd.read_csv(DATA / "dfba_model_predictions.csv")
    for exp in EXPERIMENTS:
        print(make_prediction_figure(exp, metrics, predictions))


if __name__ == "__main__":
    main()
