#!/usr/bin/env python3
"""Decide whether the reduced dFBA surrogate is statistically too easy."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
EPS = 1e-9
PRIMARY_EXPERIMENT = "3F"
PRIMARY_SPLIT = "heldout_combination"
STATE_SPACE_MODEL = "product_only_state_space"
COORD_MODEL = "coordinate_conditioned_mlp"


def pivot_curves(traj: pd.DataFrame, experiment: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sub = traj[traj["experiment"] == experiment]
    ids = sorted(sub["culture_id"].unique())
    t = np.sort(sub["time"].unique())
    env, curves, splits, meta = [], [], [], []
    for cid in ids:
        g = sub[sub["culture_id"] == cid].sort_values("time")
        env.append(g[["temperature", "pH", "DO"]].iloc[0].to_numpy(float))
        curves.append(g["B_total_true"].to_numpy(float))
        splits.append(str(g["split"].iloc[0]))
        meta.append(g[["dataset_seed", "environment_id", "culture_id", "replicate", "split"]].iloc[0].to_dict())
    return ids, np.asarray(env), np.asarray(curves), t, np.asarray(splits), pd.DataFrame(meta)


def curve_rmse(a, b, axis=None):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def pairwise_curve_rmse(curves: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            vals.append(float(curve_rmse(curves[i], curves[j])))
    return np.asarray(vals)


def trajectory_scales(curves: np.ndarray, env: np.ndarray, splits: np.ndarray, t: np.ndarray, split: str) -> dict[str, float]:
    mask = splits == split
    c = curves[mask]
    e = env[mask]
    train = curves[splits == "train"]
    out = {
        "range_scale": float(max(c.max() - c.min(), EPS)),
        "std_scale": float(max(c.std(), EPS)),
        "mean_final_titer": float(max(np.mean(c[:, -1]), EPS)),
        "median_trajectory_amplitude": float(max(np.median(c.max(axis=1) - c.min(axis=1)), EPS)),
    }
    pairwise = pairwise_curve_rmse(c)
    out["median_distinct_heldout_trajectory_difference"] = float(max(np.median(pairwise), EPS)) if len(pairwise) else EPS
    if len(e) > 1:
        e_norm = (e - env[splits == "train"].mean(axis=0)) / (env[splits == "train"].std(axis=0) + EPS)
        nearest = []
        for i in range(len(e)):
            d = np.sqrt(np.sum((e_norm - e_norm[i]) ** 2, axis=1))
            d[i] = np.inf
            j = int(np.argmin(d))
            nearest.append(float(curve_rmse(c[i], c[j])))
        out["median_neighboring_environment_difference"] = float(max(np.median(nearest), EPS))
    else:
        out["median_neighboring_environment_difference"] = EPS
    return out


def normalized_error_rows(predictions: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in sorted(predictions["experiment"].unique()):
        ids, env, curves, t, splits, _meta = pivot_curves(traj, exp)
        truth_map = {cid: curves[i] for i, cid in enumerate(ids)}
        for split in sorted(predictions[predictions["experiment"] == exp]["split"].unique()):
            scales = trajectory_scales(curves, env, splits, t, split)
            for model, g in predictions[(predictions["experiment"] == exp) & (predictions["split"] == split)].groupby("model"):
                pred_curves, true_curves = [], []
                for cid, cg in g.groupby("culture_id"):
                    pred_curves.append(cg.sort_values("time")["B_total_pred"].to_numpy(float))
                    true_curves.append(truth_map[cid])
                pred_arr = np.asarray(pred_curves)
                true_arr = np.asarray(true_curves)
                rmse_i = curve_rmse(pred_arr, true_arr, axis=1)
                amp_i = np.maximum(true_arr.max(axis=1) - true_arr.min(axis=1), EPS)
                raw = float(curve_rmse(pred_arr, true_arr))
                rows.append(
                    {
                        "experiment": exp,
                        "model": model,
                        "split": split,
                        "raw_rmse": raw,
                        "nrmse_range": raw / scales["range_scale"],
                        "nrmse_std": raw / scales["std_scale"],
                        "nrmse_trajectory": float(np.mean(rmse_i / amp_i)),
                        "rmse_over_mean_final_titer": raw / scales["mean_final_titer"],
                        "rmse_over_median_trajectory_amplitude": raw / scales["median_trajectory_amplitude"],
                        "rmse_over_median_neighboring_environment_difference": raw / scales["median_neighboring_environment_difference"],
                        "rmse_over_median_distinct_heldout_trajectory_difference": raw / scales["median_distinct_heldout_trajectory_difference"],
                        "n_cultures": int(len(pred_arr)),
                    }
                )
    return pd.DataFrame(rows)


def fit_pca(train_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_curves.mean(axis=0)
    centered = train_curves - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s**2
    explained = var / max(var.sum(), EPS)
    return mean, vt, explained


def pca_summary(traj: pd.DataFrame, experiment: str = PRIMARY_EXPERIMENT) -> tuple[pd.DataFrame, dict[str, object]]:
    ids, env, curves, t, splits, meta = pivot_curves(traj, experiment)
    train = curves[splits == "train"]
    mean, comps, explained = fit_pca(train)
    cum = np.cumsum(explained)
    rows = []
    for k in [1, 2, 3, 5, 10]:
        kk = min(k, len(cum))
        rows.append({"experiment": experiment, "summary_type": "cumulative_explained_variance", "component_count": k, "value": float(cum[kk - 1])})
    for target in [0.90, 0.95, 0.99, 0.999]:
        needed = int(np.searchsorted(cum, target) + 1) if len(cum) else 0
        rows.append({"experiment": experiment, "summary_type": f"components_for_{target}", "component_count": needed, "value": target})
    for split in ["validation", "interpolation", "heldout_combination", "extrapolation"]:
        mask = splits == split
        if not mask.any():
            continue
        max_k = min(10, comps.shape[0])
        coeff = (curves[mask] - mean) @ comps[:max_k].T
        recon = mean + coeff @ comps[:max_k]
        rows.append({"experiment": experiment, "summary_type": "reconstruction_rmse_10_components", "split": split, "component_count": max_k, "value": float(curve_rmse(recon, curves[mask]))})
    return pd.DataFrame(rows), {"ids": ids, "env": env, "curves": curves, "t": t, "splits": splits, "mean": mean, "components": comps, "explained": explained, "meta": meta}


def env_features(env: np.ndarray, polynomial: bool = False) -> np.ndarray:
    e = np.asarray(env, dtype=float)
    norm = np.column_stack([(e[:, 0] - 30.0) / 8.0, (e[:, 1] - 5.0) / 1.3, (e[:, 2] - 45.0) / 40.0])
    if not polynomial:
        return np.column_stack([np.ones(len(norm)), norm])
    a, b, c = norm[:, 0], norm[:, 1], norm[:, 2]
    return np.column_stack([np.ones(len(norm)), a, b, c, a * b, a * c, b * c, a**2, b**2, c**2])


def ridge_fit(X, Y, lam=1e-6):
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ Y)


def simple_baselines(traj: pd.DataFrame, experiment: str = PRIMARY_EXPERIMENT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids, env, curves, t, splits, meta = pivot_curves(traj, experiment)
    train = splits == "train"
    mean, comps, explained = fit_pca(curves[train])
    k = min(5, comps.shape[0])
    train_coeff = (curves[train] - mean) @ comps[:k].T
    coef_lin = ridge_fit(env_features(env[train], False), train_coeff)
    coef_poly = ridge_fit(env_features(env[train], True), train_coeff, lam=1e-4)
    train_env = env[train]
    train_curves = curves[train]
    train_scale = train_env.std(axis=0) + EPS
    predictions = []
    rows = []
    for model in ["mean_trajectory", "nearest_environment", "linear_env_to_pca", "polynomial_env_to_pca"]:
        pred = np.zeros_like(curves)
        if model == "mean_trajectory":
            pred[:] = mean
        elif model == "nearest_environment":
            train_norm = (train_env - train_env.mean(axis=0)) / train_scale
            all_norm = (env - train_env.mean(axis=0)) / train_scale
            for i, e in enumerate(all_norm):
                j = int(np.argmin(np.sqrt(np.sum((train_norm - e) ** 2, axis=1))))
                pred[i] = train_curves[j]
        elif model == "linear_env_to_pca":
            coeff = env_features(env, False) @ coef_lin
            pred = mean + coeff @ comps[:k]
        else:
            coeff = env_features(env, True) @ coef_poly
            pred = mean + coeff @ comps[:k]
        for split in ["validation", "interpolation", "heldout_combination", "extrapolation"]:
            mask = splits == split
            if not mask.any():
                continue
            scales = trajectory_scales(curves, env, splits, t, split)
            rmse_i = curve_rmse(pred[mask], curves[mask], axis=1)
            amp_i = np.maximum(curves[mask].max(axis=1) - curves[mask].min(axis=1), EPS)
            raw = float(curve_rmse(pred[mask], curves[mask]))
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "split": split,
                    "raw_rmse": raw,
                    "nrmse_range": raw / scales["range_scale"],
                    "nrmse_std": raw / scales["std_scale"],
                    "nrmse_trajectory": float(np.mean(rmse_i / amp_i)),
                    "pca_components": k if "pca" in model else 0,
                }
            )
        for i, cid in enumerate(ids):
            if splits[i] == "train":
                continue
            for j, tt in enumerate(t):
                predictions.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "culture_id": cid,
                        "split": splits[i],
                        "time": float(tt),
                        "B_total_true": float(curves[i, j]),
                        "B_total_pred": float(pred[i, j]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def seed_uncertainty(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = metrics[(metrics["experiment"] == PRIMARY_EXPERIMENT) & (metrics["split"].isin(["interpolation", "heldout_combination", "extrapolation"]))]
    for split in sorted(sub["split"].unique()):
        ss = sub[(sub["split"] == split) & (sub["model"] == STATE_SPACE_MODEL)]
        cm = sub[(sub["split"] == split) & (sub["model"] == COORD_MODEL)]
        paired = ss[["dataset_seed", "model_seed", "trajectory_rmse"]].merge(
            cm[["dataset_seed", "model_seed", "trajectory_rmse"]],
            on=["dataset_seed", "model_seed"],
            suffixes=("_state_space", "_coord_mlp"),
        )
        diffs = paired["trajectory_rmse_state_space"] - paired["trajectory_rmse_coord_mlp"]
        if len(diffs) == 0:
            continue
        rng = np.random.default_rng(123)
        boots = []
        for _ in range(1000):
            sample = diffs.to_numpy()[rng.integers(0, len(diffs), len(diffs))]
            boots.append(sample.mean())
        rows.append(
            {
                "experiment": PRIMARY_EXPERIMENT,
                "split": split,
                "comparison": f"{STATE_SPACE_MODEL}_minus_{COORD_MODEL}",
                "individual_paired_differences": ";".join(f"{x:.6f}" for x in diffs),
                "mean_difference": float(diffs.mean()),
                "std_difference": float(diffs.std(ddof=0)),
                "bootstrap_ci_low": float(np.percentile(boots, 2.5)),
                "bootstrap_ci_high": float(np.percentile(boots, 97.5)),
                "fraction_state_space_wins": float((diffs < 0).mean()),
                "fraction_coordinate_mlp_wins": float((diffs > 0).mean()),
                "n_pairs": int(len(diffs)),
            }
        )
    return pd.DataFrame(rows)


def decision_gate(norm: pd.DataFrame, simple: pd.DataFrame, pca: pd.DataFrame, uncertainty: pd.DataFrame) -> pd.DataFrame:
    held = norm[(norm["experiment"] == PRIMARY_EXPERIMENT) & (norm["split"] == PRIMARY_SPLIT)]
    coord = held[held["model"] == COORD_MODEL].iloc[0]
    ss = held[held["model"] == STATE_SPACE_MODEL].iloc[0]
    simple_held = simple[simple["split"] == PRIMARY_SPLIT].sort_values("nrmse_trajectory")
    best_simple = simple_held.iloc[0]
    pca_3 = pca[(pca["summary_type"] == "cumulative_explained_variance") & (pca["component_count"] == 3)]["value"].iloc[0]
    pca_5 = pca[(pca["summary_type"] == "cumulative_explained_variance") & (pca["component_count"] == 5)]["value"].iloc[0]
    unc = uncertainty[uncertainty["split"] == PRIMARY_SPLIT].iloc[0]
    rows = [
        ("coordinate_mlp_heldout_nrmse_trajectory_low", coord["nrmse_trajectory"], 0.02, bool(coord["nrmse_trajectory"] <= 0.02), f"coordinate MLP held-out trajectory NRMSE={coord['nrmse_trajectory']:.4f}"),
        ("state_space_heldout_nrmse_practically_small", ss["nrmse_trajectory"], 0.05, bool(ss["nrmse_trajectory"] <= 0.05), f"state-space held-out trajectory NRMSE={ss['nrmse_trajectory']:.4f}"),
        ("model_ranking_within_seed_uncertainty", abs(unc["mean_difference"]), max(abs(unc["std_difference"]), EPS), bool(abs(unc["mean_difference"]) <= max(abs(unc["std_difference"]), EPS)), f"mean paired diff={unc['mean_difference']:.4f}, std={unc['std_difference']:.4f}, n_pairs={unc['n_pairs']}"),
        ("simple_pca_baseline_close_to_neural", best_simple["nrmse_trajectory"], ss["nrmse_trajectory"] * 1.25, bool(best_simple["nrmse_trajectory"] <= ss["nrmse_trajectory"] * 1.25), f"best simple={best_simple['model']} with trajectory NRMSE={best_simple['nrmse_trajectory']:.4f}"),
        ("few_pca_components_explain_curve_family", pca_3, 0.99, bool(pca_3 >= 0.99 or pca_5 >= 0.999), f"3 PCs explain {pca_3:.4f}; 5 PCs explain {pca_5:.4f}"),
        ("true_vs_predicted_visually_near_identical_proxy", coord["nrmse_range"], 0.05, bool(coord["nrmse_range"] <= 0.05), f"coordinate MLP range NRMSE={coord['nrmse_range']:.4f}"),
        ("state_transitions_did_not_create_architecture_separation", ss["raw_rmse"] - coord["raw_rmse"], 0.0, bool(ss["raw_rmse"] > coord["raw_rmse"]), f"state-space RMSE={ss['raw_rmse']:.4f}; coord MLP RMSE={coord['raw_rmse']:.4f}"),
    ]
    out = pd.DataFrame(rows, columns=["criterion", "value", "threshold", "passed", "evidence"])
    out["decision"] = "move_to_gem" if out["passed"].mean() >= 5 / 7 else "stop_surrogate_not_solved"
    return out


def write_png(path: Path, width: int, height: int, lines: list[tuple[list[float], list[float], tuple[float, float], tuple[float, float], tuple[int, int, int]]]) -> None:
    pixels = bytearray([255, 255, 255] * width * height)
    def set_px(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(color)
    def draw(x1, y1, x2, y2, color):
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err = dx + dy
        while True:
            set_px(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x1 += sx
            if e2 <= dx:
                err += dx; y1 += sy
    for x, y, xlim, ylim, color in lines:
        pts = []
        for xi, yi in zip(x, y):
            px = int(70 + (xi - xlim[0]) / max(xlim[1] - xlim[0], EPS) * (width - 120))
            py = int(height - 60 - (yi - ylim[0]) / max(ylim[1] - ylim[0], EPS) * (height - 120))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            draw(*a, *b, color)
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def make_pca_figure(pca_info: dict[str, object]) -> None:
    explained = np.asarray(pca_info["explained"])
    cum = np.cumsum(explained)
    x = np.arange(1, min(10, len(cum)) + 1)
    y = cum[: len(x)]
    body = [
        fixed.svg_text(450, 30, "Reduced Surrogate Trajectory PCA", 18, "bold"),
        '<rect x="75" y="65" width="700" height="245" fill="#fbfbfb" stroke="#ccc"/>',
        fixed.plot_polyline_bounds(x, y, 110, 90, 610, 165, (1, max(x)), (0, 1), "#3f6f9e", 2.3),
    ]
    for xi, yi in zip(x, y):
        px = 110 + (xi - 1) / max(max(x) - 1, EPS) * 610
        py = 90 + 165 - yi * 165
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#3f6f9e"/>')
        body.append(fixed.svg_text(px, py - 8, f"{yi:.3f}", 8))
    body.append(fixed.svg_text(410, 292, "number of training-PCA components", 11))
    body.append(fixed.svg_text(42, 175, "cumulative variance", 11))
    path = FIGURES / "dfba_surrogate_trajectory_pca.svg"
    fixed.save_svg(path, 900, 345, "\n".join(body))
    write_png(path.with_suffix(".png"), 1800, 690, [(list(x), list(y), (1, max(x)), (0, 1), (63, 111, 158))])


def main() -> None:
    traj = pd.read_csv(DATA / "dfba_trajectories.csv")
    predictions = pd.read_csv(DATA / "dfba_model_predictions.csv")
    metrics = pd.read_csv(DATA / "dfba_canonical_metrics.csv")
    norm = normalized_error_rows(predictions, traj)
    pca, pca_info = pca_summary(traj)
    simple, simple_predictions = simple_baselines(traj)
    uncertainty = seed_uncertainty(metrics)
    decision = decision_gate(norm, simple, pca, uncertainty)
    DATA.mkdir(exist_ok=True)
    norm.to_csv(DATA / "dfba_surrogate_normalized_errors.csv", index=False)
    pca.to_csv(DATA / "dfba_trajectory_pca_summary.csv", index=False)
    simple.to_csv(DATA / "dfba_simple_baseline_metrics.csv", index=False)
    simple_predictions.to_csv(DATA / "dfba_simple_baseline_predictions.csv", index=False)
    uncertainty.to_csv(DATA / "dfba_model_seed_uncertainty.csv", index=False)
    decision.to_csv(DATA / "dfba_surrogate_decision_gate.csv", index=False)
    FIGURES.mkdir(exist_ok=True)
    make_pca_figure(pca_info)
    print("surrogate complexity audit complete")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
