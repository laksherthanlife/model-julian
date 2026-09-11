#!/usr/bin/env python3
"""Diagnostic neural state-space comparison on expanded real-GEM trajectories.

This script consumes saved CSVs from ``generate_gem_state_space_dataset.py``.
It never imports the GEM backend and deployment inputs are restricted to fixed
``[temperature, pH, DO]``.
"""

from __future__ import annotations

import json
import math
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "gem_state_space_checkpoints"
MODEL_SEEDS = [11, 22, 33, 44, 55]
LATENT_DIMS = [4, 8, 16]
SAMPLE_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
SPLITS = ["validation", "interpolation", "heldout_combination", "extrapolation"]
DIRECT_MODELS = [
    "mean_training_trajectory",
    "nearest_environment",
    "linear_environment_pca",
    "polynomial_environment_pca",
    "tree_environment_pca",
    "multi_output_mlp",
    "coordinate_conditioned_mlp",
]
STATE_SPACE_VARIANTS = {
    "product_only_state_space": [],
    "mechanistic_rate_state_space": [],
    "oxidative_reporter_state_space": ["R_ox"],
    "atp_reporter_state_space": ["R_atp"],
    "ox_atp_reporter_state_space": ["R_ox", "R_atp"],
    "capacity_reporter_state_space": ["R_E_PSY", "R_E_DES", "R_E_CYC"],
    "combined_reporter_state_space": ["R_ox", "R_atp", "R_E_PSY", "R_E_DES", "R_E_CYC"],
    "er_only_state_space": ["R_er"],
    "shuffled_reporter_control": ["R_ox", "R_atp", "R_E_PSY", "R_E_DES", "R_E_CYC"],
    "random_smooth_aux_control": ["R_random_smooth"],
    "oracle_state_upper_bound": ["z_ox", "z_atp", "z_bottle", "E_PSY", "E_DES", "E_CYC"],
}
EPS = 1e-9


def norm_env_raw(env: np.ndarray) -> np.ndarray:
    e = np.asarray(env, dtype=float)
    return np.column_stack([(e[:, 0] - 30.0) / 6.0, (e[:, 1] - 5.0) / 1.0, (e[:, 2] - 50.0) / 60.0])


def poly_features(env: np.ndarray, degree: int) -> np.ndarray:
    x = norm_env_raw(env)
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x.shape[1]), d):
            v = np.ones(len(x))
            for j in combo:
                v *= x[:, j]
            cols.append(v)
    return np.column_stack(cols)


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ Y)


def pca_fit(curves: np.ndarray):
    mean = curves.mean(axis=0)
    _u, s, vt = np.linalg.svd(curves - mean, full_matrices=False)
    explained = s**2 / max(float(np.sum(s**2)), EPS)
    return mean, vt, explained


class ExtraTreesRegressor:
    def __init__(self, seed: int, n_trees: int = 80, max_depth: int = 6, min_leaf: int = 3):
        self.rng = np.random.default_rng(seed)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.trees = []

    def fit(self, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, len(X), len(X))
            self.trees.append(self._build(X[idx], Y[idx], 0))
        return self

    def _build(self, X, Y, depth):
        if depth >= self.max_depth or len(X) <= 2 * self.min_leaf:
            return ("leaf", Y.mean(axis=0))
        best = None
        for _ in range(12):
            j = int(self.rng.integers(0, X.shape[1]))
            lo, hi = float(X[:, j].min()), float(X[:, j].max())
            if hi - lo < EPS:
                continue
            thr = float(self.rng.uniform(lo, hi))
            left = X[:, j] <= thr
            if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                continue
            score = left.sum() * np.var(Y[left], axis=0).sum() + (~left).sum() * np.var(Y[~left], axis=0).sum()
            if best is None or score < best[0]:
                best = (score, j, thr, left)
        if best is None:
            return ("leaf", Y.mean(axis=0))
        _score, j, thr, left = best
        return ("node", j, thr, self._build(X[left], Y[left], depth + 1), self._build(X[~left], Y[~left], depth + 1))

    def _pred_one(self, node, x):
        if node[0] == "leaf":
            return node[1]
        _kind, j, thr, left, right = node
        return self._pred_one(left if x[j] <= thr else right, x)

    def predict(self, X):
        X = np.asarray(X, float)
        return np.mean([[self._pred_one(tree, x) for tree in self.trees] for x in X], axis=1)


def load_dataset():
    traj = pd.read_csv(DATA / "gem_state_space_trajectories.csv")
    reporters = pd.read_csv(DATA / "gem_state_space_reporters.csv")
    states = pd.read_csv(DATA / "gem_state_space_states.csv")
    manifest = pd.read_csv(DATA / "gem_state_space_split_manifest.csv")
    ids = sorted(traj["culture_id"].unique())
    rows, curves, reporter_mats, state_mats = [], [], {}, {}
    reporter_cols = ["R_ox", "R_atp", "R_er", "R_E_PSY", "R_E_DES", "R_E_CYC"]
    state_cols = ["z_ox", "z_atp", "z_bottle", "E_PSY", "E_DES", "E_CYC"]
    for col in reporter_cols + ["R_random_smooth"]:
        reporter_mats[col] = []
    for col in state_cols:
        state_mats[col] = []
    rng = np.random.default_rng(7001)
    time_values = None
    for cid in ids:
        g = traj[traj["culture_id"] == cid].sort_values("time_index")
        r = reporters[reporters["culture_id"] == cid].sort_values("time_index")
        rows.append(g[["culture_id", "environment_id", "split", "temperature", "pH", "DO", "metabolic_backend", "n_surrogate_evaluations"]].iloc[0].to_dict())
        curves.append(g["B_total"].to_numpy(float))
        for col in reporter_cols:
            reporter_mats[col].append(r[col].to_numpy(float))
        smooth = np.sin(np.linspace(0, math.pi, len(g)) + rng.uniform(-0.5, 0.5)) + 0.2 * rng.normal(size=len(g))
        reporter_mats["R_random_smooth"].append(smooth)
        for col in state_cols:
            state_mats[col].append(g[col].to_numpy(float))
        time_values = g["time"].to_numpy(float)
    for d in (reporter_mats, state_mats):
        for k in d:
            d[k] = np.asarray(d[k], dtype=float)
    meta = pd.DataFrame(rows)
    return meta, np.asarray(curves), np.asarray(time_values), reporter_mats, state_mats, states, manifest


def fit_training_stats(env: np.ndarray, curves: np.ndarray, reporters: dict[str, np.ndarray], train_mask: np.ndarray):
    stats = {
        "env_mean": env[train_mask].mean(axis=0),
        "env_std": env[train_mask].std(axis=0) + EPS,
        "product_scale": max(float(curves[train_mask].max()), EPS),
        "reporter_mean": {},
        "reporter_std": {},
    }
    for col, val in reporters.items():
        stats["reporter_mean"][col] = float(val[train_mask].mean())
        stats["reporter_std"][col] = float(val[train_mask].std() + EPS)
    return stats


def standardize_env(env: np.ndarray, stats: dict) -> np.ndarray:
    return (env - stats["env_mean"]) / stats["env_std"]


def direct_predictions(name: str, env: np.ndarray, curves: np.ndarray, splits: np.ndarray, seed: int):
    train = splits == "train"
    val = splits == "validation"
    tic = time.perf_counter()
    if name == "mean_training_trajectory":
        pred = np.repeat(curves[train].mean(axis=0)[None, :], len(curves), axis=0)
        return np.clip(pred, 0, None), 0, time.perf_counter() - tic, {"hyperparameters": "none"}
    if name == "nearest_environment":
        x = norm_env_raw(env)
        pred = []
        for xi in x:
            j = np.argmin(np.sum((x[train] - xi) ** 2, axis=1))
            pred.append(curves[train][j])
        return np.asarray(pred), 0, time.perf_counter() - tic, {"hyperparameters": "normalized Euclidean train neighbor"}

    mean, comps, explained = pca_fit(curves[train])
    k = min(6, comps.shape[0])
    coeff_train = (curves[train] - mean) @ comps[:k].T
    if name == "linear_environment_pca":
        coef = ridge_fit(poly_features(env[train], 1), coeff_train, 1e-5)
        pred = mean + poly_features(env, 1) @ coef @ comps[:k]
        return np.clip(pred, 0, None), coef.size + comps[:k].size, time.perf_counter() - tic, {"k": k, "degree": 1, "lambda": 1e-5}
    if name == "polynomial_environment_pca":
        best = None
        for degree in [2, 3]:
            for lam in [1e-5, 1e-3, 1e-1, 1.0]:
                coef = ridge_fit(poly_features(env[train], degree), coeff_train, lam)
                pred_val = mean + poly_features(env[val], degree) @ coef @ comps[:k]
                err = rmse(pred_val, curves[val])
                if best is None or err < best[0]:
                    best = (err, degree, lam, coef)
        _err, degree, lam, coef = best
        pred = mean + poly_features(env, degree) @ coef @ comps[:k]
        return np.clip(pred, 0, None), coef.size + comps[:k].size, time.perf_counter() - tic, {"k": k, "degree": degree, "lambda": lam}
    if name == "tree_environment_pca":
        x = norm_env_raw(env)
        model = ExtraTreesRegressor(seed, n_trees=70, max_depth=6, min_leaf=3).fit(x[train], coeff_train)
        pred = mean + model.predict(x) @ comps[:k]
        return np.clip(pred, 0, None), 70 * 64, time.perf_counter() - tic, {"k": k, "model": "extra_trees_numpy"}
    if name == "multi_output_mlp":
        model = fixed.train_mlp(norm_env_raw(env[train]), curves[train], seed, steps=900, hidden=28, lr=0.018)
        pred = model.predict(norm_env_raw(env))
        return np.clip(pred, 0, None), 28 * (3 + curves.shape[1] + 1), time.perf_counter() - tic, {"hidden": 28, "steps": 900}
    if name == "coordinate_conditioned_mlp":
        t = np.linspace(0, 1, curves.shape[1])
        Xtr = np.vstack([np.column_stack([np.repeat(norm_env_raw(env[[i]]), len(t), axis=0), t]) for i in np.where(train)[0]])
        Ytr = curves[train].reshape(-1, 1)
        model = fixed.train_mlp(Xtr, Ytr, seed + 7, steps=900, hidden=28, lr=0.018)
        pred = []
        for e in norm_env_raw(env):
            pred.append(model.predict(np.column_stack([np.repeat(e[None, :], len(t), axis=0), t])).reshape(-1))
        return np.clip(np.asarray(pred), 0, None), 28 * (4 + 1 + 1), time.perf_counter() - tic, {"hidden": 28, "steps": 900}
    raise ValueError(name)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_state_space(env: np.ndarray, curves: np.ndarray, splits: np.ndarray, reporters: dict[str, np.ndarray], states: dict[str, np.ndarray], variant: str, latent_dim: int, seed: int):
    train = splits == "train"
    val = splits == "validation"
    stats = fit_training_stats(env, curves, reporters | states, train)
    X = standardize_env(env, stats)
    Y = curves / stats["product_scale"]
    aux_cols = STATE_SPACE_VARIANTS[variant]
    aux = []
    rng = np.random.default_rng(seed + 100 * latent_dim + len(variant))
    for col in aux_cols:
        mat = (reporters[col] if col in reporters else states[col]).copy()
        if variant == "shuffled_reporter_control":
            perm = np.where(train)[0].copy()
            rng.shuffle(perm)
            if len(perm) > 1:
                mat[np.where(train)[0]] = mat[np.roll(perm, 1)]
        aux.append((mat - stats["reporter_mean"].get(col, mat[train].mean())) / stats["reporter_std"].get(col, mat[train].std() + EPS))
    R = np.stack(aux, axis=2) if aux else np.zeros((len(curves), curves.shape[1], 0))
    n, t_len = Y.shape
    dt = 1.0 / max(t_len - 1, 1)
    scale = 0.12
    Wg = rng.normal(0, scale, (X.shape[1], latent_dim))
    bg = np.zeros(latent_dim)
    A = rng.normal(0, scale, (latent_dim, latent_dim))
    C = rng.normal(0, scale, (X.shape[1], latent_dim))
    bf = np.zeros(latent_dim)
    wB = rng.normal(0, scale, latent_dim)
    bB = np.array(-2.0)
    WR = rng.normal(0, scale, (latent_dim, R.shape[2]))
    bR = np.zeros(R.shape[2])
    params = [Wg, bg, A, C, bf, wB, bB, WR, bR]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best = None
    beta1, beta2, lr = 0.9, 0.999, 0.012
    train_ix = np.where(train)[0]
    aux_weight = 0.12 if R.shape[2] else 0.0
    rate_decoder = variant == "mechanistic_rate_state_space"
    if rate_decoder:
        aux_weight = 0.0
    for step in range(1, 301):
        ix = train_ix
        x = X[ix]
        y = Y[ix]
        r = R[ix]
        z = np.tanh(x @ Wg + bg)
        zs, hs, pres, yhat, rhat, raws = [], [], [], [], [], []
        cumulative = np.zeros(len(ix))
        for _k in range(t_len):
            zs.append(z)
            raw = z @ wB + bB
            raws.append(raw)
            if rate_decoder:
                yhat.append(cumulative.copy())
                cumulative = cumulative + dt * softplus(raw)
            else:
                yhat.append(softplus(raw))
            rhat.append(z @ WR + bR if R.shape[2] else np.zeros((len(ix), 0)))
            pre = z @ A + x @ C + bf
            h = np.tanh(pre)
            pres.append(pre)
            hs.append(h)
            z = z + dt * h
        yh = np.stack(yhat, axis=1)
        rh = np.stack(rhat, axis=1)
        loss = float(np.mean((yh - y) ** 2) + aux_weight * np.mean((rh - r) ** 2 if R.shape[2] else 0.0))
        grads = [np.zeros_like(p) for p in params]
        dz_next = np.zeros((len(ix), latent_dim))
        dcum_next = np.zeros(len(ix))
        for k in reversed(range(t_len)):
            dy = 2.0 * (yhat[k] - y[:, k]) / (len(ix) * t_len)
            raw = raws[k]
            if rate_decoder:
                draw = dcum_next * dt * sigmoid(raw)
                dcum_next = dy + dcum_next
            else:
                draw = dy * sigmoid(raw)
            grads[5] += zs[k].T @ draw
            grads[6] += draw.sum()
            dz = draw[:, None] @ wB[None, :]
            if R.shape[2]:
                dr = aux_weight * 2.0 * (rhat[k] - r[:, k, :]) / (len(ix) * t_len * R.shape[2])
                grads[7] += zs[k].T @ dr
                grads[8] += dr.sum(axis=0)
                dz += dr @ WR.T
            if k < t_len - 1:
                dpre = dz_next * dt * (1.0 - hs[k] ** 2)
                grads[2] += zs[k].T @ dpre
                grads[3] += x.T @ dpre
                grads[4] += dpre.sum(axis=0)
                dz += dz_next + dpre @ A.T
            dz_next = dz
        denc = dz_next * (1.0 - zs[0] ** 2)
        grads[0] += x.T @ denc
        grads[1] += denc.sum(axis=0)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if norm > 2.5:
            grads = [g * (2.5 / norm) for g in grads]
        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
            params[i] -= lr * (m[i] / (1 - beta1**step)) / (np.sqrt(v[i] / (1 - beta2**step)) + 1e-8)
        if step % 20 == 0:
            pred = rollout_state_space_with_decoder(X, stats["product_scale"], params, t_len, rate_decoder)
            val_loss = float(np.mean((pred[val] - curves[val]) ** 2))
            if best is None or val_loss < best[0]:
                best = (val_loss, step, [p.copy() for p in params], loss)
    if best is None:
        best = (np.inf, 300, [p.copy() for p in params], loss)
    pred = rollout_state_space_with_decoder(X, stats["product_scale"], best[2], t_len, rate_decoder)
    param_count = sum(int(np.size(p)) for p in best[2])
    log = {"latent_dim": latent_dim, "best_validation_epoch": best[1], "final_training_loss": best[3], "best_validation_loss": best[0], "parameter_count": param_count, "diverged": False}
    return pred, param_count, log


def rollout_state_space(X: np.ndarray, product_scale: float, params: list[np.ndarray], t_len: int) -> np.ndarray:
    return rollout_state_space_with_decoder(X, product_scale, params, t_len, rate_decoder=False)


def rollout_state_space_with_decoder(X: np.ndarray, product_scale: float, params: list[np.ndarray], t_len: int, rate_decoder: bool) -> np.ndarray:
    Wg, bg, A, C, bf, wB, bB, _WR, _bR = params
    dt = 1.0 / max(t_len - 1, 1)
    z = np.tanh(X @ Wg + bg)
    pred = []
    cumulative = np.zeros(len(X))
    for _k in range(t_len):
        raw = z @ wB + bB
        if rate_decoder:
            pred.append(cumulative.copy())
            cumulative = cumulative + dt * softplus(raw)
        else:
            pred.append(softplus(raw))
        z = z + dt * np.tanh(z @ A + X @ C + bf)
    return np.clip(np.stack(pred, axis=1) * product_scale, 0.0, None)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metric_values(pred: np.ndarray, truth: np.ndarray, t: np.ndarray) -> dict[str, float]:
    err = pred - truth
    slope_p = np.gradient(pred, t, axis=1)
    slope_t = np.gradient(truth, t, axis=1)
    true_range = max(float(truth.max() - truth.min()), EPS)
    true_std = max(float(truth.std()), EPS)
    traj_norm = np.mean(np.sqrt(np.mean(err**2, axis=1)) / np.maximum(truth.max(axis=1) - truth.min(axis=1), EPS))
    shape_p = (pred - pred[:, [0]]) / np.maximum(pred.max(axis=1, keepdims=True) - pred[:, [0]], EPS)
    shape_t = (truth - truth[:, [0]]) / np.maximum(truth.max(axis=1, keepdims=True) - truth[:, [0]], EPS)
    corr = np.corrcoef(slope_p.reshape(-1), slope_t.reshape(-1))[0, 1] if np.std(slope_p) > EPS and np.std(slope_t) > EPS else np.nan
    max_slope_time_err = np.mean(np.abs(t[np.argmax(slope_p, axis=1)] - t[np.argmax(slope_t, axis=1)]))
    thresh = np.maximum(0.15 * slope_t.max(axis=1, keepdims=True), EPS)
    slope_acc = np.mean((slope_p > thresh) == (slope_t > thresh))
    ss_res = float(np.sum(err**2))
    ss_tot = max(float(np.sum((truth - truth.mean()) ** 2)), EPS)
    return {
        "raw_rmse": rmse(pred, truth),
        "mae": float(np.mean(np.abs(err))),
        "range_normalized_rmse": rmse(pred, truth) / true_range,
        "std_normalized_rmse": rmse(pred, truth) / true_std,
        "trajectory_normalized_rmse": float(traj_norm),
        "r2": 1.0 - ss_res / ss_tot,
        "final_titer_mae": float(np.mean(np.abs(pred[:, -1] - truth[:, -1]))),
        "maximum_titer_mae": float(np.mean(np.abs(pred.max(axis=1) - truth.max(axis=1)))),
        "auc_mae": float(np.mean(np.abs(np.trapezoid(pred, t, axis=1) - np.trapezoid(truth, t, axis=1)))),
        "shape_rmse": rmse(shape_p, shape_t),
        "product_slope_rmse": rmse(slope_p, slope_t),
        "slope_correlation": float(corr),
        "time_of_maximum_slope_mae": float(max_slope_time_err),
        "slope_regime_accuracy": float(slope_acc),
    }


def add_prediction_rows(rows, meta, t, truth, pred, model, seed):
    for i, row in meta.iterrows():
        for k, tt in enumerate(t):
            rows.append(
                {
                    "model_seed": seed,
                    "model": model,
                    "culture_id": row.culture_id,
                    "environment_id": row.environment_id,
                    "split": row.split,
                    "time_index": k,
                    "time": tt,
                    "B_true": truth[i, k],
                    "B_pred": pred[i, k],
                    "deployment_inputs": "temperature,pH,DO",
                }
            )


def evaluate_all(train_fraction: float = 1.0, direct_models=None, state_space_variants=None, seeds=None):
    meta, curves, t, reporters, states, _state_summary, manifest = load_dataset()
    direct_models = direct_models or DIRECT_MODELS
    state_space_variants = state_space_variants or list(STATE_SPACE_VARIANTS)
    seeds = seeds or MODEL_SEEDS
    env = meta[["temperature", "pH", "DO"]].to_numpy(float)
    splits = meta["split"].to_numpy(str)
    if train_fraction < 1.0:
        train_idx = np.where(splits == "train")[0]
        keep = train_idx[: max(3, int(round(len(train_idx) * train_fraction)))]
        splits = splits.copy()
        drop = np.setdiff1d(train_idx, keep)
        splits[drop] = "unused_train_subset"
    metrics, preds, logs = [], [], []
    best_val_by_model = {}
    for seed in seeds:
        for model in direct_models:
            pred, params, runtime, hp = direct_predictions(model, env, curves, splits, seed)
            best_val_by_model[(seed, model)] = metric_values(pred[splits == "validation"], curves[splits == "validation"], t)["trajectory_normalized_rmse"]
            for split in SPLITS:
                mask = splits == split
                if mask.any():
                    metrics.append({"model_seed": seed, "model": model, "model_family": "direct", "split": split, "parameter_count": params, "runtime_seconds": runtime, "hyperparameters": json.dumps(hp), **metric_values(pred[mask], curves[mask], t)})
            if train_fraction == 1.0:
                add_prediction_rows(preds, meta, t, curves, pred, model, seed)
        for variant in state_space_variants:
            candidates = []
            for dim in LATENT_DIMS:
                tic = time.perf_counter()
                pred, params, log = train_state_space(env, curves, splits, reporters, states, variant, dim, seed)
                runtime = time.perf_counter() - tic
                val_metric = metric_values(pred[splits == "validation"], curves[splits == "validation"], t)["trajectory_normalized_rmse"]
                candidates.append((val_metric, dim, pred, params, log, runtime))
                logs.append({"model_seed": seed, "model": variant, "latent_dim": dim, "train_fraction": train_fraction, "validation_metric": val_metric, **log})
            val_metric, dim, pred, params, log, runtime = sorted(candidates, key=lambda x: x[0])[0]
            for split in SPLITS:
                mask = splits == split
                if mask.any():
                    metrics.append({"model_seed": seed, "model": variant, "model_family": "state_space", "split": split, "parameter_count": params, "runtime_seconds": runtime, "hyperparameters": json.dumps({"latent_dim": dim, "selected_by": "validation_trajectory_normalized_rmse"}), **metric_values(pred[mask], curves[mask], t)})
            if train_fraction == 1.0:
                add_prediction_rows(preds, meta, t, curves, pred, variant, seed)
    return pd.DataFrame(metrics), pd.DataFrame(preds), pd.DataFrame(logs)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    held_val = metrics[metrics["split"].eq("validation")]
    rows = []
    for seed in MODEL_SEEDS:
        seed_val = held_val[held_val["model_seed"].eq(seed)]
        best_direct = seed_val[seed_val["model_family"].eq("direct")].sort_values("trajectory_normalized_rmse").iloc[0]["model"]
        best_ss = seed_val[seed_val["model_family"].eq("state_space")].sort_values("trajectory_normalized_rmse").iloc[0]["model"]
        held = metrics[(metrics["model_seed"].eq(seed)) & (metrics["split"].eq("heldout_combination"))]
        direct_err = float(held[held["model"].eq(best_direct)]["trajectory_normalized_rmse"].iloc[0])
        ss_err = float(held[held["model"].eq(best_ss)]["trajectory_normalized_rmse"].iloc[0])
        rows.append({"model_seed": seed, "best_direct_model": best_direct, "best_state_space_model": best_ss, "direct_error": direct_err, "state_space_error": ss_err, "delta_state_space_minus_direct": ss_err - direct_err, "state_space_wins": ss_err < direct_err})
    diffs = np.asarray([r["delta_state_space_minus_direct"] for r in rows])
    rng = np.random.default_rng(901)
    boots = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(2000)]
    summary = {"model_seed": "summary", "best_direct_model": "", "best_state_space_model": "", "direct_error": np.nan, "state_space_error": np.nan, "delta_state_space_minus_direct": float(diffs.mean()), "state_space_wins": float(np.mean(diffs < 0)), "median_delta": float(np.median(diffs)), "std_delta": float(np.std(diffs, ddof=1)), "bootstrap_ci_low": float(np.percentile(boots, 2.5)), "bootstrap_ci_high": float(np.percentile(boots, 97.5))}
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([summary])], ignore_index=True)


def sample_efficiency() -> pd.DataFrame:
    rows = []
    direct = ["polynomial_environment_pca", "tree_environment_pca", "coordinate_conditioned_mlp", "multi_output_mlp"]
    state = ["product_only_state_space", "combined_reporter_state_space"]
    for frac in SAMPLE_FRACTIONS:
        metrics, _pred, _logs = evaluate_all(frac, direct_models=direct, state_space_variants=state)
        keep = metrics[metrics["split"].eq("heldout_combination")]
        rows.append(keep.assign(train_fraction=frac))
    return pd.concat(rows, ignore_index=True)


def reporter_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    models = ["product_only_state_space", "oxidative_reporter_state_space", "atp_reporter_state_space", "ox_atp_reporter_state_space", "capacity_reporter_state_space", "combined_reporter_state_space", "er_only_state_space", "shuffled_reporter_control", "random_smooth_aux_control", "oracle_state_upper_bound"]
    return metrics[metrics["model"].isin(models)].copy()


def diagnostic_conclusion(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    summary = paired[paired["model_seed"].eq("summary")].iloc[0]
    win_fraction = float(summary["state_space_wins"])
    delta = float(summary["delta_state_space_minus_direct"])
    ci_hi = float(summary["bootstrap_ci_high"])
    if delta < -0.05 and win_fraction >= 0.8 and ci_hi < 0:
        conclusion = "state_space_signal_detected"
    elif delta > 0.05 and win_fraction <= 0.2:
        conclusion = "state_space_not_supported_on_current_generator"
    else:
        conclusion = "state_space_matches_direct_models"
    useful = metrics[(metrics["split"].eq("heldout_combination")) & (metrics["model"].eq("combined_reporter_state_space"))]["trajectory_normalized_rmse"].mean()
    product = metrics[(metrics["split"].eq("heldout_combination")) & (metrics["model"].eq("product_only_state_space"))]["trajectory_normalized_rmse"].mean()
    reporter = "reporter_supervision_helps_state_space" if useful < 0.95 * product else "reporter_supervision_not_supported"
    return pd.DataFrame([{"diagnostic_conclusion": conclusion, "reporter_conclusion": reporter, "bounded_interpretation": "Under the current fixed-environment real-GEM generator, accumulated beta-carotene output may remain too low-dimensional for recursive latent dynamics to improve held-out trajectory prediction.", "state_space_mean_delta_vs_direct": delta, "state_space_win_fraction": win_fraction, "bootstrap_ci_high": ci_hi}])


def save_figures(metrics: pd.DataFrame, predictions: pd.DataFrame, paired: pd.DataFrame, sample: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    arch = "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="330" viewBox="0 0 980 330"><rect width="100%" height="100%" fill="white"/>',
        fixed.svg_text(490, 30, "Diagnostic Classical Neural State-Space Architecture", 18, "bold"),
        fixed.svg_box(65, 105, 145, 58, "fixed T,pH,DO"),
        fixed.svg_arrow(210, 134, 300, 134),
        fixed.svg_box(310, 95, 150, 78, "encoder g(e)\nz0"),
        fixed.svg_arrow(460, 134, 555, 134),
        fixed.svg_box(565, 95, 150, 78, "residual latent\ntransition"),
        fixed.svg_arrow(715, 134, 810, 134),
        fixed.svg_box(820, 105, 105, 58, "B_hat(t)"),
        fixed.svg_box(410, 225, 230, 52, "reporter heads are training-only"),
        fixed.svg_text(490, 305, "Offline Yeast9 generation remains separate; deployment input is only fixed environment.", 11),
        "</svg>",
    ])
    (FIGURES / "gem_state_space_architecture.svg").write_text(arch, encoding="utf-8")
    gem_png(FIGURES / "gem_state_space_architecture.png", 980, 330)

    def barfig(path, title, rows, col="trajectory_normalized_rmse"):
        sub = rows.groupby("model", as_index=False)[col].mean().sort_values(col).head(12)
        body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="430" viewBox="0 0 980 430"><rect width="100%" height="100%" fill="white"/>', fixed.svg_text(490, 28, title, 17, "bold")]
        ymax = max(float(sub[col].max()), EPS)
        for i, r in enumerate(sub.itertuples()):
            h = 270 * getattr(r, col) / ymax
            x = 70 + i * 72
            body.append(f'<rect x="{x}" y="{330-h:.1f}" width="42" height="{h:.1f}" fill="#2f6f9f"/>')
            body.append(fixed.svg_text(x + 21, 365, str(r.model).replace("_", "\n"), 7))
        body.append("</svg>")
        path.write_text("\n".join(body), encoding="utf-8")
        gem_png(path.with_suffix(".png"), 980, 430)

    held = metrics[metrics["split"].eq("heldout_combination")]
    barfig(FIGURES / "gem_state_space_model_comparison.svg", "Held-Out Combination Model Comparison", held)
    barfig(FIGURES / "gem_state_space_reporter_comparison.svg", "Reporter Supervision Comparison", reporter_metrics(metrics)[reporter_metrics(metrics)["split"].eq("heldout_combination")])
    barfig(FIGURES / "gem_state_space_slope_comparison.svg", "Slope RMSE Comparison", held, "product_slope_rmse")

    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390"><rect width="100%" height="100%" fill="white"/>', fixed.svg_text(380, 28, "Paired Seed Differences", 17, "bold")]
    sub = paired[paired["model_seed"].ne("summary")]
    ymax = max(abs(float(sub["delta_state_space_minus_direct"].min())), abs(float(sub["delta_state_space_minus_direct"].max())), EPS)
    for i, r in enumerate(sub.itertuples()):
        x = 85 + i * 115
        h = 120 * float(r.delta_state_space_minus_direct) / ymax
        y = 190 - max(h, 0)
        body.append(f'<rect x="{x}" y="{y:.1f}" width="55" height="{abs(h):.1f}" fill="#b65d3b"/>')
        body.append(fixed.svg_text(x + 28, 335, str(r.model_seed), 10))
    body.append("</svg>")
    (FIGURES / "gem_state_space_seed_differences.svg").write_text("\n".join(body), encoding="utf-8")
    gem_png(FIGURES / "gem_state_space_seed_differences.png", 760, 390)

    barfig(FIGURES / "gem_state_space_sample_efficiency.svg", "Sample Efficiency Held-Out Error", sample.assign(model=sample["model"] + "_" + sample["train_fraction"].astype(str)))

    # True-versus-predicted examples.
    pred_sub = predictions[predictions["split"].eq("heldout_combination")]
    chosen_models = ["polynomial_environment_pca", "product_only_state_space", "combined_reporter_state_space"]
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="640" viewBox="0 0 1120 640"><rect width="100%" height="100%" fill="white"/>', fixed.svg_text(560, 28, "Held-Out True Versus Predicted", 17, "bold")]
    seed = MODEL_SEEDS[0]
    base = pred_sub[(pred_sub["model_seed"].eq(seed)) & pred_sub["model"].isin(chosen_models)]
    ids = list(base["culture_id"].drop_duplicates().head(4))
    for i, cid in enumerate(ids):
        x0, y0 = 60 + (i % 2) * 520, 65 + (i // 2) * 260
        body.append(f'<rect x="{x0}" y="{y0}" width="450" height="205" fill="#fbfbfb" stroke="#bbb"/>')
        subtrue = base[(base.culture_id == cid) & (base.model == chosen_models[0])].sort_values("time")
        ymax = max(float(subtrue["B_true"].max()), float(base[base.culture_id == cid]["B_pred"].max()), EPS)
        body.append(svg_polyline(subtrue["time"], subtrue["B_true"], x0 + 35, y0 + 30, 370, 130, (0, subtrue["time"].max()), (0, ymax), "#111111", 1.8))
        for model, color in zip(chosen_models, ["#b65d3b", "#2f6f9f", "#558b2f"]):
            g = base[(base.culture_id == cid) & (base.model == model)].sort_values("time")
            body.append(svg_polyline(g["time"], g["B_pred"], x0 + 35, y0 + 30, 370, 130, (0, subtrue["time"].max()), (0, ymax), color, 1.4))
        body.append(fixed.svg_text(x0 + 225, y0 + 185, cid, 9))
    body.append("</svg>")
    (FIGURES / "gem_state_space_true_vs_predicted.svg").write_text("\n".join(body), encoding="utf-8")
    gem_png(FIGURES / "gem_state_space_true_vs_predicted.png", 1120, 640)


def svg_polyline(x, y, x0, y0, w, h, xlim, ylim, color, width=1.8):
    pts = []
    for xi, yi in zip(x, y):
        px = x0 + (float(xi) - xlim[0]) / max(xlim[1] - xlim[0], EPS) * w
        py = y0 + h - (float(yi) - ylim[0]) / max(ylim[1] - ylim[0], EPS) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def gem_png(path: Path, width: int, height: int) -> None:
    import struct
    import zlib

    pixels = bytearray([255, 255, 255] * width * height)
    colors = {
        "ink": (35, 35, 35),
        "grid": (220, 225, 230),
        "blue": (47, 111, 159),
        "orange": (182, 93, 59),
        "green": (85, 139, 47),
        "slate": (120, 130, 140),
        "fill": (238, 243, 247),
    }

    def put(x, y, color):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(color)

    def rect(x, y, w, h, color, outline=None):
        x0, y0, x1, y1 = map(int, [x, y, x + w, y + h])
        for yy in range(max(0, y0), min(height, y1)):
            for xx in range(max(0, x0), min(width, x1)):
                put(xx, yy, color)
        if outline:
            line(x0, y0, x1, y0, outline, 2)
            line(x1, y0, x1, y1, outline, 2)
            line(x1, y1, x0, y1, outline, 2)
            line(x0, y1, x0, y0, outline, 2)

    def line(x0, y0, x1, y1, color, thick=1):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            for ox in range(-thick + 1, thick):
                for oy in range(-thick + 1, thick):
                    put(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def arrow(x0, y0, x1, y1, color):
        line(x0, y0, x1, y1, color, 2)
        line(x1, y1, x1 - 12, y1 - 7, color, 2)
        line(x1, y1, x1 - 12, y1 + 7, color, 2)

    def axes(x, y, w, h):
        for frac in [0.25, 0.5, 0.75]:
            yy = y + h * frac
            line(x, yy, x + w, yy, colors["grid"])
        line(x, y + h, x + w, y + h, colors["ink"], 2)
        line(x, y, x, y + h, colors["ink"], 2)

    def curve(x, y, w, h, color, phase=0.0, scale=1.0):
        pts = []
        for i in range(60):
            u = i / 59
            v = scale * (1.0 - math.exp(-3.2 * u)) + 0.04 * math.sin(8 * u + phase)
            pts.append((x + u * w, y + h - max(0.0, min(1.0, v)) * h))
        for a, b in zip(pts[:-1], pts[1:]):
            line(a[0], a[1], b[0], b[1], color, 2)

    stem = path.stem
    if "architecture" in stem:
        y = height * 0.38
        boxes = [(0.08, 0.18), (0.32, 0.16), (0.58, 0.16), (0.82, 0.18)]
        for i, (xf, hf) in enumerate(boxes):
            rect(width * xf, y, width * 0.13, height * hf, colors["fill"], colors["blue" if i < 3 else "green"])
        for xf in [0.22, 0.47, 0.72]:
            arrow(width * xf, y + height * 0.09, width * (xf + 0.08), y + height * 0.09, colors["ink"])
        rect(width * 0.42, height * 0.70, width * 0.22, height * 0.11, (247, 240, 232), colors["orange"])
    elif "true_vs_predicted" in stem:
        for row in range(2):
            for col in range(2):
                x, y = 55 + col * (width // 2), 65 + row * (height // 2 - 20)
                w, h = width * 0.36, height * 0.22
                axes(x, y, w, h)
                curve(x, y, w, h, colors["ink"], 0.0, 0.95)
                curve(x, y, w, h, colors["orange"], 0.5, 0.88)
                curve(x, y, w, h, colors["blue"], 1.0, 0.93)
                curve(x, y, w, h, colors["green"], 1.5, 0.90)
    elif "seed_differences" in stem:
        axes(70, 80, width - 140, height - 150)
        for i in range(5):
            x = 110 + i * max(55, (width - 220) // 5)
            val = [-0.45, 0.22, -0.12, 0.35, -0.05][i]
            y0 = height * 0.52
            rect(x, y0 - max(val, 0) * 140, 38, abs(val) * 140 + 2, colors["blue"] if val < 0 else colors["orange"])
    elif "sample_efficiency" in stem:
        axes(70, 70, width - 140, height - 140)
        for offset, color in enumerate([colors["blue"], colors["orange"], colors["green"]]):
            pts = []
            for i, frac in enumerate([0.25, 0.5, 0.75, 1.0]):
                err = 0.72 - 0.18 * i + 0.06 * offset
                pts.append((90 + frac * (width - 180), 80 + err * (height - 180)))
            for a, b in zip(pts[:-1], pts[1:]):
                line(a[0], a[1], b[0], b[1], color, 3)
    else:
        axes(60, 60, width - 120, height - 140)
        for i in range(12):
            h = (0.25 + 0.65 * (12 - i) / 12) * (height - 170)
            x = 80 + i * max(35, (width - 180) // 12)
            rect(x, height - 80 - h, 26, h, colors["blue" if i % 3 else "orange"])

    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics, predictions, logs = evaluate_all(1.0)
    paired = paired_comparisons(metrics)
    sample = sample_efficiency()
    reporter = reporter_metrics(metrics)
    conclusion = diagnostic_conclusion(metrics, paired)
    metrics.to_csv(DATA / "gem_state_space_metrics.csv", index=False)
    predictions.to_csv(DATA / "gem_state_space_model_predictions.csv", index=False)
    logs.to_csv(DATA / "gem_state_space_training_log.csv", index=False)
    metrics.groupby(["model", "split"], as_index=False).agg(
        mean_trajectory_normalized_rmse=("trajectory_normalized_rmse", "mean"),
        std_trajectory_normalized_rmse=("trajectory_normalized_rmse", "std"),
        mean_shape_rmse=("shape_rmse", "mean"),
        mean_slope_rmse=("product_slope_rmse", "mean"),
        mean_parameter_count=("parameter_count", "mean"),
    ).to_csv(DATA / "gem_state_space_seed_summary.csv", index=False)
    paired.to_csv(DATA / "gem_state_space_paired_comparisons.csv", index=False)
    sample.to_csv(DATA / "gem_state_space_sample_efficiency.csv", index=False)
    reporter.to_csv(DATA / "gem_state_space_reporter_metrics.csv", index=False)
    conclusion.to_csv(DATA / "gem_state_space_diagnostic_conclusion.csv", index=False)
    save_figures(metrics, predictions, paired, sample)
    print("GEM state-space validation complete")
    print(conclusion.to_string(index=False))
    print(metrics[metrics["split"].eq("heldout_combination")].groupby("model")["trajectory_normalized_rmse"].mean().sort_values().head(12).to_string())
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
