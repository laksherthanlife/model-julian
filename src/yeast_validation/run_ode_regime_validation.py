#!/usr/bin/env python3
"""Predict fixed-environment ODE trajectories with direct and state-space models."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ode_regime_core as ode
import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "ode_regime_discovery"
MODEL_SEEDS = [11, 22, 33]
FAST_SEEDS = [11]
LATENT_DIMS = [4, 8, 12]
FAST_LATENT_DIMS = [4, 8]
SPLITS = ["validation", "random_test", "heldout_hole", "edge_extrapolation"]
DIRECT_MODELS = [
    "mean_training_trajectory",
    "nearest_environment",
    "linear_environment_pca",
    "polynomial_environment_pca",
    "tree_environment_pca",
    "multi_output_mlp",
    "coordinate_conditioned_mlp",
    "direct_multi_output_physiology_mlp",
]
STATE_SPACE_VARIANTS = [
    "product_only_state_space",
    "physiology_supervised_state_space",
    "shuffled_aux_state_space",
    "random_smooth_aux_state_space",
]
PHYS_LABELS = ["X", "A", "R", "E", "S", "C", "v_product", "mu"]


class ExtraTreesRegressor:
    def __init__(self, seed: int, n_trees: int = 70, max_depth: int = 7, min_leaf: int = 4):
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
        for _ in range(16):
            j = int(self.rng.integers(0, X.shape[1]))
            lo, hi = float(X[:, j].min()), float(X[:, j].max())
            if hi - lo < ode.EPS:
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


def load_dataset(prefix: str):
    traj = pd.read_csv(DATA / f"{prefix}_trajectories.csv")
    diag = pd.read_csv(DATA / f"{prefix}_diagnostics.csv")
    meta, mats, t = ode.pivot_trajectories(traj, ["P"] + PHYS_LABELS)
    regimes = diag.set_index("culture_id").loc[meta["culture_id"], "dominant_regime"].to_numpy(str)
    return meta, mats, t, regimes, diag


def train_stats(env, curves, aux, train):
    stats = {
        "env_mean": env[train].mean(axis=0),
        "env_std": env[train].std(axis=0) + ode.EPS,
        "product_scale": max(float(curves[train].max()), ode.EPS),
        "aux_mean": aux[train].mean(axis=(0, 1)),
        "aux_std": aux[train].std(axis=(0, 1)) + ode.EPS,
    }
    return stats


def metric_values(pred: np.ndarray, truth: np.ndarray, t: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = pred - truth
    dp = np.gradient(pred, t, axis=1)
    dt = np.gradient(truth, t, axis=1)
    range_norm = np.maximum(truth.max(axis=1) - truth.min(axis=1), ode.EPS)
    max_rate_time_err = np.mean(np.abs(t[np.argmax(dp, axis=1)] - t[np.argmax(dt, axis=1)]))
    late = slice(max(0, int(0.75 * truth.shape[1])), truth.shape[1])
    return {
        "absolute_rmse": ode.rmse(pred, truth),
        "normalized_rmse": float(np.mean(np.sqrt(np.mean(err**2, axis=1)) / range_norm)),
        "endpoint_mae": float(np.mean(np.abs(pred[:, -1] - truth[:, -1]))),
        "auc_mae": float(np.mean(np.abs(np.trapezoid(pred, t, axis=1) - np.trapezoid(truth, t, axis=1)))),
        "max_rate_mae": float(np.mean(np.abs(dp.max(axis=1) - dt.max(axis=1)))),
        "time_of_max_rate_mae": float(max_rate_time_err),
        "late_stage_rmse": ode.rmse(pred[:, late], truth[:, late]),
        "derivative_rmse": ode.rmse(dp, dt),
    }


def direct_predictions(name: str, env: np.ndarray, curves: np.ndarray, aux: np.ndarray, splits: np.ndarray, seed: int, fast: bool):
    train = splits == "train"
    val = splits == "validation"
    tic = time.perf_counter()
    if name == "mean_training_trajectory":
        pred = np.repeat(curves[train].mean(axis=0)[None, :], len(curves), axis=0)
        return np.clip(pred, 0, None), None, 0, time.perf_counter() - tic, {"hyperparameters": "none"}
    if name == "nearest_environment":
        x = ode.normalize_env_raw(env)
        pred = []
        for xi in x:
            j = np.argmin(np.sum((x[train] - xi) ** 2, axis=1))
            pred.append(curves[train][j])
        return np.asarray(pred), None, 0, time.perf_counter() - tic, {"distance": "normalized Euclidean"}
    mean, comps, _explained = ode.pca_fit(curves[train])
    k = min(8, comps.shape[0])
    coeff_train = (curves[train] - mean) @ comps[:k].T
    if name == "linear_environment_pca":
        coef = ode.ridge_fit(ode.polynomial_features(env[train], 1), coeff_train, 1e-5)
        pred = mean + ode.polynomial_features(env, 1) @ coef @ comps[:k]
        return np.clip(pred, 0, None), None, coef.size, time.perf_counter() - tic, {"k": k, "degree": 1}
    if name == "polynomial_environment_pca":
        best = None
        for degree in [2, 3]:
            for lam in [1e-5, 1e-3, 1e-1, 1.0]:
                coef = ode.ridge_fit(ode.polynomial_features(env[train], degree), coeff_train, lam)
                pred_val = mean + ode.polynomial_features(env[val], degree) @ coef @ comps[:k]
                err = metric_values(pred_val, curves[val], np.arange(curves.shape[1]))["normalized_rmse"]
                if best is None or err < best[0]:
                    best = (err, degree, lam, coef)
        _err, degree, lam, coef = best
        pred = mean + ode.polynomial_features(env, degree) @ coef @ comps[:k]
        return np.clip(pred, 0, None), None, coef.size, time.perf_counter() - tic, {"k": k, "degree": degree, "lambda": lam}
    if name == "tree_environment_pca":
        model = ExtraTreesRegressor(seed).fit(ode.normalize_env_raw(env[train]), coeff_train)
        pred = mean + model.predict(ode.normalize_env_raw(env)) @ comps[:k]
        return np.clip(pred, 0, None), None, 70 * 96, time.perf_counter() - tic, {"k": k, "model": "extra_trees_numpy"}
    if name == "multi_output_mlp":
        steps = 180 if fast else 550
        model = fixed.train_mlp(ode.normalize_env_raw(env[train]), curves[train], seed, steps=steps, hidden=30, lr=0.018)
        pred = model.predict(ode.normalize_env_raw(env))
        return np.clip(pred, 0, None), None, 30 * (3 + curves.shape[1] + 1), time.perf_counter() - tic, {"hidden": 30, "steps": steps}
    if name == "coordinate_conditioned_mlp":
        steps = 180 if fast else 550
        tt = np.linspace(0, 1, curves.shape[1])
        Xtr = np.vstack([np.column_stack([np.repeat(ode.normalize_env_raw(env[[i]]), len(tt), axis=0), tt]) for i in np.where(train)[0]])
        Ytr = curves[train].reshape(-1, 1)
        model = fixed.train_mlp(Xtr, Ytr, seed + 7, steps=steps, hidden=30, lr=0.018)
        pred = [model.predict(np.column_stack([np.repeat(e[None, :], len(tt), axis=0), tt])).reshape(-1) for e in ode.normalize_env_raw(env)]
        return np.clip(np.asarray(pred), 0, None), None, 30 * (4 + 1 + 1), time.perf_counter() - tic, {"hidden": 30, "steps": steps}
    if name == "direct_multi_output_physiology_mlp":
        steps = 180 if fast else 550
        stats = train_stats(env, curves, aux, train)
        y_product = curves / stats["product_scale"]
        y_aux = (aux - stats["aux_mean"]) / stats["aux_std"]
        Y = np.hstack([y_product, y_aux.reshape(len(aux), -1)])
        model = fixed.train_mlp(ode.normalize_env_raw(env[train]), Y[train], seed + 13, steps=steps, hidden=36, lr=0.016)
        Yhat = model.predict(ode.normalize_env_raw(env))
        pred = np.clip(Yhat[:, : curves.shape[1]] * stats["product_scale"], 0, None)
        aux_hat = Yhat[:, curves.shape[1] :].reshape(aux.shape) * stats["aux_std"] + stats["aux_mean"]
        return pred, aux_hat, 36 * (3 + Y.shape[1] + 1), time.perf_counter() - tic, {"hidden": 36, "steps": steps, "auxiliary_labels": PHYS_LABELS}
    raise ValueError(name)


def train_state_space(env, curves, aux, splits, variant: str, latent_dim: int, seed: int, fast: bool):
    train = splits == "train"
    val = splits == "validation"
    stats = train_stats(env, curves, aux, train)
    X = (env - stats["env_mean"]) / stats["env_std"]
    Y = curves / stats["product_scale"]
    Atrue = (aux - stats["aux_mean"]) / stats["aux_std"]
    if variant == "product_only_state_space":
        Atrue = Atrue[:, :, :0]
    elif variant == "shuffled_aux_state_space":
        rng_shuffle = np.random.default_rng(seed + 300)
        perm = np.where(train)[0].copy()
        rng_shuffle.shuffle(perm)
        shuffled = Atrue.copy()
        shuffled[np.where(train)[0]] = shuffled[np.roll(perm, 1)]
        Atrue = shuffled
    elif variant == "random_smooth_aux_state_space":
        rng_random = np.random.default_rng(seed + 400)
        tt = np.linspace(0, 1, curves.shape[1])
        random_aux = []
        for _ in range(Atrue.shape[2]):
            phase = rng_random.uniform(0, 2 * math.pi, size=len(curves))
            amp = rng_random.uniform(0.4, 1.2, size=len(curves))
            random_aux.append(amp[:, None] * np.sin(tt[None, :] * math.pi + phase[:, None]))
        Atrue = np.stack(random_aux, axis=2)
    n, t_len = Y.shape
    n_aux = Atrue.shape[2]
    rng = np.random.default_rng(seed + 1000 * latent_dim + len(variant))
    scale = 0.10
    Wg = rng.normal(0, scale, (X.shape[1], latent_dim))
    bg = np.zeros(latent_dim)
    A = rng.normal(0, scale, (latent_dim, latent_dim))
    C = rng.normal(0, scale, (X.shape[1], latent_dim))
    bf = np.zeros(latent_dim)
    wP = rng.normal(0, scale, latent_dim)
    bP = np.array(-1.3)
    WA = rng.normal(0, scale, (latent_dim, n_aux))
    bA = np.zeros(n_aux)
    params = [Wg, bg, A, C, bf, wP, bP, WA, bA]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    train_ix = np.where(train)[0]
    beta1, beta2, lr = 0.9, 0.999, 0.012
    aux_weight = 0.10 if n_aux else 0.0
    steps = 160 if fast else 300
    best = None
    dt = 1.0 / max(t_len - 1, 1)
    for step in range(1, steps + 1):
        ix = train_ix
        x = X[ix]
        y = Y[ix]
        atrue = Atrue[ix]
        z = np.tanh(x @ Wg + bg)
        zs, hs, yhat, ahat, raws = [], [], [], [], []
        for _k in range(t_len):
            zs.append(z)
            raw = z @ wP + bP
            raws.append(raw)
            yhat.append(ode.softplus(raw))
            ahat.append(z @ WA + bA if n_aux else np.zeros((len(ix), 0)))
            h = np.tanh(z @ A + x @ C + bf)
            hs.append(h)
            z = z + dt * h
        yh = np.stack(yhat, axis=1)
        ah = np.stack(ahat, axis=1)
        grads = [np.zeros_like(p) for p in params]
        dz_next = np.zeros((len(ix), latent_dim))
        for k in reversed(range(t_len)):
            dy = 2.0 * (yhat[k] - y[:, k]) / (len(ix) * t_len)
            draw = dy * ode.sigmoid(raws[k])
            grads[5] += zs[k].T @ draw
            grads[6] += draw.sum()
            dz = draw[:, None] @ wP[None, :]
            if n_aux:
                da = aux_weight * 2.0 * (ahat[k] - atrue[:, k, :]) / (len(ix) * t_len * n_aux)
                grads[7] += zs[k].T @ da
                grads[8] += da.sum(axis=0)
                dz += da @ WA.T
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
            pred, _aux, _latent = rollout_state_space(X, stats, params, t_len)
            val_metric = metric_values(pred[val], curves[val], np.arange(t_len))["normalized_rmse"]
            if best is None or val_metric < best[0]:
                best = (val_metric, step, [p.copy() for p in params])
    if best is None:
        best = (np.inf, steps, [p.copy() for p in params])
    pred, aux_pred, latent = rollout_state_space(X, stats, best[2], t_len)
    if n_aux == 0:
        aux_pred = None
    return pred, aux_pred, latent, sum(int(np.size(p)) for p in best[2]), {"latent_dim": latent_dim, "best_validation_epoch": best[1], "validation_normalized_rmse": best[0]}


def rollout_state_space(X, stats, params, t_len):
    Wg, bg, A, C, bf, wP, bP, WA, bA = params
    dt = 1.0 / max(t_len - 1, 1)
    z = np.tanh(X @ Wg + bg)
    pred, aux, latent = [], [], []
    for _k in range(t_len):
        latent.append(z.copy())
        pred.append(ode.softplus(z @ wP + bP))
        aux.append(z @ WA + bA if WA.shape[1] else np.zeros((len(X), 0)))
        z = z + dt * np.tanh(z @ A + X @ C + bf)
    p = np.clip(np.stack(pred, axis=1) * stats["product_scale"], 0.0, None)
    a = np.stack(aux, axis=1)
    if a.shape[2]:
        a = a * stats["aux_std"] + stats["aux_mean"]
    return p, a, np.stack(latent, axis=1)


def evaluate(prefix: str, fast: bool):
    meta, mats, t, regimes, _diag = load_dataset(prefix)
    env = meta[["temperature", "pH", "DO"]].to_numpy(float)
    splits = meta["split"].to_numpy(str)
    curves = mats["P"]
    aux = np.stack([mats[col] for col in PHYS_LABELS], axis=2)
    seeds = FAST_SEEDS if fast else MODEL_SEEDS
    latent_dims = FAST_LATENT_DIMS if fast else LATENT_DIMS
    metrics, preds, phys_preds, reps, logs = [], [], [], [], []
    for seed in seeds:
        for model in DIRECT_MODELS:
            pred, aux_pred, params, runtime, hp = direct_predictions(model, env, curves, aux, splits, seed, fast)
            for split in SPLITS:
                mask = splits == split
                if mask.any():
                    row = {"model_seed": seed, "model": model, "model_family": "direct", "split": split, "parameter_count": params, "runtime_seconds": runtime, "hyperparameters": json.dumps(hp), **metric_values(pred[mask], curves[mask], t)}
                    if aux_pred is not None:
                        row["physiology_rmse"] = ode.rmse(aux_pred[mask], aux[mask])
                    metrics.append(row)
            add_prediction_rows(preds, meta, t, curves, pred, model, seed, regimes)
            if aux_pred is not None:
                add_phys_rows(phys_preds, meta, t, aux, aux_pred, model, seed)
        for variant in STATE_SPACE_VARIANTS:
            candidates = []
            for dim in latent_dims:
                tic = time.perf_counter()
                pred, aux_pred, latent, params, log = train_state_space(env, curves, aux, splits, variant, dim, seed, fast)
                candidates.append((metric_values(pred[splits == "validation"], curves[splits == "validation"], t)["normalized_rmse"], dim, pred, aux_pred, latent, params, log, time.perf_counter() - tic))
                logs.append({"model_seed": seed, "model": variant, "latent_dim": dim, **log})
            _score, dim, pred, aux_pred, latent, params, log, runtime = sorted(candidates, key=lambda x: x[0])[0]
            for split in SPLITS:
                mask = splits == split
                if mask.any():
                    row = {"model_seed": seed, "model": variant, "model_family": "state_space", "split": split, "parameter_count": params, "runtime_seconds": runtime, "hyperparameters": json.dumps({"latent_dim": dim}), **metric_values(pred[mask], curves[mask], t)}
                    if aux_pred is not None:
                        row["physiology_rmse"] = ode.rmse(aux_pred[mask], aux[mask])
                    metrics.append(row)
            add_prediction_rows(preds, meta, t, curves, pred, variant, seed, regimes)
            if aux_pred is not None:
                add_phys_rows(phys_preds, meta, t, aux, aux_pred, variant, seed)
            add_representation_rows(reps, meta, latent, variant, seed, regimes)
    return pd.DataFrame(metrics), pd.DataFrame(preds), pd.DataFrame(phys_preds), pd.DataFrame(reps), pd.DataFrame(logs)


def add_prediction_rows(rows, meta, t, truth, pred, model, seed, regimes):
    for i, row in meta.iterrows():
        for k, tt in enumerate(t):
            rows.append({"model_seed": seed, "model": model, "culture_id": row.culture_id, "environment_id": row.environment_id, "split": row.split, "dominant_regime": regimes[i], "time_index": k, "time": tt, "P_true": truth[i, k], "P_pred": pred[i, k], "deployment_inputs": "temperature,pH,DO"})


def add_phys_rows(rows, meta, t, truth, pred, model, seed):
    for i, row in meta.iterrows():
        for k, tt in enumerate(t):
            out = {"model_seed": seed, "model": model, "culture_id": row.culture_id, "split": row.split, "time_index": k, "time": tt}
            for j, name in enumerate(PHYS_LABELS):
                out[f"{name}_true"] = truth[i, k, j]
                out[f"{name}_pred"] = pred[i, k, j]
            rows.append(out)


def add_representation_rows(rows, meta, latent, model, seed, regimes):
    features = np.hstack([latent.mean(axis=1), latent.max(axis=1), latent.min(axis=1), latent[:, -1, :] - latent[:, 0, :]])
    for i, row in meta.iterrows():
        out = {"model_seed": seed, "model": model, "culture_id": row.culture_id, "split": row.split, "dominant_regime": regimes[i]}
        for j, val in enumerate(features[i]):
            out[f"latent_feature_{j:02d}"] = float(val)
        rows.append(out)


def comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(metrics["model_seed"].unique()):
        val = metrics[(metrics.model_seed.eq(seed)) & metrics.split.eq("validation")]
        best_direct = val[val.model_family.eq("direct")].sort_values("normalized_rmse").iloc[0]["model"]
        best_ss = val[val.model_family.eq("state_space")].sort_values("normalized_rmse").iloc[0]["model"]
        held = metrics[(metrics.model_seed.eq(seed)) & metrics.split.eq("heldout_hole")]
        direct_err = float(held[held.model.eq(best_direct)]["normalized_rmse"].iloc[0])
        ss_err = float(held[held.model.eq(best_ss)]["normalized_rmse"].iloc[0])
        rows.append({"model_seed": seed, "best_direct_model": best_direct, "best_state_space_model": best_ss, "heldout_hole_direct_normalized_rmse": direct_err, "heldout_hole_state_space_normalized_rmse": ss_err, "delta_state_space_minus_direct": ss_err - direct_err, "state_space_wins": ss_err < direct_err})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    prefix = "ode_regime_pilot" if args.pilot else "ode_regime"
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics, preds, phys, reps, logs = evaluate(prefix, fast=args.fast)
    metrics.to_csv(DATA / f"{prefix}_model_metrics.csv", index=False)
    preds.to_csv(DATA / f"{prefix}_model_predictions.csv", index=False)
    phys.to_csv(DATA / f"{prefix}_physiology_predictions.csv", index=False)
    reps.to_csv(DATA / f"{prefix}_learned_representations.csv", index=False)
    logs.to_csv(DATA / f"{prefix}_training_log.csv", index=False)
    summary = comparison_summary(metrics)
    summary.to_csv(DATA / f"{prefix}_model_comparison_summary.csv", index=False)
    print(f"{prefix} model validation complete")
    print(metrics[metrics["split"].eq("heldout_hole")].groupby("model")["normalized_rmse"].mean().sort_values().head(12).to_string())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
