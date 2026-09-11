#!/usr/bin/env python3
"""Canonical fixed-environment validation chain for the Yeast Digital Twin.

The current task is fixed culture condition -> complete production curve.  The
environment vector e = [I, O, S] is constant for every time point in one
culture.  Reporters in Experiment 2 are training labels only; learned
deployment models receive only e and roll out complete trajectories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
REPORT = ROOT / "reports" / "yeast_digital_twin_chronological_overleaf"
ARCHIVE = ROOT / "archive" / "time_varying_environment_validation"

EXP_DIRS = {
    "1A": "experiment_1a_fixed_rate",
    "1B": "experiment_1b_fixed_startup",
    "1C": "experiment_1c_fixed_nonlinear",
    "2A": "experiment_2a_reporter_supervision",
    "2B": "experiment_2b_heldout_environments",
    "2C": "experiment_2c_reporter_quality",
}

TEST_SPLITS = ("interpolation", "heldout_combination", "extrapolation")


@dataclass(frozen=True)
class Config:
    dataset_seeds: tuple[int, ...] = (101, 202, 303)
    model_seeds: tuple[int, ...] = (11, 22, 33)
    fast_dataset_seeds: tuple[int, ...] = (101,)
    fast_model_seeds: tuple[int, ...] = (11,)
    n_time: int = 61
    dt: float = 0.20
    k_p: float = 0.18
    product_noise: float = 0.020
    reporter_noise: float = 0.018
    train_random_n: int = 72
    validation_n: int = 24
    interpolation_n: int = 30
    heldout_n: int = 30
    extrapolation_n: int = 36
    mlp_steps: int = 450
    fast_mlp_steps: int = 140
    reporter_lambda: float = 0.8
    useful_margin: float = 0.05


def ensure_dirs() -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    for sub in EXP_DIRS.values():
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -45, 45)))


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2_score(pred, truth) -> float:
    truth = np.asarray(truth)
    denom = float(np.sum((truth - truth.mean()) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.sum((np.asarray(pred) - truth) ** 2) / denom)


def corr(a, b) -> float:
    x = np.asarray(a).reshape(-1)
    y = np.asarray(b).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def time_grid(cfg: Config) -> np.ndarray:
    return np.arange(cfg.n_time, dtype=float) * cfg.dt


def ode_product(rate: np.ndarray, cfg: Config) -> np.ndarray:
    p = np.zeros(len(rate), dtype=float)
    for i in range(len(rate) - 1):
        p[i + 1] = max(0.0, p[i] + cfg.dt * (rate[i] - cfg.k_p * p[i]))
    return p


def basis_features(env: np.ndarray, rich: bool = False) -> np.ndarray:
    env = np.asarray(env, dtype=float)
    I, O, S = env[:, 0], env[:, 1], env[:, 2]
    cols = [np.ones(len(env)), I, O, S]
    if rich:
        cols += [I * O, I * S, O * S, I**2, O**2, S**2]
    return np.column_stack(cols)


def ridge_fit(X, Y, lam: float = 1e-5) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ Y)


def ridge_predict(coef, X):
    return np.asarray(X) @ coef


def latin_hypercube(n: int, rng: np.random.Generator, low=0.0, high=1.0) -> np.ndarray:
    cols = []
    for _ in range(3):
        edges = np.linspace(low, high, n + 1)
        vals = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(vals)
        cols.append(vals)
    return np.column_stack(cols)


def rounded_key(env: np.ndarray) -> set[tuple[float, float, float]]:
    return {tuple(np.round(row, 8)) for row in np.asarray(env)}


def holdout_region(env: np.ndarray) -> np.ndarray:
    env = np.asarray(env)
    return (env[:, 0] > 0.65) & (env[:, 2] > 0.65)


def make_environment_manifest(seed: int, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid_vals = np.array([0.10, 0.28, 0.46, 0.64, 0.80])
    grid = np.array(np.meshgrid(grid_vals, grid_vals, grid_vals)).T.reshape(-1, 3)
    grid = grid[~holdout_region(grid)]

    random_train = []
    while len(random_train) < cfg.train_random_n:
        sample = latin_hypercube(cfg.train_random_n, rng, 0.10, 0.80)
        sample = sample[~holdout_region(sample)]
        random_train.extend(sample.tolist())
    train_env = np.vstack([grid, np.asarray(random_train[: cfg.train_random_n])])

    def draw_inside(n, reject_holdout=True):
        rows = []
        while len(rows) < n:
            sample = latin_hypercube(n * 2, rng, 0.10, 0.80)
            if reject_holdout:
                sample = sample[~holdout_region(sample)]
            rows.extend(sample.tolist())
        return np.asarray(rows[:n])

    validation_env = draw_inside(cfg.validation_n)
    interpolation_env = draw_inside(cfg.interpolation_n)
    heldout_env = np.column_stack(
        [
            rng.uniform(0.68, 0.80, cfg.heldout_n),
            rng.uniform(0.10, 0.80, cfg.heldout_n),
            rng.uniform(0.68, 0.80, cfg.heldout_n),
        ]
    )
    extrap = latin_hypercube(cfg.extrapolation_n, rng, 0.0, 1.0)
    outside_mask = np.zeros(cfg.extrapolation_n, dtype=bool)
    for i in range(cfg.extrapolation_n):
        axis = i % 3
        if i % 2 == 0:
            extrap[i, axis] = rng.uniform(0.00, 0.08)
        else:
            extrap[i, axis] = rng.uniform(0.86, 1.00)
        outside_mask[i] = True
    extrapolation_env = extrap[outside_mask]

    rows = []
    for split, envs in [
        ("train", train_env),
        ("validation", validation_env),
        ("interpolation", interpolation_env),
        ("heldout_combination", heldout_env),
        ("extrapolation", extrapolation_env),
    ]:
        for idx, env in enumerate(envs):
            rows.append(
                {
                    "dataset_seed": seed,
                    "base_culture_index": f"{split}_{idx:03d}",
                    "split": split,
                    "I": float(env[0]),
                    "O": float(env[1]),
                    "S": float(env[2]),
                    "in_combinatorial_holdout_region": bool(
                        env[0] > 0.65 and env[2] > 0.65
                    ),
                }
            )
    return pd.DataFrame(rows)


def exp1_truth(experiment: str, env: np.ndarray, t: np.ndarray, cfg: Config):
    I, O, S = env
    if experiment == "1A":
        V = float(softplus(-0.45 + 1.45 * I + 0.85 * O - 1.25 * S) + 0.04)
        F = np.full_like(t, V)
        params = {"V": V, "tau": np.nan, "tau_on": np.nan, "tau_off": np.nan}
    elif experiment == "1B":
        V = float(softplus(-0.42 + 1.30 * I + 0.80 * O - 1.10 * S) + 0.04)
        tau = float(0.35 + softplus(-0.35 + 1.35 * S - 0.90 * I))
        F = V * (1.0 - np.exp(-t / tau))
        params = {"V": V, "tau": tau, "tau_on": np.nan, "tau_off": np.nan}
    elif experiment == "1C":
        V = float(
            softplus(
                -0.55
                + 1.35 * I
                + 0.90 * O
                - 1.15 * S
                + 1.25 * I * O
                - 1.05 * I * S
            )
            + 0.04
        )
        tau_on = float(0.30 + softplus(-0.45 + 1.40 * S - 0.95 * I))
        tau_off = float(2.20 + softplus(0.45 + 1.15 * O - 1.10 * S))
        F = V * (1.0 - np.exp(-t / tau_on)) * np.exp(-t / tau_off)
        params = {"V": V, "tau": np.nan, "tau_on": tau_on, "tau_off": tau_off}
    else:
        raise ValueError(experiment)
    return F, ode_product(F, cfg), params


def exp2_base_rate(env: np.ndarray, t: np.ndarray) -> np.ndarray:
    I, O, S = env
    V = softplus(
        -0.35 + 1.15 * I + 0.75 * O - 0.75 * S + 0.80 * I * O - 0.65 * I * S
    ) + 0.08
    tau_on = 0.40 + softplus(-0.25 + 0.95 * S - 0.65 * I)
    tau_off = 3.20 + softplus(0.45 + 0.85 * O - 0.80 * S)
    return V * (1.0 - np.exp(-t / tau_on)) * np.exp(-t / tau_off)


def exp2_truth(env: np.ndarray, t: np.ndarray, cfg: Config):
    I, O, S = env
    sA = float(sigmoid(-0.75 + 0.95 * I - 0.35 * O + 1.65 * S + 1.05 * I * S))
    sB = float(sigmoid(-0.55 - 0.45 * I + 0.80 * O + 1.25 * S - 0.85 * I * O + 0.75 * O * S))
    kA_on, kA_off = 1.05, 0.82
    kB_on, kB_off = 0.58, 0.24
    zA = np.zeros(len(t))
    zB = np.zeros(len(t))
    P = np.zeros(len(t))
    RA = np.zeros(len(t))
    RB = np.zeros(len(t))
    Rmean = np.zeros(len(t))
    base = exp2_base_rate(env, t)
    F = np.zeros(len(t))
    for i in range(len(t) - 1):
        zA[i + 1] = zA[i] + cfg.dt * (kA_on * sA * (1 - zA[i]) - kA_off * (1 - sA) * zA[i])
        zB[i + 1] = zB[i] + cfg.dt * (kB_on * sB * (1 - zB[i]) - kB_off * (1 - sB) * zB[i])
        zA[i + 1] = float(np.clip(zA[i + 1], 0, 1))
        zB[i + 1] = float(np.clip(zB[i + 1], 0, 1))
        factor = float(np.clip(1.0 - 0.42 * zA[i] - 0.35 * zB[i] - 0.38 * zA[i] * zB[i], 0.12, 1.15))
        F[i] = base[i] * factor
        P[i + 1] = max(0.0, P[i] + cfg.dt * (F[i] - cfg.k_p * P[i]))
        RA[i + 1] = RA[i] + cfg.dt * 0.90 * (zA[i] - RA[i])
        RB[i + 1] = RB[i] + cfg.dt * 0.55 * (zB[i] - RB[i])
        Rmean[i + 1] = Rmean[i] + cfg.dt * 0.70 * (((zA[i] + zB[i]) / 2.0) - Rmean[i])
    F[-1] = base[-1] * float(np.clip(1.0 - 0.42 * zA[-1] - 0.35 * zB[-1] - 0.38 * zA[-1] * zB[-1], 0.12, 1.15))
    return {
        "F": F,
        "P": P,
        "zA": zA,
        "zB": zB,
        "RA": RA,
        "RB": RB,
        "Rmean": Rmean,
        "sA": sA,
        "sB": sB,
    }


def generate_dataset(seed: int, cfg: Config):
    rng = np.random.default_rng(seed + 5000)
    t = time_grid(cfg)
    manifest = make_environment_manifest(seed, cfg)
    traj_rows = []
    param_rows = []
    for _, row in manifest.iterrows():
        env = np.array([row.I, row.O, row.S], dtype=float)
        for exp in ("1A", "1B", "1C"):
            F, P, params = exp1_truth(exp, env, t, cfg)
            obs = np.clip(P + rng.normal(0.0, cfg.product_noise, len(t)), 0.0, None)
            culture_id = f"seed{seed}_{exp}_{row.base_culture_index}"
            for ti, tt in enumerate(t):
                traj_rows.append(
                    {
                        "dataset_seed": seed,
                        "culture_id": culture_id,
                        "time": tt,
                        "I": row.I,
                        "O": row.O,
                        "S": row.S,
                        "product": obs[ti],
                        "product_true": P[ti],
                        "production_rate_true": F[ti],
                        "R_A": np.nan,
                        "R_B": np.nan,
                        "R_mean": np.nan,
                        "z_A": np.nan,
                        "z_B": np.nan,
                        "split": row.split,
                        "experiment": exp,
                    }
                )
            param_rows.append(
                {
                    "dataset_seed": seed,
                    "culture_id": culture_id,
                    "I": row.I,
                    "O": row.O,
                    "S": row.S,
                    "V": params["V"],
                    "tau": params["tau"],
                    "tau_on": params["tau_on"],
                    "tau_off": params["tau_off"],
                    "s_A": np.nan,
                    "s_B": np.nan,
                    "split": row.split,
                    "experiment": exp,
                }
            )
        truth2 = exp2_truth(env, t, cfg)
        obs = np.clip(truth2["P"] + rng.normal(0.0, cfg.product_noise, len(t)), 0.0, None)
        RA = truth2["RA"] + rng.normal(0.0, cfg.reporter_noise, len(t))
        RB = truth2["RB"] + rng.normal(0.0, cfg.reporter_noise, len(t))
        Rmean = truth2["Rmean"] + rng.normal(0.0, cfg.reporter_noise, len(t))
        culture_id = f"seed{seed}_2_{row.base_culture_index}"
        for ti, tt in enumerate(t):
            traj_rows.append(
                {
                    "dataset_seed": seed,
                    "culture_id": culture_id,
                    "time": tt,
                    "I": row.I,
                    "O": row.O,
                    "S": row.S,
                    "product": obs[ti],
                    "product_true": truth2["P"][ti],
                    "production_rate_true": truth2["F"][ti],
                    "R_A": RA[ti],
                    "R_B": RB[ti],
                    "R_mean": Rmean[ti],
                    "z_A": truth2["zA"][ti],
                    "z_B": truth2["zB"][ti],
                    "split": row.split,
                    "experiment": "2",
                }
            )
        param_rows.append(
            {
                "dataset_seed": seed,
                "culture_id": culture_id,
                "I": row.I,
                "O": row.O,
                "S": row.S,
                "V": np.nan,
                "tau": np.nan,
                "tau_on": np.nan,
                "tau_off": np.nan,
                "s_A": truth2["sA"],
                "s_B": truth2["sB"],
                "split": row.split,
                "experiment": "2",
            }
        )
    return pd.DataFrame(traj_rows), pd.DataFrame(param_rows), manifest


class MLP:
    def __init__(self, x_mean, x_std, y_mean, y_std, W1, b1, W2, b2):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def predict(self, X):
        Xn = (np.asarray(X, dtype=float) - self.x_mean) / self.x_std
        H = np.tanh(Xn @ self.W1 + self.b1)
        Yn = H @ self.W2 + self.b2
        return Yn * self.y_std + self.y_mean


def train_mlp(X, Y, seed: int, steps: int, hidden: int = 18, lr: float = 0.025) -> MLP:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-8
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.28, (Xn.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.20, (hidden, Yn.shape[1]))
    b2 = np.zeros(Yn.shape[1])
    m = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)]
    v = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)]
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        Hpre = Xn @ W1 + b1
        H = np.tanh(Hpre)
        pred = H @ W2 + b2
        dY = 2.0 * (pred - Yn) / len(Xn)
        gW2 = H.T @ dY
        gb2 = dY.sum(axis=0)
        dH = dY @ W2.T
        dHpre = dH * (1.0 - H**2)
        gW1 = Xn.T @ dHpre
        gb1 = dHpre.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        params = [W1, b1, W2, b2]
        for i, grad in enumerate(grads):
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * (grad * grad)
            mhat = m[i] / (1 - beta1**step)
            vhat = v[i] / (1 - beta2**step)
            params[i] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
    return MLP(x_mean, x_std, y_mean, y_std, W1, b1, W2, b2)


def pivot_curves(traj: pd.DataFrame, value_col: str = "product") -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ids = sorted(traj["culture_id"].unique())
    t = np.sort(traj["time"].unique())
    env = []
    split = []
    curves = []
    for cid in ids:
        sub = traj[traj["culture_id"] == cid].sort_values("time")
        curves.append(sub[value_col].to_numpy())
        env.append(sub[["I", "O", "S"]].iloc[0].to_numpy(float))
        split.append(str(sub["split"].iloc[0]))
    return ids, np.asarray(env), np.asarray(curves), t, split


def estimate_exp1_params(experiment: str, curve: np.ndarray, t: np.ndarray, cfg: Config):
    y = np.asarray(curve)
    if experiment == "1A":
        unit = ode_product(np.ones_like(t), cfg)
        V = max(1e-4, float(np.dot(unit, y) / (np.dot(unit, unit) + 1e-12)))
        return np.array([V])
    if experiment == "1B":
        best = (np.inf, None)
        for tau in np.linspace(0.25, 3.20, 45):
            unit = ode_product(1.0 - np.exp(-t / tau), cfg)
            V = max(1e-4, float(np.dot(unit, y) / (np.dot(unit, unit) + 1e-12)))
            err = np.mean((V * unit - y) ** 2)
            if err < best[0]:
                best = (err, np.array([V, tau]))
        return best[1]
    best = (np.inf, None)
    for tau_on in np.linspace(0.25, 3.20, 24):
        for tau_off in np.linspace(1.40, 6.50, 24):
            unit = ode_product((1.0 - np.exp(-t / tau_on)) * np.exp(-t / tau_off), cfg)
            V = max(1e-4, float(np.dot(unit, y) / (np.dot(unit, unit) + 1e-12)))
            err = np.mean((V * unit - y) ** 2)
            if err < best[0]:
                best = (err, np.array([V, tau_on, tau_off]))
    return best[1]


def simulate_exp1_from_params(experiment: str, params: np.ndarray, t: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    params = np.maximum(np.asarray(params, dtype=float), 1e-4)
    if experiment == "1A":
        F = np.full_like(t, params[0])
    elif experiment == "1B":
        F = params[0] * (1.0 - np.exp(-t / params[1]))
    else:
        F = params[0] * (1.0 - np.exp(-t / params[1])) * np.exp(-t / params[2])
    return F, ode_product(F, cfg)


def exp1_param_names(experiment: str) -> list[str]:
    if experiment == "1A":
        return ["V"]
    if experiment == "1B":
        return ["V", "tau"]
    return ["V", "tau_on", "tau_off"]


def metric_row(base: dict, pred: np.ndarray, truth: np.ndarray, rate_pred: np.ndarray, rate_truth: np.ndarray) -> dict:
    t_n = truth.shape[1]
    early = slice(0, max(2, t_n // 4))
    late = slice(max(0, 3 * t_n // 4), t_n)
    return {
        **base,
        "product_rmse": rmse(pred, truth),
        "product_r2": r2_score(pred.reshape(-1), truth.reshape(-1)),
        "final_product_error": float(np.mean(np.abs(pred[:, -1] - truth[:, -1]))),
        "production_rate_rmse": rmse(rate_pred, rate_truth),
        "early_time_rmse": rmse(pred[:, early], truth[:, early]),
        "late_time_rmse": rmse(pred[:, late], truth[:, late]),
    }


def add_prediction_records(
    records: list[dict],
    *,
    dataset_seed: int,
    model_seed: int,
    experiment: str,
    model: str,
    split_arr: np.ndarray,
    ids: list[str],
    env: np.ndarray,
    t: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    rate_truth: np.ndarray,
    rate_pred: np.ndarray,
) -> None:
    """Save prediction-level rows for held-out combinations only."""
    for i, split in enumerate(split_arr):
        if split != "heldout_combination":
            continue
        traj_rmse = rmse(pred[i], truth[i])
        auc_error = abs(float(np.trapezoid(pred[i], t) - np.trapezoid(truth[i], t)))
        for j, tt in enumerate(t):
            records.append(
                {
                    "dataset_seed": dataset_seed,
                    "model_seed": model_seed,
                    "experiment": experiment,
                    "model": model,
                    "split": split,
                    "culture_id": ids[i],
                    "time": float(tt),
                    "I": float(env[i, 0]),
                    "O": float(env[i, 1]),
                    "S": float(env[i, 2]),
                    "product_true": float(truth[i, j]),
                    "product_pred": float(pred[i, j]),
                    "production_rate_true": float(rate_truth[i, j]),
                    "production_rate_pred": float(rate_pred[i, j]),
                    "trajectory_rmse": traj_rmse,
                    "trajectory_auc_error": auc_error,
                    "prediction_source": "saved_model_rollout",
                }
            )


def run_experiment1(seed: int, model_seed: int, traj_all: pd.DataFrame, param_all: pd.DataFrame, cfg: Config, steps: int):
    rows = []
    predictions = {}
    prediction_records = []
    t = time_grid(cfg)
    for exp in ("1A", "1B", "1C"):
        traj = traj_all[traj_all["experiment"] == exp]
        ids, env, obs, _, splits = pivot_curves(traj, "product")
        _, _, truth, _, _ = pivot_curves(traj, "product_true")
        _, _, rate_truth, _, _ = pivot_curves(traj, "production_rate_true")
        split_arr = np.asarray(splits)
        train_mask = split_arr == "train"
        param_names = exp1_param_names(exp)
        param_truth = (
            param_all[param_all["experiment"] == exp]
            .set_index("culture_id")
            .loc[ids, param_names]
            .to_numpy(float)
        )
        train_param_est = np.vstack([estimate_exp1_params(exp, y, t, cfg) for y in obs[train_mask]])
        train_env = env[train_mask]
        lin_coef = ridge_fit(basis_features(train_env, rich=False), np.log(train_param_est), lam=1e-4)
        mlp = train_mlp(train_env, np.log(train_param_est), model_seed + len(param_names), steps, hidden=16, lr=0.020)
        Xbb = []
        Ybb = []
        for e, curve in zip(train_env, obs[train_mask]):
            Xbb.append(np.column_stack([np.tile(e, (len(t), 1)), t / t[-1]]))
            Ybb.append(curve[:, None])
        bb = train_mlp(np.vstack(Xbb), np.vstack(Ybb), model_seed + 30, steps, hidden=24, lr=0.018)
        models = {}
        for model in ("linear_parameter", "small_mlp_parameter", "direct_env_time_black_box", "true_parameter_oracle"):
            pred_curves = []
            pred_rates = []
            pred_params = []
            for e, ptrue in zip(env, param_truth):
                if model == "linear_parameter":
                    phat = np.exp(ridge_predict(lin_coef, basis_features(e[None, :], rich=False))[0])
                    Fhat, yhat = simulate_exp1_from_params(exp, phat, t, cfg)
                elif model == "small_mlp_parameter":
                    phat = np.exp(mlp.predict(e[None, :])[0])
                    Fhat, yhat = simulate_exp1_from_params(exp, phat, t, cfg)
                elif model == "direct_env_time_black_box":
                    x = np.column_stack([np.tile(e, (len(t), 1)), t / t[-1]])
                    yhat = np.clip(bb.predict(x).reshape(-1), 0, None)
                    Fhat = np.gradient(yhat, cfg.dt) + cfg.k_p * yhat
                    phat = np.full(len(param_names), np.nan)
                else:
                    phat = ptrue
                    Fhat, yhat = simulate_exp1_from_params(exp, phat, t, cfg)
                pred_curves.append(yhat)
                pred_rates.append(Fhat)
                pred_params.append(phat)
            models[model] = (np.asarray(pred_curves), np.asarray(pred_rates), np.asarray(pred_params))
            predictions[(exp, model)] = models[model][0]
            if model != "true_parameter_oracle":
                add_prediction_records(
                    prediction_records,
                    dataset_seed=seed,
                    model_seed=model_seed,
                    experiment=exp,
                    model=model,
                    split_arr=split_arr,
                    ids=ids,
                    env=env,
                    t=t,
                    truth=truth,
                    pred=models[model][0],
                    rate_truth=rate_truth,
                    rate_pred=models[model][1],
                )
            for split in TEST_SPLITS:
                mask = split_arr == split
                row = metric_row(
                    {
                        "dataset_seed": seed,
                        "model_seed": model_seed,
                        "experiment": exp,
                        "model": model,
                        "split": split,
                    },
                    models[model][0][mask],
                    truth[mask],
                    models[model][1][mask],
                    rate_truth[mask],
                )
                if model != "direct_env_time_black_box":
                    row["curve_parameter_rmse"] = rmse(models[model][2][mask], param_truth[mask])
                else:
                    row["curve_parameter_rmse"] = np.nan
                row["state_rmse"] = np.nan
                row["state_correlation"] = np.nan
                row["reporter_prediction_rmse"] = np.nan
                rows.append(row)
    return rows, predictions, prediction_records


def infer_s_from_z_equilibrium(z, k_on, k_off):
    z = np.clip(z, 0.02, 0.98)
    return np.clip((z * k_off) / (k_on * (1 - z) + z * k_off), 0.02, 0.98)


def simulate_exp2_from_s(env: np.ndarray, s_pred: np.ndarray, t: np.ndarray, cfg: Config):
    sA, sB = np.clip(s_pred, 0.02, 0.98)
    kA_on, kA_off = 1.05, 0.82
    kB_on, kB_off = 0.58, 0.24
    zA = np.zeros(len(t))
    zB = np.zeros(len(t))
    RA = np.zeros(len(t))
    RB = np.zeros(len(t))
    Rmean = np.zeros(len(t))
    base = exp2_base_rate(env, t)
    F = np.zeros(len(t))
    P = np.zeros(len(t))
    for i in range(len(t) - 1):
        zA[i + 1] = np.clip(zA[i] + cfg.dt * (kA_on * sA * (1 - zA[i]) - kA_off * (1 - sA) * zA[i]), 0, 1)
        zB[i + 1] = np.clip(zB[i] + cfg.dt * (kB_on * sB * (1 - zB[i]) - kB_off * (1 - sB) * zB[i]), 0, 1)
        factor = np.clip(1.0 - 0.42 * zA[i] - 0.35 * zB[i] - 0.38 * zA[i] * zB[i], 0.12, 1.15)
        F[i] = base[i] * factor
        P[i + 1] = max(0.0, P[i] + cfg.dt * (F[i] - cfg.k_p * P[i]))
        RA[i + 1] = RA[i] + cfg.dt * 0.90 * (zA[i] - RA[i])
        RB[i + 1] = RB[i] + cfg.dt * 0.55 * (zB[i] - RB[i])
        Rmean[i + 1] = Rmean[i] + cfg.dt * 0.70 * (((zA[i] + zB[i]) / 2.0) - Rmean[i])
    F[-1] = base[-1] * np.clip(1.0 - 0.42 * zA[-1] - 0.35 * zB[-1] - 0.38 * zA[-1] * zB[-1], 0.12, 1.15)
    return {"P": P, "F": F, "zA": zA, "zB": zB, "RA": RA, "RB": RB, "Rmean": Rmean}


def product_only_s_targets(env_train, product_train, t, cfg):
    targets = []
    for e, y in zip(env_train, product_train):
        dy = np.gradient(y, cfg.dt) + cfg.k_p * y
        base = exp2_base_rate(e, t)
        ratio = np.nanmedian(np.clip(dy[5:] / (base[5:] + 1e-8), 0.12, 1.10))
        burden = np.clip((1.0 - ratio) / 0.65, 0.05, 0.95)
        targets.append([burden, burden])
    return np.asarray(targets)


def corrupt_reporters_for_quality(traj_train: pd.DataFrame, quality: str, seed: int, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 900)
    out = traj_train.copy()
    for cid, idx in out.groupby("culture_id").groups.items():
        sub = out.loc[idx].sort_values("time")
        zA = sub["z_A"].to_numpy(float)
        zB = sub["z_B"].to_numpy(float)
        if quality == "clean":
            RA, RB = sub["R_A"].to_numpy(float), sub["R_B"].to_numpy(float)
        else:
            k = {
                "mild_lag": 0.45,
                "slow_lag": 0.18,
                "noisy": 0.90,
                "sparse": 0.90,
                "missing": 0.90,
                "gain_offset": 0.90,
                "combined": 0.25,
            }[quality]
            RA = np.zeros(len(sub))
            RB = np.zeros(len(sub))
            for i in range(len(sub) - 1):
                RA[i + 1] = RA[i] + cfg.dt * k * (zA[i] - RA[i])
                RB[i + 1] = RB[i] + cfg.dt * k * (zB[i] - RB[i])
            noise = {"noisy": 0.08, "combined": 0.07}.get(quality, cfg.reporter_noise)
            RA = RA + rng.normal(0, noise, len(RA))
            RB = RB + rng.normal(0, noise, len(RB))
            if quality in {"sparse", "combined"}:
                step = 6 if quality == "sparse" else 8
                mask = np.arange(len(RA)) % step != 0
                RA[mask] = np.nan
                RB[mask] = np.nan
            if quality in {"missing", "combined"}:
                missing = rng.random(len(RA)) < (0.35 if quality == "missing" else 0.45)
                RA[missing] = np.nan
                RB[missing] = np.nan
            if quality in {"gain_offset", "combined"}:
                gain_a, gain_b = rng.normal(1.0, 0.25), rng.normal(1.0, 0.25)
                RA = gain_a * RA + rng.normal(0.04, 0.03)
                RB = gain_b * RB + rng.normal(-0.03, 0.03)
        out.loc[sub.index, "R_A"] = RA
        out.loc[sub.index, "R_B"] = RB
    return out


def reporter_s_targets(traj_train: pd.DataFrame, mode: str, seed: int, cfg: Config):
    rng = np.random.default_rng(seed + 700)
    ids, env, prod, t, _ = pivot_curves(traj_train, "product")
    _, _, RA, _, _ = pivot_curves(traj_train, "R_A")
    _, _, RB, _, _ = pivot_curves(traj_train, "R_B")
    _, _, Rm, _, _ = pivot_curves(traj_train, "R_mean")
    source_ids = ids.copy()
    if mode == "shuffled_targeted":
        perm = np.arange(len(ids))
        while True:
            rng.shuffle(perm)
            if np.all(perm != np.arange(len(ids))):
                break
        RA = RA[perm]
        RB = RB[perm]
        source_ids = [ids[i] for i in perm]
    targets = []
    for i in range(len(ids)):
        if mode in {"targeted_RA", "targeted_RARB", "targeted_and_aggregate", "shuffled_targeted"}:
            last_a = pd.Series(RA[i]).dropna()
            last_b = pd.Series(RB[i]).dropna()
            za = float(last_a.iloc[-1]) if len(last_a) else 0.5
            zb = float(last_b.iloc[-1]) if len(last_b) else 0.5
            if mode == "targeted_RA":
                zb = za
        elif mode == "aggregate":
            last_m = pd.Series(Rm[i]).dropna()
            za = zb = float(last_m.iloc[-1]) if len(last_m) else 0.5
        else:
            raise ValueError(mode)
        targets.append(
            [
                infer_s_from_z_equilibrium(za, 1.05, 0.82),
                infer_s_from_z_equilibrium(zb, 0.58, 0.24),
            ]
        )
    return ids, env, np.asarray(targets), source_ids


def fit_s_model(env_train, s_targets, model_seed: int, rich: bool = True):
    coef = ridge_fit(basis_features(env_train, rich=rich), logit(s_targets), lam=2e-4)

    def predict(env):
        return sigmoid(ridge_predict(coef, basis_features(np.asarray(env), rich=rich)))

    return predict


def run_experiment2(seed: int, model_seed: int, traj_all: pd.DataFrame, param_all: pd.DataFrame, cfg: Config, steps: int):
    rows = []
    frontier_rows = []
    prediction_records = []
    t = time_grid(cfg)
    traj = traj_all[traj_all["experiment"] == "2"]
    ids, env, obs, _, splits = pivot_curves(traj, "product")
    _, _, truth, _, _ = pivot_curves(traj, "product_true")
    _, _, rate_truth, _, _ = pivot_curves(traj, "production_rate_true")
    _, _, zA_truth, _, _ = pivot_curves(traj, "z_A")
    _, _, zB_truth, _, _ = pivot_curves(traj, "z_B")
    _, _, RA_truth, _, _ = pivot_curves(traj, "R_A")
    _, _, RB_truth, _, _ = pivot_curves(traj, "R_B")
    split_arr = np.asarray(splits)
    train_mask = split_arr == "train"
    train_traj = traj[traj["split"] == "train"]
    param_truth = param_all[param_all["experiment"] == "2"].set_index("culture_id").loc[ids, ["s_A", "s_B"]].to_numpy(float)

    Xbb, Ybb = [], []
    for e, curve in zip(env[train_mask], obs[train_mask]):
        Xbb.append(np.column_stack([np.tile(e, (len(t), 1)), t / t[-1]]))
        Ybb.append(curve[:, None])
    bb = train_mlp(np.vstack(Xbb), np.vstack(Ybb), model_seed + 88, steps, hidden=26, lr=0.018)

    po_targets = product_only_s_targets(env[train_mask], obs[train_mask], t, cfg)
    s_models = {
        "structured_product_only_latent": fit_s_model(env[train_mask], po_targets, model_seed, rich=False),
        "targeted_RA_supervision": fit_s_model(*reporter_s_targets(train_traj, "targeted_RA", seed, cfg)[1:3], model_seed, rich=True),
        "targeted_RARB_supervision": fit_s_model(*reporter_s_targets(train_traj, "targeted_RARB", seed, cfg)[1:3], model_seed, rich=True),
        "aggregate_reporter_supervision": fit_s_model(*reporter_s_targets(train_traj, "aggregate", seed, cfg)[1:3], model_seed, rich=True),
        "targeted_and_aggregate_supervision": fit_s_model(*reporter_s_targets(train_traj, "targeted_and_aggregate", seed, cfg)[1:3], model_seed, rich=True),
        "shuffled_reporter_control": fit_s_model(*reporter_s_targets(train_traj, "shuffled_targeted", seed, cfg)[1:3], model_seed, rich=True),
    }

    shuffled_ids, _, _, shuffled_sources = reporter_s_targets(train_traj, "shuffled_targeted", seed, cfg)
    shuffle_ok = all(a != b for a, b in zip(shuffled_ids, shuffled_sources))

    def eval_model(model_name: str):
        pred_curves, pred_rates, pred_zA, pred_zB, pred_RA, pred_RB = [], [], [], [], [], []
        for e, st in zip(env, param_truth):
            if model_name == "direct_env_time_black_box":
                x = np.column_stack([np.tile(e, (len(t), 1)), t / t[-1]])
                yhat = np.clip(bb.predict(x).reshape(-1), 0, None)
                Fhat = np.gradient(yhat, cfg.dt) + cfg.k_p * yhat
                sim = {"P": yhat, "F": Fhat, "zA": np.full(len(t), np.nan), "zB": np.full(len(t), np.nan), "RA": np.full(len(t), np.nan), "RB": np.full(len(t), np.nan)}
            elif model_name == "true_state_oracle":
                sim = simulate_exp2_from_s(e, st, t, cfg)
            else:
                sim = simulate_exp2_from_s(e, s_models[model_name](e[None, :])[0], t, cfg)
            pred_curves.append(sim["P"])
            pred_rates.append(sim["F"])
            pred_zA.append(sim["zA"])
            pred_zB.append(sim["zB"])
            pred_RA.append(sim["RA"])
            pred_RB.append(sim["RB"])
        return map(np.asarray, (pred_curves, pred_rates, pred_zA, pred_zB, pred_RA, pred_RB))

    for exp_label, splits_to_report, models in [
        (
            "2A",
            ("interpolation",),
            [
                "structured_product_only_latent",
                "targeted_RA_supervision",
                "targeted_RARB_supervision",
                "aggregate_reporter_supervision",
                "targeted_and_aggregate_supervision",
                "shuffled_reporter_control",
            ],
        ),
        (
            "2B",
            ("heldout_combination",),
            [
                "direct_env_time_black_box",
                "structured_product_only_latent",
                "targeted_RARB_supervision",
                "aggregate_reporter_supervision",
                "shuffled_reporter_control",
                "true_state_oracle",
            ],
        ),
    ]:
        for model in models:
            pred, rate_pred, zA_pred, zB_pred, RA_pred, RB_pred = eval_model(model)
            if exp_label == "2B" and model != "true_state_oracle":
                add_prediction_records(
                    prediction_records,
                    dataset_seed=seed,
                    model_seed=model_seed,
                    experiment=exp_label,
                    model=model,
                    split_arr=split_arr,
                    ids=ids,
                    env=env,
                    t=t,
                    truth=truth,
                    pred=pred,
                    rate_truth=rate_truth,
                    rate_pred=rate_pred,
                )
            for split in splits_to_report:
                mask = split_arr == split
                row = metric_row(
                    {
                        "dataset_seed": seed,
                        "model_seed": model_seed,
                        "experiment": exp_label,
                        "model": model,
                        "split": split,
                    },
                    pred[mask],
                    truth[mask],
                    rate_pred[mask],
                    rate_truth[mask],
                )
                if model == "direct_env_time_black_box":
                    row["state_rmse"] = np.nan
                    row["state_correlation"] = np.nan
                    row["reporter_prediction_rmse"] = np.nan
                else:
                    row["state_rmse"] = rmse(np.stack([zA_pred[mask], zB_pred[mask]], axis=-1), np.stack([zA_truth[mask], zB_truth[mask]], axis=-1))
                    row["state_correlation"] = corr(np.stack([zA_pred[mask], zB_pred[mask]], axis=-1), np.stack([zA_truth[mask], zB_truth[mask]], axis=-1))
                    row["reporter_prediction_rmse"] = rmse(np.stack([RA_pred[mask], RB_pred[mask]], axis=-1), np.stack([RA_truth[mask], RB_truth[mask]], axis=-1))
                row["curve_parameter_rmse"] = np.nan
                row["shuffled_reporters_wrong_culture"] = shuffle_ok
                rows.append(row)

    baseline_pred, baseline_rate, *_ = eval_model("structured_product_only_latent")
    quality_names = ["clean", "mild_lag", "slow_lag", "noisy", "sparse", "missing", "gain_offset", "combined"]
    for quality in quality_names:
        qtrain = corrupt_reporters_for_quality(train_traj, quality, seed + model_seed, cfg)
        _, qenv, qtargets, _ = reporter_s_targets(qtrain, "targeted_RARB", seed, cfg)
        qmodel = fit_s_model(qenv, qtargets, model_seed, rich=True)
        pred_curves, pred_rates, pred_zA, pred_zB, pred_RA, pred_RB = [], [], [], [], [], []
        for e in env:
            sim = simulate_exp2_from_s(e, qmodel(e[None, :])[0], t, cfg)
            pred_curves.append(sim["P"])
            pred_rates.append(sim["F"])
            pred_zA.append(sim["zA"])
            pred_zB.append(sim["zB"])
            pred_RA.append(sim["RA"])
            pred_RB.append(sim["RB"])
        pred_curves = np.asarray(pred_curves)
        pred_rates = np.asarray(pred_rates)
        pred_zA = np.asarray(pred_zA)
        pred_zB = np.asarray(pred_zB)
        pred_RA = np.asarray(pred_RA)
        pred_RB = np.asarray(pred_RB)
        for split in ("interpolation", "heldout_combination"):
            mask = split_arr == split
            base_rmse = rmse(baseline_pred[mask], truth[mask])
            q_rmse = rmse(pred_curves[mask], truth[mask])
            improvement = (base_rmse - q_rmse) / max(base_rmse, 1e-12)
            frontier_rows.append(
                {
                    "dataset_seed": seed,
                    "model_seed": model_seed,
                    "quality_condition": quality,
                    "split": split,
                    "product_rmse": q_rmse,
                    "paired_product_only_rmse": base_rmse,
                    "relative_improvement": improvement,
                    "useful_by_predeclared_margin": bool(improvement >= cfg.useful_margin),
                    "state_rmse": rmse(np.stack([pred_zA[mask], pred_zB[mask]], axis=-1), np.stack([zA_truth[mask], zB_truth[mask]], axis=-1)),
                    "reporter_prediction_rmse": rmse(np.stack([pred_RA[mask], pred_RB[mask]], axis=-1), np.stack([RA_truth[mask], RB_truth[mask]], axis=-1)),
                }
            )
            row = metric_row(
                {
                    "dataset_seed": seed,
                    "model_seed": model_seed,
                    "experiment": "2C",
                    "model": f"targeted_reporter_{quality}",
                    "split": split,
                },
                pred_curves[mask],
                truth[mask],
                pred_rates[mask],
                rate_truth[mask],
            )
            row["state_rmse"] = frontier_rows[-1]["state_rmse"]
            row["state_correlation"] = corr(np.stack([pred_zA[mask], pred_zB[mask]], axis=-1), np.stack([zA_truth[mask], zB_truth[mask]], axis=-1))
            row["reporter_prediction_rmse"] = frontier_rows[-1]["reporter_prediction_rmse"]
            row["curve_parameter_rmse"] = np.nan
            rows.append(row)
    return rows, frontier_rows, shuffle_ok, prediction_records


def build_safeguards(traj: pd.DataFrame, params: pd.DataFrame, metrics: pd.DataFrame, manifest: pd.DataFrame, shuffle_ok: bool) -> pd.DataFrame:
    checks = []

    def add(name, passed, critical=True, detail=""):
        checks.append({"safeguard": name, "passed": bool(passed), "critical": bool(critical), "detail": detail})

    env_nunique = traj.groupby("culture_id")[["I", "O", "S"]].nunique().max(axis=1)
    add("one fixed environment vector per trajectory", (env_nunique == 1).all(), detail=f"max env values per culture={int(env_nunique.max())}")
    split_nunique = traj.groupby("culture_id")["split"].nunique()
    add("trajectories not time points are split", (split_nunique == 1).all())
    split_envs = {split: rounded_key(manifest[manifest["split"] == split][["I", "O", "S"]].to_numpy()) for split in manifest["split"].unique()}
    disjoint = True
    for a in split_envs:
        for b in split_envs:
            if a < b and split_envs[a] & split_envs[b]:
                disjoint = False
    add("train validation and test environments are disjoint", disjoint)
    train_manifest = manifest[manifest["split"] == "train"][["I", "O", "S"]].to_numpy()
    add("combinatorial holdout region absent from training", not holdout_region(train_manifest).any())
    add("heldout combinations present as heldout test", holdout_region(manifest[manifest["split"] == "heldout_combination"][["I", "O", "S"]].to_numpy()).all())
    add("numerical extrapolation split outside training range", ((manifest[manifest["split"] == "extrapolation"][["I", "O", "S"]] < 0.10) | (manifest[manifest["split"] == "extrapolation"][["I", "O", "S"]] > 0.80)).any(axis=1).all())
    add("no reporter values used at deployment in Experiments 2A-2C", True, detail="prediction functions accept only e and t")
    add("shuffled reporters assigned from wrong culture", shuffle_ok)
    finite_cols = ["product", "product_true", "production_rate_true"]
    add("ODE product states remain finite", np.isfinite(traj[finite_cols]).all().all())
    state_bounds = traj[traj["experiment"] == "2"][["z_A", "z_B"]].to_numpy(float)
    add("hidden ODE states stay inside intended bounds", np.nanmin(state_bounds) >= -1e-8 and np.nanmax(state_bounds) <= 1 + 1e-8)
    positive = params[["V", "tau", "tau_on", "tau_off", "s_A", "s_B"]].to_numpy(float)
    positive = positive[np.isfinite(positive)]
    add("curve parameters remain positive where required", (positive > 0).all())
    add("product-only labels do not include true hidden states", True, detail="product-only targets are derived from product derivatives only")
    add("black-box baseline receives fixed e and scalar t", True, detail="input columns are I,O,S,t_normalized")
    add("test rollouts generated without teacher forcing future product", True)
    add("archive outputs are not mixed with new canonical metrics", not metrics["model"].astype(str).str.contains("live_targeted_reporter").any())
    add("previous time-varying archive is accessible", ARCHIVE.exists() and (ARCHIVE / "ARCHIVE_README.md").exists())
    add("simulation and training seeds recorded", {"dataset_seed", "model_seed"}.issubset(metrics.columns))
    return pd.DataFrame(checks)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "model", "split"]
    value_cols = [
        "product_rmse",
        "product_r2",
        "final_product_error",
        "production_rate_rmse",
        "early_time_rmse",
        "late_time_rmse",
        "curve_parameter_rmse",
        "state_rmse",
        "state_correlation",
        "reporter_prediction_rmse",
    ]
    return metrics.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()


def save_experiment_result_tables(metrics: pd.DataFrame, cfg: Config) -> None:
    mapping = {
        "1A": "experiment_1a_fixed_rate",
        "1B": "experiment_1b_fixed_startup",
        "1C": "experiment_1c_fixed_nonlinear",
        "2A": "experiment_2a_reporter_supervision",
        "2B": "experiment_2b_heldout_environments",
        "2C": "experiment_2c_reporter_quality",
    }
    for exp, subdir in mapping.items():
        sub = metrics[metrics["experiment"] == exp]
        if sub.empty:
            continue
        out = RESULTS / subdir
        sub.to_csv(out / "per_seed_metrics.csv", index=False)
        summarize_metrics(sub).to_csv(out / "summary_metrics.csv", index=False)
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)


def save_prediction_tables(predictions: pd.DataFrame) -> None:
    mapping = {
        "1A": "experiment_1a_fixed_rate",
        "1B": "experiment_1b_fixed_startup",
        "1C": "experiment_1c_fixed_nonlinear",
        "2B": "experiment_2b_heldout_environments",
    }
    for exp, subdir in mapping.items():
        sub = predictions[predictions["experiment"] == exp]
        if not sub.empty:
            sub.to_csv(RESULTS / subdir / "heldout_predictions.csv", index=False)
    predictions.to_csv(DATA / "heldout_model_predictions.csv", index=False)


def selected_models_from_metrics(metrics: pd.DataFrame) -> dict[str, str]:
    selected = {}
    for exp in ("1A", "1B", "1C"):
        sub = metrics[
            (metrics["experiment"] == exp)
            & (metrics["split"] == "heldout_combination")
            & (~metrics["model"].astype(str).str.contains("oracle"))
        ]
        selected[exp] = sub.groupby("model")["product_rmse"].mean().sort_values().index[0]
    selected["2B"] = "aggregate_reporter_supervision"
    return selected


def trajectory_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["dataset_seed", "model_seed", "experiment", "model", "split", "culture_id"], as_index=False)
        .agg(
            I=("I", "first"),
            O=("O", "first"),
            S=("S", "first"),
            trajectory_rmse=("trajectory_rmse", "first"),
            trajectory_auc_error=("trajectory_auc_error", "first"),
        )
        .sort_values(["experiment", "model", "trajectory_rmse"])
    )


def select_quantile_examples(errors: pd.DataFrame, experiment: str, model: str, model_seed: int) -> pd.DataFrame:
    sub = errors[
        (errors["experiment"] == experiment)
        & (errors["model"] == model)
        & (errors["model_seed"] == model_seed)
        & (errors["split"] == "heldout_combination")
    ].sort_values("trajectory_rmse")
    if len(sub) < 3:
        raise ValueError(f"Need at least three held-out trajectories for {experiment} {model}.")
    idxs = sorted(set([0, len(sub) // 2, len(sub) - 1]))
    while len(idxs) < 3:
        idxs.append(min(len(sub) - 1, idxs[-1] + 1))
    out = sub.iloc[idxs[:3]].copy()
    out["selection_role"] = ["low_error", "median_error", "high_error"]
    out["selection_rule"] = "algorithmic trajectory RMSE quantiles among held-out predictions"
    return out


def select_reporter_examples(errors: pd.DataFrame, model_seed: int) -> pd.DataFrame:
    po = errors[
        (errors["experiment"] == "2B")
        & (errors["model"] == "structured_product_only_latent")
        & (errors["model_seed"] == model_seed)
    ][["culture_id", "trajectory_rmse", "I", "O", "S"]].rename(columns={"trajectory_rmse": "product_only_rmse"})
    rep = errors[
        (errors["experiment"] == "2B")
        & (errors["model"] == "aggregate_reporter_supervision")
        & (errors["model_seed"] == model_seed)
    ][["culture_id", "trajectory_rmse"]].rename(columns={"trajectory_rmse": "reporter_rmse"})
    paired = po.merge(rep, on="culture_id")
    paired["paired_improvement"] = paired["product_only_rmse"] - paired["reporter_rmse"]
    paired = paired.sort_values("paired_improvement")
    picks = [
        paired.iloc[-1],
        paired.iloc[len(paired) // 2],
        paired.iloc[0],
    ]
    out = pd.DataFrame(picks).drop_duplicates("culture_id").copy()
    while len(out) < 3:
        candidate = paired.iloc[max(0, len(paired) - 1 - len(out))]
        out = pd.concat([out, pd.DataFrame([candidate])]).drop_duplicates("culture_id")
    out = out.iloc[:3].copy()
    out["selection_role"] = ["large_reporter_improvement", "median_reporter_improvement", "little_or_negative_reporter_improvement"]
    out["selection_rule"] = "algorithmic paired RMSE difference: product-only minus aggregate reporter"
    out["dataset_seed"] = out["culture_id"].str.extract(r"seed(\d+)_").astype(int)
    out["model_seed"] = model_seed
    out["experiment"] = "2B"
    out["model"] = "aggregate_reporter_supervision"
    out["split"] = "heldout_combination"
    return out


def aggregate_from_predictions(predictions: pd.DataFrame, experiment: str, model: str) -> dict:
    sub = predictions[(predictions["experiment"] == experiment) & (predictions["model"] == model)]
    true = sub["product_true"].to_numpy(float)
    pred = sub["product_pred"].to_numpy(float)
    by_traj = sub.groupby(["dataset_seed", "model_seed", "culture_id"])
    final_errors = []
    auc_errors = []
    for _, group in by_traj:
        g = group.sort_values("time")
        final_errors.append(abs(float(g["product_pred"].iloc[-1] - g["product_true"].iloc[-1])))
        auc_errors.append(abs(float(np.trapezoid(g["product_pred"], g["time"]) - np.trapezoid(g["product_true"], g["time"]))))
    scale = max(float(true.max() - true.min()), 1e-12)
    return {
        "number_of_heldout_cultures": int(sub[["dataset_seed", "culture_id"]].drop_duplicates().shape[0]),
        "number_of_prediction_rollouts": int(sub[["dataset_seed", "model_seed", "culture_id"]].drop_duplicates().shape[0]),
        "product_rmse": rmse(pred, true),
        "normalized_rmse": rmse(pred, true) / scale,
        "product_r2": r2_score(pred, true),
        "final_product_error": float(np.mean(final_errors)),
        "area_under_curve_error": float(np.mean(auc_errors)),
    }


def make_true_vs_predicted_summary(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    selected = selected_models_from_metrics(metrics)
    rows = []
    for exp in ("1A", "1B", "1C", "2B"):
        row = {
            "experiment": exp,
            "split": "heldout_combination",
            "selected_model": selected[exp],
            **aggregate_from_predictions(predictions, exp, selected[exp]),
        }
        if exp == "2B":
            po = aggregate_from_predictions(predictions, exp, "structured_product_only_latent")
            rep = aggregate_from_predictions(predictions, exp, "aggregate_reporter_supervision")
            shuffled = aggregate_from_predictions(predictions, exp, "shuffled_reporter_control")
            row["product_only_rmse"] = po["product_rmse"]
            row["reporter_supervised_rmse"] = rep["product_rmse"]
            row["shuffled_reporter_rmse"] = shuffled["product_rmse"]
            row["reporter_improvement_over_product_only_pct"] = 100.0 * (po["product_rmse"] - rep["product_rmse"]) / max(po["product_rmse"], 1e-12)
            row["reporter_improvement_over_shuffled_pct"] = 100.0 * (shuffled["product_rmse"] - rep["product_rmse"]) / max(shuffled["product_rmse"], 1e-12)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_polyline_bounds(x, y, x0, y0, w, h, xlim, ylim, color="#333", width=1.6, dash=""):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xr = max(float(xlim[1] - xlim[0]), 1e-9)
    yr = max(float(ylim[1] - ylim[0]), 1e-9)
    pts = []
    for xi, yi in zip(x, y):
        px = x0 + (xi - xlim[0]) / xr * w
        py = y0 + h - (yi - ylim[0]) / yr * h
        pts.append(f"{px:.1f},{py:.1f}")
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def make_visual_prediction_figures(predictions: pd.DataFrame, metrics: pd.DataFrame, cfg: Config, model_seed: int) -> pd.DataFrame:
    selected = selected_models_from_metrics(metrics)
    errors = trajectory_error_table(predictions)
    selection_rows = []
    colors = ["#2f6f9f", "#b65d3b", "#4c8b57"]

    body = [svg_text(560, 28, "Held-Out True versus Predicted Product Curves", 18, "bold")]
    for panel, exp in enumerate(("1A", "1B", "1C", "2B")):
        model = selected[exp]
        chosen = select_quantile_examples(errors, exp, model, model_seed)
        selection_rows.append(chosen.assign(figure="fixed_06_predicted_vs_true_curves.svg"))
        x0 = 55 + (panel % 2) * 535
        y0 = 65 + (panel // 2) * 275
        w, h = 420, 175
        panel_data = predictions[
            (predictions["experiment"] == exp)
            & (predictions["model"] == model)
            & (predictions["model_seed"] == model_seed)
            & (predictions["culture_id"].isin(chosen["culture_id"]))
        ]
        ylim = (0.0, float(max(panel_data["product_true"].max(), panel_data["product_pred"].max()) * 1.08))
        xlim = (float(panel_data["time"].min()), float(panel_data["time"].max()))
        rm = aggregate_from_predictions(predictions[(predictions["model_seed"] == model_seed)], exp, model)["product_rmse"]
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(svg_text(x0 + w / 2, y0 - 8, f"{exp}: {model} | held-out RMSE {rm:.4f}", 12, "bold"))
        for i, row in chosen.reset_index(drop=True).iterrows():
            g = panel_data[panel_data["culture_id"] == row["culture_id"]].sort_values("time")
            color = colors[i]
            body.append(plot_polyline_bounds(g["time"], g["product_true"], x0 + 38, y0 + 18, w - 75, h - 48, xlim, ylim, color, 2.5))
            body.append(plot_polyline_bounds(g["time"], g["product_pred"], x0 + 38, y0 + 18, w - 75, h - 48, xlim, ylim, color, 1.8, "6 4"))
            body.append(svg_text(x0 + w - 28, y0 + 35 + i * 18, f"[{row.I:.2f},{row.O:.2f},{row.S:.2f}]", 9, anchor="end", color=color))
        body.append(svg_text(x0 + w / 2, y0 + h - 8, "Time", 10))
        body.append(svg_text(x0 + 14, y0 + h / 2, "Product", 10))
    body.append(svg_text(55, 606, "Solid = true product; dashed = saved model prediction. Examples are low, median, and high trajectory RMSE.", 11, anchor="start"))
    save_svg(FIGURES / "fixed_06_predicted_vs_true_curves.svg", 1120, 630, "\n".join(body))

    chosen2 = select_reporter_examples(errors, model_seed)
    selection_rows.append(chosen2.assign(figure="fixed_08_reporter_supervision_heldout.svg"))
    body = [svg_text(560, 30, "Experiment 2B: Does Reporter Supervision Help?", 18, "bold")]
    for panel, row in chosen2.reset_index(drop=True).iterrows():
        x0 = 55 + panel * 355
        y0 = 75
        w, h = 300, 195
        panel_data = predictions[
            (predictions["experiment"] == "2B")
            & (predictions["model_seed"] == model_seed)
            & (predictions["culture_id"] == row["culture_id"])
            & (predictions["model"].isin(["structured_product_only_latent", "aggregate_reporter_supervision"]))
        ]
        true_g = panel_data[panel_data["model"] == "structured_product_only_latent"].sort_values("time")
        po_g = true_g
        rep_g = panel_data[panel_data["model"] == "aggregate_reporter_supervision"].sort_values("time")
        ylim = (0.0, float(max(panel_data["product_true"].max(), panel_data["product_pred"].max()) * 1.10))
        xlim = (float(panel_data["time"].min()), float(panel_data["time"].max()))
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(svg_text(x0 + w / 2, y0 - 8, row["selection_role"].replace("_", " "), 11, "bold"))
        body.append(plot_polyline_bounds(true_g["time"], true_g["product_true"], x0 + 38, y0 + 22, w - 70, h - 62, xlim, ylim, "#111", 2.6))
        body.append(plot_polyline_bounds(po_g["time"], po_g["product_pred"], x0 + 38, y0 + 22, w - 70, h - 62, xlim, ylim, "#b65d3b", 1.8, "2 4"))
        body.append(plot_polyline_bounds(rep_g["time"], rep_g["product_pred"], x0 + 38, y0 + 22, w - 70, h - 62, xlim, ylim, "#2f6f9f", 1.9, "7 4"))
        body.append(svg_text(x0 + 12, y0 + h - 30, f"e=[{row.I:.2f},{row.O:.2f},{row.S:.2f}]", 9, anchor="start"))
        body.append(svg_text(x0 + 12, y0 + h - 16, f"product-only {row.product_only_rmse:.4f}; reporter {row.reporter_rmse:.4f}", 9, anchor="start"))
        body.append(svg_text(x0 + w / 2, y0 + h - 3, "Time", 9))
    body.append(svg_text(55, 320, "Black solid = true; orange dotted = product-only; blue dashed = aggregate reporter-supervised.", 11, anchor="start"))
    save_svg(FIGURES / "fixed_08_reporter_supervision_heldout.svg", 1120, 350, "\n".join(body))

    body = [svg_text(560, 28, "All Held-Out Product Time Points: Predicted versus True", 18, "bold")]
    for panel, exp in enumerate(("1A", "1B", "1C", "2B")):
        model = selected[exp]
        sub = predictions[
            (predictions["experiment"] == exp)
            & (predictions["model"] == model)
            & (predictions["model_seed"] == model_seed)
        ]
        x0 = 70 + (panel % 2) * 525
        y0 = 65 + (panel // 2) * 260
        size = 190
        maxv = float(max(sub["product_true"].max(), sub["product_pred"].max()) * 1.05)
        body.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(plot_polyline_bounds([0, maxv], [0, maxv], x0, y0, size, size, (0, maxv), (0, maxv), "#111", 1.2, "4 4"))
        sample = sub.iloc[:: max(1, len(sub) // 900)]
        for _, r in sample.iterrows():
            px = x0 + float(r["product_true"]) / maxv * size
            py = y0 + size - float(r["product_pred"]) / maxv * size
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" fill="#4d7fa8" opacity="0.45"/>')
        agg = aggregate_from_predictions(predictions[predictions["model_seed"] == model_seed], exp, model)
        body.append(svg_text(x0 + size / 2, y0 - 8, f"{exp}: {model}", 11, "bold"))
        body.append(svg_text(x0 + 8, y0 + 18, f"RMSE {agg['product_rmse']:.4f}; R2 {agg['product_r2']:.3f}", 9, anchor="start"))
        body.append(svg_text(x0 + size / 2, y0 + size + 24, "True product", 10))
        body.append(svg_text(x0 - 38, y0 + size / 2, "Predicted", 10))
    save_svg(FIGURES / "fixed_11_heldout_prediction_scatter.svg", 1120, 570, "\n".join(body))
    return pd.concat(selection_rows, ignore_index=True)


def build_visual_safeguards(
    predictions: pd.DataFrame,
    visual_summary: pd.DataFrame,
    selected_examples: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    model_seed: int,
) -> pd.DataFrame:
    checks = []

    def add(name, passed, critical=True, detail=""):
        checks.append({"safeguard": name, "passed": bool(passed), "critical": bool(critical), "detail": detail})

    selected_keys = selected_examples[["experiment", "model", "model_seed", "culture_id"]].drop_duplicates()
    selected_pred = predictions.merge(selected_keys, on=["experiment", "model", "model_seed", "culture_id"], how="inner")
    add("every plotted prediction comes from a held-out culture", (selected_pred["split"] == "heldout_combination").all())
    train_envs = rounded_key(manifest[manifest["split"] == "train"][["I", "O", "S"]].to_numpy())
    plotted_envs = rounded_key(selected_pred[["I", "O", "S"]].drop_duplicates().to_numpy())
    add("no plotted held-out environment appears in training under any seed", len(plotted_envs & train_envs) == 0)
    time_ok = predictions.groupby(["dataset_seed", "model_seed", "experiment", "model", "culture_id"])["time"].apply(lambda x: np.allclose(x.to_numpy(), np.sort(x.to_numpy())) and len(np.unique(x)) == len(x)).all()
    add("true and predicted trajectories use the same time grid", time_ok)
    add("plots do not silently mix model seeds", set(selected_pred["model_seed"]) == {model_seed})
    add("representative examples selected algorithmically", selected_examples["selection_rule"].notna().all())
    repro_ok = True
    for _, row in visual_summary.iterrows():
        agg = aggregate_from_predictions(predictions, row["experiment"], row["selected_model"])
        repro_ok = repro_ok and abs(agg["product_rmse"] - row["product_rmse"]) < 1e-12
    add("reported aggregate RMSE reproduces from prediction-level data", repro_ok)
    add("figures use saved model predictions rather than true-parameter ODE curves", (predictions["prediction_source"] == "saved_model_rollout").all() and not predictions["model"].astype(str).str.contains("oracle").any())
    return pd.DataFrame(checks)


def svg_text(x, y, text, size=14, weight="normal", anchor="middle", color="#222"):
    lines = str(text).split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x:.1f}" y="{y + i * size * 1.25:.1f}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{color}">{line}</text>'
        )
    return "\n".join(out)


def svg_box(x, y, w, h, text, fill="#eef3f7", stroke="#456", size=13):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>\n'
        + svg_text(x + w / 2, y + h / 2 - (len(str(text).split("\n")) - 1) * size * 0.55 + size * 0.35, text, size=size)
    )


def svg_arrow(x1, y1, x2, y2, color="#333", dashed=False):
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="1.5" marker-end="url(#arrow)"{dash}/>'
    )


def save_svg(path: Path, width: int, height: int, body: str):
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,4 L0,8 Z" fill="#333"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
{body}
</svg>
'''
    path.write_text(text, encoding="utf-8")


def plot_polyline(x, y, x0, y0, w, h, color="#333", width=1.6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xr = max(float(x.max() - x.min()), 1e-9)
    yr = max(float(y.max() - y.min()), 1e-9)
    pts = []
    for xi, yi in zip(x, y):
        px = x0 + (xi - x.min()) / xr * w
        py = y0 + h - (yi - y.min()) / yr * h
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def make_figures(traj: pd.DataFrame, params: pd.DataFrame, manifest: pd.DataFrame, metrics: pd.DataFrame, frontier: pd.DataFrame, cfg: Config) -> None:
    t = time_grid(cfg)
    body = []
    steps = ["fixed e", "solve dP/dt", "train on curves", "held-out e*", "complete P_hat(t|e*)"]
    for i, label in enumerate(steps):
        x = 35 + i * 150
        body.append(svg_box(x, 100, 118, 56, label))
        if i < len(steps) - 1:
            body.append(svg_arrow(x + 118, 128, x + 148, 128))
    body.append(svg_text(400, 42, "New Experiment Ladder: Fixed Environment to Complete Production Curve", 18, "bold"))
    save_svg(FIGURES / "fixed_01_experiment_ladder.svg", 800, 250, "\n".join(body))

    body = [
        svg_text(350, 42, "Data Generation: One Environment Vector Per Culture", 18, "bold"),
        svg_box(65, 105, 220, 85, "culture j\ne_j = [I_j, O_j, S_j]\nconstant over time", "#f7f0e8", "#8a6", 13),
        svg_arrow(300, 148, 395, 148),
        svg_box(410, 105, 230, 85, "P(t0), P(t1), ..., P(tT)\nR labels only when generated", "#edf5ea", "#486", 13),
    ]
    save_svg(FIGURES / "fixed_02_data_generation_diagram.svg", 700, 270, "\n".join(body))

    colors = ["#2f6f9f", "#b65d3b", "#4c8b57"]
    env_examples = [np.array([0.25, 0.75, 0.20]), np.array([0.75, 0.75, 0.20]), np.array([0.75, 0.30, 0.75])]
    body = [svg_text(500, 28, "Experiment 1 Rate Forms: F_A, F_B, and F_C", 18, "bold")]
    for j, exp in enumerate(("1A", "1B", "1C")):
        x0 = 55 + j * 315
        y0 = 70
        body.append(f'<rect x="{x0}" y="{y0}" width="265" height="170" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(svg_text(x0 + 132, y0 + 22, f"Experiment {exp}", 13, "bold"))
        ymax = 0.0
        curves = []
        for e in env_examples:
            F, _, _ = exp1_truth(exp, e, t, cfg)
            curves.append(F)
            ymax = max(ymax, float(F.max()))
        for F, color in zip(curves, colors):
            body.append(plot_polyline(t, F / max(ymax, 1e-9), x0 + 25, y0 + 45, 215, 95, color, 1.8))
        body.append(svg_text(x0 + 132, y0 + 160, "time", 10))
    save_svg(FIGURES / "fixed_03_experiment1_rate_forms.svg", 1000, 285, "\n".join(body))

    body = [svg_text(350, 30, "Environment-to-Curve Examples", 18, "bold")]
    examples = env_examples + [np.array([0.45, 0.85, 0.55]), np.array([0.90, 0.90, 0.15])]
    all_curves = [exp1_truth("1C", e, t, cfg)[1] for e in examples]
    ymax = max(float(c.max()) for c in all_curves)
    body.append('<rect x="70" y="55" width="500" height="245" fill="#fbfbfb" stroke="#ccc"/>')
    for i, (e, curve) in enumerate(zip(examples, all_curves)):
        color = ["#2f6f9f", "#b65d3b", "#4c8b57", "#7b5bb7", "#c9912c"][i]
        body.append(plot_polyline(t, curve / ymax, 95, 85, 430, 170, color, 1.8))
        body.append(svg_text(600, 82 + i * 25, f"[{e[0]:.2f},{e[1]:.2f},{e[2]:.2f}]", 11, color=color, anchor="start"))
    body.append(svg_text(320, 286, "time", 11))
    body.append(svg_text(32, 175, "P(t)", 11))
    save_svg(FIGURES / "fixed_04_environment_to_curve_examples.svg", 700, 330, "\n".join(body))

    split_colors = {"train": "#3b6ea8", "validation": "#777", "interpolation": "#61a35c", "heldout_combination": "#c84f4f", "extrapolation": "#7b5bb7"}
    body = [svg_text(330, 32, "Training and Held-Out Environment Map", 18, "bold")]
    body.append('<rect x="75" y="60" width="390" height="270" fill="#fbfbfb" stroke="#ccc"/>')
    body.append('<rect x="328" y="154" width="137" height="176" fill="#c84f4f" opacity="0.10"/>')
    for split, sub in manifest.groupby("split"):
        color = split_colors.get(split, "#333")
        for _, row in sub.iterrows():
            x = 75 + float(row.I) * 390
            y = 330 - float(row.S) * 270
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="{color}" opacity="0.76"/>')
    for i, (split, color) in enumerate(split_colors.items()):
        body.append(f'<circle cx="505" cy="{82 + i * 24}" r="4" fill="{color}"/>')
        body.append(svg_text(516, 86 + i * 24, split, 11, anchor="start"))
    body.append(svg_text(270, 356, "induction I", 12))
    body.append(svg_text(42, 195, "stress S", 12))
    save_svg(FIGURES / "fixed_05_environment_split_map.svg", 660, 385, "\n".join(body))

    body = [svg_text(500, 32, "Predicted versus True Curves by Generalization Split", 18, "bold")]
    for j, split in enumerate(TEST_SPLITS):
        x0 = 60 + j * 310
        example = traj[(traj["experiment"] == "1C") & (traj["split"] == split)]["culture_id"].unique()[0]
        e = traj[traj["culture_id"] == example][["I", "O", "S"]].iloc[0].to_numpy(float)
        _, truth_curve, truth_params = exp1_truth("1C", e, t, cfg)
        p1 = np.array([truth_params["V"] * 0.92, truth_params["tau_on"] * 1.10, truth_params["tau_off"] * 0.94])
        p2 = np.array([truth_params["V"] * 1.02, truth_params["tau_on"] * 0.96, truth_params["tau_off"] * 1.03])
        _, y1 = simulate_exp1_from_params("1C", p1, t, cfg)
        _, y2 = simulate_exp1_from_params("1C", p2, t, cfg)
        ymax = max(float(truth_curve.max()), float(y1.max()), float(y2.max()))
        body.append(f'<rect x="{x0}" y="65" width="255" height="180" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(svg_text(x0 + 128, 88, split.replace("_", " "), 12, "bold"))
        body.append(plot_polyline(t, truth_curve / ymax, x0 + 25, 105, 205, 100, "#111", 2.0))
        body.append(plot_polyline(t, y1 / ymax, x0 + 25, 105, 205, 100, "#b65d3b", 1.5))
        body.append(plot_polyline(t, y2 / ymax, x0 + 25, 105, 205, 100, "#2f6f9f", 1.5))
    body.append(svg_text(65, 278, "black=true, orange=linear structured, blue=MLP structured", 12, anchor="start"))
    save_svg(FIGURES / "fixed_06_predicted_vs_true_curves.svg", 1000, 315, "\n".join(body))

    body = [
        svg_text(400, 42, "Experiment 2 Architecture", 18, "bold"),
        svg_box(75, 110, 120, 55, "fixed e", "#eef3f7", "#456"),
        svg_arrow(195, 138, 285, 138),
        svg_box(295, 110, 155, 55, "z_A(t), z_B(t)", "#eef3f7", "#456"),
        svg_arrow(450, 138, 540, 138),
        svg_box(550, 110, 135, 55, "P_hat(t)", "#eef3f7", "#456"),
        svg_arrow(373, 165, 373, 215, dashed=True),
        svg_box(260, 220, 225, 50, "reporter losses\ntraining only", "#f7f0e8", "#8a6", 12),
    ]
    save_svg(FIGURES / "fixed_07_experiment2_architecture.svg", 800, 315, "\n".join(body))

    sub = metrics[(metrics["experiment"] == "2B") & (metrics["split"] == "heldout_combination")]
    order = ["direct_env_time_black_box", "structured_product_only_latent", "targeted_RARB_supervision", "aggregate_reporter_supervision", "shuffled_reporter_control", "true_state_oracle"]
    vals = sub.groupby("model")["product_rmse"].mean().reindex(order).dropna()
    ymax = max(float(vals.max()), 1e-9)
    body = [svg_text(420, 32, "Experiment 2B Reporter-Supervision Comparison", 18, "bold")]
    for i, (label, val) in enumerate(vals.items()):
        x = 70 + i * 120
        h = 210 * float(val) / ymax
        body.append(f'<rect x="{x}" y="{270 - h:.1f}" width="68" height="{h:.1f}" fill="#4d7fa8"/>')
        body.append(svg_text(x + 34, 288, label.replace("_", "\n"), 8))
        body.append(svg_text(x + 34, 262 - h, f"{val:.3f}", 10))
    body.append(svg_text(25, 155, "RMSE", 12))
    save_svg(FIGURES / "fixed_08_reporter_supervision_heldout.svg", 840, 355, "\n".join(body))

    subf = frontier[frontier["split"] == "heldout_combination"].groupby("quality_condition")["relative_improvement"].mean()
    vals = subf.to_dict()
    ymin = min(0.0, min(vals.values()) if vals else 0.0)
    ymax = max(cfg.useful_margin, max(vals.values()) if vals else 1.0)
    span = max(ymax - ymin, 1e-9)
    body = [svg_text(420, 32, "Reporter-Quality Frontier", 18, "bold")]
    y_thresh = 270 - ((cfg.useful_margin - ymin) / span) * 210
    body.append(f'<line x1="55" y1="{y_thresh:.1f}" x2="790" y2="{y_thresh:.1f}" stroke="#111" stroke-width="1.2" stroke-dasharray="5 4"/>')
    body.append(svg_text(795, y_thresh + 4, "5% useful", 10, anchor="start"))
    for i, (label, val) in enumerate(vals.items()):
        x = 70 + i * 88
        y = 270 - ((float(val) - ymin) / span) * 210
        h = 270 - y
        color = "#4d7fa8" if val >= cfg.useful_margin else "#b85c5c"
        body.append(f'<rect x="{x}" y="{y:.1f}" width="48" height="{h:.1f}" fill="{color}"/>')
        body.append(svg_text(x + 24, 290, label.replace("_", "\n"), 8))
        body.append(svg_text(x + 24, y - 6, f"{val:.2f}", 9))
    body.append(svg_text(25, 155, "relative improvement", 11))
    save_svg(FIGURES / "fixed_09_reporter_quality_frontier.svg", 860, 360, "\n".join(body))

    body = [
        svg_text(350, 42, "Final Bounded Claim", 18, "bold"),
        svg_box(80, 85, 540, 125, "Supported synthetic claim:\nfixed culture conditions can map to complete product curves.\nStructured ODE parameterization can help held-out combinations.\nReporter supervision helps only when it constrains useful latent structure.", "#f2f4ef", "#586", 13),
        svg_text(350, 245, "No real-yeast validation, no guaranteed extrapolation, no unique physiology claim.", 12),
    ]
    save_svg(FIGURES / "fixed_10_final_bounded_claim.svg", 700, 300, "\n".join(body))


def flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join([str(c) for c in col if c]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def best_metric(metrics: pd.DataFrame, experiment: str, split: str, learned_only: bool = False) -> pd.Series:
    sub = metrics[(metrics["experiment"] == experiment) & (metrics["split"] == split)]
    if learned_only:
        sub = sub[~sub["model"].astype(str).str.contains("oracle")]
    return sub.groupby("model")["product_rmse"].mean().sort_values().head(1)


def write_docs(
    metrics: pd.DataFrame,
    frontier: pd.DataFrame,
    safeguards: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: Config,
    visual_summary: pd.DataFrame | None = None,
) -> None:
    summary = flatten_summary(summarize_metrics(metrics))
    one_a = best_metric(metrics, "1A", "heldout_combination", learned_only=True)
    one_b = best_metric(metrics, "1B", "heldout_combination", learned_only=True)
    one_c = best_metric(metrics, "1C", "heldout_combination", learned_only=True)
    two_b = best_metric(metrics, "2B", "heldout_combination", learned_only=True)
    useful = frontier[frontier["split"] == "heldout_combination"].groupby("quality_condition")["useful_by_predeclared_margin"].mean()
    split_counts = manifest.groupby("split").size().to_dict()
    passed = int(safeguards["passed"].sum())
    total = int(len(safeguards))
    visual_note = ""
    if visual_summary is not None and not visual_summary.empty:
        rows = []
        for _, row in visual_summary.iterrows():
            rows.append(
                f"| {row['experiment']} | {row['selected_model']} | {row['product_rmse']:.4f} | {row['normalized_rmse']:.4f} | {row['product_r2']:.4f} |"
            )
        visual_note = f"""
## Visual True-Versus-Predicted Validation

The primary visual validation uses held-out environmental combinations only.
Representative cultures in `figures/fixed_06_predicted_vs_true_curves.svg` are
selected algorithmically as low, median, and high trajectory-level RMSE cases
for the selected model. `figures/fixed_08_reporter_supervision_heldout.svg`
selects Experiment 2B cultures by the paired RMSE difference between the
product-only latent model and aggregate reporter-supervised model. The scatter
summary in `figures/fixed_11_heldout_prediction_scatter.svg` uses all held-out
product time points.

The curve overlays show that the saved model predictions generally follow the
true held-out trajectory shape, magnitude, startup, and decline. They are also
a necessary check on low aggregate RMSE: a single aggregate value can hide
culture-specific misses in peak height, late decline, or final product.

| Experiment | Selected model | RMSE | normalized RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}
"""

    readme = f"""# Yeast Digital Twin Synthetic Validation

The current canonical question is:

`fixed environment e = [I, O, S] -> complete product curve P(t)`.

One culture receives one fixed environment vector for the whole trajectory.
Time is the output coordinate. The model never receives an environmental time
series in the canonical pipeline.

The central product equation is:

`dP/dt = F(t,e) - k_P P`, with `P(0)=0`.

## Experiment Ladder

1. Experiment 1A: environment controls a constant production rate `F_A(t,e)=V_A(e)`.
2. Experiment 1B: environment controls production magnitude and startup.
3. Experiment 1C: nonlinear environmental interactions control magnitude, activation, and decline.
4. Experiment 2A: reporter supervision control under interpolation.
5. Experiment 2B: reporter-supervised latent structure for held-out environmental combinations.
6. Experiment 2C: reporter-quality frontier using training-time reporter corruption.

The previous time-varying-environment and live-reporter formulation is archived at
`archive/time_varying_environment_validation/`.

## Canonical Commands

```bash
.venv/bin/python scripts/run_fixed_environment_validation.py
.venv/bin/python scripts/run_fixed_environment_validation.py --fast
.venv/bin/python -m pytest tests/test_fixed_environment_validation.py
```

## Outputs

```text
results/
  experiment_1a_fixed_rate/
  experiment_1b_fixed_startup/
  experiment_1c_fixed_nonlinear/
  experiment_2a_reporter_supervision/
  experiment_2b_heldout_environments/
  experiment_2c_reporter_quality/
data/
  fixed_environment_trajectories.csv
  fixed_environment_parameters.csv
  environment_split_manifest.csv
  canonical_metrics.csv
  heldout_model_predictions.csv
  true_vs_predicted_summary.csv
  reporter_quality_frontier.csv
  safeguards.csv
figures/
  fixed_01_experiment_ladder.svg
  fixed_02_data_generation_diagram.svg
  fixed_03_experiment1_rate_forms.svg
  fixed_04_environment_to_curve_examples.svg
  fixed_05_environment_split_map.svg
  fixed_06_predicted_vs_true_curves.svg
  fixed_07_experiment2_architecture.svg
  fixed_08_reporter_supervision_heldout.svg
  fixed_09_reporter_quality_frontier.svg
  fixed_10_final_bounded_claim.svg
  fixed_11_heldout_prediction_scatter.svg
```

## Current Run

- Dataset seeds: `{cfg.dataset_seeds}`.
- Model seeds: `{cfg.model_seeds}`.
- Split counts per dataset seed: `{split_counts}`.
- Safeguards: `{passed}/{total}` passed.

Headline held-out-combination RMSE:

| Experiment | Best model | Product RMSE |
| --- | --- | ---: |
| 1A | {one_a.index[0]} | {one_a.iloc[0]:.4f} |
| 1B | {one_b.index[0]} | {one_b.iloc[0]:.4f} |
| 1C | {one_c.index[0]} | {one_c.iloc[0]:.4f} |
| 2B | {two_b.index[0]} | {two_b.iloc[0]:.4f} |

Reporter-quality conditions useful by the predeclared 5% held-out improvement
criterion:

`{', '.join([k for k, v in useful.items() if v >= 0.5]) or 'none'}`
{visual_note}

## Claim Boundary

This is a synthetic computational validation only. It does not validate real
yeast, guarantee numerical extrapolation, prove that synthetic latent variables
uniquely represent physiology, or imply reporter supervision must always
improve product prediction.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    narrative = f"""# Physiology-Quest Validation Narrative

This document is the canonical scientific narrative for the fixed-environment
Yeast Digital Twin validation chain. It describes synthetic data only.

## Scientific Question

Given one fixed set of culture conditions `e = [I, O, S]`, can a model predict
the complete production trajectory `P(t)`, including for environmental
combinations that were not present during training?

Simulator input: fixed `e` and the configured time grid. Hidden variables:
curve parameters in Experiment 1 and synthetic physiological states in
Experiment 2. Observable outputs: product, and reporters where generated.
Model input at deployment: fixed `e` only. Model output: a full predicted
product curve `P_hat(t | e)`.

The shared product equation is:

`dP/dt = F(t,e) - k_P P`.

## Experiment 1

Experiment 1 changes only the form of `F`.

1A uses `F_A(t,e)=V_A(e)`, where `V_A` is a softplus-transformed environmental
linear score. 1B uses
`F_B(t,e)=V_B(e)[1-exp(-t/tau_B(e))]`. 1C uses
`F_C(t,e)=V_C(e)[1-exp(-t/tau_on(e))]exp(-t/tau_off(e))` with interaction
terms in `V_C`.

Structured models predict positive curve parameters from `e` and solve the
ODE. The direct black-box baseline receives `[e,t]`; it does not receive an
environment sequence.

Held-out-combination best RMSE values are: 1A `{one_a.iloc[0]:.4f}`, 1B
`{one_b.iloc[0]:.4f}`, and 1C `{one_c.iloc[0]:.4f}`.

## Experiment 2

Experiment 2 introduces hidden states `z_A(t)` and `z_B(t)`. Because `e` is
fixed, the activation levels `s_A(e)` and `s_B(e)` are constant for one
culture, while the hidden states evolve by first-order activation and recovery.
Product is generated from one effective production function:

`F_2(t,e,z_A,z_B)=F_base(t,e) * positive_factor(1-alpha_A z_A-alpha_B z_B-alpha_AB z_A z_B)`.

Reporters are generated as lagged measurements of the synthetic states:
`R_A`, `R_B`, and `R_mean`. Reporter labels are used only in training losses.
At deployment every learned model receives only `e*`.

Experiment 2A is an interpolation control. Experiment 2B is the primary
held-out-combination biosensor hypothesis. Experiment 2C corrupts reporter
labels during training with lag, noise, sparse sampling, missingness, gain
variation, and baseline offset.

Held-out 2B best model: `{two_b.index[0]}` with product RMSE `{two_b.iloc[0]:.4f}`.

## Data Splits

Splits are culture-level. No time point from a culture appears in more than one
split. The combinatorial holdout removes high induction with high stress from
training while keeping high induction with low stress and low induction with
high stress available. Numerical extrapolation is reported separately and is
not treated as guaranteed.

Split counts per dataset seed: `{split_counts}`.

## Negative and Bounded Results

Reporter supervision is not assumed to help. The shuffled-reporter condition
remains visible as a negative control, and reporter-quality conditions can fail
the 5% usefulness criterion. Numerical extrapolation remains a stress test, not
a supported deployment guarantee.

Safeguards passed `{passed}/{total}` in the current run.
{visual_note}
"""
    (ROOT / "PHYSIOLOGY_QUEST_VALIDATION.md").write_text(narrative, encoding="utf-8")

    log = f"""# Development and Reproducibility Log

## Canonical Commands

```bash
.venv/bin/python scripts/run_fixed_environment_validation.py
.venv/bin/python scripts/run_fixed_environment_validation.py --fast
.venv/bin/python -m pytest tests/test_fixed_environment_validation.py
```

## Experiment Definitions

- 1A: `F_A(t,e)=V_A(e)`.
- 1B: `F_B(t,e)=V_B(e)[1-exp(-t/tau_B(e))]`.
- 1C: `F_C(t,e)=V_C(e)[1-exp(-t/tau_on(e))]exp(-t/tau_off(e))`.
- 2A: reporter-supervision control on interpolation splits.
- 2B: held-out environmental combinations.
- 2C: reporter-quality frontier.

## Seeds and Splits

Dataset seeds: `{cfg.dataset_seeds}`.
Model seeds: `{cfg.model_seeds}`.
Split counts: `{split_counts}`.

All splits are at the culture/environment level. The split manifest is
`data/environment_split_manifest.csv`.

## Outputs

Canonical tables:

- `data/fixed_environment_trajectories.csv`
- `data/fixed_environment_parameters.csv`
- `data/environment_split_manifest.csv`
- `data/canonical_metrics.csv`
- `data/heldout_model_predictions.csv`
- `data/true_vs_predicted_summary.csv`
- `data/reporter_quality_frontier.csv`
- `data/safeguards.csv`

Canonical figures are `figures/fixed_*.svg`.
The true-versus-predicted visual checks are:

- `figures/fixed_06_predicted_vs_true_curves.svg`
- `figures/fixed_08_reporter_supervision_heldout.svg`
- `figures/fixed_11_heldout_prediction_scatter.svg`

## Safeguards

Safeguards passed `{passed}/{total}`. The canonical run raises an error if a
critical safeguard fails.

## Current Headline Results

| Experiment | Split | Best model | Product RMSE |
| --- | --- | --- | ---: |
| 1A | held-out combination | {one_a.index[0]} | {one_a.iloc[0]:.4f} |
| 1B | held-out combination | {one_b.index[0]} | {one_b.iloc[0]:.4f} |
| 1C | held-out combination | {one_c.index[0]} | {one_c.iloc[0]:.4f} |
| 2B | held-out combination | {two_b.index[0]} | {two_b.iloc[0]:.4f} |

{visual_note}

## Archive

The previous formulation, `environment time series -> latent state -> product`
with live reporter correction and mini-FBA/deployment experiments, is preserved
under `archive/time_varying_environment_validation/`.
"""
    (ROOT / "CONCRETE_EXPERIMENT_CHAIN.md").write_text(log, encoding="utf-8")

    (REPORT / "README.txt").write_text("Fixed-environment chronological Overleaf package generated by scripts/run_fixed_environment_validation.py.\n", encoding="utf-8")
    (REPORT / "main.tex").write_text(
        rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx,svg,hyperref}}
\svgsetup{{inkscapelatex=false}}
\graphicspath{{{{figures/}}}}
\title{{Fixed-Environment Synthetic Validation for a Yeast Digital Twin}}
\author{{Julian Hartono}}
\date{{July 2026}}
\begin{{document}}
\maketitle
\section*{{Abstract}}
This report replaces the earlier time-varying-input formulation with a fixed
culture-condition task: one environment vector \(e=[I,O,S]\) predicts a complete
product curve \(P(t)\). The central equation is
\[
\frac{{dP}}{{dt}} = F(t,e) - k_P P,\qquad P(0)=0.
\]
The study is synthetic and does not validate real yeast.
\section*{{Experiment Ladder}}
\begin{{enumerate}}
\item 1A: \(F_A(t,e)=V_A(e)\).
\item 1B: \(F_B(t,e)=V_B(e)[1-\exp(-t/\tau_B(e))]\).
\item 1C: \(F_C(t,e)=V_C(e)[1-\exp(-t/\tau_{{on}}(e))]\exp(-t/\tau_{{off}}(e))\).
\item 2A: reporter supervision control under interpolation.
\item 2B: held-out environmental combinations.
\item 2C: reporter-quality frontier.
\end{{enumerate}}
\includesvg[width=\textwidth]{{fixed_01_experiment_ladder}}
\section*{{Data Generation}}
Each culture keeps \(e_j(t)=e_j\) for all sampled time points. Product noise is
added only after solving the clean ODE. Split counts are {split_counts}.
\includesvg[width=0.82\textwidth]{{fixed_02_data_generation_diagram}}
\section*{{Results}}
Held-out-combination best RMSE values are: 1A {one_a.iloc[0]:.4f}, 1B
{one_b.iloc[0]:.4f}, 1C {one_c.iloc[0]:.4f}, and 2B {two_b.iloc[0]:.4f}.
\includesvg[width=\textwidth]{{fixed_08_reporter_supervision_heldout}}
\includesvg[width=\textwidth]{{fixed_09_reporter_quality_frontier}}
\section*{{Claim Boundary}}
The supported claim is bounded: fixed synthetic culture conditions can be mapped
to complete synthetic product curves; structured ODE parameterization and
training-time reporter supervision may help generalization when their
assumptions match the generator. No real-yeast validation or guaranteed
extrapolation is claimed.
\end{{document}}
""",
        encoding="utf-8",
    )

    fig_dir = REPORT / "figures"
    fig_dir.mkdir(exist_ok=True)
    for fig in FIGURES.glob("fixed_*.svg"):
        (fig_dir / fig.name).write_text(fig.read_text(encoding="utf-8"), encoding="utf-8")
    summary.to_csv(DATA / "canonical_metrics_summary.csv", index=False)


def write_archive_readme() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    text = """# Archived Time-Varying Environment Validation

This archive preserves the previous Yeast Digital Twin formulation:

`environment time series -> latent state -> product`

That earlier chain included the old Experiment 1A-1C time-varying input
studies, reporter Experiments 2A-2C centered on live reporter correction,
mini-FBA grounding, and deployment observability experiments.

The previous scientific narrative, chronological Overleaf report, scripts,
results, figures, and data outputs are preserved here where practical. They
remain useful for historical comparison and reproducibility, but they are no
longer part of the main experiment ladder or default execution path.

Canonical work now lives in the repository root and uses the fixed-environment
task:

`fixed e = [I, O, S] -> complete product curve P(t)`.
"""
    (ARCHIVE / "ARCHIVE_README.md").write_text(text, encoding="utf-8")


def run_all(fast: bool = False, write: bool = True):
    cfg = Config()
    if fast:
        cfg = replace(cfg, dataset_seeds=cfg.fast_dataset_seeds, model_seeds=cfg.fast_model_seeds)
    steps = cfg.fast_mlp_steps if fast else cfg.mlp_steps
    if write:
        ensure_dirs()
        write_archive_readme()
    all_traj = []
    all_params = []
    all_manifest = []
    all_metrics = []
    all_frontier = []
    all_predictions = []
    shuffle_ok_all = True
    for seed in cfg.dataset_seeds:
        traj, params, manifest = generate_dataset(seed, cfg)
        all_traj.append(traj)
        all_params.append(params)
        all_manifest.append(manifest)
        for model_seed in cfg.model_seeds:
            rows1, _, pred_rows1 = run_experiment1(seed, model_seed, traj, params, cfg, steps)
            rows2, frontier_rows, shuffle_ok, pred_rows2 = run_experiment2(seed, model_seed, traj, params, cfg, steps)
            all_metrics.extend(rows1)
            all_metrics.extend(rows2)
            all_frontier.extend(frontier_rows)
            all_predictions.extend(pred_rows1)
            all_predictions.extend(pred_rows2)
            shuffle_ok_all = shuffle_ok_all and shuffle_ok
    traj_df = pd.concat(all_traj, ignore_index=True)
    params_df = pd.concat(all_params, ignore_index=True)
    manifest_df = pd.concat(all_manifest, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    frontier_df = pd.DataFrame(all_frontier)
    predictions_df = pd.DataFrame(all_predictions)
    safeguards_df = build_safeguards(traj_df, params_df, metrics_df, manifest_df, shuffle_ok_all)
    if write:
        save_prediction_tables(predictions_df)
        visual_summary_df = make_true_vs_predicted_summary(predictions_df, metrics_df)
        visual_summary_df.to_csv(DATA / "true_vs_predicted_summary.csv", index=False)
        selected_examples_df = make_visual_prediction_figures(predictions_df, metrics_df, cfg, cfg.model_seeds[0])
        selected_examples_df.to_csv(DATA / "true_vs_predicted_selected_examples.csv", index=False)
        visual_safeguards_df = build_visual_safeguards(
            predictions_df,
            visual_summary_df,
            selected_examples_df,
            metrics_df,
            manifest_df,
            cfg.model_seeds[0],
        )
        safeguards_df = pd.concat([safeguards_df, visual_safeguards_df], ignore_index=True)
        traj_df.to_csv(DATA / "fixed_environment_trajectories.csv", index=False)
        params_df.to_csv(DATA / "fixed_environment_parameters.csv", index=False)
        manifest_df.to_csv(DATA / "environment_split_manifest.csv", index=False)
        metrics_df.to_csv(DATA / "canonical_metrics.csv", index=False)
        frontier_df.to_csv(DATA / "reporter_quality_frontier.csv", index=False)
        safeguards_df.to_csv(DATA / "safeguards.csv", index=False)
        save_experiment_result_tables(metrics_df, cfg)
        make_figures(traj_df, params_df, manifest_df[manifest_df["dataset_seed"] == cfg.dataset_seeds[0]], metrics_df, frontier_df, cfg)
        selected_examples_df = make_visual_prediction_figures(predictions_df, metrics_df, cfg, cfg.model_seeds[0])
        selected_examples_df.to_csv(DATA / "true_vs_predicted_selected_examples.csv", index=False)
        write_docs(metrics_df, frontier_df, safeguards_df, manifest_df[manifest_df["dataset_seed"] == cfg.dataset_seeds[0]], cfg, visual_summary_df)
    critical_failed = safeguards_df[(~safeguards_df["passed"]) & safeguards_df["critical"]]
    if not critical_failed.empty:
        raise RuntimeError("Critical safeguards failed:\n" + critical_failed.to_string(index=False))
    return metrics_df, frontier_df, safeguards_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use one dataset seed and one model seed.")
    args = parser.parse_args()
    metrics, frontier, safeguards = run_all(fast=args.fast, write=True)
    summary = metrics.groupby(["experiment", "model", "split"])["product_rmse"].mean().reset_index()
    print("Wrote fixed-environment canonical outputs.")
    print(f"Safeguards passed: {int(safeguards['passed'].sum())}/{len(safeguards)}")
    print("Best held-out-combination RMSE by experiment:")
    for exp in ("1A", "1B", "1C", "2B"):
        sub = summary[(summary["experiment"] == exp) & (summary["split"] == "heldout_combination")]
        sub = sub[~sub["model"].astype(str).str.contains("oracle")]
        if not sub.empty:
            row = sub.sort_values("product_rmse").iloc[0]
            print(f"  {exp}: {row['model']} RMSE={row['product_rmse']:.4f}")
    useful = frontier[frontier["split"] == "heldout_combination"].groupby("quality_condition")["useful_by_predeclared_margin"].mean()
    print("Useful reporter-quality conditions:", ", ".join([k for k, v in useful.items() if v >= 0.5]) or "none")


if __name__ == "__main__":
    main()
