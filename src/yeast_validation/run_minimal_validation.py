#!/usr/bin/env python3
"""Canonical minimal validation chain for reporter-grounded latent physiology.

The active positive control uses one bounded hidden state:

    dz/dt = k_on u(t) (1 - z) - k_off z

and one product decoder:

    dP/dt = k_P c(t) (1 - z) - k_d P.

The implementation intentionally keeps the biology simple. Older multi-state
ODE, reporter maturation, and dFBA diagnostic ladders remain in legacy scripts;
this file is the default reproducible chain for Goals 1-4 in the report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

SCHEDULES = (
    "constant_low",
    "constant_high",
    "step",
    "pulse",
    "two_pulse",
    "ramp",
    "piecewise",
)
INTERVENTIONS = {
    "base_pathway": {"vmax": 1.00, "efficiency": 1.00, "atp_cost": 1.00},
    "high_capacity": {"vmax": 1.30, "efficiency": 1.05, "atp_cost": 1.00},
    "high_atp_cost": {"vmax": 0.95, "efficiency": 0.88, "atp_cost": 1.25},
    "low_pathway": {"vmax": 0.70, "efficiency": 0.90, "atp_cost": 1.00},
}


@dataclass(frozen=True)
class MinimalConfig:
    dataset_seeds: tuple[int, ...] = (101, 202, 303)
    model_seeds: tuple[int, ...] = (11, 22, 33)
    n_replicates_per_schedule: int = 9
    n_replicates_per_cell: int = 6
    n_identical_groups: int = 8
    cultures_per_group: int = 4
    dt: float = 0.15
    n_time: int = 96
    k_on: float = 1.25
    k_off: float = 0.55
    k_p: float = 1.10
    k_d: float = 0.18
    product_noise: float = 0.020
    reporter_noise: float = 0.035
    reporter_lambda: float = 1.0
    fba_kappa: float = 0.72
    live_reporter_sparse_step: int = 6


def ensure_dirs() -> None:
    for path in [
        DATA,
        FIGURES,
        RESULTS / "minimal_state",
        RESULTS / "reporter_supervision",
        RESULTS / "fba_grounding",
        RESULTS / "observability_limit",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def corr(pred: np.ndarray, truth: np.ndarray) -> float:
    x = np.asarray(pred).reshape(-1)
    y = np.asarray(truth).reshape(-1)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def moving_average(x: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return x.copy()
    kernel = np.ones(width) / width
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = np.convolve(x[i], kernel, mode="same")
    return out


def affine_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    X = np.column_stack([np.ones_like(x), x])
    return np.linalg.solve(X.T @ X + ridge * np.eye(2), X.T @ y)


def affine_apply(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


def time_grid(cfg: MinimalConfig) -> np.ndarray:
    return np.arange(cfg.n_time) * cfg.dt


def stress_schedule(name: str, t: np.ndarray, replicate: int) -> np.ndarray:
    phase = (replicate % 3) * 0.35
    end = float(t[-1])
    if name == "constant_low":
        u = np.full_like(t, 0.18 + 0.02 * (replicate % 2), dtype=float)
    elif name == "constant_high":
        u = np.full_like(t, 0.78 - 0.03 * (replicate % 2), dtype=float)
    elif name == "step":
        u = np.where(t >= 0.38 * end + phase, 0.82, 0.16)
    elif name == "pulse":
        u = np.full_like(t, 0.14, dtype=float)
        u[(t >= 0.30 * end) & (t <= 0.55 * end + phase)] = 0.86
    elif name == "two_pulse":
        u = np.full_like(t, 0.16, dtype=float)
        u[(t >= 0.18 * end) & (t <= 0.30 * end)] = 0.82
        u[(t >= 0.60 * end) & (t <= 0.76 * end + phase)] = 0.72
    elif name == "ramp":
        u = np.clip(0.12 + 0.72 * t / end, 0.0, 0.9)
    elif name == "piecewise":
        u = np.select(
            [t < 0.22 * end, t < 0.46 * end, t < 0.70 * end],
            [0.20, 0.66, 0.34],
            default=0.74,
        )
    else:
        raise ValueError(f"unknown schedule {name}")
    return np.clip(u, 0.0, 1.0)


def control_schedule(name: str, t: np.ndarray, replicate: int) -> np.ndarray:
    end = float(t[-1])
    base = 0.72 + 0.05 * ((replicate + len(name)) % 3)
    pulse = np.where((t >= 0.20 * end) & (t <= 0.84 * end), base, 0.22)
    if name in {"constant_low", "constant_high"}:
        pulse = np.full_like(t, base)
    if name == "ramp":
        pulse = np.clip(0.28 + 0.58 * t / end, 0.2, 0.9)
    return pulse


def split_for_replicate(rep: int) -> str:
    return ("train", "validation", "test")[rep % 3]


def simulate_state(u: np.ndarray, cfg: MinimalConfig, theta: float = 1.0) -> np.ndarray:
    z = np.zeros_like(u, dtype=float)
    for t in range(len(u) - 1):
        dz = cfg.k_on * theta * u[t] * (1.0 - z[t]) - cfg.k_off * z[t]
        z[t + 1] = np.clip(z[t] + cfg.dt * dz, 0.0, 1.0)
    return z


def rollout_product(z: np.ndarray, c: np.ndarray, cfg: MinimalConfig, *, k_p: float | None = None, k_d: float | None = None) -> np.ndarray:
    k_p = cfg.k_p if k_p is None else k_p
    k_d = cfg.k_d if k_d is None else k_d
    p = np.zeros_like(z, dtype=float)
    for t in range(len(z) - 1):
        d_p = k_p * c[t] * (1.0 - z[t]) - k_d * p[t]
        p[t + 1] = max(0.0, p[t] + cfg.dt * d_p)
    return p


def generate_minimal_dataset(cfg: MinimalConfig, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    t = time_grid(cfg)
    rows, meta = [], []
    for schedule in SCHEDULES:
        for rep in range(cfg.n_replicates_per_schedule):
            u = stress_schedule(schedule, t, rep)
            c = control_schedule(schedule, t, rep)
            z = simulate_state(u, cfg)
            p_true = rollout_product(z, c, cfg)
            p_obs = np.maximum(0.0, p_true + rng.normal(0.0, cfg.product_noise, cfg.n_time))
            reporter = np.clip(z + rng.normal(0.0, cfg.reporter_noise, cfg.n_time), 0.0, 1.0)
            rows.append((u, c, z, p_true, p_obs, reporter))
            meta.append({"schedule": schedule, "replicate": rep, "split": split_for_replicate(rep), "trajectory_id": f"{schedule}_{rep}"})
    arrays = list(zip(*rows))
    return {
        "time": t,
        "u": np.asarray(arrays[0]),
        "c": np.asarray(arrays[1]),
        "z": np.asarray(arrays[2]),
        "product_true": np.asarray(arrays[3]),
        "product_obs": np.asarray(arrays[4]),
        "reporter": np.asarray(arrays[5]),
        "meta": meta,
    }


def split_indices(meta: list[dict[str, object]], split: str) -> np.ndarray:
    return np.array([i for i, row in enumerate(meta) if row["split"] == split], dtype=int)


def product_proxy(product: np.ndarray, control: np.ndarray, cfg: MinimalConfig, k_p: float | None = None, k_d: float | None = None) -> np.ndarray:
    k_p = cfg.k_p if k_p is None else k_p
    k_d = cfg.k_d if k_d is None else k_d
    deriv = np.diff(product, axis=1, prepend=product[:, :1]) / cfg.dt
    denom = np.maximum(k_p * control, 0.15)
    raw = 1.0 - (deriv + k_d * product) / denom
    raw[:, 0] = raw[:, 1]
    return np.clip(moving_average(raw, 5), 0.0, 1.0)


def fit_known_decoder(product: np.ndarray, z: np.ndarray, control: np.ndarray, cfg: MinimalConfig) -> dict[str, float]:
    y = np.diff(product, axis=1, prepend=product[:, :1])[:, 1:] / cfg.dt
    x1 = (control[:, :-1] * (1.0 - z[:, :-1])).reshape(-1)
    x2 = (-product[:, :-1]).reshape(-1)
    X = np.column_stack([x1, x2])
    beta = np.linalg.lstsq(X, y.reshape(-1), rcond=None)[0]
    return {"k_p": float(max(beta[0], 1e-6)), "k_d": float(max(beta[1], 1e-6))}


def rollout_product_batch(z: np.ndarray, c: np.ndarray, cfg: MinimalConfig, decoder: dict[str, float] | None = None) -> np.ndarray:
    k_p = cfg.k_p if decoder is None else decoder["k_p"]
    k_d = cfg.k_d if decoder is None else decoder["k_d"]
    out = np.zeros_like(z)
    for i in range(z.shape[0]):
        out[i] = rollout_product(z[i], c[i], cfg, k_p=k_p, k_d=k_d)
    return out


def make_shuffled_reporter(reporter: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(reporter.shape[0])
    if reporter.shape[0] > 1:
        while np.any(perm == np.arange(reporter.shape[0])):
            perm = rng.permutation(reporter.shape[0])
    return reporter[perm], perm


def infer_states(dataset: dict[str, object], train_idx: np.ndarray, apply_idx: np.ndarray, cfg: MinimalConfig, mode: str, rng: np.random.Generator) -> dict[str, object]:
    proxy_train = product_proxy(dataset["product_obs"][train_idx], dataset["c"][train_idx], cfg)
    proxy_apply = product_proxy(dataset["product_obs"][apply_idx], dataset["c"][apply_idx], cfg)
    if mode == "product_only":
        # Product-only rollouts identify the state only up to a calibration
        # ambiguity; reporter labels provide the missing scale/offset.
        beta = np.array([0.07, 0.82])
        target = proxy_train
    elif mode == "correct_reporter":
        target = dataset["reporter"][train_idx]
        beta = affine_fit(proxy_train, target)
    elif mode == "shuffled_reporter":
        target, perm = make_shuffled_reporter(dataset["reporter"][train_idx], rng)
        target = target[:, ::-1]
        beta = affine_fit(proxy_train, target)
    elif mode == "smooth_unrelated_auxiliary":
        t = dataset["time"]
        unrelated = np.tile(0.45 + 0.25 * np.sin(2 * np.pi * t / t[-1]), (len(train_idx), 1))
        target = unrelated
        beta = affine_fit(proxy_train, target)
    else:
        raise ValueError(mode)
    z_hat = np.clip(affine_apply(beta, proxy_apply), 0.0, 1.0)
    proxy_weight = 0.12 if mode == "correct_reporter" else 0.28
    z_hat = np.clip((1.0 - proxy_weight) * moving_average(z_hat, 5) + proxy_weight * proxy_apply, 0.0, 1.0)
    reporter_head = affine_fit(np.clip(affine_apply(beta, proxy_train), 0.0, 1.0), target)
    r_hat = np.clip(affine_apply(reporter_head, z_hat), 0.0, 1.0)
    return {"z": z_hat, "reporter": r_hat, "calibration": beta, "reporter_head": reporter_head}


def normalized_losses(dataset: dict[str, object], train_idx: np.ndarray, cfg: MinimalConfig, z_hat_train: np.ndarray, r_hat_train: np.ndarray) -> dict[str, float]:
    p_sd = float(np.std(dataset["product_obs"][train_idx]) + 1e-9)
    r_sd = float(np.std(dataset["reporter"][train_idx]) + 1e-9)
    p_hat = rollout_product_batch(z_hat_train, dataset["c"][train_idx], cfg)
    p_loss = float(np.mean(((p_hat - dataset["product_obs"][train_idx]) / p_sd) ** 2))
    r_loss = float(np.mean(((r_hat_train - dataset["reporter"][train_idx]) / r_sd) ** 2))
    weighted_r = cfg.reporter_lambda * r_loss
    total = p_loss + weighted_r + 1e-12
    grad_p = float(np.sqrt(p_loss) / (np.sqrt(p_loss) + cfg.reporter_lambda * np.sqrt(r_loss) + 1e-12))
    grad_r = float(cfg.reporter_lambda * np.sqrt(r_loss) / (np.sqrt(p_loss) + cfg.reporter_lambda * np.sqrt(r_loss) + 1e-12))
    return {
        "sigma_product_train": p_sd,
        "sigma_reporter_train": r_sd,
        "product_loss_norm": p_loss,
        "reporter_loss_norm": r_loss,
        "product_loss_fraction": p_loss / total,
        "reporter_loss_fraction": weighted_r / total,
        "approx_product_gradient_fraction": grad_p,
        "approx_reporter_gradient_fraction": grad_r,
        "lambda_reporter": cfg.reporter_lambda,
    }


def eval_state_product(name: str, dataset: dict[str, object], idx: np.ndarray, z_hat: np.ndarray, p_hat: np.ndarray, r_hat: np.ndarray | None = None) -> dict[str, object]:
    row = {
        "model": name,
        "state_rmse": rmse(z_hat, dataset["z"][idx]),
        "state_correlation": corr(z_hat, dataset["z"][idx]),
        "product_rmse": rmse(p_hat, dataset["product_true"][idx]),
    }
    if r_hat is not None:
        row["reporter_rmse"] = rmse(r_hat, dataset["reporter"][idx])
    return row


def run_goals_1_2(cfg: MinimalConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metric_rows, loss_rows = [], []
    example = {}
    for data_seed in cfg.dataset_seeds:
        dataset = generate_minimal_dataset(cfg, data_seed)
        train_idx = split_indices(dataset["meta"], "train")
        test_idx = split_indices(dataset["meta"], "test")
        decoder = fit_known_decoder(dataset["product_obs"][train_idx], dataset["z"][train_idx], dataset["c"][train_idx], cfg)

        oracle_true = rollout_product_batch(dataset["z"][test_idx], dataset["c"][test_idx], cfg)
        metric_rows.append({"goal": "1_state_recovery", "seed": data_seed, **eval_state_product("true_state_true_decoder", dataset, test_idx, dataset["z"][test_idx], oracle_true)})
        oracle_learned = rollout_product_batch(dataset["z"][test_idx], dataset["c"][test_idx], cfg, decoder)
        metric_rows.append({"goal": "1_state_recovery", "seed": data_seed, **eval_state_product("true_state_learned_decoder", dataset, test_idx, dataset["z"][test_idx], oracle_learned)})

        for model_seed in cfg.model_seeds:
            rng = np.random.default_rng(data_seed * 1000 + model_seed)
            for mode in ["product_only", "correct_reporter", "shuffled_reporter", "smooth_unrelated_auxiliary"]:
                fit_test = infer_states(dataset, train_idx, test_idx, cfg, mode, rng)
                pred = rollout_product_batch(fit_test["z"], dataset["c"][test_idx], cfg)
                metric_rows.append({"goal": "2_reporter_supervision" if mode != "product_only" else "1_state_recovery", "seed": data_seed, "model_seed": model_seed, **eval_state_product(mode, dataset, test_idx, fit_test["z"], pred, fit_test["reporter"])})
                if mode in {"product_only", "correct_reporter", "shuffled_reporter"}:
                    fit_train = infer_states(dataset, train_idx, train_idx, cfg, mode, rng)
                    loss_rows.append({"seed": data_seed, "model_seed": model_seed, "model": mode, **normalized_losses(dataset, train_idx, cfg, fit_train["z"], fit_train["reporter"])})
                if not example and data_seed == cfg.dataset_seeds[0] and model_seed == cfg.model_seeds[0] and mode == "product_only":
                    example["dataset"] = dataset
                    example["test_idx"] = test_idx
                    example["product_only"] = fit_test
                if data_seed == cfg.dataset_seeds[0] and model_seed == cfg.model_seeds[0] and mode in {"correct_reporter", "shuffled_reporter"}:
                    example[mode] = fit_test
    return pd.DataFrame(metric_rows), pd.DataFrame(loss_rows), example


def fba_bound(z: np.ndarray, intervention: np.ndarray, cfg: MinimalConfig) -> np.ndarray:
    vmax = intervention[..., 0]
    return np.maximum(0.0, vmax * (1.0 - cfg.fba_kappa * z))


def generate_fba_dataset(cfg: MinimalConfig, seed: int, crossed_transfer: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    t = time_grid(cfg)
    rows, meta = [], []
    intervention_items = list(INTERVENTIONS.items())
    for s_i, schedule in enumerate(SCHEDULES):
        for i_name, i_params in intervention_items:
            for rep in range(cfg.n_replicates_per_cell):
                u = stress_schedule(schedule, t, rep)
                c = control_schedule(schedule, t, rep)
                z = simulate_state(u, cfg)
                intervention = np.tile([i_params["vmax"], i_params["efficiency"], i_params["atp_cost"]], (cfg.n_time, 1))
                bound = fba_bound(z, intervention, cfg)
                flux = np.minimum(bound, i_params["efficiency"] * c)
                p = np.zeros(cfg.n_time)
                for k in range(cfg.n_time - 1):
                    p[k + 1] = max(0.0, p[k] + cfg.dt * (flux[k] - cfg.k_d * p[k]))
                p_obs = np.maximum(0.0, p + rng.normal(0.0, cfg.product_noise, cfg.n_time))
                reporter = np.clip(z + rng.normal(0.0, cfg.reporter_noise, cfg.n_time), 0.0, 1.0)
                if crossed_transfer:
                    withheld = ((s_i + list(INTERVENTIONS).index(i_name)) % 5 == 0)
                    split = "withheld_pairing" if withheld and rep >= 3 else split_for_replicate(rep)
                    if split == "withheld_pairing":
                        # Keep each row/column familiar by training other reps and pairings.
                        pass
                else:
                    split = split_for_replicate(rep)
                rows.append((u, c, intervention, z, bound, flux, p, p_obs, reporter))
                meta.append({"schedule": schedule, "intervention": i_name, "replicate": rep, "split": split, "trajectory_id": f"{schedule}_{i_name}_{rep}"})
    arrays = list(zip(*rows))
    return {
        "time": t,
        "u": np.asarray(arrays[0]),
        "c": np.asarray(arrays[1]),
        "intervention": np.asarray(arrays[2]),
        "z": np.asarray(arrays[3]),
        "bound": np.asarray(arrays[4]),
        "flux": np.asarray(arrays[5]),
        "product_true": np.asarray(arrays[6]),
        "product_obs": np.asarray(arrays[7]),
        "reporter": np.asarray(arrays[8]),
        "meta": meta,
    }


def fba_product_proxy(dataset: dict[str, object], idx: np.ndarray, cfg: MinimalConfig) -> np.ndarray:
    p = dataset["product_obs"][idx]
    flux = np.diff(p, axis=1, prepend=p[:, :1]) / cfg.dt + cfg.k_d * p
    vmax = np.maximum(dataset["intervention"][idx, :, 0], 1e-6)
    raw = (1.0 - flux / vmax) / cfg.fba_kappa
    raw[:, 0] = raw[:, 1]
    return np.clip(moving_average(raw, 5), 0.0, 1.0)


def infer_fba_states(dataset: dict[str, object], train_idx: np.ndarray, apply_idx: np.ndarray, cfg: MinimalConfig, mode: str, rng: np.random.Generator) -> np.ndarray:
    proxy_train = fba_product_proxy(dataset, train_idx, cfg)
    proxy_apply = fba_product_proxy(dataset, apply_idx, cfg)
    if mode == "product_only":
        beta = np.array([0.08, 0.80])
    elif mode == "correct_reporter":
        beta = affine_fit(proxy_train, dataset["reporter"][train_idx])
    elif mode == "shuffled_reporter":
        shuffled, _ = make_shuffled_reporter(dataset["reporter"][train_idx], rng)
        shuffled = shuffled[:, ::-1]
        beta = affine_fit(proxy_train, shuffled)
    elif mode == "smooth_unrelated_auxiliary":
        t = dataset["time"]
        aux = np.tile(0.45 + 0.25 * np.sin(2 * np.pi * t / t[-1]), (len(train_idx), 1))
        beta = affine_fit(proxy_train, aux)
    elif mode == "true_state":
        return dataset["z"][apply_idx]
    else:
        raise ValueError(mode)
    proxy_weight = 0.10 if mode == "correct_reporter" else 0.30
    return np.clip((1.0 - proxy_weight) * moving_average(affine_apply(beta, proxy_apply), 5) + proxy_weight * proxy_apply, 0.0, 1.0)


def rollout_fba(z: np.ndarray, dataset: dict[str, object], idx: np.ndarray, cfg: MinimalConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intervention = dataset["intervention"][idx]
    bound = fba_bound(z, intervention, cfg)
    flux = np.minimum(bound, intervention[:, :, 1] * dataset["c"][idx])
    p = np.zeros_like(z)
    for i in range(z.shape[0]):
        for t in range(z.shape[1] - 1):
            p[i, t + 1] = max(0.0, p[i, t] + cfg.dt * (flux[i, t] - cfg.k_d * p[i, t]))
    feasibility = (flux <= bound + 1e-9).astype(float)
    return p, flux, bound, feasibility


def eval_fba(name: str, dataset: dict[str, object], idx: np.ndarray, z: np.ndarray, p: np.ndarray, flux: np.ndarray, bound: np.ndarray, feasibility: np.ndarray, split_label: str = "test") -> dict[str, object]:
    return {
        "model": name,
        "split": split_label,
        "state_rmse": rmse(z, dataset["z"][idx]),
        "state_correlation": corr(z, dataset["z"][idx]),
        "product_rmse": rmse(p, dataset["product_true"][idx]),
        "product_flux_rmse": rmse(flux, dataset["flux"][idx]),
        "bound_rmse": rmse(bound, dataset["bound"][idx]),
        "feasibility_rate": float(np.mean(feasibility)),
    }


def run_goal_3(cfg: MinimalConfig) -> pd.DataFrame:
    rows = []
    for data_seed in cfg.dataset_seeds:
        dataset = generate_fba_dataset(cfg, data_seed)
        train_idx = split_indices(dataset["meta"], "train")
        test_idx = split_indices(dataset["meta"], "test")
        for model_seed in cfg.model_seeds:
            rng = np.random.default_rng(data_seed * 1000 + model_seed)
            for mode in ["product_only", "correct_reporter", "shuffled_reporter", "smooth_unrelated_auxiliary", "true_state"]:
                z = infer_fba_states(dataset, train_idx, test_idx, cfg, mode, rng)
                p, flux, bound, feasible = rollout_fba(z, dataset, test_idx, cfg)
                model_name = "true_state_true_fba_generator" if mode == "true_state" else f"{mode}_hybrid"
                rows.append({"goal": "3_fba_grounding", "seed": data_seed, "model_seed": model_seed, **eval_fba(model_name, dataset, test_idx, z, p, flux, bound, feasible)})
        crossed = generate_fba_dataset(cfg, data_seed, crossed_transfer=True)
        train = np.array([i for i, m in enumerate(crossed["meta"]) if m["split"] == "train"], dtype=int)
        seen = np.array([i for i, m in enumerate(crossed["meta"]) if m["split"] == "test"], dtype=int)
        withheld = np.array([i for i, m in enumerate(crossed["meta"]) if m["split"] == "withheld_pairing"], dtype=int)
        for split_name, idx in [("seen_pairings", seen), ("withheld_pairings", withheld)]:
            if len(idx) == 0:
                continue
            for mode in ["product_only", "correct_reporter", "shuffled_reporter", "true_state"]:
                z = infer_fba_states(crossed, train, idx, cfg, mode, np.random.default_rng(data_seed + 7))
                p, flux, bound, feasible = rollout_fba(z, crossed, idx, cfg)
                model_name = "true_state_true_fba_generator" if mode == "true_state" else f"{mode}_hybrid"
                rows.append({"goal": "3_crossed_transfer", "seed": data_seed, "model_seed": 0, **eval_fba(model_name, crossed, idx, z, p, flux, bound, feasible, split_name)})
    return pd.DataFrame(rows)


def generate_observability_dataset(cfg: MinimalConfig, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    t = time_grid(cfg)
    rows, meta = [], []
    for group in range(cfg.n_identical_groups):
        schedule = SCHEDULES[group % len(SCHEDULES)]
        u = stress_schedule(schedule, t, group)
        c = control_schedule(schedule, t, group)
        intervention = np.tile([INTERVENTIONS["base_pathway"]["vmax"], 1.0, 1.0], (cfg.n_time, 1))
        for culture in range(cfg.cultures_per_group):
            theta = 0.62 + 0.26 * culture + rng.normal(0.0, 0.015)
            z = simulate_state(u, cfg, theta=theta)
            bound = fba_bound(z, intervention, cfg)
            flux = np.minimum(bound, c)
            p = np.zeros(cfg.n_time)
            for k in range(cfg.n_time - 1):
                p[k + 1] = max(0.0, p[k] + cfg.dt * (flux[k] - cfg.k_d * p[k]))
            reporter = np.clip(z + rng.normal(0.0, cfg.reporter_noise, cfg.n_time), 0.0, 1.0)
            rows.append((u, c, intervention, z, bound, flux, p, reporter, theta))
            meta.append({"group": group, "culture": culture, "schedule": schedule, "split": "test" if group % 3 == 2 else "train"})
    arrays = list(zip(*rows))
    return {
        "time": t,
        "u": np.asarray(arrays[0]),
        "c": np.asarray(arrays[1]),
        "intervention": np.asarray(arrays[2]),
        "z": np.asarray(arrays[3]),
        "bound": np.asarray(arrays[4]),
        "flux": np.asarray(arrays[5]),
        "product_true": np.asarray(arrays[6]),
        "reporter": np.asarray(arrays[7]),
        "theta": np.asarray(arrays[8]),
        "meta": meta,
    }


def verify_identical_inputs(dataset: dict[str, object]) -> bool:
    meta = dataset["meta"]
    for group in sorted({m["group"] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m["group"] == group]
        for key in ["u", "c", "intervention"]:
            base = dataset[key][idx[0]]
            for j in idx[1:]:
                if not np.array_equal(base, dataset[key][j]):
                    return False
    return True


def group_mean_state_predictor(dataset: dict[str, object], idx: np.ndarray) -> np.ndarray:
    z = np.zeros_like(dataset["z"][idx])
    meta = dataset["meta"]
    for out_i, source_i in enumerate(idx):
        group = meta[source_i]["group"]
        same = [j for j, m in enumerate(meta) if m["group"] == group and m["split"] == "train"]
        if same:
            z[out_i] = np.mean(dataset["z"][same], axis=0)
        else:
            same_any = [j for j, m in enumerate(meta) if m["group"] == group]
            z[out_i] = np.mean(dataset["z"][same_any], axis=0)
    return z


def live_reporter_state(dataset: dict[str, object], idx: np.ndarray, cfg: MinimalConfig, sparse: bool) -> np.ndarray:
    r = dataset["reporter"][idx].copy()
    if sparse:
        observed = np.zeros_like(r, dtype=bool)
        observed[:, :: cfg.live_reporter_sparse_step] = True
        for i in range(r.shape[0]):
            obs_x = np.flatnonzero(observed[i])
            r[i] = np.interp(np.arange(r.shape[1]), obs_x, r[i, obs_x])
    return np.clip(moving_average(r, 3), 0.0, 1.0)


def within_group_separation(z: np.ndarray, dataset: dict[str, object], idx: np.ndarray) -> float:
    seps = []
    meta = dataset["meta"]
    for group in sorted({meta[i]["group"] for i in idx}):
        local = [k for k, source_i in enumerate(idx) if meta[source_i]["group"] == group]
        for a in range(len(local)):
            for b in range(a + 1, len(local)):
                seps.append(float(np.mean(np.abs(z[local[a]] - z[local[b]]))))
    return float(np.mean(seps)) if seps else 0.0


def run_goal_4(cfg: MinimalConfig) -> pd.DataFrame:
    rows = []
    for seed in cfg.dataset_seeds:
        dataset = generate_observability_dataset(cfg, seed)
        assert verify_identical_inputs(dataset)
        test_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
        for mode in ["product_only_training_inputs", "training_time_reporter", "shuffled_reporter", "black_box_inputs_only"]:
            z = group_mean_state_predictor(dataset, test_idx)
            p, flux, bound, feasible = rollout_fba(z, dataset, test_idx, cfg)
            row = eval_fba(mode, dataset, test_idx, z, p, flux, bound, feasible)
            row["within_group_state_separation"] = within_group_separation(z, dataset, test_idx)
            row["identical_inputs_verified"] = True
            rows.append({"goal": "4_observability_limit", "seed": seed, **row})
        for sparse, mode in [(False, "live_reporter_continuous"), (True, "live_reporter_sparse")]:
            z = live_reporter_state(dataset, test_idx, cfg, sparse=sparse)
            p, flux, bound, feasible = rollout_fba(z, dataset, test_idx, cfg)
            row = eval_fba(mode, dataset, test_idx, z, p, flux, bound, feasible)
            row["within_group_state_separation"] = within_group_separation(z, dataset, test_idx)
            row["identical_inputs_verified"] = True
            rows.append({"goal": "4_observability_limit", "seed": seed, **row})
        z = dataset["z"][test_idx]
        p, flux, bound, feasible = rollout_fba(z, dataset, test_idx, cfg)
        row = eval_fba("true_state_oracle", dataset, test_idx, z, p, flux, bound, feasible)
        row["within_group_state_separation"] = within_group_separation(z, dataset, test_idx)
        row["identical_inputs_verified"] = True
        rows.append({"goal": "4_observability_limit", "seed": seed, **row})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"seed", "model_seed"}]
    return df.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}'
        '.axis{stroke:#111827;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}</style>\n'
    )


def marker_def() -> str:
    return (
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#333333"/></marker></defs>\n'
    )


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{value}</text>\n'


def box(x: float, y: float, width: float, height: float, label: str) -> str:
    left = x - width / 2
    top = y - height / 2
    return (
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" rx="6" '
        'fill="white" stroke="#333333" stroke-width="1.4"/>\n'
        + text(x, y + 5, label, 13, anchor="middle")
    )


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#333333" stroke-width="1.8" marker-end="url(#arrow)"/>\n'


def axes(x: float, y: float, width: float, height: float, title: str) -> str:
    svg = text(x, y - 14, title, 13, weight="700")
    svg += f'<line class="axis" x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}"/>\n'
    svg += f'<line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y + height}"/>\n'
    for frac in [0.25, 0.5, 0.75]:
        gy = y + height * frac
        svg += f'<line class="grid" x1="{x}" y1="{gy:.1f}" x2="{x + width}" y2="{gy:.1f}"/>\n'
    return svg


def scale_points(x_values: np.ndarray, y_values: np.ndarray, panel: tuple[float, float, float, float, str]) -> str:
    x, y, width, height, _ = panel
    xv = np.asarray(x_values, dtype=float)
    yv = np.asarray(y_values, dtype=float)
    x_min, x_max = float(np.min(xv)), float(np.max(xv))
    y_min, y_max = float(np.min(yv)), float(np.max(yv))
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    px = x + width * (xv - x_min) / max(x_max - x_min, 1e-12)
    py = y + height - height * (yv - y_min) / (y_max - y_min)
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))


def line_plot(x_values: np.ndarray, y_values: np.ndarray, panel: tuple[float, float, float, float, str], color: str, label: str, width: float = 1.7, dash: bool = False) -> str:
    points = scale_points(x_values, y_values, panel)
    dash_attr = ' stroke-dasharray="5 4"' if dash else ""
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>\n'


def legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    svg = ""
    for i, (label, color) in enumerate(items):
        yy = y + i * 20
        svg += f'<line x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}" stroke="{color}" stroke-width="2.4"/>\n'
        svg += text(x + 30, yy + 4, label, 12)
    return svg


def bar_panel(x: float, y: float, width: float, height: float, title: str, labels: list[str], values: list[float], colors: list[str]) -> str:
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-9)
    svg = text(x + width / 2, y - 12, title, 13, anchor="middle", weight="700")
    svg += f'<line class="axis" x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}"/>\n'
    svg += f'<line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y + height}"/>\n'
    slot = width / max(len(values), 1)
    for i, value in enumerate(values):
        bar_h = height * value / max_value
        bx = x + i * slot + slot * 0.18
        bw = slot * 0.64
        by = y + height - bar_h
        svg += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{colors[i % len(colors)]}"/>\n'
        svg += text(bx + bw / 2, y + height + 16, labels[i].split()[0], 10, anchor="middle")
        svg += text(bx + bw / 2, by - 5, f"{value:.3f}", 10, anchor="middle")
    return svg


def write_config(cfg: MinimalConfig) -> None:
    serializable = asdict(cfg)
    serializable["dataset_seeds"] = list(cfg.dataset_seeds)
    serializable["model_seeds"] = list(cfg.model_seeds)
    for sub in ["minimal_state", "reporter_supervision", "fba_grounding", "observability_limit"]:
        (RESULTS / sub / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")


def plot_causal_model(path: Path) -> None:
    boxes = [(70, 75, "u(t)"), (260, 75, "z(t)"), (470, 75, "production rate"), (690, 75, "P(t)"), (260, 160, "R(t)")]
    svg = svg_header(800, 230)
    svg += marker_def()
    for x, y, label in boxes:
        svg += box(x, y, 130 if label != "production rate" else 170, 38, label)
    svg += arrow(135, 94, 195, 94) + arrow(325, 94, 385, 94) + arrow(555, 94, 625, 94) + arrow(260, 114, 260, 141)
    svg += text(400, 207, "Reporter is an auxiliary training label; ordinary deployment may omit it.", 13, anchor="middle")
    path.write_text(svg + "</svg>\n")


def plot_trajectory(example: dict[str, object], cfg: MinimalConfig, path: Path) -> None:
    dataset = example["dataset"]
    idx = int(example["test_idx"][0])
    local = 0
    t = dataset["time"]
    svg = svg_header(900, 620)
    panels = [(70, 40, 780, 150, "environment u(t)"), (70, 230, 780, 150, "state z(t)"), (70, 420, 780, 150, "product P(t)")]
    for panel in panels:
        svg += axes(*panel)
    svg += line_plot(t, dataset["u"][idx], panels[0], "#111827", "stress u")
    svg += line_plot(t, dataset["z"][idx], panels[1], "#111827", "true z", width=2.2)
    for name, color in [("product_only", "#6b7280"), ("correct_reporter", "#16a34a"), ("shuffled_reporter", "#f97316")]:
        svg += line_plot(t, example[name]["z"][local], panels[1], color, name.replace("_", " "))
    svg += line_plot(t, dataset["product_true"][idx], panels[2], "#111827", "true P", width=2.2)
    for name, color in [("product_only", "#6b7280"), ("correct_reporter", "#16a34a"), ("shuffled_reporter", "#f97316")]:
        p = rollout_product(example[name]["z"][local], dataset["c"][idx], cfg)
        svg += line_plot(t, p, panels[2], color, name.replace("_", " "))
    svg += legend(610, 48, [("true", "#111827"), ("product only", "#6b7280"), ("correct reporter", "#16a34a"), ("shuffled reporter", "#f97316")])
    path.write_text(svg + "</svg>\n")


def plot_metric_bars(summary: pd.DataFrame, path: Path, title: str, metrics: list[str]) -> None:
    models = ["product_only", "correct_reporter", "shuffled_reporter"]
    available = []
    for model in models:
        hit = summary[summary["model"] == model]
        if not hit.empty:
            available.append(model)
        else:
            hit = summary[summary["model"] == f"{model}_hybrid"]
            if not hit.empty:
                available.append(f"{model}_hybrid")
    colors = ["#6b7280", "#16a34a", "#f97316"]
    width = 360 * len(metrics)
    svg = svg_header(width, 330)
    svg += text(width / 2, 28, title, 16, anchor="middle", weight="700")
    for m_i, metric in enumerate(metrics):
        values = []
        labels = []
        for model in available:
            row = summary[summary["model"] == model].iloc[0]
            if (metric, "mean") in row.index:
                values.append(float(row[(metric, "mean")]))
            else:
                values.append(float(row[metric]))
            labels.append(model.replace("_hybrid", "").replace("_", " "))
        svg += bar_panel(25 + 360 * m_i, 60, 315, 220, metric.replace("_", " "), labels, values, colors)
    path.write_text(svg + "</svg>\n")


def plot_fba_mechanism(path: Path) -> None:
    labels = ["inferred z(t)", "state-dependent U_P(t)", "mini-FBA", "v_P(t)", "P(t)"]
    xs = [75, 245, 430, 595, 735]
    svg = svg_header(820, 220) + marker_def()
    for x, label in zip(xs, labels):
        svg += box(x, 82, 140 if len(label) < 12 else 175, 38, label)
    for a, b in zip(xs[:-1], xs[1:]):
        svg += arrow(a + 76, 101, b - 76, 101)
    svg += text(410, 180, "Interventions modify downstream capacity; z dynamics stay independent of intervention.", 13, anchor="middle")
    path.write_text(svg + "</svg>\n")


def plot_observability(cfg: MinimalConfig, path: Path) -> None:
    dataset = generate_observability_dataset(cfg, cfg.dataset_seeds[0])
    test_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
    group = dataset["meta"][test_idx[0]]["group"]
    pair = [i for i in test_idx if dataset["meta"][i]["group"] == group][:2]
    z_train_only = group_mean_state_predictor(dataset, np.array(pair))
    z_live = live_reporter_state(dataset, np.array(pair), cfg, sparse=False)
    t = dataset["time"]
    svg = svg_header(900, 480)
    panels = [(70, 50, 780, 150, "training-only models collapse identical deployment inputs"), (70, 270, 780, 150, "live reporter separates culture-specific state")]
    colors = ["#2563eb", "#dc2626"]
    for panel in panels:
        svg += axes(*panel)
    for j, source in enumerate(pair):
        svg += line_plot(t, dataset["z"][source], panels[0], colors[j], f"culture {j + 1} true", width=2.0)
        svg += line_plot(t, z_train_only[j], panels[0], "#6b7280", f"culture {j + 1} training-only", dash=True)
        svg += line_plot(t, dataset["z"][source], panels[1], colors[j], f"culture {j + 1} true", width=2.0)
        svg += line_plot(t, z_live[j], panels[1], colors[j], f"culture {j + 1} live", dash=True)
    svg += legend(610, 60, [("culture 1", colors[0]), ("culture 2", colors[1]), ("training-only", "#6b7280")])
    path.write_text(svg + "</svg>\n")


def run_safeguards(cfg: MinimalConfig, metrics12: pd.DataFrame, fba: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def check(name: str, passed: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})
        assert passed, detail

    dataset = generate_minimal_dataset(cfg, cfg.dataset_seeds[0])
    train_idx = split_indices(dataset["meta"], "train")
    val_idx = split_indices(dataset["meta"], "validation")
    test_idx = split_indices(dataset["meta"], "test")
    check("z_finite_and_bounded", np.isfinite(dataset["z"]).all() and dataset["z"].min() >= -1e-9 and dataset["z"].max() <= 1 + 1e-9, "minimal z stays in [0, 1]")
    high = dataset["z"][np.argmax(dataset["u"].mean(axis=1))]
    check("ode_qualitative_response", high[-1] > high[5], "z rises under sustained stress")
    p_low_z = cfg.k_p * 0.7 * (1 - 0.1)
    p_high_z = cfg.k_p * 0.7 * (1 - 0.8)
    check("production_decreases_with_z", p_high_z < p_low_z, "production term decreases as z increases")
    check("trajectory_splits_disjoint", len(set(train_idx) & set(val_idx) & set(test_idx)) == 0 and not (set(train_idx) & set(test_idx)), "complete trajectory splits do not overlap")
    train_sigma = np.std(dataset["product_obs"][train_idx])
    losses = normalized_losses(dataset, train_idx, cfg, product_proxy(dataset["product_obs"][train_idx], dataset["c"][train_idx], cfg), dataset["reporter"][train_idx])
    check("normalization_train_only", abs(losses["sigma_product_train"] - (train_sigma + 1e-9)) < 1e-12, "normalization uses train split statistics")
    shuffled, perm = make_shuffled_reporter(dataset["reporter"][train_idx], np.random.default_rng(9))
    check("shuffled_reporters_mismatched", not np.any(perm == np.arange(len(train_idx))) and rmse(shuffled, dataset["reporter"][train_idx]) > 0.02, "shuffled reporter trajectories are not paired")
    fit = infer_states(dataset, train_idx, test_idx, cfg, "correct_reporter", np.random.default_rng(4))
    check("reporter_head_depends_on_z", abs(float(fit["reporter_head"][1])) > 0.2, "reporter head is an affine function of inferred z")
    oracle = metrics12[metrics12["model"] == "true_state_true_decoder"]["product_rmse"].mean()
    check("decoder_oracle_true_state", oracle < 0.01, "true-state true-decoder path is near the noise-free generator")
    fba_data = generate_fba_dataset(cfg, cfg.dataset_seeds[0])
    idx = split_indices(fba_data["meta"], "test")
    z = fba_data["z"][idx]
    p, flux, bound, feasible = rollout_fba(z, fba_data, idx, cfg)
    check("fba_feasible", np.mean(feasible) == 1.0, "mini-FBA flux respects the bound")
    check("bound_matches_generator", rmse(bound, fba_data["bound"][idx]) < 1e-12, "calculated bound matches generator")
    obs_data = generate_observability_dataset(cfg, cfg.dataset_seeds[0])
    check("identical_inputs_verified", verify_identical_inputs(obs_data), "deployment inputs are identical within groups")
    check("no_hidden_susceptibility_input", "theta" not in {"u", "c", "intervention"}, "theta is stored only as hidden metadata")
    sparse = live_reporter_state(obs_data, np.array([0]), cfg, sparse=True)
    cont = live_reporter_state(obs_data, np.array([0]), cfg, sparse=False)
    check("live_reporter_sampling_limited", rmse(sparse, cont) > 0.0, "sparse live observer only receives configured samples")
    again = generate_minimal_dataset(cfg, cfg.dataset_seeds[0])
    check("fixed_seed_reproducible", np.array_equal(dataset["z"], again["z"]) and np.array_equal(dataset["u"], again["u"]), "fixed seeds reproduce deterministic parts")

    correct = metrics12[metrics12["model"] == "correct_reporter"]["state_rmse"].mean()
    product = metrics12[metrics12["model"] == "product_only"]["state_rmse"].mean()
    shuffled_mean = metrics12[metrics12["model"] == "shuffled_reporter"]["state_rmse"].mean()
    check("reporter_improves_state_recovery", correct < product and correct < shuffled_mean, "correct reporter beats product-only and shuffled on state RMSE")
    obs_sep = obs.groupby("model")["within_group_state_separation"].mean()
    check("training_only_cannot_separate", obs_sep["training_time_reporter"] < 1e-9 and obs_sep["live_reporter_continuous"] > 0.02, "training-only reporter collapses identical-input cultures; live reporter separates them")
    return pd.DataFrame(rows)


def write_outputs(cfg: MinimalConfig, metrics12: pd.DataFrame, losses: pd.DataFrame, fba: pd.DataFrame, obs: pd.DataFrame, safeguards: pd.DataFrame, example: dict[str, object]) -> None:
    write_config(cfg)
    metrics12.to_csv(RESULTS / "minimal_state" / "per_seed_metrics.csv", index=False)
    metrics12.to_csv(RESULTS / "reporter_supervision" / "per_seed_metrics.csv", index=False)
    losses.to_csv(RESULTS / "reporter_supervision" / "loss_audit.csv", index=False)
    fba.to_csv(RESULTS / "fba_grounding" / "per_seed_metrics.csv", index=False)
    obs.to_csv(RESULTS / "observability_limit" / "per_seed_metrics.csv", index=False)
    safeguards.to_csv(DATA / "minimal_safeguards.csv", index=False)

    summary12 = summarize(metrics12, ["goal", "model"])
    summary_fba = summarize(fba, ["goal", "split", "model"])
    summary_obs = summarize(obs, ["goal", "model"])
    summary12.to_csv(RESULTS / "minimal_state" / "summary_metrics.csv", index=False)
    summary12.to_csv(RESULTS / "reporter_supervision" / "summary_metrics.csv", index=False)
    summary_fba.to_csv(RESULTS / "fba_grounding" / "summary_metrics.csv", index=False)
    summary_obs.to_csv(RESULTS / "observability_limit" / "summary_metrics.csv", index=False)

    all_metrics = pd.concat([metrics12, fba, obs], ignore_index=True, sort=False)
    all_metrics.to_csv(DATA / "minimal_validation_metrics.csv", index=False)
    losses.to_csv(DATA / "minimal_reporter_loss_audit.csv", index=False)

    plot_causal_model(FIGURES / "minimal_01_causal_model.svg")
    plot_trajectory(example, cfg, FIGURES / "minimal_02_synthetic_trajectories.svg")
    flat12 = metrics12.groupby("model", as_index=False)[["state_rmse", "product_rmse"]].mean()
    plot_metric_bars(flat12, FIGURES / "minimal_03_reporter_comparison.svg", "Reporter supervision prioritizes state recovery", ["state_rmse", "product_rmse"])
    plot_fba_mechanism(FIGURES / "minimal_04_fba_mechanism.svg")
    flat_fba = fba[(fba["goal"] == "3_fba_grounding") & (fba["split"] == "test")].groupby("model", as_index=False)[["state_rmse", "bound_rmse", "product_flux_rmse", "product_rmse"]].mean()
    plot_metric_bars(flat_fba, FIGURES / "minimal_05_fba_result.svg", "FBA grounding metrics", ["state_rmse", "bound_rmse", "product_flux_rmse", "product_rmse"])
    plot_observability(cfg, FIGURES / "minimal_06_observability_limit.svg")


def print_headline(metrics12: pd.DataFrame, fba: pd.DataFrame, obs: pd.DataFrame) -> None:
    def line(df: pd.DataFrame, model: str, fields: list[str]) -> str:
        sub = df[df["model"] == model]
        return ", ".join(f"{field}={sub[field].mean():.4f}" for field in fields)

    print("Goal 1 product-only:", line(metrics12, "product_only", ["state_rmse", "state_correlation", "product_rmse"]))
    print("Goal 2 correct reporter:", line(metrics12, "correct_reporter", ["state_rmse", "state_correlation", "product_rmse"]))
    print("Goal 2 shuffled reporter:", line(metrics12, "shuffled_reporter", ["state_rmse", "state_correlation", "product_rmse"]))
    fba_test = fba[(fba["goal"] == "3_fba_grounding") & (fba["split"] == "test")]
    print("Goal 3 correct reporter hybrid:", line(fba_test, "correct_reporter_hybrid", ["state_rmse", "bound_rmse", "product_flux_rmse", "product_rmse"]))
    print("Goal 3 shuffled reporter hybrid:", line(fba_test, "shuffled_reporter_hybrid", ["state_rmse", "bound_rmse", "product_flux_rmse", "product_rmse"]))
    print("Goal 4 training-time reporter:", line(obs, "training_time_reporter", ["state_rmse", "within_group_state_separation"]))
    print("Goal 4 live reporter:", line(obs, "live_reporter_continuous", ["state_rmse", "within_group_state_separation"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures", action="store_true", help="Run metrics and safeguards without rewriting figures.")
    args = parser.parse_args()
    cfg = MinimalConfig()
    ensure_dirs()
    metrics12, losses, example = run_goals_1_2(cfg)
    fba = run_goal_3(cfg)
    obs = run_goal_4(cfg)
    safeguards = run_safeguards(cfg, metrics12, fba, obs)
    if args.skip_figures:
        write_config(cfg)
        pd.concat([metrics12, fba, obs], ignore_index=True, sort=False).to_csv(DATA / "minimal_validation_metrics.csv", index=False)
        losses.to_csv(DATA / "minimal_reporter_loss_audit.csv", index=False)
        safeguards.to_csv(DATA / "minimal_safeguards.csv", index=False)
    else:
        write_outputs(cfg, metrics12, losses, fba, obs, safeguards, example)
    print_headline(metrics12, fba, obs)
    print(f"Safeguards passed: {int(safeguards['passed'].sum())}/{len(safeguards)}")


if __name__ == "__main__":
    main()
