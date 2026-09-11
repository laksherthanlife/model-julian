#!/usr/bin/env python3
"""Observable-only latent GSM-interface validation.

This experiment asks whether a GSM-compatible controller can be inferred from
experimentally observable outputs without supervising the biological encoder on
the simulator's hidden controller labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import gem_backend as gem
import run_gem_dynamic_capacity as capacity
import run_reporter_grounded_hybrid_distillation as hybrid


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "observable_latent_interface"
CHECKPOINTS = RESULTS / "checkpoints"

ENV_COLUMNS = hybrid.ENV_COLUMNS
REPORTER_COLUMNS = hybrid.REPORTER_COLUMNS
INTERFACE_COLUMNS = hybrid.INTERFACE_COLUMNS
PRIMARY_SPLITS = ["interpolation", "heldout_combination", "extrapolation"]
MODEL_NAMES = [
    "privileged_controller_upper_bound",
    "latent_observable_no_reporters",
    "latent_observable_with_reporters",
]
EPS = 1e-9

# These may be used to generate synthetic data, train the frozen GSM surrogate,
# and post-hoc diagnostics. They must not enter the observable-only encoder
# inputs, losses, early stopping, or model selection.
LOCKBOX_COLUMNS = list(
    dict.fromkeys(
        INTERFACE_COLUMNS
        + [
            "z_er",
            "E_PSY",
            "E_DES",
            "E_CYC",
            "E_PSY_after",
            "E_DES_after",
            "E_CYC_after",
            "oxidative_burden_before",
            "oxidative_generation_term",
            "oxidative_repair_term",
            "oxidative_burden_after",
            "ATP_pressure_before",
            "ATP_demand_term",
            "ATP_supply_term",
            "ATP_repair_term",
            "ATP_pressure_after",
            "bottleneck_before",
            "GGPP_loading_term",
            "beta_carotene_clearance_term",
            "phytoene_congestion_term",
            "lycopene_congestion_term",
            "bottleneck_repair_term",
            "bottleneck_after",
            "ER_before",
            "ER_generation_term",
            "ER_repair_term",
            "ER_after",
            "oxygen_uptake_used",
            "oxygen_capacity_used",
            "ATP_maintenance_flux_used",
            "biomass_flux_used",
            "beta_carotene_flux_used",
            "GGPP_flux_used",
            "PSY_synthesis_term",
            "DES_synthesis_term",
            "CYC_synthesis_term",
            "PSY_damage_rate",
            "DES_damage_rate",
            "CYC_damage_rate",
            "maximum_growth",
            "preserved_growth",
            "phytoene_congestion_flux_mismatch",
            "lycopene_congestion_flux_mismatch",
            "phytoene_capacity_pressure",
            "lycopene_capacity_pressure",
            "phytoene_congestion",
            "lycopene_congestion",
        ]
    )
)

TARGET_OBSERVABLE_COLUMNS = ["B_total", "X"] + REPORTER_COLUMNS
METABOLIC_OBSERVABLES = ["B_total", "X"]


@dataclass
class RidgeSurrogate:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    coef: np.ndarray
    output_columns: list[str]
    output_slices: dict[str, tuple[int, int]]
    loss_scale: np.ndarray


@dataclass
class EncoderCheckpoint:
    model: str
    seed: int
    W: np.ndarray
    b: np.ndarray
    env_mean: np.ndarray
    env_std: np.ndarray
    control_min: np.ndarray
    control_max: np.ndarray
    selected_epoch: int
    validation_loss: float
    training_loss_type: str
    observable_loss_columns: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def required_artifacts() -> dict[str, Path | None]:
    return hybrid.required_artifacts()


def load_dataset() -> dict[str, object]:
    return hybrid.load_distillation_dataset()


def target_matrices(dataset: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        "B_total": np.asarray(dataset["product"], dtype=float),
        "X": culture_time_matrix(dataset["traj"], "X", "time_index"),
        **{col: np.asarray(dataset["reporters"][col], dtype=float) for col in REPORTER_COLUMNS},
    }


def culture_time_matrix(df: pd.DataFrame, col: str, time_col: str) -> np.ndarray:
    ids = sorted(df["culture_id"].unique())
    rows = []
    for cid in ids:
        rows.append(df[df["culture_id"].eq(cid)].sort_values(time_col)[col].to_numpy(float))
    return np.asarray(rows, dtype=float)


def controller_tensor(dataset: dict[str, object]) -> np.ndarray:
    return np.stack([np.asarray(dataset["interface"][col], dtype=float) for col in INTERFACE_COLUMNS], axis=2)


def flatten_controls(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=float).reshape(u.shape[0], -1)


def controls_dict_from_tensor(u: np.ndarray) -> dict[str, np.ndarray]:
    return {col: np.asarray(u[:, :, j], dtype=float) for j, col in enumerate(INTERFACE_COLUMNS)}


def write_column_audit(dataset: dict[str, object]) -> pd.DataFrame:
    frames = {
        "gem_state_space_environment_grid.csv": dataset["manifest"],
        "gem_state_space_trajectories.csv": dataset["traj"],
        "gem_state_space_fluxes.csv": dataset["flux_table"],
        Path(required_artifacts()["constraints"]).name: dataset["constraints_table"],
        "gem_state_space_reporters.csv": dataset["reporter_table"],
        "gem_state_space_solver_accounting.csv": dataset["solver"],
    }
    input_obs = set(ENV_COLUMNS)
    target_obs = set(TARGET_OBSERVABLE_COLUMNS) | {
        "glucose_uptake",
        "oxygen_uptake",
        "atp_maintenance_flux",
        "beta_carotene_flux",
        "biomass_flux",
        "ggpp_flux",
        "PSY_flux",
        "DES_flux",
        "CYC_flux",
    }
    lockbox = set(LOCKBOX_COLUMNS)
    metadata_keywords = ("id", "split", "time", "status", "solver", "backend", "checksum", "mode", "label", "source", "optim")
    rows = []
    seen = set()
    for source, frame in frames.items():
        for col in frame.columns:
            key = (source, col)
            if key in seen:
                continue
            seen.add(key)
            if col in input_obs:
                klass = "INPUT_OBSERVABLE"
                allowed_encoder_input = True
                allowed_observable_loss = False
            elif col in lockbox:
                klass = "HIDDEN_LOCKBOX"
                allowed_encoder_input = False
                allowed_observable_loss = False
            elif col in target_obs or col.startswith("R_"):
                klass = "TARGET_OBSERVABLE"
                allowed_encoder_input = False
                allowed_observable_loss = True
            elif any(tok in col.lower() for tok in metadata_keywords):
                klass = "SIMULATOR_METADATA"
                allowed_encoder_input = False
                allowed_observable_loss = False
            else:
                klass = "SIMULATOR_METADATA"
                allowed_encoder_input = False
                allowed_observable_loss = False
            rows.append(
                {
                    "source_table": source,
                    "column": col,
                    "classification": klass,
                    "allowed_encoder_input": allowed_encoder_input,
                    "allowed_observable_loss": allowed_observable_loss,
                    "allowed_surrogate_input": col in input_obs or col in lockbox,
                    "allowed_posthoc_lockbox_diagnostic": col in lockbox,
                }
            )
    out = pd.DataFrame(rows).sort_values(["classification", "source_table", "column"])
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "observable_latent_interface_column_audit.csv", index=False)
    pd.DataFrame({"lockbox_column": sorted(lockbox)}).to_csv(DATA / "observable_latent_interface_lockbox.csv", index=False)
    return out


def assert_no_lockbox_usage(input_columns: list[str], loss_columns: list[str], selection_metric_columns: list[str]) -> None:
    lock = set(LOCKBOX_COLUMNS)
    violations = {
        "encoder_input": sorted(lock.intersection(input_columns)),
        "observable_loss": sorted(lock.intersection(loss_columns)),
        "model_selection": sorted(lock.intersection(selection_metric_columns)),
    }
    bad = {k: v for k, v in violations.items() if v}
    if bad:
        raise AssertionError(f"Lockbox columns entered observable-only biological training: {bad}")


def assert_split_integrity(dataset: dict[str, object]) -> None:
    manifest = dataset["manifest"]
    traj = dataset["traj"]
    if manifest.groupby("environment_id")["split"].nunique().max() != 1:
        raise AssertionError("Environment appears in multiple splits.")
    if traj.groupby("culture_id")["split"].nunique().max() != 1:
        raise AssertionError("Culture appears in multiple splits.")
    if traj.groupby(["culture_id", "time_index"])["split"].nunique().max() != 1:
        raise AssertionError("Timepoint leakage across splits detected.")


def train_stats_only(arr: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(arr, dtype=float)[train_mask]
    return vals.mean(axis=0), vals.std(axis=0) + EPS


def build_surrogate_design(env: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.asarray(env, dtype=float), flatten_controls(u)])
    return x


def build_output_matrix(targets: dict[str, np.ndarray], columns: list[str]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    mats = []
    slices = {}
    start = 0
    for col in columns:
        mat = np.asarray(targets[col], dtype=float)
        mats.append(mat)
        stop = start + mat.shape[1]
        slices[col] = (start, stop)
        start = stop
    return np.concatenate(mats, axis=1), slices


def ridge_fit(X: np.ndarray, Y: np.ndarray, train_mask: np.ndarray, lam: float = 1e-4) -> RidgeSurrogate:
    xm, xs = train_stats_only(X, train_mask)
    ym, ys = train_stats_only(Y, train_mask)
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys
    eye = np.eye(Xn.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(Xn[train_mask].T @ Xn[train_mask] + lam * eye, Xn[train_mask].T @ Yn[train_mask])
    loss_scale = np.std(Y[train_mask], axis=0) + EPS
    return RidgeSurrogate(xm, xs, ym, ys, coef, [], {}, loss_scale)


def fit_frozen_surrogate(dataset: dict[str, object], output_columns: list[str]) -> RidgeSurrogate:
    train = dataset["splits"] == "train"
    targets = target_matrices(dataset)
    X = build_surrogate_design(dataset["env"], controller_tensor(dataset))
    Y, slices = build_output_matrix(targets, output_columns)
    surrogate = ridge_fit(X, Y, train, lam=1e-4)
    surrogate.output_columns = output_columns
    surrogate.output_slices = slices
    return surrogate


def surrogate_predict(surrogate: RidgeSurrogate, env: np.ndarray, u: np.ndarray) -> np.ndarray:
    X = build_surrogate_design(env, u)
    Xn = (X - surrogate.x_mean) / surrogate.x_std
    Yn = Xn @ surrogate.coef
    return Yn * surrogate.y_std + surrogate.y_mean


def surrogate_input_gradient(surrogate: RidgeSurrogate) -> np.ndarray:
    return (surrogate.coef * surrogate.y_std[None, :]) / surrogate.x_std[:, None]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -45, 45)))


def encode_controls(ckpt: EncoderCheckpoint, env: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = (np.asarray(env, dtype=float) - ckpt.env_mean) / ckpt.env_std
    logits = X @ ckpt.W + ckpt.b
    sig = sigmoid(logits)
    flat = ckpt.control_min + sig * (ckpt.control_max - ckpt.control_min)
    n = len(env)
    u = flat.reshape(n, -1, len(INTERFACE_COLUMNS))
    if "gamma_growth_fraction" in INTERFACE_COLUMNS:
        gamma_i = INTERFACE_COLUMNS.index("gamma_growth_fraction")
        u[:, :, gamma_i] = np.clip(u[:, :, gamma_i], 0.0, 1.0)
    return u, sig, X


def output_columns_for_model(model: str) -> list[str]:
    if model == "latent_observable_no_reporters":
        return METABOLIC_OBSERVABLES
    if model == "latent_observable_with_reporters":
        return METABOLIC_OBSERVABLES + REPORTER_COLUMNS
    if model == "privileged_controller_upper_bound":
        return METABOLIC_OBSERVABLES + REPORTER_COLUMNS
    raise ValueError(model)


def output_index_for_columns(surrogate: RidgeSurrogate, columns: list[str]) -> np.ndarray:
    idx = []
    for col in columns:
        start, stop = surrogate.output_slices[col]
        idx.extend(range(start, stop))
    return np.asarray(idx, dtype=int)


def train_encoder(
    dataset: dict[str, object],
    surrogate: RidgeSurrogate,
    model: str,
    seed: int,
    steps: int = 600,
    lr: float = 0.04,
) -> EncoderCheckpoint:
    assert_no_lockbox_usage(list(ENV_COLUMNS), output_columns_for_model(model), output_columns_for_model(model))
    train = dataset["splits"] == "train"
    val = dataset["splits"] == "validation"
    env = np.asarray(dataset["env"], dtype=float)
    true_u = controller_tensor(dataset)
    u_train = true_u[train]
    env_mean, env_std = train_stats_only(env, train)
    control_min = u_train.reshape(-1, len(INTERFACE_COLUMNS)).min(axis=0)
    control_max = u_train.reshape(-1, len(INTERFACE_COLUMNS)).max(axis=0)
    span = np.maximum(control_max - control_min, EPS)
    flat_dim = true_u.shape[1] * true_u.shape[2]
    rng = np.random.default_rng(seed + 101 * len(model))
    W = rng.normal(0, 0.02, (env.shape[1], flat_dim))
    b = np.zeros(flat_dim)
    repeated_min = np.tile(control_min, true_u.shape[1])
    repeated_max = np.tile(control_max, true_u.shape[1])
    targets = target_matrices(dataset)
    Y, _ = build_output_matrix(targets, surrogate.output_columns)
    if model == "privileged_controller_upper_bound":
        loss_cols = []
        selection_cols = ["controller_label_loss"]
    else:
        loss_cols = output_columns_for_model(model)
        selection_cols = loss_cols
    use_idx = output_index_for_columns(surrogate, loss_cols) if loss_cols else np.arange(0)
    G = surrogate_input_gradient(surrogate)
    control_start = len(ENV_COLUMNS)
    Gu = G[control_start:, :]
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    best = None
    beta1, beta2 = 0.9, 0.999
    for step in range(1, steps + 1):
        ckpt = EncoderCheckpoint(model, seed, W, b, env_mean, env_std, repeated_min, repeated_max, 0, math.inf, "observable_only", loss_cols)
        u, sig, Xn = encode_controls(ckpt, env)
        if model == "privileged_controller_upper_bound":
            truth_flat = flatten_controls(true_u)
            pred_flat = flatten_controls(u)
            train_scale = np.maximum(np.std(truth_flat[train], axis=0), EPS)
            diff = (pred_flat - truth_flat) / train_scale
            dflat = np.zeros_like(pred_flat)
            dflat[train] = 2.0 * diff[train] / (max(train.sum(), 1) * diff.shape[1] * train_scale)
            train_loss = float(np.mean(diff[train] ** 2))
            val_loss = float(np.mean(diff[val] ** 2))
            training_loss_type = "privileged_controller_mse_oracle_upper_bound"
        else:
            pred = surrogate_predict(surrogate, env, u)
            scale = surrogate.loss_scale[use_idx]
            diff = (pred[:, use_idx] - Y[:, use_idx]) / scale
            dpred = np.zeros_like(pred)
            dpred[:, use_idx] = diff / scale
            dpred[train] *= 2.0 / (max(train.sum(), 1) * len(use_idx))
            dpred[~train] = 0.0
            dflat = dpred @ Gu.T
            train_loss = float(np.mean(diff[train] ** 2))
            val_loss = float(np.mean(diff[val] ** 2))
            training_loss_type = "observable_only_no_controller_label_loss"
        dsig = dflat * (repeated_max - repeated_min)
        dlogits = dsig * sig.reshape(len(env), -1) * (1.0 - sig.reshape(len(env), -1))
        gW = Xn.T @ dlogits
        gb = dlogits.sum(axis=0)
        norm = math.sqrt(float(np.sum(gW * gW) + np.sum(gb * gb)))
        if norm > 5.0:
            gW *= 5.0 / norm
            gb *= 5.0 / norm
        mW = beta1 * mW + (1.0 - beta1) * gW
        vW = beta2 * vW + (1.0 - beta2) * (gW * gW)
        mb = beta1 * mb + (1.0 - beta1) * gb
        vb = beta2 * vb + (1.0 - beta2) * (gb * gb)
        W -= lr * (mW / (1.0 - beta1**step)) / (np.sqrt(vW / (1.0 - beta2**step)) + 1e-8)
        b -= lr * (mb / (1.0 - beta1**step)) / (np.sqrt(vb / (1.0 - beta2**step)) + 1e-8)
        if step % 10 == 0 or step == steps:
            if best is None or val_loss < best[0]:
                best = (val_loss, step, W.copy(), b.copy(), train_loss)
    assert best is not None
    return EncoderCheckpoint(
        model=model,
        seed=seed,
        W=best[2],
        b=best[3],
        env_mean=env_mean,
        env_std=env_std,
        control_min=repeated_min,
        control_max=repeated_max,
        selected_epoch=int(best[1]),
        validation_loss=float(best[0]),
        training_loss_type=training_loss_type,
        observable_loss_columns=loss_cols,
    )


def metrics_for_matrix(pred: np.ndarray, truth: np.ndarray, train_truth: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred) - np.asarray(truth)
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.std(train_truth) + EPS)
    ss_tot = max(float(np.sum((truth - truth.mean()) ** 2)), EPS)
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / denom,
        "r2": float(1.0 - np.sum(err**2) / ss_tot),
    }


def evaluate_surrogate_rollout(dataset: dict[str, object], surrogate: RidgeSurrogate, ckpt: EncoderCheckpoint) -> tuple[pd.DataFrame, pd.DataFrame]:
    u, _sig, _x = encode_controls(ckpt, dataset["env"])
    pred = surrogate_predict(surrogate, dataset["env"], u)
    targets = target_matrices(dataset)
    rows = []
    train = dataset["splits"] == "train"
    for split in sorted(set(dataset["splits"])) + ["all"]:
        mask = np.ones(len(dataset["splits"]), dtype=bool) if split == "all" else dataset["splits"] == split
        for col in surrogate.output_columns:
            start, stop = surrogate.output_slices[col]
            pred_mat = pred[:, start:stop]
            truth = targets[col]
            row = {
                "model": ckpt.model,
                "model_seed": ckpt.seed,
                "evaluation_backend": "frozen_differentiable_surrogate",
                "split": split,
                "observable": col,
                "n_cultures": int(mask.sum()),
                **metrics_for_matrix(pred_mat[mask], truth[mask], truth[train]),
            }
            if col == "B_total":
                row["final_product_mae"] = float(np.mean(np.abs(pred_mat[mask, -1] - truth[mask, -1])))
                row["productivity_rmse"] = float(np.sqrt(np.mean((np.gradient(pred_mat[mask], axis=1) - np.gradient(truth[mask], axis=1)) ** 2)))
            if col == "X":
                row["final_biomass_mae"] = float(np.mean(np.abs(pred_mat[mask, -1] - truth[mask, -1])))
                row["maximum_biomass_mae"] = float(np.mean(np.abs(pred_mat[mask].max(axis=1) - truth[mask].max(axis=1))))
            rows.append(row)
    return pd.DataFrame(rows), controls_prediction_table(dataset, ckpt, u)


def controls_prediction_table(dataset: dict[str, object], ckpt: EncoderCheckpoint, u: np.ndarray) -> pd.DataFrame:
    rows = []
    meta = dataset["meta"].reset_index(drop=True)
    truth = controller_tensor(dataset)
    for i, m in meta.iterrows():
        for k, tt in enumerate(dataset["interval_time"]):
            rec = {
                "model": ckpt.model,
                "model_seed": ckpt.seed,
                "culture_index": i,
                "culture_id": m.culture_id,
                "environment_id": m.environment_id,
                "split": m.split,
                "interval_index": k,
                "time": float(tt),
            }
            for j, col in enumerate(INTERFACE_COLUMNS):
                rec[f"{col}_pred"] = float(u[i, k, j])
                rec[f"{col}_true_lockbox"] = float(truth[i, k, j])
            rows.append(rec)
    return pd.DataFrame(rows)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def latent_recovery_metrics(controls: pd.DataFrame, dataset: dict[str, object]) -> pd.DataFrame:
    train = dataset["splits"] == "train"
    true_u = controller_tensor(dataset)
    rows = []
    for model in sorted(controls["model"].unique()):
        for seed in sorted(controls[controls["model"].eq(model)]["model_seed"].unique()):
            sub = controls[(controls["model"].eq(model)) & (controls["model_seed"].eq(seed))]
            for split in sorted(set(dataset["splits"])) + ["all"]:
                g = sub if split == "all" else sub[sub["split"].eq(split)]
                if g.empty:
                    continue
                for col_i, col in enumerate(INTERFACE_COLUMNS):
                    p = g[f"{col}_pred"].to_numpy(float)
                    t = g[f"{col}_true_lockbox"].to_numpy(float)
                    train_vals = true_u[train, :, col_i].reshape(-1)
                    lo, hi = float(train_vals.min()), float(train_vals.max())
                    ss_tot = max(float(np.sum((t - t.mean()) ** 2)), EPS)
                    rows.append(
                        {
                            "model": model,
                            "model_seed": seed,
                            "split": split,
                            "control": col,
                            "posthoc_only": True,
                            "pearson": pearson(p, t),
                            "spearman": pearson(rankdata(p), rankdata(t)),
                            "r2": float(1.0 - np.sum((p - t) ** 2) / ss_tot),
                            "normalized_mae": float(np.mean(np.abs(p - t)) / max(hi - lo, EPS)),
                        }
                    )
    return pd.DataFrame(rows)


def selected_exact_indices(dataset: dict[str, object], max_per_split: int) -> list[int]:
    if max_per_split <= 0:
        return []
    indices = []
    splits = dataset["splits"]
    for split in PRIMARY_SPLITS:
        hits = np.where(splits == split)[0]
        if len(hits):
            indices.extend(hits[:max_per_split].tolist())
    return sorted(set(indices))


def exact_real_gsm_replay_limited(
    controls: dict[str, np.ndarray],
    dataset: dict[str, object],
    replay_name: str,
    indices: list[int],
    max_intervals: int,
    solver_fn: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    if not indices:
        return {"trajectories": pd.DataFrame(), "controls": pd.DataFrame(), "fluxes": pd.DataFrame(), "solver": pd.DataFrame()}
    if solver_fn is None:
        source_path, augmented = capacity.load_augmented(None)
        base_cfg = capacity.selected_parameter_config()
    else:
        source_path, augmented, base_cfg = Path("injected_test_model"), None, capacity.selected_parameter_config()
    meta = dataset["meta"].reset_index(drop=True)
    n_intervals = min(max_intervals, len(dataset["interval_time"]))
    traj_rows, control_rows, flux_rows, solver_rows = [], [], [], []
    for i in indices:
        row = meta.iloc[i]
        cfg = replace(base_cfg, temperature=float(row.temperature), pH=float(row.pH), DO=float(row.DO), n_time=n_intervals + 1, capacity_mode="dynamic_congestion_feedback")
        x = np.zeros(n_intervals + 1)
        b = np.zeros(n_intervals + 1)
        x[0], b[0] = cfg.x0, cfg.b0
        totals = {
            "n_growth_optimizations": 0,
            "n_product_optimizations": 0,
            "n_pfba_optimizations": 0,
            "n_actual_lp_solves": 0,
            "n_optimal_solves": 0,
            "n_infeasible_solves": 0,
            "n_unbounded_solves": 0,
            "n_solver_errors": 0,
            "n_skipped_intervals": 0,
            "n_surrogate_evaluations": 0,
        }
        for k in range(n_intervals):
            control = {col: float(controls[col][i, k]) for col in INTERFACE_COLUMNS}
            interval_model = None
            try:
                if solver_fn is None:
                    interval_model = augmented.copy()
                    mapped = hybrid.apply_predicted_interface_controls(interval_model, cfg, control)
                    flux = gem.solve_staged(interval_model, cfg, mapped["gamma_growth_fraction"], state="balanced")
                else:
                    mapped = control
                    flux = solver_fn(control, cfg, k)
                for key in totals:
                    if key in flux:
                        totals[key] += int(flux[key])
            except Exception as exc:
                mapped = control
                totals["n_solver_errors"] += 1
                totals["n_skipped_intervals"] += 1
                flux = {"solver_error": f"{type(exc).__name__}: {exc}", "beta_carotene_flux": np.nan, "biomass_flux": np.nan, "n_surrogate_evaluations": 0}
            finally:
                if interval_model is not None:
                    del interval_model
            beta = float(flux.get("beta_carotene_flux", np.nan))
            biomass = float(flux.get("biomass_flux", np.nan))
            if np.isfinite(beta) and np.isfinite(biomass):
                beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * max(0.0, float(control.get("z_ox", 0.0))))
                x[k + 1] = max(1e-9, x[k] + cfg.dt * biomass * x[k])
                b[k + 1] = max(0.0, b[k] + cfg.dt * (beta * x[k] - beta_deg * b[k]))
            else:
                x[k + 1], b[k + 1] = x[k], b[k]
            common = {
                "replay": replay_name,
                "culture_index": i,
                "culture_id": row.culture_id,
                "environment_id": row.environment_id,
                "split": row.split,
                "interval_index": k,
                "time": float(dataset["interval_time"][k]),
                "temperature": row.temperature,
                "pH": row.pH,
                "DO": row.DO,
                "metabolic_backend": "yeast_gem_lp",
                "gem_source": str(source_path),
                "n_surrogate_evaluations": 0,
            }
            control_rows.append({**common, **mapped})
            flux_rows.append({**common, **flux})
        for k in range(n_intervals + 1):
            traj_rows.append(
                {
                    "replay": replay_name,
                    "culture_index": i,
                    "culture_id": row.culture_id,
                    "environment_id": row.environment_id,
                    "split": row.split,
                    "time_index": k,
                    "time": float(dataset["time"][k]),
                    "X_pred": float(x[k]),
                    "B_pred_pfba": float(b[k]),
                    "direct_product_head_used": False,
                    "metabolic_backend": "yeast_gem_lp",
                    "exact_replay_scope": "limited_intervals",
                }
            )
        solver_rows.append({"replay": replay_name, "culture_index": i, "culture_id": row.culture_id, "environment_id": row.environment_id, "split": row.split, **totals})
    return {"trajectories": pd.DataFrame(traj_rows), "controls": pd.DataFrame(control_rows), "fluxes": pd.DataFrame(flux_rows), "solver": pd.DataFrame(solver_rows)}


def exact_replay_metrics(rollout: dict[str, pd.DataFrame], dataset: dict[str, object], model: str, seed: int) -> pd.DataFrame:
    traj = rollout["trajectories"]
    if traj.empty:
        return pd.DataFrame()
    product = dataset["product"]
    biomass = target_matrices(dataset)["X"]
    rows = []
    for split in sorted(traj["split"].unique()) + ["all"]:
        g = traj if split == "all" else traj[traj["split"].eq(split)]
        if g.empty:
            continue
        pred_b, true_b, pred_x, true_x = [], [], [], []
        for _, row in g.iterrows():
            i = int(row["culture_index"])
            k = int(row["time_index"])
            pred_b.append(float(row["B_pred_pfba"]))
            true_b.append(float(product[i, k]))
            pred_x.append(float(row["X_pred"]))
            true_x.append(float(biomass[i, k]))
        pred_b = np.asarray(pred_b)
        true_b = np.asarray(true_b)
        pred_x = np.asarray(pred_x)
        true_x = np.asarray(true_x)
        rows.append({"model": model, "model_seed": seed, "evaluation_backend": "real_yeast9_pfba_limited", "split": split, "observable": "B_total", "n_points": len(pred_b), **metrics_for_matrix(pred_b, true_b, product[dataset["splits"] == "train"].reshape(-1))})
        rows.append({"model": model, "model_seed": seed, "evaluation_backend": "real_yeast9_pfba_limited", "split": split, "observable": "X", "n_points": len(pred_x), **metrics_for_matrix(pred_x, true_x, biomass[dataset["splits"] == "train"].reshape(-1))})
    return pd.DataFrame(rows)


def ambiguity_analysis(dataset: dict[str, object], surrogate: RidgeSurrogate, ckpt: EncoderCheckpoint, seed: int, n_env: int = 3, n_samples: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 991)
    held = np.where(dataset["splits"] == "heldout_combination")[0]
    if len(held) == 0:
        held = np.where(dataset["splits"] != "train")[0]
    chosen = held[:n_env]
    targets = target_matrices(dataset)
    train = dataset["splits"] == "train"
    true_u = controller_tensor(dataset)
    train_flat = true_u[train].reshape(-1, len(INTERFACE_COLUMNS))
    lo = train_flat.min(axis=0)
    hi = train_flat.max(axis=0)
    rows = []
    for i in chosen:
        env = dataset["env"][[i]]
        u_center, _sig, _x = encode_controls(ckpt, env)
        candidates = []
        candidates.append(u_center[0])
        for _ in range(n_samples - 1):
            if rng.random() < 0.5:
                noise = rng.normal(0.0, 0.18, size=u_center[0].shape) * (hi - lo)
                cand = np.clip(u_center[0] + noise, lo, hi)
            else:
                cand = rng.uniform(lo, hi, size=u_center[0].shape)
            candidates.append(cand)
        U = np.asarray(candidates)
        pred = surrogate_predict(surrogate, np.repeat(env, len(U), axis=0), U)
        for condition, cols in [("product_biomass_only", METABOLIC_OBSERVABLES), ("product_biomass_reporters", METABOLIC_OBSERVABLES + REPORTER_COLUMNS)]:
            idx = output_index_for_columns(surrogate, cols)
            truth_y, _ = build_output_matrix({k: v[[i]] for k, v in targets.items()}, surrogate.output_columns)
            err = np.sqrt(np.mean(((pred[:, idx] - truth_y[:, idx]) / surrogate.loss_scale[idx]) ** 2, axis=1))
            threshold = max(float(np.percentile(err, 10)), float(err.min() * 1.15 + 1e-6))
            ok = np.where(err <= threshold)[0]
            flat = U[ok].reshape(len(ok), -1)
            if len(flat) > 1:
                scale = np.tile(np.maximum(hi - lo, EPS), U.shape[1])
                norm = flat / scale
                dists = []
                for a in range(len(norm)):
                    for b in range(a + 1, len(norm)):
                        dists.append(float(np.sqrt(np.mean((norm[a] - norm[b]) ** 2))))
                spread = float(np.mean(dists))
                max_spread = float(np.max(dists))
            else:
                spread = 0.0
                max_spread = 0.0
            rows.append(
                {
                    "model": ckpt.model,
                    "model_seed": ckpt.seed,
                    "culture_index": int(i),
                    "environment_id": dataset["meta"].iloc[i]["environment_id"],
                    "ambiguity_condition": condition,
                    "candidate_count": int(len(U)),
                    "acceptable_count": int(len(ok)),
                    "error_threshold": threshold,
                    "mean_pairwise_normalized_controller_distance": spread,
                    "max_pairwise_normalized_controller_distance": max_spread,
                    "best_observable_error": float(err.min()),
                }
            )
    return pd.DataFrame(rows)


def save_checkpoint(path: Path, ckpt: EncoderCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        W=ckpt.W,
        b=ckpt.b,
        env_mean=ckpt.env_mean,
        env_std=ckpt.env_std,
        control_min=ckpt.control_min,
        control_max=ckpt.control_max,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "model": ckpt.model,
                    "seed": ckpt.seed,
                    "selected_epoch": ckpt.selected_epoch,
                    "validation_loss": ckpt.validation_loss,
                    "training_loss_type": ckpt.training_loss_type,
                    "observable_loss_columns": ckpt.observable_loss_columns,
                    "encoder_inputs": ENV_COLUMNS,
                    "lockbox_excluded_from_observable_training": ckpt.model != "privileged_controller_upper_bound",
                },
                sort_keys=True,
            )
        ),
    )


def save_figures(metrics: pd.DataFrame, latent: pd.DataFrame, ambiguity: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    architecture = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="360" viewBox="0 0 980 360"><rect width="100%" height="100%" fill="white"/>',
            '<text x="490" y="32" text-anchor="middle" font-size="18" font-weight="bold">Observable-Only Latent GSM Interface</text>',
            svg_box(55, 100, 145, 58, "environment\nT, pH, DO"),
            svg_arrow(200, 129, 300, 129),
            svg_box(310, 92, 155, 74, "encoder E_theta\npredicts u_hat"),
            svg_arrow(465, 129, 565, 129),
            svg_box(575, 88, 180, 82, "frozen differentiable\nGSM surrogate"),
            svg_arrow(755, 129, 850, 129),
            svg_box(860, 98, 80, 62, "observable\nlosses"),
            svg_box(355, 248, 245, 52, "NO controller-label loss\nNO hidden-controller selection"),
            svg_box(645, 248, 245, 52, "final check: u_hat -> real Yeast9/pFBA"),
            "</svg>",
        ]
    )
    (FIGURES / "observable_latent_interface_architecture.svg").write_text(architecture, encoding="utf-8")
    bar_metric_svg(metrics, FIGURES / "observable_latent_interface_performance.svg")
    latent_svg(latent, FIGURES / "observable_latent_interface_recovery.svg")
    ambiguity_svg(ambiguity, FIGURES / "observable_latent_interface_ambiguity.svg")
    trajectory_svg(predictions, FIGURES / "observable_latent_interface_trajectories.svg")


def svg_box(x: int, y: int, w: int, h: int, text: str) -> str:
    lines = text.split("\n")
    body = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#f5f8fb" stroke="#536b7a"/>']
    for i, line in enumerate(lines):
        body.append(f'<text x="{x + w / 2:.1f}" y="{y + 22 + i * 16}" text-anchor="middle" font-size="12">{line}</text>')
    return "\n".join(body)


def svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="2"/><polygon points="{x2},{y2} {x2-10},{y2-6} {x2-10},{y2+6}" fill="#333"/>'


def bar_metric_svg(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics[(metrics["evaluation_backend"].str.contains("surrogate")) & (metrics["split"].isin(PRIMARY_SPLITS)) & (metrics["observable"].eq("B_total"))]
    if sub.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="260"><text x="20" y="40">No metrics</text></svg>', encoding="utf-8")
        return
    agg = sub.groupby("model", as_index=False)["normalized_rmse"].mean().sort_values("normalized_rmse")
    ymax = max(float(agg["normalized_rmse"].max()), EPS)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390"><rect width="100%" height="100%" fill="white"/>', '<text x="380" y="28" text-anchor="middle" font-size="17" font-weight="bold">Held-Out Product Error</text>']
    for i, row in enumerate(agg.itertuples()):
        x = 100 + i * 170
        h = 250 * float(row.normalized_rmse) / ymax
        body.append(f'<rect x="{x}" y="{320-h:.1f}" width="80" height="{h:.1f}" fill="#4f7cac"/>')
        body.append(f'<text x="{x+40}" y="350" text-anchor="middle" font-size="10">{str(row.model).replace("_", " ")}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def latent_svg(latent: pd.DataFrame, path: Path) -> None:
    sub = latent[(latent["split"].eq("heldout_combination")) & (latent["control"].isin(["oxygen_lower_bound", "gamma_growth_fraction", "CYC_effective_upper_bound"]))]
    if sub.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="260"><text x="20" y="40">No latent metrics</text></svg>', encoding="utf-8")
        return
    agg = sub.groupby("model", as_index=False)["normalized_mae"].mean().sort_values("normalized_mae")
    ymax = max(float(agg["normalized_mae"].max()), EPS)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="390" viewBox="0 0 760 390"><rect width="100%" height="100%" fill="white"/>', '<text x="380" y="28" text-anchor="middle" font-size="17" font-weight="bold">Post-Hoc Controller Recovery</text>']
    for i, row in enumerate(agg.itertuples()):
        x = 100 + i * 170
        h = 250 * float(row.normalized_mae) / ymax
        body.append(f'<rect x="{x}" y="{320-h:.1f}" width="80" height="{h:.1f}" fill="#8f6bb2"/>')
        body.append(f'<text x="{x+40}" y="350" text-anchor="middle" font-size="10">{str(row.model).replace("_", " ")}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def ambiguity_svg(ambiguity: pd.DataFrame, path: Path) -> None:
    if ambiguity.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="260"><text x="20" y="40">No ambiguity metrics</text></svg>', encoding="utf-8")
        return
    agg = ambiguity.groupby("ambiguity_condition", as_index=False)["mean_pairwise_normalized_controller_distance"].mean()
    ymax = max(float(agg["mean_pairwise_normalized_controller_distance"].max()), EPS)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="360" viewBox="0 0 700 360"><rect width="100%" height="100%" fill="white"/>', '<text x="350" y="28" text-anchor="middle" font-size="17" font-weight="bold">Ambiguity Spread</text>']
    for i, row in enumerate(agg.itertuples()):
        x = 150 + i * 220
        h = 230 * float(row.mean_pairwise_normalized_controller_distance) / ymax
        body.append(f'<rect x="{x}" y="{295-h:.1f}" width="95" height="{h:.1f}" fill="#6b9b6b"/>')
        body.append(f'<text x="{x+48}" y="325" text-anchor="middle" font-size="10">{str(row.ambiguity_condition).replace("_", " ")}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def trajectory_svg(predictions: pd.DataFrame, path: Path) -> None:
    if predictions.empty:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="260"><text x="20" y="40">No exact trajectory smoke replay</text></svg>', encoding="utf-8")
        return
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="260"><text x="20" y="40">Exact real-GSM trajectories are stored in CSV; aggregate metrics plotted separately.</text></svg>', encoding="utf-8")


def decision_table(metrics: pd.DataFrame, latent: pd.DataFrame, ambiguity: pd.DataFrame) -> pd.DataFrame:
    held = metrics[(metrics["evaluation_backend"].str.contains("surrogate")) & (metrics["split"].eq("heldout_combination")) & (metrics["observable"].eq("B_total"))]
    exact = metrics[(metrics["evaluation_backend"].str.contains("real_yeast9")) & (metrics["split"].eq("all")) & (metrics["observable"].eq("B_total"))]
    no_rep = held[held["model"].eq("latent_observable_no_reporters")]["normalized_rmse"].mean()
    with_rep = held[held["model"].eq("latent_observable_with_reporters")]["normalized_rmse"].mean()
    latent_no = latent[(latent["model"].eq("latent_observable_no_reporters")) & (latent["split"].eq("heldout_combination"))]["normalized_mae"].mean()
    latent_rep = latent[(latent["model"].eq("latent_observable_with_reporters")) & (latent["split"].eq("heldout_combination"))]["normalized_mae"].mean()
    exact_nrmse = float(exact["normalized_rmse"].mean()) if not exact.empty else np.nan
    exact_ok = not exact.empty and exact_nrmse < 0.75
    best_observable = float(np.nanmin([no_rep, with_rep])) if np.isfinite(no_rep) or np.isfinite(with_rep) else np.nan
    prediction_result = "PARTIAL" if np.isfinite(best_observable) and best_observable < 0.75 else "FAIL"
    rows = [
        ("Can observable-only training predict held-out product?", prediction_result, f"heldout surrogate NRMSE no_reporter={no_rep:.3f}; with_reporter={with_rep:.3f}"),
        ("Does inferred controller transfer through real GSM?", "LIMITED_PASS" if exact_ok else "NOT_RUN_OR_FAIL", f"limited exact real-GSM rows={len(exact)}; mean NRMSE={exact_nrmse:.3f}"),
        ("Can hidden controllers be approximately recovered?", "PARTIAL" if np.isfinite(latent_rep) and latent_rep < 0.45 else "FAIL", f"posthoc normalized MAE with reporters={latent_rep:.3f}"),
        ("Are some controller dimensions non-identifiable?", "YES" if not ambiguity.empty and ambiguity["mean_pairwise_normalized_controller_distance"].mean() > 0.10 else "UNCLEAR", "ambiguity search after training"),
        ("Do reporters improve prediction?", "YES" if np.isfinite(no_rep) and np.isfinite(with_rep) and with_rep < no_rep else "NO_OR_UNCLEAR", f"no_reporter={no_rep:.3f}; with_reporter={with_rep:.3f}"),
        ("Do reporters improve latent identifiability?", "YES" if np.isfinite(latent_no) and np.isfinite(latent_rep) and latent_rep < latent_no else "NO_OR_UNCLEAR", f"no_reporter={latent_no:.3f}; with_reporter={latent_rep:.3f}"),
        ("Safe to proceed to DBTL benchmark?", "NO", "narrow observability validation only; exact real-GSM transfer must be scaled before DBTL"),
    ]
    return pd.DataFrame([{"question": q, "result": r, "evidence": e} for q, r, e in rows])


def write_report(decisions: pd.DataFrame, metrics: pd.DataFrame, exact_ran: bool) -> None:
    table_lines = ["| Question | Result | Evidence |", "| --- | --- | --- |"]
    for row in decisions.itertuples():
        table_lines.append(f"| {row.question} | {row.result} | {row.evidence} |")
    lines = [
        "# Observable-Only Latent Interface Validation",
        "",
        "This experiment tests whether a GSM-compatible controller can be learned without privileged controller-label supervision.",
        "",
        "Allowed biological-encoder inputs: `temperature`, `pH`, and `DO`.",
        "",
        "Allowed observable losses: product (`B_total`), biomass (`X`), and optionally reporters (`R_ox`, `R_atp`, `R_E_PSY`, `R_E_DES`, `R_E_CYC`).",
        "",
        "Privileged controller/interface values are retained only for GSM-surrogate fitting, oracle upper-bound training, real-GSM replay, and post-hoc diagnostics.",
        "",
        "The frozen differentiable surrogate is a ridge approximation of `environment + full controller trajectory -> observable trajectories`. It is used only to propagate gradients into the encoder; primary transfer checks use Yeast9/pFBA when exact replay is enabled.",
        "",
        "## Decision Table",
        "",
        "\n".join(table_lines),
        "",
        "## Exact GSM Replay",
        "",
        "Limited exact Yeast9/pFBA replay was run." if exact_ran else "Exact Yeast9/pFBA replay was not run in this invocation; rerun with `--exact-cultures-per-split` and `--exact-max-intervals` for the real-GSM transfer check.",
        "",
        "## Key Caveat",
        "",
        "This is still synthetic fixed-strain validation. Passing this experiment would justify scaling real-GSM replay and wet-lab-like calibration, not proceeding directly to metabolic-engineering claims.",
    ]
    (RESULTS / "observable_latent_interface_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    assert_split_integrity(dataset)
    audit = write_column_audit(dataset)
    surrogate = fit_frozen_surrogate(dataset, METABOLIC_OBSERVABLES + REPORTER_COLUMNS)
    artifact_paths = {k: v for k, v in required_artifacts().items() if v is not None and v.exists()}
    manifest = {
        "run_timestamp": utc_now(),
        "git_commit": git_commit(),
        "seeds": args.seeds,
        "steps": args.steps,
        "dataset_paths": {k: str(v.relative_to(ROOT)) for k, v in artifact_paths.items()},
        "dataset_sha256": {k: sha256(v) for k, v in artifact_paths.items()},
        "encoder_inputs": ENV_COLUMNS,
        "lockbox_columns": LOCKBOX_COLUMNS,
        "exact_cultures_per_split": args.exact_cultures_per_split,
        "exact_max_intervals": args.exact_max_intervals,
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    all_metrics, all_controls, all_latent, all_ambiguity, all_exact_traj = [], [], [], [], []
    for seed in args.seeds:
        for model in MODEL_NAMES:
            tic = time.perf_counter()
            ckpt = train_encoder(dataset, surrogate, model, seed, steps=args.steps, lr=args.lr)
            save_checkpoint(CHECKPOINTS / f"{model}_seed{seed}.npz", ckpt)
            metrics, controls = evaluate_surrogate_rollout(dataset, surrogate, ckpt)
            metrics["runtime_seconds"] = time.perf_counter() - tic
            all_metrics.append(metrics)
            all_controls.append(controls)
            all_latent.append(latent_recovery_metrics(controls, dataset))
            if model in {"latent_observable_no_reporters", "latent_observable_with_reporters"}:
                all_ambiguity.append(ambiguity_analysis(dataset, surrogate, ckpt, seed, n_env=args.ambiguity_envs, n_samples=args.ambiguity_samples))
            exact_indices = selected_exact_indices(dataset, args.exact_cultures_per_split)
            if exact_indices and model != "privileged_controller_upper_bound":
                u, _sig, _x = encode_controls(ckpt, dataset["env"])
                rollout = exact_real_gsm_replay_limited(controls_dict_from_tensor(u), dataset, model, exact_indices, args.exact_max_intervals)
                exact_metrics = exact_replay_metrics(rollout, dataset, model, seed)
                if not exact_metrics.empty:
                    all_metrics.append(exact_metrics)
                    all_exact_traj.append(rollout["trajectories"].assign(model=model, model_seed=seed))
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    controls_df = pd.concat(all_controls, ignore_index=True)
    latent_df = pd.concat(all_latent, ignore_index=True)
    ambiguity_df = pd.concat(all_ambiguity, ignore_index=True) if all_ambiguity else pd.DataFrame()
    exact_traj_df = pd.concat(all_exact_traj, ignore_index=True) if all_exact_traj else pd.DataFrame()
    summary = metrics_df.groupby(["model", "evaluation_backend", "split", "observable"], as_index=False).agg(
        normalized_rmse_mean=("normalized_rmse", "mean"),
        normalized_rmse_sd=("normalized_rmse", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
    )
    decisions = decision_table(metrics_df, latent_df, ambiguity_df)
    metrics_df.to_csv(DATA / "observable_latent_interface_metrics.csv", index=False)
    summary.to_csv(DATA / "observable_latent_interface_metric_summary.csv", index=False)
    controls_df.to_csv(DATA / "observable_latent_interface_control_predictions_posthoc.csv", index=False)
    latent_df.to_csv(DATA / "observable_latent_interface_latent_recovery.csv", index=False)
    ambiguity_df.to_csv(DATA / "observable_latent_interface_ambiguity.csv", index=False)
    decisions.to_csv(DATA / "observable_latent_interface_decision_table.csv", index=False)
    exact_traj_df.to_csv(DATA / "observable_latent_interface_exact_replay_trajectories.csv", index=False)
    save_figures(metrics_df, latent_df, ambiguity_df, exact_traj_df)
    write_report(decisions, metrics_df, exact_ran=not exact_traj_df.empty)
    audit.to_csv(DATA / "observable_latent_interface_column_audit.csv", index=False)
    return {
        "metrics": metrics_df,
        "summary": summary,
        "controls": controls_df,
        "latent": latent_df,
        "ambiguity": ambiguity_df,
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--ambiguity-envs", type=int, default=3)
    parser.add_argument("--ambiguity-samples", type=int, default=180)
    parser.add_argument("--exact-cultures-per-split", type=int, default=0)
    parser.add_argument("--exact-max-intervals", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run(parse_args())
    print(outputs["decisions"].to_string(index=False))
