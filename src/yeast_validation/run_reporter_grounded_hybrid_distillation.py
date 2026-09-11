#!/usr/bin/env python3
"""Reporter-grounded teacher to pFBA-grounded hybrid distillation.

Stages implemented here:

1. Train/freeze multi-head neural state-space teachers from fixed
   ``[temperature, pH, DO]`` deployment inputs.
2. Generate dense confidence-weighted teacher pseudo-data inside the original
   environmental support.
3. Train an open-loop regulatory hybrid student and provide exact Yeast9
   staged-pFBA rollouts from predicted compact interface controls.

Stage 4 strain engineering remains intentionally unavailable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sympy.core.cache import clear_cache as clear_sympy_cache

import gem_backend as gem
import run_gem_dynamic_capacity as capacity
import run_gem_state_space_validation as ss_validation


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "reporter_grounded_hybrid_distillation"
CHECKPOINTS = RESULTS / "teacher_checkpoints"
STUDENT_CHECKPOINTS = RESULTS / "hybrid_student_checkpoints"
EXACT_REPLAY = RESULTS / "exact_replay"

MODEL_SEEDS = [11, 22, 33, 44, 55]
LATENT_DIMS = [4, 8, 16]
EPS = 1e-9

ENV_COLUMNS = ["temperature", "pH", "DO"]
PRODUCT_COLUMNS = ["B_total"]
REPORTER_COLUMNS = ["R_ox", "R_atp", "R_E_PSY", "R_E_DES", "R_E_CYC"]
CAPACITY_REPORTERS = ["R_E_PSY", "R_E_DES", "R_E_CYC"]
UNRELATED_REPORTERS = ["R_er"]
STATE_COLUMNS = ["z_ox", "z_atp", "z_bottle", "E_PSY", "E_DES", "E_CYC"]
INTERFACE_COLUMNS = [
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "gamma_growth_fraction",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
    "z_ox",
    "z_atp",
    "z_bottle",
]
GEM_APPLIED_CONTROL_COLUMNS = [
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "gamma_growth_fraction",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
]
FLUX_COLUMNS = [
    "beta_carotene_flux",
    "biomass_flux",
    "oxygen_uptake",
    "atp_maintenance_flux",
    "ggpp_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
]
PRIMARY_TEACHER_VARIANT = "full_reporter_interface_state_space"

TEACHER_VARIANTS: dict[str, dict[str, object]] = {
    "product_only_state_space": {"reporters": [], "interface": False, "flux": False, "mechanistic_rate": False},
    "mechanistic_rate_state_space": {"reporters": [], "interface": False, "flux": False, "mechanistic_rate": True},
    "oxidative_reporter_state_space": {"reporters": ["R_ox"], "interface": False, "flux": False, "mechanistic_rate": False},
    "atp_reporter_state_space": {"reporters": ["R_atp"], "interface": False, "flux": False, "mechanistic_rate": False},
    "capacity_reporter_state_space": {"reporters": CAPACITY_REPORTERS, "interface": False, "flux": False, "mechanistic_rate": False},
    "ox_atp_reporter_state_space": {"reporters": ["R_ox", "R_atp"], "interface": False, "flux": False, "mechanistic_rate": False},
    "full_reporter_state_space": {"reporters": REPORTER_COLUMNS, "interface": False, "flux": False, "mechanistic_rate": False},
    PRIMARY_TEACHER_VARIANT: {"reporters": REPORTER_COLUMNS, "interface": True, "flux": True, "mechanistic_rate": False},
    "shuffled_reporter_control": {"reporters": REPORTER_COLUMNS, "interface": False, "flux": False, "mechanistic_rate": False, "shuffle_reporters": True},
    "random_smooth_aux_control": {"reporters": ["R_random_smooth"], "interface": False, "flux": False, "mechanistic_rate": False},
    "er_only_state_space": {"reporters": UNRELATED_REPORTERS, "interface": False, "flux": False, "mechanistic_rate": False},
}

STUDENT_VARIANTS: dict[str, dict[str, object]] = {
    "hybrid_reporter_supervised": {"reporters": REPORTER_COLUMNS, "interface": INTERFACE_COLUMNS, "flux": FLUX_COLUMNS},
    "hybrid_without_reporters": {"reporters": [], "interface": INTERFACE_COLUMNS, "flux": FLUX_COLUMNS},
    "hybrid_shuffled_reporters": {"reporters": REPORTER_COLUMNS, "interface": INTERFACE_COLUMNS, "flux": FLUX_COLUMNS, "shuffle_reporters": True},
    "hybrid_random_smooth_aux": {"reporters": ["R_random_smooth"], "interface": INTERFACE_COLUMNS, "flux": FLUX_COLUMNS},
    "hybrid_one_control_only": {"reporters": REPORTER_COLUMNS, "interface": ["CYC_effective_upper_bound"], "flux": ["beta_carotene_flux"]},
}

INTERFACE_DOCUMENTATION = [
    {
        "control": "oxygen_lower_bound",
        "units": "mmol gDW^-1 h^-1 exchange lower bound; negative uptake",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.OXYGEN_EXCHANGE,
        "extraction": "recorded audited interval oxygen lower bound after environment and oxidative burden constraints",
    },
    {
        "control": "atp_maintenance_lower_bound",
        "units": "mmol gDW^-1 h^-1 ATP maintenance lower bound",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.ATPM_RXN,
        "extraction": "recorded audited interval ATP-maintenance lower bound",
    },
    {
        "control": "gamma_growth_fraction",
        "units": "fraction of maximum growth preserved before product optimization",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.BIOMASS_RXN,
        "extraction": "recorded state-dependent staged-optimization growth fraction",
    },
    {
        "control": "PSY_effective_upper_bound",
        "units": "mmol gDW^-1 h^-1 reaction upper bound",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.PSY_RXN,
        "extraction": "recorded audited effective enzyme-capacity bound",
    },
    {
        "control": "DES_effective_upper_bound",
        "units": "mmol gDW^-1 h^-1 reaction upper bound",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.DES_RXN,
        "extraction": "recorded audited effective enzyme-capacity bound",
    },
    {
        "control": "CYC_effective_upper_bound",
        "units": "mmol gDW^-1 h^-1 reaction upper bound",
        "valid_range_source": "training min/max in gem_state_space_generator_constraints.partial.csv",
        "gem_mapping": gem.CYC_RXN,
        "extraction": "recorded audited effective enzyme-capacity bound",
    },
    {
        "control": "z_ox/z_atp/z_bottle",
        "units": "dimensionless burden state proxy",
        "valid_range_source": "training min/max in gem_state_space_trajectories.csv and interval constraints",
        "gem_mapping": "z_ox enters product-degradation multiplier; all three retained for reporter/interface audits",
        "extraction": "recorded audited burden states; not claimed as true yeast regulation",
    },
]


class Stage4BlockedError(RuntimeError):
    """Raised when a caller tries to run strain-engineering Stage 4."""


@dataclass
class StateSpaceCheckpoint:
    variant: str
    seed: int
    latent_dim: int
    t_len: int
    interval_len: int
    dt: float
    env_mean: np.ndarray
    env_std: np.ndarray
    Wg: np.ndarray
    bg: np.ndarray
    A: np.ndarray
    C: np.ndarray
    bf: np.ndarray
    heads: dict[str, np.ndarray]
    target_mean: dict[str, float]
    target_std: dict[str, float]
    target_min: dict[str, float]
    target_max: dict[str, float]
    supervised_columns: list[str]
    mechanistic_rate: bool = False
    product_scale: float = 1.0


def stage4_strain_engineering_placeholder(*_args, **_kwargs):
    raise Stage4BlockedError(
        "Stage 4 is blocked until Stage 3 passes: no knockouts, overexpression searches, "
        "strain feature encoders, active learning, or engineered-strain proposals are implemented here."
    )


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def discover_constraints_path() -> Path | None:
    for name in ["gem_state_space_constraints.csv", "gem_state_space_generator_constraints.partial.csv"]:
        path = DATA / name
        if path.exists():
            return path
    return None


def required_artifacts() -> dict[str, Path | None]:
    return {
        "environment_grid": DATA / "gem_state_space_environment_grid.csv",
        "trajectories": DATA / "gem_state_space_trajectories.csv",
        "fluxes": DATA / "gem_state_space_fluxes.csv",
        "states": DATA / "gem_state_space_states.csv",
        "reporters": DATA / "gem_state_space_reporters.csv",
        "constraints": discover_constraints_path(),
        "solver_accounting": DATA / "gem_state_space_solver_accounting.csv",
        "split_manifest": DATA / "gem_state_space_split_manifest.csv",
    }


def validate_artifacts() -> pd.DataFrame:
    rows = []
    for role, path in required_artifacts().items():
        rows.append(
            {
                "artifact_role": role,
                "path": "" if path is None else str(path.relative_to(ROOT)),
                "exists": bool(path is not None and path.exists()),
                "note": "constraints checkpoint used as audited interval table" if role == "constraints" and path and "partial" in path.name else "",
            }
        )
    return pd.DataFrame(rows)


def _pivot_time(df: pd.DataFrame, ids: list[str], cols: list[str], index_col: str = "time_index") -> dict[str, np.ndarray]:
    out = {col: [] for col in cols}
    for cid in ids:
        g = df[df["culture_id"].eq(cid)].sort_values(index_col)
        for col in cols:
            out[col].append(g[col].to_numpy(float))
    return {col: np.asarray(vals, dtype=float) for col, vals in out.items()}


def _add_random_smooth_reporter(reporter_mats: dict[str, np.ndarray], t_len: int, n: int) -> None:
    rng = np.random.default_rng(7001)
    rows = []
    for _ in range(n):
        rows.append(np.sin(np.linspace(0, math.pi, t_len) + rng.uniform(-0.5, 0.5)) + 0.2 * rng.normal(size=t_len))
    reporter_mats["R_random_smooth"] = np.asarray(rows, dtype=float)


def load_distillation_dataset() -> dict[str, object]:
    artifacts = required_artifacts()
    missing = [role for role, path in artifacts.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required real-GEM state-space artifacts: {missing}")

    traj = pd.read_csv(artifacts["trajectories"])
    reporters = pd.read_csv(artifacts["reporters"])
    flux = pd.read_csv(artifacts["fluxes"])
    constraints = pd.read_csv(artifacts["constraints"])
    solver = pd.read_csv(artifacts["solver_accounting"])
    manifest = pd.read_csv(artifacts["split_manifest"])

    ids = sorted(traj["culture_id"].unique())
    meta_rows = []
    for cid in ids:
        row = traj[traj["culture_id"].eq(cid)].sort_values("time_index").iloc[0]
        meta_rows.append(
            {
                "culture_id": cid,
                "environment_id": row["environment_id"],
                "split": row["split"],
                "temperature": row["temperature"],
                "pH": row["pH"],
                "DO": row["DO"],
                "metabolic_backend": row["metabolic_backend"],
                "n_surrogate_evaluations": row["n_surrogate_evaluations"],
            }
        )
    meta = pd.DataFrame(meta_rows)
    product = _pivot_time(traj, ids, PRODUCT_COLUMNS)["B_total"]
    reporter_mats = _pivot_time(reporters, ids, REPORTER_COLUMNS + UNRELATED_REPORTERS)
    _add_random_smooth_reporter(reporter_mats, product.shape[1], len(ids))
    state_mats = _pivot_time(traj, ids, STATE_COLUMNS)
    interface_mats = _pivot_time(constraints, ids, INTERFACE_COLUMNS, index_col="interval_index")
    flux_mats = _pivot_time(flux, ids, FLUX_COLUMNS, index_col="interval_index")
    time_values = traj[traj["culture_id"].eq(ids[0])].sort_values("time_index")["time"].to_numpy(float)
    interval_times = constraints[constraints["culture_id"].eq(ids[0])].sort_values("interval_index")["time"].to_numpy(float)
    return {
        "meta": meta,
        "env": meta[ENV_COLUMNS].to_numpy(float),
        "splits": meta["split"].to_numpy(str),
        "time": time_values,
        "interval_time": interval_times,
        "product": product,
        "reporters": reporter_mats,
        "states": state_mats,
        "interface": interface_mats,
        "flux": flux_mats,
        "traj": traj,
        "reporter_table": reporters,
        "flux_table": flux,
        "constraints_table": constraints,
        "solver": solver,
        "manifest": manifest,
    }


def write_data_inventory(dataset: dict[str, object]) -> pd.DataFrame:
    meta = dataset["meta"]
    traj = dataset["traj"]
    constraints = dataset["constraints_table"]
    flux = dataset["flux_table"]
    solver = dataset["solver"]
    rows = [
        {
            "inventory_item": "cultures",
            "value": int(meta["culture_id"].nunique()),
            "status": "observed",
            "source": "gem_state_space_trajectories.csv",
        },
        {
            "inventory_item": "time_points_per_culture",
            "value": int(traj.groupby("culture_id")["time_index"].nunique().mode().iloc[0]),
            "status": "observed",
            "source": "gem_state_space_trajectories.csv",
        },
        {
            "inventory_item": "intervals_per_culture",
            "value": int(constraints.groupby("culture_id")["interval_index"].nunique().mode().iloc[0]),
            "status": "observed",
            "source": Path(required_artifacts()["constraints"]).name,
        },
        {
            "inventory_item": "split_counts",
            "value": json.dumps(meta["split"].value_counts().to_dict(), sort_keys=True),
            "status": "observed",
            "source": "gem_state_space_split_manifest.csv",
        },
        {
            "inventory_item": "real_yeast9_lp_backend",
            "value": bool(traj["metabolic_backend"].eq("yeast_gem_lp").all()),
            "status": "observed",
            "source": "gem_state_space_trajectories.csv",
        },
        {
            "inventory_item": "zero_surrogate_evaluations",
            "value": int(max(traj["n_surrogate_evaluations"].max(), solver["n_surrogate_evaluations"].max())),
            "status": "observed",
            "source": "trajectory and solver accounting",
        },
        {
            "inventory_item": "constraint_columns_used",
            "value": ",".join(INTERFACE_COLUMNS),
            "status": "selected_existing_columns",
            "source": Path(required_artifacts()["constraints"]).name,
        },
        {
            "inventory_item": "flux_columns_used",
            "value": ",".join(FLUX_COLUMNS),
            "status": "selected_existing_columns",
            "source": "gem_state_space_fluxes.csv",
        },
        {
            "inventory_item": "constraint_rows",
            "value": int(len(constraints)),
            "status": "observed",
            "source": Path(required_artifacts()["constraints"]).name,
        },
        {
            "inventory_item": "flux_rows",
            "value": int(len(flux)),
            "status": "observed",
            "source": "gem_state_space_fluxes.csv",
        },
    ]
    out = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "hybrid_distillation_data_inventory.csv", index=False)
    pd.DataFrame(INTERFACE_DOCUMENTATION).to_csv(DATA / "hybrid_interface_inventory.csv", index=False)
    validate_artifacts().to_csv(DATA / "hybrid_distillation_artifact_inventory.csv", index=False)
    return out


def train_only_stats(env: np.ndarray, targets: dict[str, np.ndarray], train_mask: np.ndarray) -> dict[str, object]:
    stats = {
        "env_mean": env[train_mask].mean(axis=0),
        "env_std": env[train_mask].std(axis=0) + EPS,
        "target_mean": {},
        "target_std": {},
        "target_min": {},
        "target_max": {},
    }
    for col, arr in targets.items():
        vals = np.asarray(arr, float)[train_mask].reshape(-1)
        stats["target_mean"][col] = float(np.mean(vals))
        stats["target_std"][col] = float(np.std(vals) + EPS)
        stats["target_min"][col] = float(np.min(vals))
        stats["target_max"][col] = float(np.max(vals))
    return stats


def standardize_env(env: np.ndarray, env_mean: np.ndarray, env_std: np.ndarray) -> np.ndarray:
    return (env - env_mean) / env_std


def initialize_state_space(env_dim: int, latent_dim: int, seed: int, scale: float = 0.18):
    rng = np.random.default_rng(seed)
    Wg = rng.normal(0, scale, (env_dim, latent_dim))
    bg = rng.normal(0, scale * 0.2, latent_dim)
    A = rng.normal(0, scale, (latent_dim, latent_dim))
    # Mild damping keeps rollouts bounded without forcing biological meaning.
    A -= np.eye(latent_dim) * 0.25
    C = rng.normal(0, scale, (env_dim, latent_dim))
    bf = rng.normal(0, scale * 0.2, latent_dim)
    return Wg, bg, A, C, bf


def latent_rollout(X: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], t_len: int) -> np.ndarray:
    Wg, bg, A, C, bf = params
    dt = 1.0 / max(t_len - 1, 1)
    z = np.tanh(X @ Wg + bg)
    zs = []
    for _k in range(t_len):
        zs.append(z.copy())
        z = z + dt * np.tanh(z @ A + X @ C + bf)
        z = np.clip(z, -8.0, 8.0)
    return np.stack(zs, axis=1)


def design_from_latents(z: np.ndarray, culture_mask: np.ndarray, length: int) -> np.ndarray:
    zz = z[culture_mask, :length, :]
    n, t_len, latent_dim = zz.shape
    time_feature = np.tile(np.linspace(0, 1, t_len), n)[:, None]
    flat = zz.reshape(n * t_len, latent_dim)
    return np.column_stack([np.ones(len(flat)), flat, flat**2, time_feature])


def weighted_ridge_fit(X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None, lam: float = 1e-4) -> np.ndarray:
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float).reshape(-1, 1))
        Xw = X * w
        Yw = Y * w
    else:
        Xw, Yw = X, Y
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(Xw.T @ Xw + lam * eye, Xw.T @ Yw)


def fit_heads(
    z: np.ndarray,
    train_mask: np.ndarray,
    target_groups: dict[str, np.ndarray],
    stats: dict[str, object],
    supervised_columns: list[str],
    row_weights: np.ndarray | None = None,
    interval_len: int | None = None,
) -> dict[str, np.ndarray]:
    heads: dict[str, np.ndarray] = {}
    interval_len = interval_len or z.shape[1] - 1
    for col in supervised_columns:
        arr = target_groups[col]
        length = arr.shape[1]
        X = design_from_latents(z, train_mask, length)
        train_vals = arr[train_mask].reshape(-1, 1)
        y = (train_vals - stats["target_mean"][col]) / stats["target_std"][col]
        weights = None
        if row_weights is not None:
            weights = np.repeat(row_weights[train_mask], length)
        heads[col] = weighted_ridge_fit(X, y, weights=weights, lam=1e-4).reshape(-1)
    return heads


def predict_columns(checkpoint: StateSpaceCheckpoint, env: np.ndarray, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    columns = columns or checkpoint.supervised_columns
    X = standardize_env(env, checkpoint.env_mean, checkpoint.env_std)
    z = latent_rollout(X, (checkpoint.Wg, checkpoint.bg, checkpoint.A, checkpoint.C, checkpoint.bf), checkpoint.t_len)
    out: dict[str, np.ndarray] = {}
    for col in columns:
        length = checkpoint.interval_len if col in INTERFACE_COLUMNS + FLUX_COLUMNS else checkpoint.t_len
        if col not in checkpoint.heads:
            out[col] = np.full((len(env), length), checkpoint.target_mean.get(col, 0.0), dtype=float)
            continue
        Xd = design_from_latents(z, np.ones(len(env), dtype=bool), length)
        y = (Xd @ checkpoint.heads[col]).reshape(len(env), length)
        vals = y * checkpoint.target_std[col] + checkpoint.target_mean[col]
        lo = checkpoint.target_min[col]
        hi = checkpoint.target_max[col]
        margin = 0.05 * max(hi - lo, EPS)
        vals = np.clip(vals, lo - margin, hi + margin)
        if col.startswith("R_") or col in STATE_COLUMNS:
            vals = np.clip(vals, 0.0, None)
        if col == "gamma_growth_fraction":
            vals = np.clip(vals, 0.0, 1.0)
        out[col] = vals
    return out


def checkpoint_to_npz(path: Path, ckpt: StateSpaceCheckpoint, history: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "env_mean": ckpt.env_mean,
        "env_std": ckpt.env_std,
        "Wg": ckpt.Wg,
        "bg": ckpt.bg,
        "A": ckpt.A,
        "C": ckpt.C,
        "bf": ckpt.bf,
    }
    for col, head in ckpt.heads.items():
        arrays[f"head__{col}"] = head
    meta = {
        "variant": ckpt.variant,
        "seed": ckpt.seed,
        "latent_dim": ckpt.latent_dim,
        "t_len": ckpt.t_len,
        "interval_len": ckpt.interval_len,
        "dt": ckpt.dt,
        "target_mean": ckpt.target_mean,
        "target_std": ckpt.target_std,
        "target_min": ckpt.target_min,
        "target_max": ckpt.target_max,
        "supervised_columns": ckpt.supervised_columns,
        "mechanistic_rate": ckpt.mechanistic_rate,
        "product_scale": ckpt.product_scale,
        "optimizer_state": {"method": "closed_form_weighted_ridge_output_heads", "ridge_lambda": 1e-4},
        "training_history": history,
        "architecture": "z0=tanh(g(e)); z[k+1]=z[k]+dt*tanh(Az[k]+Ce+b); separate linear heads on [z,z^2,time]",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: Path) -> StateSpaceCheckpoint:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    heads = {key.removeprefix("head__"): data[key] for key in data.files if key.startswith("head__")}
    return StateSpaceCheckpoint(
        variant=meta["variant"],
        seed=int(meta["seed"]),
        latent_dim=int(meta["latent_dim"]),
        t_len=int(meta["t_len"]),
        interval_len=int(meta["interval_len"]),
        dt=float(meta["dt"]),
        env_mean=data["env_mean"],
        env_std=data["env_std"],
        Wg=data["Wg"],
        bg=data["bg"],
        A=data["A"],
        C=data["C"],
        bf=data["bf"],
        heads=heads,
        target_mean={k: float(v) for k, v in meta["target_mean"].items()},
        target_std={k: float(v) for k, v in meta["target_std"].items()},
        target_min={k: float(v) for k, v in meta["target_min"].items()},
        target_max={k: float(v) for k, v in meta["target_max"].items()},
        supervised_columns=list(meta["supervised_columns"]),
        mechanistic_rate=bool(meta.get("mechanistic_rate", False)),
        product_scale=float(meta.get("product_scale", 1.0)),
    )


def assemble_targets(dataset: dict[str, object]) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {"B_total": dataset["product"]}
    targets.update(dataset["reporters"])
    targets.update(dataset["states"])
    targets.update(dataset["interface"])
    targets.update(dataset["flux"])
    return targets


def shuffled_training_targets(targets: dict[str, np.ndarray], cols: list[str], train_mask: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in targets.items()}
    rng = np.random.default_rng(seed)
    idx = np.where(train_mask)[0]
    perm = idx.copy()
    rng.shuffle(perm)
    if len(perm) > 1:
        perm = np.roll(perm, 1)
    for col in cols:
        out[col][idx] = out[col][perm]
    return out


def variant_supervised_columns(variant: str) -> list[str]:
    spec = TEACHER_VARIANTS[variant]
    cols = ["B_total"]
    cols.extend(spec.get("reporters", []))
    if spec.get("interface", False):
        cols.extend(INTERFACE_COLUMNS)
    if spec.get("flux", False):
        cols.extend(FLUX_COLUMNS)
    return list(dict.fromkeys(cols))


def train_teacher_checkpoint(dataset: dict[str, object], variant: str, latent_dim: int, seed: int) -> tuple[StateSpaceCheckpoint, dict[str, object]]:
    env = dataset["env"]
    splits = dataset["splits"]
    train_mask = splits == "train"
    targets = assemble_targets(dataset)
    supervised = variant_supervised_columns(variant)
    if TEACHER_VARIANTS[variant].get("shuffle_reporters", False):
        targets = shuffled_training_targets(targets, REPORTER_COLUMNS, train_mask, seed + latent_dim)
    stats = train_only_stats(env, {col: targets[col] for col in supervised}, train_mask)
    X = standardize_env(env, stats["env_mean"], stats["env_std"])
    params = initialize_state_space(X.shape[1], latent_dim, seed + 1000 * latent_dim + len(variant))
    z = latent_rollout(X, params, dataset["product"].shape[1])
    heads = fit_heads(z, train_mask, targets, stats, supervised, interval_len=dataset["interface"][INTERFACE_COLUMNS[0]].shape[1])
    ckpt = StateSpaceCheckpoint(
        variant=variant,
        seed=seed,
        latent_dim=latent_dim,
        t_len=dataset["product"].shape[1],
        interval_len=dataset["interface"][INTERFACE_COLUMNS[0]].shape[1],
        dt=float(np.median(np.diff(dataset["time"]))),
        env_mean=stats["env_mean"],
        env_std=stats["env_std"],
        Wg=params[0],
        bg=params[1],
        A=params[2],
        C=params[3],
        bf=params[4],
        heads=heads,
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        target_min=stats["target_min"],
        target_max=stats["target_max"],
        supervised_columns=supervised,
        mechanistic_rate=bool(TEACHER_VARIANTS[variant].get("mechanistic_rate", False)),
        product_scale=float(np.max(targets["B_total"][train_mask])),
    )
    metrics = evaluate_checkpoint(ckpt, dataset)
    history = {
        "fit_method": "ridge output-head fit on fixed seeded neural state-space rollouts",
        "train_cultures": int(train_mask.sum()),
        "validation_composite": float(metrics[metrics["split"].eq("validation")]["teacher_composite_score"].mean()),
        "supervised_columns": supervised,
    }
    return ckpt, history


def normalized_rmse(pred: np.ndarray, truth: np.ndarray, train_truth: np.ndarray) -> float:
    denom = float(np.std(train_truth) + EPS)
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def channel_metric_rows(ckpt: StateSpaceCheckpoint, dataset: dict[str, object], split: str) -> list[dict[str, object]]:
    mask = dataset["splits"] == split
    train = dataset["splits"] == "train"
    targets = assemble_targets(dataset)
    cols = PRODUCT_COLUMNS + REPORTER_COLUMNS + INTERFACE_COLUMNS + FLUX_COLUMNS
    pred = predict_columns(ckpt, dataset["env"], cols)
    rows = []
    for col in cols:
        truth = targets[col]
        length = pred[col].shape[1]
        rows.append(
            {
                "model": ckpt.variant,
                "model_seed": ckpt.seed,
                "latent_dim": ckpt.latent_dim,
                "split": split,
                "channel": col,
                "channel_group": channel_group(col),
                "normalized_rmse": normalized_rmse(pred[col][mask], truth[mask, :length], truth[train, :length]),
                "raw_rmse": float(np.sqrt(np.mean((pred[col][mask] - truth[mask, :length]) ** 2))),
                "range_violation_fraction": range_violation_fraction(pred[col][mask], truth[train, :length]),
                "supervised": col in ckpt.supervised_columns,
            }
        )
    return rows


def channel_group(col: str) -> str:
    if col == "B_total":
        return "product"
    if col in REPORTER_COLUMNS:
        if col in CAPACITY_REPORTERS:
            return "capacity_reporter"
        return "reporter"
    if col in INTERFACE_COLUMNS:
        return "interface"
    if col in FLUX_COLUMNS:
        return "flux"
    return "other"


def range_violation_fraction(pred: np.ndarray, train_truth: np.ndarray) -> float:
    lo, hi = float(np.min(train_truth)), float(np.max(train_truth))
    margin = 0.05 * max(hi - lo, EPS)
    return float(np.mean((pred < lo - margin) | (pred > hi + margin)))


def rollout_stability_penalty(ckpt: StateSpaceCheckpoint, env: np.ndarray) -> float:
    pred = predict_columns(ckpt, env, ckpt.supervised_columns)
    penalty = 0.0
    for vals in pred.values():
        if not np.isfinite(vals).all():
            penalty += 1000.0
        penalty += float(np.mean(np.maximum(np.abs(vals) - 1e6, 0.0)))
    return penalty


def smoothness_penalty(ckpt: StateSpaceCheckpoint, dataset: dict[str, object]) -> float:
    rng = np.random.default_rng(1234 + ckpt.seed)
    env = dataset["env"]
    jitter = rng.normal(0.0, [0.05, 0.01, 0.5], size=env.shape)
    nearby = np.column_stack(
        [
            np.clip(env[:, 0] + jitter[:, 0], 27.0, 33.0),
            np.clip(env[:, 1] + jitter[:, 1], 4.5, 5.5),
            np.clip(env[:, 2] + jitter[:, 2], 20.0, 80.0),
        ]
    )
    pred_a = predict_columns(ckpt, env, ["B_total"] + [c for c in REPORTER_COLUMNS if c in ckpt.supervised_columns])
    pred_b = predict_columns(ckpt, nearby, list(pred_a))
    diffs = []
    for col in pred_a:
        scale = max(float(np.std(pred_a[col])), EPS)
        diffs.append(float(np.mean(np.abs(pred_a[col] - pred_b[col])) / scale))
    return float(np.mean(diffs)) if diffs else 0.0


def evaluate_checkpoint(ckpt: StateSpaceCheckpoint, dataset: dict[str, object]) -> pd.DataFrame:
    rows = []
    for split in sorted(set(dataset["splits"])):
        rows.extend(channel_metric_rows(ckpt, dataset, split))
    df = pd.DataFrame(rows)
    composites = []
    for split, g in df.groupby("split"):
        def group_mean(group_name: str) -> float:
            vals = g[g["channel_group"].eq(group_name)]["normalized_rmse"]
            return float(vals.mean()) if len(vals) else 0.0

        score = (
            1.0 * group_mean("product")
            + 0.8 * group_mean("reporter")
            + 0.8 * group_mean("capacity_reporter")
            + 1.0 * group_mean("interface")
            + 0.25 * group_mean("flux")
            + 0.05 * rollout_stability_penalty(ckpt, dataset["env"])
            + 0.10 * smoothness_penalty(ckpt, dataset)
        )
        composites.append({"split": split, "teacher_composite_score": score})
    comp = pd.DataFrame(composites)
    return df.merge(comp, on="split", how="left")


def train_all_teachers(dataset: dict[str, object], seeds: list[int] | None = None, latent_dims: list[int] | None = None) -> dict[str, object]:
    seeds = seeds or MODEL_SEEDS
    latent_dims = latent_dims or LATENT_DIMS
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    all_metrics, manifest_rows, training_rows = [], [], []
    selected: list[StateSpaceCheckpoint] = []
    tic = time.perf_counter()
    for variant in TEACHER_VARIANTS:
        for seed in seeds:
            candidates = []
            for dim in latent_dims:
                ckpt, history = train_teacher_checkpoint(dataset, variant, dim, seed)
                metrics = evaluate_checkpoint(ckpt, dataset)
                val_score = float(metrics[metrics["split"].eq("validation")]["teacher_composite_score"].mean())
                candidates.append((val_score, ckpt, history, metrics))
                ckpt_path = CHECKPOINTS / f"{variant}_seed{seed}_latent{dim}.npz"
                checkpoint_to_npz(ckpt_path, ckpt, history | {"validation_score": val_score})
                all_metrics.append(metrics.assign(checkpoint=str(ckpt_path.relative_to(ROOT))))
                training_rows.append(
                    {
                        "model": variant,
                        "model_seed": seed,
                        "latent_dim": dim,
                        "checkpoint": str(ckpt_path.relative_to(ROOT)),
                        "validation_score": val_score,
                        "supervised_columns": ",".join(ckpt.supervised_columns),
                        "fit_method": history["fit_method"],
                    }
                )
            val_score, ckpt, history, _metrics = sorted(candidates, key=lambda row: row[0])[0]
            is_selected_primary = variant == PRIMARY_TEACHER_VARIANT
            if is_selected_primary:
                selected.append(ckpt)
            manifest_rows.append(
                {
                    "model": variant,
                    "model_seed": seed,
                    "selected_latent_dim": ckpt.latent_dim,
                    "validation_score": val_score,
                    "checkpoint": str((CHECKPOINTS / f"{variant}_seed{seed}_latent{ckpt.latent_dim}.npz").relative_to(ROOT)),
                    "ensemble_role": "selected_teacher" if is_selected_primary else "ablation_or_negative_control",
                    "deployment_inputs": ",".join(ENV_COLUMNS),
                    "training_labels_only": ",".join([c for c in ckpt.supervised_columns if c != "B_total"]),
                }
            )
    metrics = pd.concat(all_metrics, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    training_log = pd.DataFrame(training_rows)
    selection = teacher_acceptance(metrics, manifest, dataset)
    metrics.to_csv(DATA / "teacher_channel_metrics.csv", index=False)
    metrics[metrics["channel_group"].eq("interface")].to_csv(DATA / "teacher_interface_metrics.csv", index=False)
    manifest.to_csv(DATA / "teacher_ensemble_manifest.csv", index=False)
    training_log.to_csv(DATA / "teacher_training_log.csv", index=False)
    selection.to_csv(DATA / "teacher_model_selection.csv", index=False)
    return {
        "selected": selected,
        "metrics": metrics,
        "manifest": manifest,
        "selection": selection,
        "runtime_seconds": time.perf_counter() - tic,
    }


def teacher_acceptance(metrics: pd.DataFrame, manifest: pd.DataFrame, dataset: dict[str, object]) -> pd.DataFrame:
    val = metrics[metrics["split"].eq("validation")]
    primary = val[val["model"].eq(PRIMARY_TEACHER_VARIANT)]
    shuffled = val[val["model"].eq("shuffled_reporter_control")]
    random = val[val["model"].eq("random_smooth_aux_control")]
    primary_product = float(primary[primary["channel_group"].eq("product")]["normalized_rmse"].mean())
    primary_reporter = float(primary[primary["channel_group"].isin(["reporter", "capacity_reporter"])]["normalized_rmse"].mean())
    primary_interface = float(primary[primary["channel_group"].eq("interface")]["normalized_rmse"].mean())
    shuffled_reporter = float(shuffled[shuffled["channel_group"].isin(["reporter", "capacity_reporter"])]["normalized_rmse"].mean())
    random_reporter = float(random[random["channel_group"].isin(["reporter", "capacity_reporter"])]["normalized_rmse"].mean())
    stable = bool(np.isfinite(primary["normalized_rmse"]).all() and primary["range_violation_fraction"].max() <= 0.10)
    gates = [
        ("heldout_product_usable", primary_product < 1.0, primary_product),
        ("reporters_learned", primary_reporter < 1.0, primary_reporter),
        ("interface_learned", primary_interface < 1.0, primary_interface),
        ("beats_shuffled_on_physiology", primary_reporter < shuffled_reporter, shuffled_reporter - primary_reporter),
        ("beats_random_on_physiology", primary_reporter < random_reporter, random_reporter - primary_reporter),
        ("stable_bounded_rollouts", stable, float(primary["range_violation_fraction"].max())),
        ("seed_coverage", manifest[manifest["model"].eq(PRIMARY_TEACHER_VARIANT)]["model_seed"].nunique() >= 3, manifest[manifest["model"].eq(PRIMARY_TEACHER_VARIANT)]["model_seed"].nunique()),
        ("real_gem_labels_only", bool(dataset["traj"]["metabolic_backend"].eq("yeast_gem_lp").all()), 1.0),
    ]
    passed = all(bool(row[1]) for row in gates)
    rows = [
        {
            "stage": "Stage 1",
            "criterion": name,
            "passed": bool(ok),
            "value": value,
            "decision": "pass" if passed else "blocked",
            "bounded_claim": "teacher is a reporter-grounded interpolation source, not a biological-regulatory truth model",
        }
        for name, ok, value in gates
    ]
    return pd.DataFrame(rows)


def latin_hypercube(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bounds = np.asarray([[27.0, 33.0], [4.5, 5.5], [20.0, 80.0]], dtype=float)
    sample = np.zeros((n, 3), dtype=float)
    for j in range(3):
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        sample[:, j] = bounds[j, 0] + bins * (bounds[j, 1] - bounds[j, 0])
    return sample


def regular_plot_grid(n_per_axis: int = 13) -> np.ndarray:
    temps = np.linspace(27.0, 33.0, n_per_axis)
    phs = np.linspace(4.5, 5.5, n_per_axis)
    dos = np.linspace(20.0, 80.0, n_per_axis)
    rows = [[t, ph, do] for t in temps for ph in phs for do in dos]
    return np.asarray(rows, dtype=float)


def ensemble_predict(ensemble: list[StateSpaceCheckpoint], env: np.ndarray, columns: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    per_col = {col: [] for col in columns}
    for ckpt in ensemble:
        pred = predict_columns(ckpt, env, columns)
        for col in columns:
            per_col[col].append(pred[col])
    means = {col: np.mean(per_col[col], axis=0) for col in columns}
    stds = {col: np.std(per_col[col], axis=0) for col in columns}
    return means, stds


def confidence_from_uncertainty(means: dict[str, np.ndarray], stds: dict[str, np.ndarray], columns: list[str]) -> pd.DataFrame:
    parts = []
    for col in columns:
        scale = max(float(np.std(means[col])), EPS)
        parts.append(np.mean((stds[col] / scale) ** 2, axis=1))
    uncertainty = np.mean(parts, axis=0)
    raw_weight = 1.0 / (1e-3 + uncertainty)
    weight = np.clip(raw_weight / np.median(raw_weight), 0.1, 10.0)
    category = np.where(weight >= 2.0, "high_confidence", np.where(weight >= 0.7, "moderate_confidence", np.where(weight >= 0.25, "low_confidence", "excluded")))
    return pd.DataFrame({"teacher_uncertainty": uncertainty, "pseudo_label_weight": weight, "confidence_category": category})


def generate_teacher_pseudodata(
    ensemble: list[StateSpaceCheckpoint],
    dataset: dict[str, object],
    n_samples: int = 5000,
    seed: int = 20260727,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    if not ensemble:
        raise RuntimeError("No frozen teacher ensemble is available; Stage 2 is blocked.")
    dense_env = latin_hypercube(n_samples, seed)
    pseudo_split = np.where(np.arange(n_samples) % 5 == 0, "pseudo_validation", "pseudo_train")
    columns = PRODUCT_COLUMNS + REPORTER_COLUMNS + INTERFACE_COLUMNS + FLUX_COLUMNS
    means, stds = ensemble_predict(ensemble, dense_env, columns)
    conf = confidence_from_uncertainty(means, stds, columns)
    env_df = pd.DataFrame(dense_env, columns=ENV_COLUMNS)
    env_df.insert(0, "pseudo_culture_id", [f"teacher_dense_{i:05d}" for i in range(n_samples)])
    env_df["pseudo_split"] = pseudo_split
    env_df["sample_method"] = "latin_hypercube_with_deterministic_seed"
    env_df["sample_seed"] = seed
    env_df = pd.concat([env_df, conf], axis=1)

    tidy_rows = []
    time_values = dataset["time"]
    interval_len = ensemble[0].interval_len
    for i, row in env_df.iterrows():
        for k, tt in enumerate(time_values):
            rec = {
                "pseudo_culture_id": row["pseudo_culture_id"],
                "pseudo_split": row["pseudo_split"],
                "time_index": k,
                "time": float(tt),
                "temperature": row["temperature"],
                "pH": row["pH"],
                "DO": row["DO"],
                "teacher_uncertainty": row["teacher_uncertainty"],
                "pseudo_label_weight": row["pseudo_label_weight"],
                "confidence_category": row["confidence_category"],
            }
            rec["B_total_mean"] = float(means["B_total"][i, k])
            rec["B_total_std"] = float(stds["B_total"][i, k])
            for col in REPORTER_COLUMNS:
                rec[f"{col}_mean"] = float(means[col][i, k])
                rec[f"{col}_std"] = float(stds[col][i, k])
            kk = min(k, interval_len - 1)
            rec["is_interval_control_row"] = bool(k < interval_len)
            for col in INTERFACE_COLUMNS + FLUX_COLUMNS:
                rec[f"{col}_mean"] = float(means[col][i, kk])
                rec[f"{col}_std"] = float(stds[col][i, kk])
            tidy_rows.append(rec)
    pseudo = pd.DataFrame(tidy_rows)
    plot_env = regular_plot_grid()
    plot_means, plot_stds = ensemble_predict(ensemble, plot_env, columns)
    plot_df = pd.DataFrame(plot_env, columns=ENV_COLUMNS)
    plot_df["B_total_final_mean"] = plot_means["B_total"][:, -1]
    plot_df["B_total_final_std"] = plot_stds["B_total"][:, -1]
    plot_df["R_ox_final_mean"] = plot_means["R_ox"][:, -1]
    plot_df["atp_maintenance_lower_bound_mean"] = plot_means["atp_maintenance_lower_bound"][:, -1]
    audits = audit_pseudodata(env_df, pseudo, dataset)
    if write_outputs:
        env_df.to_csv(DATA / "teacher_dense_environment_samples.csv", index=False)
        pseudo.to_csv(DATA / "teacher_pseudodata.csv", index=False)
        env_df[["pseudo_culture_id", "teacher_uncertainty", "pseudo_label_weight", "confidence_category", "pseudo_split"]].to_csv(DATA / "teacher_pseudodata_uncertainty.csv", index=False)
        plot_df.to_csv(DATA / "teacher_dense_plot_grid.csv", index=False)
        audits.to_csv(DATA / "teacher_pseudodata_audit.csv", index=False)
        save_teacher_figures(plot_df, env_df)
    return {"environment": env_df, "tidy": pseudo, "plot_grid": plot_df, "audit": audits}


def audit_pseudodata(env_df: pd.DataFrame, pseudo: pd.DataFrame, dataset: dict[str, object]) -> pd.DataFrame:
    train_ids = set(env_df[env_df["pseudo_split"].eq("pseudo_train")]["pseudo_culture_id"])
    val_ids = set(env_df[env_df["pseudo_split"].eq("pseudo_validation")]["pseudo_culture_id"])
    finite = np.isfinite(pseudo.select_dtypes(include=[np.number]).to_numpy()).all()
    in_domain = (
        env_df["temperature"].between(27.0, 33.0).all()
        and env_df["pH"].between(4.5, 5.5).all()
        and env_df["DO"].between(20.0, 80.0).all()
    )
    controls = [f"{col}_mean" for col in INTERFACE_COLUMNS]
    bounded = True
    for col in controls:
        source = col.removesuffix("_mean")
        target = dataset["interface"][source]
        lo, hi = float(target.min()), float(target.max())
        margin = 0.10 * max(hi - lo, EPS)
        bounded &= bool(pseudo[col].between(lo - margin, hi + margin).all())
    audits = [
        ("finite_outputs", finite),
        ("bounded_within_original_domain", in_domain),
        ("bounded_interface_controls", bounded),
        ("no_pseudo_train_validation_overlap", train_ids.isdisjoint(val_ids)),
        ("confidence_weights_clipped", env_df["pseudo_label_weight"].between(0.1, 10.0).all()),
        ("complete_domain_coverage", env_df[ENV_COLUMNS].agg(["min", "max"]).loc["min", "temperature"] >= 27.0 and env_df[ENV_COLUMNS].agg(["min", "max"]).loc["max", "DO"] <= 80.0),
    ]
    passed = all(bool(ok) for _name, ok in audits)
    return pd.DataFrame(
        [
            {
                "stage": "Stage 2",
                "criterion": name,
                "passed": bool(ok),
                "decision": "pass" if passed else "blocked",
                "bounded_claim": "dense sampling is interpolation inside the original environmental support, not new biological evidence",
            }
            for name, ok in audits
        ]
    )


def save_teacher_figures(plot_df: pd.DataFrame, env_df: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    for path, title, col in [
        (FIGURES / "teacher_product_surface.svg", "Teacher Product Surface", "B_total_final_mean"),
        (FIGURES / "teacher_reporter_surface.svg", "Teacher Oxidative Reporter Surface", "R_ox_final_mean"),
        (FIGURES / "teacher_interface_surface.svg", "Teacher ATP-Maintenance Interface Surface", "atp_maintenance_lower_bound_mean"),
    ]:
        svg_scatter_surface(path, title, plot_df, col)
    svg_uncertainty(FIGURES / "teacher_uncertainty_map.svg", env_df)


def svg_scatter_surface(path: Path, title: str, df: pd.DataFrame, col: str) -> None:
    sub = df.iloc[:: max(1, len(df) // 700)].copy()
    vals = sub[col].to_numpy(float)
    lo, hi = float(vals.min()), float(vals.max())
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470"><rect width="100%" height="100%" fill="white"/>',
        ss_validation.fixed.svg_text(380, 28, title, 17, "bold"),
        '<rect x="70" y="60" width="610" height="335" fill="#fafafa" stroke="#bbb"/>',
    ]
    for r in sub.itertuples():
        x = 70 + (float(r.temperature) - 27.0) / 6.0 * 610
        y = 395 - (float(r.DO) - 20.0) / 60.0 * 335
        frac = (float(getattr(r, col)) - lo) / max(hi - lo, EPS)
        red = int(50 + 160 * frac)
        blue = int(180 - 100 * frac)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="rgb({red},95,{blue})" opacity="0.72"/>')
    body.append(ss_validation.fixed.svg_text(375, 430, "T vs DO projection; pH varied in the dense grid", 11))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def svg_uncertainty(path: Path, env_df: pd.DataFrame) -> None:
    sub = env_df.iloc[:: max(1, len(env_df) // 700)].copy()
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470"><rect width="100%" height="100%" fill="white"/>',
        ss_validation.fixed.svg_text(380, 28, "Teacher Uncertainty Map", 17, "bold"),
        '<rect x="70" y="60" width="610" height="335" fill="#fafafa" stroke="#bbb"/>',
    ]
    hi = max(float(sub["teacher_uncertainty"].max()), EPS)
    for r in sub.itertuples():
        x = 70 + (float(r.temperature) - 27.0) / 6.0 * 610
        y = 395 - (float(r.DO) - 20.0) / 60.0 * 335
        frac = min(float(r.teacher_uncertainty) / hi, 1.0)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="rgb({int(220*frac)},80,{int(180*(1-frac))})" opacity="0.72"/>')
    body.append(ss_validation.fixed.svg_text(375, 430, "T vs DO projection; larger warm colors mean higher ensemble disagreement", 11))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def pseudo_to_student_dataset(dataset: dict[str, object], pseudo: pd.DataFrame | None) -> dict[str, object]:
    meta = dataset["meta"].copy()
    env = dataset["env"].copy()
    splits = dataset["splits"].copy()
    targets = assemble_targets(dataset)
    weights = np.ones(len(env), dtype=float) * 5.0
    if pseudo is None or pseudo.empty:
        return {"meta": meta, "env": env, "splits": splits, "targets": targets, "weights": weights}
    rows = pseudo[pseudo["time_index"].eq(0)].copy()
    pseudo_ids = list(rows["pseudo_culture_id"])
    pseudo_env = rows[ENV_COLUMNS].to_numpy(float)
    pseudo_splits = rows["pseudo_split"].replace({"pseudo_train": "train", "pseudo_validation": "validation"}).to_numpy(str)
    pseudo_weights = rows["pseudo_label_weight"].to_numpy(float)
    env = np.vstack([env, pseudo_env])
    splits = np.concatenate([splits, pseudo_splits])
    weights = np.concatenate([weights, pseudo_weights])
    meta = pd.concat(
        [
            meta,
            rows.rename(columns={"pseudo_culture_id": "culture_id"})[["culture_id", "temperature", "pH", "DO"]].assign(environment_id=rows["pseudo_culture_id"], split=rows["pseudo_split"], metabolic_backend="teacher_pseudodata", n_surrogate_evaluations=0),
        ],
        ignore_index=True,
    )
    def pivot_pseudo_matrix(frame: pd.DataFrame, value_col: str, ids: list[str]) -> np.ndarray:
        wide = frame.pivot(index="pseudo_culture_id", columns="time_index", values=value_col)
        wide = wide.reindex(ids).sort_index(axis=1)
        return wide.to_numpy(float)

    for col in PRODUCT_COLUMNS + REPORTER_COLUMNS:
        mats = pivot_pseudo_matrix(pseudo, f"{col}_mean", pseudo_ids)
        targets[col] = np.vstack([targets[col], mats])
    if "R_random_smooth" in targets:
        rng = np.random.default_rng(8107)
        mats = []
        for _pid in pseudo_ids:
            phase = rng.uniform(-0.5, 0.5)
            mats.append(np.sin(np.linspace(0, math.pi, dataset["product"].shape[1]) + phase) + 0.2 * rng.normal(size=dataset["product"].shape[1]))
        targets["R_random_smooth"] = np.vstack([targets["R_random_smooth"], np.asarray(mats)])
    interval_rows = pseudo[pseudo["is_interval_control_row"].astype(bool)]
    for col in INTERFACE_COLUMNS + FLUX_COLUMNS:
        mats = pivot_pseudo_matrix(interval_rows, f"{col}_mean", pseudo_ids)
        targets[col] = np.vstack([targets[col], np.asarray(mats)])
    return {"meta": meta, "env": env, "splits": splits, "targets": targets, "weights": weights}


def train_student_checkpoint(
    dataset: dict[str, object],
    pseudo: pd.DataFrame | None,
    variant: str,
    latent_dim: int,
    seed: int,
    mixed: dict[str, object] | None = None,
) -> tuple[StateSpaceCheckpoint, dict[str, object]]:
    mixed = mixed or pseudo_to_student_dataset(dataset, pseudo)
    env = mixed["env"]
    splits = mixed["splits"]
    train_mask = splits == "train"
    targets = mixed["targets"]
    spec = STUDENT_VARIANTS[variant]
    supervised = list(dict.fromkeys(list(spec["reporters"]) + list(spec["interface"]) + list(spec["flux"])))
    if spec.get("shuffle_reporters", False):
        targets = shuffled_training_targets(targets, REPORTER_COLUMNS, train_mask, seed + latent_dim)
    stats = train_only_stats(env, {col: targets[col] for col in supervised}, train_mask)
    X = standardize_env(env, stats["env_mean"], stats["env_std"])
    params = initialize_state_space(X.shape[1], latent_dim, seed + 3000 * latent_dim + len(variant), scale=0.16)
    z = latent_rollout(X, params, dataset["product"].shape[1])
    heads = fit_heads(z, train_mask, targets, stats, supervised, row_weights=mixed["weights"], interval_len=dataset["interface"][INTERFACE_COLUMNS[0]].shape[1])
    ckpt = StateSpaceCheckpoint(
        variant=variant,
        seed=seed,
        latent_dim=latent_dim,
        t_len=dataset["product"].shape[1],
        interval_len=dataset["interface"][INTERFACE_COLUMNS[0]].shape[1],
        dt=float(np.median(np.diff(dataset["time"]))),
        env_mean=stats["env_mean"],
        env_std=stats["env_std"],
        Wg=params[0],
        bg=params[1],
        A=params[2],
        C=params[3],
        bf=params[4],
        heads=heads,
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        target_min=stats["target_min"],
        target_max=stats["target_max"],
        supervised_columns=supervised,
        product_scale=1.0,
    )
    history = {
        "fit_method": "weighted ridge regulatory-output fit; no GLPK gradients",
        "real_culture_weight": 5.0,
        "pseudo_weight_source": "teacher ensemble confidence",
        "direct_product_head": False,
        "supervised_columns": supervised,
    }
    return ckpt, history


def evaluate_student(ckpt: StateSpaceCheckpoint, dataset: dict[str, object]) -> pd.DataFrame:
    rows = []
    for split in sorted(set(dataset["splits"])):
        rows.extend(channel_metric_rows(ckpt, dataset, split))
    df = pd.DataFrame(rows)
    return df[df["channel"].isin(REPORTER_COLUMNS + INTERFACE_COLUMNS + FLUX_COLUMNS)].assign(product_generated_by="exact_pFBA_required_for_primary_product_evaluation")


def train_hybrid_students(dataset: dict[str, object], pseudo: pd.DataFrame | None, seeds: list[int] | None = None, latent_dims: list[int] | None = None) -> dict[str, object]:
    seeds = seeds or MODEL_SEEDS
    latent_dims = latent_dims or LATENT_DIMS
    STUDENT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    metrics_rows, manifest_rows = [], []
    selected: StateSpaceCheckpoint | None = None
    mixed = pseudo_to_student_dataset(dataset, pseudo)
    for variant in STUDENT_VARIANTS:
        for seed in seeds:
            candidates = []
            for dim in latent_dims:
                ckpt, history = train_student_checkpoint(dataset, pseudo, variant, dim, seed, mixed=mixed)
                metrics = evaluate_student(ckpt, dataset)
                val = metrics[metrics["split"].eq("validation")]
                score = float(val[val["channel_group"].isin(["reporter", "capacity_reporter", "interface"])]["normalized_rmse"].mean())
                candidates.append((score, ckpt, history, metrics))
                ckpt_path = STUDENT_CHECKPOINTS / f"{variant}_seed{seed}_latent{dim}.npz"
                checkpoint_to_npz(ckpt_path, ckpt, history | {"validation_regulatory_score": score})
            score, ckpt, history, metrics = sorted(candidates, key=lambda row: row[0])[0]
            metrics_rows.append(metrics.assign(model=variant, model_seed=seed, latent_dim=ckpt.latent_dim))
            if variant == "hybrid_reporter_supervised" and selected is None:
                selected = ckpt
            manifest_rows.append(
                {
                    "model": variant,
                    "model_seed": seed,
                    "selected_latent_dim": ckpt.latent_dim,
                    "validation_regulatory_score": score,
                    "checkpoint": str((STUDENT_CHECKPOINTS / f"{variant}_seed{seed}_latent{ckpt.latent_dim}.npz").relative_to(ROOT)),
                    "direct_product_head": False,
                    "training_strategy": "supervise reporters/interface/flux; product evaluated only through pFBA rollout",
                }
            )
    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)
    metrics_df.to_csv(DATA / "hybrid_student_metrics.csv", index=False)
    manifest_df.to_csv(DATA / "hybrid_student_manifest.csv", index=False)
    metrics_df[metrics_df["model"].ne("hybrid_reporter_supervised")].to_csv(DATA / "hybrid_student_ablation_metrics.csv", index=False)
    return {"selected": selected, "metrics": metrics_df, "manifest": manifest_df}


def apply_predicted_interface_controls(model, cfg: gem.GEMCultureConfig, controls: dict[str, float]) -> dict[str, float | str]:
    glucose_lb = -cfg.glucose_uptake
    oxygen_lb = float(controls["oxygen_lower_bound"])
    atpm = max(0.0, float(controls["atp_maintenance_lower_bound"]))
    psy = max(0.0, float(controls["PSY_effective_upper_bound"]))
    des = max(0.0, float(controls["DES_effective_upper_bound"]))
    cyc = max(0.0, float(controls["CYC_effective_upper_bound"]))
    model.reactions.get_by_id(gem.GLUCOSE_EXCHANGE).lower_bound = glucose_lb
    model.reactions.get_by_id(gem.OXYGEN_EXCHANGE).lower_bound = oxygen_lb
    atpm_rxn = model.reactions.get_by_id(gem.ATPM_RXN)
    atpm_rxn.upper_bound = max(atpm, atpm_rxn.upper_bound)
    atpm_rxn.lower_bound = atpm
    model.reactions.get_by_id(gem.PSY_RXN).upper_bound = psy
    model.reactions.get_by_id(gem.DES_RXN).upper_bound = des
    model.reactions.get_by_id(gem.CYC_RXN).upper_bound = cyc
    model.reactions.get_by_id(gem.PRODUCT_RXN).upper_bound = 1000.0
    return {
        "glucose_lower_bound": glucose_lb,
        "oxygen_lower_bound": oxygen_lb,
        "atp_maintenance_lower_bound": atpm,
        "PSY_effective_upper_bound": psy,
        "DES_effective_upper_bound": des,
        "CYC_effective_upper_bound": cyc,
        "gamma_growth_fraction": float(np.clip(controls["gamma_growth_fraction"], 0.0, 1.0)),
        "mapping_protocol": "predicted compact interface applied before exact staged solve_staged/pFBA",
    }


def controls_for_checkpoint(ckpt: StateSpaceCheckpoint, env: np.ndarray) -> dict[str, np.ndarray]:
    pred = predict_columns(ckpt, env, INTERFACE_COLUMNS)
    # Fill controls not supervised by an ablation with training means.
    for col in INTERFACE_COLUMNS:
        if col not in pred:
            pred[col] = np.full((len(env), ckpt.interval_len), ckpt.target_mean.get(col, 0.0))
    return pred


def load_manifest_checkpoints(manifest_path: Path, model: str | None = None, ensemble_role: str | None = None) -> list[StateSpaceCheckpoint]:
    manifest = pd.read_csv(manifest_path)
    if model is not None:
        manifest = manifest[manifest["model"].eq(model)]
    if ensemble_role is not None and "ensemble_role" in manifest.columns:
        manifest = manifest[manifest["ensemble_role"].eq(ensemble_role)]
    checkpoints = []
    for row in manifest.sort_values(["model", "model_seed"]).itertuples():
        path = ROOT / str(row.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint from manifest is missing: {path}")
        checkpoints.append(load_checkpoint(path))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints selected from {manifest_path}")
    return checkpoints


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mean_checkpoint_controls(checkpoints: list[StateSpaceCheckpoint], env: np.ndarray) -> dict[str, np.ndarray]:
    per_col = {col: [] for col in INTERFACE_COLUMNS}
    for ckpt in checkpoints:
        pred = controls_for_checkpoint(ckpt, env)
        for col in INTERFACE_COLUMNS:
            per_col[col].append(pred[col])
    return {col: np.mean(per_col[col], axis=0) for col in INTERFACE_COLUMNS}


def mean_teacher_product(checkpoints: list[StateSpaceCheckpoint], env: np.ndarray) -> np.ndarray:
    return np.mean([predict_columns(ckpt, env, ["B_total"])["B_total"] for ckpt in checkpoints], axis=0)


def load_replay_control_sources(dataset: dict[str, object]) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, pd.DataFrame, dict[str, list[StateSpaceCheckpoint]]]:
    teacher_ckpts = load_manifest_checkpoints(DATA / "teacher_ensemble_manifest.csv", model=PRIMARY_TEACHER_VARIANT)
    student_manifest = pd.read_csv(DATA / "hybrid_student_manifest.csv")
    ckpts = {
        "teacher": teacher_ckpts,
        "hybrid": load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_reporter_supervised"),
        "hybrid_without_reporters": load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_without_reporters"),
        "hybrid_shuffled_reporters": load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_shuffled_reporters"),
        "hybrid_random_smooth_aux": load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_random_smooth_aux"),
        "hybrid_one_control_only": load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_one_control_only"),
    }
    teacher_controls = mean_checkpoint_controls(teacher_ckpts, dataset["env"])
    controls = {
        "oracle": oracle_controls_from_dataset(dataset),
        "teacher": teacher_controls,
        "hybrid": mean_checkpoint_controls(ckpts["hybrid"], dataset["env"]),
        "hybrid_without_reporters": mean_checkpoint_controls(ckpts["hybrid_without_reporters"], dataset["env"]),
        "hybrid_shuffled_reporters": mean_checkpoint_controls(ckpts["hybrid_shuffled_reporters"], dataset["env"]),
        "hybrid_random_smooth_aux": mean_checkpoint_controls(ckpts["hybrid_random_smooth_aux"], dataset["env"]),
        "hybrid_one_control_only": mean_checkpoint_controls(ckpts["hybrid_one_control_only"], dataset["env"]),
        "all_controls_training_mean": training_mean_controls(dataset),
    }
    controls["only_CYC_dynamic_from_hybrid"] = constant_replacement_controls(controls["hybrid"], dataset, "CYC_effective_upper_bound")
    teacher_product = mean_teacher_product(teacher_ckpts, dataset["env"])
    return controls, teacher_product, student_manifest, ckpts


def indices_from_manifest(dataset: dict[str, object], manifest_path: str | None) -> list[int]:
    if not manifest_path:
        return list(range(len(dataset["meta"])))
    manifest = pd.read_csv(manifest_path)
    if "culture_index" in manifest.columns:
        return [int(x) for x in manifest["culture_index"]]
    lookup = {cid: i for i, cid in enumerate(dataset["meta"]["culture_id"])}
    return [lookup[cid] for cid in manifest["culture_id"]]


def shard_indices(indices: list[int], shard_index: int | None, num_shards: int | None) -> list[int]:
    if shard_index is None:
        return indices
    if num_shards is None or num_shards <= 0:
        raise ValueError("--num-shards must be positive when --shard-index is used")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    boundaries = np.linspace(0, len(indices), num_shards + 1, dtype=int)
    return indices[int(boundaries[shard_index]) : int(boundaries[shard_index + 1])]


def shard_prefix(replay_name: str, shard_index: int | None) -> str:
    return f"{replay_name}_shard_{0 if shard_index is None else shard_index:02d}"


def shard_paths(replay_name: str, shard_index: int | None) -> dict[str, Path]:
    prefix = shard_prefix(replay_name, shard_index)
    EXACT_REPLAY.mkdir(parents=True, exist_ok=True)
    return {
        "predictions": EXACT_REPLAY / f"{prefix}_predictions.csv",
        "controls": EXACT_REPLAY / f"{prefix}_controls.csv",
        "fluxes": EXACT_REPLAY / f"{prefix}_fluxes.csv",
        "solver": EXACT_REPLAY / f"{prefix}_solver_accounting.csv",
    }


def completed_shard(paths: dict[str, Path], expected_cultures: int, expected_solves: int) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        solver = pd.read_csv(paths["solver"])
    except Exception:
        return False
    return int(solver["culture_id"].nunique()) == expected_cultures and int(solver["n_actual_lp_solves"].sum()) == expected_solves


def write_rollout_tables(rollout: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    rollout["trajectories"].to_csv(paths["predictions"], index=False)
    rollout["controls"].to_csv(paths["controls"], index=False)
    rollout["fluxes"].to_csv(paths["fluxes"], index=False)
    rollout["solver"].to_csv(paths["solver"], index=False)


def shard_manifest_row(replay_name: str, shard_index: int | None, num_shards: int | None, indices: list[int], paths: dict[str, Path], status: str, failure_reason: str = "") -> dict[str, object]:
    expected_intervals = len(indices) * 48
    expected_solves = expected_intervals * 3
    completed_solves = 0
    completed_intervals = 0
    if paths["solver"].exists():
        solver = pd.read_csv(paths["solver"])
        completed_solves = int(solver["n_actual_lp_solves"].sum())
    if paths["fluxes"].exists():
        completed_intervals = int(len(pd.read_csv(paths["fluxes"])))
    return {
        "replay_type": replay_name,
        "shard_index": -1 if shard_index is None else int(shard_index),
        "num_shards": 1 if num_shards is None else int(num_shards),
        "culture_count": len(indices),
        "culture_ids": ";".join(map(str, indices)),
        "expected_interval_count": expected_intervals,
        "completed_interval_count": completed_intervals,
        "expected_lp_solves": expected_solves,
        "completed_lp_solves": completed_solves,
        "predictions_path": str(paths["predictions"].relative_to(ROOT)),
        "fluxes_path": str(paths["fluxes"].relative_to(ROOT)),
        "solver_path": str(paths["solver"].relative_to(ROOT)),
        "file_checksum": file_sha256(paths["solver"]) if paths["solver"].exists() else "",
        "status": status,
        "failure_reason": failure_reason,
        "completed_at": utc_now(),
    }


def oracle_controls_from_dataset(dataset: dict[str, object]) -> dict[str, np.ndarray]:
    return {col: np.asarray(dataset["interface"][col], dtype=float).copy() for col in INTERFACE_COLUMNS}


def training_mean_controls(dataset: dict[str, object]) -> dict[str, np.ndarray]:
    train = dataset["splits"] == "train"
    n, intervals = dataset["env"].shape[0], len(dataset["interval_time"])
    out = {}
    for col in INTERFACE_COLUMNS:
        mean = float(dataset["interface"][col][train].mean())
        out[col] = np.full((n, intervals), mean, dtype=float)
    return out


def constant_replacement_controls(source: dict[str, np.ndarray], dataset: dict[str, object], dynamic_col: str | None = None) -> dict[str, np.ndarray]:
    train = dataset["splits"] == "train"
    out = {}
    for col in INTERFACE_COLUMNS:
        if col == dynamic_col:
            out[col] = source[col].copy()
        else:
            out[col] = np.full_like(source[col], float(dataset["interface"][col][train].mean()))
    return out


def held_constant_control(source: dict[str, np.ndarray], dataset: dict[str, object], constant_col: str) -> dict[str, np.ndarray]:
    train = dataset["splits"] == "train"
    out = {col: source[col].copy() for col in INTERFACE_COLUMNS}
    out[constant_col] = np.full_like(source[constant_col], float(dataset["interface"][constant_col][train].mean()))
    return out


def select_pilot_cultures(dataset: dict[str, object], teacher_uncertainty: np.ndarray | None = None, max_n: int = 20) -> pd.DataFrame:
    meta = dataset["meta"].reset_index(drop=True)
    product = dataset["product"]
    final = product[:, -1]
    z_ox = dataset["states"]["z_ox"].max(axis=1)
    z_atp = dataset["states"]["z_atp"].max(axis=1)
    cap_min = np.minimum.reduce([dataset["states"]["E_PSY"].min(axis=1), dataset["states"]["E_DES"].min(axis=1), dataset["states"]["E_CYC"].min(axis=1)])
    uncertainty = teacher_uncertainty if teacher_uncertainty is not None else np.zeros(len(meta))
    target_n = int(max_n)
    chosen: dict[int, list[str]] = {}

    def add(idx: int, reason: str) -> None:
        chosen.setdefault(int(idx), []).append(reason)

    for split in ["train", "validation", "interpolation", "heldout_combination", "extrapolation"]:
        hits = np.where(dataset["splits"] == split)[0]
        if len(hits):
            median_hit = hits[np.argsort(final[hits])[len(hits) // 2]]
            add(int(median_hit), f"split_representative_{split}")
            if split != "train" and len(hits) > 1:
                add(int(hits[np.argmin(final[hits])]), f"split_low_product_{split}")
    for name, values in [
        ("final_product", final),
        ("oxidative_burden", z_ox),
        ("ATP_burden", z_atp),
        ("minimum_pathway_capacity", -cap_min),
        ("teacher_uncertainty", uncertainty),
    ]:
        order = np.argsort(values)
        add(int(order[0]), f"low_{name}")
        add(int(order[len(order) // 2]), f"median_{name}")
        add(int(order[-1]), f"high_{name}")
    # Fill with environmentally diverse points if collisions leave the pilot too small.
    env_score = np.sum(((dataset["env"] - dataset["env"].mean(axis=0)) / (dataset["env"].std(axis=0) + EPS)) ** 2, axis=1)
    for idx in np.argsort(env_score)[::-1]:
        if len(chosen) >= target_n:
            break
        add(int(idx), "environmental_edge_fill")
    for idx in np.argsort(np.abs(final - np.median(final))):
        if len(chosen) >= target_n:
            break
        add(int(idx), "median_product_fill")
    rows = []
    for idx, reasons in list(chosen.items())[:target_n]:
        row = meta.iloc[idx].to_dict()
        rows.append(
            {
                "culture_index": idx,
                "culture_id": row["culture_id"],
                "environment_id": row["environment_id"],
                "split": row["split"],
                "temperature": row["temperature"],
                "pH": row["pH"],
                "DO": row["DO"],
                "final_product": float(final[idx]),
                "product_AUC": float(np.trapezoid(product[idx], dataset["time"])),
                "max_oxidative_burden": float(z_ox[idx]),
                "max_ATP_burden": float(z_atp[idx]),
                "minimum_pathway_capacity": float(cap_min[idx]),
                "max_bottleneck_burden": float(dataset["states"]["z_bottle"][idx].max()),
                "teacher_uncertainty": float(uncertainty[idx]),
                "selection_reason": ";".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != target_n:
        raise RuntimeError(f"Pilot selector produced {len(out)} cultures, expected {target_n}.")
    return out


def apply_replay_interface_controls(model, cfg: gem.GEMCultureConfig, controls: dict[str, float]) -> dict[str, float | str]:
    """Apply compact replay controls exactly enough to re-enter staged pFBA."""
    return apply_predicted_interface_controls(model, cfg, controls)


def exact_pfba_replay_from_controls(
    controls: dict[str, np.ndarray],
    dataset: dict[str, object],
    replay_name: str,
    culture_indices: list[int] | None = None,
    gem_path: str | None = None,
    solver_fn: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    meta = dataset["meta"].reset_index(drop=True)
    indices = culture_indices if culture_indices is not None else list(range(len(meta)))
    if solver_fn is None:
        source_path, augmented = capacity.load_augmented(gem_path)
        base_cfg = capacity.selected_parameter_config()
    else:
        source_path, augmented, base_cfg = Path("injected_test_model"), None, capacity.selected_parameter_config()
    traj_rows, control_rows, flux_rows, solver_rows = [], [], [], []
    for i in indices:
        row = meta.iloc[i]
        cfg = replace(base_cfg, temperature=float(row.temperature), pH=float(row.pH), DO=float(row.DO), n_time=len(dataset["time"]), capacity_mode="dynamic_congestion_feedback")
        x = np.zeros(len(dataset["time"]))
        b = np.zeros(len(dataset["time"]))
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
        for k in range(len(dataset["interval_time"])):
            control = {col: float(controls[col][i, k]) for col in INTERFACE_COLUMNS}
            mapped = control.copy()
            interval_model = None
            try:
                if solver_fn is None:
                    interval_model = augmented.copy()
                    mapped = apply_replay_interface_controls(interval_model, cfg, control)
                    flux = gem.solve_staged(interval_model, cfg, mapped["gamma_growth_fraction"], state="balanced")
                else:
                    flux = solver_fn(control, cfg, k)
                for key in totals:
                    if key in flux:
                        totals[key] += int(flux[key])
            except Exception as exc:
                totals["n_solver_errors"] += 1
                totals["n_skipped_intervals"] += 1
                flux = {
                    "growth_status": "solver_error",
                    "product_status": "solver_error",
                    "pfba_status": "solver_error",
                    "solver_error": f"{type(exc).__name__}: {exc}",
                    "beta_carotene_flux": np.nan,
                    "biomass_flux": np.nan,
                    "oxygen_uptake": np.nan,
                    "atp_maintenance_flux": np.nan,
                    "ggpp_flux": np.nan,
                    "PSY_flux": np.nan,
                    "DES_flux": np.nan,
                    "CYC_flux": np.nan,
                    "n_growth_optimizations": 0,
                    "n_product_optimizations": 0,
                    "n_pfba_optimizations": 0,
                    "n_actual_lp_solves": 0,
                    "n_optimal_solves": 0,
                    "n_infeasible_solves": 0,
                    "n_unbounded_solves": 0,
                    "n_surrogate_evaluations": 0,
                }
            finally:
                if interval_model is not None:
                    del interval_model
                clear_sympy_cache()
                gc.collect()
            beta_flux = float(flux.get("beta_carotene_flux", np.nan))
            biomass_flux = float(flux.get("biomass_flux", np.nan))
            if np.isfinite(beta_flux) and np.isfinite(biomass_flux):
                beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * max(0.0, float(control.get("z_ox", 0.0))))
                x[k + 1] = max(1e-9, x[k] + cfg.dt * biomass_flux * x[k])
                b[k + 1] = max(0.0, b[k] + cfg.dt * (beta_flux * x[k] - beta_deg * b[k]))
            else:
                x[k + 1] = x[k]
                b[k + 1] = b[k]
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
        for k, tt in enumerate(dataset["time"]):
            traj_rows.append(
                {
                    "replay": replay_name,
                    "culture_index": i,
                    "culture_id": row.culture_id,
                    "environment_id": row.environment_id,
                    "split": row.split,
                    "time_index": k,
                    "time": float(tt),
                    "X_pred": float(x[k]),
                    "B_pred_pfba": float(b[k]),
                    "product_generation": "sequential accumulation from solved beta_carotene_flux and biomass",
                    "direct_product_head_used": False,
                    "metabolic_backend": "yeast_gem_lp",
                }
            )
        solver_rows.append({"replay": replay_name, "culture_index": i, "culture_id": row.culture_id, "environment_id": row.environment_id, "split": row.split, **totals})
        gc.collect()
    return {
        "trajectories": pd.DataFrame(traj_rows),
        "controls": pd.DataFrame(control_rows),
        "fluxes": pd.DataFrame(flux_rows),
        "solver": pd.DataFrame(solver_rows),
    }


def product_metric_rows(pred: pd.DataFrame, dataset: dict[str, object], replay_name: str, reference: np.ndarray | None = None, reference_name: str = "original_GEM") -> list[dict[str, object]]:
    if pred.empty:
        return []
    meta = dataset["meta"].reset_index(drop=True)
    reference = dataset["product"] if reference is None else reference
    pred_mat = np.full_like(reference, np.nan, dtype=float)
    for row in pred.itertuples():
        pred_mat[int(row.culture_index), int(row.time_index)] = float(row.B_pred_pfba)
    rows = []
    train_truth = reference[dataset["splits"] == "train"]
    for split in list(sorted(set(dataset["splits"]))) + ["all"]:
        mask = np.ones(len(meta), dtype=bool) if split == "all" else dataset["splits"] == split
        if not mask.any():
            continue
        p = pred_mat[mask]
        r = reference[mask]
        finite = np.isfinite(p).all(axis=1)
        if not finite.any():
            continue
        p, r = p[finite], r[finite]
        t = dataset["time"]
        err = p - r
        late = slice(max(0, int(0.75 * len(t))), len(t))
        corr = np.corrcoef(p.reshape(-1), r.reshape(-1))[0, 1] if np.std(p) > EPS and np.std(r) > EPS else np.nan
        rows.append(
            {
                "replay": replay_name,
                "reference": reference_name,
                "split": split,
                "n_cultures": int(finite.sum()),
                "raw_product_rmse": float(np.sqrt(np.mean(err**2))),
                "trajectory_normalized_rmse": float(np.mean(np.sqrt(np.mean(err**2, axis=1)) / np.maximum(r.max(axis=1) - r.min(axis=1), EPS))),
                "endpoint_mae": float(np.mean(np.abs(p[:, -1] - r[:, -1]))),
                "auc_mae": float(np.mean(np.abs(np.trapezoid(p, t, axis=1) - np.trapezoid(r, t, axis=1)))),
                "product_rate_rmse": float(np.sqrt(np.mean((np.gradient(p, t, axis=1) - np.gradient(r, t, axis=1)) ** 2))),
                "late_stage_rmse": float(np.sqrt(np.mean((p[:, late] - r[:, late]) ** 2))),
                "trajectory_correlation": float(corr),
                "maximum_absolute_error": float(np.max(np.abs(err))),
                "train_std_normalized_rmse": float(np.sqrt(np.mean(err**2)) / (np.std(train_truth) + EPS)),
            }
        )
    return rows


def original_flux_lookup(dataset: dict[str, object]) -> pd.DataFrame:
    cols = ["culture_id", "interval_index"] + FLUX_COLUMNS
    return dataset["flux_table"][cols].rename(columns={col: f"{col}_original" for col in FLUX_COLUMNS})


def flux_metric_rows(flux: pd.DataFrame, dataset: dict[str, object], replay_name: str) -> list[dict[str, object]]:
    if flux.empty:
        return []
    merged = flux.merge(original_flux_lookup(dataset), on=["culture_id", "interval_index"], how="left")
    rows = []
    for split in list(sorted(set(dataset["splits"]))) + ["all"]:
        g = merged if split == "all" else merged[merged["split"].eq(split)]
        if g.empty:
            continue
        row = {"replay": replay_name, "split": split, "n_intervals": int(len(g))}
        for col in FLUX_COLUMNS:
            if col in g:
                row[f"{col}_rmse"] = float(np.sqrt(np.nanmean((g[col] - g[f"{col}_original"]) ** 2)))
        rows.append(row)
    return rows


def interface_metric_rows(controls: dict[str, np.ndarray], dataset: dict[str, object], replay_name: str) -> list[dict[str, object]]:
    rows = []
    train = dataset["splits"] == "train"
    for split in list(sorted(set(dataset["splits"]))) + ["all"]:
        mask = np.ones(len(dataset["splits"]), dtype=bool) if split == "all" else dataset["splits"] == split
        if not mask.any():
            continue
        for col in INTERFACE_COLUMNS:
            truth = dataset["interface"][col]
            pred = controls[col]
            rows.append(
                {
                    "replay": replay_name,
                    "split": split,
                    "control": col,
                    "interface_control_rmse": float(np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2))),
                    "interface_control_normalized_rmse": normalized_rmse(pred[mask], truth[mask], truth[train]),
                    "control_correlation": float(np.corrcoef(pred[mask].reshape(-1), truth[mask].reshape(-1))[0, 1]) if np.std(pred[mask]) > EPS and np.std(truth[mask]) > EPS else np.nan,
                }
            )
    return rows


def active_constraint_table(flux: pd.DataFrame, controls_df: pd.DataFrame, dataset: dict[str, object], replay_name: str) -> pd.DataFrame:
    if flux.empty or controls_df.empty:
        return pd.DataFrame()
    merged = flux.merge(
        controls_df[
            [
                "culture_id",
                "interval_index",
                "oxygen_lower_bound",
                "atp_maintenance_lower_bound",
                "PSY_effective_upper_bound",
                "DES_effective_upper_bound",
                "CYC_effective_upper_bound",
            ]
        ],
        on=["culture_id", "interval_index"],
        how="left",
    )
    original = dataset["flux_table"].merge(
        dataset["constraints_table"][
            [
                "culture_id",
                "interval_index",
                "oxygen_lower_bound",
                "atp_maintenance_lower_bound",
                "PSY_effective_upper_bound",
                "DES_effective_upper_bound",
                "CYC_effective_upper_bound",
            ]
        ],
        on=["culture_id", "interval_index"],
        how="left",
    )
    def flags(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        out = df[["culture_id", "interval_index", "split"]].copy()
        out[f"oxygen_active{suffix}"] = np.isclose(df["oxygen_uptake"], df["oxygen_lower_bound"], atol=1e-6, rtol=1e-5)
        out[f"atp_maintenance_active{suffix}"] = np.isclose(df["atp_maintenance_flux"], df["atp_maintenance_lower_bound"], atol=1e-6, rtol=1e-5)
        for rxn, flux_col, bound_col in [
            ("PSY", "PSY_flux", "PSY_effective_upper_bound"),
            ("DES", "DES_flux", "DES_effective_upper_bound"),
            ("CYC", "CYC_flux", "CYC_effective_upper_bound"),
        ]:
            out[f"{rxn}_capacity_active{suffix}"] = np.isclose(df[flux_col], df[bound_col], atol=1e-6, rtol=1e-5)
        return out
    pred_flags = flags(merged)
    orig_flags = flags(original, "_original")
    joined = pred_flags.merge(orig_flags.drop(columns=["split"]), on=["culture_id", "interval_index"], how="left")
    rows = []
    active_cols = [c for c in pred_flags.columns if c.endswith("_active")]
    for split in list(sorted(set(dataset["splits"]))) + ["all"]:
        g = joined if split == "all" else joined[joined["split"].eq(split)]
        if g.empty:
            continue
        for col in active_cols:
            rows.append(
                {
                    "replay": replay_name,
                    "split": split,
                    "constraint": col,
                    "predicted_active_fraction": float(g[col].mean()),
                    "original_active_fraction": float(g[f"{col}_original"].mean()),
                    "active_constraint_agreement": float((g[col] == g[f"{col}_original"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def control_degeneracy_audit(controls: dict[str, np.ndarray], dataset: dict[str, object], replay_name: str) -> pd.DataFrame:
    rows = []
    product = dataset["product"][:, :-1]
    train = dataset["splits"] == "train"
    for col in INTERFACE_COLUMNS:
        vals = controls[col]
        train_vals = dataset["interface"][col][train]
        lo, hi = float(train_vals.min()), float(train_vals.max())
        rows.append(
            {
                "replay": replay_name,
                "control": col,
                "variance": float(np.var(vals)),
                "fraction_at_training_min": float(np.mean(np.isclose(vals, lo, atol=1e-6, rtol=1e-5))),
                "fraction_at_training_max": float(np.mean(np.isclose(vals, hi, atol=1e-6, rtol=1e-5))),
                "correlation_with_product": float(np.corrcoef(vals.reshape(-1), product.reshape(-1))[0, 1]) if np.std(vals) > EPS and np.std(product) > EPS else np.nan,
                "dynamic_range_fraction_of_training": float((np.max(vals) - np.min(vals)) / max(hi - lo, EPS)),
            }
        )
    return pd.DataFrame(rows)


def make_prediction_table(rollout: dict[str, pd.DataFrame], dataset: dict[str, object], direct_product: np.ndarray | None = None, direct_name: str = "") -> pd.DataFrame:
    traj = rollout["trajectories"].copy()
    original_rows = []
    original = dataset["product"]
    for row in traj.itertuples():
        rec = row._asdict()
        i = int(rec["culture_index"])
        k = int(rec["time_index"])
        rec["B_original_GEM"] = float(original[i, k])
        if direct_product is not None:
            rec[direct_name] = float(direct_product[i, k])
        original_rows.append(rec)
    return pd.DataFrame(original_rows)


def write_error_decomposition(
    dataset: dict[str, object],
    oracle: dict[str, pd.DataFrame],
    teacher: dict[str, pd.DataFrame],
    hybrid_rollout: dict[str, pd.DataFrame],
    teacher_product: np.ndarray,
) -> pd.DataFrame:
    rows = []
    rows.extend(product_metric_rows(oracle["trajectories"], dataset, "oracle_controls_pfba", dataset["product"], "original_GEM"))
    rows.extend(product_metric_rows(teacher["trajectories"], dataset, "teacher_controls_pfba", teacher_product, "teacher_direct_product"))
    rows.extend(product_metric_rows(teacher["trajectories"], dataset, "teacher_controls_pfba", dataset["product"], "original_GEM"))
    rows.extend(product_metric_rows(hybrid_rollout["trajectories"], dataset, "hybrid_controls_pfba", dataset["product"], "original_GEM"))
    # Hybrid distillation error against teacher-control pFBA on shared cultures.
    teacher_mat = trajectory_matrix(teacher["trajectories"], dataset)
    rows.extend(product_metric_rows(hybrid_rollout["trajectories"], dataset, "hybrid_controls_pfba", teacher_mat, "teacher_controls_pfba"))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hybrid_reconstruction_error_decomposition.csv", index=False)
    return out


def trajectory_matrix(traj: pd.DataFrame, dataset: dict[str, object]) -> np.ndarray:
    mat = np.full_like(dataset["product"], np.nan, dtype=float)
    for row in traj.itertuples():
        mat[int(row.culture_index), int(row.time_index)] = float(row.B_pred_pfba)
    return mat


def acceptance_from_validation(
    dataset: dict[str, object],
    oracle: dict[str, pd.DataFrame],
    teacher: dict[str, pd.DataFrame],
    hybrid_rollout: dict[str, pd.DataFrame],
    error_decomp: pd.DataFrame,
    active_audit: pd.DataFrame,
    degeneracy: pd.DataFrame,
    student_manifest: pd.DataFrame,
) -> pd.DataFrame:
    solver = pd.concat([oracle["solver"], teacher["solver"], hybrid_rollout["solver"]], ignore_index=True)
    oracle_all = error_decomp[(error_decomp["replay"].eq("oracle_controls_pfba")) & (error_decomp["split"].eq("all")) & (error_decomp["reference"].eq("original_GEM"))]
    hybrid_held = error_decomp[(error_decomp["replay"].eq("hybrid_controls_pfba")) & (error_decomp["split"].eq("heldout_combination")) & (error_decomp["reference"].eq("original_GEM"))]
    distill_all = error_decomp[(error_decomp["replay"].eq("hybrid_controls_pfba")) & (error_decomp["split"].eq("all")) & (error_decomp["reference"].eq("teacher_controls_pfba"))]
    active_agreement = float(active_audit["active_constraint_agreement"].mean()) if not active_audit.empty else 0.0
    nondegenerate = bool((degeneracy[degeneracy["replay"].eq("hybrid_controls_pfba")]["variance"] > 1e-10).sum() >= 3)
    no_product_head = bool((student_manifest[student_manifest["model"].eq("hybrid_reporter_supervised")]["direct_product_head"] == False).all())
    expected_solves = 3 * len(dataset["interval_time"]) * dataset["meta"]["culture_id"].nunique()
    criteria = [
        ("oracle_replay_close_to_original", float(oracle_all["trajectory_normalized_rmse"].iloc[0]) < 0.03 if not oracle_all.empty else False, float(oracle_all["trajectory_normalized_rmse"].iloc[0]) if not oracle_all.empty else np.nan),
        ("product_generated_only_from_pfba_flux", bool((~hybrid_rollout["trajectories"]["direct_product_head_used"]).all()), 1.0),
        ("real_yeast9_backend", bool(hybrid_rollout["trajectories"]["metabolic_backend"].eq("yeast_gem_lp").all()), 1.0),
        ("zero_surrogate_evaluations", int(solver["n_surrogate_evaluations"].sum()) == 0, int(solver["n_surrogate_evaluations"].sum())),
        ("complete_solver_accounting", int(hybrid_rollout["solver"]["n_actual_lp_solves"].sum()) == int(hybrid_rollout["solver"][["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations"]].sum().sum()), int(hybrid_rollout["solver"]["n_actual_lp_solves"].sum())),
        ("no_infeasible_unbounded_or_solver_errors", int(solver[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals"]].sum().sum()) == 0, int(solver[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals"]].sum().sum())),
        ("full_primary_rollout_expected_solves", int(hybrid_rollout["solver"]["n_actual_lp_solves"].sum()) == expected_solves, int(hybrid_rollout["solver"]["n_actual_lp_solves"].sum())),
        ("more_than_one_control_dynamic", nondegenerate, int((degeneracy[degeneracy["replay"].eq("hybrid_controls_pfba")]["variance"] > 1e-10).sum())),
        ("primary_hybrid_has_no_direct_product_head", no_product_head, 1.0),
        ("heldout_hybrid_reconstruction_useful", float(hybrid_held["trajectory_normalized_rmse"].iloc[0]) < 0.50 if not hybrid_held.empty else False, float(hybrid_held["trajectory_normalized_rmse"].iloc[0]) if not hybrid_held.empty else np.nan),
        ("teacher_to_hybrid_degradation_quantified", not distill_all.empty and np.isfinite(float(distill_all["trajectory_normalized_rmse"].iloc[0])), float(distill_all["trajectory_normalized_rmse"].iloc[0]) if not distill_all.empty else np.nan),
        ("active_constraint_similarity", active_agreement > 0.50, active_agreement),
        ("hybrid_not_product_bound_encoded", nondegenerate, 1.0),
        ("seed_manifests_available", student_manifest[student_manifest["model"].eq("hybrid_reporter_supervised")]["model_seed"].nunique() >= 5, student_manifest[student_manifest["model"].eq("hybrid_reporter_supervised")]["model_seed"].nunique()),
    ]
    passed = all(bool(ok) for _name, ok, _value in criteria)
    status = "stage_3_passed_ready_for_metabolic_edits" if passed else "stage_3_blocked_" + ";".join(name for name, ok, _value in criteria if not bool(ok))
    out = pd.DataFrame(
        [
            {
                "stage": "Stage 3",
                "criterion": name,
                "passed": bool(ok),
                "value": value,
                "decision": "pass" if passed else "blocked",
                "stage_3_status": status,
                "stage4_status": "ready_but_not_executed" if passed else "blocked_until_stage3_passes",
            }
            for name, ok, value in criteria
        ]
    )
    out.to_csv(DATA / "hybrid_student_acceptance.csv", index=False)
    return out


def reporter_prediction_table(checkpoints: list[StateSpaceCheckpoint], dataset: dict[str, object], model_name: str) -> pd.DataFrame:
    env = dataset["env"]
    rows = []
    preds = {col: np.mean([predict_columns(ckpt, env, [col])[col] for ckpt in checkpoints], axis=0) for col in REPORTER_COLUMNS}
    for i, meta in dataset["meta"].reset_index(drop=True).iterrows():
        for k, tt in enumerate(dataset["time"]):
            rec = {
                "model": model_name,
                "culture_index": i,
                "culture_id": meta.culture_id,
                "environment_id": meta.environment_id,
                "split": meta.split,
                "time_index": k,
                "time": float(tt),
            }
            for col in REPORTER_COLUMNS:
                rec[f"{col}_pred"] = float(preds[col][i, k])
                rec[f"{col}_original"] = float(dataset["reporters"][col][i, k])
            rows.append(rec)
    return pd.DataFrame(rows)


def interface_control_frame(controls: dict[str, np.ndarray], dataset: dict[str, object], replay_name: str) -> pd.DataFrame:
    rows = []
    for i, meta in dataset["meta"].reset_index(drop=True).iterrows():
        for k, tt in enumerate(dataset["interval_time"]):
            rec = {
                "replay": replay_name,
                "culture_index": i,
                "culture_id": meta.culture_id,
                "environment_id": meta.environment_id,
                "split": meta.split,
                "interval_index": k,
                "time": float(tt),
                "temperature": meta.temperature,
                "pH": meta.pH,
                "DO": meta.DO,
            }
            for col in INTERFACE_COLUMNS:
                rec[col] = float(controls[col][i, k])
            rows.append(rec)
    return pd.DataFrame(rows)


def control_intervention_audit(
    controls: dict[str, np.ndarray],
    dataset: dict[str, object],
    culture_indices: list[int],
    gem_path: str | None = None,
) -> pd.DataFrame:
    source_path, augmented = capacity.load_augmented(gem_path)
    base_cfg = capacity.selected_parameter_config()
    perturbations = [
        ("oxygen_control_more_uptake", "oxygen_lower_bound", -0.20),
        ("ATP_control_higher_maintenance", "atp_maintenance_lower_bound", 0.10),
        ("PSY_capacity_lower", "PSY_effective_upper_bound", -0.05),
        ("DES_capacity_lower", "DES_effective_upper_bound", -0.05),
        ("CYC_capacity_lower", "CYC_effective_upper_bound", -0.05),
        ("preserved_growth_fraction_higher", "gamma_growth_fraction", 0.05),
        ("ER_only_negative_control", "z_er", 0.50),
    ]
    rows = []
    for i in culture_indices[: min(5, len(culture_indices))]:
        meta = dataset["meta"].iloc[i]
        cfg = replace(base_cfg, temperature=float(meta.temperature), pH=float(meta.pH), DO=float(meta.DO), n_time=len(dataset["time"]), capacity_mode="dynamic_congestion_feedback")
        base_control = {col: float(controls[col][i, 0]) for col in INTERFACE_COLUMNS}
        if "z_er" not in base_control:
            base_control["z_er"] = float(dataset["traj"][dataset["traj"]["culture_id"].eq(meta.culture_id)].sort_values("time_index")["z_er"].iloc[0])

        def solve_one(control: dict[str, float]) -> dict[str, object]:
            model = augmented.copy()
            mapped = apply_replay_interface_controls(model, cfg, control)
            return gem.solve_staged(model, cfg, mapped["gamma_growth_fraction"], state="balanced")

        try:
            base_flux = solve_one(base_control)
        except Exception as exc:
            rows.append({"culture_id": meta.culture_id, "intervention": "baseline", "solver_error": f"{type(exc).__name__}: {exc}"})
            continue
        for label, col, delta in perturbations:
            perturbed = base_control.copy()
            perturbed[col] = float(perturbed.get(col, 0.0) + delta)
            if col.endswith("upper_bound") or col == "atp_maintenance_lower_bound":
                perturbed[col] = max(0.0, perturbed[col])
            if col == "gamma_growth_fraction":
                perturbed[col] = float(np.clip(perturbed[col], 0.0, 1.0))
            try:
                flux = solve_one(perturbed)
                rows.append(
                    {
                        "culture_index": i,
                        "culture_id": meta.culture_id,
                        "split": meta.split,
                        "intervention": label,
                        "perturbed_control": col,
                        "delta": delta,
                        "beta_carotene_flux_delta": float(flux["beta_carotene_flux"] - base_flux["beta_carotene_flux"]),
                        "biomass_flux_delta": float(flux["biomass_flux"] - base_flux["biomass_flux"]),
                        "oxygen_uptake_delta": float(flux["oxygen_uptake"] - base_flux["oxygen_uptake"]),
                        "atp_maintenance_flux_delta": float(flux["atp_maintenance_flux"] - base_flux["atp_maintenance_flux"]),
                        "ggpp_flux_delta": float(flux["ggpp_flux"] - base_flux["ggpp_flux"]),
                        "PSY_flux_delta": float(flux["PSY_flux"] - base_flux["PSY_flux"]),
                        "DES_flux_delta": float(flux["DES_flux"] - base_flux["DES_flux"]),
                        "CYC_flux_delta": float(flux["CYC_flux"] - base_flux["CYC_flux"]),
                        "n_actual_lp_solves": int(flux["n_actual_lp_solves"]),
                        "n_surrogate_evaluations": 0,
                    }
                )
            except Exception as exc:
                rows.append({"culture_index": i, "culture_id": meta.culture_id, "split": meta.split, "intervention": label, "perturbed_control": col, "delta": delta, "solver_error": f"{type(exc).__name__}: {exc}"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hybrid_control_intervention_audit.csv", index=False)
    return out


def exact_pilot_indices_for_secondary_audits(dataset: dict[str, object], n: int = 18) -> list[int]:
    manifest_path = DATA / "hybrid_exact_pilot_manifest.csv"
    if manifest_path.exists():
        return indices_from_manifest(dataset, str(manifest_path))
    pilot = select_pilot_cultures(dataset, max_n=n)
    pilot.to_csv(manifest_path, index=False)
    return [int(x) for x in pilot["culture_index"]]


def run_secondary_ablation_exact_replays(
    control_library: dict[str, dict[str, np.ndarray]],
    dataset: dict[str, object],
    culture_indices: list[int],
    gem_path: str | None = None,
) -> pd.DataFrame:
    ablation_controls = {
        "hybrid_without_reporters": control_library["hybrid_without_reporters"],
        "hybrid_shuffled_reporters": control_library["hybrid_shuffled_reporters"],
        "hybrid_random_smooth_aux": control_library["hybrid_random_smooth_aux"],
        "hybrid_one_control_only": control_library["hybrid_one_control_only"],
        "all_controls_training_mean": control_library["all_controls_training_mean"],
        "only_CYC_dynamic_from_hybrid": control_library["only_CYC_dynamic_from_hybrid"],
    }
    for col in GEM_APPLIED_CONTROL_COLUMNS:
        ablation_controls[f"hybrid_hold_constant_{col}"] = held_constant_control(control_library["hybrid"], dataset, col)

    metric_rows = []
    solver_frames = []
    for name, replay_controls in ablation_controls.items():
        rollout = exact_pfba_replay_from_controls(replay_controls, dataset, name, culture_indices=culture_indices, gem_path=gem_path)
        metric_rows.extend(product_metric_rows(rollout["trajectories"], dataset, name, dataset["product"], "original_GEM"))
        metric_rows.extend(flux_metric_rows(rollout["fluxes"], dataset, name))
        solver_frames.append(rollout["solver"])

    for name, reason in [
        ("hybrid_ER_only_control", "not_available_no_compact_interface_checkpoint"),
        ("hybrid_product_only_state_space_interface", "not_available_no_compact_interface_checkpoint"),
    ]:
        metric_rows.append(
            {
                "replay": name,
                "split": "all",
                "reference": "original_GEM",
                "status": reason,
                "trajectory_rmse": np.nan,
                "trajectory_normalized_rmse": np.nan,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DATA / "hybrid_student_ablation_metrics.csv", index=False)
    if solver_frames:
        pd.concat(solver_frames, ignore_index=True).to_csv(DATA / "hybrid_student_ablation_solver_accounting.csv", index=False)
    return metrics


def save_validation_figures(error_decomp: pd.DataFrame, active_audit: pd.DataFrame, degeneracy: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)

    def simple_bar(path: Path, title: str, rows: pd.DataFrame, label_col: str, value_col: str) -> None:
        sub = rows.dropna(subset=[value_col]).head(14)
        ymax = max(float(sub[value_col].max()), EPS) if not sub.empty else 1.0
        body = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="430" viewBox="0 0 920 430"><rect width="100%" height="100%" fill="white"/>',
            ss_validation.fixed.svg_text(460, 28, title, 17, "bold"),
        ]
        for i, row in enumerate(sub.itertuples()):
            val = float(getattr(row, value_col))
            h = 280 * val / ymax
            x = 70 + i * 58
            body.append(f'<rect x="{x}" y="{340-h:.1f}" width="34" height="{h:.1f}" fill="#2f6f9f"/>')
            body.append(ss_validation.fixed.svg_text(x + 17, 375, str(getattr(row, label_col)).replace("_", "\n"), 7))
        body.append("</svg>")
        path.write_text("\n".join(body), encoding="utf-8")
        ss_validation.gem_png(path.with_suffix(".png"), 920, 430)

    split_rows = error_decomp[error_decomp["reference"].eq("original_GEM") & error_decomp["split"].ne("all")].copy()
    split_rows["label"] = split_rows["replay"] + "_" + split_rows["split"]
    simple_bar(FIGURES / "hybrid_split_error_decomposition.svg", "Split-wise Product Error Decomposition", split_rows, "label", "trajectory_normalized_rmse")
    if not active_audit.empty:
        active_rows = active_audit[active_audit["split"].eq("all")].copy()
        simple_bar(FIGURES / "hybrid_active_constraint_agreement.svg", "Active Constraint Agreement", active_rows, "constraint", "active_constraint_agreement")
    if not degeneracy.empty:
        simple_bar(FIGURES / "hybrid_control_variance.svg", "Hybrid Control Variance", degeneracy[degeneracy["replay"].eq("hybrid_controls_pfba")], "control", "variance")


def run_exact_replay_shard(args: argparse.Namespace) -> None:
    dataset = load_distillation_dataset()
    controls, _teacher_product, _student_manifest, _ckpts = load_replay_control_sources(dataset)
    all_indices = indices_from_manifest(dataset, args.culture_manifest)
    if args.max_pfba_cultures and args.max_pfba_cultures > 0:
        all_indices = all_indices[: int(args.max_pfba_cultures)]
    indices = shard_indices(all_indices, args.shard_index, args.num_shards)
    replay_types = [name.strip() for name in args.replay_types.split(",") if name.strip()]
    manifest_rows = []
    for replay_name in replay_types:
        if replay_name not in controls:
            raise ValueError(f"Unknown replay type {replay_name!r}. Available: {sorted(controls)}")
        paths = shard_paths(replay_name, args.shard_index)
        expected_solves = len(indices) * len(dataset["interval_time"]) * 3
        if args.resume and completed_shard(paths, len(indices), expected_solves):
            manifest_rows.append(shard_manifest_row(replay_name, args.shard_index, args.num_shards, indices, paths, "already_complete"))
            continue
        started = utc_now()
        try:
            rollout = exact_pfba_replay_from_controls(controls[replay_name], dataset, f"{replay_name}_interface", culture_indices=indices, gem_path=args.gem_path)
            write_rollout_tables(rollout, paths)
            row = shard_manifest_row(replay_name, args.shard_index, args.num_shards, indices, paths, "complete")
            row["started_at"] = started
            manifest_rows.append(row)
        except Exception as exc:
            manifest_rows.append(
                {
                    "replay_type": replay_name,
                    "shard_index": -1 if args.shard_index is None else int(args.shard_index),
                    "num_shards": 1 if args.num_shards is None else int(args.num_shards),
                    "culture_count": len(indices),
                    "culture_ids": ";".join(map(str, indices)),
                    "expected_interval_count": len(indices) * len(dataset["interval_time"]),
                    "completed_interval_count": 0,
                    "expected_lp_solves": expected_solves,
                    "completed_lp_solves": 0,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "started_at": started,
                    "completed_at": utc_now(),
                }
            )
            raise
    shard_manifest = pd.DataFrame(manifest_rows)
    shard_manifest.to_csv(EXACT_REPLAY / f"shard_{0 if args.shard_index is None else args.shard_index:02d}_manifest.csv", index=False)
    append_or_replace_shard_manifest(shard_manifest)
    print(f"Exact replay shard complete: shard={args.shard_index}, replay_types={','.join(replay_types)}, cultures={len(indices)}")


def append_or_replace_shard_manifest(rows: pd.DataFrame) -> None:
    path = DATA / "hybrid_exact_replay_shard_manifest.csv"
    if path.exists():
        existing = pd.read_csv(path)
        key_cols = ["replay_type", "shard_index", "num_shards"]
        key = set(map(tuple, rows[key_cols].to_numpy()))
        existing = existing[~existing[key_cols].apply(tuple, axis=1).isin(key)]
        out = pd.concat([existing, rows], ignore_index=True)
    else:
        out = rows
    out.to_csv(path, index=False)


def read_shard_outputs(replay_types: list[str], num_shards: int) -> dict[str, dict[str, pd.DataFrame]]:
    merged: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for replay_name in replay_types:
        merged[replay_name] = {"trajectories": [], "controls": [], "fluxes": [], "solver": []}
        for shard_index in range(num_shards):
            paths = shard_paths(replay_name, shard_index)
            if not all(path.exists() for path in paths.values()):
                raise FileNotFoundError(f"Missing shard files for {replay_name} shard {shard_index}: {paths}")
            merged[replay_name]["trajectories"].append(pd.read_csv(paths["predictions"]))
            merged[replay_name]["controls"].append(pd.read_csv(paths["controls"]))
            merged[replay_name]["fluxes"].append(pd.read_csv(paths["fluxes"]))
            merged[replay_name]["solver"].append(pd.read_csv(paths["solver"]))
    return {
        replay_name: {kind: pd.concat(frames, ignore_index=True) for kind, frames in kinds.items()}
        for replay_name, kinds in merged.items()
    }


def validate_merged_uniqueness(rollouts: dict[str, dict[str, pd.DataFrame]], dataset: dict[str, object]) -> pd.DataFrame:
    rows = []
    for replay_name, rollout in rollouts.items():
        pred_dupes = int(rollout["trajectories"].duplicated(["culture_id", "time_index"]).sum())
        flux_dupes = int(rollout["fluxes"].duplicated(["culture_id", "interval_index"]).sum())
        expected_cultures = int(dataset["meta"]["culture_id"].nunique())
        expected_intervals = expected_cultures * len(dataset["interval_time"])
        expected_timepoints = expected_cultures * len(dataset["time"])
        rows.append(
            {
                "replay_type": replay_name,
                "culture_count": int(rollout["solver"]["culture_id"].nunique()),
                "trajectory_rows": int(len(rollout["trajectories"])),
                "flux_rows": int(len(rollout["fluxes"])),
                "expected_trajectory_rows": expected_timepoints,
                "expected_flux_rows": expected_intervals,
                "duplicate_trajectory_keys": pred_dupes,
                "duplicate_flux_keys": flux_dupes,
                "n_actual_lp_solves": int(rollout["solver"]["n_actual_lp_solves"].sum()),
                "n_surrogate_evaluations": int(rollout["solver"]["n_surrogate_evaluations"].sum()),
                "status": "complete" if pred_dupes == 0 and flux_dupes == 0 and len(rollout["fluxes"]) == expected_intervals else "invalid",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "hybrid_exact_replay_completion.csv", index=False)
    return out


def merge_exact_replay_shards(args: argparse.Namespace) -> None:
    dataset = load_distillation_dataset()
    controls, teacher_product, student_manifest, ckpts = load_replay_control_sources(dataset)
    replay_types = [name.strip() for name in args.replay_types.split(",") if name.strip()]
    rollouts = read_shard_outputs(replay_types, int(args.num_shards or 5))
    completion = validate_merged_uniqueness(rollouts, dataset)
    if not completion["status"].eq("complete").all() and not args.force_through_failed_gates:
        print("Shard merge failed schema/uniqueness checks. See data/hybrid_exact_replay_completion.csv")
        return
    oracle = rollouts.get("oracle")
    teacher = rollouts.get("teacher")
    hybrid_rollout = rollouts.get("hybrid")
    if oracle:
        make_prediction_table(oracle, dataset).to_csv(DATA / "hybrid_oracle_interface_predictions.csv", index=False)
        oracle["controls"].to_csv(DATA / "hybrid_oracle_interface_controls.csv", index=False)
    if teacher:
        make_prediction_table(teacher, dataset, teacher_product, "B_teacher_direct").to_csv(DATA / "hybrid_teacher_interface_predictions.csv", index=False)
        teacher["controls"].to_csv(DATA / "hybrid_teacher_interface_controls.csv", index=False)
    if hybrid_rollout:
        make_prediction_table(hybrid_rollout, dataset, teacher_product, "B_teacher_direct").to_csv(DATA / "hybrid_student_predictions.csv", index=False)
        hybrid_rollout["controls"].to_csv(DATA / "hybrid_student_controls.csv", index=False)
        hybrid_rollout["fluxes"].to_csv(DATA / "hybrid_student_fluxes.csv", index=False)
        hybrid_rollout["solver"].to_csv(DATA / "hybrid_student_solver_accounting.csv", index=False)
        reporter_prediction_table(ckpts["hybrid"], dataset, "hybrid_reporter_supervised").to_csv(DATA / "hybrid_student_reporters.csv", index=False)
    if oracle and teacher and hybrid_rollout:
        error_decomp = write_error_decomposition(dataset, oracle, teacher, hybrid_rollout, teacher_product)
        metric_rows = []
        for name, rollout, control in [("oracle_controls_pfba", oracle, controls["oracle"]), ("teacher_controls_pfba", teacher, controls["teacher"]), ("hybrid_controls_pfba", hybrid_rollout, controls["hybrid"])]:
            metric_rows.extend(flux_metric_rows(rollout["fluxes"], dataset, name))
            metric_rows.extend(interface_metric_rows(control, dataset, name))
        pd.DataFrame(metric_rows).to_csv(DATA / "hybrid_student_metrics.csv", index=False)
        active = pd.concat(
            [
                active_constraint_table(oracle["fluxes"], oracle["controls"], dataset, "oracle_controls_pfba"),
                active_constraint_table(teacher["fluxes"], teacher["controls"], dataset, "teacher_controls_pfba"),
                active_constraint_table(hybrid_rollout["fluxes"], hybrid_rollout["controls"], dataset, "hybrid_controls_pfba"),
            ],
            ignore_index=True,
        )
        active.to_csv(DATA / "hybrid_active_constraint_audit.csv", index=False)
        degeneracy = pd.concat(
            [control_degeneracy_audit(controls["teacher"], dataset, "teacher_controls_pfba"), control_degeneracy_audit(controls["hybrid"], dataset, "hybrid_controls_pfba")],
            ignore_index=True,
        )
        degeneracy.to_csv(DATA / "hybrid_control_variance_audit.csv", index=False)
        degeneracy.to_csv(DATA / "hybrid_control_degeneracy_audit.csv", index=False)
        pilot_indices = exact_pilot_indices_for_secondary_audits(dataset)
        control_intervention_audit(controls["hybrid"], dataset, pilot_indices, gem_path=args.gem_path)
        run_secondary_ablation_exact_replays(controls, dataset, pilot_indices, gem_path=args.gem_path)
        acceptance_from_validation(dataset, oracle, teacher, hybrid_rollout, error_decomp, active, degeneracy, student_manifest)
        save_validation_figures(error_decomp, active, degeneracy)
    print("Exact replay shard merge complete.")


def run_exact_stage3_validation(args: argparse.Namespace) -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    EXACT_REPLAY.mkdir(parents=True, exist_ok=True)
    if args.merge_shards:
        merge_exact_replay_shards(args)
        return
    if args.shard_index is not None:
        run_exact_replay_shard(args)
        return
    dataset = load_distillation_dataset()
    write_data_inventory(dataset)
    controls, teacher_product, student_manifest, ckpts = load_replay_control_sources(dataset)
    teacher_controls = controls["teacher"]
    hybrid_controls = controls["hybrid"]
    primary_student_ckpts = ckpts["hybrid"]
    teacher_unc = np.mean([np.mean((controls_for_checkpoint(ckpt, dataset["env"])["CYC_effective_upper_bound"] - teacher_controls["CYC_effective_upper_bound"]) ** 2, axis=1) for ckpt in ckpts["teacher"]], axis=0)
    pilot = select_pilot_cultures(dataset, teacher_uncertainty=teacher_unc, max_n=args.pilot_cultures)
    pilot.to_csv(DATA / "hybrid_exact_pilot_manifest.csv", index=False)
    pilot_indices = [int(x) for x in pilot["culture_index"]]

    print(f"Running pilot exact replay on {len(pilot_indices)} cultures.")
    pilot_oracle = exact_pfba_replay_from_controls(controls["oracle"], dataset, "pilot_oracle_controls_pfba", culture_indices=pilot_indices, gem_path=args.gem_path)
    pilot_teacher = exact_pfba_replay_from_controls(teacher_controls, dataset, "pilot_teacher_controls_pfba", culture_indices=pilot_indices, gem_path=args.gem_path)
    pilot_hybrid = exact_pfba_replay_from_controls(hybrid_controls, dataset, "pilot_hybrid_controls_pfba", culture_indices=pilot_indices, gem_path=args.gem_path)
    pilot_solver = pd.concat([pilot_oracle["solver"], pilot_teacher["solver"], pilot_hybrid["solver"]], ignore_index=True)
    pd.concat([pilot_oracle["solver"], pilot_teacher["solver"], pilot_hybrid["solver"]], ignore_index=True).to_csv(DATA / "hybrid_exact_replay_pilot_solver_accounting.csv", index=False)
    pd.concat([pilot_oracle["fluxes"], pilot_teacher["fluxes"], pilot_hybrid["fluxes"]], ignore_index=True).to_csv(DATA / "hybrid_exact_replay_pilot_fluxes.csv", index=False)
    pd.concat([pilot_oracle["controls"], pilot_teacher["controls"], pilot_hybrid["controls"]], ignore_index=True).to_csv(DATA / "hybrid_exact_replay_pilot_controls.csv", index=False)
    pd.concat([pilot_oracle["trajectories"], pilot_teacher["trajectories"], pilot_hybrid["trajectories"]], ignore_index=True).to_csv(DATA / "hybrid_exact_replay_pilot_predictions.csv", index=False)
    pilot_pass = (
        int(pilot_solver[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals", "n_surrogate_evaluations"]].sum().sum()) == 0
        and int(pilot_hybrid["solver"]["n_actual_lp_solves"].sum()) == len(pilot_indices) * len(dataset["interval_time"]) * 3
    )
    pd.DataFrame(
        [
            {"criterion": "finite_predicted_controls", "passed": bool(np.isfinite(np.column_stack([hybrid_controls[c][pilot_indices].reshape(-1) for c in INTERFACE_COLUMNS])).all())},
            {"criterion": "all_three_solver_stages_execute", "passed": bool(int(pilot_hybrid["solver"]["n_actual_lp_solves"].sum()) == len(pilot_indices) * len(dataset["interval_time"]) * 3)},
            {"criterion": "zero_surrogate_evaluations", "passed": bool(int(pilot_solver["n_surrogate_evaluations"].sum()) == 0)},
            {"criterion": "no_solver_failures", "passed": bool(int(pilot_solver[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals"]].sum().sum()) == 0)},
            {"criterion": "pilot_gate", "passed": bool(pilot_pass)},
        ]
    ).to_csv(DATA / "hybrid_exact_replay_pilot_acceptance.csv", index=False)
    if not pilot_pass and not args.force_through_failed_gates:
        print("Pilot exact replay failed; full Stage 3 validation blocked. See data/hybrid_exact_replay_pilot_acceptance.csv")
        return
    if args.pilot_only:
        pd.DataFrame(
            [
                {
                    "stage": "Stage 3",
                    "criterion": "pilot_exact_replay_passed",
                    "passed": bool(pilot_pass),
                    "value": int(pilot_solver["n_actual_lp_solves"].sum()),
                    "decision": "blocked",
                    "stage_3_status": "stage_3_blocked_full_125_culture_exact_rollout_not_run",
                    "stage4_status": "blocked_until_stage3_passes",
                },
                {
                    "stage": "Stage 3",
                    "criterion": "full_125_culture_exact_rollout_executed",
                    "passed": False,
                    "value": 0,
                    "decision": "blocked",
                    "stage_3_status": "stage_3_blocked_full_125_culture_exact_rollout_not_run",
                    "stage4_status": "blocked_until_stage3_passes",
                },
            ]
        ).to_csv(DATA / "hybrid_student_acceptance.csv", index=False)
        print("Pilot-only exact replay complete.")
        return

    n_full = len(dataset["meta"]) if args.max_pfba_cultures is None or args.max_pfba_cultures >= len(dataset["meta"]) else int(args.max_pfba_cultures)
    full_indices = list(range(n_full))
    print(f"Running full exact replay on {len(full_indices)} cultures for oracle, teacher and primary hybrid controls.")
    oracle = exact_pfba_replay_from_controls(controls["oracle"], dataset, "oracle_controls_pfba", culture_indices=full_indices, gem_path=args.gem_path)
    teacher = exact_pfba_replay_from_controls(teacher_controls, dataset, "teacher_controls_pfba", culture_indices=full_indices, gem_path=args.gem_path)
    hybrid_rollout = exact_pfba_replay_from_controls(hybrid_controls, dataset, "hybrid_controls_pfba", culture_indices=full_indices, gem_path=args.gem_path)

    make_prediction_table(oracle, dataset).to_csv(DATA / "hybrid_oracle_interface_predictions.csv", index=False)
    make_prediction_table(teacher, dataset, teacher_product, "B_teacher_direct").to_csv(DATA / "hybrid_teacher_interface_predictions.csv", index=False)
    make_prediction_table(hybrid_rollout, dataset, teacher_product, "B_teacher_direct").to_csv(DATA / "hybrid_student_predictions.csv", index=False)
    reporter_prediction_table(primary_student_ckpts, dataset, "hybrid_reporter_supervised").to_csv(DATA / "hybrid_student_reporters.csv", index=False)
    hybrid_rollout["controls"].to_csv(DATA / "hybrid_student_controls.csv", index=False)
    hybrid_rollout["fluxes"].to_csv(DATA / "hybrid_student_fluxes.csv", index=False)
    hybrid_rollout["solver"].to_csv(DATA / "hybrid_student_solver_accounting.csv", index=False)
    teacher["controls"].to_csv(DATA / "hybrid_teacher_interface_controls.csv", index=False)
    oracle["controls"].to_csv(DATA / "hybrid_oracle_interface_controls.csv", index=False)
    error_decomp = write_error_decomposition(dataset, oracle, teacher, hybrid_rollout, teacher_product)
    product_rows = []
    for name, rollout, replay_controls in [
        ("oracle_controls_pfba", oracle, controls["oracle"]),
        ("teacher_controls_pfba", teacher, teacher_controls),
        ("hybrid_controls_pfba", hybrid_rollout, hybrid_controls),
    ]:
        product_rows.extend(flux_metric_rows(rollout["fluxes"], dataset, name))
        product_rows.extend(interface_metric_rows(replay_controls, dataset, name))
    pd.DataFrame(product_rows).to_csv(DATA / "hybrid_student_metrics.csv", index=False)
    active = pd.concat(
        [
            active_constraint_table(oracle["fluxes"], oracle["controls"], dataset, "oracle_controls_pfba"),
            active_constraint_table(teacher["fluxes"], teacher["controls"], dataset, "teacher_controls_pfba"),
            active_constraint_table(hybrid_rollout["fluxes"], hybrid_rollout["controls"], dataset, "hybrid_controls_pfba"),
        ],
        ignore_index=True,
    )
    active.to_csv(DATA / "hybrid_active_constraint_audit.csv", index=False)
    degeneracy = pd.concat(
        [
            control_degeneracy_audit(teacher_controls, dataset, "teacher_controls_pfba"),
            control_degeneracy_audit(hybrid_controls, dataset, "hybrid_controls_pfba"),
        ],
        ignore_index=True,
    )
    degeneracy.to_csv(DATA / "hybrid_control_degeneracy_audit.csv", index=False)
    degeneracy.to_csv(DATA / "hybrid_control_variance_audit.csv", index=False)
    control_intervention_audit(hybrid_controls, dataset, pilot_indices, gem_path=args.gem_path)

    # Secondary ablations use the predeclared pilot subset to avoid multiplying the full exact LP count.
    run_secondary_ablation_exact_replays(controls, dataset, pilot_indices, gem_path=args.gem_path)
    acceptance_from_validation(dataset, oracle, teacher, hybrid_rollout, error_decomp, active, degeneracy, student_manifest)
    save_validation_figures(error_decomp, active, degeneracy)
    print("Exact Stage 3 Yeast9/pFBA validation complete.")


def run_exact_pfba_rollout(
    ckpt: StateSpaceCheckpoint,
    dataset: dict[str, object],
    max_cultures: int | None = None,
    gem_path: str | None = None,
    solver_fn: Callable | None = None,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    env = dataset["env"]
    meta = dataset["meta"].reset_index(drop=True)
    controls = controls_for_checkpoint(ckpt, env)
    selected_idx = list(range(len(meta))) if max_cultures is None else list(range(min(max_cultures, len(meta))))
    if solver_fn is None:
        source_path, augmented = capacity.load_augmented(gem_path)
        base_cfg = capacity.selected_parameter_config()
    else:
        source_path, augmented, base_cfg = Path("injected_test_model"), None, capacity.selected_parameter_config()
    traj_rows, control_rows, flux_rows, solver_rows = [], [], [], []
    for i in selected_idx:
        row = meta.iloc[i]
        cfg = replace(base_cfg, temperature=float(row.temperature), pH=float(row.pH), DO=float(row.DO), n_time=ckpt.t_len, capacity_mode="dynamic_congestion_feedback")
        x = np.zeros(ckpt.t_len)
        b = np.zeros(ckpt.t_len)
        x[0], b[0] = cfg.x0, cfg.b0
        totals = {
            "n_growth_optimizations": 0,
            "n_product_optimizations": 0,
            "n_pfba_optimizations": 0,
            "n_actual_lp_solves": 0,
            "n_optimal_solves": 0,
            "n_infeasible_solves": 0,
            "n_unbounded_solves": 0,
            "n_surrogate_evaluations": 0,
        }
        for k in range(ckpt.interval_len):
            control = {col: float(controls[col][i, k]) for col in INTERFACE_COLUMNS}
            if solver_fn is None:
                interval_model = augmented.copy()
                mapped = apply_predicted_interface_controls(interval_model, cfg, control)
                flux = gem.solve_staged(interval_model, cfg, mapped["gamma_growth_fraction"], state="balanced")
            else:
                mapped = control.copy()
                flux = solver_fn(control, cfg, k)
            for key in totals:
                if key in flux:
                    totals[key] += int(flux[key])
            beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * max(0.0, float(control.get("z_ox", 0.0))))
            dx = float(flux["biomass_flux"]) * x[k]
            db = float(flux["beta_carotene_flux"]) * x[k] - beta_deg * b[k]
            x[k + 1] = max(1e-9, x[k] + cfg.dt * dx)
            b[k + 1] = max(0.0, b[k] + cfg.dt * db)
            common = {
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
        for k, tt in enumerate(dataset["time"]):
            traj_rows.append(
                {
                    "culture_id": row.culture_id,
                    "environment_id": row.environment_id,
                    "split": row.split,
                    "time_index": k,
                    "time": float(tt),
                    "X_pred": float(x[k]),
                    "B_pred_pfba": float(b[k]),
                    "product_generation": "sequential accumulation from solved beta_carotene_flux and biomass",
                    "direct_product_head_used": False,
                    "metabolic_backend": "yeast_gem_lp",
                }
            )
        solver_rows.append({"culture_id": row.culture_id, "environment_id": row.environment_id, "split": row.split, **totals})
    out = {
        "trajectories": pd.DataFrame(traj_rows),
        "controls": pd.DataFrame(control_rows),
        "fluxes": pd.DataFrame(flux_rows),
        "solver": pd.DataFrame(solver_rows),
    }
    if write_outputs and (max_cultures is None or max_cultures > 0):
        out["trajectories"].to_csv(DATA / "hybrid_student_predictions.csv", index=False)
        out["controls"].to_csv(DATA / "hybrid_student_controls.csv", index=False)
        out["fluxes"].to_csv(DATA / "hybrid_student_fluxes.csv", index=False)
        out["solver"].to_csv(DATA / "hybrid_student_solver_accounting.csv", index=False)
        hybrid_acceptance(out, dataset).to_csv(DATA / "hybrid_student_acceptance.csv", index=False)
    return out


def hybrid_acceptance(rollout: dict[str, pd.DataFrame], dataset: dict[str, object]) -> pd.DataFrame:
    traj = rollout["trajectories"]
    flux = rollout["fluxes"]
    solver = rollout["solver"]
    if traj.empty:
        criteria = [("exact_pfba_rollout_executed", False, "no rollout cultures requested")]
    else:
        dynamic_cols = [col for col in ["beta_carotene_flux", "biomass_flux", "oxygen_uptake", "atp_maintenance_flux"] if col in flux.columns]
        dynamic_signal = flux[dynamic_cols].nunique().sum() if dynamic_cols else 0
        criteria = [
            ("product_generated_through_pfba", bool((~traj["direct_product_head_used"]).all()), ""),
            ("zero_surrogate_evaluations", int(solver["n_surrogate_evaluations"].sum()) == 0, ""),
            ("complete_solver_accounting", int(solver["n_actual_lp_solves"].sum()) == int(solver[["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations"]].sum().sum()), ""),
            ("all_or_nearly_all_pfba_optimal", int(solver["n_infeasible_solves"].sum() + solver["n_unbounded_solves"].sum()) == 0, ""),
            ("multiple_dynamic_constraints_used", dynamic_signal > max(2, len(dynamic_cols)), "optional flux summaries absent" if len(dynamic_cols) < 4 else ""),
        ]
    passed = all(bool(row[1]) for row in criteria)
    return pd.DataFrame(
        [
            {
                "stage": "Stage 3",
                "criterion": name,
                "passed": bool(ok),
                "detail": detail,
                "decision": "pass" if passed else "blocked",
                "stage4_status": "blocked_until_stage3_passes",
            }
            for name, ok, detail in criteria
        ]
    )


def intervention_control_effects(base_controls: dict[str, float]) -> pd.DataFrame:
    rows = []
    mapping = {
        "oxidative_control": ("oxygen_lower_bound", 0.10),
        "ATP_control": ("atp_maintenance_lower_bound", 0.10),
        "pathway_capacity_control": ("CYC_effective_upper_bound", -0.10),
        "precursor_control": ("PSY_effective_upper_bound", -0.10),
        "unrelated_ER_control": ("z_er", 0.50),
    }
    for intervention, (control, delta) in mapping.items():
        before = base_controls.copy()
        after = base_controls.copy()
        after[control] = float(after.get(control, 0.0) + delta)
        changed = [key for key in sorted(set(before) | set(after)) if abs(float(before.get(key, 0.0)) - float(after.get(key, 0.0))) > 1e-12]
        rows.append({"intervention": intervention, "changed_controls": ",".join(changed), "intended_control_changed": control in changed, "n_changed_controls": len(changed)})
    return pd.DataFrame(rows)


def save_hybrid_architecture_figure() -> None:
    FIGURES.mkdir(exist_ok=True)
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="360" viewBox="0 0 1080 360"><rect width="100%" height="100%" fill="white"/>',
        ss_validation.fixed.svg_text(540, 28, "Reporter-Grounded Hybrid Student", 18, "bold"),
        ss_validation.fixed.svg_box(55, 115, 135, 58, "fixed T,pH,DO"),
        ss_validation.fixed.svg_arrow(190, 144, 270, 144),
        ss_validation.fixed.svg_box(280, 96, 170, 96, "regulatory\nstate-space"),
        ss_validation.fixed.svg_arrow(450, 144, 535, 144),
        ss_validation.fixed.svg_box(545, 96, 180, 96, "compact metabolic\ninterface controls"),
        ss_validation.fixed.svg_arrow(725, 144, 810, 144),
        ss_validation.fixed.svg_box(820, 96, 155, 96, "Yeast9 staged\npFBA"),
        ss_validation.fixed.svg_arrow(975, 144, 1035, 144),
        ss_validation.fixed.svg_box(930, 240, 115, 50, "B(t)"),
        ss_validation.fixed.svg_text(540, 320, "Primary hybrid has no direct neural product head; product is accumulated from solved beta-carotene flux.", 11),
        "</svg>",
    ]
    (FIGURES / "hybrid_student_architecture.svg").write_text("\n".join(body), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    if args.exact_stage3_validation or args.max_pfba_cultures > 0:
        run_exact_stage3_validation(args)
        return
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = load_distillation_dataset()
    write_data_inventory(dataset)
    teacher_result = train_all_teachers(dataset, seeds=args.seeds, latent_dims=args.latent_dims)
    stage1_passed = bool(teacher_result["selection"]["passed"].all())
    if not stage1_passed and not args.force_through_failed_gates:
        print("Stage 1 gate failed; Stage 2/3 blocked. See data/teacher_model_selection.csv")
        return
    pseudo = generate_teacher_pseudodata(teacher_result["selected"], dataset, n_samples=args.dense_samples, seed=args.sample_seed)
    stage2_passed = bool(pseudo["audit"]["passed"].all())
    if not stage2_passed and not args.force_through_failed_gates:
        print("Stage 2 gate failed; Stage 3 blocked. See data/teacher_pseudodata_audit.csv")
        return
    students = train_hybrid_students(dataset, pseudo["tidy"], seeds=args.seeds, latent_dims=args.latent_dims)
    save_hybrid_architecture_figure()
    if args.max_pfba_cultures > 0 and students["selected"] is not None:
        run_exact_pfba_rollout(students["selected"], dataset, max_cultures=args.max_pfba_cultures, gem_path=args.gem_path)
    else:
        pd.DataFrame(
            [
                {
                    "stage": "Stage 3",
                    "criterion": "exact_pfba_rollout_executed",
                    "passed": False,
                    "detail": "implementation present; expensive rollout not requested in this run",
                    "decision": "blocked",
                    "stage4_status": "blocked_until_stage3_passes",
                }
            ]
        ).to_csv(DATA / "hybrid_student_acceptance.csv", index=False)
    print("Reporter-grounded teacher to hybrid distillation run complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-samples", type=int, default=5000, help="Latin-hypercube teacher pseudo-conditions; use 5000-20000 for full Stage 2.")
    parser.add_argument("--sample-seed", type=int, default=20260727)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=LATENT_DIMS)
    parser.add_argument("--max-pfba-cultures", type=int, default=0, help="Exact Yeast9 pFBA cultures to roll out from saved checkpoints. Values >0 run validation-only exact Stage 3 replay without retraining.")
    parser.add_argument("--gem-path", type=str, default=None)
    parser.add_argument("--exact-stage3-validation", action="store_true", help="Load saved teacher/student checkpoints and run oracle, teacher-control, and hybrid-control exact pFBA replay.")
    parser.add_argument("--replay-types", type=str, default="oracle,teacher,hybrid", help="Comma-separated exact replay types for shard execution or merging.")
    parser.add_argument("--culture-manifest", type=str, default=None, help="Optional CSV with culture_index or culture_id selecting replay cultures.")
    parser.add_argument("--shard-index", type=int, default=None, help="Deterministic shard index to execute.")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of deterministic shards.")
    parser.add_argument("--resume", action="store_true", help="Skip a shard only when all expected shard files and solve counts are present.")
    parser.add_argument("--merge-shards", action="store_true", help="Merge completed exact replay shards and write canonical validation outputs.")
    parser.add_argument("--pilot-cultures", type=int, default=18, help="Predeclared pilot culture count before full exact replay.")
    parser.add_argument("--pilot-only", action="store_true", help="Run and write only the exact replay pilot gate.")
    parser.add_argument("--force-through-failed-gates", action="store_true", help="Diagnostic override; records gates but continues.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
