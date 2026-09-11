#!/usr/bin/env python3
"""Inverse calibration of compact controls against the real Yeast9/pFBA model.

The optimizer sees only culture-level environment and observable trajectories.
True compact controller values remain locked until post-hoc diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sympy.core.cache import clear_cache as clear_sympy_cache

import gem_backend as gem
import run_gem_dynamic_capacity as capacity
import run_reporter_grounded_hybrid_distillation as hybrid


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "inverse_gsm_calibration"

ENV_COLUMNS = hybrid.ENV_COLUMNS
INTERFACE_COLUMNS = hybrid.INTERFACE_COLUMNS
REPORTER_COLUMNS = hybrid.REPORTER_COLUMNS
LOCKBOX_COLUMNS = list(
    dict.fromkeys(
        INTERFACE_COLUMNS
        + [
            "z_er",
            "E_PSY",
            "E_DES",
            "E_CYC",
            "maximum_growth",
            "preserved_growth",
            "oxygen_uptake_used",
            "ATP_maintenance_flux_used",
            "biomass_flux_used",
            "beta_carotene_flux_used",
            "GGPP_flux_used",
            "oxidative_burden_before",
            "ATP_pressure_before",
            "bottleneck_before",
            "E_PSY_after",
            "E_DES_after",
            "E_CYC_after",
        ]
    )
)
OBSERVABLE_COLUMNS = ["B_total", "X"] + REPORTER_COLUMNS
EPS = 1e-9

REPORTER_SPECS = {
    "R_ox": ("z_ox", 1.15, 0.03, 1, 0.50),
    "R_atp": ("z_atp", 1.05, 0.04, 1, 0.45),
    "R_E_PSY": ("E_PSY_proxy", 0.92, 0.02, 1, 0.42),
    "R_E_DES": ("E_DES_proxy", 0.95, 0.02, 1, 0.42),
    "R_E_CYC": ("E_CYC_proxy", 0.98, 0.02, 1, 0.42),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset() -> dict[str, object]:
    return hybrid.load_distillation_dataset()


def culture_time_matrix(df: pd.DataFrame, col: str, index_col: str = "time_index") -> dict[str, np.ndarray]:
    return {cid: g.sort_values(index_col)[col].to_numpy(float) for cid, g in df.groupby("culture_id")}


def target_tables(dataset: dict[str, object]) -> dict[str, dict[str, np.ndarray]]:
    return {
        "B_total": culture_time_matrix(dataset["traj"], "B_total"),
        "X": culture_time_matrix(dataset["traj"], "X"),
        **{col: culture_time_matrix(dataset["reporter_table"], col) for col in REPORTER_COLUMNS},
    }


def select_diagnostic_cultures(dataset: dict[str, object]) -> pd.DataFrame:
    meta = dataset["meta"].reset_index(drop=True).copy()
    chosen: dict[int, str] = {}

    def add(idx: int, reason: str) -> None:
        chosen.setdefault(int(idx), reason)

    train = meta[meta["split"].eq("train")]
    all_meta = meta.copy()
    central_idx = int(((train[ENV_COLUMNS] - np.array([30.0, 5.0, 50.0])) ** 2).sum(axis=1).idxmin())
    add(central_idx, "central_low_stress_train_condition")

    high_t_pool = all_meta[all_meta["temperature"].eq(all_meta["temperature"].max())]
    high_t_idx = int(((high_t_pool[["pH", "DO"]] - np.array([5.0, 50.0])) ** 2).sum(axis=1).idxmin())
    add(high_t_idx, "high_temperature_environment")

    ph_pool = all_meta[(all_meta["pH"].isin([all_meta["pH"].min(), all_meta["pH"].max()])) & (~all_meta.index.isin(chosen))]
    ph_idx = int(((ph_pool[["temperature", "DO"]] - np.array([30.0, 50.0])) ** 2).sum(axis=1).idxmin())
    add(ph_idx, "pH_extreme_environment")

    low_do_pool = all_meta[(all_meta["DO"].eq(all_meta["DO"].min())) & (~all_meta.index.isin(chosen))]
    low_do_idx = int(((low_do_pool[["temperature", "pH"]] - np.array([30.0, 5.0])) ** 2).sum(axis=1).idxmin())
    add(low_do_idx, "low_DO_environment")

    held = all_meta[(all_meta["split"].eq("heldout_combination")) & (~all_meta.index.isin(chosen))]
    if held.empty:
        held = all_meta[(all_meta["split"].eq("extrapolation")) & (~all_meta.index.isin(chosen))]
    add(int(held.index[0]), "combined_heldout_like_condition")

    # Fill deterministically if any selections collided.
    for split in ["extrapolation", "heldout_combination", "train", "interpolation"]:
        for idx in all_meta[all_meta["split"].eq(split)].index:
            if len(chosen) >= 5:
                break
            add(int(idx), f"deterministic_fill_{split}")
        if len(chosen) >= 5:
            break

    rows = []
    for idx, reason in list(chosen.items())[:5]:
        row = meta.loc[idx]
        rows.append(
            {
                "culture_index": idx,
                "culture_id": row["culture_id"],
                "environment_id": row["environment_id"],
                "split": row["split"],
                "temperature": row["temperature"],
                "pH": row["pH"],
                "DO": row["DO"],
                "selection_reason": reason,
                "selected_without_fit_outcome": True,
            }
        )
    out = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    out.to_csv(DATA / "inverse_gsm_calibration_culture_manifest.csv", index=False)
    return out


def write_lockbox() -> pd.DataFrame:
    rows = []
    for col in sorted(set(LOCKBOX_COLUMNS)):
        rows.append(
            {
                "column": col,
                "classification": "HIDDEN_LOCKBOX",
                "may_enter_inverse_objective": False,
                "may_initialize_optimizer": False,
                "may_define_target_culture_bounds": False,
                "posthoc_diagnostic_only": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_calibration_lockbox.csv", index=False)
    return out


def assert_no_lockbox_usage(input_columns: list[str], objective_columns: list[str], initializer_columns: list[str]) -> None:
    lock = set(LOCKBOX_COLUMNS)
    bad = {
        "input": sorted(lock.intersection(input_columns)),
        "objective": sorted(lock.intersection(objective_columns)),
        "initializer": sorted(lock.intersection(initializer_columns)),
    }
    violations = {k: v for k, v in bad.items() if v}
    if violations:
        raise AssertionError(f"Lockbox leakage into inverse calibration: {violations}")


def global_controller_bounds(dataset: dict[str, object]) -> pd.DataFrame:
    split = dataset["splits"]
    train = split == "train"
    rows = []
    for col in INTERFACE_COLUMNS:
        vals = np.asarray(dataset["interface"][col], dtype=float)[train].reshape(-1)
        lo, hi = float(vals.min()), float(vals.max())
        if abs(hi - lo) < EPS:
            hi = lo + 1e-6
        rows.append({"control": col, "lower": lo, "upper": hi, "source": "global_training_split_controller_range"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_calibration_controller_bounds.csv", index=False)
    return out


def smooth_lag(values: np.ndarray, lag: int, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = np.r_[np.repeat(values[0], lag), values[:-lag]] if lag else values.copy()
    out = np.zeros_like(shifted)
    out[0] = shifted[0]
    for i in range(1, len(out)):
        out[i] = alpha * shifted[i] + (1.0 - alpha) * out[i - 1]
    return out


def controls_to_reporters(control: dict[str, np.ndarray], cfg: gem.GEMCultureConfig) -> dict[str, np.ndarray]:
    proxy = {
        "z_ox": control["z_ox"],
        "z_atp": control["z_atp"],
        "E_PSY_proxy": np.clip(control["PSY_effective_upper_bound"] / max(cfg.psy_base_capacity, EPS), 0.0, None),
        "E_DES_proxy": np.clip(control["DES_effective_upper_bound"] / max(cfg.des_base_capacity, EPS), 0.0, None),
        "E_CYC_proxy": np.clip(control["CYC_effective_upper_bound"] / max(cfg.cyc_base_capacity, EPS), 0.0, None),
    }
    out = {}
    for reporter, (source, scale, offset, lag, alpha) in REPORTER_SPECS.items():
        out[reporter] = np.clip(offset + scale * smooth_lag(proxy[source], lag, alpha), 0.0, None)
    return out


def decode_params(params: np.ndarray, bounds: pd.DataFrame, n_intervals: int, n_knots: int) -> dict[str, np.ndarray]:
    params = np.asarray(params, dtype=float).reshape(len(INTERFACE_COLUMNS), n_knots)
    xk = np.linspace(0, n_intervals - 1, n_knots)
    xi = np.arange(n_intervals)
    out = {}
    for j, col in enumerate(INTERFACE_COLUMNS):
        row = bounds[bounds["control"].eq(col)].iloc[0]
        vals = np.interp(xi, xk, params[j])
        vals = np.clip(vals, float(row["lower"]), float(row["upper"]))
        if col == "gamma_growth_fraction":
            vals = np.clip(vals, 0.0, 1.0)
        out[col] = vals
    return out


def mean_controller_params(dataset: dict[str, object], bounds: pd.DataFrame, n_intervals: int, n_knots: int) -> np.ndarray:
    train = dataset["splits"] == "train"
    params = []
    for col in INTERFACE_COLUMNS:
        vals = np.asarray(dataset["interface"][col], dtype=float)[train, :n_intervals]
        mean_curve = vals.mean(axis=0)
        params.extend(np.interp(np.linspace(0, n_intervals - 1, n_knots), np.arange(n_intervals), mean_curve))
    p = np.asarray(params, dtype=float)
    lo, hi = param_bounds(bounds, n_knots)
    return np.clip(p, lo, hi)


def param_bounds(bounds: pd.DataFrame, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for col in INTERFACE_COLUMNS:
        row = bounds[bounds["control"].eq(col)].iloc[0]
        lo.extend([float(row["lower"])] * n_knots)
        hi.extend([float(row["upper"])] * n_knots)
    return np.asarray(lo), np.asarray(hi)


def parameterization_audit(bounds: pd.DataFrame, stage: str, n_intervals: int, n_knots: int) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(INTERFACE_COLUMNS):
        row = bounds[bounds["control"].eq(col)].iloc[0]
        for knot in range(n_knots):
            rows.append(
                {
                    "stage": stage,
                    "n_intervals": n_intervals,
                    "controller_channel_count": len(INTERFACE_COLUMNS),
                    "knots_per_channel": n_knots,
                    "total_free_parameters": len(INTERFACE_COLUMNS) * n_knots,
                    "parameter_index": j * n_knots + knot,
                    "control": col,
                    "knot_index": knot,
                    "lower": float(row["lower"]),
                    "upper": float(row["upper"]),
                    "bound_source": row["source"],
                    "parameterization": "single_interval_direct_control" if n_intervals == 1 and n_knots == 1 else "linear_knot_interpolation",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_parameterization_audit.csv", index=False)
    return out


def fresh_replay(
    env_row: pd.Series,
    controls: dict[str, np.ndarray],
    n_intervals: int,
    source_path: Path | None = None,
    augmented=None,
    solver_fn: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if solver_fn is None and (source_path is None or augmented is None):
        source_path, augmented = capacity.load_augmented(None)
    base_cfg = capacity.selected_parameter_config()
    cfg = replace(
        base_cfg,
        temperature=float(env_row["temperature"]),
        pH=float(env_row["pH"]),
        DO=float(env_row["DO"]),
        n_time=n_intervals + 1,
        capacity_mode="dynamic_congestion_feedback",
    )
    x = np.zeros(n_intervals + 1)
    b = np.zeros(n_intervals + 1)
    x[0], b[0] = cfg.x0, cfg.b0
    flux_rows = []
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
        control = {col: float(controls[col][k]) for col in INTERFACE_COLUMNS}
        interval_model = None
        try:
            if solver_fn is None:
                interval_model = augmented.copy()
                mapped = hybrid.apply_predicted_interface_controls(interval_model, cfg, control)
                flux = gem.solve_staged(interval_model, cfg, mapped["gamma_growth_fraction"], state="balanced")
            else:
                flux = solver_fn(control, cfg, k)
            for key in totals:
                if key in flux:
                    totals[key] += int(flux[key])
        except Exception as exc:
            totals["n_solver_errors"] += 1
            totals["n_skipped_intervals"] += 1
            flux = {"solver_error": f"{type(exc).__name__}: {exc}", "beta_carotene_flux": np.nan, "biomass_flux": np.nan, "n_surrogate_evaluations": 0}
        finally:
            if interval_model is not None:
                del interval_model
            clear_sympy_cache()
            gc.collect()
        beta = float(flux.get("beta_carotene_flux", np.nan))
        biomass = float(flux.get("biomass_flux", np.nan))
        if np.isfinite(beta) and np.isfinite(biomass):
            beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * max(0.0, float(control.get("z_ox", 0.0))))
            x[k + 1] = max(1e-9, x[k] + cfg.dt * biomass * x[k])
            b[k + 1] = max(0.0, b[k] + cfg.dt * (beta * x[k] - beta_deg * b[k]))
        else:
            x[k + 1], b[k + 1] = x[k], b[k]
        flux_rows.append({"interval_index": k, "time": k * cfg.dt, **flux})
    traj = pd.DataFrame({"time_index": np.arange(n_intervals + 1), "time": np.arange(n_intervals + 1) * cfg.dt, "X_pred": x, "B_pred": b})
    return traj, pd.DataFrame(flux_rows), totals


def fit_scales(dataset: dict[str, object], n_intervals: int) -> dict[str, float]:
    train = dataset["splits"] == "train"
    targets = target_tables(dataset)
    scales = {}
    for col, by_cid in targets.items():
        mats = []
        for i, cid in enumerate(dataset["meta"]["culture_id"]):
            if train[i]:
                length = n_intervals + 1 if col in {"B_total", "X"} else n_intervals
                mats.append(by_cid[cid][:length])
        scales[col] = float(np.std(np.concatenate(mats)) + EPS)
    return scales


def objective_parts(
    culture: pd.Series,
    pred_traj: pd.DataFrame,
    pred_reporters: dict[str, np.ndarray],
    targets: dict[str, dict[str, np.ndarray]],
    scales: dict[str, float],
    reporter_weight: float,
    n_intervals: int,
) -> dict[str, float]:
    cid = culture["culture_id"]
    product_obs = targets["B_total"][cid][: n_intervals + 1]
    biomass_obs = targets["X"][cid][: n_intervals + 1]
    product = float(np.sqrt(np.mean((pred_traj["B_pred"].to_numpy(float) - product_obs) ** 2)) / scales["B_total"])
    biomass = float(np.sqrt(np.mean((pred_traj["X_pred"].to_numpy(float) - biomass_obs) ** 2)) / scales["X"])
    parts = {
        "product_loss": product,
        "biomass_loss": biomass,
    }
    reporter_losses = []
    for col in REPORTER_COLUMNS:
        obs = targets[col][cid][:n_intervals]
        pred = pred_reporters[col][:n_intervals]
        val = float(np.sqrt(np.mean((pred - obs) ** 2)) / scales[col])
        parts[f"{col}_loss"] = val
        reporter_losses.append(val)
    metabolic_block = 0.5 * (product + biomass)
    reporter_block = float(np.mean(reporter_losses)) if reporter_losses else 0.0
    parts["metabolic_block_loss"] = metabolic_block
    parts["reporter_block_loss"] = reporter_block
    parts["total_loss"] = (1.0 - reporter_weight) * metabolic_block + reporter_weight * reporter_block
    return parts


def evaluate_controller(
    params: np.ndarray,
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    n_intervals: int,
    n_knots: int,
    reporter_weight: float,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, int]]:
    controls = decode_params(params, bounds, n_intervals, n_knots)
    traj, flux, accounting = fresh_replay(culture, controls, n_intervals, source_path=source_path, augmented=augmented, solver_fn=solver_fn)
    reporters = controls_to_reporters(controls, capacity.selected_parameter_config())
    parts = objective_parts(culture, traj, reporters, target_tables(dataset), scales, reporter_weight, n_intervals)
    return parts["total_loss"], parts, traj, flux, controls, accounting


def evaluate_control_dict(
    controls: dict[str, np.ndarray],
    culture: pd.Series,
    dataset: dict[str, object],
    scales: dict[str, float],
    n_intervals: int,
    reporter_weight: float,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> tuple[float, dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, int]]:
    traj, flux, accounting = fresh_replay(culture, controls, n_intervals, source_path=source_path, augmented=augmented, solver_fn=solver_fn)
    reporters = controls_to_reporters(controls, capacity.selected_parameter_config())
    parts = objective_parts(culture, traj, reporters, target_tables(dataset), scales, reporter_weight, n_intervals)
    return parts["total_loss"], parts, traj, flux, accounting


def optimize_controller(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    n_intervals: int,
    n_knots: int,
    reporter_weight: float,
    restart_seed: int,
    iterations: int,
    population: int,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> dict[str, object]:
    lo, hi = param_bounds(bounds, n_knots)
    rng = np.random.default_rng(restart_seed)
    center = mean_controller_params(dataset, bounds, n_intervals, n_knots)
    sigma = 0.35 * (hi - lo)
    best: dict[str, object] | None = None
    history = []
    eval_count = 0
    for it in range(iterations):
        candidates = []
        if it == 0:
            candidates.append(center.copy())
        while len(candidates) < population:
            candidates.append(np.clip(center + rng.normal(0.0, sigma), lo, hi))
        scored = []
        for cand in candidates:
            loss, parts, traj, flux, controls, accounting = evaluate_controller(
                cand, culture, dataset, bounds, scales, n_intervals, n_knots, reporter_weight, source_path, augmented, solver_fn
            )
            eval_count += 1
            scored.append((loss, cand, parts, traj, flux, controls, accounting))
            history.append({"iteration": it, "evaluation": eval_count, "loss": loss, **parts})
            if best is None or loss < float(best["loss"]):
                best = {
                    "loss": loss,
                    "params": cand.copy(),
                    "parts": parts,
                    "traj": traj,
                    "flux": flux,
                    "controls": controls,
                    "accounting": accounting,
                    "iteration": it,
                    "evaluation": eval_count,
                }
        scored.sort(key=lambda row: row[0])
        elite = np.asarray([row[1] for row in scored[: max(1, population // 3)]])
        center = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.08 * (hi - lo)) * 0.75
    assert best is not None
    best["history"] = pd.DataFrame(history)
    best["evaluations"] = eval_count
    return best


def summarize_accounting(items: list[dict[str, int]]) -> dict[str, int]:
    keys = [
        "n_growth_optimizations",
        "n_product_optimizations",
        "n_pfba_optimizations",
        "n_actual_lp_solves",
        "n_optimal_solves",
        "n_infeasible_solves",
        "n_unbounded_solves",
        "n_solver_errors",
        "n_skipped_intervals",
        "n_surrogate_evaluations",
    ]
    return {key: int(sum(int(item.get(key, 0)) for item in items)) for key in keys}


def stage_a_evaluator(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    reporter_weight: float,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> Callable[[np.ndarray], tuple[float, dict[str, float], dict[str, int]]]:
    def evaluate(params: np.ndarray) -> tuple[float, dict[str, float], dict[str, int]]:
        loss, parts, _traj, _flux, _controls, accounting = evaluate_controller(
            params,
            culture,
            dataset,
            bounds,
            scales,
            n_intervals=1,
            n_knots=1,
            reporter_weight=reporter_weight,
            source_path=source_path,
            augmented=augmented,
            solver_fn=solver_fn,
        )
        return loss, parts, accounting

    return evaluate


def stage_a_reference_losses(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> pd.DataFrame:
    lo, hi = param_bounds(bounds, 1)
    oracle_controls = true_controls_for_culture(dataset, culture, n_intervals=1)
    oracle_loss, oracle_parts, _traj, _flux, oracle_accounting = evaluate_control_dict(
        oracle_controls, culture, dataset, scales, 1, 0.0, source_path, augmented, solver_fn
    )
    mean_params = mean_controller_params(dataset, bounds, 1, 1)
    random_params = np.random.default_rng(991).uniform(lo, hi)
    oracle_params = oracle_params_for_culture(dataset, culture, bounds, 1, 1)
    perturbed = np.clip(oracle_params + np.random.default_rng(992).normal(0, 0.30, size=len(lo)) * (hi - lo), lo, hi)
    rows = [
        {
            "reference": "oracle_true_controller",
            "loss": oracle_loss,
            **oracle_parts,
            **oracle_accounting,
            "metabolic_backend": "fake_solver" if solver_fn is not None else "yeast_gem_lp",
            "posthoc_controller_distance": 0.0,
            "uses_exact_recorded_controller": True,
        }
    ]
    for name, params in [
        ("training_mean_controller", mean_params),
        ("single_random_valid_controller", random_params),
        ("perturbed_true_controller", perturbed),
    ]:
        loss, parts, _traj, _flux, _controls, accounting = evaluate_controller(
            params, culture, dataset, bounds, scales, 1, 1, 0.0, source_path, augmented, solver_fn
        )
        rows.append(
            {
                "reference": name,
                "loss": loss,
                **parts,
                **accounting,
                "metabolic_backend": "fake_solver" if solver_fn is not None else "yeast_gem_lp",
                "posthoc_controller_distance": controller_distance(params, oracle_params, bounds, 1),
                "uses_exact_recorded_controller": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_stageA_reference_losses.csv", index=False)
    return out


def stage_a_random_baseline(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    n_samples: int,
    seed: int,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> pd.DataFrame:
    lo, hi = param_bounds(bounds, 1)
    oracle_params = oracle_params_for_culture(dataset, culture, bounds, 1, 1)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        params = rng.uniform(lo, hi)
        tic = time.perf_counter()
        loss, parts, _traj, _flux, _controls, accounting = evaluate_controller(
            params, culture, dataset, bounds, scales, 1, 1, 0.0, source_path, augmented, solver_fn
        )
        rows.append(
            {
                "sample_index": i,
                "rng_seed": seed,
                "loss": loss,
                "wall_time_seconds": time.perf_counter() - tic,
                "controller_vector_json": json.dumps([float(x) for x in params]),
                "posthoc_controller_distance": controller_distance(params, oracle_params, bounds, 1),
                "metabolic_backend": "fake_solver" if solver_fn is not None else "yeast_gem_lp",
                **parts,
                **accounting,
            }
        )
    out = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "n_samples": int(len(out)),
                "rng_seed": int(seed),
                "minimum": float(out["loss"].min()),
                "p05": float(out["loss"].quantile(0.05)),
                "p10": float(out["loss"].quantile(0.10)),
                "median": float(out["loss"].median()),
                "mean": float(out["loss"].mean()),
                "p90": float(out["loss"].quantile(0.90)),
                "p95": float(out["loss"].quantile(0.95)),
                "maximum": float(out["loss"].max()),
            }
        ]
    )
    out.to_csv(DATA / "inverse_gsm_stageA_random_baseline.csv", index=False)
    summary.to_csv(DATA / "inverse_gsm_stageA_random_baseline_summary.csv", index=False)
    return out


def _log_eval(
    rows: list[dict[str, object]],
    optimizer: str,
    restart: int,
    evaluation: int,
    current_loss: float,
    best_loss: float,
    parts: dict[str, float],
    params: np.ndarray,
    accounting: dict[str, int],
    tic: float,
    init_strategy: str,
) -> None:
    rows.append(
        {
            "optimizer": optimizer,
            "restart": restart,
            "evaluation_number": evaluation,
            "current_loss": float(current_loss),
            "best_loss_so_far": float(best_loss),
            "current_product_loss": float(parts.get("product_loss", np.nan)),
            "current_biomass_loss": float(parts.get("biomass_loss", np.nan)),
            "current_metabolic_block_loss": float(parts.get("metabolic_block_loss", np.nan)),
            "controller_vector_json": json.dumps([float(x) for x in params]),
            "wall_time_seconds": time.perf_counter() - tic,
            "LP_solve_count": int(accounting.get("n_actual_lp_solves", 0)),
            "initialization_strategy": init_strategy,
        }
    )


def optimizer_random_search(
    evaluate: Callable[[np.ndarray], tuple[float, dict[str, float], dict[str, int]]],
    lo: np.ndarray,
    hi: np.ndarray,
    budget: int,
    seed: int,
    restart: int,
    init_strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    tic = time.perf_counter()
    best_loss = np.inf
    best_params = None
    best_parts = {}
    accountings = []
    for ev in range(1, budget + 1):
        params = 0.5 * (lo + hi) if ev == 1 and init_strategy == "global_midpoint" else rng.uniform(lo, hi)
        loss, parts, accounting = evaluate(params)
        accountings.append(accounting)
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
            best_parts = parts
        _log_eval(rows, "random_search", restart, ev, loss, best_loss, parts, params, accounting, tic, init_strategy)
    conv = pd.DataFrame(rows)
    best_eval = int(conv["best_loss_so_far"].idxmin() + 1) if not conv.empty else 0
    return {"loss": best_loss, "params": best_params, "parts": best_parts, "accounting": summarize_accounting(accountings), "best_evaluation_number": best_eval, "wall_time_seconds": float(time.perf_counter() - tic)}, conv


def optimizer_de(
    evaluate: Callable[[np.ndarray], tuple[float, dict[str, float], dict[str, int]]],
    lo: np.ndarray,
    hi: np.ndarray,
    budget: int,
    seed: int,
    restart: int,
    init_strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dim = len(lo)
    pop_size = max(6, min(18, budget // 2))
    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    if init_strategy == "global_midpoint":
        pop[0] = 0.5 * (lo + hi)
    scores = np.full(pop_size, np.inf)
    rows = []
    tic = time.perf_counter()
    best_loss = np.inf
    best_params = None
    best_parts = {}
    accountings = []
    ev = 0
    for i in range(pop_size):
        if ev >= budget:
            break
        loss, parts, accounting = evaluate(pop[i])
        ev += 1
        accountings.append(accounting)
        scores[i] = loss
        if loss < best_loss:
            best_loss, best_params, best_parts = loss, pop[i].copy(), parts
        _log_eval(rows, "differential_evolution", restart, ev, loss, best_loss, parts, pop[i], accounting, tic, init_strategy)
    while ev < budget:
        for i in range(pop_size):
            if ev >= budget:
                break
            idx = rng.choice(pop_size, size=3, replace=False)
            mutant = pop[idx[0]] + 0.7 * (pop[idx[1]] - pop[idx[2]])
            cross = rng.random(dim) < 0.7
            if not cross.any():
                cross[int(rng.integers(0, dim))] = True
            trial = np.where(cross, mutant, pop[i])
            trial = np.clip(trial, lo, hi)
            loss, parts, accounting = evaluate(trial)
            ev += 1
            accountings.append(accounting)
            if loss <= scores[i]:
                pop[i] = trial
                scores[i] = loss
            if loss < best_loss:
                best_loss, best_params, best_parts = loss, trial.copy(), parts
            _log_eval(rows, "differential_evolution", restart, ev, loss, best_loss, parts, trial, accounting, tic, init_strategy)
    conv = pd.DataFrame(rows)
    best_eval = int(conv["best_loss_so_far"].idxmin() + 1) if not conv.empty else 0
    return {"loss": best_loss, "params": best_params, "parts": best_parts, "accounting": summarize_accounting(accountings), "best_evaluation_number": best_eval, "wall_time_seconds": float(time.perf_counter() - tic)}, conv


def optimizer_powell(
    evaluate: Callable[[np.ndarray], tuple[float, dict[str, float], dict[str, int]]],
    lo: np.ndarray,
    hi: np.ndarray,
    budget: int,
    seed: int,
    restart: int,
    init_strategy: str,
    start: np.ndarray | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = start.copy() if start is not None else rng.uniform(lo, hi)
    if init_strategy == "global_midpoint":
        x = 0.5 * (lo + hi)
    step = 0.25 * (hi - lo)
    rows = []
    tic = time.perf_counter()
    accountings = []
    loss, parts, accounting = evaluate(x)
    accountings.append(accounting)
    ev = 1
    best_loss, best_params, best_parts = loss, x.copy(), parts
    _log_eval(rows, "powell_multistart", restart, ev, loss, best_loss, parts, x, accounting, tic, init_strategy)
    dim = len(lo)
    while ev < budget:
        improved = False
        for j in range(dim):
            for sign in [1.0, -1.0]:
                if ev >= budget:
                    break
                cand = x.copy()
                cand[j] = np.clip(cand[j] + sign * step[j], lo[j], hi[j])
                loss, parts, accounting = evaluate(cand)
                accountings.append(accounting)
                ev += 1
                if loss < best_loss:
                    x = cand
                    best_loss, best_params, best_parts = loss, cand.copy(), parts
                    improved = True
                _log_eval(rows, "powell_multistart", restart, ev, loss, best_loss, parts, cand, accounting, tic, init_strategy)
            if ev >= budget:
                break
        if not improved:
            step *= 0.55
        if np.max(step / np.maximum(hi - lo, EPS)) < 0.01 and ev < budget:
            x = np.clip(best_params + rng.normal(0.0, 0.05, size=dim) * (hi - lo), lo, hi)
            step = np.maximum(step, 0.05 * (hi - lo))
    conv = pd.DataFrame(rows)
    best_eval = int(conv["best_loss_so_far"].idxmin() + 1) if not conv.empty else 0
    return {"loss": best_loss, "params": best_params, "parts": best_parts, "accounting": summarize_accounting(accountings), "best_evaluation_number": best_eval, "wall_time_seconds": float(time.perf_counter() - tic)}, conv


def optimizer_annealing(
    evaluate: Callable[[np.ndarray], tuple[float, dict[str, float], dict[str, int]]],
    lo: np.ndarray,
    hi: np.ndarray,
    budget: int,
    seed: int,
    restart: int,
    init_strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = 0.5 * (lo + hi) if init_strategy == "global_midpoint" else rng.uniform(lo, hi)
    rows = []
    tic = time.perf_counter()
    accountings = []
    loss, parts, accounting = evaluate(x)
    accountings.append(accounting)
    ev = 1
    best_loss, best_params, best_parts = loss, x.copy(), parts
    current_loss = loss
    _log_eval(rows, "annealing", restart, ev, loss, best_loss, parts, x, accounting, tic, init_strategy)
    while ev < budget:
        frac = 1.0 - ev / max(budget, 1)
        step = (0.03 + 0.35 * frac) * (hi - lo)
        cand = np.clip(x + rng.normal(0.0, step), lo, hi)
        loss, parts, accounting = evaluate(cand)
        accountings.append(accounting)
        ev += 1
        temp = max(0.02, frac)
        accept = loss < current_loss or rng.random() < math.exp(-max(loss - current_loss, 0.0) / temp)
        if accept:
            x = cand
            current_loss = loss
        if loss < best_loss:
            best_loss, best_params, best_parts = loss, cand.copy(), parts
        _log_eval(rows, "annealing", restart, ev, loss, best_loss, parts, cand, accounting, tic, init_strategy)
    conv = pd.DataFrame(rows)
    best_eval = int(conv["best_loss_so_far"].idxmin() + 1) if not conv.empty else 0
    return {"loss": best_loss, "params": best_params, "parts": best_parts, "accounting": summarize_accounting(accountings), "best_evaluation_number": best_eval, "wall_time_seconds": float(time.perf_counter() - tic)}, conv


def oracle_params_for_culture(dataset: dict[str, object], culture: pd.Series, bounds: pd.DataFrame, n_intervals: int, n_knots: int) -> np.ndarray:
    idx = int(culture["culture_index"])
    params = []
    for col in INTERFACE_COLUMNS:
        curve = np.asarray(dataset["interface"][col], dtype=float)[idx, :n_intervals]
        params.extend(np.interp(np.linspace(0, n_intervals - 1, n_knots), np.arange(n_intervals), curve))
    lo, hi = param_bounds(bounds, n_knots)
    return np.clip(np.asarray(params), lo, hi)


def true_controls_for_culture(dataset: dict[str, object], culture: pd.Series, n_intervals: int) -> dict[str, np.ndarray]:
    idx = int(culture["culture_index"])
    return {col: np.asarray(dataset["interface"][col], dtype=float)[idx, :n_intervals].copy() for col in INTERFACE_COLUMNS}


def controller_distance(a: np.ndarray, b: np.ndarray, bounds: pd.DataFrame, n_knots: int) -> float:
    lo, hi = param_bounds(bounds, n_knots)
    return float(np.sqrt(np.mean(((a - b) / np.maximum(hi - lo, EPS)) ** 2)))


def stage_a_landscape(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    points: int,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> pd.DataFrame:
    lo, hi = param_bounds(bounds, 1)
    contexts = {
        "training_mean_context": mean_controller_params(dataset, bounds, 1, 1),
        "posthoc_oracle_context": oracle_params_for_culture(dataset, culture, bounds, 1, 1),
    }
    rows = []
    for context_name, base in contexts.items():
        for j, col in enumerate(INTERFACE_COLUMNS):
            grid = np.linspace(lo[j], hi[j], points)
            for value in grid:
                params = base.copy()
                params[j] = value
                loss, parts, _traj, _flux, _controls, accounting = evaluate_controller(
                    params, culture, dataset, bounds, scales, 1, 1, 0.0, source_path, augmented, solver_fn
                )
                rows.append(
                    {
                        "culture_id": culture["culture_id"],
                        "context": context_name,
                        "control": col,
                        "value": float(value),
                        "loss": loss,
                        **parts,
                        **accounting,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_stageA_1d_landscape.csv", index=False)
    return out


def stage_a_gate(reference: pd.DataFrame, random_baseline: pd.DataFrame, optimizer_summary: pd.DataFrame, n_restarts: int) -> pd.DataFrame:
    oracle_loss = float(reference[reference["reference"].eq("oracle_true_controller")]["loss"].iloc[0])
    mean_loss = float(reference[reference["reference"].eq("training_mean_controller")]["loss"].iloc[0])
    random_median = float(random_baseline["loss"].median()) if not random_baseline.empty else np.nan
    random_best = float(random_baseline["loss"].min()) if not random_baseline.empty else np.nan
    best_inverse = float(optimizer_summary["best_loss"].min()) if not optimizer_summary.empty else np.nan
    rep = stage_a_reproducibility_analysis(reference, random_baseline, optimizer_summary, write=False)
    if rep.empty:
        best_optimizer = ""
        best_restart_losses = pd.Series(dtype=float)
        consistent = False
        best_median_closure = np.nan
    else:
        best_row = rep.sort_values(["median_best_loss", "mean_best_loss"]).iloc[0]
        best_optimizer = str(best_row["optimizer"])
        best_restart_losses = optimizer_summary[optimizer_summary["optimizer"].eq(best_optimizer)]["best_loss"]
        consistent = bool(best_row["fraction_beating_training_mean"] >= 0.8 and best_row["median_oracle_gap_closure"] > 0.10)
        best_median_closure = float(best_row["median_oracle_gap_closure"])
    closure = (mean_loss - best_inverse) / max(mean_loss - oracle_loss, EPS) if np.isfinite(best_inverse) else np.nan
    surrogate_total = int(optimizer_summary.get("n_surrogate_evaluations", pd.Series(dtype=int)).sum()) if not optimizer_summary.empty else 0
    criteria = [
        ("oracle_replay_near_zero", oracle_loss < 1e-6, oracle_loss),
        ("random_controllers_generally_poor", random_median > mean_loss, random_median),
        ("random_distribution_quantified", len(random_baseline) >= 100, len(random_baseline)),
        ("no_surrogate_evaluations", surrogate_total == 0, surrogate_total),
        ("optimizer_beats_training_mean", best_inverse < mean_loss, best_inverse),
        ("optimizer_reproducible_across_restarts", consistent, float(best_restart_losses.mean()) if len(best_restart_losses) else np.nan),
        ("positive_oracle_gap_closure", closure > 0.0, closure),
        ("positive_median_oracle_gap_closure", best_median_closure > 0.10, best_median_closure),
    ]
    passed = all(bool(ok) for _name, ok, _value in criteria)
    out = pd.DataFrame(
        [
            {
                "stage": "Stage A",
                "criterion": name,
                "passed": bool(ok),
                "value": value,
                "decision": "pass" if passed else "stop_after_stage_A",
                "oracle_loss": oracle_loss,
                "training_mean_loss": mean_loss,
                "random_median_loss": random_median,
                "random_best_loss": random_best,
                "inverse_best_loss": best_inverse,
                "fractional_closure_toward_oracle": closure,
                "best_optimizer": best_optimizer,
            }
            for name, ok, value in criteria
        ]
    )
    out.to_csv(DATA / "inverse_gsm_stageA_gate.csv", index=False)
    return out


def stage_a_auxiliary_tables(optimizer_summary: pd.DataFrame) -> None:
    if optimizer_summary.empty:
        pd.DataFrame().to_csv(DATA / "inverse_gsm_stageA_controller_solutions.csv", index=False)
        pd.DataFrame().to_csv(DATA / "inverse_gsm_stageA_loss_breakdown.csv", index=False)
        pd.DataFrame().to_csv(DATA / "inverse_gsm_stageA_solver_accounting.csv", index=False)
        return

    solution_rows = []
    for row in optimizer_summary.itertuples(index=False):
        controls = json.loads(row.best_controller_vector_json)
        solution = {
            "culture_id": row.culture_id,
            "optimizer": row.optimizer,
            "restart": row.restart,
            "restart_seed": row.restart_seed,
            "initialization_strategy": row.initialization_strategy,
            "reporter_weight": row.reporter_weight,
            "metabolic_backend": row.metabolic_backend,
            "best_loss": row.best_loss,
            "posthoc_controller_distance": row.posthoc_controller_distance,
        }
        for control, value in zip(INTERFACE_COLUMNS, controls):
            solution[control] = float(value)
        solution_rows.append(solution)

    loss_cols = [
        "culture_id",
        "optimizer",
        "restart",
        "restart_seed",
        "initialization_strategy",
        "reporter_weight",
        "metabolic_backend",
        "best_loss",
        "best_evaluation_number",
        "best_product_loss",
        "best_biomass_loss",
        "best_reporter_block_loss",
        "fractional_closure_toward_oracle",
        "posthoc_controller_distance",
    ]
    solver_cols = [
        "culture_id",
        "optimizer",
        "restart",
        "restart_seed",
        "initialization_strategy",
        "reporter_weight",
        "metabolic_backend",
        "n_evaluations",
        "best_evaluation_number",
        "wall_time_seconds",
        "best_loss",
        "n_growth_optimizations",
        "n_product_optimizations",
        "n_pfba_optimizations",
        "n_actual_lp_solves",
        "n_optimal_solves",
        "n_infeasible_solves",
        "n_unbounded_solves",
        "n_solver_errors",
        "n_skipped_intervals",
        "n_surrogate_evaluations",
    ]
    pd.DataFrame(solution_rows).to_csv(DATA / "inverse_gsm_stageA_controller_solutions.csv", index=False)
    optimizer_summary[loss_cols].to_csv(DATA / "inverse_gsm_stageA_loss_breakdown.csv", index=False)
    optimizer_summary[solver_cols].to_csv(DATA / "inverse_gsm_stageA_solver_accounting.csv", index=False)


def stage_a_reproducibility_analysis(
    reference: pd.DataFrame,
    random_baseline: pd.DataFrame,
    optimizer_summary: pd.DataFrame,
    write: bool = True,
) -> pd.DataFrame:
    if optimizer_summary.empty:
        out = pd.DataFrame()
        if write:
            out.to_csv(DATA / "inverse_gsm_stageA_reproducibility.csv", index=False)
        return out
    oracle_loss = float(reference[reference["reference"].eq("oracle_true_controller")]["loss"].iloc[0])
    mean_loss = float(reference[reference["reference"].eq("training_mean_controller")]["loss"].iloc[0])
    best_random = float(random_baseline["loss"].min()) if not random_baseline.empty else np.nan
    rows = []
    for optimizer, group in optimizer_summary[optimizer_summary["reporter_weight"].eq(0.0)].groupby("optimizer"):
        losses = group["best_loss"].to_numpy(float)
        closures = (mean_loss - losses) / max(mean_loss - oracle_loss, EPS)
        beats_mean = losses < mean_loss
        beats_random = losses < best_random if np.isfinite(best_random) else np.zeros_like(losses, dtype=bool)
        rows.append(
            {
                "optimizer": optimizer,
                "restart_count": int(len(group)),
                "best_loss": float(np.min(losses)),
                "median_best_loss": float(np.median(losses)),
                "mean_best_loss": float(np.mean(losses)),
                "std_best_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                "n_beating_training_mean": int(np.sum(beats_mean)),
                "fraction_beating_training_mean": float(np.mean(beats_mean)),
                "n_beating_best_random": int(np.sum(beats_random)),
                "fraction_beating_best_random": float(np.mean(beats_random)),
                "median_oracle_gap_closure": float(np.median(closures)),
                "min_oracle_gap_closure": float(np.min(closures)),
                "max_oracle_gap_closure": float(np.max(closures)),
                "training_mean_loss": mean_loss,
                "oracle_loss": oracle_loss,
                "best_random_loss": best_random,
            }
        )
    out = pd.DataFrame(rows).sort_values(["median_best_loss", "mean_best_loss"])
    if write:
        out.to_csv(DATA / "inverse_gsm_stageA_reproducibility.csv", index=False)
    return out


def stage_a_controller_channel_errors(optimizer_summary: pd.DataFrame, oracle_params: np.ndarray, bounds: pd.DataFrame) -> pd.DataFrame:
    lo, hi = param_bounds(bounds, 1)
    rows = []
    for row in optimizer_summary.itertuples(index=False):
        params = np.asarray(json.loads(row.best_controller_vector_json), dtype=float)
        for j, control in enumerate(INTERFACE_COLUMNS):
            rows.append(
                {
                    "culture_id": row.culture_id,
                    "optimizer": row.optimizer,
                    "restart": row.restart,
                    "restart_seed": row.restart_seed,
                    "reporter_weight": row.reporter_weight,
                    "control": control,
                    "inferred_value": float(params[j]),
                    "oracle_value": float(oracle_params[j]),
                    "absolute_error": float(abs(params[j] - oracle_params[j])),
                    "normalized_error": float(abs(params[j] - oracle_params[j]) / max(hi[j] - lo[j], EPS)),
                    "observable_loss": float(row.best_loss),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_stageA_controller_channel_errors.csv", index=False)
    return out


def stage_a_landscape_sensitivity(landscape: pd.DataFrame, dataset: dict[str, object], culture: pd.Series, bounds: pd.DataFrame) -> pd.DataFrame:
    if landscape.empty:
        out = pd.DataFrame()
        out.to_csv(DATA / "inverse_gsm_stageA_landscape_sensitivity.csv", index=False)
        return out
    contexts = {
        "training_mean_context": mean_controller_params(dataset, bounds, 1, 1),
        "posthoc_oracle_context": oracle_params_for_culture(dataset, culture, bounds, 1, 1),
    }
    rows = []
    for (context, control), group in landscape.groupby(["context", "control"]):
        g = group.sort_values("value")
        values = g["value"].to_numpy(float)
        losses = g["loss"].to_numpy(float)
        j = INTERFACE_COLUMNS.index(control)
        center = float(contexts[context][j])
        nearest = np.argsort(np.abs(values - center))[: min(3, len(values))]
        local_values = values[nearest]
        local_losses = losses[nearest]
        if len(local_values) >= 2 and np.ptp(local_values) > EPS:
            local_sensitivity = float(abs(np.polyfit(local_values, local_losses, 1)[0]))
        else:
            local_sensitivity = 0.0
        loss_range = float(losses.max() - losses.min())
        if loss_range < 0.01:
            classification = "flat"
        elif loss_range < 0.05:
            classification = "weak"
        else:
            classification = "strong"
        rows.append(
            {
                "context": context,
                "control": control,
                "loss_min": float(losses.min()),
                "loss_max": float(losses.max()),
                "loss_range": loss_range,
                "local_sensitivity": local_sensitivity,
                "identifiability_classification": classification,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_stageA_landscape_sensitivity.csv", index=False)
    return out


def stage_a_phenotype_reconstruction(
    culture: pd.Series,
    dataset: dict[str, object],
    bounds: pd.DataFrame,
    scales: dict[str, float],
    optimizer_summary: pd.DataFrame,
    source_path: Path | None,
    augmented,
    solver_fn: Callable | None = None,
) -> pd.DataFrame:
    targets = target_tables(dataset)
    mean_params = mean_controller_params(dataset, bounds, 1, 1)
    oracle_controls = true_controls_for_culture(dataset, culture, n_intervals=1)
    _mean_loss, _mean_parts, mean_traj, _mean_flux, _mean_controls, _mean_acc = evaluate_controller(
        mean_params, culture, dataset, bounds, scales, 1, 1, 0.0, source_path, augmented, solver_fn
    )
    _oracle_loss, _oracle_parts, oracle_traj, _oracle_flux, _oracle_acc = evaluate_control_dict(
        oracle_controls, culture, dataset, scales, 1, 0.0, source_path, augmented, solver_fn
    )
    rows = []
    for row in optimizer_summary[optimizer_summary["reporter_weight"].eq(0.0)].itertuples(index=False):
        params = np.asarray(json.loads(row.best_controller_vector_json), dtype=float)
        _loss, _parts, inv_traj, _flux, _controls, _acc = evaluate_controller(
            params, culture, dataset, bounds, scales, 1, 1, 0.0, source_path, augmented, solver_fn
        )
        for time_index in [0, 1]:
            rows.append(
                {
                    "culture_id": row.culture_id,
                    "optimizer": row.optimizer,
                    "restart": row.restart,
                    "restart_seed": row.restart_seed,
                    "time_index": time_index,
                    "observed_product": float(targets["B_total"][culture["culture_id"]][time_index]),
                    "inverse_fit_product": float(inv_traj["B_pred"].iloc[time_index]),
                    "mean_controller_product": float(mean_traj["B_pred"].iloc[time_index]),
                    "oracle_product": float(oracle_traj["B_pred"].iloc[time_index]),
                    "observed_biomass": float(targets["X"][culture["culture_id"]][time_index]),
                    "inverse_fit_biomass": float(inv_traj["X_pred"].iloc[time_index]),
                    "mean_controller_biomass": float(mean_traj["X_pred"].iloc[time_index]),
                    "oracle_biomass": float(oracle_traj["X_pred"].iloc[time_index]),
                    "best_loss": float(row.best_loss),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "inverse_gsm_stageA_phenotype_reconstruction.csv", index=False)
    return out


def stage_a_efficiency_analysis(reference: pd.DataFrame, convergence: pd.DataFrame, optimizer_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    milestones = [25, 50, 100, 250, 500]
    rows = []
    for (optimizer, restart), group in convergence[convergence["reporter_weight"].eq(0.0)].groupby(["optimizer", "restart"]):
        g = group.sort_values("evaluation_number")
        for milestone in milestones:
            usable = g[g["evaluation_number"] <= milestone]
            rows.append(
                {
                    "optimizer": optimizer,
                    "restart": restart,
                    "milestone_evaluations": milestone,
                    "best_loss_so_far": float(usable["best_loss_so_far"].iloc[-1]) if not usable.empty else np.nan,
                    "available": bool(not usable.empty and int(g["evaluation_number"].max()) >= milestone),
                }
            )
    efficiency = pd.DataFrame(rows)
    efficiency.to_csv(DATA / "inverse_gsm_stageA_efficiency.csv", index=False)

    if optimizer_summary.empty:
        cost = pd.DataFrame()
    else:
        rep = stage_a_reproducibility_analysis(reference, pd.DataFrame(), optimizer_summary, write=False)
        winner = str(rep.sort_values(["median_best_loss", "mean_best_loss"]).iloc[0]["optimizer"]) if not rep.empty else str(optimizer_summary.iloc[0]["optimizer"])
        win = optimizer_summary[(optimizer_summary["optimizer"].eq(winner)) & (optimizer_summary["reporter_weight"].eq(0.0))]
        wall_per_culture = float(win["wall_time_seconds"].sum())
        lp_per_culture = int(win["n_actual_lp_solves"].sum())
        rows = []
        for parallelism in [1, 8, 16, 32]:
            rows.append(
                {
                    "winning_optimizer": winner,
                    "parallelism": parallelism,
                    "lp_solves_per_fitted_culture": lp_per_culture,
                    "wall_seconds_per_fitted_culture_serial_restarts": wall_per_culture,
                    "estimated_77_culture_wall_seconds": float(math.ceil(77 / parallelism) * wall_per_culture),
                    "estimated_77_culture_lp_solves": int(77 * lp_per_culture),
                    "planning_only_not_executed": True,
                }
            )
        cost = pd.DataFrame(rows)
    cost.to_csv(DATA / "inverse_gsm_stageA_cost_estimate.csv", index=False)
    return efficiency, cost


def run_stage_a(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    dataset = load_dataset()
    manifest = select_diagnostic_cultures(dataset).head(1)
    culture = pd.Series(manifest.iloc[0].to_dict())
    write_lockbox()
    bounds = global_controller_bounds(dataset)
    parameterization = parameterization_audit(bounds, "Stage A", n_intervals=1, n_knots=1)
    assert_no_lockbox_usage(ENV_COLUMNS, ["B_total", "X"], [])
    scales = fit_scales(dataset, 1)
    source_path, augmented = (None, None) if args.fake_solver else capacity.load_augmented(None)
    solver_fn = fake_solver if args.fake_solver else None
    reference = stage_a_reference_losses(culture, dataset, bounds, scales, source_path, augmented, solver_fn)
    random_baseline = stage_a_random_baseline(
        culture, dataset, bounds, scales, args.random_samples, args.seed, source_path, augmented, solver_fn
    )
    lo, hi = param_bounds(bounds, 1)
    oracle_params = oracle_params_for_culture(dataset, culture, bounds, 1, 1)
    convergence_parts = []
    summary_rows = []
    for reporter_weight in args.stage_a_reporter_weights:
        objective_cols = ["B_total", "X"] + ([] if reporter_weight == 0 else REPORTER_COLUMNS)
        assert_no_lockbox_usage(ENV_COLUMNS, objective_cols, [])
        evaluate = stage_a_evaluator(culture, dataset, bounds, scales, reporter_weight, source_path, augmented, solver_fn)
        for optimizer in args.optimizers:
            for restart_i, seed in enumerate(args.restart_seeds):
                init_strategy = "training_mean_controller" if restart_i == 0 else "global_random_valid"
                if optimizer == "differential_evolution":
                    best, conv = optimizer_de(evaluate, lo, hi, args.max_evaluations, seed, restart_i, init_strategy)
                elif optimizer == "powell_multistart":
                    start = mean_controller_params(dataset, bounds, 1, 1) if restart_i == 0 else None
                    best, conv = optimizer_powell(evaluate, lo, hi, args.max_evaluations, seed, restart_i, init_strategy, start=start)
                elif optimizer == "annealing":
                    best, conv = optimizer_annealing(evaluate, lo, hi, args.max_evaluations, seed, restart_i, init_strategy)
                elif optimizer == "random_search":
                    best, conv = optimizer_random_search(evaluate, lo, hi, args.max_evaluations, seed, restart_i, init_strategy)
                else:
                    raise ValueError(f"Unknown optimizer {optimizer!r}")
                conv = conv.assign(culture_id=culture["culture_id"], reporter_weight=reporter_weight)
                convergence_parts.append(conv)
                mean_loss = float(reference[reference["reference"].eq("training_mean_controller")]["loss"].iloc[0])
                oracle_loss = float(reference[reference["reference"].eq("oracle_true_controller")]["loss"].iloc[0])
                summary_rows.append(
                    {
                        "culture_id": culture["culture_id"],
                        "optimizer": optimizer,
                        "restart": restart_i,
                        "restart_seed": seed,
                        "initialization_strategy": init_strategy,
                        "reporter_weight": reporter_weight,
                        "metabolic_backend": "fake_solver" if solver_fn is not None else "yeast_gem_lp",
                        "best_loss": float(best["loss"]),
                        "best_product_loss": float(best["parts"].get("product_loss", np.nan)),
                        "best_biomass_loss": float(best["parts"].get("biomass_loss", np.nan)),
                        "best_reporter_block_loss": float(best["parts"].get("reporter_block_loss", np.nan)),
                        "posthoc_controller_distance": controller_distance(best["params"], oracle_params, bounds, 1),
                        "fractional_closure_toward_oracle": (mean_loss - float(best["loss"])) / max(mean_loss - oracle_loss, EPS),
                        "best_controller_vector_json": json.dumps([float(x) for x in best["params"]]),
                        "n_evaluations": args.max_evaluations,
                        "best_evaluation_number": int(best.get("best_evaluation_number", 0)),
                        "wall_time_seconds": float(best.get("wall_time_seconds", np.nan)),
                        **best["accounting"],
                    }
                )
    convergence = pd.concat(convergence_parts, ignore_index=True) if convergence_parts else pd.DataFrame()
    if not convergence.empty:
        convergence["cumulative_LP_solve_count"] = convergence.groupby(["optimizer", "restart", "reporter_weight"])["LP_solve_count"].cumsum()
    optimizer_summary = pd.DataFrame(summary_rows)
    landscape = stage_a_landscape(culture, dataset, bounds, scales, args.landscape_points, source_path, augmented, solver_fn)
    landscape_sensitivity = stage_a_landscape_sensitivity(landscape, dataset, culture, bounds)
    channel_errors = stage_a_controller_channel_errors(optimizer_summary, oracle_params, bounds)
    reproducibility = stage_a_reproducibility_analysis(reference, random_baseline, optimizer_summary)
    phenotype = stage_a_phenotype_reconstruction(culture, dataset, bounds, scales, optimizer_summary, source_path, augmented, solver_fn)
    efficiency, cost = stage_a_efficiency_analysis(reference, convergence, optimizer_summary)
    gate = stage_a_gate(
        reference,
        random_baseline,
        optimizer_summary[optimizer_summary["reporter_weight"].eq(0.0)],
        n_restarts=len(args.restart_seeds),
    )
    convergence.to_csv(DATA / "inverse_gsm_stageA_convergence.csv", index=False)
    optimizer_summary.to_csv(DATA / "inverse_gsm_stageA_optimizer_comparison.csv", index=False)
    stage_a_auxiliary_tables(optimizer_summary)
    save_stage_a_figures(convergence, optimizer_summary, landscape, reference, random_baseline)
    decision = stage_a_decision_table(gate, optimizer_summary, reproducibility, landscape_sensitivity, cost)
    decision.to_csv(DATA / "inverse_gsm_stageA_decision_table.csv", index=False)
    decision.to_csv(DATA / "inverse_gsm_calibration_decision.csv", index=False)
    write_stage_a_summary(decision, gate, args)
    return {
        "parameterization": parameterization,
        "reference": reference,
        "random_baseline": random_baseline,
        "optimizer_summary": optimizer_summary,
        "convergence": convergence,
        "landscape": landscape,
        "landscape_sensitivity": landscape_sensitivity,
        "channel_errors": channel_errors,
        "reproducibility": reproducibility,
        "phenotype": phenotype,
        "efficiency": efficiency,
        "cost": cost,
        "gate": gate,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    dataset = load_dataset()
    manifest = select_diagnostic_cultures(dataset)
    if args.central_only:
        manifest = manifest.head(1).copy()
    lockbox = write_lockbox()
    bounds = global_controller_bounds(dataset)
    assert_no_lockbox_usage(ENV_COLUMNS, ["B_total", "X"], [])
    scales = fit_scales(dataset, args.max_intervals)
    artifacts = {k: v for k, v in hybrid.required_artifacts().items() if v is not None and v.exists()}
    (RESULTS / "run_manifest.json").write_text(
        json.dumps(
            {
                "timestamp": utc_now(),
                "culture_count": int(len(manifest)),
                "n_knots": args.knots,
                "max_intervals": args.max_intervals,
                "restarts": args.restarts,
                "iterations": args.iterations,
                "population": args.population,
                "reporter_weights": args.reporter_weights,
                "lockbox_columns": LOCKBOX_COLUMNS,
                "artifact_sha256": {k: sha256(v) for k, v in artifacts.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_path, augmented = (None, None) if args.fake_solver else capacity.load_augmented(None)
    solver_fn = fake_solver if args.fake_solver else None
    run_rows, metric_rows, solution_rows, loss_rows, solver_rows, nonid_rows = [], [], [], [], [], []
    all_traj = []
    for culture in manifest.itertuples(index=False):
        culture_s = pd.Series(culture._asdict())
        # Oracle and naive references.
        oracle_params = oracle_params_for_culture(dataset, culture_s, bounds, args.max_intervals, args.knots)
        ref_params = {
            "mean_training_controller": mean_controller_params(dataset, bounds, args.max_intervals, args.knots),
        }
        lo, hi = param_bounds(bounds, args.knots)
        rng = np.random.default_rng(args.seed + int(culture_s["culture_index"]))
        ref_params["random_valid_controller"] = rng.uniform(lo, hi)
        ref_params["perturbed_true_controller"] = np.clip(oracle_params + rng.normal(0, 0.30, size=len(lo)) * (hi - lo), lo, hi)
        reference_scores = {}
        oracle_controls = true_controls_for_culture(dataset, culture_s, args.max_intervals)
        loss, parts, traj, _flux, accounting = evaluate_control_dict(
            oracle_controls, culture_s, dataset, scales, args.max_intervals, 0.0, source_path, augmented, solver_fn
        )
        reference_scores["oracle_true_controller"] = loss
        run_rows.append({"culture_id": culture_s["culture_id"], "fit_type": "oracle_true_controller", "restart": -1, "reporter_weight": 0.0, "best_loss": loss, "n_evaluations": 1, "oracle_uses_exact_recorded_controller": True})
        loss_rows.append({"culture_id": culture_s["culture_id"], "fit_type": "oracle_true_controller", "restart": -1, "evaluation": 1, **parts})
        solver_rows.append({"culture_id": culture_s["culture_id"], "fit_type": "oracle_true_controller", "restart": -1, **accounting})
        all_traj.append(traj.assign(culture_id=culture_s["culture_id"], fit_type="oracle_true_controller", restart=-1))
        for channel, vals in oracle_controls.items():
            solution_rows.append({"culture_id": culture_s["culture_id"], "fit_type": "oracle_true_controller", "restart": -1, "control": channel, "values_json": json.dumps([float(x) for x in vals]), "posthoc_true_controller_distance": 0.0})
        for name, params in ref_params.items():
            loss, parts, traj, _flux, controls, accounting = evaluate_controller(
                params, culture_s, dataset, bounds, scales, args.max_intervals, args.knots, 0.0, source_path, augmented, solver_fn
            )
            reference_scores[name] = loss
            run_rows.append({"culture_id": culture_s["culture_id"], "fit_type": name, "restart": -1, "reporter_weight": 0.0, "best_loss": loss, "n_evaluations": 1})
            loss_rows.append({"culture_id": culture_s["culture_id"], "fit_type": name, "restart": -1, "evaluation": 1, **parts})
            solver_rows.append({"culture_id": culture_s["culture_id"], "fit_type": name, "restart": -1, **accounting})
            all_traj.append(traj.assign(culture_id=culture_s["culture_id"], fit_type=name, restart=-1))
            for channel, vals in controls.items():
                solution_rows.append({"culture_id": culture_s["culture_id"], "fit_type": name, "restart": -1, "control": channel, "values_json": json.dumps([float(x) for x in vals]), "posthoc_true_controller_distance": controller_distance(params, oracle_params, bounds, args.knots)})
        for reporter_weight in args.reporter_weights:
            fit_type = "product_biomass_only" if reporter_weight == 0 else f"product_biomass_reporters_w{reporter_weight:g}"
            assert_no_lockbox_usage(ENV_COLUMNS, ["B_total", "X"] + ([] if reporter_weight == 0 else REPORTER_COLUMNS), [])
            restarts = []
            for r in range(args.restarts):
                best = optimize_controller(
                    culture_s,
                    dataset,
                    bounds,
                    scales,
                    args.max_intervals,
                    args.knots,
                    reporter_weight,
                    args.seed + 1000 * r + int(culture_s["culture_index"]),
                    args.iterations,
                    args.population,
                    source_path,
                    augmented,
                    solver_fn,
                )
                restarts.append(best)
                parts = best["parts"]
                run_rows.append({"culture_id": culture_s["culture_id"], "fit_type": fit_type, "restart": r, "reporter_weight": reporter_weight, "best_loss": best["loss"], "n_evaluations": best["evaluations"], "best_iteration": best["iteration"]})
                loss_rows.extend(best["history"].assign(culture_id=culture_s["culture_id"], fit_type=fit_type, restart=r, reporter_weight=reporter_weight).to_dict("records"))
                solver_rows.append({"culture_id": culture_s["culture_id"], "fit_type": fit_type, "restart": r, **best["accounting"]})
                all_traj.append(best["traj"].assign(culture_id=culture_s["culture_id"], fit_type=fit_type, restart=r))
                for channel, vals in best["controls"].items():
                    solution_rows.append({"culture_id": culture_s["culture_id"], "fit_type": fit_type, "restart": r, "control": channel, "values_json": json.dumps([float(x) for x in vals]), "posthoc_true_controller_distance": controller_distance(best["params"], oracle_params, bounds, args.knots)})
                metric_rows.append(
                    {
                        "culture_id": culture_s["culture_id"],
                        "fit_type": fit_type,
                        "restart": r,
                        "split": culture_s["split"],
                        "reporter_weight": reporter_weight,
                        "improvement_vs_mean_controller": reference_scores["mean_training_controller"] - float(best["loss"]),
                        **parts,
                    }
                )
            for a in range(len(restarts)):
                for b in range(a + 1, len(restarts)):
                    loss_gap = abs(float(restarts[a]["loss"]) - float(restarts[b]["loss"]))
                    param_dist = controller_distance(restarts[a]["params"], restarts[b]["params"], bounds, args.knots)
                    phenotype_dist = float(np.sqrt(np.mean((restarts[a]["traj"]["B_pred"].to_numpy(float) - restarts[b]["traj"]["B_pred"].to_numpy(float)) ** 2)))
                    nonid_rows.append({"culture_id": culture_s["culture_id"], "fit_type": fit_type, "restart_a": a, "restart_b": b, "loss_gap": loss_gap, "controller_distance": param_dist, "product_phenotype_rmse_between_solutions": phenotype_dist, "functionally_equivalent_different_controller": bool(loss_gap < 0.05 and param_dist > 0.10)})
    runs = pd.DataFrame(run_rows)
    metrics = pd.DataFrame(metric_rows)
    solutions = pd.DataFrame(solution_rows)
    losses = pd.DataFrame(loss_rows)
    solver = pd.DataFrame(solver_rows)
    nonid = pd.DataFrame(nonid_rows)
    traj = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()
    decision = decision_table(metrics, runs, nonid)
    runs.to_csv(DATA / "inverse_gsm_calibration_runs.csv", index=False)
    metrics.to_csv(DATA / "inverse_gsm_calibration_metrics.csv", index=False)
    solutions.to_csv(DATA / "inverse_gsm_calibration_controller_solutions.csv", index=False)
    nonid.to_csv(DATA / "inverse_gsm_calibration_nonidentifiability.csv", index=False)
    losses.to_csv(DATA / "inverse_gsm_calibration_loss_breakdown.csv", index=False)
    solver.to_csv(DATA / "inverse_gsm_calibration_solver_accounting.csv", index=False)
    decision.to_csv(DATA / "inverse_gsm_calibration_decision.csv", index=False)
    traj.to_csv(DATA / "inverse_gsm_calibration_reconstruction_trajectories.csv", index=False)
    save_figures(metrics, runs, nonid)
    update_readme_snippet(decision)
    return {"runs": runs, "metrics": metrics, "solutions": solutions, "losses": losses, "solver": solver, "nonidentifiability": nonid, "decision": decision}


def fake_solver(_control, _cfg, k):
    return {
        "beta_carotene_flux": 0.10 + 0.003 * k,
        "biomass_flux": 0.20,
        "n_growth_optimizations": 1,
        "n_product_optimizations": 1,
        "n_pfba_optimizations": 1,
        "n_actual_lp_solves": 3,
        "n_optimal_solves": 3,
        "n_infeasible_solves": 0,
        "n_unbounded_solves": 0,
        "n_surrogate_evaluations": 0,
    }


def decision_table(metrics: pd.DataFrame, runs: pd.DataFrame, nonid: pd.DataFrame) -> pd.DataFrame:
    pb = metrics[metrics["fit_type"].eq("product_biomass_only")]
    rep = metrics[metrics["fit_type"].str.contains("reporters", na=False)]
    mean_ref = runs[runs["fit_type"].eq("mean_training_controller")]["best_loss"].mean()
    pb_loss = pb["total_loss"].mean() if not pb.empty else np.nan
    rep_loss = rep.groupby("reporter_weight")["total_loss"].mean().min() if not rep.empty else np.nan
    improves = np.isfinite(pb_loss) and np.isfinite(mean_ref) and pb_loss < mean_ref
    ambiguity = bool((nonid["functionally_equivalent_different_controller"].mean() > 0) if not nonid.empty else False)
    rows = [
        ("Can real Yeast9 be inverse-calibrated from product + biomass?", "YES" if improves else "NO_OR_UNCLEAR", f"mean controller loss={mean_ref:.3f}; inverse loss={pb_loss:.3f}"),
        ("Do inferred controls substantially improve reconstruction over naive controls?", "YES" if improves else "NO", f"loss reduction={mean_ref - pb_loss:.3f}" if np.isfinite(pb_loss) and np.isfinite(mean_ref) else "not available"),
        ("Does calibration generalize across 5 different environments?", "PARTIAL" if metrics["culture_id"].nunique() >= 5 and improves else "NOT_TESTED_FULLY", f"cultures fitted={metrics['culture_id'].nunique() if not metrics.empty else 0}"),
        ("Are multiple controller solutions functionally equivalent?", "YES" if ambiguity else "NO_OR_UNCLEAR", "restart pair ambiguity analysis"),
        ("Do reporters improve observable reconstruction?", "YES" if np.isfinite(rep_loss) and rep_loss < pb_loss else "NO_OR_UNCLEAR", f"best reporter-weight loss={rep_loss:.3f}; product/biomass loss={pb_loss:.3f}"),
        ("Do reporters reduce controller ambiguity?", "UNCLEAR", "requires scaled multi-restart five-culture run"),
        ("Is it reasonable to infer controllers for the full training set?", "NO", "pilot must be scaled before generating pseudo-labels"),
    ]
    return pd.DataFrame([{"question": q, "result": r, "evidence": e} for q, r, e in rows])


def save_figures(metrics: pd.DataFrame, runs: pd.DataFrame, nonid: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    simple_bar(FIGURES / "inverse_gsm_calibration_reporter_ablation.svg", "Reporter Weight Ablation", metrics, "fit_type", "total_loss")
    simple_bar(FIGURES / "inverse_gsm_calibration_convergence.svg", "Best Loss By Fit Type", runs, "fit_type", "best_loss")
    simple_bar(FIGURES / "inverse_gsm_calibration_solution_ambiguity.svg", "Controller Ambiguity", nonid, "fit_type", "controller_distance")
    (FIGURES / "inverse_gsm_calibration_reconstruction.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="260"><rect width="100%" height="100%" fill="white"/><text x="30" y="45" font-size="16">Reconstruction trajectories saved in data/inverse_gsm_calibration_reconstruction_trajectories.csv</text></svg>', encoding="utf-8")


def simple_bar(path: Path, title: str, df: pd.DataFrame, group: str, value: str) -> None:
    if df.empty or group not in df or value not in df:
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="260"><text x="30" y="45">{title}: no data</text></svg>', encoding="utf-8")
        return
    sub = df.groupby(group, as_index=False)[value].mean().sort_values(value).head(8)
    ymax = max(float(sub[value].max()), EPS)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="840" height="360" viewBox="0 0 840 360"><rect width="100%" height="100%" fill="white"/><text x="420" y="30" text-anchor="middle" font-size="17" font-weight="bold">{title}</text>']
    for i, row in enumerate(sub.itertuples(index=False)):
        x = 70 + i * 90
        val = float(getattr(row, value))
        h = 230 * val / ymax
        body.append(f'<rect x="{x}" y="{295-h:.1f}" width="46" height="{h:.1f}" fill="#4f7cac"/>')
        body.append(f'<text x="{x+23}" y="330" text-anchor="middle" font-size="8">{str(getattr(row, group)).replace("_", " ")}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def _reference_losses(reference: pd.DataFrame, random_baseline: pd.DataFrame) -> dict[str, float]:
    vals = {
        "oracle": np.nan,
        "training_mean": np.nan,
        "best_random": np.nan,
        "random_median": np.nan,
    }
    if not reference.empty:
        for name, key in [("oracle_true_controller", "oracle"), ("training_mean_controller", "training_mean")]:
            sub = reference[reference["reference"].eq(name)]
            if not sub.empty:
                vals[key] = float(sub["loss"].iloc[0])
    if not random_baseline.empty:
        vals["best_random"] = float(random_baseline["loss"].min())
        vals["random_median"] = float(random_baseline["loss"].median())
    return vals


def save_stage_a_figures(
    convergence: pd.DataFrame,
    optimizer_summary: pd.DataFrame,
    landscape: pd.DataFrame,
    reference: pd.DataFrame,
    random_baseline: pd.DataFrame,
) -> None:
    FIGURES.mkdir(exist_ok=True)
    refs = _reference_losses(reference, random_baseline)
    if convergence.empty:
        (FIGURES / "inverse_gsm_stageA_optimizer_convergence.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="260"><text x="30" y="45">No convergence data</text></svg>', encoding="utf-8")
    else:
        body = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420"><rect width="100%" height="100%" fill="white"/><text x="450" y="30" text-anchor="middle" font-size="17" font-weight="bold">Stage A Optimizer Convergence</text>']
        sub = convergence[convergence["reporter_weight"].eq(0.0)].copy()
        ref_values = [v for v in refs.values() if np.isfinite(v)]
        ymax = max(float(sub["best_loss_so_far"].max()), max(ref_values) if ref_values else EPS, EPS)
        x_axis = "cumulative_LP_solve_count" if "cumulative_LP_solve_count" in sub else "evaluation_number"
        xmax = max(float(sub[x_axis].max()), 1.0)
        colors = {"differential_evolution": "#4f7cac", "powell_multistart": "#6b9b6b", "annealing": "#b2764f", "random_search": "#8f6bb2"}
        for (opt, restart), g in sub.groupby(["optimizer", "restart"]):
            pts = []
            for row in g.itertuples():
                xval = float(getattr(row, x_axis))
                x = 70 + xval / xmax * 760
                y = 350 - float(row.best_loss_so_far) / ymax * 285
                pts.append(f"{x:.1f},{y:.1f}")
            body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors.get(opt, "#555")}" stroke-width="1.4" opacity="0.8"/>')
        ref_colors = {"oracle": "#222222", "training_mean": "#d1495b", "best_random": "#edae49", "random_median": "#777777"}
        for label, value in refs.items():
            if np.isfinite(value):
                y = 350 - value / ymax * 285
                body.append(f'<line x1="70" x2="830" y1="{y:.1f}" y2="{y:.1f}" stroke="{ref_colors[label]}" stroke-width="1" stroke-dasharray="5,4"/>')
                body.append(f'<text x="835" y="{y+3:.1f}" font-size="9" fill="{ref_colors[label]}">{label}</text>')
        body.append(f'<text x="450" y="395" text-anchor="middle" font-size="11">Exact cumulative Yeast9 LP solves</text>')
        body.append("</svg>")
        (FIGURES / "inverse_gsm_stageA_optimizer_convergence.svg").write_text("\n".join(body), encoding="utf-8")
    random_rows = random_baseline.assign(source="random", plot_loss=random_baseline["loss"]) if not random_baseline.empty else pd.DataFrame()
    inverse_rows = optimizer_summary[optimizer_summary["reporter_weight"].eq(0.0)].rename(columns={"best_loss": "plot_loss"}).assign(source="inverse")
    plot = pd.concat([random_rows[["source", "plot_loss"]], inverse_rows[["source", "plot_loss"]]], ignore_index=True) if not inverse_rows.empty or not random_rows.empty else pd.DataFrame()
    simple_bar(FIGURES / "inverse_gsm_stageA_random_vs_inverse.svg", "Stage A Random Vs Inverse", plot, "source", "plot_loss")
    simple_bar(FIGURES / "inverse_gsm_stageA_controller_distance_vs_loss.svg", "Stage A Distance Vs Loss", optimizer_summary, "optimizer", "posthoc_controller_distance")
    simple_bar(FIGURES / "inverse_gsm_stageA_landscape.svg", "Stage A 1D Landscape", landscape, "control", "loss")


def stage_a_decision_table(
    gate: pd.DataFrame,
    optimizer_summary: pd.DataFrame,
    reproducibility: pd.DataFrame | None = None,
    landscape_sensitivity: pd.DataFrame | None = None,
    cost: pd.DataFrame | None = None,
) -> pd.DataFrame:
    first = gate.iloc[0] if not gate.empty else {}
    passed = bool(gate["passed"].all()) if not gate.empty else False
    best = optimizer_summary.sort_values("best_loss").head(1)
    if best.empty:
        best_loss = np.nan
        closure = np.nan
        best_optimizer = ""
    else:
        best_loss = float(best["best_loss"].iloc[0])
        closure = float(best["fractional_closure_toward_oracle"].iloc[0])
        best_optimizer = str(best["optimizer"].iloc[0])
    rep = reproducibility if reproducibility is not None else pd.DataFrame()
    if rep.empty:
        best_rep = {}
        best_fraction = np.nan
        median_closure = np.nan
    else:
        best_rep = rep.sort_values(["median_best_loss", "mean_best_loss"]).iloc[0].to_dict()
        best_fraction = float(best_rep["fraction_beating_training_mean"])
        median_closure = float(best_rep["median_oracle_gap_closure"])
    weak_dims = 0
    flat_dims = 0
    if landscape_sensitivity is not None and not landscape_sensitivity.empty:
        posthoc = landscape_sensitivity[landscape_sensitivity["context"].eq("posthoc_oracle_context")]
        weak_dims = int(posthoc["identifiability_classification"].isin(["flat", "weak"]).sum())
        flat_dims = int(posthoc["identifiability_classification"].eq("flat").sum())
    acceptable = optimizer_summary[
        (optimizer_summary["reporter_weight"].eq(0.0))
        & (optimizer_summary["best_loss"] < float(first.get("training_mean_loss", np.inf)))
    ]
    if acceptable.empty:
        uniqueness_result = "UNRESOLVED"
        uniqueness_evidence = "no reproducibly acceptable set to compare"
    else:
        high_distance = int((acceptable["posthoc_controller_distance"] > 0.25).sum())
        uniqueness_result = "NO" if high_distance else "NO_OR_UNCLEAR"
        uniqueness_evidence = f"{len(acceptable)} acceptable fits; {high_distance} have normalized controller distance >0.25"
    if cost is None or cost.empty:
        cost_result = "NOT_ESTIMATED"
        cost_evidence = "cost table unavailable"
    else:
        serial_hours = float(cost[cost["parallelism"].eq(1)]["estimated_77_culture_wall_seconds"].iloc[0]) / 3600.0
        p16_hours = float(cost[cost["parallelism"].eq(16)]["estimated_77_culture_wall_seconds"].iloc[0]) / 3600.0
        cost_result = "PLANNING_ONLY"
        cost_evidence = f"77-culture estimate: serial {serial_hours:.1f} h; 16-way {p16_hours:.1f} h"
    rows = [
        ("Does oracle replay remain near exact?", "YES" if float(first.get("oracle_loss", np.inf)) < 1e-6 else "NO", f"oracle loss={float(first.get('oracle_loss', np.nan)):.3g}"),
        ("Are random controllers generally poor?", "YES" if bool(gate[gate["criterion"].eq("random_controllers_generally_poor")]["passed"].all()) else "NO", f"random median={float(first.get('random_median_loss', np.nan)):.3g}; mean={float(first.get('training_mean_loss', np.nan)):.3g}"),
        ("Does inverse calibration beat training mean?", "YES" if bool(gate[gate["criterion"].eq("optimizer_beats_training_mean")]["passed"].all()) else "NO", f"mean={float(first.get('training_mean_loss', np.nan)):.3g}; best={best_loss:.3g}"),
        ("Is improvement reproducible across seeds?", "YES" if bool(gate[gate["criterion"].eq("optimizer_reproducible_across_restarts")]["passed"].all()) else "NO_OR_UNCLEAR", f"best median optimizer={best_rep.get('optimizer', best_optimizer)}; fraction beating mean={best_fraction:.2f}"),
        ("Which optimizer is best by observable loss?", best_optimizer or "NONE", f"best loss={best_loss:.3g}; median closure={median_closure:.3f}"),
        ("How much of the oracle gap is closed?", "QUANTIFIED", f"best closure={closure:.3f}; median closure={median_closure:.3f}"),
        ("Are acceptable solutions unique?", uniqueness_result, uniqueness_evidence),
        ("Are some controller dimensions weakly identifiable?", "YES" if weak_dims else "NO_OR_UNCLEAR", f"{weak_dims} weak/flat post-hoc dimensions; {flat_dims} flat"),
        ("Is computational cost practical for multi-culture fitting?", cost_result, cost_evidence),
        ("Proceed to Stage B?", "YES" if passed else "NO", "Stage A gate passed" if passed else "Stage A gate did not pass"),
    ]
    return pd.DataFrame([{"question": q, "result": r, "evidence": e} for q, r, e in rows])


def write_stage_a_summary(decision: pd.DataFrame, gate: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Inverse GSM Calibration Stage A",
        "",
        "Stage A tests single-interval inverse calibration using the real Yeast9/pFBA backend.",
        "",
        f"Budget: `{args.max_evaluations}` evaluations per optimizer/restart; random baseline samples: `{args.random_samples}`.",
        "",
        "| Question | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in decision.itertuples(index=False):
        lines.append(f"| {row.question} | {row.result} | {row.evidence} |")
    lines.extend(["", "## Stage A Gate", "", "| Criterion | Passed | Value |", "| --- | --- | ---: |"])
    for row in gate.itertuples(index=False):
        lines.append(f"| {row.criterion} | {row.passed} | {row.value} |")
    (RESULTS / "inverse_gsm_stageA_summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_readme_snippet(decision: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Inverse GSM Calibration",
        "",
        "This diagnostic fits compact controller trajectories directly to observable product/biomass/reporter measurements using derivative-free optimization and the real Yeast9/pFBA backend.",
        "",
        "| Question | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in decision.itertuples(index=False):
        lines.append(f"| {row.question} | {row.result} | {row.evidence} |")
    lines.extend(
        [
            "",
            "The true controller remains a lockbox during fitting and is only used after solutions are frozen for diagnostic interpretation.",
        ]
    )
    (RESULTS / "inverse_gsm_calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", action="store_true", help="Run serious single-interval Stage A inverse calibration.")
    parser.add_argument("--central-only", action="store_true")
    parser.add_argument("--max-intervals", type=int, default=2)
    parser.add_argument("--knots", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--reporter-weights", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--stage-a-reporter-weights", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--optimizers", nargs="+", default=["differential_evolution", "powell_multistart", "annealing"])
    parser.add_argument("--restart-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--landscape-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--fake-solver", action="store_true", help="Testing only; bypasses Yeast9 with a deterministic fake solver.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    outputs = run_stage_a(cli_args) if cli_args.stage_a else run(cli_args)
    print(outputs["decision"].to_string(index=False))
