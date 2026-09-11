#!/usr/bin/env python3
"""Continuous atlas, transition-boundary search, and inverse design for ODE regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_regime_core as ode


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "ode_regime_discovery"
OUTPUTS = ["P", "X", "A", "R", "E", "S", "C", "v_product", "mu"]


def load(prefix: str):
    traj = pd.read_csv(DATA / f"{prefix}_trajectories.csv")
    diag = pd.read_csv(DATA / f"{prefix}_diagnostics.csv")
    meta, mats, t = ode.pivot_trajectories(traj, OUTPUTS)
    return meta, mats, t, diag


def fit_surrogate(meta: pd.DataFrame, mats: dict[str, np.ndarray]):
    env = meta[["temperature", "pH", "DO"]].to_numpy(float)
    train = meta["split"].eq("train").to_numpy()
    Y = np.hstack([mats[col] for col in OUTPUTS])
    mean = Y[train].mean(axis=0)
    std = Y[train].std(axis=0) + ode.EPS
    X = ode.polynomial_features(env, degree=3)
    coef = ode.ridge_fit(X[train], (Y[train] - mean) / std, lam=1e-4)
    return {"coef": coef, "mean": mean, "std": std, "n_time": mats["P"].shape[1]}


def predict_surrogate(model: dict, env: np.ndarray) -> dict[str, np.ndarray]:
    env = np.asarray(env, dtype=float).reshape(-1, 3)
    Y = ode.polynomial_features(env, degree=3) @ model["coef"] * model["std"] + model["mean"]
    out = {}
    n = model["n_time"]
    start = 0
    for col in OUTPUTS:
        out[col] = Y[:, start : start + n]
        start += n
    for col in ["P", "X", "A", "R", "E", "S", "C", "v_product", "mu"]:
        out[col] = np.clip(out[col], 0.0, None)
    return out


def diagnostics_from_prediction(model: dict, env: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    pred = predict_surrogate(model, env)
    rows = []
    for i, e in enumerate(np.asarray(env).reshape(-1, 3)):
        df = pd.DataFrame({"time_index": np.arange(len(t)), "time": t, "temperature": e[0], "pH": e[1], "DO": e[2]})
        for col in OUTPUTS:
            df[col] = pred[col][i]
        d = ode.regime_diagnostics(df)
        d["dominant_regime"] = ode.assign_regime(d)
        rows.append({"temperature": e[0], "pH": e[1], "DO": e[2], **d})
    return pd.DataFrame(rows)


def objective_value(diag: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "max_endpoint_product":
        return diag["final_product"].to_numpy(float)
    if objective == "max_total_productivity":
        return diag["auc_product"].to_numpy(float)
    if objective == "sustained_production":
        return (
            diag["final_product"].to_numpy(float)
            + 0.015 * diag["auc_product"].to_numpy(float)
            - 2.5 * diag["late_decline"].to_numpy(float)
            - 0.010 * diag["integrated_stress"].to_numpy(float)
        )
    if objective == "physiologically_constrained":
        feasible = (
            (diag["integrated_stress"] <= 15.0)
            & (diag["min_energy"] >= 0.16)
            & (diag["final_pathway_capacity"] >= 0.080)
            & (diag["endpoint_biomass"] >= 0.050)
        )
        penalty = (
            0.10 * np.maximum(0.0, diag["integrated_stress"].to_numpy(float) - 15.0)
            + 3.0 * np.maximum(0.0, 0.16 - diag["min_energy"].to_numpy(float))
            + 2.0 * np.maximum(0.0, 0.080 - diag["final_pathway_capacity"].to_numpy(float))
            + 1.5 * np.maximum(0.0, 0.050 - diag["endpoint_biomass"].to_numpy(float))
        )
        return diag["final_product"].to_numpy(float) - penalty + feasible.to_numpy(float) * 0.05
    if objective == "robust_sustained":
        base = objective_value(diag, "sustained_production")
        return base
    raise ValueError(objective)


def random_search(model: dict, t: np.ndarray, n: int, seed: int, objective: str) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unit = rng.uniform(0.0, 1.0, size=(n, 3))
    env = ode.unit_to_env(unit)
    diag = diagnostics_from_prediction(model, env, t)
    if objective == "robust_sustained":
        vals = []
        for e in env:
            local = np.clip(ode.env_to_unit(e[None, :]) + rng.normal(0, [0.025, 0.025, 0.025], size=(15, 3)), 0, 1)
            ldiag = diagnostics_from_prediction(model, ode.unit_to_env(local), t)
            score = objective_value(ldiag, "sustained_production")
            vals.append(float(score.mean() - 0.7 * score.std()))
        diag["objective_value"] = vals
    else:
        diag["objective_value"] = objective_value(diag, objective)
    best = diag.sort_values("objective_value", ascending=False).iloc[0]
    return best[["temperature", "pH", "DO"]].to_numpy(float), diag


def gradient_search(model: dict, t: np.ndarray, seed: int, objective: str) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_env = None
    best_val = -np.inf
    starts = rng.uniform(0.03, 0.97, size=(14, 3))
    for u0 in starts:
        u = u0.copy()
        step = 0.045
        for _ in range(70):
            env = ode.unit_to_env(u[None, :])[0]
            def f(e):
                return objective_value(diagnostics_from_prediction(model, np.asarray(e)[None, :], t), objective if objective != "robust_sustained" else "sustained_production")[0]
            grad_env = ode.finite_difference_sensitivity(f, env)
            grad_u = grad_env * np.array([6.0, 1.0, 60.0])
            norm = np.linalg.norm(grad_u)
            if norm > ode.EPS:
                u = np.clip(u + step * grad_u / norm, 0.0, 1.0)
            step *= 0.975
        env = ode.unit_to_env(u[None, :])[0]
        val = float(objective_value(diagnostics_from_prediction(model, env[None, :], t), objective if objective != "robust_sustained" else "sustained_production")[0])
        if val > best_val:
            best_env, best_val = env, val
    return best_env, best_val


def verify_candidate(env: np.ndarray) -> dict:
    traj, diag = ode.simulate_environment(env)
    return {"verified_" + k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))}


def nearest_training_distance(env: np.ndarray, meta: pd.DataFrame) -> float:
    train = meta[meta["split"].eq("train")][["temperature", "pH", "DO"]].to_numpy(float)
    x = ode.normalize_env_raw(train)
    y = ode.normalize_env_raw(np.asarray(env).reshape(1, 3))
    return float(np.sqrt(np.sum((x - y) ** 2, axis=1)).min())


def inverse_design(prefix: str, model: dict, meta: pd.DataFrame, mats: dict[str, np.ndarray], t: np.ndarray, fast: bool) -> pd.DataFrame:
    objectives = ["max_endpoint_product", "max_total_productivity", "sustained_production", "physiologically_constrained", "robust_sustained"]
    rows = []
    train_diag = diagnostics_from_prediction(model, meta[["temperature", "pH", "DO"]].to_numpy(float), t)
    train_diag["split"] = meta["split"].to_numpy(str)
    for i, objective in enumerate(objectives):
        n = 1200 if fast else 6000
        rand_env, rand_diag = random_search(model, t, n, seed=700 + i, objective=objective)
        grad_env, grad_score = gradient_search(model, t, seed=800 + i, objective=objective)
        best_train = train_diag[train_diag["split"].eq("train")].assign(objective_value=lambda d: objective_value(d, objective if objective != "robust_sustained" else "sustained_production")).sort_values("objective_value", ascending=False).iloc[0]
        for source, env in [("best_training_condition", best_train[["temperature", "pH", "DO"]].to_numpy(float)), ("dense_random_search", rand_env), ("gradient_inverse_design", grad_env)]:
            pred_diag = diagnostics_from_prediction(model, env[None, :], t).iloc[0].to_dict()
            verified = verify_candidate(env)
            rows.append(
                {
                    "objective": objective,
                    "proposal_source": source,
                    "temperature": float(env[0]),
                    "pH": float(env[1]),
                    "DO": float(env[2]),
                    "predicted_objective": float(objective_value(pd.DataFrame([pred_diag]), objective if objective != "robust_sustained" else "sustained_production")[0]),
                    "nearest_training_distance": nearest_training_distance(env, meta),
                    "is_exact_training_duplicate": nearest_training_distance(env, meta) < 1e-8,
                    **pred_diag,
                    **verified,
                }
            )
    return pd.DataFrame(rows)


def boundary_search(model: dict, t: np.ndarray, fast: bool) -> pd.DataFrame:
    levels = np.linspace(0.0, 1.0, 9 if fast else 15)
    unit = np.array(np.meshgrid(levels, levels, levels)).T.reshape(-1, 3)
    env = ode.unit_to_env(unit)
    diag = diagnostics_from_prediction(model, env, t)
    code = diag["dominant_regime"].to_numpy(str)
    rows = []
    shape = (len(levels), len(levels), len(levels))
    idx = np.arange(len(unit)).reshape(shape)
    for axis in range(3):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(0, -1)
        slicer_b[axis] = slice(1, None)
        for ia, ib in zip(idx[tuple(slicer_a)].reshape(-1), idx[tuple(slicer_b)].reshape(-1)):
            if code[ia] == code[ib]:
                continue
            mid = 0.5 * (env[ia] + env[ib])
            left_true = verify_candidate(env[ia])
            mid_true = verify_candidate(mid)
            right_true = verify_candidate(env[ib])
            rows.append(
                {
                    "axis": ["temperature", "pH", "DO"][axis],
                    "left_predicted_regime": code[ia],
                    "right_predicted_regime": code[ib],
                    "temperature": float(mid[0]),
                    "pH": float(mid[1]),
                    "DO": float(mid[2]),
                    "verified_left_regime": left_true["verified_dominant_regime"],
                    "verified_mid_regime": mid_true["verified_dominant_regime"],
                    "verified_right_regime": right_true["verified_dominant_regime"],
                    "simulator_confirms_qualitative_change": bool(left_true["verified_dominant_regime"] != right_true["verified_dominant_regime"]),
                    "responsible_quantity_hint": "stress/energy/precursor/congestion diagnostics saved for each verified point",
                }
            )
            if len(rows) >= (30 if fast else 120):
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def sensitivity_table(model: dict, design: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    rows = []
    selected = design[design["proposal_source"].isin(["gradient_inverse_design", "dense_random_search"])].head(10)
    outputs = ["final_product", "integrated_stress", "min_energy", "final_precursor", "max_congestion"]
    for r in selected.itertuples(index=False):
        env = np.array([r.temperature, r.pH, r.DO], dtype=float)
        for out in outputs:
            pred_fn = lambda e, out=out: diagnostics_from_prediction(model, np.asarray(e)[None, :], t).iloc[0][out]
            sim_fn = lambda e, out=out: ode.simulate_environment(np.asarray(e))[1][out]
            gp = ode.finite_difference_sensitivity(pred_fn, env)
            gs = ode.finite_difference_sensitivity(sim_fn, env)
            rows.append({"objective": r.objective, "quantity": out, "temperature": env[0], "pH": env[1], "DO": env[2], "pred_dT": gp[0], "pred_dpH": gp[1], "pred_dDO": gp[2], "sim_dT": gs[0], "sim_dpH": gs[1], "sim_dDO": gs[2]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    prefix = "ode_regime_pilot" if args.pilot else "ode_regime"
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta, mats, t, _diag = load(prefix)
    model = fit_surrogate(meta, mats)
    design = inverse_design(prefix, model, meta, mats, t, fast=args.fast)
    design.to_csv(DATA / f"{prefix}_inverse_design_verification.csv", index=False)
    boundary = boundary_search(model, t, fast=args.fast)
    boundary.to_csv(DATA / f"{prefix}_transition_boundary_verification.csv", index=False)
    sensitivity_table(model, design, t).to_csv(DATA / f"{prefix}_local_sensitivity.csv", index=False)
    atlas_unit = np.array(np.meshgrid(np.linspace(0, 1, 11 if args.fast else 21), np.linspace(0, 1, 11 if args.fast else 21), np.linspace(0, 1, 11 if args.fast else 21))).T.reshape(-1, 3)
    atlas = diagnostics_from_prediction(model, ode.unit_to_env(atlas_unit), t)
    atlas.to_csv(DATA / f"{prefix}_continuous_regime_atlas.csv", index=False)
    print(f"{prefix} inverse design complete")
    print(design[["objective", "proposal_source", "temperature", "pH", "DO", "predicted_objective", "verified_dominant_regime", "nearest_training_distance"]].to_string(index=False))
    print(boundary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
