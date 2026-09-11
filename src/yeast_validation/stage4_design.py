#!/usr/bin/env python3
"""Stage 4 probabilistic strain-environment design utilities.

This module implements the FBA-sparing Stage 4 scaffold while keeping the
formal acceptance gate tied to Stage 3 completion.  It reuses the existing
baseline Yeast9 data, frozen hybrid controls, and exact replay manifests; exact
edited-strain verification is deliberately reported as pending until Stage 3
passes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import gem_backend as gem
import run_reporter_grounded_hybrid_distillation as hybrid


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "stage4_probabilistic_design"
FIGURES = ROOT / "figures"
EPS = 1e-9

EDIT_COLUMNS = [
    "competing_sink_multiplier",
    "precursor_supply_multiplier",
    "PSY_capacity_multiplier",
    "DES_capacity_multiplier",
    "CYC_capacity_multiplier",
    "ATP_support_multiplier",
    "oxygen_support_multiplier",
    "export_capacity_multiplier",
    "glucose_uptake_multiplier",
    "product_degradation_multiplier",
]

SEARCH_OBJECTIVES = [
    "expected_production",
    "robust_production",
    "lower_confidence_production",
    "low_stress_production",
    "low_cost_production",
    "sustained_production",
]

FLUX_TARGETS = [
    "beta_carotene_flux",
    "biomass_flux",
    "oxygen_uptake",
    "ggpp_flux",
    "atp_maintenance_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
]


@dataclass(frozen=True)
class Stage3Gate:
    passed: bool
    status: str
    complete_pairs: int
    complete_culture_triples: int
    reason: str


def stage3_gate() -> Stage3Gate:
    acceptance_path = DATA / "hybrid_student_acceptance.csv"
    completion_path = DATA / "hybrid_exact_replay_completion.csv"
    status = "missing_stage3_acceptance"
    if acceptance_path.exists():
        acceptance = pd.read_csv(acceptance_path)
        if "stage_3_status" in acceptance.columns and not acceptance.empty:
            status = str(acceptance["stage_3_status"].dropna().iloc[-1])
    complete_pairs = 0
    complete_triples = 0
    if completion_path.exists():
        completion = pd.read_csv(completion_path)
        valid = completion[completion["status"].eq("complete_valid")]
        complete_pairs = int(len(valid))
        if not valid.empty:
            sets = [
                set(valid[valid["replay_type"].eq(rt)]["culture_id"])
                for rt in ["oracle", "teacher", "hybrid"]
            ]
            complete_triples = len(set.intersection(*sets)) if all(sets) else 0
    passed = status == "stage_3_passed_ready_for_metabolic_edits" and complete_pairs == 375
    reason = "stage3_passed" if passed else "stage_3_exact_replay_or_acceptance_incomplete"
    return Stage3Gate(passed, status, complete_pairs, complete_triples, reason)


def stage4_inventory() -> pd.DataFrame:
    assets = [
        ("baseline_environment_grid", DATA / "gem_state_space_environment_grid.csv"),
        ("baseline_trajectories", DATA / "gem_state_space_trajectories.csv"),
        ("baseline_fluxes", DATA / "gem_state_space_fluxes.csv"),
        ("baseline_constraints", hybrid.discover_constraints_path()),
        ("baseline_reporters", DATA / "gem_state_space_reporters.csv"),
        ("baseline_states", DATA / "gem_state_space_states.csv"),
        ("teacher_manifest", DATA / "teacher_ensemble_manifest.csv"),
        ("teacher_pseudodata", DATA / "teacher_pseudodata.csv"),
        ("teacher_pseudodata_uncertainty", DATA / "teacher_pseudodata_uncertainty.csv"),
        ("hybrid_student_manifest", DATA / "hybrid_student_manifest.csv"),
        ("exact_replay_manifest", DATA / "hybrid_exact_replay_shard_manifest.csv"),
        ("exact_replay_completion", DATA / "hybrid_exact_replay_completion.csv"),
        ("stage3_acceptance", DATA / "hybrid_student_acceptance.csv"),
    ]
    rows = []
    for role, path in assets:
        exists = bool(path is not None and Path(path).exists())
        rows.append(
            {
                "asset_role": role,
                "path": "" if path is None else str(Path(path).relative_to(ROOT)),
                "exists": exists,
                "reuse_role": "input_reused_no_regeneration",
            }
        )
    gate = stage3_gate()
    rows.append(
        {
            "asset_role": "stage3_gate",
            "path": "data/hybrid_student_acceptance.csv,data/hybrid_exact_replay_completion.csv",
            "exists": gate.complete_pairs > 0,
            "reuse_role": gate.status,
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "stage4_reused_asset_inventory.csv", index=False)
    return out


def default_edit_library() -> pd.DataFrame:
    rows = [
        {
            "edit_id": "competing_sink_multiplier",
            "biological_description": "Reduce aggregate competing consumption of native isoprenoid precursor supply.",
            "gem_reaction_id": "aggregate_native_GGPP_competing_sinks",
            "constraint_mapping": "multiplicative penalty on non-product GGPP drain in edit-aware surrogate; exact mapping requires curated sink list",
            "edit_type": "capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.35,
            "permitted_maximum": 1.0,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.35,
            "assumed_uncertainty": 0.18,
            "transfer_risk_category": "medium",
        },
        {
            "edit_id": "precursor_supply_multiplier",
            "biological_description": "Increase effective GGPP or upstream isoprenoid precursor supply.",
            "gem_reaction_id": gem.NATIVE_GGPP_RXN,
            "constraint_mapping": "multiplier on precursor/GGPP supply response",
            "edit_type": "capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.75,
            "permitted_maximum": 1.80,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.55,
            "assumed_uncertainty": 0.22,
            "transfer_risk_category": "medium",
        },
        {
            "edit_id": "PSY_capacity_multiplier",
            "biological_description": "Increase phytoene synthase pathway capacity.",
            "gem_reaction_id": gem.PSY_RXN,
            "constraint_mapping": "multiplier on PSY effective upper bound",
            "edit_type": "pathway_capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.80,
            "permitted_maximum": 2.00,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.45,
            "assumed_uncertainty": 0.16,
            "transfer_risk_category": "low",
        },
        {
            "edit_id": "DES_capacity_multiplier",
            "biological_description": "Increase phytoene desaturase pathway capacity.",
            "gem_reaction_id": gem.DES_RXN,
            "constraint_mapping": "multiplier on DES effective upper bound",
            "edit_type": "pathway_capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.80,
            "permitted_maximum": 2.00,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.45,
            "assumed_uncertainty": 0.16,
            "transfer_risk_category": "low",
        },
        {
            "edit_id": "CYC_capacity_multiplier",
            "biological_description": "Increase lycopene cyclase pathway capacity.",
            "gem_reaction_id": gem.CYC_RXN,
            "constraint_mapping": "multiplier on CYC effective upper bound",
            "edit_type": "pathway_capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.80,
            "permitted_maximum": 2.00,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.45,
            "assumed_uncertainty": 0.16,
            "transfer_risk_category": "low",
        },
        {
            "edit_id": "ATP_support_multiplier",
            "biological_description": "Improve effective ATP/cofactor support for production burden.",
            "gem_reaction_id": gem.ATPM_RXN,
            "constraint_mapping": "reduces ATP-pressure penalty; exact mapping requires curated cofactor intervention",
            "edit_type": "support_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.85,
            "permitted_maximum": 1.45,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.70,
            "assumed_uncertainty": 0.28,
            "transfer_risk_category": "high",
        },
        {
            "edit_id": "oxygen_support_multiplier",
            "biological_description": "Adjust effective respiratory or oxygen-utilisation support.",
            "gem_reaction_id": gem.OXYGEN_EXCHANGE,
            "constraint_mapping": "modifies oxygen-support penalty and oxygen uptake response",
            "edit_type": "support_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.80,
            "permitted_maximum": 1.35,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.65,
            "assumed_uncertainty": 0.26,
            "transfer_risk_category": "high",
        },
        {
            "edit_id": "export_capacity_multiplier",
            "biological_description": "Increase product export or accumulation capacity without changing the regulatory controller.",
            "gem_reaction_id": gem.PRODUCT_RXN,
            "constraint_mapping": "multiplier on product accumulation/export support in edit-aware surrogate",
            "edit_type": "product_capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.90,
            "permitted_maximum": 1.70,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.50,
            "assumed_uncertainty": 0.20,
            "transfer_risk_category": "medium",
        },
        {
            "edit_id": "glucose_uptake_multiplier",
            "biological_description": "Moderately increase available carbon uptake capacity.",
            "gem_reaction_id": gem.GLUCOSE_EXCHANGE,
            "constraint_mapping": "multiplier on carbon/biomass support; bounded to avoid whole-medium redesign",
            "edit_type": "exchange_capacity_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.85,
            "permitted_maximum": 1.30,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.60,
            "assumed_uncertainty": 0.24,
            "transfer_risk_category": "medium",
        },
        {
            "edit_id": "product_degradation_multiplier",
            "biological_description": "Reduce product loss or degradation pressure.",
            "gem_reaction_id": "beta_carotene_loss_proxy",
            "constraint_mapping": "multiplier on product degradation term during trajectory integration",
            "edit_type": "loss_multiplier",
            "baseline_value": 1.0,
            "permitted_minimum": 0.50,
            "permitted_maximum": 1.0,
            "zero_allowed": False,
            "reversible": True,
            "edit_cost_score": 0.40,
            "assumed_uncertainty": 0.20,
            "transfer_risk_category": "medium",
        },
    ]
    return pd.DataFrame(rows)


def write_edit_library() -> pd.DataFrame:
    out = default_edit_library()
    out.to_csv(DATA / "strain_edit_library.csv", index=False)
    return out


def validate_edit_vector(edit: dict[str, float], library: pd.DataFrame) -> None:
    for row in library.itertuples(index=False):
        value = float(edit.get(row.edit_id, row.baseline_value))
        if value < float(row.permitted_minimum) - EPS or value > float(row.permitted_maximum) + EPS:
            raise ValueError(f"{row.edit_id}={value} is outside permitted bounds.")
        if value == 0 and not bool(row.zero_allowed):
            raise ValueError(f"{row.edit_id} cannot be zero.")


def edit_distance(edit: dict[str, float], library: pd.DataFrame | None = None) -> float:
    lib = default_edit_library() if library is None else library
    validate_edit_vector(edit, lib)
    total = 0.0
    for row in lib.itertuples(index=False):
        value = float(edit.get(row.edit_id, row.baseline_value))
        span = max(float(row.permitted_maximum) - float(row.permitted_minimum), EPS)
        delta = abs(value - float(row.baseline_value)) / span
        risk = {"low": 1.0, "medium": 1.35, "high": 1.8}.get(str(row.transfer_risk_category), 1.5)
        total += delta * float(row.edit_cost_score) * risk
    active = sum(abs(float(edit.get(row.edit_id, row.baseline_value)) - float(row.baseline_value)) > 1e-8 for row in lib.itertuples(index=False))
    return float(total + 0.08 * active)


def edit_cost(edit: dict[str, float], library: pd.DataFrame) -> float:
    return float(
        sum(
            abs(float(edit.get(row.edit_id, row.baseline_value)) - float(row.baseline_value))
            * float(row.edit_cost_score)
            for row in library.itertuples(index=False)
        )
    )


def sample_edit_vectors(library: pd.DataFrame, n: int, seed: int, max_active: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    baseline = {row.edit_id: float(row.baseline_value) for row in library.itertuples(index=False)}
    rows.append({"edit_vector_id": "baseline_no_edit", "edit_class": "baseline", **baseline})
    for row in library.itertuples(index=False):
        for frac, label in [(0.33, "low"), (0.66, "medium"), (1.0, "high")]:
            edit = baseline.copy()
            if float(row.permitted_maximum) <= float(row.baseline_value):
                value = float(row.baseline_value) - frac * (float(row.baseline_value) - float(row.permitted_minimum))
            else:
                value = float(row.baseline_value) + frac * (float(row.permitted_maximum) - float(row.baseline_value))
            edit[row.edit_id] = float(value)
            rows.append({"edit_vector_id": f"{row.edit_id}_{label}", "edit_class": "single_edit", **edit})
    while len(rows) < n:
        edit = baseline.copy()
        active = rng.choice(library["edit_id"].to_numpy(str), size=max_active, replace=False)
        labels = []
        for eid in active:
            r = library[library["edit_id"].eq(eid)].iloc[0]
            lo, hi, base = float(r.permitted_minimum), float(r.permitted_maximum), float(r.baseline_value)
            if hi <= base:
                value = rng.uniform(lo, base)
            else:
                value = rng.uniform(base, hi)
            edit[eid] = float(value)
            labels.append(eid)
        rows.append({"edit_vector_id": "double_" + "_".join(labels) + f"_{len(rows):03d}", "edit_class": "double_edit", **edit})
    out = pd.DataFrame(rows[:n])
    out["n_active_edits"] = [sum(abs(float(row[eid]) - 1.0) > 1e-8 for eid in EDIT_COLUMNS) for _, row in out.iterrows()]
    out["edit_distance"] = [edit_distance(row.to_dict(), library) for _, row in out.iterrows()]
    out["edit_cost"] = [edit_cost(row.to_dict(), library) for _, row in out.iterrows()]
    return out


def environment_summary(dataset: dict[str, object]) -> pd.DataFrame:
    meta = dataset["meta"].reset_index(drop=True).copy()
    product = dataset["product"]
    stress = dataset["states"]["z_ox"].max(axis=1) + dataset["states"]["z_atp"].max(axis=1) + dataset["states"]["z_bottle"].max(axis=1)
    env = dataset["env"]
    center = env.mean(axis=0)
    scale = env.std(axis=0) + EPS
    meta["final_product"] = product[:, -1]
    meta["product_AUC"] = np.trapezoid(product, dataset["time"], axis=1)
    meta["stress_proxy"] = stress
    meta["environment_edge_distance"] = np.sqrt(np.sum(((env - center) / scale) ** 2, axis=1))
    meta["predicted_uncertainty"] = 0.02 + 0.02 * meta["environment_edge_distance"] / max(float(meta["environment_edge_distance"].max()), EPS)
    return meta


def select_calibration_manifest(n: int = 60, seed: int = 4401) -> pd.DataFrame:
    library = write_edit_library()
    dataset = hybrid.load_distillation_dataset()
    envs = environment_summary(dataset)
    edit_vectors = sample_edit_vectors(library, max(35, n), seed=seed)
    rng = np.random.default_rng(seed)
    chosen_envs = []
    for split in ["train", "validation", "interpolation", "heldout_combination", "extrapolation"]:
        sub = envs[envs["split"].eq(split)]
        if sub.empty:
            continue
        for col, reason in [("final_product", "product_coverage"), ("stress_proxy", "stress_coverage"), ("environment_edge_distance", "edge_coverage")]:
            order = sub.sort_values(col)
            chosen_envs.extend([(int(order.index[0]), f"low_{reason}_{split}"), (int(order.index[len(order) // 2]), f"median_{reason}_{split}"), (int(order.index[-1]), f"high_{reason}_{split}")])
    while len(chosen_envs) < n:
        idx = int(rng.integers(0, len(envs)))
        chosen_envs.append((idx, "uncertainty_diversity_fill"))
    rows = []
    gate = stage3_gate()
    for i in range(n):
        env_idx, reason = chosen_envs[i % len(chosen_envs)]
        env = envs.iloc[env_idx]
        edit = edit_vectors.iloc[i % len(edit_vectors)].to_dict()
        rows.append(
            {
                "calibration_case_id": f"stage4_cal_{i:04d}",
                "edit_vector_id": edit["edit_vector_id"],
                "edit_class": edit["edit_class"],
                "environment_id": env["environment_id"],
                "culture_id": env["culture_id"],
                "split": env["split"],
                "temperature": float(env["temperature"]),
                "pH": float(env["pH"]),
                "DO": float(env["DO"]),
                "baseline_culture_neighbour": env["culture_id"],
                "selection_reason": reason,
                "edit_distance": float(edit["edit_distance"]),
                "edit_cost": float(edit["edit_cost"]),
                "predicted_uncertainty": float(env["predicted_uncertainty"] + 0.03 * edit["edit_distance"]),
                "exact_yeast9_status": "pending_stage3_gate" if not gate.passed else "planned",
                "expected_lp_solves": 144,
                **{eid: float(edit[eid]) for eid in EDIT_COLUMNS},
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "stage4_exact_calibration_manifest.csv", index=False)
    acquisition = out[["calibration_case_id", "edit_vector_id", "environment_id", "selection_reason", "predicted_uncertainty", "edit_distance", "exact_yeast9_status"]].copy()
    acquisition.insert(0, "acquisition_round", 0)
    acquisition["acquisition_status"] = "planned_initial_batch_pending_stage3_gate" if not gate.passed else "planned_initial_batch"
    acquisition.to_csv(DATA / "stage4_acquisition_history.csv", index=False)
    return out


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> dict[str, np.ndarray]:
    xm = X.mean(axis=0)
    xs = X.std(axis=0) + EPS
    ym = Y.mean(axis=0)
    ys = Y.std(axis=0) + EPS
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys
    coef = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ Yn)
    return {"coef": coef, "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}


def _ridge_predict(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return ((X - model["x_mean"]) / model["x_std"]) @ model["coef"] * model["y_std"] + model["y_mean"]


def baseline_training_frame(dataset: dict[str, object]) -> pd.DataFrame:
    constraints = dataset["constraints_table"]
    flux = dataset["flux_table"]
    traj = dataset["traj"][["culture_id", "time_index", "X", "B_total"]].rename(columns={"time_index": "interval_index"})
    frame = constraints.merge(flux[["culture_id", "interval_index"] + FLUX_TARGETS], on=["culture_id", "interval_index"], suffixes=("", "_target"))
    frame = frame.merge(traj, on=["culture_id", "interval_index"], how="left")
    if "split" not in frame.columns:
        split_cols = [col for col in frame.columns if col.startswith("split")]
        if split_cols:
            frame["split"] = frame[split_cols[0]]
        else:
            frame = frame.merge(dataset["meta"][["culture_id", "split"]], on="culture_id", how="left")
    for col in EDIT_COLUMNS:
        frame[col] = 1.0
    return frame


def feature_columns() -> list[str]:
    return hybrid.ENV_COLUMNS + hybrid.INTERFACE_COLUMNS + ["X", "B_total"] + EDIT_COLUMNS


def train_surrogate(n_ensemble: int = 5, seed: int = 5101) -> tuple[list[dict[str, np.ndarray]], pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = hybrid.load_distillation_dataset()
    frame = baseline_training_frame(dataset)
    cols = feature_columns()
    train = frame["split"].eq("train").to_numpy()
    X = frame[cols].to_numpy(float)
    Y = frame[FLUX_TARGETS].to_numpy(float)
    rng = np.random.default_rng(seed)
    models = []
    rows = []
    for member in range(n_ensemble):
        train_idx = np.where(train)[0]
        boot = rng.choice(train_idx, size=len(train_idx), replace=True)
        model = _ridge_fit(X[boot], Y[boot], lam=1e-3)
        models.append(model)
        pred = _ridge_predict(model, X)
        for split in sorted(frame["split"].unique()) + ["all"]:
            mask = np.ones(len(frame), dtype=bool) if split == "all" else frame["split"].eq(split).to_numpy()
            row = {
                "model_member": member,
                "split": split,
                "n_intervals": int(mask.sum()),
                "metabolic_backend": "edit_aware_surrogate",
                "training_status": "baseline_exact_only_no_edited_calibration",
            }
            for j, target in enumerate(FLUX_TARGETS):
                row[f"{target}_rmse"] = float(np.sqrt(np.mean((pred[mask, j] - Y[mask, j]) ** 2)))
            rows.append(row)
    np.savez(
        RESULTS / "stage4_edit_aware_surrogate_ensemble.npz",
        **{f"member_{i}_{k}": v for i, model in enumerate(models) for k, v in model.items()},
        feature_columns=np.asarray(cols),
        target_columns=np.asarray(FLUX_TARGETS),
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DATA / "stage4_edit_surrogate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "source": "baseline_exact_gem",
                "rows": int(len(frame)),
                "weight": 1.0,
                "metabolic_backend": "yeast_gem_lp",
                "role": "control_to_flux_supervision",
            },
            {
                "source": "teacher_pseudodata",
                "rows": int(pd.read_csv(DATA / "teacher_pseudodata_uncertainty.csv").shape[0]) if (DATA / "teacher_pseudodata_uncertainty.csv").exists() else 0,
                "weight": 0.15,
                "metabolic_backend": "teacher_pseudodata_not_exact_fba",
                "role": "regulatory_control_coverage_only",
            },
            {
                "source": "edited_exact_gem",
                "rows": 0,
                "weight": 0.7,
                "metabolic_backend": "yeast_gem_lp",
                "role": "pending_stage3_gate",
            },
        ]
    ).to_csv(DATA / "stage4_edit_surrogate_training_manifest.csv", index=False)
    uncertainty = metrics.groupby("split").agg(
        mean_beta_flux_rmse=("beta_carotene_flux_rmse", "mean"),
        ensemble_beta_flux_rmse_sd=("beta_carotene_flux_rmse", "std"),
        n_intervals=("n_intervals", "max"),
    ).reset_index()
    uncertainty["calibration_status"] = "baseline_only_uncertainty_not_validated_for_edits"
    uncertainty.to_csv(DATA / "stage4_uncertainty_calibration.csv", index=False)
    return models, metrics


def _edit_effect_multiplier(edit: dict[str, float]) -> dict[str, float]:
    pathway = min(edit["PSY_capacity_multiplier"], edit["DES_capacity_multiplier"], edit["CYC_capacity_multiplier"])
    precursor = edit["precursor_supply_multiplier"] / max(edit["competing_sink_multiplier"], 0.1)
    support = math.sqrt(edit["ATP_support_multiplier"] * edit["oxygen_support_multiplier"] * edit["glucose_uptake_multiplier"])
    export = edit["export_capacity_multiplier"]
    burden = 0.035 * max(0.0, precursor - 1.0) + 0.020 * max(0.0, pathway - 1.0) + 0.015 * max(0.0, export - 1.0)
    return {
        "product": float(np.clip(pathway * precursor * support * export, 0.2, 4.0)),
        "biomass": float(np.clip(1.0 - burden + 0.03 * (support - 1.0), 0.55, 1.25)),
        "ggpp": float(np.clip(precursor, 0.3, 3.0)),
        "oxygen": float(np.clip(edit["oxygen_support_multiplier"], 0.5, 1.8)),
        "atp": float(np.clip(1.0 / max(edit["ATP_support_multiplier"], 0.2), 0.5, 1.3)),
    }


def rollout_candidate(models: list[dict[str, np.ndarray]], env: np.ndarray, controls: dict[str, np.ndarray], edit: dict[str, float], library: pd.DataFrame) -> dict[str, object]:
    interval_len = controls[hybrid.INTERFACE_COLUMNS[0]].shape[0]
    x = np.zeros(interval_len + 1)
    p = np.zeros(interval_len + 1)
    x[0] = 0.08
    features = []
    for k in range(interval_len):
        row = [env[0], env[1], env[2]]
        row.extend(float(controls[col][k]) for col in hybrid.INTERFACE_COLUMNS)
        row.extend([x[k], p[k]])
        row.extend(float(edit[col]) for col in EDIT_COLUMNS)
        features.append(row)
    X = np.asarray(features, dtype=float)
    preds = np.stack([_ridge_predict(model, X) for model in models], axis=0)
    mean_flux = preds.mean(axis=0)
    std_flux = preds.std(axis=0)
    effects = _edit_effect_multiplier(edit)
    target_index = {name: i for i, name in enumerate(FLUX_TARGETS)}
    beta_flux = np.clip(mean_flux[:, target_index["beta_carotene_flux"]] * effects["product"], 0.0, None)
    growth_flux = np.clip(mean_flux[:, target_index["biomass_flux"]] * effects["biomass"], 0.0, None)
    oxygen_flux = mean_flux[:, target_index["oxygen_uptake"]] * effects["oxygen"]
    ggpp_flux = np.clip(mean_flux[:, target_index["ggpp_flux"]] * effects["ggpp"], 0.0, None)
    atp_flux = np.clip(mean_flux[:, target_index["atp_maintenance_flux"]] * effects["atp"], 0.0, None)
    dt = 0.25
    degradation = 0.002 * edit["product_degradation_multiplier"] * (1.0 + 2.0 * np.clip(controls.get("z_ox", np.zeros(interval_len)), 0.0, None))
    for k in range(interval_len):
        x[k + 1] = max(1e-9, x[k] + dt * growth_flux[k] * x[k])
        p[k + 1] = max(0.0, p[k] + dt * (beta_flux[k] * x[k] - degradation[k] * p[k]))
    final = float(p[-1])
    auc = float(np.trapezoid(p, dx=dt))
    stress = float(np.trapezoid(np.clip(controls.get("z_ox", np.zeros(interval_len)), 0.0, None), dx=dt))
    atp_pressure = float(np.trapezoid(np.clip(controls.get("z_atp", np.zeros(interval_len)), 0.0, None), dx=dt))
    transfer = edit_distance(edit, library)
    ensemble_sd = float(np.mean(std_flux[:, target_index["beta_carotene_flux"]]))
    uncertainty = 0.05 * final + 0.6 * ensemble_sd + 0.08 * transfer
    feasibility = float(np.clip(0.99 - 0.10 * transfer - 0.05 * max(0.0, stress - 1.0), 0.05, 0.99))
    edit_c = edit_cost(edit, library)
    expected_score = 0.60 * final + 0.30 * auc + 0.10 * float(np.mean(beta_flux)) - 0.05 * stress - 0.03 * atp_pressure - 0.12 * edit_c
    robust_score = expected_score - 1.2 * uncertainty
    lcb_score = final - 1.645 * uncertainty
    return {
        "metabolic_backend": "edit_aware_surrogate",
        "expected_final_product": final,
        "final_product_p05": max(0.0, final - 1.645 * uncertainty),
        "final_product_p25": max(0.0, final - 0.675 * uncertainty),
        "final_product_p50": final,
        "final_product_p75": final + 0.675 * uncertainty,
        "final_product_p95": final + 1.645 * uncertainty,
        "expected_AUC": auc,
        "expected_productivity": auc / max(interval_len * dt, EPS),
        "final_biomass": float(x[-1]),
        "minimum_biomass": float(x.min()),
        "integrated_oxidative_stress": stress,
        "integrated_ATP_pressure": atp_pressure,
        "pathway_congestion_proxy": float(np.mean(np.maximum(0.0, ggpp_flux - beta_flux))),
        "feasibility_probability": feasibility,
        "surrogate_uncertainty": uncertainty,
        "ensemble_flux_uncertainty": ensemble_sd,
        "transfer_distance": transfer,
        "edit_cost": edit_c,
        "expected_score": expected_score,
        "robust_score": robust_score,
        "LCB_score": lcb_score,
        "mean_beta_carotene_flux": float(np.mean(beta_flux)),
        "mean_growth_flux": float(np.mean(growth_flux)),
        "mean_oxygen_flux": float(np.mean(oxygen_flux)),
        "mean_GGPP_flux": float(np.mean(ggpp_flux)),
        "mean_ATP_maintenance_flux": float(np.mean(atp_flux)),
    }


def broad_search(n_candidates: int = 20000, seed: int = 6201) -> pd.DataFrame:
    library = write_edit_library()
    models, _metrics = train_surrogate()
    dataset = hybrid.load_distillation_dataset()
    controls, _teacher_product, _student_manifest, _ckpts = hybrid.load_replay_control_sources(dataset)
    rng = np.random.default_rng(seed)
    env = np.column_stack(
        [
            rng.uniform(27.0, 33.0, n_candidates),
            rng.uniform(4.5, 5.5, n_candidates),
            rng.uniform(20.0, 80.0, n_candidates),
        ]
    )
    edit_vectors = sample_edit_vectors(library, max(200, min(n_candidates, 1500)), seed=seed + 1)
    # Reuse the frozen hybrid regulatory controller for candidate controls.
    hybrid_controls = hybrid.mean_checkpoint_controls(
        hybrid.load_manifest_checkpoints(DATA / "hybrid_student_manifest.csv", model="hybrid_reporter_supervised"),
        env,
    )
    rows = []
    for i in range(n_candidates):
        edit = edit_vectors.iloc[int(rng.integers(0, len(edit_vectors)))].to_dict()
        edit_dict = {col: float(edit[col]) for col in EDIT_COLUMNS}
        control_i = {col: hybrid_controls[col][i] for col in hybrid.INTERFACE_COLUMNS}
        result = rollout_candidate(models, env[i], control_i, edit_dict, library)
        objective = max(SEARCH_OBJECTIVES, key=lambda name: {
            "expected_production": result["expected_score"],
            "robust_production": result["robust_score"],
            "lower_confidence_production": result["LCB_score"],
            "low_stress_production": result["expected_score"] - 0.4 * result["integrated_oxidative_stress"],
            "low_cost_production": result["expected_score"] - 0.5 * result["edit_cost"],
            "sustained_production": result["expected_AUC"] + result["expected_final_product"],
        }[name])
        rows.append(
            {
                "candidate_id": f"stage4_search_{i:06d}",
                "temperature": float(env[i, 0]),
                "pH": float(env[i, 1]),
                "DO": float(env[i, 2]),
                "edit_vector_id": edit["edit_vector_id"],
                "n_active_edits": int(edit["n_active_edits"]),
                "best_objective_family": objective,
                "validated_transfer_domain_status": "pending_single_and_combination_transfer",
                **edit_dict,
                **result,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "stage4_search_candidates.csv", index=False)
    return out


def diverse_shortlist(candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    selected = []
    for objective in SEARCH_OBJECTIVES:
        score_col = {
            "expected_production": "expected_score",
            "robust_production": "robust_score",
            "lower_confidence_production": "LCB_score",
            "low_stress_production": "integrated_oxidative_stress",
            "low_cost_production": "edit_cost",
            "sustained_production": "expected_AUC",
        }[objective]
        ascending = objective in {"low_stress_production", "low_cost_production"}
        pool = candidates.sort_values(score_col, ascending=ascending).head(500)
        for row in pool.itertuples(index=False):
            if len(selected) >= n:
                break
            if any(abs(row.temperature - old["temperature"]) < 0.35 and abs(row.pH - old["pH"]) < 0.08 and abs(row.DO - old["DO"]) < 4.0 and row.edit_vector_id == old["edit_vector_id"] for old in selected):
                continue
            rec = row._asdict()
            rec["selection_objective"] = objective
            rec["reason_distinct"] = "diverse_edit_or_environment_or_objective"
            selected.append(rec)
            break
    if len(selected) < n:
        for row in candidates.sort_values("robust_score", ascending=False).itertuples(index=False):
            if len(selected) >= n:
                break
            if row.candidate_id in {old["candidate_id"] for old in selected}:
                continue
            rec = row._asdict()
            rec["selection_objective"] = "robust_fill"
            rec["reason_distinct"] = "robust_score_diversity_fill"
            selected.append(rec)
    out = pd.DataFrame(selected[:n])
    gate = stage3_gate()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["exact_yeast9_verification_status"] = "blocked_pending_stage3_gate" if not gate.passed else "not_yet_run"
    out["exact_verified_final_product"] = np.nan
    out["surrogate_prediction_error"] = np.nan
    out["main_advantage"] = "high surrogate-predicted production under bounded edit search"
    out["main_risk"] = "edited-strain transfer is not exact-validated yet"
    out.to_csv(DATA / "stage4_ranked_strain_environment_candidates.csv", index=False)
    out.to_csv(DATA / "stage4_exact_verified_candidates.csv", index=False)
    return out


def write_blocked_transfer_tables() -> None:
    library = write_edit_library()
    single_rows = []
    for row in library.itertuples(index=False):
        single_rows.append(
            {
                "edit_id": row.edit_id,
                "transfer_test": "single_edit",
                "n_exact_rollouts": 0,
                "effect_direction_accuracy": np.nan,
                "final_product_rmse": np.nan,
                "auc_rmse": np.nan,
                "biomass_rmse": np.nan,
                "feasibility_accuracy": np.nan,
                "uncertainty_coverage": np.nan,
                "status": "blocked_pending_stage3_gate_and_exact_edited_rollouts",
            }
        )
    pd.DataFrame(single_rows).to_csv(DATA / "stage4_single_edit_transfer_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "combination_class": cls,
                "n_exact_rollouts": 0,
                "max_validated_edit_count": 0,
                "max_validated_edit_distance": 0.0,
                "combination_rmse": np.nan,
                "status": "blocked_pending_single_edit_transfer",
            }
            for cls in ["double_edit", "triple_edit"]
        ]
    ).to_csv(DATA / "stage4_combination_transfer_metrics.csv", index=False)


def write_verification_placeholders(shortlist: pd.DataFrame | None = None) -> None:
    gate = stage3_gate()
    if shortlist is None:
        path = DATA / "stage4_ranked_strain_environment_candidates.csv"
        shortlist = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if shortlist.empty:
        verified = pd.DataFrame(
            [
                {
                    "candidate_id": "",
                    "exact_yeast9_verification_status": "not_run_no_shortlist",
                    "metabolic_backend": "yeast_gem_lp",
                    "n_actual_lp_solves": 0,
                }
            ]
        )
    else:
        verified = shortlist.copy()
        verified["metabolic_backend_exact"] = "yeast_gem_lp"
        verified["exact_yeast9_verification_status"] = "blocked_pending_stage3_gate" if not gate.passed else "planned"
        verified["n_actual_lp_solves"] = 0
        verified["n_surrogate_evaluations"] = 0
    verified.to_csv(DATA / "stage4_exact_verified_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "optimisation_regret",
                "value": np.nan,
                "status": "not_computable_until_exact_candidate_verification",
                "metabolic_backend": "yeast_gem_lp",
            },
            {"metric": "top_1_recovery", "value": np.nan, "status": "pending_exact_verification", "metabolic_backend": "yeast_gem_lp"},
            {"metric": "top_3_recovery", "value": np.nan, "status": "pending_exact_verification", "metabolic_backend": "yeast_gem_lp"},
            {"metric": "strain_ranking_spearman", "value": np.nan, "status": "pending_exact_verification", "metabolic_backend": "yeast_gem_lp"},
        ]
    ).to_csv(DATA / "stage4_optimisation_regret.csv", index=False)


def write_exact_rollout_placeholders() -> None:
    manifest_path = DATA / "stage4_exact_calibration_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else select_calibration_manifest()
    out = manifest[["calibration_case_id", "edit_vector_id", "environment_id", "exact_yeast9_status", "expected_lp_solves"]].copy()
    out["metabolic_backend"] = "yeast_gem_lp"
    out["n_actual_lp_solves"] = 0
    out["n_optimal_solves"] = 0
    out["n_surrogate_evaluations"] = 0
    out["status"] = "blocked_pending_stage3_gate"
    out.to_csv(DATA / "stage4_exact_calibration_solver_accounting.csv", index=False)
    out.to_csv(DATA / "stage4_exact_calibration_predictions.csv", index=False)


def audit_design_system(n_candidates: int = 20000, seed: int = 6201) -> pd.DataFrame:
    stage4_inventory()
    write_edit_library()
    select_calibration_manifest()
    write_exact_rollout_placeholders()
    train_surrogate()
    candidates = broad_search(n_candidates=n_candidates, seed=seed)
    shortlist = diverse_shortlist(candidates)
    write_blocked_transfer_tables()
    write_verification_placeholders(shortlist)
    gate = stage3_gate()
    criteria = [
        ("stage3_passed_ready_for_metabolic_edits", gate.passed, gate.status),
        ("edit_library_defined", (DATA / "strain_edit_library.csv").exists(), len(default_edit_library())),
        ("calibration_manifest_selected", (DATA / "stage4_exact_calibration_manifest.csv").exists(), 60),
        ("edited_exact_calibration_run", False, "blocked_pending_stage3_gate"),
        ("single_edit_transfer_validated", False, "blocked_pending_exact_edited_rollouts"),
        ("combination_transfer_validated", False, "blocked_pending_single_edit_transfer"),
        ("uncertainty_calibrated_for_edits", False, "baseline_only_no_edited_calibration"),
        ("candidate_search_completed_with_surrogate_labels", len(candidates) == n_candidates, n_candidates),
        ("exact_shortlist_verified", False, "blocked_pending_stage3_gate"),
    ]
    status = "stage_4_passed_probabilistic_strain_design_ready" if all(ok for _, ok, _ in criteria) else "stage_4_blocked_stage3_exact_replay_incomplete"
    out = pd.DataFrame(
        [
            {
                "stage": "Stage 4",
                "criterion": name,
                "passed": bool(ok),
                "value": value,
                "stage_3_status": gate.status,
                "stage_4_status": status,
                "decision": "pass" if status.startswith("stage_4_passed") else "blocked",
            }
            for name, ok, value in criteria
        ]
    )
    out.to_csv(DATA / "stage4_acceptance.csv", index=False)
    write_stage4_figures(shortlist)
    return out


def write_stage4_figures(shortlist: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "stage4_candidate_scores.svg"
    rows = shortlist.head(10)
    ymax = max(float(rows["expected_final_product"].max()), EPS) if not rows.empty else 1.0
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="32" text-anchor="middle" font-size="18" font-weight="bold">Stage 4 Surrogate Candidate Scores</text>',
    ]
    for i, row in enumerate(rows.itertuples(index=False)):
        h = 280 * float(row.expected_final_product) / ymax
        x = 70 + i * 75
        body.append(f'<rect x="{x}" y="{340-h:.1f}" width="42" height="{h:.1f}" fill="#297c7c"/>')
        body.append(f'<text x="{x+21}" y="370" text-anchor="middle" font-size="9">#{int(row.rank)}</text>')
    body.append('<text x="450" y="402" text-anchor="middle" font-size="12">Exact edited-strain verification pending Stage 3 gate</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(path.with_suffix(".png"), 900, 420, [])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "edit-library", "calibration", "exact-placeholders", "train-surrogate", "uncertainty", "single-transfer", "combination-transfer", "search", "verify-placeholders", "audit"])
    parser.add_argument("--n-candidates", type=int, default=20000)
    parser.add_argument("--n-calibration", type=int, default=60)
    parser.add_argument("--seed", type=int, default=6201)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        print(stage4_inventory().to_string(index=False))
    elif args.command == "edit-library":
        print(write_edit_library().to_string(index=False))
    elif args.command == "calibration":
        print(select_calibration_manifest(args.n_calibration, args.seed).head().to_string(index=False))
    elif args.command == "exact-placeholders":
        write_exact_rollout_placeholders()
    elif args.command in {"train-surrogate", "uncertainty"}:
        _models, metrics = train_surrogate()
        print(metrics.groupby("split")["beta_carotene_flux_rmse"].mean().to_string())
    elif args.command == "single-transfer":
        write_blocked_transfer_tables()
    elif args.command == "combination-transfer":
        write_blocked_transfer_tables()
    elif args.command == "search":
        candidates = broad_search(args.n_candidates, args.seed)
        shortlist = diverse_shortlist(candidates)
        write_verification_placeholders(shortlist)
        print(shortlist[["rank", "candidate_id", "expected_final_product", "feasibility_probability", "exact_yeast9_verification_status"]].to_string(index=False))
    elif args.command == "verify-placeholders":
        write_verification_placeholders()
    elif args.command == "audit":
        print(audit_design_system(args.n_candidates, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
