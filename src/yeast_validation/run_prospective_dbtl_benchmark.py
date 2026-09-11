#!/usr/bin/env python3
"""Prospective on-demand simulated DBTL benchmark.

This runner compares a conventional adaptive DBTL workflow with a frozen
pretrained hybrid digital twin workflow.  Unlike the finite-pool benchmark, it
does not precompute a candidate universe or query hidden exact outcomes for
unrequested designs.  Exact Yeast9/pFBA rollouts are called only inside the
workflow stage loops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import design_benchmark_exact as exact
import final_dbtl_benchmark_freeze as freeze
import run_deployment_verifier as verifier
import run_final_dbtl_benchmark as final_run

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "prospective_dbtl_benchmark"
ROLLOUTS = RESULTS / "on_demand_exact_rollouts"

BENCHMARK_ID = "prospective_on_demand_dbtl_v1"
WET_LAB_VERSION = "yeast9_dynamic_pfba_on_demand_v1"
PRIMARY_METHODS = ["conventional_dbtl", "pretrained_hybrid_digital_twin"]
PROSPECTIVE_WORLDS = [
    "baseline_world",
    "strong_oxidative_burden_world",
    "atp_limited_world",
    "precursor_competition_world",
    "pathway_bottleneck_damage_world",
    "mixed_multi_mechanism_world",
]
CAMPAIGN_SEEDS = list(range(86001, 86021))
FULL_CONVENTIONAL_BATCHES = [4, 4, 4, 4]
PILOT_CONVENTIONAL_BATCHES = [2, 2, 2]
FULL_HYBRID_VERIFICATION_BATCH = 8
PILOT_HYBRID_VERIFICATION_BATCH = 3
FULL_VIRTUAL_EVALUATIONS = 6000
PILOT_VIRTUAL_EVALUATIONS = 600
RELATIVE_EFFICIENCY_THRESHOLDS = [0.80, 0.90, 0.95]
MINIMUM_BIOMASS = 0.08
ENV_DOMAIN = {"temperature": (27.0, 33.0), "pH": (4.5, 5.5), "DO": (20.0, 80.0)}
OLD_EXACT_CACHE_FILES = [
    DATA / "design_benchmark_oracle_cache_index.csv",
    DATA / "final_dbtl_exact_cache_index.csv",
    DATA / "simulated_dbtl_exact_cache_index.csv",
]


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_payload(payload: object) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in [DATA, FIGURES, RESULTS, ROLLOUTS]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)


def csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            safe[key] = stable_json(value)
        else:
            safe[key] = value
    return safe


def candidate_hash(candidate: dict[str, Any]) -> str:
    return freeze.candidate_spec_hash(candidate)


def canonical_design_key(candidate: dict[str, Any]) -> str:
    return final_run.canonical_design_key(candidate.get("edits", []), candidate["environment"])


def edit_vector_hash(candidate: dict[str, Any]) -> str:
    payload = sorted(
        [
            {
                "intervention_id": str(edit.get("intervention_id", "")),
                "reaction_id": str(edit.get("reaction_id", "")),
                "edit_type": str(edit.get("edit_type", "")),
                "capacity_multiplier": round(float(edit.get("capacity_multiplier", 1.0)), 8),
                "target_lower_bound": None if pd.isna(edit.get("target_lower_bound", np.nan)) else round(float(edit.get("target_lower_bound")), 8),
                "target_upper_bound": None if pd.isna(edit.get("target_upper_bound", np.nan)) else round(float(edit.get("target_upper_bound")), 8),
            }
            for edit in candidate.get("edits", [])
        ],
        key=lambda x: (x["intervention_id"], x["reaction_id"], x["edit_type"], x["capacity_multiplier"]),
    )
    return hash_payload(payload)


def environment_key(env: dict[str, float]) -> str:
    return hash_payload({k: round(float(env[k]), 6) for k in ["temperature", "pH", "DO"]})


def load_library() -> pd.DataFrame:
    path = DATA / "final_dbtl_actionable_intervention_library.csv"
    if not path.exists():
        raise FileNotFoundError("Missing frozen intervention library. Run scripts/final_dbtl_benchmark_freeze.py first.")
    lib = pd.read_csv(path)
    if "mechanism_class" not in lib.columns and "intervention_class" in lib.columns:
        lib = lib.rename(columns={"intervention_class": "mechanism_class"})
    for col, default in [("risk_level", "medium"), ("dynamic_effect_size", 0.0), ("staged_effect_size", 0.0)]:
        if col not in lib.columns:
            lib[col] = default
    return lib


def audit_historical_training_data() -> tuple[pd.DataFrame, bool]:
    env = read_csv_if_exists(DATA / "gem_state_space_environment_grid.csv")
    traj = read_csv_if_exists(DATA / "gem_state_space_trajectories.csv")
    reporters = read_csv_if_exists(DATA / "gem_state_space_reporters.csv")
    split = read_csv_if_exists(DATA / "gem_state_space_split_manifest.csv")
    solver = read_csv_if_exists(DATA / "gem_state_space_solver_accounting.csv")
    teacher_manifest = read_csv_if_exists(DATA / "teacher_ensemble_manifest.csv")
    teacher_metrics = read_csv_if_exists(DATA / "teacher_channel_metrics.csv")
    library = load_library()

    training_tables = {"environment_grid": env, "trajectories": traj, "reporters": reporters, "split_manifest": split}
    columns = {name: set(df.columns) for name, df in training_tables.items() if not df.empty}
    forbidden = set(getattr(__import__("stage4_design"), "EDIT_COLUMNS", [])) | {
        "intervention_id",
        "reaction_id",
        "candidate_id",
        "candidate_hash",
        "strain_id",
        "edit_count",
        "edits",
    }
    forbidden_present = sorted({col for cols in columns.values() for col in cols if col in forbidden})
    env_cols = [c for c in ["temperature", "pH", "DO"] if c in env.columns]
    label_cols = []
    if not traj.empty:
        label_cols.extend([c for c in ["X", "B_total", "z_ox", "z_atp", "z_bottle", "z_er", "E_PSY", "E_DES", "E_CYC"] if c in traj.columns])
    if not reporters.empty:
        label_cols.extend([c for c in reporters.columns if c.startswith("R_") and not c.endswith("_source")])
    teacher_inputs = sorted(set(teacher_manifest.get("deployment_inputs", pd.Series(dtype=str)).dropna().astype(str)))

    n_cultures = int(env["environment_id"].nunique()) if "environment_id" in env else 0
    n_training = int((split["split"].astype(str).eq("train")).sum()) if "split" in split else 0
    n_exact = int(solver["n_actual_lp_solves"].sum()) if "n_actual_lp_solves" in solver else 0
    n_surrogate = int(solver["n_surrogate_evaluations"].sum()) if "n_surrogate_evaluations" in solver else 0
    worlds = sorted(set(traj.get("capacity_mode", pd.Series(dtype=str)).dropna().astype(str))) if not traj.empty else []
    strain_claim = "single_reference_no_edit_strain"
    benchmark_edit_overlap = "none_detected_in_teacher_training_columns" if not forbidden_present else "forbidden_edit_columns_present"
    reporters_available = ",".join(sorted(c for c in set(label_cols) & set(reporters.columns) if c.startswith("R_"))) if not reporters.empty else ""
    exact_labels = ",".join(sorted(set(label_cols)))
    env_ranges = {
        c: [float(env[c].min()), float(env[c].max()), int(env[c].nunique())]
        for c in env_cols
    }
    selected_teacher = teacher_manifest[teacher_manifest.get("ensemble_role", pd.Series(dtype=str)).eq("selected_teacher")]
    selected_models = ",".join(sorted(selected_teacher.get("model", pd.Series(dtype=str)).astype(str).unique()))
    teacher_metric_summary = ""
    if not teacher_metrics.empty and "model" in teacher_metrics:
        primary = teacher_metrics[teacher_metrics["model"].astype(str).eq("hybrid_reporter_supervised")]
        if primary.empty:
            primary = teacher_metrics[teacher_metrics["model"].astype(str).isin(selected_teacher.get("model", pd.Series(dtype=str)).astype(str))]
        if not primary.empty:
            teacher_metric_summary = f"median_normalized_rmse={primary['normalized_rmse'].median():.4f}"

    rows = [
        {
            "audit_item": "teacher_training_strains",
            "finding": strain_claim,
            "value": "1",
            "evidence": "gem_state_space culture IDs encode only capacity mode and environment; no strain/edit columns are present",
            "passes_gate": True,
        },
        {
            "audit_item": "historical_cultures",
            "finding": "fixed reference cultures",
            "value": n_cultures,
            "evidence": "data/gem_state_space_environment_grid.csv",
            "passes_gate": n_cultures > 0,
        },
        {
            "audit_item": "teacher_training_cultures",
            "finding": "training split cultures",
            "value": n_training,
            "evidence": "data/gem_state_space_split_manifest.csv",
            "passes_gate": n_training > 0,
        },
        {
            "audit_item": "environment_variables_varied",
            "finding": ",".join(env_cols),
            "value": stable_json(env_ranges),
            "evidence": "temperature, pH, and DO all vary in the historical grid",
            "passes_gate": set(env_cols) == {"temperature", "pH", "DO"},
        },
        {
            "audit_item": "metabolic_edit_dimensions_in_training",
            "finding": "absent" if not forbidden_present else "present",
            "value": ",".join(forbidden_present),
            "evidence": "checked training table columns against edit/intervention/candidate identifiers",
            "passes_gate": not forbidden_present,
        },
        {
            "audit_item": "benchmark_edit_overlap_with_training",
            "finding": benchmark_edit_overlap,
            "value": f"library_interventions={library.shape[0]}",
            "evidence": "teacher training tables contain no intervention_id, reaction_id, edit vector, candidate hash, or strain id",
            "passes_gate": not forbidden_present and library.shape[0] > 0,
        },
        {
            "audit_item": "training_worlds_or_conditions",
            "finding": ",".join(worlds),
            "value": f"exact_lp_solves={n_exact};surrogate_evaluations={n_surrogate}",
            "evidence": "data/gem_state_space_trajectories.csv and solver accounting",
            "passes_gate": n_exact > 0 and n_surrogate == 0,
        },
        {
            "audit_item": "reporter_information_available",
            "finding": reporters_available,
            "value": f"reporter_rows={reporters.shape[0] if not reporters.empty else 0}",
            "evidence": "data/gem_state_space_reporters.csv",
            "passes_gate": not reporters.empty,
        },
        {
            "audit_item": "deployment_inputs",
            "finding": ",".join(teacher_inputs),
            "value": selected_models,
            "evidence": "data/teacher_ensemble_manifest.csv",
            "passes_gate": teacher_inputs == ["temperature,pH,DO"],
        },
        {
            "audit_item": "exact_labels_used",
            "finding": exact_labels,
            "value": teacher_metric_summary,
            "evidence": "product, biomass/state, reporter, compact interface, and flux labels are labels only",
            "passes_gate": bool(exact_labels),
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(DATA / "prospective_historical_training_audit.csv", index=False)
    return audit, bool(audit["passes_gate"].all())


@dataclass
class CandidateFactory:
    library: pd.DataFrame
    campaign_id: str
    method_id: str
    rng: np.random.Generator
    proposal_counter: int = 0

    def random_env(self) -> dict[str, float]:
        return {
            "temperature": float(self.rng.uniform(*ENV_DOMAIN["temperature"])),
            "pH": float(self.rng.uniform(*ENV_DOMAIN["pH"])),
            "DO": float(self.rng.uniform(*ENV_DOMAIN["DO"])),
        }

    def grid_env(self, index: int) -> dict[str, float]:
        grid = [
            {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
            {"temperature": 30.0, "pH": 5.0, "DO": 80.0},
            {"temperature": 30.0, "pH": 5.0, "DO": 20.0},
            {"temperature": 27.0, "pH": 5.0, "DO": 65.0},
            {"temperature": 33.0, "pH": 5.0, "DO": 65.0},
            {"temperature": 30.0, "pH": 4.5, "DO": 65.0},
            {"temperature": 30.0, "pH": 5.5, "DO": 65.0},
            {"temperature": 27.0, "pH": 5.5, "DO": 80.0},
            {"temperature": 33.0, "pH": 4.5, "DO": 20.0},
        ]
        return dict(grid[index % len(grid)])

    def make(self, interventions: list[pd.Series], env: dict[str, float], hypothesis: str) -> dict[str, Any]:
        self.proposal_counter += 1
        edits = []
        for row in interventions:
            edit = final_run.edit_from_intervention(row)
            edit["intervention_id"] = str(row["intervention_id"])
            edits.append(edit)
        candidate = {
            "candidate_id": f"{self.campaign_id}_{self.method_id}_prospective_{self.proposal_counter:05d}",
            "strain_id": f"{self.campaign_id}_{self.method_id}_strain_{hash_payload(edits)[:12]}",
            "edit_tier": max(1, len(edits)),
            "edits": edits,
            "environment": {k: float(v) for k, v in env.items()},
            "requested_assay_panel": "trajectory_product_biomass_flux",
            "hypothesis": hypothesis,
            "metadata": {"benchmark_id": BENCHMARK_ID, "method_id": self.method_id, "prospective": True},
        }
        candidate["candidate_hash"] = candidate_hash(candidate)
        return candidate

    def sample_candidate(self, max_edits: int = 3) -> dict[str, Any]:
        edit_count = int(self.rng.choice(np.arange(1, max_edits + 1), p=np.array([0.50, 0.35, 0.15])[:max_edits] / np.array([0.50, 0.35, 0.15])[:max_edits].sum()))
        idx = self.rng.choice(len(self.library), size=min(edit_count, len(self.library)), replace=False)
        rows = [pd.Series(self.library.iloc[int(i)]) for i in idx]
        return self.make(rows, self.random_env(), "sampled open-ended strain-environment proposal")


def mechanism_classes(candidate: dict[str, Any], library: pd.DataFrame) -> str:
    lib = library.set_index("intervention_id")
    classes = []
    for edit in candidate.get("edits", []):
        iid = str(edit.get("intervention_id", ""))
        if iid in lib.index:
            classes.append(str(lib.loc[iid, "mechanism_class"]))
    return ";".join(sorted(set(classes)))


def trajectory_paths(culture_task_id: str) -> dict[str, Path]:
    prefix = culture_task_id[:16]
    return {
        "summary": ROLLOUTS / f"{prefix}_summary.csv",
        "trajectory": ROLLOUTS / f"{prefix}_trajectory.csv",
        "flux": ROLLOUTS / f"{prefix}_flux.csv",
        "constraints": ROLLOUTS / f"{prefix}_constraints.csv",
    }


def path_for_ledger(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def culture_task_id(campaign_id: str, method_id: str, world_id: str, stage_index: int, candidate: dict[str, Any], mock: bool = False) -> str:
    return hash_payload(
        {
            "benchmark_id": BENCHMARK_ID,
            "wet_lab_version": WET_LAB_VERSION,
            "lab_execution_mode": "mock" if mock else "fresh_exact_yeast9",
            "campaign_id": campaign_id,
            "method_id": method_id,
            "world_id": world_id,
            "stage_index": int(stage_index),
            "candidate_hash": candidate["candidate_hash"],
        }
    )[:24]


def old_cache_collision(candidate: dict[str, Any], world_id: str) -> dict[str, Any]:
    candidate_hash_value = str(candidate.get("candidate_hash", ""))
    design_key = canonical_design_key(candidate)
    collisions = []
    for path in OLD_EXACT_CACHE_FILES:
        if not path.exists():
            continue
        try:
            cache = pd.read_csv(path)
        except Exception:
            continue
        if cache.empty:
            continue
        mask = pd.Series(False, index=cache.index)
        if "candidate_hash" in cache.columns:
            mask = mask | cache["candidate_hash"].astype(str).eq(candidate_hash_value)
        if "design_key" in cache.columns:
            mask = mask | cache["design_key"].astype(str).eq(design_key)
        world_cols = [c for c in ["world_id", "hidden_regime_id", "regime_id"] if c in cache.columns]
        if world_cols:
            world_mask = pd.Series(False, index=cache.index)
            for col in world_cols:
                world_mask = world_mask | cache[col].astype(str).eq(world_id)
            mask = mask & world_mask
        if bool(mask.any()):
            collisions.append(f"{path.relative_to(ROOT)}:{int(mask.sum())}")
    return {
        "old_cache_collision": bool(collisions),
        "old_cache_collision_sources": ";".join(collisions),
        "old_result_accessed": False,
    }


def transfer_novelty_row(candidate: dict[str, Any], campaign_id: str, world_id: str, method_id: str, stage_index: int, historical_env_keys: set[str]) -> dict[str, Any]:
    env_key = environment_key(candidate["environment"])
    return {
        "benchmark_id": BENCHMARK_ID,
        "campaign_id": campaign_id,
        "world_id": world_id,
        "method_id": method_id,
        "stage_index": stage_index,
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "edit_vector_hash": edit_vector_hash(candidate),
        "environment_key": env_key,
        "edit_vector_present_in_teacher_training": False,
        "environment_present_in_teacher_training": env_key in historical_env_keys,
        "edit_environment_combination_present_in_teacher_training": False,
        "new_edit_vector": True,
        "new_environment": env_key not in historical_env_keys,
        "new_edit_environment_combination": True,
        "evidence": "historical teacher tables contain no edit dimensions; environment checked against gem_state_space_environment_grid.csv",
    }


def summarize_rollout(traj: pd.DataFrame, flux: pd.DataFrame, summary: pd.DataFrame, candidate: dict[str, Any], library: pd.DataFrame) -> dict[str, Any]:
    s = summary.iloc[0].to_dict() if not summary.empty else {}
    final_product = float(s.get("final_product", traj["B_total"].iloc[-1] if "B_total" in traj else 0.0))
    final_biomass = float(s.get("final_biomass", traj["X"].iloc[-1] if "X" in traj else 0.0))
    product_auc = float(s.get("product_AUC", np.trapezoid(traj["B_total"], traj["time"]) if {"B_total", "time"}.issubset(traj.columns) else 0.0))
    total_time = float(traj["time"].max()) if "time" in traj and len(traj) else 1.0
    productivity = final_product / max(total_time, 1e-12)
    min_biomass = float(s.get("minimum_biomass", traj["X"].min() if "X" in traj else final_biomass))
    feasible = bool(s.get("feasible", True)) and min_biomass >= MINIMUM_BIOMASS
    max_rate = 0.0
    if {"B_total", "time"}.issubset(traj.columns) and len(traj) > 1:
        dt = np.diff(traj["time"].to_numpy(float))
        db = np.diff(traj["B_total"].to_numpy(float))
        max_rate = float(np.max(db / np.maximum(dt, 1e-12)))
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "strain_id": candidate["strain_id"],
        "edit_count": len(candidate.get("edits", [])),
        "mechanism_classes": mechanism_classes(candidate, library),
        "temperature": float(candidate["environment"]["temperature"]),
        "pH": float(candidate["environment"]["pH"]),
        "DO": float(candidate["environment"]["DO"]),
        "candidate_json": stable_json(candidate),
        "final_product": final_product,
        "product_AUC": product_auc,
        "final_biomass": final_biomass,
        "minimum_biomass": min_biomass,
        "productivity": productivity,
        "maximum_instantaneous_product_rate": max_rate,
        "biologically_valid": feasible,
        "feasible": feasible,
        "n_actual_lp_solves": int(s.get("n_actual_lp_solves", 0)),
        "n_optimal_solves": int(s.get("n_optimal_solves", s.get("n_actual_lp_solves", 0))),
        "n_infeasible_solves": int(s.get("n_infeasible_solves", 0)),
        "n_unbounded_solves": int(s.get("n_unbounded_solves", 0)),
        "n_solver_errors": int(s.get("n_solver_errors", 0)),
        "n_skipped_intervals": int(s.get("n_skipped_intervals", 0)),
        "n_surrogate_evaluations": int(s.get("n_surrogate_evaluations", 0)),
        "metabolic_backend": str(s.get("metabolic_backend", "")),
        "exact_status": str(s.get("exact_status", "complete_exact_dynamic_pfba" if feasible else "invalid_or_infeasible")),
    }


def run_on_demand_exact(
    candidate: dict[str, Any],
    *,
    campaign_id: str,
    method_id: str,
    world_id: str,
    stage_index: int,
    culture_index: int,
    library: pd.DataFrame,
    mock: bool = False,
    fresh_exact: bool = False,
) -> dict[str, Any]:
    task_id = culture_task_id(campaign_id, method_id, world_id, stage_index, candidate, mock=mock)
    paths = trajectory_paths(task_id)
    cache_hit = paths["summary"].exists() and not (fresh_exact and not mock)
    cache_existed_before_request = paths["summary"].exists()
    old_cache = old_cache_collision(candidate, world_id)
    tic = time.perf_counter()
    exact_started_at = pd.Timestamp.utcnow().isoformat()
    if cache_hit:
        summary = pd.read_csv(paths["summary"])
        traj = pd.read_csv(paths["trajectory"])
        flux = pd.read_csv(paths["flux"]) if paths["flux"].exists() else pd.DataFrame()
        constraints = pd.read_csv(paths["constraints"]) if paths["constraints"].exists() else pd.DataFrame()
    elif mock:
        times = np.linspace(0.0, 12.0, 49)
        env = candidate["environment"]
        edit_bonus = 0.08 * len(candidate.get("edits", []))
        do_term = 0.0025 * float(env["DO"])
        temp_penalty = 0.015 * abs(float(env["temperature"]) - 30.0)
        ph_penalty = 0.08 * abs(float(env["pH"]) - 5.0)
        world_penalty = 0.04 * PROSPECTIVE_WORLDS.index(world_id) if world_id in PROSPECTIVE_WORLDS else 0.0
        final = max(0.02, 0.25 + edit_bonus + do_term - temp_penalty - ph_penalty - world_penalty)
        curve = final * (1.0 - np.exp(-times / 4.2))
        biomass = np.maximum(0.08, 0.08 + (1.2 - 0.03 * len(candidate.get("edits", []))) * (1.0 - np.exp(-times / 5.5)))
        traj = pd.DataFrame({"time": times, "B_total": curve, "X": biomass})
        flux = pd.DataFrame({"time": times[:-1], "dt": np.diff(times), "beta_carotene_flux": np.gradient(curve, times)[:-1], "biomass_flux": np.gradient(biomass, times)[:-1]})
        constraints = pd.DataFrame({"time": times[:-1], "mock_constraint": 1.0})
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_hash": candidate["candidate_hash"],
                    "hidden_regime_id": world_id,
                    "final_product": float(curve[-1]),
                    "product_AUC": float(np.trapezoid(curve, times)),
                    "final_biomass": float(biomass[-1]),
                    "minimum_biomass": float(biomass.min()),
                    "feasible": True,
                    "n_actual_lp_solves": 0,
                    "n_surrogate_evaluations": 0,
                    "metabolic_backend": "mock_prospective_pilot_backend",
                    "exact_status": "mock_pilot_not_scientific_evidence",
                }
            ]
        )
        summary.to_csv(paths["summary"], index=False)
        traj.to_csv(paths["trajectory"], index=False)
        flux.to_csv(paths["flux"], index=False)
        constraints.to_csv(paths["constraints"], index=False)
    else:
        if os.environ.get("PROSPECTIVE_EXACT_SUBPROCESS", "0") == "1":
            worker_inputs = RESULTS / "exact_worker_inputs"
            worker_inputs.mkdir(parents=True, exist_ok=True)
            candidate_path = worker_inputs / f"{task_id}.json"
            candidate_path.write_text(stable_json(candidate), encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "yeast_validation" / "run_prospective_exact_worker.py"),
                    "--candidate-json",
                    str(candidate_path),
                    "--world-id",
                    world_id,
                    "--summary",
                    str(paths["summary"]),
                    "--trajectory",
                    str(paths["trajectory"]),
                    "--flux",
                    str(paths["flux"]),
                    "--constraints",
                    str(paths["constraints"]),
                ],
                check=True,
                env=env,
            )
            summary = pd.read_csv(paths["summary"])
            traj = pd.read_csv(paths["trajectory"])
            flux = pd.read_csv(paths["flux"])
            constraints = pd.read_csv(paths["constraints"])
        else:
            traj, flux, constraints, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=world_id)
            summary.to_csv(paths["summary"], index=False)
            traj.to_csv(paths["trajectory"], index=False)
            flux.to_csv(paths["flux"], index=False)
            constraints.to_csv(paths["constraints"], index=False)
            exact.clear_solver_caches()
    exact_finished_at = pd.Timestamp.utcnow().isoformat()
    elapsed = time.perf_counter() - tic
    result = summarize_rollout(traj, flux, summary, candidate, library)
    result.update(
        {
            **old_cache,
            "benchmark_id": BENCHMARK_ID,
            "campaign_id": campaign_id,
            "world_id": world_id,
            "method_id": method_id,
            "stage_index": int(stage_index),
            "culture_index": int(culture_index),
            "culture_task_id": task_id,
            "physical_culture_charge": 1,
            "cache_hit": bool(cache_hit),
            "prospective_cache_existed_before_request": bool(cache_existed_before_request),
            "fresh_exact_simulation_performed": bool((not mock) and (not cache_hit)),
            "wall_time_seconds": elapsed,
            "exact_started_at": exact_started_at,
            "exact_finished_at": exact_finished_at,
            "trajectory_path": path_for_ledger(paths["trajectory"]),
            "biomass_trajectory_path": path_for_ledger(paths["trajectory"]),
            "flux_path": path_for_ledger(paths["flux"]),
            "constraint_path": path_for_ledger(paths["constraints"]),
            "requested_after_prior_exact_observations": True,
            "mock_mode": bool(mock),
        }
    )
    append_csv(DATA / "prospective_exact_simulator_call_ledger.csv", result)
    return result


def observed_score(row: dict[str, Any]) -> float:
    if not bool(row.get("biologically_valid", False)):
        return -1e9
    return float(row.get("final_product", 0.0))


def conventional_score(candidate: dict[str, Any], observations: list[dict[str, Any]], library: pd.DataFrame, rng: np.random.Generator) -> float:
    if not observations:
        return float(rng.normal())
    classes = set(mechanism_classes(candidate, library).split(";"))
    classes.discard("")
    same = [o for o in observations if classes & set(str(o.get("mechanism_classes", "")).split(";"))]
    base = float(np.mean([observed_score(o) for o in same])) if same else float(np.mean([observed_score(o) for o in observations]))
    explore = 0.25 / math.sqrt(1 + len(same))
    env = candidate["environment"]
    env_dist = abs(float(env["temperature"]) - 30.0) / 6.0 + abs(float(env["pH"]) - 5.0) + abs(float(env["DO"]) - 50.0) / 60.0
    return base + explore + 0.02 * env_dist + float(rng.normal(scale=0.005))


def propose_conventional_batch(
    factory: CandidateFactory,
    observations: list[dict[str, Any]],
    batch_size: int,
    seen_keys: set[str],
    stage_index: int,
    library: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    pool_size = 240 if observations else max(60, batch_size * 20)
    if not observations:
        sorted_library = library.sort_values(["mechanism_class", "intervention_id"]).reset_index(drop=True)
        candidates = [
            factory.make([pd.Series(sorted_library.iloc[i % len(sorted_library)])], factory.grid_env(i), "initial conventional space-filling stage")
            for i in range(pool_size)
        ]
        for i, cand in enumerate(candidates):
            cand["proposal_score"] = float(-i)
    else:
        candidates = [factory.sample_candidate(max_edits=3) for _ in range(pool_size)]
        for cand in candidates:
            cand["proposal_score"] = conventional_score(cand, observations, library, factory.rng)
        candidates.sort(key=lambda c: float(c["proposal_score"]), reverse=True)
    selected = []
    duplicates = 0
    for cand in candidates:
        key = canonical_design_key(cand)
        if key in seen_keys:
            duplicates += 1
            continue
        selected.append(cand)
        seen_keys.add(key)
        if len(selected) == batch_size:
            break
    if len(selected) < batch_size:
        raise RuntimeError(f"Conventional DBTL could only propose {len(selected)}/{batch_size} candidates at stage {stage_index}")
    return selected, duplicates


def teacher_environment_prior(env: dict[str, float], teacher_final_lookup: pd.DataFrame | None) -> float:
    if teacher_final_lookup is None or teacher_final_lookup.empty:
        return 0.0
    cols = ["temperature", "pH", "DO"]
    x = np.array([float(env[c]) for c in cols])
    ref = teacher_final_lookup[cols].to_numpy(float)
    scales = np.array([6.0, 1.0, 60.0])
    dist = np.linalg.norm((ref - x) / scales, axis=1)
    weights = 1.0 / np.maximum(dist, 1e-6)
    weights = weights / weights.sum()
    return float(np.dot(weights, teacher_final_lookup["B_total_final_mean"].to_numpy(float)))


def load_teacher_final_lookup() -> pd.DataFrame:
    pseudo = read_csv_if_exists(DATA / "teacher_pseudodata.csv")
    if pseudo.empty or "B_total_mean" not in pseudo:
        return pd.DataFrame()
    idx = pseudo.groupby("pseudo_culture_id")["time"].idxmax()
    cols = ["pseudo_culture_id", "temperature", "pH", "DO", "B_total_mean"]
    out = pseudo.loc[idx, cols].rename(columns={"B_total_mean": "B_total_final_mean"}).reset_index(drop=True)
    return out


def historical_environment_keys() -> set[str]:
    env = read_csv_if_exists(DATA / "gem_state_space_environment_grid.csv")
    if env.empty:
        return set()
    return {
        environment_key({"temperature": float(row["temperature"]), "pH": float(row["pH"]), "DO": float(row["DO"])})
        for _, row in env.iterrows()
        if {"temperature", "pH", "DO"}.issubset(env.columns)
    }


def hybrid_virtual_score(candidate: dict[str, Any], library: pd.DataFrame, teacher_lookup: pd.DataFrame) -> float:
    lib = library.set_index("intervention_id")
    intervention_score = 0.0
    risk_penalty = 0.0
    for edit in candidate.get("edits", []):
        iid = str(edit.get("intervention_id", ""))
        if iid in lib.index:
            row = lib.loc[iid]
            intervention_score += 0.75 * float(row.get("dynamic_effect_size", 0.0) or 0.0)
            intervention_score += 0.45 * float(row.get("staged_effect_size", 0.0) or 0.0)
            risk_penalty += {"low": 0.0, "medium": 0.02, "high": 0.06}.get(str(row.get("risk_level", "medium")), 0.03)
    env_score = teacher_environment_prior(candidate["environment"], teacher_lookup)
    edit_penalty = 0.025 * max(0, len(candidate.get("edits", [])) - 1)
    return float(env_score + intervention_score - risk_penalty - edit_penalty)


def hybrid_virtual_search(
    factory: CandidateFactory,
    virtual_evaluations: int,
    verification_batch: int,
    seen_keys: set[str],
    library: pd.DataFrame,
    teacher_lookup: pd.DataFrame,
    campaign_id: str,
    world_id: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    started = time.perf_counter()
    started_at = pd.Timestamp.utcnow().isoformat()
    rows = []
    candidates = []
    for i in range(virtual_evaluations):
        cand = factory.sample_candidate(max_edits=3)
        score = hybrid_virtual_score(cand, library, teacher_lookup)
        rows.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "campaign_id": campaign_id,
                "world_id": world_id,
                "method_id": "pretrained_hybrid_digital_twin",
                "virtual_evaluation_index": i + 1,
                "candidate_id": cand["candidate_id"],
                "candidate_hash": cand["candidate_hash"],
                "design_key": canonical_design_key(cand),
                "temperature": cand["environment"]["temperature"],
                "pH": cand["environment"]["pH"],
                "DO": cand["environment"]["DO"],
                "edit_count": len(cand.get("edits", [])),
                "mechanism_classes": mechanism_classes(cand, library),
                "hybrid_virtual_score": score,
                "valid_virtual_candidate": True,
                "rejection_reason": "",
                "exact_outcome_accessed": False,
                "selected_for_exact_verification": False,
                "virtual_search_started_at": started_at,
                "source": "frozen_teacher_environment_prior_plus_frozen_intervention_library",
            }
        )
        cand["proposal_score"] = score
        candidates.append(cand)
    candidates.sort(key=lambda c: float(c["proposal_score"]), reverse=True)
    selected = []
    for cand in candidates:
        key = canonical_design_key(cand)
        if key in seen_keys:
            continue
        selected.append(cand)
        seen_keys.add(key)
        if len(selected) == verification_batch:
            break
    if len(selected) < verification_batch:
        raise RuntimeError(f"Hybrid virtual search selected only {len(selected)}/{verification_batch} candidates")
    virtual = pd.DataFrame(rows)
    selected_hashes = {c["candidate_hash"] for c in selected}
    virtual["selected_for_exact_verification"] = virtual["candidate_hash"].isin(selected_hashes)
    virtual["virtual_search_finished_at"] = pd.Timestamp.utcnow().isoformat()
    virtual["virtual_runtime_seconds"] = time.perf_counter() - started
    append_virtual_ledger(virtual)
    return selected, virtual


def append_virtual_ledger(virtual: pd.DataFrame) -> None:
    path = DATA / "prospective_virtual_evaluation_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    virtual.to_csv(path, mode="a", header=not path.exists(), index=False)


def reset_outputs() -> None:
    for path in [
        DATA / "prospective_campaign_manifest.csv",
        DATA / "prospective_exact_simulator_call_ledger.csv",
        DATA / "prospective_biological_stage_ledger.csv",
        DATA / "prospective_virtual_evaluation_ledger.csv",
        DATA / "prospective_candidate_designs.csv",
        DATA / "prospective_campaign_level_summary.csv",
        DATA / "prospective_paired_statistical_comparisons.csv",
        DATA / "prospective_pareto_table.csv",
        DATA / "prospective_leakage_access_audit.csv",
        DATA / "prospective_leakage_audit.csv",
        DATA / "prospective_old_cache_collision_audit.csv",
        DATA / "prospective_transfer_novelty_audit.csv",
        DATA / "prospective_proposal_dependency_audit.csv",
        DATA / "prospective_conventional_adaptivity_audit.csv",
        DATA / "prospective_pilot_acceptance.csv",
        DATA / "prospective_exact_trajectories.csv",
        DATA / "prospective_exact_call_ledger.csv",
        DATA / "prospective_campaign_summary.csv",
        DATA / "prospective_full_run_resource_estimate.csv",
        DATA / "prospective_training_cost_amortization.csv",
    ]:
        if path.exists():
            path.unlink()


def write_stage_ledger(campaign_id: str, world_id: str, seed: int, method_id: str, stage_index: int, observations: list[dict[str, Any]], selected: list[dict[str, Any]], virtual_count: int = 0) -> None:
    valid = [o for o in observations if bool(o.get("biologically_valid", False))]
    best = max(valid, key=lambda r: float(r["final_product"])) if valid else {}
    best_prod = max(valid, key=lambda r: float(r["productivity"])) if valid else {}
    best_auc = max(valid, key=lambda r: float(r["product_AUC"])) if valid else {}
    append_csv(
        DATA / "prospective_biological_stage_ledger.csv",
        {
            "benchmark_id": BENCHMARK_ID,
            "campaign_id": campaign_id,
            "world_id": world_id,
            "campaign_seed": seed,
            "method_id": method_id,
            "stage_index": stage_index,
            "candidate_ids_requested": ",".join(c["candidate_id"] for c in selected),
            "physical_cultures_this_stage": len(selected),
            "physical_cultures_cumulative": len(observations),
            "virtual_evaluations_this_stage": virtual_count,
            "best_verified_final_product_so_far": float(best.get("final_product", np.nan)),
            "best_verified_productivity_so_far": float(best_prod.get("productivity", np.nan)),
            "best_verified_product_AUC_so_far": float(best_auc.get("product_AUC", np.nan)),
            "scenario_days_cumulative": stage_index * 10.0,
        },
    )


def run_campaign(
    *,
    world_id: str,
    seed: int,
    library: pd.DataFrame,
    teacher_lookup: pd.DataFrame,
    conventional_batches: list[int],
    hybrid_batch: int,
    virtual_evaluations: int,
    mock: bool,
    fresh_exact: bool = False,
) -> None:
    campaign_id = f"prospective_{world_id}_seed_{seed}"
    historical_env = historical_environment_keys()
    manifest_row = {
        "benchmark_id": BENCHMARK_ID,
        "campaign_id": campaign_id,
        "world_id": world_id,
        "campaign_seed": seed,
        "methods": ",".join(PRIMARY_METHODS),
        "environment_domain": stable_json(ENV_DOMAIN),
        "intervention_library_file": "data/final_dbtl_actionable_intervention_library.csv",
        "conventional_batches": stable_json(conventional_batches),
        "hybrid_verification_batch": hybrid_batch,
        "hybrid_virtual_evaluations": virtual_evaluations,
        "mock_mode": bool(mock),
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    append_csv(DATA / "prospective_campaign_manifest.csv", manifest_row)

    for method_id in PRIMARY_METHODS:
        factory = CandidateFactory(library, campaign_id, method_id, np.random.default_rng(seed + (17 if method_id == PRIMARY_METHODS[0] else 29)))
        observations: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        if method_id == "conventional_dbtl":
            for stage_index, batch_size in enumerate(conventional_batches, start=1):
                before_n = len(observations)
                proposal_started_at = pd.Timestamp.utcnow().isoformat()
                prior_latest_exact_finished_at = max([str(o.get("exact_finished_at", "")) for o in observations], default="")
                prior_best = max([observed_score(o) for o in observations], default=np.nan)
                selected, _duplicates = propose_conventional_batch(factory, observations, batch_size, seen_keys, stage_index, library)
                proposal_finished_at = pd.Timestamp.utcnow().isoformat()
                for cand in selected:
                    append_csv(
                        DATA / "prospective_proposal_dependency_audit.csv",
                        {
                            "campaign_id": campaign_id,
                            "world_id": world_id,
                            "method_id": method_id,
                            "stage_index": stage_index,
                            "candidate_id": cand["candidate_id"],
                            "candidate_hash": cand["candidate_hash"],
                            "proposal_started_at": proposal_started_at,
                            "proposal_finished_at": proposal_finished_at,
                            "prior_exact_observations_count": len(observations),
                            "prior_latest_exact_finished_at": prior_latest_exact_finished_at,
                            "proposal_after_prior_exact": True if stage_index == 1 else proposal_started_at > prior_latest_exact_finished_at,
                            "prior_best_final_product": prior_best,
                            "proposal_score": float(cand.get("proposal_score", np.nan)),
                            "adaptation_evidence": "stage1_space_filling" if stage_index == 1 else "proposal scored after prior exact observations were appended",
                        },
                    )
                for cand in selected:
                    append_csv(DATA / "prospective_candidate_designs.csv", csv_safe_row({"campaign_id": campaign_id, "world_id": world_id, "method_id": method_id, "stage_index": stage_index, "edit_vector_hash": edit_vector_hash(cand), **{k: v for k, v in cand.items() if k != "metadata"}}))
                    append_csv(DATA / "prospective_transfer_novelty_audit.csv", transfer_novelty_row(cand, campaign_id, world_id, method_id, stage_index, historical_env))
                    result = run_on_demand_exact(cand, campaign_id=campaign_id, method_id=method_id, world_id=world_id, stage_index=stage_index, culture_index=len(observations) + 1, library=library, mock=mock, fresh_exact=fresh_exact)
                    observations.append(result)
                assert len(observations) > before_n
                write_stage_ledger(campaign_id, world_id, seed, method_id, stage_index, observations, selected)
        else:
            proposal_started_at = pd.Timestamp.utcnow().isoformat()
            selected, _virtual = hybrid_virtual_search(factory, virtual_evaluations, hybrid_batch, seen_keys, library, teacher_lookup, campaign_id, world_id)
            proposal_finished_at = pd.Timestamp.utcnow().isoformat()
            stage_index = 1
            for cand in selected:
                append_csv(
                    DATA / "prospective_proposal_dependency_audit.csv",
                    {
                        "campaign_id": campaign_id,
                        "world_id": world_id,
                        "method_id": method_id,
                        "stage_index": stage_index,
                        "candidate_id": cand["candidate_id"],
                        "candidate_hash": cand["candidate_hash"],
                        "proposal_started_at": proposal_started_at,
                        "proposal_finished_at": proposal_finished_at,
                        "prior_exact_observations_count": 0,
                        "prior_latest_exact_finished_at": "",
                        "proposal_after_prior_exact": True,
                        "prior_best_final_product": np.nan,
                        "proposal_score": float(cand.get("proposal_score", np.nan)),
                        "adaptation_evidence": "frozen teacher virtual search completed before exact verification",
                    },
                )
            for cand in selected:
                append_csv(DATA / "prospective_candidate_designs.csv", csv_safe_row({"campaign_id": campaign_id, "world_id": world_id, "method_id": method_id, "stage_index": stage_index, "edit_vector_hash": edit_vector_hash(cand), **{k: v for k, v in cand.items() if k != "metadata"}}))
                append_csv(DATA / "prospective_transfer_novelty_audit.csv", transfer_novelty_row(cand, campaign_id, world_id, method_id, stage_index, historical_env))
                result = run_on_demand_exact(cand, campaign_id=campaign_id, method_id=method_id, world_id=world_id, stage_index=stage_index, culture_index=len(observations) + 1, library=library, mock=mock, fresh_exact=fresh_exact)
                observations.append(result)
            write_stage_ledger(campaign_id, world_id, seed, method_id, stage_index, observations, selected, virtual_count=virtual_evaluations)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 91) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def write_analysis() -> None:
    exact_calls = read_csv_if_exists(DATA / "prospective_exact_simulator_call_ledger.csv")
    stages = read_csv_if_exists(DATA / "prospective_biological_stage_ledger.csv")
    virtual = read_csv_if_exists(DATA / "prospective_virtual_evaluation_ledger.csv")
    if exact_calls.empty:
        return

    summary_rows = []
    for (campaign_id, world_id, method_id), group in exact_calls.groupby(["campaign_id", "world_id", "method_id"]):
        valid = group[group["biologically_valid"].astype(bool)]
        best_titer = valid.loc[valid["final_product"].idxmax()] if not valid.empty else group.iloc[0]
        best_prod = valid.loc[valid["productivity"].idxmax()] if not valid.empty else group.iloc[0]
        best_auc = valid.loc[valid["product_AUC"].idxmax()] if not valid.empty else group.iloc[0]
        summary_rows.append(
            {
                "campaign_id": campaign_id,
                "world_id": world_id,
                "method_id": method_id,
                "best_final_product": float(best_titer["final_product"]),
                "best_productivity": float(best_prod["productivity"]),
                "best_product_AUC": float(best_auc["product_AUC"]),
                "best_final_biomass": float(best_titer["final_biomass"]),
                "best_candidate_id": str(best_titer["candidate_id"]),
                "physical_cultures": int(group["physical_culture_charge"].sum()),
                "sequential_biological_stages": int(group["stage_index"].max()),
                "scenario_days": float(group["stage_index"].max()) * 10.0,
                "exact_lp_solves": int(group["n_actual_lp_solves"].sum()),
                "virtual_candidates": int(virtual[(virtual["campaign_id"].eq(campaign_id)) & (virtual["world_id"].eq(world_id)) & (virtual["method_id"].eq(method_id))].shape[0]) if not virtual.empty else 0,
                "mock_mode": bool(group["mock_mode"].astype(bool).any()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA / "prospective_campaign_level_summary.csv", index=False)
    summary.to_csv(DATA / "prospective_campaign_summary.csv", index=False)

    paired_rows = []
    piv = summary.pivot_table(index=["campaign_id", "world_id"], columns="method_id", values=["best_final_product", "best_productivity", "best_product_AUC", "physical_cultures", "scenario_days"], aggfunc="first")
    for metric in ["best_final_product", "best_productivity", "best_product_AUC", "physical_cultures", "scenario_days"]:
        if (metric, "pretrained_hybrid_digital_twin") not in piv or (metric, "conventional_dbtl") not in piv:
            continue
        diff = piv[(metric, "pretrained_hybrid_digital_twin")] - piv[(metric, "conventional_dbtl")]
        low, high = bootstrap_ci(diff.to_numpy(float))
        paired_rows.append(
            {
                "comparison": "hybrid_minus_conventional",
                "metric": metric,
                "n_paired_campaigns": int(diff.notna().sum()),
                "mean_difference": float(diff.mean()),
                "median_difference": float(diff.median()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "hybrid_win_count": int((diff > 1e-12).sum()),
                "tie_count": int((abs(diff) <= 1e-12).sum()),
                "hybrid_loss_count": int((diff < -1e-12).sum()),
            }
        )

    match_rows = []
    if not stages.empty:
        for (campaign_id, world_id), group in stages.groupby(["campaign_id", "world_id"]):
            h = group[group["method_id"].eq("pretrained_hybrid_digital_twin")]
            c = group[group["method_id"].eq("conventional_dbtl")]
            if h.empty or c.empty:
                continue
            h_final = float(h.sort_values("stage_index")["best_verified_final_product_so_far"].iloc[-1])
            h_prod = float(h.sort_values("stage_index")["best_verified_productivity_so_far"].iloc[-1])
            c_match = c[c["best_verified_final_product_so_far"].astype(float) >= h_final]
            c_prod_match = c[c["best_verified_productivity_so_far"].astype(float) >= h_prod]
            match_rows.append(
                {
                    "campaign_id": campaign_id,
                    "world_id": world_id,
                    "hybrid_best_final_product": h_final,
                    "hybrid_best_productivity": h_prod,
                    "earliest_conventional_stage_matching_hybrid_titer": int(c_match["stage_index"].min()) if not c_match.empty else np.nan,
                    "earliest_conventional_stage_matching_hybrid_productivity": int(c_prod_match["stage_index"].min()) if not c_prod_match.empty else np.nan,
                }
            )
    match = pd.DataFrame(match_rows)
    if not match.empty:
        match.to_csv(DATA / "prospective_conventional_stages_to_match_hybrid.csv", index=False)

    paired_rows.extend(write_search_performance_tables(exact_calls, stages, summary))
    pd.DataFrame(paired_rows).to_csv(DATA / "prospective_paired_statistical_comparisons.csv", index=False)

    pareto_rows = []
    for (campaign_id, world_id, method_id), group in exact_calls.groupby(["campaign_id", "world_id", "method_id"]):
        valid = group[group["biologically_valid"].astype(bool)].copy()
        for i, row in valid.iterrows():
            dominated = valid[(valid["final_product"].astype(float) >= float(row["final_product"])) & (valid["productivity"].astype(float) >= float(row["productivity"])) & ((valid["final_product"].astype(float) > float(row["final_product"])) | (valid["productivity"].astype(float) > float(row["productivity"])))]
            pareto_rows.append(
                {
                    "campaign_id": campaign_id,
                    "world_id": world_id,
                    "method_id": method_id,
                    "candidate_id": row["candidate_id"],
                    "final_product": float(row["final_product"]),
                    "productivity": float(row["productivity"]),
                    "product_AUC": float(row["product_AUC"]),
                    "final_biomass": float(row["final_biomass"]),
                    "pareto_front": dominated.empty,
                }
            )
    pd.DataFrame(pareto_rows).to_csv(DATA / "prospective_pareto_table.csv", index=False)
    write_exact_trajectory_bundle(exact_calls)
    write_old_cache_audit(exact_calls)
    write_conventional_adaptivity_audit(exact_calls)
    write_leakage_audit(exact_calls, virtual, stages)
    write_pilot_acceptance(exact_calls, virtual, stages)
    write_full_run_resource_estimate(exact_calls)
    write_training_cost_amortization()
    write_figures(exact_calls, stages, summary, match if "match" in locals() else pd.DataFrame())
    write_report(summary, paired_rows)
    shutil.copyfile(DATA / "prospective_exact_simulator_call_ledger.csv", DATA / "prospective_exact_call_ledger.csv")


def paired_metric_rows(frame: pd.DataFrame, metrics: list[str], *, index_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    pivot = frame.pivot_table(index=index_cols, columns="method_id", values=metrics, aggfunc="first")
    for metric in metrics:
        if (metric, "pretrained_hybrid_digital_twin") not in pivot or (metric, "conventional_dbtl") not in pivot:
            continue
        diff = pivot[(metric, "pretrained_hybrid_digital_twin")] - pivot[(metric, "conventional_dbtl")]
        low, high = bootstrap_ci(diff.to_numpy(float))
        rows.append(
            {
                "comparison": "hybrid_minus_conventional",
                "metric": metric,
                "n_paired_campaigns": int(diff.notna().sum()),
                "mean_difference": float(diff.mean()),
                "median_difference": float(diff.median()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "hybrid_win_count": int((diff > 1e-12).sum()),
                "tie_count": int((abs(diff) <= 1e-12).sum()),
                "hybrid_loss_count": int((diff < -1e-12).sum()),
            }
        )
    return rows


def write_search_performance_tables(exact_calls: pd.DataFrame, stages: pd.DataFrame, summary: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = ["final_product", "productivity", "product_AUC"]
    best_rows = []
    for (campaign_id, world_id, method_id), group in exact_calls.groupby(["campaign_id", "world_id", "method_id"]):
        ordered = group.sort_values(["stage_index", "culture_index", "exact_started_at"]).reset_index(drop=True).copy()
        for metric in metrics:
            ordered[f"best_{metric}_so_far"] = ordered[metric].astype(float).cummax()
        ordered["physical_culture_index"] = np.arange(1, len(ordered) + 1)
        best_rows.append(ordered)
    best = pd.concat(best_rows, ignore_index=True, sort=False) if best_rows else pd.DataFrame()
    best.to_csv(DATA / "prospective_best_so_far_by_culture.csv", index=False)
    if not stages.empty:
        stages.to_csv(DATA / "prospective_best_so_far_by_stage.csv", index=False)

    auc_rows = []
    for (campaign_id, world_id, method_id), group in best.groupby(["campaign_id", "world_id", "method_id"]):
        x = group["physical_culture_index"].to_numpy(float)
        for metric in metrics:
            y = group[f"best_{metric}_so_far"].to_numpy(float)
            auc = float(np.trapezoid(y, x)) if len(y) > 1 else float(y[0])
            auc_rows.append(
                {
                    "campaign_id": campaign_id,
                    "world_id": world_id,
                    "method_id": method_id,
                    "metric": metric,
                    "best_so_far_auc": auc,
                    "mean_best_so_far_over_measured_cultures": float(np.mean(y)),
                    "final_best_so_far": float(y[-1]),
                    "physical_cultures": int(len(y)),
                }
            )
    auc_table = pd.DataFrame(auc_rows)
    auc_table.to_csv(DATA / "prospective_best_so_far_auc.csv", index=False)

    common_auc_rows = []
    for (campaign_id, world_id), pair in best.groupby(["campaign_id", "world_id"]):
        methods_present = set(pair["method_id"].astype(str))
        if not set(PRIMARY_METHODS).issubset(methods_present):
            continue
        common_horizon = int(pair.groupby("method_id")["physical_culture_index"].max().min())
        if common_horizon <= 0:
            continue
        for method_id, group in pair.groupby("method_id"):
            common = group[group["physical_culture_index"].astype(int) <= common_horizon].copy()
            x = common["physical_culture_index"].to_numpy(float)
            for metric in metrics:
                y = common[f"best_{metric}_so_far"].to_numpy(float)
                auc = float(np.trapezoid(y, x)) if len(y) > 1 else float(y[0])
                common_auc_rows.append(
                    {
                        "campaign_id": campaign_id,
                        "world_id": world_id,
                        "method_id": method_id,
                        "metric": metric,
                        "common_physical_culture_horizon": common_horizon,
                        "common_horizon_best_so_far_auc": auc,
                        "common_horizon_mean_best_so_far": float(np.mean(y)),
                        "common_horizon_final_best_so_far": float(y[-1]),
                    }
                )
    common_auc_table = pd.DataFrame(common_auc_rows)
    common_auc_table.to_csv(DATA / "prospective_best_so_far_auc_common_budget.csv", index=False)

    threshold_rows = []
    for (campaign_id, world_id), pair in best.groupby(["campaign_id", "world_id"]):
        for metric in metrics:
            pooled_best = float(pair[metric].max())
            for fraction in RELATIVE_EFFICIENCY_THRESHOLDS:
                threshold = pooled_best * fraction
                for method_id, group in pair.groupby("method_id"):
                    hits = group[group[f"best_{metric}_so_far"].astype(float) >= threshold]
                    stage_hits = stages[
                        stages["campaign_id"].eq(campaign_id)
                        & stages["world_id"].eq(world_id)
                        & stages["method_id"].eq(method_id)
                        & (stages[f"best_verified_{metric}_so_far" if metric != "product_AUC" else "best_verified_product_AUC_so_far"].astype(float) >= threshold)
                    ] if not stages.empty else pd.DataFrame()
                    threshold_rows.append(
                        {
                            "campaign_id": campaign_id,
                            "world_id": world_id,
                            "method_id": method_id,
                            "metric": metric,
                            "relative_fraction_of_campaign_observed_best": fraction,
                            "threshold_value": threshold,
                            "campaign_observed_best": pooled_best,
                            "cultures_to_threshold": int(hits["physical_culture_index"].min()) if not hits.empty else np.nan,
                            "stages_to_threshold": int(stage_hits["stage_index"].min()) if not stage_hits.empty else np.nan,
                            "threshold_type": "retrospective_fraction_of_observed_campaign_best_not_predeclared_target",
                        }
                    )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(DATA / "prospective_relative_threshold_efficiency.csv", index=False)

    final_rows = []
    for (campaign_id, world_id, method_id), group in exact_calls.groupby(["campaign_id", "world_id", "method_id"]):
        idx = group["final_product"].astype(float).idxmax()
        row = group.loc[idx]
        best_productivity = float(group["productivity"].astype(float).max())
        best_product_auc = float(group["product_AUC"].astype(float).max())
        final_rows.append(
            {
                "campaign_id": campaign_id,
                "world_id": world_id,
                "method_id": method_id,
                "best_candidate_id": row["candidate_id"],
                "best_candidate_hash": row["candidate_hash"],
                "best_final_product": float(row["final_product"]),
                "best_productivity": best_productivity,
                "best_product_AUC": best_product_auc,
                "best_final_biomass": float(row["final_biomass"]),
                "best_minimum_biomass": float(row["minimum_biomass"]),
                "best_edit_count": int(row["edit_count"]),
                "best_mechanism_classes": row["mechanism_classes"],
                "physical_cultures": int(group["physical_culture_charge"].sum()),
                "sequential_biological_stages": int(group["stage_index"].max()),
            }
        )
    final_table = pd.DataFrame(final_rows)
    final_table.to_csv(DATA / "prospective_final_best_candidates.csv", index=False)

    paired = []
    if not final_table.empty:
        piv = final_table.pivot_table(index=["campaign_id", "world_id"], columns="method_id", values=["best_final_product", "best_productivity", "best_product_AUC", "physical_cultures", "sequential_biological_stages"], aggfunc="first")
        for metric in ["best_final_product", "best_productivity", "best_product_AUC", "physical_cultures", "sequential_biological_stages"]:
            if (metric, "pretrained_hybrid_digital_twin") in piv and (metric, "conventional_dbtl") in piv:
                paired.append((metric, piv[(metric, "pretrained_hybrid_digital_twin")] - piv[(metric, "conventional_dbtl")]))
        paired_rows = []
        for metric, diff in paired:
            for key, value in diff.items():
                paired_rows.append({"campaign_id": key[0], "world_id": key[1], "metric": metric, "hybrid_minus_conventional": float(value)})
        pd.DataFrame(paired_rows).to_csv(DATA / "prospective_final_paired_campaign_results.csv", index=False)

    world_rows = []
    paired_file = read_csv_if_exists(DATA / "prospective_final_paired_campaign_results.csv")
    if not paired_file.empty:
        for (world_id, metric), group in paired_file.groupby(["world_id", "metric"]):
            low, high = bootstrap_ci(group["hybrid_minus_conventional"].to_numpy(float))
            world_rows.append(
                {
                    "world_id": world_id,
                    "metric": metric,
                    "n_campaigns": int(group.shape[0]),
                    "mean_hybrid_minus_conventional": float(group["hybrid_minus_conventional"].mean()),
                    "median_hybrid_minus_conventional": float(group["hybrid_minus_conventional"].median()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "hybrid_win_count": int((group["hybrid_minus_conventional"] > 1e-12).sum()),
                    "hybrid_loss_count": int((group["hybrid_minus_conventional"] < -1e-12).sum()),
                }
            )
    pd.DataFrame(world_rows).to_csv(DATA / "prospective_world_paired_summary.csv", index=False)

    auc_wide = auc_table.pivot_table(index=["campaign_id", "world_id", "method_id"], columns="metric", values="best_so_far_auc", aggfunc="first").reset_index() if not auc_table.empty else pd.DataFrame()
    paired_auc_rows: list[dict[str, Any]] = []
    if not auc_wide.empty:
        auc_wide = auc_wide.rename(columns={m: f"best_so_far_auc_{m}" for m in metrics})
        auc_wide.to_csv(DATA / "prospective_best_so_far_auc_wide.csv", index=False)
        paired_auc_rows.extend(paired_metric_rows(auc_wide, [f"best_so_far_auc_{m}" for m in metrics], index_cols=["campaign_id", "world_id"]))

    if not common_auc_table.empty:
        common_auc_wide = common_auc_table.pivot_table(
            index=["campaign_id", "world_id", "method_id"],
            columns="metric",
            values=["common_horizon_best_so_far_auc", "common_horizon_mean_best_so_far", "common_horizon_final_best_so_far"],
            aggfunc="first",
        ).reset_index()
        common_auc_wide.columns = [
            "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
            for col in common_auc_wide.columns
        ]
        common_auc_wide.to_csv(DATA / "prospective_best_so_far_auc_common_budget_wide.csv", index=False)
        common_metrics = [
            f"common_horizon_best_so_far_auc_{m}" for m in metrics
        ] + [
            f"common_horizon_mean_best_so_far_{m}" for m in metrics
        ] + [
            f"common_horizon_final_best_so_far_{m}" for m in metrics
        ]
        paired_auc_rows.extend(paired_metric_rows(common_auc_wide, common_metrics, index_cols=["campaign_id", "world_id"]))
    return paired_auc_rows


def write_conventional_adaptivity_audit(exact_calls: pd.DataFrame) -> None:
    proposal = read_csv_if_exists(DATA / "prospective_proposal_dependency_audit.csv")
    out_path = DATA / "prospective_conventional_adaptivity_audit.csv"
    if proposal.empty or exact_calls.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    conv_proposals = proposal[proposal["method_id"].eq("conventional_dbtl")].copy()
    conv_calls = exact_calls[exact_calls["method_id"].eq("conventional_dbtl")].copy()
    if conv_proposals.empty or conv_calls.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    rows = []
    group_cols = ["campaign_id", "world_id", "stage_index"]
    for (campaign_id, world_id, stage_index), stage_props in conv_proposals.groupby(group_cols):
        stage_index = int(stage_index)
        stage_calls = conv_calls[
            conv_calls["campaign_id"].eq(campaign_id)
            & conv_calls["world_id"].eq(world_id)
            & (conv_calls["stage_index"].astype(int) == stage_index)
        ]
        prior_calls = conv_calls[
            conv_calls["campaign_id"].eq(campaign_id)
            & conv_calls["world_id"].eq(world_id)
            & (conv_calls["stage_index"].astype(int) < stage_index)
        ]
        proposal_started = pd.to_datetime(stage_props["proposal_started_at"], utc=True, errors="coerce").min()
        proposal_finished = pd.to_datetime(stage_props["proposal_finished_at"], utc=True, errors="coerce").max()
        current_exact_started = pd.to_datetime(stage_calls.get("exact_started_at", pd.Series(dtype=str)), utc=True, errors="coerce").min()
        prior_exact_finished = pd.to_datetime(prior_calls.get("exact_finished_at", pd.Series(dtype=str)), utc=True, errors="coerce").max()
        current_started_after_proposal = bool(pd.notna(current_exact_started) and pd.notna(proposal_finished) and current_exact_started >= proposal_finished)
        prior_observations = int(prior_calls.shape[0])
        proposal_after_prior = True if stage_index == 1 else bool(pd.notna(proposal_started) and pd.notna(prior_exact_finished) and proposal_started > prior_exact_finished)
        rows.append(
            {
                "campaign_id": campaign_id,
                "world_id": world_id,
                "method_id": "conventional_dbtl",
                "stage_index": stage_index,
                "candidate_count": int(stage_props.shape[0]),
                "current_stage_exact_cultures": int(stage_calls.shape[0]),
                "prior_exact_observations_count": prior_observations,
                "proposal_started_at": proposal_started.isoformat() if pd.notna(proposal_started) else "",
                "proposal_finished_at": proposal_finished.isoformat() if pd.notna(proposal_finished) else "",
                "prior_latest_exact_finished_at": prior_exact_finished.isoformat() if pd.notna(prior_exact_finished) else "",
                "current_stage_first_exact_started_at": current_exact_started.isoformat() if pd.notna(current_exact_started) else "",
                "proposal_after_prior_exact": proposal_after_prior,
                "exact_started_after_proposal": current_started_after_proposal,
                "prior_best_final_product_before_stage": float(prior_calls["final_product"].max()) if not prior_calls.empty else np.nan,
            }
        )
    pd.DataFrame(rows).sort_values(["world_id", "campaign_id", "stage_index"]).to_csv(out_path, index=False)


def write_leakage_audit(exact_calls: pd.DataFrame, virtual: pd.DataFrame, stages: pd.DataFrame) -> None:
    proposal = read_csv_if_exists(DATA / "prospective_proposal_dependency_audit.csv")
    adaptivity = read_csv_if_exists(DATA / "prospective_conventional_adaptivity_audit.csv")
    later_adaptivity = adaptivity[adaptivity["stage_index"].astype(int) > 1] if not adaptivity.empty else pd.DataFrame()
    rows = [
        {
            "criterion": "exact_calls_logged_for_all_physical_cultures",
            "passed": int(exact_calls.shape[0]) == int(exact_calls["physical_culture_charge"].sum()),
            "value": exact_calls.shape[0],
        },
        {
            "criterion": "virtual_search_did_not_record_exact_outcomes",
            "passed": virtual.empty or not bool(virtual.get("exact_outcome_accessed", pd.Series([False])).astype(bool).any()),
            "value": int(virtual.shape[0]),
        },
        {
            "criterion": "old_exact_cache_not_used_as_source",
            "passed": not any(str(p).startswith("data/design_benchmark_exact_rollouts") for p in exact_calls.get("trajectory_path", pd.Series(dtype=str)).astype(str)),
            "value": ",".join(sorted(set(exact_calls.get("trajectory_path", pd.Series(dtype=str)).astype(str).head(3)))),
        },
        {
            "criterion": "conventional_has_multiple_online_stages",
            "passed": int(stages[stages["method_id"].eq("conventional_dbtl")]["stage_index"].max()) >= 2 if not stages.empty else False,
            "value": int(stages[stages["method_id"].eq("conventional_dbtl")]["stage_index"].max()) if not stages.empty else 0,
        },
        {
            "criterion": "conventional_later_proposals_after_prior_exact",
            "passed": (not later_adaptivity.empty) and bool(later_adaptivity["proposal_after_prior_exact"].astype(bool).all()) and bool(later_adaptivity["exact_started_after_proposal"].astype(bool).all()),
            "value": int(later_adaptivity.shape[0]) if not later_adaptivity.empty else 0,
        },
        {
            "criterion": "hybrid_exact_verification_after_virtual_search",
            "passed": not stages[stages["method_id"].eq("pretrained_hybrid_digital_twin") & (stages["virtual_evaluations_this_stage"].astype(float) <= 0)].any(axis=None) if not stages.empty else False,
            "value": int(stages[stages["method_id"].eq("pretrained_hybrid_digital_twin")]["virtual_evaluations_this_stage"].sum()) if not stages.empty else 0,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "prospective_leakage_access_audit.csv", index=False)
    out.to_csv(DATA / "prospective_leakage_audit.csv", index=False)


def write_old_cache_audit(exact_calls: pd.DataFrame) -> None:
    cols = [
        "campaign_id",
        "world_id",
        "method_id",
        "stage_index",
        "candidate_id",
        "candidate_hash",
        "old_cache_collision",
        "old_cache_collision_sources",
        "old_result_accessed",
        "fresh_exact_simulation_performed",
        "cache_hit",
        "metabolic_backend",
    ]
    out = exact_calls[[c for c in cols if c in exact_calls.columns]].copy()
    out.to_csv(DATA / "prospective_old_cache_collision_audit.csv", index=False)


def write_exact_trajectory_bundle(exact_calls: pd.DataFrame) -> None:
    rows = []
    for _, call in exact_calls.iterrows():
        path = ROOT / str(call.get("trajectory_path", ""))
        if not path.exists():
            continue
        traj = pd.read_csv(path)
        for col in ["campaign_id", "world_id", "method_id", "stage_index", "culture_index", "candidate_id", "candidate_hash", "culture_task_id"]:
            traj[col] = call[col]
        rows.append(traj)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out.to_csv(DATA / "prospective_exact_trajectories.csv", index=False)


def write_pilot_acceptance(exact_calls: pd.DataFrame, virtual: pd.DataFrame, stages: pd.DataFrame) -> None:
    proposal = read_csv_if_exists(DATA / "prospective_proposal_dependency_audit.csv")
    adaptivity = read_csv_if_exists(DATA / "prospective_conventional_adaptivity_audit.csv")
    transfer = read_csv_if_exists(DATA / "prospective_transfer_novelty_audit.csv")

    scientific = exact_calls[~exact_calls.get("mock_mode", pd.Series(False, index=exact_calls.index)).astype(bool)].copy()
    if scientific.empty:
        scientific = exact_calls.copy()
    exact_backend_ok = bool((scientific["metabolic_backend"].astype(str) == exact.EXACT_BACKEND).all()) if not scientific.empty else False
    no_mock = bool((~exact_calls.get("mock_mode", pd.Series(False, index=exact_calls.index)).astype(bool)).all()) if not exact_calls.empty else False
    fresh_count = int(exact_calls.get("fresh_exact_simulation_performed", pd.Series(False, index=exact_calls.index)).astype(bool).sum())
    prospective_cache_count = int(exact_calls.get("cache_hit", pd.Series(False, index=exact_calls.index)).astype(bool).sum())
    on_demand_or_valid_prospective_cache = bool(no_mock and (fresh_count + prospective_cache_count == len(exact_calls)) and not exact_calls.get("old_result_accessed", pd.Series(False, index=exact_calls.index)).astype(bool).any())
    solver_health = bool(
        (scientific.get("n_infeasible_solves", pd.Series(0, index=scientific.index)).fillna(0).astype(int).sum() == 0)
        and (scientific.get("n_unbounded_solves", pd.Series(0, index=scientific.index)).fillna(0).astype(int).sum() == 0)
        and (scientific.get("n_solver_errors", pd.Series(0, index=scientific.index)).fillna(0).astype(int).sum() == 0)
        and (scientific.get("n_skipped_intervals", pd.Series(0, index=scientific.index)).fillna(0).astype(int).sum() == 0)
    )
    variation = float(scientific["final_product"].max() - scientific["final_product"].min()) if not scientific.empty else 0.0
    later_adaptivity = adaptivity[adaptivity["stage_index"].astype(int) > 1] if not adaptivity.empty else pd.DataFrame()
    adaptivity_ok = (not later_adaptivity.empty) and bool(later_adaptivity["proposal_after_prior_exact"].astype(bool).all()) and bool(later_adaptivity["exact_started_after_proposal"].astype(bool).all())
    rows = [
        ("A_exact_backend", exact_backend_ok, ",".join(sorted(scientific["metabolic_backend"].astype(str).unique())) if not scientific.empty else ""),
        ("B_no_surrogate_or_mock_lab", no_mock and bool((exact_calls["n_surrogate_evaluations"].fillna(0).astype(int) == 0).all()), f"mock_rows={int(exact_calls.get('mock_mode', pd.Series(False, index=exact_calls.index)).astype(bool).sum())};surrogate_evaluations={int(exact_calls['n_surrogate_evaluations'].fillna(0).sum())}"),
        ("C_on_demand_exact_or_valid_prospective_cache", on_demand_or_valid_prospective_cache, f"fresh={fresh_count};prospective_cache_hits={prospective_cache_count};total={len(exact_calls)}"),
        ("D_conventional_adaptivity", adaptivity_ok, f"proposal_rows={proposal.shape[0] if not proposal.empty else 0};adaptive_stages={later_adaptivity.shape[0]}"),
        ("E_hybrid_blind_virtual_search", virtual.empty or not bool(virtual.get("exact_outcome_accessed", pd.Series(False, index=virtual.index)).astype(bool).any()), f"virtual_rows={virtual.shape[0]}"),
        ("F_historical_transfer_new_edits", (not transfer.empty) and not bool(transfer["edit_vector_present_in_teacher_training"].astype(bool).any()), f"transfer_rows={transfer.shape[0] if not transfer.empty else 0}"),
        ("G_solver_health", solver_health, f"infeasible={int(scientific.get('n_infeasible_solves', pd.Series(0, index=scientific.index)).fillna(0).sum())};unbounded={int(scientific.get('n_unbounded_solves', pd.Series(0, index=scientific.index)).fillna(0).sum())};solver_errors={int(scientific.get('n_solver_errors', pd.Series(0, index=scientific.index)).fillna(0).sum())};skipped={int(scientific.get('n_skipped_intervals', pd.Series(0, index=scientific.index)).fillna(0).sum())}"),
        ("H_biological_validity", bool(scientific.get("biologically_valid", pd.Series(False, index=scientific.index)).astype(bool).all()) if not scientific.empty else False, f"valid={int(scientific.get('biologically_valid', pd.Series(False, index=scientific.index)).astype(bool).sum())}/{len(scientific)}"),
        ("I_accounting_complete", bool(len(exact_calls) > 0 and stages.shape[0] > 0 and ("wall_time_seconds" in exact_calls.columns) and ("n_actual_lp_solves" in exact_calls.columns)), f"cultures={len(exact_calls)};virtual={virtual.shape[0]};stages={stages.shape[0]};lp={int(scientific['n_actual_lp_solves'].fillna(0).sum()) if not scientific.empty else 0}"),
        ("J_reproducibility_inputs_frozen", bool(exact_calls["candidate_hash"].notna().all() and exact_calls["campaign_id"].notna().all()), f"campaigns={exact_calls['campaign_id'].nunique() if not exact_calls.empty else 0}"),
        ("candidate_quality_variation_nonconstant", variation > 1e-6, f"final_product_range={variation:.6g}"),
    ]
    out = pd.DataFrame(rows, columns=["gate", "passed", "value"])
    expected_full_cultures = 2 * 10 * (sum(FULL_CONVENTIONAL_BATCHES) + FULL_HYBRID_VERIFICATION_BATCH)
    final_status = (
        "FULL_POWERED_PROSPECTIVE_BENCHMARK_VALID"
        if bool(out["passed"].all()) and len(exact_calls) >= expected_full_cultures and exact_calls["campaign_id"].nunique() >= 20
        else "READY_FOR_FULL_PROSPECTIVE_BENCHMARK"
        if bool(out["passed"].all())
        else "BLOCKED_" + ";".join(out.loc[~out["passed"], "gate"].astype(str))
    )
    out["final_status_if_all_gates_considered"] = final_status
    out.to_csv(DATA / "prospective_pilot_acceptance.csv", index=False)


def write_full_run_resource_estimate(exact_calls: pd.DataFrame) -> None:
    scientific = exact_calls[~exact_calls.get("mock_mode", pd.Series(False, index=exact_calls.index)).astype(bool)].copy()
    if scientific.empty:
        scientific = exact_calls.copy()
    worlds = max(int(exact_calls["world_id"].nunique()), 1) if not exact_calls.empty else 2
    measured_lp_per_culture = float(scientific["n_actual_lp_solves"].median()) if not scientific.empty else float((exact.TIME_POINTS - 1) * 3)
    measured_runtime = float(scientific["wall_time_seconds"].median()) if not scientific.empty else np.nan
    full_campaigns = 10
    full_cultures_per_campaign = sum(FULL_CONVENTIONAL_BATCHES) + FULL_HYBRID_VERIFICATION_BATCH
    exact_cultures = full_campaigns * worlds * full_cultures_per_campaign
    lp_solves = exact_cultures * measured_lp_per_culture
    serial_seconds = exact_cultures * measured_runtime if np.isfinite(measured_runtime) else np.nan
    rows = []
    for parallel_workers in [1, 8, 16, 32, 64]:
        rows.append(
            {
                "scenario": f"10_campaign_seeds_x_{worlds}_worlds",
                "future_campaign_seeds": full_campaigns,
                "worlds": worlds,
                "conventional_batches": stable_json(FULL_CONVENTIONAL_BATCHES),
                "hybrid_verification_batch": FULL_HYBRID_VERIFICATION_BATCH,
                "exact_cultures_estimated": exact_cultures,
                "lp_solves_estimated": int(round(lp_solves)),
                "measured_lp_solves_per_culture_median": measured_lp_per_culture,
                "measured_wall_seconds_per_culture_median": measured_runtime,
                "parallel_workers": parallel_workers,
                "estimated_wall_hours": serial_seconds / 3600.0 / parallel_workers if np.isfinite(serial_seconds) else np.nan,
                "note": "Do not launch automatically; use one process per exact culture or bounded HPC workers with BLAS thread caps.",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "prospective_full_run_resource_estimate.csv", index=False)


def write_training_cost_amortization() -> None:
    audit = read_csv_if_exists(DATA / "prospective_historical_training_audit.csv")
    cultures = 0
    exact_solves = 0
    if not audit.empty:
        row = audit[audit["audit_item"].eq("historical_cultures")]
        cultures = int(row["value"].iloc[0]) if not row.empty else 0
        solver = audit[audit["audit_item"].eq("training_worlds_or_conditions")]
        if not solver.empty:
            text = str(solver["value"].iloc[0])
            for part in text.split(";"):
                if part.startswith("exact_lp_solves="):
                    exact_solves = int(part.split("=", 1)[1])
    rows = []
    for campaigns in [1, 2, 5, 10, 20, 50]:
        rows.append(
            {
                "future_campaigns": campaigns,
                "historical_cultures_total": cultures,
                "historical_cultures_per_future_campaign": cultures / campaigns if campaigns else np.nan,
                "historical_exact_lp_solves_total": exact_solves,
                "historical_exact_lp_solves_per_future_campaign": exact_solves / campaigns if campaigns else np.nan,
                "interpretation": "supporting amortized-development view; not primary prospective-deployment result",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "prospective_training_cost_amortization.csv", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    clean = df.reset_index()
    headers = [str(c) for c in clean.columns]
    rows = []
    for _, row in clean.iterrows():
        rows.append([str(row[c]) for c in clean.columns])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def markdown_records_table(df: pd.DataFrame) -> str:
    return markdown_table(df.reset_index(drop=True))


def line_figure(path: Path, df: pd.DataFrame, x: str, y: str, hue: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key, sub in df.groupby(hue):
        curve = sub.groupby(x)[y].mean().reset_index()
        ax.plot(curve[x], curve[y], marker="o", label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_figures(exact_calls: pd.DataFrame, stages: pd.DataFrame, summary: pd.DataFrame, match: pd.DataFrame) -> None:
    if stages.empty:
        return
    line_figure(FIGURES / "prospective_03_best_final_product_vs_stage.svg", stages, "stage_index", "best_verified_final_product_so_far", "method_id", "Best verified final product vs biological stage", "Best final product")
    line_figure(FIGURES / "prospective_04_best_productivity_vs_stage.svg", stages, "stage_index", "best_verified_productivity_so_far", "method_id", "Best verified productivity vs biological stage", "Best productivity")
    line_figure(FIGURES / "prospective_05_best_final_product_vs_time.svg", stages, "scenario_days_cumulative", "best_verified_final_product_so_far", "method_id", "Best verified final product vs scenario time", "Best final product")
    line_figure(FIGURES / "prospective_06_best_productivity_vs_time.svg", stages, "scenario_days_cumulative", "best_verified_productivity_so_far", "method_id", "Best verified productivity vs scenario time", "Best productivity")
    line_figure(FIGURES / "prospective_07_best_product_vs_cultures.svg", stages, "physical_cultures_cumulative", "best_verified_final_product_so_far", "method_id", "Best verified product vs physical cultures", "Best final product")

    import matplotlib.pyplot as plt

    best = read_csv_if_exists(DATA / "prospective_best_so_far_by_culture.csv")
    if not best.empty:
        figure_specs = [
            ("best_final_product_so_far", "Best final product", "prospective_12_best_final_product_vs_cultures_full.svg"),
            ("best_productivity_so_far", "Best productivity", "prospective_13_best_productivity_vs_cultures_full.svg"),
            ("best_product_AUC_so_far", "Best product AUC", "prospective_14_best_product_auc_vs_cultures_full.svg"),
        ]
        for y, ylabel, filename in figure_specs:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            for (_campaign_id, _world_id, method_id), sub in best.groupby(["campaign_id", "world_id", "method_id"]):
                ax.plot(sub["physical_culture_index"], sub[y], color="tab:blue" if method_id == "conventional_dbtl" else "tab:orange", alpha=0.18, linewidth=1.0)
            for method_id, sub in best.groupby("method_id"):
                curve = sub.groupby("physical_culture_index")[y].median().reset_index()
                ax.plot(curve["physical_culture_index"], curve[y], marker="o", linewidth=2.2, label=f"{method_id} median")
            ax.set_xlabel("Physical cultures requested by workflow")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs physical cultures")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(FIGURES / filename)
            plt.close(fig)

    paired_campaigns = read_csv_if_exists(DATA / "prospective_final_best_candidates.csv")
    if not paired_campaigns.empty:
        piv = paired_campaigns.pivot_table(index=["campaign_id", "world_id"], columns="method_id", values="best_final_product", aggfunc="first").reset_index()
        if {"conventional_dbtl", "pretrained_hybrid_digital_twin"}.issubset(piv.columns):
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            colors = ["tab:blue" if w == "baseline_world" else "tab:red" for w in piv["world_id"]]
            ax.scatter(piv["conventional_dbtl"], piv["pretrained_hybrid_digital_twin"], c=colors, alpha=0.8)
            lo = float(np.nanmin([piv["conventional_dbtl"].min(), piv["pretrained_hybrid_digital_twin"].min()]))
            hi = float(np.nanmax([piv["conventional_dbtl"].max(), piv["pretrained_hybrid_digital_twin"].max()]))
            ax.plot([lo, hi], [lo, hi], color="0.35", linewidth=1.2)
            ax.set_xlabel("Conventional best final product")
            ax.set_ylabel("Hybrid best final product")
            ax.set_title("Paired campaign final-product comparison")
            fig.tight_layout()
            fig.savefig(FIGURES / "prospective_15_paired_final_product_scatter.svg")
            plt.close(fig)

    world = read_csv_if_exists(DATA / "prospective_world_paired_summary.csv")
    if not world.empty:
        sub = world[world["metric"].eq("best_final_product")]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.bar(sub["world_id"], sub["mean_hybrid_minus_conventional"], color=["tab:blue" if w == "baseline_world" else "tab:red" for w in sub["world_id"]])
            ax.axhline(0.0, color="0.25", linewidth=1.0)
            ax.set_ylabel("Mean hybrid - conventional")
            ax.set_title("Per-world paired final-product difference")
            ax.tick_params(axis="x", rotation=15)
            fig.tight_layout()
            fig.savefig(FIGURES / "prospective_16_world_paired_final_product_difference.svg")
            plt.close(fig)

    thresholds = read_csv_if_exists(DATA / "prospective_relative_threshold_efficiency.csv")
    if not thresholds.empty:
        sub = thresholds[(thresholds["metric"].eq("final_product")) & (thresholds["relative_fraction_of_campaign_observed_best"].astype(float).eq(0.90))]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            plot = sub.groupby("method_id")["cultures_to_threshold"].median().sort_index()
            plot.plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange"])
            ax.set_ylabel("Median cultures to 90% observed-best")
            ax.set_title("Relative threshold efficiency")
            fig.tight_layout()
            fig.savefig(FIGURES / "prospective_17_cultures_to_relative_threshold.svg")
            plt.close(fig)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        acct = summary.groupby("method_id")[["physical_cultures", "virtual_candidates"]].sum()
        acct.plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Physical cultures and virtual evaluations")
        fig.tight_layout()
        fig.savefig(FIGURES / "prospective_18_virtual_vs_physical_accounting.svg")
        plt.close(fig)

    timeline = exact_calls.copy()
    if "exact_started_at" in timeline.columns:
        timeline["exact_started_at"] = pd.to_datetime(timeline["exact_started_at"], errors="coerce")
        timeline = timeline.sort_values("exact_started_at")
        if timeline["exact_started_at"].notna().any():
            t0 = timeline["exact_started_at"].min()
            timeline["minutes_since_start"] = (timeline["exact_started_at"] - t0).dt.total_seconds() / 60.0
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            for method_id, sub in timeline.groupby("method_id"):
                ax.scatter(sub["minutes_since_start"], sub["stage_index"], label=method_id, s=36, alpha=0.8)
            ax.set_xlabel("Minutes since first exact request")
            ax.set_ylabel("Biological stage")
            ax.set_title("Prospective exact simulator call timeline")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(FIGURES / "prospective_00_exact_call_timeline.svg")
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method_id, sub in exact_calls.groupby("method_id"):
        ax.scatter(sub["final_product"], sub["productivity"], label=method_id, alpha=0.75)
    ax.set_xlabel("Final product")
    ax.set_ylabel("Productivity")
    ax.set_title("Final-product vs productivity Pareto view")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prospective_09_pareto_frontier.svg")
    plt.close(fig)

    virtual = read_csv_if_exists(DATA / "prospective_virtual_evaluation_ledger.csv")
    if not virtual.empty and "hybrid_virtual_score" in virtual.columns:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.hist(virtual["hybrid_virtual_score"].astype(float), bins=32, color="0.75", edgecolor="white", label="virtual candidates")
        selected = virtual[virtual.get("selected_for_exact_verification", pd.Series(False, index=virtual.index)).astype(bool)]
        if not selected.empty:
            for value in selected["hybrid_virtual_score"].astype(float):
                ax.axvline(value, color="tab:red", alpha=0.35, linewidth=1.4)
        ax.set_xlabel("Hybrid virtual score")
        ax.set_ylabel("Candidate count")
        ax.set_title("Hybrid virtual distribution and selected exact verifications")
        fig.tight_layout()
        fig.savefig(FIGURES / "prospective_11_virtual_distribution_selected.svg")
        plt.close(fig)

    if not match.empty:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        vals = match["earliest_conventional_stage_matching_hybrid_titer"].fillna(999)
        labels = [">max" if v == 999 else str(int(v)) for v in vals]
        pd.Series(labels).value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Conventional stage matching hybrid titer")
        ax.set_ylabel("Campaign count")
        ax.set_title("Stages required to match hybrid result")
        fig.tight_layout()
        fig.savefig(FIGURES / "prospective_10_conventional_stages_to_match_hybrid.svg")
        plt.close(fig)

    best_rows = []
    for (_campaign_id, method_id), group in exact_calls.groupby(["campaign_id", "method_id"]):
        idx = group["final_product"].astype(float).idxmax()
        best_rows.append(exact_calls.loc[idx])
    for i, row in enumerate(best_rows[:4], start=1):
        path = ROOT / str(row["trajectory_path"])
        if not path.exists():
            continue
        traj = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(traj["time"], traj["B_total"], label="product")
        ax2 = ax.twinx()
        ax2.plot(traj["time"], traj["X"], color="tab:green", label="biomass")
        ax.set_xlabel("Time")
        ax.set_ylabel("Product")
        ax2.set_ylabel("Biomass")
        ax.set_title(f"Representative trajectory {i}: {row['method_id']}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"prospective_08_representative_trajectory_{i}.svg")
        plt.close(fig)

    workflow_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">
<style>text{font-family:Arial,sans-serif;font-size:14px}.h{font-size:18px;font-weight:bold}.box{fill:#f7f7f7;stroke:#333;stroke-width:1.4}.a{stroke:#333;stroke-width:1.5;marker-end:url(#m)}</style>
<defs><marker id="m" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#333"/></marker></defs>
<text x="40" y="35" class="h">Prospective benchmark architecture</text>
<rect x="45" y="80" width="180" height="65" class="box"/><text x="65" y="118">Method proposes culture</text>
<line x1="225" y1="112" x2="325" y2="112" class="a"/><rect x="325" y="80" width="190" height="65" class="box"/><text x="345" y="108">On-demand exact</text><text x="345" y="126">Yeast9 wet lab</text>
<line x1="515" y1="112" x2="615" y2="112" class="a"/><rect x="615" y="80" width="220" height="65" class="box"/><text x="635" y="108">Result returned and</text><text x="635" y="126">ledgered</text>
<rect x="45" y="215" width="250" height="75" class="box"/><text x="65" y="245">Conventional: later batches</text><text x="65" y="265">depend on prior exact results</text>
<rect x="390" y="215" width="300" height="75" class="box"/><text x="410" y="245">Hybrid: frozen teacher plus Yeast9</text><text x="410" y="265">virtual search before verification</text>
</svg>"""
    (FIGURES / "prospective_01_architecture.svg").write_text(workflow_svg, encoding="utf-8")
    dependency_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="320" viewBox="0 0 900 320">
<style>text{font-family:Arial,sans-serif;font-size:14px}.h{font-size:18px;font-weight:bold}.box{fill:#f7f7f7;stroke:#333;stroke-width:1.4}.a{stroke:#333;stroke-width:1.5;marker-end:url(#m)}</style>
<defs><marker id="m" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#333"/></marker></defs>
<text x="40" y="35" class="h">Conventional dependency graph vs hybrid workflow</text>
<rect x="45" y="80" width="100" height="45" class="box"/><text x="70" y="108">Batch 1</text><line x1="145" y1="102" x2="220" y2="102" class="a"/><rect x="220" y="80" width="100" height="45" class="box"/><text x="248" y="108">Exact</text><line x1="320" y1="102" x2="395" y2="102" class="a"/><rect x="395" y="80" width="100" height="45" class="box"/><text x="420" y="108">Batch 2</text>
<rect x="45" y="205" width="150" height="45" class="box"/><text x="62" y="233">Frozen twin</text><line x1="195" y1="227" x2="280" y2="227" class="a"/><rect x="280" y="205" width="170" height="45" class="box"/><text x="300" y="233">Virtual search</text><line x1="450" y1="227" x2="535" y2="227" class="a"/><rect x="535" y="205" width="170" height="45" class="box"/><text x="555" y="233">Exact verify</text>
</svg>"""
    (FIGURES / "prospective_02_dependency_graph.svg").write_text(dependency_svg, encoding="utf-8")


def write_report(summary: pd.DataFrame, paired_rows: list[dict[str, Any]]) -> None:
    path = DATA / "prospective_dbtl_final_report.md"
    lines = [
        "# Prospective On-Demand DBTL Benchmark",
        "",
        f"Benchmark ID: `{BENCHMARK_ID}`.",
        "",
        "This experiment compares `conventional_dbtl` against `pretrained_hybrid_digital_twin` using an on-demand simulated wet lab. Exact simulator calls are recorded only when a workflow requests a culture.",
        "",
    ]
    if not summary.empty:
        total_cultures = int(summary["physical_cultures"].sum())
        total_virtual = int(summary["virtual_candidates"].sum())
        total_lp_solves = int(summary["exact_lp_solves"].sum()) if "exact_lp_solves" in summary.columns else 0
        mock = bool(summary["mock_mode"].astype(bool).any())
        acceptance = read_csv_if_exists(DATA / "prospective_pilot_acceptance.csv")
        final_status = str(acceptance["final_status_if_all_gates_considered"].iloc[0]) if not acceptance.empty else "not_evaluated"
        gates_passed = int(acceptance["passed"].astype(bool).sum()) if not acceptance.empty else 0
        gates_total = int(acceptance.shape[0]) if not acceptance.empty else 0
        lines.extend(
            [
                "## Headline Accounting",
                "",
                f"- New exact/physical culture charges: `{total_cultures}`.",
                f"- Exact Yeast9 LP solves: `{total_lp_solves}`.",
                f"- Hybrid virtual candidates evaluated: `{total_virtual}`.",
                f"- Mock mode present: `{mock}`.",
                f"- Pilot acceptance gates: `{gates_passed}/{gates_total}`.",
                f"- Current status: `{final_status}`.",
                "",
                "## Method Summary",
                "",
                markdown_table(summary.groupby("method_id")[["best_final_product", "best_productivity", "best_product_AUC", "physical_cultures", "virtual_candidates"]].mean().round(6)),
                "",
            ]
        )
    if paired_rows:
        lines.extend(["## Paired Comparisons", "", markdown_records_table(pd.DataFrame(paired_rows).round(6)), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> None:
    ensure_dirs()
    audit, ok = audit_historical_training_data()
    if not ok:
        print("Historical training-data audit failed. Benchmark stopped.")
        print(audit.to_string(index=False))
        return
    if args.reset:
        reset_outputs()
        audit.to_csv(DATA / "prospective_historical_training_audit.csv", index=False)
    library = load_library()
    teacher_lookup = load_teacher_final_lookup()
    worlds = [w.strip() for w in args.world_ids.split(",") if w.strip()] if args.world_ids else PROSPECTIVE_WORLDS[: args.worlds]
    unknown_worlds = sorted(set(worlds) - set(PROSPECTIVE_WORLDS))
    if unknown_worlds:
        raise ValueError(f"Unknown prospective worlds: {unknown_worlds}")
    seeds = CAMPAIGN_SEEDS[: args.campaigns]
    conventional_batches = [int(x) for x in args.conventional_batches.split(",")] if args.conventional_batches else (PILOT_CONVENTIONAL_BATCHES if args.pilot else FULL_CONVENTIONAL_BATCHES)
    hybrid_batch = args.hybrid_batch or (PILOT_HYBRID_VERIFICATION_BATCH if args.pilot else FULL_HYBRID_VERIFICATION_BATCH)
    virtual_evaluations = args.virtual_evaluations or (PILOT_VIRTUAL_EVALUATIONS if args.pilot else FULL_VIRTUAL_EVALUATIONS)
    protocol_lp_solves = (exact.TIME_POINTS - 1) * 3
    estimate = {
        "benchmark_id": BENCHMARK_ID,
        "worlds": len(worlds),
        "world_ids": worlds,
        "campaigns_per_world": len(seeds),
        "conventional_cultures_per_campaign": sum(conventional_batches),
        "hybrid_cultures_per_campaign": hybrid_batch,
        "expected_physical_cultures": len(worlds) * len(seeds) * (sum(conventional_batches) + hybrid_batch),
        "verified_dynamic_protocol_time_points": exact.TIME_POINTS,
        "verified_dynamic_protocol_intervals": exact.TIME_POINTS - 1,
        "verified_expected_lp_solves_per_culture": protocol_lp_solves,
        "expected_lp_solves": 0 if args.mock else len(worlds) * len(seeds) * (sum(conventional_batches) + hybrid_batch) * protocol_lp_solves,
        "expected_virtual_evaluations": len(worlds) * len(seeds) * virtual_evaluations,
        "hpc_note": "Use one process per exact culture with BLAS threads capped for full exact execution.",
    }
    DATA.joinpath("prospective_resource_estimate.json").write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for world_id in worlds:
        for seed in seeds:
            run_campaign(
                world_id=world_id,
                seed=seed,
                library=library,
                teacher_lookup=teacher_lookup,
                conventional_batches=conventional_batches,
                hybrid_batch=hybrid_batch,
                virtual_evaluations=virtual_evaluations,
                mock=args.mock,
                fresh_exact=args.fresh_exact,
            )
    write_analysis()
    print(json.dumps(estimate, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Use a small pilot budget.")
    parser.add_argument("--mock", action="store_true", help="Run the workflow with a deterministic mock wet lab for smoke testing only.")
    parser.add_argument("--reset", action="store_true", help="Clear prospective benchmark output ledgers before running.")
    parser.add_argument("--campaigns", type=int, default=2, help="Number of campaign seeds per selected world.")
    parser.add_argument("--worlds", type=int, default=3, help="Number of prospective worlds from the predeclared list.")
    parser.add_argument("--world-ids", default="", help="Comma-separated frozen world ids to use instead of the first --worlds entries.")
    parser.add_argument("--conventional-batches", default="", help="Comma-separated conventional batch sizes, e.g. 2,2,2.")
    parser.add_argument("--hybrid-batch", type=int, default=0, help="Override hybrid exact verification batch size.")
    parser.add_argument("--virtual-evaluations", type=int, default=0, help="Override hybrid virtual evaluations per campaign.")
    parser.add_argument("--fresh-exact", action="store_true", help="In exact mode, ignore prospective rollout sidecars and run a fresh exact simulation for each requested culture.")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
