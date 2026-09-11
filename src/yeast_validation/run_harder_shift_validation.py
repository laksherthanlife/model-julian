#!/usr/bin/env python3
"""Harder-world and distribution-shift validation campaign scaffold.

This script prepares the next exact Yeast9 validation layer without changing
the neural architecture.  It freezes biologically interpretable hidden worlds,
source-to-target shift pairs, candidate manifests, target rules, exact-cache
fingerprints, and PBS array scripts.  Heavy exact work is deliberately exposed
as independent candidate tasks so it can run on the HPC with one fresh Python
process per candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import design_benchmark_exact as exact
import final_dbtl_benchmark_freeze as freeze
import gem_backend as gem
import run_deployment_verifier as verifier
import run_final_dbtl_benchmark as final_run


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
HARDER_RESULTS = RESULTS / "harder_dbtl"
SHIFT_RESULTS = RESULTS / "distribution_shift"
HPC_ROOT = ROOT / "hpc_bundles" / "harder_shift_validation_vanda"
HARDER_REPORT = DATA / "harder_dbtl_validation_report.md"
SHIFT_REPORT = DATA / "distribution_shift_validation_report.md"

WORLD_IDS = [
    "baseline_world",
    "strong_oxidative_burden_world",
    "atp_limited_world",
    "precursor_competition_world",
    "pathway_bottleneck_damage_world",
    "compensatory_metabolism_world",
    "mixed_multi_mechanism_world",
]
METHOD_IDS = freeze.FINAL_METHODS
CAMPAIGN_SEEDS = [62001, 62002, 62003]
ROUND_BATCHES = [8, 4, 4]
ADAPTATION_BUDGETS = [0, 2, 4, 8]
BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
SIMULATOR_VERSION = "harder_shift_exact_dynamic_pfba_v1"
TARGET_RULE_VERSION = "harder_shift_multicriterion_targets_v1"


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_payload(payload: object) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in [DATA, FIGURES, HARDER_RESULTS, SHIFT_RESULTS]:
        path.mkdir(parents=True, exist_ok=True)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    clean = frame.head(max_rows).copy() if max_rows else frame.copy()
    for col in clean.columns:
        if pd.api.types.is_float_dtype(clean[col]):
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(clean.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in clean.columns) + " |" for _, row in clean.iterrows()]
    return "\n".join([header, sep, *rows])


def safe_rank_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def selected_world_specs() -> dict[str, dict[str, object]]:
    configs = exact.regime_configs()
    missing = [world_id for world_id in WORLD_IDS if world_id not in configs]
    if missing:
        raise KeyError(f"Missing exact world definitions: {missing}")
    return {world_id: configs[world_id] for world_id in WORLD_IDS}


def world_payload(world_id: str, spec: dict[str, object], gem_checksum: str) -> dict[str, Any]:
    cfg = spec["cfg"]
    payload = {
        "world_id": world_id,
        "world_family": spec.get("world_family", world_id),
        "biological_interpretation": spec.get("biological_interpretation", ""),
        "simulator_version": SIMULATOR_VERSION,
        "regulatory_tree_version": freeze.FINAL_REGULATOR_VERSION,
        "time_grid": f"{exact.TIME_POINTS}_points_{exact.TIME_POINTS - 1}_intervals_dt_0p25",
        "gem_checksum": gem_checksum,
        "random_seed": int(spec["random_seed"]),
        "burden_scale": spec["burden_scale"],
        "repair_scale": float(spec["repair_scale"]),
        "pathway_damage_scale": float(spec["pathway_damage_scale"]),
        "degradation_rule": str(spec["degradation_rule"]),
        "cfg": asdict(cfg),
        "status": "frozen_before_method_evaluation",
    }
    payload["world_hash"] = hash_payload(payload)
    return payload


def write_world_manifest(gem_path: str | None = None) -> pd.DataFrame:
    source = gem.configured_gem_path(gem_path)
    checksum = gem.sha256(source)
    rows = [world_payload(world_id, spec, checksum) for world_id, spec in selected_world_specs().items()]
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"cfg", "burden_scale"}}
        flat["burden_scale_json"] = stable_json(row["burden_scale"])
        for key, value in row["cfg"].items():
            flat[f"cfg_{key}"] = value
        flat_rows.append(flat)
    out = pd.DataFrame(flat_rows)
    out.to_csv(DATA / "harder_shift_world_manifest.csv", index=False)
    (DATA / "harder_shift_world_specs.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_library() -> pd.DataFrame:
    path = DATA / "final_dbtl_actionable_intervention_library.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/final_dbtl_benchmark_freeze.py first to create the actionable intervention library.")
    library = pd.read_csv(path)
    if "mechanism_class" not in library.columns and "intervention_class" in library.columns:
        library = library.rename(columns={"intervention_class": "mechanism_class"})
    if "risk_level" not in library.columns:
        library["risk_level"] = library.get("pleiotropy_warning", "medium")
    return library


def candidate_hash(candidate: dict[str, Any]) -> str:
    return freeze.candidate_spec_hash(candidate)


def exact_task_hash(world_id: str, world_hash: str, candidate_hash_value: str, task_family: str) -> str:
    return hash_payload(
        {
            "task_family": task_family,
            "world_id": world_id,
            "world_hash": world_hash,
            "candidate_hash": candidate_hash_value,
            "simulator_version": SIMULATOR_VERSION,
            "lp_backend": exact.EXACT_BACKEND,
        }
    )[:24]


def make_parent_candidate(env: dict[str, float], label: str) -> dict[str, Any]:
    return freeze.parent_candidate(env, candidate_id=label)


def calibration_candidates(library: pd.DataFrame, n_random: int = 64, seed: int = 62111) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    envs = final_run.environment_grid()
    records = library.to_dict("records")
    candidates: list[dict[str, Any]] = []
    for i, env in enumerate(envs):
        candidates.append(make_parent_candidate(env, f"harder_gate_parent_env_{i:02d}"))
    for i, row in enumerate(records):
        env = dict(envs[i % len(envs)])
        candidates.append(final_run.make_candidate("landscape_calibration", "harder_gate", len(candidates) + 1, [pd.Series(row)], env, "fixed landscape calibration single edit"))
    by_mech: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_mech.setdefault(str(row.get("mechanism_class", "")), []).append(row)
    mechanisms = sorted(k for k, v in by_mech.items() if v)
    for i in range(min(n_random, 80)):
        edit_count = int(rng.choice([1, 2, 3], p=[0.30, 0.50, 0.20]))
        selected = []
        used = set()
        for _ in range(edit_count):
            mech = mechanisms[int(rng.integers(0, len(mechanisms)))] if mechanisms else ""
            pool = by_mech.get(mech, records)
            pick = pool[int(rng.integers(0, len(pool)))]
            if pick["intervention_id"] in used:
                continue
            used.add(pick["intervention_id"])
            selected.append(pd.Series(pick))
        if not selected:
            selected = [pd.Series(records[int(rng.integers(0, len(records)))])]
        env = dict(envs[int(rng.integers(0, len(envs)))])
        candidates.append(final_run.make_candidate("landscape_calibration", "harder_gate", len(candidates) + 1, selected, env, "fixed landscape calibration mechanism-balanced sample"))
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate["candidate_hash"] = candidate_hash(candidate)
        key = final_run.canonical_design_key(candidate["edits"], candidate["environment"])
        unique.setdefault(key, candidate)
    return list(unique.values())


def proposal_candidates(library: pd.DataFrame, campaign_id: str, method_id: str, seed: int) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    start_index = 0
    seen_designs: set[str] = set()
    for batch_size in ROUND_BATCHES:
        pool = final_run.proposal_pool(method_id, campaign_id, seed, library, observations, start_index, pool_size=420)
        selected = []
        for candidate in pool:
            key = final_run.canonical_design_key(candidate["edits"], candidate["environment"])
            if key in seen_designs:
                continue
            candidate["candidate_hash"] = candidate_hash(candidate)
            selected.append(candidate)
            seen_designs.add(key)
            if len(selected) == batch_size:
                break
        if len(selected) < batch_size:
            raise RuntimeError(f"{method_id} {campaign_id} generated only {len(selected)}/{batch_size} candidates")
        out.extend(selected)
        start_index += len(selected)
        observations.extend(
            {
                "candidate_hash": c["candidate_hash"],
                "utility": 0.0,
                "mechanism_classes": final_run.candidate_metadata(c, library)["mechanism_classes"],
            }
            for c in selected
        )
    return out


def task_rows_for_candidates(
    candidates: list[dict[str, Any]],
    world_manifest: pd.DataFrame,
    task_family: str,
    campaign_family: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    extra = extra or {}
    for world in world_manifest.itertuples(index=False):
        world_id = str(world.world_id)
        world_hash = str(world.world_hash)
        for candidate in candidates:
            h = str(candidate["candidate_hash"])
            task_id = exact_task_hash(world_id, world_hash, h, task_family)
            rows.append(
                {
                    "exact_task_id": task_id,
                    "task_family": task_family,
                    "campaign_family": campaign_family,
                    "world_id": world_id,
                    "world_hash": world_hash,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_hash": h,
                    "strain_id": candidate.get("strain_id", ""),
                    "candidate_json": stable_json(candidate),
                    "temperature": float(candidate["environment"]["temperature"]),
                    "pH": float(candidate["environment"]["pH"]),
                    "DO": float(candidate["environment"]["DO"]),
                    "edit_count": int(len(candidate.get("edits", []))),
                    "expected_lp_solves": exact.EXPECTED_LP_SOLVES,
                    "exact_status": "pending_exact_dynamic_pfba",
                    "scheduler_job_id": "",
                    "output_location": str((RESULTS / campaign_family / "exact_rollouts" / world_id / f"{task_id}_summary.csv").relative_to(ROOT)),
                    **extra,
                }
            )
    return rows


def write_target_rules() -> dict[str, Any]:
    rules = {
        "target_rule_id": TARGET_RULE_VERSION,
        "status": "frozen_before_world_difficulty_and_method_outcomes",
        "parent_reference_policy": "computed per world from parent no-edit exact controls; no method outcomes used",
        "utility_reference_policy": "use final_dbtl_engineering_utility_v1 components with parent values recomputed per world",
        "tiers": {
            "easy": {
                "minimum_final_titer_fold_parent": 1.15,
                "minimum_final_biomass_fold_parent": 0.85,
                "minimum_yield_fold_parent": 0.90,
                "minimum_utility_delta_parent": 0.05,
                "maximum_interventions": 3,
                "severe_burden_allowed": False,
            },
            "moderate": {
                "minimum_final_titer_fold_parent": 1.35,
                "minimum_final_biomass_fold_parent": 0.90,
                "minimum_yield_fold_parent": 0.95,
                "minimum_utility_delta_parent": 0.15,
                "maximum_interventions": 3,
                "severe_burden_allowed": False,
            },
            "hard": {
                "minimum_final_titer_fold_parent": 1.70,
                "minimum_final_biomass_fold_parent": 0.92,
                "minimum_yield_fold_parent": 1.00,
                "minimum_utility_delta_parent": 0.30,
                "maximum_interventions": 2,
                "severe_burden_allowed": False,
            },
        },
        "not_tuned_to_method_outcomes": True,
    }
    rules["target_rule_hash"] = hash_payload(rules)
    (DATA / "harder_shift_target_rules.json").write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rules


def write_shift_pairs(world_manifest: pd.DataFrame) -> pd.DataFrame:
    by_world = world_manifest.set_index("world_id")
    specs = [
        ("baseline_world", "baseline_world", "in_distribution", "source and target are the current validated world"),
        ("baseline_world", "strong_oxidative_burden_world", "strong_shift", "high oxygen/product flux produces stronger oxidative damage"),
        ("baseline_world", "atp_limited_world", "moderate_shift", "maintenance and ATP sensitivity increase"),
        ("baseline_world", "precursor_competition_world", "moderate_shift", "native isoprenoid competition increases"),
        ("baseline_world", "pathway_bottleneck_damage_world", "strong_shift", "pathway enzyme damage and limiting step change"),
        ("baseline_world", "compensatory_metabolism_world", "mild_shift", "alternative support partially buffers simple edits"),
        ("baseline_world", "mixed_multi_mechanism_world", "strong_shift", "oxygen, ATP, precursor, and enzyme damage shift together"),
    ]
    rows = []
    for i, (source, target, tier, rationale) in enumerate(specs, start=1):
        payload = {
            "pair_id": f"shift_pair_{i:03d}_{source}_to_{target}",
            "source_world_id": source,
            "target_world_id": target,
            "shift_tier": tier,
            "source_world_hash": str(by_world.loc[source, "world_hash"]),
            "target_world_hash": str(by_world.loc[target, "world_hash"]),
            "adaptation_budgets": ADAPTATION_BUDGETS,
            "rationale": rationale,
            "status": "frozen_before_evaluation",
        }
        payload["pair_hash"] = hash_payload(payload)
        rows.append(payload)
    out = pd.DataFrame(rows)
    out["adaptation_budgets"] = out["adaptation_budgets"].map(stable_json)
    out.to_csv(DATA / "distribution_shift_pair_manifest.csv", index=False)
    return out


def write_candidate_manifests(n_calibration: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    worlds = write_world_manifest()
    write_target_rules()
    pairs = write_shift_pairs(worlds)
    library = load_library()
    calibration = calibration_candidates(library, n_random=n_calibration)
    hard_rows = task_rows_for_candidates(calibration, worlds, "difficulty_gate_landscape", "harder_dbtl", {"role": "difficulty_acceptance_gate"})
    for campaign_idx, seed in enumerate(CAMPAIGN_SEEDS, start=1):
        campaign_id = f"harder_dbtl_campaign_{campaign_idx:03d}"
        for method_id in METHOD_IDS:
            candidates = proposal_candidates(library, campaign_id, method_id, seed)
            rows = task_rows_for_candidates(
                candidates,
                worlds,
                "sequential_dbtl",
                "harder_dbtl",
                {"role": "sequential_dbtl", "campaign_id": campaign_id, "campaign_seed": seed, "method_id": method_id},
            )
            for i, row in enumerate(rows):
                row["planned_culture_index"] = i % sum(ROUND_BATCHES) + 1
                row["planned_round_index"] = 1 if row["planned_culture_index"] <= 8 else 2 if row["planned_culture_index"] <= 12 else 3
            hard_rows.extend(rows)
    harder = pd.DataFrame(hard_rows).drop_duplicates(["exact_task_id"], keep="first")
    harder.insert(0, "array_index", np.arange(1, len(harder) + 1))
    harder.to_csv(DATA / "harder_dbtl_exact_task_manifest.csv", index=False)

    shift_exact_by_task: dict[str, dict[str, Any]] = {}
    shift_decision_rows = []
    for pair in pairs.itertuples(index=False):
        target_world = worlds[worlds["world_id"].eq(str(pair.target_world_id))]
        if target_world.empty:
            continue
        source_world = str(pair.source_world_id)
        target_world_id = str(pair.target_world_id)
        for campaign_idx, seed in enumerate(CAMPAIGN_SEEDS, start=1):
            campaign_id = f"distribution_shift_campaign_{campaign_idx:03d}"
            for method_id in METHOD_IDS:
                candidates = proposal_candidates(library, campaign_id, method_id, seed + 101 * campaign_idx)
                rows = task_rows_for_candidates(
                    candidates,
                    target_world,
                    "distribution_shift_dbtl",
                    "distribution_shift",
                    {
                        "role": "target_world_exact_decision_candidate",
                        "pair_id": str(pair.pair_id),
                        "pair_hash": str(pair.pair_hash),
                        "source_world_id": source_world,
                        "target_world_id": target_world_id,
                        "shift_tier": str(pair.shift_tier),
                        "campaign_id": campaign_id,
                        "campaign_seed": seed,
                        "method_id": method_id,
                    },
                )
                for i, row in enumerate(rows, start=1):
                    row["planned_culture_index"] = i
                    row["planned_round_index"] = 1 if i <= 8 else 2 if i <= 12 else 3
                    shift_exact_by_task.setdefault(str(row["exact_task_id"]), row)
                    for budget in ADAPTATION_BUDGETS:
                        shift_decision_rows.append(
                            {
                                "decision_row_id": hash_payload(
                                    {
                                        "exact_task_id": row["exact_task_id"],
                                        "adaptation_budget": budget,
                                        "method_id": method_id,
                                        "campaign_id": campaign_id,
                                        "pair_id": str(pair.pair_id),
                                    }
                                )[:24],
                                "exact_task_id": row["exact_task_id"],
                                "pair_id": str(pair.pair_id),
                                "pair_hash": str(pair.pair_hash),
                                "source_world_id": source_world,
                                "target_world_id": target_world_id,
                                "shift_tier": str(pair.shift_tier),
                                "adaptation_budget": int(budget),
                                "evaluation_role": "zero_shot_shift" if budget == 0 else "few_shot_adaptation",
                                "campaign_id": campaign_id,
                                "campaign_seed": seed,
                                "method_id": method_id,
                                "candidate_id": row["candidate_id"],
                                "candidate_hash": row["candidate_hash"],
                                "planned_culture_index": i,
                                "planned_round_index": row["planned_round_index"],
                            }
                        )
    pd.DataFrame(shift_decision_rows).to_csv(DATA / "distribution_shift_decision_manifest.csv", index=False)
    shift = pd.DataFrame(list(shift_exact_by_task.values())).drop_duplicates(["exact_task_id"], keep="first")
    shift.insert(0, "array_index", np.arange(1, len(shift) + 1))
    shift.to_csv(DATA / "distribution_shift_exact_task_manifest.csv", index=False)
    write_method_access_matrix()
    write_hpc_execution_manifest(harder, shift)
    return harder, shift


def write_method_access_matrix() -> pd.DataFrame:
    rows = []
    for method in METHOD_IDS:
        rows.append(
            {
                "method_id": method,
                "source_world_training_data": method in {"hybrid_digital_twin", "direct_trajectory_surrogate"},
                "public_static_gem_information": method in {"hybrid_digital_twin", "static_gem_public_rank", "public_bundle_dbtL_scientist_agent"},
                "own_target_world_observations_only": method in {"hybrid_digital_twin", "black_box_outcome_only_bo", "direct_trajectory_surrogate", "public_bundle_dbtL_scientist_agent"},
                "hidden_target_world_cache_access": False,
                "other_method_target_observation_access": False,
                "target_world_parameters_visible": False,
                "same_round_budget": True,
                "same_culture_budget": True,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "harder_shift_method_access_matrix.csv", index=False)
    return out


def write_hpc_execution_manifest(harder: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "campaign_family": "harder_dbtl",
            "manifest": "data/harder_dbtl_exact_task_manifest.csv",
            "array_tasks": int(len(harder)),
            "queue": "default_route_to_batch_cpu_observed_on_vanda",
            "ncpus_per_task": 1,
            "memory_per_task": "6gb",
            "walltime_per_task": "04:00:00",
            "scratch_root": "/scratch/e1471252/harder_shift_validation",
            "process_isolation": "one fresh Python process per exact task",
        },
        {
            "campaign_family": "distribution_shift",
            "manifest": "data/distribution_shift_exact_task_manifest.csv",
            "array_tasks": int(len(shift)),
            "queue": "default_route_to_batch_cpu_observed_on_vanda",
            "ncpus_per_task": 1,
            "memory_per_task": "6gb",
            "walltime_per_task": "04:00:00",
            "scratch_root": "/scratch/e1471252/harder_shift_validation",
            "process_isolation": "one fresh Python process per exact task",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "harder_shift_hpc_execution_manifest.csv", index=False)
    return out


def exact_rollout_paths(campaign_family: str, world_id: str, task_id: str) -> dict[str, Path]:
    root = RESULTS / campaign_family / "exact_rollouts" / world_id
    root.mkdir(parents=True, exist_ok=True)
    return {
        "trajectory": root / f"{task_id}_trajectory.csv",
        "flux": root / f"{task_id}_flux.csv",
        "constraints": root / f"{task_id}_constraints.csv",
        "summary": root / f"{task_id}_summary.csv",
    }


def cache_path(campaign_family: str) -> Path:
    return DATA / f"{campaign_family}_exact_cache_index.csv"


def cache_row_dir(campaign_family: str) -> Path:
    path = RESULTS / campaign_family / "exact_cache_rows"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cache_rows(campaign_family: str) -> pd.DataFrame:
    rows = []
    aggregate = cache_path(campaign_family)
    if aggregate.exists():
        try:
            rows.append(pd.read_csv(aggregate))
        except Exception:
            pass
    row_dir = cache_row_dir(campaign_family)
    for path in sorted(row_dir.glob("*.csv")):
        try:
            rows.append(pd.read_csv(path))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    key = "exact_task_id" if "exact_task_id" in out.columns else "candidate_hash"
    return out.drop_duplicates([key], keep="last")


def rollout_summary_rows(campaign_family: str) -> pd.DataFrame:
    manifest_path = DATA / f"{campaign_family}_exact_task_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(manifest_path)
    by_task = manifest.set_index("exact_task_id", drop=False)
    rows = []
    for path in sorted((RESULTS / campaign_family / "exact_rollouts").glob("*/*_summary.csv")):
        task_id = path.name.removesuffix("_summary.csv")
        try:
            summary = pd.read_csv(path)
        except Exception:
            continue
        if summary.empty:
            continue
        base = by_task.loc[task_id].to_dict() if task_id in by_task.index else {"exact_task_id": task_id}
        trajectory_path = path.with_name(path.name.replace("_summary.csv", "_trajectory.csv"))
        flux_path = path.with_name(path.name.replace("_summary.csv", "_flux.csv"))
        derived: dict[str, Any] = {}
        try:
            traj = pd.read_csv(trajectory_path)
            flux = pd.read_csv(flux_path)
            final_product = float(summary.iloc[0].get("final_product", traj["B_total"].iloc[-1] if "B_total" in traj else 0.0))
            integrated = integrated_glucose_proxy(traj, flux)
            derived = {
                "integrated_glucose_proxy": integrated,
                "yield_proxy": final_product / max(integrated, 1e-12),
                "productivity": final_product / max(float(traj["time"].iloc[-1]) if "time" in traj else 1.0, 1e-12),
                "trajectory_path": str(trajectory_path.relative_to(ROOT)),
                "flux_path": str(flux_path.relative_to(ROOT)),
            }
        except Exception:
            pass
        rows.append({**base, **summary.iloc[0].to_dict(), **derived, "summary_path": str(path.relative_to(ROOT)), "cache_hit": False})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def completed_exact_rows(campaign_family: str) -> pd.DataFrame:
    rows = [df for df in [load_cache_rows(campaign_family), rollout_summary_rows(campaign_family)] if not df.empty]
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "exact_task_id" in out.columns:
        out = out.drop_duplicates(["exact_task_id"], keep="last")
    return out


def append_cache(campaign_family: str, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(cache_row_dir(campaign_family) / f"{row['exact_task_id']}.csv", index=False)
    if os.environ.get("PBS_JOBID"):
        return
    path = cache_path(campaign_family)
    existing = load_cache_rows(campaign_family)
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    out = out.drop_duplicates(["exact_task_id"], keep="last")
    out.to_csv(path, index=False)


def manifest_row_by_index(manifest: Path, array_index: int | None, task_id: str | None) -> pd.Series:
    df = pd.read_csv(manifest)
    if task_id is not None:
        sub = df[df["exact_task_id"].astype(str).eq(task_id)]
    elif array_index is not None:
        sub = df[df["array_index"].astype(int).eq(int(array_index))]
    else:
        raise ValueError("Provide --array-index or --task-id")
    if sub.empty:
        raise KeyError(f"No task found in {manifest}")
    return sub.iloc[0]


def integrated_glucose_proxy(traj: pd.DataFrame, flux: pd.DataFrame) -> float:
    if flux.empty or not {"glucose_uptake", "dt", "time"}.issubset(flux.columns):
        return 0.0
    x_by_time = traj.set_index("time")["X"] if "time" in traj.columns else pd.Series(dtype=float)
    values = []
    for _, frow in flux.iterrows():
        x = float(x_by_time.get(float(frow["time"]), traj["X"].iloc[0] if "X" in traj else 1.0))
        values.append(max(0.0, -float(frow["glucose_uptake"])) * x * float(frow["dt"]))
    return float(np.sum(values))


def run_exact_task(row: pd.Series, force: bool = False) -> dict[str, Any]:
    for key, value in BLAS_ENV.items():
        os.environ.setdefault(key, value)
    campaign_family = str(row["campaign_family"])
    task_id = str(row["exact_task_id"])
    paths = exact_rollout_paths(campaign_family, str(row["world_id"]), task_id)
    if paths["summary"].exists() and not force:
        summary = pd.read_csv(paths["summary"]).iloc[0].to_dict()
        cached = load_cache_rows(campaign_family)
        if not cached.empty and "exact_task_id" in cached.columns:
            matched = cached[cached["exact_task_id"].astype(str).eq(task_id)]
            if not matched.empty:
                return {**matched.iloc[-1].to_dict(), "cache_hit": True}
        result = {**row.to_dict(), **summary, "cache_hit": True}
        append_cache(campaign_family, result)
        return result
    candidate = json.loads(str(row["candidate_json"]))
    candidate["candidate_hash"] = str(row["candidate_hash"])
    tic = time.perf_counter()
    traj, flux, cons, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=str(row["world_id"]))
    runtime = time.perf_counter() - tic
    traj.to_csv(paths["trajectory"], index=False)
    flux.to_csv(paths["flux"], index=False)
    cons.to_csv(paths["constraints"], index=False)
    summary.to_csv(paths["summary"], index=False)
    s = summary.iloc[0].to_dict()
    integrated_glucose = integrated_glucose_proxy(traj, flux)
    final_product = float(s.get("final_product", 0.0))
    yield_proxy = final_product / max(integrated_glucose, 1e-12)
    productivity = final_product / max(float(traj["time"].iloc[-1]) if "time" in traj else 1.0, 1e-12)
    result = {
        **row.to_dict(),
        **s,
        "integrated_glucose_proxy": integrated_glucose,
        "yield_proxy": yield_proxy,
        "productivity": productivity,
        "wall_time_seconds": runtime,
        "trajectory_path": str(paths["trajectory"].relative_to(ROOT)),
        "flux_path": str(paths["flux"].relative_to(ROOT)),
        "constraint_path": str(paths["constraints"].relative_to(ROOT)),
        "summary_path": str(paths["summary"].relative_to(ROOT)),
        "cache_hit": False,
    }
    append_cache(campaign_family, result)
    return result


def summarize_difficulty_gate() -> pd.DataFrame:
    cache = completed_exact_rows("harder_dbtl")
    if cache.empty:
        return pd.DataFrame()
    gate = cache[cache["role"].eq("difficulty_acceptance_gate")].copy()
    if gate.empty:
        return pd.DataFrame()
    try:
        library = load_library().set_index("intervention_id")
    except Exception:
        library = pd.DataFrame()
    rows = []
    baseline = gate[gate["world_id"].eq("baseline_world")][["candidate_hash", "final_product"]].rename(columns={"final_product": "baseline_final_product"})
    parent_by_world = (
        gate[gate["edit_count"].fillna(-1).astype(float).eq(0)]
        .sort_values("candidate_id")
        .groupby("world_id", as_index=True)
        .agg(parent_final_product=("final_product", "median"), parent_final_biomass=("final_biomass", "median"))
    )
    manifest_path = DATA / "harder_dbtl_exact_task_manifest.csv"
    expected_by_world: dict[str, int] = {}
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        gate_manifest = manifest[manifest["role"].eq("difficulty_acceptance_gate")]
        expected_by_world = gate_manifest.groupby("world_id")["exact_task_id"].nunique().to_dict()
    for world_id, group in gate.groupby("world_id"):
        expected = int(expected_by_world.get(world_id, len(group)))
        merged = group.merge(baseline, on="candidate_hash", how="inner")
        corr = np.nan
        if len(merged) >= 3 and merged["final_product"].nunique() > 1 and merged["baseline_final_product"].nunique() > 1:
            corr = float(pd.Series(merged["final_product"]).rank().corr(pd.Series(merged["baseline_final_product"]).rank()))
        final_product = group["final_product"].astype(float)
        biomass = group["final_biomass"].astype(float)
        top_cut = float(final_product.quantile(0.90)) if len(final_product) else np.nan
        parent_product = float(parent_by_world.loc[world_id, "parent_final_product"]) if world_id in parent_by_world.index else np.nan
        parent_biomass = float(parent_by_world.loc[world_id, "parent_final_biomass"]) if world_id in parent_by_world.index else np.nan
        active_mechanisms = []
        top_mechanisms = []
        top_group = group[group["final_product"].astype(float).ge(top_cut)] if pd.notna(top_cut) else group.iloc[0:0]
        top_task_ids = set(top_group["exact_task_id"].astype(str)) if "exact_task_id" in top_group.columns else set()
        for _, candidate_row in group.iterrows():
            text = str(candidate_row.get("candidate_json", ""))
            try:
                candidate = json.loads(text)
            except Exception:
                continue
            mechanisms = []
            for edit in candidate.get("edits", []):
                iid = str(edit.get("intervention_id", ""))
                if not library.empty and iid in library.index:
                    mechanisms.append(str(library.loc[iid, "mechanism_class"]))
                elif iid:
                    mechanisms.append(iid)
            active_mechanisms.extend(mechanisms)
            if str(candidate_row.get("exact_task_id", "")) in top_task_ids:
                top_mechanisms.extend(mechanisms)
        hard_like = pd.Series(False, index=group.index)
        if pd.notna(parent_product) and pd.notna(parent_biomass):
            hard_like = (
                group["final_product"].astype(float).ge(parent_product * 1.70)
                & group["final_biomass"].astype(float).ge(parent_biomass * 0.92)
                & ~group.get("severe_growth_collapse", pd.Series(False, index=group.index)).astype(bool)
                & group["edit_count"].fillna(99).astype(float).le(2)
            )
        rows.append(
            {
                "world_id": world_id,
                "evaluated_candidates": int(len(group)),
                "expected_candidates": expected,
                "pending_candidates": max(0, expected - int(len(group))),
                "objective_distribution_mean_final_product": float(final_product.mean()),
                "objective_distribution_sd_final_product": float(final_product.std(ddof=0)),
                "final_product_median": float(final_product.median()),
                "final_product_p90": top_cut,
                "parent_final_product_median": parent_product,
                "top10_fraction": float(final_product.ge(top_cut).mean()) if pd.notna(top_cut) else np.nan,
                "hard_like_attainment_fraction": float(hard_like.mean()) if len(hard_like) else np.nan,
                "biomass_distribution_mean": float(biomass.mean()),
                "biomass_distribution_sd": float(biomass.std(ddof=0)),
                "parent_final_biomass_median": parent_biomass,
                "growth_failure_rate": float(group["severe_growth_collapse"].astype(bool).mean()) if "severe_growth_collapse" in group else np.nan,
                "rank_correlation_with_baseline": corr,
                "fraction_ranking_changed_vs_baseline": float(np.mean(merged["final_product"].rank().to_numpy() != merged["baseline_final_product"].rank().to_numpy())) if len(merged) else np.nan,
                "distinct_mechanism_tokens": int(len(set(active_mechanisms))),
                "distinct_top_candidate_mechanism_tokens": int(len(set(top_mechanisms))),
                "difficulty_gate_status": "complete_landscape_metrics_available_no_method_filtering" if len(group) >= expected else "partial_needs_more_exact_results",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "harder_dbtl_difficulty_gate_summary.csv", index=False)
    return out


def aggregate_cache(campaign_family: str) -> pd.DataFrame:
    out = completed_exact_rows(campaign_family)
    if out.empty:
        return out
    out.to_csv(cache_path(campaign_family), index=False)
    return out


def write_completion_audit(campaign_family: str) -> pd.DataFrame:
    manifest_path = DATA / f"{campaign_family}_exact_task_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    completed = completed_exact_rows(campaign_family)
    completed_ids = set(completed["exact_task_id"].astype(str)) if not completed.empty and "exact_task_id" in completed.columns else set()
    rows = []
    for role, group in manifest.groupby("role" if "role" in manifest.columns else "task_family"):
        ids = set(group["exact_task_id"].astype(str))
        done = ids & completed_ids
        rows.append(
            {
                "campaign_family": campaign_family,
                "role": role,
                "manifest_tasks": int(len(ids)),
                "completed_tasks": int(len(done)),
                "pending_tasks": int(len(ids - done)),
                "completion_fraction": float(len(done) / max(len(ids), 1)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"{campaign_family}_completion_audit.csv", index=False)
    return out


def parent_reference_by_world(completed: pd.DataFrame) -> pd.DataFrame:
    gate = completed[completed.get("role", "").eq("difficulty_acceptance_gate")].copy()
    parents = gate[gate["edit_count"].fillna(-1).astype(float).eq(0)]
    rows = []
    for world_id, group in parents.groupby("world_id"):
        row = {
            "world_id": world_id,
            "parent_final_product": float(group["final_product"].astype(float).median()),
            "parent_product_AUC": float(group["product_AUC"].astype(float).median()),
            "parent_final_biomass": float(group["final_biomass"].astype(float).median()),
            "parent_yield_proxy": float(group["yield_proxy"].astype(float).median()) if "yield_proxy" in group else np.nan,
        }
        objective = per_world_objective_config(row)
        row["parent_utility"] = freeze.final_utility(
            {
                "final_product": row["parent_final_product"],
                "product_AUC": row["parent_product_AUC"],
                "yield_proxy": row["parent_yield_proxy"],
                "final_biomass": row["parent_final_biomass"],
                "growth_failure": False,
                "edit_count": 0,
                "construction_risk": "low",
                "infeasible": False,
            },
            objective,
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "harder_dbtl_parent_references.csv", index=False)
    return out


def per_world_objective_config(parent_row: dict[str, Any]) -> dict[str, Any]:
    base_path = DATA / "final_dbtl_objective_config.json"
    if base_path.exists():
        objective = json.loads(base_path.read_text(encoding="utf-8"))
    else:
        objective = {
            "weights": {"titer": 0.45, "productivity": 0.20, "yield": 0.20, "biomass": 0.15},
            "clipping": {"minimum_component": 0.0, "maximum_component": 2.5},
            "failure_penalties": {"growth_failure": 1.0, "infeasible": 2.0},
            "edit_count_penalty": 0.025,
            "construction_risk_penalties": {"low": 0.0, "medium": 0.04, "high": 0.10, "reaction_level_only": 0.06},
        }
    objective = dict(objective)
    def ref(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 1e-12
        return 1e-12 if pd.isna(numeric) else max(numeric, 1e-12)

    objective["reference_values"] = {
        "final_titer": ref(parent_row["parent_final_product"]),
        "productivity": ref(parent_row["parent_product_AUC"]),
        "yield": ref(parent_row["parent_yield_proxy"]),
        "final_biomass": ref(parent_row["parent_final_biomass"]),
    }
    return objective


def target_rules() -> dict[str, Any]:
    path = DATA / "harder_shift_target_rules.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else write_target_rules()


def annotate_harder_observations(completed: pd.DataFrame) -> pd.DataFrame:
    seq = completed[completed.get("role", "").eq("sequential_dbtl")].copy()
    if seq.empty:
        return pd.DataFrame()
    library = load_library()
    parents = parent_reference_by_world(completed).set_index("world_id")
    rules = target_rules()["tiers"]
    annotated = []
    for _, row in seq.iterrows():
        world_id = str(row["world_id"])
        if world_id not in parents.index:
            continue
        parent = parents.loc[world_id].to_dict()
        objective = per_world_objective_config(parent)
        candidate = json.loads(str(row["candidate_json"]))
        meta = final_run.candidate_metadata(candidate, library)
        metrics = {
            "final_product": float(row.get("final_product", 0.0)),
            "product_AUC": float(row.get("product_AUC", 0.0)),
            "yield_proxy": float(row.get("yield_proxy", 0.0)),
            "final_biomass": float(row.get("final_biomass", 0.0)),
            "growth_failure": bool(row.get("severe_growth_collapse", False)),
            "edit_count": int(row.get("edit_count", meta["edit_count"])),
            "construction_risk": meta["construction_risk"],
            "infeasible": not bool(row.get("feasible", True)),
        }
        utility = freeze.final_utility(metrics, objective)
        out = {**row.to_dict(), **meta, "utility": utility}
        for tier, rule in rules.items():
            target = (
                metrics["final_product"] >= float(parent["parent_final_product"]) * float(rule["minimum_final_titer_fold_parent"])
                and metrics["final_biomass"] >= float(parent["parent_final_biomass"]) * float(rule["minimum_final_biomass_fold_parent"])
                and metrics["yield_proxy"] >= float(parent["parent_yield_proxy"]) * float(rule["minimum_yield_fold_parent"])
                and metrics["edit_count"] <= int(rule["maximum_interventions"])
                and utility >= float(parent["parent_utility"]) + float(rule["minimum_utility_delta_parent"])
                and not metrics["growth_failure"]
            )
            out[f"reaches_{tier}_target"] = bool(target)
        annotated.append(out)
    obs = pd.DataFrame(annotated)
    if not obs.empty:
        obs.to_csv(DATA / "harder_dbtl_observations.csv", index=False)
    return obs


def dbtl_best_auc(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in best.groupby(["world_id", "campaign_id", "method_id"]):
        g = group.sort_values("planned_culture_index")
        rows.append(
            {
                "world_id": keys[0],
                "campaign_id": keys[1],
                "method_id": keys[2],
                "dbtl_auc": float(np.trapezoid(g["best_utility_so_far"].to_numpy(float), g["planned_culture_index"].to_numpy(float))) if len(g) > 1 else float(g["best_utility_so_far"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_harder_figures(world_method: pd.DataFrame, best: pd.DataFrame) -> list[str]:
    if world_method.empty:
        return []
    ensure_dirs()
    mpl_cache = RESULTS / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[str] = []
    methods = [m for m in METHOD_IDS if m in set(world_method["method_id"])]
    labels = final_run.METHOD_LABELS

    def save(name: str) -> None:
        for ext in ["svg", "png"]:
            path = FIGURES / f"{name}.{ext}"
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            figures.append(str(path.relative_to(ROOT)))
        plt.close()

    plot_specs = [
        ("harder_dbtl_success_rate_by_world", "hard_success_rate", "Hard-target success probability"),
        ("harder_dbtl_best_utility_by_world", "mean_best_utility", "Mean best utility"),
        ("harder_dbtl_dbtl_auc_by_world", "mean_dbtl_auc", "Mean DBTL AUC"),
        ("harder_dbtl_regret_by_world", "mean_regret_vs_world_campaign_best", "Mean regret"),
    ]
    for name, value_col, ylabel in plot_specs:
        pivot = world_method.pivot(index="world_id", columns="method_id", values=value_col).reindex(columns=methods)
        plt.figure(figsize=(11, 5.5))
        x = np.arange(len(pivot.index))
        width = 0.78 / max(len(methods), 1)
        for i, method in enumerate(methods):
            plt.bar(x + (i - (len(methods) - 1) / 2) * width, pivot[method].to_numpy(float), width=width, label=labels.get(method, method))
        plt.xticks(x, pivot.index, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7, ncols=2)
        save(name)

    if not best.empty:
        plt.figure(figsize=(9, 5))
        for i, method in enumerate(methods):
            sub = best[best["method_id"].eq(method)]
            if sub.empty:
                continue
            curve = sub.groupby("planned_culture_index")["best_utility_so_far"].mean()
            plt.plot(curve.index, curve.values, marker="o", label=labels.get(method, method))
        plt.xlabel("Exact cultures")
        plt.ylabel("Mean best utility so far")
        plt.legend(fontsize=7, ncols=2)
        save("harder_dbtl_best_utility_vs_cultures")
    return sorted(set(figures))


def write_harder_report(summary: pd.DataFrame, world_method: pd.DataFrame, figures: list[str], analysis: dict[str, Any]) -> None:
    lines = [
        "# Harder-World Sequential DBTL Validation Report",
        "",
        "## Status",
        "",
        f"- Analysis status: `{analysis['status']}`",
        f"- Sequential observations analysed: `{analysis.get('observations', 0)}`",
        f"- Complete method-campaigns: `{analysis.get('complete_method_campaigns', 0)}` of `{analysis.get('method_campaigns', 0)}`",
        "",
        "This report uses the frozen harder-world manifests, target rules, methods, and exact Yeast9 dynamic-pFBA rollouts. No benchmark parameters are changed during analysis.",
        "",
        "## World-Method Summary",
        "",
        markdown_table(
            world_method[
                [
                    "world_id",
                    "method_id",
                    "campaigns",
                    "hard_success_rate",
                    "median_cultures_to_hard_target_censored",
                    "mean_best_utility",
                    "mean_best_final_product",
                    "mean_best_productivity",
                    "mean_best_yield_proxy",
                    "mean_best_final_biomass",
                    "mean_dbtl_auc",
                    "mean_regret_vs_world_campaign_best",
                    "std_best_utility_across_seeds",
                ]
            ].sort_values(["world_id", "method_id"])
        ),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{fig}`" for fig in figures)
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "Use right-censored cultures-to-target when a method does not hit the hard target within the frozen `8+4+4` budget. Regret is computed against the best observed method within the same world and campaign, so it is a benchmark-relative endpoint rather than a new objective.",
            "",
        ]
    )
    HARDER_REPORT.write_text("\n".join(lines), encoding="utf-8")


def analyze_harder_dbtl() -> dict[str, Any]:
    completed = aggregate_cache("harder_dbtl")
    obs = annotate_harder_observations(completed)
    if obs.empty:
        write_completion_audit("harder_dbtl")
        return {"status": "no_sequential_observations_available", "observations": 0}
    obs = obs.sort_values(["world_id", "campaign_id", "method_id", "planned_culture_index"])
    best_rows = []
    round_rows = []
    summary_rows = []
    for (world_id, campaign_id, method_id), group in obs.groupby(["world_id", "campaign_id", "method_id"]):
        best = -np.inf
        for _, row in group.iterrows():
            best = max(best, float(row["utility"]))
            best_rows.append(
                {
                    "world_id": world_id,
                    "campaign_id": campaign_id,
                    "method_id": method_id,
                    "planned_culture_index": int(row["planned_culture_index"]),
                    "planned_round_index": int(row["planned_round_index"]),
                    "best_utility_so_far": best,
                    "hard_target_reached_so_far": bool(group[group["planned_culture_index"].le(row["planned_culture_index"])]["reaches_hard_target"].astype(bool).any()),
                }
            )
        for round_idx, rg in group.groupby("planned_round_index"):
            prefix = group[group["planned_culture_index"].le(rg["planned_culture_index"].max())]
            round_rows.append(
                {
                    "world_id": world_id,
                    "campaign_id": campaign_id,
                    "method_id": method_id,
                    "round_index": int(round_idx),
                    "cultures_cumulative": int(rg["planned_culture_index"].max()),
                    "best_utility_cumulative": float(prefix["utility"].max()),
                    "hard_target_reached_by_round": bool(prefix["reaches_hard_target"].astype(bool).any()),
                }
            )
        hard_success = group[group["reaches_hard_target"].astype(bool)]
        moderate_success = group[group["reaches_moderate_target"].astype(bool)]
        easy_success = group[group["reaches_easy_target"].astype(bool)]
        best_row = group.loc[group["utility"].astype(float).idxmax()]
        summary_rows.append(
            {
                "world_id": world_id,
                "campaign_id": campaign_id,
                "method_id": method_id,
                "observed_cultures": int(len(group)),
                "hard_success": bool(len(hard_success)),
                "right_censored_hard_target": not bool(len(hard_success)),
                "cultures_to_hard_target": int(hard_success["planned_culture_index"].iloc[0]) if len(hard_success) else np.nan,
                "rounds_to_hard_target": int(hard_success["planned_round_index"].iloc[0]) if len(hard_success) else np.nan,
                "moderate_success": bool(len(moderate_success)),
                "cultures_to_moderate_target": int(moderate_success["planned_culture_index"].iloc[0]) if len(moderate_success) else np.nan,
                "easy_success": bool(len(easy_success)),
                "cultures_to_easy_target": int(easy_success["planned_culture_index"].iloc[0]) if len(easy_success) else np.nan,
                "best_utility": float(best_row["utility"]),
                "best_final_product": float(best_row["final_product"]),
                "best_productivity": float(best_row.get("productivity", np.nan)),
                "best_final_biomass": float(best_row["final_biomass"]),
                "best_yield_proxy": float(best_row["yield_proxy"]),
                "growth_failure_rate": float(group.get("severe_growth_collapse", pd.Series(False, index=group.index)).astype(bool).mean()),
                "exact_lp_solves": int(group.get("n_actual_lp_solves", pd.Series(0, index=group.index)).fillna(0).sum()),
                "unique_effective_models": int(group.get("candidate_hash", pd.Series(dtype=str)).nunique()),
            }
        )
    best = pd.DataFrame(best_rows)
    best.to_csv(DATA / "harder_dbtl_best_so_far.csv", index=False)
    pd.DataFrame(round_rows).to_csv(DATA / "harder_dbtl_round_summary.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    auc = dbtl_best_auc(best)
    summary = summary.merge(auc, on=["world_id", "campaign_id", "method_id"], how="left")
    campaign_best = summary.groupby(["world_id", "campaign_id"])["best_utility"].max().rename("world_campaign_best_utility")
    summary = summary.merge(campaign_best.reset_index(), on=["world_id", "campaign_id"], how="left")
    summary["regret_vs_world_campaign_best"] = summary["world_campaign_best_utility"] - summary["best_utility"]
    summary["cultures_to_hard_target_censored"] = summary["cultures_to_hard_target"].fillna(sum(ROUND_BATCHES) + 1)
    summary["rounds_to_hard_target_censored"] = summary["rounds_to_hard_target"].fillna(len(ROUND_BATCHES) + 1)
    summary.to_csv(DATA / "harder_dbtl_method_summary.csv", index=False)
    world_method = (
        summary.groupby(["world_id", "method_id"], as_index=False)
        .agg(
            campaigns=("campaign_id", "nunique"),
            mean_observed_cultures=("observed_cultures", "mean"),
            hard_success_rate=("hard_success", "mean"),
            median_cultures_to_hard_target_censored=("cultures_to_hard_target", lambda s: float(np.median(pd.Series(s).fillna(17.0)))),
            median_rounds_to_hard_target_censored=("rounds_to_hard_target", lambda s: float(np.median(pd.Series(s).fillna(4.0)))),
            mean_best_utility=("best_utility", "mean"),
            mean_best_final_product=("best_final_product", "mean"),
            mean_best_productivity=("best_productivity", "mean"),
            mean_best_yield_proxy=("best_yield_proxy", "mean"),
            mean_best_final_biomass=("best_final_biomass", "mean"),
            mean_dbtl_auc=("dbtl_auc", "mean"),
            mean_regret_vs_world_campaign_best=("regret_vs_world_campaign_best", "mean"),
            std_best_utility_across_seeds=("best_utility", "std"),
            std_cultures_to_hard_target_censored=("cultures_to_hard_target_censored", "std"),
            growth_failure_rate=("growth_failure_rate", "mean"),
        )
    )
    world_method.to_csv(DATA / "harder_dbtl_world_method_summary.csv", index=False)
    audit = write_completion_audit("harder_dbtl")
    figures = write_harder_figures(world_method, best)
    analysis = {
        "status": "harder_dbtl_analysis_updated",
        "observations": int(len(obs)),
        "method_campaigns": int(len(summary)),
        "complete_method_campaigns": int((summary["observed_cultures"] >= 16).sum()),
        "audit": audit.to_dict("records"),
        "figures": figures,
    }
    write_harder_report(summary, world_method, figures, analysis)
    return analysis


def annotate_distribution_observations() -> pd.DataFrame:
    completed = aggregate_cache("distribution_shift")
    if completed.empty:
        write_completion_audit("distribution_shift")
        return pd.DataFrame()
    harder_completed = completed_exact_rows("harder_dbtl")
    if harder_completed.empty:
        return pd.DataFrame()
    parents = parent_reference_by_world(harder_completed).set_index("world_id")
    rules = target_rules()["tiers"]
    library = load_library()
    rows = []
    for _, row in completed.iterrows():
        target_world = str(row.get("target_world_id", row.get("world_id", "")))
        if target_world not in parents.index:
            continue
        parent = parents.loc[target_world].to_dict()
        objective = per_world_objective_config(parent)
        candidate = json.loads(str(row["candidate_json"]))
        meta = final_run.candidate_metadata(candidate, library)
        metrics = {
            "final_product": float(row.get("final_product", 0.0)),
            "product_AUC": float(row.get("product_AUC", 0.0)),
            "yield_proxy": float(row.get("yield_proxy", 0.0)),
            "final_biomass": float(row.get("final_biomass", 0.0)),
            "growth_failure": bool(row.get("severe_growth_collapse", False)),
            "edit_count": int(row.get("edit_count", meta["edit_count"])),
            "construction_risk": meta["construction_risk"],
            "infeasible": not bool(row.get("feasible", True)),
        }
        utility = freeze.final_utility(metrics, objective)
        out = {**row.to_dict(), **meta, "target_world_id": target_world, "utility": utility}
        for tier, rule in rules.items():
            out[f"reaches_{tier}_target"] = bool(
                metrics["final_product"] >= float(parent["parent_final_product"]) * float(rule["minimum_final_titer_fold_parent"])
                and metrics["final_biomass"] >= float(parent["parent_final_biomass"]) * float(rule["minimum_final_biomass_fold_parent"])
                and metrics["yield_proxy"] >= float(parent["parent_yield_proxy"]) * float(rule["minimum_yield_fold_parent"])
                and metrics["edit_count"] <= int(rule["maximum_interventions"])
                and utility >= float(parent["parent_utility"]) + float(rule["minimum_utility_delta_parent"])
                and not metrics["growth_failure"]
            )
        rows.append(out)
    obs = pd.DataFrame(rows)
    if not obs.empty:
        obs.to_csv(DATA / "distribution_shift_observations.csv", index=False)
    return obs


def distribution_source_reference(obs: pd.DataFrame) -> pd.DataFrame:
    source = obs[obs["target_world_id"].eq("baseline_world")].copy()
    cols = ["campaign_id", "method_id", "candidate_hash", "utility", "final_product", "planned_culture_index"]
    if source.empty:
        return pd.DataFrame(columns=[*cols[:-1], "source_utility", "source_final_product", "source_planned_culture_index"])
    return source[cols].rename(
        columns={
            "utility": "source_utility",
            "final_product": "source_final_product",
            "planned_culture_index": "source_planned_culture_index",
        }
    )


def summarize_distribution_decisions(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions_path = DATA / "distribution_shift_decision_manifest.csv"
    if not decisions_path.exists() or obs.empty:
        return pd.DataFrame(), pd.DataFrame()
    decisions = pd.read_csv(decisions_path)
    merged = decisions.merge(
        obs.drop(columns=[c for c in decisions.columns if c in obs.columns and c != "exact_task_id"], errors="ignore"),
        on="exact_task_id",
        how="left",
    )
    source = distribution_source_reference(obs)
    merged = merged.merge(source, on=["campaign_id", "method_id", "candidate_hash"], how="left")
    merged["utility_delta_vs_source"] = merged["utility"] - merged["source_utility"]
    merged["product_delta_vs_source"] = merged["final_product"] - merged["source_final_product"]
    rows = []
    for keys, group in merged.groupby(["pair_id", "source_world_id", "target_world_id", "shift_tier", "campaign_id", "method_id", "adaptation_budget"]):
        pair_id, source_world, target_world, shift_tier, campaign_id, method_id, budget = keys
        g = group.sort_values("planned_culture_index")
        completed = g[g["utility"].notna()].copy()
        budget = int(budget)
        prefix = completed[completed["planned_culture_index"].astype(float).le(float(budget))]
        hard_success = completed[completed["reaches_hard_target"].fillna(False).astype(bool)]
        prefix_hard = prefix[prefix["reaches_hard_target"].fillna(False).astype(bool)]
        auc = float(np.trapezoid(completed["utility"].cummax().to_numpy(float), completed["planned_culture_index"].to_numpy(float))) if len(completed) > 1 else (float(completed["utility"].iloc[0]) if len(completed) else np.nan)
        prefix_auc = float(np.trapezoid(prefix["utility"].cummax().to_numpy(float), prefix["planned_culture_index"].to_numpy(float))) if len(prefix) > 1 else (float(prefix["utility"].iloc[0]) if len(prefix) else np.nan)
        ranking = safe_rank_corr(completed["source_utility"], completed["utility"]) if "source_utility" in completed else np.nan
        planned_ranking = safe_rank_corr(-completed["planned_culture_index"].astype(float), completed["utility"]) if len(completed) else np.nan
        rows.append(
            {
                "pair_id": pair_id,
                "source_world_id": source_world,
                "target_world_id": target_world,
                "shift_tier": shift_tier,
                "campaign_id": campaign_id,
                "method_id": method_id,
                "adaptation_budget": budget,
                "completed_exact_candidates": int(len(completed)),
                "target_attainment": bool(len(hard_success)),
                "cultures_to_hard_target": int(hard_success["planned_culture_index"].iloc[0]) if len(hard_success) else np.nan,
                "right_censored_hard_target": not bool(len(hard_success)),
                "adaptation_target_attainment": bool(len(prefix_hard)),
                "best_utility_after_budget": float(prefix["utility"].max()) if len(prefix) else np.nan,
                "best_product_after_budget": float(prefix["final_product"].max()) if len(prefix) else np.nan,
                "adaptation_efficiency": float((prefix["utility"].max() - completed["utility"].iloc[0]) / max(budget, 1)) if len(prefix) and len(completed) else np.nan,
                "full_sequence_best_utility": float(completed["utility"].max()) if len(completed) else np.nan,
                "full_sequence_best_product": float(completed["final_product"].max()) if len(completed) else np.nan,
                "dbtl_auc": auc,
                "adaptation_budget_auc": prefix_auc,
                "prediction_quality_mae_utility_delta": float(completed["utility_delta_vs_source"].abs().mean()) if "utility_delta_vs_source" in completed else np.nan,
                "ranking_quality_source_target_spearman": ranking,
                "planned_order_target_spearman": planned_ranking,
                "transfer_degradation_mean_utility_delta": float(completed["utility_delta_vs_source"].mean()) if "utility_delta_vs_source" in completed else np.nan,
                "uncertainty_utility_delta_sd": float(completed["utility_delta_vs_source"].std(ddof=0)) if "utility_delta_vs_source" in completed else np.nan,
                "robustness_utility_delta_p10": float(completed["utility_delta_vs_source"].quantile(0.10)) if "utility_delta_vs_source" in completed and len(completed) else np.nan,
                "growth_failure_rate": float(completed.get("severe_growth_collapse", pd.Series(False, index=completed.index)).fillna(False).astype(bool).mean()) if len(completed) else np.nan,
            }
        )
    decision = pd.DataFrame(rows)
    if decision.empty:
        return decision, pd.DataFrame()
    decision["cultures_to_hard_target_censored"] = decision["cultures_to_hard_target"].fillna(sum(ROUND_BATCHES) + 1)
    decision.to_csv(DATA / "distribution_shift_decision_summary.csv", index=False)
    pair_method = (
        decision.groupby(["pair_id", "source_world_id", "target_world_id", "shift_tier", "method_id", "adaptation_budget"], as_index=False)
        .agg(
            campaigns=("campaign_id", "nunique"),
            success_probability=("target_attainment", "mean"),
            adaptation_success_probability=("adaptation_target_attainment", "mean"),
            median_cultures_to_hard_target_censored=("cultures_to_hard_target_censored", "median"),
            mean_best_utility_after_budget=("best_utility_after_budget", "mean"),
            mean_full_sequence_best_utility=("full_sequence_best_utility", "mean"),
            mean_dbtl_auc=("dbtl_auc", "mean"),
            mean_adaptation_budget_auc=("adaptation_budget_auc", "mean"),
            prediction_quality_mae_utility_delta=("prediction_quality_mae_utility_delta", "mean"),
            ranking_quality_source_target_spearman=("ranking_quality_source_target_spearman", "mean"),
            planned_order_target_spearman=("planned_order_target_spearman", "mean"),
            transfer_degradation_mean_utility_delta=("transfer_degradation_mean_utility_delta", "mean"),
            uncertainty_utility_delta_sd=("uncertainty_utility_delta_sd", "mean"),
            robustness_utility_delta_p10=("robustness_utility_delta_p10", "mean"),
            seed_variability_best_utility=("full_sequence_best_utility", "std"),
        )
    )
    pair_method.to_csv(DATA / "distribution_shift_pair_method_summary.csv", index=False)
    return decision, pair_method


def write_distribution_figures(pair_method: pd.DataFrame) -> list[str]:
    if pair_method.empty:
        return []
    ensure_dirs()
    mpl_cache = RESULTS / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[str] = []
    methods = [m for m in METHOD_IDS if m in set(pair_method["method_id"])]
    labels = final_run.METHOD_LABELS

    def save(name: str) -> None:
        for ext in ["svg", "png"]:
            path = FIGURES / f"{name}.{ext}"
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            figures.append(str(path.relative_to(ROOT)))
        plt.close()

    for name, value_col, ylabel in [
        ("distribution_shift_transfer_degradation", "transfer_degradation_mean_utility_delta", "Mean utility delta vs baseline source"),
        ("distribution_shift_ranking_quality", "ranking_quality_source_target_spearman", "Source-target rank correlation"),
        ("distribution_shift_adaptation_efficiency", "mean_best_utility_after_budget", "Mean best utility after target-culture budget"),
        ("distribution_shift_target_attainment", "adaptation_success_probability", "Target attainment within budget"),
    ]:
        plt.figure(figsize=(10, 5.5))
        sub = pair_method[pair_method["adaptation_budget"].isin([0, 2, 4, 8])]
        x_labels = [f"{row.target_world_id}\n{int(row.adaptation_budget)}" for row in sub[["target_world_id", "adaptation_budget"]].drop_duplicates().itertuples(index=False)]
        x_key = sub["target_world_id"].astype(str) + "|" + sub["adaptation_budget"].astype(str)
        keys = list(dict.fromkeys(x_key))
        x = np.arange(len(keys))
        width = 0.78 / max(len(methods), 1)
        for i, method in enumerate(methods):
            vals = []
            for key in keys:
                target, budget = key.split("|")
                m = sub[sub["method_id"].eq(method) & sub["target_world_id"].astype(str).eq(target) & sub["adaptation_budget"].astype(str).eq(budget)]
                vals.append(float(m[value_col].mean()) if not m.empty else np.nan)
            plt.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=labels.get(method, method))
        plt.xticks(x, x_labels, rotation=35, ha="right")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7, ncols=2)
        save(name)
    return sorted(set(figures))


def write_distribution_report(pair_method: pd.DataFrame, figures: list[str], analysis: dict[str, Any]) -> None:
    lines = [
        "# Distribution-Shift Validation Report",
        "",
        "## Status",
        "",
        f"- Analysis status: `{analysis['status']}`",
        f"- Exact observations analysed: `{analysis.get('observations', 0)}`",
        f"- Decision rows analysed: `{analysis.get('decision_rows', 0)}`",
        "",
        "The frozen transfer analysis uses the predefined baseline-source to target-world pairs and adaptation budgets `0`, `2`, `4`, and `8`. Source-target utility deltas and rank correlations are retrospective validation metrics from exact target-world rollouts, not new training signals.",
        "",
        "## Pair-Method Summary",
        "",
        markdown_table(
            pair_method[
                [
                    "target_world_id",
                    "method_id",
                    "adaptation_budget",
                    "success_probability",
                    "adaptation_success_probability",
                    "median_cultures_to_hard_target_censored",
                    "mean_best_utility_after_budget",
                    "mean_dbtl_auc",
                    "prediction_quality_mae_utility_delta",
                    "ranking_quality_source_target_spearman",
                    "transfer_degradation_mean_utility_delta",
                    "uncertainty_utility_delta_sd",
                ]
            ].sort_values(["target_world_id", "adaptation_budget", "method_id"]),
            max_rows=196,
        ),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{fig}`" for fig in figures)
    lines.append("")
    SHIFT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def analyze_distribution_shift() -> dict[str, Any]:
    obs = annotate_distribution_observations()
    if obs.empty:
        audit = write_completion_audit("distribution_shift")
        return {"status": "no_distribution_shift_observations_available", "observations": 0, "audit": audit.to_dict("records")}
    decision, pair_method = summarize_distribution_decisions(obs)
    audit = write_completion_audit("distribution_shift")
    figures = write_distribution_figures(pair_method)
    analysis = {
        "status": "distribution_shift_analysis_updated",
        "observations": int(len(obs)),
        "decision_rows": int(len(decision)),
        "pair_method_rows": int(len(pair_method)),
        "audit": audit.to_dict("records"),
        "figures": figures,
    }
    write_distribution_report(pair_method, figures, analysis)
    return analysis


def write_pbs(bundle: Path) -> None:
    hpc = bundle / "hpc"
    hpc.mkdir(parents=True, exist_ok=True)
    for family, manifest, job_name in [
        ("harder_dbtl", "data/harder_dbtl_exact_task_manifest.csv", "harder_dbtl"),
        ("distribution_shift", "data/distribution_shift_exact_task_manifest.csv", "shift_dbtl"),
    ]:
        array_tasks = max(1, len(pd.read_csv(bundle / manifest)))
        (hpc / f"run_{family}_array.pbs").write_text(
            f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=04:00:00
#PBS -J 1-{array_tasks}
#PBS -j oe

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd "$PBS_O_WORKDIR"

export YEAST_GEM_PATH="$PWD/models/yeast-GEM.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-hpc.txt
fi

.venv/bin/python scripts/run_harder_shift_validation.py run-exact --manifest {manifest} --array-index "$PBS_ARRAY_INDEX"
""",
            encoding="utf-8",
        )
        (hpc / f"run_{family}_smoke.pbs").write_text(
            f"""#!/bin/bash
#PBS -N {job_name}_smoke
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd "$PBS_O_WORKDIR"

export YEAST_GEM_PATH="$PWD/models/yeast-GEM.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-hpc.txt
fi

.venv/bin/python scripts/run_harder_shift_validation.py run-exact --manifest {manifest} --array-index 1
""",
            encoding="utf-8",
        )


def prepare_hpc_bundle(out_dir: Path = HPC_ROOT, overwrite: bool = False) -> Path:
    write_candidate_manifests()
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["scripts", "data", "models", "results/harder_dbtl/exact_rollouts", "results/distribution_shift/exact_rollouts", "hpc"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    scripts = [
        "run_harder_shift_validation.py",
        "run_deployment_verifier.py",
        "run_final_dbtl_benchmark.py",
        "final_dbtl_benchmark_freeze.py",
        "deployment_benchmark.py",
        "design_benchmark_exact.py",
        "design_benchmark.py",
        "gem_backend.py",
        "run_gem_dynamic_capacity.py",
        "stage4_design.py",
        "run_reporter_grounded_hybrid_distillation.py",
        "run_gem_state_space_validation.py",
        "run_fixed_environment_validation.py",
    ]
    for script in scripts:
        shutil.copy2(ROOT / "src" / "yeast_validation" / script, out_dir / "scripts" / script)
    for pattern in [
        "harder_shift_*",
        "harder_dbtl_exact_task_manifest.csv",
        "distribution_shift_*",
        "final_dbtl_*",
        "deployment_benchmark_operational_scenarios.csv",
        "deployment_benchmark_editable_reaction_universe.csv",
        "gem_capacity_selected_parameters.csv",
    ]:
        for src in sorted(DATA.glob(pattern)):
            if src.is_file():
                shutil.copy2(src, out_dir / "data" / src.name)
    yeast_gem = next(
        path
        for path in [
            ROOT / "models" / "yeast-GEM.xml",
            ROOT / "results" / "deployment_benchmark" / "hybrid_round_001_blind" / "inputs" / "yeast-GEM.xml",
            Path("/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"),
        ]
        if path.exists()
    )
    shutil.copy2(yeast_gem, out_dir / "models" / "yeast-GEM.xml")
    (out_dir / "requirements-hpc.txt").write_text(
        "\n".join(["pandas", "numpy", "matplotlib", "cobra==0.31.1", "swiglpk==5.0.13", "scipy", "scikit-learn", "sympy"]) + "\n",
        encoding="utf-8",
    )
    write_pbs(out_dir)
    (out_dir / "README_HPC.md").write_text(
        """# Harder DBTL and Distribution-Shift Vanda Bundle

Run from the bundle root on Vanda:

```bash
qsub hpc/run_harder_dbtl_smoke.pbs
qsub hpc/run_distribution_shift_smoke.pbs
qsub hpc/run_harder_dbtl_array.pbs
qsub hpc/run_distribution_shift_array.pbs
```

Each PBS array element evaluates exactly one manifest row in a fresh Python
process, with BLAS/OpenMP threads capped at one. Completed summaries are cached
by `exact_task_id`, which includes the candidate hash, hidden-world hash, and
simulator version.
""",
        encoding="utf-8",
    )
    archive = out_dir.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--n-calibration", type=int, default=64)
    run = sub.add_parser("run-exact")
    run.add_argument("--manifest", required=True)
    run.add_argument("--array-index", type=int, default=None)
    run.add_argument("--task-id", default=None)
    run.add_argument("--force", action="store_true")
    sub.add_parser("summarize-gate")
    sub.add_parser("analyze-harder")
    sub.add_parser("analyze-shift")
    aggregate = sub.add_parser("aggregate-cache")
    aggregate.add_argument("--campaign-family", choices=["harder_dbtl", "distribution_shift"], default="harder_dbtl")
    bundle = sub.add_parser("prepare-hpc-bundle")
    bundle.add_argument("--out-dir", default=str(HPC_ROOT))
    bundle.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        harder, shift = write_candidate_manifests(n_calibration=args.n_calibration)
        print(json.dumps({"harder_tasks": len(harder), "shift_tasks": len(shift)}, indent=2))
    elif args.command == "run-exact":
        row = manifest_row_by_index(Path(args.manifest), args.array_index, args.task_id)
        result = run_exact_task(row, force=args.force)
        print(json.dumps({"exact_task_id": result["exact_task_id"], "exact_status": result["exact_status"], "cache_hit": result["cache_hit"]}, indent=2, sort_keys=True))
    elif args.command == "summarize-gate":
        out = summarize_difficulty_gate()
        print(out.to_string(index=False) if not out.empty else "No harder DBTL exact cache found yet.")
    elif args.command == "analyze-harder":
        print(json.dumps(analyze_harder_dbtl(), indent=2, sort_keys=True))
    elif args.command == "analyze-shift":
        print(json.dumps(analyze_distribution_shift(), indent=2, sort_keys=True))
    elif args.command == "aggregate-cache":
        out = aggregate_cache(args.campaign_family)
        audit = write_completion_audit(args.campaign_family)
        print(json.dumps({"campaign_family": args.campaign_family, "completed_rows": len(out), "audit": audit.to_dict("records")}, indent=2, sort_keys=True))
    elif args.command == "prepare-hpc-bundle":
        archive = prepare_hpc_bundle(Path(args.out_dir), overwrite=args.overwrite)
        print(archive)


if __name__ == "__main__":
    main()
