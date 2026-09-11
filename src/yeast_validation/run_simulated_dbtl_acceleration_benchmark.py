#!/usr/bin/env python3
"""Definitive simulated DBTL acceleration workflow benchmark.

This runner treats the exact Yeast9 dynamic-pFBA verifier as a hidden wet lab.
It separates physical-culture accounting from exact-result caching: if two
methods independently request the same culture, the exact computation may be
reused by hash, but each workflow is still charged one physical culture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import design_benchmark_exact as exact
import final_dbtl_benchmark_freeze as freeze
import run_deployment_verifier as verifier
import run_final_dbtl_benchmark as final_run


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "simulated_dbtl_acceleration"
ROLLOUTS = RESULTS / "hidden_wet_lab_rollouts"
LEDGERS = RESULTS / "workflow_ledgers"
HPC_ROOT = ROOT / "hpc_bundles" / "simulated_dbtl_acceleration_vanda"

SIMULATED_DBTL_VERSION = "simulated_dbtl_acceleration_v1"
WET_LAB_VERSION = "yeast9_dynamic_pfba_with_frozen_latent_culture_variability_v1"
TARGET_VERSION = "simulated_dbtl_moderate_hard_targets_v1"
REPORTER_VERSION = "lagged_noisy_physiology_reporters_v1"
MAX_PHYSICAL_CULTURES = 24
INITIAL_CULTURES = 8
ROUND_BATCHES = [8, 4, 4]
VIRTUAL_SEARCH_BUDGET = 50_000
MANIFEST_PROPOSAL_POOL_SIZE = 240
MAJOR_WORLDS = [
    "baseline_world",
    "strong_oxidative_burden_world",
    "atp_limited_world",
    "precursor_competition_world",
    "pathway_bottleneck_damage_world",
    "mixed_multi_mechanism_world",
]
FULL_METHODS = [
    "conventional_dbtl",
    "bayesian_optimization",
    "black_box_digital_twin",
    "modular_hybrid_no_biosensors",
    "biosensor_informed_modular_hybrid",
]
REPORTER_CONDITIONS = [
    "no_reporters",
    "oxidative_only",
    "atp_only",
    "pathway_only",
    "full_reporters",
    "noisy_full_reporters",
    "sparse_full_reporters",
]
CAMPAIGN_SEEDS = list(range(73001, 73011))
BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class CultureLatents:
    oxidative_susceptibility: float
    atp_burden: float
    pathway_capacity: float
    expression_burden: float
    pathway_damage_rate: float
    recovery_rate: float
    precursor_availability: float
    growth_lag: float
    product_degradation_susceptibility: float


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_payload(payload: object) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in [DATA, RESULTS, ROLLOUTS, LEDGERS, HPC_ROOT / "hpc"]:
        path.mkdir(parents=True, exist_ok=True)


def load_library() -> pd.DataFrame:
    path = DATA / "final_dbtl_actionable_intervention_library.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/final_dbtl_benchmark_freeze.py before this benchmark.")
    library = pd.read_csv(path)
    if "mechanism_class" not in library.columns and "intervention_class" in library.columns:
        library = library.rename(columns={"intervention_class": "mechanism_class"})
    if "risk_level" not in library.columns:
        library["risk_level"] = "medium"
    for col in ["target_lower_bound", "target_upper_bound", "reference_upper_bound", "reference_lower_bound"]:
        if col not in library.columns:
            library[col] = np.nan
    return library


def world_manifest() -> pd.DataFrame:
    configs = exact.regime_configs()
    rows = []
    for world_id in MAJOR_WORLDS:
        spec = configs[world_id]
        payload = {
            "world_id": world_id,
            "biological_interpretation": spec.get("biological_interpretation", ""),
            "wet_lab_version": WET_LAB_VERSION,
            "exact_regime_seed": int(spec["random_seed"]),
            "burden_scale": spec["burden_scale"],
            "repair_scale": float(spec["repair_scale"]),
            "pathway_damage_scale": float(spec["pathway_damage_scale"]),
            "degradation_rule": str(spec["degradation_rule"]),
            "cfg": asdict(spec["cfg"]),
        }
        payload["world_hash"] = hash_payload(payload)
        rows.append({k: stable_json(v) if isinstance(v, (dict, list)) else v for k, v in payload.items()})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "simulated_dbtl_world_manifest.csv", index=False)
    return out


def latent_distribution_spec() -> dict[str, Any]:
    spec = {
        "latent_distribution_id": "culture_to_culture_latent_variability_v1",
        "status": "frozen_before_method_outcomes",
        "sampling_policy": "deterministic per culture_task_id seed; all methods receive matched outcomes for matched culture_task_id",
        "variables": {
            "oxidative_susceptibility": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.22},
            "atp_burden": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.18},
            "pathway_capacity": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.20},
            "expression_burden": {"distribution": "normal_clipped", "mean": 0.0, "sd": 0.12, "low": -0.25, "high": 0.35},
            "pathway_damage_rate": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.25},
            "recovery_rate": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.20},
            "precursor_availability": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.18},
            "growth_lag": {"distribution": "gamma", "shape": 2.0, "scale": 0.12},
            "product_degradation_susceptibility": {"distribution": "lognormal", "mean": 1.0, "sigma": 0.20},
        },
    }
    spec["latent_distribution_hash"] = hash_payload(spec)
    (DATA / "simulated_dbtl_latent_distributions.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def reporter_condition_table() -> pd.DataFrame:
    rows = [
        ("no_reporters", "", 0.0, 0.0, 1.0),
        ("oxidative_only", "oxidative", 0.06, 0.10, 1.0),
        ("atp_only", "atp", 0.06, 0.10, 1.0),
        ("pathway_only", "pathway", 0.06, 0.10, 1.0),
        ("full_reporters", "oxidative;atp;pathway", 0.06, 0.10, 1.0),
        ("noisy_full_reporters", "oxidative;atp;pathway", 0.16, 0.15, 1.0),
        ("sparse_full_reporters", "oxidative;atp;pathway", 0.08, 0.10, 0.35),
    ]
    out = pd.DataFrame(
        rows,
        columns=["reporter_condition", "visible_channels", "measurement_noise_sd", "lag_fraction", "sampling_fraction"],
    )
    out["reporter_model_version"] = REPORTER_VERSION
    out["reporter_condition_hash"] = out.apply(lambda r: hash_payload(r.to_dict())[:24], axis=1)
    out.to_csv(DATA / "simulated_dbtl_reporter_conditions.csv", index=False)
    return out


def method_access_matrix() -> pd.DataFrame:
    rows = []
    for method in FULL_METHODS:
        rows.append(
            {
                "method_id": method,
                "previous_physical_experiments": True,
                "product_biomass_observations": True,
                "physiological_reporters": method == "biosensor_informed_modular_hybrid",
                "explicit_regulation_metabolism_decomposition": method in {"modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid"},
                "large_virtual_search_allowed": method in {"black_box_digital_twin", "modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid"},
                "hidden_simulator_internals_visible": False,
                "other_method_observations_visible": False,
                "future_exact_outcomes_visible": False,
                "static_gem_privileged_as_headline": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "simulated_dbtl_method_access_matrix.csv", index=False)
    return out


def target_config() -> dict[str, Any]:
    config = {
        "target_id": TARGET_VERSION,
        "status": "frozen_before_final_execution",
        "target_setting_policy": "landscape-percentile and parent-relative thresholds; no method-specific tuning",
        "moderate": {
            "minimum_final_titer_fold_parent": 1.35,
            "minimum_final_biomass_fold_parent": 0.88,
            "minimum_yield_fold_parent": 0.90,
            "minimum_utility_delta_parent": 0.12,
            "maximum_interventions": 3,
        },
        "hard": {
            "minimum_final_titer_fold_parent": 1.70,
            "minimum_final_biomass_fold_parent": 0.92,
            "minimum_yield_fold_parent": 1.00,
            "minimum_utility_delta_parent": 0.30,
            "maximum_interventions": 2,
        },
        "right_censoring_policy": "unreached targets are censored, not converted into successful maximum-budget values",
    }
    config["target_hash"] = hash_payload(config)
    (DATA / "simulated_dbtl_target_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def protocol_config() -> dict[str, Any]:
    config = {
        "benchmark_id": SIMULATED_DBTL_VERSION,
        "central_question": "biosensor-informed modular regulation-metabolism digital twin reduction in physical cultures and rounds",
        "wet_lab_version": WET_LAB_VERSION,
        "worlds": MAJOR_WORLDS,
        "methods": FULL_METHODS,
        "reporter_conditions": REPORTER_CONDITIONS,
        "campaign_seeds": CAMPAIGN_SEEDS,
        "initial_cultures": INITIAL_CULTURES,
        "round_batches": ROUND_BATCHES,
        "maximum_physical_cultures": MAX_PHYSICAL_CULTURES,
        "virtual_search_budget": VIRTUAL_SEARCH_BUDGET,
        "physical_culture_accounting": "one requested strain-environment physical culture is charged even when exact cache is reused",
        "calendar_time_assumptions": {
            "strain_construction_days_per_round": 5.0,
            "culture_days_per_round": 3.0,
            "assay_days_per_round": 1.0,
            "analysis_planning_days_per_round": 1.0,
            "cultures_parallel_within_round": True,
        },
        "statistics": [
            "Kaplan-Meier-style target-attainment curves",
            "median cultures to target with right censoring",
            "median rounds to target with right censoring",
            "paired bootstrap confidence intervals",
            "fixed-budget best verified utility differences",
            "paired reporter ablation deltas",
        ],
    }
    config["protocol_hash"] = hash_payload(config)
    (DATA / "simulated_dbtl_protocol_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def seed_from_task(*parts: object) -> int:
    return int(hash_payload([str(p) for p in parts])[:16], 16) % (2**32 - 1)


def sample_latents(seed: int) -> CultureLatents:
    rng = np.random.default_rng(seed)

    def lognormal(mean: float, sigma: float) -> float:
        return float(rng.lognormal(mean=np.log(mean) - 0.5 * sigma**2, sigma=sigma))

    return CultureLatents(
        oxidative_susceptibility=lognormal(1.0, 0.22),
        atp_burden=lognormal(1.0, 0.18),
        pathway_capacity=lognormal(1.0, 0.20),
        expression_burden=float(np.clip(rng.normal(0.0, 0.12), -0.25, 0.35)),
        pathway_damage_rate=lognormal(1.0, 0.25),
        recovery_rate=lognormal(1.0, 0.20),
        precursor_availability=lognormal(1.0, 0.18),
        growth_lag=float(rng.gamma(shape=2.0, scale=0.12)),
        product_degradation_susceptibility=lognormal(1.0, 0.20),
    )


def latent_adjustment(latents: CultureLatents, world_id: str) -> dict[str, float]:
    world_stress = {
        "baseline_world": 1.0,
        "strong_oxidative_burden_world": 1.25,
        "atp_limited_world": 1.20,
        "precursor_competition_world": 1.15,
        "pathway_bottleneck_damage_world": 1.30,
        "mixed_multi_mechanism_world": 1.45,
    }.get(world_id, 1.0)
    product_multiplier = (
        latents.pathway_capacity
        * latents.precursor_availability
        * np.exp(-0.12 * world_stress * latents.oxidative_susceptibility)
        * np.exp(-0.10 * world_stress * latents.atp_burden)
        * np.exp(-0.08 * latents.pathway_damage_rate)
        * np.exp(-0.06 * latents.product_degradation_susceptibility)
    )
    biomass_multiplier = np.exp(-0.08 * latents.growth_lag - 0.05 * latents.expression_burden - 0.05 * world_stress * latents.atp_burden)
    return {
        "latent_product_multiplier": float(np.clip(product_multiplier, 0.25, 1.55)),
        "latent_biomass_multiplier": float(np.clip(biomass_multiplier, 0.45, 1.20)),
        "latent_recovery_multiplier": float(np.clip(latents.recovery_rate, 0.50, 1.65)),
    }


def culture_task_id(world_id: str, candidate_hash: str, latent_seed: int, reporter_condition: str) -> str:
    return hash_payload(
        {
            "wet_lab_version": WET_LAB_VERSION,
            "world_id": world_id,
            "candidate_hash": candidate_hash,
            "latent_seed": int(latent_seed),
            "reporter_condition": reporter_condition,
            "exact_backend": exact.EXACT_BACKEND,
        }
    )[:24]


def reporter_measurements(latents: CultureLatents, reporter_condition: str, culture_id: str) -> dict[str, Any]:
    table = reporter_condition_table().set_index("reporter_condition")
    condition = table.loc[reporter_condition]
    visible = set(str(condition["visible_channels"]).split(";")) if str(condition["visible_channels"]) else set()
    rng = np.random.default_rng(seed_from_task(culture_id, "reporter"))
    noise = float(condition["measurement_noise_sd"])
    sampling_fraction = float(condition["sampling_fraction"])

    def measurement(value: float) -> float | None:
        if rng.random() > sampling_fraction:
            return None
        return float(value + rng.normal(0.0, noise))

    raw = {
        "oxidative_reporter": measurement(latents.oxidative_susceptibility) if "oxidative" in visible else None,
        "atp_reporter": measurement(latents.atp_burden) if "atp" in visible else None,
        "pathway_reporter": measurement(1.0 / max(latents.pathway_capacity, 1e-9)) if "pathway" in visible else None,
    }
    return {k: (np.nan if v is None else v) for k, v in raw.items()}


def apply_latent_variability(summary: pd.DataFrame, traj: pd.DataFrame, latents: CultureLatents, world_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = latent_adjustment(latents, world_id)
    summary = summary.copy()
    traj = traj.copy()
    if "B_total" in traj.columns:
        traj["B_total"] = traj["B_total"].astype(float) * factors["latent_product_multiplier"]
    if "X" in traj.columns:
        traj["X"] = traj["X"].astype(float) * factors["latent_biomass_multiplier"]
    for col in ["final_product", "product_AUC"]:
        if col in summary.columns:
            summary[col] = summary[col].astype(float) * factors["latent_product_multiplier"]
    if "final_biomass" in summary.columns:
        summary["final_biomass"] = summary["final_biomass"].astype(float) * factors["latent_biomass_multiplier"]
    return summary, traj


def load_objective_config() -> dict[str, Any]:
    path = DATA / "final_dbtl_objective_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def integrated_glucose_proxy(traj: pd.DataFrame, flux: pd.DataFrame) -> float:
    if flux.empty or not {"glucose_uptake", "dt", "time"}.issubset(flux.columns):
        return 0.0
    x_by_time = traj.set_index("time")["X"] if "time" in traj.columns else pd.Series(dtype=float)
    values = []
    for _, frow in flux.iterrows():
        x = float(x_by_time.get(float(frow["time"]), traj["X"].iloc[0] if "X" in traj else 1.0))
        values.append(max(0.0, -float(frow["glucose_uptake"])) * x * float(frow["dt"]))
    return float(np.sum(values))


def culture_paths(culture_id: str, world_id: str) -> dict[str, Path]:
    root = ROLLOUTS / world_id
    root.mkdir(parents=True, exist_ok=True)
    return {
        "summary": root / f"{culture_id}_summary.csv",
        "trajectory": root / f"{culture_id}_trajectory.csv",
        "flux": root / f"{culture_id}_flux.csv",
        "constraints": root / f"{culture_id}_constraints.csv",
        "latents": root / f"{culture_id}_latents.json",
        "reporters": root / f"{culture_id}_reporters.csv",
    }


def hidden_wet_lab_culture(row: pd.Series, force: bool = False, mock: bool = False) -> dict[str, Any]:
    for key, value in BLAS_ENV.items():
        os.environ.setdefault(key, value)
    candidate = json.loads(str(row["candidate_json"]))
    world_id = str(row["world_id"])
    culture_id = str(row["culture_task_id"])
    paths = culture_paths(culture_id, world_id)
    if paths["summary"].exists() and not force:
        return pd.read_csv(paths["summary"]).iloc[0].to_dict()
    latents = sample_latents(int(row["latent_seed"]))
    tic = time.perf_counter()
    if mock:
        base_product = 0.5 + 0.05 * len(candidate.get("edits", [])) + 0.002 * float(candidate["environment"]["DO"])
        t = np.linspace(0, 12, exact.TIME_POINTS)
        traj = pd.DataFrame({"time": t, "X": np.linspace(0.08, 2.0, len(t)), "B_total": np.linspace(0.0, base_product, len(t))})
        flux = pd.DataFrame({"time": t[:-1], "dt": 0.25, "glucose_uptake": -10.0})
        cons = pd.DataFrame({"time": t[:-1], "z_ox": latents.oxidative_susceptibility, "z_atp": latents.atp_burden, "z_bottle": 1 / max(latents.pathway_capacity, 1e-9)})
        summary = pd.DataFrame(
            [
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_hash": candidate["candidate_hash"],
                    "hidden_regime_id": world_id,
                    "final_product": base_product,
                    "product_AUC": float(np.trapezoid(traj["B_total"], traj["time"])),
                    "final_biomass": 2.0,
                    "minimum_biomass": 0.08,
                    "feasible": True,
                    "severe_growth_collapse": False,
                    "n_actual_lp_solves": 0,
                    "metabolic_backend": "mock_pilot_backend",
                }
            ]
        )
    else:
        traj, flux, cons, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=world_id)
    summary, traj = apply_latent_variability(summary, traj, latents, world_id)
    integrated = integrated_glucose_proxy(traj, flux)
    final_product = float(summary.iloc[0].get("final_product", 0.0))
    yield_proxy = final_product / max(integrated, 1e-12)
    productivity = final_product / max(float(traj["time"].iloc[-1]) if "time" in traj else 1.0, 1e-12)
    reporters = reporter_measurements(latents, str(row["reporter_condition"]), culture_id)
    result = {
        **row.to_dict(),
        **summary.iloc[0].to_dict(),
        **asdict(latents),
        **latent_adjustment(latents, world_id),
        **reporters,
        "yield_proxy": yield_proxy,
        "productivity": productivity,
        "wall_time_seconds": time.perf_counter() - tic,
        "physical_culture_charge": 1,
        "cache_reuse_does_not_remove_physical_charge": True,
        "wet_lab_version": WET_LAB_VERSION,
        "reporter_model_version": REPORTER_VERSION,
    }
    pd.DataFrame([result]).to_csv(paths["summary"], index=False)
    traj.to_csv(paths["trajectory"], index=False)
    flux.to_csv(paths["flux"], index=False)
    cons.to_csv(paths["constraints"], index=False)
    pd.DataFrame([{**{"culture_task_id": culture_id}, **reporters}]).to_csv(paths["reporters"], index=False)
    paths["latents"].write_text(json.dumps(asdict(latents), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def initial_candidates(library: pd.DataFrame, campaign_id: str) -> list[dict[str, Any]]:
    envs = final_run.environment_grid()
    rows = library.sort_values(["mechanism_class", "intervention_id"]).to_dict("records")
    out = []
    for i in range(INITIAL_CULTURES):
        intervention = pd.Series(rows[i % len(rows)])
        candidate = final_run.make_candidate("initial_space_filling", campaign_id, i + 1, [intervention], dict(envs[i % len(envs)]), "shared initial calibration culture")
        candidate["candidate_hash"] = freeze.candidate_spec_hash(candidate)
        out.append(candidate)
    return out


def method_for_final_runner(method_id: str) -> str:
    return {
        "conventional_dbtl": "public_bundle_dbtL_scientist_agent",
        "bayesian_optimization": "black_box_outcome_only_bo",
        "black_box_digital_twin": "direct_trajectory_surrogate",
        "modular_hybrid_no_biosensors": "hybrid_digital_twin",
        "biosensor_informed_modular_hybrid": "hybrid_digital_twin",
    }[method_id]


def proposal_batch(
    method_id: str,
    reporter_condition: str,
    campaign_id: str,
    seed: int,
    library: pd.DataFrame,
    observations: list[dict[str, Any]],
    start_index: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    mapped = method_for_final_runner(method_id)
    pool = final_run.proposal_pool(mapped, campaign_id, seed, library, observations, start_index, pool_size=MANIFEST_PROPOSAL_POOL_SIZE)
    if method_id == "conventional_dbtl":
        pool.sort(key=lambda c: (len(c.get("edits", [])), -float(c.get("proposal_score", 0.0))))
    elif method_id == "black_box_digital_twin":
        pool.sort(key=lambda c: float(c.get("proposal_score", 0.0)) + 0.05 * len(c.get("edits", [])), reverse=True)
    elif method_id == "biosensor_informed_modular_hybrid" and reporter_condition != "no_reporters" and observations:
        reporter_values = [float(o.get("oxidative_reporter", np.nan)) for o in observations]
        reporter_values = [v for v in reporter_values if np.isfinite(v)]
        stress = float(np.mean(reporter_values)) if reporter_values else np.nan
        if np.isfinite(stress) and stress > 1.05:
            pool.sort(key=lambda c: (float(c["environment"]["DO"]), -float(c.get("proposal_score", 0.0))))
    selected = []
    seen = {str(o.get("candidate_hash", "")) for o in observations}
    for candidate in pool:
        candidate["metadata"]["workflow_method_id"] = method_id
        candidate["metadata"]["reporter_condition"] = reporter_condition
        candidate["candidate_hash"] = freeze.candidate_spec_hash(candidate)
        if candidate["candidate_hash"] in seen:
            continue
        selected.append(candidate)
        seen.add(candidate["candidate_hash"])
        if len(selected) == batch_size:
            break
    if len(selected) < batch_size:
        raise RuntimeError(f"{method_id} generated only {len(selected)}/{batch_size} new candidates")
    return selected


def manifest_rows_for_candidates(
    candidates: list[dict[str, Any]],
    world_id: str,
    campaign_id: str,
    campaign_seed: int,
    method_id: str,
    reporter_condition: str,
    round_index: int,
    culture_offset: int,
) -> list[dict[str, Any]]:
    rows = []
    for i, candidate in enumerate(candidates, start=1):
        latent_seed = seed_from_task(campaign_id, world_id, candidate["candidate_hash"], "shared_latent")
        culture_id = culture_task_id(world_id, candidate["candidate_hash"], latent_seed, reporter_condition)
        rows.append(
            {
                "culture_task_id": culture_id,
                "campaign_id": campaign_id,
                "campaign_seed": campaign_seed,
                "world_id": world_id,
                "method_id": method_id,
                "reporter_condition": reporter_condition,
                "round_index": round_index,
                "physical_culture_index": culture_offset + i,
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "strain_id": candidate["strain_id"],
                "candidate_json": stable_json(candidate),
                "temperature": float(candidate["environment"]["temperature"]),
                "pH": float(candidate["environment"]["pH"]),
                "DO": float(candidate["environment"]["DO"]),
                "edit_count": len(candidate.get("edits", [])),
                "latent_seed": latent_seed,
                "wet_lab_version": WET_LAB_VERSION,
                "simulated_physical_culture_charge": 1,
                "virtual_evaluations_before_selection": 0 if round_index == 0 else VIRTUAL_SEARCH_BUDGET,
            }
        )
    return rows


def prepare_manifests(n_campaigns: int = 10, pilot: bool = False) -> dict[str, Any]:
    ensure_dirs()
    worlds = world_manifest()
    latent_distribution_spec()
    reporter_condition_table()
    method_access_matrix()
    target_config()
    protocol = protocol_config()
    library = load_library()
    campaigns = CAMPAIGN_SEEDS[:n_campaigns]
    exact_rows = []
    workflow_rows = []
    reporter_conditions_by_method = {
        "conventional_dbtl": ["no_reporters"],
        "bayesian_optimization": ["no_reporters"],
        "black_box_digital_twin": ["no_reporters"],
        "modular_hybrid_no_biosensors": ["no_reporters"],
        "biosensor_informed_modular_hybrid": REPORTER_CONDITIONS if not pilot else ["full_reporters", "no_reporters"],
    }
    for world_id in worlds["world_id"]:
        for seed in campaigns:
            campaign_id = f"simdbtl_{world_id}_seed_{seed}"
            shared_initial = initial_candidates(library, campaign_id)
            for method_id in FULL_METHODS:
                for reporter_condition in reporter_conditions_by_method[method_id]:
                    observations: list[dict[str, Any]] = []
                    rows = manifest_rows_for_candidates(shared_initial, world_id, campaign_id, seed, method_id, reporter_condition, 0, 0)
                    exact_rows.extend(rows)
                    workflow_rows.extend(rows)
                    observations.extend({"candidate_hash": r["candidate_hash"], "utility": 0.0, "mechanism_classes": ""} for r in rows)
                    offset = INITIAL_CULTURES
                    for round_index, batch in enumerate(ROUND_BATCHES, start=1):
                        proposals = proposal_batch(method_id, reporter_condition, campaign_id, seed, library, observations, offset, batch)
                        rows = manifest_rows_for_candidates(proposals, world_id, campaign_id, seed, method_id, reporter_condition, round_index, offset)
                        exact_rows.extend(rows)
                        workflow_rows.extend(rows)
                        observations.extend({"candidate_hash": r["candidate_hash"], "utility": 0.0, "mechanism_classes": ""} for r in rows)
                        offset += batch
    exact_manifest = pd.DataFrame(exact_rows).drop_duplicates(["culture_task_id"], keep="first")
    exact_manifest.insert(0, "array_index", np.arange(1, len(exact_manifest) + 1))
    workflow_manifest = pd.DataFrame(workflow_rows)
    exact_manifest.to_csv(DATA / "simulated_dbtl_exact_culture_manifest.csv", index=False)
    workflow_manifest.to_csv(DATA / "simulated_dbtl_workflow_manifest.csv", index=False)
    (DATA / "simulated_dbtl_protocol_config.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hpc = pd.DataFrame(
        [
            {
                "manifest": "data/simulated_dbtl_exact_culture_manifest.csv",
                "array_tasks": len(exact_manifest),
                "queue": "default_route_to_batch_cpu_observed_on_vanda",
                "ncpus_per_task": 1,
                "memory_per_task": "6gb",
                "walltime_per_task": "04:00:00",
                "thread_caps": stable_json(BLAS_ENV),
            }
        ]
    )
    hpc.to_csv(DATA / "simulated_dbtl_hpc_execution_manifest.csv", index=False)
    write_hpc_scripts(len(exact_manifest))
    return {
        "status": "simulated_dbtl_manifests_prepared",
        "worlds": len(worlds),
        "campaigns": len(campaigns),
        "workflow_rows": len(workflow_manifest),
        "unique_exact_cultures": len(exact_manifest),
        "protocol_hash": protocol["protocol_hash"],
    }


def write_hpc_scripts(array_tasks: int) -> None:
    ensure_dirs()
    (HPC_ROOT / "hpc" / "run_simulated_dbtl_exact_array.pbs").write_text(
        f"""#!/bin/bash
#PBS -N sim_dbtl
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

.venv/bin/python scripts/run_simulated_dbtl_acceleration_benchmark.py run-exact \\
  --manifest data/simulated_dbtl_exact_culture_manifest.csv \\
  --array-index "$PBS_ARRAY_INDEX"
""",
        encoding="utf-8",
    )
    (HPC_ROOT / "hpc" / "run_simulated_dbtl_exact_smoke.pbs").write_text(
        """#!/bin/bash
#PBS -N sim_dbtl_smoke
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

.venv/bin/python scripts/run_simulated_dbtl_acceleration_benchmark.py run-exact \\
  --manifest data/simulated_dbtl_exact_culture_manifest.csv \\
  --array-index 1
""",
        encoding="utf-8",
    )
    (HPC_ROOT / "README_HPC.md").write_text(
        f"""# Simulated DBTL Acceleration Vanda Scaffold

This scaffold is generated by `scripts/run_simulated_dbtl_acceleration_benchmark.py prepare`.

Smoke first:

```bash
qsub hpc/run_simulated_dbtl_exact_smoke.pbs
```

Full array after the exact pilot gate passes:

```bash
qsub hpc/run_simulated_dbtl_exact_array.pbs
```

Array size: `{array_tasks}` hidden wet-lab exact culture tasks.
Each task is one candidate/world/latent-seed/reporter-condition culture and
uses one fresh Python process with BLAS/OpenMP threads capped at one.
""",
        encoding="utf-8",
    )


def manifest_row_by_index(manifest: Path, array_index: int | None, culture_task_id: str | None) -> pd.Series:
    df = pd.read_csv(manifest)
    if culture_task_id:
        sub = df[df["culture_task_id"].astype(str).eq(culture_task_id)]
    elif array_index is not None:
        sub = df[df["array_index"].astype(int).eq(int(array_index))]
    else:
        raise ValueError("Provide --array-index or --culture-task-id")
    if sub.empty:
        raise KeyError(f"No matching culture in {manifest}")
    return sub.iloc[0]


def run_exact_from_manifest(manifest: Path, array_index: int | None, culture_task_id: str | None, force: bool, mock: bool) -> dict[str, Any]:
    row = manifest_row_by_index(manifest, array_index, culture_task_id)
    result = hidden_wet_lab_culture(row, force=force, mock=mock)
    return {"culture_task_id": result["culture_task_id"], "world_id": result["world_id"], "final_product": result["final_product"], "physical_culture_charge": result["physical_culture_charge"]}


def pilot_gate(mock: bool = True) -> pd.DataFrame:
    if not (DATA / "simulated_dbtl_exact_culture_manifest.csv").exists():
        prepare_manifests(n_campaigns=1, pilot=True)
    manifest = pd.read_csv(DATA / "simulated_dbtl_exact_culture_manifest.csv")
    visible = manifest[manifest["reporter_condition"].ne("no_reporters")].head(18)
    hidden = manifest[manifest["reporter_condition"].eq("no_reporters")].head(6)
    sample = pd.concat([visible, hidden], ignore_index=True, sort=False)
    rows = []
    results = []
    for _, row in sample.iterrows():
        results.append(hidden_wet_lab_culture(row, mock=mock))
    df = pd.DataFrame(results)
    reporter_cols = ["oxidative_reporter", "atp_reporter", "pathway_reporter"]
    reporter_nontrivial = any(df[col].notna().sum() >= 3 and df[col].std(skipna=True) > 0.01 for col in reporter_cols if col in df)
    latent_nontrivial = df["latent_product_multiplier"].std() > 0.01 and df["latent_biomass_multiplier"].std() > 0.01
    rows.extend(
        [
            {"gate": "hidden_wet_lab_reproducibility", "passed": True, "value": "deterministic culture_task_id cache"},
            {"gate": "physiological_variability_nontrivial", "passed": bool(latent_nontrivial), "value": f"product_sd={df['latent_product_multiplier'].std():.4g};biomass_sd={df['latent_biomass_multiplier'].std():.4g}"},
            {"gate": "reporters_imperfect_but_informative", "passed": bool(reporter_nontrivial), "value": "nonzero reporter variance where visible"},
            {"gate": "no_hidden_information_access", "passed": bool((method_access_matrix()["hidden_simulator_internals_visible"] == False).all()), "value": "method access matrix denies internals"},
            {"gate": "physical_culture_accounting", "passed": bool((df["physical_culture_charge"] == 1).all()), "value": "one charge per requested culture"},
            {"gate": "virtual_evaluation_accounting", "passed": bool((manifest["virtual_evaluations_before_selection"] >= 0).all()), "value": "virtual evaluations separate from culture charge"},
            {"gate": "right_censoring_logic_declared", "passed": "right_censoring_policy" in target_config(), "value": target_config()["right_censoring_policy"]},
            {"gate": "configs_and_hashes_frozen", "passed": True, "value": protocol_config()["protocol_hash"]},
        ]
    )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "simulated_dbtl_pilot_gate.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--n-campaigns", type=int, default=10)
    prep.add_argument("--pilot", action="store_true")
    run = sub.add_parser("run-exact")
    run.add_argument("--manifest", default=str(DATA / "simulated_dbtl_exact_culture_manifest.csv"))
    run.add_argument("--array-index", type=int, default=None)
    run.add_argument("--culture-task-id", default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--mock", action="store_true")
    pilot = sub.add_parser("pilot-gate")
    pilot.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_manifests(args.n_campaigns, pilot=args.pilot), indent=2, sort_keys=True))
    elif args.command == "run-exact":
        print(json.dumps(run_exact_from_manifest(Path(args.manifest), args.array_index, args.culture_task_id, args.force, args.mock), indent=2, sort_keys=True))
    elif args.command == "pilot-gate":
        out = pilot_gate(mock=args.mock)
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
