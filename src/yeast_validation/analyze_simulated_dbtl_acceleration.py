#!/usr/bin/env python3
"""Audit and analyse the frozen simulated DBTL acceleration benchmark.

This script is intentionally read-only with respect to the frozen benchmark
protocol. It consumes the predeclared manifests plus exact hidden-wet-lab
outputs and writes completion audits, reconstructed workflow trajectories,
right-censored target summaries, paired comparisons, and publication figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "simulated_dbtl_acceleration"
ROLLOUTS = RESULTS / "hidden_wet_lab_rollouts"
FIGURES = ROOT / "figures"

EXPECTED_PROTOCOL_HASH = "2ed0c8e083537bf918013619c3ed0081bc930fd52bd725624d23940083dc1612"
EXPECTED_WET_LAB_VERSION = "yeast9_dynamic_pfba_with_frozen_latent_culture_variability_v1"
EXPECTED_BACKEND = "yeast_gem_lp"
EXPECTED_STATUS = "complete_exact_dynamic_pfba"
MAX_CULTURES = 24
MAX_ROUND_INDEX = 3
CENSORED_CULTURE_VALUE = MAX_CULTURES + 1
CENSORED_ROUND_VALUE = MAX_ROUND_INDEX + 1
PRIMARY_TARGET_TIER = "moderate"
CANONICAL_BIOSENSOR_WORKFLOW = "biosensor_informed_modular_hybrid__full_reporters"

REPORTER_CHANNELS = {
    "no_reporters": set(),
    "oxidative_only": {"oxidative_reporter"},
    "atp_only": {"atp_reporter"},
    "pathway_only": {"pathway_reporter"},
    "full_reporters": {"oxidative_reporter", "atp_reporter", "pathway_reporter"},
    "noisy_full_reporters": {"oxidative_reporter", "atp_reporter", "pathway_reporter"},
    "sparse_full_reporters": {"oxidative_reporter", "atp_reporter", "pathway_reporter"},
}

WORKFLOW_LABELS = {
    "conventional_dbtl": "Conventional DBTL",
    "bayesian_optimization": "Bayesian optimisation",
    "black_box_digital_twin": "Black-box digital twin",
    "modular_hybrid_no_biosensors": "Modular hybrid, no biosensors",
    "biosensor_informed_modular_hybrid__no_reporters": "Biosensor hybrid, no reporters",
    "biosensor_informed_modular_hybrid__oxidative_only": "Biosensor hybrid, oxidative",
    "biosensor_informed_modular_hybrid__atp_only": "Biosensor hybrid, ATP",
    "biosensor_informed_modular_hybrid__pathway_only": "Biosensor hybrid, pathway",
    "biosensor_informed_modular_hybrid__full_reporters": "Biosensor hybrid, full",
    "biosensor_informed_modular_hybrid__noisy_full_reporters": "Biosensor hybrid, noisy full",
    "biosensor_informed_modular_hybrid__sparse_full_reporters": "Biosensor hybrid, sparse full",
}

PLOT_COLORS = {
    "conventional_dbtl": "#4c78a8",
    "bayesian_optimization": "#f58518",
    "black_box_digital_twin": "#54a24b",
    "modular_hybrid_no_biosensors": "#b279a2",
    "biosensor_informed_modular_hybrid__no_reporters": "#9c755f",
    "biosensor_informed_modular_hybrid__oxidative_only": "#e45756",
    "biosensor_informed_modular_hybrid__atp_only": "#72b7b2",
    "biosensor_informed_modular_hybrid__pathway_only": "#ff9da6",
    "biosensor_informed_modular_hybrid__full_reporters": "#2f4b7c",
    "biosensor_informed_modular_hybrid__noisy_full_reporters": "#8cd17d",
    "biosensor_informed_modular_hybrid__sparse_full_reporters": "#b6992d",
}


def ensure_dirs() -> None:
    for path in [DATA, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_id(method_id: str, reporter_condition: str) -> str:
    if method_id == "biosensor_informed_modular_hybrid":
        return f"{method_id}__{reporter_condition}"
    return method_id


def workflow_label(wid: str) -> str:
    return WORKFLOW_LABELS.get(wid, wid)


def culture_paths(culture_id: str, world_id: str, rollouts: Path = ROLLOUTS) -> dict[str, Path]:
    root = rollouts / world_id
    return {
        "summary": root / f"{culture_id}_summary.csv",
        "trajectory": root / f"{culture_id}_trajectory.csv",
        "flux": root / f"{culture_id}_flux.csv",
        "constraints": root / f"{culture_id}_constraints.csv",
        "latents": root / f"{culture_id}_latents.json",
        "reporters": root / f"{culture_id}_reporters.csv",
    }


def is_finite(value: Any) -> bool:
    try:
        return bool(pd.notna(value) and np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def reporter_visibility_valid(row: pd.Series) -> bool:
    condition = str(row.get("reporter_condition", ""))
    visible = REPORTER_CHANNELS.get(condition, set())
    cols = {"oxidative_reporter", "atp_reporter", "pathway_reporter"}
    finite = {col for col in cols if is_finite(row.get(col, np.nan))}
    if condition == "sparse_full_reporters":
        return finite.issubset(visible)
    return finite == visible


def reporter_biology_valid(row: pd.Series) -> bool:
    checks = []
    if is_finite(row.get("oxidative_reporter", np.nan)):
        checks.append(abs(float(row["oxidative_reporter"]) - float(row["oxidative_susceptibility"])) < 0.65)
    if is_finite(row.get("atp_reporter", np.nan)):
        checks.append(abs(float(row["atp_reporter"]) - float(row["atp_burden"])) < 0.65)
    if is_finite(row.get("pathway_reporter", np.nan)):
        expected = 1.0 / max(float(row["pathway_capacity"]), 1e-12)
        checks.append(abs(float(row["pathway_reporter"]) - expected) < 0.65)
    return all(checks)


def read_exact_summaries(manifest: pd.DataFrame, rollouts: Path = ROLLOUTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for expected in manifest.itertuples(index=False):
        expected_dict = expected._asdict()
        culture_id = str(expected_dict["culture_task_id"])
        world_id = str(expected_dict["world_id"])
        paths = culture_paths(culture_id, world_id, rollouts=rollouts)
        artifact_missing = [name for name, path in paths.items() if not path.exists()]
        summary_present = paths["summary"].exists()
        duplicate_manifest_task = culture_id in seen_task_ids
        seen_task_ids.add(culture_id)
        summary: dict[str, Any] = {}
        read_error = ""
        if not summary_present:
            read_error = "summary_missing"
        else:
            try:
                frame = pd.read_csv(paths["summary"])
                if len(frame) != 1:
                    read_error = f"summary_row_count_{len(frame)}"
                elif "culture_task_id" in frame.columns and str(frame.iloc[0]["culture_task_id"]) != culture_id:
                    read_error = "summary_culture_task_id_mismatch"
                summary = frame.iloc[0].to_dict() if len(frame) else {}
            except Exception as exc:  # pragma: no cover - defensive audit path
                read_error = f"summary_read_error:{type(exc).__name__}"
        merged = {**expected_dict, **summary}
        if summary:
            rows.append(merged)
        required_match_cols = [
            "culture_task_id",
            "world_id",
            "campaign_seed",
            "campaign_id",
            "method_id",
            "reporter_condition",
            "round_index",
            "physical_culture_index",
            "candidate_hash",
            "wet_lab_version",
        ]
        hash_mismatch = False
        mismatch_cols = []
        for col in required_match_cols:
            if col in summary and str(summary[col]) != str(expected_dict.get(col, "")):
                hash_mismatch = True
                mismatch_cols.append(col)
        summary_series = pd.Series(merged)
        lp_sum = sum(float(summary_series.get(col, np.nan)) for col in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations"] if pd.notna(summary_series.get(col, np.nan)))
        lp_ok = is_finite(summary_series.get("n_actual_lp_solves", np.nan)) and float(summary_series["n_actual_lp_solves"]) > 0 and abs(float(summary_series["n_actual_lp_solves"]) - lp_sum) < 1e-9
        solver_ok = all(float(summary_series.get(col, 1)) == 0 for col in ["n_solver_errors", "n_infeasible_solves", "n_unbounded_solves"] if pd.notna(summary_series.get(col, np.nan)))
        latent_cols = [
            "oxidative_susceptibility",
            "atp_burden",
            "pathway_capacity",
            "expression_burden",
            "pathway_damage_rate",
            "recovery_rate",
            "precursor_availability",
            "growth_lag",
            "product_degradation_susceptibility",
        ]
        latent_ok = all(is_finite(summary_series.get(col, np.nan)) for col in latent_cols)
        charge_ok = (
            is_finite(summary_series.get("physical_culture_charge", np.nan))
            and int(float(summary_series["physical_culture_charge"])) == 1
            and is_finite(summary_series.get("simulated_physical_culture_charge", np.nan))
            and int(float(summary_series["simulated_physical_culture_charge"])) == 1
        )
        status_ok = str(summary_series.get("exact_status", "")) == EXPECTED_STATUS
        backend_ok = str(summary_series.get("metabolic_backend", "")) == EXPECTED_BACKEND
        wetlab_ok = str(summary_series.get("wet_lab_version", "")) == EXPECTED_WET_LAB_VERSION
        reporter_ok = reporter_visibility_valid(summary_series) and reporter_biology_valid(summary_series)
        valid = (
            not artifact_missing
            and not duplicate_manifest_task
            and not read_error
            and not hash_mismatch
            and status_ok
            and backend_ok
            and wetlab_ok
            and lp_ok
            and solver_ok
            and latent_ok
            and charge_ok
            and reporter_ok
        )
        audit_rows.append(
            {
                **expected_dict,
                "summary_path": str(paths["summary"]),
                "trajectory_path": str(paths["trajectory"]),
                "flux_path": str(paths["flux"]),
                "constraints_path": str(paths["constraints"]),
                "latents_path": str(paths["latents"]),
                "reporters_path": str(paths["reporters"]),
                "summary_present": summary_present,
                "artifact_missing": ";".join(artifact_missing),
                "read_error": read_error,
                "duplicate_manifest_task": duplicate_manifest_task,
                "hash_mismatch": hash_mismatch,
                "mismatch_cols": ";".join(mismatch_cols),
                "status_ok": status_ok,
                "backend_ok": backend_ok,
                "wetlab_version_ok": wetlab_ok,
                "lp_accounting_ok": lp_ok,
                "solver_ok": solver_ok,
                "latent_metadata_ok": latent_ok,
                "reporter_visibility_and_biology_ok": reporter_ok,
                "physical_charge_ok": charge_ok,
                "valid_exact_task": valid,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def completion_audit(rollouts: Path = ROLLOUTS) -> dict[str, Any]:
    ensure_dirs()
    manifest = pd.read_csv(DATA / "simulated_dbtl_exact_culture_manifest.csv")
    summaries, validity = read_exact_summaries(manifest, rollouts=rollouts)
    completed_ids = set(summaries["culture_task_id"].astype(str)) if "culture_task_id" in summaries else set()
    expected_ids = set(manifest["culture_task_id"].astype(str))
    missing_ids = sorted(expected_ids - completed_ids)
    duplicate_output_tasks = int(summaries["culture_task_id"].duplicated().sum()) if "culture_task_id" in summaries else 0
    completed_mask = validity["summary_present"].astype(bool)
    invalid = validity[completed_mask & ~validity["valid_exact_task"].astype(bool)].copy()
    hash_mismatch_tasks = int((completed_mask & validity["hash_mismatch"].astype(bool)).sum())
    failed_tasks = int((completed_mask & (~validity["status_ok"].astype(bool) | ~validity["solver_ok"].astype(bool))).sum())
    summary = {
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "expected_tasks": int(len(manifest)),
        "completed_tasks": int(len(completed_ids)),
        "missing_tasks": int(len(missing_ids)),
        "failed_tasks": failed_tasks,
        "duplicate_tasks": duplicate_output_tasks,
        "hash_mismatch_tasks": hash_mismatch_tasks,
        "invalid_output_tasks": int(len(invalid)),
        "valid_tasks": int(validity["valid_exact_task"].astype(bool).sum()),
        "all_valid_complete": bool(len(manifest) == len(completed_ids) and len(invalid) == 0),
    }
    validity.to_csv(DATA / "simulated_dbtl_exact_task_validity.csv", index=False)
    pd.DataFrame([summary]).to_csv(DATA / "simulated_dbtl_hpc_completion_audit.csv", index=False)
    pd.DataFrame({"culture_task_id": missing_ids}).to_csv(DATA / "simulated_dbtl_missing_exact_tasks.csv", index=False)
    invalid.to_csv(DATA / "simulated_dbtl_invalid_exact_tasks.csv", index=False)
    return summary


def bootstrap_ci(values: np.ndarray, fn=np.mean, n_boot: int = 5000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def parent_reference_table(objective: dict[str, Any]) -> pd.DataFrame:
    parent_path = DATA / "harder_dbtl_exact_cache_index.csv"
    if not parent_path.exists():
        raise FileNotFoundError("Need data/harder_dbtl_exact_cache_index.csv for frozen per-world parent controls.")
    parent = pd.read_csv(parent_path)
    parent = parent[parent["strain_id"].astype(str).eq("parent_no_edit")].copy()
    parent = parent[parent["world_id"].isin(pd.read_csv(DATA / "simulated_dbtl_world_manifest.csv")["world_id"].astype(str))]
    parent["env_key"] = parent.apply(lambda r: env_key(r), axis=1)
    parent["parent_utility"] = parent.apply(lambda r: utility_from_row(r, objective), axis=1)
    cols = [
        "world_id",
        "env_key",
        "final_product",
        "product_AUC",
        "final_biomass",
        "yield_proxy",
        "productivity",
        "parent_utility",
    ]
    out = parent[cols].rename(
        columns={
            "final_product": "parent_final_product",
            "product_AUC": "parent_product_AUC",
            "final_biomass": "parent_final_biomass",
            "yield_proxy": "parent_yield_proxy",
            "productivity": "parent_productivity",
        }
    )
    out.to_csv(DATA / "simulated_dbtl_parent_reference_table.csv", index=False)
    return out


def env_key(row: pd.Series) -> str:
    return f"T={float(row['temperature']):.6g}|pH={float(row['pH']):.6g}|DO={float(row['DO']):.6g}"


def utility_from_row(row: pd.Series, objective: dict[str, Any]) -> float:
    import final_dbtl_benchmark_freeze as freeze

    metrics = {
        "final_product": float(row.get("final_product", 0.0)),
        "product_AUC": float(row.get("product_AUC", 0.0)),
        "yield_proxy": float(row.get("yield_proxy", 0.0)),
        "final_biomass": float(row.get("final_biomass", 0.0)),
        "growth_failure": bool(row.get("severe_growth_collapse", False)),
        "infeasible": not bool(row.get("feasible", True)),
        "edit_count": int(float(row.get("edit_count", 0) or 0)),
        "construction_risk": str(row.get("construction_risk", "medium")),
    }
    return freeze.final_utility(metrics, objective)


def assign_targets(obs: pd.DataFrame, target_cfg: dict[str, Any], objective: dict[str, Any]) -> pd.DataFrame:
    parent = parent_reference_table(objective)
    obs = obs.copy()
    obs["env_key"] = obs.apply(lambda r: env_key(r), axis=1)
    obs = obs.merge(parent, on=["world_id", "env_key"], how="left", validate="many_to_one")
    if obs["parent_final_product"].isna().any():
        missing = obs[obs["parent_final_product"].isna()][["world_id", "env_key"]].drop_duplicates()
        raise ValueError(f"Missing frozen parent controls for {len(missing)} world/environment pairs")
    obs["utility"] = obs.apply(lambda r: utility_from_row(r, objective), axis=1)
    obs["utility_delta_vs_parent"] = obs["utility"] - obs["parent_utility"]
    obs["product_fold_parent"] = obs["final_product"].astype(float) / obs["parent_final_product"].astype(float).clip(lower=1e-12)
    obs["biomass_fold_parent"] = obs["final_biomass"].astype(float) / obs["parent_final_biomass"].astype(float).clip(lower=1e-12)
    obs["yield_fold_parent"] = obs["yield_proxy"].astype(float) / obs["parent_yield_proxy"].astype(float).clip(lower=1e-12)
    for tier in ["moderate", "hard"]:
        rules = target_cfg[tier]
        obs[f"reaches_{tier}_target"] = (
            (obs["product_fold_parent"] >= float(rules["minimum_final_titer_fold_parent"]))
            & (obs["biomass_fold_parent"] >= float(rules["minimum_final_biomass_fold_parent"]))
            & (obs["yield_fold_parent"] >= float(rules["minimum_yield_fold_parent"]))
            & (obs["utility_delta_vs_parent"] >= float(rules["minimum_utility_delta_parent"]))
            & (obs["edit_count"].astype(int) <= int(rules["maximum_interventions"]))
            & (obs["feasible"].astype(bool))
            & (~obs["severe_growth_collapse"].astype(bool))
        )
    return obs


def reconstruct_observations(rollouts: Path = ROLLOUTS) -> pd.DataFrame:
    workflow = pd.read_csv(DATA / "simulated_dbtl_workflow_manifest.csv")
    exact_manifest = pd.read_csv(DATA / "simulated_dbtl_exact_culture_manifest.csv")
    summaries, validity = read_exact_summaries(exact_manifest, rollouts=rollouts)
    valid_ids = set(validity[validity["valid_exact_task"].astype(bool)]["culture_task_id"].astype(str))
    if len(valid_ids) != len(exact_manifest):
        raise RuntimeError(f"Exact outputs are not complete/valid: {len(valid_ids)}/{len(exact_manifest)}")
    summary_cols = [c for c in summaries.columns if c not in workflow.columns or c in {"culture_task_id"}]
    merged = workflow.merge(summaries[summary_cols], on="culture_task_id", how="left", validate="many_to_one")
    merged["workflow_id"] = merged.apply(lambda r: workflow_id(str(r["method_id"]), str(r["reporter_condition"])), axis=1)
    merged["workflow_label"] = merged["workflow_id"].map(workflow_label)
    merged["campaign_key"] = (
        merged["world_id"].astype(str)
        + "|seed_"
        + merged["campaign_seed"].astype(str)
        + "|"
        + merged["workflow_id"].astype(str)
    )
    objective = load_json(DATA / "final_dbtl_objective_config.json")
    target_cfg = load_json(DATA / "simulated_dbtl_target_config.json")
    merged = assign_targets(merged, target_cfg, objective)
    merged["physical_culture_charge_workflow"] = 1
    merged["virtual_to_physical_note"] = "virtual_evaluations_before_selection is not a physical culture"
    merged.to_csv(DATA / "simulated_dbtl_reconstructed_observations.csv", index=False)
    return merged


def trajectory_tables(obs: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows = []
    round_rows = []
    target_col = f"reaches_{tier}_target"
    for keys, group in obs.groupby(["world_id", "campaign_seed", "workflow_id", "workflow_label", "method_id", "reporter_condition"], dropna=False):
        world, seed, wid, label, method, reporter = keys
        g = group.sort_values(["physical_culture_index", "culture_task_id"]).copy()
        best_utility = -np.inf
        best_product = -np.inf
        best_yield = -np.inf
        best_productivity = -np.inf
        for _, row in g.iterrows():
            prefix = g[g["physical_culture_index"].le(int(row["physical_culture_index"]))]
            best_utility = max(best_utility, float(row["utility"]))
            best_product = max(best_product, float(row["final_product"]))
            best_yield = max(best_yield, float(row["yield_proxy"]))
            best_productivity = max(best_productivity, float(row["productivity"]))
            best_rows.append(
                {
                    "world_id": world,
                    "campaign_seed": seed,
                    "workflow_id": wid,
                    "workflow_label": label,
                    "method_id": method,
                    "reporter_condition": reporter,
                    "physical_culture_index": int(row["physical_culture_index"]),
                    "round_index": int(row["round_index"]),
                    "best_utility_so_far": best_utility,
                    "best_final_product_so_far": best_product,
                    "best_yield_proxy_so_far": best_yield,
                    "best_productivity_so_far": best_productivity,
                    "target_reached_so_far": bool(prefix[target_col].astype(bool).any()),
                }
            )
        for round_idx, rg in g.groupby("round_index"):
            prefix = g[g["physical_culture_index"].le(int(rg["physical_culture_index"].max()))]
            round_rows.append(
                {
                    "world_id": world,
                    "campaign_seed": seed,
                    "workflow_id": wid,
                    "workflow_label": label,
                    "method_id": method,
                    "reporter_condition": reporter,
                    "round_index": int(round_idx),
                    "cultures_cumulative": int(rg["physical_culture_index"].max()),
                    "best_utility_round": float(rg["utility"].max()),
                    "best_utility_cumulative": float(prefix["utility"].max()),
                    "target_reached_by_round": bool(prefix[target_col].astype(bool).any()),
                }
            )
    best = pd.DataFrame(best_rows)
    rounds = pd.DataFrame(round_rows)
    best.to_csv(DATA / f"simulated_dbtl_best_so_far_{tier}.csv", index=False)
    rounds.to_csv(DATA / f"simulated_dbtl_round_summary_{tier}.csv", index=False)
    return best, rounds


def campaign_summary(obs: pd.DataFrame, best: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> pd.DataFrame:
    target_col = f"reaches_{tier}_target"
    rows = []
    auc = (
        best.sort_values("physical_culture_index")
        .groupby(["world_id", "campaign_seed", "workflow_id"], as_index=False)
        .apply(lambda g: pd.Series({"dbtl_auc": float(np.trapezoid(g["best_utility_so_far"], g["physical_culture_index"])) if len(g) > 1 else float(g["best_utility_so_far"].iloc[0])}))
        .reset_index(drop=True)
    )
    for keys, group in obs.groupby(["world_id", "campaign_seed", "workflow_id", "workflow_label", "method_id", "reporter_condition"], dropna=False):
        world, seed, wid, label, method, reporter = keys
        g = group.sort_values(["physical_culture_index", "culture_task_id"])
        hit = g[g[target_col].astype(bool)]
        best_idx = g["utility"].astype(float).idxmax()
        best_row = g.loc[best_idx]
        cultures_to_target = int(hit["physical_culture_index"].iloc[0]) if len(hit) else np.nan
        rounds_to_target = int(hit["round_index"].iloc[0]) if len(hit) else np.nan
        rows.append(
            {
                "target_tier": tier,
                "world_id": world,
                "campaign_seed": int(seed),
                "workflow_id": wid,
                "workflow_label": label,
                "method_id": method,
                "reporter_condition": reporter,
                "target_attained": bool(len(hit)),
                "cultures_to_target": cultures_to_target,
                "rounds_to_target": rounds_to_target,
                "right_censored_cultures": not bool(len(hit)),
                "right_censored_rounds": not bool(len(hit)),
                "cultures_to_target_censored": cultures_to_target if len(hit) else CENSORED_CULTURE_VALUE,
                "rounds_to_target_censored": rounds_to_target if len(hit) else CENSORED_ROUND_VALUE,
                "best_utility": float(best_row["utility"]),
                "best_final_product": float(best_row["final_product"]),
                "best_product_AUC": float(best_row["product_AUC"]),
                "best_yield_proxy": float(best_row["yield_proxy"]),
                "best_productivity": float(best_row["productivity"]),
                "best_final_biomass": float(best_row["final_biomass"]),
                "best_edit_count": int(best_row["edit_count"]),
                "best_candidate_id": str(best_row["candidate_id"]),
                "best_culture_task_id": str(best_row["culture_task_id"]),
                "growth_collapse_rate": float(g["severe_growth_collapse"].astype(bool).mean()),
                "unique_strain_constructions": int(g["strain_id"].nunique()),
                "total_simulated_physical_cultures": int(g["physical_culture_charge_workflow"].sum()),
                "total_virtual_evaluations": int(g["virtual_evaluations_before_selection"].fillna(0).sum()),
                "virtual_to_physical_leverage": float(g["virtual_evaluations_before_selection"].fillna(0).sum() / max(g["physical_culture_charge_workflow"].sum(), 1)),
                "workflow_lp_solves_if_not_cache_deduplicated": int(g["n_actual_lp_solves"].fillna(0).sum()),
                "workflow_compute_seconds_if_not_cache_deduplicated": float(g["wall_time_seconds"].fillna(0).sum()),
            }
        )
    out = pd.DataFrame(rows).merge(auc, on=["world_id", "campaign_seed", "workflow_id"], how="left")
    out.to_csv(DATA / f"simulated_dbtl_campaign_summary_{tier}.csv", index=False)
    return out


def aggregate_summary(summary: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for wid, sub in summary.groupby("workflow_id"):
        cultures = sub["cultures_to_target_censored"].to_numpy(float)
        rounds = sub["rounds_to_target_censored"].to_numpy(float)
        success = sub["target_attained"].astype(bool).to_numpy()
        culture_ci = bootstrap_ci(cultures, np.median)
        round_ci = bootstrap_ci(rounds, np.median)
        success_ci = bootstrap_ci(success.astype(float), np.mean)
        rows.append(
            {
                "target_tier": tier,
                "workflow_id": wid,
                "workflow_label": workflow_label(wid),
                "method_id": str(sub["method_id"].iloc[0]),
                "reporter_condition": str(sub["reporter_condition"].iloc[0]),
                "n_campaigns": int(len(sub)),
                "target_attainment_count": int(success.sum()),
                "target_attainment_probability": float(success.mean()),
                "target_attainment_ci_low": success_ci[0],
                "target_attainment_ci_high": success_ci[1],
                "median_cultures_to_target_censored": float(np.median(cultures)),
                "median_cultures_ci_low": culture_ci[0],
                "median_cultures_ci_high": culture_ci[1],
                "mean_cultures_to_target_observed": float(sub.loc[sub["target_attained"], "cultures_to_target"].mean()) if success.any() else np.nan,
                "median_rounds_to_target_censored": float(np.median(rounds)),
                "median_rounds_ci_low": round_ci[0],
                "median_rounds_ci_high": round_ci[1],
                "mean_rounds_to_target_observed": float(sub.loc[sub["target_attained"], "rounds_to_target"].mean()) if success.any() else np.nan,
                "censored_campaign_count": int((~success).sum()),
                "mean_best_utility": float(sub["best_utility"].mean()),
                "mean_best_final_product": float(sub["best_final_product"].mean()),
                "mean_best_yield_proxy": float(sub["best_yield_proxy"].mean()),
                "mean_best_productivity": float(sub["best_productivity"].mean()),
                "mean_best_final_biomass": float(sub["best_final_biomass"].mean()),
                "mean_dbtl_auc": float(sub["dbtl_auc"].mean()),
                "mean_total_virtual_evaluations": float(sub["total_virtual_evaluations"].mean()),
                "mean_virtual_to_physical_leverage": float(sub["virtual_to_physical_leverage"].mean()),
            }
        )
    method = pd.DataFrame(rows).sort_values(["median_cultures_to_target_censored", "median_rounds_to_target_censored", "workflow_id"])
    method.to_csv(DATA / f"simulated_dbtl_method_summary_{tier}.csv", index=False)
    world = (
        summary.groupby(["target_tier", "world_id", "workflow_id", "workflow_label", "method_id", "reporter_condition"], as_index=False)
        .agg(
            n_campaigns=("campaign_seed", "nunique"),
            target_attainment_probability=("target_attained", "mean"),
            median_cultures_to_target_censored=("cultures_to_target_censored", "median"),
            median_rounds_to_target_censored=("rounds_to_target_censored", "median"),
            mean_best_utility=("best_utility", "mean"),
            mean_best_final_product=("best_final_product", "mean"),
            mean_dbtl_auc=("dbtl_auc", "mean"),
        )
        .sort_values(["world_id", "median_cultures_to_target_censored", "workflow_id"])
    )
    world.to_csv(DATA / f"simulated_dbtl_world_method_summary_{tier}.csv", index=False)
    return method, world


def paired_effects(summary: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> pd.DataFrame:
    comparisons = [
        ("hybrid_full_vs_conventional", CANONICAL_BIOSENSOR_WORKFLOW, "conventional_dbtl"),
        ("hybrid_full_vs_bo", CANONICAL_BIOSENSOR_WORKFLOW, "bayesian_optimization"),
        ("hybrid_full_vs_black_box", CANONICAL_BIOSENSOR_WORKFLOW, "black_box_digital_twin"),
        ("modular_no_biosensors_vs_black_box", "modular_hybrid_no_biosensors", "black_box_digital_twin"),
        ("hybrid_full_vs_modular_no_biosensors", CANONICAL_BIOSENSOR_WORKFLOW, "modular_hybrid_no_biosensors"),
    ]
    reporter_workflows = sorted(w for w in summary["workflow_id"].unique() if w.startswith("biosensor_informed_modular_hybrid__"))
    for wid in reporter_workflows:
        comparisons.append((f"{wid}_vs_modular_no_biosensors", wid, "modular_hybrid_no_biosensors"))
    rows = []
    key_cols = ["world_id", "campaign_seed"]
    for comparison, a, b in comparisons:
        left = summary[summary["workflow_id"].eq(a)]
        right = summary[summary["workflow_id"].eq(b)]
        paired = left.merge(right, on=key_cols, suffixes=("_a", "_b"))
        if paired.empty:
            continue
        a_c = paired["cultures_to_target_censored_a"].to_numpy(float)
        b_c = paired["cultures_to_target_censored_b"].to_numpy(float)
        a_r = paired["rounds_to_target_censored_a"].to_numpy(float)
        b_r = paired["rounds_to_target_censored_b"].to_numpy(float)
        culture_saved = b_c - a_c
        round_saved = b_r - a_r
        quality_delta = paired["best_utility_a"].to_numpy(float) - paired["best_utility_b"].to_numpy(float)
        rows.append(
            {
                "target_tier": tier,
                "comparison": comparison,
                "workflow_a": a,
                "workflow_b": b,
                "interpretation": "positive means workflow_a is faster/better than workflow_b",
                "n_pairs": int(len(paired)),
                "a_target_attainment_probability": float(paired["target_attained_a"].mean()),
                "b_target_attainment_probability": float(paired["target_attained_b"].mean()),
                "mean_cultures_saved_by_a": float(np.mean(culture_saved)),
                "median_cultures_saved_by_a": float(np.median(culture_saved)),
                "mean_cultures_saved_ci_low": bootstrap_ci(culture_saved, np.mean)[0],
                "mean_cultures_saved_ci_high": bootstrap_ci(culture_saved, np.mean)[1],
                "mean_percent_culture_reduction_by_a": float(np.mean(100.0 * culture_saved / np.maximum(b_c, 1.0))),
                "mean_rounds_saved_by_a": float(np.mean(round_saved)),
                "median_rounds_saved_by_a": float(np.median(round_saved)),
                "mean_rounds_saved_ci_low": bootstrap_ci(round_saved, np.mean)[0],
                "mean_rounds_saved_ci_high": bootstrap_ci(round_saved, np.mean)[1],
                "mean_percent_round_reduction_by_a": float(np.mean(100.0 * round_saved / np.maximum(b_r, 1.0))),
                "culture_win_count_a": int((a_c < b_c).sum()),
                "culture_loss_count_a": int((a_c > b_c).sum()),
                "culture_tie_count": int((a_c == b_c).sum()),
                "round_win_count_a": int((a_r < b_r).sum()),
                "round_loss_count_a": int((a_r > b_r).sum()),
                "round_tie_count": int((a_r == b_r).sum()),
                "mean_best_utility_delta_a_minus_b": float(np.mean(quality_delta)),
                "mean_best_utility_delta_ci_low": bootstrap_ci(quality_delta, np.mean)[0],
                "mean_best_utility_delta_ci_high": bootstrap_ci(quality_delta, np.mean)[1],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"simulated_dbtl_paired_effects_{tier}.csv", index=False)
    return out


def target_curves(best: pd.DataFrame, rounds: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> tuple[pd.DataFrame, pd.DataFrame]:
    culture_rows = []
    for (wid, label, x), sub in best.groupby(["workflow_id", "workflow_label", "physical_culture_index"]):
        vals = sub["target_reached_so_far"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(vals, np.mean)
        culture_rows.append({"target_tier": tier, "workflow_id": wid, "workflow_label": label, "physical_culture_index": int(x), "target_attainment_probability": float(vals.mean()), "ci_low": lo, "ci_high": hi})
    round_rows = []
    for (wid, label, x), sub in rounds.groupby(["workflow_id", "workflow_label", "round_index"]):
        vals = sub["target_reached_by_round"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(vals, np.mean)
        round_rows.append({"target_tier": tier, "workflow_id": wid, "workflow_label": label, "round_index": int(x), "target_attainment_probability": float(vals.mean()), "ci_low": lo, "ci_high": hi})
    culture = pd.DataFrame(culture_rows)
    round_df = pd.DataFrame(round_rows)
    culture.to_csv(DATA / f"simulated_dbtl_target_attainment_by_culture_{tier}.csv", index=False)
    round_df.to_csv(DATA / f"simulated_dbtl_target_attainment_by_round_{tier}.csv", index=False)
    return culture, round_df


def calendar_time(summary: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> pd.DataFrame:
    protocol = load_json(DATA / "simulated_dbtl_protocol_config.json")
    assumptions = protocol["calendar_time_assumptions"]
    days_per_round = sum(float(assumptions[k]) for k in ["strain_construction_days_per_round", "culture_days_per_round", "assay_days_per_round", "analysis_planning_days_per_round"])
    out = summary.copy()
    out["assumed_days_per_completed_physical_round"] = days_per_round
    out["measured_compute_seconds_workflow_if_not_cache_deduplicated"] = out["workflow_compute_seconds_if_not_cache_deduplicated"]
    out["assumed_calendar_days_to_target_censored"] = (out["rounds_to_target_censored"] + 1) * days_per_round
    out["calendar_note"] = "Scenario-based: construction, cultivation, assay, and planning times are assumptions; exact compute is measured separately."
    cols = [
        "target_tier",
        "world_id",
        "campaign_seed",
        "workflow_id",
        "target_attained",
        "rounds_to_target",
        "rounds_to_target_censored",
        "assumed_days_per_completed_physical_round",
        "assumed_calendar_days_to_target_censored",
        "measured_compute_seconds_workflow_if_not_cache_deduplicated",
        "calendar_note",
    ]
    out[cols].to_csv(DATA / f"simulated_dbtl_calendar_time_{tier}.csv", index=False)
    return out[cols]


def choose_case_studies(obs: pd.DataFrame, summary: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> pd.DataFrame:
    paired = summary[summary["workflow_id"].isin([CANONICAL_BIOSENSOR_WORKFLOW, "conventional_dbtl", "bayesian_optimization", "modular_hybrid_no_biosensors"])].copy()
    pivot = paired.pivot_table(index=["world_id", "campaign_seed"], columns="workflow_id", values=["cultures_to_target_censored", "best_utility"], aggfunc="first")
    rows = []

    def add_case(case_id: str, description: str, selector) -> None:
        candidates = []
        for idx, row in pivot.iterrows():
            try:
                score = selector(row)
            except Exception:
                score = np.nan
            if np.isfinite(score):
                candidates.append((score, idx))
        if not candidates:
            return
        _score, (world, seed) = sorted(candidates, reverse=True)[0]
        sub = obs[(obs["world_id"].eq(world)) & (obs["campaign_seed"].eq(seed))]
        best = sub.loc[sub.groupby("workflow_id")["utility"].idxmax()][
            [
                "world_id",
                "campaign_seed",
                "workflow_id",
                "reporter_condition",
                "round_index",
                "physical_culture_index",
                "candidate_id",
                "culture_task_id",
                "temperature",
                "pH",
                "DO",
                "edit_count",
                "final_product",
                "final_biomass",
                "yield_proxy",
                "productivity",
                "utility",
                "oxidative_susceptibility",
                "atp_burden",
                "pathway_capacity",
                "oxidative_reporter",
                "atp_reporter",
                "pathway_reporter",
                f"reaches_{tier}_target",
            ]
        ]
        for _, brow in best.iterrows():
            rows.append({"case_id": case_id, "case_description": description, **brow.to_dict()})

    add_case(
        "strong_hybrid_success",
        "Canonical biosensor/full-reporter hybrid reaches target faster than conventional DBTL when such a case exists.",
        lambda r: r[("cultures_to_target_censored", "conventional_dbtl")] - r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)],
    )
    add_case(
        "conventional_or_bo_success",
        "Conventional DBTL or Bayesian optimisation matches/beats the canonical hybrid.",
        lambda r: min(r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)] - r[("cultures_to_target_censored", "conventional_dbtl")], r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)] - r[("cultures_to_target_censored", "bayesian_optimization")]),
    )
    add_case(
        "hybrid_failure",
        "Canonical hybrid has poor target timing or is censored relative to another method.",
        lambda r: r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)] - min(r[("cultures_to_target_censored", "conventional_dbtl")], r[("cultures_to_target_censored", "bayesian_optimization")]),
    )
    add_case(
        "biosensor_help",
        "Full reporters improve over modular hybrid without biosensors.",
        lambda r: r[("cultures_to_target_censored", "modular_hybrid_no_biosensors")] - r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)],
    )
    add_case(
        "reporters_little_benefit",
        "Full reporters provide little or no target-timing benefit over no-biosensor modular hybrid.",
        lambda r: -abs(r[("cultures_to_target_censored", "modular_hybrid_no_biosensors")] - r[("cultures_to_target_censored", CANONICAL_BIOSENSOR_WORKFLOW)]),
    )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"simulated_dbtl_mechanistic_case_studies_{tier}.csv", index=False)
    return out


def savefig(name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.svg")
    plt.savefig(FIGURES / f"{name}.png", dpi=220)
    plt.close()


def plot_lines(df: pd.DataFrame, x: str, y: str, name: str, title: str, workflows: list[str] | None = None, ci: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9.5, 5.2))
    data = df.copy()
    if workflows:
        data = data[data["workflow_id"].isin(workflows)]
    for wid, sub in data.groupby("workflow_id"):
        sub = sub.sort_values(x)
        color = PLOT_COLORS.get(wid)
        plt.plot(sub[x], sub[y], marker="o", linewidth=2, label=workflow_label(wid), color=color)
        if ci and {"ci_low", "ci_high"}.issubset(sub.columns):
            plt.fill_between(sub[x].to_numpy(float), sub["ci_low"].to_numpy(float), sub["ci_high"].to_numpy(float), alpha=0.16, color=color)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.25)
    savefig(name)


def plot_bar(df: pd.DataFrame, x: str, y: str, name: str, title: str, workflows: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = df.copy()
    if workflows:
        data = data[data["workflow_id"].isin(workflows)]
    data = data.sort_values(y)
    plt.figure(figsize=(10, 5.4))
    colors = [PLOT_COLORS.get(w, "#777777") for w in data["workflow_id"]]
    plt.bar(data[x], data[y], color=colors)
    plt.xticks(rotation=40, ha="right")
    plt.title(title)
    plt.ylabel(y.replace("_", " ").title())
    plt.grid(axis="y", alpha=0.25)
    savefig(name)


def write_figures(
    method: pd.DataFrame,
    world: pd.DataFrame,
    best: pd.DataFrame,
    rounds: pd.DataFrame,
    culture_curve: pd.DataFrame,
    round_curve: pd.DataFrame,
    paired: pd.DataFrame,
    calendar: pd.DataFrame,
    cases: pd.DataFrame,
    tier: str = PRIMARY_TARGET_TIER,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = [
        "conventional_dbtl",
        "bayesian_optimization",
        "black_box_digital_twin",
        "modular_hybrid_no_biosensors",
        CANONICAL_BIOSENSOR_WORKFLOW,
    ]
    plot_lines(culture_curve, "physical_culture_index", "target_attainment_probability", f"simdbtl_02_target_attainment_vs_cultures_{tier}", "Target Attainment vs Physical Cultures", headline, ci=True)
    plot_lines(round_curve, "round_index", "target_attainment_probability", f"simdbtl_03_target_attainment_vs_rounds_{tier}", "Target Attainment vs DBTL Round", headline, ci=True)
    best_curve = best.groupby(["workflow_id", "workflow_label", "physical_culture_index"], as_index=False)["best_utility_so_far"].mean()
    plot_lines(best_curve, "physical_culture_index", "best_utility_so_far", f"simdbtl_04_best_utility_vs_cultures_{tier}", "Best Verified Utility vs Physical Cultures", headline)
    round_best = rounds.groupby(["workflow_id", "workflow_label", "round_index"], as_index=False)["best_utility_cumulative"].mean()
    plot_lines(round_best, "round_index", "best_utility_cumulative", f"simdbtl_05_best_utility_vs_round_{tier}", "Best Verified Utility vs DBTL Round", headline)
    plot_bar(method, "workflow_label", "median_cultures_to_target_censored", f"simdbtl_06_cultures_to_target_{tier}", "Cultures To Target, Censored")
    plot_bar(method, "workflow_label", "median_rounds_to_target_censored", f"simdbtl_07_rounds_to_target_{tier}", "Rounds To Target, Censored")
    for name, comps in [
        ("simdbtl_08_biosensor_vs_no_biosensor", ["hybrid_full_vs_modular_no_biosensors"]),
        ("simdbtl_09_hybrid_vs_black_box", ["hybrid_full_vs_black_box"]),
        ("simdbtl_10_hybrid_vs_bo", ["hybrid_full_vs_bo"]),
        ("simdbtl_11_hybrid_vs_conventional", ["hybrid_full_vs_conventional"]),
    ]:
        sub = paired[paired["comparison"].isin(comps)]
        if not sub.empty:
            plt.figure(figsize=(7.5, 4.5))
            plt.bar(["cultures saved", "rounds saved", "utility delta"], [float(sub["mean_cultures_saved_by_a"].iloc[0]), float(sub["mean_rounds_saved_by_a"].iloc[0]), float(sub["mean_best_utility_delta_a_minus_b"].iloc[0])], color=["#4c78a8", "#f58518", "#54a24b"])
            plt.title(sub["comparison"].iloc[0])
            plt.grid(axis="y", alpha=0.25)
            savefig(f"{name}_{tier}")
    plt.figure(figsize=(11, 5.8))
    sub = world[world["workflow_id"].isin(headline)].copy()
    labels = sub["world_id"] + "\n" + sub["workflow_id"].map(workflow_label)
    plt.bar(labels, sub["median_cultures_to_target_censored"], color=[PLOT_COLORS.get(w, "#777777") for w in sub["workflow_id"]])
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.title("Performance By Biological World")
    plt.ylabel("Median cultures to target, censored")
    plt.grid(axis="y", alpha=0.25)
    savefig(f"simdbtl_12_performance_by_world_{tier}")
    reporter = method[method["method_id"].eq("biosensor_informed_modular_hybrid")]
    plot_bar(reporter, "reporter_condition", "median_cultures_to_target_censored", f"simdbtl_13_reporter_ablation_{tier}", "Reporter-Condition Ablation")
    cal = calendar.groupby("workflow_id", as_index=False)["assumed_calendar_days_to_target_censored"].median()
    cal["workflow_label"] = cal["workflow_id"].map(workflow_label)
    plot_bar(cal, "workflow_label", "assumed_calendar_days_to_target_censored", f"simdbtl_14_calendar_time_to_target_{tier}", "Scenario Calendar Time To Target")
    leverage = method.copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(leverage["mean_total_virtual_evaluations"], leverage["median_cultures_to_target_censored"], s=80, c=[PLOT_COLORS.get(w, "#777777") for w in leverage["workflow_id"]])
    for _, row in leverage.iterrows():
        plt.text(row["mean_total_virtual_evaluations"], row["median_cultures_to_target_censored"], workflow_label(row["workflow_id"]), fontsize=7)
    plt.xlabel("Mean virtual evaluations")
    plt.ylabel("Median cultures to target, censored")
    plt.title("Physical Cultures Saved vs Virtual Evaluations")
    plt.grid(alpha=0.25)
    savefig(f"simdbtl_15_cultures_vs_virtual_evaluations_{tier}")
    if not cases.empty:
        case_plot = cases.groupby(["case_id", "workflow_id"], as_index=False)["utility"].max()
        plt.figure(figsize=(10, 5))
        labels = case_plot["case_id"] + "\n" + case_plot["workflow_id"].map(workflow_label)
        plt.bar(labels, case_plot["utility"], color=[PLOT_COLORS.get(w, "#777777") for w in case_plot["workflow_id"]])
        plt.xticks(rotation=75, ha="right", fontsize=7)
        plt.ylabel("Utility")
        plt.title("Representative Mechanistic Case Study Utilities")
        plt.grid(axis="y", alpha=0.25)
        savefig(f"simdbtl_16_mechanistic_case_study_{tier}")
    # Schematic as a simple generated figure to keep the figure set complete.
    plt.figure(figsize=(10, 3.8))
    boxes = ["Frozen workflows", "Exact Yeast9 wet lab", "Visible observations", "DBTL trajectories", "Right-censored endpoints"]
    for i, text in enumerate(boxes):
        plt.gca().add_patch(plt.Rectangle((i * 2.0, 0.8), 1.55, 0.9, fill=False, linewidth=2, edgecolor="#2f4b7c"))
        plt.text(i * 2.0 + 0.775, 1.25, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            plt.arrow(i * 2.0 + 1.58, 1.25, 0.32, 0, width=0.02, head_width=0.14, head_length=0.12, color="#555555")
    plt.xlim(-0.2, 9.6)
    plt.ylim(0.4, 2.1)
    plt.axis("off")
    plt.title("End-to-End Simulated DBTL Benchmark")
    savefig(f"simdbtl_01_benchmark_schematic_{tier}")
    return sorted(str(p) for p in FIGURES.glob(f"simdbtl_*_{tier}.svg"))


def final_claim(method: pd.DataFrame, paired: pd.DataFrame, tier: str = PRIMARY_TARGET_TIER) -> pd.DataFrame:
    def comp(name: str) -> pd.Series:
        sub = paired[paired["comparison"].eq(name)]
        return sub.iloc[0] if len(sub) else pd.Series(dtype=object)

    hybrid = method[method["workflow_id"].eq(CANONICAL_BIOSENSOR_WORKFLOW)]
    no_bio = method[method["workflow_id"].eq("modular_hybrid_no_biosensors")]
    black = method[method["workflow_id"].eq("black_box_digital_twin")]
    conventional = method[method["workflow_id"].eq("conventional_dbtl")]
    bo = method[method["workflow_id"].eq("bayesian_optimization")]
    rows = []
    if not hybrid.empty:
        h = hybrid.iloc[0]
        conv = comp("hybrid_full_vs_conventional")
        bo_comp = comp("hybrid_full_vs_bo")
        mod = comp("hybrid_full_vs_modular_no_biosensors")
        bb = comp("hybrid_full_vs_black_box")
        faster_than_baselines = (
            float(conv.get("mean_cultures_saved_by_a", -np.inf)) > 0
            and float(bo_comp.get("mean_cultures_saved_by_a", -np.inf)) > 0
        )
        rounds_than_baselines = (
            float(conv.get("mean_rounds_saved_by_a", -np.inf)) > 0
            and float(bo_comp.get("mean_rounds_saved_by_a", -np.inf)) > 0
        )
        quality_ok = (
            float(conv.get("mean_best_utility_delta_a_minus_b", -np.inf)) >= -0.05
            and float(bo_comp.get("mean_best_utility_delta_a_minus_b", -np.inf)) >= -0.05
        )
        biosensors_help = float(mod.get("mean_cultures_saved_by_a", -np.inf)) > 0 or float(mod.get("mean_rounds_saved_by_a", -np.inf)) > 0
        modularity_help = float(bb.get("mean_cultures_saved_by_a", -np.inf)) > 0 or (
            not no_bio.empty
            and not black.empty
            and float(no_bio.iloc[0]["median_cultures_to_target_censored"]) < float(black.iloc[0]["median_cultures_to_target_censored"])
        )
        if faster_than_baselines and rounds_than_baselines and quality_ok and biosensors_help and modularity_help:
            outcome = "STRONG_POSITIVE"
        elif faster_than_baselines and quality_ok:
            outcome = "BOUNDED_POSITIVE"
        elif faster_than_baselines and not biosensors_help:
            outcome = "DIGITAL_TWIN_POSITIVE_BIOSENSOR_NEUTRAL"
        elif faster_than_baselines and not modularity_help:
            outcome = "MODULARITY_NEUTRAL"
        elif not faster_than_baselines and quality_ok:
            outcome = "NEUTRAL"
        else:
            outcome = "NEGATIVE_OR_TRADEOFF"
        rows.append(
            {
                "target_tier": tier,
                "claim_outcome": outcome,
                "canonical_hybrid_workflow": CANONICAL_BIOSENSOR_WORKFLOW,
                "hybrid_target_attainment_probability": float(h["target_attainment_probability"]),
                "hybrid_median_cultures_to_target_censored": float(h["median_cultures_to_target_censored"]),
                "hybrid_median_rounds_to_target_censored": float(h["median_rounds_to_target_censored"]),
                "hybrid_vs_conventional_mean_cultures_saved": float(conv.get("mean_cultures_saved_by_a", np.nan)),
                "hybrid_vs_bo_mean_cultures_saved": float(bo_comp.get("mean_cultures_saved_by_a", np.nan)),
                "hybrid_vs_conventional_mean_rounds_saved": float(conv.get("mean_rounds_saved_by_a", np.nan)),
                "hybrid_vs_bo_mean_rounds_saved": float(bo_comp.get("mean_rounds_saved_by_a", np.nan)),
                "hybrid_vs_conventional_quality_delta": float(conv.get("mean_best_utility_delta_a_minus_b", np.nan)),
                "hybrid_vs_bo_quality_delta": float(bo_comp.get("mean_best_utility_delta_a_minus_b", np.nan)),
                "biosensor_contribution_cultures_saved_vs_no_biosensors": float(mod.get("mean_cultures_saved_by_a", np.nan)),
                "modularity_contribution_cultures_saved_vs_black_box": float(bb.get("mean_cultures_saved_by_a", np.nan)),
                "conventional_target_attainment_probability": float(conventional.iloc[0]["target_attainment_probability"]) if not conventional.empty else np.nan,
                "bo_target_attainment_probability": float(bo.iloc[0]["target_attainment_probability"]) if not bo.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"simulated_dbtl_final_claim_audit_{tier}.csv", index=False)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|"))
    header = "| " + " | ".join(text.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    completion: dict[str, Any],
    method: pd.DataFrame,
    world: pd.DataFrame,
    paired: pd.DataFrame,
    claim: pd.DataFrame,
    figures: list[str],
    tier: str = PRIMARY_TARGET_TIER,
) -> None:
    report = DATA / "simulated_dbtl_final_digital_proof.md"
    lines = [
        "# Final Digital Proof: Simulated DBTL Acceleration",
        "",
        f"Protocol hash: `{EXPECTED_PROTOCOL_HASH}`.",
        f"Primary target tier analysed: `{tier}`. The hard tier is also generated when `analyze --tier hard` is run.",
        "",
        "## Exact HPC Completion Audit",
        "",
        markdown_table(pd.DataFrame([completion])),
        "",
        "## Method Summary",
        "",
        markdown_table(method[[
            "workflow_id",
            "target_attainment_probability",
            "median_cultures_to_target_censored",
            "median_rounds_to_target_censored",
            "censored_campaign_count",
            "mean_best_utility",
            "mean_best_final_product",
            "mean_dbtl_auc",
        ]]),
        "",
        "## Paired Effects",
        "",
        markdown_table(paired[[
            "comparison",
            "n_pairs",
            "mean_cultures_saved_by_a",
            "mean_percent_culture_reduction_by_a",
            "mean_rounds_saved_by_a",
            "mean_percent_round_reduction_by_a",
            "mean_best_utility_delta_a_minus_b",
        ]]) if not paired.empty else "No paired effects available.",
        "",
        "## Biological Worlds",
        "",
        markdown_table(world[[
            "world_id",
            "workflow_id",
            "target_attainment_probability",
            "median_cultures_to_target_censored",
            "median_rounds_to_target_censored",
            "mean_best_utility",
        ]]),
        "",
        "## Final Claim Audit",
        "",
        markdown_table(claim) if not claim.empty else "No claim audit available.",
        "",
        "## Final Digital Proof",
        "",
        "- What was simulated: frozen sequential DBTL workflows interacting with exact Yeast9 dynamic-pFBA hidden wet-lab rollouts under six biological worlds and ten campaign seeds.",
        "- What counted as a physical experiment: each workflow-requested strain by environment culture, even if exact-compute cache reuse avoided recomputing a duplicate culture.",
        "- Methods compared: conventional DBTL, Bayesian optimisation, black-box digital twin, modular hybrid without biosensors, and biosensor-informed modular hybrid across frozen reporter conditions.",
        f"- Physical culture budget: `{MAX_CULTURES}` per workflow campaign.",
        "- Cultures to target and rounds to target: reported above with right-censored unreached campaigns.",
        "- Percent experimental and round reduction: reported in the paired-effects table relative to conventional DBTL, Bayesian optimisation, black-box digital twin, and no-biosensor modular hybrid.",
        "- Quality tradeoff/noninferiority: reported as paired best-utility deltas and fixed-budget best product, yield, productivity, biomass, and utility tables.",
        "- Biosensor contribution: isolated by paired full/noisy/sparse/single-reporter hybrid comparisons against modular hybrid without biosensors.",
        "- Modularity contribution: isolated by paired modular-hybrid comparisons against the black-box digital twin.",
        "- Limitations: this is a computer-only simulated wet-lab benchmark; calendar time is scenario-based; exact pFBA remains a model of yeast physiology, not a replacement for physical validation.",
        "",
        "## Figures",
        "",
        *[f"- `{fig}`" for fig in figures],
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def analyze(rollouts: Path = ROLLOUTS, tier: str = PRIMARY_TARGET_TIER) -> dict[str, Any]:
    ensure_dirs()
    completion = completion_audit(rollouts=rollouts)
    if not completion["all_valid_complete"]:
        raise RuntimeError(f"Cannot analyse before exact audit passes: {completion}")
    obs = reconstruct_observations(rollouts=rollouts)
    best, rounds = trajectory_tables(obs, tier=tier)
    summary = campaign_summary(obs, best, tier=tier)
    method, world = aggregate_summary(summary, tier=tier)
    paired = paired_effects(summary, tier=tier)
    culture_curve, round_curve = target_curves(best, rounds, tier=tier)
    cal = calendar_time(summary, tier=tier)
    cases = choose_case_studies(obs, summary, tier=tier)
    claim = final_claim(method, paired, tier=tier)
    figures = write_figures(method, world, best, rounds, culture_curve, round_curve, paired, cal, cases, tier=tier)
    write_report(completion, method, world, paired, claim, figures, tier=tier)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("completion-audit")
    audit.add_argument("--rollouts", default=str(ROLLOUTS))
    ana = sub.add_parser("analyze")
    ana.add_argument("--rollouts", default=str(ROLLOUTS))
    ana.add_argument("--tier", choices=["moderate", "hard"], default=PRIMARY_TARGET_TIER)
    args = parser.parse_args()
    if args.command == "completion-audit":
        summary = completion_audit(Path(args.rollouts))
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "analyze":
        summary = analyze(Path(args.rollouts), tier=args.tier)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
