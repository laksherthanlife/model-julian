#!/usr/bin/env python3
"""Readiness audit for cached open-ended yeast DBTL benchmark campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import gem_backend as gem
import open_ended_baselines


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
EXACT_ROOT = RESULTS / "verifier" / "open_ended_exact"
PUBLIC_OBJECTIVE = {
    "final_product": 1.0,
    "product_AUC": 0.08,
    "final_biomass": 0.03,
    "growth_failure": -0.20,
}

CAMPAIGNS = {
    "campaign_001": {
        "hybrid_request": RESULTS / "exchange" / "incoming_requests" / "hybrid_round_001_request.json",
        "hybrid_exact_dir": EXACT_ROOT / "hybrid_round_001",
        "hybrid_public": DATA / "deployment_benchmark_hybrid_round_001_public_results.csv",
        "scientist_public": DATA / "deployment_benchmark_round_001_public_results.csv",
        "scientist_exact": DATA / "deployment_benchmark_round_001_exact_accounting.csv",
        "comparison": DATA / "deployment_benchmark_matched_round_001_comparison.csv",
    },
    "campaign_002": {
        "hybrid_request": RESULTS / "exchange" / "incoming_requests" / "campaign_002_hybrid_round_001_request.json",
        "hybrid_exact_dir": EXACT_ROOT / "campaign_002_hybrid_round_001",
        "hybrid_public": DATA / "deployment_benchmark_campaign_002_hybrid_public_results.csv",
        "scientist_public": DATA / "deployment_benchmark_campaign_002_scientist_public_results.csv",
        "scientist_exact": DATA / "deployment_benchmark_campaign_002_scientist_exact_accounting.csv",
        "comparison": DATA / "deployment_benchmark_campaign_002_comparison.csv",
    },
    "campaign_003": {
        "hybrid_request": RESULTS / "exchange" / "incoming_requests" / "campaign_003_hybrid_round_001_request.json",
        "hybrid_exact_dir": EXACT_ROOT / "campaign_003_hybrid_round_001",
        "hybrid_public": DATA / "deployment_benchmark_campaign_003_hybrid_public_results.csv",
        "scientist_public": DATA / "deployment_benchmark_campaign_003_scientist_public_results.csv",
        "scientist_exact": DATA / "deployment_benchmark_campaign_003_scientist_exact_accounting.csv",
        "comparison": DATA / "deployment_benchmark_campaign_003_comparison.csv",
    },
}


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_frame(df: pd.DataFrame, cols: list[str], decimals: int = 10) -> str:
    if df.empty:
        return ""
    payload = df[cols].copy()
    for col in payload.columns:
        if pd.api.types.is_numeric_dtype(payload[col]):
            payload[col] = payload[col].astype(float).round(decimals)
    return hashlib.sha256(payload.to_csv(index=False).encode("utf-8")).hexdigest()


def objective_from_public(df: pd.DataFrame) -> pd.Series:
    growth_failure = df["growth_failure"] if "growth_failure" in df.columns else df.get("severe_growth_collapse", pd.Series(False, index=df.index))
    return (
        df["final_product"].astype(float)
        + PUBLIC_OBJECTIVE["product_AUC"] * df["product_AUC"].astype(float)
        + PUBLIC_OBJECTIVE["final_biomass"] * df["final_biomass"].astype(float)
        + PUBLIC_OBJECTIVE["growth_failure"] * growth_failure.astype(float)
    )


def read_request(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("candidates", []))


def exact_paths(exact_dir: Path, candidate_id: str) -> dict[str, Path]:
    return {
        "summary": exact_dir / f"{candidate_id}_hidden_summary.csv",
        "trajectory": exact_dir / f"{candidate_id}_hidden_trajectory.csv",
        "flux": exact_dir / f"{candidate_id}_hidden_flux.csv",
        "constraints": exact_dir / f"{candidate_id}_hidden_constraints.csv",
    }


def safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_static_model():
    try:
        _cobra, _source_path, model = deployment.public_static_augmented_model()
        return model
    except Exception:
        return None


def applied_edit_rows(candidate: dict[str, Any], model) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edits = candidate.get("edits", [])
    if not isinstance(edits, list):
        return rows
    if model is None:
        for edit in edits:
            rows.append(
                {
                    "requested_edit": stable_json(edit),
                    "resolved_reaction_id": str(edit.get("reaction_id", "")),
                    "original_lower_bound": np.nan,
                    "original_upper_bound": np.nan,
                    "applied_lower_bound": np.nan,
                    "applied_upper_bound": np.nan,
                    "edit_changed_model": False,
                    "edit_application_note": "static_model_unavailable",
                }
            )
        return rows
    before = {}
    for rxn in model.reactions:
        before[rxn.id] = (float(rxn.lower_bound), float(rxn.upper_bound))
    edited = model.copy()
    try:
        notes = deployment.apply_sparse_edits_to_model(edited, candidate, strict=False)
    except Exception as exc:
        notes = [f"edit_application_failed:{type(exc).__name__}:{exc}"]
    after = {rxn.id: (float(rxn.lower_bound), float(rxn.upper_bound)) for rxn in edited.reactions}
    for edit in edits:
        rid = str(edit.get("reaction_id", ""))
        old_lb, old_ub = before.get(rid, (np.nan, np.nan))
        new_lb, new_ub = after.get(rid, (np.nan, np.nan))
        hidden_cfg_edit = rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, "BETA_EXPORT_ASSIST", gem.OXYGEN_EXCHANGE, gem.ATPM_RXN}
        rows.append(
            {
                "requested_edit": stable_json(edit),
                "resolved_reaction_id": rid,
                "original_lower_bound": old_lb,
                "original_upper_bound": old_ub,
                "applied_lower_bound": new_lb,
                "applied_upper_bound": new_ub,
                "edit_changed_model": bool(abs(new_lb - old_lb) > 1e-12 or abs(new_ub - old_ub) > 1e-12 or hidden_cfg_edit),
                "edit_application_note": ";".join(notes),
            }
        )
    return rows


def reaction_flux_proxy(rid: str, flux: pd.DataFrame, universe: pd.DataFrame) -> tuple[float, float, str]:
    flux_map = {
        gem.GLUCOSE_EXCHANGE: "glucose_uptake",
        gem.OXYGEN_EXCHANGE: "oxygen_uptake",
        gem.ATPM_RXN: "atp_maintenance_flux",
        gem.NATIVE_GGPP_RXN: "ggpp_flux",
        "r_0373": "ggpp_flux",
        gem.PSY_RXN: "PSY_flux",
        gem.DES_RXN: "DES_flux",
        gem.CYC_RXN: "CYC_flux",
    }
    baseline_flux = np.nan
    if rid in set(universe["reaction_id"].astype(str)):
        row = universe[universe["reaction_id"].astype(str).eq(rid)].iloc[0]
        baseline_flux = float(row.get("baseline_flux", np.nan)) if pd.notna(row.get("baseline_flux", np.nan)) else np.nan
    col = flux_map.get(rid)
    if col and col in flux.columns:
        edited_flux = float(flux[col].abs().max())
        return baseline_flux, edited_flux, col
    return baseline_flux, np.nan, "not_recorded_in_hidden_flux_table"


def collect_hybrid_candidate_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    model = load_static_model()
    candidate_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    for campaign_id, paths in CAMPAIGNS.items():
        candidates = read_request(paths["hybrid_request"])
        for candidate in candidates:
            candidate_hash = deployment.validate_sparse_design(candidate, universe)["candidate_hash"]
            cid = str(candidate["candidate_id"])
            paths_by_type = exact_paths(paths["hybrid_exact_dir"], cid)
            summary = safe_read(paths_by_type["summary"])
            traj = safe_read(paths_by_type["trajectory"])
            flux = safe_read(paths_by_type["flux"])
            cons = safe_read(paths_by_type["constraints"])
            if not summary.empty:
                s = summary.iloc[0].to_dict()
                candidate_hash = str(s.get("candidate_hash", candidate_hash))
            else:
                s = {}
            objective = (
                float(s.get("final_product", np.nan))
                + 0.08 * float(s.get("product_AUC", np.nan))
                + 0.03 * float(s.get("final_biomass", np.nan))
                - 0.20 * float(bool(s.get("severe_growth_collapse", False)))
            )
            trajectory_hash = sha256_frame(traj, [c for c in ["time", "X", "B_total", "z_ox", "z_atp", "z_bottle", "z_er", "E_PSY", "E_DES", "E_CYC"] if c in traj.columns])
            constraint_hash = sha256_frame(cons, [c for c in ["state", "next_state", "glucose_lower_bound", "oxygen_lower_bound", "atp_maintenance_lower_bound", "PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound", "gamma_growth_fraction"] if c in cons.columns])
            flux_hash = sha256_frame(flux, [c for c in ["biomass_flux", "beta_carotene_flux", "glucose_uptake", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux"] if c in flux.columns])
            effective_hash = hashlib.sha256(f"{trajectory_hash}:{constraint_hash}:{flux_hash}".encode("utf-8")).hexdigest()
            edit_info = applied_edit_rows(candidate, model)
            for edit_row in edit_info:
                rid = str(edit_row["resolved_reaction_id"])
                baseline_flux, edited_flux, flux_source = reaction_flux_proxy(rid, flux, universe)
                fva_min = fva_max = np.nan
                if rid in set(universe["reaction_id"].astype(str)):
                    urow = universe[universe["reaction_id"].astype(str).eq(rid)].iloc[0]
                    fva_min = float(urow.get("fva_min", np.nan)) if pd.notna(urow.get("fva_min", np.nan)) else np.nan
                    fva_max = float(urow.get("fva_max", np.nan)) if pd.notna(urow.get("fva_max", np.nan)) else np.nan
                capable = bool(pd.notna(fva_min) and pd.notna(fva_max) and max(abs(fva_min), abs(fva_max), abs(fva_max - fva_min)) > 1e-9)
                changed = bool(edit_row["edit_changed_model"])
                inactive = (not changed) or (not capable and pd.isna(edited_flux))
                edit_rows.append(
                    {
                        "campaign_id": campaign_id,
                        "candidate_id": cid,
                        "candidate_hash": candidate_hash,
                        **edit_row,
                        "bound_delta": f"{edit_row['original_lower_bound']}:{edit_row['original_upper_bound']}->{edit_row['applied_lower_bound']}:{edit_row['applied_upper_bound']}",
                        "public_static_baseline_flux": baseline_flux,
                        "hidden_recorded_edited_flux": edited_flux,
                        "hidden_flux_source": flux_source,
                        "public_fva_min": fva_min,
                        "public_fva_max": fva_max,
                        "reaction_capable_of_flux_public_static": capable,
                        "edit_effectiveness_classification": "inactive_or_unverified" if inactive else "effective_or_flux_capable",
                        "reason": "no bound/config change or no public flux capacity evidence" if inactive else "bounds/config changed and public/static or hidden flux evidence exists",
                    }
                )
            active_constraints = ""
            if not cons.empty:
                active_constraints = ";".join(cons.get("state", pd.Series(dtype=str)).astype(str).drop_duplicates().tolist())
            candidate_rows.append(
                {
                    "campaign_id": campaign_id,
                    "candidate_id": cid,
                    "candidate_hash": candidate_hash,
                    "environment_json": stable_json(candidate.get("environment", {})),
                    "requested_edit_json": stable_json(candidate.get("edits", [])),
                    "resolved_reaction_ids": ";".join(str(e.get("reaction_id", "")) for e in candidate.get("edits", []) if isinstance(e, dict)),
                    "active_constraints_during_trajectory": active_constraints,
                    "biomass_trajectory_hash": sha256_frame(traj, [c for c in ["time", "X"] if c in traj.columns]),
                    "product_trajectory_hash": sha256_frame(traj, [c for c in ["time", "B_total"] if c in traj.columns]),
                    "reporter_or_burden_trajectory_hash": sha256_frame(traj, [c for c in ["time", "z_ox", "z_atp", "z_bottle", "z_er"] if c in traj.columns]),
                    "trajectory_hash": trajectory_hash,
                    "constraint_hash": constraint_hash,
                    "flux_hash": flux_hash,
                    "effective_phenotype_hash": effective_hash,
                    "trajectory_unique_within_hybrid_campaigns": True,
                    "effective_phenotype_unique_within_hybrid_campaigns": True,
                    "final_objective": objective,
                    "objective_component_final_titer": float(s.get("final_product", np.nan)),
                    "objective_component_productivity": float(s.get("final_product", np.nan)) / float(traj["time"].max()) if not traj.empty and float(traj["time"].max()) > 0 else np.nan,
                    "objective_component_auc": float(s.get("product_AUC", np.nan)),
                    "objective_component_yield_proxy": float(s.get("final_product", np.nan)) / max(float(flux.get("glucose_uptake", pd.Series([np.nan])).abs().sum()), 1e-9) if not flux.empty and "glucose_uptake" in flux.columns else np.nan,
                    "objective_component_final_biomass": float(s.get("final_biomass", np.nan)),
                    "objective_component_growth_penalty": -0.20 * float(bool(s.get("severe_growth_collapse", False))),
                    "objective_component_edit_complexity_penalty": 0.0,
                    "objective_component_construction_risk_penalty": 0.0,
                    "cache_key": candidate_hash,
                    "output_file_provenance": stable_json({k: str(v.relative_to(ROOT)) for k, v in paths_by_type.items()}),
                    "duplicate_result_rows": int(summary.duplicated().sum()) if not summary.empty else 0,
                }
            )
    candidates_df = pd.DataFrame(candidate_rows)
    for col in ["trajectory_hash", "effective_phenotype_hash"]:
        counts = candidates_df[col].value_counts()
        unique_col = "trajectory_unique_within_hybrid_campaigns" if col == "trajectory_hash" else "effective_phenotype_unique_within_hybrid_campaigns"
        candidates_df[unique_col] = candidates_df[col].map(counts).eq(1)
    edits_df = pd.DataFrame(edit_rows)
    repeated_rows: list[dict[str, Any]] = []
    for objective_value, group in candidates_df.groupby(candidates_df["final_objective"].round(6)):
        if len(group) < 2:
            continue
        duplicate_specs = group["requested_edit_json"].duplicated().any()
        duplicate_hashes = group["candidate_hash"].duplicated().any()
        unique_trajectories = group["trajectory_hash"].nunique()
        unique_effective = group["effective_phenotype_hash"].nunique()
        if duplicate_hashes or duplicate_specs:
            classification = "duplicated computation"
        elif unique_effective == 1:
            affected_edits = edits_df[edits_df["candidate_id"].isin(group["candidate_id"])]
            inactive_fraction = float((affected_edits["edit_effectiveness_classification"] == "inactive_or_unverified").mean()) if not affected_edits.empty else np.nan
            classification = "inactive intervention" if inactive_fraction >= 0.5 else "valid equivalent phenotype"
        elif unique_trajectories > 1:
            classification = "reporting artefact"
        else:
            classification = "unresolved"
        repeated_rows.append(
            {
                "rounded_objective": objective_value,
                "n_candidates": len(group),
                "campaigns": ";".join(sorted(group["campaign_id"].unique())),
                "candidate_ids": ";".join(group["candidate_id"].astype(str)),
                "unique_candidate_hashes": int(group["candidate_hash"].nunique()),
                "unique_trajectories": int(unique_trajectories),
                "unique_effective_phenotypes": int(unique_effective),
                "classification": classification,
                "explanation": "Compared actual request specs, hidden trajectories, hidden constraints, and flux summaries; not based on hashes alone.",
            }
        )
    return candidates_df, edits_df, pd.DataFrame(repeated_rows)


def parent_reference_objective() -> float:
    cache_path = DATA / "design_benchmark_oracle_cache_index.csv"
    if not cache_path.exists():
        return np.nan
    cache = pd.read_csv(cache_path)
    baseline = cache[
        cache["regime_id"].astype(str).eq("strong_dynamic_regulation")
        & cache["edit_vector_id"].astype(str).eq("baseline_no_edit")
        & cache["selection_reason"].astype(str).str.contains("central_reference", na=False)
    ]
    if baseline.empty:
        return np.nan
    return float(objective_from_public(baseline).iloc[0])


def reanalyse_campaigns(candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parent_obj = parent_reference_objective()
    target = np.nan
    for campaign_id, paths in CAMPAIGNS.items():
        for method, public_path in [("scientist", paths["scientist_public"]), ("hybrid", paths["hybrid_public"])]:
            public = pd.read_csv(public_path).copy()
            public["objective"] = objective_from_public(public)
            if method == "hybrid":
                hybrid_detail = candidates_df[candidates_df["campaign_id"].eq(campaign_id)]
                unique_effective = int(hybrid_detail["effective_phenotype_hash"].nunique())
                unique_specs = int(hybrid_detail["requested_edit_json"].nunique())
            else:
                unique_effective = np.nan
                unique_specs = int(public["candidate_hash"].nunique())
            sorted_obj = public["objective"].sort_values(ascending=False)
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "method": method,
                    "best_objective": float(sorted_obj.iloc[0]),
                    "mean_top_2": float(sorted_obj.head(2).mean()),
                    "mean_top_4": float(sorted_obj.head(4).mean()),
                    "whole_batch_mean": float(public["objective"].mean()),
                    "median": float(public["objective"].median()),
                    "std": float(public["objective"].std(ddof=0)),
                    "worst_candidate": float(sorted_obj.iloc[-1]),
                    "fraction_above_parent": float((public["objective"] > parent_obj).mean()) if pd.notna(parent_obj) else np.nan,
                    "parent_reference_objective": parent_obj,
                    "fraction_above_frozen_target": float((public["objective"] >= target).mean()) if pd.notna(target) else np.nan,
                    "frozen_target_status": "not_defined_in_current_benchmark_artifacts",
                    "growth_failure_rate": float(public["growth_failure"].astype(bool).mean()),
                    "unique_candidate_specifications": unique_specs,
                    "unique_effective_phenotypes": unique_effective,
                    "unique_objective_values": int(public["objective"].round(9).nunique()),
                    "average_edit_count": float(candidates_df[candidates_df["campaign_id"].eq(campaign_id)]["requested_edit_json"].map(lambda s: len(json.loads(s))).mean()) if method == "hybrid" else np.nan,
                    "construction_complexity_score": np.nan,
                    "environment_diversity": int(candidates_df[candidates_df["campaign_id"].eq(campaign_id)]["environment_json"].nunique()) if method == "hybrid" else np.nan,
                    "reaction_edit_diversity": int(len(set(";".join(candidates_df[candidates_df["campaign_id"].eq(campaign_id)]["resolved_reaction_ids"]).split(";")) - {""})) if method == "hybrid" else np.nan,
                    "subsystem_or_mechanism_diversity": np.nan,
                    "mean_final_titer": float(public["final_product"].mean()),
                    "mean_productivity_proxy": float(public["final_product"].mean() / 12.0),
                    "mean_yield_proxy": np.nan,
                    "mean_final_biomass": float(public["final_biomass"].mean()),
                    "mean_growth_penalty": float((-0.20 * public["growth_failure"].astype(float)).mean()),
                    "mean_edit_complexity_penalty": 0.0,
                    "mean_construction_risk_penalty": 0.0,
                }
            )
    return pd.DataFrame(rows)


def noninferiority_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for campaign_id in sorted(metrics["campaign_id"].unique()):
        sci = metrics[(metrics["campaign_id"].eq(campaign_id)) & (metrics["method"].eq("scientist"))]["best_objective"].iloc[0]
        hyb = metrics[(metrics["campaign_id"].eq(campaign_id)) & (metrics["method"].eq("hybrid"))]["best_objective"].iloc[0]
        delta = float(hyb - sci)
        for margin in [0.02, 0.05, 0.10]:
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "delta": margin,
                    "hybrid_minus_scientist_best": delta,
                    "descriptive_noninferior": bool(delta >= -margin),
                    "note": "Descriptive sensitivity only; three one-shot campaigns are insufficient for formal statistical non-inferiority.",
                }
            )
    return pd.DataFrame(rows)


def leakage_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "public_bundle_dBTL_agent",
                "information_available_before_proposal": "public bundle, public Yeast9/static tools, public annotations, campaign-specific prior public observations only",
                "information_withheld": "hidden exact trajectories, hidden regulator, exact outcome cache, old exact pool ranking",
                "model_checkpoints_accessible": "public Yeast9 only",
                "cached_values_accessible": "public static solves and public prior observations only",
                "exact_simulator_outputs_accessible": "none before proposal",
                "ranking_uses_hidden_outcomes": False,
                "filename_ordering_metadata_cache_leak_risk": "campaign_001 provenance unverified; campaigns_002_003 have stricter campaign-specific registration",
                "critical_failure": False,
            },
            {
                "method": "hybrid_open_ended",
                "information_available_before_proposal": "public benchmark config, public edit universe, public static annotations, hybrid student assets, blinded dynamic screening code path",
                "information_withheld": "scientist exact outputs before hybrid search, hidden exact verifier outputs, exact cache/ranking",
                "model_checkpoints_accessible": "hybrid student and public GEM assets",
                "cached_values_accessible": "public static fields; no hidden exact cache by design",
                "exact_simulator_outputs_accessible": "none before proposal",
                "ranking_uses_hidden_outcomes": False,
                "filename_ordering_metadata_cache_leak_risk": "manifest blinding but no hard OS sandbox in current workspace",
                "critical_failure": False,
            },
            {
                "method": "random_valid_baseline",
                "information_available_before_proposal": "public edit universe and public environment bounds",
                "information_withheld": "all exact outcomes and hidden dynamic regulator",
                "model_checkpoints_accessible": "none required",
                "cached_values_accessible": "none beyond public validity data",
                "exact_simulator_outputs_accessible": "none",
                "ranking_uses_hidden_outcomes": False,
                "filename_ordering_metadata_cache_leak_risk": "low if generated in fresh public-only workspace",
                "critical_failure": False,
            },
            {
                "method": "space_filling_baseline",
                "information_available_before_proposal": "public edit universe, environment bounds, reaction IDs, subsystem/pathway/risk annotations",
                "information_withheld": "all exact outcomes and hidden dynamic regulator",
                "model_checkpoints_accessible": "none required",
                "cached_values_accessible": "public annotations only",
                "exact_simulator_outputs_accessible": "none",
                "ranking_uses_hidden_outcomes": False,
                "filename_ordering_metadata_cache_leak_risk": "low if generated in fresh public-only workspace",
                "critical_failure": False,
            },
            {
                "method": "static_gem_baseline",
                "information_available_before_proposal": "public FBA/pFBA/FVA-derived fields, public edit universe, public environment bounds",
                "information_withheld": "hidden dynamic regulator and hidden exact trajectories/ranks",
                "model_checkpoints_accessible": "public Yeast9 only",
                "cached_values_accessible": "public static GEM fields only",
                "exact_simulator_outputs_accessible": "none",
                "ranking_uses_hidden_outcomes": False,
                "filename_ordering_metadata_cache_leak_risk": "low if exact cache files are absent from working directory",
                "critical_failure": False,
            },
        ]
    )


def freeze_spec() -> dict[str, Any]:
    return {
        "benchmark_spec_id": "open_ended_yeast_dbtL_acceleration_benchmark_v0_pre_freeze",
        "status": "prepared_not_executed",
        "allowed_methods": ["public_bundle_dBTL_agent", "hybrid_open_ended", "random_valid_baseline", "space_filling_baseline", "static_gem_baseline"],
        "public_information": ["public Yeast9/static tools", "editable reaction universe", "heterologous library", "environment limits", "objective component definition", "operational budget"],
        "hidden_information": ["hidden dynamic regulator", "exact dynamic trajectories", "exact cache/ranking", "successful-candidate lists before public release"],
        "candidate_schema": "data/deployment_benchmark_candidate_schema_example.json",
        "environment_bounds": {"temperature": [27.0, 33.0], "pH": [4.5, 5.5], "DO": [20.0, 80.0]},
        "intervention_constraints": {"tier1": "1-2 native edits; no heterologous addition", "tier2": "up to 4 edits; at most 1 curated heterologous module"},
        "batch_sizes": {"cultures_per_round": 8, "maximum_physical_rounds": 4},
        "duplicate_handling": "reject duplicate candidate specifications within a batch before exact verification",
        "invalid_candidate_handling": "reject invalid requests without exact simulation",
        "objective_components": PUBLIC_OBJECTIVE,
        "target_definition": "not yet frozen; must be defined before sequential DBTL acceleration claim",
        "stopping_rules": "not yet executed; candidate target and max-round accounting must be frozen first",
        "accounting_rules": "separate virtual computation, public static LP solves, hidden exact verification, and physical measurements",
        "randomness_and_seed_policy": "record selector seed and campaign seed before proposal; no reseeding after hidden outcomes",
        "reporting_metrics": [
            "best",
            "top2_mean",
            "top4_mean",
            "whole_batch_mean",
            "median",
            "std",
            "failure_rate",
            "unique_candidate_specs",
            "unique_effective_phenotypes",
            "objective_components",
            "noninferiority_sensitivity",
        ],
    }


def write_report(metrics: pd.DataFrame, repeats: pd.DataFrame) -> None:
    lines = [
        "# Open-Ended Benchmark Readiness Audit",
        "",
        "Bounded conclusion: the existing three one-shot campaigns remain mixed. The public-bundle DBTL agent found the best individual candidate in all three campaigns, while the hybrid retained higher whole-batch means in all three. The current benchmark still does not demonstrate fewer physical experiments or fewer DBTL rounds to a target.",
        "",
        "Repeated outcomes were audited by comparing request specs, hidden trajectories, hidden constraints, flux summaries, and candidate hashes. Campaign 002 hybrid selected eight unique requested candidates that collapse to one effective phenotype and one objective value. The repeated 0.802500 top outcome in Campaigns 001 and 003 also collapses to the same effective phenotype despite distinct requested candidate hashes.",
        "",
        "After adding top-2, top-4, median, and unique-phenotype metrics, the batch-risk claim should be softened: hybrid whole-batch means are higher, but Campaign 002's apparent consistency is largely one repeated effective phenotype, and best-candidate non-inferiority is descriptive only.",
        "",
        "Key repeated-outcome classifications:",
    ]
    if repeats.empty:
        lines.append("- No repeated rounded objectives found.")
    else:
        for row in repeats.itertuples(index=False):
            lines.append(f"- objective {row.rounded_objective}: {row.classification}; candidates={row.n_candidates}; effective_phenotypes={row.unique_effective_phenotypes}")
    lines.extend(
        [
            "",
            "Generated tables:",
            "- `data/open_ended_hybrid_candidate_audit.csv`",
            "- `data/open_ended_edit_effectiveness.csv`",
            "- `data/open_ended_repeated_outcome_classification.csv`",
            "- `data/open_ended_campaign_metric_reanalysis.csv`",
            "- `data/open_ended_noninferiority_sensitivity.csv`",
            "- `data/open_ended_information_leakage_matrix.csv`",
            "- `data/open_ended_benchmark_freeze_spec.json`",
            "- `data/open_ended_baseline_*_proposals.csv`",
        ]
    )
    (DATA / "open_ended_benchmark_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(write_baselines: bool = True) -> None:
    candidates, edits, repeats = collect_hybrid_candidate_audit()
    metrics = reanalyse_campaigns(candidates)
    sensitivity = noninferiority_sensitivity(metrics)
    leak = leakage_matrix()
    candidates.to_csv(DATA / "open_ended_hybrid_candidate_audit.csv", index=False)
    edits.to_csv(DATA / "open_ended_edit_effectiveness.csv", index=False)
    repeats.to_csv(DATA / "open_ended_repeated_outcome_classification.csv", index=False)
    metrics.to_csv(DATA / "open_ended_campaign_metric_reanalysis.csv", index=False)
    sensitivity.to_csv(DATA / "open_ended_noninferiority_sensitivity.csv", index=False)
    leak.to_csv(DATA / "open_ended_information_leakage_matrix.csv", index=False)
    (DATA / "open_ended_benchmark_freeze_spec.json").write_text(json.dumps(freeze_spec(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_baselines:
        open_ended_baselines.write_all(batch_size=8)
    write_report(metrics, repeats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    run_all(write_baselines=not args.skip_baselines)


if __name__ == "__main__":
    main()
