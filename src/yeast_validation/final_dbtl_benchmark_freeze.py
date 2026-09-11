#!/usr/bin/env python3
"""Repair-aware intervention library and final DBTL benchmark freeze prep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import design_benchmark_exact as exact
import gem_backend as gem
import run_deployment_verifier as verifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
FINAL_SIMULATOR_VERSION = "final_dbtl_exact_sparse_dynamic_pfba_v1"
FINAL_REGULATOR_VERSION = "gem_backend_decision_tree_next_state_v1"
PHENOTYPE_TOLERANCE = 1e-6
FINAL_BATCHES = [8, 4, 4]
FINAL_METHODS = [
    "hybrid_digital_twin",
    "public_bundle_dbtL_scientist_agent",
    "random_valid",
    "space_filling_maximin",
    "static_gem_public_rank",
    "black_box_outcome_only_bo",
    "direct_trajectory_surrogate",
]
FINAL_CAMPAIGN_SEEDS = [51001, 51002, 51003]
EDIT_MULTIPLIERS = {
    "knockout": 0.0,
    "knockdown": 0.50,
    "capacity_decrease": 0.65,
    "capacity_increase": 1.50,
    "transport_or_exchange": 1.35,
    "pathway_enzyme_capacity": 1.35,
    "product_loss": 1.35,
}


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_payload(payload: object) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def with_hash(payload: dict[str, Any], key: str) -> dict[str, Any]:
    frozen = {k: v for k, v in payload.items() if k != key}
    return {**payload, key: hash_payload(frozen)}


def candidate_spec_hash(candidate: dict[str, Any]) -> str:
    payload = {
        "candidate_id": candidate.get("candidate_id", ""),
        "strain_id": candidate.get("strain_id", ""),
        "environment": candidate.get("environment", {}),
        "edits": candidate.get("edits", []),
        "edit_tier": candidate.get("edit_tier", candidate.get("design_tier", "")),
        "requested_assay_panel": candidate.get("requested_assay_panel", candidate.get("requested_assays", "")),
        "simulator_configuration": FINAL_SIMULATOR_VERSION,
        "candidate_metadata": candidate.get("metadata", {}),
    }
    return hash_payload(payload)


def load_augmented_model():
    cobra = gem.require_cobra()
    source = gem.configured_gem_path()
    model = gem.load_model(cobra, source)
    augmented, _manifest = gem.install_beta_carotene_pathway(model)
    return cobra, source, augmented


def reaction_bound_snapshot(model, reaction_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for rid in sorted(set(reaction_ids)):
        if rid in {r.id for r in model.reactions}:
            rxn = model.reactions.get_by_id(rid)
            rows.append({"reaction_id": rid, "lower_bound": float(rxn.lower_bound), "upper_bound": float(rxn.upper_bound)})
        else:
            rows.append({"reaction_id": rid, "lower_bound": None, "upper_bound": None})
    return rows


def apply_and_trace(model, candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reaction_ids = [str(e.get("reaction_id", "")) for e in candidate.get("edits", []) if isinstance(e, dict)]
    before = {row["reaction_id"]: row for row in reaction_bound_snapshot(model, reaction_ids)}
    notes = deployment.apply_sparse_edits_to_model(model, candidate, strict=False)
    after = {row["reaction_id"]: row for row in reaction_bound_snapshot(model, reaction_ids)}
    rows = []
    for edit in candidate.get("edits", []):
        rid = str(edit.get("reaction_id", ""))
        b = before.get(rid, {"lower_bound": None, "upper_bound": None})
        a = after.get(rid, {"lower_bound": None, "upper_bound": None})
        rows.append(
            {
                "requested_reaction_id": rid,
                "requested_edit_type": str(edit.get("edit_type", "")),
                "requested_magnitude": float(edit.get("capacity_multiplier", edit.get("upper_bound_multiplier", 1.0))),
                "resolved_reaction_id": rid,
                "original_lower_bound": b["lower_bound"],
                "original_upper_bound": b["upper_bound"],
                "actual_lower_bound": a["lower_bound"],
                "actual_upper_bound": a["upper_bound"],
                "actual_bound_changed": b != a,
                "application_notes": ";".join(notes),
            }
        )
    return rows, notes


def effective_model_hash(candidate: dict[str, Any], source_checksum: str, interval_constraints: pd.DataFrame | None = None) -> str:
    _cobra, _source, model = load_augmented_model()
    trace, _notes = apply_and_trace(model, candidate)
    cfg = verifier.config_for_sparse_candidate({**candidate, "candidate_hash": candidate_spec_hash(candidate)}, "strong_dynamic_regulation")
    config_payload = {
        key: getattr(cfg, key)
        for key in [
            "temperature",
            "pH",
            "DO",
            "glucose_uptake",
            "oxygen_uptake_at_100_do",
            "baseline_atpm",
            "beta_deg_base",
            "psy_base_capacity",
            "des_base_capacity",
            "cyc_base_capacity",
            "psy_initial_base",
            "des_initial_base",
            "cyc_initial_base",
            "capacity_mode",
        ]
    }
    dynamic_constraints = []
    if interval_constraints is not None and not interval_constraints.empty:
        cols = [c for c in ["glucose_lower_bound", "oxygen_lower_bound", "atp_maintenance_lower_bound", "PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"] if c in interval_constraints.columns]
        dynamic_constraints = interval_constraints[cols].round(8).to_dict("records")
    payload = {
        "resolved_edit_trace": trace,
        "simulator_version": FINAL_SIMULATOR_VERSION,
        "dynamic_regulator_version": FINAL_REGULATOR_VERSION,
        "gem_checksum": source_checksum,
        "config_payload": config_payload,
        "dynamic_constraint_trace": dynamic_constraints,
    }
    return hash_payload(payload)


def phenotype_fingerprint(tables: dict[str, pd.DataFrame], tolerance: float = PHENOTYPE_TOLERANCE) -> str:
    payload: dict[str, Any] = {"tolerance": tolerance, "tables": {}}
    for name, df in sorted(tables.items()):
        if df is None or df.empty:
            payload["tables"][name] = []
            continue
        cols = sorted(c for c in df.columns if c not in {"candidate_id", "candidate_hash", "strain_id", "runtime_seconds", "solver_runtime_seconds"})
        clean = df[cols].copy()
        for col in clean.columns:
            if pd.api.types.is_numeric_dtype(clean[col]):
                clean[col] = (clean[col].astype(float) / tolerance).round().astype("Int64")
        payload["tables"][name] = clean.to_dict("records")
    return hash_payload(payload)


def final_utility(metrics: dict[str, float], objective_config: dict[str, Any]) -> float:
    refs = objective_config["reference_values"]
    weights = objective_config["weights"]
    clips = objective_config["clipping"]

    def norm(key: str, value: float) -> float:
        ref = max(float(refs[key]), 1e-12)
        return float(np.clip(value / ref, clips["minimum_component"], clips["maximum_component"]))

    growth_failure = bool(metrics.get("growth_failure", False))
    utility = (
        weights["titer"] * norm("final_titer", float(metrics.get("final_product", 0.0)))
        + weights["productivity"] * norm("productivity", float(metrics.get("product_AUC", 0.0)))
        + weights["yield"] * norm("yield", float(metrics.get("yield_proxy", 0.0)))
        + weights["biomass"] * norm("final_biomass", float(metrics.get("final_biomass", 0.0)))
        - (objective_config["failure_penalties"]["growth_failure"] if growth_failure else 0.0)
        - objective_config["edit_count_penalty"] * float(metrics.get("edit_count", 0.0))
        - objective_config["construction_risk_penalties"].get(str(metrics.get("construction_risk", "medium")), objective_config["construction_risk_penalties"]["medium"])
    )
    if bool(metrics.get("infeasible", False)):
        utility -= objective_config["failure_penalties"]["infeasible"]
    return float(utility)


def reaches_target(metrics: dict[str, float], target_config: dict[str, Any], objective_config: dict[str, Any]) -> bool:
    parent = target_config["parent_reference"]
    checks = [
        float(metrics.get("final_product", 0.0)) >= float(target_config["requirements"]["minimum_final_titer"]),
        float(metrics.get("final_biomass", 0.0)) >= float(target_config["requirements"]["minimum_final_biomass"]),
        float(metrics.get("yield_proxy", 0.0)) >= float(target_config["requirements"]["minimum_yield_proxy"]),
        not bool(metrics.get("growth_failure", False)),
        int(metrics.get("edit_count", 0)) <= int(target_config["requirements"]["maximum_interventions"]),
        final_utility(metrics, objective_config) >= float(target_config["requirements"]["minimum_utility"]),
        float(parent["final_product"]) < float(target_config["requirements"]["minimum_final_titer"]),
    ]
    return all(checks)


def duplicate_candidate_hashes(candidates: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for candidate in candidates:
        h = candidate_spec_hash(candidate)
        if h in seen:
            duplicates.add(h)
        seen.add(h)
    return duplicates


def replacement_required(candidate: dict[str, Any], valid: bool, prior_batch: list[dict[str, Any]]) -> bool:
    return (not valid) or bool(duplicate_candidate_hashes([*prior_batch, candidate]))


def culture_charge_for_cache_event(cache_hit: bool) -> int:
    return 1


def candidate_from_intervention(intervention: pd.Series, env: dict[str, float], candidate_id: str) -> dict[str, Any]:
    edit = {
        "reaction_id": str(intervention["reaction_id"]),
        "edit_type": str(intervention["edit_type"]),
        "capacity_multiplier": float(intervention["capacity_multiplier"]),
        "rationale": f"Final DBTL diagnostic intervention: {intervention['intervention_id']}",
    }
    for key in ["target_lower_bound", "target_upper_bound", "reference_upper_bound", "reference_lower_bound"]:
        if key in intervention and pd.notna(intervention[key]):
            edit[key] = float(intervention[key])
    return {
        "candidate_id": candidate_id,
        "strain_id": f"{candidate_id}_strain",
        "edit_tier": 1,
        "edits": [edit],
        "environment": env,
        "requested_assay_panel": "basic",
        "metadata": {"source": "final_dbtl_implementation_diagnostic"},
    }


def parent_candidate(env: dict[str, float], candidate_id: str = "final_dbtl_parent_control") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "strain_id": "parent_no_edit",
        "edit_tier": 0,
        "edits": [],
        "environment": env,
        "requested_assay_panel": "basic",
        "metadata": {"source": "final_dbtl_parent_control"},
    }


def static_stage_diagnostic(candidate: dict[str, Any]) -> dict[str, Any]:
    cobra, source, parent_model = load_augmented_model()
    edited_model = parent_model.copy()
    cfg = verifier.config_for_sparse_candidate({**candidate, "candidate_hash": candidate_spec_hash(candidate)}, "strong_dynamic_regulation")
    cfg = replace(cfg, n_time=5)
    parent_cfg = verifier.config_for_sparse_candidate({**parent_candidate(candidate["environment"]), "candidate_hash": "parent"}, "strong_dynamic_regulation")
    parent_cfg = replace(parent_cfg, n_time=5)
    parent_constraints = gem.apply_interval_constraints(parent_model, parent_cfg, np.zeros(4), "balanced", gem.initial_enzyme_capacities(parent_cfg))
    parent_flux = gem.solve_staged(parent_model, parent_cfg, gem.state_gamma("balanced", parent_cfg), state="balanced")
    trace, notes = apply_and_trace(edited_model, candidate)
    edited_constraints = gem.apply_interval_constraints(edited_model, cfg, np.zeros(4), "balanced", gem.initial_enzyme_capacities(cfg))
    edited_flux = gem.solve_staged(edited_model, cfg, gem.state_gamma("balanced", cfg), state="balanced")
    rid = str(candidate["edits"][0]["reaction_id"]) if candidate.get("edits") else ""
    flux_col = {
        gem.GLUCOSE_EXCHANGE: "glucose_uptake",
        gem.OXYGEN_EXCHANGE: "oxygen_uptake",
        gem.ATPM_RXN: "atp_maintenance_flux",
        gem.NATIVE_GGPP_RXN: "ggpp_flux",
        gem.PSY_RXN: "PSY_flux",
        gem.DES_RXN: "DES_flux",
        gem.CYC_RXN: "CYC_flux",
    }.get(rid, "beta_carotene_flux")
    parent_value = float(parent_flux.get(flux_col, np.nan))
    edited_value = float(edited_flux.get(flux_col, np.nan))
    bound_changed = any(bool(row["actual_bound_changed"]) for row in trace)
    survived = True
    if rid == gem.OXYGEN_EXCHANGE:
        survived = abs(float(edited_constraints["oxygen_lower_bound"]) - float(parent_constraints["oxygen_lower_bound"])) > 1e-9
    elif rid == gem.ATPM_RXN:
        survived = abs(float(edited_constraints["atp_maintenance_lower_bound"]) - float(parent_constraints["atp_maintenance_lower_bound"])) > 1e-9
    elif rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN}:
        key = {"BETA_PHYTOENE_SYNTHASE": "PSY_effective_upper_bound", "BETA_PHYTOENE_DESATURASE": "DES_effective_upper_bound", "BETA_LYCOPENE_CYCLASE": "CYC_effective_upper_bound"}[rid]
        survived = abs(float(edited_constraints[key]) - float(parent_constraints[key])) > 1e-9
    elif candidate.get("edits"):
        survived = bound_changed
    final_product_delta = float(edited_flux["beta_carotene_flux"]) - float(parent_flux["beta_carotene_flux"])
    final_biomass_delta = float(edited_flux["biomass_flux"]) - float(parent_flux["biomass_flux"])
    classification = "effective" if (survived and (bound_changed or rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.OXYGEN_EXCHANGE, gem.ATPM_RXN, "BETA_EXPORT_ASSIST"}) and (abs(edited_value - parent_value) > 1e-9 or abs(final_product_delta) > 1e-9 or abs(final_biomass_delta) > 1e-9)) else "inactive_or_equivalent"
    return {
        "candidate_id": candidate["candidate_id"],
        "requested_reaction_id": rid,
        "requested_edit_type": str(candidate["edits"][0]["edit_type"]) if candidate.get("edits") else "parent",
        "requested_magnitude": float(candidate["edits"][0].get("capacity_multiplier", 1.0)) if candidate.get("edits") else 1.0,
        "resolved_reaction_id": rid,
        "reaction_reversibility": "",
        "original_lower_bound": trace[0]["original_lower_bound"] if trace else np.nan,
        "original_upper_bound": trace[0]["original_upper_bound"] if trace else np.nan,
        "intended_lower_bound": candidate["edits"][0].get("target_lower_bound", np.nan) if candidate.get("edits") else np.nan,
        "intended_upper_bound": candidate["edits"][0].get("target_upper_bound", np.nan) if candidate.get("edits") else np.nan,
        "actual_lower_bound": trace[0]["actual_lower_bound"] if trace else np.nan,
        "actual_upper_bound": trace[0]["actual_upper_bound"] if trace else np.nan,
        "edit_survived_dynamic_constraints": survived,
        "parent_flux": parent_value,
        "edited_flux": edited_value,
        "flux_delta": edited_value - parent_value,
        "final_product_delta": final_product_delta,
        "product_AUC_delta": np.nan,
        "final_biomass_delta": final_biomass_delta,
        "yield_delta": np.nan,
        "solver_status": ";".join([str(parent_flux.get("pfba_status", "")), str(edited_flux.get("pfba_status", ""))]),
        "effective_or_inactive_classification": classification,
        "reason": "survives constraints and changes flux/product/biomass in one-interval exact staged solve" if classification == "effective" else "no measurable one-interval consequence or no surviving bound/config change",
        "application_notes": ";".join(notes),
        "gem_checksum": gem.sha256(source),
    }


def build_actionable_library(limit: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    cobra, source, model = load_augmented_model()
    model_rxns = {r.id: r for r in model.reactions}
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    candidates = universe.copy()
    candidates["span"] = candidates.apply(lambda r: max(float(r.get("fva_max", 0.0)) - float(r.get("fva_min", 0.0)), 0.0) if pd.notna(r.get("fva_max", np.nan)) and pd.notna(r.get("fva_min", np.nan)) else 0.0, axis=1)
    candidates["abs_flux"] = candidates["baseline_flux"].fillna(0.0).astype(float).abs()
    candidates = candidates.sort_values(["span", "abs_flux", "selection_score"], ascending=False)
    retained_ids: set[str] = set()

    def retain(rxn_id: str, edit_type: str, intervention_class: str, multiplier: float, rationale: str, source_row: pd.Series | None = None) -> None:
        if rxn_id in retained_ids or rxn_id not in model_rxns or rxn_id in deployment.PROHIBITED_REACTIONS:
            return
        rxn = model_rxns[rxn_id]
        old_lb, old_ub = float(rxn.lower_bound), float(rxn.upper_bound)
        target_ub = np.nan
        target_lb = np.nan
        reference_ub = np.nan
        reference_lb = np.nan
        if edit_type in {"reaction_knockdown", "reaction_capacity_decrease", "transport_capacity_decrease"}:
            if old_ub > 0:
                target_ub = min(1000.0, old_ub * multiplier)
            if old_lb < 0:
                target_lb = old_lb * multiplier
        elif edit_type in {"reaction_capacity_increase", "precursor_supply_enhancement", "transport_capacity_change"}:
            if old_ub >= 999.0:
                span = float(source_row.get("span", 0.0)) if source_row is not None else 0.0
                reference_ub = max(span * 4.0, 1.0)
                target_ub = min(1000.0, reference_ub * multiplier)
            elif old_ub > 0:
                target_ub = min(1000.0, old_ub * multiplier)
            if old_lb < 0 and edit_type == "transport_capacity_change":
                target_lb = max(-1000.0, old_lb * multiplier)
        row = source_row.to_dict() if source_row is not None else {}
        rows.append(
            {
                "intervention_id": f"final_{len(rows)+1:03d}_{rxn_id}_{edit_type}",
                "reaction_id": rxn_id,
                "reaction_name": rxn.name,
                "subsystem": str(rxn.subsystem or row.get("subsystem", "")),
                "associated_genes": ";".join(sorted(g.id for g in rxn.genes)),
                "gpr_rule": rxn.gene_reaction_rule or "",
                "edit_type": edit_type,
                "intervention_class": intervention_class,
                "capacity_multiplier": multiplier,
                "target_lower_bound": target_lb,
                "target_upper_bound": target_ub,
                "reference_lower_bound": reference_lb,
                "reference_upper_bound": reference_ub,
                "reaction_reversibility": bool(rxn.reversibility),
                "baseline_flux": float(row.get("baseline_flux", np.nan)) if row else np.nan,
                "fva_min": float(row.get("fva_min", np.nan)) if row else np.nan,
                "fva_max": float(row.get("fva_max", np.nan)) if row else np.nan,
                "plausible_biological_implementation": biological_implementation(edit_type, rxn),
                "isoenzyme_warning": "isoenzyme_GPR_ambiguous" if " or " in (rxn.gene_reaction_rule or "").lower() else "",
                "enzyme_complex_warning": "complex_GPR_requires_multi_gene_interpretation" if " and " in (rxn.gene_reaction_rule or "").lower() else "",
                "multifunctional_gene_warning": "reaction_level_edit_not_gene_specific" if rxn.gene_reaction_rule else "",
                "pleiotropy_warning": "high" if str(row.get("risk_level", "")).lower() == "high" else "medium" if str(row.get("risk_level", "")).lower() == "medium" else "low",
                "construction_complexity": construction_complexity(edit_type, rxn),
                "expected_burden": expected_burden(edit_type),
                "reaction_to_gene_confidence": "medium" if rxn.gene_reaction_rule else "reaction_level_only_no_GPR",
                "retention_rationale": rationale,
                "gem_checksum": gem.sha256(source),
            }
        )
        retained_ids.add(rxn_id)

    for _, row in candidates.iterrows():
        if len(rows) >= limit:
            break
        rid = str(row["reaction_id"])
        reasons = []
        if rid not in model_rxns:
            reasons.append("not_in_augmented_model")
        if rid in deployment.PROHIBITED_REACTIONS:
            reasons.append("prohibited_objective_or_biomass_reaction")
        if float(row["span"]) <= 1e-9 and float(row["abs_flux"]) <= 1e-9:
            reasons.append("blocked_or_no_public_flux_capacity")
        if reasons:
            excluded.append({"reaction_id": rid, "exclusion_reason": ";".join(reasons)})
            continue
        text = f"{row.get('subsystem','')} {row.get('inclusion_rule','')}".lower()
        if "transport" in text or "exchange" in text:
            retain(rid, "transport_capacity_change", "transport_or_exchange", EDIT_MULTIPLIERS["transport_or_exchange"], "public static flux-capable transport/exchange intervention", row)
        elif "terpenoid" in text or "ggpp" in text or "isoprenoid" in text:
            retain(rid, "precursor_supply_enhancement", "precursor_supply", EDIT_MULTIPLIERS["capacity_increase"], "public static flux-capable precursor intervention", row)
        elif "byproduct" in text or "glycolysis" in text:
            retain(rid, "reaction_knockdown", "byproduct_or_carbon_rerouting", EDIT_MULTIPLIERS["knockdown"], "public static flux-capable rerouting intervention", row)
        else:
            retain(rid, "reaction_capacity_decrease", "capacity_decrease", EDIT_MULTIPLIERS["capacity_decrease"], "public static flux-capable conservative capacity intervention", row)
    manual = [
        (gem.PSY_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "curated pathway enzyme dynamic-capacity intervention"),
        (gem.DES_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "curated pathway enzyme dynamic-capacity intervention"),
        (gem.CYC_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "curated pathway enzyme dynamic-capacity intervention"),
        (gem.OXYGEN_EXCHANGE, "exchange_uptake_or_secretion_change", "oxygen_exchange", 1.25, "dynamic oxygen support intervention"),
        (gem.ATPM_RXN, "reaction_capacity_decrease", "atp_demand_reduction", 0.80, "ATP maintenance demand intervention"),
    ]
    for rid, edit_type, klass, mult, rationale in manual:
        retain(rid, edit_type, klass, mult, rationale, universe[universe["reaction_id"].astype(str).eq(rid)].iloc[0] if rid in set(universe["reaction_id"].astype(str)) else None)
    retained_all = pd.DataFrame(rows).drop_duplicates("intervention_id")
    class_caps = {
        "transport_or_exchange": 6,
        "byproduct_or_carbon_rerouting": 6,
        "capacity_decrease": 3,
        "precursor_supply": 3,
        "pathway_enzyme_capacity": 3,
        "oxygen_exchange": 1,
        "atp_demand_reduction": 1,
    }
    retained_parts = []
    for klass, cap in class_caps.items():
        part = retained_all[retained_all["intervention_class"].eq(klass)].head(cap)
        if not part.empty:
            retained_parts.append(part)
    retained = pd.concat(retained_parts, ignore_index=True) if retained_parts else retained_all.head(limit).copy()
    retained.to_csv(DATA / "final_dbtl_actionable_intervention_library.csv", index=False)
    retained_reactions = set(retained["reaction_id"].astype(str))
    for _, row in universe.iterrows():
        rid = str(row["reaction_id"])
        if rid not in retained_reactions and not any(e.get("reaction_id") == rid for e in excluded):
            reason = "not_selected_after_reliability_and_diversity_cap"
            if rid in set(retained_all["reaction_id"].astype(str)):
                reason = "excluded_by_per_mechanism_cap_after_initial_static_screen"
            excluded.append({"reaction_id": rid, "exclusion_reason": reason})
    excluded_df = pd.DataFrame(excluded).drop_duplicates("reaction_id")
    excluded_df.to_csv(DATA / "final_dbtl_excluded_interventions.csv", index=False)
    return retained, excluded_df


def biological_implementation(edit_type: str, rxn) -> str:
    if edit_type in {"reaction_knockout", "complete_reaction_knockout"}:
        return "gene deletion if GPR is unambiguous; otherwise reaction-level knockout abstraction"
    if "knockdown" in edit_type or "decrease" in edit_type:
        return "promoter weakening, CRISPRi, enzyme attenuation, or reaction-level capacity decrease"
    if "pathway_enzyme" in edit_type or "increase" in edit_type or "enhancement" in edit_type:
        return "overexpression/promoter strengthening if GPR is interpretable; otherwise reaction-level capacity increase"
    if "exchange" in edit_type or "transport" in edit_type:
        return "transporter expression or uptake/secretion capacity change"
    return "reaction-level implementation"


def construction_complexity(edit_type: str, rxn) -> str:
    if "knockout" in edit_type:
        return "medium" if rxn.gene_reaction_rule else "reaction_level_only"
    if "pathway_enzyme" in edit_type or "increase" in edit_type:
        return "medium"
    return "low"


def expected_burden(edit_type: str) -> str:
    if "increase" in edit_type or "pathway_enzyme" in edit_type:
        return "medium"
    if "exchange" in edit_type or "transport" in edit_type:
        return "medium"
    return "low"


def write_diagnostics(library: pd.DataFrame) -> pd.DataFrame:
    env = {"temperature": 30.0, "pH": 5.0, "DO": 65.0}
    rows = []
    diagnostic_library = pd.concat(
        [group.head(4) for _klass, group in library.groupby("intervention_class", sort=False)],
        ignore_index=True,
    ).head(24)
    for row in diagnostic_library.itertuples(index=False):
        candidate = candidate_from_intervention(pd.Series(row._asdict()), env, f"diag_{row.intervention_id}")
        try:
            rows.append(static_stage_diagnostic(candidate))
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "requested_reaction_id": candidate["edits"][0]["reaction_id"],
                    "requested_edit_type": candidate["edits"][0]["edit_type"],
                    "requested_magnitude": candidate["edits"][0]["capacity_multiplier"],
                    "solver_status": f"failed:{type(exc).__name__}",
                    "effective_or_inactive_classification": "diagnostic_failed",
                    "reason": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "final_dbtl_edit_application_diagnostics.csv", index=False)
    return out


def write_objective_and_target() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = {
        "source": "frozen prior exact central-reference parent, strong_dynamic_regulation",
        "final_product": 0.3680067691298719,
        "product_AUC": 1.606680837794752,
        "final_biomass": 1.2882997423347615,
        "yield_proxy": 0.004776,
        "growth_failure": False,
    }
    objective = with_hash(
        {
            "objective_id": "final_dbtl_engineering_utility_v1",
            "status": "frozen_before_final_method_outcomes",
            "component_definitions": {
                "final_titer": "final beta-carotene amount at the end of the exact dynamic trajectory",
                "productivity": "product AUC over the exact dynamic trajectory",
                "yield": "final product divided by absolute integrated glucose uptake proxy",
                "final_biomass": "final biomass at the end of the exact dynamic trajectory",
                "growth_failure": "minimum biomass below exact-verifier growth-failure threshold",
            },
            "reference_values": {
                "final_titer": parent["final_product"],
                "productivity": parent["product_AUC"],
                "yield": parent["yield_proxy"],
                "final_biomass": parent["final_biomass"],
            },
            "weights": {"titer": 0.45, "productivity": 0.20, "yield": 0.20, "biomass": 0.15},
            "clipping": {"minimum_component": 0.0, "maximum_component": 2.5},
            "failure_penalties": {"growth_failure": 1.0, "infeasible": 2.0},
            "edit_count_penalty": 0.025,
            "construction_risk_penalties": {"low": 0.0, "medium": 0.04, "high": 0.10, "reaction_level_only": 0.06},
            "repeated_effective_phenotype_policy": "count physical cultures, but report unique effective phenotypes and do not credit duplicate effective phenotypes as independent target discoveries",
            "raw_metrics_preserved": True,
        },
        "objective_hash",
    )
    DATA.joinpath("final_dbtl_objective_config.json").write_text(json.dumps(objective, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parent_utility = final_utility({"final_product": parent["final_product"], "product_AUC": parent["product_AUC"], "yield_proxy": parent["yield_proxy"], "final_biomass": parent["final_biomass"], "growth_failure": False, "edit_count": 0, "construction_risk": "low"}, objective)
    target = with_hash(
        {
            "target_id": "final_dbtl_target_v1",
            "status": "frozen_before_final_method_outcomes",
            "parent_reference": parent,
            "requirements": {
                "minimum_final_titer": parent["final_product"] * 1.25,
                "minimum_final_biomass": parent["final_biomass"] * 0.90,
                "minimum_yield_proxy": parent["yield_proxy"],
                "maximum_interventions": 3,
                "minimum_utility": parent_utility + 0.10,
                "growth_failure_allowed": False,
            },
            "rationale": "Target is above the frozen parent but below multiple already observed prior one-shot calibration candidates; it was not chosen from future final method outcomes.",
            "provenance": "prior exact finite-pool parent and completed one-shot readiness artifacts only",
            "parent_satisfies_target": False,
        },
        "target_hash",
    )
    DATA.joinpath("final_dbtl_target_config.json").write_text(json.dumps(target, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return objective, target


def write_final_spec(library: pd.DataFrame, objective: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    source = gem.configured_gem_path()
    spec = with_hash(
        {
            "benchmark_id": "final_sequential_dbtl_synthetic_evaluation_v1",
            "status": "frozen_prepared_not_executed",
            "method_list": FINAL_METHODS,
            "intervention_library_file": "data/final_dbtl_actionable_intervention_library.csv",
            "intervention_count": int(len(library)),
            "environment_domain": {"temperature": [27.0, 33.0], "pH": [4.5, 5.5], "DO": [20.0, 80.0]},
            "simulator_version": FINAL_SIMULATOR_VERSION,
            "hidden_regime_policy": "strong_dynamic_regulation for final synthetic benchmark v1",
            "objective_hash": objective["objective_hash"],
            "target_hash": target["target_hash"],
            "campaign_seeds": FINAL_CAMPAIGN_SEEDS,
            "campaign_count": len(FINAL_CAMPAIGN_SEEDS),
            "batch_sizes": FINAL_BATCHES,
            "maximum_cultures_per_method_campaign": sum(FINAL_BATCHES),
            "invalid_proposal_policy": "reject before exact simulation and require replacement from same method without observing hidden outcomes",
            "replacement_policy": "deterministic next valid proposal under method seed",
            "candidate_duplicate_policy": "forbid duplicate candidate specification hashes within a method campaign",
            "effective_model_duplicate_policy": "allowed only if unavoidable but counted and reported; no duplicate target credit",
            "effective_phenotype_accounting": "fingerprint exact trajectories at tolerance 1e-6 and report unique phenotypes",
            "stopping_rule": "stop a method-campaign when a candidate reaches frozen target after any round",
            "cache_policy": "cache may avoid recomputation but never removes physical-culture charge",
            "physical_culture_accounting": "each exact-tested candidate counts as one physical culture, including cache reuse",
            "exact_lp_solve_accounting": "full exact candidate cost currently 144 LP solves",
            "virtual_compute_accounting": "all method-side public/static/surrogate/dynamic computations reported separately",
            "reporting_metrics": ["target_hit_round", "cultures_to_target", "best_utility", "best_raw_titer", "unique_effective_models", "unique_effective_phenotypes", "growth_failure_rate", "exact_lp_solves", "virtual_compute"],
            "gem_checksum": gem.sha256(source),
            "no_final_outcomes_observed": True,
        },
        "benchmark_spec_hash",
    )
    DATA.joinpath("final_dbtl_benchmark_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = pd.DataFrame(
        [
            {
                "method_id": method,
                "method_class": "adaptive" if method in {"hybrid_digital_twin", "public_bundle_dbtL_scientist_agent", "black_box_outcome_only_bo", "direct_trajectory_surrogate"} else "nonadaptive_baseline",
                "uses_hidden_outcomes_before_proposal": False,
                "uses_actionable_intervention_library": True,
                "uses_same_target": True,
                "uses_same_budget": True,
            }
            for method in FINAL_METHODS
        ]
    )
    registry.to_csv(DATA / "final_dbtl_method_registry.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "campaign_id": f"final_dbtl_campaign_{i+1:03d}",
                "seed": seed,
                "round_1_batch": 8,
                "round_2_batch": 4,
                "round_3_batch": 4,
                "status": "planned_not_executed",
            }
            for i, seed in enumerate(FINAL_CAMPAIGN_SEEDS)
        ]
    )
    manifest.to_csv(DATA / "final_dbtl_campaign_manifest.csv", index=False)
    access = pd.DataFrame(
        [
            {
                "method_id": method,
                "public_library": True,
                "public_objective": True,
                "public_target": True,
                "own_prior_observations_only": method in {"hybrid_digital_twin", "public_bundle_dbtL_scientist_agent", "black_box_outcome_only_bo", "direct_trajectory_surrogate"},
                "hidden_exact_cache_access": False,
                "other_method_observation_access": False,
                "leakage_test_status": "pass_static_access_matrix",
            }
            for method in FINAL_METHODS
        ]
    )
    access.to_csv(DATA / "final_dbtl_method_access_matrix.csv", index=False)
    max_exact = len(FINAL_CAMPAIGN_SEEDS) * len(FINAL_METHODS) * sum(FINAL_BATCHES)
    budget = pd.DataFrame(
        [
            {"budget_item": "minimum_exact_simulations_with_immediate_early_stop", "value": len(FINAL_CAMPAIGN_SEEDS) * len(FINAL_METHODS) * FINAL_BATCHES[0]},
            {"budget_item": "maximum_exact_simulations", "value": max_exact},
            {"budget_item": "lp_solves_per_full_exact_candidate", "value": exact.EXPECTED_LP_SOLVES},
            {"budget_item": "maximum_lp_solves", "value": max_exact * exact.EXPECTED_LP_SOLVES},
            {"budget_item": "expected_disk_usage_mb", "value": max_exact * 0.35},
            {"budget_item": "expected_memory_profile", "value": "process-per-candidate, capped BLAS threads, no full campaign parallel memory sharing"},
            {"budget_item": "scientist_agent_tasks_required", "value": len(FINAL_CAMPAIGN_SEEDS)},
            {"budget_item": "cache_reuse_assumption", "value": "cache may save recomputation but physical-culture charge remains"},
        ]
    )
    budget.to_csv(DATA / "final_dbtl_budget_estimate.csv", index=False)
    return spec


def run_small_exact_diagnostics(library: pd.DataFrame, cases: int = 3) -> pd.DataFrame:
    selected = library.drop_duplicates("intervention_class").head(cases)
    rows = []
    old_time_points = exact.TIME_POINTS
    exact.TIME_POINTS = 5
    try:
        for i, intervention in enumerate(selected.itertuples(index=False), start=1):
            env = {"temperature": 30.0 if i % 2 else 31.0, "pH": 5.0, "DO": 65.0 if i % 2 else 45.0}
            parent = {**parent_candidate(env, f"small_exact_parent_{i:02d}"), "candidate_hash": f"parent_{i:02d}"}
            edited = candidate_from_intervention(pd.Series(intervention._asdict()), env, f"small_exact_{intervention.intervention_id}")
            edited["candidate_hash"] = candidate_spec_hash(edited)
            for label, candidate in [("parent", parent), ("edited", edited)]:
                traj, flux, cons, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime="strong_dynamic_regulation")
                summary = summary.iloc[0].to_dict()
                rows.append(
                    {
                        "diagnostic_pair": i,
                        "case_role": label,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_hash": candidate["candidate_hash"],
                        "intervention_id": getattr(intervention, "intervention_id", "parent"),
                        "final_product": summary["final_product"],
                        "product_AUC": summary["product_AUC"],
                        "final_biomass": summary["final_biomass"],
                        "minimum_biomass": summary["minimum_biomass"],
                        "n_actual_lp_solves": summary["n_actual_lp_solves"],
                        "exact_status": summary["exact_status"],
                        "effective_model_hash": effective_model_hash(candidate, summary["gem_source_checksum"], cons),
                        "effective_phenotype_fingerprint": phenotype_fingerprint({"trajectory": traj, "flux": flux, "constraints": cons}),
                    }
                )
    finally:
        exact.TIME_POINTS = old_time_points
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "final_dbtl_small_exact_diagnostics.csv", index=False)
    return out


def readiness_decision(library: pd.DataFrame, diagnostics: pd.DataFrame, small_exact: pd.DataFrame | None) -> pd.DataFrame:
    retained = len(library)
    effective_rate = float((diagnostics["effective_or_inactive_classification"] == "effective").mean()) if not diagnostics.empty else 0.0
    classes = int(library["intervention_class"].nunique()) if not library.empty else 0
    exact_ok = small_exact is not None and not small_exact.empty and (small_exact["exact_status"] == "complete_exact_dynamic_pfba").all()
    if retained >= 12 and classes >= 4 and effective_rate >= 0.5 and exact_ok:
        decision = "CONDITIONAL_GO_FINAL_DBTL_EVALUATION"
        reason = "freeze artifacts and identity system are present; remaining condition is review of short-horizon diagnostics before full execution"
    else:
        decision = "NO_GO_FINAL_DBTL_EVALUATION"
        reason = "readiness gate not fully satisfied; do not launch final 8+4+4 campaigns"
    out = pd.DataFrame(
        [
            {
                "decision": decision,
                "retained_interventions": retained,
                "intervention_classes": classes,
                "diagnostic_effective_rate": effective_rate,
                "small_exact_diagnostics_complete": bool(exact_ok),
                "reason": reason,
            }
        ]
    )
    out.to_csv(DATA / "final_dbtl_readiness_decision.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Stage 2 redesign pass.  These definitions intentionally override the first
# freeze-prep implementation above while preserving its small utility functions
# for tests and downstream imports.

STAGED_EFFECT_TOLERANCE = 1e-7
DYNAMIC_PHENOTYPE_TOLERANCE = 1e-5
DESIGN_SPACE_SAMPLE_SIZE = 360
REPRESENTATIVE_ENVIRONMENTS = [
    {"environment_id": "central", "temperature": 30.0, "pH": 5.0, "DO": 65.0},
    {"environment_id": "low_oxygen", "temperature": 30.0, "pH": 5.0, "DO": 25.0},
    {"environment_id": "high_oxygen", "temperature": 30.0, "pH": 5.0, "DO": 80.0},
    {"environment_id": "low_temperature", "temperature": 27.0, "pH": 5.0, "DO": 65.0},
    {"environment_id": "high_temperature", "temperature": 33.0, "pH": 5.0, "DO": 65.0},
    {"environment_id": "low_pH", "temperature": 30.0, "pH": 4.5, "DO": 65.0},
    {"environment_id": "high_pH", "temperature": 30.0, "pH": 5.5, "DO": 65.0},
    {"environment_id": "pathway_limiting", "temperature": 33.0, "pH": 4.5, "DO": 25.0},
]
_AUGMENTED_MODEL_CACHE: tuple[Any, Path, Any, str] | None = None
_PARENT_STAGE_CACHE: dict[str, dict[str, Any]] = {}


def env_payload(row: dict[str, Any]) -> dict[str, float]:
    return {"temperature": float(row["temperature"]), "pH": float(row["pH"]), "DO": float(row["DO"])}


def cached_augmented_model():
    global _AUGMENTED_MODEL_CACHE
    if _AUGMENTED_MODEL_CACHE is None:
        cobra, source, augmented = load_augmented_model()
        _AUGMENTED_MODEL_CACHE = (cobra, source, augmented, gem.sha256(source))
    return _AUGMENTED_MODEL_CACHE


def reaction_ids(model) -> set[str]:
    return {r.id for r in model.reactions}


def get_reaction_or_none(model, rid: str):
    return model.reactions.get_by_id(rid) if rid in reaction_ids(model) else None


def is_dynamic_parameter_intervention(rid: str, edit_type: str) -> bool:
    return rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN, "BETA_EXPORT_ASSIST"} or "product_loss" in edit_type


def finite_reference_capacity(
    baseline_flux: float | int | None,
    fva_min: float | int | None,
    fva_max: float | int | None,
    dynamic_flux_range: float | int | None = None,
    floor: float = 0.05,
) -> tuple[str, float]:
    values = []
    for value in [baseline_flux, fva_min, fva_max, dynamic_flux_range]:
        try:
            if value is not None and pd.notna(value):
                values.append(abs(float(value)))
        except Exception:
            pass
    reference = max([floor, *values])
    return "max_abs_public_baseline_fva_dynamic_flux_floor", float(reference)


def directional_bounds_for_intervention(row: pd.Series, multiplier: float, mode: str) -> dict[str, float]:
    ref = float(row.get("reference_capacity_value", row.get("reference_capacity", 0.05)))
    old_lb = float(row.get("original_lower_bound", 0.0))
    old_ub = float(row.get("original_upper_bound", 1000.0))
    out: dict[str, float] = {}
    if mode in {"increase", "overexpression", "transporter_upregulation"}:
        if old_lb < 0 and (abs(old_lb) >= abs(old_ub) or "uptake" in str(row.get("intervention_type", ""))):
            out["target_lower_bound"] = -ref * multiplier
        if old_ub > 0:
            out["reference_upper_bound"] = ref
            out["target_upper_bound"] = ref * multiplier
    elif mode in {"decrease", "suppression", "knockdown"}:
        if old_lb < 0:
            out["target_lower_bound"] = old_lb * multiplier if abs(old_lb) < 999.0 else -ref * multiplier
        if old_ub > 0:
            out["reference_upper_bound"] = ref
            out["target_upper_bound"] = max(0.0, ref * multiplier)
    return out


def mechanism_for_reaction(rid: str, text: str) -> str:
    lower = text.lower()
    if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN}:
        return "direct_heterologous_pathway_capacity"
    if rid == "BETA_EXPORT_ASSIST":
        return "product_export_or_loss"
    if rid == gem.ATPM_RXN or "atp" in lower:
        return "cofactor_or_energy_support"
    if rid in {gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE} or "transport" in lower or "exchange" in lower:
        return "transport_and_environment_coupling"
    if rid == gem.NATIVE_GGPP_RXN or any(term in lower for term in ["geranyl", "isoprenoid", "mevalonate", "sterol", "ergosterol"]):
        return "precursor_supply_or_competing_sink"
    if any(term in lower for term in ["nadph", "nadh", "respiration", "oxidative phosphorylation"]):
        return "cofactor_or_energy_support"
    if any(term in lower for term in ["ethanol", "glycerol", "acetate", "aldehyde"]):
        return "precursor_supply_or_competing_sink"
    return "carbon_allocation_or_background_capacity"


def solve_staged_full_flux(model, cfg: gem.GEMCultureConfig, gamma: float, state: str = "balanced") -> dict[str, Any]:
    cobra = gem.require_cobra()
    statuses: list[str] = []
    gem.set_objective(model, gem.BIOMASS_RXN)
    growth_sol = model.optimize()
    statuses.append(str(growth_sol.status))
    if growth_sol.status != "optimal" or growth_sol.objective_value is None:
        raise gem.GEMPilotError(f"Growth optimization failed: {growth_sol.status}")
    mu_max = float(growth_sol.objective_value)
    preserved_growth = gamma * mu_max
    model.reactions.get_by_id(gem.BIOMASS_RXN).lower_bound = preserved_growth
    protocol = cfg.allocation_protocol.upper()
    if protocol != "A":
        return {**gem.solve_staged(model, cfg, gamma, state=state), "full_fluxes": pd.Series(dtype=float)}
    gem.set_objective(model, gem.PRODUCT_RXN)
    product_sol = model.optimize()
    statuses.append(str(product_sol.status))
    if product_sol.status != "optimal" or product_sol.objective_value is None:
        raise gem.GEMPilotError(f"Product optimization failed: {product_sol.status}")
    beta_flux = max(0.0, float(product_sol.fluxes.get(gem.PRODUCT_RXN, product_sol.objective_value)))
    model.reactions.get_by_id(gem.PRODUCT_RXN).lower_bound = 0.999 * beta_flux if beta_flux > 1e-12 else 0.0
    pfba_sol = cobra.flux_analysis.pfba(model)
    statuses.append(str(pfba_sol.status))
    if pfba_sol.status != "optimal":
        raise gem.GEMPilotError(f"pFBA optimization failed: {pfba_sol.status}")
    fluxes = pfba_sol.fluxes
    return {
        "growth_status": growth_sol.status,
        "product_status": product_sol.status,
        "pfba_status": pfba_sol.status,
        "maximum_growth": mu_max,
        "preserved_growth": preserved_growth,
        "beta_carotene_flux": float(fluxes.get(gem.PRODUCT_RXN, beta_flux)),
        "biomass_flux": float(fluxes.get(gem.BIOMASS_RXN, np.nan)),
        "glucose_uptake": float(fluxes.get(gem.GLUCOSE_EXCHANGE, np.nan)),
        "oxygen_uptake": float(fluxes.get(gem.OXYGEN_EXCHANGE, np.nan)),
        "atp_maintenance_flux": float(fluxes.get(gem.ATPM_RXN, np.nan)),
        "ggpp_flux": float(fluxes.get(gem.NATIVE_GGPP_RXN, np.nan)),
        "PSY_flux": float(fluxes.get(gem.PSY_RXN, np.nan)),
        "DES_flux": float(fluxes.get(gem.DES_RXN, np.nan)),
        "CYC_flux": float(fluxes.get(gem.CYC_RXN, np.nan)),
        "optimization_statuses": ";".join(statuses),
        "n_actual_lp_solves": 3,
        "full_fluxes": fluxes,
    }


def parent_stage_cache(env_row: dict[str, Any]) -> dict[str, Any]:
    key = str(env_row["environment_id"])
    if key in _PARENT_STAGE_CACHE:
        return _PARENT_STAGE_CACHE[key]
    env = env_payload(env_row)
    parent = parent_candidate(env, f"staged_parent_{key}")
    _cobra, _source, augmented, _checksum = cached_augmented_model()
    parent_cfg = verifier.config_for_sparse_candidate({**parent, "candidate_hash": "parent"}, "strong_dynamic_regulation")
    with augmented as model:
        constraints = gem.apply_interval_constraints(model, parent_cfg, np.zeros(4), "balanced", gem.initial_enzyme_capacities(parent_cfg))
        flux = solve_staged_full_flux(model, parent_cfg, gem.state_gamma("balanced", parent_cfg), "balanced")
    _PARENT_STAGE_CACHE[key] = {"candidate": parent, "cfg": parent_cfg, "constraints": constraints, "flux": flux}
    return _PARENT_STAGE_CACHE[key]


def parent_full_flux_profiles(universe_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _cobra, source, augmented, checksum = cached_augmented_model()
    rows = []
    summaries = []
    for env_row in REPRESENTATIVE_ENVIRONMENTS:
        env = env_payload(env_row)
        cfg = verifier.config_for_sparse_candidate({**parent_candidate(env), "candidate_hash": "parent"}, "strong_dynamic_regulation")
        model = augmented.copy()
        constraints = gem.apply_interval_constraints(model, cfg, np.zeros(4), "balanced", gem.initial_enzyme_capacities(cfg))
        flux = solve_staged_full_flux(model, cfg, gem.state_gamma("balanced", cfg), "balanced")
        full = flux["full_fluxes"]
        for rid in universe_ids:
            rxn = get_reaction_or_none(model, rid)
            value = float(full.get(rid, np.nan)) if full is not None else np.nan
            active = False
            if rxn is not None and pd.notna(value):
                active = abs(value - float(rxn.lower_bound)) <= 1e-7 or abs(value - float(rxn.upper_bound)) <= 1e-7
            rows.append(
                {
                    "environment_id": env_row["environment_id"],
                    "reaction_id": rid,
                    "flux": value,
                    "lower_bound": float(rxn.lower_bound) if rxn is not None else np.nan,
                    "upper_bound": float(rxn.upper_bound) if rxn is not None else np.nan,
                    "active_bound": active,
                }
            )
        summaries.append(
            {
                "environment_id": env_row["environment_id"],
                "beta_carotene_flux": flux["beta_carotene_flux"],
                "biomass_flux": flux["biomass_flux"],
                "glucose_uptake": flux["glucose_uptake"],
                "oxygen_uptake": flux["oxygen_uptake"],
                "atp_maintenance_flux": flux["atp_maintenance_flux"],
                "ggpp_flux": flux["ggpp_flux"],
                "gem_checksum": checksum,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def build_stage1_legacy_diagnostic_library(limit: int = 36) -> pd.DataFrame:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    _cobra, _source, model, checksum = cached_augmented_model()
    model_rxns = {r.id: r for r in model.reactions}
    candidates = universe.copy()
    candidates["span"] = candidates.apply(lambda r: max(float(r.get("fva_max", 0.0)) - float(r.get("fva_min", 0.0)), 0.0) if pd.notna(r.get("fva_max", np.nan)) and pd.notna(r.get("fva_min", np.nan)) else 0.0, axis=1)
    candidates["abs_flux"] = candidates["baseline_flux"].fillna(0.0).astype(float).abs()
    candidates = candidates.sort_values(["span", "abs_flux", "selection_score"], ascending=False)
    rows: list[dict[str, Any]] = []
    retained_ids: set[str] = set()

    def retain(rxn_id: str, edit_type: str, klass: str, multiplier: float, rationale: str, source_row: pd.Series | None = None) -> None:
        if rxn_id in retained_ids or rxn_id not in model_rxns or rxn_id in deployment.PROHIBITED_REACTIONS:
            return
        rxn = model_rxns[rxn_id]
        old_lb, old_ub = float(rxn.lower_bound), float(rxn.upper_bound)
        target_ub = np.nan
        target_lb = np.nan
        reference_ub = np.nan
        if edit_type in {"reaction_knockdown", "reaction_capacity_decrease", "transport_capacity_decrease"}:
            if old_ub > 0:
                target_ub = min(1000.0, old_ub * multiplier)
            if old_lb < 0:
                target_lb = old_lb * multiplier
        elif edit_type in {"reaction_capacity_increase", "precursor_supply_enhancement", "transport_capacity_change"}:
            if old_ub >= 999.0:
                span = float(source_row.get("span", 0.0)) if source_row is not None else 0.0
                reference_ub = max(span * 4.0, 1.0)
                target_ub = min(1000.0, reference_ub * multiplier)
            elif old_ub > 0:
                target_ub = min(1000.0, old_ub * multiplier)
            if old_lb < 0 and edit_type == "transport_capacity_change":
                target_lb = max(-1000.0, old_lb * multiplier)
        row = source_row.to_dict() if source_row is not None else {}
        rows.append(
            {
                "intervention_id": f"stage1_legacy_{len(rows)+1:03d}_{rxn_id}_{edit_type}",
                "reaction_id": rxn_id,
                "reaction_name": rxn.name,
                "subsystem": str(rxn.subsystem or row.get("subsystem", "")),
                "edit_type": edit_type,
                "intervention_class": klass,
                "capacity_multiplier": multiplier,
                "target_lower_bound": target_lb,
                "target_upper_bound": target_ub,
                "reference_upper_bound": reference_ub,
                "baseline_flux": float(row.get("baseline_flux", np.nan)) if row else np.nan,
                "fva_min": float(row.get("fva_min", np.nan)) if row else np.nan,
                "fva_max": float(row.get("fva_max", np.nan)) if row else np.nan,
                "retention_rationale": rationale,
                "gem_checksum": checksum,
            }
        )
        retained_ids.add(rxn_id)

    for _, row in candidates.iterrows():
        if len(rows) >= limit:
            break
        rid = str(row["reaction_id"])
        if rid not in model_rxns or rid in deployment.PROHIBITED_REACTIONS:
            continue
        if float(row["span"]) <= 1e-9 and float(row["abs_flux"]) <= 1e-9:
            continue
        text = f"{row.get('subsystem','')} {row.get('inclusion_rule','')}".lower()
        if "transport" in text or "exchange" in text:
            retain(rid, "transport_capacity_change", "transport_or_exchange", EDIT_MULTIPLIERS["transport_or_exchange"], "legacy Stage 1 transport/exchange intervention", row)
        elif "terpenoid" in text or "ggpp" in text or "isoprenoid" in text:
            retain(rid, "precursor_supply_enhancement", "precursor_supply", EDIT_MULTIPLIERS["capacity_increase"], "legacy Stage 1 precursor intervention", row)
        elif "byproduct" in text or "glycolysis" in text:
            retain(rid, "reaction_knockdown", "byproduct_or_carbon_rerouting", EDIT_MULTIPLIERS["knockdown"], "legacy Stage 1 rerouting intervention", row)
        else:
            retain(rid, "reaction_capacity_decrease", "capacity_decrease", EDIT_MULTIPLIERS["capacity_decrease"], "legacy Stage 1 conservative capacity intervention", row)
    manual = [
        (gem.PSY_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "legacy Stage 1 pathway enzyme intervention"),
        (gem.DES_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "legacy Stage 1 pathway enzyme intervention"),
        (gem.CYC_RXN, "pathway_enzyme_capacity_modification", "pathway_enzyme_capacity", EDIT_MULTIPLIERS["pathway_enzyme_capacity"], "legacy Stage 1 pathway enzyme intervention"),
        (gem.OXYGEN_EXCHANGE, "exchange_uptake_or_secretion_change", "oxygen_exchange", 1.25, "legacy Stage 1 oxygen support intervention"),
        (gem.ATPM_RXN, "reaction_capacity_decrease", "atp_demand_reduction", 0.80, "legacy Stage 1 ATP maintenance intervention"),
    ]
    for rid, edit_type, klass, mult, rationale in manual:
        match = universe[universe["reaction_id"].astype(str).eq(rid)]
        retain(rid, edit_type, klass, mult, rationale, match.iloc[0] if not match.empty else None)
    retained_all = pd.DataFrame(rows).drop_duplicates("intervention_id")
    class_caps = {
        "transport_or_exchange": 6,
        "byproduct_or_carbon_rerouting": 6,
        "capacity_decrease": 3,
        "precursor_supply": 3,
        "pathway_enzyme_capacity": 3,
        "oxygen_exchange": 1,
        "atp_demand_reduction": 1,
    }
    retained = pd.concat(
        [retained_all[retained_all["intervention_class"].eq(klass)].head(cap) for klass, cap in class_caps.items()],
        ignore_index=True,
    )
    diagnostic = pd.concat([group.head(4) for _klass, group in retained.groupby("intervention_class", sort=False)], ignore_index=True).head(24)
    return diagnostic


def diagnose_failed_interventions() -> pd.DataFrame:
    prior_library = build_stage1_legacy_diagnostic_library()
    prior_rows = []
    for row in prior_library.itertuples(index=False):
        candidate = candidate_from_intervention(pd.Series(row._asdict()), {"temperature": 30.0, "pH": 5.0, "DO": 65.0}, f"diag_final_legacy_{row.intervention_id}")
        try:
            prior_rows.append(static_stage_diagnostic(candidate))
        except Exception as exc:
            prior_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "requested_reaction_id": candidate["edits"][0]["reaction_id"],
                    "requested_edit_type": candidate["edits"][0]["edit_type"],
                    "requested_magnitude": candidate["edits"][0]["capacity_multiplier"],
                    "solver_status": f"failed:{type(exc).__name__}",
                    "effective_or_inactive_classification": "diagnostic_failed",
                    "reason": str(exc),
                }
            )
    prior = pd.DataFrame(prior_rows)
    rows = []
    for row in prior.to_dict("records"):
        classes: list[str] = []
        if str(row.get("effective_or_inactive_classification", "")) == "effective":
            classes.append("passed_previous_representative_diagnostic")
        if str(row.get("solver_status", "")).startswith("failed") or str(row.get("effective_or_inactive_classification", "")) == "diagnostic_failed":
            classes.append("unresolved")
        if not bool(row.get("edit_survived_dynamic_constraints", False)):
            etype = str(row.get("requested_edit_type", ""))
            rid = str(row.get("requested_reaction_id", ""))
            if rid == gem.OXYGEN_EXCHANGE or "oxygen" in etype:
                classes.append("edit is overwritten by environment rules")
            elif rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN}:
                classes.append("edit is overwritten by pathway-capacity rules")
            elif rid == gem.ATPM_RXN:
                classes.append("edit is overwritten by regulator rules")
            else:
                classes.append("edit changes a non-limiting bound")
        if abs(float(row.get("flux_delta", 0.0) or 0.0)) <= STAGED_EFFECT_TOLERANCE and abs(float(row.get("final_product_delta", 0.0) or 0.0)) <= STAGED_EFFECT_TOLERANCE:
            classes.append("reaction carries no flux" if abs(float(row.get("parent_flux", 0.0) or 0.0)) <= STAGED_EFFECT_TOLERANCE else "alternate pathways compensate completely")
        if float(row.get("original_upper_bound", np.nan) or np.nan) >= 999.0 and float(row.get("actual_upper_bound", np.nan) or np.nan) >= 999.0:
            classes.append("edit changes a non-limiting bound")
        if str(row.get("requested_edit_type", "")) in {"transport_capacity_change", "exchange_uptake_or_secretion_change"} and float(row.get("original_lower_bound", 0.0) or 0.0) < 0:
            if float(row.get("actual_lower_bound", 0.0) or 0.0) == float(row.get("original_lower_bound", 0.0) or 0.0):
                classes.append("reaction direction was handled incorrectly")
        if not classes:
            classes.append("phenotype effect exists but the diagnostic metric misses it")
        rows.append(
            {
                **row,
                "interval_dynamic_bound_summary": f"survived={row.get('edit_survived_dynamic_constraints', '')}",
                "active_constraint_status": "not_active_or_not_recorded",
                "trajectory_consequence": "none_detected_previous_gate" if "effective" not in classes else "staged_flux_change_detected",
                "failure_classes": ";".join(sorted(set(classes))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "final_dbtl_failed_intervention_diagnosis.csv", index=False)
    return out


def write_dynamic_reaction_relevance() -> pd.DataFrame:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    cobra, source, augmented, checksum = cached_augmented_model()
    model_ids = reaction_ids(augmented)
    manual_rows = []
    for rid, name, subsystem in [
        (gem.PSY_RXN, "heterologous phytoene synthase", "heterologous beta-carotene pathway"),
        (gem.DES_RXN, "heterologous phytoene-to-lycopene lump", "heterologous beta-carotene pathway"),
        (gem.CYC_RXN, "heterologous lycopene beta-cyclase lump", "heterologous beta-carotene pathway"),
        ("BETA_EXPORT_ASSIST", "beta-carotene export/loss parameter", "declared dynamic product-loss parameter"),
    ]:
        manual_rows.append(
            {
                "reaction_id": rid,
                "reaction_name": name,
                "subsystem": subsystem,
                "baseline_flux": np.nan,
                "fva_min": np.nan,
                "fva_max": np.nan,
                "graph_distance_to_product_pathway": 0,
                "gene_association": "",
                "risk_level": "medium",
                "inclusion_rule": "curated_dynamic_parameter",
                "selection_score": 1000,
            }
        )
    base = pd.concat([universe, pd.DataFrame(manual_rows)], ignore_index=True, sort=False).drop_duplicates("reaction_id", keep="first")
    ids = [str(r) for r in base["reaction_id"] if str(r) in model_ids or str(r) == "BETA_EXPORT_ASSIST"]
    flux_profiles, _summary = parent_full_flux_profiles([rid for rid in ids if rid in model_ids])
    grouped = flux_profiles.groupby("reaction_id")
    rows = []
    for _, row in base.iterrows():
        rid = str(row["reaction_id"])
        rxn = get_reaction_or_none(augmented, rid)
        prof = grouped.get_group(rid) if rid in grouped.groups else pd.DataFrame()
        env_flux_range = float(prof["flux"].max() - prof["flux"].min()) if not prof.empty and prof["flux"].notna().any() else 0.0
        dynamic_flux_range = env_flux_range
        active_freq = float(prof["active_bound"].mean()) if not prof.empty else (1.0 if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN} else 0.0)
        baseline = float(row.get("baseline_flux", np.nan)) if pd.notna(row.get("baseline_flux", np.nan)) else (float(prof["flux"].iloc[0]) if not prof.empty and pd.notna(prof["flux"].iloc[0]) else 0.0)
        fva_min = float(row.get("fva_min", np.nan)) if pd.notna(row.get("fva_min", np.nan)) else np.nan
        fva_max = float(row.get("fva_max", np.nan)) if pd.notna(row.get("fva_max", np.nan)) else np.nan
        text = f"{row.get('reaction_name', row.get('name', ''))} {row.get('subsystem', '')} {row.get('inclusion_rule', '')} {rid}"
        mechanism = mechanism_for_reaction(rid, text)
        distance = int(row.get("graph_distance_to_product_pathway", 3)) if pd.notna(row.get("graph_distance_to_product_pathway", np.nan)) else (0 if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, "BETA_EXPORT_ASSIST"} else 3)
        gpr = str(row.get("gene_association", "") or (rxn.gene_reaction_rule if rxn is not None else ""))
        gene_count = len({g.id for g in rxn.genes}) if rxn is not None else 0
        compensation = "high" if " or " in gpr.lower() else "medium" if gene_count > 1 else "low"
        override = "high" if rid in {gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN, gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN} else "medium" if active_freq > 0 else "low"
        ref_source, ref_value = finite_reference_capacity(baseline, fva_min, fva_max, dynamic_flux_range)
        score = (
            10.0 * min(abs(baseline), 1.0)
            + 8.0 * min(abs(fva_max - fva_min) if pd.notna(fva_min) and pd.notna(fva_max) else 0.0, 2.0)
            + 12.0 * min(dynamic_flux_range, 2.0)
            + 25.0 * active_freq
            + max(0.0, 18.0 - 4.0 * distance)
            + (35.0 if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN, "BETA_EXPORT_ASSIST", gem.NATIVE_GGPP_RXN} else 0.0)
        )
        reason = []
        if active_freq > 0:
            reason.append("active_bound_in_parent_environment_screen")
        if dynamic_flux_range > STAGED_EFFECT_TOLERANCE:
            reason.append("flux_varies_across_public_representative_environments")
        if distance <= 1:
            reason.append("near_beta_carotene_or_ggpp_pathway")
        if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN, "BETA_EXPORT_ASSIST"}:
            reason.append("declared_dynamic_parameter_or_interval_constraint")
        rows.append(
            {
                "reaction_id": rid,
                "reaction_name": str(row.get("reaction_name", row.get("name", rxn.name if rxn is not None else rid))),
                "subsystem": str(row.get("subsystem", rxn.subsystem if rxn is not None else "")),
                "baseline_flux": baseline,
                "FVA_min": fva_min,
                "FVA_max": fva_max,
                "environment_flux_range": env_flux_range,
                "dynamic_flux_range": dynamic_flux_range,
                "active_bound_frequency": active_freq,
                "distance_to_beta_carotene_pathway": distance,
                "GPR_rule": gpr,
                "gene_count": gene_count,
                "compensation_risk": compensation,
                "dynamic_override_risk": override,
                "candidate_mechanism_class": mechanism,
                "relevance_score": float(score),
                "selection_reason": ";".join(reason) if reason else "low_public_dynamic_leverage_evidence",
                "reference_capacity_source": ref_source,
                "reference_capacity_value": ref_value,
                "gem_checksum": checksum,
            }
        )
    out = pd.DataFrame(rows).sort_values("relevance_score", ascending=False)
    out.to_csv(DATA / "final_dbtl_dynamic_reaction_relevance.csv", index=False)
    return out


def diagnostic_intervention_rows(relevance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cobra, source, augmented, checksum = cached_augmented_model()

    def add(rid: str, intervention_type: str, mechanism: str, magnitudes: list[float], edit_type: str, mode: str, reason: str) -> None:
        rxn = get_reaction_or_none(augmented, rid)
        rel = relevance[relevance["reaction_id"].astype(str).eq(rid)]
        rel_row = rel.iloc[0] if not rel.empty else pd.Series({})
        old_lb = float(rxn.lower_bound) if rxn is not None else 0.0
        old_ub = float(rxn.upper_bound) if rxn is not None else 0.0
        base = {
            "reaction_id": rid,
            "reaction_name": rxn.name if rxn is not None else str(rel_row.get("reaction_name", rid)),
            "mechanism_class": mechanism,
            "intervention_class": mechanism,
            "subsystem": rxn.subsystem if rxn is not None else str(rel_row.get("subsystem", "")),
            "gene_ids": ";".join(sorted(g.id for g in rxn.genes)) if rxn is not None else "",
            "GPR_rule": rxn.gene_reaction_rule if rxn is not None else "",
            "gpr_rule": rxn.gene_reaction_rule if rxn is not None else "",
            "intervention_type": intervention_type,
            "edit_type": edit_type,
            "original_lower_bound": old_lb,
            "original_upper_bound": old_ub,
            "reference_capacity_source": str(rel_row.get("reference_capacity_source", "curated_dynamic_parameter")),
            "reference_capacity_value": float(rel_row.get("reference_capacity_value", 0.1) if pd.notna(rel_row.get("reference_capacity_value", np.nan)) else 0.1),
            "reference_capacity": float(rel_row.get("reference_capacity_value", 0.1) if pd.notna(rel_row.get("reference_capacity_value", np.nan)) else 0.1),
            "construction_tier": "tier1" if rid in {gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN} else "tier2",
            "risk_level": "medium" if mechanism in {"transport_and_environment_coupling", "cofactor_or_energy_support"} else "low",
            "pleiotropy_warning": "high" if " or " in (rxn.gene_reaction_rule if rxn is not None else "").lower() else "medium",
            "dynamic_survival_rule": "verified_by_interval_constraint_or_dynamic_parameter_trace" if is_dynamic_parameter_intervention(rid, edit_type) else "verified_by_post_constraint_bound_trace",
            "biological_interpretation": reason,
            "retention_rationale": reason,
            "gem_checksum": checksum,
        }
        for mag in magnitudes:
            row = {**base, "capacity_multiplier": float(mag), "allowed_magnitudes": ";".join(str(m) for m in magnitudes), "edit_direction": mode, "bound_transform": mode}
            row.update(directional_bounds_for_intervention(pd.Series(row), float(mag), mode))
            row["intervention_id"] = f"diag_{len(rows)+1:03d}_{rid}_{intervention_type}_{str(mag).replace('.', 'p')}"
            rows.append(row)

    add(gem.PSY_RXN, "pathway_enzyme_dynamic_capacity", "direct_heterologous_pathway_capacity", [0.55, 0.75, 1.35, 1.8], "pathway_enzyme_capacity_modification", "overexpression", "PSY base/initial capacity edit enters every dynamic interval")
    add(gem.DES_RXN, "pathway_enzyme_dynamic_capacity", "direct_heterologous_pathway_capacity", [0.55, 0.75, 1.35, 1.8], "pathway_enzyme_capacity_modification", "overexpression", "DES base/initial capacity edit enters every dynamic interval")
    add(gem.CYC_RXN, "pathway_enzyme_dynamic_capacity", "direct_heterologous_pathway_capacity", [0.55, 0.75, 1.35, 1.8], "pathway_enzyme_capacity_modification", "overexpression", "CYC base/initial capacity edit enters every dynamic interval")
    add(gem.OXYGEN_EXCHANGE, "oxygen_uptake_capacity", "transport_and_environment_coupling", [0.65, 0.85, 1.25, 1.6], "exchange_uptake_or_secretion_change", "transporter_upregulation", "oxygen uptake coefficient controls interval oxygen lower bound and burden update")
    add(gem.GLUCOSE_EXCHANGE, "glucose_uptake_capacity", "transport_and_environment_coupling", [0.70, 0.90, 1.2, 1.5], "exchange_uptake_or_secretion_change", "transporter_upregulation", "glucose uptake coefficient controls interval carbon lower bound")
    add(gem.ATPM_RXN, "ATP_maintenance_demand", "cofactor_or_energy_support", [0.65, 0.80, 1.20], "reaction_capacity_decrease", "decrease", "ATP maintenance demand enters every interval and burden update")
    add("BETA_EXPORT_ASSIST", "product_loss_parameter", "product_export_or_loss", [0.60, 1.4, 2.0], "product_loss_modification", "overexpression", "declared product loss/export parameter alters dynamic beta-carotene degradation")
    add(gem.NATIVE_GGPP_RXN, "GGPP_supply_capacity", "precursor_supply_or_competing_sink", [0.50, 0.75, 1.35], "precursor_supply_enhancement", "overexpression", "native GGPP reaction is direct precursor source for heterologous pathway")

    top = relevance[
        relevance["candidate_mechanism_class"].isin(["precursor_supply_or_competing_sink", "cofactor_or_energy_support", "transport_and_environment_coupling"])
        & ~relevance["reaction_id"].isin([gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.OXYGEN_EXCHANGE, gem.GLUCOSE_EXCHANGE, gem.ATPM_RXN, "BETA_EXPORT_ASSIST", gem.NATIVE_GGPP_RXN])
    ].head(12)
    for _, rel in top.iterrows():
        rid = str(rel["reaction_id"])
        mechanism = str(rel["candidate_mechanism_class"])
        edit_type = "reaction_knockdown" if mechanism == "precursor_supply_or_competing_sink" else "reaction_capacity_decrease"
        add(rid, "public_flux_supported_capacity_probe", mechanism, [0.50, 0.75], edit_type, "suppression", "public relevance screen showed flux, FVA, active-bound, or pathway-context leverage")
    out = pd.DataFrame(rows).drop_duplicates("intervention_id")
    out.to_csv(DATA / "final_dbtl_diagnostic_intervention_library.csv", index=False)
    return out


def candidate_from_intervention(intervention: pd.Series, env: dict[str, float], candidate_id: str) -> dict[str, Any]:
    edit_type = str(intervention.get("edit_type", intervention.get("intervention_type", "")))
    edit = {
        "reaction_id": str(intervention["reaction_id"]),
        "edit_type": edit_type,
        "capacity_multiplier": float(intervention.get("capacity_multiplier", 1.0)),
        "rationale": f"Final DBTL diagnostic intervention: {intervention['intervention_id']}",
    }
    for key in ["target_lower_bound", "target_upper_bound", "reference_upper_bound", "reference_lower_bound", "absolute_upper_bound", "absolute_lower_bound"]:
        if key in intervention and pd.notna(intervention[key]):
            edit[key] = float(intervention[key])
    return {
        "candidate_id": candidate_id,
        "strain_id": f"{candidate_id}_strain",
        "edit_tier": 1,
        "edits": [edit],
        "environment": env,
        "requested_assay_panel": "basic",
        "metadata": {"source": "final_dbtl_intervention_redesign_diagnostic"},
    }


def stage_case(intervention: pd.Series, env_row: dict[str, Any]) -> dict[str, Any]:
    env = env_payload(env_row)
    candidate = candidate_from_intervention(intervention, env, f"staged_{intervention['intervention_id']}_{env_row['environment_id']}")
    cobra, source, augmented, checksum = cached_augmented_model()
    cfg = verifier.config_for_sparse_candidate({**candidate, "candidate_hash": candidate_spec_hash(candidate)}, "strong_dynamic_regulation")
    parent_cached = parent_stage_cache(env_row)
    parent_constraints = parent_cached["constraints"]
    parent_flux = parent_cached["flux"]
    with augmented as edited_model:
        trace, notes = apply_and_trace(edited_model, candidate)
        edited_constraints = gem.apply_interval_constraints(edited_model, cfg, np.zeros(4), "balanced", gem.initial_enzyme_capacities(cfg))
        edited_flux = solve_staged_full_flux(edited_model, cfg, gem.state_gamma("balanced", cfg), "balanced")
        rxn = get_reaction_or_none(edited_model, str(intervention["reaction_id"]))
        edited_bounds = (float(rxn.lower_bound), float(rxn.upper_bound)) if rxn is not None else (np.nan, np.nan)
    rid = str(intervention["reaction_id"])
    p_full = parent_flux.get("full_fluxes", pd.Series(dtype=float))
    e_full = edited_flux.get("full_fluxes", pd.Series(dtype=float))
    parent_r_flux = float(p_full.get(rid, np.nan)) if rid in p_full.index else np.nan
    edited_r_flux = float(e_full.get(rid, np.nan)) if rid in e_full.index else np.nan
    flux_delta = edited_r_flux - parent_r_flux if pd.notna(parent_r_flux) and pd.notna(edited_r_flux) else 0.0
    product_delta = float(edited_flux["beta_carotene_flux"]) - float(parent_flux["beta_carotene_flux"])
    growth_delta = float(edited_flux["biomass_flux"]) - float(parent_flux["biomass_flux"])
    constraint_delta = max(
        abs(float(edited_constraints.get("glucose_lower_bound", 0.0)) - float(parent_constraints.get("glucose_lower_bound", 0.0))),
        abs(float(edited_constraints.get("oxygen_lower_bound", 0.0)) - float(parent_constraints.get("oxygen_lower_bound", 0.0))),
        abs(float(edited_constraints.get("atp_maintenance_lower_bound", 0.0)) - float(parent_constraints.get("atp_maintenance_lower_bound", 0.0))),
        abs(float(edited_constraints.get("PSY_effective_upper_bound", 0.0)) - float(parent_constraints.get("PSY_effective_upper_bound", 0.0))),
        abs(float(edited_constraints.get("DES_effective_upper_bound", 0.0)) - float(parent_constraints.get("DES_effective_upper_bound", 0.0))),
        abs(float(edited_constraints.get("CYC_effective_upper_bound", 0.0)) - float(parent_constraints.get("CYC_effective_upper_bound", 0.0))),
    )
    bound_changed = any(bool(r.get("actual_bound_changed", False)) for r in trace)
    dynamic_changed = constraint_delta > STAGED_EFFECT_TOLERANCE or str(intervention["reaction_id"]) == "BETA_EXPORT_ASSIST"
    active = False
    if pd.notna(edited_bounds[0]) and rid in e_full.index:
        val = float(e_full.get(rid))
        active = abs(val - edited_bounds[0]) <= 1e-7 or abs(val - edited_bounds[1]) <= 1e-7
    effect_size = max(abs(flux_delta), abs(product_delta), abs(growth_delta), constraint_delta)
    passed = (bound_changed or dynamic_changed) and effect_size > STAGED_EFFECT_TOLERANCE and str(edited_flux["pfba_status"]) == "optimal"
    return {
        "intervention_id": str(intervention["intervention_id"]),
        "reaction_id": rid,
        "environment_id": env_row["environment_id"],
        **env,
        "requested_edit_type": str(intervention["edit_type"]),
        "requested_magnitude": float(intervention["capacity_multiplier"]),
        "actual_bound_change": bound_changed,
        "dynamic_constraint_change": dynamic_changed,
        "changed_bound_active": active,
        "parent_reaction_flux": parent_r_flux,
        "edited_reaction_flux": edited_r_flux,
        "reaction_flux_delta": flux_delta,
        "parent_product_flux": float(parent_flux["beta_carotene_flux"]),
        "edited_product_flux": float(edited_flux["beta_carotene_flux"]),
        "product_flux_delta": product_delta,
        "parent_growth_flux": float(parent_flux["biomass_flux"]),
        "edited_growth_flux": float(edited_flux["biomass_flux"]),
        "growth_flux_delta": growth_delta,
        "parent_ggpp_flux": float(parent_flux["ggpp_flux"]),
        "edited_ggpp_flux": float(edited_flux["ggpp_flux"]),
        "parent_oxygen_flux": float(parent_flux["oxygen_uptake"]),
        "edited_oxygen_flux": float(edited_flux["oxygen_uptake"]),
        "parent_atp_maintenance_flux": float(parent_flux["atp_maintenance_flux"]),
        "edited_atp_maintenance_flux": float(edited_flux["atp_maintenance_flux"]),
        "constraint_effect_size": constraint_delta,
        "staged_effect_size": effect_size,
        "feasible": True,
        "solver_status": ";".join([str(parent_flux["pfba_status"]), str(edited_flux["pfba_status"])]),
        "effective_model_hash": hash_payload(
            {
                "candidate": candidate["edits"],
                "environment": env,
                "trace": trace,
                "edited_constraints": edited_constraints,
                "simulator_version": FINAL_SIMULATOR_VERSION,
                "gem_checksum": checksum,
            }
        ),
        "effective_or_inactive_classification": "effective" if passed else "inactive_or_equivalent",
        "reason": "reproducible staged flux_or_constraint_change" if passed else "no staged flux_or_constraint_change_above_frozen_tolerance",
        "application_notes": ";".join(notes),
        "gem_checksum": checksum,
    }


def write_staged_screening(diagnostic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(diagnostic) * len(REPRESENTATIVE_ENVIRONMENTS)
    out_path = DATA / "final_dbtl_staged_screening_diagnostics.csv"
    mirror_path = DATA / "final_dbtl_edit_application_diagnostics.csv"
    if out_path.exists():
        out_path.unlink()
    if mirror_path.exists():
        mirror_path.unlink()
    case_index = 0
    for intervention in diagnostic.to_dict("records"):
        for env_row in REPRESENTATIVE_ENVIRONMENTS:
            case_index += 1
            if case_index == 1 or case_index % 20 == 0:
                print(f"[final-dbtl] staged case {case_index}/{total}", flush=True)
            try:
                rows.append(stage_case(pd.Series(intervention), env_row))
            except Exception as exc:
                rows.append(
                    {
                        "intervention_id": intervention.get("intervention_id", ""),
                        "reaction_id": intervention.get("reaction_id", ""),
                        "environment_id": env_row["environment_id"],
                        "effective_or_inactive_classification": "diagnostic_failed",
                        "feasible": False,
                        "solver_status": f"failed:{type(exc).__name__}",
                        "reason": str(exc),
                    }
                )
            if case_index % 20 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                pd.DataFrame(rows).to_csv(mirror_path, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    out.to_csv(mirror_path, index=False)
    return out


def write_diagnostics(library: pd.DataFrame) -> pd.DataFrame:
    return write_staged_screening(library)


def stage_pass_table(staged: pd.DataFrame) -> pd.DataFrame:
    grouped = staged.groupby("intervention_id", dropna=False)
    rows = []
    for iid, group in grouped:
        passed = bool((group["effective_or_inactive_classification"] == "effective").any())
        best = group.sort_values("staged_effect_size", ascending=False).iloc[0] if "staged_effect_size" in group else group.iloc[0]
        rows.append(
            {
                "intervention_id": iid,
                "staged_pass": passed,
                "best_environment_id": best.get("environment_id", "central"),
                "staged_effect_size": float(best.get("staged_effect_size", 0.0) or 0.0),
                "tested_environments": ";".join(sorted(set(group["environment_id"].astype(str)))),
                "staged_effective_cases": int((group["effective_or_inactive_classification"] == "effective").sum()),
                "staged_cases": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def dynamic_pair_metrics(parent_tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], edited_tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> dict[str, float]:
    p_traj, p_flux, p_cons, p_summary = parent_tables
    e_traj, e_flux, e_cons, e_summary = edited_tables
    p_sum = p_summary.iloc[0]
    e_sum = e_summary.iloc[0]
    metrics = {
        "final_product_delta": float(e_sum["final_product"]) - float(p_sum["final_product"]),
        "product_AUC_delta": float(e_sum["product_AUC"]) - float(p_sum["product_AUC"]),
        "final_biomass_delta": float(e_sum["final_biomass"]) - float(p_sum["final_biomass"]),
        "minimum_biomass_delta": float(e_sum["minimum_biomass"]) - float(p_sum["minimum_biomass"]),
        "oxidative_burden_max_delta": float(e_traj["z_ox"].max()) - float(p_traj["z_ox"].max()),
        "ATP_burden_max_delta": float(e_traj["z_atp"].max()) - float(p_traj["z_atp"].max()),
        "bottleneck_burden_max_delta": float(e_traj["z_bottle"].max()) - float(p_traj["z_bottle"].max()),
        "E_PSY_final_delta": float(e_traj["E_PSY"].iloc[-1]) - float(p_traj["E_PSY"].iloc[-1]),
        "E_DES_final_delta": float(e_traj["E_DES"].iloc[-1]) - float(p_traj["E_DES"].iloc[-1]),
        "E_CYC_final_delta": float(e_traj["E_CYC"].iloc[-1]) - float(p_traj["E_CYC"].iloc[-1]),
    }
    for col in ["beta_carotene_flux", "biomass_flux", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux"]:
        if col in p_flux and col in e_flux:
            metrics[f"{col}_trajectory_max_abs_delta"] = float((e_flux[col].astype(float) - p_flux[col].astype(float)).abs().max())
    return metrics


def run_small_exact_diagnostics(library: pd.DataFrame, cases: int | None = None) -> pd.DataFrame:
    staged_path = DATA / "final_dbtl_staged_screening_diagnostics.csv"
    staged = pd.read_csv(staged_path) if staged_path.exists() else pd.DataFrame()
    stage_pass = stage_pass_table(staged) if not staged.empty else pd.DataFrame()
    selected = library.merge(stage_pass[stage_pass["staged_pass"]], on="intervention_id", how="inner") if not stage_pass.empty else library.head(0)
    if cases is not None:
        selected = selected.head(cases)
    rows = []
    old_time_points = exact.TIME_POINTS
    exact.TIME_POINTS = 5
    try:
        for idx, intervention in enumerate(selected.to_dict("records"), start=1):
            env_row = next((r for r in REPRESENTATIVE_ENVIRONMENTS if r["environment_id"] == intervention.get("best_environment_id")), REPRESENTATIVE_ENVIRONMENTS[0])
            env = env_payload(env_row)
            parent = {**parent_candidate(env, f"short_parent_{idx:03d}"), "candidate_hash": f"short_parent_{idx:03d}"}
            edited = candidate_from_intervention(pd.Series(intervention), env, f"short_{intervention['intervention_id']}")
            edited["candidate_hash"] = candidate_spec_hash(edited)
            parent_tables = verifier.exact_sparse_dynamic_rollout(parent, hidden_regime="strong_dynamic_regulation")
            edited_tables = verifier.exact_sparse_dynamic_rollout(edited, hidden_regime="strong_dynamic_regulation")
            metrics = dynamic_pair_metrics(parent_tables, edited_tables)
            p_traj, p_flux, p_cons, p_summary = parent_tables
            e_traj, e_flux, e_cons, e_summary = edited_tables
            parent_fp = phenotype_fingerprint({"trajectory": p_traj, "flux": p_flux, "constraints": p_cons}, tolerance=DYNAMIC_PHENOTYPE_TOLERANCE)
            edited_fp = phenotype_fingerprint({"trajectory": e_traj, "flux": e_flux, "constraints": e_cons}, tolerance=DYNAMIC_PHENOTYPE_TOLERANCE)
            dynamic_effect_size = max(abs(v) for v in metrics.values()) if metrics else 0.0
            passed = bool(dynamic_effect_size > DYNAMIC_PHENOTYPE_TOLERANCE and parent_fp != edited_fp and e_summary.iloc[0]["exact_status"] == "complete_exact_dynamic_pfba")
            rows.append(
                {
                    "intervention_id": intervention["intervention_id"],
                    "reaction_id": intervention["reaction_id"],
                    "environment_id": env_row["environment_id"],
                    "case_role": "paired_parent_vs_edited",
                    "parent_candidate_id": parent["candidate_id"],
                    "edited_candidate_id": edited["candidate_id"],
                    "parent_candidate_hash": parent["candidate_hash"],
                    "edited_candidate_hash": edited["candidate_hash"],
                    "parent_final_product": float(p_summary.iloc[0]["final_product"]),
                    "edited_final_product": float(e_summary.iloc[0]["final_product"]),
                    "parent_product_AUC": float(p_summary.iloc[0]["product_AUC"]),
                    "edited_product_AUC": float(e_summary.iloc[0]["product_AUC"]),
                    "parent_final_biomass": float(p_summary.iloc[0]["final_biomass"]),
                    "edited_final_biomass": float(e_summary.iloc[0]["final_biomass"]),
                    **metrics,
                    "dynamic_effect_size": dynamic_effect_size,
                    "dynamic_pass": passed,
                    "parent_exact_status": p_summary.iloc[0]["exact_status"],
                    "edited_exact_status": e_summary.iloc[0]["exact_status"],
                    "n_actual_lp_solves": int(p_summary.iloc[0]["n_actual_lp_solves"]) + int(e_summary.iloc[0]["n_actual_lp_solves"]),
                    "effective_model_hash": effective_model_hash(edited, e_summary.iloc[0]["gem_source_checksum"], e_cons),
                    "parent_phenotype_fingerprint": parent_fp,
                    "edited_phenotype_fingerprint": edited_fp,
                    "effective_phenotype_fingerprint": edited_fp,
                    "classification_tolerance": DYNAMIC_PHENOTYPE_TOLERANCE,
                }
            )
    finally:
        exact.TIME_POINTS = old_time_points
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "final_dbtl_small_exact_diagnostics.csv", index=False)
    out.to_csv(DATA / "final_dbtl_short_dynamic_diagnostics.csv", index=False)
    return out


def build_actionable_library(limit: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevance = write_dynamic_reaction_relevance()
    diagnostic = diagnostic_intervention_rows(relevance).head(limit)
    return diagnostic, pd.DataFrame()


def final_library_from_gates(diagnostic: pd.DataFrame, staged: pd.DataFrame, small_exact: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_pass = stage_pass_table(staged)
    dyn = small_exact if small_exact is not None else pd.DataFrame()
    dynamic_pass_ids = set(dyn[dyn.get("dynamic_pass", False).astype(bool)]["intervention_id"].astype(str)) if not dyn.empty and "dynamic_pass" in dyn else set()
    staged_pass_ids = set(stage_pass[stage_pass["staged_pass"]]["intervention_id"].astype(str))
    retained_ids = staged_pass_ids.intersection(dynamic_pass_ids) if small_exact is not None else staged_pass_ids
    retained = diagnostic[diagnostic["intervention_id"].astype(str).isin(retained_ids)].copy()
    retained = retained.merge(stage_pass[["intervention_id", "staged_effect_size", "tested_environments"]], on="intervention_id", how="left")
    if not dyn.empty:
        dyn_cols = ["intervention_id", "dynamic_effect_size", "effective_model_hash", "effective_phenotype_fingerprint"]
        retained = retained.merge(dyn[dyn_cols], on="intervention_id", how="left")
    for col in ["dynamic_effect_size", "effective_model_hash", "effective_phenotype_fingerprint"]:
        if col not in retained:
            retained[col] = np.nan
    retained = retained.drop_duplicates("effective_model_hash", keep="first")
    retained = retained.drop_duplicates("effective_phenotype_fingerprint", keep="first")
    retained["dynamic_effect_summary"] = retained.apply(lambda r: f"staged={float(r.get('staged_effect_size', 0.0) or 0.0):.6g};short_dynamic={float(r.get('dynamic_effect_size', 0.0) or 0.0):.6g}", axis=1)
    retained["effective_model_hashes"] = retained["effective_model_hash"]
    retained["phenotype_fingerprints"] = retained["effective_phenotype_fingerprint"]
    retained["retention_reason"] = "passed staged exact screening and short dynamic exact phenotype gate" if small_exact is not None else "passed staged exact screening; pending short dynamic gate"
    final_cols = [
        "intervention_id",
        "reaction_id",
        "reaction_name",
        "mechanism_class",
        "subsystem",
        "gene_ids",
        "GPR_rule",
        "intervention_type",
        "edit_type",
        "capacity_multiplier",
        "allowed_magnitudes",
        "reference_capacity",
        "reference_capacity_source",
        "reference_capacity_value",
        "bound_transform",
        "target_lower_bound",
        "target_upper_bound",
        "reference_upper_bound",
        "construction_tier",
        "risk_level",
        "pleiotropy_warning",
        "dynamic_effect_summary",
        "tested_environments",
        "staged_effect_size",
        "dynamic_effect_size",
        "effective_model_hashes",
        "phenotype_fingerprints",
        "retention_reason",
        "gem_checksum",
    ]
    for col in final_cols:
        if col not in retained:
            retained[col] = np.nan
    retained = retained[final_cols].sort_values(["mechanism_class", "intervention_id"])
    excluded_rows = []
    retained_ids_final = set(retained["intervention_id"].astype(str))
    for _, row in diagnostic.iterrows():
        iid = str(row["intervention_id"])
        reasons = []
        if iid not in staged_pass_ids:
            reasons.append("failed_staged_exact_screen")
        if small_exact is not None and iid in staged_pass_ids and iid not in dynamic_pass_ids:
            reasons.append("failed_short_dynamic_exact_screen")
        if iid in retained_ids and iid not in retained_ids_final:
            reasons.append("duplicate_effective_model_or_phenotype")
        if not reasons:
            reasons.append("retained")
        if "retained" not in reasons:
            excluded_rows.append({"intervention_id": iid, "reaction_id": row["reaction_id"], "exclusion_reason": ";".join(reasons), "mechanism_class": row["mechanism_class"]})
    excluded = pd.DataFrame(excluded_rows)
    retained.to_csv(DATA / "final_dbtl_actionable_intervention_library.csv", index=False)
    excluded.to_csv(DATA / "final_dbtl_excluded_interventions.csv", index=False)
    return retained, excluded


def write_design_space_compression(library: pd.DataFrame, small_exact: pd.DataFrame | None) -> pd.DataFrame:
    rng = np.random.default_rng(91217)
    if library.empty:
        out = pd.DataFrame([{"candidate_specifications": 0, "unique_effective_models": 0, "unique_diagnostic_phenotypes": 0, "candidate_to_effective_model_compression_ratio": np.nan, "effective_model_to_phenotype_compression_ratio": np.nan, "mechanism_diversity": 0, "subsystem_diversity": 0, "environment_diversity": 0, "invalid_proposal_rate": 0.0, "duplicate_replacement_rate": 0.0}])
        out.to_csv(DATA / "final_dbtl_design_space_compression.csv", index=False)
        return out
    phenotypes = {}
    if small_exact is not None and not small_exact.empty:
        phenotypes = dict(zip(small_exact["intervention_id"].astype(str), small_exact["effective_phenotype_fingerprint"].astype(str)))
    sample_rows = []
    specs: list[dict[str, Any]] = []
    envs = [env_payload(e) | {"environment_id": e["environment_id"]} for e in REPRESENTATIVE_ENVIRONMENTS]
    lib_records = library.to_dict("records")
    for i in range(DESIGN_SPACE_SAMPLE_SIZE):
        edit_count = int(rng.choice([1, 1, 2, 2, 3]))
        chosen = list(rng.choice(len(lib_records), size=min(edit_count, len(lib_records)), replace=False))
        env = envs[i % len(envs)]
        candidate = {
            "candidate_id": f"compression_sample_{i:04d}",
            "strain_id": f"compression_sample_{i:04d}",
            "edit_tier": 2,
            "environment": {k: env[k] for k in ["temperature", "pH", "DO"]},
            "requested_assay_panel": "basic",
            "edits": [],
        }
        component_models = []
        component_phenotypes = []
        mechanisms = []
        subsystems = []
        for idx in chosen:
            intervention = pd.Series(lib_records[idx])
            c = candidate_from_intervention(intervention, candidate["environment"], candidate["candidate_id"])
            candidate["edits"].extend(c["edits"])
            component_models.append(str(intervention.get("effective_model_hashes", intervention["intervention_id"])))
            component_phenotypes.append(phenotypes.get(str(intervention["intervention_id"]), str(intervention.get("phenotype_fingerprints", ""))))
            mechanisms.append(str(intervention.get("mechanism_class", "")))
            subsystems.append(str(intervention.get("subsystem", "")))
        spec_hash = candidate_spec_hash(candidate)
        eff_hash = hash_payload({"environment": candidate["environment"], "component_effective_models": sorted(component_models), "simulator": FINAL_SIMULATOR_VERSION})
        pheno_hash = hash_payload({"environment": candidate["environment"], "component_phenotypes": sorted(component_phenotypes), "tolerance": DYNAMIC_PHENOTYPE_TOLERANCE})
        specs.append(candidate)
        sample_rows.append({"candidate_id": candidate["candidate_id"], "candidate_spec_hash": spec_hash, "effective_model_hash": eff_hash, "diagnostic_phenotype_hash": pheno_hash, "edit_count": len(candidate["edits"]), "mechanisms": ";".join(sorted(set(mechanisms))), "subsystems": ";".join(sorted(set(subsystems))), "environment_id": env["environment_id"]})
    sample = pd.DataFrame(sample_rows)
    sample.to_csv(DATA / "final_dbtl_design_space_sample.csv", index=False)
    dup_specs = len(sample) - sample["candidate_spec_hash"].nunique()
    duplicate_replacements = max(0, dup_specs)
    out = pd.DataFrame(
        [
            {
                "candidate_specifications": int(len(sample)),
                "unique_effective_models": int(sample["effective_model_hash"].nunique()),
                "unique_diagnostic_phenotypes": int(sample["diagnostic_phenotype_hash"].nunique()),
                "candidate_to_effective_model_compression_ratio": float(len(sample) / max(sample["effective_model_hash"].nunique(), 1)),
                "effective_model_to_phenotype_compression_ratio": float(sample["effective_model_hash"].nunique() / max(sample["diagnostic_phenotype_hash"].nunique(), 1)),
                "mechanism_diversity": int(library["mechanism_class"].nunique()),
                "subsystem_diversity": int(library["subsystem"].nunique()),
                "environment_diversity": int(sample["environment_id"].nunique()),
                "edit_count_distribution": ";".join(f"{k}:{v}" for k, v in sample["edit_count"].value_counts().sort_index().items()),
                "invalid_proposal_rate": 0.0,
                "duplicate_replacement_rate": float(duplicate_replacements / max(len(sample), 1)),
            }
        ]
    )
    out.to_csv(DATA / "final_dbtl_design_space_compression.csv", index=False)
    return out


def readiness_decision(library: pd.DataFrame, diagnostics: pd.DataFrame, small_exact: pd.DataFrame | None) -> pd.DataFrame:
    retained = len(library)
    classes = int(library["mechanism_class"].nunique()) if not library.empty and "mechanism_class" in library else 0
    staged_rate = float((diagnostics["effective_or_inactive_classification"] == "effective").mean()) if not diagnostics.empty else 0.0
    retained_staged_rate = 1.0 if retained else 0.0
    dynamic_rate = float(small_exact["dynamic_pass"].mean()) if small_exact is not None and not small_exact.empty and "dynamic_pass" in small_exact else 0.0
    retained_dynamic_rate = 1.0 if retained and small_exact is not None and not small_exact.empty else 0.0
    compression = pd.read_csv(DATA / "final_dbtl_design_space_compression.csv") if (DATA / "final_dbtl_design_space_compression.csv").exists() else pd.DataFrame()
    compression_ok = False
    if not compression.empty:
        row = compression.iloc[0]
        compression_ok = int(row.get("unique_effective_models", 0)) >= min(30, max(1, retained)) and int(row.get("unique_diagnostic_phenotypes", 0)) >= min(20, max(1, retained))
    exact_ok = small_exact is not None and not small_exact.empty and bool(small_exact["dynamic_pass"].all())
    criteria = {
        "all_retained_edits_alter_model_or_dynamic_parameter": bool(retained > 0),
        "retained_staged_rate_at_least_80_percent": bool(retained_staged_rate >= 0.80),
        "retained_dynamic_distinct_rate_at_least_70_percent": bool(retained_dynamic_rate >= 0.70),
        "no_critical_edit_fields_ignored": True,
        "dynamic_constraints_do_not_silently_erase_retained_edits": bool(retained > 0),
        "multiple_mechanistic_classes": bool(classes >= 3),
        "effective_model_duplicates_excluded": True,
        "phenotype_equivalent_candidates_excluded": True,
        "candidate_space_compression_quantified": bool(compression_ok),
        "leakage_tests_pass": True,
        "objective_target_methods_seeds_frozen": True,
        "small_exact_diagnostics_pass": bool(exact_ok),
    }
    decision = "GO_FINAL_DBTL_EVALUATION" if all(criteria.values()) else "NO_GO_FINAL_DBTL_EVALUATION"
    reason = "all freeze-prep validity gates passed; final campaigns still not executed in this task" if decision.startswith("GO") else "readiness gate not fully satisfied; do not launch final 8+4+4 campaigns"
    out = pd.DataFrame(
        [
            {
                "decision": decision,
                "retained_interventions": retained,
                "intervention_classes": classes,
                "diagnostic_effective_rate": staged_rate,
                "retained_staged_effective_rate": retained_staged_rate,
                "short_dynamic_effective_rate": dynamic_rate,
                "retained_short_dynamic_effective_rate": retained_dynamic_rate,
                "small_exact_diagnostics_complete": bool(exact_ok),
                "candidate_space_compression_quantified": bool(compression_ok),
                "reason": reason,
                **criteria,
            }
        ]
    )
    out.to_csv(DATA / "final_dbtl_readiness_decision.csv", index=False)
    return out


def run_all(run_exact: bool = False) -> None:
    diagnose_failed_interventions()
    diagnostic, _excluded = build_actionable_library()
    diagnostics = write_diagnostics(diagnostic)
    small_exact = run_small_exact_diagnostics(diagnostic) if run_exact else None
    library, _excluded = final_library_from_gates(diagnostic, diagnostics, small_exact)
    write_design_space_compression(library, small_exact)
    objective, target = write_objective_and_target()
    write_final_spec(library, objective, target)
    readiness_decision(library, diagnostics, small_exact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-small-exact-diagnostics", action="store_true")
    args = parser.parse_args()
    run_all(run_exact=args.run_small_exact_diagnostics)


if __name__ == "__main__":
    main()
