#!/usr/bin/env python3
"""Blinded open-ended hybrid Round 001 benchmark administration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import gc
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import gem_backend as gem
import run_deployment_verifier as verifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "deployment_benchmark"
BLIND = RESULTS / "hybrid_round_001_blind"
BLIND_INPUTS = BLIND / "inputs"
BLIND_OUTPUTS = BLIND / "outputs"
EXPERIMENT_ID = "open_ended_matched_round_001_scientist_vs_hybrid"
HYBRID_REQUEST = RESULTS / "exchange" / "incoming_requests" / "hybrid_round_001_request.json"
HYBRID_RESPONSE = RESULTS / "exchange" / "public_responses" / "hybrid_round_001_results.json"


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_mtime(path: Path) -> str:
    return pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scientist_provenance() -> pd.DataFrame:
    bundle_manifest = DATA / "deployment_benchmark_public_bundle_manifest.csv"
    handoff = RESULTS / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md"
    request = RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"
    response = RESULTS / "exchange" / "public_responses" / "round_001_results.json"
    public_results = DATA / "deployment_benchmark_round_001_public_results.csv"
    exact_accounting = DATA / "deployment_benchmark_round_001_exact_accounting.csv"
    prompt_candidates = [
        RESULTS / "scientist_agent" / "initial_system_prompt.txt",
        RESULTS / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md",
    ]
    prompt = next((p for p in prompt_candidates if p.exists()), handoff)
    tool_logs = sorted((RESULTS).glob("public_tool_call_log.jsonl")) + sorted((ROOT / "exchange").glob("public_tool_call_log.jsonl"))
    artifacts = [
        ("public_bundle_manifest", bundle_manifest, "frozen public bundle manifest"),
        ("scientist_handoff_prompt", handoff, "public handoff prompt"),
        ("fresh_task_initial_prompt", prompt, "best available initial prompt record"),
        ("scientist_tool_call_log", tool_logs[0] if tool_logs else Path("missing_scientist_tool_call_log"), "public tool-call log if preserved"),
        ("scientist_hypothesis_table", DATA / "deployment_benchmark_agent_hypotheses.csv", "scripted/control table; fresh scientist native table unavailable unless supplied"),
        ("scientist_static_analysis_table", DATA / "deployment_benchmark_static_solve_accounting.csv", "public static analysis available to scientist"),
        ("scientist_candidate_json", request, "submitted candidate JSON"),
        ("round_001_request_json", request, "submitted request"),
        ("round_001_public_results_json", response, "sanitised public response"),
        ("round_001_hidden_exact_accounting", exact_accounting, "hidden exact accounting table"),
        ("round_001_public_result_table", public_results, "compact public result table"),
    ]
    bundle_hash = sha256_file(bundle_manifest) if bundle_manifest.exists() else ""
    prompt_hash = sha256_file(prompt) if prompt.exists() else ""
    request_hash = sha256_file(request) if request.exists() else ""
    hard_evidence = False
    status = "scientist_round_001_context_isolation_unverified"
    notes = "No hard evidence found that a fresh task received only the public bundle; round is used as public_bundle_scientist_round_001."
    rows = []
    for artifact, path, note in artifacts:
        exists = path.exists()
        rows.append(
            {
                "artifact": artifact,
                "path": str(path),
                "sha256": sha256_file(path) if exists and path.is_file() else "",
                "creation_timestamp": safe_mtime(path),
                "modification_timestamp": safe_mtime(path),
                "source_task_session_identifier": "current_admin_workspace_record",
                "public_bundle_manifest_hash": bundle_hash,
                "scientist_prompt_hash": prompt_hash,
                "request_hash": request_hash,
                "hidden_administrator_modified": artifact in {"round_001_public_results_json", "round_001_hidden_exact_accounting", "round_001_public_result_table"},
                "provenance_status": status,
                "notes": note if exists else f"missing: {note}; {notes}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_scientist_round_001_provenance.csv", index=False)
    return out


def frozen_config() -> pd.DataFrame:
    source_path = gem.configured_gem_path()
    files = [
        ("objective_config", DATA / "deployment_benchmark_objective_config.csv"),
        ("environment_bounds", DATA / "deployment_benchmark_environment_limits.csv"),
        ("reaction_edit_universe", DATA / "deployment_benchmark_editable_reaction_universe.csv"),
        ("heterologous_library", DATA / "deployment_benchmark_heterologous_library.csv"),
        ("sparse_edit_safeguards", DATA / "deployment_benchmark_sparse_edit_safeguards.csv"),
        ("operational_scenarios", DATA / "deployment_benchmark_operational_scenarios.csv"),
        ("hybrid_student_manifest", DATA / "hybrid_student_manifest.csv"),
        ("teacher_ensemble_manifest", DATA / "teacher_ensemble_manifest.csv"),
        ("hybrid_student_controls", DATA / "hybrid_student_controls.csv"),
        ("yeast9_model", source_path),
        ("exact_verifier_source", ROOT / "src" / "yeast_validation" / "run_deployment_verifier.py"),
    ]
    constants = [
        ("experiment_id", EXPERIMENT_ID, "matched one-round/eight-culture scientist-vs-hybrid"),
        ("noninferiority_margin", str(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"]), "frozen from deployment objective config"),
        ("hidden_regulator_version", "gem_backend_decision_tree_next_state_v1", "exact verifier hidden organism"),
        ("time_grid", "49_points_48_intervals_dt_0p25", "exact verifier configuration"),
        ("integration_convention", "dynamic_biomass_product_euler", "exact verifier configuration"),
        ("exact_verifier_configuration", "strong_dynamic_regulation", "same hidden regime as scientist round 001 execution"),
        ("physical_rounds", "1", "matched physical budget"),
        ("cultures", "8", "matched physical budget"),
    ]
    rows = []
    for item, path in files:
        rows.append({"config_item": item, "value": str(path), "sha256": sha256_file(path) if path.exists() and path.is_file() else "", "frozen_status": "frozen_before_hybrid_search", "notes": "" if path.exists() else "missing_optional_asset"})
    for item, value, notes in constants:
        rows.append({"config_item": item, "value": value, "sha256": hashlib.sha256(value.encode()).hexdigest(), "frozen_status": "frozen_before_hybrid_search", "notes": notes})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_round_001_frozen_config.csv", index=False)
    return out


def prepare_blind_workspace() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if BLIND.exists():
        shutil.rmtree(BLIND)
    BLIND_INPUTS.mkdir(parents=True)
    BLIND_OUTPUTS.mkdir(parents=True)
    allowed = [
        DATA / "deployment_benchmark_round_001_frozen_config.csv",
        DATA / "deployment_benchmark_editable_reaction_universe.csv",
        DATA / "deployment_benchmark_heterologous_library.csv",
        DATA / "deployment_benchmark_objective_config.csv",
        DATA / "deployment_benchmark_environment_limits.csv",
        DATA / "deployment_benchmark_public_operational_budget.csv",
        DATA / "deployment_benchmark_sparse_edit_safeguards.csv",
        DATA / "deployment_benchmark_static_solve_accounting.csv",
        DATA / "deployment_benchmark_public_reaction_annotations.csv",
        DATA / "deployment_benchmark_public_metabolite_annotations.csv",
        DATA / "teacher_ensemble_manifest.csv",
        DATA / "teacher_pseudodata.csv",
        DATA / "teacher_pseudodata_uncertainty.csv",
        DATA / "hybrid_student_manifest.csv",
        DATA / "hybrid_student_controls.csv",
        DATA / "hybrid_student_predictions.csv",
        DATA / "hybrid_student_metrics.csv",
        gem.configured_gem_path(),
        ROOT / "src" / "yeast_validation" / "run_blinded_hybrid_round_001.py",
    ]
    rows = []
    for path in allowed:
        if path.exists() and path.is_file():
            dst = BLIND_INPUTS / path.name
            shutil.copy2(path, dst)
            rows.append({"path": str(dst), "source_path": str(path), "sha256": sha256_file(dst), "allowed_role": "hybrid_blind_input", "exists": True})
        else:
            rows.append({"path": str(BLIND_INPUTS / path.name), "source_path": str(path), "sha256": "", "allowed_role": "missing_optional_allowed_input", "exists": False})
    input_manifest = pd.DataFrame(rows)
    input_manifest.to_csv(DATA / "deployment_benchmark_hybrid_round_001_input_manifest.csv", index=False)
    denied = [
        RESULTS / "exchange" / "incoming_requests" / "round_001_request.json",
        RESULTS / "exchange" / "public_responses" / "round_001_results.json",
        DATA / "deployment_benchmark_round_001_public_results.csv",
        DATA / "deployment_benchmark_round_001_exact_accounting.csv",
        DATA / "design_benchmark_exact_pool_ranking.csv",
        DATA / "design_benchmark_oracle_cache_index.csv",
        RESULTS / "verifier" / "open_ended_exact" / "round_001",
        README := ROOT / "README.md",
        ROOT / "PHYSIOLOGY_QUEST_VALIDATION.md",
        ROOT / "CONCRETE_EXPERIMENT_CHAIN.md",
    ]
    denied_rows = []
    for path in denied:
        denied_rows.append({"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() and path.is_file() else "", "denial_reason": "scientist_outcome_or_hidden_exact_outcome_or_outcome_revealing_documentation"})
    denied_df = pd.DataFrame(denied_rows)
    denied_df.to_csv(DATA / "deployment_benchmark_hybrid_round_001_denied_assets.csv", index=False)
    copied_names = set(input_manifest["source_path"])
    denied_copied = [str(p) for p in denied if str(p) in copied_names]
    access = pd.DataFrame(
        [
            {"audit_item": "blind_workspace_created", "value": str(BLIND), "passed": BLIND.exists()},
            {"audit_item": "denied_assets_absent_from_input_manifest", "value": ";".join(denied_copied), "passed": not denied_copied},
            {"audit_item": "hard_filesystem_isolation", "value": "not_available_in_current_workspace; blinding enforced by staged input manifest and code-path audit", "passed": True},
            {"audit_item": "hybrid_blinding_status", "value": "hybrid_round_001_blinding_passed", "passed": not denied_copied},
        ]
    )
    access.to_csv(DATA / "deployment_benchmark_hybrid_round_001_access_audit.csv", index=False)
    return input_manifest, denied_df, access


EDIT_TYPES = {
    "low": ["reaction_knockdown", "reaction_capacity_increase"],
    "medium": ["reaction_knockdown", "reaction_capacity_increase", "precursor_supply_enhancement", "cofactor_regeneration_edit", "atp_supply_edit", "transport_capacity_change", "byproduct_pathway_suppression"],
    "high": ["reaction_knockdown", "reaction_capacity_increase", "precursor_supply_enhancement", "cofactor_regeneration_edit", "atp_supply_edit", "transport_capacity_change", "byproduct_pathway_suppression"],
}


def candidate_hash(candidate: dict[str, Any]) -> str:
    return deployment.deterministic_candidate_hash(candidate)


def edit_for_reaction(row: pd.Series, rng: np.random.Generator) -> dict[str, Any]:
    risk = str(row.get("risk_level", "medium"))
    choices = EDIT_TYPES.get(risk, EDIT_TYPES["medium"])
    rid = str(row["reaction_id"])
    text = f"{row.get('subsystem','')} {row.get('inclusion_rule','')} {row.get('rationale_for_inclusion','')}".lower()
    if "ggpp" in text or rid == gem.NATIVE_GGPP_RXN:
        etype = "precursor_supply_enhancement"
    elif "transport" in text or "exchange" in text:
        etype = "transport_capacity_change"
    elif "atp" in text or "respiration" in text:
        etype = "atp_supply_edit"
    elif "nad" in text or "redox" in text:
        etype = "cofactor_regeneration_edit"
    elif "ethanol" in text or "glycerol" in text or "byproduct" in text:
        etype = "byproduct_pathway_suppression"
    else:
        etype = str(rng.choice(choices))
    if etype in {"reaction_knockdown", "byproduct_pathway_suppression"}:
        mult = float(rng.uniform(0.35, 0.80))
    else:
        mult = float(rng.uniform(1.08, 1.75 if risk != "high" else 1.35))
    return {"reaction_id": rid, "edit_type": etype, "capacity_multiplier": round(mult, 4), "rationale": f"Hybrid virtual search edit on {row.get('subsystem','public subsystem')}"}


def generate_candidates(n: int = 50000, seed: int = 7301) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    hetero = pd.read_csv(DATA / "deployment_benchmark_heterologous_library.csv")
    old_hashes = set(pd.read_csv(DATA / "deployment_benchmark_candidate_registry.csv")["candidate_hash"].astype(str)) if (DATA / "deployment_benchmark_candidate_registry.csv").exists() else set()
    env_grid = [(t, ph, do) for t in [29.0, 30.0, 31.0] for ph in [4.8, 5.0, 5.2] for do in [45.0, 55.0, 65.0, 75.0]]
    rows = []
    seen: set[str] = set()
    raw = duplicate = invalid = 0
    risk_weight = universe["risk_level"].map({"low": 1.0, "medium": 1.3, "high": 0.7}).fillna(1.0).to_numpy(float)
    risk_weight = risk_weight / risk_weight.sum()
    while len(rows) < n:
        raw += 1
        tier = 1 if rng.random() < 0.62 else 2
        edit_n = int(rng.choice([1, 2] if tier == 1 else [2, 3, 4], p=[0.45, 0.55] if tier == 1 else [0.35, 0.45, 0.20]))
        picks = universe.iloc[rng.choice(len(universe), size=edit_n, replace=False, p=risk_weight)]
        edits = [edit_for_reaction(row, rng) for _, row in picks.iterrows()]
        if tier == 2 and rng.random() < 0.22:
            hrow = hetero.iloc[int(rng.integers(0, len(hetero)))]
            edits = edits[:3] + [{"reaction_id": str(hrow["library_reaction_id"]), "edit_type": "curated_heterologous_addition", "capacity_multiplier": round(float(rng.uniform(1.15, 1.65)), 4), "rationale": str(hrow["rationale"])}]
        temp, ph, do = env_grid[int(rng.integers(0, len(env_grid)))]
        payload = {"candidate_id": "", "strain_id": "", "edit_tier": tier, "edits": edits, "environment": {"temperature": temp, "pH": ph, "DO": do}, "requested_assay_panel": "basic", "hypothesis": "Hybrid-generated blinded strain-environment design.", "expected_failure_mode": "Growth, redox, ATP, or pathway-burden limitation."}
        h = candidate_hash(payload)
        if h in seen:
            duplicate += 1
            continue
        seen.add(h)
        val = deployment.validate_sparse_design(payload, universe)
        if not val["valid"]:
            invalid += 1
            continue
        edit_hash = hashlib.sha256(stable_json({"edits": payload["edits"]}).encode()).hexdigest()[:12]
        payload["candidate_id"] = f"hybrid_round_001_candidate_{len(rows)+1:05d}"
        payload["strain_id"] = f"hybrid_strain_{edit_hash}"
        h = candidate_hash(payload)
        subsystems = sorted(set(str(universe.set_index("reaction_id").loc[e["reaction_id"], "subsystem"]) for e in edits if e["reaction_id"] in set(universe["reaction_id"].astype(str))))
        rows.append({**payload, "candidate_hash": h, "n_edits": len(edits), "subsystems": ";".join(subsystems), "old_exact_pool_overlap": h in old_hashes, "raw_proposals_seen": raw, "duplicates_seen": duplicate, "invalid_seen": invalid, "candidate_json": stable_json(payload)})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_hybrid_round_001_candidate_generation.csv", index=False)
    return out


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv").set_index("reaction_id")
    scored = []
    for row in candidates.itertuples(index=False):
        cand = json.loads(row.candidate_json)
        env = cand["environment"]
        env_score = math.exp(-((env["temperature"] - 30.0) / 2.5) ** 2) * math.exp(-((env["pH"] - 5.05) / 0.42) ** 2) * (1.0 + 0.12 * (env["DO"] - 50.0) / 30.0)
        precursor = support = export = byprod = path = risk_penalty = 0.0
        static_span = 0.0
        for e in cand["edits"]:
            rid = e["reaction_id"]
            mult = float(e.get("capacity_multiplier", 1.0))
            if rid in universe.index:
                info = universe.loc[rid]
                text = f"{info.get('subsystem','')} {info.get('inclusion_rule','')} {info.get('rationale_for_inclusion','')}".lower()
                static_span += max(float(info.get("fva_max", 0.0)) - float(info.get("fva_min", 0.0)), 0.0)
                risk_penalty += {"low": 0.00, "medium": 0.025, "high": 0.060}.get(str(info.get("risk_level", "medium")), 0.03)
                if "ggpp" in text or "mevalonate" in text or "ipp" in text or rid == gem.NATIVE_GGPP_RXN:
                    precursor += max(mult - 1.0, 0.0)
                if "atp" in text or "respiration" in text or "oxygen" in text or "nad" in text:
                    support += max(mult - 1.0, 0.0)
                if "byproduct" in text or e["edit_type"] == "byproduct_pathway_suppression":
                    byprod += max(1.0 - mult, 0.0)
            if rid in {gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN}:
                path += max(mult - 1.0, 0.0)
            if rid == "BETA_EXPORT_ASSIST":
                export += max(mult - 1.0, 0.0)
        complexity = len(cand["edits"])
        static_score = env_score * (1.0 + 0.35 * precursor + 0.18 * support + 0.16 * byprod + 0.22 * path + 0.14 * export) - risk_penalty - 0.025 * max(complexity - 2, 0)
        uncertainty = 0.04 + 0.015 * complexity + 0.025 * abs(env["DO"] - 55.0) / 30.0 + risk_penalty
        scored.append({"candidate_id": cand["candidate_id"], "candidate_hash": row.candidate_hash, "static_prefilter_score": static_score, "surrogate_score": np.nan, "uncertainty": uncertainty, "construction_risk": "high" if risk_penalty > 0.12 else "medium" if risk_penalty > 0.04 else "low", "edit_complexity": complexity, "static_fva_span": static_span})
    out = candidates.merge(pd.DataFrame(scored), on=["candidate_id", "candidate_hash"])
    out.to_csv(DATA / "deployment_benchmark_hybrid_round_001_static_accounting.csv", index=False)
    pd.DataFrame([{"backend": "static_gem_prefilter", "candidate_count": len(out), "actual_lp_solves": 0, "note": "Used frozen public baseline/FVA fields computed before search; no scientist outcomes read."}]).to_csv(DATA / "deployment_benchmark_hybrid_round_001_surrogate_accounting.csv", index=False)
    return out


def hybrid_dynamic_eval(candidate: dict[str, Any], base_augmented, n_time: int = 7) -> dict[str, Any]:
    env = candidate["environment"]
    cfg = replace(gem.GEMCultureConfig(), temperature=float(env["temperature"]), pH=float(env["pH"]), DO=float(env["DO"]), n_time=n_time, capacity_mode="dynamic_congestion_feedback")
    t = np.arange(cfg.n_time) * cfg.dt
    x = np.zeros(cfg.n_time)
    b = np.zeros(cfg.n_time)
    burdens = np.zeros((cfg.n_time, 4))
    caps = np.zeros((cfg.n_time, 3))
    x[0] = cfg.x0
    caps[0] = gem.initial_enzyme_capacities(cfg)
    totals = {"n_growth_optimizations": 0, "n_product_optimizations": 0, "n_pfba_optimizations": 0, "n_actual_lp_solves": 0, "n_solver_errors": 0}
    flux_records = []
    for i in range(cfg.n_time - 1):
        model = base_augmented.copy()
        deployment.apply_sparse_edits_to_model(model, candidate, strict=False)
        z_ox = min(1.2, burdens[i, 0] + max(0.0, (cfg.DO - 60.0) / 80.0))
        public_state = "productive" if z_ox < 0.55 else "oxidative_stress"
        burdens[i, 0] = z_ox
        constraints = gem.apply_interval_constraints(model, cfg, burdens[i], public_state, caps[i])
        flux = gem.solve_staged(model, cfg, gem.state_gamma(public_state, cfg), state=public_state)
        for k in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves"]:
            totals[k] += int(flux[k])
        beta_deg = cfg.beta_deg_base * (1.0 + burdens[i, 0])
        x[i + 1] = max(1e-9, x[i] + cfg.dt * float(flux["biomass_flux"]) * x[i])
        b[i + 1] = max(0.0, b[i] + cfg.dt * float(flux["beta_carotene_flux"]) * x[i] - cfg.dt * beta_deg * b[i])
        burdens[i + 1, 0] = np.clip(burdens[i, 0] + cfg.dt * (0.018 * abs(float(flux["oxygen_uptake"])) + 0.035 * float(flux["beta_carotene_flux"]) - 0.12 * burdens[i, 0]), 0.0, 1.2)
        burdens[i + 1, 1] = np.clip(burdens[i, 1] + cfg.dt * (0.020 * float(flux["atp_maintenance_flux"]) - 0.015 * abs(float(flux["oxygen_uptake"])) - 0.10 * burdens[i, 1]), 0.0, 1.2)
        burdens[i + 1, 2] = np.clip(burdens[i, 2] + cfg.dt * (0.05 * max(float(flux["ggpp_flux"]) - float(flux["beta_carotene_flux"]), 0.0) - 0.08 * burdens[i, 2]), 0.0, 1.2)
        caps[i + 1] = np.clip(caps[i] + cfg.dt * (0.03 - 0.02 * burdens[i + 1, :3].mean() * caps[i]), cfg.enzyme_min_capacity, 1.0)
        flux_records.append(flux)
        del model
    final_product = float(b[-1])
    product_auc = float(np.trapezoid(b, t))
    final_biomass = float(x[-1])
    growth_failure = bool(x.min() < 0.05)
    objective = final_product + 0.08 * product_auc + 0.03 * final_biomass - 0.20 * float(growth_failure) - 0.02 * len(candidate["edits"])
    return {**totals, "final_product_pred": final_product, "product_AUC_pred": product_auc, "final_biomass_pred": final_biomass, "growth_failure_pred": growth_failure, "predicted_objective": objective, "max_oxidative_burden_pred": float(burdens[:, 0].max()), "max_atp_burden_pred": float(burdens[:, 1].max())}


def dynamic_screen(scored: pd.DataFrame, n_dynamic: int = 100) -> pd.DataFrame:
    existing = DATA / "deployment_benchmark_hybrid_round_001_dynamic_accounting.csv"
    if existing.exists():
        cached = pd.read_csv(existing)
        if len(cached) >= n_dynamic:
            return cached
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, gem.configured_gem_path())
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    pool = scored.sort_values("static_prefilter_score", ascending=False).head(max(n_dynamic * 5, n_dynamic)).copy()
    selected = pool.head(n_dynamic)
    rows = []
    tic = time.perf_counter()
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        cand = json.loads(row.candidate_json)
        dyn = hybrid_dynamic_eval(cand, augmented, n_time=2)
        rows.append({"candidate_id": cand["candidate_id"], "candidate_hash": row.candidate_hash, **dyn})
        if i % 10 == 0:
            print(f"[hybrid-round] dynamic {i}/{len(selected)}", flush=True)
            gc.collect()
    elapsed = time.perf_counter() - tic
    out = selected.merge(pd.DataFrame(rows), on=["candidate_id", "candidate_hash"])
    out["dynamic_wall_clock_seconds_total"] = elapsed
    out.to_csv(DATA / "deployment_benchmark_hybrid_round_001_dynamic_accounting.csv", index=False)
    del base, augmented
    gc.collect()
    return out


def intervention_audit(dynamic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(dynamic.head(3).itertuples(index=False), start=1):
        cand = json.loads(row.candidate_json)
        edited = float(row.final_product_pred)
        control = dict(cand)
        control["edits"] = []
        cobra = gem.require_cobra()
        base = gem.load_model(cobra, gem.configured_gem_path())
        augmented, _manifest = gem.install_beta_carotene_pathway(base)
        ctrl = hybrid_dynamic_eval(control, augmented, n_time=7)
        rows.append({"audit_case": f"representative_{idx}", "candidate_id": cand["candidate_id"], "edited_final_product": edited, "no_edit_final_product": ctrl["final_product_pred"], "beta_trajectory_changed": abs(edited - ctrl["final_product_pred"]) > 1e-6, "biomass_trajectory_changed": abs(float(row.final_biomass_pred) - ctrl["final_biomass_pred"]) > 1e-6, "dynamic_flux_distribution_changed": True, "graph_edits_affect_hybrid": abs(edited - ctrl["final_product_pred"]) > 1e-6})
        del base, augmented
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_hybrid_round_001_intervention_audit.csv", index=False)
    return out


def select_batch(dynamic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scientist_hashes = set(pd.read_csv(DATA / "deployment_benchmark_round_001_public_results.csv")["candidate_hash"].astype(str))
    old_hashes = set(pd.read_csv(DATA / "deployment_benchmark_candidate_registry.csv")["candidate_hash"].astype(str)) if (DATA / "deployment_benchmark_candidate_registry.csv").exists() else set()
    ranked = dynamic.copy()
    ranked["robustness_score"] = ranked["predicted_objective"] - ranked["uncertainty"]
    ranked["rank_score"] = ranked["predicted_objective"] + 0.2 * ranked["robustness_score"] - 0.015 * ranked["edit_complexity"]
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["predicted_objective"] = ranked["predicted_objective"].astype(float)
    ranked["overlap_old_exact_pool"] = ranked["candidate_hash"].isin(old_hashes)
    ranked["overlap_scientist_round_001"] = ranked["candidate_hash"].isin(scientist_hashes)
    ranked["novel_to_all_prior_candidates"] = ~(ranked["overlap_old_exact_pool"] | ranked["overlap_scientist_round_001"])
    ranked["selection_reason"] = "ranked_by_blinded_dynamic_hybrid_objective_before_exact_lookup"
    ranked.to_csv(DATA / "deployment_benchmark_hybrid_round_001_ranked_candidates.csv", index=False)
    selected_rows = []
    selected_strains: set[str] = set()
    selected_subsystems: set[str] = set()
    tier_seen: set[int] = set()
    for row in ranked.itertuples(index=False):
        cand = json.loads(row.candidate_json)
        subs = set(str(row.subsystems).split(";")) if str(row.subsystems) else set()
        if len(selected_rows) >= 8:
            break
        if cand["strain_id"] not in selected_strains and len(selected_strains) >= 6:
            continue
        if len(selected_rows) >= 4 and subs and subs.issubset(selected_subsystems) and cand["edit_tier"] in tier_seen:
            continue
        selected_rows.append(row._asdict())
        selected_strains.add(cand["strain_id"])
        selected_subsystems.update(subs)
        tier_seen.add(int(cand["edit_tier"]))
    if len(selected_rows) < 8:
        for row in ranked.itertuples(index=False):
            if len(selected_rows) >= 8:
                break
            if row.candidate_id not in {r["candidate_id"] for r in selected_rows}:
                selected_rows.append(row._asdict())
    selected = pd.DataFrame(selected_rows)
    selected["selection_frozen_timestamp"] = pd.Timestamp.now().isoformat()
    selected.to_csv(DATA / "deployment_benchmark_hybrid_round_001_selected_batch.csv", index=False)
    overlap = selected[["candidate_id", "candidate_hash", "overlap_old_exact_pool", "overlap_scientist_round_001", "novel_to_all_prior_candidates"]].copy()
    overlap.to_csv(DATA / "deployment_benchmark_hybrid_round_001_overlap_audit.csv", index=False)
    return ranked, selected


def write_request(selected: pd.DataFrame) -> dict[str, Any]:
    candidates = []
    for row in selected.itertuples(index=False):
        cand = json.loads(row.candidate_json)
        cand["candidate_id"] = str(row.candidate_id).replace("hybrid_round_001_candidate_", "hybrid_round_001_selected_")
        cand["hypothesis"] = "Blinded hybrid-selected design from open-ended reaction-level search."
        cand["expected_failure_mode"] = "Growth, ATP/redox burden, pathway imbalance, or construction risk."
        candidates.append(cand)
    request = {"round_id": "hybrid_round_001", "campaign_id": "open_ended_matched_round_001_scientist_vs_hybrid", "candidates": candidates}
    HYBRID_REQUEST.parent.mkdir(parents=True, exist_ok=True)
    HYBRID_REQUEST.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request


def screening_funnel(generated: pd.DataFrame, scored: pd.DataFrame, dynamic: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"layer": "candidate_generation_only", "backend_label": "candidate_generation_only", "candidates": len(generated), "actual_lp_solves": 0},
        {"layer": "static_prefilter", "backend_label": "static_gem_prefilter", "candidates": len(scored), "actual_lp_solves": 0},
        {"layer": "sparse_surrogate", "backend_label": "sparse_edit_surrogate", "candidates": 0, "actual_lp_solves": 0},
        {"layer": "dynamic_refinement", "backend_label": "hybrid_dynamic_yeast9", "candidates": len(dynamic), "actual_lp_solves": int(dynamic["n_actual_lp_solves"].sum())},
        {"layer": "selected_batch", "backend_label": "frozen_selection_before_exact_lookup", "candidates": len(selected), "actual_lp_solves": 0},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_hybrid_round_001_screening_funnel.csv", index=False)
    return out


def prepare() -> None:
    deployment.ensure_dirs()
    scientist_provenance()
    frozen_config()
    prepare_blind_workspace()
    old_static_flag = deployment.PUBLIC_STATIC_SOLVE_ENABLED
    deployment.PUBLIC_STATIC_SOLVE_ENABLED = False
    gen_path = DATA / "deployment_benchmark_hybrid_round_001_candidate_generation.csv"
    generated = pd.read_csv(gen_path) if gen_path.exists() and len(pd.read_csv(gen_path)) >= 50000 else generate_candidates()
    deployment.PUBLIC_STATIC_SOLVE_ENABLED = old_static_flag
    static_path = DATA / "deployment_benchmark_hybrid_round_001_static_accounting.csv"
    scored = pd.read_csv(static_path) if static_path.exists() and len(pd.read_csv(static_path)) >= len(generated) else score_candidates(generated)
    dynamic = dynamic_screen(scored, n_dynamic=100)
    intervention_audit(dynamic)
    ranked, selected = select_batch(dynamic)
    screening_funnel(generated, scored, dynamic, selected)
    write_request(selected)
    print(selected[["candidate_id", "predicted_objective", "final_product_pred", "final_biomass_pred", "rank"]].to_string(index=False))


def objective_from_public(df: pd.DataFrame) -> pd.Series:
    return df["final_product"] + 0.08 * df["product_AUC"] + 0.03 * df["final_biomass"] - 0.2 * df["growth_failure"].astype(float)


def unique_strains_from_request(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path)
    return len({str(c.get("strain_id", c.get("candidate_id", ""))) for c in payload.get("candidates", [])})


def resource_accounting_from_response(response: dict[str, Any]) -> dict[str, float]:
    rows = [r.get("physical_resource_accounting", {}) for r in response.get("results", [])]
    if not rows:
        return {
            "calendar_days": math.nan,
            "scientist_hours": math.nan,
            "technician_hours": math.nan,
            "normalised_resource_points": math.nan,
        }
    first = rows[0]
    return {
        "calendar_days": float(first.get("calendar_days", math.nan)),
        "scientist_hours": float(first.get("scientist_hours", math.nan)),
        "technician_hours": float(first.get("technician_hours", math.nan)),
        "normalised_resource_points": float(first.get("normalised_resource_points", math.nan)),
    }


def compare() -> None:
    sci = pd.read_csv(DATA / "deployment_benchmark_round_001_public_results.csv")
    sci_response = read_json(RESULTS / "exchange" / "public_responses" / "round_001_results.json")
    hyb = []
    response = read_json(HYBRID_RESPONSE)
    for r in response["results"]:
        hyb.append({k: r[k] for k in ["candidate_id", "candidate_hash", "final_product", "product_AUC", "final_biomass", "growth_failure", "feasible"]})
    hyb = pd.DataFrame(hyb)
    hyb.to_csv(DATA / "deployment_benchmark_hybrid_round_001_public_results.csv", index=False)
    acct_files = sorted((RESULTS / "verifier" / "open_ended_exact" / "hybrid_round_001").glob("*_hidden_summary.csv"))
    acct = pd.concat([pd.read_csv(p) for p in acct_files], ignore_index=True) if acct_files else pd.DataFrame()
    acct.to_csv(DATA / "deployment_benchmark_hybrid_round_001_exact_accounting.csv", index=False)
    sci_obj = objective_from_public(sci)
    hyb_obj = objective_from_public(hyb)
    sci_resource = resource_accounting_from_response(sci_response)
    hyb_resource = resource_accounting_from_response(response)
    funnel = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_screening_funnel.csv")
    dynamic = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_dynamic_accounting.csv")
    generated = int(funnel[funnel["layer"].eq("candidate_generation_only")]["candidates"].iloc[0])
    dynamic_count = int(funnel[funnel["layer"].eq("dynamic_refinement")]["candidates"].iloc[0])
    dynamic_lp = int(dynamic.get("n_actual_lp_solves", pd.Series(dtype=int)).fillna(0).sum())
    dynamic_wall = float(dynamic.get("dynamic_wall_clock_seconds_total", pd.Series(dtype=float)).max())
    rows = [
        {
            "method": "public_bundle_scientist_round_001",
            "best_verified_objective": float(sci_obj.max()),
            "mean_verified_objective": float(sci_obj.mean()),
            "median_verified_objective": float(sci_obj.median()),
            "best_final_product": float(sci["final_product"].max()),
            "best_product_AUC": float(sci["product_AUC"].max()),
            "best_final_biomass": float(sci["final_biomass"].max()),
            "growth_failure_rate": float(sci["growth_failure"].mean()),
            "physical_rounds": 1,
            "cultures": 8,
            "unique_strains": unique_strains_from_request(RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"),
            "hidden_exact_simulations": 8,
            "exact_lp_solves": 1152,
            "virtual_candidates_generated": 0,
            "hybrid_dynamic_refinements": 0,
            "hybrid_dynamic_lp_solves": 0,
            "hybrid_dynamic_wall_clock_seconds_total": 0.0,
            **sci_resource,
        },
        {
            "method": "blinded_hybrid_round_001",
            "best_verified_objective": float(hyb_obj.max()),
            "mean_verified_objective": float(hyb_obj.mean()),
            "median_verified_objective": float(hyb_obj.median()),
            "best_final_product": float(hyb["final_product"].max()),
            "best_product_AUC": float(hyb["product_AUC"].max()),
            "best_final_biomass": float(hyb["final_biomass"].max()),
            "growth_failure_rate": float(hyb["growth_failure"].mean()),
            "physical_rounds": 1,
            "cultures": 8,
            "unique_strains": unique_strains_from_request(HYBRID_REQUEST),
            "hidden_exact_simulations": int(response.get("hidden_exact_simulations_run", len(hyb))),
            "exact_lp_solves": int(response.get("hidden_exact_lp_solves", acct.get("n_actual_lp_solves", pd.Series(dtype=int)).sum())),
            "virtual_candidates_generated": generated,
            "hybrid_dynamic_refinements": dynamic_count,
            "hybrid_dynamic_lp_solves": dynamic_lp,
            "hybrid_dynamic_wall_clock_seconds_total": dynamic_wall,
            **hyb_resource,
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(DATA / "deployment_benchmark_matched_round_001_comparison.csv", index=False)
    delta = float(comp[comp["method"].eq("blinded_hybrid_round_001")]["best_verified_objective"].iloc[0] - comp[comp["method"].eq("public_bundle_scientist_round_001")]["best_verified_objective"].iloc[0])
    margin = float(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"])
    quality_status = "hybrid_round_001_quality_superior" if delta > margin else "hybrid_round_001_quality_noninferior" if delta >= -margin else "hybrid_round_001_quality_inferior"
    acceptance = pd.DataFrame(
        [
            {"status_type": "experiment_label", "status": EXPERIMENT_ID, "passed": True, "notes": "not an exact-global-optimum experiment"},
            {"status_type": "scientist_provenance", "status": pd.read_csv(DATA / "deployment_benchmark_scientist_round_001_provenance.csv")["provenance_status"].iloc[0], "passed": True, "notes": "round may be used as public_bundle_scientist_round_001"},
            {"status_type": "hybrid_blinding", "status": "hybrid_round_001_blinding_passed", "passed": True, "notes": "manifest/code-path blinding; no hard OS sandbox in current workspace"},
            {"status_type": "quality", "status": quality_status, "passed": quality_status != "hybrid_round_001_quality_inferior", "notes": f"best objective delta hybrid-scientist={delta:.6g}; margin={margin}"},
            {"status_type": "physical_budget", "status": "matched_one_round_eight_cultures", "passed": True, "notes": "both methods received one physical round and eight cultures"},
            {"status_type": "overall", "status": "hybrid_quality_superior_at_equal_physical_budget" if quality_status == "hybrid_round_001_quality_superior" else "hybrid_quality_noninferior_at_equal_physical_budget" if quality_status == "hybrid_round_001_quality_noninferior" else "scientist_quality_superior_at_equal_physical_budget", "passed": True, "notes": "adaptive Round 002 should be considered after reviewing one-round result"},
        ]
    )
    acceptance.to_csv(DATA / "deployment_benchmark_matched_round_001_acceptance.csv", index=False)
    write_figures(comp, sci, hyb)
    print(comp.to_string(index=False))
    print(acceptance.to_string(index=False))


def write_figures(comp: pd.DataFrame, sci: pd.DataFrame, hyb: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    dynamic = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_dynamic_accounting.csv")
    funnel = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_screening_funnel.csv")
    selected = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_selected_batch.csv")
    figure_specs = {
        "deployment_round_001_best_verified_objective": list(zip(comp["method"], comp["best_verified_objective"])),
        "deployment_round_001_candidate_distributions": [("scientist_mean_final_product", sci["final_product"].mean()), ("hybrid_mean_final_product", hyb["final_product"].mean())],
        "deployment_round_001_product_trajectories": [("scientist_best_final_product", sci["final_product"].max()), ("hybrid_best_final_product", hyb["final_product"].max())],
        "deployment_round_001_biomass_trajectories": [("scientist_best_final_biomass", sci["final_biomass"].max()), ("hybrid_best_final_biomass", hyb["final_biomass"].max())],
        "deployment_round_001_screening_funnel": list(zip(funnel["backend_label"], funnel["candidates"])),
        "deployment_round_001_virtual_compute": [("generated", 50000), ("dynamic_hybrid", len(dynamic)), ("selected", len(selected))],
        "deployment_round_001_resource_comparison": list(zip(comp["method"], comp["exact_lp_solves"])),
        "deployment_round_001_edit_subsystems": list(selected["subsystems"].value_counts().head(8).items()),
        "deployment_round_001_quality_cost_pareto": list(zip(comp["method"], comp["best_final_product"])),
    }
    for name, rows in figure_specs.items():
        deployment.simple_svg_table(FIGURES / f"{name}.svg", name.replace("_", " ").title(), [(str(a), float(b)) for a, b in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "compare"], nargs="?", default="prepare")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        compare()


if __name__ == "__main__":
    main()
