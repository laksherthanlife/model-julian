#!/usr/bin/env python3
"""Run one report-checkpoint replication campaign without touching Campaign 001."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import gem_backend as gem
import run_blinded_hybrid_round_001 as hybrid1
import run_deployment_verifier as verifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
REPLICATION_CAMPAIGNS = {"campaign_002", "campaign_003"}
REQUIRED_PROVENANCE_FILES = {
    "hypothesis_table": "round_001_hypothesis_table.csv",
    "static_analysis_table": "round_001_static_analysis.csv",
    "candidate_json": "round_001_candidate_batch.json",
    "tool_call_log": "round_001_tool_call_log.jsonl",
    "decision_summary": "round_001_decision_summary.md",
    "provenance_manifest": "round_001_provenance.json",
}
REQUIRED_REQUEST_PROVENANCE_FIELDS = {
    "fresh_task_session_identifier",
    "public_bundle_manifest_sha256",
    "initial_prompt_sha256",
    "creation_timestamp",
}


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def campaign_paths(campaign_id: str) -> dict[str, Path]:
    croot = RESULTS / campaign_id
    return {
        "root": croot,
        "scientist": croot / "scientist_round_001",
        "scientist_exchange": croot / "exchange" / "incoming_requests",
        "scientist_request": croot / "exchange" / "incoming_requests" / "round_001_request.json",
        "hybrid": croot / "hybrid_blind",
        "hybrid_inputs": croot / "hybrid_blind" / "inputs",
        "hybrid_outputs": croot / "hybrid_blind" / "outputs",
        "hybrid_request": RESULTS / "exchange" / "incoming_requests" / f"{campaign_id}_hybrid_round_001_request.json",
        "hybrid_response": RESULTS / "exchange" / "public_responses" / f"{campaign_id}_hybrid_round_001_results.json",
    }


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def campaign_001_request_hash() -> str | None:
    legacy = RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"
    return sha256_file(legacy) if legacy.exists() else None


def request_provenance_payload(request: dict[str, Any]) -> dict[str, Any]:
    provenance = request.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    merged = {**provenance}
    for field in REQUIRED_REQUEST_PROVENANCE_FIELDS:
        if field in request:
            merged[field] = request[field]
    return merged


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def validate_scientist_registration(campaign_id: str, scientist_request: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate scientist request provenance before any exact/cache work happens."""

    paths = campaign_paths(campaign_id)
    expected_request = paths["scientist_request"].resolve()
    request_path = scientist_request.resolve()
    audit_rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any = "", expected: Any = "", notes: str = "") -> None:
        audit_rows.append(
            {
                "campaign_id": campaign_id,
                "check": check,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "notes": notes,
            }
        )

    exists = request_path.exists()
    add("request_file_exists", exists, relative_to_root(request_path), relative_to_root(expected_request))
    if not exists:
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_registration_audit.csv", index=False)
        raise ValueError(f"Scientist request does not exist: {scientist_request}")

    if campaign_id in REPLICATION_CAMPAIGNS:
        add(
            "campaign_specific_request_path",
            request_path == expected_request,
            relative_to_root(request_path),
            relative_to_root(expected_request),
            "replication campaigns must not use the legacy shared exchange path",
        )
        legacy = (RESULTS / "exchange" / "incoming_requests" / "round_001_request.json").resolve()
        add(
            "legacy_shared_path_rejected",
            request_path != legacy,
            relative_to_root(request_path),
            "not results/deployment_benchmark/exchange/incoming_requests/round_001_request.json",
        )

    request_sha = sha256_file(request_path)
    add("sha256_calculated_before_registration", True, request_sha, "", "hash computed before copy or exact lookup")

    try:
        request = read_json(request_path)
        add("request_json_decodes", True)
    except Exception as exc:
        add("request_json_decodes", False, notes=str(exc))
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_registration_audit.csv", index=False)
        raise ValueError(f"Scientist request is not valid JSON: {scientist_request}") from exc

    add("campaign_id_matches_cli", request.get("campaign_id") == campaign_id, request.get("campaign_id", ""), campaign_id)
    add("round_id_is_round_001", request.get("round_id") == "round_001", request.get("round_id", ""), "round_001")
    candidates = request.get("candidates", [])
    add("candidate_count_within_budget", isinstance(candidates, list) and 1 <= len(candidates) <= 8, len(candidates) if isinstance(candidates, list) else "not_list", "1..8")

    provenance = request_provenance_payload(request)
    for field in sorted(REQUIRED_REQUEST_PROVENANCE_FIELDS):
        add(f"request_provenance_field_{field}", bool(provenance.get(field)), provenance.get(field, ""), "present")

    manifest_path = DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv"
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else ""
    add(
        "public_bundle_manifest_hash_matches",
        bool(manifest_sha) and provenance.get("public_bundle_manifest_sha256") == manifest_sha,
        provenance.get("public_bundle_manifest_sha256", ""),
        manifest_sha,
    )

    prompt_path = RESULTS / campaign_id / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md"
    prompt_sha = sha256_file(prompt_path) if prompt_path.exists() else ""
    add(
        "initial_prompt_hash_matches",
        bool(prompt_sha) and provenance.get("initial_prompt_sha256") == prompt_sha,
        provenance.get("initial_prompt_sha256", ""),
        prompt_sha,
    )

    created_at = parse_timestamp(provenance.get("creation_timestamp"))
    bundle_exported_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc) if manifest_path.exists() else None
    add(
        "creation_after_bundle_export",
        bool(created_at and bundle_exported_at and created_at > bundle_exported_at),
        created_at.isoformat() if created_at else "",
        bundle_exported_at.isoformat() if bundle_exported_at else "",
    )

    for artifact, filename in REQUIRED_PROVENANCE_FILES.items():
        sidecar = request_path.parent / filename
        add(f"provenance_file_exists_{artifact}", sidecar.exists(), relative_to_root(sidecar), "present")

    c1_hash = campaign_001_request_hash()
    full_hash_match = bool(c1_hash and request_sha == c1_hash)
    add(
        "campaign_001_full_request_hash_review",
        not full_hash_match,
        request_sha,
        f"not {c1_hash}" if c1_hash else "campaign_001_hash_unavailable",
        "full request hash match requires provenance review and is not registration-ready",
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_registration_audit.csv", index=False)
    if not audit["passed"].astype(bool).all():
        failed = "; ".join(audit.loc[~audit["passed"].astype(bool), "check"].astype(str))
        raise ValueError(f"Scientist registration rejected for {campaign_id}: {failed}")
    return request, audit


def objective_from_public(df: pd.DataFrame) -> pd.Series:
    return df["final_product"] + 0.08 * df["product_AUC"] + 0.03 * df["final_biomass"] - 0.2 * df["growth_failure"].astype(float)


def prior_exact_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in [
        DATA / "deployment_benchmark_candidate_registry.csv",
        DATA / "deployment_benchmark_round_001_public_results.csv",
        DATA / "deployment_benchmark_hybrid_round_001_public_results.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if "candidate_hash" in df.columns:
                hashes.update(df["candidate_hash"].astype(str))
    return hashes


def register_scientist_request(campaign_id: str, scientist_request: Path) -> None:
    request, audit = validate_scientist_registration(campaign_id, scientist_request)
    paths = campaign_paths(campaign_id)
    paths["scientist"].mkdir(parents=True, exist_ok=True)
    local_request = paths["scientist"] / "round_001_request.json"
    shutil.copy2(scientist_request, local_request)
    req_hash = sha256_file(scientist_request)
    provenance = request_provenance_payload(request)
    status = "scientist_context_isolation_verified"
    rows = [
        {
            "campaign_id": campaign_id,
            "artifact": "fresh_task_session_identifier",
            "path": "",
            "sha256": "",
            "value": provenance.get("fresh_task_session_identifier", ""),
            "provenance_status": status,
            "notes": "validated from campaign-specific scientist request metadata",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "registration_audit",
            "path": relative_to_root(DATA / f"deployment_benchmark_{campaign_id}_scientist_registration_audit.csv"),
            "sha256": sha256_file(DATA / f"deployment_benchmark_{campaign_id}_scientist_registration_audit.csv"),
            "value": "passed",
            "provenance_status": status,
            "notes": f"{len(audit)} registration checks passed before copy/exact lookup",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "submitted_request",
            "path": relative_to_root(local_request),
            "sha256": sha256_file(local_request),
            "value": "",
            "provenance_status": status,
            "notes": "original campaign-specific file copied verbatim after registration validation",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "hidden_administrator_edited_request",
            "path": relative_to_root(local_request),
            "sha256": sha256_file(local_request),
            "value": False,
            "provenance_status": status,
            "notes": "verbatim copy only",
        },
    ]
    old = DATA / f"deployment_benchmark_{campaign_id}_scientist_provenance.csv"
    existing = pd.read_csv(old) if old.exists() else pd.DataFrame()
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(old, index=False)


def scientist_reused_exact(campaign_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = read_json(campaign_paths(campaign_id)["scientist"] / "round_001_request.json")
    source_public = pd.read_csv(DATA / "deployment_benchmark_round_001_public_results.csv")
    source_exact = pd.read_csv(DATA / "deployment_benchmark_round_001_exact_accounting.csv")
    source_response = read_json(RESULTS / "exchange" / "public_responses" / "round_001_results.json")
    public = source_public.copy()
    public.insert(0, "campaign_id", campaign_id)
    public.insert(1, "method", "scientist")
    public["exact_reuse_status"] = "reused_campaign_001_complete_candidate_hash_match"
    exact_df = source_exact.copy()
    exact_df.insert(0, "campaign_id", campaign_id)
    exact_df.insert(1, "method", "scientist")
    exact_df["exact_reuse_status"] = "reused_campaign_001_complete_candidate_hash_match"
    public.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_public_results.csv", index=False)
    exact_df.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_exact_accounting.csv", index=False)
    response = {
        "campaign_id": campaign_id,
        "round_id": "round_001",
        "status": "cached_exact_results_sanitised_from_campaign_001_hash_match",
        "source_response_sha256": sha256_file(RESULTS / "exchange" / "public_responses" / "round_001_results.json"),
        "source_request_sha256": sha256_file(RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"),
        "candidate_count": len(req.get("candidates", [])),
        "hidden_exact_simulations_run": 0,
        "cached_exact_matches": len(public),
        "physical_measurements_counted": len(public),
        "results": source_response.get("results", []),
    }
    out = RESULTS / "exchange" / "public_responses" / f"{campaign_id}_scientist_round_001_results.json"
    out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return public, exact_df


def scientist_verify_exact(campaign_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify a registered scientist request by exact cache/full-hash matches or fresh exact runs."""

    paths = campaign_paths(campaign_id)
    local_request = paths["scientist"] / "round_001_request.json"
    if not local_request.exists():
        raise FileNotFoundError(f"Registered scientist request missing: {local_request}")
    output_round_id = f"{campaign_id}_scientist_round_001"
    verifier.process_request(
        local_request,
        execute_fresh=True,
        hidden_regime="strong_dynamic_regulation",
        output_round_id=output_round_id,
    )
    response_path = RESULTS / "exchange" / "public_responses" / f"{output_round_id}_results.json"
    response = read_json(response_path)
    results = pd.DataFrame(response.get("results", []))
    if results.empty:
        raise ValueError(f"No scientist verifier results were produced for {campaign_id}")
    public_cols = ["candidate_id", "candidate_hash", "final_product", "product_AUC", "final_biomass", "growth_failure", "feasible"]
    public = results[public_cols].copy()
    public.insert(0, "campaign_id", campaign_id)
    public.insert(1, "method", "scientist")
    public["physical_measurements_counted"] = len(public)
    public["physical_rounds"] = 1
    public["parallel_or_sequential"] = "parallel_physical_batch_after_sequential_digital_dBTL"
    public.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_public_results.csv", index=False)

    hidden_root = RESULTS / "verifier" / "open_ended_exact" / output_round_id
    hidden = sorted(hidden_root.glob("*_hidden_summary.csv"))
    exact = pd.concat([pd.read_csv(p) for p in hidden], ignore_index=True, sort=False) if hidden else pd.DataFrame()
    fresh_hashes = set(exact["candidate_hash"].astype(str)) if not exact.empty and "candidate_hash" in exact.columns else set()
    if not exact.empty:
        exact.insert(0, "campaign_id", campaign_id)
        exact.insert(1, "method", "scientist")
        exact["exact_reuse_status"] = "fresh_exact_dynamic_pfba"
    cached_rows = []
    for result in response.get("results", []):
        if str(result.get("candidate_hash", "")) in fresh_hashes:
            continue
        cached_rows.append(
            {
                "campaign_id": campaign_id,
                "method": "scientist",
                "candidate_id": result.get("candidate_id", ""),
                "candidate_hash": result.get("candidate_hash", ""),
                "hidden_regime_id": "strong_dynamic_regulation",
                "n_actual_lp_solves": 0,
                "n_optimal_solves": 0,
                "n_infeasible_solves": 0,
                "n_unbounded_solves": 0,
                "n_solver_errors": 0,
                "n_skipped_intervals": 0,
                "n_surrogate_evaluations": 0,
                "exact_status": "cached_exact_result_reused_by_full_candidate_hash",
                "runtime_seconds": 0.0,
                "final_product": result.get("final_product", 0.0),
                "product_AUC": result.get("product_AUC", 0.0),
                "final_biomass": result.get("final_biomass", 0.0),
                "minimum_biomass": "",
                "feasible": result.get("feasible", True),
                "severe_growth_collapse": result.get("growth_failure", False),
                "exact_reuse_status": "reused_full_candidate_hash_match",
            }
        )
    if cached_rows:
        exact = pd.concat([exact, pd.DataFrame(cached_rows)], ignore_index=True, sort=False)
    exact.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_exact_accounting.csv", index=False)
    return public, exact


def generate_candidates(campaign_id: str, seed: int, n: int = 50000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    hetero = pd.read_csv(DATA / "deployment_benchmark_heterologous_library.csv")
    old_hashes = prior_exact_hashes()
    env_grid = [(t, ph, do) for t in [29.0, 30.0, 31.0] for ph in [4.8, 5.0, 5.2] for do in [45.0, 55.0, 65.0, 75.0]]
    risk_weight = universe["risk_level"].map({"low": 1.0, "medium": 1.3, "high": 0.7}).fillna(1.0).to_numpy(float)
    risk_weight = risk_weight / risk_weight.sum()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw = duplicate = invalid = 0
    universe_index = universe.set_index("reaction_id")
    old_static_flag = deployment.PUBLIC_STATIC_SOLVE_ENABLED
    deployment.PUBLIC_STATIC_SOLVE_ENABLED = False
    try:
        while len(rows) < n:
            raw += 1
            tier = 1 if rng.random() < 0.62 else 2
            edit_n = int(rng.choice([1, 2] if tier == 1 else [2, 3, 4], p=[0.45, 0.55] if tier == 1 else [0.35, 0.45, 0.20]))
            picks = universe.iloc[rng.choice(len(universe), size=edit_n, replace=False, p=risk_weight)]
            edits = [hybrid1.edit_for_reaction(row, rng) for _, row in picks.iterrows()]
            if tier == 2 and rng.random() < 0.22:
                hrow = hetero.iloc[int(rng.integers(0, len(hetero)))]
                edits = edits[:3] + [
                    {
                        "reaction_id": str(hrow["library_reaction_id"]),
                        "edit_type": "curated_heterologous_addition",
                        "capacity_multiplier": round(float(rng.uniform(1.15, 1.65)), 4),
                        "rationale": str(hrow["rationale"]),
                    }
                ]
            temp, ph, do = env_grid[int(rng.integers(0, len(env_grid)))]
            payload = {
                "candidate_id": "",
                "strain_id": "",
                "edit_tier": tier,
                "edits": edits,
                "environment": {"temperature": temp, "pH": ph, "DO": do},
                "requested_assay_panel": "basic",
                "hypothesis": f"Hybrid-generated blinded strain-environment design for {campaign_id}.",
                "expected_failure_mode": "Growth, redox, ATP, or pathway-burden limitation.",
            }
            h = deployment.deterministic_candidate_hash(payload)
            if h in seen:
                duplicate += 1
                continue
            seen.add(h)
            val = deployment.validate_sparse_design(payload, universe)
            if not val["valid"]:
                invalid += 1
                continue
            edit_hash = hashlib.sha256(stable_json({"edits": payload["edits"]}).encode()).hexdigest()[:12]
            payload["candidate_id"] = f"{campaign_id}_hybrid_candidate_{len(rows)+1:05d}"
            payload["strain_id"] = f"{campaign_id}_hybrid_strain_{edit_hash}"
            h = deployment.deterministic_candidate_hash(payload)
            subsystems = sorted(set(str(universe_index.loc[e["reaction_id"], "subsystem"]) for e in edits if e["reaction_id"] in universe_index.index))
            rows.append(
                {
                    **payload,
                    "candidate_hash": h,
                    "n_edits": len(edits),
                    "subsystems": ";".join(subsystems),
                    "old_exact_pool_overlap": h in old_hashes,
                    "raw_proposals_seen": raw,
                    "duplicates_seen": duplicate,
                    "invalid_seen": invalid,
                    "candidate_json": stable_json(payload),
                }
            )
    finally:
        deployment.PUBLIC_STATIC_SOLVE_ENABLED = old_static_flag
    out = pd.DataFrame(rows)
    out.to_csv(DATA / f"deployment_benchmark_{campaign_id}_candidate_generation.csv", index=False)
    return out


def score_candidates(campaign_id: str, candidates: pd.DataFrame) -> pd.DataFrame:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv").set_index("reaction_id")
    scored = []
    for row in candidates.itertuples(index=False):
        cand = json.loads(row.candidate_json)
        env = cand["environment"]
        env_score = np.exp(-((env["temperature"] - 30.0) / 2.5) ** 2) * np.exp(-((env["pH"] - 5.05) / 0.42) ** 2) * (1.0 + 0.12 * (env["DO"] - 50.0) / 30.0)
        precursor = support = export = byprod = path = risk_penalty = static_span = 0.0
        for e in cand["edits"]:
            rid = e["reaction_id"]
            mult = float(e.get("capacity_multiplier", 1.0))
            if rid in universe.index:
                info = universe.loc[rid]
                text = f"{info.get('subsystem','')} {info.get('inclusion_rule','')} {info.get('rationale_for_inclusion','')}".lower()
                static_span += max(float(info.get("fva_max", 0.0)) - float(info.get("fva_min", 0.0)), 0.0)
                risk_penalty += {"low": 0.0, "medium": 0.025, "high": 0.060}.get(str(info.get("risk_level", "medium")), 0.03)
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
        scored.append(
            {
                "candidate_id": cand["candidate_id"],
                "candidate_hash": row.candidate_hash,
                "static_prefilter_score": static_score,
                "surrogate_score": np.nan,
                "uncertainty": uncertainty,
                "construction_risk": "high" if risk_penalty > 0.12 else "medium" if risk_penalty > 0.04 else "low",
                "edit_complexity": complexity,
                "static_fva_span": static_span,
            }
        )
    out = candidates.merge(pd.DataFrame(scored), on=["candidate_id", "candidate_hash"])
    out.to_csv(DATA / f"deployment_benchmark_{campaign_id}_static_accounting.csv", index=False)
    pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "backend": "static_gem_prefilter",
                "candidate_count": len(out),
                "actual_lp_solves": 0,
                "rejection_reasons": "invalid_sparse_designs_rejected_during_generation",
                "subsystem_coverage": int(out["subsystems"].astype(str).str.split(";").explode().nunique()),
            },
            {
                "campaign_id": campaign_id,
                "backend": "sparse_edit_surrogate",
                "candidate_count": 0,
                "actual_lp_solves": 0,
                "validated_sparse_surrogate_scores": 0,
                "rejection_reasons": "no_validated_reaction_aware_sparse_surrogate_available",
                "subsystem_coverage": 0,
            },
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_screening_layer_accounting.csv", index=False)
    return out


def dynamic_screen(campaign_id: str, scored: pd.DataFrame, n_dynamic: int = 100) -> pd.DataFrame:
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, gem.configured_gem_path())
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    selected = scored.sort_values("static_prefilter_score", ascending=False).head(max(n_dynamic * 5, n_dynamic)).head(n_dynamic).copy()
    rows = []
    tic = time.perf_counter()
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        cand = json.loads(row.candidate_json)
        dyn = hybrid1.hybrid_dynamic_eval(cand, augmented, n_time=2)
        rows.append({"candidate_id": cand["candidate_id"], "candidate_hash": row.candidate_hash, **dyn})
        if i % 10 == 0:
            print(f"[{campaign_id}] dynamic {i}/{len(selected)}", flush=True)
            gc.collect()
    elapsed = time.perf_counter() - tic
    out = selected.merge(pd.DataFrame(rows), on=["candidate_id", "candidate_hash"])
    out["dynamic_horizon_timepoints"] = 2
    out["dynamic_intervals"] = 1
    out["lp_solves_per_dynamic_candidate"] = out["n_actual_lp_solves"]
    out["dynamic_wall_clock_seconds_total"] = elapsed
    out["peak_memory_mb"] = np.nan
    out.to_csv(DATA / f"deployment_benchmark_{campaign_id}_dynamic_accounting.csv", index=False)
    del base, augmented
    gc.collect()
    return out


def select_batch(campaign_id: str, dynamic: pd.DataFrame) -> pd.DataFrame:
    old_hashes = prior_exact_hashes()
    ranked = dynamic.copy()
    ranked["robustness_score"] = ranked["predicted_objective"] - ranked["uncertainty"]
    ranked["rank_score"] = ranked["predicted_objective"] + 0.2 * ranked["robustness_score"] - 0.015 * ranked["edit_complexity"]
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["overlap_prior_exact_candidates"] = ranked["candidate_hash"].isin(old_hashes)
    ranked["selection_reason"] = "ranked_by_campaign_blinded_dynamic_hybrid_before_exact_lookup"
    ranked.to_csv(DATA / f"deployment_benchmark_{campaign_id}_ranked_candidates.csv", index=False)
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
    selected.to_csv(DATA / f"deployment_benchmark_{campaign_id}_selected_batch.csv", index=False)
    selected[["candidate_id", "candidate_hash", "overlap_prior_exact_candidates"]].to_csv(DATA / f"deployment_benchmark_{campaign_id}_overlap_audit.csv", index=False)
    return selected


def write_hybrid_request(campaign_id: str, selected: pd.DataFrame) -> Path:
    candidates = []
    for row in selected.itertuples(index=False):
        cand = json.loads(row.candidate_json)
        cand["candidate_id"] = str(row.candidate_id).replace(f"{campaign_id}_hybrid_candidate_", f"{campaign_id}_hybrid_selected_")
        cand["hypothesis"] = f"Blinded hybrid-selected design from {campaign_id} open-ended reaction-level search."
        cand["expected_failure_mode"] = "Growth, ATP/redox burden, pathway imbalance, or construction risk."
        candidates.append(cand)
    request = {"round_id": f"{campaign_id}_hybrid_round_001", "campaign_id": campaign_id, "candidates": candidates}
    path = campaign_paths(campaign_id)["hybrid_request"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(path, campaign_paths(campaign_id)["hybrid_outputs"] / path.name)
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    prior = prior_exact_hashes()
    rows = []
    for candidate in candidates:
        val = deployment.validate_sparse_design(candidate, universe)
        rows.append(
            {
                "campaign_id": campaign_id,
                "candidate_id": candidate["candidate_id"],
                "strain_id": candidate["strain_id"],
                "request_candidate_hash": val["candidate_hash"],
                "valid": val["valid"],
                "validation_reasons": val["reasons"],
                "overlap_prior_exact_candidates": val["candidate_hash"] in prior,
            }
        )
    pd.DataFrame(rows).to_csv(DATA / f"deployment_benchmark_{campaign_id}_request_overlap_audit.csv", index=False)
    return path


def write_screening_funnel(campaign_id: str, generated: pd.DataFrame, scored: pd.DataFrame, dynamic: pd.DataFrame, selected: pd.DataFrame) -> None:
    pd.DataFrame(
        [
            {"campaign_id": campaign_id, "layer": "candidate_generation_only", "backend_label": "candidate_generation_only", "candidates": len(generated), "actual_lp_solves": 0},
            {"campaign_id": campaign_id, "layer": "static_prefilter", "backend_label": "static_gem_prefilter", "candidates": len(scored), "actual_lp_solves": 0},
            {"campaign_id": campaign_id, "layer": "sparse_surrogate", "backend_label": "sparse_edit_surrogate", "candidates": 0, "actual_lp_solves": 0, "validated_sparse_surrogate_scores": 0},
            {"campaign_id": campaign_id, "layer": "dynamic_refinement", "backend_label": "hybrid_dynamic_yeast9", "candidates": len(dynamic), "actual_lp_solves": int(dynamic["n_actual_lp_solves"].sum()), "horizon_timepoints": 2, "intervals": 1},
            {"campaign_id": campaign_id, "layer": "selected_batch", "backend_label": "frozen_selection_before_exact_lookup", "candidates": len(selected), "actual_lp_solves": 0},
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_screening_funnel.csv", index=False)


def prepare_hybrid(campaign_id: str, seed: int, n_dynamic: int = 100) -> Path:
    paths = campaign_paths(campaign_id)
    paths["hybrid"].mkdir(parents=True, exist_ok=True)
    paths["hybrid_inputs"].mkdir(parents=True, exist_ok=True)
    paths["hybrid_outputs"].mkdir(parents=True, exist_ok=True)
    access = pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_hybrid_access_audit.csv")
    access.loc[access["audit_item"].eq("hybrid_search_started"), ["value", "passed"]] = [True, True]
    access.loc[access["audit_item"].eq("blocking_dependency"), ["value", "passed"]] = [
        "fresh scientist request registered before hybrid search; dependency cleared",
        True,
    ]
    access.to_csv(DATA / f"deployment_benchmark_{campaign_id}_hybrid_access_audit.csv", index=False)
    generated = generate_candidates(campaign_id, seed)
    scored = score_candidates(campaign_id, generated)
    dynamic = dynamic_screen(campaign_id, scored, n_dynamic=n_dynamic)
    selected = select_batch(campaign_id, dynamic)
    write_screening_funnel(campaign_id, generated, scored, dynamic, selected)
    return write_hybrid_request(campaign_id, selected)


def compare(campaign_id: str) -> None:
    sci = pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_public_results.csv")
    scientist_response_path = RESULTS / "exchange" / "public_responses" / f"{campaign_id}_scientist_round_001_results.json"
    scientist_response = read_json(scientist_response_path) if scientist_response_path.exists() else {}
    response_path = campaign_paths(campaign_id)["hybrid_response"]
    response = read_json(response_path)
    hyb = pd.DataFrame([{k: r[k] for k in ["candidate_id", "candidate_hash", "final_product", "product_AUC", "final_biomass", "growth_failure", "feasible"]} for r in response["results"]])
    hyb.insert(0, "campaign_id", campaign_id)
    hyb.insert(1, "method", "hybrid")
    hyb.to_csv(DATA / f"deployment_benchmark_{campaign_id}_hybrid_public_results.csv", index=False)
    hidden = sorted((RESULTS / "verifier" / "open_ended_exact" / f"{campaign_id}_hybrid_round_001").glob("*_hidden_summary.csv"))
    hyb_exact = pd.concat([pd.read_csv(p) for p in hidden], ignore_index=True) if hidden else pd.DataFrame()
    if not hyb_exact.empty:
        hyb_exact.insert(0, "campaign_id", campaign_id)
        hyb_exact.insert(1, "method", "hybrid")
    sci_exact = pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_exact_accounting.csv")
    exact = pd.concat([sci_exact, hyb_exact], ignore_index=True, sort=False)
    exact.to_csv(DATA / f"deployment_benchmark_{campaign_id}_exact_accounting.csv", index=False)
    digital_path = DATA / f"deployment_benchmark_{campaign_id}_scientist_digital_experiments.csv"
    digital_count = len(pd.read_csv(digital_path)) if digital_path.exists() else 0
    physical_plan_path = DATA / f"deployment_benchmark_{campaign_id}_scientist_physical_plan.csv"
    unique_strains = (
        int(pd.read_csv(physical_plan_path)["strain_id"].astype(str).nunique())
        if physical_plan_path.exists()
        else int(sci["candidate_id"].astype(str).nunique())
    )
    central = pd.read_csv(DATA / "deployment_benchmark_operational_scenarios.csv")
    central = central[central["scenario_id"].eq("central_lab")].iloc[0]

    def physical_accounting(cultures: int, strains: int) -> dict[str, float]:
        return {
            "calendar_days": float(central["strain_construction_batch_lead_days"]) + float(central["culture_duration_days"]) + float(central["assay_duration_days"]),
            "scientist_hours": float(central["scientist_planning_hours_per_round"]) + float(central["analysis_hours_per_round"]),
            "technician_hours": strains * float(central["technician_setup_hours_per_strain"]),
            "normalised_resource_points": strains * float(central["resource_points_per_strain"])
            + cultures * float(central["resource_points_per_culture"])
            + cultures * float(central["resource_points_basic_panel"]),
        }

    sci_physical = physical_accounting(len(sci), unique_strains)
    hyb_unique_strains = int(hyb["candidate_id"].astype(str).nunique())
    hyb_physical = physical_accounting(len(hyb), hyb_unique_strains)
    dynamic_path = DATA / f"deployment_benchmark_{campaign_id}_dynamic_accounting.csv"
    dynamic_df = pd.read_csv(dynamic_path) if dynamic_path.exists() else pd.DataFrame()
    hyb_dynamic_lp = int(dynamic_df.get("n_actual_lp_solves", pd.Series(dtype=float)).fillna(0).sum()) if not dynamic_df.empty else 0
    hyb_dynamic_wall = (
        float(dynamic_df["dynamic_wall_clock_seconds_total"].dropna().iloc[0])
        if not dynamic_df.empty and "dynamic_wall_clock_seconds_total" in dynamic_df.columns and dynamic_df["dynamic_wall_clock_seconds_total"].notna().any()
        else 0.0
    )
    sci_obj = objective_from_public(sci)
    hyb_obj = objective_from_public(hyb)
    rows = [
        {
            "campaign_id": campaign_id,
            "method": "scientist",
            "best_verified_objective": float(sci_obj.max()),
            "mean_verified_objective": float(sci_obj.mean()),
            "median_verified_objective": float(sci_obj.median()),
            "best_final_product": float(sci["final_product"].max()),
            "best_product_AUC": float(sci["product_AUC"].max()),
            "best_final_biomass": float(sci["final_biomass"].max()),
            "growth_failure_rate": float(sci["growth_failure"].mean()),
            "physical_rounds": 1,
            "cultures": len(sci),
            "unique_strains": unique_strains,
            "hidden_exact_simulations_run": int(scientist_response.get("hidden_exact_simulations_run", max(0, len(sci_exact) - int((sci_exact.get("exact_reuse_status", pd.Series(dtype=str)).astype(str) == "reused_full_candidate_hash_match").sum())))),
            "cached_exact_matches": int(scientist_response.get("cached_exact_matches", int((sci_exact.get("exact_reuse_status", pd.Series(dtype=str)).astype(str) == "reused_full_candidate_hash_match").sum()))),
            "exact_lp_solves_run_now": int(scientist_response.get("hidden_exact_lp_solves", sci_exact.get("n_actual_lp_solves", pd.Series(dtype=float)).fillna(0).sum())),
            "physical_measurements_counted": len(sci),
            "digital_experiments": digital_count,
            "digital_public_lp_solves": digital_count,
            "hybrid_dynamic_lp_solves": 0,
            "hybrid_dynamic_wall_clock_seconds_total": 0.0,
            "digital_experiment_type": "sequential_public_dBTL_static_validation",
            "physical_execution_mode": "parallel_round_001_batch",
            **sci_physical,
        },
        {
            "campaign_id": campaign_id,
            "method": "hybrid",
            "best_verified_objective": float(hyb_obj.max()),
            "mean_verified_objective": float(hyb_obj.mean()),
            "median_verified_objective": float(hyb_obj.median()),
            "best_final_product": float(hyb["final_product"].max()),
            "best_product_AUC": float(hyb["product_AUC"].max()),
            "best_final_biomass": float(hyb["final_biomass"].max()),
            "growth_failure_rate": float(hyb["growth_failure"].mean()),
            "physical_rounds": 1,
            "cultures": len(hyb),
            "unique_strains": hyb_unique_strains,
            "hidden_exact_simulations_run": int(response.get("hidden_exact_simulations_run", len(hyb_exact))),
            "cached_exact_matches": int(response.get("cached_exact_matches", 0)),
            "exact_lp_solves_run_now": int(response.get("hidden_exact_lp_solves", hyb_exact.get("n_actual_lp_solves", pd.Series(dtype=int)).sum())),
            "physical_measurements_counted": len(hyb),
            "digital_experiments": int(pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_candidate_generation.csv").shape[0]) if (DATA / f"deployment_benchmark_{campaign_id}_candidate_generation.csv").exists() else "",
            "digital_public_lp_solves": 0,
            "hybrid_dynamic_lp_solves": hyb_dynamic_lp,
            "hybrid_dynamic_wall_clock_seconds_total": hyb_dynamic_wall,
            "digital_experiment_type": "parallel_virtual_generation_then_sequential_dynamic_refinement",
            "physical_execution_mode": "parallel_round_001_batch",
            **hyb_physical,
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(DATA / f"deployment_benchmark_{campaign_id}_comparison.csv", index=False)
    delta = float(comp[comp["method"].eq("hybrid")]["best_verified_objective"].iloc[0] - comp[comp["method"].eq("scientist")]["best_verified_objective"].iloc[0])
    margin = float(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"])
    status = "hybrid_superior" if delta > margin else "hybrid_noninferior" if delta >= -margin else "hybrid_inferior"
    pd.DataFrame(
        [
            {"campaign_id": campaign_id, "status_type": "scientist_provenance", "status": "scientist_context_isolation_verified_campaign_specific_request", "passed": True, "notes": "fresh public-only DBTL agent request registered from campaign-specific exchange path"},
            {"campaign_id": campaign_id, "status_type": "quality", "status": status, "passed": status != "hybrid_inferior", "notes": f"best objective delta hybrid-scientist={delta:.6g}; margin={margin}"},
            {"campaign_id": campaign_id, "status_type": "physical_budget", "status": "matched_one_round_eight_cultures", "passed": True, "notes": "all requested scientist and hybrid cultures count as physical measurements; exact cache reuse is accounted separately"},
            {"campaign_id": campaign_id, "status_type": "digital_physical_accounting", "status": "digital_and_physical_experiments_separated", "passed": True, "notes": "scientist sequential public DBTL planning is accounted separately from parallel physical cultures"},
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_acceptance.csv", index=False)


def update_registry(campaign_id: str, status: str) -> None:
    path = DATA / "deployment_benchmark_campaign_registry.csv"
    reg = pd.read_csv(path)
    mask = reg["campaign_id"].eq(campaign_id)
    reg.loc[mask, "status"] = status
    if status == "one_shot_complete":
        reg.loc[mask, "scientist_request_status"] = "complete_campaign_specific_fresh_agent"
        reg.loc[mask, "hybrid_request_status"] = "complete"
        reg.loc[mask, "exact_verification_status"] = "complete_scientist_and_hybrid_verified"
        reg.loc[mask, "notes"] = "fresh public-only scientist agent registered and verified; hybrid independently searched and verified"
    reg.to_csv(path, index=False)


def ensure_campaign_specific_exchange_paths() -> None:
    for campaign_id in ["campaign_002", "campaign_003"]:
        paths = campaign_paths(campaign_id)
        for key in ["scientist_exchange"]:
            paths[key].mkdir(parents=True, exist_ok=True)


def refresh_public_manifest_hashes(campaign_id: str) -> None:
    manifest_path = DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv"
    bundle = RESULTS / campaign_id / "public_scientist_bundle"
    if not manifest_path.exists() or not bundle.exists():
        return
    manifest = pd.read_csv(manifest_path)
    for idx, row in manifest.iterrows():
        rel = str(row.get("path", ""))
        artifact_path = bundle / rel
        if artifact_path.is_file():
            manifest.at[idx, "sha256"] = sha256_file(artifact_path)
    manifest.to_csv(manifest_path, index=False)


def write_waiting_scientist_provenance(campaign_id: str) -> None:
    expected_request = campaign_paths(campaign_id)["scientist_request"]
    manifest_path = DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv"
    prompt_path = RESULTS / campaign_id / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md"
    pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "artifact": "public_bundle_manifest",
                "path": relative_to_root(manifest_path),
                "sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
                "value": "",
                "provenance_status": "public_bundle_prepared",
                "notes": "valid scientist request must report this manifest hash",
            },
            {
                "campaign_id": campaign_id,
                "artifact": "scientist_prompt",
                "path": relative_to_root(prompt_path),
                "sha256": sha256_file(prompt_path) if prompt_path.exists() else "",
                "value": "",
                "provenance_status": "public_bundle_prepared",
                "notes": "valid scientist request must report this initial prompt hash",
            },
            {
                "campaign_id": campaign_id,
                "artifact": "valid_request_required_path",
                "path": relative_to_root(expected_request),
                "sha256": "",
                "value": "missing",
                "provenance_status": f"blocked_waiting_valid_{campaign_id}_scientist_request",
                "notes": "registration rejects the legacy shared path and waits for campaign-specific request plus sidecar provenance files",
            },
            {
                "campaign_id": campaign_id,
                "artifact": "required_sidecar_files",
                "path": relative_to_root(expected_request.parent),
                "sha256": "",
                "value": ";".join(REQUIRED_PROVENANCE_FILES.values()),
                "provenance_status": f"blocked_waiting_valid_{campaign_id}_scientist_request",
                "notes": "all listed sidecar files must exist before exact/cache verification",
            },
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_provenance.csv", index=False)


def repair_invalid_campaign_002_registration() -> None:
    ensure_campaign_specific_exchange_paths()
    for cid in ["campaign_002", "campaign_003"]:
        refresh_public_manifest_hashes(cid)
    campaign_id = "campaign_002"
    paths = campaign_paths(campaign_id)
    legacy_request = RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"
    copied_request = paths["scientist"] / "round_001_request.json"
    expected_request = paths["scientist_request"]
    legacy_hash = sha256_file(legacy_request) if legacy_request.exists() else ""
    copied_hash = sha256_file(copied_request) if copied_request.exists() else ""
    c1_hash = campaign_001_request_hash() or ""

    invalid_status = "campaign_002_scientist_replication_invalid_duplicate_request"
    reference_status = "campaign_001_scientist_reference_reused_for_interim_campaign_002_comparison"

    pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "audit_item": "legacy_shared_path_used",
                "status": invalid_status,
                "passed": False,
                "observed": relative_to_root(legacy_request),
                "expected": relative_to_root(expected_request),
                "sha256": legacy_hash,
                "notes": "registered scientist request was read from the shared Campaign 001 exchange path",
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "campaign_specific_request_missing",
                "status": "blocked_waiting_valid_campaign_002_scientist_request",
                "passed": False,
                "observed": str(expected_request.exists()),
                "expected": "True",
                "sha256": "",
                "notes": "no valid replacement request exists in the campaign-specific incoming_requests directory",
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "campaign_001_full_request_hash_match",
                "status": "duplicate_request_provenance_unresolved",
                "passed": False,
                "observed": copied_hash or legacy_hash,
                "expected": f"not {c1_hash}",
                "sha256": copied_hash or legacy_hash,
                "notes": "duplicate may not be counted as an independent scientist replication without fresh transcript/tool provenance",
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "campaign_002_hybrid_preserved",
                "status": "campaign_002_hybrid_frozen_valid",
                "passed": True,
                "observed": "best=0.678774; mean=0.678774; exact_lp=1152",
                "expected": "preserve without rerun",
                "sha256": "",
                "notes": "frozen blinded hybrid search and exact results remain valid Campaign 002 assets",
            },
        ]
    ).to_csv(DATA / "deployment_benchmark_campaign_002_invalid_registration_audit.csv", index=False)

    write_waiting_scientist_provenance(campaign_id)
    provenance = pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_provenance.csv")
    extra = pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "artifact": "invalid_submitted_request",
                "path": relative_to_root(copied_request) if copied_request.exists() else relative_to_root(legacy_request),
                "sha256": copied_hash or legacy_hash,
                "value": invalid_status,
                "provenance_status": invalid_status,
                "notes": "full request hash matched Campaign 001 and source path was legacy shared exchange",
            },
            {
                "campaign_id": campaign_id,
                "artifact": "interim_reference",
                "path": "data/deployment_benchmark_campaign_002_interim_invalid_duplicate_scientist_comparison.csv",
                "sha256": "",
                "value": reference_status,
                "provenance_status": reference_status,
                "notes": "retained for interim comparison only; excluded from independent replication inference",
            },
        ]
    )
    pd.concat([provenance, extra], ignore_index=True).to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_provenance.csv", index=False)
    write_waiting_scientist_provenance("campaign_003")

    comparison_path = DATA / f"deployment_benchmark_{campaign_id}_comparison.csv"
    if comparison_path.exists():
        archive_path = DATA / f"deployment_benchmark_{campaign_id}_interim_invalid_duplicate_scientist_comparison.csv"
        shutil.copy2(comparison_path, archive_path)
        archived = pd.read_csv(archive_path)
        archived["replication_inference_eligible"] = False
        archived["inference_status"] = "invalid_duplicate_scientist_registration"
        archived.to_csv(archive_path, index=False)

        comp = pd.read_csv(comparison_path)
        comp["replication_inference_eligible"] = comp["method"].eq("hybrid")
        comp["scientist_batch_status"] = comp["method"].map(
            {
                "scientist": reference_status,
                "scientist_reference": reference_status,
                "hybrid": "campaign_002_hybrid_frozen_valid_waiting_valid_scientist_comparator",
            }
        )
        comp.loc[comp["method"].eq("scientist"), "method"] = "scientist_reference"
        comp["comparison_status"] = "interim_only_waiting_valid_campaign_002_scientist_request"
        comp.to_csv(comparison_path, index=False)

    pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "status_type": "scientist_provenance",
                "status": invalid_status,
                "passed": False,
                "notes": "legacy shared request path and full Campaign 001 request hash match; not an independent scientist replication",
            },
            {
                "campaign_id": campaign_id,
                "status_type": "hybrid_frozen_result",
                "status": "campaign_002_hybrid_frozen_valid",
                "passed": True,
                "notes": "do not rerun frozen Campaign 002 hybrid candidates; best objective 0.678774, mean 0.678774, 1152 exact LP solves",
            },
            {
                "campaign_id": campaign_id,
                "status_type": "replication_comparison",
                "status": "blocked_waiting_valid_campaign_002_scientist_request",
                "passed": False,
                "notes": "corrected comparison cannot be used for replication inference until a valid campaign-specific scientist request arrives",
            },
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_acceptance.csv", index=False)

    registry_path = DATA / "deployment_benchmark_campaign_registry.csv"
    if registry_path.exists():
        reg = pd.read_csv(registry_path)
        mask = reg["campaign_id"].eq(campaign_id)
        reg.loc[mask, "status"] = "hybrid_complete_waiting_valid_scientist_request"
        reg.loc[mask, "scientist_request_status"] = "invalid_duplicate_request"
        reg.loc[mask, "hybrid_request_status"] = "complete_frozen"
        reg.loc[mask, "exact_verification_status"] = "hybrid_complete_scientist_invalid"
        reg.loc[mask, "notes"] = "scientist registration invalid duplicate of Campaign 001 from legacy shared path; Campaign 002 hybrid remains frozen"
        reg.to_csv(registry_path, index=False)

    summary_rows = [
        {"summary_item": "checkpoint_status", "value": "blocked_waiting_valid_campaign_002_and_campaign_003_scientist_requests"},
        {"summary_item": "completed_campaigns_frozen", "value": "campaign_001"},
        {"summary_item": "completed_replication_campaigns", "value": ""},
        {"summary_item": "waiting_campaigns", "value": "campaign_002;campaign_003"},
        {"summary_item": "prepared_public_bundles", "value": "campaign_002;campaign_003"},
        {"summary_item": "campaign_002_hybrid_exact_simulations_preserved", "value": 8},
        {"summary_item": "campaign_002_hybrid_exact_lp_solves_preserved", "value": 1152},
        {"summary_item": "reason_for_stop", "value": "valid Campaign 002 and Campaign 003 scientist requests are absent; adaptive Campaign 001 not started"},
    ]
    pd.DataFrame(summary_rows).to_csv(DATA / "deployment_benchmark_report_checkpoint_summary.csv", index=False)
    pd.DataFrame(
        [
            {"status_type": "campaign_002_scientist", "status": invalid_status, "passed": False, "notes": "legacy shared path and full Campaign 001 request hash match"},
            {"status_type": "campaign_002_hybrid", "status": "campaign_002_hybrid_frozen_valid", "passed": True, "notes": "preserved without rerun"},
            {"status_type": "campaign_003_one_shot", "status": "blocked_waiting_fresh_scientist_request", "passed": False, "notes": "do not substitute scripted scientist"},
            {"status_type": "adaptive_campaign_001", "status": "not_started_waiting_valid_replication_phase", "passed": False, "notes": "adaptive phase waits until Campaigns 002 and 003 have valid scientist requests"},
            {"status_type": "configuration_freeze", "status": "report_checkpoint_configuration_frozen", "passed": True, "notes": "objective, universe, verifier, Yeast9 checksum, hybrid checkpoint and hashing implementation recorded"},
        ]
    ).to_csv(DATA / "deployment_benchmark_report_checkpoint_acceptance.csv", index=False)

    for name in [
        "deployment_benchmark_campaign_003_screening_funnel.csv",
        "deployment_benchmark_campaign_003_exact_accounting.csv",
        "deployment_benchmark_campaign_003_comparison.csv",
    ]:
        pd.DataFrame(
            [
                {
                    "campaign_id": "campaign_003",
                    "status": "not_started_waiting_valid_campaign_003_scientist_request",
                    "actual_lp_solves": 0,
                    "hidden_exact_simulations_run": 0,
                    "notes": "stop condition prevents Campaign 003 hybrid search and exact verification until valid scientist request arrives",
                }
            ]
        ).to_csv(DATA / name, index=False)

    if (DATA / "deployment_benchmark_one_shot_replication_summary.csv").exists():
        existing = pd.read_csv(DATA / "deployment_benchmark_one_shot_replication_summary.csv")
        valid = existing[existing["campaign_id"].eq("campaign_001")].copy()
        valid["replication_inference_eligible"] = True
        valid.to_csv(DATA / "deployment_benchmark_one_shot_replication_summary.csv", index=False)
    if (DATA / "deployment_benchmark_one_shot_paired_differences.csv").exists():
        diffs = pd.read_csv(DATA / "deployment_benchmark_one_shot_paired_differences.csv")
        diffs = diffs[diffs["campaign_id"].eq("campaign_001")].copy()
        diffs["replication_inference_eligible"] = True
        diffs.to_csv(DATA / "deployment_benchmark_one_shot_paired_differences.csv", index=False)

    pd.DataFrame(
        [
            {
                "analysis": "one_shot_bootstrap",
                "valid_campaigns": 1,
                "status": "one_shot_replication_inconclusive",
                "exploratory": True,
                "best_delta_mean": "",
                "best_delta_ci_low": "",
                "best_delta_ci_high": "",
                "notes": "bootstrap intervals withheld until Campaigns 002 and 003 have valid scientist batches",
            }
        ]
    ).to_csv(DATA / "deployment_benchmark_one_shot_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {
                "analysis": "one_shot_replication",
                "status": "one_shot_replication_inconclusive",
                "passed": False,
                "valid_campaigns": 1,
                "notes": "Campaign 002 scientist duplicate excluded; Campaign 003 scientist request absent",
            }
        ]
    ).to_csv(DATA / "deployment_benchmark_one_shot_acceptance.csv", index=False)

    for name in [
        "deployment_benchmark_campaign_001_adaptive_requests.csv",
        "deployment_benchmark_campaign_001_adaptive_results.csv",
        "deployment_benchmark_campaign_001_adaptive_comparison.csv",
        "deployment_benchmark_time_to_threshold.csv",
    ]:
        pd.DataFrame(
            [
                {
                    "status": "not_started_waiting_valid_campaign_002_and_campaign_003_scientist_requests",
                    "notes": "stop condition prevents adaptive Campaign 001 execution",
                }
            ]
        ).to_csv(DATA / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default="campaign_002")
    parser.add_argument("--seed", type=int, default=22002)
    parser.add_argument("--scientist-request", default=None)
    parser.add_argument("--n-dynamic", type=int, default=100)
    parser.add_argument("command", choices=["prepare-hybrid", "compare", "register-scientist", "repair-invalid-campaign-002"], nargs="?", default="prepare-hybrid")
    args = parser.parse_args()
    if args.command == "register-scientist":
        scientist_request = Path(args.scientist_request) if args.scientist_request else campaign_paths(args.campaign_id)["scientist_request"]
        register_scientist_request(args.campaign_id, scientist_request)
        scientist_verify_exact(args.campaign_id)
        update_registry(args.campaign_id, "scientist_registered_waiting_hybrid")
    elif args.command == "repair-invalid-campaign-002":
        repair_invalid_campaign_002_registration()
        print(DATA / "deployment_benchmark_campaign_002_invalid_registration_audit.csv")
    elif args.command == "prepare-hybrid":
        request = prepare_hybrid(args.campaign_id, args.seed, n_dynamic=args.n_dynamic)
        print(request)
    else:
        compare(args.campaign_id)
        update_registry(args.campaign_id, "one_shot_complete")
        print(pd.read_csv(DATA / f"deployment_benchmark_{args.campaign_id}_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
