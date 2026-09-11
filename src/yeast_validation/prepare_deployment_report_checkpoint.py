#!/usr/bin/env python3
"""Prepare report-checkpoint bundles without fabricating missing scientist runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

import deployment_benchmark as deployment
import gem_backend as gem


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
CAMPAIGNS = {
    "campaign_002": 12002,
    "campaign_003": 13003,
}
HYBRID_SEEDS = {
    "campaign_002": 22002,
    "campaign_003": 33003,
}
LEAK_TERMS = [
    "0.816502",
    "0.802500",
    "0.571041",
    "0.559266",
    "best_verified_objective",
    "deployment_benchmark_round_001",
    "design_benchmark_exact_pool_ranking",
    "design_benchmark_oracle_cache",
    "exact hidden",
    "hidden_regime",
    "hybrid_round_001",
    "open_ended_matched_round_001_scientist_vs_hybrid",
    "round_001_candidate_",
    "scientist_round_001",
    "strong_dynamic_regulation",
]


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_row(path: Path, item: str, value: str = "") -> dict[str, object]:
    return {
        "config_item": item,
        "value": value,
        "path": str(path.relative_to(ROOT)) if path.exists() and path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        "frozen_status": "frozen_for_report_checkpoint" if path.exists() else "missing",
    }


def write_frozen_config() -> None:
    DATA.mkdir(exist_ok=True)
    hidden_exact = DATA / "deployment_benchmark_round_001_exact_accounting.csv"
    exact_rows = pd.read_csv(hidden_exact) if hidden_exact.exists() else pd.DataFrame()
    exact_protocol = ""
    if not exact_rows.empty:
        lp_per_candidate = int(exact_rows["n_actual_lp_solves"].iloc[0])
        if {"n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations"}.issubset(exact_rows.columns):
            exact_protocol = (
                f"{lp_per_candidate}_lp_per_candidate;"
                f"{int(exact_rows['n_growth_optimizations'].iloc[0])}_growth;"
                f"{int(exact_rows['n_product_optimizations'].iloc[0])}_product;"
                f"{int(exact_rows['n_pfba_optimizations'].iloc[0])}_pfba"
            )
        else:
            exact_protocol = f"{lp_per_candidate}_lp_per_candidate;shared_full_verifier_protocol"

    rows = [
        file_row(DATA / "deployment_benchmark_objective_config.csv", "objective_configuration"),
        {
            "config_item": "objective_weights",
            "value": stable_json(deployment.OBJECTIVE_WEIGHTS),
            "path": "scripts/deployment_benchmark.py:OBJECTIVE_WEIGHTS",
            "sha256": sha256_text(stable_json(deployment.OBJECTIVE_WEIGHTS)),
            "frozen_status": "frozen_for_report_checkpoint",
        },
        {
            "config_item": "non_inferiority_margin",
            "value": str(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"]),
            "path": "scripts/deployment_benchmark.py:OBJECTIVE_WEIGHTS.noninferiority_delta",
            "sha256": sha256_text(str(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"])),
            "frozen_status": "frozen_for_report_checkpoint",
        },
        file_row(DATA / "deployment_benchmark_environment_limits.csv", "environment_limits"),
        file_row(DATA / "deployment_benchmark_editable_reaction_universe.csv", "editable_240_reaction_universe"),
        file_row(DATA / "deployment_benchmark_heterologous_library.csv", "heterologous_library"),
        file_row(DATA / "deployment_benchmark_sparse_edit_safeguards.csv", "edit_tiers_magnitudes_and_safeguards"),
        file_row(DATA / "deployment_benchmark_public_operational_budget.csv", "assay_panels_and_public_budget"),
        file_row(DATA / "deployment_benchmark_operational_scenarios.csv", "operational_scenarios"),
        file_row(ROOT / "src" / "yeast_validation" / "run_deployment_verifier.py", "exact_verifier"),
        file_row(ROOT / "src" / "yeast_validation" / "design_benchmark_exact.py", "hidden_regulator_version"),
        file_row(ROOT / "src" / "yeast_validation" / "gem_backend.py", "integration_convention_and_yeast9_backend"),
        file_row(ROOT / "src" / "yeast_validation" / "run_blinded_hybrid_round_001.py", "hybrid_controller_configuration"),
        file_row(ROOT / "src" / "yeast_validation" / "deployment_benchmark.py", "candidate_hashing_implementation"),
        file_row(DATA / "hybrid_student_manifest.csv", "hybrid_checkpoint"),
        file_row(DATA / "hybrid_student_metrics.csv", "hybrid_checkpoint_metrics"),
        {
            "config_item": "yeast9_checksum",
            "value": gem.sha256(gem.configured_gem_path()),
            "path": str(gem.configured_gem_path()),
            "sha256": gem.sha256(gem.configured_gem_path()),
            "frozen_status": "frozen_for_report_checkpoint",
        },
        {
            "config_item": "time_grid_and_exact_protocol",
            "value": exact_protocol or "not_inferred",
            "path": "data/deployment_benchmark_round_001_exact_accounting.csv",
            "sha256": sha256_text(exact_protocol),
            "frozen_status": "frozen_for_report_checkpoint" if exact_protocol else "missing",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_report_checkpoint_frozen_config.csv", index=False)


def approved_public_sources() -> list[Path]:
    existing_bundle = RESULTS / "public_scientist_bundle"
    sources = [
        existing_bundle / "scientist_agent_tools.py",
        existing_bundle / "deployment_benchmark_request_schema.json",
        existing_bundle / "deployment_benchmark_candidate_schema_example.json",
        existing_bundle / "deployment_benchmark_editable_reaction_universe.csv",
        existing_bundle / "deployment_benchmark_heterologous_library.csv",
        existing_bundle / "deployment_benchmark_objective_config.csv",
        existing_bundle / "deployment_benchmark_environment_limits.csv",
        existing_bundle / "deployment_benchmark_public_operational_budget.csv",
        existing_bundle / "deployment_benchmark_operational_scenarios.csv",
        existing_bundle / "deployment_benchmark_agent_protocol.csv",
        existing_bundle / "deployment_benchmark_public_reaction_annotations.csv",
        existing_bundle / "deployment_benchmark_public_metabolite_annotations.csv",
        existing_bundle / "deployment_benchmark_public_tool_documentation.md",
        existing_bundle / "public_yeast-GEM.xml",
        DATA / "deployment_benchmark_static_solve_accounting.csv",
        DATA / "deployment_benchmark_sparse_edit_safeguards.csv",
    ]
    return [p for p in sources if p.exists()]


def write_handoff(bundle: Path, campaign_id: str, public_seed: int) -> Path:
    handoff = bundle / "SCIENTIST_AGENT_HANDOFF.md"
    text = "\n".join(
        [
            "# Scientist Agent Handoff",
            "",
            f"Campaign ID: `{campaign_id}`.",
            f"Declared public planning seed: `{public_seed}`.",
            "",
            "Objective: improve dynamic beta-carotene production while maintaining viable growth using sparse reaction-level metabolic edits and one fixed culture environment [temperature, pH, dissolved oxygen].",
            "",
            "Use only the files in this public bundle. Static FBA, pFBA, FVA, knockout, annotation, and baseline tables are public planning aids; they are not the hidden dynamic biological verifier.",
            "",
            "Submit exactly one first-round request at `exchange/incoming_requests/round_001_request.json`. Set JSON field `campaign_id` to the campaign ID above and `round_id` to `round_001`.",
            "",
            "Physical budget for this first request: at most 8 cultures, at most 6 unique strains, Tier 1 and Tier 2 designs only, at least one conservative design, and at least two mechanistically different hypotheses.",
            "",
            "For every candidate include candidate ID, strain ID, edit tier, reaction-level edits, environment, assay panel, hypothesis, expected failure mode, batch rationale, and what result would change the next decision.",
            "",
            "Do not use any prior campaign outcomes, external project narratives, nonpublic verifier outcomes, finite-pool rankings, hybrid candidates, hybrid predictions, or other method results.",
        ]
    )
    handoff.write_text(text + "\n", encoding="utf-8")
    return handoff


def scan_text(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".json", ".md", ".py", ".txt"}:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [term for term in LEAK_TERMS if term in text]


def manifest_for_bundle(bundle: Path, campaign_id: str) -> pd.DataFrame:
    rows = []
    for path in sorted(bundle.rglob("*")):
        rel = path.relative_to(bundle)
        if path.is_dir():
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "path": str(rel),
                    "artifact_type": "directory",
                    "sha256": "",
                    "approved_public_asset": True,
                    "leak_scan_terms_found": "",
                    "contains_hidden_or_prior_outcome": False,
                }
            )
            continue
        leaks = scan_text(path)
        rows.append(
            {
                "campaign_id": campaign_id,
                "path": str(rel),
                "artifact_type": "file",
                "sha256": sha256_file(path),
                "approved_public_asset": not leaks,
                "leak_scan_terms_found": ";".join(leaks),
                "contains_hidden_or_prior_outcome": bool(leaks),
            }
        )
    return pd.DataFrame(rows)


def copy_public_bundle(campaign_id: str, public_seed: int) -> pd.DataFrame:
    campaign_root = RESULTS / campaign_id
    bundle = campaign_root / "public_scientist_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    for src in approved_public_sources():
        shutil.copy2(src, bundle / src.name)
    write_handoff(bundle, campaign_id, public_seed)
    (bundle / "exchange" / "incoming_requests").mkdir(parents=True, exist_ok=True)
    (bundle / "exchange" / "public_responses").mkdir(parents=True, exist_ok=True)
    (bundle / "exchange" / "tool_logs").mkdir(parents=True, exist_ok=True)
    manifest = manifest_for_bundle(bundle, campaign_id)
    manifest.to_csv(DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv", index=False)
    leakage = pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "audit_item": "public_manifest_files_approved",
                "value": bool(manifest["approved_public_asset"].all()),
                "passed": bool(manifest["approved_public_asset"].all()),
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "prior_or_hidden_leak_terms_found",
                "value": ";".join(sorted(set(";".join(manifest["leak_scan_terms_found"].dropna()).split(";")) - {""})),
                "passed": not bool(";".join(manifest["leak_scan_terms_found"].dropna()).strip(";")),
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "campaign_one_assets_excluded",
                "value": "round_001_results_and_hybrid_round_001_assets_not_copied",
                "passed": True,
            },
            {
                "campaign_id": campaign_id,
                "audit_item": "fresh_scientist_request_available",
                "value": False,
                "passed": False,
            },
        ]
    )
    leakage.to_csv(DATA / f"deployment_benchmark_{campaign_id}_leakage_audit.csv", index=False)
    return manifest


def write_scientist_provenance(campaign_id: str, manifest: pd.DataFrame) -> None:
    manifest_path = DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv"
    handoff_path = RESULTS / campaign_id / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md"
    rows = [
        {
            "campaign_id": campaign_id,
            "artifact": "public_bundle_manifest",
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
            "value": sha256_text(stable_json(manifest[["path", "sha256"]].to_dict(orient="records"))),
            "provenance_status": "scientist_context_isolation_unverified",
            "notes": "public bundle prepared; fresh scientist task not yet available",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "scientist_prompt",
            "path": str(handoff_path.relative_to(ROOT)),
            "sha256": sha256_file(handoff_path),
            "value": "",
            "provenance_status": "scientist_context_isolation_unverified",
            "notes": "handoff prompt prepared for future fresh task",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "fresh_task_session_identifier",
            "path": "",
            "sha256": "",
            "value": "",
            "provenance_status": "scientist_context_isolation_unverified",
            "notes": "dependency missing: no fresh task/session identifier supplied yet",
        },
        {
            "campaign_id": campaign_id,
            "artifact": "submitted_request",
            "path": "",
            "sha256": "",
            "value": "",
            "provenance_status": "scientist_context_isolation_unverified",
            "notes": "dependency missing: no scientist request supplied yet; hidden administrator has not edited any request",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_provenance.csv", index=False)


def write_hybrid_access_placeholders(campaign_id: str) -> None:
    blind = RESULTS / campaign_id / "hybrid_blind"
    inputs = blind / "inputs"
    outputs = blind / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    allowed = [
        "deployment_benchmark_report_checkpoint_frozen_config.csv",
        "deployment_benchmark_editable_reaction_universe.csv",
        "deployment_benchmark_heterologous_library.csv",
        "deployment_benchmark_objective_config.csv",
        "deployment_benchmark_environment_limits.csv",
        "deployment_benchmark_public_operational_budget.csv",
        "hybrid_student_manifest.csv",
        "hybrid_student_metrics.csv",
    ]
    denied = [
        "campaign_001_scientist_candidates_and_results",
        "campaign_002_scientist_candidates_or_results",
        "campaign_003_scientist_candidates_or_results",
        "hybrid_round_001_exact_results",
        "old_exact_pool_rankings",
        "hidden_exact_outcome_tables",
        "project_documents_revealing_successful_edits",
    ]
    pd.DataFrame({"campaign_id": campaign_id, "allowed_input": allowed}).to_csv(
        DATA / f"deployment_benchmark_{campaign_id}_hybrid_allowed_inputs.csv", index=False
    )
    pd.DataFrame({"campaign_id": campaign_id, "denied_input": denied}).to_csv(
        DATA / f"deployment_benchmark_{campaign_id}_hybrid_denied_inputs.csv", index=False
    )
    pd.DataFrame(
        [
            {"campaign_id": campaign_id, "audit_item": "hybrid_blind_workspace_created", "value": str(blind), "passed": True},
            {"campaign_id": campaign_id, "audit_item": "hybrid_seed_declared", "value": HYBRID_SEEDS[campaign_id], "passed": True},
            {"campaign_id": campaign_id, "audit_item": "hybrid_search_started", "value": False, "passed": False},
            {"campaign_id": campaign_id, "audit_item": "blocking_dependency", "value": "fresh scientist request not yet available; stopped before hybrid search", "passed": True},
        ]
    ).to_csv(DATA / f"deployment_benchmark_{campaign_id}_hybrid_access_audit.csv", index=False)


def write_campaign_001_freeze() -> None:
    artifacts = [
        RESULTS / "exchange" / "incoming_requests" / "round_001_request.json",
        RESULTS / "exchange" / "public_responses" / "round_001_results.json",
        RESULTS / "exchange" / "incoming_requests" / "hybrid_round_001_request.json",
        RESULTS / "exchange" / "public_responses" / "hybrid_round_001_results.json",
        DATA / "deployment_benchmark_round_001_public_results.csv",
        DATA / "deployment_benchmark_round_001_exact_accounting.csv",
        DATA / "deployment_benchmark_hybrid_round_001_public_results.csv",
        DATA / "deployment_benchmark_hybrid_round_001_exact_accounting.csv",
        DATA / "deployment_benchmark_matched_round_001_comparison.csv",
        DATA / "deployment_benchmark_matched_round_001_acceptance.csv",
    ]
    rows = []
    for path in artifacts:
        rows.append(
            {
                "campaign_id": "campaign_001",
                "artifact": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "frozen_status": "frozen_do_not_modify" if path.exists() else "missing",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_campaign_001_freeze_manifest.csv", index=False)


def write_registry() -> None:
    rows = [
        {
            "campaign_id": "campaign_001",
            "phase": "pilot_one_shot",
            "status": "complete_frozen",
            "scientist_request_status": "complete",
            "hybrid_request_status": "complete",
            "exact_verification_status": "complete",
            "notes": "completed pilot preserved; do not modify selected candidates or results",
        }
    ]
    for campaign_id in CAMPAIGNS:
        rows.append(
            {
                "campaign_id": campaign_id,
                "phase": "replication_one_shot",
                "status": "prepared_waiting_fresh_scientist_request",
                "scientist_request_status": "missing",
                "hybrid_request_status": "not_started_by_stop_condition",
                "exact_verification_status": "not_started",
                "notes": "public scientist bundle prepared; stop condition prevents scripted substitution",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_campaign_registry.csv", index=False)


def write_checkpoint_status() -> None:
    pd.DataFrame(
        [
            {"summary_item": "checkpoint_status", "value": "blocked_waiting_campaign_002_003_scientist_requests"},
            {"summary_item": "completed_campaigns_frozen", "value": "campaign_001"},
            {"summary_item": "prepared_public_bundles", "value": "campaign_002;campaign_003"},
            {"summary_item": "new_exact_simulations_run", "value": "0"},
            {"summary_item": "new_exact_lp_solves_run", "value": "0"},
            {"summary_item": "reason_for_stop", "value": "fresh scientist Campaign 002/003 requests not yet available"},
        ]
    ).to_csv(DATA / "deployment_benchmark_report_checkpoint_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "status_type": "report_checkpoint",
                "status": "blocked_waiting_fresh_scientist_requests",
                "passed": False,
                "notes": "per instruction, prepared bundles and stopped rather than substituting a scripted scientist",
            },
            {
                "status_type": "campaign_001_freeze",
                "status": "campaign_001_artifacts_frozen",
                "passed": True,
                "notes": "hash manifest written; selected candidates/results not modified",
            },
            {
                "status_type": "configuration_freeze",
                "status": "report_checkpoint_configuration_frozen",
                "passed": True,
                "notes": "objective, universe, verifier, Yeast9 checksum, hybrid checkpoint and hashing implementation recorded",
            },
        ]
    ).to_csv(DATA / "deployment_benchmark_report_checkpoint_acceptance.csv", index=False)


def main() -> None:
    write_frozen_config()
    write_campaign_001_freeze()
    manifests = {}
    for campaign_id, public_seed in CAMPAIGNS.items():
        manifests[campaign_id] = copy_public_bundle(campaign_id, public_seed)
        write_scientist_provenance(campaign_id, manifests[campaign_id])
        write_hybrid_access_placeholders(campaign_id)
    write_registry()
    write_checkpoint_status()
    print(pd.read_csv(DATA / "deployment_benchmark_campaign_registry.csv").to_string(index=False))
    print(pd.read_csv(DATA / "deployment_benchmark_report_checkpoint_acceptance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
