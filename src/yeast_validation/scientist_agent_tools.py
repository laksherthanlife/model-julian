#!/usr/bin/env python3
"""Public scientist-agent tool surface for the deployment benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import gem_backend as gem


ALLOWED_TOOL_NAMES = [
    "inspect_model_summary",
    "search_reactions",
    "inspect_reaction",
    "inspect_metabolite",
    "inspect_metabolite_neighbourhood",
    "list_subsystem_reactions",
    "inspect_gene_reaction_rule",
    "inspect_baseline_data",
    "run_static_fba",
    "run_static_pfba",
    "run_fva",
    "run_reaction_knockout_screen",
    "compare_static_fluxes",
    "estimate_edit_construction_risk",
    "validate_candidate",
    "submit_verification_request",
]

TOOL_LOG = deployment.RESULTS / "public_tool_call_log.jsonl"


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def log_tool_call(tool: str, payload: dict[str, Any], result: Any) -> None:
    deployment.ensure_dirs()
    row = {
        "timestamp_unix": time.time(),
        "tool": tool,
        "payload_keys": sorted(payload),
        "hidden_verifier_called": False,
        "result_status": _jsonable(result).get("status", "ok") if isinstance(_jsonable(result), dict) else "ok",
    }
    with TOOL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _universe() -> pd.DataFrame:
    path = deployment.DATA / "deployment_benchmark_editable_reaction_universe.csv"
    if not path.exists():
        deployment.build_editable_reaction_universe()
    return pd.read_csv(path)


def _annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    r_path = deployment.DATA / "deployment_benchmark_public_reaction_annotations.csv"
    m_path = deployment.DATA / "deployment_benchmark_public_metabolite_annotations.csv"
    if not r_path.exists() or not m_path.exists():
        deployment.write_public_schemas_and_annotations()
    return pd.read_csv(r_path), pd.read_csv(m_path)


def _model_with_edits(candidate: dict[str, Any] | None = None):
    cobra, source_path, model = deployment.public_static_augmented_model()
    if candidate:
        deployment.apply_sparse_edits_to_model(model, candidate)
    return cobra, source_path, model


def inspect_model_summary() -> dict[str, object]:
    universe = _universe()
    reactions, metabolites = _annotations()
    source = gem.configured_gem_path()
    result = {
        "editable_reactions": int(len(universe)),
        "native_reaction_ids": int(universe["reaction_id"].astype(str).str.match(r"r_\d+").sum()),
        "subsystems": int(universe["subsystem"].nunique()),
        "public_reaction_annotations": int(len(reactions)),
        "public_metabolite_annotations": int(len(metabolites)),
        "public_model_asset": str(source.name),
        "public_model_sha256": gem.sha256(source),
        "allowed_tool_names": ALLOWED_TOOL_NAMES,
        "hidden_regulator_available": False,
        "hidden_exact_outcomes_available_before_verification": False,
    }
    log_tool_call("inspect_model_summary", {}, result)
    return result


def search_reactions(query: str = "", limit: int = 25) -> list[dict[str, object]]:
    universe = _universe()
    text = universe.astype(str).agg(" ".join, axis=1).str.lower()
    out = universe[text.str.contains(query.lower(), regex=False)].head(limit).copy()
    result = out.to_dict(orient="records")
    log_tool_call("search_reactions", {"query": query, "limit": limit}, {"status": "ok", "rows": len(result)})
    return result


def inspect_reaction(reaction_id: str) -> dict[str, object]:
    universe = _universe()
    reactions, _metabolites = _annotations()
    match = universe[universe["reaction_id"].astype(str).eq(reaction_id)]
    ann = reactions[reactions["reaction_id"].astype(str).eq(reaction_id)]
    result = {"reaction_id": reaction_id, "found": not match.empty or not ann.empty}
    if not match.empty:
        result.update(match.iloc[0].to_dict())
    if not ann.empty:
        result["annotation"] = ann.iloc[0].to_dict()
    log_tool_call("inspect_reaction", {"reaction_id": reaction_id}, result)
    return result


def inspect_metabolite(metabolite_id: str) -> dict[str, object]:
    _reactions, metabolites = _annotations()
    match = metabolites[metabolites["metabolite_id"].astype(str).eq(metabolite_id)]
    result = {"metabolite_id": metabolite_id, "found": not match.empty}
    if not match.empty:
        result.update(match.iloc[0].to_dict())
    log_tool_call("inspect_metabolite", {"metabolite_id": metabolite_id}, result)
    return result


def inspect_metabolite_neighbourhood(metabolite_id: str, depth: int = 1, limit: int = 50) -> list[dict[str, object]]:
    _cobra, _source_path, model = _model_with_edits()
    if metabolite_id not in {m.id for m in model.metabolites}:
        result: list[dict[str, object]] = []
    else:
        met = model.metabolites.get_by_id(metabolite_id)
        rows = []
        for rxn in sorted(met.reactions, key=lambda r: r.id)[:limit]:
            rows.append({"metabolite_id": metabolite_id, "depth": depth, "reaction_id": rxn.id, "reaction_name": rxn.name, "equation": rxn.reaction})
        result = rows
    log_tool_call("inspect_metabolite_neighbourhood", {"metabolite_id": metabolite_id, "depth": depth}, {"status": "ok", "rows": len(result)})
    return result


def list_subsystem_reactions(subsystem: str, limit: int = 50) -> list[dict[str, object]]:
    universe = _universe()
    out = universe[universe["subsystem"].astype(str).str.contains(subsystem, case=False, regex=False)].head(limit)
    result = out.to_dict(orient="records")
    log_tool_call("list_subsystem_reactions", {"subsystem": subsystem, "limit": limit}, {"status": "ok", "rows": len(result)})
    return result


def inspect_gene_reaction_rule(reaction_id: str) -> dict[str, object]:
    reaction = inspect_reaction(reaction_id)
    result = {"reaction_id": reaction_id, "found": bool(reaction.get("found")), "gene_reaction_rule": reaction.get("gene_association", reaction.get("gene_reaction_rule", ""))}
    log_tool_call("inspect_gene_reaction_rule", {"reaction_id": reaction_id}, result)
    return result


def inspect_baseline_data(filters: dict[str, object] | None = None) -> dict[str, object]:
    paths = [
        deployment.DATA / "deployment_benchmark_static_solve_accounting.csv",
        deployment.DATA / "deployment_benchmark_editable_reaction_universe.csv",
        deployment.DATA / "gem_state_space_environment_grid.csv",
        deployment.DATA / "gem_state_space_reporters.csv",
    ]
    result = {path.name: {"exists": path.exists(), "rows": len(pd.read_csv(path)) if path.exists() and path.suffix == ".csv" else 0} for path in paths}
    log_tool_call("inspect_baseline_data", filters or {}, result)
    return result


def _candidate_from_payload(environment: dict[str, float] | None, edits: list[dict[str, object]] | None) -> dict[str, object]:
    return {
        "candidate_id": "public_static_tool_candidate",
        "strain_id": "public_static_tool_strain",
        "edit_tier": 2,
        "edits": edits or [],
        "environment": environment or {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
        "requested_assay_panel": "static_public",
        "hypothesis": "public static tool call",
        "expected_failure_mode": "unknown",
    }


def run_static_fba(environment: dict[str, float] | None = None, edits: list[dict[str, object]] | None = None) -> dict[str, object]:
    candidate = _candidate_from_payload(environment, edits)
    if not deployment.PUBLIC_STATIC_SOLVE_ENABLED:
        result = {"backend": "public_static_yeast9_fba", "status": "skipped_public_static_solves_disabled", "objective_value": None, "lp_solves": 0, "hidden_verifier_called": False}
        log_tool_call("run_static_fba", candidate, result)
        return result
    _cobra, source_path, model = _model_with_edits(candidate)
    model.objective = model.reactions.get_by_id(gem.BIOMASS_RXN)
    sol = model.optimize()
    result = {
        "backend": "public_static_yeast9_fba",
        "status": sol.status,
        "objective_reaction": gem.BIOMASS_RXN,
        "objective_value": float(sol.objective_value) if sol.objective_value is not None else None,
        "model_sha256": gem.sha256(source_path),
        "lp_solves": 1,
        "hidden_verifier_called": False,
    }
    log_tool_call("run_static_fba", candidate, result)
    return result


def run_static_pfba(environment: dict[str, float] | None = None, edits: list[dict[str, object]] | None = None, reaction_subset: list[str] | None = None) -> dict[str, object]:
    candidate = _candidate_from_payload(environment, edits)
    if not deployment.PUBLIC_STATIC_SOLVE_ENABLED:
        result = {"backend": "public_static_yeast9_pfba", "status": "skipped_public_static_solves_disabled", "fluxes": {}, "lp_solves": 0, "hidden_verifier_called": False}
        log_tool_call("run_static_pfba", candidate, result)
        return result
    cobra, source_path, model = _model_with_edits(candidate)
    model.objective = model.reactions.get_by_id(gem.BIOMASS_RXN)
    sol = cobra.flux_analysis.pfba(model)
    subset = reaction_subset or [gem.BIOMASS_RXN, gem.NATIVE_GGPP_RXN, gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.PRODUCT_RXN]
    result = {
        "backend": "public_static_yeast9_pfba",
        "status": sol.status,
        "objective_value": float(sol.objective_value) if sol.objective_value is not None else None,
        "fluxes": {rid: float(sol.fluxes.get(rid, np.nan)) for rid in subset},
        "model_sha256": gem.sha256(source_path),
        "lp_solves": 1,
        "hidden_verifier_called": False,
    }
    log_tool_call("run_static_pfba", {**candidate, "reaction_subset": subset}, result)
    return result


def run_fva(environment: dict[str, float] | None = None, edits: list[dict[str, object]] | None = None, reaction_subset: list[str] | None = None) -> dict[str, object]:
    candidate = _candidate_from_payload(environment, edits)
    if not deployment.PUBLIC_STATIC_SOLVE_ENABLED:
        result = {"backend": "public_static_yeast9_fva", "status": "skipped_public_static_solves_disabled", "fva": [], "lp_solves": 0, "hidden_verifier_called": False}
        log_tool_call("run_fva", candidate, result)
        return result
    cobra, source_path, model = _model_with_edits(candidate)
    subset = reaction_subset or [gem.BIOMASS_RXN, gem.NATIVE_GGPP_RXN, gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN]
    reaction_list = [model.reactions.get_by_id(rid) for rid in subset if rid in {r.id for r in model.reactions}]
    fva = cobra.flux_analysis.flux_variability_analysis(model, reaction_list=reaction_list, fraction_of_optimum=0.95, processes=1)
    result = {
        "backend": "public_static_yeast9_fva",
        "status": "complete",
        "fva": fva.reset_index(names="reaction_id").to_dict(orient="records"),
        "model_sha256": gem.sha256(source_path),
        "lp_solves": 2 * len(reaction_list),
        "hidden_verifier_called": False,
    }
    log_tool_call("run_fva", {**candidate, "reaction_subset": subset}, result)
    return result


def run_reaction_knockout_screen(reaction_ids: list[str], environment: dict[str, float] | None = None) -> dict[str, object]:
    rows = []
    for rid in reaction_ids:
        edit = [{"reaction_id": rid, "edit_type": "reaction_knockout", "capacity_multiplier": 0.0, "rationale": "public knockout screen"}]
        try:
            fba = run_static_fba(environment, edit)
            rows.append({"reaction_id": rid, "status": fba["status"], "growth_objective": fba["objective_value"]})
        except Exception as exc:
            rows.append({"reaction_id": rid, "status": f"failed:{type(exc).__name__}", "growth_objective": None})
    result = {"backend": "public_static_yeast9_knockout_screen", "status": "complete", "rows": rows, "hidden_verifier_called": False}
    log_tool_call("run_reaction_knockout_screen", {"reaction_ids": reaction_ids}, result)
    return result


def compare_static_fluxes(candidate_a: dict[str, object], candidate_b: dict[str, object], reaction_subset: list[str] | None = None) -> dict[str, object]:
    a = run_static_pfba(candidate_a.get("environment"), candidate_a.get("edits"), reaction_subset)
    b = run_static_pfba(candidate_b.get("environment"), candidate_b.get("edits"), reaction_subset)
    flux_a = a["fluxes"]
    flux_b = b["fluxes"]
    result = {"backend": "public_static_flux_comparison", "status": "complete", "delta_fluxes": {rid: float(flux_b.get(rid, np.nan) - flux_a.get(rid, np.nan)) for rid in flux_a}, "hidden_verifier_called": False}
    log_tool_call("compare_static_fluxes", {"candidate_a": candidate_a.get("candidate_id"), "candidate_b": candidate_b.get("candidate_id")}, result)
    return result


def estimate_edit_construction_risk(candidate: dict[str, object]) -> dict[str, object]:
    universe = _universe().set_index("reaction_id")
    risks = []
    for edit in candidate.get("edits", []):
        rid = str(edit.get("reaction_id", ""))
        risks.append(str(universe.loc[rid, "risk_level"]) if rid in universe.index else "curated_or_unknown")
    result = {"status": "complete", "edit_count": len(candidate.get("edits", [])), "risk_categories": risks, "construction_risk": "high" if "high" in risks else "medium" if "medium" in risks or "curated_or_unknown" in risks else "low", "hidden_verifier_called": False}
    log_tool_call("estimate_edit_construction_risk", candidate, result)
    return result


def validate_candidate(candidate: dict[str, object]) -> dict[str, object]:
    result = deployment.validate_sparse_design(candidate, _universe())
    result["hidden_verifier_called"] = False
    log_tool_call("validate_candidate", candidate, result)
    return result


def submit_verification_request(request: dict[str, object]) -> dict[str, object]:
    deployment.ensure_dirs()
    round_id = str(request.get("round_id", "round_001"))
    out = deployment.INCOMING_REQUESTS / f"{round_id}_request.json"
    out.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "request_written_not_executed", "path": str(out), "candidate_count": len(request.get("candidates", [])), "hidden_verifier_called": False}
    log_tool_call("submit_verification_request", {"round_id": round_id}, result)
    return result


def compare_fluxes(candidate_a: dict[str, object], candidate_b: dict[str, object]) -> dict[str, object]:
    return compare_static_fluxes(candidate_a, candidate_b)


def request_verification(candidate_batch: list[dict[str, object]], assay_panel: str = "basic_product_biomass") -> dict[str, object]:
    request = {
        "round_id": "round_001",
        "campaign_id": "legacy_public_tool_request",
        "assay_panel": assay_panel,
        "candidates": candidate_batch,
    }
    result = submit_verification_request(request)
    result["status"] = "not_executed_by_public_tool"
    return result


def dispatch(command: str, payload: dict[str, Any]) -> Any:
    table = {
        "inspect_model_summary": lambda p: inspect_model_summary(),
        "search_reactions": lambda p: search_reactions(str(p.get("query", "")), int(p.get("limit", 25))),
        "inspect_reaction": lambda p: inspect_reaction(str(p["reaction_id"])),
        "inspect_metabolite": lambda p: inspect_metabolite(str(p["metabolite_id"])),
        "inspect_metabolite_neighbourhood": lambda p: inspect_metabolite_neighbourhood(str(p["metabolite_id"]), int(p.get("depth", 1)), int(p.get("limit", 50))),
        "list_subsystem_reactions": lambda p: list_subsystem_reactions(str(p["subsystem"]), int(p.get("limit", 50))),
        "inspect_gene_reaction_rule": lambda p: inspect_gene_reaction_rule(str(p["reaction_id"])),
        "inspect_baseline_data": lambda p: inspect_baseline_data(p),
        "run_static_fba": lambda p: run_static_fba(p.get("environment"), p.get("edits")),
        "run_static_pfba": lambda p: run_static_pfba(p.get("environment"), p.get("edits"), p.get("reaction_subset")),
        "run_fva": lambda p: run_fva(p.get("environment"), p.get("edits"), p.get("reaction_subset")),
        "run_reaction_knockout_screen": lambda p: run_reaction_knockout_screen(list(p.get("reaction_ids", [])), p.get("environment")),
        "compare_static_fluxes": lambda p: compare_static_fluxes(p["candidate_a"], p["candidate_b"], p.get("reaction_subset")),
        "estimate_edit_construction_risk": lambda p: estimate_edit_construction_risk(p["candidate"]),
        "validate_candidate": lambda p: validate_candidate(p["candidate"]),
        "submit_verification_request": lambda p: submit_verification_request(p),
    }
    if command not in table:
        raise SystemExit(f"Unknown command {command!r}. Allowed: {', '.join(ALLOWED_TOOL_NAMES)}")
    return table[command](payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=ALLOWED_TOOL_NAMES)
    parser.add_argument("--json", default="{}", help="JSON payload for the command")
    args = parser.parse_args()
    payload = json.loads(args.json)
    print(json.dumps(_jsonable(dispatch(args.command, payload)), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
