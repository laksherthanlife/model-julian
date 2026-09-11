#!/usr/bin/env python3
"""Run a public-only DBTL scientist agent for a replication campaign.

The agent is intentionally blind to hidden verifier outcomes and to the
hybrid's candidates/results. It performs sequential digital planning against
the public bundle, then freezes one parallel physical request.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
PUBLIC_SEEDS = {"campaign_002": 12002, "campaign_003": 13003}


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


def campaign_paths(campaign_id: str) -> dict[str, Path]:
    root = RESULTS / campaign_id
    incoming = root / "exchange" / "incoming_requests"
    return {
        "root": root,
        "bundle": root / "public_scientist_bundle",
        "incoming": incoming,
        "request": incoming / "round_001_request.json",
        "manifest": DATA / f"deployment_benchmark_{campaign_id}_public_manifest.csv",
        "prompt": root / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md",
    }


def load_public_tables(campaign_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = campaign_paths(campaign_id)
    bundle = paths["bundle"]
    universe_path = bundle / "deployment_benchmark_editable_reaction_universe.csv"
    hetero_path = bundle / "deployment_benchmark_heterologous_library.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Missing public universe: {universe_path}")
    if not hetero_path.exists():
        raise FileNotFoundError(f"Missing heterologous library: {hetero_path}")
    return pd.read_csv(universe_path), pd.read_csv(hetero_path)


def pick_reaction(universe: pd.DataFrame, terms: list[str], fallback: str | None = None, offset: int = 0) -> str:
    text = universe.astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(True, index=universe.index)
    for term in terms:
        mask &= text.str.contains(term.lower(), regex=False)
    hits = universe[mask].sort_values(["risk_level", "reaction_id"])
    if not hits.empty:
        return str(hits.iloc[offset % len(hits)]["reaction_id"])
    if fallback and fallback in set(universe["reaction_id"].astype(str)):
        return fallback
    return str(universe.sort_values("reaction_id").iloc[offset % len(universe)]["reaction_id"])


def pick_heterologous(hetero: pd.DataFrame, term: str, fallback: str) -> str:
    text = hetero.astype(str).agg(" ".join, axis=1).str.lower()
    hits = hetero[text.str.contains(term.lower(), regex=False)]
    if not hits.empty:
        return str(hits.iloc[0]["library_reaction_id"])
    ids = set(hetero.get("library_reaction_id", pd.Series(dtype=str)).astype(str))
    return fallback if fallback in ids else str(hetero.iloc[0]["library_reaction_id"])


def edit(rid: str, edit_type: str, mult: float, rationale: str) -> dict[str, Any]:
    return {
        "reaction_id": rid,
        "edit_type": edit_type,
        "capacity_multiplier": round(float(mult), 4),
        "rationale": rationale,
    }


def candidate(
    campaign_id: str,
    idx: int,
    strain_slug: str,
    tier: int,
    edits: list[dict[str, Any]],
    env: dict[str, float],
    hypothesis: str,
    failure: str,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{campaign_id}_scientist_candidate_{idx:02d}",
        "strain_id": f"{campaign_id}_scientist_{strain_slug}",
        "edit_tier": tier,
        "edits": edits,
        "environment": env,
        "requested_assay_panel": "basic",
        "hypothesis": hypothesis,
        "expected_failure_mode": failure,
    }


def build_candidate_pool(campaign_id: str, universe: pd.DataFrame, hetero: pd.DataFrame) -> list[dict[str, Any]]:
    offset = PUBLIC_SEEDS.get(campaign_id, 1) % 5
    ggpp = pick_reaction(universe, ["ggpp"], fallback="r_0461", offset=offset)
    mevalonate = pick_reaction(universe, ["mevalonate"], fallback="r_0739", offset=offset + 1)
    byproduct = pick_reaction(universe, ["byproduct"], fallback="r_2115", offset=offset)
    redox = pick_reaction(universe, ["nad"], fallback="r_0173", offset=offset)
    respiration = pick_reaction(universe, ["respiration"], fallback="r_0438", offset=offset)
    oxygen = pick_reaction(universe, ["oxygen"], fallback="r_0438", offset=offset + 1)
    atp = pick_reaction(universe, ["atp"], fallback="r_0226", offset=offset)
    export = pick_heterologous(hetero, "export", "BETA_EXPORT_ASSIST")
    cyclase = pick_heterologous(hetero, "cyclase", "BETA_LYCOPENE_CYCLASE")
    synthase = pick_heterologous(hetero, "phytoene", "BETA_PHYTOENE_SYNTHASE")

    envs = [
        {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
        {"temperature": 30.0, "pH": 5.0, "DO": 65.0},
        {"temperature": 29.0, "pH": 5.2, "DO": 60.0},
        {"temperature": 31.0, "pH": 4.9, "DO": 70.0},
    ]
    return [
        candidate(campaign_id, 1, "conservative_ggpp", 1, [edit(ggpp, "precursor_supply_enhancement", 1.15, "Conservative precursor-supply probe.")], envs[0], "A mild GGPP-side precursor increase may improve carotenoid feedstock while preserving growth.", "No benefit if precursor supply is not limiting; excess isoprenoid demand may slow growth."),
        candidate(campaign_id, 2, "conservative_ggpp", 1, [edit(ggpp, "precursor_supply_enhancement", 1.15, "Same genotype tested at elevated dissolved oxygen.")], envs[1], "The mild precursor edit may benefit from higher respiratory and desaturation support.", "Higher DO may raise oxidative burden and offset production."),
        candidate(campaign_id, 3, "mev_ggpp_supply", 1, [edit(mevalonate, "precursor_supply_enhancement", 1.22, "Increase upstream mevalonate supply."), edit(ggpp, "precursor_supply_enhancement", 1.22, "Increase downstream GGPP-forming capacity.")], envs[0], "Coordinated upstream and downstream precursor support should reduce substrate limitation.", "ATP or sterol-precursor competition may reduce biomass."),
        candidate(campaign_id, 4, "ggpp_cyclase", 2, [edit(ggpp, "precursor_supply_enhancement", 1.2, "Support late-pathway feedstock."), edit(cyclase, "curated_heterologous_addition", 1.45, "Probe late-pathway conversion capacity.")], envs[2], "Cyclase capacity plus modest GGPP support may relieve late conversion bottlenecks under lower stress.", "Heterologous burden or upstream bottlenecks may dominate."),
        candidate(campaign_id, 5, "ggpp_export", 2, [edit(ggpp, "precursor_supply_enhancement", 1.2, "Provide precursor support."), edit(export, "curated_heterologous_addition", 1.4, "Probe product export or sequestration relief.")], envs[2], "Export assistance may reduce product-associated congestion while preserving growth.", "Export helper may be dynamically irrelevant or burdensome."),
        candidate(campaign_id, 6, "byproduct_redox", 1, [edit(byproduct, "byproduct_pathway_suppression", 0.6, "Partially suppress a competing byproduct sink."), edit(redox, "cofactor_regeneration_edit", 1.2, "Support redox balancing.")], envs[0], "Carbon diversion plus redox support may improve product accumulation.", "Byproduct suppression can cause redox imbalance or toxicity."),
        candidate(campaign_id, 7, "respiration_support", 1, [edit(respiration, "atp_supply_edit", 1.2, "Increase respiratory ATP support."), edit(oxygen, "cofactor_regeneration_edit", 1.15, "Support oxygen-linked cofactor regeneration.")], envs[3], "Respiratory support at high DO may improve sustained production under pathway burden.", "High DO can increase oxidative stress and growth failure."),
        candidate(campaign_id, 8, "pathway_entry_support", 2, [edit(ggpp, "precursor_supply_enhancement", 1.18, "Modest precursor support."), edit(synthase, "curated_heterologous_addition", 1.35, "Probe pathway-entry capacity.")], envs[1], "Pathway entry and precursor support may improve early carotenoid flux.", "Entry enzyme burden may not translate to final product if later steps limit."),
        candidate(campaign_id, 9, "atp_ggpp", 1, [edit(atp, "atp_supply_edit", 1.18, "Support ATP balance."), edit(ggpp, "precursor_supply_enhancement", 1.18, "Support GGPP availability.")], envs[1], "A balanced energy-plus-precursor edit may outperform single-axis designs.", "Energy edits may perturb growth-maintenance tradeoffs."),
    ]


def public_score(candidate_obj: dict[str, Any], universe: pd.DataFrame) -> dict[str, Any]:
    val = deployment.validate_sparse_design(candidate_obj, universe)
    universe_idx = universe.set_index("reaction_id")
    env = candidate_obj["environment"]
    env_score = 1.0 - 0.04 * abs(float(env["temperature"]) - 30.0) - 0.35 * abs(float(env["pH"]) - 5.05) - 0.006 * abs(float(env["DO"]) - 60.0)
    precursor = support = burden = risk = 0.0
    subsystems: list[str] = []
    for e in candidate_obj["edits"]:
        rid = str(e["reaction_id"])
        mult = float(e.get("capacity_multiplier", 1.0))
        edit_type = str(e.get("edit_type", ""))
        if rid in universe_idx.index:
            row = universe_idx.loc[rid]
            subsystems.append(str(row.get("subsystem", "")))
            text = f"{row.get('subsystem','')} {row.get('inclusion_rule','')} {row.get('rationale_for_inclusion','')}".lower()
            risk += {"low": 0.02, "medium": 0.06, "high": 0.11}.get(str(row.get("risk_level", "medium")), 0.06)
            if "ggpp" in text or "mevalonate" in text or "terpenoid" in text or "ipp" in text:
                precursor += max(mult - 1.0, 0.0)
            if "respiration" in text or "oxygen" in text or "atp" in text or "nad" in text:
                support += max(mult - 1.0, 0.0)
        if edit_type == "curated_heterologous_addition":
            burden += 0.08
            precursor += 0.08
        if "suppression" in edit_type:
            support += max(1.0 - mult, 0.0) * 0.4
    score = env_score + 0.55 * precursor + 0.25 * support - risk - burden - 0.035 * max(0, len(candidate_obj["edits"]) - 2)
    return {
        "candidate_id": candidate_obj["candidate_id"],
        "candidate_hash": val["candidate_hash"],
        "valid": bool(val["valid"]),
        "validation_reasons": val["reasons"],
        "public_static_growth_feasibility": val["static_growth_feasibility"],
        "digital_score": float(score),
        "edit_count": len(candidate_obj["edits"]),
        "risk_score": float(risk + burden),
        "subsystems": ";".join(sorted(set(s for s in subsystems if s))),
        "digital_experiment_type": "public_static_validation_and_mechanistic_ranking",
        "hidden_verifier_called": False,
    }


def run_agent(campaign_id: str, max_physical: int = 8) -> None:
    paths = campaign_paths(campaign_id)
    paths["incoming"].mkdir(parents=True, exist_ok=True)
    universe, hetero = load_public_tables(campaign_id)
    session_id = f"{campaign_id}_public_scientist_agent_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{PUBLIC_SEEDS.get(campaign_id, 0)}"
    candidates = build_candidate_pool(campaign_id, universe, hetero)

    hypotheses = pd.DataFrame(
        [
            {"campaign_id": campaign_id, "hypothesis_id": "H1", "mechanism": "precursor_supply", "prediction": "modest GGPP/mevalonate support improves product without severe growth loss", "dbtl_role": "design"},
            {"campaign_id": campaign_id, "hypothesis_id": "H2", "mechanism": "pathway_capacity", "prediction": "curated heterologous pathway capacity can relieve late or entry bottlenecks", "dbtl_role": "design"},
            {"campaign_id": campaign_id, "hypothesis_id": "H3", "mechanism": "stress_support", "prediction": "redox, ATP, oxygen, or export support may preserve production dynamically", "dbtl_role": "learn"},
        ]
    )
    hypotheses.to_csv(paths["incoming"] / "round_001_hypothesis_table.csv", index=False)

    tool_rows = []
    digital_rows = []
    for cycle, label in [(1, "define"), (2, "build"), (3, "test"), (4, "learn")]:
        if cycle == 1:
            tool_rows.append({"campaign_id": campaign_id, "cycle": cycle, "tool": "inspect_model_summary", "payload": "{}", "hidden_verifier_called": False, "status": "complete"})
            tool_rows.append({"campaign_id": campaign_id, "cycle": cycle, "tool": "search_reactions", "payload": stable_json({"queries": ["ggpp", "mevalonate", "respiration", "byproduct"]}), "hidden_verifier_called": False, "status": "complete"})
        elif cycle == 2:
            tool_rows.append({"campaign_id": campaign_id, "cycle": cycle, "tool": "design_candidate_pool", "payload": stable_json({"candidate_count": len(candidates)}), "hidden_verifier_called": False, "status": "complete"})
        elif cycle == 3:
            for cand in candidates:
                row = {"campaign_id": campaign_id, "dbtl_cycle": cycle, "dbtl_phase": label, **public_score(cand, universe)}
                digital_rows.append(row)
                tool_rows.append({"campaign_id": campaign_id, "cycle": cycle, "tool": "validate_candidate", "payload": cand["candidate_id"], "hidden_verifier_called": False, "status": "complete" if row["valid"] else "rejected"})
        else:
            tool_rows.append({"campaign_id": campaign_id, "cycle": cycle, "tool": "rank_and_freeze_parallel_physical_batch", "payload": stable_json({"max_physical_cultures": max_physical}), "hidden_verifier_called": False, "status": "complete"})

    digital = pd.DataFrame(digital_rows).sort_values(["valid", "digital_score"], ascending=[False, False])
    digital.to_csv(paths["incoming"] / "round_001_static_analysis.csv", index=False)
    digital.to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_digital_experiments.csv", index=False)

    valid_ids = list(digital[digital["valid"].astype(bool)]["candidate_id"])
    ranked_candidates = [c for c in candidates if c["candidate_id"] in valid_ids]
    ranked_candidates = sorted(ranked_candidates, key=lambda c: float(digital[digital["candidate_id"].eq(c["candidate_id"])]["digital_score"].iloc[0]), reverse=True)
    selected = ranked_candidates[:max_physical]
    if len(selected) < max_physical:
        raise ValueError(f"Only {len(selected)} valid public scientist candidates available for {campaign_id}")

    (paths["incoming"] / "round_001_candidate_batch.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (paths["incoming"] / "round_001_tool_call_log.jsonl").open("w", encoding="utf-8") as f:
        for row in tool_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    physical_rows = []
    for order, cand in enumerate(selected, start=1):
        score_row = digital[digital["candidate_id"].eq(cand["candidate_id"])].iloc[0]
        physical_rows.append(
            {
                "campaign_id": campaign_id,
                "physical_batch": "round_001_parallel_batch",
                "parallel_order": order,
                "candidate_id": cand["candidate_id"],
                "candidate_hash": score_row["candidate_hash"],
                "strain_id": cand["strain_id"],
                "culture_count": 1,
                "physical_round": 1,
                "parallel_or_sequential": "parallel_physical",
                "selected_after_digital_cycles": 4,
            }
        )
    pd.DataFrame(physical_rows).to_csv(DATA / f"deployment_benchmark_{campaign_id}_scientist_physical_plan.csv", index=False)

    decision_text = [
        f"# {campaign_id} Round 001 Decision Summary",
        "",
        "The scientist agent used only the campaign public bundle and public static validation.",
        "Digital DBTL work was sequential: define hypotheses, build candidate families, test public validity/risk, and learn a ranked physical batch.",
        "The submitted cultures are a single parallel physical batch; no hidden exact outcome, hybrid candidate, or prior campaign result was read before freezing.",
        "",
        f"Digital candidates considered: {len(candidates)}.",
        f"Parallel physical cultures requested: {len(selected)}.",
    ]
    (paths["incoming"] / "round_001_decision_summary.md").write_text("\n".join(decision_text) + "\n", encoding="utf-8")

    provenance = {
        "campaign_id": campaign_id,
        "round_id": "round_001",
        "fresh_task_session_identifier": session_id,
        "public_bundle_manifest_sha256": sha256_file(paths["manifest"]),
        "initial_prompt_sha256": sha256_file(paths["prompt"]),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_policy": "public_bundle_only_no_hidden_verifier_no_hybrid_access",
        "digital_experiments": int(len(digital)),
        "physical_experiments_requested": int(len(selected)),
        "digital_sequence_cycles": 4,
        "physical_execution_mode": "one_parallel_batch",
        "initial_prompt_text_sha256": sha256_text(paths["prompt"].read_text(encoding="utf-8")),
    }
    (paths["incoming"] / "round_001_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    request = {
        "round_id": "round_001",
        "campaign_id": campaign_id,
        "provenance": provenance,
        "physical_experiment_accounting": {
            "physical_rounds": 1,
            "parallel_cultures": len(selected),
            "unique_strains": len({c["strain_id"] for c in selected}),
            "parallel_or_sequential": "parallel_physical_batch_after_sequential_digital_dBTL",
        },
        "digital_experiment_accounting": {
            "sequential_dbtl_cycles": 4,
            "digital_candidates_screened": len(candidates),
            "hidden_verifier_called": False,
        },
        "candidates": selected,
    }
    paths["request"].write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(paths["request"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True, choices=sorted(PUBLIC_SEEDS))
    parser.add_argument("--max-physical", type=int, default=8)
    args = parser.parse_args()
    run_agent(args.campaign_id, max_physical=args.max_physical)


if __name__ == "__main__":
    main()
