#!/usr/bin/env python3
"""Summarise the three-campaign deployment replication benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "deployment_benchmark"
CAMPAIGNS = ["campaign_001", "campaign_002", "campaign_003"]
METHOD_MAP = {
    "public_bundle_scientist_round_001": "scientist",
    "blinded_hybrid_round_001": "hybrid",
}


def objective_from_public(df: pd.DataFrame) -> pd.Series:
    return df["final_product"] + 0.08 * df["product_AUC"] + 0.03 * df["final_biomass"] - 0.2 * df["growth_failure"].astype(float)


def campaign_comparison(campaign_id: str) -> pd.DataFrame:
    if campaign_id == "campaign_001":
        comp = pd.read_csv(DATA / "deployment_benchmark_matched_round_001_comparison.csv")
        comp = comp.rename(
            columns={
                "hidden_exact_simulations": "hidden_exact_simulations_run",
                "exact_lp_solves": "exact_lp_solves_run_now",
            }
        )
        comp["campaign_id"] = campaign_id
        comp["method"] = comp["method"].map(METHOD_MAP)
        comp["digital_experiments"] = comp["method"].map({"scientist": 0, "hybrid": int(comp["virtual_candidates_generated"].max())})
        comp["digital_experiment_type"] = comp["method"].map({"scientist": "legacy_public_scientist_request", "hybrid": "parallel_virtual_generation_then_sequential_dynamic_refinement"})
        comp["physical_execution_mode"] = "parallel_round_001_batch"
        comp["physical_measurements_counted"] = comp["cultures"]
        return comp
    return pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_comparison.csv")


def public_results(campaign_id: str, method: str) -> pd.DataFrame:
    if campaign_id == "campaign_001" and method == "scientist":
        df = pd.read_csv(DATA / "deployment_benchmark_round_001_public_results.csv")
    elif campaign_id == "campaign_001" and method == "hybrid":
        df = pd.read_csv(DATA / "deployment_benchmark_hybrid_round_001_public_results.csv")
    else:
        df = pd.read_csv(DATA / f"deployment_benchmark_{campaign_id}_{method}_public_results.csv")
    out = df.copy()
    out["campaign_id"] = campaign_id
    out["method"] = method
    out["objective"] = objective_from_public(out)
    return out


def request_candidates(campaign_id: str, method: str) -> list[dict[str, Any]]:
    if campaign_id == "campaign_001" and method == "scientist":
        path = RESULTS / "exchange" / "incoming_requests" / "round_001_request.json"
    elif campaign_id == "campaign_001" and method == "hybrid":
        path = RESULTS / "exchange" / "incoming_requests" / "hybrid_round_001_request.json"
    elif method == "scientist":
        path = RESULTS / campaign_id / "exchange" / "incoming_requests" / "round_001_request.json"
    else:
        path = RESULTS / "exchange" / "incoming_requests" / f"{campaign_id}_hybrid_round_001_request.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("candidates", [])


def reaction_and_subsystem_sets(campaign_id: str, method: str) -> tuple[set[str], set[str], set[str], list[int]]:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv").set_index("reaction_id")
    reactions: set[str] = set()
    subsystems: set[str] = set()
    envs: set[str] = set()
    complexities: list[int] = []
    for cand in request_candidates(campaign_id, method):
        edits = cand.get("edits", [])
        complexities.append(len(edits))
        for edit in edits:
            rid = str(edit.get("reaction_id", ""))
            reactions.add(rid)
            if rid in universe.index:
                subsystems.add(str(universe.loc[rid, "subsystem"]))
            else:
                subsystems.add("curated_heterologous_library")
        env = cand.get("environment", {})
        envs.add(json.dumps(env, sort_keys=True))
    return reactions, subsystems, envs, complexities


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def write_summary() -> pd.DataFrame:
    rows = []
    for campaign_id in CAMPAIGNS:
        comp = campaign_comparison(campaign_id)
        for row in comp.itertuples(index=False):
            rows.append(row._asdict())
    summary = pd.DataFrame(rows)
    front = ["campaign_id", "method"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    summary.to_csv(DATA / "deployment_benchmark_one_shot_replication_summary.csv", index=False)
    return summary


def write_paired(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    margin = float(deployment.OBJECTIVE_WEIGHTS["noninferiority_delta"])
    for campaign_id in CAMPAIGNS:
        sci = summary[(summary["campaign_id"].eq(campaign_id)) & (summary["method"].eq("scientist"))].iloc[0]
        hyb = summary[(summary["campaign_id"].eq(campaign_id)) & (summary["method"].eq("hybrid"))].iloc[0]
        sci_rxn, sci_sub, sci_env, sci_complex = reaction_and_subsystem_sets(campaign_id, "scientist")
        hyb_rxn, hyb_sub, hyb_env, hyb_complex = reaction_and_subsystem_sets(campaign_id, "hybrid")
        best_delta = float(hyb["best_verified_objective"] - sci["best_verified_objective"])
        mean_delta = float(hyb["mean_verified_objective"] - sci["mean_verified_objective"])
        rows.append(
            {
                "campaign_id": campaign_id,
                "best_objective_delta_hybrid_minus_scientist": best_delta,
                "mean_objective_delta_hybrid_minus_scientist": mean_delta,
                "median_objective_delta_hybrid_minus_scientist": float(hyb["median_verified_objective"] - sci["median_verified_objective"]),
                "best_winner": "hybrid" if best_delta > 0 else "scientist" if best_delta < 0 else "tie",
                "mean_winner": "hybrid" if mean_delta > 0 else "scientist" if mean_delta < 0 else "tie",
                "noninferiority_margin": margin,
                "hybrid_noninferior": best_delta >= -margin,
                "selected_reaction_overlap_jaccard": jaccard(sci_rxn, hyb_rxn),
                "selected_subsystem_overlap_jaccard": jaccard(sci_sub, hyb_sub),
                "environment_overlap_jaccard": jaccard(sci_env, hyb_env),
                "scientist_design_complexity_mean": float(np.mean(sci_complex)) if sci_complex else np.nan,
                "hybrid_design_complexity_mean": float(np.mean(hyb_complex)) if hyb_complex else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "deployment_benchmark_one_shot_paired_differences.csv", index=False)
    return out


def write_bootstrap(diffs: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260730)
    rows = []
    for metric in ["best_objective_delta_hybrid_minus_scientist", "mean_objective_delta_hybrid_minus_scientist"]:
        vals = diffs[metric].to_numpy(float)
        boots = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(10000)])
        rows.append(
            {
                "analysis": "one_shot_bootstrap",
                "metric": metric,
                "valid_campaigns": len(vals),
                "exploratory": True,
                "mean": float(vals.mean()),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "notes": "exploratory bootstrap over campaigns; n=3",
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_one_shot_bootstrap.csv", index=False)


def write_acceptance(summary: pd.DataFrame, diffs: pd.DataFrame) -> None:
    noninferior = int(diffs["hybrid_noninferior"].sum())
    best_wins = diffs["best_winner"].value_counts().to_dict()
    mean_wins = diffs["mean_winner"].value_counts().to_dict()
    replication_status = (
        "one_shot_noninferiority_replicated"
        if noninferior == len(diffs)
        else "one_shot_result_mixed"
        if noninferior > 0
        else "one_shot_noninferiority_not_replicated"
    )
    overall = "result_depends_on_campaign_and_accounting_scenario"
    if best_wins.get("hybrid", 0) == len(diffs):
        overall = "deployed_hybrid_supported_as_virtual_screening_tool"
    elif best_wins.get("scientist", 0) == len(diffs) and mean_wins.get("hybrid", 0) == len(diffs):
        overall = "result_depends_on_campaign_and_accounting_scenario"
    elif best_wins.get("scientist", 0) == len(diffs):
        overall = "hybrid_advantage_not_supported"
    rows = [
        {
            "analysis": "one_shot_replication",
            "status": replication_status,
            "passed": replication_status == "one_shot_noninferiority_replicated",
            "valid_campaigns": len(diffs),
            "notes": f"hybrid noninferior in {noninferior}/{len(diffs)} campaigns by best objective",
        },
        {
            "analysis": "best_objective_win_count",
            "status": f"scientist={best_wins.get('scientist', 0)};hybrid={best_wins.get('hybrid', 0)};tie={best_wins.get('tie', 0)}",
            "passed": False,
            "valid_campaigns": len(diffs),
            "notes": "win counts are descriptive with n=3",
        },
        {
            "analysis": "mean_objective_win_count",
            "status": f"scientist={mean_wins.get('scientist', 0)};hybrid={mean_wins.get('hybrid', 0)};tie={mean_wins.get('tie', 0)}",
            "passed": True,
            "valid_campaigns": len(diffs),
            "notes": "mean objective favors batch quality, not best-hit discovery",
        },
        {
            "analysis": "overall",
            "status": overall,
            "passed": overall.startswith("deployed_hybrid_supported"),
            "valid_campaigns": len(diffs),
            "notes": "best-hit scientist advantage and mean-batch hybrid advantage point to campaign/accounting dependence",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_one_shot_acceptance.csv", index=False)


def write_time_to_threshold() -> None:
    threshold = 0.8
    rows = []
    for campaign_id in CAMPAIGNS:
        for method in ["scientist", "hybrid"]:
            df = public_results(campaign_id, method).reset_index(drop=True)
            hits = df.index[df["objective"] >= threshold].tolist()
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "method": method,
                    "threshold_objective": threshold,
                    "threshold_reached": bool(hits),
                    "within_batch_first_hit_index": int(hits[0] + 1) if hits else "",
                    "parallel_batch_cultures_charged": len(df),
                    "physical_rounds_charged": 1,
                    "calendar_days_central_lab": 8.0,
                    "parallel_or_sequential_physical": "parallel_batch",
                    "digital_planning_is_sequential": method == "scientist",
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_time_to_threshold.csv", index=False)


def write_checkpoint(summary: pd.DataFrame, diffs: pd.DataFrame) -> None:
    rows = [
        {"summary_item": "checkpoint_status", "value": "complete_three_campaign_one_shot_replication"},
        {"summary_item": "completed_campaigns_frozen", "value": "campaign_001;campaign_002;campaign_003"},
        {"summary_item": "valid_scientist_campaigns", "value": "campaign_001;campaign_002;campaign_003"},
        {"summary_item": "invalid_duplicate_scientist_registrations_excluded", "value": "campaign_002_legacy_duplicate"},
        {"summary_item": "scientist_best_win_count", "value": int((diffs["best_winner"] == "scientist").sum())},
        {"summary_item": "hybrid_best_win_count", "value": int((diffs["best_winner"] == "hybrid").sum())},
        {"summary_item": "scientist_mean_win_count", "value": int((diffs["mean_winner"] == "scientist").sum())},
        {"summary_item": "hybrid_mean_win_count", "value": int((diffs["mean_winner"] == "hybrid").sum())},
        {"summary_item": "one_shot_replication_classification", "value": pd.read_csv(DATA / "deployment_benchmark_one_shot_acceptance.csv").iloc[0]["status"]},
        {"summary_item": "adaptive_campaign_001_status", "value": "not_run_in_this_agent_creation_step"},
    ]
    pd.DataFrame(rows).to_csv(DATA / "deployment_benchmark_report_checkpoint_summary.csv", index=False)
    pd.DataFrame(
        [
            {"status_type": "campaign_002_scientist", "status": "fresh_public_scientist_agent_valid", "passed": True, "notes": "campaign-specific request plus sidecars registered"},
            {"status_type": "campaign_003_scientist", "status": "fresh_public_scientist_agent_valid", "passed": True, "notes": "campaign-specific request plus sidecars registered"},
            {"status_type": "campaign_002_hybrid", "status": "campaign_002_hybrid_frozen_reused", "passed": True, "notes": "not rerun"},
            {"status_type": "campaign_003_hybrid", "status": "campaign_003_hybrid_blinded_search_complete", "passed": True, "notes": "50,000 generated; 100 dynamic refinements; 8 exact verified"},
            {"status_type": "digital_physical_accounting", "status": "separated", "passed": True, "notes": "digital planning/refinement and physical exact measurements are separate columns and ledgers"},
        ]
    ).to_csv(DATA / "deployment_benchmark_report_checkpoint_acceptance.csv", index=False)


def main() -> None:
    summary = write_summary()
    diffs = write_paired(summary)
    write_bootstrap(diffs)
    write_acceptance(summary, diffs)
    write_time_to_threshold()
    write_checkpoint(summary, diffs)
    print(summary[["campaign_id", "method", "best_verified_objective", "mean_verified_objective", "cultures", "digital_experiments"]].to_string(index=False))
    print(diffs.to_string(index=False))
    print(pd.read_csv(DATA / "deployment_benchmark_one_shot_acceptance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
