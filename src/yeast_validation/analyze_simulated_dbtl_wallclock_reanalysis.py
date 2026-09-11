#!/usr/bin/env python3
"""Corrected wall-clock/depth re-analysis for the simulated DBTL benchmark.

This script is cache-only: it consumes the completed exact outcome table from
``analyze_simulated_dbtl_acceleration.py`` and never launches new Yeast9 runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

OBS_PATH = DATA / "simulated_dbtl_reconstructed_observations.csv"
PROTOCOL_PATH = DATA / "simulated_dbtl_protocol_config.json"
COMPLETION_PATH = DATA / "simulated_dbtl_hpc_completion_audit.csv"

TARGET_TIER = "moderate"
TARGET_COL = f"reaches_{TARGET_TIER}_target"
QUALITY_MARGIN = -0.05
FINAL_BATCH_SIZES = [4, 8, 16, 32]

ADAPTIVE_WORKFLOWS = ["conventional_dbtl", "bayesian_optimization"]
DIGITAL_TWIN_WORKFLOWS = [
    "black_box_digital_twin",
    "modular_hybrid_no_biosensors",
    "biosensor_informed_modular_hybrid__no_reporters",
    "biosensor_informed_modular_hybrid__oxidative_only",
    "biosensor_informed_modular_hybrid__atp_only",
    "biosensor_informed_modular_hybrid__pathway_only",
    "biosensor_informed_modular_hybrid__full_reporters",
    "biosensor_informed_modular_hybrid__noisy_full_reporters",
    "biosensor_informed_modular_hybrid__sparse_full_reporters",
]
CANONICAL_WORKFLOWS = [
    "conventional_dbtl",
    "bayesian_optimization",
    "black_box_digital_twin",
    "modular_hybrid_no_biosensors",
    "biosensor_informed_modular_hybrid__full_reporters",
]
LABELS = {
    "conventional_dbtl": "Conventional DBTL",
    "bayesian_optimization": "Bayesian optimisation",
    "black_box_digital_twin": "Black-box twin",
    "modular_hybrid_no_biosensors": "Modular hybrid, no biosensors",
    "biosensor_informed_modular_hybrid__full_reporters": "Biosensor modular hybrid",
    "biosensor_informed_modular_hybrid__no_reporters": "Biosensor hybrid, no reporters",
    "biosensor_informed_modular_hybrid__oxidative_only": "Biosensor hybrid, oxidative",
    "biosensor_informed_modular_hybrid__atp_only": "Biosensor hybrid, ATP",
    "biosensor_informed_modular_hybrid__pathway_only": "Biosensor hybrid, pathway",
    "biosensor_informed_modular_hybrid__noisy_full_reporters": "Biosensor hybrid, noisy full",
    "biosensor_informed_modular_hybrid__sparse_full_reporters": "Biosensor hybrid, sparse full",
}
COLORS = {
    "conventional_dbtl": "#4c78a8",
    "bayesian_optimization": "#f58518",
    "black_box_digital_twin": "#54a24b",
    "modular_hybrid_no_biosensors": "#b279a2",
    "biosensor_informed_modular_hybrid__full_reporters": "#2f4b7c",
}


def ensure_dirs() -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def stage_days(protocol: dict[str, Any]) -> float:
    assumptions = protocol["calendar_time_assumptions"]
    return float(assumptions["strain_construction_days_per_round"]) + float(assumptions["culture_days_per_round"]) + float(assumptions["assay_days_per_round"])


def planning_days(protocol: dict[str, Any]) -> float:
    return float(protocol["calendar_time_assumptions"]["analysis_planning_days_per_round"])


def workflow_label(workflow_id: str) -> str:
    return LABELS.get(workflow_id, workflow_id)


def end_of_batch_cultures(round_index: int) -> int:
    return {0: 8, 1: 16, 2: 20, 3: 24}[int(round_index)]


def best_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "best_verified_quality": np.nan,
            "best_final_product": np.nan,
            "best_final_biomass": np.nan,
            "best_yield_proxy": np.nan,
            "best_productivity": np.nan,
            "best_edit_count": np.nan,
            "best_culture_task_id": "",
            "best_candidate_id": "",
        }
    idx = frame["utility"].astype(float).idxmax()
    row = frame.loc[idx]
    return {
        "best_verified_quality": float(row["utility"]),
        "best_final_product": float(row["final_product"]),
        "best_final_biomass": float(row["final_biomass"]),
        "best_yield_proxy": float(row["yield_proxy"]),
        "best_productivity": float(row["productivity"]),
        "best_edit_count": int(row["edit_count"]),
        "best_culture_task_id": str(row["culture_task_id"]),
        "best_candidate_id": str(row["candidate_id"]),
    }


def adaptive_campaign_rows(obs: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exp_days = stage_days(protocol)
    plan_days = planning_days(protocol)
    for (world, seed, workflow), group in obs[obs["workflow_id"].isin(ADAPTIVE_WORKFLOWS)].groupby(["world_id", "campaign_seed", "workflow_id"]):
        group = group.sort_values(["round_index", "physical_culture_index"]).copy()
        target_rows = group[group[TARGET_COL].astype(bool)]
        target_attained = not target_rows.empty
        if target_attained:
            target_round = int(target_rows["round_index"].min())
            stages = target_round + 1
            committed_cultures = end_of_batch_cultures(target_round)
            used = group[group["round_index"].astype(int) <= target_round]
        else:
            stages = 4
            committed_cultures = 24
            used = group
        metrics = best_metrics(used)
        rows.append(
            {
                "deployment_protocol": "adaptive_feedback",
                "world_id": world,
                "campaign_seed": int(seed),
                "workflow_id": workflow,
                "workflow_label": workflow_label(workflow),
                "method_family": workflow,
                "final_verification_batch_size": 0,
                "effective_final_verification_batch_size": 0,
                "cache_hits": int(len(used)),
                "cache_misses": 0,
                "target_attained": bool(target_attained),
                "sequential_biological_stages": int(stages),
                "normalized_culture_time_Tc": float(stages),
                "simulated_calendar_days_excluding_compute": float(stages * exp_days + max(stages - 1, 0) * plan_days),
                "total_physical_cultures": int(committed_cultures),
                "unique_strains_constructed": int(used["strain_id"].nunique()),
                "total_assays": int(committed_cultures),
                "virtual_evaluations": int(used["virtual_evaluations_before_selection"].sum()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def one_shot_campaign_rows(obs: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exp_days = stage_days(protocol)
    for (world, seed, workflow), group in obs[obs["workflow_id"].isin(DIGITAL_TWIN_WORKFLOWS)].groupby(["world_id", "campaign_seed", "workflow_id"]):
        group = group.sort_values("physical_culture_index").copy()
        initial = group[group["round_index"].astype(int).eq(0)].copy()
        post = group[group["round_index"].astype(int).gt(0)].copy().sort_values("physical_culture_index")
        initial_hit = bool(initial[TARGET_COL].astype(bool).any())
        for requested_k in FINAL_BATCH_SIZES:
            effective_k = min(int(requested_k), len(post))
            final = post.head(effective_k)
            selected = pd.concat([initial, final], ignore_index=True)
            final_hit = bool(final[TARGET_COL].astype(bool).any())
            target_attained = initial_hit or final_hit
            stages = 1 if initial_hit else 2
            committed_cultures = 8 if initial_hit else 8 + effective_k
            metrics = best_metrics(selected)
            rows.append(
                {
                    "deployment_protocol": "one_shot_parallel_twin",
                    "world_id": world,
                    "campaign_seed": int(seed),
                    "workflow_id": workflow,
                    "workflow_label": workflow_label(workflow),
                    "method_family": str(group["method_id"].iloc[0]),
                    "final_verification_batch_size": int(requested_k),
                    "effective_final_verification_batch_size": int(effective_k),
                    "cache_hits": int(len(selected)),
                    "cache_misses": 0,
                    "target_attained": bool(target_attained),
                    "sequential_biological_stages": int(stages),
                    "normalized_culture_time_Tc": float(stages),
                    "simulated_calendar_days_excluding_compute": float(stages * exp_days),
                    "total_physical_cultures": int(committed_cultures),
                    "unique_strains_constructed": int(selected["strain_id"].nunique()),
                    "total_assays": int(committed_cultures),
                    "virtual_evaluations": int(final["virtual_evaluations_before_selection"].sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def attach_conventional_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    conv = rows[(rows["workflow_id"].eq("conventional_dbtl")) & (rows["deployment_protocol"].eq("adaptive_feedback"))][
        ["world_id", "campaign_seed", "best_verified_quality", "simulated_calendar_days_excluding_compute", "sequential_biological_stages"]
    ].rename(
        columns={
            "best_verified_quality": "conventional_best_verified_quality",
            "simulated_calendar_days_excluding_compute": "conventional_calendar_days",
            "sequential_biological_stages": "conventional_biological_stages",
        }
    )
    out = rows.merge(conv, on=["world_id", "campaign_seed"], how="left", validate="many_to_one")
    out["quality_delta_vs_conventional"] = out["best_verified_quality"] - out["conventional_best_verified_quality"]
    out["quality_noninferior_vs_conventional"] = out["quality_delta_vs_conventional"] >= QUALITY_MARGIN
    out["wall_clock_speedup_vs_conventional"] = out["conventional_calendar_days"] / out["simulated_calendar_days_excluding_compute"]
    out["biological_stage_speedup_vs_conventional"] = out["conventional_biological_stages"] / out["sequential_biological_stages"]
    out["wall_clock_reduction_percent_vs_conventional"] = 100.0 * (out["conventional_calendar_days"] - out["simulated_calendar_days_excluding_compute"]) / out["conventional_calendar_days"]
    out["biological_stage_reduction_percent_vs_conventional"] = 100.0 * (out["conventional_biological_stages"] - out["sequential_biological_stages"]) / out["conventional_biological_stages"]
    return out


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["deployment_protocol", "workflow_id", "workflow_label", "final_verification_batch_size", "effective_final_verification_batch_size"]
    return (
        rows.groupby(keys, dropna=False)
        .agg(
            n_campaigns=("campaign_seed", "size"),
            target_attainment=("target_attained", "mean"),
            mean_total_physical_cultures=("total_physical_cultures", "mean"),
            median_total_physical_cultures=("total_physical_cultures", "median"),
            mean_unique_strains_constructed=("unique_strains_constructed", "mean"),
            mean_total_assays=("total_assays", "mean"),
            mean_sequential_biological_stages=("sequential_biological_stages", "mean"),
            median_sequential_biological_stages=("sequential_biological_stages", "median"),
            mean_normalized_culture_time_Tc=("normalized_culture_time_Tc", "mean"),
            mean_simulated_calendar_days_excluding_compute=("simulated_calendar_days_excluding_compute", "mean"),
            mean_best_verified_quality=("best_verified_quality", "mean"),
            mean_best_final_product=("best_final_product", "mean"),
            mean_best_final_biomass=("best_final_biomass", "mean"),
            mean_best_yield_proxy=("best_yield_proxy", "mean"),
            mean_best_productivity=("best_productivity", "mean"),
            mean_best_edit_count=("best_edit_count", "mean"),
            mean_virtual_evaluations=("virtual_evaluations", "mean"),
            cache_misses=("cache_misses", "sum"),
            mean_wall_clock_speedup_vs_conventional=("wall_clock_speedup_vs_conventional", "mean"),
            mean_biological_stage_speedup_vs_conventional=("biological_stage_speedup_vs_conventional", "mean"),
            mean_wall_clock_reduction_percent_vs_conventional=("wall_clock_reduction_percent_vs_conventional", "mean"),
            mean_biological_stage_reduction_percent_vs_conventional=("biological_stage_reduction_percent_vs_conventional", "mean"),
            mean_quality_delta_vs_conventional=("quality_delta_vs_conventional", "mean"),
            quality_noninferiority_rate=("quality_noninferior_vs_conventional", "mean"),
        )
        .reset_index()
    )


def world_summary(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["world_id", "deployment_protocol", "workflow_id", "workflow_label", "final_verification_batch_size", "effective_final_verification_batch_size"]
    return (
        rows.groupby(keys, dropna=False)
        .agg(
            n_campaigns=("campaign_seed", "size"),
            target_attainment=("target_attained", "mean"),
            mean_total_physical_cultures=("total_physical_cultures", "mean"),
            mean_sequential_biological_stages=("sequential_biological_stages", "mean"),
            mean_simulated_calendar_days_excluding_compute=("simulated_calendar_days_excluding_compute", "mean"),
            mean_best_verified_quality=("best_verified_quality", "mean"),
            mean_wall_clock_speedup_vs_conventional=("wall_clock_speedup_vs_conventional", "mean"),
            mean_biological_stage_speedup_vs_conventional=("biological_stage_speedup_vs_conventional", "mean"),
            quality_noninferiority_rate=("quality_noninferior_vs_conventional", "mean"),
        )
        .reset_index()
    )


def final_table(summary: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for workflow in CANONICAL_WORKFLOWS:
        if workflow in ADAPTIVE_WORKFLOWS:
            sub = summary[(summary["workflow_id"].eq(workflow)) & (summary["deployment_protocol"].eq("adaptive_feedback"))]
        else:
            sub = summary[
                (summary["workflow_id"].eq(workflow))
                & (summary["deployment_protocol"].eq("one_shot_parallel_twin"))
                & (summary["final_verification_batch_size"].eq(16))
            ]
        if not sub.empty:
            chosen.append(sub.iloc[0])
    out = pd.DataFrame(chosen).copy()
    out["quality_noninferior"] = out["mean_quality_delta_vs_conventional"] >= QUALITY_MARGIN
    return out[
        [
            "workflow_id",
            "workflow_label",
            "target_attainment",
            "mean_total_physical_cultures",
            "mean_sequential_biological_stages",
            "mean_normalized_culture_time_Tc",
            "mean_simulated_calendar_days_excluding_compute",
            "mean_best_verified_quality",
            "quality_noninferior",
            "quality_noninferiority_rate",
            "mean_virtual_evaluations",
            "mean_wall_clock_speedup_vs_conventional",
            "mean_biological_stage_speedup_vs_conventional",
            "mean_wall_clock_reduction_percent_vs_conventional",
        ]
    ]


def stage_curve(rows: pd.DataFrame, k: int = 16) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    canonical = rows[rows["workflow_id"].isin(CANONICAL_WORKFLOWS)].copy()
    for workflow in CANONICAL_WORKFLOWS:
        if workflow in ADAPTIVE_WORKFLOWS:
            sub = canonical[(canonical["workflow_id"].eq(workflow)) & (canonical["deployment_protocol"].eq("adaptive_feedback"))]
            for stage in [1, 2, 3, 4]:
                records.append(
                    {
                        "workflow_id": workflow,
                        "workflow_label": workflow_label(workflow),
                        "sequential_biological_stage": stage,
                        "target_attainment_probability": float(((sub["target_attained"]) & (sub["sequential_biological_stages"] <= stage)).mean()),
                    }
                )
        else:
            sub = canonical[
                (canonical["workflow_id"].eq(workflow))
                & (canonical["deployment_protocol"].eq("one_shot_parallel_twin"))
                & (canonical["final_verification_batch_size"].eq(k))
            ]
            # Stage 1 success is equivalent to two-stage row having one stage.
            for stage in [1, 2, 3, 4]:
                p = float(((sub["target_attained"]) & (sub["sequential_biological_stages"] <= min(stage, 2))).mean()) if stage <= 2 else float(sub["target_attained"].mean())
                records.append(
                    {
                        "workflow_id": workflow,
                        "workflow_label": workflow_label(workflow),
                        "sequential_biological_stage": stage,
                        "target_attainment_probability": p,
                    }
                )
    return pd.DataFrame(records)


def culture_curve(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for workflow in CANONICAL_WORKFLOWS:
        if workflow in ADAPTIVE_WORKFLOWS:
            sub = rows[(rows["workflow_id"].eq(workflow)) & (rows["deployment_protocol"].eq("adaptive_feedback"))]
            for cultures in [8, 16, 20, 24]:
                records.append(
                    {
                        "workflow_id": workflow,
                        "workflow_label": workflow_label(workflow),
                        "total_physical_cultures": cultures,
                        "target_attainment_probability": float(((sub["target_attained"]) & (sub["total_physical_cultures"] <= cultures)).mean()),
                    }
                )
        else:
            for k in FINAL_BATCH_SIZES:
                sub = rows[
                    (rows["workflow_id"].eq(workflow))
                    & (rows["deployment_protocol"].eq("one_shot_parallel_twin"))
                    & (rows["final_verification_batch_size"].eq(k))
                ]
                records.append(
                    {
                        "workflow_id": workflow,
                        "workflow_label": workflow_label(workflow),
                        "total_physical_cultures": int(sub["total_physical_cultures"].median()) if not sub.empty else np.nan,
                        "target_attainment_probability": float(sub["target_attained"].mean()) if not sub.empty else np.nan,
                    }
                )
    return pd.DataFrame(records)


def dependency_audit() -> pd.DataFrame:
    rows = [
        {
            "workflow_id": "conventional_dbtl",
            "previous_round_accounting": "round_index 0,1,2,3 counted as up to four physical rounds",
            "implementation_dependency": "manifest generation used placeholder observations, not exact outcomes",
            "deployment_interpretation": "retain as adaptive feedback baseline by protocol semantics",
            "collapsible_without_leakage": False,
            "dependency_graph": "calibration -> outcome analysis -> batch1 -> outcome analysis -> batch2 -> outcome analysis -> batch3",
        },
        {
            "workflow_id": "bayesian_optimization",
            "previous_round_accounting": "round_index 0,1,2,3 counted as up to four physical rounds",
            "implementation_dependency": "manifest generation used placeholder observations, not exact outcomes",
            "deployment_interpretation": "retain as adaptive feedback baseline by protocol semantics",
            "collapsible_without_leakage": False,
            "dependency_graph": "calibration -> BO update -> batch1 -> BO update -> batch2 -> BO update -> batch3",
        },
        {
            "workflow_id": "black_box_digital_twin",
            "previous_round_accounting": "round_index 0,1,2,3 counted as up to four physical rounds",
            "implementation_dependency": "later proposals can be regenerated from calibration/manifest provenance without exact outcomes",
            "deployment_interpretation": "collapse post-calibration candidates into one parallel verification batch",
            "collapsible_without_leakage": True,
            "dependency_graph": "parallel calibration -> train/search -> one parallel verification batch",
        },
        {
            "workflow_id": "modular_hybrid_no_biosensors",
            "previous_round_accounting": "round_index 0,1,2,3 counted as up to four physical rounds",
            "implementation_dependency": "later proposals can be regenerated from calibration/manifest provenance without exact outcomes",
            "deployment_interpretation": "collapse post-calibration candidates into one parallel verification batch",
            "collapsible_without_leakage": True,
            "dependency_graph": "parallel calibration -> mechanistic/twin search -> one parallel verification batch",
        },
        {
            "workflow_id": "biosensor_informed_modular_hybrid",
            "previous_round_accounting": "round_index 0,1,2,3 counted as up to four physical rounds",
            "implementation_dependency": "biosensor-aware branches had no exact reporter values in manifest-generation observations, so existing post-calibration candidates are cache-collapsible",
            "deployment_interpretation": "collapse post-calibration candidates into one parallel verification batch; biosensors affect reliability/quality, not stage depth",
            "collapsible_without_leakage": True,
            "dependency_graph": "parallel calibration + allowed reporters -> twin search -> one parallel verification batch",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return ""
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}g}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x).replace("|", "\\|"))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *body])


def write_figures(rows: pd.DataFrame, summary: pd.DataFrame, world: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    outputs: list[str] = []

    def save(name: str) -> None:
        for ext in ["png", "svg"]:
            path = FIGURES / f"{name}.{ext}"
            plt.savefig(path, bbox_inches="tight", dpi=220)
            outputs.append(str(path.relative_to(ROOT)))
        plt.close()

    curves = stage_curve(rows, k=16)
    plt.figure(figsize=(7.2, 4.8))
    for wid, sub in curves.groupby("workflow_id", sort=False):
        plt.plot(sub["sequential_biological_stage"], sub["target_attainment_probability"], marker="o", label=workflow_label(wid), color=COLORS.get(wid))
    plt.xlabel("Sequential biological culture stages")
    plt.ylabel("Probability target reached")
    plt.ylim(0, 1.02)
    plt.xticks([1, 2, 3, 4])
    plt.legend(frameon=False, fontsize=8)
    plt.title("Target Attainment vs Biological Depth")
    save("simulated_dbtl_wallclock_01_stage_attainment")

    ccurve = culture_curve(rows)
    plt.figure(figsize=(7.2, 4.8))
    for wid, sub in ccurve.groupby("workflow_id", sort=False):
        plt.plot(sub["total_physical_cultures"], sub["target_attainment_probability"], marker="o", label=workflow_label(wid), color=COLORS.get(wid))
    plt.xlabel("Total physical cultures committed")
    plt.ylabel("Probability target reached")
    plt.ylim(0, 1.02)
    plt.legend(frameon=False, fontsize=8)
    plt.title("Target Attainment vs Total Cultures")
    save("simulated_dbtl_wallclock_02_attainment_vs_cultures")

    q = summary[(summary["deployment_protocol"].eq("one_shot_parallel_twin")) & (summary["workflow_id"].isin(["black_box_digital_twin", "modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid__full_reporters"]))]
    plt.figure(figsize=(7.2, 4.8))
    for wid, sub in q.groupby("workflow_id", sort=False):
        plt.plot(sub["final_verification_batch_size"], sub["mean_best_verified_quality"], marker="o", label=workflow_label(wid), color=COLORS.get(wid))
    plt.xlabel("Requested final verification batch size")
    plt.ylabel("Mean best verified quality")
    plt.legend(frameon=False, fontsize=8)
    plt.title("Quality vs One-Shot Verification Capacity")
    save("simulated_dbtl_wallclock_03_quality_vs_batch_size")

    ws = world[(world["workflow_id"].eq("biosensor_informed_modular_hybrid__full_reporters")) & (world["final_verification_batch_size"].eq(16))]
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(ws["world_id"], ws["mean_biological_stage_speedup_vs_conventional"], color="#2f4b7c")
    plt.ylabel("Biological-stage speedup vs conventional")
    plt.xticks(rotation=35, ha="right")
    plt.title("One-Shot Biosensor Hybrid Stage Speedup by World")
    save("simulated_dbtl_wallclock_04_stage_speedup_by_world")

    ft = final_table(summary)
    plt.figure(figsize=(7.2, 4.8))
    plt.barh(ft["workflow_label"], ft["mean_simulated_calendar_days_excluding_compute"], color=[COLORS.get(w, "#777777") for w in ft["workflow_id"]])
    plt.xlabel("Scenario calendar days excluding compute")
    plt.title("Wall-Clock Time to Target/Censoring")
    save("simulated_dbtl_wallclock_05_calendar_time")

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(ft["mean_total_physical_cultures"], ft["mean_simulated_calendar_days_excluding_compute"], s=80, c=[COLORS.get(w, "#777777") for w in ft["workflow_id"]])
    for _, row in ft.iterrows():
        plt.text(row["mean_total_physical_cultures"] + 0.1, row["mean_simulated_calendar_days_excluding_compute"], row["workflow_label"], fontsize=8)
    plt.xlabel("Mean total physical cultures")
    plt.ylabel("Mean calendar days excluding compute")
    plt.title("Cultures vs Wall-Clock Pareto")
    save("simulated_dbtl_wallclock_06_cultures_calendar_pareto")

    bio = summary[(summary["deployment_protocol"].eq("one_shot_parallel_twin")) & (summary["workflow_id"].isin(["modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid__full_reporters"]))]
    x = np.arange(len(FINAL_BATCH_SIZES))
    width = 0.36
    plt.figure(figsize=(7.2, 4.8))
    for offset, wid in [(-width / 2, "modular_hybrid_no_biosensors"), (width / 2, "biosensor_informed_modular_hybrid__full_reporters")]:
        sub = bio[bio["workflow_id"].eq(wid)].set_index("final_verification_batch_size").loc[FINAL_BATCH_SIZES]
        plt.bar(x + offset, sub["target_attainment"], width=width, label=workflow_label(wid), color=COLORS.get(wid))
    plt.xticks(x, FINAL_BATCH_SIZES)
    plt.xlabel("Requested final verification batch size")
    plt.ylabel("Target attainment probability")
    plt.ylim(0, 1.02)
    plt.legend(frameon=False)
    plt.title("Biosensor vs No-Biosensor One-Shot Reliability")
    save("simulated_dbtl_wallclock_07_biosensor_comparison")

    return outputs


def write_report(
    completion: pd.DataFrame,
    dependency: pd.DataFrame,
    final: pd.DataFrame,
    summary: pd.DataFrame,
    world: pd.DataFrame,
    figures: list[str],
) -> None:
    canonical_16 = summary[
        (summary["workflow_id"].isin(CANONICAL_WORKFLOWS))
        & (
            (summary["deployment_protocol"].eq("adaptive_feedback"))
            | ((summary["deployment_protocol"].eq("one_shot_parallel_twin")) & (summary["final_verification_batch_size"].eq(16)))
        )
    ].copy()
    biosensor = summary[
        (summary["workflow_id"].isin(["modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid__full_reporters"]))
        & (summary["deployment_protocol"].eq("one_shot_parallel_twin"))
    ].copy()
    modularity = summary[
        (summary["workflow_id"].isin(["black_box_digital_twin", "biosensor_informed_modular_hybrid__full_reporters"]))
        & (summary["deployment_protocol"].eq("one_shot_parallel_twin"))
    ].copy()
    report = DATA / "simulated_dbtl_wallclock_reanalysis.md"
    lines = [
        "# Corrected Wall-Clock Re-Analysis: Simulated DBTL",
        "",
        "This re-analysis reuses the completed exact Yeast9 cache. No new exact simulations were launched.",
        "",
        "## Completion Gate",
        "",
        markdown_table(completion),
        "",
        "## Dependency Audit",
        "",
        markdown_table(dependency),
        "",
        "Implementation note: the definitive manifest generator appended placeholder observations (`utility=0.0`) during proposal generation. Therefore the existing digital-twin post-calibration candidates can be collapsed without using future exact outcomes. Conventional DBTL and Bayesian optimisation are retained as adaptive baselines by deployment semantics.",
        "",
        "```mermaid",
        "flowchart LR",
        "  C0[Conventional/BO calibration] --> A1[Outcome-dependent design 1] --> B1[Physical batch 1] --> A2[Outcome-dependent design 2] --> B2[Physical batch 2] --> A3[Outcome-dependent design 3] --> B3[Physical batch 3]",
        "  T0[Twin calibration] --> V[Virtual search] --> TV[One parallel verification batch]",
        "```",
        "",
        "## Final Deliverable Table",
        "",
        markdown_table(final),
        "",
        "## One-Shot Verification Capacity",
        "",
        markdown_table(canonical_16),
        "",
        "## Biosensor Value",
        "",
        markdown_table(biosensor),
        "",
        "## Modularity Value",
        "",
        markdown_table(modularity),
        "",
        "## World-Specific Canonical Top-16 Results",
        "",
        markdown_table(world[(world["workflow_id"].isin(CANONICAL_WORKFLOWS)) & ((world["deployment_protocol"].eq("adaptive_feedback")) | ((world["deployment_protocol"].eq("one_shot_parallel_twin")) & (world["final_verification_batch_size"].eq(16))))]),
        "",
        "## Verdict",
        "",
        "Verdict: **B. Digital twin accelerates wall-clock time but requires more parallel culture capacity.**",
        "",
        "The corrected time-depth analysis supports a wall-clock biological-depth advantage for the cached one-shot digital-twin deployment: digital twins can operate as calibration plus one parallel verification batch. However, the full-reporter biosensor hybrid does not reduce total culture count versus conventional DBTL and is quality-noninferior only by the predeclared mean-utility margin, not superior. Biosensors do not reduce biological stages because both modular variants are two-stage; their value is reliability/quality at a fixed final-batch capacity.",
        "",
        "## Figures",
        "",
        *[f"- `{fig}`" for fig in figures],
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    protocol = load_protocol()
    completion = pd.read_csv(COMPLETION_PATH)
    if not bool(completion.iloc[0]["all_valid_complete"]):
        raise RuntimeError("Exact completion audit has not passed; refusing wall-clock re-analysis.")
    obs = pd.read_csv(OBS_PATH)
    if TARGET_COL not in obs.columns:
        raise KeyError(f"Missing target column {TARGET_COL}; rerun final analysis first.")
    adaptive = adaptive_campaign_rows(obs, protocol)
    one_shot = one_shot_campaign_rows(obs, protocol)
    rows = attach_conventional_comparisons(pd.concat([adaptive, one_shot], ignore_index=True))
    summary = aggregate(rows)
    world = world_summary(rows)
    final = final_table(summary)
    dependency = dependency_audit()
    curves_stage = stage_curve(rows)
    curves_culture = culture_curve(rows)

    rows.to_csv(DATA / "simulated_dbtl_wallclock_campaigns_moderate.csv", index=False)
    summary.to_csv(DATA / "simulated_dbtl_wallclock_summary_moderate.csv", index=False)
    world.to_csv(DATA / "simulated_dbtl_wallclock_world_summary_moderate.csv", index=False)
    final.to_csv(DATA / "simulated_dbtl_wallclock_final_table_moderate.csv", index=False)
    dependency.to_csv(DATA / "simulated_dbtl_wallclock_dependency_audit.csv", index=False)
    curves_stage.to_csv(DATA / "simulated_dbtl_wallclock_stage_attainment_curves.csv", index=False)
    curves_culture.to_csv(DATA / "simulated_dbtl_wallclock_culture_attainment_curves.csv", index=False)
    figures = write_figures(rows, summary, world)
    write_report(completion, dependency, final, summary, world, figures)
    print(
        json.dumps(
            {
                "status": "simulated_dbtl_wallclock_reanalysis_complete",
                "campaign_rows": int(len(rows)),
                "summary_rows": int(len(summary)),
                "world_rows": int(len(world)),
                "final_table": str(DATA / "simulated_dbtl_wallclock_final_table_moderate.csv"),
                "report": str(DATA / "simulated_dbtl_wallclock_reanalysis.md"),
                "figures": figures,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
