#!/usr/bin/env python3
"""Cache-only follow-up analysis for the final simulated DBTL benchmark.

This script reuses the completed exact Yeast9 hidden-wet-lab outputs. It does
not submit jobs or run new exact simulations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SUMMARY_PATH = DATA / "simulated_dbtl_best_found_followup.md"

OBS_PATH = DATA / "simulated_dbtl_reconstructed_observations.csv"
VALIDITY_PATH = DATA / "simulated_dbtl_exact_task_validity.csv"
ACCESS_PATH = DATA / "simulated_dbtl_method_access_matrix.csv"
DEPENDENCY_PATH = DATA / "simulated_dbtl_wallclock_dependency_audit.csv"
PROTOCOL_PATH = DATA / "simulated_dbtl_protocol_config.json"
REPORTERS_PATH = DATA / "simulated_dbtl_reporter_conditions.csv"

PHYSICAL_BUDGETS = [8, 12, 16, 20, 24]
STAGE_BUDGETS = [1, 2, 3, 4]
TIME_BUDGETS = [9.0, 18.0, 19.0, 29.0, 39.0]
CANONICAL_WORKFLOWS = [
    "conventional_dbtl",
    "bayesian_optimization",
    "black_box_digital_twin",
    "modular_hybrid_no_biosensors",
    "biosensor_informed_modular_hybrid__full_reporters",
]
DIGITAL_WORKFLOW_PREFIXES = (
    "black_box_digital_twin",
    "modular_hybrid_no_biosensors",
    "biosensor_informed_modular_hybrid",
)
ADAPTIVE_WORKFLOWS = {"conventional_dbtl", "bayesian_optimization"}
TIE_TOL = 1e-9
BOOTSTRAP_SEED = 260812
BOOTSTRAP_N = 2000

COLORS = {
    "conventional_dbtl": "#4c78a8",
    "bayesian_optimization": "#f58518",
    "black_box_digital_twin": "#54a24b",
    "modular_hybrid_no_biosensors": "#b279a2",
    "biosensor_informed_modular_hybrid__full_reporters": "#2f4b7c",
    "biosensor_informed_modular_hybrid__no_reporters": "#9c755f",
    "biosensor_informed_modular_hybrid__oxidative_only": "#72b7b2",
    "biosensor_informed_modular_hybrid__atp_only": "#eeca3b",
    "biosensor_informed_modular_hybrid__pathway_only": "#ff9da6",
    "biosensor_informed_modular_hybrid__noisy_full_reporters": "#bab0ac",
    "biosensor_informed_modular_hybrid__sparse_full_reporters": "#59a14f",
}


def ensure_dirs() -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_label_map(obs: pd.DataFrame) -> dict[str, str]:
    labels = (
        obs[["workflow_id", "workflow_label"]]
        .drop_duplicates()
        .sort_values("workflow_id")
        .set_index("workflow_id")["workflow_label"]
        .to_dict()
    )
    labels["biosensor_informed_modular_hybrid__full_reporters"] = "Biosensor modular hybrid"
    labels["black_box_digital_twin"] = "Black-box twin"
    return labels


def culture_days(protocol: dict[str, Any]) -> float:
    a = protocol["calendar_time_assumptions"]
    return float(a["strain_construction_days_per_round"]) + float(a["culture_days_per_round"]) + float(a["assay_days_per_round"])


def planning_days(protocol: dict[str, Any]) -> float:
    return float(protocol["calendar_time_assumptions"]["analysis_planning_days_per_round"])


def available_at_stage(group: pd.DataFrame, workflow_id: str, stage_budget: int) -> pd.DataFrame:
    if workflow_id in ADAPTIVE_WORKFLOWS:
        return group[group["round_index"].astype(int) <= stage_budget - 1]
    if stage_budget <= 1:
        return group[group["round_index"].astype(int).eq(0)]
    return group


def available_at_time(group: pd.DataFrame, workflow_id: str, days: float, protocol: dict[str, Any]) -> pd.DataFrame:
    exp_days = culture_days(protocol)
    plan_days = planning_days(protocol)
    if workflow_id in ADAPTIVE_WORKFLOWS:
        max_stage = int(np.floor((days + plan_days) / (exp_days + plan_days) + 1e-9))
        max_stage = max(0, min(4, max_stage))
        if max_stage == 0:
            return group.iloc[0:0]
        return available_at_stage(group, workflow_id, max_stage)
    if days + 1e-9 < exp_days:
        return group.iloc[0:0]
    if days + 1e-9 < 2 * exp_days:
        return group[group["round_index"].astype(int).eq(0)]
    return group


def best_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    idx = frame["utility"].astype(float).idxmax()
    return frame.loc[idx]


def summarize_available(frame: pd.DataFrame, universe_best: float) -> dict[str, Any]:
    row = best_row(frame)
    if row is None:
        return {
            "best_verified_utility": np.nan,
            "best_final_product": np.nan,
            "best_yield_proxy": np.nan,
            "best_productivity": np.nan,
            "best_final_biomass": np.nan,
            "best_minimum_biomass": np.nan,
            "best_edit_count": np.nan,
            "target_attainment": False,
            "regret_to_frozen_universe_best": np.nan,
            "n_available_cultures": 0,
            "best_culture_task_id": "",
            "best_candidate_id": "",
        }
    utility = float(row["utility"])
    return {
        "best_verified_utility": utility,
        "best_final_product": float(row["final_product"]),
        "best_yield_proxy": float(row["yield_proxy"]),
        "best_productivity": float(row["productivity"]),
        "best_final_biomass": float(row["final_biomass"]),
        "best_minimum_biomass": float(row["minimum_biomass"]),
        "best_edit_count": int(row["edit_count"]),
        "target_attainment": bool(frame["reaches_moderate_target"].astype(bool).any()),
        "regret_to_frozen_universe_best": float(universe_best - utility),
        "n_available_cultures": int(len(frame)),
        "best_culture_task_id": str(row["culture_task_id"]),
        "best_candidate_id": str(row["candidate_id"]),
    }


def fixed_budget_rows(obs: pd.DataFrame, protocol: dict[str, Any], labels: dict[str, str]) -> pd.DataFrame:
    universe = obs.groupby(["world_id", "campaign_seed"], as_index=False).agg(
        frozen_universe_best_utility=("utility", "max")
    )
    universe_map = {
        (r.world_id, int(r.campaign_seed)): float(r.frozen_universe_best_utility)
        for r in universe.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (world, seed, workflow), group in obs.groupby(["world_id", "campaign_seed", "workflow_id"]):
        group = group.sort_values(["physical_culture_index", "round_index"]).copy()
        universe_best = universe_map[(world, int(seed))]
        for budget in PHYSICAL_BUDGETS:
            available = group[group["physical_culture_index"].astype(int) <= budget]
            rows.append(
                {
                    "budget_axis": "physical_cultures",
                    "budget": float(budget),
                    "world_id": world,
                    "campaign_seed": int(seed),
                    "workflow_id": workflow,
                    "workflow_label": labels.get(workflow, workflow),
                    **summarize_available(available, universe_best),
                }
            )
        for budget in STAGE_BUDGETS:
            available = available_at_stage(group, workflow, budget)
            rows.append(
                {
                    "budget_axis": "sequential_biological_stages",
                    "budget": float(budget),
                    "world_id": world,
                    "campaign_seed": int(seed),
                    "workflow_id": workflow,
                    "workflow_label": labels.get(workflow, workflow),
                    **summarize_available(available, universe_best),
                }
            )
        for budget in TIME_BUDGETS:
            available = available_at_time(group, workflow, budget, protocol)
            rows.append(
                {
                    "budget_axis": "scenario_calendar_days",
                    "budget": float(budget),
                    "world_id": world,
                    "campaign_seed": int(seed),
                    "workflow_id": workflow,
                    "workflow_label": labels.get(workflow, workflow),
                    **summarize_available(available, universe_best),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, higher_is_better: bool = True) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + len(values))
    samples = rng.choice(values, size=(BOOTSTRAP_N, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def aggregate_fixed_budget(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["budget_axis", "budget", "workflow_id", "workflow_label"], dropna=False)
    out = grouped.agg(
        n_campaigns=("campaign_seed", "size"),
        mean_best_verified_utility=("best_verified_utility", "mean"),
        median_best_verified_utility=("best_verified_utility", "median"),
        mean_best_final_product=("best_final_product", "mean"),
        mean_best_yield_proxy=("best_yield_proxy", "mean"),
        mean_best_productivity=("best_productivity", "mean"),
        mean_best_final_biomass=("best_final_biomass", "mean"),
        median_best_edit_count=("best_edit_count", "median"),
        target_attainment_probability=("target_attainment", "mean"),
        mean_regret_to_frozen_universe_best=("regret_to_frozen_universe_best", "mean"),
        median_regret_to_frozen_universe_best=("regret_to_frozen_universe_best", "median"),
    ).reset_index()
    ci_rows = []
    for keys, sub in grouped:
        lo, hi = bootstrap_ci(sub["best_verified_utility"].to_numpy(float))
        ci_rows.append((*keys, lo, hi))
    ci = pd.DataFrame(
        ci_rows,
        columns=["budget_axis", "budget", "workflow_id", "workflow_label", "mean_best_utility_ci_low", "mean_best_utility_ci_high"],
    )
    return out.merge(ci, on=["budget_axis", "budget", "workflow_id", "workflow_label"], how="left")


def paired_effects(rows: pd.DataFrame, reference: str = "black_box_digital_twin") -> pd.DataFrame:
    effects: list[dict[str, Any]] = []
    for (axis, budget), sub in rows.groupby(["budget_axis", "budget"]):
        ref = sub[sub["workflow_id"].eq(reference)][["world_id", "campaign_seed", "best_verified_utility"]].rename(
            columns={"best_verified_utility": "reference_best_verified_utility"}
        )
        for workflow, wsub in sub.groupby("workflow_id"):
            if workflow == reference:
                continue
            merged = wsub.merge(ref, on=["world_id", "campaign_seed"], how="inner")
            diffs = (merged["best_verified_utility"] - merged["reference_best_verified_utility"]).to_numpy(float)
            lo, hi = bootstrap_ci(diffs)
            wins = int((diffs > TIE_TOL).sum())
            losses = int((diffs < -TIE_TOL).sum())
            ties = int(len(diffs) - wins - losses)
            effects.append(
                {
                    "budget_axis": axis,
                    "budget": float(budget),
                    "reference_workflow_id": reference,
                    "workflow_id": workflow,
                    "workflow_label": str(wsub["workflow_label"].iloc[0]),
                    "n_pairs": int(len(diffs)),
                    "mean_delta_best_utility_vs_reference": float(np.mean(diffs)) if len(diffs) else np.nan,
                    "median_delta_best_utility_vs_reference": float(np.median(diffs)) if len(diffs) else np.nan,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / len(diffs) if len(diffs) else np.nan,
                }
            )
    return pd.DataFrame(effects)


def winner_table(rows: pd.DataFrame) -> pd.DataFrame:
    winners: list[dict[str, Any]] = []
    for (axis, budget, world, seed), sub in rows.groupby(["budget_axis", "budget", "world_id", "campaign_seed"]):
        max_u = sub["best_verified_utility"].max()
        tied = sub[np.isclose(sub["best_verified_utility"], max_u, atol=TIE_TOL)]
        for r in tied.itertuples(index=False):
            winners.append(
                {
                    "budget_axis": axis,
                    "budget": float(budget),
                    "world_id": world,
                    "campaign_seed": int(seed),
                    "winner_workflow_id": r.workflow_id,
                    "winner_workflow_label": r.workflow_label,
                    "winner_best_verified_utility": float(max_u),
                    "n_tied_winners": int(len(tied)),
                }
            )
    return pd.DataFrame(winners)


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    final_rows = summary[
        ((summary["budget_axis"].eq("physical_cultures")) & (summary["budget"].eq(24)))
        | ((summary["budget_axis"].eq("sequential_biological_stages")) & (summary["budget"].eq(2)))
        | ((summary["budget_axis"].eq("scenario_calendar_days")) & (summary["budget"].eq(18.0)))
    ].copy()
    final_rows["cost_metric"] = final_rows["budget_axis"]
    final_rows["cost_value"] = final_rows["budget"]
    rows = []
    for axis, sub in final_rows.groupby("budget_axis"):
        for i, row in sub.iterrows():
            dominated_by = []
            for j, other in sub.iterrows():
                if i == j:
                    continue
                no_worse_cost = float(other["budget"]) <= float(row["budget"])
                no_worse_quality = float(other["mean_best_verified_utility"]) >= float(row["mean_best_verified_utility"]) - TIE_TOL
                strictly_better = (
                    float(other["budget"]) < float(row["budget"]) - TIE_TOL
                    or float(other["mean_best_verified_utility"]) > float(row["mean_best_verified_utility"]) + TIE_TOL
                )
                if no_worse_cost and no_worse_quality and strictly_better:
                    dominated_by.append(str(other["workflow_id"]))
            rows.append(
                {
                    "budget_axis": axis,
                    "workflow_id": row["workflow_id"],
                    "workflow_label": row["workflow_label"],
                    "budget": float(row["budget"]),
                    "mean_best_verified_utility": float(row["mean_best_verified_utility"]),
                    "mean_regret_to_frozen_universe_best": float(row["mean_regret_to_frozen_universe_best"]),
                    "pareto_status": "dominated" if dominated_by else "pareto_optimal_or_tied",
                    "dominated_by": ";".join(dominated_by),
                }
            )
    return pd.DataFrame(rows)


def blackbox_hybrid_analysis(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = rows[
        (rows["budget_axis"].eq("physical_cultures"))
        & (rows["budget"].eq(24))
        & rows["workflow_id"].isin(
            [
                "black_box_digital_twin",
                "modular_hybrid_no_biosensors",
                "biosensor_informed_modular_hybrid__full_reporters",
            ]
        )
    ].copy()
    world_rows = []
    for world, sub in main.groupby("world_id"):
        pivot = sub.pivot_table(index="campaign_seed", columns="workflow_id", values="best_verified_utility", aggfunc="first")
        for workflow in ["modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid__full_reporters"]:
            diffs = (pivot[workflow] - pivot["black_box_digital_twin"]).dropna().to_numpy(float)
            lo, hi = bootstrap_ci(diffs)
            world_rows.append(
                {
                    "world_id": world,
                    "comparison": f"{workflow}_minus_black_box_digital_twin",
                    "n_pairs": int(len(diffs)),
                    "mean_delta_best_utility": float(np.mean(diffs)),
                    "median_delta_best_utility": float(np.median(diffs)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "wins": int((diffs > TIE_TOL).sum()),
                    "ties": int(np.isclose(diffs, 0.0, atol=TIE_TOL).sum()),
                    "losses": int((diffs < -TIE_TOL).sum()),
                }
            )
    corr_rows = []
    pivot_all = main.pivot_table(index=["world_id", "campaign_seed"], columns="workflow_id", values="best_verified_utility", aggfunc="first")
    for workflow in ["modular_hybrid_no_biosensors", "biosensor_informed_modular_hybrid__full_reporters"]:
        valid = pivot_all[["black_box_digital_twin", workflow]].dropna()
        pearson = float(valid.corr(method="pearson").iloc[0, 1])
        spearman = float(valid.corr(method="spearman").iloc[0, 1])
        corr_rows.append(
            {
                "comparison": f"{workflow}_vs_black_box_digital_twin",
                "n_pairs": int(len(valid)),
                "pearson_campaign_best_utility": pearson,
                "spearman_campaign_best_utility": spearman,
            }
        )
    return pd.DataFrame(world_rows), pd.DataFrame(corr_rows)


def design_feature_rows(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.drop_duplicates("culture_task_id", keep="first").copy()
    reaction_ids = sorted(
        {
            edit.get("reaction_id", "")
            for text in obs["candidate_json"].dropna().astype(str)
            for edit in json.loads(text).get("edits", [])
            if edit.get("reaction_id")
        }
    )
    mechanism_classes = sorted(
        {
            edit.get("edit_type", "")
            for text in obs["candidate_json"].dropna().astype(str)
            for edit in json.loads(text).get("edits", [])
            if edit.get("edit_type")
        }
    )
    rows = []
    for r in obs.itertuples(index=False):
        candidate = json.loads(r.candidate_json)
        row = {
            "culture_task_id": r.culture_task_id,
            "temperature": float(r.temperature),
            "pH": float(r.pH),
            "DO": float(r.DO),
            "edit_count": int(r.edit_count),
        }
        for reaction_id in reaction_ids:
            row[f"rxn_{reaction_id}_count"] = 0.0
            row[f"rxn_{reaction_id}_capacity_multiplier_mean"] = 1.0
        for mechanism in mechanism_classes:
            row[f"mechanism_{mechanism}_count"] = 0.0
        multipliers: dict[str, list[float]] = defaultdict(list)
        for edit in candidate.get("edits", []):
            rid = edit.get("reaction_id", "")
            mechanism = edit.get("edit_type", "")
            if rid:
                row[f"rxn_{rid}_count"] += 1.0
                if "capacity_multiplier" in edit:
                    multipliers[rid].append(float(edit["capacity_multiplier"]))
                elif "target_upper_bound" in edit and "reference_upper_bound" in edit and float(edit["reference_upper_bound"]) != 0:
                    multipliers[rid].append(float(edit["target_upper_bound"]) / float(edit["reference_upper_bound"]))
            if mechanism:
                row[f"mechanism_{mechanism}_count"] += 1.0
        for rid, values in multipliers.items():
            row[f"rxn_{rid}_capacity_multiplier_mean"] = float(np.mean(values))
        rows.append(row)
    return pd.DataFrame(rows)


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xt = (x_train - mean) / sd
    xv = (x_test - mean) / sd
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])
    penalty = lam * np.eye(xt.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(xt.T @ xt + penalty) @ xt.T @ y_train
    return xv @ beta


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def top_recovery(group: pd.DataFrame, pred_col: str, truth_col: str = "utility", frac: float = 0.2) -> float:
    n = len(group)
    if n == 0:
        return np.nan
    k = max(1, int(np.ceil(frac * n)))
    truth_top = set(group.nlargest(k, truth_col)["culture_task_id"])
    pred_top = set(group.nlargest(k, pred_col)["culture_task_id"])
    return len(truth_top & pred_top) / k


def sensor_proxy_analysis(obs: pd.DataFrame, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This is a bounded proxy: final exact cache stores one reporter summary per
    # culture, not a reporter time series. We test same-culture reporter
    # assimilation using those observed reporter values, under leave-one-world-out
    # splits, and record the limitation in the feasibility audit.
    target = obs[
        obs["workflow_id"].str.startswith("biosensor_informed_modular_hybrid")
        | obs["workflow_id"].isin(["black_box_digital_twin", "modular_hybrid_no_biosensors"])
    ].drop_duplicates("culture_task_id", keep="first").copy()
    overlapping_features = [c for c in feature_df.columns if c != "culture_task_id" and c in target.columns]
    target = target.drop(columns=overlapping_features)
    target = target.merge(feature_df, on="culture_task_id", how="left", validate="one_to_one")
    design_cols = [
        c
        for c in feature_df.columns
        if c != "culture_task_id"
    ]
    reporter_sets = {
        "oxidative_reporter": ["oxidative_reporter"],
        "atp_reporter": ["atp_reporter"],
        "pathway_reporter": ["pathway_reporter"],
        "full_reporters": ["oxidative_reporter", "atp_reporter", "pathway_reporter"],
    }
    metric_rows = []
    prediction_frames = []
    for heldout_world in sorted(target["world_id"].unique()):
        train = target[target["world_id"].ne(heldout_world)].copy()
        test = target[target["world_id"].eq(heldout_world)].copy()
        for reporter_name, reporter_cols in reporter_sets.items():
            usable_test = test[test[reporter_cols].notna().any(axis=1)].copy()
            usable_train = train[train[reporter_cols].notna().any(axis=1)].copy()
            if usable_test.empty or len(usable_train) < 10:
                continue
            for cols, model_name in [
                (design_cols, "nominal_design_only"),
                (design_cols + reporter_cols, f"same_culture_{reporter_name}"),
            ]:
                train_model = usable_train.copy()
                test_model = usable_test.copy()
                for col in cols:
                    if col in reporter_cols:
                        fill = float(train_model[col].mean(skipna=True))
                        train_model[col] = train_model[col].fillna(fill)
                        test_model[col] = test_model[col].fillna(fill)
                x_train = train_model[cols].to_numpy(float)
                y_train = train_model["utility"].to_numpy(float)
                x_test = test_model[cols].to_numpy(float)
                pred = ridge_fit_predict(x_train, y_train, x_test)
                pred_col = "predicted_utility"
                out = test_model[
                    [
                        "culture_task_id",
                        "world_id",
                        "campaign_seed",
                        "workflow_id",
                        "reporter_condition",
                        "utility",
                        "final_product",
                        "final_biomass",
                        "yield_proxy",
                        "productivity",
                    ]
                ].copy()
                out["heldout_world"] = heldout_world
                out["sensor_model"] = model_name
                out[pred_col] = pred
                prediction_frames.append(out)
                err = pred - out["utility"].to_numpy(float)
                campaign_acc = []
                recovery = []
                rank_corrs = []
                poor_actual = out["utility"] <= out["utility"].quantile(0.2)
                poor_pred = out[pred_col] <= out[pred_col].quantile(0.2)
                for _, g in out.groupby(["campaign_seed", "workflow_id", "reporter_condition"]):
                    if len(g) < 2:
                        continue
                    actual_best = g.loc[g["utility"].idxmax(), "culture_task_id"]
                    pred_best = g.loc[g[pred_col].idxmax(), "culture_task_id"]
                    campaign_acc.append(float(actual_best == pred_best))
                    recovery.append(top_recovery(g, pred_col))
                    rank_corrs.append(spearman_corr(g[pred_col].to_numpy(float), g["utility"].to_numpy(float)))
                metric_rows.append(
                    {
                        "heldout_world": heldout_world,
                        "reporter_test": reporter_name,
                        "sensor_model": model_name,
                        "n_train": int(len(train_model)),
                        "n_test": int(len(test_model)),
                        "utility_rmse": float(np.sqrt(np.mean(err**2))),
                        "utility_mae": float(np.mean(np.abs(err))),
                        "utility_spearman": spearman_corr(out[pred_col].to_numpy(float), out["utility"].to_numpy(float)),
                        "campaign_best_identification_accuracy": float(np.mean(campaign_acc)) if campaign_acc else np.nan,
                        "top_20pct_recovery": float(np.mean(recovery)) if recovery else np.nan,
                        "poor_culture_detection_accuracy": float(np.mean(np.asarray(poor_actual) == np.asarray(poor_pred))),
                        "mean_campaign_rank_spearman": float(np.nanmean(rank_corrs)) if rank_corrs else np.nan,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions = pd.DataFrame()
    if metrics.empty:
        return metrics, pd.DataFrame()
    baseline = metrics[metrics["sensor_model"].eq("nominal_design_only")].copy()
    enriched = metrics[~metrics["sensor_model"].eq("nominal_design_only")].copy()
    compare = enriched.merge(
        baseline[["heldout_world", "reporter_test", "utility_rmse", "utility_spearman", "campaign_best_identification_accuracy", "top_20pct_recovery"]],
        on=["heldout_world", "reporter_test"],
        how="left",
        suffixes=("", "_nominal"),
    )
    compare["delta_utility_rmse_vs_nominal"] = compare["utility_rmse"] - compare["utility_rmse_nominal"]
    compare["delta_utility_spearman_vs_nominal"] = compare["utility_spearman"] - compare["utility_spearman_nominal"]
    compare["delta_best_identification_accuracy_vs_nominal"] = compare["campaign_best_identification_accuracy"] - compare["campaign_best_identification_accuracy_nominal"]
    compare["delta_top20_recovery_vs_nominal"] = compare["top_20pct_recovery"] - compare["top_20pct_recovery_nominal"]
    return metrics, compare


def feasibility_audit(validity: pd.DataFrame) -> pd.DataFrame:
    sample_paths = validity["reporters_path"].head(50).tolist()
    reporter_shapes = []
    reporter_cols = set()
    for path in sample_paths:
        df = pd.read_csv(path)
        reporter_shapes.append(tuple(df.shape))
        reporter_cols.update(df.columns)
    trajectory_cols = set()
    for path in validity["trajectory_path"].head(10):
        trajectory_cols.update(pd.read_csv(path, nrows=1).columns)
    return pd.DataFrame(
        [
            {
                "question": "per_time_reporter_trajectories_saved",
                "answer": False,
                "evidence": f"sampled reporter files had shapes {sorted(set(reporter_shapes))} and columns {sorted(reporter_cols)}",
            },
            {
                "question": "hidden_physiology_trajectories_saved",
                "answer": True,
                "evidence": f"trajectory files include hidden columns {sorted(c for c in trajectory_cols if c.startswith('z_') or c.startswith('E_'))}",
            },
            {
                "question": "exact_requested_early_fraction_test_possible_without_reconstruction",
                "answer": False,
                "evidence": "reporter files contain one culture-level reporter row, not 10/20/30% reporter time points",
            },
            {
                "question": "bounded_same_culture_reporter_proxy_possible",
                "answer": True,
                "evidence": "saved reporter summaries can be used as observed same-culture physiological covariates without exposing hidden latents directly",
            },
        ]
    )


def information_flow_table(access: pd.DataFrame, dependency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dep = dependency.set_index("workflow_id").to_dict("index")
    for r in access.itertuples(index=False):
        rows.append(
            {
                "method_id": r.method_id,
                "reporters_allowed_by_access_matrix": bool(r.physiological_reporters),
                "reporters_used_only_in_training_or_calibration": bool(r.method_id == "biosensor_informed_modular_hybrid"),
                "reporters_observed_during_each_physical_culture": bool(r.method_id == "biosensor_informed_modular_hybrid"),
                "reporters_available_during_virtual_search": False,
                "online_regulatory_state_update_from_reporters": False,
                "later_candidate_selection_used_exact_reporters": False,
                "evidence": dep.get(r.method_id, {}).get(
                    "implementation_dependency",
                    "manifest-generation observations used placeholders, not exact reporter/outcome feedback",
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_best_curves(summary: pd.DataFrame) -> None:
    axes = [
        ("physical_cultures", "Physical cultures", "simulated_dbtl_followup_01_utility_vs_cultures.svg"),
        ("sequential_biological_stages", "Sequential biological stages", "simulated_dbtl_followup_02_utility_vs_stages.svg"),
        ("scenario_calendar_days", "Scenario calendar days", "simulated_dbtl_followup_03_utility_vs_time.svg"),
    ]
    for axis, xlabel, filename in axes:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        sub = summary[summary["budget_axis"].eq(axis) & summary["workflow_id"].isin(CANONICAL_WORKFLOWS)]
        for workflow, g in sub.groupby("workflow_id"):
            g = g.sort_values("budget")
            ax.plot(
                g["budget"],
                g["mean_best_verified_utility"],
                marker="o",
                linewidth=2,
                color=COLORS.get(workflow, "#333333"),
                label=str(g["workflow_label"].iloc[0]),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean best verified utility")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / filename)
        plt.close(fig)


def plot_pareto(pareto: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    names = [
        ("physical_cultures", "Cultures"),
        ("sequential_biological_stages", "Stages"),
        ("scenario_calendar_days", "Days"),
    ]
    for ax, (axis, label) in zip(axes, names, strict=True):
        sub = pareto[pareto["budget_axis"].eq(axis)]
        for row in sub.itertuples(index=False):
            ax.scatter(
                row.budget,
                row.mean_best_verified_utility,
                s=70,
                color=COLORS.get(row.workflow_id, "#555555"),
                marker="o" if row.pareto_status.startswith("pareto") else "x",
            )
            ax.text(row.budget, row.mean_best_verified_utility, str(row.workflow_label).split(",")[0], fontsize=6)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean best verified utility")
    fig.tight_layout()
    fig.savefig(FIGURES / "simulated_dbtl_followup_04_pareto.svg")
    plt.close(fig)


def plot_blackbox_world(world: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot = world.pivot(index="world_id", columns="comparison", values="mean_delta_best_utility")
    pivot.plot(kind="bar", ax=ax, color=["#b279a2", "#2f4b7c"])
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_ylabel("Mean delta utility vs black-box")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "simulated_dbtl_followup_05_blackbox_vs_hybrid_world.svg")
    plt.close(fig)


def plot_sensor_proxy(compare: pd.DataFrame) -> None:
    if compare.empty:
        return
    agg = compare.groupby("reporter_test", as_index=False).agg(
        delta_rmse=("delta_utility_rmse_vs_nominal", "mean"),
        delta_spearman=("delta_utility_spearman_vs_nominal", "mean"),
        delta_top20=("delta_top20_recovery_vs_nominal", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, metric, title in [
        (axes[0], "delta_rmse", "Utility RMSE delta"),
        (axes[1], "delta_spearman", "Spearman delta"),
        (axes[2], "delta_top20", "Top-20% recovery delta"),
    ]:
        colors = ["#2f4b7c" if v >= 0 else "#d62728" for v in agg[metric]]
        if metric == "delta_rmse":
            colors = ["#2f4b7c" if v <= 0 else "#d62728" for v in agg[metric]]
        ax.bar(agg["reporter_test"], agg[metric], color=colors)
        ax.axhline(0, color="#111111", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "simulated_dbtl_followup_07_sensor_proxy.svg")
    plt.close(fig)


def write_sensor_flow_svg() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="980" height="280" viewBox="0 0 980 280">
<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#111827}.box{fill:#f8fafc;stroke:#334155;stroke-width:1.2}.weak{fill:#fff7ed;stroke:#c2410c}.no{fill:#fee2e2;stroke:#b91c1c}.yes{fill:#ecfdf5;stroke:#047857}.arrow{stroke:#334155;stroke-width:1.4;marker-end:url(#arrow)}</style>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#334155"/></marker></defs>
<rect x="25" y="55" width="150" height="80" rx="6" class="yes"/><text x="45" y="88">Initial physical</text><text x="45" y="108">cultures</text><text x="45" y="128">product/biomass</text>
<rect x="215" y="55" width="160" height="80" rx="6" class="weak"/><text x="235" y="88">Reporter labels</text><text x="235" y="108">allowed for</text><text x="235" y="128">biosensor method</text>
<rect x="420" y="55" width="150" height="80" rx="6" class="box"/><text x="444" y="88">Train/calibrate</text><text x="444" y="108">twin/search</text><text x="444" y="128">models</text>
<rect x="625" y="55" width="145" height="80" rx="6" class="no"/><text x="645" y="86">No exact live</text><text x="645" y="106">reporter feedback</text><text x="645" y="126">in manifest loop</text>
<rect x="815" y="55" width="135" height="80" rx="6" class="box"/><text x="835" y="88">Candidate</text><text x="835" y="108">selection and</text><text x="835" y="128">verification</text>
<line x1="175" y1="95" x2="215" y2="95" class="arrow"/><line x1="375" y1="95" x2="420" y2="95" class="arrow"/><line x1="570" y1="95" x2="625" y2="95" class="arrow"/><line x1="770" y1="95" x2="815" y2="95" class="arrow"/>
<text x="40" y="190">Conclusion: final benchmark tested reporter-enabled training/calibration and reporter-condition branches, not online state assimilation from early same-culture reporter time series.</text>
<text x="40" y="220">Therefore weak final biosensor contribution does not by itself prove live biosensors are biologically useless.</text>
</svg>
"""
    (FIGURES / "simulated_dbtl_followup_06_sensor_information_flow.svg").write_text(svg, encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    use = df[cols].copy()
    if n is not None:
        use = use.head(n)
    for c in use.columns:
        if pd.api.types.is_float_dtype(use[c]):
            use[c] = use[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    use = use.fillna("")
    header = "| " + " | ".join(str(c) for c in use.columns) + " |"
    separator = "| " + " | ".join("---" for _ in use.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in use.to_numpy()]
    return "\n".join([header, separator, *body])


def write_summary(
    fixed_summary: pd.DataFrame,
    winners: pd.DataFrame,
    pareto: pd.DataFrame,
    bb_world: pd.DataFrame,
    bb_corr: pd.DataFrame,
    flow: pd.DataFrame,
    feasibility: pd.DataFrame,
    sensor_compare: pd.DataFrame,
) -> None:
    phys24 = fixed_summary[
        fixed_summary["budget_axis"].eq("physical_cultures")
        & fixed_summary["budget"].eq(24)
        & fixed_summary["workflow_id"].isin(CANONICAL_WORKFLOWS)
    ].sort_values("mean_best_verified_utility", ascending=False)
    stage2 = fixed_summary[
        fixed_summary["budget_axis"].eq("sequential_biological_stages")
        & fixed_summary["budget"].eq(2)
        & fixed_summary["workflow_id"].isin(CANONICAL_WORKFLOWS)
    ].sort_values("mean_best_verified_utility", ascending=False)
    time18 = fixed_summary[
        fixed_summary["budget_axis"].eq("scenario_calendar_days")
        & fixed_summary["budget"].eq(18.0)
        & fixed_summary["workflow_id"].isin(CANONICAL_WORKFLOWS)
    ].sort_values("mean_best_verified_utility", ascending=False)
    win24 = winners[
        winners["budget_axis"].eq("physical_cultures")
        & winners["budget"].eq(24)
    ]["winner_workflow_id"].value_counts()
    sensor_agg = (
        sensor_compare.groupby("reporter_test", as_index=False)
        .agg(
            mean_delta_rmse=("delta_utility_rmse_vs_nominal", "mean"),
            mean_delta_spearman=("delta_utility_spearman_vs_nominal", "mean"),
            mean_delta_top20=("delta_top20_recovery_vs_nominal", "mean"),
        )
        .sort_values("reporter_test")
        if not sensor_compare.empty
        else pd.DataFrame(columns=["reporter_test", "mean_delta_rmse", "mean_delta_spearman", "mean_delta_top20"])
    )
    text = f"""# Simulated DBTL Best-Found Follow-Up Analysis

Status: cache-only analysis. No new exact Yeast9 simulations were launched.

Inputs reused:

- `data/simulated_dbtl_reconstructed_observations.csv`
- `data/simulated_dbtl_exact_task_validity.csv`
- `data/simulated_dbtl_protocol_config.json`
- `data/simulated_dbtl_method_access_matrix.csv`
- `data/simulated_dbtl_wallclock_dependency_audit.csv`

## Main Result

At the full matched physical-culture budget of 24 cultures, the highest mean
best verified utility across all workflow variants is the ATP-only biosensor
hybrid (`1.8861`). Among the five canonical headline workflows, the
full-reporter biosensor modular hybrid is highest (`1.8842`), but the margin
over conventional DBTL (`1.8808`) and the black-box twin (`1.8641`) is small.
At matched sequential-stage or scenario-time budgets, the digital-twin
workflows dominate conventional/BO because they can collapse post-calibration
verification into one parallel stage. This strengthens the interpretation that
the robust advantage is workflow depth reduction, not a clean mechanistic or
biosensor-specific performance win.

This fixed-budget endpoint deliberately spends the whole budget. It is
therefore complementary to, and numerically different from, the earlier
time-to-target analysis that stops a campaign when the target is reached.

## Fixed Physical-Culture Budget, 24 Cultures

{markdown_table(phys24, ["workflow_label", "mean_best_verified_utility", "median_best_verified_utility", "mean_best_final_product", "target_attainment_probability", "mean_regret_to_frozen_universe_best"])}

Winner counts across world/campaign pairs at 24 cultures, including ties and
all reporter-condition variants:

{markdown_table(win24.rename_axis("winner_workflow_id").reset_index(name="winner_count"), ["winner_workflow_id", "winner_count"])}

## Fixed Sequential-Stage Budget, 2 Stages

{markdown_table(stage2, ["workflow_label", "mean_best_verified_utility", "median_best_verified_utility", "mean_best_final_product", "target_attainment_probability", "mean_regret_to_frozen_universe_best"])}

## Fixed Scenario-Time Budget, 18 Days

{markdown_table(time18, ["workflow_label", "mean_best_verified_utility", "median_best_verified_utility", "mean_best_final_product", "target_attainment_probability", "mean_regret_to_frozen_universe_best"])}

## Black-Box Versus Modular Hybrid

{markdown_table(bb_world, ["world_id", "comparison", "mean_delta_best_utility", "bootstrap_ci_low", "bootstrap_ci_high", "wins", "ties", "losses"])}

Campaign-level ranking correlations:

{markdown_table(bb_corr, ["comparison", "n_pairs", "pearson_campaign_best_utility", "spearman_campaign_best_utility"])}

Interpretation: black-box and modular/hybrid workflows are not decisively
separated in the completed exact benchmark. Any modular advantage is
world-specific and small enough that interpretability/auditability is currently
better supported than a broad performance claim.

## Pareto Interpretation

{markdown_table(pareto[pareto["workflow_id"].isin(CANONICAL_WORKFLOWS)], ["budget_axis", "workflow_label", "budget", "mean_best_verified_utility", "pareto_status", "dominated_by"])}

## Biosensor Information Regime

{markdown_table(flow, ["method_id", "reporters_allowed_by_access_matrix", "reporters_available_during_virtual_search", "online_regulatory_state_update_from_reporters", "later_candidate_selection_used_exact_reporters"])}

The final benchmark did not test live online state assimilation. The manifest
generation loop used placeholder observations, and the wall-clock dependency
audit explicitly notes that biosensor-aware branches did not have exact
reporter values in manifest generation.

## Live Sensor Assimilation Feasibility

{markdown_table(feasibility, ["question", "answer", "evidence"])}

Because per-time reporter trajectories are absent, the exact requested 10%,
20%, and 30% early-reporter test cannot be performed literally from saved
reporter files. The script therefore runs only a bounded same-culture reporter
proxy using saved culture-level reporter summaries.

## Same-Culture Reporter Proxy

{markdown_table(sensor_agg, ["reporter_test", "mean_delta_rmse", "mean_delta_spearman", "mean_delta_top20"])}

Negative RMSE deltas are better; positive Spearman/top-20 deltas are better.
These proxy results should not be promoted as a full live-biosensor result
because they do not use true early reporter time series.

## Answer To The Updated Scientific Questions

1. Equal physical-culture budget: at 24 cultures the full-reporter biosensor
   modular hybrid has the highest mean best verified utility among canonical
   workflows, but by a small margin over the black-box twin.
2. Equal biological-stage budget: at two stages, the one-shot digital-twin
   workflows outperform adaptive conventional/BO because they have already
   completed calibration plus parallel verification.
3. Equal scenario-time budget: at 18 days, the same digital-twin advantage
   appears; conventional and BO have only completed their first stage by then.
4. Hybrid versus black-box: no broad decisive hybrid performance win is
   demonstrated. Modularity is currently better supported as an interpretability
   and auditability feature.
5. The black-box/hybrid tie persists across harder worlds enough that it should
   not be dismissed as only a baseline-world artifact.
6. Biosensors were previously tested mainly as allowed training/calibration
   information and reporter-condition branches, not as live deployment-time
   measurements.
7. The saved exact cache does not contain early reporter time series, so the
   decisive early live-biosensor assimilation test remains missing.
8. Saved culture-level reporter summaries support only a proxy test of
   same-culture reporter information.
9. Live biosensors may still create an additional DBTL time-saving mechanism,
   but the current exact campaign does not yet demonstrate it.
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    obs = pd.read_csv(OBS_PATH)
    validity = pd.read_csv(VALIDITY_PATH)
    access = pd.read_csv(ACCESS_PATH)
    dependency = pd.read_csv(DEPENDENCY_PATH)
    protocol = read_json(PROTOCOL_PATH)
    labels = workflow_label_map(obs)

    fixed = fixed_budget_rows(obs, protocol, labels)
    summary = aggregate_fixed_budget(fixed)
    effects = paired_effects(fixed)
    winners = winner_table(fixed)
    pareto = pareto_table(summary)
    bb_world, bb_corr = blackbox_hybrid_analysis(fixed)
    feature_df = design_feature_rows(obs)
    sensor_metrics, sensor_compare = sensor_proxy_analysis(obs, feature_df)
    feasibility = feasibility_audit(validity)
    flow = information_flow_table(access, dependency)

    fixed.to_csv(DATA / "simulated_dbtl_fixed_budget_best_found.csv", index=False)
    summary.to_csv(DATA / "simulated_dbtl_fixed_budget_best_found_summary.csv", index=False)
    effects.to_csv(DATA / "simulated_dbtl_fixed_budget_paired_effects_vs_blackbox.csv", index=False)
    winners.to_csv(DATA / "simulated_dbtl_fixed_budget_winners.csv", index=False)
    pareto.to_csv(DATA / "simulated_dbtl_best_found_pareto.csv", index=False)
    bb_world.to_csv(DATA / "simulated_dbtl_blackbox_hybrid_world_comparison.csv", index=False)
    bb_corr.to_csv(DATA / "simulated_dbtl_blackbox_hybrid_ranking_correlation.csv", index=False)
    flow.to_csv(DATA / "simulated_dbtl_sensor_information_flow.csv", index=False)
    feasibility.to_csv(DATA / "simulated_dbtl_live_reporter_feasibility_audit.csv", index=False)
    sensor_metrics.to_csv(DATA / "simulated_dbtl_sensor_assimilation_proxy_metrics.csv", index=False)
    sensor_compare.to_csv(DATA / "simulated_dbtl_sensor_assimilation_proxy_deltas.csv", index=False)

    plot_best_curves(summary)
    plot_pareto(pareto)
    plot_blackbox_world(bb_world)
    write_sensor_flow_svg()
    plot_sensor_proxy(sensor_compare)
    write_summary(summary, winners, pareto, bb_world, bb_corr, flow, feasibility, sensor_compare)

    print("cache_only_followup_complete")
    print(f"observations={len(obs)} exact_valid_tasks={int(validity['valid_exact_task'].sum())}/{len(validity)}")
    print(f"wrote={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
