#!/usr/bin/env python3
"""Execute the frozen final sequential DBTL benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import design_benchmark_exact as exact
import final_dbtl_benchmark_freeze as freeze
import gem_backend as gem
import run_deployment_verifier as verifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "final_dbtl_benchmark"
ROLLOUTS = RESULTS / "exact_rollouts"
REPORT = DATA / "final_dbtl_final_report.md"
SUMMARY = DATA / "final_dbtl_method_summary.csv"
OBSERVATIONS = DATA / "final_dbtl_observations.csv"
ROUNDS = DATA / "final_dbtl_round_summary.csv"
BEST = DATA / "final_dbtl_best_so_far.csv"
STATS = DATA / "final_dbtl_statistical_analysis.csv"
EXECUTION = DATA / "final_dbtl_execution_manifest.csv"
EXACT_CACHE = DATA / "final_dbtl_exact_cache_index.csv"
FINAL_FIGURES = [
    "best_utility_vs_cultures",
    "best_utility_vs_rounds",
    "cultures_to_target",
    "rounds_to_target",
    "product_comparison",
    "biomass_comparison",
    "yield_comparison",
    "success_rate_comparison",
    "effective_phenotype_diversity",
    "lp_accounting",
    "method_comparison_summary",
    "dbtl_improvement_curves",
]
METHOD_LABELS = {
    "hybrid_digital_twin": "Hybrid digital twin",
    "public_bundle_dbtL_scientist_agent": "Scientist agent",
    "random_valid": "Random",
    "space_filling_maximin": "Space filling",
    "static_gem_public_rank": "Static GEM",
    "black_box_outcome_only_bo": "Black-box BO",
    "direct_trajectory_surrogate": "Direct surrogate",
}


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for path in [DATA, FIGURES, RESULTS, ROLLOUTS]:
        path.mkdir(parents=True, exist_ok=True)


def edit_from_intervention(row: pd.Series) -> dict[str, Any]:
    edit = {
        "reaction_id": str(row["reaction_id"]),
        "edit_type": str(row["edit_type"]),
        "capacity_multiplier": float(row["capacity_multiplier"]),
        "rationale": f"Frozen final DBTL intervention {row['intervention_id']}",
    }
    for key in ["target_lower_bound", "target_upper_bound", "reference_upper_bound", "reference_lower_bound"]:
        if key in row and pd.notna(row[key]):
            edit[key] = float(row[key])
    return edit


def candidate_hash(candidate: dict[str, Any]) -> str:
    return freeze.candidate_spec_hash(candidate)


def canonical_design_key(edits: list[dict[str, Any]], env: dict[str, float]) -> str:
    payload = {
        "edits": sorted(edits, key=lambda e: (str(e["reaction_id"]), str(e["edit_type"]), float(e.get("capacity_multiplier", 1.0)))),
        "environment": {k: round(float(v), 6) for k, v in sorted(env.items())},
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def intervention_feature(row: pd.Series) -> np.ndarray:
    mechanism = str(row.get("mechanism_class", ""))
    return np.array(
        [
            float(row.get("capacity_multiplier", 1.0)),
            float(row.get("staged_effect_size", 0.0) or 0.0),
            float(row.get("dynamic_effect_size", 0.0) or 0.0),
            1.0 if mechanism == "direct_heterologous_pathway_capacity" else 0.0,
            1.0 if mechanism == "transport_and_environment_coupling" else 0.0,
            1.0 if mechanism == "precursor_supply_or_competing_sink" else 0.0,
        ],
        dtype=float,
    )


def candidate_features(candidate: dict[str, Any], library: pd.DataFrame) -> np.ndarray:
    lib = library.set_index("intervention_id")
    feats = []
    for edit in candidate["edits"]:
        iid = str(edit.get("intervention_id", ""))
        if iid in lib.index:
            feats.append(intervention_feature(lib.loc[iid]))
        else:
            feats.append(np.zeros(6))
    while len(feats) < 3:
        feats.append(np.zeros(6))
    env = candidate["environment"]
    env_vec = np.array(
        [
            (float(env["temperature"]) - 30.0) / 3.0,
            (float(env["pH"]) - 5.0) / 0.5,
            (float(env["DO"]) - 50.0) / 30.0,
        ],
        dtype=float,
    )
    return np.concatenate([env_vec, *feats[:3]])


def make_candidate(
    method_id: str,
    campaign_id: str,
    proposal_index: int,
    interventions: list[pd.Series],
    env: dict[str, float],
    hypothesis: str,
) -> dict[str, Any]:
    edits = []
    for row in interventions:
        edit = edit_from_intervention(row)
        edit["intervention_id"] = str(row["intervention_id"])
        edits.append(edit)
    return {
        "candidate_id": f"{campaign_id}_{method_id}_candidate_{proposal_index:03d}",
        "strain_id": f"{campaign_id}_{method_id}_strain_{hashlib.sha256(stable_json(edits).encode('utf-8')).hexdigest()[:12]}",
        "edit_tier": 1 if len(edits) == 1 else 2,
        "edits": edits,
        "environment": {k: float(v) for k, v in env.items()},
        "requested_assay_panel": "basic",
        "hypothesis": hypothesis,
        "metadata": {
            "method_id": method_id,
            "campaign_id": campaign_id,
            "frozen_final_benchmark": True,
        },
    }


def environment_grid() -> list[dict[str, float]]:
    return [
        {"temperature": 30.0, "pH": 5.0, "DO": 65.0},
        {"temperature": 30.0, "pH": 5.0, "DO": 80.0},
        {"temperature": 30.0, "pH": 5.0, "DO": 25.0},
        {"temperature": 27.0, "pH": 5.0, "DO": 65.0},
        {"temperature": 33.0, "pH": 5.0, "DO": 65.0},
        {"temperature": 30.0, "pH": 4.5, "DO": 65.0},
        {"temperature": 30.0, "pH": 5.5, "DO": 65.0},
        {"temperature": 33.0, "pH": 4.5, "DO": 25.0},
        {"temperature": 27.0, "pH": 5.5, "DO": 80.0},
    ]


def method_score(method_id: str, row: pd.Series, env: dict[str, float], observations: list[dict[str, Any]], rng: np.random.Generator) -> float:
    staged = float(row.get("staged_effect_size", 0.0) or 0.0)
    dynamic = float(row.get("dynamic_effect_size", 0.0) or 0.0)
    mult = float(row.get("capacity_multiplier", 1.0))
    mechanism = str(row.get("mechanism_class", ""))
    env_fit = 1.0 - 0.08 * abs(float(env["temperature"]) - 30.0) - 0.25 * abs(float(env["pH"]) - 5.0) + 0.003 * float(env["DO"])
    if method_id == "hybrid_digital_twin":
        score = 1.3 * staged + 1.1 * dynamic + 0.25 * env_fit
        if observations:
            mech_obs = [o for o in observations if mechanism in str(o.get("mechanism_classes", ""))]
            if mech_obs:
                score += 0.4 * max(float(o["utility"]) for o in mech_obs)
    elif method_id == "public_bundle_dbtL_scientist_agent":
        score = 0.8 * staged + 0.6 * dynamic + 0.4 * env_fit - 0.08 * abs(mult - 1.2)
        if mechanism == "precursor_supply_or_competing_sink":
            score += 0.15
    elif method_id == "random_valid":
        score = float(rng.normal())
    elif method_id == "space_filling_maximin":
        score = float(rng.normal(scale=0.02))
    elif method_id == "static_gem_public_rank":
        score = staged + 0.2 * env_fit - 0.04 * abs(mult - 1.0)
    elif method_id == "black_box_outcome_only_bo":
        if not observations:
            score = float(rng.normal())
        else:
            same = [o for o in observations if mechanism in str(o.get("mechanism_classes", ""))]
            mean = np.mean([float(o["utility"]) for o in same]) if same else np.mean([float(o["utility"]) for o in observations])
            explore = 1.0 / math.sqrt(1 + len(same))
            score = float(mean + 0.35 * explore + rng.normal(scale=0.01))
    elif method_id == "direct_trajectory_surrogate":
        score = dynamic + 0.35 * staged + 0.08 * env_fit
    else:
        score = staged + dynamic
    return float(score)


def proposal_pool(
    method_id: str,
    campaign_id: str,
    seed: int,
    library: pd.DataFrame,
    observations: list[dict[str, Any]],
    start_index: int,
    pool_size: int = 320,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 1009 * (start_index + 1) + 17 * len(observations))
    envs = environment_grid()
    rows = library.to_dict("records")
    candidates = []
    for i in range(pool_size):
        if method_id in {"hybrid_digital_twin", "public_bundle_dbtL_scientist_agent", "direct_trajectory_surrogate"}:
            edit_count = int(rng.choice([1, 1, 2, 2, 3], p=[0.25, 0.25, 0.25, 0.15, 0.10]))
        elif method_id == "black_box_outcome_only_bo" and observations:
            edit_count = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
        else:
            edit_count = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))
        picks = list(rng.choice(len(rows), size=min(edit_count, len(rows)), replace=False))
        selected = [pd.Series(rows[j]) for j in picks]
        env = dict(envs[int(rng.integers(0, len(envs)))])
        if method_id in {"hybrid_digital_twin", "direct_trajectory_surrogate", "static_gem_public_rank"}:
            # Give deterministic optimisers a mild preference for environments
            # already represented in the frozen diagnostic gate.
            env = dict(envs[i % min(len(envs), 8)])
        cand = make_candidate(method_id, campaign_id, start_index + i + 1, selected, env, f"{method_id} frozen-protocol proposal")
        score = sum(method_score(method_id, row, env, observations, rng) for row in selected) - 0.03 * max(0, len(selected) - 1)
        cand["proposal_score"] = float(score)
        candidates.append(cand)
    if method_id == "space_filling_maximin":
        feats = np.vstack([candidate_features(c, library) for c in candidates])
        selected_idx = [int(np.argmax(np.linalg.norm(feats - feats.mean(axis=0), axis=1)))]
        remaining = set(range(len(candidates))) - set(selected_idx)
        while len(selected_idx) < min(64, len(candidates)):
            idx = max(remaining, key=lambda j: min(float(np.linalg.norm(feats[j] - feats[k])) for k in selected_idx))
            selected_idx.append(idx)
            remaining.remove(idx)
        candidates = [candidates[i] for i in selected_idx]
    else:
        candidates.sort(key=lambda c: float(c["proposal_score"]), reverse=True)
    return candidates


def candidate_metadata(candidate: dict[str, Any], library: pd.DataFrame) -> dict[str, Any]:
    lib = library.set_index("intervention_id")
    mechanisms = []
    risks = []
    for edit in candidate["edits"]:
        iid = str(edit.get("intervention_id", ""))
        if iid in lib.index:
            mechanisms.append(str(lib.loc[iid, "mechanism_class"]))
            risks.append(str(lib.loc[iid, "risk_level"]))
    return {
        "edit_count": len(candidate["edits"]),
        "mechanism_classes": ";".join(sorted(set(mechanisms))),
        "construction_risk": "high" if "high" in risks else "medium" if "medium" in risks else "low",
    }


def load_exact_cache() -> pd.DataFrame:
    if EXACT_CACHE.exists():
        return pd.read_csv(EXACT_CACHE)
    return pd.DataFrame()


def write_cache_row(row: dict[str, Any]) -> None:
    cache = load_exact_cache()
    out = pd.concat([cache, pd.DataFrame([row])], ignore_index=True, sort=False)
    out = out.drop_duplicates("candidate_hash", keep="last")
    out.to_csv(EXACT_CACHE, index=False)


def rollout_paths(candidate_hash_value: str) -> dict[str, Path]:
    prefix = candidate_hash_value[:16]
    return {
        "trajectory": ROLLOUTS / f"{prefix}_trajectory.csv",
        "flux": ROLLOUTS / f"{prefix}_flux.csv",
        "constraints": ROLLOUTS / f"{prefix}_constraints.csv",
        "summary": ROLLOUTS / f"{prefix}_summary.csv",
    }


def run_exact_candidate(candidate: dict[str, Any], library: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    h = candidate["candidate_hash"]
    paths = rollout_paths(h)
    cache = load_exact_cache()
    if not cache.empty and h in set(cache["candidate_hash"].astype(str)) and paths["summary"].exists():
        row = cache[cache["candidate_hash"].astype(str).eq(h)].iloc[-1].to_dict()
        return row, True
    tic = time.perf_counter()
    traj, flux, cons, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime="strong_dynamic_regulation")
    runtime = time.perf_counter() - tic
    traj.to_csv(paths["trajectory"], index=False)
    flux.to_csv(paths["flux"], index=False)
    cons.to_csv(paths["constraints"], index=False)
    summary.to_csv(paths["summary"], index=False)
    s = summary.iloc[0].to_dict()
    meta = candidate_metadata(candidate, library)
    integrated_glucose = 0.0
    if not flux.empty and {"glucose_uptake", "dt"}.issubset(flux.columns):
        # Biomass-scaled integrated uptake proxy matching the objective docs.
        x_by_interval = traj.set_index("time")["X"] if "time" in traj.columns else pd.Series(dtype=float)
        values = []
        for _, frow in flux.iterrows():
            x = float(x_by_interval.get(float(frow["time"]), traj["X"].iloc[0] if "X" in traj else 1.0))
            values.append(max(0.0, -float(frow["glucose_uptake"])) * x * float(frow["dt"]))
        integrated_glucose = float(np.sum(values))
    yield_proxy = float(s["final_product"]) / max(integrated_glucose, 1e-12)
    productivity = float(s["final_product"]) / max(float(traj["time"].iloc[-1]) if "time" in traj else 1.0, 1e-12)
    phenotype = freeze.phenotype_fingerprint({"trajectory": traj, "flux": flux, "constraints": cons}, tolerance=freeze.PHENOTYPE_TOLERANCE)
    effective_model = freeze.effective_model_hash(candidate, str(s["gem_source_checksum"]), cons)
    result = {
        **s,
        **meta,
        "candidate_hash": h,
        "candidate_id": candidate["candidate_id"],
        "strain_id": candidate["strain_id"],
        "candidate_json": stable_json(candidate),
        "integrated_glucose_proxy": integrated_glucose,
        "yield_proxy": yield_proxy,
        "productivity": productivity,
        "utility": freeze.final_utility(
            {
                "final_product": float(s["final_product"]),
                "product_AUC": float(s["product_AUC"]),
                "yield_proxy": yield_proxy,
                "final_biomass": float(s["final_biomass"]),
                "growth_failure": bool(s["severe_growth_collapse"]),
                "edit_count": meta["edit_count"],
                "construction_risk": meta["construction_risk"],
                "infeasible": not bool(s["feasible"]),
            },
            load_json(DATA / "final_dbtl_objective_config.json"),
        ),
        "reaches_target": freeze.reaches_target(
            {
                "final_product": float(s["final_product"]),
                "product_AUC": float(s["product_AUC"]),
                "yield_proxy": yield_proxy,
                "final_biomass": float(s["final_biomass"]),
                "growth_failure": bool(s["severe_growth_collapse"]),
                "edit_count": meta["edit_count"],
                "construction_risk": meta["construction_risk"],
                "infeasible": not bool(s["feasible"]),
            },
            load_json(DATA / "final_dbtl_target_config.json"),
            load_json(DATA / "final_dbtl_objective_config.json"),
        ),
        "effective_model_hash": effective_model,
        "effective_phenotype_fingerprint": phenotype,
        "wall_time_seconds": runtime,
        "trajectory_path": str(paths["trajectory"].relative_to(ROOT)),
        "flux_path": str(paths["flux"].relative_to(ROOT)),
        "constraint_path": str(paths["constraints"].relative_to(ROOT)),
        "summary_path": str(paths["summary"].relative_to(ROOT)),
    }
    write_cache_row(result)
    exact.clear_solver_caches()
    return result, False


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=not exists, index=False)


def already_completed_keys() -> set[tuple[str, str, str, int]]:
    if not OBSERVATIONS.exists():
        return set()
    obs = pd.read_csv(OBSERVATIONS)
    if obs.empty:
        return set()
    return set(zip(obs["campaign_id"].astype(str), obs["method_id"].astype(str), obs["candidate_hash"].astype(str), obs["culture_index"].astype(int)))


def run_benchmark(max_new_exact: int | None = None) -> None:
    ensure_dirs()
    spec = load_json(DATA / "final_dbtl_benchmark_spec.json")
    library = pd.read_csv(DATA / "final_dbtl_actionable_intervention_library.csv")
    objective = load_json(DATA / "final_dbtl_objective_config.json")
    target = load_json(DATA / "final_dbtl_target_config.json")
    completed = already_completed_keys()
    if not OBSERVATIONS.exists():
        for p in [OBSERVATIONS, ROUNDS, BEST, SUMMARY, STATS, EXECUTION]:
            if p.exists():
                p.unlink()
    new_exact = 0
    for campaign_idx, seed in enumerate(spec["campaign_seeds"], start=1):
        campaign_id = f"final_dbtl_campaign_{campaign_idx:03d}"
        for method_id in spec["method_list"]:
            observations = []
            if OBSERVATIONS.exists():
                existing = pd.read_csv(OBSERVATIONS)
                sub = existing[existing["campaign_id"].eq(campaign_id) & existing["method_id"].eq(method_id)]
                if not sub.empty:
                    observations = sub.sort_values("culture_index").to_dict("records")
            seen_designs = {str(o.get("design_key", "")) for o in observations}
            target_hit = any(bool(o.get("reaches_target", False)) for o in observations)
            proposal_index = len(observations)
            virtual_evals = 0
            for round_index, batch_size in enumerate(spec["batch_sizes"], start=1):
                if target_hit:
                    break
                already_in_round = [o for o in observations if int(o["round_index"]) == round_index]
                needed = int(batch_size) - len(already_in_round)
                if needed <= 0:
                    continue
                pool = proposal_pool(method_id, campaign_id, int(seed), library, observations, proposal_index, pool_size=420)
                virtual_evals += len(pool)
                selected: list[dict[str, Any]] = []
                invalid = 0
                duplicate = 0
                for cand in pool:
                    key = canonical_design_key(cand["edits"], cand["environment"])
                    if key in seen_designs:
                        duplicate += 1
                        continue
                    h = candidate_hash(cand)
                    cand["candidate_hash"] = h
                    if any(h == c.get("candidate_hash") for c in selected):
                        duplicate += 1
                        continue
                    selected.append(cand)
                    seen_designs.add(key)
                    if len(selected) == needed:
                        break
                if len(selected) < needed:
                    raise RuntimeError(f"{method_id} {campaign_id} could only propose {len(selected)}/{needed} valid non-duplicates")
                for cand in selected:
                    proposal_index += 1
                    culture_index = len(observations) + 1
                    key_tuple = (campaign_id, method_id, cand["candidate_hash"], culture_index)
                    if key_tuple in completed:
                        continue
                    if max_new_exact is not None and new_exact >= max_new_exact:
                        append_csv(
                            EXECUTION,
                            {
                                "status": "paused_after_max_new_exact",
                                "max_new_exact": max_new_exact,
                                "timestamp": pd.Timestamp.utcnow().isoformat(),
                            },
                        )
                        return
                    print(f"[final-dbtl] {campaign_id} {method_id} round={round_index} culture={culture_index} {cand['candidate_id']}", flush=True)
                    try:
                        result, cache_hit = run_exact_candidate(cand, library)
                    except Exception as exc:
                        result = {
                            **candidate_metadata(cand, library),
                            "candidate_hash": cand["candidate_hash"],
                            "candidate_id": cand["candidate_id"],
                            "strain_id": cand["strain_id"],
                            "candidate_json": stable_json(cand),
                            "feasible": False,
                            "severe_growth_collapse": True,
                            "exact_status": "failed_exact_dynamic_pfba",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "n_actual_lp_solves": 0,
                            "n_infeasible_solves": 0,
                            "n_unbounded_solves": 0,
                            "n_solver_errors": 1,
                            "utility": -math.inf,
                            "final_product": 0.0,
                            "product_AUC": 0.0,
                            "final_biomass": 0.0,
                            "yield_proxy": 0.0,
                            "productivity": 0.0,
                            "reaches_target": False,
                            "effective_model_hash": "",
                            "effective_phenotype_fingerprint": "",
                            "wall_time_seconds": 0.0,
                        }
                        cache_hit = False
                    new_exact += 0 if cache_hit else 1
                    obs_row = {
                        "campaign_id": campaign_id,
                        "campaign_seed": seed,
                        "method_id": method_id,
                        "round_index": round_index,
                        "culture_index": culture_index,
                        "physical_culture_charge": 1,
                        "cache_hit": bool(cache_hit),
                        "virtual_candidate_evaluations_cumulative": virtual_evals,
                        "invalid_proposals_this_round": invalid,
                        "duplicate_proposals_this_round": duplicate,
                        "design_key": canonical_design_key(cand["edits"], cand["environment"]),
                        "objective_hash": objective["objective_hash"],
                        "target_hash": target["target_hash"],
                        "benchmark_spec_hash": spec["benchmark_spec_hash"],
                        **result,
                    }
                    append_csv(OBSERVATIONS, obs_row)
                    observations.append(obs_row)
                    target_hit = bool(obs_row["reaches_target"])
                    if target_hit:
                        break
                write_intermediate_summaries(spec)
    write_all_outputs()


def write_intermediate_summaries(spec: dict[str, Any]) -> None:
    if not OBSERVATIONS.exists():
        return
    obs = pd.read_csv(OBSERVATIONS)
    best_rows = []
    round_rows = []
    for (campaign_id, method_id), group in obs.groupby(["campaign_id", "method_id"]):
        group = group.sort_values("culture_index")
        best = -np.inf
        for _, row in group.iterrows():
            best = max(best, float(row["utility"]))
            best_rows.append(
                {
                    "campaign_id": campaign_id,
                    "method_id": method_id,
                    "culture_index": int(row["culture_index"]),
                    "round_index": int(row["round_index"]),
                    "best_utility_so_far": best,
                    "target_reached_so_far": bool(group[group["culture_index"].le(row["culture_index"])]["reaches_target"].astype(bool).any()),
                }
            )
        for round_idx, rg in group.groupby("round_index"):
            round_rows.append(
                {
                    "campaign_id": campaign_id,
                    "method_id": method_id,
                    "round_index": int(round_idx),
                    "cultures_cumulative": int(rg["culture_index"].max()),
                    "best_utility_round": float(rg["utility"].max()),
                    "best_utility_cumulative": float(group[group["culture_index"].le(rg["culture_index"].max())]["utility"].max()),
                    "target_reached_by_round": bool(group[group["culture_index"].le(rg["culture_index"].max())]["reaches_target"].astype(bool).any()),
                }
            )
    pd.DataFrame(best_rows).to_csv(BEST, index=False)
    pd.DataFrame(round_rows).to_csv(ROUNDS, index=False)


def method_campaign_summary(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (campaign_id, method_id), group in obs.groupby(["campaign_id", "method_id"]):
        group = group.sort_values("culture_index")
        success = group[group["reaches_target"].astype(bool)]
        n_star = int(success["culture_index"].iloc[0]) if not success.empty else np.nan
        round_star = int(success["round_index"].iloc[0]) if not success.empty else np.nan
        best_idx = group["utility"].astype(float).idxmax()
        best = group.loc[best_idx]
        rows.append(
            {
                "campaign_id": campaign_id,
                "method_id": method_id,
                "success": not success.empty,
                "cultures_to_target": n_star,
                "right_censored_cultures": np.nan if not success.empty else ">16",
                "rounds_to_target": round_star,
                "right_censored_rounds": np.nan if not success.empty else ">3",
                "best_utility": float(best["utility"]),
                "best_final_product": float(best["final_product"]),
                "best_product_AUC": float(best["product_AUC"]),
                "best_productivity": float(best["productivity"]),
                "best_yield_proxy": float(best["yield_proxy"]),
                "best_final_biomass": float(best["final_biomass"]),
                "growth_failure_rate": float(group["severe_growth_collapse"].astype(bool).mean()) if "severe_growth_collapse" in group else 0.0,
                "failure_rate": float((group["exact_status"] != "complete_exact_dynamic_pfba").mean()) if "exact_status" in group else 0.0,
                "invalid_proposals": int(group["invalid_proposals_this_round"].fillna(0).sum()),
                "duplicate_proposals": int(group["duplicate_proposals_this_round"].fillna(0).sum()),
                "unique_effective_phenotypes": int(group["effective_phenotype_fingerprint"].nunique()),
                "unique_effective_models": int(group["effective_model_hash"].nunique()),
                "exact_cultures_charged": int(group["physical_culture_charge"].sum()),
                "cache_hits": int(group["cache_hit"].astype(bool).sum()),
                "exact_lp_solves": int(group["n_actual_lp_solves"].fillna(0).sum()),
                "total_compute_time_seconds": float(group["wall_time_seconds"].fillna(0).sum()),
                "virtual_candidate_evaluations": int(group["virtual_candidate_evaluations_cumulative"].max()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, fn=np.mean, n_boot: int = 5000, seed: int = 991) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = [fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def write_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = list(METHOD_LABELS)
    hybrid = summary[summary["method_id"].eq("hybrid_digital_twin")]
    for method in methods:
        sub = summary[summary["method_id"].eq(method)]
        successes = sub["success"].astype(bool)
        n_values = sub["cultures_to_target"].astype(float)
        censored = n_values.fillna(17.0).to_numpy(float)
        ci = bootstrap_ci(censored, np.median)
        rows.append(
            {
                "comparison": f"{method}_overall",
                "method_id": method,
                "success_rate": float(successes.mean()) if len(successes) else np.nan,
                "median_cultures_to_target_censored_at_17": float(np.median(censored)) if len(censored) else np.nan,
                "median_cultures_bootstrap_ci_low": ci[0],
                "median_cultures_bootstrap_ci_high": ci[1],
                "mean_best_utility": float(sub["best_utility"].mean()) if len(sub) else np.nan,
            }
        )
    for method in methods:
        if method == "hybrid_digital_twin":
            continue
        merged = hybrid[["campaign_id", "cultures_to_target", "best_utility", "success"]].merge(
            summary[summary["method_id"].eq(method)][["campaign_id", "cultures_to_target", "best_utility", "success"]],
            on="campaign_id",
            suffixes=("_hybrid", "_other"),
        )
        if merged.empty:
            continue
        h_c = merged["cultures_to_target_hybrid"].fillna(17.0).to_numpy(float)
        o_c = merged["cultures_to_target_other"].fillna(17.0).to_numpy(float)
        diff = h_c - o_c
        ci = bootstrap_ci(diff, np.mean)
        rows.append(
            {
                "comparison": f"hybrid_vs_{method}",
                "method_id": method,
                "paired_mean_cultures_delta_hybrid_minus_other": float(np.mean(diff)),
                "paired_mean_cultures_delta_ci_low": ci[0],
                "paired_mean_cultures_delta_ci_high": ci[1],
                "hybrid_culture_win_count": int((h_c < o_c).sum()),
                "other_culture_win_count": int((h_c > o_c).sum()),
                "tie_count": int((h_c == o_c).sum()),
                "paired_mean_best_utility_delta": float((merged["best_utility_hybrid"] - merged["best_utility_other"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(STATS, index=False)
    return out


def dbtl_auc(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (campaign_id, method_id), group in best.groupby(["campaign_id", "method_id"]):
        g = group.sort_values("culture_index")
        rows.append(
            {
                "campaign_id": campaign_id,
                "method_id": method_id,
                "dbtl_auc": float(np.trapezoid(g["best_utility_so_far"].to_numpy(float), g["culture_index"].to_numpy(float))) if len(g) > 1 else float(g["best_utility_so_far"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_all_outputs() -> None:
    obs = pd.read_csv(OBSERVATIONS)
    write_intermediate_summaries(load_json(DATA / "final_dbtl_benchmark_spec.json"))
    best = pd.read_csv(BEST)
    summary = method_campaign_summary(obs)
    auc = dbtl_auc(best)
    summary = summary.merge(auc, on=["campaign_id", "method_id"], how="left")
    summary.to_csv(SUMMARY, index=False)
    stats = write_statistics(summary)
    make_figures(obs, summary, best)
    write_report(obs, summary, stats)
    pd.DataFrame(
        [
            {
                "status": "complete_final_dbtl_benchmark",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "observations": len(obs),
                "methods": obs["method_id"].nunique(),
                "campaigns": obs["campaign_id"].nunique(),
                "final_campaigns_executed": True,
            }
        ]
    ).to_csv(EXECUTION, index=False)


def make_figures(obs: pd.DataFrame, summary: pd.DataFrame, best: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    colors = plt.get_cmap("tab10")
    methods = list(METHOD_LABELS)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIGURES / f"final_dbtl_{name}.svg")
        plt.savefig(FIGURES / f"final_dbtl_{name}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        sub = best[best["method_id"].eq(method)]
        if sub.empty:
            continue
        curve = sub.groupby("culture_index")["best_utility_so_far"].mean()
        plt.plot(curve.index, curve.values, marker="o", label=METHOD_LABELS[method], color=colors(i))
    plt.xlabel("Exact cultures")
    plt.ylabel("Best utility so far")
    plt.legend(fontsize=7)
    save("best_utility_vs_cultures")

    plt.figure(figsize=(8, 5))
    rounds = pd.read_csv(ROUNDS)
    for i, method in enumerate(methods):
        sub = rounds[rounds["method_id"].eq(method)]
        if sub.empty:
            continue
        curve = sub.groupby("round_index")["best_utility_cumulative"].mean()
        plt.plot(curve.index, curve.values, marker="o", label=METHOD_LABELS[method], color=colors(i))
    plt.xlabel("DBTL round")
    plt.ylabel("Best utility so far")
    plt.legend(fontsize=7)
    save("best_utility_vs_rounds")

    plot_specs = [
        ("cultures_to_target", "cultures_to_target", "Cultures to target (17 = censored)"),
        ("rounds_to_target", "rounds_to_target", "Rounds to target (4 = censored)"),
        ("product_comparison", "best_final_product", "Best final beta-carotene"),
        ("biomass_comparison", "best_final_biomass", "Best final biomass"),
        ("yield_comparison", "best_yield_proxy", "Best yield proxy"),
        ("effective_phenotype_diversity", "unique_effective_phenotypes", "Unique effective phenotypes"),
        ("lp_accounting", "exact_lp_solves", "Exact LP solves"),
        ("method_comparison_summary", "best_utility", "Best utility"),
    ]
    for name, col, ylabel in plot_specs:
        plt.figure(figsize=(8, 5))
        data = []
        labels = []
        for method in methods:
            sub = summary[summary["method_id"].eq(method)]
            if sub.empty:
                continue
            vals = sub[col].copy()
            if col == "cultures_to_target":
                vals = vals.fillna(17)
            if col == "rounds_to_target":
                vals = vals.fillna(4)
            data.append(vals.to_numpy(float))
            labels.append(METHOD_LABELS[method])
        try:
            plt.boxplot(data, tick_labels=labels, showmeans=True)
        except TypeError:
            plt.boxplot(data, labels=labels, showmeans=True)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(ylabel)
        save(name)

    plt.figure(figsize=(8, 5))
    success = summary.groupby("method_id")["success"].mean().reindex(methods)
    plt.bar([METHOD_LABELS[m] for m in methods], success.values)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Success rate")
    save("success_rate_comparison")

    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        sub = best[best["method_id"].eq(method)]
        if sub.empty:
            continue
        parent = float(load_json(DATA / "final_dbtl_objective_config.json")["reference_values"]["final_titer"])
        curve = sub.groupby("culture_index")["best_utility_so_far"].mean()
        plt.plot(curve.index, curve.values - float(curve.iloc[0]), marker="o", label=METHOD_LABELS[method], color=colors(i))
    plt.xlabel("Exact cultures")
    plt.ylabel("Utility improvement from first culture")
    plt.legend(fontsize=7)
    save("dbtl_improvement_curves")


def write_report(obs: pd.DataFrame, summary: pd.DataFrame, stats: pd.DataFrame) -> None:
    spec = load_json(DATA / "final_dbtl_benchmark_spec.json")
    objective = load_json(DATA / "final_dbtl_objective_config.json")
    target = load_json(DATA / "final_dbtl_target_config.json")
    success = summary.groupby("method_id")["success"].mean().sort_values(ascending=False)
    cultures = summary.assign(cultures_to_target_censored=summary["cultures_to_target"].fillna(17)).groupby("method_id")["cultures_to_target_censored"].median().sort_values()
    best_utility = summary.groupby("method_id")["best_utility"].mean().sort_values(ascending=False)
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        clean = frame.copy()
        for col in clean.columns:
            if pd.api.types.is_float_dtype(clean[col]):
                clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
            else:
                clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(clean.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
        rows = ["| " + " | ".join(str(row[col]) for col in clean.columns) + " |" for _, row in clean.iterrows()]
        return "\n".join([header, sep, *rows])

    lines = [
        "# Final Sequential DBTL Benchmark Report",
        "",
        "## Experimental setup",
        "",
        f"Benchmark `{spec['benchmark_id']}` executed the frozen `8+4+4` sequential protocol across `{len(spec['campaign_seeds'])}` campaign seeds and `{len(spec['method_list'])}` methods. Each exact culture used Yeast9 dynamic pFBA through the frozen strong-regulation verifier. Cached rollouts, if any, were still charged as physical cultures.",
        "",
        "## Frozen hashes",
        "",
        f"- Objective hash: `{objective['objective_hash']}`",
        f"- Target hash: `{target['target_hash']}`",
        f"- Benchmark spec hash: `{spec['benchmark_spec_hash']}`",
        f"- Intervention count: `{spec['intervention_count']}`",
        "",
        "## Execution summary",
        "",
        f"- Exact culture observations: `{len(obs)}`",
        f"- Exact LP solves recorded: `{int(obs['n_actual_lp_solves'].fillna(0).sum())}`",
        f"- Solver failures: `{int((obs['exact_status'] != 'complete_exact_dynamic_pfba').sum())}`",
        f"- Cache hits: `{int(obs['cache_hit'].astype(bool).sum())}`",
        "",
        "## Method-by-method results",
        "",
    ]
    lines.append(markdown_table(summary.sort_values(["method_id", "campaign_id"])))
    lines.extend(
        [
            "",
            "## Statistical analysis",
            "",
            markdown_table(stats),
            "",
            "## Biological interpretation",
            "",
            "The retained intervention library concentrates on mechanisms that survived the exact dynamic constraint pipeline: heterologous pathway capacities, oxygen/glucose environment coupling, and GGPP/precursor leverage. Method differences therefore reflect experimental efficiency over a phenotype-effective design space rather than nominal edits that collapse to no-ops.",
            "",
            "## Limitations",
            "",
            "The benchmark remains synthetic, uses three campaign seeds, and should be interpreted as exact-simulator evidence rather than wet-lab proof. Right-censored failures are reported as `>16` cultures, and small-sample bootstrap intervals should not be over-interpreted.",
            "",
            "## Final conclusion",
            "",
        ]
    )
    hybrid_success = float(success.get("hybrid_digital_twin", np.nan))
    hybrid_cultures = float(cultures.get("hybrid_digital_twin", np.nan))
    best_method = str(best_utility.index[0]) if len(best_utility) else ""
    fastest = str(cultures.index[0]) if len(cultures) else ""
    lines.append(
        f"Hybrid success rate was `{hybrid_success:.3g}` with median censored cultures-to-target `{hybrid_cultures:.3g}`. The highest mean best utility was `{best_method}`, and the lowest median censored cultures-to-target was `{fastest}`. Use the right-censored endpoint and paired bootstrap table above as the primary evidence for whether the hybrid reduced exact simulated DBTL experiments."
    )
    REPORT.write_text("\n".join(str(x) for x in lines if x is not None) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-exact", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_all_outputs()
    else:
        run_benchmark(max_new_exact=args.max_new_exact)


if __name__ == "__main__":
    main()
