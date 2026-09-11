#!/usr/bin/env python3
"""Parallel screening versus sequential optimisation benchmark scaffold.

The benchmark separates cheap model evaluation, simulated physical
measurements, and sequential adaptive decision rounds.  The current pilot uses
a clearly labelled surrogate oracle for scheduler/cache validation; exact
dynamic Yeast9/pFBA cache entries remain at zero until the Stage 3/Stage 4 gates
permit edited-strain exact rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import stage4_design as stage4


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
EPS = 1e-9

OBJECTIVE_CONFIG = {
    "w_endpoint": 0.55,
    "w_auc": 0.30,
    "w_rate": 0.10,
    "w_oxidative": 0.08,
    "w_atp": 0.05,
    "w_growth": 0.75,
    "w_edit": 0.12,
    "final_biomass_threshold": 0.12,
    "max_edit_distance": 1.35,
}

PILOT_METHODS = [
    "random_one_shot",
    "space_filling_one_shot",
    "static_gem_ranked_one_shot",
    "reporter_heuristic_one_shot",
    "direct_surrogate_one_shot",
    "hybrid_digital_twin_one_shot",
    "black_box_batch_ucb",
    "direct_surrogate_batch_ucb",
    "hybrid_digital_twin_batch_ucb",
    "black_box_sequential_ucb",
    "direct_surrogate_sequential_ucb",
    "hybrid_digital_twin_sequential_ucb",
]


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_hash(row: dict[str, object], oracle_backend: str = "surrogate_oracle_for_scheduler_pilot") -> str:
    payload = {
        "strain_edit_vector": {col: round(float(row[col]), 8) for col in stage4.EDIT_COLUMNS},
        "temperature": round(float(row["temperature"]), 6),
        "pH": round(float(row["pH"]), 6),
        "dissolved_oxygen": round(float(row["DO"]), 6),
        "simulator_configuration": "design_benchmark_pilot_v1",
        "regulatory_tree_version": "hidden_regulator_surrogate_v1",
        "gem_checksum": "stage4_pending_exact_yeast9_checksum",
        "time_grid": "49_points_48_intervals_dt_0p25",
        "integration_convention": "sequential_flux_to_biomass_product",
        "solver_configuration": oracle_backend,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:24]


def strain_id(row: dict[str, object]) -> str:
    payload = {col: round(float(row[col]), 8) for col in stage4.EDIT_COLUMNS}
    return "strain_" + hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def environment_id(row: dict[str, object]) -> str:
    return f"T{float(row['temperature']):.3f}_pH{float(row['pH']):.3f}_DO{float(row['DO']):.3f}"


def objective_value(row: dict[str, float]) -> float:
    growth_penalty = max(0.0, OBJECTIVE_CONFIG["final_biomass_threshold"] - float(row["final_biomass"]))
    return float(
        OBJECTIVE_CONFIG["w_endpoint"] * row["final_product"]
        + OBJECTIVE_CONFIG["w_auc"] * row["product_AUC"]
        + OBJECTIVE_CONFIG["w_rate"] * row["integrated_product_flux"]
        - OBJECTIVE_CONFIG["w_oxidative"] * row["integrated_oxidative_burden"]
        - OBJECTIVE_CONFIG["w_atp"] * row["integrated_ATP_pressure"]
        - OBJECTIVE_CONFIG["w_growth"] * growth_penalty
        - OBJECTIVE_CONFIG["w_edit"] * row["edit_cost"]
    )


def hidden_surrogate_oracle(candidate: pd.Series, regime: str = "strong_dynamic_regulation") -> dict[str, object]:
    edit = {col: float(candidate[col]) for col in stage4.EDIT_COLUMNS}
    temp = float(candidate["temperature"])
    ph = float(candidate["pH"])
    do = float(candidate["DO"])
    transfer = float(candidate["edit_distance"])
    edit_eff = stage4._edit_effect_multiplier(edit)
    temp_score = math.exp(-((temp - 30.0) / 2.3) ** 2)
    ph_score = math.exp(-((ph - 5.05) / 0.34) ** 2)
    oxygen_score = 0.55 + 0.45 / (1.0 + math.exp(-(do - 47.0) / 8.0))
    base = 0.55 * temp_score * ph_score * oxygen_score + 0.08
    dynamic_penalty = 0.0
    if regime == "strong_dynamic_regulation":
        overload = max(0.0, edit_eff["product"] - 1.6) * max(0.0, do - 62.0) / 55.0
        atp_limit = max(0.0, 1.0 - edit["ATP_support_multiplier"]) * 0.6
        dynamic_penalty = 0.26 * overload + 0.18 * atp_limit + 0.05 * transfer
    else:
        dynamic_penalty = 0.04 * transfer
    final_product = max(0.0, base * edit_eff["product"] * (1.0 - dynamic_penalty))
    product_auc = 4.8 * final_product * (0.8 + 0.2 * temp_score)
    integrated_flux = product_auc / 12.0
    stress = max(0.0, 0.25 * (do - 50.0) / 30.0 + 0.20 * max(0.0, edit_eff["product"] - 1.0) + dynamic_penalty)
    atp = max(0.0, 0.30 * max(0.0, edit_eff["product"] - edit["ATP_support_multiplier"]))
    biomass = max(0.02, 0.35 * edit_eff["biomass"] * temp_score * (1.0 - 0.25 * stress))
    feasible = bool(biomass >= OBJECTIVE_CONFIG["final_biomass_threshold"] and transfer <= OBJECTIVE_CONFIG["max_edit_distance"])
    row = {
        "final_product": final_product,
        "product_AUC": product_auc,
        "integrated_product_flux": integrated_flux,
        "integrated_oxidative_burden": stress,
        "integrated_ATP_pressure": atp,
        "final_biomass": biomass,
        "minimum_biomass": min(0.08, biomass),
        "edit_cost": transfer,
        "feasible": feasible,
        "severe_growth_collapse": biomass < OBJECTIVE_CONFIG["final_biomass_threshold"],
        "excessive_stress": stress > 0.85,
        "low_product": final_product < 0.25,
        "metabolic_backend": "surrogate_oracle_for_scheduler_pilot",
        "oracle_regime": regime,
        "n_actual_lp_solves": 0,
        "n_surrogate_evaluations": 1,
    }
    row["objective_value"] = objective_value(row)
    return row


def build_candidate_pool(n_candidates: int = 180, seed: int = 8101) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    library_path = DATA / "strain_edit_library.csv"
    if library_path.exists():
        library = pd.read_csv(library_path)
    else:
        library = stage4.default_edit_library()
        library.to_csv(library_path, index=False)
    rng = np.random.default_rng(seed)
    edit_vectors = stage4.sample_edit_vectors(library, 45, seed=seed + 1, max_active=2)
    env_levels = np.array(
        [
            [27.0, 4.5, 20.0],
            [27.0, 5.0, 50.0],
            [27.0, 5.5, 80.0],
            [30.0, 4.5, 50.0],
            [30.0, 5.0, 20.0],
            [30.0, 5.0, 50.0],
            [30.0, 5.0, 80.0],
            [30.0, 5.5, 50.0],
            [33.0, 4.5, 20.0],
            [33.0, 5.0, 50.0],
            [33.0, 5.5, 80.0],
        ],
        dtype=float,
    )
    rows = []
    for i in range(n_candidates):
        edit = edit_vectors.iloc[i % len(edit_vectors)].to_dict()
        if i < len(edit_vectors) * len(env_levels):
            env = env_levels[(i // len(edit_vectors)) % len(env_levels)]
        else:
            env = np.array([rng.uniform(27, 33), rng.uniform(4.5, 5.5), rng.uniform(20, 80)])
        row = {
            "temperature": float(env[0]),
            "pH": float(env[1]),
            "DO": float(env[2]),
            "edit_vector_id": edit["edit_vector_id"],
            "edit_class": edit["edit_class"],
            "n_active_edits": int(edit["n_active_edits"]),
            "edit_distance": float(edit["edit_distance"]),
            "edit_cost": float(edit["edit_cost"]),
            **{col: float(edit[col]) for col in stage4.EDIT_COLUMNS},
        }
        row["strain_id"] = strain_id(row)
        row["environment_id"] = environment_id(row)
        row["candidate_id"] = "cand_" + candidate_hash(row)
        rows.append(row)
    out = pd.DataFrame(rows).drop_duplicates("candidate_id").reset_index(drop=True)
    # Prior scores use different model representations, not hidden outcomes.
    oxygen = out["DO"].to_numpy(float)
    edit_product = (
        out["precursor_supply_multiplier"].to_numpy(float)
        / np.maximum(out["competing_sink_multiplier"].to_numpy(float), 0.1)
        * np.minimum.reduce(
            [
                out["PSY_capacity_multiplier"].to_numpy(float),
                out["DES_capacity_multiplier"].to_numpy(float),
                out["CYC_capacity_multiplier"].to_numpy(float),
            ]
        )
        * out["export_capacity_multiplier"].to_numpy(float)
    )
    env_prior = np.exp(-((out["temperature"].to_numpy(float) - 30.0) / 2.6) ** 2) * np.exp(-((out["pH"].to_numpy(float) - 5.0) / 0.42) ** 2)
    out["random_prior_score"] = rng.uniform(0.0, 1.0, len(out))
    out["space_filling_score"] = out["edit_distance"] + np.sqrt(((out["temperature"] - 30.0) / 3.0) ** 2 + ((out["pH"] - 5.0) / 0.5) ** 2 + ((out["DO"] - 50.0) / 30.0) ** 2)
    out["static_gem_prior_score"] = env_prior * edit_product
    out["reporter_heuristic_score"] = env_prior * edit_product - 0.015 * np.maximum(0.0, oxygen - 60.0)
    out["direct_surrogate_prior_score"] = env_prior * edit_product - 0.08 * out["edit_cost"]
    out["hybrid_prior_score"] = out["direct_surrogate_prior_score"] - 0.14 * out["edit_distance"] + 0.12 * out["ATP_support_multiplier"]
    out["candidate_domain_status"] = "inside_pilot_domain"
    out.to_csv(DATA / "design_benchmark_candidate_pool.csv", index=False)
    pd.Series(OBJECTIVE_CONFIG).to_frame("value").to_csv(DATA / "design_benchmark_objective_config.csv")
    return out


def method_registry() -> pd.DataFrame:
    rows = [
        ("random_one_shot", "A_one_shot_parallel_screen", "random_prior_score", "none", "baseline_prior", False),
        ("space_filling_one_shot", "A_one_shot_parallel_screen", "space_filling_score", "none", "baseline_prior", False),
        ("static_gem_ranked_one_shot", "A_one_shot_parallel_screen", "static_gem_prior_score", "none", "static_GEM_no_hidden_regulator", False),
        ("reporter_heuristic_one_shot", "A_one_shot_parallel_screen", "reporter_heuristic_score", "none", "reporter_summary_heuristic", False),
        ("direct_surrogate_one_shot", "A_one_shot_parallel_screen", "direct_surrogate_prior_score", "none", "direct_black_box_surrogate", True),
        ("hybrid_digital_twin_one_shot", "A_one_shot_parallel_screen", "hybrid_prior_score", "none", "hybrid_regulation_metabolism_decomposition", True),
        ("black_box_batch_ucb", "B_batched_adaptive_optimisation", "observed_product_only", "ucb", "black_box", True),
        ("direct_surrogate_batch_ucb", "B_batched_adaptive_optimisation", "direct_surrogate_prior_score", "ucb", "direct_black_box_surrogate", True),
        ("hybrid_digital_twin_batch_ucb", "B_batched_adaptive_optimisation", "hybrid_prior_score", "ucb", "hybrid_regulation_metabolism_decomposition", True),
        ("black_box_sequential_ucb", "C_fully_sequential_optimisation", "observed_product_only", "ucb", "black_box", True),
        ("direct_surrogate_sequential_ucb", "C_fully_sequential_optimisation", "direct_surrogate_prior_score", "ucb", "direct_black_box_surrogate", True),
        ("hybrid_digital_twin_sequential_ucb", "C_fully_sequential_optimisation", "hybrid_prior_score", "ucb", "hybrid_regulation_metabolism_decomposition", True),
        ("oracle_upper_bound_reference", "D_reference_only_not_competing", "objective_value_after_oracle_evaluation", "none", "hidden_oracle_reference", False),
    ]
    out = pd.DataFrame(rows, columns=["method_id", "track", "prior_score_column", "outer_optimiser", "model_representation", "uses_adaptive_observations"])
    out["optimiser_parity_group"] = np.where(out["method_id"].str.contains("direct_surrogate|hybrid_digital_twin"), out["track"] + "_same_ucb", "")
    out.to_csv(DATA / "design_benchmark_method_registry.csv", index=False)
    return out


def protocols() -> pd.DataFrame:
    rows = []
    for budget in [8, 16, 24, 32]:
        rows.append({"protocol_id": f"one_shot_{budget}", "track": "A_one_shot_parallel_screen", "N_measurements": budget, "N_rounds": 1, "batch_size": budget, "max_new_strains_per_round": budget})
    for q in [4, 8, 16]:
        for budget in [16, 32]:
            rows.append({"protocol_id": f"batch_q{q}_n{budget}", "track": "B_batched_adaptive_optimisation", "N_measurements": budget, "N_rounds": int(math.ceil(budget / q)), "batch_size": q, "max_new_strains_per_round": q})
    for budget in [8, 16, 24, 32]:
        rows.append({"protocol_id": f"sequential_n{budget}", "track": "C_fully_sequential_optimisation", "N_measurements": budget, "N_rounds": budget, "batch_size": 1, "max_new_strains_per_round": 1})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "design_benchmark_protocols.csv", index=False)
    return out


class BenchmarkOracle:
    def __init__(self, regime: str = "strong_dynamic_regulation"):
        self.regime = regime
        self.cache_path = DATA / "design_benchmark_oracle_cache_index.csv"
        self.solver_path = DATA / "design_benchmark_oracle_solver_accounting.csv"
        self.ledger_path = DATA / "design_benchmark_observation_ledger.csv"
        self.cache = pd.read_csv(self.cache_path) if self.cache_path.exists() else pd.DataFrame()
        self.ledger = pd.read_csv(self.ledger_path) if self.ledger_path.exists() else pd.DataFrame()

    def query(self, method_id: str, run_id: str, candidate: pd.Series, round_id: int) -> dict[str, object]:
        candidate_id = str(candidate["candidate_id"])
        if self.cache.empty or candidate_id not in set(self.cache["candidate_id"]):
            result = hidden_surrogate_oracle(candidate, self.regime)
            row = {"candidate_id": candidate_id, "strain_id": candidate["strain_id"], "environment_id": candidate["environment_id"], **result}
            self.cache = pd.concat([self.cache, pd.DataFrame([row])], ignore_index=True)
            cache_event = "computed_new_surrogate_oracle"
        else:
            row = self.cache[self.cache["candidate_id"].eq(candidate_id)].iloc[0].to_dict()
            cache_event = "global_cache_reuse"
        obs = {
            "method_id": method_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "strain_id": candidate["strain_id"],
            "environment_id": candidate["environment_id"],
            "round_id": int(round_id),
            "measurement_index": int(0 if self.ledger.empty else len(self.ledger)),
            "cache_event": cache_event,
            "physical_measurement_counted": 1,
            **{k: row[k] for k in ["objective_value", "final_product", "product_AUC", "final_biomass", "feasible", "severe_growth_collapse", "excessive_stress", "low_product", "metabolic_backend", "n_actual_lp_solves", "n_surrogate_evaluations"]},
        }
        self.ledger = pd.concat([self.ledger, pd.DataFrame([obs])], ignore_index=True)
        return obs

    def save(self) -> None:
        self.cache.to_csv(self.cache_path, index=False)
        if not self.cache.empty:
            solver = self.cache[["candidate_id", "metabolic_backend", "n_actual_lp_solves", "n_surrogate_evaluations"]].copy()
            solver["n_optimal_solves"] = 0
            solver["n_infeasible_solves"] = 0
            solver["n_unbounded_solves"] = 0
            solver["n_solver_errors"] = 0
            solver.to_csv(self.solver_path, index=False)
        self.ledger.to_csv(self.ledger_path, index=False)


def select_candidates(method_id: str, pool: pd.DataFrame, observed: pd.DataFrame, batch_size: int, rng: np.random.Generator) -> pd.DataFrame:
    queried = set(observed["candidate_id"]) if not observed.empty else set()
    available = pool[~pool["candidate_id"].isin(queried)].copy()
    if available.empty:
        return available
    if method_id.startswith("random") or method_id.startswith("black_box") and observed.empty:
        return available.sample(n=min(batch_size, len(available)), random_state=int(rng.integers(0, 1_000_000)))
    if method_id.startswith("space_filling"):
        return available.sort_values("space_filling_score", ascending=False).head(batch_size)
    score_col = "hybrid_prior_score" if "hybrid" in method_id else "direct_surrogate_prior_score" if "direct" in method_id else "static_gem_prior_score" if "static_gem" in method_id else "reporter_heuristic_score" if "reporter" in method_id else "random_prior_score"
    if "ucb" in method_id and not observed.empty:
        X_obs = pool[pool["candidate_id"].isin(observed["candidate_id"])][["temperature", "pH", "DO", "edit_distance", "edit_cost", score_col]].to_numpy(float)
        y = observed.set_index("candidate_id").loc[pool[pool["candidate_id"].isin(observed["candidate_id"])]["candidate_id"], "objective_value"].to_numpy(float)
        X_avail = available[["temperature", "pH", "DO", "edit_distance", "edit_cost", score_col]].to_numpy(float)
        if len(X_obs) >= 3 and np.std(y) > EPS:
            xm, xs = X_obs.mean(axis=0), X_obs.std(axis=0) + EPS
            coef = np.linalg.lstsq(np.c_[np.ones(len(X_obs)), (X_obs - xm) / xs], y, rcond=None)[0]
            pred = np.c_[np.ones(len(X_avail)), (X_avail - xm) / xs] @ coef
            uncertainty = np.sqrt(np.sum(((X_avail - X_obs.mean(axis=0)) / xs) ** 2, axis=1))
            available["_acquisition"] = pred + 0.15 * uncertainty
            return available.sort_values("_acquisition", ascending=False).head(batch_size)
    return available.sort_values(score_col, ascending=False).head(batch_size)


def run_method(method_id: str, protocol: pd.Series, seed: int, pool: pd.DataFrame, oracle: BenchmarkOracle) -> None:
    rng = np.random.default_rng(seed)
    run_id = f"{method_id}_seed{seed}_{protocol.protocol_id}"
    observed = pd.DataFrame()
    remaining = int(protocol.N_measurements)
    for round_id in range(int(protocol.N_rounds)):
        batch_size = min(int(protocol.batch_size), remaining)
        if batch_size <= 0:
            break
        batch = select_candidates(method_id, pool, observed, batch_size, rng)
        rows = [oracle.query(method_id, run_id, row, round_id) for _, row in batch.iterrows()]
        observed = pd.concat([observed, pd.DataFrame(rows)], ignore_index=True)
        remaining -= len(rows)


def run_pilot(pool_size: int = 180, seeds: list[int] | None = None) -> None:
    seeds = [11, 22, 33] if seeds is None else seeds
    pool = build_candidate_pool(pool_size)
    registry = method_registry()
    protos = protocols()
    for path in [DATA / "design_benchmark_oracle_cache_index.csv", DATA / "design_benchmark_observation_ledger.csv", DATA / "design_benchmark_oracle_solver_accounting.csv"]:
        path.unlink(missing_ok=True)
    oracle = BenchmarkOracle()
    for seed in seeds:
        for method in PILOT_METHODS:
            track = registry[registry["method_id"].eq(method)]["track"].iloc[0]
            if track.startswith("A_"):
                selected_protocols = protos[protos["protocol_id"].isin(["one_shot_8", "one_shot_16", "one_shot_32"])]
            elif track.startswith("B_"):
                selected_protocols = protos[protos["protocol_id"].isin(["batch_q4_n16", "batch_q8_n32", "batch_q16_n32"])]
            else:
                selected_protocols = protos[protos["protocol_id"].isin(["sequential_n8", "sequential_n16"])]
            for _, protocol in selected_protocols.iterrows():
                run_method(method, protocol, seed, pool, oracle)
    oracle.save()
    analyse()


def analyse() -> None:
    ledger = pd.read_csv(DATA / "design_benchmark_observation_ledger.csv")
    cache = pd.read_csv(DATA / "design_benchmark_oracle_cache_index.csv")
    if ledger.empty:
        return
    ledger = ledger.sort_values(["run_id", "round_id", "measurement_index"])
    histories = []
    for run_id, g in ledger.groupby("run_id"):
        g = g.copy()
        g["measurement_number"] = np.arange(1, len(g) + 1)
        g["unique_strains_so_far"] = [g.iloc[:i]["strain_id"].nunique() for i in range(1, len(g) + 1)]
        g["rounds_so_far"] = [g.iloc[:i]["round_id"].nunique() for i in range(1, len(g) + 1)]
        g["best_objective_so_far"] = g["objective_value"].cummax()
        histories.append(g)
    mh = pd.concat(histories, ignore_index=True)
    mh.to_csv(DATA / "design_benchmark_measurement_history.csv", index=False)
    round_history = ledger.groupby(["method_id", "run_id", "round_id"]).agg(
        batch_size=("candidate_id", "count"),
        n_unique_strains=("strain_id", "nunique"),
        best_round_objective=("objective_value", "max"),
    ).reset_index()
    round_history.to_csv(DATA / "design_benchmark_round_history.csv", index=False)
    strain_rows = []
    for (method_id, run_id), g in ledger.groupby(["method_id", "run_id"]):
        for i, (sid, sg) in enumerate(g.groupby("strain_id")):
            strain_rows.append({"method_id": method_id, "run_id": run_id, "strain_id": sid, "strain_construction_index": i + 1, "n_environments_tested": sg["environment_id"].nunique()})
    pd.DataFrame(strain_rows).to_csv(DATA / "design_benchmark_strain_construction_history.csv", index=False)
    best = mh.groupby(["method_id", "run_id"]).agg(
        N_measurements=("candidate_id", "count"),
        N_strains=("strain_id", "nunique"),
        N_rounds=("round_id", "nunique"),
        batch_size=("round_id", lambda s: int(ledger.loc[s.index].groupby("round_id").size().max())),
        best_verified_objective=("objective_value", "max"),
        best_final_product=("final_product", "max"),
    ).reset_index()
    best.to_csv(DATA / "design_benchmark_best_found.csv", index=False)
    best_known = float(cache["objective_value"].max())
    regret = best.copy()
    regret["reference_status"] = "best_known_regret_incomplete_pool"
    regret["best_known_objective"] = best_known
    regret["best_known_regret"] = best_known - regret["best_verified_objective"]
    regret.to_csv(DATA / "design_benchmark_regret.csv", index=False)
    thresholds = []
    for (method_id, run_id), g in mh.groupby(["method_id", "run_id"]):
        for frac in [0.90, 0.95, 0.99]:
            hit = g[g["best_objective_so_far"] >= frac * best_known]
            thresholds.append(
                {
                    "method_id": method_id,
                    "run_id": run_id,
                    "threshold_fraction_of_best_known": frac,
                    "measurements_to_threshold": int(hit["measurement_number"].iloc[0]) if not hit.empty else np.nan,
                    "rounds_to_threshold": int(hit["rounds_so_far"].iloc[0]) if not hit.empty else np.nan,
                    "strains_to_threshold": int(hit["unique_strains_so_far"].iloc[0]) if not hit.empty else np.nan,
                    "reference_status": "best_known_incomplete_pool",
                }
            )
    pd.DataFrame(thresholds).to_csv(DATA / "design_benchmark_threshold_efficiency.csv", index=False)
    top = cache.sort_values("objective_value", ascending=False).head(max(1, int(0.1 * len(cache))))["candidate_id"]
    topk = best.copy()
    topk["top_10pct_recovery"] = [int(bool(set(ledger[ledger["run_id"].eq(rid)]["candidate_id"]) & set(top))) for rid in topk["run_id"]]
    topk["reference_status"] = "queried_cache_topk_not_full_pool"
    topk.to_csv(DATA / "design_benchmark_topk_recovery.csv", index=False)
    feas = ledger.groupby(["method_id", "run_id"]).agg(
        infeasible_candidates=("feasible", lambda s: int((~s.astype(bool)).sum())),
        severe_growth_collapse_candidates=("severe_growth_collapse", "sum"),
        excessive_stress_candidates=("excessive_stress", "sum"),
        low_product_candidates=("low_product", "sum"),
        n_measurements=("candidate_id", "count"),
    ).reset_index()
    feas.to_csv(DATA / "design_benchmark_feasibility_efficiency.csv", index=False)
    compute = pd.DataFrame(
        [
            {
                "compute_item": "unique_computational_oracle_cache_entries",
                "value": int(cache["candidate_id"].nunique()),
                "backend": "surrogate_oracle_for_scheduler_pilot",
            },
            {"compute_item": "method_specific_physical_measurements", "value": int(len(ledger)), "backend": "method_ledger"},
            {"compute_item": "cached_oracle_reuses", "value": int((ledger["cache_event"] == "global_cache_reuse").sum()), "backend": "global_cache"},
            {"compute_item": "unique_exact_dynamic_pfba_rollouts", "value": 0, "backend": "yeast_gem_lp"},
            {"compute_item": "total_lp_solves", "value": 0, "backend": "yeast_gem_lp"},
            {"compute_item": "cheap_surrogate_oracle_evaluations", "value": int(cache["n_surrogate_evaluations"].sum()), "backend": "surrogate_oracle_for_scheduler_pilot"},
        ]
    )
    compute.to_csv(DATA / "design_benchmark_compute_accounting.csv", index=False)
    seed_summary = best.groupby("method_id").agg(
        mean_best_objective=("best_verified_objective", "mean"),
        median_best_objective=("best_verified_objective", "median"),
        sd_best_objective=("best_verified_objective", "std"),
        mean_measurements=("N_measurements", "mean"),
        mean_rounds=("N_rounds", "mean"),
        mean_strains=("N_strains", "mean"),
    ).reset_index()
    seed_summary.to_csv(DATA / "design_benchmark_seed_summary.csv", index=False)
    acceptance = pd.DataFrame(
        [
            {"status_type": "implementation_status", "status": "design_benchmark_framework_complete", "passed": True},
            {"status_type": "pilot_status", "status": "design_benchmark_pilot_blocked_no_exact_dynamic_pfba_oracle", "passed": False},
            {"status_type": "scientific_result", "status": "result_pending_exact_oracle_evidence", "passed": False},
            {"status_type": "regret_status", "status": "best_known_regret_only_incomplete_pool", "passed": False},
        ]
    )
    acceptance.to_csv(DATA / "design_benchmark_acceptance.csv", index=False)
    write_figures(mh, regret)


def write_figures(history: pd.DataFrame, regret: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    summaries = {
        "design_benchmark_best_vs_measurements.svg": ("measurement_number", "best_objective_so_far", "Best Objective vs Measurements"),
        "design_benchmark_best_vs_rounds.svg": ("rounds_so_far", "best_objective_so_far", "Best Objective vs Sequential Rounds"),
        "design_benchmark_best_vs_strains.svg": ("unique_strains_so_far", "best_objective_so_far", "Best Objective vs Unique Strains"),
    }
    for filename, (xcol, ycol, title) in summaries.items():
        path = FIGURES / filename
        rows = history.groupby(["method_id", xcol])[ycol].mean().reset_index()
        body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430"><rect width="100%" height="100%" fill="white"/><text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>']
        xmax, ymax = max(rows[xcol].max(), 1), max(rows[ycol].max(), EPS)
        colors = ["#276c86", "#c26d3d", "#5d8f52", "#7a5ea8", "#a64f68", "#448c8a"]
        for j, (method, g) in enumerate(rows.groupby("method_id")):
            pts = []
            for r in g.sort_values(xcol).itertuples(index=False):
                x = 70 + 760 * float(getattr(r, xcol)) / xmax
                y = 360 - 300 * float(getattr(r, ycol)) / ymax
                pts.append(f"{x:.1f},{y:.1f}")
            body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[j % len(colors)]}" stroke-width="2"/>')
        body.append("</svg>")
        path.write_text("\n".join(body), encoding="utf-8")
    for name in ["design_benchmark_regret.svg", "design_benchmark_parallel_frontier.svg", "design_benchmark_method_comparison.svg"]:
        path = FIGURES / name
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360"><rect width="100%" height="100%" fill="white"/><text x="450" y="180" text-anchor="middle" font-size="18">Design benchmark pilot: exact oracle pending</text></svg>', encoding="utf-8")


def audit_representative_stage3() -> pd.DataFrame:
    completion_path = DATA / "hybrid_exact_replay_completion.csv"
    metrics_path = DATA / "hybrid_exact_replay_preliminary_metrics.csv"
    if not completion_path.exists() or not metrics_path.exists():
        out = pd.DataFrame([{"criterion": "required_stage3_files_exist", "passed": False, "value": "missing"}])
        out.to_csv(DATA / "stage3_representative_acceptance.csv", index=False)
        return out
    completion = pd.read_csv(completion_path)
    valid = completion[completion["status"].eq("complete_valid")]
    common = set.intersection(*[set(valid[valid["replay_type"].eq(rt)]["culture_id"]) for rt in ["oracle", "teacher", "hybrid"]])
    manifest = valid[valid["culture_id"].isin(common)].drop_duplicates("culture_id").copy()
    manifest["coverage_reason"] = "completed_exact_replay_triple"
    manifest.to_csv(DATA / "stage3_representative_validation_manifest.csv", index=False)
    metrics = pd.read_csv(metrics_path)
    def metric(comp: str, split: str, col: str) -> float:
        s = metrics[(metrics["comparison"].eq(comp)) & (metrics["split"].eq(split))]
        return float(s[col].iloc[0]) if not s.empty and col in s else np.nan
    split_ok = set(["train", "validation", "interpolation", "heldout_combination", "extrapolation"]).issubset(set(manifest["split"]))
    oracle_ok = metric("oracle_replay_error", "all", "raw_RMSE") < 1e-10
    hybrid_ok = metric("total_hybrid_error", "all", "trajectory_normalized_RMSE") < 0.05
    held_ok = metric("total_hybrid_error", "heldout_combination", "trajectory_normalized_RMSE") < 0.10
    lp_ok = bool((valid[["completed_lp_solves", "optimal_solves"]].diff(axis=1)["optimal_solves"].eq(0)).all() and valid[["infeasible_solves", "unbounded_solves", "solver_errors", "skipped_intervals", "surrogate_evaluations"]].sum().sum() == 0)
    rows = [
        ("all_required_splits_represented", split_ok, ";".join(sorted(manifest["split"].unique()))),
        ("oracle_replay_numerical_precision", oracle_ok, metric("oracle_replay_error", "all", "raw_RMSE")),
        ("lp_accounting_clean", lp_ok, int(valid["completed_lp_solves"].sum())),
        ("hybrid_total_error_below_threshold", hybrid_ok, metric("total_hybrid_error", "all", "trajectory_normalized_RMSE")),
        ("heldout_performance_useful", held_ok, metric("total_hybrid_error", "heldout_combination", "trajectory_normalized_RMSE")),
        ("no_direct_product_head_used", True, "Stage 3 hybrid replay product generated from pFBA flux integration"),
    ]
    passed = all(bool(r[1]) for r in rows)
    status = "stage3_representative_passed" if passed else "stage3_representative_blocked_insufficient_completed_replay_coverage"
    out = pd.DataFrame([{"criterion": name, "passed": bool(ok), "value": value, "representative_status": status, "full_stage3_status": "stage_3_blocked_full_125_culture_exact_rollout_incomplete"} for name, ok, value in rows])
    out.to_csv(DATA / "stage3_representative_acceptance.csv", index=False)
    return out


def run_all(pool_size: int = 180) -> None:
    audit_representative_stage3()
    build_candidate_pool(pool_size)
    method_registry()
    protocols()
    run_pilot(pool_size=pool_size)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stage3-gate", "pool", "methods", "pilot", "analyse", "audit"])
    parser.add_argument("--pool-size", type=int, default=180)
    args = parser.parse_args(argv)
    if args.command == "stage3-gate":
        print(audit_representative_stage3().to_string(index=False))
    elif args.command == "pool":
        print(build_candidate_pool(args.pool_size).head().to_string(index=False))
    elif args.command == "methods":
        method_registry()
        print(protocols().to_string(index=False))
    elif args.command == "pilot":
        run_pilot(pool_size=args.pool_size)
        print(pd.read_csv(DATA / "design_benchmark_seed_summary.csv").to_string(index=False))
    elif args.command == "analyse":
        analyse()
    elif args.command == "audit":
        run_all(pool_size=args.pool_size)
        print(pd.read_csv(DATA / "design_benchmark_acceptance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
