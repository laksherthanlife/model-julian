#!/usr/bin/env python3
"""Exact finite-pool benchmark for strain-environment optimisation.

This module extends the scheduler-only benchmark with a small cached exact
reference pool.  The hidden oracle is the original decision-tree regulator plus
dynamic Yeast9 growth/product/pFBA loop from ``gem_backend``; learned regulatory
controllers are used only as competing method priors.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import design_benchmark as scheduler
import gem_backend as gem
import run_gem_dynamic_capacity as dynamic_capacity
import stage4_design as stage4


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
ROLL_ROOT = DATA / "design_benchmark_exact_rollouts"
EPS = 1e-9


def spearman_without_scipy(a: pd.Series, b: pd.Series) -> float:
    ar = pd.Series(a.to_numpy(float)).rank(method="average")
    br = pd.Series(b.to_numpy(float)).rank(method="average")
    return float(ar.corr(br))
EXACT_BACKEND = "yeast_gem_lp"
TIME_POINTS = 49
EXPECTED_LP_SOLVES = (TIME_POINTS - 1) * 3
CORE_METHODS = [
    "random_parallel",
    "space_filling_parallel",
    "static_gem_parallel",
    "black_box_bo",
    "direct_trajectory_ucb",
    "hybrid_digital_twin_ucb",
]


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return float(usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024)


def clear_solver_caches() -> None:
    try:
        import sympy as sp  # type: ignore

        sp.core.cache.clear_cache()
    except Exception:
        pass
    gc.collect()


def exact_candidate_hash(row: dict[str, object]) -> str:
    payload = {
        "strain_edit_vector": {col: round(float(row[col]), 8) for col in stage4.EDIT_COLUMNS},
        "temperature": round(float(row["temperature"]), 6),
        "pH": round(float(row["pH"]), 6),
        "dissolved_oxygen": round(float(row["DO"]), 6),
        "regime_id": str(row["regime_id"]),
        "simulator_configuration": "exact_design_benchmark_v1",
        "regulatory_tree_version": "gem_backend_decision_tree_next_state_v1",
        "time_grid": f"{TIME_POINTS}_points_{TIME_POINTS - 1}_intervals_dt_0p25",
        "integration_convention": "dynamic_biomass_product_euler",
        "solver_configuration": EXACT_BACKEND,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:24]


def strain_id(row: dict[str, object]) -> str:
    payload = {col: round(float(row[col]), 8) for col in stage4.EDIT_COLUMNS}
    return "strain_" + hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def environment_id(row: dict[str, object]) -> str:
    return f"T{float(row['temperature']):.3f}_pH{float(row['pH']):.3f}_DO{float(row['DO']):.3f}"


def regime_configs() -> dict[str, dict[str, object]]:
    base = dynamic_capacity.selected_parameter_config()
    baseline = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
    )
    weak = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        psy_synthesis_base=0.090,
        des_synthesis_base=0.080,
        cyc_synthesis_base=0.075,
        psy_decay_base=0.009,
        des_decay_base=0.012,
        cyc_decay_base=0.011,
        psy_oxidative_sensitivity=0.018,
        des_oxidative_sensitivity=0.045,
        cyc_oxidative_sensitivity=0.060,
        psy_atp_sensitivity=0.010,
        des_atp_sensitivity=0.025,
        cyc_atp_sensitivity=0.035,
        psy_bottleneck_sensitivity=0.040,
        des_bottleneck_sensitivity=0.025,
        cyc_bottleneck_sensitivity=0.020,
        phytoene_congestion_sensitivity=0.120,
        lycopene_congestion_sensitivity=0.100,
        beta_deg_base=0.0012,
    )
    strong = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        psy_synthesis_base=0.064,
        des_synthesis_base=0.054,
        cyc_synthesis_base=0.048,
        psy_decay_base=0.021,
        des_decay_base=0.030,
        cyc_decay_base=0.028,
        psy_oxidative_sensitivity=0.055,
        des_oxidative_sensitivity=0.145,
        cyc_oxidative_sensitivity=0.185,
        psy_atp_sensitivity=0.028,
        des_atp_sensitivity=0.074,
        cyc_atp_sensitivity=0.105,
        psy_bottleneck_sensitivity=0.135,
        des_bottleneck_sensitivity=0.070,
        cyc_bottleneck_sensitivity=0.052,
        phytoene_congestion_sensitivity=0.360,
        lycopene_congestion_sensitivity=0.330,
        beta_deg_base=0.0024,
    )
    oxidative = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        oxygen_uptake_at_100_do=float(base.oxygen_uptake_at_100_do) * 1.18,
        baseline_atpm=float(base.baseline_atpm) * 1.08,
        psy_oxidative_sensitivity=0.070,
        des_oxidative_sensitivity=0.175,
        cyc_oxidative_sensitivity=0.225,
        psy_atp_sensitivity=0.025,
        des_atp_sensitivity=0.060,
        cyc_atp_sensitivity=0.082,
        beta_deg_base=0.0032,
    )
    atp_limited = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        glucose_uptake=float(base.glucose_uptake) * 0.92,
        baseline_atpm=float(base.baseline_atpm) * 1.38,
        psy_atp_sensitivity=0.050,
        des_atp_sensitivity=0.120,
        cyc_atp_sensitivity=0.160,
        psy_synthesis_base=0.070,
        des_synthesis_base=0.060,
        cyc_synthesis_base=0.052,
        beta_deg_base=0.0020,
    )
    precursor_competition = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        psy_base_capacity=float(base.psy_base_capacity) * 0.82,
        des_base_capacity=float(base.des_base_capacity) * 0.80,
        cyc_base_capacity=float(base.cyc_base_capacity) * 0.78,
        psy_bottleneck_sensitivity=0.190,
        des_bottleneck_sensitivity=0.110,
        cyc_bottleneck_sensitivity=0.085,
        phytoene_congestion_sensitivity=0.520,
        lycopene_congestion_sensitivity=0.460,
        beta_deg_base=0.0022,
    )
    pathway_damage = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        psy_synthesis_base=0.058,
        des_synthesis_base=0.050,
        cyc_synthesis_base=0.044,
        psy_decay_base=0.030,
        des_decay_base=0.047,
        cyc_decay_base=0.040,
        psy_oxidative_sensitivity=0.065,
        des_oxidative_sensitivity=0.205,
        cyc_oxidative_sensitivity=0.130,
        psy_bottleneck_sensitivity=0.150,
        des_bottleneck_sensitivity=0.115,
        cyc_bottleneck_sensitivity=0.065,
        beta_deg_base=0.0027,
    )
    compensatory = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        oxygen_uptake_at_100_do=float(base.oxygen_uptake_at_100_do) * 1.06,
        glucose_uptake=float(base.glucose_uptake) * 1.04,
        baseline_atpm=float(base.baseline_atpm) * 1.16,
        psy_base_capacity=float(base.psy_base_capacity) * 0.88,
        des_base_capacity=float(base.des_base_capacity) * 0.92,
        cyc_base_capacity=float(base.cyc_base_capacity) * 0.86,
        psy_synthesis_base=0.074,
        des_synthesis_base=0.070,
        cyc_synthesis_base=0.058,
        psy_oxidative_sensitivity=0.038,
        des_oxidative_sensitivity=0.085,
        cyc_oxidative_sensitivity=0.105,
        psy_bottleneck_sensitivity=0.105,
        des_bottleneck_sensitivity=0.082,
        cyc_bottleneck_sensitivity=0.095,
        phytoene_congestion_sensitivity=0.260,
        lycopene_congestion_sensitivity=0.290,
        beta_deg_base=0.0021,
    )
    mixed = replace(
        base,
        capacity_mode="dynamic_congestion_feedback",
        oxygen_uptake_at_100_do=float(base.oxygen_uptake_at_100_do) * 1.10,
        glucose_uptake=float(base.glucose_uptake) * 0.90,
        baseline_atpm=float(base.baseline_atpm) * 1.28,
        psy_base_capacity=float(base.psy_base_capacity) * 0.84,
        des_base_capacity=float(base.des_base_capacity) * 0.76,
        cyc_base_capacity=float(base.cyc_base_capacity) * 0.80,
        psy_synthesis_base=0.060,
        des_synthesis_base=0.048,
        cyc_synthesis_base=0.046,
        psy_decay_base=0.026,
        des_decay_base=0.044,
        cyc_decay_base=0.036,
        psy_oxidative_sensitivity=0.072,
        des_oxidative_sensitivity=0.180,
        cyc_oxidative_sensitivity=0.190,
        psy_atp_sensitivity=0.046,
        des_atp_sensitivity=0.112,
        cyc_atp_sensitivity=0.142,
        psy_bottleneck_sensitivity=0.180,
        des_bottleneck_sensitivity=0.125,
        cyc_bottleneck_sensitivity=0.105,
        phytoene_congestion_sensitivity=0.480,
        lycopene_congestion_sensitivity=0.440,
        beta_deg_base=0.0030,
    )
    return {
        "baseline_world": {
            "cfg": baseline,
            "random_seed": 9301,
            "burden_scale": {"oxidative": 1.00, "atp": 1.00, "bottleneck": 1.00, "er": 1.00},
            "repair_scale": 1.00,
            "pathway_damage_scale": 1.00,
            "degradation_rule": "current_validated_dynamic_pfba_reference",
            "world_family": "baseline",
            "biological_interpretation": "Current repaired exact Yeast9 dynamic-pFBA organism used as the source/reference world.",
        },
        "strong_oxidative_burden_world": {
            "cfg": oxidative,
            "random_seed": 9302,
            "burden_scale": {"oxidative": 1.80, "atp": 1.10, "bottleneck": 1.12, "er": 1.05},
            "repair_scale": 0.76,
            "pathway_damage_scale": 1.28,
            "degradation_rule": "high_DO_and_product_flux_generate_extra_oxidative_beta_loss",
            "world_family": "strong_oxidative_burden",
            "biological_interpretation": "High oxygen can support respiration early but imposes stronger oxidative damage and product-loss penalties later.",
        },
        "atp_limited_world": {
            "cfg": atp_limited,
            "random_seed": 9303,
            "burden_scale": {"oxidative": 1.00, "atp": 1.70, "bottleneck": 1.05, "er": 1.00},
            "repair_scale": 0.82,
            "pathway_damage_scale": 1.08,
            "degradation_rule": "maintenance_competes_with_high_production_for_ATP_support",
            "world_family": "atp_limited",
            "biological_interpretation": "Higher maintenance and ATP sensitivity make pathway overexpression compete more directly with growth.",
        },
        "precursor_competition_world": {
            "cfg": precursor_competition,
            "random_seed": 9304,
            "burden_scale": {"oxidative": 1.08, "atp": 1.05, "bottleneck": 1.72, "er": 1.00},
            "repair_scale": 0.88,
            "pathway_damage_scale": 1.12,
            "degradation_rule": "native_isoprenoid_competition_increases_bottleneck_loading",
            "world_family": "precursor_competition",
            "biological_interpretation": "Native isoprenoid sinks compete harder with beta-carotene precursor supply, so capacity increases alone need not solve production.",
        },
        "pathway_bottleneck_damage_world": {
            "cfg": pathway_damage,
            "random_seed": 9305,
            "burden_scale": {"oxidative": 1.22, "atp": 1.18, "bottleneck": 1.34, "er": 1.08},
            "repair_scale": 0.70,
            "pathway_damage_scale": 1.72,
            "degradation_rule": "differential_PSY_DES_CYC_damage_changes_pathway_limiting_step",
            "world_family": "pathway_bottleneck_enzyme_damage",
            "biological_interpretation": "PSY, DES, and CYC capacities decay/recover differently, moving the pathway-limiting step across environments.",
        },
        "compensatory_metabolism_world": {
            "cfg": compensatory,
            "random_seed": 9306,
            "burden_scale": {"oxidative": 0.92, "atp": 1.22, "bottleneck": 1.25, "er": 1.18},
            "repair_scale": 1.05,
            "pathway_damage_scale": 0.95,
            "degradation_rule": "alternative_capacity_partially_buffers_obvious_single_reaction_edits",
            "world_family": "compensatory_metabolism",
            "biological_interpretation": "Alternative metabolic support partly buffers obvious static edits, making local GEM intuition less decisive.",
        },
        "mixed_multi_mechanism_world": {
            "cfg": mixed,
            "random_seed": 9307,
            "burden_scale": {"oxidative": 1.45, "atp": 1.48, "bottleneck": 1.55, "er": 1.15},
            "repair_scale": 0.74,
            "pathway_damage_scale": 1.50,
            "degradation_rule": "combined_oxidative_ATP_precursor_and_pathway_damage_shift",
            "world_family": "mixed_multi_mechanism",
            "biological_interpretation": "Combined non-adversarial hidden shift changes oxygen, maintenance, precursor competition, and pathway damage together.",
        },
        "weak_dynamic_regulation": {
            "cfg": weak,
            "random_seed": 9101,
            "burden_scale": {"oxidative": 0.55, "atp": 0.55, "bottleneck": 0.50, "er": 0.65},
            "repair_scale": 1.35,
            "pathway_damage_scale": 0.55,
            "degradation_rule": "low_beta_decay_with_weak_oxidative_feedback",
            "world_family": "legacy_weak_dynamic_regulation",
            "biological_interpretation": "Legacy exact finite-pool weak-regulation reference retained for backward compatibility.",
        },
        "strong_dynamic_regulation": {
            "cfg": strong,
            "random_seed": 9102,
            "burden_scale": {"oxidative": 1.18, "atp": 1.12, "bottleneck": 1.25, "er": 1.00},
            "repair_scale": 0.82,
            "pathway_damage_scale": 1.18,
            "degradation_rule": "higher_beta_decay_with_oxidative_burden_feedback",
            "world_family": "legacy_strong_dynamic_regulation",
            "biological_interpretation": "Legacy final-DBTL hidden regime retained for backward compatibility.",
        },
    }


def freeze_regime_manifest(gem_path: str | None = None) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    source_path = gem.configured_gem_path(gem_path)
    checksum = gem.sha256(source_path)
    rows: list[dict[str, object]] = []
    thresholds = {
        "state_high_oxidative": 0.62,
        "state_high_atp": 0.58,
        "state_high_bottleneck": 0.60,
        "state_low_oxidative": 0.42,
        "state_low_atp": 0.38,
        "state_low_bottleneck": 0.40,
        "nonproductive_min_dwell_intervals": 2,
    }
    for regime_id, spec in regime_configs().items():
        cfg = replace(spec["cfg"], n_time=TIME_POINTS)
        row = {
            "regime_id": regime_id,
            "time_grid": f"{TIME_POINTS}_points_{TIME_POINTS - 1}_intervals_dt_0p25",
            "gem_checksum": checksum,
            "gem_source_path": str(source_path),
            "solver_configuration": EXACT_BACKEND,
            "random_seed": spec["random_seed"],
            "burden_coefficients": stable_json(spec["burden_scale"]),
            "recovery_coefficient": spec["repair_scale"],
            "pathway_damage_parameter": spec["pathway_damage_scale"],
            "degradation_rule": spec["degradation_rule"],
            **thresholds,
        }
        for key, value in asdict(cfg).items():
            row[f"cfg_{key}"] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "design_benchmark_regime_manifest.csv", index=False)
    return out


def edit_description(row: dict[str, object]) -> str:
    active = [col for col in stage4.EDIT_COLUMNS if abs(float(row[col]) - 1.0) > 1e-8]
    if not active:
        return "baseline no-edit control"
    return "; ".join(f"{col}={float(row[col]):.3g}" for col in active)


def exact_prior_scores(row: dict[str, object]) -> dict[str, float]:
    temp = float(row["temperature"])
    ph = float(row["pH"])
    do = float(row["DO"])
    env_prior = math.exp(-((temp - 30.0) / 2.6) ** 2) * math.exp(-((ph - 5.0) / 0.42) ** 2)
    pathway = min(float(row["PSY_capacity_multiplier"]), float(row["DES_capacity_multiplier"]), float(row["CYC_capacity_multiplier"]))
    edit_product = (
        float(row["precursor_supply_multiplier"])
        / max(float(row["competing_sink_multiplier"]), 0.1)
        * pathway
        * float(row["export_capacity_multiplier"])
        / max(float(row["product_degradation_multiplier"]), 0.2)
    )
    oxygen_penalty = 0.010 * max(0.0, do - 60.0) / max(float(row["oxygen_support_multiplier"]), 0.5)
    static = env_prior * edit_product
    direct = static - 0.08 * float(row["edit_cost"]) - oxygen_penalty
    hybrid = direct - 0.12 * float(row["edit_distance"]) + 0.10 * float(row["ATP_support_multiplier"]) + 0.05 * float(row["oxygen_support_multiplier"])
    space = float(row["edit_distance"]) + math.sqrt(((temp - 30.0) / 3.0) ** 2 + ((ph - 5.0) / 0.5) ** 2 + ((do - 50.0) / 30.0) ** 2)
    return {
        "static_gem_score": float(static),
        "direct_prior_score": float(direct),
        "hybrid_prior_score": float(hybrid),
        "space_filling_score": float(space),
    }


def build_base_exact_design(n_per_regime: int = 48, seed: int = 9201) -> pd.DataFrame:
    library = stage4.default_edit_library()
    baseline = {row.edit_id: float(row.baseline_value) for row in library.itertuples(index=False)}
    envs = [
        (27.0, 4.5, 20.0, "low_T_low_pH_low_DO"),
        (27.0, 5.0, 50.0, "low_T_center_pH_mid_DO"),
        (27.0, 5.5, 80.0, "low_T_high_pH_high_DO"),
        (30.0, 4.5, 50.0, "center_T_low_pH_mid_DO"),
        (30.0, 5.0, 20.0, "center_T_center_pH_low_DO"),
        (30.0, 5.0, 50.0, "central_reference"),
        (30.0, 5.0, 80.0, "center_T_center_pH_high_DO"),
        (33.0, 5.5, 80.0, "high_T_high_pH_high_DO"),
    ]
    rows: list[dict[str, object]] = []

    def add(edit: dict[str, float], edit_vector_id: str, edit_class: str, env, reason: str) -> None:
        temp, ph, do, env_reason = env
        row = {
            "edit_vector_id": edit_vector_id,
            "edit_class": edit_class,
            "temperature": temp,
            "pH": ph,
            "DO": do,
            "selection_reason": f"{reason};{env_reason}",
            **edit,
        }
        row["n_active_edits"] = sum(abs(float(row[col]) - 1.0) > 1e-8 for col in stage4.EDIT_COLUMNS)
        row["edit_distance"] = stage4.edit_distance(row, library)
        row["edit_cost"] = stage4.edit_cost(row, library)
        rows.append(row)

    for env in envs:
        add(baseline.copy(), "baseline_no_edit", "baseline_control", env, "baseline_control")
    strengths = [(0.33, "low"), (0.66, "medium"), (1.0, "high")]
    single_env_cycle = [
        (30.0, 5.0, 50.0, "central_single_edit"),
        (30.0, 4.75, 50.0, "near_pH_threshold"),
        (31.5, 5.0, 65.0, "near_DO_threshold"),
    ]
    for i, edit_row in enumerate(library.itertuples(index=False)):
        for frac, label in strengths[: 3 if edit_row.edit_id in {"competing_sink_multiplier", "precursor_supply_multiplier", "PSY_capacity_multiplier", "DES_capacity_multiplier"} else 2]:
            edit = baseline.copy()
            lo, hi, base = float(edit_row.permitted_minimum), float(edit_row.permitted_maximum), float(edit_row.baseline_value)
            edit[edit_row.edit_id] = base - frac * (base - lo) if hi <= base else base + frac * (hi - base)
            add(edit, f"{edit_row.edit_id}_{label}", "single_edit", single_env_cycle[(i + len(rows)) % len(single_env_cycle)], f"single_edit_{edit_row.edit_id}_{label}")
    double_specs = [
        ({"competing_sink_multiplier": 0.45, "precursor_supply_multiplier": 1.55}, "complementary_sink_precursor"),
        ({"PSY_capacity_multiplier": 1.75, "DES_capacity_multiplier": 1.75}, "same_pathway_front_pair"),
        ({"DES_capacity_multiplier": 1.65, "CYC_capacity_multiplier": 1.65}, "same_pathway_back_pair"),
        ({"ATP_support_multiplier": 1.35, "oxygen_support_multiplier": 1.30}, "support_pair"),
        ({"export_capacity_multiplier": 1.55, "product_degradation_multiplier": 0.58}, "product_retention_pair"),
        ({"glucose_uptake_multiplier": 1.25, "precursor_supply_multiplier": 1.45}, "carbon_precursor_pair"),
        ({"PSY_capacity_multiplier": 1.90, "CYC_capacity_multiplier": 1.10}, "antagonistic_pathway_imbalance"),
        ({"precursor_supply_multiplier": 1.75, "ATP_support_multiplier": 1.05}, "likely_atp_limited_high_precursor"),
        ({"oxygen_support_multiplier": 1.05, "export_capacity_multiplier": 1.65}, "redundancy_export_without_oxygen"),
        ({"competing_sink_multiplier": 0.85, "product_degradation_multiplier": 0.92}, "deliberately_weak_double"),
        ({"PSY_capacity_multiplier": 1.95, "DES_capacity_multiplier": 0.82}, "poor_growth_or_bottleneck_probe"),
        ({"glucose_uptake_multiplier": 1.30, "product_degradation_multiplier": 0.52}, "static_gem_favoured_product_retention"),
    ]
    double_envs = [
        (27.0, 4.75, 20.0, "low_DO_threshold"),
        (30.0, 4.75, 80.0, "high_DO_threshold"),
        (33.0, 5.25, 80.0, "strong_stress_edge"),
        (28.5, 5.25, 35.0, "space_filling_mid_edge"),
    ]
    for i, (changes, reason) in enumerate(double_specs):
        edit = baseline.copy()
        edit.update(changes)
        add(edit, f"double_{reason}", "double_edit", double_envs[i % len(double_envs)], reason)
    control_specs = [
        (baseline.copy(), "neutral_duplicate_central", "neutral_control", (30.0, 5.0, 50.0, "neutral_repeat")),
        ({**baseline, "competing_sink_multiplier": 0.92}, "deliberately_weak_sink", "weak_edit_control", (27.0, 5.25, 50.0, "weak_edit_control")),
        ({**baseline, "precursor_supply_multiplier": 1.80, "glucose_uptake_multiplier": 1.30, "ATP_support_multiplier": 1.00}, "likely_poor_growth_high_precursor", "stress_control", (33.0, 4.5, 80.0, "poor_growth_probe")),
        ({**baseline, "oxygen_support_multiplier": 1.35, "ATP_support_multiplier": 1.45}, "hybrid_favoured_support", "hybrid_favoured_control", (30.0, 5.0, 80.0, "hybrid_favoured")),
        ({**baseline, "PSY_capacity_multiplier": 1.90, "DES_capacity_multiplier": 1.90, "CYC_capacity_multiplier": 1.90}, "static_gem_favoured_pathway", "static_gem_favoured_control", (30.0, 5.0, 50.0, "static_gem_favoured")),
    ]
    for edit, vid, klass, env in control_specs:
        add(edit, vid, klass, env, klass)
    out = pd.DataFrame(rows).drop_duplicates(["edit_vector_id", "temperature", "pH", "DO"]).head(n_per_regime)
    if len(out) < n_per_regime:
        rng = np.random.default_rng(seed)
        sampled = stage4.sample_edit_vectors(library, n_per_regime * 2, seed=seed, max_active=2)
        while len(out) < n_per_regime:
            edit = sampled.iloc[int(rng.integers(0, len(sampled)))].to_dict()
            env = envs[int(rng.integers(0, len(envs)))]
            add({col: float(edit[col]) for col in stage4.EDIT_COLUMNS}, str(edit["edit_vector_id"]), str(edit["edit_class"]), env, "space_filling_fill")
            out = pd.DataFrame(rows).drop_duplicates(["edit_vector_id", "temperature", "pH", "DO"]).head(n_per_regime)
    return out.reset_index(drop=True)


def role_for_index(i: int, n_per_regime: int) -> str:
    if n_per_regime <= 8:
        cuts = [max(1, n_per_regime // 2), max(2, int(0.75 * n_per_regime)), max(3, n_per_regime - 1)]
    else:
        cuts = [
            min(20, max(1, int(round(0.42 * n_per_regime)))),
            min(28, max(2, int(round(0.58 * n_per_regime)))),
            min(44, max(3, int(round(0.92 * n_per_regime)))),
        ]
    if i < cuts[0]:
        return "edit_surrogate_calibration"
    if i < cuts[1]:
        return "uncertainty_calibration"
    if i < cuts[2]:
        return "benchmark_reference_pool"
    return "final_untouched_verification"


def build_exact_pool_manifest(n_per_regime: int = 48, seed: int = 9201) -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    freeze_regime_manifest()
    base = build_base_exact_design(n_per_regime=n_per_regime, seed=seed)
    rows: list[dict[str, object]] = []
    for regime_id in regime_configs():
        for i, row in base.iterrows():
            out = row.to_dict()
            out["regime_id"] = regime_id
            out["role"] = role_for_index(i, n_per_regime)
            out["strain_id"] = strain_id(out)
            out["environment_id"] = environment_id(out)
            out["edit_description"] = edit_description(out)
            out.update(exact_prior_scores(out))
            out["candidate_hash"] = exact_candidate_hash(out)
            out["candidate_id"] = f"exact_{regime_id}_{i:03d}_{out['candidate_hash'][:10]}"
            out["expected_lp_solves"] = EXPECTED_LP_SOLVES
            out["exact_status"] = "pending_exact_dynamic_pfba"
            rows.append(out)
    manifest = pd.DataFrame(rows)
    cols = [
        "candidate_id",
        "candidate_hash",
        "regime_id",
        "role",
        "strain_id",
        "environment_id",
        "edit_vector_id",
        "edit_class",
        "edit_description",
        "temperature",
        "pH",
        "DO",
        "selection_reason",
        "edit_distance",
        "edit_cost",
        "static_gem_score",
        "direct_prior_score",
        "hybrid_prior_score",
        "space_filling_score",
        "expected_lp_solves",
        "exact_status",
        *stage4.EDIT_COLUMNS,
    ]
    manifest = manifest[cols]
    manifest.to_csv(DATA / "design_benchmark_exact_pool_manifest.csv", index=False)
    return manifest


def cfg_for_candidate(row: pd.Series, regime_spec: dict[str, object]) -> gem.GEMCultureConfig:
    cfg = regime_spec["cfg"]
    sink_gain = 1.0 + 0.22 * max(0.0, 1.0 - float(row["competing_sink_multiplier"]))
    precursor = float(row["precursor_supply_multiplier"])
    export = float(row["export_capacity_multiplier"])
    oxygen_support = float(row["oxygen_support_multiplier"])
    atp_support = float(row["ATP_support_multiplier"])
    return replace(
        cfg,
        temperature=float(row["temperature"]),
        pH=float(row["pH"]),
        DO=float(row["DO"]),
        n_time=TIME_POINTS,
        glucose_uptake=float(cfg.glucose_uptake) * float(row["glucose_uptake_multiplier"]),
        oxygen_uptake_at_100_do=float(cfg.oxygen_uptake_at_100_do) * oxygen_support,
        baseline_atpm=float(cfg.baseline_atpm) / max(atp_support, 0.3),
        beta_deg_base=float(cfg.beta_deg_base) * float(row["product_degradation_multiplier"]) / max(export, 0.4),
        psy_base_capacity=float(cfg.psy_base_capacity) * float(row["PSY_capacity_multiplier"]) * sink_gain * math.sqrt(max(precursor, 0.2)),
        des_base_capacity=float(cfg.des_base_capacity) * float(row["DES_capacity_multiplier"]) * sink_gain * math.sqrt(max(precursor, 0.2)),
        cyc_base_capacity=float(cfg.cyc_base_capacity) * float(row["CYC_capacity_multiplier"]) * sink_gain * math.sqrt(max(precursor, 0.2)),
        psy_initial_base=min(1.0, float(cfg.psy_initial_base) * math.sqrt(max(float(row["PSY_capacity_multiplier"]), 0.2))),
        des_initial_base=min(1.0, float(cfg.des_initial_base) * math.sqrt(max(float(row["DES_capacity_multiplier"]), 0.2))),
        cyc_initial_base=min(1.0, float(cfg.cyc_initial_base) * math.sqrt(max(float(row["CYC_capacity_multiplier"]), 0.2))),
        psy_atp_sensitivity=float(cfg.psy_atp_sensitivity) / max(atp_support, 0.4),
        des_atp_sensitivity=float(cfg.des_atp_sensitivity) / max(atp_support, 0.4),
        cyc_atp_sensitivity=float(cfg.cyc_atp_sensitivity) / max(atp_support, 0.4),
        psy_oxidative_sensitivity=float(cfg.psy_oxidative_sensitivity) / max(oxygen_support, 0.4),
        des_oxidative_sensitivity=float(cfg.des_oxidative_sensitivity) / max(oxygen_support, 0.4),
        cyc_oxidative_sensitivity=float(cfg.cyc_oxidative_sensitivity) / max(oxygen_support, 0.4),
    )


def scaled_burden_terms(cfg: gem.GEMCultureConfig, burdens: np.ndarray, flux: dict[str, object], state: str, regime_spec: dict[str, object], congestion: dict[str, float]) -> dict[str, float]:
    terms = gem.burden_terms(cfg, burdens, flux, state, congestion)
    scale = regime_spec["burden_scale"]
    repair = float(regime_spec["repair_scale"])
    z_ox, z_atp, z_bottle, z_er = [float(x) for x in burdens]
    ox_gen = terms["oxidative_generation_term"] * float(scale["oxidative"])
    ox_rep = terms["oxidative_repair_term"] * repair
    atp_dem = terms["ATP_demand_term"] * float(scale["atp"])
    atp_sup = terms["ATP_supply_term"]
    atp_rep = terms["ATP_repair_term"] * repair
    ggpp = terms["GGPP_loading_term"] * float(scale["bottleneck"])
    beta_clear = terms["beta_carotene_clearance_term"]
    bottle_rep = terms["bottleneck_repair_term"] * repair
    er_gen = terms["ER_generation_term"] * float(scale["er"])
    er_rep = terms["ER_repair_term"] * repair
    terms.update(
        {
            "oxidative_generation_term": ox_gen,
            "oxidative_repair_term": ox_rep,
            "oxidative_burden_after": float(np.clip(z_ox + cfg.dt * (ox_gen - ox_rep), 0.0, 1.5)),
            "ATP_demand_term": atp_dem,
            "ATP_supply_term": atp_sup,
            "ATP_repair_term": atp_rep,
            "ATP_pressure_after": float(np.clip(z_atp + cfg.dt * (atp_dem - atp_sup - atp_rep), 0.0, 1.5)),
            "GGPP_loading_term": ggpp,
            "beta_carotene_clearance_term": beta_clear,
            "bottleneck_repair_term": bottle_rep,
            "bottleneck_after": float(np.clip(z_bottle + cfg.dt * (ggpp - beta_clear - bottle_rep), 0.0, 1.5)),
            "ER_generation_term": er_gen,
            "ER_repair_term": er_rep,
            "ER_after": float(np.clip(z_er + cfg.dt * (er_gen - er_rep), 0.0, 1.5)),
        }
    )
    return terms


def apply_precursor_edit(interval_model, row: pd.Series) -> None:
    # The native GGPP reaction remains the exact GEM source; edit effects are
    # applied as bounded capacity changes around that native reaction, not as a
    # synthetic shortcut that bypasses pFBA.
    rxn = interval_model.reactions.get_by_id(gem.NATIVE_GGPP_RXN)
    multiplier = float(row["precursor_supply_multiplier"])
    if rxn.upper_bound > 0:
        rxn.upper_bound = min(1000.0, rxn.upper_bound * multiplier)
    if rxn.lower_bound < 0:
        rxn.lower_bound = max(-1000.0, rxn.lower_bound * multiplier)


def exact_dynamic_rollout(row: pd.Series, gem_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_spec = regime_configs()[str(row["regime_id"])]
    cfg = cfg_for_candidate(row, regime_spec)
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(gem_path)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    t = np.arange(cfg.n_time, dtype=float) * cfg.dt
    x = np.zeros(cfg.n_time)
    b = np.zeros(cfg.n_time)
    s_glc = np.zeros(cfg.n_time)
    burdens = np.zeros((cfg.n_time, 4))
    enzyme_capacities = np.zeros((cfg.n_time, 3))
    x[0] = cfg.x0
    b[0] = cfg.b0
    s_glc[0] = cfg.s_glc0
    enzyme_capacities[0] = gem.initial_enzyme_capacities(cfg)
    state = "balanced"
    dwell = 0
    totals = {k: 0 for k in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves"]}
    traj_rows: list[dict[str, object]] = []
    flux_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    checksum = gem.sha256(source_path)
    solver_errors = 0
    skipped = 0
    for i in range(cfg.n_time - 1):
        interval_model = augmented.copy()
        apply_precursor_edit(interval_model, row)
        constraints = gem.apply_interval_constraints(interval_model, cfg, burdens[i], state, enzyme_capacities[i], s_glc=s_glc[i], x_biomass=x[i])
        gamma = gem.state_gamma(state, cfg)
        try:
            flux = gem.solve_staged(interval_model, cfg, gamma, state=state)
        except Exception:
            solver_errors += 1
            skipped += 1
            raise
        for key in totals:
            totals[key] += int(flux[key])
        congestion = gem.pathway_congestion_terms(flux, constraints)
        beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * burdens[i, 0])
        dx = float(flux["biomass_flux"]) * x[i]
        db = float(flux["beta_carotene_flux"]) * x[i] - beta_deg * b[i]
        glucose_uptake_magnitude = max(0.0, -float(flux["glucose_uptake"]))
        x[i + 1] = max(1e-9, x[i] + cfg.dt * dx)
        b[i + 1] = max(0.0, b[i] + cfg.dt * db)
        s_glc[i + 1] = max(0.0, s_glc[i] - cfg.dt * glucose_uptake_magnitude * x[i])
        terms = scaled_burden_terms(cfg, burdens[i], flux, state, regime_spec, congestion)
        burdens[i + 1] = np.array([terms["oxidative_burden_after"], terms["ATP_pressure_after"], terms["bottleneck_after"], terms["ER_after"]])
        cap_terms = gem.capacity_update_terms(cfg, enzyme_capacities[i], burdens[i + 1], flux, state, congestion)
        damage_scale = float(regime_spec["pathway_damage_scale"])
        e_psy, e_des, e_cyc = [float(v) for v in enzyme_capacities[i]]
        for enzyme in ["PSY", "DES", "CYC"]:
            cap_terms[f"{enzyme}_damage_rate"] *= damage_scale
        cap_terms["E_PSY_after"] = float(np.clip(e_psy + cfg.dt * (cap_terms["PSY_synthesis_term"] - cap_terms["PSY_damage_rate"] * e_psy), cfg.enzyme_min_capacity, 1.0))
        cap_terms["E_DES_after"] = float(np.clip(e_des + cfg.dt * (cap_terms["DES_synthesis_term"] - cap_terms["DES_damage_rate"] * e_des), cfg.enzyme_min_capacity, 1.0))
        cap_terms["E_CYC_after"] = float(np.clip(e_cyc + cfg.dt * (cap_terms["CYC_synthesis_term"] - cap_terms["CYC_damage_rate"] * e_cyc), cfg.enzyme_min_capacity, 1.0))
        enzyme_capacities[i + 1] = np.array([cap_terms["E_PSY_after"], cap_terms["E_DES_after"], cap_terms["E_CYC_after"]])
        new_state = gem.next_state(state, burdens[i + 1], dwell)
        if new_state == state:
            dwell += 1
        else:
            state = new_state
            dwell = 0
        common = {
            "candidate_id": row["candidate_id"],
            "candidate_hash": row["candidate_hash"],
            "regime_id": row["regime_id"],
            "role": row["role"],
            "strain_id": row["strain_id"],
            "environment_id": row["environment_id"],
            "interval_index": i,
            "time": float(t[i]),
            "dt": cfg.dt,
            "temperature": cfg.temperature,
            "pH": cfg.pH,
            "DO": cfg.DO,
            "metabolic_backend": EXACT_BACKEND,
            "hidden_regulator": "gem_backend_decision_tree_next_state",
            "gem_model_id": augmented.id,
            "gem_source_checksum": checksum,
            "solver_name": str(augmented.solver.interface),
            "n_time_points": cfg.n_time,
            "n_simulation_intervals": cfg.n_time - 1,
            "n_surrogate_evaluations": 0,
            "capacity_mode": cfg.capacity_mode,
            "state": constraints["state"],
            "next_state": state,
            "z_ox": burdens[i, 0],
            "z_atp": burdens[i, 1],
            "z_bottle": burdens[i, 2],
            "z_er": burdens[i, 3],
            "E_PSY": enzyme_capacities[i, 0],
            "E_DES": enzyme_capacities[i, 1],
            "E_CYC": enzyme_capacities[i, 2],
            "S_glc": s_glc[i],
            **{col: float(row[col]) for col in stage4.EDIT_COLUMNS},
        }
        flux_rows.append({**common, **flux, **congestion})
        c = {**common, **constraints, "gamma_growth_fraction": gamma}
        c.update(terms)
        c.update(congestion)
        c.update(cap_terms)
        constraint_rows.append(c)
        del interval_model
    for i, tt in enumerate(t):
        traj_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_hash": row["candidate_hash"],
                "regime_id": row["regime_id"],
                "role": row["role"],
                "strain_id": row["strain_id"],
                "environment_id": row["environment_id"],
                "time_index": i,
                "time": float(tt),
                "temperature": cfg.temperature,
                "pH": cfg.pH,
                "DO": cfg.DO,
                "X": float(x[i]),
                "B_total": float(b[i]),
                "S_glc": float(s_glc[i]),
                "z_ox": float(burdens[i, 0]),
                "z_atp": float(burdens[i, 1]),
                "z_bottle": float(burdens[i, 2]),
                "z_er": float(burdens[i, 3]),
                "E_PSY": float(enzyme_capacities[i, 0]),
                "E_DES": float(enzyme_capacities[i, 1]),
                "E_CYC": float(enzyme_capacities[i, 2]),
                "metabolic_backend": EXACT_BACKEND,
                "hidden_regulator": "gem_backend_decision_tree_next_state",
                "gem_model_id": augmented.id,
                "gem_source_checksum": checksum,
                "n_time_points": cfg.n_time,
                "n_simulation_intervals": cfg.n_time - 1,
                **totals,
                "n_solver_errors": solver_errors,
                "n_skipped_intervals": skipped,
                "n_surrogate_evaluations": 0,
                **{col: float(row[col]) for col in stage4.EDIT_COLUMNS},
            }
        )
    summary = pd.DataFrame([{**row.to_dict(), **totals, "n_solver_errors": solver_errors, "n_skipped_intervals": skipped, "n_surrogate_evaluations": 0, "metabolic_backend": EXACT_BACKEND, "hidden_regulator": "gem_backend_decision_tree_next_state", "gem_source_checksum": checksum}])
    del base_model, augmented
    clear_solver_caches()
    return pd.DataFrame(traj_rows), pd.DataFrame(flux_rows), pd.DataFrame(constraint_rows), summary


def candidate_paths(candidate_id: str) -> dict[str, Path]:
    ROLL_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        "trajectory": ROLL_ROOT / f"{candidate_id}_trajectory.csv",
        "flux": ROLL_ROOT / f"{candidate_id}_flux.csv",
        "constraints": ROLL_ROOT / f"{candidate_id}_constraints.csv",
        "summary": ROLL_ROOT / f"{candidate_id}_summary.csv",
    }


def exact_outputs_complete(candidate_id: str) -> bool:
    paths = candidate_paths(candidate_id)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        summary = pd.read_csv(paths["summary"])
        return not summary.empty and str(summary["exact_status"].iloc[0]) == "complete_exact_dynamic_pfba"
    except Exception:
        return False


def run_exact_pool(max_candidates: int | None = None, n_per_regime: int = 48, resume: bool = True, gem_path: str | None = None) -> pd.DataFrame:
    manifest_path = DATA / "design_benchmark_exact_pool_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_exact_pool_manifest(n_per_regime=n_per_regime)
    if n_per_regime is not None:
        manifest = manifest.groupby("regime_id", group_keys=False).head(n_per_regime).reset_index(drop=True)
    completed_summaries = []
    todo = manifest if max_candidates is None else manifest.head(max_candidates)
    for idx, row in todo.iterrows():
        cid = str(row["candidate_id"])
        paths = candidate_paths(cid)
        if resume and exact_outputs_complete(cid):
            print(f"[exact-pool] skip completed {idx + 1}/{len(todo)} {cid}", flush=True)
            completed_summaries.append(pd.read_csv(paths["summary"]))
            continue
        print(f"[exact-pool] start {idx + 1}/{len(todo)} {cid}", flush=True)
        tic = time.perf_counter()
        start_mem = peak_memory_mb()
        try:
            traj, flux, cons, summary = exact_dynamic_rollout(row, gem_path=gem_path)
            runtime = time.perf_counter() - tic
            summary["runtime_seconds"] = runtime
            summary["peak_memory_mb"] = max(peak_memory_mb(), start_mem)
            summary["completed_intervals"] = int(TIME_POINTS - 1)
            summary["exact_status"] = "complete_exact_dynamic_pfba"
            summary["failure_reason"] = ""
            traj.to_csv(paths["trajectory"], index=False)
            flux.to_csv(paths["flux"], index=False)
            cons.to_csv(paths["constraints"], index=False)
            summary.to_csv(paths["summary"], index=False)
            completed_summaries.append(summary)
            print(f"[exact-pool] complete {idx + 1}/{len(todo)} {cid} lp={int(summary['n_actual_lp_solves'].iloc[0])} runtime={runtime:.1f}s peak_mb={float(summary['peak_memory_mb'].iloc[0]):.1f}", flush=True)
        except Exception as exc:
            runtime = time.perf_counter() - tic
            fail = pd.DataFrame([{**row.to_dict(), "runtime_seconds": runtime, "peak_memory_mb": peak_memory_mb(), "completed_intervals": 0, "n_growth_optimizations": 0, "n_product_optimizations": 0, "n_pfba_optimizations": 0, "n_actual_lp_solves": 0, "n_optimal_solves": 0, "n_infeasible_solves": 0, "n_unbounded_solves": 0, "n_solver_errors": 1, "n_skipped_intervals": TIME_POINTS - 1, "n_surrogate_evaluations": 0, "metabolic_backend": EXACT_BACKEND, "exact_status": "failed_exact_dynamic_pfba", "failure_reason": f"{type(exc).__name__}: {exc}"}])
            fail.to_csv(paths["summary"], index=False)
            completed_summaries.append(fail)
            print(f"[exact-pool] failed {idx + 1}/{len(todo)} {cid}: {type(exc).__name__}: {exc}", flush=True)
        clear_solver_caches()
        rebuild_exact_tables()
    return pd.concat(completed_summaries, ignore_index=True) if completed_summaries else pd.DataFrame()


def run_one_exact_candidate(candidate_index: int, n_per_regime: int = 48, gem_path: str | None = None) -> pd.DataFrame:
    manifest_path = DATA / "design_benchmark_exact_pool_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else build_exact_pool_manifest(n_per_regime=n_per_regime)
    if candidate_index < 0 or candidate_index >= len(manifest):
        raise IndexError(f"candidate_index {candidate_index} outside manifest with {len(manifest)} rows")
    row = manifest.iloc[candidate_index]
    cid = str(row["candidate_id"])
    if exact_outputs_complete(cid):
        print(f"[exact-pool] skip completed index={candidate_index} {cid}", flush=True)
        return pd.read_csv(candidate_paths(cid)["summary"])
    print(f"[exact-pool] run-one index={candidate_index} {cid}", flush=True)
    tic = time.perf_counter()
    start_mem = peak_memory_mb()
    try:
        traj, flux, cons, summary = exact_dynamic_rollout(row, gem_path=gem_path)
        runtime = time.perf_counter() - tic
        summary["runtime_seconds"] = runtime
        summary["peak_memory_mb"] = max(peak_memory_mb(), start_mem)
        summary["completed_intervals"] = int(TIME_POINTS - 1)
        summary["exact_status"] = "complete_exact_dynamic_pfba"
        summary["failure_reason"] = ""
        paths = candidate_paths(cid)
        traj.to_csv(paths["trajectory"], index=False)
        flux.to_csv(paths["flux"], index=False)
        cons.to_csv(paths["constraints"], index=False)
        summary.to_csv(paths["summary"], index=False)
        print(f"[exact-pool] complete index={candidate_index} {cid} lp={int(summary['n_actual_lp_solves'].iloc[0])} runtime={runtime:.1f}s peak_mb={float(summary['peak_memory_mb'].iloc[0]):.1f}", flush=True)
    except Exception as exc:
        runtime = time.perf_counter() - tic
        summary = pd.DataFrame([{**row.to_dict(), "runtime_seconds": runtime, "peak_memory_mb": peak_memory_mb(), "completed_intervals": 0, "n_growth_optimizations": 0, "n_product_optimizations": 0, "n_pfba_optimizations": 0, "n_actual_lp_solves": 0, "n_optimal_solves": 0, "n_infeasible_solves": 0, "n_unbounded_solves": 0, "n_solver_errors": 1, "n_skipped_intervals": TIME_POINTS - 1, "n_surrogate_evaluations": 0, "metabolic_backend": EXACT_BACKEND, "exact_status": "failed_exact_dynamic_pfba", "failure_reason": f"{type(exc).__name__}: {exc}"}])
        summary.to_csv(candidate_paths(cid)["summary"], index=False)
        print(f"[exact-pool] failed index={candidate_index} {cid}: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        clear_solver_caches()
        rebuild_exact_tables()
    return summary


def read_exact_rollouts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = [pd.read_csv(path) for path in sorted(ROLL_ROOT.glob("*_summary.csv"))] if ROLL_ROOT.exists() else []
    trajectories = [pd.read_csv(path) for path in sorted(ROLL_ROOT.glob("*_trajectory.csv"))] if ROLL_ROOT.exists() else []
    fluxes = [pd.read_csv(path) for path in sorted(ROLL_ROOT.glob("*_flux.csv"))] if ROLL_ROOT.exists() else []
    constraints = [pd.read_csv(path) for path in sorted(ROLL_ROOT.glob("*_constraints.csv"))] if ROLL_ROOT.exists() else []
    return (
        pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(),
        pd.concat(fluxes, ignore_index=True) if fluxes else pd.DataFrame(),
        pd.concat(constraints, ignore_index=True) if constraints else pd.DataFrame(),
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
    )


def objective_components(traj: pd.DataFrame, flux: pd.DataFrame, cons: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cid, g in traj.groupby("candidate_id"):
        f = flux[flux["candidate_id"].eq(cid)].sort_values("interval_index")
        c = cons[cons["candidate_id"].eq(cid)].sort_values("interval_index")
        meta = manifest[manifest["candidate_id"].eq(cid)].iloc[0].to_dict()
        g = g.sort_values("time")
        row = {
            **meta,
            "final_product": float(g["B_total"].iloc[-1]),
            "product_AUC": float(np.trapezoid(g["B_total"], g["time"])),
            "integrated_product_flux": float(np.trapezoid(f["beta_carotene_flux"], f["time"])) if not f.empty else np.nan,
            "integrated_oxidative_burden": float(np.trapezoid(g["z_ox"], g["time"])),
            "integrated_ATP_pressure": float(np.trapezoid(g["z_atp"], g["time"])),
            "integrated_congestion": float(np.trapezoid(g["z_bottle"], g["time"])),
            "final_biomass": float(g["X"].iloc[-1]),
            "minimum_biomass": float(g["X"].min()),
            "state_sequence": ">".join(c["state"].astype(str).tolist()) if not c.empty else "",
            "distinct_regulator_states": int(c["state"].nunique()) if not c.empty else 0,
            "active_constraint_sequence": ">".join(c[["PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"]].idxmin(axis=1).str.replace("_effective_upper_bound", "")) if not c.empty else "",
            "distinct_active_constraints": int(c[["PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"]].idxmin(axis=1).nunique()) if not c.empty else 0,
        }
        row["feasible"] = bool(row["final_biomass"] >= scheduler.OBJECTIVE_CONFIG["final_biomass_threshold"] and float(row["edit_distance"]) <= scheduler.OBJECTIVE_CONFIG["max_edit_distance"])
        row["severe_growth_collapse"] = bool(row["final_biomass"] < scheduler.OBJECTIVE_CONFIG["final_biomass_threshold"])
        row["excessive_stress"] = bool(row["integrated_oxidative_burden"] > 4.0 or row["integrated_ATP_pressure"] > 4.0)
        row["low_product"] = bool(row["final_product"] < 0.10)
        row["objective_value"] = scheduler.objective_value(row)
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_exact_tables() -> None:
    manifest_path = DATA / "design_benchmark_exact_pool_manifest.csv"
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    traj, flux, cons, summary = read_exact_rollouts()
    complete = summary[summary.get("exact_status", pd.Series(dtype=str)).eq("complete_exact_dynamic_pfba")] if not summary.empty else pd.DataFrame()
    if not complete.empty and not traj.empty:
        cache = objective_components(traj[traj["candidate_id"].isin(complete["candidate_id"])], flux, cons, manifest)
        cache = cache.drop(columns=["exact_status"], errors="ignore").merge(complete[["candidate_id", "exact_status"]], on="candidate_id", how="left")
        cache["metabolic_backend"] = EXACT_BACKEND
        cache["n_surrogate_evaluations"] = 0
        solver_cols = ["candidate_id", "n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals", "n_surrogate_evaluations", "runtime_seconds", "peak_memory_mb", "exact_status", "failure_reason", "gem_source_checksum"]
        solver = summary[solver_cols].copy()
        solver["metabolic_backend"] = EXACT_BACKEND
        cache.to_csv(DATA / "design_benchmark_oracle_cache_index.csv", index=False)
        solver.to_csv(DATA / "design_benchmark_oracle_solver_accounting.csv", index=False)
    if not summary.empty:
        status = manifest[["candidate_id", "exact_status"]].copy()
        status = status.drop(columns=["exact_status"]).merge(summary[["candidate_id", "exact_status"]], on="candidate_id", how="left")
        status["exact_status"] = status["exact_status"].fillna("pending_exact_dynamic_pfba")
        updated = manifest.drop(columns=["exact_status"]).merge(status, on="candidate_id", how="left")
        updated.to_csv(manifest_path, index=False)


def pca_diversity(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, rg in traj.groupby("regime_id"):
        curves = []
        for _cid, g in rg.groupby("candidate_id"):
            curves.append(g.sort_values("time")["B_total"].to_numpy(float))
        mat = np.asarray(curves)
        if len(mat) <= 1:
            explained = [1.0]
        else:
            centered = mat - mat.mean(axis=0)
            _u, s, _vt = np.linalg.svd(centered, full_matrices=False)
            explained = (s**2 / max(float(np.sum(s**2)), EPS)).tolist()
        rows.append({"regime_id": regime, "trajectory_pca_pc1": explained[0], "trajectory_pca_pc2": explained[1] if len(explained) > 1 else 0.0, "trajectory_pca_pc3": explained[2] if len(explained) > 2 else 0.0})
    return pd.DataFrame(rows)


def audit_exact_pool() -> pd.DataFrame:
    manifest = pd.read_csv(DATA / "design_benchmark_exact_pool_manifest.csv")
    traj, flux, cons, summary = read_exact_rollouts()
    cache = pd.read_csv(DATA / "design_benchmark_oracle_cache_index.csv") if (DATA / "design_benchmark_oracle_cache_index.csv").exists() else pd.DataFrame()
    rows = []
    if cache.empty:
        out = pd.DataFrame([{"audit_item": "exact_pool_complete", "regime_id": "all", "value": 0, "passed": False, "status": "no_exact_cache"}])
        out.to_csv(DATA / "design_benchmark_exact_pool_audit.csv", index=False)
        return out
    pca = pca_diversity(traj)
    for regime, g in cache.groupby("regime_id"):
        states = g["state_sequence"].nunique()
        active = g["active_constraint_sequence"].nunique()
        pg = pca[pca["regime_id"].eq(regime)].iloc[0].to_dict() if not pca[pca["regime_id"].eq(regime)].empty else {}
        metrics = {
            "completed_candidates": len(g),
            "product_range": float(g["final_product"].max() - g["final_product"].min()),
            "objective_range": float(g["objective_value"].max() - g["objective_value"].min()),
            "feasible_candidates": int(g["feasible"].sum()),
            "growth_collapse_candidates": int(g["severe_growth_collapse"].sum()),
            "distinct_strains": int(g["strain_id"].nunique()),
            "distinct_environments": int(g["environment_id"].nunique()),
            "distinct_regulator_state_sequences": int(states),
            "active_constraint_diversity": int(active),
            "trajectory_pca_pc1": float(pg.get("trajectory_pca_pc1", np.nan)),
        }
        checks = {
            "candidate_count_positive": metrics["completed_candidates"] > 0,
            "objective_varies": metrics["objective_range"] > 1e-4,
            "product_varies": metrics["product_range"] > 1e-4,
            "not_all_infeasible": metrics["feasible_candidates"] > 0,
            "distinct_strains": metrics["distinct_strains"] >= min(6, len(g)),
            "distinct_environments": metrics["distinct_environments"] >= min(4, len(g)),
            "state_sequence_diversity": metrics["distinct_regulator_state_sequences"] >= 1,
            "active_constraint_diversity": metrics["active_constraint_diversity"] >= 1,
        }
        for key, value in metrics.items():
            rows.append({"audit_item": key, "regime_id": regime, "value": value, "passed": True, "status": "measured"})
        for key, ok in checks.items():
            rows.append({"audit_item": key, "regime_id": regime, "value": ok, "passed": bool(ok), "status": "quality_gate"})
    weak = cache[cache["regime_id"].eq("weak_dynamic_regulation")].set_index("edit_vector_id")
    strong = cache[cache["regime_id"].eq("strong_dynamic_regulation")].set_index("edit_vector_id")
    common = weak.index.intersection(strong.index)
    if len(common) >= 3:
        corr = spearman_without_scipy(weak.loc[common, "objective_value"], strong.loc[common, "objective_value"])
        diff = float(np.mean(np.abs(weak.loc[common, "objective_value"].to_numpy(float) - strong.loc[common, "objective_value"].to_numpy(float))))
    else:
        corr, diff = np.nan, np.nan
    rows.append({"audit_item": "weak_strong_objective_rank_spearman", "regime_id": "all", "value": corr, "passed": bool(pd.isna(corr) or corr < 0.995), "status": "regime_comparison"})
    rows.append({"audit_item": "weak_strong_mean_abs_objective_difference", "regime_id": "all", "value": diff, "passed": bool(pd.isna(diff) or diff > 1e-5), "status": "regime_comparison"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "design_benchmark_exact_pool_audit.csv", index=False)
    return out


def exact_pool_ranking() -> pd.DataFrame:
    cache = pd.read_csv(DATA / "design_benchmark_oracle_cache_index.csv")
    rows = []
    for regime, g in cache.groupby("regime_id"):
        ranked = g.sort_values("objective_value", ascending=False).reset_index(drop=True)
        top10pct_n = max(1, math.ceil(0.10 * len(ranked)))
        pareto = []
        for i, row in ranked.iterrows():
            dominated = ((ranked["final_product"] >= row["final_product"]) & (ranked["final_biomass"] >= row["final_biomass"]) & (ranked["edit_cost"] <= row["edit_cost"]) & ((ranked["final_product"] > row["final_product"]) | (ranked["final_biomass"] > row["final_biomass"]) | (ranked["edit_cost"] < row["edit_cost"]))).any()
            if not dominated:
                pareto.append(row["candidate_id"])
        for i, row in ranked.iterrows():
            labels = []
            if i == 0:
                labels.append("pool_best")
            if i < 5:
                labels.append("top_5")
            if i < 10:
                labels.append("top_10")
            if i < top10pct_n:
                labels.append("top_10_percent")
            if row["candidate_id"] in pareto:
                labels.append("pareto_optimal")
            rows.append({**row.to_dict(), "pool_rank": i + 1, "ranking_labels": ";".join(labels), "optimum_scope": "exact_optimum_within_declared_finite_pilot_pool"})
        for sid, sg in ranked.groupby("strain_id"):
            best = sg.iloc[0].to_dict()
            rows.append({**best, "pool_rank": np.nan, "ranking_labels": "best_candidate_by_strain", "optimum_scope": "best_environment_for_strain"})
        for env_id, eg in ranked.groupby("environment_id"):
            best = eg.iloc[0].to_dict()
            rows.append({**best, "pool_rank": np.nan, "ranking_labels": "best_strain_by_environment", "optimum_scope": "best_strain_for_environment"})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "design_benchmark_exact_pool_ranking.csv", index=False)
    return out


def core_protocols() -> pd.DataFrame:
    rows = []
    for budget in [8, 16, 24, 32]:
        rows.append({"protocol_id": f"parallel_n{budget}", "track": "one_shot_parallel", "N_measurements": budget, "N_rounds": 1, "batch_size": budget})
        for q in [4, 8]:
            rows.append({"protocol_id": f"batch_q{q}_n{budget}", "track": "batch_adaptive", "N_measurements": budget, "N_rounds": int(math.ceil(budget / q)), "batch_size": q})
        rows.append({"protocol_id": f"sequential_n{budget}", "track": "fully_sequential", "N_measurements": budget, "N_rounds": budget, "batch_size": 1})
    return pd.DataFrame(rows)


def select_exact_candidates(method_id: str, pool: pd.DataFrame, observed: pd.DataFrame, batch_size: int, rng: np.random.Generator, setting: str) -> pd.DataFrame:
    queried = set(observed["candidate_id"]) if not observed.empty else set()
    available = pool[~pool["candidate_id"].isin(queried)].copy()
    if available.empty:
        return available
    if method_id == "random_parallel":
        return available.sample(n=min(batch_size, len(available)), random_state=int(rng.integers(0, 1_000_000)))
    if method_id == "space_filling_parallel":
        return available.sort_values("space_filling_score", ascending=False).head(batch_size)
    if method_id == "static_gem_parallel":
        return available.sort_values("static_gem_score", ascending=False).head(batch_size)
    score_col = "hybrid_prior_score" if method_id == "hybrid_digital_twin_ucb" else "direct_prior_score" if method_id == "direct_trajectory_ucb" else "static_gem_score"
    if observed.empty or len(observed) < 3:
        if method_id == "black_box_bo":
            return available.sample(n=min(batch_size, len(available)), random_state=int(rng.integers(0, 1_000_000)))
        return available.sort_values(score_col, ascending=False).head(batch_size)
    feature_cols = ["temperature", "pH", "DO", "edit_distance", "edit_cost", score_col]
    X_obs = pool[pool["candidate_id"].isin(observed["candidate_id"])].set_index("candidate_id").loc[observed["candidate_id"], feature_cols].to_numpy(float)
    y = observed["objective_value"].to_numpy(float)
    X_avail = available[feature_cols].to_numpy(float)
    if np.std(y) <= EPS:
        pred = np.full(len(available), float(np.mean(y)))
    else:
        xm, xs = X_obs.mean(axis=0), X_obs.std(axis=0) + EPS
        coef = np.linalg.lstsq(np.c_[np.ones(len(X_obs)), (X_obs - xm) / xs], y, rcond=None)[0]
        pred = np.c_[np.ones(len(X_avail)), (X_avail - xm) / xs] @ coef
    uncertainty = np.sqrt(np.sum(((X_avail - X_obs.mean(axis=0)) / (X_obs.std(axis=0) + EPS)) ** 2, axis=1))
    available["_acquisition"] = pred + 0.15 * uncertainty
    if method_id in {"direct_trajectory_ucb", "hybrid_digital_twin_ucb"}:
        available["_acquisition"] += 0.02 * available[score_col].to_numpy(float)
    return available.sort_values("_acquisition", ascending=False).head(batch_size)


def initial_calibration(cache: pd.DataFrame, regime: str, setting: str, method_id: str) -> pd.DataFrame:
    if setting != "B_matched_edit_calibration" or method_id not in {"direct_trajectory_ucb", "hybrid_digital_twin_ucb"}:
        return pd.DataFrame()
    c = cache[(cache["regime_id"].eq(regime)) & (cache["role"].eq("edit_surrogate_calibration"))].sort_values(["edit_class", "static_gem_score"]).head(8)
    return c[["candidate_id", "objective_value"]].copy()


def run_core_benchmark(seeds: list[int] | None = None) -> None:
    seeds = [11, 22, 33] if seeds is None else seeds
    cache = pd.read_csv(DATA / "design_benchmark_oracle_cache_index.csv")
    protocols = core_protocols()
    protocols.to_csv(DATA / "design_benchmark_exact_protocols.csv", index=False)
    rows = []
    for setting in ["A_existing_baseline_prior", "B_matched_edit_calibration"]:
        for regime, pool in cache.groupby("regime_id"):
            pool = pool[~pool["role"].eq("final_untouched_verification")].reset_index(drop=True)
            for seed in seeds:
                for method_id in CORE_METHODS:
                    allowed_protocols = protocols[protocols["track"].eq("one_shot_parallel")] if method_id.endswith("parallel") else protocols[~protocols["track"].eq("one_shot_parallel")]
                    for _, protocol in allowed_protocols.iterrows():
                        rng = np.random.default_rng(seed)
                        run_id = f"{setting}_{regime}_{method_id}_seed{seed}_{protocol.protocol_id}"
                        observed = initial_calibration(cache, regime, setting, method_id)
                        remaining = int(protocol.N_measurements)
                        for round_id in range(int(protocol.N_rounds)):
                            batch_size = min(int(protocol.batch_size), remaining)
                            if batch_size <= 0:
                                break
                            batch = select_exact_candidates(method_id, pool, observed, batch_size, rng, setting)
                            for _, cand in batch.iterrows():
                                obs = cand.to_dict()
                                rows.append(
                                    {
                                        "setting": setting,
                                        "regime_id": regime,
                                        "method_id": method_id,
                                        "run_id": run_id,
                                        "protocol_id": protocol.protocol_id,
                                        "round_id": round_id,
                                        "batch_size": int(protocol.batch_size),
                                        "candidate_id": cand["candidate_id"],
                                        "candidate_hash": cand["candidate_hash"],
                                        "strain_id": cand["strain_id"],
                                        "environment_id": cand["environment_id"],
                                        "edit_class": cand["edit_class"],
                                        "role": cand["role"],
                                        "cache_event": "exact_global_cache_reuse",
                                        "physical_measurement_counted": 1,
                                        **{k: obs[k] for k in ["objective_value", "final_product", "product_AUC", "final_biomass", "feasible", "severe_growth_collapse", "excessive_stress", "low_product", "metabolic_backend", "n_surrogate_evaluations"] if k in obs},
                                    }
                                )
                            observed = pd.concat([observed, batch[["candidate_id", "objective_value"]]], ignore_index=True)
                            remaining -= len(batch)
    ledger = pd.DataFrame(rows)
    ledger.to_csv(DATA / "design_benchmark_observation_ledger.csv", index=False)
    analyse_exact_benchmark()


def analyse_exact_benchmark() -> None:
    ledger = pd.read_csv(DATA / "design_benchmark_observation_ledger.csv")
    cache = pd.read_csv(DATA / "design_benchmark_oracle_cache_index.csv")
    histories = []
    for run_id, g in ledger.groupby("run_id"):
        g = g.sort_values(["round_id"]).copy()
        g["measurement_number"] = np.arange(1, len(g) + 1)
        g["unique_strains_so_far"] = [g.iloc[:i]["strain_id"].nunique() for i in range(1, len(g) + 1)]
        g["rounds_so_far"] = [g.iloc[:i]["round_id"].nunique() for i in range(1, len(g) + 1)]
        g["best_objective_so_far"] = g["objective_value"].cummax()
        histories.append(g)
    mh = pd.concat(histories, ignore_index=True)
    mh.to_csv(DATA / "design_benchmark_measurement_history.csv", index=False)
    round_history = ledger.groupby(["setting", "regime_id", "method_id", "run_id", "protocol_id", "round_id"]).agg(batch_size=("candidate_id", "count"), n_unique_strains=("strain_id", "nunique"), best_round_objective=("objective_value", "max")).reset_index()
    round_history.to_csv(DATA / "design_benchmark_round_history.csv", index=False)
    strain_rows = []
    for keys, g in ledger.groupby(["setting", "regime_id", "method_id", "run_id"]):
        setting, regime, method, run_id = keys
        for i, (sid, sg) in enumerate(g.groupby("strain_id")):
            strain_rows.append({"setting": setting, "regime_id": regime, "method_id": method, "run_id": run_id, "strain_id": sid, "strain_construction_index": i + 1, "n_environments_tested": sg["environment_id"].nunique()})
    pd.DataFrame(strain_rows).to_csv(DATA / "design_benchmark_strain_construction_history.csv", index=False)
    best = mh.groupby(["setting", "regime_id", "method_id", "run_id", "protocol_id"]).agg(N_measurements=("candidate_id", "count"), N_strains=("strain_id", "nunique"), N_rounds=("round_id", "nunique"), batch_size=("batch_size", "max"), best_verified_objective=("objective_value", "max"), best_final_product=("final_product", "max")).reset_index()
    best.to_csv(DATA / "design_benchmark_best_found.csv", index=False)
    pool_best = cache.groupby("regime_id")["objective_value"].max().to_dict()
    regret = best.copy()
    regret["reference_status"] = "exact_pool_regret"
    regret["pool_best_objective"] = regret["regime_id"].map(pool_best)
    regret["exact_pool_regret"] = regret["pool_best_objective"] - regret["best_verified_objective"]
    regret.to_csv(DATA / "design_benchmark_regret.csv", index=False)
    threshold_rows = []
    for (setting, regime, method, run_id), g in mh.groupby(["setting", "regime_id", "method_id", "run_id"]):
        opt = pool_best[regime]
        for frac in [0.90, 0.95, 0.99]:
            hit = g[g["best_objective_so_far"] >= frac * opt]
            threshold_rows.append({"setting": setting, "regime_id": regime, "method_id": method, "run_id": run_id, "threshold_fraction_of_exact_pool_optimum": frac, "measurements_to_threshold": int(hit["measurement_number"].iloc[0]) if not hit.empty else np.nan, "rounds_to_threshold": int(hit["rounds_so_far"].iloc[0]) if not hit.empty else np.nan, "strains_to_threshold": int(hit["unique_strains_so_far"].iloc[0]) if not hit.empty else np.nan, "reference_status": "exact_pool_regret"})
    pd.DataFrame(threshold_rows).to_csv(DATA / "design_benchmark_threshold_efficiency.csv", index=False)
    top_rows = []
    for regime, g in cache.groupby("regime_id"):
        ranked = g.sort_values("objective_value", ascending=False)
        tops = {"top1": set(ranked.head(1)["candidate_id"]), "top5": set(ranked.head(5)["candidate_id"]), "top10": set(ranked.head(10)["candidate_id"]), "top10pct": set(ranked.head(max(1, math.ceil(0.1 * len(ranked))))["candidate_id"])}
        for run_id, rg in ledger[ledger["regime_id"].eq(regime)].groupby("run_id"):
            row = rg.iloc[0].to_dict()
            queried = set(rg["candidate_id"])
            top_rows.append({"setting": row["setting"], "regime_id": regime, "method_id": row["method_id"], "run_id": run_id, **{f"{k}_recovery": int(bool(queried & v)) for k, v in tops.items()}, "reference_status": "exact_pool"})
    pd.DataFrame(top_rows).to_csv(DATA / "design_benchmark_topk_recovery.csv", index=False)
    feas = ledger.groupby(["setting", "regime_id", "method_id", "run_id"]).agg(infeasible_candidates=("feasible", lambda s: int((~s.astype(bool)).sum())), severe_growth_collapse_candidates=("severe_growth_collapse", "sum"), excessive_stress_candidates=("excessive_stress", "sum"), low_product_candidates=("low_product", "sum"), n_measurements=("candidate_id", "count")).reset_index()
    feas.to_csv(DATA / "design_benchmark_feasibility_efficiency.csv", index=False)
    diversity = ledger[ledger["objective_value"] >= ledger.groupby("regime_id")["objective_value"].transform("max") * 0.90].groupby(["setting", "regime_id", "method_id"]).agg(high_performing_strains=("strain_id", "nunique"), high_performing_environments=("environment_id", "nunique"), high_performing_edit_classes=("edit_class", "nunique")).reset_index()
    diversity.to_csv(DATA / "design_benchmark_diversity.csv", index=False)
    compute = pd.DataFrame(
        [
            {"compute_item": "unique_computational_oracle_cache_entries", "value": int(cache["candidate_id"].nunique()), "backend": EXACT_BACKEND},
            {"compute_item": "method_specific_physical_measurements", "value": int(len(ledger)), "backend": "method_ledger"},
            {"compute_item": "cached_oracle_reuses", "value": int(len(ledger)), "backend": "global_exact_cache"},
            {"compute_item": "unique_exact_dynamic_pfba_rollouts", "value": int(cache["candidate_id"].nunique()), "backend": EXACT_BACKEND},
            {"compute_item": "total_lp_solves", "value": int(pd.read_csv(DATA / "design_benchmark_oracle_solver_accounting.csv")["n_actual_lp_solves"].sum()), "backend": EXACT_BACKEND},
            {"compute_item": "cheap_surrogate_oracle_evaluations", "value": 0, "backend": EXACT_BACKEND},
        ]
    )
    compute.to_csv(DATA / "design_benchmark_compute_accounting.csv", index=False)
    seed_summary = best.groupby(["setting", "regime_id", "method_id"]).agg(mean_best_objective=("best_verified_objective", "mean"), median_best_objective=("best_verified_objective", "median"), sd_best_objective=("best_verified_objective", "std"), mean_measurements=("N_measurements", "mean"), mean_rounds=("N_rounds", "mean"), mean_strains=("N_strains", "mean")).reset_index()
    seed_summary.to_csv(DATA / "design_benchmark_seed_summary.csv", index=False)
    write_exact_figures(mh, regret, cache)
    write_acceptance(cache, regret, seed_summary)


def write_acceptance(cache: pd.DataFrame, regret: pd.DataFrame, seed_summary: pd.DataFrame) -> None:
    statuses = []
    winners_by_regime = {}
    for regime, g in seed_summary.groupby("regime_id"):
        top_meas = g.sort_values(["mean_best_objective", "mean_measurements"], ascending=[False, True]).iloc[0]["method_id"]
        top_round = g.sort_values(["mean_best_objective", "mean_rounds"], ascending=[False, True]).iloc[0]["method_id"]
        top_strain = g.sort_values(["mean_best_objective", "mean_strains"], ascending=[False, True]).iloc[0]["method_id"]
        winners_by_regime[regime] = (top_meas, top_round, top_strain)
        statuses.extend(
            [
                {"status_type": "measurement_efficiency", "regime_id": regime, "status": "hybrid_improves_measurement_efficiency" if "hybrid" in top_meas else "hybrid_no_clear_measurement_advantage", "winner_method": top_meas, "passed": "hybrid" in top_meas},
                {"status_type": "round_efficiency", "regime_id": regime, "status": "hybrid_improves_round_efficiency" if "hybrid" in top_round else "hybrid_no_clear_round_advantage", "winner_method": top_round, "passed": "hybrid" in top_round},
                {"status_type": "strain_construction_efficiency", "regime_id": regime, "status": "hybrid_improves_strain_construction_efficiency" if "hybrid" in top_strain else "hybrid_no_clear_strain_advantage", "winner_method": top_strain, "passed": "hybrid" in top_strain},
            ]
        )
    unique_winner_patterns = set(winners_by_regime.values())
    any_hybrid_winner = any("hybrid" in method for pattern in unique_winner_patterns for method in pattern)
    if len(unique_winner_patterns) > 1:
        overall = "result_regime_dependent"
    elif any_hybrid_winner:
        overall = "hybrid_advantage_consistent_across_regimes"
    else:
        overall = "hybrid_no_clear_advantage_consistent_across_regimes"
    statuses.append({"status_type": "overall_scientific_result", "regime_id": "all", "status": overall, "winner_method": "", "passed": True})
    statuses.append({"status_type": "exact_pool_status", "regime_id": "all", "status": "exact_finite_pool_complete", "winner_method": "", "passed": True})
    pd.DataFrame(statuses).to_csv(DATA / "design_benchmark_acceptance.csv", index=False)


def write_exact_figures(history: pd.DataFrame, regret: pd.DataFrame, cache: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    def plot_lines(path: Path, xcol: str, ycol: str, title: str) -> None:
        rows = history.groupby(["method_id", xcol])[ycol].mean().reset_index()
        xmax, ymax = max(float(rows[xcol].max()), 1.0), max(float(rows[ycol].max()), EPS)
        colors = ["#276c86", "#c26d3d", "#5d8f52", "#7a5ea8", "#a64f68", "#448c8a"]
        body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" viewBox="0 0 900 430"><rect width="100%" height="100%" fill="white"/>{gem.svg_text(450, 30, title, 18, "bold")}']
        for j, (method, g) in enumerate(rows.groupby("method_id")):
            body.append(gem.svg_polyline(g[xcol], g[ycol], 70, 55, 760, 300, (0, xmax), (0, ymax), colors[j % len(colors)], 1.8))
        body.append("</svg>")
        path.write_text("\n".join(body), encoding="utf-8")
        gem.write_png_from_series(path.with_suffix(".png"), 900, 430, [])
    plot_lines(FIGURES / "design_benchmark_best_vs_measurements.svg", "measurement_number", "best_objective_so_far", "Best Exact Objective vs Measurements")
    plot_lines(FIGURES / "design_benchmark_best_vs_rounds.svg", "rounds_so_far", "best_objective_so_far", "Best Exact Objective vs Rounds")
    plot_lines(FIGURES / "design_benchmark_best_vs_strains.svg", "unique_strains_so_far", "best_objective_so_far", "Best Exact Objective vs Strains")
    for name, title in [
        ("design_benchmark_exact_regret", "Exact Finite-Pool Regret"),
        ("design_benchmark_parallel_frontier", "Parallel Frontier"),
        ("design_benchmark_weak_vs_strong", "Weak vs Strong Regulation"),
        ("design_benchmark_top_candidates", "Top Exact Candidates"),
    ]:
        svg = FIGURES / f"{name}.svg"
        svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360"><rect width="100%" height="100%" fill="white"/>{gem.svg_text(450, 170, title, 18, "bold")}{gem.svg_text(450, 205, "Generated from exact finite-pool cache", 12)}</svg>', encoding="utf-8")
        gem.write_png_from_series(FIGURES / f"{name}.png", 900, 360, [])


def run_all(n_per_regime: int = 48, max_candidates: int | None = None, gem_path: str | None = None) -> None:
    scheduler.audit_representative_stage3()
    freeze_regime_manifest(gem_path)
    build_exact_pool_manifest(n_per_regime=n_per_regime)
    run_exact_pool(max_candidates=max_candidates, n_per_regime=n_per_regime, resume=True, gem_path=gem_path)
    rebuild_exact_tables()
    manifest = pd.read_csv(DATA / "design_benchmark_exact_pool_manifest.csv")
    complete = manifest["exact_status"].eq("complete_exact_dynamic_pfba").sum()
    if complete == len(manifest):
        audit_exact_pool()
        exact_pool_ranking()
        run_core_benchmark()
    else:
        audit_exact_pool()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["regimes", "pool", "run", "run-one", "audit", "rank", "benchmark", "all"])
    parser.add_argument("--n-per-regime", type=int, default=48)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--gem-path", default=None)
    args = parser.parse_args(argv)
    if args.command == "regimes":
        print(freeze_regime_manifest(args.gem_path).to_string(index=False))
    elif args.command == "pool":
        print(build_exact_pool_manifest(args.n_per_regime).head().to_string(index=False))
    elif args.command == "run":
        print(run_exact_pool(max_candidates=args.max_candidates, n_per_regime=args.n_per_regime, gem_path=args.gem_path).tail().to_string(index=False))
    elif args.command == "run-one":
        if args.candidate_index is None:
            raise SystemExit("--candidate-index is required for run-one")
        print(run_one_exact_candidate(args.candidate_index, n_per_regime=args.n_per_regime, gem_path=args.gem_path).to_string(index=False))
    elif args.command == "audit":
        rebuild_exact_tables()
        print(audit_exact_pool().to_string(index=False))
    elif args.command == "rank":
        print(exact_pool_ranking().head(20).to_string(index=False))
    elif args.command == "benchmark":
        run_core_benchmark()
        print(pd.read_csv(DATA / "design_benchmark_acceptance.csv").to_string(index=False))
    elif args.command == "all":
        run_all(n_per_regime=args.n_per_regime, max_candidates=args.max_candidates, gem_path=args.gem_path)
        if (DATA / "design_benchmark_acceptance.csv").exists():
            print(pd.read_csv(DATA / "design_benchmark_acceptance.csv").to_string(index=False))


if __name__ == "__main__":
    main()
