#!/usr/bin/env python3
"""Outcome-blind baseline selectors for the open-ended deployment benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import deployment_benchmark as deployment


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ENV_GRID = [(t, ph, do) for t in [29.0, 30.0, 31.0] for ph in [4.8, 5.0, 5.2] for do in [45.0, 55.0, 65.0, 75.0]]
FORBIDDEN_OUTCOME_COLUMNS = {
    "objective_value",
    "final_product",
    "product_AUC",
    "integrated_product_flux",
    "final_biomass",
    "minimum_biomass",
    "feasible",
    "severe_growth_collapse",
    "exact_status",
    "hidden_regime_id",
}


def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _candidate_hash(candidate: dict[str, Any]) -> str:
    return deployment.deterministic_candidate_hash(candidate)


def _risk_weights(universe: pd.DataFrame) -> np.ndarray:
    weights = universe["risk_level"].map({"low": 1.0, "medium": 1.15, "high": 0.65}).fillna(1.0).to_numpy(float)
    return weights / weights.sum()


def _edit_for_row(row: pd.Series, rng: np.random.Generator) -> dict[str, Any]:
    rid = str(row["reaction_id"])
    text = f"{row.get('subsystem', '')} {row.get('inclusion_rule', '')} {row.get('rationale_for_inclusion', '')}".lower()
    if "ggpp" in text or "mevalonate" in text or "terpenoid" in text:
        edit_type = "precursor_supply_enhancement"
    elif "transport" in text or "exchange" in text:
        edit_type = "transport_capacity_change"
    elif "atp" in text or "respiration" in text:
        edit_type = "atp_supply_edit"
    elif "nad" in text or "redox" in text:
        edit_type = "cofactor_regeneration_edit"
    elif "byproduct" in text or "ethanol" in text or "glycerol" in text:
        edit_type = "byproduct_pathway_suppression"
    else:
        edit_type = str(rng.choice(["reaction_knockdown", "reaction_capacity_increase"]))
    if edit_type in {"reaction_knockdown", "byproduct_pathway_suppression"}:
        multiplier = float(rng.uniform(0.35, 0.85))
    else:
        multiplier = float(rng.uniform(1.08, 1.75))
    return {
        "reaction_id": rid,
        "edit_type": edit_type,
        "capacity_multiplier": round(multiplier, 4),
        "rationale": f"Outcome-blind baseline edit on {row.get('subsystem', 'public subsystem')}",
    }


def candidate_to_row(candidate: dict[str, Any], method_id: str, rank: int, score: float | None = None) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "rank": rank,
        "candidate_id": candidate["candidate_id"],
        "strain_id": candidate["strain_id"],
        "candidate_hash": _candidate_hash(candidate),
        "edit_tier": candidate["edit_tier"],
        "edits": stable_json(candidate["edits"]),
        "environment": stable_json(candidate["environment"]),
        "requested_assay_panel": candidate["requested_assay_panel"],
        "proposal_score": score,
        "candidate_json": stable_json(candidate),
        "uses_hidden_outcomes": False,
    }


def sample_valid_candidates(
    universe: pd.DataFrame,
    batch_size: int,
    seed: int,
    method_id: str,
    max_attempts: int = 50_000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    weights = _risk_weights(universe)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    old_static_flag = deployment.PUBLIC_STATIC_SOLVE_ENABLED
    deployment.PUBLIC_STATIC_SOLVE_ENABLED = False
    try:
        attempts = 0
        while len(rows) < batch_size and attempts < max_attempts:
            attempts += 1
            tier = 1 if rng.random() < 0.7 else 2
            edit_count = int(rng.choice([1, 2] if tier == 1 else [2, 3, 4]))
            picks = universe.iloc[rng.choice(len(universe), size=edit_count, replace=False, p=weights)]
            temp, ph, do = ENV_GRID[int(rng.integers(0, len(ENV_GRID)))]
            candidate = {
                "candidate_id": f"{method_id}_candidate_{len(rows) + 1:03d}",
                "strain_id": f"{method_id}_strain_pending",
                "edit_tier": tier,
                "edits": [_edit_for_row(row, rng) for _, row in picks.iterrows()],
                "environment": {"temperature": temp, "pH": ph, "DO": do},
                "requested_assay_panel": "basic",
                "hypothesis": f"{method_id} outcome-blind proposal.",
                "expected_failure_mode": "Growth, redox, ATP, pathway-burden, or construction limitation.",
            }
            edit_digest = hashlib.sha256(stable_json(candidate["edits"]).encode("utf-8")).hexdigest()[:12]
            candidate["strain_id"] = f"{method_id}_strain_{edit_digest}"
            h = _candidate_hash(candidate)
            if h in seen:
                continue
            val = deployment.validate_sparse_design(candidate, universe)
            if not val["valid"]:
                continue
            seen.add(h)
            rows.append(candidate)
    finally:
        deployment.PUBLIC_STATIC_SOLVE_ENABLED = old_static_flag
    if len(rows) < batch_size:
        raise RuntimeError(f"Only sampled {len(rows)} valid candidates for {method_id}.")
    return rows


def random_valid_baseline(universe: pd.DataFrame, batch_size: int = 8, seed: int = 9001) -> pd.DataFrame:
    candidates = sample_valid_candidates(universe, batch_size, seed, "random_valid")
    return pd.DataFrame([candidate_to_row(c, "random_valid", i + 1) for i, c in enumerate(candidates)])


def candidate_features(candidate: dict[str, Any], universe: pd.DataFrame) -> np.ndarray:
    u = universe.set_index("reaction_id")
    env = candidate["environment"]
    env_vec = np.array(
        [
            (float(env["temperature"]) - 29.0) / 2.0,
            (float(env["pH"]) - 4.8) / 0.4,
            (float(env["DO"]) - 45.0) / 30.0,
        ],
        dtype=float,
    )
    edit_vec = []
    for edit in candidate["edits"]:
        rid = str(edit["reaction_id"])
        info = u.loc[rid] if rid in u.index else pd.Series(dtype=object)
        edit_vec.extend(
            [
                (float(edit.get("capacity_multiplier", 1.0)) - 1.0) / 0.75,
                float(info.get("graph_distance_to_product_pathway", 3.0)) / 3.0,
                {"low": 0.0, "medium": 0.5, "high": 1.0}.get(str(info.get("risk_level", "medium")), 0.5),
            ]
        )
    while len(edit_vec) < 12:
        edit_vec.append(0.0)
    return np.concatenate([env_vec, np.array(edit_vec[:12], dtype=float)])


def space_filling_baseline(universe: pd.DataFrame, batch_size: int = 8, seed: int = 9002, candidate_pool_size: int = 256) -> pd.DataFrame:
    pool = sample_valid_candidates(universe, candidate_pool_size, seed, "space_filling_pool")
    features = np.vstack([candidate_features(c, universe) for c in pool])
    center = features.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    remaining = set(range(len(pool))) - {first}
    while len(selected) < batch_size:
        best_idx = max(
            remaining,
            key=lambda idx: min(float(np.linalg.norm(features[idx] - features[j])) for j in selected),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
    rows = []
    for rank, idx in enumerate(selected, start=1):
        candidate = dict(pool[idx])
        candidate["candidate_id"] = f"space_filling_candidate_{rank:03d}"
        row_score = min(float(np.linalg.norm(features[idx] - features[j])) for j in selected if j != idx) if len(selected) > 1 else math.nan
        rows.append(candidate_to_row(candidate, "space_filling_maximin", rank, row_score))
    return pd.DataFrame(rows)


def static_gem_baseline(universe: pd.DataFrame, batch_size: int = 8, seed: int = 9003, candidate_pool_size: int = 256) -> pd.DataFrame:
    pool = sample_valid_candidates(universe, candidate_pool_size, seed, "static_gem_pool")
    u = universe.set_index("reaction_id")
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in pool:
        env = candidate["environment"]
        env_score = np.exp(-((float(env["temperature"]) - 30.0) / 2.5) ** 2) * np.exp(-((float(env["pH"]) - 5.0) / 0.4) ** 2)
        edit_score = 0.0
        risk_penalty = 0.0
        for edit in candidate["edits"]:
            rid = str(edit["reaction_id"])
            info = u.loc[rid] if rid in u.index else pd.Series(dtype=object)
            span = max(float(info.get("fva_max", 0.0)) - float(info.get("fva_min", 0.0)), 0.0)
            flux = abs(float(info.get("baseline_flux", 0.0))) if pd.notna(info.get("baseline_flux", np.nan)) else 0.0
            text = f"{info.get('subsystem', '')} {info.get('inclusion_rule', '')}".lower()
            pathway_bonus = 0.35 if any(term in text for term in ["terpenoid", "ggpp", "mevalonate", "isoprenoid"]) else 0.0
            edit_score += pathway_bonus + 0.03 * np.log1p(span) + 0.01 * np.log1p(flux)
            risk_penalty += {"low": 0.0, "medium": 0.025, "high": 0.07}.get(str(info.get("risk_level", "medium")), 0.03)
        score = float(env_score + edit_score - risk_penalty - 0.02 * max(len(candidate["edits"]) - 2, 0))
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    rows = []
    seen_specs: set[str] = set()
    for score, candidate in scored:
        if len(rows) >= batch_size:
            break
        spec = stable_json({"edits": candidate["edits"], "environment": candidate["environment"]})
        if spec in seen_specs:
            continue
        seen_specs.add(spec)
        candidate = dict(candidate)
        candidate["candidate_id"] = f"static_gem_candidate_{len(rows) + 1:03d}"
        rows.append(candidate_to_row(candidate, "static_gem_public_rank", len(rows) + 1, score))
    return pd.DataFrame(rows)


def assert_no_hidden_inputs(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"Baseline proposal table contains hidden/outcome columns: {leaked}")
    if "uses_hidden_outcomes" in frame.columns and frame["uses_hidden_outcomes"].astype(bool).any():
        raise ValueError("Baseline marked as using hidden outcomes.")


def write_all(batch_size: int = 8) -> None:
    universe = pd.read_csv(DATA / "deployment_benchmark_editable_reaction_universe.csv")
    outputs = {
        "open_ended_baseline_random_valid_proposals.csv": random_valid_baseline(universe, batch_size=batch_size),
        "open_ended_baseline_space_filling_proposals.csv": space_filling_baseline(universe, batch_size=batch_size),
        "open_ended_baseline_static_gem_proposals.csv": static_gem_baseline(universe, batch_size=batch_size),
    }
    for name, frame in outputs.items():
        assert_no_hidden_inputs(frame)
        frame.to_csv(DATA / name, index=False)
    pd.DataFrame(
        [
            {
                "baseline": "random_valid",
                "proposal_logic": "uniform/random valid sparse candidate sampling from the public edit universe and public environment grid",
                "distance_or_score": "none",
                "hidden_outcomes_used": False,
            },
            {
                "baseline": "space_filling_maximin",
                "proposal_logic": "sample public valid candidates, then greedily maximize minimum Euclidean distance",
                "distance_or_score": "environment, edit magnitude, graph distance to pathway, and public risk representation",
                "hidden_outcomes_used": False,
            },
            {
                "baseline": "static_gem_public_rank",
                "proposal_logic": "rank sampled valid candidates by public FVA span, pFBA baseline flux, public subsystem/pathway annotations, environment prior, and public construction risk",
                "distance_or_score": "public static-GEM heuristic score",
                "hidden_outcomes_used": False,
            },
        ]
    ).to_csv(DATA / "open_ended_baseline_selector_documentation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    write_all(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
