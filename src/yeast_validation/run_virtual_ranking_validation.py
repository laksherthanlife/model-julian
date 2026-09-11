#!/usr/bin/env python3
"""Step 3/4 of the virtual-scorer repair: generate a candidate pool, score it
with the repaired-model virtual scorer (cheap, no LP), select a stratified
subset for exact Yeast9 verification, and write everything Vanda's existing
`run_prospective_exact_worker.py` needs to run that subset.

Exact evaluation itself is NOT run here -- see hpc_bundles/virtual_ranking_validation/
for the candidate JSONs and the PBS array script; results are pulled back
and scored by scripts/analyze_virtual_ranking_validation.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HPC_DIR = ROOT / "hpc_bundles" / "virtual_ranking_validation"
WORLD_ID = "baseline_world"
CHECKPOINT = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11.npz"
N_POOL = 200
N_PER_STRATUM = 6
SEED = 4021


def generate_pool(n: int, seed: int) -> list[dict]:
    library = bench.load_library()
    rng = np.random.default_rng(seed)
    factory = bench.CandidateFactory(library, "virtual_ranking_validation", "repaired_hybrid", rng)
    return [factory.sample_candidate(max_edits=3) for _ in range(n)]


def score_pool(candidates: list[dict]) -> pd.DataFrame:
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    scorer = scorer_mod.RepairedVirtualScorer.load(CHECKPOINT, env_mean, env_std)
    rows = [scorer.score(c) for c in candidates]
    df = pd.DataFrame(rows)
    df["candidate_json"] = [json.dumps(c) for c in candidates]
    return df


def select_stratified(scored: pd.DataFrame, n_per_stratum: int, seed: int) -> pd.DataFrame:
    ranked = scored.sort_values("virtual_score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    top = ranked.iloc[:n_per_stratum].copy()
    top["stratum"] = "top"
    mid_start = n // 2 - n_per_stratum // 2
    mid = ranked.iloc[mid_start : mid_start + n_per_stratum].copy()
    mid["stratum"] = "middle"
    bottom = ranked.iloc[-n_per_stratum:].copy()
    bottom["stratum"] = "bottom"
    remaining = ranked.drop(index=list(top.index) + list(mid.index) + list(bottom.index))
    random_sel = remaining.sample(n=min(n_per_stratum, len(remaining)), random_state=seed).copy()
    random_sel["stratum"] = "random"
    return pd.concat([top, mid, bottom, random_sel], ignore_index=True)


def main() -> None:
    HPC_DIR.mkdir(parents=True, exist_ok=True)
    (HPC_DIR / "candidates").mkdir(exist_ok=True)
    (HPC_DIR / "results").mkdir(exist_ok=True)

    candidates = generate_pool(N_POOL, SEED)
    scored = score_pool(candidates)
    scored.drop(columns=["candidate_json"]).to_csv(DATA / "virtual_ranking_validation_pool_scores.csv", index=False)

    selected = select_stratified(scored, N_PER_STRATUM, SEED)
    manifest_rows = []
    for i, row in selected.iterrows():
        candidate = json.loads(row["candidate_json"])
        candidate_path = HPC_DIR / "candidates" / f"{i:03d}_{candidate['candidate_id']}.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        manifest_rows.append(
            {
                "index": i,
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "stratum": row["stratum"],
                "virtual_score": row["virtual_score"],
                "candidate_json_path": str(candidate_path.relative_to(ROOT)),
                "world_id": WORLD_ID,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(HPC_DIR / "selected_candidates_manifest.csv", index=False)
    manifest.to_csv(DATA / "virtual_ranking_validation_selected_manifest.csv", index=False)

    print(f"Pool: {len(scored)} candidates, virtual_score range [{scored['virtual_score'].min():.4f}, {scored['virtual_score'].max():.4f}]")
    print(f"Selected {len(manifest)} candidates for exact verification -> {HPC_DIR}")
    print(manifest["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
