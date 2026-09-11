#!/usr/bin/env python3
"""Shared configuration for the repaired-hybrid prospective DBTL rerun.

Reuses `run_prospective_dbtl_benchmark.py` (the frozen headline file) almost
entirely unmodified -- `run_campaign`, `hybrid_virtual_search`,
`propose_conventional_batch`, every audit-ledger writer, and the leakage
audit machinery are all called as-is. Three things are swapped, all via
runtime monkey-patch from this module (never by editing the frozen file):

1. `bench.hybrid_virtual_score` -> the repaired model-driven scorer
   (`gem_repaired_virtual_scorer.RepairedVirtualScorer.score`), so
   `hybrid_virtual_search`'s call site is unchanged but now genuinely
   executes the repaired learner instead of a lookup table.
2. `bench.DATA` / `bench.RESULTS` / `bench.ROLLOUTS` -> a fresh, isolated
   directory tree per run (and per campaign, when parallelised), so the
   original benchmark's ledgers under `data/` are never appended to,
   overwritten, or mixed with -- it remains untouched historical evidence.
3. `bench.BENCHMARK_ID` -> a distinct id, so `culture_task_id` hashes (and
   therefore any cache-hit checks) can never collide with the original run's.

Common design domain (Step 1 of the rerun): the frozen 24-intervention
library targets 7 reactions; only 4 (BETA_PHYTOENE_SYNTHASE,
BETA_PHYTOENE_DESATURASE, BETA_LYCOPENE_CYCLASE, r_1992/oxygen exchange, 16
of 24 interventions) are representable by the repaired virtual scorer's
6-dim control vector (see LEARNER_ARCHITECTURE_REPAIR.md Section H.2 for the
audit). Extending true surrogate sensitivity to the other 3 reactions would
require new exact training data (the historical dataset never varies those
bounds) -- out of scope for this pass. Per the instructions, we therefore use
Option 2: a restricted **common** intervention domain (only the 16 covered
interventions) for *both* arms, so no candidate edit is ever silently
skipped in the powered benchmark, and the candidate domain is identical
between workflows (an explicit leakage/fairness gate).
"""

from __future__ import annotations

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

REPAIRED_BENCHMARK_ID = "repaired_hybrid_prospective_dbtl_v1"
CHECKPOINT = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11.npz"

# Core, power-matched to the original benchmark's 20 paired campaigns (10+10).
CORE_WORLDS = ["baseline_world", "strong_oxidative_burden_world"]
CORE_SEEDS = {
    "baseline_world": list(range(97001, 97011)),
    "strong_oxidative_burden_world": list(range(97011, 97021)),
}
# Supplementary, added only if compute allows (Section: worlds and pairing).
SUPPLEMENTARY_WORLDS = ["atp_limited_world"]
SUPPLEMENTARY_SEEDS = {"atp_limited_world": list(range(97021, 97031))}

CONVENTIONAL_BATCHES = [4, 4, 4, 4]
HYBRID_BATCH = 8
VIRTUAL_EVALUATIONS = 6000


def all_campaigns(include_supplementary: bool = False) -> list[tuple[str, int]]:
    campaigns = [(w, s) for w in CORE_WORLDS for s in CORE_SEEDS[w]]
    if include_supplementary:
        campaigns += [(w, s) for w in SUPPLEMENTARY_WORLDS for s in SUPPLEMENTARY_SEEDS[w]]
    return campaigns


def build_common_intervention_library() -> pd.DataFrame:
    full = bench.load_library()
    covered_reactions = set(scorer_mod.REACTION_TO_CONTROL.keys())
    restricted = full[full["reaction_id"].isin(covered_reactions)].reset_index(drop=True)
    return restricted


def write_common_library(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA / "repaired_benchmark_common_intervention_library.csv")
    restricted = build_common_intervention_library()
    path.parent.mkdir(parents=True, exist_ok=True)
    restricted.to_csv(path, index=False)
    return restricted


def load_scorer() -> "scorer_mod.RepairedVirtualScorer":
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    return scorer_mod.RepairedVirtualScorer.load(CHECKPOINT, env_mean, env_std)


def repaired_hybrid_virtual_score(candidate: dict, library: pd.DataFrame, scorer: "scorer_mod.RepairedVirtualScorer") -> float:
    return float(scorer.score(candidate)["virtual_score"])


def patch_benchmark_module(campaign_data_dir: Path) -> None:
    """Redirect bench's I/O to an isolated directory and swap in the repaired scorer.
    Call once per process before calling bench.run_campaign."""
    bench.hybrid_virtual_score = repaired_hybrid_virtual_score
    bench.BENCHMARK_ID = REPAIRED_BENCHMARK_ID
    bench.DATA = campaign_data_dir / "data"
    bench.RESULTS = campaign_data_dir / "results"
    bench.ROLLOUTS = bench.RESULTS / "on_demand_exact_rollouts"
    bench.DATA.mkdir(parents=True, exist_ok=True)
    bench.ROLLOUTS.mkdir(parents=True, exist_ok=True)


def campaign_id_for(world_id: str, seed: int) -> str:
    return f"prospective_{world_id}_seed_{seed}"
