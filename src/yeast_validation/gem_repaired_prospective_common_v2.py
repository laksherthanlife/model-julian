#!/usr/bin/env python3
"""Shared configuration for the v2 (surrogate-repaired, exact-Yeast9-reranked)
powered prospective DBTL rerun -- the direct successor to Section K, using:

  - the retrained physiology checkpoint
    (`hybrid_learned_physiology__full__seed11__new_surrogate.npz`), and
  - the new MLP GSM surrogate (`gsm_surrogate_mlp_v1.npz`)

isolated exactly like Section K's `gem_repaired_prospective_common.py`
(never editing `run_prospective_dbtl_benchmark.py`, only monkey-patching it
per campaign), plus one new capability K didn't have: a Stage-2 exact-Yeast9
reranking of the surrogate's top-50 shortlist before the final top-8 goes to
biological-style verification (`exact_sparse_dynamic_rollout`, the true
hidden generator -- untouched).

Three-phase pipeline (each phase is a separate script, so the expensive
reranking work can run as its own PBS array independent of any single
campaign's process):

  Phase 1 (`generate_repaired_prospective_v2_shortlists.py`, cheap, no LP):
    for each of the 20 campaigns, deterministically reproduce the SAME
    6000-candidate hybrid-arm virtual screen `run_campaign` will reproduce
    in Phase 3 (`generate_hybrid_shortlist` below -- the single source of
    truth for this logic, called by both phases), select the top-50 by new
    surrogate score, write 50 candidate JSONs + a shortlist manifest.

  Phase 2 (PBS array, expensive): rerank those 1000 candidates (20 x 50)
    through real Yeast9 using the SAME learned physiology + edit mapping
    (not the hidden generator's own physiology) -- reuses
    `run_repaired_controls_real_gsm_replay_worker.py` UNMODIFIED, pointed at
    the new checkpoint via `--checkpoint`. This is in-silico metabolic
    compute, never a biological-style culture.

  Phase 3 (`run_repaired_prospective_campaign_worker_v2.py`, one PBS array
    task per campaign): runs `bench.run_campaign` for real (both arms), with
    `bench.hybrid_virtual_search` monkey-patched to a closure that
    regenerates the identical shortlist (Phase 1's function, bit-identical
    given the same seed/library/scorer), looks up each of the 50 candidates'
    precomputed Phase-2 rerank score, and selects the true top-8 by rerank
    score for biological-style verification via the unmodified
    `run_on_demand_exact` -> `exact_sparse_dynamic_rollout` path.

Determinism note: candidate generation is a pure function of
(campaign_id, method_id, seed, library, virtual_evaluations) via
`np.random.default_rng(seed + 29)` (the exact offset `run_campaign` uses for
the non-first `PRIMARY_METHODS` entry, i.e. the hybrid arm) -- as long as
Phase 1 and Phase 3 construct the factory identically and call
`generate_hybrid_shortlist` with no other RNG draws in between, the 50
shortlisted candidate hashes are guaranteed bit-identical between phases.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_repaired_prospective_common as common_v1  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

REPAIRED_BENCHMARK_ID_V2 = "repaired_hybrid_prospective_dbtl_v2_reranked"
NEW_CHECKPOINT = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11__new_surrogate.npz"
NEW_SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"

# New, non-colliding campaign seeds (Section K used 97001-97030).
CORE_WORLDS = common_v1.CORE_WORLDS  # ["baseline_world", "strong_oxidative_burden_world"]
CORE_SEEDS = {
    "baseline_world": list(range(98001, 98011)),
    "strong_oxidative_burden_world": list(range(98011, 98021)),
}

CONVENTIONAL_BATCHES = [4, 4, 4, 4]
HYBRID_VERIFICATION_BATCH = 8  # final biological-style culture count, unchanged from Section K
SHORTLIST_SIZE = 50  # new Stage-2 exact-Yeast9 reranking shortlist, frozen
VIRTUAL_EVALUATIONS = 6000

PRIMARY_METHODS = bench.PRIMARY_METHODS  # ["conventional_dbtl", "pretrained_hybrid_digital_twin"]
HYBRID_METHOD_ID = PRIMARY_METHODS[1]


def all_campaigns() -> list[tuple[str, int]]:
    return [(w, s) for w in CORE_WORLDS for s in CORE_SEEDS[w]]


def campaign_id_for(world_id: str, seed: int) -> str:
    return f"prospective_{world_id}_seed_{seed}"


def build_common_intervention_library() -> pd.DataFrame:
    # Identical restricted 16/24-intervention domain as Section K -- reused
    # unchanged, not re-derived, so the candidate domain is provably the
    # same design space as the previous clean powered benchmark.
    return common_v1.build_common_intervention_library()


def load_scorer_v2() -> "scorer_mod.RepairedVirtualScorer":
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    new_surrogate = mlp.GSMSurrogateMLP.load(NEW_SURROGATE_PATH)
    return scorer_mod.RepairedVirtualScorer.load(NEW_CHECKPOINT, env_mean, env_std, surrogate=new_surrogate)


def patch_benchmark_module_v2(campaign_data_dir: Path, hybrid_virtual_search_fn) -> None:
    """Redirect bench's I/O to an isolated directory, swap BENCHMARK_ID, and
    install the reranking-aware hybrid_virtual_search replacement. Call once
    per process before calling bench.run_campaign."""
    bench.hybrid_virtual_search = hybrid_virtual_search_fn
    bench.BENCHMARK_ID = REPAIRED_BENCHMARK_ID_V2
    bench.DATA = campaign_data_dir / "data"
    bench.RESULTS = campaign_data_dir / "results"
    bench.ROLLOUTS = bench.RESULTS / "on_demand_exact_rollouts"
    bench.DATA.mkdir(parents=True, exist_ok=True)
    bench.ROLLOUTS.mkdir(parents=True, exist_ok=True)


def make_factory(library: pd.DataFrame, campaign_id: str, seed: int) -> "bench.CandidateFactory":
    """Reconstructs the SAME hybrid-arm CandidateFactory run_campaign builds
    internally (seed + 29, since HYBRID_METHOD_ID != PRIMARY_METHODS[0]) --
    used by Phase 1 to independently reproduce Phase 3's factory state."""
    return bench.CandidateFactory(library, campaign_id, HYBRID_METHOD_ID, np.random.default_rng(seed + 29))


def generate_hybrid_shortlist(
    factory: "bench.CandidateFactory",
    virtual_evaluations: int,
    shortlist_size: int,
    seen_keys: set[str],
    library: pd.DataFrame,
    scorer: "scorer_mod.RepairedVirtualScorer",
    campaign_id: str,
    world_id: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Single source of truth for the cheap virtual screen (Stage 1 of the
    hybrid workflow): score `virtual_evaluations` candidates with the new
    checkpoint+surrogate (no LP -- RepairedVirtualScorer.score is pure
    numpy), sort by surrogate score, dedupe by canonical_design_key, return
    the top `shortlist_size` plus the FULL 6000-row virtual ledger
    (with surrogate_rank / entered_shortlist diagnostic columns already
    filled in; rerank columns are filled in later by the caller once Phase
    2's precomputed scores are available). Called identically by Phase 1
    (standalone) and Phase 3 (inside the patched hybrid_virtual_search) --
    bit-identical given the same factory/library/scorer/seed."""
    started = time.perf_counter()
    started_at = pd.Timestamp.utcnow().isoformat()
    rows = []
    candidates = []
    for i in range(virtual_evaluations):
        cand = factory.sample_candidate(max_edits=3)
        result = scorer.score(cand)
        score = float(result["virtual_score"])
        rows.append(
            {
                "benchmark_id": REPAIRED_BENCHMARK_ID_V2,
                "campaign_id": campaign_id,
                "world_id": world_id,
                "method_id": HYBRID_METHOD_ID,
                "virtual_evaluation_index": i + 1,
                "candidate_id": cand["candidate_id"],
                "candidate_hash": cand["candidate_hash"],
                "design_key": bench.canonical_design_key(cand),
                "temperature": cand["environment"]["temperature"],
                "pH": cand["environment"]["pH"],
                "DO": cand["environment"]["DO"],
                "edit_count": len(cand.get("edits", [])),
                "mechanism_classes": bench.mechanism_classes(cand, library),
                "hybrid_virtual_score": score,
                "surrogate_score": score,
                "n_control_values_clipped": int(result.get("n_control_values_clipped", 0)),
                "valid_virtual_candidate": True,
                "rejection_reason": "",
                "exact_outcome_accessed": False,
                "selected_for_exact_verification": False,
                "entered_shortlist_top50": False,
                "rerank_score": np.nan,
                "rerank_rank": np.nan,
                "entered_final_top8": False,
                "virtual_search_started_at": started_at,
                "source": "repaired_physiology_v2_new_checkpoint_new_surrogate",
            }
        )
        cand["_surrogate_score"] = score
        candidates.append(cand)

    candidates.sort(key=lambda c: float(c["_surrogate_score"]), reverse=True)
    rank_by_hash = {}
    for rank, c in enumerate(candidates, start=1):
        rank_by_hash[c["candidate_hash"]] = rank

    shortlist: list[dict[str, Any]] = []
    for cand in candidates:
        key = bench.canonical_design_key(cand)
        if key in seen_keys:
            continue
        shortlist.append(cand)
        seen_keys.add(key)
        if len(shortlist) == shortlist_size:
            break
    if len(shortlist) < shortlist_size:
        raise RuntimeError(f"Hybrid virtual screen selected only {len(shortlist)}/{shortlist_size} shortlist candidates for {campaign_id}")

    virtual = pd.DataFrame(rows)
    virtual["surrogate_rank"] = virtual["candidate_hash"].map(rank_by_hash)
    shortlist_hashes = {c["candidate_hash"] for c in shortlist}
    virtual["entered_shortlist_top50"] = virtual["candidate_hash"].isin(shortlist_hashes)
    virtual["virtual_search_finished_at"] = pd.Timestamp.utcnow().isoformat()
    virtual["virtual_runtime_seconds"] = time.perf_counter() - started
    return shortlist, virtual
