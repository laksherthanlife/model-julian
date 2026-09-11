#!/usr/bin/env python3
"""Phase 3 of the v2 powered rerun: run ONE paired campaign (both
conventional_dbtl and pretrained_hybrid_digital_twin) with
bench.hybrid_virtual_search monkey-patched to a reranking-aware replacement
that LOADS Phase 1's persisted output (never recomputes it):

  1. load the exact 6000-row virtual ledger Phase 1 wrote for this campaign,
     and the 50 shortlisted candidate JSONs (see
     generate_repaired_prospective_v2_shortlists.py -- an earlier version
     regenerated the screen via the same RNG seed and assumed it would be
     bit-identical to Phase 1's; it wasn't, across machines, right at the
     rank-50 cutoff boundary, so this version loads instead of recomputing);
  2. look up each of the 50 shortlist candidates' precomputed Phase-2
     real-Yeast9 rerank score (from
     hpc_bundles/repaired_prospective_v2/rerank_results/*_summary.csv,
     produced by the unmodified run_repaired_controls_real_gsm_replay_worker.py);
  3. select the true top-8 by rerank score for biological-style verification
     via the unchanged run_on_demand_exact -> exact_sparse_dynamic_rollout
     path (the true hidden generator; never touched by steps 1-2).

Everything else -- the conventional arm, ledger writers, leakage audit
machinery -- is `bench.run_campaign` unmodified, exactly as in Section K's
worker. One PBS array task per campaign; campaigns are independent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_shortlist_and_virtual(campaign_id: str, candidates_dir: Path, shortlists_dir: Path, virtual_ledgers_dir: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Loads Phase 1's persisted output directly -- no recomputation, so no
    cross-machine floating-point determinism dependency (see module
    docstring)."""
    # Deliberately do NOT attach any extra bookkeeping keys to the candidate
    # dicts themselves -- run_prospective_dbtl_benchmark.py's
    # prospective_candidate_designs.csv writer spreads **cand wholesale
    # (csv_safe_row({..., **cand})), so any extra key added here would leak
    # into that shared-schema ledger and break it for every row after the
    # first (append_csv only writes a header once). Keep scores in a
    # separate dict keyed by candidate_hash instead (see below).
    manifest = pd.read_csv(shortlists_dir / f"{campaign_id}_shortlist_manifest.csv")
    shortlist = []
    for row in manifest.itertuples():
        cand_path = candidates_dir / f"{row.candidate_id}.json"
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        shortlist.append(cand)
    virtual = pd.read_csv(virtual_ledgers_dir / f"{campaign_id}_virtual.csv")
    return shortlist, virtual


def load_rerank_lookup(campaign_id: str, shortlists_dir: Path, rerank_results_dir: Path) -> dict[str, float]:
    manifest_path = shortlists_dir / f"{campaign_id}_shortlist_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    lookup: dict[str, float] = {}
    missing = []
    for row in manifest.itertuples():
        summary_path = rerank_results_dir / f"{row.candidate_id}_summary.csv"
        if not summary_path.exists():
            missing.append(row.candidate_id)
            continue
        summary = pd.read_csv(summary_path)
        lookup[str(row.candidate_hash)] = float(summary["final_product_pred_controls_via_real_gsm"].iloc[0])
    if missing:
        raise RuntimeError(f"{campaign_id}: {len(missing)}/{len(manifest)} rerank results missing (Phase 2 incomplete): {missing[:5]}")
    return lookup


def make_reranked_hybrid_search(rerank_lookup: dict[str, float], candidates_dir: Path, shortlists_dir: Path, virtual_ledgers_dir: Path):
    def reranked_hybrid_virtual_search(
        factory: "bench.CandidateFactory",
        virtual_evaluations: int,
        verification_batch: int,
        seen_keys: set[str],
        library: pd.DataFrame,
        teacher_lookup: Any,
        campaign_id: str,
        world_id: str,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        # factory/seen_keys/teacher_lookup/virtual_evaluations are accepted
        # only for signature compatibility with bench.run_campaign's call
        # site -- Phase 1's persisted output is the single source of truth
        # (see module docstring), never recomputed here.
        shortlist, virtual = load_shortlist_and_virtual(campaign_id, candidates_dir, shortlists_dir, virtual_ledgers_dir)
        missing = [c["candidate_hash"] for c in shortlist if c["candidate_hash"] not in rerank_lookup]
        if missing:
            raise RuntimeError(f"{campaign_id}: {len(missing)} shortlist candidates have no precomputed rerank score (Phase 2 incomplete?): {missing[:5]}")

        shortlist.sort(key=lambda c: float(rerank_lookup[c["candidate_hash"]]), reverse=True)
        rerank_score_by_hash = {c["candidate_hash"]: float(rerank_lookup[c["candidate_hash"]]) for c in shortlist}
        rerank_rank_by_hash = {c["candidate_hash"]: rank for rank, c in enumerate(shortlist, start=1)}

        selected = shortlist[:verification_batch]
        if len(selected) < verification_batch:
            raise RuntimeError(f"{campaign_id}: reranked shortlist has only {len(selected)}/{verification_batch} candidates")
        selected_hashes = {c["candidate_hash"] for c in selected}
        for cand in selected:
            cand["proposal_score"] = rerank_score_by_hash[cand["candidate_hash"]]  # what actually determined final selection; the ONLY key run_campaign's own code expects on every candidate (conventional sets it too), so safe to add
        virtual["rerank_score"] = virtual["candidate_hash"].map(rerank_score_by_hash)
        virtual["rerank_rank"] = virtual["candidate_hash"].map(rerank_rank_by_hash)
        virtual["entered_final_top8"] = virtual["candidate_hash"].isin(selected_hashes)
        virtual["selected_for_exact_verification"] = virtual["candidate_hash"].isin(selected_hashes)

        bench.append_virtual_ledger(virtual)
        return selected, virtual

    return reranked_hybrid_virtual_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conventional-batches", type=int, nargs="+", default=v2.CONVENTIONAL_BATCHES)
    parser.add_argument("--hybrid-batch", type=int, default=v2.HYBRID_VERIFICATION_BATCH)
    parser.add_argument("--virtual-evaluations", type=int, default=v2.VIRTUAL_EVALUATIONS)
    parser.add_argument("--out-root", type=str, default=str(ROOT / "results" / "repaired_prospective_campaigns_v2"))
    # cwd-relative by default (matches the PBS convention of cd "$PBS_O_WORKDIR"
    # before invoking this script) -- NOT resolved from this file's own
    # location, since that would follow the scripts/ symlink back to the
    # read-only home-dir repo root rather than the scratch working directory
    # actually holding Phase 1/2's outputs.
    parser.add_argument("--shortlists-dir", type=str, default="repaired_prospective_v2/shortlists")
    parser.add_argument("--rerank-results-dir", type=str, default="repaired_prospective_v2/rerank_results")
    parser.add_argument("--candidates-dir", type=str, default="repaired_prospective_v2/candidates")
    parser.add_argument("--virtual-ledgers-dir", type=str, default="repaired_prospective_v2/virtual_ledgers")
    args = parser.parse_args()

    campaign_id = v2.campaign_id_for(args.world_id, args.seed)
    campaign_dir = Path(args.out_root) / campaign_id
    done_marker = campaign_dir / "_DONE"
    if done_marker.exists():
        print(f"already done: {campaign_id}")
        return

    library = v2.build_common_intervention_library()
    scorer = v2.load_scorer_v2()
    rerank_lookup = load_rerank_lookup(campaign_id, Path(args.shortlists_dir), Path(args.rerank_results_dir))
    reranked_search = make_reranked_hybrid_search(rerank_lookup, Path(args.candidates_dir), Path(args.shortlists_dir), Path(args.virtual_ledgers_dir))
    v2.patch_benchmark_module_v2(campaign_dir, reranked_search)

    tic = time.perf_counter()
    bench.run_campaign(
        world_id=args.world_id,
        seed=args.seed,
        library=library,
        teacher_lookup=scorer,
        conventional_batches=args.conventional_batches,
        hybrid_batch=args.hybrid_batch,
        virtual_evaluations=args.virtual_evaluations,
        mock=False,
        fresh_exact=False,  # allow resume: reuse this campaign's own isolated exact-solve cache
    )
    elapsed = time.perf_counter() - tic
    done_marker.write_text(f"elapsed_seconds={elapsed:.1f}\n", encoding="utf-8")
    print(f"campaign {campaign_id} done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
