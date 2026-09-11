#!/usr/bin/env python3
"""Write the freeze manifest for the v2 (surrogate-repaired, exact-Yeast9-
reranked) powered prospective DBTL rerun. Must be run, and its output
committed, before any powered campaign in this pass -- not retuned after
seeing results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    _params, ckpt_meta = tss.load_checkpoint(v2.NEW_CHECKPOINT)
    library = v2.build_common_intervention_library()
    library_path = DATA / "repaired_benchmark_common_intervention_library.csv"

    manifest = {
        "benchmark_id": v2.REPAIRED_BENCHMARK_ID_V2,
        "predecessor_benchmark_id": "repaired_hybrid_prospective_dbtl_v1",
        "predecessor_status": "Section K, untouched historical evidence; not modified or re-run by this manifest",
        "question": "Does the clean hybrid digital twin outperform conventional adaptive DBTL once the GSM-surrogate ranking problem (Section L) has been repaired?",
        "architecture": {
            "module": "scripts/gem_trainable_state_space.py",
            "type": "residual-MLP latent transition, hand-derived BPTT, D_model=16, H=24 (UNCHANGED architecture/hyperparameters from Section K)",
            "checkpoint_path": str(v2.NEW_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(v2.NEW_CHECKPOINT),
            "checkpoint_meta": ckpt_meta,
            "note": "Retrained (Section L.5) THROUGH the new surrogate only -- same seed=11, same D_model/hidden width/epochs/lr/reporter_mode/training data as Section K's checkpoint. Not retuned for this benchmark.",
        },
        "gsm_surrogate": {
            "module": "scripts/gem_gsm_surrogate_mlp.py",
            "type": "1-hidden-layer tanh MLP (width 32), exact analytic Jacobian (Section L.3)",
            "checkpoint_path": str(v2.NEW_SURROGATE_PATH.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(v2.NEW_SURROGATE_PATH),
            "trained_from": "9,640 purpose-built exact-Yeast9 points (Section L.2) -- trajectory + edge-case/decorrelated coverage, NOT the old reference-strain-only training set",
            "note": "Frozen; not retrained or retuned based on this benchmark's results (per instructions).",
        },
        "edit_semantics": {
            "reaction_to_control": scorer_mod.REACTION_TO_CONTROL,
            "bound_transform": "mirrors deployment_benchmark.apply_sparse_edits_to_model's scaled_positive_upper/scaled_negative_lower (unchanged)",
            "control_clip_stds": scorer_mod.CONTROL_CLIP_STDS,
            "note": "Same clip code as Section K (unchanged) -- but the new surrogate's own control_mean/std are ~3.4-5.5x wider (fit on Stage-2 data spanning +/-8std of the old surrogate's units), so the clip is effectively far less restrictive without any code change (verified empirically: 0/1 test candidates clipped vs. 100% under the old surrogate).",
        },
        "candidate_domain": {
            "common_intervention_library_path": str(library_path.relative_to(ROOT)),
            "common_intervention_library_sha256": file_sha256(library_path),
            "n_interventions": int(len(library)),
            "reactions_covered": sorted(library["reaction_id"].unique().tolist()),
            "identical_domain_for_both_arms": True,
            "note": "Byte-identical restricted 16/24-intervention library file as Section K -- not regenerated, not broadened.",
        },
        "hybrid_workflow": {
            "stage1_virtual_screen": "environment -> new checkpoint -> z(t) -> predicted c(t) -> edit -> new MLP surrogate -> mechanistic integration -> virtual_score (RepairedVirtualScorer.score, unmodified code, new checkpoint+surrogate injected)",
            "stage1_virtual_evaluations": v2.VIRTUAL_EVALUATIONS,
            "stage2_exact_rerank": "same learned c(t) + same edit -> REAL Yeast9 (apply_predicted_interface_controls + gem_backend.solve_staged, run_repaired_controls_real_gsm_replay_worker.py UNMODIFIED, new checkpoint) -- NOT the hidden generator's own physiology. In-silico metabolic compute, not a biological-style culture.",
            "stage2_shortlist_size": v2.SHORTLIST_SIZE,
            "stage2_shortlist_selection": "deduplicated top-50 by Stage-1 surrogate score (canonical_design_key dedup, same as Section K)",
            "stage3_biological_verification": "true hidden generator's exact_sparse_dynamic_rollout (run_on_demand_exact, unmodified) on the reranked top-8",
            "stage3_verification_batch": v2.HYBRID_VERIFICATION_BATCH,
            "final_selection_rule": "top-8 by Stage-2 real-Yeast9 rerank score (not Stage-1 surrogate score)",
        },
        "conventional_arm": {
            "batches": v2.CONVENTIONAL_BATCHES,
            "total_exact_cultures": sum(v2.CONVENTIONAL_BATCHES),
            "algorithm": "propose_conventional_batch, unmodified from run_prospective_dbtl_benchmark.py",
        },
        "worlds_and_campaigns": {
            "core_worlds": v2.CORE_WORLDS,
            "core_seeds": v2.CORE_SEEDS,
            "campaign_count": len(v2.all_campaigns()),
            "seed_collision_check": "98001-98020, non-colliding with Section K's 97001-97030",
            "exact_biological_cultures_per_campaign": sum(v2.CONVENTIONAL_BATCHES) + v2.HYBRID_VERIFICATION_BATCH,
            "total_biological_cultures": len(v2.all_campaigns()) * (sum(v2.CONVENTIONAL_BATCHES) + v2.HYBRID_VERIFICATION_BATCH),
            "total_stage1_virtual_evaluations": len(v2.all_campaigns()) * v2.VIRTUAL_EVALUATIONS,
            "total_stage2_rerank_trajectories": len(v2.all_campaigns()) * v2.SHORTLIST_SIZE,
        },
        "isolation": {
            "note": "bench.DATA/RESULTS/ROLLOUTS/BENCHMARK_ID/hybrid_virtual_search are runtime-monkey-patched per campaign to isolated directories and a reranking-aware closure; run_prospective_dbtl_benchmark.py itself and Section K's data/results are never written to.",
            "hpc_storage_note": "Vanda home-dir quota was full during this pass; Phase 2/3 outputs (candidates, rerank_results, per-campaign results) were written under /scratch/e1471252/prospective_dbtl_benchmark_v2/ (scripts/models/.venv/checkpoints symlinked read-only from the home-dir repo) and synced back to the repo after completion.",
        },
        "not_changed_from_section_k": [
            "D_model=16, hidden width=24, learner architecture",
            "historical biological training dataset, reporter set, reference-strain-only training contract",
            "deployment inputs (temperature/pH/DO)",
            "conventional DBTL algorithm and batch structure [4,4,4,4]",
            "worlds (baseline_world, strong_oxidative_burden_world)",
            "paired-analysis procedure (bootstrap_ci/write_analysis, unmodified)",
            "restricted common intervention library",
        ],
    }
    out_path = DATA / "repaired_prospective_benchmark_v2_freeze_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(manifest["worlds_and_campaigns"], indent=2, default=str))


if __name__ == "__main__":
    main()
