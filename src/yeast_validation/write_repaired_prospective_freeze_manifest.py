#!/usr/bin/env python3
"""Write the freeze manifest for the repaired-hybrid prospective DBTL rerun.
Must be run, and its output committed to data/, before any powered campaign."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_repaired_prospective_common as common  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    _params, meta = tss.load_checkpoint(common.CHECKPOINT)
    library = common.write_common_library()

    manifest = {
        "benchmark_id": common.REPAIRED_BENCHMARK_ID,
        "predecessor_benchmark_id": "prospective_on_demand_dbtl_v1",
        "predecessor_status": "untouched historical evidence; not modified or re-run by this manifest",
        "architecture": {
            "module": "scripts/gem_trainable_state_space.py",
            "type": "residual-MLP latent transition, hand-derived BPTT, D_model=16, H=24",
            "checkpoint_path": str(common.CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(common.CHECKPOINT),
            "checkpoint_meta": meta,
        },
        "training": {
            "variant": "hybrid_learned_physiology",
            "reporter_mode": "full",
            "model_seed": 11,
            "optimizer": "Adam",
            "lr": 0.02,
            "beta1": 0.9,
            "beta2": 0.999,
            "epochs": 600,
            "lambda_dyn": 2e-4,
            "training_contract": "gem_clean_teacher_targets.py (unchanged) -- deployment inputs temperature/pH/DO only; supervision B_total_observed/X_observed + R_ox/R_atp/R_E_PSY/R_E_DES/R_E_CYC; no raw z_*, no internal GEM bounds, no internal LP fluxes ever used as training targets",
            "reporter_interface": "R_ox, R_atp, R_E_PSY, R_E_DES, R_E_CYC as direct linear heads on shared z(t)",
        },
        "gsm_surrogate": {
            "module": "scripts/gem_gsm_surrogate.py",
            "features": "[1, c, c^2, env, env^2] ridge regression",
            "ridge_lambda": 1.0,
            "control_columns": sur.CONTROL_COLUMNS,
            "trained_from": "gem_state_space_generator_constraints.partial.csv + gem_state_space_fluxes.csv, train split only",
        },
        "edit_semantics": {
            "reaction_to_control": scorer_mod.REACTION_TO_CONTROL,
            "bound_transform": "mirrors deployment_benchmark.apply_sparse_edits_to_model's scaled_positive_upper/scaled_negative_lower",
            "control_clip_stds": scorer_mod.CONTROL_CLIP_STDS,
        },
        "candidate_domain": {
            "common_intervention_library_path": "data/repaired_benchmark_common_intervention_library.csv",
            "common_intervention_library_sha256": file_sha256(DATA / "repaired_benchmark_common_intervention_library.csv"),
            "n_interventions": int(len(library)),
            "reactions_covered": sorted(library["reaction_id"].unique().tolist()),
            "reactions_excluded": ["r_0461", "r_1714", "r_0773"],
            "exclusion_reason": "no corresponding control dimension in the repaired 6-dim control vector; extending would require new exact training data (redesign, out of scope this pass)",
            "identical_domain_for_both_arms": True,
        },
        "virtual_scorer": {
            "module": "scripts/gem_repaired_virtual_scorer.py",
            "objective": "final_product (same objective as run_prospective_dbtl_benchmark.observed_score)",
            "deterministic": True,
        },
        "worlds_and_campaigns": {
            "core_worlds": common.CORE_WORLDS,
            "core_seeds": common.CORE_SEEDS,
            "supplementary_worlds": common.SUPPLEMENTARY_WORLDS,
            "supplementary_seeds": common.SUPPLEMENTARY_SEEDS,
            "conventional_batches": common.CONVENTIONAL_BATCHES,
            "hybrid_verification_batch": common.HYBRID_BATCH,
            "hybrid_virtual_evaluations": common.VIRTUAL_EVALUATIONS,
            "exact_cultures_per_campaign": sum(common.CONVENTIONAL_BATCHES) + common.HYBRID_BATCH,
            "core_campaign_count": len(common.all_campaigns(False)),
            "core_total_exact_cultures": len(common.all_campaigns(False)) * (sum(common.CONVENTIONAL_BATCHES) + common.HYBRID_BATCH),
        },
        "isolation": {
            "note": "bench.DATA/RESULTS/ROLLOUTS/BENCHMARK_ID are runtime-monkey-patched per campaign to isolated directories; the original run_prospective_dbtl_benchmark.py file and its data/*.csv ledgers are never written to",
        },
    }
    out_path = DATA / "repaired_prospective_benchmark_freeze_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({k: manifest[k] for k in ("worlds_and_campaigns", "candidate_domain")}, indent=2, default=str))


if __name__ == "__main__":
    main()
