#!/usr/bin/env python3
"""Stage 1: isolate whether the GSM surrogate materially scrambles virtual
candidate ranking, holding the learned physiology and edit mapping fixed.

Path A: same c(t) + same edits -> current surrogate -> objective (this IS
gem_repaired_virtual_scorer.RepairedVirtualScorer.score, computed here
directly, no LP).
Path B: same c(t) + same edits -> real Yeast9 -> objective (computed on
Vanda by run_repaired_controls_real_gsm_replay_worker.py, reused unmodified
from the previous pass).

This script: samples a candidate pool from the actual prospective design
domain (same restricted common library + CandidateFactory used by the
powered rerun), scores Path A + records support-distance/clipping metadata
for every candidate, and writes candidate JSONs for the Vanda Path B array.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_repaired_prospective_common as common  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HPC_DIR = ROOT / "hpc_bundles" / "surrogate_diagnostic"
N_POOL = 300
SEED = 51001


def generate_pool(n: int, seed: int) -> list[dict]:
    library = common.build_common_intervention_library()
    rng = np.random.default_rng(seed)
    factory = bench.CandidateFactory(library, "surrogate_diagnostic", "repaired_hybrid", rng)
    candidates = []
    for _ in range(n):
        edit_count = int(rng.choice(np.arange(0, 4), p=[0.15, 0.35, 0.30, 0.20]))
        if edit_count == 0:
            cand = factory.make([], factory.random_env(), "surrogate diagnostic: no-edit reference-like point")
        else:
            idx = rng.choice(len(library), size=min(edit_count, len(library)), replace=False)
            rows = [pd.Series(library.iloc[int(i)]) for i in idx]
            cand = factory.make(rows, factory.random_env(), "surrogate diagnostic: sampled edited candidate")
        candidates.append(cand)
    return candidates


def support_distance_stats(c_baseline: np.ndarray, c_edited: np.ndarray, control_mean: np.ndarray, control_std: np.ndarray) -> dict:
    z_baseline = np.abs((c_baseline - control_mean) / control_std)
    z_edited = np.abs((c_edited - control_mean) / control_std)
    return {
        "max_abs_z_baseline": float(z_baseline.max()),
        "mean_abs_z_baseline": float(z_baseline.mean()),
        "max_abs_z_edited": float(z_edited.max()),
        "mean_abs_z_edited": float(z_edited.mean()),
    }


def main() -> None:
    HPC_DIR.mkdir(parents=True, exist_ok=True)
    (HPC_DIR / "candidates").mkdir(exist_ok=True)
    (HPC_DIR / "results").mkdir(exist_ok=True)

    candidates = generate_pool(N_POOL, SEED)
    scorer = common.load_scorer()

    rows = []
    for cand in candidates:
        env = cand["environment"]
        env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
        env_std_in = (env_raw - scorer.env_mean) / scorer.env_std
        import gem_trainable_state_space as tss

        cache = tss.forward(scorer.params, env_std_in, scorer_mod.T_LEN)
        c_pre = cache.z[:-1, 0, :] @ scorer.params.Wc + scorer.params.bc
        c_baseline = c_pre * scorer.surrogate.control_std + scorer.surrogate.control_mean
        c_edited, edit_notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, cand.get("edits", []))
        dist = support_distance_stats(c_baseline, c_edited, scorer.surrogate.control_mean, scorer.surrogate.control_std)

        result = scorer.score(cand)
        edits = cand.get("edits", [])
        row = {
            "candidate_id": cand["candidate_id"],
            "candidate_hash": cand["candidate_hash"],
            "temperature": env["temperature"],
            "pH": env["pH"],
            "DO": env["DO"],
            "n_edits": len(edits),
            "edit_reactions": ";".join(sorted({e.get("reaction_id", "") for e in edits})),
            "path_A_virtual_score": result["virtual_score"],
            "n_control_values_clipped": result["n_control_values_clipped"],
            **dist,
        }
        rows.append(row)
        (HPC_DIR / "candidates" / f"{cand['candidate_id']}.json").write_text(json.dumps(cand), encoding="utf-8")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "surrogate_diagnostic_pool_path_A.csv", index=False)
    print(f"Pool: {len(df)} candidates. n_edits distribution:\n{df['n_edits'].value_counts()}")
    print(f"clipped>0 fraction: {(df['n_control_values_clipped']>0).mean():.3f}")
    print(f"max_abs_z_edited>4 fraction (near/at clip boundary): {(df['max_abs_z_edited']>4).mean():.3f}")


if __name__ == "__main__":
    main()
