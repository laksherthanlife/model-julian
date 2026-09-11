#!/usr/bin/env python3
"""Stage 4 (ranking half): rescore the exact same Stage 1 300-candidate pool
with the NEW MLP surrogate (Path A') -- same physiology checkpoint, same
edit mapping, same candidates -- and compare to the already-computed Path B
(real Yeast9) scores. This isolates whether the new surrogate closes the
ranking gap found in Stage 1, without spending any new LP solves (Path B is
reused unmodified).

Unlike the old scorer, no CONTROL_CLIP_STDS safety clip is applied here --
the new surrogate is trained explicitly to cover the extrapolative region
that clip existed to guard against. The old clip's trigger rate is still
recorded (informational) for the before/after clipping-rate comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CAND_DIR = ROOT / "hpc_bundles" / "surrogate_diagnostic" / "candidates"
CHECKPOINT = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11.npz"
NEW_SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"
OUT_CSV = DATA / "surrogate_diagnostic_pool_path_A_new.csv"

# Old safety-clip thresholds, used here only to report the "before" clipping
# rate for comparison -- not applied to the new surrogate's scoring.
OLD_CONTROL_CLIP_STDS = scorer_mod.CONTROL_CLIP_STDS


def main() -> None:
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    params, _meta = tss.load_checkpoint(CHECKPOINT)
    new_surrogate = mlp.GSMSurrogateMLP.load(NEW_SURROGATE_PATH)

    # Old surrogate's control_mean/std, purely to compute the "would the old
    # 4-std clip have fired" informational statistic on the same edited c(t).
    old_scorer = scorer_mod.RepairedVirtualScorer.load(CHECKPOINT, env_mean, env_std)
    old_control_mean = old_scorer.surrogate.control_mean
    old_control_std = old_scorer.surrogate.control_std

    cand_files = sorted(CAND_DIR.glob("*.json"))
    rows = []
    for f in cand_files:
        candidate = json.loads(f.read_text(encoding="utf-8"))
        env = candidate["environment"]
        env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
        env_std_in = (env_raw - env_mean) / env_std
        cache = tss.forward(params, env_std_in, scorer_mod.T_LEN)
        c_pre = cache.z[:-1, 0, :] @ params.Wc + params.bc
        c_baseline = c_pre * old_scorer.surrogate.control_std + old_scorer.surrogate.control_mean
        c_edited, _notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, candidate.get("edits", []))

        old_clip_lo = old_control_mean - OLD_CONTROL_CLIP_STDS * old_control_std
        old_clip_hi = old_control_mean + OLD_CONTROL_CLIP_STDS * old_control_std
        n_would_have_clipped = int(np.sum((c_edited < old_clip_lo) | (c_edited > old_clip_hi)))

        X = np.zeros(scorer_mod.T_LEN)
        B = np.zeros(scorer_mod.T_LEN)
        X[0], B[0] = scorer_mod.X0, scorer_mod.B0
        env_broadcast = env_raw
        for t in range(scorer_mod.T_LEN - 1):
            values = new_surrogate.predict(c_edited[t : t + 1], env_broadcast)
            biomass_flux = float(values[0, sur.SURROGATE_OUT_INDEX["biomass_flux"]])
            beta_flux = float(values[0, sur.SURROGATE_OUT_INDEX["beta_carotene_flux"]])
            X[t + 1] = max(1e-9, X[t] + scorer_mod.DT_REAL * biomass_flux * X[t])
            B[t + 1] = max(0.0, B[t] + scorer_mod.DT_REAL * beta_flux * X[t] - scorer_mod.DT_REAL * scorer_mod.BETA_DEG_BASE * B[t])

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "path_A_new_virtual_score": float(B[-1]),
                "path_A_new_final_biomass": float(X[-1]),
                "n_control_values_would_have_clipped_old": n_would_have_clipped,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(df)} candidates with new surrogate.")
    print(f"Would-have-clipped>0 fraction (old 4std clip, informational): {(df['n_control_values_would_have_clipped_old']>0).mean():.3f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
