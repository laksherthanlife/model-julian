#!/usr/bin/env python3
"""Stage 7: Path A'' -- NEW retrained physiology checkpoint + NEW MLP
surrogate, no safety clip (drift is now small enough that the old clip
would rarely fire; recorded here for completeness anyway). Scores the same
60-candidate Stage 7 subset used for Path B'' so the two are directly
comparable without spending any extra LP solves.
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
CANDIDATE_LIST = ROOT / "hpc_bundles" / "surrogate_diagnostic" / "stage7_candidate_list.txt"
CHECKPOINT = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11__new_surrogate.npz"
NEW_SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"
OUT_CSV = DATA / "stage7_path_A_new_checkpoint_new_surrogate.csv"


def main() -> None:
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    params, _meta = tss.load_checkpoint(CHECKPOINT)
    new_surrogate = mlp.GSMSurrogateMLP.load(NEW_SURROGATE_PATH)

    rel_paths = [line.strip() for line in CANDIDATE_LIST.read_text().splitlines() if line.strip()]
    cand_files = [ROOT / "hpc_bundles" / p for p in rel_paths]

    rows = []
    for f in cand_files:
        candidate = json.loads(f.read_text(encoding="utf-8"))
        env = candidate["environment"]
        env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
        env_std_in = (env_raw - env_mean) / env_std
        cache = tss.forward(params, env_std_in, scorer_mod.T_LEN)
        c_pre = cache.z[:-1, 0, :] @ params.Wc + params.bc
        c_baseline = c_pre * new_surrogate.control_std + new_surrogate.control_mean
        c_edited, _notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, candidate.get("edits", []))

        z_edited = np.abs((c_edited - new_surrogate.control_mean) / new_surrogate.control_std)

        X = np.zeros(scorer_mod.T_LEN)
        B = np.zeros(scorer_mod.T_LEN)
        X[0], B[0] = scorer_mod.X0, scorer_mod.B0
        for t in range(scorer_mod.T_LEN - 1):
            values = new_surrogate.predict(c_edited[t : t + 1], env_raw)
            biomass_flux = float(values[0, sur.SURROGATE_OUT_INDEX["biomass_flux"]])
            beta_flux = float(values[0, sur.SURROGATE_OUT_INDEX["beta_carotene_flux"]])
            X[t + 1] = max(1e-9, X[t] + scorer_mod.DT_REAL * biomass_flux * X[t])
            B[t + 1] = max(0.0, B[t] + scorer_mod.DT_REAL * beta_flux * X[t] - scorer_mod.DT_REAL * scorer_mod.BETA_DEG_BASE * B[t])

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "n_edits": len(candidate.get("edits", [])),
                "path_A_new_ckpt_new_sur_score": float(B[-1]),
                "path_A_new_ckpt_new_sur_biomass": float(X[-1]),
                "max_abs_z_edited": float(z_edited.max()),
                "mean_abs_z_edited": float(z_edited.mean()),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Scored {len(df)} candidates.")
    print(f"max_abs_z_edited: mean={df['max_abs_z_edited'].mean():.2f}, max={df['max_abs_z_edited'].max():.2f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
