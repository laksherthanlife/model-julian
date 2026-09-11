#!/usr/bin/env python3
"""Step 4 worker: replay the repaired model's predicted (edit-adjusted)
control trajectory c(t) through REAL Yeast9 (not the surrogate), for one
candidate. Holding the learned physiology + edit mapping fixed and only
swapping surrogate-vs-real-solver isolates the surrogate's own
approximation error from physiology/edit-mapping error (Section: Step 4 of
the virtual-scorer repair pass).

Reuses apply_predicted_interface_controls unmodified (same function already
used for the G7 validation in LEARNER_ARCHITECTURE_REPAIR.md) -- it takes
only the compact 6-key control dict, no z_* input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

DT_REAL = 0.25
BETA_DEG_BASE = 0.002


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    scorer = scorer_mod.RepairedVirtualScorer.load(Path(args.checkpoint), env_mean, env_std)

    env = candidate["environment"]
    env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
    env_std_in = (env_raw - scorer.env_mean) / scorer.env_std
    cache = tss.forward(scorer.params, env_std_in, scorer_mod.T_LEN)
    c_pre = cache.z[:-1, 0, :] @ scorer.params.Wc + scorer.params.bc
    c_baseline = c_pre * scorer.surrogate.control_std + scorer.surrogate.control_mean
    c_edited, edit_notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, candidate.get("edits", []))
    clip_lo = scorer.surrogate.control_mean - scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    clip_hi = scorer.surrogate.control_mean + scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    c_final = np.clip(c_edited, clip_lo, clip_hi)

    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    cfg = gem.GEMCultureConfig(temperature=float(env["temperature"]), pH=float(env["pH"]), DO=float(env["DO"]))

    n_intervals = scorer_mod.T_LEN - 1
    X = np.zeros(scorer_mod.T_LEN)
    B = np.zeros(scorer_mod.T_LEN)
    X[0], B[0] = scorer_mod.X0, scorer_mod.B0
    infeasible = 0
    for t in range(n_intervals):
        interval_model = augmented.copy()
        controls = {col: float(c_final[t, i]) for i, col in enumerate(sur.CONTROL_COLUMNS)}
        prod.apply_predicted_interface_controls(interval_model, cfg, controls)
        try:
            flux = gem.solve_staged(interval_model, cfg, gamma=float(np.clip(controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")
        except Exception:
            infeasible += 1
            X[t + 1], B[t + 1] = X[t], B[t]
            continue
        biomass_flux = float(flux["biomass_flux"])
        beta_flux = float(flux["beta_carotene_flux"])
        X[t + 1] = max(1e-9, X[t] + DT_REAL * biomass_flux * X[t])
        B[t + 1] = max(0.0, B[t] + DT_REAL * beta_flux * X[t] - DT_REAL * BETA_DEG_BASE * B[t])

    summary = pd.DataFrame(
        [
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_hash": candidate["candidate_hash"],
                "final_product_pred_controls_via_real_gsm": float(B[-1]),
                "final_biomass_pred_controls_via_real_gsm": float(X[-1]),
                "n_infeasible_intervals": infeasible,
                "n_intervals": n_intervals,
            }
        ]
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
