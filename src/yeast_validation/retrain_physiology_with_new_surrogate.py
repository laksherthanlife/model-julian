#!/usr/bin/env python3
"""Stage 5: retrain the physiology model THROUGH the new MLP surrogate,
justified by the Stage 5 gate check (Jacobian comparison in
data/stage5_jacobian_comparison.md): the old and new surrogates' gradients
w.r.t. the control vector differ materially in the region the physiology
model actually visits (beta_carotene_flux gradient cosine similarity ~0.49
with high variance, biomass_flux gradient magnitude ~5.7x different).

Same frozen architecture/config/seed/clean reference-strain training data as
the existing hybrid_learned_physiology__full__seed11 checkpoint -- only the
frozen surrogate injected into BPTT changes. No retuning of D_model, hidden
width, epochs, or learning rate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "results" / "repaired_hybrid_training" / "checkpoints"
NEW_SURROGATE_PATH = CHECKPOINTS / "gsm_surrogate_mlp_v1.npz"
SEED = 11
VARIANT = "hybrid_learned_physiology"
REPORTER_MODE = "full"


def main() -> None:
    tic = time.perf_counter()
    dataset = train_mod.load_dataset()
    new_surrogate = mlp.GSMSurrogateMLP.load(NEW_SURROGATE_PATH)
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)

    stats: dict = {}
    trained = train_mod.train_state_space_variant(
        dataset, VARIANT, REPORTER_MODE, SEED, new_surrogate, env_mean, env_std, stats,
        epochs=train_mod.EPOCHS, lr=train_mod.LR,
    )
    metrics = train_mod.evaluate_state_space(trained, dataset, new_surrogate)

    ckpt_path = CHECKPOINTS / f"{VARIANT}__{REPORTER_MODE}__seed{SEED}__new_surrogate.npz"
    tss.save_checkpoint(
        ckpt_path, trained["params"],
        {
            "variant": VARIANT, "reporter_mode": REPORTER_MODE, "seed": SEED,
            "epochs": train_mod.EPOCHS, "lr": train_mod.LR,
            "hidden_dim": train_mod.HIDDEN_DIM, "d_model": train_mod.D_MODEL,
            "final_loss": trained["loss_history"][-1],
            "surrogate": "gsm_surrogate_mlp_v1",
        },
    )
    metrics.to_csv(DATA / "stage5_new_surrogate_training_metrics.csv", index=False)

    print(f"final_loss={trained['loss_history'][-1]:.5f}")
    print(metrics.to_string(index=False))
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Done in {time.perf_counter() - tic:.1f}s")


if __name__ == "__main__":
    main()
