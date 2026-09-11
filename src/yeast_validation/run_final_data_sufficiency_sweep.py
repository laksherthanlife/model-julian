#!/usr/bin/env python3
"""Final validation: train the repaired hybrid learner on whole-culture subsets.

This script varies only the number of independent physical culture
trajectories available to the learned physiology component. It keeps the v2
GSM MLP surrogate fixed because that layer is trained from in-silico Yeast9
solves, not additional physical cultures.
"""

from __future__ import annotations

import argparse
import json
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
RESULTS = ROOT / "results" / "final_validation_data_sufficiency"
CHECKPOINTS = RESULTS / "checkpoints"
SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"

TRAINING_COUNTS = [77, 58, 39, 20, 10]
SUBSAMPLE_SEEDS = [101, 202, 303]
MODEL_SEED = 11


def split_dataset_for_subset(dataset: dict[str, object], selected_train_ids: set[str]) -> dict[str, object]:
    out = dict(dataset)
    meta = dataset["meta"].copy()
    splits = np.array(dataset["splits"], dtype=object).copy()
    culture_ids = list(dataset["culture_ids"])
    for i, cid in enumerate(culture_ids):
        if splits[i] == "train" and cid not in selected_train_ids:
            splits[i] = "train_unused"
    meta["split"] = splits
    out["meta"] = meta
    out["splits"] = splits.astype(str)
    return out


def subset_ids(train_ids: list[str], n: int, seed: int) -> list[str]:
    if n == len(train_ids):
        return list(train_ids)
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(train_ids, size=n, replace=False).tolist())
    return picked


def metric_summary(metrics: pd.DataFrame, n_train: int, subsample_seed: int, checkpoint: Path, stats_path: Path) -> pd.DataFrame:
    rows = metrics.copy()
    rows.insert(0, "training_cultures_n", n_train)
    rows.insert(1, "subsample_seed", subsample_seed)
    rows.insert(2, "model_seed", MODEL_SEED)
    rows["checkpoint"] = str(checkpoint.relative_to(ROOT))
    rows["normalization_stats"] = str(stats_path.relative_to(ROOT))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=TRAINING_COUNTS)
    parser.add_argument("--subsample-seeds", type=int, nargs="+", default=SUBSAMPLE_SEEDS)
    parser.add_argument("--epochs", type=int, default=train_mod.EPOCHS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    dataset = train_mod.load_dataset()
    train_ids = [cid for cid, split in zip(dataset["culture_ids"], dataset["splits"]) if split == "train"]
    counts = sorted(set(args.counts), reverse=True)
    if any(n < 2 or n > len(train_ids) for n in counts):
        raise ValueError(f"counts must be in [2, {len(train_ids)}], got {counts}")
    surrogate = mlp.GSMSurrogateMLP.load(SURROGATE_PATH)

    manifest_rows: list[dict[str, object]] = []
    all_metrics: list[pd.DataFrame] = []
    started = time.perf_counter()
    for n_train in counts:
        seeds = [0] if n_train == len(train_ids) else args.subsample_seeds
        for subsample_seed in seeds:
            selected = subset_ids(train_ids, n_train, subsample_seed)
            ds = split_dataset_for_subset(dataset, set(selected))
            train_mask = ds["splits"] == "train"
            env_mean, env_std = train_mod.standardize_fit(ds["env_raw"], train_mask)
            stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            ckpt = CHECKPOINTS / f"hybrid_learned_physiology__full__N{n_train:03d}__subseed{subsample_seed}__seed{MODEL_SEED}.npz"
            stats_path = CHECKPOINTS / f"hybrid_learned_physiology__full__N{n_train:03d}__subseed{subsample_seed}__seed{MODEL_SEED}__norm.npz"
            if ckpt.exists() and stats_path.exists() and not args.force:
                params, meta = tss.load_checkpoint(ckpt)
                trained = {
                    "params": params,
                    "loss_history": [float(meta.get("final_loss", np.nan))],
                    "head_targets": {},
                    "head_masks": {},
                    "surrogate_kwargs": {},
                    "stats": {},
                    "env": (ds["env_raw"] - env_mean) / env_std,
                    "reporter_cols_used": train_mod.REPORTER_COLS,
                    "variant": "hybrid_learned_physiology",
                    "reporter_mode": "full",
                    "seed": MODEL_SEED,
                }
                # Reconstruct only the fields evaluate_state_space needs for
                # reporter heads and product/biomass mechanistic rollout.
                stat_npz = np.load(stats_path)
                trained["stats"] = {k[:-6]: (stat_npz[k], stat_npz[k[:-6] + "__std"]) for k in stat_npz.files if k.endswith("__mean")}
                trained["head_targets"] = {col: np.empty((0, 0)) for col in train_mod.REPORTER_COLS}
                print(f"reuse checkpoint N={n_train} subsample_seed={subsample_seed}")
            else:
                trained = train_mod.train_state_space_variant(
                    ds,
                    "hybrid_learned_physiology",
                    "full",
                    MODEL_SEED,
                    surrogate,
                    env_mean,
                    env_std,
                    stats,
                    epochs=args.epochs,
                    lr=train_mod.LR,
                )
                meta = {
                    "variant": "hybrid_learned_physiology",
                    "reporter_mode": "full",
                    "seed": MODEL_SEED,
                    "epochs": args.epochs,
                    "lr": train_mod.LR,
                    "hidden_dim": train_mod.HIDDEN_DIM,
                    "d_model": train_mod.D_MODEL,
                    "lambda_dyn": train_mod.LAMBDA_DYN,
                    "training_cultures_n": n_train,
                    "subsample_seed": subsample_seed,
                    "selected_train_culture_ids": selected,
                    "fixed_gsm_surrogate": str(SURROGATE_PATH.relative_to(ROOT)),
                    "final_loss": float(trained["loss_history"][-1]),
                }
                tss.save_checkpoint(ckpt, trained["params"], meta)
                stat_arrays = {"env_mean": env_mean, "env_std": env_std}
                for key, (mean, std) in trained["stats"].items():
                    stat_arrays[f"{key}__mean"] = np.asarray(mean)
                    stat_arrays[f"{key}__std"] = np.asarray(std)
                np.savez_compressed(stats_path, **stat_arrays)
                print(f"trained N={n_train} subsample_seed={subsample_seed} final_loss={trained['loss_history'][-1]:.6f}")

            metrics = train_mod.evaluate_state_space(trained, ds, surrogate)
            all_metrics.append(metric_summary(metrics, n_train, subsample_seed, ckpt, stats_path))
            manifest_rows.append(
                {
                    "training_cultures_n": n_train,
                    "subsample_seed": subsample_seed,
                    "model_seed": MODEL_SEED,
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                    "normalization_stats": str(stats_path.relative_to(ROOT)),
                    "selected_train_culture_ids_json": json.dumps(selected),
                    "held_out_training_cultures_unused": len(train_ids) - n_train,
                    "epochs": args.epochs,
                    "architecture_retuned": False,
                    "fixed_gsm_surrogate": str(SURROGATE_PATH.relative_to(ROOT)),
                    "params_fingerprint": tss.params_fingerprint(trained["params"]),
                    "final_loss": float(trained["loss_history"][-1]),
                }
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(DATA / "final_data_sufficiency_predictive_metrics.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DATA / "final_data_sufficiency_training_manifest.csv", index=False)
    print(f"wrote {len(metrics_df)} metric rows and {len(manifest)} training rows in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
