#!/usr/bin/env python3
"""Train and evaluate the repaired learner architecture on the existing clean
G3 baseline dataset (Section 11 of the learner-repair handover: "first test
the repaired model on the existing clean D_true~=current G3 dataset" before
returning to the latent-capacity-mismatch generator work).

Model variants (Section 13):
  - direct_blackbox: (environment, time) -> outputs, closed-form ridge, no
    latent state at all. Tests whether any dynamical/hybrid structure is
    needed for prediction at all.
  - product_only_state_space: trainable z(t) (gem_trainable_state_space),
    direct linear heads for product/biomass, NO reporters, NO surrogate/GSM
    coupling. Negative control for the reporter question.
  - hybrid_learned_physiology (PRIMARY): trainable z(t); reporter heads read
    directly from the shared z(t); product/biomass are NOT given a direct
    head -- they are produced only via a control head c(t) -> the frozen
    differentiable GSM surrogate (gem_gsm_surrogate) -> one-step
    teacher-forced mechanistic integration. This is what makes the model
    genuinely hybrid rather than "a state-space model that happens to also
    predict beta-carotene" (Section 8).

All three respect the clean, leak-free training contract
(gem_clean_teacher_targets.py): deployment inputs are environment only;
supervision is noisy observed product/biomass/glucose plus partial reporter
coverage. No raw z_*, no internal GEM bounds, no internal LP fluxes are ever
a training target.

D_model is fixed at 16 throughout (frozen per the handover spec, Section 5).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import design_benchmark_exact as dbe  # noqa: E402  (reuse spearman_without_scipy)
import gem_clean_teacher_targets as targets  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_observation_noise as noise  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402  (reuse _pivot_time, _add_random_smooth_reporter)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "repaired_hybrid_training"
CHECKPOINTS = RESULTS / "checkpoints"

MODEL_SEEDS = [11, 22, 33, 44, 55]
D_MODEL = 16
HIDDEN_DIM = 24  # transition MLP width; a separate, smaller, not-tuned-against-D_true architecture choice
EPOCHS = 600
LR = 0.02
LAMBDA_DYN = 2e-4
REPORTER_COLS = list(targets.CORE_REPORTER_COLUMNS)  # R_ox, R_atp, R_E_PSY, R_E_DES, R_E_CYC


def load_dataset(apply_noise: bool = True, noise_seed: int = 20260819) -> dict[str, object]:
    traj = pd.read_csv(DATA / "gem_state_space_trajectories.csv")
    reporters = pd.read_csv(DATA / "gem_state_space_reporters.csv")
    if apply_noise:
        traj = noise.apply_observation_noise(traj, seed=noise_seed)

    ids = sorted(traj["culture_id"].unique())
    meta_rows = []
    for cid in ids:
        row = traj[traj["culture_id"].eq(cid)].sort_values("time_index").iloc[0]
        meta_rows.append({"culture_id": cid, "environment_id": row["environment_id"], "split": row["split"], "temperature": row["temperature"], "pH": row["pH"], "DO": row["DO"]})
    meta = pd.DataFrame(meta_rows)

    traj_cols = [c for c in targets.TRAJECTORY_OBSERVED_COLUMNS if c in traj.columns] if apply_noise else [c for c in targets.TRAJECTORY_BASE_COLUMNS if c in traj.columns]
    traj_mats = prod._pivot_time(traj, ids, traj_cols)  # each (n_cultures, T)
    reporter_mats = prod._pivot_time(reporters, ids, REPORTER_COLS)
    t_len = traj_mats[traj_cols[0]].shape[1]
    prod._add_random_smooth_reporter(reporter_mats, t_len, len(ids))

    # This module's convention throughout is (T, B) -- transpose once here.
    traj_mats = {k: v.T for k, v in traj_mats.items()}
    reporter_mats = {k: v.T for k, v in reporter_mats.items()}

    time_values = traj[traj["culture_id"].eq(ids[0])].sort_values("time_index")["time"].to_numpy(float)
    return {
        "meta": meta,
        "env_raw": meta[targets.DEPLOYMENT_INPUTS].to_numpy(float),
        "splits": meta["split"].to_numpy(str),
        "time": time_values,
        "t_len": t_len,
        "product": traj_mats[traj_cols[0]],  # B_total_observed (or B_total), (T,B)
        "biomass": traj_mats[traj_cols[1]],  # X_observed (or X), (T,B)
        "reporters": reporter_mats,
        "culture_ids": ids,
    }


def standardize_fit(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For (B, feature) arrays such as env_raw -- mask selects rows (axis 0)."""
    return x[mask].mean(axis=0), x[mask].std(axis=0) + 1e-9


def standardize_fit_series(x_tb: np.ndarray, culture_mask: np.ndarray) -> tuple[float, float]:
    """For (T, B) trajectory/reporter arrays -- a single scalar mean/std over
    all (time, train-culture) entries, matching the existing scalar
    target_mean/std convention used elsewhere in this codebase."""
    vals = x_tb[:, culture_mask]
    return float(vals.mean()), float(vals.std() + 1e-9)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return dbe.spearman_without_scipy(pd.Series(a), pd.Series(b))


def fit_frozen_surrogate() -> sur.GSMSurrogate:
    pairs = sur.load_surrogate_training_pairs(
        DATA / "gem_state_space_generator_constraints.partial.csv",
        DATA / "gem_state_space_fluxes.csv",
        DATA / "gem_state_space_split_manifest.csv",
    )
    train_mask = pairs["split"].eq("train").to_numpy()
    return sur.fit_surrogate(pairs, train_mask=train_mask, ridge_lambda=1.0)


def direct_blackbox_fit(dataset: dict[str, object], env_mean: np.ndarray, env_std: np.ndarray, train_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Closed-form ridge (environment, time) -> outputs. No latent state."""
    env = (dataset["env_raw"] - env_mean) / env_std
    T = dataset["t_len"]
    time_norm = np.linspace(0.0, 1.0, T)
    rows_env = np.repeat(env, T, axis=0)
    rows_time = np.tile(time_norm, env.shape[0])[:, None]
    phi = np.concatenate([np.ones((len(rows_env), 1)), rows_env, rows_env**2, rows_time, rows_time**2], axis=1)
    row_train_mask = np.repeat(train_mask, T)

    heads: dict[str, np.ndarray] = {}
    for name, mat in [("product", dataset["product"]), ("biomass", dataset["biomass"])] + [(col, dataset["reporters"][col]) for col in REPORTER_COLS]:
        y = mat.T.reshape(-1, 1)  # mat is (T,B); flatten culture-major to match rows_env's ordering
        eye = np.eye(phi.shape[1])
        eye[0, 0] = 0.0
        W = np.linalg.solve(phi[row_train_mask].T @ phi[row_train_mask] + 1.0 * eye, phi[row_train_mask].T @ y[row_train_mask])
        heads[name] = W
    return {"W": heads, "env_mean": env_mean, "env_std": env_std, "T": T}


def direct_blackbox_predict(model: dict[str, object], env_raw: np.ndarray, col: str) -> np.ndarray:
    env = (env_raw - model["env_mean"]) / model["env_std"]
    T = model["T"]
    time_norm = np.linspace(0.0, 1.0, T)
    rows_env = np.repeat(env, T, axis=0)
    rows_time = np.tile(time_norm, env.shape[0])[:, None]
    phi = np.concatenate([np.ones((len(rows_env), 1)), rows_env, rows_env**2, rows_time, rows_time**2], axis=1)
    pred = (phi @ model["W"][col]).reshape(env.shape[0], T)
    return pred.T  # (T, B)


def shuffle_columns(mats: dict[str, np.ndarray], cols: list[str], train_mask: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in mats.items()}
    rng = np.random.default_rng(seed)
    idx = np.where(train_mask)[0]
    perm = idx.copy()
    rng.shuffle(perm)
    if len(perm) > 1:
        perm = np.roll(perm, 1)
    for col in cols:
        out[col][:, idx] = out[col][:, perm]
    return out


def train_state_space_variant(
    dataset: dict[str, object],
    variant: str,
    reporter_mode: str,
    seed: int,
    surrogate: sur.GSMSurrogate | None,
    env_mean: np.ndarray,
    env_std: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int = EPOCHS,
    lr: float = LR,
) -> dict[str, object]:
    env = (dataset["env_raw"] - env_mean) / env_std
    train_mask = dataset["splits"] == "train"
    T = dataset["t_len"]

    reporter_mats = dataset["reporters"]
    if variant != "hybrid_learned_physiology":
        # product_only_state_space (and any future non-hybrid variant) is the
        # explicit "no reporters" negative control -- reporter_mode is only
        # meaningful for hybrid_learned_physiology.
        reporter_cols_used: list[str] = []
    elif reporter_mode == "none":
        reporter_cols_used = []
    elif reporter_mode == "random":
        reporter_cols_used = ["R_random_smooth"]
    else:
        if reporter_mode == "shuffled":
            reporter_mats = shuffle_columns(reporter_mats, REPORTER_COLS, train_mask, seed)
        reporter_cols_used = REPORTER_COLS

    head_targets: dict[str, np.ndarray] = {}
    head_masks: dict[str, np.ndarray] = {}
    for col in reporter_cols_used:
        mean, std = stats.setdefault(col, standardize_fit_series(reporter_mats[col], train_mask))
        head_targets[col] = (reporter_mats[col] - mean) / std
        head_masks[col] = np.tile(train_mask.astype(float), (T, 1))

    surrogate_kwargs = {}
    n_controls = 0
    if variant == "hybrid_learned_physiology":
        n_controls = len(sur.CONTROL_COLUMNS)
        X_mean, X_std = stats.setdefault("biomass_raw", standardize_fit_series(dataset["biomass"], train_mask))
        B_mean, B_std = stats.setdefault("product_raw", standardize_fit_series(dataset["product"], train_mask))
        surrogate_kwargs = dict(
            surrogate=surrogate,
            control_mean=surrogate.control_mean,
            control_std=surrogate.control_std,
            surrogate_out_index=sur.SURROGATE_OUT_INDEX,
            X_true=dataset["biomass"],
            B_true=dataset["product"],
            X_std=X_std,
            B_std=B_std,
            dt_real=0.25,
            beta_deg_base=0.002,
        )
    else:
        for col, mat in [("product", dataset["product"]), ("biomass", dataset["biomass"])]:
            mean, std = stats.setdefault(col, standardize_fit_series(mat, train_mask))
            head_targets[col] = (mat - mean) / std
            head_masks[col] = np.tile(train_mask.astype(float), (T, 1))

    params = tss.init_params(env.shape[1], D_MODEL, HIDDEN_DIM, list(head_targets.keys()), n_controls, seed=seed)
    adam = tss.AdamState(params, lr=lr)
    loss_history = []
    for epoch in range(epochs):
        loss, grads, diag = tss.compute_loss_and_grads(
            params, env, T, head_targets, head_masks, lambda_dyn=LAMBDA_DYN, **surrogate_kwargs
        )
        adam.step(params, grads)
        loss_history.append(loss)
    return {
        "params": params,
        "loss_history": loss_history,
        "head_targets": head_targets,
        "head_masks": head_masks,
        "surrogate_kwargs": surrogate_kwargs,
        "stats": stats,
        "env": env,
        "reporter_cols_used": reporter_cols_used,
        "variant": variant,
        "reporter_mode": reporter_mode,
        "seed": seed,
    }


def evaluate_state_space(trained: dict[str, object], dataset: dict[str, object], surrogate: sur.GSMSurrogate | None) -> pd.DataFrame:
    params = trained["params"]
    env = trained["env"]
    T = dataset["t_len"]
    cache = tss.forward(params, env, T)
    rows = []

    def channel_metric(name: str, pred: np.ndarray, true_raw: np.ndarray, split: str) -> None:
        mask = dataset["splits"] == split
        train_mask = dataset["splits"] == "train"
        train_true = true_raw[:, train_mask]
        denom = float(np.std(train_true) + 1e-9)
        rows.append(
            {
                "variant": trained["variant"],
                "reporter_mode": trained["reporter_mode"],
                "seed": trained["seed"],
                "split": split,
                "channel": name,
                "normalized_rmse": float(np.sqrt(np.mean((pred[:, mask] - true_raw[:, mask]) ** 2)) / denom),
                "final_timepoint_abs_error_mean": float(np.mean(np.abs(pred[-1, mask] - true_raw[-1, mask]))),
            }
        )

    for col in [c for c in trained["head_targets"] if c not in ("product", "biomass")]:
        Wh, bh = params.heads[col]
        y_pre = (cache.z @ Wh + bh)[..., 0]
        mean, std = trained["stats"][col]
        pred = y_pre * std + mean
        true_raw = dataset["reporters"][col] if col in dataset["reporters"] else None
        if true_raw is None:
            continue
        for split in sorted(set(dataset["splits"])):
            channel_metric(col, pred, true_raw, split)

    if "product" in trained["head_targets"]:
        for name, mat in [("product", dataset["product"]), ("biomass", dataset["biomass"])]:
            Wh, bh = params.heads[name]
            y_pre = (cache.z @ Wh + bh)[..., 0]
            mean, std = trained["stats"][name]
            pred = y_pre * std + mean
            for split in sorted(set(dataset["splits"])):
                channel_metric(name, pred, mat, split)
    else:
        c_pre = cache.z[:-1] @ params.Wc + params.bc
        c = c_pre * surrogate.control_std + surrogate.control_mean
        pred_X = np.zeros((T, env.shape[0]))
        pred_B = np.zeros((T, env.shape[0]))
        for t in range(T - 1):
            values = surrogate.predict(c[t], dataset["env_raw"])
            biomass_flux = values[:, sur.SURROGATE_OUT_INDEX["biomass_flux"]]
            beta_flux = values[:, sur.SURROGATE_OUT_INDEX["beta_carotene_flux"]]
            X_true_t = dataset["biomass"][t]
            B_true_t = dataset["product"][t]
            pred_X[t + 1] = X_true_t + 0.25 * biomass_flux * X_true_t
            pred_B[t + 1] = B_true_t + 0.25 * beta_flux * X_true_t - 0.25 * 0.002 * B_true_t
        pred_X[0] = dataset["biomass"][0]
        pred_B[0] = dataset["product"][0]
        for split in sorted(set(dataset["splits"])):
            channel_metric("product", pred_B, dataset["product"], split)
            channel_metric("biomass", pred_X, dataset["biomass"], split)

    return pd.DataFrame(rows)


def ranking_correlation(pred_final: np.ndarray, true_final: np.ndarray, splits: np.ndarray, split_name: str) -> float:
    mask = splits == split_name
    if mask.sum() < 3:
        return float("nan")
    return spearman(pred_final[mask], true_final[mask])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=MODEL_SEEDS)
    parser.add_argument("--reporter-modes", type=str, nargs="+", default=["full"])
    parser.add_argument("--variants", type=str, nargs="+", default=["direct_blackbox", "product_only_state_space", "hybrid_learned_physiology"])
    args = parser.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    tic = time.perf_counter()
    dataset = load_dataset()
    surrogate = fit_frozen_surrogate()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = standardize_fit(dataset["env_raw"], train_mask)

    all_metrics = []
    manifest_rows = []
    for variant in args.variants:
        reporter_modes = args.reporter_modes if variant == "hybrid_learned_physiology" else ["full"]
        for reporter_mode in reporter_modes:
            for seed in args.seeds:
                stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                if variant == "direct_blackbox":
                    model = direct_blackbox_fit(dataset, env_mean, env_std, train_mask)
                    rows = []
                    for name, mat in [("product", dataset["product"]), ("biomass", dataset["biomass"])] + [(c, dataset["reporters"][c]) for c in REPORTER_COLS]:
                        pred = direct_blackbox_predict(model, dataset["env_raw"], name)
                        for split in sorted(set(dataset["splits"])):
                            mask = dataset["splits"] == split
                            train_true = mat[:, train_mask]
                            denom = float(np.std(train_true) + 1e-9)
                            rows.append({"variant": variant, "reporter_mode": "n/a", "seed": seed, "split": split, "channel": name, "normalized_rmse": float(np.sqrt(np.mean((pred[:, mask] - mat[:, mask]) ** 2)) / denom), "final_timepoint_abs_error_mean": float(np.mean(np.abs(pred[-1, mask] - mat[-1, mask])))})
                    all_metrics.append(pd.DataFrame(rows))
                    manifest_rows.append({"variant": variant, "reporter_mode": "n/a", "seed": seed, "epochs": 0, "final_loss": float("nan"), "params_fingerprint": "closed_form_ridge_no_recurrent_state"})
                    continue

                trained = train_state_space_variant(dataset, variant, reporter_mode, seed, surrogate, env_mean, env_std, stats, epochs=args.epochs)
                metrics = evaluate_state_space(trained, dataset, surrogate)
                all_metrics.append(metrics)
                ckpt_path = CHECKPOINTS / f"{variant}__{reporter_mode}__seed{seed}.npz"
                tss.save_checkpoint(ckpt_path, trained["params"], {"variant": variant, "reporter_mode": reporter_mode, "seed": seed, "epochs": args.epochs, "lr": LR, "hidden_dim": HIDDEN_DIM, "d_model": D_MODEL, "final_loss": trained["loss_history"][-1]})
                manifest_rows.append({"variant": variant, "reporter_mode": reporter_mode, "seed": seed, "epochs": args.epochs, "final_loss": trained["loss_history"][-1], "checkpoint": str(ckpt_path.relative_to(ROOT)), "params_fingerprint": tss.params_fingerprint(trained["params"])})
                print(f"{variant} / {reporter_mode} / seed={seed}: final_loss={trained['loss_history'][-1]:.5f}")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)
    metrics_df.to_csv(DATA / "repaired_hybrid_training_metrics.csv", index=False)
    manifest_df.to_csv(DATA / "repaired_hybrid_training_manifest.csv", index=False)
    print(f"Done in {time.perf_counter() - tic:.1f}s. Wrote {len(metrics_df)} metric rows, {len(manifest_df)} manifest rows.")


if __name__ == "__main__":
    main()
