#!/usr/bin/env python3
"""Diagnose why reporter supervision hurt active-rxncon product prediction.

Learner-only diagnostic.  Reuses the completed active-rxncon structural
validation dataset and hidden lockbox tables for post-hoc analyses only.  It
does not regenerate Yeast9 cultures or modify the biological generator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_trainable_state_space as tss  # noqa: E402
import run_rxncon_active_structural_validation as active  # noqa: E402
import run_rxncon_external_complexity_screen as rxncon_base  # noqa: E402


OUT_DIR = ROOT / "data" / "rxncon_reporter_supervision_diagnostic"
FIG_DIR = ROOT / "figures"
ACTIVE_DIR = active.OUT_DIR
N_TIMEPOINTS = active.N_TIMEPOINTS
DT = active.DT
REPORTERS = active.REPORTER_COLS
PRIMARY = ["B_total", "X"]
SEEDS = [11, 22, 33, 44, 55]
LAMBDA_GRID = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
LATENT_DIMS = [4, 8, 16, 32]
SUBSAMPLE_N = [10, 25, 40, 50, 69]
SUBSAMPLE_SEEDS = [101, 202, 303, 404, 505]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def shared_arrays(params: tss.StateSpaceParams) -> list[np.ndarray]:
    return [params.Wg0, params.bg0, params.Wz, params.We, params.b1, params.Wo, params.bo]


def shared_norm(grads: tss.StateSpaceParams) -> float:
    return float(np.sqrt(sum(np.sum(g**2) for g in shared_arrays(grads))))


def shared_dot(a: tss.StateSpaceParams, b: tss.StateSpaceParams) -> float:
    return float(sum(np.sum(x * y) for x, y in zip(shared_arrays(a), shared_arrays(b))))


def shared_cosine(a: tss.StateSpaceParams, b: tss.StateSpaceParams) -> float:
    na = shared_norm(a)
    nb = shared_norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return shared_dot(a, b) / (na * nb)


def zero_like(params: tss.StateSpaceParams) -> tss.StateSpaceParams:
    z = params.copy()
    for arr in [z.Wg0, z.bg0, z.Wz, z.We, z.b1, z.Wo, z.bo]:
        arr[:] = 0.0
    for w, b in z.heads.values():
        w[:] = 0.0
        b[:] = 0.0
    if z.Wc is not None:
        z.Wc[:] = 0.0
        z.bc[:] = 0.0
    return z


def add_scaled(dst: tss.StateSpaceParams, src: tss.StateSpaceParams, scale: float) -> None:
    for name in ("Wg0", "bg0", "Wz", "We", "b1", "Wo", "bo"):
        getattr(dst, name)[:] += scale * getattr(src, name)
    for head, (sw, sb) in src.heads.items():
        if head in dst.heads:
            dw, db = dst.heads[head]
            dw[:] += scale * sw
            db[:] += scale * sb
    if dst.Wc is not None and src.Wc is not None:
        dst.Wc[:] += scale * src.Wc
        dst.bc[:] += scale * src.bc


def copy_params(params: tss.StateSpaceParams) -> tss.StateSpaceParams:
    return params.copy()


def dataset() -> dict[str, object]:
    return active.model_dataset()


def standardize_env(env: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    return active.standardize_env(env, train_mask)[0]


def channel_stats(channels: dict[str, np.ndarray], train_mask: np.ndarray) -> dict[str, tuple[float, float]]:
    return {name: active.train_stats(arr, train_mask) for name, arr in channels.items()}


def standardized_targets(channels: dict[str, np.ndarray], stats: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    return {name: (arr - stats[name][0]) / stats[name][1] for name, arr in channels.items()}


def mask_for(train_mask: np.ndarray) -> np.ndarray:
    return np.tile(train_mask.astype(float), (N_TIMEPOINTS, 1))


def task_grad(params, env_z, targets, masks, channel: str, lambda_dyn: float = 2e-4):
    loss, grads, diag = tss.compute_loss_and_grads(
        params,
        env_z,
        N_TIMEPOINTS,
        {channel: targets[channel]},
        {channel: masks[channel]},
        lambda_dyn=lambda_dyn,
    )
    return float(diag[f"loss__{channel}"]), grads


def combined_loss_grads(
    params,
    env_z,
    targets,
    masks,
    reporter_set: list[str],
    lambda_r: float,
    reporter_mode: str,
    lambda_dyn: float = 2e-4,
):
    weights = {ch: 1.0 for ch in PRIMARY}
    for r in reporter_set:
        weights[r] = lambda_r if reporter_mode == "sum" else lambda_r / max(len(reporter_set), 1)
    return weighted_loss_and_grads(params, env_z, targets, masks, weights, lambda_dyn=lambda_dyn)


def weighted_loss_and_grads(params, env, targets, masks, weights: dict[str, float], lambda_dyn: float = 2e-4):
    """Single-pass weighted direct-head objective for the diagnostic runs."""
    cache = tss.forward(params, env, N_TIMEPOINTS)
    t_len, n_batch, latent_dim = cache.z.shape
    grads = zero_like(params)
    dz_grad = np.zeros_like(cache.z)
    losses = {}
    total = 0.0
    for col, weight in weights.items():
        if weight == 0 or col not in params.heads:
            continue
        wh, bh = params.heads[col]
        y_pre = (cache.z @ wh + bh)[..., 0]
        mask = masks[col]
        n = max(float(np.sum(mask)), 1.0)
        diff = (y_pre - targets[col]) * mask
        col_loss = float(np.sum(diff**2)) / n
        losses[col] = col_loss
        total += weight * col_loss
        dldy = weight * (2.0 / n) * diff
        gw, gb = grads.heads[col]
        gw += np.einsum("tbd,tb->d", cache.z, dldy).reshape(latent_dim, 1)
        gb += np.array([np.sum(dldy)])
        dz_grad += dldy[..., None] * wh[:, 0][None, None, :]
    rep_losses = [losses[r] for r in REPORTERS if r in losses]
    losses["reporter_mean"] = float(np.mean(rep_losses)) if rep_losses else np.nan
    losses["reporter_sum"] = float(np.sum(rep_losses)) if rep_losses else 0.0
    dyn = lambda_dyn * float(np.mean(cache.dz_pre**2))
    total += dyn
    losses["dyn_reg"] = dyn
    losses["total"] = float(total)

    g = dz_grad[t_len - 1].copy()
    for t in range(t_len - 2, -1, -1):
        pre_clip = cache.z[t] + cache.dt_latent * cache.dz[t]
        clip_mask = (np.abs(pre_clip) < tss.Z_CLIP).astype(float)
        g_clipped = g * clip_mask
        dld_dz = g_clipped * cache.dt_latent
        dld_dz_pre = dld_dz * (1 - cache.dz[t] ** 2)
        dld_dz_pre += lambda_dyn * 2.0 * cache.dz_pre[t] / cache.dz_pre.size
        grads.Wo += np.einsum("bh,bd->hd", cache.h[t], dld_dz_pre)
        grads.bo += np.sum(dld_dz_pre, axis=0)
        dld_h = dld_dz_pre @ params.Wo.T
        dld_h_pre = dld_h * (1 - cache.h[t] ** 2)
        grads.Wz += np.einsum("bd,bh->dh", cache.z[t], dld_h_pre)
        grads.We += np.einsum("be,bh->eh", env, dld_h_pre)
        grads.b1 += np.sum(dld_h_pre, axis=0)
        g = g_clipped + dld_h_pre @ params.Wz.T + dz_grad[t]
    dld_a0 = g * (1 - cache.z[0] ** 2)
    grads.Wg0 += np.einsum("be,bd->ed", env, dld_a0)
    grads.bg0 += np.sum(dld_a0, axis=0)
    return float(total), grads, losses


def combined_loss_grads_slow(
    params,
    env_z,
    targets,
    masks,
    reporter_set: list[str],
    lambda_r: float,
    reporter_mode: str,
    lambda_dyn: float = 2e-4,
):
    total_grads = zero_like(params)
    losses = {}
    total = 0.0
    for ch in PRIMARY:
        loss, grads = task_grad(params, env_z, targets, masks, ch, lambda_dyn=0.0)
        losses[ch] = loss
        total += loss
        add_scaled(total_grads, grads, 1.0)
    rep_losses = []
    for r in reporter_set:
        loss, grads = task_grad(params, env_z, targets, masks, r, lambda_dyn=0.0)
        losses[r] = loss
        rep_losses.append(loss)
        weight = lambda_r if reporter_mode == "sum" else lambda_r / max(len(reporter_set), 1)
        add_scaled(total_grads, grads, weight)
    total += lambda_r * (sum(rep_losses) if reporter_mode == "sum" else np.mean(rep_losses) if rep_losses else 0.0)
    loss_dyn, dyn_grads, diag = tss.compute_loss_and_grads(params, env_z, N_TIMEPOINTS, {}, {}, lambda_dyn=lambda_dyn)
    dyn = float(diag["loss__dyn_reg"])
    total += dyn
    add_scaled(total_grads, dyn_grads, 1.0)
    losses["reporter_mean"] = float(np.mean(rep_losses)) if rep_losses else np.nan
    losses["reporter_sum"] = float(np.sum(rep_losses)) if rep_losses else 0.0
    losses["dyn_reg"] = dyn
    losses["total"] = float(total)
    return float(total), total_grads, losses


def train_model(
    ds: dict[str, object],
    *,
    reporter_set: list[str],
    lambda_r: float,
    reporter_mode: str,
    seed: int,
    latent_dim: int = 16,
    epochs: int = 240,
    train_ids: np.ndarray | None = None,
    lr: float = 0.018,
    log_every: int = 5,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = ds["splits"]
    train_mask = splits == "train"
    if train_ids is not None:
        train_mask = train_mask & np.isin(np.asarray(ds["ids"]), train_ids)
    val_mask = splits == "validation"
    env_z = standardize_env(ds["env"], train_mask)
    channels = ds["channels"]
    stats = channel_stats(channels, train_mask)
    targets = standardized_targets(channels, stats)
    train_masks = {ch: mask_for(train_mask) for ch in PRIMARY + REPORTERS}
    val_masks = {ch: mask_for(val_mask) for ch in PRIMARY + REPORTERS}
    heads = PRIMARY + [r for r in reporter_set if r not in PRIMARY]
    params = tss.init_params(ds["env"].shape[1], latent_dim=latent_dim, hidden_dim=24, head_names=heads, n_controls=0, seed=seed)
    adam = tss.AdamState(params, lr=lr)
    logs = []
    best_total = {"value": math.inf, "params": copy_params(params), "epoch": 0}
    best_product = {"value": math.inf, "params": copy_params(params), "epoch": 0}
    grad_epochs = {0, 1, 5, 20, max(epochs // 2, 1), epochs - 1}
    grad_rows = []
    for epoch in range(epochs):
        total, grads, losses = combined_loss_grads(params, env_z, targets, train_masks, reporter_set, lambda_r, reporter_mode)
        adam.step(params, grads)
        if epoch in grad_epochs or epoch % log_every == 0 or epoch == epochs - 1:
            train_eval = evaluate_losses(params, env_z, targets, train_masks, reporter_set, lambda_r, reporter_mode)
            val_eval = evaluate_losses(params, env_z, targets, val_masks, reporter_set, lambda_r, reporter_mode)
            cache = tss.forward(params, env_z, N_TIMEPOINTS)
            logs.append(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "lambda_R": lambda_r,
                    "reporter_mode": reporter_mode,
                    "reporter_set": ",".join(reporter_set) if reporter_set else "none",
                    "latent_dim": latent_dim,
                    "train_product_loss": train_eval["product_loss"],
                    "val_product_loss": val_eval["product_loss"],
                    "train_reporter_loss": train_eval["reporter_mean"],
                    "val_reporter_loss": val_eval["reporter_mean"],
                    "train_total_loss": train_eval["total"],
                    "val_total_loss": val_eval["total"],
                    "shared_grad_norm_total": shared_norm(grads),
                    "param_norm": param_norm(params),
                    "latent_norm": float(np.sqrt(np.mean(cache.z**2))),
                }
            )
            if epoch in grad_epochs:
                grad_rows.extend(gradient_diagnostics(params, env_z, targets, train_masks, reporter_set, epoch, seed, lambda_r, reporter_mode, latent_dim))
            if val_eval["total"] < best_total["value"]:
                best_total = {"value": val_eval["total"], "params": copy_params(params), "epoch": epoch}
            if val_eval["product_loss"] < best_product["value"]:
                best_product = {"value": val_eval["product_loss"], "params": copy_params(params), "epoch": epoch}
    rows, pred_rows = [], []
    for rule, chosen in [("final", {"params": params, "epoch": epochs - 1}), ("best_total_val", best_total), ("best_product_val", best_product)]:
        metrics, preds = evaluate_model(ds, chosen["params"], env_z, stats, reporter_set, seed, lambda_r, reporter_mode, latent_dim, rule, chosen["epoch"])
        rows.append(metrics)
        pred_rows.append(preds)
    return (
        {"params": params, "env_z": env_z, "stats": stats},
        pd.concat(rows, ignore_index=True),
        pd.concat(pred_rows, ignore_index=True),
        pd.concat([pd.DataFrame(logs), pd.DataFrame(grad_rows)], ignore_index=True, sort=False),
    )


def param_norm(params) -> float:
    total = sum(np.sum(x**2) for x in shared_arrays(params))
    for w, b in params.heads.values():
        total += np.sum(w**2) + np.sum(b**2)
    return float(np.sqrt(total))


def evaluate_losses(params, env_z, targets, masks, reporter_set, lambda_r, reporter_mode) -> dict[str, float]:
    total, _grads, losses = combined_loss_grads(params, env_z, targets, masks, reporter_set, lambda_r, reporter_mode)
    product_loss = losses["B_total"] + losses["X"]
    return {"total": total, "product_loss": product_loss, **losses}


def gradient_diagnostics(params, env_z, targets, masks, reporter_set, epoch, seed, lambda_r, reporter_mode, latent_dim) -> list[dict[str, object]]:
    rows = []
    losses = {}
    grads = {}
    for ch in PRIMARY + reporter_set:
        losses[ch], grads[ch] = task_grad(params, env_z, targets, masks, ch, lambda_dyn=0.0)
        rows.append(
            {
                "record_type": "gradient_norm",
                "epoch": epoch,
                "seed": seed,
                "lambda_R": lambda_r,
                "reporter_mode": reporter_mode,
                "reporter_set": ",".join(reporter_set) if reporter_set else "none",
                "latent_dim": latent_dim,
                "task": ch,
                "loss": losses[ch],
                "shared_grad_norm": shared_norm(grads[ch]),
            }
        )
    if "B_total" in grads:
        rep_combo = zero_like(params)
        for r in reporter_set:
            add_scaled(rep_combo, grads[r], 1.0 / max(len(reporter_set), 1))
            rows.append(
                {
                    "record_type": "gradient_cosine",
                    "epoch": epoch,
                    "seed": seed,
                    "lambda_R": lambda_r,
                    "reporter_mode": reporter_mode,
                    "reporter_set": ",".join(reporter_set) if reporter_set else "none",
                    "latent_dim": latent_dim,
                    "task": f"B_total_vs_{r}",
                    "shared_grad_cosine": shared_cosine(grads["B_total"], grads[r]),
                }
            )
        if reporter_set:
            rows.append(
                {
                    "record_type": "gradient_cosine",
                    "epoch": epoch,
                    "seed": seed,
                    "lambda_R": lambda_r,
                    "reporter_mode": reporter_mode,
                    "reporter_set": ",".join(reporter_set),
                    "latent_dim": latent_dim,
                    "task": "B_total_vs_reporter_mean",
                    "shared_grad_cosine": shared_cosine(grads["B_total"], rep_combo),
                }
            )
    return rows


def predict_channel(params, env_z, channel, stats):
    cache = tss.forward(params, env_z, N_TIMEPOINTS)
    if channel not in params.heads:
        return None
    wh, bh = params.heads[channel]
    zpred = (cache.z @ wh + bh)[..., 0]
    return zpred * stats[channel][1] + stats[channel][0]


def evaluate_model(ds, params, env_z, stats, reporter_set, seed, lambda_r, reporter_mode, latent_dim, checkpoint_rule, checkpoint_epoch):
    rows, preds = [], []
    cache = tss.forward(params, env_z, N_TIMEPOINTS)
    for channel in PRIMARY + REPORTERS:
        pred = predict_channel(params, env_z, channel, stats)
        if pred is None:
            continue
        truth = ds["truth"][channel]
        for split in sorted(set(ds["splits"])):
            mask = ds["splits"] == split
            row = active.metric_row(ds, "diagnostic_state_space", seed, split, channel, pred[:, mask], truth[:, mask])
            row.update(
                {
                    "lambda_R": lambda_r,
                    "reporter_mode": reporter_mode,
                    "reporter_set": ",".join(reporter_set) if reporter_set else "none",
                    "latent_dim": latent_dim,
                    "checkpoint_rule": checkpoint_rule,
                    "checkpoint_epoch": checkpoint_epoch,
                }
            )
            rows.append(row)
        if channel in PRIMARY:
            for b, cid in enumerate(ds["ids"]):
                for t in range(N_TIMEPOINTS):
                    preds.append({"culture_id": cid, "time_index": t, "channel": channel, "prediction": pred[t, b], "truth": truth[t, b], "split": ds["splits"][b], "seed": seed, "lambda_R": lambda_r, "reporter_mode": reporter_mode, "reporter_set": ",".join(reporter_set) if reporter_set else "none", "latent_dim": latent_dim, "checkpoint_rule": checkpoint_rule})
    latent = cache.z.reshape(N_TIMEPOINTS, len(ds["ids"]), -1).transpose(1, 0, 2).reshape(len(ds["ids"]), -1)
    rec = recovery_metrics(latent)
    rec.update(
        {
            "world": active.WORLD,
            "model": "diagnostic_state_space",
            "seed": seed,
            "split": "all",
            "channel": "posthoc_recovery",
            "metric_type": "posthoc_latent_alignment",
            "rmse": np.nan,
            "nrmse_train_std": np.nan,
            "endpoint_abs_error": np.nan,
            "auc_abs_error": np.nan,
            "lambda_R": lambda_r,
            "reporter_mode": reporter_mode,
            "reporter_set": ",".join(reporter_set) if reporter_set else "none",
            "latent_dim": latent_dim,
            "checkpoint_rule": checkpoint_rule,
            "checkpoint_epoch": checkpoint_epoch,
        }
    )
    rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(preds)


def recovery_metrics(latent_flat: np.ndarray) -> dict[str, float]:
    paths = {
        "rxncon": ACTIVE_DIR / "hidden_rxncon.npz",
        "module": ACTIVE_DIR / "hidden_module_trajectories_wide.csv",
        "interface": ACTIVE_DIR / "hidden_interface_trajectories_wide.csv",
        "metabolic": ACTIVE_DIR / "fluxes.csv",
    }
    out = {}
    x, _ = rxncon_base.remove_constant_columns(latent_flat)
    if x.shape[1] == 0:
        return {f"{k}_recovery_r2": np.nan for k in paths}
    xz = rxncon_base.standardize_matrix(x)
    for name, path in paths.items():
        if name == "rxncon":
            import run_rxncon_gsm_generator_gate as gate

            model = gate.load_rxncon_model()
            arr = np.load(path)["traj"]
            state_ids = [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]
            y = gate.flatten_by_culture(arr[:, :, state_ids])
        elif name == "metabolic":
            manifest = pd.read_csv(ACTIVE_DIR / "culture_manifest.csv")
            flux = pd.read_csv(path)
            ids = manifest["culture_id"].tolist()
            cols = [c for c in ["biomass_flux", "beta_carotene_flux", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux"] if c in flux.columns]
            y = np.concatenate([flux.pivot(index="interval_index", columns="culture_id", values=c)[ids].T.to_numpy(float) for c in cols], axis=1)
        else:
            y = pd.read_csv(path).to_numpy(float)
        y, _ = rxncon_base.remove_constant_columns(y)
        if y.shape[1] == 0:
            out[f"{name}_recovery_r2"] = np.nan
            continue
        yz = rxncon_base.standardize_matrix(y)
        _u, _s, vt = np.linalg.svd(yz, full_matrices=False)
        pcs = yz @ vt[: min(8, vt.shape[0])].T
        coef = np.linalg.solve(xz.T @ xz + 1e-6 * np.eye(xz.shape[1]), xz.T @ pcs)
        pred = xz @ coef
        out[f"{name}_recovery_r2"] = float(1.0 - np.sum((pcs - pred) ** 2) / max(np.sum((pcs - pcs.mean(axis=0)) ** 2), 1e-12))
    return out


def initial_loss_audit(ds: dict[str, object]) -> pd.DataFrame:
    train_mask = ds["splits"] == "train"
    env_z = standardize_env(ds["env"], train_mask)
    stats = channel_stats(ds["channels"], train_mask)
    targets = standardized_targets(ds["channels"], stats)
    masks = {ch: mask_for(train_mask) for ch in PRIMARY + REPORTERS}
    rows = []
    for seed in SEEDS:
        params = tss.init_params(ds["env"].shape[1], latent_dim=16, hidden_dim=24, head_names=PRIMARY + REPORTERS, n_controls=0, seed=seed)
        for stage, epochs in [("initialization", 0), ("after_20_epochs_original_sum", 20)]:
            if epochs:
                adam = tss.AdamState(params, lr=0.018)
                for _ in range(epochs):
                    _loss, grads, _ = combined_loss_grads(params, env_z, targets, masks, REPORTERS, 1.0, "sum")
                    adam.step(params, grads)
            task_rows = gradient_diagnostics(params, env_z, targets, masks, REPORTERS, epochs, seed, 1.0, "sum", 16)
            norms = [r for r in task_rows if r.get("record_type") == "gradient_norm"]
            total_grad = sum(r["shared_grad_norm"] for r in norms)
            for r in norms:
                rows.append({**r, "stage": stage, "effective_grad_fraction_by_l1_norm": r["shared_grad_norm"] / max(total_grad, 1e-12)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "loss_gradient_audit.csv", index=False)
    return out


def reporter_relevance(ds: dict[str, object]) -> pd.DataFrame:
    manifest = pd.read_csv(ACTIVE_DIR / "culture_manifest.csv")
    train_mask = manifest["split"].eq("train").to_numpy()
    obs = pd.read_csv(ACTIVE_DIR / "observable_trajectories.csv")
    ids = manifest["culture_id"].tolist()
    rows = []
    product = active.pivot_time(obs, ids, "B_total").T
    y_targets = {
        "product_final": product[:, -1],
        "product_auc": np.trapezoid(product, dx=DT, axis=1),
    }
    _u, _s, vt = np.linalg.svd(rxncon_base.standardize_matrix(product), full_matrices=False)
    for pc in range(min(3, vt.shape[0])):
        y_targets[f"product_pc{pc+1}"] = rxncon_base.standardize_matrix(product) @ vt[pc]
    env = rxncon_base.standardize_matrix(manifest[active.LEARNER_INPUTS].to_numpy(float))
    for rep in REPORTERS:
        arr = active.pivot_time(obs, ids, rep).T
        feats = {
            f"{rep}_mean": arr.mean(axis=1),
            f"{rep}_final": arr[:, -1],
            f"{rep}_auc": np.trapezoid(arr, dx=DT, axis=1),
        }
        for fname, x in feats.items():
            for tname, y in y_targets.items():
                c = corr(x[train_mask], y[train_mask])
                xr = residualize(x[:, None], env)[:, 0]
                yr = residualize(y[:, None], env)[:, 0]
                pc = corr(xr[train_mask], yr[train_mask])
                rows.append({"reporter": rep, "feature": fname, "target": tname, "train_corr": c, "partial_corr_after_external_controls": pc})
    # Hidden post-hoc relevance: reporter feature to interface/metabolic PC1.
    hidden_targets = hidden_pc_targets()
    for rep in REPORTERS:
        arr = active.pivot_time(obs, ids, rep).T
        x = arr.mean(axis=1)
        for tname, y in hidden_targets.items():
            rows.append({"reporter": rep, "feature": f"{rep}_mean", "target": tname, "train_corr": corr(x[train_mask], y[train_mask]), "partial_corr_after_external_controls": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reporter_relevance.csv", index=False)
    return out


def hidden_pc_targets() -> dict[str, np.ndarray]:
    out = {}
    for name, path in {"interface": ACTIVE_DIR / "hidden_interface_trajectories_wide.csv", "module": ACTIVE_DIR / "hidden_module_trajectories_wide.csv"}.items():
        x, _ = rxncon_base.remove_constant_columns(pd.read_csv(path).to_numpy(float))
        z = rxncon_base.standardize_matrix(x)
        _u, _s, vt = np.linalg.svd(z, full_matrices=False)
        for i in range(min(3, vt.shape[0])):
            out[f"{name}_pc{i+1}"] = z @ vt[i]
    manifest = pd.read_csv(ACTIVE_DIR / "culture_manifest.csv")
    flux = pd.read_csv(ACTIVE_DIR / "fluxes.csv")
    ids = manifest["culture_id"].tolist()
    cols = [c for c in ["biomass_flux", "beta_carotene_flux", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux"] if c in flux.columns]
    x = np.concatenate([flux.pivot(index="interval_index", columns="culture_id", values=c)[ids].T.to_numpy(float) for c in cols], axis=1)
    x, _ = rxncon_base.remove_constant_columns(x)
    z = rxncon_base.standardize_matrix(x)
    _u, _s, vt = np.linalg.svd(z, full_matrices=False)
    for i in range(min(3, vt.shape[0])):
        out[f"metabolic_pc{i+1}"] = z @ vt[i]
    return out


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.solve(x1.T @ x1 + 1e-6 * np.eye(x1.shape[1]), x1.T @ y)
    return y - x1 @ coef


def corr(a, b) -> float:
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_sweep(ds: dict[str, object], epochs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, logs = [], []
    # Original behavior: standardized per channel but reporter losses summed.
    for seed in SEEDS:
        _ckpt, m, _p, log = train_model(ds, reporter_set=REPORTERS, lambda_r=1.0, reporter_mode="sum", seed=seed, epochs=epochs)
        frames.append(m.assign(condition_family="original_sum"))
        logs.append(log.assign(condition_family="original_sum"))
    for lam in LAMBDA_GRID:
        for seed in SEEDS:
            _ckpt, m, _p, log = train_model(ds, reporter_set=REPORTERS if lam > 0 else [], lambda_r=lam, reporter_mode="mean", seed=seed, epochs=epochs)
            frames.append(m.assign(condition_family="lambda_sweep"))
            logs.append(log.assign(condition_family="lambda_sweep"))
    metrics = pd.concat(frames, ignore_index=True)
    log_df = pd.concat(logs, ignore_index=True, sort=False)
    metrics.to_csv(OUT_DIR / "lambda_sweep_metrics.csv", index=False)
    log_df.to_csv(OUT_DIR / "training_dynamics_and_gradients.csv", index=False)
    return metrics, log_df


def choose_best_positive_lambda(metrics: pd.DataFrame) -> float:
    p = metrics[
        (metrics["condition_family"].eq("lambda_sweep"))
        & (metrics["metric_type"].eq("prediction"))
        & (metrics["channel"].eq("B_total"))
        & (metrics["split"].eq("validation"))
        & (metrics["checkpoint_rule"].eq("best_product_val"))
        & (metrics["lambda_R"] > 0)
    ]
    if p.empty:
        return 0.03
    return float(p.groupby("lambda_R")["nrmse_train_std"].mean().idxmin())


def run_ablations(ds: dict[str, object], lambda_best: float, epochs: int) -> pd.DataFrame:
    groups = {
        "R_stress": ["R_stress"],
        "R_resource": ["R_resource"],
        "R_checkpoint": ["R_checkpoint"],
        "R_pathway_capacity": ["R_pathway_capacity"],
        "R_morphology": ["R_morphology"],
        "R_energy": ["R_energy"],
        "metabolic_resource": ["R_resource", "R_pathway_capacity", "R_energy"],
        "stress": ["R_stress", "R_checkpoint"],
        "cell_state": ["R_morphology", "R_checkpoint"],
        "all_six": REPORTERS,
    }
    frames = []
    for label, reps in groups.items():
        for seed in SEEDS[:3]:
            _ckpt, m, _p, _log = train_model(ds, reporter_set=reps, lambda_r=lambda_best, reporter_mode="mean", seed=seed, epochs=epochs)
            frames.append(m.assign(ablation=label, condition_family="reporter_ablation"))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "reporter_ablation_metrics.csv", index=False)
    return out


def run_latent_dim(ds: dict[str, object], lambda_best: float, epochs: int) -> pd.DataFrame:
    frames = []
    for d in LATENT_DIMS:
        for label, reps, lam in [("product_only", [], 0.0), ("reporter_supervised", REPORTERS, lambda_best)]:
            for seed in SEEDS[:3]:
                _ckpt, m, _p, _log = train_model(ds, reporter_set=reps, lambda_r=lam, reporter_mode="mean", seed=seed, latent_dim=d, epochs=epochs)
                frames.append(m.assign(latent_sweep_condition=label, condition_family="latent_dim"))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "latent_dim_metrics.csv", index=False)
    return out


def run_sample_size(ds: dict[str, object], lambda_best: float, epochs: int) -> pd.DataFrame:
    train_ids = np.array(ds["ids"])[ds["splits"] == "train"]
    frames = []
    for n in SUBSAMPLE_N:
        if n > len(train_ids):
            continue
        for ss in SUBSAMPLE_SEEDS:
            rng = np.random.default_rng(ss)
            chosen = np.array(sorted(rng.choice(train_ids, size=n, replace=False)))
            for label, reps, lam in [("product_only", [], 0.0), ("reporter_supervised", REPORTERS, lambda_best)]:
                _ckpt, m, _p, _log = train_model(ds, reporter_set=reps, lambda_r=lam, reporter_mode="mean", seed=11, train_ids=chosen, epochs=epochs)
                frames.append(m.assign(training_cultures_N=n, subsample_seed=ss, sample_size_condition=label, condition_family="sample_size"))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "sample_size_metrics.csv", index=False)
    return out


def central_summary(metrics, ablations, latent, sample, logs, relevance) -> pd.DataFrame:
    rows = []
    all_metrics = pd.concat([metrics, ablations, latent, sample], ignore_index=True, sort=False)
    pred = all_metrics[(all_metrics["metric_type"].eq("prediction")) & (all_metrics["channel"].eq("B_total")) & (all_metrics["checkpoint_rule"].eq("best_product_val"))]
    for keys, g in pred.groupby(["condition_family", "lambda_R", "reporter_mode", "reporter_set", "latent_dim", "checkpoint_rule"], dropna=False):
        rec = {
            "condition_family": keys[0],
            "lambda_R": keys[1],
            "reporter_mode": keys[2],
            "reporter_set": keys[3],
            "latent_dim": keys[4],
            "checkpoint_rule": keys[5],
            "product_NRMSE": g[g["split"].ne("train")]["nrmse_train_std"].mean(),
            "heldout_combination_NRMSE": g[g["split"].eq("heldout_combination")]["nrmse_train_std"].mean(),
            "hard_perturbation_NRMSE": g[g["split"].eq("hard_perturbation")]["nrmse_train_std"].mean(),
        }
        rec.update(recovery_aggregate(all_metrics, keys))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reporter_diagnostic_summary_table.csv", index=False)
    decision = decision_table(metrics, ablations, latent, sample, logs, relevance)
    decision.to_csv(OUT_DIR / "reporter_failure_decision_tree.csv", index=False)
    return out


def recovery_aggregate(all_metrics, keys) -> dict[str, float]:
    fam, lam, mode, reps, d, rule = keys
    r = all_metrics[
        (all_metrics["metric_type"].eq("posthoc_latent_alignment"))
        & (all_metrics["condition_family"].eq(fam))
        & (all_metrics["lambda_R"].eq(lam))
        & (all_metrics["reporter_mode"].eq(mode))
        & (all_metrics["reporter_set"].eq(reps))
        & (all_metrics["latent_dim"].eq(d))
        & (all_metrics["checkpoint_rule"].eq(rule))
    ]
    out = {}
    for col in ["rxncon_recovery_r2", "module_recovery_r2", "interface_recovery_r2", "metabolic_recovery_r2"]:
        out[col] = r[col].mean() if col in r else np.nan
    return out


def decision_table(metrics, ablations, latent, sample, logs, relevance) -> pd.DataFrame:
    rows = []
    p = metrics[(metrics["condition_family"].eq("lambda_sweep")) & (metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["checkpoint_rule"].eq("best_product_val")) & (metrics["split"].ne("train"))]
    g = p.groupby("lambda_R")["nrmse_train_std"].mean()
    best_lam = float(g.idxmin())
    lam0 = float(g.loc[0.0])
    best_pos = float(g[g.index > 0].min()) if any(g.index > 0) else np.nan
    rows.append({"classification": "reporter_overweighting", "selected": bool(best_pos < 0.95 * original_product_error(metrics)), "evidence": f"lambda0={lam0:.3f}; best_positive={best_pos:.3f}; original_sum={original_product_error(metrics):.3f}"})
    rows.append({"classification": "loss_scaling_problem", "selected": bool(best_pos < 0.95 * original_product_error(metrics)), "evidence": "per-channel standardized mean reporter loss sweep compared with original summed reporter objective"})
    # Product vs total checkpoint.
    chk = metrics[(metrics["condition_family"].eq("lambda_sweep")) & (metrics["lambda_R"].eq(best_lam)) & (metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["split"].ne("train"))]
    by_rule = chk.groupby("checkpoint_rule")["nrmse_train_std"].mean()
    rows.append({"classification": "checkpoint_overoptimization", "selected": bool(by_rule.get("best_product_val", np.inf) < 0.9 * by_rule.get("best_total_val", np.inf)), "evidence": json.dumps(by_rule.to_dict())})
    # Gradient conflict.
    cos = logs[(logs["record_type"].eq("gradient_cosine")) & (logs["task"].eq("B_total_vs_reporter_mean"))]["shared_grad_cosine"].dropna()
    rows.append({"classification": "global_multitask_negative_transfer", "selected": bool(cos.mean() < -0.05), "evidence": f"mean product-vs-reporter gradient cosine={cos.mean():.3f}"})
    # Capacity.
    latp = latent[(latent["metric_type"].eq("prediction")) & (latent["channel"].eq("B_total")) & (latent["split"].ne("train")) & (latent["checkpoint_rule"].eq("best_product_val"))]
    lpiv = latp.groupby(["latent_dim", "latent_sweep_condition"])["nrmse_train_std"].mean().unstack()
    gap32 = lpiv.loc[32, "reporter_supervised"] - lpiv.loc[32, "product_only"] if 32 in lpiv.index else np.nan
    rows.append({"classification": "insufficient_latent_capacity", "selected": bool(np.isfinite(gap32) and gap32 < 0.05), "evidence": f"d32 reporter-product gap={gap32:.3f}"})
    # Data.
    sp = sample[(sample["metric_type"].eq("prediction")) & (sample["channel"].eq("B_total")) & (sample["split"].ne("train")) & (sample["checkpoint_rule"].eq("best_product_val"))]
    spiv = sp.groupby(["training_cultures_N", "sample_size_condition"])["nrmse_train_std"].mean().unstack()
    gaps = (spiv["reporter_supervised"] - spiv["product_only"]).dropna()
    shrink = bool(len(gaps) >= 2 and gaps.iloc[-1] < gaps.iloc[0])
    rows.append({"classification": "insufficient_data", "selected": shrink, "evidence": json.dumps({str(k): float(v) for k, v in gaps.items()})})
    # Reporter subset.
    abp = ablations[(ablations["metric_type"].eq("prediction")) & (ablations["channel"].eq("B_total")) & (ablations["split"].ne("train")) & (ablations["checkpoint_rule"].eq("best_product_val"))]
    abg = abp.groupby("ablation")["nrmse_train_std"].mean().sort_values()
    rows.append({"classification": "reporter_specific_negative_transfer", "selected": bool(abg.iloc[-1] - abg.iloc[0] > 0.15), "evidence": json.dumps(abg.round(3).to_dict())})
    rows.append({"classification": "reporter_information_not_product_relevant", "selected": bool(relevance["partial_corr_after_external_controls"].abs().median(skipna=True) < 0.2), "evidence": f"median abs partial reporter-product/control correlation={relevance['partial_corr_after_external_controls'].abs().median(skipna=True):.3f}"})
    rows.append({"classification": "optimizer_instability", "selected": bool(seed_variability(metrics) > 0.25), "evidence": f"mean seed std across lambda sweep={seed_variability(metrics):.3f}"})
    return pd.DataFrame(rows)


def original_product_error(metrics) -> float:
    p = metrics[(metrics["condition_family"].eq("original_sum")) & (metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["split"].ne("train")) & (metrics["checkpoint_rule"].eq("best_product_val"))]
    return float(p["nrmse_train_std"].mean())


def seed_variability(metrics) -> float:
    p = metrics[(metrics["condition_family"].eq("lambda_sweep")) & (metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["split"].ne("train")) & (metrics["checkpoint_rule"].eq("best_product_val"))]
    return float(p.groupby("lambda_R")["nrmse_train_std"].std().mean())


def figures() -> None:
    metrics = pd.read_csv(OUT_DIR / "lambda_sweep_metrics.csv")
    logs = pd.read_csv(OUT_DIR / "training_dynamics_and_gradients.csv")
    ab = pd.read_csv(OUT_DIR / "reporter_ablation_metrics.csv")
    lat = pd.read_csv(OUT_DIR / "latent_dim_metrics.csv")
    sample = pd.read_csv(OUT_DIR / "sample_size_metrics.csv")
    summary = pd.read_csv(OUT_DIR / "reporter_diagnostic_summary_table.csv")
    p = metrics[(metrics["condition_family"].eq("lambda_sweep")) & (metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["split"].ne("train")) & (metrics["checkpoint_rule"].eq("best_product_val"))]
    g = p.groupby("lambda_R", as_index=False)["nrmse_train_std"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g["lambda_R"], g["mean"], yerr=g["std"], marker="o")
    ax.set_xscale("symlog", linthresh=0.003)
    ax.set_xlabel("lambda_R")
    ax.set_ylabel("Product NRMSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_lambda_product_error.svg")
    plt.close(fig)

    rec = metrics[(metrics["metric_type"].eq("posthoc_latent_alignment")) & (metrics["condition_family"].eq("lambda_sweep")) & (metrics["checkpoint_rule"].eq("best_product_val"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ["rxncon_recovery_r2", "module_recovery_r2", "interface_recovery_r2", "metabolic_recovery_r2"]:
        ax.plot(rec.groupby("lambda_R")[col].mean().index, rec.groupby("lambda_R")[col].mean().values, marker="o", label=col.replace("_recovery_r2", ""))
    ax.set_xscale("symlog", linthresh=0.003)
    ax.set_xlabel("lambda_R")
    ax.set_ylabel("Latent->hidden PC R2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_lambda_physiology_recovery.svg")
    plt.close(fig)

    merged = p.groupby("lambda_R")["nrmse_train_std"].mean().to_frame("product").join(rec.groupby("lambda_R")["interface_recovery_r2"].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(merged["product"], merged["interface_recovery_r2"])
    for lam, row in merged.iterrows():
        ax.text(row["product"], row["interface_recovery_r2"], str(lam), fontsize=7)
    ax.set_xlabel("Product NRMSE")
    ax.set_ylabel("Interface recovery R2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_product_physiology_pareto.svg")
    plt.close(fig)

    grad = logs[logs["record_type"].eq("gradient_norm")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for task in ["B_total", "X", *REPORTERS]:
        gg = grad[(grad["task"].eq(task)) & (grad["condition_family"].eq("lambda_sweep")) & (grad["lambda_R"].eq(1.0))]
        if not gg.empty:
            ax.plot(gg.groupby("epoch")["shared_grad_norm"].mean(), label=task)
    ax.set_xlabel("epoch")
    ax.set_ylabel("shared gradient norm")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_gradient_norms.svg")
    plt.close(fig)

    cos = logs[logs["record_type"].eq("gradient_cosine")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for task in sorted(cos["task"].dropna().unique()):
        if task == "B_total_vs_reporter_mean" or any(r in task for r in REPORTERS):
            cc = cos[(cos["task"].eq(task)) & (cos["condition_family"].eq("lambda_sweep")) & (cos["lambda_R"].eq(1.0))]
            if not cc.empty:
                ax.plot(cc.groupby("epoch")["shared_grad_cosine"].mean(), label=task.replace("B_total_vs_", ""))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cosine with product gradient")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_gradient_cosines.svg")
    plt.close(fig)

    dyn = logs[(logs["condition_family"].eq("lambda_sweep")) & (logs["lambda_R"].isin([0.0, 0.03, 1.0])) & logs["val_product_loss"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for lam, gg in dyn.groupby("lambda_R"):
        ax.plot(gg.groupby("epoch")["val_product_loss"].mean(), label=f"product val lambda={lam}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation product loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_validation_dynamics.svg")
    plt.close(fig)

    abp = ab[(ab["metric_type"].eq("prediction")) & (ab["channel"].eq("B_total")) & (ab["split"].ne("train")) & (ab["checkpoint_rule"].eq("best_product_val"))]
    ag = abp.groupby("ablation", as_index=False)["nrmse_train_std"].mean().sort_values("nrmse_train_std")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ag["ablation"], ag["nrmse_train_std"])
    ax.set_ylabel("Product NRMSE")
    ax.set_xticklabels(ag["ablation"], rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_ablation.svg")
    plt.close(fig)

    lp = lat[(lat["metric_type"].eq("prediction")) & (lat["channel"].eq("B_total")) & (lat["split"].ne("train")) & (lat["checkpoint_rule"].eq("best_product_val"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, gg in lp.groupby("latent_sweep_condition"):
        ax.plot(gg.groupby("latent_dim")["nrmse_train_std"].mean(), marker="o", label=cond)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("Product NRMSE")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_latent_dimension.svg")
    plt.close(fig)

    sp = sample[(sample["metric_type"].eq("prediction")) & (sample["channel"].eq("B_total")) & (sample["split"].ne("train")) & (sample["checkpoint_rule"].eq("best_product_val"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, gg in sp.groupby("sample_size_condition"):
        ax.plot(gg.groupby("training_cultures_N")["nrmse_train_std"].mean(), marker="o", label=cond)
    ax.set_xlabel("training cultures")
    ax.set_ylabel("Product NRMSE")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_sample_size.svg")
    plt.close(fig)

    rel = pd.read_csv(OUT_DIR / "reporter_relevance.csv")
    top = rel[rel["target"].isin(["product_final", "product_auc", "interface_pc1", "metabolic_pc1"])]
    pivot = top.pivot_table(index="reporter", columns="target", values="train_corr", aggfunc=lambda x: np.nanmax(np.abs(x)))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.fillna(0).to_numpy(), vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pivot.index, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_reporter_alignment_heatmap.svg")
    plt.close(fig)


def update_docs() -> None:
    decision = pd.read_csv(OUT_DIR / "reporter_failure_decision_tree.csv")
    summ = pd.read_csv(OUT_DIR / "reporter_diagnostic_summary_table.csv")
    best = best_rows_for_docs()
    selected = decision[decision["selected"].astype(bool)]["classification"].tolist()
    section = f"""

### Reporter-Supervision Diagnostic

The active-rxncon benchmark unexpectedly showed worse product prediction when
the state-space model was supervised with all six training-time biosensors. The
diagnostic in `scripts/run_rxncon_reporter_supervision_diagnostic.py` reused
the same 125 cultures, splits, reporters, and hidden lockbox tables; it did not
regenerate Yeast9 data or alter the generator.

The original reporter setting was already per-channel standardized, but it
summed six reporter losses alongside product and biomass. The diagnostic
therefore compared that original summed objective with a normalized objective
`L = L_product + L_biomass + lambda_R * mean(L_reporters)`, swept
`lambda_R`, computed shared-gradient norms and product-vs-reporter gradient
cosines, tested reporter ablations, latent dimensions, and culture-count
subsamples, and used hidden simulator quantities only for post-hoc recovery.

The best validation/product setting was `lambda_R={best['best_lambda']}` with
non-train product NRMSE {best['best_lambda_product']:.3f}, compared with
`lambda_R=0` product-only NRMSE {best['lambda0_product']:.3f}. The original
summed reporter objective had product NRMSE {best['original_sum_product']:.3f}.
Selected failure classifications were: {', '.join(selected) if selected else 'none'}.
The central result table is
`data/rxncon_reporter_supervision_diagnostic/reporter_diagnostic_summary_table.csv`.

The bounded interpretation is that this is a negative result for
training-time auxiliary reporter supervision in the current architecture and
dataset. It does not test live reporter assimilation during a new culture, and
it does not imply that the biosensors lack biological information.
"""
    for rel in ["README.md", "PHYSIOLOGY_QUEST_VALIDATION.md", "CONCRETE_EXPERIMENT_CHAIN.md"]:
        path = ROOT / rel
        text = path.read_text()
        marker = "### Reporter-Supervision Diagnostic"
        if marker not in text:
            path.write_text(text.rstrip() + "\n" + section)


def best_rows_for_docs() -> dict[str, float]:
    m = pd.read_csv(OUT_DIR / "lambda_sweep_metrics.csv")
    p = m[(m["condition_family"].eq("lambda_sweep")) & (m["metric_type"].eq("prediction")) & (m["channel"].eq("B_total")) & (m["split"].ne("train")) & (m["checkpoint_rule"].eq("best_product_val"))]
    g = p.groupby("lambda_R")["nrmse_train_std"].mean()
    orig = original_product_error(m)
    return {"best_lambda": float(g.idxmin()), "best_lambda_product": float(g.min()), "lambda0_product": float(g.loc[0.0]), "original_sum_product": orig}


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    ds = dataset()
    if args.phase in {"all", "audit"}:
        initial_loss_audit(ds)
        reporter_relevance(ds)
        if args.phase == "audit":
            return
    if args.phase in {"all", "sweep"}:
        metrics, logs = run_sweep(ds, args.epochs)
        if args.phase == "sweep":
            return
    else:
        metrics = pd.read_csv(OUT_DIR / "lambda_sweep_metrics.csv")
        logs = pd.read_csv(OUT_DIR / "training_dynamics_and_gradients.csv")
    lambda_best = choose_best_positive_lambda(metrics)
    if args.phase in {"all", "ablations"}:
        run_ablations(ds, lambda_best, max(160, args.epochs // 2))
        if args.phase == "ablations":
            return
    if args.phase in {"all", "latent"}:
        run_latent_dim(ds, lambda_best, max(160, args.epochs // 2))
        if args.phase == "latent":
            return
    if args.phase in {"all", "sample"}:
        run_sample_size(ds, lambda_best, max(150, args.epochs // 2))
        if args.phase == "sample":
            return
    if args.phase in {"all", "report"}:
        if not (OUT_DIR / "reporter_relevance.csv").exists():
            reporter_relevance(ds)
        ab = pd.read_csv(OUT_DIR / "reporter_ablation_metrics.csv")
        lat = pd.read_csv(OUT_DIR / "latent_dim_metrics.csv")
        sample = pd.read_csv(OUT_DIR / "sample_size_metrics.csv")
        rel = pd.read_csv(OUT_DIR / "reporter_relevance.csv")
        central_summary(metrics, ab, lat, sample, logs, rel)
        figures()
        update_docs()
        print(pd.read_csv(OUT_DIR / "reporter_failure_decision_tree.csv").to_string(index=False))
        print(best_rows_for_docs())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["all", "audit", "sweep", "ablations", "latent", "sample", "report"], default="all")
    p.add_argument("--epochs", type=int, default=220)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
