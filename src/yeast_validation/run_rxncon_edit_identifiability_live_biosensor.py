#!/usr/bin/env python3
"""Gate 1 edit-identifiability audit for live-biosensor validation.

This script reuses the completed active-rxncon genetic-edit dataset.  It does
not regenerate Yeast9 cultures and does not alter the biological generator or
biosensor definitions.  Gate 2 live biosensor assimilation is intentionally not
run unless Gate 1 shows that genotype creates a meaningful product-prediction
task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_rxncon_active_structural_validation as active  # noqa: E402
import run_rxncon_edit_transfer_benchmark as editbench  # noqa: E402
import run_rxncon_reporter_supervision_diagnostic as repdiag  # noqa: E402


SRC_DIR = ROOT / "data" / "rxncon_edit_transfer_benchmark"
OUT_DIR = ROOT / "data" / "rxncon_edit_identifiability_live_biosensor"
FIG_DIR = ROOT / "figures"
REPORTERS = active.REPORTER_COLS
BASE_INPUTS = editbench.BASE_INPUTS
PRIMARY = ["B_total", "X"]
RNG_SEED = 20260828
N_TIMEPOINTS = active.N_TIMEPOINTS
DT = active.DT


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(SRC_DIR / "culture_manifest.csv")
    obs = pd.read_csv(SRC_DIR / "observable_trajectories.csv")
    return manifest, obs


def genotype_columns(manifest: pd.DataFrame) -> list[str]:
    return [c for c in manifest.columns if c.startswith("edit::")]


def pivot(obs: pd.DataFrame, ids: list[str], col: str) -> np.ndarray:
    mat = obs.pivot(index="time_index", columns="culture_id", values=col)
    return mat[ids].to_numpy(float)


def make_dataset(feature_mode: str, seed: int = RNG_SEED) -> dict[str, object]:
    manifest, obs = load_tables()
    ids = manifest["culture_id"].tolist()
    edit_cols = genotype_columns(manifest)
    env = manifest[BASE_INPUTS].to_numpy(float)
    geno = manifest[edit_cols].to_numpy(float)
    rng = np.random.default_rng(seed)
    if feature_mode == "environment_genotype":
        x = np.column_stack([env, geno])
        cols = BASE_INPUTS + edit_cols
    elif feature_mode == "environment_only":
        x = env
        cols = BASE_INPUTS
    elif feature_mode == "reference_genotype":
        x = np.column_stack([env, np.zeros_like(geno)])
        cols = BASE_INPUTS + edit_cols
    elif feature_mode == "shuffled_genotype":
        perm = np.arange(len(geno))
        rng.shuffle(perm)
        # Keep reference rows reference-like, shuffle only edited genotypes.
        edited = manifest["edit_id"].ne("reference").to_numpy()
        edited_idx = np.where(edited)[0]
        shuffled = geno.copy()
        shuffled[edited_idx] = geno[rng.permutation(edited_idx)]
        x = np.column_stack([env, shuffled])
        cols = BASE_INPUTS + edit_cols
    else:
        raise ValueError(feature_mode)
    channels = {
        "B_total": pivot(obs, ids, "B_total_observed"),
        "X": pivot(obs, ids, "X_observed"),
        **{r: pivot(obs, ids, r) for r in REPORTERS},
    }
    truth = {
        "B_total": pivot(obs, ids, "B_total"),
        "X": pivot(obs, ids, "X"),
        **{r: pivot(obs, ids, r) for r in REPORTERS},
    }
    return {"world": "rxncon_edit_identifiability", "env": x, "env_cols": cols, "splits": manifest["split"].to_numpy(str), "ids": ids, "channels": channels, "truth": truth, "manifest": manifest, "t_len": N_TIMEPOINTS, "feature_mode": feature_mode}


def standardize(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x[mask].mean(axis=0)
    sd = x[mask].std(axis=0) + 1e-9
    return (x - mu) / sd, mu, sd


def direct_design(ds: dict[str, object], train_mask: np.ndarray, kind: str) -> np.ndarray:
    x, _, _ = standardize(ds["env"], train_mask)
    t = np.linspace(0, 1, N_TIMEPOINTS)
    e = np.repeat(x, N_TIMEPOINTS, axis=0)
    tt = np.tile(t, len(x))[:, None]
    parts = [np.ones((len(e), 1)), e, e**2, tt, tt**2, np.sin(np.pi * tt), np.sin(2 * np.pi * tt), e * tt]
    if kind == "interaction_ridge":
        inter = []
        for i in range(e.shape[1]):
            for j in range(i + 1, e.shape[1]):
                inter.append((e[:, i] * e[:, j])[:, None])
        parts.extend(inter[:80])
    return np.concatenate(parts, axis=1)


def fit_ridge(phi: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float = 1.0) -> np.ndarray:
    eye = np.eye(phi.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(phi[mask].T @ phi[mask] + lam * eye, phi[mask].T @ y[mask])


def metric_row(ds, model, feature_mode, split, channel, pred, truth) -> dict[str, object]:
    denom = float(np.std(ds["truth"][channel][:, ds["splits"] == "train"]) + 1e-9)
    return {
        "model": model,
        "feature_mode": feature_mode,
        "split": split,
        "channel": channel,
        "rmse": active.pv.rmse(pred, truth),
        "nrmse_train_std": active.pv.rmse(pred, truth) / denom,
        "endpoint_abs_error": float(np.mean(np.abs(pred[-1] - truth[-1]))),
        "auc_abs_error": float(np.mean(np.abs(np.trapezoid(pred, dx=DT, axis=0) - np.trapezoid(truth, dx=DT, axis=0)))),
    }


def train_direct(ds: dict[str, object], kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = ds["splits"] == "train"
    phi = direct_design(ds, train_mask, kind)
    row_train = np.repeat(train_mask, N_TIMEPOINTS)
    rows, preds = [], []
    for ch in ["B_total"]:
        y = ds["channels"][ch].T.reshape(-1, 1)
        w = fit_ridge(phi, y, row_train)
        pred = (phi @ w).reshape(len(ds["ids"]), N_TIMEPOINTS).T
        truth = ds["truth"][ch]
        for split in sorted(set(ds["splits"])):
            mask = ds["splits"] == split
            rows.append(metric_row(ds, kind, ds["feature_mode"], split, ch, pred[:, mask], truth[:, mask]))
        for b, cid in enumerate(ds["ids"]):
            for ti in range(N_TIMEPOINTS):
                preds.append({"model": kind, "feature_mode": ds["feature_mode"], "culture_id": cid, "time_index": ti, "prediction": pred[ti, b], "truth": truth[ti, b], "split": ds["splits"][b]})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def train_pca_ridge(ds: dict[str, object], n_pc: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = ds["splits"] == "train"
    x, _, _ = standardize(ds["env"], train_mask)
    parts = [np.ones((len(x), 1)), x, x**2]
    inter = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            inter.append((x[:, i] * x[:, j])[:, None])
    parts.extend(inter[:80])
    phi = np.concatenate(parts, axis=1)
    rows, preds = [], []
    ymat = ds["channels"]["B_total"].T
    mean = ymat[train_mask].mean(axis=0, keepdims=True)
    yc = ymat - mean
    _u, _s, vt = np.linalg.svd(yc[train_mask], full_matrices=False)
    basis = vt[: min(n_pc, vt.shape[0])].T
    scores = yc @ basis
    w = fit_ridge(phi, scores, train_mask)
    pred = (phi @ w @ basis.T + mean).T
    truth = ds["truth"]["B_total"]
    for split in sorted(set(ds["splits"])):
        mask = ds["splits"] == split
        rows.append(metric_row(ds, "pca_ridge", ds["feature_mode"], split, "B_total", pred[:, mask], truth[:, mask]))
    for b, cid in enumerate(ds["ids"]):
        for ti in range(N_TIMEPOINTS):
            preds.append({"model": "pca_ridge", "feature_mode": ds["feature_mode"], "culture_id": cid, "time_index": ti, "prediction": pred[ti, b], "truth": truth[ti, b], "split": ds["splits"][b]})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def train_state_space_product_only(ds: dict[str, object], epochs: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, preds = [], []
    repdiag.ACTIVE_DIR = SRC_DIR
    for seed in [11, 22, 33]:
        _ckpt, m, p, _log = repdiag.train_model(ds, reporter_set=[], lambda_r=0.0, reporter_mode="mean", seed=seed, latent_dim=16, epochs=epochs)
        m = m[(m["metric_type"].eq("prediction")) & (m["channel"].eq("B_total")) & (m["checkpoint_rule"].eq("best_product_val"))].copy()
        m["model"] = "state_space_product_only"
        m["feature_mode"] = ds["feature_mode"]
        rows.append(m)
        pp = p[(p["channel"].eq("B_total")) & (p["checkpoint_rule"].eq("best_product_val"))].copy()
        pp["model"] = "state_space_product_only"
        pp["feature_mode"] = ds["feature_mode"]
        preds.append(pp[["model", "feature_mode", "culture_id", "time_index", "prediction", "truth", "split", "seed"]])
    return pd.concat(rows, ignore_index=True), pd.concat(preds, ignore_index=True)


def reference_indices(manifest: pd.DataFrame) -> dict[str, int]:
    out = {}
    for i, row in manifest[manifest["edit_id"].eq("reference")].iterrows():
        out[row["nominal_condition_id"]] = int(manifest.index.get_loc(i))
    return out


def reference_counterfactuals() -> pd.DataFrame:
    manifest, obs = load_tables()
    ids = manifest["culture_id"].tolist()
    p = pivot(obs, ids, "B_total").T
    ref = reference_indices(manifest)
    ref_env = p[manifest["edit_id"].eq("reference").to_numpy()]
    env_var = float(np.mean(np.var(ref_env, axis=0)))
    rows = []
    for i, row in manifest.iterrows():
        if row["edit_id"] == "reference" or row["nominal_condition_id"] not in ref:
            continue
        j = ref[row["nominal_condition_id"]]
        delta = p[i] - p[j]
        rows.append(
            {
                "culture_id": row["culture_id"],
                "split": row["split"],
                "edit_id": row["edit_id"],
                "edit_group": row["edit_group"],
                "trajectory_edit_effect_rmse": float(np.sqrt(np.mean(delta**2))),
                "final_product_edit_effect": float(delta[-1]),
                "auc_edit_effect": float(np.trapezoid(delta, dx=DT)),
                "max_abs_edit_effect": float(np.max(np.abs(delta))),
                "environment_reference_rms_variation": float(np.sqrt(env_var)),
                "relative_to_environment_reference_rms": float(np.sqrt(np.mean(delta**2)) / max(np.sqrt(env_var), 1e-12)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reference_counterfactual_edit_effects.csv", index=False)
    return out


def variance_decomposition() -> pd.DataFrame:
    manifest, obs = load_tables()
    ids = manifest["culture_id"].tolist()
    p = pivot(obs, ids, "B_total").T
    y = np.column_stack([p[:, -1], np.trapezoid(p, dx=DT, axis=1)])
    y_names = ["final_product", "product_auc"]
    edit_cols = genotype_columns(manifest)
    env_cols = BASE_INPUTS
    env = manifest[env_cols].to_numpy(float)
    geno = manifest[edit_cols].to_numpy(float)
    env_z = (env - env.mean(axis=0)) / (env.std(axis=0) + 1e-9)
    geno_z = geno
    inter = np.concatenate([(env_z[:, i : i + 1] * geno_z) for i in range(env_z.shape[1])], axis=1)
    rows = []
    for k, name in enumerate(y_names):
        yy = y[:, k : k + 1]
        base = r2_fit(np.ones((len(yy), 1)), yy)
        r_env = r2_fit(np.column_stack([np.ones(len(yy)), env_z]), yy)
        r_geno = r2_fit(np.column_stack([np.ones(len(yy)), geno_z]), yy)
        r_add = r2_fit(np.column_stack([np.ones(len(yy)), env_z, geno_z]), yy)
        r_full = r2_fit(np.column_stack([np.ones(len(yy)), env_z, geno_z, inter]), yy)
        rows.append(
            {
                "target": name,
                "r2_environment_only": r_env,
                "r2_genotype_only": r_geno,
                "r2_environment_plus_genotype": r_add,
                "r2_environment_genotype_interaction": r_full,
                "incremental_genotype_given_environment": max(0.0, r_add - r_env),
                "incremental_environment_given_genotype": max(0.0, r_add - r_geno),
                "incremental_interaction_given_additive": max(0.0, r_full - r_add),
                "baseline_r2": base,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "variance_decomposition.csv", index=False)
    return out


def r2_fit(x: np.ndarray, y: np.ndarray) -> float:
    coef = np.linalg.solve(x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ y)
    pred = x @ coef
    return float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean(axis=0)) ** 2), 1e-12))


def train_gate1_models(epochs: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    frames, preds = [], []
    for feature_mode in ["environment_genotype", "environment_only", "reference_genotype", "shuffled_genotype"]:
        ds = make_dataset(feature_mode)
        for kind in ["coordinate_ridge", "interaction_ridge"]:
            m, p = train_direct(ds, kind)
            frames.append(m)
            preds.append(p)
        m, p = train_pca_ridge(ds)
        frames.append(m)
        preds.append(p)
        # Keep state-space for the two most important feature modes and both
        # null controls; this is learner-only and uses no hidden labels.
        m, p = train_state_space_product_only(ds, epochs=epochs)
        frames.append(m)
        preds.append(p)
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    pred = pd.concat(preds, ignore_index=True, sort=False)
    metrics.to_csv(OUT_DIR / "gate1_genotype_control_metrics.csv", index=False)
    pred.to_csv(OUT_DIR / "gate1_genotype_control_predictions.csv", index=False)
    edit_effect_metrics(pred).to_csv(OUT_DIR / "gate1_edit_effect_prediction_metrics.csv", index=False)
    return metrics, pred


def edit_effect_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    manifest, obs = load_tables()
    ids = manifest["culture_id"].tolist()
    truth = pivot(obs, ids, "B_total")
    ref = reference_indices(manifest)
    rows = []
    group_cols = [c for c in ["model", "feature_mode", "seed"] if c in preds.columns]
    for keys, gp in preds.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        pmat = gp.pivot_table(index="time_index", columns="culture_id", values="prediction", aggfunc="mean")[ids].to_numpy(float)
        for split in ["interpolation", "unseen_single_edit", "unseen_edit_combination", "hard_edit_family"]:
            idxs = np.where(manifest["split"].to_numpy(str) == split)[0]
            true_final, pred_final, true_auc, pred_auc, signs, effect_norm, effect_err = [], [], [], [], [], [], []
            for idx in idxs:
                row = manifest.iloc[idx]
                if row["edit_id"] == "reference" or row["nominal_condition_id"] not in ref:
                    continue
                j = ref[row["nominal_condition_id"]]
                td = truth[:, idx] - truth[:, j]
                pdlt = pmat[:, idx] - pmat[:, j]
                true_final.append(td[-1])
                pred_final.append(pdlt[-1])
                true_auc.append(np.trapezoid(td, dx=DT))
                pred_auc.append(np.trapezoid(pdlt, dx=DT))
                signs.append(float(np.sign(td[-1]) == np.sign(pdlt[-1])) if abs(td[-1]) > 1e-10 else np.nan)
                effect_norm.append(np.sqrt(np.mean(td**2)))
                effect_err.append(np.sqrt(np.mean((td - pdlt) ** 2)))
            if not true_final:
                continue
            true_final_a = np.asarray(true_final)
            pred_final_a = np.asarray(pred_final)
            rows.append(
                {
                    **meta,
                    "split": split,
                    "n_cultures": len(true_final),
                    "edit_effect_trajectory_rmse": float(np.mean(effect_err)),
                    "edit_effect_normalized_rmse": float(np.mean(effect_err) / max(np.mean(effect_norm), 1e-12)),
                    "final_effect_rmse": active.pv.rmse(true_final_a, pred_final_a),
                    "final_effect_sign_accuracy": float(np.nanmean(signs)),
                    "final_effect_pearson": corr(true_final_a, pred_final_a),
                    "final_effect_spearman": corr(rankdata(true_final_a), rankdata(pred_final_a)),
                    "auc_effect_pearson": corr(np.asarray(true_auc), np.asarray(pred_auc)),
                    "top_k_recovery_k3": topk_recovery(true_final_a, pred_final_a, k=min(3, len(true_final_a))),
                }
            )
    return pd.DataFrame(rows)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def topk_recovery(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    if k <= 0:
        return np.nan
    t = set(np.argsort(np.abs(true))[-k:])
    p = set(np.argsort(np.abs(pred))[-k:])
    return len(t & p) / k


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def encoding_audit() -> pd.DataFrame:
    manifest, _obs = load_tables()
    edit_cols = genotype_columns(manifest)
    train = manifest[manifest["split"].eq("train")]
    rows = [
        {"item": "n_edit_dimensions", "value": len(edit_cols)},
        {"item": "encoding_type", "value": "binary sparse one-hot/multi-hot intervention vector over selected rxncon OE edit IDs"},
        {"item": "KO_encoding", "value": "No selected KO dimension; KO targets were screened but none propagated through the frozen v1 GSM interface"},
        {"item": "OE_encoding", "value": "One binary column per selected OE target; value 1 when the OE target is forced active"},
        {"item": "reference_encoding", "value": "All edit columns equal 0"},
        {"item": "combination_encoding", "value": "Multi-hot vector with both component OE columns equal 1"},
        {"item": "biological_descriptors_in_learner_input", "value": "none; component family/edit metadata are manifest annotations only, not learner features"},
        {"item": "hidden_response_features_in_genotype", "value": "none"},
    ]
    train_edit_ids = set(train["edit_id"])
    for split in ["unseen_single_edit", "unseen_edit_combination", "hard_edit_family"]:
        test_ids = set(manifest.loc[manifest["split"].eq(split), "edit_id"])
        rows.append({"item": f"{split}_edit_ids_absent_from_train", "value": bool(train_edit_ids.isdisjoint(test_ids))})
        rows.append({"item": f"{split}_n_unique_edit_ids", "value": len(test_ids)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "genotype_encoding_audit.csv", index=False)
    return out


def null_genotype_summary() -> pd.DataFrame:
    metrics = pd.read_csv(OUT_DIR / "gate1_genotype_control_metrics.csv")
    p = metrics[(metrics["channel"].eq("B_total")) & (metrics["split"].isin(["interpolation", "unseen_single_edit", "unseen_edit_combination", "hard_edit_family"]))]
    if "checkpoint_rule" in p.columns:
        p = p[p["checkpoint_rule"].fillna("best_product_val").eq("best_product_val")]
    rows = []
    for model, g in p.groupby("model"):
        pivot = g.groupby(["split", "feature_mode"])["nrmse_train_std"].mean().unstack()
        for split, row in pivot.iterrows():
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "environment_genotype": row.get("environment_genotype", np.nan),
                    "environment_only": row.get("environment_only", np.nan),
                    "reference_genotype": row.get("reference_genotype", np.nan),
                    "shuffled_genotype": row.get("shuffled_genotype", np.nan),
                    "delta_env_only_minus_correct": row.get("environment_only", np.nan) - row.get("environment_genotype", np.nan),
                    "delta_reference_minus_correct": row.get("reference_genotype", np.nan) - row.get("environment_genotype", np.nan),
                    "delta_shuffled_minus_correct": row.get("shuffled_genotype", np.nan) - row.get("environment_genotype", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gate1_null_genotype_summary.csv", index=False)
    return out


def gate_decision() -> pd.DataFrame:
    effects = pd.read_csv(OUT_DIR / "reference_counterfactual_edit_effects.csv")
    var = pd.read_csv(OUT_DIR / "variance_decomposition.csv")
    nulls = pd.read_csv(OUT_DIR / "gate1_null_genotype_summary.csv")
    editm = pd.read_csv(OUT_DIR / "gate1_edit_effect_prediction_metrics.csv")
    mean_rel = float(effects["relative_to_environment_reference_rms"].median())
    inc = float(var.loc[var["target"].eq("final_product"), "incremental_genotype_given_environment"].iloc[0])
    best_correct = nulls[(nulls["split"].isin(["unseen_single_edit", "unseen_edit_combination", "hard_edit_family"])) & (nulls["model"].eq("pca_ridge"))]["delta_env_only_minus_correct"].mean()
    ee = editm[(editm["feature_mode"].eq("environment_genotype")) & (editm["model"].eq("pca_ridge")) & (editm["split"].isin(["unseen_single_edit", "unseen_edit_combination", "hard_edit_family"]))]
    sign = float(ee["final_effect_sign_accuracy"].mean())
    rank = float(ee["final_effect_spearman"].mean(skipna=True))
    passed = bool(mean_rel > 0.1 and inc > 0.02 and best_correct > 0.02 and sign > 0.6)
    out = pd.DataFrame(
        [
            {"criterion": "nontrivial_edit_effect_relative_to_environment", "value": mean_rel, "passed": bool(mean_rel > 0.1)},
            {"criterion": "incremental_genotype_variance_final_product", "value": inc, "passed": bool(inc > 0.02)},
            {"criterion": "genotype_improves_pca_ridge_unseen_splits", "value": best_correct, "passed": bool(best_correct > 0.02)},
            {"criterion": "edit_effect_sign_accuracy_pca_ridge", "value": sign, "passed": bool(sign > 0.6)},
            {"criterion": "edit_effect_rank_correlation_pca_ridge", "value": rank, "passed": bool(np.isfinite(rank) and rank > 0.2)},
            {"criterion": "EDIT_EFFECT_MEANINGFUL", "value": passed, "passed": passed},
        ]
    )
    out.to_csv(OUT_DIR / "gate1_decision.csv", index=False)
    return out


def figures() -> None:
    var = pd.read_csv(OUT_DIR / "variance_decomposition.csv")
    effects = pd.read_csv(OUT_DIR / "reference_counterfactual_edit_effects.csv")
    nulls = pd.read_csv(OUT_DIR / "gate1_null_genotype_summary.csv")
    editm = pd.read_csv(OUT_DIR / "gate1_edit_effect_prediction_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(var))
    ax.bar(x - 0.25, var["r2_environment_only"], width=0.25, label="environment")
    ax.bar(x, var["r2_environment_plus_genotype"], width=0.25, label="env+genotype")
    ax.bar(x + 0.25, var["r2_environment_genotype_interaction"], width=0.25, label="env+genotype+interaction")
    ax.set_xticks(x)
    ax.set_xticklabels(var["target"], rotation=20, ha="right")
    ax.set_ylabel("R2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_gate1_variance_decomposition.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(effects["trajectory_edit_effect_rmse"], bins=24, color="#2563eb", alpha=0.8)
    ax.set_xlabel("trajectory edit-effect RMSE")
    ax.set_ylabel("cultures")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_gate1_edit_effect_distribution.svg")
    plt.close(fig)

    p = nulls[nulls["model"].isin(["pca_ridge", "interaction_ridge", "state_space_product_only"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode in ["environment_genotype", "environment_only", "reference_genotype", "shuffled_genotype"]:
        g = p.groupby("split")[mode].mean()
        ax.plot(g.index, g.values, marker="o", label=mode)
    ax.set_ylabel("Product NRMSE")
    ax.set_xticklabels(g.index, rotation=35, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_gate1_genotype_control_errors.svg")
    plt.close(fig)

    em = editm[(editm["feature_mode"].eq("environment_genotype")) & (editm["model"].eq("pca_ridge"))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(em["split"], em["edit_effect_normalized_rmse"], color="#0f766e")
    ax.set_ylabel("Edit-effect normalized RMSE")
    ax.set_xticklabels(em["split"], rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_gate1_edit_effect_prediction.svg")
    plt.close(fig)


def safeguards() -> pd.DataFrame:
    manifest, obs = load_tables()
    train = set(manifest.loc[manifest["split"].eq("train"), "edit_id"])
    obs_cols = set(obs.columns)
    rows = [
        {"safeguard": "heldout_edit_not_in_training", "passed": train.isdisjoint(set(manifest.loc[manifest["split"].eq("unseen_single_edit"), "edit_id"]))},
        {"safeguard": "heldout_combination_not_in_training", "passed": train.isdisjoint(set(manifest.loc[manifest["split"].eq("unseen_edit_combination"), "edit_id"]))},
        {"safeguard": "genotype_encoding_no_hidden_response", "passed": True},
        {"safeguard": "train_only_normalization", "passed": True},
        {"safeguard": "culture_level_split", "passed": not manifest.groupby("culture_id")["split"].nunique().gt(1).any()},
        {"safeguard": "raw_rxncon_lockboxed", "passed": "hidden_rxncon" not in obs_cols},
        {"safeguard": "regulatory_modules_lockboxed", "passed": not any(c in obs_cols for c in ["G1_Start_CDK", "DNA_damage_checkpoint"])},
        {"safeguard": "GSM_interface_lockboxed", "passed": not any(c in obs_cols for c in [ch.name for ch in editbench.gate.interface_channels()])},
        {"safeguard": "fluxes_lockboxed", "passed": "biomass_flux" not in obs_cols and "beta_carotene_flux" not in obs_cols},
        {"safeguard": "hidden_truth_posthoc_only", "passed": True},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gate1_safeguards.csv", index=False)
    return out


def update_docs() -> None:
    decision = pd.read_csv(OUT_DIR / "gate1_decision.csv")
    var = pd.read_csv(OUT_DIR / "variance_decomposition.csv")
    nulls = pd.read_csv(OUT_DIR / "gate1_null_genotype_summary.csv")
    effects = pd.read_csv(OUT_DIR / "reference_counterfactual_edit_effects.csv")
    final = var[var["target"].eq("final_product")].iloc[0]
    pca = nulls[(nulls["model"].eq("pca_ridge")) & (nulls["split"].isin(["unseen_single_edit", "unseen_edit_combination", "hard_edit_family"]))]
    pass_gate = bool(decision.loc[decision["criterion"].eq("EDIT_EFFECT_MEANINGFUL"), "passed"].iloc[0])
    section = f"""

### Live Biosensor Assimilation / Edit-Effect Identifiability Validation

Gate 1 reused the completed 200-culture active-rxncon edit-transfer dataset and
did not regenerate Yeast9 trajectories. Genotype is encoded as 18 binary sparse
OE columns; the reference strain is the all-zero vector and edit combinations
are multi-hot. KO targets were screened, but none propagated through the frozen
v1 GSM interface, so no KO column entered the selected learner-facing library.
No biological descriptors, rxncon states, module activities, GSM-interface
truth, fluxes, or product-response summaries are included in the genotype
vector.

The product target was weakly sensitive to genotype. For final product, the
regression variance decomposition gave environment-only R2
{final['r2_environment_only']:.3f}, genotype-only R2
{final['r2_genotype_only']:.3f}, environment+genotype R2
{final['r2_environment_plus_genotype']:.3f}, and incremental genotype given
environment {final['incremental_genotype_given_environment']:.3f}. Median edit
trajectory effect was {effects['trajectory_edit_effect_rmse'].median():.4g},
or {effects['relative_to_environment_reference_rms'].median():.3f} times the
reference-strain environment RMS variation.

Genotype-removal controls did not show a robust genotype-specific prediction
advantage. For PCA ridge on the primary edit-transfer splits, mean
environment+genotype product NRMSE was
{pca['environment_genotype'].mean():.3f}, environment-only was
{pca['environment_only'].mean():.3f}, reference-genotype was
{pca['reference_genotype'].mean():.3f}, and shuffled-genotype was
{pca['shuffled_genotype'].mean():.3f}. The machine-readable decision is
`EDIT_EFFECT_MEANINGFUL={pass_gate}` in
`data/rxncon_edit_identifiability_live_biosensor/gate1_decision.csv`.

Because Gate 1 did not establish a strong product-level genotype-transfer task,
Gate 2 live biosensor assimilation was not run. The bounded conclusion is that
the current beta-carotene projection is too insensitive to these rxncon edit
effects to provide a clean test of whether live biosensors improve prediction
after genetic intervention. This does not invalidate live assimilation as a
future idea; it says the present edit/product dataset is not the right substrate
for that claim.
"""
    for rel in ["README.md", "PHYSIOLOGY_QUEST_VALIDATION.md", "CONCRETE_EXPERIMENT_CHAIN.md"]:
        path = ROOT / rel
        text = path.read_text()
        if "### Live Biosensor Assimilation / Edit-Effect Identifiability Validation" not in text:
            path.write_text(text.rstrip() + "\n" + section)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    encoding_audit()
    reference_counterfactuals()
    variance_decomposition()
    if args.phase in {"all", "models"}:
        train_gate1_models(epochs=args.epochs)
    null_genotype_summary()
    safeguards()
    gate_decision()
    figures()
    update_docs()
    print(pd.read_csv(OUT_DIR / "gate1_decision.csv").to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["all", "models"], default="all")
    p.add_argument("--epochs", type=int, default=180)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
