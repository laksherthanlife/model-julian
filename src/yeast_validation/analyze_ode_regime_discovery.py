#!/usr/bin/env python3
"""Unsupervised regime discovery from product, physiology, and learned states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_regime_core as ode
from audit_ode_regime_experiment import pairwise_regime_distances


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load(prefix: str):
    traj = pd.read_csv(DATA / f"{prefix}_trajectories.csv")
    diag = pd.read_csv(DATA / f"{prefix}_diagnostics.csv")
    reps = pd.read_csv(DATA / f"{prefix}_learned_representations.csv")
    meta, mats, _t = ode.pivot_trajectories(traj, ode.STATE_COLUMNS + ode.AUX_COLUMNS)
    regimes = diag.set_index("culture_id").loc[meta["culture_id"], "dominant_regime"].to_numpy(str)
    return meta, mats, diag, reps, regimes


def feature_sets(meta: pd.DataFrame, mats: dict[str, np.ndarray], reps: pd.DataFrame, model_seed: int) -> dict[str, np.ndarray]:
    curves = mats["P"]
    product_norm = ode.normalized_curves(curves)
    product_mean, product_comps, _ = ode.pca_fit(product_norm)
    phys = np.hstack([mats[col] for col in ["X", "A", "R", "E", "S", "C", "v_product", "mu"]])
    phys_mean, phys_comps, _ = ode.pca_fit(phys)
    out = {
        "raw_product_trajectories": curves,
        "normalized_product_pca": (product_norm - product_mean) @ product_comps[:8].T,
        "true_physiology_upper_bound": (phys - phys_mean) @ phys_comps[:8].T,
    }
    for model in sorted(reps["model"].unique()):
        sub = reps[(reps["model"].eq(model)) & (reps["model_seed"].eq(model_seed))].set_index("culture_id")
        if sub.empty:
            continue
        feature_cols = [c for c in sub.columns if c.startswith("latent_feature_")]
        aligned = sub.loc[meta["culture_id"], feature_cols].to_numpy(float)
        out[f"learned_latent_{model}"] = aligned
    return out


def evaluate_clusters(features: dict[str, np.ndarray], regimes: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, X in features.items():
        X = np.asarray(X, dtype=float)
        if X.shape[1] > 12:
            mean, comps, _ = ode.pca_fit(X)
            X = (X - mean) @ comps[: min(12, comps.shape[0])].T
        for k in [5, 6, 7, 8]:
            if k >= len(X):
                continue
            labels, _centers = ode.kmeans(X, k, seed=1234 + k)
            rows.append(
                {
                    "feature_set": name,
                    "n_clusters": k,
                    "ari": ode.adjusted_rand_index(regimes, labels),
                    "nmi": ode.normalized_mutual_info(regimes, labels),
                    "purity": ode.cluster_purity(regimes, labels),
                    "silhouette": ode.silhouette_score(X, labels),
                }
            )
    return pd.DataFrame(rows)


def latent_correlations(reps: pd.DataFrame, diag: pd.DataFrame, model_seed: int) -> pd.DataFrame:
    diagnostic_cols = [
        "integrated_stress",
        "min_energy",
        "final_precursor",
        "pathway_capacity_loss",
        "max_congestion",
        "endpoint_biomass",
        "growth_to_product_allocation",
    ]
    rows = []
    for model in sorted(reps["model"].unique()):
        sub = reps[(reps["model"].eq(model)) & (reps["model_seed"].eq(model_seed))].merge(diag[["culture_id"] + diagnostic_cols], on="culture_id")
        feature_cols = [c for c in sub.columns if c.startswith("latent_feature_")]
        for feature in feature_cols:
            x = sub[feature].to_numpy(float)
            for dcol in diagnostic_cols:
                y = sub[dcol].to_numpy(float)
                corr = 0.0 if np.std(x) < ode.EPS or np.std(y) < ode.EPS else float(np.corrcoef(x, y)[0, 1])
                if abs(corr) >= 0.35:
                    rows.append({"model": model, "latent_feature": feature, "diagnostic": dcol, "correlation": corr, "interpretation_guardrail": "correlation_only_not_unique_biological_identity"})
    return pd.DataFrame(rows).sort_values(["model", "diagnostic", "correlation"], ascending=[True, True, False])


def confusion_for_best(features: dict[str, np.ndarray], regimes: np.ndarray, cluster_metrics: pd.DataFrame) -> pd.DataFrame:
    best_name = cluster_metrics.sort_values(["ari", "purity"], ascending=False).iloc[0]["feature_set"]
    X = features[best_name]
    if X.shape[1] > 12:
        mean, comps, _ = ode.pca_fit(X)
        X = (X - mean) @ comps[: min(12, comps.shape[0])].T
    labels, _centers = ode.kmeans(X, int(cluster_metrics[cluster_metrics.feature_set.eq(best_name)].sort_values("ari", ascending=False).iloc[0]["n_clusters"]), seed=999)
    rows = []
    for true in sorted(set(regimes)):
        for cluster in sorted(set(labels)):
            rows.append({"feature_set": best_name, "true_regime": true, "cluster": int(cluster), "count": int(np.sum((regimes == true) & (labels == cluster)))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--model-seed", type=int, default=11)
    args = parser.parse_args()
    prefix = "ode_regime_pilot" if args.pilot else "ode_regime"
    meta, mats, diag, reps, regimes = load(prefix)
    features = feature_sets(meta, mats, reps, args.model_seed)
    cluster_metrics = evaluate_clusters(features, regimes)
    cluster_metrics.to_csv(DATA / f"{prefix}_regime_discovery_metrics.csv", index=False)
    latent_correlations(reps, diag, args.model_seed).to_csv(DATA / f"{prefix}_latent_diagnostic_correlations.csv", index=False)
    confusion_for_best(features, regimes, cluster_metrics).to_csv(DATA / f"{prefix}_regime_discovery_confusion.csv", index=False)
    best_latent = [name for name in features if name.startswith("learned_latent_")]
    distance_rows = []
    for name in ["normalized_product_pca", "true_physiology_upper_bound"] + best_latent:
        distance_rows.append(pairwise_regime_distances(features[name], regimes).assign(feature_set=name))
    pd.concat(distance_rows, ignore_index=True).to_csv(DATA / f"{prefix}_regime_discovery_pairwise_distances.csv", index=False)
    print(f"{prefix} regime discovery analysis complete")
    print(cluster_metrics.sort_values(["ari", "purity"], ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
