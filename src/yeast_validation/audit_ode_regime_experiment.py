#!/usr/bin/env python3
"""Audit numerical stability and trajectory diversity of the ODE regime testbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ode_regime_core as ode


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def load(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(DATA / f"{prefix}_environment_manifest.csv")
    traj = pd.read_csv(DATA / f"{prefix}_trajectories.csv")
    diag = pd.read_csv(DATA / f"{prefix}_diagnostics.csv")
    return manifest, traj, diag


def pca_summary(name: str, X: np.ndarray) -> pd.DataFrame:
    _mean, _comps, explained = ode.pca_fit(X)
    rows = []
    for k in [1, 2, 3, 5, 8, 12]:
        kk = min(k, len(explained))
        rows.append({"basis": name, "components": kk, "cumulative_explained_variance": float(np.sum(explained[:kk]))})
    return pd.DataFrame(rows)


def pairwise_regime_distances(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    regimes = sorted(set(labels))
    for a in regimes:
        ia = np.where(labels == a)[0]
        if len(ia) < 2:
            continue
        da = np.sqrt(np.sum((features[ia][:, None, :] - features[ia][None, :, :]) ** 2, axis=2))
        rows.append({"comparison": "within", "regime_a": a, "regime_b": a, "mean_distance": float(np.mean(da[np.triu_indices(len(ia), 1)]))})
    for i, a in enumerate(regimes):
        for b in regimes[i + 1 :]:
            ia = np.where(labels == a)[0]
            ib = np.where(labels == b)[0]
            if len(ia) and len(ib):
                d = np.sqrt(np.sum((features[ia][:, None, :] - features[ib][None, :, :]) ** 2, axis=2))
                rows.append({"comparison": "between", "regime_a": a, "regime_b": b, "mean_distance": float(np.mean(d))})
    return pd.DataFrame(rows)


def cluster_audit(curves: np.ndarray, physiology: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, X in [("normalized_product", ode.normalized_curves(curves)), ("physiological_trajectories", physiology)]:
        mean, comps, explained = ode.pca_fit(X)
        k = min(8, comps.shape[0])
        feat = (X - mean) @ comps[:k].T
        for n_clusters in [5, 6, 7, 8]:
            pred, _centers = ode.kmeans(feat, n_clusters, seed=900 + n_clusters)
            rows.append(
                {
                    "feature_set": name,
                    "n_clusters": n_clusters,
                    "ari": ode.adjusted_rand_index(regimes, pred),
                    "nmi": ode.normalized_mutual_info(regimes, pred),
                    "purity": ode.cluster_purity(regimes, pred),
                    "silhouette": ode.silhouette_score(feat, pred),
                    "pc1_explained": float(explained[0]),
                }
            )
    return pd.DataFrame(rows)


def smoothness_audit(diag: pd.DataFrame) -> pd.DataFrame:
    atlas = diag[diag["split"].eq("atlas_grid")].copy()
    if atlas.empty:
        return pd.DataFrame()
    env = atlas[["temperature", "pH", "DO"]].to_numpy(float)
    y = atlas[["final_product", "integrated_stress", "min_energy", "final_precursor", "max_congestion"]].to_numpy(float)
    rows = []
    for i in range(len(atlas)):
        dist = np.sum(((env - env[i]) / np.array([6.0, 1.0, 60.0])) ** 2, axis=1)
        nn = np.argsort(dist)[1:7]
        delta = np.sqrt(np.sum((y[nn] - y[i]) ** 2, axis=1))
        regime_switch = atlas.iloc[nn]["dominant_regime"].ne(atlas.iloc[i]["dominant_regime"]).mean()
        rows.append({"culture_id": atlas.iloc[i]["culture_id"], "mean_neighbor_delta": float(delta.mean()), "neighbor_regime_switch_fraction": float(regime_switch)})
    out = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "mean_neighbor_delta": float(out["mean_neighbor_delta"].mean()),
                "p95_neighbor_delta": float(out["mean_neighbor_delta"].quantile(0.95)),
                "mean_neighbor_regime_switch_fraction": float(out["neighbor_regime_switch_fraction"].mean()),
                "boundary_like_fraction": float(out["neighbor_regime_switch_fraction"].gt(0).mean()),
            }
        ]
    )


def save_figures(traj: pd.DataFrame, diag: pd.DataFrame, prefix: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    counts = diag["dominant_regime"].value_counts().reindex(ode.REGIME_ORDER).dropna()
    plt.figure(figsize=(9, 4.8))
    counts.plot(kind="bar", color="#2f6f9f")
    plt.ylabel("cultures")
    plt.title("ODE Physiological Regime Counts")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{prefix}_regime_counts.png", dpi=180)
    plt.close()

    reps = []
    for regime in counts.index:
        sub = diag[diag["dominant_regime"].eq(regime)]
        if len(sub):
            target = sub["final_product"].median()
            reps.append(sub.iloc[(sub["final_product"] - target).abs().argsort().iloc[0]]["culture_id"])
    fig, axes = plt.subplots(len(reps), 2, figsize=(10, max(5, 2.1 * len(reps))), sharex=True)
    if len(reps) == 1:
        axes = np.array([axes])
    for axrow, cid in zip(axes, reps):
        g = traj[traj["culture_id"].eq(cid)].sort_values("time")
        regime = diag[diag["culture_id"].eq(cid)]["dominant_regime"].iloc[0]
        axrow[0].plot(g["time"], g["P"], color="#111111", lw=2)
        axrow[0].set_ylabel(regime.replace("_", "\n"), fontsize=8)
        for col, color in zip(["X", "A", "R", "E", "S", "C"], ["#2f6f9f", "#558b2f", "#b65d3b", "#7d5fb2", "#b23b4a", "#444444"]):
            axrow[1].plot(g["time"], g[col], label=col, color=color, lw=1.4)
    axes[0, 1].legend(ncol=6, fontsize=7)
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    axes[0, 0].set_title("beta-carotene P(t)")
    axes[0, 1].set_title("physiological states")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{prefix}_representative_regime_trajectories.png", dpi=180)
    plt.close()

    atlas = diag[diag["split"].eq("atlas_grid") & np.isclose(diag["pH"], 5.0, atol=0.09)]
    if not atlas.empty:
        code = {name: i for i, name in enumerate(ode.REGIME_ORDER)}
        plt.figure(figsize=(7, 5.5))
        plt.scatter(atlas["temperature"], atlas["DO"], c=atlas["dominant_regime"].map(code), cmap="tab10", s=80, edgecolor="k", linewidth=0.2)
        plt.xlabel("temperature")
        plt.ylabel("dissolved oxygen")
        plt.title("Dominant regime atlas near pH 5.0")
        plt.tight_layout()
        plt.savefig(FIGURES / f"{prefix}_regime_atlas_T_DO.png", dpi=180)
        plt.close()


def audit(prefix: str) -> dict[str, pd.DataFrame]:
    manifest, traj, diag = load(prefix)
    meta, mats, _time = ode.pivot_trajectories(traj, ode.STATE_COLUMNS + ode.AUX_COLUMNS)
    curves = mats["P"]
    physiology = np.hstack([mats[col] for col in ["X", "A", "R", "E", "S", "C", "v_product", "mu"]])
    regimes = diag.set_index("culture_id").loc[meta["culture_id"], "dominant_regime"].to_numpy(str)
    pca = pd.concat(
        [
            pca_summary("raw_product", curves),
            pca_summary("normalized_product", ode.normalized_curves(curves)),
            pca_summary("physiological_trajectories", physiology),
        ],
        ignore_index=True,
    )
    clusters = cluster_audit(curves, physiology, regimes)
    distances = pairwise_regime_distances(ode.normalized_curves(curves), regimes)
    smoothness = smoothness_audit(diag)
    normalized_pc1 = float(pca[(pca.basis == "normalized_product") & (pca.components == 1)]["cumulative_explained_variance"].iloc[0])
    physiology_pc1 = float(pca[(pca.basis == "physiological_trajectories") & (pca.components == 1)]["cumulative_explained_variance"].iloc[0])
    safeguards = pd.DataFrame(
        [
            ("finite_nonnegative_states", np.isfinite(traj[ode.STATE_COLUMNS + ode.AUX_COLUMNS]).all().all() and (traj[ode.STATE_COLUMNS] >= -1e-10).all().all()),
            ("multiple_regimes", diag["dominant_regime"].nunique() >= 6),
            ("normalized_product_not_one_dimensional", normalized_pc1 < 0.985),
            ("physiology_richer_than_product", physiology_pc1 < normalized_pc1 - 0.20),
            ("transition_regions_exist", diag["dominant_regime"].eq("mixed_transition").any()),
            ("heldout_blocks_exist", manifest["split"].eq("heldout_hole").any()),
        ],
        columns=["audit_check", "passed"],
    )
    save_figures(traj, diag, prefix)
    return {
        "pca": pca,
        "cluster_audit": clusters,
        "pairwise_distances": distances,
        "smoothness": smoothness,
        "safeguards": safeguards,
        "regime_counts": diag["dominant_regime"].value_counts().rename_axis("dominant_regime").reset_index(name="count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    prefix = "ode_regime_pilot" if args.pilot else "ode_regime"
    outputs = audit(prefix)
    for name, df in outputs.items():
        df.to_csv(DATA / f"{prefix}_{name}.csv", index=False)
    print(f"{prefix} audit complete")
    print(outputs["safeguards"].to_string(index=False))
    print(outputs["regime_counts"].to_string(index=False))


if __name__ == "__main__":
    main()
