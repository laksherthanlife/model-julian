#!/usr/bin/env python3
"""Aggregate final validation boundary-condition experiments.

This script intentionally derives all prospective engineering endpoints from
exact hidden-simulator outputs.  The primary comparison is the common
8-physical-culture horizon.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


DATA_SUFF_WORLDS = ("baseline_world", "strong_oxidative_burden_world")
DATA_SUFF_SEEDS = {
    "baseline_world": (98001, 98002, 98003),
    "strong_oxidative_burden_world": (98011, 98012, 98013),
}
ROBUSTNESS_WORLDS = (
    "baseline_world",
    "strong_oxidative_burden_world",
    "atp_limited_world",
    "pathway_bottleneck_damage_world",
)
ROBUSTNESS_SEEDS = {
    "baseline_world": (98001, 98002, 98003),
    "strong_oxidative_burden_world": (98011, 98012, 98013),
    "atp_limited_world": (99001, 99002, 99003),
    "pathway_bottleneck_damage_world": (99011, 99012, 99013),
}


def bootstrap_ci(values: np.ndarray, seed: int = 12345, n_boot: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[sample_idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def endpoint_from_ledger(df: pd.DataFrame, horizon: int) -> pd.Series:
    physical_col = "physical_culture_index" if "physical_culture_index" in df.columns else "culture_index"
    limited = df[df[physical_col] <= horizon].sort_values(physical_col)
    if limited.empty:
        raise ValueError(f"No rows at <= {horizon} cultures")
    if {"best_final_product_so_far", "best_productivity_so_far", "best_product_AUC_so_far"} <= set(limited.columns):
        row = limited.iloc[-1]
        return pd.Series(
            {
                "best_final_product": row["best_final_product_so_far"],
                "best_productivity": row["best_productivity_so_far"],
                "best_product_AUC": row["best_product_AUC_so_far"],
            }
        )
    best_row = limited.loc[limited["final_product"].idxmax()]
    return pd.Series(
        {
            "best_final_product": best_row["final_product"],
            "best_productivity": best_row["productivity"],
            "best_product_AUC": best_row["product_AUC"],
        }
    )


def load_new_hybrid_exact(root: Path, horizon: int) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "final_validation_hpc" / "hybrid_exact_results").glob("*_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].copy()
        rows.append(row)
    raw = pd.DataFrame(rows).reset_index(drop=True)
    if raw.empty:
        return raw
    raw["physical_culture_index"] = 1
    grouped = []
    for campaign_id, group in raw.groupby("campaign_id", sort=False):
        limited = group.sort_values(["rerank_rank", "surrogate_rank"]).head(horizon)
        best = limited.loc[limited["final_product"].idxmax()]
        grouped.append(
            {
                "source": "new_hpc",
                "experiment_group": best["experiment_group"],
                "world_id": best["world_id"],
                "campaign_seed": int(best["campaign_seed"]),
                "campaign_id": campaign_id,
                "method_id": "pretrained_hybrid_digital_twin",
                "training_cultures_n": int(best["training_cultures_n"]),
                "subsample_seed": int(best["subsample_seed"]),
                "physical_culture_budget": len(limited),
                "best_final_product": float(best["final_product"]),
                "best_productivity": float(best["productivity"]),
                "best_product_AUC": float(best["product_AUC"]),
                "best_candidate_id": best["candidate_id"],
                "n_exact_verified": len(limited),
            }
        )
    return pd.DataFrame(grouped)


def load_v2_common8(root: Path, horizon: int) -> pd.DataFrame:
    path = root / "results/repaired_prospective_benchmark_v2_aggregate/data/prospective_best_so_far_by_culture.csv"
    df = pd.read_csv(path)
    out = []
    for world, seeds in DATA_SUFF_SEEDS.items():
        for seed in seeds:
            campaign_id = f"prospective_{world}_seed_{seed}"
            for method_id in ("conventional_dbtl", "pretrained_hybrid_digital_twin"):
                subset = df[
                    (df["campaign_id"] == campaign_id)
                    & (df["method_id"] == method_id)
                    & (df["physical_culture_index"] <= horizon)
                ].sort_values("physical_culture_index")
                if subset.empty:
                    raise ValueError(f"Missing v2 endpoint for {campaign_id} {method_id}")
                row = subset.iloc[-1]
                out.append(
                    {
                        "source": "v2_reuse",
                        "experiment_group": "v2_primary_common8",
                        "world_id": world,
                        "campaign_seed": seed,
                        "campaign_id": campaign_id,
                        "method_id": method_id,
                        "training_cultures_n": 77 if method_id == "pretrained_hybrid_digital_twin" else np.nan,
                        "subsample_seed": 0 if method_id == "pretrained_hybrid_digital_twin" else np.nan,
                        "physical_culture_budget": horizon,
                        "best_final_product": float(row["best_final_product_so_far"]),
                        "best_productivity": float(row["best_productivity_so_far"]),
                        "best_product_AUC": float(row["best_product_AUC_so_far"]),
                        "best_candidate_id": row["candidate_id"],
                        "n_exact_verified": horizon,
                    }
                )
    return pd.DataFrame(out)


def load_new_conventional(root: Path, horizon: int) -> pd.DataFrame:
    rows = []
    for path in sorted(
        (root / "results/final_validation_conventional_campaigns").glob(
            "*/data/prospective_exact_simulator_call_ledger.csv"
        )
    ):
        df = pd.read_csv(path)
        if df.empty:
            continue
        meta = df.iloc[0]
        endpoint8 = endpoint_from_ledger(df, horizon)
        endpoint16 = endpoint_from_ledger(df, 16)
        rows.append(
            {
                "source": "new_hpc",
                "experiment_group": "physiology_robustness",
                "world_id": meta["world_id"],
                "campaign_seed": int(str(meta["campaign_id"]).rsplit("_seed_", 1)[1]),
                "campaign_id": meta["campaign_id"],
                "method_id": "conventional_dbtl",
                "training_cultures_n": np.nan,
                "subsample_seed": np.nan,
                "physical_culture_budget": horizon,
                "best_final_product": float(endpoint8["best_final_product"]),
                "best_productivity": float(endpoint8["best_productivity"]),
                "best_product_AUC": float(endpoint8["best_product_AUC"]),
                "best_candidate_id": "",
                "n_exact_verified": horizon,
                "raw16_best_final_product": float(endpoint16["best_final_product"]),
                "raw16_best_productivity": float(endpoint16["best_productivity"]),
                "raw16_best_product_AUC": float(endpoint16["best_product_AUC"]),
            }
        )
    return pd.DataFrame(rows)


def build_data_sufficiency(hybrid_endpoints: pd.DataFrame, v2_endpoints: pd.DataFrame) -> pd.DataFrame:
    hybrid_new = hybrid_endpoints[
        (hybrid_endpoints["experiment_group"] == "data_sufficiency")
        & (hybrid_endpoints["world_id"].isin(DATA_SUFF_WORLDS))
    ].copy()
    hybrid_v2 = v2_endpoints[v2_endpoints["method_id"] == "pretrained_hybrid_digital_twin"].copy()
    hybrid = pd.concat([hybrid_v2, hybrid_new], ignore_index=True)
    conv = v2_endpoints[v2_endpoints["method_id"] == "conventional_dbtl"].copy()
    comparisons = []
    for _, h in hybrid.iterrows():
        c = conv[(conv["world_id"] == h["world_id"]) & (conv["campaign_seed"] == h["campaign_seed"])]
        if c.empty:
            raise ValueError(f"Missing conventional comparator for {h['world_id']} {h['campaign_seed']}")
        c = c.iloc[0]
        comparisons.append(
            {
                "comparison_group": "data_sufficiency_common8",
                "world_id": h["world_id"],
                "campaign_seed": int(h["campaign_seed"]),
                "training_cultures_n": int(h["training_cultures_n"]),
                "subsample_seed": int(h["subsample_seed"]),
                "hybrid_best_final_product": h["best_final_product"],
                "conventional_best_final_product": c["best_final_product"],
                "delta_final_product": h["best_final_product"] - c["best_final_product"],
                "hybrid_best_productivity": h["best_productivity"],
                "conventional_best_productivity": c["best_productivity"],
                "delta_productivity": h["best_productivity"] - c["best_productivity"],
                "hybrid_best_product_AUC": h["best_product_AUC"],
                "conventional_best_product_AUC": c["best_product_AUC"],
                "delta_product_AUC": h["best_product_AUC"] - c["best_product_AUC"],
                "hybrid_source": h["source"],
                "conventional_source": c["source"],
                "physical_culture_budget_each": 8,
            }
        )
    return pd.DataFrame(comparisons)


def build_robustness(
    hybrid_endpoints: pd.DataFrame, v2_endpoints: pd.DataFrame, new_conventional: pd.DataFrame
) -> pd.DataFrame:
    hybrid_new = hybrid_endpoints[
        (hybrid_endpoints["experiment_group"] == "physiology_robustness")
        & (hybrid_endpoints["training_cultures_n"] == 77)
    ].copy()
    hybrid_v2 = v2_endpoints[
        (v2_endpoints["method_id"] == "pretrained_hybrid_digital_twin")
        & (v2_endpoints["world_id"].isin(("baseline_world", "strong_oxidative_burden_world")))
    ].copy()
    hybrid_v2["experiment_group"] = "physiology_robustness"
    hybrid = pd.concat([hybrid_v2, hybrid_new], ignore_index=True)
    conv_v2 = v2_endpoints[
        (v2_endpoints["method_id"] == "conventional_dbtl")
        & (v2_endpoints["world_id"].isin(("baseline_world", "strong_oxidative_burden_world")))
    ].copy()
    conv = pd.concat([conv_v2, new_conventional], ignore_index=True)
    comparisons = []
    for _, h in hybrid.iterrows():
        c = conv[(conv["world_id"] == h["world_id"]) & (conv["campaign_seed"] == h["campaign_seed"])]
        if c.empty:
            raise ValueError(f"Missing conventional comparator for {h['world_id']} {h['campaign_seed']}")
        c = c.iloc[0]
        comparisons.append(
            {
                "comparison_group": "physiology_robustness_common8",
                "world_id": h["world_id"],
                "campaign_seed": int(h["campaign_seed"]),
                "training_cultures_n": int(h["training_cultures_n"]),
                "subsample_seed": int(h["subsample_seed"]),
                "hybrid_best_final_product": h["best_final_product"],
                "conventional_best_final_product": c["best_final_product"],
                "delta_final_product": h["best_final_product"] - c["best_final_product"],
                "hybrid_best_productivity": h["best_productivity"],
                "conventional_best_productivity": c["best_productivity"],
                "delta_productivity": h["best_productivity"] - c["best_productivity"],
                "hybrid_best_product_AUC": h["best_product_AUC"],
                "conventional_best_product_AUC": c["best_product_AUC"],
                "delta_product_AUC": h["best_product_AUC"] - c["best_product_AUC"],
                "hybrid_source": h["source"],
                "conventional_source": c["source"],
                "physical_culture_budget_each": 8,
                "conventional_raw16_best_final_product": c.get("raw16_best_final_product", np.nan),
            }
        )
    return pd.DataFrame(comparisons)


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        deltas = group["delta_final_product"].to_numpy(float)
        ci_low, ci_high = bootstrap_ci(deltas)
        rec.update(
            {
                "n_paired_runs": len(group),
                "mean_hybrid_best_final_product": group["hybrid_best_final_product"].mean(),
                "mean_conventional_best_final_product": group["conventional_best_final_product"].mean(),
                "mean_delta_final_product": deltas.mean(),
                "median_delta_final_product": np.median(deltas),
                "delta_final_product_ci95_low": ci_low,
                "delta_final_product_ci95_high": ci_high,
                "win_rate": (deltas > 0).mean(),
                "mean_delta_productivity": group["delta_productivity"].mean(),
                "mean_delta_product_AUC": group["delta_product_AUC"].mean(),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_world_seed_averaged_data_sufficiency(comparisons: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        comparisons.groupby(["training_cultures_n", "world_id", "campaign_seed"], as_index=False)
        .agg(
            hybrid_best_final_product=("hybrid_best_final_product", "mean"),
            conventional_best_final_product=("conventional_best_final_product", "first"),
            delta_final_product=("delta_final_product", "mean"),
            delta_productivity=("delta_productivity", "mean"),
            delta_product_AUC=("delta_product_AUC", "mean"),
        )
    )
    return summarize_group(averaged, ["training_cultures_n"])


def summarize_predictive(root: Path) -> pd.DataFrame:
    path = root / "data/final_data_sufficiency_predictive_metrics.csv"
    df = pd.read_csv(path)
    product = df[df["channel"] == "product"].copy()
    if product.empty:
        product = df[df["channel"].astype(str).str.lower().isin(["b_total", "beta_carotene", "product"])]
    summary = (
        product.groupby(["training_cultures_n", "split"], as_index=False)
        .agg(
            mean_normalized_rmse=("normalized_rmse", "mean"),
            sd_normalized_rmse=("normalized_rmse", "std"),
            n=("normalized_rmse", "size"),
        )
        .sort_values(["training_cultures_n", "split"])
    )
    train = summary[summary["split"] == "train"][["training_cultures_n", "mean_normalized_rmse"]].rename(
        columns={"mean_normalized_rmse": "train_product_rmse"}
    )
    held = summary[summary["split"] == "validation"][["training_cultures_n", "mean_normalized_rmse"]].rename(
        columns={"mean_normalized_rmse": "validation_product_rmse"}
    )
    extra = summary[summary["split"] == "extrapolation"][["training_cultures_n", "mean_normalized_rmse"]].rename(
        columns={"mean_normalized_rmse": "extrapolation_product_rmse"}
    )
    wide = train.merge(held, on="training_cultures_n", how="outer").merge(extra, on="training_cultures_n", how="outer")
    wide["validation_generalization_gap"] = wide["validation_product_rmse"] - wide["train_product_rmse"]
    wide["extrapolation_generalization_gap"] = wide["extrapolation_product_rmse"] - wide["train_product_rmse"]
    return wide.sort_values("training_cultures_n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / "data/final_validation_prospective"
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid_new = load_new_hybrid_exact(root, args.horizon)
    v2 = load_v2_common8(root, args.horizon)
    conv_new = load_new_conventional(root, args.horizon)
    endpoints = pd.concat([v2, hybrid_new, conv_new], ignore_index=True, sort=False)
    endpoints.to_csv(out_dir / "final_validation_exact_endpoints_common8.csv", index=False)

    data_suff = build_data_sufficiency(hybrid_new, v2)
    robustness = build_robustness(hybrid_new, v2, conv_new)
    data_suff.to_csv(out_dir / "final_validation_data_sufficiency_common8_comparisons.csv", index=False)
    robustness.to_csv(out_dir / "final_validation_physiology_robustness_common8_comparisons.csv", index=False)

    summarize_group(data_suff, ["training_cultures_n"]).to_csv(
        out_dir / "final_validation_data_sufficiency_summary_by_N.csv", index=False
    )
    summarize_world_seed_averaged_data_sufficiency(data_suff).to_csv(
        out_dir / "final_validation_data_sufficiency_summary_by_N_worldseed_averaged.csv", index=False
    )
    summarize_group(robustness, ["world_id"]).to_csv(
        out_dir / "final_validation_physiology_robustness_summary_by_world.csv", index=False
    )
    summarize_group(robustness, ["comparison_group"]).to_csv(
        out_dir / "final_validation_physiology_robustness_summary_overall.csv", index=False
    )
    summarize_predictive(root).to_csv(out_dir / "final_validation_predictive_overfitting_summary.csv", index=False)

    print(f"Wrote aggregated outputs to {out_dir}")
    print("Data sufficiency summary:")
    print(pd.read_csv(out_dir / "final_validation_data_sufficiency_summary_by_N.csv").to_string(index=False))
    print("\nPhysiology robustness summary:")
    print(pd.read_csv(out_dir / "final_validation_physiology_robustness_summary_by_world.csv").to_string(index=False))


if __name__ == "__main__":
    main()
