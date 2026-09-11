#!/usr/bin/env python3
"""Step 3/4 analysis: exact-vs-virtual ranking metrics, and surrogate-vs-real-GSM
error isolation, for the repaired-model virtual scorer validation.

Reads back the exact Yeast9 results (Vanda, run_exact_validation_array.pbs)
and the surrogate-isolation replay results (Vanda,
run_surrogate_isolation_array.pbs), joins them against the local virtual
scores, and reports the metrics requested in Step 3/4 of the virtual-scorer
repair pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import design_benchmark_exact as dbe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HPC_DIR = ROOT / "hpc_bundles" / "virtual_ranking_validation"


def load_exact_results() -> pd.DataFrame:
    rows = []
    for path in sorted((HPC_DIR / "results").glob("*_summary.csv")):
        df = pd.read_csv(path)
        rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def load_surrogate_isolation_results() -> pd.DataFrame:
    rows = []
    for path in sorted((HPC_DIR / "surrogate_isolation_results").glob("*_summary.csv")):
        df = pd.read_csv(path)
        rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return dbe.spearman_without_scipy(pd.Series(a), pd.Series(b))


def main() -> None:
    manifest = pd.read_csv(HPC_DIR / "selected_candidates_manifest.csv")
    exact = load_exact_results()
    surr_iso = load_surrogate_isolation_results()

    merged = manifest.merge(exact[["candidate_id", "candidate_hash", "final_product", "final_biomass", "feasible", "severe_growth_collapse", "n_infeasible_solves"]], on=["candidate_id", "candidate_hash"], how="left")
    surr_cols = ["candidate_id", "candidate_hash", "final_product_pred_controls_via_real_gsm", "final_biomass_pred_controls_via_real_gsm", "n_infeasible_intervals"]
    if not surr_iso.empty:
        merged = merged.merge(surr_iso[surr_cols], on=["candidate_id", "candidate_hash"], how="left")
    else:
        for col in surr_cols:
            if col not in merged.columns:
                merged[col] = np.nan

    pool = pd.read_csv(DATA / "virtual_ranking_validation_pool_scores.csv")
    merged = merged.merge(pool[["candidate_id", "final_product_pred", "final_biomass_pred"]], on="candidate_id", how="left", suffixes=("", "_pool"))

    merged.to_csv(DATA / "virtual_ranking_validation_merged.csv", index=False)

    n = len(merged)
    valid = merged.dropna(subset=["final_product"])
    print(f"=== Step 3: exact-vs-virtual ranking ({len(valid)}/{n} exact results available) ===")

    rho = spearman(valid["virtual_score"].to_numpy(float), valid["final_product"].to_numpy(float))
    print(f"Spearman rank correlation (virtual_score vs exact final_product): {rho:.3f}")

    k = 8
    ranked_by_virtual = valid.sort_values("virtual_score", ascending=False)
    ranked_by_exact = valid.sort_values("final_product", ascending=False)
    virtual_topk_ids = set(ranked_by_virtual.head(k)["candidate_id"])
    exact_topk_ids = set(ranked_by_exact.head(k)["candidate_id"])
    enrichment = len(virtual_topk_ids & exact_topk_ids) / k
    print(f"Top-{k} enrichment (fraction of virtual top-{k} that are also exact top-{k}): {enrichment:.3f}")

    best_exact_overall = float(valid["final_product"].max())
    best_exact_in_virtual_topk = float(ranked_by_virtual.head(k)["final_product"].max())
    regret = best_exact_overall - best_exact_in_virtual_topk
    print(f"Top-{k} exact regret (best possible - best found via virtual top-{k}): {regret:.4f}  (best_overall={best_exact_overall:.4f}, best_in_virtual_top{k}={best_exact_in_virtual_topk:.4f})")

    print(f"Mean exact final_product, virtual top-{k}: {ranked_by_virtual.head(k)['final_product'].mean():.4f}")
    print(f"Mean exact final_product, all {len(valid)} candidates: {valid['final_product'].mean():.4f}")
    print(f"Mean exact final_product, virtual bottom-{k}: {ranked_by_virtual.tail(k)['final_product'].mean():.4f}")

    print("\nBy stratum (mean exact final_product):")
    print(valid.groupby("stratum")["final_product"].agg(["mean", "min", "max", "count"]).to_string())

    print("\n=== Conventional prediction error (secondary) ===")
    rmse_product = float(np.sqrt(np.mean((valid["virtual_score"] - valid["final_product"]) ** 2)))
    print(f"RMSE(virtual_score, exact final_product): {rmse_product:.4f}  (std of exact final_product: {valid['final_product'].std():.4f})")

    print("\n=== Step 4: surrogate vs real-GSM error isolation ===")
    valid4 = merged.dropna(subset=["final_product", "final_product_pred_controls_via_real_gsm"])
    if len(valid4):
        surrogate_err = float(np.sqrt(np.mean((valid4["final_product_pred"] - valid4["final_product_pred_controls_via_real_gsm"]) ** 2)))
        physiology_err = float(np.sqrt(np.mean((valid4["final_product_pred_controls_via_real_gsm"] - valid4["final_product"]) ** 2)))
        total_err = float(np.sqrt(np.mean((valid4["final_product_pred"] - valid4["final_product"]) ** 2)))
        print(f"n={len(valid4)}")
        print(f"RMSE(surrogate path final_product, real-GSM-replayed-same-controls final_product) -- surrogate approximation error alone: {surrogate_err:.4f}")
        print(f"RMSE(real-GSM-replayed-same-controls, true independent exact rollout) -- physiology+edit-mapping error (vs the true generator's own dynamics): {physiology_err:.4f}")
        print(f"RMSE(surrogate path, true independent exact rollout) -- total end-to-end error: {total_err:.4f}")
        print(f"Mean n_infeasible_intervals in real-GSM replay: {valid4['n_infeasible_intervals'].mean():.2f}/48")
    else:
        print("No matched rows for Step 4 comparison yet.")


if __name__ == "__main__":
    main()
