#!/usr/bin/env python3
"""Stage 1 analysis: merge Path A (surrogate) vs Path B (real Yeast9) scores
for the surrogate_diagnostic candidate pool and compute ranking-degradation
metrics, stratified by environment, edit count/target, support distance, and
clipping.

Path A score  = candidate's ``path_A_virtual_score`` column
                 (data/surrogate_diagnostic_pool_path_A.csv).
Path B score  = ``final_product_pred_controls_via_real_gsm`` from each
                 per-candidate summary CSV produced on Vanda by
                 run_repaired_controls_real_gsm_replay_worker.py
                 (hpc_bundles/surrogate_diagnostic/results/*_summary.csv).

Both are the same quantity (final beta-carotene product B[-1]) computed
through two different metabolic-flux paths, holding c(t)+edits fixed -- so
any ranking divergence between them isolates surrogate-induced ranking
error, not physiology-model or edit-mapping error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS_DIR = ROOT / "hpc_bundles" / "surrogate_diagnostic" / "results"
OUT_CSV = DATA / "surrogate_diagnostic_merged.csv"
OUT_REPORT = DATA / "surrogate_diagnostic_stage1_report.md"


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    x = np.asarray(x)
    y = np.asarray(y)
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        sign = np.sign(dx) * np.sign(dy)
        concordant += int(np.sum(sign > 0))
        discordant += int(np.sum(sign < 0))
    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def load_path_b(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_summary.csv"))
    if not files:
        raise SystemExit(f"No Path B summary CSVs found in {results_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def topk_overlap(a_rank_ids: list[str], b_rank_ids: list[str], k: int) -> float:
    a_top = set(a_rank_ids[:k])
    b_top = set(b_rank_ids[:k])
    return len(a_top & b_top) / k


def topk_regret(df: pd.DataFrame, k: int) -> float:
    """Best true (Path B) score achievable in top-k, vs best true score if
    selection had used Path B ranking directly -- i.e. how much true product
    is left on the table by selecting according to Path A instead of Path B."""
    by_b = df.sort_values("path_B_score", ascending=False)
    best_possible = by_b["path_B_score"].iloc[:k].max()
    by_a = df.sort_values("path_A_virtual_score", ascending=False)
    achieved = by_a["path_B_score"].iloc[:k].max()
    return float(best_possible - achieved)


def rank_agreement_at_tail(df: pd.DataFrame, frac: float) -> dict:
    n = len(df)
    k = max(1, int(round(n * frac)))
    a_ids = df.sort_values("path_A_virtual_score", ascending=False)["candidate_id"].tolist()
    b_ids = df.sort_values("path_B_score", ascending=False)["candidate_id"].tolist()
    overlap = len(set(a_ids[:k]) & set(b_ids[:k])) / k
    return {"frac": frac, "k": k, "overlap": overlap}


def stratified_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, observed=True):
        if len(g) < 4:
            continue
        rho = spearman_rho(g["path_A_virtual_score"].to_numpy(), g["path_B_score"].to_numpy())
        tau = kendall_tau(g["path_A_virtual_score"].to_numpy(), g["path_B_score"].to_numpy())
        rmse = float(np.sqrt(np.mean((g["path_A_virtual_score"] - g["path_B_score"]) ** 2)))
        rows.append({group_col: key, "n": len(g), "spearman": rho, "kendall_tau": tau, "rmse": rmse})
    return pd.DataFrame(rows)


def main() -> None:
    path_a = pd.read_csv(DATA / "surrogate_diagnostic_pool_path_A.csv")
    path_b = load_path_b(RESULTS_DIR)
    path_b = path_b.rename(columns={"final_product_pred_controls_via_real_gsm": "path_B_score",
                                     "final_biomass_pred_controls_via_real_gsm": "path_B_biomass"})

    df = path_a.merge(
        path_b[["candidate_id", "path_B_score", "path_B_biomass", "n_infeasible_intervals", "n_intervals"]],
        on="candidate_id", how="inner",
    )
    df.to_csv(OUT_CSV, index=False)

    n_pool = len(path_a)
    n_matched = len(df)

    overall_rho = spearman_rho(df["path_A_virtual_score"].to_numpy(), df["path_B_score"].to_numpy())
    overall_tau = kendall_tau(df["path_A_virtual_score"].to_numpy(), df["path_B_score"].to_numpy())
    overall_rmse = float(np.sqrt(np.mean((df["path_A_virtual_score"] - df["path_B_score"]) ** 2)))

    a_ids = df.sort_values("path_A_virtual_score", ascending=False)["candidate_id"].tolist()
    b_ids = df.sort_values("path_B_score", ascending=False)["candidate_id"].tolist()

    topk = {k: topk_overlap(a_ids, b_ids, k) for k in (8, 25, 50) if k <= n_matched}
    regret = {k: topk_regret(df, k) for k in (8, 25, 50) if k <= n_matched}
    tail = [rank_agreement_at_tail(df, f) for f in (0.01, 0.05, 0.10) if int(round(n_matched * f)) >= 1]

    strat_edits = stratified_metrics(df, "n_edits")
    df["clipped_bucket"] = pd.cut(df["n_control_values_clipped"], bins=[-1, 0, 20, 80, 1000],
                                   labels=["0", "1-20", "21-80", "81+"])
    strat_clip = stratified_metrics(df, "clipped_bucket")

    def z_bucket(z):
        if z < 4:
            return "in_support(<4std)"
        if z < 6:
            return "mild_extrap(4-6std)"
        if z < 8:
            return "moderate_extrap(6-8std)"
        return "severe_extrap(8std+)"
    df["support_bucket"] = df["max_abs_z_edited"].apply(z_bucket)
    strat_support = stratified_metrics(df, "support_bucket")

    def mean_z_bucket(z):
        if z < 3:
            return "mean_in_support(<3std)"
        if z < 4:
            return "mean_mild(3-4std)"
        if z < 5:
            return "mean_moderate(4-5std)"
        return "mean_severe(5std+)"
    df["mean_support_bucket"] = df["mean_abs_z_edited"].apply(mean_z_bucket)
    strat_mean_support = stratified_metrics(df, "mean_support_bucket")

    df["temp_bucket"] = pd.cut(df["temperature"], bins=4)
    strat_env = stratified_metrics(df, "temp_bucket")

    infeasible_frac = (df["n_infeasible_intervals"] > 0).mean()

    lines = []
    lines.append("# Stage 1 Surrogate Ranking Diagnostic -- Path A vs Path B\n")
    lines.append(f"Pool: {n_pool} candidates generated; {n_matched} matched (Path A + Path B present).\n")
    lines.append(f"Path B LP-infeasible-interval fraction (candidates w/ >=1 infeasible interval): {infeasible_frac:.3f}\n")
    lines.append("## Overall ranking agreement\n")
    lines.append(f"- Spearman rho: {overall_rho:.4f}")
    lines.append(f"- Kendall tau: {overall_tau:.4f}")
    lines.append(f"- RMSE (final product, Path A vs Path B): {overall_rmse:.4f}")
    lines.append("")
    lines.append("## Top-k overlap (Path A top-k vs Path B top-k)\n")
    for k, v in topk.items():
        lines.append(f"- top-{k}: {v:.3f}")
    lines.append("")
    lines.append("## Top-k regret (true product left on table selecting via Path A vs Path B)\n")
    for k, v in regret.items():
        lines.append(f"- top-{k} regret: {v:.4f}")
    lines.append("")
    lines.append("## Tail rank agreement\n")
    for t in tail:
        lines.append(f"- top {t['frac']*100:.0f}% (k={t['k']}): overlap={t['overlap']:.3f}")
    lines.append("")
    lines.append("## Stratified by edit count\n")
    lines.append(strat_edits.to_string(index=False))
    lines.append("")
    lines.append("## Stratified by clipping severity (n_control_values_clipped)\n")
    lines.append(strat_clip.to_string(index=False))
    lines.append("")
    lines.append("## Stratified by support distance (max_abs_z_edited -- note: saturated, see below)\n")
    lines.append(strat_support.to_string(index=False))
    lines.append("")
    lines.append("## Stratified by support distance (mean_abs_z_edited, unsaturated)\n")
    lines.append(strat_mean_support.to_string(index=False))
    lines.append("")
    lines.append("## Stratified by temperature quartile\n")
    lines.append(strat_env.to_string(index=False))
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote merged data to {OUT_CSV}")
    print(f"Wrote report to {OUT_REPORT}")


if __name__ == "__main__":
    main()
