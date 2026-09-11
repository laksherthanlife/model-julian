#!/usr/bin/env python3
"""Stage 3/4: aggregate the Stage 2 purpose-built exact-Yeast9 solves, fit
the new MLP surrogate on a train split, and report held-out flux accuracy.

Held-out split is candidate-aware for "traj" points (all 8 timesteps of a
given candidate go together, so held-out isn't leaking nearby-timestep
correlation of the same trajectory) and simple random for "edge_random" /
"edge_corner" points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_gsm_surrogate_mlp as mlp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS_DIR = ROOT / "hpc_bundles" / "stage2_points" / "results"
OUT_DATASET = DATA / "gsm_surrogate_stage2_solved.csv"
OUT_SURROGATE = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"
OUT_REPORT = DATA / "stage3_stage4_surrogate_report.md"
HELD_OUT_FRACTION = 0.15
SEED = 8001


def load_solved(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_solved.csv"))
    if not files:
        raise SystemExit(f"No solved shard CSVs found in {results_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def candidate_aware_split(df: pd.DataFrame, held_out_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_mask = np.ones(len(df), dtype=bool)

    traj = df[df["source"] == "traj"]
    cand_ids = traj["candidate_id"].dropna().unique()
    n_held_cands = int(round(len(cand_ids) * held_out_fraction))
    held_cands = set(rng.choice(cand_ids, size=n_held_cands, replace=False))
    train_mask &= ~(df["source"].eq("traj") & df["candidate_id"].isin(held_cands))

    non_traj_idx = df.index[df["source"] != "traj"].to_numpy()
    n_held_non_traj = int(round(len(non_traj_idx) * held_out_fraction))
    held_non_traj = rng.choice(non_traj_idx, size=n_held_non_traj, replace=False)
    train_mask[held_non_traj] = False

    return train_mask


def main() -> None:
    raw = load_solved(RESULTS_DIR)
    n_total = len(raw)
    ok = raw[raw["status"] == "ok"].copy()
    n_ok = len(ok)
    for col in sur.SURROGATE_OUTPUT_COLUMNS:
        ok = ok[np.isfinite(ok[col])]
    ok = ok.reset_index(drop=True)
    n_finite = len(ok)
    ok.to_csv(OUT_DATASET, index=False)

    train_mask = candidate_aware_split(ok, HELD_OUT_FRACTION, SEED)
    n_train = int(train_mask.sum())
    n_held = int((~train_mask).sum())

    surrogate = mlp.fit_surrogate_mlp(ok, train_mask, hidden_dim=32, epochs=2000, lr=0.01, seed=7001, verbose=True)
    OUT_SURROGATE.parent.mkdir(parents=True, exist_ok=True)
    surrogate.save(OUT_SURROGATE)

    metrics_new = mlp.evaluate_surrogate_mlp(surrogate, ok, train_mask)

    # Comparison: refit the OLD ridge surrogate on this same Stage 2 data/split
    # (not its original reference-strain-only data) so the accuracy comparison
    # isolates "new deployment-domain data" from "new architecture" -- and
    # separately evaluate the OLD surrogate (as actually deployed, trained on
    # its original reference-strain data) on this held-out set, to see how it
    # does on the deployment-domain points it never saw.
    old_refit = sur.fit_surrogate(ok, train_mask=train_mask, ridge_lambda=1.0)
    metrics_old_refit = sur.evaluate_surrogate(old_refit, ok, train_mask)

    lines = []
    lines.append("# Stage 3/4: New MLP Surrogate vs Old Ridge Surrogate\n")
    lines.append(f"Stage 2 raw rows: {n_total}; status==ok: {n_ok}; finite outputs: {n_finite}")
    lines.append(f"Train points: {n_train}; held-out points: {n_held} (candidate-aware split, held_out_fraction={HELD_OUT_FRACTION})\n")
    lines.append("## New MLP surrogate -- held-out normalized RMSE per channel\n")
    lines.append(metrics_new.to_string(index=False))
    lines.append("")
    lines.append("## Old ridge surrogate, REFIT on the same Stage 2 train split -- held-out normalized RMSE per channel\n")
    lines.append(metrics_old_refit.to_string(index=False))
    lines.append("")
    lines.append(f"New surrogate saved to {OUT_SURROGATE}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
