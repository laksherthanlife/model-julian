#!/usr/bin/env python3
"""Audit saved diagnostic GEM state-space validation outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    required = [
        "gem_state_space_environment_grid.csv",
        "gem_state_space_trajectories.csv",
        "gem_state_space_fluxes.csv",
        "gem_state_space_states.csv",
        "gem_state_space_reporters.csv",
        "gem_state_space_solver_accounting.csv",
        "gem_state_space_split_manifest.csv",
        "gem_state_space_model_predictions.csv",
        "gem_state_space_metrics.csv",
        "gem_state_space_seed_summary.csv",
        "gem_state_space_paired_comparisons.csv",
        "gem_state_space_sample_efficiency.csv",
        "gem_state_space_reporter_metrics.csv",
        "gem_state_space_training_log.csv",
        "gem_state_space_diagnostic_conclusion.csv",
    ]
    missing = [name for name in required if not (DATA / name).exists()]
    if missing:
        raise SystemExit(f"Missing state-space artifacts: {', '.join(missing)}")
    grid = pd.read_csv(DATA / "gem_state_space_environment_grid.csv")
    traj = pd.read_csv(DATA / "gem_state_space_trajectories.csv")
    solver = pd.read_csv(DATA / "gem_state_space_solver_accounting.csv")
    metrics = pd.read_csv(DATA / "gem_state_space_metrics.csv")
    paired = pd.read_csv(DATA / "gem_state_space_paired_comparisons.csv")
    conclusion = pd.read_csv(DATA / "gem_state_space_diagnostic_conclusion.csv")
    print("GEM state-space validation audit")
    print(f"environments={grid['environment_id'].nunique()} cultures={traj['culture_id'].nunique()} time_points={traj['time_index'].nunique()}")
    print("split counts")
    print(grid["split"].value_counts().to_string())
    print("solver accounting")
    print(solver[["n_actual_lp_solves", "n_surrogate_evaluations", "n_infeasible_solves", "n_unbounded_solves"]].sum().to_string())
    print("heldout-combination headline")
    print(metrics[metrics["split"].eq("heldout_combination")].groupby("model")["trajectory_normalized_rmse"].mean().sort_values().head(12).to_string())
    print("paired comparison")
    print(paired.to_string(index=False))
    print("conclusion")
    print(conclusion.to_string(index=False))


if __name__ == "__main__":
    main()
