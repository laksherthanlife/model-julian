#!/usr/bin/env python3
"""Summarize saved dynamic-capacity generator artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    required = [
        "gem_dynamic_capacity_trajectories.csv",
        "gem_dynamic_capacity_fluxes.csv",
        "gem_dynamic_capacity_constraints.csv",
        "gem_dynamic_capacity_audit.csv",
        "gem_dynamic_capacity_solver_accounting.csv",
        "gem_dynamic_capacity_pca.csv",
        "gem_dynamic_capacity_shape_metrics.csv",
        "gem_dynamic_capacity_acceptance.csv",
        "gem_dynamic_capacity_control_comparison.csv",
    ]
    missing = [name for name in required if not (DATA / name).exists()]
    if missing:
        raise SystemExit(f"Missing dynamic-capacity artifacts: {', '.join(missing)}")
    traj = pd.read_csv(DATA / "gem_dynamic_capacity_trajectories.csv")
    flux = pd.read_csv(DATA / "gem_dynamic_capacity_fluxes.csv")
    audit = pd.read_csv(DATA / "gem_dynamic_capacity_audit.csv")
    solver = pd.read_csv(DATA / "gem_dynamic_capacity_solver_accounting.csv")
    pca = pd.read_csv(DATA / "gem_dynamic_capacity_pca.csv")
    shape = pd.read_csv(DATA / "gem_dynamic_capacity_shape_metrics.csv")
    accept = pd.read_csv(DATA / "gem_dynamic_capacity_acceptance.csv")
    print("Dynamic-capacity artifact audit")
    print(f"cultures={traj['culture_id'].nunique()} time_points={traj['time_index'].nunique()} intervals={flux.shape[0]}")
    print(f"LP solves={int(solver['n_actual_lp_solves'].sum())} surrogate_evaluations={int(solver['n_surrogate_evaluations'].sum())}")
    print(f"infeasible={int(solver['n_infeasible_solves'].sum())} unbounded={int(solver['n_unbounded_solves'].sum())}")
    print("capacity minima")
    print(audit[["minimum_capacity", "capacity_recovery_occurrence", "n_limiting_reaction_switches"]].agg(["min", "mean", "max"]).to_string())
    print("shape metrics")
    print(shape.to_string(index=False))
    print("PCA headline")
    print(pca[pca["matrix"].isin(["raw_accumulated_product", "amplitude_normalized_product", "beta_carotene_flux"])].head(15).to_string(index=False))
    print("Acceptance")
    print(accept[["criterion", "passed", "classification", "state_space_ml_unlocked"]].to_string(index=False))


if __name__ == "__main__":
    main()
