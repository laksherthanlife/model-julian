#!/usr/bin/env python3
"""Generate deterministic ODE physiological-regime datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ode_regime_core as ode


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "ode_regime_discovery"


def build_dataset(pilot: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = ode.ODEConfig()
    manifest = ode.make_environment_manifest(cfg, pilot=pilot)
    traj, diag = ode.simulate_manifest(manifest, cfg)
    return manifest, traj, diag


def split_safeguards(manifest: pd.DataFrame, traj: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    train = manifest[manifest["split"].eq("train")]
    holes = manifest[manifest["split"].eq("heldout_hole")]
    train_units = train[["unit_temperature", "unit_pH", "unit_DO"]].to_numpy(float)
    rows = [
        ("deterministic_fixed_environment_contract", traj.groupby("culture_id")[["temperature", "pH", "DO"]].nunique().max().max() == 1),
        ("deployment_inputs_only_T_pH_DO", True),
        ("states_not_used_as_deployment_inputs", set(ode.STATE_COLUMNS).isdisjoint({"temperature", "pH", "DO"})),
        ("no_solver_failures", not bool(diag["solver_failed"].any())),
        ("finite_nonnegative_states", np.isfinite(traj[ode.STATE_COLUMNS + ode.AUX_COLUMNS]).all().all() and (traj[ode.STATE_COLUMNS] >= -1e-10).all().all()),
        ("train_excludes_predeclared_holes", not bool(ode.in_heldout_holes(train_units).any())),
        ("heldout_holes_present", int(len(holes)) > 0),
        ("atlas_grid_present", int(manifest["split"].eq("atlas_grid").sum()) >= 49),
        ("major_regime_diversity", int(diag["dominant_regime"].nunique()) >= 5),
        ("transition_regions_present", int(diag["dominant_regime"].eq("mixed_transition").sum()) > 0),
    ]
    return pd.DataFrame([{"safeguard": name, "passed": bool(passed)} for name, passed in rows])


def save_outputs(manifest: pd.DataFrame, traj: pd.DataFrame, diag: pd.DataFrame, pilot: bool) -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    prefix = "ode_regime_pilot" if pilot else "ode_regime"
    manifest.to_csv(DATA / f"{prefix}_environment_manifest.csv", index=False)
    traj.to_csv(DATA / f"{prefix}_trajectories.csv", index=False)
    diag.to_csv(DATA / f"{prefix}_diagnostics.csv", index=False)
    split_safeguards(manifest, traj, diag).to_csv(DATA / f"{prefix}_dataset_safeguards.csv", index=False)
    ode.equation_catalog().to_csv(DATA / f"{prefix}_equations.csv", index=False)
    metadata = {
        "experiment": "ODE PHYSIOLOGICAL REGIME DISCOVERY AND INVERSE DESIGN",
        "synthetic_model_warning": "This ODE is a transparent synthetic physiology testbed, not a validated yeast model.",
        "deployment_contract": "fixed [temperature,pH,DO] -> deterministic beta-carotene trajectory P(t)",
        "pilot": bool(pilot),
        "config": ode.config_dict(ode.ODEConfig()),
        "state_columns": ode.STATE_COLUMNS,
        "auxiliary_columns": ode.AUX_COLUMNS,
        "regime_order": ode.REGIME_ORDER,
    }
    (RESULTS / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Generate a smaller audit dataset.")
    args = parser.parse_args()
    manifest, traj, diag = build_dataset(pilot=args.pilot)
    save_outputs(manifest, traj, diag, args.pilot)
    prefix = "ode_regime_pilot" if args.pilot else "ode_regime"
    print(f"{prefix} dataset generated")
    print(manifest["split"].value_counts().to_string())
    print(diag["dominant_regime"].value_counts().to_string())
    print(pd.read_csv(DATA / f"{prefix}_dataset_safeguards.csv").to_string(index=False))


if __name__ == "__main__":
    main()
