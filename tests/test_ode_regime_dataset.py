import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import generate_ode_regime_dataset as gen
import ode_regime_core as ode


def test_manifest_uses_continuous_splits_and_heldout_holes():
    manifest = ode.make_environment_manifest(ode.ODEConfig(), pilot=True)
    assert {"train", "validation", "random_test", "heldout_hole", "edge_extrapolation", "atlas_grid"}.issubset(set(manifest["split"]))
    train_units = manifest[manifest["split"].eq("train")][["unit_temperature", "unit_pH", "unit_DO"]].to_numpy(float)
    hole_units = manifest[manifest["split"].eq("heldout_hole")][["unit_temperature", "unit_pH", "unit_DO"]].to_numpy(float)
    assert not bool(ode.in_heldout_holes(train_units).any())
    assert bool(ode.in_heldout_holes(hole_units).all())
    assert manifest[manifest["split"].eq("train")][["temperature", "pH", "DO"]].nunique().min() > 20


def test_pilot_dataset_safeguards_are_constructible():
    cfg = ode.ODEConfig(n_time=9, pilot_train=8, pilot_validation=4, pilot_test=4)
    manifest = ode.make_environment_manifest(cfg, pilot=True)
    tiny = manifest.groupby("split", group_keys=False).head(2).reset_index(drop=True)
    traj, diag = ode.simulate_manifest(tiny, cfg)
    safeguards = gen.split_safeguards(tiny, traj, diag)
    required = {"deterministic_fixed_environment_contract", "finite_nonnegative_states", "no_solver_failures"}
    assert required.issubset(set(safeguards["safeguard"]))
    assert safeguards[safeguards["safeguard"].isin(required)]["passed"].all()
