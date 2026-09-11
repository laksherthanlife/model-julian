import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_fixed_environment_validation as fixed


def test_fixed_environment_fast_safeguards_pass():
    metrics, frontier, safeguards = fixed.run_all(fast=True, write=False)
    assert not metrics.empty
    assert not frontier.empty
    assert safeguards["passed"].all()
    assert {"1A", "1B", "1C", "2A", "2B", "2C"}.issubset(set(metrics["experiment"]))
    assert {"interpolation", "heldout_combination", "extrapolation"}.issubset(
        set(metrics["split"])
    )
    assert {
        "product_rmse",
        "product_r2",
        "final_product_error",
        "production_rate_rmse",
        "early_time_rmse",
        "late_time_rmse",
    }.issubset(metrics.columns)


def test_generated_trajectories_keep_environment_fixed():
    cfg = fixed.Config(dataset_seeds=(101,), model_seeds=(11,))
    traj, params, manifest = fixed.generate_dataset(101, cfg)
    env_counts = traj.groupby("culture_id")[["I", "O", "S"]].nunique().max(axis=1)
    assert (env_counts == 1).all()
    split_counts = traj.groupby("culture_id")["split"].nunique()
    assert (split_counts == 1).all()
    train_env = manifest[manifest["split"] == "train"][["I", "O", "S"]].to_numpy()
    assert not fixed.holdout_region(train_env).any()
    present = params[["V", "tau", "tau_on", "tau_off", "s_A", "s_B"]].to_numpy(float)
    present = present[np.isfinite(present)]
    assert (present > 0).all()


def test_black_box_inputs_are_fixed_environment_plus_time():
    cfg = fixed.Config(dataset_seeds=(101,), model_seeds=(11,))
    traj, params, _ = fixed.generate_dataset(101, cfg)
    rows, _, pred_rows = fixed.run_experiment1(101, 11, traj, params, cfg, steps=20)
    assert any(row["model"] == "direct_env_time_black_box" for row in rows)
    assert np.isfinite([row["product_rmse"] for row in rows]).all()
    assert pred_rows
    assert all(row["split"] == "heldout_combination" for row in pred_rows)
    assert not any("oracle" in row["model"] for row in pred_rows)
    assert {row["prediction_source"] for row in pred_rows} == {"saved_model_rollout"}
