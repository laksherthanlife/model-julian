import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_observable_latent_interface_validation as obs


def test_column_audit_marks_controller_columns_as_lockbox():
    dataset = obs.load_dataset()
    audit = obs.write_column_audit(dataset)
    locked = audit[audit["classification"].eq("HIDDEN_LOCKBOX")]
    assert "PSY_effective_upper_bound" in set(locked["column"])
    assert "gamma_growth_fraction" in set(locked["column"])
    assert not locked["allowed_encoder_input"].any()
    assert not locked["allowed_observable_loss"].any()


def test_lockbox_guard_rejects_hidden_training_usage():
    with pytest.raises(AssertionError):
        obs.assert_no_lockbox_usage(["temperature"], ["B_total", "z_ox"], ["B_total"])
    with pytest.raises(AssertionError):
        obs.assert_no_lockbox_usage(["temperature", "gamma_growth_fraction"], ["B_total"], ["B_total"])


def test_split_integrity_and_train_only_stats():
    dataset = obs.load_dataset()
    obs.assert_split_integrity(dataset)
    arr = np.array([[0.0, 1.0], [2.0, 3.0], [200.0, 300.0]])
    train = np.array([True, True, False])
    mean, std = obs.train_stats_only(arr, train)
    assert np.allclose(mean, [1.0, 2.0])
    assert np.allclose(std, [1.0, 1.0])


def test_observable_encoder_excludes_controller_label_loss():
    dataset = obs.load_dataset()
    surrogate = obs.fit_frozen_surrogate(dataset, obs.METABOLIC_OBSERVABLES + obs.REPORTER_COLUMNS)
    ckpt = obs.train_encoder(dataset, surrogate, "latent_observable_with_reporters", seed=11, steps=5, lr=0.01)
    assert ckpt.training_loss_type == "observable_only_no_controller_label_loss"
    assert set(ckpt.observable_loss_columns).isdisjoint(set(obs.LOCKBOX_COLUMNS))
    assert ckpt.observable_loss_columns == obs.METABOLIC_OBSERVABLES + obs.REPORTER_COLUMNS


def test_privileged_upper_bound_is_explicitly_not_headline_observable_training():
    dataset = obs.load_dataset()
    surrogate = obs.fit_frozen_surrogate(dataset, obs.METABOLIC_OBSERVABLES + obs.REPORTER_COLUMNS)
    ckpt = obs.train_encoder(dataset, surrogate, "privileged_controller_upper_bound", seed=11, steps=5, lr=0.01)
    assert ckpt.training_loss_type == "privileged_controller_mse_oracle_upper_bound"
    assert ckpt.observable_loss_columns == []


def test_exact_replay_path_is_real_gsm_backend_not_surrogate_with_fake_solver():
    dataset = obs.load_dataset()
    controls = {col: dataset["interface"][col][:1].copy() for col in obs.INTERFACE_COLUMNS}
    for col in obs.INTERFACE_COLUMNS:
        full = np.zeros_like(dataset["interface"][col])
        full[:1] = controls[col]
        controls[col] = full

    def fake_solver(_control, _cfg, k):
        return {
            "beta_carotene_flux": 0.1 + 0.01 * k,
            "biomass_flux": 0.2,
            "n_growth_optimizations": 1,
            "n_product_optimizations": 1,
            "n_pfba_optimizations": 1,
            "n_actual_lp_solves": 3,
            "n_optimal_solves": 3,
            "n_infeasible_solves": 0,
            "n_unbounded_solves": 0,
            "n_surrogate_evaluations": 0,
        }

    rollout = obs.exact_real_gsm_replay_limited(
        controls,
        dataset,
        "latent_observable_with_reporters",
        indices=[0],
        max_intervals=2,
        solver_fn=fake_solver,
    )
    traj = rollout["trajectories"]
    solver = rollout["solver"].iloc[0]
    assert traj["metabolic_backend"].eq("yeast_gem_lp").all()
    assert (~traj["direct_product_head_used"]).all()
    assert solver["n_actual_lp_solves"] == 6
    assert solver["n_surrogate_evaluations"] == 0
