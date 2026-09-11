import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_inverse_gsm_calibration as inv


def test_diagnostic_culture_manifest_is_algorithmic_and_mixed():
    dataset = inv.load_dataset()
    manifest = inv.select_diagnostic_cultures(dataset)
    assert len(manifest) == 5
    assert manifest["culture_id"].is_unique
    assert manifest["selected_without_fit_outcome"].all()
    assert {"train", "heldout_combination", "extrapolation"}.intersection(set(manifest["split"]))
    assert "central_low_stress_train_condition" in set(manifest["selection_reason"])


def test_lockbox_guard_blocks_controller_objective_and_initialization():
    with pytest.raises(AssertionError):
        inv.assert_no_lockbox_usage(inv.ENV_COLUMNS, ["B_total", "gamma_growth_fraction"], [])
    with pytest.raises(AssertionError):
        inv.assert_no_lockbox_usage(inv.ENV_COLUMNS, ["B_total"], ["z_ox"])
    inv.assert_no_lockbox_usage(inv.ENV_COLUMNS, ["B_total", "X", "R_ox"], [])


def test_global_bounds_are_from_training_split_not_target_culture():
    dataset = inv.load_dataset()
    bounds = inv.global_controller_bounds(dataset)
    assert set(bounds["control"]) == set(inv.INTERFACE_COLUMNS)
    assert bounds["source"].eq("global_training_split_controller_range").all()
    assert (bounds["upper"] > bounds["lower"]).all()


def test_reporter_free_objective_contains_no_reporter_block_weight():
    dataset = inv.load_dataset()
    bounds = inv.global_controller_bounds(dataset)
    culture = inv.select_diagnostic_cultures(dataset).iloc[0]
    params = inv.mean_controller_params(dataset, bounds, n_intervals=2, n_knots=3)
    controls = inv.decode_params(params, bounds, n_intervals=2, n_knots=3)
    traj, _flux, _accounting = inv.fresh_replay(culture, controls, n_intervals=2, solver_fn=inv.fake_solver)
    reporters = inv.controls_to_reporters(controls, inv.capacity.selected_parameter_config())
    parts = inv.objective_parts(culture, traj, reporters, inv.target_tables(dataset), inv.fit_scales(dataset, 2), reporter_weight=0.0, n_intervals=2)
    assert parts["total_loss"] == parts["metabolic_block_loss"]


def test_reporter_enabled_objective_uses_reporter_measurements_not_hidden_states():
    dataset = inv.load_dataset()
    bounds = inv.global_controller_bounds(dataset)
    culture = inv.select_diagnostic_cultures(dataset).iloc[0]
    params = inv.mean_controller_params(dataset, bounds, n_intervals=2, n_knots=3)
    controls = inv.decode_params(params, bounds, n_intervals=2, n_knots=3)
    traj, _flux, _accounting = inv.fresh_replay(culture, controls, n_intervals=2, solver_fn=inv.fake_solver)
    reporters = inv.controls_to_reporters(controls, inv.capacity.selected_parameter_config())
    parts = inv.objective_parts(culture, traj, reporters, inv.target_tables(dataset), inv.fit_scales(dataset, 2), reporter_weight=0.3, n_intervals=2)
    assert parts["reporter_block_loss"] > 0
    assert parts["total_loss"] != parts["metabolic_block_loss"]
    assert not any(key.startswith("z_") for key in parts)


def test_fresh_replay_uses_real_backend_contract_with_fake_solver():
    dataset = inv.load_dataset()
    bounds = inv.global_controller_bounds(dataset)
    culture = inv.select_diagnostic_cultures(dataset).iloc[0]
    params = inv.mean_controller_params(dataset, bounds, n_intervals=2, n_knots=3)
    controls = inv.decode_params(params, bounds, n_intervals=2, n_knots=3)
    traj, flux, accounting = inv.fresh_replay(culture, controls, n_intervals=2, solver_fn=inv.fake_solver)
    assert len(traj) == 3
    assert len(flux) == 2
    assert accounting["n_actual_lp_solves"] == 6
    assert accounting["n_surrogate_evaluations"] == 0


def test_true_controls_are_posthoc_lockbox_shape():
    dataset = inv.load_dataset()
    culture = inv.select_diagnostic_cultures(dataset).iloc[0]
    controls = inv.true_controls_for_culture(dataset, culture, n_intervals=3)
    assert set(controls) == set(inv.INTERFACE_COLUMNS)
    assert all(v.shape == (3,) for v in controls.values())


def test_stage_a_parameterization_is_single_interval_direct_control():
    dataset = inv.load_dataset()
    bounds = inv.global_controller_bounds(dataset)
    audit = inv.parameterization_audit(bounds, "Stage A", n_intervals=1, n_knots=1)
    assert audit["total_free_parameters"].nunique() == 1
    assert int(audit["total_free_parameters"].iloc[0]) == len(inv.INTERFACE_COLUMNS)
    assert audit["parameterization"].eq("single_interval_direct_control").all()


def test_stage_a_fake_solver_writes_required_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "DATA", tmp_path / "data")
    monkeypatch.setattr(inv, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(inv, "RESULTS", tmp_path / "results")
    args = inv.parse_args([])
    args.stage_a = True
    args.fake_solver = True
    args.random_samples = 3
    args.max_evaluations = 3
    args.restart_seeds = [11]
    args.optimizers = ["differential_evolution", "powell_multistart", "annealing"]
    args.stage_a_reporter_weights = [0.0]
    args.landscape_points = 2
    out = inv.run_stage_a(args)
    assert not out["parameterization"].empty
    assert not out["reference"].empty
    assert len(out["random_baseline"]) == 3
    assert set(out["optimizer_summary"]["optimizer"]) == {"differential_evolution", "powell_multistart", "annealing"}
    assert (inv.DATA / "inverse_gsm_stageA_convergence.csv").exists()
    assert (inv.DATA / "inverse_gsm_stageA_decision_table.csv").exists()
    assert (inv.DATA / "inverse_gsm_stageA_controller_solutions.csv").exists()
    assert (inv.DATA / "inverse_gsm_stageA_loss_breakdown.csv").exists()
    assert (inv.DATA / "inverse_gsm_stageA_solver_accounting.csv").exists()
