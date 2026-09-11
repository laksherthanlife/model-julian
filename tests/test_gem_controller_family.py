"""Tests for the variable-dimension hidden controller family (scripts/gem_controller_family.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import gem_controller_family as gcf  # noqa: E402

pytest.importorskip("cobra")


@pytest.fixture(scope="module")
def augmented_model():
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    return augmented, source_path


def test_build_controller_sizes():
    for d in (3, 8, 14, gcf.MAX_D_TRUE):
        ctrl = gcf.build_controller(d)
        assert len(ctrl) == d - 3
        assert len(ctrl) == len(set(s.name for s in ctrl))  # no duplicate names


def test_build_controller_rejects_below_core_dimension():
    with pytest.raises(ValueError):
        gcf.build_controller(2)


def test_build_controller_rejects_above_library_capacity():
    with pytest.raises(ValueError):
        gcf.build_controller(gcf.MAX_D_TRUE + 1)


def test_every_state_has_a_declared_evidence_class():
    valid = {"mechanistic", "literature-shaped", "declared design", "arbitrary"}
    for spec in gcf.STATE_LIBRARY:
        assert spec.evidence_class in valid, spec.name


def test_d_true_3_matches_gem_backend_bit_for_bit(augmented_model):
    augmented, source_path = augmented_model
    cfg = gem.GEMCultureConfig(n_time=4)
    traj_ref, _flux_ref, _cons_ref, _summary_ref = gem.run_dynamic_culture(augmented.copy(), source_path, cfg, culture_id="ref")
    traj_gen, _flux_gen, _cons_gen, _summary_gen = gcf.run_dynamic_culture_generic(augmented.copy(), source_path, cfg, extra_states=[], culture_id="ref")
    for col in ("X", "B_total", "S_glc", "z_ox", "z_atp", "z_bottle", "z_er", "E_PSY", "E_DES", "E_CYC"):
        assert np.allclose(traj_ref[col].to_numpy(), traj_gen[col].to_numpy()), col


def test_generic_rollout_runs_with_extra_states(augmented_model):
    augmented, source_path = augmented_model
    cfg = gem.GEMCultureConfig(n_time=3)
    extra_states = gcf.build_controller(6)  # 3 core + 3 extra
    traj, flux, cons, summary = gcf.run_dynamic_culture_generic(augmented.copy(), source_path, cfg, extra_states=extra_states, culture_id="d6")
    for spec in extra_states:
        assert spec.name in traj.columns
        assert (traj[spec.name] >= 0.0).all()
        assert (traj[spec.name] <= 1.5).all()
    assert int(summary["n_actual_lp_solves"].iloc[0]) == 3 * (cfg.n_time - 1)


def test_sensitivity_matrix_shape_and_gate(augmented_model):
    augmented, _source_path = augmented_model
    cfg = gem.GEMCultureConfig()
    result = gcf.sensitivity_matrix(6, cfg, augmented)
    assert result["jacobian"].shape[1] == 6
    assert len(result["state_names"]) == 6
    assert 0.0 <= result["effective_rank"] <= 6.0 + 1e-6
    assert isinstance(result["gate_pass"], bool)


def test_er_equivalent_decoy_registers_near_zero_effect(augmented_model):
    """z_er is the existing confirmed-null decoy: perturbing it must not move
    any causal flux output at all through the generalized coupling path used
    by gem_controller_family (mirrors gem_backend.er_isolation_check)."""
    augmented, _source_path = augmented_model
    cfg = gem.GEMCultureConfig()
    causal_keys = ["maximum_growth", "preserved_growth", "beta_carotene_flux", "biomass_flux", "glucose_uptake", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux"]
    outputs = []
    for z_er in (0.0, 1.2):
        model = augmented.copy()
        core_burdens = np.array([0.3, 0.3, 0.3, z_er])
        gem.apply_interval_constraints(model, cfg, core_burdens, "productive", enzyme_capacities=None)
        outputs.append(gem.solve_staged(model, cfg, gem.state_gamma("productive", cfg), state="productive"))
    max_abs = max(abs(float(outputs[0][k]) - float(outputs[1][k])) for k in causal_keys)
    assert max_abs < 1e-9
