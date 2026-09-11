import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem
import run_gem_dynamic_capacity as dynamic_capacity


def test_dynamic_capacity_initialization_and_clipping():
    cfg = gem.GEMCultureConfig(temperature=33.0, pH=4.5, DO=80.0, capacity_mode="dynamic_congestion_feedback")
    caps = gem.initial_enzyme_capacities(cfg)
    assert caps.shape == (3,)
    assert np.all(caps >= cfg.enzyme_min_capacity)
    assert np.all(caps <= 1.0)
    assert not np.allclose(caps, np.ones(3))

    low = replace(cfg, enzyme_min_capacity=0.5, psy_initial_base=0.1, des_initial_base=0.1, cyc_initial_base=0.1)
    assert np.all(gem.initial_enzyme_capacities(low) >= 0.5)


def test_capacity_update_loss_recovery_and_congestion_feedback():
    cfg = gem.GEMCultureConfig(capacity_mode="dynamic_congestion_feedback")
    caps = np.array([0.8, 0.75, 0.7])
    flux = {"oxygen_uptake": -4.0, "beta_carotene_flux": 0.15, "atp_maintenance_flux": 0.8, "biomass_flux": 0.2, "ggpp_flux": 0.3}
    stress = np.array([0.8, 0.7, 0.6, 0.0])
    no_cong, terms0 = gem.update_enzyme_capacities(cfg, caps, stress, flux, "oxidative_stress", {"phytoene_congestion": 0.0, "lycopene_congestion": 0.0})
    cong, terms1 = gem.update_enzyme_capacities(cfg, caps, stress, flux, "oxidative_stress", {"phytoene_congestion": 0.1, "lycopene_congestion": 0.1})
    assert np.all(no_cong < caps)
    assert terms1["DES_damage_rate"] > terms0["DES_damage_rate"]
    assert np.all(cong <= no_cong)

    depleted = np.array([0.3, 0.25, 0.2])
    recovering, _terms = gem.update_enzyme_capacities(cfg, depleted, np.array([0.1, 0.1, 0.1, 0.0]), flux, "recovering", {})
    assert np.all(recovering > depleted)


def test_congestion_calculation_tracks_flux_mismatch_and_capacity_pressure():
    flux = {"phytoene_flux": 0.20, "desaturase_flux": 0.15, "cyclase_flux": 0.10}
    terms = gem.pathway_congestion_terms(flux)
    assert terms["phytoene_congestion_flux_mismatch"] == pytest.approx(0.05)
    assert terms["lycopene_congestion_flux_mismatch"] == pytest.approx(0.05)

    pressure = gem.pathway_congestion_terms(
        {"phytoene_flux": 0.12, "desaturase_flux": 0.12, "cyclase_flux": 0.12},
        {"PSY_effective_upper_bound": 0.30, "DES_effective_upper_bound": 0.20, "CYC_effective_upper_bound": 0.15},
    )
    assert pressure["phytoene_capacity_pressure"] > 0
    assert pressure["lycopene_capacity_pressure"] > 0


@pytest.fixture(scope="module")
def augmented_asset():
    assets = gem.discover_assets()
    if not assets:
        pytest.skip("external yeast GEM asset is unavailable")
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, assets[0])
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    return assets[0], augmented


def test_reaction_specific_upper_bounds_and_real_lp_response(augmented_asset):
    _path, augmented = augmented_asset
    cfg = gem.GEMCultureConfig(capacity_mode="constant_pathway_caps")
    caps = np.array([0.8, 0.6, 0.4])
    model = augmented.copy()
    constraints = gem.apply_interval_constraints(model, cfg, np.zeros(4), "balanced", caps)
    assert model.reactions.get_by_id(gem.PSY_RXN).upper_bound == pytest.approx(cfg.psy_base_capacity * caps[0])
    assert model.reactions.get_by_id(gem.DES_RXN).upper_bound == pytest.approx(cfg.des_base_capacity * caps[1])
    assert model.reactions.get_by_id(gem.CYC_RXN).upper_bound == pytest.approx(cfg.cyc_base_capacity * caps[2])
    flux = gem.solve_staged(model, cfg, gem.state_gamma("balanced", cfg), "balanced")
    assert 0 < flux["beta_carotene_flux"] <= constraints["CYC_effective_upper_bound"] + 1e-8
    assert flux["n_actual_lp_solves"] == 3


def test_dynamic_capacity_run_accounting_reproducible_and_no_surrogate(augmented_asset):
    path, augmented = augmented_asset
    cfg = gem.GEMCultureConfig(n_time=4, capacity_mode="dynamic_damage_recovery")
    t1, f1, c1, _s1 = gem.run_dynamic_culture(augmented, path, cfg, "cap_rep1")
    t2, f2, c2, _s2 = gem.run_dynamic_culture(augmented, path, cfg, "cap_rep2")
    assert int(t1["n_actual_lp_solves"].iloc[0]) == 9
    assert int(t1["n_surrogate_evaluations"].max()) == 0
    assert np.allclose(t1["B_total"], t2["B_total"])
    assert np.allclose(t1[["E_PSY", "E_DES", "E_CYC"]], t2[["E_PSY", "E_DES", "E_CYC"]])
    assert c1[["PSY_synthesis_term", "DES_damage_rate", "CYC_damage_rate"]].notna().all().all()
    assert f1[["PSY_flux", "DES_flux", "CYC_flux"]].notna().all().all()


def test_pca_shape_metrics_and_gate_do_not_unlock_without_ready_classification(augmented_asset):
    path, augmented = augmented_asset
    base = dynamic_capacity.selected_parameter_config()
    envs = [(30.0, 5.0, 40.0, "central"), (33.0, 4.5, 80.0, "stress")]
    traj, flux, cons, _summary, _runtime = dynamic_capacity.run_envs(augmented, path, base, envs, 4, "dynamic_congestion_feedback")
    audit = dynamic_capacity.culture_audit(traj, flux, cons)
    pca = dynamic_capacity.pca_table(traj, flux)
    shape = dynamic_capacity.shape_metrics(traj, flux, audit)
    ladder = pytest.importorskip("pandas").DataFrame(
        [
            {"capacity_mode": "constant_pathway_caps", "distinct_normalized_trajectory_clusters": 1, "normalized_shape_PC1": 0.999},
            {"capacity_mode": "dynamic_congestion_feedback", "distinct_normalized_trajectory_clusters": 1, "normalized_shape_PC1": 0.999},
        ]
    )
    accept = dynamic_capacity.acceptance(traj, flux, cons, audit, pca, shape, ladder)
    assert "classification" in accept
    if accept["classification"].iloc[0] != "gem_dynamic_capacity_ready_for_state_space":
        assert not bool(accept["state_space_ml_unlocked"].iloc[0])
