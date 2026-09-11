import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem
import run_gem_temporal_diversity as temporal


@pytest.fixture(scope="module")
def augmented_asset():
    assets = gem.discover_assets()
    if not assets:
        pytest.skip("external yeast GEM asset is unavailable")
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, assets[0])
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    return assets[0], augmented


def test_allocation_protocols_are_real_lp_and_change_product_policy(augmented_asset):
    _path, augmented = augmented_asset
    rows = []
    for protocol in ("A", "B", "C"):
        cfg = gem.GEMCultureConfig(n_time=3, allocation_protocol=protocol)
        model = augmented.copy()
        gem.apply_interval_constraints(model, cfg, np.zeros(4), "balanced")
        rows.append(gem.solve_staged(model, cfg, gem.state_gamma("balanced", cfg), "balanced"))

    assert {row["allocation_protocol"] for row in rows} == {"A", "B", "C"}
    assert all(row["n_actual_lp_solves"] == 3 for row in rows)
    assert all(row["n_optimal_solves"] == 3 for row in rows)
    assert all(row["n_infeasible_solves"] == 0 for row in rows)
    assert rows[0]["beta_carotene_flux"] > rows[2]["beta_carotene_flux"] > rows[1]["beta_carotene_flux"]


def test_flux_driven_burden_terms_have_expected_directionality():
    cfg = gem.GEMCultureConfig()
    burdens = np.array([0.2, 0.2, 0.2, 0.0])
    base_flux = {
        "oxygen_uptake": -2.0,
        "beta_carotene_flux": 0.10,
        "atp_maintenance_flux": 0.7,
        "biomass_flux": 0.2,
        "ggpp_flux": 0.20,
    }
    base = gem.burden_terms(cfg, burdens, base_flux, "productive")
    respiratory = gem.burden_terms(cfg, burdens, {**base_flux, "oxygen_uptake": -4.0}, "productive")
    atp_surplus = gem.burden_terms(cfg, burdens, {**base_flux, "oxygen_uptake": -4.0, "atp_maintenance_flux": 0.2}, "productive")
    ggpp_loaded = gem.burden_terms(cfg, burdens, {**base_flux, "ggpp_flux": 0.35}, "productive")

    assert respiratory["oxidative_burden_after"] > base["oxidative_burden_after"]
    assert atp_surplus["ATP_pressure_after"] < base["ATP_pressure_after"]
    assert ggpp_loaded["bottleneck_after"] > base["bottleneck_after"]
    assert base["ER_after"] == pytest.approx(gem.burden_terms(cfg, burdens, base_flux, "productive")["ER_after"])


def test_growth_quantity_audit_reconciles_default_medium_and_interval_flux(augmented_asset):
    path, augmented = augmented_asset
    cfg = gem.GEMCultureConfig(n_time=3)
    traj, fluxes, constraints, _summary = gem.run_dynamic_culture(augmented, path, cfg, culture_id="audit")
    for df in (fluxes, constraints):
        df["environment_id"] = "central"
        df["allocation_protocol"] = "A"

    audit = temporal.growth_quantity_audit(augmented, path, fluxes, constraints)
    assert {"raw_model_default_medium_max_growth", "interval_pfba_biomass_flux"}.issubset(set(audit["stage"]))
    raw = audit[audit["stage"].eq("raw_model_default_medium_max_growth")].iloc[0]
    interval = audit[audit["stage"].eq("interval_pfba_biomass_flux")].iloc[0]
    assert raw["glucose_bound"] == pytest.approx(-1.0)
    assert interval["glucose_bound"] == pytest.approx(-cfg.glucose_uptake)
    assert interval["flux_value"] > raw["flux_value"]
    assert int(traj["n_surrogate_evaluations"].max()) == 0
