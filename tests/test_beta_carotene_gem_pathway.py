import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem


@pytest.fixture(scope="module")
def augmented_model():
    assets = gem.discover_assets()
    if not assets:
        pytest.skip("external yeast GEM asset is unavailable")
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, assets[0])
    augmented, manifest = gem.install_beta_carotene_pathway(base)
    return base, augmented, manifest


def test_pathway_installation_does_not_modify_raw_model(augmented_model):
    base, augmented, manifest = augmented_model
    assert gem.PRODUCT_RXN not in {rxn.id for rxn in base.reactions}
    assert gem.PRODUCT_RXN in {rxn.id for rxn in augmented.reactions}
    assert set(gem.PATHWAY_RXNS).issubset(set(manifest["reaction_id"]))
    assert "native" in set(manifest["status"])


def test_augmented_model_feasible_and_product_requires_pathway(augmented_model):
    _base, augmented, _manifest = augmented_model
    cfg = gem.GEMCultureConfig(n_time=3)
    model = augmented.copy()
    gem.apply_interval_constraints(model, cfg, np.zeros(4), "balanced")
    flux = gem.solve_staged(model, cfg, gem.state_gamma("balanced", cfg))
    assert flux["beta_carotene_flux"] > 0
    for rxn_id in ("BETA_PHYTOENE_SYNTHASE", "BETA_PHYTOENE_DESATURASE", "BETA_LYCOPENE_CYCLASE"):
        disabled = augmented.copy()
        gem.apply_interval_constraints(disabled, cfg, np.zeros(4), "balanced")
        disabled.reactions.get_by_id(rxn_id).upper_bound = 0.0
        blocked = gem.solve_staged(disabled, cfg, gem.state_gamma("balanced", cfg))
        assert blocked["beta_carotene_flux"] <= 1e-8


def test_growth_atp_and_oxygen_change_product_allocation(augmented_model):
    _base, augmented, _manifest = augmented_model
    cfg = gem.GEMCultureConfig(n_time=3)
    low_growth = augmented.copy()
    high_growth = augmented.copy()
    gem.apply_interval_constraints(low_growth, cfg, np.zeros(4), "balanced")
    gem.apply_interval_constraints(high_growth, cfg, np.zeros(4), "balanced")
    low = gem.solve_staged(low_growth, cfg, 0.50)
    high = gem.solve_staged(high_growth, cfg, 0.90)
    assert high["beta_carotene_flux"] < low["beta_carotene_flux"]

    low_atp = augmented.copy()
    high_atp = augmented.copy()
    gem.apply_interval_constraints(low_atp, cfg, np.zeros(4), "balanced")
    gem.apply_interval_constraints(high_atp, cfg, np.array([0.0, 1.0, 0.0, 0.0]), "balanced")
    assert gem.solve_staged(high_atp, cfg, gem.state_gamma("balanced", cfg))["beta_carotene_flux"] < gem.solve_staged(low_atp, cfg, gem.state_gamma("balanced", cfg))["beta_carotene_flux"]

    low_o2_cfg = gem.GEMCultureConfig(DO=10.0, n_time=3)
    high_o2_cfg = gem.GEMCultureConfig(DO=80.0, n_time=3)
    lo = augmented.copy()
    hi = augmented.copy()
    gem.apply_interval_constraints(lo, low_o2_cfg, np.zeros(4), "balanced")
    gem.apply_interval_constraints(hi, high_o2_cfg, np.zeros(4), "balanced")
    assert abs(gem.solve_staged(hi, high_o2_cfg, gem.state_gamma("balanced", high_o2_cfg))["oxygen_uptake"] - gem.solve_staged(lo, low_o2_cfg, gem.state_gamma("balanced", low_o2_cfg))["oxygen_uptake"]) > 1e-8
