"""Regression test for the oxidative-beta coupling ablation switch (GENERATOR_VALIDITY.md Section 2.4/2.6).

``burden_terms`` is pure math over dicts/arrays -- no COBRApy/GEM required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402


def _flux(beta: float) -> dict[str, object]:
    return {
        "oxygen_uptake": -4.0,
        "beta_carotene_flux": beta,
        "atp_maintenance_flux": 0.8,
        "biomass_flux": 0.2,
        "ggpp_flux": 0.3,
    }


def test_default_coefficient_matches_historical_hardcoded_value():
    cfg = gem.GEMCultureConfig()
    assert cfg.oxidative_beta_coupling == 0.22


def test_ablation_zeroes_only_the_beta_contribution_to_oxidative_generation():
    burdens = np.array([0.3, 0.25, 0.2, 0.0])
    flux = _flux(beta=0.4)

    cfg_default = gem.GEMCultureConfig()
    cfg_ablated = gem.GEMCultureConfig(oxidative_beta_coupling=0.0)

    terms_default = gem.burden_terms(cfg_default, burdens, flux, "balanced")
    terms_ablated = gem.burden_terms(cfg_ablated, burdens, flux, "balanced")

    # The oxygen-stress contribution (independent of beta) must be identical.
    oxygen_stress_term = 0.16 * max(0.0, 4.0 / max(0.25 + 12.0 * 0.4, 1e-9))
    assert terms_default["oxidative_generation_term"] > terms_ablated["oxidative_generation_term"]
    assert np.isclose(terms_default["oxidative_generation_term"] - terms_ablated["oxidative_generation_term"], 0.22 * 0.4)

    # No other burden channel (ATP, bottleneck, ER) should change at all.
    for key in ("ATP_demand_term", "ATP_supply_term", "GGPP_loading_term", "ER_generation_term"):
        assert terms_default[key] == terms_ablated[key]


def test_ablation_with_zero_beta_flux_is_a_true_no_op():
    burdens = np.array([0.3, 0.25, 0.2, 0.0])
    flux = _flux(beta=0.0)
    cfg_default = gem.GEMCultureConfig()
    cfg_ablated = gem.GEMCultureConfig(oxidative_beta_coupling=0.0)
    terms_default = gem.burden_terms(cfg_default, burdens, flux, "balanced")
    terms_ablated = gem.burden_terms(cfg_ablated, burdens, flux, "balanced")
    assert terms_default["oxidative_generation_term"] == terms_ablated["oxidative_generation_term"]
