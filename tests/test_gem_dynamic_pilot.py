import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem


@pytest.fixture(scope="module")
def augmented_asset():
    assets = gem.discover_assets()
    if not assets:
        pytest.skip("external yeast GEM asset is unavailable")
    cobra = gem.require_cobra()
    base = gem.load_model(cobra, assets[0])
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    return assets[0], augmented


def test_dynamic_pilot_uses_real_lp_accounting_and_no_surrogate(augmented_asset):
    path, augmented = augmented_asset
    cfg = gem.GEMCultureConfig(n_time=4)
    traj, fluxes, constraints, _summary = gem.run_dynamic_culture(augmented, path, cfg, culture_id="test_short")
    assert traj["metabolic_backend"].eq("yeast_gem_lp").all()
    assert int(traj["n_surrogate_evaluations"].max()) == 0
    assert int(traj["n_actual_lp_solves"].iloc[0]) == 3 * (cfg.n_time - 1)
    assert (fluxes["n_optimal_solves"] == 3).all()
    assert (fluxes["n_infeasible_solves"] == 0).all()
    assert (fluxes["n_unbounded_solves"] == 0).all()
    assert np.isfinite(traj[["X", "B_total"]].to_numpy()).all()
    assert (traj[["X", "B_total"]] >= 0).all().all()
    assert constraints[["oxygen_lower_bound", "atp_maintenance_lower_bound", "pathway_capacity_upper_bound"]].nunique().max() > 1
    assert fluxes[["beta_carotene_flux", "biomass_flux", "oxygen_uptake", "ggpp_flux"]].nunique().max() > 1


def test_er_isolation_and_reproducibility(augmented_asset):
    path, augmented = augmented_asset
    cfg = gem.GEMCultureConfig(n_time=3)
    er = gem.er_isolation_check(augmented, cfg)
    assert er["er_isolation_passed"]
    traj1, flux1, _c1, _s1 = gem.run_dynamic_culture(augmented, path, cfg, culture_id="rep1")
    traj2, flux2, _c2, _s2 = gem.run_dynamic_culture(augmented, path, cfg, culture_id="rep2")
    assert np.allclose(traj1["B_total"], traj2["B_total"])
    assert np.allclose(flux1["beta_carotene_flux"], flux2["beta_carotene_flux"])
