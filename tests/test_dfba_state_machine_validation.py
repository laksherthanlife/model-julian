import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_dfba_state_machine_validation as dfba


def test_dfba_fast_runs_all_experiments_and_safeguards():
    metrics, predictions, sample_eff, robustness, safeguards, traj, reporters, fluxes, manifest = dfba.run_all(fast=True, write=False)
    assert {"3A", "3B", "3C", "3D", "3E", "3F"}.issubset(set(metrics["experiment"]))
    assert {"interpolation", "heldout_combination", "extrapolation"}.issubset(set(metrics["split"]))
    assert {"trajectory_rmse", "normalized_rmse", "product_r2", "auc_error", "final_titer_error", "max_titer_error", "peak_time_error", "early_slope_error", "late_decline_error"}.issubset(metrics.columns)
    assert not predictions.empty
    assert not sample_eff.empty
    assert not robustness.empty
    assert safeguards["passed"].all()


def test_dfba_environment_splits_are_unique_across_seeds_and_replicates():
    cfg = dfba.active_config(fast=True)
    traj, fluxes, reporters, transitions, manifest, summary = dfba.generate_dataset(cfg.dataset_seeds[0], cfg, fast=True)
    env_counts = traj.groupby("culture_id")[["temperature", "pH", "DO"]].nunique().max(axis=1)
    assert (env_counts == 1).all()
    split_counts = traj.groupby("culture_id")["split"].nunique()
    assert (split_counts == 1).all()
    grid = dfba.make_environment_grid(fast=True)
    assert grid.groupby("environment_id")["split"].nunique().eq(1).all()
    train_regions = set(grid[grid["split"] == "train"]["heldout_region"])
    assert train_regions == {"training_support"}


def test_er_state_is_causally_isolated_from_fluxes():
    cfg = dfba.active_config(fast=True)
    env = np.array([33.0, 4.5, 60.0])
    burdens_low_er = np.array([0.4, 0.3, 0.2, 0.0])
    burdens_high_er = np.array([0.4, 0.3, 0.2, 1.3])
    low = dfba.reduced_flux_solution(env, burdens_low_er, "productive", cfg)
    high = dfba.reduced_flux_solution(env, burdens_high_er, "productive", cfg)
    for key in ["biomass_flux", "v_beta", "v_precursor", "v_atpm", "rho_atp"]:
        assert low[key] == pytest.approx(high[key])
    assert "R_bottle" not in dfba.generate_dataset(cfg.dataset_seeds[0], cfg, fast=True)[2].columns


def test_nonfast_requires_explicit_metabolic_asset(monkeypatch, tmp_path):
    monkeypatch.delenv("YEAST_GEM_PATH", raising=False)
    monkeypatch.setattr(dfba, "MODELS", tmp_path)
    cfg = dfba.active_config(fast=False)
    with pytest.raises(dfba.MissingMetabolicAsset, match="YEAST_GEM_PATH"):
        dfba.require_metabolic_asset(cfg)


def test_nonfast_refuses_configured_asset_without_lp_backend(monkeypatch, tmp_path):
    fake = tmp_path / "yeast_gem.xml"
    fake.write_text("<sbml></sbml>", encoding="utf-8")
    monkeypatch.setenv("YEAST_GEM_PATH", str(fake))
    cfg = dfba.active_config(fast=False)
    with pytest.raises(dfba.MissingMetabolicBackend, match="real repeated LP/FBA"):
        dfba.require_metabolic_asset(cfg)
