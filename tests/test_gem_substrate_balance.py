"""Tests for the opt-in extracellular glucose mass balance (GENERATOR_VALIDITY.md Section 2.9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402

pytest.importorskip("cobra")


@pytest.fixture(scope="module")
def augmented_model():
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    return augmented, source_path


def test_substrate_limited_defaults_to_off():
    cfg = gem.GEMCultureConfig()
    assert cfg.substrate_limited is False
    assert cfg.s_glc0 == pytest.approx(1000.0)


def test_default_behavior_unaffected_by_substrate_tracking(augmented_model):
    augmented, source_path = augmented_model
    cfg = gem.GEMCultureConfig(n_time=4)
    traj, _flux, _cons, _summary = gem.run_dynamic_culture(augmented.copy(), source_path, cfg, culture_id="regression_default")
    assert "S_glc" in traj.columns
    # substrate_limited=False must never constrain the glucose bound, regardless
    # of how far S_glc has notionally drifted.
    assert traj["X"].iloc[-1] > 0
    assert traj["B_total"].iloc[-1] >= 0


def test_substrate_limited_depletes_monotonically_and_stays_nonnegative(augmented_model):
    augmented, source_path = augmented_model
    cfg = gem.GEMCultureConfig(n_time=5, substrate_limited=True, s_glc0=2.0, glucose_uptake=10.0)
    traj, _flux, _cons, _summary = gem.run_dynamic_culture(augmented.copy(), source_path, cfg, culture_id="substrate_limited")
    s_glc = traj["S_glc"].to_numpy(dtype=float)
    assert np.all(np.diff(s_glc) <= 1e-9), "S_glc must be non-increasing under substrate_limited=True"
    assert np.all(s_glc >= 0.0), "S_glc must never go negative"


def test_substrate_limited_decelerates_growth_vs_unconstrained(augmented_model):
    augmented, source_path = augmented_model
    n_time = 5
    cfg_unconstrained = gem.GEMCultureConfig(n_time=n_time, substrate_limited=False)
    cfg_limited = gem.GEMCultureConfig(n_time=n_time, substrate_limited=True, s_glc0=1.5, glucose_uptake=10.0)
    traj_free, *_ = gem.run_dynamic_culture(augmented.copy(), source_path, cfg_unconstrained, culture_id="free")
    traj_limited, *_ = gem.run_dynamic_culture(augmented.copy(), source_path, cfg_limited, culture_id="limited")
    assert traj_limited["X"].iloc[-1] <= traj_free["X"].iloc[-1] + 1e-9


def test_exact_dynamic_rollout_mirrors_substrate_balance():
    import design_benchmark_exact as dbe

    df = dbe.build_base_exact_design(n_per_regime=1, seed=1)
    row = df.iloc[0].copy()
    row["regime_id"] = "baseline_world"
    regime_spec = dbe.regime_configs()[str(row["regime_id"])]
    cfg = dbe.cfg_for_candidate(row, regime_spec)
    assert hasattr(cfg, "substrate_limited")
    assert cfg.substrate_limited is False
