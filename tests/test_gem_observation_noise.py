"""Tests for the seeded post-hoc observation-noise layer (scripts/gem_observation_noise.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

from gem_observation_noise import apply_observation_noise  # noqa: E402


def _toy_traj() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "culture_id": ["c1"] * 4 + ["c2"] * 4,
            "time_index": [0, 1, 2, 3] * 2,
            "B_total": [0.0, 0.3, 0.6, 0.9, 0.0, 0.25, 0.55, 0.8],
            "X": [0.08, 0.10, 0.14, 0.20, 0.08, 0.09, 0.13, 0.18],
            "S_glc": [1000.0, 900.0, 750.0, 600.0, 1000.0, 920.0, 800.0, 650.0],
        }
    )


def test_deterministic_columns_untouched():
    traj = _toy_traj()
    out = apply_observation_noise(traj, seed=7)
    for col in ("B_total", "X", "S_glc"):
        assert (out[f"{col}_true"] == traj[col]).all()
        assert (out[col] == traj[col]).all()  # unmodified alias for existing readers


def test_reproducible_given_same_seed():
    traj = _toy_traj()
    out_a = apply_observation_noise(traj, seed=123)
    out_b = apply_observation_noise(traj, seed=123)
    for col in ("B_total_observed", "X_observed", "S_glc_observed"):
        assert np.allclose(out_a[col], out_b[col])


def test_different_seeds_diverge():
    traj = _toy_traj()
    out_a = apply_observation_noise(traj, seed=1)
    out_b = apply_observation_noise(traj, seed=2)
    assert not np.allclose(out_a["B_total_observed"], out_b["B_total_observed"])


def test_noise_magnitude_is_bounded_and_nonnegative():
    traj = _toy_traj()
    out = apply_observation_noise(traj, seed=11, sigma_product=0.03, sigma_biomass=0.02, sigma_glc=0.03)
    for col in ("B_total_observed", "X_observed", "S_glc_observed"):
        assert (out[col] >= 0.0).all()
    # noise should be small relative to the trajectory's own scale, not dominate it
    rel_err = (out["B_total_observed"] - out["B_total_true"]).abs() / out["B_total_true"].replace(0, np.nan)
    assert rel_err.dropna().mean() < 0.25


def test_per_culture_biological_variability_distinguishes_cultures():
    traj = _toy_traj()
    out = apply_observation_noise(traj, seed=99)
    factors = out.groupby("culture_id")["biological_variability_factor"].first()
    assert factors.nunique() == 2
