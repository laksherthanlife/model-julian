import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import ode_regime_core as ode


def test_simulator_is_deterministic_nonnegative_and_finite():
    env = np.array([30.5, 5.0, 55.0])
    a, da = ode.simulate_environment(env)
    b, db = ode.simulate_environment(env)
    assert np.allclose(a[ode.STATE_COLUMNS + ode.AUX_COLUMNS], b[ode.STATE_COLUMNS + ode.AUX_COLUMNS])
    assert da["dominant_regime"] == db["dominant_regime"]
    assert np.isfinite(a[ode.STATE_COLUMNS + ode.AUX_COLUMNS]).all().all()
    assert (a[ode.STATE_COLUMNS] >= -1e-12).all().all()


def test_environment_maps_to_one_full_product_trajectory():
    env = np.array([31.0, 4.9, 65.0])
    traj, diag = ode.simulate_environment(env)
    assert len(traj) == ode.ODEConfig().n_time
    assert traj[["temperature", "pH", "DO"]].nunique().max() == 1
    assert "P" in traj
    assert diag["dominant_regime"] in ode.REGIME_ORDER


def test_regime_labels_are_posthoc_not_curve_generators():
    source = Path(ode.__file__).read_text()
    assert "use oxidative-collapse curve" not in source
    assert "hand-drawn production" not in source
    assert "def assign_regime" in source
    assert source.index("def simulate_environment") < source.index("def assign_regime")


def test_small_environment_panel_contains_multiple_mechanisms():
    envs = np.array(
        [
            [30.8, 5.05, 55.0],
            [28.5, 5.05, 55.0],
            [32.7, 5.0, 80.0],
            [30.0, 5.0, 20.0],
            [31.5, 4.75, 35.0],
            [27.0, 5.5, 80.0],
        ]
    )
    regimes = [ode.simulate_environment(e)[1]["dominant_regime"] for e in envs]
    assert len(set(regimes)) >= 4
