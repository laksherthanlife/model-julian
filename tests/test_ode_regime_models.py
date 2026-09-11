import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import ode_regime_core as ode
import run_ode_regime_validation as validation


def test_model_families_include_required_baselines_and_controls():
    assert "nearest_environment" in validation.DIRECT_MODELS
    assert "polynomial_environment_pca" in validation.DIRECT_MODELS
    assert "coordinate_conditioned_mlp" in validation.DIRECT_MODELS
    assert "direct_multi_output_physiology_mlp" in validation.DIRECT_MODELS
    assert "product_only_state_space" in validation.STATE_SPACE_VARIANTS
    assert "physiology_supervised_state_space" in validation.STATE_SPACE_VARIANTS
    assert "shuffled_aux_state_space" in validation.STATE_SPACE_VARIANTS
    assert "random_smooth_aux_state_space" in validation.STATE_SPACE_VARIANTS


def test_validation_script_never_imports_gem_backend_or_scipy_stack():
    source = Path(validation.__file__).read_text().lower()
    assert "import gem_backend" not in source
    assert "from gem_backend" not in source
    assert "scipy" not in source
    assert "sklearn" not in source
    assert "torch" not in source


def test_metric_values_cover_requested_prediction_quantities():
    t = np.linspace(0, 1, 5)
    truth = np.array([[0.0, 0.1, 0.3, 0.35, 0.36]])
    pred = truth + 0.01
    metrics = validation.metric_values(pred, truth, t)
    for key in ["normalized_rmse", "endpoint_mae", "auc_mae", "max_rate_mae", "time_of_max_rate_mae", "late_stage_rmse", "derivative_rmse"]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_representation_rows_use_environment_only_deployment_metadata():
    meta = pd.DataFrame({"culture_id": ["c1"], "environment_id": ["e1"], "split": ["validation"]})
    rows = []
    validation.add_prediction_rows(rows, meta, np.array([0.0, 1.0]), np.array([[0.0, 1.0]]), np.array([[0.0, 0.9]]), "m", 11, np.array(["balanced_sustained"]))
    assert rows[0]["deployment_inputs"] == "temperature,pH,DO"
    assert "X" not in rows[0]["deployment_inputs"]
