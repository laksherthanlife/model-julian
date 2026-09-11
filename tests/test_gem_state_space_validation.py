import inspect
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import generate_gem_state_space_dataset as dataset
import run_gem_state_space_validation as validation


def test_split_manifest_is_environment_level_and_has_no_overlap():
    manifest = dataset.assign_splits(dataset.expanded_grid())
    assert manifest["environment_id"].is_unique
    assert set(["train", "validation", "interpolation", "heldout_combination", "extrapolation"]).issubset(set(manifest["split"]))
    assert manifest.groupby("environment_id")["split"].nunique().max() == 1
    train = manifest[manifest["split"].eq("train")]
    assert train["temperature"].nunique() == 5
    assert train["pH"].nunique() == 5
    assert train["DO"].nunique() == 5


def test_validation_script_does_not_import_gem_backend():
    source = Path(validation.__file__).read_text()
    assert "import gem_backend" not in source
    assert "from gem_backend" not in source
    assert "cobra" not in source.lower()


def test_deployment_inputs_are_only_fixed_environment_coordinates():
    assert validation.DIRECT_MODELS
    env = np.array([[27.0, 4.5, 20.0], [30.0, 5.0, 50.0], [33.0, 5.5, 80.0]])
    x = validation.norm_env_raw(env)
    assert x.shape == (3, 3)


def test_preprocessing_uses_training_statistics_only():
    env = np.array([[27.0, 4.5, 20.0], [30.0, 5.0, 50.0], [33.0, 5.5, 80.0]])
    curves = np.array([[0, 1, 2], [0, 2, 4], [100, 200, 300]], dtype=float)
    reporters = {"R_ox": curves.copy()}
    train = np.array([True, True, False])
    stats = validation.fit_training_stats(env, curves, reporters, train)
    assert stats["product_scale"] == 4
    assert np.allclose(stats["env_mean"], env[:2].mean(axis=0))
    assert stats["reporter_mean"]["R_ox"] == reporters["R_ox"][:2].mean()


def test_training_only_pca_ignores_extreme_test_curve():
    train_curves = np.array([[0, 1, 2, 3], [0, 1.1, 2.1, 3.1], [0, 0.9, 1.9, 2.9]])
    test_curve = np.array([[100, 0, 100, 0]])
    mean_train, comps_train, _ = validation.pca_fit(train_curves)
    mean_with_leak, comps_with_leak, _ = validation.pca_fit(np.vstack([train_curves, test_curve]))
    assert not np.allclose(mean_train, mean_with_leak)
    assert not np.allclose(comps_train[0], comps_with_leak[0])


def test_state_space_variants_include_negative_controls_and_oracle():
    variants = set(validation.STATE_SPACE_VARIANTS)
    assert "er_only_state_space" in variants
    assert "shuffled_reporter_control" in variants
    assert "random_smooth_aux_control" in variants
    assert "oracle_state_upper_bound" in variants
