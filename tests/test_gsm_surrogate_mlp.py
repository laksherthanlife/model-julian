import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_gsm_surrogate_mlp as mlp  # noqa: E402


def _toy_training_pairs(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, len(mlp.CONTROL_COLUMNS))) * np.array([2.0, 0.1, 0.02, 0.03, 0.03, 0.02]) + np.array(
        [-5.0, 1.0, 0.7, 0.19, 0.13, 0.10]
    )
    env = rng.uniform(low=[27, 4.5, 20], high=[33, 5.5, 80], size=(n, 3))
    # synthetic nonlinear ground truth so the MLP has something real to fit
    y0 = 0.3 + 0.05 * np.tanh(c[:, 2] * 3) - 0.01 * c[:, 0]
    y1 = 0.15 + 0.02 * c[:, 3] + 0.01 * np.sin(c[:, 4] * 5)
    y_rest = np.tile((c[:, :1] * 0.1), (1, len(mlp.SURROGATE_OUTPUT_COLUMNS) - 2))
    y = np.concatenate([y0[:, None], y1[:, None], y_rest], axis=1)
    df = pd.DataFrame(c, columns=mlp.CONTROL_COLUMNS)
    for i, col in enumerate(mlp.ENV_COLUMNS):
        df[col] = env[:, i]
    for i, col in enumerate(mlp.SURROGATE_OUTPUT_COLUMNS):
        df[col] = y[:, i]
    return df


def test_gradient_matches_finite_difference():
    df = _toy_training_pairs(400, seed=1)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[: int(0.8 * len(df))] = True
    surrogate = mlp.fit_surrogate_mlp(df, train_mask, hidden_dim=16, epochs=50, seed=3)

    c0 = df[mlp.CONTROL_COLUMNS].to_numpy(float)[:5]
    env0 = df[mlp.ENV_COLUMNS].to_numpy(float)[:5]
    values, jac = surrogate.predict_and_grad(c0, env0)

    eps = 1e-4
    max_rel_err = 0.0
    for b in range(c0.shape[0]):
        for i in range(len(mlp.CONTROL_COLUMNS)):
            c_plus = c0[b : b + 1].copy()
            c_minus = c0[b : b + 1].copy()
            c_plus[0, i] += eps
            c_minus[0, i] -= eps
            v_plus = surrogate.predict(c_plus, env0[b : b + 1])
            v_minus = surrogate.predict(c_minus, env0[b : b + 1])
            numeric = (v_plus[0] - v_minus[0]) / (2 * eps)
            analytic = jac[b, :, i]
            denom = max(1e-6, np.max(np.abs(numeric)))
            rel_err = np.max(np.abs(numeric - analytic)) / denom
            max_rel_err = max(max_rel_err, rel_err)

    assert max_rel_err < 1e-3, f"gradient check failed, max relative error {max_rel_err}"


def test_save_load_roundtrip(tmp_path):
    df = _toy_training_pairs(200, seed=2)
    train_mask = np.ones(len(df), dtype=bool)
    surrogate = mlp.fit_surrogate_mlp(df, train_mask, hidden_dim=8, epochs=20, seed=5)
    path = tmp_path / "surrogate.npz"
    surrogate.save(path)
    loaded = mlp.GSMSurrogateMLP.load(path)

    c0 = df[mlp.CONTROL_COLUMNS].to_numpy(float)[:10]
    env0 = df[mlp.ENV_COLUMNS].to_numpy(float)[:10]
    np.testing.assert_allclose(surrogate.predict(c0, env0), loaded.predict(c0, env0))


def test_fit_reduces_held_out_error_vs_mean_baseline():
    df = _toy_training_pairs(600, seed=9)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[: int(0.8 * len(df))] = True
    surrogate = mlp.fit_surrogate_mlp(df, train_mask, hidden_dim=16, epochs=300, seed=11)
    metrics = mlp.evaluate_surrogate_mlp(surrogate, df, train_mask)
    assert (metrics["held_out_normalized_rmse"] < 1.0).all()
