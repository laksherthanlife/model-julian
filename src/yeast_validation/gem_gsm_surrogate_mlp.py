#!/usr/bin/env python3
"""Stage 3 replacement for the polynomial-ridge GSM surrogate
(gem_gsm_surrogate.GSMSurrogate): a small, deliberately boring differentiable
MLP fit on the Stage 2 purpose-built exact-Yeast9 dataset (which, unlike the
original surrogate's training data, explicitly covers edited/edge-case/
extrapolative control combinations -- see generate_gsm_surrogate_stage2_points.py
and LEARNER_ARCHITECTURE_REPAIR.md).

Same public contract as GSMSurrogate so it is a drop-in replacement:
  - .predict(c, env) -> (B, n_out)
  - .predict_and_grad(c, env) -> (values (B,n_out), jacobian (B,n_out,n_controls))
    exact analytic Jacobian (hand-derived backprop through one tanh layer,
    same "no autodiff library available" constraint and hand-BPTT convention
    as gem_trainable_state_space.py), not finite-differenced.
  - control_mean/control_std/env_mean/env_std fields with identical meaning.

Architecture: x = [standardized control (nc), standardized env (ne)] ->
tanh(x @ W1 + b1) -> h @ W2 + b2 -> standardized output, unstandardized on
the way out. One hidden layer, width configurable (default 32) -- no reason
for more capacity than the ~9.8k-point Stage 2 dataset supports.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402

CONTROL_COLUMNS = sur.CONTROL_COLUMNS
ENV_COLUMNS = sur.ENV_COLUMNS
SURROGATE_OUTPUT_COLUMNS = sur.SURROGATE_OUTPUT_COLUMNS
EPS = 1e-9


@dataclass
class GSMSurrogateMLP:
    control_mean: np.ndarray
    control_std: np.ndarray
    env_mean: np.ndarray
    env_std: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray
    W1: np.ndarray  # (n_in, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, n_out)
    b2: np.ndarray  # (n_out,)
    n_controls: int
    n_env: int

    def _standardize(self, c: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (c - self.control_mean) / self.control_std, (env - self.env_mean) / self.env_std

    def predict(self, c: np.ndarray, env: np.ndarray) -> np.ndarray:
        cs, envs = self._standardize(c, env)
        x = np.concatenate([cs, envs], axis=1)
        h = np.tanh(x @ self.W1 + self.b1)
        ys = h @ self.W2 + self.b2
        return ys * self.out_std + self.out_mean

    def predict_and_grad(self, c: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (values (B,n_out), jacobian (B,n_out,n_controls)) --
        d(values)/d(raw c), exact analytic gradient through the single tanh
        layer (chain rule through standardization on both input and output)."""
        cs, envs = self._standardize(c, env)
        x = np.concatenate([cs, envs], axis=1)
        pre_h = x @ self.W1 + self.b1
        h = np.tanh(pre_h)
        ys = h @ self.W2 + self.b2
        values = ys * self.out_std + self.out_mean

        B = c.shape[0]
        nc = self.n_controls
        n_out = self.W2.shape[1]
        dh_dpre = 1.0 - h**2  # (B, H)
        # d(ys)/d(x_i) = ((1-h^2) * W1[i,:]) @ W2 -- (B, n_out) for each input i
        jac = np.zeros((B, n_out, nc))
        for i in range(nc):
            dpre_dxi = self.W1[i, :]  # (H,)
            dh_dxi = dh_dpre * dpre_dxi[None, :]  # (B, H)
            dys_dxi = dh_dxi @ self.W2  # (B, n_out)
            dxi_dci = 1.0 / self.control_std[i]
            jac[:, :, i] = dys_dxi * self.out_std[None, :] * dxi_dci
        return values, jac

    def save(self, path: Path) -> None:
        np.savez(
            path,
            control_mean=self.control_mean, control_std=self.control_std,
            env_mean=self.env_mean, env_std=self.env_std,
            out_mean=self.out_mean, out_std=self.out_std,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            n_controls=self.n_controls, n_env=self.n_env,
        )

    @classmethod
    def load(cls, path: Path) -> "GSMSurrogateMLP":
        d = np.load(path)
        return cls(
            control_mean=d["control_mean"], control_std=d["control_std"],
            env_mean=d["env_mean"], env_std=d["env_std"],
            out_mean=d["out_mean"], out_std=d["out_std"],
            W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"],
            n_controls=int(d["n_controls"]), n_env=int(d["n_env"]),
        )


def _init_weights(n_in: int, hidden_dim: int, n_out: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale1 = np.sqrt(1.0 / n_in)
    scale2 = np.sqrt(1.0 / hidden_dim)
    W1 = rng.normal(0, scale1, size=(n_in, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, scale2, size=(hidden_dim, n_out))
    b2 = np.zeros(n_out)
    return W1, b1, W2, b2


def fit_surrogate_mlp(
    training_pairs: pd.DataFrame,
    train_mask: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 2000,
    lr: float = 0.01,
    batch_size: int = 512,
    l2: float = 1e-4,
    seed: int = 7001,
    verbose: bool = False,
) -> GSMSurrogateMLP:
    c = training_pairs[CONTROL_COLUMNS].to_numpy(float)
    env = training_pairs[ENV_COLUMNS].to_numpy(float)
    y_cols = [col for col in SURROGATE_OUTPUT_COLUMNS if col in training_pairs.columns]
    y = training_pairs[y_cols].to_numpy(float)
    if len(y_cols) < len(SURROGATE_OUTPUT_COLUMNS):
        pad = np.zeros((len(training_pairs), len(SURROGATE_OUTPUT_COLUMNS) - len(y_cols)))
        y = np.concatenate([y, pad], axis=1)
        y_cols = y_cols + [col for col in SURROGATE_OUTPUT_COLUMNS if col not in y_cols]
    order = [y_cols.index(col) for col in SURROGATE_OUTPUT_COLUMNS]
    y = y[:, order]

    control_mean = c[train_mask].mean(axis=0)
    control_std = c[train_mask].std(axis=0) + EPS
    env_mean = env[train_mask].mean(axis=0)
    env_std = env[train_mask].std(axis=0) + EPS
    out_mean = y[train_mask].mean(axis=0)
    out_std = y[train_mask].std(axis=0) + EPS

    cs = (c - control_mean) / control_std
    envs = (env - env_mean) / env_std
    x_all = np.concatenate([cs, envs], axis=1)
    ys_all = (y - out_mean) / out_std

    x_train = x_all[train_mask]
    ys_train = ys_all[train_mask]
    n_in = x_train.shape[1]
    n_out = ys_train.shape[1]

    W1, b1, W2, b2 = _init_weights(n_in, hidden_dim, n_out, seed)
    m = {k: np.zeros_like(v) for k, v in {"W1": W1, "b1": b1, "W2": W2, "b2": b2}.items()}
    v = {k: np.zeros_like(val) for k, val in {"W1": W1, "b1": b1, "W2": W2, "b2": b2}.items()}
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    rng = np.random.default_rng(seed + 1)
    n = x_train.shape[0]
    t = 0
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            yb = ys_train[idx]
            bsz = xb.shape[0]

            pre_h = xb @ W1 + b1
            h = np.tanh(pre_h)
            pred = h @ W2 + b2
            err = pred - yb  # (bsz, n_out)

            dW2 = h.T @ err / bsz + l2 * W2
            db2 = err.mean(axis=0)
            dh = err @ W2.T
            dpre_h = dh * (1.0 - h**2)
            dW1 = xb.T @ dpre_h / bsz + l2 * W1
            db1 = dpre_h.mean(axis=0)

            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            for name, p, g in (("W1", W1, dW1), ("b1", b1, db1), ("W2", W2, dW2), ("b2", b2, db2)):
                m[name][:] = beta1 * m[name] + (1 - beta1) * g
                v[name][:] = beta2 * v[name] + (1 - beta2) * (g**2)
                p -= lr_t * m[name] / (np.sqrt(v[name]) + adam_eps)

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            pred_all = np.tanh(x_train @ W1 + b1) @ W2 + b2
            train_loss = float(np.mean((pred_all - ys_train) ** 2))
            print(f"epoch {epoch}: train_mse(standardized)={train_loss:.5f}")

    return GSMSurrogateMLP(
        control_mean=control_mean, control_std=control_std,
        env_mean=env_mean, env_std=env_std,
        out_mean=out_mean, out_std=out_std,
        W1=W1, b1=b1, W2=W2, b2=b2,
        n_controls=len(CONTROL_COLUMNS), n_env=len(ENV_COLUMNS),
    )


def evaluate_surrogate_mlp(surrogate: GSMSurrogateMLP, training_pairs: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    c = training_pairs[CONTROL_COLUMNS].to_numpy(float)
    env = training_pairs[ENV_COLUMNS].to_numpy(float)
    pred = surrogate.predict(c, env)
    rows = []
    for i, col in enumerate(SURROGATE_OUTPUT_COLUMNS):
        if col not in training_pairs.columns:
            continue
        truth = training_pairs[col].to_numpy(float)
        train_truth = truth[mask]
        denom = float(np.std(train_truth) + EPS)
        rows.append(
            {
                "channel": col,
                "held_out_normalized_rmse": float(np.sqrt(np.mean((pred[~mask, i] - truth[~mask]) ** 2)) / denom),
                "train_normalized_rmse": float(np.sqrt(np.mean((pred[mask, i] - truth[mask]) ** 2)) / denom),
            }
        )
    return pd.DataFrame(rows)
