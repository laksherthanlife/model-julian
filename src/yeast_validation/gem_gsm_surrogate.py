#!/usr/bin/env python3
"""A small, closed-form, genuinely differentiable surrogate for the real
Yeast9 GSM response function `(compact control vector c, environment) ->
flux outputs`.

Why this exists: Yeast9 is solved by LP (COBRA/GLPK) -- gradients do not pass
through that solve. Per Option A of the handover spec, this module trains a
differentiable approximation of the *mechanistic* GSM response surface from
legitimate historical exact Yeast9 solves (the existing
`gem_state_space_constraints.csv` / `gem_state_space_fluxes.csv`, produced by
real LP solves), then freezes it. The physiology model's control head is
trained *through* this frozen surrogate -- so a product/biomass loss can
shape the predicted control vector `c(t)` (and, via the shared latent state,
the transition dynamics) without ever supervising `c(t)` directly against
privileged simulator values.

The surrogate is deliberately simple: ridge regression on a fixed polynomial
feature map of `(c, env)`, fit in closed form (no iterative training,
`data/gem_gsm_surrogate_fit.csv` for details) -- this keeps it small,
auditable, and trivially differentiable analytically (the Jacobian of a
polynomial-feature ridge model w.r.t. its inputs is exact, not
approximated). It approximates the mechanistic map only; it never sees or
represents hidden regulatory state.

Control vector definition (`CONTROL_COLUMNS`, unchanged from the existing
`GEM_APPLIED_CONTROL_COLUMNS` convention already used by the hybrid-student
code in `run_reporter_grounded_hybrid_distillation.py` -- reused rather than
reinvented): `oxygen_lower_bound, atp_maintenance_lower_bound,
gamma_growth_fraction, PSY_effective_upper_bound, DES_effective_upper_bound,
CYC_effective_upper_bound`. Note this list contains derived GEM bound values,
not raw burden states (`z_ox` etc. are excluded) -- it is the same "compact
metabolic-control vector" Section 8 of the handover spec describes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

CONTROL_COLUMNS = [
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "gamma_growth_fraction",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
]
ENV_COLUMNS = ["temperature", "pH", "DO"]
SURROGATE_OUTPUT_COLUMNS = [
    "biomass_flux",
    "beta_carotene_flux",
    "oxygen_uptake",
    "atp_maintenance_flux",
    "ggpp_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
]
SURROGATE_OUT_INDEX = {col: i for i, col in enumerate(SURROGATE_OUTPUT_COLUMNS)}
EPS = 1e-9


@dataclass
class GSMSurrogate:
    control_mean: np.ndarray
    control_std: np.ndarray
    env_mean: np.ndarray
    env_std: np.ndarray
    W: np.ndarray  # (n_features, n_out)
    n_controls: int
    n_env: int

    def _standardize(self, c: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (c - self.control_mean) / self.control_std, (env - self.env_mean) / self.env_std

    def _features(self, cs: np.ndarray, envs: np.ndarray) -> np.ndarray:
        ones = np.ones((cs.shape[0], 1))
        return np.concatenate([ones, cs, cs**2, envs, envs**2], axis=1)

    def predict(self, c: np.ndarray, env: np.ndarray) -> np.ndarray:
        cs, envs = self._standardize(c, env)
        phi = self._features(cs, envs)
        return phi @ self.W

    def predict_and_grad(self, c: np.ndarray, env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (values (B,n_out), jacobian (B,n_out,n_controls)) --
        d(values)/d(raw c), exact (not finite-differenced), via the chain
        rule through standardization and the polynomial feature map."""
        cs, envs = self._standardize(c, env)
        phi = self._features(cs, envs)
        values = phi @ self.W  # (B, n_out)
        B = c.shape[0]
        n_out = self.W.shape[1]
        jac = np.zeros((B, n_out, self.n_controls))
        # d(phi)/d(cs_i): the "cs_i" feature column has derivative 1, the
        # "cs_i^2" column has derivative 2*cs_i; every other feature has zero
        # derivative w.r.t. cs_i. Feature layout: [1, cs(nc), cs^2(nc), env(ne), env^2(ne)].
        nc = self.n_controls
        for i in range(nc):
            dphi_dcsi = np.zeros((B, phi.shape[1]))
            dphi_dcsi[:, 1 + i] = 1.0
            dphi_dcsi[:, 1 + nc + i] = 2.0 * cs[:, i]
            dvalues_dcsi = dphi_dcsi @ self.W  # (B, n_out)
            jac[:, :, i] = dvalues_dcsi / self.control_std[i]  # chain rule: d(cs_i)/d(c_i) = 1/std_i
        return values, jac


def load_surrogate_training_pairs(constraints_path: Path, fluxes_path: Path, split_manifest_path: Path | None = None) -> pd.DataFrame:
    cons = pd.read_csv(constraints_path)
    flux = pd.read_csv(fluxes_path)
    cols_needed = ["culture_id", "interval_index", "environment_id"] + CONTROL_COLUMNS + ENV_COLUMNS
    cons = cons[[c for c in cols_needed if c in cons.columns]]
    flux_cols = ["culture_id", "interval_index"] + [c for c in SURROGATE_OUTPUT_COLUMNS if c in flux.columns]
    flux = flux[flux_cols]
    merged = cons.merge(flux, on=["culture_id", "interval_index"], how="inner")
    if split_manifest_path is not None and "environment_id" in merged.columns:
        manifest = pd.read_csv(split_manifest_path)
        split_lookup = manifest.set_index("environment_id")["split"].to_dict()
        merged["split"] = merged["environment_id"].map(split_lookup)
    return merged.dropna(subset=CONTROL_COLUMNS + ENV_COLUMNS)


def fit_surrogate(training_pairs: pd.DataFrame, train_mask: np.ndarray | None = None, ridge_lambda: float = 1.0) -> GSMSurrogate:
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

    mask = train_mask if train_mask is not None else np.ones(len(training_pairs), dtype=bool)
    control_mean = c[mask].mean(axis=0)
    control_std = c[mask].std(axis=0) + EPS
    env_mean = env[mask].mean(axis=0)
    env_std = env[mask].std(axis=0) + EPS

    cs = (c - control_mean) / control_std
    envs = (env - env_mean) / env_std
    ones = np.ones((len(training_pairs), 1))
    phi = np.concatenate([ones, cs, cs**2, envs, envs**2], axis=1)

    phi_train = phi[mask]
    y_train = y[mask]
    eye = np.eye(phi_train.shape[1])
    eye[0, 0] = 0.0
    W = np.linalg.solve(phi_train.T @ phi_train + ridge_lambda * eye, phi_train.T @ y_train)

    return GSMSurrogate(control_mean=control_mean, control_std=control_std, env_mean=env_mean, env_std=env_std, W=W, n_controls=len(CONTROL_COLUMNS), n_env=len(ENV_COLUMNS))


def evaluate_surrogate(surrogate: GSMSurrogate, training_pairs: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
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
