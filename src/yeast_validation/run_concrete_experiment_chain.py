#!/usr/bin/env python3
"""Run the concrete four-step synthetic physiology validation chain.

The point of this runner is not to be a full biological simulator. It is a
small, auditable proof sequence:

1. recover a latent controller in an ODE expression/product system;
2. show fluorescent reporter supervision improves recovery with 3 latent
   controllers;
3. move the same controller idea into a tiny dFBA/GSM-like simulator with
   product and reporter sinks;
4. benchmark the learned state-space simulator against pure ML rollout models.

Only numpy and pandas are required. Figures are written as simple SVGs so the
bundle remains runnable in minimal environments.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
NOTEBOOKS = ROOT / "notebooks"
DT = 0.5
T_STEPS = 73
TRAIN_N = 72
TEST_N = 28
STATE_NAMES = ["product_controller", "proteotoxic_reporter", "oxygen_stress"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def corr_alignment(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if pred.ndim == 1:
        pred = pred[:, None]
    if truth.ndim == 1:
        truth = truth[:, None]
    scores = []
    for j in range(truth.shape[1]):
        best = 0.0
        y = truth[:, j]
        y = y - y.mean()
        for k in range(pred.shape[1]):
            x = pred[:, k] - pred[:, k].mean()
            den = np.linalg.norm(x) * np.linalg.norm(y)
            if den > 1e-12:
                best = max(best, abs(float(np.dot(x, y) / den)))
        scores.append(best)
    return scores


def ridge_fit(X, Y, lam=1e-5):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ Y)


def design_env(n, rng):
    """Create diverse time-varying environmental conditions."""
    all_env = []
    t = np.arange(T_STEPS) * DT
    for i in range(n):
        light_amp = rng.uniform(0.35, 1.1)
        temp_shift = rng.uniform(-1.0, 1.0)
        feed = rng.uniform(0.45, 1.0)
        oxy_supply = rng.uniform(0.45, 1.0)
        pulse_start = rng.uniform(3.0, 10.0)
        pulse_len = rng.uniform(5.0, 18.0)
        pulse = ((t >= pulse_start) & (t <= pulse_start + pulse_len)).astype(float)
        ramp = np.clip((t - pulse_start) / max(pulse_len, 1e-6), 0, 1)
        light = light_amp * np.maximum(pulse, 0.25 * ramp)
        temp = 30.0 + temp_shift + rng.normal(0, 0.08, T_STEPS)
        glucose = np.clip(feed + 0.12 * np.sin(0.35 * t + i), 0.15, 1.2)
        oxygen = np.clip(oxy_supply + 0.08 * np.cos(0.2 * t + 0.3 * i), 0.12, 1.1)
        env = np.column_stack(
            [
                light,
                (temp - 30.0) / 4.0,
                glucose,
                oxygen,
                np.full(T_STEPS, light_amp),
                np.full(T_STEPS, feed),
                np.full(T_STEPS, oxy_supply),
            ]
        )
        all_env.append(env)
    return np.asarray(all_env)


def simulate_ode_one(env, rng):
    n, T, _ = env.shape
    z = np.zeros((n, T))
    expr = np.zeros((n, T))
    product = np.zeros((n, T))
    for i in range(n):
        z[i, 0] = 0.25 * env[i, 0, 0] - 0.2 * (1 - env[i, 0, 3])
        for t in range(T - 1):
            light, temp, glucose, oxygen = env[i, t, :4]
            drive = 1.15 * light + 0.35 * glucose - 0.45 * (1 - oxygen) - 0.25 * temp
            z[i, t + 1] = 0.86 * z[i, t] + DT * (0.42 * drive - 0.25 * z[i, t])
            expr[i, t + 1] = expr[i, t] + DT * (
                1.15 * sigmoid(1.8 * z[i, t]) * light - 0.32 * expr[i, t]
            )
            product[i, t + 1] = product[i, t] + DT * (
                0.9 * expr[i, t] - 0.045 * product[i, t]
            )
        product[i] += rng.normal(0, 0.025, T)
    return {"z": z, "expr": expr, "product": product}


def infer_ode1_state(product):
    d = np.diff(product, axis=1, prepend=product[:, :1]) / DT
    # The observer only knows the output curve. A smoothed derivative is its
    # latent proxy; affine scaling is handled in the alignment metric.
    proxy = np.zeros_like(product)
    for i in range(product.shape[0]):
        proxy[i] = np.convolve(d[i], np.ones(5) / 5, mode="same")
    return (proxy - proxy.mean()) / (proxy.std() + 1e-9)


@dataclass
class StateSpaceFit:
    dyn: np.ndarray
    out: np.ndarray
    z0: np.ndarray
    latent_dim: int


def fit_state_space(env, outputs, z_proxy):
    """Fit z[t+1]=A[1,z[t],env[t]] and y[t+1]=C[1,y[t],z[t]].

    The output head intentionally does not receive environment directly. The
    contract being tested is environment -> learned latent state -> simulation.
    """
    X_dyn, Y_dyn, X_out, Y_out = [], [], [], []
    y = outputs
    if y.ndim == 2:
        y = y[:, :, None]
    if z_proxy.ndim == 2:
        z_proxy = z_proxy[:, :, None]
    for i in range(env.shape[0]):
        for t in range(env.shape[1] - 1):
            X_dyn.append(np.r_[1.0, z_proxy[i, t], env[i, t]])
            Y_dyn.append(z_proxy[i, t + 1])
            X_out.append(np.r_[1.0, y[i, t], z_proxy[i, t]])
            Y_out.append(y[i, t + 1])
    z0 = ridge_fit(
        np.column_stack([np.ones(env.shape[0]), env[:, 0, :]]),
        z_proxy[:, 0, :],
        lam=1e-4,
    )
    return StateSpaceFit(
        dyn=ridge_fit(np.asarray(X_dyn), np.asarray(Y_dyn), lam=1e-4),
        out=ridge_fit(np.asarray(X_out), np.asarray(Y_out), lam=1e-4),
        z0=z0,
        latent_dim=z_proxy.shape[2],
    )


def rollout_state_space(fit, env, output_dim, initial_y=None):
    n, T, _ = env.shape
    z = np.zeros((n, T, fit.latent_dim))
    y = np.zeros((n, T, output_dim))
    z[:, 0, :] = np.column_stack([np.ones(n), env[:, 0, :]]) @ fit.z0
    if initial_y is not None:
        y[:, 0, :] = initial_y
    for i in range(n):
        for t in range(T - 1):
            z[i, t + 1] = np.r_[1.0, z[i, t], env[i, t]] @ fit.dyn
            y[i, t + 1] = np.r_[1.0, y[i, t], z[i, t]] @ fit.out
            y[i, t + 1] = np.maximum(y[i, t + 1], 0.0)
    return z, y


def simulate_ode_three(env, rng):
    n, T, _ = env.shape
    z = np.zeros((n, T, 3))
    expr = np.zeros((n, T))
    product = np.zeros((n, T))
    reporters = np.zeros((n, T, 2))
    B = np.array(
        [
            [0.80, -0.20, 0.25, 0.05],
            [0.15, 0.90, 0.10, -0.35],
            [-0.20, 0.15, -0.15, -1.05],
        ]
    )
    for i in range(n):
        z[i, 0] = np.array([0.2, -0.1, 0.1])
        for t in range(T - 1):
            env4 = env[i, t, :4]
            drive = B @ env4 + np.array([-0.25, 0.1, 0.55])
            z[i, t + 1] = 0.84 * z[i, t] + DT * (0.45 * drive - 0.18 * z[i, t])
            effective = 1.2 * z[i, t, 0] - 0.85 * z[i, t, 1] - 0.75 * z[i, t, 2]
            expr[i, t + 1] = expr[i, t] + DT * (
                env[i, t, 0] * sigmoid(effective) - 0.28 * expr[i, t]
            )
            product[i, t + 1] = product[i, t] + DT * (
                1.05 * expr[i, t] - 0.04 * product[i, t]
            )
            for j in range(2):
                reporters[i, t + 1, j] = reporters[i, t, j] + DT * (
                    (0.85 + 0.48 * z[i, t, j]) - 0.30 * reporters[i, t, j]
                )
        product[i] += rng.normal(0, 0.03, T)
        reporters[i] += rng.normal(0, 0.015, (T, 2))
    return {"z": z, "expr": expr, "product": product, "reporters": reporters}


def infer_reporter_state(F):
    d = np.diff(F, axis=1, prepend=F[:, :1]) / DT
    z = (d + 0.30 * F - 0.85) / 0.48
    for i in range(z.shape[0]):
        z[i] = np.convolve(z[i], np.ones(3) / 3, mode="same")
    return np.clip(z, -4, 4)


def build_ode3_proxy(product, reporters=None):
    prod_proxy = infer_ode1_state(product)
    if reporters is None:
        return prod_proxy[:, :, None]
    cols = [infer_reporter_state(reporters[:, :, j]) for j in range(reporters.shape[2])]
    residual = prod_proxy
    for c in cols:
        residual = residual - 0.2 * c
    cols.append(residual)
    return np.stack(cols, axis=2)


def fba_vertex(product_cap, r1_cap, r2_cap, glucose_cap, oxygen_cap, stress, growth_cap=None):
    """Tiny FBA: choose biomass/product/reporter fluxes under C/O limits."""
    # Variables: biomass, beta-carotene sink, reporter1 sink, reporter2 sink.
    if growth_cap is None:
        growth_cap = 0.16 * (1 - 0.25 * stress)
    limits = np.array([growth_cap, product_cap, r1_cap, r2_cap])
    A = [
        np.array([2.0, 1.35, 0.65, 0.65]),
        np.array([1.0, 1.10, 0.40, 0.40]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]
    b = np.array([glucose_cap, oxygen_cap, *limits])
    obj = np.array([1.0, 0.18, 0.06, 0.06])
    best, best_score = np.zeros(4), -1e9
    # Enumerate intersections of active constraints. Four variables, six
    # inequality constraints: small enough to do exactly without scipy.
    from itertools import combinations

    for idx in combinations(range(6), 4):
        M = np.vstack([A[k] for k in idx])
        rhs = b[list(idx)]
        try:
            v = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        if np.all(v >= -1e-8) and all(float(a @ v) <= bb + 1e-7 for a, bb in zip(A, b)):
            score = float(obj @ v)
            if score > best_score:
                best, best_score = np.maximum(v, 0), score
    return best


def simulate_dfba(env, rng):
    n, T, _ = env.shape
    z = np.zeros((n, T, 3))
    out = np.zeros((n, T, 4))  # product, biomass, reporter1, reporter2
    oxygen = np.zeros((n, T))
    flux = np.zeros((n, T, 4))
    for i in range(n):
        out[i, 0, 1] = 0.08
        oxygen[i, 0] = env[i, 0, 3]
        for t in range(T - 1):
            light, temp, glucose_feed, oxy_supply = env[i, t, :4]
            oxygen_deficit = max(0.0, 0.55 - oxygen[i, t])
            product_burden = min(1.0, out[i, t, 0] / 4.0)
            drive = np.array(
                [
                    0.9 * light + 0.25 * glucose_feed - 0.2 * temp - 0.35 * product_burden,
                    0.45 * light + 0.4 * product_burden + 0.35 * temp,
                    1.25 * oxygen_deficit + 0.2 * product_burden - 0.25 * oxy_supply,
                ]
            )
            z[i, t + 1] = 0.82 * z[i, t] + DT * (0.55 * drive - 0.22 * z[i, t])
            caps = 0.03 + np.array([0.50, 0.22, 0.22]) * sigmoid(2.0 * z[i, t])
            v = fba_vertex(
                caps[0],
                caps[1],
                caps[2],
                glucose_cap=1.15 * glucose_feed,
                oxygen_cap=max(0.05, 1.2 * oxygen[i, t]),
                stress=sigmoid(z[i, t, 2]),
                growth_cap=0.16 * (1 - 0.25 * sigmoid(z[i, t, 2])),
            )
            flux[i, t] = v
            X = out[i, t, 1]
            out[i, t + 1, 1] = max(0.0, X + DT * v[0] * X)
            out[i, t + 1, 0] = out[i, t, 0] + DT * v[1] * X
            out[i, t + 1, 2] = out[i, t, 2] + DT * (v[2] * X - 0.08 * out[i, t, 2])
            out[i, t + 1, 3] = out[i, t, 3] + DT * (v[3] * X - 0.08 * out[i, t, 3])
            oxygen[i, t + 1] = np.clip(
                oxygen[i, t]
                + DT * (0.35 * (oxy_supply - oxygen[i, t]) - 0.18 * v[0] * X - 0.08 * v[1] * X),
                0.02,
                1.2,
            )
        out[i] += rng.normal(0, [0.004, 0.002, 0.002, 0.002], (T, 4))
        out[i] = np.maximum(out[i], 0)
    return {"z": z, "outputs": out, "oxygen": oxygen, "flux": flux}


def infer_dfba_proxy(outputs, reporter_count):
    product, biomass = outputs[:, :, 0], outputs[:, :, 1]
    X = np.maximum(biomass, 0.03)
    prod_flux_proxy = np.diff(product, axis=1, prepend=product[:, :1]) / DT / X
    cols = [(prod_flux_proxy - prod_flux_proxy.mean()) / (prod_flux_proxy.std() + 1e-9)]
    for j in range(reporter_count):
        F = outputs[:, :, 2 + j]
        dF = np.diff(F, axis=1, prepend=F[:, :1]) / DT
        rep_flux = (dF + 0.08 * F) / X
        cols.append((rep_flux - rep_flux.mean()) / (rep_flux.std() + 1e-9))
    return np.stack(cols, axis=2)


def fit_dfba_state_space(env, outputs, z_proxy):
    X_dyn, Y_dyn = [], []
    if z_proxy.ndim == 2:
        z_proxy = z_proxy[:, :, None]
    for i in range(env.shape[0]):
        for t in range(env.shape[1] - 1):
            oxygen_feedback = env[i, t, 3]
            product_burden = min(1.0, outputs[i, t, 0] / 4.0)
            X_dyn.append(np.r_[1.0, z_proxy[i, t], env[i, t], oxygen_feedback, product_burden])
            Y_dyn.append(z_proxy[i, t + 1])
    X0 = np.column_stack([np.ones(env.shape[0]), env[:, 0, :]])
    z0 = ridge_fit(X0, z_proxy[:, 0, :], lam=1e-4)

    X_cap, y_growth, y_product, y_r1, y_r2 = [], [], [], [], []
    biomass = np.maximum(outputs[:, :, 1], 0.035)
    growth_flux = np.diff(outputs[:, :, 1], axis=1, prepend=outputs[:, :1, 1]) / DT / biomass
    product_flux = np.diff(outputs[:, :, 0], axis=1, prepend=outputs[:, :1, 0]) / DT / biomass
    r1_flux = (np.diff(outputs[:, :, 2], axis=1, prepend=outputs[:, :1, 2]) / DT + 0.08 * outputs[:, :, 2]) / biomass
    r2_flux = (np.diff(outputs[:, :, 3], axis=1, prepend=outputs[:, :1, 3]) / DT + 0.08 * outputs[:, :, 3]) / biomass
    for i in range(env.shape[0]):
        for t in range(env.shape[1]):
            X_cap.append(np.r_[1.0, z_proxy[i, t]])
            y_growth.append(growth_flux[i, t])
            y_product.append(product_flux[i, t])
            y_r1.append(r1_flux[i, t])
            y_r2.append(r2_flux[i, t])
    X_cap = np.asarray(X_cap)
    return {
        "dyn": ridge_fit(np.asarray(X_dyn), np.asarray(Y_dyn), lam=1e-4),
        "z0": z0,
        "latent_dim": z_proxy.shape[2],
        "growth_cap": ridge_fit(X_cap, np.asarray(y_growth), lam=1e-3),
        "product_cap": ridge_fit(X_cap, np.asarray(y_product), lam=1e-3),
        "r1_cap": ridge_fit(X_cap, np.asarray(y_r1), lam=1e-3),
        "r2_cap": ridge_fit(X_cap, np.asarray(y_r2), lam=1e-3),
    }


def rollout_dfba_model(model, env, output_dim=4):
    n, T, _ = env.shape
    z = np.zeros((n, T, model["latent_dim"]))
    out = np.zeros((n, T, output_dim))
    oxygen = np.zeros((n, T))
    z[:, 0] = np.column_stack([np.ones(n), env[:, 0, :]]) @ model["z0"]
    out[:, 0, 1] = 0.08
    oxygen[:, 0] = env[:, 0, 3]
    for i in range(n):
        for t in range(T - 1):
            product_burden = min(1.0, out[i, t, 0] / 4.0)
            feat = np.r_[1.0, z[i, t], env[i, t], oxygen[i, t], product_burden]
            z[i, t + 1] = feat @ model["dyn"]
            cap_feat = np.r_[1.0, z[i, t]]
            growth_cap = float(np.clip(cap_feat @ model["growth_cap"], 0.005, 0.22))
            product_cap = float(np.clip(cap_feat @ model["product_cap"], 0.01, 0.75))
            r1_cap = float(np.clip(cap_feat @ model["r1_cap"], 0.01, 0.35))
            r2_cap = float(np.clip(cap_feat @ model["r2_cap"], 0.01, 0.35))
            stress = 0.0 if model["latent_dim"] < 3 else float(sigmoid(z[i, t, -1]))
            v = fba_vertex(
                product_cap,
                r1_cap,
                r2_cap,
                glucose_cap=1.15 * env[i, t, 2],
                oxygen_cap=max(0.05, 1.2 * oxygen[i, t]),
                stress=stress,
                growth_cap=growth_cap,
            )
            X = out[i, t, 1]
            out[i, t + 1, 1] = max(0.0, X + DT * v[0] * X)
            out[i, t + 1, 0] = out[i, t, 0] + DT * v[1] * X
            out[i, t + 1, 2] = out[i, t, 2] + DT * (v[2] * X - 0.08 * out[i, t, 2])
            out[i, t + 1, 3] = out[i, t, 3] + DT * (v[3] * X - 0.08 * out[i, t, 3])
            oxygen[i, t + 1] = np.clip(
                oxygen[i, t]
                + DT * (0.35 * (env[i, t, 3] - oxygen[i, t]) - 0.18 * v[0] * X - 0.08 * v[1] * X),
                0.02,
                1.2,
            )
    return z, out


def baseline_features(env, outputs):
    X, Y = [], []
    for i in range(env.shape[0]):
        for t in range(env.shape[1] - 1):
            X.append(np.r_[1.0, env[i, t], outputs[i, t], t / (env.shape[1] - 1)])
            Y.append(outputs[i, t + 1])
    return np.asarray(X), np.asarray(Y)


class SimpleTree:
    def __init__(self, depth=5, min_leaf=18):
        self.depth = depth
        self.min_leaf = min_leaf
        self.node = None

    def fit(self, X, Y):
        self.node = self._build(X, Y, self.depth)
        return self

    def _build(self, X, Y, depth):
        if depth == 0 or len(X) < 2 * self.min_leaf:
            return {"value": Y.mean(axis=0)}
        base = np.var(Y, axis=0).sum() * len(Y)
        best = None
        for f in range(1, X.shape[1]):
            qs = np.quantile(X[:, f], [0.2, 0.35, 0.5, 0.65, 0.8])
            for thr in np.unique(qs):
                left = X[:, f] <= thr
                if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                    continue
                loss = np.var(Y[left], axis=0).sum() * left.sum()
                loss += np.var(Y[~left], axis=0).sum() * (~left).sum()
                gain = base - loss
                if best is None or gain > best[0]:
                    best = (gain, f, thr, left)
        if best is None:
            return {"value": Y.mean(axis=0)}
        _, f, thr, left = best
        return {
            "feature": f,
            "threshold": float(thr),
            "left": self._build(X[left], Y[left], depth - 1),
            "right": self._build(X[~left], Y[~left], depth - 1),
        }

    def predict_one(self, x):
        node = self.node
        while "value" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]


def fit_mlp(X, Y, rng, hidden=24, epochs=450, lr=0.015):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    ym, ys = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    Xn, Yn = (X - mu) / sd, (Y - ym) / ys
    W1 = rng.normal(0, 0.12, (X.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.12, (hidden, Y.shape[1]))
    b2 = np.zeros(Y.shape[1])
    for _ in range(epochs):
        H = np.tanh(Xn @ W1 + b1)
        pred = H @ W2 + b2
        d = 2 * (pred - Yn) / len(Xn)
        gW2 = H.T @ d
        gb2 = d.sum(axis=0)
        dH = (d @ W2.T) * (1 - H * H)
        gW1 = Xn.T @ dH
        gb1 = dH.sum(axis=0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    return {"mu": mu, "sd": sd, "ym": ym, "ys": ys, "W1": W1, "b1": b1, "W2": W2, "b2": b2}


def mlp_predict(model, X):
    Xn = (X - model["mu"]) / model["sd"]
    H = np.tanh(Xn @ model["W1"] + model["b1"])
    return (H @ model["W2"] + model["b2"]) * model["ys"] + model["ym"]


def rollout_ml_model(kind, model, env, output_dim=4):
    n, T, _ = env.shape
    out = np.zeros((n, T, output_dim))
    out[:, 0, 1] = 0.08
    for i in range(n):
        for t in range(T - 1):
            x = np.r_[1.0, env[i, t], out[i, t], t / (T - 1)]
            if kind == "ridge":
                y = x @ model
            elif kind == "tree":
                y = model.predict_one(x)
            else:
                y = mlp_predict(model, x[None, :])[0]
            out[i, t + 1] = np.maximum(y, 0.0)
    return out


INTERVENTIONS = {
    "base": {"product_scale": 1.00, "growth_scale": 1.00, "atp_cost": 1.00, "maintenance": 0.00},
    "high_capacity": {"product_scale": 1.35, "growth_scale": 0.92, "atp_cost": 1.08, "maintenance": 0.02},
    "high_atp_cost": {"product_scale": 0.95, "growth_scale": 0.82, "atp_cost": 1.45, "maintenance": 0.05},
    "low_pathway": {"product_scale": 0.55, "growth_scale": 1.08, "atp_cost": 0.92, "maintenance": 0.00},
}

PHYSIOLOGY_REGIMES = {
    "A_hot_induction": {"temp": 1.00, "light": 0.95, "feed": 0.82, "oxygen": 0.88, "pulse_len": 15.0},
    "B_low_oxygen": {"temp": -0.20, "light": 0.70, "feed": 0.62, "oxygen": 0.42, "pulse_len": 13.0},
    "balanced": {"temp": 0.35, "light": 0.82, "feed": 0.75, "oxygen": 0.68, "pulse_len": 11.0},
    "unseen_long_double": {"temp": 0.75, "light": 1.02, "feed": 0.68, "oxygen": 0.50, "pulse_len": 23.0},
}

TRAIN_COMBOS = {
    ("A_hot_induction", "base"),
    ("A_hot_induction", "high_capacity"),
    ("B_low_oxygen", "high_capacity"),
    ("B_low_oxygen", "high_atp_cost"),
    ("balanced", "base"),
    ("balanced", "high_atp_cost"),
    ("balanced", "low_pathway"),
}

CROSSED_TEST_COMBOS = {
    ("A_hot_induction", "high_atp_cost"),
    ("B_low_oxygen", "base"),
    ("balanced", "high_capacity"),
}

UNSEEN_TEST_COMBOS = {
    ("unseen_long_double", "base"),
    ("unseen_long_double", "low_pathway"),
}


def intervention_vector(name):
    p = INTERVENTIONS[name]
    return np.array([p["product_scale"], p["growth_scale"], p["atp_cost"], p["maintenance"]], dtype=float)


def make_crossed_env(regime, replicate, rng):
    cfg = PHYSIOLOGY_REGIMES[regime]
    t = np.arange(T_STEPS) * DT
    start = 4.0 + 0.45 * (replicate % 5) + rng.normal(0, 0.2)
    pulse_len = cfg["pulse_len"] + rng.normal(0, 0.7)
    pulse = ((t >= start) & (t <= start + pulse_len)).astype(float)
    if regime == "unseen_long_double":
        second = ((t >= start + 13.0) & (t <= start + 13.0 + 8.0)).astype(float)
        pulse = np.maximum(pulse, 0.85 * second)
    ramp = np.clip((t - start) / max(pulse_len, 1e-6), 0, 1)
    light = cfg["light"] * np.maximum(pulse, 0.18 * ramp) + rng.normal(0, 0.015, T_STEPS)
    temp = cfg["temp"] + 0.08 * np.sin(0.25 * t + replicate) + rng.normal(0, 0.02, T_STEPS)
    feed = cfg["feed"] + 0.06 * np.cos(0.19 * t + 0.2 * replicate)
    oxygen = cfg["oxygen"] + 0.05 * np.sin(0.21 * t + 0.5 * replicate)
    init_x = 0.075 + 0.012 * ((replicate % 4) - 1.5)
    return np.column_stack(
        [
            np.clip(light, 0, 1.35),
            np.clip(temp, -1.4, 1.4),
            np.clip(feed, 0.12, 1.15),
            np.clip(oxygen, 0.06, 1.1),
            np.full(T_STEPS, init_x),
            np.full(T_STEPS, start / 24.0),
            np.full(T_STEPS, pulse_len / 24.0),
        ]
    )


def crossed_fba(growth_cap, product_cap, rep_a_cap, rep_b_cap, env_t, intervention, b_state):
    product_scale, growth_scale, atp_cost, maintenance = intervention
    glucose_cap = 1.12 * env_t[2]
    oxygen_cap = max(0.04, 1.18 * env_t[3] / max(atp_cost, 1e-6))
    return fba_vertex(
        product_cap * product_scale,
        rep_a_cap,
        rep_b_cap,
        glucose_cap=glucose_cap,
        oxygen_cap=oxygen_cap,
        stress=sigmoid(b_state),
        growth_cap=max(0.004, growth_cap * growth_scale - maintenance),
    )


def simulate_crossed_dataset(rng, world="separable", reps_per_combo=14):
    envs, interventions, outputs, states, reporters, fluxes, meta = [], [], [], [], [], [], []
    combos = sorted(TRAIN_COMBOS | CROSSED_TEST_COMBOS | UNSEEN_TEST_COMBOS)
    for regime, intervention_name in combos:
        for rep in range(reps_per_combo):
            env = make_crossed_env(regime, rep, rng)
            interv = intervention_vector(intervention_name)
            z = np.zeros((T_STEPS, 2))
            out = np.zeros((T_STEPS, 4))  # product, biomass, reporter A, reporter B
            reporter_mrna = np.zeros((T_STEPS, 2))
            flux = np.zeros((T_STEPS, 4))
            out[0, 1] = env[0, 4]
            for t in range(T_STEPS - 1):
                light, temp, feed, oxygen = env[t, :4]
                oxygen_deficit = max(0.0, 0.55 - oxygen)
                base_drive = np.array(
                    [
                        0.92 * temp + 0.48 * light - 0.28 * feed + 0.12,
                        1.12 * oxygen_deficit + 0.58 * (0.72 - feed) + 0.24 * light - 0.10,
                    ]
                )
                if world == "entangled":
                    base_drive[0] += 0.42 * (interv[0] - 1.0) + 0.12 * flux[t - 1, 1] if t else 0.0
                    base_drive[1] += 0.36 * (interv[2] - 1.0) + 0.18 * flux[t - 1, 0] if t else 0.0
                z[t + 1] = 0.84 * z[t] + DT * (0.48 * base_drive - 0.20 * z[t])
                growth_cap = 0.16 * sigmoid(1.15 - 1.25 * z[t, 1])
                product_cap = 0.12 + 0.42 * sigmoid(1.25 - 1.55 * z[t, 0])
                rep_a_cap = 0.055 + 0.15 * sigmoid(1.8 * z[t, 0])
                rep_b_cap = 0.055 + 0.15 * sigmoid(1.8 * z[t, 1])
                v = crossed_fba(growth_cap, product_cap, rep_a_cap, rep_b_cap, env[t], interv, z[t, 1])
                flux[t] = v
                x = out[t, 1]
                out[t + 1, 1] = max(0.0, x + DT * v[0] * x)
                out[t + 1, 0] = out[t, 0] + DT * v[1] * x
                reporter_drive = np.array(
                    [
                        sigmoid(1.25 * (z[t, 0] + 0.02 * z[t, 1])),
                        sigmoid(1.20 * (z[t, 1] + 0.02 * z[t, 0])),
                    ]
                )
                reporter_mrna[t + 1] = reporter_mrna[t] + DT * (1.25 * reporter_drive - 0.42 * reporter_mrna[t])
                out[t + 1, 2] = out[t, 2] + DT * (0.62 * reporter_mrna[t, 0] - 0.10 * out[t, 2])
                out[t + 1, 3] = out[t, 3] + DT * (0.58 * reporter_mrna[t, 1] - 0.10 * out[t, 3])
            out += rng.normal(0, [0.003, 0.0015, 0.003, 0.003], out.shape)
            out = np.maximum(out, 0.0)
            envs.append(env)
            interventions.append(interv)
            outputs.append(out)
            states.append(z)
            reporters.append(out[:, 2:4])
            fluxes.append(flux)
            if (regime, intervention_name) in TRAIN_COMBOS:
                split = "train"
            elif (regime, intervention_name) in CROSSED_TEST_COMBOS:
                split = "withheld_crossed"
            else:
                split = "unseen_perturbation"
            meta.append(
                {
                    "world": world,
                    "physiology_regime": regime,
                    "intervention": intervention_name,
                    "split": split,
                    "replicate": rep,
                }
            )
    return {
        "env": np.asarray(envs),
        "intervention": np.asarray(interventions),
        "outputs": np.asarray(outputs),
        "states": np.asarray(states),
        "reporters": np.asarray(reporters),
        "flux": np.asarray(fluxes),
        "meta": meta,
    }


def reporter_proxy_from_fluorescence(reporters):
    proxies = []
    for j in range(2):
        F = reporters[:, :, j]
        dF = np.diff(F, axis=1, prepend=F[:, :1]) / DT
        mrna_proxy = (dF + 0.10 * F) / (0.62 if j == 0 else 0.58)
        dm = np.diff(mrna_proxy, axis=1, prepend=mrna_proxy[:, :1]) / DT
        hill_proxy = (dm + 0.42 * mrna_proxy) / 1.25
        z = logit(hill_proxy) / (1.25 if j == 0 else 1.20)
        for i in range(z.shape[0]):
            z[i] = np.convolve(z[i], np.ones(3) / 3, mode="same")
        proxies.append(np.clip(z, -4, 4))
    return np.stack(proxies, axis=2)


def product_only_crossed_proxy(outputs):
    product, biomass = outputs[:, :, 0], outputs[:, :, 1]
    x = np.maximum(biomass, 0.035)
    prod_flux = np.diff(product, axis=1, prepend=product[:, :1]) / DT / x
    lagged_flux = np.column_stack([prod_flux[:, :1], prod_flux[:, :-1]])
    cols = []
    # Product/biomass-only supervision is intentionally ambiguous: it sees the
    # product trajectory and a local product acceleration proxy, but not the
    # calibrated reporter channels that distinguish A-like from B-like causes.
    for arr in [prod_flux, lagged_flux]:
        cols.append((arr - arr.mean()) / (arr.std() + 1e-9))
    return np.stack(cols, axis=2)


def crossed_cap_features(z_t, intervention):
    z_t = np.asarray(z_t)
    intervention = np.asarray(intervention)
    active = sigmoid(z_t)
    return np.r_[1.0, z_t, active, intervention, active[0] * intervention, active[-1] * intervention]


def crossed_dyn_features(z_t, env_t):
    z_t = np.asarray(z_t)
    return np.r_[1.0, z_t, env_t]


def fit_crossed_hybrid(dataset, train_idx, proxy):
    env = dataset["env"][train_idx]
    interv = dataset["intervention"][train_idx]
    outputs = dataset["outputs"][train_idx]
    if proxy.ndim == 2:
        proxy = proxy[:, :, None]
    X_dyn, Y_dyn = [], []
    for i in range(len(train_idx)):
        for t in range(T_STEPS - 1):
            X_dyn.append(crossed_dyn_features(proxy[i, t], env[i, t]))
            Y_dyn.append(proxy[i, t + 1])
    z0 = ridge_fit(np.column_stack([np.ones(len(train_idx)), env[:, 0, :]]), proxy[:, 0, :], lam=1e-4)

    biomass = np.maximum(outputs[:, :, 1], 0.035)
    growth_flux = np.diff(outputs[:, :, 1], axis=1, prepend=outputs[:, :1, 1]) / DT / biomass
    product_flux = np.diff(outputs[:, :, 0], axis=1, prepend=outputs[:, :1, 0]) / DT / biomass
    r_a_flux = (np.diff(outputs[:, :, 2], axis=1, prepend=outputs[:, :1, 2]) / DT + 0.10 * outputs[:, :, 2]) / biomass
    r_b_flux = (np.diff(outputs[:, :, 3], axis=1, prepend=outputs[:, :1, 3]) / DT + 0.10 * outputs[:, :, 3]) / biomass
    X_cap, yg, yp, yra, yrb = [], [], [], [], []
    for i in range(len(train_idx)):
        for t in range(T_STEPS):
            X_cap.append(crossed_cap_features(proxy[i, t], interv[i]))
            yg.append(growth_flux[i, t])
            yp.append(product_flux[i, t])
            yra.append(r_a_flux[i, t])
            yrb.append(r_b_flux[i, t])
    X_cap = np.asarray(X_cap)
    return {
        "dyn": ridge_fit(np.asarray(X_dyn), np.asarray(Y_dyn), lam=1e-4),
        "z0": z0,
        "latent_dim": proxy.shape[2],
        "growth_cap": ridge_fit(X_cap, np.asarray(yg), lam=1e-3),
        "product_cap": ridge_fit(X_cap, np.asarray(yp), lam=1e-3),
        "r_a_cap": ridge_fit(X_cap, np.asarray(yra), lam=1e-3),
        "r_b_cap": ridge_fit(X_cap, np.asarray(yrb), lam=1e-3),
    }


def rollout_crossed_hybrid(model, env, interventions):
    n = env.shape[0]
    z = np.zeros((n, T_STEPS, model["latent_dim"]))
    out = np.zeros((n, T_STEPS, 4))
    z[:, 0] = np.column_stack([np.ones(n), env[:, 0, :]]) @ model["z0"]
    out[:, 0, 1] = env[:, 0, 4]
    for i in range(n):
        for t in range(T_STEPS - 1):
            z[i, t + 1] = crossed_dyn_features(z[i, t], env[i, t]) @ model["dyn"]
            cap_feat = crossed_cap_features(z[i, t], interventions[i])
            growth_cap = float(np.clip(cap_feat @ model["growth_cap"], 0.002, 0.22))
            product_cap = float(np.clip(cap_feat @ model["product_cap"], 0.002, 0.75))
            r_a_cap = float(np.clip(cap_feat @ model["r_a_cap"], 0.002, 0.35))
            r_b_cap = float(np.clip(cap_feat @ model["r_b_cap"], 0.002, 0.35))
            b_state = z[i, t, min(1, model["latent_dim"] - 1)]
            v = crossed_fba(growth_cap, product_cap, r_a_cap, r_b_cap, env[i, t], interventions[i], b_state)
            x = out[i, t, 1]
            out[i, t + 1, 1] = max(0.0, x + DT * v[0] * x)
            out[i, t + 1, 0] = out[i, t, 0] + DT * v[1] * x
            out[i, t + 1, 2] = out[i, t, 2] + DT * (v[2] * x - 0.10 * out[i, t, 2])
            out[i, t + 1, 3] = out[i, t, 3] + DT * (v[3] * x - 0.10 * out[i, t, 3])
    return z, out


def fit_recurrent_reservoir(dataset, train_idx, rng, hidden=96):
    env = dataset["env"][train_idx]
    interv = dataset["intervention"][train_idx]
    outputs = dataset["outputs"][train_idx]
    in_dim = env.shape[2] + interv.shape[1] + outputs.shape[2] + 1
    W_in = rng.normal(0, 0.18, (hidden, in_dim))
    W_h = rng.normal(0, 0.08, (hidden, hidden))
    # Keep recurrent dynamics stable.
    radius = max(abs(np.linalg.eigvals(W_h)).max(), 1e-6)
    W_h *= 0.82 / radius
    states, targets = [], []
    for i in range(len(train_idx)):
        h = np.zeros(hidden)
        for t in range(T_STEPS - 1):
            x = np.r_[env[i, t], interv[i], outputs[i, t], t / (T_STEPS - 1)]
            h = np.tanh(W_in @ x + W_h @ h)
            states.append(np.r_[1.0, h, x])
            targets.append(outputs[i, t + 1])
    readout = ridge_fit(np.asarray(states), np.asarray(targets), lam=1e-2)
    return {"W_in": W_in, "W_h": W_h, "readout": readout, "hidden": hidden}


def rollout_recurrent_reservoir(model, env, interventions):
    n = env.shape[0]
    out = np.zeros((n, T_STEPS, 4))
    out[:, 0, 1] = env[:, 0, 4]
    for i in range(n):
        h = np.zeros(model["hidden"])
        for t in range(T_STEPS - 1):
            x = np.r_[env[i, t], interventions[i], out[i, t], t / (T_STEPS - 1)]
            h = np.tanh(model["W_in"] @ x + model["W_h"] @ h)
            out[i, t + 1] = np.maximum(np.r_[1.0, h, x] @ model["readout"], 0.0)
    return out


def evaluate_crossed_model(world, model_name, pred_out, true_out, pred_z, true_z, meta, idx, parameter_count):
    rows = []
    for split in ["withheld_crossed", "unseen_perturbation", "all_transfer"]:
        if split == "all_transfer":
            local = list(range(len(idx)))
        else:
            local = [k for k, j in enumerate(idx) if meta[j]["split"] == split]
        if not local:
            continue
        pred = pred_out[local]
        truth = true_out[local]
        z_align = ""
        cause_acc = ""
        if pred_z is not None:
            aligns = corr_alignment(pred_z[local].reshape(-1, pred_z.shape[2]), true_z[local].reshape(-1, 2))
            z_align = float(np.mean(aligns))
            true_cause = (true_z[local, :, 0].mean(axis=1) > true_z[local, :, 1].mean(axis=1)).astype(int)
            pred_cause = (pred_z[local, :, 0].mean(axis=1) > pred_z[local, :, min(1, pred_z.shape[2] - 1)].mean(axis=1)).astype(int)
            cause_acc = float((true_cause == pred_cause).mean())
        rows.append(
            {
                "world": world,
                "split": split,
                "model": model_name,
                "product_rmse": rmse(pred[:, :, 0], truth[:, :, 0]),
                "biomass_rmse": rmse(pred[:, :, 1], truth[:, :, 1]),
                "reporter_rmse": rmse(pred[:, :, 2:4], truth[:, :, 2:4]),
                "state_alignment": z_align,
                "cause_discrimination_accuracy": cause_acc,
                "parameter_count": parameter_count,
            }
        )
    return rows


def run_crossed_intervention_transfer(rng):
    metric_rows, split_rows = [], []
    colors = {
        "product-only hybrid": "#64748b",
        "reporter-supervised hybrid": "#16a34a",
        "shuffled-reporter control": "#f97316",
        "oracle true-state hybrid": "#2563eb",
        "recurrent reservoir black box": "#7c3aed",
    }
    figure_rows = []
    for world in ["separable", "entangled"]:
        data = simulate_crossed_dataset(rng, world=world, reps_per_combo=14)
        train_idx = np.array([i for i, m in enumerate(data["meta"]) if m["split"] == "train"], dtype=int)
        test_idx = np.array([i for i, m in enumerate(data["meta"]) if m["split"] != "train"], dtype=int)
        for i, m in enumerate(data["meta"]):
            row = dict(m)
            row["trajectory_index"] = i
            split_rows.append(row)

        train_outputs = data["outputs"][train_idx]
        reporter_proxy = reporter_proxy_from_fluorescence(data["reporters"][train_idx])
        shuffled = data["reporters"][train_idx].copy()
        shuffled = shuffled[rng.permutation(len(shuffled))]
        proxies = {
            "product-only hybrid": product_only_crossed_proxy(train_outputs),
            "reporter-supervised hybrid": reporter_proxy,
            "shuffled-reporter control": reporter_proxy_from_fluorescence(shuffled),
            "oracle true-state hybrid": data["states"][train_idx],
        }
        for model_name, proxy in proxies.items():
            model = fit_crossed_hybrid(data, train_idx, proxy)
            pred_z, pred_out = rollout_crossed_hybrid(model, data["env"][test_idx], data["intervention"][test_idx])
            metric_rows.extend(
                evaluate_crossed_model(
                    world,
                    model_name,
                    pred_out,
                    data["outputs"][test_idx],
                    pred_z,
                    data["states"][test_idx],
                    data["meta"],
                    test_idx,
                    int(model["dyn"].size + model["growth_cap"].size + model["product_cap"].size + model["r_a_cap"].size + model["r_b_cap"].size),
                )
            )
        reservoir = fit_recurrent_reservoir(data, train_idx, rng, hidden=96)
        bb_out = rollout_recurrent_reservoir(reservoir, data["env"][test_idx], data["intervention"][test_idx])
        metric_rows.extend(
            evaluate_crossed_model(
                world,
                "recurrent reservoir black box",
                bb_out,
                data["outputs"][test_idx],
                None,
                data["states"][test_idx],
                data["meta"],
                test_idx,
                int(reservoir["W_in"].size + reservoir["W_h"].size + reservoir["readout"].size),
            )
        )

    metrics = pd.DataFrame(metric_rows)
    split_df = pd.DataFrame(split_rows)
    metrics.to_csv(DATA / "crossed_intervention_transfer_metrics.csv", index=False)
    split_df.to_csv(DATA / "crossed_intervention_split.csv", index=False)

    for world in ["separable", "entangled"]:
        subset = metrics[(metrics["world"] == world) & (metrics["split"] == "all_transfer")]
        for _, row in subset.iterrows():
            figure_rows.append(
                (
                    row["model"].replace(" ", "\n"),
                    float(row["product_rmse"]),
                    colors[row["model"]],
                )
            )
        write_svg_bar(
            FIGURES / f"concrete_05_{world}_crossed_transfer.svg",
            f"Experiment 5: {world} crossed plus unseen transfer",
            figure_rows[-len(subset) :],
            ylabel="transfer product RMSE",
        )
    return metrics, split_df


def write_svg_bar(path, title, rows, ylabel="RMSE"):
    width, height = 920, 470
    margin = 70
    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows], dtype=float)
    colors = [r[2] for r in rows]
    ymax = max(values.max() * 1.25, 1e-6)
    bar_w = (width - 2 * margin) / len(rows) * 0.68
    gap = (width - 2 * margin) / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#222"/>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>',
    ]
    for tick in range(5):
        val = ymax * tick / 4
        y = height - margin - val / ymax * (height - 2 * margin)
        parts.append(f'<line x1="{margin-5}" y1="{y:.1f}" x2="{width-margin}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>')
    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        x = margin + i * gap + (gap - bar_w) / 2
        h = val / ymax * (height - 2 * margin)
        y = height - margin - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val:.3f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-margin+18}" text-anchor="middle" font-family="Arial" font-size="12">{lab}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_svg_line(path, title, series):
    width, height = 920, 470
    margin = 65
    xmax = max(len(v) for _, v, _ in series) - 1
    ymin = min(float(np.min(v)) for _, v, _ in series)
    ymax = max(float(np.max(v)) for _, v, _ in series)
    span = max(ymax - ymin, 1e-6)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#222"/>',
    ]
    for name, vals, color in series:
        pts = []
        for i, val in enumerate(vals):
            x = margin + i / xmax * (width - 2 * margin)
            y = height - margin - (float(val) - ymin) / span * (height - 2 * margin)
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(pts)}"/>')
    for i, (name, _, color) in enumerate(series):
        y = margin + 22 * i
        parts.append(f'<rect x="{width-margin-210}" y="{y-12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width-margin-188}" y="{y}" font-family="Arial" font-size="14">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def markdown_table(df):
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(float(v)) else f"{float(v):.4f}"
        return str(v)

    rows = [[fmt(v) for v in row] for row in df.values.tolist()]
    headers = list(df.columns)
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def make_notebook(metrics_summary, transfer_summary):
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": text.splitlines(True),
            }
        )

    md(
        """# Concrete State-Space Experiment Chain

This notebook follows the proof sequence requested for the proposed model:

1. recover a latent controller in a simple ODE;
2. show reporter supervision improves recovery in a multi-state ODE;
3. repeat the logic inside a tiny dFBA/GSM simulator with product and reporter sinks;
4. benchmark the learned state-space simulator against pure ML rollout models.
5. test crossed physiology-metabolism transfer under withheld interventions.
"""
    )
    md(
        """## Reproducibility

Run the first code cell before any analysis cells. Leave `RERUN = False` to use the checked-in outputs, or set it to `True` to regenerate the tables and figures with the command declared in the cell.
"""
    )
    code(
        """from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src" / "yeast_validation").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

from notebook_repro import ExperimentSpec, configure_plots, load_tables

spec = ExperimentSpec(
    name="Concrete state-space experiment chain",
    command=[".venv/bin/python", "scripts/run_concrete_experiment_chain.py"],
    outputs={
        "metrics": "data/concrete_experiment_chain_metrics.csv",
        "transfer": "data/crossed_intervention_transfer_metrics.csv",
        "split": "data/crossed_intervention_split.csv",
    },
)
RERUN = False
tables = load_tables(spec, rerun=RERUN, root=ROOT)
metrics = tables["metrics"]
transfer = tables["transfer"]
split = tables["split"]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
configure_plots()
metrics, transfer, split.head()
"""
    )
    md("## 1. ODE Latent State Recovery\n\n![ODE latent recovery](../figures/concrete_01_ode_latent_recovery.svg)")
    md("## 2. Biosensors Improve Multi-State ODE Recovery\n\n![ODE biosensor comparison](../figures/concrete_02_biosensor_supervision.svg)")
    md("## 3. dFBA/GSM With Product and Reporter Sinks\n\n![dFBA reporter comparison](../figures/concrete_03_dfba_reporters.svg)")
    md("## 4. Benchmark Against Pure ML Rollout Models\n\n![dFBA benchmark](../figures/concrete_04_dfba_benchmark.svg)")
    md(
        "## 5. Crossed Physiology-Intervention Transfer\n\n"
        "![Separable crossed transfer](../figures/concrete_05_separable_crossed_transfer.svg)\n\n"
        "![Entangled crossed transfer](../figures/concrete_05_entangled_crossed_transfer.svg)"
    )
    md("## Headline Metrics\n\n" + metrics_summary)
    md("## Crossed-Transfer Metrics\n\n" + transfer_summary)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "00_concrete_experiment_chain.ipynb").write_text(json.dumps(nb, indent=2))


def main():
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    NOTEBOOKS.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    metrics = []

    env = design_env(TRAIN_N + TEST_N, rng)
    tr, te = slice(0, TRAIN_N), slice(TRAIN_N, None)
    # Heldout trajectories are still generated by the same equations, but they
    # occupy a harder oxygen/induction regime. This is where state dynamics and
    # reporter-supervised physiology should matter more than one-step curve fit.
    env[te, :, 0] = np.clip(env[te, :, 0] * 1.18, 0, 1.35)
    env[te, :, 2] = np.clip(env[te, :, 2] * 0.82, 0.08, 1.2)
    env[te, :, 3] = np.clip(env[te, :, 3] * 0.58, 0.05, 1.2)
    env[te, :, 4] = np.clip(env[te, :, 4] * 1.18, 0, 1.35)
    env[te, :, 5] = np.clip(env[te, :, 5] * 0.82, 0.08, 1.2)
    env[te, :, 6] = np.clip(env[te, :, 6] * 0.58, 0.05, 1.2)

    # 1. ODE one-state recovery.
    ode1 = simulate_ode_one(env, rng)
    z_proxy = infer_ode1_state(ode1["product"][tr])
    fit1 = fit_state_space(env[tr], ode1["product"][tr], z_proxy)
    z_pred1, y_pred1 = rollout_state_space(fit1, env[te], 1)
    align1 = corr_alignment(z_pred1.reshape(-1, 1), ode1["z"][te].reshape(-1, 1))[0]
    metrics.append(
        {
            "stage": "1_ode_latent_recovery",
            "model": "state-space latent ODE",
            "product_rmse": rmse(y_pred1[:, :, 0], ode1["product"][te]),
            "state_alignment": align1,
            "biomass_rmse": "",
            "reporter_rmse": "",
            "notes": "One learned latent controller from environment to product curve.",
        }
    )
    write_svg_line(
        FIGURES / "concrete_01_ode_latent_recovery.svg",
        "Experiment 1: learned latent state follows hidden controller",
        [
            ("true z", ode1["z"][te][0], "#2563eb"),
            ("learned z", z_pred1[0, :, 0], "#dc2626"),
            ("product", ode1["product"][te][0] / max(ode1["product"][te][0].max(), 1e-9), "#16a34a"),
        ],
    )

    # 2. ODE multi-state with biosensors.
    ode3 = simulate_ode_three(env, rng)
    model_defs = [
        ("product only", None),
        ("product + 1 reporter", 1),
        ("product + 2 reporters", 2),
    ]
    bars = []
    colors = ["#64748b", "#0ea5e9", "#16a34a"]
    for idx, (name, reps) in enumerate(model_defs):
        rep_train = None if reps is None else ode3["reporters"][tr, :, :reps]
        proxy = build_ode3_proxy(ode3["product"][tr], rep_train)
        outputs_train = ode3["product"][tr]
        if reps:
            outputs_train = np.dstack([ode3["product"][tr], ode3["reporters"][tr, :, :reps]])
        fit = fit_state_space(env[tr], outputs_train, proxy)
        z_pred, y_pred = rollout_state_space(fit, env[te], output_dim=1 + (reps or 0))
        aligns = corr_alignment(z_pred.reshape(-1, z_pred.shape[2]), ode3["z"][te].reshape(-1, 3))
        prod_rmse = rmse(y_pred[:, :, 0], ode3["product"][te])
        reporter_rmse = ""
        if reps:
            reporter_rmse = rmse(y_pred[:, :, 1 : 1 + reps], ode3["reporters"][te, :, :reps])
        metrics.append(
            {
                "stage": "2_ode_biosensor_supervision",
                "model": name,
                "product_rmse": prod_rmse,
                "state_alignment": float(np.mean(aligns)),
                "biomass_rmse": "",
                "reporter_rmse": reporter_rmse,
                "notes": f"Mean alignment across 3 true latent controllers; reporter_count={reps or 0}.",
            }
        )
        bars.append((name.replace(" ", "\n"), float(np.mean(aligns)), colors[idx]))
    write_svg_bar(
        FIGURES / "concrete_02_biosensor_supervision.svg",
        "Experiment 2: reporters improve latent-state recovery",
        bars,
        ylabel="mean state alignment",
    )

    # 3. dFBA/GSM state-space models with product/reporter sinks.
    dfba = simulate_dfba(env, rng)
    dfba_defs = [
        ("dFBA state-space product only", 0),
        ("dFBA state-space + 1 reporter sink", 1),
        ("dFBA state-space + 2 reporter sinks", 2),
    ]
    bars = []
    for idx, (name, reps) in enumerate(dfba_defs):
        proxy = infer_dfba_proxy(dfba["outputs"][tr], reps)
        model = fit_dfba_state_space(env[tr], dfba["outputs"][tr], proxy)
        z_pred, out_pred = rollout_dfba_model(model, env[te])
        aligns = corr_alignment(z_pred.reshape(-1, z_pred.shape[2]), dfba["z"][te].reshape(-1, 3))
        metrics.append(
            {
                "stage": "3_dfba_gsm_reporter_sinks",
                "model": name,
                "product_rmse": rmse(out_pred[:, :, 0], dfba["outputs"][te, :, 0]),
                "state_alignment": float(np.mean(aligns)),
                "biomass_rmse": rmse(out_pred[:, :, 1], dfba["outputs"][te, :, 1]),
                "reporter_rmse": "" if reps == 0 else rmse(out_pred[:, :, 2 : 2 + reps], dfba["outputs"][te, :, 2 : 2 + reps]),
                "notes": f"Tiny FBA allocates biomass, beta-carotene sink, and {reps} reporter sinks.",
            }
        )
        bars.append((name.replace("dFBA state-space ", "").replace(" ", "\n"), rmse(out_pred[:, :, 0], dfba["outputs"][te, :, 0]), colors[idx]))
    write_svg_bar(
        FIGURES / "concrete_03_dfba_reporters.svg",
        "Experiment 3: reporter sinks improve dFBA product rollout",
        bars,
        ylabel="product RMSE",
    )

    # 4. Pure ML baselines on dFBA-generated curves.
    X_train, Y_train = baseline_features(env[tr], dfba["outputs"][tr])
    ridge = ridge_fit(X_train, Y_train, lam=1e-3)
    tree = SimpleTree(depth=6, min_leaf=24).fit(X_train, Y_train)
    mlp = fit_mlp(X_train, Y_train, rng)
    # Use the strongest dFBA state-space model from stage 3.
    best_dfba = fit_dfba_state_space(env[tr], dfba["outputs"][tr], infer_dfba_proxy(dfba["outputs"][tr], 2))
    _, hybrid_out = rollout_dfba_model(best_dfba, env[te])
    baseline_rows = [
        ("hybrid state-space dFBA-GSM", hybrid_out),
        ("lagged ridge ML", rollout_ml_model("ridge", ridge, env[te])),
        ("shallow tree ML", rollout_ml_model("tree", tree, env[te])),
        ("one-hidden-layer MLP", rollout_ml_model("mlp", mlp, env[te])),
    ]
    bars = []
    base_colors = ["#16a34a", "#64748b", "#f97316", "#7c3aed"]
    for idx, (name, pred) in enumerate(baseline_rows):
        metrics.append(
            {
                "stage": "4_dfba_vs_pure_ml",
                "model": name,
                "product_rmse": rmse(pred[:, :, 0], dfba["outputs"][te, :, 0]),
                "state_alignment": "" if name != "hybrid state-space dFBA-GSM" else metrics[-1]["state_alignment"],
                "biomass_rmse": rmse(pred[:, :, 1], dfba["outputs"][te, :, 1]),
                "reporter_rmse": rmse(pred[:, :, 2:4], dfba["outputs"][te, :, 2:4]),
                "notes": "Pure ML models receive environment plus previous outputs, not explicit internal states.",
            }
        )
        bars.append((name.replace(" ", "\n"), rmse(pred[:, :, 0], dfba["outputs"][te, :, 0]), base_colors[idx]))
    write_svg_bar(
        FIGURES / "concrete_04_dfba_benchmark.svg",
        "Experiment 4: dFBA state-space model vs pure ML rollouts",
        bars,
        ylabel="product RMSE",
    )

    transfer_metrics, _ = run_crossed_intervention_transfer(rng)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(DATA / "concrete_experiment_chain_metrics.csv", index=False)

    pred_rows = []
    for t in range(T_STEPS):
        pred_rows.append(
            {
                "time_h": t * DT,
                "true_product": dfba["outputs"][te][0, t, 0],
                "hybrid_product": hybrid_out[0, t, 0],
                "true_biomass": dfba["outputs"][te][0, t, 1],
                "hybrid_biomass": hybrid_out[0, t, 1],
                "true_reporter_1": dfba["outputs"][te][0, t, 2],
                "hybrid_reporter_1": hybrid_out[0, t, 2],
                "true_reporter_2": dfba["outputs"][te][0, t, 3],
                "hybrid_reporter_2": hybrid_out[0, t, 3],
            }
        )
    pd.DataFrame(pred_rows).to_csv(DATA / "concrete_dfba_example_rollout.csv", index=False)

    summary = markdown_table(metrics_df)
    transfer_summary = markdown_table(transfer_metrics)
    make_notebook(summary, transfer_summary)
    print(metrics_df.to_string(index=False))
    print(transfer_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
