#!/usr/bin/env python3
"""Bridge diagnostics for reporter supervision.

This runner diagnoses where reporter supervision stops helping as the concrete
chain moves from ODE dynamics to crossed dFBA transfer. It intentionally reuses
the lightweight state-space/decoder style in run_concrete_experiment_chain.py
instead of introducing a separate modeling framework.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
NOTEBOOKS = ROOT / "notebooks"
CHAIN_PATH = ROOT / "src" / "yeast_validation" / "run_concrete_experiment_chain.py"

spec = importlib.util.spec_from_file_location("concrete_chain", CHAIN_PATH)
chain = importlib.util.module_from_spec(spec)
sys.modules["concrete_chain"] = chain
spec.loader.exec_module(chain)

DT = chain.DT
T_STEPS = chain.T_STEPS
LAMBDA_SWEEP = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
REPAIRED_LAMBDAS = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
REPAIRED_DECODER_TRUE = {
    "theta0": 1.15,
    "beta_A": 1.28,
    "beta_B": 0.92,
    "delta_E": 0.30,
    "q_product": 0.96,
    "delta_P": 0.035,
    "mu": 0.080,
    "gamma0": 0.95,
    "gamma_B": 1.18,
}
STAGE_E_LAMBDAS = [0, 0.03, 0.1, 0.3, 1.0, 3.0]
STAGE_E_INTERVENTIONS = {
    "base_pathway": {"product_scale": 1.00, "atp_cost": 1.00, "maintenance": 0.018, "precursor": 1.00},
    "high_capacity": {"product_scale": 1.35, "atp_cost": 1.04, "maintenance": 0.020, "precursor": 1.05},
    "high_atp_cost": {"product_scale": 1.05, "atp_cost": 1.45, "maintenance": 0.026, "precursor": 1.05},
    "low_pathway": {"product_scale": 0.68, "atp_cost": 0.95, "maintenance": 0.017, "precursor": 0.92},
}


def rmse(a, b):
    return chain.rmse(a, b)


def sigmoid(x):
    return chain.sigmoid(x)


def ridge_fit(X, Y, lam=1e-5):
    return chain.ridge_fit(X, Y, lam=lam)


def env_schedule(regime, rep, rng, unseen=False):
    base = chain.make_crossed_env(regime, rep, rng)
    if unseen:
        base = base.copy()
        pulse = base[:, 0] > 0.15
        shifted = np.zeros_like(base[:, 0])
        shifted[6:] = base[:-6, 0]
        base[:, 0] = np.maximum(base[:, 0], 0.75 * shifted)
        base[:, 2] = np.clip(base[:, 2] * 0.82, 0.1, 1.2)
        base[:, 3] = np.clip(base[:, 3] * 0.70, 0.05, 1.1)
        base[:, 6] *= 1.35
    return base


def evolve_states(env, rng, entangled=False, intervention=None):
    z = np.zeros((T_STEPS, 2))
    flux_memory = 0.0
    intervention = np.ones(4) if intervention is None else np.asarray(intervention)
    for t in range(T_STEPS - 1):
        light, temp, feed, oxygen = env[t, :4]
        oxygen_deficit = max(0.0, 0.55 - oxygen)
        drive = np.array(
            [
                0.88 * temp + 0.52 * light - 0.24 * feed + 0.08,
                1.06 * oxygen_deficit + 0.54 * (0.72 - feed) + 0.22 * light - 0.08,
            ]
        )
        if entangled:
            drive += np.array([0.36 * (intervention[0] - 1.0), 0.30 * (intervention[2] - 1.0)])
            drive += 0.12 * flux_memory
        z[t + 1] = 0.84 * z[t] + DT * (0.48 * drive - 0.20 * z[t])
        flux_memory = 0.8 * flux_memory + 0.2 * sigmoid(z[t, 0] - z[t, 1])
    return z


def reporter_from_states(z, rng, noise=True):
    mrna = np.zeros((T_STEPS, 2))
    fluor = np.zeros((T_STEPS, 2))
    for t in range(T_STEPS - 1):
        drive = np.array([sigmoid(1.25 * z[t, 0]), sigmoid(1.20 * z[t, 1])])
        mrna[t + 1] = mrna[t] + DT * (1.25 * drive - 0.42 * mrna[t])
        fluor[t + 1, 0] = fluor[t, 0] + DT * (0.62 * mrna[t, 0] - 0.10 * fluor[t, 0])
        fluor[t + 1, 1] = fluor[t, 1] + DT * (0.58 * mrna[t, 1] - 0.10 * fluor[t, 1])
    if noise:
        fluor = np.maximum(0, fluor + rng.normal(0, 0.003, fluor.shape))
    return fluor


def decode_ode(env, z, rng, noise=True):
    expr = np.zeros(T_STEPS)
    product = np.zeros(T_STEPS)
    biomass = np.zeros(T_STEPS)
    biomass[0] = env[0, 4]
    for t in range(T_STEPS - 1):
        light = env[t, 0]
        expr[t + 1] = expr[t] + DT * (light * sigmoid(1.15 - 1.20 * z[t, 0] - 0.85 * z[t, 1]) - 0.28 * expr[t])
        growth = 0.075 * sigmoid(1.0 - z[t, 1])
        biomass[t + 1] = max(0, biomass[t] + DT * growth * biomass[t])
        product[t + 1] = product[t] + DT * (0.92 * expr[t] * biomass[t] - 0.035 * product[t])
    out = np.column_stack([product, biomass, np.zeros((T_STEPS, 2))])
    if noise:
        out[:, :2] = np.maximum(0, out[:, :2] + rng.normal(0, [0.003, 0.0015], (T_STEPS, 2)))
    return out


def decode_algebraic(env, z, intervention, rng, noise=True):
    product = np.zeros(T_STEPS)
    biomass = np.zeros(T_STEPS)
    biomass[0] = env[0, 4]
    product_scale, growth_scale, atp_cost, maintenance = intervention
    for t in range(T_STEPS - 1):
        cap = product_scale * sigmoid(1.2 - 1.25 * z[t, 0])
        growth = max(0, 0.10 * growth_scale * sigmoid(1.0 - 1.15 * z[t, 1]) - maintenance)
        product[t + 1] = product[t] + DT * (0.42 * cap * biomass[t] / atp_cost)
        biomass[t + 1] = max(0, biomass[t] + DT * growth * biomass[t])
    out = np.column_stack([product, biomass, np.zeros((T_STEPS, 2))])
    if noise:
        out[:, :2] = np.maximum(0, out[:, :2] + rng.normal(0, [0.003, 0.0015], (T_STEPS, 2)))
    return out


def decode_fba(env, z, intervention, rng, noise=True, feedback=False):
    out = np.zeros((T_STEPS, 4))
    flux = np.zeros((T_STEPS, 4))
    out[0, 1] = env[0, 4]
    for t in range(T_STEPS - 1):
        growth_cap = 0.16 * sigmoid(1.15 - 1.25 * z[t, 1])
        product_cap = 0.12 + 0.42 * sigmoid(1.25 - 1.55 * z[t, 0])
        rep_a_cap = 0.055 + 0.15 * sigmoid(1.8 * z[t, 0])
        rep_b_cap = 0.055 + 0.15 * sigmoid(1.8 * z[t, 1])
        v = chain.crossed_fba(growth_cap, product_cap, rep_a_cap, rep_b_cap, env[t], intervention, z[t, 1])
        flux[t] = v
        x = out[t, 1]
        out[t + 1, 1] = max(0.0, x + DT * v[0] * x)
        out[t + 1, 0] = out[t, 0] + DT * v[1] * x
        out[t + 1, 2] = out[t, 2] + DT * (v[2] * x - 0.10 * out[t, 2])
        out[t + 1, 3] = out[t, 3] + DT * (v[3] * x - 0.10 * out[t, 3])
    if noise:
        out += rng.normal(0, [0.003, 0.0015, 0.003, 0.003], out.shape)
        out = np.maximum(out, 0)
    return out, flux


def make_dataset(stage, seed, mode="fast", world="separable"):
    rng = np.random.default_rng(seed)
    reps = 8 if mode == "fast" else 14
    regimes = ["A_hot_induction", "B_low_oxygen", "balanced"]
    interventions = ["base"]
    split_plan = []
    decoder = "ode"
    entangled = world == "entangled"
    if stage == "A":
        split_plan = [(r, "base", rep, "train" if rep < 5 else "test") for r in regimes for rep in range(reps)]
        decoder = "ode"
    elif stage == "B":
        split_plan = [(r, "base", rep, "train" if rep < 5 else "test") for r in regimes for rep in range(reps)]
        split_plan += [("unseen_long_double", "base", rep, "test") for rep in range(3 if mode == "fast" else 8)]
        decoder = "ode"
    elif stage == "C":
        interventions = ["base", "high_capacity", "high_atp_cost"]
        split_plan = [(r, itv, rep, "train" if rep < 5 else "test") for r in regimes for itv in interventions for rep in range(reps)]
        decoder = "algebraic"
    elif stage == "D":
        split_plan = [(r, "base", rep, "train" if rep < 5 else "test") for r in regimes for rep in range(reps)]
        decoder = "fba"
    elif stage == "E":
        interventions = ["base", "high_capacity", "high_atp_cost", "low_pathway"]
        split_plan = [(r, itv, rep, "train" if rep < 5 else ("validation" if rep < 6 else "test")) for r in regimes for itv in interventions for rep in range(reps)]
        decoder = "fba"
    elif stage == "F":
        decoder = "fba"
        combos_train = chain.TRAIN_COMBOS
        combos_crossed = chain.CROSSED_TEST_COMBOS
        combos_unseen = chain.UNSEEN_TEST_COMBOS
        all_combos = sorted(combos_train | combos_crossed | combos_unseen)
        for r, itv in all_combos:
            split = "train" if (r, itv) in combos_train else "test"
            for rep in range(reps):
                split_plan.append((r, itv, rep, split))
    else:
        raise ValueError(stage)

    rows, envs, states, outputs, noiseless, reporters, interventions_arr = [], [], [], [], [], [], []
    for idx, (regime, intervention_name, rep, split) in enumerate(split_plan):
        unseen = regime == "unseen_long_double"
        env = env_schedule(regime, rep, rng, unseen=unseen)
        intervention = chain.intervention_vector(intervention_name)
        z = evolve_states(env, rng, entangled=entangled, intervention=intervention)
        rep_f = reporter_from_states(z, rng, noise=True)
        if decoder == "ode":
            out_clean = decode_ode(env, z, rng, noise=False)
            out = decode_ode(env, z, rng, noise=True)
        elif decoder == "algebraic":
            out_clean = decode_algebraic(env, z, intervention, rng, noise=False)
            out = decode_algebraic(env, z, intervention, rng, noise=True)
        else:
            out_clean, _ = decode_fba(env, z, intervention, rng, noise=False)
            out, _ = decode_fba(env, z, intervention, rng, noise=True)
        out[:, 2:4] = rep_f
        out_clean[:, 2:4] = reporter_from_states(z, rng, noise=False)
        envs.append(env)
        states.append(z)
        outputs.append(out)
        noiseless.append(out_clean)
        reporters.append(rep_f)
        interventions_arr.append(intervention)
        rows.append(
            {
                "trajectory_id": idx,
                "stage": stage,
                "world": world,
                "split": split,
                "physiology_regime": regime,
                "intervention": intervention_name,
                "replicate": rep,
                "decoder": decoder,
            }
        )
    return {
        "meta": rows,
        "env": np.asarray(envs),
        "states": np.asarray(states),
        "outputs": np.asarray(outputs),
        "noiseless": np.asarray(noiseless),
        "reporters": np.asarray(reporters),
        "intervention": np.asarray(interventions_arr),
        "decoder": decoder,
        "world": world,
    }


def product_proxy(outputs):
    product, biomass = outputs[:, :, 0], outputs[:, :, 1]
    x = np.maximum(biomass, 0.035)
    prod_flux = np.diff(product, axis=1, prepend=product[:, :1]) / DT / x
    bio_flux = np.diff(biomass, axis=1, prepend=biomass[:, :1]) / DT / x
    cols = []
    for arr in [prod_flux, -bio_flux]:
        cols.append((arr - arr.mean()) / (arr.std() + 1e-9))
    return np.stack(cols, axis=2)


def reporter_proxy(reporters, head="state_constrained", env=None):
    if head == "direct_env":
        if env is None:
            raise ValueError("direct_env head needs env")
        z_a = env[:, :, 1]
        z_b = 1 - env[:, :, 3]
        return np.stack([(z_a - z_a.mean()) / (z_a.std() + 1e-9), (z_b - z_b.mean()) / (z_b.std() + 1e-9)], axis=2)
    proxy = chain.reporter_proxy_from_fluorescence(reporters)
    if head == "shared_hidden":
        mix0 = 0.75 * proxy[:, :, 0] + 0.25 * proxy[:, :, 1]
        mix1 = 0.25 * proxy[:, :, 0] + 0.75 * proxy[:, :, 1]
        return np.stack([mix0, mix1], axis=2)
    return proxy


def blend_proxy(outputs, reporters, lam, head="state_constrained", env=None):
    pp = product_proxy(outputs)
    if lam <= 0:
        return pp
    rp = reporter_proxy(reporters, head=head, env=env)
    alpha = lam / (1.0 + lam)
    return (1 - alpha) * pp + alpha * rp


def fit_bridge_model(dataset, train_idx, proxy):
    env = dataset["env"][train_idx]
    outputs = dataset["outputs"][train_idx]
    intervention = dataset["intervention"][train_idx]
    X_dyn, Y_dyn = [], []
    for i in range(len(train_idx)):
        for t in range(T_STEPS - 1):
            X_dyn.append(chain.crossed_dyn_features(proxy[i, t], env[i, t]))
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
            X_cap.append(chain.crossed_cap_features(proxy[i, t], intervention[i]))
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


def rollout_bridge(model, env, intervention, decoder):
    if decoder in {"fba", "algebraic"}:
        return chain.rollout_crossed_hybrid(model, env, intervention)
    n = env.shape[0]
    z = np.zeros((n, T_STEPS, model["latent_dim"]))
    out = np.zeros((n, T_STEPS, 4))
    z[:, 0] = np.column_stack([np.ones(n), env[:, 0, :]]) @ model["z0"]
    out[:, 0, 1] = env[:, 0, 4]
    for i in range(n):
        expr = 0.0
        for t in range(T_STEPS - 1):
            z[i, t + 1] = chain.crossed_dyn_features(z[i, t], env[i, t]) @ model["dyn"]
            cap_feat = chain.crossed_cap_features(z[i, t], intervention[i])
            product_cap = float(np.clip(cap_feat @ model["product_cap"], 0.002, 0.75))
            growth_cap = float(np.clip(cap_feat @ model["growth_cap"], 0.002, 0.22))
            expr += DT * (env[i, t, 0] * product_cap - 0.28 * expr)
            x = out[i, t, 1]
            out[i, t + 1, 1] = max(0, x + DT * growth_cap * x)
            out[i, t + 1, 0] = out[i, t, 0] + DT * expr * x
            out[i, t + 1, 2] = out[i, t, 2] + DT * (0.05 * x - 0.10 * out[i, t, 2])
            out[i, t + 1, 3] = out[i, t, 3] + DT * (0.05 * x - 0.10 * out[i, t, 3])
    return z, out


def affine_rmse(pred, truth):
    X = np.column_stack([np.ones(pred.size), pred.ravel()])
    y = truth.ravel()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return rmse(X @ coef, y)


def timing_metrics(pred, truth):
    p = pred.mean(axis=0)
    t = truth.mean(axis=0)
    return {
        "onset_error_h": abs(np.argmax(p > np.percentile(p, 65)) - np.argmax(t > np.percentile(t, 65))) * DT,
        "peak_error_h": abs(np.argmax(p) - np.argmax(t)) * DT,
        "recovery_error_h": abs(np.argmin(abs(p - p[-1])) - np.argmin(abs(t - t[-1]))) * DT,
        "amplitude_error": abs((p.max() - p.min()) - (t.max() - t.min())),
    }


def evaluate(dataset, test_idx, model_name, pred_z, pred_out, parameter_count, lam=np.nan, head=""):
    truth = dataset["outputs"][test_idx]
    true_z = dataset["states"][test_idx]
    align = chain.corr_alignment(pred_z.reshape(-1, pred_z.shape[2]), true_z.reshape(-1, 2)) if pred_z is not None else [np.nan, np.nan]
    state_a_rmse = affine_rmse(pred_z[:, :, 0], true_z[:, :, 0]) if pred_z is not None else np.nan
    state_b_rmse = affine_rmse(pred_z[:, :, min(1, pred_z.shape[2] - 1)], true_z[:, :, 1]) if pred_z is not None else np.nan
    cause_true = (true_z[:, :, 0].mean(axis=1) > true_z[:, :, 1].mean(axis=1)).astype(int)
    if pred_z is not None:
        cause_pred = (pred_z[:, :, 0].mean(axis=1) > pred_z[:, :, min(1, pred_z.shape[2] - 1)].mean(axis=1)).astype(int)
        cause_acc = float((cause_true == cause_pred).mean())
    else:
        cause_acc = np.nan
    tm = timing_metrics(pred_z[:, :, 0], true_z[:, :, 0]) if pred_z is not None else {}
    product_sd = float(np.std(truth[:, :, 0]) + 1e-9)
    return {
        "stage": dataset["stage"],
        "world": dataset["world"],
        "model": model_name,
        "lambda_reporter": lam,
        "reporter_head": head,
        "product_rmse": rmse(pred_out[:, :, 0], truth[:, :, 0]),
        "product_nrmse": rmse(pred_out[:, :, 0], truth[:, :, 0]) / product_sd,
        "biomass_rmse": rmse(pred_out[:, :, 1], truth[:, :, 1]),
        "reporter_rmse": rmse(pred_out[:, :, 2:4], truth[:, :, 2:4]),
        "state_A_alignment": align[0],
        "state_B_alignment": align[1],
        "state_A_affine_rmse": state_a_rmse,
        "state_B_affine_rmse": state_b_rmse,
        "cause_accuracy": cause_acc,
        "parameter_count": parameter_count,
        **tm,
    }


def run_variant(dataset, train_idx, test_idx, model_name, lam, head, rng):
    if model_name == "product_only":
        proxy = blend_proxy(dataset["outputs"][train_idx], dataset["reporters"][train_idx], 0, head=head, env=dataset["env"][train_idx])
    elif model_name == "reporter":
        proxy = blend_proxy(dataset["outputs"][train_idx], dataset["reporters"][train_idx], lam, head=head, env=dataset["env"][train_idx])
    elif model_name == "shuffled":
        shuffled_idx = rng.permutation(len(train_idx))
        attempts = 0
        while np.any(shuffled_idx == np.arange(len(train_idx))) and attempts < 20:
            shuffled_idx = rng.permutation(len(train_idx))
            attempts += 1
        if np.any(shuffled_idx == np.arange(len(train_idx))) and len(train_idx) > 1:
            shuffled_idx = np.roll(np.arange(len(train_idx)), 1)
        assert not np.any(shuffled_idx == np.arange(len(train_idx)))
        proxy = blend_proxy(dataset["outputs"][train_idx], dataset["reporters"][train_idx][shuffled_idx], lam, head=head, env=dataset["env"][train_idx])
    elif model_name == "oracle_learned":
        proxy = dataset["states"][train_idx]
    else:
        raise ValueError(model_name)
    model = fit_bridge_model(dataset, train_idx, proxy)
    pred_z, pred_out = rollout_bridge(model, dataset["env"][test_idx], dataset["intervention"][test_idx], dataset["decoder"])
    count = int(model["dyn"].size + model["growth_cap"].size + model["product_cap"].size + model["r_a_cap"].size + model["r_b_cap"].size)
    return evaluate(dataset, test_idx, model_name, pred_z, pred_out, count, lam=lam, head=head)


def oracle_true_decoder(dataset, test_idx):
    pred = dataset["noiseless"][test_idx]
    z = dataset["states"][test_idx]
    row = evaluate(dataset, test_idx, "oracle_true_decoder", z, pred, 0)
    if row["product_rmse"] > 0.02 or row["biomass_rmse"] > 0.02:
        raise RuntimeError(f"Oracle true decoder failed noise-floor check: {row}")
    return row


def target_audit(dataset, train_idx):
    rows = []
    names = ["product", "biomass", "reporter_A", "reporter_B"]
    Y = dataset["outputs"][train_idx]
    for j, name in enumerate(names):
        raw = Y[:, :, j].ravel()
        norm = (raw - raw.mean()) / (raw.std() + 1e-9)
        rows.append(
            {
                "stage": dataset["stage"],
                "target": name,
                "raw_min": raw.min(),
                "raw_max": raw.max(),
                "raw_variance": raw.var(),
                "normalized_min": norm.min(),
                "normalized_max": norm.max(),
                "normalized_variance": norm.var(),
                "train_only_statistics": True,
            }
        )
    return rows


def loss_audit(dataset, train_idx, lam):
    y = dataset["outputs"][train_idx]
    prod_var = float(np.var(y[:, :, 0]) + np.var(y[:, :, 1]))
    rep_var = float(np.var(y[:, :, 2]) + np.var(y[:, :, 3]))
    total = prod_var + lam * rep_var
    return {
        "stage": dataset["stage"],
        "lambda_reporter": lam,
        "product_loss_proxy": prod_var,
        "reporter_loss_proxy": rep_var,
        "weighted_product_fraction": prod_var / total if total else np.nan,
        "weighted_reporter_fraction": lam * rep_var / total if total else np.nan,
        "product_gradient_norm_proxy": math.sqrt(prod_var),
        "reporter_gradient_norm_proxy": lam * math.sqrt(rep_var),
    }


def cause_leakage(dataset, train_idx, test_idx):
    true_z = dataset["states"]
    y = (true_z[:, :, 0].mean(axis=1) > true_z[:, :, 1].mean(axis=1)).astype(int)
    feats = {
        "environment_only": dataset["env"].mean(axis=1),
        "intervention_only": dataset["intervention"],
        "product_biomass_only": dataset["outputs"][:, :, :2].mean(axis=1),
        "true_state": dataset["states"].mean(axis=1),
    }
    rows = []
    for name, X in feats.items():
        Xtr = np.column_stack([np.ones(len(train_idx)), X[train_idx]])
        ytr = y[train_idx]
        coef = ridge_fit(Xtr, ytr, lam=1e-3)
        score = np.column_stack([np.ones(len(test_idx)), X[test_idx]]) @ coef
        pred = (score > 0.5).astype(int)
        acc = float((pred == y[test_idx]).mean())
        classes = sorted(set(y[test_idx]))
        recalls = []
        for c in classes:
            mask = y[test_idx] == c
            recalls.append(float((pred[mask] == c).mean()) if mask.any() else np.nan)
        rows.append(
            {
                "stage": dataset["stage"],
                "feature_set": name,
                "accuracy": acc,
                "balanced_accuracy": float(np.nanmean(recalls)),
                "macro_f1": acc,
                "class_0_count": int((y[test_idx] == 0).sum()),
                "class_1_count": int((y[test_idx] == 1).sum()),
                "confusion_00": int(((pred == 0) & (y[test_idx] == 0)).sum()),
                "confusion_01": int(((pred == 1) & (y[test_idx] == 0)).sum()),
                "confusion_10": int(((pred == 0) & (y[test_idx] == 1)).sum()),
                "confusion_11": int(((pred == 1) & (y[test_idx] == 1)).sum()),
            }
        )
    return rows


def modularity_audit(seed=900):
    rng = np.random.default_rng(seed)
    rows = []
    env = env_schedule("A_hot_induction", 0, rng)
    for world in ["separable", "entangled"]:
        z_by = []
        out_by = []
        for name in ["base", "high_atp_cost"]:
            intervention = chain.intervention_vector(name)
            z = evolve_states(env, rng, entangled=(world == "entangled"), intervention=intervention)
            out, _ = decode_fba(env, z, intervention, rng, noise=False)
            z_by.append(z)
            out_by.append(out)
        rows.append(
            {
                "world": world,
                "environment_identifier": "A_hot_induction_rep0",
                "intervention_pair": "base_vs_high_atp_cost",
                "state_A_trajectory_difference": rmse(z_by[0][:, 0], z_by[1][:, 0]),
                "state_B_trajectory_difference": rmse(z_by[0][:, 1], z_by[1][:, 1]),
                "product_difference": rmse(out_by[0][:, 0], out_by[1][:, 0]),
                "biomass_difference": rmse(out_by[0][:, 1], out_by[1][:, 1]),
            }
        )
    sep = rows[0]
    if sep["state_A_trajectory_difference"] > 1e-10 or sep["state_B_trajectory_difference"] > 1e-10:
        raise RuntimeError("Separable modularity check failed")
    if rows[1]["state_A_trajectory_difference"] <= 1e-4 and rows[1]["state_B_trajectory_difference"] <= 1e-4:
        raise RuntimeError("Entangled modularity check failed")
    return rows


def write_svg_scatter(path, title, rows, xkey, ykey, labelkey):
    width, height, margin = 920, 500, 70
    xs = np.array([float(r[xkey]) for r in rows])
    ys = np.array([float(r[ykey]) for r in rows])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="23">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="14">{xkey}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Arial" font-size="14">{ykey}</text>',
    ]
    for r in rows:
        x = margin + (float(r[xkey]) - xmin) / (xmax - xmin) * (width - 2 * margin)
        y = height - margin - (float(r[ykey]) - ymin) / (ymax - ymin) * (height - 2 * margin)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#2563eb"/>')
        parts.append(f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-family="Arial" font-size="12">{r[labelkey]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_svg_bars(path, title, rows, key="product_rmse"):
    chain.write_svg_bar(path, title, [(r["label"], float(r[key]), r.get("color", "#2563eb")) for r in rows], ylabel=key)


def run_stage(stage, seed, mode, world="separable", lambda_reporter=1.0, head="state_constrained"):
    dataset = make_dataset(stage, seed, mode=mode, world=world)
    dataset["stage"] = stage
    idx_train = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "train"], dtype=int)
    idx_test = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
    rng = np.random.default_rng(seed + 777)
    rows = [
        run_variant(dataset, idx_train, idx_test, "product_only", 0, head, rng),
        run_variant(dataset, idx_train, idx_test, "reporter", lambda_reporter, head, rng),
        run_variant(dataset, idx_train, idx_test, "shuffled", lambda_reporter, head, rng),
        run_variant(dataset, idx_train, idx_test, "oracle_learned", lambda_reporter, head, rng),
        oracle_true_decoder(dataset, idx_test),
    ]
    if stage in {"E", "F"}:
        reservoir = chain.fit_recurrent_reservoir(dataset, idx_train, rng, hidden=64)
        pred = chain.rollout_recurrent_reservoir(reservoir, dataset["env"][idx_test], dataset["intervention"][idx_test])
        rows.append(evaluate(dataset, idx_test, "recurrent_reservoir", None, pred, int(reservoir["W_in"].size + reservoir["W_h"].size + reservoir["readout"].size)))
    return dataset, idx_train, idx_test, rows


def run_stage_rows(dataset, train_idx, test_idx, seed, lambda_reporter=1.0, head="state_constrained"):
    rng = np.random.default_rng(seed + 777)
    rows = [
        run_variant(dataset, train_idx, test_idx, "product_only", 0, head, rng),
        run_variant(dataset, train_idx, test_idx, "reporter", lambda_reporter, head, rng),
        run_variant(dataset, train_idx, test_idx, "shuffled", lambda_reporter, head, rng),
        run_variant(dataset, train_idx, test_idx, "oracle_learned", lambda_reporter, head, rng),
        oracle_true_decoder(dataset, test_idx),
    ]
    if dataset["stage"] in {"E", "F"}:
        reservoir = chain.fit_recurrent_reservoir(dataset, train_idx, rng, hidden=48)
        pred = chain.rollout_recurrent_reservoir(reservoir, dataset["env"][test_idx], dataset["intervention"][test_idx])
        rows.append(evaluate(dataset, test_idx, "recurrent_reservoir", None, pred, int(reservoir["W_in"].size + reservoir["W_h"].size + reservoir["readout"].size)))
    return rows


def run_diagnostics(mode):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    NOTEBOOKS.mkdir(exist_ok=True)
    seeds = [11, 23] if mode == "fast" else [11, 23, 37, 51]
    lambdas = [0, 0.1, 1.0, 3.0] if mode == "fast" else LAMBDA_SWEEP
    stage_rows, seed_rows, audit_rows, loss_rows, target_rows, cause_rows = [], [], [], [], [], []
    stages = list("ABCDEF")

    for seed in seeds:
        for stage in stages:
            dataset = make_dataset(stage, seed, mode)
            dataset["stage"] = stage
            train_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "train"], dtype=int)
            test_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
            rows = run_stage_rows(dataset, train_idx, test_idx, seed)
            seed_rows.extend([{**r, "seed": seed} for r in rows])
            for lam in lambdas if stage in {"A", "D", "E", "F"} else [1.0]:
                sweep_rows = [run_variant(dataset, train_idx, test_idx, "reporter", lam, "state_constrained", np.random.default_rng(seed + 991))]
                loss_rows.append(loss_audit(dataset, train_idx, lam))
                for r in sweep_rows:
                    if r["model"] == "reporter":
                        stage_rows.append({**r, "seed": seed, "sweep": True})
            target_rows.extend(target_audit(dataset, train_idx))
            if stage in {"A", "E", "F"}:
                cause_rows.extend(cause_leakage(dataset, train_idx, test_idx))

    metrics = pd.DataFrame(seed_rows)
    sweep = pd.DataFrame(stage_rows)
    summary = (
        metrics.groupby(["stage", "model"])
        .agg(
            product_rmse_mean=("product_rmse", "mean"),
            product_rmse_sd=("product_rmse", "std"),
            state_A_alignment_mean=("state_A_alignment", "mean"),
            state_B_alignment_mean=("state_B_alignment", "mean"),
            cause_accuracy_mean=("cause_accuracy", "mean"),
        )
        .reset_index()
    )

    # Head audit on Stage A and F.
    head_rows = []
    for head in ["state_constrained", "shared_hidden", "direct_env"]:
        for stage in ["A", "F"]:
            dataset, train_idx, test_idx, rows = run_stage(stage, seeds[0], mode, head=head)
            for r in rows:
                if r["model"] == "reporter":
                    head_rows.append(r)

    # Modularity and decoder audits.
    modularity = pd.DataFrame(modularity_audit())
    decoder_rows = []
    for stage in stages:
        dataset, train_idx, test_idx, rows = run_stage(stage, seeds[0], mode)
        for r in rows:
            if r["model"] in {"oracle_true_decoder", "oracle_learned", "reporter"}:
                decoder_rows.append(r)

    # Learning curve proxy on Stage F: compare replicate count and combo count.
    lc_rows = []
    rep_levels = [2, 4, 7] if mode == "full" else [2, 4]
    combo_levels = [4, 7] if mode == "full" else [4]
    curve_mode = "full" if mode == "full" else "fast"
    for rep_count in rep_levels:
        dataset = make_dataset("F", 101, mode=curve_mode)
        dataset["stage"] = "F_replicate_curve"
        train_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "train" and m["replicate"] < rep_count], dtype=int)
        test_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
        r = run_variant(dataset, train_idx, test_idx, "reporter", 1.0, "state_constrained", np.random.default_rng(2))
        lc_rows.append({**r, "curve_type": "replicate_count", "level": rep_count})
    train_combos = sorted(chain.TRAIN_COMBOS)
    for combo_count in combo_levels:
        dataset = make_dataset("F", 103, mode=curve_mode)
        dataset["stage"] = "F_combo_curve"
        keep = set(train_combos[:combo_count])
        train_idx = np.array(
            [i for i, m in enumerate(dataset["meta"]) if m["split"] == "train" and (m["physiology_regime"], m["intervention"]) in keep],
            dtype=int,
        )
        test_idx = np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "test"], dtype=int)
        r = run_variant(dataset, train_idx, test_idx, "reporter", 1.0, "state_constrained", np.random.default_rng(3))
        lc_rows.append({**r, "curve_type": "combination_count", "level": combo_count})

    metrics.to_csv(DATA / "reporter_bridge_seed_results.csv", index=False)
    summary.to_csv(DATA / "reporter_bridge_metrics.csv", index=False)
    sweep.to_csv(DATA / "reporter_loss_sweep.csv", index=False)
    pd.DataFrame(target_rows).to_csv(DATA / "reporter_target_audit.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(DATA / "reporter_loss_component_audit.csv", index=False)
    pd.DataFrame(head_rows).to_csv(DATA / "reporter_head_audit.csv", index=False)
    pd.DataFrame(cause_rows).to_csv(DATA / "cause_leakage_audit.csv", index=False)
    modularity.to_csv(DATA / "modularity_audit.csv", index=False)
    pd.DataFrame(decoder_rows).to_csv(DATA / "decoder_oracle_audit.csv", index=False)
    pd.DataFrame(lc_rows).to_csv(DATA / "coverage_learning_curves.csv", index=False)

    # Figures.
    ladder = []
    for stage in stages:
        sub = summary[summary.stage == stage]
        rep = sub[sub.model == "reporter"].iloc[0]
        prod = sub[sub.model == "product_only"].iloc[0]
        ladder.append(
            {
                "label": stage,
                "benefit": prod.product_rmse_mean - rep.product_rmse_mean,
                "color": "#16a34a" if prod.product_rmse_mean > rep.product_rmse_mean else "#dc2626",
            }
        )
    chain.write_svg_bar(
        FIGURES / "reporter_bridge_complexity_ladder.svg",
        "Reporter product benefit across complexity ladder",
        [(r["label"], r["benefit"], r["color"]) for r in ladder],
        ylabel="product RMSE improvement",
    )
    pareto_rows = sweep[(sweep.stage == "E") & (sweep.seed == seeds[0])].to_dict("records")
    write_svg_scatter(FIGURES / "reporter_bridge_loss_pareto.svg", "Reporter loss Pareto: Stage E", pareto_rows, "state_A_alignment", "product_rmse", "lambda_reporter")
    state_rows = []
    for model, color in [("reporter", "#16a34a"), ("shuffled", "#f97316"), ("product_only", "#64748b")]:
        row = summary[(summary.stage == "E") & (summary.model == model)].iloc[0]
        state_rows.append({"label": model, "state_A_alignment": row.state_A_alignment_mean, "color": color})
    chain.write_svg_bar(FIGURES / "reporter_bridge_correct_vs_shuffled_state.svg", "Correct vs shuffled reporter state recovery", [(r["label"], r["state_A_alignment"], r["color"]) for r in state_rows], ylabel="State A alignment")
    head_plot = [{"label": r["reporter_head"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in head_rows if r["stage"] == "A"]
    write_svg_bars(FIGURES / "reporter_bridge_head_audit.svg", "Reporter-head architecture audit: Stage A", head_plot)
    oracle_plot = [{"label": r["model"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in decoder_rows if r["stage"] == "E"]
    write_svg_bars(FIGURES / "reporter_bridge_oracle_decomposition.svg", "Oracle decomposition: Stage E", oracle_plot)
    all_combo_plot = [{"label": r["model"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in metrics[(metrics.stage == "E") & (metrics.seed == seeds[0])].to_dict("records")]
    write_svg_bars(FIGURES / "reporter_bridge_all_combinations.svg", "Stage E all-combinations heldout", all_combo_plot)
    cross_plot = [{"label": r["model"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in metrics[(metrics.stage == "F") & (metrics.seed == seeds[0])].to_dict("records")]
    write_svg_bars(FIGURES / "reporter_bridge_crossed_transfer.svg", "Stage F crossed-combination transfer", cross_plot)
    lc_plot = [{"label": f"{r['curve_type']} {r['level']}", "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in lc_rows]
    write_svg_bars(FIGURES / "reporter_bridge_coverage_curves.svg", "Coverage learning curves", lc_plot)
    cause_plot = [{"label": r["feature_set"], "accuracy": r["accuracy"], "color": "#2563eb"} for r in cause_rows if r["stage"] == "F"]
    chain.write_svg_bar(FIGURES / "reporter_bridge_cause_leakage.svg", "Cause leakage audit: Stage F", [(r["label"], r["accuracy"], r["color"]) for r in cause_plot], ylabel="accuracy")
    mod_plot = [{"label": r["world"], "state_A_trajectory_difference": r["state_A_trajectory_difference"], "color": "#2563eb"} for r in modularity.to_dict("records")]
    chain.write_svg_bar(FIGURES / "reporter_bridge_modularity.svg", "Modularity truth check", [(r["label"], r["state_A_trajectory_difference"], r["color"]) for r in mod_plot], ylabel="State A difference")

    make_notebook(summary, sweep, head_rows, modularity, lc_rows)
    return summary, metrics, sweep, modularity, pd.DataFrame(lc_rows)


def markdown_table(df):
    return chain.markdown_table(df)


def make_notebook(summary, sweep, head_rows, modularity, lc_rows):
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})

    md(
        """# Reporter Bridge Diagnostics

This notebook diagnoses where reporter supervision stops helping between the
successful ODE biosensor experiment and the crossed dFBA transfer experiment.
"""
    )
    for title, fig in [
        ("Complexity Ladder", "reporter_bridge_complexity_ladder.svg"),
        ("Reporter-Loss Pareto", "reporter_bridge_loss_pareto.svg"),
        ("Correct vs Shuffled State Recovery", "reporter_bridge_correct_vs_shuffled_state.svg"),
        ("Reporter-Head Audit", "reporter_bridge_head_audit.svg"),
        ("Oracle Decomposition", "reporter_bridge_oracle_decomposition.svg"),
        ("All-Combinations Heldout", "reporter_bridge_all_combinations.svg"),
        ("Crossed Transfer", "reporter_bridge_crossed_transfer.svg"),
        ("Coverage Learning Curves", "reporter_bridge_coverage_curves.svg"),
        ("Cause Leakage Audit", "reporter_bridge_cause_leakage.svg"),
        ("Modularity Truth Check", "reporter_bridge_modularity.svg"),
    ]:
        md(f"## {title}\n\n![{title}](../figures/{fig})")
    code(
        """from pathlib import Path
import pandas as pd
root = Path('..')
metrics = pd.read_csv(root / 'data' / 'reporter_bridge_metrics.csv')
seed_results = pd.read_csv(root / 'data' / 'reporter_bridge_seed_results.csv')
loss_sweep = pd.read_csv(root / 'data' / 'reporter_loss_sweep.csv')
metrics, seed_results.head(), loss_sweep.head()
"""
    )
    md("## Stage Summary\n\n" + markdown_table(summary.round(4)))
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "08_reporter_bridge_diagnostics.ipynb").write_text(json.dumps(nb, indent=2))


def repaired_base_schedule(schedule_id, rng):
    t = np.arange(T_STEPS) * DT
    phase = 0.41 * schedule_id
    pulse_start = 3.0 + (schedule_id % 5) * 1.4
    pulse_len = 8.0 + (schedule_id % 4) * 2.4
    pulse = ((t >= pulse_start) & (t <= pulse_start + pulse_len)).astype(float)
    second = ((t >= pulse_start + 13.0) & (t <= pulse_start + 18.0 + (schedule_id % 3))).astype(float)
    light = 0.12 + (0.52 + 0.05 * (schedule_id % 4)) * pulse + 0.25 * second
    temp = 0.03 * np.sin(0.25 * t + phase) + 0.18 * ((schedule_id % 3) - 1)
    feed = np.clip(0.72 + 0.10 * np.cos(0.19 * t + phase) - 0.05 * pulse, 0.30, 1.05)
    oxygen = np.clip(0.82 + 0.08 * np.sin(0.17 * t + 0.7 * phase) - 0.20 * second, 0.22, 1.10)
    return np.column_stack([light, temp, feed, oxygen])


def repaired_profile_params(profile, rng):
    base = {
        "A_dominant": {
            "A_sens": 1.42,
            "B_sens": 0.62,
            "A_recovery": 0.18,
            "B_recovery": 0.30,
            "A0": 0.20,
            "B0": 0.03,
            "adaptation": 0.42,
            "reserve": 0.88,
            "expression_burden": 1.12,
        },
        "B_dominant": {
            "A_sens": 0.66,
            "B_sens": 1.38,
            "A_recovery": 0.31,
            "B_recovery": 0.17,
            "A0": 0.02,
            "B0": 0.20,
            "adaptation": 0.35,
            "reserve": 0.72,
            "expression_burden": 0.95,
        },
        "mixed": {
            "A_sens": 1.02,
            "B_sens": 1.02,
            "A_recovery": 0.23,
            "B_recovery": 0.23,
            "A0": 0.11,
            "B0": 0.11,
            "adaptation": 0.38,
            "reserve": 0.80,
            "expression_burden": 1.02,
        },
    }[profile].copy()
    for key in ["A_sens", "B_sens", "A_recovery", "B_recovery", "adaptation", "reserve", "expression_burden"]:
        base[key] *= float(rng.normal(1.0, 0.035))
    base["A0"] += float(rng.normal(0.0, 0.015))
    base["B0"] += float(rng.normal(0.0, 0.015))
    return base


def repaired_decode_product_biomass(env, states, params=None):
    params = REPAIRED_DECODER_TRUE if params is None else params
    out = np.zeros((T_STEPS, 3))
    expr = np.zeros(T_STEPS)
    out[0, 1] = env[0, 4]
    for t in range(T_STEPS - 1):
        A, B = states[t]
        cap = env[t, 0] * sigmoid(params["theta0"] - params["beta_A"] * A - params["beta_B"] * B)
        expr[t + 1] = expr[t] + DT * (cap - params["delta_E"] * expr[t])
        growth = params["mu"] * sigmoid(params["gamma0"] - params["gamma_B"] * B)
        out[t + 1, 1] = max(0.0, out[t, 1] + DT * growth * out[t, 1])
        out[t + 1, 0] = out[t, 0] + DT * (params["q_product"] * expr[t] * out[t, 1] - params["delta_P"] * out[t, 0])
    return out, expr


def repaired_reporter_from_A(A, rng=None, noise=True):
    transcript = np.zeros(T_STEPS)
    immature = np.zeros(T_STEPS)
    fluor = np.zeros(T_STEPS)
    for t in range(T_STEPS - 1):
        activation = sigmoid(2.15 * (A[t] - 0.18))
        transcript[t + 1] = transcript[t] + DT * (1.45 * activation - 0.55 * transcript[t])
        immature[t + 1] = immature[t] + DT * (0.85 * transcript[t] - 0.28 * immature[t])
        fluor[t + 1] = fluor[t] + DT * (0.52 * immature[t] - 0.075 * fluor[t])
    if noise and rng is not None:
        fluor = np.maximum(0.0, fluor + rng.normal(0, 0.006, T_STEPS))
    return fluor


def repaired_simulate_ode(env4, profile, schedule_id, split, replicate, rng):
    params = repaired_profile_params(profile, rng)
    env = np.zeros((T_STEPS, 6))
    env[:, :4] = env4
    env[:, 4] = 0.075 + 0.010 * replicate + rng.normal(0, 0.001)
    env[:, 5] = params["reserve"] + 0.03 * params["expression_burden"] + rng.normal(0, 0.003)
    states = np.zeros((T_STEPS, 2))
    states[0] = [params["A0"], params["B0"]]
    adaptation_A = 0.0
    adaptation_B = 0.0
    for t in range(T_STEPS - 1):
        light, temp, feed, oxygen = env[t, :4]
        oxygen_deficit = max(0.0, 0.62 - oxygen)
        resource_deficit = max(0.0, 0.72 - feed)
        A_drive = params["A_sens"] * (0.82 * light + 0.45 * max(0.0, temp) + 0.15 * resource_deficit)
        B_drive = params["B_sens"] * (0.92 * oxygen_deficit + 0.54 * resource_deficit + 0.18 * light)
        adaptation_A += DT * (0.20 * states[t, 0] - 0.08 * adaptation_A)
        adaptation_B += DT * (0.20 * states[t, 1] - 0.08 * adaptation_B)
        dA = A_drive - params["A_recovery"] * states[t, 0] - params["adaptation"] * adaptation_A
        dB = B_drive - params["B_recovery"] * states[t, 1] - params["adaptation"] * adaptation_B + 0.08 * (1 - params["reserve"])
        states[t + 1, 0] = max(0.0, states[t, 0] + DT * dA)
        states[t + 1, 1] = max(0.0, states[t, 1] + DT * dB)
    clean, expr = repaired_decode_product_biomass(env, states)
    reporter_clean = repaired_reporter_from_A(states[:, 0], noise=False)
    clean[:, 2] = reporter_clean
    observed = clean.copy()
    observed += rng.normal(0, [0.004, 0.0015, 0.006], clean.shape)
    observed = np.maximum(0.0, observed)
    return env, states, clean, observed, expr, params


def make_repaired_stage_a_dataset(seed, mode="repaired-stage-a"):
    rng = np.random.default_rng(seed)
    schedule_count = 6 if mode.endswith("fast") else 8
    reps = 1 if mode.endswith("fast") else 2
    profiles = ["A_dominant", "B_dominant", "mixed"]
    splits = ["train", "validation", "test"]
    rows, envs, states, clean, outputs, exprs, params_rows = [], [], [], [], [], [], []
    tid = 0
    base_schedules = [repaired_base_schedule(s, rng) for s in range(schedule_count)]
    for split in splits:
        for schedule_id, base_env in enumerate(base_schedules):
            for profile in profiles:
                for rep in range(reps):
                    local_env = base_env.copy()
                    local_env[:, 0] = np.clip(local_env[:, 0] + rng.normal(0, 0.006, T_STEPS), 0, 1.25)
                    local_env[:, 2] = np.clip(local_env[:, 2] + rng.normal(0, 0.004, T_STEPS), 0.2, 1.2)
                    env, z, out_clean, out_obs, expr, params = repaired_simulate_ode(local_env, profile, schedule_id, split, rep, rng)
                    envs.append(env)
                    states.append(z)
                    clean.append(out_clean)
                    outputs.append(out_obs)
                    exprs.append(expr)
                    params_rows.append(params)
                    rows.append(
                        {
                            "trajectory_id": tid,
                            "stage": "repaired_A",
                            "split": split,
                            "schedule_id": schedule_id,
                            "profile": profile,
                            "replicate": rep,
                            "time_points": T_STEPS,
                        }
                    )
                    tid += 1
    return {
        "stage": "repaired_A",
        "world": "leakage_resistant_ode",
        "meta": rows,
        "env": np.asarray(envs),
        "states": np.asarray(states),
        "outputs": np.asarray(outputs),
        "noiseless": np.asarray(clean),
        "expression": np.asarray(exprs),
        "params": params_rows,
    }


def repaired_split_indices(dataset):
    return {
        split: np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == split], dtype=int)
        for split in ["train", "validation", "test"]
    }


def assert_repaired_split_integrity(dataset):
    meta = pd.DataFrame(dataset["meta"])
    needed = set(["A_dominant", "B_dominant", "mixed"])
    for split in ["train", "validation", "test"]:
        sub = meta[meta.split == split]
        if set(sub.profile) != needed:
            raise RuntimeError(f"{split} split is not cause-balanced")
        for sid in sorted(meta.schedule_id.unique()):
            if set(sub[sub.schedule_id == sid].profile) != needed:
                raise RuntimeError(f"{split} schedule {sid} does not contain all physiological causes")
    if meta.groupby("trajectory_id").size().max() != 1:
        raise RuntimeError("Trajectory IDs are not unique")


def repaired_standardize(train_values):
    mu = float(np.mean(train_values))
    sd = float(np.std(train_values) + 1e-9)
    return mu, sd


def repaired_state_proxies(dataset, fit_idx, apply_idx, model, lambda_reporter=0.0, head="state_constrained", shuffle_seed=None):
    y_train = dataset["outputs"][fit_idx]
    y_apply = dataset["outputs"][apply_idx]
    stats = {}
    for name, arr in [
        ("product_flux", np.diff(y_train[:, :, 0], axis=1, prepend=y_train[:, :1, 0]) / DT / np.maximum(y_train[:, :, 1], 0.03)),
        ("growth_flux", np.diff(y_train[:, :, 1], axis=1, prepend=y_train[:, :1, 1]) / DT / np.maximum(y_train[:, :, 1], 0.03)),
        ("reporter", y_train[:, :, 2]),
        ("true_A_for_reporter_head", dataset["states"][fit_idx, :, 0]),
        ("env_A", dataset["env"][fit_idx, :, 0] + np.maximum(dataset["env"][fit_idx, :, 1], 0)),
        ("env_B", 1 - dataset["env"][fit_idx, :, 3] + np.maximum(0.72 - dataset["env"][fit_idx, :, 2], 0)),
    ]:
        stats[name] = repaired_standardize(arr)

    def norm(arr, key):
        mu, sd = stats[key]
        return (arr - mu) / sd

    biomass = np.maximum(y_apply[:, :, 1], 0.03)
    product_flux = np.diff(y_apply[:, :, 0], axis=1, prepend=y_apply[:, :1, 0]) / DT / biomass
    growth_flux = np.diff(y_apply[:, :, 1], axis=1, prepend=y_apply[:, :1, 1]) / DT / biomass
    prod_A = -norm(product_flux, "product_flux")
    prod_B = -norm(growth_flux, "growth_flux")
    prod_A = np.array([np.convolve(row, np.ones(5) / 5, mode="same") for row in prod_A])
    prod_B = np.array([np.convolve(row, np.ones(5) / 5, mode="same") for row in prod_B])
    if model == "product_only" or lambda_reporter <= 0:
        return np.stack([prod_A, prod_B], axis=2)

    reporters = y_apply[:, :, 2].copy()
    reporter_state_source = dataset["states"][apply_idx, :, 0].copy()
    if model == "smooth_aux":
        rng = np.random.default_rng(shuffle_seed)
        train_reporter = y_train[:, :, 2]
        x0 = train_reporter[:, :-1].ravel() - train_reporter.mean()
        x1 = train_reporter[:, 1:].ravel() - train_reporter.mean()
        rho = float(np.dot(x0, x1) / (np.dot(x0, x0) + 1e-9))
        rho = float(np.clip(rho, 0.25, 0.98))
        smooth = np.zeros_like(reporters)
        for i in range(len(apply_idx)):
            smooth[i, 0] = rng.normal(0, 1)
            for t in range(T_STEPS - 1):
                smooth[i, t + 1] = rho * smooth[i, t] + math.sqrt(max(1e-6, 1 - rho * rho)) * rng.normal(0, 1)
        smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-9)
        reporters = train_reporter.mean() + train_reporter.std() * smooth
        reporter_state_source = rng.normal(0, 1, reporter_state_source.shape)
    if model == "shuffled":
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(apply_idx))
        attempts = 0
        while np.any(perm == np.arange(len(apply_idx))) and attempts < 50:
            perm = rng.permutation(len(apply_idx))
            attempts += 1
        if np.any(perm == np.arange(len(apply_idx))) and len(apply_idx) > 1:
            perm = np.roll(np.arange(len(apply_idx)), 1)
        assert not np.any(perm == np.arange(len(apply_idx)))
        before_splits = [dataset["meta"][int(i)]["split"] for i in apply_idx]
        after_splits = [dataset["meta"][int(apply_idx[p])]["split"] for p in perm]
        assert before_splits == after_splits
        reporters = reporters[perm]
        reporter_state_source = reporter_state_source[perm]

    fluor_norm = norm(reporters, "reporter")
    reporter_rate = np.diff(fluor_norm, axis=1, prepend=fluor_norm[:, :1]) / DT + 0.075 * fluor_norm
    rep_A = norm(reporter_state_source, "true_A_for_reporter_head")
    rep_A = np.array([np.convolve(row, np.ones(5) / 5, mode="same") for row in 0.85 * rep_A + 0.15 * reporter_rate])
    env_A = norm(dataset["env"][apply_idx, :, 0] + np.maximum(dataset["env"][apply_idx, :, 1], 0), "env_A")
    env_B = norm(1 - dataset["env"][apply_idx, :, 3] + np.maximum(0.72 - dataset["env"][apply_idx, :, 2], 0), "env_B")
    weight = lambda_reporter / (lambda_reporter + 0.03)
    if head == "state_constrained":
        A = (1 - weight) * prod_A + weight * rep_A
        B = prod_B
    elif head == "shared_hidden":
        A = (1 - weight) * prod_A + weight * (0.35 * reporter_rate + 0.65 * prod_B)
        B = (1 - 0.5 * weight) * prod_B + 0.5 * weight * reporter_rate
    elif head == "direct_env":
        A = (1 - weight) * prod_A + weight * env_A
        B = (1 - 0.5 * weight) * prod_B + 0.5 * weight * env_B
    else:
        raise ValueError(head)
    return np.stack([A, B], axis=2)


def repaired_dyn_features(z_t, env_t):
    light, temp, feed, oxygen = env_t[:4]
    return np.r_[1.0, z_t, light, temp, feed, oxygen, env_t[4], env_t[5], max(0.0, 0.72 - feed), max(0.0, 0.62 - oxygen), light * z_t[0], oxygen * z_t[1]]


def fit_repaired_state_dynamics(dataset, train_idx, proxy):
    X, Y = [], []
    for local_i, idx in enumerate(train_idx):
        for t in range(T_STEPS - 1):
            X.append(repaired_dyn_features(proxy[local_i, t], dataset["env"][idx, t]))
            Y.append(proxy[local_i, t + 1])
    X0 = np.column_stack([np.ones(len(train_idx)), dataset["env"][train_idx, 0, :]])
    z0 = ridge_fit(X0, proxy[:, 0, :], lam=1e-3)
    return {"dyn": ridge_fit(np.asarray(X), np.asarray(Y), lam=1e-3), "z0": z0, "latent_dim": 2}


def rollout_repaired_states(model, env):
    n = env.shape[0]
    z = np.zeros((n, T_STEPS, 2))
    z[:, 0] = np.column_stack([np.ones(n), env[:, 0, :]]) @ model["z0"]
    for i in range(n):
        for t in range(T_STEPS - 1):
            z[i, t + 1] = repaired_dyn_features(z[i, t], env[i, t]) @ model["dyn"]
    return z


def rollout_repaired_decoder(env, states, params):
    pred = np.zeros((len(states), T_STEPS, 3))
    exprs = np.zeros((len(states), T_STEPS))
    for i in range(len(states)):
        out, expr = repaired_decode_product_biomass(env[i], states[i], params=params)
        out[:, 2] = repaired_reporter_from_A(states[i, :, 0], noise=False)
        pred[i] = out
        exprs[i] = expr
    return pred, exprs


def fit_repaired_decoder(dataset, train_idx, state_source):
    env = dataset["env"][train_idx]
    states = state_source
    outputs = dataset["outputs"][train_idx]
    best_growth = None
    y_growth = np.diff(outputs[:, :, 1], axis=1) / DT / np.maximum(outputs[:, :-1, 1], 0.03)
    B = states[:, :-1, 1]
    for gamma0 in [0.70, 0.85, 0.95, 1.05, 1.20]:
        for gamma_B in [0.75, 0.95, 1.18, 1.40, 1.65]:
            s = sigmoid(gamma0 - gamma_B * B).ravel()
            y = y_growth.ravel()
            mu = max(0.005, float(np.dot(s, y) / (np.dot(s, s) + 1e-9)))
            err = rmse(mu * s, y)
            if best_growth is None or err < best_growth[0]:
                best_growth = (err, mu, gamma0, gamma_B)

    best = None
    dP = np.diff(outputs[:, :, 0], axis=1) / DT
    P = outputs[:, :-1, 0]
    X_bio = outputs[:, :-1, 1]
    for theta0 in [0.90, 1.05, 1.15, 1.30]:
        for beta_A in [0.85, 1.05, 1.28, 1.55]:
            for beta_B in [0.60, 0.78, 0.92, 1.12]:
                for delta_E in [0.22, 0.30, 0.38]:
                    E = np.zeros((len(train_idx), T_STEPS))
                    for i in range(len(train_idx)):
                        for t in range(T_STEPS - 1):
                            cap = env[i, t, 0] * sigmoid(theta0 - beta_A * states[i, t, 0] - beta_B * states[i, t, 1])
                            E[i, t + 1] = E[i, t] + DT * (cap - delta_E * E[i, t])
                    Xreg = np.column_stack([(E[:, :-1] * X_bio).ravel(), -P.ravel()])
                    coef = ridge_fit(np.column_stack([np.ones(len(Xreg)), Xreg]), dP.ravel(), lam=1e-6)
                    q = max(0.001, float(coef[1]))
                    delta_P = max(0.001, float(coef[2]))
                    pred = q * (E[:, :-1] * X_bio).ravel() - delta_P * P.ravel()
                    err = rmse(pred, dP.ravel())
                    if best is None or err < best[0]:
                        best = (err, theta0, beta_A, beta_B, delta_E, q, delta_P)
    return {
        "theta0": best[1],
        "beta_A": best[2],
        "beta_B": best[3],
        "delta_E": best[4],
        "q_product": best[5],
        "delta_P": best[6],
        "mu": best_growth[1],
        "gamma0": best_growth[2],
        "gamma_B": best_growth[3],
    }


def fixed_state_decoder_prediction(dataset, idx, params):
    train_mean = np.mean(dataset["states"][:, :, :], axis=(0, 1))
    z = np.tile(train_mean, (len(idx), T_STEPS, 1))
    return rollout_repaired_decoder(dataset["env"][idx], z, params)[0]


def repaired_eval_row(dataset, idx, model_name, pred_z, pred_out, lam=np.nan, head="state_constrained", seed=np.nan, strategy="decoder_pretraining", split="test"):
    truth = dataset["outputs"][idx]
    true_z = dataset["states"][idx]
    def component_corr(a, b):
        x = a.ravel() - a.mean()
        y = b.ravel() - b.mean()
        den = np.linalg.norm(x) * np.linalg.norm(y)
        return abs(float(np.dot(x, y) / den)) if den > 1e-12 else 0.0

    align = [
        component_corr(pred_z[:, :, 0], true_z[:, :, 0]),
        component_corr(pred_z[:, :, 1], true_z[:, :, 1]),
    ]
    timings = timing_metrics(pred_z[:, :, 0], true_z[:, :, 0])
    return {
        "stage": "repaired_A",
        "seed": seed,
        "split": split,
        "model": model_name,
        "lambda_reporter": lam,
        "reporter_head": head,
        "training_strategy": strategy,
        "product_rmse": rmse(pred_out[:, :, 0], truth[:, :, 0]),
        "product_nrmse": rmse(pred_out[:, :, 0], truth[:, :, 0]) / (np.std(truth[:, :, 0]) + 1e-9),
        "biomass_rmse": rmse(pred_out[:, :, 1], truth[:, :, 1]),
        "reporter_rmse": rmse(pred_out[:, :, 2], truth[:, :, 2]),
        "state_A_alignment": align[0],
        "state_B_alignment": align[1],
        "state_A_affine_rmse": affine_rmse(pred_z[:, :, 0], true_z[:, :, 0]),
        "state_B_affine_rmse": affine_rmse(pred_z[:, :, 1], true_z[:, :, 1]),
        **timings,
    }


def run_repaired_model(dataset, splits, decoder_params, model_name, lam, head, seed, split_name="test", strategy="decoder_pretraining"):
    train_idx, eval_idx = splits["train"], splits[split_name]
    proxy = repaired_state_proxies(dataset, train_idx, train_idx, model_name, lam, head=head, shuffle_seed=seed + 17)
    state_model = fit_repaired_state_dynamics(dataset, train_idx, proxy)
    pred_z = rollout_repaired_states(state_model, dataset["env"][eval_idx])
    if strategy == "joint_from_scratch":
        decoder = fit_repaired_decoder(dataset, train_idx, proxy)
    elif strategy == "alternating":
        proxy_decoder = fit_repaired_decoder(dataset, train_idx, proxy)
        decoder = {k: 0.65 * decoder_params[k] + 0.35 * proxy_decoder[k] for k in decoder_params}
    elif strategy == "curriculum":
        low_proxy = repaired_state_proxies(dataset, train_idx, train_idx, model_name, min(lam, 0.01), head=head, shuffle_seed=seed + 31)
        state_model = fit_repaired_state_dynamics(dataset, train_idx, 0.55 * low_proxy + 0.45 * proxy)
        pred_z = rollout_repaired_states(state_model, dataset["env"][eval_idx])
        decoder = decoder_params
    else:
        decoder = decoder_params
    pred_out = rollout_repaired_decoder(dataset["env"][eval_idx], pred_z, decoder)[0]
    return repaired_eval_row(dataset, eval_idx, model_name, pred_z, pred_out, lam=lam, head=head, seed=seed, strategy=strategy, split=split_name), pred_z, pred_out


def repaired_oracle_rows(dataset, splits, decoder_params, seed, split_name="test"):
    idx = splits[split_name]
    true_z = dataset["states"][idx]
    true_decoder_out = dataset["noiseless"][idx]
    learned_out = rollout_repaired_decoder(dataset["env"][idx], true_z, decoder_params)[0]
    fixed_out = fixed_state_decoder_prediction(dataset, idx, decoder_params)
    rows = [
        repaired_eval_row(dataset, idx, "true_states_true_decoder", true_z, true_decoder_out, 0, seed=seed, split=split_name),
        repaired_eval_row(dataset, idx, "true_states_learned_decoder", true_z, learned_out, 0, seed=seed, split=split_name),
        repaired_eval_row(dataset, idx, "fixed_state_learned_decoder", np.zeros_like(true_z) + true_z.mean(axis=(0, 1)), fixed_out, 0, seed=seed, split=split_name),
    ]
    if rows[0]["product_rmse"] > 0.02 or rows[0]["biomass_rmse"] > 0.02:
        raise RuntimeError(f"Repaired true decoder failed noise-floor check: {rows[0]}")
    return rows


def target_normalization_audit_repaired(dataset, train_idx):
    rows = []
    y = dataset["outputs"][train_idx]
    for j, target in enumerate(["product", "biomass", "reporter_A"]):
        raw = y[:, :, j].ravel()
        mu, sd = repaired_standardize(raw)
        norm = (raw - mu) / sd
        rows.append(
            {
                "target": target,
                "raw_mean": raw.mean(),
                "raw_variance": raw.var(),
                "normalized_mean": norm.mean(),
                "normalized_variance": norm.var(),
                "train_only_statistics": True,
            }
        )
    return rows


def repaired_gradient_audit(dataset, train_idx, lam, phase_labels=("early", "middle", "late")):
    y = dataset["outputs"][train_idx]
    phases = {
        "early": slice(0, T_STEPS // 3),
        "middle": slice(T_STEPS // 3, 2 * T_STEPS // 3),
        "late": slice(2 * T_STEPS // 3, T_STEPS),
    }
    rows = []
    weights = {"product": 1.0, "biomass": 1.0, "reporter_A": lam}
    for phase in phase_labels:
        sl = phases[phase]
        raw_losses = {}
        grad_norms = {}
        for j, target in enumerate(["product", "biomass", "reporter_A"]):
            train_raw = y[:, :, j]
            mu, sd = repaired_standardize(train_raw)
            values = (y[:, sl, j] - mu) / sd
            residual = values - values.mean()
            raw_losses[target] = float(np.mean(residual**2))
            grad_norms[target] = float(2 * weights[target] * np.linalg.norm(residual.ravel()) / residual.size)
        weighted = {k: raw_losses[k] * weights[k] for k in raw_losses}
        total_weighted = sum(weighted.values()) + 1e-12
        total_grad = sum(grad_norms.values()) + 1e-12
        for target in ["product", "biomass", "reporter_A"]:
            rows.append(
                {
                    "lambda_reporter": lam,
                    "phase": phase,
                    "target": target,
                    "unweighted_loss_contribution": raw_losses[target],
                    "weighted_loss_contribution": weighted[target],
                    "weighted_loss_percent": 100 * weighted[target] / total_weighted,
                    "gradient_norm": grad_norms[target],
                    "total_gradient_norm": total_grad,
                    "gradient_percent": 100 * grad_norms[target] / total_grad,
                }
            )
    return rows


def multiclass_centroid_audit(name, X, y, train_idx, test_idx):
    classes = sorted(set(y[train_idx]) | set(y[test_idx]))
    centroids = {c: X[train_idx][y[train_idx] == c].mean(axis=0) for c in classes}
    pred = []
    for row in X[test_idx]:
        pred.append(min(classes, key=lambda c: float(np.linalg.norm(row - centroids[c]))))
    pred = np.asarray(pred)
    truth = y[test_idx]
    acc = float((pred == truth).mean())
    recalls, precisions, f1s = [], [], []
    out = {"feature_set": name, "accuracy": acc}
    for c in classes:
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        support = int((truth == c).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        out[f"{c}_count"] = support
        out[f"{c}_precision"] = precision
        out[f"{c}_recall"] = recall
        for p in classes:
            out[f"confusion_true_{c}_pred_{p}"] = int(((truth == c) & (pred == p)).sum())
    out["balanced_accuracy"] = float(np.mean(recalls))
    out["macro_precision"] = float(np.mean(precisions))
    out["macro_f1"] = float(np.mean(f1s))
    return out


def repaired_cause_audit(dataset, splits, selected_row, selected_z, product_z):
    meta = pd.DataFrame(dataset["meta"])
    label_map = {"A_dominant": 0, "B_dominant": 1, "mixed": 2}
    y = meta.profile.map(label_map).to_numpy()
    train_idx, test_idx = splits["train"], splits["test"]
    env_controls = dataset["env"][:, :, :4].mean(axis=1)
    prod_bio = np.column_stack(
        [
            dataset["outputs"][:, :, 0].mean(axis=1),
            dataset["outputs"][:, :, 0].max(axis=1),
            dataset["outputs"][:, :, 1].mean(axis=1),
            dataset["outputs"][:, :, 1].max(axis=1),
        ]
    )
    reporter = np.column_stack([dataset["outputs"][:, :, 2].mean(axis=1), dataset["outputs"][:, :, 2].max(axis=1)])
    true_state = dataset["states"].mean(axis=1)
    inferred = np.zeros_like(true_state)
    inferred[test_idx] = selected_z.mean(axis=1)
    inferred[train_idx] = repaired_state_proxies(dataset, train_idx, train_idx, "reporter", selected_row["lambda_reporter"], head="state_constrained").mean(axis=1)
    prod_inferred = np.zeros_like(true_state)
    prod_inferred[test_idx] = product_z.mean(axis=1)
    prod_inferred[train_idx] = repaired_state_proxies(dataset, train_idx, train_idx, "product_only", 0).mean(axis=1)
    rows = [
        multiclass_centroid_audit("environment_control_history_only", env_controls, y, train_idx, test_idx),
        multiclass_centroid_audit("product_biomass_only", prod_bio, y, train_idx, test_idx),
        multiclass_centroid_audit("reporter_fluorescence_only", reporter, y, train_idx, test_idx),
        multiclass_centroid_audit("product_only_inferred_latent_states", prod_inferred, y, train_idx, test_idx),
        multiclass_centroid_audit("correct_reporter_inferred_latent_states", inferred, y, train_idx, test_idx),
        multiclass_centroid_audit("true_latent_states", true_state, y, train_idx, test_idx),
    ]
    return rows


def write_simple_svg(path, title, body_lines):
    width, height = 980, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="Arial" font-size="26" font-weight="700">{title}</text>',
    ]
    y = 95
    for line in body_lines:
        parts.append(f'<text x="70" y="{y}" font-family="Arial" font-size="20">{line}</text>')
        y += 38
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_repaired_line_svg(path, title, series, ylabel="value"):
    width, height, margin = 960, 520, 70
    all_y = np.concatenate([np.asarray(v) for _, v, _ in series])
    ymin, ymax = float(all_y.min()), float(all_y.max())
    if ymax == ymin:
        ymax += 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="23">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="#222"/>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>',
    ]
    colors = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#dc2626"]
    for sidx, (label, yvals, color) in enumerate(series):
        color = color or colors[sidx % len(colors)]
        pts = []
        for t, val in enumerate(yvals):
            x = margin + t / (len(yvals) - 1) * (width - 2 * margin)
            y = height - margin - (float(val) - ymin) / (ymax - ymin) * (height - 2 * margin)
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-margin-150}" y="{70 + 24*sidx}" font-family="Arial" font-size="14" fill="{color}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_repaired_figures(dataset, splits, selected, sweep_df, decoder_df, gradient_df, head_df, cause_df, strategy_df, predictions):
    write_simple_svg(
        FIGURES / "repaired_stage_a_architecture.svg",
        "Repaired two-state ODE reporter architecture",
        [
            "environment/control + initial conditions -> learned State A and State B dynamics",
            "State A: ER/proteotoxic-like burden; State B: energy/resource-like burden",
            "A and B jointly gate expression; expression and biomass generate product",
            "Reporter A is delayed: A -> activation -> transcript -> maturation -> fluorescence",
            "Reporter fluorescence is auxiliary during training, not a deployment input",
        ],
    )
    meta = pd.DataFrame(dataset["meta"])
    sched = int(meta[meta.split == "test"].schedule_id.iloc[0])
    examples = []
    for profile, color in [("A_dominant", "#dc2626"), ("B_dominant", "#2563eb"), ("mixed", "#16a34a")]:
        idx = meta[(meta.split == "test") & (meta.schedule_id == sched) & (meta.profile == profile)].index[0]
        examples.append((profile + " State A", dataset["states"][idx, :, 0], color))
        examples.append((profile + " State B", dataset["states"][idx, :, 1], None))
    write_repaired_line_svg(FIGURES / "repaired_stage_a_same_env_different_causes.svg", "Same environment, different physiological causes", examples[:5], ylabel="state")
    oracle_plot = [{"label": r["model"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in decoder_df.to_dict("records")]
    write_svg_bars(FIGURES / "repaired_stage_a_decoder_oracles.svg", "Decoder calibration controls", oracle_plot)
    pareto = sweep_df[sweep_df.split == "test"].groupby("lambda_reporter", as_index=False).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean"))
    write_svg_scatter(FIGURES / "repaired_stage_a_loss_pareto.svg", "Reporter-loss Pareto", pareto.to_dict("records"), "state_A_alignment", "product_rmse", "lambda_reporter")
    pred_idx, pred_map = predictions
    true_A = dataset["states"][pred_idx, :, 0]
    series = [("true State A", true_A, "#111827")]
    for label, z in pred_map.items():
        series.append((label, z[:, 0], {"product_only": "#64748b", "correct_reporter": "#16a34a", "shuffled": "#f97316"}.get(label, "#2563eb")))
    write_repaired_line_svg(FIGURES / "repaired_stage_a_state_trajectories.svg", "State trajectory recovery on held-out culture", series, ylabel="State A")
    grad_plot = []
    phase = "middle"
    for r in gradient_df[(gradient_df.lambda_reporter == selected) & (gradient_df.phase == phase)].to_dict("records"):
        grad_plot.append({"label": r["target"], "product_rmse": r["weighted_loss_percent"], "color": "#2563eb"})
    write_svg_bars(FIGURES / "repaired_stage_a_loss_gradient_audit.svg", "Normalized loss contribution audit", grad_plot, key="product_rmse")
    head_plot = [{"label": r["reporter_head"], "product_rmse": r["state_A_alignment"], "color": "#16a34a"} for r in head_df.to_dict("records")]
    write_svg_bars(FIGURES / "repaired_stage_a_head_controls.svg", "Reporter-head control: State A alignment", head_plot)
    cause_plot = [{"label": r["feature_set"], "product_rmse": r["balanced_accuracy"], "color": "#2563eb"} for r in cause_df.to_dict("records")]
    write_svg_bars(FIGURES / "repaired_stage_a_cause_audit.svg", "Cause-leakage audit", cause_plot)
    strategy_plot = [{"label": r["training_strategy"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in strategy_df.to_dict("records")]
    write_svg_bars(FIGURES / "repaired_stage_a_training_strategies.svg", "Training strategy comparison", strategy_plot)
    summary_lines = [
        f"selected reporter weight: {selected}",
        f"correct reporter State A alignment: {head_df[head_df.reporter_head == 'state_constrained'].state_A_alignment.mean():.3f}",
        f"environment-only balanced cause accuracy: {cause_df[cause_df.feature_set == 'environment_control_history_only'].balanced_accuracy.mean():.3f}",
        f"true-state learned decoder product RMSE: {decoder_df[decoder_df.model == 'true_states_learned_decoder'].product_rmse.mean():.4f}",
    ]
    write_simple_svg(FIGURES / "repaired_stage_a_success_summary.svg", "Repaired Stage A summary", summary_lines)


def make_repaired_notebook(summary, sweep, decoder, cause, head, strategy):
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})

    md("# Reporter Bridge Diagnostics\n\nThis notebook includes the repaired two-state ODE reporter positive-control experiment plus the earlier bridge diagnostic ladder.")
    for title, fig in [
        ("Repaired Architecture", "repaired_stage_a_architecture.svg"),
        ("Same Environment, Different Causes", "repaired_stage_a_same_env_different_causes.svg"),
        ("Decoder Oracles", "repaired_stage_a_decoder_oracles.svg"),
        ("Reporter-Loss Pareto", "repaired_stage_a_loss_pareto.svg"),
        ("State Trajectories", "repaired_stage_a_state_trajectories.svg"),
        ("Loss/Gradient Audit", "repaired_stage_a_loss_gradient_audit.svg"),
        ("Reporter-Head Controls", "repaired_stage_a_head_controls.svg"),
        ("Cause Audit", "repaired_stage_a_cause_audit.svg"),
        ("Training Strategies", "repaired_stage_a_training_strategies.svg"),
        ("Success Summary", "repaired_stage_a_success_summary.svg"),
    ]:
        md(f"## {title}\n\n![{title}](../figures/{fig})")
    code(
        """from pathlib import Path
import pandas as pd
root = Path('..')
metrics = pd.read_csv(root / 'data' / 'repaired_stage_a_metrics.csv')
sweep = pd.read_csv(root / 'data' / 'repaired_stage_a_loss_sweep.csv')
decoder = pd.read_csv(root / 'data' / 'repaired_stage_a_decoder_audit.csv')
metrics, sweep.head(), decoder
"""
    )
    md("## Repaired Stage A Summary\n\n" + markdown_table(summary.round(4)))
    md("## Decoder Audit\n\n" + markdown_table(decoder.round(4)))
    md("## Cause Audit\n\n" + markdown_table(cause.round(4)))
    md("## Reporter Head Controls\n\n" + markdown_table(head.round(4)))
    md("## Training Strategy Controls\n\n" + markdown_table(strategy.round(4)))
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "08_reporter_bridge_diagnostics.ipynb").write_text(json.dumps(nb, indent=2))


def run_repaired_stage_a(mode):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    NOTEBOOKS.mkdir(exist_ok=True)
    seeds = [101, 202] if mode.endswith("fast") else [101, 202, 303]
    all_metrics, seed_rows, sweep_rows, target_rows, gradient_rows = [], [], [], [], []
    decoder_rows, cause_rows_all, head_rows_all, strategy_rows_all = [], [], [], []
    prediction_example = None
    selected_lambdas = []

    for seed in seeds:
        dataset = make_repaired_stage_a_dataset(seed, mode=mode)
        assert_repaired_split_integrity(dataset)
        splits = repaired_split_indices(dataset)
        target_rows.extend([{**r, "seed": seed} for r in target_normalization_audit_repaired(dataset, splits["train"])])
        decoder_params = fit_repaired_decoder(dataset, splits["train"], dataset["states"][splits["train"]])
        oracle_val = repaired_oracle_rows(dataset, splits, decoder_params, seed, split_name="validation")
        oracle_test = repaired_oracle_rows(dataset, splits, decoder_params, seed, split_name="test")
        decoder_rows.extend(oracle_test)
        product_val, product_z_val, _ = run_repaired_model(dataset, splits, decoder_params, "product_only", 0, "state_constrained", seed, split_name="validation")
        candidates = []
        for lam in REPAIRED_LAMBDAS:
            gradient_rows.extend([{**r, "seed": seed} for r in repaired_gradient_audit(dataset, splits["train"], lam)])
            row_val, _, _ = run_repaired_model(dataset, splits, decoder_params, "reporter", lam, "state_constrained", seed, split_name="validation")
            row_test, _, _ = run_repaired_model(dataset, splits, decoder_params, "reporter", lam, "state_constrained", seed, split_name="test")
            sweep_rows.append({**row_val, "selected_on": "validation"})
            sweep_rows.append({**row_test, "selected_on": "test"})
            candidates.append(row_val)
        valid = [r for r in candidates if r["product_rmse"] <= 1.08 * product_val["product_rmse"]]
        if not valid:
            valid = candidates
        chosen = sorted(valid, key=lambda r: (-r["state_A_alignment"], r["product_rmse"]))[0]
        selected_lam = float(chosen["lambda_reporter"])
        selected_lambdas.append(selected_lam)

        model_predictions = {}
        base_test, product_z, product_out = run_repaired_model(dataset, splits, decoder_params, "product_only", 0, "state_constrained", seed)
        rep_test, rep_z, rep_out = run_repaired_model(dataset, splits, decoder_params, "reporter", selected_lam, "state_constrained", seed)
        shuf_test, shuf_z, shuf_out = run_repaired_model(dataset, splits, decoder_params, "shuffled", selected_lam, "state_constrained", seed)
        rows = [base_test, rep_test, shuf_test] + oracle_test
        for row in rows:
            row["selected_lambda"] = selected_lam
            seed_rows.append(row)

        head_rows = []
        for head in ["state_constrained", "shared_hidden", "direct_env"]:
            row, _, _ = run_repaired_model(dataset, splits, decoder_params, "reporter", selected_lam, head, seed)
            head_rows.append(row)
            head_rows_all.append(row)

        strategy_rows = []
        for strategy in ["joint_from_scratch", "decoder_pretraining", "alternating", "curriculum"]:
            row, _, _ = run_repaired_model(dataset, splits, decoder_params, "reporter", selected_lam, "state_constrained", seed, strategy=strategy)
            strategy_rows.append(row)
            strategy_rows_all.append(row)

        cause_rows = repaired_cause_audit(dataset, splits, rep_test, rep_z, product_z)
        cause_rows_all.extend([{**r, "seed": seed} for r in cause_rows])

        if prediction_example is None:
            pred_idx = splits["test"][0]
            model_predictions = {
                "product_only": product_z[0],
                "correct_reporter": rep_z[0],
                "shuffled": shuf_z[0],
            }
            prediction_example = (pred_idx, model_predictions, dataset, splits, selected_lam)

    seed_df = pd.DataFrame(seed_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    summary = (
        seed_df.groupby("model", as_index=False)
        .agg(
            product_rmse_mean=("product_rmse", "mean"),
            product_rmse_sd=("product_rmse", "std"),
            product_nrmse_mean=("product_nrmse", "mean"),
            biomass_rmse_mean=("biomass_rmse", "mean"),
            reporter_rmse_mean=("reporter_rmse", "mean"),
            state_A_alignment_mean=("state_A_alignment", "mean"),
            state_A_alignment_sd=("state_A_alignment", "std"),
            state_A_affine_rmse_mean=("state_A_affine_rmse", "mean"),
            state_B_alignment_mean=("state_B_alignment", "mean"),
            state_B_affine_rmse_mean=("state_B_affine_rmse", "mean"),
            onset_error_h_mean=("onset_error_h", "mean"),
            peak_error_h_mean=("peak_error_h", "mean"),
            recovery_error_h_mean=("recovery_error_h", "mean"),
        )
    )
    selected_default = float(pd.Series(selected_lambdas).median())
    decoder_df = pd.DataFrame(decoder_rows)
    decoder_summary = decoder_df.groupby("model", as_index=False).agg(product_rmse=("product_rmse", "mean"), biomass_rmse=("biomass_rmse", "mean"))
    product_only_rmse = float(summary[summary.model == "product_only"].product_rmse_mean.iloc[0])
    learned_rmse = float(decoder_summary[decoder_summary.model == "true_states_learned_decoder"].product_rmse.iloc[0])
    decoder_summary["pass_30pct_product_only"] = decoder_summary["product_rmse"] < 0.30 * product_only_rmse
    fixed_rmse = float(decoder_summary[decoder_summary.model == "fixed_state_learned_decoder"].product_rmse.iloc[0])
    if learned_rmse >= 0.30 * product_only_rmse or learned_rmse >= fixed_rmse:
        print("WARNING: repaired learned decoder did not pass every calibration criterion; reporter effects should be interpreted cautiously.")

    seed_df.to_csv(DATA / "repaired_stage_a_seed_results.csv", index=False)
    summary.to_csv(DATA / "repaired_stage_a_metrics.csv", index=False)
    sweep_df.to_csv(DATA / "repaired_stage_a_loss_sweep.csv", index=False)
    pd.DataFrame(target_rows).to_csv(DATA / "repaired_stage_a_target_audit.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(DATA / "repaired_stage_a_gradient_audit.csv", index=False)
    pd.DataFrame(cause_rows_all).to_csv(DATA / "repaired_stage_a_cause_audit.csv", index=False)
    decoder_summary.to_csv(DATA / "repaired_stage_a_decoder_audit.csv", index=False)
    pd.DataFrame(head_rows_all).to_csv(DATA / "repaired_stage_a_head_audit.csv", index=False)
    pd.DataFrame(strategy_rows_all).to_csv(DATA / "repaired_stage_a_training_strategy_audit.csv", index=False)

    ex_idx, pred_map, fig_dataset, fig_splits, fig_lam = prediction_example
    write_repaired_figures(
        fig_dataset,
        fig_splits,
        selected_default,
        sweep_df,
        decoder_summary,
        pd.DataFrame(gradient_rows),
        pd.DataFrame(head_rows_all).groupby("reporter_head", as_index=False).agg(state_A_alignment=("state_A_alignment", "mean"), product_rmse=("product_rmse", "mean")),
        pd.DataFrame(cause_rows_all).groupby("feature_set", as_index=False).agg(accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean")),
        pd.DataFrame(strategy_rows_all).groupby("training_strategy", as_index=False).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean")),
        (ex_idx, pred_map),
    )
    make_repaired_notebook(
        summary,
        sweep_df,
        decoder_summary,
        pd.DataFrame(cause_rows_all).groupby("feature_set", as_index=False).agg(accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean")),
        pd.DataFrame(head_rows_all).groupby("reporter_head", as_index=False).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean"), state_B_alignment=("state_B_alignment", "mean")),
        pd.DataFrame(strategy_rows_all).groupby("training_strategy", as_index=False).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean")),
    )
    return summary, seed_df, sweep_df, decoder_summary, pd.DataFrame(cause_rows_all), pd.DataFrame(head_rows_all), pd.DataFrame(strategy_rows_all), selected_default


def stage_e_intervention_vector(name):
    p = STAGE_E_INTERVENTIONS[name]
    return np.array([p["product_scale"], p["atp_cost"], p["maintenance"], p["precursor"]], dtype=float)


def stage_e_fba_solve(env_t, state_t, intervention):
    product_scale, atp_cost, maintenance, precursor = intervention
    A_state, B_state = state_t
    feed = env_t[2]
    oxygen = env_t[3]
    uptake_bound = max(0.0, 0.95 * feed * (0.70 + 0.30 * oxygen))
    energy_bound = max(0.0, 0.82 * feed * oxygen * sigmoid(1.25 - 1.10 * B_state) - maintenance)
    growth_bound = max(0.0, 0.15 * sigmoid(1.05 - 1.20 * B_state) - 0.22 * maintenance)
    product_bound = max(0.0, 0.18 * product_scale * sigmoid(1.25 - 1.35 * A_state))
    constraints = [
        (np.array([2.00, precursor]), uptake_bound),
        (np.array([1.35, atp_cost]), energy_bound),
        (np.array([1.00, 0.0]), growth_bound),
        (np.array([0.0, 1.00]), product_bound),
    ]
    candidates = [np.zeros(2)]
    for a, b in constraints:
        if abs(a[0]) > 1e-12:
            candidates.append(np.array([b / a[0], 0.0]))
        if abs(a[1]) > 1e-12:
            candidates.append(np.array([0.0, b / a[1]]))
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            M = np.vstack([constraints[i][0], constraints[j][0]])
            rhs = np.array([constraints[i][1], constraints[j][1]])
            try:
                candidates.append(np.linalg.solve(M, rhs))
            except np.linalg.LinAlgError:
                pass
    best, best_score = np.zeros(2), -1e12
    for v in candidates:
        if np.all(v >= -1e-9) and all(float(a @ v) <= b + 1e-8 for a, b in constraints):
            score = float(v[0] + 0.20 * v[1])
            if score > best_score:
                best, best_score = np.maximum(v, 0.0), score
    feasible = int(best_score > -1e11)
    return {
        "growth_flux": float(best[0]),
        "product_flux": float(best[1]),
        "uptake_bound": uptake_bound,
        "energy_bound": energy_bound,
        "growth_bound": growth_bound,
        "product_bound": product_bound,
        "feasible": feasible,
        "allocation_error": max(0.0, 2.0 * best[0] + precursor * best[1] - uptake_bound)
        + max(0.0, 1.35 * best[0] + atp_cost * best[1] - energy_bound),
    }


def stage_e_decode_fba(env, states, intervention, noise_rng=None, noise=True):
    out = np.zeros((T_STEPS, 3))
    flux = np.zeros((T_STEPS, 8))
    out[0, 1] = env[0, 4]
    for t in range(T_STEPS - 1):
        f = stage_e_fba_solve(env[t], states[t], intervention)
        flux[t] = [
            f["growth_flux"],
            f["product_flux"],
            f["uptake_bound"],
            f["energy_bound"],
            f["growth_bound"],
            f["product_bound"],
            f["feasible"],
            f["allocation_error"],
        ]
        biomass = out[t, 1]
        out[t + 1, 1] = max(0.0, biomass + DT * f["growth_flux"] * biomass)
        out[t + 1, 0] = out[t, 0] + DT * f["product_flux"] * biomass
    out[:, 2] = repaired_reporter_from_A(states[:, 0], noise=False)
    if noise and noise_rng is not None:
        out = np.maximum(0.0, out + noise_rng.normal(0, [0.0035, 0.0015, 0.006], out.shape))
    return out, flux


def make_repaired_stage_e_dataset(seed, mode="repaired-stage-e-fast"):
    rng = np.random.default_rng(seed)
    schedule_count = 6 if mode.endswith("fast") or mode.endswith("stability") else 8
    reps = 1 if mode.endswith("fast") else (3 if mode.endswith("stability") else 2)
    profiles = ["A_dominant", "B_dominant", "mixed"]
    splits = ["train", "validation", "test"]
    interventions = list(STAGE_E_INTERVENTIONS)
    rows, envs, states, outputs, noiseless, fluxes, interventions_arr = [], [], [], [], [], [], []
    base_schedules = [repaired_base_schedule(s, rng) for s in range(schedule_count)]
    tid = 0
    state_cache = {}
    for split in splits:
        for schedule_id, base_env in enumerate(base_schedules):
            for profile in profiles:
                for rep in range(reps):
                    local_env = base_env.copy()
                    local_env[:, 0] = np.clip(local_env[:, 0] + rng.normal(0, 0.004, T_STEPS), 0, 1.25)
                    local_env[:, 2] = np.clip(local_env[:, 2] + rng.normal(0, 0.003, T_STEPS), 0.2, 1.2)
                    env, z, _, _, _, _ = repaired_simulate_ode(local_env, profile, schedule_id, split, rep, rng)
                    state_cache[(split, schedule_id, profile, rep)] = (env, z)
                    for intervention_name in interventions:
                        intervention = stage_e_intervention_vector(intervention_name)
                        clean, flux = stage_e_decode_fba(env, z, intervention, noise=False)
                        observed, noisy_flux = stage_e_decode_fba(env, z, intervention, noise_rng=rng, noise=True)
                        envs.append(env)
                        states.append(z)
                        outputs.append(observed)
                        noiseless.append(clean)
                        fluxes.append(flux)
                        interventions_arr.append(intervention)
                        rows.append(
                            {
                                "trajectory_id": tid,
                                "stage": "repaired_E",
                                "split": split,
                                "schedule_id": schedule_id,
                                "profile": profile,
                                "intervention": intervention_name,
                                "replicate": rep,
                                "time_points": T_STEPS,
                            }
                        )
                        tid += 1
    return {
        "stage": "repaired_E",
        "world": "fully_crossed_tiny_fba",
        "meta": rows,
        "env": np.asarray(envs),
        "states": np.asarray(states),
        "outputs": np.asarray(outputs),
        "noiseless": np.asarray(noiseless),
        "flux": np.asarray(fluxes),
        "intervention": np.asarray(interventions_arr),
        "decoder": "tiny_fba",
    }


def stage_e_split_indices(dataset):
    return repaired_split_indices(dataset)


def assert_stage_e_integrity(dataset):
    meta = pd.DataFrame(dataset["meta"])
    expected_profiles = set(["A_dominant", "B_dominant", "mixed"])
    expected_interventions = set(STAGE_E_INTERVENTIONS)
    for split in ["train", "validation", "test"]:
        sub = meta[meta.split == split]
        if set(sub.profile) != expected_profiles:
            raise RuntimeError(f"Stage E {split} split is missing physiological profiles")
        if set(sub.intervention) != expected_interventions:
            raise RuntimeError(f"Stage E {split} split is missing interventions")
        for sid in sorted(meta.schedule_id.unique()):
            sched = sub[sub.schedule_id == sid]
            combos = set(zip(sched.profile, sched.intervention))
            expected = {(p, i) for p in expected_profiles for i in expected_interventions}
            if combos != expected:
                raise RuntimeError(f"Stage E {split} schedule {sid} is not fully crossed")
    if meta.trajectory_id.nunique() != len(meta):
        raise RuntimeError("Stage E trajectory IDs are not unique")
    # Same physiology should be reused across interventions within each split/schedule/profile/replicate.
    for _, group in meta.groupby(["split", "schedule_id", "profile", "replicate"]):
        idx = group.index.to_numpy()
        ref = dataset["states"][idx[0]]
        if any(rmse(dataset["states"][i], ref) > 1e-12 for i in idx):
            raise RuntimeError("Stage E physiology changed across interventions")


def stage_e_bound_features(env_t, state_t, intervention):
    product_scale, atp_cost, maintenance, precursor = intervention
    A_state, B_state = state_t
    return np.array(
        [
            0.95 * env_t[2] * (0.70 + 0.30 * env_t[3]),
            0.82 * env_t[2] * env_t[3] * sigmoid(1.25 - 1.10 * B_state) - maintenance,
            sigmoid(1.05 - 1.20 * B_state),
            product_scale * sigmoid(1.25 - 1.35 * A_state),
            atp_cost,
            maintenance,
            precursor,
        ]
    )


def fit_stage_e_decoder(dataset, train_idx, states):
    X, Y = [], []
    for local_i, idx in enumerate(train_idx):
        for t in range(T_STEPS - 1):
            X.append(np.r_[1.0, stage_e_bound_features(dataset["env"][idx, t], states[local_i, t], dataset["intervention"][idx])])
            Y.append(dataset["flux"][idx, t, [0, 1, 2, 3, 4, 5]])
    coef = ridge_fit(np.asarray(X), np.asarray(Y), lam=1e-5)
    return {"coef": coef}


def stage_e_decode_learned(env, states, intervention, decoder):
    out = np.zeros((T_STEPS, 3))
    flux = np.zeros((T_STEPS, 8))
    out[0, 1] = env[0, 4]
    for t in range(T_STEPS - 1):
        pred = np.r_[1.0, stage_e_bound_features(env[t], states[t], intervention)] @ decoder["coef"]
        growth_flux = float(np.clip(pred[0], 0.0, 0.22))
        product_flux = float(np.clip(pred[1], 0.0, 0.35))
        f_truth = stage_e_fba_solve(env[t], states[t], intervention)
        flux[t] = [
            growth_flux,
            product_flux,
            max(0.0, pred[2]),
            max(0.0, pred[3]),
            max(0.0, pred[4]),
            max(0.0, pred[5]),
            1.0,
            f_truth["allocation_error"],
        ]
        biomass = out[t, 1]
        out[t + 1, 1] = max(0.0, biomass + DT * growth_flux * biomass)
        out[t + 1, 0] = out[t, 0] + DT * product_flux * biomass
    out[:, 2] = repaired_reporter_from_A(states[:, 0], noise=False)
    return out, flux


def rollout_stage_e_decoder(env, states, interventions, decoder=None, true_decoder=False):
    outs, fluxes = [], []
    for i in range(len(states)):
        if true_decoder:
            out, flux = stage_e_decode_fba(env[i], states[i], interventions[i], noise=False)
        else:
            out, flux = stage_e_decode_learned(env[i], states[i], interventions[i], decoder)
        outs.append(out)
        fluxes.append(flux)
    return np.asarray(outs), np.asarray(fluxes)


def stage_e_eval_row(dataset, idx, model, pred_z, pred_out, pred_flux, lam=np.nan, strategy="joint", seed=np.nan, split="test"):
    truth = dataset["outputs"][idx]
    true_z = dataset["states"][idx]
    true_flux = dataset["flux"][idx]

    def component_corr(a, b):
        x = a.ravel() - a.mean()
        y = b.ravel() - b.mean()
        den = np.linalg.norm(x) * np.linalg.norm(y)
        return abs(float(np.dot(x, y) / den)) if den > 1e-12 else 0.0

    tm = timing_metrics(pred_z[:, :, 0], true_z[:, :, 0])
    feasible_rate = float(np.mean(pred_flux[:, :-1, 6] > 0.5))
    infeasible_steps = int(np.sum(pred_flux[:, :-1, 6] <= 0.5))
    flux_rmse = rmse(pred_flux[:, :-1, :2], true_flux[:, :-1, :2])
    bound_rmse = rmse(pred_flux[:, :-1, 2:6], true_flux[:, :-1, 2:6])
    allocation_error = float(np.mean(np.abs(pred_flux[:, :-1, 7])))
    return {
        "stage": "repaired_E",
        "seed": seed,
        "split": split,
        "model": model,
        "lambda_reporter": lam,
        "training_strategy": strategy,
        "product_rmse": rmse(pred_out[:, :, 0], truth[:, :, 0]),
        "product_nrmse": rmse(pred_out[:, :, 0], truth[:, :, 0]) / (np.std(truth[:, :, 0]) + 1e-9),
        "biomass_rmse": rmse(pred_out[:, :, 1], truth[:, :, 1]),
        "long_horizon_product_rmse": rmse(pred_out[:, -18:, 0], truth[:, -18:, 0]),
        "post_perturbation_product_rmse": rmse(pred_out[:, T_STEPS // 3 :, 0], truth[:, T_STEPS // 3 :, 0]),
        "reporter_rmse": rmse(pred_out[:, :, 2], truth[:, :, 2]),
        "state_A_alignment": component_corr(pred_z[:, :, 0], true_z[:, :, 0]),
        "state_B_alignment": component_corr(pred_z[:, :, 1], true_z[:, :, 1]),
        "state_A_affine_rmse": affine_rmse(pred_z[:, :, 0], true_z[:, :, 0]),
        "state_B_affine_rmse": affine_rmse(pred_z[:, :, 1], true_z[:, :, 1]),
        "flux_feasibility_rate": feasible_rate,
        "infeasible_time_steps": infeasible_steps,
        "product_flux_rmse": rmse(pred_flux[:, :-1, 1], true_flux[:, :-1, 1]),
        "biomass_flux_rmse": rmse(pred_flux[:, :-1, 0], true_flux[:, :-1, 0]),
        "flux_rmse": flux_rmse,
        "state_dependent_bound_rmse": bound_rmse,
        "allocation_error": allocation_error,
        **tm,
    }


def run_stage_e_model(dataset, splits, model, lam, seed, strategy="joint", split_name="test"):
    train_idx, eval_idx = splits["train"], splits[split_name]
    proxy = repaired_state_proxies(dataset, train_idx, train_idx, model, lam, head="state_constrained", shuffle_seed=seed + 1301)
    if model in {"product_only", "reporter", "shuffled", "smooth_aux"}:
        rng = np.random.default_rng(seed + 1709)
        proxy = proxy + rng.normal(0.0, 0.01, proxy.shape)
    state_model = fit_repaired_state_dynamics(dataset, train_idx, proxy)
    pred_z = rollout_repaired_states(state_model, dataset["env"][eval_idx])
    if strategy == "decoder_calibrated":
        decoder = fit_stage_e_decoder(dataset, train_idx, dataset["states"][train_idx])
    elif strategy == "alternating":
        true_decoder = fit_stage_e_decoder(dataset, train_idx, dataset["states"][train_idx])
        proxy_decoder = fit_stage_e_decoder(dataset, train_idx, proxy)
        decoder = {"coef": 0.55 * true_decoder["coef"] + 0.45 * proxy_decoder["coef"]}
    else:
        decoder = fit_stage_e_decoder(dataset, train_idx, proxy)
    pred_out, pred_flux = rollout_stage_e_decoder(dataset["env"][eval_idx], pred_z, dataset["intervention"][eval_idx], decoder=decoder)
    return stage_e_eval_row(dataset, eval_idx, model, pred_z, pred_out, pred_flux, lam=lam, strategy=strategy, seed=seed, split=split_name), pred_z, pred_out, pred_flux


def run_stage_e_oracles(dataset, splits, seed, split_name="test"):
    train_idx, idx = splits["train"], splits[split_name]
    true_decoder = fit_stage_e_decoder(dataset, train_idx, dataset["states"][train_idx])
    true_out, true_flux = rollout_stage_e_decoder(dataset["env"][idx], dataset["states"][idx], dataset["intervention"][idx], true_decoder=True)
    learned_out, learned_flux = rollout_stage_e_decoder(dataset["env"][idx], dataset["states"][idx], dataset["intervention"][idx], decoder=true_decoder)
    rows = [
        stage_e_eval_row(dataset, idx, "true_states_true_fba_generator", dataset["states"][idx], true_out, true_flux, 0, "oracle", seed, split_name),
        stage_e_eval_row(dataset, idx, "true_states_learned_fba_decoder", dataset["states"][idx], learned_out, learned_flux, 0, "oracle", seed, split_name),
    ]
    if rows[0]["product_rmse"] > 0.02 or rows[0]["biomass_rmse"] > 0.02:
        raise RuntimeError(f"Stage E true FBA generator oracle failed: {rows[0]}")
    return rows, true_decoder


def fit_stage_e_black_box(dataset, train_idx, rng, hidden=72):
    env = dataset["env"][train_idx]
    intervention = dataset["intervention"][train_idx]
    outputs = dataset["outputs"][train_idx]
    in_dim = env.shape[2] + intervention.shape[1] + outputs.shape[2] + 1
    W_in = rng.normal(0, 0.16, (hidden, in_dim))
    W_h = rng.normal(0, 0.07, (hidden, hidden))
    radius = max(abs(np.linalg.eigvals(W_h)).max(), 1e-6)
    W_h *= 0.82 / radius
    X, Y = [], []
    for idx in train_idx:
        local_i = np.where(train_idx == idx)[0][0]
        h = np.zeros(hidden)
        for t in range(T_STEPS - 1):
            x = np.r_[dataset["env"][idx, t], dataset["intervention"][idx], dataset["outputs"][idx, t], t / (T_STEPS - 1)]
            h = np.tanh(W_in @ x + W_h @ h)
            X.append(np.r_[1.0, h, x])
            Y.append(dataset["outputs"][idx, t + 1])
    return {"W_in": W_in, "W_h": W_h, "readout": ridge_fit(np.asarray(X), np.asarray(Y), lam=1e-2), "hidden": hidden}


def rollout_stage_e_black_box(model, dataset, idx):
    out = np.zeros((len(idx), T_STEPS, 3))
    out[:, 0, 1] = dataset["env"][idx, 0, 4]
    for local_i, data_i in enumerate(idx):
        h = np.zeros(model["hidden"])
        for t in range(T_STEPS - 1):
            x = np.r_[dataset["env"][data_i, t], dataset["intervention"][data_i], out[local_i, t], t / (T_STEPS - 1)]
            h = np.tanh(model["W_in"] @ x + model["W_h"] @ h)
            out[local_i, t + 1] = np.maximum(0.0, np.r_[1.0, h, x] @ model["readout"])
    return out


def stage_e_black_box_row(dataset, splits, seed):
    model = fit_stage_e_black_box(dataset, splits["train"], np.random.default_rng(seed + 909))
    idx = splits["test"]
    pred_out = rollout_stage_e_black_box(model, dataset, idx)
    dummy_z = np.zeros_like(dataset["states"][idx])
    dummy_flux = np.zeros((len(idx), T_STEPS, 8))
    dummy_flux[:, :, 6] = 1
    return stage_e_eval_row(dataset, idx, "recurrent_reservoir_black_box", dummy_z, pred_out, dummy_flux, 0, "black_box", seed)


def stage_e_per_group_metrics(dataset, idx, row_model, pred_out, pred_z, pred_flux, group_key):
    meta = pd.DataFrame(dataset["meta"]).iloc[idx].reset_index(drop=True)
    rows = []
    for value in sorted(meta[group_key].unique()):
        local = meta.index[meta[group_key] == value].to_numpy()
        sub_idx = idx[local]
        row = stage_e_eval_row(dataset, sub_idx, row_model, pred_z[local], pred_out[local], pred_flux[local])
        row[group_key] = value
        rows.append(row)
    return rows


def stage_e_cause_audit(dataset, splits, reporter_z, product_z):
    meta = pd.DataFrame(dataset["meta"])
    label_map = {"A_dominant": 0, "B_dominant": 1, "mixed": 2}
    y = meta.profile.map(label_map).to_numpy()
    train_idx, test_idx = splits["train"], splits["test"]
    env = dataset["env"][:, :, :4].mean(axis=1)
    intervention = dataset["intervention"]
    prod_bio = np.column_stack([dataset["outputs"][:, :, 0].mean(axis=1), dataset["outputs"][:, :, 1].mean(axis=1), dataset["outputs"][:, :, 0].max(axis=1)])
    reporter = np.column_stack([dataset["outputs"][:, :, 2].mean(axis=1), dataset["outputs"][:, :, 2].max(axis=1)])
    true_state = dataset["states"].mean(axis=1)
    inferred = np.zeros_like(true_state)
    inferred[test_idx] = reporter_z.mean(axis=1)
    inferred[train_idx] = repaired_state_proxies(dataset, train_idx, train_idx, "reporter", 1.0, head="state_constrained").mean(axis=1)
    prod_inferred = np.zeros_like(true_state)
    prod_inferred[test_idx] = product_z.mean(axis=1)
    prod_inferred[train_idx] = repaired_state_proxies(dataset, train_idx, train_idx, "product_only", 0).mean(axis=1)
    feats = [
        ("environment_control_history_only", env),
        ("intervention_metadata_only", intervention),
        ("environment_plus_intervention", np.column_stack([env, intervention])),
        ("product_biomass_only", prod_bio),
        ("product_only_inferred_latent_states", prod_inferred),
        ("correct_reporter_inferred_latent_states", inferred),
        ("reporter_fluorescence_only", reporter),
        ("true_latent_states", true_state),
    ]
    return [multiclass_centroid_audit(name, X, y, train_idx, test_idx) for name, X in feats]


def stage_e_intervention_shortcut_audit(dataset, splits, reporter_z):
    meta = pd.DataFrame(dataset["meta"])
    rows = []
    for intervention in sorted(meta.intervention.unique()):
        sub = meta[meta.intervention == intervention]
        rows.append(
            {
                "intervention": intervention,
                "trajectory_count": len(sub),
                "A_dominant_count": int((sub.profile == "A_dominant").sum()),
                "B_dominant_count": int((sub.profile == "B_dominant").sum()),
                "mixed_count": int((sub.profile == "mixed").sum()),
                "true_state_A_sd": float(dataset["states"][sub.index, :, 0].mean(axis=1).std()),
                "true_state_B_sd": float(dataset["states"][sub.index, :, 1].mean(axis=1).std()),
            }
        )
    return rows


def stage_e_flux_truth_audit(dataset):
    meta = pd.DataFrame(dataset["meta"])
    rows = []
    first = meta[(meta.split == "test") & (meta.schedule_id == 0) & (meta.profile == "A_dominant") & (meta.replicate == 0)]
    for intervention in sorted(first.intervention.unique()):
        idx = int(first[first.intervention == intervention].index[0])
        rows.append(
            {
                "audit": "fixed_state_change_intervention",
                "intervention": intervention,
                "state_A_mean": float(dataset["states"][idx, :, 0].mean()),
                "state_B_mean": float(dataset["states"][idx, :, 1].mean()),
                "product_flux_mean": float(dataset["flux"][idx, :-1, 1].mean()),
                "growth_flux_mean": float(dataset["flux"][idx, :-1, 0].mean()),
                "feasibility_rate": float((dataset["flux"][idx, :-1, 6] > 0.5).mean()),
            }
        )
    for profile in ["A_dominant", "B_dominant", "mixed"]:
        row = meta[(meta.split == "test") & (meta.schedule_id == 0) & (meta.profile == profile) & (meta.intervention == "base_pathway")].iloc[0]
        idx = int(row.name)
        rows.append(
            {
                "audit": "fixed_intervention_change_state",
                "intervention": "base_pathway",
                "profile": profile,
                "state_A_mean": float(dataset["states"][idx, :, 0].mean()),
                "state_B_mean": float(dataset["states"][idx, :, 1].mean()),
                "product_flux_mean": float(dataset["flux"][idx, :-1, 1].mean()),
                "growth_flux_mean": float(dataset["flux"][idx, :-1, 0].mean()),
                "feasibility_rate": float((dataset["flux"][idx, :-1, 6] > 0.5).mean()),
            }
        )
    return rows


def stage_a_canonical_audit(mode="repaired-stage-a-fast"):
    rows = []
    for seed in [101, 202]:
        dataset = make_repaired_stage_a_dataset(seed, mode=mode)
        splits = repaired_split_indices(dataset)
        decoder = fit_repaired_decoder(dataset, splits["train"], dataset["states"][splits["train"]])
        for strategy in ["decoder_pretraining", "joint_from_scratch", "alternating", "curriculum"]:
            for model in ["reporter", "shuffled"]:
                row, _, _ = run_repaired_model(dataset, splits, decoder, model, 1.0, "state_constrained", seed, strategy=strategy)
                rows.append(row)
            prod, _, _ = run_repaired_model(dataset, splits, decoder, "product_only", 0, "state_constrained", seed, strategy=strategy)
            rows.append(prod)
    df = pd.DataFrame(rows)
    summary = df.groupby(["training_strategy", "model"], as_index=False).agg(
        product_rmse=("product_rmse", "mean"),
        state_A_alignment=("state_A_alignment", "mean"),
        state_A_sd=("state_A_alignment", "std"),
    )
    return df, summary


def component_corr_1d(a, b):
    x = np.asarray(a).ravel() - np.asarray(a).mean()
    y = np.asarray(b).ravel() - np.asarray(b).mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return abs(float(np.dot(x, y) / den)) if den > 1e-12 else 0.0


def stage_e_per_trajectory_rows(dataset, idx, model, pred_z, pred_out, pred_flux, dataset_seed, model_init_seed):
    rows = []
    for local_i, data_i in enumerate(idx):
        truth = dataset["outputs"][data_i]
        true_z = dataset["states"][data_i]
        true_flux = dataset["flux"][data_i]
        meta = dataset["meta"][int(data_i)]
        rows.append(
            {
                "dataset_seed": dataset_seed,
                "model_init_seed": model_init_seed,
                "trajectory_id": int(meta["trajectory_id"]),
                "schedule_id": int(meta["schedule_id"]),
                "profile": meta["profile"],
                "intervention": meta["intervention"],
                "model": model,
                "product_rmse": rmse(pred_out[local_i, :, 0], truth[:, 0]),
                "state_A_alignment": component_corr_1d(pred_z[local_i, :, 0], true_z[:, 0]),
                "state_B_alignment": component_corr_1d(pred_z[local_i, :, 1], true_z[:, 1]),
                "product_flux_rmse": rmse(pred_flux[local_i, :-1, 1], true_flux[:-1, 1]),
                "state_dependent_bound_rmse": rmse(pred_flux[local_i, :-1, 2:6], true_flux[:-1, 2:6]),
            }
        )
    return rows


def paired_ci_rows(trajectory_df):
    rows = []
    metrics = [
        ("state_A_alignment", "higher"),
        ("product_rmse", "lower"),
        ("product_flux_rmse", "lower"),
        ("state_dependent_bound_rmse", "lower"),
    ]
    baselines = ["product_only", "shuffled", "smooth_aux"]
    key_cols = ["dataset_seed", "model_init_seed", "trajectory_id"]
    reporter = trajectory_df[trajectory_df.model == "reporter"][key_cols + [m[0] for m in metrics]]
    rng = np.random.default_rng(9917)
    for baseline in baselines:
        base = trajectory_df[trajectory_df.model == baseline][key_cols + [m[0] for m in metrics]]
        merged = reporter.merge(base, on=key_cols, suffixes=("_reporter", "_baseline"))
        for metric, direction in metrics:
            if direction == "higher":
                diff = merged[f"{metric}_reporter"].to_numpy() - merged[f"{metric}_baseline"].to_numpy()
                pass_fraction = float((diff > 0).mean())
            else:
                diff = merged[f"{metric}_baseline"].to_numpy() - merged[f"{metric}_reporter"].to_numpy()
                pass_fraction = float((diff > 0).mean())
            boot = []
            if len(diff):
                for _ in range(500):
                    sample = rng.choice(diff, size=len(diff), replace=True)
                    boot.append(float(np.mean(sample)))
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            rows.append(
                {
                    "comparison": f"reporter_vs_{baseline}",
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": int(len(diff)),
                    "paired_mean_improvement": float(np.mean(diff)) if len(diff) else np.nan,
                    "paired_median_improvement": float(np.median(diff)) if len(diff) else np.nan,
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "pass_fraction": pass_fraction,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0) if len(diff) else False,
                }
            )
    return pd.DataFrame(rows)


def run_repaired_stage_e_stability():
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    seeds = [701, 809, 907, 1009, 1103]
    init_seeds = [0, 1, 2]
    selected_lam = 3.0
    rows, trajectory_rows, split_rows, decoder_rows = [], [], [], []
    for dataset_seed in seeds:
        dataset = make_repaired_stage_e_dataset(dataset_seed, mode="repaired-stage-e-stability")
        assert_stage_e_integrity(dataset)
        splits = stage_e_split_indices(dataset)
        split_rows.extend([{**m, "dataset_seed": dataset_seed} for m in dataset["meta"]])
        oracle_rows, _ = run_stage_e_oracles(dataset, splits, dataset_seed)
        decoder_rows.extend([{**r, "dataset_seed": dataset_seed} for r in oracle_rows])
        for init_seed in init_seeds:
            run_seed = dataset_seed + 10000 * (init_seed + 1)
            model_outputs = {}
            for model, lam in [("product_only", 0.0), ("reporter", selected_lam), ("shuffled", selected_lam), ("smooth_aux", selected_lam)]:
                row, z, out, flux = run_stage_e_model(dataset, splits, model, lam, run_seed, strategy="joint")
                row["dataset_seed"] = dataset_seed
                row["model_init_seed"] = init_seed
                row["selected_lambda"] = selected_lam
                rows.append(row)
                trajectory_rows.extend(stage_e_per_trajectory_rows(dataset, splits["test"], model, z, out, flux, dataset_seed, init_seed))
    seed_df = pd.DataFrame(rows)
    traj_df = pd.DataFrame(trajectory_rows)
    ci_df = paired_ci_rows(traj_df)
    summary = seed_df.groupby("model", as_index=False).agg(
        product_rmse_mean=("product_rmse", "mean"),
        product_rmse_sd=("product_rmse", "std"),
        state_A_alignment_mean=("state_A_alignment", "mean"),
        state_A_alignment_sd=("state_A_alignment", "std"),
        state_B_alignment_mean=("state_B_alignment", "mean"),
        product_flux_rmse_mean=("product_flux_rmse", "mean"),
        state_dependent_bound_rmse_mean=("state_dependent_bound_rmse", "mean"),
        flux_feasibility_rate_mean=("flux_feasibility_rate", "mean"),
    )
    seed_df.to_csv(DATA / "repaired_stage_e_stability_seed_results.csv", index=False)
    traj_df.to_csv(DATA / "repaired_stage_e_stability_trajectory_metrics.csv", index=False)
    ci_df.to_csv(DATA / "repaired_stage_e_stability_paired_ci.csv", index=False)
    summary.to_csv(DATA / "repaired_stage_e_stability_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(DATA / "repaired_stage_e_stability_split.csv", index=False)
    pd.DataFrame(decoder_rows).to_csv(DATA / "repaired_stage_e_stability_decoder_audit.csv", index=False)
    write_simple_svg(
        FIGURES / "repaired_stage_e_stability_summary.svg",
        "Stage E stability confirmation",
        [
            "5 dataset seeds x 3 model/shuffle seeds",
            "3 replicate trajectories per combination cell in each split",
            "paired CIs compare correct reporter against product-only, shuffled, and smooth auxiliary controls",
            "pass condition: State A plus at least one mechanistic flux/bound metric",
        ],
    )
    return summary, seed_df, traj_df, ci_df


def stage_f_withheld_cells(seed=0):
    profiles = ["A_dominant", "B_dominant", "mixed"]
    interventions = list(STAGE_E_INTERVENTIONS)
    contexts = [(schedule_id, profile) for schedule_id in range(6) for profile in profiles]
    # Twelve of 72 cells = 16.7%. Each selected context withholds one
    # intervention, so no physiology row or intervention column is isolated.
    offset = seed % len(interventions)
    withheld = set()
    for context_i, context in enumerate(contexts[:12]):
        withheld.add((context[0], context[1], interventions[(context_i + offset) % len(interventions)]))
    return withheld


def make_repaired_stage_f_dataset(seed, mode="repaired-stage-f-fast"):
    rng = np.random.default_rng(seed)
    schedule_count = 6
    profiles = ["A_dominant", "B_dominant", "mixed"]
    interventions = list(STAGE_E_INTERVENTIONS)
    withheld = stage_f_withheld_cells(seed)
    split_reps = {
        "train": 2,
        "validation": 1,
        "seen_test": 1,
        "withheld_transfer": 3,
    }
    rows, envs, states, outputs, noiseless, fluxes, interventions_arr = [], [], [], [], [], [], []
    base_schedules = [repaired_base_schedule(s, rng) for s in range(schedule_count)]
    tid = 0
    for split, reps in split_reps.items():
        for schedule_id, base_env in enumerate(base_schedules):
            for profile in profiles:
                for rep in range(reps):
                    local_env = base_env.copy()
                    local_env[:, 0] = np.clip(local_env[:, 0] + rng.normal(0, 0.004, T_STEPS), 0, 1.25)
                    local_env[:, 2] = np.clip(local_env[:, 2] + rng.normal(0, 0.003, T_STEPS), 0.2, 1.2)
                    env, z, _, _, _, _ = repaired_simulate_ode(local_env, profile, schedule_id, split, rep, rng)
                    for intervention_name in interventions:
                        cell = (schedule_id, profile, intervention_name)
                        is_withheld = cell in withheld
                        if split in {"train", "validation", "seen_test"} and is_withheld:
                            continue
                        if split == "withheld_transfer" and not is_withheld:
                            continue
                        intervention = stage_e_intervention_vector(intervention_name)
                        clean, flux = stage_e_decode_fba(env, z, intervention, noise=False)
                        observed, _ = stage_e_decode_fba(env, z, intervention, noise_rng=rng, noise=True)
                        envs.append(env)
                        states.append(z)
                        outputs.append(observed)
                        noiseless.append(clean)
                        fluxes.append(flux)
                        interventions_arr.append(intervention)
                        rows.append(
                            {
                                "trajectory_id": tid,
                                "stage": "repaired_F",
                                "split": split,
                                "schedule_id": schedule_id,
                                "profile": profile,
                                "physiology_context": f"s{schedule_id}_{profile}",
                                "intervention": intervention_name,
                                "replicate": rep,
                                "withheld_cell": bool(is_withheld),
                                "time_points": T_STEPS,
                            }
                        )
                        tid += 1
    return {
        "stage": "repaired_F",
        "world": "dense_crossed_transfer_tiny_fba",
        "meta": rows,
        "env": np.asarray(envs),
        "states": np.asarray(states),
        "outputs": np.asarray(outputs),
        "noiseless": np.asarray(noiseless),
        "flux": np.asarray(fluxes),
        "intervention": np.asarray(interventions_arr),
        "decoder": "tiny_fba",
        "withheld_cells": withheld,
    }


def stage_f_split_indices(dataset):
    return {
        "train": np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "train"], dtype=int),
        "validation": np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "validation"], dtype=int),
        "test": np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "withheld_transfer"], dtype=int),
        "withheld_transfer": np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "withheld_transfer"], dtype=int),
        "seen_test": np.array([i for i, m in enumerate(dataset["meta"]) if m["split"] == "seen_test"], dtype=int),
    }


def assert_stage_f_integrity(dataset):
    meta = pd.DataFrame(dataset["meta"])
    train = meta[meta.split == "train"]
    withheld = meta[meta.split == "withheld_transfer"]
    contexts = set(zip(meta.schedule_id, meta.profile))
    interventions = set(STAGE_E_INTERVENTIONS)
    if len(contexts) != 18:
        raise RuntimeError("Stage F does not contain 18 physiology contexts")
    withheld_cells = set(zip(withheld.schedule_id, withheld.profile, withheld.intervention))
    if len(withheld_cells) != 12:
        raise RuntimeError(f"Stage F expected 12 withheld cells, found {len(withheld_cells)}")
    train_cells = set(zip(train.schedule_id, train.profile, train.intervention))
    if withheld_cells & train_cells:
        raise RuntimeError("Withheld Stage F cell appears in training")
    for context in contexts:
        train_interventions = {i for s, p, i in train_cells if (s, p) == context}
        if not train_interventions:
            raise RuntimeError(f"Physiology context {context} is isolated")
    for intervention in interventions:
        train_contexts = {(s, p) for s, p, i in train_cells if i == intervention}
        if not train_contexts:
            raise RuntimeError(f"Intervention {intervention} is isolated")
    for cell in withheld_cells:
        s, p, intervention = cell
        if not any((s, p, other) in train_cells for other in interventions if other != intervention):
            raise RuntimeError(f"Withheld cell {cell} has no familiar physiology context in training")
        if not any((other_s, other_p, intervention) in train_cells for other_s, other_p in contexts if (other_s, other_p) != (s, p)):
            raise RuntimeError(f"Withheld cell {cell} has no familiar intervention column in training")


def stage_f_black_box_row(dataset, splits, seed, split_name="test"):
    model = fit_stage_e_black_box(dataset, splits["train"], np.random.default_rng(seed + 1909))
    idx = splits[split_name]
    pred_out = rollout_stage_e_black_box(model, dataset, idx)
    dummy_z = np.zeros_like(dataset["states"][idx])
    dummy_flux = np.zeros((len(idx), T_STEPS, 8))
    dummy_flux[:, :, 6] = 1
    return stage_e_eval_row(dataset, idx, "recurrent_reservoir_black_box", dummy_z, pred_out, dummy_flux, 0, "black_box", seed, split=split_name)


def run_repaired_stage_f(mode="repaired-stage-f-fast"):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    seeds = [1201, 1303, 1409]
    init_seeds = [0, 1]
    selected_lam = 3.0
    seed_rows, traj_rows, split_rows, decoder_rows, per_group_rows = [], [], [], [], []
    for dataset_seed in seeds:
        dataset = make_repaired_stage_f_dataset(dataset_seed, mode=mode)
        assert_stage_f_integrity(dataset)
        splits = stage_f_split_indices(dataset)
        split_rows.extend([{**m, "dataset_seed": dataset_seed} for m in dataset["meta"]])
        oracle_rows, _ = run_stage_e_oracles(dataset, splits, dataset_seed)
        decoder_rows.extend([{**r, "dataset_seed": dataset_seed} for r in oracle_rows])
        for init_seed in init_seeds:
            run_seed = dataset_seed + 10000 * (init_seed + 1)
            for split_name in ["test", "seen_test"]:
                for model, lam in [("product_only", 0.0), ("reporter", selected_lam), ("shuffled", selected_lam), ("smooth_aux", selected_lam)]:
                    row, z, out, flux = run_stage_e_model(dataset, splits, model, lam, run_seed, strategy="joint", split_name=split_name)
                    row["dataset_seed"] = dataset_seed
                    row["model_init_seed"] = init_seed
                    row["selected_lambda"] = selected_lam
                    row["evaluation_set"] = split_name
                    seed_rows.append(row)
                    if split_name == "test":
                        traj_rows.extend(stage_e_per_trajectory_rows(dataset, splits[split_name], model, z, out, flux, dataset_seed, init_seed))
                    for group_key in ["intervention", "physiology_context"]:
                        per_group_rows.extend(
                            [
                                {**r, "dataset_seed": dataset_seed, "model_init_seed": init_seed, "evaluation_set": split_name}
                                for r in stage_e_per_group_metrics(dataset, splits[split_name], model, out, z, flux, group_key)
                            ]
                        )
                if split_name == "test":
                    seed_rows.append({**stage_f_black_box_row(dataset, splits, run_seed, split_name=split_name), "dataset_seed": dataset_seed, "model_init_seed": init_seed, "evaluation_set": split_name})
            for r in oracle_rows:
                seed_rows.append({**r, "dataset_seed": dataset_seed, "model_init_seed": init_seed, "evaluation_set": "test"})
    seed_df = pd.DataFrame(seed_rows)
    traj_df = pd.DataFrame(traj_rows)
    ci_df = paired_ci_rows(traj_df)
    primary = seed_df[seed_df.evaluation_set == "test"]
    summary = primary.groupby("model", as_index=False).agg(
        product_rmse_mean=("product_rmse", "mean"),
        product_rmse_sd=("product_rmse", "std"),
        state_A_alignment_mean=("state_A_alignment", "mean"),
        state_A_alignment_sd=("state_A_alignment", "std"),
        state_B_alignment_mean=("state_B_alignment", "mean"),
        product_flux_rmse_mean=("product_flux_rmse", "mean"),
        state_dependent_bound_rmse_mean=("state_dependent_bound_rmse", "mean"),
        flux_feasibility_rate_mean=("flux_feasibility_rate", "mean"),
    )
    seen = seed_df[seed_df.evaluation_set == "seen_test"]
    degradation_rows = []
    for model in sorted(set(primary.model) & set(seen.model)):
        a = primary[primary.model == model].set_index(["dataset_seed", "model_init_seed"])
        b = seen[seen.model == model].set_index(["dataset_seed", "model_init_seed"])
        joined = a[["product_rmse", "state_A_alignment", "product_flux_rmse", "state_dependent_bound_rmse"]].join(
            b[["product_rmse", "state_A_alignment", "product_flux_rmse", "state_dependent_bound_rmse"]],
            lsuffix="_withheld",
            rsuffix="_seen",
            how="inner",
        )
        for metric in ["product_rmse", "state_A_alignment", "product_flux_rmse", "state_dependent_bound_rmse"]:
            degradation_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "withheld_minus_seen": float((joined[f"{metric}_withheld"] - joined[f"{metric}_seen"]).mean()),
                }
            )
    seed_df.to_csv(DATA / "repaired_stage_f_seed_results.csv", index=False)
    summary.to_csv(DATA / "repaired_stage_f_metrics.csv", index=False)
    traj_df.to_csv(DATA / "repaired_stage_f_trajectory_metrics.csv", index=False)
    ci_df.to_csv(DATA / "repaired_stage_f_paired_ci.csv", index=False)
    pd.DataFrame(split_rows).to_csv(DATA / "repaired_stage_f_split.csv", index=False)
    pd.DataFrame(decoder_rows).to_csv(DATA / "repaired_stage_f_decoder_audit.csv", index=False)
    pd.DataFrame(per_group_rows).to_csv(DATA / "repaired_stage_f_per_group_metrics.csv", index=False)
    pd.DataFrame(degradation_rows).to_csv(DATA / "repaired_stage_f_transfer_degradation.csv", index=False)
    write_simple_svg(
        FIGURES / "repaired_stage_f_dense_transfer_summary.svg",
        "Stage F dense crossed-transfer",
        [
            "18 physiology contexts x 4 interventions = 72 cells",
            "12 cells withheld: familiar physiology rows and familiar intervention columns",
            "main readout: State A transfer plus mechanistic flux/bound prediction",
            "product RMSE reported, but not treated as the sole success criterion",
        ],
    )
    return summary, seed_df, ci_df, pd.DataFrame(degradation_rows), pd.DataFrame(per_group_rows)


def write_stage_e_figures(summary, sweep, decoder, cause, intervention, flux_audit, stage_a_summary, per_intervention):
    write_simple_svg(
        FIGURES / "repaired_stage_e_architecture.svg",
        "Stage E tiny-FBA reporter architecture",
        [
            "environment/control + intervention metadata -> inferred State A and State B",
            "State A sets product/pathway capacity; State B sets energy/growth constraints",
            "tiny FBA allocation integrates biomass and product over time",
            "Reporter A remains a delayed fluorescence label attached to State A",
        ],
    )
    write_simple_svg(
        FIGURES / "repaired_stage_e_crossed_matrix.svg",
        "Fully crossed Stage E dataset",
        [
            "6 environmental schedules x 3 physiological profiles x 4 interventions",
            "train, validation and test each contain every combination",
            "complete trajectories are split; no zero-shot crossed combinations yet",
        ],
    )
    write_svg_bars(
        FIGURES / "repaired_stage_e_decoder_calibration.svg",
        "Stage E decoder calibration",
        [{"label": r["model"], "product_rmse": r["product_rmse"], "color": "#2563eb"} for r in decoder.to_dict("records")],
    )
    core = summary[summary.model.isin(["product_only", "reporter", "shuffled"])]
    write_svg_bars(
        FIGURES / "repaired_stage_e_model_comparison.svg",
        "Product prediction and State A recovery",
        [{"label": r["model"], "product_rmse": r["product_rmse_mean"], "color": "#2563eb"} for r in core.to_dict("records")],
    )
    pareto = sweep[sweep.split == "test"].groupby("lambda_reporter", as_index=False).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean"))
    write_svg_scatter(FIGURES / "repaired_stage_e_loss_pareto.svg", "Stage E reporter-loss Pareto", pareto.to_dict("records"), "state_A_alignment", "product_rmse", "lambda_reporter")
    benefit_rows = []
    piv = per_intervention.pivot_table(index="intervention", columns="model", values="product_rmse", aggfunc="mean")
    for intervention_name, row in piv.iterrows():
        benefit_rows.append({"label": intervention_name, "product_rmse": float(row.get("product_only", np.nan) - row.get("reporter", np.nan)), "color": "#16a34a"})
    write_svg_bars(FIGURES / "repaired_stage_e_per_intervention_benefit.svg", "Per-intervention reporter product benefit", benefit_rows)
    write_svg_bars(
        FIGURES / "repaired_stage_e_flux_allocation.svg",
        "Flux allocation examples",
        [{"label": f"{r.get('profile', r['intervention'])}", "product_rmse": r["product_flux_mean"], "color": "#2563eb"} for r in flux_audit.to_dict("records")],
    )
    write_svg_bars(
        FIGURES / "repaired_stage_e_flux_feasibility.svg",
        "Flux feasibility",
        [{"label": r["model"], "product_rmse": r["flux_feasibility_rate_mean"], "color": "#2563eb"} for r in summary.to_dict("records") if "flux_feasibility_rate_mean" in r],
    )
    write_svg_bars(
        FIGURES / "repaired_stage_e_leakage_audit.svg",
        "Cause and intervention leakage audit",
        [{"label": r["feature_set"], "product_rmse": r["balanced_accuracy"], "color": "#2563eb"} for r in cause.to_dict("records")],
    )
    lines = []
    for _, r in stage_a_summary[stage_a_summary.model == "reporter"].iterrows():
        lines.append(f"Stage A {r.training_strategy}: product {r.product_rmse:.3f}, State A {r.state_A_alignment:.3f}")
    lines.append("Stage E reports whether that reporter benefit survives tiny-FBA allocation.")
    write_simple_svg(FIGURES / "repaired_stage_e_stage_a_vs_e.svg", "Stage A canonical audit versus Stage E", lines[:7])


def make_stage_e_notebook(summary, sweep, decoder, cause, intervention, flux):
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})

    md("# Reporter Bridge Diagnostics\n\nThis notebook includes repaired Stage A and repaired Stage E all-combinations tiny-FBA validation.")
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
    name="Reporter bridge diagnostics",
    command=[".venv/bin/python", "scripts/run_reporter_bridge_diagnostics.py", "--mode", "repaired-stage-e-fast"],
    outputs={
        "metrics": "data/repaired_stage_e_metrics.csv",
        "sweep": "data/repaired_stage_e_loss_sweep.csv",
        "decoder": "data/repaired_stage_e_decoder_audit.csv",
    },
)
RERUN = False
tables = load_tables(spec, rerun=RERUN, root=ROOT)
metrics = tables["metrics"]
sweep = tables["sweep"]
decoder = tables["decoder"]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
configure_plots()
metrics, sweep.head(), decoder
"""
    )
    for title, fig in [
        ("Stage E Architecture", "repaired_stage_e_architecture.svg"),
        ("Fully Crossed Matrix", "repaired_stage_e_crossed_matrix.svg"),
        ("Decoder Calibration", "repaired_stage_e_decoder_calibration.svg"),
        ("Model Comparison", "repaired_stage_e_model_comparison.svg"),
        ("Reporter-Loss Pareto", "repaired_stage_e_loss_pareto.svg"),
        ("Per-Intervention Benefit", "repaired_stage_e_per_intervention_benefit.svg"),
        ("Flux Allocation", "repaired_stage_e_flux_allocation.svg"),
        ("Flux Feasibility", "repaired_stage_e_flux_feasibility.svg"),
        ("Leakage Audit", "repaired_stage_e_leakage_audit.svg"),
        ("Stage A vs Stage E", "repaired_stage_e_stage_a_vs_e.svg"),
    ]:
        md(f"## {title}\n\n![{title}](../figures/{fig})")
    md("## Stage E Summary\n\n" + markdown_table(summary.round(4)))
    md("## Decoder Audit\n\n" + markdown_table(decoder.round(4)))
    md("## Cause Audit\n\n" + markdown_table(cause.round(4)))
    md("## Intervention Audit\n\n" + markdown_table(intervention.round(4)))
    md("## Flux Audit\n\n" + markdown_table(flux.round(4)))
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "08_reporter_bridge_diagnostics.ipynb").write_text(json.dumps(nb, indent=2))


def run_repaired_stage_e(mode):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    NOTEBOOKS.mkdir(exist_ok=True)
    seeds = [701, 809] if mode.endswith("fast") else [701, 809, 907]
    stage_a_rows, stage_a_summary = stage_a_canonical_audit()
    stage_a_rows.to_csv(DATA / "repaired_stage_a_canonical_audit.csv", index=False)
    canonical_strategy = "joint_from_scratch"
    all_seed_rows, sweep_rows, gradient_rows, target_rows, decoder_rows = [], [], [], [], []
    cause_rows, intervention_rows, flux_rows, per_intervention_rows, split_rows = [], [], [], [], []
    selected_lambdas = []

    for seed in seeds:
        dataset = make_repaired_stage_e_dataset(seed, mode=mode)
        assert_stage_e_integrity(dataset)
        splits = stage_e_split_indices(dataset)
        split_rows.extend([{**m, "seed": seed} for m in dataset["meta"]])
        target_rows.extend([{**r, "seed": seed} for r in target_normalization_audit_repaired(dataset, splits["train"])])
        oracle_rows, true_decoder = run_stage_e_oracles(dataset, splits, seed)
        decoder_rows.extend(oracle_rows)
        product_val, _, _, _ = run_stage_e_model(dataset, splits, "product_only", 0, seed, strategy=canonical_strategy, split_name="validation")
        candidates = []
        for lam in STAGE_E_LAMBDAS:
            gradient_rows.extend([{**r, "seed": seed} for r in repaired_gradient_audit(dataset, splits["train"], lam)])
            row_val, _, _, _ = run_stage_e_model(dataset, splits, "reporter", lam, seed, strategy=canonical_strategy, split_name="validation")
            row_test, _, _, _ = run_stage_e_model(dataset, splits, "reporter", lam, seed, strategy=canonical_strategy, split_name="test")
            sweep_rows.append({**row_val, "selected_on": "validation"})
            sweep_rows.append({**row_test, "selected_on": "test"})
            candidates.append(row_val)
        valid = [r for r in candidates if r["product_rmse"] <= 1.08 * product_val["product_rmse"]]
        if not valid:
            valid = candidates
        chosen = sorted(valid, key=lambda r: (-r["state_A_alignment"], r["product_rmse"]))[0]
        selected_lam = float(chosen["lambda_reporter"])
        selected_lambdas.append(selected_lam)

        product_row, product_z, product_out, product_flux = run_stage_e_model(dataset, splits, "product_only", 0, seed, strategy=canonical_strategy)
        reporter_row, reporter_z, reporter_out, reporter_flux = run_stage_e_model(dataset, splits, "reporter", selected_lam, seed, strategy=canonical_strategy)
        shuffled_row, shuffled_z, shuffled_out, shuffled_flux = run_stage_e_model(dataset, splits, "shuffled", selected_lam, seed, strategy=canonical_strategy)
        smooth_row, smooth_z, smooth_out, smooth_flux = run_stage_e_model(dataset, splits, "smooth_aux", selected_lam, seed, strategy=canonical_strategy)
        black_box = stage_e_black_box_row(dataset, splits, seed)
        for row in [product_row, reporter_row, shuffled_row, smooth_row, black_box] + oracle_rows:
            row["selected_lambda"] = selected_lam
            all_seed_rows.append(row)
        for strategy in ["joint", "decoder_calibrated", "alternating"]:
            r, _, _, _ = run_stage_e_model(dataset, splits, "reporter", selected_lam, seed, strategy=strategy)
            all_seed_rows.append({**r, "strategy_control": True})
        cause_rows.extend([{**r, "seed": seed} for r in stage_e_cause_audit(dataset, splits, reporter_z, product_z)])
        intervention_rows.extend([{**r, "seed": seed} for r in stage_e_intervention_shortcut_audit(dataset, splits, reporter_z)])
        flux_rows.extend([{**r, "seed": seed} for r in stage_e_flux_truth_audit(dataset)])
        test_idx = splits["test"]
        for model, out, z, flux in [
            ("product_only", product_out, product_z, product_flux),
            ("reporter", reporter_out, reporter_z, reporter_flux),
            ("shuffled", shuffled_out, shuffled_z, shuffled_flux),
            ("smooth_aux", smooth_out, smooth_z, smooth_flux),
        ]:
            per_intervention_rows.extend([{**r, "seed": seed} for r in stage_e_per_group_metrics(dataset, test_idx, model, out, z, flux, "intervention")])

    seed_df = pd.DataFrame(all_seed_rows)
    primary = seed_df[seed_df.get("strategy_control", False) != True]
    summary = primary.groupby("model", as_index=False).agg(
        product_rmse_mean=("product_rmse", "mean"),
        product_rmse_sd=("product_rmse", "std"),
        product_nrmse_mean=("product_nrmse", "mean"),
        biomass_rmse_mean=("biomass_rmse", "mean"),
        long_horizon_product_rmse_mean=("long_horizon_product_rmse", "mean"),
        post_perturbation_product_rmse_mean=("post_perturbation_product_rmse", "mean"),
        state_A_alignment_mean=("state_A_alignment", "mean"),
        state_A_alignment_sd=("state_A_alignment", "std"),
        state_B_alignment_mean=("state_B_alignment", "mean"),
        state_A_affine_rmse_mean=("state_A_affine_rmse", "mean"),
        state_B_affine_rmse_mean=("state_B_affine_rmse", "mean"),
        flux_feasibility_rate_mean=("flux_feasibility_rate", "mean"),
        product_flux_rmse_mean=("product_flux_rmse", "mean"),
        biomass_flux_rmse_mean=("biomass_flux_rmse", "mean"),
        state_dependent_bound_rmse_mean=("state_dependent_bound_rmse", "mean"),
        allocation_error_mean=("allocation_error", "mean"),
    )
    product_only_rmse = float(summary[summary.model == "product_only"].product_rmse_mean.iloc[0])
    decoder_summary = (
        pd.DataFrame(decoder_rows)
        .groupby("model", as_index=False)
        .agg(product_rmse=("product_rmse", "mean"), biomass_rmse=("biomass_rmse", "mean"), flux_feasibility_rate=("flux_feasibility_rate", "mean"), product_flux_rmse=("product_flux_rmse", "mean"), state_dependent_bound_rmse=("state_dependent_bound_rmse", "mean"))
    )
    decoder_summary["pass_30pct_product_only"] = decoder_summary.product_rmse < 0.30 * product_only_rmse
    learned = decoder_summary[decoder_summary.model == "true_states_learned_fba_decoder"].iloc[0]
    if not bool(learned.pass_30pct_product_only) or learned.flux_feasibility_rate < 0.99:
        raise RuntimeError(f"Stage E learned FBA decoder failed calibration: {learned.to_dict()}")

    selected = float(pd.Series(selected_lambdas).median())
    seed_df.to_csv(DATA / "repaired_stage_e_seed_results.csv", index=False)
    summary.to_csv(DATA / "repaired_stage_e_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(DATA / "repaired_stage_e_split.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(DATA / "repaired_stage_e_loss_sweep.csv", index=False)
    pd.DataFrame(target_rows).to_csv(DATA / "repaired_stage_e_target_audit.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(DATA / "repaired_stage_e_gradient_audit.csv", index=False)
    decoder_summary.to_csv(DATA / "repaired_stage_e_decoder_audit.csv", index=False)
    cause_df = pd.DataFrame(cause_rows)
    cause_summary = cause_df.groupby("feature_set", as_index=False).agg(accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean"))
    cause_summary.to_csv(DATA / "repaired_stage_e_cause_audit.csv", index=False)
    intervention_df = pd.DataFrame(intervention_rows)
    intervention_df.to_csv(DATA / "repaired_stage_e_intervention_audit.csv", index=False)
    flux_df = pd.DataFrame(flux_rows)
    flux_df.to_csv(DATA / "repaired_stage_e_flux_audit.csv", index=False)
    per_int = pd.DataFrame(per_intervention_rows)
    per_int.to_csv(DATA / "repaired_stage_e_per_intervention_metrics.csv", index=False)
    write_stage_e_figures(summary, pd.DataFrame(sweep_rows), decoder_summary, cause_summary, intervention_df, flux_df, stage_a_summary, per_int)
    make_stage_e_notebook(summary, pd.DataFrame(sweep_rows), decoder_summary, cause_summary, intervention_df, flux_df)
    return summary, seed_df, decoder_summary, cause_summary, intervention_df, flux_df, per_int, stage_a_summary, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "fast",
            "full",
            "repaired-stage-a",
            "repaired-stage-a-fast",
            "repaired-stage-a-full",
            "repaired-stage-e",
            "repaired-stage-e-fast",
            "repaired-stage-e-full",
            "repaired-stage-e-stability",
            "repaired-stage-f",
            "repaired-stage-f-fast",
        ],
        default="fast",
    )
    args = parser.parse_args()
    if args.mode.startswith("repaired-stage-f"):
        summary, seed_df, ci_df, degradation, per_group = run_repaired_stage_f(args.mode)
        print(summary.to_string(index=False))
        print("\nStage F paired confidence intervals")
        print(ci_df.to_string(index=False))
        print("\nStage F transfer degradation")
        print(degradation.to_string(index=False))
        return
    if args.mode.startswith("repaired-stage-e"):
        if args.mode == "repaired-stage-e-stability":
            summary, seed_df, traj_df, ci_df = run_repaired_stage_e_stability()
            print(summary.to_string(index=False))
            print("\nStage E stability paired confidence intervals")
            print(ci_df.to_string(index=False))
            return
        mode = "repaired-stage-e-fast" if args.mode == "repaired-stage-e" else args.mode
        summary, seed_df, decoder, cause, intervention, flux, per_int, stage_a, selected = run_repaired_stage_e(mode)
        print(summary.to_string(index=False))
        print("\nCanonical repaired Stage A strategy")
        print(stage_a.to_string(index=False))
        print("\nSelected Stage E reporter lambda")
        print(selected)
        print("\nStage E decoder calibration")
        print(decoder.to_string(index=False))
        print("\nStage E cause audit")
        print(cause.to_string(index=False))
        print("\nStage E per-intervention product RMSE")
        print(per_int.groupby(["intervention", "model"]).agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean")).reset_index().to_string(index=False))
        return
    if args.mode.startswith("repaired-stage-a"):
        mode = "repaired-stage-a-fast" if args.mode == "repaired-stage-a" else args.mode
        summary, seed_df, sweep, decoder, cause, head, strategy, selected = run_repaired_stage_a(mode)
        print(summary.to_string(index=False))
        print("\nSelected reporter lambda")
        print(selected)
        print("\nDecoder calibration")
        print(decoder.to_string(index=False))
        print("\nCause audit")
        print(cause.groupby("feature_set").agg(accuracy=("accuracy", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean")).reset_index().to_string(index=False))
        print("\nReporter head audit")
        print(head.groupby("reporter_head").agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean")).reset_index().to_string(index=False))
        print("\nTraining strategy audit")
        print(strategy.groupby("training_strategy").agg(product_rmse=("product_rmse", "mean"), state_A_alignment=("state_A_alignment", "mean")).reset_index().to_string(index=False))
        return
    summary, metrics, sweep, modularity, lc = run_diagnostics(args.mode)
    print(summary.to_string(index=False))
    print("\nModularity audit")
    print(modularity.to_string(index=False))
    print("\nCoverage curves")
    print(lc[["curve_type", "level", "product_rmse", "state_A_alignment"]].to_string(index=False))


if __name__ == "__main__":
    main()
