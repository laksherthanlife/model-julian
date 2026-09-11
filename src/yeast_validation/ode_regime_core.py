#!/usr/bin/env python3
"""Synthetic ODE physiology for fixed-environment beta-carotene regimes.

This module is intentionally separate from the Yeast9/pFBA pipeline.  It is a
transparent synthetic testbed whose deployment contract is still
``[temperature, pH, DO] -> full beta-carotene trajectory``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd


EPS = 1e-9
STATE_COLUMNS = ["X", "A", "R", "E", "S", "C", "P"]
AUX_COLUMNS = ["v_product", "mu"]
REGIME_ORDER = [
    "balanced_sustained",
    "delayed_pathway_activation",
    "precursor_limited_plateau",
    "atp_limited_slow",
    "oxidative_stress_collapse",
    "congestion_recovery",
    "growth_dominated_low_product",
    "mixed_transition",
]


@dataclass(frozen=True)
class ODEConfig:
    """Frozen parameters for the synthetic physiological generator."""

    seed: int = 20260724
    t_end: float = 48.0
    n_time: int = 73
    temperature_min: float = 27.0
    temperature_max: float = 33.0
    pH_min: float = 4.5
    pH_max: float = 5.5
    DO_min: float = 20.0
    DO_max: float = 80.0
    mu_max: float = 0.105
    death_base: float = 0.002
    k_product: float = 0.235
    initial_X: float = 0.055
    initial_A: float = 0.045
    initial_R: float = 0.86
    initial_E: float = 0.55
    initial_S: float = 0.055
    initial_C: float = 0.03
    max_state: float = 8.0
    final_dataset_train: int = 720
    final_dataset_validation: int = 180
    final_dataset_test: int = 220
    pilot_train: int = 160
    pilot_validation: int = 45
    pilot_test: int = 55


def config_dict(cfg: ODEConfig) -> dict[str, float | int]:
    return asdict(cfg)


def time_grid(cfg: ODEConfig = ODEConfig()) -> np.ndarray:
    return np.linspace(0.0, cfg.t_end, cfg.n_time)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -45, 45)))


def softplus(x):
    x = np.asarray(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def hill(x, k, n=2.0):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x**n / (k**n + x**n + EPS)


def env_to_unit(env: np.ndarray, cfg: ODEConfig = ODEConfig()) -> np.ndarray:
    e = np.asarray(env, dtype=float)
    return np.column_stack(
        [
            (e[:, 0] - cfg.temperature_min) / (cfg.temperature_max - cfg.temperature_min),
            (e[:, 1] - cfg.pH_min) / (cfg.pH_max - cfg.pH_min),
            (e[:, 2] - cfg.DO_min) / (cfg.DO_max - cfg.DO_min),
        ]
    )


def unit_to_env(unit: np.ndarray, cfg: ODEConfig = ODEConfig()) -> np.ndarray:
    u = np.asarray(unit, dtype=float)
    return np.column_stack(
        [
            cfg.temperature_min + u[:, 0] * (cfg.temperature_max - cfg.temperature_min),
            cfg.pH_min + u[:, 1] * (cfg.pH_max - cfg.pH_min),
            cfg.DO_min + u[:, 2] * (cfg.DO_max - cfg.DO_min),
        ]
    )


def normalize_env_raw(env: np.ndarray) -> np.ndarray:
    e = np.asarray(env, dtype=float)
    return np.column_stack([(e[:, 0] - 30.0) / 3.0, (e[:, 1] - 5.0) / 0.5, (e[:, 2] - 50.0) / 30.0])


def environmental_responses(T: float, pH: float, DO: float) -> dict[str, float]:
    """Smooth environmental response functions used by every ODE term."""

    phi_T = 0.08 + 0.92 * np.exp(-0.5 * ((T - 30.2) / 1.55) ** 2)
    phi_pH = 0.10 + 0.90 * np.exp(-0.5 * ((pH - 5.05) / 0.28) ** 2)
    phi_DO = 0.24 + 0.86 * hill(DO, 35.0, 2.2)
    low_oxygen = sigmoid((34.0 - DO) / 5.2)
    high_oxygen = sigmoid((DO - 63.0) / 5.0)
    temp_stress = 0.18 * ((T - 30.0) / 3.0) ** 2 + 0.38 * sigmoid((T - 31.6) / 0.55)
    ph_stress = 0.20 * ((pH - 5.05) / 0.55) ** 2
    oxygen_stress = 0.70 * high_oxygen * hill(DO, 66.0, 3.0)
    energy_from_oxygen = 0.30 + 0.88 * hill(DO, 32.0, 2.0)
    product_bias = sigmoid((T - 29.45) / 0.44) * (0.55 + 0.45 * np.exp(-0.5 * ((pH - 4.82) / 0.28) ** 2))
    growth_bias = phi_T * phi_pH * (1.0 - 0.32 * product_bias)
    return {
        "phi_T": float(phi_T),
        "phi_pH": float(phi_pH),
        "phi_DO": float(phi_DO),
        "low_oxygen": float(low_oxygen),
        "high_oxygen": float(high_oxygen),
        "temp_stress": float(temp_stress),
        "ph_stress": float(ph_stress),
        "oxygen_stress": float(oxygen_stress),
        "energy_from_oxygen": float(energy_from_oxygen),
        "product_bias": float(product_bias),
        "growth_bias": float(growth_bias),
    }


def initial_state(env: np.ndarray, cfg: ODEConfig = ODEConfig()) -> np.ndarray:
    T, pH, DO = map(float, env)
    r = environmental_responses(T, pH, DO)
    A0 = cfg.initial_A * (0.45 + 1.45 * r["product_bias"]) * (0.65 + 0.35 * r["phi_pH"])
    R0 = cfg.initial_R * (0.65 + 0.55 * r["growth_bias"])
    E0 = cfg.initial_E * (0.55 + 0.65 * r["energy_from_oxygen"])
    S0 = cfg.initial_S + 0.065 * r["temp_stress"] + 0.055 * r["oxygen_stress"]
    C0 = cfg.initial_C * (0.7 + 1.2 * r["product_bias"])
    return np.array([cfg.initial_X, A0, R0, E0, S0, C0, 0.0], dtype=float)


def support_terms(y: np.ndarray) -> dict[str, float]:
    X, A, R, E, S, C, _P = np.maximum(np.asarray(y, dtype=float), 0.0)
    return {
        "pathway_support": float(hill(A, 0.20, 2.0)),
        "precursor_support": float(hill(R, 0.22, 2.0)),
        "energy_support": float(0.18 + 0.92 * hill(E, 0.30, 2.4)),
        "stress_inhibition": float(1.0 / (1.0 + (S / 0.34) ** 3.0)),
        "congestion_inhibition": float(1.0 / (1.0 + (C / 0.32) ** 2.5)),
    }


def ode_rhs(_t: float, y: np.ndarray, env: np.ndarray, cfg: ODEConfig = ODEConfig()) -> np.ndarray:
    """Return derivatives for X, A, R, E, S, C, P.

    Biological interpretation:
    - X grows from smooth environmental suitability and is inhibited by ATP
      shortage, stress, and congestion.
    - A is synthesized according to product-allocation tendency and damaged by
      oxidative stress and congestion.
    - R is supplied by biomass and environmental carbon fitness, then consumed
      by growth and beta-carotene flux.
    - E is generated from oxygen-supported respiration, spent on growth,
      product flux, and stress repair.
    - S accumulates from temperature, high oxygen, product flux, and congestion,
      then recovers using oxygen and ATP capacity.
    - C accumulates when pathway flux exceeds processing capacity and clears
      through energy-dependent cellular maintenance.
    - P accumulates from the emergent product flux and can decline mildly under
      severe oxidative burden.
    """

    y = np.clip(np.asarray(y, dtype=float), 0.0, cfg.max_state)
    X, A, R, E, S, C, P = y
    T, pH, DO = map(float, env)
    r = environmental_responses(T, pH, DO)
    s = support_terms(y)
    mu = (
        cfg.mu_max
        * r["phi_T"]
        * r["phi_pH"]
        * r["phi_DO"]
        * s["energy_support"]
        * s["stress_inhibition"]
        * s["congestion_inhibition"]
        * (0.72 + 0.36 * r["growth_bias"])
    )
    v_product = (
        cfg.k_product
        * X
        * s["pathway_support"]
        * s["precursor_support"]
        * s["energy_support"]
        * s["stress_inhibition"]
        * s["congestion_inhibition"]
        * (0.55 + 0.95 * r["product_bias"])
    )
    death = cfg.death_base + 0.036 * hill(S, 0.58, 3.0) + 0.012 * hill(C, 0.70, 3.0)
    dX = (mu - death) * X
    target_A = 0.10 + 0.70 * r["product_bias"] * r["phi_pH"] * (0.72 + 0.28 * r["energy_from_oxygen"])
    synthesis = 0.045 * target_A * (0.42 + 0.58 * hill(E, 0.24, 2.0))
    dA = synthesis - 0.040 * A - 0.105 * hill(S, 0.38, 2.6) * A - 0.070 * hill(C, 0.42, 2.2) * A
    precursor_supply = 0.072 * X * (0.45 + 0.65 * r["phi_T"] * r["phi_pH"]) * (0.55 + 0.45 * r["growth_bias"])
    growth_use = 0.47 * mu * X * (0.85 + 0.70 * r["growth_bias"])
    product_use = 1.05 * v_product
    dR = precursor_supply - growth_use - product_use - 0.020 * R
    energy_generation = 0.080 * X * r["energy_from_oxygen"] * r["phi_T"] * (0.78 + 0.22 * r["phi_pH"])
    dE = (
        energy_generation
        - 0.54 * mu * X
        - 0.84 * v_product
        - 0.060 * S * (0.25 + hill(E, 0.28, 2.0))
        - 0.042 * (E - 0.56 * r["energy_from_oxygen"])
    )
    dS = (
        0.018 * r["temp_stress"]
        + 0.042 * r["oxygen_stress"]
        + 0.135 * v_product
        + 0.032 * hill(C, 0.36, 2.0)
        + 0.006 * r["low_oxygen"]
        - 0.085 * S * (0.30 + 0.70 * hill(E, 0.32, 2.0)) * (0.45 + 0.55 * hill(DO, 28.0, 2.0))
    )
    processing_capacity = 0.014 + 0.080 * hill(E, 0.31, 2.0) + 0.070 * hill(X, 0.34, 2.0)
    overload = softplus(16.0 * (v_product - processing_capacity)) / 16.0
    dC = 0.95 * overload + 0.026 * hill(A, 0.38, 2.0) * r["product_bias"] - 0.155 * C * (0.22 + hill(E, 0.34, 2.0) + 0.45 * hill(X, 0.34, 2.0))
    degradation = (0.003 + 0.034 * hill(S, 0.62, 4.0) + 0.008 * hill(C, 0.72, 3.0)) * P
    dP = v_product - degradation
    deriv = np.array([dX, dA, dR, dE, dS, dC, dP], dtype=float)
    for i, val in enumerate(y):
        if val <= 0.0 and deriv[i] < 0.0:
            deriv[i] = 0.0
    return deriv


def fluxes_for_state(y: np.ndarray, env: np.ndarray, cfg: ODEConfig = ODEConfig()) -> tuple[float, float]:
    y = np.clip(np.asarray(y, dtype=float), 0.0, cfg.max_state)
    X = y[0]
    T, pH, DO = map(float, env)
    r = environmental_responses(T, pH, DO)
    s = support_terms(y)
    mu = (
        cfg.mu_max
        * r["phi_T"]
        * r["phi_pH"]
        * r["phi_DO"]
        * s["energy_support"]
        * s["stress_inhibition"]
        * s["congestion_inhibition"]
        * (0.72 + 0.36 * r["growth_bias"])
    )
    v_product = (
        cfg.k_product
        * X
        * s["pathway_support"]
        * s["precursor_support"]
        * s["energy_support"]
        * s["stress_inhibition"]
        * s["congestion_inhibition"]
        * (0.55 + 0.95 * r["product_bias"])
    )
    return float(v_product), float(mu)


def rk4_step(y: np.ndarray, env: np.ndarray, dt: float, cfg: ODEConfig, t: float) -> np.ndarray:
    k1 = ode_rhs(t, y, env, cfg)
    k2 = ode_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, env, cfg)
    k3 = ode_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, env, cfg)
    k4 = ode_rhs(t + dt, y + dt * k3, env, cfg)
    out = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.clip(out, 0.0, cfg.max_state)


def simulate_environment(env: np.ndarray, cfg: ODEConfig = ODEConfig()) -> tuple[pd.DataFrame, dict[str, float | str | bool]]:
    env = np.asarray(env, dtype=float)
    t = time_grid(cfg)
    y = initial_state(env, cfg)
    rows = []
    solver_failed = False
    for k, tt in enumerate(t):
        v_product, mu = fluxes_for_state(y, env, cfg)
        rows.append(
            {
                "time_index": k,
                "time": float(tt),
                "temperature": float(env[0]),
                "pH": float(env[1]),
                "DO": float(env[2]),
                **{name: float(y[i]) for i, name in enumerate(STATE_COLUMNS)},
                "v_product": v_product,
                "mu": mu,
            }
        )
        if k < len(t) - 1:
            y = rk4_step(y, env, float(t[k + 1] - tt), cfg, float(tt))
            solver_failed = solver_failed or (not np.isfinite(y).all()) or bool((y < -1e-10).any())
    df = pd.DataFrame(rows)
    diagnostics = regime_diagnostics(df)
    diagnostics["solver_failed"] = bool(solver_failed or not np.isfinite(df[STATE_COLUMNS + AUX_COLUMNS]).all().all())
    diagnostics["dominant_regime"] = assign_regime(diagnostics)
    return df, diagnostics


def regime_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    t = df["time"].to_numpy(float)
    P = df["P"].to_numpy(float)
    v = df["v_product"].to_numpy(float)
    X = df["X"].to_numpy(float)
    A = df["A"].to_numpy(float)
    R = df["R"].to_numpy(float)
    E = df["E"].to_numpy(float)
    S = df["S"].to_numpy(float)
    C = df["C"].to_numpy(float)
    dP = np.gradient(P, t)
    onset_thresh = max(0.28 * float(np.max(v)), 1e-5)
    active = np.where(v >= onset_thresh)[0]
    flux_onset_time = float(t[active[0]]) if len(active) else float(t[-1])
    a_thresh = A[0] + 0.45 * max(float(np.max(A) - A[0]), EPS)
    a_active = np.where(A >= a_thresh)[0]
    pathway_activation_time = max(flux_onset_time, float(t[a_active[0]]) if len(a_active) else float(t[-1]))
    early_slope = float(np.mean(dP[1 : max(3, len(dP) // 5)]))
    late_slope = float(np.mean(dP[-max(3, len(dP) // 5) :]))
    decline = float(max(0.0, np.max(P) - P[-1]))
    return {
        "final_product": float(P[-1]),
        "max_product": float(np.max(P)),
        "auc_product": float(np.trapezoid(P, t)),
        "max_product_rate": float(np.max(v)),
        "endpoint_biomass": float(X[-1]),
        "mean_DO": float(df["DO"].mean()),
        "mean_temperature": float(df["temperature"].mean()),
        "growth_to_product_allocation": float(np.trapezoid(X * df["mu"].to_numpy(float), t) / (np.trapezoid(v, t) + EPS)),
        "integrated_stress": float(np.trapezoid(S, t)),
        "max_stress": float(np.max(S)),
        "min_energy": float(np.min(E)),
        "final_precursor": float(R[-1]),
        "precursor_depletion": float(max(0.0, R[0] - R[-1])),
        "pathway_capacity_loss": float(max(0.0, np.max(A) - A[-1])),
        "final_pathway_capacity": float(A[-1]),
        "max_congestion": float(np.max(C)),
        "congestion_recovery": float(max(0.0, np.max(C) - C[-1])),
        "pathway_activation_time": pathway_activation_time,
        "late_decline": decline,
        "early_slope": early_slope,
        "late_slope": late_slope,
        "plateau_index": float(1.0 - late_slope / (early_slope + EPS)) if early_slope > EPS else 0.0,
    }


def assign_regime(d: dict[str, float]) -> str:
    """Transparent post-simulation diagnostic; never used to generate curves."""

    if d["max_stress"] > 0.62 and d["late_decline"] > 0.006 and d["pathway_capacity_loss"] > 0.045:
        return "oxidative_stress_collapse"
    if d["pathway_activation_time"] > 12.0 and d["final_product"] > 0.075 and d["max_stress"] < 0.62:
        return "delayed_pathway_activation"
    if d["growth_to_product_allocation"] > 3.4 and d["endpoint_biomass"] > 0.085 and d["final_product"] < 0.080:
        return "growth_dominated_low_product"

    scores = {
        "oxidative_stress_collapse": 2.3 * hill(d["integrated_stress"], 13.0, 3.0) * hill(d["mean_DO"], 58.0, 4.0)
        + 1.8 * hill(d["late_decline"], 0.10, 2.0)
        + 1.2 * hill(d["pathway_capacity_loss"], 0.30, 2.0),
        "atp_limited_slow": 2.3 * (1.0 - hill(d["min_energy"], 0.18, 3.0)) * (1.0 - hill(d["mean_DO"], 34.0, 4.0))
        + 0.7 * (1.0 - hill(d["max_product_rate"], 0.006, 2.0)),
        "precursor_limited_plateau": 2.1 * (1.0 - hill(d["final_precursor"], 0.15, 3.0))
        + 1.2 * hill(d["plateau_index"], 0.45, 2.0)
        + 0.5 * hill(d["final_pathway_capacity"], 0.20, 2.0),
        "congestion_recovery": 2.4 * hill(d["max_congestion"], 0.24, 3.0)
        + 1.8 * hill(d["congestion_recovery"], 0.025, 2.0)
        + 0.4 * hill(d["late_slope"], 0.002, 2.0),
        "delayed_pathway_activation": 2.3 * hill(d["pathway_activation_time"], 10.0, 3.0)
        + 0.6 * hill(d["final_product"], 0.10, 2.0),
        "growth_dominated_low_product": 1.8 * hill(d["endpoint_biomass"], 0.85, 2.0)
        + 1.7 * hill(d["growth_to_product_allocation"], 7.5, 2.0)
        + 0.8 * (1.0 - hill(d["final_product"], 0.12, 2.0)),
        "balanced_sustained": 1.5 * hill(d["final_product"], 0.16, 2.0)
        + 0.9 * hill(d["endpoint_biomass"], 0.35, 2.0)
        + 0.7 * hill(d["late_slope"], 0.002, 2.0)
        + 0.7 * hill(d["min_energy"], 0.24, 2.0)
        + 0.5 * (1.0 - hill(d["max_stress"], 0.50, 3.0)),
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.10:
        return "mixed_transition"
    return ranked[0][0]


def env_id(env: np.ndarray, prefix: str, i: int) -> str:
    T, pH, DO = map(float, env)
    return f"{prefix}_{i:04d}_T{T:.3f}_pH{pH:.3f}_DO{DO:.3f}".replace(".", "p")


def latin_hypercube(n: int, rng: np.random.Generator, dim: int = 3, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    cols = []
    for _ in range(dim):
        edges = np.linspace(low, high, n + 1)
        vals = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(vals)
        cols.append(vals)
    return np.column_stack(cols)


def in_heldout_holes(unit: np.ndarray) -> np.ndarray:
    u = np.asarray(unit, dtype=float)
    hot_oxygen_hole = (u[:, 0] > 0.66) & (u[:, 2] > 0.62)
    low_oxygen_hole = (u[:, 2] < 0.25) & (u[:, 1] > 0.58)
    return hot_oxygen_hole | low_oxygen_hole


def make_environment_manifest(cfg: ODEConfig = ODEConfig(), pilot: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + (17 if pilot else 0))
    n_train = cfg.pilot_train if pilot else cfg.final_dataset_train
    n_val = cfg.pilot_validation if pilot else cfg.final_dataset_validation
    n_test = cfg.pilot_test if pilot else cfg.final_dataset_test

    def draw(n: int, reject_holes: bool, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        rows = []
        while len(rows) < n:
            u = latin_hypercube(max(n, 12), rng, low=low, high=high)
            if reject_holes:
                u = u[~in_heldout_holes(u)]
            rows.extend(u.tolist())
        return np.asarray(rows[:n])

    train = draw(n_train, True, 0.02, 0.98)
    val = draw(n_val, True, 0.02, 0.98)
    random_test = draw(n_test, False, 0.02, 0.98)
    hole = []
    while len(hole) < max(90 if not pilot else 24, n_test // 3):
        u = latin_hypercube(max(60, n_test // 2), rng, low=0.02, high=0.98)
        u = u[in_heldout_holes(u)]
        hole.extend(u.tolist())
    hole = np.asarray(hole[: max(90 if not pilot else 24, n_test // 3)])
    edge_levels = np.array([0.0, 0.08, 0.92, 1.0])
    edges = np.array(np.meshgrid(edge_levels, edge_levels, edge_levels)).T.reshape(-1, 3)
    interior = np.linspace(0.0, 1.0, 7)
    dense = np.array(np.meshgrid(interior, interior, interior)).T.reshape(-1, 3)
    parts = [
        ("train", train, "latin_hypercube_excluding_predeclared_holes"),
        ("validation", val, "independent_random_continuous_excluding_holes"),
        ("random_test", random_test, "independent_random_continuous_global"),
        ("heldout_hole", hole, "compact_interior_blocks_absent_from_training_not_extrapolation"),
        ("edge_extrapolation", edges, "environmental_edges_within_allowed_bounds"),
        ("atlas_grid", dense, "regular_grid_for_regime_atlas"),
    ]
    rows = []
    culture_i = 0
    for split, unit, rule in parts:
        env = unit_to_env(unit, cfg)
        for i, e in enumerate(env):
            rows.append(
                {
                    "culture_id": env_id(e, split, i),
                    "environment_id": env_id(e, split, i),
                    "split": split,
                    "temperature": float(e[0]),
                    "pH": float(e[1]),
                    "DO": float(e[2]),
                    "unit_temperature": float(unit[i, 0]),
                    "unit_pH": float(unit[i, 1]),
                    "unit_DO": float(unit[i, 2]),
                    "heldout_rule": rule,
                    "culture_index": culture_i,
                }
            )
            culture_i += 1
    return pd.DataFrame(rows)


def simulate_manifest(manifest: pd.DataFrame, cfg: ODEConfig = ODEConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_rows, diag_rows = [], []
    for row in manifest.itertuples(index=False):
        env = np.array([row.temperature, row.pH, row.DO], dtype=float)
        traj, diag = simulate_environment(env, cfg)
        for col in ["culture_id", "environment_id", "split"]:
            traj[col] = getattr(row, col)
        traj["culture_index"] = row.culture_index
        traj_rows.append(traj)
        diag_rows.append(
            {
                "culture_id": row.culture_id,
                "environment_id": row.environment_id,
                "split": row.split,
                "temperature": row.temperature,
                "pH": row.pH,
                "DO": row.DO,
                **diag,
            }
        )
    return pd.concat(traj_rows, ignore_index=True), pd.DataFrame(diag_rows)


def pivot_trajectories(traj: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    meta = traj.groupby("culture_id", as_index=False).first()[
        ["culture_id", "environment_id", "split", "temperature", "pH", "DO", "culture_index"]
    ].sort_values("culture_index")
    time = np.sort(traj["time"].unique())
    mats: dict[str, np.ndarray] = {}
    for col in value_cols:
        wide = traj.pivot(index="culture_id", columns="time_index", values=col).loc[meta["culture_id"]]
        mats[col] = wide.to_numpy(float)
    return meta.reset_index(drop=True), mats, time


def pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    _u, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    explained = s**2 / max(float(np.sum(s**2)), EPS)
    return mean, vt, explained


def normalized_curves(curves: np.ndarray) -> np.ndarray:
    c = np.asarray(curves, dtype=float)
    return (c - c[:, [0]]) / np.maximum(c.max(axis=1, keepdims=True) - c[:, [0]], EPS)


def polynomial_features(env: np.ndarray, degree: int = 2) -> np.ndarray:
    x = normalize_env_raw(env)
    cols = [np.ones(len(x))]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x.shape[1]), d):
            v = np.ones(len(x))
            for j in combo:
                v *= x[:, j]
            cols.append(v)
    return np.column_stack(cols)


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-5) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    eye = np.eye(X.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * eye, X.T @ Y)


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(axis=0)
            else:
                centers[j] = X[int(rng.integers(0, len(X)))]
    return labels, centers


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    n = len(a)
    contingency = np.zeros((a.max() + 1, b.max() + 1), dtype=float)
    for i in range(n):
        contingency[a[i], b[i]] += 1

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_comb = comb2(contingency).sum()
    row_comb = comb2(contingency.sum(axis=1)).sum()
    col_comb = comb2(contingency.sum(axis=0)).sum()
    total_comb = comb2(float(n))
    expected = row_comb * col_comb / max(total_comb, EPS)
    denom = 0.5 * (row_comb + col_comb) - expected
    return float((sum_comb - expected) / max(denom, EPS))


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    n = float(len(a))
    mi = 0.0
    ha = 0.0
    hb = 0.0
    for i in np.unique(a):
        pa = np.mean(a == i)
        ha -= pa * np.log(pa + EPS)
    for j in np.unique(b):
        pb = np.mean(b == j)
        hb -= pb * np.log(pb + EPS)
    for i in np.unique(a):
        for j in np.unique(b):
            pab = np.mean((a == i) & (b == j))
            if pab > 0:
                pa = np.mean(a == i)
                pb = np.mean(b == j)
                mi += pab * np.log(pab / (pa * pb + EPS) + EPS)
    return float(mi / max(np.sqrt(ha * hb), EPS))


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    total = 0
    for c in np.unique(labels_pred):
        vals, counts = np.unique(labels_true[labels_pred == c], return_counts=True)
        if len(vals):
            total += int(np.max(counts))
    return float(total / max(len(labels_true), 1))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return 0.0
    dist = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2), 0.0))
    vals = []
    for i in range(len(X)):
        same = labels == labels[i]
        a = float(np.mean(dist[i, same & (np.arange(len(X)) != i)])) if np.sum(same) > 1 else 0.0
        b = min(float(np.mean(dist[i, labels == c])) for c in np.unique(labels) if c != labels[i])
        vals.append((b - a) / max(a, b, EPS))
    return float(np.mean(vals))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def encode_regimes(labels: np.ndarray) -> np.ndarray:
    order = {name: i for i, name in enumerate(REGIME_ORDER)}
    return np.array([order.get(x, len(order)) for x in labels], dtype=int)


def finite_difference_sensitivity(fn, env: np.ndarray, step: np.ndarray | None = None) -> np.ndarray:
    env = np.asarray(env, dtype=float)
    step = np.asarray(step if step is not None else [0.05, 0.01, 0.5], dtype=float)
    bounds = np.array([[27.0, 33.0], [4.5, 5.5], [20.0, 80.0]], dtype=float)
    grads = []
    for j in range(3):
        lo = env.copy()
        hi = env.copy()
        lo[j] = max(bounds[j, 0], lo[j] - step[j])
        hi[j] = min(bounds[j, 1], hi[j] + step[j])
        denom = hi[j] - lo[j]
        grads.append((float(fn(hi)) - float(fn(lo))) / max(denom, EPS))
    return np.asarray(grads, dtype=float)


def equation_catalog() -> pd.DataFrame:
    rows = [
        ("growth", "mu = mu_max * phi_T * phi_pH * phi_DO * energy_support(E) * stress_inhibition(S) * congestion_inhibition(C)", "Environment and internal physiological state jointly determine biomass growth."),
        ("biomass", "dX/dt = (mu - death(S,C)) * X", "Stress and congestion can reduce viable biomass accumulation."),
        ("pathway_capacity", "dA/dt = synthesis(T,pH,DO,E) - decay*A - oxidative_damage(S)*A - congestion_damage(C)*A", "Pathway capacity builds smoothly but is damaged by stress and bottlenecks."),
        ("precursor_reserve", "dR/dt = supply(T,pH,DO,X) - growth_use(mu,X) - product_use(v_product) - loss*R", "Carbon precursor is shared by growth and heterologous product flux."),
        ("energy_capacity", "dE/dt = generation(DO,X) - growth_cost - product_cost - stress_repair_cost - relaxation", "Low oxygen limits ATP; high demand depletes energy."),
        ("oxidative_stress", "dS/dt = temperature_stress + high_oxygen_stress + flux_stress + congestion_stress - recovery(DO,E,S)", "Stress emerges continuously from environment and product burden."),
        ("congestion", "dC/dt = overload(v_product,processing_capacity) - energy_dependent_clearance(E,C)", "Flux can overload the pathway and later recover as clearance catches up."),
        ("product_flux", "v_product = k_product * X * pathway_support(A) * precursor_support(R) * energy_support(E) * stress_inhibition(S) * congestion_inhibition(C)", "Product formation emerges from coupled state interactions."),
        ("product", "dP/dt = v_product - degradation(S,C)*P", "Accumulated beta-carotene can plateau or decline under severe oxidative/congestion burden."),
    ]
    return pd.DataFrame(rows, columns=["term", "equation", "interpretation"])
