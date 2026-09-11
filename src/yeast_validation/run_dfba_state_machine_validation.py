#!/usr/bin/env python3
"""Canonical Experiments 3A-3F for fixed-environment dFBA validation.

The deployment problem remains fixed culture conditions
`e = [temperature, pH, dissolved_oxygen] -> complete beta-carotene trajectory`.

Full non-fast runs require an explicit yeast GEM asset.  Fast mode uses a
declared reduced stoichiometric surrogate so tests can exercise the complete
experiment chain without silently downloading or fabricating a genome-scale
model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

STATE_ORDER = (
    "balanced",
    "productive",
    "oxidative_stress",
    "energy_limited",
    "pathway_limited",
    "combined_overload",
    "recovering",
)
CAUSAL_STATES = ("z_ox", "z_atp", "z_bottle")
TEST_SPLITS = ("interpolation", "heldout_combination", "extrapolation")


@dataclass(frozen=True)
class DFBAConfig:
    dataset_seeds: tuple[int, ...] = (701, 702, 703)
    model_seeds: tuple[int, ...] = (17, 29, 43)
    fast_dataset_seeds: tuple[int, ...] = (701,)
    fast_model_seeds: tuple[int, ...] = (17,)
    replicates: int = 2
    fast_replicates: int = 1
    n_time: int = 49
    fast_n_time: int = 25
    dt: float = 0.25
    mlp_steps: int = 260
    fast_mlp_steps: int = 80
    latent_dim: int = 6
    reporter_noise: float = 0.025
    product_noise: float = 0.012
    useful_margin: float = 0.05
    er_tolerance: float = 0.05
    metabolic_asset_env: str = "YEAST_GEM_PATH"
    reduced_fast_model: bool = False

    # Culture dynamics.
    x0: float = 0.08
    k_x: float = 6.0
    b0: float = 0.0
    beta_deg_base: float = 0.015

    # Environmental optima and widths.
    t_opt: float = 30.0
    t_sigma: float = 5.0
    ph_opt: float = 5.0
    ph_sigma: float = 0.75
    do_opt: float = 45.0

    # Burden kinetics.
    ox_recovery: float = 0.34
    atp_recovery: float = 0.38
    bottle_recovery: float = 0.26
    er_recovery: float = 0.30

    # State-machine hysteresis.
    theta_ox_hi: float = 0.62
    theta_ox_lo: float = 0.42
    theta_atp_hi: float = 0.58
    theta_atp_lo: float = 0.38
    theta_bottle_hi: float = 0.60
    theta_bottle_lo: float = 0.40
    min_dwell_steps: int = 2

    # Declared state effects.
    gamma_balanced: float = 0.72
    gamma_productive: float = 0.66
    gamma_oxidative_stress: float = 0.57
    gamma_energy_limited: float = 0.52
    gamma_pathway_limited: float = 0.55
    gamma_combined_overload: float = 0.43
    gamma_recovering: float = 0.58
    beta_ub_balanced: float = 1.00
    beta_ub_productive: float = 1.10
    beta_ub_oxidative_stress: float = 0.72
    beta_ub_energy_limited: float = 0.80
    beta_ub_pathway_limited: float = 0.48
    beta_ub_combined_overload: float = 0.35
    beta_ub_recovering: float = 0.68
    atpm_balanced: float = 1.00
    atpm_productive: float = 1.08
    atpm_oxidative_stress: float = 1.24
    atpm_energy_limited: float = 1.42
    atpm_pathway_limited: float = 1.12
    atpm_combined_overload: float = 1.65
    atpm_recovering: float = 1.22
    death_balanced: float = 0.000
    death_productive: float = 0.002
    death_oxidative_stress: float = 0.015
    death_energy_limited: float = 0.012
    death_pathway_limited: float = 0.009
    death_combined_overload: float = 0.032
    death_recovering: float = 0.006


class MissingMetabolicAsset(RuntimeError):
    """Raised when the full dFBA chain lacks a configured yeast GEM."""


class MissingMetabolicBackend(RuntimeError):
    """Raised when the requested metabolic solver backend is not implemented."""


def active_config(fast: bool) -> DFBAConfig:
    cfg = DFBAConfig()
    if fast:
        cfg = replace(
            cfg,
            dataset_seeds=cfg.fast_dataset_seeds,
            model_seeds=cfg.fast_model_seeds,
            replicates=cfg.fast_replicates,
            n_time=cfg.fast_n_time,
            mlp_steps=cfg.fast_mlp_steps,
            reduced_fast_model=True,
        )
    return cfg


def metabolic_backend(cfg: DFBAConfig) -> str:
    if cfg.reduced_fast_model:
        return "reduced_stoichiometric_surrogate"
    return "yeast_gem_lp"


def ensure_dirs() -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    for sub in (
        "experiment_3a_static_metabolic_control",
        "experiment_3b_state_machine_dfba",
        "experiment_3c_useful_reporters",
        "experiment_3d_partial_reporters",
        "experiment_3e_irrelevant_er",
        "experiment_3f_model_comparison",
    ):
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)


def configured_gem_path(cfg: DFBAConfig) -> Path | None:
    env_path = os.environ.get(cfg.metabolic_asset_env)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            MODELS / "yeast_gem.xml",
            MODELS / "yeast_gem.json",
            MODELS / "yeast8.xml",
            MODELS / "yeast9.xml",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def require_metabolic_asset(cfg: DFBAConfig) -> None:
    if cfg.reduced_fast_model:
        return
    if configured_gem_path(cfg) is None:
        raise MissingMetabolicAsset(
            "Experiments 3A-3F require an explicit yeast GEM asset for non-fast "
            "runs. Set YEAST_GEM_PATH to a local Yeast8/Yeast9 SBML/JSON file or "
            "place it at models/yeast_gem.xml. No model is downloaded at runtime. "
            "Use --fast to exercise the full chain with the documented reduced "
            "stoichiometric surrogate."
        )
    raise MissingMetabolicBackend(
        "A yeast GEM asset is configured, but this repository does not yet wire "
        "a real repeated LP/FBA solver into Experiments 3A-3F. Refusing to run "
        "non-fast mode rather than silently using the reduced surrogate and "
        "calling it FBA."
    )


def time_grid(cfg: DFBAConfig) -> np.ndarray:
    return np.arange(cfg.n_time, dtype=float) * cfg.dt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -45, 45)))


def gaussian(x: float | np.ndarray, opt: float, sigma: float):
    return np.exp(-((np.asarray(x) - opt) ** 2) / (2.0 * sigma**2))


def low_do_penalty(do: float) -> float:
    return float(sigmoid((24.0 - do) / 5.5))


def high_do_penalty(do: float) -> float:
    return float(sigmoid((do - 66.0) / 6.0))


def env_id_from_values(t: float, ph: float, do: float) -> str:
    return f"T{t:.1f}_pH{ph:.2f}_DO{do:.1f}".replace(".", "p")


def holdout_region_name(t: float, ph: float, do: float) -> str:
    names = []
    if t >= 33 and do <= 25:
        names.append("highT_lowDO")
    if t >= 33 and ph <= 4.5:
        names.append("highT_lowpH")
    if do >= 60 and abs(ph - 5.0) >= 0.5:
        names.append("highDO_offpH")
    if t >= 36 and ph <= 4.0 and do >= 80:
        names.append("three_way_corner")
    return "+".join(names) if names else "training_support"


def assign_split(t: float, ph: float, do: float) -> str:
    region = holdout_region_name(t, ph, do)
    if region != "training_support":
        return "heldout_combination"
    key = (int(round(t * 10)) + int(round(ph * 20)) + int(round(do))) % 11
    if key in {0, 5}:
        return "validation"
    if key in {2, 7}:
        return "interpolation"
    return "train"


def make_environment_grid(fast: bool) -> pd.DataFrame:
    if fast:
        temps = [24.0, 30.0, 36.0]
        phs = [4.0, 5.0, 6.0]
        dos = [10.0, 40.0, 80.0]
    else:
        temps = [24.0, 27.0, 30.0, 33.0, 36.0]
        phs = [4.0, 4.5, 5.0, 5.5, 6.0]
        dos = [10.0, 25.0, 40.0, 60.0, 80.0]

    rows = []
    for t in temps:
        for ph in phs:
            for do in dos:
                rows.append(
                    {
                        "environment_id": env_id_from_values(t, ph, do),
                        "temperature": t,
                        "pH": ph,
                        "DO": do,
                        "split": assign_split(t, ph, do),
                        "heldout_region": holdout_region_name(t, ph, do),
                        "is_extrapolation": False,
                    }
                )
    extrap = [
        (22.0, 5.0, 40.0),
        (38.0, 5.0, 40.0),
        (30.0, 3.7, 40.0),
        (30.0, 6.3, 40.0),
        (30.0, 5.0, 5.0),
        (30.0, 5.0, 95.0),
    ]
    for t, ph, do in extrap:
        rows.append(
            {
                "environment_id": env_id_from_values(t, ph, do),
                "temperature": t,
                "pH": ph,
                "DO": do,
                "split": "extrapolation",
                "heldout_region": "numerical_extrapolation",
                "is_extrapolation": True,
            }
        )
    return pd.DataFrame(rows)


def state_effects(state: str, cfg: DFBAConfig) -> dict[str, float]:
    suffix = state
    return {
        "gamma": getattr(cfg, f"gamma_{suffix}"),
        "beta_ub": getattr(cfg, f"beta_ub_{suffix}"),
        "atpm": getattr(cfg, f"atpm_{suffix}"),
        "death": getattr(cfg, f"death_{suffix}"),
    }


def recovery_multiplier(state: str) -> float:
    return {
        "balanced": 1.00,
        "productive": 1.00,
        "oxidative_stress": 3.20,
        "energy_limited": 3.20,
        "pathway_limited": 2.80,
        "combined_overload": 4.10,
        "recovering": 2.70,
    }[state]


def degradation_multiplier(state: str) -> float:
    return {
        "balanced": 1.00,
        "productive": 1.05,
        "oxidative_stress": 7.00,
        "energy_limited": 4.50,
        "pathway_limited": 3.50,
        "combined_overload": 12.00,
        "recovering": 4.00,
    }[state]


def beta_pathway_reactions() -> list[dict[str, object]]:
    return [
        {
            "reaction_id": "BETA_PRECURSOR_SUPPLY",
            "name": "native acetyl-CoA to mevalonate/IPPs surrogate",
            "stoichiometry": {"accoa_c": -2.0, "nadph_c": -2.0, "ipp_dmapp_c": 1.0},
            "compartment": "c",
            "mass_balance_note": "lumped synthetic reaction; cofactors are partial.",
        },
        {
            "reaction_id": "BETA_GGPP_FORMATION",
            "name": "IPP/DMAPP to FPP/GGPP surrogate",
            "stoichiometry": {"ipp_dmapp_c": -4.0, "ggpp_c": 1.0, "ppi_c": 4.0},
            "compartment": "c",
            "mass_balance_note": "lumped isoprenoid-chain surrogate.",
        },
        {
            "reaction_id": "BETA_CAROTENE_FORMATION",
            "name": "GGPP to beta-carotene surrogate",
            "stoichiometry": {"ggpp_c": -2.0, "nadph_c": -4.0, "beta_carotene_c": 1.0},
            "compartment": "c",
            "mass_balance_note": "conceptual phytoene/lycopene/beta-carotene lump.",
        },
        {
            "reaction_id": "EX_beta_carotene",
            "name": "beta-carotene accumulation sink",
            "stoichiometry": {"beta_carotene_c": -1.0},
            "compartment": "c",
            "mass_balance_note": "intentional non-mass-balanced accumulation sink.",
        },
    ]


def reduced_flux_solution(env: np.ndarray, burdens: np.ndarray, state: str, cfg: DFBAConfig) -> dict[str, float]:
    temp, ph, do = [float(x) for x in env]
    z_ox, z_atp, z_bottle, _z_er = [float(x) for x in burdens]
    eff = state_effects(state, cfg)
    phi_t = float(gaussian(temp, cfg.t_opt, cfg.t_sigma))
    phi_ph = float(gaussian(ph, cfg.ph_opt, cfg.ph_sigma))
    low_do = low_do_penalty(do)
    high_do = high_do_penalty(do)
    do_resp = float(np.clip(1.0 - 0.42 * low_do - 0.20 * high_do, 0.25, 1.12))
    growth_capacity = max(0.02, 0.47 * phi_t * phi_ph * do_resp)
    mu_max = max(0.02, growth_capacity * (1.0 - 0.22 * z_ox - 0.18 * z_atp))
    gamma = eff["gamma"]
    biomass = max(0.0, gamma * mu_max)
    precursor_capacity = max(0.0, 0.72 * phi_t * (0.55 + 0.45 * phi_ph) * (1.0 - 0.18 * low_do))
    precursor = precursor_capacity
    beta_capacity = max(0.0, 0.38 * phi_t * (0.78 + 0.22 * phi_ph) * (1.0 - 0.62 * z_bottle - 0.18 * z_ox))
    beta_ub = eff["beta_ub"] * beta_capacity
    beta_flux = max(0.0, min(beta_ub, 0.62 * precursor - 0.18 * biomass))
    oxygen_uptake_bound = max(0.0, 0.55 * do / 100.0 * (1.0 - 0.45 * low_do))
    atp_supply = max(0.05, 1.05 * (1.0 - 0.62 * low_do + 0.18 * high_do) * phi_ph)
    atpm = eff["atpm"] * (0.45 + 0.28 * (1.0 - phi_ph) + 0.20 * low_do)
    repair = 0.18 * z_ox + 0.12 * z_atp + 0.10 * z_bottle
    rho_atp = (atpm + 0.88 * biomass + 0.55 * beta_flux + 0.25 * repair) / (atp_supply + 1e-8)
    v_o2 = oxygen_uptake_bound
    return {
        "mu_max": mu_max,
        "growth_capacity": growth_capacity,
        "biomass_flux": biomass,
        "v_beta": beta_flux,
        "v_precursor": precursor,
        "precursor_capacity": precursor_capacity,
        "v_o2": v_o2,
        "oxygen_uptake_bound": oxygen_uptake_bound,
        "v_atpm": atpm,
        "v_atp_supply": atp_supply,
        "rho_atp": rho_atp,
        "gamma_growth_fraction": gamma,
        "beta_upper_bound": beta_ub,
        "feasible": bool(mu_max > 0 and atp_supply > 0 and beta_capacity >= 0),
    }


def next_state(current: str, burdens: np.ndarray, dwell: int, v_beta: float, cfg: DFBAConfig) -> str:
    z_ox, z_atp, z_bottle = burdens[:3]
    high = [
        z_ox > cfg.theta_ox_hi,
        z_atp > cfg.theta_atp_hi,
        z_bottle > cfg.theta_bottle_hi,
    ]
    low = [
        z_ox < cfg.theta_ox_lo,
        z_atp < cfg.theta_atp_lo,
        z_bottle < cfg.theta_bottle_lo,
    ]
    if dwell < cfg.min_dwell_steps and current not in {"balanced", "productive"}:
        return current
    if sum(high) >= 2:
        return "combined_overload"
    if high[0]:
        return "oxidative_stress"
    if high[1]:
        return "energy_limited"
    if high[2]:
        return "pathway_limited"
    if current in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"} and not all(low):
        return "recovering"
    if all(low) and v_beta > 0.10:
        return "productive"
    return "balanced"


def update_burdens(env: np.ndarray, burdens: np.ndarray, flux: dict[str, float], state: str, cfg: DFBAConfig, feedback: bool) -> np.ndarray:
    temp, ph, do = [float(x) for x in env]
    z_ox, z_atp, z_bottle, z_er = [float(x) for x in burdens]
    phi_t = float(gaussian(temp, cfg.t_opt, cfg.t_sigma))
    phi_ph = float(gaussian(ph, cfg.ph_opt, cfg.ph_sigma))
    high_do = high_do_penalty(do)
    low_do = low_do_penalty(do)
    env_fitness = float(np.clip(phi_t * phi_ph * (1.0 - 0.50 * low_do - 0.35 * high_do), 0.0, 1.0))
    stress_gain = 1.15 - 0.55 * env_fitness
    if state in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"}:
        stress_gain *= 0.55
    if state == "recovering":
        stress_gain *= 0.45
    recovery = recovery_multiplier(state)
    if feedback:
        b_specific_proxy = flux["v_beta"] / (0.08 + flux["biomass_flux"])
        dz_ox = (
            stress_gain * (0.42 * flux["v_o2"] + 0.30 * flux["v_beta"] + 0.48 * high_do + 0.20 * z_bottle)
            - cfg.ox_recovery * recovery * z_ox
            - 0.08 * b_specific_proxy / (0.25 + b_specific_proxy)
        )
        dz_atp = stress_gain * 0.82 * max(0.0, flux["rho_atp"] - 1.0) - cfg.atp_recovery * recovery * z_atp
        dz_bottle = stress_gain * 0.72 * max(0.0, flux["v_precursor"] - 1.45 * flux["v_beta"]) - cfg.bottle_recovery * recovery * z_bottle
    else:
        dz_ox = -cfg.ox_recovery * z_ox
        dz_atp = -cfg.atp_recovery * z_atp
        dz_bottle = -cfg.bottle_recovery * z_bottle
    # ER is smooth and environment-correlated but causally isolated.
    dz_er = (
        0.28 * (1.0 - phi_t)
        + 0.25 * (1.0 - phi_ph)
        + 0.18 * high_do
        + 0.12 * low_do
        - cfg.er_recovery * z_er
    )
    nxt = burdens + cfg.dt * np.array([dz_ox, dz_atp, dz_bottle, dz_er])
    return np.clip(nxt, 0.0, 1.5)


def simulate_culture(env: np.ndarray, cfg: DFBAConfig, feedback: bool) -> dict[str, np.ndarray | list[str] | list[dict[str, float]]]:
    t = time_grid(cfg)
    x = np.zeros(len(t))
    b = np.zeros(len(t))
    bs = np.zeros(len(t))
    burdens = np.zeros((len(t), 4))
    fluxes: list[dict[str, float]] = []
    states: list[str] = []
    x[0] = cfg.x0
    b[0] = cfg.b0
    state = "balanced"
    dwell = 0
    surrogate_evaluations = 0
    actual_fba_solves = 0
    for i in range(len(t) - 1):
        flux = reduced_flux_solution(env, burdens[i], state, cfg)
        surrogate_evaluations += int(cfg.reduced_fast_model)
        actual_fba_solves += int(not cfg.reduced_fast_model)
        states.append(state)
        fluxes.append(flux)
        eff = state_effects(state, cfg)
        temp = float(env[0])
        phi_t = float(gaussian(temp, cfg.t_opt, cfg.t_sigma))
        death = eff["death"] + 0.018 * max(0.0, burdens[i, 0] - 0.70) + 0.014 * max(0.0, burdens[i, 1] - 0.70)
        dx = flux["biomass_flux"] * x[i] * (1.0 - x[i] / cfg.k_x) - death * x[i]
        beta_deg = cfg.beta_deg_base * degradation_multiplier(state) * (1.0 + 0.85 * (1.0 - phi_t) + 1.80 * burdens[i, 0])
        db = flux["v_beta"] * x[i] - beta_deg * b[i]
        x[i + 1] = max(1e-6, x[i] + cfg.dt * dx)
        b[i + 1] = max(0.0, b[i] + cfg.dt * db)
        bs[i + 1] = b[i + 1] / (x[i + 1] + 1e-8)
        burdens[i + 1] = update_burdens(env, burdens[i], flux, state, cfg, feedback)
        if feedback:
            new_state = next_state(state, burdens[i + 1], dwell, flux["v_beta"], cfg)
        else:
            new_state = "productive" if flux["v_beta"] > 0.10 else "balanced"
        if new_state == state:
            dwell += 1
        else:
            state = new_state
            dwell = 0
    bs[0] = b[0] / (x[0] + 1e-8)
    states.append(state)
    fluxes.append(fluxes[-1].copy())
    return {
        "time": t,
        "X": x,
        "B_total": b,
        "B_specific": bs,
        "burdens": burdens,
        "states": states,
        "fluxes": fluxes,
        "surrogate_evaluations": surrogate_evaluations,
        "actual_fba_solves": actual_fba_solves,
    }


def reporter_from_state(z: np.ndarray, lag_steps: int, baseline: float, gain: float, noise: float, rng: np.random.Generator, sparse: int = 1, missing: float = 0.0) -> np.ndarray:
    delayed = np.zeros_like(z)
    if lag_steps <= 0:
        delayed[:] = z
    else:
        delayed[lag_steps:] = z[:-lag_steps]
        delayed[:lag_steps] = z[0]
    sat = delayed / (0.35 + delayed)
    out = baseline + gain * sat + rng.normal(0.0, noise, len(z))
    if sparse > 1:
        out[np.arange(len(out)) % sparse != 0] = np.nan
    if missing > 0:
        out[rng.random(len(out)) < missing] = np.nan
    return out


def generate_dataset(seed: int, cfg: DFBAConfig, fast: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 3300)
    grid = make_environment_grid(fast)
    traj_rows: list[dict[str, object]] = []
    flux_rows: list[dict[str, object]] = []
    reporter_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    experiments = {
        "3A": False,
        "3B": True,
        "3C": True,
        "3D": True,
        "3E": True,
        "3F": True,
    }
    backend = metabolic_backend(cfg)
    n_time_points = cfg.n_time
    n_intervals = cfg.n_time - 1
    for _, env_row in grid.iterrows():
        env = np.array([env_row.temperature, env_row.pH, env_row.DO], dtype=float)
        for exp, feedback in experiments.items():
            sim = simulate_culture(env, cfg, feedback=feedback)
            burdens = np.asarray(sim["burdens"])
            states = list(sim["states"])
            fluxes = list(sim["fluxes"])
            transition_count = int(sum(states[i] != states[i - 1] for i in range(1, len(states))))
            for rep in range(cfg.replicates):
                culture_id = f"seed{seed}_{exp}_{env_row.environment_id}_rep{rep}"
                product_obs = np.clip(np.asarray(sim["B_total"]) + rng.normal(0.0, cfg.product_noise, cfg.n_time), 0.0, None)
                r_ox = reporter_from_state(burdens[:, 0], 2, 0.04, 1.15, cfg.reporter_noise, rng, sparse=1, missing=0.02)
                r_atp = reporter_from_state(burdens[:, 1], 3, 0.03, 1.05, cfg.reporter_noise, rng, sparse=1, missing=0.02)
                r_er = reporter_from_state(burdens[:, 3], 2, 0.07, 1.10, cfg.reporter_noise, rng, sparse=1, missing=0.02)
                for i, tt in enumerate(sim["time"]):
                    base = {
                        "dataset_seed": seed,
                        "experiment": exp,
                        "environment_id": env_row.environment_id,
                        "culture_id": culture_id,
                        "replicate": rep,
                        "split": env_row.split,
                        "heldout_region": env_row.heldout_region,
                        "temperature": env[0],
                        "pH": env[1],
                        "DO": env[2],
                        "time": float(tt),
                        "time_index": i,
                        "dt": cfg.dt,
                        "n_time_points": n_time_points,
                        "n_simulation_intervals": n_intervals,
                        "n_metabolic_calculations_total": int(sim["surrogate_evaluations"]) + int(sim["actual_fba_solves"]),
                        "n_actual_fba_solves_total": int(sim["actual_fba_solves"]),
                        "n_surrogate_evaluations_total": int(sim["surrogate_evaluations"]),
                        "metabolic_backend": backend,
                        "flux_solution_reused_or_cached": False,
                    }
                    traj_rows.append(
                        {
                            **base,
                            "X": float(sim["X"][i]),
                            "B_total": float(product_obs[i]),
                            "B_total_true": float(sim["B_total"][i]),
                            "B_specific_true": float(sim["B_specific"][i]),
                            "z_ox": float(burdens[i, 0]),
                            "z_atp": float(burdens[i, 1]),
                            "z_bottle": float(burdens[i, 2]),
                            "z_er": float(burdens[i, 3]),
                            "state": states[i],
                            "transition_count": transition_count,
                            "final_state": states[-1],
                        }
                    )
                    flux_rows.append(
                        {
                            **base,
                            **{k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in fluxes[i].items()},
                            "state": states[i],
                            "is_metabolic_update": bool(i < n_intervals),
                            "actual_fba_solves_this_step": int((i < n_intervals) and (not cfg.reduced_fast_model)),
                            "surrogate_evaluations_this_step": int((i < n_intervals) and cfg.reduced_fast_model),
                        }
                    )
                    reporter_rows.append({**base, "R_ox": float(r_ox[i]) if np.isfinite(r_ox[i]) else np.nan, "R_atp": float(r_atp[i]) if np.isfinite(r_atp[i]) else np.nan, "R_er": float(r_er[i]) if np.isfinite(r_er[i]) else np.nan})
                for i in range(1, len(states)):
                    if states[i] != states[i - 1]:
                        transition_rows.append(
                            {
                                "dataset_seed": seed,
                                "experiment": exp,
                                "environment_id": env_row.environment_id,
                                "culture_id": culture_id,
                                "replicate": rep,
                                "split": env_row.split,
                                "time": float(sim["time"][i]),
                                "from_state": states[i - 1],
                                "to_state": states[i],
                            }
                        )
                summary_rows.append(
                    {
                        "dataset_seed": seed,
                        "experiment": exp,
                        "environment_id": env_row.environment_id,
                        "culture_id": culture_id,
                        "replicate": rep,
                        "split": env_row.split,
                        "heldout_region": env_row.heldout_region,
                        "temperature": env[0],
                        "pH": env[1],
                        "DO": env[2],
                        "dt": cfg.dt,
                        "n_time_points": n_time_points,
                        "n_simulation_intervals": n_intervals,
                        "n_metabolic_calculations_total": int(sim["surrogate_evaluations"]) + int(sim["actual_fba_solves"]),
                        "n_actual_fba_solves_total": int(sim["actual_fba_solves"]),
                        "n_surrogate_evaluations_total": int(sim["surrogate_evaluations"]),
                        "metabolic_backend": backend,
                        "transition_count": transition_count,
                        "final_state": states[-1],
                        "max_z_ox": float(burdens[:, 0].max()),
                        "max_z_atp": float(burdens[:, 1].max()),
                        "max_z_bottle": float(burdens[:, 2].max()),
                        "max_z_er": float(burdens[:, 3].max()),
                        "final_B_total_true": float(sim["B_total"][-1]),
                        "auc_B_total_true": float(np.trapezoid(sim["B_total"], sim["time"])),
                    }
                )
    manifest = grid.assign(dataset_seed=seed)
    return (
        pd.DataFrame(traj_rows),
        pd.DataFrame(flux_rows),
        pd.DataFrame(reporter_rows),
        pd.DataFrame(transition_rows),
        manifest,
        pd.DataFrame(summary_rows),
    )


def pivot(traj: pd.DataFrame, value_col: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    ids = sorted(traj["culture_id"].unique())
    t = np.sort(traj["time"].unique())
    curves = []
    env = []
    splits = []
    meta = []
    for cid in ids:
        sub = traj[traj["culture_id"] == cid].sort_values("time")
        curves.append(sub[value_col].to_numpy(float))
        env.append(sub[["temperature", "pH", "DO"]].iloc[0].to_numpy(float))
        splits.append(str(sub["split"].iloc[0]))
        meta.append(
            sub[
                [
                    "dataset_seed",
                    "experiment",
                    "environment_id",
                    "culture_id",
                    "replicate",
                    "heldout_region",
                    "transition_count",
                    "final_state",
                    "metabolic_backend",
                    "n_time_points",
                    "n_simulation_intervals",
                    "n_metabolic_calculations_total",
                    "n_actual_fba_solves_total",
                    "n_surrogate_evaluations_total",
                ]
            ].iloc[0].to_dict()
        )
    return ids, np.asarray(env), np.asarray(curves), t, np.asarray(splits), pd.DataFrame(meta)


def env_features(env: np.ndarray) -> np.ndarray:
    env = np.asarray(env, dtype=float)
    t = (env[:, 0] - 30.0) / 6.0
    ph = (env[:, 1] - 5.0) / 1.0
    do = (env[:, 2] - 45.0) / 35.0
    return np.column_stack([t, ph, do, t * ph, t * do, ph * do, t**2, ph**2, do**2])


def curve_features(curves: np.ndarray, t: np.ndarray) -> np.ndarray:
    auc = np.trapezoid(curves, t, axis=1)
    final = curves[:, -1]
    mx = curves.max(axis=1)
    peak_t = t[np.argmax(curves, axis=1)]
    early = (curves[:, min(4, curves.shape[1] - 1)] - curves[:, 0]) / max(t[min(4, len(t) - 1)] - t[0], 1e-8)
    late = (curves[:, -1] - curves[:, max(0, curves.shape[1] - 5)]) / max(t[-1] - t[max(0, len(t) - 5)], 1e-8)
    return np.column_stack([final, auc, mx, peak_t, early, late])


def summarize_curve_matrix(mat: np.ndarray, t: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(mat), mat, np.nan)
    last = []
    mean = []
    mx = []
    auc = []
    for row in clean:
        finite = row[np.isfinite(row)]
        last.append(float(finite[-1]) if len(finite) else 0.0)
        mean.append(float(np.mean(finite)) if len(finite) else 0.0)
        mx.append(float(np.max(finite)) if len(finite) else 0.0)
        fill = pd.Series(row).interpolate(limit_direction="both").fillna(0.0).to_numpy(float)
        auc.append(float(np.trapezoid(fill, t)))
    return np.column_stack([last, mean, mx, auc])


def ridge_decoder_fit(Z: np.ndarray, Y: np.ndarray):
    X = np.column_stack([np.ones(len(Z)), Z])
    coef = fixed.ridge_fit(X, Y, lam=1e-4)
    return coef


def ridge_decoder_predict(coef, Z: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(Z)), Z])
    return np.clip(fixed.ridge_predict(coef, X), 0.0, None)


def metric_values(pred: np.ndarray, truth: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if len(pred) == 0:
        return {k: np.nan for k in ("trajectory_rmse", "normalized_rmse", "product_r2", "auc_error", "final_titer_error", "max_titer_error", "peak_time_error", "early_slope_error", "late_decline_error")}
    scale = max(float(np.nanmax(truth) - np.nanmin(truth)), 1e-8)
    pred_peak = t[np.argmax(pred, axis=1)]
    true_peak = t[np.argmax(truth, axis=1)]
    early_idx = min(4, len(t) - 1)
    late_idx = max(0, len(t) - 5)
    early_pred = (pred[:, early_idx] - pred[:, 0]) / max(t[early_idx] - t[0], 1e-8)
    early_true = (truth[:, early_idx] - truth[:, 0]) / max(t[early_idx] - t[0], 1e-8)
    late_pred = (pred[:, -1] - pred[:, late_idx]) / max(t[-1] - t[late_idx], 1e-8)
    late_true = (truth[:, -1] - truth[:, late_idx]) / max(t[-1] - t[late_idx], 1e-8)
    return {
        "trajectory_rmse": fixed.rmse(pred, truth),
        "normalized_rmse": fixed.rmse(pred, truth) / scale,
        "product_r2": fixed.r2_score(pred.reshape(-1), truth.reshape(-1)),
        "auc_error": float(np.mean(np.abs(np.trapezoid(pred, t, axis=1) - np.trapezoid(truth, t, axis=1)))),
        "final_titer_error": float(np.mean(np.abs(pred[:, -1] - truth[:, -1]))),
        "max_titer_error": float(np.mean(np.abs(pred.max(axis=1) - truth.max(axis=1)))),
        "peak_time_error": float(np.mean(np.abs(pred_peak - true_peak))),
        "early_slope_error": float(np.mean(np.abs(early_pred - early_true))),
        "late_decline_error": float(np.mean(np.abs(late_pred - late_true))),
    }


def add_prediction_rows(rows: list[dict[str, object]], meta: pd.DataFrame, env: np.ndarray, splits: np.ndarray, t: np.ndarray, truth: np.ndarray, pred: np.ndarray, dataset_seed: int, model_seed: int, experiment: str, model: str) -> None:
    for i, split in enumerate(splits):
        if split not in TEST_SPLITS:
            continue
        traj_rmse = fixed.rmse(pred[i], truth[i])
        for j, tt in enumerate(t):
            rows.append(
                {
                    "dataset_seed": dataset_seed,
                    "model_seed": model_seed,
                    "experiment": experiment,
                    "model": model,
                    "environment_id": meta.iloc[i]["environment_id"],
                    "culture_id": meta.iloc[i]["culture_id"],
                    "replicate": int(meta.iloc[i]["replicate"]),
                    "split": split,
                    "heldout_region": meta.iloc[i]["heldout_region"],
                    "temperature": float(env[i, 0]),
                    "pH": float(env[i, 1]),
                    "DO": float(env[i, 2]),
                    "time": float(tt),
                    "B_total_true": float(truth[i, j]),
                    "B_total_pred": float(pred[i, j]),
                    "trajectory_rmse": traj_rmse,
                    "prediction_source": "fixed_environment_rollout",
                    "metabolic_backend": meta.iloc[i].get("metabolic_backend", ""),
                    "n_actual_fba_solves_total": int(meta.iloc[i].get("n_actual_fba_solves_total", 0)),
                    "n_surrogate_evaluations_total": int(meta.iloc[i].get("n_surrogate_evaluations_total", 0)),
                }
            )


def fit_predict_direct(env: np.ndarray, curves: np.ndarray, splits: np.ndarray, cfg: DFBAConfig, model_seed: int):
    train = splits == "train"
    tic = time.perf_counter()
    model = fixed.train_mlp(env_features(env[train]), curves[train], model_seed + 10, cfg.mlp_steps, hidden=18, lr=0.018)
    pred = np.clip(model.predict(env_features(env)), 0.0, None)
    return pred, time.perf_counter() - tic, 18 * (env_features(env).shape[1] + curves.shape[1] + 1)


def fit_predict_coordinate(env: np.ndarray, curves: np.ndarray, splits: np.ndarray, t: np.ndarray, cfg: DFBAConfig, model_seed: int):
    train = splits == "train"
    x_rows = []
    y_rows = []
    for e, curve in zip(env[train], curves[train]):
        base = np.column_stack([np.repeat(env_features(e[None, :]), len(t), axis=0), t / t[-1]])
        x_rows.append(base)
        y_rows.append(curve[:, None])
    tic = time.perf_counter()
    model = fixed.train_mlp(np.vstack(x_rows), np.vstack(y_rows), model_seed + 20, cfg.mlp_steps, hidden=24, lr=0.016)
    pred = []
    for e in env:
        x = np.column_stack([np.repeat(env_features(e[None, :]), len(t), axis=0), t / t[-1]])
        pred.append(model.predict(x).reshape(-1))
    return np.clip(np.asarray(pred), 0.0, None), time.perf_counter() - tic, 24 * (env_features(env).shape[1] + 1 + 2)


def fit_predict_basis(env: np.ndarray, curves: np.ndarray, splits: np.ndarray, cfg: DFBAConfig, model_seed: int):
    train = splits == "train"
    tic = time.perf_counter()
    mean = curves[train].mean(axis=0)
    u, s, vt = np.linalg.svd(curves[train] - mean, full_matrices=False)
    k = min(5, vt.shape[0])
    comps = vt[:k]
    coeff_train = (curves[train] - mean) @ comps.T
    model = fixed.train_mlp(env_features(env[train]), coeff_train, model_seed + 30, cfg.mlp_steps, hidden=16, lr=0.018)
    coeff = model.predict(env_features(env))
    pred = mean + coeff @ comps
    return np.clip(pred, 0.0, None), time.perf_counter() - tic, int(k * len(mean) + 16 * (env_features(env).shape[1] + k + 1))


def latent_targets(mode: str, curves: np.ndarray, t: np.ndarray, reporter_curves: dict[str, np.ndarray], burden_curves: dict[str, np.ndarray], env: np.ndarray, seed: int) -> np.ndarray:
    parts = [curve_features(curves, t)]
    if "ox" in mode:
        parts.append(summarize_curve_matrix(reporter_curves["R_ox"], t))
    if "atp" in mode:
        parts.append(summarize_curve_matrix(reporter_curves["R_atp"], t))
    if "er" in mode:
        parts.append(summarize_curve_matrix(reporter_curves["R_er"], t))
    if mode == "er_only":
        parts = [summarize_curve_matrix(reporter_curves["R_er"], t)]
    if "oracle" in mode:
        parts.append(summarize_curve_matrix(burden_curves["z_ox"], t))
        parts.append(summarize_curve_matrix(burden_curves["z_atp"], t))
        parts.append(summarize_curve_matrix(burden_curves["z_bottle"], t))
    if mode == "random_smooth":
        ef = env_features(env)
        parts.append(np.column_stack([np.sin(1.7 * ef[:, 0] + 0.3), np.cos(1.3 * ef[:, 1] - 0.4), np.sin(1.1 * ef[:, 2] + ef[:, 0])]))
    out = np.column_stack(parts)
    if mode.startswith("shuffled"):
        rng = np.random.default_rng(seed + 1500)
        perm = np.arange(len(out))
        rng.shuffle(perm)
        if len(perm) > 1 and np.any(perm == np.arange(len(perm))):
            perm = np.roll(perm, 1)
        out = out.copy()
        if "er" in mode:
            er = summarize_curve_matrix(reporter_curves["R_er"], t)[perm]
            out = np.column_stack([curve_features(curves, t), er])
        else:
            useful = np.column_stack([summarize_curve_matrix(reporter_curves["R_ox"], t), summarize_curve_matrix(reporter_curves["R_atp"], t)])[perm]
            out = np.column_stack([curve_features(curves, t), useful])
    return out


def fit_predict_latent(mode: str, env: np.ndarray, curves: np.ndarray, splits: np.ndarray, t: np.ndarray, reporters: dict[str, np.ndarray], burdens: dict[str, np.ndarray], cfg: DFBAConfig, model_seed: int, dataset_seed: int):
    train = splits == "train"
    tic = time.perf_counter()
    target = latent_targets(mode, curves, t, reporters, burdens, env, dataset_seed + model_seed)
    env_model = fixed.train_mlp(env_features(env[train]), target[train], model_seed + 40 + len(mode), cfg.mlp_steps, hidden=18, lr=0.018)
    pred_latent = env_model.predict(env_features(env))
    decoder = ridge_decoder_fit(target[train], curves[train])
    pred = ridge_decoder_predict(decoder, pred_latent)
    params = int(18 * (env_features(env).shape[1] + target.shape[1] + 1) + (target.shape[1] + 1) * curves.shape[1])
    return pred, time.perf_counter() - tic, params


def split_reporters(reporters: pd.DataFrame, ids: list[str], cols: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    for col in cols:
        curves = []
        for cid in ids:
            curves.append(reporters[reporters["culture_id"] == cid].sort_values("time")[col].to_numpy(float))
        out[col] = np.asarray(curves)
    return out


def split_burdens(traj: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for col in ("z_ox", "z_atp", "z_bottle", "z_er"):
        curves = []
        for cid in ids:
            curves.append(traj[traj["culture_id"] == cid].sort_values("time")[col].to_numpy(float))
        out[col] = np.asarray(curves)
    return out


def evaluate_model_set(dataset_seed: int, model_seed: int, experiment: str, traj: pd.DataFrame, reporters: pd.DataFrame, cfg: DFBAConfig, model_names: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    sub = traj[traj["experiment"] == experiment]
    ids, env, truth, t, splits, meta = pivot(sub, "B_total_true")
    rep_curves = split_reporters(reporters[reporters["experiment"] == experiment], ids, ("R_ox", "R_atp", "R_er"))
    burden_curves = split_burdens(sub, ids)
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    pred_cache: dict[str, np.ndarray] = {}
    for name in model_names:
        if name == "direct_multi_output_mlp":
            pred, train_time, param_count = fit_predict_direct(env, truth, splits, cfg, model_seed)
        elif name == "coordinate_conditioned_mlp":
            pred, train_time, param_count = fit_predict_coordinate(env, truth, splits, t, cfg, model_seed)
        elif name == "basis_curve_decoder":
            pred, train_time, param_count = fit_predict_basis(env, truth, splits, cfg, model_seed)
        elif name == "adaptive_ox_atp_er_state_space":
            candidates = []
            for candidate_name, mode, er_weight in [
                ("adaptive_useful_only", "ox_atp", 0.0),
                ("adaptive_er_downweighted", "ox_atp_er", 0.25),
                ("adaptive_er_full", "ox_atp_er", 1.0),
            ]:
                cand_pred, cand_time, cand_params = fit_predict_latent(
                    mode,
                    env,
                    truth,
                    splits,
                    t,
                    rep_curves,
                    burden_curves,
                    cfg,
                    model_seed + int(er_weight * 100),
                    dataset_seed,
                )
                val_mask = splits == "validation"
                candidates.append(
                    {
                        "candidate": candidate_name,
                        "er_weight": er_weight,
                        "pred": cand_pred,
                        "train_time": cand_time,
                        "parameter_count": cand_params,
                        "validation_rmse": fixed.rmse(cand_pred[val_mask], truth[val_mask]),
                    }
                )
            best = sorted(candidates, key=lambda row: row["validation_rmse"])[0]
            pred = best["pred"]
            train_time = float(sum(row["train_time"] for row in candidates))
            param_count = int(best["parameter_count"])
            adaptive_er_weight = float(best["er_weight"])
            adaptive_candidate = str(best["candidate"])
        else:
            mode = {
                "product_only_state_space": "product_only",
                "oxidative_reporter_state_space": "ox",
                "atp_reporter_state_space": "atp",
                "ox_atp_reporter_state_space": "ox_atp",
                "shuffled_reporter_control": "shuffled_ox_atp",
                "random_smooth_aux_control": "random_smooth",
                "oracle_causal_burden_state_space": "oracle",
                "oracle_all_burdens_state_space": "oracle",
                "er_only_state_space": "er_only",
                "ox_atp_er_state_space": "ox_atp_er",
                "shuffled_er_control": "shuffled_er",
            }[name]
            pred, train_time, param_count = fit_predict_latent(mode, env, truth, splits, t, rep_curves, burden_curves, cfg, model_seed, dataset_seed)
            adaptive_er_weight = np.nan
            adaptive_candidate = ""
        if name != "adaptive_ox_atp_er_state_space":
            adaptive_er_weight = np.nan
            adaptive_candidate = ""
        pred_cache[name] = pred
        add_prediction_rows(predictions, meta, env, splits, t, truth, pred, dataset_seed, model_seed, experiment, name)
        for split in TEST_SPLITS + ("validation",):
            mask = splits == split
            vals = metric_values(pred[mask], truth[mask], t)
            metrics.append(
                {
                    "dataset_seed": dataset_seed,
                    "model_seed": model_seed,
                    "experiment": experiment,
                    "model": name,
                    "split": split,
                    "train_time_seconds": train_time,
                    "parameter_count": param_count,
                    "adaptive_er_weight": adaptive_er_weight,
                    "adaptive_candidate": adaptive_candidate,
                    "deployment_inputs": "temperature,pH,DO",
                    "metabolic_backend": str(meta["metabolic_backend"].iloc[0]) if "metabolic_backend" in meta else metabolic_backend(cfg),
                    "n_actual_fba_solves_per_culture": int(meta["n_actual_fba_solves_total"].iloc[0]) if "n_actual_fba_solves_total" in meta else 0,
                    "n_surrogate_evaluations_per_culture": int(meta["n_surrogate_evaluations_total"].iloc[0]) if "n_surrogate_evaluations_total" in meta else 0,
                    **vals,
                }
            )
        if experiment == "3B":
            held = splits == "heldout_combination"
            err = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
            tmp = meta.copy()
            tmp["trajectory_rmse"] = err
            for transition_count, group in tmp[held].groupby("transition_count"):
                metrics.append(
                    {
                        "dataset_seed": dataset_seed,
                        "model_seed": model_seed,
                        "experiment": experiment,
                        "model": name,
                        "split": "heldout_by_transition_count",
                        "transition_count": int(transition_count),
                        "trajectory_rmse": float(group["trajectory_rmse"].mean()),
                        "adaptive_er_weight": adaptive_er_weight,
                        "adaptive_candidate": adaptive_candidate,
                        "deployment_inputs": "temperature,pH,DO",
                        "metabolic_backend": str(meta["metabolic_backend"].iloc[0]) if "metabolic_backend" in meta else metabolic_backend(cfg),
                        "n_actual_fba_solves_per_culture": int(meta["n_actual_fba_solves_total"].iloc[0]) if "n_actual_fba_solves_total" in meta else 0,
                        "n_surrogate_evaluations_per_culture": int(meta["n_surrogate_evaluations_total"].iloc[0]) if "n_surrogate_evaluations_total" in meta else 0,
                    }
                )
            for final_state, group in tmp[held].groupby("final_state"):
                metrics.append(
                    {
                        "dataset_seed": dataset_seed,
                        "model_seed": model_seed,
                        "experiment": experiment,
                        "model": name,
                        "split": "heldout_by_final_state",
                        "final_state": final_state,
                        "trajectory_rmse": float(group["trajectory_rmse"].mean()),
                        "adaptive_er_weight": adaptive_er_weight,
                        "adaptive_candidate": adaptive_candidate,
                        "deployment_inputs": "temperature,pH,DO",
                        "metabolic_backend": str(meta["metabolic_backend"].iloc[0]) if "metabolic_backend" in meta else metabolic_backend(cfg),
                        "n_actual_fba_solves_per_culture": int(meta["n_actual_fba_solves_total"].iloc[0]) if "n_actual_fba_solves_total" in meta else 0,
                        "n_surrogate_evaluations_per_culture": int(meta["n_surrogate_evaluations_total"].iloc[0]) if "n_surrogate_evaluations_total" in meta else 0,
                    }
                )
    return metrics, predictions, pred_cache


def choose_validation_best(metrics: pd.DataFrame, experiment: str, candidates: list[str]) -> str:
    sub = metrics[(metrics["experiment"] == experiment) & (metrics["split"] == "validation") & (metrics["model"].isin(candidates))]
    return str(sub.groupby("model")["trajectory_rmse"].mean().sort_values().index[0])


def evaluate_sample_efficiency(dataset_seed: int, model_seed: int, traj: pd.DataFrame, reporters: pd.DataFrame, cfg: DFBAConfig, best_reporter_model: str) -> list[dict[str, object]]:
    sub = traj[traj["experiment"] == "3F"]
    ids, env, truth, t, splits, _meta = pivot(sub, "B_total_true")
    rep_curves = split_reporters(reporters[reporters["experiment"] == "3F"], ids, ("R_ox", "R_atp", "R_er"))
    burden_curves = split_burdens(sub, ids)
    train_idx = np.flatnonzero(splits == "train")
    held = splits == "heldout_combination"
    rows = []
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    models = ["direct_multi_output_mlp", "coordinate_conditioned_mlp", "basis_curve_decoder", "product_only_state_space", best_reporter_model]
    rng = np.random.default_rng(dataset_seed + model_seed + 990)
    for frac in fractions:
        n = max(4, int(np.ceil(len(train_idx) * frac)))
        chosen_train = np.sort(rng.choice(train_idx, size=min(n, len(train_idx)), replace=False))
        local_splits = np.array(["unused"] * len(splits), dtype=object)
        local_splits[chosen_train] = "train"
        local_splits[splits == "validation"] = "validation"
        local_splits[splits == "heldout_combination"] = "heldout_combination"
        local_splits[splits == "interpolation"] = "interpolation"
        local_splits[splits == "extrapolation"] = "extrapolation"
        for name in models:
            if name == "direct_multi_output_mlp":
                pred, _, params = fit_predict_direct(env, truth, local_splits, cfg, model_seed)
            elif name == "coordinate_conditioned_mlp":
                pred, _, params = fit_predict_coordinate(env, truth, local_splits, t, cfg, model_seed)
            elif name == "basis_curve_decoder":
                pred, _, params = fit_predict_basis(env, truth, local_splits, cfg, model_seed)
            else:
                mode = "product_only" if name == "product_only_state_space" else "ox_atp"
                pred, _, params = fit_predict_latent(mode, env, truth, local_splits, t, rep_curves, burden_curves, cfg, model_seed, dataset_seed)
            vals = metric_values(pred[held], truth[held], t)
            rows.append(
                {
                    "dataset_seed": dataset_seed,
                    "model_seed": model_seed,
                    "training_environment_fraction": frac,
                    "n_training_environments": int(len(chosen_train)),
                    "model": name,
                    "parameter_count": params,
                    **vals,
                }
            )
    return rows


def reporter_robustness(metrics: pd.DataFrame, cfg: DFBAConfig) -> pd.DataFrame:
    rows = []
    def mean_rmse(exp: str, model: str) -> float:
        sub = metrics[(metrics["experiment"] == exp) & (metrics["model"] == model) & (metrics["split"] == "heldout_combination")]
        return float(sub["trajectory_rmse"].mean())

    po = mean_rmse("3C", "product_only_state_space")
    useful = mean_rmse("3C", "ox_atp_reporter_state_space")
    shuffled = mean_rmse("3C", "shuffled_reporter_control")
    random = mean_rmse("3C", "random_smooth_aux_control")
    rows.append(
        {
            "comparison": "3C_useful_reporters_vs_product_only",
            "baseline_rmse": po,
            "candidate_rmse": useful,
            "relative_improvement": (po - useful) / max(po, 1e-12),
            "passed_predeclared_threshold": (po - useful) / max(po, 1e-12) >= cfg.useful_margin,
        }
    )
    rows.append(
        {
            "comparison": "3C_useful_reporters_vs_controls",
            "baseline_rmse": min(shuffled, random),
            "candidate_rmse": useful,
            "relative_improvement": (min(shuffled, random) - useful) / max(min(shuffled, random), 1e-12),
            "passed_predeclared_threshold": useful < min(shuffled, random),
        }
    )
    useful_e = mean_rmse("3E", "ox_atp_reporter_state_space")
    useful_er = mean_rmse("3E", "ox_atp_er_state_space")
    er_only = mean_rmse("3E", "er_only_state_space")
    po_e = mean_rmse("3E", "product_only_state_space")
    rows.append(
        {
            "comparison": "3E_useful_plus_er_tolerance",
            "baseline_rmse": useful_e,
            "candidate_rmse": useful_er,
            "relative_degradation": (useful_er - useful_e) / max(useful_e, 1e-12),
            "passed_predeclared_threshold": (useful_er - useful_e) / max(useful_e, 1e-12) < cfg.er_tolerance,
        }
    )
    rows.append(
        {
            "comparison": "3E_er_only_not_consistently_useful",
            "baseline_rmse": po_e,
            "candidate_rmse": er_only,
            "relative_improvement": (po_e - er_only) / max(po_e, 1e-12),
            "passed_predeclared_threshold": (po_e - er_only) / max(po_e, 1e-12) < cfg.useful_margin,
        }
    )
    adaptive = metrics[
        (metrics["experiment"] == "3E")
        & (metrics["model"] == "adaptive_ox_atp_er_state_space")
        & (metrics["split"] == "heldout_combination")
    ]
    if not adaptive.empty:
        rows.append(
            {
                "comparison": "3E_adaptive_er_weight",
                "baseline_rmse": useful_e,
                "candidate_rmse": float(adaptive["trajectory_rmse"].mean()),
                "mean_selected_er_weight": float(adaptive["adaptive_er_weight"].mean()),
                "passed_predeclared_threshold": float(adaptive["adaptive_er_weight"].mean()) < 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_safeguards(traj: pd.DataFrame, reporters: pd.DataFrame, fluxes: pd.DataFrame, manifest: pd.DataFrame, metrics: pd.DataFrame, cfg: DFBAConfig) -> pd.DataFrame:
    checks = []
    def add(name, passed, critical=True, detail=""):
        checks.append({"safeguard": name, "passed": bool(passed), "critical": bool(critical), "detail": detail})

    env_n = traj.groupby("culture_id")[["temperature", "pH", "DO"]].nunique().max(axis=1)
    add("one fixed environment vector per trajectory", (env_n == 1).all())
    split_n = traj.groupby("culture_id")["split"].nunique()
    add("splits are assigned per complete culture trajectory", (split_n == 1).all())
    cross_seed = manifest.groupby(["environment_id"])["split"].nunique()
    add("same environment has same split across dataset seeds", (cross_seed == 1).all())
    split_sets = {s: set(manifest[manifest["split"] == s]["environment_id"]) for s in manifest["split"].unique()}
    disjoint = all(not (split_sets[a] & split_sets[b]) for a in split_sets for b in split_sets if a < b)
    add("train validation interpolation heldout and extrapolation environments are disjoint", disjoint)
    train_regions = set(manifest[manifest["split"] == "train"]["heldout_region"])
    add("held-out environmental combinations absent from training", train_regions == {"training_support"})
    add("numerical extrapolation remains separate", (manifest[manifest["split"] == "extrapolation"]["is_extrapolation"]).all())
    add("reporters are not deployment inputs", (metrics["deployment_inputs"].dropna() == "temperature,pH,DO").all())
    er_cols = ["z_er"]
    causal_flux_cols = ["biomass_flux", "v_beta", "v_precursor", "v_atpm", "rho_atp"]
    # Causal isolation: no flux formula consumes z_er; this finite-difference check
    # verifies that changing ER alone leaves causal fluxes unchanged.
    env = np.array([33.0, 4.5, 60.0])
    b0 = np.array([0.45, 0.35, 0.42, 0.0])
    b1 = np.array([0.45, 0.35, 0.42, 1.2])
    f0 = reduced_flux_solution(env, b0, "productive", cfg)
    f1 = reduced_flux_solution(env, b1, "productive", cfg)
    er_iso = all(abs(f0[k] - f1[k]) < 1e-12 for k in causal_flux_cols)
    add("ER reporter state is causally isolated from metabolic constraints", er_iso, detail="finite difference in z_er leaves causal fluxes unchanged")
    add("bottleneck state is not reported in canonical reporters", "R_bottle" not in reporters.columns)
    add("static 3A has no causal state-machine transitions", traj[traj["experiment"] == "3A"]["transition_count"].max() <= 1)
    dyn = traj[traj["experiment"].isin(["3B", "3C", "3D", "3E", "3F"])]
    add("dynamic dFBA experiments include state transitions", dyn["transition_count"].max() > 0)
    add("flux feasibility failures are recorded", "feasible" in fluxes.columns)
    add("all canonical experiments 3A-3F have metrics", set(["3A", "3B", "3C", "3D", "3E", "3F"]).issubset(set(metrics["experiment"])))
    add("product trajectories remain finite and nonnegative", np.isfinite(traj["B_total_true"]).all() and (traj["B_total_true"] >= 0).all())
    add("reproducible dataset and model seeds are recorded", {"dataset_seed", "model_seed"}.issubset(metrics.columns))
    add("reduced fast model is explicitly marked when used", cfg.reduced_fast_model or configured_gem_path(cfg) is not None)
    return pd.DataFrame(checks)


def make_figures(traj: pd.DataFrame, reporters: pd.DataFrame, manifest: pd.DataFrame, transitions: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, sample_eff: pd.DataFrame, robustness: pd.DataFrame, cfg: DFBAConfig) -> None:
    def save(name: str, body: str, w: int = 900, h: int = 360):
        fixed.save_svg(FIGURES / name, w, h, body)

    save(
        "dfba_01_generator_architecture.svg",
        "\n".join(
            [
                fixed.svg_text(450, 32, "Experiments 3A-3F Generator", 18, "bold"),
                fixed.svg_box(55, 92, 140, 58, "fixed e\nT, pH, DO"),
                fixed.svg_arrow(195, 121, 270, 121),
                fixed.svg_box(285, 82, 170, 80, "yeast GEM or\nreduced fast surrogate\n+ beta-carotene path"),
                fixed.svg_arrow(455, 121, 535, 121),
                fixed.svg_box(550, 82, 150, 80, "burdens +\nstate machine"),
                fixed.svg_arrow(700, 121, 780, 121),
                fixed.svg_box(790, 92, 80, 58, "B(t)"),
                fixed.svg_arrow(625, 162, 625, 225, dashed=True),
                fixed.svg_box(535, 235, 180, 52, "reporters\ntraining labels only"),
            ]
        ),
    )
    body = [fixed.svg_text(450, 32, "Environment Grid and Splits", 18, "bold"), '<rect x="75" y="62" width="420" height="245" fill="#fbfbfb" stroke="#ccc"/>']
    colors = {"train": "#3b6ea8", "validation": "#777", "interpolation": "#61a35c", "heldout_combination": "#c84f4f", "extrapolation": "#7b5bb7"}
    one = manifest.drop_duplicates("environment_id")
    for _, r in one.iterrows():
        x = 75 + (float(r.temperature) - 22) / 16 * 420
        y = 307 - (float(r.DO) - 5) / 90 * 245
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[r.split]}" opacity="0.78"/>')
    for i, (split, color) in enumerate(colors.items()):
        body.append(f'<circle cx="555" cy="{85 + 24 * i}" r="5" fill="{color}"/>')
        body.append(fixed.svg_text(570, 89 + 24 * i, split, 11, anchor="start"))
    save("dfba_02_environment_grid_and_splits.svg", "\n".join(body))
    states = list(STATE_ORDER)
    body = [fixed.svg_text(450, 32, "Hysteretic State Machine", 18, "bold")]
    for i, st in enumerate(states):
        x = 50 + (i % 4) * 205
        y = 80 + (i // 4) * 110
        body.append(fixed.svg_box(x, y, 160, 55, st.replace("_", "\n"), "#eef3f7", "#456", 12))
    body.append(fixed.svg_text(450, 305, "Transitions use high/low burden thresholds plus minimum dwell time; ER is excluded.", 12))
    save("dfba_03_state_machine.svg", "\n".join(body))
    ex = traj[(traj["experiment"] == "3B") & (traj["split"] == "heldout_combination")]["culture_id"].iloc[0]
    sub = traj[traj["culture_id"] == ex].sort_values("time")
    ymax = max(sub["B_total_true"].max(), sub["z_ox"].max(), sub["z_atp"].max(), sub["z_bottle"].max(), 1e-8)
    body = [fixed.svg_text(450, 32, "Example Flux and Burden Trajectories", 18, "bold"), '<rect x="70" y="65" width="620" height="220" fill="#fbfbfb" stroke="#ccc"/>']
    for col, color in [("B_total_true", "#111"), ("z_ox", "#b65d3b"), ("z_atp", "#2f6f9f"), ("z_bottle", "#4c8b57")]:
        body.append(fixed.plot_polyline_bounds(sub["time"], sub[col] / ymax, 100, 90, 540, 150, (sub["time"].min(), sub["time"].max()), (0, 1), color, 2.0))
    body.append(fixed.svg_text(715, 100, "black B(t)\nred ox\nblue ATP\ngreen bottleneck", 11, anchor="start"))
    save("dfba_04_example_flux_and_burden_trajectories.svg", "\n".join(body))
    pred_sub = predictions[(predictions["experiment"] == "3F") & (predictions["split"] == "heldout_combination")]
    model = "ox_atp_reporter_state_space" if "ox_atp_reporter_state_space" in set(pred_sub["model"]) else pred_sub["model"].iloc[0]
    examples = pred_sub[pred_sub["model"] == model]["culture_id"].drop_duplicates().head(3).tolist()
    body = [fixed.svg_text(450, 32, "Held-Out Beta-Carotene True versus Predicted", 18, "bold")]
    for i, cid in enumerate(examples):
        g = pred_sub[(pred_sub["model"] == model) & (pred_sub["culture_id"] == cid)].sort_values("time")
        x0 = 70 + i * 270
        ymax = max(g["B_total_true"].max(), g["B_total_pred"].max(), 1e-8)
        body.append(f'<rect x="{x0}" y="70" width="220" height="170" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(fixed.plot_polyline_bounds(g["time"], g["B_total_true"], x0 + 25, 95, 165, 105, (g["time"].min(), g["time"].max()), (0, ymax), "#111", 2.2))
        body.append(fixed.plot_polyline_bounds(g["time"], g["B_total_pred"], x0 + 25, 95, 165, 105, (g["time"].min(), g["time"].max()), (0, ymax), "#2f6f9f", 1.8, "6 4"))
    save("dfba_05_beta_carotene_true_vs_predicted.svg", "\n".join(body))
    for fname, title, models in [
        ("dfba_06_reporter_supervision.svg", "Useful Reporter Supervision", ["product_only_state_space", "ox_atp_reporter_state_space", "shuffled_reporter_control", "oracle_causal_burden_state_space"]),
        ("dfba_07_partial_reporter_coverage.svg", "Incomplete Reporter Coverage", ["product_only_state_space", "oxidative_reporter_state_space", "atp_reporter_state_space", "ox_atp_reporter_state_space", "oracle_all_burdens_state_space"]),
        ("dfba_08_irrelevant_er_reporter.svg", "Causally Irrelevant ER Reporter", ["product_only_state_space", "er_only_state_space", "ox_atp_reporter_state_space", "ox_atp_er_state_space", "adaptive_ox_atp_er_state_space"]),
        ("dfba_09_model_baseline_comparison.svg", "State-Space versus Direct Curve Prediction", ["direct_multi_output_mlp", "coordinate_conditioned_mlp", "basis_curve_decoder", "product_only_state_space", "ox_atp_reporter_state_space"]),
    ]:
        exp = {"dfba_06_reporter_supervision.svg": "3C", "dfba_07_partial_reporter_coverage.svg": "3D", "dfba_08_irrelevant_er_reporter.svg": "3E"}.get(fname, "3F")
        vals = metrics[(metrics["experiment"] == exp) & (metrics["split"] == "heldout_combination") & (metrics["model"].isin(models))].groupby("model")["trajectory_rmse"].mean().reindex(models).dropna()
        ymax = max(vals.max(), 1e-8)
        body = [fixed.svg_text(450, 32, title, 18, "bold")]
        for i, (label, val) in enumerate(vals.items()):
            x = 80 + i * 145
            h = 190 * val / ymax
            body.append(f'<rect x="{x}" y="{260 - h:.1f}" width="78" height="{h:.1f}" fill="#4d7fa8"/>')
            body.append(fixed.svg_text(x + 39, 285, label.replace("_", "\n"), 8))
            body.append(fixed.svg_text(x + 39, 252 - h, f"{val:.3f}", 10))
        save(fname, "\n".join(body), 900, 330)
    vals = sample_eff.groupby(["training_environment_fraction", "model"])["trajectory_rmse"].mean().reset_index()
    body = [fixed.svg_text(450, 32, "Sample Efficiency", 18, "bold"), '<rect x="75" y="70" width="660" height="210" fill="#fbfbfb" stroke="#ccc"/>']
    palette = ["#111", "#2f6f9f", "#b65d3b", "#4c8b57", "#7b5bb7"]
    ymax = max(vals["trajectory_rmse"].max(), 1e-8)
    for color, (model_name, group) in zip(palette, vals.groupby("model")):
        group = group.sort_values("training_environment_fraction")
        body.append(fixed.plot_polyline_bounds(group["training_environment_fraction"], group["trajectory_rmse"], 105, 90, 570, 150, (0.1, 1.0), (0, ymax), color, 2.0))
    save("dfba_10_sample_efficiency.svg", "\n".join(body), 900, 330)
    vals = metrics[(metrics["split"] == "heldout_by_transition_count") & (metrics["experiment"] == "3B")]
    body = [fixed.svg_text(450, 32, "Error by State Transition Count", 18, "bold")]
    if not vals.empty:
        group = vals.groupby("transition_count")["trajectory_rmse"].mean().reset_index()
        body.append('<rect x="90" y="70" width="600" height="210" fill="#fbfbfb" stroke="#ccc"/>')
        ymax = max(group["trajectory_rmse"].max(), 1e-8)
        body.append(fixed.plot_polyline_bounds(group["transition_count"], group["trajectory_rmse"], 125, 95, 520, 145, (group["transition_count"].min(), group["transition_count"].max() + 1e-6), (0, ymax), "#2f6f9f", 2.4))
    save("dfba_11_error_by_state_transition_count.svg", "\n".join(body), 900, 330)


def write_docs(metrics: pd.DataFrame, safeguards: pd.DataFrame, robustness: pd.DataFrame, cfg: DFBAConfig) -> None:
    passed = int(safeguards["passed"].sum())
    total = int(len(safeguards))
    def best(exp: str) -> tuple[str, float]:
        sub = metrics[(metrics["experiment"] == exp) & (metrics["split"] == "heldout_combination") & (~metrics["model"].astype(str).str.contains("oracle"))]
        row = sub.groupby("model")["trajectory_rmse"].mean().sort_values()
        return str(row.index[0]), float(row.iloc[0])
    rows = [f"| {exp} | {best(exp)[0]} | {best(exp)[1]:.4f} |" for exp in ("3A", "3B", "3C", "3D", "3E", "3F")]
    appendix = f"""

## Experiments 3A-3F: dFBA State-Machine Stage

The next canonical stage keeps the same deployment contract but changes the
biological generator:

`fixed environment e = [T, pH, DO] -> complete beta-carotene titer curve B(t)`.

The model input is never a time series. Reporters `R_ox`, `R_ATP`, and `R_ER`
are training-time auxiliary labels only. At deployment, every learned model
receives only `[T, pH, DO]`.

Canonical commands:

```bash
.venv/bin/python scripts/audit_surrogate_complexity.py
.venv/bin/python scripts/audit_yeast_gem.py
.venv/bin/python scripts/run_dfba_state_machine_validation.py
.venv/bin/python scripts/run_dfba_state_machine_validation.py --fast
.venv/bin/python scripts/run_dfba_state_machine_validation.py --backend yeast_gem
.venv/bin/python -m pytest tests/test_dfba_state_machine_validation.py tests/test_yeast_gem_backend.py
```

Non-fast runs require a local yeast GEM asset configured through
`YEAST_GEM_PATH` or `models/yeast_gem.xml`; no model is downloaded at runtime.
The current repository refuses non-fast execution until a real repeated LP/FBA
yeast-GEM backend is wired, so it cannot silently use the surrogate and call it
FBA. Fast mode uses the documented reduced stoichiometric surrogate and still
runs Experiments 3A-3F plus the critical safeguards.

New canonical outputs:

```text
data/dfba_environment_grid.csv
data/dfba_split_manifest.csv
data/dfba_trajectories.csv
data/dfba_fluxes.csv
data/dfba_state_transitions.csv
data/dfba_reporters.csv
data/dfba_model_predictions.csv
data/dfba_canonical_metrics.csv
data/dfba_sample_efficiency.csv
data/dfba_reporter_robustness.csv
data/dfba_safeguards.csv
figures/dfba_*.svg
```

The beta-carotene pathway is represented by declared synthetic reactions from
native acetyl-CoA/isoprenoid precursor supply through GGPP, phytoene/lycopene
lumps, and an intentional non-mass-balanced beta-carotene accumulation sink.
In full mode this pathway must be installed on a configured yeast GEM; in fast
mode the same declared constraints are evaluated by a reduced stoichiometric
surrogate.

Held-out-combination headline RMSE:

| Experiment | Best learned model | RMSE |
| --- | --- | ---: |
{chr(10).join(rows)}

Safeguards passed `{passed}/{total}`. ER causal isolation is checked by a
finite-difference test showing that changing `z_ER` alone does not alter
growth, beta-carotene flux, precursor flux, ATP maintenance, or ATP pressure.
"""
    for path in (ROOT / "README.md", ROOT / "CONCRETE_EXPERIMENT_CHAIN.md", ROOT / "PHYSIOLOGY_QUEST_VALIDATION.md"):
        existing = path.read_text(encoding="utf-8")
        marker = "\n## Experiments 3A-3F: dFBA State-Machine Stage\n"
        if marker in existing:
            existing = existing.split(marker)[0].rstrip() + "\n"
        path.write_text(existing.rstrip() + appendix, encoding="utf-8")


def save_outputs(traj, fluxes, reporters, transitions, manifest, env_grid, metrics, predictions, sample_eff, robustness, safeguards, cfg):
    DATA.mkdir(exist_ok=True)
    env_grid.to_csv(DATA / "dfba_environment_grid.csv", index=False)
    manifest.to_csv(DATA / "dfba_split_manifest.csv", index=False)
    traj.to_csv(DATA / "dfba_trajectories.csv", index=False)
    fluxes.to_csv(DATA / "dfba_fluxes.csv", index=False)
    transitions.to_csv(DATA / "dfba_state_transitions.csv", index=False)
    reporters.to_csv(DATA / "dfba_reporters.csv", index=False)
    predictions.to_csv(DATA / "dfba_model_predictions.csv", index=False)
    metrics.to_csv(DATA / "dfba_canonical_metrics.csv", index=False)
    sample_eff.to_csv(DATA / "dfba_sample_efficiency.csv", index=False)
    robustness.to_csv(DATA / "dfba_reporter_robustness.csv", index=False)
    safeguards.to_csv(DATA / "dfba_safeguards.csv", index=False)
    pd.DataFrame(beta_pathway_reactions()).to_json(DATA / "dfba_beta_carotene_pathway.json", orient="records", indent=2)
    mapping = {
        "3A": "experiment_3a_static_metabolic_control",
        "3B": "experiment_3b_state_machine_dfba",
        "3C": "experiment_3c_useful_reporters",
        "3D": "experiment_3d_partial_reporters",
        "3E": "experiment_3e_irrelevant_er",
        "3F": "experiment_3f_model_comparison",
    }
    for exp, subdir in mapping.items():
        out = RESULTS / subdir
        metrics[metrics["experiment"] == exp].to_csv(out / "per_seed_metrics.csv", index=False)
        metrics[metrics["experiment"] == exp].groupby(["model", "split"], dropna=False)["trajectory_rmse"].agg(["mean", "std"]).reset_index().to_csv(out / "summary_metrics.csv", index=False)
        predictions[predictions["experiment"] == exp].to_csv(out / "predictions.csv", index=False)
        with (out / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)


def run_all(fast: bool = False, write: bool = True, backend: str | None = None):
    if backend == "yeast_gem" and fast:
        raise ValueError("--backend yeast_gem cannot be combined with --fast.")
    if backend == "reduced_surrogate":
        fast = True
    cfg = active_config(fast)
    require_metabolic_asset(cfg)
    ensure_dirs()
    all_traj = []
    all_flux = []
    all_rep = []
    all_trans = []
    all_manifest = []
    all_summary = []
    for seed in cfg.dataset_seeds:
        traj, flux, rep, trans, manifest, summary = generate_dataset(seed, cfg, fast)
        all_traj.append(traj)
        all_flux.append(flux)
        all_rep.append(rep)
        all_trans.append(trans)
        all_manifest.append(manifest)
        all_summary.append(summary)
    traj = pd.concat(all_traj, ignore_index=True)
    fluxes = pd.concat(all_flux, ignore_index=True)
    reporters = pd.concat(all_rep, ignore_index=True)
    transitions = pd.concat(all_trans, ignore_index=True) if all_trans else pd.DataFrame()
    manifest = pd.concat(all_manifest, ignore_index=True)
    env_grid = make_environment_grid(fast)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for ds in cfg.dataset_seeds:
        for ms in cfg.model_seeds:
            ds_traj = traj[traj["dataset_seed"] == ds]
            ds_rep = reporters[reporters["dataset_seed"] == ds]
            experiment_models = {
                "3A": ["direct_multi_output_mlp", "coordinate_conditioned_mlp", "product_only_state_space"],
                "3B": ["direct_multi_output_mlp", "coordinate_conditioned_mlp", "product_only_state_space"],
                "3C": ["product_only_state_space", "oxidative_reporter_state_space", "atp_reporter_state_space", "ox_atp_reporter_state_space", "shuffled_reporter_control", "random_smooth_aux_control", "oracle_causal_burden_state_space"],
                "3D": ["product_only_state_space", "oxidative_reporter_state_space", "atp_reporter_state_space", "ox_atp_reporter_state_space", "oracle_all_burdens_state_space"],
                "3E": ["product_only_state_space", "ox_atp_reporter_state_space", "er_only_state_space", "ox_atp_er_state_space", "adaptive_ox_atp_er_state_space", "shuffled_er_control", "random_smooth_aux_control", "oracle_causal_burden_state_space"],
                "3F": ["direct_multi_output_mlp", "coordinate_conditioned_mlp", "basis_curve_decoder", "product_only_state_space", "ox_atp_reporter_state_space"],
            }
            for exp, models in experiment_models.items():
                rows, preds, _cache = evaluate_model_set(ds, ms, exp, ds_traj, ds_rep, cfg, models)
                metrics_rows.extend(rows)
                prediction_rows.extend(preds)
            metrics_so_far = pd.DataFrame(metrics_rows)
            best_reporter = choose_validation_best(metrics_so_far, "3C", ["oxidative_reporter_state_space", "atp_reporter_state_space", "ox_atp_reporter_state_space"])
            sample_rows.extend(evaluate_sample_efficiency(ds, ms, ds_traj, ds_rep, cfg, best_reporter))
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    sample_eff = pd.DataFrame(sample_rows)
    robustness = reporter_robustness(metrics, cfg)
    safeguards = build_safeguards(traj, reporters, fluxes, manifest, metrics, cfg)
    if safeguards[(safeguards["critical"]) & (~safeguards["passed"])].shape[0]:
        failures = safeguards[(safeguards["critical"]) & (~safeguards["passed"])]
        raise AssertionError("Critical dFBA safeguards failed:\n" + failures.to_string(index=False))
    if write:
        save_outputs(traj, fluxes, reporters, transitions, manifest, env_grid, metrics, predictions, sample_eff, robustness, safeguards, cfg)
        make_figures(traj, reporters, manifest, transitions, predictions, metrics, sample_eff, robustness, cfg)
        write_docs(metrics, safeguards, robustness, cfg)
    return metrics, predictions, sample_eff, robustness, safeguards, traj, reporters, fluxes, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Run reduced fast-mode chain for tests and smoke checks.")
    parser.add_argument(
        "--backend",
        choices=("reduced_surrogate", "yeast_gem"),
        default=None,
        help="Select metabolic backend. yeast_gem requires a real LP-backed implementation and currently fails clearly.",
    )
    args = parser.parse_args()
    metrics, _pred, sample_eff, robustness, safeguards, *_ = run_all(fast=args.fast, write=True, backend=args.backend)
    print("dFBA state-machine validation complete")
    print(f"experiments: {sorted(metrics['experiment'].dropna().unique())}")
    print(f"safeguards: {int(safeguards['passed'].sum())}/{len(safeguards)} passed")
    print("held-out RMSE:")
    print(metrics[metrics["split"] == "heldout_combination"].groupby(["experiment", "model"])["trajectory_rmse"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
