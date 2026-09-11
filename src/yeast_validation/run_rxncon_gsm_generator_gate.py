#!/usr/bin/env python3
"""Gate the rxncon -> GSM hidden-generator interface.

This is the first gated implementation of the high-dimensional rxncon hidden
organism.  It preserves the published rxncon Boolean model as the regulatory
ground truth, builds a frozen deterministic regulatory-module -> GSM interface,
generates mixed delayed reporters, audits lockbox leakage, and optionally runs a
tiny Yeast9 dense-vs-event-triggered smoke comparison.

It does not train ML, run DBTL, or generate the canonical 125-culture dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import run_rxncon_external_complexity_screen as rxncon_screen  # noqa: E402
import run_rxncon_observable_complexity_audit as rxncon_obs  # noqa: E402


OUT_DIR = ROOT / "data/rxncon_gsm_generator_gate"
FIG_DIR = ROOT / "figures"
RNG_SEED = 20260823
N_TIMEPOINTS = 49
DT = 0.25
EPS = 1e-12

RXNCON_INPUTS = ["[Nutrients]", "[Pheromone]", "[HU]", "[LatA]", "[Nocodazole]"]
CONT_ENV_AXES = {
    "temperature": {"allowed_values": [27.0, 30.0, 33.0], "unit": "degC"},
    "pH": {"allowed_values": [4.5, 5.0, 5.5], "unit": "pH"},
    "DO": {"allowed_values": [20.0, 40.0, 80.0], "unit": "percent_air_saturation"},
    "glucose_uptake": {"allowed_values": [6.0, 10.0, 14.0], "unit": "mmol/gDW/h bound scale"},
}


@dataclass(frozen=True)
class InterfaceChannel:
    name: str
    target_gsm_quantity: str
    mapping_type: str
    justification: str
    directness: str
    direction: str
    alpha: float
    fn: Callable[[dict[str, float], dict[str, float]], float]


@dataclass(frozen=True)
class ReporterSpec:
    name: str
    contributors: tuple[str, ...]
    weights: tuple[float, ...]
    lag_steps: int
    filter_alpha: float
    gain: float
    baseline: float
    noise_sd: float
    interpretation: str


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def stable_hash(obj: object, n: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def load_rxncon_model() -> rxncon_screen.BoolNet:
    prefix = rxncon_screen.MODELS_DIR / "CDC_S_cerevisiae_perturbable"
    return rxncon_screen.load_boolnet(prefix)


def build_environment_contract() -> pd.DataFrame:
    rows = []
    for name in RXNCON_INPUTS:
        rows.append(
            {
                "environment_variable": name.strip("[]"),
                "learner_input": True,
                "allowed_values_or_range": "0/1",
                "regulatory_effect": "clamped published rxncon external input",
                "direct_metabolic_effect": {
                    "[Nutrients]": "scales glucose/carbon exchange capacity",
                    "[Pheromone]": "no direct exchange effect; affects metabolism through mating/arrest modules",
                    "[HU]": "adds replication-stress maintenance demand through interface",
                    "[LatA]": "adds morphogenesis/cytoskeletal stress maintenance through interface",
                    "[Nocodazole]": "adds spindle-checkpoint maintenance demand through interface",
                }[name],
                "observable_to_learner": True,
                "hidden_oracle": False,
            }
        )
    for name, spec in CONT_ENV_AXES.items():
        rows.append(
            {
                "environment_variable": name,
                "learner_input": True,
                "allowed_values_or_range": json.dumps(spec["allowed_values"]),
                "unit": spec["unit"],
                "regulatory_effect": "none in v1 rxncon; direct culture condition",
                "direct_metabolic_effect": {
                    "temperature": "affects ATP maintenance/environment factor in Yeast9 constraints",
                    "pH": "affects ATP maintenance/environment factor in Yeast9 constraints",
                    "DO": "sets oxygen exchange capacity",
                    "glucose_uptake": "sets maximum glucose exchange magnitude",
                }[name],
                "observable_to_learner": True,
                "hidden_oracle": False,
            }
        )
    return pd.DataFrame(rows)


def build_environment_panel(n: int = 60) -> pd.DataFrame:
    rows = []
    cont = []
    for t in CONT_ENV_AXES["temperature"]["allowed_values"]:
        for ph in CONT_ENV_AXES["pH"]["allowed_values"]:
            for do in CONT_ENV_AXES["DO"]["allowed_values"]:
                for glc in CONT_ENV_AXES["glucose_uptake"]["allowed_values"]:
                    cont.append((t, ph, do, glc))
    # All 32 constant rxncon input combinations, paired with spread continuous
    # conditions, then biologically meaningful single-stress pulses.
    for mask in range(32):
        env = {name: int((mask >> i) & 1) for i, name in enumerate(RXNCON_INPUTS)}
        env["[Nutrients]"] = env.get("[Nutrients]", 1)
        t, ph, do, glc = cont[(mask * 7) % len(cont)]
        rows.append(
            {
                "culture_id": f"rxncon_gsm_env_{len(rows):03d}",
                "environment_program": "constant",
                **{k.strip("[]"): v for k, v in env.items()},
                "temperature": t,
                "pH": ph,
                "DO": do,
                "glucose_uptake": glc,
            }
        )
    pulse_inputs = ["[Pheromone]", "[HU]", "[LatA]", "[Nocodazole]", "[Nutrients]"]
    for pulse in pulse_inputs:
        for program in ["late_on", "late_off", "early_pulse", "late_pulse"]:
            t, ph, do, glc = cont[((len(rows) + 3) * 11) % len(cont)]
            env = {name.strip("[]"): 0 for name in RXNCON_INPUTS}
            env["Nutrients"] = 1
            rows.append(
                {
                    "culture_id": f"rxncon_gsm_env_{len(rows):03d}",
                    "environment_program": f"{program}:{pulse.strip('[]')}",
                    **env,
                    "temperature": t,
                    "pH": ph,
                    "DO": do,
                    "glucose_uptake": glc,
                }
            )
    rng = np.random.default_rng(RNG_SEED)
    while len(rows) < n:
        t, ph, do, glc = cont[int(rng.integers(0, len(cont)))]
        env = {name.strip("[]"): int(rng.random() < p) for name, p in zip(RXNCON_INPUTS, [0.85, 0.25, 0.25, 0.25, 0.25])}
        rows.append(
            {
                "culture_id": f"rxncon_gsm_env_{len(rows):03d}",
                "environment_program": "stratified_sample",
                **env,
                "temperature": t,
                "pH": ph,
                "DO": do,
                "glucose_uptake": glc,
            }
        )
    return pd.DataFrame(rows[:n])


def force_array_for_panel(model: rxncon_screen.BoolNet, panel: pd.DataFrame) -> np.ndarray:
    name_to_sym = {v: k for k, v in model.names.items()}
    force = np.full((N_TIMEPOINTS, len(panel), len(model.symbol_order)), -1, dtype=np.int8)
    for i, row in panel.iterrows():
        program = str(row["environment_program"])
        for inp in RXNCON_INPUTS:
            col = inp.strip("[]")
            val = int(row[col])
            idx = model.index[name_to_sym[inp]]
            if program.startswith("late_on") and col in program:
                force[: N_TIMEPOINTS // 3, i, idx] = 0
                force[N_TIMEPOINTS // 3 :, i, idx] = 1
            elif program.startswith("late_off") and col in program:
                force[: N_TIMEPOINTS // 3, i, idx] = 1
                force[N_TIMEPOINTS // 3 :, i, idx] = 0
            elif program.startswith("early_pulse") and col in program:
                force[:, i, idx] = 0
                force[N_TIMEPOINTS // 6 : N_TIMEPOINTS // 3, i, idx] = 1
            elif program.startswith("late_pulse") and col in program:
                force[:, i, idx] = 0
                force[N_TIMEPOINTS // 2 : 2 * N_TIMEPOINTS // 3, i, idx] = 1
            else:
                force[:, i, idx] = val
    return force


def module_activity(model: rxncon_screen.BoolNet, traj: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mods = rxncon_obs.module_symbols(model)
    arrays = []
    names = []
    for module, syms in mods.items():
        idx = [model.index[s] for s in syms]
        if not idx:
            continue
        arrays.append(traj[:, :, idx].mean(axis=2))
        names.append(module)
    return np.stack(arrays, axis=2), names


def env_time_arrays(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(panel)
    env = {}
    for col in ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole", "temperature", "pH", "DO", "glucose_uptake"]:
        schedule_cols = [f"{col}_t{t:02d}" for t in range(N_TIMEPOINTS)]
        if all(c in panel.columns for c in schedule_cols):
            env[col] = panel[schedule_cols].to_numpy(float)
        else:
            env[col] = np.repeat(panel[col].to_numpy(float)[:, None], N_TIMEPOINTS, axis=1)
    return env


def interface_channels() -> list[InterfaceChannel]:
    def m(mods: dict[str, float], name: str) -> float:
        return float(mods.get(name, 0.0))

    def e(env: dict[str, float], name: str) -> float:
        return float(env.get(name, 0.0))

    return [
        InterfaceChannel(
            "carbon_uptake_capacity",
            f"{gem.GLUCOSE_EXCHANGE} lower bound",
            "direct environment plus checkpoint/growth-arrest module attenuation",
            "Nutrients and glucose directly set carbon availability; cell-cycle arrest/checkpoints reduce uptake/resource demand.",
            "direct_environment_and_higher_level_physiology",
            "higher channel increases glucose uptake magnitude",
            0.55,
            lambda mods, env: clip01(0.12 + 0.78 * e(env, "Nutrients") * (e(env, "glucose_uptake") / 14.0) * (1 - 0.18 * m(mods, "DNA_damage_checkpoint") - 0.14 * m(mods, "Mating_polarity"))),
        ),
        InterfaceChannel(
            "oxygen_capacity",
            f"{gem.OXYGEN_EXCHANGE} lower bound",
            "direct DO plus stress attenuation",
            "DO directly limits oxygen exchange; checkpoint/stress states reduce effective respiratory capacity.",
            "direct_environment_and_higher_level_physiology",
            "higher channel increases oxygen uptake magnitude",
            0.60,
            lambda mods, env: clip01((e(env, "DO") / 80.0) * (1 - 0.12 * m(mods, "DNA_damage_checkpoint") - 0.10 * m(mods, "Spindle_checkpoint"))),
        ),
        InterfaceChannel(
            "atp_maintenance_multiplier",
            f"{gem.ATPM_RXN} lower bound",
            "many-to-one checkpoint/stress maintenance load",
            "Replication, spindle, cytoskeletal and mating stress consume maintenance resources.",
            "higher_level_physiology",
            "higher channel increases ATP maintenance demand",
            0.45,
            lambda mods, env: float(np.clip(0.55 + 0.25 * e(env, "HU") + 0.22 * e(env, "LatA") + 0.22 * e(env, "Nocodazole") + 0.42 * m(mods, "DNA_damage_checkpoint") + 0.32 * m(mods, "Spindle_checkpoint") + 0.22 * m(mods, "Morphogenesis_spindle_actin"), 0.35, 1.8)),
        ),
        InterfaceChannel(
            "growth_allocation_gamma",
            f"{gem.BIOMASS_RXN} preserved-growth fraction",
            "cell-cycle progression versus checkpoint/arrest allocation",
            "Active growth/cell-cycle modules support biomass allocation; checkpoint/mating arrest diverts resources.",
            "higher_level_physiology",
            "higher channel preserves more maximum growth before product optimization",
            0.50,
            lambda mods, env: float(np.clip(0.42 + 0.14 * m(mods, "G1_Start_CDK") + 0.12 * m(mods, "S_phase_replication") + 0.10 * m(mods, "G2_M_cyclins") - 0.22 * m(mods, "DNA_damage_checkpoint") - 0.18 * m(mods, "Mating_polarity"), 0.25, 0.86)),
        ),
        InterfaceChannel(
            "stress_maintenance_load",
            "beta-carotene degradation and maintenance burden",
            "overlapping environmental and checkpoint stress",
            "Stress states increase product loss/maintenance without exposing a single hidden state.",
            "higher_level_physiology",
            "higher channel increases product degradation and maintenance penalty",
            0.40,
            lambda mods, env: float(np.clip(0.10 + 0.18 * e(env, "HU") + 0.12 * e(env, "LatA") + 0.12 * e(env, "Nocodazole") + 0.38 * m(mods, "DNA_damage_checkpoint") + 0.28 * m(mods, "Spindle_checkpoint") + 0.18 * m(mods, "Mating_polarity"), 0.0, 1.5)),
        ),
        InterfaceChannel(
            "precursor_availability",
            f"{gem.NATIVE_GGPP_RXN} effective upper bound",
            "resource/cell-cycle precursor allocation",
            "Biosynthetic precursor availability follows growth/resource programs and falls under checkpoint stress.",
            "higher_level_physiology",
            "higher channel increases GGPP precursor availability",
            0.35,
            lambda mods, env: clip01(0.35 + 0.26 * m(mods, "G1_Start_CDK") + 0.18 * m(mods, "S_phase_replication") + 0.16 * m(mods, "G2_M_cyclins") - 0.18 * m(mods, "DNA_damage_checkpoint") - 0.12 * m(mods, "Spindle_checkpoint")),
        ),
        InterfaceChannel(
            "resource_translation_capacity",
            "global protein/resource allocation",
            "cell-cycle resource activity with stress diversion",
            "Growth and biosynthesis programs support enzyme availability; stress diverts proteome capacity.",
            "higher_level_physiology",
            "higher channel increases heterologous enzyme capacity",
            0.35,
            lambda mods, env: clip01(0.30 + 0.22 * m(mods, "G1_Start_CDK") + 0.18 * m(mods, "S_phase_replication") + 0.14 * m(mods, "G2_M_cyclins") + 0.10 * m(mods, "APC_exit_separase") - 0.20 * m(mods, "DNA_damage_checkpoint") - 0.15 * m(mods, "Mating_polarity")),
        ),
        InterfaceChannel(
            "PSY_capacity",
            f"{gem.PSY_RXN} upper bound",
            "resource-sensitive heterologous pathway capacity",
            "PSY is a heterologous enzyme; capacity follows resource allocation and stress burden, not direct rxncon GPR.",
            "heterologous_higher_level_physiology",
            "higher channel increases PSY upper bound",
            0.30,
            lambda mods, env: clip01(0.20 + 0.70 * (0.70 * m(mods, "G1_Start_CDK") + 0.30 * m(mods, "G2_M_cyclins")) * (1 - 0.22 * m(mods, "DNA_damage_checkpoint"))),
        ),
        InterfaceChannel(
            "DES_capacity",
            f"{gem.DES_RXN} upper bound",
            "stress-sensitive heterologous pathway capacity",
            "Downstream carotenoid desaturation is treated as more stress/resource sensitive than PSY.",
            "heterologous_higher_level_physiology",
            "higher channel increases DES upper bound",
            0.28,
            lambda mods, env: clip01(0.18 + 0.66 * (0.45 * m(mods, "S_phase_replication") + 0.35 * m(mods, "G2_M_cyclins") + 0.20 * m(mods, "APC_exit_separase")) * (1 - 0.30 * m(mods, "DNA_damage_checkpoint") - 0.16 * m(mods, "Spindle_checkpoint"))),
        ),
        InterfaceChannel(
            "CYC_capacity",
            f"{gem.CYC_RXN} upper bound",
            "stress-sensitive terminal pathway capacity",
            "Terminal cyclase capacity follows resource availability and is penalized by checkpoint/morphogenesis stress.",
            "heterologous_higher_level_physiology",
            "higher channel increases CYC upper bound",
            0.25,
            lambda mods, env: clip01(0.16 + 0.66 * (0.40 * m(mods, "G2_M_cyclins") + 0.35 * m(mods, "APC_exit_separase") + 0.25 * m(mods, "Morphogenesis_spindle_actin")) * (1 - 0.25 * m(mods, "Spindle_checkpoint") - 0.20 * m(mods, "DNA_damage_checkpoint"))),
        ),
    ]


def compute_interface(module_arr: np.ndarray, module_names: list[str], panel: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    env_arrays = env_time_arrays(panel)
    channels = interface_channels()
    raw = np.zeros((module_arr.shape[0], module_arr.shape[1], len(channels)))
    smooth = np.zeros_like(raw)
    for i in range(module_arr.shape[0]):
        for t in range(module_arr.shape[1]):
            mods = {name: float(module_arr[i, t, k]) for k, name in enumerate(module_names)}
            env = {name: float(arr[i, t]) for name, arr in env_arrays.items()}
            for j, ch in enumerate(channels):
                raw[i, t, j] = ch.fn(mods, env)
                if t == 0:
                    smooth[i, t, j] = raw[i, t, j]
                else:
                    smooth[i, t, j] = smooth[i, t - 1, j] + ch.alpha * (raw[i, t, j] - smooth[i, t - 1, j])
    return smooth, [ch.name for ch in channels]


def reporter_specs() -> list[ReporterSpec]:
    return [
        ReporterSpec("R_stress", ("DNA_damage_checkpoint", "Spindle_checkpoint", "Mating_polarity", "stress_maintenance_load"), (1.1, 0.8, 0.5, 0.9), 2, 0.35, 1.0, 0.05, 0.035, "oxidative/checkpoint-like stress reporter"),
        ReporterSpec("R_resource", ("G1_Start_CDK", "S_phase_replication", "resource_translation_capacity", "growth_allocation_gamma"), (0.7, 0.6, 1.0, 0.8), 1, 0.40, 1.0, 0.03, 0.030, "resource and growth-program reporter"),
        ReporterSpec("R_checkpoint", ("DNA_damage_checkpoint", "Spindle_checkpoint", "HU", "Nocodazole"), (1.0, 0.9, 0.4, 0.4), 3, 0.30, 1.0, 0.02, 0.030, "replication/spindle checkpoint reporter"),
        ReporterSpec("R_pathway_capacity", ("precursor_availability", "PSY_capacity", "DES_capacity", "CYC_capacity"), (0.8, 0.7, 0.7, 0.7), 2, 0.32, 1.0, 0.04, 0.035, "heterologous pathway resource-capacity reporter"),
        ReporterSpec("R_morphology", ("Morphogenesis_spindle_actin", "Mating_polarity", "LatA", "Pheromone"), (1.0, 0.5, 0.4, 0.4), 2, 0.34, 1.0, 0.05, 0.030, "morphology/polarity reporter"),
        ReporterSpec("R_energy", ("atp_maintenance_multiplier", "oxygen_capacity", "carbon_uptake_capacity", "stress_maintenance_load"), (0.9, -0.45, -0.35, 0.7), 1, 0.38, 1.0, 0.04, 0.035, "ATP/resource-demand reporter"),
    ]


def build_reporters(
    module_arr: np.ndarray,
    module_names: list[str],
    interface_arr: np.ndarray,
    interface_names: list[str],
    panel: pd.DataFrame,
    noise: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(RNG_SEED + (1 if noise else 0))
    module_lookup = {name: module_arr[:, :, i] for i, name in enumerate(module_names)}
    iface_lookup = {name: interface_arr[:, :, i] for i, name in enumerate(interface_names)}
    env_lookup = env_time_arrays(panel)
    signals = {**module_lookup, **iface_lookup, **env_lookup}
    specs = reporter_specs()
    clean = np.zeros((module_arr.shape[0], module_arr.shape[1], len(specs)))
    noisy = np.zeros_like(clean)
    for k, spec in enumerate(specs):
        x = np.zeros((module_arr.shape[0], module_arr.shape[1]))
        for contrib, weight in zip(spec.contributors, spec.weights):
            val = signals[contrib]
            if spec.lag_steps:
                lagged = np.concatenate([np.repeat(val[:, :1], spec.lag_steps, axis=1), val[:, :-spec.lag_steps]], axis=1)
            else:
                lagged = val
            x += weight * lagged
        y = sigmoid(spec.gain * (x - np.nanmedian(x))) + spec.baseline
        filt = np.zeros_like(y)
        filt[:, 0] = y[:, 0]
        for t in range(1, y.shape[1]):
            filt[:, t] = filt[:, t - 1] + spec.filter_alpha * (y[:, t] - filt[:, t - 1])
        clean[:, :, k] = filt
        gain = rng.normal(1.0, 0.035, size=(filt.shape[0], 1)) if noise else 1.0
        baseline = rng.normal(0.0, 0.015, size=(filt.shape[0], 1)) if noise else 0.0
        eps = rng.normal(0.0, spec.noise_sd, size=filt.shape) if noise else 0.0
        noisy[:, :, k] = np.clip(gain * filt + baseline + eps, 0.0, 1.5)
    return clean, [s.name for s in specs], noisy


def write_interface_inventory(model, source_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    gpr_rows = []
    if source_path and source_path.exists():
        labels = parse_gpr_labels(source_path)
        cobra = gem.require_cobra()
        yeast = gem.load_model(cobra, source_path)
        rxncon_components = set()
        for name in model.names.values():
            for token in re.findall(r"[A-Z][A-Za-z0-9]+", name):
                rxncon_components.add(token.upper())
        for rxn in yeast.reactions:
            for gene in rxn.genes:
                label = labels.get(gene.id, "")
                if label and label.upper() in rxncon_components:
                    gpr_rows.append(
                        {
                            "rxncon_component": label,
                            "yeast9_gene_id": gene.id,
                            "yeast9_reaction": rxn.id,
                            "yeast9_reaction_name": rxn.name,
                            "gene_reaction_rule": rxn.gene_reaction_rule,
                            "mapping_status": "direct_GPR_name_label_match",
                        }
                    )
    interface_rows = []
    for ch in interface_channels():
        interface_rows.append(
            {
                "gsm_control_channel": ch.name,
                "target_GSM_quantity": ch.target_gsm_quantity,
                "mapping_type": ch.mapping_type,
                "literature_or_model_justification": ch.justification,
                "direct_GPR_or_higher_level": ch.directness,
                "expected_direction": ch.direction,
                "smoothing_alpha_per_step": ch.alpha,
                "equation_summary": "bounded deterministic weighted function of rxncon module summaries and/or environment, then first-order smoothing",
            }
        )
    return pd.DataFrame(interface_rows), pd.DataFrame(gpr_rows)


def parse_gpr_labels(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    out = {}
    for gid, label in re.findall(r'fbc:id="([^"]+)"\s+fbc:label="([^"]+)"', text):
        out[gid] = label
    return out


def flatten_by_culture(arr: np.ndarray) -> np.ndarray:
    return arr.transpose(0, 2, 1).reshape(arr.shape[0], -1)


def complexity_rows(layers: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for layer, x in layers.items():
        row, _spec = rxncon_screen.summarize_layer(layer, x, compute_nonlinear=x.shape[1] <= 2500)
        row["nominal_dimension"] = x.shape[1]
        rows.append(row)
    return pd.DataFrame(rows)


def intervention_response(panel: pd.DataFrame, y: np.ndarray, names: list[str], layer: str) -> dict[str, float]:
    u_cols = ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole", "temperature", "pH", "DO", "glucose_uptake"]
    u = panel[u_cols].to_numpy(float)
    ymat, kept = rxncon_screen.remove_constant_columns(y, names)
    u, _ = rxncon_screen.remove_constant_columns(u, u_cols)
    if ymat.shape[1] == 0 or u.shape[1] == 0:
        return {"layer": layer, "entropy_rank": np.nan}
    uz = rxncon_screen.standardize_matrix(u)
    yz = rxncon_screen.standardize_matrix(ymat)
    j = np.linalg.solve(uz.T @ uz + 1e-6 * np.eye(uz.shape[1]), uz.T @ yz)
    s = np.linalg.svd(j, compute_uv=False)
    return {
        "layer": layer,
        "n_design_features": u.shape[1],
        "n_response_features": ymat.shape[1],
        **rxncon_screen.dimension_metrics(s**2),
        "pc1_variance_explained": float((s[0] ** 2) / np.sum(s**2)) if len(s) else np.nan,
    }


def reporter_audit(model, traj, module_arr, module_names, interface_arr, interface_names, reporters, reporter_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Culture-time table for correlations.
    module_flat = module_arr.reshape(-1, module_arr.shape[2])
    iface_flat = interface_arr.reshape(-1, interface_arr.shape[2])
    rep_flat = reporters.reshape(-1, reporters.shape[2])
    rows = []
    for r_i, r_name in enumerate(reporter_names):
        for m_i, m_name in enumerate(module_names):
            c = corr(rep_flat[:, r_i], module_flat[:, m_i])
            rows.append({"reporter": r_name, "target_type": "module", "target": m_name, "pearson": c})
        for c_i, c_name in enumerate(interface_names):
            rows.append({"reporter": r_name, "target_type": "interface", "target": c_name, "pearson": corr(rep_flat[:, r_i], iface_flat[:, c_i])})
    # Single reporter vs raw rxncon states on a deterministic subsample.
    state_syms = model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()
    idx_states = [model.index[s] for s in state_syms]
    raw = traj[:, :, idx_states].reshape(-1, len(idx_states)).astype(float)
    rng = np.random.default_rng(RNG_SEED)
    sample = rng.choice(np.arange(raw.shape[0]), size=min(5000, raw.shape[0]), replace=False)
    max_direct = 0.0
    max_pair = ("", "")
    for r_i, r_name in enumerate(reporter_names):
        rv = rep_flat[sample, r_i]
        for j, sym in enumerate(state_syms):
            if np.nanstd(raw[sample, j]) < 1e-10:
                continue
            c = abs(corr(rv, raw[sample, j]))
            if c > max_direct:
                max_direct = c
                max_pair = (r_name, model.names[sym])
    cross = np.corrcoef(rep_flat, rowvar=False)
    tri = np.triu_indices_from(cross, k=1)
    max_cross = float(np.nanmax(np.abs(cross[tri]))) if len(tri[0]) else np.nan
    # Approximate non-invertibility: regress reporters to top raw-state PCs.
    raw_sample = raw[sample]
    raw_sample, _ = rxncon_screen.remove_constant_columns(raw_sample)
    rz = rxncon_screen.standardize_matrix(rep_flat[sample])
    hz = rxncon_screen.standardize_matrix(raw_sample)
    _u, s, vt = np.linalg.svd(hz, full_matrices=False)
    pcs = hz @ vt[: min(12, vt.shape[0])].T
    coef = np.linalg.solve(rz.T @ rz + 1e-6 * np.eye(rz.shape[1]), rz.T @ pcs)
    pred = rz @ coef
    r2 = 1.0 - np.sum((pcs - pred) ** 2, axis=0) / np.maximum(np.sum((pcs - pcs.mean(axis=0)) ** 2, axis=0), EPS)
    safeguard = pd.DataFrame(
        [
            {
                "safeguard": "no_single_reporter_direct_hidden_state_exposure",
                "value": max_direct,
                "threshold": 0.95,
                "passed": bool(max_direct < 0.95),
                "max_pair": json.dumps({"reporter": max_pair[0], "hidden_state": max_pair[1]}),
            },
            {
                "safeguard": "reporters_not_duplicate_channels",
                "value": max_cross,
                "threshold": 0.98,
                "passed": bool(max_cross < 0.98),
            },
            {
                "safeguard": "reporters_informative_but_noninvertible",
                "value": float(np.nanmean(r2)),
                "threshold": "0.02 < mean_top_hidden_pc_R2 < 0.80",
                "passed": bool(float(np.nanmean(r2)) > 0.02 and float(np.nanmean(r2)) < 0.80),
            },
        ]
    )
    return pd.DataFrame(rows), safeguard


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-10 or np.nanstd(b) < 1e-10:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def lockbox_spec() -> pd.DataFrame:
    rows = []
    lock = [
        "raw_rxncon_node",
        "regulatory_module_truth",
        "regulatory_to_GSM_interface_state",
        "reaction_capacity_multiplier_truth",
        "true_GSM_bound",
        "exact_flux",
        "event_trigger_decision",
        "oracle_future_phenotype",
        "generator_regime_label",
    ]
    for item in lock:
        rows.append({"variable_class": item, "learner_facing_allowed": False, "table_role": "simulator_lockbox"})
    obs = ["environment", "time", "biomass_X", "product_B_total", "predefined_biosensor", "culture_metadata"]
    for item in obs:
        rows.append({"variable_class": item, "learner_facing_allowed": True, "table_role": "observable_dataset"})
    return pd.DataFrame(rows)


def apply_rxncon_constraints(model, cfg: gem.GEMCultureConfig, controls: dict[str, float], env_row: pd.Series) -> tuple[dict[str, float], float, float]:
    glucose_lb = -float(env_row["glucose_uptake"]) * max(0.02, controls["carbon_uptake_capacity"])
    oxygen_lb = -cfg.oxygen_uptake_at_100_do * float(env_row["DO"]) / 100.0 * max(0.05, controls["oxygen_capacity"])
    temp_cfg = gem.GEMCultureConfig(temperature=float(env_row["temperature"]), pH=float(env_row["pH"]), DO=float(env_row["DO"]))
    atpm = cfg.baseline_atpm * gem.env_factor(float(env_row["temperature"]), float(env_row["pH"]), temp_cfg) * max(0.2, controls["atp_maintenance_multiplier"] + 0.30 * controls["stress_maintenance_load"])
    model.reactions.get_by_id(gem.GLUCOSE_EXCHANGE).lower_bound = glucose_lb
    model.reactions.get_by_id(gem.OXYGEN_EXCHANGE).lower_bound = oxygen_lb
    atpm_rxn = model.reactions.get_by_id(gem.ATPM_RXN)
    atpm_rxn.lower_bound = atpm
    atpm_rxn.upper_bound = max(atpm_rxn.upper_bound, atpm)
    model.reactions.get_by_id(gem.PSY_RXN).upper_bound = 0.34 * max(0.04, controls["PSY_capacity"]) * max(0.1, controls["resource_translation_capacity"])
    model.reactions.get_by_id(gem.DES_RXN).upper_bound = 0.28 * max(0.04, controls["DES_capacity"]) * max(0.1, controls["resource_translation_capacity"])
    model.reactions.get_by_id(gem.CYC_RXN).upper_bound = 0.23 * max(0.04, controls["CYC_capacity"]) * max(0.1, controls["resource_translation_capacity"])
    if gem.NATIVE_GGPP_RXN in {r.id for r in model.reactions}:
        rxn = model.reactions.get_by_id(gem.NATIVE_GGPP_RXN)
        rxn.upper_bound = min(rxn.upper_bound, 0.75 * max(0.05, controls["precursor_availability"]))
    model.reactions.get_by_id(gem.PRODUCT_RXN).upper_bound = 1000.0
    gamma = float(np.clip(controls["growth_allocation_gamma"], 0.25, 0.86))
    beta_deg = cfg.beta_deg_base * (1.0 + 1.4 * controls["stress_maintenance_load"])
    cons = {
        "glucose_lower_bound": glucose_lb,
        "oxygen_lower_bound": oxygen_lb,
        "atp_maintenance_lower_bound": atpm,
        "PSY_effective_upper_bound": model.reactions.get_by_id(gem.PSY_RXN).upper_bound,
        "DES_effective_upper_bound": model.reactions.get_by_id(gem.DES_RXN).upper_bound,
        "CYC_effective_upper_bound": model.reactions.get_by_id(gem.CYC_RXN).upper_bound,
        "gamma_growth_fraction": gamma,
        "beta_degradation_rate": beta_deg,
    }
    return cons, gamma, beta_deg


def run_gsm_rollout(
    augmented,
    source_path: Path,
    panel: pd.DataFrame,
    interface_arr: np.ndarray,
    interface_names: list[str],
    culture_indices: list[int],
    n_time: int,
    sparse: bool,
    delta: float = 0.05,
    max_gap: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg0 = gem.GEMCultureConfig(n_time=n_time, dt=DT, capacity_mode="no_dynamic_capacity")
    totals = {k: 0 for k in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves"]}
    traj_rows, flux_rows, cons_rows, summary_rows = [], [], [], []
    checksum = gem.sha256(source_path)
    for idx in culture_indices:
        row = panel.iloc[idx]
        x = np.zeros(n_time)
        b = np.zeros(n_time)
        x[0] = cfg0.x0
        last_controls = None
        last_flux = None
        last_cons = None
        last_solve_i = -999
        culture_totals = {k: 0 for k in totals}
        for i in range(n_time - 1):
            controls = {name: float(interface_arr[idx, i, j]) for j, name in enumerate(interface_names)}
            if (not sparse) or last_controls is None or i == n_time - 2 or (i - last_solve_i) >= max_gap:
                solve_now = True
                trigger = "dense_or_required"
            else:
                dist = max(abs(controls[k] - last_controls[k]) / (abs(last_controls[k]) + 1e-6) for k in controls)
                solve_now = dist > delta
                trigger = f"interface_distance_{dist:.4f}" if solve_now else "carried_forward"
            if solve_now:
                interval_model = augmented.copy()
                do_value = float(row.get(f"DO_t{i:02d}", row["DO"]))
                cfg = gem.GEMCultureConfig(
                    temperature=float(row["temperature"]),
                    pH=float(row["pH"]),
                    DO=do_value,
                    glucose_uptake=float(row["glucose_uptake"]),
                    n_time=n_time,
                    dt=DT,
                    capacity_mode="no_dynamic_capacity",
                )
                cons, gamma, beta_deg = apply_rxncon_constraints(interval_model, cfg, controls, row)
                flux = gem.solve_staged(interval_model, cfg, gamma, state="balanced")
                for k in culture_totals:
                    culture_totals[k] += int(flux[k])
                    totals[k] += int(flux[k])
                last_controls = controls
                last_flux = flux
                last_cons = cons
                last_solve_i = i
            else:
                flux = dict(last_flux)
                cons = dict(last_cons)
                beta_deg = cons["beta_degradation_rate"]
            dx = max(0.0, float(flux["biomass_flux"])) * x[i]
            db = max(0.0, float(flux["beta_carotene_flux"])) * x[i] - beta_deg * b[i]
            x[i + 1] = max(1e-9, x[i] + DT * dx)
            b[i + 1] = max(0.0, b[i] + DT * db)
            common = {
                "culture_id": row["culture_id"],
                "interval_index": i,
                "time": i * DT,
                "sparse_event_triggered": sparse,
                "solve_anchor": bool(solve_now),
                "solve_trigger": trigger,
                "metabolic_backend": "yeast_gem_lp",
                "gem_source_checksum": checksum,
                **{k: (do_value if k == "DO" else row[k]) for k in ["temperature", "pH", "DO", "glucose_uptake", "Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]},
            }
            flux_rows.append({**common, **flux})
            cons_rows.append({**common, **cons, **controls})
        for i in range(n_time):
            traj_rows.append(
                {
                    "culture_id": row["culture_id"],
                    "time_index": i,
                    "time": i * DT,
                    "sparse_event_triggered": sparse,
                    "X": x[i],
                    "B_total": b[i],
                    **{k: row[k] for k in ["temperature", "pH", "DO", "glucose_uptake", "Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]},
                }
            )
        summary_rows.append({"culture_id": row["culture_id"], "sparse_event_triggered": sparse, **culture_totals, "n_surrogate_evaluations": 0})
    return pd.DataFrame(traj_rows), pd.DataFrame(flux_rows), pd.DataFrame(cons_rows), pd.DataFrame(summary_rows)


def dense_sparse_smoke(panel, interface_arr, interface_names) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    indices = [0, 7, 19]
    dense = run_gsm_rollout(augmented, source_path, panel, interface_arr, interface_names, indices, n_time=13, sparse=False)
    sparse = run_gsm_rollout(augmented, source_path, panel, interface_arr, interface_names, indices, n_time=13, sparse=True, delta=0.04, max_gap=4)
    dense_traj, dense_flux, dense_cons, dense_sum = dense
    sparse_traj, sparse_flux, sparse_cons, sparse_sum = sparse
    joined = dense_traj.merge(sparse_traj, on=["culture_id", "time_index"], suffixes=("_dense", "_sparse"))
    metrics = []
    for cid, g in joined.groupby("culture_id"):
        metrics.append(
            {
                "culture_id": cid,
                "biomass_rmse": rmse(g["X_dense"], g["X_sparse"]),
                "product_rmse": rmse(g["B_total_dense"], g["B_total_sparse"]),
                "final_product_abs_diff": float(abs(g.sort_values("time_index")["B_total_dense"].iloc[-1] - g.sort_values("time_index")["B_total_sparse"].iloc[-1])),
            }
        )
    summary = pd.DataFrame(metrics)
    dense_lp = int(dense_sum["n_actual_lp_solves"].sum())
    sparse_lp = int(sparse_sum["n_actual_lp_solves"].sum())
    summary["dense_lp_solves_total"] = dense_lp
    summary["sparse_lp_solves_total"] = sparse_lp
    summary["lp_reduction_fraction"] = 1.0 - sparse_lp / max(dense_lp, 1)
    return pd.concat([dense_traj, sparse_traj], ignore_index=True), pd.concat([dense_flux, sparse_flux], ignore_index=True), pd.concat([dense_cons, sparse_cons], ignore_index=True), summary


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def save_figures(panel, module_arr, module_names, interface_arr, interface_names, reporters, reporter_names, complexity) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    # Complexity ladder.
    plot = complexity.copy()
    plt.figure(figsize=(8, 4.8))
    plt.bar(np.arange(len(plot)), plot["entropy_rank"], color="#2563eb")
    plt.xticks(np.arange(len(plot)), plot["layer"], rotation=35, ha="right", fontsize=8)
    plt.ylabel("Entropy effective rank")
    plt.title("rxncon -> GSM generator gate: effective-rank ladder")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_gsm_generator_complexity_ladder.svg")
    plt.close()

    # Representative module/interface/reporter trajectories.
    cid = 0
    t = np.arange(module_arr.shape[1]) * DT
    plt.figure(figsize=(9, 5))
    for name in ["G1_Start_CDK", "DNA_damage_checkpoint", "Spindle_checkpoint", "Morphogenesis_spindle_actin"]:
        if name in module_names:
            plt.plot(t, module_arr[cid, :, module_names.index(name)], label=name)
    plt.xlabel("time")
    plt.ylabel("module activity")
    plt.title("Representative rxncon regulatory-module trajectories")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_gsm_generator_module_trajectories.svg")
    plt.close()

    plt.figure(figsize=(9, 5))
    for name in ["carbon_uptake_capacity", "atp_maintenance_multiplier", "growth_allocation_gamma", "PSY_capacity", "DES_capacity", "CYC_capacity"]:
        if name in interface_names:
            plt.plot(t, interface_arr[cid, :, interface_names.index(name)], label=name)
    plt.xlabel("time")
    plt.ylabel("interface value")
    plt.title("Representative time-varying GSM interface controls")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_gsm_generator_interface_trajectories.svg")
    plt.close()

    plt.figure(figsize=(9, 5))
    for name in reporter_names:
        plt.plot(t, reporters[cid, :, reporter_names.index(name)], label=name)
    plt.xlabel("time")
    plt.ylabel("reporter signal")
    plt.title("Representative mixed delayed biosensor trajectories")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_gsm_generator_reporter_trajectories.svg")
    plt.close()

    flat = reporters.reshape(-1, reporters.shape[2])
    corr = np.corrcoef(flat, rowvar=False)
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(np.arange(len(reporter_names)), reporter_names, rotation=45, ha="right", fontsize=7)
    plt.yticks(np.arange(len(reporter_names)), reporter_names, fontsize=7)
    plt.title("Reporter correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_gsm_generator_reporter_correlation.svg")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cultures", type=int, default=60)
    parser.add_argument("--with-gsm-smoke", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_rxncon_model()
    env_contract = build_environment_contract()
    env_contract.to_csv(OUT_DIR / "rxncon_gsm_environment_contract.csv", index=False)
    panel = build_environment_panel(args.n_cultures)
    force = force_array_for_panel(model, panel)
    traj, runtime = rxncon_screen.simulate_panel(model, force)
    panel.to_csv(OUT_DIR / "rxncon_gsm_environment_panel.csv", index=False)
    np.savez_compressed(OUT_DIR / "rxncon_gsm_hidden_rxncon_envonly_panel.npz", traj=traj)
    runtime.to_csv(OUT_DIR / "rxncon_gsm_rxncon_runtime.csv", index=False)

    module_arr, module_names = module_activity(model, traj)
    interface_arr, interface_names = compute_interface(module_arr, module_names, panel)
    reporters_clean, reporter_names, reporters_noisy = build_reporters(module_arr, module_names, interface_arr, interface_names, panel, noise=True)
    _clean2, _, reporters_noiseless = build_reporters(module_arr, module_names, interface_arr, interface_names, panel, noise=False)

    interface_inventory, gpr = write_interface_inventory(model, Path(pd.read_csv(gem.DATA / "yeast_gem_selected_asset.csv").iloc[0]["full_path"]))
    interface_inventory.to_csv(OUT_DIR / "rxncon_gsm_interface_inventory.csv", index=False)
    gpr.to_csv(OUT_DIR / "rxncon_gsm_direct_gpr_overlap_inventory.csv", index=False)
    reporter_def = pd.DataFrame([s.__dict__ for s in reporter_specs()])
    reporter_def["contributors"] = reporter_def["contributors"].apply(lambda x: ";".join(x))
    reporter_def["weights"] = reporter_def["weights"].apply(lambda x: ";".join(map(str, x)))
    reporter_def.to_csv(OUT_DIR / "rxncon_gsm_biosensor_definitions.csv", index=False)
    lockbox_spec().to_csv(OUT_DIR / "rxncon_gsm_lockbox_spec.csv", index=False)

    # Long observable dataset preview: environment + reporters only. No hidden
    # rxncon/modules/interface/flux variables are written here.
    obs_rows = []
    for i, row in panel.iterrows():
        for t in range(N_TIMEPOINTS):
            rec = {"culture_id": row["culture_id"], "time_index": t, "time": t * DT, **{k: row[k] for k in ["temperature", "pH", "DO", "glucose_uptake", "Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]}}
            for j, name in enumerate(reporter_names):
                rec[name] = reporters_noisy[i, t, j]
            obs_rows.append(rec)
    observable_preview = pd.DataFrame(obs_rows)
    observable_preview.to_csv(OUT_DIR / "rxncon_gsm_observable_dataset_preview_no_gsm.csv", index=False)

    # Lockbox hidden tables for generator audit only.
    pd.DataFrame(flatten_by_culture(module_arr), index=panel["culture_id"]).to_csv(OUT_DIR / "rxncon_gsm_hidden_module_trajectories_wide.csv")
    pd.DataFrame(flatten_by_culture(interface_arr), index=panel["culture_id"]).to_csv(OUT_DIR / "rxncon_gsm_hidden_interface_trajectories_wide.csv")

    layers = {
        "rxncon_state": flatten_by_culture(traj[:, :, [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]]),
        "regulatory_modules": flatten_by_culture(module_arr),
        "GSM_interface": flatten_by_culture(interface_arr),
        "biosensors_noiseless": flatten_by_culture(reporters_noiseless),
        "biosensors_noisy": flatten_by_culture(reporters_noisy),
    }
    complexity = complexity_rows(layers)
    complexity.to_csv(OUT_DIR / "rxncon_gsm_generator_complexity_ladder.csv", index=False)
    ratios = []
    get_rank = dict(zip(complexity["layer"], complexity["entropy_rank"]))
    for a, b in [("regulatory_modules", "rxncon_state"), ("GSM_interface", "regulatory_modules"), ("biosensors_noiseless", "rxncon_state"), ("biosensors_noisy", "rxncon_state")]:
        ratios.append({"ratio": f"D_{a}/D_{b}", "value": get_rank.get(a, np.nan) / max(get_rank.get(b, np.nan), EPS)})
    pd.DataFrame(ratios).to_csv(OUT_DIR / "rxncon_gsm_information_collapse_ratios.csv", index=False)

    # Intervention/environment response on interface and reporter summaries.
    temporal_reporters, reporter_feature_names = rxncon_obs.temporal_features(reporters_noisy, reporter_names)
    temporal_interface, interface_feature_names = rxncon_obs.temporal_features(interface_arr, interface_names)
    pd.DataFrame(
        [
            intervention_response(panel, temporal_interface, interface_feature_names, "environment_to_GSM_interface"),
            intervention_response(panel, temporal_reporters, reporter_feature_names, "environment_to_biosensors"),
        ]
    ).to_csv(OUT_DIR / "rxncon_gsm_intervention_response_complexity.csv", index=False)

    reporter_corr, reporter_safeguards = reporter_audit(model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noisy, reporter_names)
    reporter_corr.to_csv(OUT_DIR / "rxncon_gsm_reporter_information_correlations.csv", index=False)
    reporter_safeguards.to_csv(OUT_DIR / "rxncon_gsm_reporter_safeguards.csv", index=False)
    boolnet_ids = set(model.symbol_order)
    observable_cols = set(observable_preview.columns)
    safeguards = [
        {"safeguard": "environment_only_primary_input_contract", "passed": True, "detail": "observable preview contains environment/time/reporters only; GSM outputs added only after smoke/dataset generation"},
        {"safeguard": "no_raw_rxncon_state_in_observable_preview", "passed": len(observable_cols.intersection(boolnet_ids)) == 0, "detail": "raw Boolean node IDs only in npz lockbox; reporter names starting with R are allowed"},
        {"safeguard": "interface_complexity_above_old_hidden", "passed": bool(get_rank.get("GSM_interface", 0) > 2.789), "detail": f"GSM interface entropy rank {get_rank.get('GSM_interface', np.nan):.3f}"},
        {"safeguard": "biosensor_complexity_not_noise_only", "passed": bool(get_rank.get("biosensors_noiseless", 0) > 2.108), "detail": f"noiseless reporter entropy rank {get_rank.get('biosensors_noiseless', np.nan):.3f}"},
        {"safeguard": "noisy_reporter_rank_not_used_as_biology", "passed": bool(get_rank.get("biosensors_noisy", 0) / max(get_rank.get("biosensors_noiseless", np.nan), EPS) < 2.0), "detail": f"noisy/noiseless reporter entropy-rank ratio {get_rank.get('biosensors_noisy', np.nan) / max(get_rank.get('biosensors_noiseless', np.nan), EPS):.3f}"},
    ]
    safeguards.extend(reporter_safeguards.to_dict("records"))
    pd.DataFrame(safeguards).to_csv(OUT_DIR / "rxncon_gsm_generator_safeguards.csv", index=False)

    if args.with_gsm_smoke:
        gsm_traj, gsm_flux, gsm_cons, gsm_summary = dense_sparse_smoke(panel, interface_arr, interface_names)
        gsm_traj.to_csv(OUT_DIR / "rxncon_gsm_smoke_dense_sparse_trajectories.csv", index=False)
        gsm_flux.to_csv(OUT_DIR / "rxncon_gsm_smoke_dense_sparse_fluxes.csv", index=False)
        gsm_cons.to_csv(OUT_DIR / "rxncon_gsm_smoke_dense_sparse_constraints.csv", index=False)
        gsm_summary.to_csv(OUT_DIR / "rxncon_gsm_smoke_dense_sparse_summary.csv", index=False)

    provenance = {
        "rxncon_model_file": str(rxncon_screen.MODEL_XLS.relative_to(ROOT)),
        "rxncon_model_commit": json.loads((rxncon_screen.OUT_DIR / "rxncon_model_inventory.json").read_text()).get("models_repo_commit"),
        "rxncon_runtime_commit": json.loads((rxncon_screen.OUT_DIR / "rxncon_model_inventory.json").read_text()).get("rxncon_repo_commit"),
        "yeast_gem_selected_asset": pd.read_csv(gem.DATA / "yeast_gem_selected_asset.csv").iloc[0].to_dict(),
        "generator_stage": "rxncon_to_GSM_interface_gate",
        "rxncon_state_lockbox": True,
        "learner_input_contract": "environment only; historical outputs are biomass/product/reporters after GSM generation",
    }
    (OUT_DIR / "rxncon_gsm_generator_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    save_figures(panel, module_arr, module_names, interface_arr, interface_names, reporters_noisy, reporter_names, complexity)
    print("rxncon -> GSM generator gate complete")
    print(complexity.to_string(index=False))
    print(pd.DataFrame(safeguards).to_string(index=False))


if __name__ == "__main__":
    main()
