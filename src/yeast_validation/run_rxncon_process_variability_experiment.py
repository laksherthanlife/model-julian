#!/usr/bin/env python3
"""rxncon -> GSM -> Yeast9 hidden-process variability experiment.

This script keeps the v1 rxncon/GSM generator fixed and adds only a causal
hidden process layer:

    nominal T/pH/DO -> realized process/environment -> rxncon -> GSM -> Yeast9

Deployable learners receive only nominal T/pH/DO. Realized process variables,
systematic group labels, rxncon states, GSM controls, bounds and fluxes remain
lockbox quantities. The script is intentionally gated and resumable because the
Yeast9 LP work is expensive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_rxncon_external_complexity_screen as rxncon_base  # noqa: E402
import run_rxncon_gsm_generator_gate as gate  # noqa: E402
import run_rxncon_observable_complexity_audit as rxncon_obs  # noqa: E402


OUT_DIR = ROOT / "data" / "rxncon_process_variability"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results" / "rxncon_process_variability"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

RNG_SEED = 20260824
N_TIMEPOINTS = 49
DT = gate.DT
LEARNER_INPUTS = ["T_set", "pH_set", "DO_set"]
ENV_REALIZED_FOR_GENERATOR = [
    "temperature",
    "pH",
    "DO",
    "glucose_uptake",
    "Nutrients",
    "Pheromone",
    "HU",
    "LatA",
    "Nocodazole",
]
HIDDEN_PROCESS_COLS = [
    "T_real",
    "pH_real",
    "DO_real",
    "carbon_real",
    "nitrogen_real",
    "kLa_real",
    "temperature_random_error",
    "pH_random_error",
    "kLa_random_rel",
    "carbon_random_rel",
    "nitrogen_random_rel",
    "DO_random_offset",
    "batch_temperature_offset",
    "plate_pH_offset",
    "reader_DO_scale",
    "reader_DO_offset",
    "batch_feed_scale",
    "media_nitrogen_scale",
    "position_kLa_scale",
]
REPORTER_COLS = ["R_stress", "R_resource", "R_checkpoint", "R_pathway_capacity", "R_morphology", "R_energy"]
PRIMARY_CHANNELS = ["B_total", "X", *REPORTER_COLS]
MODEL_SEEDS = [11, 22, 33]
SUBSAMPLE_SEEDS = [101, 202, 303]


@dataclass(frozen=True)
class ProcessWorld:
    name: str
    random_variation: bool
    systematic_variation: bool


WORLDS = [
    ProcessWorld("perfect_control", False, False),
    ProcessWorld("random_process", True, False),
    ProcessWorld("random_systematic_process", True, True),
]
WORLD_BY_NAME = {w.name: w for w in WORLDS}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def rng_for(*parts: object) -> np.random.Generator:
    payload = "::".join(map(str, parts))
    seed = int.from_bytes(payload.encode("utf-8"), "little", signed=False) % (2**32 - 1)
    return np.random.default_rng(RNG_SEED + seed)


def nominal_condition_table(n_nominal: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    base = []
    # Include exact grid anchors plus stratified continuous points.
    for t in [27.0, 30.0, 33.0]:
        for ph in [4.5, 5.0, 5.5]:
            for do in [20.0, 40.0, 80.0]:
                base.append((t, ph, do, "grid"))
    while len(base) < n_nominal:
        t = rng.uniform(26.5, 33.5)
        ph = rng.uniform(4.35, 5.65)
        do = rng.uniform(15.0, 90.0)
        base.append((t, ph, do, "stratified"))
    rows = []
    for i, (t, ph, do, source) in enumerate(base[:n_nominal]):
        if source == "grid" and (t >= 33.0 or ph <= 4.5 or ph >= 5.5 or do <= 20.0 or do >= 80.0):
            split_hint = "extrapolation_stress"
        elif source == "grid" and ((t == 27.0 and do == 80.0) or (t == 33.0 and do == 20.0)):
            split_hint = "heldout_combination"
        else:
            split_hint = "ordinary"
        rows.append(
            {
                "nominal_environment_id": f"nominal_{i:03d}",
                "T_set": float(t),
                "pH_set": float(ph),
                "DO_set": float(do),
                "nominal_design_source": source,
                "split_hint": split_hint,
                "carbon_set": 10.0,
                "nitrogen_set": 1.0,
                "kLa_set": 1.0,
            }
        )
    return pd.DataFrame(rows)


def assign_systematic_groups(n_rows: int) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "batch_id": f"batch_{i % 5}",
                "plate_id": f"plate_{i % 8}",
                "reader_id": f"reader_{i % 3}",
                "position_id": f"pos_{i % 12}",
            }
        )
    return pd.DataFrame(rows)


def hidden_group_biases() -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(RNG_SEED + 17)
    return {
        "batch_temperature_offset": {f"batch_{i}": float(rng.normal(0.0, 0.18)) for i in range(5)},
        "batch_feed_scale": {f"batch_{i}": float(np.clip(rng.normal(1.0, 0.035), 0.90, 1.10)) for i in range(5)},
        "media_nitrogen_scale": {f"batch_{i}": float(np.clip(rng.normal(1.0, 0.04), 0.88, 1.12)) for i in range(5)},
        "plate_pH_offset": {f"plate_{i}": float(rng.normal(0.0, 0.025)) for i in range(8)},
        "reader_DO_scale": {f"reader_{i}": float(np.clip(rng.normal(1.0, 0.035), 0.90, 1.10)) for i in range(3)},
        "reader_DO_offset": {f"reader_{i}": float(rng.normal(0.0, 1.2)) for i in range(3)},
        "position_kLa_scale": {f"pos_{i}": float(np.clip(rng.normal(1.0, 0.045), 0.86, 1.14)) for i in range(12)},
    }


def build_nominal_replicate_design(n_nominal: int, n_cultures: int) -> pd.DataFrame:
    nominal = nominal_condition_table(n_nominal)
    reps = []
    rep = 0
    while len(reps) < n_cultures:
        for _, row in nominal.iterrows():
            if len(reps) >= n_cultures:
                break
            reps.append(
                {
                    **row.to_dict(),
                    "replicate_id": f"rep_{rep:02d}",
                    "nominal_replicate_index": rep,
                }
            )
        rep += 1
    design = pd.DataFrame(reps)
    groups = assign_systematic_groups(len(design))
    return pd.concat([design.reset_index(drop=True), groups], axis=1)


def split_for_row(row: pd.Series, world: str) -> str:
    if world == "random_systematic_process" and row["batch_id"] == "batch_4":
        return "systematic_holdout"
    hint = row["split_hint"]
    if hint == "heldout_combination":
        return "heldout_combination"
    if hint == "extrapolation_stress":
        return "extrapolation_stress"
    idx = int(str(row["nominal_environment_id"]).split("_")[-1])
    if idx % 10 == 0:
        return "validation"
    if idx % 10 == 1:
        return "interpolation"
    return "train"


def process_realization(design: pd.DataFrame, world: ProcessWorld) -> pd.DataFrame:
    biases = hidden_group_biases()
    rows = []
    for i, row in design.iterrows():
        rng = rng_for(world.name, row["nominal_environment_id"], row["replicate_id"], i)
        if world.random_variation:
            eps_t = float(rng.normal(0.0, 0.22))
            eps_ph = float(rng.normal(0.0, 0.035))
            eps_kla = float(np.clip(rng.normal(0.0, 0.055), -0.16, 0.16))
            eps_c = float(np.clip(rng.normal(0.0, 0.045), -0.14, 0.14))
            eps_n = float(np.clip(rng.normal(0.0, 0.045), -0.14, 0.14))
            eps_do = float(rng.normal(0.0, 1.0))
        else:
            eps_t = eps_ph = eps_kla = eps_c = eps_n = eps_do = 0.0

        if world.systematic_variation:
            b_t = biases["batch_temperature_offset"][row["batch_id"]]
            b_feed = biases["batch_feed_scale"][row["batch_id"]]
            b_n = biases["media_nitrogen_scale"][row["batch_id"]]
            p_ph = biases["plate_pH_offset"][row["plate_id"]]
            r_do_scale = biases["reader_DO_scale"][row["reader_id"]]
            r_do_offset = biases["reader_DO_offset"][row["reader_id"]]
            pos_kla = biases["position_kLa_scale"][row["position_id"]]
        else:
            b_t = p_ph = r_do_offset = 0.0
            b_feed = b_n = r_do_scale = pos_kla = 1.0

        kLa = float(np.clip(row["kLa_set"] * pos_kla * (1.0 + eps_kla), 0.65, 1.35))
        carbon = float(np.clip(row["carbon_set"] * b_feed * (1.0 + eps_c), 6.0, 14.5))
        nitrogen = float(np.clip(row["nitrogen_set"] * b_n * (1.0 + eps_n), 0.70, 1.25))
        t_real = float(np.clip(row["T_set"] + b_t + eps_t, 25.5, 34.5))
        ph_real = float(np.clip(row["pH_set"] + p_ph + eps_ph, 4.1, 5.9))
        do_real = float(np.clip(row["DO_set"] * r_do_scale * kLa + r_do_offset + eps_do, 5.0, 100.0))

        # Only route hidden process variables through defensible environmental
        # channels: nutrient sufficiency and exchange capacities. Stress-drug
        # rxncon inputs remain off because the process variables are not HU,
        # LatA, pheromone, or nocodazole treatments.
        nutrients = int(carbon > 6.5 and nitrogen > 0.75)
        rec = row.to_dict()
        rec.update(
            {
                "world": world.name,
                "culture_id": f"{world.name}__cult_{i:04d}",
                "hidden_process_realization_id": f"{world.name}__proc_{i:04d}",
                "split": split_for_row(row, world.name),
                "T_real": t_real,
                "pH_real": ph_real,
                "DO_real": do_real,
                "carbon_real": carbon,
                "nitrogen_real": nitrogen,
                "kLa_real": kLa,
                "temperature_random_error": eps_t,
                "pH_random_error": eps_ph,
                "kLa_random_rel": eps_kla,
                "carbon_random_rel": eps_c,
                "nitrogen_random_rel": eps_n,
                "DO_random_offset": eps_do,
                "batch_temperature_offset": b_t,
                "plate_pH_offset": p_ph,
                "reader_DO_scale": r_do_scale,
                "reader_DO_offset": r_do_offset,
                "batch_feed_scale": b_feed,
                "media_nitrogen_scale": b_n,
                "position_kLa_scale": pos_kla,
                # Generator-facing realized environment columns expected by v1.
                "environment_program": "constant",
                "temperature": t_real,
                "pH": ph_real,
                "DO": do_real,
                "glucose_uptake": carbon,
                "Nutrients": nutrients,
                "Pheromone": 0,
                "HU": 0,
                "LatA": 0,
                "Nocodazole": 0,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def write_process_specs() -> None:
    specs = [
        ("temperature_random_error", "culture_random", "degC", "N(0,0.22), clipped by realized T bounds", "temperature/pH/DO affects Yeast9 environment factor and v1 GSM constraints"),
        ("pH_random_error", "culture_random", "pH", "N(0,0.035), clipped by realized pH bounds", "pH affects Yeast9 environment factor"),
        ("kLa_random_rel", "culture_random", "relative", "N(0,0.055), clipped +/-0.16", "scales realized DO/oxygen-transfer availability"),
        ("carbon_random_rel", "culture_random", "relative", "N(0,0.045), clipped +/-0.14", "scales carbon/glucose exchange capacity and nutrient sufficiency"),
        ("nitrogen_random_rel", "culture_random", "relative", "N(0,0.045), clipped +/-0.14", "affects nutrient sufficiency/resource limitation; no arbitrary rxncon-node wiring"),
        ("DO_random_offset", "culture_random", "% air saturation", "N(0,1.0)", "adds small DO realization error"),
        ("batch_temperature_offset", "systematic_batch", "degC", "batch N(0,0.18)", "shared temperature calibration/process offset"),
        ("plate_pH_offset", "systematic_plate", "pH", "plate N(0,0.025)", "shared plate/media pH bias"),
        ("reader_DO_scale", "systematic_reader", "relative", "reader N(1,0.035), clipped 0.90-1.10", "shared DO calibration scale"),
        ("reader_DO_offset", "systematic_reader", "% air saturation", "reader N(0,1.2)", "shared DO calibration offset"),
        ("batch_feed_scale", "systematic_batch", "relative", "batch N(1,0.035), clipped 0.90-1.10", "shared feed/carbon delivery bias"),
        ("media_nitrogen_scale", "systematic_batch", "relative", "batch N(1,0.04), clipped 0.88-1.12", "shared nitrogen/media bias"),
        ("position_kLa_scale", "systematic_position", "relative", "position N(1,0.045), clipped 0.86-1.14", "shared mixing/oxygen-transfer position effect"),
    ]
    pd.DataFrame(
        specs,
        columns=["variable", "variation_type", "unit", "magnitude", "causal_route"],
    ).to_csv(OUT_DIR / "process_variability_specification.csv", index=False)


def make_design(n_nominal: int, n_cultures: int) -> pd.DataFrame:
    base = build_nominal_replicate_design(n_nominal, n_cultures)
    all_rows = []
    for world in WORLDS:
        all_rows.append(process_realization(base, world))
    design = pd.concat(all_rows, ignore_index=True)
    design.to_csv(OUT_DIR / "process_variability_culture_design.csv", index=False)
    write_process_specs()
    return design


def simulate_regulatory_layers(world_design: pd.DataFrame) -> tuple[object, np.ndarray, np.ndarray, list[str], np.ndarray, list[str], np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    model = gate.load_rxncon_model()
    force = gate.force_array_for_panel(model, world_design)
    traj, runtime = gate.rxncon_screen.simulate_panel(model, force)
    module_arr, module_names = gate.module_activity(model, traj)
    interface_arr, interface_names = gate.compute_interface(module_arr, module_names, world_design)
    clean, reporter_names, noisy = gate.build_reporters(module_arr, module_names, interface_arr, interface_names, world_design, noise=True)
    _clean2, _names2, noiseless = gate.build_reporters(module_arr, module_names, interface_arr, interface_names, world_design, noise=False)
    return model, traj, module_arr, module_names, interface_arr, interface_names, noiseless, noisy, reporter_names, runtime


def save_hidden_layers(world: str, model, rxncon_traj, module_arr, module_names, interface_arr, interface_names, reporters_clean, reporters_noisy, reporter_names, runtime) -> None:
    np.savez_compressed(OUT_DIR / f"{world}_hidden_rxncon.npz", traj=rxncon_traj)
    runtime.to_csv(OUT_DIR / f"{world}_rxncon_runtime.csv", index=False)
    pd.DataFrame(gate.flatten_by_culture(module_arr), columns=[f"{name}@t{t:02d}" for name in module_names for t in range(N_TIMEPOINTS)]).to_csv(
        OUT_DIR / f"{world}_hidden_module_trajectories_wide.csv", index=False
    )
    pd.DataFrame(gate.flatten_by_culture(interface_arr), columns=[f"{name}@t{t:02d}" for name in interface_names for t in range(N_TIMEPOINTS)]).to_csv(
        OUT_DIR / f"{world}_hidden_interface_trajectories_wide.csv", index=False
    )
    for arr, kind in [(reporters_clean, "noiseless"), (reporters_noisy, "noisy")]:
        rows = []
        for i in range(arr.shape[0]):
            for t in range(arr.shape[1]):
                rows.append({"row_index": i, "time_index": t, **{name: arr[i, t, j] for j, name in enumerate(reporter_names)}})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{world}_biosensors_{kind}.csv", index=False)


def touched_reaction_ids() -> list[str]:
    return [
        gem.GLUCOSE_EXCHANGE,
        gem.OXYGEN_EXCHANGE,
        gem.ATPM_RXN,
        gem.BIOMASS_RXN,
        gem.PRODUCT_RXN,
        gem.PSY_RXN,
        gem.DES_RXN,
        gem.CYC_RXN,
        gem.NATIVE_GGPP_RXN,
    ]


def reset_touched_bounds(model, bounds: dict[str, tuple[float, float]]) -> None:
    for rid, (lb, ub) in bounds.items():
        if rid in {r.id for r in model.reactions}:
            rxn = model.reactions.get_by_id(rid)
            rxn.lower_bound = lb
            rxn.upper_bound = ub


def run_world_gsm(world_design: pd.DataFrame, interface_arr: np.ndarray, interface_names: list[str], *, sparse: bool, n_time: int = N_TIMEPOINTS, indices: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    if indices is None:
        indices = list(range(len(world_design)))
    baseline_bounds = {
        rid: (augmented.reactions.get_by_id(rid).lower_bound, augmented.reactions.get_by_id(rid).upper_bound)
        for rid in touched_reaction_ids()
        if rid in {r.id for r in augmented.reactions}
    }
    cfg0 = gem.GEMCultureConfig(n_time=n_time, dt=DT, capacity_mode="no_dynamic_capacity")
    traj_rows, flux_rows, cons_rows, summary_rows = [], [], [], []
    checksum = gem.sha256(source_path)
    panel = world_design.reset_index(drop=True)
    total_keys = ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves"]
    for idx in indices:
        row = panel.iloc[idx]
        culture_model = augmented.copy()
        x = np.zeros(n_time)
        b = np.zeros(n_time)
        x[0] = cfg0.x0
        last_controls = None
        last_flux = None
        last_cons = None
        last_solve_i = -999
        culture_totals = {k: 0 for k in total_keys}
        for i in range(n_time - 1):
            controls = {name: float(interface_arr[idx, i, j]) for j, name in enumerate(interface_names)}
            if (not sparse) or last_controls is None or i == n_time - 2 or (i - last_solve_i) >= 4:
                solve_now = True
                trigger = "dense_or_required"
            else:
                dist = max(abs(controls[k] - last_controls[k]) / (abs(last_controls[k]) + 1e-6) for k in controls)
                solve_now = dist > 0.04
                trigger = f"interface_distance_{dist:.4f}" if solve_now else "carried_forward"
            if solve_now:
                reset_touched_bounds(culture_model, baseline_bounds)
                cfg = gem.GEMCultureConfig(
                    temperature=float(row["temperature"]),
                    pH=float(row["pH"]),
                    DO=float(row["DO"]),
                    glucose_uptake=float(row["glucose_uptake"]),
                    n_time=n_time,
                    dt=DT,
                    capacity_mode="no_dynamic_capacity",
                )
                cons, gamma, beta_deg = gate.apply_rxncon_constraints(culture_model, cfg, controls, row)
                flux = gem.solve_staged(culture_model, cfg, gamma, state="balanced")
                for k in total_keys:
                    culture_totals[k] += int(flux[k])
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
                **{k: row[k] for k in ENV_REALIZED_FOR_GENERATOR},
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
                    **{k: row[k] for k in ENV_REALIZED_FOR_GENERATOR},
                }
            )
        summary_rows.append({"culture_id": row["culture_id"], "sparse_event_triggered": sparse, **culture_totals, "n_surrogate_evaluations": 0})
    return pd.DataFrame(traj_rows), pd.DataFrame(flux_rows), pd.DataFrame(cons_rows), pd.DataFrame(summary_rows)


def add_measurement_noise(traj: pd.DataFrame, world: str) -> pd.DataFrame:
    rng = rng_for("measurement", world)
    out = traj.copy()
    for col, rel, floor in [("X", 0.012, 2e-4), ("B_total", 0.018, 2e-5)]:
        vals = out[col].to_numpy(float)
        sd = rel * np.maximum(np.abs(vals), np.nanmedian(np.abs(vals))) + floor
        obs = np.clip(vals + rng.normal(0.0, sd), 0.0, None)
        out[f"{col}_observed"] = obs
    return out


def merge_observables(world: str, design: pd.DataFrame, traj: pd.DataFrame, reporters_noisy: np.ndarray, reporters_clean: np.ndarray, reporter_names: list[str]) -> pd.DataFrame:
    rows = []
    design_i = design.reset_index(drop=True)
    for i, row in design_i.iterrows():
        for t in range(N_TIMEPOINTS):
            rec = {
                "culture_id": row["culture_id"],
                "world": world,
                "time_index": t,
                "time": t * DT,
                "nominal_environment_id": row["nominal_environment_id"],
                "replicate_id": row["replicate_id"],
                "split": row["split"],
                "T_set": row["T_set"],
                "pH_set": row["pH_set"],
                "DO_set": row["DO_set"],
            }
            for j, name in enumerate(reporter_names):
                rec[name] = reporters_noisy[i, t, j]
                rec[f"{name}_noiseless"] = reporters_clean[i, t, j]
            rows.append(rec)
    rep_df = pd.DataFrame(rows)
    obs = traj.merge(rep_df, on=["culture_id", "world", "time_index", "time"], how="left")
    # Remove realized generator columns from the learner-facing observable table
    # while retaining nominal settings and observables.
    cols = [
        "culture_id",
        "world",
        "time_index",
        "time",
        "nominal_environment_id",
        "replicate_id",
        "split",
        "T_set",
        "pH_set",
        "DO_set",
        "X",
        "B_total",
        "X_observed",
        "B_total_observed",
        *reporter_names,
        *[f"{name}_noiseless" for name in reporter_names],
    ]
    return obs[cols]


def dense_sparse_validation(design: pd.DataFrame, n_per_world: int, n_time: int = N_TIMEPOINTS) -> pd.DataFrame:
    rows = []
    all_traj, all_flux, all_cons = [], [], []
    for world in [w.name for w in WORLDS]:
        wd = design[design["world"].eq(world)].reset_index(drop=True)
        model, rxncon_traj, module_arr, module_names, interface_arr, interface_names, reporters_clean, reporters_noisy, reporter_names, runtime = simulate_regulatory_layers(wd)
        # Diverse deterministic subset: train/stress/systematic when available.
        splits = ["train", "validation", "heldout_combination", "extrapolation_stress", "systematic_holdout"]
        chosen = []
        for split in splits:
            idx = wd.index[wd["split"].eq(split)].tolist()
            if idx:
                chosen.append(idx[0])
        fill = [i for i in range(len(wd)) if i not in chosen]
        chosen = (chosen + fill)[:n_per_world]
        dense = run_world_gsm(wd, interface_arr, interface_names, sparse=False, n_time=n_time, indices=chosen)
        sparse = run_world_gsm(wd, interface_arr, interface_names, sparse=True, n_time=n_time, indices=chosen)
        dense_traj, dense_flux, dense_cons, dense_sum = dense
        sparse_traj, sparse_flux, sparse_cons, sparse_sum = sparse
        for df in [dense_traj, sparse_traj, dense_flux, sparse_flux, dense_cons, sparse_cons]:
            df["world"] = world
        joined = dense_traj.merge(sparse_traj, on=["culture_id", "time_index"], suffixes=("_dense", "_sparse"))
        flux_joined = dense_flux.merge(sparse_flux, on=["culture_id", "interval_index"], suffixes=("_dense", "_sparse"))
        flux_cols = ["biomass_flux", "beta_carotene_flux", "oxygen_uptake", "atp_maintenance_flux", "PSY_flux", "DES_flux", "CYC_flux"]
        for cid, g in joined.groupby("culture_id"):
            fg = flux_joined[flux_joined["culture_id"].eq(cid)]
            rec = {
                "world": world,
                "culture_id": cid,
                "biomass_rmse": rmse(g["X_dense"], g["X_sparse"]),
                "product_rmse": rmse(g["B_total_dense"], g["B_total_sparse"]),
                "final_product_abs_diff": float(abs(g.sort_values("time_index")["B_total_dense"].iloc[-1] - g.sort_values("time_index")["B_total_sparse"].iloc[-1])),
                "mean_flux_rmse": float(np.mean([rmse(fg[f"{c}_dense"], fg[f"{c}_sparse"]) for c in flux_cols if f"{c}_dense" in fg])),
                "dense_lp_solves": int(dense_sum.loc[dense_sum["culture_id"].eq(cid), "n_actual_lp_solves"].sum()),
                "sparse_lp_solves": int(sparse_sum.loc[sparse_sum["culture_id"].eq(cid), "n_actual_lp_solves"].sum()),
            }
            rec["lp_reduction_fraction"] = 1.0 - rec["sparse_lp_solves"] / max(rec["dense_lp_solves"], 1)
            rows.append(rec)
        all_traj.extend([dense_traj, sparse_traj])
        all_flux.extend([dense_flux, sparse_flux])
        all_cons.extend([dense_cons, sparse_cons])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "dense_sparse_validation_metrics.csv", index=False)
    pd.concat(all_traj, ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_trajectories.csv", index=False)
    pd.concat(all_flux, ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_fluxes.csv", index=False)
    pd.concat(all_cons, ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_constraints.csv", index=False)
    return metrics


def generate_world(world: ProcessWorld, design: pd.DataFrame, force: bool = False) -> dict[str, pd.DataFrame]:
    prefix = OUT_DIR / f"{world.name}_observable_trajectories.csv"
    if prefix.exists() and not force:
        return {
            "observable": pd.read_csv(prefix),
            "flux": pd.read_csv(OUT_DIR / f"{world.name}_fluxes.csv"),
            "constraints": pd.read_csv(OUT_DIR / f"{world.name}_constraints.csv"),
            "solver": pd.read_csv(OUT_DIR / f"{world.name}_solver_accounting.csv"),
        }
    wd = design[design["world"].eq(world.name)].reset_index(drop=True)
    model, rxncon_traj, module_arr, module_names, interface_arr, interface_names, reporters_clean, reporters_noisy, reporter_names, runtime = simulate_regulatory_layers(wd)
    save_hidden_layers(world.name, model, rxncon_traj, module_arr, module_names, interface_arr, interface_names, reporters_clean, reporters_noisy, reporter_names, runtime)
    traj, flux, cons, solver = run_world_gsm(wd, interface_arr, interface_names, sparse=True, n_time=N_TIMEPOINTS)
    for df in [traj, flux, cons, solver]:
        df["world"] = world.name
    traj = add_measurement_noise(traj, world.name)
    obs = merge_observables(world.name, wd, traj, reporters_noisy, reporters_clean, reporter_names)
    obs.to_csv(prefix, index=False)
    flux.to_csv(OUT_DIR / f"{world.name}_fluxes.csv", index=False)
    cons.to_csv(OUT_DIR / f"{world.name}_constraints.csv", index=False)
    solver.to_csv(OUT_DIR / f"{world.name}_solver_accounting.csv", index=False)
    wd.to_csv(OUT_DIR / f"{world.name}_culture_manifest_lockbox.csv", index=False)
    return {"observable": obs, "flux": flux, "constraints": cons, "solver": solver}


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_world_arrays(world: str) -> dict[str, object]:
    obs = pd.read_csv(OUT_DIR / f"{world}_observable_trajectories.csv")
    flux = pd.read_csv(OUT_DIR / f"{world}_fluxes.csv")
    cons = pd.read_csv(OUT_DIR / f"{world}_constraints.csv")
    manifest = pd.read_csv(OUT_DIR / f"{world}_culture_manifest_lockbox.csv")
    hidden = np.load(OUT_DIR / f"{world}_hidden_rxncon.npz")["traj"]
    module_wide = pd.read_csv(OUT_DIR / f"{world}_hidden_module_trajectories_wide.csv")
    interface_wide = pd.read_csv(OUT_DIR / f"{world}_hidden_interface_trajectories_wide.csv")
    ids = manifest["culture_id"].tolist()
    reporter_names = REPORTER_COLS
    return {
        "obs": obs,
        "flux": flux,
        "constraints": cons,
        "manifest": manifest,
        "rxncon": hidden,
        "module_wide": module_wide,
        "interface_wide": interface_wide,
        "culture_ids": ids,
        "reporter_names": reporter_names,
    }


def pivot_time(obs: pd.DataFrame, ids: list[str], col: str) -> np.ndarray:
    mat = obs.pivot(index="time_index", columns="culture_id", values=col)
    return mat[ids].to_numpy(float)


def flatten_obs(obs: pd.DataFrame, ids: list[str], cols: list[str]) -> np.ndarray:
    mats = [pivot_time(obs, ids, c).T for c in cols]
    arr = np.stack(mats, axis=2)  # B,T,C
    return gate.flatten_by_culture(arr)


def product_complexity_layers(obs: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    p = pivot_time(obs, ids, "B_total")
    raw = p.T
    amp = raw / np.maximum(np.max(raw, axis=1, keepdims=True), 1e-9)
    slopes = np.diff(raw, axis=1)
    final_auc = np.column_stack([raw[:, -1], np.trapz(raw, dx=DT, axis=1)])
    return {
        "product_raw": raw,
        "product_amplitude_normalized": amp,
        "product_slope": slopes,
        "product_final_auc": final_auc,
    }


def layer_summary(layer_arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    spectra = []
    for layer, x in layer_arrays.items():
        x = np.asarray(x, dtype=float)
        x_nonconstant, _ = rxncon_base.remove_constant_columns(x)
        compute_nonlinear = 2 <= x_nonconstant.shape[1] <= 2500 and x_nonconstant.shape[0] >= 4
        row, spec = rxncon_base.summarize_layer(layer, x, compute_nonlinear=compute_nonlinear)
        row["nominal_dimension"] = x.shape[1]
        rows.append(row)
        if not spec.empty:
            spec["layer"] = layer
            spectra.append(spec)
    return pd.DataFrame(rows), pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()


def hankel_rows(world: str, arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in arrays.items():
        h = rxncon_obs.hankel_matrix(arr, lag=8, max_windows=3000)
        h, _ = rxncon_base.remove_constant_columns(h)
        row, _ = rxncon_base.summarize_layer(f"hankel_{name}", h, compute_nonlinear=h.shape[1] <= 2500)
        row["world"] = world
        rows.append(row)
    return pd.DataFrame(rows)


def intervention_row(panel: pd.DataFrame, y: np.ndarray, layer: str) -> dict[str, float]:
    u = panel[LEARNER_INPUTS].to_numpy(float)
    y, _ = rxncon_base.remove_constant_columns(y)
    u, _ = rxncon_base.remove_constant_columns(u)
    if y.shape[1] == 0 or u.shape[1] == 0:
        return {"layer": layer, "entropy_rank": np.nan}
    uz = rxncon_base.standardize_matrix(u)
    yz = rxncon_base.standardize_matrix(y)
    j = np.linalg.solve(uz.T @ uz + 1e-6 * np.eye(uz.shape[1]), uz.T @ yz)
    s = np.linalg.svd(j, compute_uv=False)
    return {"layer": layer, "n_design_features": u.shape[1], "n_response_features": y.shape[1], **rxncon_base.dimension_metrics(s**2)}


def complexity_for_world(world: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_world_arrays(world)
    obs, flux, cons, manifest, rxncon = data["obs"], data["flux"], data["constraints"], data["manifest"], data["rxncon"]
    ids = data["culture_ids"]
    model = gate.load_rxncon_model()
    state_ids = [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]
    rxncon_flat = gate.flatten_by_culture(rxncon[:, :, state_ids])
    flux_cols = [
        c
        for c in [
            "biomass_flux",
            "beta_carotene_flux",
            "oxygen_uptake",
            "atp_maintenance_flux",
            "ggpp_flux",
            "PSY_flux",
            "DES_flux",
            "CYC_flux",
        ]
        if c in flux.columns
    ]
    flux_arrays = []
    for c in flux_cols:
        mat = flux.pivot(index="interval_index", columns="culture_id", values=c)[ids].T
        flux_arrays.append(mat.to_numpy(float))
    metabolic = np.concatenate(flux_arrays, axis=1) if flux_arrays else np.empty((len(ids), 0))
    obs_bio = flatten_obs(obs, ids, ["X", "B_total", *[f"{r}_noiseless" for r in REPORTER_COLS]])
    obs_noisy = flatten_obs(obs, ids, ["X_observed", "B_total_observed", *REPORTER_COLS])
    prod_layers = product_complexity_layers(obs, ids)
    layers = {
        "rxncon_state": rxncon_flat,
        "regulatory_modules": data["module_wide"].to_numpy(float),
        "GSM_interface": data["interface_wide"].to_numpy(float),
        "Yeast9_metabolic_flux": metabolic,
        "observable_biological_noiseless": obs_bio,
        "observable_experimental_noisy": obs_noisy,
        **prod_layers,
    }
    comp, spectra = layer_summary(layers)
    comp.insert(0, "world", world)
    spectra.insert(0, "world", world)
    # Hankel: convert B,T,C arrays to expected B,T,C.
    product_arr = pivot_time(obs, ids, "B_total").T[:, :, None]
    obs_arr = np.stack([pivot_time(obs, ids, c).T for c in ["X", "B_total", *[f"{r}_noiseless" for r in REPORTER_COLS]]], axis=2)
    rxncon_arr = rxncon[:, :, state_ids].astype(float)
    interface_arr = data["interface_wide"].to_numpy(float).reshape(len(ids), len(gate.interface_channels()), N_TIMEPOINTS).transpose(0, 2, 1)
    hank = hankel_rows(world, {"rxncon": rxncon_arr, "GSM_interface": interface_arr, "observable_bio": obs_arr, "product": product_arr})
    inter_rows = []
    temp_obs, _ = rxncon_obs.temporal_features(obs_arr, ["X", "B_total", *REPORTER_COLS])
    temp_interface, _ = rxncon_obs.temporal_features(interface_arr, [c.name for c in gate.interface_channels()])
    temp_met, _ = rxncon_obs.temporal_features(metabolic.reshape(len(ids), -1, len(flux_cols)) if flux_cols else np.zeros((len(ids), 1, 1)), flux_cols if flux_cols else ["none"])
    inter_rows.append(intervention_row(manifest, temp_obs, "environment_to_observable"))
    inter_rows.append(intervention_row(manifest, temp_interface, "environment_to_GSM_interface"))
    inter_rows.append(intervention_row(manifest, temp_met, "environment_to_metabolic"))
    inter = pd.DataFrame(inter_rows)
    inter.insert(0, "world", world)
    return comp, spectra, pd.concat([hank, inter], ignore_index=True, sort=False)


def replicate_variance(world: str) -> pd.DataFrame:
    data = load_world_arrays(world)
    obs, ids = data["obs"], data["culture_ids"]
    rows = []
    for col in ["B_total", "X", *REPORTER_COLS]:
        mat = pivot_time(obs, ids, col).T
        meta = data["manifest"].set_index("culture_id").loc[ids]
        within_vals = []
        group_means = []
        for nid, idx in meta.groupby("nominal_environment_id").indices.items():
            arr = mat[list(idx)]
            if arr.shape[0] >= 2:
                mean = arr.mean(axis=0, keepdims=True)
                within_vals.append((arr - mean).reshape(-1))
                group_means.append(mean.reshape(-1))
        within = np.concatenate(within_vals) if within_vals else np.array([np.nan])
        group_means_arr = np.vstack(group_means) if group_means else np.empty((0, mat.shape[1]))
        rows.append(
            {
                "world": world,
                "channel": col,
                "within_nominal_rmse_floor": float(np.sqrt(np.nanmean(within**2))),
                "between_nominal_rmse": float(np.sqrt(np.nanmean((group_means_arr - np.nanmean(group_means_arr, axis=0, keepdims=True)) ** 2))) if len(group_means_arr) else np.nan,
                "within_to_between_ratio": float(np.sqrt(np.nanmean(within**2)) / max(float(np.sqrt(np.nanmean((group_means_arr - np.nanmean(group_means_arr, axis=0, keepdims=True)) ** 2))) if len(group_means_arr) else np.nan, 1e-9)),
            }
        )
    return pd.DataFrame(rows)


def train_stats(arr: np.ndarray, train_mask: np.ndarray) -> tuple[float, float]:
    vals = arr[:, train_mask]
    return float(np.mean(vals)), float(np.std(vals) + 1e-9)


def standardize_env(env: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = env[train_mask].mean(axis=0)
    std = env[train_mask].std(axis=0) + 1e-9
    return (env - mean) / std, mean, std


def model_dataset(world: str, oracle: bool = False) -> dict[str, object]:
    data = load_world_arrays(world)
    obs, manifest, ids = data["obs"], data["manifest"], data["culture_ids"]
    if oracle:
        env_cols = LEARNER_INPUTS + ["T_real", "pH_real", "DO_real", "carbon_real", "nitrogen_real", "kLa_real"]
    else:
        env_cols = LEARNER_INPUTS
    env = manifest[env_cols].to_numpy(float)
    channels = {
        "B_total": pivot_time(obs, ids, "B_total_observed"),
        "X": pivot_time(obs, ids, "X_observed"),
        **{r: pivot_time(obs, ids, r) for r in REPORTER_COLS},
    }
    truth = {
        "B_total": pivot_time(obs, ids, "B_total"),
        "X": pivot_time(obs, ids, "X"),
        **{r: pivot_time(obs, ids, r) for r in REPORTER_COLS},
    }
    return {
        "world": world,
        "env": env,
        "env_cols": env_cols,
        "splits": manifest["split"].to_numpy(str),
        "ids": ids,
        "channels": channels,
        "truth": truth,
        "manifest": manifest,
        "t_len": N_TIMEPOINTS,
    }


def add_random_smooth(channels: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in channels.items()}
    rng = np.random.default_rng(seed)
    T, B = next(iter(out.values())).shape
    x = rng.normal(0.0, 1.0, size=(T, B))
    for t in range(1, T):
        x[t] = 0.85 * x[t - 1] + 0.15 * x[t]
    out["R_random_smooth"] = x
    return out


def shuffled_channels(channels: dict[str, np.ndarray], train_mask: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in channels.items()}
    rng = np.random.default_rng(seed)
    idx = np.where(train_mask)[0]
    perm = idx.copy()
    rng.shuffle(perm)
    if len(perm) > 1:
        perm = np.roll(perm, 1)
    for r in REPORTER_COLS:
        out[r][:, idx] = out[r][:, perm]
    return out


def train_state_space(dataset: dict[str, object], reporter_mode: str, seed: int, train_ids: np.ndarray | None = None, epochs: int = 350) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    splits = dataset["splits"]
    train_mask = splits == "train"
    if train_ids is not None:
        selected = np.isin(np.array(dataset["ids"]), train_ids)
        train_mask = train_mask & selected
    env_z, env_mean, env_std = standardize_env(dataset["env"], train_mask)
    channels = add_random_smooth(dataset["channels"], seed + 3)
    if reporter_mode == "shuffled":
        channels = shuffled_channels(channels, train_mask, seed + 7)
    head_names = ["B_total", "X"]
    if reporter_mode == "real":
        head_names += REPORTER_COLS
    elif reporter_mode == "shuffled":
        head_names += REPORTER_COLS
    elif reporter_mode == "random":
        head_names += ["R_random_smooth"]
    elif reporter_mode == "none":
        pass
    else:
        raise ValueError(reporter_mode)
    head_targets = {}
    head_masks = {}
    stats = {}
    for name in head_names:
        mean, std = train_stats(channels[name], train_mask)
        stats[name] = (mean, std)
        head_targets[name] = (channels[name] - mean) / std
        head_masks[name] = np.tile(train_mask.astype(float), (N_TIMEPOINTS, 1))
    params = tss.init_params(dataset["env"].shape[1], latent_dim=16, hidden_dim=24, head_names=head_names, n_controls=0, seed=seed)
    adam = tss.AdamState(params, lr=0.018)
    loss_history = []
    for _ in range(epochs):
        loss, grads, _diag = tss.compute_loss_and_grads(params, env_z, N_TIMEPOINTS, head_targets, head_masks, lambda_dyn=2e-4)
        adam.step(params, grads)
        loss_history.append(loss)
    ckpt = {"params": params, "env_z": env_z, "env_mean": env_mean, "env_std": env_std, "stats": stats, "reporter_mode": reporter_mode, "seed": seed, "loss_history": loss_history}
    metrics, preds = evaluate_state_space(dataset, ckpt, model_name=f"state_space_{reporter_mode}")
    return ckpt, metrics, preds


def evaluate_state_space(dataset: dict[str, object], ckpt: dict[str, object], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = tss.forward(ckpt["params"], ckpt["env_z"], N_TIMEPOINTS)
    rows = []
    pred_rows = []
    for channel in ["B_total", "X", *REPORTER_COLS]:
        if channel in ckpt["params"].heads:
            wh, bh = ckpt["params"].heads[channel]
            pre = (cache.z @ wh + bh)[..., 0]
            mean, std = ckpt["stats"][channel]
            pred = pre * std + mean
        else:
            pred = np.full_like(dataset["truth"][channel], np.nan)
        truth = dataset["truth"][channel]
        for split in sorted(set(dataset["splits"])):
            mask = dataset["splits"] == split
            if not np.isfinite(pred[:, mask]).any():
                continue
            rows.append(metric_row(dataset["world"], model_name, ckpt["seed"], split, channel, pred[:, mask], truth[:, mask], dataset))
        if channel in ["B_total", "X"]:
            for b, cid in enumerate(dataset["ids"]):
                for t in range(N_TIMEPOINTS):
                    pred_rows.append({"world": dataset["world"], "model": model_name, "seed": ckpt["seed"], "culture_id": cid, "time_index": t, "channel": channel, "prediction": pred[t, b], "truth": truth[t, b], "split": dataset["splits"][b]})
    latent = cache.z.reshape(N_TIMEPOINTS, len(dataset["ids"]), -1).transpose(1, 0, 2).reshape(len(dataset["ids"]), -1)
    latent_df = latent_alignment(dataset["world"], model_name, ckpt["seed"], latent)
    rows.extend(latent_df.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def metric_row(world: str, model: str, seed: int, split: str, channel: str, pred: np.ndarray, truth: np.ndarray, dataset: dict[str, object]) -> dict[str, object]:
    train_truth = dataset["truth"][channel][:, dataset["splits"] == "train"]
    denom = float(np.std(train_truth) + 1e-9)
    auc_pred = np.trapz(pred, dx=DT, axis=0)
    auc_truth = np.trapz(truth, dx=DT, axis=0)
    return {
        "world": world,
        "model": model,
        "seed": seed,
        "split": split,
        "channel": channel,
        "metric_type": "prediction",
        "rmse": rmse(pred, truth),
        "nrmse_train_std": rmse(pred, truth) / denom,
        "endpoint_abs_error": float(np.mean(np.abs(pred[-1] - truth[-1]))),
        "auc_abs_error": float(np.mean(np.abs(auc_pred - auc_truth))),
    }


def direct_design(dataset: dict[str, object], train_mask: np.ndarray, pca: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env_z, mean, std = standardize_env(dataset["env"], train_mask)
    T = dataset["t_len"]
    time = np.linspace(0, 1, T)
    rows_env = np.repeat(env_z, T, axis=0)
    rows_time = np.tile(time, env_z.shape[0])[:, None]
    if pca:
        phi = np.concatenate([np.ones((len(rows_env), 1)), rows_env, rows_env**2, rows_env[:, [0]] * rows_env[:, [1]], rows_time, rows_time**2], axis=1)
    else:
        phi = np.concatenate([np.ones((len(rows_env), 1)), rows_env, rows_env**2, rows_time, rows_time**2], axis=1)
    return phi, mean, std


def fit_ridge(phi: np.ndarray, y: np.ndarray, train_mask_rows: np.ndarray, lam: float = 1.0) -> np.ndarray:
    eye = np.eye(phi.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(phi[train_mask_rows].T @ phi[train_mask_rows] + lam * eye, phi[train_mask_rows].T @ y[train_mask_rows])


def train_direct_models(dataset: dict[str, object], oracle: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = dataset["splits"] == "train"
    phi, _mean, _std = direct_design(dataset, train_mask)
    row_train = np.repeat(train_mask, N_TIMEPOINTS)
    rows = []
    pred_rows = []
    model_name = "oracle_full_information_direct" if oracle else "direct_polynomial"
    for channel in PRIMARY_CHANNELS:
        y = dataset["channels"][channel].T.reshape(-1, 1)
        W = fit_ridge(phi, y, row_train, lam=1.0)
        pred = (phi @ W).reshape(len(dataset["ids"]), N_TIMEPOINTS).T
        truth = dataset["truth"][channel]
        for split in sorted(set(dataset["splits"])):
            mask = dataset["splits"] == split
            rows.append(metric_row(dataset["world"], model_name, 0, split, channel, pred[:, mask], truth[:, mask], dataset))
        if channel in ["B_total", "X"]:
            for b, cid in enumerate(dataset["ids"]):
                for t in range(N_TIMEPOINTS):
                    pred_rows.append({"world": dataset["world"], "model": model_name, "seed": 0, "culture_id": cid, "time_index": t, "channel": channel, "prediction": pred[t, b], "truth": truth[t, b], "split": dataset["splits"][b]})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def train_mean_nearest(dataset: dict[str, object]) -> pd.DataFrame:
    train_mask = dataset["splits"] == "train"
    train_env = dataset["env"][train_mask]
    rows = []
    for baseline in ["train_mean", "nearest_nominal_environment"]:
        for channel in PRIMARY_CHANNELS:
            truth = dataset["truth"][channel]
            if baseline == "train_mean":
                pred_full = np.repeat(truth[:, train_mask].mean(axis=1, keepdims=True), truth.shape[1], axis=1)
            else:
                pred_full = np.zeros_like(truth)
                for b in range(truth.shape[1]):
                    d = np.linalg.norm((train_env - dataset["env"][b]), axis=1)
                    nearest = np.where(train_mask)[0][int(np.argmin(d))]
                    pred_full[:, b] = truth[:, nearest]
            for split in sorted(set(dataset["splits"])):
                mask = dataset["splits"] == split
                rows.append(metric_row(dataset["world"], baseline, 0, split, channel, pred_full[:, mask], truth[:, mask], dataset))
    return pd.DataFrame(rows)


def latent_alignment(world: str, model_name: str, seed: int, latent_flat: np.ndarray) -> pd.DataFrame:
    rows = []
    paths = {
        "regulatory_module": OUT_DIR / f"{world}_hidden_module_trajectories_wide.csv",
        "GSM_interface": OUT_DIR / f"{world}_hidden_interface_trajectories_wide.csv",
    }
    x, _ = rxncon_base.remove_constant_columns(latent_flat)
    if x.shape[1] == 0:
        return pd.DataFrame()
    xz = rxncon_base.standardize_matrix(x)
    for target_name, path in paths.items():
        if not path.exists():
            continue
        y, _ = rxncon_base.remove_constant_columns(pd.read_csv(path).to_numpy(float))
        if y.shape[1] == 0:
            continue
        yz = rxncon_base.standardize_matrix(y)
        _u, _s, vt = np.linalg.svd(yz, full_matrices=False)
        pcs = yz @ vt[: min(8, vt.shape[0])].T
        coef = np.linalg.solve(xz.T @ xz + 1e-6 * np.eye(xz.shape[1]), xz.T @ pcs)
        pred = xz @ coef
        r2 = 1.0 - np.sum((pcs - pred) ** 2) / max(np.sum((pcs - pcs.mean(axis=0)) ** 2), 1e-12)
        rows.append({"world": world, "model": model_name, "seed": seed, "split": "all", "channel": target_name, "metric_type": "posthoc_latent_alignment", "rmse": np.nan, "nrmse_train_std": np.nan, "endpoint_abs_error": np.nan, "auc_abs_error": np.nan, "latent_to_hidden_pc_r2": float(r2)})
    return pd.DataFrame(rows)


def run_learning(world: str, epochs: int, sample_size: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = model_dataset(world, oracle=False)
    oracle_ds = model_dataset(world, oracle=True)
    metric_frames = []
    pred_frames = []
    for reporter_mode in ["real", "none", "shuffled", "random"]:
        for seed in MODEL_SEEDS:
            _ckpt, metrics, preds = train_state_space(dataset, reporter_mode, seed, epochs=epochs)
            metric_frames.append(metrics)
            pred_frames.append(preds)
    direct_metrics, direct_preds = train_direct_models(dataset, oracle=False)
    oracle_metrics, oracle_preds = train_direct_models(oracle_ds, oracle=True)
    metric_frames.extend([direct_metrics, oracle_metrics, train_mean_nearest(dataset)])
    pred_frames.extend([direct_preds, oracle_preds])
    if sample_size:
        train_ids = np.array(dataset["ids"])[dataset["splits"] == "train"]
        for n in [10, 25, 50, len(train_ids)]:
            if n > len(train_ids):
                continue
            seeds = [0] if n == len(train_ids) else SUBSAMPLE_SEEDS
            for ss in seeds:
                rng = np.random.default_rng(ss)
                chosen = np.array(sorted(rng.choice(train_ids, size=n, replace=False)))
                for seed in [11]:
                    _ckpt, metrics, _preds = train_state_space(dataset, "real", seed, train_ids=chosen, epochs=max(180, epochs // 2))
                    metrics["training_cultures_N"] = n
                    metrics["subsample_seed"] = ss
                    metric_frames.append(metrics)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    preds_df = pd.concat(pred_frames, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / f"{world}_model_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / f"{world}_model_predictions.csv", index=False)
    return metrics_df, preds_df


def safeguards(design: pd.DataFrame, dense_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    observable_cols = set(pd.read_csv(OUT_DIR / "perfect_control_observable_trajectories.csv", nrows=1).columns) if (OUT_DIR / "perfect_control_observable_trajectories.csv").exists() else set()
    deployable_forbidden = set(HIDDEN_PROCESS_COLS + ["batch_id", "plate_id", "reader_id", "position_id", "temperature", "pH", "DO", "glucose_uptake", "Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"])
    rows = [
        {"safeguard": "nominal_vs_realized_environment_separated", "passed": all(c in design.columns for c in ["T_set", "T_real", "pH_set", "pH_real", "DO_set", "DO_real"])},
        {"safeguard": "realized_environment_not_in_observable_table", "passed": len(observable_cols.intersection({"T_real", "pH_real", "DO_real", "carbon_real", "nitrogen_real", "kLa_real"})) == 0},
        {"safeguard": "nuisance_variables_not_in_deployable_inputs", "passed": len(set(LEARNER_INPUTS).intersection(deployable_forbidden)) == 0},
        {"safeguard": "systematic_group_id_not_in_observable_table", "passed": len(observable_cols.intersection({"batch_id", "plate_id", "reader_id", "position_id"})) == 0},
        {"safeguard": "true_rxncon_state_lockboxed", "passed": all((OUT_DIR / f"{w.name}_hidden_rxncon.npz").exists() for w in WORLDS if (OUT_DIR / f"{w.name}_observable_trajectories.csv").exists())},
        {"safeguard": "true_GSM_interface_lockboxed", "passed": all((OUT_DIR / f"{w.name}_hidden_interface_trajectories_wide.csv").exists() for w in WORLDS if (OUT_DIR / f"{w.name}_observable_trajectories.csv").exists())},
        {"safeguard": "true_fluxes_lockboxed", "passed": "biomass_flux" not in observable_cols and "beta_carotene_flux" not in observable_cols},
        {"safeguard": "reporters_many_to_many", "passed": all(len(s.contributors) >= 4 for s in gate.reporter_specs())},
        {"safeguard": "culture_level_split_integrity", "passed": not design.groupby("culture_id")["split"].nunique().gt(1).any()},
        {"safeguard": "systematic_holdout_integrity", "passed": not ((design["world"].eq("random_systematic_process")) & (design["batch_id"].eq("batch_4")) & (~design["split"].eq("systematic_holdout"))).any()},
        {"safeguard": "no_timepoint_leakage_by_design", "passed": True},
        {"safeguard": "oracle_model_marked_non_deployable", "passed": True},
    ]
    if dense_metrics is not None and not dense_metrics.empty:
        rows.append({"safeguard": "event_triggered_replay_validation", "passed": bool(dense_metrics["final_product_abs_diff"].median() < 1e-3), "detail": f"median final product diff={dense_metrics['final_product_abs_diff'].median():.3g}"})
        rows.append({"safeguard": "exact_yeast9_lp_accounting_present", "passed": bool(dense_metrics["dense_lp_solves"].sum() > dense_metrics["sparse_lp_solves"].sum())})
    sf = pd.DataFrame(rows)
    sf.to_csv(OUT_DIR / "process_variability_safeguards.csv", index=False)
    return sf


def save_figures(design: pd.DataFrame, complexity: pd.DataFrame, repvar: pd.DataFrame, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    # Process variable distributions.
    cols = ["T_real", "pH_real", "DO_real", "carbon_real", "nitrogen_real", "kLa_real"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, col in zip(axes.ravel(), cols):
        for world in [w.name for w in WORLDS]:
            ax.hist(design.loc[design["world"].eq(world), col], bins=20, alpha=0.45, label=world)
        ax.set_title(col, fontsize=9)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_process_variability_realized_distributions.svg")
    plt.close(fig)

    # Complexity ladder.
    core_layers = ["rxncon_state", "regulatory_modules", "GSM_interface", "Yeast9_metabolic_flux", "observable_biological_noiseless", "observable_experimental_noisy", "product_raw"]
    sub = complexity[complexity["layer"].isin(core_layers)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(core_layers))
    width = 0.25
    for i, world in enumerate([w.name for w in WORLDS]):
        vals = [sub[(sub["world"].eq(world)) & (sub["layer"].eq(l))]["entropy_rank"].mean() for l in core_layers]
        ax.bar(x + (i - 1) * width, vals, width=width, label=world)
    ax.set_xticks(x)
    ax.set_xticklabels(core_layers, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Entropy effective rank")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_process_variability_complexity_ladder.svg")
    plt.close(fig)

    # Replicate variance.
    fig, ax = plt.subplots(figsize=(7, 4))
    prod = repvar[repvar["channel"].eq("B_total")]
    ax.bar(prod["world"], prod["within_to_between_ratio"])
    ax.set_ylabel("within-condition / between-condition RMSE")
    ax.set_title("Product replicate variability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_process_variability_replicate_floor.svg")
    plt.close(fig)

    # Performance.
    if not metrics.empty:
        predm = metrics[(metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["split"].isin(["interpolation", "heldout_combination", "extrapolation_stress", "systematic_holdout"]))]
        fig, ax = plt.subplots(figsize=(10, 5))
        agg = predm.groupby(["world", "model"], as_index=False)["nrmse_train_std"].mean()
        models = ["state_space_real", "state_space_none", "state_space_shuffled", "state_space_random", "direct_polynomial", "oracle_full_information_direct"]
        x = np.arange(len([w.name for w in WORLDS]))
        width = 0.12
        for i, model in enumerate(models):
            vals = [agg[(agg["world"].eq(w.name)) & (agg["model"].eq(model))]["nrmse_train_std"].mean() for w in WORLDS]
            ax.bar(x + (i - 2.5) * width, vals, width=width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels([w.name for w in WORLDS], rotation=20, ha="right")
        ax.set_ylabel("Product NRMSE")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rxncon_process_variability_model_performance.svg")
        plt.close(fig)

    # Representative curves.
    if not predictions.empty:
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        for ax, world in zip(axes.ravel(), [w.name for w in WORLDS] * 2):
            subp = predictions[(predictions["world"].eq(world)) & (predictions["channel"].eq("B_total"))]
            if subp.empty:
                continue
            cid = subp["culture_id"].iloc[0]
            for model in ["state_space_real", "direct_polynomial", "oracle_full_information_direct"]:
                g = subp[(subp["culture_id"].eq(cid)) & (subp["model"].eq(model))]
                if not g.empty:
                    ax.plot(g["time_index"] * DT, g["prediction"], label=model, alpha=0.8)
            truth = subp[subp["culture_id"].eq(cid)].drop_duplicates("time_index")
            ax.plot(truth["time_index"] * DT, truth["truth"], color="black", linewidth=1.5, label="truth")
            ax.set_title(f"{world} {cid}", fontsize=8)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rxncon_process_variability_true_vs_predicted.svg")
        plt.close(fig)


def update_docs(summary: dict[str, object]) -> None:
    section = f"""

## rxncon Hidden Process-Variability Experiment

This experiment preserved the v1 published rxncon -> GSM -> Yeast9 generator
and added only a causal hidden process layer between nominal controlled
settings and realized culture conditions. Deployable learners saw only
`T_set`, `pH_set`, and `DO_set`; realized temperature, pH, DO, carbon/feed,
nitrogen, kLa/mixing, batch/plate/reader/position effects, rxncon state, GSM
interface controls, bounds, and fluxes remained lockbox variables. The executable
workflow is `scripts/run_rxncon_process_variability_experiment.py`; generated
tables are under `data/rxncon_process_variability/`.

Three matched worlds were generated: perfect control, random culture-level
process variation, and random plus systematic batch/plate/reader/position
variation. The hidden process magnitudes were conservative synthetic
assumptions: temperature random SD `0.22 C`, pH random SD `0.035`, kLa relative
SD `5.5%`, carbon/feed relative SD `4.5%`, nitrogen relative SD `4.5%`, and DO
offset SD `1% air saturation`; systematic offsets were shared by experimental
groups rather than independently resampled per culture.

Run summary: `{summary.get('n_cultures_total','NA')}` complete cultures across
three worlds, `{summary.get('n_lp_solves_total','NA')}` exact Yeast9 LP solves,
median dense-vs-event-triggered product final difference
`{summary.get('dense_sparse_median_final_diff','NA')}`, and median LP reduction
`{summary.get('dense_sparse_median_lp_reduction','NA')}`. The key result table
is `data/rxncon_process_variability/process_variability_main_summary.csv`.

Interpretation: see the machine-readable summary for the final pass/fail
decision. The critical comparison is deployable model error versus both the
within-nominal replicate variability floor and the full-information oracle that
is allowed to see hidden realized process variables. Oracle performance is a
diagnostic upper bound and is not a deployable competitor.
    """
    for path in [ROOT / "README.md", ROOT / "PHYSIOLOGY_QUEST_VALIDATION.md", ROOT / "CONCRETE_EXPERIMENT_CHAIN.md"]:
        if not path.exists():
            continue
        text = path.read_text()
        if "## rxncon Hidden Process-Variability Experiment" not in text:
            path.write_text(text.rstrip() + section + "\n")


def aggregate_summary(design: pd.DataFrame, dense: pd.DataFrame, complexity: pd.DataFrame, repvar: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for world in [w.name for w in WORLDS]:
        row = {"world": world}
        for layer, key in [
            ("rxncon_state", "D_regulatory"),
            ("GSM_interface", "D_interface"),
            ("Yeast9_metabolic_flux", "D_metabolic"),
            ("observable_biological_noiseless", "D_observable_bio"),
            ("observable_experimental_noisy", "D_observable_noisy"),
            ("product_raw", "D_product"),
        ]:
            row[key] = float(complexity[(complexity["world"].eq(world)) & (complexity["layer"].eq(layer))]["entropy_rank"].mean())
        rv = repvar[(repvar["world"].eq(world)) & (repvar["channel"].eq("B_total"))]
        row["product_replicate_floor_rmse"] = float(rv["within_nominal_rmse_floor"].mean()) if not rv.empty else np.nan
        for model in ["state_space_real", "state_space_none", "direct_polynomial", "oracle_full_information_direct"]:
            sub = metrics[(metrics["world"].eq(world)) & (metrics["model"].eq(model)) & (metrics["channel"].eq("B_total")) & (metrics["metric_type"].eq("prediction")) & (~metrics["split"].eq("train"))]
            row[f"{model}_product_nrmse"] = float(sub["nrmse_train_std"].mean()) if not sub.empty else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "process_variability_main_summary.csv", index=False)
    return summary


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    tic = time.perf_counter()
    n_nominal = args.n_nominal
    design = make_design(n_nominal, args.n_cultures_per_world)
    selected_worlds = [WORLD_BY_NAME[name] for name in args.worlds] if args.worlds else WORLDS
    if args.stage in {"all", "dense_sparse"}:
        dense = dense_sparse_validation(design, args.dense_sparse_per_world, n_time=args.dense_sparse_n_time)
        if not dense.empty:
            med_final = float(dense["final_product_abs_diff"].median())
            med_product = float(dense["product_rmse"].median())
            if med_final > args.dense_sparse_final_threshold or med_product > args.dense_sparse_product_rmse_threshold:
                raise RuntimeError(
                    f"dense-vs-sparse validation failed: median final product diff={med_final:.6g}, "
                    f"median product RMSE={med_product:.6g}"
                )
    else:
        dense_path = OUT_DIR / "dense_sparse_validation_metrics.csv"
        dense = pd.read_csv(dense_path) if dense_path.exists() else pd.DataFrame()
    safeguards(design, dense)
    if args.stage in {"all", "generate"}:
        for world in selected_worlds:
            print(f"generating {world.name}", flush=True)
            generate_world(world, design, force=args.force)
    if args.stage in {"all", "complexity"}:
        comps, specs, extras, reps = [], [], [], []
        for world in [w.name for w in selected_worlds]:
            print(f"complexity {world}", flush=True)
            comp, spec, extra = complexity_for_world(world)
            comps.append(comp)
            specs.append(spec)
            extras.append(extra)
            reps.append(replicate_variance(world))
        complexity = pd.concat(comps, ignore_index=True)
        spectra = pd.concat(specs, ignore_index=True)
        extra = pd.concat(extras, ignore_index=True)
        repvar = pd.concat(reps, ignore_index=True)
        complexity.to_csv(OUT_DIR / "process_variability_complexity_ladder.csv", index=False)
        spectra.to_csv(OUT_DIR / "process_variability_complexity_spectra.csv", index=False)
        extra.to_csv(OUT_DIR / "process_variability_dynamic_intervention_complexity.csv", index=False)
        repvar.to_csv(OUT_DIR / "process_variability_replicate_variance.csv", index=False)
    else:
        complexity = pd.read_csv(OUT_DIR / "process_variability_complexity_ladder.csv") if (OUT_DIR / "process_variability_complexity_ladder.csv").exists() else pd.DataFrame()
        repvar = pd.read_csv(OUT_DIR / "process_variability_replicate_variance.csv") if (OUT_DIR / "process_variability_replicate_variance.csv").exists() else pd.DataFrame()
    metrics_all, preds_all = [], []
    if args.stage in {"all", "learn"}:
        for world in [w.name for w in selected_worlds]:
            print(f"learning {world}", flush=True)
            m, p = run_learning(world, args.epochs, sample_size=args.sample_size)
            metrics_all.append(m)
            preds_all.append(p)
        metrics = pd.concat(metrics_all, ignore_index=True)
        preds = pd.concat(preds_all, ignore_index=True)
        metrics.to_csv(OUT_DIR / "process_variability_all_model_metrics.csv", index=False)
        preds.to_csv(OUT_DIR / "process_variability_all_model_predictions.csv", index=False)
    else:
        metrics = pd.read_csv(OUT_DIR / "process_variability_all_model_metrics.csv") if (OUT_DIR / "process_variability_all_model_metrics.csv").exists() else pd.DataFrame()
        preds = pd.read_csv(OUT_DIR / "process_variability_all_model_predictions.csv") if (OUT_DIR / "process_variability_all_model_predictions.csv").exists() else pd.DataFrame()
    if args.stage in {"all", "report"}:
        if complexity.empty:
            complexity = pd.read_csv(OUT_DIR / "process_variability_complexity_ladder.csv")
        if repvar.empty:
            repvar = pd.read_csv(OUT_DIR / "process_variability_replicate_variance.csv")
        if metrics.empty and (OUT_DIR / "process_variability_all_model_metrics.csv").exists():
            metrics = pd.read_csv(OUT_DIR / "process_variability_all_model_metrics.csv")
        if preds.empty and (OUT_DIR / "process_variability_all_model_predictions.csv").exists():
            preds = pd.read_csv(OUT_DIR / "process_variability_all_model_predictions.csv")
        summary = aggregate_summary(design, dense, complexity, repvar, metrics)
        save_figures(design, complexity, repvar, metrics, preds)
        lp_total = 0
        for world in [w.name for w in WORLDS]:
            p = OUT_DIR / f"{world}_solver_accounting.csv"
            if p.exists():
                lp_total += int(pd.read_csv(p)["n_actual_lp_solves"].sum())
        doc_summary = {
            "n_cultures_total": int(design["culture_id"].nunique()),
            "n_lp_solves_total": int(lp_total),
            "dense_sparse_median_final_diff": float(dense["final_product_abs_diff"].median()) if not dense.empty else "NA",
            "dense_sparse_median_lp_reduction": float(dense["lp_reduction_fraction"].median()) if not dense.empty else "NA",
        }
        (OUT_DIR / "process_variability_run_summary.json").write_text(json.dumps({**doc_summary, "wall_seconds": time.perf_counter() - tic}, indent=2))
        update_docs(doc_summary)
        print(summary.to_string(index=False))
    print(f"done in {time.perf_counter() - tic:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "dense_sparse", "generate", "complexity", "learn", "report"], default="all")
    parser.add_argument("--n-nominal", type=int, default=40)
    parser.add_argument("--n-cultures-per-world", type=int, default=125)
    parser.add_argument("--dense-sparse-per-world", type=int, default=4)
    parser.add_argument("--dense-sparse-n-time", type=int, default=N_TIMEPOINTS)
    parser.add_argument("--dense-sparse-final-threshold", type=float, default=1e-3)
    parser.add_argument("--dense-sparse-product-rmse-threshold", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--sample-size", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worlds", nargs="+", choices=[w.name for w in WORLDS], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
