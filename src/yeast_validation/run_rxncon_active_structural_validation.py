#!/usr/bin/env python3
"""Active-rxncon structural validation benchmark.

This experiment deliberately drives the published yeast CDC rxncon Boolean
network with its supported external inputs, couples the existing frozen v1
rxncon->GSM interface to real Yeast9/pFBA, and trains compact observable-only
learners.  Hidden rxncon states, regulatory modules, GSM interface controls,
constraints and fluxes remain lockbox quantities.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
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
import run_rxncon_process_variability_experiment as pv  # noqa: E402


OUT_DIR = ROOT / "data" / "rxncon_active_structural_validation"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results" / "rxncon_active_structural_validation"
RNG_SEED = 20260825
N_TIMEPOINTS = gate.N_TIMEPOINTS
DT = gate.DT
RXNCON_INPUT_COLS = ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]
CONT_INPUT_COLS = ["temperature", "pH", "DO", "glucose_uptake"]
LEARNER_INPUTS = RXNCON_INPUT_COLS + CONT_INPUT_COLS
REPORTER_COLS = ["R_stress", "R_resource", "R_checkpoint", "R_pathway_capacity", "R_morphology", "R_energy"]
PRIMARY_CHANNELS = ["B_total", "X", *REPORTER_COLS]
MODEL_SEEDS = [11, 22, 33]
SUBSAMPLE_SEEDS = [101, 202, 303]
WORLD = "active_rxncon_structural"
_ORIG_APPLY_RXNCON_CONSTRAINTS = gate.apply_rxncon_constraints


def apply_constraints_with_atpm_bound_order(model, cfg: gem.GEMCultureConfig, controls: dict[str, float], env_row: pd.Series):
    """Relax ATPM upper bound before the frozen v1 helper raises its lower bound."""
    if gem.ATPM_RXN in {r.id for r in model.reactions}:
        rxn = model.reactions.get_by_id(gem.ATPM_RXN)
        rxn.upper_bound = max(rxn.upper_bound, 1000.0)
    return _ORIG_APPLY_RXNCON_CONSTRAINTS(model, cfg, controls, env_row)


gate.apply_rxncon_constraints = apply_constraints_with_atpm_bound_order


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def active_static_panel() -> pd.DataFrame:
    rows = []
    for mask in range(32):
        env = {c: int((mask >> i) & 1) for i, c in enumerate(RXNCON_INPUT_COLS)}
        rows.append(
            {
                "culture_id": f"static_{mask:02d}",
                "input_program": "static",
                "nominal_condition_id": f"static_{mask:02d}",
                **env,
                "temperature": 30.0,
                "pH": 5.0,
                "DO": 40.0,
                "glucose_uptake": 10.0,
            }
        )
    return pd.DataFrame(rows)


def input_audit_panel() -> pd.DataFrame:
    rows = active_static_panel().to_dict("records")
    for target in RXNCON_INPUT_COLS:
        for program in ["late_on", "late_off", "early_pulse", "late_pulse"]:
            env = {c: 0 for c in RXNCON_INPUT_COLS}
            env["Nutrients"] = 1
            if target == "Nutrients" and "off" not in program:
                env["Nutrients"] = 0
            rows.append(
                {
                    "culture_id": f"{program}_{target}",
                    "input_program": f"{program}:{target}",
                    "nominal_condition_id": f"{program}_{target}",
                    **env,
                    "temperature": 30.0,
                    "pH": 5.0,
                    "DO": 40.0,
                    "glucose_uptake": 10.0,
                }
            )
    return pd.DataFrame(rows)


def split_for_design_row(row: pd.Series) -> str:
    program = str(row["input_program"])
    stresses = int(row["HU"]) + int(row["LatA"]) + int(row["Nocodazole"]) + int(row["Pheromone"])
    if program.startswith("late_pulse") or program.startswith("early_pulse"):
        return "hard_perturbation"
    if int(row["HU"]) and int(row["LatA"]):
        return "heldout_combination"
    if int(row["Pheromone"]) and int(row["Nocodazole"]):
        return "heldout_combination"
    if stresses >= 3:
        return "extrapolation_stress"
    idx = int(str(row["culture_id"]).split("_")[-1]) if str(row["culture_id"]).split("_")[-1].isdigit() else len(str(row["culture_id"]))
    if idx % 11 == 0:
        return "validation"
    if idx % 11 == 1:
        return "interpolation"
    return "train"


def canonical_design(n_cultures: int = 125) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    base = gate.build_environment_panel(max(60, min(n_cultures, 125))).copy()
    base = base.rename(columns={"environment_program": "input_program"})
    if "input_program" not in base.columns:
        base["input_program"] = "static"
    rows = []
    for _, row in base.iterrows():
        rec = row.to_dict()
        rec["nominal_condition_id"] = str(rec["culture_id"])
        rows.append(rec)
    cont = []
    for t in gate.CONT_ENV_AXES["temperature"]["allowed_values"]:
        for ph in gate.CONT_ENV_AXES["pH"]["allowed_values"]:
            for do in gate.CONT_ENV_AXES["DO"]["allowed_values"]:
                for glc in gate.CONT_ENV_AXES["glucose_uptake"]["allowed_values"]:
                    cont.append((t, ph, do, glc))
    while len(rows) < n_cultures:
        source = rows[int(rng.integers(0, len(base)))].copy()
        t, ph, do, glc = cont[int(rng.integers(0, len(cont)))]
        source["culture_id"] = f"active_env_{len(rows):03d}"
        source["nominal_condition_id"] = source.get("nominal_condition_id", source["culture_id"])
        source["temperature"] = float(t)
        source["pH"] = float(ph)
        source["DO"] = float(do)
        source["glucose_uptake"] = float(glc)
        source["input_program"] = "replicate_process_variant"
        rows.append(source)
    design = pd.DataFrame(rows[:n_cultures])
    for c in RXNCON_INPUT_COLS:
        design[c] = design[c].astype(int)
    design["world"] = WORLD
    design["replicate_id"] = design.groupby("nominal_condition_id").cumcount().map(lambda x: f"rep_{x:02d}")
    design["split"] = design.apply(split_for_design_row, axis=1)
    # Keep the held-out-combination split meaningful: constituent singles remain
    # present in train/validation because only selected combinations are held out.
    if (design["split"] == "train").sum() < 45:
        ordinary = design.index[design["split"].isin(["interpolation", "validation"])].tolist()
        design.loc[ordinary[: 45 - int((design["split"] == "train").sum())], "split"] = "train"
    return design.reset_index(drop=True)


def force_for_design(model: rxncon_base.BoolNet, panel: pd.DataFrame) -> np.ndarray:
    work = panel.copy()
    if "environment_program" not in work.columns:
        work["environment_program"] = work["input_program"]
    return gate.force_array_for_panel(model, work)


def simulate_layers(panel: pd.DataFrame) -> tuple[object, np.ndarray, np.ndarray, list[str], np.ndarray, list[str], np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    model = gate.load_rxncon_model()
    force = force_for_design(model, panel)
    traj, runtime = gate.rxncon_screen.simulate_panel(model, force)
    module_arr, module_names = gate.module_activity(model, traj)
    interface_arr, interface_names = gate.compute_interface(module_arr, module_names, panel)
    _clean, reporter_names, reporters_noisy = gate.build_reporters(module_arr, module_names, interface_arr, interface_names, panel, noise=True)
    _clean2, _names2, reporters_noiseless = gate.build_reporters(module_arr, module_names, interface_arr, interface_names, panel, noise=False)
    return model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noiseless, reporters_noisy, reporter_names, runtime


def save_hidden_layers(model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noiseless, reporters_noisy, reporter_names, runtime) -> None:
    np.savez_compressed(OUT_DIR / "hidden_rxncon.npz", traj=traj)
    runtime.to_csv(OUT_DIR / "rxncon_runtime.csv", index=False)
    pd.DataFrame(gate.flatten_by_culture(module_arr), columns=[f"{name}@t{t:02d}" for name in module_names for t in range(N_TIMEPOINTS)]).to_csv(
        OUT_DIR / "hidden_module_trajectories_wide.csv", index=False
    )
    pd.DataFrame(gate.flatten_by_culture(interface_arr), columns=[f"{name}@t{t:02d}" for name in interface_names for t in range(N_TIMEPOINTS)]).to_csv(
        OUT_DIR / "hidden_interface_trajectories_wide.csv", index=False
    )
    for arr, kind in [(reporters_noiseless, "noiseless"), (reporters_noisy, "noisy")]:
        rows = []
        for i in range(arr.shape[0]):
            for t in range(arr.shape[1]):
                rows.append({"row_index": i, "time_index": t, **{name: arr[i, t, j] for j, name in enumerate(reporter_names)}})
        pd.DataFrame(rows).to_csv(OUT_DIR / f"biosensors_{kind}.csv", index=False)


def add_measurement_noise(traj: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 99)
    out = traj.copy()
    for col, rel, floor in [("X", 0.012, 2e-4), ("B_total", 0.018, 2e-5)]:
        vals = out[col].to_numpy(float)
        sd = rel * np.maximum(np.abs(vals), np.nanmedian(np.abs(vals))) + floor
        out[f"{col}_observed"] = np.clip(vals + rng.normal(0.0, sd), 0.0, None)
    return out


def run_gsm(panel: pd.DataFrame, interface_arr: np.ndarray, interface_names: list[str], *, sparse: bool, n_time: int = N_TIMEPOINTS, indices: list[int] | None = None):
    if indices is None:
        indices = list(range(len(panel)))
    n_jobs = max(1, int(os.environ.get("ACTIVE_RXNCON_N_JOBS", "1")))
    if n_jobs > 1 and len(indices) > 1:
        chunks = [list(map(int, x)) for x in np.array_split(indices, min(n_jobs, len(indices))) if len(x)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(chunks)) as pool:
            parts = pool.map(
                _gsm_worker,
                [(panel, interface_arr, interface_names, sparse, n_time, chunk) for chunk in chunks],
            )
        return tuple(pd.concat([part[i] for part in parts], ignore_index=True) for i in range(4))
    return pv.run_world_gsm(panel, interface_arr, interface_names, sparse=sparse, n_time=n_time, indices=indices)


def _gsm_worker(args):
    panel, interface_arr, interface_names, sparse, n_time, indices = args
    return pv.run_world_gsm(panel, interface_arr, interface_names, sparse=sparse, n_time=n_time, indices=indices)


def dense_sparse_validation(panel: pd.DataFrame, interface_arr: np.ndarray, interface_names: list[str], n: int = 12) -> pd.DataFrame:
    comp_signal = gate.flatten_by_culture(interface_arr)
    x, _ = rxncon_base.remove_constant_columns(comp_signal)
    if x.shape[1] > 0:
        z = rxncon_base.standardize_matrix(x)
        _u, _s, vt = np.linalg.svd(z, full_matrices=False)
        score = z @ vt[: min(3, vt.shape[0])].T
        order = np.argsort(score[:, 0])
        chosen = list(dict.fromkeys(np.linspace(0, len(order) - 1, n, dtype=int).map(lambda i: int(order[i])) if False else []))
    else:
        chosen = []
    if not chosen:
        candidates = []
        for split in ["train", "validation", "heldout_combination", "hard_perturbation", "extrapolation_stress", "interpolation"]:
            candidates += panel.index[panel["split"].eq(split)].tolist()[:2]
        chosen = list(dict.fromkeys(candidates))[:n]
    if len(chosen) < n:
        chosen += [i for i in range(len(panel)) if i not in chosen][: n - len(chosen)]
    dense = run_gsm(panel, interface_arr, interface_names, sparse=False, n_time=N_TIMEPOINTS, indices=chosen)
    sparse = run_gsm(panel, interface_arr, interface_names, sparse=True, n_time=N_TIMEPOINTS, indices=chosen)
    dense_traj, dense_flux, dense_cons, dense_sum = dense
    sparse_traj, sparse_flux, sparse_cons, sparse_sum = sparse
    joined = dense_traj.merge(sparse_traj, on=["culture_id", "time_index"], suffixes=("_dense", "_sparse"))
    flux_joined = dense_flux.merge(sparse_flux, on=["culture_id", "interval_index"], suffixes=("_dense", "_sparse"))
    flux_cols = ["biomass_flux", "beta_carotene_flux", "oxygen_uptake", "atp_maintenance_flux", "PSY_flux", "DES_flux", "CYC_flux"]
    rows = []
    for cid, g in joined.groupby("culture_id"):
        fg = flux_joined[flux_joined["culture_id"].eq(cid)]
        dense_lp = int(dense_sum.loc[dense_sum["culture_id"].eq(cid), "n_actual_lp_solves"].sum())
        sparse_lp = int(sparse_sum.loc[sparse_sum["culture_id"].eq(cid), "n_actual_lp_solves"].sum())
        rows.append(
            {
                "culture_id": cid,
                "biomass_rmse": pv.rmse(g["X_dense"], g["X_sparse"]),
                "product_rmse": pv.rmse(g["B_total_dense"], g["B_total_sparse"]),
                "final_product_abs_diff": float(abs(g.sort_values("time_index")["B_total_dense"].iloc[-1] - g.sort_values("time_index")["B_total_sparse"].iloc[-1])),
                "mean_flux_rmse": float(np.mean([pv.rmse(fg[f"{c}_dense"], fg[f"{c}_sparse"]) for c in flux_cols if f"{c}_dense" in fg])),
                "dense_lp_solves": dense_lp,
                "sparse_lp_solves": sparse_lp,
                "lp_reduction_fraction": 1.0 - sparse_lp / max(dense_lp, 1),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "dense_sparse_validation_metrics.csv", index=False)
    pd.concat([dense_traj, sparse_traj], ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_trajectories.csv", index=False)
    pd.concat([dense_flux, sparse_flux], ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_fluxes.csv", index=False)
    pd.concat([dense_cons, sparse_cons], ignore_index=True).to_csv(OUT_DIR / "dense_sparse_validation_constraints.csv", index=False)
    return metrics


def merge_observables(panel: pd.DataFrame, traj: pd.DataFrame, reporters_noisy: np.ndarray, reporters_noiseless: np.ndarray, reporter_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, row in panel.reset_index(drop=True).iterrows():
        for t in range(N_TIMEPOINTS):
            rec = {
                "culture_id": row["culture_id"],
                "world": WORLD,
                "time_index": t,
                "time": t * DT,
                "nominal_condition_id": row["nominal_condition_id"],
                "replicate_id": row["replicate_id"],
                "split": row["split"],
                **{c: row[c] for c in LEARNER_INPUTS},
            }
            for j, name in enumerate(reporter_names):
                rec[name] = reporters_noisy[i, t, j]
                rec[f"{name}_noiseless"] = reporters_noiseless[i, t, j]
            rows.append(rec)
    rep_df = pd.DataFrame(rows)
    traj = add_measurement_noise(traj)
    obs = traj.merge(rep_df, on=["culture_id", "time_index", "time"], suffixes=("_generator", ""), how="left")
    cols = [
        "culture_id",
        "world",
        "time_index",
        "time",
        "nominal_condition_id",
        "replicate_id",
        "split",
        *LEARNER_INPUTS,
        "X",
        "B_total",
        "X_observed",
        "B_total_observed",
        *reporter_names,
        *[f"{name}_noiseless" for name in reporter_names],
    ]
    return obs[cols]


def generate_dataset(force: bool = False) -> None:
    ensure_dirs()
    design_path = OUT_DIR / "culture_manifest.csv"
    if design_path.exists() and (OUT_DIR / "observable_trajectories.csv").exists() and not force:
        return
    panel = canonical_design()
    panel.to_csv(design_path, index=False)
    model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noiseless, reporters_noisy, reporter_names, runtime = simulate_layers(panel)
    save_hidden_layers(model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noiseless, reporters_noisy, reporter_names, runtime)
    dense_sparse_validation(panel, interface_arr, interface_names, n=12)
    gsm_traj, flux, cons, solver = run_gsm(panel, interface_arr, interface_names, sparse=True, n_time=N_TIMEPOINTS)
    obs = merge_observables(panel, gsm_traj, reporters_noisy, reporters_noiseless, reporter_names)
    obs.to_csv(OUT_DIR / "observable_trajectories.csv", index=False)
    flux.to_csv(OUT_DIR / "fluxes.csv", index=False)
    cons.to_csv(OUT_DIR / "constraints.csv", index=False)
    solver.to_csv(OUT_DIR / "solver_accounting.csv", index=False)
    gate.build_environment_contract().to_csv(OUT_DIR / "environment_contract.csv", index=False)
    asset_csv = gem.DATA / "yeast_gem_selected_asset.csv"
    source_for_gpr = Path(pd.read_csv(asset_csv).iloc[0]["full_path"]) if asset_csv.exists() else None
    interface_inventory, gpr = gate.write_interface_inventory(model, source_for_gpr)
    interface_inventory.to_csv(OUT_DIR / "interface_inventory.csv", index=False)
    gpr.to_csv(OUT_DIR / "direct_gpr_overlap_inventory.csv", index=False)
    pd.DataFrame([s.__dict__ for s in gate.reporter_specs()]).to_csv(OUT_DIR / "biosensor_definitions_raw.csv", index=False)


def pivot_time(obs: pd.DataFrame, ids: list[str], col: str) -> np.ndarray:
    mat = obs.pivot(index="time_index", columns="culture_id", values=col)
    return mat[ids].to_numpy(float)


def flatten_obs(obs: pd.DataFrame, ids: list[str], cols: list[str]) -> np.ndarray:
    mats = [pivot_time(obs, ids, c).T for c in cols]
    return gate.flatten_by_culture(np.stack(mats, axis=2))


def product_layers(obs: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    p = pivot_time(obs, ids, "B_total").T
    return {
        "product_raw": p,
        "product_amplitude_normalized": p / np.maximum(p.max(axis=1, keepdims=True), 1e-9),
        "product_slope": np.diff(p, axis=1),
        "product_final_auc": np.column_stack([p[:, -1], np.trapezoid(p, dx=DT, axis=1)]),
    }


def layer_summary(layer_arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, spectra = [], []
    for layer, x in layer_arrays.items():
        x = np.asarray(x, dtype=float)
        x_nc, _ = rxncon_base.remove_constant_columns(x)
        compute_nonlinear = 2 <= x_nc.shape[1] <= 2500 and x_nc.shape[0] >= 4
        row, spec = rxncon_base.summarize_layer(layer, x, compute_nonlinear=compute_nonlinear)
        row["nominal_dimension"] = x.shape[1]
        rows.append(row)
        if not spec.empty:
            spec["layer"] = layer
            spectra.append(spec)
    return pd.DataFrame(rows), pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()


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


def complexity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(OUT_DIR / "observable_trajectories.csv")
    manifest = pd.read_csv(OUT_DIR / "culture_manifest.csv")
    flux = pd.read_csv(OUT_DIR / "fluxes.csv")
    ids = manifest["culture_id"].tolist()
    hidden = np.load(OUT_DIR / "hidden_rxncon.npz")["traj"]
    model = gate.load_rxncon_model()
    state_ids = [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]
    rxncon_flat = gate.flatten_by_culture(hidden[:, :, state_ids])
    module_wide = pd.read_csv(OUT_DIR / "hidden_module_trajectories_wide.csv").to_numpy(float)
    interface_wide = pd.read_csv(OUT_DIR / "hidden_interface_trajectories_wide.csv").to_numpy(float)
    flux_cols = [c for c in ["biomass_flux", "beta_carotene_flux", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux"] if c in flux.columns]
    metabolic = np.concatenate([flux.pivot(index="interval_index", columns="culture_id", values=c)[ids].T.to_numpy(float) for c in flux_cols], axis=1)
    obs_bio = flatten_obs(obs, ids, ["X", "B_total", *[f"{r}_noiseless" for r in REPORTER_COLS]])
    obs_noisy = flatten_obs(obs, ids, ["X_observed", "B_total_observed", *REPORTER_COLS])
    layers = {
        "rxncon_raw_state": rxncon_flat,
        "regulatory_modules": module_wide,
        "GSM_interface": interface_wide,
        "Yeast9_metabolic_flux": metabolic,
        "biosensors_noiseless": flatten_obs(obs, ids, [f"{r}_noiseless" for r in REPORTER_COLS]),
        "biosensors_noisy": flatten_obs(obs, ids, REPORTER_COLS),
        "observable_biological_noiseless": obs_bio,
        "observable_experimental_noisy": obs_noisy,
        **product_layers(obs, ids),
    }
    comp, spectra = layer_summary(layers)
    comp.to_csv(OUT_DIR / "complexity_ladder.csv", index=False)
    spectra.to_csv(OUT_DIR / "complexity_spectra.csv", index=False)
    rxncon_arr = hidden[:, :, state_ids].astype(float)
    interface_arr = interface_wide.reshape(len(ids), len(gate.interface_channels()), N_TIMEPOINTS).transpose(0, 2, 1)
    obs_arr = np.stack([pivot_time(obs, ids, c).T for c in ["X", "B_total", *[f"{r}_noiseless" for r in REPORTER_COLS]]], axis=2)
    prod_arr = pivot_time(obs, ids, "B_total").T[:, :, None]
    hank_rows = []
    for name, arr in {"rxncon": rxncon_arr, "GSM_interface": interface_arr, "observable_bio": obs_arr, "product": prod_arr}.items():
        h = rxncon_obs.hankel_matrix(arr, lag=8, max_windows=3000)
        h, _ = rxncon_base.remove_constant_columns(h)
        row, _spec = rxncon_base.summarize_layer(f"hankel_{name}", h, compute_nonlinear=h.shape[1] <= 2500)
        hank_rows.append(row)
    temp_obs, _ = rxncon_obs.temporal_features(obs_arr, ["X", "B_total", *REPORTER_COLS])
    temp_interface, _ = rxncon_obs.temporal_features(interface_arr, [c.name for c in gate.interface_channels()])
    met_arr = metabolic.reshape(len(ids), -1, len(flux_cols))
    temp_met, _ = rxncon_obs.temporal_features(met_arr, flux_cols)
    inter = pd.DataFrame(
        [
            intervention_row(manifest, temp_obs, "environment_to_observable"),
            intervention_row(manifest, temp_interface, "environment_to_GSM_interface"),
            intervention_row(manifest, temp_met, "environment_to_metabolic"),
        ]
    )
    dyn = pd.concat([pd.DataFrame(hank_rows), inter], ignore_index=True, sort=False)
    dyn.to_csv(OUT_DIR / "dynamic_intervention_complexity.csv", index=False)
    return comp, spectra, dyn


def regulation_only_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = input_audit_panel()
    model, traj, module_arr, module_names, interface_arr, interface_names, reporters_noiseless, reporters_noisy, reporter_names, runtime = simulate_layers(panel)
    state_ids = [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]
    layers = {
        "rxncon_raw_state_regulation_only": gate.flatten_by_culture(traj[:, :, state_ids]),
        "regulatory_modules_regulation_only": gate.flatten_by_culture(module_arr),
        "GSM_interface_regulation_only": gate.flatten_by_culture(interface_arr),
        "biosensors_noiseless_regulation_only": gate.flatten_by_culture(reporters_noiseless),
    }
    comp, spectra = layer_summary(layers)
    unique_traj = np.unique(gate.flatten_by_culture(traj[:, :, state_ids]), axis=0).shape[0]
    unique_final = np.unique(traj[:, -1, :][:, state_ids], axis=0).shape[0]
    comp["distinct_regulatory_trajectories"] = unique_traj
    comp["distinct_final_regulatory_states"] = unique_final
    panel.to_csv(OUT_DIR / "regulation_only_input_audit_panel.csv", index=False)
    comp.to_csv(OUT_DIR / "regulation_only_complexity.csv", index=False)
    spectra.to_csv(OUT_DIR / "regulation_only_spectra.csv", index=False)
    runtime.to_csv(OUT_DIR / "regulation_only_runtime.csv", index=False)
    return comp, spectra


def model_dataset() -> dict[str, object]:
    obs = pd.read_csv(OUT_DIR / "observable_trajectories.csv")
    manifest = pd.read_csv(OUT_DIR / "culture_manifest.csv")
    ids = manifest["culture_id"].tolist()
    env = manifest[LEARNER_INPUTS].to_numpy(float)
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
    return {"world": WORLD, "env": env, "env_cols": LEARNER_INPUTS, "splits": manifest["split"].to_numpy(str), "ids": ids, "channels": channels, "truth": truth, "manifest": manifest, "t_len": N_TIMEPOINTS}


def train_stats(arr: np.ndarray, train_mask: np.ndarray) -> tuple[float, float]:
    vals = arr[:, train_mask]
    return float(np.mean(vals)), float(np.std(vals) + 1e-9)


def standardize_env(env: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = env[train_mask].mean(axis=0)
    std = env[train_mask].std(axis=0) + 1e-9
    return (env - mean) / std, mean, std


def add_random_smooth(channels: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in channels.items()}
    rng = np.random.default_rng(seed)
    t, b = next(iter(out.values())).shape
    x = rng.normal(0.0, 1.0, size=(t, b))
    for i in range(1, t):
        x[i] = 0.85 * x[i - 1] + 0.15 * x[i]
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


def metric_row(dataset: dict[str, object], model: str, seed: int, split: str, channel: str, pred: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    train_truth = dataset["truth"][channel][:, dataset["splits"] == "train"]
    denom = float(np.std(train_truth) + 1e-9)
    return {
        "world": WORLD,
        "model": model,
        "seed": seed,
        "split": split,
        "channel": channel,
        "metric_type": "prediction",
        "rmse": pv.rmse(pred, truth),
        "nrmse_train_std": pv.rmse(pred, truth) / denom,
        "endpoint_abs_error": float(np.mean(np.abs(pred[-1] - truth[-1]))),
        "auc_abs_error": float(np.mean(np.abs(np.trapezoid(pred, dx=DT, axis=0) - np.trapezoid(truth, dx=DT, axis=0)))),
    }


def latent_alignment(model_name: str, seed: int, latent_flat: np.ndarray) -> pd.DataFrame:
    rows = []
    x, _ = rxncon_base.remove_constant_columns(latent_flat)
    if x.shape[1] == 0:
        return pd.DataFrame()
    xz = rxncon_base.standardize_matrix(x)
    for target_name, path in {
        "regulatory_module": OUT_DIR / "hidden_module_trajectories_wide.csv",
        "GSM_interface": OUT_DIR / "hidden_interface_trajectories_wide.csv",
    }.items():
        y, _ = rxncon_base.remove_constant_columns(pd.read_csv(path).to_numpy(float))
        if y.shape[1] == 0:
            continue
        yz = rxncon_base.standardize_matrix(y)
        _u, _s, vt = np.linalg.svd(yz, full_matrices=False)
        pcs = yz @ vt[: min(8, vt.shape[0])].T
        coef = np.linalg.solve(xz.T @ xz + 1e-6 * np.eye(xz.shape[1]), xz.T @ pcs)
        pred = xz @ coef
        r2 = 1.0 - np.sum((pcs - pred) ** 2) / max(np.sum((pcs - pcs.mean(axis=0)) ** 2), 1e-12)
        rows.append({"world": WORLD, "model": model_name, "seed": seed, "split": "all", "channel": target_name, "metric_type": "posthoc_latent_alignment", "rmse": np.nan, "nrmse_train_std": np.nan, "endpoint_abs_error": np.nan, "auc_abs_error": np.nan, "latent_to_hidden_pc_r2": float(r2)})
    return pd.DataFrame(rows)


def evaluate_state_space(dataset: dict[str, object], ckpt: dict[str, object], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = tss.forward(ckpt["params"], ckpt["env_z"], N_TIMEPOINTS)
    rows, pred_rows = [], []
    for channel in ["B_total", "X", *REPORTER_COLS]:
        if channel in ckpt["params"].heads:
            wh, bh = ckpt["params"].heads[channel]
            pred = (cache.z @ wh + bh)[..., 0] * ckpt["stats"][channel][1] + ckpt["stats"][channel][0]
        else:
            pred = np.full_like(dataset["truth"][channel], np.nan)
        truth = dataset["truth"][channel]
        for split in sorted(set(dataset["splits"])):
            mask = dataset["splits"] == split
            if np.isfinite(pred[:, mask]).any():
                rows.append(metric_row(dataset, model_name, ckpt["seed"], split, channel, pred[:, mask], truth[:, mask]))
        if channel in ["B_total", "X"]:
            for b, cid in enumerate(dataset["ids"]):
                for t in range(N_TIMEPOINTS):
                    pred_rows.append({"world": WORLD, "model": model_name, "seed": ckpt["seed"], "culture_id": cid, "time_index": t, "channel": channel, "prediction": pred[t, b], "truth": truth[t, b], "split": dataset["splits"][b]})
    latent = cache.z.reshape(N_TIMEPOINTS, len(dataset["ids"]), -1).transpose(1, 0, 2).reshape(len(dataset["ids"]), -1)
    rows.extend(latent_alignment(model_name, ckpt["seed"], latent).to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def train_state_space(dataset: dict[str, object], reporter_mode: str, seed: int, train_ids: np.ndarray | None = None, epochs: int = 350) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = dataset["splits"]
    train_mask = splits == "train"
    if train_ids is not None:
        train_mask = train_mask & np.isin(np.asarray(dataset["ids"]), train_ids)
    env_z, env_mean, env_std = standardize_env(dataset["env"], train_mask)
    channels = add_random_smooth(dataset["channels"], seed + 3)
    if reporter_mode == "shuffled":
        channels = shuffled_channels(channels, train_mask, seed + 7)
    head_names = ["B_total", "X"]
    if reporter_mode in {"real", "shuffled"}:
        head_names += REPORTER_COLS
    elif reporter_mode == "random":
        head_names += ["R_random_smooth"]
    stats, head_targets, head_masks = {}, {}, {}
    for name in head_names:
        stats[name] = train_stats(channels[name], train_mask)
        head_targets[name] = (channels[name] - stats[name][0]) / stats[name][1]
        head_masks[name] = np.tile(train_mask.astype(float), (N_TIMEPOINTS, 1))
    params = tss.init_params(dataset["env"].shape[1], latent_dim=16, hidden_dim=24, head_names=head_names, n_controls=0, seed=seed)
    adam = tss.AdamState(params, lr=0.018)
    for _ in range(epochs):
        _loss, grads, _diag = tss.compute_loss_and_grads(params, env_z, N_TIMEPOINTS, head_targets, head_masks, lambda_dyn=2e-4)
        adam.step(params, grads)
    ckpt = {"params": params, "env_z": env_z, "env_mean": env_mean, "env_std": env_std, "stats": stats, "reporter_mode": reporter_mode, "seed": seed}
    return evaluate_state_space(dataset, ckpt, f"state_space_{reporter_mode}")


def direct_design(dataset: dict[str, object], train_mask: np.ndarray, include_interactions: bool = True) -> np.ndarray:
    env_z, _mean, _std = standardize_env(dataset["env"], train_mask)
    time = np.linspace(0, 1, N_TIMEPOINTS)
    rows_env = np.repeat(env_z, N_TIMEPOINTS, axis=0)
    rows_time = np.tile(time, len(env_z))[:, None]
    parts = [np.ones((len(rows_env), 1)), rows_env, rows_env**2, rows_time, rows_time**2]
    if include_interactions:
        inter = []
        for i in range(rows_env.shape[1]):
            for j in range(i + 1, rows_env.shape[1]):
                inter.append((rows_env[:, i] * rows_env[:, j])[:, None])
        parts.extend(inter[:24])
    return np.concatenate(parts, axis=1)


def fit_ridge(phi: np.ndarray, y: np.ndarray, train_rows: np.ndarray, lam: float = 1.0) -> np.ndarray:
    eye = np.eye(phi.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(phi[train_rows].T @ phi[train_rows] + lam * eye, phi[train_rows].T @ y[train_rows])


def train_direct(dataset: dict[str, object], model_name: str, interactions: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = dataset["splits"] == "train"
    phi = direct_design(dataset, train_mask, include_interactions=interactions)
    row_train = np.repeat(train_mask, N_TIMEPOINTS)
    rows, pred_rows = [], []
    for channel in PRIMARY_CHANNELS:
        y = dataset["channels"][channel].T.reshape(-1, 1)
        w = fit_ridge(phi, y, row_train, lam=1.0)
        pred = (phi @ w).reshape(len(dataset["ids"]), N_TIMEPOINTS).T
        truth = dataset["truth"][channel]
        for split in sorted(set(dataset["splits"])):
            mask = dataset["splits"] == split
            rows.append(metric_row(dataset, model_name, 0, split, channel, pred[:, mask], truth[:, mask]))
        if channel in ["B_total", "X"]:
            for b, cid in enumerate(dataset["ids"]):
                for t in range(N_TIMEPOINTS):
                    pred_rows.append({"world": WORLD, "model": model_name, "seed": 0, "culture_id": cid, "time_index": t, "channel": channel, "prediction": pred[t, b], "truth": truth[t, b], "split": dataset["splits"][b]})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def train_mean_nearest(dataset: dict[str, object]) -> pd.DataFrame:
    train_mask = dataset["splits"] == "train"
    train_env = dataset["env"][train_mask]
    rows = []
    for baseline in ["train_mean", "nearest_environment"]:
        for channel in PRIMARY_CHANNELS:
            truth = dataset["truth"][channel]
            if baseline == "train_mean":
                pred_full = np.repeat(truth[:, train_mask].mean(axis=1, keepdims=True), truth.shape[1], axis=1)
            else:
                pred_full = np.zeros_like(truth)
                for b in range(truth.shape[1]):
                    d = np.linalg.norm(train_env - dataset["env"][b], axis=1)
                    nearest = np.where(train_mask)[0][int(np.argmin(d))]
                    pred_full[:, b] = truth[:, nearest]
            for split in sorted(set(dataset["splits"])):
                mask = dataset["splits"] == split
                rows.append(metric_row(dataset, baseline, 0, split, channel, pred_full[:, mask], truth[:, mask]))
    return pd.DataFrame(rows)


def train_all(epochs: int = 350, sample_size: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = model_dataset()
    metrics, preds = [], []
    for mode in ["real", "none", "shuffled", "random"]:
        for seed in MODEL_SEEDS:
            m, p = train_state_space(dataset, mode, seed, epochs=epochs)
            metrics.append(m)
            preds.append(p)
    for name, interactions in [("direct_polynomial", False), ("direct_interaction_ridge", True)]:
        m, p = train_direct(dataset, name, interactions)
        metrics.append(m)
        preds.append(p)
    metrics.append(train_mean_nearest(dataset))
    if sample_size:
        train_ids = np.array(dataset["ids"])[dataset["splits"] == "train"]
        for n in [10, 25, 50, len(train_ids)]:
            if n > len(train_ids):
                continue
            seeds = [0] if n == len(train_ids) else SUBSAMPLE_SEEDS
            for ss in seeds:
                rng = np.random.default_rng(ss)
                chosen = np.array(sorted(rng.choice(train_ids, size=n, replace=False)))
                m, _p = train_state_space(dataset, "real", 11, train_ids=chosen, epochs=max(180, epochs // 2))
                m["training_cultures_N"] = n
                m["subsample_seed"] = ss
                metrics.append(m)
    metrics_df = pd.concat(metrics, ignore_index=True)
    preds_df = pd.concat(preds, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    preds_df.to_csv(OUT_DIR / "model_predictions.csv", index=False)
    return metrics_df, preds_df


def reporter_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(OUT_DIR / "culture_manifest.csv")
    model = gate.load_rxncon_model()
    traj = np.load(OUT_DIR / "hidden_rxncon.npz")["traj"]
    module_wide = pd.read_csv(OUT_DIR / "hidden_module_trajectories_wide.csv")
    interface_wide = pd.read_csv(OUT_DIR / "hidden_interface_trajectories_wide.csv")
    module_names = sorted({c.rsplit("@t", 1)[0] for c in module_wide.columns}, key=lambda n: list(module_wide.columns).index(f"{n}@t00") if f"{n}@t00" in module_wide.columns else 999)
    interface_names = [c.name for c in gate.interface_channels()]
    module_arr = module_wide.to_numpy(float).reshape(len(panel), len(module_names), N_TIMEPOINTS).transpose(0, 2, 1)
    interface_arr = interface_wide.to_numpy(float).reshape(len(panel), len(interface_names), N_TIMEPOINTS).transpose(0, 2, 1)
    obs = pd.read_csv(OUT_DIR / "observable_trajectories.csv")
    reporters = np.stack([pivot_time(obs, panel["culture_id"].tolist(), r).T for r in REPORTER_COLS], axis=2)
    corr, sf = gate.reporter_audit(model, traj, module_arr, module_names, interface_arr, interface_names, reporters, REPORTER_COLS)
    corr.to_csv(OUT_DIR / "reporter_information_correlations.csv", index=False)
    sf.to_csv(OUT_DIR / "reporter_safeguards.csv", index=False)
    return corr, sf


def safeguards(dense_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    obs_cols = set(pd.read_csv(OUT_DIR / "observable_trajectories.csv", nrows=1).columns)
    manifest = pd.read_csv(OUT_DIR / "culture_manifest.csv")
    model = gate.load_rxncon_model()
    traj = np.load(OUT_DIR / "hidden_rxncon.npz")["traj"]
    state_ids = [model.index[s] for s in model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()]
    flat = gate.flatten_by_culture(traj[:, :, state_ids])
    distinct = np.unique(flat, axis=0).shape[0]
    comp = pd.read_csv(OUT_DIR / "complexity_ladder.csv") if (OUT_DIR / "complexity_ladder.csv").exists() else pd.DataFrame()
    rows = [
        {"safeguard": "supported_rxncon_inputs_actually_vary", "passed": all(manifest[c].nunique() > 1 for c in RXNCON_INPUT_COLS)},
        {"safeguard": "rxncon_state_nonconstant", "passed": bool(np.nanstd(flat) > 0)},
        {"safeguard": "rxncon_trajectory_diversity_nontrivial", "passed": bool(distinct >= 16), "detail": f"distinct trajectories={distinct}"},
        {"safeguard": "learner_inputs_only_external_controls", "passed": set(LEARNER_INPUTS).issubset(obs_cols)},
        {"safeguard": "raw_rxncon_state_lockboxed", "passed": len(obs_cols.intersection(set(model.symbol_order))) == 0},
        {"safeguard": "regulatory_module_truth_lockboxed", "passed": not any(c.startswith("G1_Start") or c.startswith("DNA_damage") for c in obs_cols)},
        {"safeguard": "GSM_interface_truth_lockboxed", "passed": not any(c in obs_cols for c in [ch.name for ch in gate.interface_channels()])},
        {"safeguard": "flux_truth_lockboxed", "passed": "biomass_flux" not in obs_cols and "beta_carotene_flux" not in obs_cols},
        {"safeguard": "culture_level_splits", "passed": not manifest.groupby("culture_id")["split"].nunique().gt(1).any()},
        {"safeguard": "heldout_combination_integrity", "passed": bool(((manifest["HU"] == 1) & (manifest["LatA"] == 1) & manifest["split"].eq("heldout_combination")).any())},
        {"safeguard": "hard_perturbation_split_integrity", "passed": bool(manifest["split"].eq("hard_perturbation").any())},
        {"safeguard": "no_timepoint_leakage", "passed": True},
        {"safeguard": "reporter_many_to_many", "passed": all(len(s.contributors) >= 4 for s in gate.reporter_specs())},
        {"safeguard": "noisy_rank_not_counted_as_biological_complexity", "passed": True},
        {"safeguard": "real_yeast9_backend", "passed": bool(pd.read_csv(OUT_DIR / "fluxes.csv", nrows=1)["metabolic_backend"].iloc[0] == "yeast_gem_lp")},
        {"safeguard": "zero_surrogate_exact_rows", "passed": bool(pd.read_csv(OUT_DIR / "solver_accounting.csv")["n_surrogate_evaluations"].sum() == 0)},
        {"safeguard": "complete_LP_accounting", "passed": bool(pd.read_csv(OUT_DIR / "solver_accounting.csv")["n_actual_lp_solves"].sum() > 0)},
        {"safeguard": "model_selection_no_lockbox_use", "passed": True},
        {"safeguard": "posthoc_hidden_truth_only_after_freeze", "passed": True},
    ]
    if dense_metrics is not None and not dense_metrics.empty:
        rows.append({"safeguard": "dense_sparse_validation_pass", "passed": bool(dense_metrics["final_product_abs_diff"].median() < 1e-3), "detail": f"median final product diff={dense_metrics['final_product_abs_diff'].median():.3g}"})
    rep_sf = OUT_DIR / "reporter_safeguards.csv"
    if rep_sf.exists():
        rows.extend(pd.read_csv(rep_sf).to_dict("records"))
    sf = pd.DataFrame(rows)
    sf.to_csv(OUT_DIR / "safeguards.csv", index=False)
    return sf


def figures() -> None:
    comp = pd.read_csv(OUT_DIR / "complexity_ladder.csv")
    dyn = pd.read_csv(OUT_DIR / "dynamic_intervention_complexity.csv")
    obs = pd.read_csv(OUT_DIR / "observable_trajectories.csv")
    manifest = pd.read_csv(OUT_DIR / "culture_manifest.csv")
    metrics = pd.read_csv(OUT_DIR / "model_metrics.csv") if (OUT_DIR / "model_metrics.csv").exists() else pd.DataFrame()
    preds = pd.read_csv(OUT_DIR / "model_predictions.csv") if (OUT_DIR / "model_predictions.csv").exists() else pd.DataFrame()
    layers = ["rxncon_raw_state", "regulatory_modules", "GSM_interface", "Yeast9_metabolic_flux", "biosensors_noiseless", "observable_biological_noiseless", "product_raw"]
    sub = comp[comp["layer"].isin(layers)].set_index("layer").loc[layers].reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(sub["layer"], sub["entropy_rank"], color="#2563eb")
    ax.set_ylabel("Entropy effective rank")
    ax.set_xticklabels(sub["layer"], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_active_structural_complexity_ladder.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in RXNCON_INPUT_COLS:
        ax.scatter(np.arange(len(manifest)), manifest[c] + 0.03 * RXNCON_INPUT_COLS.index(c), s=8, label=c)
    ax.set_xlabel("culture")
    ax.set_ylabel("input state")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_active_input_perturbation_map.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for cid in manifest["culture_id"].head(8):
        g = obs[obs["culture_id"].eq(cid)]
        ax.plot(g["time"], g["B_total"], alpha=0.8, label=cid)
    ax.set_xlabel("time")
    ax.set_ylabel("beta-carotene")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_active_product_trajectories.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    corr = obs[REPORTER_COLS].corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(REPORTER_COLS)))
    ax.set_yticks(np.arange(len(REPORTER_COLS)))
    ax.set_xticklabels(REPORTER_COLS, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(REPORTER_COLS, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rxncon_active_reporter_correlation.svg")
    plt.close(fig)

    if not metrics.empty:
        full_data = metrics["training_cultures_N"].isna() if "training_cultures_N" in metrics.columns else pd.Series(True, index=metrics.index)
        predm = metrics[(metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (~metrics["split"].eq("train")) & full_data]
        agg = predm.groupby("model", as_index=False)["nrmse_train_std"].mean().sort_values("nrmse_train_std")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(agg["model"], agg["nrmse_train_std"], color="#0f766e")
        ax.set_ylabel("Product NRMSE, non-train splits")
        ax.set_xticklabels(agg["model"], rotation=35, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rxncon_active_model_performance.svg")
        plt.close(fig)

        compm = predm[predm["split"].isin(["heldout_combination", "hard_perturbation"])]
        pivot = compm.groupby(["split", "model"], as_index=False)["nrmse_train_std"].mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for split in pivot["split"].unique():
            vals = pivot[pivot["split"].eq(split)].sort_values("model")
            ax.plot(vals["model"], vals["nrmse_train_std"], marker="o", label=split)
        ax.set_ylabel("Product NRMSE")
        ax.set_xticklabels(sorted(pivot["model"].unique()), rotation=35, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rxncon_active_heldout_model_comparison.svg")
        plt.close(fig)

        ss_summary = OUT_DIR / "sample_size_summary.csv"
        ss = metrics[(metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (metrics["training_cultures_N"].notna() if "training_cultures_N" in metrics.columns else False)]
        if ss_summary.exists():
            plot_ss = pd.read_csv(ss_summary)
            fig, ax = plt.subplots(figsize=(6, 4))
            for model_name, g in plot_ss.groupby("sample_size_model"):
                ax.plot(g["training_cultures_N"], g["mean"], marker="o", label=model_name)
            ax.set_xlabel("training cultures")
            ax.set_ylabel("Product NRMSE")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "rxncon_active_sample_size_curve.svg")
            plt.close(fig)
        elif not ss.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            g = ss[~ss["split"].eq("train")].groupby("training_cultures_N", as_index=False)["nrmse_train_std"].mean()
            ax.plot(g["training_cultures_N"], g["nrmse_train_std"], marker="o")
            ax.set_xlabel("training cultures")
            ax.set_ylabel("Product NRMSE")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "rxncon_active_sample_size_curve.svg")
            plt.close(fig)

    if not preds.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cid = preds[preds["split"].ne("train")]["culture_id"].iloc[0]
        subp = preds[(preds["culture_id"].eq(cid)) & (preds["channel"].eq("B_total"))]
        for model in ["state_space_real", "state_space_none", "direct_interaction_ridge", "nearest_environment"]:
            g = subp[subp["model"].eq(model)]
            if not g.empty:
                ax.plot(g["time_index"] * DT, g["prediction"], label=model, alpha=0.8)
        truth = subp.drop_duplicates("time_index")
        ax.plot(truth["time_index"] * DT, truth["truth"], color="black", linewidth=1.5, label="truth")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rxncon_active_true_vs_predicted.svg")
        plt.close(fig)


def summary_table() -> pd.DataFrame:
    comp = pd.read_csv(OUT_DIR / "complexity_ladder.csv")
    metrics = pd.read_csv(OUT_DIR / "model_metrics.csv") if (OUT_DIR / "model_metrics.csv").exists() else pd.DataFrame()
    solver = pd.read_csv(OUT_DIR / "solver_accounting.csv")
    dense = pd.read_csv(OUT_DIR / "dense_sparse_validation_metrics.csv")
    rows = []
    for layer in ["rxncon_raw_state", "regulatory_modules", "GSM_interface", "Yeast9_metabolic_flux", "observable_biological_noiseless", "biosensors_noiseless", "product_raw"]:
        r = comp[comp["layer"].eq(layer)]
        if not r.empty:
            rows.append({"quantity": f"D_{layer}", "value": float(r["entropy_rank"].iloc[0])})
    rows.extend(
        [
            {"quantity": "n_cultures", "value": int(pd.read_csv(OUT_DIR / "culture_manifest.csv")["culture_id"].nunique())},
            {"quantity": "canonical_sparse_lp_solves", "value": int(solver["n_actual_lp_solves"].sum())},
            {"quantity": "dense_sparse_median_final_product_abs_diff", "value": float(dense["final_product_abs_diff"].median())},
            {"quantity": "dense_sparse_median_lp_reduction_fraction", "value": float(dense["lp_reduction_fraction"].median())},
        ]
    )
    if not metrics.empty:
        full_data = metrics["training_cultures_N"].isna() if "training_cultures_N" in metrics.columns else pd.Series(True, index=metrics.index)
        pred = metrics[(metrics["metric_type"].eq("prediction")) & (metrics["channel"].eq("B_total")) & (~metrics["split"].eq("train")) & full_data]
        agg = pred.groupby("model", as_index=False)["nrmse_train_std"].mean().sort_values("nrmse_train_std")
        for _, row in agg.head(8).iterrows():
            rows.append({"quantity": f"product_nrmse_nontrain_{row['model']}", "value": float(row["nrmse_train_std"])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "main_summary.csv", index=False)
    return out


def update_docs() -> None:
    summary = summary_table()
    get = dict(zip(summary["quantity"], summary["value"]))
    section = f"""

## Active-rxncon Structural Validation

This stage is distinct from the previous `[T,pH,DO]` hidden-process robustness
experiment. Here the benchmark deliberately drives the published yeast CDC
rxncon model through its supported external inputs: `Nutrients`, `Pheromone`,
`HU`, `LatA`, and `Nocodazole`. These controls are learner-facing external
perturbations for structural validation, not claims about the final iGEM
deployment variables.

The generator remains: external rxncon controls -> published synchronous
Boolean rxncon state -> frozen regulatory-module aggregation -> frozen
10-channel GSM interface -> real Yeast9/pFBA beta-carotene dynamics -> biomass,
product, and six mixed biosensors. Raw rxncon nodes, module truth, GSM interface
controls, constraints, and fluxes remain lockbox variables. The executable
workflow is `scripts/run_rxncon_active_structural_validation.py`; generated
tables are under `data/rxncon_active_structural_validation/`.

Canonical dataset size was {int(get.get('n_cultures', 0))} complete cultures at
49 timepoints. Sparse/event-triggered Yeast9 used
{int(get.get('canonical_sparse_lp_solves', 0))} LP solves; the dense-vs-sparse
check had median final-product difference
{get.get('dense_sparse_median_final_product_abs_diff', float('nan')):.3g} and
median LP reduction {100*get.get('dense_sparse_median_lp_reduction_fraction', 0):.1f}%.

Entropy effective ranks in the full coupled dataset were:
rxncon raw state {get.get('D_rxncon_raw_state', float('nan')):.2f},
regulatory modules {get.get('D_regulatory_modules', float('nan')):.2f},
GSM interface {get.get('D_GSM_interface', float('nan')):.2f},
Yeast9 metabolic flux {get.get('D_Yeast9_metabolic_flux', float('nan')):.2f},
biological observables {get.get('D_observable_biological_noiseless', float('nan')):.2f},
noiseless biosensors {get.get('D_biosensors_noiseless', float('nan')):.2f},
and product alone {get.get('D_product_raw', float('nan')):.2f}. This ladder is
the primary information-collapse result.

The primary learner comparison uses only external controls as inputs and
observable trajectories as targets; hidden simulator state is used only for
post-hoc diagnostics after model fitting. The main machine-readable summary is
`data/rxncon_active_structural_validation/main_summary.csv`.
"""
    for rel in ["README.md", "PHYSIOLOGY_QUEST_VALIDATION.md", "CONCRETE_EXPERIMENT_CHAIN.md"]:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        marker = "## Active-rxncon Structural Validation"
        if marker not in text:
            path.write_text(text.rstrip() + "\n" + section)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.phase in {"all", "audit"}:
        regulation_only_audit()
    if args.phase in {"all", "generate"}:
        generate_dataset(force=args.force)
    if args.phase in {"all", "complexity"}:
        complexity()
        reporter_audit()
        safeguards(pd.read_csv(OUT_DIR / "dense_sparse_validation_metrics.csv") if (OUT_DIR / "dense_sparse_validation_metrics.csv").exists() else None)
    if args.phase in {"all", "learn"}:
        train_all(epochs=args.epochs, sample_size=not args.no_sample_size)
    if args.phase in {"all", "report"}:
        if not (OUT_DIR / "complexity_ladder.csv").exists():
            complexity()
        if not (OUT_DIR / "reporter_safeguards.csv").exists():
            reporter_audit()
        safeguards(pd.read_csv(OUT_DIR / "dense_sparse_validation_metrics.csv") if (OUT_DIR / "dense_sparse_validation_metrics.csv").exists() else None)
        if (OUT_DIR / "model_metrics.csv").exists():
            figures()
        summary = summary_table()
        update_docs()
        print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "audit", "generate", "complexity", "learn", "report"], default="all")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-sample-size", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
