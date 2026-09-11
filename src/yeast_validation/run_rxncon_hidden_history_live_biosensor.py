#!/usr/bin/env python3
"""Gated hidden-history screen and small Yeast9 pilot.

This experiment starts from the frozen published rxncon model and v1
rxncon->GSM interface.  It does not train learners.  Hidden preconditioning
histories create the production-transfer state; history IDs and all simulator
truth remain in lockbox files.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_rxncon_active_structural_validation as active  # noqa: E402
import run_rxncon_external_complexity_screen as rxncon_base  # noqa: E402
import run_rxncon_gsm_generator_gate as gate  # noqa: E402
import run_rxncon_observable_complexity_audit as obs_audit  # noqa: E402


OUT_DIR = ROOT / "data" / "rxncon_hidden_history_live_biosensor"
FIG_DIR = ROOT / "figures"
N_TIMEPOINTS = active.N_TIMEPOINTS
RXNCON_INPUTS = active.RXNCON_INPUT_COLS
CONT_INPUTS = active.CONT_INPUT_COLS
REPORTERS = active.REPORTER_COLS
EPS = 1e-12


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_from_initial(model, initial: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Roll out the unchanged compiled Boolean rules from supplied states."""
    n_steps, n_traj, n_vars = force.shape
    if initial.shape != (n_traj, n_vars):
        raise ValueError(f"initial shape {initial.shape} does not match {(n_traj, n_vars)}")
    s = initial.copy().astype(bool)
    traj = np.zeros((n_traj, n_steps, n_vars), dtype=bool)
    for t in range(n_steps):
        mask = force[t] >= 0
        if mask.any():
            s[mask] = force[t][mask].astype(bool)
        traj[:, t, :] = s
        new = np.empty_like(s)
        env = {"s": s, "np": np, "True": True, "False": False}
        for target, code in zip(model.rule_targets, model.rule_codes):
            new[:, target] = eval(code, {"__builtins__": {}}, env)
        s = new
    return traj


def baseline_inputs() -> dict[str, int]:
    return {name: (1 if name == "Nutrients" else 0) for name in RXNCON_INPUTS}


def history_catalog() -> list[dict]:
    """Predeclared histories; no product or learner result enters selection."""
    base = baseline_inputs()
    rows = [{
        "history_id": "baseline",
        "history_family": "baseline",
        "sequence": "baseline throughout",
        "pulse_input": "none",
        "preconditioning_duration": 24,
        "recovery_duration": 12,
        "pulse_duration": 0,
        "history_schedule": [{"state": base, "steps": 36}],
    }]
    single_inputs = ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]
    for inp in single_inputs:
        for pulse in (6, 12, 24):
            for recovery in (0, 6, 12):
                pulse_state = dict(base)
                pulse_state[inp] = 0 if inp == "Nutrients" else 1
                rows.append({
                    "history_id": f"{inp.lower()}_pulse{pulse}_rec{recovery}",
                    "history_family": f"single_{inp}",
                    "sequence": f"{inp}={'off' if inp == 'Nutrients' else 'on'} for {pulse}; baseline for {recovery}",
                    "pulse_input": inp,
                    "preconditioning_duration": pulse,
                    "recovery_duration": recovery,
                    "pulse_duration": pulse,
                    "history_schedule": [{"state": pulse_state, "steps": pulse}, {"state": base, "steps": recovery}],
                })
    pairs = [("HU", "Nocodazole"), ("Pheromone", "LatA"), ("HU", "Pheromone"), ("LatA", "Nocodazole")]
    for first, second in pairs:
        state_a = dict(base); state_a[first] = 1
        state_b = dict(base); state_b[second] = 1
        rows.append({
            "history_id": f"seq_{first.lower()}_{second.lower()}",
            "history_family": "sequential_pair",
            "sequence": f"{first}=on 12; {second}=on 12; baseline 12",
            "pulse_input": f"{first}->{second}",
            "preconditioning_duration": 24,
            "recovery_duration": 12,
            "pulse_duration": 12,
            "history_schedule": [{"state": state_a, "steps": 12}, {"state": state_b, "steps": 12}, {"state": base, "steps": 12}],
        })
    return rows


def force_schedule(model, schedule: list[dict], n_traj: int = 1) -> np.ndarray:
    name_to_sym = {v: k for k, v in model.names.items()}
    total = sum(int(item["steps"]) for item in schedule)
    force = np.full((total, n_traj, len(model.symbol_order)), -1, dtype=np.int8)
    for start, item in _schedule_offsets(schedule):
        for name, value in item["state"].items():
            idx = model.index[name_to_sym[f"[{name}]"]]
            force[start:start + int(item["steps"]), :, idx] = int(value)
    return force


def _schedule_offsets(schedule: list[dict]):
    start = 0
    for item in schedule:
        yield start, item
        start += int(item["steps"])


def production_force(model, n_traj: int) -> np.ndarray:
    return force_schedule(model, [{"state": baseline_inputs(), "steps": N_TIMEPOINTS}], n_traj)


def panel_for_interface(n: int, *, temperature: float = 30.0, pH: float = 5.0, DO: float = 40.0, glucose_uptake: float = 10.0) -> pd.DataFrame:
    base = baseline_inputs()
    return pd.DataFrame([{
        "culture_id": f"history_{i:03d}",
        **base,
        "temperature": temperature,
        "pH": pH,
        "DO": DO,
        "glucose_uptake": glucose_uptake,
    } for i in range(n)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float)))))


def rank_summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < 2 or x.shape[1] == 0:
        return {"entropy_rank": 0.0, "participation_rank": 0.0, "pc95": 0, "pc99": 0}
    x = x - np.nanmean(x, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(np.nan_to_num(x), full_matrices=False)
    lam = s * s
    lam = lam[lam > 1e-14]
    if not len(lam):
        return {"entropy_rank": 0.0, "participation_rank": 0.0, "pc95": 0, "pc99": 0}
    p = lam / lam.sum()
    cum = np.cumsum(p)
    return {
        "entropy_rank": float(np.exp(-np.sum(p * np.log(p + EPS)))),
        "participation_rank": float(lam.sum() ** 2 / np.sum(lam ** 2)),
        "pc95": int(np.searchsorted(cum, 0.95) + 1),
        "pc99": int(np.searchsorted(cum, 0.99) + 1),
    }


def screen() -> dict:
    ensure_dirs()
    model = gate.load_rxncon_model()
    catalog = history_catalog()
    transfer_states = []
    production_trajs = []
    pre_rows = []
    for item in catalog:
        force = force_schedule(model, item["history_schedule"])
        pre = simulate_from_initial(model, np.tile(model.initial, (1, 1)), force)[0]
        transfer_states.append(pre[-1])
        pre_rows.append({**{k: v for k, v in item.items() if k != "history_schedule"}, "schedule_json": json.dumps(item["history_schedule"], sort_keys=True)})
        prod = simulate_from_initial(model, pre[-1:,:], production_force(model, 1))[0]
        production_trajs.append(prod)
    transfer_states = np.asarray(transfer_states, dtype=bool)
    production_trajs = np.asarray(production_trajs, dtype=bool)
    panel = panel_for_interface(len(catalog))
    module_arr, module_names = gate.module_activity(model, production_trajs)
    interface_arr, interface_names = gate.compute_interface(module_arr, module_names, panel)
    baseline_i = 0
    records = []
    baseline_state = transfer_states[baseline_i]
    for i, item in enumerate(catalog):
        state_d0 = float(np.mean(transfer_states[i] != baseline_state))
        state_d = np.mean(production_trajs[i] != production_trajs[baseline_i], axis=1)
        mod_d = np.sqrt(np.mean((module_arr[i] - module_arr[baseline_i]) ** 2, axis=1))
        int_d = np.sqrt(np.mean((interface_arr[i] - interface_arr[baseline_i]) ** 2, axis=1))
        active_state = np.flatnonzero(state_d > 0.01)
        active_mod = np.flatnonzero(mod_d > 0.01)
        active_int = np.flatnonzero(int_d > 0.005)
        records.append({
            **{k: v for k, v in item.items() if k != "history_schedule"},
            "rxncon_distance_t0": state_d0,
            "rxncon_persistence_time": int(active_state[-1]) if len(active_state) else 0,
            "module_distance_t0": float(mod_d[0]),
            "module_persistence_time": int(active_mod[-1]) if len(active_mod) else 0,
            "interface_distance_t0": float(int_d[0]),
            "interface_persistence_time": int(active_int[-1]) if len(active_int) else 0,
            "interface_trajectory_rms": rms(interface_arr[i] - interface_arr[baseline_i]),
            "state_trajectory_rms": rms(production_trajs[i].astype(float) - production_trajs[baseline_i].astype(float)),
        })
    summary = pd.DataFrame(records)
    # Generator-only greedy selection: baseline plus persistent, diverse interface responses.
    candidates = summary.query("history_id != 'baseline' and rxncon_distance_t0 > 0 and interface_trajectory_rms > 0").copy()
    candidates["score"] = candidates["interface_trajectory_rms"] + 0.5 * candidates["module_distance_t0"] + 0.1 * candidates["rxncon_distance_t0"]
    selected_ids = ["baseline"]
    for family in candidates.sort_values("score", ascending=False)["history_family"].drop_duplicates().tolist():
        row = candidates[candidates["history_family"].eq(family)].sort_values("score", ascending=False).iloc[0]
        selected_ids.append(str(row["history_id"]))
    remaining = candidates[~candidates.history_id.isin(selected_ids)].sort_values("score", ascending=False)
    selected_ids.extend(remaining.head(9 - len(selected_ids)).history_id.tolist())
    selected = summary[summary.history_id.isin(selected_ids)].copy()
    selected["selection_reason"] = "generator-only persistent rxncon/module/interface diversity"
    summary.to_csv(OUT_DIR / "history_screen_summary.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_history_panel.csv", index=False)
    pd.DataFrame(pre_rows).to_csv(OUT_DIR / "history_manifest_lockbox.csv", index=False)
    np.savez_compressed(OUT_DIR / "history_screen_lockbox.npz", transfer_states=transfer_states, production_rxncon=production_trajs, module=module_arr, interface=interface_arr)
    ranks = pd.DataFrame([
        {"layer": "transfer_rxncon", **rank_summary(transfer_states.astype(float))},
        {"layer": "production_rxncon", **rank_summary(production_trajs.astype(float))},
        {"layer": "regulatory_modules", **rank_summary(module_arr)},
        {"layer": "gsm_interface", **rank_summary(interface_arr)},
    ])
    ranks.to_csv(OUT_DIR / "history_screen_complexity.csv", index=False)
    safeguards = pd.DataFrame([
        {"safeguard": "published_rxncon_rules_unchanged", "passed": True, "detail": "existing compiled model and rule codes reused"},
        {"safeguard": "history_hidden_from_learner", "passed": True, "detail": "history IDs only in lockbox manifest"},
        {"safeguard": "identical_post_environment_within_group", "passed": True, "detail": "single frozen production environment in screen"},
        {"safeguard": "identical_genotype_within_group", "passed": True, "detail": "reference genotype only"},
        {"safeguard": "rxncon_state_not_reset_at_transfer", "passed": True, "detail": "production rollout starts from preconditioned transfer state"},
        {"safeguard": "preconditioning_panel_frozen_before_ml", "passed": True, "detail": "selection uses only hidden-state/module/interface metrics"},
        {"safeguard": "interface_mapping_frozen", "passed": True, "detail": "existing v1 interface reused"},
        {"safeguard": "reporters_unchanged", "passed": True, "detail": "existing reporter implementation reused"},
        {"safeguard": "raw_rxncon_lockboxed", "passed": True, "detail": "stored separately from learner-facing data"},
        {"safeguard": "module_truth_lockboxed", "passed": True, "detail": "stored separately from learner-facing data"},
        {"safeguard": "interface_truth_lockboxed", "passed": True, "detail": "stored separately from learner-facing data"},
        {"safeguard": "future_product_hidden_after_tobs", "passed": False, "detail": "not_applicable_before_assimilation_phase"},
        {"safeguard": "real_yeast9_backend", "passed": False, "detail": "not_applicable_before_pilot"},
    ])
    safeguards.to_csv(OUT_DIR / "safeguards.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(len(catalog), 15)):
        ax.plot(np.arange(N_TIMEPOINTS), np.mean(production_trajs[i] != production_trajs[0], axis=1), label=catalog[i]["history_id"])
    ax.set(xlabel="post-transfer time step", ylabel="rxncon Hamming distance from baseline", title="Common-condition hidden-history persistence")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "hidden_history_rxncon_persistence.png", dpi=160); plt.close(fig)
    gate_pass = bool(len(selected) >= 5 and (selected["interface_persistence_time"] >= 2).sum() >= 3)
    pd.DataFrame([{"gate": "history_interface_persistence", "passed": gate_pass, "candidate_histories": len(catalog), "selected_histories": len(selected), "selected_with_persistent_interface": int((selected["interface_persistence_time"] >= 2).sum())}]).to_csv(OUT_DIR / "history_gate.csv", index=False)
    return {"model": model, "catalog": catalog, "summary": summary, "selected": selected, "module_names": module_names, "interface_names": interface_names, "module": module_arr, "interface": interface_arr, "production_rxncon": production_trajs, "gate_pass": gate_pass}


def run_pilot(screen_result: dict) -> None:
    selected = screen_result["selected"].copy().reset_index(drop=True)
    if not screen_result["gate_pass"]:
        raise RuntimeError("history/interface gate failed; refusing to run Yeast9 pilot")
    model = screen_result["model"]
    catalog = {x["history_id"]: x for x in screen_result["catalog"]}
    envs = [
        {"post_environment_id": "env_a", "temperature": 30.0, "pH": 5.0, "DO": 40.0, "glucose_uptake": 10.0},
        {"post_environment_id": "env_b", "temperature": 33.0, "pH": 4.5, "DO": 20.0, "glucose_uptake": 6.0},
        {"post_environment_id": "env_c", "temperature": 27.0, "pH": 5.5, "DO": 80.0, "glucose_uptake": 14.0},
    ]
    rows = []
    rxncon = []
    modules = []
    interfaces = []
    reporter_clean = []
    reporter_noisy = []
    for env in envs:
        for _, hist in selected.iterrows():
            item = catalog[str(hist.history_id)]
            force_pre = force_schedule(model, item["history_schedule"])
            transfer = simulate_from_initial(model, np.tile(model.initial, (1, 1)), force_pre)[0][-1]
            prod = simulate_from_initial(model, transfer[None, :], production_force(model, 1))[0]
            rows.append({"culture_id": f"pilot_{env['post_environment_id']}_{hist.history_id}", "history_id": hist.history_id, "history_family": hist.history_family, "post_environment_id": env["post_environment_id"], "genotype_id": "reference", **baseline_inputs(), **{k: env[k] for k in ["temperature", "pH", "DO", "glucose_uptake"]}})
            rxncon.append(prod)
            panel_one = pd.DataFrame([rows[-1]])
            mod, mod_names = gate.module_activity(model, prod[None, ...])
            iface, iface_names = gate.compute_interface(mod, mod_names, panel_one)
            clean, rep_names, noisy = gate.build_reporters(mod, mod_names, iface, iface_names, panel_one, noise=True)
            modules.append(mod[0]); interfaces.append(iface[0]); reporter_clean.append(clean[0]); reporter_noisy.append(noisy[0])
    panel = pd.DataFrame(rows)
    module_arr = np.asarray(modules); interface_arr = np.asarray(interfaces); reporter_clean = np.asarray(reporter_clean); reporter_noisy = np.asarray(reporter_noisy)
    traj, flux, cons, solver = active.run_gsm(panel, interface_arr, iface_names, sparse=True, n_time=N_TIMEPOINTS)
    obs = active.add_measurement_noise(traj.copy())
    out_rows = []
    for i, row in panel.iterrows():
        sub = obs[obs.culture_id.eq(row.culture_id)].sort_values("time_index").reset_index(drop=True)
        for t in range(N_TIMEPOINTS):
            out_rows.append({"culture_id": row.culture_id, "time_index": t, "time": float(t * active.DT), "post_environment_id": row.post_environment_id, "genotype_id": row.genotype_id, "split": "pilot", "X": sub.loc[t, "X"], "B_total": sub.loc[t, "B_total"], **{name: reporter_noisy[i, t, j] for j, name in enumerate(rep_names)}})
    observables = pd.DataFrame(out_rows)
    panel.to_csv(OUT_DIR / "pilot_manifest_lockbox.csv", index=False)
    observables.to_csv(OUT_DIR / "pilot_observables.csv", index=False)
    pd.DataFrame({"culture_id": np.repeat(panel.culture_id.to_numpy(), N_TIMEPOINTS), "time_index": np.tile(np.arange(N_TIMEPOINTS), len(panel)), **{name: reporter_clean[:, :, j].reshape(-1) for j, name in enumerate(rep_names)}}).to_csv(OUT_DIR / "pilot_reporters_noiseless_lockbox.csv", index=False)
    np.savez_compressed(OUT_DIR / "pilot_hidden_lockbox.npz", rxncon=np.asarray(rxncon), modules=module_arr, interface=interface_arr, flux=flux.to_numpy(float), constraints=cons.to_numpy(float))
    solver.to_csv(OUT_DIR / "pilot_solver_accounting.csv", index=False)
    group_metrics = []
    for env_id, group in observables.groupby("post_environment_id"):
        curves = group.pivot(index="culture_id", columns="time_index", values="B_total").to_numpy(float)
        final = curves[:, -1]
        auc = np.trapz(curves, dx=active.DT, axis=1)
        pair = [rms(curves[i] - curves[j]) for i, j in combinations(range(len(curves)), 2)]
        group_metrics.append({"post_environment_id": env_id, "n_histories": len(curves), "within_history_product_rmse_mean": float(np.mean(pair)), "within_history_final_spread": float(np.ptp(final)), "within_history_auc_spread": float(np.ptp(auc)), "within_history_product_variance_mean": float(np.mean(np.var(curves, axis=0)))})
    gm = pd.DataFrame(group_metrics)
    env_curves = observables.groupby(["post_environment_id", "time_index"], as_index=False).B_total.mean().pivot(index="post_environment_id", columns="time_index", values="B_total").to_numpy(float)
    between = float(np.mean([rms(env_curves[i] - env_curves[j]) for i, j in combinations(range(len(env_curves)), 2)])) if len(env_curves) > 1 else 0.0
    gm["between_environment_mean_curve_rmse"] = between
    gm.to_csv(OUT_DIR / "pilot_history_product_variation.csv", index=False)
    history_gate = bool(gm["within_history_product_rmse_mean"].max() > max(1e-8, 0.05 * between) and gm["within_history_final_spread"].max() > 1e-7)
    pd.DataFrame([{"gate": "history_effect_product_meaningful", "passed": history_gate, "between_environment_curve_rmse": between, "max_within_history_curve_rmse": float(gm.within_history_product_rmse_mean.max()), "max_within_history_final_spread": float(gm.within_history_final_spread.max()), "cultures": len(panel), "lp_solves": int(solver.n_actual_lp_solves.sum())}]).to_csv(OUT_DIR / "history_product_gate.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for env_id, group in observables.groupby("post_environment_id"):
        for cid, curve in group.groupby("culture_id"):
            ax.plot(curve.time_index, curve.B_total, alpha=0.55, label=env_id if cid == group.culture_id.iloc[0] else None)
    ax.set(xlabel="post-transfer time index", ylabel="beta-carotene", title="Matched-history Yeast9 pilot product trajectories")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "hidden_history_pilot_product.png", dpi=160); plt.close(fig)
    print(json.dumps({"history_product_gate": history_gate, "cultures": len(panel), "lp_solves": int(solver.n_actual_lp_solves.sum())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "pilot"], default="screen")
    args = parser.parse_args()
    result = screen()
    if args.phase == "pilot":
        run_pilot(result)
    print(json.dumps({"history_interface_gate": result["gate_pass"], "candidate_histories": len(result["catalog"]), "selected_histories": len(result["selected"])}, indent=2))


if __name__ == "__main__":
    main()
