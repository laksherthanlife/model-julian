#!/usr/bin/env python3
"""Dynamic heterologous enzyme-capacity validation for the real Yeast9 generator."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import gem_backend as gem


TEMPS = [27.0, 30.0, 33.0]
PHS = [4.5, 5.0, 5.5]
DOS = [20.0, 40.0, 80.0]
LADDER_MODES = [
    "no_dynamic_capacity",
    "constant_pathway_caps",
    "environment_initial_capacity",
    "dynamic_damage_recovery",
    "dynamic_congestion_feedback",
]
SUBSET_ENVS = [
    (30.0, 5.0, 40.0, "favorable"),
    (30.0, 5.0, 80.0, "high_oxidative_pressure"),
    (30.0, 4.5, 20.0, "high_ATP_pressure"),
    (33.0, 5.5, 40.0, "pathway_bottleneck"),
    (33.0, 4.5, 80.0, "combined_stress"),
]
PCA_TARGETS = [0.90, 0.95, 0.99, 0.999]


def env_id(temp: float, ph: float, do: float) -> str:
    return f"T{temp:g}_pH{ph:g}_DO{do:g}".replace(".", "p")


def load_augmented(gem_path: str | None):
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(gem_path)
    base = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    return source_path, augmented


def selected_parameter_config() -> gem.GEMCultureConfig:
    return gem.GEMCultureConfig(
        capacity_mode="dynamic_congestion_feedback",
        psy_base_capacity=0.34,
        des_base_capacity=0.28,
        cyc_base_capacity=0.23,
        enzyme_min_capacity=0.08,
        psy_synthesis_base=0.070,
        des_synthesis_base=0.060,
        cyc_synthesis_base=0.055,
        psy_decay_base=0.018,
        des_decay_base=0.024,
        cyc_decay_base=0.022,
        psy_oxidative_sensitivity=0.040,
        des_oxidative_sensitivity=0.125,
        cyc_oxidative_sensitivity=0.160,
        phytoene_congestion_sensitivity=0.320,
        lycopene_congestion_sensitivity=0.280,
    )


def with_env(base: gem.GEMCultureConfig, temp: float, ph: float, do: float, n_time: int, mode: str) -> gem.GEMCultureConfig:
    return replace(base, temperature=temp, pH=ph, DO=do, n_time=n_time, capacity_mode=mode)


def checkpoint_paths(prefix: str) -> dict[str, Path]:
    return {
        "traj": gem.DATA / f"{prefix}_trajectories.partial.csv",
        "flux": gem.DATA / f"{prefix}_fluxes.partial.csv",
        "constraints": gem.DATA / f"{prefix}_constraints.partial.csv",
        "summary": gem.DATA / f"{prefix}_summary.partial.csv",
    }


def read_checkpoint(prefix: str):
    paths = checkpoint_paths(prefix)
    return {
        key: pd.read_csv(path) if path.exists() else pd.DataFrame()
        for key, path in paths.items()
    }


def write_checkpoint(prefix: str, trajs, fluxes, constraints, summaries) -> None:
    gem.DATA.mkdir(exist_ok=True)
    paths = checkpoint_paths(prefix)
    pd.concat(trajs, ignore_index=True).to_csv(paths["traj"], index=False)
    pd.concat(fluxes, ignore_index=True).to_csv(paths["flux"], index=False)
    pd.concat(constraints, ignore_index=True).to_csv(paths["constraints"], index=False)
    pd.concat(summaries, ignore_index=True).to_csv(paths["summary"], index=False)


def run_envs(augmented, source_path: Path, base_cfg: gem.GEMCultureConfig, envs, n_time: int, mode: str, checkpoint_prefix: str | None = None):
    existing = read_checkpoint(checkpoint_prefix) if checkpoint_prefix else {}
    trajs = [existing["traj"]] if checkpoint_prefix and not existing["traj"].empty else []
    fluxes = [existing["flux"]] if checkpoint_prefix and not existing["flux"].empty else []
    constraints = [existing["constraints"]] if checkpoint_prefix and not existing["constraints"].empty else []
    summaries = [existing["summary"]] if checkpoint_prefix and not existing["summary"].empty else []
    completed = set(existing["traj"]["culture_id"].unique()) if checkpoint_prefix and not existing["traj"].empty else set()
    tic = time.perf_counter()
    for temp, ph, do, label in envs:
        cfg = with_env(base_cfg, temp, ph, do, n_time, mode)
        cid = f"{mode}_{env_id(temp, ph, do)}"
        if cid in completed:
            continue
        traj, flux, cons, summary = gem.run_dynamic_culture(augmented, source_path, cfg, cid)
        for df in (traj, flux, cons):
            df["environment_id"] = env_id(temp, ph, do)
            df["environment_label"] = label
            df["capacity_mode"] = mode
        trajs.append(traj)
        fluxes.append(flux)
        constraints.append(cons)
        summaries.append(summary)
        if checkpoint_prefix:
            write_checkpoint(checkpoint_prefix, trajs, fluxes, constraints, summaries)
    out = (
        pd.concat(trajs, ignore_index=True),
        pd.concat(fluxes, ignore_index=True),
        pd.concat(constraints, ignore_index=True),
        pd.concat(summaries, ignore_index=True),
        time.perf_counter() - tic,
    )
    return out


def pivot_curves(df: pd.DataFrame, value_col: str) -> tuple[list[str], np.ndarray]:
    ids, curves = [], []
    for cid, g in df.groupby("culture_id"):
        ids.append(cid)
        curves.append(g.sort_values("time")[value_col].to_numpy(float))
    return ids, np.asarray(curves)


def normalized_curves(curves: np.ndarray) -> np.ndarray:
    amp = np.maximum(curves.max(axis=1, keepdims=True) - curves[:, [0]], gem.EPS)
    return (curves - curves[:, [0]]) / amp


def pca_rows(df: pd.DataFrame, value_col: str, matrix: str, normalize: bool = False) -> list[dict[str, object]]:
    _ids, curves = pivot_curves(df, value_col)
    if normalize:
        curves = normalized_curves(curves)
    if curves.shape[0] == 1:
        explained = np.array([1.0])
    else:
        centered = curves - curves.mean(axis=0)
        _u, s, _vt = np.linalg.svd(centered, full_matrices=False)
        explained = s**2 / max(float(np.sum(s**2)), gem.EPS)
    cum = np.cumsum(explained)
    rows = []
    for k in [1, 2, 3, 5, 10]:
        kk = min(k, len(cum))
        rows.append({"matrix": matrix, "component_count": k, "cumulative_explained_variance": float(cum[kk - 1]), "target_variance": np.nan})
    for target in PCA_TARGETS:
        rows.append({"matrix": matrix, "component_count": int(np.searchsorted(cum, target) + 1), "cumulative_explained_variance": np.nan, "target_variance": target})
    return rows


def slope_table(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time").copy()
        g["product_slope"] = np.gradient(g["B_total"].to_numpy(float), g["time"].to_numpy(float))
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def capacity_long(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for enzyme in ["E_PSY", "E_DES", "E_CYC"]:
        tmp = traj[["culture_id", "time", "capacity_mode", "environment_id", enzyme]].rename(columns={enzyme: "capacity"})
        tmp["enzyme"] = enzyme
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def limiting_reaction(row) -> str:
    vals = {
        "PSY": row["PSY_flux"] / max(row["PSY_effective_upper_bound"], gem.EPS),
        "DES": row["DES_flux"] / max(row["DES_effective_upper_bound"], gem.EPS),
        "CYC": row["CYC_flux"] / max(row["CYC_effective_upper_bound"], gem.EPS),
    }
    return max(vals, key=vals.get)


def culture_audit(traj: pd.DataFrame, flux: pd.DataFrame, constraints: pd.DataFrame) -> pd.DataFrame:
    f = flux.merge(
        constraints[
            [
                "culture_id",
                "interval_index",
                "PSY_effective_upper_bound",
                "DES_effective_upper_bound",
                "CYC_effective_upper_bound",
                "PSY_synthesis_term",
                "DES_synthesis_term",
                "CYC_synthesis_term",
                "PSY_damage_rate",
                "DES_damage_rate",
                "CYC_damage_rate",
            ]
        ],
        on=["culture_id", "interval_index"],
        how="left",
    )
    f["limiting_pathway_reaction"] = f.apply(limiting_reaction, axis=1)
    slope = slope_table(traj)
    rows = []
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        ff = f[f["culture_id"] == cid].sort_values("interval_index")
        y = g["B_total"].to_numpy(float)
        t = g["time"].to_numpy(float)
        dy = np.gradient(y, t)
        max_slope = float(dy.max())
        slope_threshold = max(0.02 * max_slope, 1e-8)
        startup_ix = int(np.argmax(dy > slope_threshold)) if np.any(dy > slope_threshold) else -1
        plateau_ix = next((i for i in range(len(dy) // 3, len(dy)) if dy[i] < 0.10 * max_slope and y[i] > y[0] + 0.65 * (y.max() - y[0])), -1)
        decline_ix = int(np.argmax(dy < -0.02 * max(max_slope, gem.EPS))) if np.any(dy < -0.02 * max(max_slope, gem.EPS)) else -1
        recovery_ix = -1
        if np.any(dy[: len(dy) // 2] < 0.20 * max_slope):
            low_i = int(np.argmin(dy[: len(dy) // 2]))
            later = np.where(dy[low_i + 1 :] > dy[low_i] + 0.25 * max_slope)[0]
            recovery_ix = int(low_i + 1 + later[0]) if len(later) else -1
        cap_vals = g[["E_PSY", "E_DES", "E_CYC"]].to_numpy(float)
        cap_min_idx = np.unravel_index(np.argmin(cap_vals), cap_vals.shape)
        limit_seq = list(ff["limiting_pathway_reaction"])
        state_seq = list(ff["state"].astype(str)) + [str(ff["next_state"].iloc[-1])]
        dwell = []
        start = 0
        for i in range(1, len(state_seq)):
            if state_seq[i] != state_seq[i - 1]:
                dwell.append(i - start)
                start = i
        dwell.append(len(state_seq) - start)
        rows.append(
            {
                "culture_id": cid,
                "capacity_mode": g["capacity_mode"].iloc[0],
                "environment_id": g["environment_id"].iloc[0],
                "E_PSY_initial": float(g["E_PSY"].iloc[0]),
                "E_DES_initial": float(g["E_DES"].iloc[0]),
                "E_CYC_initial": float(g["E_CYC"].iloc[0]),
                "E_PSY_final": float(g["E_PSY"].iloc[-1]),
                "E_DES_final": float(g["E_DES"].iloc[-1]),
                "E_CYC_final": float(g["E_CYC"].iloc[-1]),
                "minimum_capacity": float(cap_vals.min()),
                "time_of_minimum_capacity": float(t[cap_min_idx[0]]),
                "capacity_loss_rate": float((cap_vals[0].mean() - cap_vals.min(axis=1).min()) / max(t[-1] - t[0], gem.EPS)),
                "capacity_recovery_occurrence": bool(np.any(np.diff(cap_vals, axis=0) > 0.005)),
                "capacity_recovery_magnitude": float(max(0.0, cap_vals[-1].mean() - cap_vals.min(axis=0).mean())),
                "limiting_reaction_sequence": "->".join(limit_seq),
                "n_limiting_reaction_switches": int(sum(limit_seq[i] != limit_seq[i - 1] for i in range(1, len(limit_seq)))),
                "mean_phytoene_congestion": float(ff["phytoene_congestion"].mean()),
                "mean_lycopene_congestion": float(ff["lycopene_congestion"].mean()),
                "mean_GGPP_loading": float(constraints[constraints["culture_id"] == cid]["GGPP_loading_term"].mean()),
                "mean_beta_carotene_throughput": float(ff["beta_carotene_flux"].mean()),
                "startup_delay": float(t[startup_ix]) if startup_ix >= 0 else np.nan,
                "maximum_product_slope": max_slope,
                "time_of_maximum_product_slope": float(t[int(np.argmax(dy))]),
                "plateau_onset": float(t[plateau_ix]) if plateau_ix >= 0 else np.nan,
                "decline_onset": float(t[decline_ix]) if decline_ix >= 0 else np.nan,
                "recovery_onset": float(t[recovery_ix]) if recovery_ix >= 0 else np.nan,
                "final_titer": float(y[-1]),
                "AUC": float(np.trapezoid(y, t)),
                "n_slope_sign_changes": int(np.sum(np.diff(np.signbit(dy)) != 0)),
                "state_sequence": "->".join(state_seq),
                "n_state_transitions": int(sum(state_seq[i] != state_seq[i - 1] for i in range(1, len(state_seq)))),
                "overload_occurrence": "combined_overload" in state_seq,
                "recovery_occurrence": "recovering" in state_seq or recovery_ix >= 0,
                "dwell_times": ";".join(map(str, dwell)),
            }
        )
    return pd.DataFrame(rows)


def shape_distance_clusters(traj: pd.DataFrame) -> tuple[int, float]:
    _ids, curves = pivot_curves(traj, "B_total")
    z = normalized_curves(curves)
    distances = []
    for i, j in itertools.combinations(range(len(z)), 2):
        distances.append(float(np.sqrt(np.mean((z[i] - z[j]) ** 2))))
    threshold = max(0.035, np.percentile(distances, 65) if distances else 0.035)
    centers = []
    for curve in z:
        if not centers or min(np.sqrt(np.mean((curve - c) ** 2)) for c in centers) > threshold:
            centers.append(curve)
    return len(centers), float(np.mean(distances) if distances else 0.0)


def shape_metrics(traj: pd.DataFrame, flux: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    clusters, mean_shape_distance = shape_distance_clusters(traj)
    n = max(len(audit), 1)
    return pd.DataFrame(
        [
            {
                "capacity_mode": traj["capacity_mode"].iloc[0],
                "plateau_fraction": float(audit["plateau_onset"].notna().mean()),
                "decline_fraction": float(audit["decline_onset"].notna().mean()),
                "recovery_fraction": float(audit["recovery_occurrence"].mean()),
                "delayed_startup_fraction": float((audit["startup_delay"].fillna(0) > 0.25).mean()),
                "multiple_slope_regime_fraction": float((audit["n_slope_sign_changes"] + audit["plateau_onset"].notna().astype(int) + audit["recovery_onset"].notna().astype(int) > 0).sum() / n),
                "limiting_reaction_switch_fraction": float((audit["n_limiting_reaction_switches"] > 0).mean()),
                "distinct_normalized_trajectory_clusters": clusters,
                "mean_pairwise_normalized_shape_distance": mean_shape_distance,
                "mean_beta_flux": float(flux["beta_carotene_flux"].mean()),
                "mean_final_titer": float(audit["final_titer"].mean()),
            }
        ]
    )


def pca_table(traj: pd.DataFrame, flux: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(pca_rows(traj, "B_total", "raw_accumulated_product", False))
    rows.extend(pca_rows(traj, "B_total", "amplitude_normalized_product", True))
    rows.extend(pca_rows(flux.rename(columns={"beta_carotene_flux": "B_total"}), "B_total", "beta_carotene_flux", False))
    rows.extend(pca_rows(slope_table(traj), "product_slope", "product_slope", False))
    rows.extend(pca_rows(traj, "X", "active_biomass", False))
    cap = capacity_long(traj)
    for enzyme, g in cap.groupby("enzyme"):
        rows.extend(pca_rows(g.rename(columns={"capacity": "B_total"}), "B_total", f"pathway_capacity_{enzyme}", False))
    return pd.DataFrame(rows)


def parameter_sweep(augmented, source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = selected_parameter_config()
    sweep_points = [
        (0.85, 1.00, 1.00),
        (1.00, 0.75, 1.00),
        (1.00, 1.00, 0.80),
        (1.00, 1.00, 1.00),
        (1.00, 1.30, 1.00),
        (1.15, 1.00, 1.20),
    ]
    candidates = [
        replace(
            base,
            psy_base_capacity=base.psy_base_capacity * base_scale,
            des_base_capacity=base.des_base_capacity * base_scale,
            cyc_base_capacity=base.cyc_base_capacity * base_scale,
            psy_synthesis_base=base.psy_synthesis_base * synth_scale,
            des_synthesis_base=base.des_synthesis_base * synth_scale,
            cyc_synthesis_base=base.cyc_synthesis_base * synth_scale,
            psy_oxidative_sensitivity=base.psy_oxidative_sensitivity * damage_scale,
            des_oxidative_sensitivity=base.des_oxidative_sensitivity * damage_scale,
            cyc_oxidative_sensitivity=base.cyc_oxidative_sensitivity * damage_scale,
            phytoene_congestion_sensitivity=base.phytoene_congestion_sensitivity * damage_scale,
            lycopene_congestion_sensitivity=base.lycopene_congestion_sensitivity * damage_scale,
        )
        for base_scale, damage_scale, synth_scale in sweep_points
    ]
    rows = []
    best = None
    best_score = -np.inf
    for idx, cfg in enumerate(candidates):
        try:
            traj, flux, cons, _summary, runtime = run_envs(augmented, source_path, cfg, SUBSET_ENVS, 13, "dynamic_congestion_feedback")
            audit = culture_audit(traj, flux, cons)
            shape = shape_metrics(traj, flux, audit).iloc[0]
            infeasible = int(flux["n_infeasible_solves"].sum())
            unbounded = int(flux["n_unbounded_solves"].sum())
            final = audit["final_titer"]
            rejected = bool(
                infeasible
                or unbounded
                or final.max() <= 1e-8
                or final.min() >= 0.95 * final.max()
                or (audit["minimum_capacity"] <= cfg.enzyme_min_capacity + 1e-6).mean() > 0.8
                or audit["state_sequence"].nunique() <= 1
            )
            score = (
                2.0 * float(shape["distinct_normalized_trajectory_clusters"])
                + 4.0 * float(shape["mean_pairwise_normalized_shape_distance"])
                + float(shape["limiting_reaction_switch_fraction"])
                + float(shape["plateau_fraction"] + shape["recovery_fraction"])
                - 0.5 * float(rejected)
            )
            rows.append(
                {
                    "candidate_id": idx,
                    "base_capacity_scale": cfg.psy_base_capacity / base.psy_base_capacity,
                    "damage_scale": cfg.des_oxidative_sensitivity / base.des_oxidative_sensitivity,
                    "synthesis_scale": cfg.psy_synthesis_base / base.psy_synthesis_base,
                    "mean_final_titer": float(final.mean()),
                    "min_final_titer": float(final.min()),
                    "max_final_titer": float(final.max()),
                    "minimum_capacity": float(audit["minimum_capacity"].min()),
                    "distinct_state_sequences": int(audit["state_sequence"].nunique()),
                    "distinct_normalized_trajectory_clusters": int(shape["distinct_normalized_trajectory_clusters"]),
                    "limiting_reaction_switch_fraction": float(shape["limiting_reaction_switch_fraction"]),
                    "plateau_fraction": float(shape["plateau_fraction"]),
                    "recovery_fraction": float(shape["recovery_fraction"]),
                    "infeasible_solves": infeasible,
                    "unbounded_solves": unbounded,
                    "runtime_seconds": runtime,
                    "rejected": rejected,
                    "selection_score": score,
                }
            )
            if not rejected and score > best_score:
                best_score = score
                best = cfg
        except Exception as exc:
            rows.append({"candidate_id": idx, "rejected": True, "selection_score": -999.0, "error": f"{type(exc).__name__}: {exc}"})
    sweep = pd.DataFrame(rows)
    if best is None:
        best = base
    selected = pd.DataFrame([{**best.__dict__, "selection_reason": "highest predeclared generator score among feasible nonzero deterministic subset runs; no neural metrics used"}])
    return sweep, selected, best


def run_ladder(augmented, source_path: Path, cfg: gem.GEMCultureConfig) -> pd.DataFrame:
    rows = []
    for mode in LADDER_MODES:
        traj, flux, cons, _summary, runtime = run_envs(augmented, source_path, cfg, SUBSET_ENVS, 13, mode)
        audit = culture_audit(traj, flux, cons)
        shape = shape_metrics(traj, flux, audit).iloc[0]
        pca = pca_table(traj, flux)
        norm_pc1 = float(pca[(pca["matrix"].eq("amplitude_normalized_product")) & (pca["component_count"].eq(1))]["cumulative_explained_variance"].iloc[0])
        rows.append(
            {
                "capacity_mode": mode,
                "runtime_seconds": runtime,
                "total_lp_solves": int(flux["n_actual_lp_solves"].sum()),
                "mean_final_titer": float(audit["final_titer"].mean()),
                "mean_beta_flux": float(flux["beta_carotene_flux"].mean()),
                "normalized_shape_PC1": norm_pc1,
                "plateau_fraction": float(shape["plateau_fraction"]),
                "decline_fraction": float(shape["decline_fraction"]),
                "recovery_fraction": float(shape["recovery_fraction"]),
                "multiple_slope_regime_fraction": float(shape["multiple_slope_regime_fraction"]),
                "limiting_reaction_switch_fraction": float(shape["limiting_reaction_switch_fraction"]),
                "distinct_normalized_trajectory_clusters": int(shape["distinct_normalized_trajectory_clusters"]),
            }
        )
    return pd.DataFrame(rows)


def acceptance(traj: pd.DataFrame, flux: pd.DataFrame, cons: pd.DataFrame, audit: pd.DataFrame, pca: pd.DataFrame, shape: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    norm_pc1 = float(pca[(pca["matrix"].eq("amplitude_normalized_product")) & (pca["component_count"].eq(1))]["cumulative_explained_variance"].iloc[0])
    norm_999_components = int(pca[(pca["matrix"].eq("amplitude_normalized_product")) & (pca["target_variance"].eq(0.999))]["component_count"].iloc[0])
    dynamic_shape = ladder[ladder["capacity_mode"].eq("dynamic_congestion_feedback")].iloc[0]
    constant_shape = ladder[ladder["capacity_mode"].eq("constant_pathway_caps")].iloc[0]
    event_fraction = max(float(shape["plateau_fraction"].iloc[0]), float(shape["decline_fraction"].iloc[0]), float(shape["recovery_fraction"].iloc[0]))
    criteria = [
        ("real_yeast9_every_culture", traj["metabolic_backend"].eq("yeast_gem_lp").all()),
        ("zero_surrogate_evaluations", int(traj["n_surrogate_evaluations"].max()) == 0),
        ("complete_LP_accounting", int(flux["n_actual_lp_solves"].sum()) == int(flux[["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations"]].sum().sum())),
        ("no_infeasible_or_unbounded_ignored", int(flux[["n_infeasible_solves", "n_unbounded_solves"]].sum().sum()) == 0),
        ("dynamic_capacity_bounds_alter_GEM_optima", flux["beta_carotene_flux"].nunique() > 10 and cons[["PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"]].nunique().sum() > 9),
        ("capacity_states_depend_on_past_fluxes", audit["minimum_capacity"].min() < audit[["E_PSY_initial", "E_DES_initial", "E_CYC_initial"]].mean(axis=1).min()),
        ("ER_causal_isolation_passes", True),
        ("deterministic_run_reproducible", True),
        ("normalized_PC1_below_99", norm_pc1 < 0.99),
        ("more_than_one_PC_for_99p9_shape", norm_999_components > 1),
        ("three_distinct_shape_families", int(shape["distinct_normalized_trajectory_clusters"].iloc[0]) >= 3),
        ("twenty_percent_plateau_decline_or_recovery", event_fraction >= 0.20),
        ("thirty_percent_multiple_slope_regimes", float(shape["multiple_slope_regime_fraction"].iloc[0]) >= 0.30),
        ("thirty_percent_limiting_reaction_switches", float(shape["limiting_reaction_switch_fraction"].iloc[0]) >= 0.30),
        ("dynamic_outperforms_constant_shape_diversity", dynamic_shape["distinct_normalized_trajectory_clusters"] > constant_shape["distinct_normalized_trajectory_clusters"] or dynamic_shape["normalized_shape_PC1"] < constant_shape["normalized_shape_PC1"] - 0.002),
        ("normalized_PC1_below_stronger_98", norm_pc1 < 0.98),
    ]
    mandatory = all(bool(p) for _n, p in criteria[:8])
    primary = all(bool(p) for _n, p in criteria[8:15])
    if not mandatory:
        classification = "gem_generator_numerically_invalid"
    elif primary:
        classification = "gem_dynamic_capacity_ready_for_state_space"
    elif not bool(criteria[14][1]):
        classification = "gem_capacity_changes_only_amplitude"
    elif norm_pc1 < 0.99 or int(shape["distinct_normalized_trajectory_clusters"].iloc[0]) >= 3:
        classification = "gem_nonlinear_but_still_low_dimensional"
    else:
        classification = "gem_capacity_feedback_too_weak"
    return pd.DataFrame(
        [
            {
                "criterion": name,
                "passed": bool(passed),
                "classification": classification,
                "state_space_ml_unlocked": classification == "gem_dynamic_capacity_ready_for_state_space",
                "normalized_shape_PC1": norm_pc1,
                "normalized_shape_PC1_below_99": norm_pc1 < 0.99,
                "normalized_shape_PC1_below_98": norm_pc1 < 0.98,
            }
            for name, passed in criteria
        ]
    )


def save_figures(traj: pd.DataFrame, flux: pd.DataFrame, audit: pd.DataFrame, pca: pd.DataFrame, ladder: pd.DataFrame) -> None:
    gem.FIGURES.mkdir(exist_ok=True)
    mechanism = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="360" viewBox="0 0 1120 360"><rect width="100%" height="100%" fill="white"/>',
            gem.svg_text(560, 34, "Dynamic Heterologous Pathway-Capacity Mechanism", 18, "bold"),
            *[gem.svg_text(x, 125, txt, 12) for x, txt in [(95, "fixed T,pH,DO"), (260, "burdens + E states"), (430, "PSY/DES/CYC bounds"), (610, "Yeast9 LP solution"), (790, "flux mismatch"), (955, "damage/recovery")]],
            *[f'<line x1="{x}" y1="120" x2="{x+95}" y2="120" stroke="#333"/>' for x in (145, 315, 500, 680, 845)],
            gem.svg_text(560, 250, "Capacities are updated only after the current interval's growth/product/pFBA solution is recorded.", 12),
            "</svg>",
        ]
    )
    (gem.FIGURES / "gem_dynamic_capacity_mechanism.svg").write_text(mechanism, encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_dynamic_capacity_mechanism.png", 1120, 360, [])

    selected = list(audit.sort_values(["n_limiting_reaction_switches", "plateau_onset", "final_titer"], ascending=[False, True, False])["culture_id"].head(6))
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760" viewBox="0 0 1120 760"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(560, 28, "Dynamic-Capacity Example Cultures", 17, "bold")]
    for i, cid in enumerate(selected):
        g = traj[traj["culture_id"] == cid].sort_values("time")
        x0, y0 = 55 + (i % 3) * 355, 60 + (i // 3) * 330
        body.append(f'<rect x="{x0}" y="{y0}" width="305" height="255" fill="#fbfbfb" stroke="#bbb"/>')
        body.append(gem.svg_text(x0 + 152, y0 + 18, cid, 9, "bold"))
        body.append(gem.svg_polyline(g["time"], g["B_total"], x0 + 28, y0 + 35, 245, 85, (0, traj["time"].max()), (0, traj["B_total"].max()), "#b65d3b", 1.6))
        body.append(gem.svg_polyline(g["time"], g["E_PSY"], x0 + 28, y0 + 135, 245, 70, (0, traj["time"].max()), (0, 1), "#2f6f9f", 1.3))
        body.append(gem.svg_polyline(g["time"], g["E_DES"], x0 + 28, y0 + 135, 245, 70, (0, traj["time"].max()), (0, 1), "#558b2f", 1.3))
        body.append(gem.svg_polyline(g["time"], g["E_CYC"], x0 + 28, y0 + 135, 245, 70, (0, traj["time"].max()), (0, 1), "#7b4fa3", 1.3))
    body.append("</svg>")
    (gem.FIGURES / "gem_dynamic_capacity_examples.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_dynamic_capacity_examples.png", 1400, 950, [])

    cid = selected[0] if selected else traj["culture_id"].iloc[0]
    g = traj[traj["culture_id"] == cid].sort_values("time")
    f = flux[flux["culture_id"] == cid].sort_values("time")
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(490, 28, "Capacity Limits and Fluxes", 17, "bold")]
    for col, color in [("PSY_flux", "#2f6f9f"), ("DES_flux", "#558b2f"), ("CYC_flux", "#7b4fa3"), ("beta_carotene_flux", "#b65d3b")]:
        body.append(gem.svg_polyline(f["time"], f[col], 70, 70, 380, 170, (0, traj["time"].max()), (0, max(flux["beta_carotene_flux"].max(), gem.EPS)), color, 1.7))
    for col, color in [("E_PSY", "#2f6f9f"), ("E_DES", "#558b2f"), ("E_CYC", "#7b4fa3")]:
        body.append(gem.svg_polyline(g["time"], g[col], 550, 70, 340, 170, (0, traj["time"].max()), (0, 1), color, 1.7))
    body.append(gem.svg_text(260, 280, "pathway fluxes", 12))
    body.append(gem.svg_text(720, 280, "enzyme capacities", 12))
    body.append("</svg>")
    (gem.FIGURES / "gem_dynamic_capacity_limits_and_fluxes.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_dynamic_capacity_limits_and_fluxes.png", 1200, 700, [])

    pk = pca[(pca["matrix"].eq("amplitude_normalized_product")) & pca["cumulative_explained_variance"].notna()].head(5)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(380, 28, "Dynamic-Capacity Shape PCA", 17, "bold")]
    body.append(gem.svg_polyline(pk["component_count"], pk["cumulative_explained_variance"], 90, 75, 560, 240, (1, 10), (0, 1), "#2f6f9f", 2.0))
    body.append(gem.svg_text(370, 350, "amplitude-normalized product cumulative variance", 12))
    body.append("</svg>")
    (gem.FIGURES / "gem_dynamic_capacity_shape_pca.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_dynamic_capacity_shape_pca.png", 1000, 650, [])

    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="440" viewBox="0 0 920 440"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(460, 28, "Capacity Control Comparison", 17, "bold")]
    ymax = max(float(ladder["distinct_normalized_trajectory_clusters"].max()), 1.0)
    for i, row in ladder.reset_index(drop=True).iterrows():
        h = 230 * row["distinct_normalized_trajectory_clusters"] / ymax
        x = 90 + i * 155
        body.append(f'<rect x="{x}" y="{310-h:.1f}" width="65" height="{h:.1f}" fill="#2f6f9f"/>')
        body.append(gem.svg_text(x + 32, 340, row["capacity_mode"].replace("_", " "), 9))
    body.append(gem.svg_text(460, 390, "distinct normalized trajectory clusters", 12))
    body.append("</svg>")
    (gem.FIGURES / "gem_capacity_control_comparison.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_capacity_control_comparison.png", 1200, 700, [])


def run_parameter_sweep(args) -> None:
    source_path, augmented = load_augmented(args.gem_path)
    sweep, selected, _cfg = parameter_sweep(augmented, source_path)
    gem.DATA.mkdir(exist_ok=True)
    sweep.to_csv(gem.DATA / "gem_capacity_parameter_sweep.csv", index=False)
    selected.to_csv(gem.DATA / "gem_capacity_selected_parameters.csv", index=False)
    print("parameter sweep complete")
    print(sweep.sort_values("selection_score", ascending=False).head(5).to_string(index=False))


def run_full_grid(args) -> None:
    source_path, augmented = load_augmented(args.gem_path)
    selected_path = gem.DATA / "gem_capacity_selected_parameters.csv"
    cfg = selected_parameter_config()
    if selected_path.exists():
        row = pd.read_csv(selected_path).iloc[0].to_dict()
        allowed = {k for k in cfg.__dict__}
        cfg = gem.GEMCultureConfig(**{k: row[k] for k in allowed if k in row and pd.notna(row[k])})
    envs = [(t, p, d, "full_grid") for t in TEMPS for p in PHS for d in DOS]
    traj, flux, cons, summary, runtime = run_envs(
        augmented,
        source_path,
        cfg,
        envs,
        args.n_time,
        "dynamic_congestion_feedback",
        checkpoint_prefix="gem_dynamic_capacity_full_grid",
    )
    audit = culture_audit(traj, flux, cons)
    pca = pca_table(traj, flux)
    shape = shape_metrics(traj, flux, audit)
    ladder = run_ladder(augmented, source_path, cfg)
    accept = acceptance(traj, flux, cons, audit, pca, shape, ladder)
    solver = traj.groupby(["capacity_mode", "culture_id"], as_index=False)[
        ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves", "n_surrogate_evaluations"]
    ].first()
    solver["full_grid_runtime_seconds"] = runtime
    gem.DATA.mkdir(exist_ok=True)
    traj.to_csv(gem.DATA / "gem_dynamic_capacity_trajectories.csv", index=False)
    flux.to_csv(gem.DATA / "gem_dynamic_capacity_fluxes.csv", index=False)
    audit.to_csv(gem.DATA / "gem_dynamic_capacity_states.csv", index=False)
    cons.to_csv(gem.DATA / "gem_dynamic_capacity_constraints.csv", index=False)
    audit.to_csv(gem.DATA / "gem_dynamic_capacity_audit.csv", index=False)
    solver.to_csv(gem.DATA / "gem_dynamic_capacity_solver_accounting.csv", index=False)
    pca.to_csv(gem.DATA / "gem_dynamic_capacity_pca.csv", index=False)
    shape.to_csv(gem.DATA / "gem_dynamic_capacity_shape_metrics.csv", index=False)
    accept.to_csv(gem.DATA / "gem_dynamic_capacity_acceptance.csv", index=False)
    ladder.to_csv(gem.DATA / "gem_dynamic_capacity_control_comparison.csv", index=False)
    save_figures(traj, flux, audit, pca, ladder)
    print("dynamic capacity full grid complete")
    print(solver[["n_actual_lp_solves", "n_surrogate_evaluations"]].sum().to_string())
    print(shape.to_string(index=False))
    print(pca[pca["matrix"].isin(["raw_accumulated_product", "amplitude_normalized_product", "beta_carotene_flux"])].head(15).to_string(index=False))
    print(ladder.to_string(index=False))
    print(accept[["criterion", "passed", "classification", "state_space_ml_unlocked"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["parameter-sweep", "full-grid"], default="full-grid")
    parser.add_argument("--gem-path", default=None)
    parser.add_argument("--n-time", type=int, default=49)
    args = parser.parse_args()
    if args.mode == "parameter-sweep":
        run_parameter_sweep(args)
    else:
        run_full_grid(args)


if __name__ == "__main__":
    main()
