#!/usr/bin/env python3
"""Current-system complexity audit from cached data only.

This script does not invoke Yeast9, retrain models, or launch DBTL campaigns.
It reads the canonical 125-culture `gem_state_space_*` tables and cached exact
candidate ledgers, then writes empirical dimensionality summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12
RNG_SEED = 20260822

OBS_TRAJ_COLS = ["X", "B_total"]
REPORTER_COLS = ["R_ox", "R_atp", "R_er", "R_E_PSY", "R_E_DES", "R_E_CYC"]
HIDDEN_COMPACT_COLS = ["z_ox", "z_atp", "z_bottle", "z_er"]
CAPACITY_COLS = ["E_PSY", "E_DES", "E_CYC"]
SELECTED_FLUX_COLS = [
    "maximum_growth",
    "preserved_growth",
    "beta_carotene_flux",
    "biomass_flux",
    "glucose_uptake",
    "oxygen_uptake",
    "atp_maintenance_flux",
    "ggpp_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
    "phytoene_flux",
    "desaturase_flux",
    "cyclase_flux",
    "lycopene_flux",
    "phytoene_congestion",
    "lycopene_congestion",
]
ACTIVE_CONSTRAINT_COLS = [
    "glucose_lower_bound",
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "pathway_capacity_upper_bound",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
    "gamma_growth_fraction",
    "oxidative_burden_after",
    "ATP_pressure_after",
    "bottleneck_after",
    "ER_after",
    "phytoene_capacity_pressure",
    "lycopene_capacity_pressure",
]
PHENOTYPE_COLS = [
    "final_product",
    "product_AUC",
    "productivity",
    "final_biomass",
    "minimum_biomass",
    "maximum_instantaneous_product_rate",
]


def standardize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x), posinf=np.nanmax(x[np.isfinite(x)]), neginf=np.nanmin(x[np.isfinite(x)]))
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (x - mu) / sd


def remove_constant_columns(x: np.ndarray, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    sd = np.nanstd(x, axis=0)
    keep = sd > 1e-10
    if names is None:
        names = [f"f{i}" for i in range(x.shape[1])]
    return x[:, keep], [n for n, k in zip(names, keep) if k]


def eigenvalues_from_samples(x: np.ndarray, standardize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x, _ = remove_constant_columns(x)
    if x.shape[1] == 0:
        return np.array([])
    x = standardize_matrix(x) if standardize else x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    eig = (s**2) / max(x.shape[0] - 1, 1)
    eig = eig[eig > EPS]
    return eig


def dimension_metrics(eig: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eig, dtype=float)
    eig = eig[eig > EPS]
    if eig.size == 0:
        return {
            "n_components_nonzero": 0,
            "entropy_rank": np.nan,
            "participation_rank": np.nan,
            "pc90": np.nan,
            "pc95": np.nan,
            "pc99": np.nan,
            "pc999": np.nan,
        }
    p = eig / eig.sum()
    c = np.cumsum(p)
    return {
        "n_components_nonzero": int(eig.size),
        "entropy_rank": float(np.exp(-(p * np.log(p + EPS)).sum())),
        "participation_rank": float((eig.sum() ** 2) / np.sum(eig**2)),
        "pc90": int(np.searchsorted(c, 0.90) + 1),
        "pc95": int(np.searchsorted(c, 0.95) + 1),
        "pc99": int(np.searchsorted(c, 0.99) + 1),
        "pc999": int(np.searchsorted(c, 0.999) + 1),
    }


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = standardize_matrix(np.asarray(x, dtype=float))
    x, _ = remove_constant_columns(x)
    if x.shape[0] < 3 or x.shape[1] == 0:
        return np.empty((x.shape[0], x.shape[0]))
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    return d


def twonn_id(x: np.ndarray) -> float:
    d = pairwise_distances(x)
    if d.size == 0:
        return np.nan
    nearest = np.sort(d, axis=1)[:, :2]
    good = (nearest[:, 0] > EPS) & np.isfinite(nearest[:, 1])
    mu = nearest[good, 1] / nearest[good, 0]
    if mu.size < 5:
        return np.nan
    return float(1.0 / np.mean(np.log(mu + EPS)))


def levina_bickel_id(x: np.ndarray, ks: Iterable[int] = (5, 8, 12)) -> float:
    d = pairwise_distances(x)
    if d.size == 0:
        return np.nan
    vals = []
    sorted_d = np.sort(d, axis=1)
    for k in ks:
        if sorted_d.shape[1] <= k:
            continue
        r = sorted_d[:, : k + 1]
        rk = r[:, k]
        good = (r[:, 0] > EPS) & (rk > EPS) & np.isfinite(rk)
        if good.sum() < 5:
            continue
        logs = np.log((rk[good, None] + EPS) / (r[good, :k] + EPS))
        inv = np.mean(logs, axis=1)
        local = 1.0 / np.maximum(inv, EPS)
        vals.append(np.median(local[np.isfinite(local)]))
    return float(np.mean(vals)) if vals else np.nan


def nonlinear_id(x: np.ndarray) -> tuple[float, float]:
    return twonn_id(x), levina_bickel_id(x)


def wide_by_culture(df: pd.DataFrame, cols: list[str], order_col: str = "time_index") -> tuple[np.ndarray, list[str], list[str]]:
    cols = [c for c in cols if c in df.columns]
    ids = sorted(df["culture_id"].unique())
    blocks = []
    names = []
    for col in cols:
        pivot = df.pivot(index="culture_id", columns=order_col, values=col).reindex(ids)
        arr = pivot.to_numpy(float)
        blocks.append(arr)
        names.extend([f"{col}@{t}" for t in pivot.columns])
    if not blocks:
        return np.empty((len(ids), 0)), ids, []
    return np.concatenate(blocks, axis=1), ids, names


def amplitude_normalized_product(traj: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ids = sorted(traj["culture_id"].unique())
    pivot = traj.pivot(index="culture_id", columns="time_index", values="B_total").reindex(ids)
    y = pivot.to_numpy(float)
    amp = np.maximum(y.max(axis=1, keepdims=True) - y.min(axis=1, keepdims=True), EPS)
    yn = (y - y[:, :1]) / amp
    return yn, ids


def product_slopes(traj: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ids = sorted(traj["culture_id"].unique())
    p = traj.pivot(index="culture_id", columns="time_index", values="B_total").reindex(ids)
    t = traj.drop_duplicates("time_index").sort_values("time_index")["time"].to_numpy(float)
    y = p.to_numpy(float)
    dt = np.diff(t)
    slopes = np.diff(y, axis=1) / dt[None, :]
    return slopes, ids


def delay_embed_by_culture(df: pd.DataFrame, cols: list[str], lag: int = 8, order_col: str = "time_index") -> tuple[np.ndarray, list[str]]:
    cols = [c for c in cols if c in df.columns]
    ids = sorted(df["culture_id"].unique())
    windows = []
    window_cultures = []
    for cid in ids:
        g = df[df["culture_id"] == cid].sort_values(order_col)
        arr = g[cols].to_numpy(float)
        if arr.shape[0] < lag:
            continue
        for start in range(arr.shape[0] - lag + 1):
            windows.append(arr[start : start + lag].reshape(-1))
            window_cultures.append(cid)
    return np.asarray(windows, dtype=float), window_cultures


def summarize_dataset(
    layer: str,
    x: np.ndarray,
    standardize: bool = True,
    bootstrap: bool = True,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = rng or np.random.default_rng(RNG_SEED)
    x = np.asarray(x, dtype=float)
    x, _ = remove_constant_columns(x)
    eig = eigenvalues_from_samples(x, standardize=standardize)
    row = {"layer": layer, "n_samples": x.shape[0], "n_features_varying": x.shape[1], **dimension_metrics(eig)}
    twonn, lb = nonlinear_id(x)
    row["twonn_id"] = twonn
    row["levina_bickel_id"] = lb
    boot_rows = []
    if bootstrap and x.shape[0] >= 20:
        # Bootstrap with replacement is appropriate for covariance/effective-rank
        # uncertainty. Nearest-neighbor intrinsic-dimension estimators are not
        # bootstrapped here because duplicate resampled trajectories create zero
        # neighbor distances and unstable artificial IDs; use the separate
        # complete-culture subsampling table for finite-N ID sensitivity.
        metrics = ["entropy_rank", "participation_rank", "pc95", "pc99"]
        vals = {m: [] for m in metrics}
        for _ in range(n_boot):
            idx = rng.integers(0, x.shape[0], size=x.shape[0])
            eig_b = eigenvalues_from_samples(x[idx], standardize=standardize)
            mb = dimension_metrics(eig_b)
            for m in metrics:
                vals[m].append(mb[m])
        for m, arr in vals.items():
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                row[f"{m}_ci_low"] = float(np.quantile(arr, 0.025))
                row[f"{m}_ci_high"] = float(np.quantile(arr, 0.975))
            else:
                row[f"{m}_ci_low"] = np.nan
                row[f"{m}_ci_high"] = np.nan
    spectrum = pd.DataFrame(
        {
            "layer": layer,
            "pc": np.arange(1, eig.size + 1),
            "eigenvalue": eig,
            "variance_explained": eig / eig.sum() if eig.size else [],
            "cumulative_variance": np.cumsum(eig / eig.sum()) if eig.size else [],
        }
    )
    return row, spectrum


def channel_standardized_flatten(df: pd.DataFrame, cols: list[str], order_col: str = "time_index") -> tuple[np.ndarray, list[str], list[str]]:
    blocks = []
    names = []
    ids_out = None
    for col in cols:
        if col not in df.columns:
            continue
        x, ids, col_names = wide_by_culture(df, [col], order_col=order_col)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        if not np.isfinite(sd) or sd < EPS:
            sd = 1.0
        x = (x - mu) / sd  # one scale per biological measurement channel
        blocks.append(x)
        names.extend(col_names)
        ids_out = ids
    if not blocks:
        return np.empty((0, 0)), [], []
    return np.concatenate(blocks, axis=1), ids_out or [], names


def parse_candidate_designs(ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    edit_types = set()
    mechanisms = set()
    parsed_cache: list[dict] = []
    for _, r in ledger.iterrows():
        try:
            cand = json.loads(r.get("candidate_json", "{}"))
        except Exception:
            cand = {}
        edits = cand.get("edits", []) or []
        parsed_cache.append(cand)
        for edit in edits:
            edit_types.add(str(edit.get("edit_type", "unknown")))
        for mech in str(r.get("mechanism_classes", "")).split(";"):
            mech = mech.strip()
            if mech and mech != "nan":
                mechanisms.add(mech)
    edit_types = sorted(edit_types)
    mechanisms = sorted(mechanisms)
    for (_, r), cand in zip(ledger.iterrows(), parsed_cache):
        edits = cand.get("edits", []) or []
        rec = {
            "temperature": float(r.get("temperature", np.nan)),
            "pH": float(r.get("pH", np.nan)),
            "DO": float(r.get("DO", np.nan)),
            "edit_count": float(r.get("edit_count", len(edits))),
        }
        mults = []
        for edit in edits:
            try:
                mults.append(float(edit.get("capacity_multiplier", np.nan)))
            except Exception:
                pass
        mults = [m for m in mults if np.isfinite(m)]
        rec["edit_multiplier_mean"] = float(np.mean(mults)) if mults else 1.0
        rec["edit_multiplier_sd"] = float(np.std(mults)) if len(mults) > 1 else 0.0
        rec["edit_multiplier_min"] = float(np.min(mults)) if mults else 1.0
        rec["edit_multiplier_max"] = float(np.max(mults)) if mults else 1.0
        for et in edit_types:
            rec[f"edit_type::{et}"] = sum(1 for e in edits if str(e.get("edit_type", "unknown")) == et)
        mech_set = {m.strip() for m in str(r.get("mechanism_classes", "")).split(";") if m.strip()}
        for mech in mechanisms:
            rec[f"mechanism::{mech}"] = 1.0 if mech in mech_set else 0.0
        rows.append(rec)
    design = pd.DataFrame(rows).fillna(0.0)
    return design, design.columns.tolist()


def load_candidate_ledgers(root: Path) -> pd.DataFrame:
    paths = [
        root / "results/repaired_prospective_benchmark_v2_aggregate/data/prospective_exact_simulator_call_ledger.csv",
        root / "results/repaired_prospective_benchmark_aggregate/data/prospective_exact_simulator_call_ledger.csv",
    ]
    frames = []
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            df["source_file"] = str(p)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df.get("mock_mode", False).astype(str).str.lower().isin(["false", "0", "nan"])]
    df = df.drop_duplicates("candidate_hash")
    return df


def intervention_response_summary(ledger: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    ledger = ledger.copy()
    ledger = ledger[[c for c in ledger.columns if c in set(PHENOTYPE_COLS + [
        "candidate_hash", "candidate_json", "temperature", "pH", "DO", "edit_count", "mechanism_classes",
        "world_id", "method_id", "final_product", "product_AUC", "final_biomass", "productivity",
        "minimum_biomass", "maximum_instantaneous_product_rate",
    ])]]
    ledger = ledger.dropna(subset=["final_product", "product_AUC", "final_biomass", "productivity"])
    design, design_cols = parse_candidate_designs(ledger)
    y_cols = [c for c in PHENOTYPE_COLS if c in ledger.columns]
    y = ledger[y_cols].astype(float).to_numpy()
    u = design.to_numpy(float)
    u, kept_u = remove_constant_columns(u, design_cols)
    y, kept_y = remove_constant_columns(y, y_cols)
    uz = standardize_matrix(u)
    yz = standardize_matrix(y)
    ridge = 1e-6
    j = np.linalg.solve(uz.T @ uz + ridge * np.eye(uz.shape[1]), uz.T @ yz)
    s = np.linalg.svd(j, compute_uv=False)
    row = {
        "layer": "Intervention response",
        "n_samples": int(ledger.shape[0]),
        "n_features_varying": int(uz.shape[1]),
        "n_outputs_varying": int(yz.shape[1]),
        **dimension_metrics(s**2),
        "condition_number": float(s[0] / s[-1]) if len(s) and s[-1] > EPS else np.nan,
        "twonn_id": twonn_id(np.concatenate([uz, yz], axis=1)),
        "levina_bickel_id": levina_bickel_id(np.concatenate([uz, yz], axis=1)),
    }
    spectrum = pd.DataFrame(
        {
            "layer": "Intervention response",
            "pc": np.arange(1, len(s) + 1),
            "singular_value": s,
            "variance_explained": (s**2) / np.sum(s**2) if len(s) else [],
            "cumulative_variance": np.cumsum((s**2) / np.sum(s**2)) if len(s) else [],
        }
    )
    coeff = pd.DataFrame(j, index=kept_u, columns=kept_y)
    coeff.index.name = "design_feature"
    return row, spectrum, coeff.reset_index()


def save_spectrum_plot(spectra: pd.DataFrame, out: Path, title: str, layers: list[str]) -> None:
    plt.figure(figsize=(8, 5))
    for layer in layers:
        g = spectra[spectra["layer"] == layer]
        if g.empty:
            continue
        plt.plot(g["pc"], g["cumulative_variance"], marker="o", markersize=3, linewidth=1.3, label=layer)
    plt.axhline(0.95, color="#777777", linestyle="--", linewidth=0.8)
    plt.axhline(0.99, color="#999999", linestyle=":", linewidth=0.8)
    plt.xlabel("Principal component / singular direction")
    plt.ylabel("Cumulative variance")
    plt.ylim(0, 1.01)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def write_state_space_inventory(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    def add(table: str, variable: str, category: str, rationale: str) -> None:
        rows.append({"table": table, "variable": variable, "category": category, "rationale": rationale})
    for c in ["X", "B_total"]:
        add("data/gem_state_space_trajectories.csv", c, "Observable state", "Plausible biomass/product trajectory measurement")
    for c in REPORTER_COLS:
        add("data/gem_state_space_reporters.csv", c, "Observable state", "Explicit simulated biosensor/reporter signal")
    for c in HIDDEN_COMPACT_COLS:
        add("data/gem_state_space_trajectories.csv", c, "Hidden physiological state", "Simulator-only latent regulatory/burden state")
    for c in CAPACITY_COLS:
        add("data/gem_state_space_trajectories.csv", c, "Hidden physiological state", "Simulator-only engineered pathway capacity state")
    for c in SELECTED_FLUX_COLS:
        add("data/gem_state_space_fluxes.csv", c, "Metabolic state", "Selected Yeast9/pFBA flux summary")
    for c in ACTIVE_CONSTRAINT_COLS:
        add("data/gem_state_space_generator_constraints.partial.csv", c, "Metabolic state", "Active dynamic bound/constraint or burden update term")
    for c in ["temperature", "pH", "DO"]:
        add("data/gem_state_space_environment_grid.csv", c, "Intervention/design state", "Controlled environment variable")
    add("results/repaired_prospective_benchmark_v2_aggregate/data/prospective_exact_simulator_call_ledger.csv", "candidate_json", "Intervention/design state", "Strain edit specification for cached DBTL candidates")
    for c in PHENOTYPE_COLS:
        add("results/repaired_prospective_benchmark_v2_aggregate/data/prospective_exact_simulator_call_ledger.csv", c, "Exact candidate outcome", "Hidden exact simulator phenotype used for post-selection verification")
    inv = pd.DataFrame(rows)
    inv.to_csv(out_dir / "current_system_state_space_inventory.csv", index=False)
    return inv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--subsample-reps", type=int, default=200)
    parser.add_argument("--delay-lag", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / "data/current_system_complexity_audit"
    fig_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    traj = pd.read_csv(root / "data/gem_state_space_trajectories.csv")
    reporters = pd.read_csv(root / "data/gem_state_space_reporters.csv")
    states = pd.read_csv(root / "data/gem_state_space_states.csv")
    flux = pd.read_csv(root / "data/gem_state_space_fluxes.csv")
    constraints = pd.read_csv(root / "data/gem_state_space_generator_constraints.partial.csv")
    env = pd.read_csv(root / "data/gem_state_space_environment_grid.csv")
    yeast_audit = pd.read_csv(root / "data/yeast_gem_audit.csv")
    ledger = load_candidate_ledgers(root)

    inventory = write_state_space_inventory(root, out_dir)
    dataset_rows = [
        {"artifact": "data/gem_state_space_trajectories.csv", "rows": len(traj), "cultures": traj["culture_id"].nunique(), "timepoints": traj["time_index"].nunique(), "role": "primary 125-culture trajectory table"},
        {"artifact": "data/gem_state_space_reporters.csv", "rows": len(reporters), "cultures": reporters["culture_id"].nunique(), "timepoints": reporters["time_index"].nunique(), "role": "experimentally plausible reporter table"},
        {"artifact": "data/gem_state_space_states.csv", "rows": len(states), "cultures": states["culture_id"].nunique(), "timepoints": np.nan, "role": "culture-level hidden physiology summaries"},
        {"artifact": "data/gem_state_space_fluxes.csv", "rows": len(flux), "cultures": flux["culture_id"].nunique(), "timepoints": flux["interval_index"].nunique(), "role": "selected Yeast9/pFBA flux summaries"},
        {"artifact": "data/gem_state_space_generator_constraints.partial.csv", "rows": len(constraints), "cultures": constraints["culture_id"].nunique(), "timepoints": constraints["interval_index"].nunique(), "role": "active dynamic constraints and hidden update terms"},
        {"artifact": "data/gem_state_space_environment_grid.csv", "rows": len(env), "cultures": len(env), "timepoints": np.nan, "role": "environment/design grid"},
        {"artifact": "results/repaired_prospective_benchmark*_aggregate/data/prospective_exact_simulator_call_ledger.csv", "rows": len(ledger), "cultures": np.nan, "timepoints": np.nan, "role": "cached exact candidate outcomes for intervention-response audit"},
    ]
    pd.DataFrame(dataset_rows).to_csv(out_dir / "current_system_dataset_inventory.csv", index=False)

    rows = []
    spectra = []
    matrices: dict[str, np.ndarray] = {}

    raw_prod, ids, _ = wide_by_culture(traj, ["B_total"])
    norm_prod, _ = amplitude_normalized_product(traj)
    slopes, _ = product_slopes(traj)
    observable, _, _ = channel_standardized_flatten(
        traj.merge(reporters[["culture_id", "time_index"] + REPORTER_COLS], on=["culture_id", "time_index"], how="left"),
        OBS_TRAJ_COLS + REPORTER_COLS,
    )
    hidden_compact, _, _ = channel_standardized_flatten(traj, HIDDEN_COMPACT_COLS)
    hidden_capacity, _, _ = channel_standardized_flatten(traj, HIDDEN_COMPACT_COLS + CAPACITY_COLS)
    hidden_flux, _, _ = channel_standardized_flatten(
        flux,
        [c for c in HIDDEN_COMPACT_COLS + CAPACITY_COLS + SELECTED_FLUX_COLS if c in flux.columns],
        order_col="interval_index",
    )
    flux_state, _, flux_names = channel_standardized_flatten(flux, [c for c in SELECTED_FLUX_COLS if c in flux.columns], order_col="interval_index")
    constraint_state, _, constraint_names = channel_standardized_flatten(constraints, [c for c in ACTIVE_CONSTRAINT_COLS if c in constraints.columns], order_col="interval_index")
    metabolic = np.concatenate([flux_state, constraint_state], axis=1)

    matrices.update(
        {
            "Product raw": raw_prod,
            "Product normalized": norm_prod,
            "Product slopes": slopes,
            "Observable cell": observable,
            "Hidden physiology": hidden_compact,
            "Hidden physiology + capacities": hidden_capacity,
            "Hidden + selected fluxes": hidden_flux,
            "Metabolic flux/constraint state": metabolic,
            "Selected flux state": flux_state,
            "Active constraint state": constraint_state,
        }
    )

    for layer, x in matrices.items():
        standardize = layer != "Product raw"
        row, spec = summarize_dataset(layer, x, standardize=standardize, bootstrap=True, n_boot=args.bootstrap, rng=rng)
        rows.append(row)
        spectra.append(spec)

    dyn_prod, _ = delay_embed_by_culture(traj, ["B_total"], lag=args.delay_lag)
    dyn_obs_df = traj.merge(reporters[["culture_id", "time_index"] + REPORTER_COLS], on=["culture_id", "time_index"], how="left")
    dyn_obs, _ = delay_embed_by_culture(dyn_obs_df, OBS_TRAJ_COLS + REPORTER_COLS, lag=args.delay_lag)
    dyn_hidden, _ = delay_embed_by_culture(traj, HIDDEN_COMPACT_COLS + CAPACITY_COLS, lag=args.delay_lag)
    for layer, x in {
        "Dynamical product": dyn_prod,
        "Dynamical observable state": dyn_obs,
        "Dynamical hidden state": dyn_hidden,
    }.items():
        row, spec = summarize_dataset(layer, x, standardize=True, bootstrap=False, n_boot=0, rng=rng)
        rows.append(row)
        spectra.append(spec)

    if not ledger.empty:
        irow, ispec, coeff = intervention_response_summary(ledger)
        rows.append(irow)
        spectra.append(ispec.rename(columns={"singular_value": "eigenvalue"}))
        coeff.to_csv(out_dir / "current_system_intervention_response_coefficients.csv", index=False)

    summary = pd.DataFrame(rows)
    spectra_df = pd.concat(spectra, ignore_index=True, sort=False)
    summary.to_csv(out_dir / "current_system_complexity_fingerprint.csv", index=False)
    spectra_df.to_csv(out_dir / "current_system_complexity_spectra.csv", index=False)

    # Subsampling stability by complete cultures.
    subsample_rows = []
    culture_ids = np.array(sorted(traj["culture_id"].unique()))
    for n in [25, 50, 75, 100, len(culture_ids)]:
        reps = 1 if n == len(culture_ids) else args.subsample_reps
        for rep in range(reps):
            chosen = culture_ids if n == len(culture_ids) else rng.choice(culture_ids, size=n, replace=False)
            mask = np.isin(culture_ids, chosen)
            for layer in ["Product raw", "Product normalized", "Observable cell", "Hidden physiology", "Metabolic flux/constraint state"]:
                x = matrices[layer][mask]
                eig = eigenvalues_from_samples(x, standardize=(layer != "Product raw"))
                m = dimension_metrics(eig)
                subsample_rows.append(
                    {
                        "layer": layer,
                        "subsample_cultures": n,
                        "rep": rep,
                        "entropy_rank": m["entropy_rank"],
                        "participation_rank": m["participation_rank"],
                        "pc95": m["pc95"],
                        "pc99": m["pc99"],
                    }
                )
    subs = pd.DataFrame(subsample_rows)
    subs.to_csv(out_dir / "current_system_complexity_subsampling.csv", index=False)
    subs_summary = (
        subs.groupby(["layer", "subsample_cultures"], as_index=False)
        .agg(
            entropy_rank_mean=("entropy_rank", "mean"),
            entropy_rank_ci_low=("entropy_rank", lambda s: np.quantile(s, 0.025)),
            entropy_rank_ci_high=("entropy_rank", lambda s: np.quantile(s, 0.975)),
            participation_rank_mean=("participation_rank", "mean"),
            pc95_median=("pc95", "median"),
            pc99_median=("pc99", "median"),
        )
    )
    subs_summary.to_csv(out_dir / "current_system_complexity_subsampling_summary.csv", index=False)

    nonlinear_rows = []
    for layer in [
        "Product raw",
        "Product normalized",
        "Observable cell",
        "Hidden physiology",
        "Metabolic flux/constraint state",
    ]:
        x = matrices[layer]
        nonlinear_rows.append({"layer": layer, "estimator": "TwoNN", "neighborhood": "1-2", "intrinsic_dimension": twonn_id(x)})
        for k in [5, 8, 12]:
            nonlinear_rows.append(
                {
                    "layer": layer,
                    "estimator": "Levina-Bickel MLE",
                    "neighborhood": f"k={k}",
                    "intrinsic_dimension": levina_bickel_id(x, ks=[k]),
                }
            )
    pd.DataFrame(nonlinear_rows).to_csv(out_dir / "current_system_nonlinear_id_sensitivity.csv", index=False)

    # Nominal-vs-empirical metabolic inventory.
    flux_numeric = flux[[c for c in SELECTED_FLUX_COLS if c in flux.columns]].select_dtypes(include=[np.number])
    constraint_numeric = constraints[[c for c in ACTIVE_CONSTRAINT_COLS if c in constraints.columns]].select_dtypes(include=[np.number])
    meta_inventory = pd.DataFrame(
        [
            {
                "quantity": "Yeast9 nominal reactions",
                "value": int(yeast_audit["reactions"].iloc[0]),
                "source": "data/yeast_gem_audit.csv",
            },
            {
                "quantity": "Yeast9 nominal metabolites",
                "value": int(yeast_audit["metabolites"].iloc[0]),
                "source": "data/yeast_gem_audit.csv",
            },
            {
                "quantity": "Selected cached flux channels",
                "value": int(flux_numeric.shape[1]),
                "source": "data/gem_state_space_fluxes.csv",
            },
            {
                "quantity": "Selected cached flux channels with variation",
                "value": int((flux_numeric.std(axis=0) > 1e-10).sum()),
                "source": "data/gem_state_space_fluxes.csv",
            },
            {
                "quantity": "Cached active constraint channels",
                "value": int(constraint_numeric.shape[1]),
                "source": "data/gem_state_space_generator_constraints.partial.csv",
            },
            {
                "quantity": "Cached active constraint channels with variation",
                "value": int((constraint_numeric.std(axis=0) > 1e-10).sum()),
                "source": "data/gem_state_space_generator_constraints.partial.csv",
            },
        ]
    )
    meta_inventory.to_csv(out_dir / "current_system_metabolic_nominal_vs_empirical.csv", index=False)

    # Compact fingerprint table requested by the prompt.
    aliases = {
        "Product raw": "Product raw",
        "Product normalized": "Product normalized",
        "Observable cell": "Observable cell",
        "Hidden physiology": "Hidden physiology",
        "Metabolic flux/constraint state": "Metabolic flux state",
        "Intervention response": "Intervention response",
        "Dynamical observable state": "Dynamical observable state",
    }
    fp = summary[summary["layer"].isin(aliases)].copy()
    fp["Layer"] = fp["layer"].map(aliases)
    fp = fp[
        [
            "Layer",
            "entropy_rank",
            "participation_rank",
            "twonn_id",
            "levina_bickel_id",
            "pc95",
            "pc99",
            "n_samples",
            "n_features_varying",
        ]
    ].rename(
        columns={
            "entropy_rank": "Entropy rank",
            "participation_rank": "Participation rank",
            "twonn_id": "TwoNN intrinsic dimension",
            "levina_bickel_id": "Levina-Bickel intrinsic dimension",
            "pc95": "PCs/singular directions for 95%",
            "pc99": "PCs/singular directions for 99%",
            "n_samples": "Samples",
            "n_features_varying": "Varying features",
        }
    )
    fp.to_csv(out_dir / "current_system_complexity_fingerprint_compact.csv", index=False)

    save_spectrum_plot(
        spectra_df,
        fig_dir / "current_system_complexity_product_observable_spectra.svg",
        "Current synthetic system: product and observable spectra",
        ["Product raw", "Product normalized", "Product slopes", "Observable cell"],
    )
    save_spectrum_plot(
        spectra_df,
        fig_dir / "current_system_complexity_hidden_metabolic_spectra.svg",
        "Current synthetic system: hidden and metabolic spectra",
        ["Hidden physiology", "Hidden physiology + capacities", "Hidden + selected fluxes", "Metabolic flux/constraint state"],
    )
    save_spectrum_plot(
        spectra_df,
        fig_dir / "current_system_complexity_dynamic_intervention_spectra.svg",
        "Current synthetic system: dynamical and intervention spectra",
        ["Dynamical product", "Dynamical observable state", "Dynamical hidden state", "Intervention response"],
    )

    print("Wrote current-system complexity audit outputs to", out_dir)
    print(fp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
