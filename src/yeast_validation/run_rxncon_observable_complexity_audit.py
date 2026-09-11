#!/usr/bin/env python3
"""Observable-complexity audit for the published yeast rxncon CDC model.

This is a regulation-only analysis.  It reuses the already compiled rxncon
BoolNet and cached perturbation panel produced by
`run_rxncon_external_complexity_screen.py`; it does not train ML, run DBTL,
invoke Yeast9/pFBA, or change rxncon rules/perturbations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_rxncon_external_complexity_screen as base  # noqa: E402


OUT_DIR = ROOT / "data/rxncon_observable_complexity_audit"
FIG_DIR = ROOT / "figures"
RNG_SEED = 20260823
N_MATCHED = base.N_MATCHED
N_TIMEPOINTS = base.N_TIMEPOINTS
EPS = 1e-12

DIRECT_PHENOTYPE_STATES = base.OBSERVABLE_STATES

MODULES: dict[str, dict[str, object]] = {
    "G1_Start_CDK": {
        "patterns": ["Cln1", "Cln2", "Cln3", "Cdc28", "Whi5", "Swi4", "Swi6", "SBF", "Cks1"],
        "interpretation": "G1 cyclin/CDK and Start transcriptional control",
        "measurement": "fluorescent protein fusions, phosphosite reporters, cyclin abundance, transcriptional reporters",
    },
    "S_phase_replication": {
        "patterns": ["Clb5", "Clb6", "Sic1", "Cdc6", "Orc", "Mcm", "Cdc45", "Dbf4", "Sld2", "Sld3"],
        "interpretation": "DNA replication licensing/origin firing and S-phase CDK control",
        "measurement": "replication markers, tagged licensing factors, phosphoproteomics, transcriptional reporters",
    },
    "G2_M_cyclins": {
        "patterns": ["Clb1", "Clb2", "Clb3", "Clb4", "Mcm1", "Fkh1", "Fkh2", "Ndd1", "Cdc5"],
        "interpretation": "G2/M cyclins, mitotic transcription, and polo kinase control",
        "measurement": "cyclin abundance, tagged transcription factors, kinase/phosphosite reporters",
    },
    "APC_exit_separase": {
        "patterns": ["APC", "Cdc20", "Cdh1", "Cdc14", "Net1", "Tem1", "Cdc15", "Dbf2", "Mob1", "Spo12", "Bfa1", "Bub2", "Pds1", "Esp1"],
        "interpretation": "APC/C activation, securin/separase, and mitotic exit network",
        "measurement": "APC/C substrate reporters, tagged Cdc14 localization, securin abundance, phosphosite reporters",
    },
    "DNA_damage_checkpoint": {
        "patterns": ["Mec1", "Rad53", "Dun1", "Chk1", "Mrc1", "Rfa1", "Rad9", "ssDNA", "dNTP"],
        "interpretation": "replication stress and DNA damage checkpoint signaling",
        "measurement": "Rad53/Mec1 pathway phosphoproteomics, ssDNA/RPA markers, checkpoint reporters",
    },
    "Spindle_checkpoint": {
        "patterns": ["Mad1", "Mad2", "Mad3", "Bub1", "Bub3", "Mps1", "Ipl1", "Ndc80", "Dam1", "Nocodazole"],
        "interpretation": "spindle assembly checkpoint and kinetochore attachment state",
        "measurement": "spindle/kinetochore imaging, checkpoint protein localization, phosphosite reporters",
    },
    "Morphogenesis_spindle_actin": {
        "patterns": ["Act1", "Bni1", "Bnr1", "Cdc42", "Bem1", "Bem2", "Bem3", "Cdc24", "Cdc3", "Cdc10", "Cdc11", "Cdc12", "Tub1", "Tub2", "Tub3", "Tub4", "Kar9", "Bim1", "Myo2", "Kip1", "Cin8", "Ase1", "LatA"],
        "interpretation": "bud polarity, actin, microtubules, spindle positioning, and cytokinetic morphology",
        "measurement": "time-lapse microscopy of tagged cytoskeleton/polarity proteins and morphology markers",
    },
    "Mating_polarity": {
        "patterns": ["Pheromone", "Far1", "Fus3", "Kss1", "Ste2", "Ste4", "Ste5", "Ste7", "Ste11", "Ste12"],
        "interpretation": "pheromone response and mating/polarity interface with cell-cycle arrest",
        "measurement": "pheromone-response reporters, Far1 abundance/phosphorylation, tagged MAPK pathway components",
    },
}

TRANSCRIPTION_DELAY_RE = re.compile(r"^\[[A-Za-z0-9]+TRSC\d+\]$")


def name_to_symbol(model: base.BoolNet) -> dict[str, str]:
    return {v: k for k, v in model.names.items()}


def state_symbols(model: base.BoolNet) -> list[str]:
    return model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()


def direct_symbols(model: base.BoolNet) -> list[str]:
    rev = name_to_symbol(model)
    return [rev[name] for name in DIRECT_PHENOTYPE_STATES if name in rev]


def module_symbols(model: base.BoolNet) -> dict[str, list[str]]:
    syms = state_symbols(model)
    result = {}
    for module, spec in MODULES.items():
        pats = spec["patterns"]
        selected = []
        for sym in syms:
            n = model.names[sym]
            if TRANSCRIPTION_DELAY_RE.match(n):
                continue
            if any(p in n for p in pats):
                selected.append(sym)
        result[module] = selected
    return result


def rich_marker_symbols(model: base.BoolNet) -> list[str]:
    selected = []
    seen = set()
    for sym in direct_symbols(model):
        selected.append(sym)
        seen.add(sym)
    for syms in module_symbols(model).values():
        for sym in syms:
            if sym not in seen:
                selected.append(sym)
                seen.add(sym)
    return selected


def build_observable_inventory(model: base.BoolNet, traj: np.ndarray) -> pd.DataFrame:
    direct = set(direct_symbols(model))
    mods = module_symbols(model)
    rows = []
    for sym in direct_symbols(model):
        idx = model.index[sym]
        rows.append(
            {
                "observable_name": model.names[sym],
                "tier": "A_conservative",
                "module": "macroscopic_CDC_output",
                "boolnet_id": sym,
                "underlying_rxncon_nodes": model.names[sym],
                "biological_interpretation": "published model global CDC/phenotype or stress/output state",
                "experimental_measurement_analogue": "time-lapse microscopy, DNA/spindle/cytokinesis assay, stress reporter, or cell-cycle phenotype call",
                "measurement_type": "direct Boolean phenotype/state trajectory",
                "publication_support": "explicit model output/global state",
                "quantity_status": "direct model state",
                "dynamic_or_endpoint": "both",
                "expected_redundancy": "high with other macroscopic CDC phase flags",
                "n_active_points": int(traj[:, :, idx].sum()),
                "varies": bool(np.nanstd(traj[:, :, idx].astype(float)) > 1e-10),
            }
        )
    for module, syms in mods.items():
        spec = MODULES[module]
        rows.append(
            {
                "observable_name": f"{module}_aggregate_activity",
                "tier": "B_systems_module",
                "module": module,
                "boolnet_id": ";".join(syms),
                "underlying_rxncon_nodes": ";".join(model.names[s] for s in syms[:20]) + (";..." if len(syms) > 20 else ""),
                "biological_interpretation": spec["interpretation"],
                "experimental_measurement_analogue": spec["measurement"],
                "measurement_type": "derived module activity from measurable marker states",
                "publication_support": "derived from explicit rxncon elemental states in published CDC model",
                "quantity_status": "derived but biologically measurable",
                "dynamic_or_endpoint": "both",
                "expected_redundancy": "moderate within module, lower across modules",
                "n_underlying_states": len(syms),
                "n_varying_underlying_states": int(sum(np.nanstd(traj[:, :, model.index[s]].astype(float)) > 1e-10 for s in syms)),
            }
        )
        for sym in syms:
            if sym in direct:
                continue
            idx = model.index[sym]
            rows.append(
                {
                    "observable_name": model.names[sym],
                    "tier": "C_rich_marker",
                    "module": module,
                    "boolnet_id": sym,
                    "underlying_rxncon_nodes": model.names[sym],
                    "biological_interpretation": spec["interpretation"],
                    "experimental_measurement_analogue": spec["measurement"],
                    "measurement_type": "multiplexed reporter/phosphoproteomic/localization/complex-state marker",
                    "publication_support": "explicit rxncon elemental state; experimental analogue depends on marker technology",
                    "quantity_status": "direct model state, experimentally plausible marker",
                    "dynamic_or_endpoint": "both",
                    "expected_redundancy": "module-correlated; audit quantifies effective rank",
                    "n_active_points": int(traj[:, :, idx].sum()),
                    "varies": bool(np.nanstd(traj[:, :, idx].astype(float)) > 1e-10),
                }
            )
    return pd.DataFrame(rows)


def signal_array(model: base.BoolNet, traj: np.ndarray, tier: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    if tier == "A_conservative":
        syms = direct_symbols(model)
        arr = traj[:, :, [model.index[s] for s in syms]].astype(float)
        names = [model.names[s] for s in syms]
        meta = pd.DataFrame({"signal": names, "module": "macroscopic_CDC_output", "kind": "direct_model_output"})
        return arr, names, meta
    if tier == "B_systems_module":
        arrays = []
        names = []
        modules = []
        # Include direct macroscopic outputs plus eight module aggregate signals.
        a_arr, a_names, _ = signal_array(model, traj, "A_conservative")
        arrays.append(a_arr)
        names.extend(a_names)
        modules.extend(["macroscopic_CDC_output"] * len(a_names))
        for module, syms in module_symbols(model).items():
            idx = [model.index[s] for s in syms]
            if not idx:
                continue
            arrays.append(traj[:, :, idx].mean(axis=2, keepdims=True).astype(float))
            names.append(f"{module}_aggregate_activity")
            modules.append(module)
        arr = np.concatenate(arrays, axis=2)
        meta = pd.DataFrame({"signal": names, "module": modules, "kind": "direct_output_or_derived_module_activity"})
        return arr, names, meta
    if tier == "C_rich_marker":
        syms = rich_marker_symbols(model)
        idx = [model.index[s] for s in syms]
        arr = traj[:, :, idx].astype(float)
        names = [model.names[s] for s in syms]
        mod_lookup = {}
        for mod, syms_mod in module_symbols(model).items():
            for sym in syms_mod:
                mod_lookup.setdefault(sym, mod)
        meta = pd.DataFrame(
            {
                "signal": names,
                "module": [mod_lookup.get(s, "macroscopic_CDC_output") for s in syms],
                "kind": ["direct_model_output" if s in set(direct_symbols(model)) else "direct_measurable_marker_state" for s in syms],
            }
        )
        return arr, names, meta
    raise ValueError(f"Unknown tier {tier}")


def temporal_features(arr: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    final = arr[:, -1, :]
    mean = arr.mean(axis=1)
    maxv = arr.max(axis=1)
    minv = arr.min(axis=1)
    tv = np.abs(np.diff(arr, axis=1)).sum(axis=1)
    thresh = 0.5
    active = arr > thresh
    first = np.full((arr.shape[0], arr.shape[2]), arr.shape[1], dtype=float)
    last = np.full((arr.shape[0], arr.shape[2]), -1.0, dtype=float)
    duration = active.sum(axis=1).astype(float)
    transitions = np.abs(np.diff(active.astype(int), axis=1)).sum(axis=1).astype(float)
    for j in range(arr.shape[2]):
        a = active[:, :, j]
        any_on = a.any(axis=1)
        first[any_on, j] = a[any_on].argmax(axis=1)
        rev = a[any_on, ::-1].argmax(axis=1)
        last[any_on, j] = arr.shape[1] - 1 - rev
    blocks = [
        ("final", final),
        ("mean", mean),
        ("max", maxv),
        ("min", minv),
        ("total_variation", tv),
        ("first_active", first),
        ("last_active", last),
        ("duration_active", duration),
        ("transition_count", transitions),
    ]
    x = np.concatenate([b for _, b in blocks], axis=1)
    cols = [f"{prefix}::{name}" for prefix, block in blocks for name in names]
    return x, cols


def representation_matrix(arr: np.ndarray, names: list[str], representation: str) -> tuple[np.ndarray, list[str]]:
    if representation == "endpoint":
        return arr[:, -1, :], [f"final::{n}" for n in names]
    if representation == "trajectory":
        cols = [f"{n}@t{t}" for n in names for t in range(arr.shape[1])]
        return arr.transpose(0, 2, 1).reshape(arr.shape[0], -1), cols
    if representation == "temporal_features":
        return temporal_features(arr, names)
    if representation == "combined":
        xt, ct = representation_matrix(arr, names, "trajectory")
        xf, cf = representation_matrix(arr, names, "temporal_features")
        return np.concatenate([xt, xf], axis=1), ct + cf
    raise ValueError(representation)


def hankel_matrix(arr: np.ndarray, lag: int = 8, max_windows: int = 2500) -> np.ndarray:
    return base.delay_embed(arr, lag=lag, max_windows=max_windows, seed=RNG_SEED)


def summarize_matched(
    model: base.BoolNet,
    panel: pd.DataFrame,
    traj: np.ndarray,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    spectra = []
    intervention_rows = []
    intervention_spectra = []
    redundancy_rows = []
    tiers = ["A_conservative", "B_systems_module", "C_rich_marker"]
    reps = ["endpoint", "trajectory", "temporal_features", "combined"]
    full_arrays = {tier: signal_array(model, traj, tier) for tier in tiers}
    for b in range(n_bootstrap):
        print(f"observable bootstrap {b + 1}/{n_bootstrap}", flush=True)
        idx = rng.choice(np.arange(traj.shape[0]), size=N_MATCHED, replace=False)
        sub_panel = panel.iloc[idx].reset_index(drop=True)
        for tier in tiers:
            full_arr, names, signal_meta = full_arrays[tier]
            arr = full_arr[idx]
            for rep in reps:
                x, cols = representation_matrix(arr, names, rep)
                row, spec = base.summarize_layer(f"{tier}::{rep}", x, compute_nonlinear=(b == 0 and rep != "combined"))
                row.update(
                    {
                        "tier": tier,
                        "representation": rep,
                        "bootstrap_id": b,
                        "n_observable_signals": arr.shape[2],
                        "n_observable_features_nominal": len(cols),
                        "effective_rank_per_nominal_feature": row["entropy_rank"] / max(len(cols), 1),
                        "pc1_variance_explained": float(spec["variance_explained"].iloc[0]) if not spec.empty else np.nan,
                    }
                )
                rows.append(row)
                if b == 0:
                    spectra.append(spec.assign(tier=tier, representation=rep))
                if rep in ("temporal_features", "combined"):
                    irow, ispec, _ = base.intervention_response(sub_panel, x, cols)
                    irow.update(
                        {
                            "tier": tier,
                            "representation": rep,
                            "bootstrap_id": b,
                            "n_observable_signals": arr.shape[2],
                            "n_response_features_nominal": len(cols),
                            "pc1_variance_explained": float(ispec["variance_explained"].iloc[0]) if not ispec.empty else np.nan,
                        }
                    )
                    intervention_rows.append(irow)
                    if b == 0:
                        intervention_spectra.append(ispec.assign(tier=tier, representation=rep))
            h = hankel_matrix(arr, lag=8, max_windows=2500 if tier != "C_rich_marker" else 1500)
            hrow, hspec = base.summarize_layer(f"{tier}::hankel_lag8", h, compute_nonlinear=(b == 0 and tier != "C_rich_marker"))
            hrow.update(
                {
                    "tier": tier,
                    "representation": "hankel_lag8",
                    "bootstrap_id": b,
                    "n_observable_signals": arr.shape[2],
                    "n_observable_features_nominal": h.shape[1],
                    "effective_rank_per_nominal_feature": hrow["entropy_rank"] / max(h.shape[1], 1),
                    "pc1_variance_explained": float(hspec["variance_explained"].iloc[0]) if not hspec.empty else np.nan,
                }
            )
            rows.append(hrow)
            if b == 0:
                spectra.append(hspec.assign(tier=tier, representation="hankel_lag8"))
            # Redundancy on trajectory channels from the matched subset.
            xtraj, _ = representation_matrix(arr, names, "trajectory")
            xtraj, kept = base.remove_constant_columns(xtraj)
            redundancy_rows.append(
                {
                    "tier": tier,
                    "bootstrap_id": b,
                    "n_observable_signals": arr.shape[2],
                    "n_varying_trajectory_features": xtraj.shape[1],
                    "feature_redundancy_fraction": 1.0 - min(xtraj.shape[0] - 1, xtraj.shape[1]) / max(xtraj.shape[1], 1),
                }
            )
    obs = pd.DataFrame(rows)
    inter = pd.DataFrame(intervention_rows)
    obs_summary = summarize_bootstrap(obs)
    inter_summary = summarize_bootstrap(inter)
    return obs, obs_summary, pd.concat(spectra, ignore_index=True), inter, inter_summary, pd.concat(intervention_spectra, ignore_index=True), pd.DataFrame(redundancy_rows)


def summarize_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["tier", "representation"]
    metrics = [
        "entropy_rank",
        "participation_rank",
        "pc90",
        "pc95",
        "pc99",
        "pc999",
        "pc1_variance_explained",
        "twonn_id",
        "levina_bickel_id",
        "effective_rank_per_nominal_feature",
    ]
    rows = []
    for key, g in df.groupby(keys):
        rec = {"tier": key[0], "representation": key[1], "n_bootstrap": int(g.shape[0])}
        for fixed in ["n_observable_signals", "n_observable_features_nominal", "n_features_varying"]:
            if fixed in g.columns:
                vals = g[fixed].dropna()
                if not vals.empty:
                    rec[fixed] = float(np.median(vals))
        for metric in metrics:
            if metric not in g.columns:
                continue
            vals = g[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rec[f"{metric}_median"] = float(np.median(vals))
                rec[f"{metric}_lo"] = float(np.percentile(vals, 2.5))
                rec[f"{metric}_hi"] = float(np.percentile(vals, 97.5))
        rows.append(rec)
    return pd.DataFrame(rows)


def trajectory_diversity(model: base.BoolNet, traj: np.ndarray) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for tier in ["A_conservative", "B_systems_module", "C_rich_marker"]:
        arr, names, _ = signal_array(model, traj, tier)
        endpoint, _ = representation_matrix(arr, names, "endpoint")
        trajectory, _ = representation_matrix(arr, names, "trajectory")
        temporal, _ = representation_matrix(arr, names, "temporal_features")
        endpoint_v, _ = base.remove_constant_columns(endpoint)
        traj_v, _ = base.remove_constant_columns(trajectory)
        temp_v, _ = base.remove_constant_columns(temporal)
        sample_idx = rng.choice(np.arange(arr.shape[0]), size=min(500, arr.shape[0]), replace=False)
        ez = base.standardize_matrix(endpoint_v[sample_idx]) if endpoint_v.shape[1] else np.empty((len(sample_idx), 0))
        tz = base.standardize_matrix(traj_v[sample_idx]) if traj_v.shape[1] else np.empty((len(sample_idx), 0))
        if ez.shape[1] and tz.shape[1]:
            ed = pdist(ez)
            td = pdist(tz)
            pear = pearsonr(ed, td).statistic if np.nanstd(ed) > EPS and np.nanstd(td) > EPS else np.nan
            spear = spearmanr(ed, td).statistic if np.nanstd(ed) > EPS and np.nanstd(td) > EPS else np.nan
            same_endpoint_different_traj = float(np.mean((ed <= np.percentile(ed, 10)) & (td >= np.percentile(td, 90))))
            z = tz
            link = linkage(z, method="average", metric="euclidean") if z.shape[0] > 2 else None
            clusters = len(np.unique(fcluster(link, t=np.percentile(td, 50), criterion="distance"))) if link is not None else np.nan
        else:
            pear = spear = same_endpoint_different_traj = clusters = np.nan
        endpoint_unique = np.unique(endpoint.round(6), axis=0).shape[0] if endpoint.shape[1] else 0
        trajectory_unique = np.unique(trajectory.round(6), axis=0).shape[0] if trajectory.shape[1] else 0
        temporal_unique = np.unique(temporal.round(6), axis=0).shape[0] if temporal.shape[1] else 0
        rows.append(
            {
                "tier": tier,
                "n_trajectories": arr.shape[0],
                "n_signals": arr.shape[2],
                "unique_endpoint_patterns": int(endpoint_unique),
                "unique_trajectory_patterns": int(trajectory_unique),
                "unique_temporal_feature_patterns": int(temporal_unique),
                "endpoint_vs_trajectory_pearson_distance_corr": pear,
                "endpoint_vs_trajectory_spearman_distance_corr": spear,
                "fraction_pairs_same_endpoint_decile_different_trajectory_decile": same_endpoint_different_traj,
                "trajectory_family_clusters_at_median_distance": clusters,
            }
        )
    # Internal attractor count for context, not an observable claim.
    state_syms = model.symbols.loc[model.symbols["kind"].eq("S"), "boolnet_id"].tolist()
    state_idx = [model.index[s] for s in state_syms if model.names[s] not in base.EXTERNAL_INPUTS]
    rows.append(
        {
            "tier": "internal_context_not_observable_claim",
            "n_trajectories": traj.shape[0],
            "n_signals": len(state_idx),
            "unique_endpoint_patterns": int(np.unique(traj[:, -1, state_idx], axis=0).shape[0]),
            "unique_trajectory_patterns": int(np.unique(traj[:, :, state_idx].reshape(traj.shape[0], -1), axis=0).shape[0]),
        }
    )
    return pd.DataFrame(rows)


def previous_collapse_diagnosis() -> pd.DataFrame:
    prev = pd.read_csv(ROOT / "data/rxncon_external_complexity_screen/rxncon_complexity_representative_matched_subset.csv")
    spectra = pd.read_csv(ROOT / "data/rxncon_external_complexity_screen/rxncon_complexity_spectra.csv")
    rows = []
    for layer in ["Observable/phenotypic state", "Dynamical observable state", "Intervention response"]:
        g = prev[prev["layer"].eq(layer)]
        s = spectra[spectra["layer"].eq(layer)]
        if g.empty:
            continue
        rows.append(
            {
                "previous_layer": layer,
                "n_samples": int(g["n_samples"].iloc[0]),
                "n_features_varying": int(g["n_features_varying"].iloc[0]),
                "entropy_rank": float(g["entropy_rank"].iloc[0]),
                "participation_rank": float(g["participation_rank"].iloc[0]),
                "pc95": float(g["pc95"].iloc[0]),
                "pc99": float(g["pc99"].iloc[0]),
                "pc1_variance_explained": float(s["variance_explained"].iloc[0]) if not s.empty else np.nan,
                "diagnosis": "macroscopic endpoint/phase flags are synchronized and redundant under the panel",
            }
        )
    return pd.DataFrame(rows)


def gate_decision(obs_summary: pd.DataFrame, inter_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in obs_summary.iterrows():
        if r["representation"] not in ("trajectory", "temporal_features", "combined", "hankel_lag8"):
            continue
        rows.append(
            {
                "tier": r["tier"],
                "representation": r["representation"],
                "observable_entropy": r.get("entropy_rank_median", np.nan),
                "observable_pr": r.get("participation_rank_median", np.nan),
                "observable_strong": bool(r.get("entropy_rank_median", 0) >= 10),
                "observable_acceptable": bool(r.get("entropy_rank_median", 0) > 6),
            }
        )
    gate = pd.DataFrame(rows)
    best_inter = inter_summary[inter_summary["representation"].isin(["temporal_features", "combined"])].copy()
    best_inter = best_inter.rename(
        columns={
            "entropy_rank_median": "intervention_entropy",
            "participation_rank_median": "intervention_pr",
        }
    )[["tier", "representation", "intervention_entropy", "intervention_pr"]]
    gate = gate.merge(best_inter, on=["tier", "representation"], how="left")
    gate["intervention_strong"] = gate["intervention_entropy"] >= 5
    gate["intervention_acceptable"] = gate["intervention_entropy"] > 4
    gate["strong_success"] = gate["observable_strong"] & gate["intervention_strong"]
    gate["acceptable_success"] = gate["observable_acceptable"] & gate["intervention_acceptable"]
    return gate


def save_plots(obs_spectra: pd.DataFrame, inter_spectra: pd.DataFrame, obs_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(8.5, 5.5))
    for tier in ["A_conservative", "B_systems_module", "C_rich_marker"]:
        for rep, style in [("trajectory", "-"), ("temporal_features", "--"), ("hankel_lag8", ":")]:
            g = obs_spectra[(obs_spectra["tier"].eq(tier)) & (obs_spectra["representation"].eq(rep))]
            if g.empty:
                continue
            plt.plot(g["pc"], g["cumulative_variance"], linestyle=style, linewidth=1.2, label=f"{tier}:{rep}")
    plt.axhline(0.95, color="#777", linestyle="--", linewidth=0.8)
    plt.axhline(0.99, color="#999", linestyle=":", linewidth=0.8)
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance")
    plt.ylim(0, 1.01)
    plt.title("rxncon observable tiers: matched-subset spectra")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_observable_tier_spectra.svg")
    plt.close()

    plot = obs_summary[obs_summary["representation"].isin(["endpoint", "trajectory", "temporal_features", "hankel_lag8"])].copy()
    plot["label"] = plot["tier"] + "\n" + plot["representation"]
    x = np.arange(plot.shape[0])
    med = plot["entropy_rank_median"].to_numpy(float)
    lo = plot["entropy_rank_lo"].to_numpy(float)
    hi = plot["entropy_rank_hi"].to_numpy(float)
    plt.figure(figsize=(10, 5.2))
    plt.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", capsize=3)
    plt.axhline(6, color="#b44", linestyle="--", linewidth=0.9, label="acceptable observable gate")
    plt.axhline(10, color="#844", linestyle=":", linewidth=0.9, label="strong observable gate")
    plt.xticks(x, plot["label"], rotation=45, ha="right", fontsize=7)
    plt.ylabel("Entropy effective rank")
    plt.title("rxncon observable complexity by biologically plausible tier")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rxncon_observable_tier_bootstrap.svg")
    plt.close()

    if not inter_spectra.empty:
        plt.figure(figsize=(8, 5))
        for tier in ["A_conservative", "B_systems_module", "C_rich_marker"]:
            for rep, style in [("temporal_features", "-"), ("combined", "--")]:
                g = inter_spectra[(inter_spectra["tier"].eq(tier)) & (inter_spectra["representation"].eq(rep))]
                if g.empty:
                    continue
                plt.plot(g["pc"], g["cumulative_variance"], linestyle=style, linewidth=1.4, label=f"{tier}:{rep}")
        plt.axhline(0.95, color="#777", linestyle="--", linewidth=0.8)
        plt.axhline(0.99, color="#999", linestyle=":", linewidth=0.8)
        plt.xlabel("Intervention-response singular direction")
        plt.ylabel("Cumulative response variance")
        plt.ylim(0, 1.01)
        plt.title("rxncon observable-tier intervention-response spectra")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "rxncon_observable_intervention_response_spectra.svg")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = base.MODELS_DIR / "CDC_S_cerevisiae_perturbable"
    model = base.load_boolnet(prefix)
    panel = pd.read_csv(base.OUT_DIR / "rxncon_perturbation_panel.csv")
    traj = np.load(base.OUT_DIR / "rxncon_trajectory_panel_bool.npz")["traj"].astype(bool)

    inventory = build_observable_inventory(model, traj)
    inventory.to_csv(OUT_DIR / "rxncon_candidate_observable_inventory.csv", index=False)
    previous_collapse_diagnosis().to_csv(OUT_DIR / "rxncon_previous_observable_collapse_diagnosis.csv", index=False)

    obs, obs_summary, obs_spectra, inter, inter_summary, inter_spectra, redundancy = summarize_matched(
        model, panel, traj, args.n_bootstrap
    )
    obs.to_csv(OUT_DIR / "rxncon_observable_complexity_matched_bootstrap.csv", index=False)
    obs_summary.to_csv(OUT_DIR / "rxncon_observable_complexity_summary.csv", index=False)
    obs_spectra.to_csv(OUT_DIR / "rxncon_observable_complexity_spectra.csv", index=False)
    inter.to_csv(OUT_DIR / "rxncon_observable_intervention_response_bootstrap.csv", index=False)
    inter_summary.to_csv(OUT_DIR / "rxncon_observable_intervention_response_summary.csv", index=False)
    inter_spectra.to_csv(OUT_DIR / "rxncon_observable_intervention_response_spectra.csv", index=False)
    redundancy.to_csv(OUT_DIR / "rxncon_observable_redundancy.csv", index=False)
    diversity = trajectory_diversity(model, traj)
    diversity.to_csv(OUT_DIR / "rxncon_trajectory_family_diversity.csv", index=False)
    gate = gate_decision(obs_summary, inter_summary)
    gate.to_csv(OUT_DIR / "rxncon_observable_complexity_gate.csv", index=False)
    signal_rows = []
    for tier in ["A_conservative", "B_systems_module", "C_rich_marker"]:
        arr, names, meta = signal_array(model, traj, tier)
        signal_rows.append(
            {
                "tier": tier,
                "n_signals": arr.shape[2],
                "n_varying_signals": int(sum(np.nanstd(arr[:, :, j]) > 1e-10 for j in range(arr.shape[2]))),
                "median_pairwise_signal_abs_corr": median_signal_abs_corr(arr),
            }
        )
        meta.to_csv(OUT_DIR / f"rxncon_{tier}_signals.csv", index=False)
    pd.DataFrame(signal_rows).to_csv(OUT_DIR / "rxncon_observable_tier_signal_counts.csv", index=False)
    save_plots(obs_spectra, inter_spectra, obs_summary)

    print("rxncon observable-complexity audit complete")
    print(obs_summary.to_string(index=False))
    print(inter_summary.to_string(index=False))
    print(gate.to_string(index=False))
    print(diversity.to_string(index=False))


def median_signal_abs_corr(arr: np.ndarray) -> float:
    x = arr.reshape(-1, arr.shape[2])
    x, _ = base.remove_constant_columns(x)
    if x.shape[1] < 2:
        return np.nan
    c = np.corrcoef(x, rowvar=False)
    tri = np.triu_indices_from(c, k=1)
    vals = np.abs(c[tri])
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


if __name__ == "__main__":
    main()
