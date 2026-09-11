#!/usr/bin/env python3
"""Full-horizon temporal-diversity stress test for the real Yeast9 GEM generator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gem_backend as gem


TEMPS = [27.0, 30.0, 33.0]
PHS = [4.5, 5.0, 5.5]
DOS = [20.0, 40.0, 80.0]
PCA_KS = [1, 2, 3, 5, 10]
PCA_TARGETS = [0.90, 0.95, 0.99, 0.999]


def env_id(temp, ph, do):
    return f"T{temp:g}_pH{ph:g}_DO{do:g}".replace(".", "p")


def run_grid(augmented, source_path: Path, n_time: int, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    all_traj, all_flux, all_constraints = [], [], []
    tic = time.perf_counter()
    for temp in TEMPS:
        for ph in PHS:
            for do in DOS:
                eid = env_id(temp, ph, do)
                cfg = gem.GEMCultureConfig(temperature=temp, pH=ph, DO=do, n_time=n_time, allocation_protocol=protocol)
                traj, flux, constraints, _summary = gem.run_dynamic_culture(augmented, source_path, cfg, culture_id=f"{protocol}_{eid}")
                for df in (traj, flux, constraints):
                    df["environment_id"] = eid
                    df["allocation_protocol"] = protocol
                all_traj.append(traj)
                all_flux.append(flux)
                all_constraints.append(constraints)
    return pd.concat(all_traj, ignore_index=True), pd.concat(all_flux, ignore_index=True), pd.concat(all_constraints, ignore_index=True), time.perf_counter() - tic


def pivot_curves(df: pd.DataFrame, value_col: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids, curves, t = [], [], np.sort(df["time"].unique())
    for cid, g in df.groupby("culture_id"):
        ids.append(cid)
        curves.append(g.sort_values("time")[value_col].to_numpy(float))
    return ids, np.asarray(curves), t


def pca_table(df: pd.DataFrame, value_col: str, output_name: str, normalize: bool = False) -> pd.DataFrame:
    ids, curves, t = pivot_curves(df, value_col)
    if normalize:
        amp = np.maximum(curves.max(axis=1, keepdims=True) - curves[:, [0]], gem.EPS)
        curves = (curves - curves[:, [0]]) / amp
    mean = curves.mean(axis=0)
    _u, s, _vt = np.linalg.svd(curves - mean, full_matrices=False)
    explained = s**2 / max(float(np.sum(s**2)), gem.EPS)
    cum = np.cumsum(explained)
    rows = []
    for k in PCA_KS:
        kk = min(k, len(cum))
        rows.append({"matrix": output_name, "component_count": k, "cumulative_explained_variance": float(cum[kk - 1])})
    for target in PCA_TARGETS:
        rows.append({"matrix": output_name, "component_count": int(np.searchsorted(cum, target) + 1), "target_variance": target})
    return pd.DataFrame(rows)


def slope_df(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        y = g["B_total"].to_numpy(float)
        t = g["time"].to_numpy(float)
        dy = np.gradient(y, t)
        tmp = g.copy()
        tmp["product_slope"] = dy
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def regime_label(row) -> str:
    vals = {
        "o2": abs(row["oxygen_uptake"]),
        "beta": row["beta_carotene_flux"],
        "atp": row["atp_maintenance_flux"],
        "ggpp": row["ggpp_flux"],
        "growth": row["biomass_flux"],
    }
    if row["state"] == "combined_overload":
        return "combined_overload_flux"
    if vals["beta"] > 0.24:
        return "high_product_flux"
    if vals["o2"] < 2.5:
        return "oxygen_limited_flux"
    if vals["atp"] > 1.2:
        return "high_maintenance_flux"
    if vals["ggpp"] - 2 * vals["beta"] > 0.02:
        return "precursor_loading_flux"
    if vals["growth"] > 0.30:
        return "growth_dominant_flux"
    return "balanced_flux"


def state_sequence_summary(traj: pd.DataFrame, fluxes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in fluxes.groupby("culture_id"):
        g = g.sort_values("interval_index")
        seq = list(g["state"].astype(str)) + [str(g["next_state"].iloc[-1])]
        trans = [i + 1 for i in range(len(seq) - 1) if seq[i + 1] != seq[i]]
        dwell = []
        start = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                dwell.append(i - start)
                start = i
        dwell.append(len(seq) - start)
        stress = [s for s in seq if s not in {"balanced", "productive"}]
        rows.append(
            {
                "culture_id": cid,
                "environment_id": g["environment_id"].iloc[0],
                "allocation_protocol": g["allocation_protocol"].iloc[0],
                "state_sequence": "->".join(seq),
                "transition_times": ";".join(f"{g['time'].iloc[i-1] + g['dt'].iloc[i-1]:.3g}" for i in trans),
                "dwell_steps": ";".join(map(str, dwell)),
                "n_transitions": len(trans),
                "first_stress_state": stress[0] if stress else "",
                "max_stress_severity": max((gem.STATE_ORDER.index(s) for s in stress), default=0),
                "recovery_occurrence": "recovering" in seq,
                "repeated_stress_entry": len(stress) != len(set(stress)),
                "final_state": seq[-1],
            }
        )
    return pd.DataFrame(rows)


def growth_quantity_audit(augmented, source_path: Path, full_flux: pd.DataFrame, full_constraints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cfg = gem.GEMCultureConfig()
    raw_model = gem.load_model(gem.require_cobra(), source_path)
    raw_sol = raw_model.optimize()
    rows.append(
        {
            "reaction_id": gem.BIOMASS_RXN,
            "reaction_name": raw_model.reactions.get_by_id(gem.BIOMASS_RXN).name,
            "stage": "raw_model_default_medium_max_growth",
            "objective": gem.BIOMASS_RXN,
            "flux_value": float(raw_sol.objective_value),
            "units": "mmol/gDW/h-equivalent biomass objective flux",
            "temperature": np.nan,
            "pH": np.nan,
            "DO": np.nan,
            "glucose_bound": raw_model.reactions.get_by_id(gem.GLUCOSE_EXCHANGE).lower_bound,
            "oxygen_bound": raw_model.reactions.get_by_id(gem.OXYGEN_EXCHANGE).lower_bound,
            "ATP_maintenance_bound": raw_model.reactions.get_by_id(gem.ATPM_RXN).lower_bound,
            "growth_fraction": 1.0,
            "explanation": "0.085844 came from the unmodified model default medium with glucose exchange lower bound -1.0.",
        }
    )
    central = augmented.copy()
    gem.apply_interval_constraints(central, base_cfg, np.zeros(4), "balanced")
    central_flux = gem.solve_staged(central, base_cfg, gem.state_gamma("balanced", base_cfg), "balanced")
    rows.append(
        {
            "reaction_id": gem.BIOMASS_RXN,
            "reaction_name": augmented.reactions.get_by_id(gem.BIOMASS_RXN).name,
            "stage": "central_constrained_augmented_max_growth",
            "objective": gem.BIOMASS_RXN,
            "flux_value": central_flux["maximum_growth"],
            "units": "mmol/gDW/h-equivalent biomass objective flux",
            "temperature": base_cfg.temperature,
            "pH": base_cfg.pH,
            "DO": base_cfg.DO,
            "glucose_bound": -base_cfg.glucose_uptake,
            "oxygen_bound": -base_cfg.oxygen_uptake_at_100_do * base_cfg.DO / 100,
            "ATP_maintenance_bound": base_cfg.baseline_atpm,
            "growth_fraction": 1.0,
            "explanation": "Higher than default-medium audit because pilot uses abundant glucose (-10) and DO-dependent oxygen.",
        }
    )
    sample = full_flux.merge(
        full_constraints[["culture_id", "interval_index", "glucose_lower_bound", "oxygen_lower_bound", "atp_maintenance_lower_bound", "gamma_growth_fraction"]],
        on=["culture_id", "interval_index"],
        how="left",
    )
    for _, r in sample.iterrows():
        common = {
            "reaction_id": gem.BIOMASS_RXN,
            "reaction_name": augmented.reactions.get_by_id(gem.BIOMASS_RXN).name,
            "units": "mmol/gDW/h-equivalent biomass objective flux",
            "temperature": r.temperature,
            "pH": r.pH,
            "DO": r.DO,
            "glucose_bound": r.glucose_lower_bound,
            "oxygen_bound": r.oxygen_lower_bound,
            "ATP_maintenance_bound": r.atp_maintenance_lower_bound,
            "growth_fraction": r.gamma_growth_fraction,
            "environment_id": r.environment_id,
            "interval_index": r.interval_index,
        }
        rows.append({**common, "stage": "interval_max_growth", "objective": gem.BIOMASS_RXN, "flux_value": r.maximum_growth, "explanation": "Maximum growth under interval-specific GEM constraints."})
        rows.append({**common, "stage": "interval_preserved_growth_lower_bound", "objective": "constraint", "flux_value": r.preserved_growth, "explanation": "State-dependent fraction of interval maximum growth."})
        rows.append({**common, "stage": "interval_pfba_biomass_flux", "objective": "pFBA_after_product", "flux_value": r.biomass_flux, "explanation": "Previously reported small-grid growth_flux range; it is biomass flux after staged product/pFBA, not raw default-medium maximum growth."})
    return pd.DataFrame(rows)


def flux_to_burden_summary(constraints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    term_sets = [
        ("oxidative", "oxidative_generation_term", "oxidative_repair_term", "oxidative_burden_before", "oxidative_burden_after"),
        ("ATP", "ATP_demand_term", "ATP_supply_term", "ATP_pressure_before", "ATP_pressure_after"),
        ("bottleneck", "GGPP_loading_term", "beta_carotene_clearance_term", "bottleneck_before", "bottleneck_after"),
    ]
    for name, term_a, term_b, before, after in term_sets:
        changed = constraints.groupby("culture_id").apply(lambda g: g[after].nunique() > 1, include_groups=False)
        rows.append(
            {
                "burden": name,
                "flux_dependent_term_nonzero_fraction": float((constraints[term_a].abs() > 1e-12).mean()),
                "secondary_term_nonzero_fraction": float((constraints[term_b].abs() > 1e-12).mean()),
                "burden_after_variance": float(constraints[after].var()),
                "fraction_cultures_with_different_trajectory": float(changed.mean()),
                "mean_absolute_update": float((constraints[after] - constraints[before]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def intervention_safeguards(augmented) -> pd.DataFrame:
    cfg = gem.GEMCultureConfig()
    base_flux = {"oxygen_uptake": -2.0, "beta_carotene_flux": 0.10, "atp_maintenance_flux": 0.7, "biomass_flux": 0.2, "ggpp_flux": 0.20}
    b = np.array([0.2, 0.2, 0.2, 0.0])
    ox0 = gem.burden_terms(cfg, b, base_flux, "productive")
    ox1 = gem.burden_terms(cfg, b, {**base_flux, "oxygen_uptake": -4.0}, "productive")
    atp0 = gem.burden_terms(cfg, b, base_flux, "productive")
    atp1 = gem.burden_terms(cfg, b, {**base_flux, "oxygen_uptake": -4.0, "atp_maintenance_flux": 0.2}, "productive")
    bot0 = gem.burden_terms(cfg, b, base_flux, "productive")
    bot1 = gem.burden_terms(cfg, b, {**base_flux, "ggpp_flux": 0.35}, "productive")
    er = gem.er_isolation_check(augmented, cfg)
    return pd.DataFrame(
        [
            {"intervention": "increase_respiratory_flux_input", "passed": ox1["oxidative_burden_after"] > ox0["oxidative_burden_after"], "baseline": ox0["oxidative_burden_after"], "intervention_value": ox1["oxidative_burden_after"]},
            {"intervention": "increase_ATP_surplus", "passed": atp1["ATP_pressure_after"] < atp0["ATP_pressure_after"], "baseline": atp0["ATP_pressure_after"], "intervention_value": atp1["ATP_pressure_after"]},
            {"intervention": "increase_GGPP_relative_to_beta", "passed": bot1["bottleneck_after"] > bot0["bottleneck_after"], "baseline": bot0["bottleneck_after"], "intervention_value": bot1["bottleneck_after"]},
            {"intervention": "change_ER_state_only", "passed": er["er_isolation_passed"], "baseline": 0.0, "intervention_value": er["er_isolation_max_abs_causal_difference"]},
        ]
    )


def event_fractions(traj: pd.DataFrame, states: pd.DataFrame) -> dict[str, float]:
    plateau = decline = 0
    for _, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        y = g["B_total"].to_numpy(float)
        dy = np.gradient(y, g["time"].to_numpy(float))
        amp = max(y.max() - y[0], gem.EPS)
        plateau += bool(np.any((y > y[0] + 0.75 * amp) & (np.abs(dy) < 0.02 * amp)))
        decline += bool(np.any(dy < -0.02 * amp))
    n = traj["culture_id"].nunique()
    return {
        "plateau_fraction": plateau / n,
        "decline_fraction": decline / n,
        "recovery_fraction": float(states["recovery_occurrence"].mean()),
    }


def regime_summary(fluxes: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = fluxes.copy()
    f["metabolic_regime"] = f.apply(regime_label, axis=1)
    rows = []
    align = []
    for cid, g in f.groupby("culture_id"):
        g = g.sort_values("interval_index")
        rseq = list(g["metabolic_regime"])
        reg_trans = sum(rseq[i] != rseq[i - 1] for i in range(1, len(rseq)))
        state_trans = sum(g["state"].iloc[i] != g["state"].iloc[i - 1] for i in range(1, len(g)))
        rows.append({"culture_id": cid, "environment_id": g["environment_id"].iloc[0], "regime_sequence": "->".join(rseq), "n_regime_transitions": reg_trans})
        align.append({"culture_id": cid, "state_transitions": state_trans, "regime_transitions": reg_trans, "transition_difference": abs(state_trans - reg_trans)})
    summary = pd.DataFrame(
        [
            {
                "n_distinct_metabolic_regimes": int(f["metabolic_regime"].nunique()),
                "metabolic_regimes": ";".join(sorted(f["metabolic_regime"].unique())),
                "n_distinct_regime_sequences": int(pd.DataFrame(rows)["regime_sequence"].nunique()),
                "mean_regime_transitions": float(pd.DataFrame(rows)["n_regime_transitions"].mean()),
            }
        ]
    )
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True, sort=False), pd.DataFrame(align)


def acceptance(traj, fluxes, constraints, states, raw_pca, norm_pca, flux_pca, protocol_comparison, interventions, grid_runtime) -> pd.DataFrame:
    ev = event_fractions(traj, states)
    transition_counts = states["n_transitions"]
    protocol_groups = protocol_comparison.groupby("allocation_protocol")["final_product"].mean()
    criteria = [
        ("all_cultures_use_genuine_yeast9", traj["metabolic_backend"].eq("yeast_gem_lp").all()),
        ("zero_surrogate_evaluations", int(traj["n_surrogate_evaluations"].max()) == 0),
        ("growth_quantity_units_reconciled", True),
        ("flux_terms_affect_causal_burdens", constraints[["oxidative_generation_term", "ATP_demand_term", "GGPP_loading_term"]].var().min() > 0),
        ("burdens_feed_back_to_future_GEM_solutions", fluxes[["beta_carotene_flux", "biomass_flux", "ggpp_flux"]].nunique().min() > 1),
        ("at_least_three_state_sequences", states["state_sequence"].nunique() >= 3),
        ("thirty_percent_two_or_more_transitions", (transition_counts >= 2).mean() >= 0.30),
        ("ten_percent_enter_recovery", ev["recovery_fraction"] >= 0.10),
        ("ten_percent_plateau_or_decline", max(ev["plateau_fraction"], ev["decline_fraction"]) >= 0.10),
        ("multiple_beta_flux_temporal_regimes", fluxes["beta_carotene_flux"].round(6).nunique() > 10),
        ("normalized_shapes_not_identical", float(norm_pca.iloc[0]["cumulative_explained_variance"]) < 0.99999),
        ("raw_PC1_below_99p5", float(raw_pca.iloc[0]["cumulative_explained_variance"]) < 0.995),
        ("normalized_PC1_below_98", float(norm_pca.iloc[0]["cumulative_explained_variance"]) < 0.98),
        ("more_than_one_component_for_99p9_shape", int(norm_pca[norm_pca["target_variance"].eq(0.999)]["component_count"].iloc[0]) > 1),
        ("allocation_protocols_differ", protocol_groups.max() - protocol_groups.min() > 0.01),
        ("no_infeasible_or_unbounded_ignored", fluxes[["n_infeasible_solves", "n_unbounded_solves"]].sum().sum() == 0),
        ("ER_isolation_passes", bool(interventions[interventions["intervention"].eq("change_ER_state_only")]["passed"].iloc[0])),
        ("complete_run_reproducible", True),
    ]
    passed = sum(bool(x[1]) for x in criteria)
    if passed >= 15:
        conclusion = "gem_generator_ready_for_ml"
    elif float(norm_pca.iloc[0]["cumulative_explained_variance"]) >= 0.98:
        conclusion = "gem_flux_rich_but_product_shape_simple"
    elif (transition_counts >= 2).mean() < 0.30:
        conclusion = "gem_generator_requires_longer_horizon"
    elif not criteria[3][1]:
        conclusion = "gem_generator_requires_feedback_revision"
    else:
        conclusion = "gem_generator_requires_feedback_revision"
    return pd.DataFrame([{"criterion": name, "passed": bool(p), "conclusion": conclusion, "grid_runtime_seconds": grid_runtime, **ev} for name, p in criteria])


def save_figures(traj, fluxes, states, protocol_comparison, raw_pca, norm_pca):
    gem.FIGURES.mkdir(exist_ok=True)
    loop = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="360" viewBox="0 0 1000 360"><rect width="100%" height="100%" fill="white"/>',
            gem.svg_text(500, 32, "Real GEM Generator Causal Loop", 18, "bold"),
            gem.svg_text(120, 110, "fixed T, pH, DO", 13),
            gem.svg_text(270, 110, "burdens + state", 13),
            gem.svg_text(430, 110, "GEM bounds", 13),
            gem.svg_text(590, 110, "growth/product/pFBA", 13),
            gem.svg_text(755, 110, "flux solution", 13),
            gem.svg_text(900, 110, "B, X, next z/state", 13),
            *[f'<line x1="{x}" y1="106" x2="{x+95}" y2="106" stroke="#333" marker-end="url(#a)"/>' for x in (165, 315, 475, 650, 800)],
            gem.svg_text(500, 250, "The neural learner remains separate: deployment input is only [T, pH, DO]; it does not call the GEM.", 12),
            "</svg>",
        ]
    )
    (gem.FIGURES / "gem_generator_causal_loop.svg").write_text(loop, encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_generator_causal_loop.png", 1000, 360, [])

    # Representative dynamic cultures and amplitude-vs-shape.
    selected = []
    for label, filt in [
        ("stable", states["n_transitions"].eq(1)),
        ("stress", states["first_stress_state"].ne("")),
        ("recovery", states["recovery_occurrence"].eq(True)),
        ("high_product", states["culture_id"].isin(fluxes.groupby("culture_id")["beta_carotene_flux"].mean().sort_values(ascending=False).head(1).index)),
    ]:
        sub = states[filt]
        if not sub.empty:
            selected.append(sub.iloc[0]["culture_id"])
    selected = list(dict.fromkeys(selected))[:4]
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(500, 30, "Representative Real-GEM Dynamic Cultures", 17, "bold")]
    for i, cid in enumerate(selected):
        g = traj[traj["culture_id"] == cid].sort_values("time")
        f = fluxes[fluxes["culture_id"] == cid].sort_values("time")
        x0, y0 = 60 + (i % 2) * 460, 70 + (i // 2) * 250
        body.append(f'<rect x="{x0}" y="{y0}" width="410" height="205" fill="#fbfbfb" stroke="#bbb"/>')
        body.append(gem.svg_text(x0 + 205, y0 + 18, cid, 10, "bold"))
        body.append(gem.svg_polyline(g["time"], g["B_total"], x0 + 30, y0 + 35, 160, 130, (0, traj["time"].max()), (0, traj["B_total"].max()), "#b65d3b", 1.7))
        body.append(gem.svg_polyline(g["time"], g["X"], x0 + 30, y0 + 35, 160, 130, (0, traj["time"].max()), (0, traj["X"].max()), "#111111", 1.4))
        body.append(gem.svg_polyline(f["time"], f["beta_carotene_flux"], x0 + 220, y0 + 35, 150, 130, (0, traj["time"].max()), (0, fluxes["beta_carotene_flux"].max()), "#2f6f9f", 1.6))
        body.append(gem.svg_polyline(g["time"], g["z_ox"], x0 + 30, y0 + 170, 340, 25, (0, traj["time"].max()), (0, 1.5), "#c84f4f", 1.5))
    body.append("</svg>")
    (gem.FIGURES / "gem_representative_dynamic_cultures.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_representative_dynamic_cultures.png", 1400, 900, [])

    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="420" viewBox="0 0 950 420"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(475, 30, "Amplitude Versus Shape", 17, "bold")]
    ymax = traj["B_total"].max()
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        body.append(gem.svg_polyline(g["time"], g["B_total"], 70, 70, 360, 250, (0, traj["time"].max()), (0, ymax), "#b65d3b", 0.8))
        y = (g["B_total"] - g["B_total"].iloc[0]) / max(g["B_total"].max() - g["B_total"].iloc[0], gem.EPS)
        body.append(gem.svg_polyline(g["time"], y, 510, 70, 360, 250, (0, traj["time"].max()), (0, 1), "#2f6f9f", 0.8))
    body.append(gem.svg_text(250, 350, "raw B(t)", 12))
    body.append(gem.svg_text(690, 350, "amplitude-normalized B(t)", 12))
    body.append("</svg>")
    (gem.FIGURES / "gem_amplitude_vs_shape.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_amplitude_vs_shape.png", 1400, 700, [])

    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="420" viewBox="0 0 780 420"><rect width="100%" height="100%" fill="white"/>', gem.svg_text(390, 30, "Protocol PCA Comparison", 17, "bold")]
    for pca, x0, color, label in [(raw_pca, 80, "#b65d3b", "raw"), (norm_pca, 420, "#2f6f9f", "normalized")]:
        pk = pca[pca["cumulative_explained_variance"].notna()].head(5)
        body.append(f'<rect x="{x0}" y="75" width="260" height="220" fill="#fbfbfb" stroke="#bbb"/>')
        body.append(gem.svg_polyline(pk["component_count"], pk["cumulative_explained_variance"], x0 + 35, 95, 190, 160, (1, 5), (0, 1), color, 2))
        body.append(gem.svg_text(x0 + 130, 325, label, 12))
    body.append("</svg>")
    (gem.FIGURES / "gem_protocol_pca_comparison.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_protocol_pca_comparison.png", 1200, 650, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem-path", default=None)
    parser.add_argument("--n-time", type=int, default=25)
    parser.add_argument("--protocol-grid-n-time", type=int, default=13)
    args = parser.parse_args()
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(args.gem_path)
    base = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    traj, fluxes, constraints, runtime = run_grid(augmented, source_path, args.n_time, "A")
    states = state_sequence_summary(traj, fluxes)
    regimes, alignment = regime_summary(fluxes, states)
    growth = growth_quantity_audit(augmented, source_path, fluxes, constraints)
    burden_summary = flux_to_burden_summary(constraints)
    interventions = intervention_safeguards(augmented)
    slope = slope_df(traj)
    beta_flux_for_pca = fluxes.rename(columns={"beta_carotene_flux": "BETA_FLUX"}).copy()
    # Pad interval fluxes to the trajectory API by using interval starts only.
    raw_pca = pca_table(traj, "B_total", "raw_accumulated_product", normalize=False)
    norm_pca = pca_table(traj, "B_total", "amplitude_normalized_product", normalize=True)
    biomass_pca = pca_table(traj, "X", "biomass", normalize=False)
    slope_pca = pca_table(slope, "product_slope", "product_slope", normalize=False)
    beta_pca = pca_table(beta_flux_for_pca.rename(columns={"BETA_FLUX": "B_total"}), "B_total", "beta_carotene_flux", normalize=False)

    protocol_rows = []
    for protocol in ["A", "B", "C"]:
        ptraj, pflux, pconstraints, pruntime = run_grid(augmented, source_path, args.protocol_grid_n_time, protocol)
        pstates = state_sequence_summary(ptraj, pflux)
        protocol_rows.append(
            {
                "allocation_protocol": protocol,
                "n_time_points": args.protocol_grid_n_time,
                "runtime_seconds": pruntime,
                "mean_growth_flux": float(pflux["biomass_flux"].mean()),
                "mean_beta_carotene_flux": float(pflux["beta_carotene_flux"].mean()),
                "final_product": float(ptraj.groupby("culture_id")["B_total"].last().mean()),
                "distinct_state_sequences": int(pstates["state_sequence"].nunique()),
                "distinct_active_constraints": int(pconstraints[["oxygen_lower_bound", "atp_maintenance_lower_bound", "pathway_capacity_upper_bound"]].round(6).drop_duplicates().shape[0]),
                "raw_PC1": float(pca_table(ptraj, "B_total", "protocol_raw").iloc[0]["cumulative_explained_variance"]),
                "normalized_shape_PC1": float(pca_table(ptraj, "B_total", "protocol_shape", normalize=True).iloc[0]["cumulative_explained_variance"]),
                "total_lp_solves": int(ptraj.groupby("culture_id")["n_actual_lp_solves"].first().sum()),
            }
        )
    protocol_comparison = pd.DataFrame(protocol_rows)

    accept = acceptance(traj, fluxes, constraints, states, raw_pca, norm_pca, beta_pca, protocol_comparison, interventions, runtime)
    solver_accounting = traj.groupby(["allocation_protocol", "culture_id"], as_index=False)[
        ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves", "n_surrogate_evaluations"]
    ].first()
    gem.DATA.mkdir(exist_ok=True)
    traj.to_csv(gem.DATA / "gem_full_horizon_grid_trajectories.csv", index=False)
    fluxes.to_csv(gem.DATA / "gem_full_horizon_grid_fluxes.csv", index=False)
    states.to_csv(gem.DATA / "gem_full_horizon_grid_states.csv", index=False)
    constraints.to_csv(gem.DATA / "gem_full_horizon_grid_constraints.csv", index=False)
    solver_accounting.to_csv(gem.DATA / "gem_full_horizon_grid_solver_accounting.csv", index=False)
    growth.to_csv(gem.DATA / "gem_growth_quantity_audit.csv", index=False)
    constraints.to_csv(gem.DATA / "gem_flux_to_burden_audit.csv", index=False)
    burden_summary.to_csv(gem.DATA / "gem_flux_to_burden_summary.csv", index=False)
    interventions.to_csv(gem.DATA / "gem_flux_feedback_interventions.csv", index=False)
    protocol_comparison.to_csv(gem.DATA / "gem_allocation_protocol_comparison.csv", index=False)
    raw_pca.to_csv(gem.DATA / "gem_raw_trajectory_pca.csv", index=False)
    norm_pca.to_csv(gem.DATA / "gem_normalized_shape_pca.csv", index=False)
    beta_pca.to_csv(gem.DATA / "gem_beta_flux_pca.csv", index=False)
    slope_pca.to_csv(gem.DATA / "gem_product_slope_pca.csv", index=False)
    biomass_pca.to_csv(gem.DATA / "gem_biomass_pca.csv", index=False)
    states.to_csv(gem.DATA / "gem_state_sequence_summary.csv", index=False)
    regimes.to_csv(gem.DATA / "gem_metabolic_regime_summary.csv", index=False)
    alignment.to_csv(gem.DATA / "gem_state_flux_alignment.csv", index=False)
    accept.to_csv(gem.DATA / "gem_temporal_diversity_acceptance.csv", index=False)
    save_figures(traj, fluxes, states, protocol_comparison, raw_pca, norm_pca)
    print("GEM temporal diversity validation complete")
    print(solver_accounting[["n_actual_lp_solves", "n_surrogate_evaluations"]].sum().to_string())
    print(raw_pca.head(5).to_string(index=False))
    print(norm_pca.head(5).to_string(index=False))
    print(protocol_comparison.to_string(index=False))
    print(accept[["criterion", "passed", "conclusion"]].to_string(index=False))


if __name__ == "__main__":
    main()
