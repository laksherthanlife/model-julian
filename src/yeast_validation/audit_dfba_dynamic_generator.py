#!/usr/bin/env python3
"""Audit dynamic metabolic behavior in Experiments 3A-3F."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import run_fixed_environment_validation as fixed


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TOL = 1e-6
STATE_ORDER = [
    "balanced",
    "productive",
    "oxidative_stress",
    "energy_limited",
    "pathway_limited",
    "combined_overload",
    "recovering",
]


def ensure_backend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "metabolic_backend" not in out.columns:
        out["metabolic_backend"] = "reduced_stoichiometric_surrogate"
    return out


def adjacent_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) <= 1:
        return {"unique": len(values), "mean_change": 0.0, "max_change": 0.0, "fraction_constant": 1.0}
    diff = np.abs(np.diff(values))
    return {
        "unique": int(len(np.unique(np.round(values / TOL).astype(int)))),
        "mean_change": float(diff.mean()),
        "max_change": float(diff.max()),
        "fraction_constant": float((diff <= TOL).mean()),
    }


def active_flags(flux: pd.DataFrame) -> pd.DataFrame:
    out = flux.copy()
    precursor_limit = np.maximum(0.0, 0.62 * out["v_precursor"] - 0.18 * out["biomass_flux"])
    out["beta_capacity_active"] = np.isclose(out["v_beta"], out["beta_upper_bound"], atol=1e-5, rtol=1e-4)
    out["precursor_capacity_active"] = np.isclose(out["v_beta"], precursor_limit, atol=1e-5, rtol=1e-4)
    out["minimum_biomass_active"] = np.isclose(out["biomass_flux"], out["gamma_growth_fraction"] * out["mu_max"], atol=1e-8, rtol=1e-6)
    out["oxygen_uptake_active"] = np.isclose(out["v_o2"], out["oxygen_uptake_bound"], atol=1e-8, rtol=1e-6)
    out["atp_maintenance_active"] = out["rho_atp"] >= 1.0
    out["growth_capacity_active"] = np.isclose(out["mu_max"], out["growth_capacity"], atol=1e-8, rtol=1e-6)
    return out


def build_dynamic_audit(traj: pd.DataFrame, flux: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flux = active_flags(flux)
    active_cols = [
        "beta_capacity_active",
        "precursor_capacity_active",
        "minimum_biomass_active",
        "oxygen_uptake_active",
        "atp_maintenance_active",
        "growth_capacity_active",
    ]
    for cid, g in flux.groupby("culture_id"):
        g = g.sort_values("time")
        trow = traj[traj["culture_id"] == cid].sort_values("time")
        beta = adjacent_stats(g["v_beta"].to_numpy(float))
        growth = adjacent_stats(g["biomass_flux"].to_numpy(float))
        precursor = adjacent_stats(g["v_precursor"].to_numpy(float))
        state_changes = int((g["state"].astype(str) != g["state"].astype(str).shift()).sum() - 1)
        active_pattern = g[active_cols].astype(int).astype(str).agg("".join, axis=1)
        regime_changes = int((active_pattern != active_pattern.shift()).sum() - 1 + state_changes)
        rows.append(
            {
                "culture_id": cid,
                "experiment": g["experiment"].iloc[0],
                "environment_id": g["environment_id"].iloc[0],
                "split": g["split"].iloc[0],
                "temperature": float(g["temperature"].iloc[0]),
                "pH": float(g["pH"].iloc[0]),
                "DO": float(g["DO"].iloc[0]),
                "n_time_points": int(g["n_time_points"].iloc[0]),
                "n_simulation_intervals": int(g["n_simulation_intervals"].iloc[0]),
                "dt": float(g["dt"].iloc[0]),
                "n_metabolic_updates": int(g["n_metabolic_calculations_total"].iloc[0]),
                "n_actual_fba_solves": int(g["n_actual_fba_solves_total"].iloc[0]),
                "n_surrogate_evaluations": int(g["n_surrogate_evaluations_total"].iloc[0]),
                "one_optimization_or_surrogate_per_time_interval": bool(g["n_metabolic_calculations_total"].iloc[0] >= g["n_simulation_intervals"].iloc[0]),
                "flux_solutions_reused_or_cached": bool(g["flux_solution_reused_or_cached"].any()),
                "n_state_transitions": int(trow["transition_count"].iloc[0]),
                "n_metabolic_regime_changes": regime_changes,
                "n_unique_beta_fluxes": beta["unique"],
                "n_unique_growth_fluxes": growth["unique"],
                "n_unique_precursor_fluxes": precursor["unique"],
                "mean_beta_flux_change": beta["mean_change"],
                "max_beta_flux_change": beta["max_change"],
                "fraction_constant_beta_flux": beta["fraction_constant"],
                "mean_growth_flux_change": growth["mean_change"],
                "max_growth_flux_change": growth["max_change"],
                "fraction_constant_growth_flux": growth["fraction_constant"],
                "mean_precursor_flux_change": precursor["mean_change"],
                "max_precursor_flux_change": precursor["max_change"],
                "fraction_constant_precursor_flux": precursor["fraction_constant"],
                "initial_state": str(g["state"].iloc[0]),
                "final_state": str(g["state"].iloc[-1]),
                "metabolic_backend": str(g["metabolic_backend"].iloc[0]),
                "has_plateau_or_decline": bool((np.diff(trow["B_total_true"].to_numpy(float)) <= TOL).any()),
            }
        )
    return pd.DataFrame(rows)


def build_constraint_audit(flux: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flux = active_flags(flux)
    specs = [
        ("beta_carotene_pathway_capacity", "beta_upper_bound", "v_beta", "beta_capacity_active", "z_bottle/state"),
        ("minimum_biomass_requirement", "gamma_growth_fraction", "biomass_flux", "minimum_biomass_active", "discrete state"),
        ("oxygen_uptake_capacity", "oxygen_uptake_bound", "v_o2", "oxygen_uptake_active", "DO"),
        ("ATP_maintenance_requirement", "v_atpm", "rho_atp", "atp_maintenance_active", "pH/DO/state/burdens"),
        ("precursor_capacity", "precursor_capacity", "v_precursor", "precursor_capacity_active", "T/pH/DO"),
        ("growth_capacity", "growth_capacity", "mu_max", "growth_capacity_active", "T/pH/DO/causal burdens"),
    ]
    detailed = []
    summary = []
    for name, bound_col, flux_col, active_col, driver in specs:
        rows = flux[
            [
                "culture_id",
                "experiment",
                "environment_id",
                "split",
                "temperature",
                "pH",
                "DO",
                "time",
                "state",
                "metabolic_backend",
                bound_col,
                flux_col,
                active_col,
            ]
        ].copy()
        rows = rows.rename(columns={bound_col: "configured_or_effective_bound", flux_col: "solved_flux_or_proxy", active_col: "constraint_active"})
        rows["constraint"] = name
        rows["changed_by"] = driver
        detailed.append(rows)

        per_culture = rows.groupby("culture_id")
        bound_change = per_culture["configured_or_effective_bound"].agg(lambda x: float(np.abs(np.diff(x.to_numpy(float))).mean()) if len(x) > 1 else 0.0)
        flux_change = per_culture["solved_flux_or_proxy"].agg(lambda x: float(np.abs(np.diff(x.to_numpy(float))).mean()) if len(x) > 1 else 0.0)
        ever_changes = per_culture["configured_or_effective_bound"].agg(lambda x: bool(np.ptp(x.to_numpy(float)) > TOL))
        solution_altered = (bound_change > TOL) & (flux_change > TOL)
        summary.append(
            {
                "constraint": name,
                "changed_by": driver,
                "fraction_cultures_ever_changes": float(ever_changes.mean()),
                "fraction_time_intervals_active": float(rows["constraint_active"].mean()),
                "typical_absolute_bound_change": float(bound_change.median()),
                "fraction_cultures_where_bound_change_alters_solution": float(solution_altered.mean()),
                "metabolic_backend": str(rows["metabolic_backend"].iloc[0]),
            }
        )
    return pd.DataFrame(summary), pd.concat(detailed, ignore_index=True)


def build_state_audit(traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    one = traj.drop_duplicates("culture_id")
    dynamic = traj[traj["experiment"] != "3A"]
    dist = one[one["experiment"] != "3A"]["transition_count"].value_counts().sort_index()
    transition_rows = []
    for bucket, mask in [
        ("0 transitions", one[one["experiment"] != "3A"]["transition_count"] == 0),
        ("1 transition", one[one["experiment"] != "3A"]["transition_count"] == 1),
        ("2 transitions", one[one["experiment"] != "3A"]["transition_count"] == 2),
        ("3 or more transitions", one[one["experiment"] != "3A"]["transition_count"] >= 3),
    ]:
        transition_rows.append({"transition_bucket": bucket, "n_cultures": int(mask.sum()), "fraction_cultures": float(mask.mean())})
    transition_distribution = pd.DataFrame(transition_rows)
    occupancy = (
        dynamic["state"]
        .value_counts(normalize=True)
        .reindex(STATE_ORDER, fill_value=0.0)
        .rename_axis("state")
        .reset_index(name="occupancy_fraction")
    )
    pairs = []
    dwell_rows = []
    resumed = []
    for cid, g in dynamic.groupby("culture_id"):
        states = g.sort_values("time")["state"].astype(str).to_list()
        times = g.sort_values("time")["time"].to_numpy(float)
        if len(states) > 1:
            for a, b in zip(states[:-1], states[1:]):
                if a != b:
                    pairs.append({"from_state": a, "to_state": b, "count": 1})
        start = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                dwell_rows.append({"culture_id": cid, "state": states[start], "dwell_time": float(times[i - 1] - times[start])})
                start = i
        stress_seen = any(s in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"} for s in states)
        resume_after_stress = False
        if stress_seen:
            stress_indices = [i for i, s in enumerate(states) if s in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"}]
            resume_after_stress = any(s == "productive" for s in states[max(stress_indices) + 1 :])
        resumed.append({"culture_id": cid, "resumed_production_after_stress": resume_after_stress})
    matrix = pd.DataFrame(pairs).groupby(["from_state", "to_state"], as_index=False)["count"].sum() if pairs else pd.DataFrame(columns=["from_state", "to_state", "count"])
    special = pd.DataFrame(
        [
            {"quantity": "average_dwell_time", "value": float(pd.DataFrame(dwell_rows)["dwell_time"].mean()) if dwell_rows else 0.0},
            {"quantity": "cultures_never_leave_balanced", "value": int((dynamic.groupby("culture_id")["state"].nunique() == 1).loc[lambda x: x.index.map(lambda cid: dynamic[dynamic.culture_id == cid]["state"].iloc[0] == "balanced")].sum()) if not dynamic.empty else 0},
            {"quantity": "cultures_permanently_productive", "value": int(dynamic.groupby("culture_id")["state"].agg(lambda x: (x == "productive").all()).sum())},
            {"quantity": "cultures_entering_combined_overload", "value": int(dynamic.groupby("culture_id")["state"].agg(lambda x: (x == "combined_overload").any()).sum())},
            {"quantity": "cultures_entering_recovering", "value": int(dynamic.groupby("culture_id")["state"].agg(lambda x: (x == "recovering").any()).sum())},
            {"quantity": "cultures_resuming_production_after_stress", "value": int(pd.DataFrame(resumed)["resumed_production_after_stress"].sum()) if resumed else 0},
        ]
    )
    return transition_distribution, occupancy, matrix, special


def acceptance_criteria(dynamic_audit: pd.DataFrame, constraint_audit: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    dyn = dynamic_audit[dynamic_audit["experiment"] != "3A"]
    sequences = traj[traj["experiment"] != "3A"].groupby("culture_id")["state"].agg(lambda x: "->".join(pd.Series(x).loc[lambda s: s != s.shift()].astype(str)))
    near_opt = dyn[(dyn["temperature"].between(28, 32)) & (dyn["pH"].between(4.7, 5.3)) & (dyn["DO"].between(35, 55))]
    off = dyn[~dyn.index.isin(near_opt.index)]
    beta_active = constraint_audit[constraint_audit["constraint"] == "beta_carotene_pathway_capacity"]["fraction_time_intervals_active"].iloc[0]
    atp_active = constraint_audit[constraint_audit["constraint"] == "ATP_maintenance_requirement"]["fraction_time_intervals_active"].iloc[0]
    rows = [
        ("at_least_30pct_dynamic_cultures_have_two_or_more_transitions", float((dyn["n_state_transitions"] >= 2).mean()), 0.30, ">="),
        ("at_least_10pct_dynamic_cultures_have_three_or_more_transitions", float((dyn["n_state_transitions"] >= 3).mean()), 0.10, ">="),
        ("at_least_three_distinct_state_sequences", float(sequences.nunique()), 3.0, ">="),
        ("at_least_25pct_show_substantial_beta_flux_change", float((dyn["max_beta_flux_change"] >= 0.02).mean()), 0.25, ">="),
        ("at_least_10pct_show_partial_recovery_after_stress", float(dyn["final_state"].eq("recovering").mean() + traj[traj["experiment"] != "3A"].groupby("culture_id")["state"].agg(lambda x: (x == "recovering").any()).mean()), 0.10, ">="),
        ("beta_capacity_constraint_active_meaningfully", float(beta_active), 0.05, ">="),
        ("ATP_maintenance_active_under_some_conditions", float(atp_active), 0.05, ">="),
        ("some_cultures_show_plateau_or_decline", float(dyn["has_plateau_or_decline"].mean()), 0.05, ">="),
        ("near_optimal_cultures_remain_comparatively_productive", float(near_opt["max_beta_flux_change"].median() < off["max_beta_flux_change"].median()) if len(near_opt) and len(off) else 0.0, 1.0, "=="),
        ("3A_and_3B_distinguishable_by_transition_counts", float(dynamic_audit[dynamic_audit["experiment"].eq("3B")]["n_state_transitions"].mean() - dynamic_audit[dynamic_audit["experiment"].eq("3A")]["n_state_transitions"].mean()), 0.5, ">="),
        ("dynamic_behavior_differs_across_fixed_environments", float(dyn.groupby(["temperature", "pH", "DO"])["n_state_transitions"].mean().nunique()), 3.0, ">="),
        ("integration_stable_no_nan_or_infeasible_states", float(np.isfinite(traj[["X", "B_total_true", "z_ox", "z_atp", "z_bottle", "z_er"]]).all().all()), 1.0, "=="),
    ]
    out = []
    for name, value, target, op in rows:
        passed = value >= target if op == ">=" else abs(value - target) < 1e-12
        out.append({"criterion": name, "observed_value": value, "target": target, "operator": op, "passed": bool(passed)})
    return pd.DataFrame(out)


def write_png(path: Path, width: int, height: int, lines: list[tuple[list[float], list[float], tuple[float, float], tuple[float, float], tuple[int, int, int]]]) -> None:
    pixels = bytearray([255, 255, 255] * width * height)

    def set_px(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def line(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            for ox in (0, 1):
                for oy in (0, 1):
                    set_px(x1 + ox, y1 + oy, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    for x, y, xlim, ylim, color in lines:
        pts = []
        for xi, yi in zip(x, y):
            px = int((xi - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * (width - 120) + 60)
            py = int(height - 60 - (yi - ylim[0]) / max(ylim[1] - ylim[0], 1e-9) * (height - 120))
            pts.append((px, py))
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            line(x1, y1, x2, y2, color)
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def save_svg_and_png(path: Path, width: int, height: int, body: str, png_lines: list) -> None:
    fixed.save_svg(path, width, height, body)
    write_png(path.with_suffix(".png"), width * 2, height * 2, png_lines)


def make_figures(traj: pd.DataFrame, flux: pd.DataFrame, dynamic_audit: pd.DataFrame, constraint_audit: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    dyn = dynamic_audit[dynamic_audit["experiment"] != "3A"].sort_values(["n_state_transitions", "max_beta_flux_change"], ascending=False)
    cid = dyn.iloc[0]["culture_id"]
    t = traj[traj["culture_id"] == cid].sort_values("time")
    f = flux[flux["culture_id"] == cid].sort_values("time")
    xlim = (float(t["time"].min()), float(t["time"].max()))
    body = [fixed.svg_text(520, 30, "Dynamic Audit: Single Culture Simulation Loop", 18, "bold")]
    png_lines = []
    panels = [
        ("B(t), X(t)", ["B_total_true", "X"], t, ["#111", "#3f6f9e"]),
        ("Fluxes", ["v_beta", "v_precursor", "biomass_flux"], f, ["#111", "#4c8b57", "#b65d3b"]),
        ("Burdens", ["z_ox", "z_atp", "z_bottle"], t, ["#b65d3b", "#3f6f9e", "#4c8b57"]),
        ("Bounds", ["beta_upper_bound", "v_atpm", "gamma_growth_fraction", "oxygen_uptake_bound"], f, ["#111", "#7b5bb7", "#b65d3b", "#4c8b57"]),
        ("State index", ["state_index"], t.assign(state_index=t["state"].map({s: i for i, s in enumerate(STATE_ORDER)})), ["#111"]),
    ]
    for pi, (title, cols, df, colors) in enumerate(panels):
        x0, y0, w, h = 70, 65 + pi * 100, 870, 72
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(fixed.svg_text(x0 - 8, y0 + 40, title, 9, anchor="end"))
        ymax = max(float(df[c].max()) for c in cols) * 1.1 + 1e-9
        for col, color in zip(cols, colors):
            body.append(fixed.plot_polyline_bounds(df["time"], df[col], x0 + 20, y0 + 8, w - 45, h - 20, xlim, (0, ymax), color, 1.8, ""))
            png_lines.append((df["time"].to_list(), df[col].to_list(), xlim, (0, ymax), rgb(color)))
        for tt in t.loc[t["state"] != t["state"].shift(), "time"].iloc[1:]:
            px = x0 + 20 + (float(tt) - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * (w - 45)
            body.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + h}" stroke="#c44" stroke-width="1" stroke-dasharray="4 3"/>')
    save_svg_and_png(FIGURES / "dfba_dynamic_audit_single_culture.svg", 1040, 590, "\n".join(body), png_lines)

    choices = dyn.head(3)["culture_id"].to_list()
    body = [fixed.svg_text(520, 30, "Production Flux versus Accumulated Product", 18, "bold")]
    png_lines = []
    for idx, cc in enumerate(choices):
        tt = traj[traj["culture_id"] == cc].sort_values("time")
        ff = flux[flux["culture_id"] == cc].sort_values("time")
        deriv = np.gradient(tt["B_total_true"].to_numpy(float), tt["time"].to_numpy(float))
        x0, y0, w, h = 60 + idx * 330, 75, 285, 235
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        ymax = max(float(tt["B_total_true"].max()), float(ff["v_beta"].max()), float(np.max(deriv)), 1e-8) * 1.1
        for y, color in [(tt["B_total_true"], "#111"), (ff["v_beta"], "#3f6f9e"), (deriv, "#b65d3b")]:
            body.append(fixed.plot_polyline_bounds(tt["time"], y, x0 + 30, y0 + 20, w - 55, h - 55, xlim, (0, ymax), color, 1.8))
            png_lines.append((tt["time"].to_list(), list(y), xlim, (0, ymax), rgb(color)))
        body.append(fixed.svg_text(x0 + w / 2, y0 + h - 13, "black=B(t), blue=v_beta, red=dB/dt", 8))
    save_svg_and_png(FIGURES / "dfba_flux_vs_accumulated_product.svg", 1040, 355, "\n".join(body), png_lines)

    env_id = str(dyn.iloc[0]["environment_id"])
    body = [fixed.svg_text(520, 30, "Experiment 3A versus 3B Mechanism", 18, "bold")]
    png_lines = []
    for idx, exp in enumerate(["3A", "3B"]):
        cc = traj[(traj["experiment"] == exp) & (traj["environment_id"] == env_id)]["culture_id"].iloc[0]
        tt = traj[traj["culture_id"] == cc].sort_values("time")
        ff = flux[flux["culture_id"] == cc].sort_values("time")
        x0, y0, w, h = 70 + idx * 480, 75, 410, 250
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(fixed.svg_text(x0 + w / 2, y0 - 9, exp, 13, "bold"))
        ymax = max(float(tt["B_total_true"].max()), float(ff["v_beta"].max()), float(tt["X"].max()), 1e-8) * 1.1
        for y, color in [(tt["B_total_true"], "#111"), (ff["v_beta"], "#3f6f9e"), (tt["X"], "#4c8b57")]:
            body.append(fixed.plot_polyline_bounds(tt["time"], y, x0 + 35, y0 + 20, w - 70, h - 55, xlim, (0, ymax), color, 1.8))
            png_lines.append((tt["time"].to_list(), list(y), xlim, (0, ymax), rgb(color)))
    save_svg_and_png(FIGURES / "dfba_3a_vs_3b_mechanism.svg", 1040, 365, "\n".join(body), png_lines)

    body = [fixed.svg_text(520, 30, "Population-Level Dynamic Complexity", 18, "bold")]
    png_lines = []
    summaries = [
        ("state transitions", dynamic_audit[dynamic_audit["experiment"] != "3A"]["n_state_transitions"]),
        ("unique beta fluxes", dynamic_audit[dynamic_audit["experiment"] != "3A"]["n_unique_beta_fluxes"]),
        ("constant beta-flux fraction", dynamic_audit[dynamic_audit["experiment"] != "3A"]["fraction_constant_beta_flux"]),
    ]
    for idx, (title, series) in enumerate(summaries):
        counts = series.value_counts().sort_index()
        x0, y0, w, h = 70 + idx * 310, 80, 250, 190
        body.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfb" stroke="#ccc"/>')
        body.append(fixed.svg_text(x0 + w / 2, y0 - 9, title, 11, "bold"))
        ymax = max(float(counts.max()), 1e-8)
        for j, (xv, val) in enumerate(counts.items()):
            bx = x0 + 25 + j * max(10, (w - 60) / max(len(counts), 1))
            bh = (h - 55) * float(val) / ymax
            body.append(f'<rect x="{bx:.1f}" y="{y0 + h - 30 - bh:.1f}" width="16" height="{bh:.1f}" fill="#6d8f5f"/>')
    y0 = 310
    active = constraint_audit.set_index("constraint")["fraction_time_intervals_active"]
    body.append(fixed.svg_text(520, y0 - 10, "Constraint active fractions", 11, "bold"))
    for i, (name, val) in enumerate(active.items()):
        x = 80 + i * 155
        body.append(f'<rect x="{x}" y="{y0}" width="{float(val) * 120:.1f}" height="18" fill="#8ca3b8"/>')
        body.append(fixed.svg_text(x, y0 + 36, name.replace("_", "\n"), 7, anchor="start"))
    save_svg_and_png(FIGURES / "dfba_dynamic_complexity_summary.svg", 1040, 440, "\n".join(body), png_lines)


def explanation_table(dynamic_audit: pd.DataFrame, constraint_audit: pd.DataFrame, state_special: pd.DataFrame) -> pd.DataFrame:
    dyn = dynamic_audit[dynamic_audit["experiment"] != "3A"]
    active = constraint_audit.set_index("constraint")
    recovering = float(state_special[state_special["quantity"] == "cultures_entering_recovering"]["value"].iloc[0])
    beta_active = float(active.loc["beta_carotene_pathway_capacity", "fraction_time_intervals_active"])
    atp_active = float(active.loc["ATP_maintenance_requirement", "fraction_time_intervals_active"])
    plateau_fraction = float(dyn["has_plateau_or_decline"].mean())
    median_max_beta_change = float(dyn["max_beta_flux_change"].median())
    rows = [
        ("only_one_or_two_state_transitions", f"median transitions={dyn['n_state_transitions'].median():.1f}; fraction >=2={(dyn['n_state_transitions'] >= 2).mean():.3f}", "partial"),
        ("burdens_do_not_cross_thresholds", f"transitions occur in {(dyn['n_state_transitions'] > 0).mean():.3f} of dynamic cultures", "not_supported"),
        ("state_multipliers_too_weak", f"median max beta-flux change={median_max_beta_change:.4f}", "not_supported" if median_max_beta_change >= 0.02 else "partial"),
        ("bounds_change_but_not_active", f"beta capacity active={beta_active:.3f}; ATP active={atp_active:.3f}", "not_supported" if beta_active >= 0.05 and atp_active >= 0.05 else "partial"),
        ("fast_surrogate_composed_of_simple_rules", f"backend={dyn['metabolic_backend'].mode().iloc[0]}", "supported"),
        ("piecewise_constant_flux_integrates_to_linear_segments", f"median constant beta-flux fraction={dyn['fraction_constant_beta_flux'].median():.3f}", "partial"),
        ("biomass_dominates_trajectory", f"median mean beta-flux change={dyn['mean_beta_flux_change'].median():.4f}", "supported"),
        ("product_degradation_too_weak_for_plateaus", f"plateau/decline fraction={plateau_fraction:.3f}", "not_supported" if plateau_fraction >= 0.05 else "supported"),
        ("recovery_transitions_never_occur", f"cultures entering recovering={recovering:.0f}", "supported" if recovering == 0 else "not_supported"),
        ("full_GEM_not_used", f"backend={dyn['metabolic_backend'].mode().iloc[0]}; actual FBA solves={dyn['n_actual_fba_solves'].max()}", "supported"),
    ]
    return pd.DataFrame(rows, columns=["explanation", "evidence", "support"])


def main() -> None:
    traj = ensure_backend(pd.read_csv(DATA / "dfba_trajectories.csv"))
    flux = ensure_backend(pd.read_csv(DATA / "dfba_fluxes.csv"))
    dynamic_audit = build_dynamic_audit(traj, flux)
    constraint_summary, constraint_detail = build_constraint_audit(flux)
    trans_dist, occupancy, matrix, state_special = build_state_audit(traj)
    criteria = acceptance_criteria(dynamic_audit, constraint_summary, traj)
    explanations = explanation_table(dynamic_audit, constraint_summary, state_special)

    dynamic_audit.to_csv(DATA / "dfba_dynamic_audit.csv", index=False)
    constraint_summary.to_csv(DATA / "dfba_active_constraint_audit.csv", index=False)
    constraint_detail.to_csv(DATA / "dfba_active_constraint_audit_long.csv", index=False)
    trans_dist.to_csv(DATA / "dfba_state_transition_distribution.csv", index=False)
    occupancy.to_csv(DATA / "dfba_state_occupancy.csv", index=False)
    matrix.to_csv(DATA / "dfba_state_transition_matrix.csv", index=False)
    state_special.to_csv(DATA / "dfba_state_special_cases.csv", index=False)
    criteria.to_csv(DATA / "dfba_generator_acceptance_criteria.csv", index=False)
    explanations.to_csv(DATA / "dfba_dynamic_explanation_audit.csv", index=False)
    make_figures(traj, flux, dynamic_audit, constraint_summary)

    print("dynamic audit complete")
    print(dynamic_audit[dynamic_audit["experiment"] != "3A"]["n_state_transitions"].value_counts().sort_index().to_string())
    print(criteria.to_string(index=False))


if __name__ == "__main__":
    main()
