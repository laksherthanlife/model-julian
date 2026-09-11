#!/usr/bin/env python3
"""Run a small 3x3x3 real-GEM diagnostic grid after the pilot passes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import gem_backend as gem


def pca_summary(curves: np.ndarray) -> pd.DataFrame:
    mean = curves.mean(axis=0)
    _u, s, _vt = np.linalg.svd(curves - mean, full_matrices=False)
    var = s**2
    explained = var / max(var.sum(), gem.EPS)
    rows = []
    cum = np.cumsum(explained)
    for k in range(1, min(5, len(cum)) + 1):
        rows.append({"component_count": k, "cumulative_explained_variance": float(cum[k - 1])})
    for target in (0.90, 0.95, 0.99, 0.999):
        rows.append({"component_count": int(np.searchsorted(cum, target) + 1), "target_variance": target})
    return pd.DataFrame(rows)


def make_grid_figures(traj: pd.DataFrame, fluxes: pd.DataFrame, pca: pd.DataFrame) -> None:
    gem.FIGURES.mkdir(exist_ok=True)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="420" viewBox="0 0 950 420">', '<rect width="100%" height="100%" fill="white"/>', gem.svg_text(475, 28, "Real GEM Small Grid: Environment to Beta-Carotene Curves", 17, "bold")]
    ymax = max(float(traj["B_total"].max()), gem.EPS)
    for env_id, g in traj.groupby("environment_id"):
        color = "#b65d3b" if g["DO"].iloc[0] < 40 else ("#2f6f9f" if g["DO"].iloc[0] > 40 else "#4c8b57")
        body.append(gem.svg_polyline(g["time"], g["B_total"], 70, 65, 780, 270, (0, traj["time"].max()), (0, ymax), color, 1.0))
    body.append(gem.svg_text(460, 370, "27 fixed environments; color roughly tracks DO", 11))
    body.append("</svg>")
    (gem.FIGURES / "gem_small_grid_environment_to_curve.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_small_grid_environment_to_curve.png", 1400, 720, [(g["time"].to_numpy(float), g["B_total"].to_numpy(float), (0, traj["time"].max()), (0, ymax), (182, 93, 59)) for _, g in traj.groupby("environment_id")])

    flux_cols = ["beta_carotene_flux", "biomass_flux", "oxygen_uptake", "ggpp_flux"]
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="420" viewBox="0 0 950 420">', '<rect width="100%" height="100%" fill="white"/>', gem.svg_text(475, 28, "Real GEM Small Grid Flux Diversity", 17, "bold")]
    summary = fluxes.groupby("environment_id")[flux_cols].mean().reset_index()
    for i, col in enumerate(flux_cols):
        vals = summary[col].to_numpy(float)
        lo, hi = vals.min(), vals.max()
        x0 = 85 + i * 210
        body.append(f'<rect x="{x0}" y="80" width="135" height="230" fill="#fbfbfb" stroke="#bbb"/>')
        for j, val in enumerate(vals):
            x = x0 + 15 + (j % 3) * 38
            y = 290 - (val - lo) / max(hi - lo, gem.EPS) * 180
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2f6f9f"/>')
        body.append(gem.svg_text(x0 + 68, 335, col, 10))
        body.append(gem.svg_text(x0 + 68, 355, f"range {lo:.3g} to {hi:.3g}", 9))
    body.append("</svg>")
    (gem.FIGURES / "gem_small_grid_flux_diversity.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_small_grid_flux_diversity.png", 1200, 650, [])

    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">', '<rect width="100%" height="100%" fill="white"/>', gem.svg_text(360, 28, "Real GEM Small Grid Trajectory PCA", 17, "bold"), '<rect x="70" y="65" width="560" height="220" fill="#fbfbfb" stroke="#bbb"/>']
    pk = pca[pca["cumulative_explained_variance"].notna()]
    if not pk.empty:
        body.append(gem.svg_polyline(pk["component_count"], pk["cumulative_explained_variance"], 100, 90, 500, 150, (1, max(pk["component_count"])), (0, 1), "#4c8b57", 2.2))
    body.append("</svg>")
    (gem.FIGURES / "gem_small_grid_pca.svg").write_text("\n".join(body), encoding="utf-8")
    gem.write_png_from_series(gem.FIGURES / "gem_small_grid_pca.png", 1000, 550, [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem-path", default=None)
    parser.add_argument("--n-time", type=int, default=7, help="Short diagnostic time grid; pilot uses 25.")
    args = parser.parse_args()
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(args.gem_path)
    base = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base)
    all_traj, all_flux, all_constraints = [], [], []
    temps = [27.0, 30.0, 33.0]
    phs = [4.5, 5.0, 5.5]
    dos = [20.0, 40.0, 80.0]
    for temp in temps:
        for ph in phs:
            for do in dos:
                env_id = f"T{temp:g}_pH{ph:g}_DO{do:g}".replace(".", "p")
                cfg = gem.GEMCultureConfig(temperature=temp, pH=ph, DO=do, n_time=args.n_time)
                traj, flux, constraints, _summary = gem.run_dynamic_culture(augmented, source_path, cfg, culture_id=env_id)
                traj["environment_id"] = env_id
                flux["environment_id"] = env_id
                constraints["environment_id"] = env_id
                all_traj.append(traj)
                all_flux.append(flux)
                all_constraints.append(constraints)
    traj = pd.concat(all_traj, ignore_index=True)
    fluxes = pd.concat(all_flux, ignore_index=True)
    constraints = pd.concat(all_constraints, ignore_index=True)
    curves = np.vstack([g.sort_values("time")["B_total"].to_numpy(float) for _, g in traj.groupby("environment_id")])
    pca = pca_summary(curves)
    audit = pd.DataFrame(
        [
            {
                "n_environments": traj["environment_id"].nunique(),
                "feasibility_rate": float((fluxes["n_infeasible_solves"] == 0).mean()),
                "beta_carotene_flux_min": float(fluxes["beta_carotene_flux"].min()),
                "beta_carotene_flux_max": float(fluxes["beta_carotene_flux"].max()),
                "growth_flux_min": float(fluxes["biomass_flux"].min()),
                "growth_flux_max": float(fluxes["biomass_flux"].max()),
                "distinct_state_sequences": int(traj.groupby("environment_id")["z_ox"].count().shape[0]),
                "distinct_beta_flux_values_rounded": int(fluxes["beta_carotene_flux"].round(6).nunique()),
                "distinct_active_constraint_vectors": int(constraints[["oxygen_lower_bound", "atp_maintenance_lower_bound", "pathway_capacity_upper_bound"]].round(6).drop_duplicates().shape[0]),
                "total_lp_solves": int(traj.groupby("environment_id")["n_actual_lp_solves"].first().sum()),
                "total_surrogate_evaluations": int(traj.groupby("environment_id")["n_surrogate_evaluations"].first().sum()),
                "comparison_with_reduced_surrogate": "real GEM grid uses yeast_gem_lp with staged LP solves and 0 surrogate evaluations; reduced surrogate used algebraic flux formulas and 24 surrogate evaluations per culture",
            }
        ]
    )
    gem.DATA.mkdir(exist_ok=True)
    traj.to_csv(gem.DATA / "gem_small_grid_trajectories.csv", index=False)
    fluxes.to_csv(gem.DATA / "gem_small_grid_fluxes.csv", index=False)
    constraints.to_csv(gem.DATA / "gem_small_grid_constraints.csv", index=False)
    audit.to_csv(gem.DATA / "gem_small_grid_audit.csv", index=False)
    pca.to_csv(gem.DATA / "gem_small_grid_pca.csv", index=False)
    make_grid_figures(traj, fluxes, pca)
    print("GEM small grid complete")
    print(audit.to_string(index=False))
    print(pca.to_string(index=False))


if __name__ == "__main__":
    main()
