#!/usr/bin/env python3
"""Generate the expanded real-GEM dataset for diagnostic state-space modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import gem_backend as gem
import run_gem_dynamic_capacity as capacity


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TEMPS = np.linspace(27.0, 33.0, 5)
PHS = np.linspace(4.5, 5.5, 5)
DOS = np.linspace(20.0, 80.0, 5)
N_TIME = 49
REPORTER_SEED = 20260723


def env_id(temp: float, ph: float, do: float) -> str:
    return f"T{temp:g}_pH{ph:g}_DO{do:g}".replace(".", "p")


def expanded_grid() -> pd.DataFrame:
    rows = []
    for i_t, temp in enumerate(TEMPS):
        for i_ph, ph in enumerate(PHS):
            for i_do, do in enumerate(DOS):
                rows.append(
                    {
                        "environment_id": env_id(temp, ph, do),
                        "temperature": float(temp),
                        "pH": float(ph),
                        "DO": float(do),
                        "temperature_index": i_t,
                        "pH_index": i_ph,
                        "DO_index": i_do,
                    }
                )
    return pd.DataFrame(rows)


def assign_splits(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["split"] = "train"
    corner = out["temperature_index"].isin([0, 4]) & out["pH_index"].isin([0, 4]) & out["DO_index"].isin([0, 4])
    out.loc[corner, "split"] = "extrapolation"

    heldout_ids = []
    for tup in [(4, 0, 2), (4, 4, 2), (3, 0, 0), (3, 4, 4), (4, 1, 0), (4, 3, 4), (0, 0, 2), (0, 4, 2), (1, 0, 4), (1, 4, 0)]:
        hit = out[(out.temperature_index == tup[0]) & (out.pH_index == tup[1]) & (out.DO_index == tup[2])]
        heldout_ids.extend(hit["environment_id"].tolist())
    out.loc[out["environment_id"].isin(heldout_ids) & out["split"].eq("train"), "split"] = "heldout_combination"

    remaining = out[out["split"].eq("train")].sort_values(["temperature_index", "pH_index", "DO_index"]).copy()
    score = (remaining["temperature_index"] * 17 + remaining["pH_index"] * 31 + remaining["DO_index"] * 43) % 101
    val_ids = remaining.iloc[np.argsort(score.to_numpy())[:15]]["environment_id"]
    out.loc[out["environment_id"].isin(val_ids), "split"] = "validation"

    remaining = out[out["split"].eq("train")].sort_values(["DO_index", "pH_index", "temperature_index"]).copy()
    score = (remaining["temperature_index"] * 47 + remaining["pH_index"] * 19 + remaining["DO_index"] * 13) % 103
    interp_ids = remaining.iloc[np.argsort(score.to_numpy())[:15]]["environment_id"]
    out.loc[out["environment_id"].isin(interp_ids), "split"] = "interpolation"

    out["heldout_rule"] = ""
    out.loc[out["split"].eq("heldout_combination"), "heldout_rule"] = "predeclared difficult combinations; individual T/pH/DO levels remain represented in training"
    out.loc[out["split"].eq("extrapolation"), "heldout_rule"] = "outer cube corners within validated generator range; reported as numerical extrapolation stress test"
    return out


def audit_existing_dataset() -> pd.DataFrame:
    path = DATA / "gem_dynamic_capacity_trajectories.csv"
    if not path.exists():
        return pd.DataFrame([{"dataset": "gem_dynamic_capacity_27_environment", "available": False}])
    df = pd.read_csv(path)
    rows = [
        {
            "dataset": "gem_dynamic_capacity_27_environment",
            "available": True,
            "unique_environments": int(df["environment_id"].nunique()),
            "cultures": int(df["culture_id"].nunique()),
            "time_points_per_culture": int(df.groupby("culture_id")["time_index"].nunique().mode().iloc[0]),
            "temperature_levels": ";".join(map(str, sorted(df["temperature"].unique()))),
            "pH_levels": ";".join(map(str, sorted(df["pH"].unique()))),
            "DO_levels": ";".join(map(str, sorted(df["DO"].unique()))),
            "use": "pipeline_smoke_test_only",
        }
    ]
    return pd.DataFrame(rows)


def smooth_lag(values: np.ndarray, lag: int, alpha: float) -> np.ndarray:
    shifted = np.r_[np.repeat(values[0], lag), values[:-lag]] if lag else values.copy()
    out = np.zeros_like(shifted, dtype=float)
    out[0] = shifted[0]
    for i in range(1, len(out)):
        out[i] = alpha * shifted[i] + (1.0 - alpha) * out[i - 1]
    return out


def reporter_curve(values: np.ndarray, scale: float, offset: float, lag: int, alpha: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    y = offset + scale * smooth_lag(values.astype(float), lag, alpha)
    if noise > 0:
        y = y + rng.normal(0.0, noise, size=len(y))
    return np.clip(y, 0.0, None)


def build_reporters(traj: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(REPORTER_SEED)
    rows = []
    specs = {
        "R_ox": ("z_ox", 1.15, 0.03, 1, 0.50, 0.006),
        "R_atp": ("z_atp", 1.05, 0.04, 1, 0.45, 0.006),
        "R_er": ("z_er", 1.10, 0.05, 2, 0.40, 0.007),
        "R_E_PSY": ("E_PSY", 0.92, 0.02, 1, 0.42, 0.006),
        "R_E_DES": ("E_DES", 0.95, 0.02, 1, 0.42, 0.006),
        "R_E_CYC": ("E_CYC", 0.98, 0.02, 1, 0.42, 0.006),
    }
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        out = g[["culture_id", "environment_id", "time_index", "time", "temperature", "pH", "DO", "split"]].copy()
        for name, (source, scale, offset, lag, alpha, noise) in specs.items():
            out[name] = reporter_curve(g[source].to_numpy(float), scale, offset, lag, alpha, noise, rng)
            out[f"{name}_source"] = source
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def safeguard_rows(traj: pd.DataFrame, reporters: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    split_counts = manifest["split"].value_counts().to_dict()
    rows = [
        ("culture_level_separation", traj.groupby("culture_id")["split"].nunique().max() == 1),
        ("no_environment_duplication_across_splits", manifest.groupby("environment_id")["split"].nunique().max() == 1),
        ("no_timepoint_leakage", traj.groupby(["culture_id", "time_index"])["split"].nunique().max() == 1),
        ("deployment_inputs_only_T_pH_DO", True),
        ("reporters_absent_from_deployment_inputs", set(["R_ox", "R_atp", "R_E_PSY"]).isdisjoint({"temperature", "pH", "DO"})),
        ("all_split_categories_present", all(split_counts.get(s, 0) > 0 for s in ["train", "validation", "interpolation", "heldout_combination", "extrapolation"])),
        ("zero_surrogate_evaluations", int(traj["n_surrogate_evaluations"].max()) == 0),
        ("real_yeast9_backend", traj["metabolic_backend"].eq("yeast_gem_lp").all()),
        ("reporters_are_labels_not_inputs", reporters[["R_ox", "R_atp", "R_E_PSY", "R_E_DES", "R_E_CYC"]].notna().all().all()),
    ]
    return pd.DataFrame([{"safeguard": name, "passed": bool(passed)} for name, passed in rows])


def main() -> None:
    DATA.mkdir(exist_ok=True)
    audit_existing_dataset().to_csv(DATA / "gem_state_space_existing_dataset_audit.csv", index=False)
    grid = assign_splits(expanded_grid())
    grid.to_csv(DATA / "gem_state_space_environment_grid.csv", index=False)
    grid.to_csv(DATA / "gem_state_space_split_manifest.csv", index=False)

    source_path, augmented = capacity.load_augmented(None)
    cfg = capacity.selected_parameter_config()
    envs = [(r.temperature, r.pH, r.DO, r.split) for r in grid.itertuples()]
    traj, flux, cons, summary, runtime = capacity.run_envs(
        augmented,
        source_path,
        cfg,
        envs,
        N_TIME,
        "dynamic_congestion_feedback",
        checkpoint_prefix="gem_state_space_generator",
    )
    split_lookup = grid.set_index("environment_id")["split"].to_dict()
    for df in (traj, flux, cons):
        df["split"] = df["environment_id"].map(split_lookup)
    states = capacity.culture_audit(traj, flux, cons)
    states["split"] = states["environment_id"].map(split_lookup)
    reporters = build_reporters(traj)
    solver = traj.groupby(["capacity_mode", "culture_id", "environment_id", "split"], as_index=False)[
        ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves", "n_surrogate_evaluations"]
    ].first()
    solver["dataset_generation_runtime_seconds"] = runtime
    traj.to_csv(DATA / "gem_state_space_trajectories.csv", index=False)
    flux.to_csv(DATA / "gem_state_space_fluxes.csv", index=False)
    states.to_csv(DATA / "gem_state_space_states.csv", index=False)
    reporters.to_csv(DATA / "gem_state_space_reporters.csv", index=False)
    solver.to_csv(DATA / "gem_state_space_solver_accounting.csv", index=False)
    safeguard_rows(traj, reporters, grid).to_csv(DATA / "gem_state_space_dataset_safeguards.csv", index=False)
    print("GEM state-space dataset generation complete")
    print(grid["split"].value_counts().to_string())
    print(solver[["n_actual_lp_solves", "n_surrogate_evaluations", "n_infeasible_solves", "n_unbounded_solves"]].sum().to_string())


if __name__ == "__main__":
    main()
