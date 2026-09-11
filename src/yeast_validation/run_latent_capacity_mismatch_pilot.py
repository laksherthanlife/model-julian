#!/usr/bin/env python3
"""Stage D3 pilot driver: generate a small environment-grid dataset for a
given D_true controller, retrain the clean D_model=16 teacher on it, and
report predictive metrics + wall-clock/LP-solve accounting (for the Stage F
powered-run compute estimate).

This is explicitly a *mechanics and power-estimation* pilot, not the powered
benchmark: the environment grid and time resolution are both far smaller than
the eventual full run (see --grid-size / --n-time), by design (Section 14,
Stage D of the handover spec).

Usage:
    python run_latent_capacity_mismatch_pilot.py --d-true 16 --grid-size 2 --n-time 13
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_backend as gem  # noqa: E402
import gem_controller_family as gcf  # noqa: E402
import run_clean_teacher_retrain as retrain  # noqa: E402
import run_gem_dynamic_capacity as dynamic_capacity  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PILOT_DIR = DATA / "latent_capacity_mismatch_pilot"


def _env_id(temp: float, ph: float, do: float) -> str:
    return f"T{temp:g}_pH{ph:g}_DO{do:g}".replace(".", "p")


def small_grid(grid_size: int, seed: int = 20260818) -> pd.DataFrame:
    """A grid_size^3 environment grid (default reduced from the 5x5x5=125
    production grid), with a val/train split assignment reused from
    generate_gem_state_space_dataset's corner/held-out logic simplified to a
    fixed fraction."""
    temps = np.linspace(27.0, 33.0, grid_size)
    phs = np.linspace(4.5, 5.5, grid_size)
    dos = np.linspace(20.0, 80.0, grid_size)
    rng = np.random.default_rng(seed)
    rows = []
    for temp in temps:
        for ph in phs:
            for do in dos:
                rows.append({"environment_id": _env_id(temp, ph, do), "temperature": float(temp), "pH": float(ph), "DO": float(do)})
    grid = pd.DataFrame(rows)
    n_val = max(1, int(round(0.2 * len(grid))))
    val_idx = rng.choice(len(grid), size=n_val, replace=False)
    grid["split"] = "train"
    grid.loc[val_idx, "split"] = "validation"
    return grid


def generate_pilot_dataset(d_true: int, grid_size: int, n_time: int, seed: int = 20260818) -> dict[str, object]:
    extra_states = gcf.build_controller(d_true)
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    cfg_base = dynamic_capacity.selected_parameter_config()
    cfg_base = replace(cfg_base, n_time=n_time, capacity_mode="dynamic_congestion_feedback")

    grid = small_grid(grid_size, seed=seed)
    traj_frames, reporter_frames = [], []
    totals = {"n_actual_lp_solves": 0, "n_infeasible_solves": 0, "n_unbounded_solves": 0}
    tic = time.perf_counter()
    for row in grid.itertuples():
        cfg = replace(cfg_base, temperature=row.temperature, pH=row.pH, DO=row.DO)
        culture_id = row.environment_id
        traj, _flux, _cons, summary = gcf.run_dynamic_culture_generic(augmented.copy(), source_path, cfg, extra_states=extra_states, culture_id=culture_id)
        traj["environment_id"] = row.environment_id
        traj["split"] = row.split
        for key in totals:
            totals[key] += int(summary[key].iloc[0])
        reporters = gcf.build_reporters_generic(traj, extra_states, seed=seed)
        traj_frames.append(traj)
        reporter_frames.append(reporters)
    wall_clock = time.perf_counter() - tic

    traj_all = pd.concat(traj_frames, ignore_index=True)
    reporters_all = pd.concat(reporter_frames, ignore_index=True)
    return {
        "traj": traj_all,
        "reporters": reporters_all,
        "grid": grid,
        "wall_clock_seconds": wall_clock,
        "n_environments": len(grid),
        "n_time": n_time,
        **totals,
    }


def run_pilot_condition(d_true: int, grid_size: int, n_time: int, seed: int = 20260818) -> dict[str, object]:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    gen = generate_pilot_dataset(d_true, grid_size, n_time, seed=seed)
    traj_path = PILOT_DIR / f"traj_d{d_true}.csv"
    reporters_path = PILOT_DIR / f"reporters_d{d_true}.csv"
    split_path = PILOT_DIR / f"splits_d{d_true}.csv"
    gen["traj"].to_csv(traj_path, index=False)
    gen["reporters"].to_csv(reporters_path, index=False)
    gen["grid"].to_csv(split_path, index=False)

    extra_states = gcf.build_controller(d_true)
    tic = time.perf_counter()
    out = retrain.run_clean_retrain_for_condition(d_true, traj_path, reporters_path, split_path, out_prefix="pilot_clean_teacher")
    train_wall_clock = time.perf_counter() - tic
    acceptance = retrain.acceptance_check(out["metrics"], d_true)

    summary_row = {
        "d_true": d_true,
        "grid_size": grid_size,
        "n_environments": gen["n_environments"],
        "n_time": n_time,
        "n_actual_lp_solves": gen["n_actual_lp_solves"],
        "n_infeasible_solves": gen["n_infeasible_solves"],
        "n_unbounded_solves": gen["n_unbounded_solves"],
        "generation_wall_clock_seconds": round(gen["wall_clock_seconds"], 2),
        "teacher_retrain_wall_clock_seconds": round(train_wall_clock, 2),
        "n_extra_states": len(extra_states),
        "acceptance_all_pass": bool(acceptance[~acceptance["gate"].str.startswith("INFO_")]["passed"].all()),
    }
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    acceptance.to_csv(PILOT_DIR / f"acceptance_d{d_true}.csv", index=False)
    pd.DataFrame([summary_row]).to_csv(PILOT_DIR / f"summary_d{d_true}.csv", index=False)
    return {"summary": summary_row, "acceptance": acceptance, "metrics": out["metrics"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-true", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=2, help="grid_size^3 environments (default 2 -> 8 envs, a mechanics-check pilot; production grid is 5 -> 125 envs)")
    parser.add_argument("--n-time", type=int, default=13, help="time points per culture (default 13 -> 12 intervals; production is 49 -> 48 intervals)")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args(argv)

    result = run_pilot_condition(args.d_true, args.grid_size, args.n_time, seed=args.seed)
    print(f"D_true={args.d_true} pilot summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    print(result["acceptance"].to_string(index=False))


if __name__ == "__main__":
    main()
