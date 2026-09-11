#!/usr/bin/env python3
"""Stage 2: sample the actual deployment joint space (environment x predicted
controls x edit variables) for a purpose-built Yeast9 surrogate training
dataset -- not just naturally-visited reference-strain trajectories (which is
all the current surrogate was trained on; see gem_gsm_surrogate.py docstring).

Two point sources, mixed into one point list, all going to exact Yeast9 LP
solves on Vanda:

1. "traj" points: the physiology model's predicted, edit-adjusted control
   trajectory c(t) (same computation as Stage 1 / the production scorer),
   for a large diverse candidate pool, evaluated at several stratified
   timesteps across the rollout (captures the real growing-drift +
   edit-driven support extension found in Stage 1 -- unclipped, so the new
   surrogate can learn this region instead of needing a safety clip).

2. "edge" points: deliberate, predeclared decorrelated/correlated corner
   coverage -- independent per-control-dimension sampling out to +/-8
   training-std (roughly double the old CONTROL_CLIP_STDS=4 safety clip),
   plus explicit combinatorial extreme corners (e.g. low-O2/high-ATP,
   high-O2/low-ATP, capacity-crash, capacity-boost), each replicated across
   several environments. Every value is clipped to the physical sign
   constraint of its control (oxygen_lower_bound negative, ATP/PSY/DES/CYC
   non-negative, gamma in [0.05, 0.95]) -- not to astronomical values.

No LP solves happen in this script (it is cheap/local). It writes a single
point-list CSV; a companion Vanda worker does the exact solves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate as sur  # noqa: E402
import gem_repaired_prospective_common as common  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_CSV = DATA / "gsm_surrogate_stage2_points.csv"

N_TRAJ_CANDIDATES = 300
TRAJ_SEED = 51002
STRATIFIED_T = [0, 6, 13, 20, 27, 34, 41, 47]  # 8 timesteps spanning the rollout

N_EDGE_RANDOM = 6800
N_EDGE_CORNERS_PER_ENV = 16  # 2^4 sign corners on (O2, ATP, capacity-group, gamma) x envs
EDGE_SEED = 51003
EDGE_ENVS = 40
EDGE_Z_RANGE = 8.0  # sample within +/- this many training-std of each control dim

PHYSICAL_BOUNDS = {
    # (floor, ceiling) applied AFTER sampling, in raw (unstandardized) units.
    "oxygen_lower_bound": (-30.0, -0.05),
    "atp_maintenance_lower_bound": (0.0, None),
    "gamma_growth_fraction": (0.05, 0.95),
    "PSY_effective_upper_bound": (0.0, None),
    "DES_effective_upper_bound": (0.0, None),
    "CYC_effective_upper_bound": (0.0, None),
}


def clip_physical(col: str, values: np.ndarray) -> np.ndarray:
    lo, hi = PHYSICAL_BOUNDS[col]
    if lo is not None:
        values = np.maximum(values, lo)
    if hi is not None:
        values = np.minimum(values, hi)
    return values


def generate_traj_points(scorer, n_candidates: int, seed: int) -> list[dict]:
    library = common.build_common_intervention_library()
    rng = np.random.default_rng(seed)
    factory = bench.CandidateFactory(library, "stage2_points", "repaired_hybrid", rng)
    rows = []
    for _ in range(n_candidates):
        edit_count = int(rng.choice(np.arange(0, 4), p=[0.15, 0.35, 0.30, 0.20]))
        env = factory.random_env()
        if edit_count == 0:
            cand = factory.make([], env, "stage2 traj point: no-edit")
        else:
            idx = rng.choice(len(library), size=min(edit_count, len(library)), replace=False)
            rows_lib = [pd.Series(library.iloc[int(i)]) for i in idx]
            cand = factory.make(rows_lib, env, "stage2 traj point: edited")

        env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
        env_std_in = (env_raw - scorer.env_mean) / scorer.env_std
        cache = tss.forward(scorer.params, env_std_in, scorer_mod.T_LEN)
        c_pre = cache.z[:-1, 0, :] @ scorer.params.Wc + scorer.params.bc
        c_baseline = c_pre * scorer.surrogate.control_std + scorer.surrogate.control_mean
        c_edited, _ = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, cand.get("edits", []))

        for t in STRATIFIED_T:
            row = {
                "source": "traj",
                "candidate_id": cand["candidate_id"],
                "t": t,
                "n_edits": len(cand.get("edits", [])),
                "temperature": env["temperature"],
                "pH": env["pH"],
                "DO": env["DO"],
            }
            for i, col in enumerate(sur.CONTROL_COLUMNS):
                row[col] = float(clip_physical(col, np.array([c_edited[t, i]]))[0])
            rows.append(row)
    return rows


def generate_edge_random_points(surrogate, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        env = {
            "temperature": float(rng.uniform(*bench.ENV_DOMAIN["temperature"])),
            "pH": float(rng.uniform(*bench.ENV_DOMAIN["pH"])),
            "DO": float(rng.uniform(*bench.ENV_DOMAIN["DO"])),
        }
        row = {"source": "edge_random", "candidate_id": "", "t": -1, "n_edits": -1, **env}
        for i, col in enumerate(sur.CONTROL_COLUMNS):
            z = rng.uniform(-EDGE_Z_RANGE, EDGE_Z_RANGE)
            raw = surrogate.control_mean[i] + z * surrogate.control_std[i]
            row[col] = float(clip_physical(col, np.array([raw]))[0])
        rows.append(row)
    return rows


def generate_edge_corner_points(surrogate, n_envs: int, seed: int) -> list[dict]:
    """Deliberate combinatorial corners: sign combos of (oxygen, ATP,
    capacity-group [PSY/DES/CYC together], gamma) at +/-6std, held at the
    surrogate's training mean for whichever dims aren't part of the corner
    axis being tested, replicated across several environments."""
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(sur.CONTROL_COLUMNS)}
    rows = []
    corner_z = 6.0
    signs = [-1, 1]
    for _ in range(n_envs):
        env = {
            "temperature": float(rng.uniform(*bench.ENV_DOMAIN["temperature"])),
            "pH": float(rng.uniform(*bench.ENV_DOMAIN["pH"])),
            "DO": float(rng.uniform(*bench.ENV_DOMAIN["DO"])),
        }
        for s_ox in signs:
            for s_atp in signs:
                for s_cap in signs:
                    for s_gamma in signs:
                        row = {"source": "edge_corner", "candidate_id": "", "t": -1, "n_edits": -1, **env}
                        base = dict(zip(sur.CONTROL_COLUMNS, surrogate.control_mean))
                        base["oxygen_lower_bound"] += s_ox * corner_z * surrogate.control_std[idx["oxygen_lower_bound"]]
                        base["atp_maintenance_lower_bound"] += s_atp * corner_z * surrogate.control_std[idx["atp_maintenance_lower_bound"]]
                        for cap_col in ("PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"):
                            base[cap_col] += s_cap * corner_z * surrogate.control_std[idx[cap_col]]
                        base["gamma_growth_fraction"] += s_gamma * corner_z * surrogate.control_std[idx["gamma_growth_fraction"]]
                        for col in sur.CONTROL_COLUMNS:
                            row[col] = float(clip_physical(col, np.array([base[col]]))[0])
                        rows.append(row)
    return rows


def main() -> None:
    scorer = common.load_scorer()
    surrogate = scorer.surrogate

    traj_rows = generate_traj_points(scorer, N_TRAJ_CANDIDATES, TRAJ_SEED)
    edge_random_rows = generate_edge_random_points(surrogate, N_EDGE_RANDOM, EDGE_SEED)
    edge_corner_rows = generate_edge_corner_points(surrogate, EDGE_ENVS, EDGE_SEED + 1)

    all_rows = traj_rows + edge_random_rows + edge_corner_rows
    df = pd.DataFrame(all_rows)
    df.insert(0, "point_id", [f"pt_{i:06d}" for i in range(len(df))])
    df.to_csv(OUT_CSV, index=False)

    print(f"traj points: {len(traj_rows)}")
    print(f"edge_random points: {len(edge_random_rows)}")
    print(f"edge_corner points: {len(edge_corner_rows)}")
    print(f"total: {len(df)}")
    print(df["source"].value_counts())
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
