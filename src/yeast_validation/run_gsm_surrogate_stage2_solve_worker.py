#!/usr/bin/env python3
"""Stage 2 worker: for one shard of (env, control-vector) points, run the
exact Yeast9 staged solve (growth LP + product LP + pFBA -- same 3-solve
pattern as gem_backend.solve_staged, same apply_predicted_interface_controls
bound mapping already used for Path B / production) and record the full
flux-output vector. This is the purpose-built exact-LP training data for the
replacement surrogate (Stage 3) -- unlike the original surrogate's training
data, these points explicitly cover edited/edge-case/extrapolative control
combinations, not just naturally-visited reference-strain trajectories.

Fresh model.copy() per point (matching the validated Path B / production
pattern in run_repaired_controls_real_gsm_replay_worker.py) so bounds from
one solve never leak into the next point's solve.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    points = pd.read_csv(args.points_csv)

    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)

    rows = []
    for _, p in points.iterrows():
        cfg = gem.GEMCultureConfig(temperature=float(p["temperature"]), pH=float(p["pH"]), DO=float(p["DO"]))
        controls = {col: float(p[col]) for col in sur.CONTROL_COLUMNS}
        interval_model = augmented.copy()
        prod.apply_predicted_interface_controls(interval_model, cfg, controls)
        gamma = float(np.clip(controls["gamma_growth_fraction"], 0.05, 0.95))

        tic = time.perf_counter()
        try:
            flux = gem.solve_staged(interval_model, cfg, gamma=gamma, state="balanced")
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            flux = {col: float("nan") for col in sur.SURROGATE_OUTPUT_COLUMNS}
            status = f"error:{type(exc).__name__}"
        runtime = time.perf_counter() - tic

        row = {
            "point_id": p["point_id"],
            "source": p["source"],
            "candidate_id": p.get("candidate_id", ""),
            "t": p.get("t", -1),
            "temperature": p["temperature"],
            "pH": p["pH"],
            "DO": p["DO"],
            "status": status,
            "solve_runtime_seconds": runtime,
        }
        for col in sur.CONTROL_COLUMNS:
            row[col] = p[col]
        for col in sur.SURROGATE_OUTPUT_COLUMNS:
            row[col] = flux.get(col, float("nan"))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
