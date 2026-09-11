#!/usr/bin/env python3
"""Run one hidden exact simulator verification task for final validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import design_benchmark_exact as exact  # noqa: E402
import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import run_deployment_verifier as verifier  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

BENCHMARK_ID = "final_validation_boundary_conditions_v1"
HYBRID_METHOD_ID = "pretrained_hybrid_digital_twin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-row", required=True)
    parser.add_argument("--out-dir", default="final_validation_hpc/hybrid_exact_results")
    args = parser.parse_args()

    with Path(args.task_row).open(newline="", encoding="utf-8") as handle:
        task = next(csv.DictReader(handle))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{int(task['array_index']):05d}_{task['candidate_id']}"
    summary_path = out_dir / f"{prefix}_summary.csv"
    if summary_path.exists():
        print(f"already done: {summary_path}")
        return

    candidate = json.loads(Path(str(task["candidate_json"])).read_text(encoding="utf-8"))
    library = v2.build_common_intervention_library()
    traj, flux, constraints, hidden_summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=str(task["world_id"]))
    result = bench.summarize_rollout(traj, flux, hidden_summary, candidate, library)
    result.update(
        {
            **task,
            "benchmark_id": BENCHMARK_ID,
            "method_id": HYBRID_METHOD_ID,
            "physical_culture_charge": 1,
            "mock_mode": False,
            "hidden_exact_simulator": "exact_sparse_dynamic_rollout",
        }
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    traj.to_csv(out_dir / f"{prefix}_trajectory.csv", index=False)
    flux.to_csv(out_dir / f"{prefix}_flux.csv", index=False)
    constraints.to_csv(out_dir / f"{prefix}_constraints.csv", index=False)
    exact.clear_solver_caches()
    print(f"done: {prefix}")


if __name__ == "__main__":
    main()
