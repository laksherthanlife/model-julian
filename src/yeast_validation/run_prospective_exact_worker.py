#!/usr/bin/env python3
"""Run one prospective exact Yeast9 rollout and exit to release solver memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import design_benchmark_exact as exact  # noqa: E402
import run_deployment_verifier as verifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--flux", required=True)
    parser.add_argument("--constraints", required=True)
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    traj, flux, constraints, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=args.world_id)
    for path in [args.summary, args.trajectory, args.flux, args.constraints]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    traj.to_csv(args.trajectory, index=False)
    flux.to_csv(args.flux, index=False)
    constraints.to_csv(args.constraints, index=False)
    exact.clear_solver_caches()


if __name__ == "__main__":
    main()
