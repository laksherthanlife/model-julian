#!/usr/bin/env python3
"""Step 5: small paired DBTL dry run -- conventional adaptive arm.

Runs a small (3-stage x 4-candidate = 12 exact culture) conventional
adaptive campaign in-process (sequential: each stage's proposals depend on
the previous stage's real exact observations), reusing
run_prospective_dbtl_benchmark's own propose_conventional_batch/observed_score
unmodified. The paired hybrid arm reuses the already-completed Step 3
virtual-ranking-validation exact results (top-12-by-virtual-score subset,
same 12-culture budget, same world/library/candidate distribution) --
no new hybrid-side exact compute is spent, since Step 3 already produced it.

This is explicitly a dry run for mechanics/compute-estimation, not a powered
result (n=1 seed, budget=12 per arm).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_deployment_verifier as verifier  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

DATA = ROOT / "data"
WORLD_ID = "baseline_world"
SEED = 9101
N_STAGES = 3
BATCH_SIZE = 4


def main() -> None:
    library = bench.load_library()
    rng = np.random.default_rng(SEED)
    factory = bench.CandidateFactory(library, "repaired_dry_run", "conventional", rng)

    observations: list[dict] = []
    seen_keys: set[str] = set()
    exact_rows = []
    tic = time.perf_counter()

    for stage in range(N_STAGES):
        selected, _dupes = bench.propose_conventional_batch(factory, observations, BATCH_SIZE, seen_keys, stage, library)
        for candidate in selected:
            traj, flux, constraints, summary = verifier.exact_sparse_dynamic_rollout(candidate, hidden_regime=WORLD_ID)
            row = summary.iloc[0].to_dict()
            row["stage"] = stage
            row["candidate_id"] = candidate["candidate_id"]
            row["candidate_hash"] = candidate["candidate_hash"]
            row["mechanism_classes"] = bench.mechanism_classes(candidate, library)
            exact_rows.append(row)
            observations.append({**row, "final_product": row.get("final_product", 0.0), "biologically_valid": bool(row.get("feasible", False))})
            print(f"stage={stage} candidate={candidate['candidate_id']} final_product={row.get('final_product'):.4f} feasible={row.get('feasible')}")

    elapsed = time.perf_counter() - tic
    df = pd.DataFrame(exact_rows)
    df.to_csv(DATA / "repaired_dry_run_conventional_exact_results.csv", index=False)

    best = df["final_product"].max()
    print(f"\nConventional dry run: {len(df)} exact cultures, best final_product={best:.4f}, wall_clock={elapsed:.1f}s")


if __name__ == "__main__":
    main()
