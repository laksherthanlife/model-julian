#!/usr/bin/env python3
"""Run one conventional DBTL campaign for final-validation robustness worlds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

BENCHMARK_ID = "final_validation_boundary_conditions_v1"
CONVENTIONAL_METHOD_ID = "conventional_dbtl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", default="results/final_validation_conventional_campaigns")
    args = parser.parse_args()

    campaign_id = f"prospective_{args.world_id}_seed_{args.seed}"
    campaign_dir = Path(args.out_root) / campaign_id
    done = campaign_dir / "_DONE"
    if done.exists():
        print(f"already done: {campaign_id}")
        return
    library = v2.build_common_intervention_library()
    teacher_lookup = bench.load_teacher_final_lookup()
    bench.BENCHMARK_ID = BENCHMARK_ID
    bench.PRIMARY_METHODS = [CONVENTIONAL_METHOD_ID]
    bench.DATA = campaign_dir / "data"
    bench.RESULTS = campaign_dir / "results"
    bench.ROLLOUTS = bench.RESULTS / "on_demand_exact_rollouts"
    bench.DATA.mkdir(parents=True, exist_ok=True)
    bench.ROLLOUTS.mkdir(parents=True, exist_ok=True)
    bench.run_campaign(
        world_id=args.world_id,
        seed=args.seed,
        library=library,
        teacher_lookup=teacher_lookup,
        conventional_batches=[4, 4, 4, 4],
        hybrid_batch=0,
        virtual_evaluations=0,
        mock=False,
        fresh_exact=False,
    )
    done.write_text("done\n", encoding="utf-8")
    print(f"done: {campaign_id}")


if __name__ == "__main__":
    main()
