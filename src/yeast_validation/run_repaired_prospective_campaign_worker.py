#!/usr/bin/env python3
"""Run ONE paired campaign (both conventional_dbtl and pretrained_hybrid_digital_twin
methods, one world, one seed) of the repaired-hybrid prospective DBTL rerun.

Reuses run_prospective_dbtl_benchmark.run_campaign unmodified (via
gem_repaired_prospective_common.patch_benchmark_module, which monkey-patches
the scorer and isolates all I/O to a per-campaign directory -- see that
module's docstring). Designed to be one PBS array task; campaigns are
independent of each other and safe to run fully in parallel.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_repaired_prospective_common as common  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conventional-batches", type=int, nargs="+", default=common.CONVENTIONAL_BATCHES)
    parser.add_argument("--hybrid-batch", type=int, default=common.HYBRID_BATCH)
    parser.add_argument("--virtual-evaluations", type=int, default=common.VIRTUAL_EVALUATIONS)
    parser.add_argument("--out-root", type=str, default=str(ROOT / "results" / "repaired_prospective_campaigns"))
    args = parser.parse_args()

    campaign_id = common.campaign_id_for(args.world_id, args.seed)
    campaign_dir = Path(args.out_root) / campaign_id
    done_marker = campaign_dir / "_DONE"
    if done_marker.exists():
        print(f"already done: {campaign_id}")
        return

    # Load the library and fit the scorer BEFORE patching bench.DATA to the
    # isolated campaign directory -- load_library()/the dataset loaders read
    # from bench.DATA / the real data/ directory respectively.
    library = common.build_common_intervention_library()
    scorer = common.load_scorer()
    common.patch_benchmark_module(campaign_dir)

    tic = time.perf_counter()
    bench.run_campaign(
        world_id=args.world_id,
        seed=args.seed,
        library=library,
        teacher_lookup=scorer,  # opaque pass-through; consumed only by our patched hybrid_virtual_score
        conventional_batches=args.conventional_batches,
        hybrid_batch=args.hybrid_batch,
        virtual_evaluations=args.virtual_evaluations,
        mock=False,
        fresh_exact=False,  # allow resume: reuse this campaign's own already-solved cultures (isolated cache), never a different config's cache
    )
    elapsed = time.perf_counter() - tic
    done_marker.write_text(f"elapsed_seconds={elapsed:.1f}\n", encoding="utf-8")
    print(f"campaign {campaign_id} done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
