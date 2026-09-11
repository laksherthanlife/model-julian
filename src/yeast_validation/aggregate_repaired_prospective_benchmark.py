#!/usr/bin/env python3
"""Step 5: merge per-campaign isolated ledgers from the repaired-hybrid
prospective DBTL rerun into one aggregate dataset, then reuse
run_prospective_dbtl_benchmark.write_analysis() UNMODIFIED (monkey-patching
only bench.DATA to the aggregate directory) -- this gives the exact same
paired bootstrap statistics, leakage audit, conventional-adaptivity audit,
search-performance/AUC tables, and report format as the original benchmark,
so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

LEDGER_FILES = [
    "prospective_campaign_manifest.csv",
    "prospective_exact_simulator_call_ledger.csv",
    "prospective_biological_stage_ledger.csv",
    "prospective_virtual_evaluation_ledger.csv",
    "prospective_proposal_dependency_audit.csv",
    "prospective_candidate_designs.csv",
    "prospective_transfer_novelty_audit.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns-root", type=str, default=str(ROOT / "results" / "repaired_prospective_campaigns"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "repaired_prospective_benchmark_aggregate"))
    args = parser.parse_args()

    campaigns_root = Path(args.campaigns_root)
    out_data = Path(args.out_dir) / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    campaign_dirs = sorted(d for d in campaigns_root.iterdir() if d.is_dir())
    done = [d for d in campaign_dirs if (d / "_DONE").exists()]
    incomplete = [d.name for d in campaign_dirs if not (d / "_DONE").exists()]
    print(f"campaign dirs: {len(campaign_dirs)}, done: {len(done)}, incomplete: {incomplete}")

    for fname in LEDGER_FILES:
        frames = []
        for d in done:
            p = d / "data" / fname
            if p.exists() and p.stat().st_size > 0:
                frames.append(pd.read_csv(p))
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged.to_csv(out_data / fname, index=False)
            print(f"{fname}: {len(merged)} rows from {len(frames)} campaigns")
        else:
            print(f"{fname}: no data found")

    bench.DATA = out_data
    bench.RESULTS = Path(args.out_dir)
    bench.FIGURES = Path(args.out_dir) / "figures"
    bench.FIGURES.mkdir(parents=True, exist_ok=True)
    bench.write_analysis()
    print(f"Wrote analysis outputs to {out_data}")


if __name__ == "__main__":
    main()
