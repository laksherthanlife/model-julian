#!/usr/bin/env python3
"""Select top-8 hybrid candidates per campaign after rerank completion."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HYBRID_BATCH = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpc-dir", default="final_validation_hpc")
    parser.add_argument("--rerank-results-dir", default="final_validation_hpc/rerank_results")
    parser.add_argument("--out", default="final_validation_hpc/hybrid_exact_task_manifest.csv")
    args = parser.parse_args()

    hpc = Path(args.hpc_dir)
    with (hpc / "campaign_table.csv").open(newline="", encoding="utf-8") as handle:
        campaigns = list(csv.DictReader(handle))
    task_rows = []
    missing = []
    for campaign in campaigns:
        with Path(campaign["shortlist_manifest"]).open(newline="", encoding="utf-8") as handle:
            short = list(csv.DictReader(handle))
        rows = []
        for cand in short:
            summary_path = Path(args.rerank_results_dir) / f"{cand['candidate_id']}_summary.csv"
            if not summary_path.exists():
                missing.append(str(summary_path))
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            row["candidate_json"] = cand["candidate_json"]
            rows.append(row)
        if len(rows) != len(short):
            continue
        ranked = sorted(rows, key=lambda r: float(r["final_product_pred_controls_via_real_gsm"]), reverse=True)
        for rank, row in enumerate(ranked[:HYBRID_BATCH], start=1):
            task_rows.append(
                {
                    "array_index": len(task_rows) + 1,
                    "experiment_group": row["experiment_group"],
                    "campaign_id": row["campaign_id"],
                    "world_id": row["world_id"],
                    "campaign_seed": int(float(row["campaign_seed"])),
                    "training_cultures_n": int(float(row["training_cultures_n"])),
                    "subsample_seed": int(float(row["subsample_seed"])),
                    "checkpoint": row["checkpoint"],
                    "normalization_stats": row["normalization_stats"],
                    "candidate_id": row["candidate_id"],
                    "candidate_hash": row["candidate_hash"],
                    "candidate_json": row["candidate_json"],
                    "surrogate_rank": int(float(row["surrogate_rank"])),
                    "rerank_rank": rank,
                    "rerank_score": float(row["final_product_pred_controls_via_real_gsm"]),
                }
            )
    if missing:
        raise SystemExit(f"missing {len(missing)} rerank summaries; first: {missing[:5]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)
    print(f"wrote {len(task_rows)} hidden exact hybrid tasks")


if __name__ == "__main__":
    main()
