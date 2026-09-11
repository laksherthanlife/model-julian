#!/usr/bin/env python3
"""One-time repair for a schema-drift bug in
prospective_candidate_designs.csv across all v2 campaigns.

Root cause: run_repaired_prospective_campaign_worker_v2.py's reranking
closure attached extra internal-only keys (_surrogate_score, _rerank_score,
_rerank_rank) directly onto candidate dicts for bookkeeping. Those dicts are
later spread wholesale into prospective_candidate_designs.csv's row
(**cand, via csv_safe_row in run_prospective_dbtl_benchmark.py, unmodified)
for the 8 selected hybrid candidates per campaign -- so those rows got 17
columns instead of the conventional arm's 14, and since append_csv only
writes a header on the file's first row, pandas can't re-read the file.

Fix here is data-only: every 20 campaigns' hybrid rows have the exact same
extra-column shape (verified against the header), so the 3 garbage columns
(positions 13,14,15 -- _surrogate_score, _rerank_score, _rerank_rank) are
dropped, keeping the real proposal_score (position 16) in its rightful
place. No exact-culture data is touched; this only fixes an audit/design
ledger's column layout. The worker script is also fixed separately so this
never recurs on a future run.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS_ROOT = ROOT / "results" / "repaired_prospective_campaigns_v2"

EXPECTED_HEADER = [
    "campaign_id", "world_id", "method_id", "stage_index", "edit_vector_hash",
    "candidate_id", "strain_id", "edit_tier", "edits", "environment",
    "requested_assay_panel", "hypothesis", "candidate_hash", "proposal_score",
]


def repair_file(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if header != EXPECTED_HEADER:
        raise SystemExit(f"{path}: unexpected header {header}")

    fixed_rows = [header]
    n_fixed = 0
    for row in rows[1:]:
        if len(row) == 14:
            fixed_rows.append(row)
        elif len(row) == 17:
            fixed_rows.append(row[0:13] + [row[16]])
            n_fixed += 1
        else:
            raise SystemExit(f"{path}: unexpected row width {len(row)}: {row[:5]}...")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    return len(fixed_rows) - 1, n_fixed


def main() -> None:
    total_rows = 0
    total_fixed = 0
    for path in sorted(CAMPAIGNS_ROOT.glob("*/data/prospective_candidate_designs.csv")):
        n_rows, n_fixed = repair_file(path)
        total_rows += n_rows
        total_fixed += n_fixed
        print(f"{path.parent.parent.name}: {n_rows} rows, {n_fixed} repaired")
    print(f"TOTAL: {total_rows} rows, {total_fixed} repaired")


if __name__ == "__main__":
    main()
