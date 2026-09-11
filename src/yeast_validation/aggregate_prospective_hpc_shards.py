#!/usr/bin/env python3
"""Merge prospective DBTL campaign shards and regenerate canonical analysis."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_prospective_dbtl_benchmark as prospective  # noqa: E402


RAW_TABLES = [
    "prospective_campaign_manifest.csv",
    "prospective_exact_simulator_call_ledger.csv",
    "prospective_biological_stage_ledger.csv",
    "prospective_virtual_evaluation_ledger.csv",
    "prospective_candidate_designs.csv",
    "prospective_transfer_novelty_audit.csv",
    "prospective_proposal_dependency_audit.csv",
]


def merge_table(shard_root: Path, table: str) -> pd.DataFrame:
    frames = []
    for path in sorted(shard_root.glob(f"*/data/{table}")):
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", default="results/prospective_dbtl_benchmark/hpc_shards")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shard_root = ROOT / args.shard_root
    if not shard_root.exists():
        raise FileNotFoundError(shard_root)

    completions = []
    for path in sorted(shard_root.glob("*/completion.json")):
        completions.append(pd.read_json(path, typ="series").to_dict())
    completion_df = pd.DataFrame(completions)
    completion_df.to_csv(ROOT / "data" / "prospective_full_hpc_completion_audit.csv", index=False)

    if args.overwrite:
        for table in RAW_TABLES:
            path = ROOT / "data" / table
            if path.exists():
                path.unlink()
    for table in RAW_TABLES:
        merged = merge_table(shard_root, table)
        if not merged.empty:
            merged.to_csv(ROOT / "data" / table, index=False)

    historical = ROOT / "results" / "prospective_dbtl_benchmark" / "frozen_exact_pilot_20260812" / "prospective_historical_training_audit.csv"
    if historical.exists():
        shutil.copy2(historical, ROOT / "data" / "prospective_historical_training_audit.csv")

    prospective.write_analysis()
    print(
        {
            "shards_completed": int(completion_df.shape[0]),
            "exact_cultures": int(completion_df["exact_cultures"].sum()) if not completion_df.empty else 0,
            "exact_lp_solves": int(completion_df["exact_lp_solves"].sum()) if not completion_df.empty else 0,
            "fresh_exact_simulations": int(completion_df["fresh_exact_simulations"].sum()) if not completion_df.empty else 0,
            "cache_hits": int(completion_df["cache_hits"].sum()) if not completion_df.empty else 0,
        }
    )


if __name__ == "__main__":
    main()
