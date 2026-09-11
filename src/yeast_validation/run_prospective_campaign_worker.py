#!/usr/bin/env python3
"""Run one frozen prospective DBTL world-seed campaign in an isolated shard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_prospective_dbtl_benchmark as prospective  # noqa: E402


INPUT_FILES = [
    "final_dbtl_actionable_intervention_library.csv",
    "gem_state_space_environment_grid.csv",
    "gem_state_space_trajectories.csv",
    "gem_state_space_reporters.csv",
    "gem_state_space_split_manifest.csv",
    "gem_state_space_solver_accounting.csv",
    "teacher_ensemble_manifest.csv",
    "teacher_channel_metrics.csv",
    "teacher_pseudodata.csv",
    "design_benchmark_oracle_cache_index.csv",
    "final_dbtl_exact_cache_index.csv",
    "simulated_dbtl_exact_cache_index.csv",
    "prospective_full_run_frozen_manifest.json",
]


def clear_shard_outputs(shard_dir: Path) -> None:
    """Remove rerunnable shard-local ledgers while preserving shared exact sidecars."""
    shard_data = shard_dir / "data"
    if shard_data.exists():
        for path in shard_data.glob("prospective_*.csv"):
            path.unlink()
    shard_figures = shard_dir / "figures"
    if shard_figures.exists():
        for path in shard_figures.glob("prospective_*"):
            if path.is_file():
                path.unlink()
    completion = shard_dir / "completion.json"
    if completion.exists():
        completion.unlink()


def copy_inputs(shard_data: Path) -> None:
    shard_data.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        src = ROOT / "data" / name
        if src.exists():
            shutil.copy2(src, shard_data / name)


def campaign_from_manifest(path: Path, array_index: int) -> tuple[str, int]:
    manifest = pd.read_csv(path)
    row = manifest[manifest["array_index"].astype(int).eq(int(array_index))]
    if row.empty:
        raise ValueError(f"Array index {array_index} not found in {path}")
    first = row.iloc[0]
    return str(first["world_id"]), int(first["campaign_seed"])


def configure_paths(shard_dir: Path) -> None:
    prospective.DATA = shard_dir / "data"
    prospective.FIGURES = shard_dir / "figures"
    prospective.RESULTS = ROOT / "results" / "prospective_dbtl_benchmark"
    prospective.ROLLOUTS = prospective.RESULTS / "on_demand_exact_rollouts"
    prospective.OLD_EXACT_CACHE_FILES = [
        prospective.DATA / "design_benchmark_oracle_cache_index.csv",
        prospective.DATA / "final_dbtl_exact_cache_index.csv",
        prospective.DATA / "simulated_dbtl_exact_cache_index.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/prospective_full_campaign_manifest.csv")
    parser.add_argument("--array-index", type=int, default=0)
    parser.add_argument("--world-id", default="")
    parser.add_argument("--campaign-seed", type=int, default=0)
    parser.add_argument("--shard-root", default="results/prospective_dbtl_benchmark/hpc_shards")
    args = parser.parse_args()

    array_index = args.array_index
    if array_index <= 0:
        array_index = int(__import__("os").environ.get("PBS_ARRAY_INDEX", "0"))
    if args.world_id and args.campaign_seed:
        world_id, seed = args.world_id, int(args.campaign_seed)
    else:
        world_id, seed = campaign_from_manifest(ROOT / args.manifest, array_index)

    shard_dir = ROOT / args.shard_root / f"{array_index:03d}_{world_id}_seed_{seed}"
    clear_shard_outputs(shard_dir)
    copy_inputs(shard_dir / "data")
    configure_paths(shard_dir)
    prospective.ensure_dirs()

    audit, ok = prospective.audit_historical_training_data()
    if not ok:
        raise RuntimeError("Historical training-data audit failed in prospective shard")
    library = prospective.load_library()
    teacher_lookup = prospective.load_teacher_final_lookup()
    prospective.run_campaign(
        world_id=world_id,
        seed=seed,
        library=library,
        teacher_lookup=teacher_lookup,
        conventional_batches=prospective.FULL_CONVENTIONAL_BATCHES,
        hybrid_batch=prospective.FULL_HYBRID_VERIFICATION_BATCH,
        virtual_evaluations=prospective.FULL_VIRTUAL_EVALUATIONS,
        mock=False,
        fresh_exact=False,
    )
    prospective.write_analysis()

    exact_calls = pd.read_csv(prospective.DATA / "prospective_exact_simulator_call_ledger.csv")
    completion = {
        "array_index": array_index,
        "world_id": world_id,
        "campaign_seed": seed,
        "campaign_id": f"prospective_{world_id}_seed_{seed}",
        "exact_cultures": int(exact_calls.shape[0]),
        "exact_lp_solves": int(exact_calls["n_actual_lp_solves"].sum()),
        "cache_hits": int(exact_calls["cache_hit"].astype(bool).sum()),
        "fresh_exact_simulations": int(exact_calls["fresh_exact_simulation_performed"].astype(bool).sum()),
        "solver_failures": int(exact_calls[["n_infeasible_solves", "n_unbounded_solves", "n_solver_errors", "n_skipped_intervals"]].fillna(0).sum().sum()),
    }
    (shard_dir / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
