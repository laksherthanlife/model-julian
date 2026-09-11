#!/usr/bin/env python3
"""Generate persisted hybrid shortlists for the final validation HPC run.

This is Phase 1. It is pure NumPy/model scoring: no Yeast9 LP solves and no
hidden exact simulator calls. The output is the single source of truth for the
candidate JSONs used by the Vanda rerank array, avoiding cross-machine
floating-point differences at the top-50 cutoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"
BENCHMARK_ID = "final_validation_boundary_conditions_v1"
HYBRID_METHOD_ID = "pretrained_hybrid_digital_twin"
VIRTUAL_EVALUATIONS = 6000
SHORTLIST_SIZE = 50

DATA_SUFF_WORLDS = {
    "baseline_world": [98001, 98002, 98003],
    "strong_oxidative_burden_world": [98011, 98012, 98013],
}
ROBUSTNESS_WORLDS = {
    "atp_limited_world": [99001, 99002, 99003],
    "pathway_bottleneck_damage_world": [99011, 99012, 99013],
}


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["env_mean"], d["env_std"]


def campaign_id(group: str, world_id: str, seed: int, n_train: int, subsample_seed: int) -> str:
    return f"final_{group}_{world_id}_seed_{seed}_N{n_train}_sub{subsample_seed}"


def campaign_rows(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples():
        n_train = int(row.training_cultures_n)
        subsample_seed = int(row.subsample_seed)
        if n_train != 77:
            for world_id, seeds in DATA_SUFF_WORLDS.items():
                for seed in seeds:
                    rows.append(
                        {
                            "experiment_group": "data_sufficiency",
                            "world_id": world_id,
                            "campaign_seed": seed,
                            "training_cultures_n": n_train,
                            "subsample_seed": subsample_seed,
                            "checkpoint": row.checkpoint,
                            "normalization_stats": row.normalization_stats,
                        }
                    )
        elif subsample_seed == 0:
            for world_id, seeds in ROBUSTNESS_WORLDS.items():
                for seed in seeds:
                    rows.append(
                        {
                            "experiment_group": "physiology_robustness",
                            "world_id": world_id,
                            "campaign_seed": seed,
                            "training_cultures_n": n_train,
                            "subsample_seed": subsample_seed,
                            "checkpoint": row.checkpoint,
                            "normalization_stats": row.normalization_stats,
                        }
                    )
    for row in rows:
        row["campaign_id"] = campaign_id(
            str(row["experiment_group"]),
            str(row["world_id"]),
            int(row["campaign_seed"]),
            int(row["training_cultures_n"]),
            int(row["subsample_seed"]),
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="final_validation_hpc")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    candidates_dir = out_dir / "candidates"
    shortlists_dir = out_dir / "shortlists"
    virtual_dir = out_dir / "virtual_ledgers"
    for path in [candidates_dir, shortlists_dir, virtual_dir]:
        path.mkdir(parents=True, exist_ok=True)

    bench.BENCHMARK_ID = BENCHMARK_ID
    library = v2.build_common_intervention_library()
    surrogate = mlp.GSMSurrogateMLP.load(SURROGATE_PATH)
    training_manifest = pd.read_csv(DATA / "final_data_sufficiency_training_manifest.csv")
    campaigns = campaign_rows(training_manifest)

    campaign_table_rows = []
    task_rows = []
    for campaign_index, spec in enumerate(campaigns, start=1):
        cid = str(spec["campaign_id"])
        manifest_path = shortlists_dir / f"{cid}_shortlist_manifest.csv"
        virtual_path = virtual_dir / f"{cid}_virtual.csv"
        if manifest_path.exists() and virtual_path.exists() and not args.force:
            print(f"already done: {cid}")
        else:
            env_mean, env_std = load_stats(ROOT / str(spec["normalization_stats"]))
            scorer = scorer_mod.RepairedVirtualScorer.load(ROOT / str(spec["checkpoint"]), env_mean, env_std, surrogate=surrogate)
            factory = v2.make_factory(library, cid, int(spec["campaign_seed"]))
            shortlist, virtual = v2.generate_hybrid_shortlist(
                factory,
                VIRTUAL_EVALUATIONS,
                SHORTLIST_SIZE,
                set(),
                library,
                scorer,
                cid,
                str(spec["world_id"]),
            )
            rows = []
            for cand in shortlist:
                cand_path = candidates_dir / f"{cand['candidate_id']}.json"
                cand_path.write_text(json.dumps(cand, sort_keys=True), encoding="utf-8")
                rows.append(
                    {
                        **spec,
                        "candidate_id": cand["candidate_id"],
                        "candidate_hash": cand["candidate_hash"],
                        "candidate_json": str(cand_path),
                        "surrogate_score": float(cand["_surrogate_score"]),
                    }
                )
            short = pd.DataFrame(rows).sort_values("surrogate_score", ascending=False).reset_index(drop=True)
            short["surrogate_rank"] = short.index + 1
            short.to_csv(manifest_path, index=False)
            virtual["benchmark_id"] = BENCHMARK_ID
            virtual["experiment_group"] = spec["experiment_group"]
            virtual["training_cultures_n"] = spec["training_cultures_n"]
            virtual["subsample_seed"] = spec["subsample_seed"]
            virtual.to_csv(virtual_path, index=False)
            print(f"{cid}: shortlist={len(shortlist)}")

        campaign_table_rows.append({**spec, "campaign_index": campaign_index, "shortlist_manifest": str(manifest_path), "virtual_ledger": str(virtual_path)})
        short_df = pd.read_csv(manifest_path)
        for row in short_df.itertuples():
            task_rows.append(
                {
                    "array_index": len(task_rows) + 1,
                    "campaign_index": campaign_index,
                    "campaign_id": cid,
                    "experiment_group": spec["experiment_group"],
                    "world_id": spec["world_id"],
                    "campaign_seed": spec["campaign_seed"],
                    "training_cultures_n": spec["training_cultures_n"],
                    "subsample_seed": spec["subsample_seed"],
                    "checkpoint": spec["checkpoint"],
                    "normalization_stats": spec["normalization_stats"],
                    "candidate_id": row.candidate_id,
                    "candidate_hash": row.candidate_hash,
                    "candidate_json": row.candidate_json,
                    "surrogate_rank": int(row.surrogate_rank),
                    "surrogate_score": float(row.surrogate_score),
                }
            )

    pd.DataFrame(campaign_table_rows).to_csv(out_dir / "campaign_table.csv", index=False)
    pd.DataFrame(task_rows).to_csv(out_dir / "rerank_task_manifest.csv", index=False)
    conventional_rows = []
    for world_id, seeds in ROBUSTNESS_WORLDS.items():
        for seed in seeds:
            conventional_rows.append({"array_index": len(conventional_rows) + 1, "world_id": world_id, "campaign_seed": seed})
    pd.DataFrame(conventional_rows).to_csv(out_dir / "conventional_task_manifest.csv", index=False)
    print(f"wrote {len(campaign_table_rows)} campaigns and {len(task_rows)} rerank tasks")


if __name__ == "__main__":
    main()
