#!/usr/bin/env python3
"""Prepare a Vanda/PBS bundle for the full prospective DBTL benchmark."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "hpc_bundles" / "prospective_dbtl_benchmark_vanda"
YEAST_GEM_CANDIDATES = [
    ROOT / "models" / "yeast-GEM.xml",
    ROOT / "results" / "deployment_benchmark" / "hybrid_round_001_blind" / "inputs" / "yeast-GEM.xml",
    Path("/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"),
]
SCRIPT_FILES = [
    "run_prospective_dbtl_benchmark.py",
    "run_prospective_campaign_worker.py",
    "run_prospective_exact_worker.py",
    "aggregate_prospective_hpc_shards.py",
    "run_final_dbtl_benchmark.py",
    "final_dbtl_benchmark_freeze.py",
    "run_deployment_verifier.py",
    "deployment_benchmark.py",
    "design_benchmark_exact.py",
    "design_benchmark.py",
    "gem_backend.py",
    "run_gem_dynamic_capacity.py",
    "stage4_design.py",
    "run_reporter_grounded_hybrid_distillation.py",
    "run_gem_state_space_validation.py",
    "run_fixed_environment_validation.py",
]
DATA_FILES = [
    "final_dbtl_actionable_intervention_library.csv",
    "gem_capacity_selected_parameters.csv",
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


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def selected_yeast_gem() -> Path:
    for path in YEAST_GEM_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find yeast-GEM.xml for the HPC bundle.")


def write_requirements(bundle: Path) -> None:
    requirements = [
        "pandas",
        "numpy",
        "matplotlib",
        "cobra==0.31.1",
        "swiglpk==5.0.13",
        "scipy",
        "scikit-learn",
        "sympy",
    ]
    (bundle / "requirements-hpc.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def write_campaign_manifest(bundle: Path) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))
    import run_prospective_dbtl_benchmark as prospective

    worlds = ["baseline_world", "strong_oxidative_burden_world"]
    seeds = prospective.CAMPAIGN_SEEDS[:10]
    rows = []
    idx = 1
    for world_id in worlds:
        for seed in seeds:
            rows.append({"array_index": idx, "world_id": world_id, "campaign_seed": seed, "campaign_id": f"prospective_{world_id}_seed_{seed}"})
            idx += 1
    pd.DataFrame(rows).to_csv(bundle / "data" / "prospective_full_campaign_manifest.csv", index=False)


def write_pbs(bundle: Path) -> None:
    path = bundle / "hpc" / "run_prospective_dbtl_campaign_array.pbs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
"""#!/bin/bash
#PBS -N prosp_dbtl
#PBS -J 1-20
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=08:00:00
#PBS -j oe

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
python3 --version
cd "$PBS_O_WORKDIR"

export YEAST_GEM_PATH="$PWD/models/yeast-GEM.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig"
export PROSPECTIVE_EXACT_SUBPROCESS=1
mkdir -p "$MPLCONFIGDIR"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-hpc.txt
fi

.venv/bin/python -m py_compile scripts/run_prospective_campaign_worker.py
.venv/bin/python scripts/run_prospective_campaign_worker.py --array-index "$PBS_ARRAY_INDEX"
""",
        encoding="utf-8",
    )


def prepare_bundle(out_dir: Path, overwrite: bool = False) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["scripts", "data", "models", "results/prospective_dbtl_benchmark/on_demand_exact_rollouts", "figures", "hpc"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_FILES:
        copy_file(ROOT / "src" / "yeast_validation" / name, out_dir / "scripts" / name)
    for name in DATA_FILES:
        src = ROOT / "data" / name
        if src.exists():
            copy_file(src, out_dir / "data" / name)
    copy_file(selected_yeast_gem(), out_dir / "models" / "yeast-GEM.xml")
    source_rollouts = ROOT / "results" / "prospective_dbtl_benchmark" / "on_demand_exact_rollouts"
    if source_rollouts.exists():
        for src in sorted(source_rollouts.glob("*.csv")):
            copy_file(src, out_dir / "results" / "prospective_dbtl_benchmark" / "on_demand_exact_rollouts" / src.name)
    write_campaign_manifest(out_dir)
    write_requirements(out_dir)
    write_pbs(out_dir)
    (out_dir / "README_HPC.md").write_text(
        """# Prospective DBTL Vanda Bundle

Run from the bundle root on Vanda:

```bash
qsub hpc/run_prospective_dbtl_campaign_array.pbs
```

This is the frozen full prospective on-demand DBTL benchmark: 10 seeds x 2
worlds, conventional batches 4,4,4,4, hybrid verification batch 8, and 6000
hybrid virtual evaluations per campaign. Each PBS array element runs one
world-seed campaign in an isolated shard and shares only the on-demand exact
rollout sidecar directory. After all 20 shards complete, merge with:

```bash
.venv/bin/python scripts/aggregate_prospective_hpc_shards.py --overwrite
```
""",
        encoding="utf-8",
    )
    archive = out_dir.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    archive = prepare_bundle(Path(args.out_dir), overwrite=args.overwrite)
    print(archive)


if __name__ == "__main__":
    main()
