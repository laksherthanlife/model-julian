#!/usr/bin/env python3
"""Prepare a Vanda/PBS bundle for the full Stage A inverse GSM run."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "hpc_bundles" / "inverse_gsm_stageA_vanda"
YEAST_GEM_CANDIDATES = [
    ROOT / "models" / "yeast-GEM.xml",
    ROOT / "results" / "deployment_benchmark" / "hybrid_round_001_blind" / "inputs" / "yeast-GEM.xml",
    Path("/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"),
]
SCRIPT_FILES = [
    "aggregate_inverse_gsm_stageA_shards.py",
    "gem_backend.py",
    "run_gem_dynamic_capacity.py",
    "run_gem_state_space_validation.py",
    "run_inverse_gsm_calibration.py",
    "run_inverse_gsm_stageA_worker.py",
    "run_reporter_grounded_hybrid_distillation.py",
]
DATA_FILES = [
    "gem_state_space_environment_grid.csv",
    "gem_state_space_trajectories.csv",
    "gem_state_space_fluxes.csv",
    "gem_state_space_states.csv",
    "gem_state_space_reporters.csv",
    "gem_state_space_generator_constraints.partial.csv",
    "gem_state_space_solver_accounting.csv",
    "gem_state_space_split_manifest.csv",
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
    raise FileNotFoundError("Could not find yeast-GEM.xml for the Stage A Vanda bundle.")


def write_requirements(bundle: Path) -> None:
    requirements = [
        "pandas",
        "numpy",
        "cobra==0.31.1",
        "swiglpk==5.0.13",
        "sympy",
    ]
    (bundle / "requirements-hpc.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")


def write_manifest(bundle: Path) -> pd.DataFrame:
    optimizers = ["differential_evolution", "powell_multistart", "annealing"]
    restart_seeds = [11, 22, 33, 44, 55]
    rows = [
        {
            "array_index": 1,
            "mode": "reference",
            "optimizer": "",
            "restart_seed": "",
            "restart_index": "",
            "output_subdir": "reference",
        }
    ]
    index = 2
    for optimizer in optimizers:
        for restart_index, restart_seed in enumerate(restart_seeds):
            rows.append(
                {
                    "array_index": index,
                    "mode": "optimizer",
                    "optimizer": optimizer,
                    "restart_seed": restart_seed,
                    "restart_index": restart_index,
                    "output_subdir": f"{optimizer}_seed{restart_seed}",
                }
            )
            index += 1
    out = pd.DataFrame(rows)
    out.to_csv(bundle / "hpc" / "inverse_gsm_stageA_manifest.csv", index=False)
    return out


def write_worker_script(bundle: Path, array_tasks: int) -> None:
    (bundle / "hpc" / "run_inverse_gsm_stageA_array.pbs").write_text(
        f"""#!/bin/bash
#PBS -N inv_stageA
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=08:00:00
#PBS -J 1-{array_tasks}
#PBS -j oe

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd "$PBS_O_WORKDIR"

export YEAST_GEM_PATH="$PWD/models/yeast-GEM.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.mplconfig"
mkdir -p "$MPLCONFIGDIR" results/inverse_gsm_calibration/stageA_shards

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-hpc.txt
fi

row=$(.venv/bin/python - <<'PY'
import os
import pandas as pd
idx = int(os.environ["PBS_ARRAY_INDEX"])
df = pd.read_csv("hpc/inverse_gsm_stageA_manifest.csv")
row = df[df.array_index.eq(idx)].iloc[0]
print("|".join(str(row[c]) for c in ["mode", "optimizer", "restart_seed", "restart_index", "output_subdir"]))
PY
)
IFS='|' read -r mode optimizer restart_seed restart_index output_subdir <<< "$row"
output_root="results/inverse_gsm_calibration/stageA_shards/$output_subdir"

if [ "$mode" = "reference" ]; then
  .venv/bin/python scripts/run_inverse_gsm_stageA_worker.py \\
    --mode reference \\
    --output-root "$output_root" \\
    --random-samples 100 \\
    --landscape-points 5
else
  .venv/bin/python scripts/run_inverse_gsm_stageA_worker.py \\
    --mode optimizer \\
    --output-root "$output_root" \\
    --optimizer "$optimizer" \\
    --restart-seed "$restart_seed" \\
    --restart-index "$restart_index" \\
    --max-evaluations 500
fi
""",
        encoding="utf-8",
    )


def write_aggregate_script(bundle: Path) -> None:
    (bundle / "hpc" / "aggregate_inverse_gsm_stageA.pbs").write_text(
        """#!/bin/bash
#PBS -N inv_stageA_agg
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd "$PBS_O_WORKDIR"

export YEAST_GEM_PATH="$PWD/models/yeast-GEM.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

.venv/bin/python scripts/aggregate_inverse_gsm_stageA_shards.py \\
  --shard-root results/inverse_gsm_calibration/stageA_shards \\
  --max-evaluations 500 \\
  --random-samples 100 \\
  --restart-seeds 11 22 33 44 55
""",
        encoding="utf-8",
    )


def prepare_bundle(out_dir: Path, overwrite: bool = False) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["scripts", "data", "models", "figures", "hpc", "results/inverse_gsm_calibration/stageA_shards"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_FILES:
        copy_file(ROOT / "src" / "yeast_validation" / name, out_dir / "scripts" / name)
    for name in DATA_FILES:
        copy_file(ROOT / "data" / name, out_dir / "data" / name)
    copy_file(selected_yeast_gem(), out_dir / "models" / "yeast-GEM.xml")
    write_requirements(out_dir)
    manifest = write_manifest(out_dir)
    write_worker_script(out_dir, int(len(manifest)))
    write_aggregate_script(out_dir)
    (out_dir / "README_HPC.md").write_text(
        """# Inverse GSM Stage A Vanda Bundle

Run from the bundle root on Vanda:

```bash
qsub hpc/run_inverse_gsm_stageA_array.pbs
```

This submits 16 fresh-process tasks:

- array index 1: reference losses, 100 valid random controllers, 1D landscape
- array indices 2-16: one optimizer x one restart seed, 500 exact Yeast9 evaluations

After all 16 tasks complete, aggregate:

```bash
qsub hpc/aggregate_inverse_gsm_stageA.pbs
```

The aggregate job writes the canonical `data/inverse_gsm_stageA_*` CSVs and
`figures/inverse_gsm_stageA_*` SVGs. All jobs use queue-less PBS routing and
cap BLAS/OpenMP thread counts at one.
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
