#!/usr/bin/env python3
"""Prepare a minimal Vanda/HPC bundle for the final DBTL benchmark."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "hpc_bundles" / "final_dbtl_benchmark_vanda"
YEAST_GEM_CANDIDATES = [
    ROOT / "models" / "yeast-GEM.xml",
    ROOT / "results" / "deployment_benchmark" / "hybrid_round_001_blind" / "inputs" / "yeast-GEM.xml",
    Path("/Users/julianedberthartono/Jelly/igem/data/raw/yeast-GEM.xml"),
]
SCRIPT_FILES = [
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
DATA_PATTERNS = [
    "final_dbtl_*",
    "gem_capacity_selected_parameters.csv",
    "deployment_benchmark_operational_scenarios.csv",
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


def write_pbs(bundle: Path) -> None:
    jobs = bundle / "hpc" / "run_final_dbtl_vanda.pbs"
    jobs.parent.mkdir(parents=True, exist_ok=True)
    jobs.write_text(
"""#!/bin/bash
#PBS -N final_dbtl
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=72:00:00
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

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-hpc.txt
fi

.venv/bin/python -m py_compile scripts/run_final_dbtl_benchmark.py
.venv/bin/python scripts/run_final_dbtl_benchmark.py
.venv/bin/python scripts/run_final_dbtl_benchmark.py --summarize-only
""",
        encoding="utf-8",
    )


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


def prepare_bundle(out_dir: Path, overwrite: bool = False) -> Path:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["scripts", "data", "models", "results/final_dbtl_benchmark/exact_rollouts", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in SCRIPT_FILES:
        copy_file(ROOT / "src" / "yeast_validation" / name, out_dir / "scripts" / name)
    for pattern in DATA_PATTERNS:
        for src in sorted((ROOT / "data").glob(pattern)):
            if src.is_file():
                copy_file(src, out_dir / "data" / src.name)
    rollouts = ROOT / "results" / "final_dbtl_benchmark" / "exact_rollouts"
    if rollouts.exists():
        for src in sorted(rollouts.glob("*")):
            if src.is_file():
                copy_file(src, out_dir / "results" / "final_dbtl_benchmark" / "exact_rollouts" / src.name)
    copy_file(selected_yeast_gem(), out_dir / "models" / "yeast-GEM.xml")
    write_requirements(out_dir)
    write_pbs(out_dir)
    (out_dir / "README_HPC.md").write_text(
        """# Final DBTL Vanda Bundle

Run from the bundle root on Vanda:

```bash
qsub hpc/run_final_dbtl_vanda.pbs
```

The job resumes from `data/final_dbtl_observations.csv` and
`data/final_dbtl_exact_cache_index.csv` if present. `YEAST_GEM_PATH` is set to
`models/yeast-GEM.xml`. The final sequential protocol, methods, seeds,
objective, target, and intervention library are read from the frozen
`data/final_dbtl_*` files.
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
