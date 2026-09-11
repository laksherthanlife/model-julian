#!/usr/bin/env python3
"""Prepare a local yeast GEM asset for non-fast dFBA validation.

The canonical experiment runner never downloads or substitutes a GEM.  This
helper locates a user-provided asset, records dependency versions, and prints
clear placement instructions when the model is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DEFAULT_PATH = MODELS / "yeast_gem.xml"


def candidate_paths(explicit: str | None = None) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    env_path = os.environ.get("YEAST_GEM_PATH")
    if env_path:
        paths.append(Path(env_path))
    paths.extend([DEFAULT_PATH, MODELS / "yeast_gem.json", MODELS / "yeast9.xml", MODELS / "yeast8.xml"])
    return paths


def find_asset(explicit: str | None = None) -> Path | None:
    for path in candidate_paths(explicit):
        if path.exists() and path.is_file():
            return path
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_rows() -> list[dict[str, str]]:
    rows = []
    for package, module_name in [
        ("cobra", "cobra"),
        ("swiglpk", "swiglpk"),
        ("optlang", "optlang"),
        ("python-libsbml", "libsbml"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
    ]:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            status = "installed"
        except Exception as exc:
            version = ""
            status = f"missing: {type(exc).__name__}"
        rows.append({"package": package, "import_name": module_name, "version": str(version), "status": status})
    return rows


def write_dependency_versions() -> None:
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(dependency_rows()).to_csv(DATA / "yeast_gem_dependency_versions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem-path", default=None, help="Explicit Yeast8/Yeast9 SBML or JSON file.")
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    write_dependency_versions()
    asset = find_asset(args.gem_path)
    if asset is None:
        print(
            f"""Yeast GEM asset not found.

Provide a documented Saccharomyces cerevisiae GEM, preferably Yeast9 or Yeast8.
The canonical runner will not download a model and will not silently substitute
the reduced surrogate.

Supported locations:
  1. Set YEAST_GEM_PATH=/absolute/path/to/yeast_gem.xml
  2. Place the asset at {DEFAULT_PATH}

Then run:
  .venv/bin/python scripts/audit_yeast_gem.py
  .venv/bin/python scripts/run_dfba_state_machine_validation.py --backend yeast_gem

Recorded dependency versions:
  data/yeast_gem_dependency_versions.csv
"""
        )
        return

    rows = [
        {
            "asset_path": str(asset.resolve()),
            "asset_name": asset.name,
            "suffix": asset.suffix.lower(),
            "sha256": sha256(asset),
            "source": "YEAST_GEM_PATH" if os.environ.get("YEAST_GEM_PATH") and Path(os.environ["YEAST_GEM_PATH"]) == asset else "local_file",
            "canonical_copy_expected": str(DEFAULT_PATH),
            "status": "found",
        }
    ]
    pd.DataFrame(rows).to_csv(DATA / "yeast_gem_asset_manifest.csv", index=False)
    print(f"Found yeast GEM asset: {asset}")
    print(f"sha256: {rows[0]['sha256']}")
    print("Wrote data/yeast_gem_asset_manifest.csv")
    print("Wrote data/yeast_gem_dependency_versions.csv")


if __name__ == "__main__":
    main()
