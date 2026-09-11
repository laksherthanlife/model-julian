"""Small helpers used by the experiment notebooks.

The notebooks are intended to be readable first, so common repo-root discovery,
output checks, and table loading live here instead of being repeated in every
notebook.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    """Reproducibility contract for one notebook."""

    name: str
    command: Sequence[str] | None
    outputs: Mapping[str, str]
    notes: str = ""


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root from either repo root or notebooks/ cwd."""

    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "scripts").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing scripts/ and notebooks/.")


def configure_plots() -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-matplotlib"))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def command_text(spec: ExperimentSpec) -> str:
    if spec.command is None:
        return "No rerun command is registered for this historical bundled-output notebook."
    return " ".join(spec.command)


def ensure_outputs(spec: ExperimentSpec, *, rerun: bool = False, root: Path | None = None) -> Path:
    """Check notebook inputs and optionally rerun the owning experiment."""

    root = repo_root(root)
    missing = [rel for rel in spec.outputs.values() if not (root / rel).exists()]
    if missing and rerun and spec.command:
        subprocess.run(spec.command, cwd=root, check=True)
        missing = [rel for rel in spec.outputs.values() if not (root / rel).exists()]
    if missing:
        missing_text = "\n".join(f"- {rel}" for rel in missing)
        raise FileNotFoundError(
            f"{spec.name} is missing required outputs:\n{missing_text}\n\n"
            f"Rerun command: {command_text(spec)}"
        )
    return root


def load_tables(spec: ExperimentSpec, *, rerun: bool = False, root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = ensure_outputs(spec, rerun=rerun, root=root)
    return {name: pd.read_csv(root / rel) for name, rel in spec.outputs.items()}
