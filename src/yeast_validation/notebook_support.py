"""Small, explicit utilities shared by the research notebooks."""

from __future__ import annotations

from pathlib import Path
import sys


def repository_root(start: Path | None = None) -> Path:
    """Return the repository root when a notebook runs from root or notebooks/."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "yeast_validation").exists() and (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing src/yeast_validation")


def configure_notebook(start: Path | None = None) -> Path:
    """Make the local package importable and return the repository root."""
    root = repository_root(start)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def require_cached(path: Path) -> Path:
    """Explain a missing cache without attempting an implicit expensive rerun."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cached artifact is missing: {path}. Set REGENERATE = True in this notebook "
            "and run the marked generation cell, or restore the artifact bundle."
        )
    return path
