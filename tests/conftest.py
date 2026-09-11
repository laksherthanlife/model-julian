"""Test configuration for the notebook-support package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "src"), str(ROOT / "src" / "yeast_validation")):
    if path not in sys.path:
        sys.path.insert(0, path)
