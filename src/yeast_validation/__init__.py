"""Reusable implementation for the Yeast Digital Twin validation notebooks.

The notebooks are the public, scientific entry points.  This package contains
only the generators, model implementations, metrics, and expensive evaluation
functions that those notebooks exercise.  It deliberately keeps the original
module names for now because cached artifacts and published reports record
them; compatibility aliases are a safer migration than silently changing an
experiment's semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Historical modules use absolute sibling imports (for example ``import
# gem_backend``).  Keep them importable while notebooks progressively use the
# package-qualified API.  This does not execute experiments or alter paths
# outside this source tree.
_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

