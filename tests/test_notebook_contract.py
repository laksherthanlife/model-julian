"""Guard the notebook-first research interface without running costly campaigns."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_sequence_is_complete_and_scientific() -> None:
    paths = sorted(NOTEBOOKS.glob("0[1-8]_*.ipynb"))
    assert len(paths) == 8
    for path in paths:
        source = _source(path)
        assert "Question / motivation" in source
        assert "Experimental setup" in source
        assert "Interpretation and next phase" in source
        assert "REGENERATE = False" in source
        assert "yeast_validation" in source
        assert "subprocess.run" not in source


def test_notebooks_default_to_cached_evidence_without_shell_runners() -> None:
    for path in NOTEBOOKS.glob("0[1-8]_*.ipynb"):
        source = _source(path)
        assert "if REGENERATE:" in source
        assert "python scripts/" not in source
        assert "subprocess.run(" not in source


def test_research_implementation_is_not_a_scripts_forest() -> None:
    assert not (ROOT / "scripts").exists()
    assert (ROOT / "src" / "yeast_validation" / "notebook_support.py").exists()
