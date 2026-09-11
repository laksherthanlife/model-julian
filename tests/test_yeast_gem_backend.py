import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem


def asset_or_skip() -> Path:
    assets = gem.discover_assets()
    if not assets:
        pytest.skip("external yeast GEM asset is unavailable")
    return assets[0]


def test_asset_discovery_finds_raw_yeast_gem():
    assets = gem.discover_assets()
    assert any(path.name == "yeast-GEM.xml" for path in assets)


def test_missing_asset_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("YEAST_GEM_PATH", raising=False)
    monkeypatch.setattr(gem, "discover_assets", lambda: [])
    monkeypatch.setattr(gem, "MODELS", tmp_path)
    with pytest.raises(gem.GEMPilotError, match="No yeast GEM asset"):
        gem.configured_gem_path()


def test_cobra_loading_and_solver_minimal_lp():
    cobra = gem.require_cobra()
    path = asset_or_skip()
    model = gem.load_model(cobra, path)
    assert len(model.reactions) > 1000
    assert len(model.genes) > 100
    solver = gem.verify_minimal_lp(cobra)
    assert solver["minimal_lp_status"] == "optimal"
    assert solver["minimal_lp_objective"] == pytest.approx(10.0)
