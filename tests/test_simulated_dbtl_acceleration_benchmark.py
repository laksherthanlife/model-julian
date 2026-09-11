from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_simulated_dbtl_acceleration_benchmark as simdbtl  # noqa: E402


def test_protocol_declares_physical_culture_and_censoring_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(simdbtl, "DATA", tmp_path)
    monkeypatch.setattr(simdbtl, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(simdbtl, "ROLLOUTS", tmp_path / "results" / "hidden_wet_lab_rollouts")
    monkeypatch.setattr(simdbtl, "LEDGERS", tmp_path / "results" / "workflow_ledgers")
    protocol = simdbtl.protocol_config()
    target = simdbtl.target_config()
    assert "physical culture" in protocol["physical_culture_accounting"]
    assert "censored" in target["right_censoring_policy"]
    assert protocol["maximum_physical_cultures"] == 24


def test_method_access_matrix_blocks_hidden_information(tmp_path, monkeypatch):
    monkeypatch.setattr(simdbtl, "DATA", tmp_path)
    matrix = simdbtl.method_access_matrix()
    assert set(matrix["method_id"]) == set(simdbtl.FULL_METHODS)
    assert not matrix["hidden_simulator_internals_visible"].any()
    assert not matrix["other_method_observations_visible"].any()
    assert bool(matrix[matrix["method_id"].eq("biosensor_informed_modular_hybrid")]["physiological_reporters"].iloc[0])
    assert not bool(matrix[matrix["method_id"].eq("modular_hybrid_no_biosensors")]["physiological_reporters"].iloc[0])


def test_latents_are_reproducible_and_nontrivial():
    a = simdbtl.sample_latents(123)
    b = simdbtl.sample_latents(123)
    c = simdbtl.sample_latents(124)
    assert a == b
    assert a != c
    factors = simdbtl.latent_adjustment(a, "mixed_multi_mechanism_world")
    assert 0.25 <= factors["latent_product_multiplier"] <= 1.55
    assert 0.45 <= factors["latent_biomass_multiplier"] <= 1.20


def test_reporter_condition_visibility(tmp_path, monkeypatch):
    monkeypatch.setattr(simdbtl, "DATA", tmp_path)
    latent = simdbtl.sample_latents(42)
    full = simdbtl.reporter_measurements(latent, "full_reporters", "culture")
    none = simdbtl.reporter_measurements(latent, "no_reporters", "culture")
    assert pd.notna(full["oxidative_reporter"])
    assert pd.notna(full["atp_reporter"])
    assert pd.notna(full["pathway_reporter"])
    assert pd.isna(none["oxidative_reporter"])
    assert pd.isna(none["atp_reporter"])
    assert pd.isna(none["pathway_reporter"])


def test_culture_task_id_includes_simulator_fingerprint():
    first = simdbtl.culture_task_id("baseline_world", "candidate", 1, "full_reporters")
    second = simdbtl.culture_task_id("baseline_world", "candidate", 2, "full_reporters")
    third = simdbtl.culture_task_id("baseline_world", "candidate", 1, "no_reporters")
    assert first != second
    assert first != third
    assert len(first) == 24
