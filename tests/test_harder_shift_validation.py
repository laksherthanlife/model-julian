from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_harder_shift_validation as hsv  # noqa: E402


def test_selected_world_specs_are_predeclared_and_interpretable():
    specs = hsv.selected_world_specs()
    assert len(specs) >= 6
    assert "baseline_world" in specs
    for world_id, spec in specs.items():
        assert spec["world_family"]
        assert spec["biological_interpretation"]
        assert world_id.endswith("_world")


def test_world_hash_changes_with_hidden_world_parameters():
    checksum = "gem"
    specs = hsv.selected_world_specs()
    baseline = hsv.world_payload("baseline_world", specs["baseline_world"], checksum)
    oxidative = hsv.world_payload("strong_oxidative_burden_world", specs["strong_oxidative_burden_world"], checksum)
    assert baseline["world_hash"] != oxidative["world_hash"]
    assert baseline["status"] == "frozen_before_method_evaluation"


def test_exact_task_hash_includes_world_hash():
    h1 = hsv.exact_task_hash("baseline_world", "world_a", "candidate", "difficulty_gate_landscape")
    h2 = hsv.exact_task_hash("baseline_world", "world_b", "candidate", "difficulty_gate_landscape")
    assert h1 != h2


def test_target_rules_are_method_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(hsv, "DATA", tmp_path)
    rules = hsv.write_target_rules()
    assert rules["not_tuned_to_method_outcomes"]
    assert set(rules["tiers"]) == {"easy", "moderate", "hard"}
    assert rules["tiers"]["hard"]["minimum_final_titer_fold_parent"] > rules["tiers"]["moderate"]["minimum_final_titer_fold_parent"]
    assert (tmp_path / "harder_shift_target_rules.json").exists()


def test_shift_exact_manifest_is_unique_while_decisions_keep_budgets():
    exact_manifest = ROOT / "data" / "distribution_shift_exact_task_manifest.csv"
    decision_manifest = ROOT / "data" / "distribution_shift_decision_manifest.csv"
    if not exact_manifest.exists() or not decision_manifest.exists():
        return
    exact_rows = pd.read_csv(exact_manifest)
    decision_rows = pd.read_csv(decision_manifest)
    assert exact_rows["exact_task_id"].is_unique
    assert set(decision_rows["adaptation_budget"].unique()) == set(hsv.ADAPTATION_BUDGETS)
    assert set(decision_rows["exact_task_id"]).issubset(set(exact_rows["exact_task_id"]))
