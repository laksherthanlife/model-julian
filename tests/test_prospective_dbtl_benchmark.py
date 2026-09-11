from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_prospective_dbtl_benchmark as prospective  # noqa: E402


def small_library() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "intervention_id": "i_psy",
                "reaction_id": "BETA_PHYTOENE_SYNTHASE",
                "edit_type": "pathway_enzyme_capacity_modification",
                "capacity_multiplier": 1.35,
                "mechanism_class": "direct_heterologous_pathway_capacity",
                "dynamic_effect_size": 0.20,
                "staged_effect_size": 0.10,
                "risk_level": "low",
            },
            {
                "intervention_id": "i_atp",
                "reaction_id": "r_4046",
                "edit_type": "reaction_capacity_decrease_demand",
                "capacity_multiplier": 0.80,
                "mechanism_class": "atp_demand_reduction",
                "dynamic_effect_size": 0.12,
                "staged_effect_size": 0.05,
                "risk_level": "medium",
            },
            {
                "intervention_id": "i_o2",
                "reaction_id": "r_1992",
                "edit_type": "exchange_uptake_or_secretion_change",
                "capacity_multiplier": 1.25,
                "mechanism_class": "oxygen_exchange",
                "dynamic_effect_size": 0.08,
                "staged_effect_size": 0.04,
                "risk_level": "medium",
            },
        ]
    )


def test_historical_training_audit_passes_current_canonical_artifacts():
    audit, ok = prospective.audit_historical_training_data()
    assert ok
    table = audit.set_index("audit_item")
    assert table.loc["teacher_training_strains", "finding"] == "single_reference_no_edit_strain"
    assert table.loc["metabolic_edit_dimensions_in_training", "finding"] == "absent"
    assert table.loc["deployment_inputs", "finding"] == "temperature,pH,DO"


def test_mock_campaign_has_online_conventional_stages_and_virtual_only_hybrid(tmp_path, monkeypatch):
    monkeypatch.setattr(prospective, "DATA", tmp_path / "data")
    monkeypatch.setattr(prospective, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(prospective, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(prospective, "ROLLOUTS", tmp_path / "results" / "rollouts")
    prospective.ensure_dirs()
    lib = small_library()
    teacher_lookup = pd.DataFrame(
        [
            {"temperature": 30.0, "pH": 5.0, "DO": 50.0, "B_total_final_mean": 0.5},
            {"temperature": 30.0, "pH": 5.0, "DO": 80.0, "B_total_final_mean": 0.7},
        ]
    )

    prospective.run_campaign(
        world_id="baseline_world",
        seed=1,
        library=lib,
        teacher_lookup=teacher_lookup,
        conventional_batches=[1, 1],
        hybrid_batch=1,
        virtual_evaluations=5,
        mock=True,
    )

    exact = pd.read_csv(tmp_path / "data" / "prospective_exact_simulator_call_ledger.csv")
    stages = pd.read_csv(tmp_path / "data" / "prospective_biological_stage_ledger.csv")
    virtual = pd.read_csv(tmp_path / "data" / "prospective_virtual_evaluation_ledger.csv")

    assert exact.shape[0] == 3
    assert int(stages[stages["method_id"].eq("conventional_dbtl")]["stage_index"].max()) == 2
    assert virtual.shape[0] == 5
    assert not virtual["exact_outcome_accessed"].astype(bool).any()
    assert set(exact["method_id"]) == {"conventional_dbtl", "pretrained_hybrid_digital_twin"}


def test_culture_task_id_includes_stage_and_method():
    candidate = {
        "candidate_id": "c",
        "candidate_hash": "h",
        "strain_id": "s",
        "edits": [],
        "environment": {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
    }
    a = prospective.culture_task_id("camp", "conventional_dbtl", "baseline_world", 1, candidate)
    b = prospective.culture_task_id("camp", "conventional_dbtl", "baseline_world", 2, candidate)
    c = prospective.culture_task_id("camp", "pretrained_hybrid_digital_twin", "baseline_world", 1, candidate)
    assert a != b
    assert a != c
    assert len(a) == 24
