import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import stage4_design as stage4


def test_edit_library_bounds_and_mapping():
    library = stage4.default_edit_library()
    assert 8 <= len(library) <= 15
    required = {
        "edit_id",
        "biological_description",
        "gem_reaction_id",
        "constraint_mapping",
        "edit_type",
        "baseline_value",
        "permitted_minimum",
        "permitted_maximum",
        "zero_allowed",
        "reversible",
        "edit_cost_score",
        "assumed_uncertainty",
        "transfer_risk_category",
    }
    assert required.issubset(library.columns)
    assert set(stage4.EDIT_COLUMNS) == set(library["edit_id"])
    assert (library["permitted_minimum"] <= library["baseline_value"]).all()
    assert (library["baseline_value"] <= library["permitted_maximum"]).all()
    assert not library["gem_reaction_id"].isna().any()


def test_baseline_edit_identity_and_distance():
    library = stage4.default_edit_library()
    baseline = {row.edit_id: float(row.baseline_value) for row in library.itertuples(index=False)}
    assert stage4.edit_distance(baseline, library) == pytest.approx(0.0)
    assert stage4.edit_cost(baseline, library) == pytest.approx(0.0)
    edited = baseline.copy()
    edited["PSY_capacity_multiplier"] = 1.5
    assert stage4.edit_distance(edited, library) > 0.0


def test_edit_bounds_reject_out_of_domain():
    library = stage4.default_edit_library()
    bad = {row.edit_id: float(row.baseline_value) for row in library.itertuples(index=False)}
    bad["PSY_capacity_multiplier"] = 99.0
    with pytest.raises(ValueError):
        stage4.validate_edit_vector(bad, library)


def test_stage3_gate_blocks_stage4_acceptance_until_exact_replay_complete():
    gate = stage4.stage3_gate()
    assert gate.complete_pairs < 375 or gate.status != "stage_3_passed_ready_for_metabolic_edits"
    assert not gate.passed


def test_calibration_manifest_has_no_exact_test_leakage_labels():
    manifest = stage4.select_calibration_manifest(n=12, seed=123)
    assert len(manifest) == 12
    assert manifest["calibration_case_id"].is_unique
    assert manifest["expected_lp_solves"].eq(144).all()
    assert manifest["exact_yeast9_status"].eq("pending_stage3_gate").all()
    assert set(stage4.EDIT_COLUMNS).issubset(manifest.columns)


def test_surrogate_training_uses_backend_labels_and_no_surrogate_as_fba():
    _models, metrics = stage4.train_surrogate(n_ensemble=2, seed=222)
    assert metrics["metabolic_backend"].eq("edit_aware_surrogate").all()
    manifest = pd.read_csv(stage4.DATA / "stage4_edit_surrogate_training_manifest.csv")
    assert "teacher_pseudodata_not_exact_fba" in set(manifest["metabolic_backend"])
    assert "yeast_gem_lp" in set(manifest["metabolic_backend"])


def test_candidate_rollout_is_trajectory_based_and_bounded():
    library = stage4.default_edit_library()
    models, _metrics = stage4.train_surrogate(n_ensemble=2, seed=333)
    interval_len = 48
    controls = {col: np.ones(interval_len) for col in stage4.hybrid.INTERFACE_COLUMNS}
    controls["oxygen_lower_bound"] = np.full(interval_len, -4.0)
    controls["gamma_growth_fraction"] = np.full(interval_len, 0.6)
    controls["z_ox"] = np.zeros(interval_len)
    controls["z_atp"] = np.zeros(interval_len)
    controls["z_bottle"] = np.zeros(interval_len)
    edit = {row.edit_id: float(row.baseline_value) for row in library.itertuples(index=False)}
    out = stage4.rollout_candidate(models, np.array([30.0, 5.0, 50.0]), controls, edit, library)
    assert out["metabolic_backend"] == "edit_aware_surrogate"
    assert out["expected_final_product"] >= 0.0
    assert 0.0 <= out["feasibility_probability"] <= 1.0
    assert out["surrogate_uncertainty"] >= 0.0


def test_diverse_shortlist_and_verification_placeholders():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": f"c{i}",
                "temperature": 27 + i * 0.5,
                "pH": 4.5 + (i % 5) * 0.1,
                "DO": 20 + i * 3,
                "edit_vector_id": f"edit_{i % 4}",
                "expected_score": 10 - i * 0.1,
                "robust_score": 9 - i * 0.1,
                "LCB_score": 8 - i * 0.1,
                "integrated_oxidative_stress": i * 0.1,
                "edit_cost": i * 0.05,
                "expected_AUC": 5 + i,
                "expected_final_product": 1 + i * 0.1,
                "feasibility_probability": 0.9,
            }
            for i in range(30)
        ]
    )
    shortlist = stage4.diverse_shortlist(candidates, n=6)
    assert len(shortlist) == 6
    assert shortlist["candidate_id"].is_unique
    assert shortlist["exact_yeast9_verification_status"].eq("blocked_pending_stage3_gate").all()


def test_audit_writes_blocked_status_with_surrogate_candidates():
    acceptance = stage4.audit_design_system(n_candidates=120, seed=777)
    assert acceptance["stage_4_status"].str.startswith("stage_4_blocked_").all()
    search = pd.read_csv(stage4.DATA / "stage4_search_candidates.csv")
    assert len(search) == 120
    assert search["metabolic_backend"].eq("edit_aware_surrogate").all()
    verified = pd.read_csv(stage4.DATA / "stage4_exact_verified_candidates.csv")
    assert verified["exact_yeast9_verification_status"].eq("blocked_pending_stage3_gate").all()
