from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_final_dbtl_benchmark as final_run  # noqa: E402


def test_edit_from_intervention_preserves_finite_bounds():
    row = pd.Series(
        {
            "intervention_id": "int_001",
            "reaction_id": "r_001",
            "edit_type": "reaction_capacity_increase",
            "capacity_multiplier": 1.5,
            "target_lower_bound": np.nan,
            "target_upper_bound": 6.0,
            "reference_upper_bound": 4.0,
        }
    )
    edit = final_run.edit_from_intervention(row)
    assert edit["reaction_id"] == "r_001"
    assert edit["target_upper_bound"] == 6.0
    assert edit["reference_upper_bound"] == 4.0
    assert "target_lower_bound" not in edit


def test_canonical_design_key_is_invariant_to_edit_order():
    env = {"temperature": 30.0, "pH": 5.0, "DO": 65.0}
    edits = [
        {"reaction_id": "b", "edit_type": "reaction_capacity_decrease", "capacity_multiplier": 0.5},
        {"reaction_id": "a", "edit_type": "reaction_capacity_increase", "capacity_multiplier": 1.2},
    ]
    assert final_run.canonical_design_key(edits, env) == final_run.canonical_design_key(list(reversed(edits)), dict(reversed(env.items())))


def test_method_campaign_summary_accounts_for_success_and_censoring():
    obs = pd.DataFrame(
        [
            {
                "campaign_id": "c1",
                "method_id": "m1",
                "culture_index": 1,
                "round_index": 1,
                "reaches_target": False,
                "utility": 0.5,
                "final_product": 1.0,
                "product_AUC": 2.0,
                "productivity": 0.1,
                "yield_proxy": 0.2,
                "final_biomass": 3.0,
                "severe_growth_collapse": False,
                "exact_status": "complete_exact_dynamic_pfba",
                "invalid_proposals_this_round": 0,
                "duplicate_proposals_this_round": 0,
                "effective_phenotype_fingerprint": "p1",
                "effective_model_hash": "e1",
                "physical_culture_charge": 1,
                "cache_hit": False,
                "n_actual_lp_solves": 144,
                "wall_time_seconds": 1.0,
                "virtual_candidate_evaluations_cumulative": 420,
            },
            {
                "campaign_id": "c1",
                "method_id": "m1",
                "culture_index": 2,
                "round_index": 1,
                "reaches_target": True,
                "utility": 1.5,
                "final_product": 2.0,
                "product_AUC": 3.0,
                "productivity": 0.2,
                "yield_proxy": 0.3,
                "final_biomass": 4.0,
                "severe_growth_collapse": False,
                "exact_status": "complete_exact_dynamic_pfba",
                "invalid_proposals_this_round": 0,
                "duplicate_proposals_this_round": 1,
                "effective_phenotype_fingerprint": "p2",
                "effective_model_hash": "e2",
                "physical_culture_charge": 1,
                "cache_hit": False,
                "n_actual_lp_solves": 144,
                "wall_time_seconds": 2.0,
                "virtual_candidate_evaluations_cumulative": 420,
            },
            {
                "campaign_id": "c1",
                "method_id": "m2",
                "culture_index": 1,
                "round_index": 1,
                "reaches_target": False,
                "utility": 0.25,
                "final_product": 0.5,
                "product_AUC": 1.0,
                "productivity": 0.05,
                "yield_proxy": 0.1,
                "final_biomass": 2.0,
                "severe_growth_collapse": False,
                "exact_status": "complete_exact_dynamic_pfba",
                "invalid_proposals_this_round": 0,
                "duplicate_proposals_this_round": 0,
                "effective_phenotype_fingerprint": "p3",
                "effective_model_hash": "e3",
                "physical_culture_charge": 1,
                "cache_hit": False,
                "n_actual_lp_solves": 144,
                "wall_time_seconds": 1.0,
                "virtual_candidate_evaluations_cumulative": 420,
            },
        ]
    )
    summary = final_run.method_campaign_summary(obs).set_index("method_id")
    assert bool(summary.loc["m1", "success"])
    assert summary.loc["m1", "cultures_to_target"] == 2
    assert summary.loc["m1", "best_utility"] == 1.5
    assert summary.loc["m1", "exact_lp_solves"] == 288
    assert not bool(summary.loc["m2", "success"])
    assert summary.loc["m2", "right_censored_cultures"] == ">16"


def test_bootstrap_ci_returns_finite_bounds_for_nonempty_input():
    low, high = final_run.bootstrap_ci(np.array([1.0, 2.0, 3.0]), n_boot=100, seed=7)
    assert np.isfinite(low)
    assert np.isfinite(high)
    assert low <= high
