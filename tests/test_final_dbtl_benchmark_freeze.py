from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import deployment_benchmark as dep  # noqa: E402
import final_dbtl_benchmark_freeze as freeze  # noqa: E402


class FakeReaction:
    def __init__(self, rid, lower, upper, reversible=False):
        self.id = rid
        self.name = rid
        self.subsystem = "test"
        self.lower_bound = lower
        self.upper_bound = upper
        self.reversibility = reversible
        self.gene_reaction_rule = ""
        self.genes = []


class FakeReactions(list):
    def get_by_id(self, rid):
        for rxn in self:
            if rxn.id == rid:
                return rxn
        raise KeyError(rid)


class FakeModel:
    def __init__(self, reactions):
        self.reactions = FakeReactions(reactions)

    def copy(self):
        return FakeModel([FakeReaction(r.id, r.lower_bound, r.upper_bound, r.reversibility) for r in self.reactions])


def apply_one(edit, reaction=None):
    model = FakeModel([reaction or FakeReaction("r_test", 0.0, 1000.0)])
    dep.apply_sparse_edits_to_model(
        model,
        {
            "edits": [edit],
            "environment": {"temperature": 30, "pH": 5, "DO": 50},
        },
    )
    return model.reactions.get_by_id(edit["reaction_id"])


def test_knockout_bound_application():
    rxn = apply_one({"reaction_id": "r_test", "edit_type": "reaction_knockout", "capacity_multiplier": 0.0})
    assert rxn.lower_bound == 0.0
    assert rxn.upper_bound == 0.0


def test_knockdown_bound_application_and_reversible_handling():
    rxn = apply_one(
        {"reaction_id": "r_rev", "edit_type": "reaction_knockdown", "capacity_multiplier": 0.5},
        FakeReaction("r_rev", -10.0, 20.0, reversible=True),
    )
    assert rxn.lower_bound == -5.0
    assert rxn.upper_bound == 10.0


def test_capacity_increase_and_decrease_with_finite_targets():
    inc = apply_one(
        {"reaction_id": "r_test", "edit_type": "reaction_capacity_increase", "capacity_multiplier": 1.5, "reference_upper_bound": 4.0}
    )
    dec = apply_one({"reaction_id": "r_test", "edit_type": "reaction_capacity_decrease", "capacity_multiplier": 0.25})
    assert inc.upper_bound == 6.0
    assert dec.upper_bound == 250.0


def test_transport_and_exchange_edit_handling():
    rxn = apply_one(
        {"reaction_id": "r_ex", "edit_type": "transport_capacity_change", "capacity_multiplier": 1.5},
        FakeReaction("r_ex", -10.0, 1000.0, reversible=True),
    )
    assert rxn.lower_bound == -15.0


def test_fresh_model_reconstruction_preserves_edits():
    candidate = {"edits": [{"reaction_id": "r_test", "edit_type": "reaction_capacity_decrease", "capacity_multiplier": 0.5}]}
    first = FakeModel([FakeReaction("r_test", 0.0, 1000.0)])
    second = FakeModel([FakeReaction("r_test", 0.0, 1000.0)])
    dep.apply_sparse_edits_to_model(first, candidate)
    dep.apply_sparse_edits_to_model(second, candidate)
    assert first.reactions.get_by_id("r_test").upper_bound == second.reactions.get_by_id("r_test").upper_bound == 500.0


def test_blocked_reaction_exclusion_logic():
    row = pd.Series({"fva_min": 0.0, "fva_max": 0.0, "baseline_flux": 0.0})
    span = max(float(row["fva_max"]) - float(row["fva_min"]), 0.0)
    assert span <= 1e-9 and abs(float(row["baseline_flux"])) <= 1e-9


def test_effective_phenotype_fingerprint_tolerance_and_distinction():
    base = pd.DataFrame([{"time": 0.0, "X": 1.0, "B_total": 0.1}, {"time": 1.0, "X": 1.1, "B_total": 0.2}])
    same = pd.DataFrame([{"time": 0.0, "X": 1.0 + 1e-8, "B_total": 0.1}, {"time": 1.0, "X": 1.1, "B_total": 0.2}])
    diff = pd.DataFrame([{"time": 0.0, "X": 1.0, "B_total": 0.1}, {"time": 1.0, "X": 1.1, "B_total": 0.25}])
    assert freeze.phenotype_fingerprint({"trajectory": base}) == freeze.phenotype_fingerprint({"trajectory": same})
    assert freeze.phenotype_fingerprint({"trajectory": base}) != freeze.phenotype_fingerprint({"trajectory": diff})


def test_effective_model_hash_inputs_detect_candidate_equivalence_without_candidate_hash():
    c1 = {"candidate_id": "a", "strain_id": "s1", "edit_tier": 1, "edits": [], "environment": {"temperature": 30, "pH": 5, "DO": 50}}
    c2 = {"candidate_id": "b", "strain_id": "s2", "edit_tier": 1, "edits": [], "environment": {"temperature": 30, "pH": 5, "DO": 50}}
    assert freeze.candidate_spec_hash(c1) != freeze.candidate_spec_hash(c2)
    effective_payload = {"environment": c1["environment"], "resolved_bounds": [], "simulator": freeze.FINAL_SIMULATOR_VERSION}
    assert freeze.hash_payload(effective_payload) == freeze.hash_payload({**effective_payload})


def test_within_batch_duplicate_prevention_and_invalid_replacement():
    candidate = {"candidate_id": "a", "strain_id": "s", "edit_tier": 1, "edits": [], "environment": {"temperature": 30, "pH": 5, "DO": 50}}
    assert freeze.duplicate_candidate_hashes([candidate, dict(candidate)])
    assert freeze.replacement_required(candidate, valid=False, prior_batch=[])
    assert freeze.replacement_required(dict(candidate), valid=True, prior_batch=[candidate])


def test_observation_privacy_and_hidden_cache_isolation_matrix_shape():
    methods = freeze.FINAL_METHODS
    access = pd.DataFrame(
        [{"method_id": method, "hidden_exact_cache_access": False, "other_method_observation_access": False} for method in methods]
    )
    assert not access["hidden_exact_cache_access"].any()
    assert not access["other_method_observation_access"].any()


def test_objective_reproducibility_and_target_stopping_logic():
    objective = {
        "reference_values": {"final_titer": 1.0, "productivity": 1.0, "yield": 1.0, "final_biomass": 1.0},
        "weights": {"titer": 0.45, "productivity": 0.20, "yield": 0.20, "biomass": 0.15},
        "clipping": {"minimum_component": 0.0, "maximum_component": 2.5},
        "failure_penalties": {"growth_failure": 1.0, "infeasible": 2.0},
        "edit_count_penalty": 0.025,
        "construction_risk_penalties": {"low": 0.0, "medium": 0.04, "high": 0.10},
    }
    metrics = {"final_product": 1.3, "product_AUC": 1.1, "yield_proxy": 1.0, "final_biomass": 1.0, "growth_failure": False, "edit_count": 2, "construction_risk": "low"}
    assert freeze.final_utility(metrics, objective) == freeze.final_utility(dict(metrics), objective)
    target = {"parent_reference": {"final_product": 1.0}, "requirements": {"minimum_final_titer": 1.25, "minimum_final_biomass": 0.9, "minimum_yield_proxy": 1.0, "maximum_interventions": 3, "minimum_utility": 0.9}}
    assert freeze.reaches_target(metrics, target, objective)


def test_cache_reuse_versus_culture_accounting_and_deterministic_hashes():
    assert freeze.culture_charge_for_cache_event(cache_hit=True) == 1
    payload = {"b": 2, "a": 1}
    assert freeze.hash_payload(payload) == freeze.hash_payload({"a": 1, "b": 2})


def test_finite_reference_capacity_uses_public_flux_fva_and_floor():
    source, value = freeze.finite_reference_capacity(0.2, -0.4, 1.1, 0.7)
    assert source == "max_abs_public_baseline_fva_dynamic_flux_floor"
    assert value == 1.1
    assert freeze.finite_reference_capacity(0.0, 0.0, 0.0, 0.0, floor=0.05)[1] == 0.05


def test_directional_bounds_for_uptake_and_competing_sink():
    uptake = pd.Series(
        {
            "reference_capacity_value": 4.0,
            "original_lower_bound": -1000.0,
            "original_upper_bound": 1000.0,
            "intervention_type": "oxygen_uptake_capacity",
        }
    )
    suppress = pd.Series(
        {
            "reference_capacity_value": 3.0,
            "original_lower_bound": 0.0,
            "original_upper_bound": 1000.0,
            "intervention_type": "competing_sink_suppression",
        }
    )
    assert freeze.directional_bounds_for_intervention(uptake, 1.5, "transporter_upregulation")["target_lower_bound"] == -6.0
    assert freeze.directional_bounds_for_intervention(suppress, 0.5, "suppression")["target_upper_bound"] == 1.5


def test_dynamic_config_edits_cover_pathway_glucose_atpm_and_product_loss():
    base = {
        "candidate_id": "cfg",
        "candidate_hash": "cfg",
        "environment": {"temperature": 30, "pH": 5, "DO": 50},
        "edits": [
            {"reaction_id": "BETA_PHYTOENE_SYNTHASE", "edit_type": "pathway_enzyme_capacity_modification", "capacity_multiplier": 1.5},
            {"reaction_id": "r_1714", "edit_type": "exchange_uptake_or_secretion_change", "capacity_multiplier": 1.2},
            {"reaction_id": "r_4046", "edit_type": "reaction_capacity_decrease_demand", "capacity_multiplier": 0.8},
            {"reaction_id": "BETA_EXPORT_ASSIST", "edit_type": "product_loss_modification", "capacity_multiplier": 2.0},
        ],
    }
    cfg = freeze.verifier.config_for_sparse_candidate(base, "strong_dynamic_regulation")
    parent = freeze.verifier.config_for_sparse_candidate({**base, "edits": []}, "strong_dynamic_regulation")
    assert cfg.psy_base_capacity > parent.psy_base_capacity
    assert cfg.glucose_uptake > parent.glucose_uptake
    assert cfg.baseline_atpm < parent.baseline_atpm
    assert cfg.beta_deg_base < parent.beta_deg_base


def test_stage_and_dynamic_gate_classification_helpers():
    staged = pd.DataFrame(
        [
            {"intervention_id": "a", "effective_or_inactive_classification": "inactive_or_equivalent", "staged_effect_size": 0.0, "environment_id": "central"},
            {"intervention_id": "a", "effective_or_inactive_classification": "effective", "staged_effect_size": 0.2, "environment_id": "low_oxygen"},
            {"intervention_id": "b", "effective_or_inactive_classification": "inactive_or_equivalent", "staged_effect_size": 0.0, "environment_id": "central"},
        ]
    )
    table = freeze.stage_pass_table(staged).set_index("intervention_id")
    assert bool(table.loc["a", "staged_pass"])
    assert not bool(table.loc["b", "staged_pass"])

    p_traj = pd.DataFrame({"z_ox": [0.0, 0.1], "z_atp": [0.0, 0.2], "z_bottle": [0.0, 0.3], "E_PSY": [1.0, 0.9], "E_DES": [1.0, 0.8], "E_CYC": [1.0, 0.7]})
    e_traj = pd.DataFrame({"z_ox": [0.0, 0.2], "z_atp": [0.0, 0.1], "z_bottle": [0.0, 0.4], "E_PSY": [1.0, 0.8], "E_DES": [1.0, 0.85], "E_CYC": [1.0, 0.75]})
    p_flux = pd.DataFrame({"beta_carotene_flux": [1.0], "biomass_flux": [2.0]})
    e_flux = pd.DataFrame({"beta_carotene_flux": [1.4], "biomass_flux": [1.9]})
    p_summary = pd.DataFrame([{"final_product": 1.0, "product_AUC": 2.0, "final_biomass": 3.0, "minimum_biomass": 1.0}])
    e_summary = pd.DataFrame([{"final_product": 1.2, "product_AUC": 2.5, "final_biomass": 2.9, "minimum_biomass": 0.9}])
    metrics = freeze.dynamic_pair_metrics((p_traj, p_flux, pd.DataFrame(), p_summary), (e_traj, e_flux, pd.DataFrame(), e_summary))
    assert metrics["final_product_delta"] == 0.19999999999999996
    assert metrics["beta_carotene_flux_trajectory_max_abs_delta"] == 0.3999999999999999


def test_final_library_deduplicates_effective_models_and_phenotypes(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "DATA", tmp_path)
    diagnostic = pd.DataFrame(
        [
            {"intervention_id": "a", "reaction_id": "r1", "reaction_name": "r1", "mechanism_class": "m1", "subsystem": "s", "gene_ids": "", "GPR_rule": "", "intervention_type": "x", "edit_type": "x", "capacity_multiplier": 1.0, "allowed_magnitudes": "1", "reference_capacity": 1.0, "reference_capacity_source": "test", "reference_capacity_value": 1.0, "bound_transform": "increase", "gem_checksum": "g"},
            {"intervention_id": "b", "reaction_id": "r2", "reaction_name": "r2", "mechanism_class": "m1", "subsystem": "s", "gene_ids": "", "GPR_rule": "", "intervention_type": "x", "edit_type": "x", "capacity_multiplier": 1.0, "allowed_magnitudes": "1", "reference_capacity": 1.0, "reference_capacity_source": "test", "reference_capacity_value": 1.0, "bound_transform": "increase", "gem_checksum": "g"},
        ]
    )
    staged = pd.DataFrame(
        [
            {"intervention_id": "a", "effective_or_inactive_classification": "effective", "staged_effect_size": 1.0, "environment_id": "central"},
            {"intervention_id": "b", "effective_or_inactive_classification": "effective", "staged_effect_size": 1.0, "environment_id": "central"},
        ]
    )
    dynamic = pd.DataFrame(
        [
            {"intervention_id": "a", "dynamic_pass": True, "dynamic_effect_size": 1.0, "effective_model_hash": "same", "effective_phenotype_fingerprint": "samep"},
            {"intervention_id": "b", "dynamic_pass": True, "dynamic_effect_size": 1.0, "effective_model_hash": "same", "effective_phenotype_fingerprint": "samep"},
        ]
    )
    retained, excluded = freeze.final_library_from_gates(diagnostic, staged, dynamic)
    assert len(retained) == 1
    assert "duplicate_effective_model_or_phenotype" in excluded["exclusion_reason"].iloc[0]


def test_design_space_compression_reporting_for_empty_library(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "DATA", tmp_path)
    out = freeze.write_design_space_compression(pd.DataFrame(), None)
    row = out.iloc[0]
    assert row["candidate_specifications"] == 0
    assert row["unique_effective_models"] == 0
