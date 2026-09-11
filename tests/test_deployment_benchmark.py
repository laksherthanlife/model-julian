from pathlib import Path
import json
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import deployment_benchmark as dep  # noqa: E402
import run_deployment_replication_campaign as repl  # noqa: E402
import scientist_agent_tools as tools  # noqa: E402
import stage4_design as stage4  # noqa: E402


def synthetic_cache(n_per_regime: int = 48) -> pd.DataFrame:
    rows = []
    for regime, scale in [("weak_dynamic_regulation", 2.0), ("strong_dynamic_regulation", 0.8)]:
        for i in range(n_per_regime):
            row = {
                "candidate_id": f"{regime}_{i:03d}",
                "candidate_hash": f"h{i:03d}",
                "regime_id": regime,
                "role": "benchmark_reference_pool",
                "strain_id": f"strain_{i % 40}",
                "environment_id": f"env_{i % 12}",
                "edit_vector_id": "baseline_no_edit" if i == 0 else f"edit_{i}",
                "edit_class": f"class_{i % 6}",
                "edit_description": "synthetic",
                "temperature": 30.0,
                "pH": 5.0,
                "DO": 50.0,
                "selection_reason": "test",
                "edit_distance": float(i % 5) / 2.0,
                "edit_cost": float(i % 4) / 10.0,
                "static_gem_score": float(i) / n_per_regime,
                "direct_prior_score": float(i) / n_per_regime,
                "hybrid_prior_score": float(i) / n_per_regime,
                "space_filling_score": float(n_per_regime - i) / n_per_regime,
                "expected_lp_solves": 144,
                "final_product": scale * (0.1 + i / n_per_regime),
                "product_AUC": scale * (0.5 + i / 10.0),
                "integrated_product_flux": scale * (0.1 + i / 100.0),
                "integrated_oxidative_burden": 0.1,
                "integrated_ATP_pressure": 0.1,
                "integrated_congestion": 0.1,
                "final_biomass": 1.0,
                "minimum_biomass": 0.1,
                "feasible": True,
                "severe_growth_collapse": False,
                "excessive_stress": False,
                "low_product": False,
                "objective_value": scale * (0.2 + i / n_per_regime),
                "exact_status": "complete_exact_dynamic_pfba",
                "metabolic_backend": "yeast_gem_lp",
                "n_surrogate_evaluations": 0,
            }
            for col in stage4.EDIT_COLUMNS:
                row[col] = 1.0
            if i % 3 == 1:
                row["precursor_supply_multiplier"] = 1.4
            if i % 3 == 2:
                row["PSY_capacity_multiplier"] = 1.5
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_universe(data: Path) -> pd.DataFrame:
    required = ["r_0461", "r_0373", "r_0226", "r_0438", dep.gem.GLUCOSE_EXCHANGE]
    subsystems = [
        "central carbon",
        "pentose phosphate",
        "TCA",
        "respiration",
        "ATP",
        "NADPH",
        "mevalonate",
        "IPP DMAPP GGPP",
        "byproduct",
        "transport exchange",
    ]
    rows = []
    for i in range(120):
        rid = required[i] if i < len(required) else f"r_{5000 + i:04d}"
        rows.append(
            {
                "reaction_id": rid,
                "edit_parameter": rid,
                "reaction_name": f"reaction {i}",
                "name": f"reaction {i}",
                "subsystem": subsystems[i % len(subsystems)],
                "equation": "a -> b",
                "stoichiometry_summary": "a -> b",
                "reversible": False,
                "gene_association": "",
                "baseline_flux_distribution": "skipped_public_static_solves_disabled",
                "baseline_flux": 0.0,
                "fva_min": 0.0,
                "fva_max": 1.0,
                "graph_distance_to_product_pathway": i % 4,
                "allowed_edit_types": "reaction_knockdown;reaction_capacity_increase;reaction_knockout",
                "edit_types_allowed": "reaction_knockdown;reaction_capacity_increase;reaction_knockout",
                "allowed_magnitude_range": "0.0..2.5",
                "allowed_minimum": 0.0,
                "allowed_maximum": 2.5,
                "risk_level": ["low", "medium", "high"][i % 3],
                "risk_flag": ["low", "medium", "high"][i % 3],
                "inclusion_rule": "synthetic_test_public_universe",
                "rationale_for_inclusion": "test",
                "mass_balance_status": "balanced",
                "universe_backend": "test",
                "selection_score": 1,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(data / "deployment_benchmark_editable_reaction_universe.csv", index=False)
    pd.DataFrame(
        [
            {"library_reaction_id": dep.gem.PSY_RXN, "name": "PSY", "mass_balance_status": "curated", "max_copies_or_multiplier": 2.0, "allowed_in_tiers": "tier2", "rationale": "test"},
            {"library_reaction_id": dep.gem.DES_RXN, "name": "DES", "mass_balance_status": "curated", "max_copies_or_multiplier": 2.0, "allowed_in_tiers": "tier2", "rationale": "test"},
            {"library_reaction_id": dep.gem.CYC_RXN, "name": "CYC", "mass_balance_status": "curated", "max_copies_or_multiplier": 2.0, "allowed_in_tiers": "tier2", "rationale": "test"},
            {"library_reaction_id": "BETA_EXPORT_ASSIST", "name": "export", "mass_balance_status": "curated", "max_copies_or_multiplier": 2.0, "allowed_in_tiers": "tier2", "rationale": "test"},
        ]
    ).to_csv(data / "deployment_benchmark_heterologous_library.csv", index=False)
    pd.DataFrame([{"analysis": "public_static_fva_universe_subset", "lp_solves": 0, "status": "skipped_public_static_solves_disabled", "reaction_count": len(out)}]).to_csv(data / "deployment_benchmark_static_solve_accounting.csv", index=False)
    return out


def configure_tmp(monkeypatch, tmp_path):
    data = tmp_path / "data"
    figures = tmp_path / "figures"
    results = tmp_path / "results" / "deployment_benchmark"
    data.mkdir()
    synthetic_cache().to_csv(data / "design_benchmark_oracle_cache_index.csv", index=False)
    universe = synthetic_universe(data)
    public_model = data / "public_test_yeast_gem.xml"
    public_model.write_text("<sbml>test</sbml>\n", encoding="utf-8")

    def lightweight_schemas():
        pd.DataFrame([{"reaction_id": "r_0461", "reaction_name": "test", "subsystem": "IPP DMAPP GGPP", "equation": "a -> b"}]).to_csv(data / "deployment_benchmark_public_reaction_annotations.csv", index=False)
        pd.DataFrame([{"metabolite_id": "s_1311", "metabolite_name": "GGPP", "compartment": "c", "formula": "", "charge": "", "reaction_count": 1}]).to_csv(data / "deployment_benchmark_public_metabolite_annotations.csv", index=False)
        (data / "deployment_benchmark_request_schema.json").write_text("{}\n", encoding="utf-8")
        (data / "deployment_benchmark_candidate_schema_example.json").write_text("{}\n", encoding="utf-8")
        pd.DataFrame([{"field": "temperature", "minimum": 27, "maximum": 33}]).to_csv(data / "deployment_benchmark_environment_limits.csv", index=False)
        pd.DataFrame([{"budget_item": "maximum_physical_rounds", "value": 4}]).to_csv(data / "deployment_benchmark_public_operational_budget.csv", index=False)
        (data / "deployment_benchmark_public_tool_documentation.md").write_text("tools\n", encoding="utf-8")

    monkeypatch.setattr(dep, "DATA", data)
    monkeypatch.setattr(dep, "FIGURES", figures)
    monkeypatch.setattr(dep, "RESULTS", results)
    monkeypatch.setattr(dep, "SCIENTIST_RESULTS", results / "scientist_agent")
    monkeypatch.setattr(dep, "HYBRID_RESULTS", results / "hybrid")
    monkeypatch.setattr(dep, "VERIFIER_RESULTS", results / "verifier")
    monkeypatch.setattr(dep, "PUBLIC_SANDBOX", results / "scientist_agent" / "public_sandbox")
    monkeypatch.setattr(dep, "PUBLIC_BUNDLE", results / "public_scientist_bundle")
    monkeypatch.setattr(dep, "EXCHANGE", results / "exchange")
    monkeypatch.setattr(dep, "INCOMING_REQUESTS", results / "exchange" / "incoming_requests")
    monkeypatch.setattr(dep, "PROCESSED_REQUESTS", results / "exchange" / "processed_requests")
    monkeypatch.setattr(dep, "PUBLIC_RESPONSES", results / "exchange" / "public_responses")
    monkeypatch.setattr(dep, "EVALUATOR_LOGS", results / "exchange" / "evaluator_logs")
    monkeypatch.setattr(dep, "PUBLIC_STATIC_SOLVE_ENABLED", False)
    monkeypatch.setattr(dep, "build_editable_reaction_universe", lambda: universe.copy())
    monkeypatch.setattr(dep, "write_public_schemas_and_annotations", lightweight_schemas)
    monkeypatch.setattr(dep.gem, "configured_gem_path", lambda explicit=None: public_model)
    monkeypatch.setattr(tools.deployment, "DATA", data)
    monkeypatch.setattr(tools.deployment, "PUBLIC_STATIC_SOLVE_ENABLED", False)
    monkeypatch.setattr(tools.deployment.gem, "configured_gem_path", lambda explicit=None: public_model)
    return data, figures, results


def test_cached_deployment_reanalysis_writes_zero_new_fba_outputs(tmp_path, monkeypatch):
    data, _figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.run_all()
    summary = pd.read_csv(data / "deployment_benchmark_run_summary.csv")
    acceptance = pd.read_csv(data / "deployment_benchmark_acceptance.csv")
    classes = pd.read_csv(data / "deployment_benchmark_classification.csv")
    assert int(summary[summary["summary_item"].eq("new_exact_fba_rollouts")]["value"].iloc[0]) == 0
    assert dep.BENCHMARK_TYPE_OLD in set(classes["benchmark_type"])
    assert dep.BENCHMARK_TYPE_NEW in set(classes["benchmark_type"])
    assert dep.LLM_STATUS_UNAVAILABLE in set(acceptance["status"])


def test_sparse_edit_validation_accepts_curated_and_rejects_cheats(tmp_path, monkeypatch):
    data, _figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.ensure_dirs()
    dep.build_editable_reaction_universe()
    valid = {
        "strain_id": "valid",
        "design_tier": "tier1",
        "edits": [{"edit_type": "precursor_supply_modification", "reaction_id": "r_0461", "capacity_multiplier": 1.4}],
        "environment": {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
    }
    cheat = {
        "strain_id": "cheat",
        "design_tier": "tier1",
        "edits": [{"edit_type": "reaction_capacity_increase", "reaction_id": dep.gem.PRODUCT_RXN, "capacity_multiplier": 2.0}],
        "environment": {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
    }
    assert dep.validate_sparse_design(valid)["valid"]
    rejected = dep.validate_sparse_design(cheat)
    assert not rejected["valid"]
    assert "direct_objective_or_biomass_cheating_rejected" in rejected["reasons"]


def test_virtual_and_physical_accounting_are_separated(tmp_path, monkeypatch):
    data, _figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.run_all()
    virtuals = pd.read_csv(data / "deployment_benchmark_virtual_evaluations.csv")
    results = pd.read_csv(data / "deployment_benchmark_verification_results.csv")
    assert set(virtuals["event_category"]) == {"virtual_computation"}
    assert set(results["event_category"]) == {"method_requested_biological_verification"}
    assert set(results["verification_backend"]) == {dep.HIDDEN_VERIFIER_BACKEND}
    assert int(results["benchmark_evaluator_lp_solves_charged_to_method"].sum()) == 0


def test_operational_scenarios_and_amortisation_are_explicit(tmp_path, monkeypatch):
    data, _figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.run_all()
    scenarios = pd.read_csv(data / "deployment_benchmark_operational_scenarios.csv")
    amort = pd.read_csv(data / "deployment_benchmark_greenfield_deployed_amortisation.csv")
    assert set(scenarios["scenario_id"]) == {"optimistic_lab", "central_lab", "conservative_lab"}
    assert {1, 3, 5, 10}.issubset(set(amort["future_campaigns"]))
    assert "resource_points_per_campaign" in amort.columns


def test_public_scientist_tools_do_not_expose_hidden_verifier(tmp_path, monkeypatch):
    data, _figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.run_all()
    summary = tools.inspect_model_summary()
    static = tools.run_static_pfba({"temperature": 30.0, "pH": 5.0, "DO": 50.0}, [])
    request = tools.request_verification([], "basic_product_biomass")
    assert summary["hidden_regulator_available"] is False
    assert static["hidden_verifier_called"] is False
    assert request["hidden_verifier_called"] is False
    assert request["status"] == "not_executed_by_public_tool"


def test_required_deployment_figures_are_written(tmp_path, monkeypatch):
    _data, figures, _results = configure_tmp(monkeypatch, tmp_path)
    dep.run_all()
    required = [
        "deployment_best_verified_vs_calendar_time",
        "deployment_virtual_to_physical_leverage",
        "deployment_quality_cost_pareto",
        "deployment_top_strain_designs",
    ]
    for name in required:
        assert (figures / f"{name}.svg").exists()
        assert (figures / f"{name}.png").exists()


def test_open_ended_hybrid_round_001_outputs_are_consistent_when_present():
    request_path = ROOT / "results" / "deployment_benchmark" / "exchange" / "incoming_requests" / "hybrid_round_001_request.json"
    generated_path = ROOT / "data" / "deployment_benchmark_hybrid_round_001_candidate_generation.csv"
    funnel_path = ROOT / "data" / "deployment_benchmark_hybrid_round_001_screening_funnel.csv"
    access_path = ROOT / "data" / "deployment_benchmark_hybrid_round_001_access_audit.csv"
    if not all(p.exists() for p in [request_path, generated_path, funnel_path, access_path]):
        pytest.skip("open-ended hybrid Round 001 artifacts have not been generated in this checkout")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    generated = pd.read_csv(generated_path)
    funnel = pd.read_csv(funnel_path)
    access = pd.read_csv(access_path)

    assert request["campaign_id"] == "open_ended_matched_round_001_scientist_vs_hybrid"
    assert request["round_id"] == "hybrid_round_001"
    assert len(request["candidates"]) == 8
    assert len(generated) >= 50_000
    assert generated["candidate_hash"].nunique() == len(generated)
    assert 1.0 - generated["old_exact_pool_overlap"].astype(bool).mean() >= 0.90
    assert int(funnel[funnel["layer"].eq("dynamic_refinement")]["candidates"].iloc[0]) >= 100
    assert int(funnel[funnel["layer"].eq("selected_batch")]["candidates"].iloc[0]) == 8
    assert access["passed"].astype(bool).all()


def test_report_checkpoint_bundles_are_public_only_when_present():
    frozen = ROOT / "data" / "deployment_benchmark_report_checkpoint_frozen_config.csv"
    registry_path = ROOT / "data" / "deployment_benchmark_campaign_registry.csv"
    if not frozen.exists() or not registry_path.exists():
        pytest.skip("report checkpoint bundles have not been prepared in this checkout")

    registry = pd.read_csv(registry_path)
    frozen_config = pd.read_csv(frozen)
    assert {"campaign_001", "campaign_002", "campaign_003"}.issubset(set(registry["campaign_id"]))
    assert registry[registry["campaign_id"].eq("campaign_001")]["status"].iloc[0] == "complete_frozen"
    assert "non_inferiority_margin" in set(frozen_config["config_item"])
    assert "yeast9_checksum" in set(frozen_config["config_item"])

    for campaign_id in ["campaign_002", "campaign_003"]:
        manifest = pd.read_csv(ROOT / "data" / f"deployment_benchmark_{campaign_id}_public_manifest.csv")
        leakage = pd.read_csv(ROOT / "data" / f"deployment_benchmark_{campaign_id}_leakage_audit.csv")
        handoff = ROOT / "results" / "deployment_benchmark" / campaign_id / "public_scientist_bundle" / "SCIENTIST_AGENT_HANDOFF.md"
        assert manifest["approved_public_asset"].astype(bool).all()
        assert leakage[leakage["audit_item"].eq("prior_or_hidden_leak_terms_found")]["passed"].astype(bool).iloc[0]
        text = handoff.read_text(encoding="utf-8")
        assert "at most 8 cultures" in text
        assert "at most 6 unique strains" in text
        assert campaign_id in text


def test_replication_scientist_registration_rejects_legacy_shared_path(tmp_path, monkeypatch):
    data = tmp_path / "data"
    results = tmp_path / "results" / "deployment_benchmark"
    legacy_dir = results / "exchange" / "incoming_requests"
    campaign_dir = results / "campaign_002" / "exchange" / "incoming_requests"
    bundle_dir = results / "campaign_002" / "public_scientist_bundle"
    data.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    campaign_dir.mkdir(parents=True)
    bundle_dir.mkdir(parents=True)

    request = {
        "round_id": "round_001",
        "campaign_id": "campaign_002",
        "candidates": [
            {
                "candidate_id": "c1",
                "strain_id": "s1",
                "edit_tier": 1,
                "edits": [],
                "environment": {"temperature": 30.0, "pH": 5.0, "DO": 50.0},
                "requested_assay_panel": "basic",
                "hypothesis": "test",
                "expected_failure_mode": "test",
            }
        ],
    }
    legacy_request = legacy_dir / "round_001_request.json"
    legacy_request.write_text(json.dumps(request), encoding="utf-8")
    (data / "deployment_benchmark_campaign_002_public_manifest.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (bundle_dir / "SCIENTIST_AGENT_HANDOFF.md").write_text("handoff\n", encoding="utf-8")

    monkeypatch.setattr(repl, "ROOT", tmp_path)
    monkeypatch.setattr(repl, "DATA", data)
    monkeypatch.setattr(repl, "RESULTS", results)

    with pytest.raises(ValueError, match="campaign_specific_request_path"):
        repl.validate_scientist_registration("campaign_002", legacy_request)

    audit = pd.read_csv(data / "deployment_benchmark_campaign_002_scientist_registration_audit.csv")
    failed = set(audit.loc[~audit["passed"].astype(bool), "check"])
    assert "campaign_specific_request_path" in failed
    assert "legacy_shared_path_rejected" in failed
