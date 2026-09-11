from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import design_benchmark_exact as exact  # noqa: E402


def test_exact_manifest_freezes_two_regimes_and_role_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(exact, "DATA", tmp_path)
    manifest = exact.build_exact_pool_manifest(n_per_regime=48)
    roles = manifest.groupby(["regime_id", "role"]).size().unstack(fill_value=0)
    assert set(manifest["regime_id"]) == {"weak_dynamic_regulation", "strong_dynamic_regulation"}
    assert (manifest.groupby("regime_id")["candidate_id"].count() == 48).all()
    assert (roles["edit_surrogate_calibration"] == 20).all()
    assert (roles["uncertainty_calibration"] == 8).all()
    assert (roles["benchmark_reference_pool"] == 16).all()
    assert (roles["final_untouched_verification"] == 4).all()
    assert set(manifest["exact_status"]) == {"pending_exact_dynamic_pfba"}
    assert set(manifest["expected_lp_solves"]) == {144}
    assert (tmp_path / "design_benchmark_regime_manifest.csv").exists()


def test_exact_candidate_hash_is_regime_and_backend_sensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(exact, "DATA", tmp_path)
    manifest = exact.build_exact_pool_manifest(n_per_regime=8)
    weak = manifest[manifest["regime_id"].eq("weak_dynamic_regulation")].iloc[0].to_dict()
    strong = dict(weak, regime_id="strong_dynamic_regulation")
    assert exact.exact_candidate_hash(weak) == exact.exact_candidate_hash(weak)
    assert exact.exact_candidate_hash(weak) != exact.exact_candidate_hash(strong)
    assert exact.EXACT_BACKEND == "yeast_gem_lp"


def test_exact_protocols_cover_required_budgets_and_batch_sizes():
    protocols = exact.core_protocols()
    assert set(protocols["N_measurements"]) == {8, 16, 24, 32}
    assert set(protocols[protocols["track"].eq("one_shot_parallel")]["batch_size"]) == {8, 16, 24, 32}
    assert set(protocols[protocols["protocol_id"].str.startswith("batch_q4")]["batch_size"]) == {4}
    assert set(protocols[protocols["protocol_id"].str.startswith("batch_q8")]["batch_size"]) == {8}
    assert set(protocols[protocols["track"].eq("fully_sequential")]["batch_size"]) == {1}
    assert protocols[protocols["protocol_id"].eq("sequential_n32")]["N_rounds"].iloc[0] == 32


def test_initial_matched_calibration_is_only_for_direct_and_hybrid():
    cache = pd.DataFrame(
        [
            {
                "candidate_id": f"c{i}",
                "objective_value": float(i),
                "regime_id": "weak_dynamic_regulation",
                "role": "edit_surrogate_calibration" if i < 10 else "benchmark_reference_pool",
                "edit_class": f"class_{i % 3}",
                "static_gem_score": float(10 - i),
            }
            for i in range(14)
        ]
    )
    direct = exact.initial_calibration(cache, "weak_dynamic_regulation", "B_matched_edit_calibration", "direct_trajectory_ucb")
    hybrid = exact.initial_calibration(cache, "weak_dynamic_regulation", "B_matched_edit_calibration", "hybrid_digital_twin_ucb")
    random = exact.initial_calibration(cache, "weak_dynamic_regulation", "B_matched_edit_calibration", "random_parallel")
    assert len(direct) == len(hybrid) == 8
    assert set(direct["candidate_id"]).issubset({f"c{i}" for i in range(10)})
    assert random.empty


def test_core_benchmark_keeps_final_verification_candidates_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(exact, "DATA", tmp_path)
    monkeypatch.setattr(exact, "FIGURES", tmp_path / "figures")
    rows = []
    for regime in ["weak_dynamic_regulation", "strong_dynamic_regulation"]:
        for i in range(44):
            role = "edit_surrogate_calibration" if i < 20 else "uncertainty_calibration" if i < 28 else "benchmark_reference_pool"
            rows.append(
                {
                    "candidate_id": f"{regime}_bench_{i}",
                    "candidate_hash": f"h{i}",
                    "regime_id": regime,
                    "role": role,
                    "strain_id": f"s{i}",
                    "environment_id": f"e{i % 8}",
                    "edit_class": f"class_{i % 5}",
                    "temperature": 30.0,
                    "pH": 5.0,
                    "DO": 50.0,
                    "edit_distance": float(i % 4),
                    "edit_cost": 0.1,
                    "static_gem_score": float(i),
                    "direct_prior_score": float(i),
                    "hybrid_prior_score": float(i),
                    "space_filling_score": float(i),
                    "objective_value": float(i),
                    "final_product": float(i) / 10.0,
                    "product_AUC": float(i),
                    "final_biomass": 1.0,
                    "feasible": True,
                    "severe_growth_collapse": False,
                    "excessive_stress": False,
                    "low_product": False,
                    "metabolic_backend": exact.EXACT_BACKEND,
                    "n_surrogate_evaluations": 0,
                }
            )
        rows.append(
            {
                "candidate_id": f"{regime}_final_best",
                "candidate_hash": "final",
                "regime_id": regime,
                "role": "final_untouched_verification",
                "strain_id": "final_strain",
                "environment_id": "final_env",
                "edit_class": "final_class",
                "temperature": 30.0,
                "pH": 5.0,
                "DO": 50.0,
                "edit_distance": 10.0,
                "edit_cost": 0.1,
                "static_gem_score": 999.0,
                "direct_prior_score": 999.0,
                "hybrid_prior_score": 999.0,
                "space_filling_score": 999.0,
                "objective_value": 999.0,
                "final_product": 99.0,
                "product_AUC": 99.0,
                "final_biomass": 1.0,
                "feasible": True,
                "severe_growth_collapse": False,
                "excessive_stress": False,
                "low_product": False,
                "metabolic_backend": exact.EXACT_BACKEND,
                "n_surrogate_evaluations": 0,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "design_benchmark_oracle_cache_index.csv", index=False)
    pd.DataFrame({"n_actual_lp_solves": [0]}).to_csv(tmp_path / "design_benchmark_oracle_solver_accounting.csv", index=False)
    exact.run_core_benchmark(seeds=[11])
    ledger = pd.read_csv(tmp_path / "design_benchmark_observation_ledger.csv")
    regret = pd.read_csv(tmp_path / "design_benchmark_regret.csv")
    assert "final_untouched_verification" not in set(ledger["role"])
    assert regret["exact_pool_regret"].max() > 0


def test_acceptance_overall_is_not_false_regime_dependent(tmp_path, monkeypatch):
    monkeypatch.setattr(exact, "DATA", tmp_path)
    seed_summary = pd.DataFrame(
        [
            {"regime_id": regime, "method_id": method, "mean_best_objective": score, "mean_measurements": 8, "mean_rounds": 1, "mean_strains": 8}
            for regime in ["weak_dynamic_regulation", "strong_dynamic_regulation"]
            for method, score in [("space_filling_parallel", 2.0), ("hybrid_digital_twin_ucb", 1.5)]
        ]
    )
    exact.write_acceptance(pd.DataFrame(), pd.DataFrame(), seed_summary)
    acceptance = pd.read_csv(tmp_path / "design_benchmark_acceptance.csv")
    overall = acceptance[acceptance["status_type"].eq("overall_scientific_result")].iloc[0]
    assert overall["status"] == "hybrid_no_clear_advantage_consistent_across_regimes"


def test_spearman_without_scipy_matches_reversed_rank_expectation():
    corr = exact.spearman_without_scipy(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([4.0, 3.0, 2.0, 1.0]))
    assert corr == -1.0
