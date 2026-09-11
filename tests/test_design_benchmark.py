from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import design_benchmark as db  # noqa: E402
import stage4_design as stage4  # noqa: E402


def test_candidate_hash_is_deterministic_and_configuration_sensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    pool = db.build_candidate_pool(24, seed=100)
    row = pool.iloc[0].to_dict()
    assert db.candidate_hash(row) == db.candidate_hash(row)
    assert db.candidate_hash(row) != db.candidate_hash(row, oracle_backend="exact_dynamic_pfba")


def test_oracle_cache_reuse_keeps_method_observations_private(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    pool = db.build_candidate_pool(12, seed=101)
    oracle = db.BenchmarkOracle()
    candidate = pool.iloc[0]
    first = oracle.query("method_a", "run_a", candidate, round_id=0)
    second = oracle.query("method_b", "run_b", candidate, round_id=0)
    oracle.save()
    cache = pd.read_csv(tmp_path / "design_benchmark_oracle_cache_index.csv")
    ledger = pd.read_csv(tmp_path / "design_benchmark_observation_ledger.csv")
    assert first["cache_event"] == "computed_new_surrogate_oracle"
    assert second["cache_event"] == "global_cache_reuse"
    assert len(cache) == 1
    assert len(ledger) == 2
    assert set(ledger[ledger["method_id"].eq("method_a")]["run_id"]) == {"run_a"}
    assert set(ledger[ledger["method_id"].eq("method_b")]["run_id"]) == {"run_b"}


def test_protocols_separate_measurements_rounds_batches_and_strains(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    protocols = db.protocols()
    one = protocols[protocols["protocol_id"].eq("one_shot_32")].iloc[0]
    batch = protocols[protocols["protocol_id"].eq("batch_q8_n32")].iloc[0]
    batch16 = protocols[protocols["protocol_id"].eq("batch_q16_n32")].iloc[0]
    seq = protocols[protocols["protocol_id"].eq("sequential_n16")].iloc[0]
    assert (one.N_measurements, one.N_rounds, one.batch_size) == (32, 1, 32)
    assert (batch.N_measurements, batch.N_rounds, batch.batch_size) == (32, 4, 8)
    assert (batch16.N_measurements, batch16.N_rounds, batch16.batch_size) == (32, 2, 16)
    assert (seq.N_measurements, seq.N_rounds, seq.batch_size) == (16, 16, 1)


def test_method_registry_contains_required_tracks_and_optimiser_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    registry = db.method_registry()
    tracks = set(registry["track"])
    assert "A_one_shot_parallel_screen" in tracks
    assert "B_batched_adaptive_optimisation" in tracks
    assert "C_fully_sequential_optimisation" in tracks
    direct = registry[registry["method_id"].eq("direct_surrogate_batch_ucb")].iloc[0]
    hybrid = registry[registry["method_id"].eq("hybrid_digital_twin_batch_ucb")].iloc[0]
    assert direct["outer_optimiser"] == hybrid["outer_optimiser"] == "ucb"
    assert direct["optimiser_parity_group"] == hybrid["optimiser_parity_group"]


def test_candidate_pool_respects_bounds_and_stage4_edit_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    pool = db.build_candidate_pool(48, seed=102)
    assert pool["temperature"].between(27.0, 33.0).all()
    assert pool["pH"].between(4.5, 5.5).all()
    assert pool["DO"].between(20.0, 80.0).all()
    assert set(stage4.EDIT_COLUMNS).issubset(pool.columns)
    assert pool["candidate_id"].is_unique


def test_objective_rewards_product_and_penalises_stress_growth_and_edit_cost():
    good = {
        "final_product": 1.0,
        "product_AUC": 4.0,
        "integrated_product_flux": 0.4,
        "integrated_oxidative_burden": 0.1,
        "integrated_ATP_pressure": 0.1,
        "final_biomass": 0.3,
        "edit_cost": 0.1,
    }
    bad = dict(good, final_product=0.4, product_AUC=1.0, integrated_oxidative_burden=1.0, final_biomass=0.02, edit_cost=1.0)
    assert db.objective_value(good) > db.objective_value(bad)


def test_surrogate_pilot_outputs_are_not_labelled_as_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    monkeypatch.setattr(db, "FIGURES", tmp_path / "figures")
    db.run_all(pool_size=40)
    compute = pd.read_csv(tmp_path / "design_benchmark_compute_accounting.csv")
    cache = pd.read_csv(tmp_path / "design_benchmark_oracle_cache_index.csv")
    regret = pd.read_csv(tmp_path / "design_benchmark_regret.csv")
    acceptance = pd.read_csv(tmp_path / "design_benchmark_acceptance.csv")
    assert int(compute[compute["compute_item"].eq("unique_exact_dynamic_pfba_rollouts")]["value"].iloc[0]) == 0
    assert int(compute[compute["compute_item"].eq("total_lp_solves")]["value"].iloc[0]) == 0
    assert set(cache["metabolic_backend"]) == {"surrogate_oracle_for_scheduler_pilot"}
    assert set(regret["reference_status"]) == {"best_known_regret_incomplete_pool"}
    assert not bool(acceptance[acceptance["status_type"].eq("pilot_status")]["passed"].iloc[0])


def test_representative_stage3_gate_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA", tmp_path)
    completion = pd.DataFrame(
        [
            {
                "culture_id": cid,
                "split": split,
                "replay_type": replay,
                "status": "complete_valid",
                "completed_lp_solves": 10,
                "optimal_solves": 10,
                "infeasible_solves": 0,
                "unbounded_solves": 0,
                "solver_errors": 0,
                "skipped_intervals": 0,
                "surrogate_evaluations": 0,
            }
            for cid, split in enumerate(["train", "validation", "interpolation", "heldout_combination", "extrapolation"])
            for replay in ["oracle", "teacher", "hybrid"]
        ]
    )
    metrics = pd.DataFrame(
        [
            {"comparison": "oracle_replay_error", "split": "all", "raw_RMSE": 0.0, "trajectory_normalized_RMSE": 0.0},
            {"comparison": "total_hybrid_error", "split": "all", "raw_RMSE": 0.0, "trajectory_normalized_RMSE": 0.02},
            {"comparison": "total_hybrid_error", "split": "heldout_combination", "raw_RMSE": 0.0, "trajectory_normalized_RMSE": 0.06},
        ]
    )
    completion.to_csv(tmp_path / "hybrid_exact_replay_completion.csv", index=False)
    metrics.to_csv(tmp_path / "hybrid_exact_replay_preliminary_metrics.csv", index=False)
    gate = db.audit_representative_stage3()
    assert gate["passed"].all()
    assert (tmp_path / "stage3_representative_validation_manifest.csv").exists()
