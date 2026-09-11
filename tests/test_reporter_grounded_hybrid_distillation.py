import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem
import run_reporter_grounded_hybrid_distillation as hybrid


def test_artifact_inventory_uses_existing_constraints_checkpoint():
    artifacts = hybrid.required_artifacts()
    assert artifacts["constraints"] is not None
    assert artifacts["constraints"].name in {"gem_state_space_constraints.csv", "gem_state_space_generator_constraints.partial.csv"}
    inventory = hybrid.validate_artifacts()
    assert inventory["exists"].all()


def test_no_culture_leakage_and_real_gem_zero_surrogate():
    dataset = hybrid.load_distillation_dataset()
    manifest = dataset["manifest"]
    traj = dataset["traj"]
    assert manifest.groupby("environment_id")["split"].nunique().max() == 1
    assert traj.groupby("culture_id")["split"].nunique().max() == 1
    assert traj["metabolic_backend"].eq("yeast_gem_lp").all()
    assert int(traj["n_surrogate_evaluations"].max()) == 0


def test_train_only_normalisation_ignores_validation_extreme():
    env = np.array([[27.0, 4.5, 20.0], [30.0, 5.0, 50.0], [33.0, 5.5, 80.0]])
    targets = {"B_total": np.array([[0, 1], [0, 2], [100, 200]], dtype=float)}
    train = np.array([True, True, False])
    stats = hybrid.train_only_stats(env, targets, train)
    assert np.allclose(stats["env_mean"], env[:2].mean(axis=0))
    assert stats["target_max"]["B_total"] == 2.0


def test_deterministic_teacher_rollout_and_bounded_outputs():
    dataset = hybrid.load_distillation_dataset()
    ckpt_a, _ = hybrid.train_teacher_checkpoint(dataset, hybrid.PRIMARY_TEACHER_VARIANT, latent_dim=4, seed=11)
    ckpt_b, _ = hybrid.train_teacher_checkpoint(dataset, hybrid.PRIMARY_TEACHER_VARIANT, latent_dim=4, seed=11)
    pred_a = hybrid.predict_columns(ckpt_a, dataset["env"][:3], ["B_total", "R_ox", "gamma_growth_fraction"])
    pred_b = hybrid.predict_columns(ckpt_b, dataset["env"][:3], ["B_total", "R_ox", "gamma_growth_fraction"])
    assert np.allclose(pred_a["B_total"], pred_b["B_total"])
    assert np.isfinite(pred_a["R_ox"]).all()
    assert (pred_a["R_ox"] >= 0).all()
    assert ((pred_a["gamma_growth_fraction"] >= 0) & (pred_a["gamma_growth_fraction"] <= 1)).all()


def test_shuffled_reporter_changes_training_labels_only():
    dataset = hybrid.load_distillation_dataset()
    targets = hybrid.assemble_targets(dataset)
    train = dataset["splits"] == "train"
    shuffled = hybrid.shuffled_training_targets(targets, ["R_ox"], train, seed=123)
    assert not np.allclose(shuffled["R_ox"][train], targets["R_ox"][train])
    assert np.allclose(shuffled["R_ox"][~train], targets["R_ox"][~train])


def test_pseudo_sample_split_integrity_and_uncertainty_weights():
    dataset = hybrid.load_distillation_dataset()
    ckpt, _ = hybrid.train_teacher_checkpoint(dataset, hybrid.PRIMARY_TEACHER_VARIANT, latent_dim=4, seed=11)
    pseudo = hybrid.generate_teacher_pseudodata([ckpt], dataset, n_samples=12, seed=99, write_outputs=False)
    env_df = pseudo["environment"]
    train_ids = set(env_df[env_df["pseudo_split"].eq("pseudo_train")]["pseudo_culture_id"])
    val_ids = set(env_df[env_df["pseudo_split"].eq("pseudo_validation")]["pseudo_culture_id"])
    assert train_ids.isdisjoint(val_ids)
    assert env_df["pseudo_label_weight"].between(0.1, 10.0).all()
    assert pseudo["audit"]["passed"].all()


class _FakeReaction:
    def __init__(self):
        self.lower_bound = -1000.0
        self.upper_bound = 1000.0


class _FakeReactions:
    def __init__(self):
        self._rxns = {rid: _FakeReaction() for rid in [gem.GLUCOSE_EXCHANGE, gem.OXYGEN_EXCHANGE, gem.ATPM_RXN, gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN, gem.PRODUCT_RXN]}

    def get_by_id(self, rid):
        return self._rxns[rid]


class _FakeModel:
    def __init__(self):
        self.reactions = _FakeReactions()


def test_exact_interface_to_gem_mapping():
    model = _FakeModel()
    cfg = gem.GEMCultureConfig()
    controls = {
        "oxygen_lower_bound": -3.2,
        "atp_maintenance_lower_bound": 0.91,
        "gamma_growth_fraction": 0.63,
        "PSY_effective_upper_bound": 0.20,
        "DES_effective_upper_bound": 0.18,
        "CYC_effective_upper_bound": 0.12,
    }
    mapped = hybrid.apply_predicted_interface_controls(model, cfg, controls)
    assert model.reactions.get_by_id(gem.OXYGEN_EXCHANGE).lower_bound == -3.2
    assert model.reactions.get_by_id(gem.ATPM_RXN).lower_bound == 0.91
    assert model.reactions.get_by_id(gem.CYC_RXN).upper_bound == 0.12
    assert mapped["gamma_growth_fraction"] == 0.63


def _minimal_interface_checkpoint(dataset):
    target_mean = {col: float(dataset["interface"][col].mean()) for col in hybrid.INTERFACE_COLUMNS}
    target_std = {col: 1.0 for col in hybrid.INTERFACE_COLUMNS}
    target_min = {col: float(dataset["interface"][col].min()) for col in hybrid.INTERFACE_COLUMNS}
    target_max = {col: float(dataset["interface"][col].max()) for col in hybrid.INTERFACE_COLUMNS}
    return hybrid.StateSpaceCheckpoint(
        variant="hybrid_reporter_supervised",
        seed=1,
        latent_dim=1,
        t_len=dataset["product"].shape[1],
        interval_len=dataset["interface"][hybrid.INTERFACE_COLUMNS[0]].shape[1],
        dt=float(np.median(np.diff(dataset["time"]))),
        env_mean=dataset["env"].mean(axis=0),
        env_std=dataset["env"].std(axis=0) + hybrid.EPS,
        Wg=np.zeros((3, 1)),
        bg=np.zeros(1),
        A=np.zeros((1, 1)),
        C=np.zeros((3, 1)),
        bf=np.zeros(1),
        heads={},
        target_mean=target_mean,
        target_std=target_std,
        target_min=target_min,
        target_max=target_max,
        supervised_columns=hybrid.INTERFACE_COLUMNS,
    )


def test_pfba_rollout_uses_solved_product_flux_and_complete_accounting():
    dataset = hybrid.load_distillation_dataset()
    ckpt = _minimal_interface_checkpoint(dataset)

    def fake_solver(_controls, _cfg, k):
        return {
            "beta_carotene_flux": 0.1 + 0.001 * k,
            "biomass_flux": 0.2,
            "n_growth_optimizations": 1,
            "n_product_optimizations": 1,
            "n_pfba_optimizations": 1,
            "n_actual_lp_solves": 3,
            "n_optimal_solves": 3,
            "n_infeasible_solves": 0,
            "n_unbounded_solves": 0,
            "n_surrogate_evaluations": 0,
        }

    rollout = hybrid.run_exact_pfba_rollout(ckpt, dataset, max_cultures=1, solver_fn=fake_solver, write_outputs=False)
    traj = rollout["trajectories"]
    solver = rollout["solver"].iloc[0]
    assert traj["direct_product_head_used"].eq(False).all()
    assert traj["B_pred_pfba"].iloc[-1] > traj["B_pred_pfba"].iloc[1]
    assert solver["n_actual_lp_solves"] == solver["n_growth_optimizations"] + solver["n_product_optimizations"] + solver["n_pfba_optimizations"]
    assert solver["n_surrogate_evaluations"] == 0


def test_exact_replay_shards_are_contiguous():
    indices = list(range(125))
    assert hybrid.shard_indices(indices, 0, 5) == list(range(0, 25))
    assert hybrid.shard_indices(indices, 1, 5) == list(range(25, 50))
    assert hybrid.shard_indices(indices, 4, 5) == list(range(100, 125))


def test_primary_hybrid_student_has_no_direct_product_head():
    dataset = hybrid.load_distillation_dataset()
    ckpt, history = hybrid.train_student_checkpoint(dataset, pseudo=None, variant="hybrid_reporter_supervised", latent_dim=4, seed=11)
    assert "B_total" not in ckpt.supervised_columns
    assert history["direct_product_head"] is False


def test_multiple_active_dynamic_constraints_and_intervention_isolation():
    dataset = hybrid.load_distillation_dataset()
    interface = dataset["interface"]
    dynamic_counts = {col: int(np.unique(interface[col].round(8)).size) for col in ["oxygen_lower_bound", "atp_maintenance_lower_bound", "PSY_effective_upper_bound", "DES_effective_upper_bound", "CYC_effective_upper_bound"]}
    assert sum(v > 1 for v in dynamic_counts.values()) >= 3
    base = {col: float(interface[col][0, 0]) for col in hybrid.INTERFACE_COLUMNS}
    effects = hybrid.intervention_control_effects(base)
    assert effects["intended_control_changed"].all()
    assert effects[effects["intervention"].eq("unrelated_ER_control")]["n_changed_controls"].iloc[0] == 1


def test_stage4_remains_disabled():
    with pytest.raises(hybrid.Stage4BlockedError):
        hybrid.stage4_strain_engineering_placeholder()
