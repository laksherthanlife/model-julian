"""Acceptance-gate tests (G1-G10) for the repaired learner architecture.

Some gates are checked against fresh, cheap synthetic-data computations
(G1, G2, G4-mechanics); others are checked against the already-generated
CSVs/checkpoints from `run_repaired_hybrid_training.py` and
`run_hybrid_gsm_replay_validation.py` (G4-empirical, G5, G6, G7, G9, G10),
which this file does not re-run (they involve real Yeast9 LP solves / a
multi-seed training sweep) -- if those artifacts are missing, the relevant
tests are skipped rather than silently passing.

G3 (no privileged leakage) reuses `tests/test_clean_teacher_no_leakage.py`'s
contract directly. G8 (live virtual score) is not re-tested here: it is a
documented, currently-unmet gate (Section 19 of the design doc) -- this pass
audits and documents it but does not implement a fix, so there is nothing to
assert as passing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_trainable_state_space as tss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "results" / "repaired_hybrid_training" / "checkpoints"


def _toy_setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    B, T, env_dim, D, H = 6, 8, 3, 4, 6
    env = rng.normal(size=(B, env_dim))
    targets = {"R_ox": rng.normal(size=(T, B))}
    masks = {"R_ox": np.ones((T, B))}
    return env, T, targets, masks, env_dim, D, H


# --- G1: trainable dynamics -- changing training data changes learned transition parameters ---

def test_g1_different_training_targets_yield_different_transition_params():
    env, T, targets, masks, env_dim, D, H = _toy_setup()
    params_a = tss.init_params(env_dim, D, H, ["R_ox"], n_controls=0, seed=1)
    params_b = params_a.copy()

    adam_a = tss.AdamState(params_a, lr=0.05)
    for _ in range(50):
        _, grads, _ = tss.compute_loss_and_grads(params_a, env, T, targets, masks, lambda_dyn=1e-3)
        adam_a.step(params_a, grads)

    rng2 = np.random.default_rng(99)
    targets_b = {"R_ox": rng2.normal(size=targets["R_ox"].shape)}
    adam_b = tss.AdamState(params_b, lr=0.05)
    for _ in range(50):
        _, grads, _ = tss.compute_loss_and_grads(params_b, env, T, targets_b, masks, lambda_dyn=1e-3)
        adam_b.step(params_b, grads)

    assert not np.allclose(params_a.Wz, params_b.Wz), "transition weights must differ when training targets differ"
    assert tss.params_fingerprint(params_a) != tss.params_fingerprint(params_b)


# --- G2: reporter gradient -- reporter-only loss produces nonzero, mathematically-correct gradient into shared dynamics ---

def test_g2_reporter_only_loss_gradient_check_passes():
    env, T, targets, masks, env_dim, D, H = _toy_setup()
    params = tss.init_params(env_dim, D, H, ["R_ox"], n_controls=0, seed=2)

    def loss_fn(p):
        return tss.compute_loss_and_grads(p, env, T, targets, masks, lambda_dyn=1e-3)

    result = tss.gradient_check(params, env, T, loss_fn, n_params_to_check=14)
    assert result["passed"], result
    _, grads, _ = loss_fn(params)
    assert np.any(np.abs(grads.Wz) > 1e-8), "reporter-only loss must produce a nonzero gradient in the shared transition weights (Wz)"
    assert np.any(np.abs(grads.Wo) > 1e-8)


# --- G4 (mechanics): later outputs genuinely depend on earlier latent evolution ---

def test_g4_state_evolves_over_time_not_constant():
    env, T, _targets, _masks, env_dim, D, H = _toy_setup()
    params = tss.init_params(env_dim, D, H, ["R_ox"], n_controls=0, seed=3)
    cache = tss.forward(params, env, T)
    per_step_change = np.mean(np.abs(np.diff(cache.z, axis=0)))
    assert per_step_change > 1e-4, "latent state must actually evolve across time steps, not stay constant"


def test_g4_empirical_dynamics_beat_direct_env_time_regression():
    path = DATA / "repaired_hybrid_training_metrics.csv"
    if not path.exists():
        pytest.skip("run_repaired_hybrid_training.py has not been run in this environment")
    m = pd.read_csv(path)
    for split in ("validation", "heldout_combination", "extrapolation"):
        db = m[(m["variant"] == "direct_blackbox") & (m["channel"] == "product") & (m["split"] == split)]["normalized_rmse"].mean()
        pos = m[(m["variant"] == "product_only_state_space") & (m["channel"] == "product") & (m["split"] == split)]["normalized_rmse"].mean()
        assert pos < db, f"trainable-dynamics model should beat the no-dynamics direct regression on {split} (got {pos} vs {db})"


# --- G5: predictive signal -- clean model beats constant/train-mean baseline on held-out cultures ---

def test_g5_beats_training_mean_baseline_on_heldout():
    path = DATA / "repaired_hybrid_training_metrics.csv"
    if not path.exists():
        pytest.skip("run_repaired_hybrid_training.py has not been run in this environment")
    m = pd.read_csv(path)
    for variant in ("product_only_state_space", "hybrid_learned_physiology"):
        for channel in ("product", "biomass"):
            vals = m[(m["variant"] == variant) & (m["channel"] == channel) & (m["split"] == "validation")]["normalized_rmse"]
            assert (vals < 1.0).all(), f"{variant}/{channel} must beat the normalized_rmse=1.0 constant-baseline on validation"


# --- G6: OOD evaluation is explicitly measured ---

def test_g6_ood_splits_are_present_in_metrics():
    path = DATA / "repaired_hybrid_training_metrics.csv"
    if not path.exists():
        pytest.skip("run_repaired_hybrid_training.py has not been run in this environment")
    m = pd.read_csv(path)
    for split in ("heldout_combination", "extrapolation"):
        assert split in set(m["split"]), f"OOD split {split!r} must appear in the metrics table"
        assert len(m[m["split"] == split]) > 0


# --- G7: real hybrid coupling, checked live against Yeast9 in run_hybrid_gsm_replay_validation.py ---

def test_g7_real_gsm_coupling_gate():
    path = DATA / "hybrid_gsm_replay_gate.csv"
    if not path.exists():
        pytest.skip("run_hybrid_gsm_replay_validation.py has not been run in this environment")
    gate = pd.read_csv(path)
    assert gate["passed"].all(), gate.to_string()


# --- G9: full vs shuffled reporters mechanically differ (regardless of performance direction) ---

def test_g9_full_vs_shuffled_reporters_yield_different_representations():
    path = DATA / "repaired_hybrid_training_manifest.csv"
    metrics_path = DATA / "repaired_hybrid_training_metrics.csv"
    if not path.exists() or not metrics_path.exists():
        pytest.skip("run_repaired_hybrid_training.py has not been run in this environment")
    manifest = pd.read_csv(path)
    hyb = manifest[manifest["variant"] == "hybrid_learned_physiology"]
    full_fp = set(hyb[hyb["reporter_mode"] == "full"]["params_fingerprint"])
    shuffled_fp = set(hyb[hyb["reporter_mode"] == "shuffled"]["params_fingerprint"])
    assert full_fp.isdisjoint(shuffled_fp), "full and shuffled reporter training must produce different learned parameters"

    m = pd.read_csv(metrics_path)
    full_rox = m[(m["variant"] == "hybrid_learned_physiology") & (m["reporter_mode"] == "full") & (m["channel"] == "R_ox") & (m["split"] == "validation")]["normalized_rmse"].mean()
    shuffled_rox = m[(m["variant"] == "hybrid_learned_physiology") & (m["reporter_mode"] == "shuffled") & (m["channel"] == "R_ox") & (m["split"] == "validation")]["normalized_rmse"].mean()
    assert abs(full_rox - shuffled_rox) > 0.01, "shuffling reporters must measurably change reporter-reconstruction quality (mechanical effect, independent of whether it helps product prediction)"


# --- G10: reproducibility -- seeds, checkpoints, configs saved ---

def test_g10_checkpoints_and_manifest_are_reproducible_and_complete():
    path = DATA / "repaired_hybrid_training_manifest.csv"
    if not path.exists():
        pytest.skip("run_repaired_hybrid_training.py has not been run in this environment")
    manifest = pd.read_csv(path)
    state_space_rows = manifest[manifest["variant"] != "direct_blackbox"]
    assert not state_space_rows.empty
    for _, row in state_space_rows.iterrows():
        ckpt_path = ROOT / row["checkpoint"]
        assert ckpt_path.exists(), f"missing checkpoint: {ckpt_path}"
        _params, meta = tss.load_checkpoint(ckpt_path)
        assert meta["params_fingerprint"] == row["params_fingerprint"]
        assert "seed" in meta and "variant" in meta and "reporter_mode" in meta and "hidden_dim" in meta and "d_model" in meta
        assert meta["d_model"] == 16, "D_model must be frozen at 16 per the handover spec"
