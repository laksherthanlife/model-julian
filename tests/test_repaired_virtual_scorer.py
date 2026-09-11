"""Step 2 verification: the repaired-model virtual scorer genuinely depends
on the model, environment, edits, and predicted physiology -- not on a
nearest-neighbour pseudodata lookup (LEARNER_ARCHITECTURE_REPAIR.md, virtual
scoring section)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "results" / "repaired_hybrid_training" / "checkpoints"
CKPT_A = CKPT_DIR / "hybrid_learned_physiology__full__seed11.npz"
CKPT_B = CKPT_DIR / "hybrid_learned_physiology__full__seed22.npz"

pytestmark = pytest.mark.skipif(not CKPT_A.exists() or not CKPT_B.exists(), reason="run_repaired_hybrid_training.py checkpoints not present in this environment")


@pytest.fixture(scope="module")
def env_stats():
    dataset = train_mod.load_dataset()
    train_mask = dataset["splits"] == "train"
    return train_mod.standardize_fit(dataset["env_raw"], train_mask)


@pytest.fixture(scope="module")
def scorer_a(env_stats):
    return scorer_mod.RepairedVirtualScorer.load(CKPT_A, *env_stats)


@pytest.fixture(scope="module")
def scorer_b(env_stats):
    return scorer_mod.RepairedVirtualScorer.load(CKPT_B, *env_stats)


BASE_CANDIDATE = {"candidate_id": "t1", "candidate_hash": "h1", "edits": [], "environment": {"temperature": 30.0, "pH": 5.0, "DO": 50.0}}


def test_checkpoint_changes_score(scorer_a, scorer_b):
    a = scorer_a.score(BASE_CANDIDATE)["virtual_score"]
    b = scorer_b.score(BASE_CANDIDATE)["virtual_score"]
    assert a != b, "different checkpoints (different training seeds) must give different scores"


def test_environment_changes_score(scorer_a):
    other = {**BASE_CANDIDATE, "environment": {"temperature": 33.0, "pH": 4.5, "DO": 20.0}}
    a = scorer_a.score(BASE_CANDIDATE)["virtual_score"]
    b = scorer_a.score(other)["virtual_score"]
    assert a != b


def test_edits_change_score_and_are_logged(scorer_a):
    edited = {**BASE_CANDIDATE, "edits": [{"reaction_id": "BETA_PHYTOENE_SYNTHASE", "edit_type": "pathway_enzyme_capacity_modification", "capacity_multiplier": 1.3}]}
    base_result = scorer_a.score(BASE_CANDIDATE)
    edit_result = scorer_a.score(edited)
    assert base_result["virtual_score"] != edit_result["virtual_score"]
    assert edit_result["n_edits_applied"] == 1
    assert "applied:BETA_PHYTOENE_SYNTHASE" in edit_result["edit_notes"]


def test_uncovered_edit_is_skipped_not_silently_dropped(scorer_a):
    uncovered = {**BASE_CANDIDATE, "edits": [{"reaction_id": "r_1714", "edit_type": "exchange_uptake_or_secretion_change", "capacity_multiplier": 1.5}]}
    result = scorer_a.score(uncovered)
    assert result["n_edits_skipped"] == 1
    assert "skipped_uncovered_reaction:r_1714" in result["edit_notes"]
    # score must equal the no-edit baseline since the only edit present is uncovered
    assert result["virtual_score"] == scorer_a.score(BASE_CANDIDATE)["virtual_score"]


def test_predicted_physiology_actually_drives_the_score(scorer_a, env_stats):
    """Perturbing the control head's weights (a stand-in for 'predicted
    physiology changes') must change the score -- confirms the score is
    read from the live forward pass, not a cached/independent table."""
    perturbed_params = scorer_a.params.copy()
    perturbed_params.Wc += 0.5
    perturbed_scorer = scorer_mod.RepairedVirtualScorer(params=perturbed_params, surrogate=scorer_a.surrogate, env_mean=scorer_a.env_mean, env_std=scorer_a.env_std, checkpoint_meta=scorer_a.checkpoint_meta)
    assert perturbed_scorer.score(BASE_CANDIDATE)["virtual_score"] != scorer_a.score(BASE_CANDIDATE)["virtual_score"]


def test_reproducible_given_same_inputs(scorer_a):
    r1 = scorer_a.score(BASE_CANDIDATE)
    r2 = scorer_a.score(BASE_CANDIDATE)
    assert r1["virtual_score"] == r2["virtual_score"]


def test_no_raw_hidden_state_or_prospective_exact_result_is_read():
    """Static check: the scorer module must never reference privileged
    simulator columns, teacher_pseudodata.csv, or the prospective exact
    result ledger -- the exact things the old lookup-based scorer depended on."""
    # Scan only the executable code (functions/classes), not the module
    # docstring, which deliberately *names* what this module replaces.
    code_only = "\n".join(inspect.getsource(obj) for _name, obj in vars(scorer_mod).items() if inspect.isfunction(obj) or inspect.isclass(obj))
    for forbidden in ("z_ox", "z_atp", "z_bottle", "teacher_pseudodata", "prospective_exact_simulator_call_ledger", "teacher_lookup", "nearest"):
        assert forbidden not in code_only, f"scorer code must not reference {forbidden!r}"


def test_score_depends_only_on_checkpoint_surrogate_and_candidate():
    """The scorer's public API takes no hidden global state -- score() is a
    pure function of (loaded checkpoint, frozen surrogate, candidate dict)."""
    sig = inspect.signature(scorer_mod.RepairedVirtualScorer.score)
    assert list(sig.parameters) == ["self", "candidate"]
