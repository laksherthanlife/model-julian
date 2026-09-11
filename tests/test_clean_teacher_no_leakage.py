"""Leakage guard for the clean teacher training-target contract (Stage D1).

Asserts the clean SUPERVISION_COLUMNS contract never includes a raw hidden
state, an internal GEM constraint bound, or an internal LP flux -- the exact
information-boundary violation found in the production teacher during the
Stage A audit (run_reporter_grounded_hybrid_distillation.INTERFACE_COLUMNS /
FLUX_COLUMNS).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))

import gem_clean_teacher_targets as targets  # noqa: E402
import gem_controller_family as gcf  # noqa: E402


def test_deployment_inputs_are_environment_only():
    assert targets.DEPLOYMENT_INPUTS == ["temperature", "pH", "DO"]


def test_supervision_columns_disjoint_from_hidden_states():
    for d_true in (3, 8, 12, 16, 20, gcf.MAX_D_TRUE):
        extra = gcf.build_controller(d_true)
        cols = targets.supervision_columns_for_controller(extra)
        leaked_states = set(cols) & set(targets.EXCLUDED_HIDDEN_STATE_COLUMNS)
        assert not leaked_states, f"D_true={d_true} leaks hidden states: {leaked_states}"


def test_supervision_columns_disjoint_from_internal_constraints_and_fluxes():
    for d_true in (3, 8, 12, 16, 20, gcf.MAX_D_TRUE):
        extra = gcf.build_controller(d_true)
        cols = targets.supervision_columns_for_controller(extra)
        leaked = set(cols) & (set(targets.EXCLUDED_INTERNAL_CONSTRAINT_COLUMNS) | set(targets.EXCLUDED_INTERNAL_FLUX_COLUMNS))
        assert not leaked, f"D_true={d_true} leaks internal constraints/fluxes: {leaked}"


def test_production_teacher_leak_is_reproduced_as_a_documented_regression():
    """Not a leakage guard on new code -- a pinned regression check that the
    known leak in the existing production teacher module is still present
    (so this test starts failing, as an alert, the day someone fixes it
    in-place rather than via the clean retrain path)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "yeast_validation"))
    import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402

    spec = prod.TEACHER_VARIANTS[prod.PRIMARY_TEACHER_VARIANT]
    assert spec["interface"] is True
    assert spec["flux"] is True
    assert {"z_ox", "z_atp", "z_bottle"}.issubset(set(prod.INTERFACE_COLUMNS))
    assert set(prod.FLUX_COLUMNS) & {"ggpp_flux", "PSY_flux", "DES_flux", "CYC_flux", "atp_maintenance_flux"}


def test_reporter_coverage_is_partial_not_total():
    """Section 12 requires partial observability: not every hidden state gets
    a reporter. Confirm the state library actually has both reporter and
    non-reporter states, and that a matched-capacity D_true still leaves some
    states unobserved."""
    with_reporter = [s for s in gcf.STATE_LIBRARY if s.reporter_spec is not None]
    without_reporter = [s for s in gcf.STATE_LIBRARY if s.reporter_spec is None]
    assert with_reporter, "expected at least one state with a reporter"
    assert without_reporter, "expected at least one state with no reporter (partial observability)"

    extra = gcf.build_controller(16)
    observed = sum(1 for s in extra if s.reporter_spec is not None)
    assert 0 < observed < len(extra), "D_true=16 should have both observed and unobserved extra states"
