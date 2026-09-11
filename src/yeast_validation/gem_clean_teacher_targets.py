#!/usr/bin/env python3
"""The clean (leak-free) training-target contract for the Latent Capacity
Mismatch Benchmark's teacher retrain (Stage D1).

Background: the production teacher (`run_reporter_grounded_hybrid_distillation.
PRIMARY_TEACHER_VARIANT`) was found, during the Stage A audit, to train
directly on privileged simulator quantities -- `INTERFACE_COLUMNS` there
includes raw `z_ox, z_atp, z_bottle`, and `FLUX_COLUMNS` includes internal LP
fluxes (`ggpp_flux`, `PSY_flux`, `DES_flux`, `CYC_flux`,
`atp_maintenance_flux`) that are not plausible wet-lab observables. That is
exactly the information-boundary violation Section 6 of the handover spec
prohibits.

This module is the single place the clean contract is defined, so both the
retrain script and its leakage test import the same list -- the contract
cannot silently drift between the two.

Deployment inputs: environment only (temperature, pH, DO), unchanged from the
existing production contract. Historical/training supervision: product
trajectory, biomass trajectory, extracellular glucose (a genuine wet-lab
observable now that Stage B1 adds it), and reporter trajectories for whichever
hidden states declare a `reporter_spec` (partial observability -- most hidden
states have none). Raw hidden states, internal GEM constraint bounds, and
internal LP fluxes are never included, by construction and by test
(`tests/test_clean_teacher_no_leakage.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_controller_family as gcf  # noqa: E402

DEPLOYMENT_INPUTS: list[str] = ["temperature", "pH", "DO"]

# Trajectory columns: prefer the noisy *_observed variant (scripts/gem_observation_noise.py)
# when present; the *_true / bare column remains a legitimate fallback for datasets
# generated before the noise layer was applied.
TRAJECTORY_BASE_COLUMNS: list[str] = ["B_total", "X", "S_glc"]
TRAJECTORY_OBSERVED_COLUMNS: list[str] = [f"{c}_observed" for c in TRAJECTORY_BASE_COLUMNS]

# The core reporter layer (generate_gem_state_space_dataset.build_reporters) is
# always present for D_true >= 3, since the core ox/atp/bottle triad (plus the
# always-present decoy z_er and the always-present enzyme capacities) is
# always active. Note there is deliberately no direct z_bottle reporter --
# bottleneck is only observed indirectly via the three capacity reporters,
# matching the existing (pre-audit) partial-observability design.
CORE_REPORTER_COLUMNS: list[str] = ["R_ox", "R_atp", "R_E_PSY", "R_E_DES", "R_E_CYC"]

# Explicitly excluded: never add any of these to a supervision-column list.
# Raw hidden states (every name gem_controller_family can ever produce, core +
# full library), internal GEM constraint bounds, and internal LP fluxes.
EXCLUDED_HIDDEN_STATE_COLUMNS: list[str] = list(gcf.CORE_STATES) + ["z_er"] + [s.name for s in gcf.STATE_LIBRARY]
EXCLUDED_INTERNAL_CONSTRAINT_COLUMNS: list[str] = [
    "glucose_lower_bound",
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "gamma_growth_fraction",
    "pathway_capacity_upper_bound",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
    "s_glc_available",
    "substrate_limited",
] + [f"{name}_probe_bound" for name in gcf.EXTRA_BOUND_PROBES]
EXCLUDED_INTERNAL_FLUX_COLUMNS: list[str] = [
    "beta_carotene_flux",
    "biomass_flux",
    "oxygen_uptake",
    "atp_maintenance_flux",
    "ggpp_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
    "glucose_uptake",
    "E_PSY",
    "E_DES",
    "E_CYC",
]
EXCLUDED_COLUMNS: set[str] = set(EXCLUDED_HIDDEN_STATE_COLUMNS) | set(EXCLUDED_INTERNAL_CONSTRAINT_COLUMNS) | set(EXCLUDED_INTERNAL_FLUX_COLUMNS)


def reporter_column_for_state(state_name: str) -> str:
    return f"R_{state_name.removeprefix('z_')}"


def supervision_columns_for_controller(extra_states: list["gcf.HiddenStateSpec"], use_observed: bool = True) -> list[str]:
    """The clean SUPERVISION_COLUMNS list for a given D_true controller
    (extra_states beyond the core triad, from gem_controller_family.build_controller).
    Reporter coverage is partial by design: only states with a declared
    reporter_spec contribute a reporter column."""
    trajectory_cols = TRAJECTORY_OBSERVED_COLUMNS if use_observed else TRAJECTORY_BASE_COLUMNS
    reporter_cols = list(CORE_REPORTER_COLUMNS)
    for spec in extra_states:
        if spec.reporter_spec is not None:
            reporter_cols.append(reporter_column_for_state(spec.name))
    cols = trajectory_cols + reporter_cols
    leaked = set(cols) & EXCLUDED_COLUMNS
    if leaked:
        raise AssertionError(f"Clean supervision-column contract would leak privileged columns: {sorted(leaked)}")
    return cols
