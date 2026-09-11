#!/usr/bin/env python3
"""The repaired-model virtual scorer for prospective DBTL candidates.

Replaces `run_prospective_dbtl_benchmark.py::hybrid_virtual_score` (a frozen
linear intervention-effect table + nearest-neighbour lookup into
`teacher_pseudodata.csv`, confirmed in `LEARNER_ARCHITECTURE_REPAIR.md`
Section H to run no model at all) with a scorer that actually executes the
repaired learner:

    candidate edits + (T, pH, DO)
        -> repaired trainable z(t)  (gem_trainable_state_space, D_model=16)
        -> predicted control vector c(t)
        -> candidate edits applied to c(t) using the SAME declared bound-transform
           semantics as the exact simulator's deployment_benchmark.apply_sparse_edits_to_model
        -> frozen GSM surrogate (gem_gsm_surrogate)
        -> one-step mechanistic integration (same ODE the real generator uses)
        -> final_product (the same objective run_prospective_dbtl_benchmark.observed_score uses)

`run_prospective_dbtl_benchmark.py` (the frozen headline file) is not
modified -- its `hybrid_virtual_score` remains as historical evidence of the
original benchmark's contract. This module is a standalone, explicitly
separate scorer; `run_repaired_prospective_dry_run.py` is what actually uses
it for the new dry run.

Strain-edit coverage (audited against `data/final_dbtl_actionable_intervention_library.csv`):
16 of 24 frozen interventions target a reaction already present in the
control vector (`BETA_PHYTOENE_SYNTHASE`, `BETA_PHYTOENE_DESATURASE`,
`BETA_LYCOPENE_CYCLASE`, `r_1992`/oxygen exchange) and are applied here with
the same bound-transform rules as the exact simulator. The remaining 8 (GGPP
precursor `r_0461`, glucose exchange `r_1714`, one NADH:ubiquinone
knockdown `r_0773`) have no corresponding control dimension in the current
6-column control vector and are **not yet wired** -- they are skipped with an
explicit, logged note (mirroring `apply_sparse_edits_to_model`'s own
`strict=False` skip-and-note behaviour), not silently dropped. This is
disclosed, not hidden, and is the honest "simplest defensible version":
extending the control vector/surrogate to cover them is future work, not
attempted in this pass.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

T_LEN = 49  # fixed by the training dataset (generate_gem_state_space_dataset.N_TIME); baked into dt_latent=1/(T_LEN-1)
DT_REAL = 0.25
BETA_DEG_BASE = 0.002
X0 = 0.08
B0 = 0.0
CONTROL_CLIP_STDS = 4.0  # disclosed surrogate-safety clip; see score() docstring comment

REACTION_TO_CONTROL = {
    gem.PSY_RXN: "PSY_effective_upper_bound",
    gem.DES_RXN: "DES_effective_upper_bound",
    gem.CYC_RXN: "CYC_effective_upper_bound",
    gem.OXYGEN_EXCHANGE: "oxygen_lower_bound",
}
CONTROL_IS_LOWER_BOUND = {"oxygen_lower_bound", "atp_maintenance_lower_bound"}

# edit_type -> whether it is a "decrease-family" transform (knockdown-like) or
# an "increase-family" transform, mirroring deployment_benchmark.apply_sparse_edits_to_model's
# own dispatch table exactly (reaction_knockout is not supported here: it targets
# named pathway/exchange reactions, none of which are ever knocked out in the frozen library).
DECREASE_EDIT_TYPES = {"reaction_knockdown", "partial_knockdown", "byproduct_suppression", "byproduct_pathway_suppression", "loss_pathway_reduction", "reaction_capacity_decrease", "capacity_decrease", "transport_capacity_decrease"}
INCREASE_EDIT_TYPES = {
    "reaction_capacity_increase", "capacity_increase", "precursor_supply_edit", "precursor_supply_modification", "precursor_supply_enhancement",
    "cofactor_regeneration_edit", "atp_supply_edit", "ATP_supply_edit", "pathway_enzyme_capacity_edit", "pathway_enzyme_capacity_modification",
    "product_export_edit", "product_export_modification", "transport_capacity_change", "secretion_capacity_change", "byproduct_enhancement",
    "byproduct_pathway_enhancement", "exchange_uptake_change", "exchange_uptake_or_secretion_change", "curated_heterologous_addition",
}


def _scaled_bound(old_value: float, multiplier: float, edit: dict[str, Any], *, is_lower_bound: bool) -> float:
    """Mirrors deployment_benchmark.apply_sparse_edits_to_model's scaled_positive_upper /
    scaled_negative_lower helpers, applied to the physiology model's predicted
    control value in place of a live reaction bound."""
    if is_lower_bound:
        if "target_lower_bound" in edit:
            return float(edit["target_lower_bound"])
        return min(0.0, old_value * max(0.0, multiplier))
    if "target_upper_bound" in edit:
        return float(edit["target_upper_bound"])
    if "reference_upper_bound" in edit and old_value >= 999.0:
        return min(1000.0, float(edit["reference_upper_bound"]) * max(0.0, multiplier))
    return min(1000.0, old_value * max(0.0, multiplier))


def apply_edits_to_controls(c: np.ndarray, control_cols: list[str], edits: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """c: (T-1, n_controls) physiology-predicted baseline controls. Returns
    (edited_c, notes) -- notes record which edits were applied vs. skipped,
    for auditability (never silent)."""
    c_edited = c.copy()
    notes: list[str] = []
    for edit in edits:
        rxn_id = str(edit.get("reaction_id", ""))
        edit_type = str(edit.get("edit_type", ""))
        if rxn_id not in REACTION_TO_CONTROL:
            notes.append(f"skipped_uncovered_reaction:{rxn_id}")
            continue
        control_name = REACTION_TO_CONTROL[rxn_id]
        if control_name not in control_cols:
            notes.append(f"skipped_uncovered_control:{control_name}")
            continue
        idx = control_cols.index(control_name)
        is_lb = control_name in CONTROL_IS_LOWER_BOUND
        cap = float(edit.get("capacity_multiplier", 1.0))
        if edit_type in DECREASE_EDIT_TYPES:
            mult = max(0.0, min(cap, float(edit.get("lower_bound_multiplier", cap)), float(edit.get("upper_bound_multiplier", cap))))
        elif edit_type in INCREASE_EDIT_TYPES:
            mult = cap
        else:
            notes.append(f"skipped_unsupported_edit_type:{edit_type}")
            continue
        for t in range(c_edited.shape[0]):
            c_edited[t, idx] = _scaled_bound(float(c[t, idx]), mult, edit, is_lower_bound=is_lb)
        notes.append(f"applied:{rxn_id}:{edit_type}:mult={mult:.4f}")
    return c_edited, notes


@dataclass
class RepairedVirtualScorer:
    params: "tss.StateSpaceParams"
    surrogate: "sur.GSMSurrogate"
    env_mean: np.ndarray
    env_std: np.ndarray
    checkpoint_meta: dict[str, Any]

    @classmethod
    def load(cls, checkpoint_path: Path, env_mean: np.ndarray, env_std: np.ndarray, surrogate: "sur.GSMSurrogate" | None = None) -> "RepairedVirtualScorer":
        params, meta = tss.load_checkpoint(checkpoint_path)
        if surrogate is None:
            pairs = sur.load_surrogate_training_pairs(
                DATA / "gem_state_space_generator_constraints.partial.csv",
                DATA / "gem_state_space_fluxes.csv",
                DATA / "gem_state_space_split_manifest.csv",
            )
            train_mask = pairs["split"].eq("train").to_numpy()
            surrogate = sur.fit_surrogate(pairs, train_mask=train_mask, ridge_lambda=1.0)
        return cls(params=params, surrogate=surrogate, env_mean=env_mean, env_std=env_std, checkpoint_meta=meta)

    def score(self, candidate: dict[str, Any]) -> dict[str, Any]:
        env = candidate["environment"]
        env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
        env_std_in = (env_raw - self.env_mean) / self.env_std

        cache = tss.forward(self.params, env_std_in, T_LEN)
        c_pre = cache.z[:-1, 0, :] @ self.params.Wc + self.params.bc  # (T_LEN-1, n_controls)
        c_baseline = c_pre * self.surrogate.control_std + self.surrogate.control_mean

        edits = candidate.get("edits", [])
        c_edited, edit_notes = apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, edits)

        # Disclosed safety clip: the surrogate is a polynomial-in-standardized-c
        # ridge fit with no principled extrapolation behaviour. An edit that
        # pushes a control many std outside its training support (observed:
        # capacity_multiplier=2.0 on PSY -> ~12 std out) can flip the
        # quadratic term's sign and predict a nonsensical negative flux, which
        # the mechanistic integration's max(0, ...) then locks at exactly 0
        # for the rest of the trajectory -- a surrogate-approximation failure,
        # not a physiology-model or edit-mapping failure (see
        # LEARNER_ARCHITECTURE_REPAIR.md Step 4). Clipping to a bounded
        # multiple of the training std is a standard, disclosed surrogate
        # safety measure (same spirit as predict_columns' existing
        # min/max+margin clip elsewhere in this codebase), not silent tuning.
        clip_lo = self.surrogate.control_mean - CONTROL_CLIP_STDS * self.surrogate.control_std
        clip_hi = self.surrogate.control_mean + CONTROL_CLIP_STDS * self.surrogate.control_std
        n_clipped = int(np.sum((c_edited < clip_lo) | (c_edited > clip_hi)))
        c_final = np.clip(c_edited, clip_lo, clip_hi)

        X = np.zeros(T_LEN)
        B = np.zeros(T_LEN)
        X[0], B[0] = X0, B0
        env_broadcast = env_raw
        for t in range(T_LEN - 1):
            values = self.surrogate.predict(c_final[t : t + 1], env_broadcast)
            biomass_flux = float(values[0, sur.SURROGATE_OUT_INDEX["biomass_flux"]])
            beta_flux = float(values[0, sur.SURROGATE_OUT_INDEX["beta_carotene_flux"]])
            X[t + 1] = max(1e-9, X[t] + DT_REAL * biomass_flux * X[t])
            B[t + 1] = max(0.0, B[t] + DT_REAL * beta_flux * X[t] - DT_REAL * BETA_DEG_BASE * B[t])

        n_applied = sum(1 for n in edit_notes if n.startswith("applied"))
        n_skipped = len(edit_notes) - n_applied
        return {
            "candidate_id": candidate.get("candidate_id"),
            "candidate_hash": candidate.get("candidate_hash"),
            "virtual_score": float(B[-1]),
            "final_product_pred": float(B[-1]),
            "final_biomass_pred": float(X[-1]),
            "product_AUC_pred": float(np.trapezoid(B, dx=DT_REAL)),
            "n_edits_total": len(edits),
            "n_edits_applied": n_applied,
            "n_edits_skipped": n_skipped,
            "n_control_values_clipped": n_clipped,
            "edit_notes": ";".join(edit_notes),
            "checkpoint_params_fingerprint": self.checkpoint_meta.get("params_fingerprint"),
            "scorer": "repaired_hybrid_learned_physiology",
        }
