#!/usr/bin/env python3
"""Variable-dimension hidden physiology controller family (Latent Capacity
Mismatch Benchmark, Stage C).

This module does NOT modify ``gem_backend.py``. It builds a parallel,
generalized rollout that reuses ``gem_backend`` primitives (``solve_staged``,
``apply_interval_constraints``, ``apply_enzyme_capacity_bounds``,
``pathway_congestion_terms``, ``initial_enzyme_capacities``, ``state_gamma``,
``next_state``, ``burden_terms``) so the frozen 3-LP-per-interval structure and
the existing discrete 7-state regulator (still keyed on the original
ox/atp/bottleneck triad only) are untouched.

Every ``HiddenStateSpec`` beyond the core three (``z_ox``, ``z_atp``,
``z_bottle``, always present, unmodified, mechanistic/existing) couples into a
real Yeast9 constraint via its ``coupling_fn``, applied *after*
``gem_backend.apply_interval_constraints`` has set the interval's baseline
bounds. ``D_true = N`` selects the first ``N - 3`` entries of
``STATE_LIBRARY``. See ``GENERATOR_VALIDITY.md`` and
``LATENT_CAPACITY_MISMATCH_DESIGN.md`` for the mechanism/evidence-class
documentation of each state, and for the frozen causal-rank acceptance gate
computed by :func:`sensitivity_matrix`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_backend as gem  # noqa: E402
from generate_gem_state_space_dataset import build_reporters, reporter_curve  # noqa: E402

EPS = gem.EPS

# Reaction IDs confirmed present (and uniquely resolved via gem.reaction_search)
# in the augmented Yeast9 model used by this project.
NADPH_SOURCE_RXN = "r_0466"  # glucose-6-phosphate dehydrogenase (main cytosolic PPP/NADPH source)
AMMONIUM_EXCHANGE_RXN = "r_1654"  # ammonium exchange (general nutrient-transport proxy)

STRESSED_STATES = {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"}
RECOVERY_ELIGIBLE_STATES = STRESSED_STATES | {"recovering"}


@dataclass
class BurdenContext:
    flux: dict[str, object]
    z: dict[str, float]
    cfg: "gem.GEMCultureConfig"
    congestion: dict[str, float]
    state: str


def recovery_multiplier(state: str) -> float:
    """Same discrete-regulator-gated repair acceleration as gem_backend.burden_terms."""
    return 2.8 if state in RECOVERY_ELIGIBLE_STATES else 1.0


# ---------------------------------------------------------------------------
# Generation functions: dz_i/dt = generation_fn(ctx) - (1/tau_hours) * rho * z_i
# ---------------------------------------------------------------------------

def _gen_nadph(ctx: BurdenContext) -> float:
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    beta = max(0.0, float(ctx.flux["beta_carotene_flux"]))
    return 0.10 * growth + 0.14 * beta


def _gen_proteostasis(ctx: BurdenContext) -> float:
    psy = max(0.0, float(ctx.flux.get("PSY_flux", 0.0)))
    des = max(0.0, float(ctx.flux.get("DES_flux", 0.0)))
    cyc = max(0.0, float(ctx.flux.get("CYC_flux", 0.0)))
    return 0.09 * (psy + des + cyc)


def _gen_membrane(ctx: BurdenContext) -> float:
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    oxygen_capacity = max(0.25, ctx.cfg.oxygen_uptake_at_100_do * ctx.cfg.DO / 100.0)
    oxygen_use = max(0.0, -float(ctx.flux["oxygen_uptake"]))
    return 0.07 * growth + 0.05 * (oxygen_use / max(oxygen_capacity, EPS))


def _gen_precursor_dyn(ctx: BurdenContext) -> float:
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    ggpp = max(0.0, float(ctx.flux.get("ggpp_flux", 0.0)))
    return 0.10 * growth + 0.06 * ggpp


def _gen_translation(ctx: BurdenContext) -> float:
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    return 0.20 * growth


def _gen_recovery(ctx: BurdenContext) -> float:
    return 0.05 if ctx.state in STRESSED_STATES else 0.0


def _gen_osmotic(ctx: BurdenContext) -> float:
    return 0.09 * abs(ctx.cfg.temperature - ctx.cfg.t_opt) / 8.0 + 0.09 * abs(ctx.cfg.pH - ctx.cfg.ph_opt)


def _gen_ox_slow(ctx: BurdenContext) -> float:
    oxygen_capacity = max(0.25, ctx.cfg.oxygen_uptake_at_100_do * ctx.cfg.DO / 100.0)
    oxygen_use = max(0.0, -float(ctx.flux["oxygen_uptake"]))
    beta = max(0.0, float(ctx.flux["beta_carotene_flux"]))
    return 0.05 * (oxygen_use / max(oxygen_capacity, EPS)) + 0.07 * beta


def _gen_atp_slow(ctx: BurdenContext) -> float:
    atpm = max(0.0, float(ctx.flux["atp_maintenance_flux"]))
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    return 0.05 * atpm + 0.04 * growth


def _gen_bottle_slow(ctx: BurdenContext) -> float:
    ggpp = max(0.0, float(ctx.flux.get("ggpp_flux", 0.0)))
    return 0.20 * ggpp


def _gen_nadph_slow(ctx: BurdenContext) -> float:
    growth = max(0.0, float(ctx.flux["biomass_flux"]))
    beta = max(0.0, float(ctx.flux["beta_carotene_flux"]))
    return 0.05 * growth + 0.06 * beta


# ---------------------------------------------------------------------------
# Coupling functions: mutate `model` reaction bounds after
# gem.apply_interval_constraints has already set the interval's baseline
# ox/atp/bottleneck/pathway/glucose/oxygen bounds. `z` is the dict of current
# (pre-update, i.e. this interval's) burden values keyed by state name.
# ---------------------------------------------------------------------------

def _couple_nadph(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(NADPH_SOURCE_RXN)
    factor = max(0.05, 1.0 - 0.6 * z["z_nadph"])
    if rxn.upper_bound > 0:
        rxn.upper_bound = rxn.upper_bound * factor
    if rxn.lower_bound < 0:
        rxn.lower_bound = rxn.lower_bound * factor


def _couple_proteostasis(model, cfg, z: dict[str, float]) -> None:
    factor = max(0.05, 1.0 - 0.5 * z["z_proteostasis"])
    for rxn_id in (gem.PSY_RXN, gem.DES_RXN, gem.CYC_RXN):
        rxn = model.reactions.get_by_id(rxn_id)
        rxn.upper_bound = rxn.upper_bound * factor


def _couple_membrane(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(AMMONIUM_EXCHANGE_RXN)
    factor = max(0.15, 1.0 - 0.5 * z["z_membrane"])
    if rxn.lower_bound < 0:
        rxn.lower_bound = rxn.lower_bound * factor


def _couple_precursor_dyn(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.NATIVE_GGPP_RXN)
    factor = max(0.1, 1.0 - 0.5 * z["z_precursor_dyn"])
    if rxn.upper_bound > 0:
        rxn.upper_bound = rxn.upper_bound * factor
    if rxn.lower_bound < 0:
        rxn.lower_bound = rxn.lower_bound * factor


def _couple_translation(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.BIOMASS_RXN)
    factor = max(0.05, 1.0 - 0.6 * z["z_translation"])
    rxn.upper_bound = 1000.0 * factor


def _couple_recovery(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.ATPM_RXN)
    relief = max(0.5, 1.0 - 0.3 * z["z_recovery"])
    rxn.lower_bound = rxn.lower_bound * relief
    rxn.upper_bound = max(rxn.lower_bound, rxn.upper_bound)


def _couple_osmotic(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.GLUCOSE_EXCHANGE)
    factor = max(0.2, 1.0 - 0.4 * z["z_osmotic"])
    if rxn.lower_bound < 0:
        rxn.lower_bound = rxn.lower_bound * factor


def _couple_ox_slow(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.PSY_RXN)
    factor = max(0.1, 1.0 - 0.3 * z["z_ox_slow"])
    rxn.upper_bound = rxn.upper_bound * factor


def _couple_atp_slow(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.CYC_RXN)
    factor = max(0.1, 1.0 - 0.3 * z["z_atp_slow"])
    rxn.upper_bound = rxn.upper_bound * factor


def _couple_bottle_slow(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.DES_RXN)
    factor = max(0.1, 1.0 - 0.3 * z["z_bottle_slow"])
    rxn.upper_bound = rxn.upper_bound * factor


def _couple_nadph_slow(model, cfg, z: dict[str, float]) -> None:
    rxn = model.reactions.get_by_id(gem.OXYGEN_EXCHANGE)
    factor = max(0.3, 1.0 - 0.2 * z["z_nadph_slow"])
    if rxn.lower_bound < 0:
        rxn.lower_bound = rxn.lower_bound * factor


def _make_scaling_coupling(state_name: str, rxn_id: str, side: str, floor: float, slope: float) -> Callable[[object, "gem.GEMCultureConfig", dict[str, float]], None]:
    """Factory for a tier-2 dimensionality-scaling coupling function: an
    independent multiplicative layer on an already-used reaction bound,
    keyed by its own state name. Reused for the D_true=15..24 library tier
    (declared design, not independently calibrated -- see STATE_LIBRARY
    docstring-equivalent entries below); whether these turn out causally
    distinct from their tier-1 counterparts is exactly what
    sensitivity_matrix's acceptance gate is for, not something asserted here."""

    def _coupling(model, cfg, z: dict[str, float]) -> None:
        rxn = model.reactions.get_by_id(rxn_id)
        factor = max(floor, 1.0 - slope * z[state_name])
        if side == "upper" and rxn.upper_bound > 0:
            rxn.upper_bound = rxn.upper_bound * factor
        elif side == "lower" and rxn.lower_bound < 0:
            rxn.lower_bound = rxn.lower_bound * factor

    return _coupling


@dataclass(frozen=True)
class HiddenStateSpec:
    name: str
    evidence_class: str  # "mechanistic" | "literature-shaped" | "declared design" | "arbitrary"
    tau_hours: float
    mechanism: str
    generation_fn: Callable[[BurdenContext], float]
    coupling_fn: Callable[[object, "gem.GEMCultureConfig", dict[str, float]], None]
    reporter_spec: dict[str, float] | None = None


CORE_STATES = ("z_ox", "z_atp", "z_bottle")  # always present; mechanistic; unmodified gem_backend dynamics

STATE_LIBRARY: list[HiddenStateSpec] = [
    HiddenStateSpec(
        "z_nadph", "literature-shaped", 1.5,
        "NADPH/redox cofactor availability constrains PPP-linked reductive biosynthetic capacity.",
        _gen_nadph, _couple_nadph,
        reporter_spec={"scale": 1.05, "offset": 0.03, "lag": 1, "alpha": 0.45, "noise": 0.006},
    ),
    HiddenStateSpec(
        "z_proteostasis", "literature-shaped", 3.0,
        "Heterologous protein/proteostasis burden imposes a shared folding-capacity ceiling on PSY+DES+CYC jointly.",
        _gen_proteostasis, _couple_proteostasis,
        reporter_spec={"scale": 1.00, "offset": 0.02, "lag": 2, "alpha": 0.35, "noise": 0.007},
    ),
    HiddenStateSpec(
        "z_membrane", "declared design", 4.0,
        "Membrane/transport burden reduces general nutrient transport capacity (ammonium exchange used as proxy).",
        _gen_membrane, _couple_membrane, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_precursor_dyn", "literature-shaped", 2.5,
        "Dynamic native-isoprenoid competition for GGPP, distinct from the static per-candidate edit multiplier.",
        _gen_precursor_dyn, _couple_precursor_dyn, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_translation", "literature-shaped", 1.0,
        "Ribosome/protein-allocation trade-off imposes a dynamic ceiling on the maximum growth rate itself.",
        _gen_translation, _couple_translation, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_recovery", "declared design", 8.0,
        "Slow cumulative-stress-exposure recovery/repair capacity; relieves ATP maintenance independently of z_atp.",
        _gen_recovery, _couple_recovery, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_osmotic", "declared design", 1.0,
        "General osmotic/environmental stress reduces glucose transport capacity, independent of substrate depletion.",
        _gen_osmotic, _couple_osmotic, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_ox_slow", "declared design", 7.0,
        "Dimensionality-scaling variant of z_ox (slow lipid-peroxidation-type damage) -- not independently calibrated.",
        _gen_ox_slow, _couple_ox_slow, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_atp_slow", "declared design", 8.0,
        "Dimensionality-scaling variant of z_atp (chronic energy-charge drift) -- not independently calibrated.",
        _gen_atp_slow, _couple_atp_slow, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_bottle_slow", "declared design", 9.0,
        "Dimensionality-scaling variant of z_bottle (slow chronic congestion) -- not independently calibrated.",
        _gen_bottle_slow, _couple_bottle_slow, reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_nadph_slow", "declared design", 10.0,
        "Dimensionality-scaling variant of z_nadph (chronic respiratory/redox capacity drift) -- not independently calibrated.",
        _gen_nadph_slow, _couple_nadph_slow, reporter_spec=None,
    ),
    # Tier 2 (states 15-24): declared-design, dimensionality-scaling variants
    # reusing the ten distinct reaction targets already established above,
    # each with its own generation dynamics and time constant, added purely
    # to reach the higher end of the D_true sweep (>~14). Whether each is
    # causally distinguishable from its tier-1 counterpart on the same
    # reaction is an empirical question for sensitivity_matrix, not assumed.
    HiddenStateSpec(
        "z_redox2", "declared design", 2.0,
        "Tier-2 variant: second independent redox-linked layer on the NADPH source reaction.",
        _gen_nadph, _make_scaling_coupling("z_redox2", NADPH_SOURCE_RXN, "upper", 0.05, 0.4), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_transport2", "declared design", 5.0,
        "Tier-2 variant: second independent transport-burden layer on the ammonium exchange reaction.",
        _gen_membrane, _make_scaling_coupling("z_transport2", AMMONIUM_EXCHANGE_RXN, "lower", 0.2, 0.3), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_precursor2", "declared design", 3.5,
        "Tier-2 variant: second independent precursor-competition layer on the native GGPP reaction.",
        _gen_precursor_dyn, _make_scaling_coupling("z_precursor2", gem.NATIVE_GGPP_RXN, "upper", 0.15, 0.35), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_growth2", "declared design", 1.5,
        "Tier-2 variant: second independent allocation-burden layer on the biomass reaction upper bound.",
        _gen_translation, _make_scaling_coupling("z_growth2", gem.BIOMASS_RXN, "upper", 0.1, 0.4), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_maintenance2", "declared design", 9.0,
        "Tier-2 variant: second independent slow relief layer on the ATP maintenance lower bound.",
        _gen_recovery, _make_scaling_coupling("z_maintenance2", gem.ATPM_RXN, "lower", 0.4, 0.25), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_uptake2", "declared design", 1.2,
        "Tier-2 variant: second independent stress-linked layer on the glucose exchange lower bound.",
        _gen_osmotic, _make_scaling_coupling("z_uptake2", gem.GLUCOSE_EXCHANGE, "lower", 0.15, 0.35), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_psy2", "declared design", 6.0,
        "Tier-2 variant: second independent slow-damage layer on the PSY reaction upper bound.",
        _gen_ox_slow, _make_scaling_coupling("z_psy2", gem.PSY_RXN, "upper", 0.05, 0.35), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_des2", "declared design", 7.5,
        "Tier-2 variant: second independent slow-damage layer on the DES reaction upper bound.",
        _gen_bottle_slow, _make_scaling_coupling("z_des2", gem.DES_RXN, "upper", 0.05, 0.35), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_cyc2", "declared design", 8.5,
        "Tier-2 variant: second independent slow-damage layer on the CYC reaction upper bound.",
        _gen_atp_slow, _make_scaling_coupling("z_cyc2", gem.CYC_RXN, "upper", 0.05, 0.35), reporter_spec=None,
    ),
    HiddenStateSpec(
        "z_respiration2", "declared design", 4.5,
        "Tier-2 variant: second independent chronic-respiratory-drift layer on the oxygen exchange lower bound.",
        _gen_nadph_slow, _make_scaling_coupling("z_respiration2", gem.OXYGEN_EXCHANGE, "lower", 0.25, 0.3), reporter_spec=None,
    ),
]

MAX_D_TRUE = 3 + len(STATE_LIBRARY)


def build_controller(d_true: int) -> list[HiddenStateSpec]:
    """Return the ordered list of *extra* (beyond the core 3) HiddenStateSpecs for D_true."""
    if d_true < 3:
        raise ValueError("D_true must be >= 3 -- the core ox/atp/bottleneck triad is always present.")
    if d_true > MAX_D_TRUE:
        raise ValueError(f"D_true={d_true} exceeds STATE_LIBRARY capacity (max {MAX_D_TRUE}).")
    return STATE_LIBRARY[: d_true - 3]


# ---------------------------------------------------------------------------
# Generalized rollout
# ---------------------------------------------------------------------------

def run_dynamic_culture_generic(
    base_augmented_model,
    source_path: Path,
    cfg: "gem.GEMCultureConfig",
    extra_states: list[HiddenStateSpec],
    culture_id: str = "controller_family_culture",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generalized counterpart of gem_backend.run_dynamic_culture. With
    extra_states=[] this reproduces gem_backend.run_dynamic_culture bit-for-bit
    (see tests/test_gem_controller_family.py)."""
    t = np.arange(cfg.n_time, dtype=float) * cfg.dt
    x = np.zeros(cfg.n_time)
    b = np.zeros(cfg.n_time)
    s_glc = np.zeros(cfg.n_time)
    burdens = np.zeros((cfg.n_time, 4))  # z_ox, z_atp, z_bottle, z_er (core, unchanged)
    enzyme_capacities = np.zeros((cfg.n_time, 3))
    extra_names = [s.name for s in extra_states]
    extra_z = np.zeros((cfg.n_time, len(extra_states)))
    x[0] = cfg.x0
    b[0] = cfg.b0
    s_glc[0] = cfg.s_glc0
    enzyme_capacities[0] = gem.initial_enzyme_capacities(cfg)
    state = "balanced"
    dwell = 0
    traj_rows, flux_rows, constraint_rows = [], [], []
    totals = {k: 0 for k in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves"]}
    checksum = gem.sha256(source_path)

    for i in range(cfg.n_time - 1):
        interval_model = base_augmented_model.copy()
        constraints = gem.apply_interval_constraints(interval_model, cfg, burdens[i], state, enzyme_capacities[i], s_glc=s_glc[i], x_biomass=x[i])
        z_extra_now = {name: float(extra_z[i, k]) for k, name in enumerate(extra_names)}
        for spec in extra_states:
            spec.coupling_fn(interval_model, cfg, z_extra_now)
        gamma = gem.state_gamma(state, cfg)
        flux = gem.solve_staged(interval_model, cfg, gamma, state=state)
        for key in totals:
            totals[key] += int(flux[key])
        congestion = gem.pathway_congestion_terms(flux, constraints)

        beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * burdens[i, 0])
        dx = float(flux["biomass_flux"]) * x[i]
        db = float(flux["beta_carotene_flux"]) * x[i] - beta_deg * b[i]
        glucose_uptake_magnitude = max(0.0, -float(flux["glucose_uptake"]))
        x[i + 1] = max(1e-9, x[i] + cfg.dt * dx)
        b[i + 1] = max(0.0, b[i] + cfg.dt * db)
        s_glc[i + 1] = max(0.0, s_glc[i] - cfg.dt * glucose_uptake_magnitude * x[i])

        terms = gem.burden_terms(cfg, burdens[i], flux, state, congestion if cfg.capacity_mode == "dynamic_congestion_feedback" else None)
        burdens[i + 1] = np.array([terms["oxidative_burden_after"], terms["ATP_pressure_after"], terms["bottleneck_after"], terms["ER_after"]])

        ctx = BurdenContext(flux=flux, z={**z_extra_now, "z_ox": burdens[i, 0], "z_atp": burdens[i, 1], "z_bottle": burdens[i, 2]}, cfg=cfg, congestion=congestion, state=state)
        for k, spec in enumerate(extra_states):
            gen = spec.generation_fn(ctx)
            repair = (1.0 / spec.tau_hours) * recovery_multiplier(state) * extra_z[i, k]
            extra_z[i + 1, k] = float(np.clip(extra_z[i, k] + cfg.dt * (gen - repair), 0.0, 1.5))

        enzyme_capacities[i + 1], capacity_terms = gem.update_enzyme_capacities(cfg, enzyme_capacities[i], burdens[i + 1], flux, state, congestion)
        new_state = gem.next_state(state, burdens[i + 1], dwell)
        if new_state == state:
            dwell += 1
        else:
            state = new_state
            dwell = 0

        common = {
            "culture_id": culture_id,
            "interval_index": i,
            "time": float(t[i]),
            "dt": cfg.dt,
            "temperature": cfg.temperature,
            "pH": cfg.pH,
            "DO": cfg.DO,
            "metabolic_backend": "yeast_gem_lp",
            "gem_model_id": base_augmented_model.id,
            "gem_source_checksum": checksum,
            "solver_name": str(base_augmented_model.solver.interface),
            "n_time_points": cfg.n_time,
            "n_simulation_intervals": cfg.n_time - 1,
            "n_surrogate_evaluations": 0,
            "capacity_mode": cfg.capacity_mode,
            "d_true": 3 + len(extra_states),
            "state": constraints["state"],
            "next_state": state,
            "z_ox": burdens[i, 0],
            "z_atp": burdens[i, 1],
            "z_bottle": burdens[i, 2],
            "z_er": burdens[i, 3],
            "E_PSY": enzyme_capacities[i, 0],
            "E_DES": enzyme_capacities[i, 1],
            "E_CYC": enzyme_capacities[i, 2],
            "S_glc": s_glc[i],
            **z_extra_now,
        }
        flux_rows.append({**common, **flux, **congestion})
        c = {**common, **constraints, "gamma_growth_fraction": gamma}
        c.update(terms)
        c.update(congestion)
        c.update(capacity_terms)
        constraint_rows.append(c)

    for i, tt in enumerate(t):
        traj_rows.append(
            {
                "culture_id": culture_id,
                "time_index": i,
                "time": float(tt),
                "temperature": cfg.temperature,
                "pH": cfg.pH,
                "DO": cfg.DO,
                "X": float(x[i]),
                "B_total": float(b[i]),
                "S_glc": float(s_glc[i]),
                "z_ox": float(burdens[i, 0]),
                "z_atp": float(burdens[i, 1]),
                "z_bottle": float(burdens[i, 2]),
                "z_er": float(burdens[i, 3]),
                "E_PSY": float(enzyme_capacities[i, 0]),
                "E_DES": float(enzyme_capacities[i, 1]),
                "E_CYC": float(enzyme_capacities[i, 2]),
                "d_true": 3 + len(extra_states),
                "capacity_mode": cfg.capacity_mode,
                "metabolic_backend": "yeast_gem_lp",
                "gem_model_id": base_augmented_model.id,
                "gem_source_checksum": checksum,
                "n_time_points": cfg.n_time,
                "n_simulation_intervals": cfg.n_time - 1,
                **totals,
                "n_surrogate_evaluations": 0,
                **{name: float(extra_z[i, k]) for k, name in enumerate(extra_names)},
            }
        )
    return pd.DataFrame(traj_rows), pd.DataFrame(flux_rows), pd.DataFrame(constraint_rows), pd.DataFrame([{"culture_id": culture_id, "d_true": 3 + len(extra_states), **totals, "n_surrogate_evaluations": 0}])


def build_reporters_generic(traj: pd.DataFrame, extra_states: list[HiddenStateSpec], seed: int) -> pd.DataFrame:
    """Core reporters (R_ox, R_atp, R_er, R_E_PSY/DES/CYC) via the existing,
    already-validated generate_gem_state_space_dataset.build_reporters, plus
    additional reporters for any extra_states that declare a reporter_spec
    (partial observability -- states without a spec have no reporter)."""
    core = build_reporters(traj)
    rng = np.random.default_rng(seed)
    for cid, g in traj.groupby("culture_id"):
        g = g.sort_values("time")
        idx = core["culture_id"] == cid
        for spec in extra_states:
            if spec.reporter_spec is None or spec.name not in g.columns:
                continue
            r = spec.reporter_spec
            curve = reporter_curve(g[spec.name].to_numpy(float), r["scale"], r["offset"], r["lag"], r["alpha"], r["noise"], rng)
            col = f"R_{spec.name.removeprefix('z_')}"
            core.loc[idx, col] = curve
            core.loc[idx, f"{col}_source"] = spec.name
    return core


# ---------------------------------------------------------------------------
# Causal-rank sensitivity diagnostic (Section 9 of the handover spec) --
# frozen acceptance gate, computed BEFORE any model training.
# ---------------------------------------------------------------------------

CONSTRAINT_KEYS = (
    "glucose_lower_bound",
    "oxygen_lower_bound",
    "atp_maintenance_lower_bound",
    "pathway_capacity_upper_bound",
    "PSY_effective_upper_bound",
    "DES_effective_upper_bound",
    "CYC_effective_upper_bound",
)
FLUX_KEYS = (
    "biomass_flux",
    "beta_carotene_flux",
    "oxygen_uptake",
    "atp_maintenance_flux",
    "ggpp_flux",
    "PSY_flux",
    "DES_flux",
    "CYC_flux",
)
EXTRA_BOUND_PROBES: dict[str, tuple[str, str]] = {
    # state_name -> (reaction_id, "upper"|"lower")
    "z_nadph": (NADPH_SOURCE_RXN, "upper"),
    "z_proteostasis": (gem.PSY_RXN, "upper"),
    "z_membrane": (AMMONIUM_EXCHANGE_RXN, "lower"),
    "z_precursor_dyn": (gem.NATIVE_GGPP_RXN, "upper"),
    "z_translation": (gem.BIOMASS_RXN, "upper"),
    "z_recovery": (gem.ATPM_RXN, "lower"),
    "z_osmotic": (gem.GLUCOSE_EXCHANGE, "lower"),
    "z_ox_slow": (gem.PSY_RXN, "upper"),
    "z_atp_slow": (gem.CYC_RXN, "upper"),
    "z_bottle_slow": (gem.DES_RXN, "upper"),
    "z_nadph_slow": (gem.OXYGEN_EXCHANGE, "lower"),
}


def _solve_at_reference(base_model, cfg: "gem.GEMCultureConfig", z_all: dict[str, float], extra_states: list[HiddenStateSpec], state: str = "balanced") -> tuple[dict, dict]:
    model = base_model.copy()
    core_burdens = np.array([z_all["z_ox"], z_all["z_atp"], z_all["z_bottle"], 0.0])
    constraints = gem.apply_interval_constraints(model, cfg, core_burdens, state, enzyme_capacities=None)
    for spec in extra_states:
        spec.coupling_fn(model, cfg, z_all)
    probe_bounds = {}
    for name, (rxn_id, side) in EXTRA_BOUND_PROBES.items():
        if name not in z_all or name not in {s.name for s in extra_states}:
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        probe_bounds[f"{name}_probe_bound"] = float(rxn.upper_bound if side == "upper" else rxn.lower_bound)
    gamma = gem.state_gamma(state, cfg)
    flux = gem.solve_staged(model, cfg, gamma, state=state)
    output = {k: float(constraints.get(k, np.nan)) for k in CONSTRAINT_KEYS}
    output.update(probe_bounds)
    output.update({k: float(flux.get(k, np.nan)) for k in FLUX_KEYS})
    return output, flux


def sensitivity_matrix(d_true: int, cfg: "gem.GEMCultureConfig", base_model, delta: float = 0.15, reference: float = 0.30, state: str = "balanced") -> dict[str, object]:
    """Finite-difference Jacobian of the constraint+flux output vector w.r.t.
    each of the D_true hidden states, evaluated at a single reference point
    (z_i = reference for all i, fixed environment/state). D_true+1 LP solves
    total. Returns SVD-derived effective rank, condition number, and pairwise
    column cosine similarities -- the diagnostics behind the frozen acceptance
    gate (effective_rank >= 0.6*D_true, at most 1 pair with |cos| > 0.9)."""
    extra_states = build_controller(d_true)
    names = list(CORE_STATES) + [s.name for s in extra_states]
    z_ref = {name: reference for name in names}
    baseline_out, _ = _solve_at_reference(base_model, cfg, z_ref, extra_states, state)
    output_keys = list(baseline_out.keys())
    baseline_vec = np.array([baseline_out[k] for k in output_keys])

    columns = []
    for name in names:
        z_perturbed = dict(z_ref)
        z_perturbed[name] = reference + delta
        out, _ = _solve_at_reference(base_model, cfg, z_perturbed, extra_states, state)
        vec = np.array([out.get(k, np.nan) for k in output_keys])
        columns.append((vec - baseline_vec) / delta)
    J = np.column_stack(columns)  # shape (n_outputs, D_true)
    J = np.nan_to_num(J, nan=0.0)

    singular_values = np.linalg.svd(J, compute_uv=False)
    sv_sum = float(np.sum(singular_values))
    sv_sq_sum = float(np.sum(singular_values**2))
    effective_rank = (sv_sum**2 / sv_sq_sum) if sv_sq_sum > EPS else 0.0
    condition_number = float(singular_values[0] / max(singular_values[-1], EPS)) if len(singular_values) else float("nan")

    norms = np.linalg.norm(J, axis=0)
    norms_safe = np.where(norms > EPS, norms, EPS)
    J_normed = J / norms_safe
    cos_sim = J_normed.T @ J_normed
    high_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(cos_sim[i, j]) > 0.9:
                high_pairs.append((names[i], names[j], float(cos_sim[i, j])))

    gate_rank = effective_rank >= 0.6 * d_true
    gate_collinearity = len(high_pairs) <= 1
    return {
        "d_true": d_true,
        "state_names": names,
        "output_keys": output_keys,
        "jacobian": J,
        "singular_values": singular_values,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "high_cosine_pairs": high_pairs,
        "gate_effective_rank_pass": bool(gate_rank),
        "gate_collinearity_pass": bool(gate_collinearity),
        "gate_pass": bool(gate_rank and gate_collinearity),
    }
