#!/usr/bin/env python3
"""Shared real yeast-GEM helpers for the beta-carotene dFBA pilot."""

from __future__ import annotations

import gzip
import hashlib
import os
import platform
import shutil
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RAW_ROOT = Path("/Users/julianedberthartono/Jelly/igem/data/raw")
MODELS = ROOT / "models"

BIOMASS_RXN = "r_2111"
BIOMASS_PSEUDO_RXN = "r_4041"
GLUCOSE_EXCHANGE = "r_1714"
OXYGEN_EXCHANGE = "r_1992"
ATPM_RXN = "r_4046"
NATIVE_GGPP_RXN = "r_0461"
NATIVE_GGPP_MET = "s_1311"
NATIVE_PPI_MET = "s_0633"
PRODUCT_RXN = "DM_beta_carotene_c"
PSY_RXN = "BETA_PHYTOENE_SYNTHASE"
DES_RXN = "BETA_PHYTOENE_DESATURASE"
CYC_RXN = "BETA_LYCOPENE_CYCLASE"
PATHWAY_RXNS = (
    PSY_RXN,
    DES_RXN,
    CYC_RXN,
    PRODUCT_RXN,
)
CAPACITY_MODES = (
    "no_dynamic_capacity",
    "constant_pathway_caps",
    "environment_initial_capacity",
    "dynamic_damage_recovery",
    "dynamic_congestion_feedback",
)
STATE_ORDER = (
    "balanced",
    "productive",
    "oxidative_stress",
    "energy_limited",
    "pathway_limited",
    "combined_overload",
    "recovering",
)
EPS = 1e-9


class GEMPilotError(RuntimeError):
    pass


@dataclass(frozen=True)
class GEMCultureConfig:
    temperature: float = 30.0
    pH: float = 5.0
    DO: float = 40.0
    n_time: int = 25
    dt: float = 0.25
    x0: float = 0.08
    b0: float = 0.0
    beta_deg_base: float = 0.002
    glucose_uptake: float = 10.0
    oxygen_uptake_at_100_do: float = 12.0
    baseline_atpm: float = 0.70
    t_opt: float = 30.0
    ph_opt: float = 5.0
    gamma_balanced: float = 0.72
    gamma_productive: float = 0.66
    gamma_oxidative_stress: float = 0.57
    gamma_energy_limited: float = 0.52
    gamma_pathway_limited: float = 0.55
    gamma_combined_overload: float = 0.43
    gamma_recovering: float = 0.58
    pathway_cap_balanced: float = 1.00
    pathway_cap_productive: float = 1.10
    pathway_cap_oxidative_stress: float = 0.70
    pathway_cap_energy_limited: float = 0.85
    pathway_cap_pathway_limited: float = 0.45
    pathway_cap_combined_overload: float = 0.30
    pathway_cap_recovering: float = 0.65
    allocation_protocol: str = "A"
    capacity_mode: str = "no_dynamic_capacity"
    psy_base_capacity: float = 0.34
    des_base_capacity: float = 0.28
    cyc_base_capacity: float = 0.23
    psy_initial_base: float = 0.82
    des_initial_base: float = 0.76
    cyc_initial_base: float = 0.72
    enzyme_min_capacity: float = 0.08
    psy_synthesis_base: float = 0.070
    des_synthesis_base: float = 0.060
    cyc_synthesis_base: float = 0.055
    psy_decay_base: float = 0.018
    des_decay_base: float = 0.024
    cyc_decay_base: float = 0.022
    psy_oxidative_sensitivity: float = 0.040
    des_oxidative_sensitivity: float = 0.125
    cyc_oxidative_sensitivity: float = 0.160
    psy_atp_sensitivity: float = 0.020
    des_atp_sensitivity: float = 0.060
    cyc_atp_sensitivity: float = 0.085
    psy_bottleneck_sensitivity: float = 0.110
    des_bottleneck_sensitivity: float = 0.055
    cyc_bottleneck_sensitivity: float = 0.040
    phytoene_congestion_sensitivity: float = 0.320
    lycopene_congestion_sensitivity: float = 0.280
    oxidative_beta_coupling: float = 0.22
    substrate_limited: bool = False
    s_glc0: float = 1000.0


def require_cobra():
    try:
        import cobra  # type: ignore
    except Exception as exc:
        raise GEMPilotError("COBRApy is required for the real GEM backend and is not available.") from exc
    return cobra


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_assets(raw_root: Path = RAW_ROOT) -> list[Path]:
    if not raw_root.exists():
        return []
    suffixes = (".xml", ".sbml", ".xml.gz", ".sbml.gz")
    return sorted(path for path in raw_root.rglob("*") if path.is_file() and any(str(path).lower().endswith(s) for s in suffixes))


def configured_gem_path(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("YEAST_GEM_PATH"):
        candidates.append(Path(os.environ["YEAST_GEM_PATH"]))
    candidates.extend([MODELS / "yeast_gem.xml", MODELS / "yeast_gem.json", MODELS / "yeast9.xml", MODELS / "yeast8.xml"])
    candidates.extend(discover_assets())
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise GEMPilotError("No yeast GEM asset found. Set YEAST_GEM_PATH or place a Yeast8/Yeast9 XML under the raw data directory.")


def _load_path_for_cobra(path: Path) -> tuple[Path, str | None]:
    if str(path).lower().endswith(".gz"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path.stem).suffix or ".xml")
        with gzip.open(path, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return Path(tmp.name), tmp.name
    return path, None


def load_model(cobra, path: Path):
    load_path, tmp_name = _load_path_for_cobra(path)
    try:
        if load_path.suffix.lower() == ".json":
            model = cobra.io.load_json_model(str(load_path))
        else:
            model = cobra.io.read_sbml_model(str(load_path))
    except Exception as exc:
        raise GEMPilotError(f"Could not load GEM asset with COBRApy: {path}") from exc
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
    return model


def reaction_search(model, patterns: tuple[str, ...], boundary_only: bool = False):
    collection = model.boundary if boundary_only else model.reactions
    rows = []
    for rxn in collection:
        text = f"{rxn.id} {rxn.name} {rxn.reaction} {' '.join(m.id + ' ' + m.name for m in rxn.metabolites)}".lower()
        if any(p.lower() in text for p in patterns):
            rows.append(rxn)
    return rows


def objective_reactions(model) -> list[str]:
    ids = []
    for rxn in model.reactions:
        coeffs = model.objective.get_linear_coefficients([rxn.forward_variable, rxn.reverse_variable])
        if abs(coeffs.get(rxn.forward_variable, 0.0)) > 0 or abs(coeffs.get(rxn.reverse_variable, 0.0)) > 0:
            ids.append(rxn.id)
    return sorted(ids)


def evaluate_candidate(cobra, path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "full_path": str(path.resolve()),
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "compression_status": "gzip" if str(path).lower().endswith(".gz") else "plain",
        "cobra_can_load": False,
        "selection_score": 0,
    }
    try:
        model = load_model(cobra, path)
        sol = model.optimize()
        biomass = reaction_search(model, ("biomass", "growth"))
        glucose = reaction_search(model, ("glucose", "glc"), boundary_only=True)
        oxygen = reaction_search(model, ("oxygen", "o2"), boundary_only=True)
        atpm = reaction_search(model, ("ATPM", "maintenance", "atp maintenance"))
        grr_count = sum(1 for r in model.reactions if r.gene_reaction_rule)
        score = (
            10 * int(sol.status == "optimal")
            + 5 * int(len(model.genes) > 0)
            + 5 * int(grr_count > 0)
            + 5 * int(bool(objective_reactions(model)))
            + 3 * int(bool(biomass))
            + 3 * int(bool(glucose))
            + 3 * int(bool(oxygen))
            + 3 * int(bool(atpm))
            + int("yeast" in f"{model.id} {model.name}".lower())
        )
        row.update(
            {
                "cobra_can_load": True,
                "sbml_model_id": model.id,
                "model_name": model.name,
                "organism_annotation": str(getattr(model, "annotation", {}).get("taxonomy", "") or getattr(model, "annotation", {}).get("organism", "")),
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
                "gene_reaction_rule_count": grr_count,
                "objective_reactions": ";".join(objective_reactions(model)),
                "biomass_candidates": ";".join(r.id for r in biomass[:20]),
                "glucose_exchange_candidates": ";".join(r.id for r in glucose[:20]),
                "oxygen_exchange_candidates": ";".join(r.id for r in oxygen[:20]),
                "atp_maintenance_candidates": ";".join(r.id for r in atpm[:20]),
                "baseline_status": sol.status,
                "baseline_objective": float(sol.objective_value) if sol.objective_value is not None else np.nan,
                "selection_score": score,
            }
        )
    except Exception as exc:
        row["load_error"] = f"{type(exc).__name__}: {exc}"
    return row


def discover_and_select_assets(raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    cobra = require_cobra()
    rows = [evaluate_candidate(cobra, path) for path in discover_assets(raw_root)]
    if not rows:
        raise GEMPilotError(f"No XML/SBML candidates found under {raw_root}.")
    candidates = pd.DataFrame(rows).sort_values(["selection_score", "file_size_bytes"], ascending=[False, False])
    selected = candidates[candidates["cobra_can_load"]].head(1)
    if selected.empty:
        raise GEMPilotError("No discovered GEM candidate could be loaded by COBRApy.")
    selected_path = Path(str(selected.iloc[0]["full_path"]))
    selected = selected.assign(selection_reason="highest compatible score: parses, has genes/GPRs, objective, exchanges, ATP maintenance, and yeast metadata")
    DATA.mkdir(exist_ok=True)
    candidates.to_csv(DATA / "yeast_gem_asset_candidates.csv", index=False)
    selected.to_csv(DATA / "yeast_gem_selected_asset.csv", index=False)
    return candidates, selected, selected_path


def installed_versions() -> pd.DataFrame:
    rows = [{"package": "python", "version": platform.python_version(), "status": "installed"}]
    for package, module_name in [
        ("cobra", "cobra"),
        ("optlang", "optlang"),
        ("swiglpk", "swiglpk"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            status = "installed"
        except Exception as exc:
            version = ""
            status = f"missing: {type(exc).__name__}"
        rows.append({"package": package, "version": str(version), "status": status})
    return pd.DataFrame(rows)


def verify_minimal_lp(cobra) -> dict[str, object]:
    from cobra import Metabolite, Model, Reaction

    model = Model("minimal_solver_check")
    a = Metabolite("a_c", compartment="c")
    source = Reaction("SOURCE_A")
    source.lower_bound = 0
    source.upper_bound = 10
    source.add_metabolites({a: 1})
    sink = Reaction("SINK_A")
    sink.lower_bound = 0
    sink.upper_bound = 1000
    sink.add_metabolites({a: -1})
    model.add_reactions([source, sink])
    model.objective = sink
    sol = model.optimize()
    if sol.status != "optimal":
        raise GEMPilotError("Minimal LP solver check failed.")
    return {"minimal_lp_status": sol.status, "minimal_lp_objective": float(sol.objective_value), "solver": str(model.solver.interface)}


def install_beta_carotene_pathway(model):
    cobra = require_cobra()
    from cobra import Metabolite, Reaction

    aug = model.copy()
    existing = {rxn.id for rxn in aug.reactions}
    collisions = sorted(existing.intersection(PATHWAY_RXNS))
    if collisions:
        raise GEMPilotError(f"Beta-carotene pathway reaction IDs already exist: {collisions}")
    ggpp = aug.metabolites.get_by_id(NATIVE_GGPP_MET)
    ppi = aug.metabolites.get_by_id(NATIVE_PPI_MET)
    phytoene = Metabolite("beta_phytoene_c", name="phytoene", compartment="c")
    lycopene = Metabolite("beta_lycopene_c", name="lycopene", compartment="c")
    beta_carotene = Metabolite("beta_carotene_c", name="beta-carotene", compartment="c")

    rxns = []
    r1 = Reaction("BETA_PHYTOENE_SYNTHASE")
    r1.name = "heterologous phytoene synthase"
    r1.lower_bound = 0
    r1.upper_bound = 1000
    r1.add_metabolites({ggpp: -2.0, phytoene: 1.0, ppi: 2.0})
    rxns.append(r1)

    r2 = Reaction("BETA_PHYTOENE_DESATURASE")
    r2.name = "heterologous phytoene-to-lycopene lump"
    r2.lower_bound = 0
    r2.upper_bound = 1000
    r2.add_metabolites({phytoene: -1.0, lycopene: 1.0})
    rxns.append(r2)

    r3 = Reaction("BETA_LYCOPENE_CYCLASE")
    r3.name = "heterologous lycopene beta-cyclase lump"
    r3.lower_bound = 0
    r3.upper_bound = 1000
    r3.add_metabolites({lycopene: -1.0, beta_carotene: 1.0})
    rxns.append(r3)

    r4 = Reaction(PRODUCT_RXN)
    r4.name = "beta-carotene accumulation demand"
    r4.lower_bound = 0
    r4.upper_bound = 1000
    r4.add_metabolites({beta_carotene: -1.0})
    rxns.append(r4)
    aug.add_reactions(rxns)
    manifest_rows = []
    for rxn in rxns:
        manifest_rows.append(
            {
                "reaction_id": rxn.id,
                "name": rxn.name,
                "stoichiometry": "; ".join(f"{coef:g} {met.id} ({met.name})" for met, coef in rxn.metabolites.items()),
                "compartment": "c",
                "lower_bound": rxn.lower_bound,
                "upper_bound": rxn.upper_bound,
                "reversibility": bool(rxn.lower_bound < 0 and rxn.upper_bound > 0),
                "status": "heterologous" if rxn.id != PRODUCT_RXN else "synthetic_accumulation_sink",
                "mass_balance_status": "not_formula_balanced_lumped_step" if rxn.id != PRODUCT_RXN else "intentional_unbalanced_product_sink",
                "biological_interpretation": {
                    "BETA_PHYTOENE_SYNTHASE": "drains native GGPP into phytoene using heterologous crtB-style chemistry",
                    "BETA_PHYTOENE_DESATURASE": "lumps carotenoid desaturation from phytoene to lycopene",
                    "BETA_LYCOPENE_CYCLASE": "lumps lycopene cyclization to beta-carotene",
                    PRODUCT_RXN: "accumulated beta-carotene titer sink outside the GEM biomass objective",
                }[rxn.id],
                "lumping_or_sink_justification": "Pilot-scale pathway audit: drains native GGPP without creating ATP, reducing equivalents, carbon, or precursor-producing cycles.",
            }
        )
    native_rows = [
        {
            "reaction_id": NATIVE_GGPP_RXN,
            "name": aug.reactions.get_by_id(NATIVE_GGPP_RXN).name,
            "stoichiometry": aug.reactions.get_by_id(NATIVE_GGPP_RXN).reaction,
            "compartment": "c",
            "lower_bound": aug.reactions.get_by_id(NATIVE_GGPP_RXN).lower_bound,
            "upper_bound": aug.reactions.get_by_id(NATIVE_GGPP_RXN).upper_bound,
            "reversibility": bool(aug.reactions.get_by_id(NATIVE_GGPP_RXN).reversibility),
            "status": "native",
            "mass_balance_status": "native_model_reaction",
            "biological_interpretation": "native geranylgeranyl diphosphate formation from farnesyl diphosphate and IPP",
            "lumping_or_sink_justification": "Reused native Yeast9 isoprenoid reaction; no duplicate GGPP formation added.",
        }
    ]
    return aug, pd.DataFrame(native_rows + manifest_rows)


def set_objective(model, rxn_id: str) -> None:
    if rxn_id not in {rxn.id for rxn in model.reactions}:
        raise GEMPilotError(f"Objective reaction {rxn_id} is absent.")
    model.objective = model.reactions.get_by_id(rxn_id)


def env_factor(temp: float, ph: float, cfg: GEMCultureConfig) -> float:
    temp_factor = 1.0 + 0.018 * abs(temp - cfg.t_opt)
    ph_factor = 1.0 + 0.12 * abs(ph - cfg.ph_opt)
    return temp_factor * ph_factor


def state_gamma(state: str, cfg: GEMCultureConfig) -> float:
    return getattr(cfg, f"gamma_{state}")


def state_pathway_cap(state: str, cfg: GEMCultureConfig) -> float:
    return getattr(cfg, f"pathway_cap_{state}")


def capacity_env_factors(cfg: GEMCultureConfig) -> dict[str, float]:
    temp_dev = cfg.temperature - cfg.t_opt
    ph_dev = cfg.pH - cfg.ph_opt
    do_dev = (cfg.DO - 40.0) / 40.0
    return {
        "PSY": float(np.clip(np.exp(-0.010 * temp_dev**2) * np.exp(-0.090 * ph_dev**2) * (0.94 + 0.06 * np.tanh((cfg.DO - 25.0) / 35.0)), 0.45, 1.0)),
        "DES": float(np.clip(np.exp(-0.016 * temp_dev**2) * np.exp(-0.055 * ph_dev**2) * (0.88 + 0.12 * np.exp(-do_dev**2)), 0.40, 1.0)),
        "CYC": float(np.clip(np.exp(-0.020 * temp_dev**2) * np.exp(-0.120 * ph_dev**2) * (0.90 + 0.10 * np.tanh((cfg.DO - 35.0) / 45.0)), 0.38, 1.0)),
    }


def initial_enzyme_capacities(cfg: GEMCultureConfig) -> np.ndarray:
    mode = cfg.capacity_mode
    if mode not in CAPACITY_MODES:
        raise GEMPilotError(f"Unknown capacity mode {mode!r}.")
    if mode == "no_dynamic_capacity":
        return np.ones(3, dtype=float)
    if mode == "constant_pathway_caps":
        vals = np.array([0.78, 0.74, 0.70], dtype=float)
    else:
        env = capacity_env_factors(cfg)
        vals = np.array(
            [
                cfg.psy_initial_base * env["PSY"],
                cfg.des_initial_base * env["DES"],
                cfg.cyc_initial_base * env["CYC"],
            ],
            dtype=float,
        )
    return np.clip(vals, cfg.enzyme_min_capacity, 1.0)


def base_pathway_capacities(cfg: GEMCultureConfig) -> dict[str, float]:
    return {
        "PSY_base_capacity": cfg.psy_base_capacity,
        "DES_base_capacity": cfg.des_base_capacity,
        "CYC_base_capacity": cfg.cyc_base_capacity,
    }


def recovery_multiplier(state: str) -> dict[str, float]:
    table = {
        "balanced": (1.00, 1.00, 1.00),
        "productive": (1.05, 1.05, 1.05),
        "oxidative_stress": (0.55, 0.40, 0.35),
        "energy_limited": (0.62, 0.52, 0.48),
        "pathway_limited": (0.78, 0.58, 0.50),
        "combined_overload": (0.32, 0.25, 0.22),
        "recovering": (1.45, 1.60, 1.70),
    }
    psy, des, cyc = table[state]
    return {"PSY": psy, "DES": des, "CYC": cyc}


def apply_enzyme_capacity_bounds(model, cfg: GEMCultureConfig, pathway_ub: float, capacities: np.ndarray | None) -> dict[str, float | str]:
    bases = base_pathway_capacities(cfg)
    mode = cfg.capacity_mode
    if mode == "no_dynamic_capacity" or capacities is None:
        effective = {
            "PSY_effective_upper_bound": pathway_ub,
            "DES_effective_upper_bound": pathway_ub,
            "CYC_effective_upper_bound": pathway_ub,
        }
    else:
        e_psy, e_des, e_cyc = [float(x) for x in capacities]
        effective = {
            "PSY_effective_upper_bound": min(pathway_ub, bases["PSY_base_capacity"] * e_psy),
            "DES_effective_upper_bound": min(pathway_ub, bases["DES_base_capacity"] * e_des),
            "CYC_effective_upper_bound": min(pathway_ub, bases["CYC_base_capacity"] * e_cyc),
        }
    model.reactions.get_by_id(PSY_RXN).upper_bound = float(effective["PSY_effective_upper_bound"])
    model.reactions.get_by_id(DES_RXN).upper_bound = float(effective["DES_effective_upper_bound"])
    model.reactions.get_by_id(CYC_RXN).upper_bound = float(effective["CYC_effective_upper_bound"])
    return {**bases, **effective, "capacity_mode": mode}


def apply_interval_constraints(
    model,
    cfg: GEMCultureConfig,
    burdens: np.ndarray,
    state: str,
    enzyme_capacities: np.ndarray | None = None,
    s_glc: float | None = None,
    x_biomass: float | None = None,
) -> dict[str, float | str]:
    z_ox, z_atp, z_bottle, z_er = [float(x) for x in burdens]
    if cfg.substrate_limited and s_glc is not None and x_biomass is not None:
        # Extracellular glucose mass balance: the realizable uptake bound cannot
        # exceed what remains in the pool this interval, S_glc(t) / (dt * X(t)).
        # See GENERATOR_VALIDITY.md Section 2.9 for units and derivation.
        glucose_lb = -min(cfg.glucose_uptake, max(0.0, s_glc) / max(cfg.dt * x_biomass, EPS))
    else:
        glucose_lb = -cfg.glucose_uptake
    oxygen_capacity = max(0.25, cfg.oxygen_uptake_at_100_do * cfg.DO / 100.0)
    oxygen_capacity *= max(0.20, 1.0 - 0.45 * z_ox)
    atpm = cfg.baseline_atpm * env_factor(cfg.temperature, cfg.pH, cfg) * (1.0 + 1.35 * z_atp)
    if state in {"energy_limited", "combined_overload"}:
        atpm *= 1.25
    pathway_ub = 1000.0 * state_pathway_cap(state, cfg) * max(0.05, 1.0 - 0.62 * z_bottle - 0.18 * z_ox)
    if state in {"pathway_limited", "combined_overload"}:
        pathway_ub *= 0.55

    model.reactions.get_by_id(GLUCOSE_EXCHANGE).lower_bound = glucose_lb
    model.reactions.get_by_id(OXYGEN_EXCHANGE).lower_bound = -oxygen_capacity
    atpm_rxn = model.reactions.get_by_id(ATPM_RXN)
    atpm_rxn.upper_bound = max(atpm, atpm_rxn.upper_bound)
    atpm_rxn.lower_bound = atpm
    capacity_bounds = apply_enzyme_capacity_bounds(model, cfg, pathway_ub, enzyme_capacities)
    model.reactions.get_by_id(PRODUCT_RXN).upper_bound = 1000.0
    return {
        "glucose_exchange": GLUCOSE_EXCHANGE,
        "glucose_lower_bound": glucose_lb,
        "substrate_limited": bool(cfg.substrate_limited),
        "s_glc_available": float(s_glc) if s_glc is not None else float("nan"),
        "oxygen_exchange": OXYGEN_EXCHANGE,
        "oxygen_lower_bound": -oxygen_capacity,
        "atp_maintenance_reaction": ATPM_RXN,
        "atp_maintenance_lower_bound": atpm,
        "pathway_capacity_upper_bound": pathway_ub,
        **capacity_bounds,
        "state": state,
        "z_ox": z_ox,
        "z_atp": z_atp,
        "z_bottle": z_bottle,
        "z_er": z_er,
    }


def solve_staged(model, cfg: GEMCultureConfig, gamma: float, state: str = "balanced") -> dict[str, object]:
    cobra = require_cobra()
    statuses: list[str] = []
    tic = time.perf_counter()
    n_growth = n_product = n_pfba = 0

    set_objective(model, BIOMASS_RXN)
    growth_sol = model.optimize()
    n_growth += 1
    statuses.append(str(growth_sol.status))
    if growth_sol.status != "optimal" or growth_sol.objective_value is None:
        raise GEMPilotError(f"Growth optimization failed: {growth_sol.status}")
    mu_max = float(growth_sol.objective_value)
    preserved_growth = gamma * mu_max
    model.reactions.get_by_id(BIOMASS_RXN).lower_bound = preserved_growth

    protocol = cfg.allocation_protocol.upper()
    if protocol == "A":
        product_target_fraction = 0.999
        set_objective(model, PRODUCT_RXN)
    elif protocol == "B":
        # Fixed heterologous-capacity protocol: product is required at a modest
        # state-dependent fraction of current pathway capacity, then pFBA/growth
        # determines the rest instead of pushing beta-carotene to its envelope.
        cap = model.reactions.get_by_id(PSY_RXN).upper_bound
        state_fraction = {
            "balanced": 0.00004,
            "productive": 0.00006,
            "oxidative_stress": 0.000025,
            "energy_limited": 0.00003,
            "pathway_limited": 0.000015,
            "combined_overload": 0.00001,
            "recovering": 0.00002,
        }[state]
        model.reactions.get_by_id(PRODUCT_RXN).lower_bound = min(0.20, cap * state_fraction)
        set_objective(model, BIOMASS_RXN)
        product_target_fraction = 1.0
    elif protocol == "C":
        # Weighted allocation protocol implemented as a softer lexicographic
        # product target after preserving growth. Weights are predeclared by
        # state and intentionally lower than Protocol A.
        set_objective(model, PRODUCT_RXN)
        product_target_fraction = {
            "balanced": 0.45,
            "productive": 0.60,
            "oxidative_stress": 0.25,
            "energy_limited": 0.20,
            "pathway_limited": 0.15,
            "combined_overload": 0.10,
            "recovering": 0.25,
        }[state]
    else:
        raise GEMPilotError(f"Unknown allocation protocol {cfg.allocation_protocol!r}.")
    product_sol = model.optimize()
    n_product += 1
    statuses.append(str(product_sol.status))
    if product_sol.status != "optimal" or product_sol.objective_value is None:
        raise GEMPilotError(f"Product optimization failed: {product_sol.status}")
    beta_flux = max(0.0, float(product_sol.fluxes.get(PRODUCT_RXN, product_sol.objective_value)))
    model.reactions.get_by_id(PRODUCT_RXN).lower_bound = product_target_fraction * beta_flux if beta_flux > 1e-12 else 0.0

    pfba_sol = cobra.flux_analysis.pfba(model)
    n_pfba += 1
    statuses.append(str(pfba_sol.status))
    if pfba_sol.status != "optimal":
        raise GEMPilotError(f"pFBA optimization failed: {pfba_sol.status}")
    runtime = time.perf_counter() - tic
    fluxes = pfba_sol.fluxes
    return {
        "growth_status": growth_sol.status,
        "product_status": product_sol.status,
        "pfba_status": pfba_sol.status,
        "maximum_growth": mu_max,
        "preserved_growth": preserved_growth,
        "beta_carotene_flux": float(fluxes.get(PRODUCT_RXN, beta_flux)),
        "biomass_flux": float(fluxes.get(BIOMASS_RXN, np.nan)),
        "glucose_uptake": float(fluxes.get(GLUCOSE_EXCHANGE, np.nan)),
        "oxygen_uptake": float(fluxes.get(OXYGEN_EXCHANGE, np.nan)),
        "atp_maintenance_flux": float(fluxes.get(ATPM_RXN, np.nan)),
        "ggpp_flux": float(fluxes.get(NATIVE_GGPP_RXN, np.nan)),
        "PSY_flux": float(fluxes.get(PSY_RXN, np.nan)),
        "DES_flux": float(fluxes.get(DES_RXN, np.nan)),
        "CYC_flux": float(fluxes.get(CYC_RXN, np.nan)),
        "phytoene_flux": float(fluxes.get(PSY_RXN, np.nan)),
        "desaturase_flux": float(fluxes.get(DES_RXN, np.nan)),
        "cyclase_flux": float(fluxes.get(CYC_RXN, np.nan)),
        "lycopene_flux": float(fluxes.get(DES_RXN, np.nan)),
        "allocation_protocol": protocol,
        "product_target_fraction": product_target_fraction,
        "n_growth_optimizations": n_growth,
        "n_product_optimizations": n_product,
        "n_pfba_optimizations": n_pfba,
        "n_actual_lp_solves": n_growth + n_product + n_pfba,
        "n_optimal_solves": sum(s == "optimal" for s in statuses),
        "n_infeasible_solves": sum(s == "infeasible" for s in statuses),
        "n_unbounded_solves": sum(s == "unbounded" for s in statuses),
        "optimization_statuses": ";".join(statuses),
        "solver_runtime_seconds": runtime,
    }


def next_state(current: str, burdens: np.ndarray, dwell: int) -> str:
    z_ox, z_atp, z_bottle = burdens[:3]
    high = [z_ox > 0.62, z_atp > 0.58, z_bottle > 0.60]
    low = [z_ox < 0.42, z_atp < 0.38, z_bottle < 0.40]
    if dwell < 2 and current not in {"balanced", "productive"}:
        return current
    if sum(high) >= 2:
        return "combined_overload"
    if high[0]:
        return "oxidative_stress"
    if high[1]:
        return "energy_limited"
    if high[2]:
        return "pathway_limited"
    if current in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload"} and not all(low):
        return "recovering"
    if all(low):
        return "productive"
    return "balanced"


def pathway_congestion_terms(flux: dict[str, object], constraints: dict[str, object] | pd.Series | None = None) -> dict[str, float]:
    psy = max(0.0, float(flux.get("phytoene_flux", flux.get(PSY_RXN, 0.0))))
    des = max(0.0, float(flux.get("desaturase_flux", flux.get("lycopene_flux", flux.get(DES_RXN, 0.0)))))
    cyc = max(0.0, float(flux.get("cyclase_flux", flux.get(CYC_RXN, flux.get("beta_carotene_flux", 0.0)))))
    phytoene_flux_mismatch = max(0.0, psy - des)
    lycopene_flux_mismatch = max(0.0, des - cyc)
    phytoene_capacity_pressure = 0.0
    lycopene_capacity_pressure = 0.0
    if constraints is not None:
        get = constraints.get if hasattr(constraints, "get") else lambda key, default=None: default
        psy_ub = float(get("PSY_effective_upper_bound", 0.0) or 0.0)
        des_ub = float(get("DES_effective_upper_bound", 0.0) or 0.0)
        cyc_ub = float(get("CYC_effective_upper_bound", 0.0) or 0.0)
        psy_sat = psy / max(psy_ub, EPS) if psy_ub < 999.0 else 0.0
        des_sat = des / max(des_ub, EPS) if des_ub < 999.0 else 0.0
        phytoene_capacity_pressure = max(0.0, psy_ub - des_ub) * np.clip(psy_sat, 0.0, 1.0)
        lycopene_capacity_pressure = max(0.0, des_ub - cyc_ub) * np.clip(des_sat, 0.0, 1.0)
    return {
        "phytoene_congestion_flux_mismatch": float(phytoene_flux_mismatch),
        "lycopene_congestion_flux_mismatch": float(lycopene_flux_mismatch),
        "phytoene_capacity_pressure": float(phytoene_capacity_pressure),
        "lycopene_capacity_pressure": float(lycopene_capacity_pressure),
        "phytoene_congestion": float(phytoene_flux_mismatch + phytoene_capacity_pressure),
        "lycopene_congestion": float(lycopene_flux_mismatch + lycopene_capacity_pressure),
    }


def burden_terms(cfg: GEMCultureConfig, burdens: np.ndarray, flux: dict[str, object], state: str, congestion: dict[str, float] | None = None) -> dict[str, float]:
    z_ox, z_atp, z_bottle, z_er = [float(x) for x in burdens]
    oxygen_capacity = max(0.25, cfg.oxygen_uptake_at_100_do * cfg.DO / 100.0)
    oxygen_use = max(0.0, -float(flux["oxygen_uptake"]))
    oxygen_stress = max(0.0, oxygen_use / max(oxygen_capacity, EPS))
    beta = max(0.0, float(flux["beta_carotene_flux"]))
    atpm = max(0.0, float(flux["atp_maintenance_flux"]))
    growth = max(0.0, float(flux["biomass_flux"]))
    ggpp = max(0.0, float(flux["ggpp_flux"]))
    recovery = 2.8 if state in {"oxidative_stress", "energy_limited", "pathway_limited", "combined_overload", "recovering"} else 1.0
    oxidative_generation = 0.16 * oxygen_stress + cfg.oxidative_beta_coupling * beta
    oxidative_repair = 0.36 * recovery * z_ox
    atp_demand = 0.14 * atpm + 0.11 * growth + 0.12 * beta
    atp_supply = 0.05 * oxygen_use / max(oxygen_capacity, EPS)
    atp_repair = 0.38 * recovery * z_atp
    congestion = congestion or {}
    phytoene_congestion = float(congestion.get("phytoene_congestion", 0.0))
    lycopene_congestion = float(congestion.get("lycopene_congestion", 0.0))
    ggpp_loading = 0.85 * ggpp + 2.40 * phytoene_congestion + 3.00 * lycopene_congestion
    beta_clearance = 1.45 * beta
    bottleneck_repair = 0.28 * recovery * z_bottle
    er_generation = 0.08 * abs(cfg.temperature - cfg.t_opt) / 8.0 + 0.07 * abs(cfg.pH - cfg.ph_opt)
    er_repair = 0.30 * z_er
    after_ox = float(np.clip(z_ox + cfg.dt * (oxidative_generation - oxidative_repair), 0.0, 1.5))
    after_atp = float(np.clip(z_atp + cfg.dt * (atp_demand - atp_supply - atp_repair), 0.0, 1.5))
    after_bottle = float(np.clip(z_bottle + cfg.dt * (ggpp_loading - beta_clearance - bottleneck_repair), 0.0, 1.5))
    after_er = float(np.clip(z_er + cfg.dt * (er_generation - er_repair), 0.0, 1.5))
    return {
        "oxidative_burden_before": z_ox,
        "oxidative_generation_term": oxidative_generation,
        "oxidative_repair_term": oxidative_repair,
        "oxidative_burden_after": after_ox,
        "ATP_pressure_before": z_atp,
        "ATP_demand_term": atp_demand,
        "ATP_supply_term": atp_supply,
        "ATP_repair_term": atp_repair,
        "ATP_pressure_after": after_atp,
        "bottleneck_before": z_bottle,
        "GGPP_loading_term": ggpp_loading,
        "beta_carotene_clearance_term": beta_clearance,
        "phytoene_congestion_term": phytoene_congestion,
        "lycopene_congestion_term": lycopene_congestion,
        "bottleneck_repair_term": bottleneck_repair,
        "bottleneck_after": after_bottle,
        "ER_before": z_er,
        "ER_generation_term": er_generation,
        "ER_repair_term": er_repair,
        "ER_after": after_er,
        "oxygen_uptake_used": oxygen_use,
        "oxygen_capacity_used": oxygen_capacity,
        "ATP_maintenance_flux_used": atpm,
        "biomass_flux_used": growth,
        "beta_carotene_flux_used": beta,
        "GGPP_flux_used": ggpp,
    }


def update_burdens(cfg: GEMCultureConfig, burdens: np.ndarray, flux: dict[str, object], state: str) -> np.ndarray:
    terms = burden_terms(cfg, burdens, flux, state)
    return np.array(
        [
            terms["oxidative_burden_after"],
            terms["ATP_pressure_after"],
            terms["bottleneck_after"],
            terms["ER_after"],
        ],
        dtype=float,
    )


def capacity_update_terms(
    cfg: GEMCultureConfig,
    capacities: np.ndarray,
    burdens_after: np.ndarray,
    flux: dict[str, object],
    state: str,
    congestion: dict[str, float] | None = None,
) -> dict[str, float]:
    e_psy, e_des, e_cyc = [float(x) for x in capacities]
    z_ox, z_atp, z_bottle = [float(x) for x in burdens_after[:3]]
    congestion = congestion or {}
    mult = recovery_multiplier(state)
    env = capacity_env_factors(cfg)
    synth = {
        "PSY": cfg.psy_synthesis_base * env["PSY"] * mult["PSY"] * (1.0 - e_psy),
        "DES": cfg.des_synthesis_base * env["DES"] * mult["DES"] * (1.0 - e_des),
        "CYC": cfg.cyc_synthesis_base * env["CYC"] * mult["CYC"] * (1.0 - e_cyc),
    }
    phy = float(congestion.get("phytoene_congestion", 0.0)) if cfg.capacity_mode == "dynamic_congestion_feedback" else 0.0
    lyc = float(congestion.get("lycopene_congestion", 0.0)) if cfg.capacity_mode == "dynamic_congestion_feedback" else 0.0
    damage = {
        "PSY": cfg.psy_decay_base + cfg.psy_oxidative_sensitivity * z_ox + cfg.psy_atp_sensitivity * z_atp + cfg.psy_bottleneck_sensitivity * z_bottle + cfg.phytoene_congestion_sensitivity * phy,
        "DES": cfg.des_decay_base + cfg.des_oxidative_sensitivity * z_ox + cfg.des_atp_sensitivity * z_atp + cfg.des_bottleneck_sensitivity * z_bottle + 0.35 * cfg.phytoene_congestion_sensitivity * phy + cfg.lycopene_congestion_sensitivity * lyc,
        "CYC": cfg.cyc_decay_base + cfg.cyc_oxidative_sensitivity * z_ox + cfg.cyc_atp_sensitivity * z_atp + cfg.cyc_bottleneck_sensitivity * z_bottle + 0.40 * cfg.lycopene_congestion_sensitivity * lyc,
    }
    after = {
        "PSY": float(np.clip(e_psy + cfg.dt * (synth["PSY"] - damage["PSY"] * e_psy), cfg.enzyme_min_capacity, 1.0)),
        "DES": float(np.clip(e_des + cfg.dt * (synth["DES"] - damage["DES"] * e_des), cfg.enzyme_min_capacity, 1.0)),
        "CYC": float(np.clip(e_cyc + cfg.dt * (synth["CYC"] - damage["CYC"] * e_cyc), cfg.enzyme_min_capacity, 1.0)),
    }
    return {
        "PSY_synthesis_term": synth["PSY"],
        "DES_synthesis_term": synth["DES"],
        "CYC_synthesis_term": synth["CYC"],
        "PSY_damage_rate": damage["PSY"],
        "DES_damage_rate": damage["DES"],
        "CYC_damage_rate": damage["CYC"],
        "E_PSY_after": after["PSY"],
        "E_DES_after": after["DES"],
        "E_CYC_after": after["CYC"],
    }


def update_enzyme_capacities(
    cfg: GEMCultureConfig,
    capacities: np.ndarray,
    burdens_after: np.ndarray,
    flux: dict[str, object],
    state: str,
    congestion: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    mode = cfg.capacity_mode
    if mode in {"no_dynamic_capacity", "constant_pathway_caps", "environment_initial_capacity"}:
        terms = {
            "PSY_synthesis_term": 0.0,
            "DES_synthesis_term": 0.0,
            "CYC_synthesis_term": 0.0,
            "PSY_damage_rate": 0.0,
            "DES_damage_rate": 0.0,
            "CYC_damage_rate": 0.0,
            "E_PSY_after": float(capacities[0]),
            "E_DES_after": float(capacities[1]),
            "E_CYC_after": float(capacities[2]),
        }
        return capacities.copy(), terms
    terms = capacity_update_terms(cfg, capacities, burdens_after, flux, state, congestion)
    return np.array([terms["E_PSY_after"], terms["E_DES_after"], terms["E_CYC_after"]], dtype=float), terms


def run_dynamic_culture(base_augmented_model, source_path: Path, cfg: GEMCultureConfig, culture_id: str = "gem_pilot_central") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = np.arange(cfg.n_time, dtype=float) * cfg.dt
    x = np.zeros(cfg.n_time)
    b = np.zeros(cfg.n_time)
    s_glc = np.zeros(cfg.n_time)
    burdens = np.zeros((cfg.n_time, 4))
    enzyme_capacities = np.zeros((cfg.n_time, 3))
    x[0] = cfg.x0
    b[0] = cfg.b0
    s_glc[0] = cfg.s_glc0
    enzyme_capacities[0] = initial_enzyme_capacities(cfg)
    state = "balanced"
    dwell = 0
    traj_rows, flux_rows, constraint_rows = [], [], []
    totals = {
        "n_growth_optimizations": 0,
        "n_product_optimizations": 0,
        "n_pfba_optimizations": 0,
        "n_actual_lp_solves": 0,
        "n_optimal_solves": 0,
        "n_infeasible_solves": 0,
        "n_unbounded_solves": 0,
    }
    checksum = sha256(source_path)
    for i in range(cfg.n_time - 1):
        interval_model = base_augmented_model.copy()
        constraints = apply_interval_constraints(interval_model, cfg, burdens[i], state, enzyme_capacities[i], s_glc=s_glc[i], x_biomass=x[i])
        gamma = state_gamma(state, cfg)
        flux = solve_staged(interval_model, cfg, gamma, state=state)
        for key in totals:
            totals[key] += int(flux[key])
        congestion = pathway_congestion_terms(flux, constraints)
        beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * burdens[i, 0])
        dx = float(flux["biomass_flux"]) * x[i]
        db = float(flux["beta_carotene_flux"]) * x[i] - beta_deg * b[i]
        glucose_uptake_magnitude = max(0.0, -float(flux["glucose_uptake"]))
        x[i + 1] = max(1e-9, x[i] + cfg.dt * dx)
        b[i + 1] = max(0.0, b[i] + cfg.dt * db)
        s_glc[i + 1] = max(0.0, s_glc[i] - cfg.dt * glucose_uptake_magnitude * x[i])
        terms = burden_terms(cfg, burdens[i], flux, state, congestion if cfg.capacity_mode == "dynamic_congestion_feedback" else None)
        burdens[i + 1] = np.array(
            [
                terms["oxidative_burden_after"],
                terms["ATP_pressure_after"],
                terms["bottleneck_after"],
                terms["ER_after"],
            ]
        )
        enzyme_capacities[i + 1], capacity_terms = update_enzyme_capacities(cfg, enzyme_capacities[i], burdens[i + 1], flux, state, congestion)
        new_state = next_state(state, burdens[i + 1], dwell)
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
        }
        flux_rows.append({**common, **flux, **congestion})
        constraint_rows.append({**common, **constraints, "gamma_growth_fraction": gamma})
        constraint_rows[-1].update(terms)
        constraint_rows[-1].update(congestion)
        constraint_rows[-1].update(capacity_terms)

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
                "capacity_mode": cfg.capacity_mode,
                "metabolic_backend": "yeast_gem_lp",
                "gem_model_id": base_augmented_model.id,
                "gem_source_checksum": checksum,
                "n_time_points": cfg.n_time,
                "n_simulation_intervals": cfg.n_time - 1,
                **totals,
                "n_surrogate_evaluations": 0,
            }
        )
    return pd.DataFrame(traj_rows), pd.DataFrame(flux_rows), pd.DataFrame(constraint_rows), pd.DataFrame([{"culture_id": culture_id, **totals, "n_surrogate_evaluations": 0}])


def er_isolation_check(base_augmented_model, cfg: GEMCultureConfig) -> dict[str, object]:
    b0 = np.array([0.35, 0.25, 0.20, 0.0])
    b1 = np.array([0.35, 0.25, 0.20, 1.2])
    outputs = []
    for burdens in (b0, b1):
        model = base_augmented_model.copy()
        apply_interval_constraints(model, cfg, burdens, "productive")
        outputs.append(solve_staged(model, cfg, state_gamma("productive", cfg)))
    causal_keys = ["maximum_growth", "preserved_growth", "beta_carotene_flux", "biomass_flux", "glucose_uptake", "oxygen_uptake", "atp_maintenance_flux", "ggpp_flux"]
    max_abs = max(abs(float(outputs[0][k]) - float(outputs[1][k])) for k in causal_keys)
    return {"er_isolation_passed": bool(max_abs < 1e-9), "er_isolation_max_abs_causal_difference": max_abs}


def write_png_from_series(path: Path, width: int, height: int, series: list[tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], tuple[int, int, int]]]) -> None:
    pixels = bytearray([255, 255, 255] * width * height)

    def set_px(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def line(x1, y1, x2, y2, color):
        dx, dy = abs(x2 - x1), -abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err = dx + dy
        while True:
            set_px(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    for x, y, xlim, ylim, color in series:
        pts = []
        for xi, yi in zip(x, y):
            px = int(55 + (xi - xlim[0]) / max(xlim[1] - xlim[0], EPS) * (width - 95))
            py = int(height - 35 - (yi - ylim[0]) / max(ylim[1] - ylim[0], EPS) * (height - 70))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            line(a[0], a[1], b[0], b[1], color)
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def svg_text(x, y, text, size=11, weight="normal", anchor="middle"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{text}</text>'


def svg_polyline(x, y, x0, y0, w, h, xlim, ylim, color, width=1.8):
    pts = []
    for xi, yi in zip(x, y):
        px = x0 + (xi - xlim[0]) / max(xlim[1] - xlim[0], EPS) * w
        py = y0 + h - (yi - ylim[0]) / max(ylim[1] - ylim[0], EPS) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"/>'
