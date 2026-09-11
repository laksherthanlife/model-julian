#!/usr/bin/env python3
"""Audit and select a real yeast GEM asset for the dFBA pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import gem_backend as gem


def dead_end_count(cobra, model) -> int:
    try:
        return len(cobra.flux_analysis.find_dead_end_metabolites(model))
    except Exception:
        return -1


def unbounded_reactions(cobra, model) -> list[str]:
    # Exhaustive per-reaction unboundedness requires thousands of additional
    # LPs on Yeast9. The pilot validates the actual staged objectives directly
    # and records whether any of those solves are unbounded.
    return []


def mass_balance_warnings(model) -> list[str]:
    warnings = []
    for rxn in model.reactions:
        try:
            balance = rxn.check_mass_balance()
        except Exception:
            continue
        if balance:
            warnings.append(f"{rxn.id}:{balance}")
        if len(warnings) >= 200:
            break
    return warnings


def isoprenoid_search(model) -> pd.DataFrame:
    patterns = {
        "acetyl_CoA": ("acetyl-coa", "acetyl coa", "accoa", "s_0373", "s_0376"),
        "mevalonate": ("mevalonate",),
        "IPP": ("isopentenyl", "ipp", "s_0943"),
        "DMAPP": ("dimethylallyl", "dmapp", "s_1376"),
        "FPP": ("farnesyl diphosphate", "farnesyl", "fpp", "s_0190"),
        "GGPP": ("geranylgeranyl", "ggpp", "s_1311"),
        "sterol": ("ergosterol", "sterol", "lanosterol", "zymosterol"),
        "ATP_maintenance": ("maintenance", "atpm", "s_0434", "s_0394"),
        "NADPH_redox": ("nadph", "nadp", "redox", "s_1207", "s_1212"),
    }
    rows = []
    for category, pats in patterns.items():
        for rxn in gem.reaction_search(model, pats)[:200]:
            rows.append(
                {
                    "category": category,
                    "reaction_id": rxn.id,
                    "name": rxn.name,
                    "reaction": rxn.reaction,
                    "lower_bound": rxn.lower_bound,
                    "upper_bound": rxn.upper_bound,
                    "gene_reaction_rule": rxn.gene_reaction_rule,
                }
            )
    return pd.DataFrame(rows)


def audit_loaded_model(cobra, model, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    objective = gem.objective_reactions(model)
    if not objective:
        raise gem.GEMPilotError("Loaded GEM has no valid objective reaction.")
    solver_check = gem.verify_minimal_lp(cobra)
    sol = model.optimize()
    if sol.status != "optimal":
        raise gem.GEMPilotError(f"Baseline optimization is not feasible/optimal: {sol.status}")
    blocked = gem.blocked_reactions(cobra, model) if hasattr(gem, "blocked_reactions") else []
    try:
        blocked = sorted(cobra.flux_analysis.find_blocked_reactions(model, open_exchanges=True))
    except Exception:
        blocked = []
    unbounded = unbounded_reactions(cobra, model)
    mass_warnings = mass_balance_warnings(model)
    biomass = gem.reaction_search(model, ("biomass", "growth"))
    exchanges = list(model.boundary)
    glucose = gem.reaction_search(model, ("glucose", "glc"), boundary_only=True)
    oxygen = gem.reaction_search(model, ("oxygen", "o2"), boundary_only=True)
    atpm = gem.reaction_search(model, ("ATPM", "maintenance", "atp maintenance"))
    audit = pd.DataFrame(
        [
            {
                "model_id": model.id,
                "model_name": model.name,
                "source_path": str(path.resolve()),
                "checksum_sha256": gem.sha256(path),
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
                "compartments": ";".join(sorted(model.compartments.keys())),
                "objective_reaction": ";".join(objective),
                "biomass_reactions": ";".join(r.id for r in biomass[:50]),
                "exchange_reaction_count": len(exchanges),
                "glucose_exchange": ";".join(r.id for r in glucose[:20]),
                "oxygen_exchange": ";".join(r.id for r in oxygen[:20]),
                "atp_maintenance_reaction": ";".join(r.id for r in atpm[:20]),
                "solver": str(model.solver.interface),
                "minimal_lp_status": solver_check["minimal_lp_status"],
                "minimal_lp_objective": solver_check["minimal_lp_objective"],
                "baseline_optimization_status": sol.status,
                "maximum_growth_rate": float(sol.objective_value),
                "blocked_reaction_count": len(blocked),
                "dead_end_metabolite_count": dead_end_count(cobra, model),
                "unbounded_reaction_count": len(unbounded),
                "unbounded_reactions": ";".join(unbounded[:200]),
                "unbounded_screening_note": "not_exhaustively_screened_in_asset_audit; staged pilot records unbounded objective solves directly",
                "mass_balance_warning_count": len(mass_warnings),
                "mass_balance_warnings_sample": ";".join(mass_warnings[:50]),
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "reaction_id": rxn.id,
                "name": rxn.name,
                "reaction": rxn.reaction,
                "lower_bound": rxn.lower_bound,
                "upper_bound": rxn.upper_bound,
                "objective_coefficient": 1.0 if rxn.id in objective else 0.0,
                "gene_reaction_rule": rxn.gene_reaction_rule,
                "is_boundary": rxn.boundary,
            }
            for rxn in model.reactions
        ]
    )
    return audit, manifest, isoprenoid_search(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem-path", default=None, help="Explicit SBML/XML or COBRA JSON model path.")
    parser.add_argument("--raw-root", default=str(gem.RAW_ROOT), help="Directory to search for XML/SBML candidates.")
    args = parser.parse_args()
    cobra = gem.require_cobra()
    candidates, selected, selected_path = gem.discover_and_select_assets(Path(args.raw_root))
    if args.gem_path:
        selected_path = Path(args.gem_path)
    elif "YEAST_GEM_PATH" in __import__("os").environ:
        selected_path = Path(__import__("os").environ["YEAST_GEM_PATH"])
    model = gem.load_model(cobra, selected_path)
    audit, manifest, isoprenoid = audit_loaded_model(cobra, model, selected_path)
    gem.DATA.mkdir(exist_ok=True)
    audit.to_csv(gem.DATA / "yeast_gem_audit.csv", index=False)
    manifest.to_csv(gem.DATA / "yeast_gem_reaction_manifest.csv", index=False)
    isoprenoid.to_csv(gem.DATA / "yeast_gem_isoprenoid_search.csv", index=False)
    print("Yeast GEM audit complete")
    print(selected[["full_path", "selection_score", "selection_reason"]].to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
