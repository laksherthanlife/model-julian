#!/usr/bin/env python3
"""Hidden-administrator request/response verifier interface.

This command validates public scientist requests, rejects invalid candidates
before any exact work, and only reveals sanitised public observations.  Fresh
open-ended exact execution is gated by ``--execute-fresh`` and is not invoked by
the public bundle preparation task.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np
import pandas as pd

import deployment_benchmark as deployment
import design_benchmark_exact as exact
import gem_backend as gem


ALLOWED_PUBLIC_RESULT_COLUMNS = [
    "candidate_id",
    "candidate_hash",
    "feasible",
    "product_trajectory",
    "final_product",
    "product_AUC",
    "biomass_trajectory",
    "final_biomass",
    "growth_failure",
    "requested_public_assays",
    "sanitised_failure_category",
    "physical_resource_accounting",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def operational_accounting(candidates: list[dict[str, Any]], scenario_id: str = "central_lab") -> dict[str, float | int | str]:
    scenarios = pd.read_csv(deployment.DATA / "deployment_benchmark_operational_scenarios.csv")
    scenario = scenarios[scenarios["scenario_id"].eq(scenario_id)].iloc[0]
    strains = len({str(c.get("strain_id", c.get("candidate_id", ""))) for c in candidates})
    cultures = len(candidates)
    assay_points = sum(float(scenario["resource_points_basic_panel"]) for _ in candidates)
    resource = strains * float(scenario["resource_points_per_strain"]) + cultures * float(scenario["resource_points_per_culture"]) + assay_points
    return {
        "scenario_id": scenario_id,
        "unique_strains": strains,
        "cultures": cultures,
        "physical_rounds": 1,
        "calendar_days": float(scenario["strain_construction_batch_lead_days"]) + float(scenario["culture_duration_days"]) + float(scenario["assay_duration_days"]),
        "scientist_hours": float(scenario["scientist_planning_hours_per_round"]) + float(scenario["analysis_hours_per_round"]),
        "technician_hours": strains * float(scenario["technician_setup_hours_per_strain"]),
        "assay_usage": cultures,
        "instrument_days": float(scenario["culture_duration_days"]) + float(scenario["assay_duration_days"]),
        "normalised_resource_points": resource,
    }


def sanitise_cache_result(candidate: dict[str, Any], cache_row: pd.Series) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id", cache_row.get("candidate_id", ""))),
        "candidate_hash": str(cache_row.get("candidate_hash", "")),
        "feasible": bool(cache_row.get("feasible", True)),
        "product_trajectory": [],
        "final_product": float(cache_row.get("final_product", 0.0)),
        "product_AUC": float(cache_row.get("product_AUC", 0.0)),
        "biomass_trajectory": [],
        "final_biomass": float(cache_row.get("final_biomass", 0.0)),
        "growth_failure": bool(cache_row.get("severe_growth_collapse", False)),
        "requested_public_assays": str(candidate.get("requested_assay_panel", "basic")),
        "sanitised_failure_category": "none" if bool(cache_row.get("feasible", True)) else "infeasible_or_growth_failure",
        "physical_resource_accounting": {},
    }


def hidden_output_paths(round_id: str, candidate_id: str) -> dict[str, Path]:
    root = deployment.VERIFIER_RESULTS / "open_ended_exact" / round_id
    root.mkdir(parents=True, exist_ok=True)
    return {
        "trajectory": root / f"{candidate_id}_hidden_trajectory.csv",
        "flux": root / f"{candidate_id}_hidden_flux.csv",
        "constraints": root / f"{candidate_id}_hidden_constraints.csv",
        "summary": root / f"{candidate_id}_hidden_summary.csv",
    }


def deployment_exact_cache() -> pd.DataFrame:
    """Return exact-result rows reusable by full candidate-hash match only."""

    caches: list[pd.DataFrame] = []
    try:
        finite = deployment.read_exact_cache().copy()
        finite["exact_cache_source"] = "finite_pool_exact_cache"
        caches.append(finite)
    except Exception:
        pass
    root = deployment.VERIFIER_RESULTS / "open_ended_exact"
    if root.exists():
        summaries = sorted(root.glob("*/*_hidden_summary.csv"))
        rows = []
        for path in summaries:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "candidate_hash" not in df.columns or df.empty:
                continue
            df = df.copy()
            df["exact_cache_source"] = str(path.relative_to(deployment.ROOT))
            rows.append(df)
        if rows:
            caches.append(pd.concat(rows, ignore_index=True, sort=False))
    if not caches:
        return pd.DataFrame()
    out = pd.concat(caches, ignore_index=True, sort=False)
    return out.drop_duplicates("candidate_hash", keep="first")


def config_for_sparse_candidate(candidate: dict[str, Any], hidden_regime: str) -> gem.GEMCultureConfig:
    spec = exact.regime_configs()[hidden_regime]
    cfg = spec["cfg"]
    env = candidate.get("environment", {})
    cfg = replace(
        cfg,
        temperature=float(env.get("temperature", 30.0)),
        pH=float(env.get("pH", 5.0)),
        DO=float(env.get("DO", 50.0)),
        n_time=exact.TIME_POINTS,
    )
    for edit in candidate.get("edits", []):
        rid = str(edit.get("reaction_id", ""))
        etype = str(edit.get("edit_type", ""))
        mult = float(edit.get("capacity_multiplier", edit.get("upper_bound_multiplier", 1.0)))
        if rid == gem.PSY_RXN:
            cfg = replace(cfg, psy_base_capacity=float(cfg.psy_base_capacity) * mult, psy_initial_base=min(1.0, float(cfg.psy_initial_base) * np.sqrt(max(mult, 0.2))))
        elif rid == gem.DES_RXN:
            cfg = replace(cfg, des_base_capacity=float(cfg.des_base_capacity) * mult, des_initial_base=min(1.0, float(cfg.des_initial_base) * np.sqrt(max(mult, 0.2))))
        elif rid == gem.CYC_RXN:
            cfg = replace(cfg, cyc_base_capacity=float(cfg.cyc_base_capacity) * mult, cyc_initial_base=min(1.0, float(cfg.cyc_initial_base) * np.sqrt(max(mult, 0.2))))
        elif rid == "BETA_EXPORT_ASSIST":
            cfg = replace(cfg, beta_deg_base=float(cfg.beta_deg_base) / max(mult, 0.4))
        elif rid == gem.GLUCOSE_EXCHANGE or "glucose" in etype or "carbon_uptake" in etype:
            cfg = replace(cfg, glucose_uptake=float(cfg.glucose_uptake) * mult)
        elif rid == gem.OXYGEN_EXCHANGE or "oxygen" in etype:
            cfg = replace(cfg, oxygen_uptake_at_100_do=float(cfg.oxygen_uptake_at_100_do) * mult)
        elif rid == gem.ATPM_RXN and "demand" in etype:
            cfg = replace(cfg, baseline_atpm=float(cfg.baseline_atpm) * mult)
    return cfg


def exact_sparse_dynamic_rollout(candidate: dict[str, Any], hidden_regime: str = "strong_dynamic_regulation") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime_spec = exact.regime_configs()[hidden_regime]
    cfg = config_for_sparse_candidate(candidate, hidden_regime)
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path()
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    t = np.arange(cfg.n_time, dtype=float) * cfg.dt
    x = np.zeros(cfg.n_time)
    b = np.zeros(cfg.n_time)
    burdens = np.zeros((cfg.n_time, 4))
    enzyme_capacities = np.zeros((cfg.n_time, 3))
    x[0] = cfg.x0
    b[0] = cfg.b0
    enzyme_capacities[0] = gem.initial_enzyme_capacities(cfg)
    state = "balanced"
    dwell = 0
    totals = {k: 0 for k in ["n_growth_optimizations", "n_product_optimizations", "n_pfba_optimizations", "n_actual_lp_solves", "n_optimal_solves", "n_infeasible_solves", "n_unbounded_solves"]}
    traj_rows: list[dict[str, object]] = []
    flux_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    solver_errors = 0
    skipped = 0
    checksum = gem.sha256(source_path)
    candidate_id = str(candidate["candidate_id"])
    candidate_hash = str(candidate["candidate_hash"])
    env = candidate.get("environment", {})
    for i in range(cfg.n_time - 1):
        interval_model = augmented.copy()
        deployment.apply_sparse_edits_to_model(interval_model, candidate, strict=False)
        constraints = gem.apply_interval_constraints(interval_model, cfg, burdens[i], state, enzyme_capacities[i])
        gamma = gem.state_gamma(state, cfg)
        try:
            flux = gem.solve_staged(interval_model, cfg, gamma, state=state)
        except Exception:
            solver_errors += 1
            skipped += 1
            raise
        for key in totals:
            totals[key] += int(flux[key])
        congestion = gem.pathway_congestion_terms(flux, constraints)
        beta_deg = cfg.beta_deg_base * (1.0 + 2.0 * burdens[i, 0])
        dx = float(flux["biomass_flux"]) * x[i]
        db = float(flux["beta_carotene_flux"]) * x[i] - beta_deg * b[i]
        x[i + 1] = max(1e-9, x[i] + cfg.dt * dx)
        b[i + 1] = max(0.0, b[i] + cfg.dt * db)
        terms = exact.scaled_burden_terms(cfg, burdens[i], flux, state, regime_spec, congestion)
        burdens[i + 1] = np.array([terms["oxidative_burden_after"], terms["ATP_pressure_after"], terms["bottleneck_after"], terms["ER_after"]])
        cap_terms = gem.capacity_update_terms(cfg, enzyme_capacities[i], burdens[i + 1], flux, state, congestion)
        damage_scale = float(regime_spec["pathway_damage_scale"])
        e_psy, e_des, e_cyc = [float(v) for v in enzyme_capacities[i]]
        for enzyme in ["PSY", "DES", "CYC"]:
            cap_terms[f"{enzyme}_damage_rate"] *= damage_scale
        cap_terms["E_PSY_after"] = float(np.clip(e_psy + cfg.dt * (cap_terms["PSY_synthesis_term"] - cap_terms["PSY_damage_rate"] * e_psy), cfg.enzyme_min_capacity, 1.0))
        cap_terms["E_DES_after"] = float(np.clip(e_des + cfg.dt * (cap_terms["DES_synthesis_term"] - cap_terms["DES_damage_rate"] * e_des), cfg.enzyme_min_capacity, 1.0))
        cap_terms["E_CYC_after"] = float(np.clip(e_cyc + cfg.dt * (cap_terms["CYC_synthesis_term"] - cap_terms["CYC_damage_rate"] * e_cyc), cfg.enzyme_min_capacity, 1.0))
        enzyme_capacities[i + 1] = np.array([cap_terms["E_PSY_after"], cap_terms["E_DES_after"], cap_terms["E_CYC_after"]])
        new_state = gem.next_state(state, burdens[i + 1], dwell)
        if new_state == state:
            dwell += 1
        else:
            state = new_state
            dwell = 0
        common = {
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "hidden_regime_id": hidden_regime,
            "strain_id": candidate.get("strain_id", ""),
            "interval_index": i,
            "time": float(t[i]),
            "dt": cfg.dt,
            "temperature": float(env.get("temperature", cfg.temperature)),
            "pH": float(env.get("pH", cfg.pH)),
            "DO": float(env.get("DO", cfg.DO)),
            "metabolic_backend": exact.EXACT_BACKEND,
            "hidden_regulator": "gem_backend_decision_tree_next_state",
            "gem_source_checksum": checksum,
            "state": constraints["state"],
            "next_state": state,
            "z_ox": burdens[i, 0],
            "z_atp": burdens[i, 1],
            "z_bottle": burdens[i, 2],
            "z_er": burdens[i, 3],
            "E_PSY": enzyme_capacities[i, 0],
            "E_DES": enzyme_capacities[i, 1],
            "E_CYC": enzyme_capacities[i, 2],
        }
        flux_rows.append({**common, **flux, **congestion})
        c = {**common, **constraints, "gamma_growth_fraction": gamma}
        c.update(terms)
        c.update(congestion)
        c.update(cap_terms)
        constraint_rows.append(c)
        del interval_model
    for i, tt in enumerate(t):
        traj_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "hidden_regime_id": hidden_regime,
                "strain_id": candidate.get("strain_id", ""),
                "time_index": i,
                "time": float(tt),
                "temperature": float(env.get("temperature", cfg.temperature)),
                "pH": float(env.get("pH", cfg.pH)),
                "DO": float(env.get("DO", cfg.DO)),
                "X": float(x[i]),
                "B_total": float(b[i]),
                "z_ox": float(burdens[i, 0]),
                "z_atp": float(burdens[i, 1]),
                "z_bottle": float(burdens[i, 2]),
                "z_er": float(burdens[i, 3]),
                "E_PSY": float(enzyme_capacities[i, 0]),
                "E_DES": float(enzyme_capacities[i, 1]),
                "E_CYC": float(enzyme_capacities[i, 2]),
                **totals,
                "n_solver_errors": solver_errors,
                "n_skipped_intervals": skipped,
            }
        )
    traj = pd.DataFrame(traj_rows)
    flux = pd.DataFrame(flux_rows)
    cons = pd.DataFrame(constraint_rows)
    summary = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "hidden_regime_id": hidden_regime,
                "strain_id": candidate.get("strain_id", ""),
                "final_product": float(traj["B_total"].iloc[-1]),
                "product_AUC": float(np.trapezoid(traj["B_total"], traj["time"])),
                "final_biomass": float(traj["X"].iloc[-1]),
                "minimum_biomass": float(traj["X"].min()),
                "feasible": True,
                "severe_growth_collapse": bool(float(traj["X"].min()) < 0.05),
                **totals,
                "n_solver_errors": solver_errors,
                "n_skipped_intervals": skipped,
                "n_surrogate_evaluations": 0,
                "metabolic_backend": exact.EXACT_BACKEND,
                "exact_status": "complete_exact_dynamic_pfba",
                "failure_reason": "",
                "gem_source_checksum": checksum,
            }
        ]
    )
    del base_model, augmented
    exact.clear_solver_caches()
    gc.collect()
    return traj, flux, cons, summary


def public_result_from_exact(candidate: dict[str, Any], traj: pd.DataFrame, summary: pd.Series, accounting: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_hash": str(candidate["candidate_hash"]),
        "feasible": bool(summary.get("feasible", True)),
        "product_trajectory": [{"time": float(r.time), "B_total": float(r.B_total)} for r in traj[["time", "B_total"]].itertuples(index=False)],
        "final_product": float(summary["final_product"]),
        "product_AUC": float(summary["product_AUC"]),
        "biomass_trajectory": [{"time": float(r.time), "X": float(r.X)} for r in traj[["time", "X"]].itertuples(index=False)],
        "final_biomass": float(summary["final_biomass"]),
        "growth_failure": bool(summary.get("severe_growth_collapse", False)),
        "requested_public_assays": str(candidate.get("requested_assay_panel", "basic")),
        "sanitised_failure_category": "none" if bool(summary.get("feasible", True)) and not bool(summary.get("severe_growth_collapse", False)) else "infeasible_or_growth_failure",
        "physical_resource_accounting": accounting,
    }


def validate_request(request: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    candidates = request.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return [], [{"scope": "request", "error": "candidates_must_be_nonempty_list"}]
    if len(candidates) > 8:
        errors.append({"scope": "request", "error": "maximum_8_candidates_per_parallel_round"})
    universe = pd.read_csv(deployment.DATA / "deployment_benchmark_editable_reaction_universe.csv")
    validated = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append({"scope": "candidate", "candidate_id": "", "error": "candidate_must_be_object"})
            continue
        result = deployment.validate_sparse_design(candidate, universe)
        candidate_hash = result["candidate_hash"]
        validated.append({**candidate, "candidate_hash": candidate_hash, "validation": result})
        if not result["valid"]:
            errors.append({"scope": "candidate", "candidate_id": candidate.get("candidate_id", ""), "candidate_hash": candidate_hash, "error": result["reasons"]})
    return validated, errors


def process_request(
    path: Path,
    execute_fresh: bool = False,
    smoke_label: str | None = None,
    hidden_regime: str = "strong_dynamic_regulation",
    output_round_id: str | None = None,
) -> dict[str, Any]:
    deployment.ensure_dirs()
    request = load_json(path)
    round_id = str(request.get("round_id", path.stem.replace("_request", "")))
    effective_round_id = output_round_id or round_id
    campaign_id = str(request.get("campaign_id", ""))
    validated, errors = validate_request(request)
    log_rows = []
    if errors:
        response = {
            "round_id": round_id,
            "output_round_id": effective_round_id,
            "campaign_id": campaign_id,
            "status": "validation_failed_no_exact_simulation_run",
            "errors": errors,
            "hidden_exact_simulations_run": 0,
            "smoke_label": smoke_label,
        }
        out = deployment.PUBLIC_RESPONSES / f"{effective_round_id}_validation_errors.json"
        out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_rows.append({"round_id": round_id, "status": response["status"], "hidden_exact_simulations_run": 0, "response_path": str(out)})
    else:
        cache = deployment_exact_cache()
        cache_by_hash = cache.drop_duplicates("candidate_hash").set_index("candidate_hash") if "candidate_hash" in cache.columns else pd.DataFrame()
        results = []
        fresh_needed = []
        for candidate in validated:
            h = str(candidate["candidate_hash"])
            if not cache_by_hash.empty and h in cache_by_hash.index:
                results.append(sanitise_cache_result(candidate, cache_by_hash.loc[h]))
            else:
                fresh_needed.append(candidate)
        if fresh_needed and not execute_fresh:
            response = {
                "round_id": round_id,
                "output_round_id": effective_round_id,
                "campaign_id": campaign_id,
                "status": "validation_passed_fresh_exact_execution_required_not_run",
                "candidate_hashes_requiring_fresh_exact": [c["candidate_hash"] for c in fresh_needed],
                "cached_exact_matches": len(results),
                "hidden_exact_simulations_run": 0,
            }
            out = deployment.PUBLIC_RESPONSES / f"{effective_round_id}_validation_accepted_pending_execution.json"
            out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log_rows.append({"round_id": round_id, "status": response["status"], "hidden_exact_simulations_run": 0, "hidden_exact_lp_solves": 0, "response_path": str(out)})
        elif fresh_needed and execute_fresh:
            accounting = operational_accounting(validated)
            hidden_runs = 0
            hidden_lp_solves = 0
            failed = []
            for idx, candidate in enumerate(fresh_needed, start=1):
                cid = str(candidate["candidate_id"])
                print(f"[verifier] fresh exact {idx}/{len(fresh_needed)} {cid}", flush=True)
                tic = time.perf_counter()
                try:
                    traj, flux, cons, summary = exact_sparse_dynamic_rollout(candidate, hidden_regime=hidden_regime)
                    runtime = time.perf_counter() - tic
                    summary["runtime_seconds"] = runtime
                    summary["hidden_regime_id"] = hidden_regime
                    paths = hidden_output_paths(effective_round_id, cid)
                    traj.to_csv(paths["trajectory"], index=False)
                    flux.to_csv(paths["flux"], index=False)
                    cons.to_csv(paths["constraints"], index=False)
                    summary.to_csv(paths["summary"], index=False)
                    hidden_runs += 1
                    hidden_lp_solves += int(summary["n_actual_lp_solves"].iloc[0])
                    results.append(public_result_from_exact(candidate, traj, summary.iloc[0], accounting))
                    print(f"[verifier] complete {cid} lp={int(summary['n_actual_lp_solves'].iloc[0])} final_product={float(summary['final_product'].iloc[0]):.6g}", flush=True)
                except Exception as exc:
                    failed.append({"candidate_id": cid, "candidate_hash": candidate["candidate_hash"], "sanitised_failure_category": "solver_or_feasibility_failure", "error_type": type(exc).__name__})
                    paths = hidden_output_paths(effective_round_id, cid)
                    pd.DataFrame(
                        [
                            {
                                "candidate_id": cid,
                                "candidate_hash": candidate["candidate_hash"],
                                "hidden_regime_id": hidden_regime,
                                "n_actual_lp_solves": 0,
                                "n_solver_errors": 1,
                                "exact_status": "failed_exact_dynamic_pfba",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                            }
                        ]
                    ).to_csv(paths["summary"], index=False)
                    print(f"[verifier] failed {cid}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    exact.clear_solver_caches()
                    gc.collect()
            response = {
                "round_id": round_id,
                "output_round_id": effective_round_id,
                "campaign_id": campaign_id,
                "status": "fresh_exact_results_sanitised" if not failed else "fresh_exact_results_partial_failures_sanitised",
                "results": results,
                "failures": failed,
                "allowed_columns": ALLOWED_PUBLIC_RESULT_COLUMNS,
                "cached_exact_matches": len(results) - hidden_runs,
                "hidden_exact_simulations_run": hidden_runs,
                "hidden_exact_lp_solves": hidden_lp_solves,
            }
            out = deployment.PUBLIC_RESPONSES / f"{effective_round_id}_results.json"
            out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log_rows.append({"round_id": round_id, "status": response["status"], "hidden_exact_simulations_run": hidden_runs, "hidden_exact_lp_solves": hidden_lp_solves, "response_path": str(out)})
        else:
            accounting = operational_accounting(validated)
            for result in results:
                result["physical_resource_accounting"] = accounting
            response = {
                "round_id": round_id,
                "output_round_id": effective_round_id,
                "campaign_id": campaign_id,
                "status": "cached_exact_results_sanitised",
                "results": results,
                "allowed_columns": ALLOWED_PUBLIC_RESULT_COLUMNS,
                "hidden_exact_simulations_run": 0,
            }
            out = deployment.PUBLIC_RESPONSES / f"{effective_round_id}_results.json"
            out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log_rows.append({"round_id": round_id, "status": response["status"], "hidden_exact_simulations_run": 0, "response_path": str(out)})
    processed = deployment.PROCESSED_REQUESTS / (f"{effective_round_id}_request.json" if output_round_id else path.name)
    if path.exists():
        shutil.copy2(path, processed)
    pd.DataFrame(log_rows).to_csv(deployment.EVALUATOR_LOGS / f"{effective_round_id}_evaluator_log.csv", index=False)
    return {"round_id": round_id, "output_round_id": effective_round_id, "log_rows": log_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--execute-fresh", action="store_true")
    parser.add_argument("--smoke-label", default=None)
    parser.add_argument("--hidden-regime", default="strong_dynamic_regulation", choices=sorted(exact.regime_configs()))
    parser.add_argument("--output-round-id", default=None)
    args = parser.parse_args()
    result = process_request(
        Path(args.request),
        execute_fresh=args.execute_fresh,
        smoke_label=args.smoke_label,
        hidden_regime=args.hidden_regime,
        output_round_id=args.output_round_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
