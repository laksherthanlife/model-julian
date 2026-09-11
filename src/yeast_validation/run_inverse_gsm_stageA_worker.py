#!/usr/bin/env python3
"""Shard worker for the Stage A inverse GSM calibration run.

Each optimizer shard is one optimizer x one restart seed in a fresh process.
This avoids a long-lived COBRA model process and makes interrupted runs
resumable without changing the inverse-calibration objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_gem_dynamic_capacity as capacity
import run_inverse_gsm_calibration as inv


def set_output_root(output_root: Path) -> None:
    inv.DATA = output_root / "data"
    inv.FIGURES = output_root / "figures"
    inv.RESULTS = output_root / "results"
    inv.DATA.mkdir(parents=True, exist_ok=True)
    inv.FIGURES.mkdir(parents=True, exist_ok=True)
    inv.RESULTS.mkdir(parents=True, exist_ok=True)


def context(fake_solver: bool):
    dataset = inv.load_dataset()
    manifest = inv.select_diagnostic_cultures(dataset).head(1)
    culture = pd.Series(manifest.iloc[0].to_dict())
    inv.write_lockbox()
    bounds = inv.global_controller_bounds(dataset)
    audit = inv.parameterization_audit(bounds, "Stage A", n_intervals=1, n_knots=1)
    if not (
        audit["total_free_parameters"].eq(len(inv.INTERFACE_COLUMNS)).all()
        and audit["knots_per_channel"].eq(1).all()
        and audit["controller_channel_count"].eq(len(inv.INTERFACE_COLUMNS)).all()
    ):
        raise SystemExit("Stage A parameterization audit failed")
    inv.assert_no_lockbox_usage(inv.ENV_COLUMNS, ["B_total", "X"], [])
    scales = inv.fit_scales(dataset, 1)
    source_path, augmented = (None, None) if fake_solver else capacity.load_augmented(None)
    solver_fn = inv.fake_solver if fake_solver else None
    return dataset, culture, bounds, scales, source_path, augmented, solver_fn


def run_reference(args: argparse.Namespace) -> None:
    dataset, culture, bounds, scales, source_path, augmented, solver_fn = context(args.fake_solver)
    reference = inv.stage_a_reference_losses(culture, dataset, bounds, scales, source_path, augmented, solver_fn)
    random_baseline = inv.stage_a_random_baseline(
        culture, dataset, bounds, scales, args.random_samples, args.seed, source_path, augmented, solver_fn
    )
    landscape = inv.stage_a_landscape(culture, dataset, bounds, scales, args.landscape_points, source_path, augmented, solver_fn)
    inv.stage_a_landscape_sensitivity(landscape, dataset, culture, bounds)
    print(
        json.dumps(
            {
                "mode": "reference",
                "output_root": str(args.output_root),
                "oracle_loss": float(reference[reference["reference"].eq("oracle_true_controller")]["loss"].iloc[0]),
                "random_samples": int(len(random_baseline)),
            },
            sort_keys=True,
        )
    )


def run_optimizer(args: argparse.Namespace) -> None:
    dataset, culture, bounds, scales, source_path, augmented, solver_fn = context(args.fake_solver)
    reference = inv.stage_a_reference_losses(culture, dataset, bounds, scales, source_path, augmented, solver_fn)
    lo, hi = inv.param_bounds(bounds, 1)
    oracle_params = inv.oracle_params_for_culture(dataset, culture, bounds, 1, 1)
    mean_loss = float(reference[reference["reference"].eq("training_mean_controller")]["loss"].iloc[0])
    oracle_loss = float(reference[reference["reference"].eq("oracle_true_controller")]["loss"].iloc[0])
    init_strategy = "training_mean_controller" if args.restart_index == 0 else "global_random_valid"
    evaluate = inv.stage_a_evaluator(culture, dataset, bounds, scales, 0.0, source_path, augmented, solver_fn)

    if args.optimizer == "differential_evolution":
        best, convergence = inv.optimizer_de(evaluate, lo, hi, args.max_evaluations, args.restart_seed, args.restart_index, init_strategy)
    elif args.optimizer == "powell_multistart":
        start = inv.mean_controller_params(dataset, bounds, 1, 1) if args.restart_index == 0 else None
        best, convergence = inv.optimizer_powell(
            evaluate, lo, hi, args.max_evaluations, args.restart_seed, args.restart_index, init_strategy, start=start
        )
    elif args.optimizer == "annealing":
        best, convergence = inv.optimizer_annealing(evaluate, lo, hi, args.max_evaluations, args.restart_seed, args.restart_index, init_strategy)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer!r}")

    convergence = convergence.assign(culture_id=culture["culture_id"], reporter_weight=0.0)
    if not convergence.empty:
        convergence["cumulative_LP_solve_count"] = convergence.groupby(["optimizer", "restart", "reporter_weight"])[
            "LP_solve_count"
        ].cumsum()
    summary = pd.DataFrame(
        [
            {
                "culture_id": culture["culture_id"],
                "optimizer": args.optimizer,
                "restart": args.restart_index,
                "restart_seed": args.restart_seed,
                "initialization_strategy": init_strategy,
                "reporter_weight": 0.0,
                "metabolic_backend": "fake_solver" if solver_fn is not None else "yeast_gem_lp",
                "best_loss": float(best["loss"]),
                "best_product_loss": float(best["parts"].get("product_loss", np.nan)),
                "best_biomass_loss": float(best["parts"].get("biomass_loss", np.nan)),
                "best_reporter_block_loss": float(best["parts"].get("reporter_block_loss", np.nan)),
                "posthoc_controller_distance": inv.controller_distance(best["params"], oracle_params, bounds, 1),
                "fractional_closure_toward_oracle": (mean_loss - float(best["loss"])) / max(mean_loss - oracle_loss, inv.EPS),
                "best_controller_vector_json": json.dumps([float(x) for x in best["params"]]),
                "n_evaluations": args.max_evaluations,
                "best_evaluation_number": int(best.get("best_evaluation_number", 0)),
                "wall_time_seconds": float(best.get("wall_time_seconds", np.nan)),
                **best["accounting"],
            }
        ]
    )
    convergence.to_csv(inv.DATA / "inverse_gsm_stageA_convergence.csv", index=False)
    summary.to_csv(inv.DATA / "inverse_gsm_stageA_optimizer_comparison.csv", index=False)
    inv.stage_a_auxiliary_tables(summary)
    inv.stage_a_controller_channel_errors(summary, oracle_params, bounds)
    inv.stage_a_phenotype_reconstruction(culture, dataset, bounds, scales, summary, source_path, augmented, solver_fn)
    print(
        json.dumps(
            {
                "mode": "optimizer",
                "output_root": str(args.output_root),
                "optimizer": args.optimizer,
                "restart_seed": args.restart_seed,
                "best_loss": float(best["loss"]),
                "wall_time_seconds": float(best.get("wall_time_seconds", np.nan)),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "optimizer"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--optimizer", choices=["differential_evolution", "powell_multistart", "annealing"])
    parser.add_argument("--restart-seed", type=int)
    parser.add_argument("--restart-index", type=int)
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--landscape-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--fake-solver", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "optimizer" and (args.optimizer is None or args.restart_seed is None or args.restart_index is None):
        parser.error("--mode optimizer requires --optimizer, --restart-seed, and --restart-index")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    set_output_root(parsed.output_root)
    if parsed.mode == "reference":
        run_reference(parsed)
    else:
        run_optimizer(parsed)
