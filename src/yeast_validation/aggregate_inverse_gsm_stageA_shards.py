#!/usr/bin/env python3
"""Aggregate fresh-process Stage A inverse GSM shards into canonical outputs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import run_gem_dynamic_capacity as capacity
import run_inverse_gsm_calibration as inv


def read_many(shard_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(shard_root.glob(f"*/data/{filename}")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def copy_reference_artifacts(shard_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_dir = shard_root / "reference" / "data"
    if not reference_dir.exists():
        raise SystemExit(f"Missing reference shard at {reference_dir}")
    copied = {}
    for name in [
        "inverse_gsm_calibration_culture_manifest.csv",
        "inverse_gsm_calibration_lockbox.csv",
        "inverse_gsm_calibration_controller_bounds.csv",
        "inverse_gsm_parameterization_audit.csv",
        "inverse_gsm_stageA_reference_losses.csv",
        "inverse_gsm_stageA_random_baseline.csv",
        "inverse_gsm_stageA_random_baseline_summary.csv",
        "inverse_gsm_stageA_1d_landscape.csv",
        "inverse_gsm_stageA_landscape_sensitivity.csv",
    ]:
        src = reference_dir / name
        if src.exists():
            dst = inv.DATA / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[name] = dst
    required = [
        "inverse_gsm_stageA_reference_losses.csv",
        "inverse_gsm_stageA_random_baseline.csv",
        "inverse_gsm_stageA_1d_landscape.csv",
    ]
    missing = [name for name in required if name not in copied]
    if missing:
        raise SystemExit(f"Reference shard missing required artifacts: {missing}")
    return (
        pd.read_csv(copied["inverse_gsm_stageA_reference_losses.csv"]),
        pd.read_csv(copied["inverse_gsm_stageA_random_baseline.csv"]),
        pd.read_csv(copied["inverse_gsm_stageA_1d_landscape.csv"]),
    )


def set_output_root(output_root: Path | None) -> None:
    if output_root is None:
        return
    inv.DATA = output_root / "data"
    inv.FIGURES = output_root / "figures"
    inv.RESULTS = output_root / "results"


def aggregate(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    set_output_root(args.output_root)
    inv.DATA.mkdir(exist_ok=True)
    inv.FIGURES.mkdir(exist_ok=True)
    inv.RESULTS.mkdir(parents=True, exist_ok=True)
    reference, random_baseline, landscape = copy_reference_artifacts(args.shard_root)
    optimizer_summary = read_many(args.shard_root, "inverse_gsm_stageA_optimizer_comparison.csv")
    convergence = read_many(args.shard_root, "inverse_gsm_stageA_convergence.csv")
    expected = len(args.optimizers) * len(args.restart_seeds)
    if len(optimizer_summary) != expected:
        raise SystemExit(f"Expected {expected} optimizer shard rows, found {len(optimizer_summary)}")
    if not convergence.empty:
        convergence = convergence.sort_values(["optimizer", "restart", "evaluation_number"]).reset_index(drop=True)
        convergence["cumulative_LP_solve_count"] = convergence.groupby(["optimizer", "restart", "reporter_weight"])[
            "LP_solve_count"
        ].cumsum()
    optimizer_summary = optimizer_summary.sort_values(["optimizer", "restart"]).reset_index(drop=True)

    dataset = inv.load_dataset()
    culture = pd.Series(inv.select_diagnostic_cultures(dataset).head(1).iloc[0].to_dict())
    bounds = inv.global_controller_bounds(dataset)
    scales = inv.fit_scales(dataset, 1)
    source_path, augmented = capacity.load_augmented(None)
    oracle_params = inv.oracle_params_for_culture(dataset, culture, bounds, 1, 1)

    convergence.to_csv(inv.DATA / "inverse_gsm_stageA_convergence.csv", index=False)
    optimizer_summary.to_csv(inv.DATA / "inverse_gsm_stageA_optimizer_comparison.csv", index=False)
    inv.stage_a_auxiliary_tables(optimizer_summary)
    channel_errors = inv.stage_a_controller_channel_errors(optimizer_summary, oracle_params, bounds)
    landscape_sensitivity = inv.stage_a_landscape_sensitivity(landscape, dataset, culture, bounds)
    reproducibility = inv.stage_a_reproducibility_analysis(reference, random_baseline, optimizer_summary)
    phenotype = inv.stage_a_phenotype_reconstruction(culture, dataset, bounds, scales, optimizer_summary, source_path, augmented)
    efficiency, cost = inv.stage_a_efficiency_analysis(reference, convergence, optimizer_summary)
    gate = inv.stage_a_gate(reference, random_baseline, optimizer_summary, n_restarts=len(args.restart_seeds))
    decision = inv.stage_a_decision_table(gate, optimizer_summary, reproducibility, landscape_sensitivity, cost)
    decision.to_csv(inv.DATA / "inverse_gsm_stageA_decision_table.csv", index=False)
    decision.to_csv(inv.DATA / "inverse_gsm_calibration_decision.csv", index=False)
    inv.save_stage_a_figures(convergence, optimizer_summary, landscape, reference, random_baseline)
    inv.write_stage_a_summary(decision, gate, args)
    return {
        "optimizer_summary": optimizer_summary,
        "convergence": convergence,
        "channel_errors": channel_errors,
        "landscape_sensitivity": landscape_sensitivity,
        "reproducibility": reproducibility,
        "phenotype": phenotype,
        "efficiency": efficiency,
        "cost": cost,
        "gate": gate,
        "decision": decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--optimizers", nargs="+", default=["differential_evolution", "powell_multistart", "annealing"])
    parser.add_argument("--restart-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    return parser.parse_args(argv)


if __name__ == "__main__":
    outputs = aggregate(parse_args())
    print(outputs["decision"].to_string(index=False))
