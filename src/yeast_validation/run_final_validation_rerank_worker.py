#!/usr/bin/env python3
"""Vanda worker for one final-validation real-Yeast9 rerank candidate.

Primary mode intentionally preserves the actual v2 rerank implementation
contract: the learned state-space checkpoint and edit mapping are used, and
the control unstandardization falls back to the legacy GSM surrogate statistics
when `--scale-source legacy_v2` is selected. This matches the positive v2
benchmark path and is kept separate from optional corrected-MLP sensitivity
runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402

DT_REAL = 0.25
BETA_DEG_BASE = 0.002
MLP_SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"


def load_env_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["env_mean"], d["env_std"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-row", required=True, help="CSV row extracted from rerank_task_manifest.csv")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--scale-source", choices=["legacy_v2", "mlp_corrected"], default="legacy_v2")
    args = parser.parse_args()

    with Path(args.task_row).open(newline="", encoding="utf-8") as handle:
        task = next(csv.DictReader(handle))
    candidate = json.loads(Path(str(task["candidate_json"])).read_text(encoding="utf-8"))
    env_mean, env_std = load_env_stats(ROOT / str(task["normalization_stats"]))
    surrogate = mlp.GSMSurrogateMLP.load(MLP_SURROGATE_PATH) if args.scale_source == "mlp_corrected" else None
    scorer = scorer_mod.RepairedVirtualScorer.load(ROOT / str(task["checkpoint"]), env_mean, env_std, surrogate=surrogate)

    env = candidate["environment"]
    env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
    env_std_in = (env_raw - scorer.env_mean) / scorer.env_std
    cache = tss.forward(scorer.params, env_std_in, scorer_mod.T_LEN)
    c_pre = cache.z[:-1, 0, :] @ scorer.params.Wc + scorer.params.bc
    c_baseline = c_pre * scorer.surrogate.control_std + scorer.surrogate.control_mean
    c_edited, edit_notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, candidate.get("edits", []))
    clip_lo = scorer.surrogate.control_mean - scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    clip_hi = scorer.surrogate.control_mean + scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    n_clipped = int(np.sum((c_edited < clip_lo) | (c_edited > clip_hi)))
    c_final = np.clip(c_edited, clip_lo, clip_hi)

    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)
    cfg = gem.GEMCultureConfig(temperature=float(env["temperature"]), pH=float(env["pH"]), DO=float(env["DO"]))

    n_intervals = scorer_mod.T_LEN - 1
    x = np.zeros(scorer_mod.T_LEN)
    b = np.zeros(scorer_mod.T_LEN)
    x[0], b[0] = scorer_mod.X0, scorer_mod.B0
    flux_rows = []
    infeasible = 0
    actual_lp = 0
    for t in range(n_intervals):
        interval_model = augmented.copy()
        controls = {col: float(c_final[t, i]) for i, col in enumerate(sur.CONTROL_COLUMNS)}
        prod.apply_predicted_interface_controls(interval_model, cfg, controls)
        try:
            flux = gem.solve_staged(interval_model, cfg, gamma=float(np.clip(controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")
        except Exception:
            infeasible += 1
            x[t + 1], b[t + 1] = x[t], b[t]
            continue
        actual_lp += int(flux.get("n_actual_lp_solves", 0))
        biomass_flux = float(flux["biomass_flux"])
        beta_flux = float(flux["beta_carotene_flux"])
        x[t + 1] = max(1e-9, x[t] + DT_REAL * biomass_flux * x[t])
        b[t + 1] = max(0.0, b[t] + DT_REAL * beta_flux * x[t] - DT_REAL * BETA_DEG_BASE * b[t])
        flux_rows.append(
            {
                "interval_index": t,
                "biomass_flux": biomass_flux,
                "beta_carotene_flux": beta_flux,
                "n_actual_lp_solves": int(flux.get("n_actual_lp_solves", 0)),
            }
        )

    summary = {
        **task,
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "scale_source": args.scale_source,
        "final_product_pred_controls_via_real_gsm": float(b[-1]),
        "final_biomass_pred_controls_via_real_gsm": float(x[-1]),
        "product_AUC_pred_controls_via_real_gsm": float(np.trapezoid(b, dx=DT_REAL)),
        "n_infeasible_intervals": infeasible,
        "n_intervals": n_intervals,
        "n_actual_lp_solves": actual_lp,
        "n_control_values_clipped": n_clipped,
        "edit_notes": ";".join(edit_notes),
        "metabolic_backend": "yeast_gem_lp",
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    flux_path = out.with_name(out.stem.replace("_summary", "_flux") + ".csv")
    with flux_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["interval_index", "biomass_flux", "beta_carotene_flux", "n_actual_lp_solves"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flux_rows)


if __name__ == "__main__":
    main()
