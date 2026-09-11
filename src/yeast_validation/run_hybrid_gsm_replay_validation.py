#!/usr/bin/env python3
"""G7 validation: does the repaired hybrid model's predicted control vector
c(t) actually change real Yeast9 constraints and outputs?

This replays a trained `hybrid_learned_physiology` checkpoint's predicted
`c(t)` for a handful of cultures through the *real* Yeast9 LP solver, reusing
`run_reporter_grounded_hybrid_distillation.apply_predicted_interface_controls`
unmodified -- that function already takes only the compact 6-key control
dict (no `z_*`/hidden-state input at all), so no new leaky bridge is needed.

Two checks:
  1. Different cultures' predicted c(t) produce different real Yeast9 flux
     outputs (the model isn't predicting a constant control regardless of
     environment).
  2. A direct perturbation of one control value (holding the rest fixed)
     changes at least one real Yeast9 output -- a literal causal-manipulation
     check, not just a correlational one.

Kept intentionally small-scale (a handful of cultures, not all 125) --
this is a validation check, not a campaign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402
import run_repaired_hybrid_training as train_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def replay_culture(model, cfg: gem.GEMCultureConfig, controls_over_time: np.ndarray, control_cols: list[str], n_intervals: int) -> pd.DataFrame:
    rows = []
    for t in range(n_intervals):
        interval_model = model.copy()
        controls = {col: float(controls_over_time[t, i]) for i, col in enumerate(control_cols)}
        applied = prod.apply_predicted_interface_controls(interval_model, cfg, controls)
        flux = gem.solve_staged(interval_model, cfg, gamma=float(np.clip(controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")
        rows.append({"interval": t, **{f"control__{k}": v for k, v in controls.items()}, "biomass_flux": flux["biomass_flux"], "beta_carotene_flux": flux["beta_carotene_flux"], "growth_status": flux["growth_status"], "product_status": flux["product_status"], "pfba_status": flux["pfba_status"], **applied})
    return pd.DataFrame(rows)


def main() -> None:
    ckpt_path = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "hybrid_learned_physiology__full__seed11.npz"
    params, meta = tss.load_checkpoint(ckpt_path)
    print("Loaded checkpoint:", meta)

    dataset = train_mod.load_dataset()
    surrogate = train_mod.fit_frozen_surrogate()
    train_mask = dataset["splits"] == "train"
    env_mean, env_std = train_mod.standardize_fit(dataset["env_raw"], train_mask)
    env_std_data = (dataset["env_raw"] - env_mean) / env_std

    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model = gem.load_model(cobra, source_path)
    augmented, _manifest = gem.install_beta_carotene_pathway(base_model)

    rng = np.random.default_rng(7)
    culture_idx = rng.choice(len(dataset["culture_ids"]), size=4, replace=False)

    n_intervals = 6  # small-scale validation, not a full 48-interval replay
    cache = tss.forward(params, env_std_data, dataset["t_len"])
    all_replays = []
    for ci in culture_idx:
        env_row = dataset["env_raw"][ci]
        culture_id = dataset["culture_ids"][ci]
        cfg = gem.GEMCultureConfig(temperature=float(env_row[0]), pH=float(env_row[1]), DO=float(env_row[2]))
        c_pre = cache.z[:n_intervals, ci, :] @ params.Wc + params.bc
        c = c_pre * surrogate.control_std + surrogate.control_mean
        replay = replay_culture(augmented, cfg, c, sur.CONTROL_COLUMNS, n_intervals)
        replay["culture_id"] = culture_id
        all_replays.append(replay)
    replay_df = pd.concat(all_replays, ignore_index=True)
    replay_df.to_csv(DATA / "hybrid_gsm_replay_validation.csv", index=False)

    print("\n=== Check 1: different cultures -> different real Yeast9 outputs ===")
    by_culture = replay_df.groupby("culture_id")[["biomass_flux", "beta_carotene_flux"]].mean()
    print(by_culture)
    spread_biomass = float(by_culture["biomass_flux"].std())
    spread_product = float(by_culture["beta_carotene_flux"].std())
    check1_pass = spread_biomass > 1e-4 or spread_product > 1e-4
    print(f"cross-culture std: biomass_flux={spread_biomass:.6f}, beta_carotene_flux={spread_product:.6f} -> {'PASS' if check1_pass else 'FAIL'}")

    print("\n=== Check 2: direct control perturbation changes real Yeast9 output ===")
    ci = culture_idx[0]
    env_row = dataset["env_raw"][ci]
    cfg = gem.GEMCultureConfig(temperature=float(env_row[0]), pH=float(env_row[1]), DO=float(env_row[2]))
    c_pre = cache.z[0, ci, :] @ params.Wc + params.bc
    c_base = (c_pre * surrogate.control_std + surrogate.control_mean).reshape(1, -1)
    base_controls = {col: float(c_base[0, i]) for i, col in enumerate(sur.CONTROL_COLUMNS)}
    model_base = augmented.copy()
    prod.apply_predicted_interface_controls(model_base, cfg, base_controls)
    flux_base = gem.solve_staged(model_base, cfg, gamma=float(np.clip(base_controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")

    perturbed_controls = dict(base_controls)
    idx = sur.CONTROL_COLUMNS.index("PSY_effective_upper_bound")
    perturbed_controls["PSY_effective_upper_bound"] = base_controls["PSY_effective_upper_bound"] * 0.3
    model_pert = augmented.copy()
    prod.apply_predicted_interface_controls(model_pert, cfg, perturbed_controls)
    flux_pert = gem.solve_staged(model_pert, cfg, gamma=float(np.clip(perturbed_controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")

    delta_product = abs(flux_pert["beta_carotene_flux"] - flux_base["beta_carotene_flux"])
    print(f"base beta_carotene_flux={flux_base['beta_carotene_flux']:.6f}, perturbed (PSY UB x0.3)={flux_pert['beta_carotene_flux']:.6f}, delta={delta_product:.6f}")
    check2_pass = delta_product > 1e-6
    print(f"-> {'PASS' if check2_pass else 'FAIL'}")

    pd.DataFrame(
        [
            {"gate": "G7_check1_cross_culture_output_varies", "passed": bool(check1_pass), "evidence": f"std biomass_flux={spread_biomass:.6f}, std beta_carotene_flux={spread_product:.6f}"},
            {"gate": "G7_check2_control_perturbation_changes_output", "passed": bool(check2_pass), "evidence": f"delta_beta_carotene_flux={delta_product:.6f}"},
        ]
    ).to_csv(DATA / "hybrid_gsm_replay_gate.csv", index=False)


if __name__ == "__main__":
    main()
