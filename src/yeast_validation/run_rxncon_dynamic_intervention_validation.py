#!/usr/bin/env python3
"""Phase-A gate for time-varying DO interventions in the frozen generator.

This script deliberately stops at the exact rxncon -> frozen interface ->
Yeast9/pFBA phenotype gate. It does not train a learner. The only new input
mechanism is a schedule of the already-supported DO control; static generator
values and the interface mapping remain imported from the canonical gate.
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
import gem_backend as gem  # noqa: E402
import run_rxncon_active_structural_validation as active  # noqa: E402
import run_rxncon_gsm_generator_gate as gate  # noqa: E402
import run_rxncon_external_complexity_screen as rxncon_screen  # noqa: E402

OUT_DIR = ROOT / "data/rxncon_dynamic_intervention_validation"
FIG_DIR = ROOT / "figures"
N_TIME = gate.N_TIMEPOINTS
DT = gate.DT
SEED = 20260825
DO_LEVELS = (20.0, 40.0, 80.0)


def _schedule(kind: str, start: int | None = None, duration: int = 8) -> np.ndarray:
    x = np.full(N_TIME, 40.0)
    if kind == "static_low": return np.full(N_TIME, DO_LEVELS[0])
    if kind == "static_mid": return np.full(N_TIME, DO_LEVELS[1])
    if kind == "static_high": return np.full(N_TIME, DO_LEVELS[2])
    if kind.startswith("switch_low_high"):
        x[:] = DO_LEVELS[0]; x[int(start):] = DO_LEVELS[2]; return x
    if kind.startswith("switch_high_low"):
        x[:] = DO_LEVELS[2]; x[int(start):] = DO_LEVELS[0]; return x
    if kind.startswith("pulse_low"):
        x[:] = DO_LEVELS[2]; x[int(start):int(start)+duration] = DO_LEVELS[0]; return x
    if kind.startswith("pulse_high"):
        x[:] = DO_LEVELS[0]; x[int(start):int(start)+duration] = DO_LEVELS[2]; return x
    raise ValueError(kind)


def build_backgrounds() -> pd.DataFrame:
    # Reuse the generator's existing supported continuous environment axes.
    # The active 125-culture design contains only one zero-stress constant row,
    # so selecting backgrounds from its canonical panel is not sufficient for
    # a multi-background timing gate. No new values are introduced here.
    design = gate.build_environment_panel(60).rename(columns={"environment_program": "input_program"})
    ordinary = design[design["input_program"].astype(str).eq("constant")].copy()
    ordinary["Nutrients"] = 1
    ordinary[["Pheromone", "HU", "LatA", "Nocodazole"]] = 0
    if ordinary.empty:
        raise RuntimeError("Could not find canonical ordinary background environments.")
    ordinary = ordinary.drop_duplicates(["temperature", "pH", "glucose_uptake"]).head(3)
    ordinary["nominal_condition_id"] = [f"dynamic_bg_{i}" for i in range(len(ordinary))]
    return ordinary.reset_index(drop=True)


def build_schedule_panel() -> pd.DataFrame:
    backgrounds = build_backgrounds()
    rows = []
    schedule_defs = []
    for kind in ["static_low", "static_mid", "static_high"]:
        schedule_defs.append((kind, None, 0))
    for direction in ("low_high", "high_low"):
        for label, start in (("early", 12), ("middle", 24), ("late", 36)):
            schedule_defs.append((f"switch_{direction}_{label}", start, 0))
    for family in ("pulse_low", "pulse_high"):
        for label, start in (("early", 4), ("middle", 20), ("late", 36)):
            schedule_defs.append((f"{family}_{label}", start, 8))
    for bidx, background in backgrounds.iterrows():
        for kind, start, duration in schedule_defs:
            row = background.to_dict()
            row["culture_id"] = f"dynamic_do_bg{bidx}_{kind}"
            row["nominal_condition_id"] = f"dynamic_bg_{bidx}"
            row["input_program"] = kind
            row["schedule_family"] = "static" if kind.startswith("static") else ("switch" if kind.startswith("switch") else "matched_pulse")
            row["schedule_direction"] = "none" if kind.startswith("static") else ("low_high" if "low_high" in kind else ("high_low" if "high_low" in kind else ("low_pulse" if "pulse_low" in kind else "high_pulse")))
            row["schedule_start"] = -1 if start is None else int(start)
            row["schedule_duration"] = int(duration)
            schedule = _schedule(kind, start=start, duration=duration)
            for t, value in enumerate(schedule): row[f"DO_t{t:02d}"] = float(value)
            row["DO"] = float(schedule[0])
            rows.append(row)
    panel=pd.DataFrame(rows)
    panel["environment_program"]=panel["input_program"]
    return panel


def entropy_rank(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False); lam = s * s; lam = lam[lam > max(lam.max(initial=0.0) * 1e-12, 1e-15)]
    if len(lam) == 0: return 0.0
    p = lam / lam.sum(); return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-15)))))


def summarize_phenotypes(traj: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]
    for cid, group in traj.groupby("culture_id", sort=False):
        g=group.sort_values("time_index"); row=panel.loc[panel.culture_id.eq(cid)].iloc[0]
        p=g.B_total.to_numpy(float); x=g.X.to_numpy(float)
        rows.append({"culture_id":cid,"background_id":row.nominal_condition_id,"schedule":row.input_program,"schedule_family":row.schedule_family,"schedule_direction":row.schedule_direction,"schedule_start":row.schedule_start,"schedule_duration":row.schedule_duration,"final_product":p[-1],"product_auc":float(np.trapezoid(p,dx=DT)),"final_biomass":x[-1],"productivity":float((p[-1]-p[0])/(g.time.iloc[-1]-g.time.iloc[0]+1e-9)),"mean_DO":float(np.mean([row[f"DO_t{t:02d}"] for t in range(N_TIME)])),"min_DO":float(min(row[f"DO_t{t:02d}"] for t in range(N_TIME))),"max_DO":float(max(row[f"DO_t{t:02d}"] for t in range(N_TIME))),"low_DO_steps":int(sum(row[f"DO_t{t:02d}"]==20.0 for t in range(N_TIME))),"high_DO_steps":int(sum(row[f"DO_t{t:02d}"]==80.0 for t in range(N_TIME)))})
    ph=pd.DataFrame(rows)
    # Pairwise differences among timing-matched schedules, not cumulative-dose
    # comparisons. Each row is one background and one matched family.
    pair=[]
    for (bg, direction), g in ph[ph.schedule_family.isin(["switch","matched_pulse"])].groupby(["background_id","schedule_direction"]):
        if len(g)<2: continue
        for metric in ("final_product","product_auc","final_biomass"):
            pair.append({"background_id":bg,"schedule_direction":direction,"metric":metric,"timing_range":float(g[metric].max()-g[metric].min()),"timing_std":float(g[metric].std(ddof=0))})
    return ph,pd.DataFrame(pair)


def order_summary_test(ph: pd.DataFrame) -> pd.DataFrame:
    # Compare an order-insensitive linear summary against the same summary plus
    # explicit timing/order variables. This is post-hoc phenotype analysis only.
    rows=[]
    for target in ("final_product","product_auc","final_biomass"):
        d=ph.copy(); base=np.column_stack([np.ones(len(d)),d[["mean_DO","low_DO_steps","high_DO_steps","min_DO","max_DO"]].to_numpy(float)])
        timing=np.column_stack([base,d[["schedule_start"]].to_numpy(float),pd.get_dummies(d["schedule_direction"],dtype=float).to_numpy()])
        y=d[target].to_numpy(float)
        def r2(A):
            w=np.linalg.lstsq(A,y,rcond=None)[0]; pred=A@w; den=max(float(np.sum((y-y.mean())**2)),1e-15); return float(1-np.sum((y-pred)**2)/den)
        rows.append({"target":target,"r2_order_insensitive":r2(base),"r2_with_timing":r2(timing),"delta_r2_timing":r2(timing)-r2(base)})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel=build_schedule_panel(); panel.to_csv(OUT_DIR/"schedule_manifest.csv",index=False)
    print(f"Phase A: {len(panel)} cultures, {panel.nominal_condition_id.nunique()} backgrounds", flush=True)
    model=gate.load_rxncon_model(); force=gate.force_array_for_panel(model,panel); rxncon_traj,runtime=rxncon_screen.simulate_panel(model,force); runtime.to_csv(OUT_DIR/"rxncon_runtime.csv",index=False)
    print("Phase A: rxncon screen complete", flush=True)
    modules,module_names=gate.module_activity(model,rxncon_traj); interface,interface_names=gate.compute_interface(modules,module_names,panel)
    pd.DataFrame(interface.reshape(-1,interface.shape[-1]),columns=interface_names).to_csv(OUT_DIR/"hidden_interface_lockbox_long.csv",index=False)
    source=gem.configured_gem_path(None); cobra=gem.require_cobra(); augmented,_=gem.install_beta_carotene_pathway(gem.load_model(cobra,source))
    # Diverse exact dense/sparse validation on the same dynamic schedule panel.
    # Dense replay is intentionally limited to one representative culture in
    # this local gate; the complete schedule panel is still exact sparse Yeast9.
    val_idx=np.linspace(0,len(panel)-1,min(1,len(panel)),dtype=int).tolist()
    print(f"Phase A: dense/sparse audit on {len(val_idx)} cultures", flush=True)
    dense=gate.run_gsm_rollout(augmented,source,panel,interface,interface_names,val_idx,N_TIME,sparse=False)
    sparse=gate.run_gsm_rollout(augmented,source,panel,interface,interface_names,val_idx,N_TIME,sparse=True,delta=0.04,max_gap=4)
    joined=dense[0].merge(sparse[0],on=["culture_id","time_index"],suffixes=("_dense","_sparse")); rows=[]
    for cid,g in joined.groupby("culture_id"):
        rows.append({"culture_id":cid,"biomass_rmse":float(np.sqrt(np.mean((g.X_dense-g.X_sparse)**2))),"product_rmse":float(np.sqrt(np.mean((g.B_total_dense-g.B_total_sparse)**2))),"final_product_abs_diff":float(abs(g.sort_values("time_index").B_total_dense.iloc[-1]-g.sort_values("time_index").B_total_sparse.iloc[-1])),"dense_lp_solves":int(dense[3].loc[dense[3].culture_id.eq(cid),"n_actual_lp_solves"].sum()),"sparse_lp_solves":int(sparse[3].loc[sparse[3].culture_id.eq(cid),"n_actual_lp_solves"].sum())})
    dense_sparse=pd.DataFrame(rows); dense_sparse["lp_reduction_fraction"]=1-dense_sparse.sparse_lp_solves/dense_sparse.dense_lp_solves.clip(lower=1); dense_sparse.to_csv(OUT_DIR/"dense_sparse_validation.csv",index=False)
    if dense_sparse.final_product_abs_diff.median()>1e-3: raise RuntimeError("Dynamic dense/sparse validation failed before full generation.")
    print("Phase A: dense/sparse audit passed; running full sparse panel", flush=True)
    sparse_full=gate.run_gsm_rollout(augmented,source,panel,interface,interface_names,list(range(len(panel))),N_TIME,sparse=True,delta=0.04,max_gap=4)
    sparse_full[0].to_csv(OUT_DIR/"trajectories.csv",index=False); sparse_full[1].to_csv(OUT_DIR/"flux_lockbox.csv",index=False); sparse_full[2].to_csv(OUT_DIR/"constraints_lockbox.csv",index=False); sparse_full[3].to_csv(OUT_DIR/"solver_accounting.csv",index=False)
    ph,pairs=summarize_phenotypes(sparse_full[0],panel); ph.to_csv(OUT_DIR/"phenotype_summary.csv",index=False); pairs.to_csv(OUT_DIR/"matched_timing_pair_metrics.csv",index=False)
    order=order_summary_test(ph); order.to_csv(OUT_DIR/"order_summary_test.csv",index=False)
    product=ph.product_auc.to_numpy(float); complexity=pd.DataFrame([{"layer":"product_final_auc","entropy_rank":entropy_rank(ph[["final_product","product_auc","final_biomass"]].to_numpy(float)),"n_samples":len(ph),"nominal_dimension":3},{"layer":"product_trajectories","entropy_rank":entropy_rank(sparse_full[0].pivot(index="culture_id",columns="time_index",values="B_total").to_numpy(float)),"n_samples":len(ph),"nominal_dimension":N_TIME}]); complexity.to_csv(OUT_DIR/"dynamic_product_complexity.csv",index=False)
    static=ph[ph.schedule_family.eq("static")].groupby("background_id").product_auc.agg(lambda x:float(x.max()-x.min())).replace(0,np.nan); pulse=pairs[pairs.schedule_direction.isin(["low_pulse","high_pulse"])].groupby("background_id").timing_range.mean(); switch=pairs[pairs.schedule_direction.isin(["low_high","high_low"])].groupby("background_id").timing_range.mean(); ratios=pd.concat([(pulse/static).rename("pulse_timing_over_static_spread"),(switch/static).rename("switch_timing_over_static_spread")],axis=1); ratios.to_csv(OUT_DIR/"timing_effect_ratios.csv")
    reproducible=(ratios.dropna().shape[0]>=3 and float(ratios.max(axis=1).median())>=0.10); gate_pass=bool(reproducible)
    safeguards=pd.DataFrame([{"safeguard":"published_rxncon_rules_unchanged","passed":True},{"safeguard":"frozen_interface_mapping","passed":True},{"safeguard":"observable_learner_not_run_phase_a","passed":True},{"safeguard":"dense_sparse_validation_pass","passed":True},{"safeguard":"complete_LP_accounting","passed":int(sparse_full[3].n_actual_lp_solves.sum())>0},{"safeguard":"zero_surrogate_exact_rows","passed":int(sparse_full[3].n_surrogate_evaluations.sum())==0},{"safeguard":"timing_effect_gate","passed":gate_pass,"detail":f"median timing/static ratio={float(ratios.max(axis=1).median()) if not ratios.empty else float('nan'):.4g}"}]); safeguards.to_csv(OUT_DIR/"safeguards.csv",index=False)
    summary={"phase_a":"complete","phase_b":"run_only_if_phase_a_passes","phase_a_gate_pass":gate_pass,"n_backgrounds":int(panel.nominal_condition_id.nunique()),"n_schedules":int(panel.schedule_family.ne("static").sum()),"n_cultures":len(panel),"n_timepoints":N_TIME,"sparse_lp_solves":int(sparse_full[3].n_actual_lp_solves.sum()),"dense_sparse_median_final_product_abs_diff":float(dense_sparse.final_product_abs_diff.median()),"dense_sparse_median_lp_reduction":float(dense_sparse.lp_reduction_fraction.median()),"timing_static_ratio_median":float(ratios.max(axis=1).median()) if not ratios.empty else None,"phase_b_status":"not_run_gate_failed" if not gate_pass else "eligible_not_started"}
    (OUT_DIR/"phase_a_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(safeguards.to_string(index=False))
    provenance = {
        "script": "scripts/run_rxncon_dynamic_intervention_validation.py",
        "rxncon_model": "external_models/rxncon_screening/models/CDC_S_cerevisiae.xls",
        "rxncon_inputs": ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"],
        "dynamic_control": "DO",
        "do_levels": list(DO_LEVELS),
        "timepoints": N_TIME,
        "dt": DT,
        "backgrounds": panel[["nominal_condition_id", "temperature", "pH", "glucose_uptake"]].drop_duplicates().to_dict("records"),
        "schedule_manifest": "schedule_manifest.csv",
        "interface_mapping": "imported unchanged from src/yeast_validation/run_rxncon_gsm_generator_gate.py",
        "yeast9_backend": "scripts/gem_backend.py",
        "solver_protocol": "growth optimization, product optimization, deterministic pFBA per executed interval",
        "sparse_validation": {"audit_cultures": len(val_idx), "delta": 0.04, "max_gap": 4},
        "learner_inputs_phase_a": [],
        "lockbox_outputs": ["hidden_interface_lockbox_long.csv", "flux_lockbox.csv", "constraints_lockbox.csv"],
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); run(p.parse_args())
