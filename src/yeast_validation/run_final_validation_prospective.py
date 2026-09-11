#!/usr/bin/env python3
"""Final validation prospective runs for data sufficiency and physiology worlds.

The hybrid path is the v2 protocol:
6000 virtual candidates -> top-50 by frozen learner/surrogate -> exact Yeast9
rerank using learned controls -> top-8 hidden exact simulator verification.

Conventional DBTL is reused from the v2 aggregate for baseline/oxidative worlds
when the world and campaign seed match. For new robustness worlds it is run
with the same benchmark policy and compared at the common 8-culture horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gem_backend as gem  # noqa: E402
import gem_gsm_surrogate as sur  # noqa: E402
import gem_gsm_surrogate_mlp as mlp  # noqa: E402
import gem_repaired_prospective_common_v2 as v2  # noqa: E402
import gem_repaired_virtual_scorer as scorer_mod  # noqa: E402
import gem_trainable_state_space as tss  # noqa: E402
import run_prospective_dbtl_benchmark as bench  # noqa: E402
import run_reporter_grounded_hybrid_distillation as prod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_ROOT = ROOT / "results" / "final_validation_prospective"
OUT_DATA = DATA / "final_validation_prospective"
SURROGATE_PATH = ROOT / "results" / "repaired_hybrid_training" / "checkpoints" / "gsm_surrogate_mlp_v1.npz"
V2_AGG_DATA = ROOT / "results" / "repaired_prospective_benchmark_v2_aggregate" / "data"

BENCHMARK_ID = "final_validation_boundary_conditions_v1"
HYBRID_METHOD_ID = "pretrained_hybrid_digital_twin"
CONVENTIONAL_METHOD_ID = "conventional_dbtl"
DATA_SUFF_WORLDS = ["baseline_world", "strong_oxidative_burden_world"]
DATA_SUFF_SEEDS = {"baseline_world": [98001, 98002, 98003], "strong_oxidative_burden_world": [98011, 98012, 98013]}
ROBUSTNESS_WORLDS = ["baseline_world", "strong_oxidative_burden_world", "atp_limited_world", "pathway_bottleneck_damage_world"]
ROBUSTNESS_SEEDS = {
    "baseline_world": [98001, 98002, 98003],
    "strong_oxidative_burden_world": [98011, 98012, 98013],
    "atp_limited_world": [99001, 99002, 99003],
    "pathway_bottleneck_damage_world": [99011, 99012, 99013],
}
VIRTUAL_EVALUATIONS = 6000
SHORTLIST_SIZE = 50
HYBRID_VERIFICATION_BATCH = 8
CONVENTIONAL_BATCHES_FULL = [4, 4, 4, 4]


def append_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["env_mean"], d["env_std"]


def load_scorer(checkpoint: Path, stats_path: Path) -> scorer_mod.RepairedVirtualScorer:
    env_mean, env_std = load_stats(stats_path)
    surrogate = mlp.GSMSurrogateMLP.load(SURROGATE_PATH)
    return scorer_mod.RepairedVirtualScorer.load(checkpoint, env_mean, env_std, surrogate=surrogate)


def patch_bench_outputs() -> None:
    bench.BENCHMARK_ID = BENCHMARK_ID
    bench.DATA = OUT_DATA
    bench.RESULTS = OUT_ROOT
    bench.ROLLOUTS = OUT_ROOT / "on_demand_exact_rollouts"
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    bench.ROLLOUTS.mkdir(parents=True, exist_ok=True)


def real_gsm_replay_score(candidate: dict[str, Any], scorer: scorer_mod.RepairedVirtualScorer, base_model: Any) -> dict[str, Any]:
    env = candidate["environment"]
    env_raw = np.array([[float(env["temperature"]), float(env["pH"]), float(env["DO"])]])
    env_std_in = (env_raw - scorer.env_mean) / scorer.env_std
    cache = tss.forward(scorer.params, env_std_in, scorer_mod.T_LEN)
    c_pre = cache.z[:-1, 0, :] @ scorer.params.Wc + scorer.params.bc
    c_baseline = c_pre * scorer.surrogate.control_std + scorer.surrogate.control_mean
    c_edited, edit_notes = scorer_mod.apply_edits_to_controls(c_baseline, sur.CONTROL_COLUMNS, candidate.get("edits", []))
    clip_lo = scorer.surrogate.control_mean - scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    clip_hi = scorer.surrogate.control_mean + scorer_mod.CONTROL_CLIP_STDS * scorer.surrogate.control_std
    c_final = np.clip(c_edited, clip_lo, clip_hi)

    cfg = gem.GEMCultureConfig(temperature=float(env["temperature"]), pH=float(env["pH"]), DO=float(env["DO"]))
    x = np.zeros(scorer_mod.T_LEN)
    b = np.zeros(scorer_mod.T_LEN)
    x[0], b[0] = scorer_mod.X0, scorer_mod.B0
    infeasible = 0
    for t in range(scorer_mod.T_LEN - 1):
        with base_model as interval_model:
            try:
                interval_model.solver.configuration.timeout = 10
            except Exception:
                pass
            controls = {col: float(c_final[t, i]) for i, col in enumerate(sur.CONTROL_COLUMNS)}
            prod.apply_predicted_interface_controls(interval_model, cfg, controls)
            try:
                flux = gem.solve_staged(interval_model, cfg, gamma=float(np.clip(controls["gamma_growth_fraction"], 0.05, 0.95)), state="balanced")
            except Exception:
                infeasible += 1
                x[t + 1], b[t + 1] = x[t], b[t]
                continue
        biomass_flux = float(flux["biomass_flux"])
        beta_flux = float(flux["beta_carotene_flux"])
        x[t + 1] = max(1e-9, x[t] + scorer_mod.DT_REAL * biomass_flux * x[t])
        b[t + 1] = max(0.0, b[t] + scorer_mod.DT_REAL * beta_flux * x[t] - scorer_mod.DT_REAL * scorer_mod.BETA_DEG_BASE * b[t])
    return {
        "candidate_hash": candidate["candidate_hash"],
        "rerank_score": float(b[-1]),
        "rerank_productivity": float(b[-1] / max(scorer_mod.T_LEN * scorer_mod.DT_REAL, 1e-9)),
        "rerank_final_biomass": float(x[-1]),
        "n_infeasible_intervals": infeasible,
        "edit_notes": ";".join(edit_notes),
    }


def hybrid_campaign(
    *,
    checkpoint: Path,
    stats_path: Path,
    training_cultures_n: int,
    subsample_seed: int,
    experiment_group: str,
    world_id: str,
    campaign_seed: int,
    library: pd.DataFrame,
    base_model: Any,
    force: bool,
) -> dict[str, Any]:
    campaign_id = f"final_{experiment_group}_{world_id}_seed_{campaign_seed}_N{training_cultures_n}_sub{subsample_seed}"
    done_path = OUT_DATA / "final_validation_hybrid_campaign_summary.csv"
    if done_path.exists() and not force:
        existing = pd.read_csv(done_path)
        mask = existing["campaign_id"].eq(campaign_id)
        if mask.any():
            row = existing[mask].iloc[-1].to_dict()
            print(f"reuse hybrid campaign {campaign_id}")
            return row

    scorer = load_scorer(checkpoint, stats_path)
    factory = v2.make_factory(library, campaign_id, campaign_seed)
    shortlist, virtual = v2.generate_hybrid_shortlist(
        factory,
        VIRTUAL_EVALUATIONS,
        SHORTLIST_SIZE,
        set(),
        library,
        scorer,
        campaign_id,
        world_id,
    )
    virtual["benchmark_id"] = BENCHMARK_ID
    virtual["experiment_group"] = experiment_group
    virtual["training_cultures_n"] = training_cultures_n
    virtual["subsample_seed"] = subsample_seed
    virtual_path = OUT_DATA / f"virtual_ledger__{campaign_id}.csv"
    virtual.to_csv(virtual_path, index=False)

    rerank_rows = []
    rerank_path = OUT_DATA / "final_validation_rerank_ledger.csv"
    existing_rerank = pd.read_csv(rerank_path) if rerank_path.exists() else pd.DataFrame()
    for rank, cand in enumerate(shortlist, start=1):
        prior = pd.DataFrame()
        if not existing_rerank.empty:
            prior = existing_rerank[
                existing_rerank["campaign_id"].eq(campaign_id)
                & existing_rerank["candidate_hash"].eq(cand["candidate_hash"])
                & existing_rerank["training_cultures_n"].astype(int).eq(training_cultures_n)
                & existing_rerank["subsample_seed"].astype(int).eq(subsample_seed)
            ]
        if not prior.empty and not force:
            replay = prior.iloc[-1].to_dict()
            print(f"reuse rerank {campaign_id} {rank}/{SHORTLIST_SIZE}", flush=True)
        else:
            replay = real_gsm_replay_score(cand, scorer, base_model)
            replay.update(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "experiment_group": experiment_group,
                    "campaign_id": campaign_id,
                    "world_id": world_id,
                    "campaign_seed": campaign_seed,
                    "training_cultures_n": training_cultures_n,
                    "subsample_seed": subsample_seed,
                    "candidate_id": cand["candidate_id"],
                    "surrogate_shortlist_rank": rank,
                    "surrogate_score": float(cand["_surrogate_score"]),
                }
            )
            append_frame(rerank_path, [replay])
            print(f"reranked {campaign_id} {rank}/{SHORTLIST_SIZE} score={float(replay['rerank_score']):.6g}", flush=True)
        cand["_rerank_score"] = replay["rerank_score"]
        rerank_rows.append(replay)

    shortlist.sort(key=lambda c: float(c["_rerank_score"]), reverse=True)
    selected = shortlist[:HYBRID_VERIFICATION_BATCH]
    selected_hashes = {c["candidate_hash"] for c in selected}
    rerank_rank = {c["candidate_hash"]: i for i, c in enumerate(shortlist, start=1)}
    virtual["rerank_score"] = virtual["candidate_hash"].map({r["candidate_hash"]: r["rerank_score"] for r in rerank_rows})
    virtual["rerank_rank"] = virtual["candidate_hash"].map(rerank_rank)
    virtual["entered_final_top8"] = virtual["candidate_hash"].isin(selected_hashes)
    virtual["selected_for_exact_verification"] = virtual["entered_final_top8"]
    virtual["exact_outcome_accessed"] = False
    virtual.to_csv(virtual_path, index=False)

    exact_rows = []
    for culture_index, cand in enumerate(selected, start=1):
        result = bench.run_on_demand_exact(
            cand,
            campaign_id=campaign_id,
            method_id=HYBRID_METHOD_ID,
            world_id=world_id,
            stage_index=1,
            culture_index=culture_index,
            library=library,
            mock=False,
            fresh_exact=False,
        )
        result.update({"experiment_group": experiment_group, "training_cultures_n": training_cultures_n, "subsample_seed": subsample_seed})
        exact_rows.append(result)

    exact_df = pd.DataFrame(exact_rows)
    valid = exact_df[exact_df["biologically_valid"].astype(bool)]
    best_titer = valid.loc[valid["final_product"].idxmax()] if not valid.empty else exact_df.iloc[0]
    best_prod = valid.loc[valid["productivity"].idxmax()] if not valid.empty else exact_df.iloc[0]
    summary = {
        "benchmark_id": BENCHMARK_ID,
        "experiment_group": experiment_group,
        "campaign_id": campaign_id,
        "world_id": world_id,
        "campaign_seed": campaign_seed,
        "method_id": HYBRID_METHOD_ID,
        "training_cultures_n": training_cultures_n,
        "subsample_seed": subsample_seed,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "physical_cultures": HYBRID_VERIFICATION_BATCH,
        "virtual_candidates": VIRTUAL_EVALUATIONS,
        "shortlist_size": SHORTLIST_SIZE,
        "reranked_with_exact_yeast9": True,
        "best_final_product": float(best_titer["final_product"]),
        "best_productivity": float(best_prod["productivity"]),
        "best_product_AUC": float(valid["product_AUC"].max()) if not valid.empty else float("nan"),
        "best_candidate_hash": str(best_titer["candidate_hash"]),
        "exact_lp_solves": int(exact_df["n_actual_lp_solves"].sum()),
        "mock_mode": False,
    }
    append_frame(done_path, [summary])
    return summary


def existing_conventional_common8(world_id: str, seed: int) -> dict[str, Any] | None:
    path = V2_AGG_DATA / "prospective_best_so_far_by_culture.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cid = f"prospective_{world_id}_seed_{seed}"
    mask = df["campaign_id"].eq(cid) & df["world_id"].eq(world_id) & df["method_id"].eq(CONVENTIONAL_METHOD_ID) & (df["physical_culture_index"].astype(int) <= 8)
    sub = df[mask].sort_values("physical_culture_index")
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {
        "benchmark_id": "repaired_hybrid_prospective_dbtl_v2_reranked",
        "source": "reused_existing_v2_conventional_common8",
        "campaign_id": cid,
        "world_id": world_id,
        "campaign_seed": seed,
        "method_id": CONVENTIONAL_METHOD_ID,
        "physical_cultures": 8,
        "best_final_product": float(row["best_final_product_so_far"]),
        "best_productivity": float(row["best_productivity_so_far"]),
        "best_product_AUC": float(row["best_product_AUC_so_far"]),
    }


def run_conventional_if_needed(world_id: str, seed: int, library: pd.DataFrame, force: bool) -> dict[str, Any]:
    reused = existing_conventional_common8(world_id, seed)
    if reused is not None:
        return reused
    out_path = OUT_DATA / "final_validation_conventional_common8.csv"
    cid = f"prospective_{world_id}_seed_{seed}"
    if out_path.exists() and not force:
        existing = pd.read_csv(out_path)
        mask = existing["campaign_id"].eq(cid)
        if mask.any():
            print(f"reuse conventional campaign {cid}")
            return existing[mask].iloc[-1].to_dict()
    old_methods = list(bench.PRIMARY_METHODS)
    try:
        bench.PRIMARY_METHODS = [CONVENTIONAL_METHOD_ID]
        bench.run_campaign(
            world_id=world_id,
            seed=seed,
            library=library,
            teacher_lookup=bench.load_teacher_final_lookup(),
            conventional_batches=CONVENTIONAL_BATCHES_FULL,
            hybrid_batch=0,
            virtual_evaluations=0,
            mock=False,
            fresh_exact=False,
        )
    finally:
        bench.PRIMARY_METHODS = old_methods
    by_culture = pd.read_csv(OUT_DATA / "prospective_exact_simulator_call_ledger.csv")
    sub = by_culture[by_culture["campaign_id"].eq(cid) & by_culture["world_id"].eq(world_id) & by_culture["method_id"].eq(CONVENTIONAL_METHOD_ID)].sort_values("culture_index")
    common = sub.iloc[:8].copy()
    common["best_final_product_so_far"] = common["final_product"].cummax()
    common["best_productivity_so_far"] = common["productivity"].cummax()
    common["best_product_AUC_so_far"] = common["product_AUC"].cummax()
    row = common.iloc[-1]
    summary = {
        "benchmark_id": BENCHMARK_ID,
        "source": "new_final_validation_conventional_common8_from_full16_run",
        "campaign_id": cid,
        "world_id": world_id,
        "campaign_seed": seed,
        "method_id": CONVENTIONAL_METHOD_ID,
        "physical_cultures": 8,
        "best_final_product": float(row["best_final_product_so_far"]),
        "best_productivity": float(row["best_productivity_so_far"]),
        "best_product_AUC": float(row["best_product_AUC_so_far"]),
    }
    append_frame(out_path, [summary])
    return summary


def selected_training_rows(limit_counts: list[int] | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(DATA / "final_data_sufficiency_training_manifest.csv")
    if limit_counts:
        manifest = manifest[manifest["training_cultures_n"].isin(limit_counts)]
    return manifest.sort_values(["training_cultures_n", "subsample_seed"], ascending=[False, True]).reset_index(drop=True)


def run_data_sufficiency(args: argparse.Namespace, library: pd.DataFrame, base_model: Any) -> None:
    rows = selected_training_rows(args.counts)
    for _, ck in rows.iterrows():
        n = int(ck["training_cultures_n"])
        subs = int(ck["subsample_seed"])
        for world in DATA_SUFF_WORLDS:
            for seed in DATA_SUFF_SEEDS[world]:
                conv = run_conventional_if_needed(world, seed, library, force=args.force)
                hyb = hybrid_campaign(
                    checkpoint=ROOT / str(ck["checkpoint"]),
                    stats_path=ROOT / str(ck["normalization_stats"]),
                    training_cultures_n=n,
                    subsample_seed=subs,
                    experiment_group="data_sufficiency",
                    world_id=world,
                    campaign_seed=seed,
                    library=library,
                    base_model=base_model,
                    force=args.force,
                )
                append_comparison("data_sufficiency", hyb, conv)


def run_robustness(args: argparse.Namespace, library: pd.DataFrame, base_model: Any) -> None:
    rows = selected_training_rows([77])
    ck = rows.iloc[0]
    for world in ROBUSTNESS_WORLDS:
        for seed in ROBUSTNESS_SEEDS[world]:
            conv = run_conventional_if_needed(world, seed, library, force=args.force)
            hyb = hybrid_campaign(
                checkpoint=ROOT / str(ck["checkpoint"]),
                stats_path=ROOT / str(ck["normalization_stats"]),
                training_cultures_n=int(ck["training_cultures_n"]),
                subsample_seed=int(ck["subsample_seed"]),
                experiment_group="physiology_robustness",
                world_id=world,
                campaign_seed=seed,
                library=library,
                base_model=base_model,
                force=args.force,
            )
            append_comparison("physiology_robustness", hyb, conv)


def append_comparison(experiment_group: str, hyb: dict[str, Any], conv: dict[str, Any]) -> None:
    row = {
        "benchmark_id": BENCHMARK_ID,
        "experiment_group": experiment_group,
        "world_id": hyb["world_id"],
        "campaign_seed": int(hyb["campaign_seed"]),
        "training_cultures_n": int(hyb["training_cultures_n"]),
        "subsample_seed": int(hyb["subsample_seed"]),
        "hybrid_campaign_id": hyb["campaign_id"],
        "conventional_campaign_id": conv["campaign_id"],
        "conventional_source": conv.get("source", ""),
        "common_physical_cultures": 8,
        "hybrid_best_final_product": float(hyb["best_final_product"]),
        "conventional_best_final_product": float(conv["best_final_product"]),
        "delta_final_product": float(hyb["best_final_product"]) - float(conv["best_final_product"]),
        "hybrid_best_productivity": float(hyb["best_productivity"]),
        "conventional_best_productivity": float(conv["best_productivity"]),
        "delta_productivity": float(hyb["best_productivity"]) - float(conv["best_productivity"]),
        "hybrid_win_final_product": float(hyb["best_final_product"]) > float(conv["best_final_product"]) + 1e-12,
    }
    append_frame(OUT_DATA / "final_validation_common8_comparisons.csv", [row])


def summarize() -> None:
    path = OUT_DATA / "final_validation_common8_comparisons.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).drop_duplicates(["experiment_group", "world_id", "campaign_seed", "training_cultures_n", "subsample_seed"], keep="last")
    rows = []
    for keys, group in df.groupby(["experiment_group", "training_cultures_n"]):
        exp, n = keys
        vals = group["delta_final_product"].to_numpy(float)
        rng = np.random.default_rng(811)
        boots = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(2000)] if len(vals) else [np.nan]
        rows.append(
            {
                "experiment_group": exp,
                "training_cultures_n": int(n),
                "n_campaigns": int(len(group)),
                "n_worlds": int(group["world_id"].nunique()),
                "mean_hybrid_best_final_product": float(group["hybrid_best_final_product"].mean()),
                "mean_conventional_best_final_product": float(group["conventional_best_final_product"].mean()),
                "mean_delta_final_product": float(np.mean(vals)),
                "bootstrap_ci_low": float(np.percentile(boots, 2.5)),
                "bootstrap_ci_high": float(np.percentile(boots, 97.5)),
                "win_count": int(group["hybrid_win_final_product"].sum()),
                "loss_count": int((~group["hybrid_win_final_product"].astype(bool)).sum()),
                "win_rate": float(group["hybrid_win_final_product"].mean()),
            }
        )
    pd.DataFrame(rows).sort_values(["experiment_group", "training_cultures_n"], ascending=[True, False]).to_csv(OUT_DATA / "final_validation_summary_by_training_culture_count.csv", index=False)
    world_rows = []
    for keys, group in df.groupby(["experiment_group", "world_id", "training_cultures_n"]):
        exp, world, n = keys
        world_rows.append(
            {
                "experiment_group": exp,
                "world_id": world,
                "training_cultures_n": int(n),
                "n_campaigns": int(len(group)),
                "mean_delta_final_product": float(group["delta_final_product"].mean()),
                "win_count": int(group["hybrid_win_final_product"].sum()),
                "loss_count": int((~group["hybrid_win_final_product"].astype(bool)).sum()),
            }
        )
    pd.DataFrame(world_rows).sort_values(["experiment_group", "world_id", "training_cultures_n"]).to_csv(OUT_DATA / "final_validation_summary_by_world.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "data-sufficiency", "robustness"], default="all")
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    library = v2.build_common_intervention_library()
    patch_bench_outputs()
    cobra = gem.require_cobra()
    source_path = gem.configured_gem_path(None)
    base_model_raw = gem.load_model(cobra, source_path)
    base_model, _manifest = gem.install_beta_carotene_pathway(base_model_raw)

    tic = time.perf_counter()
    if args.mode in ("all", "data-sufficiency"):
        run_data_sufficiency(args, library, base_model)
    if args.mode in ("all", "robustness"):
        run_robustness(args, library, base_model)
    summarize()
    print(f"prospective final validation complete in {time.perf_counter() - tic:.1f}s")


if __name__ == "__main__":
    main()
