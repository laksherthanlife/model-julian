from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import audit_open_ended_benchmark_readiness as audit  # noqa: E402
import open_ended_baselines as baselines  # noqa: E402


def synthetic_universe(n: int = 32) -> pd.DataFrame:
    rows = []
    subsystems = [
        "Terpenoid backbone biosynthesis",
        "Oxidative phosphorylation",
        "Transport [c, e]",
        "Glycolysis / gluconeogenesis",
    ]
    for i in range(n):
        rows.append(
            {
                "reaction_id": f"r_{1000 + i:04d}",
                "reaction_name": f"reaction {i}",
                "subsystem": subsystems[i % len(subsystems)],
                "inclusion_rule": "ipp_dmapp_ggpp" if i % 4 == 0 else "transport_exchange" if i % 4 == 2 else "atp_energy",
                "rationale_for_inclusion": "public synthetic universe",
                "risk_level": ["low", "medium", "high"][i % 3],
                "graph_distance_to_product_pathway": i % 4,
                "baseline_flux": float(i % 5),
                "fva_min": 0.0,
                "fva_max": float((i % 4) + 1) / 10.0,
            }
        )
    return pd.DataFrame(rows)


def assert_clean_selector_output(df: pd.DataFrame, batch_size: int) -> None:
    assert len(df) == batch_size
    assert df["candidate_hash"].nunique() == batch_size
    assert not df["uses_hidden_outcomes"].astype(bool).any()
    assert not baselines.FORBIDDEN_OUTCOME_COLUMNS.intersection(df.columns)


def test_minimum_baseline_selectors_are_outcome_blind_and_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines.deployment, "DATA", tmp_path)
    monkeypatch.setattr(baselines.deployment, "PUBLIC_STATIC_SOLVE_ENABLED", False)
    universe = synthetic_universe()
    random_df = baselines.random_valid_baseline(universe, batch_size=4, seed=1)
    space_df = baselines.space_filling_baseline(universe, batch_size=4, seed=2, candidate_pool_size=24)
    static_df = baselines.static_gem_baseline(universe, batch_size=4, seed=3, candidate_pool_size=24)
    assert_clean_selector_output(random_df, 4)
    assert_clean_selector_output(space_df, 4)
    assert_clean_selector_output(static_df, 4)


def test_objective_helper_accepts_public_and_hidden_growth_failure_names():
    public = pd.DataFrame(
        [{"final_product": 1.0, "product_AUC": 2.0, "final_biomass": 3.0, "growth_failure": False}]
    )
    hidden = pd.DataFrame(
        [{"final_product": 1.0, "product_AUC": 2.0, "final_biomass": 3.0, "severe_growth_collapse": True}]
    )
    assert audit.objective_from_public(public).iloc[0] == 1.25
    assert audit.objective_from_public(hidden).iloc[0] == 1.05
