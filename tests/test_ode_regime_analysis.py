import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "yeast_validation"))

import analyze_ode_regime_discovery as discovery
import ode_regime_core as ode
import run_ode_inverse_design as design


def test_cluster_metrics_are_label_free_until_evaluation():
    X = {
        "toy": np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
                [9.0, 0.0],
                [9.1, 0.1],
            ]
        )
    }
    regimes = np.array(["a", "a", "b", "b", "c", "c"])
    out = discovery.evaluate_clusters(X, regimes)
    assert {"ari", "nmi", "purity", "silhouette"}.issubset(set(out.columns))
    assert out["purity"].between(0, 1).all()


def test_inverse_design_objectives_and_bounds():
    diag = pd.DataFrame(
        {
            "final_product": [1.0, 0.5],
            "auc_product": [10.0, 9.0],
            "late_decline": [0.0, 0.4],
            "integrated_stress": [5.0, 20.0],
            "min_energy": [0.3, 0.05],
            "final_pathway_capacity": [0.2, 0.01],
            "endpoint_biomass": [0.2, 0.01],
        }
    )
    assert design.objective_value(diag, "sustained_production")[0] > design.objective_value(diag, "sustained_production")[1]
    assert design.objective_value(diag, "physiologically_constrained")[0] > design.objective_value(diag, "physiologically_constrained")[1]
    env = ode.unit_to_env(np.array([[0.0, 0.5, 1.0]]))[0]
    assert 27.0 <= env[0] <= 33.0
    assert 4.5 <= env[1] <= 5.5
    assert 20.0 <= env[2] <= 80.0


def test_boundary_search_reports_simulator_verification_columns():
    source = Path(design.__file__).read_text()
    assert "verify_candidate" in source
    assert "simulator_confirms_qualitative_change" in source
    assert "ode.simulate_environment" in source
