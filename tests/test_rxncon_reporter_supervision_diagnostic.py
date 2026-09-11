from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data" / "rxncon_reporter_supervision_diagnostic"


def test_reporter_diagnostic_outputs_exist():
    required = [
        "loss_gradient_audit.csv",
        "lambda_sweep_metrics.csv",
        "training_dynamics_and_gradients.csv",
        "reporter_ablation_metrics.csv",
        "latent_dim_metrics.csv",
        "sample_size_metrics.csv",
        "reporter_relevance.csv",
        "reporter_failure_decision_tree.csv",
        "required_reporter_diagnostic_summary_table.csv",
    ]
    for rel in required:
        assert (DATA / rel).exists(), rel


def test_reporter_weight_grid_and_core_comparison_present():
    metrics = pd.read_csv(DATA / "lambda_sweep_metrics.csv")
    lambdas = set(metrics.loc[metrics["condition_family"].eq("lambda_sweep"), "lambda_R"].round(3))
    assert {0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}.issubset(lambdas)
    product = metrics[
        metrics["condition_family"].eq("lambda_sweep")
        & metrics["metric_type"].eq("prediction")
        & metrics["channel"].eq("B_total")
        & metrics["split"].ne("train")
        & metrics["checkpoint_rule"].eq("best_product_val")
    ]
    by_lambda = product.groupby("lambda_R")["nrmse_train_std"].mean()
    assert by_lambda.loc[0.0] <= by_lambda.loc[1.0]


def test_decision_tree_contains_selected_classifications():
    decision = pd.read_csv(DATA / "reporter_failure_decision_tree.csv")
    selected = set(decision.loc[decision["selected"].astype(bool), "classification"])
    assert "reporter_overweighting" in selected
    assert "reporter_information_not_product_relevant" in selected
