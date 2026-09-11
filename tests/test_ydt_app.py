from pathlib import Path

import numpy as np

from webapp.ydt_app.core import (
    build_culture_manifest,
    infer_column_roles,
    load_dataset,
    predict_trajectory,
    predict_with_observations,
    screen_candidates,
    split_cultures,
    train_candidate_models,
    validate_dataset,
    verify_with_yeast9,
)


DEMO = Path(__file__).parents[1] / "webapp" / "demo" / "yeast_demo.csv"


def loaded():
    df, digest = load_dataset(DEMO.read_text(), "demo.csv")
    suggestions = infer_column_roles(df)
    mapping = {role: [col for col, suggested in suggestions.items() if suggested == role] for role in (
        "culture_id", "time", "environment", "product", "biomass", "reporter", "genotype", "ignore"
    )}
    return df, mapping, digest


def test_schema_inference_and_validation():
    df, mapping, digest = loaded()
    result = validate_dataset(df, mapping)
    assert digest
    assert result["ok"]
    assert result["metrics"]["cultures"] == 6
    assert len(mapping["environment"]) == 3
    assert len(mapping["reporter"]) == 6


def test_culture_level_split_has_no_row_leakage():
    df, mapping, _ = loaded()
    manifest = build_culture_manifest(df, mapping)
    splits = split_cultures(manifest, seed=7)
    sets = [set(splits[name]) for name in ("train", "validation", "test")]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set().union(*sets) == set(df["culture_id"])


def test_training_prediction_and_screen_smoke():
    df, mapping, _ = loaded()
    splits = split_cultures(build_culture_manifest(df, mapping), seed=42)
    training = train_candidate_models(df, mapping, splits)
    assert training["best"]["test"]["trajectory_nrmse"] >= 0
    prediction = predict_trajectory(training["best"], {"temperature": 30, "pH": 5, "DO": 40})
    assert len(prediction["predicted_product"]) == training["target_points"]
    rows = screen_candidates(training["best"], {"temperature": {"min": 28, "max": 32}, "pH": {"min": 4.5, "max": 5.5}, "DO": {"min": 30, "max": 70}}, 7)
    assert len(rows) == 7
    assert rows[0]["final_product"] >= rows[-1]["final_product"]


def test_missing_reporter_is_warning_not_failure():
    df, mapping, _ = loaded()
    mapping["reporter"] = []
    result = validate_dataset(df, mapping)
    assert result["ok"]
    assert any("No reporter" in item["message"] for item in result["warnings"])


def test_hidden_simulator_column_is_flagged():
    df, mapping, _ = loaded()
    df["rxncon_state"] = np.arange(len(df))
    mapping["ignore"].append("rxncon_state")
    result = validate_dataset(df, mapping)
    assert result["ok"]
    assert not any("simulator-only" in item["message"] for item in result["warnings"])


def test_exact_yeast9_boundary_is_explicit():
    result = verify_with_yeast9()
    assert result["available"] is False
    assert "not exposed" in result["reason"]


def test_observation_conditioned_model_and_partial_update_smoke():
    df, mapping, _ = loaded()
    splits = split_cultures(build_culture_manifest(df, mapping), seed=42)
    training = train_candidate_models(df, mapping, splits)
    observers = [c for c in training["candidates"] if c["model"].get("kind") == "observable_observer"]
    assert observers
    model = observers[0]
    result = predict_with_observations(model, {"temperature": 30, "pH": 5, "DO": 40}, {
        "cutoff_fraction": 0.4,
        "product": [0.0, 0.04, 0.09],
        "biomass": [0.10, 0.12, 0.15],
        "R_stress": [0.08, 0.10, 0.14],
        "R_resource": [0.22, 0.24, 0.28],
        "R_checkpoint": [0.12, 0.14, 0.18],
        "R_pathway_capacity": [0.18, 0.20, 0.23],
        "R_morphology": [0.15, 0.16, 0.19],
        "R_energy": [0.21, 0.24, 0.29],
    })
    assert result["observed_cutoff_fraction"] > 0
    assert len(result["predicted_product"]) == training["target_points"]


def test_categorical_genotype_is_encoded_for_direct_prediction():
    df, mapping, _ = loaded()
    df["strain"] = ["reference" if i % 2 else "edited" for i in range(len(df))]
    mapping["genotype"] = ["strain"]
    splits = split_cultures(build_culture_manifest(df, mapping), seed=42)
    training = train_candidate_models(df, mapping, splits)
    genotype_model = next(x for x in training["candidates"] if x["name"] == "Environment + genotype")
    result = predict_trajectory(genotype_model, {"temperature": 30, "pH": 5, "DO": 40, "strain": "edited"})
    assert len(result["predicted_product"]) == training["target_points"]
