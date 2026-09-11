from pathlib import Path

import pandas as pd
import pytest


DATA = Path(__file__).resolve().parents[1] / "data" / "rxncon_process_variability"


@pytest.mark.skipif(not DATA.exists(), reason="process-variability outputs not generated")
def test_process_variability_observable_tables_do_not_expose_lockbox_columns():
    forbidden = {
        "T_real",
        "pH_real",
        "DO_real",
        "carbon_real",
        "nitrogen_real",
        "kLa_real",
        "batch_id",
        "plate_id",
        "reader_id",
        "position_id",
        "biomass_flux",
        "beta_carotene_flux",
        "GSM_interface",
    }
    for world in ["perfect_control", "random_process", "random_systematic_process"]:
        obs = pd.read_csv(DATA / f"{world}_observable_trajectories.csv", nrows=5)
        assert forbidden.isdisjoint(obs.columns)
        assert {"T_set", "pH_set", "DO_set", "B_total_observed", "X_observed"}.issubset(obs.columns)


@pytest.mark.skipif(not DATA.exists(), reason="process-variability outputs not generated")
def test_process_variability_has_replicates_and_systematic_holdout():
    design = pd.read_csv(DATA / "process_variability_culture_design.csv")
    assert design["culture_id"].nunique() == 375
    reps = design.groupby(["world", "nominal_environment_id"])["culture_id"].nunique()
    assert reps.max() >= 3
    systematic = design[design["world"].eq("random_systematic_process")]
    assert systematic[systematic["batch_id"].eq("batch_4")]["split"].eq("systematic_holdout").all()


@pytest.mark.skipif(not DATA.exists(), reason="process-variability outputs not generated")
def test_process_variability_safeguards_pass():
    safeguards = pd.read_csv(DATA / "process_variability_safeguards.csv")
    assert safeguards["passed"].all()

