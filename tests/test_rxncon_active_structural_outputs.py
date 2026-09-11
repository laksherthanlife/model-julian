from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data" / "rxncon_active_structural_validation"


def test_active_rxncon_outputs_exist_and_are_complete():
    required = [
        "culture_manifest.csv",
        "observable_trajectories.csv",
        "hidden_rxncon.npz",
        "hidden_module_trajectories_wide.csv",
        "hidden_interface_trajectories_wide.csv",
        "fluxes.csv",
        "constraints.csv",
        "solver_accounting.csv",
        "complexity_ladder.csv",
        "model_metrics.csv",
        "safeguards.csv",
    ]
    for rel in required:
        assert (DATA / rel).exists(), rel
    obs = pd.read_csv(DATA / "observable_trajectories.csv")
    assert obs["culture_id"].nunique() == 125
    assert obs["time_index"].nunique() == 49


def test_active_inputs_vary_and_lockbox_stays_out_of_observables():
    manifest = pd.read_csv(DATA / "culture_manifest.csv")
    for col in ["Nutrients", "Pheromone", "HU", "LatA", "Nocodazole"]:
        assert manifest[col].nunique() == 2
    obs_cols = set(pd.read_csv(DATA / "observable_trajectories.csv", nrows=1).columns)
    forbidden = {
        "carbon_uptake_capacity",
        "oxygen_capacity",
        "atp_maintenance_multiplier",
        "growth_allocation_gamma",
        "stress_maintenance_load",
        "precursor_availability",
        "resource_translation_capacity",
        "PSY_capacity",
        "DES_capacity",
        "CYC_capacity",
        "biomass_flux",
        "beta_carotene_flux",
    }
    assert not (obs_cols & forbidden)


def test_active_structural_safeguards_pass():
    safeguards = pd.read_csv(DATA / "safeguards.csv")
    failed = safeguards.loc[~safeguards["passed"].astype(bool), "safeguard"].tolist()
    assert failed == []
