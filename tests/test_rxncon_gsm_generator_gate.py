import numpy as np

import run_rxncon_gsm_generator_gate as gate


def test_environment_contract_is_learner_visible_only():
    contract = gate.build_environment_contract()
    assert contract["learner_input"].all()
    assert contract["observable_to_learner"].all()
    assert not contract["hidden_oracle"].any()


def test_lockbox_marks_hidden_generator_quantities_unobservable():
    lockbox = gate.lockbox_spec()
    hidden = lockbox[lockbox["learner_facing_allowed"] == False]  # noqa: E712
    assert {"raw_rxncon_node", "regulatory_to_GSM_interface_state", "exact_flux"}.issubset(
        set(hidden["variable_class"])
    )


def test_reporters_are_mixed_and_not_one_to_one_hidden_states():
    reporters = gate.reporter_specs()
    assert len(reporters) >= 3
    for reporter in reporters:
        assert len(reporter.contributors) >= 4
        assert len(set(reporter.contributors)) == len(reporter.contributors)
        assert len(reporter.weights) == len(reporter.contributors)


def test_interface_has_multiple_independent_control_channels():
    channels = gate.interface_channels()
    assert len(channels) >= 8
    targets = {channel.target_gsm_quantity for channel in channels}
    assert {"r_1714 lower bound", "r_1992 lower bound", "r_4046 lower bound"}.issubset(targets)
    assert {"PSY_capacity", "DES_capacity", "CYC_capacity"}.issubset({c.name for c in channels})


def test_rxncon_rollout_is_deterministic_for_fixed_panel():
    model = gate.load_rxncon_model()
    panel = gate.build_environment_panel(4)
    force = gate.force_array_for_panel(model, panel)
    out1, _runtime1 = gate.rxncon_screen.simulate_panel(model, force)
    out2, _runtime2 = gate.rxncon_screen.simulate_panel(model, force)
    assert np.array_equal(out1, out2)
