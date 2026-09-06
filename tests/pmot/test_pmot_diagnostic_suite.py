from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pmot.pmot.diagnostic_suite import _fixed_population_force
from pmot.pmot.diagnostic_suite import _production_kernel_detuning_probe
from pmot.pmot.diagnostic_suite import run_ordered_diagnostic_suite
from pmot.pmot.vector_only_trajectories import build_vector_only_trajectory_context


def test_production_kernel_doppler_and_stark_sign_probe() -> None:
    context = build_vector_only_trajectory_context()
    checks, table = _production_kernel_detuning_probe(context)

    assert all(checks.values())
    assert len(table) == 7
    assert set(table["case"]) == {
        "baseline",
        "plus_velocity_zero_shift",
        "minus_velocity_zero_shift",
        "zero_velocity_plus_shift",
        "zero_velocity_minus_shift",
        "plus_velocity_minus_shift",
        "minus_velocity_plus_shift",
    }


def test_fixed_population_two_beam_force_is_balanced_and_damping() -> None:
    context = build_vector_only_trajectory_context()
    beams = [
        beam
        for beam in context.cooling_repump_beams
        if beam.family == "cooling" and beam.axis_name == "vertical_z"
    ]

    zero = _fixed_population_force(context, beams, 0.0)["total_force_n"][2]
    plus = _fixed_population_force(context, beams, 1.0e-3)["total_force_n"][2]
    minus = _fixed_population_force(context, beams, -1.0e-3)["total_force_n"][2]

    assert zero == 0.0
    assert plus < 0.0 < minus
    assert np.isclose(plus, -minus, rtol=1.0e-12, atol=1.0e-30)


def test_ordered_campaign_records_test2_failure_and_stops(tmp_path) -> None:
    output = tmp_path / "diagnostics"
    manifest = run_ordered_diagnostic_suite(
        output_root=output,
        test1_sample_count=21,
        profile_points=21,
        repeat_profile_points=41,
    )

    assert manifest["first_failed_test"] == 2
    index = pd.read_csv(output / "result_index.csv")
    assert index.loc[index["test_number"] == 0, "status"].iloc[0] == (
        "PASS_WITH_QUALIFICATIONS"
    )
    assert index.loc[index["test_number"] == 1, "status"].iloc[0] == (
        "PASS_WITH_QUALIFICATIONS"
    )
    assert index.loc[index["test_number"] == 2, "status"].iloc[0] == "FAIL"
    assert set(index.loc[index["test_number"] >= 3, "status"]) == {"NOT_RUN"}

    result = json.loads(
        (output / "test_02_signed_vector_shift" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["criteria"]["optical_signed_profile_is_odd"] is True
    assert result["criteria"]["named_transition_shift_is_odd_in_fixed_state_labels"] is False
    assert result["criteria"]["refinement_repeats_same_failure"] is True
    assert (
        output
        / "test_02_signed_vector_shift"
        / "figures"
        / "signed_vector_shift_profile.png"
    ).is_file()
