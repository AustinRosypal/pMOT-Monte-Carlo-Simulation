from __future__ import annotations

import numpy as np
import pytest

from pmot.pmot.temporary_tests.effective_field_quantization_axis_diagnostic import (
    calculate_beamwise_diagnostic,
)
from pmot.pmot.temporary_tests.run_local_field_axis_trajectories import (
    polarization_history,
    reconstruct_component_fields_t,
)
from pmot.pmot.vector_only_trajectories import (
    build_vector_only_trajectory_context,
    inward_launch_state,
    simulate_vector_only_pmot_trajectory,
    vector_only_trajectory_observable,
)
from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryConfig


def test_symmetric_off_center_point_has_expected_axis_and_normalized_weights():
    frame, summary = calculate_beamwise_diagnostic()

    expected_axis = np.asarray((1.0, 1.0, -1.0)) / np.sqrt(3.0)
    assert summary["quantization_axis_from_vector_proxy"] == pytest.approx(
        expected_axis,
        abs=1.0e-12,
    )
    assert summary["net_transition_equivalent_field_g"] == pytest.approx(
        (0.19669074196089747, 0.19669074196089747, -0.19669074196089747),
        rel=2.0e-9,
    )
    weights = frame[
        ["sigma_plus_fraction", "pi_fraction", "sigma_minus_fraction"]
    ].to_numpy()
    assert np.sum(weights, axis=1) == pytest.approx(np.ones(6), abs=2.0e-15)
    assert frame.loc[0, "sigma_plus_fraction"] == pytest.approx(
        0.6220084679281461
    )
    assert frame.loc[0, "pi_fraction"] == pytest.approx(1.0 / 3.0)
    assert frame.loc[0, "sigma_minus_fraction"] == pytest.approx(
        0.044658198738520456
    )


def test_equivalent_field_scales_linearly_with_uniform_path_power():
    _, lower = calculate_beamwise_diagnostic(power_w_per_path=0.010)
    _, upper = calculate_beamwise_diagnostic(power_w_per_path=0.020)

    assert upper["net_transition_equivalent_field_g"] == pytest.approx(
        2.0 * np.asarray(lower["net_transition_equivalent_field_g"]),
        rel=2.0e-12,
        abs=1.0e-14,
    )
    assert upper["quantization_axis_from_vector_proxy"] == pytest.approx(
        lower["quantization_axis_from_vector_proxy"],
        abs=1.0e-12,
    )


def test_short_trajectory_reconstructs_field_and_normalizes_every_beam():
    context = build_vector_only_trajectory_context(
        trapping_power_w_per_path=0.010,
    )
    record = simulate_vector_only_pmot_trajectory(
        inward_launch_state(),
        duration_s=5.0e-6,
        context=context,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=2.5e-6,
            include_diffusion=False,
            escape_radius_m=30.0e-3,
        ),
    )

    component_fields, record_error, component_sum_roundoff = (
        reconstruct_component_fields_t(record, context)
    )
    weights, beam_records, normalization_error = polarization_history(
        record,
        context,
    )

    assert component_fields.shape == (3, 6, 3)
    assert weights.shape == (3, 18, 3)
    assert len(beam_records) == 18
    assert record_error < 2.0e-14
    assert component_sum_roundoff < 2.0e-14
    assert normalization_error < 2.0e-14
    assert np.asarray(record.rate_equation.magnetic_fields_t) == pytest.approx(
        np.zeros((3, 3)),
        abs=0.0,
    )


def test_vector_shift_produces_odd_restoring_static_force_on_x():
    enabled = build_vector_only_trajectory_context()
    disabled = build_vector_only_trajectory_context(
        trapping_power_w_per_path=0.0,
    )
    coordinates_m = (-0.5e-3, 0.5e-3)
    enabled_forces = []
    disabled_forces = []
    for coordinate_m in coordinates_m:
        axis = (np.sign(coordinate_m), 0.0, 0.0)
        enabled_forces.append(
            vector_only_trajectory_observable(
                enabled,
                (coordinate_m, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                axis,
            ).rate_equation.force_n[0]
        )
        disabled_forces.append(
            vector_only_trajectory_observable(
                disabled,
                (coordinate_m, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                axis,
            ).rate_equation.force_n[0]
        )

    assert enabled_forces[0] > 0.0
    assert enabled_forces[1] < 0.0
    assert enabled_forces[0] == pytest.approx(-enabled_forces[1], rel=1.0e-12)
    assert np.asarray(disabled_forces) == pytest.approx(
        np.zeros(2),
        abs=1.0e-33,
    )
