"""Checks for the plotting-free vector-only pMOT trajectory engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryRecord
from pmot.pmot.vector_only_trajectories import PMOTBeamHelicities
from pmot.pmot.vector_only_trajectories import build_vector_only_apparatus
from pmot.pmot.vector_only_trajectories import build_vector_only_trajectory_context
from pmot.pmot.vector_only_trajectories import classify_vector_only_trajectory
from pmot.pmot.vector_only_trajectories import inward_launch_state
from pmot.pmot.vector_only_trajectories import simulate_vector_only_pmot_trajectory
from pmot.pmot.vector_only_trajectories import vector_only_trajectory_dataframe
from pmot.pmot.vector_only_trajectories import vector_only_trajectory_observable


@pytest.fixture(scope="module")
def context():
    return build_vector_only_trajectory_context(
        trapping_power_w_per_path=1.0e-6,
    )


def test_default_helicities_are_individually_resolved_known_working_tuple(
    context,
) -> None:
    assert context.trapping_code == "++-++-"
    expected = {"horizontal_x": "sigma+", "horizontal_y": "sigma+", "vertical_z": "sigma-"}
    for beam in context.cooling_repump_beams:
        assert beam.circular_polarization == expected[beam.axis_name]
    for beam in context.trapping_beams:
        assert beam.helicity == expected[beam.axis_name]


def test_each_780_family_and_trapping_direction_can_be_changed() -> None:
    helicities = PMOTBeamHelicities(
        cooling_incident_xyz=("pi", "sigma-", "sigma+"),
        cooling_retro_xyz=("sigma-", "pi", "sigma+"),
        repump_incident_xyz=("sigma-", "sigma+", "pi"),
        repump_retro_xyz=("sigma+", "sigma-", "pi"),
        trapping_incident_xyz=("sigma-", "sigma+", "sigma-"),
        trapping_retro_xyz=("sigma+", "sigma-", "sigma+"),
    )
    configured = build_vector_only_trajectory_context(
        helicities=helicities,
        trapping_power_w_per_path=0.0,
    )
    assert configured.trapping_code == "-+-+-+"
    for beam in configured.cooling_repump_beams:
        index = ("horizontal_x", "horizontal_y", "vertical_z").index(beam.axis_name)
        values = getattr(
            helicities,
            f"{beam.family}_{beam.propagation_sense}_xyz",
        )
        assert beam.circular_polarization == values[index]
    for beam in configured.trapping_beams:
        values = getattr(helicities, f"trapping_{beam.propagation_sense}_xyz")
        index = ("horizontal_x", "horizontal_y", "vertical_z").index(beam.axis_name)
        assert beam.helicity == values[index]


def test_notebook_apparatus_controls_propagate_into_context() -> None:
    apparatus = build_vector_only_apparatus(
        cooling_power_w_per_beam=25.0e-3,
        repump_power_w_per_beam=0.2e-3,
        cooling_detuning_hz=-18.0e6,
        beam_diameter_m=14.0e-3,
        trapping_wavelength_m=1529.268881e-9,
        trapping_focus_offset_m=8.0e-3,
        trapping_incident_waist_radius_m=5.0e-6,
        trapping_retro_waist_radius_m=6.0e-6,
    )
    configured = build_vector_only_trajectory_context(
        apparatus=apparatus,
        trapping_power_w_per_path=2.0e-3,
    )
    cooling = [beam for beam in configured.cooling_repump_beams if beam.family == "cooling"]
    repump = [beam for beam in configured.cooling_repump_beams if beam.family == "repump"]
    assert all(beam.power_w == pytest.approx(25.0e-3) for beam in cooling)
    assert all(beam.power_w == pytest.approx(0.2e-3) for beam in repump)
    assert all(2.0 * beam.beam_radius_m == pytest.approx(14.0e-3) for beam in cooling + repump)
    assert configured.multilevel_config.cooling_detuning_rad_per_s == pytest.approx(
        2.0 * np.pi * -18.0e6
    )
    assert configured.trapping_power_w_per_path == pytest.approx(2.0e-3)
    incident = [beam for beam in configured.trapping_beams if beam.propagation_sense == "incident"]
    retro = [beam for beam in configured.trapping_beams if beam.propagation_sense == "retro"]
    assert all(beam.waist_radius_m == pytest.approx(5.0e-6) for beam in incident)
    assert all(beam.waist_radius_m == pytest.approx(6.0e-6) for beam in retro)


def test_standard_launch_is_15_mm_at_17_m_per_s_and_radially_inward() -> None:
    state = inward_launch_state()
    position = np.asarray(state.position_m)
    velocity = np.asarray(state.velocity_m_per_s)
    assert np.linalg.norm(position) == pytest.approx(15.0e-3)
    assert np.linalg.norm(velocity) == pytest.approx(17.0)
    assert np.dot(position, velocity) == pytest.approx(-15.0e-3 * 17.0)
    np.testing.assert_allclose(position, (15.0e-3, 0.0, 0.0), atol=2.0e-18)
    np.testing.assert_allclose(velocity, (-17.0, 0.0, 0.0), atol=2.0e-15)


def test_vector_only_observable_has_zero_external_field_and_only_vector_shift(
    context,
) -> None:
    observable = vector_only_trajectory_observable(
        context,
        (0.2e-3, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(observable.rate_equation.magnetic_field_t, 0.0, atol=0.0)
    np.testing.assert_allclose(
        observable.applied_transition_shift_rad_per_s,
        observable.stark_diagnostic.vector_transition_energy_j / 1.054571817e-34,
        rtol=1.0e-8,
    )
    assert not np.allclose(
        observable.applied_transition_shift_rad_per_s,
        observable.stark_diagnostic.transition_angular_frequency_shift_rad_per_s,
    )


def test_short_default_run_is_deterministic_and_records_all_diagnostics(context) -> None:
    no_gravity = replace(
        context,
        multilevel_config=replace(context.multilevel_config, include_gravity=False),
        _observable_context=replace(
            context._observable_context,
            multilevel_config=replace(context.multilevel_config, include_gravity=False),
        ),
    )
    numerical = RateEquationTrajectoryConfig(
        time_step_s=5.0e-6,
        include_diffusion=False,
        escape_radius_m=30.0e-3,
    )
    first = simulate_vector_only_pmot_trajectory(
        duration_s=10.0e-6,
        context=no_gravity,
        trajectory_config=numerical,
    )
    second = simulate_vector_only_pmot_trajectory(
        duration_s=10.0e-6,
        context=no_gravity,
        trajectory_config=numerical,
    )
    assert len(first.rate_equation.times_s) == 3
    np.testing.assert_allclose(first.rate_equation.positions_m, second.rate_equation.positions_m)
    np.testing.assert_allclose(first.rate_equation.velocities_m_per_s, second.rate_equation.velocities_m_per_s)
    np.testing.assert_allclose(first.rate_equation.magnetic_fields_t, 0.0, atol=0.0)
    history_lengths = (
        len(first.atom_frame_wavelengths_nm),
        len(first.atom_frame_frequencies_hz),
        len(first.trapping_component_intensities_w_per_m2),
        len(first.effective_fields_t),
        len(first.quantization_axes),
        len(first.reference_vector_shift_hz),
    )
    assert history_lengths == (3, 3, 3, 3, 3, 3)
    assert first.model_metadata["scalar_transition_shift_included"] is False
    assert first.model_metadata["tensor_transition_shift_included"] is False
    assert first.capture is not None
    frame = vector_only_trajectory_dataframe(first)
    assert len(frame) == 3
    assert {"time_s", "x_m", "vx_m_per_s", "Fx_n", "total_780_scattering_rate_per_s"} <= set(frame)


def _record(times_s, radii_m, termination_reason="duration"):
    record = RateEquationTrajectoryRecord()
    record.times_s = list(times_s)
    record.positions_m = [(radius, 0.0, 0.0) for radius in radii_m]
    record.termination_reason = termination_reason
    return record


def test_capture_status_accepts_two_entries_with_intervening_exit() -> None:
    status = classify_vector_only_trajectory(
        _record(
            [0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3],
            [3.0e-3, 1.0e-3, 3.0e-3, 1.0e-3, 1.0e-3],
        )
    )
    assert status.trapped
    assert status.core_entry_count == 2
    assert status.classification == "two_core_entries"


def test_capture_status_accepts_five_ms_continuous_residence() -> None:
    status = classify_vector_only_trajectory(
        _record(
            [0.0, 1.0e-3, 6.0e-3],
            [3.0e-3, 1.0e-3, 1.0e-3],
        )
    )
    assert status.trapped
    assert status.classification == "bounded_core_residence"
    assert status.maximum_continuous_core_residence_s == pytest.approx(5.0e-3)
