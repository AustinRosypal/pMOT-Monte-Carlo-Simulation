"""Validation checks for the explicitly provisional pMOT Stark layer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.configuration import SPEED_OF_LIGHT_M_PER_S
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.mot_multilevel.rate_equations import build_rate_equation_model
from pmot.mot_multilevel.rate_equations import RateEquationAtomState
from pmot.mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from pmot.mot_multilevel.rate_equations import rate_equation_observable_from_local_environment
from pmot.pmot.ac_stark import ProvisionalStarkConfig
from pmot.pmot.ac_stark import atom_frame_trapping_frequencies_hz
from pmot.pmot.ac_stark import atom_frame_trapping_wavelengths_m
from pmot.pmot.ac_stark import build_physics_trapping_beams
from pmot.pmot.ac_stark import provisional_power_for_target_gradient_w_per_path
from pmot.pmot.ac_stark import provisional_transition_stark_shifts
from pmot.pmot.configuration import build_pmot_cooling_and_repump_beams
from pmot.pmot.configuration import default_pmot_apparatus_config
from pmot.pmot.polarizability import interpolate_differential_polarizability_arrays
from pmot.pmot.polarizability import load_differential_polarizability_table
from pmot.pmot.trapping_beams import TrappingLaserConfig
from pmot.pmot.stark_trajectories import provisional_pmot_observable
from pmot.pmot.stark_trajectories import simulate_provisional_pmot_trajectory


@pytest.fixture(scope="module")
def provisional_apparatus():
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    apparatus = default_pmot_apparatus_config()
    return model, apparatus, config


def _cycling_transition_index(model) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise AssertionError("missing cycling transition")


def test_retained_cooling_and_repump_beams_exactly_reuse_multilevel_builder(
    provisional_apparatus,
) -> None:
    _, apparatus, config = provisional_apparatus
    actual = build_pmot_cooling_and_repump_beams(apparatus, config)
    expected = build_multilevel_mot_beams(apparatus.mot_light, config)
    assert actual == expected
    assert len(actual) == 12
    assert {
        beam.wavelength_m for beam in actual if beam.family == "repump"
    } == {config.repump_wavelength_m}


def test_atom_frame_trapping_wavelength_uses_full_direction_dot_product(
    provisional_apparatus,
) -> None:
    _, apparatus, _ = provisional_apparatus
    stark = ProvisionalStarkConfig.uniform_power(1.0e-3)
    beams = build_physics_trapping_beams(apparatus.trapping_laser, stark)
    velocity = np.asarray((3.0, -4.0, 5.0))
    actual_frequencies = atom_frame_trapping_frequencies_hz(beams, velocity)
    actual = atom_frame_trapping_wavelengths_m(beams, velocity)

    for index, beam in enumerate(beams):
        projected_speed = float(np.dot(beam.direction, velocity))
        expected_frequency = (
            SPEED_OF_LIGHT_M_PER_S
            / beam.wavelength_m
            * (1.0 - projected_speed / SPEED_OF_LIGHT_M_PER_S)
        )
        expected = beam.wavelength_m / (
            1.0 - projected_speed / SPEED_OF_LIGHT_M_PER_S
        )
        assert actual_frequencies[index] == pytest.approx(
            expected_frequency,
            rel=2.0e-16,
        )
        assert actual[index] == pytest.approx(expected, rel=2.0e-16)

    x_indices = [index for index, beam in enumerate(beams) if beam.axis_name == "horizontal_x"]
    transverse = atom_frame_trapping_wavelengths_m(beams, (0.0, 7.0, 0.0))
    assert all(transverse[index] == beams[index].wavelength_m for index in x_indices)


def test_vectorized_narrow_table_interpolation_is_strict_and_finite() -> None:
    table = load_differential_polarizability_table()
    alpha = interpolate_differential_polarizability_arrays(
        np.asarray((1529.268876, 1529.268881, 1529.268886)),
        table,
    )
    assert all(values.shape == (3,) for values in alpha)
    assert all(np.all(np.isfinite(values)) for values in alpha)
    with pytest.raises(ValueError, match="outside the narrow table range"):
        interpolate_differential_polarizability_arrays((1500.0,), table)


def test_zero_path_power_produces_zero_stark_shift(provisional_apparatus) -> None:
    model, apparatus, _ = provisional_apparatus
    stark = ProvisionalStarkConfig.uniform_power(0.0)
    beams = build_physics_trapping_beams(apparatus.trapping_laser, stark)
    observable = provisional_transition_stark_shifts(
        model,
        beams,
        (0.7e-3, -0.4e-3, 0.2e-3),
        (1.0, -2.0, 0.5),
        apparatus.trapping_laser,
        stark,
    )
    np.testing.assert_allclose(observable.total_transition_energy_j, 0.0)
    np.testing.assert_allclose(observable.effective_field_t, 0.0)
    assert observable.total_intensity_w_per_m2 == 0.0


def test_single_aligned_circular_component_recovers_reference_magic_cancellation(
    provisional_apparatus,
) -> None:
    model, _, _ = provisional_apparatus
    laser = TrappingLaserConfig(retro_power_fraction=0.0)
    stark = ProvisionalStarkConfig(
        incident_path_powers_w=(10.0e-3, 0.0, 0.0),
    )
    beams = build_physics_trapping_beams(laser, stark)
    observable = provisional_transition_stark_shifts(
        model,
        beams,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        laser,
        stark,
        previous_axis=(1.0, 0.0, 0.0),
    )
    index = _cycling_transition_index(model)
    scalar = observable.scalar_transition_energy_j[index]
    tensor = observable.tensor_transition_energy_j[index]
    assert abs(scalar + tensor) / max(abs(scalar), abs(tensor)) < 2.0e-5


def test_symmetric_origin_has_zero_vector_proxy_but_nonzero_scalar_shift(
    provisional_apparatus,
) -> None:
    model, apparatus, _ = provisional_apparatus
    stark = ProvisionalStarkConfig.uniform_power(38.0e-3)
    beams = build_physics_trapping_beams(apparatus.trapping_laser, stark)
    observable = provisional_transition_stark_shifts(
        model,
        beams,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        apparatus.trapping_laser,
        stark,
    )
    index = _cycling_transition_index(model)
    np.testing.assert_allclose(observable.vector_reference_energy_vector_j, 0.0, atol=1.0e-36)
    assert observable.scalar_transition_energy_j[index] < 0.0
    assert abs(observable.tensor_transition_energy_j[index]) < 1.0e-33


def test_twenty_gauss_per_cm_proxy_power_and_force_signs(provisional_apparatus) -> None:
    model, apparatus, config = provisional_apparatus
    power = provisional_power_for_target_gradient_w_per_path(
        model,
        apparatus.trapping_laser,
        target_gradient_g_per_cm=20.0,
    )
    assert power == pytest.approx(38.29e-3, rel=2.0e-3)
    stark = ProvisionalStarkConfig.uniform_power(power)
    trapping_beams = build_physics_trapping_beams(apparatus.trapping_laser, stark)
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)

    for component in range(3):
        position = np.zeros(3)
        position[component] = 1.0e-3
        shift = provisional_transition_stark_shifts(
            model,
            trapping_beams,
            position,
            (0.0, 0.0, 0.0),
            apparatus.trapping_laser,
            stark,
        )
        observable = rate_equation_observable_from_local_environment(
            model,
            mot_beams,
            tuple(position),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            shift.quantization_axis,
            config,
            transition_resonance_shift_rad_per_s=(
                shift.transition_angular_frequency_shift_rad_per_s
            ),
        )
        assert observable.magnetic_field_t == (0.0, 0.0, 0.0)
        assert observable.force_n[component] < 0.0


def test_rate_kernel_rejects_invalid_transition_shift_shape(provisional_apparatus) -> None:
    model, apparatus, config = provisional_apparatus
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    with pytest.raises(ValueError, match="one value per absorption transition"):
        rate_equation_observable_from_local_environment(
            model,
            mot_beams,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            config,
            transition_resonance_shift_rad_per_s=np.zeros(2),
        )


def test_zero_power_pmot_wrapper_recovers_field_free_rate_kernel(
    provisional_apparatus,
) -> None:
    model, apparatus, config = provisional_apparatus
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    stark_config = ProvisionalStarkConfig.uniform_power(0.0)
    trapping_beams = build_physics_trapping_beams(
        apparatus.trapping_laser,
        stark_config,
    )
    position = (0.4e-3, -0.2e-3, 0.1e-3)
    velocity = (0.3, -0.1, 0.2)
    pmot = provisional_pmot_observable(
        model,
        mot_beams,
        trapping_beams,
        position,
        velocity,
        apparatus.trapping_laser,
        stark_config,
        config,
    )
    reference = rate_equation_observable_from_local_environment(
        model,
        mot_beams,
        position,
        velocity,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        config,
    )
    np.testing.assert_allclose(pmot.rate_equation.force_n, reference.force_n)
    np.testing.assert_allclose(
        pmot.rate_equation.beam_scattering_rates_per_s,
        reference.beam_scattering_rates_per_s,
    )
    np.testing.assert_allclose(pmot.rate_equation.populations, reference.populations)


def test_short_pmot_trajectory_has_exact_step_count_and_zero_external_field(
    provisional_apparatus,
) -> None:
    model, apparatus, config = provisional_apparatus
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    stark_config = ProvisionalStarkConfig.uniform_power(1.0e-3)
    trapping_beams = build_physics_trapping_beams(
        apparatus.trapping_laser,
        stark_config,
    )
    record = simulate_provisional_pmot_trajectory(
        RateEquationAtomState((0.5e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
        10.0e-6,
        model,
        mot_beams,
        trapping_beams,
        apparatus.trapping_laser,
        stark_config,
        config,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=5.0e-6,
            include_diffusion=False,
        ),
    )
    assert record.rate_equation.times_s == pytest.approx((0.0, 5.0e-6, 10.0e-6))
    np.testing.assert_allclose(record.rate_equation.magnetic_fields_t, 0.0)


def test_short_pmot_trajectory_rejects_nonpositive_escape_radius(
    provisional_apparatus,
) -> None:
    model, apparatus, config = provisional_apparatus
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus, config)
    stark_config = ProvisionalStarkConfig.uniform_power(1.0e-3)
    trapping_beams = build_physics_trapping_beams(
        apparatus.trapping_laser,
        stark_config,
    )
    with pytest.raises(ValueError, match="escape radius must be positive"):
        simulate_provisional_pmot_trajectory(
            RateEquationAtomState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            5.0e-6,
            model,
            mot_beams,
            trapping_beams,
            apparatus.trapping_laser,
            stark_config,
            config,
            trajectory_config=RateEquationTrajectoryConfig(
                time_step_s=5.0e-6,
                include_diffusion=False,
                escape_radius_m=0.0,
            ),
        )
