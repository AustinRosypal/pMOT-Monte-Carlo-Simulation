from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.fields import beam_intensity_w_per_m2
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    MODEL_OUTPUT_NAMESPACE,
    assemble_rate_matrix,
    assert_rate_matrix_conserves_probability,
    build_beam_transition_quantities,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    default_multilevel_mot_config,
    multilevel_mot_paths,
    rate_equation_observable,
    steady_state_populations,
)


def test_new_solver_uses_a_fresh_output_namespace(tmp_path) -> None:
    paths = multilevel_mot_paths(tmp_path)
    assert MODEL_OUTPUT_NAMESPACE == "mot_multilevel_population_rate_v1"
    assert paths["statistics"] == tmp_path / "outputs/statistics" / MODEL_OUTPUT_NAMESPACE


def test_default_multilevel_cooling_power_is_27_mw_per_traveling_beam() -> None:
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    assert len(cooling) == len(repump) == 6
    assert all(beam.power_w == pytest.approx(27.0e-3) for beam in cooling)
    assert all(beam.power_w == pytest.approx(0.1e-3) for beam in repump)
    expected_peak_w_per_m2 = 2.0 * 27.0e-3 / (np.pi * (6.35e-3) ** 2)
    assert beam_intensity_w_per_m2(cooling[0], (0.0, 0.0, 0.0)) == pytest.approx(
        expected_peak_w_per_m2
    )


def test_two_level_limit_recovers_the_saturated_scattering_rate() -> None:
    gamma = 2.0 * np.pi * 6.07e6
    saturation = 3.5
    detuning = -0.8 * gamma
    w = 0.5 * gamma * saturation / (1.0 + 4.0 * detuning**2 / gamma**2)
    matrix = assemble_rate_matrix(
        np.asarray([[w]]),
        np.asarray([[gamma]]),
    )
    populations = steady_state_populations(matrix)
    expected_excited = w / (2.0 * w + gamma)
    expected_scattering = 0.5 * gamma * saturation / (
        1.0 + saturation + 4.0 * detuning**2 / gamma**2
    )
    assert populations[1] == pytest.approx(expected_excited, rel=2.0e-14)
    assert w * (populations[0] - populations[1]) == pytest.approx(
        expected_scattering,
        rel=2.0e-14,
    )
    assert gamma * populations[1] == pytest.approx(expected_scattering, rel=2.0e-14)


def test_weakly_driven_positive_excited_population_preserves_photon_flow() -> None:
    gamma = 4.0e7
    w = 4.0e-5
    matrix = assemble_rate_matrix(np.asarray([[w]]), np.asarray([[gamma]]))
    populations = steady_state_populations(matrix)
    assert 0.0 < populations[1] < 1.0e-11
    assert w * (populations[0] - populations[1]) == pytest.approx(
        gamma * populations[1], rel=1.0e-10
    )


def test_column_conservation_allows_roundoff_but_rejects_probability_leak() -> None:
    matrix = np.asarray([[-4.0e7, 4.0e7], [4.0e7, -4.0e7]])
    matrix[0, 0] += 1.1e-7
    assert_rate_matrix_conserves_probability(matrix)
    matrix[0, 0] += 1.0e-4
    with pytest.raises(RuntimeError, match="does not conserve probability"):
        assert_rate_matrix_conserves_probability(matrix)


def test_section_12_w_uses_rabi_frequency_without_a_saturation_denominator() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    quantities = build_beam_transition_quantities(
        model,
        beams,
        (0.0, 0.0, 0.0),
        (0.4, -0.2, 0.1),
        2.0e-4,
        (0.0, 0.0, 1.0),
        config,
    )
    beam_index = 0
    nonzero = np.flatnonzero(np.abs(quantities.rabi_frequencies_rad_per_s[beam_index]) > 0.0)
    transition_index = int(nonzero[0])
    excited = model.transition_excited[transition_index]
    ground = model.transition_ground[transition_index]
    omega = quantities.rabi_frequencies_rad_per_s[beam_index, transition_index]
    detuning = quantities.effective_detunings_rad_per_s[beam_index, transition_index]
    gamma_e = model.excited_decay_rates_per_s[excited]
    expected = gamma_e * abs(omega) ** 2 / (gamma_e**2 + 4.0 * detuning**2)
    assert quantities.stimulated_coefficients_per_s[beam_index, excited, ground] == pytest.approx(
        expected,
        rel=2.0e-14,
    )


def test_rabi_squared_and_w_scale_linearly_with_beam_intensity() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beam = build_multilevel_mot_beams(config=config)[0]
    brighter = replace(beam, power_w=4.0 * beam.power_w)
    base = build_beam_transition_quantities(
        model, [beam], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config
    )
    scaled = build_beam_transition_quantities(
        model, [brighter], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config
    )
    mask = np.abs(base.rabi_frequencies_rad_per_s[0]) > 0.0
    np.testing.assert_allclose(
        np.abs(scaled.rabi_frequencies_rad_per_s[0, mask]) ** 2,
        4.0 * np.abs(base.rabi_frequencies_rad_per_s[0, mask]) ** 2,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        scaled.stimulated_coefficients_per_s,
        4.0 * base.stimulated_coefficients_per_s,
        rtol=2.0e-14,
    )


def test_full_rate_matrix_conserves_probability_and_has_a_physical_null_vector() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    observable = rate_equation_observable(
        model,
        beams,
        (0.4e-3, -0.2e-3, 0.7e-3),
        (0.3, -0.1, 0.2),
        default_anti_helmholtz_config(),
        config,
        store_rate_matrix=True,
    )
    matrix = observable.rate_matrix_per_s
    assert matrix is not None
    assert matrix.shape == (24, 24)
    np.testing.assert_allclose(np.sum(matrix, axis=0), 0.0, rtol=0.0, atol=1.0e-7)
    assert np.all(observable.populations >= 0.0)
    assert np.sum(observable.populations) == pytest.approx(1.0, abs=1.0e-13)
    np.testing.assert_allclose(matrix @ observable.populations, 0.0, rtol=0.0, atol=2.0e-3)


def test_rate_matrix_blocks_match_section_12() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    quantities = build_beam_transition_quantities(
        model, beams, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config
    )
    stimulated = np.sum(quantities.stimulated_coefficients_per_s, axis=0)
    matrix = assemble_rate_matrix(stimulated, model.spontaneous_decay_matrix_per_s)
    ground_count = model.ground_count
    np.testing.assert_allclose(matrix[ground_count:, :ground_count], stimulated)
    np.testing.assert_allclose(
        matrix[:ground_count, ground_count:],
        stimulated.T + model.spontaneous_decay_matrix_per_s,
    )


def test_force_uses_beam_resolved_absorption_minus_stimulated_emission() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    observable = rate_equation_observable(
        model,
        beams,
        (0.0, 0.0, 0.0),
        (0.6, -0.3, 0.2),
        default_anti_helmholtz_config(),
        config,
    )
    absorption = np.asarray(observable.beam_absorption_rates_per_s)
    emission = np.asarray(observable.beam_stimulated_emission_rates_per_s)
    np.testing.assert_allclose(
        observable.beam_effective_scattering_rates_per_s,
        absorption - emission,
        rtol=0.0,
        atol=1.0e-9,
    )
    assert observable.total_effective_scattering_rate_per_s == pytest.approx(
        observable.total_spontaneous_scattering_rate_per_s,
        rel=2.0e-12,
    )


@pytest.mark.parametrize(
    ("position", "component"),
    [
        ((1.0e-3, 0.0, 0.0), 0),
        ((0.0, 1.0e-3, 0.0), 1),
        ((0.0, 0.0, 1.0e-3), 2),
    ],
)
def test_force_is_restoring_on_all_three_axes(position, component) -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    observable = rate_equation_observable(
        model,
        build_multilevel_mot_beams(config=config),
        position,
        (0.0, 0.0, 0.0),
        default_anti_helmholtz_config(),
        config,
    )
    assert observable.force_n[component] < 0.0


@pytest.mark.parametrize("component", [0, 1, 2])
def test_force_is_damping_on_all_three_axes(component) -> None:
    velocity = [0.0, 0.0, 0.0]
    velocity[component] = 1.0
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    observable = rate_equation_observable(
        model,
        build_multilevel_mot_beams(config=config),
        (0.0, 0.0, 0.0),
        tuple(velocity),
        default_anti_helmholtz_config(),
        config,
    )
    assert observable.force_n[component] < 0.0


def test_symmetric_origin_force_is_zero() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    observable = rate_equation_observable(
        model,
        build_multilevel_mot_beams(config=config),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        default_anti_helmholtz_config(),
        config,
    )
    np.testing.assert_allclose(observable.force_n, 0.0, rtol=0.0, atol=1.0e-32)


def test_gradient_and_global_helicity_reversal_reverse_the_restoring_force() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=config)
    coil = default_anti_helmholtz_config()
    position = (1.0e-3, 0.0, 0.0)
    force = rate_equation_observable(
        model, beams, position, (0.0, 0.0, 0.0), coil, config
    ).force_n[0]
    reversed_gradient = rate_equation_observable(
        model,
        beams,
        position,
        (0.0, 0.0, 0.0),
        replace(coil, current_a=-coil.current_a),
        config,
    ).force_n[0]
    opposite = {"sigma+": "sigma-", "sigma-": "sigma+", "pi": "pi"}
    reversed_beams = [
        replace(beam, circular_polarization=opposite[beam.circular_polarization])
        for beam in beams
    ]
    reversed_helicity = rate_equation_observable(
        model, reversed_beams, position, (0.0, 0.0, 0.0), coil, config
    ).force_n[0]
    assert reversed_gradient == pytest.approx(-force, rel=2.0e-12)
    assert reversed_helicity == pytest.approx(-force, rel=2.0e-12)


def test_field_free_symmetric_light_has_no_position_force() -> None:
    model = build_rate_equation_model()
    config = default_multilevel_mot_config()
    coil = replace(default_anti_helmholtz_config(), current_a=0.0)
    observable = rate_equation_observable(
        model,
        build_multilevel_mot_beams(config=config),
        (1.0e-3, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        coil,
        config,
    )
    np.testing.assert_allclose(observable.force_n, 0.0, rtol=0.0, atol=1.0e-32)
