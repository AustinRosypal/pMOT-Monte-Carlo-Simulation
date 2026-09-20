"""Regression tests for the isolated fixed-stretched-transition diagnostic."""

from __future__ import annotations

from dataclasses import replace
import json
from math import pi
from pathlib import Path

import numpy as np
import pytest

from pmot.configuration import HBAR_J_S
from pmot.configuration import PLANCK_CONSTANT_J_S
from pmot.configuration import SPEED_OF_LIGHT_M_PER_S
from pmot.pmot.ac_stark import ProvisionalStarkConfig
from pmot.pmot.ac_stark import build_physics_trapping_beams
from pmot.pmot.temporary_tests.fixed_stretched_shift import (
    cooling_effective_detunings_hz,
)
from pmot.pmot.temporary_tests.fixed_stretched_shift import (
    evaluate_fixed_stretched_transition_shift,
)
from pmot.pmot.temporary_tests import run_fixed_stretched_shift_diagnostic as runner


@pytest.fixture(scope="module")
def diagnostic_context():
    """Use the documented demonstration scale and the real narrow table."""

    return runner._build_context(None)


def _evaluate(context, position, velocity, axis):
    return evaluate_fixed_stretched_transition_shift(
        context["trapping_beams"],
        position,
        velocity,
        axis,
        context["apparatus"].trapping_laser,
        context["stark"],
        context["table"],
    )


def test_all_six_components_use_the_full_three_dimensional_doppler_projection(
    diagnostic_context,
) -> None:
    velocity = np.asarray((3.0, -4.0, 5.0))
    observable = _evaluate(
        diagnostic_context,
        (0.3e-3, -0.2e-3, 0.1e-3),
        velocity,
        (0.0, 0.0, 1.0),
    )
    beams = diagnostic_context["trapping_beams"]
    directions = np.asarray([beam.direction for beam in beams])
    wavelengths_m = np.asarray([beam.wavelength_m for beam in beams])
    projected = directions @ velocity
    expected_frequencies = (
        SPEED_OF_LIGHT_M_PER_S
        / wavelengths_m
        * (1.0 - projected / SPEED_OF_LIGHT_M_PER_S)
    )
    expected_wavelengths_nm = (
        1.0e9 * wavelengths_m / (1.0 - projected / SPEED_OF_LIGHT_M_PER_S)
    )

    assert len(beams) == 6
    assert len(set(observable.trapping_component_labels)) == 6
    np.testing.assert_allclose(
        observable.projected_speeds_m_per_s,
        projected,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        observable.atom_frame_frequencies_hz,
        expected_frequencies,
        rtol=2.0e-16,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observable.atom_frame_wavelengths_nm,
        expected_wavelengths_nm,
        rtol=2.0e-16,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observable.trapping_doppler_shifts_hz,
        -projected / wavelengths_m,
        rtol=0.0,
        atol=0.1,
    )
    # A velocity component transverse to a path must not leak into its shift.
    for index, beam in enumerate(beams):
        expected_projection = {
            "horizontal_x": velocity[0],
            "horizontal_y": velocity[1],
            "vertical_z": velocity[2],
        }[beam.axis_name]
        if beam.propagation_sense == "retro":
            expected_projection *= -1.0
        assert observable.projected_speeds_m_per_s[index] == pytest.approx(
            expected_projection, abs=1.0e-15
        )


@pytest.mark.parametrize(
    ("axis", "displacement"),
    (
        ((1.0, 0.0, 0.0), (0.1e-3, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.1e-3, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.1e-3)),
    ),
)
def test_fixed_axis_vector_shift_is_signed_and_odd(
    diagnostic_context,
    axis,
    displacement,
) -> None:
    plus = _evaluate(diagnostic_context, displacement, (0.0, 0.0, 0.0), axis)
    minus = _evaluate(
        diagnostic_context,
        tuple(-value for value in displacement),
        (0.0, 0.0, 0.0),
        axis,
    )

    assert abs(plus.total_vector_shift_hz) > 1.0
    assert plus.total_vector_shift_hz == pytest.approx(
        -minus.total_vector_shift_hz,
        rel=2.0e-13,
        abs=1.0e-6,
    )
    assert plus.total_scalar_shift_hz == pytest.approx(
        minus.total_scalar_shift_hz, rel=2.0e-13
    )
    assert plus.total_tensor_shift_hz == pytest.approx(
        minus.total_tensor_shift_hz, rel=2.0e-13, abs=1.0e-6
    )


def test_one_aligned_circular_component_recovers_magic_scalar_tensor_cancellation(
    diagnostic_context,
) -> None:
    laser = replace(
        diagnostic_context["apparatus"].trapping_laser,
        retro_power_fraction=0.0,
    )
    stark = ProvisionalStarkConfig(
        incident_path_powers_w=(10.0e-3, 0.0, 0.0),
        incident_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
        retro_helicities_by_axis=("sigma+", "sigma+", "sigma-"),
    )
    beams = build_physics_trapping_beams(laser, stark)
    observable = evaluate_fixed_stretched_transition_shift(
        beams,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        laser,
        stark,
        diagnostic_context["table"],
    )

    assert np.count_nonzero(observable.component_intensities_w_per_m2) == 1
    assert observable.total_scalar_shift_hz < 0.0
    assert observable.total_tensor_shift_hz > 0.0
    assert observable.total_vector_shift_hz != 0.0
    residual_fraction = abs(
        (observable.total_scalar_shift_hz + observable.total_tensor_shift_hz)
        / observable.total_vector_shift_hz
    )
    assert residual_fraction == pytest.approx(4.787e-6, rel=2.0e-3)
    assert residual_fraction < 2.0e-5


def test_symmetric_six_beam_origin_keeps_scalar_but_cancels_vector_and_tensor(
    diagnostic_context,
) -> None:
    origin = _evaluate(
        diagnostic_context,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    assert diagnostic_context["power_w_per_path"] == pytest.approx(
        0.03829448617292197, rel=2.0e-12
    )
    assert origin.total_intensity_w_per_m2 == pytest.approx(
        30797.974634914408, rel=2.0e-12
    )
    assert origin.total_scalar_shift_hz / 1.0e6 == pytest.approx(
        -16.33969143, rel=2.0e-9
    )
    assert abs(origin.total_vector_shift_hz) < 1.0e-6
    assert abs(origin.total_tensor_shift_hz) < 1.0e-6
    assert origin.total_frequency_shift_hz == pytest.approx(
        origin.total_scalar_shift_hz, rel=2.0e-15, abs=1.0e-6
    )
    # The aligned one-beam cancellation must not be silently promoted to the
    # non-collinear six-beam fixed-basis sum.
    assert abs(
        origin.total_scalar_shift_hz + origin.total_tensor_shift_hz
    ) > 0.999 * abs(origin.total_scalar_shift_hz)


def test_energy_frequency_shifted_resonance_and_effective_detuning_algebra(
    diagnostic_context,
) -> None:
    velocity = np.asarray((1.5, -2.0, 0.75))
    observable = _evaluate(
        diagnostic_context,
        (0.25e-3, -0.15e-3, 0.05e-3),
        velocity,
        (0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        observable.component_total_shift_hz,
        observable.component_total_energy_j / PLANCK_CONSTANT_J_S,
        rtol=2.0e-16,
        atol=1.0e-9,
    )
    assert observable.total_frequency_shift_hz == pytest.approx(
        observable.total_energy_j / PLANCK_CONSTANT_J_S,
        rel=2.0e-16,
        abs=1.0e-9,
    )
    assert observable.total_energy_j / HBAR_J_S == pytest.approx(
        2.0 * pi * observable.total_frequency_shift_hz,
        rel=1.0e-9,
    )

    bare = diagnostic_context["apparatus"].mot_light.cooling.resonance_frequency_hz
    assert observable.shifted_resonance_frequency_hz(bare) == (
        bare + observable.total_frequency_shift_hz
    )

    hyperfine_offset_hz = 1234.5
    carrier_offset_hz = 2.5e6
    actual = cooling_effective_detunings_hz(
        diagnostic_context["cooling_repump_beams"],
        velocity,
        observable.total_frequency_shift_hz,
        hyperfine_offset_hz=hyperfine_offset_hz,
        cooling_carrier_offset_hz=carrier_offset_hz,
    )
    cooling = [
        beam
        for beam in diagnostic_context["cooling_repump_beams"]
        if beam.family == "cooling"
    ]
    expected = np.asarray(
        [
            beam.detuning_hz
            + carrier_offset_hz
            - hyperfine_offset_hz
            - np.dot(beam.direction, velocity) / beam.wavelength_m
            - observable.total_frequency_shift_hz
            for beam in cooling
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)

    origin = _evaluate(
        diagnostic_context,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    recentered = cooling_effective_detunings_hz(
        diagnostic_context["cooling_repump_beams"],
        (0.0, 0.0, 0.0),
        origin.total_frequency_shift_hz,
        cooling_carrier_offset_hz=origin.total_frequency_shift_hz,
    )
    np.testing.assert_allclose(recentered, -15.0e6, rtol=0.0, atol=1.0e-9)


def test_zero_trapping_power_produces_exactly_zero_energy_and_shift(
    diagnostic_context,
) -> None:
    zero_stark = replace(
        diagnostic_context["stark"],
        incident_path_powers_w=(0.0, 0.0, 0.0),
    )
    beams = build_physics_trapping_beams(
        diagnostic_context["apparatus"].trapping_laser,
        zero_stark,
    )
    observable = evaluate_fixed_stretched_transition_shift(
        beams,
        (0.4e-3, -0.3e-3, 0.2e-3),
        (2.0, -3.0, 4.0),
        (0.0, 0.0, 1.0),
        diagnostic_context["apparatus"].trapping_laser,
        zero_stark,
        diagnostic_context["table"],
    )

    np.testing.assert_array_equal(observable.component_intensities_w_per_m2, 0.0)
    np.testing.assert_array_equal(observable.component_scalar_energy_j, 0.0)
    np.testing.assert_array_equal(observable.component_vector_energy_j, 0.0)
    np.testing.assert_array_equal(observable.component_tensor_energy_j, 0.0)
    np.testing.assert_array_equal(observable.component_total_shift_hz, 0.0)
    assert observable.total_energy_j == 0.0
    assert observable.total_frequency_shift_hz == 0.0


def test_lightweight_end_to_end_diagnostic_writes_complete_self_consistent_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "LINEOUT_SAMPLE_COUNT", 21)
    output = tmp_path / "fixed_stretched_diagnostic"
    result = runner.run_diagnostic(
        output_directory=output,
        power_w_per_path=5.0e-3,
    )

    assert result["status"] == "DIAGNOSTIC_COMPLETE_NOT_PRODUCTION_PHYSICS"
    assert result["qa_status"] == "PASS_WITH_QUALIFICATIONS"
    assert len(result["data_files"]) == 8
    assert len(result["figure_files"]) == 8
    for path in (
        result["readme"],
        result["summary"],
        result["manifest"],
        result["qa"],
        *result["data_files"],
        *result["figure_files"],
    ):
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    qa_payload = json.loads(Path(result["qa"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert qa_payload["status"] == "PASS_WITH_QUALIFICATIONS"
    assert all(qa_payload["checks"].values())
    assert summary["origin"]["fixed_carrier_effective_detuning_mhz"] == pytest.approx(
        -15.0 - summary["origin"]["total_shift_mhz"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert summary["origin"]["center_compensated_effective_detuning_mhz"] == -15.0
    assert summary["limitations"]
    assert manifest["polarizability_row_count"] == 70_400
    assert manifest["configuration"]["incident_helicities_xyz"] == [
        "sigma+",
        "sigma+",
        "sigma-",
    ]
    assert not any("trajectory" in Path(path).name.lower() for path in result["data_files"])
