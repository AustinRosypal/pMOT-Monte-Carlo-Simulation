"""Geometry tests for the effective two-level pMOT foundation."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import component_intensities_w_per_m2
from pmot.pMOT_2Level import evaluate_trapping_geometry
from pmot.pMOT_2Level import optical_spin_vector_w_per_m2
from pmot.pMOT_2Level import quantization_axis_at_point
from pmot.pMOT_2Level import total_intensity_w_per_m2


def test_default_map_reuses_six_component_geometry_and_candidate_helicities() -> None:
    beams = build_pmot_2level_trapping_beams()

    assert len(beams) == 6
    assert [beam.helicity for beam in beams] == [
        "sigma+",
        "sigma+",
        "sigma+",
        "sigma+",
        "sigma-",
        "sigma-",
    ]
    directions = np.asarray([beam.direction for beam in beams])
    np.testing.assert_allclose(directions[0::2], np.eye(3))
    np.testing.assert_allclose(directions[1::2], -np.eye(3))
    np.testing.assert_allclose(
        np.asarray([beam.waist_position_m for beam in beams[0::2]]),
        -10.0e-3 * np.eye(3),
    )
    np.testing.assert_allclose(
        np.asarray([beam.waist_position_m for beam in beams[1::2]]),
        10.0e-3 * np.eye(3),
    )


def test_intensity_functions_apply_path_power_and_return_consistent_total() -> None:
    beams = build_pmot_2level_trapping_beams()
    point = (0.4e-3, -0.2e-3, 0.7e-3)
    powers = (0.1, 0.2, 0.3)

    components = component_intensities_w_per_m2(
        point,
        beams,
        path_powers_w=powers,
    )
    total = total_intensity_w_per_m2(point, beams, path_powers_w=powers)

    assert components.shape == (6,)
    assert np.all(np.isfinite(components))
    assert np.all(components >= 0.0)
    assert total == pytest.approx(float(np.sum(components)))

    doubled = component_intensities_w_per_m2(
        point,
        beams,
        path_powers_w=2.0 * np.asarray(powers),
    )
    np.testing.assert_allclose(doubled, 2.0 * components)


def test_point_sample_exposes_each_k_hat_and_local_quantization_axis() -> None:
    beams = build_pmot_2level_trapping_beams()
    sample = evaluate_trapping_geometry(
        (0.5e-3, -0.3e-3, 0.8e-3),
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
    )

    assert len(sample.component_intensities_w_per_m2) == 6
    assert len(sample.component_k_hat) == 6
    assert sample.quantization_axis_from_optical_spin is True
    assert np.linalg.norm(sample.quantization_axis) == pytest.approx(1.0)
    np.testing.assert_allclose(
        sample.component_k_hat,
        [beam.direction for beam in beams],
    )


def test_balanced_origin_reports_axis_degeneracy_and_uses_selected_fallback() -> None:
    beams = build_pmot_2level_trapping_beams()
    axis, defined = quantization_axis_at_point(
        (0.0, 0.0, 0.0),
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
        fallback_axis=(1.0, 1.0, 0.0),
    )

    assert defined is False
    np.testing.assert_allclose(axis, np.asarray((1.0, 1.0, 0.0)) / np.sqrt(2.0))
    np.testing.assert_allclose(
        optical_spin_vector_w_per_m2(
            (0.0, 0.0, 0.0),
            beams,
            path_powers_w=(1.0, 1.0, 1.0),
        ),
        np.zeros(3),
        atol=1.0e-8,
    )


def test_vector_geometry_is_odd_and_axis_reverses_across_origin() -> None:
    beams = build_pmot_2level_trapping_beams()
    point = np.asarray((0.2e-3, -0.4e-3, 0.7e-3))
    plus = optical_spin_vector_w_per_m2(
        point,
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
    )
    minus = optical_spin_vector_w_per_m2(
        -point,
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
    )
    axis_plus, defined_plus = quantization_axis_at_point(
        point,
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
    )
    axis_minus, defined_minus = quantization_axis_at_point(
        -point,
        beams,
        path_powers_w=(1.0, 1.0, 1.0),
    )

    np.testing.assert_allclose(minus, -plus, rtol=2.0e-13, atol=1.0e-8)
    assert defined_plus and defined_minus
    np.testing.assert_allclose(axis_minus, -axis_plus, rtol=2.0e-13, atol=1.0e-13)


@pytest.mark.parametrize(
    ("powers", "match"),
    [
        ((1.0, 2.0), "three finite"),
        ((1.0, -1.0, 1.0), "non-negative"),
    ],
)
def test_invalid_path_power_inputs_fail_closed(powers, match: str) -> None:
    beams = build_pmot_2level_trapping_beams()
    with pytest.raises(ValueError, match=match):
        component_intensities_w_per_m2(
            (0.0, 0.0, 0.0),
            beams,
            path_powers_w=powers,
        )
