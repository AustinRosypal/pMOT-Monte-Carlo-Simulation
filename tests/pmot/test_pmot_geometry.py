"""Focused geometry-contract tests for the first no-coil pMOT apparatus."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

import numpy as np
import pytest

from pmot.beams import axis_direction_from_name
from pmot.pmot.configuration import (
    PMOTApparatusConfig,
    build_pmot_cooling_and_repump_beams,
    default_pmot_apparatus_config,
    describe_pmot_configuration,
)
from pmot.pmot.geometry_validation import sample_axis_lineout, sample_intensity_plane
from pmot.pmot.trapping_beams import (
    DEFAULT_TRAPPING_AXES,
    DEFAULT_TRAPPING_WAVELENGTH_M,
    TrappingLaserConfig,
    beams_for_trapping_axis,
    build_trapping_beams,
    helicity_sign,
    total_trapping_intensity_response_m_inv2,
    trapping_beam_intensity_response_m_inv2,
    vector_intensity_response_m_inv2,
)


def test_exact_default_wavelength_and_user_configuration_are_preserved() -> None:
    default = TrappingLaserConfig()
    assert DEFAULT_TRAPPING_WAVELENGTH_M == 1529.268881e-9
    assert default.wavelength_m == 1529.268881e-9
    assert default.incident_helicity == "sigma+"
    assert default.retro_helicity == "sigma+"
    assert default.resolved_incident_waist_radius_m == pytest.approx(
        2.2336312398559133e-6,
        rel=1.0e-12,
    )

    custom_wavelength_m = 1530.125e-9
    apparatus = default_pmot_apparatus_config(
        trapping_wavelength_m=custom_wavelength_m,
        incident_trapping_helicity="sigma-",
        retro_trapping_helicity="sigma+",
    )
    assert apparatus.trapping_laser.wavelength_m == custom_wavelength_m
    assert apparatus.trapping_laser.incident_helicity == "sigma-"
    assert apparatus.trapping_laser.retro_helicity == "sigma+"
    assert (
        apparatus.trapping_laser.resolved_incident_waist_radius_m
        / default.resolved_incident_waist_radius_m
    ) == pytest.approx(custom_wavelength_m / default.wavelength_m)


@pytest.mark.parametrize("field_name", ("incident_helicity", "retro_helicity"))
def test_trapping_helicity_is_explicitly_validated(field_name: str) -> None:
    with pytest.raises(ValueError, match="helicity must be"):
        TrappingLaserConfig(**{field_name: "linear-x"})

    assert helicity_sign("sigma+") == -1.0
    assert helicity_sign("sigma-") == 1.0
    assert helicity_sign("pi") == 0.0


def test_one_laser_metadata_and_all_optical_component_counts() -> None:
    apparatus = default_pmot_apparatus_config()
    mot_beams = build_pmot_cooling_and_repump_beams(apparatus)
    trapping_beams = build_trapping_beams(apparatus.trapping_laser)
    summary = describe_pmot_configuration(apparatus)

    assert summary["trapping_laser_count"] == 1
    assert summary["trapping_path_count"] == 3
    assert summary["trapping_component_count"] == 6
    assert summary["cooling_component_count"] == 6
    assert summary["repump_component_count"] == 6
    assert summary["trapping_wavelength_nm"] == pytest.approx(1529.268881)
    assert summary["trapping_focus_positions_mm"] == pytest.approx([-10.0, 10.0])

    assert len(mot_beams) == 12
    assert Counter(beam.family for beam in mot_beams) == {
        "cooling": 6,
        "repump": 6,
    }
    for family in ("cooling", "repump"):
        for axis_name in DEFAULT_TRAPPING_AXES:
            assert sum(
                beam.family == family and beam.axis_name == axis_name
                for beam in mot_beams
            ) == 2

    assert len(trapping_beams) == 6
    assert {beam.axis_name for beam in trapping_beams} == set(DEFAULT_TRAPPING_AXES)
    for axis_name in DEFAULT_TRAPPING_AXES:
        pair = beams_for_trapping_axis(trapping_beams, axis_name)
        assert {beam.propagation_sense for beam in pair} == {"incident", "retro"}


def test_pmot_apparatus_structurally_excludes_coils_and_external_field() -> None:
    apparatus = default_pmot_apparatus_config()

    field_names = {field.name for field in fields(PMOTApparatusConfig)}
    assert {"mot_light", "trapping_laser"} <= field_names
    assert not any("coil" in field_name for field_name in field_names)
    assert apparatus.external_magnetic_field_t == (0.0, 0.0, 0.0)
    assert apparatus.anti_helmholtz_coils_present is False


def test_each_round_trip_has_opposite_propagation_and_waists_at_plus_minus_10_mm() -> None:
    config = TrappingLaserConfig()
    beams = build_trapping_beams(config)

    for axis_name in DEFAULT_TRAPPING_AXES:
        axis_direction = np.asarray(axis_direction_from_name(axis_name), dtype=float)
        pair = beams_for_trapping_axis(beams, axis_name)
        incident = next(beam for beam in pair if beam.propagation_sense == "incident")
        retro = next(beam for beam in pair if beam.propagation_sense == "retro")

        np.testing.assert_allclose(incident.direction, axis_direction)
        np.testing.assert_allclose(retro.direction, -axis_direction)
        np.testing.assert_allclose(
            incident.waist_position_m,
            -10.0e-3 * axis_direction,
        )
        np.testing.assert_allclose(
            retro.waist_position_m,
            10.0e-3 * axis_direction,
        )
        assert np.dot(incident.direction, retro.direction) == pytest.approx(-1.0)
        assert incident.wavelength_m == config.wavelength_m
        assert retro.wavelength_m == config.wavelength_m


def test_balanced_scalar_intensity_is_even_finite_and_nonnegative() -> None:
    beams = build_trapping_beams()
    positions_m = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.3e-3, -0.8e-3, 1.1e-3),
            (-1.7e-3, 0.4e-3, 0.9e-3),
            (10.0e-3, 0.0, 0.0),
        ]
    )

    positive = total_trapping_intensity_response_m_inv2(beams, positions_m)
    inverted = total_trapping_intensity_response_m_inv2(beams, -positions_m)
    np.testing.assert_allclose(positive, inverted, rtol=2.0e-13, atol=0.0)
    assert np.all(np.isfinite(positive))
    assert np.all(positive >= 0.0)

    for beam in beams:
        component = trapping_beam_intensity_response_m_inv2(beam, positions_m)
        assert component.shape == (len(positions_m),)
        assert np.all(np.isfinite(component))
        assert np.all(component >= 0.0)


def test_vector_proxy_is_odd_zero_at_origin_and_flips_with_helicity() -> None:
    plus_beams = build_trapping_beams(
        TrappingLaserConfig(incident_helicity="sigma+", retro_helicity="sigma+")
    )
    minus_beams = build_trapping_beams(
        TrappingLaserConfig(incident_helicity="sigma-", retro_helicity="sigma-")
    )
    positions_m = np.asarray(
        [
            (0.25e-3, 0.0, 0.0),
            (0.0, -0.6e-3, 0.0),
            (0.0, 0.0, 1.2e-3),
            (0.5e-3, -0.4e-3, 0.7e-3),
        ]
    )

    plus = vector_intensity_response_m_inv2(plus_beams, positions_m)
    inverted = vector_intensity_response_m_inv2(plus_beams, -positions_m)
    minus = vector_intensity_response_m_inv2(minus_beams, positions_m)
    origin = np.asarray(
        vector_intensity_response_m_inv2(plus_beams, (0.0, 0.0, 0.0))
    )

    np.testing.assert_allclose(inverted, -plus, rtol=2.0e-13, atol=1.0e-8)
    np.testing.assert_allclose(minus, -plus, rtol=2.0e-13, atol=1.0e-8)
    np.testing.assert_allclose(origin, np.zeros(3), atol=1.0e-8)
    assert np.any(np.linalg.norm(plus, axis=1) > 0.0)
    assert np.all(np.isfinite(plus))


def test_lightweight_line_and_plane_samplers_have_stable_shapes() -> None:
    beams = build_trapping_beams()

    for axis_name in DEFAULT_TRAPPING_AXES:
        line = sample_axis_lineout(
            beams,
            axis_name,
            extent_m=1.0e-3,
            sample_count=7,
        )
        assert set(line) == {
            "coordinate_m",
            "incident_response_m_inv2",
            "retro_response_m_inv2",
            "total_response_m_inv2",
            "signed_axis_response_m_inv2",
        }
        assert all(values.shape == (7,) for values in line.values())
        assert np.all(np.isfinite(np.stack(list(line.values()))))
        assert np.all(line["total_response_m_inv2"] >= 0.0)

    for plane_name in ("xy", "xz", "yz"):
        plane = sample_intensity_plane(
            beams,
            plane_name,
            extent_m=1.0e-3,
            sample_count=5,
        )
        assert plane["axis_1_m"].shape == (5,)
        assert plane["axis_2_m"].shape == (5,)
        assert plane["total_response_m_inv2"].shape == (5, 5)
        assert plane["vector_response_m_inv2"].shape == (5, 5, 3)
        assert np.all(np.isfinite(plane["total_response_m_inv2"]))
        assert np.all(np.isfinite(plane["vector_response_m_inv2"]))
        assert np.all(plane["total_response_m_inv2"] >= 0.0)
