"""Stage-B light, polarization, Doppler, and Zeeman validation."""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np

from pmot.fields import build_mot_beams
from pmot.mot_multilevel import build_atomic_structure
from pmot.mot_multilevel import doppler_shift_rad_per_s
from pmot.mot_multilevel import ground_laser_channels
from pmot.mot_multilevel import polarization_weights
from pmot.mot_multilevel import propagation_frame_polarization
from pmot.mot_multilevel import quantization_axis
from pmot.mot_multilevel import zeeman_shift_rad_per_s
from pmot.mot_multilevel.configuration import default_multilevel_mot_config


def test_analytic_polarization_decompositions() -> None:
    axis = (0.0, 0.0, 1.0)
    sigma_plus = propagation_frame_polarization(axis, "sigma+")
    assert np.allclose(list(polarization_weights(sigma_plus, axis).values()), [0.0, 0.0, 1.0], atol=1.0e-14)
    assert np.allclose(list(polarization_weights((0j, 0j, 1 + 0j), axis).values()), [0.0, 1.0, 0.0], atol=1.0e-14)
    transverse = polarization_weights((1 + 0j, 0j, 0j), axis)
    assert np.isclose(transverse[-1], 0.5)
    assert np.isclose(transverse[0], 0.0)
    assert np.isclose(transverse[+1], 0.5)


def test_quantization_axis_fallback_is_stable() -> None:
    previous = (1.0, 0.0, 0.0)
    assert quantization_axis((0.0, 0.0, 0.0), previous, 1.0e-12) == previous
    assert np.allclose(quantization_axis((0.0, 0.0, 2.0e-4), previous, 1.0e-12), (0.0, 0.0, 1.0))


def test_zeeman_sign_and_reversal_symmetry() -> None:
    structure = build_atomic_structure()
    plus = next(t for t in structure.absorption_transitions if (t.ground_m_f, t.excited_f, t.excited_m_f) == (2, 3, 3))
    minus = next(t for t in structure.absorption_transitions if (t.ground_m_f, t.excited_f, t.excited_m_f) == (-2, 3, -3))
    shift_plus = zeeman_shift_rad_per_s(structure, plus, 1.0e-4)
    shift_minus = zeeman_shift_rad_per_s(structure, minus, 1.0e-4)
    assert shift_plus > 0.0
    assert np.isclose(shift_plus, -shift_minus, rtol=1.0e-12)
    assert np.isclose(zeeman_shift_rad_per_s(structure, plus, -1.0e-4), -shift_plus, rtol=1.0e-12)


def test_doppler_sign_regresses_to_validated_convention() -> None:
    beams = [beam for beam in build_mot_beams() if beam.family == "cooling" and beam.axis_name == "horizontal_x"]
    incident = next(beam for beam in beams if beam.direction[0] > 0.0)
    retro = next(beam for beam in beams if beam.direction[0] < 0.0)
    velocity = (1.0, 0.0, 0.0)
    assert doppler_shift_rad_per_s(incident, velocity) > 0.0
    assert doppler_shift_rad_per_s(retro, velocity) < 0.0


def test_off_resonant_manifold_rate_ordering() -> None:
    structure = build_atomic_structure()
    beams = [beam for beam in build_mot_beams() if beam.family == "cooling" and beam.direction == (0.0, 0.0, 1.0)]
    state_index = structure.state_index("ground", 2, 0)
    config = replace(default_multilevel_mot_config(), cooling_detuning_rad_per_s=-2.0 * pi * 15.0e6)
    channels = ground_laser_channels(
        structure, state_index, beams, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config
    )
    rates = {f: sum(channel.rate_per_s for channel in channels if channel.transition.excited_f == f) for f in (1, 2, 3)}
    assert rates[3] > rates[2] > rates[1]
