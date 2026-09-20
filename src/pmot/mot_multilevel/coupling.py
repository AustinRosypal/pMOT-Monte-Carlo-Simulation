"""Section-12 laser coupling, detuning, and stimulated-rate primitives."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np

from ..configuration import HBAR_J_S, SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M
from ..fields import MOTBeam
from .polarization import ComplexVec3, Vec3, propagation_frame_polarization


def wavevector_rad_per_m(beam: MOTBeam) -> Vec3:
    magnitude = 2.0 * pi / beam.wavelength_m
    return tuple(magnitude * component for component in beam.direction)


def beam_polarization_vector(beam: MOTBeam) -> ComplexVec3:
    return propagation_frame_polarization(
        beam.direction,
        beam.circular_polarization,
    )


def electric_field_amplitude_v_per_m(intensity_w_per_m2: float) -> float:
    """Return E=sqrt(2 I / (c epsilon_0))."""

    intensity = float(intensity_w_per_m2)
    if not np.isfinite(intensity) or intensity < 0.0:
        raise ValueError("intensity must be finite and non-negative")
    return sqrt(
        2.0 * intensity
        / (SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M)
    )


def rabi_frequency_rad_per_s(
    electric_field_v_per_m: float,
    polarization_amplitude: complex,
    dipole_matrix_element_c_m: float,
) -> complex:
    """Return Eq. (39), including its physically irrelevant overall sign."""

    return (
        -float(electric_field_v_per_m)
        * complex(polarization_amplitude)
        * float(dipole_matrix_element_c_m)
        / HBAR_J_S
    )


def effective_detuning_rad_per_s(
    laser_angular_frequency_rad_per_s: float,
    transition_angular_frequency_rad_per_s: float,
    wavevector_rad_per_m_value: Vec3,
    velocity_m_per_s: Vec3,
    zeeman_shift_rad_per_s: float,
    extra_transition_shift_rad_per_s: float = 0.0,
) -> float:
    """Return laser minus Doppler, zero-field resonance, and Zeeman shifts."""

    return float(
        laser_angular_frequency_rad_per_s
        - transition_angular_frequency_rad_per_s
        - np.dot(wavevector_rad_per_m_value, velocity_m_per_s)
        - zeeman_shift_rad_per_s
        - extra_transition_shift_rad_per_s
    )


def stimulated_transition_coefficient_per_s(
    excited_decay_rate_per_s: float,
    rabi_frequency_rad_per_s_value: complex,
    detuning_rad_per_s: float,
) -> float:
    """Return Section 12 Eq. (41), with no two-level saturation denominator."""

    gamma_e = float(excited_decay_rate_per_s)
    if gamma_e <= 0.0:
        raise ValueError("excited-state decay rate must be positive")
    omega_squared = abs(complex(rabi_frequency_rad_per_s_value)) ** 2
    return float(
        gamma_e
        * omega_squared
        / (gamma_e**2 + 4.0 * float(detuning_rad_per_s) ** 2)
    )


__all__ = [
    "beam_polarization_vector",
    "effective_detuning_rad_per_s",
    "electric_field_amplitude_v_per_m",
    "rabi_frequency_rad_per_s",
    "stimulated_transition_coefficient_per_s",
    "wavevector_rad_per_m",
]
