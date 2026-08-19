"""Beam-specific Doppler, Zeeman, polarization, and laser-rate calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from ..fields import MOTBeam
from ..fields import beam_intensity_w_per_m2
from .atomic_structure import AtomicStructure
from .atomic_structure import BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T
from .atomic_structure import DipoleTransition
from .configuration import MultilevelMOTConfig
from .polarization import ComplexVec3
from .polarization import Vec3
from .polarization import polarization_weights
from .polarization import propagation_frame_polarization


@dataclass(frozen=True, slots=True)
class LaserChannel:
    """One beam-resolved laser-driven internal-state channel."""

    beam_index: int
    source_state_index: int
    destination_state_index: int
    transition: DipoleTransition
    event_type: str
    polarization_weight: float
    saturation_parameter: float
    detuning_rad_per_s: float
    rate_per_s: float


def wavevector_rad_per_m(beam: MOTBeam) -> Vec3:
    magnitude = 2.0 * pi / beam.wavelength_m
    return tuple(magnitude * value for value in beam.direction)


def doppler_shift_rad_per_s(beam: MOTBeam, velocity_m_per_s: Vec3) -> float:
    return float(np.dot(np.asarray(wavevector_rad_per_m(beam)), np.asarray(velocity_m_per_s)))


def zeeman_shift_rad_per_s(
    structure: AtomicStructure,
    transition: DipoleTransition,
    field_magnitude_t: float,
) -> float:
    ground = structure.states[transition.ground_state_index]
    excited = structure.states[transition.excited_state_index]
    return BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T * field_magnitude_t * (
        excited.lande_g * excited.m_f - ground.lande_g * ground.m_f
    )


def effective_detuning_rad_per_s(
    beam: MOTBeam,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    transition: DipoleTransition,
    structure: AtomicStructure,
    config: MultilevelMOTConfig,
) -> float:
    """Return delta_L-F' - k.v - delta_Z in angular-frequency units."""

    if beam.family == "cooling":
        laser_detuning = config.cooling_detuning_rad_per_s
        hyperfine_offset = transition.hyperfine_offset_rad_per_s
    elif beam.family == "repump":
        laser_detuning = config.repump_detuning_rad_per_s
        hyperfine_offset = transition.hyperfine_offset_rad_per_s - structure.states[
            structure.state_index("excited", 2, 0)
        ].energy_offset_rad_per_s
    else:
        raise ValueError(f"unsupported beam family: {beam.family}")
    return (
        laser_detuning
        - hyperfine_offset
        - doppler_shift_rad_per_s(beam, velocity_m_per_s)
        - zeeman_shift_rad_per_s(structure, transition, field_magnitude_t)
    )


def laser_driven_rate_per_s(saturation_parameter: float, detuning_rad_per_s: float, gamma_rad_per_s: float) -> float:
    """Return the specified incoherent laser-driven transition rate."""

    return 0.5 * gamma_rad_per_s * saturation_parameter / (
        1.0 + 4.0 * detuning_rad_per_s**2 / gamma_rad_per_s**2
    )


def beam_polarization_vector(beam: MOTBeam) -> ComplexVec3:
    return propagation_frame_polarization(beam.direction, beam.circular_polarization)


def ground_laser_channels(
    structure: AtomicStructure,
    state_index: int,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig,
) -> list[LaserChannel]:
    """Enumerate laser-driven transitions for the addressed ground manifold."""

    state = structure.states[state_index]
    if not state.is_ground:
        return []
    channels: list[LaserChannel] = []
    for beam_index, beam in enumerate(beams):
        if beam.family == "cooling":
            if state.f != 2:
                continue
            enabled_manifolds = config.enabled_excited_manifolds
        elif beam.family == "repump":
            if not config.repumper_enabled or state.f != 1 or beam.power_w <= 0.0:
                continue
            enabled_manifolds = config.enabled_repump_excited_manifolds
        else:
            continue
        intensity = beam_intensity_w_per_m2(beam, position_m)
        weights = polarization_weights(beam_polarization_vector(beam), quantization_axis_vector)
        for transition in structure.absorption_by_ground[state_index]:
            if transition.excited_f not in enabled_manifolds:
                continue
            polarization_weight = weights[transition.q]
            saturation = intensity / config.saturation_intensity_w_per_m2 * transition.c_squared * polarization_weight
            detuning = effective_detuning_rad_per_s(
                beam, velocity_m_per_s, field_magnitude_t, transition, structure, config
            )
            rate = laser_driven_rate_per_s(saturation, detuning, config.natural_linewidth_rad_per_s)
            if rate > 0.0:
                channels.append(
                    LaserChannel(
                        beam_index,
                        state_index,
                        transition.excited_state_index,
                        transition,
                        "absorption",
                        polarization_weight,
                        saturation,
                        detuning,
                        rate,
                    )
                )
    return channels


def stimulated_emission_channels(
    structure: AtomicStructure,
    state_index: int,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig,
) -> list[LaserChannel]:
    """Enumerate beam-resolved stimulated-emission channels from an excited state."""

    if not structure.states[state_index].is_excited:
        return []
    channels: list[LaserChannel] = []
    for beam_index, beam in enumerate(beams):
        if beam.family == "cooling":
            enabled_manifolds = config.enabled_excited_manifolds
            addressed_ground_f = 2
        elif beam.family == "repump":
            if not config.repumper_enabled or beam.power_w <= 0.0:
                continue
            enabled_manifolds = config.enabled_repump_excited_manifolds
            addressed_ground_f = 1
        else:
            continue
        intensity = beam_intensity_w_per_m2(beam, position_m)
        weights = polarization_weights(beam_polarization_vector(beam), quantization_axis_vector)
        for transition in structure.transitions_by_excited[state_index]:
            if transition.ground_f != addressed_ground_f:
                continue
            if transition.excited_f not in enabled_manifolds:
                continue
            polarization_weight = weights[transition.q]
            saturation = intensity / config.saturation_intensity_w_per_m2 * transition.c_squared * polarization_weight
            detuning = effective_detuning_rad_per_s(
                beam, velocity_m_per_s, field_magnitude_t, transition, structure, config
            )
            rate = laser_driven_rate_per_s(saturation, detuning, config.natural_linewidth_rad_per_s)
            if rate > 0.0:
                channels.append(
                    LaserChannel(
                        beam_index,
                        state_index,
                        transition.ground_state_index,
                        transition,
                        "stimulated_emission",
                        polarization_weight,
                        saturation,
                        detuning,
                        rate,
                    )
                )
    return channels
