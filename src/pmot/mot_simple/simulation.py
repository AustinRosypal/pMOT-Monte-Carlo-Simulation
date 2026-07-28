"""Deterministic effective two-level MOT simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from ..configuration import HBAR_J_S
from ..configuration import RB87_MASS_KG
from ..fields import MOTBeam
from ..fields import beam_intensity_w_per_m2
from ..fields import build_mot_beams
from ..forces import AtomState
from ..mot.magnetic_fields import anti_helmholtz_field_t
from .configuration import default_simple_mot_apparatus
from .configuration import default_simple_mot_config
from .configuration import SimpleMOTConfig


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SimpleMOTBeam:
    """One effective two-level cooling beam."""

    label: str
    axis_name: str
    propagation_sense: str
    circular_polarization: str
    direction: Vec3
    wavelength_m: float
    intensity_beam: MOTBeam
    polarization_sign: float
    detuning_hz: float


@dataclass(frozen=True, slots=True)
class SimpleMOTRateSample:
    """One beam-resolved deterministic scattering sample."""

    beam_label: str
    axis_name: str
    propagation_sense: str
    circular_polarization: str
    intensity_w_per_m2: float
    saturation_parameter: float
    total_saturation_parameter: float
    doppler_shift_hz: float
    zeeman_shift_hz: float
    effective_detuning_hz: float
    scattering_rate_per_s: float


@dataclass(frozen=True, slots=True)
class SimpleMOTTrajectoryRecord:
    """Saved deterministic trajectory and force histories."""

    times_s: list[float]
    positions_m: list[Vec3]
    velocities_m_per_s: list[Vec3]
    forces_n: list[Vec3]
    magnetic_fields_t: list[Vec3]
    total_scattering_rates_per_s: list[float]
    beam_scattering_rates_per_s: dict[str, list[float]]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def wavevector_magnitude_m_inv(beam: SimpleMOTBeam) -> float:
    return 2.0 * pi / beam.wavelength_m


def build_simple_mot_beams(
    apparatus_config=None,
    simple_config: SimpleMOTConfig | None = None,
) -> list[SimpleMOTBeam]:
    """Build the six cooling beams used by the simplified MOT model.

    The effective polarization sign is axis-dependent, not tied directly to the
    multilevel handedness interpretation. This is an explicit simplifying
    convention for the two-level model.
    """

    apparatus = apparatus_config or default_simple_mot_apparatus()
    simple = simple_config or default_simple_mot_config()
    beams: list[SimpleMOTBeam] = []
    for beam in build_mot_beams(apparatus):
        if beam.family != "cooling":
            continue
        beams.append(
            SimpleMOTBeam(
                label=beam.label,
                axis_name=beam.axis_name,
                propagation_sense=beam.propagation_sense,
                circular_polarization=beam.circular_polarization,
                direction=beam.direction,
                wavelength_m=beam.wavelength_m,
                intensity_beam=beam,
                polarization_sign=simple.axis_polarization_sign[beam.axis_name],
                detuning_hz=simple.cooling_detuning_hz,
            )
        )
    return beams


def local_magnetic_field_t(position_m: Vec3, coil_config) -> Vec3:
    bx, by, bz = anti_helmholtz_field_t(position_m[0], position_m[1], position_m[2], coil_config)
    return (float(np.asarray(bx)), float(np.asarray(by)), float(np.asarray(bz)))


def beam_intensity(beam: SimpleMOTBeam, position_m: Vec3) -> float:
    return beam_intensity_w_per_m2(beam.intensity_beam, position_m)


def doppler_shift_hz(beam: SimpleMOTBeam, velocity_m_per_s: Vec3) -> float:
    return wavevector_magnitude_m_inv(beam) * dot(beam.direction, velocity_m_per_s) / (2.0 * pi)


def zeeman_shift_hz(
    beam: SimpleMOTBeam,
    magnetic_field_t: Vec3,
    simple_config: SimpleMOTConfig,
) -> float:
    b_parallel_t = dot(magnetic_field_t, beam.direction)
    return beam.polarization_sign * simple_config.effective_magnetic_moment_hz_per_t * b_parallel_t


def rate_samples(
    beams: list[SimpleMOTBeam],
    atom_state: AtomState,
    coil_config,
    simple_config: SimpleMOTConfig | None = None,
) -> tuple[list[SimpleMOTRateSample], Vec3]:
    """Return deterministic beam-resolved scattering rates."""

    simple = simple_config or default_simple_mot_config()
    magnetic_field = local_magnetic_field_t(atom_state.position_m, coil_config)
    beam_intensities = {beam.label: beam_intensity(beam, atom_state.position_m) for beam in beams}
    beam_saturations = {
        beam.label: beam_intensities[beam.label] / simple.saturation_intensity_w_per_m2 for beam in beams
    }
    total_saturation = sum(beam_saturations.values())

    samples: list[SimpleMOTRateSample] = []
    for beam in beams:
        doppler = doppler_shift_hz(beam, atom_state.velocity_m_per_s)
        zeeman = zeeman_shift_hz(beam, magnetic_field, simple)
        delta_eff = beam.detuning_hz - doppler - zeeman
        rate = 0.5 * simple.linewidth_hz * beam_saturations[beam.label] / (
            1.0 + total_saturation + (2.0 * delta_eff / simple.linewidth_hz) ** 2
        )
        samples.append(
            SimpleMOTRateSample(
                beam_label=beam.label,
                axis_name=beam.axis_name,
                propagation_sense=beam.propagation_sense,
                circular_polarization=beam.circular_polarization,
                intensity_w_per_m2=beam_intensities[beam.label],
                saturation_parameter=beam_saturations[beam.label],
                total_saturation_parameter=total_saturation,
                doppler_shift_hz=doppler,
                zeeman_shift_hz=zeeman,
                effective_detuning_hz=delta_eff,
                scattering_rate_per_s=rate,
            )
        )
    return samples, magnetic_field


def mean_force_n(
    beams: list[SimpleMOTBeam],
    atom_state: AtomState,
    coil_config,
    simple_config: SimpleMOTConfig | None = None,
) -> tuple[Vec3, list[SimpleMOTRateSample], Vec3]:
    """Return the deterministic mean radiation-pressure force."""

    samples, magnetic_field = rate_samples(beams, atom_state, coil_config, simple_config=simple_config)
    force = (0.0, 0.0, 0.0)
    for beam, sample in zip(beams, samples):
        force = add(force, scale(HBAR_J_S * wavevector_magnitude_m_inv(beam) * sample.scattering_rate_per_s, beam.direction))
    return force, samples, magnetic_field


def acceleration_m_per_s2(
    beams: list[SimpleMOTBeam],
    atom_state: AtomState,
    coil_config,
    simple_config: SimpleMOTConfig | None = None,
) -> tuple[Vec3, Vec3, list[SimpleMOTRateSample], Vec3]:
    """Return the deterministic MOT acceleration, force, rate samples, and B-field."""

    simple = simple_config or default_simple_mot_config()
    force, samples, magnetic_field = mean_force_n(beams, atom_state, coil_config, simple_config=simple)
    acceleration = scale(1.0 / RB87_MASS_KG, force)
    if simple.include_gravity:
        acceleration = add(acceleration, simple.gravity_acceleration_m_per_s2)
    return acceleration, force, samples, magnetic_field


def _derivative(
    beams: list[SimpleMOTBeam],
    atom_state: AtomState,
    coil_config,
    simple_config: SimpleMOTConfig,
) -> tuple[Vec3, Vec3, Vec3, list[SimpleMOTRateSample], Vec3]:
    acceleration, force, samples, magnetic_field = acceleration_m_per_s2(
        beams,
        atom_state,
        coil_config,
        simple_config=simple_config,
    )
    return atom_state.velocity_m_per_s, acceleration, force, samples, magnetic_field


def rk4_step(
    beams: list[SimpleMOTBeam],
    atom_state: AtomState,
    time_step_s: float,
    coil_config,
    simple_config: SimpleMOTConfig | None = None,
) -> tuple[AtomState, Vec3, list[SimpleMOTRateSample], Vec3]:
    """Advance one deterministic RK4 step."""

    simple = simple_config or default_simple_mot_config()

    k1_r, k1_v, force_1, samples_1, magnetic_field_1 = _derivative(beams, atom_state, coil_config, simple)
    state_2 = AtomState(
        position_m=add(atom_state.position_m, scale(0.5 * time_step_s, k1_r)),
        velocity_m_per_s=add(atom_state.velocity_m_per_s, scale(0.5 * time_step_s, k1_v)),
    )
    k2_r, k2_v, _, _, _ = _derivative(beams, state_2, coil_config, simple)
    state_3 = AtomState(
        position_m=add(atom_state.position_m, scale(0.5 * time_step_s, k2_r)),
        velocity_m_per_s=add(atom_state.velocity_m_per_s, scale(0.5 * time_step_s, k2_v)),
    )
    k3_r, k3_v, _, _, _ = _derivative(beams, state_3, coil_config, simple)
    state_4 = AtomState(
        position_m=add(atom_state.position_m, scale(time_step_s, k3_r)),
        velocity_m_per_s=add(atom_state.velocity_m_per_s, scale(time_step_s, k3_v)),
    )
    k4_r, k4_v, _, _, _ = _derivative(beams, state_4, coil_config, simple)

    position_increment = scale(
        time_step_s / 6.0,
        add(add(k1_r, scale(2.0, k2_r)), add(scale(2.0, k3_r), k4_r)),
    )
    velocity_increment = scale(
        time_step_s / 6.0,
        add(add(k1_v, scale(2.0, k2_v)), add(scale(2.0, k3_v), k4_v)),
    )
    next_state = AtomState(
        position_m=add(atom_state.position_m, position_increment),
        velocity_m_per_s=add(atom_state.velocity_m_per_s, velocity_increment),
    )
    return next_state, force_1, samples_1, magnetic_field_1


def simulate_simple_mot_trajectory(
    beams: list[SimpleMOTBeam],
    initial_state: AtomState,
    duration_s: float,
    time_step_s: float,
    coil_config,
    simple_config: SimpleMOTConfig | None = None,
) -> SimpleMOTTrajectoryRecord:
    """Run a deterministic two-level MOT trajectory."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    simple = simple_config or default_simple_mot_config()
    step_count = int(np.ceil(duration_s / time_step_s))

    atom_state = initial_state
    force_0, samples_0, magnetic_field_0 = mean_force_n(beams, atom_state, coil_config, simple_config=simple)
    beam_histories = {beam.label: [0.0] for beam in beams}
    for sample in samples_0:
        beam_histories[sample.beam_label][0] = sample.scattering_rate_per_s

    times_s = [0.0]
    positions_m = [atom_state.position_m]
    velocities_m_per_s = [atom_state.velocity_m_per_s]
    forces_n = [force_0]
    magnetic_fields_t = [magnetic_field_0]
    total_scattering_rates_per_s = [sum(sample.scattering_rate_per_s for sample in samples_0)]

    for step_index in range(step_count):
        atom_state, force, samples, magnetic_field = rk4_step(
            beams,
            atom_state,
            time_step_s,
            coil_config,
            simple_config=simple,
        )
        current_time = min(duration_s, (step_index + 1) * time_step_s)
        times_s.append(current_time)
        positions_m.append(atom_state.position_m)
        velocities_m_per_s.append(atom_state.velocity_m_per_s)
        forces_n.append(force)
        magnetic_fields_t.append(magnetic_field)
        total_scattering_rates_per_s.append(sum(sample.scattering_rate_per_s for sample in samples))
        sample_map = {sample.beam_label: sample.scattering_rate_per_s for sample in samples}
        for beam in beams:
            beam_histories.setdefault(beam.label, [])
            beam_histories[beam.label].append(sample_map.get(beam.label, 0.0))

    return SimpleMOTTrajectoryRecord(
        times_s=times_s,
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        forces_n=forces_n,
        magnetic_fields_t=magnetic_fields_t,
        total_scattering_rates_per_s=total_scattering_rates_per_s,
        beam_scattering_rates_per_s=beam_histories,
    )
