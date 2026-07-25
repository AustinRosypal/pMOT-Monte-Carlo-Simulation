"""Velocity-dependent photon scattering and recoil utilities for Rb-87 MOT beams."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .atomic_data import RB87CoolingTransition
from .atomic_data import RB87RepumpTransition
from .configuration import HBAR_J_S
from .configuration import GRAVITY_ACCELERATION_M_PER_S2
from .configuration import RB87_MASS_KG
from .fields import MOTBeam
from .fields import beam_intensity_w_per_m2
from .fields import total_intensity_w_per_m2


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AtomState:
    """One classical atom state in the Monte Carlo simulation."""

    position_m: Vec3
    velocity_m_per_s: Vec3


@dataclass(frozen=True, slots=True)
class ScatteringSample:
    """Local scattering information for one beam at one atom state."""

    beam_label: str
    family: str
    intensity_w_per_m2: float
    saturation_parameter: float
    effective_detuning_hz: float
    scattering_rate_per_s: float


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """Saved single-atom trajectory outputs."""

    times_s: list[float]
    positions_m: list[Vec3]
    velocities_m_per_s: list[Vec3]
    total_scattering_rates_per_s: list[float]
    cooling_scattering_rates_per_s: list[float]
    repump_scattering_rates_per_s: list[float]
    total_intensities_w_per_m2: list[float]
    absorption_kick_directions: list[Vec3]
    emission_kick_directions: list[Vec3]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(vector: Vec3) -> float:
    return float(np.sqrt(dot(vector, vector)))


def gravitational_velocity_increment(time_step_s: float) -> Vec3:
    """Return the velocity increment from gravity over one timestep."""

    return scale(time_step_s, GRAVITY_ACCELERATION_M_PER_S2)


def transition_for_beam(beam: MOTBeam) -> RB87CoolingTransition | RB87RepumpTransition:
    """Return the natural-linewidth and saturation data for one beam family."""

    if beam.family == "cooling":
        return RB87CoolingTransition()
    if beam.family == "repump":
        return RB87RepumpTransition()
    raise ValueError(f"unsupported beam family: {beam.family}")


def transition_for_active_manifold(active_transition: str) -> RB87CoolingTransition | RB87RepumpTransition:
    """Return the transition model for the hyperfine manifold occupied by the atom."""

    if active_transition == "cooling":
        return RB87CoolingTransition()
    if active_transition == "repump":
        return RB87RepumpTransition()
    raise ValueError("active_transition must be 'cooling' or 'repump'")


def wavevector_magnitude_m_inv(beam: MOTBeam) -> float:
    """Return the optical wavenumber magnitude."""

    return 2.0 * pi / beam.wavelength_m


def recoil_velocity_m_per_s(beam: MOTBeam, atom_mass_kg: float = RB87_MASS_KG) -> float:
    """Return the one-photon recoil speed for the beam wavelength."""

    return HBAR_J_S * wavevector_magnitude_m_inv(beam) / atom_mass_kg


def doppler_shift_hz(beam: MOTBeam, velocity_m_per_s: Vec3) -> float:
    """Return the first-order Doppler shift seen by the moving atom."""

    return dot(scale(wavevector_magnitude_m_inv(beam), beam.direction), velocity_m_per_s) / (2.0 * pi)


def effective_detuning_hz(
    beam: MOTBeam,
    velocity_m_per_s: Vec3,
    active_transition: str = "cooling",
) -> float:
    """Return the laser detuning in the atom frame for the chosen active manifold."""

    transition = transition_for_active_manifold(active_transition)
    laser_detuning_hz = beam.laser_frequency_hz - transition.resonance_frequency_hz
    return laser_detuning_hz - doppler_shift_hz(beam, velocity_m_per_s)


def saturation_parameter(
    beam: MOTBeam,
    position_m: Vec3,
    active_transition: str = "cooling",
) -> float:
    """Return the local saturation parameter s = I / I_sat."""

    transition = transition_for_active_manifold(active_transition)
    intensity = beam_intensity_w_per_m2(beam, position_m)
    return intensity / transition.saturation_intensity_w_per_m2


def scattering_rate_per_s(
    beam: MOTBeam,
    atom_state: AtomState,
    active_transition: str = "cooling",
) -> float:
    """Return the photon scattering rate for one beam at the current atom state."""

    transition = transition_for_active_manifold(active_transition)
    s_value = saturation_parameter(beam, atom_state.position_m, active_transition=active_transition)
    delta_hz = effective_detuning_hz(beam, atom_state.velocity_m_per_s, active_transition=active_transition)
    linewidth_hz = transition.linewidth_hz
    denominator = 1.0 + s_value + (2.0 * delta_hz / linewidth_hz) ** 2
    return 0.5 * linewidth_hz * s_value / denominator


def scattering_sample(
    beam: MOTBeam,
    atom_state: AtomState,
    active_transition: str = "cooling",
) -> ScatteringSample:
    """Return local intensity, detuning, saturation, and rate for one beam."""

    intensity = beam_intensity_w_per_m2(beam, atom_state.position_m)
    transition = transition_for_active_manifold(active_transition)
    s_value = intensity / transition.saturation_intensity_w_per_m2
    delta_hz = effective_detuning_hz(beam, atom_state.velocity_m_per_s, active_transition=active_transition)
    linewidth_hz = transition.linewidth_hz
    rate = 0.5 * linewidth_hz * s_value / (1.0 + s_value + (2.0 * delta_hz / linewidth_hz) ** 2)
    return ScatteringSample(
        beam_label=beam.label,
        family=beam.family,
        intensity_w_per_m2=intensity,
        saturation_parameter=s_value,
        effective_detuning_hz=delta_hz,
        scattering_rate_per_s=rate,
    )


def beam_scattering_samples(
    beams: list[MOTBeam],
    atom_state: AtomState,
    active_transition: str = "cooling",
) -> list[ScatteringSample]:
    """Return scattering diagnostics for every beam."""

    return [scattering_sample(beam, atom_state, active_transition=active_transition) for beam in beams]


def total_scattering_rate_per_s(
    beams: list[MOTBeam],
    atom_state: AtomState,
    family: str | None = None,
    active_transition: str = "cooling",
) -> float:
    """Return the summed scattering rate across all selected beams."""

    rate = 0.0
    for beam in beams:
        if family is not None and beam.family != family:
            continue
        rate += scattering_rate_per_s(beam, atom_state, active_transition=active_transition)
    return rate


def isotropic_unit_vector(rng: np.random.Generator) -> Vec3:
    """Sample a random 3D unit vector for spontaneous emission."""

    z_value = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * pi)
    radial = float(np.sqrt(max(0.0, 1.0 - z_value * z_value)))
    return (radial * float(np.cos(azimuth)), radial * float(np.sin(azimuth)), z_value)


def absorption_kick_velocity_m_per_s(beam: MOTBeam, atom_mass_kg: float = RB87_MASS_KG) -> Vec3:
    """Return the directed recoil velocity from absorbing one photon."""

    return scale(recoil_velocity_m_per_s(beam, atom_mass_kg), beam.direction)


def emission_kick_velocity_m_per_s(
    beam: MOTBeam,
    rng: np.random.Generator,
    atom_mass_kg: float = RB87_MASS_KG,
) -> Vec3:
    """Return the isotropic recoil velocity from one spontaneous emission."""

    return scale(recoil_velocity_m_per_s(beam, atom_mass_kg), isotropic_unit_vector(rng))


def simulate_scattering_trajectory(
    beams: list[MOTBeam],
    initial_state: AtomState,
    duration_s: float,
    time_step_s: float,
    seed: int = 0,
    atom_mass_kg: float = RB87_MASS_KG,
    active_transition: str = "cooling",
) -> TrajectoryRecord:
    """Propagate one atom using stochastic absorption and spontaneous-emission kicks."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    step_count = int(np.ceil(duration_s / time_step_s))
    rng = np.random.default_rng(seed)

    position = initial_state.position_m
    velocity = initial_state.velocity_m_per_s

    times_s: list[float] = [0.0]
    positions_m: list[Vec3] = [position]
    velocities_m_per_s: list[Vec3] = [velocity]
    total_scattering_rates_per_s: list[float] = [
        total_scattering_rate_per_s(beams, initial_state, active_transition=active_transition)
    ]
    cooling_scattering_rates_per_s: list[float] = [
        total_scattering_rate_per_s(beams, initial_state, family="cooling", active_transition=active_transition)
    ]
    repump_scattering_rates_per_s: list[float] = [
        total_scattering_rate_per_s(beams, initial_state, family="repump", active_transition=active_transition)
    ]
    total_intensities_w_per_m2: list[float] = [total_intensity_w_per_m2(beams, position)]
    absorption_kick_directions: list[Vec3] = []
    emission_kick_directions: list[Vec3] = []

    for step_index in range(step_count):
        atom_state = AtomState(position_m=position, velocity_m_per_s=velocity)
        updated_velocity = velocity
        for beam in beams:
            rate_per_s = scattering_rate_per_s(beam, atom_state, active_transition=active_transition)
            mean_event_count = rate_per_s * time_step_s
            event_count = int(rng.poisson(mean_event_count))
            if event_count == 0:
                continue
            absorption_kick = absorption_kick_velocity_m_per_s(beam, atom_mass_kg)
            for _ in range(event_count):
                absorption_kick_directions.append(beam.direction)
                updated_velocity = add(updated_velocity, absorption_kick)
                emission_direction = isotropic_unit_vector(rng)
                emission_kick_directions.append(emission_direction)
                updated_velocity = add(
                    updated_velocity,
                    scale(recoil_velocity_m_per_s(beam, atom_mass_kg), emission_direction),
                )

        updated_velocity = add(updated_velocity, gravitational_velocity_increment(time_step_s))

        position = add(position, scale(time_step_s, updated_velocity))
        velocity = updated_velocity

        current_time_s = min(duration_s, (step_index + 1) * time_step_s)
        atom_state = AtomState(position_m=position, velocity_m_per_s=velocity)
        times_s.append(current_time_s)
        positions_m.append(position)
        velocities_m_per_s.append(velocity)
        total_scattering_rates_per_s.append(
            total_scattering_rate_per_s(beams, atom_state, active_transition=active_transition)
        )
        cooling_scattering_rates_per_s.append(
            total_scattering_rate_per_s(beams, atom_state, family="cooling", active_transition=active_transition)
        )
        repump_scattering_rates_per_s.append(
            total_scattering_rate_per_s(beams, atom_state, family="repump", active_transition=active_transition)
        )
        total_intensities_w_per_m2.append(total_intensity_w_per_m2(beams, position))

    return TrajectoryRecord(
        times_s=times_s,
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        total_scattering_rates_per_s=total_scattering_rates_per_s,
        cooling_scattering_rates_per_s=cooling_scattering_rates_per_s,
        repump_scattering_rates_per_s=repump_scattering_rates_per_s,
        total_intensities_w_per_m2=total_intensities_w_per_m2,
        absorption_kick_directions=absorption_kick_directions,
        emission_kick_directions=emission_kick_directions,
    )
