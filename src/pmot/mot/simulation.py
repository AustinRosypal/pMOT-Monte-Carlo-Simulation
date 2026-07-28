"""Zeeman-aware MOT scattering and trajectory simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from math import pi
from math import sqrt

import numpy as np

from ..atomic_data import RB87CoolingTransition
from ..configuration import GRAVITY_ACCELERATION_M_PER_S2
from ..configuration import HBAR_J_S
from ..configuration import RB87_MASS_KG
from ..fields import MOTBeam
from ..fields import beam_intensity_w_per_m2
from .configuration import AntiHelmholtzCoilConfig
from .magnetic_fields import anti_helmholtz_field_t
from .zeeman import BOHR_MAGNETON_OVER_H_HZ_PER_T
from .zeeman import ExcitedState
from .zeeman import GroundState
from .zeeman import addressed_excited_states_for_beam_family
from .zeeman import beam_polarization_vector
from .zeeman import decay_branching_probabilities
from .zeeman import dot
from .zeeman import initial_ground_state
from .zeeman import norm
from .zeeman import polarization_weights_for_quantization_axis
from .zeeman import transition_strength
from .zeeman import zero_field_polarization_weights
from .zeeman import zero_field_transition_frequency_hz
from .zeeman import zeeman_shift_hz


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MOTAtomState:
    """One classical-plus-internal MOT atom state."""

    position_m: Vec3
    velocity_m_per_s: Vec3
    ground_state: GroundState
    last_quantization_axis: Vec3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class TransitionRateSample:
    """One beam-resolved Zeeman transition rate."""

    beam_label: str
    beam_family: str
    axis_name: str
    q_value: int
    excited_state: ExcitedState
    polarization_weight: float
    line_strength: float
    intensity_w_per_m2: float
    beam_saturation_parameter: float
    total_saturation_parameter: float
    effective_detuning_hz: float
    scattering_rate_per_s: float


@dataclass(frozen=True, slots=True)
class MOTTrajectoryRecord:
    """Saved single-atom MOT simulation outputs."""

    times_s: list[float]
    positions_m: list[Vec3]
    velocities_m_per_s: list[Vec3]
    ground_states: list[GroundState]
    quantization_axes: list[Vec3]
    magnetic_fields_t: list[Vec3]
    total_scattering_rates_per_s: list[float]
    cooling_scattering_rates_per_s: list[float]
    repump_scattering_rates_per_s: list[float]
    axis_scattering_rates_per_s: dict[str, list[float]]
    event_counts: list[int]


@dataclass(frozen=True, slots=True)
class ManyAtomRecord:
    """Saved many-atom MOT evolution."""

    times_s: list[float]
    positions_m_by_time: list[np.ndarray]
    speeds_m_per_s_by_time: list[np.ndarray]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def unit(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


def wavevector_magnitude_m_inv(beam: MOTBeam) -> float:
    return 2.0 * pi / beam.wavelength_m


def doppler_shift_hz(beam: MOTBeam, velocity_m_per_s: Vec3) -> float:
    return wavevector_magnitude_m_inv(beam) * dot(beam.direction, velocity_m_per_s) / (2.0 * pi)


def recoil_velocity_m_per_s(beam: MOTBeam, atom_mass_kg: float = RB87_MASS_KG) -> float:
    return HBAR_J_S * wavevector_magnitude_m_inv(beam) / atom_mass_kg


def gravitational_velocity_increment(time_step_s: float) -> Vec3:
    """Return the velocity increment from gravity over one timestep."""

    return scale(time_step_s, GRAVITY_ACCELERATION_M_PER_S2)


def isotropic_unit_vector(rng: np.random.Generator) -> Vec3:
    z_value = rng.uniform(-1.0, 1.0)
    azimuth = rng.uniform(0.0, 2.0 * pi)
    radial = sqrt(max(0.0, 1.0 - z_value * z_value))
    return (radial * float(np.cos(azimuth)), radial * float(np.sin(azimuth)), z_value)


def quantization_axis(field_vector_t: Vec3, fallback_axis: Vec3) -> Vec3:
    magnitude = norm(field_vector_t)
    if magnitude <= 1.0e-18:
        return fallback_axis
    return unit(field_vector_t)


def local_magnetic_field_t(position_m: Vec3, coil_config: AntiHelmholtzCoilConfig) -> Vec3:
    bx, by, bz = anti_helmholtz_field_t(position_m[0], position_m[1], position_m[2], coil_config)
    return (float(np.asarray(bx)), float(np.asarray(by)), float(np.asarray(bz)))


def effective_detuning_hz(
    beam: MOTBeam,
    ground_state: GroundState,
    excited_state: ExcitedState,
    field_magnitude_t: float,
    velocity_m_per_s: Vec3,
) -> float:
    laser_frequency_hz = beam.laser_frequency_hz
    resonance_hz = zero_field_transition_frequency_hz(ground_state.f, excited_state.f_prime)
    return (
        laser_frequency_hz
        - resonance_hz
        - doppler_shift_hz(beam, velocity_m_per_s)
        - zeeman_shift_hz(field_magnitude_t, ground_state, excited_state)
    )


def transition_rate_samples(
    beams: list[MOTBeam],
    atom_state: MOTAtomState,
    coil_config: AntiHelmholtzCoilConfig,
) -> tuple[list[TransitionRateSample], Vec3]:
    """Return all allowed beam-resolved Zeeman transition rates."""

    field_vector_t = local_magnetic_field_t(atom_state.position_m, coil_config)
    field_magnitude_t = norm(field_vector_t)
    q_axis = quantization_axis(field_vector_t, atom_state.last_quantization_axis)
    transition = RB87CoolingTransition()
    beam_intensities = {beam.label: beam_intensity_w_per_m2(beam, atom_state.position_m) for beam in beams}
    active_beams = [
        beam for beam in beams if addressed_excited_states_for_beam_family(atom_state.ground_state, beam.family)
    ]
    total_saturation = sum(beam_intensities[beam.label] / transition.saturation_intensity_w_per_m2 for beam in active_beams)

    samples: list[TransitionRateSample] = []
    for beam in beams:
        intensity = beam_intensities[beam.label]
        if field_magnitude_t <= 1.0e-18:
            polarization_weights = zero_field_polarization_weights()
        else:
            polarization_weights = polarization_weights_for_quantization_axis(beam, q_axis)
        beam_saturation = intensity / transition.saturation_intensity_w_per_m2
        for excited_state in addressed_excited_states_for_beam_family(atom_state.ground_state, beam.family):
            q_value = excited_state.m_f_prime - atom_state.ground_state.m_f
            polarization_weight = polarization_weights[q_value]
            line_strength = transition_strength(
                atom_state.ground_state.f,
                atom_state.ground_state.m_f,
                excited_state.f_prime,
                excited_state.m_f_prime,
            )
            effective_saturation = beam_saturation * polarization_weight * line_strength
            if effective_saturation <= 0.0:
                continue
            detuning_hz = effective_detuning_hz(
                beam=beam,
                ground_state=atom_state.ground_state,
                excited_state=excited_state,
                field_magnitude_t=field_magnitude_t,
                velocity_m_per_s=atom_state.velocity_m_per_s,
            )
            linewidth_hz = transition.linewidth_hz
            rate = 0.5 * linewidth_hz * effective_saturation / (
                1.0 + total_saturation + (2.0 * detuning_hz / linewidth_hz) ** 2
            )
            samples.append(
                TransitionRateSample(
                    beam_label=beam.label,
                    beam_family=beam.family,
                    axis_name=beam.axis_name,
                    q_value=q_value,
                    excited_state=excited_state,
                    polarization_weight=polarization_weight,
                    line_strength=line_strength,
                    intensity_w_per_m2=intensity,
                    beam_saturation_parameter=effective_saturation,
                    total_saturation_parameter=total_saturation,
                    effective_detuning_hz=detuning_hz,
                    scattering_rate_per_s=rate,
                )
            )
    return samples, q_axis


def total_scattering_rate_per_s(samples: list[TransitionRateSample]) -> float:
    return sum(sample.scattering_rate_per_s for sample in samples)


def _weighted_choice(weights: list[float], rng: np.random.Generator) -> int:
    cumulative = np.cumsum(np.asarray(weights, dtype=float))
    target = rng.uniform(0.0, float(cumulative[-1]))
    return int(np.searchsorted(cumulative, target, side="right"))


def beam_scattering_rates(samples: list[TransitionRateSample]) -> dict[str, float]:
    """Return the total scattering rate for each beam."""

    rates: dict[str, float] = {}
    for sample in samples:
        rates[sample.beam_label] = rates.get(sample.beam_label, 0.0) + sample.scattering_rate_per_s
    return rates


def advance_mot_atom_one_step(
    beams: list[MOTBeam],
    coil_config: AntiHelmholtzCoilConfig,
    atom_state: MOTAtomState,
    time_step_s: float,
    rng: np.random.Generator,
    atom_mass_kg: float = RB87_MASS_KG,
) -> tuple[MOTAtomState, int]:
    """Advance one MOT atom by one timestep."""

    transition_samples, q_axis = transition_rate_samples(beams, atom_state, coil_config)
    total_rate = total_scattering_rate_per_s(transition_samples)
    event_count = 0
    updated_velocity = atom_state.velocity_m_per_s
    updated_ground_state = atom_state.ground_state

    if total_rate > 0.0:
        scatter_probability = 1.0 - exp(-total_rate * time_step_s)
        if rng.uniform() < scatter_probability:
            rates_by_beam = beam_scattering_rates(transition_samples)
            beam_labels = list(rates_by_beam)
            beam_weights = [rates_by_beam[label] for label in beam_labels]
            selected_beam_label = beam_labels[_weighted_choice(beam_weights, rng)]
            beam_samples = [sample for sample in transition_samples if sample.beam_label == selected_beam_label]
            transition_weights = [sample.scattering_rate_per_s for sample in beam_samples]
            selected = beam_samples[_weighted_choice(transition_weights, rng)]
            selected_beam = next(beam for beam in beams if beam.label == selected.beam_label)
            kick_speed = recoil_velocity_m_per_s(selected_beam, atom_mass_kg=atom_mass_kg)
            updated_velocity = add(updated_velocity, scale(kick_speed, selected_beam.direction))
            decay_channels = decay_branching_probabilities(selected.excited_state)
            decay_weights = [weight for _, weight in decay_channels]
            updated_ground_state = decay_channels[_weighted_choice(decay_weights, rng)][0]
            updated_velocity = add(updated_velocity, scale(kick_speed, isotropic_unit_vector(rng)))
            event_count = 1

    updated_velocity = add(updated_velocity, gravitational_velocity_increment(time_step_s))

    updated_position = add(atom_state.position_m, scale(time_step_s, updated_velocity))
    next_state = MOTAtomState(
        position_m=updated_position,
        velocity_m_per_s=updated_velocity,
        ground_state=updated_ground_state,
        last_quantization_axis=q_axis,
    )
    return next_state, event_count


def simulate_mot_trajectory(
    beams: list[MOTBeam],
    coil_config: AntiHelmholtzCoilConfig,
    initial_state: MOTAtomState,
    duration_s: float,
    time_step_s: float,
    seed: int = 0,
    atom_mass_kg: float = RB87_MASS_KG,
) -> MOTTrajectoryRecord:
    """Propagate one atom with Zeeman-aware MOT scattering."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    step_count = int(np.ceil(duration_s / time_step_s))
    rng = np.random.default_rng(seed)

    atom_state = initial_state
    transition_samples, q_axis = transition_rate_samples(beams, atom_state, coil_config)
    magnetic_field = local_magnetic_field_t(atom_state.position_m, coil_config)

    times_s = [0.0]
    positions_m = [atom_state.position_m]
    velocities_m_per_s = [atom_state.velocity_m_per_s]
    ground_states = [atom_state.ground_state]
    quantization_axes = [q_axis]
    magnetic_fields_t = [magnetic_field]
    total_scattering_rates_per_s = [total_scattering_rate_per_s(transition_samples)]
    cooling_scattering_rates_per_s = [
        sum(sample.scattering_rate_per_s for sample in transition_samples if sample.beam_family == "cooling")
    ]
    repump_scattering_rates_per_s = [
        sum(sample.scattering_rate_per_s for sample in transition_samples if sample.beam_family == "repump")
    ]
    axis_scattering_rates_per_s = {
        "horizontal_x": [
            sum(sample.scattering_rate_per_s for sample in transition_samples if sample.axis_name == "horizontal_x")
        ],
        "horizontal_y": [
            sum(sample.scattering_rate_per_s for sample in transition_samples if sample.axis_name == "horizontal_y")
        ],
        "vertical_z": [
            sum(sample.scattering_rate_per_s for sample in transition_samples if sample.axis_name == "vertical_z")
        ],
    }
    event_counts = [0]

    for step_index in range(step_count):
        atom_state, event_count = advance_mot_atom_one_step(
            beams=beams,
            coil_config=coil_config,
            atom_state=atom_state,
            time_step_s=time_step_s,
            rng=rng,
            atom_mass_kg=atom_mass_kg,
        )
        magnetic_field = local_magnetic_field_t(atom_state.position_m, coil_config)
        transition_samples, next_axis = transition_rate_samples(beams, atom_state, coil_config)

        current_time_s = min(duration_s, (step_index + 1) * time_step_s)
        times_s.append(current_time_s)
        positions_m.append(atom_state.position_m)
        velocities_m_per_s.append(atom_state.velocity_m_per_s)
        ground_states.append(atom_state.ground_state)
        quantization_axes.append(next_axis)
        magnetic_fields_t.append(magnetic_field)
        total_scattering_rates_per_s.append(total_scattering_rate_per_s(transition_samples))
        cooling_scattering_rates_per_s.append(
            sum(sample.scattering_rate_per_s for sample in transition_samples if sample.beam_family == "cooling")
        )
        repump_scattering_rates_per_s.append(
            sum(sample.scattering_rate_per_s for sample in transition_samples if sample.beam_family == "repump")
        )
        for axis_name in axis_scattering_rates_per_s:
            axis_scattering_rates_per_s[axis_name].append(
                sum(sample.scattering_rate_per_s for sample in transition_samples if sample.axis_name == axis_name)
            )
        event_counts.append(event_count)

    return MOTTrajectoryRecord(
        times_s=times_s,
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        ground_states=ground_states,
        quantization_axes=quantization_axes,
        magnetic_fields_t=magnetic_fields_t,
        total_scattering_rates_per_s=total_scattering_rates_per_s,
        cooling_scattering_rates_per_s=cooling_scattering_rates_per_s,
        repump_scattering_rates_per_s=repump_scattering_rates_per_s,
        axis_scattering_rates_per_s=axis_scattering_rates_per_s,
        event_counts=event_counts,
    )


def random_initial_mot_state(
    radius_m: float,
    speed_m_per_s: float,
    rng: np.random.Generator,
) -> MOTAtomState:
    """Sample a many-atom initial condition on a sphere, launched inward."""

    cos_theta = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * pi)
    sin_theta = sqrt(max(0.0, 1.0 - cos_theta**2))
    outward = (
        sin_theta * float(np.cos(phi)),
        sin_theta * float(np.sin(phi)),
        cos_theta,
    )
    return MOTAtomState(
        position_m=scale(radius_m, outward),
        velocity_m_per_s=scale(-speed_m_per_s, outward),
        ground_state=initial_ground_state(rng),
        last_quantization_axis=(0.0, 0.0, 1.0),
    )


def simulate_many_atoms(
    beams: list[MOTBeam],
    coil_config: AntiHelmholtzCoilConfig,
    atom_count: int,
    radius_m: float,
    speed_m_per_s: float,
    duration_s: float,
    time_step_s: float,
    seed: int = 0,
) -> ManyAtomRecord:
    """Simulate many atoms independently and save synchronized snapshots."""

    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    rng = np.random.default_rng(seed)
    states = [random_initial_mot_state(radius_m, speed_m_per_s, rng) for _ in range(atom_count)]
    step_count = int(np.ceil(duration_s / time_step_s))

    times_s = [0.0]
    positions_m_by_time = [np.array([state.position_m for state in states], dtype=float)]
    speeds_m_per_s_by_time = [np.array([norm(state.velocity_m_per_s) for state in states], dtype=float)]

    for step_index in range(step_count):
        next_states: list[MOTAtomState] = []
        for state in states:
            next_state, _ = advance_mot_atom_one_step(
                beams=beams,
                coil_config=coil_config,
                time_step_s=time_step_s,
                atom_state=state,
                rng=rng,
            )
            next_states.append(next_state)
        states = next_states
        times_s.append(min(duration_s, (step_index + 1) * time_step_s))
        positions_m_by_time.append(np.array([state.position_m for state in states], dtype=float))
        speeds_m_per_s_by_time.append(np.array([norm(state.velocity_m_per_s) for state in states], dtype=float))

    return ManyAtomRecord(
        times_s=times_s,
        positions_m_by_time=positions_m_by_time,
        speeds_m_per_s_by_time=speeds_m_per_s_by_time,
    )
