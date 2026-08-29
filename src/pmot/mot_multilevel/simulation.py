"""Event-driven classical trajectories and diagnostic mean observables.

The mean force here is conditional on a specified ground-state population. It
is a force-law diagnostic, not a replacement for stochastic optical pumping.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import inf, pi

import numpy as np

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2, HBAR_J_S, RB87_MASS_KG
from ..configuration import RB87_REPUMP_RESONANCE_HZ
from ..fields import MOTBeam, build_mot_beams
from ..magnetic_fields import anti_helmholtz_field_t
from .atomic_structure import AtomicStructure, build_atomic_structure
from .configuration import DarkStateBehavior, MultilevelMOTConfig, default_multilevel_mot_config
from .coupling import ground_laser_channels, wavevector_rad_per_m
from .events import EventChannel, outgoing_channels, sample_next_event
from .polarization import Vec3, quantization_axis
from .trajectory import (
    MultilevelAtomState,
    RepumpAbsorptionRecord,
    TrajectoryCounters,
    absorption_velocity_kick,
    spontaneous_emission_velocity_kick,
    stimulated_emission_velocity_kick,
)


@dataclass(frozen=True, slots=True)
class MeanObservable:
    force_n: Vec3
    total_absorption_rate_per_s: float
    beam_absorption_rates_per_s: tuple[float, ...]
    magnetic_field_t: Vec3


@dataclass(slots=True)
class MultilevelTrajectoryRecord:
    times_s: list[float] = field(default_factory=list)
    positions_m: list[Vec3] = field(default_factory=list)
    velocities_m_per_s: list[Vec3] = field(default_factory=list)
    mean_forces_n: list[Vec3] = field(default_factory=list)
    total_scattering_rates_per_s: list[float] = field(default_factory=list)
    beam_scattering_rates_per_s: list[tuple[float, ...]] = field(default_factory=list)
    magnetic_fields_t: list[Vec3] = field(default_factory=list)
    internal_state_indices: list[int] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    event_beam_indices: list[int | None] = field(default_factory=list)
    f1_visit_durations_s: list[float] = field(default_factory=list)
    repump_photons_per_f1_return: list[int] = field(default_factory=list)
    repump_absorptions: list[RepumpAbsorptionRecord] = field(default_factory=list)
    counters: TrajectoryCounters = field(default_factory=TrajectoryCounters)
    termination_reason: str = "duration"


def build_multilevel_cooling_beams(apparatus_config=None) -> list[MOTBeam]:
    """Build six cooling beams with the propagation-frame MOT helicities.

    Opposite fixed-axis circular polarizations correspond to equal helicity
    labels when each label is defined looking along that beam's own k vector.
    The z choice reverses because the quadrupole axial gradient has the opposite
    sign to its two radial gradients.
    """

    helicity_by_axis = {"horizontal_x": "sigma+", "horizontal_y": "sigma+", "vertical_z": "sigma-"}
    return [
        replace(beam, circular_polarization=helicity_by_axis[beam.axis_name])
        for beam in build_mot_beams(apparatus_config)
        if beam.family == "cooling"
    ]


def build_multilevel_repump_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build repump components sharing the six cooling-beam paths."""

    cfg = config or default_multilevel_mot_config()
    repump_detuning_hz = cfg.repump_detuning_rad_per_s / (2.0 * pi)
    return [
        replace(
            beam,
            label=beam.label.replace("_cooling_", "_repump_"),
            family="repump",
            wavelength_m=cfg.repump_wavelength_m,
            resonance_frequency_hz=RB87_REPUMP_RESONANCE_HZ,
            detuning_hz=repump_detuning_hz,
            power_w=cfg.repump_power_w_per_beam,
        )
        for beam in build_multilevel_cooling_beams(apparatus_config)
    ]


def build_multilevel_mot_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build cooling beams and, when enabled, co-propagating repump components."""

    cfg = config or default_multilevel_mot_config()
    beams = build_multilevel_cooling_beams(apparatus_config)
    if cfg.repumper_enabled and cfg.repump_power_w_per_beam > 0.0:
        beams += build_multilevel_repump_beams(apparatus_config, cfg)
    return beams


def local_magnetic_field_t(position_m: Vec3, coil_config) -> Vec3:
    values = anti_helmholtz_field_t(*position_m, coil_config)
    return tuple(float(np.asarray(value)) for value in values)


def _axis_and_field(position_m: Vec3, previous_axis: Vec3, coil_config, config: MultilevelMOTConfig):
    field_t = local_magnetic_field_t(position_m, coil_config)
    axis = quantization_axis(field_t, previous_axis, config.magnetic_field_epsilon_t)
    return axis, field_t


def ground_state_mean_observable(
    structure: AtomicStructure,
    ground_state_index: int,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    coil_config,
    config: MultilevelMOTConfig | None = None,
    previous_axis: Vec3 = (0.0, 0.0, 1.0),
) -> MeanObservable:
    """Return absorption rate and mean radiation force for one ground sublevel."""

    cfg = config or default_multilevel_mot_config()
    axis, field_t = _axis_and_field(position_m, previous_axis, coil_config, cfg)
    channels = ground_laser_channels(
        structure, ground_state_index, beams, position_m, velocity_m_per_s,
        float(np.linalg.norm(field_t)), axis, cfg,
    )
    beam_rates = np.zeros(len(beams), dtype=float)
    force = np.zeros(3, dtype=float)
    for channel in channels:
        beam_rates[channel.beam_index] += channel.rate_per_s
        force += HBAR_J_S * np.asarray(wavevector_rad_per_m(beams[channel.beam_index])) * channel.rate_per_s
    return MeanObservable(tuple(force), float(np.sum(beam_rates)), tuple(beam_rates), field_t)


def unpolarized_f2_mean_observable(
    structure: AtomicStructure,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    coil_config,
    config: MultilevelMOTConfig | None = None,
) -> MeanObservable:
    """Average the instantaneous observable equally over all five F=2 states."""

    samples = [
        ground_state_mean_observable(
            structure, structure.state_index("ground", 2, m_f), beams,
            position_m, velocity_m_per_s, coil_config, config,
        )
        for m_f in range(-2, 3)
    ]
    return MeanObservable(
        tuple(np.mean([sample.force_n for sample in samples], axis=0)),
        float(np.mean([sample.total_absorption_rate_per_s for sample in samples])),
        tuple(np.mean([sample.beam_absorption_rates_per_s for sample in samples], axis=0)),
        samples[0].magnetic_field_t,
    )


def _propagate_ballistic(position: Vec3, velocity: Vec3, dt_s: float, include_gravity: bool):
    gravity = np.asarray(GRAVITY_ACCELERATION_M_PER_S2 if include_gravity else (0.0, 0.0, 0.0))
    r = np.asarray(position) + np.asarray(velocity) * dt_s + 0.5 * gravity * dt_s**2
    v = np.asarray(velocity) + gravity * dt_s
    return tuple(r), tuple(v)


def _is_outward_escape(position: Vec3, velocity: Vec3, escape_radius_m: float | None) -> bool:
    """Return whether the atom is beyond the escape sphere and moving outward."""

    if escape_radius_m is None:
        return False
    position_array = np.asarray(position)
    radius_m = float(np.linalg.norm(position_array))
    if radius_m < escape_radius_m:
        return False
    radial_velocity_m_per_s = float(np.dot(position_array, velocity)) / max(radius_m, 1.0e-15)
    return radial_velocity_m_per_s > 0.0


def _append_record(record, time_s, state, observable, field_t, event_type, beam_index):
    record.times_s.append(time_s)
    record.positions_m.append(state.position_m)
    record.velocities_m_per_s.append(state.velocity_m_per_s)
    record.mean_forces_n.append(observable.force_n)
    record.total_scattering_rates_per_s.append(observable.total_absorption_rate_per_s)
    record.beam_scattering_rates_per_s.append(observable.beam_absorption_rates_per_s)
    record.magnetic_fields_t.append(field_t)
    record.internal_state_indices.append(state.internal_state_index)
    record.event_types.append(event_type)
    record.event_beam_indices.append(beam_index)


def simulate_multilevel_trajectory(
    initial_state: MultilevelAtomState,
    duration_s: float,
    coil_config,
    *,
    beams: list[MOTBeam] | None = None,
    structure: AtomicStructure | None = None,
    config: MultilevelMOTConfig | None = None,
    seed: int = 12345,
    max_events: int = 100_000,
    escape_radius_m: float | None = None,
) -> MultilevelTrajectoryRecord:
    """Simulate one piecewise-ballistic, event-driven multilevel trajectory.

    Rates are held constant over each sampled Gillespie interval. This is
    accurate while the atom moves negligibly on an optical-event timescale;
    long dark intervals are propagated exactly under gravity.
    """

    if duration_s <= 0.0 or max_events <= 0:
        raise ValueError("duration_s and max_events must be positive")
    if escape_radius_m is not None and escape_radius_m <= 0.0:
        raise ValueError("escape_radius_m must be positive when provided")
    cfg = config or default_multilevel_mot_config()
    atom_structure = structure or build_atomic_structure()
    optical_beams = build_multilevel_mot_beams(config=cfg) if beams is None else beams
    rng = np.random.default_rng(seed)
    state = initial_state
    record = MultilevelTrajectoryRecord()
    time_s = 0.0
    event_count = 0
    f1_visit_start_s = 0.0 if atom_structure.states[state.internal_state_index].is_dark else None
    repump_photons_since_f1_entry = 0
    last_absorption_wavelength_m = cfg.wavelength_m
    if f1_visit_start_s is not None:
        record.counters.f1_visit_count = 1

    axis, field_t = _axis_and_field(state.position_m, state.last_quantization_axis, coil_config, cfg)
    internal = atom_structure.states[state.internal_state_index]
    if internal.is_ground:
        observable = ground_state_mean_observable(
            atom_structure, state.internal_state_index, optical_beams,
            state.position_m, state.velocity_m_per_s, coil_config, cfg, axis,
        )
    else:
        observable = MeanObservable((0.0, 0.0, 0.0), 0.0, tuple(0.0 for _ in optical_beams), field_t)
    _append_record(record, time_s, state, observable, field_t, "initial", None)

    while True:
        axis, field_t = _axis_and_field(state.position_m, state.last_quantization_axis, coil_config, cfg)
        internal = atom_structure.states[state.internal_state_index]
        if time_s >= duration_s:
            record.termination_reason = "duration"
            break
        if _is_outward_escape(state.position_m, state.velocity_m_per_s, escape_radius_m):
            record.termination_reason = "escaped"
            break
        if internal.is_dark and not cfg.repumper_enabled:
            record.counters.dark_entry_time_s = record.counters.dark_entry_time_s or time_s
            if cfg.dark_state_behavior == DarkStateBehavior.BALLISTIC:
                dt_s = duration_s - time_s
                position, velocity = _propagate_ballistic(
                    state.position_m, state.velocity_m_per_s, dt_s, cfg.include_gravity
                )
                time_s = duration_s
                state = MultilevelAtomState(position, velocity, state.internal_state_index, axis, True)
                dark_observable = MeanObservable(
                    (0.0, 0.0, 0.0), 0.0, tuple(0.0 for _ in optical_beams),
                    local_magnetic_field_t(position, coil_config),
                )
                _append_record(
                    record, time_s, state, dark_observable,
                    dark_observable.magnetic_field_t, "dark_ballistic_end", None,
                )
                record.termination_reason = "duration_dark_ballistic"
            else:
                record.termination_reason = "dark_state"
            break
        if event_count >= max_events:
            record.termination_reason = "max_events"
            break

        channels = outgoing_channels(
            atom_structure, state.internal_state_index, optical_beams,
            state.position_m, state.velocity_m_per_s, float(np.linalg.norm(field_t)), axis, cfg,
        )
        sampled = sample_next_event(channels, rng)
        waiting = sampled.waiting_time_s if sampled else inf
        dt_s = min(waiting, duration_s - time_s)
        position, velocity = _propagate_ballistic(state.position_m, state.velocity_m_per_s, dt_s, cfg.include_gravity)
        time_s += dt_s
        if _is_outward_escape(position, velocity, escape_radius_m):
            state = MultilevelAtomState(position, velocity, state.internal_state_index, axis, state.dark)
            field_at_escape = local_magnetic_field_t(position, coil_config)
            if internal.is_ground:
                escape_observable = ground_state_mean_observable(
                    atom_structure, state.internal_state_index, optical_beams,
                    position, velocity, coil_config, cfg, axis,
                )
            else:
                escape_observable = MeanObservable(
                    (0.0, 0.0, 0.0), 0.0, tuple(0.0 for _ in optical_beams), field_at_escape,
                )
            _append_record(record, time_s, state, escape_observable, field_at_escape, "escaped", None)
            record.termination_reason = "escaped"
            break
        if sampled is None or waiting > dt_s:
            state = MultilevelAtomState(position, velocity, state.internal_state_index, axis, state.dark)
            _append_record(record, time_s, state, observable, field_t, "duration", None)
            continue

        channel: EventChannel = sampled.channel
        velocity_array = np.asarray(velocity)
        source_internal = atom_structure.states[state.internal_state_index]
        if channel.event_type == "absorption":
            beam = optical_beams[channel.beam_index]
            wavelength_m = cfg.repump_wavelength_m if beam.family == "repump" else cfg.wavelength_m
            last_absorption_wavelength_m = wavelength_m
            velocity_array += absorption_velocity_kick(beam.direction, wavelength_m)
            record.counters.absorption_events += 1
            if beam.family == "repump":
                record.counters.repump_absorption_events += 1
                repump_photons_since_f1_entry += 1
                destination_state = atom_structure.states[channel.destination_state_index]
                record.repump_absorptions.append(
                    RepumpAbsorptionRecord(
                        time_s=time_s,
                        beam_index=channel.beam_index,
                        initial_f=source_internal.f,
                        initial_m_f=source_internal.m_f,
                        excited_f=destination_state.f,
                        excited_m_f=destination_state.m_f,
                        position_m=position,
                        velocity_m_per_s=tuple(velocity_array),
                        magnetic_field_t=field_t,
                        detuning_rad_per_s=channel.detuning_rad_per_s or 0.0,
                        polarization_weight=channel.polarization_weight or 0.0,
                        saturation_parameter=channel.saturation_parameter or 0.0,
                        rate_per_s=channel.rate_per_s,
                    )
                )
            if source_internal.is_dark and f1_visit_start_s is not None:
                duration = time_s - f1_visit_start_s
                record.counters.total_f1_time_s += duration
                record.f1_visit_durations_s.append(duration)
                f1_visit_start_s = None
        elif channel.event_type == "stimulated_emission":
            beam = optical_beams[channel.beam_index]
            wavelength_m = cfg.repump_wavelength_m if beam.family == "repump" else cfg.wavelength_m
            velocity_array += stimulated_emission_velocity_kick(beam.direction, wavelength_m)
            record.counters.stimulated_emissions += 1
        else:
            velocity_array += spontaneous_emission_velocity_kick(last_absorption_wavelength_m, rng)
            record.counters.spontaneous_emissions += 1
            record.counters.photons_before_dark += 1
        destination = atom_structure.states[channel.destination_state_index]
        if destination.is_dark and not source_internal.is_dark:
            if record.counters.dark_entry_time_s is None:
                record.counters.dark_entry_time_s = time_s
            record.counters.dark_parent_excited_f = channel.excited_f
            f1_visit_start_s = time_s
            record.counters.f1_visit_count += 1
        if source_internal.is_excited and destination.is_ground and destination.f == 2 and repump_photons_since_f1_entry > 0:
            record.counters.f1_returns_to_f2 += 1
            record.repump_photons_per_f1_return.append(repump_photons_since_f1_entry)
            repump_photons_since_f1_entry = 0
        state = MultilevelAtomState(position, tuple(velocity_array), destination.index, axis, destination.is_dark)
        event_count += 1
        # Replace the generic state label recorded next iteration with event metadata now.
        axis2, field2 = _axis_and_field(state.position_m, axis, coil_config, cfg)
        if destination.is_ground:
            obs2 = ground_state_mean_observable(atom_structure, destination.index, optical_beams, position, tuple(velocity_array), coil_config, cfg, axis2)
        else:
            obs2 = MeanObservable((0.0, 0.0, 0.0), 0.0, tuple(0.0 for _ in optical_beams), field2)
        _append_record(record, time_s, state, obs2, field2, channel.event_type, channel.beam_index)

    if f1_visit_start_s is not None and record.times_s:
        duration = record.times_s[-1] - f1_visit_start_s
        if duration > 0.0:
            record.counters.total_f1_time_s += duration
            record.f1_visit_durations_s.append(duration)
    return record
