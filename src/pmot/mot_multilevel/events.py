"""Continuous-time Gillespie event construction and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from ..fields import MOTBeam
from .atomic_structure import AtomicStructure
from .configuration import MultilevelMOTConfig
from .coupling import ground_laser_channels
from .coupling import stimulated_emission_channels
from .polarization import Vec3


@dataclass(frozen=True, slots=True)
class EventChannel:
    """One possible outgoing Markov-process event."""

    event_type: str
    destination_state_index: int
    rate_per_s: float
    beam_index: int | None = None
    excited_f: int | None = None
    q: int | None = None
    detuning_rad_per_s: float | None = None
    polarization_weight: float | None = None
    saturation_parameter: float | None = None


@dataclass(frozen=True, slots=True)
class SampledEvent:
    """A selected event and its waiting time."""

    waiting_time_s: float
    channel: EventChannel


def spontaneous_channels(
    structure: AtomicStructure,
    excited_state_index: int,
    gamma_per_s: float,
) -> list[EventChannel]:
    """Return branches whose rates sum exactly to Gamma."""

    return [
        EventChannel(
            "spontaneous_emission",
            branch.ground_state_index,
            gamma_per_s * branch.branch_probability,
            excited_f=structure.states[excited_state_index].f,
            q=branch.q,
        )
        for branch in structure.decay_by_excited[excited_state_index]
    ]


def outgoing_channels(
    structure: AtomicStructure,
    state_index: int,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig,
) -> list[EventChannel]:
    """Gather direct adjacency-list events for the current internal state."""

    state = structure.states[state_index]
    if state.is_dark and not config.repumper_enabled:
        return []
    if state.is_ground:
        return [
            EventChannel(
                channel.event_type,
                channel.destination_state_index,
                channel.rate_per_s,
                beam_index=channel.beam_index,
                excited_f=channel.transition.excited_f,
                q=channel.transition.q,
                detuning_rad_per_s=channel.detuning_rad_per_s,
                polarization_weight=channel.polarization_weight,
                saturation_parameter=channel.saturation_parameter,
            )
            for channel in ground_laser_channels(
                structure,
                state_index,
                beams,
                position_m,
                velocity_m_per_s,
                field_magnitude_t,
                quantization_axis_vector,
                config,
            )
        ]

    channels = spontaneous_channels(structure, state_index, config.natural_linewidth_rad_per_s)
    channels.extend(
        EventChannel(
            channel.event_type,
            channel.destination_state_index,
            channel.rate_per_s,
            beam_index=channel.beam_index,
            excited_f=channel.transition.excited_f,
            q=channel.transition.q,
            detuning_rad_per_s=channel.detuning_rad_per_s,
            polarization_weight=channel.polarization_weight,
            saturation_parameter=channel.saturation_parameter,
        )
        for channel in stimulated_emission_channels(
            structure,
            state_index,
            beams,
            position_m,
            velocity_m_per_s,
            field_magnitude_t,
            quantization_axis_vector,
            config,
        )
    )
    return channels


def sample_waiting_time_s(total_rate_per_s: float, rng: np.random.Generator) -> float:
    """Sample an exponential waiting time for a constant total rate."""

    if total_rate_per_s <= 0.0:
        return float("inf")
    uniform = max(float(rng.random()), np.finfo(float).tiny)
    return -log(uniform) / total_rate_per_s


def sample_channel(channels: list[EventChannel], rng: np.random.Generator) -> EventChannel:
    """Select a channel with probability proportional to its rate."""

    if not channels:
        raise ValueError("cannot sample from an empty channel list")
    rates = np.asarray([channel.rate_per_s for channel in channels], dtype=float)
    if np.any(rates < 0.0) or float(np.sum(rates)) <= 0.0:
        raise ValueError("channel rates must be nonnegative with positive total")
    cumulative = np.cumsum(rates)
    target = float(rng.uniform(0.0, cumulative[-1]))
    index = min(int(np.searchsorted(cumulative, target, side="right")), len(channels) - 1)
    return channels[index]


def sample_next_event(channels: list[EventChannel], rng: np.random.Generator) -> SampledEvent | None:
    """Sample both Gillespie waiting time and event identity."""

    total_rate = sum(channel.rate_per_s for channel in channels)
    if total_rate <= 0.0:
        return None
    return SampledEvent(sample_waiting_time_s(total_rate, rng), sample_channel(channels, rng))
