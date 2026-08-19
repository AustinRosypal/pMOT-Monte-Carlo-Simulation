"""Stage-C Gillespie, lifetime, dark-state, and recoil primitives."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pmot.mot_multilevel import EventChannel
from pmot.mot_multilevel import absorption_velocity_kick
from pmot.mot_multilevel import build_atomic_structure
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import outgoing_channels
from pmot.mot_multilevel import recoil_speed_m_per_s
from pmot.mot_multilevel import sample_channel
from pmot.mot_multilevel import sample_waiting_time_s
from pmot.mot_multilevel import spontaneous_channels
from pmot.mot_multilevel import spontaneous_emission_velocity_kick
from pmot.mot_multilevel.configuration import default_multilevel_mot_config


def test_gillespie_waiting_time_statistics() -> None:
    rng = np.random.default_rng(12345)
    rate = 2.5e6
    samples = np.asarray([sample_waiting_time_s(rate, rng) for _ in range(100_000)])
    assert abs(float(np.mean(samples)) - 1.0 / rate) / (1.0 / rate) < 0.01


def test_gillespie_channel_statistics() -> None:
    rng = np.random.default_rng(12345)
    channels = [EventChannel("a", 0, 1.0), EventChannel("b", 1, 3.0)]
    count_b = sum(sample_channel(channels, rng).event_type == "b" for _ in range(100_000))
    assert abs(count_b / 100_000 - 0.75) < 0.01


def test_excited_state_lifetime_statistics() -> None:
    structure = build_atomic_structure()
    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    excited_index = structure.state_index("excited", 3, 0)
    total_rate = sum(channel.rate_per_s for channel in spontaneous_channels(structure, excited_index, gamma))
    rng = np.random.default_rng(54)
    samples = np.asarray([sample_waiting_time_s(total_rate, rng) for _ in range(100_000)])
    assert abs(float(np.mean(samples)) - 1.0 / gamma) / (1.0 / gamma) < 0.01


def test_dark_ground_states_have_no_outgoing_channels_without_repumper() -> None:
    structure = build_atomic_structure()
    config = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=replace(config, repumper_enabled=True))
    for m_f in (-1, 0, 1):
        state_index = structure.state_index("ground", 1, m_f)
        assert outgoing_channels(structure, state_index, beams, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config) == []


def test_f1_ground_states_have_repump_channels_when_enabled() -> None:
    structure = build_atomic_structure()
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    beams = build_multilevel_mot_beams(config=config)
    for m_f in (-1, 0, 1):
        state_index = structure.state_index("ground", 1, m_f)
        channels = outgoing_channels(structure, state_index, beams, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 1.0), config)
        assert channels
        assert all(beams[channel.beam_index].family == "repump" for channel in channels)


def test_recoil_kick_magnitudes() -> None:
    wavelength = 780.0e-9
    expected = recoil_speed_m_per_s(wavelength)
    absorption = absorption_velocity_kick((1.0, 0.0, 0.0), wavelength)
    emission = spontaneous_emission_velocity_kick(wavelength, np.random.default_rng(8))
    assert np.isclose(np.linalg.norm(absorption), expected, rtol=1.0e-12)
    assert np.isclose(np.linalg.norm(emission), expected, rtol=1.0e-12)
