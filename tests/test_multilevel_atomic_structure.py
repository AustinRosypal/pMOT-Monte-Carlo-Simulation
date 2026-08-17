"""Stage-A validation tests from MULTILEVEL_MOT.md."""

from __future__ import annotations

import numpy as np

from pmot.mot_multilevel import build_atomic_structure
from pmot.mot_multilevel import normalized_dipole_strength
from pmot.mot_multilevel import spontaneous_channels
from pmot.mot_multilevel.configuration import default_multilevel_mot_config


def test_state_count() -> None:
    structure = build_atomic_structure()
    ground = [structure.states[index] for index in structure.ground_state_indices]
    excited = [structure.states[index] for index in structure.excited_state_indices]
    assert len([state for state in ground if state.f == 1]) == 3
    assert len([state for state in ground if state.f == 2]) == 5
    assert len([state for state in excited if state.f == 1]) == 3
    assert len([state for state in excited if state.f == 2]) == 5
    assert len([state for state in excited if state.f == 3]) == 7
    assert len(ground) == 8
    assert len(excited) == 15
    assert len(structure.states) == 23


def test_selection_rules_and_required_channels() -> None:
    structure = build_atomic_structure()
    for transition in structure.absorption_transitions:
        assert transition.excited_f - transition.ground_f in (-1, 0, 1)
        assert transition.excited_m_f == transition.ground_m_f + transition.q
        assert transition.q in (-1, 0, 1)
        assert abs(transition.ground_m_f) <= transition.ground_f
        assert abs(transition.excited_m_f) <= transition.excited_f
    assert any(
        (t.ground_f, t.ground_m_f, t.excited_f, t.excited_m_f, t.q) == (2, 2, 3, 3, 1)
        for t in structure.absorption_transitions
    )
    assert not any(
        (t.ground_f, t.ground_m_f, t.excited_f, t.excited_m_f) == (2, 2, 3, -3)
        for t in structure.absorption_transitions
    )
    assert not any(
        structure.states[channel.excited_state_index].f == 3
        and structure.states[channel.ground_state_index].f == 1
        for channel in structure.decay_channels
    )


def test_clebsch_gordan_normalization_and_symmetry() -> None:
    structure = build_atomic_structure()
    stretched = normalized_dipole_strength(2, 2, 3, 3)
    assert np.isclose(stretched, 1.0, atol=1.0e-12)
    assert all(-1.0e-15 <= transition.c_squared <= 1.0 + 1.0e-15 for transition in structure.absorption_transitions)
    for transition in structure.absorption_transitions:
        mirror = normalized_dipole_strength(
            transition.ground_f,
            -transition.ground_m_f,
            transition.excited_f,
            -transition.excited_m_f,
        )
        assert np.isclose(transition.c_squared, mirror, atol=1.0e-12)


def test_hyperfine_branching_ratios() -> None:
    structure = build_atomic_structure()
    expected = {1: {1: 5.0 / 6.0, 2: 1.0 / 6.0}, 2: {1: 0.5, 2: 0.5}, 3: {1: 0.0, 2: 1.0}}
    for excited_index in structure.excited_state_indices:
        excited = structure.states[excited_index]
        totals = {1: 0.0, 2: 0.0}
        for channel in structure.decay_by_excited[excited_index]:
            totals[structure.states[channel.ground_state_index].f] += channel.branch_probability
        assert np.isclose(sum(totals.values()), 1.0, atol=1.0e-12)
        assert np.isclose(totals[1], expected[excited.f][1], atol=1.0e-12)
        assert np.isclose(totals[2], expected[excited.f][2], atol=1.0e-12)


def test_spontaneous_rates_sum_to_gamma() -> None:
    structure = build_atomic_structure()
    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    for excited_index in structure.excited_state_indices:
        channels = spontaneous_channels(structure, excited_index, gamma)
        assert np.isclose(sum(channel.rate_per_s for channel in channels), gamma, rtol=1.0e-12)
