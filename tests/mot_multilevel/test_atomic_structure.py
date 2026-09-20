from __future__ import annotations

import numpy as np
import pytest

from pmot.mot_multilevel.atomic_structure import build_atomic_structure
from pmot.mot_multilevel.rate_equations import build_rate_equation_model


def test_arc_precomputes_the_complete_24_state_d2_graph_once() -> None:
    structure = build_atomic_structure()
    assert build_atomic_structure() is structure
    assert len(structure.states) == 24
    assert len(structure.ground_state_indices) == 8
    assert len(structure.excited_state_indices) == 16
    assert len(structure.transitions) == 54
    assert structure.arc_version == "3.8.0"


def test_arc_cycling_dipole_and_selection_rule() -> None:
    structure = build_atomic_structure()
    cycling = next(
        item
        for item in structure.transitions
        if (
            item.ground_f,
            item.ground_m_f,
            item.excited_f,
            item.excited_m_f,
        )
        == (2, 2, 3, 3)
    )
    assert cycling.q == 1
    assert cycling.dipole_matrix_element_ea0 == pytest.approx(2.98915, abs=1.0e-12)
    assert not any(
        item.ground_f == 2
        and item.ground_m_f == 2
        and item.excited_f == 3
        and item.excited_m_f == 0
        for item in structure.transitions
    )


def test_pairwise_arc_dipoles_sum_to_each_excited_state_decay_constant() -> None:
    model = build_rate_equation_model()
    assert build_rate_equation_model() is model
    np.testing.assert_allclose(
        np.sum(model.spontaneous_decay_matrix_per_s, axis=0),
        model.excited_decay_rates_per_s,
        rtol=0.0,
        atol=1.0e-9,
    )
    linewidths_mhz = model.excited_decay_rates_per_s / (2.0 * np.pi * 1.0e6)
    assert np.all(linewidths_mhz > 6.065)
    assert np.all(linewidths_mhz < 6.067)
    assert np.all(model.transition_spontaneous_decay_rate_per_s > 0.0)


def test_reference_frequencies_are_the_intended_hyperfine_lines() -> None:
    structure = build_atomic_structure()
    cooling = structure.cooling_reference_angular_frequency_rad_per_s
    repump = structure.repump_reference_angular_frequency_rad_per_s
    assert cooling > 0.0
    assert repump > cooling
    assert (repump - cooling) / (2.0 * np.pi) == pytest.approx(6.568e9, rel=2.0e-3)
