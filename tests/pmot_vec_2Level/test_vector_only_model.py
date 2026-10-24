"""Checks for the vector-only effective two-level pMOT diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.pmot_vec_2Level import build_pmot_vec_2level_mot_light
from pmot.pmot_vec_2Level import build_pmot_vec_2level_trapping_beams
from pmot.pmot_vec_2Level import evaluate_pmot_vec_2level_local_observable
from pmot.pmot_vec_2Level import evaluate_vector_only_ac_stark_shift
from pmot.pmot_vec_2Level import load_vector_only_polarizability_table


def test_vector_only_table_preserves_vector_and_zeros_other_ranks() -> None:
    table = load_vector_only_polarizability_table()
    assert np.all(table.scalar_si == 0.0)
    assert np.all(table.tensor_si == 0.0)
    assert np.any(table.vector_si != 0.0)
    assert not table.scalar_si.flags.writeable
    assert not table.tensor_si.flags.writeable


def test_vector_only_stark_sample_has_no_scalar_or_tensor_energy() -> None:
    sample = evaluate_vector_only_ac_stark_shift(
        build_pmot_vec_2level_trapping_beams(),
        (0.3e-3, -0.2e-3, 0.1e-3),
        (0.0, 0.0, 0.0),
        path_powers_w=(0.010, 0.011, 0.012),
    )
    assert sample.total_scalar_energy_j == 0.0
    assert sample.total_tensor_energy_j == 0.0
    assert sample.total_vector_energy_j != 0.0
    assert sample.transition_shifts_hz_by_q[1] == 0.0
    assert sample.transition_shifts_hz_by_q[0] == pytest.approx(
        -sample.transition_shifts_hz_by_q[2]
    )


def test_vector_only_bundle_includes_passive_resonant_repump_geometry() -> None:
    bundle = build_pmot_vec_2level_mot_light(cooling_power_w_per_beam=1.0571e-3)
    assert len(bundle.cooling_beams) == len(bundle.repump_beams) == 6
    assert all(beam.intensity_beam.power_w == pytest.approx(1.0571e-3) for beam in bundle.cooling_beams)
    assert all(beam.power_w == pytest.approx(0.1e-3) for beam in bundle.repump_beams)
    assert all(beam.detuning_hz == 0.0 for beam in bundle.repump_beams)


def test_vector_only_local_force_uses_zero_common_stark_shift() -> None:
    light = build_pmot_vec_2level_mot_light(cooling_power_w_per_beam=1.0571e-3)
    trapping = build_pmot_vec_2level_trapping_beams()
    observable = evaluate_pmot_vec_2level_local_observable(
        light.cooling_beams,
        trapping,
        (0.2e-3, 0.1e-3, -0.1e-3),
        (0.0, 0.0, 0.0),
        path_powers_w=(0.010, 0.010, 0.020),
    )
    shifts = np.asarray(observable.stark.transition_shifts_hz_by_q)
    assert shifts[1] == 0.0
    assert shifts[0] == pytest.approx(-shifts[2])
