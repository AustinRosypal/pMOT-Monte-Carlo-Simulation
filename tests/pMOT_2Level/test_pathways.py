"""Tests for the two fixed-wavelength pMOT_2Level pathways."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.mot_strange.trapping_beams import TrappingLaserConfig
from pmot.pMOT_2Level import PMOT_2LEVEL_PATHWAYS
from pmot.pMOT_2Level import SCALAR_TENSOR_CANCELLATION_PATHWAY
from pmot.pMOT_2Level import SCALAR_ZERO_PATHWAY
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import get_pmot_2level_pathway
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level.scalar_tensor_cancellation import (
    build_trapping_beams as build_scalar_tensor_cancellation_beams,
)
from pmot.pMOT_2Level.scalar_zero import (
    build_trapping_beams as build_scalar_zero_beams,
)


def _polarizabilities_at(wavelength_nm: float) -> tuple[float, float, float]:
    table = load_pmot_2level_polarizability_table()
    return tuple(
        float(np.interp(wavelength_nm, table.wavelengths_nm, values))
        for values in (table.scalar_si, table.vector_si, table.tensor_si)
    )


def test_registry_exposes_only_the_two_named_wavelength_paths() -> None:
    assert tuple(PMOT_2LEVEL_PATHWAYS) == (
        "scalar_tensor_cancellation",
        "scalar_zero",
    )
    assert get_pmot_2level_pathway("scalar_tensor_cancellation") is (
        SCALAR_TENSOR_CANCELLATION_PATHWAY
    )
    assert get_pmot_2level_pathway("scalar_zero") is SCALAR_ZERO_PATHWAY
    with pytest.raises(ValueError, match="unknown pMOT_2Level pathway"):
        get_pmot_2level_pathway("custom")


def test_pathway_builders_change_only_the_fixed_wavelength() -> None:
    scalar_tensor_beams = build_scalar_tensor_cancellation_beams()
    scalar_zero_beams = build_scalar_zero_beams()

    assert len(scalar_tensor_beams) == len(scalar_zero_beams) == 6
    for scalar_tensor_beam, scalar_zero_beam in zip(
        scalar_tensor_beams,
        scalar_zero_beams,
        strict=True,
    ):
        assert scalar_tensor_beam.wavelength_m == pytest.approx(1529.268881e-9)
        assert scalar_zero_beam.wavelength_m == pytest.approx(
            1529.273195181438e-9
        )
        assert scalar_tensor_beam.label == scalar_zero_beam.label
        assert scalar_tensor_beam.axis_name == scalar_zero_beam.axis_name
        assert scalar_tensor_beam.propagation_sense == (
            scalar_zero_beam.propagation_sense
        )
        assert scalar_tensor_beam.helicity == scalar_zero_beam.helicity
        np.testing.assert_allclose(
            scalar_tensor_beam.direction,
            scalar_zero_beam.direction,
        )
        np.testing.assert_allclose(
            scalar_tensor_beam.waist_position_m,
            scalar_zero_beam.waist_position_m,
        )


def test_pathway_config_accepts_geometry_but_rejects_wavelength_override() -> None:
    base = TrappingLaserConfig(focus_offset_m=9.0e-3)
    config = SCALAR_ZERO_PATHWAY.build_laser_config(
        base,
        retro_power_fraction=0.8,
    )

    assert config.wavelength_m == pytest.approx(1529.273195181438e-9)
    assert config.focus_offset_m == pytest.approx(9.0e-3)
    assert config.retro_power_fraction == pytest.approx(0.8)
    with pytest.raises(ValueError, match="wavelength_m is fixed"):
        SCALAR_ZERO_PATHWAY.build_laser_config(wavelength_m=1530.0e-9)


def test_tabulated_conditions_distinguish_the_two_pathways() -> None:
    scalar_a, vector_a, tensor_a = _polarizabilities_at(
        SCALAR_TENSOR_CANCELLATION_PATHWAY.trapping_wavelength_nm
    )
    scalar_b, vector_b, tensor_b = _polarizabilities_at(
        SCALAR_ZERO_PATHWAY.trapping_wavelength_nm
    )

    assert abs(scalar_a - 0.5 * tensor_a) < 1.0e-5 * abs(vector_a)
    assert abs(vector_a) > 0.0
    assert abs(scalar_b) < 1.0e-10 * abs(vector_b)
    assert abs(vector_b) > 0.0
    assert abs(tensor_b) > 0.0


def test_balanced_origin_exposes_the_geometric_reason_for_the_split() -> None:
    table = load_pmot_2level_polarizability_table()
    powers = (0.01, 0.01, 0.01)
    scalar_tensor = evaluate_differential_ac_stark_shift(
        build_scalar_tensor_cancellation_beams(),
        position_m=(0.0, 0.0, 0.0),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        path_powers_w=powers,
        polarizability_table=table,
    )
    scalar_zero = evaluate_differential_ac_stark_shift(
        build_scalar_zero_beams(),
        position_m=(0.0, 0.0, 0.0),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        path_powers_w=powers,
        polarizability_table=table,
    )

    assert abs(scalar_tensor.total_tensor_energy_j) < 1.0e-12 * sum(
        abs(value) for value in scalar_tensor.component_tensor_energies_j
    )
    assert abs(scalar_tensor.total_scalar_energy_j) > 0.0
    assert abs(scalar_zero.total_scalar_energy_j) < 1.0e-10 * sum(
        abs(value) for value in scalar_zero.component_tensor_energies_j
    )
    assert abs(scalar_zero.total_tensor_energy_j) < 1.0e-12 * sum(
        abs(value) for value in scalar_zero.component_tensor_energies_j
    )

