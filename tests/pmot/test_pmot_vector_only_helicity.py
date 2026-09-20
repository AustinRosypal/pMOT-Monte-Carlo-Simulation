"""Checks for the distinct ideal-magic vector-only pMOT audit."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.configuration import HBAR_J_S
from pmot.pmot.helicity_sweep import CURRENT_COMBINED_CODE
from pmot.pmot.helicity_sweep import POSITION_RESTORING_COMBINED_CODE
from pmot.pmot.helicity_sweep import path_helicity_codes
from pmot.pmot.vector_only_helicity_study import bare_780_jacobians
from pmot.pmot.vector_only_helicity_study import build_helicity_sweep_context
from pmot.pmot.vector_only_helicity_study import evaluate_vector_only_configuration
from pmot.pmot.vector_only_helicity_study import vector_only_pmot_observable


@pytest.fixture(scope="module")
def vector_only_context():
    return build_helicity_sweep_context(target_gradient_g_per_cm=20.0)


def test_authoritative_780_beam_helicities_remain_fixed(vector_only_context) -> None:
    expected = {
        "horizontal_x": "sigma+",
        "horizontal_y": "sigma+",
        "vertical_z": "sigma-",
    }
    assert len(vector_only_context.cooling_repump_beams) == 12
    for beam in vector_only_context.cooling_repump_beams:
        assert beam.circular_polarization == expected[beam.axis_name]


def test_vector_only_observable_applies_no_scalar_or_tensor_shift(
    vector_only_context,
) -> None:
    observable = vector_only_pmot_observable(
        vector_only_context,
        CURRENT_COMBINED_CODE,
        (0.25e-3, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    expected = observable.stark_diagnostic.vector_transition_energy_j / HBAR_J_S
    np.testing.assert_allclose(
        observable.applied_transition_shift_rad_per_s,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    full_shift = observable.stark_diagnostic.transition_angular_frequency_shift_rad_per_s
    assert not np.allclose(observable.applied_transition_shift_rad_per_s, full_shift)


def test_bare_780_response_is_damping_without_static_position_slope(
    vector_only_context,
) -> None:
    position, velocity = bare_780_jacobians(vector_only_context)
    np.testing.assert_allclose(position, 0.0, atol=1.0e-30)
    assert np.all(np.real(np.linalg.eigvals(velocity)) < 0.0)


def test_current_is_unique_restoring_damped_orientation_and_reverse_is_anti(
    vector_only_context,
) -> None:
    centered = {
        path_code + path_code: evaluate_vector_only_configuration(
            vector_only_context,
            path_code + path_code,
        )
        for path_code in path_helicity_codes()
    }
    assert [
        code
        for code, record in centered.items()
        if record["combined_classification"] == "restoring_and_damping"
    ] == [CURRENT_COMBINED_CODE]
    current = centered[CURRENT_COMBINED_CODE]
    reversed_configuration = centered[POSITION_RESTORING_COMBINED_CODE]
    assert current["combined_classification"] == "restoring_and_damping"
    assert reversed_configuration["combined_classification"] == (
        "anti_restoring_but_damping"
    )
    np.testing.assert_allclose(
        np.diag(current["position_jacobian_n_per_m"]),
        -np.diag(reversed_configuration["position_jacobian_n_per_m"]),
        rtol=2.0e-3,
        atol=0.0,
    )
    assert np.all(np.diag(current["velocity_jacobian_n_s_per_m"]) < 0.0)
    assert np.all(
        np.diag(reversed_configuration["velocity_jacobian_n_s_per_m"]) < 0.0
    )


def test_biased_configuration_is_recorded_but_not_classified(
    vector_only_context,
) -> None:
    record = evaluate_vector_only_configuration(vector_only_context, "++--+-")
    assert record["mismatched_path_count"] == 1
    assert not record["centered_origin_equilibrium"]
    assert record["position_jacobian_n_per_m"] is None
    assert record["velocity_jacobian_n_s_per_m"] is None
    assert record["combined_classification"] == "not_classified_origin_biased"
