"""Focused checks for the exhaustive provisional pMOT helicity audit."""

from __future__ import annotations

import numpy as np
import pytest

from pmot.pmot.helicity_sweep import CURRENT_COMBINED_CODE
from pmot.pmot.helicity_sweep import POSITION_RESTORING_COMBINED_CODE
from pmot.pmot.helicity_sweep import all_helicity_codes
from pmot.pmot.helicity_sweep import build_helicity_sweep_context
from pmot.pmot.helicity_sweep import classify_position_jacobian
from pmot.pmot.helicity_sweep import classify_velocity_jacobian
from pmot.pmot.helicity_sweep import decode_helicity_code
from pmot.pmot.helicity_sweep import evaluate_helicity_configuration


def test_sigma_only_code_space_is_complete_and_unambiguous() -> None:
    codes = all_helicity_codes()
    assert len(codes) == 64
    assert len(set(codes)) == 64
    assert sum(code[:3] == code[3:] for code in codes) == 8
    assert decode_helicity_code("++---+") == (
        ("sigma+", "sigma+", "sigma-"),
        ("sigma-", "sigma-", "sigma+"),
    )
    with pytest.raises(ValueError, match="six"):
        decode_helicity_code("++-")
    with pytest.raises(ValueError, match="six"):
        decode_helicity_code("++-pi+")


def test_jacobian_classifiers_use_force_slope_signs() -> None:
    assert classify_position_jacobian(-np.eye(3)) == "restoring"
    assert classify_position_jacobian(np.eye(3)) == "anti_restoring"
    assert classify_position_jacobian(np.diag((-1.0, 1.0, -1.0))) == "saddle"
    assert classify_velocity_jacobian(-np.eye(3)) == "damping"
    assert classify_velocity_jacobian(np.eye(3)) == "anti_damping"
    assert classify_velocity_jacobian(np.diag((-1.0, 1.0, -1.0))) == "mixed"


@pytest.fixture(scope="module")
def sweep_context():
    return build_helicity_sweep_context(target_gradient_g_per_cm=20.0)


def test_current_and_reversed_centered_configurations_have_expected_local_signs(
    sweep_context,
) -> None:
    current = evaluate_helicity_configuration(sweep_context, CURRENT_COMBINED_CODE)
    reversed_configuration = evaluate_helicity_configuration(
        sweep_context,
        POSITION_RESTORING_COMBINED_CODE,
    )
    assert current["centered_origin_equilibrium"]
    assert reversed_configuration["centered_origin_equilibrium"]
    assert current["position_jacobian_signature"] == "anti_restoring"
    assert reversed_configuration["position_jacobian_signature"] == "restoring"
    assert current["velocity_classification"] == "anti_damping"
    assert reversed_configuration["velocity_classification"] == "anti_damping"
    np.testing.assert_allclose(
        np.diag(current["position_jacobian_n_per_m"]),
        -np.diag(reversed_configuration["position_jacobian_n_per_m"]),
        rtol=2.0e-3,
        atol=0.0,
    )


def test_one_mismatched_path_is_labeled_as_biased_not_equilibrium(
    sweep_context,
) -> None:
    record = evaluate_helicity_configuration(sweep_context, "++--+-")
    assert record["mismatched_path_count"] == 1
    assert not record["pair_centered"]
    assert not record["origin_effective_field_zero"]
    assert not record["centered_origin_equilibrium"]
    assert record["origin_position_classification"] == (
        "not_a_centered_origin_equilibrium"
    )
    assert record["dynamic_linearization_classification"] == (
        "not_applicable_origin_not_equilibrium"
    )
