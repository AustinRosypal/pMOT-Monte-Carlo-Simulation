"""Temperature-definition checks for the multilevel ensemble diagnostic."""

import numpy as np
import pytest

from pmot.configuration import HBAR_J_S, RB87_MASS_KG
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.temperature import (
    BOLTZMANN_CONSTANT_J_PER_K,
    doppler_temperature_k,
    effective_saturation_parameter,
    temperature_components_k,
)


def test_multilevel_doppler_formula_reduces_to_146_microkelvin_at_ideal_point() -> None:
    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    doppler = doppler_temperature_k(gamma, -0.5 * gamma, 0.0)

    assert np.isclose(doppler, 145.7e-6, rtol=2e-3)


def test_multilevel_doppler_temperature_uses_detuning_and_effective_saturation() -> None:
    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    detuning = -2.5 * gamma
    effective_saturation = 0.75
    expected = (
        -HBAR_J_S
        * gamma**2
        / (8.0 * BOLTZMANN_CONSTANT_J_PER_K * detuning)
        * (1.0 + effective_saturation + (2.0 * detuning / gamma) ** 2)
    )

    assert doppler_temperature_k(gamma, detuning, effective_saturation) == pytest.approx(
        expected
    )
    assert effective_saturation_parameter(20.0, detuning, gamma) == pytest.approx(
        20.0 / 26.0
    )


@pytest.mark.parametrize(
    ("gamma", "detuning", "effective_saturation", "message"),
    [
        (0.0, -1.0, 0.0, "linewidth"),
        (1.0, 0.0, 0.0, "negative"),
        (1.0, 1.0, 0.0, "negative"),
        (1.0, -1.0, -0.1, "non-negative"),
        (1.0, -1.0, np.nan, "finite"),
    ],
)
def test_multilevel_doppler_temperature_rejects_unphysical_inputs(
    gamma: float,
    detuning: float,
    effective_saturation: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        doppler_temperature_k(gamma, detuning, effective_saturation)


def test_temperature_subtracts_center_of_mass_velocity() -> None:
    target = 200.0e-6
    sigma = np.sqrt(BOLTZMANN_CONSTANT_J_PER_K * target / RB87_MASS_KG)
    velocities = sigma * np.asarray([
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]) + np.asarray([10.0, -4.0, 2.0])
    assert np.allclose(temperature_components_k(velocities), target)
