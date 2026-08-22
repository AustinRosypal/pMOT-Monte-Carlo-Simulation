"""Temperature-definition checks for the multilevel ensemble diagnostic."""

import numpy as np

from pmot.configuration import RB87_MASS_KG
from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.temperature import (
    BOLTZMANN_CONSTANT_J_PER_K,
    doppler_temperature_k,
    temperature_components_k,
)


def test_rb87_doppler_temperature_is_about_146_microkelvin() -> None:
    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    assert np.isclose(doppler_temperature_k(gamma), 145.7e-6, rtol=2e-3)


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
