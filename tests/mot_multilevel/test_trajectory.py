from __future__ import annotations

import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    simulate_rate_equation_trajectory,
)


def test_default_five_microsecond_rk4_step_records_normalized_populations() -> None:
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(
            position_m=(1.0e-3, 0.0, 0.0),
            velocity_m_per_s=(0.0, 0.0, 0.0),
        ),
        5.0e-6,
        default_anti_helmholtz_config(),
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=5.0e-6,
            store_rate_matrices=True,
            store_beam_transition_quantities=True,
        ),
    )
    assert record.times_s == [0.0, 5.0e-6]
    assert len(record.positions_m) == 2
    assert len(record.rate_matrices_per_s) == 2
    assert len(record.beam_transition_quantities) == 2
    assert record.beam_transition_quantities[0].stimulated_coefficients_per_s.shape == (
        12,
        16,
        8,
    )
    for populations, matrix in zip(record.populations, record.rate_matrices_per_s):
        np.testing.assert_allclose(np.sum(populations), 1.0, rtol=0.0, atol=1.0e-13)
        np.testing.assert_allclose(np.sum(matrix, axis=0), 0.0, rtol=0.0, atol=1.0e-7)
