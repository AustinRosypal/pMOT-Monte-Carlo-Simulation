"""Physics and numerical checks for the efficient population-rate MOT model."""

from dataclasses import replace

import numpy as np
import pytest

from pmot.mot import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams, default_multilevel_mot_config
from pmot.mot_multilevel.rate_equations import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    assemble_rate_matrix,
    build_beam_stimulated_rate_matrices,
    build_rate_equation_model,
    rate_equation_observable,
    simulate_rate_equation_trajectory,
    steady_state_populations,
)
from pmot.mot_multilevel.rate_diagnostics import hyperfine_manifold_occupation_percent


@pytest.fixture(scope="module")
def rate_apparatus():
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    beams = build_multilevel_mot_beams(config=config)
    return model, beams, default_anti_helmholtz_config(), config


def test_precomputed_dipole_and_decay_arrays_include_full_repumper_graph(rate_apparatus) -> None:
    model, _, _, config = rate_apparatus
    assert model.ground_count == 8
    assert model.excited_count == 16  # includes repumper-accessible F'=0
    assert model.dipole_tensor.shape == (3, 16, 8)
    assert np.allclose(
        np.sum(model.decay_rate_matrix_per_s, axis=0),
        config.natural_linewidth_rad_per_s,
        rtol=1e-13,
    )


def test_rate_matrix_conserves_population_and_steady_state_is_physical(rate_apparatus) -> None:
    model, beams, coil, config = rate_apparatus
    observable = rate_equation_observable(
        model, beams, (0.7e-3, -0.2e-3, 0.4e-3), (1.0, -0.3, 0.2),
        coil, config, store_rate_matrix=True,
    )
    matrix = observable.rate_matrix_per_s
    assert matrix is not None
    assert np.max(np.abs(np.sum(matrix, axis=0))) < 1e-7
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.min(off_diagonal) >= 0.0
    assert np.isclose(np.sum(observable.populations), 1.0)
    assert np.min(observable.populations) >= 0.0
    assert np.max(np.abs(matrix @ observable.populations)) < 1e-6


def test_explicit_rate_matrix_has_symmetric_stimulated_links() -> None:
    stimulated = np.asarray([[2.0, 3.0], [5.0, 7.0]])
    decay = np.asarray([[11.0, 13.0], [17.0, 19.0]])
    matrix = assemble_rate_matrix(stimulated, decay)
    assert np.allclose(matrix[2:, :2], stimulated)
    assert np.allclose(matrix[:2, 2:] - decay, stimulated.T)
    assert np.allclose(np.sum(matrix, axis=0), 0.0)
    populations = steady_state_populations(matrix, 10.0)
    assert np.isclose(np.sum(populations), 1.0)


def test_symmetric_origin_has_zero_mean_force(rate_apparatus) -> None:
    model, beams, coil, config = rate_apparatus
    observable = rate_equation_observable(model, beams, (0, 0, 0), (0, 0, 0), coil, config)
    assert np.linalg.norm(observable.force_n) < 1e-30
    assert observable.total_scattering_rate_per_s > 0.0
    assert observable.diffusion_kg2_m2_per_s3 > 0.0


@pytest.mark.parametrize("component", range(3))
def test_rate_equation_force_is_restoring(rate_apparatus, component: int) -> None:
    model, beams, coil, config = rate_apparatus
    position = np.zeros(3)
    position[component] = 1.0e-3
    force = rate_equation_observable(model, beams, tuple(position), (0, 0, 0), coil, config).force_n
    assert force[component] < 0.0


@pytest.mark.parametrize("component", range(3))
def test_rate_equation_force_damps_velocity(rate_apparatus, component: int) -> None:
    model, beams, coil, config = rate_apparatus
    velocity = np.zeros(3)
    velocity[component] = 1.0
    force = rate_equation_observable(model, beams, (0, 0, 0), tuple(velocity), coil, config).force_n
    assert force[component] < 0.0


def test_fixed_timestep_trajectory_is_reproducible(rate_apparatus) -> None:
    model, beams, coil, config = rate_apparatus
    numerical = RateEquationTrajectoryConfig(time_step_s=5e-6, include_diffusion=True, seed=91)
    initial = RateEquationAtomState((1e-3, 0, 0), (-1.0, 0.1, 0.0))
    first = simulate_rate_equation_trajectory(
        initial, 50e-6, coil, beams=beams, model=model, config=config, trajectory_config=numerical,
    )
    second = simulate_rate_equation_trajectory(
        initial, 50e-6, coil, beams=beams, model=model, config=config, trajectory_config=numerical,
    )
    assert len(first.times_s) == 11
    assert np.allclose(np.diff(first.times_s), 5e-6)
    assert np.allclose(first.positions_m, second.positions_m)
    assert np.allclose(first.velocities_m_per_s, second.velocities_m_per_s)
    assert np.asarray(first.populations).shape == (11, model.state_count)


def test_deterministic_trajectory_refines_with_smaller_timestep(rate_apparatus) -> None:
    model, beams, coil, config = rate_apparatus
    initial = RateEquationAtomState((1e-3, 0, 0), (-1.0, 0.1, 0.0))

    trajectories = []
    for time_step_s in (5e-6, 2.5e-6, 1.25e-6):
        numerical = RateEquationTrajectoryConfig(
            time_step_s=time_step_s,
            include_diffusion=False,
        )
        trajectories.append(
            simulate_rate_equation_trajectory(
                initial,
                100e-6,
                coil,
                beams=beams,
                model=model,
                config=config,
                trajectory_config=numerical,
            )
        )

    coarse_position_error = np.linalg.norm(
        np.asarray(trajectories[0].positions_m[-1])
        - np.asarray(trajectories[2].positions_m[-1])
    )
    fine_position_error = np.linalg.norm(
        np.asarray(trajectories[1].positions_m[-1])
        - np.asarray(trajectories[2].positions_m[-1])
    )
    coarse_velocity_error = np.linalg.norm(
        np.asarray(trajectories[0].velocities_m_per_s[-1])
        - np.asarray(trajectories[2].velocities_m_per_s[-1])
    )
    fine_velocity_error = np.linalg.norm(
        np.asarray(trajectories[1].velocities_m_per_s[-1])
        - np.asarray(trajectories[2].velocities_m_per_s[-1])
    )

    assert fine_position_error < coarse_position_error
    assert fine_velocity_error < coarse_velocity_error


def test_hyperfine_occupation_groups_mf_states_and_sums_to_100_percent(rate_apparatus) -> None:
    model, beams, coil, config = rate_apparatus
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState((1e-3, 0, 0), (-1.0, 0.1, 0.0)),
        25e-6,
        coil,
        beams=beams,
        model=model,
        config=config,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=5e-6,
            include_diffusion=False,
        ),
    )
    labels, percentages = hyperfine_manifold_occupation_percent(record, model)
    assert labels == [
        r"$|g\rangle\ F=1$",
        r"$|g\rangle\ F=2$",
        r"$|e\rangle\ F=0$",
        r"$|e\rangle\ F=1$",
        r"$|e\rangle\ F=2$",
        r"$|e\rangle\ F=3$",
    ]
    assert np.all(percentages >= 0.0)
    assert np.sum(percentages) == pytest.approx(100.0)
