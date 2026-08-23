"""Focused checks for the 10x25 multilevel temperature-versus-detuning study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pmot.mot_multilevel.temperature_sweep as sweep
from pmot.configuration import RB87_MASS_KG
from pmot.mot_multilevel.temperature import BOLTZMANN_CONSTANT_J_PER_K
from pmot.mot_multilevel.temperature_sweep import (
    DETUNING_N_VALUES,
    PHYSICAL_MODEL_STATEMENT,
    REQUIRED_ATOMS_PER_ENSEMBLE,
    REQUIRED_ENSEMBLE_REALIZATION_COUNT,
    REQUIRED_TRAJECTORY_COUNT,
    TemperatureSweepWorkerResult,
    _mean_sem_t_interval,
    _resume_signature,
    _summarize_temperature_point,
    _summarize_temperature_realization,
    build_argument_parser,
    build_temperature_sweep_configuration,
    detuning_n_grid,
    ensemble_temperature_metrics,
    generate_common_initial_ensembles,
    physical_model_markdown,
    plot_temperature_vs_detuning,
    recoil_seed,
    run_temperature_detuning_sweep,
    verify_only_cooling_detuning_changes,
)


def test_detuning_grid_has_25_established_points_and_extends_to_minus_ten() -> None:
    expected = np.asarray(
        [
            -10.0,
            -9.0,
            -8.0,
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            -2.5,
            -2.25,
            -2.0,
            -1.75,
            -1.5,
            -1.25,
            -1.0,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.25,
            -0.2,
            -0.15,
            -0.1,
        ]
    )
    actual = detuning_n_grid()

    assert np.array_equal(actual, expected)
    assert len(actual) == 25
    assert np.count_nonzero(np.isclose(actual, -0.5)) == 1
    actual[0] = 99.0
    assert DETUNING_N_VALUES[0] == -10.0


def test_detuning_configuration_is_24_state_repumper_model_and_audit_passes() -> None:
    config, apparatus, beams = build_temperature_sweep_configuration(-0.5)
    expected_rad_per_s = -0.5 * config.natural_linewidth_rad_per_s

    assert config.repumper_enabled
    assert config.cooling_detuning_rad_per_s == expected_rad_per_s
    assert apparatus.cooling.detuning_hz == expected_rad_per_s / (2.0 * np.pi)
    cooling_beams = [beam for beam in beams if beam.family == "cooling"]
    repump_beams = [beam for beam in beams if beam.family == "repump"]
    assert len(cooling_beams) == 6
    assert len(repump_beams) == 6
    assert all(
        beam.detuning_hz == expected_rad_per_s / (2.0 * np.pi)
        for beam in cooling_beams
    )

    audit = verify_only_cooling_detuning_changes((-10.0, -2.5, -0.5, -0.1))
    assert audit["passed"]
    assert audit["point_count"] == 4
    assert len(audit["fixed_configuration_sha256"]) == 64
    assert audit["allowed_varied_fields"] == [
        "multilevel_config.cooling_detuning_rad_per_s",
        "apparatus_config.cooling.detuning_hz",
        "cooling_beams[*].detuning_hz",
    ]


def test_common_clouds_and_recoil_streams_are_random_reproducible_and_independent() -> None:
    first = generate_common_initial_ensembles(10, 25, seed=1234)
    second = generate_common_initial_ensembles(10, 25, seed=1234)
    different = generate_common_initial_ensembles(10, 25, seed=1235)

    assert first[0].shape == (10, 25, 3)
    assert first[1].shape == (10, 25, 3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0][0], first[0][1])
    assert not np.array_equal(first[1], different[1])

    seeds = {
        recoil_seed(1234, point_index, ensemble_index, atom_index)
        for point_index in range(2)
        for ensemble_index in range(10)
        for atom_index in range(25)
    }
    assert len(seeds) == 500
    assert recoil_seed(1234, 1, 9, 24) == recoil_seed(1234, 1, 9, 24)


def test_plateau_metrics_use_unbiased_com_subtracted_variance_and_detect_stationarity() -> None:
    target_k = 250.0e-6
    # Each component of this tetrahedral sample has sample variance 1 after
    # scaling, so the unbiased estimator should recover target_k exactly.
    tetrahedron = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    )
    scale = np.sqrt(
        0.75 * BOLTZMANN_CONSTANT_J_PER_K * target_k / RB87_MASS_KG
    )
    ensemble = scale * tetrahedron
    times = np.linspace(0.0, 10.0e-3, 101)
    history = np.repeat(ensemble[:, None, :], len(times), axis=1)

    metrics = ensemble_temperature_metrics(
        history,
        times,
        plateau_window_s=5.0e-3,
    )

    assert np.allclose(metrics["plateau_temperature_components_k"], target_k)
    assert np.isclose(metrics["plateau_temperature_mean_k"], target_k)
    assert metrics["stationarity_metric"] < 1e-12
    assert abs(metrics["relative_drift"]) < 1e-12
    assert metrics["stationarity_pass"]


def test_plateau_metrics_flag_a_monotonic_temperature_drift() -> None:
    rng = np.random.default_rng(9)
    ensemble = rng.normal(size=(25, 3))
    times = np.linspace(0.0, 10.0e-3, 101)
    amplitude = np.sqrt(np.linspace(1.0, 3.0, len(times)))
    history = ensemble[:, None, :] * amplitude[None, :, None]

    metrics = ensemble_temperature_metrics(
        history,
        times,
        plateau_window_s=5.0e-3,
    )

    assert metrics["relative_drift"] > 0.15
    assert not metrics["stationarity_pass"]


def _stationary_worker_results(atom_count: int = 25) -> tuple[list[TemperatureSweepWorkerResult], np.ndarray]:
    rng = np.random.default_rng(42)
    velocities = rng.normal(scale=0.12, size=(atom_count, 3))
    times = np.linspace(0.0, 10.0e-3, 101)
    results = [
        TemperatureSweepWorkerResult(
            ensemble_index=0,
            atom_index=atom_index,
            complete=True,
            trapped=True,
            termination_reason="duration",
            boundedness_reason="complete_final_plateau_core",
            final_radius_m=0.4e-3,
            maximum_radius_m=0.8e-3,
            plateau_max_radius_m=0.6e-3,
            plateau_mean_radius_m=0.4e-3,
            times_s=times,
            velocities_m_per_s=np.repeat(
                velocities[atom_index][None, :], len(times), axis=0
            ),
        )
        for atom_index in range(atom_count)
    ]
    return results, velocities


def test_realization_temperature_is_calculated_from_25_time_aligned_survivors() -> None:
    results, initial_velocities = _stationary_worker_results()
    row = _summarize_temperature_realization(
        point_index=0,
        detuning_n=-2.5,
        ensemble_index=0,
        results=results,
        initial_velocities_m_per_s=initial_velocities,
        requested_atom_count=25,
        failed_atom_count=0,
        plateau_window_s=5.0e-3,
        minimum_survivor_count=5,
        stationarity_limit=0.15,
    )
    expected_components = (
        RB87_MASS_KG
        * np.var(initial_velocities, axis=0, ddof=1)
        / BOLTZMANN_CONSTANT_J_PER_K
    )

    assert row["trapped_atom_count"] == 25
    assert row["aligned_trapped_atom_count"] == 25
    assert np.allclose(
        [
            row["plateau_temperature_x_k"],
            row["plateau_temperature_y_k"],
            row["plateau_temperature_z_k"],
        ],
        expected_components,
    )
    assert row["stationarity_pass"]
    assert row["statistics_pass"]
    assert row["valid"]
    assert row["quality_status"] == "valid"


def _ensemble_summary_row(index: int, temperature_k: float) -> dict[str, object]:
    return {
        "point_index": 0,
        "detuning_n": -2.5,
        "ensemble_index": index,
        "requested_atom_count": 25,
        "successful_atom_count": 25,
        "complete_atom_count": 25,
        "trapped_atom_count": 24,
        "aligned_trapped_atom_count": 24,
        "complete_outside_core_count": 1,
        "escaped_atom_count": 0,
        "other_incomplete_atom_count": 0,
        "failed_atom_count": 0,
        "duration_complete_fraction": 1.0,
        "trapped_fraction": 24 / 25,
        "initial_temperature_mean_k": 2.0e-3,
        "plateau_temperature_x_k": temperature_k,
        "plateau_temperature_y_k": temperature_k,
        "plateau_temperature_z_k": temperature_k,
        "plateau_temperature_mean_k": temperature_k,
        "final_temperature_mean_k": 1.01 * temperature_k,
        "stationarity_pass": True,
        "statistics_pass": True,
        "survivor_conditioning_warning": True,
        "valid": True,
        "quality_status": "valid_survivor_conditioned",
        "plateau_max_radius_median_m": 0.7e-3,
        "plateau_max_radius_p90_m": 1.1e-3,
        "plateau_max_radius_max_m": 2.5e-3,
        "termination_counts_json": json.dumps({"duration": 25}),
        "boundedness_counts_json": json.dumps(
            {
                "complete_final_plateau_core": 24,
                "complete_outside_final_plateau_core": 1,
            }
        ),
    }


def test_point_is_mean_and_sem_of_ten_ensemble_temperatures() -> None:
    temperatures = 1.0e-4 + np.arange(10) * 1.0e-5
    ensemble_rows = [
        _ensemble_summary_row(index, temperature)
        for index, temperature in enumerate(temperatures)
    ]
    row = _summarize_temperature_point(
        point_index=0,
        detuning_n=-2.5,
        ensemble_rows=ensemble_rows,
        requested_ensemble_count=10,
        atoms_per_ensemble=25,
        point_wall_time_s=1.0,
    )

    assert row["temperature_ensemble_count"] == 10
    assert row["requested_atom_count"] == 250
    assert np.isclose(row["plateau_temperature_mean_k"], np.mean(temperatures))
    assert np.isclose(
        row["temperature_sem_k"],
        np.std(temperatures, ddof=1) / np.sqrt(10),
    )
    assert row["temperature_ci_low_k"] < row["plateau_temperature_mean_k"]
    assert row["temperature_ci_high_k"] > row["plateau_temperature_mean_k"]
    assert np.isclose(row["trapped_fraction"], 24 / 25)
    assert row["trapped_atom_count"] == 240
    assert row["valid"]
    assert row["quality_status"] == "valid_survivor_conditioned"


def test_student_t_interval_uses_t9_for_ten_clouds() -> None:
    values = np.arange(10, dtype=float)
    count, mean, sem, low, high = _mean_sem_t_interval(values)

    assert count == 10
    assert mean == 4.5
    assert np.isclose(high - mean, 2.262157163 * sem)
    assert np.isclose(mean - low, 2.262157163 * sem)


def test_plotting_macro_shows_temperature_errors_survival_and_doppler_reference(
    tmp_path: Path,
) -> None:
    rows = []
    for index, value in enumerate(DETUNING_N_VALUES):
        temperature = 150.0e-6 + index * 2.0e-6
        rows.append(
            {
                "point_index": index,
                "detuning_n": value,
                "plateau_temperature_mean_k": temperature,
                "temperature_sem_k": 4.0e-6,
                "temperature_ci_low_k": temperature - 9.0e-6,
                "temperature_ci_high_k": temperature + 9.0e-6,
                "trapped_fraction": 0.9,
                "trapped_fraction_ci_low": 0.84,
                "trapped_fraction_ci_high": 0.96,
                "valid": index % 2 == 0,
            }
        )
    destination = tmp_path / "temperature_vs_detuning.png"

    assert plot_temperature_vs_detuning(rows, destination) == destination
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_signature_and_human_readable_model_record_the_complete_physics() -> None:
    audit = verify_only_cooling_detuning_changes((-10.0, -0.5))
    signature = _resume_signature(
        ensemble_realization_count=10,
        atoms_per_ensemble=25,
        duration_s=25e-3,
        time_step_s=5e-6,
        initial_temperature_k=2e-3,
        initial_position_sigma_m=0.25e-3,
        seed=20260822,
        plateau_window_s=5e-3,
        record_interval_s=0.1e-3,
        minimum_survivor_count=5,
        stationarity_limit=0.15,
        configuration_audit=audit,
    )
    markdown = physical_model_markdown(signature)

    assert json.loads(json.dumps(signature)) == signature
    assert signature["trajectory_count_per_point"] == 250
    assert signature["multilevel_config"]["repumper_enabled"]
    assert PHYSICAL_MODEL_STATEMENT in markdown
    assert "preloaded cloud" in markdown
    assert "center-of-mass" in markdown
    assert "Langevin recoil diffusion" in markdown
    assert "only" in markdown.lower() and "cooling detuning" in markdown
    assert "sub-Doppler" in markdown
    assert "trapped fraction could equal one" in markdown


class _ImmediateFuture:
    def __init__(self, value: object) -> None:
        self._value = value

    def result(self) -> object:
        return self._value


class _InlineExecutor:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function, payload):
        return _ImmediateFuture(function(payload))


def test_runner_writes_physical_model_and_both_statistical_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_worker(payload: sweep.TemperatureSweepWorkerPayload):
        times = np.linspace(0.0, payload.duration_s, 251)
        velocity = np.asarray(payload.velocity_m_per_s)
        return TemperatureSweepWorkerResult(
            ensemble_index=payload.ensemble_index,
            atom_index=payload.atom_index,
            complete=True,
            trapped=True,
            termination_reason="duration",
            boundedness_reason="complete_final_plateau_core",
            final_radius_m=0.4e-3,
            maximum_radius_m=0.8e-3,
            plateau_max_radius_m=0.6e-3,
            plateau_mean_radius_m=0.4e-3,
            times_s=times,
            velocities_m_per_s=np.repeat(velocity[None, :], len(times), axis=0),
        )

    monkeypatch.setattr(sweep, "DETUNING_N_VALUES", (-0.5,))
    monkeypatch.setattr(sweep, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(sweep, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(sweep, "_temperature_sweep_worker", fake_worker)
    output_directory = tmp_path / "statistics"
    figure_directory = tmp_path / "figures"

    result = run_temperature_detuning_sweep(
        output_directory=output_directory,
        figure_directory=figure_directory,
        resume=False,
    )

    assert result["status"] == "completed"
    assert len(result["rows"]) == 1
    assert len(result["ensemble_rows"]) == 10
    assert result["rows"][0]["requested_atom_count"] == 250
    physical_model_path = Path(result["outputs"]["physical_model_markdown"])
    assert physical_model_path.exists()
    assert PHYSICAL_MODEL_STATEMENT in physical_model_path.read_text(encoding="utf-8")
    assert Path(result["outputs"]["temperature_summary_csv"]).exists()
    assert Path(result["outputs"]["temperature_ensemble_csv"]).exists()
    assert Path(result["outputs"]["temperature_plot"]).exists()


def test_cli_defaults_and_production_cardinality_are_exact() -> None:
    args = build_argument_parser().parse_args([])
    assert args.ensemble_realizations == REQUIRED_ENSEMBLE_REALIZATION_COUNT == 10
    assert args.atoms_per_ensemble == REQUIRED_ATOMS_PER_ENSEMBLE == 25
    assert REQUIRED_TRAJECTORY_COUNT == 250
    assert args.duration_ms == 25.0
    assert args.dt_us == 5.0
    assert args.workers == 8
    assert args.resume

    with pytest.raises(ValueError, match="exactly 10"):
        run_temperature_detuning_sweep(ensemble_realization_count=9)
    with pytest.raises(ValueError, match="exactly 25"):
        run_temperature_detuning_sweep(atoms_per_ensemble=24)
