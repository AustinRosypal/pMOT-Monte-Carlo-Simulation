from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import pmot.mot_multilevel.capture as capture_module
from pmot.capture_statistics import CaptureVelocitySample, TrajectoryClassification
from pmot.launch_geometry import PointSample
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    CaptureSearchConfig,
    IndeterminateCaptureError,
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    capture_cross_section_spectrum,
    find_capture_velocity,
    generate_capture_launches,
    run_capture_loading_study,
    simulate_rate_equation_trajectory,
)


def _point() -> PointSample:
    return PointSample(
        disc_index=0,
        point_index=0,
        theta_rad=0.0,
        phi_rad=0.0,
        theta_prime_rad=0.0,
        s_m=0.0,
        radial_distance_m=15.0e-3,
        initial_position_m=(0.0, 0.0, 15.0e-3),
        incident_unit_vector=(0.0, 0.0, -1.0),
        launch_axis_unit_vector=(0.0, 0.0, -1.0),
    )


def _classification(speed: float, trapped: bool, reason: str) -> TrajectoryClassification:
    return TrajectoryClassification(
        trapped=trapped,
        termination_reason=reason,
        entered_trap_core=trapped,
        core_entry_count=2 if trapped else 1,
        elapsed_time_s=0.01,
        minimum_radius_m=0.0,
        final_radius_m=0.001 if trapped else 0.03,
        final_position_m=(0.0, 0.0, 0.0),
        final_velocity_m_per_s=(0.0, 0.0, speed),
    )


def test_full_sphere_launch_sampling_is_seeded_and_uniform_area() -> None:
    search = CaptureSearchConfig(disc_count=3, points_per_disc=4, seed=17)
    discs_a, points_a = generate_capture_launches(search)
    discs_b, points_b = generate_capture_launches(search)
    assert discs_a == discs_b
    assert points_a == points_b
    assert len(discs_a) == 3
    assert len(points_a) == 12
    assert all(0.0 <= point.s_m <= search.disc_radius_m for point in points_a)


def test_capture_velocity_search_requires_a_trapped_escaped_bracket() -> None:
    search = CaptureSearchConfig(
        disc_count=1,
        points_per_disc=1,
        initial_velocity_guess_m_per_s=5.0,
        velocity_tolerance_m_per_s=0.25,
    )

    def classifier(point, speed, search, **kwargs):
        trapped = speed <= 7.0
        return _classification(speed, trapped, "two_core_entries" if trapped else "escaped")

    sample = find_capture_velocity(_point(), search, classifier=classifier)
    assert sample.trapped_velocity_lower_m_per_s <= 7.0
    assert sample.untrapped_velocity_upper_m_per_s > 7.0
    assert sample.velocity_resolution_m_per_s <= 0.25
    assert sample.lower_classification == "two_core_entries"
    assert sample.upper_classification == "escaped"


def test_capture_velocity_search_fails_closed_on_timeout() -> None:
    search = CaptureSearchConfig(disc_count=1, points_per_disc=1)

    def classifier(point, speed, search, **kwargs):
        return _classification(speed, False, "timeout")

    with pytest.raises(IndeterminateCaptureError, match="increase the trajectory duration"):
        find_capture_velocity(_point(), search, classifier=classifier)


def test_capture_config_rejects_nonphysical_values() -> None:
    with pytest.raises(ValueError):
        replace(CaptureSearchConfig(), time_step_s=0.0)


def test_multilevel_trajectory_defaults_use_five_microsecond_steps() -> None:
    assert CaptureSearchConfig().time_step_s == pytest.approx(5.0e-6)
    assert RateEquationTrajectoryConfig().time_step_s == pytest.approx(5.0e-6)


def test_uncaptured_zero_threshold_is_not_counted_at_zero_speed(tmp_path) -> None:
    point = _point()
    trapped = _classification(1.0, True, "two_core_entries")
    escaped = _classification(2.0, False, "escaped")
    samples = [
        capture_module._capture_sample(point, 0.0, 0.25, escaped, escaped),
        capture_module._capture_sample(point, 1.0, 1.25, trapped, escaped),
    ]
    search = CaptureSearchConfig(
        disc_count=1,
        points_per_disc=2,
        analysis_velocity_step_m_per_s=0.5,
        analysis_velocity_max_m_per_s=2.0,
        analysis_s_bin_count=2,
    )
    spectrum = capture_cross_section_spectrum(samples, search)
    assert spectrum[0].velocity_m_per_s == 0.0
    assert spectrum[0].captured_count == 1
    assert spectrum[-1].captured_count == 0
    assert capture_module.plot_capture_probability_heatmap(
        samples,
        search,
        tmp_path,
    ).exists()


def test_capture_rk4_step_matches_main_trajectory_integrator() -> None:
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams()
    coil = default_anti_helmholtz_config()
    initial = RateEquationAtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0))
    dt = 5.0e-6
    record = simulate_rate_equation_trajectory(
        initial,
        dt,
        coil,
        beams=beams,
        model=model,
        trajectory_config=RateEquationTrajectoryConfig(time_step_s=dt),
    )
    position, velocity, _ = capture_module._rk4_step(
        np.asarray(initial.position_m),
        np.asarray(initial.velocity_m_per_s),
        dt,
        initial.last_quantization_axis,
        model,
        beams,
        coil,
        capture_module.default_multilevel_mot_config(),
    )
    np.testing.assert_allclose(position, record.positions_m[-1], rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(velocity, record.velocities_m_per_s[-1], rtol=0.0, atol=1e-12)


def test_fast_real_multilevel_launch_definitively_escapes() -> None:
    search = CaptureSearchConfig(
        disc_count=1,
        points_per_disc=1,
        maximum_simulation_time_s=5.0e-3,
        time_step_s=10.0e-6,
    )
    point = generate_capture_launches(search)[1][0]
    result = capture_module.classify_capture_trajectory(point, 60.0, search)
    assert result.termination_reason == "escaped"
    assert not result.trapped


def test_capture_loading_workflow_writes_cross_section_and_loading_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_worker(payload):
        point, _, _, _ = payload
        threshold = 1.0 + point.point_index
        return CaptureVelocitySample(
            disc_index=point.disc_index,
            point_index=point.point_index,
            theta_rad=point.theta_rad,
            phi_rad=point.phi_rad,
            theta_prime_rad=point.theta_prime_rad,
            s_m=point.s_m,
            radial_distance_m=point.radial_distance_m,
            initial_position_m=point.initial_position_m,
            incident_unit_vector=point.incident_unit_vector,
            capture_velocity_m_per_s=threshold,
            velocity_resolution_m_per_s=0.25,
            trapped_velocity_lower_m_per_s=threshold,
            untrapped_velocity_upper_m_per_s=threshold + 0.25,
            lower_classification="two_core_entries",
            upper_classification="escaped",
            lower_entered_trap_core=True,
            upper_entered_trap_core=True,
            lower_core_entry_count=2,
            upper_core_entry_count=1,
        )

    monkeypatch.setattr(capture_module, "_worker", fake_worker)
    search = CaptureSearchConfig(
        disc_count=1,
        points_per_disc=2,
        worker_count=1,
        analysis_velocity_step_m_per_s=0.5,
        analysis_velocity_min_m_per_s=0.0,
        analysis_velocity_max_m_per_s=3.0,
        analysis_s_bin_count=2,
    )
    result = run_capture_loading_study(
        search,
        output_directory=tmp_path / "statistics",
        figure_directory=tmp_path / "figures",
    )
    assert len(result.samples) == 2
    assert len(result.spectrum) == 7
    assert result.loading.loading_rate_atoms_per_s > 0.0
    assert all(path.exists() for path in result.output_paths.values())

    def should_not_run(payload):
        raise AssertionError("a completed matching run must resume without new capture work")

    monkeypatch.setattr(capture_module, "_worker", should_not_run)
    resumed = run_capture_loading_study(
        search,
        output_directory=tmp_path / "statistics",
        figure_directory=tmp_path / "figures",
        resume=True,
    )
    assert resumed.samples == result.samples
    with pytest.raises(ValueError, match="does not match"):
        run_capture_loading_study(
            replace(search, seed=search.seed + 1),
            output_directory=tmp_path / "statistics",
            figure_directory=tmp_path / "figures",
            resume=True,
        )
    with pytest.raises(ValueError, match="does not match"):
        run_capture_loading_study(
            search,
            coil_config=default_anti_helmholtz_config(target_gradient_g_per_cm=11.0),
            output_directory=tmp_path / "statistics",
            figure_directory=tmp_path / "figures",
            resume=True,
        )


def test_capture_loading_does_not_overwrite_unmanifested_output(tmp_path) -> None:
    output = tmp_path / "statistics"
    output.mkdir()
    existing = output / "capture_velocity_samples.csv"
    existing.write_text("existing user data\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="without a run manifest"):
        run_capture_loading_study(
            CaptureSearchConfig(disc_count=1, points_per_disc=1),
            output_directory=output,
            figure_directory=tmp_path / "figures",
        )
    assert existing.read_text(encoding="utf-8") == "existing user data\n"
