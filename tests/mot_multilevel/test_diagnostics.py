from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    create_population_histogram_animation,
    create_trajectory_animation,
    plot_time_diagnostics,
    plot_trajectory_3d,
    save_trajectory,
    simulate_rate_equation_trajectory,
    trajectory_summary,
    trajectory_table,
)


def _short_record():
    return simulate_rate_equation_trajectory(
        RateEquationAtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
        5.0e-6,
        default_anti_helmholtz_config(),
        trajectory_config=RateEquationTrajectoryConfig(time_step_s=5.0e-6),
    )


def test_trajectory_diagnostics_use_new_record_fields(tmp_path) -> None:
    record = _short_record()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams()
    summary = trajectory_summary(record, minimum_duration_s=0.0)
    assert summary["step_count"] == 1
    assert summary["maximum_optical_force_n"] > 0.0
    frame = trajectory_table(record)
    assert len(frame) == 2
    assert {"Fx_N", "spontaneous_scattering_rate_per_s", "population_23"} <= set(frame)
    files = save_trajectory(record, model, beams, tmp_path / "trajectory")
    assert all(path.exists() for path in files)
    metadata = json.loads(files[-1].read_text(encoding="utf-8"))
    assert metadata["schema"] == "pmot.mot_multilevel.population-rate-trajectory.v1"
    assert metadata["diffusion"] == "not implemented; deterministic mean force"


def test_static_plots_and_animation_are_constructible() -> None:
    record = _short_record()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams()
    trajectory_figure = plot_trajectory_3d(record, beams)
    diagnostic_figure = plot_time_diagnostics(record, model, beams)
    movie = create_trajectory_animation(record, beams, max_frames=2, fps=10)
    population_movie = create_population_histogram_animation(record, model, max_frames=2, fps=10)
    assert len(trajectory_figure.axes) >= 1
    assert len(diagnostic_figure.axes) == 6
    assert movie._save_count == 2
    assert population_movie._save_count == 2
    movie._draw_was_started = True
    population_movie._draw_was_started = True
    plt.close(trajectory_figure)
    plt.close(diagnostic_figure)
    plt.close(movie._fig)
    plt.close(population_movie._fig)
