from __future__ import annotations

import numpy as np
import pytest

import pmot.mot_strange.surrogate_capture_campaign as campaign_module
from pmot.mot_strange.surrogate_capture_campaign import (
    CampaignConfig,
    OUTCOME_ESCAPE,
    OUTCOME_INDETERMINATE,
    OUTCOME_OFF_ORIGIN,
    OUTCOME_ORIGIN,
    _mean_ci,
    aggregate_results,
    build_launch_tasks,
    classify_trajectory,
)


def test_full_sphere_campaign_geometry_is_complete_and_inside_cell() -> None:
    config = CampaignConfig()
    tasks = build_launch_tasks(config)
    assert len(tasks) == (
        config.direction_discs
        * config.points_per_disc
        * len(config.speeds_m_per_s)
    )
    assert max(max(abs(value) for value in task.initial_position_m) for task in tasks) < 14.1421356e-3


def test_origin_residence_classification() -> None:
    config = CampaignConfig()
    times = np.linspace(0.0, 10.0e-3, 1001)
    positions = np.zeros((len(times), 3))
    positions[:, 0] = 1.0e-3
    velocities = np.zeros_like(positions)
    result = classify_trajectory(times, positions, velocities, "duration", config)
    assert result["outcome"] == OUTCOME_ORIGIN


def test_conservative_off_origin_and_wall_classifications() -> None:
    config = CampaignConfig()
    times = np.linspace(0.0, 20.0e-3, 2001)
    positions = np.zeros((len(times), 3))
    positions[:, 0] = 5.0e-3
    velocities = np.zeros_like(positions)
    assert classify_trajectory(times, positions, velocities, "duration", config)["outcome"] == OUTCOME_OFF_ORIGIN
    assert classify_trajectory(times, positions, velocities, "wall_loss", config)["outcome"] == OUTCOME_ESCAPE


def test_cluster_interval_is_recomputed_from_each_parameter_distribution() -> None:
    _, narrow_low, narrow_high = _mean_ci((1.0, 1.1, 0.9, 1.0))
    _, wide_low, wide_high = _mean_ci((0.0, 2.0, 0.0, 2.0))
    assert wide_high - wide_low > narrow_high - narrow_low


def test_loading_is_withheld_when_direct_mask_contains_indeterminate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CampaignConfig(
        direction_discs=2,
        points_per_disc=1,
        speeds_m_per_s=(1.0, 3.0),
    )
    outcomes = (OUTCOME_ORIGIN, OUTCOME_ESCAPE, OUTCOME_INDETERMINATE, OUTCOME_ESCAPE)
    rows = []
    for index, (disc, speed) in enumerate(((0, 1.0), (0, 3.0), (1, 1.0), (1, 3.0))):
        rows.append(
            {
                "disc_index": str(disc),
                "point_index": "0",
                "speed_m_per_s": str(speed),
                "outcome": outcomes[index],
                "final_centroid_m": "[0.0, 0.0, 0.0]",
            }
        )
    monkeypatch.setattr(campaign_module, "make_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign_module, "write_report", lambda *args, **kwargs: None)

    summary = aggregate_results(rows, config, tmp_path)

    assert summary["loading"] == {}
    assert summary["loading_status"] == "withheld_unresolved_trajectories"
    assert summary["loading_withheld_indeterminate_count"] == 1


def test_in_cell_cross_sections_use_each_clipped_area(tmp_path, monkeypatch):
    config = CampaignConfig(direction_discs=2, points_per_disc=1,
                            speeds_m_per_s=(1.0, 3.0), in_cell_launches=True)
    rows = []
    for disc, fraction in enumerate((0.5, 0.25)):
        for speed in config.speeds_m_per_s:
            rows.append({"disc_index": str(disc), "point_index": "0",
                         "speed_m_per_s": str(speed), "outcome": OUTCOME_ORIGIN,
                         "launch_area_m2": str(config.sample_area_m2*fraction),
                         "final_centroid_m": "[0.0, 0.0, 0.0]"})
    monkeypatch.setattr(campaign_module, "make_plots", lambda *a: None)
    monkeypatch.setattr(campaign_module, "write_report", lambda *a: None)
    result = aggregate_results(rows, config, tmp_path)
    import csv
    with (tmp_path / "cross_section_spectrum.csv").open() as handle:
        spectra = list(csv.DictReader(handle))
    origin = [r for r in spectra if r["category"] == "origin"]
    assert all(float(r["mean_cross_section_m2"]) == pytest.approx(
        config.sample_area_m2 * 0.375) for r in origin)
    assert result["launch_area_weighting"] == "disc intersection with cell"
    for row in rows:
        del row["launch_area_m2"]
    with pytest.raises(ValueError, match="clipped launch area"):
        aggregate_results(rows, config, tmp_path)


def test_trajectory_worker_receives_calibrated_field_and_cell(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from pmot.mot_multilevel import rate_equations
    from pmot.mot_strange.surrogate_effective_field import calibrated_surrogate_field_config
    field = replace(calibrated_surrogate_field_config(), cell_half_length_m=0.012)
    cfg = CampaignConfig(direction_discs=1, points_per_disc=1,
                         speeds_m_per_s=(1.0,), in_cell_launches=True,
                         effective_field_config=field)
    task = build_launch_tasks(cfg)[0]
    monkeypatch.setattr(campaign_module, "_worker_context", lambda: (None, None, None))

    def fake_trajectory(*args, magnetic_field_function, spatial_domain_function, **kwargs):
        step = 1e-8
        derivative = (magnetic_field_function((step, 0, 0))
                      - magnetic_field_function((-step, 0, 0))) / (2*step)
        assert derivative[0] * 100 == pytest.approx(10.0, rel=1e-8)
        assert not spatial_domain_function((0.013, 0, 0))
        return SimpleNamespace(times_s=[0.0, 0.006],
                               positions_m=[(0, 0, 0), (0, 0, 0)],
                               velocities_m_per_s=[(0, 0, 0), (0, 0, 0)],
                               termination_reason="duration")

    monkeypatch.setattr(rate_equations, "simulate_rate_equation_trajectory", fake_trajectory)
    row = campaign_module._run_task(task, cfg, str(tmp_path))
    assert row["outcome"] == OUTCOME_ORIGIN
    assert row["launch_area_m2"] == task.launch_area_m2
