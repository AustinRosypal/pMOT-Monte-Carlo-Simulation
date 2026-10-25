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
