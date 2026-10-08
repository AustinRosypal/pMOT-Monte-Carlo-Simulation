from __future__ import annotations

import numpy as np

from pmot.pmot.surrogate_capture_campaign import (
    CampaignConfig,
    OUTCOME_ESCAPE,
    OUTCOME_OFF_ORIGIN,
    OUTCOME_ORIGIN,
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
