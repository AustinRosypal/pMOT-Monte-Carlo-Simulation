"""Focused provenance checks for the September 2026 MOT refinement."""

from __future__ import annotations

from math import prod
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from pmot.mot_multilevel import refined_relationship_campaign as campaign
from pmot.mot_multilevel import relationship_sweeps as historical
from pmot.mot_multilevel.force_sweep import build_force_sweep_configuration
from pmot.mot_multilevel.power_loading_study import generate_study_geometry


def test_exact_refined_grids_counts_endpoints_and_steps() -> None:
    assert campaign.RAW_SATURATION_VALUES == (
        0.25,
        0.5,
        0.75,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        125.0,
    )
    assert len(campaign.EFFECTIVE_SATURATION_VALUES) == 20
    assert campaign.EFFECTIVE_SATURATION_VALUES == pytest.approx(
        np.arange(0.25, 5.0 + 0.125, 0.25)
    )
    assert len(campaign.DETUNING_N_VALUES) == 23
    assert campaign.DETUNING_N_VALUES == pytest.approx(
        -np.arange(0.5, 6.0 + 0.125, 0.25)
    )
    assert len(campaign.FORCE_DETUNING_N_VALUES) == 111
    assert campaign.FORCE_DETUNING_N_VALUES[0] == pytest.approx(-0.5)
    assert campaign.FORCE_DETUNING_N_VALUES[-1] == pytest.approx(-6.0)
    assert np.diff(campaign.FORCE_DETUNING_N_VALUES) == pytest.approx(-0.05)


def test_refined_loading_design_is_25x25_full_sphere_at_r15mm() -> None:
    search = campaign.default_refined_search_config(worker_count=1)
    assert search.phase_space == "full_sphere"
    assert search.disc_count == campaign.DEFAULT_DISC_COUNT == 25
    assert search.points_per_disc == campaign.DEFAULT_POINTS_PER_DISC == 25
    assert search.disc_radius_m == campaign.DEFAULT_DISC_RADIUS_M == pytest.approx(
        15.0e-3
    )
    assert search.seed == campaign.DEFAULT_SEED == 20260903
    assert not search.include_center_point

    discs, points = generate_study_geometry(search)
    assert len(discs) == 25
    assert len(points) == 625
    centers = np.asarray([disc.center_position_m for disc in discs])
    assert np.all(np.min(centers, axis=0) < 0.0)
    assert np.all(np.max(centers, axis=0) > 0.0)


def test_refined_loading_studies_are_sequential_groups_not_a_product() -> None:
    groups = campaign.requested_refined_points()
    assert tuple(groups) == (
        historical.RAW_STUDY_KEY,
        historical.EFFECTIVE_STUDY_KEY,
        historical.DETUNING_STUDY_KEY,
    )
    group_sizes = tuple(map(len, groups.values()))
    assert group_sizes == (24, 20, 23)
    assert sum(group_sizes) == 67
    assert prod(group_sizes) == 11_040
    assert all(
        point.study_key == study_key
        for study_key, points in groups.items()
        for point in points
    )

    detuning_points = groups[historical.DETUNING_STUDY_KEY]
    assert all(
        point.cooling_power_w_per_beam
        == pytest.approx(historical.DEFAULT_COOLING_POWER_W_PER_BEAM)
        for point in detuning_points
    )
    assert historical.DEFAULT_COOLING_POWER_W_PER_BEAM == pytest.approx(27.0e-3)

    effective_endpoint = groups[historical.EFFECTIVE_STUDY_KEY][-1]
    assert effective_endpoint.effective_saturation == pytest.approx(5.0)
    assert effective_endpoint.on_resonance_saturation == pytest.approx(
        127.133592437323
    )
    assert effective_endpoint.cooling_power_w_per_beam == pytest.approx(
        0.134395269744593
    )


def test_refined_outputs_and_manifest_are_isolated_and_explicit(tmp_path: Path) -> None:
    paths = campaign.default_refined_campaign_paths(tmp_path)
    expected_statistics = (
        tmp_path
        / "outputs"
        / "statistics"
        / "mot_multilevel"
        / campaign.CAMPAIGN_NAME
    )
    expected_figures = (
        tmp_path
        / "outputs"
        / "figures"
        / "mot_multilevel"
        / campaign.CAMPAIGN_NAME
    )
    assert paths.statistics == expected_statistics
    assert paths.figures == expected_figures
    assert "25x25" in campaign.CAMPAIGN_NAME
    assert "r15mm" in campaign.CAMPAIGN_NAME
    assert "27mW_reference" in campaign.CAMPAIGN_NAME
    assert paths != historical.default_campaign_paths(tmp_path)
    assert paths != historical.default_remaining_campaign_paths(tmp_path)

    search = campaign.default_refined_search_config(worker_count=1)
    metadata = campaign._new_campaign_metadata(paths, search)
    assert metadata["combinatorial_product_used"] is False
    assert metadata["loading_studies_are_independent_and_sequential"] is True
    assert metadata["loading_point_count"] == 67
    assert metadata["capture_threshold_search_count"] == 67 * 25 * 25
    assert metadata["fixed_parameters"] == {
        "detuning_and_temperature_cooling_power_w_per_beam": pytest.approx(
            27.0e-3
        ),
        "repump_power_w_per_beam": pytest.approx(0.1e-3),
        "disc_radius_m": pytest.approx(15.0e-3),
    }


def test_temperature_wrapper_uses_25_preloaded_clouds_by_25_atoms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = campaign.default_refined_campaign_paths(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_temperature_sweep(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "completed_point_count": len(campaign.DETUNING_N_VALUES),
            "outputs": {},
        }

    monkeypatch.setattr(
        campaign,
        "run_temperature_detuning_sweep",
        fake_temperature_sweep,
    )
    result = campaign.run_refined_temperature_sweep(
        worker_count=1,
        paths=paths,
        resume=False,
    )

    assert result["status"] == "completed"
    assert len(calls) == 1
    call = calls[0]
    assert call["ensemble_realization_count"] == 25
    assert call["atoms_per_ensemble"] == 25
    assert call["ensemble_realization_count"] * call["atoms_per_ensemble"] == 625
    assert call["detuning_n_values"] == campaign.DETUNING_N_VALUES
    assert call["cooling_power_w_per_beam"] == pytest.approx(27.0e-3)
    assert call["seed"] == campaign.TEMPERATURE_SEED
    assert call["output_directory"] == paths.temperature_statistics
    assert call["figure_directory"] == paths.temperature_figures

    metadata = campaign._new_campaign_metadata(
        paths,
        campaign.default_refined_search_config(worker_count=1),
    )
    design = metadata["temperature_design"]
    assert design["independent_preloaded_cloud_count"] == 25
    assert design["atoms_per_cloud"] == 25
    assert design["trajectories_per_detuning"] == 625
    assert "rather than incident sampling discs" in design["physical_interpretation"]
    assert "detuning-dependent multilevel Doppler" in design["theory_overlay"]


def test_force_wrapper_uses_dense_grid_and_power_qualified_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = campaign.default_refined_campaign_paths(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_force_sweep(**kwargs):
        calls.append(kwargs)
        return {
            "status": "completed",
            "completed_point_count": len(campaign.FORCE_DETUNING_N_VALUES),
            "outputs": {},
        }

    monkeypatch.setattr(campaign, "run_force_detuning_sweep", fake_force_sweep)
    result = campaign.run_refined_force_sweep(paths=paths, resume=False)

    assert result["status"] == "completed"
    assert len(calls) == 1
    call = calls[0]
    assert call["detuning_n_values"] == campaign.FORCE_DETUNING_N_VALUES
    assert call["output_directory"] == paths.statistics / "04_force_vs_detuning_27mW"
    assert call["figure_directory"] == paths.figures / "04_force_vs_detuning_27mW"

    config, apparatus, beams = build_force_sweep_configuration(-1.0)
    assert config.repumper_enabled
    assert apparatus.cooling.power_w_per_beam == pytest.approx(27.0e-3)
    assert apparatus.repump.power_w_per_beam == pytest.approx(0.1e-3)
    assert [beam.power_w for beam in beams if beam.family == "cooling"] == pytest.approx(
        [27.0e-3] * 6
    )
    assert [beam.power_w for beam in beams if beam.family == "repump"] == pytest.approx(
        [0.1e-3] * 6
    )


@pytest.mark.parametrize(
    ("study_key", "expected_limits"),
    (
        (historical.RAW_STUDY_KEY, (0.0, 127.5)),
        (historical.EFFECTIVE_STUDY_KEY, (0.0, 5.1)),
        (historical.DETUNING_STUDY_KEY, (-6.11, -0.39)),
    ),
)
def test_refined_loading_plots_export_requested_x_ranges(
    study_key: str,
    expected_limits: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_limits: list[tuple[float, float]] = []
    original_set = matplotlib.axes.Axes.set

    def spy_set(axis, *args, **kwargs):
        if "xlim" in kwargs:
            captured_limits.append(tuple(kwargs["xlim"]))
        return original_set(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set", spy_set)
    rows = [
        {
            "s0": 0.25,
            "seff": 0.25,
            "detuning_n": -6.0,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 10.0e6,
            "loading_rate_t95_lower_atoms_per_s": 9.0e6,
            "loading_rate_t95_upper_atoms_per_s": 12.0e6,
        },
        {
            "s0": 125.0,
            "seff": 5.0,
            "detuning_n": -0.5,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 20.0e6,
            "loading_rate_t95_lower_atoms_per_s": 18.0e6,
            "loading_rate_t95_upper_atoms_per_s": 23.0e6,
        },
    ]
    destination = tmp_path / f"{study_key}.png"

    assert campaign.plot_refined_loading_relationship(
        rows,
        study_key,
        destination,
        search_config=campaign.default_refined_search_config(worker_count=1),
    ) == destination
    assert destination.is_file()
    assert destination.stat().st_size > 1_000
    assert captured_limits[-1] == expected_limits


def test_raw_saturation_plot_marks_125_without_clipping_and_uses_clear_ci_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_ticks: list[np.ndarray] = []
    captured_labels: list[str | None] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks
    original_errorbar = matplotlib.axes.Axes.errorbar

    def spy_set_xticks(axis, ticks, *args, **kwargs):
        captured_ticks.append(np.asarray(ticks, dtype=float))
        return original_set_xticks(axis, ticks, *args, **kwargs)

    def spy_errorbar(axis, *args, **kwargs):
        captured_labels.append(kwargs.get("label"))
        return original_errorbar(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy_errorbar)
    rows = [
        {
            "s0": 0.25,
            "seff": 0.25,
            "detuning_n": -6.0,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 10.0e6,
            "loading_rate_t95_lower_atoms_per_s": 9.0e6,
            "loading_rate_t95_upper_atoms_per_s": 12.0e6,
        },
        {
            "s0": 125.0,
            "seff": 5.0,
            "detuning_n": -0.5,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 20.0e6,
            "loading_rate_t95_lower_atoms_per_s": 18.0e6,
            "loading_rate_t95_upper_atoms_per_s": 23.0e6,
        },
    ]

    campaign.plot_refined_loading_relationship(
        rows,
        historical.RAW_STUDY_KEY,
        tmp_path / "raw_saturation.png",
        search_config=campaign.default_refined_search_config(worker_count=1),
    )

    assert captured_ticks
    assert 125.0 in captured_ticks[-1]
    # The full-range trace owns the legend entry.  A second unlabeled
    # errorbar call supplies the low-s0 inset without duplicating the legend.
    assert captured_labels[0] == "Mean ± 95% CI"
    assert all(label is None for label in captured_labels[1:])


@pytest.mark.parametrize(
    ("study_key", "expected_endpoints"),
    (
        (historical.EFFECTIVE_STUDY_KEY, (0.0, 5.0)),
        (historical.DETUNING_STUDY_KEY, (-6.0, -0.5)),
    ),
)
def test_refined_loading_plot_keeps_requested_endpoints_visible_and_labeled(
    study_key: str,
    expected_endpoints: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_ticks: list[np.ndarray] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks

    def spy_set_xticks(axis, ticks, *args, **kwargs):
        captured_ticks.append(np.asarray(ticks, dtype=float))
        return original_set_xticks(axis, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)
    rows = [
        {
            "s0": 0.25,
            "seff": 0.25,
            "detuning_n": -6.0,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 10.0e6,
            "loading_rate_t95_lower_atoms_per_s": 9.0e6,
            "loading_rate_t95_upper_atoms_per_s": 12.0e6,
        },
        {
            "s0": 125.0,
            "seff": 5.0,
            "detuning_n": -0.5,
            "cooling_detuning_n": -2.4711696869851727,
            "loading_rate_mean_atoms_per_s": 20.0e6,
            "loading_rate_t95_lower_atoms_per_s": 18.0e6,
            "loading_rate_t95_upper_atoms_per_s": 23.0e6,
        },
    ]

    campaign.plot_refined_loading_relationship(
        rows,
        study_key,
        tmp_path / f"{study_key}.png",
        search_config=campaign.default_refined_search_config(worker_count=1),
    )

    assert captured_ticks
    for endpoint in expected_endpoints:
        assert endpoint in captured_ticks[-1]
