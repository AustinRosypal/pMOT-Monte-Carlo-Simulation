"""Focused checks for the independent multilevel loading relationships."""

from __future__ import annotations

import csv
import json
from math import pi
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from pmot.mot_multilevel import relationship_sweeps as sweeps
from pmot.mot_multilevel.power_loading_study import (
    generate_study_geometry,
    geometry_rows,
    geometry_sha256,
)


def _geometry_digest(search) -> str:
    discs, points = generate_study_geometry(search)
    return geometry_sha256(geometry_rows(discs, points))


def _write_temperature_products(
    paths: sweeps.CampaignPaths,
    *,
    status: str = "completed",
    ensemble_count: int = sweeps.TEMPERATURE_ENSEMBLE_COUNT,
    atoms_per_ensemble: int = sweeps.TEMPERATURE_ATOMS_PER_ENSEMBLE,
    detuning_values: tuple[float, ...] | None = None,
) -> tuple[Path, Path]:
    detunings = detuning_values or sweeps.DETUNING_N_VALUES
    statistics = paths.temperature_statistics
    statistics.mkdir(parents=True, exist_ok=True)
    summary_csv = statistics / "temperature_vs_detuning.csv"
    metadata_json = statistics / "temperature_vs_detuning_metadata.json"
    requested_atom_count = ensemble_count * atoms_per_ensemble

    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "detuning_n",
                "requested_ensemble_count",
                "requested_atom_count",
                "cooling_power_w_per_beam",
            ),
        )
        writer.writeheader()
        for detuning_n in detunings:
            writer.writerow(
                {
                    "detuning_n": detuning_n,
                    "requested_ensemble_count": ensemble_count,
                    "requested_atom_count": requested_atom_count,
                    "cooling_power_w_per_beam": (
                        sweeps.DEFAULT_COOLING_POWER_W_PER_BEAM
                    ),
                }
            )

    metadata = {
        "status": status,
        "completed_point_count": len(detunings),
        "completed_ensemble_row_count": len(detunings) * ensemble_count,
        "resume_signature": {
            "solver": (
                "24_state_repumper_adiabatic_population_rate_equation_langevin"
            ),
            "ensemble_realization_count": ensemble_count,
            "atoms_per_ensemble": atoms_per_ensemble,
            "trajectory_count_per_point": requested_atom_count,
            "detuning_n_values": list(detunings),
            "cooling_power_w_per_beam": (
                sweeps.DEFAULT_COOLING_POWER_W_PER_BEAM
            ),
            "multilevel_config": {
                "repumper_enabled": True,
                "repump_power_w_per_beam": sweeps.REPUMP_POWER_W_PER_BEAM,
            },
            "apparatus_config": {
                "repump": {
                    "power_w_per_beam": sweeps.REPUMP_POWER_W_PER_BEAM,
                }
            },
        },
    }
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_csv, metadata_json


def test_exact_requested_relationship_grids() -> None:
    assert sweeps.RAW_SATURATION_VALUES == (
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
    )
    assert sweeps.EFFECTIVE_SATURATION_VALUES == (
        0.25,
        0.5,
        0.75,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
        10.0,
        12.0,
        15.0,
        18.0,
        20.0,
        22.0,
        25.0,
    )
    assert sweeps.DETUNING_N_VALUES == (
        -0.1,
        -0.25,
        -0.5,
        -0.75,
        -1.0,
        -2.0,
        -2.5,
        -3.0,
        -4.0,
        -5.0,
        -7.0,
        -10.0,
        -12.0,
        -15.0,
    )


def test_saturation_power_and_detuning_conversions_are_exact() -> None:
    power_for_s0_one = sweeps.saturation_power_w_per_beam(1.0)
    assert power_for_s0_one == pytest.approx(1.0571184781827414e-3)
    assert sweeps.on_resonance_saturation_parameter(power_for_s0_one) == pytest.approx(
        1.0
    )

    reference_s0 = sweeps.on_resonance_saturation_parameter(
        sweeps.DEFAULT_COOLING_POWER_W_PER_BEAM
    )
    linewidth_hz = (
        sweeps.default_multilevel_mot_config().natural_linewidth_rad_per_s
        / (2.0 * pi)
    )
    reference_detuning_n = sweeps.COOLING_DETUNING_HZ / linewidth_hz
    assert reference_s0 == pytest.approx(25.54112955263306)
    assert sweeps.detuning_reduction_denominator(reference_detuning_n) == pytest.approx(
        25.426718487464557
    )
    assert sweeps.effective_saturation_from_s0(
        reference_s0, reference_detuning_n
    ) == pytest.approx(1.0044996392124688)

    effective_points = sweeps.build_relationship_points(
        sweeps.EFFECTIVE_STUDY_KEY,
        (0.25, 1.0, 25.0),
    )
    assert [point.effective_saturation for point in effective_points] == pytest.approx(
        [0.25, 1.0, 25.0]
    )
    for point in effective_points:
        reconstructed_s0 = sweeps.on_resonance_saturation_parameter(
            point.cooling_power_w_per_beam
        )
        assert reconstructed_s0 == pytest.approx(point.on_resonance_saturation)
        assert sweeps.effective_saturation_from_s0(
            reconstructed_s0, point.cooling_detuning_n
        ) == pytest.approx(point.scan_value)

    detuning_points = sweeps.build_relationship_points(
        sweeps.DETUNING_STUDY_KEY,
        (-0.5, -2.5, -15.0),
    )
    assert all(
        point.cooling_power_w_per_beam
        == pytest.approx(sweeps.DEFAULT_COOLING_POWER_W_PER_BEAM)
        for point in detuning_points
    )
    assert [point.cooling_detuning_hz / linewidth_hz for point in detuning_points] == (
        pytest.approx([-0.5, -2.5, -15.0])
    )


def test_stage_specific_paths_and_geometry_preserve_30x30_and_use_remaining_15x15(
    tmp_path: Path,
) -> None:
    saturation_paths = sweeps.default_campaign_paths(tmp_path)
    remaining_paths = sweeps.default_remaining_campaign_paths(tmp_path)
    assert saturation_paths.statistics.name == sweeps.SATURATION_CAMPAIGN_NAME
    assert remaining_paths.statistics.name == sweeps.REMAINING_CAMPAIGN_NAME
    assert saturation_paths.statistics != remaining_paths.statistics
    assert saturation_paths.figures != remaining_paths.figures
    assert "30x30" in saturation_paths.statistics.name
    assert "15x15" in remaining_paths.statistics.name

    saturation_search = sweeps.default_saturation_search_config(worker_count=1)
    remaining_search = sweeps.default_relationship_search_config(worker_count=1)
    assert (saturation_search.disc_count, saturation_search.points_per_disc) == (30, 30)
    assert (remaining_search.disc_count, remaining_search.points_per_disc) == (15, 15)
    for search, expected_discs, expected_points in (
        (saturation_search, 30, 900),
        (remaining_search, 15, 225),
    ):
        assert search.disc_radius_m == sweeps.DEFAULT_DISC_RADIUS_M == pytest.approx(
            15.0e-3
        )
        assert search.phase_space == "full_sphere"
        assert not search.include_center_point
        discs, points = generate_study_geometry(search)
        assert len(discs) == expected_discs
        assert len(points) == expected_points
        assert all(0.0 < point.s_m < search.disc_radius_m for point in points)
        centers = np.asarray([disc.center_position_m for disc in discs])
        assert np.all(np.min(centers, axis=0) < 0.0)
        assert np.all(np.max(centers, axis=0) > 0.0)
        recreated = (
            sweeps.default_saturation_search_config(worker_count=1)
            if expected_discs == 30
            else sweeps.default_relationship_search_config(worker_count=1)
        )
        assert _geometry_digest(search) == _geometry_digest(recreated)


def test_requested_plan_is_three_sequential_studies_not_a_cartesian_product() -> None:
    groups = sweeps.requested_relationship_points()
    assert tuple(groups) == (
        sweeps.RAW_STUDY_KEY,
        sweeps.EFFECTIVE_STUDY_KEY,
        sweeps.DETUNING_STUDY_KEY,
    )
    assert tuple(map(len, groups.values())) == (16, 16, 14)
    assert sum(map(len, groups.values())) == 46
    assert all(
        point.study_key == study_key
        for study_key, points in groups.items()
        for point in points
    )


def test_remaining_detuning_orchestration_reuses_15x15_geometry_and_counts_3150_searches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, object, str]] = []

    def fake_run(study_key, points, **kwargs):
        search = kwargs["search_config"]
        calls.append((study_key, len(points), search, _geometry_digest(search)))
        return [{"status": "completed"} for _ in points]

    monkeypatch.setattr(sweeps, "run_loading_relationship", fake_run)
    paths = sweeps.CampaignPaths(
        statistics=tmp_path / "statistics",
        figures=tmp_path / "figures",
    )
    metadata = sweeps.run_relationship_loading_campaign(
        worker_count=1,
        paths=paths,
        resume=False,
        selected_studies=(sweeps.DETUNING_STUDY_KEY,),
    )

    assert [(key, count) for key, count, _search, _digest in calls] == [
        (sweeps.DETUNING_STUDY_KEY, 14),
    ]
    searches = [search for _key, _count, search, _digest in calls]
    assert all(search == searches[0] for search in searches[1:])
    assert len({digest for _key, _count, _search, digest in calls}) == 1
    assert searches[0].phase_space == "full_sphere"
    assert searches[0].disc_count * searches[0].points_per_disc == 225
    assert metadata["loading_point_count_in_full_campaign"] == 14
    assert metadata["capture_threshold_search_count_in_full_campaign"] == 3_150
    assert metadata["combinatorial_product_used"] is False
    assert metadata["status"] == "completed"

    saved = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert saved["capture_threshold_search_count_in_full_campaign"] == 3_150
    assert saved["search_config"]["phase_space"] == "full_sphere"
    assert saved["search_config"]["disc_count"] == 15
    assert saved["search_config"]["points_per_disc"] == 15


def test_top_level_campaign_routes_only_remaining_stages_to_15x15_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []
    paths = sweeps.CampaignPaths(
        statistics=tmp_path / sweeps.REMAINING_CAMPAIGN_NAME / "statistics",
        figures=tmp_path / sweeps.REMAINING_CAMPAIGN_NAME / "figures",
    )

    def fake_loading(**kwargs):
        calls.append(("loading", kwargs))
        paths.statistics.mkdir(parents=True, exist_ok=True)
        paths.metadata_json.write_text(
            json.dumps({"status": "completed"}) + "\n",
            encoding="utf-8",
        )
        return {"status": "completed"}

    def fake_temperature(**kwargs):
        calls.append(("temperature", kwargs))
        return {
            "status": "completed",
            "completed_point_count": len(sweeps.DETUNING_N_VALUES),
            "outputs": {},
        }

    monkeypatch.setattr(sweeps, "run_relationship_loading_campaign", fake_loading)
    monkeypatch.setattr(sweeps, "run_relationship_temperature_campaign", fake_temperature)

    result = sweeps.run_relationship_campaign(worker_count=1, paths=paths)

    loading_kwargs = calls[0][1]
    temperature_kwargs = calls[1][1]
    assert loading_kwargs["selected_studies"] == (sweeps.DETUNING_STUDY_KEY,)
    assert loading_kwargs["paths"] == paths
    assert loading_kwargs["search_config"].disc_count == 15
    assert loading_kwargs["search_config"].points_per_disc == 15
    assert temperature_kwargs["paths"] == paths
    assert result["status"] == "completed"
    assert "30x30" in result["preserved_saturation_outputs"]["statistics_root"]


def test_relationship_plot_uses_saved_asymmetric_t95_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, np.ndarray] = {}
    original_errorbar = matplotlib.axes.Axes.errorbar

    def spy_errorbar(axis, x, y, *args, **kwargs):
        captured["x"] = np.asarray(x, dtype=float)
        captured["y"] = np.asarray(y, dtype=float)
        captured["yerr"] = np.asarray(kwargs["yerr"], dtype=float)
        return original_errorbar(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy_errorbar)
    rows = [
        {
            "s0": 2.0,
            "loading_rate_mean_atoms_per_s": 20.0e6,
            "loading_rate_t95_lower_atoms_per_s": 18.0e6,
            "loading_rate_t95_upper_atoms_per_s": 25.0e6,
        },
        {
            "s0": 1.0,
            "loading_rate_mean_atoms_per_s": 10.0e6,
            "loading_rate_t95_lower_atoms_per_s": 9.0e6,
            "loading_rate_t95_upper_atoms_per_s": 14.0e6,
        },
    ]
    destination = tmp_path / "loading_rate_vs_s0.png"

    assert sweeps.plot_loading_relationship(
        rows, sweeps.RAW_STUDY_KEY, destination
    ) == destination
    assert destination.is_file()
    assert destination.stat().st_size > 1_000
    assert captured["x"] == pytest.approx([1.0, 2.0])
    assert captured["y"] == pytest.approx([10.0, 20.0])
    assert captured["yerr"] == pytest.approx(
        np.asarray([[1.0, 2.0], [4.0, 5.0]])
    )


def test_temperature_plot_only_accepts_complete_14_point_15x15_products(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = sweeps.CampaignPaths(
        statistics=tmp_path / "statistics",
        figures=tmp_path / "figures",
    )
    summary_csv, metadata_json = _write_temperature_products(paths)
    plot_calls: list[dict[str, object]] = []

    def fake_plot(summary, destination, **kwargs):
        plot_calls.append(
            {
                "summary": summary,
                "destination": destination,
                **kwargs,
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
        return destination

    def production_run_forbidden(**_kwargs):
        raise AssertionError("plot-only validation must not launch temperature trajectories")

    monkeypatch.setattr(sweeps, "plot_temperature_vs_detuning", fake_plot)
    monkeypatch.setattr(
        sweeps,
        "run_temperature_detuning_sweep",
        production_run_forbidden,
    )

    result = sweeps.run_relationship_temperature_campaign(
        worker_count=1,
        paths=paths,
        plot_only=True,
    )

    assert result["status"] == "completed"
    assert result["completed_point_count"] == len(sweeps.DETUNING_N_VALUES) == 14
    assert len(plot_calls) == 2
    assert {call["include_survivor_panel"] for call in plot_calls} == {True, False}
    assert all(call["summary"] == summary_csv for call in plot_calls)
    assert all(
        call["cooling_power_w_per_beam"]
        == pytest.approx(sweeps.DEFAULT_COOLING_POWER_W_PER_BEAM)
        for call in plot_calls
    )
    assert all(
        call["ensemble_realization_count"] == sweeps.TEMPERATURE_ENSEMBLE_COUNT == 15
        for call in plot_calls
    )
    assert all(
        call["atoms_per_ensemble"] == sweeps.TEMPERATURE_ATOMS_PER_ENSEMBLE == 15
        for call in plot_calls
    )
    validated = sweeps._validate_temperature_campaign_products(
        summary_csv,
        metadata_json,
    )
    signature = validated["resume_signature"]
    assert signature["trajectory_count_per_point"] == 225
    assert signature["cooling_power_w_per_beam"] == pytest.approx(27.0e-3)
    assert signature["multilevel_config"]["repump_power_w_per_beam"] == pytest.approx(
        0.1e-3
    )
    assert signature["apparatus_config"]["repump"][
        "power_w_per_beam"
    ] == pytest.approx(0.1e-3)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("stale_10x25", "wrong cloud-row count"),
        ("incomplete", "not marked completed"),
        ("wrong_grid", "wrong detuning grid or order"),
    ),
)
def test_temperature_plot_only_rejects_stale_incomplete_or_wrong_grid_products(
    case: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = sweeps.CampaignPaths(
        statistics=tmp_path / case / "statistics",
        figures=tmp_path / case / "figures",
    )
    if case == "stale_10x25":
        _write_temperature_products(
            paths,
            ensemble_count=10,
            atoms_per_ensemble=25,
        )
    elif case == "incomplete":
        _write_temperature_products(paths, status="running")
    else:
        wrong_grid = tuple(reversed(sweeps.DETUNING_N_VALUES))
        _write_temperature_products(paths, detuning_values=wrong_grid)

    def plotting_forbidden(*_args, **_kwargs):
        raise AssertionError("invalid products must be rejected before plotting")

    def production_run_forbidden(**_kwargs):
        raise AssertionError("plot-only validation must not launch temperature trajectories")

    monkeypatch.setattr(sweeps, "plot_temperature_vs_detuning", plotting_forbidden)
    monkeypatch.setattr(
        sweeps,
        "run_temperature_detuning_sweep",
        production_run_forbidden,
    )

    with pytest.raises(ValueError, match=message):
        sweeps.run_relationship_temperature_campaign(
            worker_count=1,
            paths=paths,
            plot_only=True,
        )
