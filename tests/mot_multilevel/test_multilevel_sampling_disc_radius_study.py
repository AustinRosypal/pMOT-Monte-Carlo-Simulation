"""Checks for the full-sphere sampling-disc-radius production study."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from math import pi
from pathlib import Path

import numpy as np
import pytest

from pmot.mot_multilevel import sampling_disc_radius_study as study
from pmot.mot_multilevel.power_loading_study import (
    REPUMP_POWER_W_PER_BEAM,
    build_27mw_multilevel_configuration,
    configuration_invariance_audit,
    default_study_search_config,
    generate_study_geometry,
)
from pmot.mot.magnetic_fields import default_anti_helmholtz_config


def _fit_rows(rate_function, *, sem_atoms_per_s: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for radius_mm in study.SAMPLING_DISC_RADII_MM:
        rate = float(rate_function(radius_mm))
        rows.append(
            {
                "sampling_disc_radius_mm": radius_mm,
                "loading_rate_mean_atoms_per_s": rate,
                "loading_rate_disc_cluster_sem_atoms_per_s": sem_atoms_per_s,
                "loading_rate_t95_lower_atoms_per_s": max(
                    0.0, rate - 2.0 * sem_atoms_per_s
                ),
                "loading_rate_t95_upper_atoms_per_s": rate
                + 2.0 * sem_atoms_per_s,
            }
        )
    return rows


def _write_spectrum(path: Path, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "velocity_m_per_s",
                "capture_cross_section_m2",
                "capture_cross_section_t95_lower_m2",
                "capture_cross_section_t95_upper_m2",
            ),
        )
        writer.writeheader()
        for velocity, fraction in ((0.0, 1.0), (10.0, 0.7), (20.0, 0.2), (30.0, 0.0)):
            mean = scale * fraction
            writer.writerow(
                {
                    "velocity_m_per_s": velocity,
                    "capture_cross_section_m2": mean,
                    "capture_cross_section_t95_lower_m2": max(0.0, 0.9 * mean),
                    "capture_cross_section_t95_upper_m2": 1.1 * mean,
                }
            )


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_exact_production_design_defaults() -> None:
    assert study.SAMPLING_DISC_RADII_MM == (
        3.0,
        5.0,
        8.0,
        12.0,
        15.0,
        20.0,
        25.0,
        30.0,
    )
    assert study.DISC_COUNT == 100
    assert study.POINTS_PER_DISC == 100
    assert study.COOLING_POWER_W_PER_BEAM == pytest.approx(27.0e-3)
    assert REPUMP_POWER_W_PER_BEAM == pytest.approx(0.1e-3)

    baseline = default_study_search_config()
    search = study.search_config_for_radius(30.0)
    assert search.disc_radius_m == pytest.approx(30.0e-3)
    assert search.disc_count == 100
    assert search.points_per_disc == 100
    assert search.worker_count == study.DEFAULT_WORKER_COUNT == 24
    assert search.seed == study.BASE_RANDOM_SEED
    assert search.phase_space == "full_sphere"
    assert not search.include_center_point

    allowed_changes = {
        "disc_radius_m",
        "disc_count",
        "points_per_disc",
        "worker_count",
        "seed",
        "phase_space",
    }
    baseline_values = asdict(baseline)
    search_values = asdict(search)
    for name in baseline_values.keys() - allowed_changes:
        assert search_values[name] == baseline_values[name]

    for radius_mm in study.SAMPLING_DISC_RADII_MM:
        radius_search = study.search_config_for_radius(radius_mm)
        assert radius_search.disc_radius_m == pytest.approx(1.0e-3 * radius_mm)


def test_radius_search_invariance_audit_allows_only_sampling_and_execution() -> None:
    config, apparatus, _ = build_27mw_multilevel_configuration()
    search = study.search_config_for_radius(30.0)

    audit = configuration_invariance_audit(
        config,
        apparatus,
        default_anti_helmholtz_config(),
        search,
    )

    assert audit["capture_search_only_sampling_design_changed"] is True
    assert audit["capture_search_non_sampling_changed_fields"] == []
    assert set(audit["capture_search_changed_fields"]) == {
        "disc_count",
        "disc_radius_m",
        "include_center_point",
        "phase_space",
        "points_per_disc",
        "seed",
        "worker_count",
    }


def test_run_one_radius_passes_exact_production_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_geometry_plot(search, path, **kwargs):
        captured["geometry_search"] = search
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def fake_geometry_metrics(search):
        captured["metrics_search"] = search
        return {"phase_space": search.phase_space, "direction_count": search.disc_count}

    def fake_run(search, **kwargs):
        captured["run_search"] = search
        captured["run_kwargs"] = kwargs
        return {
            "sample_count": 10_000,
            "capture_velocity_mean_m_per_s": 20.0,
            "capture_velocity_min_m_per_s": 0.0,
            "capture_velocity_max_m_per_s": 30.0,
            "zero_capture_velocity_count": 3,
            "valid_bracket_count": 9_997,
            "loading_rate": {
                "loading_rate_mean_atoms_per_s": 5.0e7,
                "loading_rate_sample_std_atoms_per_s": 1.0e7,
                "loading_rate_disc_cluster_sem_atoms_per_s": 1.0e6,
                "loading_rate_t95_lower_atoms_per_s": 4.8e7,
                "loading_rate_t95_upper_atoms_per_s": 5.2e7,
                "student_t_critical_95": 1.9842169515,
            },
        }

    monkeypatch.setattr(
        study, "plot_full_sphere_geometry_with_cooling_beams", fake_geometry_plot
    )
    monkeypatch.setattr(study, "full_sphere_geometry_metrics", fake_geometry_metrics)
    monkeypatch.setattr(study, "run_power_loading_study", fake_run)

    paths = study.RadiusStudyPaths(
        statistics=tmp_path / "statistics",
        figures=tmp_path / "figures",
    )
    row = study.run_one_radius(12.0, paths, resume=False)

    search = captured["run_search"]
    assert search == captured["geometry_search"] == captured["metrics_search"]
    assert search.disc_count == 100
    assert search.points_per_disc == 100
    assert search.phase_space == "full_sphere"
    assert not search.include_center_point
    kwargs = captured["run_kwargs"]
    assert kwargs["cooling_power_w_per_beam"] == pytest.approx(27.0e-3)
    assert kwargs["repump_power_w_per_beam"] == pytest.approx(0.1e-3)
    assert kwargs["worker_count"] == 24
    assert kwargs["resume"] is False
    assert row["sample_count"] == 10_000
    assert row["phase_space"] == "full_sphere"


def test_full_sphere_geometry_metrics_cover_all_signs_and_use_random_points() -> None:
    search = study.search_config_for_radius(
        15.0,
        disc_count=100,
        points_per_disc=4,
        worker_count=1,
    )
    discs, points = generate_study_geometry(search)
    assert len(discs) == 100
    assert len(points) == 400
    assert all(0.0 < point.s_m < search.disc_radius_m for point in points)
    assert all(
        np.isclose(np.linalg.norm(disc.outward_unit_vector), 1.0) for disc in discs
    )

    metrics = study.full_sphere_geometry_metrics(search)
    assert metrics["direction_count"] == 100
    assert metrics["point_count"] == 400
    assert metrics["phase_space"] == "full_sphere"
    assert metrics["solid_angle_sr"] == pytest.approx(4.0 * pi)
    assert all(count > 0 for count in metrics["signed_octant_counts"])
    assert all(
        0 < count < 100 for count in metrics["positive_direction_counts"].values()
    )
    assert np.linalg.norm(metrics["mean_direction"]) < 0.2
    second_moment = np.asarray(metrics["direction_second_moment"], dtype=float)
    assert np.trace(second_moment) == pytest.approx(1.0)
    assert np.all(np.asarray(metrics["direction_second_moment_eigenvalues"]) > 0.2)
    assert metrics["normalized_disc_area_coordinate_mean"] == pytest.approx(
        0.5, abs=0.06
    )


def test_all_phase_radii_reuse_the_same_normalized_random_geometry() -> None:
    reference_search = study.search_config_for_radius(
        study.SAMPLING_DISC_RADII_MM[0],
        disc_count=12,
        points_per_disc=7,
        worker_count=1,
    )
    reference_discs, reference_points = generate_study_geometry(reference_search)
    for radius_mm in study.SAMPLING_DISC_RADII_MM[1:]:
        search = study.search_config_for_radius(
            radius_mm,
            disc_count=12,
            points_per_disc=7,
            worker_count=1,
        )
        discs, points = generate_study_geometry(search)
        assert [disc.theta_rad for disc in discs] == pytest.approx(
            [disc.theta_rad for disc in reference_discs]
        )
        assert [disc.phi_rad for disc in discs] == pytest.approx(
            [disc.phi_rad for disc in reference_discs]
        )
        assert [point.theta_prime_rad for point in points] == pytest.approx(
            [point.theta_prime_rad for point in reference_points]
        )
        assert [point.s_m / search.disc_radius_m for point in points] == pytest.approx(
            [
                point.s_m / reference_search.disc_radius_m
                for point in reference_points
            ]
        )


def test_30_mm_disc_has_one_quarter_of_points_outside_escape_sphere() -> None:
    assert study.initial_point_fraction_outside_escape_sphere(
        30.0e-3,
        15.0e-3,
        30.0e-3,
    ) == pytest.approx(0.25)
    assert study.initial_point_fraction_outside_escape_sphere(
        25.0e-3,
        15.0e-3,
        30.0e-3,
    ) == 0.0
    assert study.initial_point_fraction_outside_escape_sphere(
        30.0e-3,
        30.0e-3,
        30.0e-3,
    ) == 1.0


def test_synthetic_saturating_fit_selects_rounded_convergence_radius() -> None:
    asymptote = 1.0e8
    scale_mm = 8.0
    shape = 2.0
    rows = _fit_rows(
        lambda radius: asymptote
        * (1.0 - np.exp(-((radius / scale_mm) ** shape))),
        sem_atoms_per_s=1.0e6,
    )

    fit = study.fit_loading_rate_convergence(rows)

    expected_radius = scale_mm * np.sqrt(-np.log(0.05))
    assert fit["convergence_ascertainable"] is True
    assert fit["fallback_used"] is False
    assert fit["parameters"]["asymptote_atoms_per_s"] == pytest.approx(asymptote)
    assert fit["parameters"]["scale_mm"] == pytest.approx(scale_mm)
    assert fit["parameters"]["shape"] == pytest.approx(shape)
    assert fit["convergence_radius_mm"] == pytest.approx(expected_radius)
    assert fit["convergence_radius_upper_95_mm"] > expected_radius
    assert fit["selected_confirmation_radius_mm"] == np.ceil(
        fit["convergence_radius_upper_95_mm"]
    )


def test_poor_saturating_fit_fails_chi_squared_gate() -> None:
    rows = _fit_rows(
        lambda radius: 1.0e8 * (1.0 - np.exp(-((radius / 8.0) ** 2.0))),
        sem_atoms_per_s=1.0e4,
    )
    rows[2]["loading_rate_mean_atoms_per_s"] += 2.0e7
    rows[2]["loading_rate_t95_lower_atoms_per_s"] += 2.0e7
    rows[2]["loading_rate_t95_upper_atoms_per_s"] += 2.0e7

    fit = study.fit_loading_rate_convergence(rows)

    assert fit["goodness_of_fit_accepted"] is False
    assert fit["convergence_ascertainable"] is False
    assert fit["selected_confirmation_radius_mm"] == 12.0
    assert "chi-squared" in fit["fallback_reason"]


def test_paired_disc_covariance_is_finite_and_positive_definite(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for radius_index, radius_mm in enumerate(study.SAMPLING_DISC_RADII_MM):
        path = tmp_path / f"radius_{radius_mm:g}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("disc_index", "loading_rate_atoms_per_s"),
            )
            writer.writeheader()
            for disc_index in range(8):
                writer.writerow(
                    {
                        "disc_index": disc_index,
                        "loading_rate_atoms_per_s": (
                            1.0e7 * radius_index
                            + 2.0e5 * disc_index
                            + 1.0e4 * radius_index * disc_index**2
                        ),
                    }
                )
        rows.append({"loading_rate_by_disc_csv": str(path)})

    result = study._paired_disc_mean_covariance(rows)

    assert result is not None
    covariance, disc_count, regularization = result
    expected_radius_count = len(study.SAMPLING_DISC_RADII_MM)
    assert covariance.shape == (expected_radius_count, expected_radius_count)
    assert disc_count == 8
    assert regularization >= 0.0
    assert np.all(np.isfinite(covariance))
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_nonconvergent_synthetic_curve_falls_back_to_12_mm() -> None:
    rows = _fit_rows(
        lambda radius: 1.0e6 * radius,
        sem_atoms_per_s=1.0e4,
    )

    fit = study.fit_loading_rate_convergence(rows)

    assert fit["convergence_ascertainable"] is False
    assert fit["fallback_used"] is True
    assert fit["selected_confirmation_radius_mm"] == 12.0
    assert fit["fallback_reason"]


def test_nonfinite_fit_covariance_serializes_as_finite_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _fit_rows(
        lambda radius: 8.0e7 * (1.0 - np.exp(-((radius / 9.0) ** 2.0))),
        sem_atoms_per_s=1.0e6,
    )

    monkeypatch.setattr(
        study,
        "curve_fit",
        lambda *args, **kwargs: (
            np.asarray([8.0e7, 9.0, 2.0]),
            np.full((3, 3), np.inf),
        ),
    )
    fit = study.fit_loading_rate_convergence(rows)

    assert fit["convergence_ascertainable"] is False
    assert fit["selected_confirmation_radius_mm"] == 12.0
    json.dumps(fit, allow_nan=False)


def test_phase_one_fit_and_independent_confirmation_are_sequenced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[float, int, bool]] = []

    def fake_run_one(radius_mm, paths, *, seed, confirmation=False, **kwargs):
        del paths, kwargs
        calls.append((float(radius_mm), int(seed), bool(confirmation)))
        rate = 1.0e7 * float(radius_mm)
        return {
            "sampling_disc_radius_mm": float(radius_mm),
            "random_seed": int(seed),
            "disc_count": 100,
            "points_per_disc": 100,
            "sample_count": 10_000,
            "phase_space": "full_sphere",
            "cooling_power_w_per_beam": 27.0e-3,
            "repump_power_w_per_beam": 0.1e-3,
            "loading_rate_mean_atoms_per_s": rate,
            "loading_rate_sample_std_atoms_per_s": 2.0e6,
            "loading_rate_disc_cluster_sem_atoms_per_s": 2.0e5,
            "loading_rate_t95_lower_atoms_per_s": rate - 4.0e5,
            "loading_rate_t95_upper_atoms_per_s": rate + 4.0e5,
            "loading_rate_t95_half_width_atoms_per_s": 4.0e5,
            "student_t_critical_95": 1.984,
        }

    fit = {
        "model": "test",
        "weighted_by": "test",
        "convergence_fraction": 0.95,
        "convergence_ascertainable": True,
        "convergence_radius_mm": 19.2,
        "selected_confirmation_radius_mm": 20.0,
        "fallback_used": False,
        "fallback_reason": None,
    }

    def fake_plot(rows, *args):
        del rows
        path = Path(args[-1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(study, "run_one_radius", fake_run_one)
    monkeypatch.setattr(study, "fit_loading_rate_convergence", lambda rows: fit)
    monkeypatch.setattr(study, "plot_loading_rate_vs_radius", fake_plot)
    monkeypatch.setattr(study, "plot_combined_cross_sections", fake_plot)
    paths = study.RadiusStudyPaths(
        statistics=tmp_path / "statistics",
        figures=tmp_path / "figures",
    )

    result = study.run_sampling_disc_radius_study(paths=paths)

    radius_count = len(study.SAMPLING_DISC_RADII_MM)
    assert calls[:radius_count] == [
        (radius, study.BASE_RANDOM_SEED, False)
        for radius in study.SAMPLING_DISC_RADII_MM
    ]
    assert calls[radius_count] == (20.0, study.CONFIRMATION_RANDOM_SEED, True)
    assert result["confirmation"]["random_seed"] == study.CONFIRMATION_RANDOM_SEED
    assert paths.aggregate_csv.is_file()
    assert paths.convergence_json.is_file()
    assert paths.confirmation_json.is_file()
    assert paths.metadata_json.is_file()


def test_extension_cli_selects_only_requested_phase_radii() -> None:
    args = study.build_argument_parser().parse_args(
        ["--execute-radii-mm", "3", "8", "--skip-confirmation"]
    )
    assert args.execute_radii_mm == [3.0, 8.0]
    assert args.skip_confirmation is True


def test_plot_helpers_create_pngs_from_small_synthetic_inputs(tmp_path: Path) -> None:
    rows = _fit_rows(
        lambda radius: 8.0e7 * (1.0 - np.exp(-((radius / 9.0) ** 1.8))),
        sem_atoms_per_s=8.0e5,
    )
    fit = study.fit_loading_rate_convergence(rows)
    loading_plot = study.plot_loading_rate_vs_radius(
        rows,
        fit,
        tmp_path / "loading_rate_vs_radius.png",
    )
    _assert_png(loading_plot)

    spectrum_rows: list[dict[str, object]] = []
    for radius_mm in (5.0, 12.0, 30.0):
        spectrum_path = tmp_path / f"spectrum_{radius_mm:g}.csv"
        _write_spectrum(spectrum_path, pi * (1.0e-3 * radius_mm) ** 2)
        spectrum_rows.append(
            {
                "sampling_disc_radius_mm": radius_mm,
                "capture_cross_section_csv": str(spectrum_path),
            }
        )
    combined_plot = study.plot_combined_cross_sections(
        spectrum_rows,
        tmp_path / "combined_cross_sections.png",
    )
    _assert_png(combined_plot)

    geometry_search = study.search_config_for_radius(
        5.0,
        disc_count=8,
        points_per_disc=2,
        worker_count=1,
    )
    geometry_plot = study.plot_full_sphere_geometry_with_cooling_beams(
        geometry_search,
        tmp_path / "full_sphere_geometry.png",
    )
    _assert_png(geometry_plot)
