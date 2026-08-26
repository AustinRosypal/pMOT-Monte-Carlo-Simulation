from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import pi

import numpy as np
import pytest

from pmot.mot_simple import power_loading_study as study
from pmot.mot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_simple.power_loading_study import COOLING_POWER_W_PER_BEAM
from pmot.mot_simple.power_loading_study import DEFAULT_WORKER_COUNT
from pmot.mot_simple.power_loading_study import DISC_COUNT
from pmot.mot_simple.power_loading_study import POINTS_PER_DISC
from pmot.mot_simple.power_loading_study import RANDOM_SEED
from pmot.mot_simple.power_loading_study import STUDY_NAME
from pmot.mot_simple.power_loading_study import build_27mw_apparatus
from pmot.mot_simple.power_loading_study import build_argument_parser
from pmot.mot_simple.power_loading_study import calculate_clustered_cross_section
from pmot.mot_simple.power_loading_study import calculate_disc_clustered_loading
from pmot.mot_simple.power_loading_study import configuration_invariance_audit
from pmot.mot_simple.power_loading_study import default_study_paths
from pmot.mot_simple.power_loading_study import default_study_search_config
from pmot.mot_simple.power_loading_study import effective_saturation_metrics
from pmot.mot_simple.power_loading_study import generate_study_geometry
from pmot.mot_simple.power_loading_study import geometry_rows
from pmot.mot_simple.power_loading_study import geometry_sha256
from pmot.mot_simple.power_loading_study import study_signature
from pmot.mot_simple.power_loading_study import study_signature_payload
from pmot.mot_simple.sampling import CaptureVelocitySample


def _capture_sample(disc_index: int, point_index: int, threshold: float) -> CaptureVelocitySample:
    return CaptureVelocitySample(
        disc_index=disc_index,
        point_index=point_index,
        theta_rad=0.1 * (disc_index + 1),
        phi_rad=0.2 * (disc_index + 1),
        theta_prime_rad=0.3 * (point_index + 1),
        s_m=1.0e-3 * (point_index + 1),
        radial_distance_m=15.0e-3,
        initial_position_m=(15.0e-3, 0.0, 0.0),
        incident_unit_vector=(-1.0, 0.0, 0.0),
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


def test_production_defaults_change_only_requested_power_and_sample_counts() -> None:
    apparatus = build_27mw_apparatus()
    baseline = default_simple_mot_apparatus()
    simple = default_simple_mot_config()
    coil = default_anti_helmholtz_config()
    search = default_study_search_config()

    assert apparatus.cooling.power_w_per_beam == pytest.approx(27.0e-3)
    assert apparatus.repump == baseline.repump
    assert replace(apparatus, cooling=replace(apparatus.cooling, power_w_per_beam=20.0e-3)) == baseline
    assert search.disc_count == DISC_COUNT == 50
    assert search.points_per_disc == POINTS_PER_DISC == 50
    assert search.seed == RANDOM_SEED == 0
    assert search.bounded_core_residence_s == pytest.approx(5.0e-3)
    assert not search.include_center_point

    audit = configuration_invariance_audit(apparatus, simple, coil, search)
    assert audit["apparatus_only_cooling_power_changed"]
    assert audit["simple_config_matches_default"]
    assert audit["coil_config_matches_default"]
    assert audit["search_only_requested_counts_changed"]


def test_effective_saturation_and_cli_defaults() -> None:
    metrics = effective_saturation_metrics()
    assert metrics["beam_center_peak_intensity_w_per_m2"] == pytest.approx(426.2814521402368)
    assert metrics["beam_center_on_resonance_saturation_parameter"] == pytest.approx(25.54112954704834)
    assert metrics["beam_center_effective_saturation_parameter"] == pytest.approx(1.0029104151628667)

    args = build_argument_parser().parse_args([])
    assert args.workers == DEFAULT_WORKER_COUNT == 20
    assert args.resume
    assert not args.analyze_only
    assert args.output_dir is None
    assert args.figures_dir is None


def test_default_paths_and_seeded_geometry_signature_are_reproducible(tmp_path) -> None:
    paths = default_study_paths(tmp_path)
    assert paths.statistics == tmp_path / "outputs" / "statistics" / "mot_simple" / STUDY_NAME
    assert paths.figures == tmp_path / "outputs" / "figures" / "mot_simple" / STUDY_NAME

    search = replace(default_study_search_config(), disc_count=2, points_per_disc=3)
    discs_a, points_a = generate_study_geometry(search)
    discs_b, points_b = generate_study_geometry(search)
    rows_a = geometry_rows(discs_a, points_a)
    rows_b = geometry_rows(discs_b, points_b)
    assert rows_a == rows_b
    assert len(rows_a) == 6
    assert all(0.0 < float(row["s_m"]) < search.disc_radius_m for row in rows_a)
    geometry_hash = geometry_sha256(rows_a)
    assert geometry_hash == geometry_sha256(rows_b)

    apparatus = build_27mw_apparatus()
    payload = study_signature_payload(
        apparatus,
        default_simple_mot_config(),
        default_anti_helmholtz_config(),
        search,
        geometry_hash,
        source_hashes={"unit-test": "fixed"},
    )
    signature = study_signature(payload)
    assert signature == study_signature(payload)
    changed = dict(payload)
    changed["geometry_sha256"] = "different"
    assert signature != study_signature(changed)


def test_clustered_cross_section_uses_disc_clusters_and_clips_interval() -> None:
    samples = [
        _capture_sample(0, 0, 1.0),
        _capture_sample(0, 1, 3.0),
        _capture_sample(1, 0, 2.0),
        _capture_sample(1, 1, 4.0),
    ]
    search = replace(
        default_study_search_config(),
        disc_count=2,
        points_per_disc=2,
        disc_radius_m=1.0,
    )
    rows = calculate_clustered_cross_section(
        samples,
        search,
        velocity_grid_m_per_s=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert [row["captured_count"] for row in rows] == [4, 3, 2, 1]
    assert [row["capture_fraction"] for row in rows] == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert [row["capture_cross_section_m2"] for row in rows] == pytest.approx(
        [pi, 0.75 * pi, 0.5 * pi, 0.25 * pi]
    )
    assert rows[0]["capture_cross_section_disc_cluster_sem_m2"] == pytest.approx(0.0)
    assert rows[1]["capture_cross_section_disc_cluster_sem_m2"] > 0.0
    assert all(0.0 <= row["capture_cross_section_t95_lower_m2"] <= pi for row in rows)
    assert all(0.0 <= row["capture_cross_section_t95_upper_m2"] <= pi for row in rows)


def test_loading_rate_is_mean_of_disc_integrals_with_clustered_uncertainty() -> None:
    samples = [
        _capture_sample(0, 0, 1.0),
        _capture_sample(0, 1, 3.0),
        _capture_sample(1, 0, 2.0),
        _capture_sample(1, 1, 4.0),
    ]
    search = replace(
        default_study_search_config(),
        disc_count=2,
        points_per_disc=2,
        disc_radius_m=1.0,
    )
    spectrum = calculate_clustered_cross_section(
        samples,
        search,
        velocity_grid_m_per_s=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    by_disc, summary = calculate_disc_clustered_loading(samples, search, spectrum)
    rates = np.asarray([row["loading_rate_atoms_per_s"] for row in by_disc])
    assert len(by_disc) == 2
    assert summary["loading_rate_mean_atoms_per_s"] == pytest.approx(float(np.mean(rates)))
    assert summary["loading_rate_from_mean_spectrum_atoms_per_s"] == pytest.approx(float(np.mean(rates)))
    assert summary["loading_rate_disc_cluster_sem_atoms_per_s"] == pytest.approx(
        float(np.std(rates, ddof=1) / np.sqrt(2.0))
    )
    assert summary["loading_rate_t95_lower_atoms_per_s"] >= 0.0
    assert summary["loading_rate_t95_upper_atoms_per_s"] >= summary["loading_rate_mean_atoms_per_s"]
    assert summary["raw_integral_units"] == "m^6/s^4"
    assert asdict(default_simple_mot_config())["cooling_detuning_hz"] == -15.0e6
    assert COOLING_POWER_W_PER_BEAM == pytest.approx(27.0e-3)


def test_mocked_end_to_end_run_writes_completed_resumable_products(tmp_path, monkeypatch) -> None:
    search = replace(default_study_search_config(), disc_count=2, points_per_disc=2)
    statistics = tmp_path / "statistics"
    figures = tmp_path / "figures"

    def fake_capture(point):
        threshold = 4.0 + point.disc_index + 0.25 * point.point_index
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
            lower_classification="bounded_core_residence",
            upper_classification="escaped",
            lower_entered_trap_core=True,
            upper_entered_trap_core=True,
            lower_core_entry_count=1,
            upper_core_entry_count=1,
        )

    monkeypatch.setattr(study, "_capture_worker", fake_capture)
    summary = study.run_power_loading_study(
        search,
        worker_count=1,
        output_directory=statistics,
        figure_directory=figures,
        resume=False,
    )
    assert summary["sample_count"] == 4
    metadata = json.loads((statistics / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["completed_sample_count"] == 4
    for filename in (
        "launch_geometry.csv",
        "capture_velocity_partial_samples.csv",
        "capture_velocity_samples.csv",
        "capture_velocity_summary.json",
        "capture_velocity_spectrum.csv",
        "loading_rate_by_disc.csv",
        "loading_rate_result.json",
    ):
        assert (statistics / filename).is_file()
    for filename in (
        "capture_cross_section_vs_velocity.png",
        "capture_velocity_vs_impact_parameter.png",
        "loading_rate_by_disc.png",
    ):
        assert (figures / filename).is_file()

    resumed = study.run_power_loading_study(
        search,
        worker_count=1,
        output_directory=statistics,
        figure_directory=figures,
        resume=True,
        analyze_only=True,
    )
    assert resumed["run_signature_sha256"] == summary["run_signature_sha256"]
