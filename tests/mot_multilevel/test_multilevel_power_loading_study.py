"""Checks for the fixed 27 mW cooling/0.1 mW repump loading runner."""

from __future__ import annotations

import json
from dataclasses import replace
from math import pi

import numpy as np
import pytest

from pmot.mot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel.power_loading_study import (
    COOLING_POWER_W_PER_BEAM,
    DISC_COUNT,
    POINTS_PER_DISC,
    REPUMP_POWER_W_PER_BEAM,
    STUDY_NAME,
    StudyPaths,
    analyze_completed_samples,
    build_27mw_multilevel_configuration,
    calculate_clustered_cross_section,
    calculate_disc_clustered_loading,
    configuration_invariance_audit,
    default_study_paths,
    default_study_search_config,
    effective_saturation_metrics,
    generate_study_geometry,
    geometry_rows,
    geometry_sha256,
    build_run_metadata,
    study_signature,
    study_signature_payload,
    validate_checkpoint_samples,
)
from pmot.mot_multilevel.rate_equations import build_rate_equation_model
from pmot.mot_simple.power_loading_study import (
    default_study_search_config as default_simple_power_search,
)
from pmot.mot_simple.power_loading_study import (
    generate_study_geometry as generate_simple_power_geometry,
)
from pmot.mot_simple.power_loading_study import geometry_rows as simple_geometry_rows
from pmot.mot_simple.power_loading_study import geometry_sha256 as simple_geometry_sha256
from pmot.mot_simple.sampling import CaptureVelocitySample, PointSample


def _sample(point: PointSample, capture_velocity: float) -> CaptureVelocitySample:
    lower_reason = "bounded_core_residence" if capture_velocity > 0.0 else "timeout"
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
        capture_velocity_m_per_s=capture_velocity,
        velocity_resolution_m_per_s=0.125,
        trapped_velocity_lower_m_per_s=capture_velocity,
        untrapped_velocity_upper_m_per_s=capture_velocity + 0.125,
        lower_classification=lower_reason,
        upper_classification="escaped",
        lower_entered_trap_core=capture_velocity > 0.0,
        upper_entered_trap_core=False,
        lower_core_entry_count=1 if capture_velocity > 0.0 else 0,
        upper_core_entry_count=0,
    )


def test_default_study_is_50_by_50_random_points() -> None:
    search = default_study_search_config()
    assert search.disc_count == DISC_COUNT == 50
    assert search.points_per_disc == POINTS_PER_DISC == 50
    assert search.seed == 0
    assert not search.include_center_point
    discs, points = generate_study_geometry(search)
    assert len(discs) == 50
    assert len(points) == 2500
    assert all(0.0 < point.s_m < search.disc_radius_m for point in points)


def test_geometry_exactly_matches_completed_simple_power_study() -> None:
    multilevel_discs, multilevel_points = generate_study_geometry(
        default_study_search_config()
    )
    simple_discs, simple_points = generate_simple_power_geometry(
        default_simple_power_search()
    )
    assert geometry_sha256(geometry_rows(multilevel_discs, multilevel_points)) == (
        simple_geometry_sha256(simple_geometry_rows(simple_discs, simple_points))
    )


def test_cooling_is_27_mw_repump_is_baseline_0p1_mw_and_model_has_24_states() -> None:
    config, apparatus, beams = build_27mw_multilevel_configuration()
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    assert config.repumper_enabled
    assert model.state_count == 24
    assert (model.ground_count, model.excited_count) == (8, 16)
    assert len(cooling) == len(repump) == 6
    assert apparatus.cooling.power_w_per_beam == COOLING_POWER_W_PER_BEAM == 0.027
    assert apparatus.repump.power_w_per_beam == REPUMP_POWER_W_PER_BEAM == 0.0001
    assert config.repump_power_w_per_beam == 0.0001
    assert all(np.isclose(beam.power_w, 0.027) for beam in cooling)
    assert all(np.isclose(beam.power_w, 0.0001) for beam in repump)
    assert all(np.isclose(beam.detuning_hz, -15.0e6) for beam in cooling)
    assert all(np.isclose(beam.detuning_hz, 0.0) for beam in repump)
    assert all(np.isclose(2.0 * beam.beam_radius_m, 12.7e-3) for beam in beams)


def test_effective_saturation_distinguishes_cooling_from_resonant_repump() -> None:
    config, apparatus, _ = build_27mw_multilevel_configuration()
    metrics = effective_saturation_metrics(config, apparatus)
    assert np.isclose(
        metrics["cooling"]["beam_center_effective_saturation_parameter"],
        1.0044996392124688,
    )
    assert np.isclose(
        metrics["repump"]["beam_center_effective_saturation_parameter"],
        0.09459677610017904,
    )


def test_only_requested_power_and_sampling_fields_change() -> None:
    config, apparatus, _ = build_27mw_multilevel_configuration()
    audit = configuration_invariance_audit(
        config,
        apparatus,
        default_anti_helmholtz_config(),
        default_study_search_config(),
    )
    assert audit["apparatus_only_cooling_and_repump_power_changed"]
    assert audit["multilevel_only_repumper_enable_changed"]
    assert audit["repump_power_matches_authoritative_multilevel_default"] is True
    assert audit["coil_config_matches_default"]
    assert audit["capture_search_only_sampling_design_changed"]


def test_dedicated_paths_are_under_multilevel_outputs(tmp_path) -> None:
    paths = default_study_paths(tmp_path)
    assert paths.statistics == tmp_path / "outputs/statistics/mot_multilevel" / STUDY_NAME
    assert paths.figures == tmp_path / "outputs/figures/mot_multilevel" / STUDY_NAME


def test_signature_binds_powers_search_geometry_and_source_hashes() -> None:
    config, apparatus, _ = build_27mw_multilevel_configuration()
    search = default_study_search_config()
    payload = study_signature_payload(
        config,
        apparatus,
        default_anti_helmholtz_config(),
        search,
        "geometry",
        source_hashes={"test.py": "abc"},
    )
    baseline = study_signature(payload)
    changed = dict(payload)
    changed["geometry_sha256"] = "different"
    assert baseline != study_signature(changed)
    assert payload["multilevel_config"]["repump_power_w_per_beam"] == 0.0001
    assert payload["apparatus_config"]["cooling"]["power_w_per_beam"] == 0.027
    assert payload["apparatus_config"]["repump"]["power_w_per_beam"] == 0.0001


def test_run_metadata_is_json_serializable_and_audits_family_powers(tmp_path) -> None:
    config, apparatus, beams = build_27mw_multilevel_configuration()
    search = default_study_search_config()
    coil = default_anti_helmholtz_config()
    signature_payload = study_signature_payload(
        config,
        apparatus,
        coil,
        search,
        "geometry",
        source_hashes={"test.py": "abc"},
    )
    signature = study_signature(signature_payload)
    metadata = build_run_metadata(
        config,
        apparatus,
        beams,
        coil,
        search,
        signature_payload,
        signature,
        StudyPaths(tmp_path / "statistics", tmp_path / "figures"),
        worker_count=2,
        status="ready",
        completed_sample_count=0,
        started_utc="now",
        elapsed_wall_time_s=0.0,
    )
    json.dumps(metadata, allow_nan=False)
    assert metadata["cooling_power_w_per_beam"] == 0.027
    assert metadata["repump_power_w_per_beam"] == 0.0001
    assert metadata["all_built_beams_match_requested_family_powers"] is True


def test_clustered_cross_section_and_loading_use_direction_discs() -> None:
    search = replace(
        default_study_search_config(),
        disc_count=2,
        points_per_disc=2,
        analysis_velocity_max_m_per_s=5.0,
    )
    _, points = generate_study_geometry(search)
    samples = [
        _sample(points[0], 1.0),
        _sample(points[1], 3.0),
        _sample(points[2], 2.0),
        _sample(points[3], 4.0),
    ]
    velocity = np.asarray([0.0, 2.0, 5.0])
    spectrum = calculate_clustered_cross_section(samples, search, velocity)
    area = pi * search.disc_radius_m**2
    assert np.isclose(spectrum[0]["capture_cross_section_m2"], area)
    assert np.isclose(spectrum[1]["capture_cross_section_m2"], 0.75 * area)
    assert spectrum[1]["disc_count"] == 2
    assert spectrum[2]["capture_cross_section_m2"] == 0.0
    by_disc, loading = calculate_disc_clustered_loading(samples, search, spectrum)
    assert len(by_disc) == 2
    assert loading["disc_count"] == 2
    assert loading["raw_integral_units"] == "m^6/s^4"
    assert np.isclose(
        loading["loading_rate_mean_atoms_per_s"],
        loading["loading_rate_from_mean_spectrum_atoms_per_s"],
    )


def test_checkpoint_validation_rejects_geometry_or_endpoint_tampering() -> None:
    search = replace(default_study_search_config(), disc_count=1, points_per_disc=1)
    _, points = generate_study_geometry(search)
    valid = _sample(points[0], 2.0)
    assert validate_checkpoint_samples([valid], points) == {(0, 0): valid}
    with pytest.raises(ValueError, match="upper endpoint"):
        validate_checkpoint_samples(
            [replace(valid, upper_classification="bounded_core_residence")],
            points,
        )
    with pytest.raises(ValueError, match="geometry"):
        validate_checkpoint_samples([replace(valid, s_m=valid.s_m + 1.0e-3)], points)


def test_analysis_writes_three_plots_and_correct_loading_units(tmp_path) -> None:
    search = replace(
        default_study_search_config(),
        disc_count=2,
        points_per_disc=2,
        analysis_velocity_max_m_per_s=5.0,
    )
    _, points = generate_study_geometry(search)
    samples = [_sample(point, 1.0 + index) for index, point in enumerate(points)]
    paths = StudyPaths(tmp_path / "statistics", tmp_path / "figures")
    paths.statistics.mkdir()
    paths.figures.mkdir()
    # The production runner writes this before analysis.
    from pmot.mot_multilevel.power_loading_study import save_samples_atomic

    save_samples_atomic(paths.final_samples_csv, samples)
    summary = analyze_completed_samples(
        samples,
        search,
        paths,
        signature="signed",
        geometry_hash="geometry",
    )
    assert summary["sample_count"] == 4
    assert summary["valid_bracket_count"] == 4
    assert summary["loading_rate"]["raw_integral_units"] == "m^6/s^4"
    assert paths.cross_section_png.is_file()
    assert paths.impact_parameter_png.is_file()
    assert paths.loading_by_disc_png.is_file()
    assert paths.spectrum_csv.is_file()
    assert paths.loading_json.is_file()
