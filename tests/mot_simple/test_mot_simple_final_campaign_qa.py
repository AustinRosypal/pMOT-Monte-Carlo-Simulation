from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pmot.mot_simple import final_campaign_qa as qa
from pmot.mot_simple import refined_relationship_campaign as campaign
from pmot.mot_simple.timeout_audit import (
    VelocityResolvedCaptureOverride,
    velocity_overrides_to_payload,
)


def _direction_dependent_threshold_samples():
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(search)
    samples = []
    for point in points:
        threshold = 0.25 * (1 + point.disc_index % 6 + point.point_index % 3)
        samples.append(
            campaign.CaptureVelocitySample(
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
                upper_entered_trap_core=False,
                lower_core_entry_count=2,
                upper_core_entry_count=0,
            )
        )
    return samples


def test_final_qa_contract_has_exact_requested_grids_and_counts() -> None:
    assert len(qa.EXPECTED_RAW_SATURATION) == 24
    assert qa.EXPECTED_RAW_SATURATION[-8:] == (
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        125.0,
    )
    assert np.array_equal(
        qa.EXPECTED_EFFECTIVE_SATURATION,
        np.arange(0.25, 5.0 + 0.125, 0.25),
    )
    assert np.array_equal(
        qa.EXPECTED_LOADING_DETUNING_N,
        np.arange(-0.5, -6.0 - 0.125, -0.25),
    )
    assert np.array_equal(
        qa.EXPECTED_FORCE_DETUNING_N,
        np.round(np.arange(-0.5, -6.0 - 0.025, -0.05), 12),
    )
    assert qa.EXPECTED_LOADING_POINT_COUNT == 24 + 20 + 23
    assert qa.EXPECTED_TOTAL_RAY_COUNT == 67 * 25 * 25 == 41_875
    assert len(qa.FINAL_FIGURE_RELATIVE_PATHS) == 5


def test_final_qa_search_contract_accepts_only_canonical_configuration() -> None:
    search = campaign.default_search_config()
    reconstructed = qa._search_from_metadata({"search_config": asdict(search)})
    assert reconstructed == search

    changed = replace(search, disc_count=24)
    with pytest.raises(qa.CampaignQAFailure, match="canonical 25x25"):
        qa._search_from_metadata({"search_config": asdict(changed)})


def test_signature_payload_accepts_json_round_trip_but_rejects_tamper() -> None:
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    expected = campaign._point_signature_payload(
        point, campaign.default_search_config(), qa.EXPECTED_GEOMETRY_SHA256
    )
    persisted = json.loads(json.dumps(expected, allow_nan=False))
    expected_signature = campaign._signature(expected)

    # Dataclass tuples become JSON arrays on disk, so raw Python equality is
    # too strict even though the canonical signed JSON is unchanged.
    assert persisted != expected
    assert campaign._signature(persisted) == expected_signature
    qa._validate_signature_payload(
        persisted,
        expected,
        expected_signature,
        label="test signature payload",
    )

    persisted["simple_mot_config"]["cooling_detuning_hz"] += 1.0
    with pytest.raises(qa.CampaignQAFailure, match="canonical signature differs"):
        qa._validate_signature_payload(
            persisted,
            expected,
            expected_signature,
            label="test signature payload",
        )


def test_point_identity_omits_points_segment_but_output_paths_include_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(campaign, "_git_provenance", lambda _root: {})
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    search = campaign.default_search_config()
    campaign_paths = campaign.CampaignPaths(
        tmp_path / "statistics", tmp_path / "figures"
    )
    point_paths = campaign_paths.point_paths(point)
    payload = campaign._point_signature_payload(
        point, search, qa.EXPECTED_GEOMETRY_SHA256
    )
    metadata = campaign._new_point_metadata(
        point,
        search,
        payload,
        campaign._signature(payload),
        qa.EXPECTED_GEOMETRY_SHA256,
        point_paths,
        worker_count=1,
        completed_sample_count=0,
    )

    identity_suffix = Path(campaign.CAMPAIGN_NAME) / point.study_key / point.slug
    output_suffix = Path(point.study_key) / "points" / point.slug
    qa._require_path_suffix(
        metadata["study_name"], identity_suffix, label="point identity"
    )
    qa._require_path_suffix(
        point_paths.statistics, output_suffix, label="point statistics path"
    )
    qa._require_path_suffix(
        point_paths.figures, output_suffix, label="point figures path"
    )
    with pytest.raises(qa.CampaignQAFailure, match="does not end with"):
        qa._require_path_suffix(
            metadata["study_name"], output_suffix, label="point identity"
        )


def test_independent_loading_reconstruction_matches_linearity() -> None:
    # A deliberately direction-dependent but monotone threshold field exercises
    # both the per-disc estimator and integration of the mean spectrum.
    samples = _direction_dependent_threshold_samples()
    overrides = velocity_overrides_to_payload([])
    spectrum, by_disc, loading = qa._reconstruct_loading(samples, overrides)
    assert len(spectrum) == 121
    assert len(by_disc) == 25
    assert loading["student_t_critical_95"] == qa.EXPECTED_T_CRITICAL_95_DF24
    assert loading["disc_count"] == 25
    assert loading["point_count"] == 625
    assert np.isclose(
        loading["loading_rate_mean_atoms_per_s"],
        loading["loading_rate_from_mean_spectrum_atoms_per_s"],
        rtol=2e-15,
        atol=1e-12,
    )


def test_independent_loading_reconstruction_uses_zero_flux_anchor_for_null_v0() -> None:
    samples = _direction_dependent_threshold_samples()
    # The scalar row is an explicit compatibility sentinel: the direct mask is
    # authoritative and its sole indeterminate value is exactly at zero speed.
    samples[0] = replace(
        samples[0],
        capture_velocity_m_per_s=0.0,
        trapped_velocity_lower_m_per_s=0.0,
        lower_classification="indeterminate_zero_flux",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )
    captured_positive = tuple(
        bool(velocity <= 1.0) for velocity in qa.EXPECTED_VELOCITY_GRID[1:]
    )
    tristate = VelocityResolvedCaptureOverride(
        disc_index=0,
        point_index=0,
        velocity_m_per_s=tuple(qa.EXPECTED_VELOCITY_GRID),
        captured=(None, *captured_positive),
    )
    definite = replace(tristate, captured=(False, *captured_positive))
    definite_samples = list(samples)
    definite_samples[0] = replace(
        definite_samples[0], lower_classification="escaped"
    )

    spectrum, by_disc, loading = qa._reconstruct_loading(
        samples, velocity_overrides_to_payload([tristate])
    )
    _, definite_by_disc, definite_loading = qa._reconstruct_loading(
        definite_samples, velocity_overrides_to_payload([definite])
    )

    assert len(spectrum) == len(qa.EXPECTED_VELOCITY_GRID) - 1
    assert spectrum[0]["velocity_m_per_s"] == 0.25
    assert all(row["velocity_m_per_s"] > 0.0 for row in spectrum)
    assert loading["capture_spectrum_row_count"] == 120
    # The physical quadrature still begins at the exact, analytic g(0)=0
    # anchor, even though no capture cross section is invented there.
    assert loading["velocity_grid_sample_count"] == 121
    assert loading["indeterminate_zero_flux_ray_count"] == 1
    assert loading["zero_flux_quadrature_anchor_used"] is True
    assert loading["zero_speed_cross_section_imputed"] is False
    assert np.allclose(
        [row["loading_rate_atoms_per_s"] for row in by_disc],
        [row["loading_rate_atoms_per_s"] for row in definite_by_disc],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.isclose(
        loading["loading_rate_mean_atoms_per_s"],
        definite_loading["loading_rate_mean_atoms_per_s"],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_independent_loading_reconstruction_rejects_null_at_positive_speed() -> None:
    samples = _direction_dependent_threshold_samples()
    malformed_mask: list[bool | None] = [True] * len(qa.EXPECTED_VELOCITY_GRID)
    malformed_mask[1] = None
    malformed_mask[-1] = False
    malformed = VelocityResolvedCaptureOverride(
        disc_index=0,
        point_index=0,
        velocity_m_per_s=tuple(qa.EXPECTED_VELOCITY_GRID),
        captured=tuple(malformed_mask),
    )

    with pytest.raises(qa.CampaignQAFailure, match="zero|positive|indeterminate"):
        qa._reconstruct_loading(samples, velocity_overrides_to_payload([malformed]))


def test_audit_counter_reconstruction_is_independent_and_complete() -> None:
    rows = {
        (0, 0): {
            "base_timeout_detected": "True",
            "adaptive_audit_level_count": "3",
            "zero_threshold_audit_status": (
                "velocity_resolved_capture_with_indeterminate_zero_flux"
            ),
            "timeout_resolution_status": "adaptive_dual_step_research_recovered_boundary",
            "positive_boundary_grid_audit_status": "not_applicable",
            "pre_adaptive_coarse_evaluation_timeout_count": "0",
            "pre_adaptive_fine_evaluation_timeout_count": "0",
        },
        (0, 1): {
            "base_timeout_detected": "False",
            "adaptive_audit_level_count": "0",
            "zero_threshold_audit_status": "not_applicable",
            "timeout_resolution_status": "complete_scalar_diagnostics_and_velocity_grid_recovered_capture",
            "positive_boundary_grid_audit_status": "velocity_resolved_capture",
            "pre_adaptive_coarse_evaluation_timeout_count": "1",
            "pre_adaptive_fine_evaluation_timeout_count": "0",
        },
    }

    assert qa._reconstruct_audit_counts(rows, 2) == {
        "automatically_extended_timeout_ray_count": 1,
        "adaptively_extended_velocity_grid_ray_count": 1,
        "adaptively_extended_audit_ray_count": 1,
        "adaptively_extended_positive_boundary_ray_count": 1,
        "complete_longer_boundary_fallback_ray_count": 1,
        "positive_boundary_grid_override_count": 1,
        "pre_adaptive_positive_research_timeout_ray_count": 1,
        "velocity_resolved_override_count": 2,
        "indeterminate_zero_flux_ray_count": 1,
    }


def test_png_integrity_check_rejects_low_resolution_and_accepts_qa_resolution(
    tmp_path: Path,
) -> None:
    def valid_png(path: Path, width: int, height: int) -> None:
        pixels = np.random.default_rng(20260908).integers(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        Image.fromarray(pixels, mode="RGB").save(path, format="PNG")

    accepted = tmp_path / "accepted.png"
    valid_png(accepted, 1800, 1200)
    assert qa._png_dimensions(accepted) == (1800, 1200)

    rejected = tmp_path / "rejected.png"
    valid_png(rejected, 800, 600)
    with pytest.raises(qa.CampaignQAFailure, match="resolution is too low"):
        qa._png_dimensions(rejected)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(accepted.read_bytes()[:100_000])
    with pytest.raises(qa.CampaignQAFailure, match="corrupt or truncated"):
        qa._png_dimensions(corrupt)


def test_json_reader_fails_closed_on_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps({"value": float("nan")}), encoding="utf-8")
    with pytest.raises(qa.CampaignQAFailure, match="nonfinite JSON"):
        qa._read_json(path)
