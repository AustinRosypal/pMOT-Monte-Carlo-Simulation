from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pmot.mot_simple import refined_relationship_campaign as campaign
from pmot.mot_simple import refined_relationship_campaign_v5_migration as migration


def _fake_source_hashes() -> dict[str, str]:
    return {
        "pmot/mot_simple/refined_relationship_campaign.py": "1" * 64,
        "pmot/mot_simple/timeout_audit.py": "2" * 64,
    }


def _source_point_metadata(point: campaign.RelationshipPoint) -> dict[str, object]:
    hashes = _fake_source_hashes()
    payload = migration._expected_source_point_payload(point, hashes)
    signature = migration._canonical_hash(payload)
    apparatus, simple, coil, _ = campaign.build_point_configuration(point)
    return {
        "schema_version": migration.SOURCE_POINT_SCHEMA_VERSION,
        "study_name": (
            f"{migration.SOURCE_CAMPAIGN_NAME}/{point.study_key}/{point.slug}"
        ),
        "run_signature_sha256": signature,
        "signature_payload": payload,
        "geometry_sha256": migration.SOURCE_GEOMETRY_SHA256,
        "relationship_point": payload["point"],
        "apparatus_config": payload["apparatus_config"],
        "simple_mot_config": payload["simple_mot_config"],
        "coil_config": payload["coil_config"],
        "capture_search_config": payload["search_config"],
        "endpoint_timeout_policy": payload["endpoint_timeout_policy"],
        "built_cooling_beam_count": 6,
        "built_cooling_beam_powers_w": [
            apparatus.cooling.power_w_per_beam
        ] * 6,
        "built_cooling_beam_detunings_hz": [simple.cooling_detuning_hz] * 6,
        "repumper_included": False,
    }


def test_source_payload_reconstruction_pins_schema5_policy(monkeypatch) -> None:
    hashes = _fake_source_hashes()
    monkeypatch.setattr(
        migration,
        "SOURCE_PHYSICS_HASH_MAP_SHA256",
        migration._canonical_hash(hashes),
    )
    point = campaign.requested_points()[campaign.DETUNING_STUDY_KEY][0]
    metadata = _source_point_metadata(point)

    signature, actual_hashes = migration._validate_source_point_signature(
        metadata, point
    )

    payload = metadata["signature_payload"]
    assert signature == metadata["run_signature_sha256"]
    assert actual_hashes == hashes
    assert payload["schema_version"] == 5
    assert payload["endpoint_timeout_policy"] == dict(
        migration.SOURCE_ENDPOINT_TIMEOUT_POLICY
    )
    assert len(payload["endpoint_timeout_policy"]["adaptive_node_levels"]) == 3
    assert len(
        payload["endpoint_timeout_policy"][
            "complete_positive_boundary_search_levels"
        ]
    ) == 2


def test_source_signature_rejects_policy_tamper(monkeypatch) -> None:
    hashes = _fake_source_hashes()
    monkeypatch.setattr(
        migration,
        "SOURCE_PHYSICS_HASH_MAP_SHA256",
        migration._canonical_hash(hashes),
    )
    point = campaign.requested_points()[campaign.DETUNING_STUDY_KEY][0]
    metadata = _source_point_metadata(point)
    payload = dict(metadata["signature_payload"])
    policy = dict(payload["endpoint_timeout_policy"])
    policy["adaptive_node_levels"] = [*policy["adaptive_node_levels"], {}]
    payload["endpoint_timeout_policy"] = policy
    metadata["signature_payload"] = payload
    metadata["run_signature_sha256"] = migration._canonical_hash(payload)
    metadata["endpoint_timeout_policy"] = policy

    with pytest.raises(ValueError, match="signed contract mismatch"):
        migration._validate_source_point_signature(metadata, point)


def test_destination_payload_accepts_canonical_json_tuple_list_round_trip() -> None:
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    expected = campaign._point_signature_payload(
        point,
        campaign.default_search_config(),
        migration.SOURCE_GEOMETRY_SHA256,
    )
    persisted = json.loads(json.dumps(expected, sort_keys=True))
    signature = migration._canonical_hash(expected)

    assert persisted != expected
    migration._validate_destination_point_payload(
        persisted,
        expected,
        signature,
        point_slug=point.slug,
    )

    persisted["simple_mot_config"]["cooling_detuning_hz"] = -1.0
    with pytest.raises(ValueError, match="destination point payload mismatch"):
        migration._validate_destination_point_payload(
            persisted,
            expected,
            signature,
            point_slug=point.slug,
        )


def test_append_migration_history_preserves_prior_record() -> None:
    prior = {
        "schema_version": 1,
        "source_campaign_name": "schema4-source",
        "reason": "prior migration",
    }
    latest = {
        "schema_version": 2,
        "source_campaign_name": migration.SOURCE_CAMPAIGN_NAME,
        "reason": "new migration",
    }
    payload: dict[str, object] = {"migration": dict(prior)}

    migration._append_migration_history(payload, latest)

    assert payload["migration"] == latest
    assert payload["migration_history"] == [prior, latest]
    migration._append_migration_history(payload, {**latest, "scope": "second"})
    assert payload["migration_history"][0] == prior
    assert payload["migration_history"][1] == latest


def test_point_migration_record_pins_source_point_signature() -> None:
    signature = "a" * 64

    record = migration._migration_record(
        migrated_utc="2026-09-08T00:00:00+00:00",
        scope="point",
        source_point_signature_sha256=signature,
    )

    assert record["source_campaign_name"] == migration.SOURCE_CAMPAIGN_NAME
    assert (
        record["source_campaign_signature_sha256"]
        == migration.SOURCE_CAMPAIGN_SIGNATURE_SHA256
    )
    assert record["source_point_signature_sha256"] == signature


def _diagnostic_payload() -> dict[str, object]:
    coarse_position = np.asarray([0.0107, -0.0087, -0.01424])
    fine_position = coarse_position + np.asarray([1.0e-14, -2.0e-14, 3.0e-14])
    coarse_velocity = np.asarray([-0.0014, 0.00011, 0.00384])
    fine_velocity = coarse_velocity + np.asarray([1.0e-12, -1.0e-12, 1.0e-12])
    minimum_radii = (0.0198493077452215, 0.0198493077452207)
    final_radii = (0.0198493077450271, 0.0198493077449886)
    pair = []
    for step, position, velocity, minimum_radius, final_radius in zip(
        (1.25e-6, 0.625e-6),
        (coarse_position, fine_position),
        (coarse_velocity, fine_velocity),
        minimum_radii,
        final_radii,
        strict=True,
    ):
        pair.append(
            {
                "duration_s": 2.0,
                "time_step_s": step,
                "termination_reason": "timeout",
                "trapped": False,
                "entered_trap_core": False,
                "core_entry_count": 0,
                "elapsed_time_s": 2.0,
                "minimum_radius_m": minimum_radius,
                "final_radius_m": final_radius,
                "final_position_m": position.tolist(),
                "final_velocity_m_per_s": velocity.tolist(),
            }
        )
    return {
        "production_failure": {
            "source_campaign": migration.SOURCE_CAMPAIGN_NAME,
            "source_campaign_signature_sha256": (
                migration.SOURCE_CAMPAIGN_SIGNATURE_SHA256
            ),
            "point_status": "failed",
            "completed_sample_count": 615,
            "expected_sample_count": 625,
        },
        "ray": {
            "disc_index": 7,
            "point_index": 20,
            "incident_speed_m_per_s": 0.0,
            "geometry_seed": campaign.DEFAULT_SEED,
            "geometry_sha256": migration.SOURCE_GEOMETRY_SHA256,
            "initial_velocity_m_per_s": [0.0, 0.0, 0.0],
        },
        "physics_configuration": {
            "cooling_detuning_n": -4.25,
            "cooling_power_w_per_beam": 0.027,
            "gravity_enabled": True,
            "trap_core_radius_m": 0.002,
            "trapped_criterion": (
                "continuously inside the 2 mm-radius core for at least 5 ms OR "
                "two core entries with an intervening exit"
            ),
        },
        "two_second_convergence_pair": pair,
        "convergence": {
            "same_termination_reason": True,
            "both_finite": True,
            "both_timeout": True,
            "both_zero_core_entries": True,
            "position_difference_norm_m": float(
                np.linalg.norm(coarse_position - fine_position)
            ),
            "velocity_difference_norm_m_per_s": float(
                np.linalg.norm(coarse_velocity - fine_velocity)
            ),
            "minimum_radius_difference_m": abs(
                minimum_radii[0] - minimum_radii[1]
            ),
            "final_radius_difference_m": abs(final_radii[0] - final_radii[1]),
        },
        "conclusion": {
            "physical_capture_classification": "indeterminate",
            "loading_classification": "indeterminate_zero_flux",
            "requirements": ["Retain the exact quadrature anchor g(0)=0."],
        },
    }


def _write_diagnostic(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / migration.ZERO_FLUX_DIAGNOSTIC_DIRECTORY
    directory.mkdir(parents=True)
    readme = directory / "README.md"
    data = directory / "diagnostic.json"
    readme.write_text("diagnostic fixture\n", encoding="utf-8")
    data.write_text(
        json.dumps(_diagnostic_payload(), indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        migration,
        "ZERO_FLUX_DIAGNOSTIC_README_SHA256",
        migration._sha256_file(readme),
    )
    monkeypatch.setattr(
        migration,
        "ZERO_FLUX_DIAGNOSTIC_README_BYTES",
        readme.stat().st_size,
    )
    monkeypatch.setattr(
        migration,
        "ZERO_FLUX_DIAGNOSTIC_JSON_SHA256",
        migration._sha256_file(data),
    )
    monkeypatch.setattr(
        migration,
        "ZERO_FLUX_DIAGNOSTIC_JSON_BYTES",
        data.stat().st_size,
    )
    return data


def test_zero_flux_diagnostic_requires_converged_matched_timeouts(
    tmp_path: Path, monkeypatch
) -> None:
    _write_diagnostic(tmp_path, monkeypatch)

    migration._validate_zero_flux_diagnostic(tmp_path)


def test_zero_flux_diagnostic_rejects_relabelled_escape(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = _write_diagnostic(tmp_path, monkeypatch)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["two_second_convergence_pair"][1]["termination_reason"] = "escaped"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        migration,
        "ZERO_FLUX_DIAGNOSTIC_JSON_SHA256",
        migration._sha256_file(data_path),
    )

    with pytest.raises(ValueError, match="relabeled|entered"):
        migration._validate_zero_flux_diagnostic(tmp_path)


def test_transformation_whitelist_preserves_every_other_file() -> None:
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    verified = migration.VerifiedCampaign(
        points=(migration.VerifiedPoint(point, "a" * 64, "completed", 625),),
        source_physics_hashes={},
        source_file_hashes={},
        source_total_bytes=0,
    )
    changed = migration._expected_transformed_files(verified)
    unchanged = {
        "statistics/launch_geometry.csv": "a" * 64,
        "statistics/" + migration.PRIOR_MIGRATION_MANIFEST: (
            migration.PRIOR_MIGRATION_MANIFEST_SHA256
        ),
        "figures/example.png": "b" * 64,
    }
    source = {**unchanged, **{key: "c" * 64 for key in changed}}
    destination = {**unchanged, **{key: "d" * 64 for key in changed}}

    migration._validate_transformation_whitelist(source, destination, verified)
    destination["figures/example.png"] = "e" * 64
    with pytest.raises(ValueError, match="unexpected file set"):
        migration._validate_transformation_whitelist(source, destination, verified)


def test_manifest_records_partial_checkpoint_and_prior_history() -> None:
    points = campaign.requested_points()[campaign.DETUNING_STUDY_KEY]
    completed = migration.VerifiedPoint(points[0], "1" * 64, "completed", 625)
    partial = migration.VerifiedPoint(points[15], "2" * 64, "failed", 615)
    source_hashes = {
        "statistics/" + migration.PRIOR_MIGRATION_MANIFEST: (
            migration.PRIOR_MIGRATION_MANIFEST_SHA256
        )
    }
    verified = migration.VerifiedCampaign(
        points=(completed, partial),
        source_physics_hashes={},
        source_file_hashes=source_hashes,
        source_total_bytes=123,
    )
    signatures = {
        (completed.point.study_key, completed.point.point_index): "3" * 64,
        (partial.point.study_key, partial.point.point_index): "4" * 64,
    }

    # The active contract may still be schema 5 while the parent patch is in
    # flight, so isolate this unit from the active campaign signature.
    original = campaign._campaign_signature
    try:
        campaign._campaign_signature = lambda *_args, **_kwargs: "5" * 64
        manifest = migration._migration_manifest(
            verified, source_hashes, 456, signatures
        )
    finally:
        campaign._campaign_signature = original

    assert manifest["checkpoint_counts"] == {
        "point_directories": 2,
        "completed_points": 1,
        "partial_points": 1,
        "complete_samples": 625,
        "partial_samples": 615,
        "total_samples": 1240,
    }
    assert manifest["prior_migration"]["preserved_byte_for_byte"] is True
    assert manifest["zero_flux_policy_rationale"][
        "physical_capture_classification"
    ] == "indeterminate"


def test_active_destination_contract_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "CAMPAIGN_NAME", "wrong")
    with pytest.raises(RuntimeError, match="active campaign name"):
        migration._active_destination_contract()
