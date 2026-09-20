from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from pmot.mot_simple import refined_relationship_campaign as campaign
from pmot.mot_simple import refined_relationship_campaign_migration as migration


def _fixture_point_metadata(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, object]:
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    source_hashes = {"pmot/example.py": "0" * 64}
    monkeypatch.setattr(
        migration,
        "SOURCE_PHYSICS_HASH_MAP_SHA256",
        migration._canonical_hash(source_hashes),
    )
    payload = migration._expected_source_point_payload(point, source_hashes)
    metadata = {
        "schema_version": migration.SOURCE_POINT_SCHEMA_VERSION,
        "study_name": (
            f"{migration.SOURCE_CAMPAIGN_NAME}/{point.study_key}/{point.slug}"
        ),
        "run_signature_sha256": migration._canonical_hash(payload),
        "signature_payload": payload,
        "geometry_sha256": migration.SOURCE_GEOMETRY_SHA256,
        "relationship_point": asdict(point),
        "apparatus_config": deepcopy(payload["apparatus_config"]),
        "simple_mot_config": deepcopy(payload["simple_mot_config"]),
        "coil_config": deepcopy(payload["coil_config"]),
        "capture_search_config": deepcopy(payload["search_config"]),
        "endpoint_timeout_policy": deepcopy(payload["endpoint_timeout_policy"]),
        "model": payload["model"],
        "phase_space": payload["phase_space"],
        "built_cooling_beam_count": 6,
        "built_cooling_beam_powers_w": [point.cooling_power_w_per_beam] * 6,
        "built_cooling_beam_detunings_hz": [point.cooling_detuning_hz] * 6,
        "repumper_included": False,
    }
    return metadata, point


def test_source_point_signature_accepts_exact_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, point = _fixture_point_metadata(monkeypatch)

    signature, source_hashes = migration._validate_source_point_signature(
        metadata, point
    )

    assert signature == metadata["run_signature_sha256"]
    assert source_hashes == {"pmot/example.py": "0" * 64}


def test_source_point_signature_refuses_payload_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, point = _fixture_point_metadata(monkeypatch)
    metadata["signature_payload"]["phase_space"] = "octant"

    with pytest.raises(ValueError, match="signature payload was tampered"):
        migration._validate_source_point_signature(metadata, point)


@pytest.mark.parametrize(
    "tamper",
    (
        lambda payload: payload["apparatus_config"]["cooling"].update(
            {"beam_diameter_m": 0.02}
        ),
        lambda payload: payload["endpoint_timeout_policy"].update(
            {"audit_duration_s": 0.19}
        ),
    ),
    ids=("apparatus", "endpoint-policy"),
)
def test_source_point_refuses_resigned_physics_or_policy_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tamper,
) -> None:
    metadata, point = _fixture_point_metadata(monkeypatch)
    tamper(metadata["signature_payload"])
    # An unkeyed digest alone cannot prevent tamper-and-rehash.  The exact
    # independently reconstructed schema-4 payload must still reject it.
    metadata["run_signature_sha256"] = migration._canonical_hash(
        metadata["signature_payload"]
    )

    with pytest.raises(ValueError, match="physics/configuration payload mismatch"):
        migration._validate_source_point_signature(metadata, point)


def test_source_point_refuses_top_level_physics_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, point = _fixture_point_metadata(monkeypatch)
    metadata["apparatus_config"]["cooling"]["beam_diameter_m"] = 0.02

    with pytest.raises(ValueError, match="top-level apparatus_config disagrees"):
        migration._validate_source_point_signature(metadata, point)


def test_source_campaign_identity_refuses_signature_tamper() -> None:
    metadata = {
        "schema_version": migration.SOURCE_CAMPAIGN_SCHEMA_VERSION,
        "campaign_name": migration.SOURCE_CAMPAIGN_NAME,
        "campaign_signature_sha256": "f" * 64,
        "common_geometry_sha256": migration.SOURCE_GEOMETRY_SHA256,
        "loading_point_count": 67,
        "capture_threshold_search_count": 41_875,
    }

    with pytest.raises(ValueError, match="source campaign signature mismatch"):
        migration._validate_source_campaign_identity(metadata)


def test_delayed_capture_diagnostic_is_hash_and_semantics_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / migration.DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY
    directory.mkdir(parents=True)
    readme = directory / "README.md"
    data_path = directory / "diagnostic.json"
    readme.write_text("validated delayed capture\n", encoding="utf-8")
    data = {
        "ray": {
            "disc_index": 23,
            "point_index": 15,
            "geometry_seed": campaign.DEFAULT_SEED,
            "geometry_sha256": migration.SOURCE_GEOMETRY_SHA256,
        },
        "physics_configuration": {
            "cooling_detuning_n": -3.75,
            "cooling_power_w_per_beam": 0.027,
            "gravity_enabled": True,
            "trap_core_radius_m": 0.002,
            "trapped_criterion": (
                "continuously inside the 2 mm-radius core for at least 5 ms OR "
                "two core entries with an intervening exit"
            ),
        },
        "terminal_pair": [
            {
                "time_step_s": 1.25e-6,
                "termination_reason": "two_core_entries",
                "absolute_elapsed_time_s": 0.8064525,
            },
            {
                "time_step_s": 0.625e-6,
                "termination_reason": "two_core_entries",
                "absolute_elapsed_time_s": 0.806451875,
            },
        ],
        "conclusion": {
            "classification": "delayed capture",
            "minimum_demonstrated_terminal_time_s": 0.8064525,
        },
    }
    data_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        migration, "DELAYED_CAPTURE_README_SHA256", migration._sha256_file(readme)
    )
    monkeypatch.setattr(
        migration, "DELAYED_CAPTURE_JSON_SHA256", migration._sha256_file(data_path)
    )
    migration._validate_delayed_capture_diagnostic(tmp_path)

    data["terminal_pair"][1]["termination_reason"] = "timeout"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        migration, "DELAYED_CAPTURE_JSON_SHA256", migration._sha256_file(data_path)
    )
    with pytest.raises(ValueError, match="terminal classification mismatch"):
        migration._validate_delayed_capture_diagnostic(tmp_path)


def test_copy_and_destination_rewrite_leave_v3_bytes_unchanged(
    tmp_path: Path,
) -> None:
    source_statistics = tmp_path / "source" / "statistics"
    source_figures = tmp_path / "source" / "figures"
    source_statistics.mkdir(parents=True)
    source_figures.mkdir(parents=True)
    source_json = source_statistics / "metadata.json"
    source_binary = source_figures / "plot.png"
    source_json.write_text(
        json.dumps({"campaign": migration.SOURCE_CAMPAIGN_NAME}) + "\n",
        encoding="utf-8",
    )
    source_binary.write_bytes(b"\x89PNG\r\nfixture")
    before, _ = migration._hash_tree_pair(
        source_statistics, source_figures, label="test source"
    )

    staged_statistics = tmp_path / "stage" / "statistics"
    staged_figures = tmp_path / "stage" / "figures"
    migration._copy_tree_pair(
        source_statistics, source_figures, staged_statistics, staged_figures
    )
    migration._rewrite_json(
        staged_statistics / "metadata.json",
        ((migration.SOURCE_CAMPAIGN_NAME, campaign.CAMPAIGN_NAME),),
    )

    after, _ = migration._hash_tree_pair(
        source_statistics, source_figures, label="test source recheck"
    )
    copied, _ = migration._hash_tree_pair(
        staged_statistics, staged_figures, label="test destination"
    )
    assert after == before
    assert source_json.read_text(encoding="utf-8") == (
        json.dumps({"campaign": migration.SOURCE_CAMPAIGN_NAME}) + "\n"
    )
    assert copied["figures/plot.png"] == before["figures/plot.png"]
    assert copied["statistics/metadata.json"] != before["statistics/metadata.json"]
    assert hashlib.sha256(source_binary.read_bytes()).hexdigest() == before[
        "figures/plot.png"
    ]


def test_manifest_records_copy2_and_old_new_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, point = _fixture_point_metadata(monkeypatch)
    old_signature, source_hashes = migration._validate_source_point_signature(
        metadata, point
    )
    verified = migration.VerifiedCampaign(
        points=(
            migration.VerifiedPoint(
                point=point,
                source_signature_sha256=old_signature,
                status="completed",
                completed_sample_count=625,
            ),
        ),
        source_physics_hashes=source_hashes,
        source_file_hashes={"statistics/a.json": "a" * 64},
        source_total_bytes=10,
    )
    new_signature = "b" * 64

    manifest = migration._migration_manifest(
        verified,
        {"statistics/a.json": "c" * 64},
        11,
        {(point.study_key, point.point_index): new_signature},
    )

    assert manifest["copy_method"].startswith("copy2")
    assert manifest["point_signatures"][0]["source_signature_sha256"] == old_signature
    assert manifest["point_signatures"][0]["destination_signature_sha256"] == new_signature
    assert manifest["files"][0]["method"] == "copy2"
    assert manifest["files"][0]["transformed_after_copy"] is True


def test_end_to_end_staged_copy_resigns_and_commits_without_mutating_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = campaign.requested_points()[campaign.RAW_STUDY_KEY][0]
    old_signature = "a" * 64
    source_statistics = tmp_path / "statistics" / migration.SOURCE_CAMPAIGN_NAME
    source_figures = tmp_path / "figures" / migration.SOURCE_CAMPAIGN_NAME
    destination_statistics = tmp_path / "statistics" / campaign.CAMPAIGN_NAME
    destination_figures = tmp_path / "figures" / campaign.CAMPAIGN_NAME
    point_stats = source_statistics / point.study_key / "points" / point.slug
    point_figs = source_figures / point.study_key / "points" / point.slug
    point_stats.mkdir(parents=True)
    point_figs.mkdir(parents=True)
    (point_figs / "capture_cross_section_vs_velocity.png").write_bytes(b"figure")

    absolute_old = str(source_statistics.resolve())
    (point_stats / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "study_name": f"{migration.SOURCE_CAMPAIGN_NAME}/{point.study_key}/{point.slug}",
                "run_signature_sha256": old_signature,
                "signature_payload": {},
                "endpoint_timeout_policy": {},
                "outputs": {"statistics_directory": str(point_stats.resolve())},
                "loading_rate": {
                    "run_signature_sha256": old_signature,
                    "path": absolute_old,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in ("loading_rate_result.json", "capture_velocity_summary.json"):
        (point_stats / name).write_text(
            json.dumps(
                {
                    "run_signature_sha256": old_signature,
                    "path": absolute_old,
                }
            ),
            encoding="utf-8",
        )
    for study_key in campaign.STUDY_ORDER:
        study_stats = source_statistics / study_key
        study_stats.mkdir(parents=True, exist_ok=True)
        rows = []
        if study_key == point.study_key:
            row = {field: "" for field in campaign.AGGREGATE_FIELDNAMES}
            row.update(
                {
                    "point_index": "0",
                    "run_signature_sha256": old_signature,
                    "statistics_directory": absolute_old,
                    "figures_directory": str(source_figures.resolve()),
                    "capture_cross_section_csv": absolute_old + "\\spectrum.csv",
                    "capture_cross_section_plot": str(source_figures.resolve())
                    + "\\plot.png",
                }
            )
            rows.append(row)
        campaign._atomic_write_csv(
            study_stats / "aggregate.csv", rows, campaign.AGGREGATE_FIELDNAMES
        )
        campaign._atomic_write_json(
            study_stats / "sweep_metadata.json",
            {
                "schema_version": 4,
                "study_key": study_key,
                "aggregate_csv": str((study_stats / "aggregate.csv").resolve()),
            },
        )
    campaign._atomic_write_json(
        source_statistics / "campaign_metadata.json",
        {
            "schema_version": 4,
            "campaign_name": migration.SOURCE_CAMPAIGN_NAME,
            "campaign_signature_sha256": migration.SOURCE_CAMPAIGN_SIGNATURE_SHA256,
            "statistics_directory": absolute_old,
            "figures_directory": str(source_figures.resolve()),
        },
    )
    force_directory = source_statistics / "04_force_vs_detuning_27mW"
    force_directory.mkdir()
    campaign._atomic_write_json(
        force_directory / "force_vs_detuning_metadata.json",
        {"outputs": {"force_sweep_csv": absolute_old + "\\force.csv"}},
    )

    verified = migration.VerifiedCampaign(
        points=(
            migration.VerifiedPoint(
                point=point,
                source_signature_sha256=old_signature,
                status="completed",
                completed_sample_count=625,
            ),
        ),
        source_physics_hashes={},
        source_file_hashes={},
        source_total_bytes=0,
    )

    def accept_source(*args, source_file_hashes, source_total_bytes, **kwargs):
        return migration.VerifiedCampaign(
            points=verified.points,
            source_physics_hashes={},
            source_file_hashes=dict(source_file_hashes),
            source_total_bytes=source_total_bytes,
        )

    monkeypatch.setattr(migration, "validate_source_campaign", accept_source)
    monkeypatch.setattr(migration, "_validate_destination", lambda *args, **kwargs: None)
    source_hashes_before, _ = migration._hash_tree_pair(
        source_statistics, source_figures, label="fixture before"
    )

    manifest_path = migration.migrate_campaign(
        project_root=tmp_path,
        source_statistics=source_statistics,
        source_figures=source_figures,
        destination_statistics=destination_statistics,
        destination_figures=destination_figures,
    )

    source_hashes_after, _ = migration._hash_tree_pair(
        source_statistics, source_figures, label="fixture after"
    )
    assert source_hashes_after == source_hashes_before
    assert manifest_path.is_file()
    destination_metadata = json.loads(
        (destination_statistics / point.study_key / "points" / point.slug / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    expected_payload = campaign._point_signature_payload(
        point, campaign.default_search_config(), migration.SOURCE_GEOMETRY_SHA256
    )
    expected_signature = campaign._signature(expected_payload)
    assert destination_metadata["schema_version"] == campaign.POINT_SCHEMA_VERSION
    assert destination_metadata["run_signature_sha256"] == expected_signature
    assert campaign._signature(destination_metadata["signature_payload"]) == (
        campaign._signature(expected_payload)
    )
    assert destination_metadata["migration"]["source_campaign_name"] == (
        migration.SOURCE_CAMPAIGN_NAME
    )
    assert destination_metadata["loading_rate"]["run_signature_sha256"] == (
        expected_signature
    )
    assert migration._stale_v3_references(destination_metadata) == []
    result = json.loads(
        (
            destination_statistics
            / point.study_key
            / "points"
            / point.slug
            / "loading_rate_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["run_signature_sha256"] == expected_signature
    assert str(destination_statistics.resolve()) in result["path"]
    assert destination_figures.is_dir()


def test_final_root_validation_parses_json_paths_instead_of_raw_escaped_text(
    tmp_path: Path,
) -> None:
    final_statistics = tmp_path / "statistics" / campaign.CAMPAIGN_NAME
    final_figures = tmp_path / "figures" / campaign.CAMPAIGN_NAME
    metadata_path = tmp_path / "campaign_metadata.json"
    campaign._atomic_write_json(
        metadata_path,
        {
            "statistics_directory": str(final_statistics.resolve()),
            "figures_directory": str(final_figures.resolve()),
        },
    )

    migration._validate_serialized_final_roots(
        metadata_path, final_statistics, final_figures
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["statistics_directory"] = str((tmp_path / ".staging").resolve())
    campaign._atomic_write_json(metadata_path, payload)
    with pytest.raises(ValueError, match="statistics_directory"):
        migration._validate_serialized_final_roots(
            metadata_path, final_statistics, final_figures
        )
