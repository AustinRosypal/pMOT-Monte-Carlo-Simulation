"""Authenticated migration of the stopped schema-5 simple-MOT campaign.

The schema-5 campaign stopped at exactly zero incident speed for one ray at
``Delta/Gamma=-4.25``.  Independent 2 s integrations at 1.25 and 0.625 us
steps both remained finite, mutually converged, outside the core, and
non-terminal.  Schema 6 preserves that physical result as the explicit
``indeterminate_zero_flux`` state.  It is neither captured nor escaped; it can
nevertheless be excluded from the capture cross section at exactly zero speed
because the effusive loading integrand has the exact kinematic anchor
``g(0)=sigma(0)*0**3=0``.

This module is intentionally separate from the historical v3-to-v4 migrator.
It accepts one exact stopped v4 identity, validates every retained scientific
ledger, copies both output trees with :func:`shutil.copy2`, re-signs only the
active schema-coupled metadata, preserves the prior v3-to-v4 manifest and
provenance byte-for-byte, and atomically publishes a new v5 root.  Normal
campaign resume never performs this migration implicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import refined_relationship_campaign as campaign
from . import refined_relationship_campaign_migration as v3_to_v4
from .power_loading_study import geometry_csv_text, geometry_rows, validate_checkpoint_samples
from .sampling import CaptureVelocitySample, load_capture_velocity_samples
from .simple_force_sweep import (
    DETUNING_N_VALUES as FORCE_DETUNING_N_VALUES,
    SimpleForceSweepNumerics,
    _resume_signature as force_resume_signature,
)
from .timeout_audit import VelocityResolvedCaptureOverride, velocity_overrides_from_payload


SOURCE_CAMPAIGN_NAME = (
    "refined_relationships_full_sphere_25x25_r15mm_"
    "27mW_reference_optimized_v4_20260908"
)
DESTINATION_CAMPAIGN_NAME = (
    "refined_relationships_full_sphere_25x25_r15mm_"
    "27mW_reference_optimized_v5_20260908"
)
SOURCE_CAMPAIGN_SCHEMA_VERSION = 5
SOURCE_POINT_SCHEMA_VERSION = 5
DESTINATION_CAMPAIGN_SCHEMA_VERSION = 6
DESTINATION_POINT_SCHEMA_VERSION = 6
SOURCE_CAMPAIGN_SIGNATURE_SHA256 = (
    "c58cde3796ecb4e51394d84f490051e48b8bab202df831fb73fd905d6db18128"
)
SOURCE_GEOMETRY_SHA256 = (
    "02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17"
)
SOURCE_PHYSICS_HASH_MAP_SHA256 = (
    "d9a0afd3c56c1fd3e44f54769ba00a28e74dff3d45d6ec60015833e6f6096801"
)
SOURCE_COMPLETED_POINT_COUNT = 59
SOURCE_PARTIAL_STUDY_KEY = "03_detuning"
SOURCE_PARTIAL_POINT_INDEX = 15
SOURCE_PARTIAL_SAMPLE_COUNT = 615
EXPECTED_SOURCE_POINT_COUNT = 60
EXPECTED_SAMPLE_COUNT_PER_POINT = 625
MIGRATION_SCHEMA_VERSION = 2

PRIOR_MIGRATION_MANIFEST = "migration_manifest.json"
PRIOR_MIGRATION_MANIFEST_SHA256 = (
    "a2f86c1812fdf80d962e9b4a9d1c91c0f8b6de9e2716c6c550abd2562a562e44"
)
NEW_MIGRATION_MANIFEST = "migration_manifest_v4_to_v5.json"

DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY = (
    "diagnostics/zero_speed_delayed_capture_n_m3p75_disc23_point15"
)
ZERO_FLUX_DIAGNOSTIC_DIRECTORY = (
    "diagnostics/zero_speed_indeterminate_n_m4p25_disc7_point20"
)
ZERO_FLUX_DIAGNOSTIC_JSON_SHA256 = (
    "831b4f17c85b7784a93d85e8c1cc63d77059477a84fb7b0f5fc8fd11e8ba8c81"
)
ZERO_FLUX_DIAGNOSTIC_JSON_BYTES = 4_343
ZERO_FLUX_DIAGNOSTIC_README_SHA256 = (
    "678afe6b434f343e654f198377bdf796aea61082eefa86149850e2b1fbd3aab8"
)
ZERO_FLUX_DIAGNOSTIC_README_BYTES = 1_841

SOURCE_ENDPOINT_TIMEOUT_POLICY: Mapping[str, object] = {
    "base_max_time_s": 0.05,
    "audit_duration_s": 0.2,
    "audit_coarse_time_step_s": 5.0e-6,
    "fine_time_step_s": 2.5e-6,
    "accepted_integration_fields": {
        "duration": "accepted_max_simulation_time_s",
        "coarse_time_step": "accepted_coarse_time_step_s",
        "fine_time_step": "accepted_time_step_s",
        "meaning": (
            "authoritative complete dual-timestep boundary level: normally "
            "200 ms, or the first clean bounded longer-duration level after "
            "a censored/disputed positive boundary"
        ),
    },
    "adaptive_integration_fields": {
        "duration": "adaptive_max_duration_s",
        "coarse_time_step": "adaptive_coarse_time_step_s",
        "fine_time_step": "adaptive_fine_time_step_s",
        "meaning": (
            "highest node-only direct-grid level actually evaluated; all "
            "zeros when no grid node escalation was required"
        ),
    },
    "adaptive_node_levels": [
        {
            "duration_s": 0.25,
            "coarse_time_step_s": 5.0e-6,
            "fine_time_step_s": 2.5e-6,
        },
        {
            "duration_s": 0.4,
            "coarse_time_step_s": 2.5e-6,
            "fine_time_step_s": 1.25e-6,
        },
        {
            "duration_s": 1.0,
            "coarse_time_step_s": 1.25e-6,
            "fine_time_step_s": 0.625e-6,
        },
    ],
    "complete_positive_boundary_search_levels": [
        {
            "duration_s": 0.25,
            "coarse_time_step_s": 5.0e-6,
            "fine_time_step_s": 2.5e-6,
        },
        {
            "duration_s": 0.4,
            "coarse_time_step_s": 2.5e-6,
            "fine_time_step_s": 1.25e-6,
        },
    ],
    "acceptance": (
        "Every actual timeout-contaminated speed in the base search and both "
        "saved endpoints are classified at 200 ms with 5 and 2.5 microsecond "
        "steps. The saved bracket is retained only when both endpoints are "
        "definitive and every timeout speed becomes escaped at both steps; "
        "otherwise complete instrumented dual-step searches must recover "
        "compatible definitive brackets. A censored or incompatible 200 ms "
        "positive boundary is re-searched completely at 250 ms and, only if "
        "needed, 400 ms. Every evaluated search node is persisted. Once a "
        "positive scalar search has failed its premise, an independent full "
        "loading-grid scan is retained as the authoritative boolean mask even "
        "when monotone. Every scalar zero threshold is independently scanned "
        "from 0 to 30 m/s by 0.25 m/s at both timesteps, with finite-speed "
        "capture islands retained as direct boolean masks. Only non-definitive "
        "or timestep-disagreeing grid or positive-boundary nodes escalate to "
        "250 ms at 5/2.5 microseconds and then, only if still required, to "
        "400 ms at 2.5/1.25 microseconds and finally 1 s at 1.25/0.625 "
        "microseconds. The 1 s level is node-only, not a complete "
        "positive-boundary re-search. A node is accepted only when both "
        "timesteps give the same definitive trapped/escaped classification."
    ),
}

# This is the exact schema-5 aggregate header.  It is deliberately independent
# of the active schema-6 source so the stopped source is authenticated rather
# than interpreted through a moving header.
SOURCE_AGGREGATE_FIELDNAMES: tuple[str, ...] = (
    "point_index",
    "study_key",
    "scan_variable",
    "scan_value",
    "s0",
    "seff",
    "detuning_n",
    "cooling_power_w_per_beam",
    "cooling_power_mw_per_beam",
    "cooling_beam_diameter_m",
    "cooling_beam_diameter_mm",
    "cooling_beam_center_peak_intensity_w_per_m2",
    "cooling_beam_center_on_resonance_saturation_parameter",
    "cooling_beam_center_effective_saturation_parameter",
    "cooling_detuning_hz",
    "cooling_detuning_mhz",
    "linewidth_hz",
    "loading_rate_mean_atoms_per_s",
    "loading_rate_from_mean_spectrum_atoms_per_s",
    "loading_rate_sample_std_atoms_per_s",
    "loading_rate_disc_cluster_sem_atoms_per_s",
    "loading_rate_t95_lower_atoms_per_s",
    "loading_rate_t95_upper_atoms_per_s",
    "student_t_critical_95",
    "confidence_level",
    "disc_count",
    "points_per_disc",
    "capture_threshold_search_count",
    "base_timeout_ray_count",
    "zero_capture_velocity_count",
    "velocity_resolved_override_count",
    "unresolved_timeout_count",
    "lower_classification_counts_json",
    "upper_classification_counts_json",
    "disc_radius_m",
    "disc_radius_mm",
    "phase_space",
    "geometry_sha256",
    "run_signature_sha256",
    "point_elapsed_wall_time_s",
    "statistics_directory",
    "figures_directory",
    "capture_cross_section_csv",
    "capture_cross_section_plot",
    "status",
)


@dataclass(frozen=True, slots=True)
class VerifiedPoint:
    point: campaign.RelationshipPoint
    source_signature_sha256: str
    status: str
    completed_sample_count: int


@dataclass(frozen=True, slots=True)
class VerifiedCampaign:
    points: tuple[VerifiedPoint, ...]
    source_physics_hashes: Mapping[str, str]
    source_file_hashes: Mapping[str, str]
    source_total_bytes: int


def _canonical_hash(payload: Mapping[str, object]) -> str:
    return v3_to_v4._canonical_hash(payload)


def _sha256_file(path: Path) -> str:
    return v3_to_v4._sha256_file(path)


def _json_equivalent(left: object, right: object) -> bool:
    return v3_to_v4._json_equivalent(left, right)


def _assert_close(
    actual: object,
    expected: float,
    *,
    label: str,
    atol: float = 1.0e-15,
) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(value) or not np.isclose(value, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label} mismatch: {value!r} != {expected!r}")


def _active_destination_contract() -> None:
    checks = (
        (campaign.CAMPAIGN_NAME, DESTINATION_CAMPAIGN_NAME, "campaign name"),
        (
            campaign.CAMPAIGN_SCHEMA_VERSION,
            DESTINATION_CAMPAIGN_SCHEMA_VERSION,
            "campaign schema",
        ),
        (
            campaign.POINT_SCHEMA_VERSION,
            DESTINATION_POINT_SCHEMA_VERSION,
            "point schema",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RuntimeError(
                f"active {label} is {actual!r}; migration requires {expected!r}"
            )


def _expected_source_point_payload(
    point: campaign.RelationshipPoint,
    source_physics_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Reconstruct the exact signed schema-5 payload using pinned old policy."""

    expected = campaign._point_signature_payload(
        point, campaign.default_search_config(), SOURCE_GEOMETRY_SHA256
    )
    expected["schema_version"] = SOURCE_POINT_SCHEMA_VERSION
    expected["physics_source_sha256"] = dict(source_physics_hashes)
    expected["endpoint_timeout_policy"] = json.loads(
        json.dumps(SOURCE_ENDPOINT_TIMEOUT_POLICY)
    )
    # Schema-6 may sign an explicit zero-flux interpretation at top level.  It
    # did not exist in schema 5 and therefore cannot be allowed into the
    # reconstructed source payload.
    for key in (
        "indeterminate_zero_flux_policy",
        "zero_flux_policy",
        "loading_zero_flux_policy",
    ):
        expected.pop(key, None)
    return expected


def _validate_source_point_signature(
    metadata: Mapping[str, object],
    point: campaign.RelationshipPoint,
) -> tuple[str, dict[str, str]]:
    signature = metadata.get("run_signature_sha256")
    payload = metadata.get("signature_payload")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(f"source point {point.slug} lacks a valid signature")
    if not isinstance(payload, Mapping) or _canonical_hash(payload) != signature:
        raise ValueError(f"source point {point.slug} signature payload was tampered")
    physics_hashes = payload.get("physics_source_sha256")
    if not isinstance(physics_hashes, Mapping) or not physics_hashes:
        raise ValueError(f"source point {point.slug} lacks physics source hashes")
    normalized = {str(key): str(value) for key, value in physics_hashes.items()}
    if any(len(value) != 64 for value in normalized.values()):
        raise ValueError(f"source point {point.slug} has malformed physics hashes")
    if _canonical_hash(normalized) != SOURCE_PHYSICS_HASH_MAP_SHA256:
        raise ValueError(f"source point {point.slug} physics hash map mismatch")
    expected = _expected_source_point_payload(point, normalized)
    if not _json_equivalent(payload, expected):
        changed = sorted(
            key
            for key in set(payload).union(expected)
            if not _json_equivalent(payload.get(key), expected.get(key))
        )
        raise ValueError(
            f"source point {point.slug} signed contract mismatch: {changed}"
        )
    if metadata.get("schema_version") != SOURCE_POINT_SCHEMA_VERSION:
        raise ValueError(f"source point {point.slug} schema mismatch")
    if metadata.get("study_name") != (
        f"{SOURCE_CAMPAIGN_NAME}/{point.study_key}/{point.slug}"
    ):
        raise ValueError(f"source point {point.slug} study identity mismatch")
    if metadata.get("geometry_sha256") != SOURCE_GEOMETRY_SHA256:
        raise ValueError(f"source point {point.slug} geometry mismatch")
    if metadata.get("relationship_point") != asdict(point):
        raise ValueError(f"source point {point.slug} plan mismatch")
    duplicates = {
        "apparatus_config": "apparatus_config",
        "simple_mot_config": "simple_mot_config",
        "coil_config": "coil_config",
        "capture_search_config": "search_config",
        "endpoint_timeout_policy": "endpoint_timeout_policy",
    }
    for metadata_key, payload_key in duplicates.items():
        if not _json_equivalent(metadata.get(metadata_key), payload[payload_key]):
            raise ValueError(
                f"source point {point.slug} top-level {metadata_key} mismatch"
            )
    if metadata.get("built_cooling_beam_count") != 6:
        raise ValueError(f"source point {point.slug} cooling beam count mismatch")
    if metadata.get("repumper_included") is not False:
        raise ValueError(f"source point {point.slug} unexpectedly has repumping")
    return signature, normalized


def _expected_imported_points() -> tuple[campaign.RelationshipPoint, ...]:
    groups = campaign.requested_points()
    return (
        *groups[campaign.RAW_STUDY_KEY],
        *groups[campaign.EFFECTIVE_STUDY_KEY],
        *groups[campaign.DETUNING_STUDY_KEY][: SOURCE_PARTIAL_POINT_INDEX + 1],
    )


def _load_and_validate_ledgers(
    point_paths,
    launch_points: Sequence[object],
    search,
) -> tuple[
    dict[tuple[int, int], CaptureVelocitySample],
    dict[tuple[int, int], dict[str, object]],
    dict[tuple[int, int], VelocityResolvedCaptureOverride],
]:
    sample_path = (
        point_paths.final_samples_csv
        if point_paths.final_samples_csv.is_file()
        else point_paths.partial_samples_csv
    )
    if not sample_path.is_file():
        raise FileNotFoundError(f"point has no sample checkpoint: {point_paths.statistics}")
    samples = validate_checkpoint_samples(
        load_capture_velocity_samples(sample_path), launch_points
    )
    audit_rows = campaign._load_audit_rows(campaign._audit_path(point_paths))
    if set(samples) != set(audit_rows):
        raise ValueError("sample and endpoint-audit keys differ")
    override_path = campaign._overrides_path(point_paths)
    if not override_path.is_file():
        raise FileNotFoundError(f"missing velocity-override ledger: {override_path}")
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    overrides = {
        item.key: item for item in velocity_overrides_from_payload(payload)
    }
    campaign._validate_completed_audit_ledger(
        samples, audit_rows, overrides, search=search
    )
    return samples, audit_rows, overrides


def _validate_source_identity(metadata: Mapping[str, object]) -> None:
    checks = (
        (metadata.get("schema_version"), SOURCE_CAMPAIGN_SCHEMA_VERSION, "schema"),
        (metadata.get("campaign_name"), SOURCE_CAMPAIGN_NAME, "name"),
        (
            metadata.get("campaign_signature_sha256"),
            SOURCE_CAMPAIGN_SIGNATURE_SHA256,
            "signature",
        ),
        (metadata.get("common_geometry_sha256"), SOURCE_GEOMETRY_SHA256, "geometry"),
        (metadata.get("loading_point_count"), 67, "point count"),
        (metadata.get("capture_threshold_search_count"), 41_875, "ray count"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"source campaign {label} mismatch")
    if metadata.get("execution_order") != list(campaign.STUDY_ORDER):
        raise ValueError("source execution order mismatch")
    if metadata.get("phase_space") != "full_sphere":
        raise ValueError("source is not full-sphere")
    if metadata.get("stage_status") != {
        campaign.RAW_STUDY_KEY: "completed",
        campaign.EFFECTIVE_STUDY_KEY: "completed",
        campaign.DETUNING_STUDY_KEY: "failed",
    }:
        raise ValueError("source stage status is not the pinned stopped state")
    error = str(metadata.get("last_error", ""))
    if "(7, 20)" not in error or "0 m/s" not in error:
        raise ValueError("source campaign does not retain the pinned failure")
    if metadata.get("search_config") != asdict(campaign.default_search_config()):
        raise ValueError("source search configuration mismatch")
    expected_grids = {
        "s0": list(campaign.RAW_SATURATION_VALUES),
        "s_eff": list(campaign.EFFECTIVE_SATURATION_VALUES),
        "detuning_delta_over_gamma": list(campaign.DETUNING_N_VALUES),
    }
    if metadata.get("requested_grids") != expected_grids:
        raise ValueError("source requested grids mismatch")


def _validate_source_aggregates(
    statistics: Path,
    figures: Path,
    verified: Mapping[tuple[str, int], VerifiedPoint],
) -> None:
    completed_by_study = {
        campaign.RAW_STUDY_KEY: 24,
        campaign.EFFECTIVE_STUDY_KEY: 20,
        campaign.DETUNING_STUDY_KEY: 15,
    }
    plans = campaign.requested_points()
    for study_key, count in completed_by_study.items():
        aggregate_path = statistics / study_key / "aggregate.csv"
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(SOURCE_AGGREGATE_FIELDNAMES):
                raise ValueError(f"source {study_key} aggregate schema mismatch")
            rows = list(reader)
        if len(rows) != count:
            raise ValueError(f"source {study_key} aggregate count mismatch")
        for index, row in enumerate(rows):
            key = (study_key, index)
            if int(row["point_index"]) != index or row["status"] != "completed":
                raise ValueError(f"source {study_key} aggregate order/status mismatch")
            if row["run_signature_sha256"] != verified[key].source_signature_sha256:
                raise ValueError(f"source {study_key} aggregate signature mismatch")
            if row["geometry_sha256"] != SOURCE_GEOMETRY_SHA256:
                raise ValueError(f"source {study_key} aggregate geometry mismatch")
        sweep = json.loads(
            (statistics / study_key / "sweep_metadata.json").read_text(encoding="utf-8")
        )
        if sweep.get("schema_version") != SOURCE_CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"source {study_key} sweep schema mismatch")
        if sweep.get("study_key") != study_key:
            raise ValueError(f"source {study_key} sweep identity mismatch")
        if int(sweep.get("point_count_completed", -1)) != count:
            raise ValueError(f"source {study_key} sweep completed count mismatch")
        if sweep.get("ordered_point_plan") != [asdict(item) for item in plans[study_key]]:
            raise ValueError(f"source {study_key} sweep plan mismatch")
        relationship_plot = campaign.CampaignPaths(
            statistics, figures
        ).relationship_plot(study_key)
        if not relationship_plot.is_file() or relationship_plot.stat().st_size == 0:
            raise FileNotFoundError(f"source relationship plot missing: {relationship_plot}")


def _validate_force_outputs(statistics: Path, figures: Path) -> None:
    stats = statistics / "04_force_vs_detuning_27mW"
    figs = figures / "04_force_vs_detuning_27mW"
    metadata = json.loads(
        (stats / "force_vs_detuning_metadata.json").read_text(encoding="utf-8")
    )
    expected_signature = force_resume_signature(
        FORCE_DETUNING_N_VALUES, SimpleForceSweepNumerics()
    )
    if metadata.get("status") != "completed" or metadata.get(
        "resume_signature"
    ) != expected_signature:
        raise ValueError("force sweep identity/status mismatch")
    if metadata.get("all_convergence_checks_passed") is not True:
        raise ValueError("force sweep convergence failed")
    with (stats / "force_vs_detuning.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 111 or not all(
        str(row.get("all_converged", "")).lower() == "true" for row in rows
    ):
        raise ValueError("force sweep rows are incomplete or unconverged")
    grid = np.asarray([float(row["detuning_n"]) for row in rows])
    if not np.array_equal(grid, np.asarray(FORCE_DETUNING_N_VALUES)):
        raise ValueError("force sweep grid mismatch")
    for name in (
        "restoring_slope_vs_detuning.png",
        "damping_turnaround_vs_detuning.png",
    ):
        path = figs / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"force figure missing: {path}")


def _validate_prior_migration_manifest(statistics: Path) -> None:
    path = statistics / PRIOR_MIGRATION_MANIFEST
    if _sha256_file(path) != PRIOR_MIGRATION_MANIFEST_SHA256:
        raise ValueError("prior v3-to-v4 migration manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source_campaign")
    destination = payload.get("destination_campaign")
    if not isinstance(source, Mapping) or not isinstance(destination, Mapping):
        raise ValueError("prior migration manifest is malformed")
    if source.get("name") != v3_to_v4.SOURCE_CAMPAIGN_NAME:
        raise ValueError("prior manifest v3 identity mismatch")
    if destination.get("name") != SOURCE_CAMPAIGN_NAME:
        raise ValueError("prior manifest v4 identity mismatch")
    if destination.get("campaign_signature_sha256") != SOURCE_CAMPAIGN_SIGNATURE_SHA256:
        raise ValueError("prior manifest v4 signature mismatch")


def _validate_zero_flux_diagnostic(statistics: Path) -> None:
    directory = statistics / ZERO_FLUX_DIAGNOSTIC_DIRECTORY
    readme = directory / "README.md"
    data_path = directory / "diagnostic.json"
    if readme.stat().st_size != ZERO_FLUX_DIAGNOSTIC_README_BYTES:
        raise ValueError("zero-flux diagnostic README byte count mismatch")
    if data_path.stat().st_size != ZERO_FLUX_DIAGNOSTIC_JSON_BYTES:
        raise ValueError("zero-flux diagnostic JSON byte count mismatch")
    if _sha256_file(readme) != ZERO_FLUX_DIAGNOSTIC_README_SHA256:
        raise ValueError("zero-flux diagnostic README hash mismatch")
    if _sha256_file(data_path) != ZERO_FLUX_DIAGNOSTIC_JSON_SHA256:
        raise ValueError("zero-flux diagnostic JSON hash mismatch")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    failure = data.get("production_failure")
    if not isinstance(failure, Mapping) or (
        failure.get("source_campaign"),
        failure.get("source_campaign_signature_sha256"),
        failure.get("point_status"),
        failure.get("completed_sample_count"),
        failure.get("expected_sample_count"),
    ) != (
        SOURCE_CAMPAIGN_NAME,
        SOURCE_CAMPAIGN_SIGNATURE_SHA256,
        "failed",
        SOURCE_PARTIAL_SAMPLE_COUNT,
        EXPECTED_SAMPLE_COUNT_PER_POINT,
    ):
        raise ValueError("zero-flux diagnostic production identity mismatch")
    ray = data.get("ray")
    if not isinstance(ray, Mapping) or (
        ray.get("disc_index"),
        ray.get("point_index"),
        ray.get("incident_speed_m_per_s"),
        ray.get("geometry_seed"),
        ray.get("geometry_sha256"),
    ) != (7, 20, 0.0, campaign.DEFAULT_SEED, SOURCE_GEOMETRY_SHA256):
        raise ValueError("zero-flux diagnostic ray identity mismatch")
    velocity = np.asarray(ray.get("initial_velocity_m_per_s"), dtype=float)
    if velocity.shape != (3,) or not np.array_equal(velocity, np.zeros(3)):
        raise ValueError("zero-flux diagnostic initial velocity is not exactly zero")
    physics = data.get("physics_configuration")
    expected_criterion = (
        "continuously inside the 2 mm-radius core for at least 5 ms OR two core "
        "entries with an intervening exit"
    )
    if not isinstance(physics, Mapping) or (
        physics.get("cooling_detuning_n"),
        physics.get("cooling_power_w_per_beam"),
        physics.get("gravity_enabled"),
        physics.get("trap_core_radius_m"),
        physics.get("trapped_criterion"),
    ) != (-4.25, 0.027, True, 0.002, expected_criterion):
        raise ValueError("zero-flux diagnostic physics/criterion mismatch")
    pair = data.get("two_second_convergence_pair")
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError("zero-flux diagnostic pair is malformed")
    expected_steps = (1.25e-6, 0.625e-6)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    minimum_radii: list[float] = []
    final_radii: list[float] = []
    for record, step in zip(pair, expected_steps, strict=True):
        if not isinstance(record, Mapping):
            raise ValueError("zero-flux diagnostic trajectory record is malformed")
        if (
            record.get("termination_reason") != "timeout"
            or record.get("trapped") is not False
            or record.get("entered_trap_core") is not False
            or record.get("core_entry_count") != 0
        ):
            raise ValueError("zero-flux diagnostic was relabeled or entered the core")
        _assert_close(record.get("duration_s"), 2.0, label="diagnostic duration")
        _assert_close(record.get("time_step_s"), step, label="diagnostic time step")
        _assert_close(record.get("elapsed_time_s"), 2.0, label="diagnostic elapsed time")
        position = np.asarray(record.get("final_position_m"), dtype=float)
        final_velocity = np.asarray(record.get("final_velocity_m_per_s"), dtype=float)
        if position.shape != (3,) or final_velocity.shape != (3,):
            raise ValueError("zero-flux diagnostic terminal vector shape mismatch")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(final_velocity)):
            raise ValueError("zero-flux diagnostic terminal vector is nonfinite")
        minimum_radius = float(record.get("minimum_radius_m", np.nan))
        final_radius = float(record.get("final_radius_m", np.nan))
        if not np.isfinite(minimum_radius) or not np.isfinite(final_radius):
            raise ValueError("zero-flux diagnostic radius is nonfinite")
        if minimum_radius <= 0.002 or final_radius >= 0.03:
            raise ValueError("zero-flux diagnostic crossed a terminal region")
        positions.append(position)
        velocities.append(final_velocity)
        minimum_radii.append(minimum_radius)
        final_radii.append(final_radius)
    convergence = data.get("convergence")
    if not isinstance(convergence, Mapping) or any(
        convergence.get(key) is not True
        for key in (
            "same_termination_reason",
            "both_finite",
            "both_timeout",
            "both_zero_core_entries",
        )
    ):
        raise ValueError("zero-flux diagnostic convergence flags mismatch")
    reconstructed = {
        "position_difference_norm_m": float(np.linalg.norm(positions[0] - positions[1])),
        "velocity_difference_norm_m_per_s": float(
            np.linalg.norm(velocities[0] - velocities[1])
        ),
        "minimum_radius_difference_m": abs(minimum_radii[0] - minimum_radii[1]),
        "final_radius_difference_m": abs(final_radii[0] - final_radii[1]),
    }
    for key, value in reconstructed.items():
        _assert_close(convergence.get(key), value, label=f"diagnostic {key}", atol=1.0e-18)
    if reconstructed["position_difference_norm_m"] >= 1.0e-10 or reconstructed[
        "velocity_difference_norm_m_per_s"
    ] >= 1.0e-8:
        raise ValueError("zero-flux diagnostic timesteps did not converge tightly")
    conclusion = data.get("conclusion")
    if not isinstance(conclusion, Mapping) or (
        conclusion.get("physical_capture_classification"),
        conclusion.get("loading_classification"),
    ) != ("indeterminate", "indeterminate_zero_flux"):
        raise ValueError("zero-flux diagnostic conclusion mismatch")
    requirements = conclusion.get("requirements")
    if not isinstance(requirements, list) or not any(
        "g(0)=0" in str(item) for item in requirements
    ):
        raise ValueError("zero-flux diagnostic does not preserve the quadrature anchor")


def validate_source_campaign(
    source_statistics: Path,
    source_figures: Path,
    *,
    source_file_hashes: Mapping[str, str],
    source_total_bytes: int,
) -> VerifiedCampaign:
    """Authenticate the exact stopped v4 campaign and every saved checkpoint."""

    metadata = json.loads(
        (source_statistics / "campaign_metadata.json").read_text(encoding="utf-8")
    )
    _validate_source_identity(metadata)
    if _sha256_file(source_statistics / "launch_geometry.csv") != SOURCE_GEOMETRY_SHA256:
        raise ValueError("source common geometry bytes changed")
    search = campaign.default_search_config()
    discs, launch_points = campaign.generate_common_geometry(search)
    geometry = geometry_csv_text(geometry_rows(discs, launch_points))
    if hashlib.sha256(geometry.encode("utf-8")).hexdigest() != SOURCE_GEOMETRY_SHA256:
        raise RuntimeError("current geometry generator does not reproduce source geometry")

    points = _expected_imported_points()
    discovered = {
        path.relative_to(source_statistics).as_posix()
        for path in source_statistics.glob("0[123]_*/points/*/run_metadata.json")
    }
    expected = {
        f"{point.study_key}/points/{point.slug}/run_metadata.json" for point in points
    }
    if discovered != expected:
        raise ValueError("source point-directory inventory mismatch")

    verified: list[VerifiedPoint] = []
    common_hashes: dict[str, str] | None = None
    paths = campaign.CampaignPaths(source_statistics, source_figures)
    print(f"[v4->v5] validating {len(points)} point checkpoints", flush=True)
    for ordinal, point in enumerate(points, start=1):
        point_paths = paths.point_paths(point)
        point_metadata = json.loads(
            point_paths.metadata_json.read_text(encoding="utf-8")
        )
        signature, physics_hashes = _validate_source_point_signature(
            point_metadata, point
        )
        if common_hashes is None:
            common_hashes = physics_hashes
        elif physics_hashes != common_hashes:
            raise ValueError("source point physics hash maps differ")
        if _sha256_file(point_paths.geometry_csv) != SOURCE_GEOMETRY_SHA256:
            raise ValueError(f"source point geometry changed: {point.slug}")
        samples, _, _ = _load_and_validate_ledgers(
            point_paths, launch_points, search
        )
        partial = (
            point.study_key == SOURCE_PARTIAL_STUDY_KEY
            and point.point_index == SOURCE_PARTIAL_POINT_INDEX
        )
        status = "failed" if partial else "completed"
        count = SOURCE_PARTIAL_SAMPLE_COUNT if partial else EXPECTED_SAMPLE_COUNT_PER_POINT
        if point_metadata.get("status") != status or int(
            point_metadata.get("completed_sample_count", -1)
        ) != count:
            raise ValueError(f"source point status/count mismatch: {point.slug}")
        if len(samples) != count:
            raise ValueError(f"source point ledger count mismatch: {point.slug}")
        if int(point_metadata.get("expected_sample_count", -1)) != EXPECTED_SAMPLE_COUNT_PER_POINT:
            raise ValueError(f"source point expected count mismatch: {point.slug}")
        if partial:
            if point_paths.final_samples_csv.exists():
                raise ValueError("partial source point unexpectedly has final samples")
            if "(7, 20)" not in str(point_metadata.get("last_error", "")):
                raise ValueError("partial source point lacks pinned failure")
        else:
            if not point_paths.final_samples_csv.is_file():
                raise FileNotFoundError(f"completed point lacks final samples: {point.slug}")
            if _sha256_file(point_paths.final_samples_csv) != _sha256_file(
                point_paths.partial_samples_csv
            ):
                raise ValueError(f"completed point final/partial ledgers differ: {point.slug}")
            loading = json.loads(point_paths.loading_json.read_text(encoding="utf-8"))
            summary = json.loads(
                point_paths.capture_summary_json.read_text(encoding="utf-8")
            )
            duplicated = (
                point_metadata.get("loading_rate"),
                summary.get("loading_rate"),
                loading,
            )
            if summary.get("run_signature_sha256") != signature:
                raise ValueError(f"source summary signature mismatch: {point.slug}")
            if not all(
                isinstance(item, Mapping)
                and item.get("run_signature_sha256") == signature
                for item in duplicated
            ) or not all(_json_equivalent(duplicated[0], item) for item in duplicated[1:]):
                raise ValueError(f"source loading duplication mismatch: {point.slug}")
        verified.append(VerifiedPoint(point, signature, status, count))
        if ordinal == len(points) or ordinal % 5 == 0:
            print(f"[v4->v5] validated point ledgers {ordinal}/{len(points)}", flush=True)

    if common_hashes is None:
        raise ValueError("source has no physics hash map")
    if sum(item.status == "completed" for item in verified) != SOURCE_COMPLETED_POINT_COUNT:
        raise ValueError("source completed-point count mismatch")
    by_key = {(item.point.study_key, item.point.point_index): item for item in verified}
    _validate_source_aggregates(source_statistics, source_figures, by_key)
    _validate_force_outputs(source_statistics, source_figures)
    _validate_prior_migration_manifest(source_statistics)
    # The first diagnostic is part of the prior authenticated migration; the
    # historical validator pins both hashes and scientific semantics.
    v3_to_v4._validate_delayed_capture_diagnostic(source_statistics)
    _validate_zero_flux_diagnostic(source_statistics)
    return VerifiedCampaign(
        points=tuple(verified),
        source_physics_hashes=common_hashes,
        source_file_hashes=dict(source_file_hashes),
        source_total_bytes=source_total_bytes,
    )


def _replace_strings(value: object, replacements: Sequence[tuple[str, str]]) -> object:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _path_replacements(
    source_statistics: Path,
    source_figures: Path,
    destination_statistics: Path,
    destination_figures: Path,
) -> tuple[tuple[str, str], ...]:
    pairs = (
        (str(source_statistics.resolve()), str(destination_statistics.resolve())),
        (str(source_figures.resolve()), str(destination_figures.resolve())),
        (SOURCE_CAMPAIGN_NAME, DESTINATION_CAMPAIGN_NAME),
    )
    expanded: list[tuple[str, str]] = []
    for old, new in pairs:
        expanded.extend(((old, new), (old.replace("\\", "/"), new.replace("\\", "/"))))
    return tuple(expanded)


def _read_rewrite_json(
    path: Path,
    replacements: Sequence[tuple[str, str]],
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewritten = _replace_strings(payload, replacements)
    if not isinstance(rewritten, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return rewritten


def _append_migration_history(
    payload: dict[str, object],
    new_record: Mapping[str, object],
) -> None:
    history_raw = payload.get("migration_history", [])
    if not isinstance(history_raw, list):
        raise ValueError("migration_history is not a list")
    history = list(history_raw)
    prior = payload.get("migration")
    if isinstance(prior, Mapping) and not any(_json_equivalent(prior, item) for item in history):
        history.append(dict(prior))
    history.append(dict(new_record))
    payload["migration_history"] = history
    payload["migration"] = dict(new_record)


def _migration_record(
    *,
    migrated_utc: str,
    scope: str,
    source_point_signature_sha256: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migrated_utc": migrated_utc,
        "source_campaign_name": SOURCE_CAMPAIGN_NAME,
        "source_campaign_signature_sha256": SOURCE_CAMPAIGN_SIGNATURE_SHA256,
        "scope": scope,
        "reason": (
            "schema 6 preserves a converged matched dual-timestep timeout at exactly "
            "zero incident speed as indeterminate_zero_flux without classifying it as "
            "captured or escaped"
        ),
        "diagnostic_reference": ZERO_FLUX_DIAGNOSTIC_DIRECTORY + "/README.md",
        "prior_manifest_sha256": PRIOR_MIGRATION_MANIFEST_SHA256,
    }
    if source_point_signature_sha256 is not None:
        record["source_point_signature_sha256"] = source_point_signature_sha256
    return record


def _resign_destination(
    source_statistics: Path,
    source_figures: Path,
    staged_statistics: Path,
    staged_figures: Path,
    final_statistics: Path,
    final_figures: Path,
    verified: VerifiedCampaign,
) -> dict[tuple[str, int], str]:
    replacements = _path_replacements(
        source_statistics,
        source_figures,
        final_statistics,
        final_figures,
    )
    search = campaign.default_search_config()
    paths = campaign.CampaignPaths(staged_statistics, staged_figures)
    signatures: dict[tuple[str, int], str] = {}
    migrated_utc = campaign._utc_now()
    for item in verified.points:
        point = item.point
        point_paths = paths.point_paths(point)
        new_payload = campaign._point_signature_payload(
            point, search, SOURCE_GEOMETRY_SHA256
        )
        signature = campaign._signature(new_payload)
        signatures[(point.study_key, point.point_index)] = signature
        point_replacements = (*replacements, (item.source_signature_sha256, signature))

        loading: dict[str, object] | None = None
        summary: dict[str, object] | None = None
        if item.status == "completed":
            loading = _read_rewrite_json(point_paths.loading_json, point_replacements)
            loading.update(
                {
                    "run_signature_sha256": signature,
                    "indeterminate_zero_flux_ray_count": 0,
                    "capture_spectrum_row_count": 121,
                    "zero_flux_quadrature_anchor_used": False,
                    "zero_speed_cross_section_imputed": False,
                }
            )
            campaign._atomic_write_json(point_paths.loading_json, loading)
            summary = _read_rewrite_json(
                point_paths.capture_summary_json, point_replacements
            )
            summary.update(
                {
                    "run_signature_sha256": signature,
                    "indeterminate_zero_flux_ray_count": 0,
                    "loading_rate": loading,
                }
            )
            campaign._atomic_write_json(point_paths.capture_summary_json, summary)

        metadata = _read_rewrite_json(point_paths.metadata_json, point_replacements)
        metadata.update(
            {
                "schema_version": DESTINATION_POINT_SCHEMA_VERSION,
                "study_name": f"{DESTINATION_CAMPAIGN_NAME}/{point.study_key}/{point.slug}",
                "run_signature_sha256": signature,
                "signature_payload": new_payload,
                "endpoint_timeout_policy": new_payload["endpoint_timeout_policy"],
                "indeterminate_zero_flux_ray_count": 0,
            }
        )
        if loading is not None:
            metadata["loading_rate"] = loading
        _append_migration_history(
            metadata,
            _migration_record(
                migrated_utc=migrated_utc,
                scope="point",
                source_point_signature_sha256=item.source_signature_sha256,
            ),
        )
        campaign._atomic_write_json(point_paths.metadata_json, metadata)

    for study_key in campaign.STUDY_ORDER:
        aggregate_path = paths.aggregate_csv(study_key)
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(SOURCE_AGGREGATE_FIELDNAMES):
                raise ValueError(f"staged {study_key} source aggregate schema changed")
            rows = list(reader)
        for row in rows:
            key = (study_key, int(row["point_index"]))
            row["run_signature_sha256"] = signatures[key]
            row["indeterminate_zero_flux_ray_count"] = 0
            for field in (
                "statistics_directory",
                "figures_directory",
                "capture_cross_section_csv",
                "capture_cross_section_plot",
            ):
                row[field] = _replace_strings(row[field], replacements)
        campaign._atomic_write_csv(aggregate_path, rows, campaign.AGGREGATE_FIELDNAMES)

        sweep_path = paths.study_metadata_json(study_key)
        sweep = _read_rewrite_json(sweep_path, replacements)
        sweep["schema_version"] = DESTINATION_CAMPAIGN_SCHEMA_VERSION
        sweep["indeterminate_zero_flux_ray_count"] = 0
        _append_migration_history(
            sweep,
            _migration_record(migrated_utc=migrated_utc, scope="study"),
        )
        campaign._atomic_write_json(sweep_path, sweep)

    campaign_metadata = _read_rewrite_json(paths.metadata_json, replacements)
    campaign_metadata.update(
        {
            "schema_version": DESTINATION_CAMPAIGN_SCHEMA_VERSION,
            "campaign_name": DESTINATION_CAMPAIGN_NAME,
            "campaign_signature_sha256": campaign._campaign_signature(
                search, SOURCE_GEOMETRY_SHA256
            ),
            "indeterminate_zero_flux_ray_count": 0,
        }
    )
    _append_migration_history(
        campaign_metadata,
        _migration_record(migrated_utc=migrated_utc, scope="campaign"),
    )
    campaign._atomic_write_json(paths.metadata_json, campaign_metadata)

    force_metadata = (
        staged_statistics
        / "04_force_vs_detuning_27mW"
        / "force_vs_detuning_metadata.json"
    )
    rewritten_force = _read_rewrite_json(force_metadata, replacements)
    campaign._atomic_write_json(force_metadata, rewritten_force)
    return signatures


def _expected_transformed_files(verified: VerifiedCampaign) -> set[str]:
    transformed = {
        "statistics/campaign_metadata.json",
        "statistics/04_force_vs_detuning_27mW/force_vs_detuning_metadata.json",
    }
    for study_key in campaign.STUDY_ORDER:
        transformed.add(f"statistics/{study_key}/aggregate.csv")
        transformed.add(f"statistics/{study_key}/sweep_metadata.json")
    for item in verified.points:
        prefix = (
            f"statistics/{item.point.study_key}/points/{item.point.slug}/"
        )
        transformed.add(prefix + "run_metadata.json")
        if item.status == "completed":
            transformed.add(prefix + "loading_rate_result.json")
            transformed.add(prefix + "capture_velocity_summary.json")
    return transformed


def _validate_transformation_whitelist(
    source_hashes: Mapping[str, str],
    destination_hashes: Mapping[str, str],
    verified: VerifiedCampaign,
) -> None:
    if set(source_hashes) != set(destination_hashes):
        raise ValueError("source/destination inventories differ before new manifest")
    changed = {
        key for key in source_hashes if source_hashes[key] != destination_hashes[key]
    }
    expected = _expected_transformed_files(verified)
    if changed != expected:
        raise ValueError(
            "migration transformed an unexpected file set: "
            f"missing={sorted(expected - changed)[:5]}, extra={sorted(changed - expected)[:5]}"
        )
    prior_key = "statistics/" + PRIOR_MIGRATION_MANIFEST
    if destination_hashes.get(prior_key) != PRIOR_MIGRATION_MANIFEST_SHA256:
        raise ValueError("prior migration manifest changed in destination")


def _validate_active_paths(
    payload: Mapping[str, object],
    final_statistics: Path,
    final_figures: Path,
) -> None:
    for field, expected in (
        ("statistics_directory", final_statistics.resolve()),
        ("figures_directory", final_figures.resolve()),
    ):
        raw = payload.get(field)
        if not isinstance(raw, str) or Path(raw).resolve() != expected:
            raise ValueError(f"destination active {field} does not name final root")


def _validate_destination_point_payload(
    persisted_payload: object,
    expected_payload: Mapping[str, object],
    expected_signature: str,
    *,
    point_slug: str,
) -> None:
    """Validate a JSON-restored signed payload without tuple/list false alarms."""

    if (
        not isinstance(persisted_payload, Mapping)
        or _canonical_hash(persisted_payload) != expected_signature
        or not _json_equivalent(persisted_payload, expected_payload)
    ):
        raise ValueError(f"destination point payload mismatch: {point_slug}")


def _validate_destination(
    staged_statistics: Path,
    staged_figures: Path,
    final_statistics: Path,
    final_figures: Path,
    verified: VerifiedCampaign,
    signatures: Mapping[tuple[str, int], str],
) -> None:
    search = campaign.default_search_config()
    paths = campaign.CampaignPaths(staged_statistics, staged_figures)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != DESTINATION_CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("destination campaign schema mismatch")
    if metadata.get("campaign_name") != DESTINATION_CAMPAIGN_NAME:
        raise ValueError("destination campaign name mismatch")
    if metadata.get("campaign_signature_sha256") != campaign._campaign_signature(
        search, SOURCE_GEOMETRY_SHA256
    ):
        raise ValueError("destination campaign signature mismatch")
    if int(metadata.get("indeterminate_zero_flux_ray_count", -1)) != 0:
        raise ValueError("destination campaign zero-flux count mismatch")
    _validate_active_paths(metadata, final_statistics, final_figures)

    _, launch_points = campaign.generate_common_geometry(search)
    for item in verified.points:
        point = item.point
        point_paths = paths.point_paths(point)
        point_metadata = json.loads(
            point_paths.metadata_json.read_text(encoding="utf-8")
        )
        expected_payload = campaign._point_signature_payload(
            point, search, SOURCE_GEOMETRY_SHA256
        )
        expected_signature = campaign._signature(expected_payload)
        key = (point.study_key, point.point_index)
        if signatures.get(key) != expected_signature:
            raise RuntimeError("destination signature map is inconsistent")
        if point_metadata.get("run_signature_sha256") != expected_signature:
            raise ValueError(f"destination point signature mismatch: {point.slug}")
        _validate_destination_point_payload(
            point_metadata.get("signature_payload"),
            expected_payload,
            expected_signature,
            point_slug=point.slug,
        )
        if point_metadata.get("schema_version") != DESTINATION_POINT_SCHEMA_VERSION:
            raise ValueError(f"destination point schema mismatch: {point.slug}")
        if point_metadata.get("status") != item.status or int(
            point_metadata.get("completed_sample_count", -1)
        ) != item.completed_sample_count:
            raise ValueError(f"destination point status/count changed: {point.slug}")
        if int(point_metadata.get("indeterminate_zero_flux_ray_count", -1)) != 0:
            raise ValueError(f"migrated point has a synthetic zero-flux result: {point.slug}")
        history = point_metadata.get("migration_history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"destination point lacks migration history: {point.slug}")
        latest_migration = point_metadata.get("migration")
        if not isinstance(latest_migration, Mapping) or (
            latest_migration.get("source_campaign_name"),
            latest_migration.get("source_campaign_signature_sha256"),
            latest_migration.get("source_point_signature_sha256"),
        ) != (
            SOURCE_CAMPAIGN_NAME,
            SOURCE_CAMPAIGN_SIGNATURE_SHA256,
            item.source_signature_sha256,
        ):
            raise ValueError(f"destination point migration provenance mismatch: {point.slug}")
        samples, _, _ = _load_and_validate_ledgers(point_paths, launch_points, search)
        if len(samples) != item.completed_sample_count:
            raise ValueError(f"destination point ledger count changed: {point.slug}")
        if item.status == "completed":
            loading = json.loads(point_paths.loading_json.read_text(encoding="utf-8"))
            summary = json.loads(
                point_paths.capture_summary_json.read_text(encoding="utf-8")
            )
            if (
                loading.get("run_signature_sha256") != expected_signature
                or summary.get("run_signature_sha256") != expected_signature
                or summary.get("loading_rate") != loading
                or point_metadata.get("loading_rate") != loading
            ):
                raise ValueError(f"destination result signatures differ: {point.slug}")
            if any(
                int(payload.get("indeterminate_zero_flux_ray_count", -1)) != 0
                for payload in (point_metadata, summary, loading)
            ):
                raise ValueError(f"migrated completed point has zero-flux count: {point.slug}")
            if loading.get("zero_flux_quadrature_anchor_used") is not False or loading.get(
                "zero_speed_cross_section_imputed"
            ) is not False:
                raise ValueError(f"migrated loading flags mismatch: {point.slug}")
            if int(loading.get("capture_spectrum_row_count", -1)) != 121:
                raise ValueError(f"migrated spectrum-row count mismatch: {point.slug}")
        elif point_paths.final_samples_csv.exists():
            raise ValueError("partial destination point gained final samples")

    for study_key in campaign.STUDY_ORDER:
        aggregate_path = paths.aggregate_csv(study_key)
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(campaign.AGGREGATE_FIELDNAMES):
                raise ValueError(f"destination {study_key} aggregate schema mismatch")
            rows = list(reader)
        for row in rows:
            key = (study_key, int(row["point_index"]))
            if row.get("run_signature_sha256") != signatures.get(key):
                raise ValueError(f"destination aggregate signature mismatch: {key}")
            if int(row.get("indeterminate_zero_flux_ray_count", -1)) != 0:
                raise ValueError(f"destination aggregate synthetic zero-flux result: {key}")
        sweep = json.loads(paths.study_metadata_json(study_key).read_text(encoding="utf-8"))
        if sweep.get("schema_version") != DESTINATION_CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"destination {study_key} sweep schema mismatch")
        if int(sweep.get("indeterminate_zero_flux_ray_count", -1)) != 0:
            raise ValueError(f"destination {study_key} sweep zero-flux count mismatch")
    _validate_force_outputs(staged_statistics, staged_figures)
    _validate_prior_migration_manifest(staged_statistics)
    v3_to_v4._validate_delayed_capture_diagnostic(staged_statistics)
    _validate_zero_flux_diagnostic(staged_statistics)


def _migration_manifest(
    verified: VerifiedCampaign,
    destination_hashes: Mapping[str, str],
    destination_total_bytes: int,
    signatures: Mapping[tuple[str, int], str],
) -> dict[str, object]:
    if set(verified.source_file_hashes) != set(destination_hashes):
        raise ValueError("manifest source/destination inventories differ")
    files = []
    for key in sorted(verified.source_file_hashes):
        source_hash = verified.source_file_hashes[key]
        destination_hash = destination_hashes[key]
        files.append(
            {
                "logical_path": key,
                "method": "copy2",
                "source_sha256": source_hash,
                "destination_sha256": destination_hash,
                "transformed_after_copy": source_hash != destination_hash,
            }
        )
    return {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "status": "completed",
        "created_utc": campaign._utc_now(),
        "copy_method": "copy2 for every source file; no hard links or aliases",
        "source_campaign": {
            "name": SOURCE_CAMPAIGN_NAME,
            "schema_version": SOURCE_CAMPAIGN_SCHEMA_VERSION,
            "campaign_signature_sha256": SOURCE_CAMPAIGN_SIGNATURE_SHA256,
            "geometry_sha256": SOURCE_GEOMETRY_SHA256,
            "physics_hash_map_sha256": SOURCE_PHYSICS_HASH_MAP_SHA256,
            "file_count": len(verified.source_file_hashes),
            "total_bytes": verified.source_total_bytes,
        },
        "destination_campaign": {
            "name": DESTINATION_CAMPAIGN_NAME,
            "schema_version": DESTINATION_CAMPAIGN_SCHEMA_VERSION,
            "campaign_signature_sha256": campaign._campaign_signature(
                campaign.default_search_config(), SOURCE_GEOMETRY_SHA256
            ),
            "file_count_excluding_new_manifest": len(destination_hashes),
            "total_bytes_excluding_new_manifest": destination_total_bytes,
        },
        "prior_migration": {
            "manifest": PRIOR_MIGRATION_MANIFEST,
            "sha256": PRIOR_MIGRATION_MANIFEST_SHA256,
            "preserved_byte_for_byte": True,
        },
        "checkpoint_counts": {
            "point_directories": len(verified.points),
            "completed_points": sum(item.status == "completed" for item in verified.points),
            "partial_points": sum(item.status != "completed" for item in verified.points),
            "complete_samples": sum(
                item.completed_sample_count
                for item in verified.points
                if item.status == "completed"
            ),
            "partial_samples": sum(
                item.completed_sample_count
                for item in verified.points
                if item.status != "completed"
            ),
            "total_samples": sum(item.completed_sample_count for item in verified.points),
        },
        "point_signatures": [
            {
                "study_key": item.point.study_key,
                "point_index": item.point.point_index,
                "point_slug": item.point.slug,
                "status": item.status,
                "completed_sample_count": item.completed_sample_count,
                "source_signature_sha256": item.source_signature_sha256,
                "destination_signature_sha256": signatures[
                    (item.point.study_key, item.point.point_index)
                ],
            }
            for item in verified.points
        ],
        "zero_flux_policy_rationale": {
            "ray": {
                "detuning_n": -4.25,
                "disc_index": 7,
                "point_index": 20,
                "incident_speed_m_per_s": 0.0,
            },
            "physical_capture_classification": "indeterminate",
            "loading_classification": "indeterminate_zero_flux",
            "diagnostic_directory": ZERO_FLUX_DIAGNOSTIC_DIRECTORY,
            "diagnostic_json_bytes": ZERO_FLUX_DIAGNOSTIC_JSON_BYTES,
            "diagnostic_json_sha256": ZERO_FLUX_DIAGNOSTIC_JSON_SHA256,
            "diagnostic_readme_bytes": ZERO_FLUX_DIAGNOSTIC_README_BYTES,
            "diagnostic_readme_sha256": ZERO_FLUX_DIAGNOSTIC_README_SHA256,
            "quadrature_rule": (
                "sigma_capture(0) remains undefined; the full velocity grid retains "
                "the exact kinematic anchor g(0)=0"
            ),
        },
        "validation": {
            "source_identity": "exact fixed schema/name/campaign signature",
            "source_point_signatures": "canonical SHA-256 recomputed for every payload",
            "scientific_ledgers": "all saved sample/audit/override ledgers validated",
            "transformed_file_whitelist": "all unlisted files remained byte-identical",
            "source_immutability": "full source-tree SHA-256 map rechecked before commit",
            "prior_provenance": "v3-to-v4 manifest retained byte-for-byte",
        },
        "files": files,
    }


def migrate_campaign(
    *,
    project_root: Path,
    source_statistics: Path | None = None,
    source_figures: Path | None = None,
    destination_statistics: Path | None = None,
    destination_figures: Path | None = None,
) -> Path:
    """Copy, authenticate, re-sign, and atomically publish v4 as schema-6 v5."""

    _active_destination_contract()
    root = project_root.resolve()
    source_statistics = (
        source_statistics
        or root / "outputs" / "statistics" / "mot_simple" / SOURCE_CAMPAIGN_NAME
    ).resolve()
    source_figures = (
        source_figures
        or root / "outputs" / "figures" / "mot_simple" / SOURCE_CAMPAIGN_NAME
    ).resolve()
    destination_statistics = (
        destination_statistics
        or root / "outputs" / "statistics" / "mot_simple" / DESTINATION_CAMPAIGN_NAME
    ).resolve()
    destination_figures = (
        destination_figures
        or root / "outputs" / "figures" / "mot_simple" / DESTINATION_CAMPAIGN_NAME
    ).resolve()
    if source_statistics.name != SOURCE_CAMPAIGN_NAME or source_figures.name != SOURCE_CAMPAIGN_NAME:
        raise ValueError("source roots must use the exact v4 campaign name")
    if destination_statistics.name != DESTINATION_CAMPAIGN_NAME or destination_figures.name != DESTINATION_CAMPAIGN_NAME:
        raise ValueError("destination roots must use the exact v5 campaign name")
    for destination in (destination_statistics, destination_figures):
        if destination.exists():
            raise FileExistsError(f"migration destination already exists: {destination}")

    token = hashlib.sha256(str(destination_statistics).encode("utf-8")).hexdigest()[:12]
    # Add process-specific entropy without importing a shell or using a shared
    # predictable staging directory.
    import uuid

    token += uuid.uuid4().hex
    staged_statistics = destination_statistics.with_name(
        f".{destination_statistics.name}.migration-{token}"
    )
    staged_figures = destination_figures.with_name(
        f".{destination_figures.name}.migration-{token}"
    )

    source_hashes, source_bytes = v3_to_v4._hash_tree_pair(
        source_statistics, source_figures, label="source v4"
    )
    verified = validate_source_campaign(
        source_statistics,
        source_figures,
        source_file_hashes=source_hashes,
        source_total_bytes=source_bytes,
    )
    try:
        v3_to_v4._copy_tree_pair(
            source_statistics, source_figures, staged_statistics, staged_figures
        )
        raw_hashes, _ = v3_to_v4._hash_tree_pair(
            staged_statistics, staged_figures, label="raw staged v4 copy"
        )
        if raw_hashes != source_hashes:
            raise ValueError("raw copy2 destination differs from authenticated v4")
        signatures = _resign_destination(
            source_statistics,
            source_figures,
            staged_statistics,
            staged_figures,
            destination_statistics,
            destination_figures,
            verified,
        )
        _validate_destination(
            staged_statistics,
            staged_figures,
            destination_statistics,
            destination_figures,
            verified,
            signatures,
        )
        destination_hashes, destination_bytes = v3_to_v4._hash_tree_pair(
            staged_statistics, staged_figures, label="validated staged v5"
        )
        _validate_transformation_whitelist(
            source_hashes, destination_hashes, verified
        )
        source_hashes_after, _ = v3_to_v4._hash_tree_pair(
            source_statistics, source_figures, label="source v4 immutability recheck"
        )
        if source_hashes_after != source_hashes:
            raise RuntimeError("source v4 changed during migration; refusing commit")
        manifest = _migration_manifest(
            verified, destination_hashes, destination_bytes, signatures
        )
        campaign._atomic_write_json(
            staged_statistics / NEW_MIGRATION_MANIFEST, manifest
        )

        destination_statistics.parent.mkdir(parents=True, exist_ok=True)
        destination_figures.parent.mkdir(parents=True, exist_ok=True)
        staged_figures.replace(destination_figures)
        try:
            staged_statistics.replace(destination_statistics)
        except BaseException:
            destination_figures.replace(staged_figures)
            raise
    finally:
        import shutil

        for staged in (staged_statistics, staged_figures):
            if staged.exists():
                shutil.rmtree(staged)
    manifest_path = destination_statistics / NEW_MIGRATION_MANIFEST
    print(f"[v4->v5] committed schema-6 campaign: {manifest_path}", flush=True)
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated one-time copy of the stopped simple-MOT v4 campaign to v5"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = migrate_campaign(project_root=args.project_root)
    print(json.dumps({"migration_manifest": str(result)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DESTINATION_CAMPAIGN_NAME",
    "SOURCE_CAMPAIGN_NAME",
    "SOURCE_CAMPAIGN_SIGNATURE_SHA256",
    "SOURCE_GEOMETRY_SHA256",
    "ZERO_FLUX_DIAGNOSTIC_DIRECTORY",
    "ZERO_FLUX_DIAGNOSTIC_JSON_SHA256",
    "ZERO_FLUX_DIAGNOSTIC_README_SHA256",
    "migrate_campaign",
    "validate_source_campaign",
]
