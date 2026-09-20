"""Authenticated one-time migration of the September 2026 simple-MOT campaign.

The schema-4 campaign stopped after a rare zero-speed trajectory remained
non-terminal at every configured adaptive duration.  A separate convergence
diagnostic subsequently demonstrated a second core entry at about 806 ms, so
schema 5 adds a bounded 1 s *node-only* audit level.  This module copies the
validated schema-4 checkpoint into a new schema-5 campaign root and re-signs
only metadata that is intentionally coupled to the campaign source/policy.

The migration is deliberately not part of normal resume handling.  It accepts
one exact source campaign identity, refuses pre-existing destinations, copies
every file with :func:`shutil.copy2`, validates all sample/audit/override
ledgers through the production campaign validator, and verifies that every
source-file SHA-256 is unchanged before committing the staged destination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import refined_relationship_campaign as campaign
from .power_loading_study import geometry_csv_text, geometry_rows, validate_checkpoint_samples
from .sampling import CaptureVelocitySample, load_capture_velocity_samples
from .simple_force_sweep import (
    DETUNING_N_VALUES as FORCE_DETUNING_N_VALUES,
    SimpleForceSweepNumerics,
    _resume_signature as force_resume_signature,
)
from .timeout_audit import (
    VelocityResolvedCaptureOverride,
    velocity_overrides_from_payload,
)


SOURCE_CAMPAIGN_NAME = (
    "refined_relationships_full_sphere_25x25_r15mm_"
    "27mW_reference_optimized_v3_20260906"
)
SOURCE_CAMPAIGN_SCHEMA_VERSION = 4
SOURCE_POINT_SCHEMA_VERSION = 4
SOURCE_CAMPAIGN_SIGNATURE_SHA256 = (
    "1a778bdb0f00659936b5f9beb93b4c5000abb0235d76aca950cdf10123257659"
)
SOURCE_GEOMETRY_SHA256 = (
    "02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17"
)
SOURCE_PHYSICS_HASH_MAP_SHA256 = (
    "26f1d5a5fbacf16424c4ddf816384145a1a37971223a87e7dcbf3e61ea4d19c1"
)
SOURCE_ENDPOINT_POLICY_ACCEPTANCE = (
    "Every actual timeout-contaminated speed in the base search and both saved "
    "endpoints are classified at 200 ms with 5 and 2.5 microsecond steps. The saved "
    "bracket is retained only when both endpoints are definitive and every timeout "
    "speed becomes escaped at both steps; otherwise complete instrumented dual-step "
    "searches must recover compatible definitive brackets. A censored or incompatible "
    "200 ms positive boundary is re-searched completely at 250 ms and, only if needed, "
    "400 ms. Every evaluated search node is persisted. Once a positive scalar search "
    "has failed its premise, an independent full loading-grid scan is retained as the "
    "authoritative boolean mask even when monotone. Every scalar zero threshold is "
    "independently scanned from 0 to 30 m/s by 0.25 m/s at both timesteps, with "
    "finite-speed capture islands retained as direct boolean masks. Only non-definitive "
    "or timestep-disagreeing grid or positive-boundary nodes escalate to 250 ms at "
    "5/2.5 microseconds and then, only if still required, to 400 ms at 2.5/1.25 "
    "microseconds. A node is accepted only when both timesteps give the same definitive "
    "trapped/escaped classification."
)
SOURCE_COMPLETED_POINT_COUNT = 57
SOURCE_PARTIAL_STUDY_KEY = campaign.DETUNING_STUDY_KEY
SOURCE_PARTIAL_POINT_INDEX = 13
SOURCE_PARTIAL_SAMPLE_COUNT = 620
EXPECTED_SOURCE_POINT_COUNT = 58
EXPECTED_SAMPLE_COUNT_PER_POINT = 625
MIGRATION_SCHEMA_VERSION = 1

DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY = (
    "diagnostics/zero_speed_delayed_capture_n_m3p75_disc23_point15"
)
DELAYED_CAPTURE_README_SHA256 = (
    "711561b695a8844559a6a30c6171228f4f05f1632a58dffc961c0e3c9034ac8c"
)
DELAYED_CAPTURE_JSON_SHA256 = (
    "26c163d13902f53297211273221b52a5701b64beeb34135403e73a923b811b51"
)


@dataclass(frozen=True, slots=True)
class VerifiedPoint:
    """One authenticated and ledger-validated schema-4 point checkpoint."""

    point: campaign.RelationshipPoint
    source_signature_sha256: str
    status: str
    completed_sample_count: int


@dataclass(frozen=True, slots=True)
class VerifiedCampaign:
    """Validated source campaign state needed for the staged migration."""

    points: tuple[VerifiedPoint, ...]
    source_physics_hashes: Mapping[str, str]
    source_file_hashes: Mapping[str, str]
    source_total_bytes: int


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_files(statistics: Path, figures: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for label, root in (("statistics", statistics), ("figures", figures)):
        if not root.is_dir():
            raise FileNotFoundError(f"missing {label} campaign root: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"migration refuses symbolic links: {path}")
            if path.is_file():
                key = f"{label}/{path.relative_to(root).as_posix()}"
                files[key] = path
    return files


def _hash_tree_pair(
    statistics: Path,
    figures: Path,
    *,
    label: str,
) -> tuple[dict[str, str], int]:
    files = _logical_files(statistics, figures)
    hashes: dict[str, str] = {}
    total_bytes = 0
    count = len(files)
    print(f"[migration] hashing {label}: {count} file(s)", flush=True)
    for index, (key, path) in enumerate(files.items(), start=1):
        hashes[key] = _sha256_file(path)
        total_bytes += path.stat().st_size
        if index == count or index % 25 == 0:
            print(
                f"[migration] hashing {label}: {index}/{count} files, "
                f"{total_bytes / (1024**3):.3f} GiB",
                flush=True,
            )
    return hashes, total_bytes


def _require_exact_mapping(
    actual: object,
    expected: Mapping[str, object],
    *,
    description: str,
) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ValueError(f"{description} mismatch")


def _json_equivalent(left: object, right: object) -> bool:
    """Compare JSON-compatible values while normalizing tuples to arrays."""

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _validate_source_campaign_identity(metadata: Mapping[str, object]) -> None:
    """Refuse any source other than the exact stopped schema-4 campaign."""

    checks = (
        (metadata.get("schema_version"), SOURCE_CAMPAIGN_SCHEMA_VERSION, "schema"),
        (metadata.get("campaign_name"), SOURCE_CAMPAIGN_NAME, "name"),
        (
            metadata.get("campaign_signature_sha256"),
            SOURCE_CAMPAIGN_SIGNATURE_SHA256,
            "signature",
        ),
        (
            metadata.get("common_geometry_sha256"),
            SOURCE_GEOMETRY_SHA256,
            "geometry",
        ),
        (metadata.get("loading_point_count"), 67, "point count"),
        (metadata.get("capture_threshold_search_count"), 41_875, "ray count"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"source campaign {label} mismatch")
    if metadata.get("execution_order") != list(campaign.STUDY_ORDER):
        raise ValueError("source campaign execution order mismatch")
    if metadata.get("phase_space") != "full_sphere":
        raise ValueError("source campaign is not full-sphere")
    if metadata.get("stage_status") != {
        campaign.RAW_STUDY_KEY: "completed",
        campaign.EFFECTIVE_STUDY_KEY: "completed",
        campaign.DETUNING_STUDY_KEY: "failed",
    }:
        raise ValueError("source campaign stage status is not the known stopped state")
    _require_exact_mapping(
        metadata.get("search_config"),
        asdict(campaign.default_search_config()),
        description="source campaign search configuration",
    )
    expected_grids = {
        "s0": list(campaign.RAW_SATURATION_VALUES),
        "s_eff": list(campaign.EFFECTIVE_SATURATION_VALUES),
        "detuning_delta_over_gamma": list(campaign.DETUNING_N_VALUES),
    }
    _require_exact_mapping(
        metadata.get("requested_grids"),
        expected_grids,
        description="source campaign requested grids",
    )


def _validate_source_point_signature(
    metadata: Mapping[str, object],
    point: campaign.RelationshipPoint,
) -> tuple[str, dict[str, str]]:
    """Validate the stored schema-4 payload and its top-level SHA-256."""

    signature = metadata.get("run_signature_sha256")
    payload = metadata.get("signature_payload")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError(f"source point {point.slug} has no valid signature string")
    if not isinstance(payload, Mapping):
        raise ValueError(f"source point {point.slug} has no signature payload")
    if _canonical_hash(payload) != signature:
        raise ValueError(f"source point {point.slug} signature payload was tampered")
    physics_hashes = payload.get("physics_source_sha256")
    if not isinstance(physics_hashes, Mapping) or not physics_hashes:
        raise ValueError(f"source point {point.slug} has no physics source hash map")
    normalized = {str(key): str(value) for key, value in physics_hashes.items()}
    if any(len(value) != 64 for value in normalized.values()):
        raise ValueError(f"source point {point.slug} has malformed physics hashes")
    if _canonical_hash(normalized) != SOURCE_PHYSICS_HASH_MAP_SHA256:
        raise ValueError(f"source point {point.slug} physics hash map mismatch")
    expected_payload = _expected_source_point_payload(point, normalized)
    if _canonical_hash(payload) != _canonical_hash(expected_payload):
        changed_keys = sorted(
            key
            for key in set(payload).union(expected_payload)
            if not _json_equivalent(payload.get(key), expected_payload.get(key))
        )
        raise ValueError(
            f"source point {point.slug} signed physics/configuration payload mismatch: "
            f"{changed_keys}"
        )
    if metadata.get("schema_version") != SOURCE_POINT_SCHEMA_VERSION:
        raise ValueError(f"source point {point.slug} top-level schema mismatch")
    expected_study_name = f"{SOURCE_CAMPAIGN_NAME}/{point.study_key}/{point.slug}"
    if metadata.get("study_name") != expected_study_name:
        raise ValueError(f"source point {point.slug} top-level study name mismatch")
    if metadata.get("geometry_sha256") != SOURCE_GEOMETRY_SHA256:
        raise ValueError(f"source point {point.slug} top-level geometry mismatch")
    _require_exact_mapping(
        metadata.get("relationship_point"),
        asdict(point),
        description=f"source point {point.slug} top-level plan",
    )
    duplicated_fields = {
        "apparatus_config": "apparatus_config",
        "simple_mot_config": "simple_mot_config",
        "coil_config": "coil_config",
        "capture_search_config": "search_config",
        "endpoint_timeout_policy": "endpoint_timeout_policy",
    }
    for metadata_key, payload_key in duplicated_fields.items():
        if not _json_equivalent(metadata.get(metadata_key), payload[payload_key]):
            raise ValueError(
                f"source point {point.slug} top-level {metadata_key} disagrees "
                "with its signed payload"
            )
    if metadata.get("model") != payload["model"] or metadata.get(
        "phase_space"
    ) != payload["phase_space"]:
        raise ValueError(
            f"source point {point.slug} top-level model/phase disagrees with signed payload"
        )
    if metadata.get("built_cooling_beam_count") != 6:
        raise ValueError(f"source point {point.slug} cooling-beam count mismatch")
    powers = np.asarray(metadata.get("built_cooling_beam_powers_w"), dtype=float)
    detunings = np.asarray(metadata.get("built_cooling_beam_detunings_hz"), dtype=float)
    if powers.shape != (6,) or not np.allclose(
        powers,
        point.cooling_power_w_per_beam,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(f"source point {point.slug} built cooling powers mismatch")
    if detunings.shape != (6,) or not np.allclose(
        detunings,
        point.cooling_detuning_hz,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(f"source point {point.slug} built cooling detunings mismatch")
    if metadata.get("repumper_included") is not False:
        raise ValueError(f"source point {point.slug} unexpectedly includes repumping")
    return signature, normalized


def _expected_source_point_payload(
    point: campaign.RelationshipPoint,
    source_physics_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Reconstruct the exact schema-4 signed payload from authoritative controls."""

    search = campaign.default_search_config()
    expected = campaign._point_signature_payload(
        point, search, SOURCE_GEOMETRY_SHA256
    )
    expected["schema_version"] = SOURCE_POINT_SCHEMA_VERSION
    expected["physics_source_sha256"] = dict(source_physics_hashes)
    policy = dict(expected["endpoint_timeout_policy"])
    policy["adaptive_node_levels"] = policy["adaptive_node_levels"][:2]
    policy.pop("complete_positive_boundary_search_levels", None)
    policy["acceptance"] = SOURCE_ENDPOINT_POLICY_ACCEPTANCE
    expected["endpoint_timeout_policy"] = policy
    return expected


def _load_and_validate_ledgers(
    point_paths,
    launch_points,
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
        raise FileNotFoundError(f"point checkpoint has no sample ledger: {point_paths.statistics}")
    samples = validate_checkpoint_samples(
        load_capture_velocity_samples(sample_path), launch_points
    )
    audit_rows = campaign._load_audit_rows(campaign._audit_path(point_paths))
    if set(audit_rows) != set(samples):
        raise ValueError("endpoint-audit checkpoint keys do not match sample keys")
    override_path = campaign._overrides_path(point_paths)
    if not override_path.is_file():
        raise FileNotFoundError(f"point checkpoint has no override ledger: {override_path}")
    raw_overrides = json.loads(override_path.read_text(encoding="utf-8"))
    overrides = {
        item.key: item for item in velocity_overrides_from_payload(raw_overrides)
    }
    campaign._validate_completed_audit_ledger(
        samples, audit_rows, overrides, search=search
    )
    return samples, audit_rows, overrides


def _expected_imported_points() -> tuple[campaign.RelationshipPoint, ...]:
    groups = campaign.requested_points()
    return (
        *groups[campaign.RAW_STUDY_KEY],
        *groups[campaign.EFFECTIVE_STUDY_KEY],
        *groups[campaign.DETUNING_STUDY_KEY][: SOURCE_PARTIAL_POINT_INDEX + 1],
    )


def _validate_source_aggregates(
    statistics: Path,
    figures: Path,
    verified: Mapping[tuple[str, int], VerifiedPoint],
) -> None:
    completed_by_study = {
        campaign.RAW_STUDY_KEY: 24,
        campaign.EFFECTIVE_STUDY_KEY: 20,
        campaign.DETUNING_STUDY_KEY: 13,
    }
    points_by_study = campaign.requested_points()
    for study_key, completed_count in completed_by_study.items():
        aggregate_path = statistics / study_key / "aggregate.csv"
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != completed_count:
            raise ValueError(f"source {study_key} aggregate row count mismatch")
        for expected_index, row in enumerate(rows):
            key = (study_key, expected_index)
            if int(row["point_index"]) != expected_index or row["status"] != "completed":
                raise ValueError(f"source {study_key} aggregate ordering/status mismatch")
            if row["run_signature_sha256"] != verified[key].source_signature_sha256:
                raise ValueError(f"source {study_key} aggregate signature mismatch")
            if row["geometry_sha256"] != SOURCE_GEOMETRY_SHA256:
                raise ValueError(f"source {study_key} aggregate geometry mismatch")
        sweep_path = statistics / study_key / "sweep_metadata.json"
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        if sweep.get("schema_version") != SOURCE_CAMPAIGN_SCHEMA_VERSION:
            raise ValueError(f"source {study_key} sweep schema mismatch")
        if sweep.get("study_key") != study_key:
            raise ValueError(f"source {study_key} sweep identity mismatch")
        if int(sweep.get("point_count_completed", -1)) != completed_count:
            raise ValueError(f"source {study_key} sweep completed count mismatch")
        if sweep.get("ordered_point_plan") != [
            asdict(point) for point in points_by_study[study_key]
        ]:
            raise ValueError(f"source {study_key} sweep plan mismatch")
        relationship_plot = campaign.CampaignPaths(
            statistics=statistics, figures=figures
        ).relationship_plot(study_key)
        if not relationship_plot.is_file() or relationship_plot.stat().st_size == 0:
            raise FileNotFoundError(f"source relationship plot missing: {relationship_plot}")


def _validate_force_outputs(statistics: Path, figures: Path) -> None:
    stats = statistics / "04_force_vs_detuning_27mW"
    figs = figures / "04_force_vs_detuning_27mW"
    metadata_path = stats / "force_vs_detuning_metadata.json"
    csv_path = stats / "force_vs_detuning.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_signature = force_resume_signature(
        FORCE_DETUNING_N_VALUES, SimpleForceSweepNumerics()
    )
    if metadata.get("status") != "completed" or metadata.get(
        "resume_signature"
    ) != expected_signature:
        raise ValueError("source force sweep identity or status mismatch")
    if metadata.get("all_convergence_checks_passed") is not True:
        raise ValueError("source force sweep did not pass convergence")
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(FORCE_DETUNING_N_VALUES):
        raise ValueError("source force sweep row count mismatch")
    actual_grid = np.asarray([float(row["detuning_n"]) for row in rows])
    if not np.array_equal(actual_grid, np.asarray(FORCE_DETUNING_N_VALUES)):
        raise ValueError("source force sweep detuning grid mismatch")
    if not all(str(row["all_converged"]).strip().lower() == "true" for row in rows):
        raise ValueError("source force sweep contains a failed convergence row")
    for filename in (
        "restoring_slope_vs_detuning.png",
        "damping_turnaround_vs_detuning.png",
    ):
        path = figs / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"source force figure missing: {path}")


def _validate_delayed_capture_diagnostic(statistics: Path) -> None:
    directory = statistics / DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY
    readme = directory / "README.md"
    data_path = directory / "diagnostic.json"
    if _sha256_file(readme) != DELAYED_CAPTURE_README_SHA256:
        raise ValueError("delayed-capture README hash mismatch")
    if _sha256_file(data_path) != DELAYED_CAPTURE_JSON_SHA256:
        raise ValueError("delayed-capture diagnostic JSON hash mismatch")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    ray = data.get("ray")
    if not isinstance(ray, Mapping) or (
        ray.get("disc_index"),
        ray.get("point_index"),
        ray.get("geometry_seed"),
        ray.get("geometry_sha256"),
    ) != (23, 15, campaign.DEFAULT_SEED, SOURCE_GEOMETRY_SHA256):
        raise ValueError("delayed-capture diagnostic ray identity mismatch")
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
    ) != (-3.75, 0.027, True, 0.002, expected_criterion):
        raise ValueError("delayed-capture diagnostic physics/criterion mismatch")
    terminal = data.get("terminal_pair")
    if not isinstance(terminal, list) or len(terminal) != 2:
        raise ValueError("delayed-capture diagnostic terminal pair is malformed")
    expected_steps = (1.25e-6, 0.625e-6)
    for record, expected_step in zip(terminal, expected_steps, strict=True):
        if not isinstance(record, Mapping):
            raise ValueError("delayed-capture terminal record is malformed")
        if record.get("termination_reason") != "two_core_entries" or not np.isclose(
            float(record.get("time_step_s", np.nan)),
            expected_step,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("delayed-capture terminal classification mismatch")
        if not np.isclose(
            float(record.get("absolute_elapsed_time_s", np.nan)),
            0.806452,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("delayed-capture terminal time mismatch")
    conclusion = data.get("conclusion")
    if not isinstance(conclusion, Mapping) or conclusion.get("classification") != (
        "delayed capture"
    ):
        raise ValueError("delayed-capture diagnostic conclusion mismatch")
    if not np.isclose(
        float(conclusion.get("minimum_demonstrated_terminal_time_s", np.nan)),
        0.8064525,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("delayed-capture demonstrated terminal time mismatch")


def validate_source_campaign(
    source_statistics: Path,
    source_figures: Path,
    *,
    source_file_hashes: Mapping[str, str],
    source_total_bytes: int,
) -> VerifiedCampaign:
    """Fully authenticate the stopped v3 root and validate every saved ledger."""

    metadata_path = source_statistics / "campaign_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_source_campaign_identity(metadata)
    if _sha256_file(source_statistics / "launch_geometry.csv") != SOURCE_GEOMETRY_SHA256:
        raise ValueError("source common launch geometry bytes changed")

    search = campaign.default_search_config()
    discs, launch_points = campaign.generate_common_geometry(search)
    expected_geometry = geometry_csv_text(geometry_rows(discs, launch_points))
    if hashlib.sha256(expected_geometry.encode("utf-8")).hexdigest() != SOURCE_GEOMETRY_SHA256:
        raise RuntimeError("current geometry generator does not reproduce the source geometry")

    expected_points = _expected_imported_points()
    discovered_metadata = {
        path.relative_to(source_statistics).as_posix()
        for path in source_statistics.glob("0[123]_*/points/*/run_metadata.json")
    }
    expected_metadata = {
        f"{point.study_key}/points/{point.slug}/run_metadata.json"
        for point in expected_points
    }
    if discovered_metadata != expected_metadata:
        raise ValueError("source point-directory inventory mismatch")

    verified_points: list[VerifiedPoint] = []
    common_source_hashes: dict[str, str] | None = None
    print(
        f"[migration] validating {len(expected_points)} point signatures and ledgers",
        flush=True,
    )
    source_paths = campaign.CampaignPaths(source_statistics, source_figures)
    for ordinal, point in enumerate(expected_points, start=1):
        paths = source_paths.point_paths(point)
        point_metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        signature, source_hashes = _validate_source_point_signature(
            point_metadata, point
        )
        if common_source_hashes is None:
            common_source_hashes = source_hashes
        elif source_hashes != common_source_hashes:
            raise ValueError("source point physics hash maps are not identical")
        if _sha256_file(paths.geometry_csv) != SOURCE_GEOMETRY_SHA256:
            raise ValueError(f"source point {point.slug} launch geometry changed")
        samples, _, _ = _load_and_validate_ledgers(paths, launch_points, search)

        is_partial = (
            point.study_key == SOURCE_PARTIAL_STUDY_KEY
            and point.point_index == SOURCE_PARTIAL_POINT_INDEX
        )
        expected_status = "failed" if is_partial else "completed"
        expected_count = SOURCE_PARTIAL_SAMPLE_COUNT if is_partial else EXPECTED_SAMPLE_COUNT_PER_POINT
        if point_metadata.get("status") != expected_status:
            raise ValueError(f"source point {point.slug} status mismatch")
        if int(point_metadata.get("completed_sample_count", -1)) != expected_count:
            raise ValueError(f"source point {point.slug} metadata sample count mismatch")
        if len(samples) != expected_count:
            raise ValueError(f"source point {point.slug} ledger sample count mismatch")
        if int(point_metadata.get("expected_sample_count", -1)) != EXPECTED_SAMPLE_COUNT_PER_POINT:
            raise ValueError(f"source point {point.slug} expected sample count mismatch")
        if is_partial:
            if paths.final_samples_csv.exists():
                raise ValueError("known partial source point unexpectedly has final samples")
        else:
            if not paths.final_samples_csv.is_file():
                raise FileNotFoundError(f"completed source point lacks final samples: {point.slug}")
            capture_summary = json.loads(
                paths.capture_summary_json.read_text(encoding="utf-8")
            )
            loading_result = json.loads(paths.loading_json.read_text(encoding="utf-8"))
            duplicated_loading = (
                point_metadata.get("loading_rate"),
                capture_summary.get("loading_rate"),
                loading_result,
            )
            if capture_summary.get("run_signature_sha256") != signature:
                raise ValueError(
                    f"source point result signature mismatch: {paths.capture_summary_json}"
                )
            for label, result in zip(
                ("run metadata", "capture summary", "loading result"),
                duplicated_loading,
                strict=True,
            ):
                if not isinstance(result, Mapping) or result.get(
                    "run_signature_sha256"
                ) != signature:
                    raise ValueError(
                        f"source {label} loading signature mismatch for {point.slug}"
                    )
            if not all(
                _json_equivalent(duplicated_loading[0], result)
                for result in duplicated_loading[1:]
            ):
                raise ValueError(
                    f"source duplicated loading result mismatch for {point.slug}"
                )
        verified_points.append(
            VerifiedPoint(point, signature, expected_status, expected_count)
        )
        if ordinal == len(expected_points) or ordinal % 5 == 0:
            print(
                f"[migration] validated point ledgers: {ordinal}/{len(expected_points)}",
                flush=True,
            )

    if common_source_hashes is None:
        raise ValueError("source campaign contains no point source hash map")
    completed = sum(item.status == "completed" for item in verified_points)
    if completed != SOURCE_COMPLETED_POINT_COUNT:
        raise ValueError("source completed point count mismatch")
    verified_by_key = {
        (item.point.study_key, item.point.point_index): item
        for item in verified_points
    }
    _validate_source_aggregates(
        source_statistics, source_figures, verified_by_key
    )
    _validate_force_outputs(source_statistics, source_figures)
    _validate_delayed_capture_diagnostic(source_statistics)
    return VerifiedCampaign(
        points=tuple(verified_points),
        source_physics_hashes=common_source_hashes,
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
        return {
            key: _replace_strings(item, replacements) for key, item in value.items()
        }
    return value


def _stale_v3_references(
    value: object,
    *,
    location: tuple[str, ...] = (),
) -> list[str]:
    """Return stale v3 references while allowing explicit migration provenance."""

    if isinstance(value, str):
        if SOURCE_CAMPAIGN_NAME not in value:
            return []
        if (
            len(location) >= 2
            and location[-2] == "migration"
            and location[-1] == "source_campaign_name"
            and value == SOURCE_CAMPAIGN_NAME
        ):
            return []
        return [".".join(location) or "<root>"]
    if isinstance(value, list):
        stale: list[str] = []
        for index, item in enumerate(value):
            stale.extend(
                _stale_v3_references(item, location=(*location, str(index)))
            )
        return stale
    if isinstance(value, Mapping):
        stale = []
        for key, item in value.items():
            stale.extend(
                _stale_v3_references(item, location=(*location, str(key)))
            )
        return stale
    return []


def _path_replacements(
    source_statistics: Path,
    source_figures: Path,
    destination_statistics: Path,
    destination_figures: Path,
) -> tuple[tuple[str, str], ...]:
    pairs = [
        (str(source_statistics.resolve()), str(destination_statistics.resolve())),
        (str(source_figures.resolve()), str(destination_figures.resolve())),
        (SOURCE_CAMPAIGN_NAME, campaign.CAMPAIGN_NAME),
    ]
    # Stored products use both native absolute paths and relative Windows paths.
    expanded: list[tuple[str, str]] = []
    for old, new in pairs:
        expanded.extend(((old, new), (old.replace("\\", "/"), new.replace("\\", "/"))))
    return tuple(expanded)


def _copy_tree_pair(
    source_statistics: Path,
    source_figures: Path,
    staged_statistics: Path,
    staged_figures: Path,
) -> None:
    source_files = _logical_files(source_statistics, source_figures)
    copied_count = 0
    copied_bytes = 0

    def copy_with_progress(source: str, destination: str) -> str:
        nonlocal copied_count, copied_bytes
        result = shutil.copy2(source, destination)
        copied_count += 1
        copied_bytes += Path(source).stat().st_size
        if copied_count == len(source_files) or copied_count % 25 == 0:
            print(
                f"[migration] copy2: {copied_count}/{len(source_files)} files, "
                f"{copied_bytes / (1024**3):.3f} GiB",
                flush=True,
            )
        return result

    print("[migration] copying statistics with copy2", flush=True)
    shutil.copytree(
        source_statistics, staged_statistics, copy_function=copy_with_progress
    )
    print("[migration] copying figures with copy2", flush=True)
    shutil.copytree(source_figures, staged_figures, copy_function=copy_with_progress)


def _rewrite_json(path: Path, replacements: Sequence[tuple[str, str]]) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewritten = _replace_strings(payload, replacements)
    if not isinstance(rewritten, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    campaign._atomic_write_json(path, rewritten)
    return rewritten


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
        # Replacements must name the original and final roots, not staging roots.
        source_statistics,
        source_figures,
        final_statistics,
        final_figures,
    )
    search = campaign.default_search_config()
    new_paths = campaign.CampaignPaths(staged_statistics, staged_figures)
    signatures: dict[tuple[str, int], str] = {}
    migrated_utc = campaign._utc_now()
    for item in verified.points:
        point = item.point
        paths = new_paths.point_paths(point)
        new_payload = campaign._point_signature_payload(
            point, search, SOURCE_GEOMETRY_SHA256
        )
        new_signature = campaign._signature(new_payload)
        signatures[(point.study_key, point.point_index)] = new_signature
        point_replacements = (
            *replacements,
            (item.source_signature_sha256, new_signature),
        )
        metadata = _rewrite_json(paths.metadata_json, point_replacements)
        metadata.update(
            {
                "schema_version": campaign.POINT_SCHEMA_VERSION,
                "study_name": f"{campaign.CAMPAIGN_NAME}/{point.study_key}/{point.slug}",
                "run_signature_sha256": new_signature,
                "signature_payload": new_payload,
                "endpoint_timeout_policy": new_payload["endpoint_timeout_policy"],
                "migration": {
                    "schema_version": MIGRATION_SCHEMA_VERSION,
                    "migrated_utc": migrated_utc,
                    "source_campaign_name": SOURCE_CAMPAIGN_NAME,
                    "source_point_signature_sha256": item.source_signature_sha256,
                    "reason": (
                        "add bounded 1 s node-only dual-timestep audit level after "
                        "independent delayed-capture convergence evidence"
                    ),
                    "diagnostic_reference": DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY
                    + "/README.md",
                },
            }
        )
        campaign._atomic_write_json(paths.metadata_json, metadata)
        if item.status == "completed":
            for result_path in (paths.loading_json, paths.capture_summary_json):
                _rewrite_json(result_path, point_replacements)

    for study_key in campaign.STUDY_ORDER:
        aggregate_path = new_paths.aggregate_csv(study_key)
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if fieldnames != list(campaign.AGGREGATE_FIELDNAMES):
            raise ValueError(f"destination {study_key} aggregate columns changed")
        for row in rows:
            key = (study_key, int(row["point_index"]))
            row["run_signature_sha256"] = signatures[key]
            for field in (
                "statistics_directory",
                "figures_directory",
                "capture_cross_section_csv",
                "capture_cross_section_plot",
            ):
                row[field] = _replace_strings(row[field], replacements)
        campaign._atomic_write_csv(
            aggregate_path, rows, campaign.AGGREGATE_FIELDNAMES
        )
        sweep_path = new_paths.study_metadata_json(study_key)
        sweep = _rewrite_json(sweep_path, replacements)
        sweep["schema_version"] = campaign.CAMPAIGN_SCHEMA_VERSION
        sweep["migration"] = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "migrated_utc": migrated_utc,
            "source_campaign_name": SOURCE_CAMPAIGN_NAME,
        }
        campaign._atomic_write_json(sweep_path, sweep)

    campaign_metadata = _rewrite_json(new_paths.metadata_json, replacements)
    campaign_metadata.update(
        {
            "schema_version": campaign.CAMPAIGN_SCHEMA_VERSION,
            "campaign_name": campaign.CAMPAIGN_NAME,
            "campaign_signature_sha256": campaign._campaign_signature(
                search, SOURCE_GEOMETRY_SHA256
            ),
            "migration": {
                "schema_version": MIGRATION_SCHEMA_VERSION,
                "migrated_utc": migrated_utc,
                "source_campaign_name": SOURCE_CAMPAIGN_NAME,
                "source_campaign_signature_sha256": SOURCE_CAMPAIGN_SIGNATURE_SHA256,
                "reason": (
                    "schema 5 adds a bounded 1 s node-only dual-timestep audit "
                    "level; all schema-4 scientific ledgers were validated before copy"
                ),
                "diagnostic_reference": DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY
                + "/README.md",
            },
        }
    )
    campaign._atomic_write_json(new_paths.metadata_json, campaign_metadata)

    force_metadata_path = (
        staged_statistics
        / "04_force_vs_detuning_27mW"
        / "force_vs_detuning_metadata.json"
    )
    _rewrite_json(force_metadata_path, replacements)
    return signatures


def _validate_destination(
    staged_statistics: Path,
    staged_figures: Path,
    final_statistics: Path,
    final_figures: Path,
    verified: VerifiedCampaign,
    signatures: Mapping[tuple[str, int], str],
) -> None:
    search = campaign.default_search_config()
    discs, launch_points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, launch_points))
    if _sha256_file(staged_statistics / "launch_geometry.csv") != SOURCE_GEOMETRY_SHA256:
        raise ValueError("destination common geometry changed during migration")
    paths = campaign.CampaignPaths(staged_statistics, staged_figures)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != campaign.CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("destination campaign schema mismatch")
    if metadata.get("campaign_name") != campaign.CAMPAIGN_NAME:
        raise ValueError("destination campaign name mismatch")
    if metadata.get("campaign_signature_sha256") != campaign._campaign_signature(
        search, SOURCE_GEOMETRY_SHA256
    ):
        raise ValueError("destination campaign signature mismatch")
    migration_record = metadata.get("migration")
    if not isinstance(migration_record, Mapping) or (
        migration_record.get("source_campaign_name"),
        migration_record.get("source_campaign_signature_sha256"),
    ) != (SOURCE_CAMPAIGN_NAME, SOURCE_CAMPAIGN_SIGNATURE_SHA256):
        raise ValueError("destination campaign migration provenance mismatch")

    for item in verified.points:
        point = item.point
        point_paths = paths.point_paths(point)
        metadata = json.loads(point_paths.metadata_json.read_text(encoding="utf-8"))
        expected_payload = campaign._point_signature_payload(
            point, search, SOURCE_GEOMETRY_SHA256
        )
        expected_signature = campaign._signature(expected_payload)
        if expected_signature != signatures[(point.study_key, point.point_index)]:
            raise RuntimeError("internally inconsistent destination signature map")
        if metadata.get("run_signature_sha256") != expected_signature:
            raise ValueError(f"destination point {point.slug} signature mismatch")
        persisted_payload = metadata.get("signature_payload")
        if not isinstance(persisted_payload, Mapping) or campaign._signature(
            persisted_payload
        ) != expected_signature:
            raise ValueError(f"destination point {point.slug} signature payload mismatch")
        expected_study_name = f"{campaign.CAMPAIGN_NAME}/{point.study_key}/{point.slug}"
        if metadata.get("schema_version") != campaign.POINT_SCHEMA_VERSION or metadata.get(
            "study_name"
        ) != expected_study_name:
            raise ValueError(f"destination point {point.slug} identity mismatch")
        if metadata.get("geometry_sha256") != SOURCE_GEOMETRY_SHA256 or _sha256_file(
            point_paths.geometry_csv
        ) != SOURCE_GEOMETRY_SHA256:
            raise ValueError(f"destination point {point.slug} geometry mismatch")
        duplicated_fields = {
            "apparatus_config": "apparatus_config",
            "simple_mot_config": "simple_mot_config",
            "coil_config": "coil_config",
            "capture_search_config": "search_config",
            "endpoint_timeout_policy": "endpoint_timeout_policy",
            "relationship_point": "point",
        }
        for metadata_key, payload_key in duplicated_fields.items():
            if not _json_equivalent(
                metadata.get(metadata_key), persisted_payload[payload_key]
            ):
                raise ValueError(
                    f"destination point {point.slug} top-level {metadata_key} "
                    "disagrees with its signed payload"
                )
        if metadata.get("status") != item.status or int(
            metadata.get("completed_sample_count", -1)
        ) != item.completed_sample_count:
            raise ValueError(f"destination point {point.slug} checkpoint status changed")
        migration_record = metadata.get("migration")
        if not isinstance(migration_record, Mapping) or (
            migration_record.get("source_campaign_name"),
            migration_record.get("source_point_signature_sha256"),
        ) != (SOURCE_CAMPAIGN_NAME, item.source_signature_sha256):
            raise ValueError(f"destination point {point.slug} migration provenance mismatch")
        samples, _, _ = _load_and_validate_ledgers(
            point_paths, launch_points, search
        )
        if len(samples) != item.completed_sample_count:
            raise ValueError(f"destination point {point.slug} sample count changed")
        if item.status == "completed":
            loading_result = json.loads(
                point_paths.loading_json.read_text(encoding="utf-8")
            )
            capture_summary = json.loads(
                point_paths.capture_summary_json.read_text(encoding="utf-8")
            )
            duplicated_loading = (
                metadata.get("loading_rate"),
                capture_summary.get("loading_rate"),
                loading_result,
            )
            for label, result in zip(
                ("run metadata", "capture summary", "loading result"),
                duplicated_loading,
                strict=True,
            ):
                if not isinstance(result, Mapping) or result.get(
                    "run_signature_sha256"
                ) != expected_signature:
                    raise ValueError(
                        f"destination {label} was not re-signed for {point.slug}"
                    )
            if not all(
                _json_equivalent(duplicated_loading[0], result)
                for result in duplicated_loading[1:]
            ):
                raise ValueError(
                    f"destination duplicated loading result mismatch for {point.slug}"
                )

    for path in staged_statistics.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stale = _stale_v3_references(payload)
        if stale:
            raise ValueError(
                f"destination JSON retains a stale v3 reference at "
                f"{', '.join(stale[:3])}: {path}"
            )
    for study_key in campaign.STUDY_ORDER:
        aggregate_path = paths.aggregate_csv(study_key)
        aggregate_text = aggregate_path.read_text(encoding="utf-8")
        if SOURCE_CAMPAIGN_NAME in aggregate_text:
            raise ValueError(f"destination aggregate retains a v3 path: {study_key}")
        with aggregate_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            key = (study_key, int(row["point_index"]))
            if row.get("run_signature_sha256") != signatures.get(key):
                raise ValueError(
                    f"destination aggregate signature mismatch for {study_key}/{key[1]}"
                )
    _validate_force_outputs(staged_statistics, staged_figures)
    _validate_serialized_final_roots(
        paths.metadata_json, final_statistics, final_figures
    )
    if hashlib.sha256(geometry_text.encode("utf-8")).hexdigest() != SOURCE_GEOMETRY_SHA256:
        raise RuntimeError("current generated destination geometry changed unexpectedly")


def _validate_serialized_final_roots(
    metadata_path: Path,
    final_statistics: Path,
    final_figures: Path,
) -> None:
    """Require parsed metadata paths to name published roots, not staging roots."""

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "statistics_directory": final_statistics.resolve(),
        "figures_directory": final_figures.resolve(),
    }
    for field, expected_path in expected.items():
        raw_path = payload.get(field)
        if not isinstance(raw_path, str) or Path(raw_path).resolve() != expected_path:
            raise ValueError(
                f"destination campaign metadata {field} does not name final root"
            )


def _migration_manifest(
    verified: VerifiedCampaign,
    destination_hashes: Mapping[str, str],
    destination_total_bytes: int,
    signatures: Mapping[tuple[str, int], str],
) -> dict[str, object]:
    if set(verified.source_file_hashes) != set(destination_hashes):
        raise ValueError("source/destination file inventories differ before manifest")
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
            "name": campaign.CAMPAIGN_NAME,
            "schema_version": campaign.CAMPAIGN_SCHEMA_VERSION,
            "campaign_signature_sha256": campaign._campaign_signature(
                campaign.default_search_config(), SOURCE_GEOMETRY_SHA256
            ),
            "file_count_excluding_manifest": len(destination_hashes),
            "total_bytes_excluding_manifest": destination_total_bytes,
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
        "delayed_capture_policy_rationale": {
            "ray": {"detuning_n": -3.75, "disc_index": 23, "point_index": 15},
            "classification": "delayed capture by two core entries",
            "first_demonstrated_terminal_time_s": 0.8064525,
            "new_node_only_level": {
                "duration_s": 1.0,
                "coarse_time_step_s": 1.25e-6,
                "fine_time_step_s": 0.625e-6,
            },
            "readme": DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY + "/README.md",
            "data": DELAYED_CAPTURE_DIAGNOSTIC_DIRECTORY + "/diagnostic.json",
            "readme_sha256": DELAYED_CAPTURE_README_SHA256,
            "data_sha256": DELAYED_CAPTURE_JSON_SHA256,
        },
        "validation": {
            "source_identity": "exact fixed schema/name/campaign signature",
            "source_point_signatures": "canonical SHA-256 recomputed for every payload",
            "source_physics_hash_maps": "all identical and equal fixed expected digest",
            "scientific_ledgers": (
                "all sample, endpoint-audit, and velocity-override ledgers passed "
                "the schema-5 production validator before and after copy"
            ),
            "source_immutability": "full source-tree SHA-256 map rechecked before commit",
            "manifest_self_hash": "not included to avoid recursive self-reference",
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
    """Copy, validate, and atomically publish the exact v3 checkpoint as v4."""

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
        or root / "outputs" / "statistics" / "mot_simple" / campaign.CAMPAIGN_NAME
    ).resolve()
    destination_figures = (
        destination_figures
        or root / "outputs" / "figures" / "mot_simple" / campaign.CAMPAIGN_NAME
    ).resolve()
    if source_statistics.name != SOURCE_CAMPAIGN_NAME or source_figures.name != SOURCE_CAMPAIGN_NAME:
        raise ValueError("source roots must use the exact authenticated v3 campaign name")
    if destination_statistics.name != campaign.CAMPAIGN_NAME or destination_figures.name != campaign.CAMPAIGN_NAME:
        raise ValueError("destination roots must use the current v4 campaign name")
    for destination in (destination_statistics, destination_figures):
        if destination.exists():
            raise FileExistsError(f"migration destination already exists: {destination}")

    token = uuid.uuid4().hex
    staged_statistics = destination_statistics.with_name(
        f".{destination_statistics.name}.migration-{token}"
    )
    staged_figures = destination_figures.with_name(
        f".{destination_figures.name}.migration-{token}"
    )
    source_hashes, source_bytes = _hash_tree_pair(
        source_statistics, source_figures, label="source v3"
    )
    verified = validate_source_campaign(
        source_statistics,
        source_figures,
        source_file_hashes=source_hashes,
        source_total_bytes=source_bytes,
    )
    try:
        _copy_tree_pair(
            source_statistics, source_figures, staged_statistics, staged_figures
        )
        raw_copy_hashes, _ = _hash_tree_pair(
            staged_statistics, staged_figures, label="raw staged copy"
        )
        if raw_copy_hashes != source_hashes:
            raise ValueError("raw copy2 destination differs from authenticated source")

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
        destination_hashes, destination_bytes = _hash_tree_pair(
            staged_statistics, staged_figures, label="validated staged v4"
        )
        source_hashes_after, _ = _hash_tree_pair(
            source_statistics, source_figures, label="source v3 immutability recheck"
        )
        if source_hashes_after != source_hashes:
            raise RuntimeError("source v3 changed during migration; refusing commit")
        manifest = _migration_manifest(
            verified, destination_hashes, destination_bytes, signatures
        )
        campaign._atomic_write_json(
            staged_statistics / "migration_manifest.json", manifest
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
        for staged in (staged_statistics, staged_figures):
            if staged.exists():
                shutil.rmtree(staged)

    manifest_path = destination_statistics / "migration_manifest.json"
    print(f"[migration] committed schema-5 campaign: {manifest_path}", flush=True)
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated one-time copy of the stopped simple-MOT v3 campaign to v4"
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
    "SOURCE_CAMPAIGN_NAME",
    "SOURCE_CAMPAIGN_SIGNATURE_SHA256",
    "SOURCE_GEOMETRY_SHA256",
    "migrate_campaign",
    "validate_source_campaign",
]
