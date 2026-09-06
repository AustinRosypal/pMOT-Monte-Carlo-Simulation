"""Post-campaign convergence audit for finite-horizon capture timeouts.

The production loading search treats ``timeout`` as untrapped within its
configured trajectory horizon.  That is useful for a bounded production run,
but it is not equivalent to an escaped upper capture bracket.  This module
audits completed refined relationship points whose saved bracket endpoint is a
timeout. Raw saturation and detuning are selectable, independent studies. It
is deliberately separate from the live campaign runner.

Audit mode never changes production files.  ``--apply`` is required before a
resolved correction can be installed, and application is refused until the
entire refined campaign is complete.  The audit report is written before that
guard is evaluated, production hashes are rechecked, and recoverable backups
are made before any replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import shutil
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ..capture_statistics import (
    CaptureVelocitySample,
    TrajectoryClassification,
    load_capture_velocity_samples,
)
from ..configuration import AntiHelmholtzCoilConfig
from ..launch_geometry import PointSample
from .loading_sweeps import (
    classify_multilevel_loading_trajectory,
    find_multilevel_capture_velocity,
)
from .power_loading_study import (
    LOADING_BY_DISC_FIELDNAMES,
    SPECTRUM_FIELDNAMES,
    TRAPPED_TERMINATION_REASONS,
    StudyPaths,
    _atomic_write_csv,
    analyze_completed_samples,
    build_27mw_multilevel_configuration,
    generate_study_geometry,
    geometry_rows,
    geometry_sha256,
    plot_clustered_cross_section,
    plot_loading_rate_by_disc,
    save_samples_atomic,
    study_signature,
    validate_checkpoint_samples,
)
from .velocity_resolved_capture import (
    VelocityResolvedCaptureOverride,
    calculate_clustered_cross_section_with_overrides,
    calculate_disc_clustered_loading_with_overrides,
)
from .rate_capture import RateCaptureSearchConfig
from .rate_equations import build_rate_equation_model
from .refined_relationship_campaign import (
    default_refined_campaign_paths,
    plot_refined_loading_relationship,
    requested_refined_points,
)
from .relationship_sweeps import (
    DETUNING_STUDY_KEY,
    RAW_STUDY_KEY,
    CampaignPaths,
    RelationshipPoint,
    _atomic_write_text,
    _csv_text,
    _point_row,
)


SCHEMA_VERSION = 1
AUDIT_DURATION_S = 100.0e-3
COARSE_TIME_STEP_S = 5.0e-6
FINE_TIME_STEP_S = 2.5e-6
# Strictly less than the requested 0.25 m/s maximum bracket width.
AUDIT_VELOCITY_TOLERANCE_M_PER_S = 0.249
AUDIT_REPORT_NAME = "timeout_audit_report.json"
APPLY_INTENT_NAME = "timeout_apply_intent.json"
APPLY_RESULT_NAME = "timeout_apply_result.json"
CLI_STUDIES = {
    "detuning": DETUNING_STUDY_KEY,
    "raw-s0": RAW_STUDY_KEY,
}
MAX_AUDIT_WORKERS = 24


Classifier = Callable[..., TrajectoryClassification]
Rebisector = Callable[..., CaptureVelocitySample]


class ApplyTransactionError(RuntimeError):
    """An apply failed after backups were created."""

    def __init__(self, message: str, *, rollback_complete: bool) -> None:
        super().__init__(message)
        self.rollback_complete = rollback_complete


@dataclass(frozen=True, slots=True)
class TimeoutCase:
    """One saved timeout endpoint and the files needed to audit it."""

    point: RelationshipPoint
    paths: StudyPaths
    sample: CaptureVelocitySample
    run_metadata: Mapping[str, object]
    source_sample_sha256: str
    source_metadata_sha256: str
    endpoint_kind: str = "upper"


@dataclass(frozen=True, slots=True)
class TimeoutOutcome:
    """In-memory result for one timeout endpoint."""

    case: TimeoutCase
    status: str
    reason: str
    coarse_lower_result: TrajectoryClassification
    fine_lower_result: TrajectoryClassification
    coarse_upper_result: TrajectoryClassification
    fine_upper_result: TrajectoryClassification
    coarse_rebisected: CaptureVelocitySample | None
    fine_rebisected: CaptureVelocitySample | None
    replacement_sample: CaptureVelocitySample | None
    velocity_scan_m_per_s: tuple[float, ...] = ()
    coarse_velocity_scan: tuple[TrajectoryClassification, ...] = ()
    fine_velocity_scan: tuple[TrajectoryClassification, ...] = ()

    @property
    def safe_to_apply(self) -> bool:
        return self.replacement_sample is not None and self.status in {
            "confirmed_escaped",
            "confirmed_zero_capture",
            "rebisected",
            "velocity_resolved_capture",
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_signed_run(metadata: Mapping[str, object]) -> dict[str, str]:
    """Verify the stored run signature and every signed physics source file."""

    payload = metadata.get("run_signature_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("run metadata has no run_signature_payload")
    recorded_signature = metadata.get("run_signature_sha256")
    if not isinstance(recorded_signature, str):
        raise ValueError("run metadata has no run_signature_sha256")
    calculated_signature = study_signature(payload)
    if calculated_signature != recorded_signature:
        raise RuntimeError("stored run-signature payload does not match its SHA-256")

    recorded_sources = payload.get("physics_source_sha256")
    if not isinstance(recorded_sources, Mapping) or not recorded_sources:
        raise ValueError("run signature has no physics_source_sha256 mapping")
    source_root = Path(__file__).resolve().parent
    current_sources: dict[str, str] = {}
    for raw_name, raw_digest in sorted(recorded_sources.items()):
        name = str(raw_name)
        expected_digest = str(raw_digest)
        relative = Path(name)
        if relative.is_absolute() or relative.name != name:
            raise ValueError(f"invalid signed physics-source name: {name!r}")
        source = source_root / relative
        if not source.is_file():
            raise RuntimeError(f"signed physics source is missing: {source}")
        current_digest = _sha256(source)
        current_sources[name] = current_digest
        if current_digest != expected_digest:
            raise RuntimeError(
                "signed physics source changed since the production run: "
                f"{name} (recorded {expected_digest}, current {current_digest})"
            )
    return current_sources


def _audit_runtime_source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        "mot_multilevel/timeout_audit.py": Path(__file__).resolve(),
        "mot_multilevel/velocity_resolved_capture.py": package_root
        / "mot_multilevel"
        / "velocity_resolved_capture.py",
        "mot_multilevel/rate_capture.py": package_root
        / "mot_multilevel"
        / "rate_capture.py",
        "capture_statistics.py": package_root / "capture_statistics.py",
        "launch_geometry.py": package_root / "launch_geometry.py",
        "loading.py": package_root / "loading.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equivalent(expected, actual, *, location: str) -> None:
    """Recursively verify that a rebuilt signed configuration is unchanged."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"configuration keys differ at {location}")
        for key in expected:
            _assert_equivalent(
                expected[key], actual[key], location=f"{location}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f"configuration sequence differs at {location}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_equivalent(left, right, location=f"{location}[{index}]")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(f"configuration value differs at {location}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(float(expected), float(actual), rtol=0.0, atol=2.0e-14):
            raise ValueError(f"configuration value differs at {location}")
        return
    if expected != actual:
        raise ValueError(f"configuration value differs at {location}")


def _point_sample(sample: CaptureVelocitySample) -> PointSample:
    return PointSample(
        disc_index=sample.disc_index,
        point_index=sample.point_index,
        theta_rad=sample.theta_rad,
        phi_rad=sample.phi_rad,
        theta_prime_rad=sample.theta_prime_rad,
        s_m=sample.s_m,
        radial_distance_m=sample.radial_distance_m,
        initial_position_m=sample.initial_position_m,
        incident_unit_vector=sample.incident_unit_vector,
        launch_axis_unit_vector=sample.incident_unit_vector,
    )


def _production_search(metadata: Mapping[str, object]) -> RateCaptureSearchConfig:
    payload = metadata.get("capture_search_config")
    if not isinstance(payload, Mapping):
        raise ValueError("run metadata has no capture_search_config")
    return RateCaptureSearchConfig(**dict(payload))


def _validate_audit_search_parameters(
    audit_duration_s: float,
    coarse_time_step_s: float,
    fine_time_step_s: float,
) -> None:
    values = (audit_duration_s, coarse_time_step_s, fine_time_step_s)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("audit duration and timesteps must be finite and positive")
    if fine_time_step_s >= coarse_time_step_s:
        raise ValueError("fine audit timestep must be strictly smaller than coarse")


def _audit_searches(
    metadata: Mapping[str, object],
    audit_duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
) -> tuple[RateCaptureSearchConfig, RateCaptureSearchConfig]:
    _validate_audit_search_parameters(
        audit_duration_s, coarse_time_step_s, fine_time_step_s
    )
    production = _production_search(metadata)
    coarse = replace(
        production,
        max_simulation_time_s=audit_duration_s,
        time_step_s=coarse_time_step_s,
        velocity_tolerance_m_per_s=AUDIT_VELOCITY_TOLERANCE_M_PER_S,
        worker_count=1,
    )
    return coarse, replace(coarse, time_step_s=fine_time_step_s)


def _reconstruct_physics(case: TimeoutCase):
    """Rebuild and verify the exact signed production configuration."""

    metadata = case.run_metadata
    _verify_signed_run(metadata)
    config, apparatus, beams = build_27mw_multilevel_configuration(
        cooling_power_w_per_beam=float(metadata["cooling_power_w_per_beam"]),
        repump_power_w_per_beam=float(metadata["repump_power_w_per_beam"]),
        cooling_detuning_hz=float(metadata["cooling_detuning_hz"]),
    )
    coil_payload = metadata.get("coil_config")
    if not isinstance(coil_payload, Mapping):
        raise ValueError("run metadata has no coil_config")
    coil = AntiHelmholtzCoilConfig(**dict(coil_payload))
    _assert_equivalent(
        metadata["multilevel_config"],
        asdict(config),
        location="multilevel_config",
    )
    _assert_equivalent(
        metadata["apparatus_config"],
        asdict(apparatus),
        location="apparatus_config",
    )
    _assert_equivalent(coil_payload, asdict(coil), location="coil_config")
    expected_cooling = [float(metadata["cooling_power_w_per_beam"])] * 6
    expected_repump = [float(metadata["repump_power_w_per_beam"])] * 6
    actual_cooling = [beam.power_w for beam in beams if beam.family == "cooling"]
    actual_repump = [beam.power_w for beam in beams if beam.family == "repump"]
    if not np.allclose(actual_cooling, expected_cooling, rtol=0.0, atol=0.0):
        raise ValueError("rebuilt cooling powers differ from signed metadata")
    if not np.allclose(actual_repump, expected_repump, rtol=0.0, atol=0.0):
        raise ValueError("rebuilt repump powers differ from signed metadata")
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    if (model.state_count, model.ground_count, model.excited_count) != (24, 8, 16):
        raise ValueError("rebuilt model is not the authoritative 24-state model")
    return model, beams, coil, config


def _validate_completed_point_inputs(
    point: RelationshipPoint,
    point_paths: StudyPaths,
    metadata: Mapping[str, object],
    samples: Sequence[CaptureVelocitySample],
) -> None:
    """Validate signed physics, regenerated geometry, and the complete checkpoint."""

    if metadata.get("status") != "completed":
        raise ValueError(f"point {point.slug} is not completed")
    _verify_signed_run(metadata)
    search = _production_search(metadata)
    expected_count = search.disc_count * search.points_per_disc
    if expected_count != 25 * 25:
        raise ValueError(
            f"point {point.slug} is not the requested 25-disc by 25-point run"
        )
    if int(metadata.get("expected_sample_count", -1)) != expected_count:
        raise ValueError(f"point {point.slug} metadata has the wrong expected count")
    if int(metadata.get("completed_sample_count", -1)) != expected_count:
        raise ValueError(f"point {point.slug} metadata has the wrong completed count")
    if len(samples) != expected_count:
        raise ValueError(
            f"completed point {point.slug} has {len(samples)} of {expected_count} samples"
        )
    expected_search_values = {
        "disc_count": 25,
        "points_per_disc": 25,
        "disc_radius_m": 15.0e-3,
        "phase_space": "full_sphere",
        "analysis_velocity_step_m_per_s": 0.25,
        "analysis_velocity_max_m_per_s": 30.0,
    }
    for field, expected in expected_search_values.items():
        actual = getattr(search, field)
        if isinstance(expected, float):
            matches = np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-15)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"point {point.slug} capture search has {field}={actual!r}; "
                f"expected {expected!r}"
            )
    expected_metadata_values = {
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "repump_power_w_per_beam": 0.1e-3,
        "cooling_detuning_hz": point.cooling_detuning_hz,
    }
    for field, expected in expected_metadata_values.items():
        actual = float(metadata.get(field, np.nan))
        if not np.isclose(actual, expected, rtol=2.0e-13, atol=1.0e-15):
            raise ValueError(
                f"point {point.slug} metadata has {field}={actual!r}; "
                f"expected {expected!r}"
            )
    saturation = metadata.get("effective_saturation")
    cooling_saturation = (
        saturation.get("cooling") if isinstance(saturation, Mapping) else None
    )
    if not isinstance(cooling_saturation, Mapping):
        raise ValueError(f"point {point.slug} has no cooling saturation metadata")
    expected_saturation_values = {
        "beam_center_on_resonance_saturation_parameter": (
            point.on_resonance_saturation
        ),
        "beam_center_effective_saturation_parameter": point.effective_saturation,
    }
    for field, expected in expected_saturation_values.items():
        actual = float(cooling_saturation.get(field, np.nan))
        if not np.isclose(actual, expected, rtol=2.0e-13, atol=1.0e-15):
            raise ValueError(
                f"point {point.slug} metadata has cooling {field}={actual!r}; "
                f"expected {expected!r}"
            )

    discs, points = generate_study_geometry(search)
    generated_hash = geometry_sha256(geometry_rows(discs, points))
    signature_payload = metadata["run_signature_payload"]
    if not isinstance(signature_payload, Mapping):
        raise ValueError(f"point {point.slug} has no signed geometry payload")
    recorded_hash = str(signature_payload.get("geometry_sha256", ""))
    if generated_hash != recorded_hash:
        raise RuntimeError(f"point {point.slug} generated geometry hash changed")
    if not point_paths.geometry_csv.is_file():
        raise RuntimeError(f"point {point.slug} launch_geometry.csv is missing")
    if _sha256(point_paths.geometry_csv) != recorded_hash:
        raise RuntimeError(f"point {point.slug} launch_geometry.csv is corrupt")
    validate_checkpoint_samples(samples, points)


def scan_completed_timeouts(
    paths: CampaignPaths,
    study_key: str,
) -> list[TimeoutCase]:
    """Return every saved timeout endpoint in completed selected-study points."""

    cases: list[TimeoutCase] = []
    groups = requested_refined_points()
    if study_key not in {RAW_STUDY_KEY, DETUNING_STUDY_KEY}:
        raise ValueError(f"timeout audit does not support study {study_key!r}")
    points = groups[study_key]
    for point in points:
        point_paths = paths.point_paths(point)
        if not point_paths.metadata_json.is_file() or not point_paths.final_samples_csv.is_file():
            continue
        metadata = _read_json(point_paths.metadata_json)
        if metadata.get("status") != "completed":
            continue
        samples = load_capture_velocity_samples(point_paths.final_samples_csv)
        _validate_completed_point_inputs(point, point_paths, metadata, samples)
        for sample in samples:
            endpoint_kind = None
            if sample.lower_classification == "timeout":
                endpoint_kind = "lower"
            elif sample.upper_classification == "timeout":
                endpoint_kind = "upper"
            if endpoint_kind is not None:
                cases.append(
                    TimeoutCase(
                        point=point,
                        paths=point_paths,
                        sample=sample,
                        run_metadata=metadata,
                        source_sample_sha256=_sha256(point_paths.final_samples_csv),
                        source_metadata_sha256=_sha256(point_paths.metadata_json),
                        endpoint_kind=endpoint_kind,
                    )
                )
    return cases


def scan_completed_detuning_timeouts(paths: CampaignPaths) -> list[TimeoutCase]:
    """Backward-compatible detuning-only scan entry point."""

    return scan_completed_timeouts(paths, DETUNING_STUDY_KEY)


def _classify(
    classifier: Classifier,
    point: PointSample,
    speed: float,
    search: RateCaptureSearchConfig,
    physics,
) -> TrajectoryClassification:
    model, beams, coil, config = physics
    return classifier(
        point,
        speed,
        search,
        model=model,
        beams=beams,
        coil_config=coil,
        config=config,
    )


def _rebisect(
    rebisector: Rebisector,
    point: PointSample,
    search: RateCaptureSearchConfig,
    physics,
) -> CaptureVelocitySample:
    model, beams, coil, config = physics
    return rebisector(
        point,
        search,
        model=model,
        beams=beams,
        coil_config=coil,
        config=config,
    )


def _definitive_bracket(sample: CaptureVelocitySample) -> bool:
    lower = sample.trapped_velocity_lower_m_per_s
    upper = sample.untrapped_velocity_upper_m_per_s
    width = upper - lower
    return bool(
        sample.lower_classification in TRAPPED_TERMINATION_REASONS
        and sample.upper_classification == "escaped"
        and 0.0 <= lower < upper
        and 0.0 < width < AUDIT_VELOCITY_TOLERANCE_M_PER_S + 1.0e-12
        and np.isclose(
            sample.capture_velocity_m_per_s, lower, rtol=0.0, atol=1.0e-12
        )
        and np.isclose(
            sample.velocity_resolution_m_per_s, width, rtol=0.0, atol=1.0e-12
        )
    )


def _compatible_brackets(
    coarse: CaptureVelocitySample, fine: CaptureVelocitySample
) -> bool:
    overlap = max(
        coarse.trapped_velocity_lower_m_per_s,
        fine.trapped_velocity_lower_m_per_s,
    ) <= min(
        coarse.untrapped_velocity_upper_m_per_s,
        fine.untrapped_velocity_upper_m_per_s,
    )
    endpoint_tolerance = AUDIT_VELOCITY_TOLERANCE_M_PER_S
    return bool(
        overlap
        and abs(
            coarse.trapped_velocity_lower_m_per_s
            - fine.trapped_velocity_lower_m_per_s
        )
        <= endpoint_tolerance
        and abs(
            coarse.untrapped_velocity_upper_m_per_s
            - fine.untrapped_velocity_upper_m_per_s
        )
        <= endpoint_tolerance
    )


def _classify_velocity_grid(
    classifier: Classifier,
    point: PointSample,
    search: RateCaptureSearchConfig,
    physics,
    velocities: Sequence[float],
    cached: Mapping[float, TrajectoryClassification],
) -> tuple[TrajectoryClassification, ...]:
    results: list[TrajectoryClassification] = []
    for velocity in velocities:
        cached_result = next(
            (
                result
                for cached_velocity, result in cached.items()
                if np.isclose(velocity, cached_velocity, rtol=0.0, atol=1.0e-12)
            ),
            None,
        )
        results.append(
            cached_result
            if cached_result is not None
            else _classify(classifier, point, velocity, search, physics)
        )
    return tuple(results)


def _velocity_resolved_capture_fallback(
    case: TimeoutCase,
    velocities: Sequence[float],
    coarse_scan: Sequence[TrajectoryClassification],
    fine_scan: Sequence[TrajectoryClassification],
) -> tuple[CaptureVelocitySample | None, str]:
    """Validate a direct capture mask and build its zero-threshold fallback.

    The fallback keeps legacy scalar readers conservative.  Corrected spectra
    and loading rates use the complete direct mask stored in the immutable
    audit report; they never infer the finite-speed band from this scalar row.
    """

    search = _production_search(case.run_metadata)
    step = float(search.analysis_velocity_step_m_per_s)
    stop = float(search.analysis_velocity_max_m_per_s)
    expected_velocities = tuple(
        float(value) for value in np.arange(0.0, stop + 0.5 * step, step)
    )
    if not (
        len(velocities) == len(expected_velocities)
        and len(coarse_scan) == len(expected_velocities)
        and len(fine_scan) == len(expected_velocities)
        and np.allclose(
            velocities,
            expected_velocities,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        return None, "velocity scan does not match the exact configured analysis grid"

    for velocity, coarse_result, fine_result in zip(
        expected_velocities, coarse_scan, fine_scan, strict=True
    ):
        if coarse_result.trapped != fine_result.trapped:
            return (
                None,
                f"velocity-grid classification changes with timestep at {velocity:g} m/s",
            )
        for label, result in (("coarse", coarse_result), ("fine", fine_result)):
            definitive = (
                result.trapped
                and result.termination_reason in TRAPPED_TERMINATION_REASONS
            ) or (
                not result.trapped and result.termination_reason == "escaped"
            )
            if not definitive:
                return (
                    None,
                    f"{label} velocity-grid point {velocity:g} m/s is not definitive",
                )

    trapped_indices = [
        index for index, result in enumerate(coarse_scan) if result.trapped
    ]
    if not trapped_indices:
        return None, "velocity scan contains no finite-speed capture band"
    if coarse_scan[0].trapped:
        return None, "zero speed is trapped, so a standard threshold search is required"
    if coarse_scan[-1].trapped:
        return None, "velocity-resolved scan has no escaped high-speed endpoint"
    upper_index = next(
        (
            index
            for index in range(1, len(expected_velocities))
            if not coarse_scan[index].trapped
        ),
        None,
    )
    if upper_index is None:  # pragma: no cover - final point was checked above
        return None, "velocity-resolved scan has no positive escaped fallback endpoint"
    upper_velocity = expected_velocities[upper_index]
    replacement = replace(
        case.sample,
        capture_velocity_m_per_s=0.0,
        velocity_resolution_m_per_s=upper_velocity,
        trapped_velocity_lower_m_per_s=0.0,
        untrapped_velocity_upper_m_per_s=upper_velocity,
        lower_classification=fine_scan[0].termination_reason,
        upper_classification=fine_scan[upper_index].termination_reason,
        lower_entered_trap_core=fine_scan[0].entered_trap_core,
        upper_entered_trap_core=fine_scan[upper_index].entered_trap_core,
        lower_core_entry_count=fine_scan[0].core_entry_count,
        upper_core_entry_count=fine_scan[upper_index].core_entry_count,
    )
    captured_velocities = [expected_velocities[index] for index in trapped_indices]
    return (
        replacement,
        "dual-timestep direct velocity mask retained for a nonmonotone capture band "
        f"from {min(captured_velocities):g} to {max(captured_velocities):g} m/s; "
        "the scalar sample is a zero-capture fallback and is not used for the "
        "corrected spectrum or loading integral",
    )


def _velocity_override_from_outcome(
    outcome: TimeoutOutcome,
) -> VelocityResolvedCaptureOverride:
    if outcome.status != "velocity_resolved_capture":
        raise ValueError("outcome is not velocity resolved")
    replacement, _ = _velocity_resolved_capture_fallback(
        outcome.case,
        outcome.velocity_scan_m_per_s,
        outcome.coarse_velocity_scan,
        outcome.fine_velocity_scan,
    )
    if replacement is None:
        raise RuntimeError("velocity-resolved outcome no longer validates")
    return VelocityResolvedCaptureOverride(
        disc_index=outcome.case.sample.disc_index,
        point_index=outcome.case.sample.point_index,
        velocity_m_per_s=outcome.velocity_scan_m_per_s,
        captured=tuple(result.trapped for result in outcome.fine_velocity_scan),
    )


def _velocity_override_payload(
    outcome: TimeoutOutcome,
    *,
    report_path: Path,
    report_sha256: str,
) -> dict[str, object]:
    override = _velocity_override_from_outcome(outcome)
    captured_velocities = [
        velocity
        for velocity, captured in zip(
            override.velocity_m_per_s, override.captured, strict=True
        )
        if captured
    ]
    return {
        "disc_index": override.disc_index,
        "point_index": override.point_index,
        "representation": "direct_boolean_capture_mask",
        "velocity_grid_m_per_s": list(override.velocity_m_per_s),
        "captured_mask": list(override.captured),
        "captured_velocity_nodes_m_per_s": captured_velocities,
        "scalar_sample_role": "zero-capture fallback only",
        "audit_report_json": str(report_path.resolve()),
        "audit_report_sha256": report_sha256,
        "dual_timestep_agreement_required": True,
    }


def _audit_lower_timeout(
    case: TimeoutCase,
    *,
    classifier: Classifier,
    rebisector: Rebisector,
    physics,
    point: PointSample,
    coarse_search: RateCaptureSearchConfig,
    fine_search: RateCaptureSearchConfig,
    coarse_lower: TrajectoryClassification,
    fine_lower: TrajectoryClassification,
    coarse_upper: TrajectoryClassification,
    fine_upper: TrajectoryClassification,
) -> TimeoutOutcome:
    if coarse_lower.trapped != fine_lower.trapped:
        return TimeoutOutcome(
            case,
            "unresolved",
            "zero-speed lower endpoint changes trapped classification with timestep",
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            None,
        )
    if coarse_lower.trapped and fine_lower.trapped:
        coarse_rebisected = _rebisect(rebisector, point, coarse_search, physics)
        fine_rebisected = _rebisect(rebisector, point, fine_search, physics)
        if not (
            _definitive_bracket(coarse_rebisected)
            and _definitive_bracket(fine_rebisected)
            and _compatible_brackets(coarse_rebisected, fine_rebisected)
        ):
            return TimeoutOutcome(
                case,
                "unresolved",
                "later-trapped zero endpoint did not yield compatible brackets",
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
                coarse_rebisected,
                fine_rebisected,
                None,
            )
        return TimeoutOutcome(
            case,
            "rebisected",
            (
                "saved zero-speed timeout traps at "
                f"{1.0e3 * coarse_search.max_simulation_time_s:g} ms; "
                "compatible brackets resolved it"
            ),
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            coarse_rebisected,
            fine_rebisected,
            fine_rebisected,
        )
    if not (
        coarse_lower.termination_reason == "escaped"
        and fine_lower.termination_reason == "escaped"
    ):
        return TimeoutOutcome(
            case,
            "unresolved",
            (
                "zero-speed lower endpoint remains non-definitive at "
                f"{1.0e3 * coarse_search.max_simulation_time_s:g} ms"
            ),
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            None,
        )

    step = float(coarse_search.analysis_velocity_step_m_per_s)
    stop = float(coarse_search.analysis_velocity_max_m_per_s)
    velocities = tuple(float(value) for value in np.arange(0.0, stop + 0.5 * step, step))
    coarse_scan = _classify_velocity_grid(
        classifier,
        point,
        coarse_search,
        physics,
        velocities,
        {
            case.sample.trapped_velocity_lower_m_per_s: coarse_lower,
            case.sample.untrapped_velocity_upper_m_per_s: coarse_upper,
        },
    )
    fine_scan = _classify_velocity_grid(
        classifier,
        point,
        fine_search,
        physics,
        velocities,
        {
            case.sample.trapped_velocity_lower_m_per_s: fine_lower,
            case.sample.untrapped_velocity_upper_m_per_s: fine_upper,
        },
    )
    all_escaped = all(
        not coarse_result.trapped
        and coarse_result.termination_reason == "escaped"
        and not fine_result.trapped
        and fine_result.termination_reason == "escaped"
        for coarse_result, fine_result in zip(
            coarse_scan, fine_scan, strict=True
        )
    )
    if all_escaped:
        replacement = replace(
            case.sample,
            capture_velocity_m_per_s=0.0,
            velocity_resolution_m_per_s=step,
            trapped_velocity_lower_m_per_s=0.0,
            untrapped_velocity_upper_m_per_s=step,
            lower_classification="escaped",
            upper_classification="escaped",
            lower_entered_trap_core=fine_scan[0].entered_trap_core,
            upper_entered_trap_core=fine_scan[1].entered_trap_core,
            lower_core_entry_count=fine_scan[0].core_entry_count,
            upper_core_entry_count=fine_scan[1].core_entry_count,
        )
        return TimeoutOutcome(
            case,
            "confirmed_zero_capture",
            (
                "zero speed escapes and no trapped island appears on the dual-"
                f"timestep 0-{stop:g} m/s grid at {step:g} m/s spacing"
            ),
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            replacement,
            velocities,
            coarse_scan,
            fine_scan,
        )
    replacement, reason = _velocity_resolved_capture_fallback(
        case, velocities, coarse_scan, fine_scan
    )
    if replacement is not None:
        return TimeoutOutcome(
            case,
            "velocity_resolved_capture",
            reason,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            replacement,
            velocities,
            coarse_scan,
            fine_scan,
        )
    return TimeoutOutcome(
        case,
        "unresolved",
        reason,
        coarse_lower,
        fine_lower,
        coarse_upper,
        fine_upper,
        None,
        None,
        None,
        velocities,
        coarse_scan,
        fine_scan,
    )


def audit_timeout_case(
    case: TimeoutCase,
    *,
    classifier: Classifier = classify_multilevel_loading_trajectory,
    rebisector: Rebisector = find_multilevel_capture_velocity,
    physics=None,
    audit_duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
) -> TimeoutOutcome:
    """Audit one timeout at both timesteps without writing any output."""

    physics = _reconstruct_physics(case) if physics is None else physics
    point = _point_sample(case.sample)
    coarse_search, fine_search = _audit_searches(
        case.run_metadata,
        audit_duration_s,
        coarse_time_step_s,
        fine_time_step_s,
    )
    lower_speed = case.sample.trapped_velocity_lower_m_per_s
    upper_speed = case.sample.untrapped_velocity_upper_m_per_s
    coarse_lower = _classify(classifier, point, lower_speed, coarse_search, physics)
    fine_lower = _classify(classifier, point, lower_speed, fine_search, physics)
    coarse_upper = _classify(classifier, point, upper_speed, coarse_search, physics)
    fine_upper = _classify(classifier, point, upper_speed, fine_search, physics)

    if case.endpoint_kind == "lower":
        return _audit_lower_timeout(
            case,
            classifier=classifier,
            rebisector=rebisector,
            physics=physics,
            point=point,
            coarse_search=coarse_search,
            fine_search=fine_search,
            coarse_lower=coarse_lower,
            fine_lower=fine_lower,
            coarse_upper=coarse_upper,
            fine_upper=fine_upper,
        )
    if case.endpoint_kind != "upper":
        raise ValueError(f"unsupported timeout endpoint kind {case.endpoint_kind!r}")

    endpoint_results = (coarse_lower, fine_lower, coarse_upper, fine_upper)
    if any(result.termination_reason == "non_finite" for result in endpoint_results):
        return TimeoutOutcome(
            case,
            "unresolved",
            (
                "a saved bracket endpoint produced a non-finite trajectory; "
                "fresh threshold search is refused"
            ),
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            None,
        )

    endpoints_are_definitive = bool(
        coarse_lower.trapped
        and coarse_lower.termination_reason in TRAPPED_TERMINATION_REASONS
        and fine_lower.trapped
        and fine_lower.termination_reason in TRAPPED_TERMINATION_REASONS
        and not coarse_upper.trapped
        and coarse_upper.termination_reason == "escaped"
        and not fine_upper.trapped
        and fine_upper.termination_reason == "escaped"
    )
    if endpoints_are_definitive:
        replacement = replace(
            case.sample,
            lower_classification=fine_lower.termination_reason,
            upper_classification="escaped",
            lower_entered_trap_core=fine_lower.entered_trap_core,
            upper_entered_trap_core=fine_upper.entered_trap_core,
            lower_core_entry_count=fine_lower.core_entry_count,
            upper_core_entry_count=fine_upper.core_entry_count,
        )
        return TimeoutOutcome(
            case,
            "confirmed_escaped",
            "saved bracket is definitively trapped/escaped at both audit timesteps",
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            None,
            None,
            replacement,
        )

    coarse_rebisected = _rebisect(rebisector, point, coarse_search, physics)
    fine_rebisected = _rebisect(rebisector, point, fine_search, physics)
    if not (
        _definitive_bracket(coarse_rebisected)
        and _definitive_bracket(fine_rebisected)
        and _compatible_brackets(coarse_rebisected, fine_rebisected)
    ):
        return TimeoutOutcome(
            case,
            "unresolved",
            (
                "saved endpoints were not jointly definitive and fresh coarse/fine "
                "searches did not produce compatible trapped/escaped brackets"
            ),
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            coarse_rebisected,
            fine_rebisected,
            None,
        )
    return TimeoutOutcome(
        case,
        "rebisected",
        (
            "saved endpoints were not jointly definitive; fresh coarse/fine "
            "searches produced compatible trapped/escaped brackets"
        ),
        coarse_lower,
        fine_lower,
        coarse_upper,
        fine_upper,
        coarse_rebisected,
        fine_rebisected,
        fine_rebisected,
    )


def _audit_case_worker(
    case: TimeoutCase,
    audit_duration_s: float,
    coarse_time_step_s: float,
    fine_time_step_s: float,
) -> TimeoutOutcome:
    """Spawn-safe production worker for one deterministic timeout case."""

    return audit_timeout_case(
        case,
        audit_duration_s=audit_duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
    )


def _case_error(case: TimeoutCase, error: BaseException) -> dict[str, object]:
    return {
        "point_index": case.point.point_index,
        "point_slug": case.point.slug,
        "disc_index": case.sample.disc_index,
        "disc_point_index": case.sample.point_index,
        "timeout_endpoint_kind": case.endpoint_kind,
        "error": f"{type(error).__name__}: {error}",
    }


def _audit_cases(
    cases: Sequence[TimeoutCase],
    *,
    workers: int,
    audit_duration_s: float,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    classifier: Classifier,
    rebisector: Rebisector,
) -> tuple[list[TimeoutOutcome], list[dict[str, object]]]:
    if not 1 <= workers <= MAX_AUDIT_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_AUDIT_WORKERS}")
    _validate_audit_search_parameters(
        audit_duration_s, coarse_time_step_s, fine_time_step_s
    )
    if not cases:
        return [], []
    if workers == 1:
        outcomes: list[TimeoutOutcome] = []
        errors: list[dict[str, object]] = []
        for index, case in enumerate(cases, start=1):
            print(
                f"[timeout-audit] {index}/{len(cases)}: "
                f"{case.point.slug}, disc={case.sample.disc_index}, "
                f"point={case.sample.point_index}",
                flush=True,
            )
            try:
                outcomes.append(
                    audit_timeout_case(
                        case,
                        classifier=classifier,
                        rebisector=rebisector,
                        audit_duration_s=audit_duration_s,
                        coarse_time_step_s=coarse_time_step_s,
                        fine_time_step_s=fine_time_step_s,
                    )
                )
            except Exception as error:
                errors.append(_case_error(case, error))
        return outcomes, errors

    if (
        classifier is not classify_multilevel_loading_trajectory
        or rebisector is not find_multilevel_capture_velocity
    ):
        raise ValueError(
            "workers > 1 requires the production classifier and rebisector"
        )
    process_count = min(workers, len(cases))
    print(
        f"[timeout-audit] dispatching {len(cases)} cases to "
        f"{process_count} spawn-process workers",
        flush=True,
    )
    ordered: list[TimeoutOutcome | None] = [None] * len(cases)
    indexed_errors: list[tuple[int, dict[str, object]]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _audit_case_worker,
                case,
                audit_duration_s,
                coarse_time_step_s,
                fine_time_step_s,
            ): index
            for index, case in enumerate(cases)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            case = cases[index]
            completed += 1
            try:
                ordered[index] = future.result()
                status = ordered[index].status
            except Exception as error:
                indexed_errors.append((index, _case_error(case, error)))
                status = f"error: {type(error).__name__}"
            print(
                f"[timeout-audit] completed {completed}/{len(cases)}: "
                f"{case.point.slug}, disc={case.sample.disc_index}, "
                f"point={case.sample.point_index}; {status}",
                flush=True,
            )
    outcomes = [outcome for outcome in ordered if outcome is not None]
    errors = [item for _, item in sorted(indexed_errors)]
    return outcomes, errors


def _outcome_payload(
    outcome: TimeoutOutcome,
    audit_duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
) -> dict[str, object]:
    sample = outcome.case.sample
    coarse_search, fine_search = _audit_searches(
        outcome.case.run_metadata,
        audit_duration_s,
        coarse_time_step_s,
        fine_time_step_s,
    )
    signature_payload = outcome.case.run_metadata.get("run_signature_payload", {})
    return {
        "point_index": outcome.case.point.point_index,
        **_point_context_payload(outcome.case.point),
        "point_slug": outcome.case.point.slug,
        "disc_index": sample.disc_index,
        "disc_point_index": sample.point_index,
        "timeout_endpoint_kind": outcome.case.endpoint_kind,
        "source_sample_sha256": outcome.case.source_sample_sha256,
        "source_metadata_sha256": outcome.case.source_metadata_sha256,
        "base_run_signature_sha256": outcome.case.run_metadata.get(
            "run_signature_sha256"
        ),
        "base_run_signature_payload": signature_payload,
        "signed_physics_source_sha256": (
            signature_payload.get("physics_source_sha256", {})
            if isinstance(signature_payload, Mapping)
            else {}
        ),
        "production_capture_search_config": dict(
            outcome.case.run_metadata["capture_search_config"]
        ),
        "audit_capture_search_configs": {
            "coarse": asdict(coarse_search),
            "fine": asdict(fine_search),
        },
        "status": outcome.status,
        "safe_to_apply": outcome.safe_to_apply,
        "reason": outcome.reason,
        "original_sample": asdict(sample),
        "coarse_lower_result": asdict(outcome.coarse_lower_result),
        "fine_lower_result": asdict(outcome.fine_lower_result),
        "coarse_upper_result": asdict(outcome.coarse_upper_result),
        "fine_upper_result": asdict(outcome.fine_upper_result),
        "coarse_rebisected": (
            None
            if outcome.coarse_rebisected is None
            else asdict(outcome.coarse_rebisected)
        ),
        "fine_rebisected": (
            None
            if outcome.fine_rebisected is None
            else asdict(outcome.fine_rebisected)
        ),
        "replacement_sample": (
            None
            if outcome.replacement_sample is None
            else asdict(outcome.replacement_sample)
        ),
        "velocity_scan": [
            {
                "velocity_m_per_s": velocity,
                "coarse_result": asdict(coarse_result),
                "fine_result": asdict(fine_result),
            }
            for velocity, coarse_result, fine_result in zip(
                outcome.velocity_scan_m_per_s,
                outcome.coarse_velocity_scan,
                outcome.fine_velocity_scan,
                strict=True,
            )
        ],
    }


def _classification_from_payload(
    payload: Mapping[str, object],
) -> TrajectoryClassification:
    values = dict(payload)
    values["final_position_m"] = tuple(values["final_position_m"])
    values["final_velocity_m_per_s"] = tuple(values["final_velocity_m_per_s"])
    return TrajectoryClassification(**values)


def _sample_from_payload(payload: Mapping[str, object]) -> CaptureVelocitySample:
    values = dict(payload)
    values["initial_position_m"] = tuple(values["initial_position_m"])
    values["incident_unit_vector"] = tuple(values["incident_unit_vector"])
    return CaptureVelocitySample(**values)


def _payload_audit_context(
    case: TimeoutCase,
    payload: Mapping[str, object],
) -> tuple[float, float, float]:
    configs = payload.get("audit_capture_search_configs")
    if not isinstance(configs, Mapping):
        raise ValueError("prior outcome has no audit capture-search configurations")
    coarse_payload = configs.get("coarse")
    fine_payload = configs.get("fine")
    if not isinstance(coarse_payload, Mapping) or not isinstance(fine_payload, Mapping):
        raise ValueError("prior outcome audit capture-search configurations are malformed")
    coarse_duration = float(coarse_payload.get("max_simulation_time_s", np.nan))
    fine_duration = float(fine_payload.get("max_simulation_time_s", np.nan))
    if not np.isclose(coarse_duration, fine_duration, rtol=0.0, atol=1.0e-15):
        raise ValueError("prior outcome coarse/fine durations differ")
    coarse_step = float(coarse_payload.get("time_step_s", np.nan))
    fine_step = float(fine_payload.get("time_step_s", np.nan))
    _validate_audit_search_parameters(coarse_duration, coarse_step, fine_step)
    expected_coarse, expected_fine = _audit_searches(
        case.run_metadata,
        coarse_duration,
        coarse_step,
        fine_step,
    )
    _assert_equivalent(
        coarse_payload,
        asdict(expected_coarse),
        location="prior_report.audit_capture_search_configs.coarse",
    )
    _assert_equivalent(
        fine_payload,
        asdict(expected_fine),
        location="prior_report.audit_capture_search_configs.fine",
    )
    return coarse_duration, coarse_step, fine_step


def _optional_sample_from_payload(payload: object) -> CaptureVelocitySample | None:
    return None if payload is None else _sample_from_payload(payload)


def _outcome_from_payload(
    case: TimeoutCase, payload: Mapping[str, object]
) -> TimeoutOutcome:
    if _payload_case_key(payload) != _case_key(case):
        raise ValueError("prior outcome key does not match the current timeout case")
    if payload.get("source_sample_sha256") != case.source_sample_sha256:
        raise RuntimeError("prior outcome sample fingerprint changed")
    if payload.get("source_metadata_sha256") != case.source_metadata_sha256:
        raise RuntimeError("prior outcome metadata fingerprint changed")
    _assert_equivalent(
        payload["original_sample"],
        asdict(case.sample),
        location="prior_report.original_sample",
    )
    _payload_audit_context(case, payload)
    velocity_scan = payload.get("velocity_scan", [])
    if not isinstance(velocity_scan, list):
        raise ValueError("prior outcome velocity_scan is malformed")
    outcome = TimeoutOutcome(
        case=case,
        status=str(payload["status"]),
        reason=str(payload["reason"]),
        coarse_lower_result=_classification_from_payload(
            payload["coarse_lower_result"]
        ),
        fine_lower_result=_classification_from_payload(payload["fine_lower_result"]),
        coarse_upper_result=_classification_from_payload(
            payload["coarse_upper_result"]
        ),
        fine_upper_result=_classification_from_payload(payload["fine_upper_result"]),
        coarse_rebisected=_optional_sample_from_payload(
            payload.get("coarse_rebisected")
        ),
        fine_rebisected=_optional_sample_from_payload(payload.get("fine_rebisected")),
        replacement_sample=_optional_sample_from_payload(
            payload.get("replacement_sample")
        ),
        velocity_scan_m_per_s=tuple(
            float(row["velocity_m_per_s"]) for row in velocity_scan
        ),
        coarse_velocity_scan=tuple(
            _classification_from_payload(row["coarse_result"])
            for row in velocity_scan
        ),
        fine_velocity_scan=tuple(
            _classification_from_payload(row["fine_result"])
            for row in velocity_scan
        ),
    )
    if bool(payload.get("safe_to_apply")) != outcome.safe_to_apply:
        raise RuntimeError("prior outcome safe-to-apply flag is inconsistent")
    _validate_deserialized_outcome(outcome)
    return outcome


def _validate_deserialized_outcome(outcome: TimeoutOutcome) -> None:
    """Fail closed when a prior report's claimed result is self-inconsistent."""

    if not outcome.safe_to_apply:
        if outcome.replacement_sample is not None:
            raise RuntimeError("unsafe prior outcome unexpectedly has a replacement")
        return
    replacement = outcome.replacement_sample
    if replacement is None:  # pragma: no cover - implied by safe_to_apply
        raise RuntimeError("safe prior outcome has no replacement")
    if outcome.status == "confirmed_escaped":
        if outcome.case.endpoint_kind != "upper":
            raise RuntimeError("confirmed-escaped prior outcome is not an upper timeout")
        if not (
            outcome.coarse_lower_result.trapped
            and outcome.coarse_lower_result.termination_reason
            in TRAPPED_TERMINATION_REASONS
            and outcome.fine_lower_result.trapped
            and outcome.fine_lower_result.termination_reason
            in TRAPPED_TERMINATION_REASONS
            and not outcome.coarse_upper_result.trapped
            and outcome.coarse_upper_result.termination_reason == "escaped"
            and not outcome.fine_upper_result.trapped
            and outcome.fine_upper_result.termination_reason == "escaped"
        ):
            raise RuntimeError("confirmed-escaped prior outcome lacks definitive endpoints")
        expected = replace(
            outcome.case.sample,
            lower_classification=outcome.fine_lower_result.termination_reason,
            upper_classification="escaped",
            lower_entered_trap_core=outcome.fine_lower_result.entered_trap_core,
            upper_entered_trap_core=outcome.fine_upper_result.entered_trap_core,
            lower_core_entry_count=outcome.fine_lower_result.core_entry_count,
            upper_core_entry_count=outcome.fine_upper_result.core_entry_count,
        )
        _assert_equivalent(
            asdict(expected), asdict(replacement), location="prior_report.replacement_sample"
        )
        return
    if outcome.status == "rebisected":
        if outcome.coarse_rebisected is None or outcome.fine_rebisected is None:
            raise RuntimeError("rebisected prior outcome is missing a timestep bracket")
        if not (
            _definitive_bracket(outcome.coarse_rebisected)
            and _definitive_bracket(outcome.fine_rebisected)
            and _compatible_brackets(
                outcome.coarse_rebisected, outcome.fine_rebisected
            )
        ):
            raise RuntimeError("rebisected prior outcome lacks compatible definitive brackets")
        geometry_fields = (
            "disc_index",
            "point_index",
            "theta_rad",
            "phi_rad",
            "theta_prime_rad",
            "s_m",
            "radial_distance_m",
            "initial_position_m",
            "incident_unit_vector",
        )
        for label, sample in (
            ("coarse", outcome.coarse_rebisected),
            ("fine", outcome.fine_rebisected),
        ):
            _assert_equivalent(
                {
                    field: getattr(outcome.case.sample, field)
                    for field in geometry_fields
                },
                {field: getattr(sample, field) for field in geometry_fields},
                location=f"prior_report.{label}_rebisected_geometry",
            )
        _assert_equivalent(
            asdict(outcome.fine_rebisected),
            asdict(replacement),
            location="prior_report.replacement_sample",
        )
        return
    if outcome.status == "velocity_resolved_capture":
        if outcome.case.endpoint_kind != "lower":
            raise RuntimeError(
                "velocity-resolved prior outcome is not a lower timeout"
            )
        expected, _ = _velocity_resolved_capture_fallback(
            outcome.case,
            outcome.velocity_scan_m_per_s,
            outcome.coarse_velocity_scan,
            outcome.fine_velocity_scan,
        )
        if expected is None:
            raise RuntimeError(
                "velocity-resolved prior outcome lacks a definitive dual-timestep mask"
            )
        _assert_equivalent(
            asdict(expected),
            asdict(replacement),
            location="prior_report.replacement_sample",
        )
        return
    if outcome.status == "confirmed_zero_capture":
        if outcome.case.endpoint_kind != "lower":
            raise RuntimeError("zero-capture prior outcome is not a lower timeout")
        search = _production_search(outcome.case.run_metadata)
        step = float(search.analysis_velocity_step_m_per_s)
        stop = float(search.analysis_velocity_max_m_per_s)
        expected_velocities = tuple(
            float(value) for value in np.arange(0.0, stop + 0.5 * step, step)
        )
        if (
            len(expected_velocities) < 2
            or len(outcome.velocity_scan_m_per_s) != len(expected_velocities)
            or not np.allclose(
                outcome.velocity_scan_m_per_s,
                expected_velocities,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError("zero-capture prior outcome lacks the exact velocity grid")
        if not (
            len(outcome.coarse_velocity_scan) == len(expected_velocities)
            and len(outcome.fine_velocity_scan) == len(expected_velocities)
            and all(
                coarse.termination_reason == "escaped"
                and fine.termination_reason == "escaped"
                and not coarse.trapped
                and not fine.trapped
                for coarse, fine in zip(
                    outcome.coarse_velocity_scan,
                    outcome.fine_velocity_scan,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("zero-capture prior outcome lacks dual escaped scans")
        expected = replace(
            outcome.case.sample,
            capture_velocity_m_per_s=0.0,
            velocity_resolution_m_per_s=step,
            trapped_velocity_lower_m_per_s=0.0,
            untrapped_velocity_upper_m_per_s=step,
            lower_classification="escaped",
            upper_classification="escaped",
            lower_entered_trap_core=outcome.fine_velocity_scan[0].entered_trap_core,
            upper_entered_trap_core=outcome.fine_velocity_scan[1].entered_trap_core,
            lower_core_entry_count=outcome.fine_velocity_scan[0].core_entry_count,
            upper_core_entry_count=outcome.fine_velocity_scan[1].core_entry_count,
        )
        _assert_equivalent(
            asdict(expected), asdict(replacement), location="prior_report.replacement_sample"
        )
        return
    raise RuntimeError(f"unsupported safe prior outcome status {outcome.status!r}")


def _predicted_loading_changes(
    outcomes: Sequence[TimeoutOutcome],
) -> list[dict[str, object]]:
    grouped: dict[int, list[TimeoutOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.case.point.point_index, []).append(outcome)
    changes: list[dict[str, object]] = []
    for point_index, point_outcomes in sorted(grouped.items()):
        case = point_outcomes[0].case
        original = load_capture_velocity_samples(case.paths.final_samples_csv)
        replacements = {
            (item.case.sample.disc_index, item.case.sample.point_index): item.replacement_sample
            for item in point_outcomes
            if item.replacement_sample is not None
        }
        corrected = [
            replacements.get((sample.disc_index, sample.point_index), sample)
            for sample in original
        ]
        search = _production_search(case.run_metadata)
        overrides = [
            _velocity_override_from_outcome(item)
            for item in point_outcomes
            if item.status == "velocity_resolved_capture"
        ]

        def loading(samples, velocity_overrides=()):
            spectrum = calculate_clustered_cross_section_with_overrides(
                samples, search, velocity_overrides
            )
            _, result = calculate_disc_clustered_loading_with_overrides(
                samples, search, spectrum, velocity_overrides
            )
            return result

        old = loading(original)
        new = loading(corrected, overrides)
        old_rate = float(old["loading_rate_mean_atoms_per_s"])
        new_rate = float(new["loading_rate_mean_atoms_per_s"])
        changes.append(
            {
                "point_index": point_index,
                **_point_context_payload(case.point),
                "original_loading_rate_atoms_per_s": old_rate,
                "corrected_loading_rate_atoms_per_s": new_rate,
                "absolute_change_atoms_per_s": new_rate - old_rate,
                "relative_change": (
                    None if old_rate == 0.0 else new_rate / old_rate - 1.0
                ),
                "resolved_timeout_count": len(replacements),
                "velocity_resolved_capture_count": len(overrides),
            }
        )
    return changes


def _point_context_payload(point: RelationshipPoint) -> dict[str, object]:
    """Identify the independent sweep coordinate and physical laser setting."""

    return {
        "study_key": point.study_key,
        "scan_variable": point.scan_variable,
        "scan_value": point.scan_value,
        "s0": point.on_resonance_saturation,
        "seff": point.effective_saturation,
        "detuning_n": point.cooling_detuning_n,
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "cooling_power_mw_per_beam": 1.0e3 * point.cooling_power_w_per_beam,
    }


def _default_audit_directory(paths: CampaignPaths, study_key: str) -> Path:
    output_root = paths.statistics.parents[2]
    return (
        output_root
        / "diagnostics"
        / "mot_multilevel"
        / paths.statistics.name
        / f"{study_key}_timeout_audit"
        / _timestamp_slug()
    )


@contextmanager
def _exclusive_campaign_apply_lock(paths: CampaignPaths):
    """Exclude concurrent raw/detuning apply transactions for one campaign."""

    lock_directory = (
        paths.statistics.parent
        / f".{paths.statistics.name}.timeout_audit_apply.lock"
    )
    if not lock_directory.parent.is_dir():
        raise RuntimeError(
            f"campaign statistics parent does not exist: {lock_directory.parent}"
        )
    try:
        lock_directory.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "another timeout-audit apply transaction holds the campaign lock: "
            f"{lock_directory}. Inspect owner.json; after a hard-killed process, "
            "remove the lock directory only after confirming no apply is active"
        ) from error
    owner = lock_directory / "owner.json"
    try:
        _atomic_write_json(
            owner,
            {
                "created_utc": _utc_now(),
                "process_id": os.getpid(),
                "statistics_root": str(paths.statistics.resolve()),
            },
        )
        yield lock_directory
    finally:
        if owner.exists():
            owner.unlink()
        lock_directory.rmdir()


def _point_product_paths(point_paths: StudyPaths) -> set[Path]:
    return {
        point_paths.metadata_json,
        point_paths.geometry_csv,
        point_paths.partial_samples_csv,
        point_paths.final_samples_csv,
        point_paths.capture_summary_json,
        point_paths.spectrum_csv,
        point_paths.loading_by_disc_csv,
        point_paths.loading_json,
        point_paths.cross_section_png,
        point_paths.impact_parameter_png,
        point_paths.loading_by_disc_png,
    }


def _all_study_product_paths(paths: CampaignPaths, study_key: str) -> list[Path]:
    files = {
        paths.metadata_json,
        paths.aggregate_csv(study_key),
        paths.study_metadata_json(study_key),
        paths.relationship_plot(study_key),
    }
    for point in requested_refined_points()[study_key]:
        files.update(_point_product_paths(paths.point_paths(point)))
    return sorted(files, key=str)


def _modified_product_paths(
    paths: CampaignPaths,
    outcomes: Sequence[TimeoutOutcome],
    study_key: str,
) -> list[Path]:
    files = {
        paths.metadata_json,
        paths.aggregate_csv(study_key),
        paths.study_metadata_json(study_key),
        paths.relationship_plot(study_key),
    }
    for outcome in outcomes:
        files.update(_point_product_paths(outcome.case.paths) - {outcome.case.paths.geometry_csv})
    return sorted(files, key=str)


def _snapshot_files(files: Sequence[Path], *, require_all: bool) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for path in files:
        resolved = str(path.resolve())
        if path.is_file():
            snapshot[resolved] = _sha256(path)
        elif require_all:
            raise RuntimeError(f"required production output is missing: {path}")
        else:
            snapshot[resolved] = None
    return snapshot


def _verify_snapshot(snapshot: Mapping[str, str | None]) -> None:
    for raw_path, expected in snapshot.items():
        path = Path(raw_path)
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            raise RuntimeError(f"production output changed during audit: {path}")


def _campaign_completion_issues(
    paths: CampaignPaths, study_key: str
) -> list[str]:
    issues: list[str] = []
    if not paths.metadata_json.is_file():
        issues.append("campaign_metadata.json is missing")
    else:
        campaign = _read_json(paths.metadata_json)
        if campaign.get("status") != "completed":
            issues.append("campaign_metadata.json status is not 'completed'")
    points = requested_refined_points()[study_key]
    for point in points:
        point_paths = paths.point_paths(point)
        if not point_paths.metadata_json.is_file():
            issues.append(f"{point.slug}: run_metadata.json is missing")
            continue
        metadata = _read_json(point_paths.metadata_json)
        if metadata.get("status") != "completed":
            issues.append(f"{point.slug}: status is not 'completed'")
        if not point_paths.final_samples_csv.is_file():
            issues.append(f"{point.slug}: final sample CSV is missing")
            continue
        try:
            count = len(load_capture_velocity_samples(point_paths.final_samples_csv))
        except Exception as error:
            issues.append(f"{point.slug}: sample CSV cannot be read ({error})")
            continue
        if count != 625:
            issues.append(f"{point.slug}: has {count} samples, expected 625")
        if not point_paths.partial_samples_csv.is_file():
            issues.append(f"{point.slug}: partial sample CSV is missing")
        elif (
            point_paths.partial_samples_csv.read_bytes()
            != point_paths.final_samples_csv.read_bytes()
        ):
            issues.append(f"{point.slug}: partial and final sample CSVs differ")
    if not paths.study_metadata_json(study_key).is_file():
        issues.append(f"{study_key} sweep_metadata.json is missing")
    else:
        sweep = _read_json(paths.study_metadata_json(study_key))
        if sweep.get("status") != "completed":
            issues.append(f"{study_key} sweep_metadata.json status is not 'completed'")
    aggregate_path = paths.aggregate_csv(study_key)
    if not aggregate_path.is_file():
        issues.append(f"{study_key} aggregate.csv is missing")
    else:
        try:
            with aggregate_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            indices = [int(row["point_index"]) for row in rows]
            expected_indices = list(range(len(points)))
            if sorted(indices) != expected_indices or len(indices) != len(set(indices)):
                issues.append(
                    f"{study_key} aggregate.csv is not one complete row per planned point"
                )
        except Exception as error:
            issues.append(f"{study_key} aggregate.csv cannot be read ({error})")
    return issues


def _case_fingerprints(
    cases: Sequence[TimeoutCase],
) -> dict[tuple[int, int, int, str], tuple[str, str]]:
    return {
        _case_key(case): (case.source_sample_sha256, case.source_metadata_sha256)
        for case in cases
    }


def _case_key(case: TimeoutCase) -> tuple[int, int, int, str]:
    return (
        case.point.point_index,
        case.sample.disc_index,
        case.sample.point_index,
        case.endpoint_kind,
    )


def _payload_case_key(payload: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        int(payload["point_index"]),
        int(payload["disc_index"]),
        int(payload["disc_point_index"]),
        str(payload.get("timeout_endpoint_kind", "upper")),
    )


def _backup_files(
    paths: CampaignPaths,
    files: Sequence[Path],
    audit_directory: Path,
) -> list[dict[str, str]]:
    backup_root = audit_directory / "production_backups"
    if backup_root.exists():
        raise FileExistsError(f"backup directory already exists: {backup_root}")
    records: list[dict[str, str]] = []
    for source in files:
        if not source.is_file():
            raise RuntimeError(f"cannot back up missing production output: {source}")
        relative: Path | None = None
        for label, root in (
            ("statistics_root", paths.statistics),
            ("figures_root", paths.figures),
        ):
            try:
                relative = Path(label) / source.resolve().relative_to(root.resolve())
                break
            except ValueError:
                continue
        if relative is None:
            path_hash = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()
            relative = Path("external") / path_hash[:16] / source.name
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "source": str(source.resolve()),
                "backup": str(destination.resolve()),
                "sha256": _sha256(destination),
            }
        )
    return records


def _restore_backups(records: Sequence[Mapping[str, str]]) -> list[str]:
    errors: list[str] = []
    for record in records:
        source = Path(record["source"])
        backup = Path(record["backup"])
        try:
            if _sha256(backup) != record["sha256"]:
                raise RuntimeError("backup hash changed")
            temporary = source.with_name(f".{source.name}.timeout-audit-restore.tmp")
            shutil.copy2(backup, temporary)
            temporary.replace(source)
            if _sha256(source) != record["sha256"]:
                raise RuntimeError("restored source hash does not match backup")
        except Exception as error:
            errors.append(f"{source}: {type(error).__name__}: {error}")
    return errors


def _plot_context(point: RelationshipPoint) -> str:
    return (
        f"s0={point.on_resonance_saturation:.6g}; "
        f"seff={point.effective_saturation:.6g}; "
        f"P={1e3 * point.cooling_power_w_per_beam:.6g} mW; "
        f"Delta/Gamma={point.cooling_detuning_n:.6g}; "
        "0.1 mW repump; full sphere; r=15 mm"
    )


def _append_provenance(path: Path, record: Mapping[str, object]) -> None:
    payload = _read_json(path)
    history = list(payload.get("post_campaign_timeout_audits", []))
    history.append(dict(record))
    payload["post_campaign_timeout_audits"] = history
    payload["updated_utc"] = _utc_now()
    _atomic_write_json(path, payload)


def _rebuild_study_aggregate(
    paths: CampaignPaths, study_key: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    common_geometry_hash: str | None = None
    first_search: RateCaptureSearchConfig | None = None
    for point in requested_refined_points()[study_key]:
        point_paths = paths.point_paths(point)
        metadata = _read_json(point_paths.metadata_json)
        search = _production_search(metadata)
        summary = _read_json(point_paths.capture_summary_json)
        row = _point_row(point, search, point_paths, summary)
        geometry_hash = str(row["geometry_sha256"])
        if common_geometry_hash is None:
            common_geometry_hash = geometry_hash
            first_search = search
        elif geometry_hash != common_geometry_hash:
            raise RuntimeError(f"{study_key} points do not share one seeded geometry")
        rows.append(row)
    expected_indices = list(range(len(rows)))
    if [int(row["point_index"]) for row in rows] != expected_indices:
        raise RuntimeError(f"rebuilt {study_key} aggregate has an invalid point grid")
    _atomic_write_text(paths.aggregate_csv(study_key), _csv_text(rows))
    if first_search is None:
        raise RuntimeError(f"cannot plot an empty {study_key} aggregate")
    plot_refined_loading_relationship(
        rows,
        study_key,
        paths.relationship_plot(study_key),
        search_config=first_search,
    )
    return rows


def _reanalyze_with_velocity_overrides(
    case: TimeoutCase,
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    summary: Mapping[str, object],
    outcomes: Sequence[TimeoutOutcome],
    *,
    report_path: Path,
    report_sha256: str,
) -> dict[str, object]:
    """Replace threshold-only products with directly masked capture products."""

    resolved = [
        outcome
        for outcome in outcomes
        if outcome.status == "velocity_resolved_capture"
    ]
    if not resolved:
        return dict(summary)
    overrides = [_velocity_override_from_outcome(outcome) for outcome in resolved]
    spectrum = calculate_clustered_cross_section_with_overrides(
        samples, search, overrides
    )
    by_disc, loading = calculate_disc_clustered_loading_with_overrides(
        samples, search, spectrum, overrides
    )
    _atomic_write_csv(case.paths.spectrum_csv, spectrum, SPECTRUM_FIELDNAMES)
    _atomic_write_csv(
        case.paths.loading_by_disc_csv, by_disc, LOADING_BY_DISC_FIELDNAMES
    )
    override_payloads = [
        _velocity_override_payload(
            outcome,
            report_path=report_path,
            report_sha256=report_sha256,
        )
        for outcome in resolved
    ]
    loading_payload = _read_json(case.paths.loading_json)
    loading_payload.update(loading)
    loading_payload.update(
        {
            "capture_representation": "hybrid_threshold_and_velocity_resolved",
            "scalar_threshold_sample_count": len(samples) - len(overrides),
            "velocity_resolved_sample_count": len(overrides),
            "velocity_resolved_capture_overrides": override_payloads,
            "velocity_resolved_note": (
                "The listed nonmonotone rays use direct audited capture masks; "
                "their zero-valued scalar rows are conservative compatibility "
                "fallbacks and are not used in this spectrum or loading integral."
            ),
        }
    )
    _atomic_write_json(case.paths.loading_json, loading_payload)
    context = _plot_context(case.point)
    plot_clustered_cross_section(
        spectrum,
        case.paths.cross_section_png,
        title=f"24-State MOT Capture Cross Section ({context})",
    )
    plot_loading_rate_by_disc(
        by_disc,
        loading,
        case.paths.loading_by_disc_png,
        title=f"24-State MOT Loading Rate by Direction ({context})",
    )
    revised_summary = dict(summary)
    revised_summary.update(
        {
            "loading_rate": loading_payload,
            "capture_representation": "hybrid_threshold_and_velocity_resolved",
            "scalar_threshold_sample_count": len(samples) - len(overrides),
            "velocity_resolved_sample_count": len(overrides),
            "velocity_resolved_capture_overrides": override_payloads,
        }
    )
    _atomic_write_json(case.paths.capture_summary_json, revised_summary)
    return revised_summary


def _dataset_revision(
    outcomes: Sequence[TimeoutOutcome], report_path: Path, study_key: str
) -> tuple[str, dict[str, object]]:
    by_point: dict[int, TimeoutCase] = {}
    for outcome in outcomes:
        by_point[outcome.case.point.point_index] = outcome.case
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "post_campaign_timeout_correction",
        "study_key": study_key,
        "audit_report_sha256": _sha256(report_path),
        "hybrid_dataset": True,
        "uncorrected_sample_search": "production 50 ms / 5 us",
        "corrected_timeout_search": (
            "duration and ordered coarse/fine timesteps documented per case in "
            "the immutable audit report; fine bracket retained"
        ),
        "velocity_resolved_capture_count": sum(
            outcome.status == "velocity_resolved_capture"
            for outcome in outcomes
        ),
        "velocity_resolved_capture_representation": (
            "direct boolean masks in the immutable audit report; scalar rows are "
            "zero-capture compatibility fallbacks"
        ),
        "points": [
            {
                "point_index": point_index,
                "point_slug": case.point.slug,
                "base_run_signature_sha256": case.run_metadata[
                    "run_signature_sha256"
                ],
                "corrected_samples_sha256": _sha256(
                    case.paths.final_samples_csv
                ),
            }
            for point_index, case in sorted(by_point.items())
        ],
    }
    return study_signature(payload), payload


def _validate_post_apply(
    paths: CampaignPaths,
    outcomes: Sequence[TimeoutOutcome],
    study_key: str,
) -> None:
    affected: dict[int, TimeoutCase] = {}
    expected_replacements: dict[
        tuple[int, int, int], CaptureVelocitySample
    ] = {}
    for outcome in outcomes:
        affected[outcome.case.point.point_index] = outcome.case
        if outcome.replacement_sample is None:
            raise RuntimeError("post-apply validation received an unresolved outcome")
        expected_replacements[
            (
                outcome.case.point.point_index,
                outcome.case.sample.disc_index,
                outcome.case.sample.point_index,
            )
        ] = outcome.replacement_sample
    for point_index, case in sorted(affected.items()):
        if case.paths.partial_samples_csv.read_bytes() != case.paths.final_samples_csv.read_bytes():
            raise RuntimeError(f"partial/final sample CSV mismatch: {case.point.slug}")
        metadata = _read_json(case.paths.metadata_json)
        samples = load_capture_velocity_samples(case.paths.final_samples_csv)
        _validate_completed_point_inputs(case.point, case.paths, metadata, samples)
        keyed = {(sample.disc_index, sample.point_index): sample for sample in samples}
        for key, expected_sample in expected_replacements.items():
            expected_point_index, disc_index, disc_point_index = key
            if expected_point_index != point_index:
                continue
            actual_sample = keyed[(disc_index, disc_point_index)]
            _assert_equivalent(
                asdict(expected_sample),
                asdict(actual_sample),
                location=(
                    f"post_apply.{case.point.slug}."
                    f"{disc_index}.{disc_point_index}"
                ),
            )
            if "timeout" in {
                actual_sample.lower_classification,
                actual_sample.upper_classification,
            }:
                raise RuntimeError(
                    f"corrected timeout remains in {case.point.slug}: "
                    f"{disc_index}/{disc_point_index}"
                )
        point_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.case.point.point_index == point_index
        ]
        overrides = [
            _velocity_override_from_outcome(outcome)
            for outcome in point_outcomes
            if outcome.status == "velocity_resolved_capture"
        ]
        if overrides:
            search = _production_search(metadata)
            expected_spectrum = calculate_clustered_cross_section_with_overrides(
                samples, search, overrides
            )
            with case.paths.spectrum_csv.open(
                newline="", encoding="utf-8"
            ) as stream:
                actual_spectrum = list(csv.DictReader(stream))
            if len(actual_spectrum) != len(expected_spectrum):
                raise RuntimeError(
                    f"velocity-resolved spectrum length differs: {case.point.slug}"
                )
            for row_index, (expected_row, actual_row) in enumerate(
                zip(expected_spectrum, actual_spectrum, strict=True)
            ):
                for field, expected_value in expected_row.items():
                    actual_value = (
                        int(actual_row[field])
                        if isinstance(expected_value, int)
                        else float(actual_row[field])
                    )
                    _assert_equivalent(
                        expected_value,
                        actual_value,
                        location=(
                            f"post_apply.{case.point.slug}.spectrum."
                            f"{row_index}.{field}"
                        ),
                    )
            expected_by_disc, expected_loading = (
                calculate_disc_clustered_loading_with_overrides(
                    samples, search, expected_spectrum, overrides
                )
            )
            with case.paths.loading_by_disc_csv.open(
                newline="", encoding="utf-8"
            ) as stream:
                actual_by_disc = list(csv.DictReader(stream))
            if len(actual_by_disc) != len(expected_by_disc):
                raise RuntimeError(
                    f"velocity-resolved per-disc row count differs: {case.point.slug}"
                )
            for row_index, (expected_row, actual_row) in enumerate(
                zip(expected_by_disc, actual_by_disc, strict=True)
            ):
                for field, expected_value in expected_row.items():
                    actual_value = (
                        int(actual_row[field])
                        if isinstance(expected_value, int)
                        else float(actual_row[field])
                    )
                    _assert_equivalent(
                        expected_value,
                        actual_value,
                        location=(
                            f"post_apply.{case.point.slug}.by_disc."
                            f"{row_index}.{field}"
                        ),
                    )
            actual_loading = _read_json(case.paths.loading_json)
            for field, expected_value in expected_loading.items():
                _assert_equivalent(
                    expected_value,
                    actual_loading[field],
                    location=f"post_apply.{case.point.slug}.loading.{field}",
                )
            if int(actual_loading.get("velocity_resolved_sample_count", -1)) != len(
                overrides
            ):
                raise RuntimeError(
                    f"velocity-resolved provenance count differs: {case.point.slug}"
                )
            actual_summary = _read_json(case.paths.capture_summary_json)
            if int(actual_summary.get("velocity_resolved_sample_count", -1)) != len(
                overrides
            ):
                raise RuntimeError(
                    f"capture-summary override count differs: {case.point.slug}"
                )
            _assert_equivalent(
                actual_loading,
                actual_summary["loading_rate"],
                location=f"post_apply.{case.point.slug}.summary_loading",
            )
    aggregate_path = paths.aggregate_csv(study_key)
    with aggregate_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(requested_refined_points()[study_key])
    indices = [int(row["point_index"]) for row in rows]
    if len(rows) != expected or indices != list(range(expected)):
        raise RuntimeError(
            f"post-apply {study_key} aggregate is not the exact planned grid"
        )


def _prepare_continuation(
    prior_report_path: Path,
    cases: Sequence[TimeoutCase],
    *,
    paths: CampaignPaths,
    study_key: str,
    audit_duration_s: float,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
) -> tuple[
    list[TimeoutOutcome],
    list[TimeoutCase],
    dict[tuple[int, int, int, str], dict[str, object]],
    dict[str, object],
]:
    _validate_audit_search_parameters(
        audit_duration_s, coarse_time_step_s, fine_time_step_s
    )
    if not prior_report_path.is_file():
        raise FileNotFoundError(f"prior audit report does not exist: {prior_report_path}")
    prior_hash = _sha256(prior_report_path)
    prior = _read_json(prior_report_path)
    if _sha256(prior_report_path) != prior_hash:
        raise RuntimeError("prior audit report changed while it was being read")
    if prior.get("status") != "audited":
        raise ValueError("prior report status is not 'audited'")
    if int(prior.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("prior report schema version is incompatible")
    if prior.get("study_key") != study_key:
        raise ValueError("prior report study does not match --study")
    if Path(str(prior.get("production_statistics_root", ""))).resolve() != paths.statistics.resolve():
        raise ValueError("prior report statistics root does not match this campaign")
    if Path(str(prior.get("production_figures_root", ""))).resolve() != paths.figures.resolve():
        raise ValueError("prior report figures root does not match this campaign")
    prior_duration = float(prior.get("audit_duration_s", 0.0))
    if audit_duration_s < prior_duration and not np.isclose(
        audit_duration_s, prior_duration, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError(
            "continuation duration cannot be shorter than the prior audit duration "
            f"({1.0e3 * prior_duration:g} ms)"
        )
    raw_prior_steps = prior.get("audit_time_steps_s", [])
    if not isinstance(raw_prior_steps, list) or len(raw_prior_steps) != 2:
        raise ValueError("prior report has no valid dual-timestep pair")
    prior_coarse_time_step_s, prior_fine_time_step_s = map(
        float, raw_prior_steps
    )
    _validate_audit_search_parameters(
        prior_duration, prior_coarse_time_step_s, prior_fine_time_step_s
    )
    tolerance = 1.0e-15
    same_coarse_step = np.isclose(
        coarse_time_step_s,
        prior_coarse_time_step_s,
        rtol=0.0,
        atol=tolerance,
    )
    same_fine_step = np.isclose(
        fine_time_step_s,
        prior_fine_time_step_s,
        rtol=0.0,
        atol=tolerance,
    )
    strictly_finer_pair = bool(
        coarse_time_step_s < prior_coarse_time_step_s - tolerance
        and fine_time_step_s < prior_fine_time_step_s - tolerance
    )
    if (
        coarse_time_step_s > prior_coarse_time_step_s + tolerance
        or fine_time_step_s > prior_fine_time_step_s + tolerance
    ):
        raise ValueError("continuation timesteps cannot be coarser than the prior pair")
    prior_runtime = prior.get("audit_runtime_source_sha256")
    current_runtime = _audit_runtime_source_hashes()
    if not isinstance(prior_runtime, Mapping):
        raise ValueError("prior report has no audit runtime source hashes")
    added_runtime_sources = set(current_runtime) - set(prior_runtime)
    if (
        set(prior_runtime) - set(current_runtime)
        or added_runtime_sources
        - {"mot_multilevel/velocity_resolved_capture.py"}
    ):
        raise ValueError("prior report audit runtime source set is incompatible")
    # The continuation feature necessarily changes timeout_audit.py itself.
    # Every independent runtime/scientific helper must still match bit for bit;
    # safe prior outcomes are additionally reconstructed and checked below.
    for name, digest in current_runtime.items():
        if name not in prior_runtime:
            continue
        if name == "mot_multilevel/timeout_audit.py":
            continue
        if str(prior_runtime[name]) != digest:
            raise RuntimeError(f"audit runtime source changed since prior report: {name}")

    current = {_case_key(case): case for case in cases}
    prior_payloads: dict[tuple[int, int, int, str], dict[str, object]] = {}
    raw_payloads = prior.get("cases", [])
    if not isinstance(raw_payloads, list):
        raise ValueError("prior report cases field is malformed")
    for raw_payload in raw_payloads:
        if not isinstance(raw_payload, Mapping):
            raise ValueError("prior report contains a malformed case")
        payload = dict(raw_payload)
        key = _payload_case_key(payload)
        if key in prior_payloads:
            raise ValueError(f"prior report contains duplicate case {key}")
        prior_payloads[key] = payload
    error_keys: set[tuple[int, int, int, str]] = set()
    unkeyed_errors: list[Mapping[str, object]] = []
    raw_errors = prior.get("audit_errors", [])
    if not isinstance(raw_errors, list):
        raise ValueError("prior report audit_errors field is malformed")
    for item in raw_errors:
        if not isinstance(item, Mapping) or "point_index" not in item:
            if isinstance(item, Mapping):
                unkeyed_errors.append(item)
            else:
                raise ValueError("prior report contains a malformed audit error")
            continue
        key = (
            int(item["point_index"]),
            int(item["disc_index"]),
            int(item["disc_point_index"]),
            str(item.get("timeout_endpoint_kind", "upper")),
        )
        if key in error_keys:
            raise ValueError(f"prior report contains duplicate audit error {key}")
        error_keys.add(key)
    if unkeyed_errors:
        raise RuntimeError(
            "prior report contains non-case audit errors and cannot be continued"
        )
    overlap = set(prior_payloads) & error_keys
    if overlap:
        raise RuntimeError(
            f"prior report lists cases as both outcomes and audit errors: {sorted(overlap)}"
        )
    if set(current) != set(prior_payloads) | error_keys:
        raise RuntimeError("prior report does not cover the current timeout case set")
    if int(prior.get("timeout_case_count", -1)) != len(current):
        raise RuntimeError("prior report timeout count does not match current production")
    prior_safe_case_count = sum(
        bool(payload.get("safe_to_apply")) for payload in prior_payloads.values()
    )
    prior_unsafe_case_count = len(prior_payloads) - prior_safe_case_count + len(
        error_keys
    )
    if int(prior.get("resolved_case_count", -1)) != prior_safe_case_count:
        raise RuntimeError("prior report resolved-case count is inconsistent")
    if int(prior.get("unresolved_case_count", -1)) != prior_unsafe_case_count:
        raise RuntimeError("prior report unresolved-case count is inconsistent")
    expected_prior_safe = bool(
        prior.get("campaign_complete") is True
        and not raw_errors
        and prior_safe_case_count == len(current)
    )
    if bool(prior.get("safe_to_apply")) != expected_prior_safe:
        raise RuntimeError("prior report top-level safe-to-apply flag is inconsistent")

    retained: list[TimeoutOutcome] = []
    rerun: list[TimeoutCase] = []
    retained_payloads: dict[
        tuple[int, int, int, str], dict[str, object]
    ] = {}
    reclassified_case_count = 0
    for key, case in current.items():
        payload = prior_payloads.get(key)
        if payload is not None:
            outcome = _outcome_from_payload(case, payload)
            payload_duration, payload_coarse, payload_fine = (
                _payload_audit_context(case, payload)
            )
            reclassified = False
            if (
                not outcome.safe_to_apply
                and outcome.case.endpoint_kind == "lower"
                and outcome.status == "unresolved"
                and outcome.velocity_scan_m_per_s
            ):
                replacement, reason = _velocity_resolved_capture_fallback(
                    outcome.case,
                    outcome.velocity_scan_m_per_s,
                    outcome.coarse_velocity_scan,
                    outcome.fine_velocity_scan,
                )
                if replacement is not None:
                    outcome = replace(
                        outcome,
                        status="velocity_resolved_capture",
                        reason=reason,
                        replacement_sample=replacement,
                    )
                    _validate_deserialized_outcome(outcome)
                    reclassified = True
                    reclassified_case_count += 1
            if outcome.safe_to_apply:
                retained.append(outcome)
                if reclassified:
                    promoted_payload = _outcome_payload(
                        outcome,
                        payload_duration,
                        payload_coarse,
                        payload_fine,
                    )
                    retained_payloads[key] = {
                        **promoted_payload,
                        "reclassified_from_prior_report": str(
                            prior_report_path.resolve()
                        ),
                        "reclassified_from_prior_report_sha256": prior_hash,
                        "reclassification_used_existing_trajectory_results": True,
                    }
                else:
                    retained_payloads[key] = {
                        **payload,
                        "retained_from_prior_report": str(
                            prior_report_path.resolve()
                        ),
                        "retained_from_prior_report_sha256": prior_hash,
                    }
                continue
            if payload is not None:
                if not (
                    np.isclose(
                        payload_duration, prior_duration, rtol=0.0, atol=tolerance
                    )
                    and np.isclose(
                        payload_coarse,
                        prior_coarse_time_step_s,
                        rtol=0.0,
                        atol=tolerance,
                    )
                    and np.isclose(
                        payload_fine,
                        prior_fine_time_step_s,
                        rtol=0.0,
                        atol=tolerance,
                    )
                ):
                    raise RuntimeError(
                        "unresolved prior outcome search context differs from the "
                        "prior report continuation context"
                    )
        rerun.append(case)
    same_duration = np.isclose(
        audit_duration_s, prior_duration, rtol=0.0, atol=1.0e-15
    )
    same_timestep_pair = bool(same_coarse_step and same_fine_step)
    if rerun and same_duration and not strictly_finer_pair:
        raise ValueError(
            "same-duration continuation with unresolved cases requires both coarse "
            "and fine timesteps to be strictly finer than the prior pair"
        )
    if not rerun and not (same_duration and same_timestep_pair):
        raise ValueError(
            "zero-recompute safe-report reuse requires the exact prior duration "
            "and timestep pair"
        )
    if (
        not rerun
        and prior.get("safe_to_apply") is not True
        and reclassified_case_count == 0
    ):
        raise RuntimeError(
            "zero-recompute reuse requires a fully safe prior report or validated "
            "velocity-resolved reclassification"
        )
    duration_extended = audit_duration_s > prior_duration + tolerance
    timestep_refined = not same_timestep_pair
    if not rerun and reclassified_case_count:
        continuation_axis = "zero_recompute_velocity_resolved_reclassification"
    elif not rerun:
        continuation_axis = "zero_recompute_safe_report_reuse"
    elif duration_extended and timestep_refined:
        continuation_axis = "duration_and_timestep_refinement"
    elif duration_extended:
        continuation_axis = "duration_extension"
    else:
        continuation_axis = "timestep_refinement"
    lineage = {
        "prior_audit_report": str(prior_report_path.resolve()),
        "prior_audit_report_sha256": prior_hash,
        "prior_audit_duration_s": prior_duration,
        "prior_audit_time_steps_s": [
            prior_coarse_time_step_s,
            prior_fine_time_step_s,
        ],
        "continuation_audit_duration_s": audit_duration_s,
        "continuation_audit_time_steps_s": [
            coarse_time_step_s,
            fine_time_step_s,
        ],
        "continuation_axis": continuation_axis,
        "prior_audit_runtime_source_sha256": dict(prior_runtime),
        "current_audit_runtime_source_sha256": current_runtime,
        "added_audit_runtime_sources": sorted(added_runtime_sources),
        "audit_utility_source_changed_for_continuation": (
            str(prior_runtime["mot_multilevel/timeout_audit.py"])
            != current_runtime["mot_multilevel/timeout_audit.py"]
        ),
        "retained_resolved_case_count": len(retained),
        "velocity_resolved_reclassified_case_count": reclassified_case_count,
        "rerun_case_count": len(rerun),
        "zero_recompute_safe_report_reuse": bool(
            not rerun and reclassified_case_count == 0
        ),
        "zero_recompute_velocity_resolved_reclassification": bool(
            not rerun and reclassified_case_count > 0
        ),
    }
    return retained, rerun, retained_payloads, lineage


def _apply_corrections(
    paths: CampaignPaths,
    outcomes: Sequence[TimeoutOutcome],
    *,
    report_path: Path,
    audit_directory: Path,
    initial_cases: Sequence[TimeoutCase],
    initial_snapshot: Mapping[str, str | None],
    intent_path: Path,
    study_key: str,
) -> dict[str, object]:
    completion_issues = _campaign_completion_issues(paths, study_key)
    if completion_issues:
        raise RuntimeError(
            "--apply completion check failed: " + "; ".join(completion_issues)
        )
    if not all(outcome.safe_to_apply for outcome in outcomes):
        raise RuntimeError("--apply is refused while any timeout remains unresolved")
    rescanned_cases = scan_completed_timeouts(paths, study_key)
    if _case_fingerprints(rescanned_cases) != _case_fingerprints(initial_cases):
        raise RuntimeError("timeout case set changed during audit; rerun the audit")
    intent = _read_json(intent_path)
    expected_report_hash = str(intent.get("audit_report_sha256", ""))
    initial_report_hash = _sha256(report_path)
    if not expected_report_hash or initial_report_hash != expected_report_hash:
        raise RuntimeError("apply intent does not bind the current audit report")
    initial_intent_hash = _sha256(intent_path)
    audit_report = _read_json(report_path)
    raw_report_cases = audit_report.get("cases", [])
    if not isinstance(raw_report_cases, list) or not all(
        isinstance(payload, Mapping) for payload in raw_report_cases
    ):
        raise RuntimeError("audit report cases are malformed")
    report_case_payloads: dict[
        tuple[int, int, int, str], Mapping[str, object]
    ] = {}
    for payload in raw_report_cases:
        key = _payload_case_key(payload)
        if key in report_case_payloads:
            raise RuntimeError(f"audit report contains duplicate case {key}")
        report_case_payloads[key] = payload
    outcome_by_key = {_case_key(outcome.case): outcome for outcome in outcomes}
    outcome_keys = set(outcome_by_key)
    if outcome_keys != set(report_case_payloads):
        raise RuntimeError(
            "audit report per-case provenance does not exactly match apply outcomes"
        )
    search_configuration_counts: dict[tuple[float, float, float], int] = {}
    for key in sorted(outcome_keys):
        payload = report_case_payloads[key]
        duration, coarse_step, fine_step = _payload_audit_context(
            outcome_by_key[key].case, payload
        )
        search_key = (duration, coarse_step, fine_step)
        search_configuration_counts[search_key] = (
            search_configuration_counts.get(search_key, 0) + 1
        )
    search_configurations_used = [
        {
            "max_simulation_time_s": key[0],
            "coarse_time_step_s": key[1],
            "fine_time_step_s": key[2],
            "corrected_sample_count": count,
        }
        for key, count in sorted(search_configuration_counts.items())
    ]
    _verify_snapshot(initial_snapshot)
    modified_files = _modified_product_paths(paths, outcomes, study_key)
    backups = _backup_files(paths, modified_files, audit_directory)
    for record in backups:
        expected = initial_snapshot.get(record["source"])
        if expected != record["sha256"]:
            raise RuntimeError(f"backup does not match pre-audit snapshot: {record['source']}")
    # Close the backup-window race: nothing may change after its individual
    # backup was copied and before the first production write begins.
    _verify_snapshot(initial_snapshot)
    grouped: dict[int, list[TimeoutOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.case.point.point_index, []).append(outcome)
    try:
        if (
            _sha256(report_path) != initial_report_hash
            or _sha256(intent_path) != initial_intent_hash
        ):
            raise RuntimeError("audit report or apply intent changed before production write")
        point_summaries: dict[int, Mapping[str, object]] = {}
        for point_index, point_outcomes in sorted(grouped.items()):
            case = point_outcomes[0].case
            replacements = {
                (
                    outcome.case.sample.disc_index,
                    outcome.case.sample.point_index,
                ): outcome.replacement_sample
                for outcome in point_outcomes
            }
            samples = load_capture_velocity_samples(case.paths.final_samples_csv)
            corrected = [
                replacements.get((sample.disc_index, sample.point_index), sample)
                for sample in samples
            ]
            if any(sample is None for sample in corrected):
                raise RuntimeError("a resolved timeout has no replacement sample")
            save_samples_atomic(case.paths.partial_samples_csv, corrected)
            save_samples_atomic(case.paths.final_samples_csv, corrected)
            search = _production_search(case.run_metadata)
            signature_payload = case.run_metadata["run_signature_payload"]
            if not isinstance(signature_payload, Mapping):
                raise RuntimeError("run signature payload disappeared during apply")
            base_summary = analyze_completed_samples(
                corrected,
                search,
                case.paths,
                signature=str(case.run_metadata["run_signature_sha256"]),
                geometry_hash=str(signature_payload["geometry_sha256"]),
                plot_context=_plot_context(case.point),
            )
            point_summaries[point_index] = _reanalyze_with_velocity_overrides(
                case,
                corrected,
                search,
                base_summary,
                point_outcomes,
                report_path=report_path,
                report_sha256=initial_report_hash,
            )

        revision_hash, revision_payload = _dataset_revision(
            outcomes, report_path, study_key
        )
        provenance = {
            "study_key": study_key,
            "audit_report_json": str(report_path.resolve()),
            "apply_intent_json": str(intent_path.resolve()),
            "apply_result_json": str(
                (audit_directory / APPLY_RESULT_NAME).resolve()
            ),
            "applied_utc": _utc_now(),
            "base_run_signatures_preserved": True,
            "hybrid_dataset": True,
            "hybrid_dataset_note": (
                "Only audited timeout samples use the longer-duration and/or finer-"
                "timestep searches documented per case in the immutable audit report; "
                "all other samples retain the production 50 ms, 5 us search."
            ),
            "dataset_revision_sha256": revision_hash,
            "per_case_audit_search_configs_are_authoritative": True,
            "audited_timeout_search_configurations": search_configurations_used,
            "velocity_tolerance_m_per_s": AUDIT_VELOCITY_TOLERANCE_M_PER_S,
            "velocity_resolved_capture_count": sum(
                outcome.status == "velocity_resolved_capture"
                for outcome in outcomes
            ),
            "velocity_resolved_capture_note": (
                "Any nonmonotone capture band is integrated from its complete "
                "dual-timestep boolean mask in the immutable audit report. Its "
                "scalar CSV row is a conservative zero-capture compatibility "
                "fallback and is not used for corrected spectra or loading."
            ),
        }
        for point_index, point_outcomes in sorted(grouped.items()):
            case = point_outcomes[0].case
            point_provenance = {
                **provenance,
                "corrected_samples": [
                    {
                        "disc_index": outcome.case.sample.disc_index,
                        "point_index": outcome.case.sample.point_index,
                        "action": outcome.status,
                        "audit_capture_search_configs": report_case_payloads[
                            _case_key(outcome.case)
                        ]["audit_capture_search_configs"],
                    }
                    for outcome in point_outcomes
                ],
            }
            loading_payload = _read_json(case.paths.loading_json)
            loading_history = list(
                loading_payload.get("post_campaign_timeout_audits", [])
            )
            loading_history.append(point_provenance)
            loading_payload["post_campaign_timeout_audits"] = loading_history
            loading_payload["updated_utc"] = _utc_now()
            _atomic_write_json(case.paths.loading_json, loading_payload)

            summary_payload = _read_json(case.paths.capture_summary_json)
            summary_history = list(
                summary_payload.get("post_campaign_timeout_audits", [])
            )
            summary_history.append(point_provenance)
            summary_payload["post_campaign_timeout_audits"] = summary_history
            summary_payload["updated_utc"] = _utc_now()
            summary_payload["loading_rate"] = loading_payload
            _atomic_write_json(case.paths.capture_summary_json, summary_payload)
            point_summaries[point_index] = summary_payload

            metadata = _read_json(case.paths.metadata_json)
            metadata["loading_rate"] = loading_payload
            history = list(metadata.get("post_campaign_timeout_audits", []))
            history.append(point_provenance)
            metadata["post_campaign_timeout_audits"] = history
            metadata["updated_utc"] = _utc_now()
            _atomic_write_json(case.paths.metadata_json, metadata)

        _rebuild_study_aggregate(paths, study_key)
        sweep_record = {
            **provenance,
            "corrected_timeout_count": len(outcomes),
            "corrected_point_count": len(grouped),
        }
        _append_provenance(paths.study_metadata_json(study_key), sweep_record)
        _append_provenance(paths.metadata_json, sweep_record)
        _validate_post_apply(paths, outcomes, study_key)
        if (
            _sha256(report_path) != initial_report_hash
            or _sha256(intent_path) != initial_intent_hash
        ):
            raise RuntimeError("audit report or apply intent changed during apply")
        post_hashes = _snapshot_files(modified_files, require_all=True)
        application = {
            "backups": backups,
            "pre_apply_sha256": {
                str(path.resolve()): initial_snapshot[str(path.resolve())]
                for path in modified_files
            },
            "post_apply_sha256": post_hashes,
            "dataset_revision_sha256": revision_hash,
            "dataset_revision_payload": revision_payload,
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "applied",
            "production_modified": True,
            "audit_report_json": str(report_path.resolve()),
            "audit_report_sha256": initial_report_hash,
            "apply_intent_json": str(intent_path.resolve()),
            "apply_intent_sha256": initial_intent_hash,
            **application,
        }
        # This record is part of the transaction: if it cannot be committed,
        # the surrounding exception handler restores every production backup.
        _atomic_write_json(audit_directory / APPLY_RESULT_NAME, result)
        return result
    except BaseException as error:
        rollback_errors = _restore_backups(backups)
        if not rollback_errors:
            try:
                _verify_snapshot(initial_snapshot)
            except Exception as verify_error:
                rollback_errors.append(
                    f"snapshot verification: {type(verify_error).__name__}: {verify_error}"
                )
        detail = f"{type(error).__name__}: {error}"
        if rollback_errors:
            detail += "; rollback errors: " + "; ".join(rollback_errors)
        raise ApplyTransactionError(
            detail,
            rollback_complete=not rollback_errors,
        ) from error


def _run_timeout_audit(
    *,
    paths: CampaignPaths | None = None,
    audit_directory: Path | None = None,
    apply: bool = False,
    study_key: str = DETUNING_STUDY_KEY,
    workers: int = 1,
    audit_duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    prior_report_path: Path | None = None,
    classifier: Classifier = classify_multilevel_loading_trajectory,
    rebisector: Rebisector = find_multilevel_capture_velocity,
) -> dict[str, object]:
    """Run the audit, write its report, and optionally apply safe corrections."""

    if study_key not in {RAW_STUDY_KEY, DETUNING_STUDY_KEY}:
        raise ValueError(f"timeout audit does not support study {study_key!r}")
    if not 1 <= workers <= MAX_AUDIT_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_AUDIT_WORKERS}")
    _validate_audit_search_parameters(
        audit_duration_s, coarse_time_step_s, fine_time_step_s
    )
    campaign_paths = paths or default_refined_campaign_paths()
    destination = audit_directory or _default_audit_directory(
        campaign_paths, study_key
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    report_path = destination / AUDIT_REPORT_NAME
    apply_result_path = destination / APPLY_RESULT_NAME
    completion_issues = _campaign_completion_issues(campaign_paths, study_key)
    common_report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "apply_requested": bool(apply),
        "production_modified": False,
        "production_modified_at_report_write": False,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "study_key": study_key,
        "scope": (
            f"completed refined {study_key} points with saved timeout bracket endpoints"
        ),
        "worker_count": workers,
        "parallel_execution": "spawn-process" if workers > 1 else "serial",
        "continuation_requested": prior_report_path is not None,
        "audit_duration_s": audit_duration_s,
        "audit_time_steps_s": [coarse_time_step_s, fine_time_step_s],
        "per_case_audit_search_configs_are_authoritative": True,
        "velocity_tolerance_m_per_s": AUDIT_VELOCITY_TOLERANCE_M_PER_S,
        "audit_runtime_source_sha256": _audit_runtime_source_hashes(),
        "source_provenance_note": (
            "Recorded production hashes are verified where present in each run "
            "signature; additional audit/runtime sources were not signed by the "
            "original campaign and are recorded here at audit time."
        ),
        "campaign_complete": not completion_issues,
        "campaign_completion_issues": completion_issues,
        "production_statistics_root": str(campaign_paths.statistics.resolve()),
        "production_figures_root": str(campaign_paths.figures.resolve()),
        "report_json": str(report_path.resolve()),
        "report_is_immutable_after_first_write": True,
    }
    if apply and completion_issues:
        report = {
            **common_report,
            "status": "apply_preflight_refused",
            "timeout_case_count": 0,
            "resolved_case_count": 0,
            "unresolved_case_count": 0,
            "safe_to_apply": False,
            "cases": [],
            "predicted_loading_changes": [],
        }
        _atomic_write_json(report_path, report)
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "refused",
            "production_modified": False,
            "audit_report_json": str(report_path.resolve()),
            "audit_report_sha256": _sha256(report_path),
            "error": "campaign is incomplete: " + "; ".join(completion_issues),
        }
        _atomic_write_json(apply_result_path, result)
        raise RuntimeError(result["error"])

    try:
        cases = scan_completed_timeouts(campaign_paths, study_key)
    except Exception as error:
        report = {
            **common_report,
            "status": "audit_failed",
            "timeout_case_count": 0,
            "resolved_case_count": 0,
            "unresolved_case_count": 0,
            "safe_to_apply": False,
            "cases": [],
            "predicted_loading_changes": [],
            "audit_errors": [f"{type(error).__name__}: {error}"],
        }
        _atomic_write_json(report_path, report)
        if apply:
            _atomic_write_json(
                apply_result_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_utc": _utc_now(),
                    "status": "refused",
                    "production_modified": False,
                    "audit_report_json": str(report_path.resolve()),
                    "audit_report_sha256": _sha256(report_path),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise

    try:
        initial_snapshot = (
            _snapshot_files(
                _all_study_product_paths(campaign_paths, study_key),
                require_all=True,
            )
            if apply
            else {}
        )
    except Exception as error:
        report = {
            **common_report,
            "status": "apply_snapshot_refused",
            "timeout_case_count": len(cases),
            "resolved_case_count": 0,
            "unresolved_case_count": len(cases),
            "safe_to_apply": False,
            "cases": [],
            "predicted_loading_changes": [],
            "audit_errors": [f"{type(error).__name__}: {error}"],
        }
        _atomic_write_json(report_path, report)
        _atomic_write_json(
            apply_result_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "refused",
                "production_modified": False,
                "audit_report_json": str(report_path.resolve()),
                "audit_report_sha256": _sha256(report_path),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    retained_outcomes: list[TimeoutOutcome] = []
    rerun_cases = list(cases)
    retained_payloads: dict[
        tuple[int, int, int, str], dict[str, object]
    ] = {}
    continuation_lineage: dict[str, object] = {}
    if prior_report_path is not None:
        try:
            (
                retained_outcomes,
                rerun_cases,
                retained_payloads,
                continuation_lineage,
            ) = _prepare_continuation(
                prior_report_path,
                cases,
                paths=campaign_paths,
                study_key=study_key,
                audit_duration_s=audit_duration_s,
                coarse_time_step_s=coarse_time_step_s,
                fine_time_step_s=fine_time_step_s,
            )
        except Exception as error:
            report = {
                **common_report,
                "status": "continuation_refused",
                "timeout_case_count": len(cases),
                "resolved_case_count": 0,
                "unresolved_case_count": len(cases),
                "safe_to_apply": False,
                "cases": [],
                "predicted_loading_changes": [],
                "audit_errors": [f"{type(error).__name__}: {error}"],
            }
            _atomic_write_json(report_path, report)
            raise

    new_outcomes, audit_errors = _audit_cases(
        rerun_cases,
        workers=workers,
        audit_duration_s=audit_duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
        classifier=classifier,
        rebisector=rebisector,
    )
    outcome_by_key = {
        _case_key(outcome.case): outcome
        for outcome in [*retained_outcomes, *new_outcomes]
    }
    outcomes = [
        outcome_by_key[_case_key(case)]
        for case in cases
        if _case_key(case) in outcome_by_key
    ]
    case_payloads = dict(retained_payloads)
    case_payloads.update(
        {
            _case_key(outcome.case): _outcome_payload(
                outcome,
                audit_duration_s,
                coarse_time_step_s,
                fine_time_step_s,
            )
            for outcome in new_outcomes
        }
    )
    if prior_report_path is not None:
        expected_prior_hash = str(
            continuation_lineage["prior_audit_report_sha256"]
        )
        if _sha256(prior_report_path) != expected_prior_hash:
            audit_errors.append(
                {
                    "stage": "continuation_provenance",
                    "error": "prior audit report changed during continuation",
                }
            )
    try:
        predicted_changes = _predicted_loading_changes(outcomes)
    except Exception as error:
        predicted_changes = []
        audit_errors.append(
            {"stage": "predicted_loading_changes", "error": f"{type(error).__name__}: {error}"}
        )
    safe_to_apply = bool(
        not completion_issues
        and not audit_errors
        and len(outcomes) == len(cases)
        and all(outcome.safe_to_apply for outcome in outcomes)
    )
    report: dict[str, object] = {
        **common_report,
        **continuation_lineage,
        "status": "audited",
        "timeout_case_count": len(cases),
        "upper_endpoint_timeout_case_count": sum(
            case.endpoint_kind == "upper" for case in cases
        ),
        "lower_endpoint_timeout_case_count": sum(
            case.endpoint_kind == "lower" for case in cases
        ),
        "resolved_case_count": sum(outcome.safe_to_apply for outcome in outcomes),
        "unresolved_case_count": (
            sum(not outcome.safe_to_apply for outcome in outcomes) + len(audit_errors)
        ),
        "safe_to_apply": safe_to_apply,
        "audit_errors": audit_errors,
        "cases": [
            case_payloads[_case_key(case)]
            for case in cases
            if _case_key(case) in case_payloads
        ],
        "predicted_loading_changes": predicted_changes,
    }
    _atomic_write_json(report_path, report)
    if apply and not safe_to_apply:
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "refused",
            "production_modified": False,
            "audit_report_json": str(report_path.resolve()),
            "audit_report_sha256": _sha256(report_path),
            "error": "audit contains unresolved cases or errors",
        }
        _atomic_write_json(apply_result_path, result)
        raise RuntimeError(result["error"])
    if apply and not outcomes:
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "no_action",
            "production_modified": False,
            "audit_report_json": str(report_path.resolve()),
            "audit_report_sha256": _sha256(report_path),
            "reason": "the completed campaign contains no saved endpoint timeouts",
        }
        _atomic_write_json(apply_result_path, result)
        report["application_result"] = result
        return report
    if apply:
        intent_path = destination / APPLY_INTENT_NAME
        intent = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "ready_to_apply",
            "audit_report_json": str(report_path.resolve()),
            "audit_report_sha256": _sha256(report_path),
            "timeout_case_fingerprints": [
                {
                    "point_index": key[0],
                    "disc_index": key[1],
                    "point_index_within_disc": key[2],
                    "timeout_endpoint_kind": key[3],
                    "source_sample_sha256": value[0],
                    "source_metadata_sha256": value[1],
                }
                for key, value in sorted(_case_fingerprints(cases).items())
            ],
            "pre_audit_production_sha256": initial_snapshot,
        }
        _atomic_write_json(intent_path, intent)
        try:
            result = _apply_corrections(
                campaign_paths,
                outcomes,
                report_path=report_path,
                audit_directory=destination,
                initial_cases=cases,
                initial_snapshot=initial_snapshot,
                intent_path=intent_path,
                study_key=study_key,
            )
        except BaseException as error:
            rollback_complete = getattr(error, "rollback_complete", True)
            result = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "failed",
                "production_modified": not rollback_complete,
                "rollback_complete": bool(rollback_complete),
                "audit_report_json": str(report_path.resolve()),
                "audit_report_sha256": _sha256(report_path),
                "apply_intent_json": str(intent_path.resolve()),
                "apply_intent_sha256": _sha256(intent_path),
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_write_json(apply_result_path, result)
            raise
        report["application_result"] = result
    return report


def run_timeout_audit(
    *,
    paths: CampaignPaths | None = None,
    audit_directory: Path | None = None,
    apply: bool = False,
    study_key: str = DETUNING_STUDY_KEY,
    workers: int = 1,
    audit_duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    prior_report_path: Path | None = None,
    classifier: Classifier = classify_multilevel_loading_trajectory,
    rebisector: Rebisector = find_multilevel_capture_velocity,
) -> dict[str, object]:
    """Run one read-only audit, or an exclusively locked apply transaction."""

    campaign_paths = paths or default_refined_campaign_paths()
    keywords = {
        "paths": campaign_paths,
        "audit_directory": audit_directory,
        "apply": apply,
        "study_key": study_key,
        "workers": workers,
        "audit_duration_s": audit_duration_s,
        "coarse_time_step_s": coarse_time_step_s,
        "fine_time_step_s": fine_time_step_s,
        "prior_report_path": prior_report_path,
        "classifier": classifier,
        "rebisector": rebisector,
    }
    if not apply:
        return _run_timeout_audit(**keywords)
    with _exclusive_campaign_apply_lock(campaign_paths):
        return _run_timeout_audit(**keywords)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit completed refined relationship-sweep timeout endpoints at "
            "a configurable duration and two timesteps; production is read-only unless --apply "
            "is explicit"
        )
    )
    parser.add_argument("--statistics-root", type=Path, default=None)
    parser.add_argument("--figures-root", type=Path, default=None)
    parser.add_argument("--audit-directory", type=Path, default=None)
    parser.add_argument(
        "--study",
        choices=tuple(CLI_STUDIES),
        default="detuning",
        help="independent relationship study to audit (default: detuning)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="spawn-process case workers; use 24 for the full campaign host",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=1.0e3 * AUDIT_DURATION_S,
        help="audit trajectory horizon in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--coarse-step-us",
        type=float,
        default=1.0e6 * COARSE_TIME_STEP_S,
        help="coarse audit trajectory timestep in microseconds (default: 5)",
    )
    parser.add_argument(
        "--fine-step-us",
        type=float,
        default=1.0e6 * FINE_TIME_STEP_S,
        help="fine audit trajectory timestep in microseconds (default: 2.5)",
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=None,
        help=(
            "immutable prior audit report; retain its resolved cases and rerun only "
            "unresolved/error cases using a longer duration and/or a strictly finer "
            "dual-timestep pair; exact settings reuse a fully safe report without "
            "trajectory recomputation"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="after writing the audit report, back up and apply resolved corrections",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if (args.statistics_root is None) != (args.figures_root is None):
        raise SystemExit("--statistics-root and --figures-root must be supplied together")
    paths = (
        None
        if args.statistics_root is None
        else CampaignPaths(
            statistics=args.statistics_root,
            figures=args.figures_root,
        )
    )
    report = run_timeout_audit(
        paths=paths,
        audit_directory=args.audit_directory,
        apply=args.apply,
        study_key=CLI_STUDIES[args.study],
        workers=args.workers,
        audit_duration_s=1.0e-3 * args.duration_ms,
        coarse_time_step_s=1.0e-6 * args.coarse_step_us,
        fine_time_step_s=1.0e-6 * args.fine_step_us,
        prior_report_path=args.prior_report,
    )
    print(json.dumps(_json_ready(report), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_DURATION_S",
    "AUDIT_VELOCITY_TOLERANCE_M_PER_S",
    "COARSE_TIME_STEP_S",
    "CLI_STUDIES",
    "FINE_TIME_STEP_S",
    "TimeoutCase",
    "TimeoutOutcome",
    "audit_timeout_case",
    "build_argument_parser",
    "main",
    "run_timeout_audit",
    "scan_completed_timeouts",
    "scan_completed_detuning_timeouts",
]
