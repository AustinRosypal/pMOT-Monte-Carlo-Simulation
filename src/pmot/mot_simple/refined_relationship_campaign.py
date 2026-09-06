"""Refined loading-relationship campaign for the effective two-level MOT.

This campaign is the two-level counterpart of the September 2026 multilevel
relationship study.  The three loading sweeps are independent and sequential:

1. loading rate versus the on-resonance, single-beam saturation ``s0``;
2. loading rate versus the detuning-reduced, single-beam saturation ``s_eff``;
3. loading rate versus cooling detuning ``Delta/Gamma``.

Every point reuses one seeded set of 25 full-sphere incident-direction discs
and 25 independent uniform-area launch points per disc.  The implementation is
strictly isolated under ``mot_simple`` and never reads or writes multilevel
campaign products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..configuration import AntiHelmholtzCoilConfig, MOTApparatusConfig
from ..launch_geometry import (
    DiscSample,
    PointSample,
    sample_disc_points,
    sample_incident_disc_full_sphere,
)
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import (
    SimpleMOTConfig,
    default_simple_mot_apparatus,
    default_simple_mot_config,
    simple_mot_paths,
)
from .batched_sampling import classify_trajectory_batch
from .power_loading_study import (
    LOADING_BY_DISC_FIELDNAMES,
    SPECTRUM_FIELDNAMES,
    StudyPaths,
    geometry_csv_text,
    geometry_rows,
    plot_capture_velocity_vs_impact_parameter,
    plot_clustered_cross_section,
    plot_loading_rate_by_disc,
    save_samples_atomic,
    validate_checkpoint_samples,
)
from .sampling import (
    CaptureSearchConfig,
    CaptureVelocitySample,
    classify_trajectory,
    load_capture_velocity_samples,
)
from .simulation import SimpleMOTBeam, build_simple_mot_beams
from .timeout_audit import (
    AUDIT_DURATION_S,
    AdaptiveAuditLevel,
    AdaptiveAuditEvidence,
    COARSE_TIME_STEP_S,
    DEFAULT_ADAPTIVE_AUDIT_LEVELS,
    FINE_TIME_STEP_S,
    TimeoutAuditCase,
    TimeoutAuditOutcome,
    VelocityResolvedCaptureOverride,
    adaptive_audit_evidence_to_payload,
    audit_capture_boundaries_on_velocity_grid_batched,
    audit_capture_boundary,
    audit_searches,
    audit_zero_capture_boundaries_batched,
    calculate_clustered_cross_section_with_overrides,
    calculate_disc_clustered_loading_with_overrides,
    velocity_overrides_from_payload,
    velocity_overrides_to_payload,
)


RAW_SATURATION_VALUES: tuple[float, ...] = (
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
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    125.0,
)
EFFECTIVE_SATURATION_VALUES: tuple[float, ...] = tuple(
    0.25 * index for index in range(1, 21)
)
DETUNING_N_VALUES: tuple[float, ...] = tuple(
    -0.25 * index for index in range(2, 25)
)

CAMPAIGN_NAME = (
    "refined_relationships_full_sphere_25x25_r15mm_"
    "27mW_reference_optimized_v3_20260906"
)
RAW_STUDY_KEY = "01_raw_saturation"
EFFECTIVE_STUDY_KEY = "02_effective_saturation"
DETUNING_STUDY_KEY = "03_detuning"
STUDY_ORDER = (RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY, DETUNING_STUDY_KEY)

DEFAULT_DISC_COUNT = 25
DEFAULT_POINTS_PER_DISC = 25
DEFAULT_DISC_RADIUS_M = 15.0e-3
DEFAULT_SEED = 20260903
DEFAULT_COOLING_POWER_W_PER_BEAM = 27.0e-3
DEFAULT_WORKER_COUNT = min(24, os.cpu_count() or 1)
ZERO_GRID_RAY_BATCH_SIZE = 5
PROGRESS_EVERY = 10
CHECKPOINT_EVERY = 25
CAMPAIGN_SCHEMA_VERSION = 4
POINT_SCHEMA_VERSION = 4
TRAPPED_TERMINATION_REASONS = {"two_core_entries", "bounded_core_residence"}

AGGREGATE_FIELDNAMES: tuple[str, ...] = (
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

ENDPOINT_AUDIT_FIELDNAMES: tuple[str, ...] = (
    "disc_index",
    "point_index",
    "base_timeout_detected",
    "base_evaluation_timeout_count",
    "base_evaluation_timeout_speeds_m_per_s_json",
    "accepted_max_simulation_time_s",
    "accepted_coarse_time_step_s",
    "accepted_time_step_s",
    "adaptive_audit_level_count",
    "adaptive_audit_evidence_json",
    "adaptive_max_duration_s",
    "adaptive_coarse_time_step_s",
    "adaptive_fine_time_step_s",
    "accepted_trapped_velocity_lower_m_per_s",
    "accepted_untrapped_velocity_upper_m_per_s",
    "accepted_lower_classification",
    "accepted_upper_classification",
    "base_capture_velocity_m_per_s",
    "coarse_capture_velocity_m_per_s",
    "fine_capture_velocity_m_per_s",
    "coarse_trapped_velocity_lower_m_per_s",
    "coarse_untrapped_velocity_upper_m_per_s",
    "fine_trapped_velocity_lower_m_per_s",
    "fine_untrapped_velocity_upper_m_per_s",
    "coarse_lower_classification",
    "coarse_upper_classification",
    "fine_lower_classification",
    "fine_upper_classification",
    "timeout_node_coarse_results_json",
    "timeout_node_fine_results_json",
    "coarse_evaluation_timeout_count",
    "coarse_evaluation_timeout_speeds_m_per_s_json",
    "fine_evaluation_timeout_count",
    "fine_evaluation_timeout_speeds_m_per_s_json",
    "pre_adaptive_coarse_evaluation_timeout_count",
    "pre_adaptive_coarse_evaluation_timeout_speeds_m_per_s_json",
    "pre_adaptive_fine_evaluation_timeout_count",
    "pre_adaptive_fine_evaluation_timeout_speeds_m_per_s_json",
    "complete_boundary_audit_evidence_json",
    "positive_boundary_grid_base_level_index",
    "diagnostic_scalar_boundary_level_index",
    "diagnostic_scalar_boundary_converged",
    "diagnostic_scalar_grid_agreement",
    "timeout_resolution_status",
    "timeout_resolution_reason",
    "zero_threshold_audit_status",
    "zero_threshold_audit_reason",
    "zero_threshold_grid_evidence_json",
    "positive_boundary_grid_audit_status",
    "positive_boundary_grid_audit_reason",
    "positive_boundary_grid_evidence_json",
    "positive_boundary_grid_duration_s",
    "positive_boundary_grid_coarse_time_step_s",
    "positive_boundary_grid_fine_time_step_s",
    "velocity_resolved_override",
)


@dataclass(frozen=True, slots=True)
class RelationshipPoint:
    """One point in one independent loading relationship."""

    study_key: str
    point_index: int
    scan_variable: str
    scan_value: float
    cooling_power_w_per_beam: float
    cooling_detuning_n: float
    cooling_detuning_hz: float
    on_resonance_saturation: float
    effective_saturation: float

    @property
    def slug(self) -> str:
        value = format(self.scan_value, ".12g").replace("-", "m").replace(".", "p")
        return f"{self.point_index:03d}_{self.scan_variable}_{value}"


@dataclass(frozen=True, slots=True)
class CampaignPaths:
    """Stable statistics and figure roots for the two-level campaign."""

    statistics: Path
    figures: Path

    @property
    def metadata_json(self) -> Path:
        return self.statistics / "campaign_metadata.json"

    @property
    def geometry_csv(self) -> Path:
        return self.statistics / "launch_geometry.csv"

    def study_statistics(self, study_key: str) -> Path:
        return self.statistics / study_key

    def study_figures(self, study_key: str) -> Path:
        return self.figures / study_key

    def aggregate_csv(self, study_key: str) -> Path:
        return self.study_statistics(study_key) / "aggregate.csv"

    def study_metadata_json(self, study_key: str) -> Path:
        return self.study_statistics(study_key) / "sweep_metadata.json"

    def relationship_plot(self, study_key: str) -> Path:
        names = {
            RAW_STUDY_KEY: "loading_rate_vs_saturation_parameter.png",
            EFFECTIVE_STUDY_KEY: "loading_rate_vs_effective_saturation_parameter.png",
            DETUNING_STUDY_KEY: "loading_rate_vs_detuning.png",
        }
        return self.study_figures(study_key) / names[study_key]

    def point_paths(self, point: RelationshipPoint) -> StudyPaths:
        return StudyPaths(
            statistics=self.study_statistics(point.study_key) / "points" / point.slug,
            figures=self.study_figures(point.study_key) / "points" / point.slug,
        )


@dataclass(frozen=True, slots=True)
class CaptureWorkerResult:
    """A capture threshold plus evidence for automatic timeout resolution."""

    sample: CaptureVelocitySample
    audit_row: dict[str, object]
    velocity_override: VelocityResolvedCaptureOverride | None = None


def default_campaign_paths(root: Path | None = None) -> CampaignPaths:
    paths = simple_mot_paths(root)
    return CampaignPaths(
        statistics=paths["outputs_statistics_simple_mot"] / CAMPAIGN_NAME,
        figures=paths["outputs_figures_simple_mot"] / CAMPAIGN_NAME,
    )


def default_search_config(*, seed: int = DEFAULT_SEED) -> CaptureSearchConfig:
    """Return the exact 25-by-25 full-sphere campaign sampling controls."""

    return replace(
        CaptureSearchConfig(),
        disc_count=DEFAULT_DISC_COUNT,
        points_per_disc=DEFAULT_POINTS_PER_DISC,
        disc_radius_m=DEFAULT_DISC_RADIUS_M,
        include_center_point=False,
        analysis_velocity_min_m_per_s=0.0,
        seed=int(seed),
        save_every=CHECKPOINT_EVERY,
    )


def saturation_power_w_per_beam(
    saturation_parameter: float,
    *,
    beam_diameter_m: float = 12.7e-3,
    saturation_intensity_w_per_m2: float | None = None,
) -> float:
    """Convert single-beam Gaussian-center ``s0`` to beam power."""

    simple = default_simple_mot_config()
    saturation_intensity = (
        simple.saturation_intensity_w_per_m2
        if saturation_intensity_w_per_m2 is None
        else float(saturation_intensity_w_per_m2)
    )
    if saturation_parameter <= 0.0 or beam_diameter_m <= 0.0 or saturation_intensity <= 0.0:
        raise ValueError("saturation, beam diameter, and saturation intensity must be positive")
    radius_m = 0.5 * float(beam_diameter_m)
    return float(saturation_parameter * saturation_intensity * pi * radius_m**2 / 2.0)


def on_resonance_saturation_parameter(
    power_w_per_beam: float,
    *,
    beam_diameter_m: float = 12.7e-3,
    saturation_intensity_w_per_m2: float | None = None,
) -> float:
    """Return the single-beam Gaussian-center ``s0=I0/I_sat``."""

    simple = default_simple_mot_config()
    saturation_intensity = (
        simple.saturation_intensity_w_per_m2
        if saturation_intensity_w_per_m2 is None
        else float(saturation_intensity_w_per_m2)
    )
    if power_w_per_beam <= 0.0 or beam_diameter_m <= 0.0 or saturation_intensity <= 0.0:
        raise ValueError("power, beam diameter, and saturation intensity must be positive")
    radius_m = 0.5 * float(beam_diameter_m)
    peak_intensity = 2.0 * float(power_w_per_beam) / (pi * radius_m**2)
    return float(peak_intensity / saturation_intensity)


def detuning_reduction_denominator(detuning_n: float) -> float:
    if not np.isfinite(detuning_n):
        raise ValueError("detuning_n must be finite")
    return float(1.0 + (2.0 * float(detuning_n)) ** 2)


def effective_saturation_from_s0(s0: float, detuning_n: float) -> float:
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    return float(s0 / detuning_reduction_denominator(detuning_n))


def build_relationship_points(
    study_key: str,
    values: Sequence[float],
) -> tuple[RelationshipPoint, ...]:
    """Map one requested grid to exact powers, detunings, and saturation values."""

    simple = default_simple_mot_config()
    gamma_hz = simple.linewidth_hz
    baseline_n = simple.cooling_detuning_hz / gamma_hz
    reference_s0 = on_resonance_saturation_parameter(DEFAULT_COOLING_POWER_W_PER_BEAM)
    points: list[RelationshipPoint] = []
    for point_index, raw_value in enumerate(values):
        value = float(raw_value)
        if study_key == RAW_STUDY_KEY:
            if value <= 0.0:
                raise ValueError("raw saturation values must be positive")
            s0 = value
            detuning_n = baseline_n
            detuning_hz = simple.cooling_detuning_hz
            power = saturation_power_w_per_beam(s0)
            seff = effective_saturation_from_s0(s0, detuning_n)
            variable = "s0"
        elif study_key == EFFECTIVE_STUDY_KEY:
            if value <= 0.0:
                raise ValueError("effective saturation values must be positive")
            detuning_n = baseline_n
            detuning_hz = simple.cooling_detuning_hz
            seff = value
            s0 = value * detuning_reduction_denominator(detuning_n)
            power = saturation_power_w_per_beam(s0)
            variable = "seff"
        elif study_key == DETUNING_STUDY_KEY:
            if not np.isfinite(value) or value >= 0.0:
                raise ValueError("detuning values must be finite and red")
            detuning_n = value
            detuning_hz = value * gamma_hz
            power = DEFAULT_COOLING_POWER_W_PER_BEAM
            s0 = reference_s0
            seff = effective_saturation_from_s0(s0, detuning_n)
            variable = "detuning_n"
        else:
            raise ValueError(f"unknown relationship study: {study_key}")
        points.append(
            RelationshipPoint(
                study_key=study_key,
                point_index=point_index,
                scan_variable=variable,
                scan_value=value,
                cooling_power_w_per_beam=power,
                cooling_detuning_n=detuning_n,
                cooling_detuning_hz=detuning_hz,
                on_resonance_saturation=s0,
                effective_saturation=seff,
            )
        )
    return tuple(points)


def requested_points() -> dict[str, tuple[RelationshipPoint, ...]]:
    return {
        RAW_STUDY_KEY: build_relationship_points(RAW_STUDY_KEY, RAW_SATURATION_VALUES),
        EFFECTIVE_STUDY_KEY: build_relationship_points(
            EFFECTIVE_STUDY_KEY, EFFECTIVE_SATURATION_VALUES
        ),
        DETUNING_STUDY_KEY: build_relationship_points(
            DETUNING_STUDY_KEY, DETUNING_N_VALUES
        ),
    }


def generate_common_geometry(
    search: CaptureSearchConfig,
) -> tuple[list[DiscSample], list[PointSample]]:
    """Generate seeded full-sphere directions and uniform-area launch points."""

    if search.disc_count <= 0 or search.points_per_disc <= 0:
        raise ValueError("disc_count and points_per_disc must be positive")
    if search.include_center_point:
        raise ValueError("production geometry cannot contain a forced disc-center point")
    rng = np.random.default_rng(search.seed)
    discs: list[DiscSample] = []
    points: list[PointSample] = []
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc_full_sphere(
            disc_index, search.radial_distance_m, rng
        )
        discs.append(disc)
        points.extend(
            sample_disc_points(
                disc,
                search.points_per_disc,
                search.disc_radius_m,
                False,
                rng,
            )
        )
    return discs, points


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8", newline="")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, _json_text(payload))


def _csv_text(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    _atomic_write_text(path, _csv_text(rows, fieldnames))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    mot_simple_directory = Path(__file__).resolve().parent
    package_directory = mot_simple_directory.parent
    source_root = package_directory.parent
    files = (
        mot_simple_directory / "refined_relationship_campaign.py",
        mot_simple_directory / "configuration.py",
        mot_simple_directory / "simulation.py",
        mot_simple_directory / "sampling.py",
        mot_simple_directory / "power_loading_study.py",
        mot_simple_directory / "loading.py",
        mot_simple_directory / "timeout_audit.py",
        mot_simple_directory / "batched_sampling.py",
        package_directory / "configuration.py",
        package_directory / "beams.py",
        package_directory / "fields.py",
        package_directory / "magnetic_fields.py",
        package_directory / "state.py",
        package_directory / "launch_geometry.py",
        package_directory / "capture_statistics.py",
        package_directory / "loading.py",
    )
    return {
        path.relative_to(source_root).as_posix(): _sha256_file(path) for path in files
    }


def _git_provenance(root: Path) -> dict[str, object]:
    def command(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(command("status", "--porcelain")),
    }


def build_point_configuration(
    point: RelationshipPoint,
) -> tuple[MOTApparatusConfig, SimpleMOTConfig, AntiHelmholtzCoilConfig, list[SimpleMOTBeam]]:
    """Build a synchronized two-level configuration for one relationship point."""

    apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=point.cooling_power_w_per_beam,
            detuning_hz=point.cooling_detuning_hz,
        ),
    )
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=point.cooling_detuning_hz,
    )
    coil = default_anti_helmholtz_config()
    beams = build_simple_mot_beams(apparatus, simple)
    if len(beams) != 6:
        raise RuntimeError("the two-level campaign requires exactly six cooling beams")
    if not all(
        np.isclose(
            beam.intensity_beam.power_w,
            point.cooling_power_w_per_beam,
            rtol=0.0,
            atol=1.0e-15,
        )
        for beam in beams
    ):
        raise RuntimeError("built cooling-beam powers do not match the point plan")
    if not all(
        np.isclose(beam.detuning_hz, point.cooling_detuning_hz, rtol=0.0, atol=1.0e-9)
        for beam in beams
    ):
        raise RuntimeError("built cooling-beam detunings do not match the point plan")
    return apparatus, simple, coil, beams


def _point_signature_payload(
    point: RelationshipPoint,
    search: CaptureSearchConfig,
    geometry_hash: str,
) -> dict[str, object]:
    apparatus, simple, coil, _ = build_point_configuration(point)
    return {
        "schema_version": POINT_SCHEMA_VERSION,
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "point": asdict(point),
        "search_config": asdict(search),
        "phase_space": "full_sphere",
        "geometry_sha256": geometry_hash,
        "apparatus_config": asdict(apparatus),
        "simple_mot_config": asdict(simple),
        "coil_config": asdict(coil),
        "physics_source_sha256": _source_hashes(),
        "execution_batching": {
            "zero_grid_ray_batch_size": ZERO_GRID_RAY_BATCH_SIZE,
            "interpretation": (
                "independent ray-velocity states share vectorized array calls only; "
                "no physical or statistical coupling"
            ),
        },
        "endpoint_timeout_policy": {
            "base_max_time_s": search.max_simulation_time_s,
            "audit_duration_s": AUDIT_DURATION_S,
            "audit_coarse_time_step_s": COARSE_TIME_STEP_S,
            "fine_time_step_s": FINE_TIME_STEP_S,
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
                "when monotone. Every scalar "
                "zero threshold is "
                "independently scanned from 0 to 30 m/s by 0.25 m/s at both timesteps, "
                "with finite-speed capture islands retained as direct boolean masks. "
                "Only non-definitive or timestep-disagreeing grid or positive-boundary "
                "nodes escalate to "
                "250 ms at 5/2.5 microseconds and then, only if still required, to "
                "400 ms at 2.5/1.25 microseconds. A node is accepted only when both "
                "timesteps give the same definitive trapped/escaped classification."
            ),
        },
    }


def _signature(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _campaign_signature(
    search: CaptureSearchConfig,
    geometry_hash: str,
) -> str:
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "campaign_name": CAMPAIGN_NAME,
        "study_order": list(STUDY_ORDER),
        "points": {
            key: [asdict(point) for point in values]
            for key, values in requested_points().items()
        },
        "search_config": asdict(search),
        "zero_grid_ray_batch_size": ZERO_GRID_RAY_BATCH_SIZE,
        "phase_space": "full_sphere",
        "geometry_sha256": geometry_hash,
        "physics_source_sha256": _source_hashes(),
    }
    return _signature(payload)


def _valid_capture_endpoint(sample: CaptureVelocitySample) -> bool:
    if sample.upper_classification != "escaped":
        return False
    if sample.capture_velocity_m_per_s > 0.0:
        return sample.lower_classification in TRAPPED_TERMINATION_REASONS
    return sample.lower_classification not in {"timeout", "non_finite"}


def _sample_has_timeout(sample: CaptureVelocitySample) -> bool:
    return "timeout" in (sample.lower_classification, sample.upper_classification)


_WORKER_BEAMS: list[SimpleMOTBeam] | None = None
_WORKER_APPARATUS: MOTApparatusConfig | None = None
_WORKER_COIL: AntiHelmholtzCoilConfig | None = None
_WORKER_SIMPLE: SimpleMOTConfig | None = None
_WORKER_SEARCH: CaptureSearchConfig | None = None


def _initialize_worker(
    apparatus: MOTApparatusConfig,
    simple: SimpleMOTConfig,
    coil: AntiHelmholtzCoilConfig,
    search: CaptureSearchConfig,
) -> None:
    global _WORKER_BEAMS, _WORKER_APPARATUS, _WORKER_COIL, _WORKER_SIMPLE, _WORKER_SEARCH
    _WORKER_BEAMS = build_simple_mot_beams(apparatus, simple)
    _WORKER_APPARATUS = apparatus
    _WORKER_COIL = coil
    _WORKER_SIMPLE = simple
    _WORKER_SEARCH = search


def _evaluate_capture_instrumented(
    point: PointSample, search: CaptureSearchConfig
) -> tuple[
    CaptureVelocitySample,
    tuple[float, ...],
    tuple[tuple[float, object], ...],
]:
    """Return a threshold plus the complete evaluated-node ledger.

    Unlike the legacy search, this campaign explicitly evaluates zero speed
    first. If zero does not trap, the scalar result is deliberately zero and a
    direct-grid audit determines whether a finite-speed island exists. This
    both enforces the monotonic-threshold precondition and avoids redundant
    geometric halving far below the 0.25 m/s velocity resolution.
    """

    if _WORKER_BEAMS is None or _WORKER_COIL is None or _WORKER_SIMPLE is None:
        raise RuntimeError("capture worker was not initialized")
    evaluations = {}

    def evaluate(speed_m_per_s: float):
        speed = round(float(speed_m_per_s), 12)
        if speed not in evaluations:
            evaluations[speed] = classify_trajectory(
                _WORKER_BEAMS,
                point,
                speed,
                _WORKER_COIL,
                _WORKER_SIMPLE,
                search,
            )
        return evaluations[speed]

    zero_result = evaluate(0.0)
    if not zero_result.trapped:
        upper_speed = search.analysis_velocity_step_m_per_s
        upper_result = evaluate(upper_speed)
        sample = CaptureVelocitySample(
            disc_index=point.disc_index,
            point_index=point.point_index,
            theta_rad=point.theta_rad,
            phi_rad=point.phi_rad,
            theta_prime_rad=point.theta_prime_rad,
            s_m=point.s_m,
            radial_distance_m=point.radial_distance_m,
            initial_position_m=point.initial_position_m,
            incident_unit_vector=point.incident_unit_vector,
            capture_velocity_m_per_s=0.0,
            velocity_resolution_m_per_s=upper_speed,
            trapped_velocity_lower_m_per_s=0.0,
            untrapped_velocity_upper_m_per_s=upper_speed,
            lower_classification=zero_result.termination_reason,
            upper_classification=upper_result.termination_reason,
            lower_entered_trap_core=zero_result.entered_trap_core,
            upper_entered_trap_core=upper_result.entered_trap_core,
            lower_core_entry_count=zero_result.core_entry_count,
            upper_core_entry_count=upper_result.core_entry_count,
        )
    else:
        trial = max(search.analysis_velocity_step_m_per_s, search.initial_velocity_guess_m_per_s)
        trial_result = evaluate(trial)
        if trial_result.trapped:
            lower_speed = trial
            upper_speed = max(1.0, trial)
            for _ in range(search.max_bracket_iterations):
                upper_speed *= 2.0
                if not evaluate(upper_speed).trapped:
                    break
            else:
                raise RuntimeError("failed to find an untrapped upper capture bracket")
        else:
            upper_speed = trial
            lower_speed = trial
            for _ in range(search.max_bracket_iterations):
                candidate = 0.5 * lower_speed
                if candidate <= search.velocity_tolerance_m_per_s:
                    lower_speed = 0.0
                    break
                lower_speed = candidate
                if evaluate(lower_speed).trapped:
                    break
            else:
                raise RuntimeError("failed to find a trapped lower capture bracket")
        for _ in range(search.max_search_iterations):
            if upper_speed - lower_speed <= search.velocity_tolerance_m_per_s:
                break
            midpoint = round(0.5 * (lower_speed + upper_speed), 12)
            result = evaluations.get(midpoint)
            if result is None:
                result = classify_trajectory(
                    _WORKER_BEAMS,
                    point,
                    midpoint,
                    _WORKER_COIL,
                    _WORKER_SIMPLE,
                    search,
                )
                evaluations[midpoint] = result
            if result.trapped:
                lower_speed = midpoint
            else:
                upper_speed = midpoint
        lower_result = evaluations[round(lower_speed, 12)]
        upper_result = evaluations[round(upper_speed, 12)]
        sample = CaptureVelocitySample(
            disc_index=point.disc_index,
            point_index=point.point_index,
            theta_rad=point.theta_rad,
            phi_rad=point.phi_rad,
            theta_prime_rad=point.theta_prime_rad,
            s_m=point.s_m,
            radial_distance_m=point.radial_distance_m,
            initial_position_m=point.initial_position_m,
            incident_unit_vector=point.incident_unit_vector,
            capture_velocity_m_per_s=lower_speed,
            velocity_resolution_m_per_s=upper_speed - lower_speed,
            trapped_velocity_lower_m_per_s=lower_speed,
            untrapped_velocity_upper_m_per_s=upper_speed,
            lower_classification=lower_result.termination_reason,
            upper_classification=upper_result.termination_reason,
            lower_entered_trap_core=lower_result.entered_trap_core,
            upper_entered_trap_core=upper_result.entered_trap_core,
            lower_core_entry_count=lower_result.core_entry_count,
            upper_core_entry_count=upper_result.core_entry_count,
        )
    timeout_speeds = tuple(
        sorted(
            speed
            for speed, result in evaluations.items()
            if result.termination_reason == "timeout"
        )
    )
    evaluation_ledger = tuple(sorted(evaluations.items()))
    return sample, timeout_speeds, evaluation_ledger


def _evaluate_capture(
    point: PointSample, search: CaptureSearchConfig
) -> tuple[CaptureVelocitySample, tuple[float, ...]]:
    """Return a threshold and every timeout-contaminated evaluated speed."""

    sample, timeout_speeds, _ = _evaluate_capture_instrumented(point, search)
    return sample, timeout_speeds


def _apply_zero_threshold_audit(
    sample: CaptureVelocitySample,
    audit_row: dict[str, object],
    *,
    precomputed_outcome: TimeoutAuditOutcome | None = None,
) -> CaptureWorkerResult:
    """Fail closed on every zero threshold and retain any finite-speed island."""

    if sample.capture_velocity_m_per_s > 0.0:
        audit_row.update(
            {
                "accepted_trapped_velocity_lower_m_per_s": (
                    sample.trapped_velocity_lower_m_per_s
                ),
                "accepted_untrapped_velocity_upper_m_per_s": (
                    sample.untrapped_velocity_upper_m_per_s
                ),
                "accepted_lower_classification": sample.lower_classification,
                "accepted_upper_classification": sample.upper_classification,
                "zero_threshold_audit_status": "not_applicable",
                "zero_threshold_audit_reason": "positive scalar capture threshold",
                "velocity_resolved_override": False,
            }
        )
        return CaptureWorkerResult(sample, audit_row, None)
    if (
        _WORKER_APPARATUS is None
        or _WORKER_SIMPLE is None
        or _WORKER_COIL is None
        or _WORKER_SEARCH is None
    ):
        raise RuntimeError("capture worker was not initialized")
    rebisected_evidence: list[
        tuple[CaptureVelocitySample, tuple[float, ...], CaptureSearchConfig]
    ] = []

    def instrumented_rebisector(_beams, audit_point, _coil, _simple, audit_search):
        replacement, timeout_speeds = _evaluate_capture(audit_point, audit_search)
        rebisected_evidence.append((replacement, timeout_speeds, audit_search))
        return replacement

    outcome = precomputed_outcome
    if outcome is None:
        outcome = audit_capture_boundary(
            TimeoutAuditCase(
                sample=sample,
                apparatus=_WORKER_APPARATUS,
                simple_config=_WORKER_SIMPLE,
                coil_config=_WORKER_COIL,
                search_config=_WORKER_SEARCH,
            ),
            rebisector=instrumented_rebisector,
            scan_zero_threshold=True,
        )
    elif outcome.case.sample != sample:
        raise RuntimeError("precomputed zero-grid audit belongs to a different ray")
    if not outcome.resolved or outcome.replacement_sample is None:
        raise RuntimeError(
            "zero-threshold direct velocity audit failed for "
            f"{(sample.disc_index, sample.point_index)}: {outcome.reason}"
        )
    coarse_boundary = outcome.coarse_boundary_sample
    fine_boundary = outcome.fine_boundary_sample
    if coarse_boundary is None or fine_boundary is None:
        raise RuntimeError(
            "zero-threshold audit resolved without complete coarse/fine bracket "
            f"provenance for {(sample.disc_index, sample.point_index)}"
        )
    if fine_boundary != outcome.replacement_sample:
        raise RuntimeError(
            "zero-threshold audit fine bracket does not equal its replacement "
            f"sample for {(sample.disc_index, sample.point_index)}"
        )
    accepted = fine_boundary
    coarse_timeout_speeds: tuple[float, ...] = ()
    fine_timeout_speeds: tuple[float, ...] = ()
    if outcome.status == "rebisected":
        if len(rebisected_evidence) != 2:
            raise RuntimeError("zero-threshold re-bisection evidence is incomplete")
        coarse, coarse_timeout_speeds, _ = rebisected_evidence[0]
        fine, fine_timeout_speeds, _ = rebisected_evidence[1]
        if (
            coarse_timeout_speeds
            or fine_timeout_speeds
            or not _compatible_capture_brackets(coarse, fine, _WORKER_SEARCH)
        ):
            raise RuntimeError(
                "zero-threshold re-bisection contains a hidden timeout or "
                "timestep-dependent boundary for "
                f"{(sample.disc_index, sample.point_index)}"
            )
        if coarse != coarse_boundary or fine != fine_boundary:
            raise RuntimeError(
                "zero-threshold re-bisection bracket provenance is inconsistent"
            )

    coarse_search, fine_search = audit_searches(_WORKER_SEARCH)
    zero_grid_evidence = (
        _velocity_grid_evidence_payload(outcome)
        if outcome.status
        in {
            "confirmed_zero_capture",
            "grid_resolved_capture",
            "velocity_resolved_capture",
        }
        else []
    )
    adaptive_payload = adaptive_audit_evidence_to_payload(outcome.adaptive_evidence)
    final_level = None
    if outcome.adaptive_evidence:
        final_level = max(
            outcome.adaptive_evidence,
            key=lambda item: (item.level_index, item.duration_s),
        )
    audit_row.update(
        {
            # These describe the primary dual-step boundary/grid audit. Any
            # node-only escalation is recorded separately below.
            "accepted_max_simulation_time_s": fine_search.max_simulation_time_s,
            "accepted_coarse_time_step_s": coarse_search.time_step_s,
            "accepted_time_step_s": fine_search.time_step_s,
            "adaptive_audit_level_count": max(
                (item.level_index for item in outcome.adaptive_evidence), default=0
            ),
            "adaptive_audit_evidence_json": json.dumps(
                adaptive_payload, separators=(",", ":")
            ),
            "adaptive_max_duration_s": (
                final_level.duration_s if final_level is not None else 0.0
            ),
            "adaptive_coarse_time_step_s": (
                final_level.coarse_time_step_s if final_level is not None else 0.0
            ),
            "adaptive_fine_time_step_s": (
                final_level.fine_time_step_s if final_level is not None else 0.0
            ),
            "accepted_trapped_velocity_lower_m_per_s": (
                accepted.trapped_velocity_lower_m_per_s
            ),
            "accepted_untrapped_velocity_upper_m_per_s": (
                accepted.untrapped_velocity_upper_m_per_s
            ),
            "accepted_lower_classification": accepted.lower_classification,
            "accepted_upper_classification": accepted.upper_classification,
            "coarse_capture_velocity_m_per_s": (
                coarse_boundary.capture_velocity_m_per_s
            ),
            "fine_capture_velocity_m_per_s": fine_boundary.capture_velocity_m_per_s,
            "coarse_trapped_velocity_lower_m_per_s": (
                coarse_boundary.trapped_velocity_lower_m_per_s
            ),
            "coarse_untrapped_velocity_upper_m_per_s": (
                coarse_boundary.untrapped_velocity_upper_m_per_s
            ),
            "fine_trapped_velocity_lower_m_per_s": (
                fine_boundary.trapped_velocity_lower_m_per_s
            ),
            "fine_untrapped_velocity_upper_m_per_s": (
                fine_boundary.untrapped_velocity_upper_m_per_s
            ),
            "coarse_lower_classification": coarse_boundary.lower_classification,
            "coarse_upper_classification": coarse_boundary.upper_classification,
            "fine_lower_classification": fine_boundary.lower_classification,
            "fine_upper_classification": fine_boundary.upper_classification,
            "coarse_evaluation_timeout_count": len(coarse_timeout_speeds),
            "coarse_evaluation_timeout_speeds_m_per_s_json": _speed_json(
                coarse_timeout_speeds
            ),
            "fine_evaluation_timeout_count": len(fine_timeout_speeds),
            "fine_evaluation_timeout_speeds_m_per_s_json": _speed_json(
                fine_timeout_speeds
            ),
            "zero_threshold_audit_status": outcome.status,
            "zero_threshold_audit_reason": outcome.reason,
            "zero_threshold_grid_evidence_json": json.dumps(
                zero_grid_evidence, separators=(",", ":")
            ),
            "velocity_resolved_override": outcome.velocity_override is not None,
        }
    )
    if bool(audit_row["base_timeout_detected"]):
        audit_row.update(
            {
                "timeout_resolution_status": f"zero_threshold_{outcome.status}",
                "timeout_resolution_reason": (
                    "base-search timeout evidence was superseded by the complete "
                    "dual-timestep zero-threshold audit, including adaptive "
                    "node-only escalation where required"
                ),
            }
        )
    return CaptureWorkerResult(
        accepted,
        audit_row,
        outcome.velocity_override,
    )


def _speed_json(speeds: Sequence[float]) -> str:
    return json.dumps([float(speed) for speed in speeds], separators=(",", ":"))


def _classification_json(speeds: Sequence[float], results: Sequence[object]) -> str:
    return json.dumps(
        [
            {
                "speed_m_per_s": float(speed),
                "trapped": bool(getattr(result, "trapped")),
                "termination_reason": str(getattr(result, "termination_reason")),
            }
            for speed, result in zip(speeds, results, strict=True)
        ],
        separators=(",", ":"),
    )


def _classification_payload(result: object) -> dict[str, object]:
    """Return a JSON-safe terminal-classification record."""

    return {
        "trapped": bool(getattr(result, "trapped")),
        "termination_reason": str(getattr(result, "termination_reason")),
        "entered_trap_core": bool(getattr(result, "entered_trap_core")),
        "core_entry_count": int(getattr(result, "core_entry_count")),
        "elapsed_time_s": float(getattr(result, "elapsed_time_s")),
        "minimum_radius_m": float(getattr(result, "minimum_radius_m")),
        "final_radius_m": float(getattr(result, "final_radius_m")),
        "final_position_m": [
            float(value) for value in getattr(result, "final_position_m")
        ],
        "final_velocity_m_per_s": [
            float(value) for value in getattr(result, "final_velocity_m_per_s")
        ],
    }


def _capture_sample_payload(sample: CaptureVelocitySample) -> dict[str, object]:
    """Return the boundary fields needed to independently revalidate a search."""

    return {
        "capture_velocity_m_per_s": float(sample.capture_velocity_m_per_s),
        "velocity_resolution_m_per_s": float(sample.velocity_resolution_m_per_s),
        "trapped_velocity_lower_m_per_s": float(
            sample.trapped_velocity_lower_m_per_s
        ),
        "untrapped_velocity_upper_m_per_s": float(
            sample.untrapped_velocity_upper_m_per_s
        ),
        "lower_classification": sample.lower_classification,
        "upper_classification": sample.upper_classification,
        "lower_entered_trap_core": bool(sample.lower_entered_trap_core),
        "upper_entered_trap_core": bool(sample.upper_entered_trap_core),
        "lower_core_entry_count": int(sample.lower_core_entry_count),
        "upper_core_entry_count": int(sample.upper_core_entry_count),
    }


def _evaluation_ledger_payload(
    evaluations: Sequence[tuple[float, object]],
) -> list[dict[str, object]]:
    return [
        {
            "speed_m_per_s": float(speed),
            "classification": _classification_payload(result),
        }
        for speed, result in evaluations
    ]


def _velocity_grid_evidence_payload(
    outcome: TimeoutAuditOutcome,
) -> list[dict[str, object]]:
    """Serialize every final coarse/fine node behind a direct capture mask."""

    velocities = tuple(outcome.velocity_grid_m_per_s)
    coarse_results = tuple(outcome.coarse_velocity_grid_results)
    fine_results = tuple(outcome.fine_velocity_grid_results)
    if not (
        velocities
        and len(velocities) == len(coarse_results) == len(fine_results)
    ):
        raise RuntimeError("resolved velocity-grid audit lacks complete dual-step evidence")
    return [
        {
            "speed_m_per_s": float(speed),
            "coarse_result": _classification_payload(coarse_result),
            "fine_result": _classification_payload(fine_result),
        }
        for speed, coarse_result, fine_result in zip(
            velocities, coarse_results, fine_results, strict=True
        )
    ]


def _classify_targeted_audit_speeds(
    point: PointSample,
    speeds_m_per_s: Sequence[float],
    search: CaptureSearchConfig,
) -> dict[float, object]:
    """Classify one ray's audit speeds in one vectorized trajectory batch.

    Positive-threshold timeout audits can contain several actual timeout nodes
    plus the saved lower and upper endpoints.  Advancing those speeds together
    is mathematically equivalent to independent deterministic trajectories and
    avoids repeating the Python integration loop for every speed.  Coarse and
    fine timesteps still call this helper separately, so timestep-convergence
    evidence remains independent and explicit.
    """

    if _WORKER_BEAMS is None or _WORKER_COIL is None or _WORKER_SIMPLE is None:
        raise RuntimeError("capture worker was not initialized")
    unique_speeds = tuple(
        dict.fromkeys(round(float(speed), 12) for speed in speeds_m_per_s)
    )
    if not unique_speeds:
        return {}
    classified = classify_trajectory_batch(
        _WORKER_BEAMS,
        point,
        unique_speeds,
        _WORKER_COIL,
        _WORKER_SIMPLE,
        search,
    )
    if len(classified) != len(unique_speeds):
        raise RuntimeError("targeted timeout audit returned an incomplete batch")
    return dict(zip(unique_speeds, classified, strict=True))


def _is_definitive_trapped(result: object) -> bool:
    return bool(
        getattr(result, "trapped", False)
        and getattr(result, "termination_reason", "") in TRAPPED_TERMINATION_REASONS
    )


def _is_definitive_escaped(result: object) -> bool:
    return bool(
        not getattr(result, "trapped", True)
        and getattr(result, "termination_reason", "") == "escaped"
    )


def _adaptively_resolve_positive_timeout_nodes(
    point: PointSample,
    speeds_m_per_s: Sequence[float],
) -> tuple[tuple[AdaptiveAuditEvidence, ...], tuple[float, ...]]:
    """Escalate only censored nodes from an otherwise definitive re-search.

    Each level reruns only nodes that remained non-definitive or disagreed at
    the preceding level. A node is removed from the unresolved set only when
    the coarse and fine integrations both terminate definitively with the same
    trapped/escaped result.
    """

    if _WORKER_SEARCH is None:
        raise RuntimeError("capture worker was not initialized")
    unresolved = tuple(
        dict.fromkeys(round(float(speed), 12) for speed in speeds_m_per_s)
    )
    evidence: list[AdaptiveAuditEvidence] = []
    for level_index, level in enumerate(DEFAULT_ADAPTIVE_AUDIT_LEVELS, start=1):
        if not unresolved:
            break
        coarse_search, fine_search = audit_searches(
            _WORKER_SEARCH,
            duration_s=level.duration_s,
            coarse_time_step_s=level.coarse_time_step_s,
            fine_time_step_s=level.fine_time_step_s,
        )
        coarse_results = _classify_targeted_audit_speeds(
            point, unresolved, coarse_search
        )
        fine_results = _classify_targeted_audit_speeds(point, unresolved, fine_search)
        still_unresolved: list[float] = []
        for speed in unresolved:
            item = AdaptiveAuditEvidence(
                level_index=level_index,
                duration_s=level.duration_s,
                coarse_time_step_s=level.coarse_time_step_s,
                fine_time_step_s=level.fine_time_step_s,
                velocity_m_per_s=speed,
                coarse_result=coarse_results[speed],
                fine_result=fine_results[speed],
            )
            evidence.append(item)
            if not item.resolved:
                still_unresolved.append(speed)
        unresolved = tuple(still_unresolved)
    return tuple(evidence), unresolved


def _adaptive_evidence_preserves_bracket(
    evidence: Sequence[AdaptiveAuditEvidence],
    coarse: CaptureVelocitySample,
    fine: CaptureVelocitySample,
) -> bool:
    """Check final adaptive classifications against both saved boundaries."""

    latest: dict[float, AdaptiveAuditEvidence] = {}
    for item in evidence:
        latest[item.velocity_m_per_s] = item
    epsilon = 1.0e-12
    for speed, item in latest.items():
        if not item.resolved:
            return False
        for sample in (coarse, fine):
            if speed <= sample.trapped_velocity_lower_m_per_s + epsilon:
                if not _is_definitive_trapped(item.fine_result):
                    return False
            elif speed >= sample.untrapped_velocity_upper_m_per_s - epsilon:
                if not _is_definitive_escaped(item.fine_result):
                    return False
    return True


def _compatible_capture_brackets(
    coarse: CaptureVelocitySample,
    fine: CaptureVelocitySample,
    search: CaptureSearchConfig,
) -> bool:
    tolerance = search.velocity_tolerance_m_per_s + 1.0e-12
    overlap_width = min(
        coarse.untrapped_velocity_upper_m_per_s,
        fine.untrapped_velocity_upper_m_per_s,
    ) - max(
        coarse.trapped_velocity_lower_m_per_s,
        fine.trapped_velocity_lower_m_per_s,
    )
    return bool(
        _valid_capture_endpoint(coarse)
        and _valid_capture_endpoint(fine)
        # Merely touching intervals assign escaped and trapped to the same
        # shared speed. Require a numerically meaningful, nonzero overlap.
        and overlap_width > 1.0e-12
        and abs(coarse.capture_velocity_m_per_s - fine.capture_velocity_m_per_s)
        <= tolerance
    )


def _complete_boundary_level_payload(
    *,
    level_index: int,
    coarse_search: CaptureSearchConfig,
    fine_search: CaptureSearchConfig,
    coarse_sample: CaptureVelocitySample,
    fine_sample: CaptureVelocitySample,
    coarse_timeout_speeds: Sequence[float],
    fine_timeout_speeds: Sequence[float],
    coarse_evaluations: Sequence[tuple[float, object]],
    fine_evaluations: Sequence[tuple[float, object]],
    production_search: CaptureSearchConfig,
) -> dict[str, object]:
    """Serialize one complete dual-timestep boundary-search level."""

    return {
        "level_index": int(level_index),
        "duration_s": float(fine_search.max_simulation_time_s),
        "coarse_time_step_s": float(coarse_search.time_step_s),
        "fine_time_step_s": float(fine_search.time_step_s),
        "coarse_sample": _capture_sample_payload(coarse_sample),
        "fine_sample": _capture_sample_payload(fine_sample),
        "coarse_timeout_speeds_m_per_s": [
            float(speed) for speed in coarse_timeout_speeds
        ],
        "fine_timeout_speeds_m_per_s": [
            float(speed) for speed in fine_timeout_speeds
        ],
        "coarse_evaluations": _evaluation_ledger_payload(coarse_evaluations),
        "fine_evaluations": _evaluation_ledger_payload(fine_evaluations),
        "compatible_definitive_brackets": _compatible_capture_brackets(
            coarse_sample, fine_sample, production_search
        ),
        "timeout_free": not coarse_timeout_speeds and not fine_timeout_speeds,
    }


def _recover_positive_boundary_with_complete_search_and_grid(
    point: PointSample,
    base: CaptureVelocitySample,
    audit_row: dict[str, object],
    *,
    initial_coarse_search: CaptureSearchConfig,
    initial_fine_search: CaptureSearchConfig,
    initial_coarse: CaptureVelocitySample,
    initial_fine: CaptureVelocitySample,
    initial_coarse_timeout_speeds: Sequence[float],
    initial_fine_timeout_speeds: Sequence[float],
    initial_coarse_evaluations: Sequence[tuple[float, object]],
    initial_fine_evaluations: Sequence[tuple[float, object]],
) -> CaptureWorkerResult:
    """Recover a censored/disputed positive boundary without forcing a scalar.

    Complete dual-timestep searches are repeated at bounded longer durations.
    The first timeout-free, compatible pair supplies a diagnostic sub-grid
    bracket.  Because the original scalar search already failed its premise, a
    complete dual-timestep scan on the exact loading grid is retained as the
    authoritative capture mask even when that mask is monotone.
    """

    if (
        _WORKER_APPARATUS is None
        or _WORKER_SIMPLE is None
        or _WORKER_COIL is None
        or _WORKER_SEARCH is None
    ):
        raise RuntimeError("capture worker was not initialized")

    boundary_evidence: list[dict[str, object]] = [
        _complete_boundary_level_payload(
            level_index=0,
            coarse_search=initial_coarse_search,
            fine_search=initial_fine_search,
            coarse_sample=initial_coarse,
            fine_sample=initial_fine,
            coarse_timeout_speeds=initial_coarse_timeout_speeds,
            fine_timeout_speeds=initial_fine_timeout_speeds,
            coarse_evaluations=initial_coarse_evaluations,
            fine_evaluations=initial_fine_evaluations,
            production_search=_WORKER_SEARCH,
        )
    ]
    grid_base_level_index: int | None = None
    grid_base_level: AdaptiveAuditLevel | None = None
    grid_base_coarse_search: CaptureSearchConfig | None = None
    grid_base_fine_search: CaptureSearchConfig | None = None
    diagnostic_coarse: CaptureVelocitySample | None = None
    diagnostic_fine: CaptureVelocitySample | None = None
    diagnostic_scalar_boundary_level_index = -1
    diagnostic_scalar_boundary_converged = False

    for level_index, level in enumerate(DEFAULT_ADAPTIVE_AUDIT_LEVELS, start=1):
        coarse_search, fine_search = audit_searches(
            _WORKER_SEARCH,
            duration_s=level.duration_s,
            coarse_time_step_s=level.coarse_time_step_s,
            fine_time_step_s=level.fine_time_step_s,
        )
        coarse, coarse_timeouts, coarse_evaluations = _evaluate_capture_instrumented(
            point, coarse_search
        )
        fine, fine_timeouts, fine_evaluations = _evaluate_capture_instrumented(
            point, fine_search
        )
        boundary_evidence.append(
            _complete_boundary_level_payload(
                level_index=level_index,
                coarse_search=coarse_search,
                fine_search=fine_search,
                coarse_sample=coarse,
                fine_sample=fine,
                coarse_timeout_speeds=coarse_timeouts,
                fine_timeout_speeds=fine_timeouts,
                coarse_evaluations=coarse_evaluations,
                fine_evaluations=fine_evaluations,
                production_search=_WORKER_SEARCH,
            )
        )
        grid_base_level_index = level_index
        grid_base_level = level
        grid_base_coarse_search = coarse_search
        grid_base_fine_search = fine_search
        diagnostic_coarse = coarse
        diagnostic_fine = fine
        if (
            not coarse_timeouts
            and not fine_timeouts
            and _compatible_capture_brackets(coarse, fine, _WORKER_SEARCH)
        ):
            diagnostic_scalar_boundary_level_index = level_index
            diagnostic_scalar_boundary_converged = True
            break

    if (
        grid_base_level_index is None
        or grid_base_level is None
        or grid_base_coarse_search is None
        or grid_base_fine_search is None
        or diagnostic_coarse is None
        or diagnostic_fine is None
    ):
        raise RuntimeError("positive capture boundary fallback has no duration level")

    remaining_grid_levels = DEFAULT_ADAPTIVE_AUDIT_LEVELS[grid_base_level_index:]
    (grid_outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (
            TimeoutAuditCase(
                sample=diagnostic_fine,
                apparatus=_WORKER_APPARATUS,
                simple_config=_WORKER_SIMPLE,
                coil_config=_WORKER_COIL,
                search_config=_WORKER_SEARCH,
            ),
        ),
        always_override=True,
        duration_s=grid_base_level.duration_s,
        coarse_time_step_s=grid_base_level.coarse_time_step_s,
        fine_time_step_s=grid_base_level.fine_time_step_s,
        adaptive_levels=remaining_grid_levels,
    )
    if (
        not grid_outcome.resolved
        or grid_outcome.velocity_override is None
        or grid_outcome.coarse_boundary_sample is None
        or grid_outcome.fine_boundary_sample is None
    ):
        raise RuntimeError(
            "positive capture boundary exact loading-grid audit failed closed for "
            f"{(point.disc_index, point.point_index)}: {grid_outcome.reason}"
        )

    accepted_coarse = grid_outcome.coarse_boundary_sample
    accepted_fine = grid_outcome.fine_boundary_sample
    diagnostic_scalar_grid_agreement = "not_converged"
    if diagnostic_scalar_boundary_converged:
        diagnostic_scalar_grid_agreement = (
            "agree"
            if _compatible_capture_brackets(
                diagnostic_fine, accepted_fine, _WORKER_SEARCH
            )
            else "disagree"
        )

    adaptive_payload = adaptive_audit_evidence_to_payload(
        grid_outcome.adaptive_evidence
    )
    final_grid_level = (
        max(grid_outcome.adaptive_evidence, key=lambda item: item.level_index)
        if grid_outcome.adaptive_evidence
        else None
    )
    audit_row.update(
        {
            "accepted_max_simulation_time_s": grid_base_fine_search.max_simulation_time_s,
            "accepted_coarse_time_step_s": grid_base_coarse_search.time_step_s,
            "accepted_time_step_s": grid_base_fine_search.time_step_s,
            "adaptive_audit_level_count": (
                final_grid_level.level_index if final_grid_level is not None else 0
            ),
            "adaptive_audit_evidence_json": json.dumps(
                adaptive_payload, separators=(",", ":")
            ),
            "adaptive_max_duration_s": (
                final_grid_level.duration_s if final_grid_level is not None else 0.0
            ),
            "adaptive_coarse_time_step_s": (
                final_grid_level.coarse_time_step_s
                if final_grid_level is not None
                else 0.0
            ),
            "adaptive_fine_time_step_s": (
                final_grid_level.fine_time_step_s
                if final_grid_level is not None
                else 0.0
            ),
            "accepted_trapped_velocity_lower_m_per_s": (
                accepted_fine.trapped_velocity_lower_m_per_s
            ),
            "accepted_untrapped_velocity_upper_m_per_s": (
                accepted_fine.untrapped_velocity_upper_m_per_s
            ),
            "accepted_lower_classification": accepted_fine.lower_classification,
            "accepted_upper_classification": accepted_fine.upper_classification,
            "coarse_capture_velocity_m_per_s": accepted_coarse.capture_velocity_m_per_s,
            "fine_capture_velocity_m_per_s": accepted_fine.capture_velocity_m_per_s,
            "coarse_trapped_velocity_lower_m_per_s": (
                accepted_coarse.trapped_velocity_lower_m_per_s
            ),
            "coarse_untrapped_velocity_upper_m_per_s": (
                accepted_coarse.untrapped_velocity_upper_m_per_s
            ),
            "fine_trapped_velocity_lower_m_per_s": (
                accepted_fine.trapped_velocity_lower_m_per_s
            ),
            "fine_untrapped_velocity_upper_m_per_s": (
                accepted_fine.untrapped_velocity_upper_m_per_s
            ),
            "coarse_lower_classification": accepted_coarse.lower_classification,
            "coarse_upper_classification": accepted_coarse.upper_classification,
            "fine_lower_classification": accepted_fine.lower_classification,
            "fine_upper_classification": accepted_fine.upper_classification,
            "coarse_evaluation_timeout_count": 0,
            "coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
            "fine_evaluation_timeout_count": 0,
            "fine_evaluation_timeout_speeds_m_per_s_json": "[]",
            "complete_boundary_audit_evidence_json": json.dumps(
                boundary_evidence, separators=(",", ":")
            ),
            "positive_boundary_grid_base_level_index": grid_base_level_index,
            "diagnostic_scalar_boundary_level_index": (
                diagnostic_scalar_boundary_level_index
            ),
            "diagnostic_scalar_boundary_converged": (
                diagnostic_scalar_boundary_converged
            ),
            "diagnostic_scalar_grid_agreement": diagnostic_scalar_grid_agreement,
            "timeout_resolution_status": (
                "complete_scalar_diagnostics_and_velocity_grid_recovered_capture"
            ),
            "timeout_resolution_reason": (
                "the 200 ms scalar search was censored or incompatible; bounded "
                "longer-duration scalar searches were retained as diagnostics and "
                "an independent exact loading-grid scan supplied the authoritative "
                "per-velocity capture mask"
            ),
            "zero_threshold_audit_status": "not_applicable",
            "zero_threshold_audit_reason": "positive scalar capture threshold",
            "positive_boundary_grid_audit_status": grid_outcome.status,
            "positive_boundary_grid_audit_reason": grid_outcome.reason,
            "positive_boundary_grid_evidence_json": json.dumps(
                _velocity_grid_evidence_payload(grid_outcome), separators=(",", ":")
            ),
            "positive_boundary_grid_duration_s": grid_base_level.duration_s,
            "positive_boundary_grid_coarse_time_step_s": (
                grid_base_level.coarse_time_step_s
            ),
            "positive_boundary_grid_fine_time_step_s": (
                grid_base_level.fine_time_step_s
            ),
            "velocity_resolved_override": True,
        }
    )
    return CaptureWorkerResult(
        accepted_fine,
        audit_row,
        grid_outcome.velocity_override,
    )


def _base_audit_row(
    point: PointSample,
    sample: CaptureVelocitySample,
    timeout_speeds: Sequence[float],
    search: CaptureSearchConfig,
) -> dict[str, object]:
    return {
        "disc_index": point.disc_index,
        "point_index": point.point_index,
        "base_timeout_detected": bool(timeout_speeds),
        "base_evaluation_timeout_count": len(timeout_speeds),
        "base_evaluation_timeout_speeds_m_per_s_json": _speed_json(timeout_speeds),
        "accepted_max_simulation_time_s": search.max_simulation_time_s,
        "accepted_coarse_time_step_s": search.time_step_s,
        "accepted_time_step_s": search.time_step_s,
        "adaptive_audit_level_count": 0,
        "adaptive_audit_evidence_json": "[]",
        "adaptive_max_duration_s": 0.0,
        "adaptive_coarse_time_step_s": 0.0,
        "adaptive_fine_time_step_s": 0.0,
        "accepted_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "accepted_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "accepted_lower_classification": sample.lower_classification,
        "accepted_upper_classification": sample.upper_classification,
        "base_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "coarse_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "fine_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "coarse_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "coarse_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "fine_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "fine_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "coarse_lower_classification": sample.lower_classification,
        "coarse_upper_classification": sample.upper_classification,
        "fine_lower_classification": sample.lower_classification,
        "fine_upper_classification": sample.upper_classification,
        "timeout_node_coarse_results_json": "[]",
        "timeout_node_fine_results_json": "[]",
        "coarse_evaluation_timeout_count": 0,
        "coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
        "fine_evaluation_timeout_count": 0,
        "fine_evaluation_timeout_speeds_m_per_s_json": "[]",
        "pre_adaptive_coarse_evaluation_timeout_count": 0,
        "pre_adaptive_coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
        "pre_adaptive_fine_evaluation_timeout_count": 0,
        "pre_adaptive_fine_evaluation_timeout_speeds_m_per_s_json": "[]",
        "complete_boundary_audit_evidence_json": "[]",
        "positive_boundary_grid_base_level_index": 0,
        "diagnostic_scalar_boundary_level_index": -1,
        "diagnostic_scalar_boundary_converged": False,
        "diagnostic_scalar_grid_agreement": "not_applicable",
        "timeout_resolution_status": (
            "pending" if timeout_speeds else "not_required"
        ),
        "timeout_resolution_reason": (
            "base search evaluated one or more timeout-contaminated speeds"
            if timeout_speeds
            else "base search contained no timeout"
        ),
        "zero_threshold_audit_status": "pending",
        "zero_threshold_audit_reason": "pending",
        "zero_threshold_grid_evidence_json": "[]",
        "positive_boundary_grid_audit_status": "not_applicable",
        "positive_boundary_grid_audit_reason": (
            "positive-boundary grid fallback was not required"
        ),
        "positive_boundary_grid_evidence_json": "[]",
        "positive_boundary_grid_duration_s": 0.0,
        "positive_boundary_grid_coarse_time_step_s": 0.0,
        "positive_boundary_grid_fine_time_step_s": 0.0,
        "velocity_resolved_override": False,
    }


def _complete_capture_worker_from_base(
    point: PointSample,
    base: CaptureVelocitySample,
    base_timeout_speeds: Sequence[float],
    audit_row: dict[str, object],
) -> CaptureWorkerResult:
    """Complete all required audits for one already-evaluated base search."""

    if _WORKER_SEARCH is None:
        raise RuntimeError("capture worker was not initialized")

    # Every scalar zero is checked over the complete 0--30 m/s analysis grid,
    # regardless of whether its short production search observed a timeout.
    if base.capture_velocity_m_per_s == 0.0:
        return _apply_zero_threshold_audit(base, audit_row)

    if not base_timeout_speeds:
        if not _valid_capture_endpoint(base):
            raise RuntimeError(
                f"invalid capture bracket for {(point.disc_index, point.point_index)}"
            )
        return _apply_zero_threshold_audit(base, audit_row)

    if _WORKER_APPARATUS is None or _WORKER_COIL is None or _WORKER_SIMPLE is None:
        raise RuntimeError("capture worker was not initialized")
    coarse_search, fine_search = audit_searches(_WORKER_SEARCH)

    audit_speeds = tuple(
        dict.fromkeys(
            round(float(speed), 12)
            for speed in (
                *base_timeout_speeds,
                base.trapped_velocity_lower_m_per_s,
                base.untrapped_velocity_upper_m_per_s,
            )
        )
    )

    coarse_results = _classify_targeted_audit_speeds(
        point, audit_speeds, coarse_search
    )
    fine_results = _classify_targeted_audit_speeds(point, audit_speeds, fine_search)
    coarse_nodes = tuple(coarse_results[round(speed, 12)] for speed in base_timeout_speeds)
    fine_nodes = tuple(fine_results[round(speed, 12)] for speed in base_timeout_speeds)
    lower_speed = round(base.trapped_velocity_lower_m_per_s, 12)
    upper_speed = round(base.untrapped_velocity_upper_m_per_s, 12)
    coarse_lower = coarse_results[lower_speed]
    fine_lower = fine_results[lower_speed]
    coarse_upper = coarse_results[upper_speed]
    fine_upper = fine_results[upper_speed]
    audit_row.update(
        {
            "accepted_max_simulation_time_s": fine_search.max_simulation_time_s,
            "accepted_coarse_time_step_s": coarse_search.time_step_s,
            "accepted_time_step_s": fine_search.time_step_s,
            "coarse_capture_velocity_m_per_s": (
                base.trapped_velocity_lower_m_per_s
            ),
            "fine_capture_velocity_m_per_s": (
                base.trapped_velocity_lower_m_per_s
            ),
            "coarse_trapped_velocity_lower_m_per_s": (
                base.trapped_velocity_lower_m_per_s
            ),
            "coarse_untrapped_velocity_upper_m_per_s": (
                base.untrapped_velocity_upper_m_per_s
            ),
            "fine_trapped_velocity_lower_m_per_s": (
                base.trapped_velocity_lower_m_per_s
            ),
            "fine_untrapped_velocity_upper_m_per_s": (
                base.untrapped_velocity_upper_m_per_s
            ),
            "coarse_lower_classification": coarse_lower.termination_reason,
            "coarse_upper_classification": coarse_upper.termination_reason,
            "fine_lower_classification": fine_lower.termination_reason,
            "fine_upper_classification": fine_upper.termination_reason,
            "timeout_node_coarse_results_json": _classification_json(
                base_timeout_speeds, coarse_nodes
            ),
            "timeout_node_fine_results_json": _classification_json(
                base_timeout_speeds, fine_nodes
            ),
        }
    )
    saved_boundary_is_definitive = bool(
        _is_definitive_trapped(coarse_lower)
        and _is_definitive_trapped(fine_lower)
        and _is_definitive_escaped(coarse_upper)
        and _is_definitive_escaped(fine_upper)
    )
    timeout_nodes_are_escaped = all(
        _is_definitive_escaped(coarse_result)
        and _is_definitive_escaped(fine_result)
        for coarse_result, fine_result in zip(coarse_nodes, fine_nodes, strict=True)
    )
    if saved_boundary_is_definitive and timeout_nodes_are_escaped:
        accepted = replace(
            base,
            lower_classification=fine_lower.termination_reason,
            upper_classification=fine_upper.termination_reason,
            lower_entered_trap_core=fine_lower.entered_trap_core,
            upper_entered_trap_core=fine_upper.entered_trap_core,
            lower_core_entry_count=fine_lower.core_entry_count,
            upper_core_entry_count=fine_upper.core_entry_count,
        )
        audit_row.update(
            {
                "timeout_resolution_status": "targeted_timeout_nodes_confirmed_escaped",
                "timeout_resolution_reason": (
                    "every actual base-search timeout speed is escaped at 200 ms "
                    "with 5 and 2.5 microsecond steps, and the saved endpoints remain "
                    "a definitive trapped/escaped bracket"
                ),
                "zero_threshold_audit_status": "not_applicable",
                "zero_threshold_audit_reason": "positive scalar capture threshold",
                "accepted_lower_classification": accepted.lower_classification,
                "accepted_upper_classification": accepted.upper_classification,
            }
        )
        return CaptureWorkerResult(accepted, audit_row, None)

    # Only changed, timestep-dependent, or still non-definitive evidence pays
    # for complete 200 ms coarse/fine re-searches. Both searches expose every
    # evaluated timeout speed, including internal bisection nodes.
    coarse, coarse_timeout_speeds, coarse_evaluations = (
        _evaluate_capture_instrumented(point, coarse_search)
    )
    fine, fine_timeout_speeds, fine_evaluations = _evaluate_capture_instrumented(
        point, fine_search
    )
    audit_row.update(
        {
            "pre_adaptive_coarse_evaluation_timeout_count": len(
                coarse_timeout_speeds
            ),
            "pre_adaptive_coarse_evaluation_timeout_speeds_m_per_s_json": (
                _speed_json(coarse_timeout_speeds)
            ),
            "pre_adaptive_fine_evaluation_timeout_count": len(fine_timeout_speeds),
            "pre_adaptive_fine_evaluation_timeout_speeds_m_per_s_json": (
                _speed_json(fine_timeout_speeds)
            ),
        }
    )
    if (
        coarse_timeout_speeds
        or fine_timeout_speeds
        or not _compatible_capture_brackets(coarse, fine, _WORKER_SEARCH)
    ):
        return _recover_positive_boundary_with_complete_search_and_grid(
            point,
            base,
            audit_row,
            initial_coarse_search=coarse_search,
            initial_fine_search=fine_search,
            initial_coarse=coarse,
            initial_fine=fine,
            initial_coarse_timeout_speeds=coarse_timeout_speeds,
            initial_fine_timeout_speeds=fine_timeout_speeds,
            initial_coarse_evaluations=coarse_evaluations,
            initial_fine_evaluations=fine_evaluations,
        )

    adaptive_evidence: tuple[AdaptiveAuditEvidence, ...] = ()
    complete_boundary_evidence = [
        _complete_boundary_level_payload(
            level_index=0,
            coarse_search=coarse_search,
            fine_search=fine_search,
            coarse_sample=coarse,
            fine_sample=fine,
            coarse_timeout_speeds=coarse_timeout_speeds,
            fine_timeout_speeds=fine_timeout_speeds,
            coarse_evaluations=coarse_evaluations,
            fine_evaluations=fine_evaluations,
            production_search=_WORKER_SEARCH,
        )
    ]

    accepted = fine
    adaptive_payload = adaptive_audit_evidence_to_payload(adaptive_evidence)
    final_level = (
        max(adaptive_evidence, key=lambda item: item.level_index)
        if adaptive_evidence
        else None
    )
    audit_row.update(
        {
            "coarse_capture_velocity_m_per_s": coarse.capture_velocity_m_per_s,
            "fine_capture_velocity_m_per_s": fine.capture_velocity_m_per_s,
            "coarse_trapped_velocity_lower_m_per_s": (
                coarse.trapped_velocity_lower_m_per_s
            ),
            "coarse_untrapped_velocity_upper_m_per_s": (
                coarse.untrapped_velocity_upper_m_per_s
            ),
            "fine_trapped_velocity_lower_m_per_s": (
                fine.trapped_velocity_lower_m_per_s
            ),
            "fine_untrapped_velocity_upper_m_per_s": (
                fine.untrapped_velocity_upper_m_per_s
            ),
            "coarse_lower_classification": coarse.lower_classification,
            "coarse_upper_classification": coarse.upper_classification,
            "fine_lower_classification": fine.lower_classification,
            "fine_upper_classification": fine.upper_classification,
            "coarse_evaluation_timeout_count": 0,
            "coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
            "fine_evaluation_timeout_count": 0,
            "fine_evaluation_timeout_speeds_m_per_s_json": "[]",
            "accepted_max_simulation_time_s": (
                fine_search.max_simulation_time_s
            ),
            "accepted_coarse_time_step_s": (
                coarse_search.time_step_s
            ),
            "accepted_time_step_s": (
                fine_search.time_step_s
            ),
            "adaptive_audit_level_count": (
                final_level.level_index if final_level is not None else 0
            ),
            "adaptive_audit_evidence_json": json.dumps(
                adaptive_payload, separators=(",", ":")
            ),
            "adaptive_max_duration_s": (
                final_level.duration_s if final_level is not None else 0.0
            ),
            "adaptive_coarse_time_step_s": (
                final_level.coarse_time_step_s if final_level is not None else 0.0
            ),
            "adaptive_fine_time_step_s": (
                final_level.fine_time_step_s if final_level is not None else 0.0
            ),
            "complete_boundary_audit_evidence_json": json.dumps(
                complete_boundary_evidence, separators=(",", ":")
            ),
            "positive_boundary_grid_base_level_index": 0,
            "timeout_resolution_status": (
                "adaptive_dual_step_research_recovered_boundary"
                if adaptive_evidence
                else "dual_step_research_recovered_boundary"
            ),
            "timeout_resolution_reason": (
                "targeted timeout-node or endpoint evidence changed or remained "
                "non-definitive; complete 200 ms coarse/fine searches recovered "
                "compatible, timeout-free definitive brackets"
            ),
        }
    )
    return _apply_zero_threshold_audit(accepted, audit_row)


def _capture_worker(point: PointSample) -> CaptureWorkerResult:
    """Evaluate one threshold and fail closed on every observed timeout."""

    if _WORKER_SEARCH is None:
        raise RuntimeError("capture worker was not initialized")
    base, base_timeout_speeds = _evaluate_capture(point, _WORKER_SEARCH)
    audit_row = _base_audit_row(point, base, base_timeout_speeds, _WORKER_SEARCH)
    return _complete_capture_worker_from_base(
        point,
        base,
        base_timeout_speeds,
        audit_row,
    )


def _capture_worker_batch(
    points: Sequence[PointSample],
) -> tuple[CaptureWorkerResult, ...]:
    """Evaluate up to five rays while coalescing their zero-grid audits.

    Base searches and ordinary positive-threshold audits retain their
    scalar/targeted execution paths. Rays whose scalar threshold is zero share
    the large ``ray x velocity`` NumPy integration; an exceptional censored or
    disputed positive boundary receives the same direct grid independently.
    Each returned envelope remains a complete, independently validated Monte
    Carlo sample.
    """

    batch_points = tuple(points)
    if not batch_points:
        return ()
    if len(batch_points) > ZERO_GRID_RAY_BATCH_SIZE:
        raise ValueError(
            "capture worker batches may contain at most "
            f"{ZERO_GRID_RAY_BATCH_SIZE} rays"
        )
    if (
        _WORKER_APPARATUS is None
        or _WORKER_SIMPLE is None
        or _WORKER_COIL is None
        or _WORKER_SEARCH is None
    ):
        raise RuntimeError("capture worker was not initialized")

    base_rows: list[
        tuple[
            PointSample,
            CaptureVelocitySample,
            tuple[float, ...],
            dict[str, object],
        ]
    ] = []
    zero_cases: list[TimeoutAuditCase] = []
    zero_keys: list[tuple[int, int]] = []
    for point in batch_points:
        base, timeout_speeds = _evaluate_capture(point, _WORKER_SEARCH)
        audit_row = _base_audit_row(
            point, base, timeout_speeds, _WORKER_SEARCH
        )
        base_rows.append((point, base, timeout_speeds, audit_row))
        if base.capture_velocity_m_per_s == 0.0:
            zero_cases.append(
                TimeoutAuditCase(
                    sample=base,
                    apparatus=_WORKER_APPARATUS,
                    simple_config=_WORKER_SIMPLE,
                    coil_config=_WORKER_COIL,
                    search_config=_WORKER_SEARCH,
                )
            )
            zero_keys.append((point.disc_index, point.point_index))

    zero_outcomes = audit_zero_capture_boundaries_batched(zero_cases)
    if len(zero_outcomes) != len(zero_cases):
        raise RuntimeError("multi-ray zero-grid audit returned incomplete outcomes")
    outcomes_by_key = {
        key: outcome for key, outcome in zip(zero_keys, zero_outcomes, strict=True)
    }

    completed: list[CaptureWorkerResult] = []
    for point, base, timeout_speeds, audit_row in base_rows:
        key = (point.disc_index, point.point_index)
        if key in outcomes_by_key:
            completed.append(
                _apply_zero_threshold_audit(
                    base,
                    audit_row,
                    precomputed_outcome=outcomes_by_key[key],
                )
            )
        else:
            completed.append(
                _complete_capture_worker_from_base(
                    point,
                    base,
                    timeout_speeds,
                    audit_row,
                )
            )
    return tuple(completed)


def _ray_batches(
    points: Sequence[PointSample],
    batch_size: int = ZERO_GRID_RAY_BATCH_SIZE,
) -> tuple[tuple[PointSample, ...], ...]:
    """Return stable, order-preserving worker batches for resumable sampling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = tuple(points)
    return tuple(
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    )


def _audit_path(paths: StudyPaths) -> Path:
    return paths.statistics / "capture_endpoint_audit.csv"


def _overrides_path(paths: StudyPaths) -> Path:
    return paths.statistics / "capture_velocity_overrides.json"


def _load_audit_rows(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    if not path.is_file():
        return {}
    rows: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            key = (int(raw["disc_index"]), int(raw["point_index"]))
            if key in rows:
                raise ValueError(f"duplicate endpoint-audit row {key}")
            rows[key] = dict(raw)
    return rows


def _audit_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"endpoint-audit field {field} is not boolean")


def _audit_speed_set(
    values: object,
    *,
    key: tuple[int, int],
    field: str,
) -> set[float]:
    """Return a canonical set of finite, nonnegative serialized speeds."""

    if not isinstance(values, list):
        raise ValueError(f"endpoint-audit field {field} is not a list for {key}")
    canonical: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError(
                f"endpoint-audit field {field} contains a boolean speed for {key}"
            )
        try:
            speed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"endpoint-audit field {field} contains a nonnumeric speed for {key}"
            ) from exc
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError(
                f"endpoint-audit field {field} contains an invalid speed for {key}"
            )
        canonical.append(round(speed, 12))
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            f"endpoint-audit field {field} contains duplicate speeds for {key}"
        )
    return set(canonical)


def _audit_classification_state(
    payload: object,
    *,
    key: tuple[int, int],
    field: str,
) -> str:
    """Recompute trapped/escaped/unresolved from serialized trajectory evidence."""

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"endpoint-audit classification {field} is not a mapping for {key}"
        )
    trapped = payload.get("trapped")
    if type(trapped) is not bool:
        raise ValueError(
            f"endpoint-audit classification {field}.trapped is not boolean for {key}"
        )
    reason = payload.get("termination_reason")
    if not isinstance(reason, str) or reason not in {
        *TRAPPED_TERMINATION_REASONS,
        "escaped",
        "timeout",
        "non_finite",
    }:
        raise ValueError(
            f"endpoint-audit classification {field} has an invalid reason for {key}"
        )
    if trapped:
        if reason not in TRAPPED_TERMINATION_REASONS:
            raise ValueError(
                f"endpoint-audit classification {field} has inconsistent trapped "
                f"state for {key}"
            )
        return "trapped"
    if reason in TRAPPED_TERMINATION_REASONS:
        raise ValueError(
            f"endpoint-audit classification {field} has inconsistent trapped "
            f"reason for {key}"
        )
    return "escaped" if reason == "escaped" else "unresolved"


def _audit_classification_records(
    raw_payload: object,
    *,
    key: tuple[int, int],
    field: str,
) -> dict[float, str]:
    """Parse one targeted-node JSON field without trusting its classifications."""

    try:
        payload = json.loads(str(raw_payload) or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"endpoint-audit field {field} is not valid JSON for {key}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError(f"endpoint-audit field {field} is not a list for {key}")
    parsed: dict[float, str] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"endpoint-audit field {field}[{index}] is not a mapping for {key}"
            )
        speeds = _audit_speed_set(
            [record.get("speed_m_per_s")],
            key=key,
            field=f"{field}[{index}].speed_m_per_s",
        )
        speed = next(iter(speeds))
        if speed in parsed:
            raise ValueError(
                f"endpoint-audit field {field} contains duplicate speeds for {key}"
            )
        parsed[speed] = _audit_classification_state(
            record,
            key=key,
            field=f"{field}[{index}]",
        )
    return parsed


def _validate_completed_audit_ledger(
    samples: Mapping[tuple[int, int], CaptureVelocitySample],
    audit_rows: Mapping[tuple[int, int], Mapping[str, object]],
    overrides: Mapping[tuple[int, int], VelocityResolvedCaptureOverride],
    *,
    search: CaptureSearchConfig | None = None,
) -> None:
    """Reject incomplete or internally inconsistent per-ray QA evidence."""

    if set(audit_rows) != set(samples):
        raise ValueError("endpoint-audit keys do not match capture-sample keys")
    if not set(overrides).issubset(samples):
        raise ValueError("velocity-override keys do not match capture-sample keys")
    bracket_search = search if search is not None else default_search_config()
    allowed_zero_statuses = {
        "not_applicable",
        "confirmed_zero_capture",
        "grid_resolved_capture",
        "rebisected",
        "velocity_resolved_capture",
    }
    positive_timeout_statuses = {
        "targeted_timeout_nodes_confirmed_escaped",
        "dual_step_research_recovered_boundary",
        "adaptive_dual_step_research_recovered_boundary",
        "complete_scalar_diagnostics_and_velocity_grid_recovered_capture",
    }
    for key, row in audit_rows.items():
        sample = samples[key]
        if not _valid_capture_endpoint(sample):
            raise ValueError(f"endpoint-audit accepted sample is invalid for {key}")

        timeout_count = int(row.get("base_evaluation_timeout_count", 0) or 0)
        try:
            raw_speeds = json.loads(
                str(row.get("base_evaluation_timeout_speeds_m_per_s_json", "[]"))
                or "[]"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"endpoint-audit base timeout speeds are not valid JSON for {key}"
            ) from exc
        base_timeout_speeds = _audit_speed_set(
            raw_speeds,
            key=key,
            field="base_evaluation_timeout_speeds_m_per_s_json",
        )
        if len(base_timeout_speeds) != timeout_count:
            raise ValueError(f"endpoint-audit timeout-speed count mismatch for {key}")
        detected = _audit_bool(
            row.get("base_timeout_detected"), field="base_timeout_detected"
        )
        if detected != bool(timeout_count):
            raise ValueError(f"endpoint-audit timeout flag mismatch for {key}")

        timeout_status = str(row.get("timeout_resolution_status", "")).strip()
        zero_status = str(row.get("zero_threshold_audit_status", "")).strip()
        positive_grid_status = str(
            row.get("positive_boundary_grid_audit_status", "not_applicable")
            or "not_applicable"
        ).strip()
        if positive_grid_status not in {
            "not_applicable",
            "velocity_resolved_capture",
        }:
            raise ValueError(
                f"endpoint-audit positive-boundary grid status is invalid for {key}"
            )
        positive_grid_fallback = timeout_status == (
            "complete_scalar_diagnostics_and_velocity_grid_recovered_capture"
        )
        base_capture_velocity = float(row.get("base_capture_velocity_m_per_s", np.nan))
        if not np.isfinite(base_capture_velocity) or base_capture_velocity < 0.0:
            raise ValueError(
                f"endpoint-audit base capture velocity is invalid for {key}"
            )

        def persisted_bracket(prefix: str) -> CaptureVelocitySample:
            try:
                capture_velocity = float(
                    row[f"{prefix}_capture_velocity_m_per_s"]
                )
                lower = float(
                    row[f"{prefix}_trapped_velocity_lower_m_per_s"]
                )
                upper = float(
                    row[f"{prefix}_untrapped_velocity_upper_m_per_s"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"endpoint-audit {prefix} bracket bounds are malformed for {key}"
                ) from exc
            if (
                not all(np.isfinite(value) for value in (capture_velocity, lower, upper))
                or lower < 0.0
                or upper <= lower
                or not np.isclose(
                    capture_velocity, lower, rtol=0.0, atol=1.0e-12
                )
            ):
                raise ValueError(
                    f"endpoint-audit {prefix} capture velocity does not equal a "
                    f"valid trapped lower bound for {key}"
                )
            bracket = replace(
                sample,
                capture_velocity_m_per_s=capture_velocity,
                velocity_resolution_m_per_s=upper - lower,
                trapped_velocity_lower_m_per_s=lower,
                untrapped_velocity_upper_m_per_s=upper,
                lower_classification=str(row[f"{prefix}_lower_classification"]),
                upper_classification=str(row[f"{prefix}_upper_classification"]),
            )
            if not _valid_capture_endpoint(bracket):
                raise ValueError(
                    f"endpoint-audit {prefix} persisted bracket is not definitive "
                    f"for {key}"
                )
            return bracket

        coarse_bracket = persisted_bracket("coarse")
        fine_bracket = persisted_bracket("fine")
        has_positive_persisted_bracket = bool(
            coarse_bracket.capture_velocity_m_per_s > 0.0
            or fine_bracket.capture_velocity_m_per_s > 0.0
        )
        if has_positive_persisted_bracket:
            if not (
                coarse_bracket.capture_velocity_m_per_s > 0.0
                and fine_bracket.capture_velocity_m_per_s > 0.0
                and _compatible_capture_brackets(
                    coarse_bracket, fine_bracket, bracket_search
                )
            ):
                raise ValueError(
                    f"endpoint-audit persisted coarse/fine capture brackets are "
                    f"incompatible for {key}"
                )
        if (
            not np.isclose(
                fine_bracket.capture_velocity_m_per_s,
                sample.capture_velocity_m_per_s,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                fine_bracket.trapped_velocity_lower_m_per_s,
                sample.trapped_velocity_lower_m_per_s,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                fine_bracket.untrapped_velocity_upper_m_per_s,
                sample.untrapped_velocity_upper_m_per_s,
                rtol=0.0,
                atol=1.0e-12,
            )
            or not np.isclose(
                fine_bracket.velocity_resolution_m_per_s,
                sample.velocity_resolution_m_per_s,
                rtol=0.0,
                atol=1.0e-12,
            )
            or fine_bracket.lower_classification != sample.lower_classification
            or fine_bracket.upper_classification != sample.upper_classification
        ):
            raise ValueError(
                f"endpoint-audit fine persisted bracket does not equal the "
                f"accepted sample for {key}"
            )
        if zero_status not in allowed_zero_statuses:
            raise ValueError(
                f"endpoint-audit zero-threshold status is invalid for {key}"
            )
        allowed_timeout_statuses = {
            "not_required",
            *positive_timeout_statuses,
            *(f"zero_threshold_{status}" for status in allowed_zero_statuses if status != "not_applicable"),
        }
        if timeout_status not in allowed_timeout_statuses:
            raise ValueError(f"endpoint-audit timeout status is invalid for {key}")
        if detected:
            if timeout_status == "not_required":
                raise ValueError(
                    f"endpoint-audit detected timeout is marked not required for {key}"
                )
        elif timeout_status != "not_required":
            raise ValueError(
                f"endpoint-audit timeout-free base has a resolution status for {key}"
            )
        if zero_status == "not_applicable":
            if np.isclose(base_capture_velocity, 0.0, rtol=0.0, atol=1.0e-15):
                raise ValueError(
                    f"endpoint-audit zero base threshold was not grid-audited for {key}"
                )
            if timeout_status.startswith("zero_threshold_"):
                raise ValueError(
                    f"endpoint-audit zero/timeout status combination is invalid for {key}"
                )
        elif not np.isclose(
            base_capture_velocity, 0.0, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                f"endpoint-audit zero-grid result has a nonzero base threshold for {key}"
            )
        elif detected and timeout_status != f"zero_threshold_{zero_status}":
            raise ValueError(
                f"endpoint-audit zero/timeout status combination is invalid for {key}"
            )
        elif not detected and timeout_status != "not_required":
            raise ValueError(
                f"endpoint-audit zero/timeout status combination is invalid for {key}"
            )

        targeted_coarse = _audit_classification_records(
            row.get("timeout_node_coarse_results_json", "[]"),
            key=key,
            field="timeout_node_coarse_results_json",
        )
        targeted_fine = _audit_classification_records(
            row.get("timeout_node_fine_results_json", "[]"),
            key=key,
            field="timeout_node_fine_results_json",
        )
        if targeted_coarse or targeted_fine or timeout_status in positive_timeout_statuses:
            if set(targeted_coarse) != base_timeout_speeds or set(
                targeted_fine
            ) != base_timeout_speeds:
                raise ValueError(
                    f"endpoint-audit targeted-node evidence does not exactly cover "
                    f"the base timeout speeds for {key}"
                )
        if timeout_status == "targeted_timeout_nodes_confirmed_escaped" and (
            any(state != "escaped" for state in targeted_coarse.values())
            or any(state != "escaped" for state in targeted_fine.values())
        ):
            raise ValueError(
                f"endpoint-audit targeted success retains a non-escaped node for {key}"
            )

        pre_adaptive_timeout_speeds: set[float] = set()
        for prefix in ("pre_adaptive_coarse", "pre_adaptive_fine"):
            count = int(row.get(f"{prefix}_evaluation_timeout_count", 0) or 0)
            try:
                values = json.loads(
                    str(
                        row.get(
                            f"{prefix}_evaluation_timeout_speeds_m_per_s_json",
                            "[]",
                        )
                    )
                    or "[]"
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"endpoint-audit {prefix} timeout speeds are not valid JSON "
                    f"for {key}"
                ) from exc
            parsed = _audit_speed_set(
                values,
                key=key,
                field=f"{prefix}_evaluation_timeout_speeds_m_per_s_json",
            )
            if len(parsed) != count:
                raise ValueError(
                    f"endpoint-audit {prefix} timeout-speed count mismatch for {key}"
                )
            pre_adaptive_timeout_speeds.update(parsed)

        try:
            complete_boundary_evidence = json.loads(
                str(row.get("complete_boundary_audit_evidence_json", "[]")) or "[]"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"endpoint-audit complete-boundary evidence is not valid JSON for {key}"
            ) from exc
        if not isinstance(complete_boundary_evidence, list):
            raise ValueError(
                f"endpoint-audit complete-boundary evidence is not a list for {key}"
            )
        try:
            positive_grid_base_level = int(
                row.get("positive_boundary_grid_base_level_index", 0) or 0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"endpoint-audit positive-grid base level is malformed for {key}"
            ) from exc
        parsed_boundary_levels: list[
            tuple[int, CaptureVelocitySample, CaptureVelocitySample, bool, bool]
        ] = []
        for evidence_index, item in enumerate(complete_boundary_evidence):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"endpoint-audit complete-boundary item is malformed for {key}"
                )
            try:
                level_index = int(item["level_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"endpoint-audit complete-boundary level is malformed for {key}"
                ) from exc
            if level_index != evidence_index or level_index > len(
                DEFAULT_ADAPTIVE_AUDIT_LEVELS
            ):
                raise ValueError(
                    f"endpoint-audit complete-boundary levels are not contiguous for {key}"
                )
            expected_level = (
                AdaptiveAuditLevel(AUDIT_DURATION_S, COARSE_TIME_STEP_S, FINE_TIME_STEP_S)
                if level_index == 0
                else DEFAULT_ADAPTIVE_AUDIT_LEVELS[level_index - 1]
            )
            for field, expected in (
                ("duration_s", expected_level.duration_s),
                ("coarse_time_step_s", expected_level.coarse_time_step_s),
                ("fine_time_step_s", expected_level.fine_time_step_s),
            ):
                try:
                    value = float(item[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"endpoint-audit complete-boundary {field} is malformed for {key}"
                    ) from exc
                if not np.isfinite(value) or not np.isclose(
                    value, expected, rtol=0.0, atol=1.0e-15
                ):
                    raise ValueError(
                        f"endpoint-audit complete-boundary {field} is invalid for {key}"
                    )

            def evidence_sample(name: str) -> CaptureVelocitySample:
                payload = item.get(name)
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"endpoint-audit complete-boundary {name} is malformed for {key}"
                    )
                try:
                    lower = float(payload["trapped_velocity_lower_m_per_s"])
                    upper = float(payload["untrapped_velocity_upper_m_per_s"])
                    capture = float(payload["capture_velocity_m_per_s"])
                    resolution = float(payload["velocity_resolution_m_per_s"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"endpoint-audit complete-boundary {name} bounds are malformed for {key}"
                    ) from exc
                if (
                    not all(
                        np.isfinite(value)
                        for value in (lower, upper, capture, resolution)
                    )
                    or lower < 0.0
                    or upper <= lower
                    or not np.isclose(capture, lower, rtol=0.0, atol=1.0e-12)
                    or not np.isclose(
                        resolution, upper - lower, rtol=0.0, atol=1.0e-12
                    )
                ):
                    raise ValueError(
                        f"endpoint-audit complete-boundary {name} bounds are invalid for {key}"
                    )
                return replace(
                    sample,
                    capture_velocity_m_per_s=capture,
                    velocity_resolution_m_per_s=resolution,
                    trapped_velocity_lower_m_per_s=lower,
                    untrapped_velocity_upper_m_per_s=upper,
                    lower_classification=str(payload.get("lower_classification", "")),
                    upper_classification=str(payload.get("upper_classification", "")),
                )

            evidence_coarse = evidence_sample("coarse_sample")
            evidence_fine = evidence_sample("fine_sample")
            timeout_sets: list[set[float]] = []
            for prefix in ("coarse", "fine"):
                timeout_values = item.get(f"{prefix}_timeout_speeds_m_per_s")
                timeout_set = _audit_speed_set(
                    timeout_values,
                    key=key,
                    field=(
                        f"complete_boundary_evidence[{evidence_index}]."
                        f"{prefix}_timeout_speeds_m_per_s"
                    ),
                )
                evaluations = item.get(f"{prefix}_evaluations")
                if not isinstance(evaluations, list) or not evaluations:
                    raise ValueError(
                        f"endpoint-audit complete-boundary {prefix} evaluations are missing for {key}"
                    )
                evaluation_speeds: set[float] = set()
                derived_timeouts: set[float] = set()
                evaluation_reasons: dict[float, str] = {}
                for evaluation_index, evaluation in enumerate(evaluations):
                    if not isinstance(evaluation, Mapping):
                        raise ValueError(
                            f"endpoint-audit complete-boundary evaluation is malformed for {key}"
                        )
                    speed = next(
                        iter(
                            _audit_speed_set(
                                [evaluation.get("speed_m_per_s")],
                                key=key,
                                field=(
                                    f"complete_boundary_evidence[{evidence_index}]."
                                    f"{prefix}_evaluations[{evaluation_index}].speed"
                                ),
                            )
                        )
                    )
                    if speed in evaluation_speeds:
                        raise ValueError(
                            f"endpoint-audit complete-boundary evaluations duplicate a speed for {key}"
                        )
                    evaluation_speeds.add(speed)
                    classification = evaluation.get("classification")
                    _audit_classification_state(
                        classification,
                        key=key,
                        field=(
                            f"complete_boundary_evidence[{evidence_index}]."
                            f"{prefix}_evaluations[{evaluation_index}].classification"
                        ),
                    )
                    if isinstance(classification, Mapping) and (
                        classification.get("termination_reason") == "timeout"
                    ):
                        derived_timeouts.add(speed)
                    if isinstance(classification, Mapping):
                        evaluation_reasons[speed] = str(
                            classification.get("termination_reason")
                        )
                if derived_timeouts != timeout_set:
                    raise ValueError(
                        f"endpoint-audit complete-boundary timeout evidence disagrees with its evaluations for {key}"
                    )
                if not {
                    round(evidence_sample_value, 12)
                    for evidence_sample_value in (
                        evidence_coarse.trapped_velocity_lower_m_per_s
                        if prefix == "coarse"
                        else evidence_fine.trapped_velocity_lower_m_per_s,
                        evidence_coarse.untrapped_velocity_upper_m_per_s
                        if prefix == "coarse"
                        else evidence_fine.untrapped_velocity_upper_m_per_s,
                    )
                }.issubset(evaluation_speeds):
                    raise ValueError(
                        f"endpoint-audit complete-boundary endpoints lack evaluation evidence for {key}"
                    )
                prefix_sample = (
                    evidence_coarse if prefix == "coarse" else evidence_fine
                )
                if (
                    evaluation_reasons[
                        round(prefix_sample.trapped_velocity_lower_m_per_s, 12)
                    ]
                    != prefix_sample.lower_classification
                    or evaluation_reasons[
                        round(prefix_sample.untrapped_velocity_upper_m_per_s, 12)
                    ]
                    != prefix_sample.upper_classification
                ):
                    raise ValueError(
                        f"endpoint-audit complete-boundary endpoint classifications disagree with their evaluations for {key}"
                    )
                timeout_sets.append(timeout_set)
            recomputed_compatible = _compatible_capture_brackets(
                evidence_coarse, evidence_fine, bracket_search
            )
            compatible_flag = item.get("compatible_definitive_brackets")
            timeout_free_flag = item.get("timeout_free")
            if type(compatible_flag) is not bool or (
                compatible_flag != recomputed_compatible
            ):
                raise ValueError(
                    f"endpoint-audit complete-boundary compatibility flag is invalid for {key}"
                )
            recomputed_timeout_free = not timeout_sets[0] and not timeout_sets[1]
            if type(timeout_free_flag) is not bool or (
                timeout_free_flag != recomputed_timeout_free
            ):
                raise ValueError(
                    f"endpoint-audit complete-boundary timeout-free flag is invalid for {key}"
                )
            parsed_boundary_levels.append(
                (
                    level_index,
                    evidence_coarse,
                    evidence_fine,
                    recomputed_compatible,
                    recomputed_timeout_free,
                )
            )

        if positive_grid_fallback:
            if (
                positive_grid_base_level < 1
                or positive_grid_base_level >= len(parsed_boundary_levels)
                or len(parsed_boundary_levels) != positive_grid_base_level + 1
            ):
                raise ValueError(
                    f"endpoint-audit positive-grid fallback boundary levels are incomplete for {key}"
                )
            _, _, diagnostic_fine, final_compatible, final_timeout_free = (
                parsed_boundary_levels[-1]
            )
            serialized_scalar_converged = row.get(
                "diagnostic_scalar_boundary_converged"
            )
            if type(serialized_scalar_converged) is not bool:
                serialized_scalar_converged = _audit_bool(
                    serialized_scalar_converged,
                    field="diagnostic_scalar_boundary_converged",
                )
            recomputed_scalar_converged = bool(
                final_compatible and final_timeout_free
            )
            if serialized_scalar_converged != recomputed_scalar_converged:
                raise ValueError(
                    f"endpoint-audit diagnostic scalar convergence flag is invalid for {key}"
                )
            diagnostic_scalar_level = int(
                row.get("diagnostic_scalar_boundary_level_index", -1)
            )
            expected_scalar_level = (
                positive_grid_base_level if recomputed_scalar_converged else -1
            )
            if diagnostic_scalar_level != expected_scalar_level:
                raise ValueError(
                    f"endpoint-audit diagnostic scalar level is invalid for {key}"
                )
            if not recomputed_scalar_converged and positive_grid_base_level != len(
                DEFAULT_ADAPTIVE_AUDIT_LEVELS
            ):
                raise ValueError(
                    f"endpoint-audit unconverged scalar hierarchy stopped early for {key}"
                )
            agreement = str(row.get("diagnostic_scalar_grid_agreement", ""))
            expected_agreement = (
                (
                    "agree"
                    if _compatible_capture_brackets(
                        diagnostic_fine, fine_bracket, bracket_search
                    )
                    else "disagree"
                )
                if recomputed_scalar_converged
                else "not_converged"
            )
            if agreement != expected_agreement:
                raise ValueError(
                    f"endpoint-audit scalar/grid agreement diagnostic is invalid for {key}"
                )
        elif timeout_status == "dual_step_research_recovered_boundary":
            if len(parsed_boundary_levels) != 1:
                raise ValueError(
                    f"endpoint-audit 200 ms recovered boundary lacks complete search evidence for {key}"
                )
            _, evidence_coarse, evidence_fine, compatible, timeout_free = (
                parsed_boundary_levels[0]
            )
            if (
                not compatible
                or not timeout_free
                or evidence_coarse != coarse_bracket
                or evidence_fine != fine_bracket
            ):
                raise ValueError(
                    f"endpoint-audit 200 ms recovered boundary evidence is inconsistent for {key}"
                )
        elif positive_grid_base_level != 0:
            raise ValueError(
                f"endpoint-audit unused longer boundary level is nonzero for {key}"
            )

        adaptive_level_count = int(row.get("adaptive_audit_level_count", 0) or 0)
        adaptive_schedule = (
            DEFAULT_ADAPTIVE_AUDIT_LEVELS[positive_grid_base_level:]
            if positive_grid_fallback
            else DEFAULT_ADAPTIVE_AUDIT_LEVELS
        )
        try:
            adaptive_evidence = json.loads(
                str(row.get("adaptive_audit_evidence_json", "[]")) or "[]"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"endpoint-audit adaptive evidence is not valid JSON for {key}"
            ) from exc
        if not isinstance(adaptive_evidence, list):
            raise ValueError(f"endpoint-audit adaptive evidence is not a list for {key}")
        if adaptive_level_count < 0 or adaptive_level_count > len(adaptive_schedule):
            raise ValueError(f"endpoint-audit adaptive level count is invalid for {key}")

        evidence_by_speed: dict[float, list[tuple[int, str]]] = {}
        seen_level_speed: set[tuple[int, float]] = set()
        for index, item in enumerate(adaptive_evidence):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"endpoint-audit adaptive evidence item is malformed for {key}"
                )
            raw_level = item.get("level_index")
            if isinstance(raw_level, bool):
                raise ValueError(
                    f"endpoint-audit adaptive level is malformed for {key}"
                )
            try:
                level_index = int(raw_level)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"endpoint-audit adaptive level is malformed for {key}"
                ) from exc
            if level_index < 1 or level_index > len(adaptive_schedule):
                raise ValueError(f"endpoint-audit adaptive level is invalid for {key}")
            expected_level = adaptive_schedule[level_index - 1]
            for field, expected in (
                ("duration_s", expected_level.duration_s),
                ("coarse_time_step_s", expected_level.coarse_time_step_s),
                ("fine_time_step_s", expected_level.fine_time_step_s),
            ):
                try:
                    value = float(item[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"endpoint-audit adaptive field {field} is malformed for {key}"
                    ) from exc
                if not np.isfinite(value) or not np.isclose(
                    value, expected, rtol=0.0, atol=1.0e-15
                ):
                    raise ValueError(
                        f"endpoint-audit adaptive field {field} is invalid for {key}"
                    )
            speed = next(
                iter(
                    _audit_speed_set(
                        [item.get("velocity_m_per_s")],
                        key=key,
                        field=f"adaptive_evidence[{index}].velocity_m_per_s",
                    )
                )
            )
            if (level_index, speed) in seen_level_speed:
                raise ValueError(
                    f"endpoint-audit adaptive evidence duplicates a level/node for {key}"
                )
            seen_level_speed.add((level_index, speed))
            coarse_state = _audit_classification_state(
                item.get("coarse_result"),
                key=key,
                field=f"adaptive_evidence[{index}].coarse_result",
            )
            fine_state = _audit_classification_state(
                item.get("fine_result"),
                key=key,
                field=f"adaptive_evidence[{index}].fine_result",
            )
            computed_resolved = bool(
                coarse_state in {"trapped", "escaped"}
                and coarse_state == fine_state
            )
            serialized_resolved = item.get("resolved")
            if type(serialized_resolved) is not bool or (
                serialized_resolved != computed_resolved
            ):
                raise ValueError(
                    f"endpoint-audit adaptive resolved flag disagrees with its "
                    f"dual-step classifications for {key}"
                )
            evidence_by_speed.setdefault(speed, []).append(
                (level_index, coarse_state if computed_resolved else "unresolved")
            )

        adaptive_evidence_speeds = set(evidence_by_speed)
        if adaptive_evidence:
            evidence_levels = {level for level, _speed in seen_level_speed}
            if max(evidence_levels) != adaptive_level_count:
                raise ValueError(f"endpoint-audit adaptive level count mismatch for {key}")
            for speed, records in evidence_by_speed.items():
                ordered = sorted(records)
                expected_indices = list(range(1, ordered[-1][0] + 1))
                if [level for level, _state in ordered] != expected_indices:
                    raise ValueError(
                        f"endpoint-audit adaptive evidence skips a level for {key}"
                    )
                if any(state != "unresolved" for _level, state in ordered[:-1]):
                    raise ValueError(
                        f"endpoint-audit re-evaluated an already resolved node for {key}"
                    )
                if ordered[-1][1] not in {"trapped", "escaped"}:
                    raise ValueError(
                        f"endpoint-audit retained unresolved adaptive evidence for {key}"
                    )
            expected_final = adaptive_schedule[adaptive_level_count - 1]
            for field, expected in (
                ("adaptive_max_duration_s", expected_final.duration_s),
                ("adaptive_coarse_time_step_s", expected_final.coarse_time_step_s),
                ("adaptive_fine_time_step_s", expected_final.fine_time_step_s),
            ):
                if not np.isclose(
                    float(row.get(field, np.nan)),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError(
                        f"endpoint-audit adaptive integration field {field} "
                        f"does not match its final level for {key}"
                    )
        else:
            if adaptive_level_count:
                raise ValueError(f"endpoint-audit adaptive evidence is missing for {key}")
            for field in (
                "adaptive_max_duration_s",
                "adaptive_coarse_time_step_s",
                "adaptive_fine_time_step_s",
            ):
                if not np.isclose(
                    float(row.get(field, np.nan)), 0.0, rtol=0.0, atol=1.0e-15
                ):
                    raise ValueError(
                        f"endpoint-audit unused adaptive field {field} is nonzero "
                        f"for {key}"
                    )

        if pre_adaptive_timeout_speeds and not positive_grid_fallback:
            if pre_adaptive_timeout_speeds != adaptive_evidence_speeds:
                raise ValueError(
                    f"endpoint-audit adaptive evidence does not cover the exact "
                    f"pre-adaptive timeout union for {key}"
                )
        elif timeout_status == "adaptive_dual_step_research_recovered_boundary":
            raise ValueError(
                f"endpoint-audit adaptive boundary status lacks pre-adaptive "
                f"timeout evidence for {key}"
            )
        if zero_status == "not_applicable":
            if adaptive_evidence and timeout_status not in {
                "adaptive_dual_step_research_recovered_boundary",
                "complete_scalar_diagnostics_and_velocity_grid_recovered_capture",
            }:
                raise ValueError(
                    f"endpoint-audit positive-boundary adaptive evidence has an "
                    f"invalid status for {key}"
                )
            if timeout_status == "adaptive_dual_step_research_recovered_boundary" and (
                not adaptive_evidence
            ):
                raise ValueError(
                    f"endpoint-audit adaptive boundary status lacks evidence for {key}"
                )
            if timeout_status in {
                "targeted_timeout_nodes_confirmed_escaped",
                "dual_step_research_recovered_boundary",
            } and adaptive_evidence:
                raise ValueError(
                    f"endpoint-audit nonadaptive boundary status retains adaptive "
                    f"evidence for {key}"
                )
        else:
            if pre_adaptive_timeout_speeds:
                raise ValueError(
                    f"endpoint-audit zero-grid result contains positive-boundary "
                    f"pre-adaptive timeouts for {key}"
                )
            if zero_status == "rebisected" and adaptive_evidence:
                raise ValueError(
                    f"endpoint-audit rebisected zero result contains adaptive grid "
                    f"evidence for {key}"
                )

        if positive_grid_fallback:
            if positive_grid_status != "velocity_resolved_capture":
                raise ValueError(
                    f"endpoint-audit positive-grid fallback lacks a resolved mask for {key}"
                )
            accepted_level = DEFAULT_ADAPTIVE_AUDIT_LEVELS[
                positive_grid_base_level - 1
            ]
            for field, expected in (
                ("positive_boundary_grid_duration_s", accepted_level.duration_s),
                (
                    "positive_boundary_grid_coarse_time_step_s",
                    accepted_level.coarse_time_step_s,
                ),
                (
                    "positive_boundary_grid_fine_time_step_s",
                    accepted_level.fine_time_step_s,
                ),
            ):
                if not np.isclose(
                    float(row.get(field, np.nan)),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError(
                        f"endpoint-audit positive-grid integration field {field} is invalid for {key}"
                    )
        else:
            if positive_grid_status != "not_applicable":
                raise ValueError(
                    f"endpoint-audit unexpected positive-grid status for {key}"
                )
            if int(row.get("diagnostic_scalar_boundary_level_index", -1) or -1) != -1:
                raise ValueError(
                    f"endpoint-audit unused diagnostic scalar level is nonnegative for {key}"
                )
            if _audit_bool(
                row.get("diagnostic_scalar_boundary_converged", False),
                field="diagnostic_scalar_boundary_converged",
            ):
                raise ValueError(
                    f"endpoint-audit unused diagnostic scalar convergence flag is true for {key}"
                )
            if str(
                row.get("diagnostic_scalar_grid_agreement", "not_applicable")
                or "not_applicable"
            ) != "not_applicable":
                raise ValueError(
                    f"endpoint-audit unused scalar/grid agreement diagnostic is set for {key}"
                )
            for field in (
                "positive_boundary_grid_duration_s",
                "positive_boundary_grid_coarse_time_step_s",
                "positive_boundary_grid_fine_time_step_s",
            ):
                if not np.isclose(
                    float(row.get(field, 0.0) or 0.0),
                    0.0,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError(
                        f"endpoint-audit unused positive-grid field {field} is nonzero for {key}"
                    )

        # The accepted_* values describe the authoritative complete boundary
        # level (normally 200 ms; a bounded longer level for a positive-grid
        # fallback). Adaptive grid-node work has its own fields above.
        primary_audited = timeout_status != "not_required" or zero_status != "not_applicable"
        if primary_audited:
            accepted_integration_level = (
                DEFAULT_ADAPTIVE_AUDIT_LEVELS[positive_grid_base_level - 1]
                if positive_grid_fallback
                else AdaptiveAuditLevel(
                    AUDIT_DURATION_S, COARSE_TIME_STEP_S, FINE_TIME_STEP_S
                )
            )
            for field, expected in (
                (
                    "accepted_max_simulation_time_s",
                    accepted_integration_level.duration_s,
                ),
                (
                    "accepted_coarse_time_step_s",
                    accepted_integration_level.coarse_time_step_s,
                ),
                ("accepted_time_step_s", accepted_integration_level.fine_time_step_s),
            ):
                if not np.isclose(
                    float(row.get(field, np.nan)),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError(
                        f"endpoint-audit accepted primary integration field {field} "
                        f"is invalid for {key}"
                    )

        # Re-evaluate the latest adaptive state against the direct mask or both
        # persisted brackets. Runtime positive-boundary QA applies each bracket
        # independently; the ledger must not weaken that rule by consulting only
        # the accepted fine interval. The serialized resolved flag alone never
        # authorizes a row.
        override = overrides.get(key)
        for speed, records in evidence_by_speed.items():
            latest_state = max(records)[1]
            if override is not None:
                matches = np.flatnonzero(
                    np.isclose(
                        np.asarray(override.velocity_m_per_s, dtype=float),
                        speed,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"endpoint-audit adaptive node is absent from its velocity "
                        f"override for {key}"
                    )
                expected_state = (
                    "trapped" if override.captured[int(matches[0])] else "escaped"
                )
                if latest_state != expected_state:
                    raise ValueError(
                        f"endpoint-audit latest adaptive node contradicts the "
                        f"accepted capture override for {key}"
                    )
                continue

            brackets = (
                ("coarse", coarse_bracket),
                ("fine", fine_bracket),
            )
            for bracket_name, bracket in brackets:
                expected_state: str | None = None
                if speed <= (
                    bracket.trapped_velocity_lower_m_per_s + 1.0e-12
                ):
                    expected_state = (
                        "trapped"
                        if bracket.lower_classification
                        in TRAPPED_TERMINATION_REASONS
                        else "escaped"
                    )
                elif speed >= (
                    bracket.untrapped_velocity_upper_m_per_s - 1.0e-12
                ):
                    expected_state = "escaped"
                if expected_state is not None and latest_state != expected_state:
                    raise ValueError(
                        f"endpoint-audit latest adaptive node contradicts the "
                        f"persisted {bracket_name} capture bracket for {key}"
                    )

        for prefix in ("coarse", "fine"):
            extended_count = int(
                row.get(f"{prefix}_evaluation_timeout_count", 0) or 0
            )
            raw_extended = row.get(
                f"{prefix}_evaluation_timeout_speeds_m_per_s_json", "[]"
            )
            extended_speeds = json.loads(str(raw_extended) or "[]")
            if not isinstance(extended_speeds, list) or len(extended_speeds) != extended_count:
                raise ValueError(
                    f"endpoint-audit {prefix} timeout-speed count mismatch for {key}"
                )
            if extended_count:
                raise ValueError(f"endpoint-audit retained unresolved {prefix} timeouts for {key}")

        try:
            positive_grid_evidence = json.loads(
                str(row.get("positive_boundary_grid_evidence_json", "[]")) or "[]"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"endpoint-audit positive-grid evidence is not valid JSON for {key}"
            ) from exc
        if not isinstance(positive_grid_evidence, list):
            raise ValueError(
                f"endpoint-audit positive-grid evidence is not a list for {key}"
            )
        recomputed_grid_mask: tuple[bool, ...] | None = None
        if positive_grid_fallback:
            expected_grid = np.arange(
                bracket_search.analysis_velocity_min_m_per_s,
                bracket_search.analysis_velocity_max_m_per_s
                + 0.5 * bracket_search.analysis_velocity_step_m_per_s,
                bracket_search.analysis_velocity_step_m_per_s,
                dtype=float,
            )
            if len(positive_grid_evidence) != len(expected_grid):
                raise ValueError(
                    f"endpoint-audit positive-grid evidence does not cover the exact analysis grid for {key}"
                )
            coarse_grid_reasons: list[str] = []
            fine_grid_reasons: list[str] = []
            captured_mask: list[bool] = []
            for grid_index, (expected_speed, item) in enumerate(
                zip(expected_grid, positive_grid_evidence, strict=True)
            ):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"endpoint-audit positive-grid item is malformed for {key}"
                    )
                speed = next(
                    iter(
                        _audit_speed_set(
                            [item.get("speed_m_per_s")],
                            key=key,
                            field=f"positive_grid_evidence[{grid_index}].speed",
                        )
                    )
                )
                if not np.isclose(
                    speed, expected_speed, rtol=0.0, atol=1.0e-12
                ):
                    raise ValueError(
                        f"endpoint-audit positive-grid speed ordering is invalid for {key}"
                    )
                coarse_payload = item.get("coarse_result")
                fine_payload = item.get("fine_result")
                coarse_state = _audit_classification_state(
                    coarse_payload,
                    key=key,
                    field=f"positive_grid_evidence[{grid_index}].coarse_result",
                )
                fine_state = _audit_classification_state(
                    fine_payload,
                    key=key,
                    field=f"positive_grid_evidence[{grid_index}].fine_result",
                )
                if coarse_state not in {"trapped", "escaped"} or (
                    coarse_state != fine_state
                ):
                    raise ValueError(
                        f"endpoint-audit positive-grid node is unresolved or timestep-dependent for {key}"
                    )
                if not isinstance(coarse_payload, Mapping) or not isinstance(
                    fine_payload, Mapping
                ):
                    raise ValueError(
                        f"endpoint-audit positive-grid classifications are malformed for {key}"
                    )
                coarse_grid_reasons.append(
                    str(coarse_payload.get("termination_reason"))
                )
                fine_grid_reasons.append(str(fine_payload.get("termination_reason")))
                captured_mask.append(fine_state == "trapped")
            if captured_mask[-1]:
                raise ValueError(
                    f"endpoint-audit positive-grid evidence lacks an escaped high-speed endpoint for {key}"
                )
            recomputed_grid_mask = tuple(captured_mask)

            def expected_grid_bracket(
                reasons: Sequence[str],
            ) -> tuple[float, float, str, str]:
                if captured_mask[0]:
                    upper_index = next(
                        index for index, value in enumerate(captured_mask) if not value
                    )
                    lower_index = upper_index - 1
                    return (
                        float(expected_grid[lower_index]),
                        float(expected_grid[upper_index]),
                        reasons[lower_index],
                        reasons[upper_index],
                    )
                upper_index = next(
                    index
                    for index in range(1, len(expected_grid))
                    if reasons[index] == "escaped"
                )
                return (
                    float(expected_grid[0]),
                    float(expected_grid[upper_index]),
                    reasons[0],
                    reasons[upper_index],
                )

            for bracket_name, bracket, reasons in (
                ("coarse", coarse_bracket, coarse_grid_reasons),
                ("fine", fine_bracket, fine_grid_reasons),
            ):
                lower, upper, lower_reason, upper_reason = expected_grid_bracket(
                    reasons
                )
                if not (
                    np.isclose(
                        bracket.trapped_velocity_lower_m_per_s,
                        lower,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and np.isclose(
                        bracket.untrapped_velocity_upper_m_per_s,
                        upper,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and bracket.lower_classification == lower_reason
                    and bracket.upper_classification == upper_reason
                ):
                    raise ValueError(
                        f"endpoint-audit persisted {bracket_name} bracket does not match its positive-grid evidence for {key}"
                    )
        elif positive_grid_evidence:
            raise ValueError(
                f"endpoint-audit unused positive-grid evidence is nonempty for {key}"
            )

        try:
            zero_grid_evidence = json.loads(
                str(row.get("zero_threshold_grid_evidence_json", "[]")) or "[]"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"endpoint-audit zero-grid evidence is not valid JSON for {key}"
            ) from exc
        if not isinstance(zero_grid_evidence, list):
            raise ValueError(
                f"endpoint-audit zero-grid evidence is not a list for {key}"
            )
        recomputed_zero_grid_mask: tuple[bool, ...] | None = None
        if zero_status in {
            "confirmed_zero_capture",
            "grid_resolved_capture",
            "velocity_resolved_capture",
        }:
            expected_zero_grid = np.arange(
                bracket_search.analysis_velocity_min_m_per_s,
                bracket_search.analysis_velocity_max_m_per_s
                + 0.5 * bracket_search.analysis_velocity_step_m_per_s,
                bracket_search.analysis_velocity_step_m_per_s,
                dtype=float,
            )
            if len(zero_grid_evidence) != len(expected_zero_grid):
                raise ValueError(
                    "endpoint-audit zero-grid evidence does not cover the exact "
                    f"analysis grid for {key}"
                )
            zero_coarse_reasons: list[str] = []
            zero_fine_reasons: list[str] = []
            zero_captured_mask: list[bool] = []
            for grid_index, (expected_speed, item) in enumerate(
                zip(expected_zero_grid, zero_grid_evidence, strict=True)
            ):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"endpoint-audit zero-grid item is malformed for {key}"
                    )
                speed = next(
                    iter(
                        _audit_speed_set(
                            [item.get("speed_m_per_s")],
                            key=key,
                            field=f"zero_grid_evidence[{grid_index}].speed",
                        )
                    )
                )
                if not np.isclose(
                    speed, expected_speed, rtol=0.0, atol=1.0e-12
                ):
                    raise ValueError(
                        f"endpoint-audit zero-grid speed ordering is invalid for {key}"
                    )
                coarse_payload = item.get("coarse_result")
                fine_payload = item.get("fine_result")
                coarse_state = _audit_classification_state(
                    coarse_payload,
                    key=key,
                    field=f"zero_grid_evidence[{grid_index}].coarse_result",
                )
                fine_state = _audit_classification_state(
                    fine_payload,
                    key=key,
                    field=f"zero_grid_evidence[{grid_index}].fine_result",
                )
                if coarse_state not in {"trapped", "escaped"} or (
                    coarse_state != fine_state
                ):
                    raise ValueError(
                        "endpoint-audit zero-grid node is unresolved or "
                        f"timestep-dependent for {key}"
                    )
                if not isinstance(coarse_payload, Mapping) or not isinstance(
                    fine_payload, Mapping
                ):
                    raise ValueError(
                        f"endpoint-audit zero-grid classifications are malformed for {key}"
                    )
                zero_coarse_reasons.append(
                    str(coarse_payload.get("termination_reason"))
                )
                zero_fine_reasons.append(
                    str(fine_payload.get("termination_reason"))
                )
                zero_captured_mask.append(fine_state == "trapped")
            if zero_captured_mask[-1]:
                raise ValueError(
                    "endpoint-audit zero-grid evidence lacks an escaped "
                    f"high-speed endpoint for {key}"
                )
            recomputed_zero_grid_mask = tuple(zero_captured_mask)

            def expected_zero_grid_bracket(
                reasons: Sequence[str],
            ) -> tuple[float, float, str, str]:
                if zero_captured_mask[0]:
                    upper_index = next(
                        index
                        for index, captured in enumerate(zero_captured_mask)
                        if not captured
                    )
                    lower_index = upper_index - 1
                    return (
                        float(expected_zero_grid[lower_index]),
                        float(expected_zero_grid[upper_index]),
                        reasons[lower_index],
                        reasons[upper_index],
                    )
                upper_index = next(
                    index
                    for index in range(1, len(expected_zero_grid))
                    if reasons[index] == "escaped"
                )
                return (
                    float(expected_zero_grid[0]),
                    float(expected_zero_grid[upper_index]),
                    reasons[0],
                    reasons[upper_index],
                )

            for bracket_name, bracket, reasons in (
                ("coarse", coarse_bracket, zero_coarse_reasons),
                ("fine", fine_bracket, zero_fine_reasons),
            ):
                lower, upper, lower_reason, upper_reason = (
                    expected_zero_grid_bracket(reasons)
                )
                if not (
                    np.isclose(
                        bracket.trapped_velocity_lower_m_per_s,
                        lower,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and np.isclose(
                        bracket.untrapped_velocity_upper_m_per_s,
                        upper,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    and bracket.lower_classification == lower_reason
                    and bracket.upper_classification == upper_reason
                ):
                    raise ValueError(
                        f"endpoint-audit persisted {bracket_name} bracket does "
                        f"not match its zero-grid evidence for {key}"
                    )

            first_escape = next(
                (
                    index
                    for index, captured in enumerate(zero_captured_mask)
                    if not captured
                ),
                len(zero_captured_mask),
            )
            has_later_capture = any(zero_captured_mask[first_escape + 1 :])
            if zero_status == "confirmed_zero_capture":
                if any(zero_captured_mask) or override is not None:
                    raise ValueError(
                        f"endpoint-audit confirmed-zero status disagrees with its grid for {key}"
                    )
            elif zero_status == "grid_resolved_capture":
                if (
                    not zero_captured_mask[0]
                    or has_later_capture
                    or override is not None
                ):
                    raise ValueError(
                        f"endpoint-audit contiguous zero-grid status is invalid for {key}"
                    )
            elif zero_status == "velocity_resolved_capture":
                if override is None:
                    raise ValueError(
                        f"endpoint-audit zero-grid mask/status is invalid for {key}"
                    )
        elif zero_status == "rebisected":
            if zero_grid_evidence:
                raise ValueError(
                    f"endpoint-audit rebisected zero result has grid evidence for {key}"
                )
        elif zero_grid_evidence:
            raise ValueError(
                f"endpoint-audit unused zero-grid evidence is nonempty for {key}"
            )

        override_flag = _audit_bool(
            row.get("velocity_resolved_override"), field="velocity_resolved_override"
        )
        if override_flag != (key in overrides):
            raise ValueError(f"endpoint-audit velocity-override flag mismatch for {key}")
        expected_override = bool(
            zero_status == "velocity_resolved_capture"
            or positive_grid_status == "velocity_resolved_capture"
        )
        if expected_override != override_flag:
            raise ValueError(
                f"endpoint-audit velocity-resolved status/override mismatch for {key}"
            )
        if override is not None:
            expected_grid = np.arange(
                bracket_search.analysis_velocity_min_m_per_s,
                bracket_search.analysis_velocity_max_m_per_s
                + 0.5 * bracket_search.analysis_velocity_step_m_per_s,
                bracket_search.analysis_velocity_step_m_per_s,
                dtype=float,
            )
            if not np.array_equal(
                np.asarray(override.velocity_m_per_s, dtype=float), expected_grid
            ):
                raise ValueError(
                    f"endpoint-audit velocity override does not cover the exact analysis grid for {key}"
                )
            if len(override.captured) != len(expected_grid) or bool(
                override.captured[-1]
            ):
                raise ValueError(
                    f"endpoint-audit velocity override lacks an escaped high-speed endpoint for {key}"
                )
            if positive_grid_fallback and tuple(override.captured) != (
                recomputed_grid_mask
            ):
                raise ValueError(
                    f"endpoint-audit positive-grid override disagrees with dual-step evidence for {key}"
                )
            if zero_status == "velocity_resolved_capture" and tuple(
                override.captured
            ) != recomputed_zero_grid_mask:
                raise ValueError(
                    f"endpoint-audit zero-grid override disagrees with dual-step evidence for {key}"
                )
        if not np.isclose(
            float(row["accepted_trapped_velocity_lower_m_per_s"]),
            sample.trapped_velocity_lower_m_per_s,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.isclose(
            float(row["accepted_untrapped_velocity_upper_m_per_s"]),
            sample.untrapped_velocity_upper_m_per_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"endpoint-audit accepted bracket mismatch for {key}")
        if (
            str(row["accepted_lower_classification"]) != sample.lower_classification
            or str(row["accepted_upper_classification"]) != sample.upper_classification
        ):
            raise ValueError(f"endpoint-audit accepted classification mismatch for {key}")


def _save_checkpoint(
    paths: StudyPaths,
    samples: Mapping[tuple[int, int], CaptureVelocitySample],
    audit_rows: Mapping[tuple[int, int], Mapping[str, object]],
    overrides: Mapping[tuple[int, int], VelocityResolvedCaptureOverride],
    metadata: dict[str, object],
    *,
    elapsed_wall_time_s: float,
    eta_s: float | None,
    status: str = "running",
    error: str | None = None,
) -> None:
    save_samples_atomic(paths.partial_samples_csv, samples.values())
    ordered_audit = [audit_rows[key] for key in sorted(audit_rows)]
    _atomic_write_csv(_audit_path(paths), ordered_audit, ENDPOINT_AUDIT_FIELDNAMES)
    _atomic_write_json(
        _overrides_path(paths),
        velocity_overrides_to_payload([overrides[key] for key in sorted(overrides)]),
    )
    updated = dict(metadata)
    updated.update(
        {
            "status": status,
            "updated_utc": _utc_now(),
            "completed_sample_count": len(samples),
            "completion_fraction": len(samples) / int(metadata["expected_sample_count"]),
            "elapsed_wall_time_s": elapsed_wall_time_s,
            "eta_s": eta_s,
            "automatically_extended_timeout_ray_count": sum(
                str(row["base_timeout_detected"]).lower() == "true"
                for row in audit_rows.values()
            ),
            "adaptively_extended_velocity_grid_ray_count": sum(
                int(row.get("adaptive_audit_level_count", 0) or 0) > 0
                and str(row.get("zero_threshold_audit_status", ""))
                != "not_applicable"
                for row in audit_rows.values()
            ),
            "adaptively_extended_audit_ray_count": sum(
                int(row.get("adaptive_audit_level_count", 0) or 0) > 0
                for row in audit_rows.values()
            ),
            "adaptively_extended_positive_boundary_ray_count": sum(
                str(row.get("timeout_resolution_status", ""))
                == "adaptive_dual_step_research_recovered_boundary"
                for row in audit_rows.values()
            ),
            "complete_longer_boundary_fallback_ray_count": sum(
                str(row.get("timeout_resolution_status", ""))
                == "complete_scalar_diagnostics_and_velocity_grid_recovered_capture"
                for row in audit_rows.values()
            ),
            "positive_boundary_grid_override_count": sum(
                str(row.get("positive_boundary_grid_audit_status", ""))
                == "velocity_resolved_capture"
                for row in audit_rows.values()
            ),
            "pre_adaptive_positive_research_timeout_ray_count": sum(
                (
                    int(
                        row.get(
                            "pre_adaptive_coarse_evaluation_timeout_count", 0
                        )
                        or 0
                    )
                    + int(
                        row.get("pre_adaptive_fine_evaluation_timeout_count", 0)
                        or 0
                    )
                )
                > 0
                for row in audit_rows.values()
            ),
            "velocity_resolved_override_count": len(overrides),
        }
    )
    if error is not None:
        updated["last_error"] = error
    _atomic_write_json(paths.metadata_json, updated)
    metadata.clear()
    metadata.update(updated)


def _known_point_outputs(paths: StudyPaths) -> tuple[Path, ...]:
    return (
        paths.metadata_json,
        paths.geometry_csv,
        paths.partial_samples_csv,
        paths.final_samples_csv,
        paths.capture_summary_json,
        paths.spectrum_csv,
        paths.loading_by_disc_csv,
        paths.loading_json,
        _audit_path(paths),
        _overrides_path(paths),
        paths.cross_section_png,
        paths.impact_parameter_png,
        paths.loading_by_disc_png,
    )


def _velocity_grid(
    samples: Sequence[CaptureVelocitySample], search: CaptureSearchConfig
) -> np.ndarray:
    if not samples:
        raise ValueError("at least one capture sample is required")
    step = search.analysis_velocity_step_m_per_s
    start = search.analysis_velocity_min_m_per_s
    stop = search.analysis_velocity_max_m_per_s
    if step <= 0.0 or start < 0.0 or stop <= start:
        raise ValueError("analysis velocity grid must be finite, positive, and ordered")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def _clamp_spectrum_roundoff(
    rows: Sequence[dict[str, int | float]], search: CaptureSearchConfig
) -> None:
    """Keep floating-point cross-section intervals physically ordered.

    A mean of identical full-disc areas can differ from ``pi*R**2`` by one
    rounding unit. Clamping that numerical dust prevents a nominal lower
    confidence bound from appearing infinitesimally above its upper bound.
    """

    area = pi * search.disc_radius_m**2
    for row in rows:
        mean = float(np.clip(float(row["capture_cross_section_m2"]), 0.0, area))
        lower = float(
            np.clip(float(row["capture_cross_section_t95_lower_m2"]), 0.0, mean)
        )
        upper = float(
            np.clip(float(row["capture_cross_section_t95_upper_m2"]), mean, area)
        )
        row["capture_cross_section_m2"] = mean
        row["capture_fraction"] = mean / area
        row["capture_cross_section_t95_lower_m2"] = lower
        row["capture_cross_section_t95_upper_m2"] = upper


def _analyze_point(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    point: RelationshipPoint,
    paths: StudyPaths,
    *,
    signature: str,
    geometry_hash: str,
    overrides: Sequence[VelocityResolvedCaptureOverride] = (),
) -> dict[str, object]:
    expected = search.disc_count * search.points_per_disc
    if len(samples) != expected:
        raise ValueError(f"point analysis requires {expected} samples")
    grouped = Counter(sample.disc_index for sample in samples)
    if set(grouped) != set(range(search.disc_count)) or any(
        count != search.points_per_disc for count in grouped.values()
    ):
        raise ValueError("point samples do not contain the complete disc geometry")
    if any(not _valid_capture_endpoint(sample) for sample in samples):
        raise ValueError("point samples contain an unresolved or invalid endpoint")

    velocity_grid = _velocity_grid(samples, search)
    if overrides:
        override_grid = np.asarray(overrides[0].velocity_m_per_s, dtype=float)
        if any(
            not np.array_equal(np.asarray(item.velocity_m_per_s), override_grid)
            for item in overrides[1:]
        ):
            raise ValueError("velocity-resolved overrides do not share one exact grid")
        if not np.array_equal(velocity_grid, override_grid):
            raise ValueError(
                "velocity-resolved override grid does not cover the complete point spectrum"
            )
    spectrum = calculate_clustered_cross_section_with_overrides(
        samples,
        search,
        overrides,
        velocity_grid,
    )
    _clamp_spectrum_roundoff(spectrum, search)
    by_disc, loading = calculate_disc_clustered_loading_with_overrides(
        samples, search, spectrum, overrides
    )
    _atomic_write_csv(paths.spectrum_csv, spectrum, SPECTRUM_FIELDNAMES)
    _atomic_write_csv(paths.loading_by_disc_csv, by_disc, LOADING_BY_DISC_FIELDNAMES)
    loading_payload = {
        **loading,
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "run_signature_sha256": signature,
        "geometry_sha256": geometry_hash,
        "spectrum_csv": str(paths.spectrum_csv.resolve()),
        "samples_csv": str(paths.final_samples_csv.resolve()),
        "loading_rate_by_disc_csv": str(paths.loading_by_disc_csv.resolve()),
        "velocity_resolved_override_count": len(overrides),
        "velocity_resolved_overrides_json": str(_overrides_path(paths).resolve()),
        "velocity_override_interpretation": (
            "Cross-section and loading integrals use every saved direct boolean mask. "
            "For an overridden ray, the scalar threshold in the samples CSV and "
            "impact-parameter plot represents only the first contiguous low-speed "
            "capture interval (or a zero fallback); the direct mask is authoritative "
            "for the physical velocity-resolved capture result."
        ),
    }
    _atomic_write_json(paths.loading_json, loading_payload)
    plot_clustered_cross_section(
        spectrum,
        paths.cross_section_png,
        power_w_per_beam=point.cooling_power_w_per_beam,
    )
    plot_capture_velocity_vs_impact_parameter(
        samples,
        paths.impact_parameter_png,
        power_w_per_beam=point.cooling_power_w_per_beam,
        velocity_resolved_override_count=len(overrides),
    )
    plot_loading_rate_by_disc(
        by_disc,
        loading,
        paths.loading_by_disc_png,
        power_w_per_beam=point.cooling_power_w_per_beam,
    )
    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples])
    summary = {
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "run_signature_sha256": signature,
        "geometry_sha256": geometry_hash,
        "sample_count": len(samples),
        "expected_sample_count": expected,
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "capture_velocity_mean_m_per_s": float(np.mean(capture)),
        "capture_velocity_sample_std_m_per_s": (
            float(np.std(capture, ddof=1)) if len(capture) > 1 else 0.0
        ),
        "capture_velocity_min_m_per_s": float(np.min(capture)),
        "capture_velocity_max_m_per_s": float(np.max(capture)),
        "zero_capture_velocity_count": int(np.count_nonzero(capture == 0.0)),
        "lower_classification_counts": dict(
            sorted(Counter(sample.lower_classification for sample in samples).items())
        ),
        "upper_classification_counts": dict(
            sorted(Counter(sample.upper_classification for sample in samples).items())
        ),
        "unresolved_timeout_count": sum(_sample_has_timeout(sample) for sample in samples),
        "velocity_resolved_override_count": len(overrides),
        "velocity_resolved_overrides_json": str(_overrides_path(paths).resolve()),
        "velocity_override_interpretation": loading_payload[
            "velocity_override_interpretation"
        ],
        "loading_rate": loading_payload,
        "relationship_point": asdict(point),
        "search_config": asdict(search),
        "phase_space": "full_sphere",
    }
    _atomic_write_json(paths.capture_summary_json, summary)
    return summary


def _new_point_metadata(
    point: RelationshipPoint,
    search: CaptureSearchConfig,
    signature_payload: Mapping[str, object],
    signature: str,
    geometry_hash: str,
    paths: StudyPaths,
    *,
    worker_count: int,
    completed_sample_count: int,
    started_utc: str | None = None,
    elapsed_wall_time_s: float = 0.0,
) -> dict[str, object]:
    apparatus, simple, coil, beams = build_point_configuration(point)
    expected = search.disc_count * search.points_per_disc
    return {
        "schema_version": POINT_SCHEMA_VERSION,
        "study_name": f"{CAMPAIGN_NAME}/{point.study_key}/{point.slug}",
        "status": "running",
        "started_utc": started_utc or _utc_now(),
        "updated_utc": _utc_now(),
        "completed_sample_count": completed_sample_count,
        "expected_sample_count": expected,
        "completion_fraction": completed_sample_count / expected,
        "elapsed_wall_time_s": elapsed_wall_time_s,
        "eta_s": None,
        "run_signature_sha256": signature,
        "signature_payload": signature_payload,
        "geometry_sha256": geometry_hash,
        "relationship_point": asdict(point),
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "state_model": "effective two-level Rb-87 D2 atom; no repumper or state engine",
        "capture_dynamics": "deterministic RK4 mean force with gravity; recoil diffusion absent",
        "phase_space": "full_sphere",
        "sampling_method": (
            "disc normals uniform over 4pi; disc points independent and uniform in area; "
            "all velocities on a disc are parallel to its inward normal"
        ),
        "cross_section_convention": (
            "direction-disc average of projected capture areas; no octant or 4pi "
            "multiplicity factor"
        ),
        "trapped_definition": (
            f"inside the {1e3 * search.trap_core_radius_m:g} mm core continuously for "
            f"{1e3 * search.bounded_core_residence_s:g} ms OR "
            f"{search.required_core_entries} entries with an intervening exit"
        ),
        "uncertainty": (
            f"two-sided 95% Student-t interval across {search.disc_count} independent "
            f"direction-disc clusters (df={search.disc_count - 1}); the "
            f"{expected} individual rays are not treated as independent replicates"
        ),
        "saturation_convention": (
            "s0=I0/I_sat for one Gaussian beam at its center, I0=2P/(pi*w^2); "
            "s_eff=s0/[1+(2*Delta/Gamma)^2]. The force law separately uses the "
            "sum of all six local beam saturation parameters in its denominator."
        ),
        "endpoint_timeout_policy": signature_payload["endpoint_timeout_policy"],
        "apparatus_config": asdict(apparatus),
        "simple_mot_config": asdict(simple),
        "coil_config": asdict(coil),
        "capture_search_config": asdict(search),
        "built_cooling_beam_count": len(beams),
        "built_cooling_beam_powers_w": [beam.intensity_beam.power_w for beam in beams],
        "built_cooling_beam_detunings_hz": [beam.detuning_hz for beam in beams],
        "repumper_included": False,
        "worker_count": worker_count,
        "zero_grid_ray_batch_size": ZERO_GRID_RAY_BATCH_SIZE,
        "execution_batching_interpretation": (
            "independent ray-velocity states share vectorized array calls only; "
            "statistics and terminal histories remain per ray"
        ),
        "git": _git_provenance(simple_mot_paths()["root"]),
        "outputs": {
            "statistics_directory": str(paths.statistics.resolve()),
            "figures_directory": str(paths.figures.resolve()),
            "samples_csv": str(paths.final_samples_csv.resolve()),
            "capture_cross_section_csv": str(paths.spectrum_csv.resolve()),
            "loading_rate_json": str(paths.loading_json.resolve()),
            "capture_endpoint_audit_csv": str(_audit_path(paths).resolve()),
            "capture_velocity_overrides_json": str(_overrides_path(paths).resolve()),
        },
    }


def _abort_process_pool(
    executor: ProcessPoolExecutor,
    futures: Iterable[object],
) -> None:
    """Cancel queued work and promptly stop workers after the first failure.

    ``ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)`` cancels
    work that has not started, but on Python versions before 3.14 it does not
    stop already-running calls.  A long 200 ms timeout audit can therefore keep
    the interpreter alive for minutes after one sibling has already failed.
    Prefer the public 3.14 termination API when present; otherwise retain the
    worker handles before non-waiting shutdown and terminate them explicitly.
    Every operation is best-effort because this function runs while preserving
    the original worker exception and the caller's checkpoint.
    """

    for future in futures:
        cancel = getattr(future, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except BaseException:
                pass

    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):
        try:
            terminate_workers()
            return
        except BaseException:
            # Fall through to the pre-3.14 compatibility path if a partially
            # initialized pool cannot use the public termination method.
            pass

    process_mapping = getattr(executor, "_processes", None)
    processes = tuple(process_mapping.values()) if process_mapping else ()
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except BaseException:
        pass

    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except BaseException:
            pass

    # Use one shared deadline rather than waiting once per worker.  In normal
    # operation terminate() is immediate; kill() is only a final fallback.
    join_deadline = perf_counter() + 1.0
    for process in processes:
        try:
            process.join(timeout=max(0.0, join_deadline - perf_counter()))
        except BaseException:
            pass
    for process in processes:
        try:
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                else:
                    process.terminate()
        except BaseException:
            pass


def run_relationship_point(
    point: RelationshipPoint,
    *,
    search: CaptureSearchConfig,
    discs: Sequence[DiscSample],
    points: Sequence[PointSample],
    geometry_text: str,
    geometry_hash: str,
    paths: StudyPaths,
    worker_count: int,
    resume: bool,
    analyze_only: bool = False,
) -> dict[str, object]:
    """Run or resume one complete 625-ray relationship point."""

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    if len(discs) != search.disc_count or len(points) != search.disc_count * search.points_per_disc:
        raise ValueError("common geometry does not match the search design")
    paths.statistics.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    signature_payload = _point_signature_payload(point, search, geometry_hash)
    signature = _signature(signature_payload)
    expected = len(points)
    known_outputs = _known_point_outputs(paths)
    existing_state = any(path.exists() for path in known_outputs)
    if existing_state and not resume:
        for path in known_outputs:
            if path.is_file():
                path.unlink()
        existing_state = False

    prior_metadata: dict[str, object] | None = None
    results: dict[tuple[int, int], CaptureVelocitySample] = {}
    audit_rows: dict[tuple[int, int], dict[str, object]] = {}
    overrides: dict[tuple[int, int], VelocityResolvedCaptureOverride] = {}
    if existing_state:
        if not paths.metadata_json.is_file() or not paths.geometry_csv.is_file():
            raise ValueError("resume state is missing metadata or launch geometry")
        prior_metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if prior_metadata.get("run_signature_sha256") != signature:
            raise ValueError("point resume signature mismatch")
        if _sha256_file(paths.geometry_csv) != geometry_hash:
            raise ValueError("point resume geometry hash mismatch")
        sample_path = (
            paths.final_samples_csv
            if paths.final_samples_csv.is_file()
            else paths.partial_samples_csv
        )
        if sample_path.is_file():
            results = validate_checkpoint_samples(
                load_capture_velocity_samples(sample_path), points
            )
        audit_rows = _load_audit_rows(_audit_path(paths))
        if set(audit_rows) != set(results):
            raise ValueError("endpoint-audit checkpoint keys do not match sample keys")
        if _overrides_path(paths).is_file():
            override_payload = json.loads(
                _overrides_path(paths).read_text(encoding="utf-8")
            )
            loaded_overrides = velocity_overrides_from_payload(override_payload)
            overrides = {item.key: item for item in loaded_overrides}
        elif results:
            raise ValueError("capture checkpoint is missing its velocity-override ledger")
        if not set(overrides).issubset(results):
            raise ValueError("velocity-override keys do not match saved capture samples")
        if results:
            _validate_completed_audit_ledger(
                results, audit_rows, overrides, search=search
            )
    else:
        _atomic_write_text(paths.geometry_csv, geometry_text)

    apparatus, simple, coil, _ = build_point_configuration(point)
    prior_elapsed = (
        float(prior_metadata.get("elapsed_wall_time_s", 0.0))
        if prior_metadata
        else 0.0
    )
    metadata = _new_point_metadata(
        point,
        search,
        signature_payload,
        signature,
        geometry_hash,
        paths,
        worker_count=worker_count,
        completed_sample_count=len(results),
        started_utc=str(prior_metadata["started_utc"]) if prior_metadata else None,
        elapsed_wall_time_s=prior_elapsed,
    )
    _atomic_write_json(paths.metadata_json, metadata)

    if analyze_only:
        if len(results) != expected:
            raise ValueError("analyze-only requires a complete point checkpoint")
        ordered = sorted(results.values(), key=lambda item: (item.disc_index, item.point_index))
        _validate_completed_audit_ledger(
            results, audit_rows, overrides, search=search
        )
        save_samples_atomic(paths.final_samples_csv, ordered)
        summary = _analyze_point(
            ordered,
            search,
            point,
            paths,
            signature=signature,
            geometry_hash=geometry_hash,
            overrides=[overrides[key] for key in sorted(overrides)],
        )
        metadata.update(
            {
                "status": "completed",
                "updated_utc": _utc_now(),
                "completed_sample_count": expected,
                "completion_fraction": 1.0,
                "analysis_only_invocation": True,
                "velocity_resolved_override_count": len(overrides),
                "loading_rate": summary["loading_rate"],
            }
        )
        _atomic_write_json(paths.metadata_json, metadata)
        return summary

    point_map = {(item.disc_index, item.point_index): item for item in points}
    missing = [item for key, item in point_map.items() if key not in results]
    print(
        f"[{_utc_now()}] [two-level {point.study_key}/{point.slug}] "
        f"{len(results)}/{expected} complete; running {len(missing)} with "
        f"{worker_count} worker(s); P={1e3 * point.cooling_power_w_per_beam:.9g} mW, "
        f"Delta/Gamma={point.cooling_detuning_n:.9g}",
        flush=True,
    )
    segment_start = perf_counter()
    initial_count = len(results)
    last_eta: float | None = None
    next_progress_count = (
        (initial_count // PROGRESS_EVERY) + 1
    ) * PROGRESS_EVERY
    next_checkpoint_count = (
        (initial_count // CHECKPOINT_EVERY) + 1
    ) * CHECKPOINT_EVERY
    executor: ProcessPoolExecutor | None = None
    submitted_futures: list[object] = []
    worker_batches = _ray_batches(missing)
    try:
        if worker_count == 1:
            _initialize_worker(apparatus, simple, coil, search)
            batch_iterator: Iterable[tuple[CaptureWorkerResult, ...]] = (
                _capture_worker_batch(batch) for batch in worker_batches
            )
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_worker,
                initargs=(apparatus, simple, coil, search),
            )
            futures = {}
            for batch in worker_batches:
                future = executor.submit(_capture_worker_batch, batch)
                submitted_futures.append(future)
                futures[future] = batch
            batch_iterator = (
                future.result() for future in as_completed(futures)
            )

        iterator = (
            envelope
            for completed_batch in batch_iterator
            for envelope in completed_batch
        )

        for envelope in iterator:
            sample = envelope.sample
            key = (sample.disc_index, sample.point_index)
            validate_checkpoint_samples([sample], [point_map[key]])
            if not _valid_capture_endpoint(sample):
                raise RuntimeError(f"worker returned invalid endpoint for {key}")
            results[key] = sample
            audit_rows[key] = envelope.audit_row
            if envelope.velocity_override is None:
                overrides.pop(key, None)
            else:
                if envelope.velocity_override.key != key:
                    raise RuntimeError("worker returned a velocity override for the wrong ray")
                overrides[key] = envelope.velocity_override
            completed = len(results)
            segment_elapsed = perf_counter() - segment_start
            new_count = completed - initial_count
            rate = new_count / segment_elapsed if segment_elapsed > 0.0 else 0.0
            last_eta = (expected - completed) / rate if rate > 0.0 else None
            if completed >= next_progress_count or completed == expected:
                eta_text = "unknown" if last_eta is None else f"{last_eta / 60.0:.1f} min"
                print(
                    f"[{_utc_now()}] [two-level {point.study_key}/{point.slug}] "
                    f"{completed}/{expected}; vc={sample.capture_velocity_m_per_s:.3f} m/s; "
                    f"point ETA={eta_text}",
                    flush=True,
                )
                while next_progress_count <= completed:
                    next_progress_count += PROGRESS_EVERY
            if completed >= next_checkpoint_count:
                _save_checkpoint(
                    paths,
                    results,
                    audit_rows,
                    overrides,
                    metadata,
                    elapsed_wall_time_s=prior_elapsed + segment_elapsed,
                    eta_s=last_eta,
                )
                while next_checkpoint_count <= completed:
                    next_checkpoint_count += CHECKPOINT_EVERY
    except BaseException as exc:
        if executor is not None:
            _abort_process_pool(executor, submitted_futures)
            executor = None
        _save_checkpoint(
            paths,
            results,
            audit_rows,
            overrides,
            metadata,
            elapsed_wall_time_s=prior_elapsed + perf_counter() - segment_start,
            eta_s=last_eta,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if len(results) != expected:
        raise RuntimeError(f"point sampling produced {len(results)} of {expected} samples")
    _validate_completed_audit_ledger(
        results, audit_rows, overrides, search=search
    )
    ordered = sorted(results.values(), key=lambda item: (item.disc_index, item.point_index))
    save_samples_atomic(paths.partial_samples_csv, ordered)
    save_samples_atomic(paths.final_samples_csv, ordered)
    _atomic_write_csv(
        _audit_path(paths),
        [audit_rows[key] for key in sorted(audit_rows)],
        ENDPOINT_AUDIT_FIELDNAMES,
    )
    _atomic_write_json(
        _overrides_path(paths),
        velocity_overrides_to_payload([overrides[key] for key in sorted(overrides)]),
    )
    summary = _analyze_point(
        ordered,
        search,
        point,
        paths,
        signature=signature,
        geometry_hash=geometry_hash,
        overrides=[overrides[key] for key in sorted(overrides)],
    )
    elapsed = prior_elapsed + perf_counter() - segment_start
    metadata.update(
        {
            "status": "completed",
            "updated_utc": _utc_now(),
            "completed_sample_count": expected,
            "completion_fraction": 1.0,
            "elapsed_wall_time_s": elapsed,
            "eta_s": 0.0,
            "automatically_extended_timeout_ray_count": sum(
                str(row["base_timeout_detected"]).lower() == "true"
                for row in audit_rows.values()
            ),
            "velocity_resolved_override_count": len(overrides),
            "loading_rate": summary["loading_rate"],
        }
    )
    _atomic_write_json(paths.metadata_json, metadata)
    return summary


def _point_aggregate_row(
    point: RelationshipPoint,
    search: CaptureSearchConfig,
    paths: StudyPaths,
    summary: Mapping[str, object],
) -> dict[str, object]:
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    loading = summary["loading_rate"]
    if not isinstance(loading, Mapping):
        raise ValueError("point summary has no loading-rate mapping")
    apparatus = metadata["apparatus_config"]
    simple = metadata["simple_mot_config"]
    if not isinstance(apparatus, Mapping) or not isinstance(simple, Mapping):
        raise ValueError("point metadata configuration is malformed")
    expected = search.disc_count * search.points_per_disc
    if (
        int(summary.get("sample_count", -1)) != expected
        or int(metadata.get("completed_sample_count", -1)) != expected
        or metadata.get("status") != "completed"
    ):
        raise ValueError(f"point {point.slug} is not a complete {expected}-ray result")
    cooling = apparatus["cooling"]
    if not isinstance(cooling, Mapping):
        raise ValueError("point cooling configuration is malformed")
    diameter_m = float(cooling["beam_diameter_m"])
    peak_intensity = 2.0 * point.cooling_power_w_per_beam / (
        pi * (0.5 * diameter_m) ** 2
    )
    actual_s0 = peak_intensity / float(simple["saturation_intensity_w_per_m2"])
    actual_seff = effective_saturation_from_s0(actual_s0, point.cooling_detuning_n)
    if not np.isclose(actual_s0, point.on_resonance_saturation, rtol=2.0e-13, atol=0.0):
        raise ValueError(f"point {point.slug} recorded power does not reproduce its s0")
    if not np.isclose(actual_seff, point.effective_saturation, rtol=2.0e-13, atol=0.0):
        raise ValueError(f"point {point.slug} recorded power does not reproduce its s_eff")
    if int(summary.get("unresolved_timeout_count", -1)) != 0:
        raise ValueError(f"point {point.slug} contains unresolved trajectory timeouts")
    mean_loading = float(loading["loading_rate_mean_atoms_per_s"])
    spectrum_loading = float(loading["loading_rate_from_mean_spectrum_atoms_per_s"])
    if not np.isclose(mean_loading, spectrum_loading, rtol=5.0e-13, atol=1.0e-6):
        raise ValueError(f"point {point.slug} loading estimators disagree")
    return {
        "point_index": point.point_index,
        "study_key": point.study_key,
        "scan_variable": point.scan_variable,
        "scan_value": point.scan_value,
        "s0": point.on_resonance_saturation,
        "seff": point.effective_saturation,
        "detuning_n": point.cooling_detuning_n,
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "cooling_power_mw_per_beam": 1.0e3 * point.cooling_power_w_per_beam,
        "cooling_beam_diameter_m": diameter_m,
        "cooling_beam_diameter_mm": 1.0e3 * diameter_m,
        "cooling_beam_center_peak_intensity_w_per_m2": peak_intensity,
        "cooling_beam_center_on_resonance_saturation_parameter": actual_s0,
        "cooling_beam_center_effective_saturation_parameter": actual_seff,
        "cooling_detuning_hz": point.cooling_detuning_hz,
        "cooling_detuning_mhz": point.cooling_detuning_hz / 1.0e6,
        "linewidth_hz": float(simple["linewidth_hz"]),
        "loading_rate_mean_atoms_per_s": float(loading["loading_rate_mean_atoms_per_s"]),
        "loading_rate_from_mean_spectrum_atoms_per_s": float(
            loading["loading_rate_from_mean_spectrum_atoms_per_s"]
        ),
        "loading_rate_sample_std_atoms_per_s": float(
            loading["loading_rate_sample_std_atoms_per_s"]
        ),
        "loading_rate_disc_cluster_sem_atoms_per_s": float(
            loading["loading_rate_disc_cluster_sem_atoms_per_s"]
        ),
        "loading_rate_t95_lower_atoms_per_s": float(
            loading["loading_rate_t95_lower_atoms_per_s"]
        ),
        "loading_rate_t95_upper_atoms_per_s": float(
            loading["loading_rate_t95_upper_atoms_per_s"]
        ),
        "student_t_critical_95": float(loading["student_t_critical_95"]),
        "confidence_level": float(loading["confidence_level"]),
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "capture_threshold_search_count": search.disc_count * search.points_per_disc,
        "base_timeout_ray_count": int(
            metadata.get("automatically_extended_timeout_ray_count", 0)
        ),
        "zero_capture_velocity_count": int(summary["zero_capture_velocity_count"]),
        "velocity_resolved_override_count": int(
            summary["velocity_resolved_override_count"]
        ),
        "unresolved_timeout_count": int(summary["unresolved_timeout_count"]),
        "lower_classification_counts_json": json.dumps(
            summary["lower_classification_counts"], sort_keys=True, separators=(",", ":")
        ),
        "upper_classification_counts_json": json.dumps(
            summary["upper_classification_counts"], sort_keys=True, separators=(",", ":")
        ),
        "disc_radius_m": search.disc_radius_m,
        "disc_radius_mm": 1.0e3 * search.disc_radius_m,
        "phase_space": "full_sphere",
        "geometry_sha256": summary["geometry_sha256"],
        "run_signature_sha256": summary["run_signature_sha256"],
        "point_elapsed_wall_time_s": float(metadata["elapsed_wall_time_s"]),
        "statistics_directory": str(paths.statistics.resolve()),
        "figures_directory": str(paths.figures.resolve()),
        "capture_cross_section_csv": str(paths.spectrum_csv.resolve()),
        "capture_cross_section_plot": str(paths.cross_section_png.resolve()),
        "status": "completed",
    }


def _read_aggregate(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_loading_relationship(
    rows: Sequence[Mapping[str, object]],
    study_key: str,
    path: Path,
) -> Path:
    """Plot a two-level loading relationship with clustered error bars."""

    if not rows:
        raise ValueError("at least one aggregate row is required")
    definitions = {
        RAW_STUDY_KEY: (
            "s0",
            r"On-resonance single-beam saturation $s_0$",
            "Two-Level MOT Loading Rate vs. On-Resonance Saturation "
            "(cooling detuning = -15 MHz)",
        ),
        EFFECTIVE_STUDY_KEY: (
            "seff",
            r"Effective single-beam saturation $s_{\mathrm{eff}}$",
            "Two-Level MOT Loading Rate vs. Effective Saturation "
            "(cooling detuning = -15 MHz)",
        ),
        DETUNING_STUDY_KEY: (
            "detuning_n",
            r"Cooling detuning $\Delta/\Gamma$",
            "Two-Level MOT Loading Rate vs. Cooling Detuning (27 mW per beam)",
        ),
    }
    if study_key not in definitions:
        raise ValueError(f"unknown relationship study: {study_key}")
    xfield, xlabel, title = definitions[study_key]
    ordered = sorted(rows, key=lambda row: float(row[xfield]))
    x = np.asarray([float(row[xfield]) for row in ordered])
    mean = np.asarray([float(row["loading_rate_mean_atoms_per_s"]) for row in ordered])
    lower = np.asarray([float(row["loading_rate_t95_lower_atoms_per_s"]) for row in ordered])
    upper = np.asarray([float(row["loading_rate_t95_upper_atoms_per_s"]) for row in ordered])
    yerr = np.vstack((mean - lower, upper - mean)) / 1.0e6
    figure, axis = plt.subplots(figsize=(9.4, 6.8))
    figure.subplots_adjust(left=0.125, right=0.985, top=0.90, bottom=0.22)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.errorbar(
        x,
        mean / 1.0e6,
        yerr=yerr,
        fmt="o-",
        color="#0f766e",
        ecolor="#9f4a13",
        linewidth=1.8,
        markersize=5.4,
        capsize=3.0,
        label="Mean with 95% direction-cluster t interval",
    )

    reference_s0 = on_resonance_saturation_parameter(
        DEFAULT_COOLING_POWER_W_PER_BEAM
    )
    baseline_n = (
        default_simple_mot_config().cooling_detuning_hz
        / default_simple_mot_config().linewidth_hz
    )
    if study_key in {RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY}:
        reference_x = (
            reference_s0
            if study_key == RAW_STUDY_KEY
            else effective_saturation_from_s0(reference_s0, baseline_n)
        )
        axis.axvline(
            reference_x,
            color="#475569",
            linestyle="--",
            linewidth=1.3,
            label=f"27 mW reference ({reference_x:.3f})",
        )

    if study_key == RAW_STUDY_KEY:
        axis.set_xlim(0.0, 127.5)
        axis.set_xticks((0.0, 25.0, 50.0, 75.0, 100.0, 125.0))
        inset = axis.inset_axes((0.49, 0.46, 0.47, 0.48))
        inset_mask = x <= 5.0 + 1.0e-12
        inset.errorbar(
            x[inset_mask],
            mean[inset_mask] / 1.0e6,
            yerr=yerr[:, inset_mask],
            fmt="o-",
            color="#0f766e",
            ecolor="#9f4a13",
            linewidth=1.3,
            markersize=4.0,
            capsize=2.0,
        )
        inset.set_xlim(0.0, 5.15)
        inset.set_title(r"Low-$s_0$ detail", fontsize=9)
        inset.grid(True, alpha=0.22)
        inset.tick_params(labelsize=8)
    elif study_key == EFFECTIVE_STUDY_KEY:
        axis.set_xlim(0.15, 5.10)
        axis.set_xticks((0.25, 1.0, 2.0, 3.0, 4.0, 5.0))
    else:
        axis.set_xlim(-6.12, -0.38)
        axis.set_xticks(np.arange(-6.0, -0.49, 0.5))

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Loading rate [$10^6$ atoms/s]")
    axis.grid(True, alpha=0.25)
    axis.legend(
        frameon=False,
        loc="upper left" if study_key == RAW_STUDY_KEY else "best",
    )
    figure.text(
        0.5,
        0.065,
        "25 full-sphere direction discs x 25 random points; disc radius 15 mm; "
        "error bars: 95% Student-t interval across discs (df=24).",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def _study_metadata(
    study_key: str,
    points: Sequence[RelationshipPoint],
    rows: Sequence[Mapping[str, object]],
    search: CaptureSearchConfig,
    geometry_hash: str,
    paths: CampaignPaths,
) -> dict[str, object]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "study_key": study_key,
        "status": "completed" if len(rows) == len(points) else "running",
        "updated_utc": _utc_now(),
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "point_count_requested": len(points),
        "point_count_completed": len(rows),
        "ordered_point_plan": [asdict(point) for point in points],
        "search_config": asdict(search),
        "zero_grid_ray_batch_size": ZERO_GRID_RAY_BATCH_SIZE,
        "phase_space": "full_sphere",
        "common_geometry_sha256": geometry_hash,
        "common_random_numbers": (
            "The same seeded full-sphere direction discs and uniform-area points are "
            "reused at every parameter point; only the named scan variable changes."
        ),
        "uncertainty": (
            f"Each point is the mean of {search.disc_count} direction-disc loading "
            f"estimates. Error bars are 95% Student-t intervals across those clusters "
            f"(df={search.disc_count - 1}), not across individual rays."
        ),
        "aggregate_csv": str(paths.aggregate_csv(study_key).resolve()),
        "relationship_plot": str(paths.relationship_plot(study_key).resolve()),
    }


def run_loading_relationship(
    study_key: str,
    points_to_run: Sequence[RelationshipPoint],
    *,
    search: CaptureSearchConfig,
    campaign_paths: CampaignPaths,
    worker_count: int,
    resume: bool,
    plot_only: bool = False,
    overall_offset: int = 0,
    overall_total: int = 67,
) -> list[dict[str, object]]:
    """Run one whole relationship before starting the next relationship."""

    if (
        search.disc_count != DEFAULT_DISC_COUNT
        or search.points_per_disc != DEFAULT_POINTS_PER_DISC
        or not np.isclose(search.disc_radius_m, DEFAULT_DISC_RADIUS_M)
        or search.seed != DEFAULT_SEED
        or search.include_center_point
    ):
        raise ValueError(
            "refined production loading requires the canonical seed-20260903 "
            "full-sphere 25x25 r=15 mm geometry"
        )
    discs, launch_points = generate_common_geometry(search)
    rows_geometry = geometry_rows(discs, launch_points)
    geometry_text = geometry_csv_text(rows_geometry)
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    rows: list[dict[str, object]] = []
    for local_index, point in enumerate(points_to_run, start=1):
        point_paths = campaign_paths.point_paths(point)
        print(
            f"[{_utc_now()}] [two-level campaign {overall_offset + local_index}/"
            f"{overall_total}] starting {point.study_key}/{point.slug}",
            flush=True,
        )
        if plot_only:
            if not point_paths.capture_summary_json.is_file():
                raise FileNotFoundError(
                    f"plot-only point summary missing: {point_paths.capture_summary_json}"
                )
            summary = json.loads(
                point_paths.capture_summary_json.read_text(encoding="utf-8")
            )
        else:
            complete = False
            if resume and point_paths.metadata_json.is_file():
                prior = json.loads(point_paths.metadata_json.read_text(encoding="utf-8"))
                complete = (
                    prior.get("status") == "completed"
                    and int(prior.get("completed_sample_count", -1))
                    == search.disc_count * search.points_per_disc
                    and point_paths.final_samples_csv.is_file()
                )
            summary = run_relationship_point(
                point,
                search=search,
                discs=discs,
                points=launch_points,
                geometry_text=geometry_text,
                geometry_hash=geometry_hash,
                paths=point_paths,
                worker_count=worker_count,
                resume=resume,
                analyze_only=complete,
            )
        if summary.get("geometry_sha256") != geometry_hash:
            raise ValueError(f"point {point.slug} did not use the common geometry")
        row = _point_aggregate_row(point, search, point_paths, summary)
        rows.append(row)
        _atomic_write_csv(
            campaign_paths.aggregate_csv(study_key), rows, AGGREGATE_FIELDNAMES
        )
        plot_loading_relationship(
            rows, study_key, campaign_paths.relationship_plot(study_key)
        )
        _atomic_write_json(
            campaign_paths.study_metadata_json(study_key),
            _study_metadata(
                study_key,
                points_to_run,
                rows,
                search,
                geometry_hash,
                campaign_paths,
            ),
        )
        print(
            f"[{_utc_now()}] [two-level campaign {overall_offset + local_index}/"
            f"{overall_total}] completed {point.study_key}/{point.slug}; "
            f"R={float(row['loading_rate_mean_atoms_per_s']):.6g} atoms/s",
            flush=True,
        )
    return rows


def _new_campaign_metadata(
    paths: CampaignPaths,
    search: CaptureSearchConfig,
    geometry_hash: str,
) -> dict[str, object]:
    point_groups = requested_points()
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_name": CAMPAIGN_NAME,
        "campaign_signature_sha256": _campaign_signature(search, geometry_hash),
        "status": "running",
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "state_model": "effective two-level Rb-87 D2 atom; no repumper",
        "execution_order": list(STUDY_ORDER),
        "stage_status": {key: "pending" for key in STUDY_ORDER},
        "requested_grids": {
            "s0": list(RAW_SATURATION_VALUES),
            "s_eff": list(EFFECTIVE_SATURATION_VALUES),
            "detuning_delta_over_gamma": list(DETUNING_N_VALUES),
        },
        "loading_point_count": sum(len(values) for values in point_groups.values()),
        "capture_threshold_search_count": sum(
            len(values) * search.disc_count * search.points_per_disc
            for values in point_groups.values()
        ),
        "search_config": asdict(search),
        "zero_grid_ray_batch_size": ZERO_GRID_RAY_BATCH_SIZE,
        "phase_space": "full_sphere",
        "common_geometry_sha256": geometry_hash,
        "cooling_reference_power_w_per_beam": DEFAULT_COOLING_POWER_W_PER_BEAM,
        "temperature_stage": "omitted: deterministic mot_simple has no recoil diffusion",
        "statistics_directory": str(paths.statistics.resolve()),
        "figures_directory": str(paths.figures.resolve()),
    }


def run_campaign(
    *,
    paths: CampaignPaths | None = None,
    search: CaptureSearchConfig | None = None,
    worker_count: int = DEFAULT_WORKER_COUNT,
    resume: bool = True,
    selected_studies: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run selected relationships in canonical order."""

    campaign_paths = paths or default_campaign_paths()
    campaign_search = search or default_search_config()
    campaign_paths.statistics.mkdir(parents=True, exist_ok=True)
    campaign_paths.figures.mkdir(parents=True, exist_ok=True)
    discs, launch_points = generate_common_geometry(campaign_search)
    geometry_text = geometry_csv_text(geometry_rows(discs, launch_points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    expected_signature = _campaign_signature(campaign_search, geometry_hash)
    if resume and campaign_paths.metadata_json.is_file():
        metadata = json.loads(campaign_paths.metadata_json.read_text(encoding="utf-8"))
        if metadata.get("campaign_signature_sha256") != expected_signature:
            raise ValueError("campaign resume signature mismatch")
        if not campaign_paths.geometry_csv.is_file() or _sha256_file(
            campaign_paths.geometry_csv
        ) != geometry_hash:
            raise ValueError("campaign launch geometry is missing or changed")
    else:
        metadata = _new_campaign_metadata(campaign_paths, campaign_search, geometry_hash)
        _atomic_write_text(campaign_paths.geometry_csv, geometry_text)
        _atomic_write_json(campaign_paths.metadata_json, metadata)

    selected = set(STUDY_ORDER if selected_studies is None else selected_studies)
    unknown = selected.difference(STUDY_ORDER)
    if unknown:
        raise ValueError(f"unknown relationship studies: {sorted(unknown)}")
    if not selected:
        raise ValueError("at least one relationship study is required")
    groups = requested_points()
    offsets = {
        RAW_STUDY_KEY: 0,
        EFFECTIVE_STUDY_KEY: len(groups[RAW_STUDY_KEY]),
        DETUNING_STUDY_KEY: len(groups[RAW_STUDY_KEY]) + len(groups[EFFECTIVE_STUDY_KEY]),
    }
    total = sum(len(values) for values in groups.values())
    for study_key in STUDY_ORDER:
        if study_key not in selected:
            continue
        metadata["stage_status"][study_key] = "running"
        metadata["current_stage"] = study_key
        metadata["updated_utc"] = _utc_now()
        _atomic_write_json(campaign_paths.metadata_json, metadata)
        try:
            run_loading_relationship(
                study_key,
                groups[study_key],
                search=campaign_search,
                campaign_paths=campaign_paths,
                worker_count=worker_count,
                resume=resume,
                overall_offset=offsets[study_key],
                overall_total=total,
            )
        except BaseException as exc:
            metadata["stage_status"][study_key] = "failed"
            metadata["status"] = "failed"
            metadata["last_error"] = f"{type(exc).__name__}: {exc}"
            metadata["updated_utc"] = _utc_now()
            _atomic_write_json(campaign_paths.metadata_json, metadata)
            raise
        metadata["stage_status"][study_key] = "completed"
        metadata["updated_utc"] = _utc_now()
        metadata["status"] = (
            "completed"
            if all(metadata["stage_status"].get(key) == "completed" for key in STUDY_ORDER)
            else "running"
        )
        _atomic_write_json(campaign_paths.metadata_json, metadata)
    return metadata


def regenerate_plots(*, paths: CampaignPaths | None = None) -> dict[str, str]:
    campaign_paths = paths or default_campaign_paths()
    outputs: dict[str, str] = {}
    for study_key in STUDY_ORDER:
        rows = _read_aggregate(campaign_paths.aggregate_csv(study_key))
        expected = len(requested_points()[study_key])
        if len(rows) != expected:
            raise ValueError(
                f"{study_key} aggregate has {len(rows)} of {expected} requested points"
            )
        result = plot_loading_relationship(
            rows, study_key, campaign_paths.relationship_plot(study_key)
        )
        outputs[study_key] = str(result.resolve())
    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated refined two-level MOT loading relationships"
    )
    parser.add_argument(
        "--study",
        choices=("campaign", "raw-s", "effective-s", "detuning", "plots"),
        default="campaign",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    paths = default_campaign_paths(args.project_root)
    if args.study == "plots":
        result: Mapping[str, object] = regenerate_plots(paths=paths)
    else:
        selection = {
            "raw-s": (RAW_STUDY_KEY,),
            "effective-s": (EFFECTIVE_STUDY_KEY,),
            "detuning": (DETUNING_STUDY_KEY,),
        }
        result = run_campaign(
            paths=paths,
            worker_count=args.workers,
            resume=args.resume,
            selected_studies=None if args.study == "campaign" else selection[args.study],
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_NAME",
    "DETUNING_N_VALUES",
    "DETUNING_STUDY_KEY",
    "EFFECTIVE_SATURATION_VALUES",
    "EFFECTIVE_STUDY_KEY",
    "RAW_SATURATION_VALUES",
    "RAW_STUDY_KEY",
    "CampaignPaths",
    "RelationshipPoint",
    "build_relationship_points",
    "default_campaign_paths",
    "default_search_config",
    "detuning_reduction_denominator",
    "effective_saturation_from_s0",
    "generate_common_geometry",
    "on_resonance_saturation_parameter",
    "plot_loading_relationship",
    "regenerate_plots",
    "requested_points",
    "run_campaign",
    "run_loading_relationship",
    "run_relationship_point",
    "saturation_power_w_per_beam",
]
