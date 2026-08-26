"""Resumable 27 mW capture-cross-section and loading study for ``mot_simple``.

The production defaults in this module encode one deliberately narrow study:
the validated deterministic effective two-level MOT is sampled on 50 random
incident-direction discs with 50 independent uniform-area launch points per
disc, with only the cooling-beam power changed from 20 mW to 27 mW per beam.
All trajectory and apparatus settings otherwise come from the authoritative
``mot_simple`` defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..configuration import PMOTSimulationConfig
from ..mot.configuration import AntiHelmholtzCoilConfig
from ..mot.magnetic_fields import anti_helmholtz_axial_gradient_t_per_m
from ..mot.magnetic_fields import default_anti_helmholtz_config
from .configuration import default_simple_mot_apparatus
from .configuration import default_simple_mot_config
from .configuration import simple_mot_paths
from .configuration import SimpleMOTConfig
from .loading import LOADING_RATE_PREFACTOR
from .loading import THERMAL_SCALE_M2_PER_S2
from .loading import calculate_loading_rate_from_spectrum
from .sampling import CaptureSearchConfig
from .sampling import CaptureVelocitySample
from .sampling import DiscSample
from .sampling import PointSample
from .sampling import find_capture_velocity
from .sampling import load_capture_velocity_samples
from .sampling import sample_disc_points
from .sampling import sample_incident_disc
from .sampling import velocity_grid_from_samples
from .simulation import build_simple_mot_beams


STUDY_NAME = "loading_rate_27mW_50_discs_50_points"
COOLING_POWER_W_PER_BEAM = 27.0e-3
DISC_COUNT = 50
POINTS_PER_DISC = 50
RANDOM_SEED = 0
DEFAULT_WORKER_COUNT = 20
CHECKPOINT_EVERY = 25
PROGRESS_EVERY = 10
METADATA_SCHEMA_VERSION = 1

TRAPPED_TERMINATION_REASONS = {"two_core_entries", "bounded_core_residence"}

SAMPLE_FIELDNAMES = [
    "disc_index",
    "point_index",
    "theta_rad",
    "phi_rad",
    "theta_prime_rad",
    "s_m",
    "radial_distance_m",
    "capture_velocity_m_per_s",
    "velocity_resolution_m_per_s",
    "trapped_velocity_lower_m_per_s",
    "untrapped_velocity_upper_m_per_s",
    "lower_classification",
    "upper_classification",
    "lower_entered_trap_core",
    "upper_entered_trap_core",
    "lower_core_entry_count",
    "upper_core_entry_count",
    "x0_m",
    "y0_m",
    "z0_m",
    "vx_hat",
    "vy_hat",
    "vz_hat",
]

GEOMETRY_FIELDNAMES = [
    "disc_index",
    "point_index",
    "theta_rad",
    "phi_rad",
    "theta_prime_rad",
    "s_m",
    "radial_distance_m",
    "x0_m",
    "y0_m",
    "z0_m",
    "vx_hat",
    "vy_hat",
    "vz_hat",
    "disc_center_x_m",
    "disc_center_y_m",
    "disc_center_z_m",
    "basis_u_x",
    "basis_u_y",
    "basis_u_z",
    "basis_v_x",
    "basis_v_y",
    "basis_v_z",
]


@dataclass(frozen=True, slots=True)
class StudyPaths:
    """Filesystem products for one power-loading study."""

    statistics: Path
    figures: Path

    @property
    def metadata_json(self) -> Path:
        return self.statistics / "run_metadata.json"

    @property
    def geometry_csv(self) -> Path:
        return self.statistics / "launch_geometry.csv"

    @property
    def partial_samples_csv(self) -> Path:
        return self.statistics / "capture_velocity_partial_samples.csv"

    @property
    def final_samples_csv(self) -> Path:
        return self.statistics / "capture_velocity_samples.csv"

    @property
    def capture_summary_json(self) -> Path:
        return self.statistics / "capture_velocity_summary.json"

    @property
    def spectrum_csv(self) -> Path:
        return self.statistics / "capture_velocity_spectrum.csv"

    @property
    def loading_by_disc_csv(self) -> Path:
        return self.statistics / "loading_rate_by_disc.csv"

    @property
    def loading_json(self) -> Path:
        return self.statistics / "loading_rate_result.json"

    @property
    def cross_section_png(self) -> Path:
        return self.figures / "capture_cross_section_vs_velocity.png"

    @property
    def impact_parameter_png(self) -> Path:
        return self.figures / "capture_velocity_vs_impact_parameter.png"

    @property
    def loading_by_disc_png(self) -> Path:
        return self.figures / "loading_rate_by_disc.png"


def default_study_paths(root: Path | None = None) -> StudyPaths:
    """Return the dedicated output directories for the production study."""

    paths = simple_mot_paths(root)
    return StudyPaths(
        statistics=paths["outputs_statistics_simple_mot"] / STUDY_NAME,
        figures=paths["outputs_figures_simple_mot"] / STUDY_NAME,
    )


def default_study_search_config() -> CaptureSearchConfig:
    """Return exact production sampling defaults: 50 discs x 50 points."""

    return replace(
        CaptureSearchConfig(),
        disc_count=DISC_COUNT,
        points_per_disc=POINTS_PER_DISC,
        seed=RANDOM_SEED,
        save_every=CHECKPOINT_EVERY,
        include_center_point=False,
    )


def build_27mw_apparatus() -> PMOTSimulationConfig:
    """Return the default apparatus with only cooling power changed to 27 mW."""

    apparatus = default_simple_mot_apparatus()
    return replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=COOLING_POWER_W_PER_BEAM,
        ),
    )


def effective_saturation_metrics(
    apparatus: PMOTSimulationConfig | None = None,
    simple_config: SimpleMOTConfig | None = None,
) -> dict[str, float | str]:
    """Return the requested on-axis, single-beam effective saturation metric."""

    apparatus = apparatus or build_27mw_apparatus()
    simple = simple_config or default_simple_mot_config()
    radius_m = apparatus.cooling.beam_radius_m
    peak_intensity = 2.0 * apparatus.cooling.power_w_per_beam / (pi * radius_m**2)
    on_resonance_saturation = peak_intensity / simple.saturation_intensity_w_per_m2
    detuning_factor = 1.0 + (
        2.0 * simple.cooling_detuning_hz / simple.linewidth_hz
    ) ** 2
    return {
        "beam_center_peak_intensity_w_per_m2": peak_intensity,
        "beam_center_on_resonance_saturation_parameter": on_resonance_saturation,
        "detuning_reduction_denominator": detuning_factor,
        "beam_center_effective_saturation_parameter": on_resonance_saturation / detuning_factor,
        "definition": "s_eff = (I0/I_sat) / [1 + (2*Delta/Gamma)^2]",
    }


def generate_study_geometry(
    search: CaptureSearchConfig,
) -> tuple[list[DiscSample], list[PointSample]]:
    """Generate all seeded incident discs and independent uniform-area points."""

    if search.disc_count <= 0 or search.points_per_disc <= 0:
        raise ValueError("disc_count and points_per_disc must be positive")
    if search.include_center_point:
        raise ValueError(
            "production power-loading geometry requires include_center_point=False "
            "so every point is a random uniform-area draw"
        )
    rng = np.random.default_rng(search.seed)
    discs: list[DiscSample] = []
    points: list[PointSample] = []
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc(disc_index, search.radial_distance_m, rng)
        discs.append(disc)
        points.extend(sample_disc_points(
            disc,
            search.points_per_disc,
            search.disc_radius_m,
            search.include_center_point,
            rng,
        ))
    return discs, points


def geometry_rows(
    discs: Sequence[DiscSample],
    points: Sequence[PointSample],
) -> list[dict[str, int | float]]:
    """Return the exact point geometry in stable, CSV-friendly form."""

    disc_map = {disc.disc_index: disc for disc in discs}
    rows: list[dict[str, int | float]] = []
    for point in sorted(points, key=lambda value: (value.disc_index, value.point_index)):
        disc = disc_map[point.disc_index]
        rows.append({
            "disc_index": point.disc_index,
            "point_index": point.point_index,
            "theta_rad": point.theta_rad,
            "phi_rad": point.phi_rad,
            "theta_prime_rad": point.theta_prime_rad,
            "s_m": point.s_m,
            "radial_distance_m": point.radial_distance_m,
            "x0_m": point.initial_position_m[0],
            "y0_m": point.initial_position_m[1],
            "z0_m": point.initial_position_m[2],
            "vx_hat": point.incident_unit_vector[0],
            "vy_hat": point.incident_unit_vector[1],
            "vz_hat": point.incident_unit_vector[2],
            "disc_center_x_m": disc.center_position_m[0],
            "disc_center_y_m": disc.center_position_m[1],
            "disc_center_z_m": disc.center_position_m[2],
            "basis_u_x": disc.basis_u[0],
            "basis_u_y": disc.basis_u[1],
            "basis_u_z": disc.basis_u[2],
            "basis_v_x": disc.basis_v[0],
            "basis_v_y": disc.basis_v[1],
            "basis_v_z": disc.basis_v[2],
        })
    return rows


def _csv_text(
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def geometry_csv_text(rows: Sequence[dict[str, object]]) -> str:
    """Serialize geometry canonically so its SHA-256 is reproducible."""

    return _csv_text(rows, GEOMETRY_FIELDNAMES)


def geometry_sha256(rows: Sequence[dict[str, object]]) -> str:
    return hashlib.sha256(geometry_csv_text(rows).encode("utf-8")).hexdigest()


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8", newline="")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, _json_text(payload))


def _atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    _atomic_write_text(path, _csv_text(rows, fieldnames))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physics_source_hashes() -> dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    names = ("power_loading_study.py", "sampling.py", "simulation.py")
    return {name: _sha256_file(source_directory / name) for name in names}


def _flatten_differences(
    baseline: object,
    candidate: object,
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(set(baseline) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline or key not in candidate:
                differences.append(path)
            else:
                differences.extend(_flatten_differences(baseline[key], candidate[key], path))
        return differences
    if isinstance(baseline, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if len(baseline) != len(candidate):
            differences.append(prefix)
            return differences
        for index, (left, right) in enumerate(zip(baseline, candidate)):
            differences.extend(_flatten_differences(left, right, f"{prefix}[{index}]"))
        return differences
    if baseline != candidate:
        differences.append(prefix)
    return differences


def configuration_invariance_audit(
    apparatus: PMOTSimulationConfig,
    simple: SimpleMOTConfig,
    coil: AntiHelmholtzCoilConfig,
    search: CaptureSearchConfig,
) -> dict[str, object]:
    """Prove which fields differ from the repository defaults."""

    apparatus_changes = _flatten_differences(
        asdict(default_simple_mot_apparatus()), asdict(apparatus),
    )
    simple_changes = _flatten_differences(
        asdict(default_simple_mot_config()), asdict(simple),
    )
    coil_changes = _flatten_differences(
        asdict(default_anti_helmholtz_config()), asdict(coil),
    )
    search_changes = _flatten_differences(
        asdict(CaptureSearchConfig()), asdict(search),
    )
    expected_search_changes = {"disc_count", "points_per_disc"}
    if CaptureSearchConfig().include_center_point:
        expected_search_changes.add("include_center_point")
    return {
        "apparatus_changed_fields": apparatus_changes,
        "apparatus_only_cooling_power_changed": apparatus_changes == ["cooling.power_w_per_beam"],
        "simple_config_changed_fields": simple_changes,
        "simple_config_matches_default": not simple_changes,
        "coil_config_changed_fields": coil_changes,
        "coil_config_matches_default": not coil_changes,
        "search_config_changed_fields": search_changes,
        "search_only_requested_counts_changed": set(search_changes) == expected_search_changes,
    }


def study_signature_payload(
    apparatus: PMOTSimulationConfig,
    simple: SimpleMOTConfig,
    coil: AntiHelmholtzCoilConfig,
    search: CaptureSearchConfig,
    geometry_hash: str,
    *,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return every scientific input that must match before resuming."""

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "apparatus_config": asdict(apparatus),
        "simple_mot_config": asdict(simple),
        "coil_config": asdict(coil),
        "capture_search_config": asdict(search),
        "geometry_sha256": geometry_hash,
        "physics_source_sha256": source_hashes or _physics_source_hashes(),
    }


def study_signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sample_to_row(sample: CaptureVelocitySample) -> dict[str, object]:
    return {
        "disc_index": sample.disc_index,
        "point_index": sample.point_index,
        "theta_rad": sample.theta_rad,
        "phi_rad": sample.phi_rad,
        "theta_prime_rad": sample.theta_prime_rad,
        "s_m": sample.s_m,
        "radial_distance_m": sample.radial_distance_m,
        "capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "velocity_resolution_m_per_s": sample.velocity_resolution_m_per_s,
        "trapped_velocity_lower_m_per_s": sample.trapped_velocity_lower_m_per_s,
        "untrapped_velocity_upper_m_per_s": sample.untrapped_velocity_upper_m_per_s,
        "lower_classification": sample.lower_classification,
        "upper_classification": sample.upper_classification,
        "lower_entered_trap_core": sample.lower_entered_trap_core,
        "upper_entered_trap_core": sample.upper_entered_trap_core,
        "lower_core_entry_count": sample.lower_core_entry_count,
        "upper_core_entry_count": sample.upper_core_entry_count,
        "x0_m": sample.initial_position_m[0],
        "y0_m": sample.initial_position_m[1],
        "z0_m": sample.initial_position_m[2],
        "vx_hat": sample.incident_unit_vector[0],
        "vy_hat": sample.incident_unit_vector[1],
        "vz_hat": sample.incident_unit_vector[2],
    }


def _sorted_samples(samples: Iterable[CaptureVelocitySample]) -> list[CaptureVelocitySample]:
    return sorted(samples, key=lambda sample: (sample.disc_index, sample.point_index))


def save_samples_atomic(path: Path, samples: Iterable[CaptureVelocitySample]) -> None:
    rows = [_sample_to_row(sample) for sample in _sorted_samples(samples)]
    _atomic_write_csv(path, rows, SAMPLE_FIELDNAMES)


def _validate_sample_against_point(
    sample: CaptureVelocitySample,
    point: PointSample,
) -> None:
    if (sample.disc_index, sample.point_index) != (point.disc_index, point.point_index):
        raise ValueError("checkpoint sample key does not match seeded launch geometry")
    scalar_pairs = (
        (sample.theta_rad, point.theta_rad),
        (sample.phi_rad, point.phi_rad),
        (sample.theta_prime_rad, point.theta_prime_rad),
        (sample.s_m, point.s_m),
        (sample.radial_distance_m, point.radial_distance_m),
    )
    if any(not np.isclose(left, right, rtol=0.0, atol=2.0e-14) for left, right in scalar_pairs):
        raise ValueError("checkpoint sample geometry differs from seeded launch geometry")
    if not np.allclose(sample.initial_position_m, point.initial_position_m, rtol=0.0, atol=2.0e-14):
        raise ValueError("checkpoint initial position differs from seeded launch geometry")
    if not np.allclose(sample.incident_unit_vector, point.incident_unit_vector, rtol=0.0, atol=2.0e-14):
        raise ValueError("checkpoint incident direction differs from seeded launch geometry")
    numeric = np.asarray([
        sample.capture_velocity_m_per_s,
        sample.velocity_resolution_m_per_s,
        sample.trapped_velocity_lower_m_per_s,
        sample.untrapped_velocity_upper_m_per_s,
    ])
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("checkpoint contains an invalid capture-speed bracket")
    if sample.untrapped_velocity_upper_m_per_s < sample.trapped_velocity_lower_m_per_s:
        raise ValueError("checkpoint capture-speed bracket is reversed")
    if (
        sample.capture_velocity_m_per_s > 0.0
        and sample.lower_classification not in TRAPPED_TERMINATION_REASONS
    ):
        raise ValueError("checkpoint lower bracket is not classified as trapped")
    if sample.upper_classification in TRAPPED_TERMINATION_REASONS:
        raise ValueError("checkpoint upper bracket is still classified as trapped")


def validate_checkpoint_samples(
    samples: Sequence[CaptureVelocitySample],
    points: Sequence[PointSample],
) -> dict[tuple[int, int], CaptureVelocitySample]:
    """Validate checkpoint uniqueness, bounds, and exact seeded geometry."""

    point_map = {(point.disc_index, point.point_index): point for point in points}
    result: dict[tuple[int, int], CaptureVelocitySample] = {}
    for sample in samples:
        key = (sample.disc_index, sample.point_index)
        if key in result:
            raise ValueError(f"duplicate checkpoint sample {key}")
        if key not in point_map:
            raise ValueError(f"checkpoint contains unknown sample {key}")
        _validate_sample_against_point(sample, point_map[key])
        result[key] = sample
    return result


def _student_t_critical_95(sample_count: int) -> float:
    if sample_count <= 1:
        return 0.0
    return float(student_t.ppf(0.975, df=sample_count - 1))


def calculate_clustered_cross_section(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    velocity_grid_m_per_s: np.ndarray | None = None,
) -> list[dict[str, int | float]]:
    """Calculate direction-clustered cross section and a clipped t interval."""

    if not samples:
        raise ValueError("at least one capture-velocity sample is required")
    velocity = (
        np.asarray(velocity_grid_m_per_s, dtype=float)
        if velocity_grid_m_per_s is not None
        else velocity_grid_from_samples(
            list(samples),
            search.analysis_velocity_step_m_per_s,
            search.analysis_velocity_min_m_per_s,
            search.analysis_velocity_max_m_per_s,
        )
    )
    if velocity.ndim != 1 or len(velocity) < 2 or np.any(np.diff(velocity) <= 0.0):
        raise ValueError("velocity grid must be a strictly increasing one-dimensional array")

    grouped: dict[int, list[float]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample.capture_velocity_m_per_s)
    disc_ids = sorted(grouped)
    disc_count = len(disc_ids)
    t_critical = _student_t_critical_95(disc_count)
    area = pi * search.disc_radius_m**2
    thresholds = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    rows: list[dict[str, int | float]] = []
    for speed in velocity:
        disc_fractions = np.asarray([
            np.mean(np.asarray(grouped[disc_index]) >= speed - 1.0e-12)
            for disc_index in disc_ids
        ])
        disc_cross_sections = area * disc_fractions
        mean = float(np.mean(disc_cross_sections))
        sample_std = float(np.std(disc_cross_sections, ddof=1)) if disc_count > 1 else 0.0
        sem = sample_std / np.sqrt(disc_count) if disc_count else 0.0
        half_width = t_critical * sem
        captured_count = int(np.count_nonzero(thresholds >= speed - 1.0e-12))
        rows.append({
            "velocity_m_per_s": float(speed),
            "captured_count": captured_count,
            "launched_count": len(samples),
            "capture_fraction": mean / area,
            "capture_cross_section_m2": mean,
            "capture_cross_section_sample_std_m2": sample_std,
            "capture_cross_section_disc_cluster_sem_m2": sem,
            "capture_cross_section_t95_lower_m2": max(0.0, mean - half_width),
            "capture_cross_section_t95_upper_m2": min(area, mean + half_width),
            "disc_count": disc_count,
            "student_t_critical_95": t_critical,
        })
    return rows


SPECTRUM_FIELDNAMES = [
    "velocity_m_per_s",
    "captured_count",
    "launched_count",
    "capture_fraction",
    "capture_cross_section_m2",
    "capture_cross_section_sample_std_m2",
    "capture_cross_section_disc_cluster_sem_m2",
    "capture_cross_section_t95_lower_m2",
    "capture_cross_section_t95_upper_m2",
    "disc_count",
    "student_t_critical_95",
]


def calculate_disc_clustered_loading(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    spectrum_rows: Sequence[dict[str, int | float]],
) -> tuple[list[dict[str, int | float]], dict[str, int | float | str]]:
    """Integrate one loading rate per direction disc, then summarize clusters."""

    if not spectrum_rows:
        raise ValueError("capture-cross-section spectrum must not be empty")
    velocity = np.asarray([row["velocity_m_per_s"] for row in spectrum_rows], dtype=float)
    area = pi * search.disc_radius_m**2
    grouped: dict[int, list[float]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample.capture_velocity_m_per_s)
    by_disc: list[dict[str, int | float]] = []
    for disc_index in sorted(grouped):
        thresholds = np.asarray(grouped[disc_index], dtype=float)
        sigma = area * np.asarray([
            np.mean(thresholds >= speed - 1.0e-12) for speed in velocity
        ])
        result = calculate_loading_rate_from_spectrum(velocity, sigma)
        by_disc.append({
            "disc_index": disc_index,
            "point_count": len(thresholds),
            "loading_integral_m6_per_s4": result.integral_value_m5_per_s4,
            "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
        })

    rates = np.asarray([row["loading_rate_atoms_per_s"] for row in by_disc], dtype=float)
    integrals = np.asarray([row["loading_integral_m6_per_s4"] for row in by_disc], dtype=float)
    mean_sigma = np.asarray([row["capture_cross_section_m2"] for row in spectrum_rows], dtype=float)
    mean_spectrum_result = calculate_loading_rate_from_spectrum(velocity, mean_sigma)
    disc_count = len(by_disc)
    sample_std = float(np.std(rates, ddof=1)) if disc_count > 1 else 0.0
    sem = sample_std / np.sqrt(disc_count) if disc_count else 0.0
    t_critical = _student_t_critical_95(disc_count)
    half_width = t_critical * sem
    mean_rate = float(np.mean(rates))
    summary: dict[str, int | float | str] = {
        "loading_rate_mean_atoms_per_s": mean_rate,
        "loading_rate_from_mean_spectrum_atoms_per_s": mean_spectrum_result.loading_rate_atoms_per_s,
        "loading_rate_sample_std_atoms_per_s": sample_std,
        "loading_rate_disc_cluster_sem_atoms_per_s": sem,
        "loading_rate_t95_lower_atoms_per_s": max(0.0, mean_rate - half_width),
        "loading_rate_t95_upper_atoms_per_s": mean_rate + half_width,
        "student_t_critical_95": t_critical,
        "confidence_level": 0.95,
        "disc_count": disc_count,
        "point_count": len(samples),
        "loading_integral_mean_m6_per_s4": float(np.mean(integrals)),
        "loading_integral_from_mean_spectrum_m6_per_s4": mean_spectrum_result.integral_value_m5_per_s4,
        "velocity_min_m_per_s": float(np.min(velocity)),
        "velocity_max_m_per_s": float(np.max(velocity)),
        "velocity_grid_sample_count": len(velocity),
        "quadrature_method": "trapezoid",
        "formula": (
            "R = 9.1196e5 * integral sigma_capture(v) * v^3 * "
            "exp[-v^2/(5.667e4)] dv"
        ),
        "raw_integral_units": "m^6/s^4",
        "loading_rate_prefactor": LOADING_RATE_PREFACTOR,
        "thermal_scale_m2_per_s2": THERMAL_SCALE_M2_PER_S2,
        "primary_uncertainty": "disc-clustered standard error across incident directions",
    }
    return by_disc, summary


LOADING_BY_DISC_FIELDNAMES = [
    "disc_index",
    "point_count",
    "loading_integral_m6_per_s4",
    "loading_rate_atoms_per_s",
]


def plot_clustered_cross_section(
    spectrum_rows: Sequence[dict[str, int | float]],
    path: Path,
    *,
    power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
) -> Path:
    """Plot the mean direction-averaged cross section and clustered t interval."""

    velocity = np.asarray([row["velocity_m_per_s"] for row in spectrum_rows], dtype=float)
    mean_mm2 = 1.0e6 * np.asarray([
        row["capture_cross_section_m2"] for row in spectrum_rows
    ], dtype=float)
    lower_mm2 = 1.0e6 * np.asarray([
        row["capture_cross_section_t95_lower_m2"] for row in spectrum_rows
    ], dtype=float)
    upper_mm2 = 1.0e6 * np.asarray([
        row["capture_cross_section_t95_upper_m2"] for row in spectrum_rows
    ], dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.fill_between(
        velocity, lower_mm2, upper_mm2,
        color="#99c8c2", alpha=0.5, linewidth=0.0,
        label="95% t interval across direction discs",
    )
    axis.plot(velocity, mean_mm2, color="#0f766e", linewidth=2.2, label="Mean cross section")
    axis.set_title(f"Two-Level MOT Capture Cross Section ({1e3 * power_w_per_beam:g} mW/beam)")
    axis.set_xlabel("Launch speed [m/s]")
    axis.set_ylabel(r"Capture cross section [mm$^2$]")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_capture_velocity_vs_impact_parameter(
    samples: Sequence[CaptureVelocitySample],
    path: Path,
    *,
    power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
) -> Path:
    """Plot all capture thresholds against sampled disc impact parameter."""

    s_mm = 1.0e3 * np.asarray([sample.s_m for sample in samples], dtype=float)
    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.scatter(s_mm, capture, s=12, alpha=0.25, color="#0f766e", edgecolors="none")
    if len(samples) >= 10:
        edges = np.linspace(0.0, float(np.max(s_mm)), 13)
        centers: list[float] = []
        means: list[float] = []
        sems: list[float] = []
        for index in range(len(edges) - 1):
            selected = (s_mm >= edges[index]) & (
                s_mm <= edges[index + 1] if index == len(edges) - 2 else s_mm < edges[index + 1]
            )
            values = capture[selected]
            if len(values) == 0:
                continue
            centers.append(0.5 * (edges[index] + edges[index + 1]))
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        axis.errorbar(
            centers, means, yerr=sems, color="#9f4a13", marker="o",
            markersize=4, linewidth=1.7, capsize=2,
            label="Impact-parameter-bin mean +/- SEM",
        )
        axis.legend(frameon=False)
    axis.set_title(f"Capture Velocity vs Impact Parameter ({1e3 * power_w_per_beam:g} mW/beam)")
    axis.set_xlabel("Impact parameter [mm]")
    axis.set_ylabel("Capture velocity [m/s]")
    axis.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_loading_rate_by_disc(
    by_disc: Sequence[dict[str, int | float]],
    summary: dict[str, int | float | str],
    path: Path,
    *,
    power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
) -> Path:
    """Plot direction-disc loading estimates with the clustered mean interval."""

    disc_number = 1 + np.asarray([row["disc_index"] for row in by_disc], dtype=int)
    rate = np.asarray([row["loading_rate_atoms_per_s"] for row in by_disc], dtype=float)
    mean = float(summary["loading_rate_mean_atoms_per_s"])
    lower = float(summary["loading_rate_t95_lower_atoms_per_s"])
    upper = float(summary["loading_rate_t95_upper_atoms_per_s"])
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.axhspan(
        lower, upper, color="#99c8c2", alpha=0.5, linewidth=0.0,
        label="95% t interval for the mean",
    )
    axis.axhline(mean, color="#0f766e", linewidth=2.2, label="Mean loading rate")
    axis.scatter(
        disc_number, rate, color="#9f4a13", marker="o", s=22, alpha=0.75,
        label="Direction-disc estimates",
    )
    axis.set_title(f"Two-Level MOT Loading Rate by Direction ({1e3 * power_w_per_beam:g} mW/beam)")
    axis.set_xlabel("Random incident-direction disc")
    axis.set_ylabel("Loading rate [atoms/s]")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _git_provenance(project_root: Path) -> dict[str, object]:
    """Return bounded, best-effort Git provenance without affecting the run."""

    def command(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    head = command("rev-parse", "HEAD")
    branch = command("branch", "--show-current")
    status = command("status", "--short", "--untracked-files=all")
    lines = [] if not status else status.splitlines()
    return {
        "head_commit": head,
        "branch": branch,
        "is_dirty": bool(lines),
        "dirty_path_count": len(lines),
        "dirty_status_lines": lines[:100],
        "dirty_status_truncated": len(lines) > 100,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_manifest(paths: StudyPaths) -> dict[str, str]:
    return {
        "statistics_directory": str(paths.statistics.resolve()),
        "figures_directory": str(paths.figures.resolve()),
        "metadata_json": str(paths.metadata_json.resolve()),
        "geometry_csv": str(paths.geometry_csv.resolve()),
        "partial_samples_csv": str(paths.partial_samples_csv.resolve()),
        "final_samples_csv": str(paths.final_samples_csv.resolve()),
        "capture_summary_json": str(paths.capture_summary_json.resolve()),
        "spectrum_csv": str(paths.spectrum_csv.resolve()),
        "loading_by_disc_csv": str(paths.loading_by_disc_csv.resolve()),
        "loading_json": str(paths.loading_json.resolve()),
        "cross_section_plot": str(paths.cross_section_png.resolve()),
        "impact_parameter_plot": str(paths.impact_parameter_png.resolve()),
        "loading_rate_by_disc_plot": str(paths.loading_by_disc_png.resolve()),
    }


def build_run_metadata(
    apparatus: PMOTSimulationConfig,
    simple: SimpleMOTConfig,
    coil: AntiHelmholtzCoilConfig,
    search: CaptureSearchConfig,
    signature_payload: dict[str, object],
    signature: str,
    paths: StudyPaths,
    *,
    worker_count: int,
    status: str,
    completed_sample_count: int,
    started_utc: str | None = None,
    elapsed_wall_time_s: float = 0.0,
    eta_s: float | None = None,
) -> dict[str, object]:
    """Build an auditable metadata snapshot for running or completed work."""

    beams = build_simple_mot_beams(apparatus, simple)
    powers = [beam.intensity_beam.power_w for beam in beams]
    axial_gradient = anti_helmholtz_axial_gradient_t_per_m(
        coil.radius_m, coil.turns_per_coil, coil.current_a,
    )
    expected_count = search.disc_count * search.points_per_disc
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "status": status,
        "started_utc": started_utc or _utc_now(),
        "updated_utc": _utc_now(),
        "completed_sample_count": completed_sample_count,
        "expected_sample_count": expected_count,
        "completion_fraction": completed_sample_count / expected_count,
        "elapsed_wall_time_s": elapsed_wall_time_s,
        "eta_s": eta_s,
        "run_signature_sha256": signature,
        "signature_payload": signature_payload,
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "state_model": "effective two-level Rb-87 D2 atom; no multilevel state engine",
        "capture_dynamics": "deterministic RK4 mean force with gravity; recoil diffusion absent",
        "trapped_definition": (
            f"inside the {1e3 * search.trap_core_radius_m:g} mm-radius core continuously "
            f"for {1e3 * search.bounded_core_residence_s:g} ms OR "
            f"{search.required_core_entries} core entries with an intervening exit"
        ),
        "sampling_method": (
            "disc normals uniform in solid angle in one symmetry octant; each disc is "
            "perpendicular to its incident direction; independent points uniform in area "
            "using s=R*sqrt(U); every launch velocity is parallel to the disc normal"
        ),
        "cross_section_method": (
            "for each speed and direction disc, sigma=pi*R_disc^2 times the captured point "
            "fraction; reported sigma is the mean over discs with disc-cluster SEM and a "
            "Student-t 95% interval clipped to [0, pi*R_disc^2]"
        ),
        "loading_method": (
            "integrate one capture spectrum per direction disc by trapezoid quadrature, then "
            "report the mean, disc-cluster SEM, and Student-t 95% interval"
        ),
        "cooling_power_w_per_beam": apparatus.cooling.power_w_per_beam,
        "cooling_power_mw_per_beam": 1.0e3 * apparatus.cooling.power_w_per_beam,
        "built_cooling_beam_count": len(beams),
        "built_cooling_beam_powers_w": powers,
        "all_built_beams_have_requested_power": (
            len(beams) == 6
            and all(np.isclose(power, COOLING_POWER_W_PER_BEAM, rtol=0.0, atol=1.0e-15) for power in powers)
        ),
        "effective_saturation": effective_saturation_metrics(apparatus, simple),
        "anti_helmholtz_axial_gradient_t_per_m": axial_gradient,
        "anti_helmholtz_axial_gradient_g_per_cm": 100.0 * axial_gradient,
        "apparatus_config": asdict(apparatus),
        "simple_mot_config": asdict(simple),
        "coil_config": asdict(coil),
        "capture_search_config": asdict(search),
        "configuration_invariance_audit": configuration_invariance_audit(
            apparatus, simple, coil, search,
        ),
        "worker_count": worker_count,
        "checkpoint_every_completed_samples": CHECKPOINT_EVERY,
        "progress_every_completed_samples": PROGRESS_EVERY,
        "output_manifest": _output_manifest(paths),
        "git": _git_provenance(simple_mot_paths()["root"]),
        "limitations": [
            "The deterministic effective two-level model omits recoil diffusion and internal-state structure.",
            "The capture-speed binary search assumes a locally monotonic trapped/untrapped boundary.",
            "The sampling follows the authoritative one-octant convention, although gravity breaks exact z-reflection symmetry.",
            "The 12 mm sampling-disc radius caps the reported projected capture cross section.",
            "The quoted uncertainty is Monte Carlo direction-cluster uncertainty, not a full model-systematic uncertainty.",
        ],
    }


def _capture_summary(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    signature: str,
    geometry_hash: str,
    loading_summary: dict[str, object],
    paths: StudyPaths,
) -> dict[str, object]:
    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    resolution = np.asarray([sample.velocity_resolution_m_per_s for sample in samples], dtype=float)
    lower_counts = Counter(sample.lower_classification for sample in samples)
    upper_counts = Counter(sample.upper_classification for sample in samples)
    valid_brackets = sum(
        (
            sample.capture_velocity_m_per_s == 0.0
            or sample.lower_classification in TRAPPED_TERMINATION_REASONS
        )
        and sample.upper_classification not in TRAPPED_TERMINATION_REASONS
        for sample in samples
    )
    return {
        "model": "mot_simple deterministic effective two-level mean-force MOT",
        "run_signature_sha256": signature,
        "geometry_sha256": geometry_hash,
        "sample_count": len(samples),
        "expected_sample_count": search.disc_count * search.points_per_disc,
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "capture_velocity_mean_m_per_s": float(np.mean(capture)),
        "capture_velocity_sample_std_m_per_s": float(np.std(capture, ddof=1)) if len(capture) > 1 else 0.0,
        "capture_velocity_min_m_per_s": float(np.min(capture)),
        "capture_velocity_max_m_per_s": float(np.max(capture)),
        "zero_capture_velocity_count": int(np.count_nonzero(capture == 0.0)),
        "velocity_resolution_mean_m_per_s": float(np.mean(resolution)),
        "velocity_resolution_max_m_per_s": float(np.max(resolution)),
        "valid_bracket_count": int(valid_brackets),
        "lower_classification_counts": dict(sorted(lower_counts.items())),
        "upper_classification_counts": dict(sorted(upper_counts.items())),
        "loading_rate": loading_summary,
        "search_config": asdict(search),
        "outputs": _output_manifest(paths),
    }


def analyze_completed_samples(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    paths: StudyPaths,
    *,
    signature: str,
    geometry_hash: str,
) -> dict[str, object]:
    """Write cross-section, loading, summary, and compact diagnostic products."""

    expected = search.disc_count * search.points_per_disc
    if len(samples) != expected:
        raise ValueError(f"analysis requires {expected} samples, received {len(samples)}")
    counts = Counter(sample.disc_index for sample in samples)
    if set(counts) != set(range(search.disc_count)):
        raise ValueError("analysis samples do not contain every expected direction disc")
    if any(count != search.points_per_disc for count in counts.values()):
        raise ValueError("analysis samples do not contain the requested points per disc")

    spectrum = calculate_clustered_cross_section(samples, search)
    by_disc, loading = calculate_disc_clustered_loading(samples, search, spectrum)
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
    }
    _atomic_write_json(paths.loading_json, loading_payload)
    plot_clustered_cross_section(spectrum, paths.cross_section_png)
    plot_capture_velocity_vs_impact_parameter(samples, paths.impact_parameter_png)
    plot_loading_rate_by_disc(by_disc, loading, paths.loading_by_disc_png)
    summary = _capture_summary(
        samples, search, signature, geometry_hash, loading_payload, paths,
    )
    _atomic_write_json(paths.capture_summary_json, summary)
    return summary


_WORKER_BEAMS = None
_WORKER_COIL: AntiHelmholtzCoilConfig | None = None
_WORKER_SIMPLE: SimpleMOTConfig | None = None
_WORKER_SEARCH: CaptureSearchConfig | None = None


def _initialize_capture_worker(
    apparatus: PMOTSimulationConfig,
    simple: SimpleMOTConfig,
    coil: AntiHelmholtzCoilConfig,
    search: CaptureSearchConfig,
) -> None:
    """Build immutable beam/config state once in each worker process."""

    global _WORKER_BEAMS, _WORKER_COIL, _WORKER_SIMPLE, _WORKER_SEARCH
    _WORKER_BEAMS = build_simple_mot_beams(apparatus, simple)
    _WORKER_COIL = coil
    _WORKER_SIMPLE = simple
    _WORKER_SEARCH = search


def _capture_worker(point: PointSample) -> CaptureVelocitySample:
    if _WORKER_BEAMS is None or _WORKER_COIL is None or _WORKER_SIMPLE is None or _WORKER_SEARCH is None:
        raise RuntimeError("capture worker was not initialized")
    return find_capture_velocity(
        _WORKER_BEAMS,
        point,
        _WORKER_COIL,
        _WORKER_SIMPLE,
        _WORKER_SEARCH,
    )


def _validate_resume_state(
    paths: StudyPaths,
    expected_signature: str,
    expected_geometry_hash: str,
) -> dict[str, object]:
    if not paths.metadata_json.is_file():
        raise ValueError("resume requested but run_metadata.json is missing")
    if not paths.geometry_csv.is_file():
        raise ValueError("resume requested but launch_geometry.csv is missing")
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("run_signature_sha256") != expected_signature:
        raise ValueError("resume signature mismatch: physics, configuration, or geometry changed")
    actual_geometry_hash = _sha256_file(paths.geometry_csv)
    if actual_geometry_hash != expected_geometry_hash:
        raise ValueError("resume geometry mismatch: launch_geometry.csv was changed or corrupted")
    return metadata


def _load_existing_results(
    paths: StudyPaths,
    points: Sequence[PointSample],
) -> dict[tuple[int, int], CaptureVelocitySample]:
    candidate = None
    if paths.final_samples_csv.is_file():
        candidate = paths.final_samples_csv
    elif paths.partial_samples_csv.is_file():
        candidate = paths.partial_samples_csv
    if candidate is None:
        return {}
    return validate_checkpoint_samples(load_capture_velocity_samples(candidate), points)


def _reset_known_study_outputs(paths: StudyPaths) -> None:
    """Remove only this runner's known products for an explicit fresh restart."""

    for path in (
        paths.metadata_json,
        paths.geometry_csv,
        paths.partial_samples_csv,
        paths.final_samples_csv,
        paths.capture_summary_json,
        paths.spectrum_csv,
        paths.loading_by_disc_csv,
        paths.loading_json,
        paths.cross_section_png,
        paths.impact_parameter_png,
        paths.loading_by_disc_png,
    ):
        if path.is_file():
            path.unlink()


def _save_running_checkpoint(
    paths: StudyPaths,
    results: dict[tuple[int, int], CaptureVelocitySample],
    metadata: dict[str, object],
    *,
    elapsed_wall_time_s: float,
    eta_s: float | None,
    status: str = "running",
    error: str | None = None,
) -> None:
    save_samples_atomic(paths.partial_samples_csv, results.values())
    updated = dict(metadata)
    updated.update({
        "status": status,
        "updated_utc": _utc_now(),
        "completed_sample_count": len(results),
        "completion_fraction": len(results) / int(metadata["expected_sample_count"]),
        "elapsed_wall_time_s": elapsed_wall_time_s,
        "eta_s": eta_s,
    })
    if error is not None:
        updated["last_error"] = error
    _atomic_write_json(paths.metadata_json, updated)
    metadata.clear()
    metadata.update(updated)


def run_power_loading_study(
    search_config: CaptureSearchConfig | None = None,
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
    analyze_only: bool = False,
) -> dict[str, object]:
    """Run or resume the complete 27 mW two-level capture/loading study."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if analyze_only and not resume:
        raise ValueError("--analyze-only requires resume mode and existing completed samples")
    search = search_config or default_study_search_config()
    if search.bounded_core_residence_s != 5.0e-3:
        raise ValueError("the production trapped criterion requires 5 ms core residence")
    default_paths = default_study_paths()
    paths = StudyPaths(
        statistics=Path(output_directory) if output_directory is not None else default_paths.statistics,
        figures=Path(figure_directory) if figure_directory is not None else default_paths.figures,
    )
    paths.statistics.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    apparatus = build_27mw_apparatus()
    simple = default_simple_mot_config()
    coil = default_anti_helmholtz_config()
    discs, points = generate_study_geometry(search)
    rows = geometry_rows(discs, points)
    geometry_text = geometry_csv_text(rows)
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    signature_payload = study_signature_payload(apparatus, simple, coil, search, geometry_hash)
    signature = study_signature(signature_payload)
    expected = search.disc_count * search.points_per_disc

    existing_state = any(path.exists() for path in (
        paths.metadata_json,
        paths.geometry_csv,
        paths.partial_samples_csv,
        paths.final_samples_csv,
        paths.capture_summary_json,
        paths.spectrum_csv,
        paths.loading_by_disc_csv,
        paths.loading_json,
        paths.cross_section_png,
        paths.impact_parameter_png,
        paths.loading_by_disc_png,
    ))
    if not resume and existing_state:
        _reset_known_study_outputs(paths)
        existing_state = False
    prior_metadata: dict[str, object] | None = None
    if resume and existing_state:
        prior_metadata = _validate_resume_state(paths, signature, geometry_hash)
        results = _load_existing_results(paths, points)
    else:
        _atomic_write_text(paths.geometry_csv, geometry_text)
        results = {}

    if len(results) > expected:
        raise ValueError("checkpoint contains more samples than the requested geometry")

    prior_elapsed = float(prior_metadata.get("elapsed_wall_time_s", 0.0)) if prior_metadata else 0.0
    started_utc = str(prior_metadata.get("started_utc")) if prior_metadata else _utc_now()
    metadata = build_run_metadata(
        apparatus,
        simple,
        coil,
        search,
        signature_payload,
        signature,
        paths,
        worker_count=worker_count,
        status="running",
        completed_sample_count=len(results),
        started_utc=started_utc,
        elapsed_wall_time_s=prior_elapsed,
    )
    _atomic_write_json(paths.metadata_json, metadata)

    if analyze_only:
        if len(results) != expected:
            raise ValueError(
                f"--analyze-only requires all {expected} samples; checkpoint contains {len(results)}"
            )
        samples = _sorted_samples(results.values())
        save_samples_atomic(paths.final_samples_csv, samples)
        summary = analyze_completed_samples(
            samples, search, paths, signature=signature, geometry_hash=geometry_hash,
        )
        metadata.update({
            "status": "completed",
            "updated_utc": _utc_now(),
            "completed_sample_count": expected,
            "completion_fraction": 1.0,
            "analysis_only_invocation": True,
            "loading_rate": summary["loading_rate"],
        })
        _atomic_write_json(paths.metadata_json, metadata)
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    point_map = {(point.disc_index, point.point_index): point for point in points}
    missing = [point for key, point in point_map.items() if key not in results]
    print(
        f"[mot_simple 27 mW] {len(results)}/{expected} samples already complete; "
        f"running {len(missing)} with {worker_count} worker(s)",
        flush=True,
    )
    segment_start = perf_counter()
    initial_completed = len(results)
    last_eta: float | None = None
    executor: ProcessPoolExecutor | None = None
    execution_failed = False
    try:
        if worker_count == 1:
            _initialize_capture_worker(apparatus, simple, coil, search)
            iterator = (_capture_worker(point) for point in missing)
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_capture_worker,
                initargs=(apparatus, simple, coil, search),
            )
            futures = {executor.submit(_capture_worker, point): point for point in missing}
            iterator = (future.result() for future in as_completed(futures))

        for sample in iterator:
            key = (sample.disc_index, sample.point_index)
            _validate_sample_against_point(sample, point_map[key])
            results[key] = sample
            completed = len(results)
            new_completed = completed - initial_completed
            segment_elapsed = perf_counter() - segment_start
            rate = new_completed / segment_elapsed if segment_elapsed > 0.0 else 0.0
            remaining = expected - completed
            last_eta = remaining / rate if rate > 0.0 else None
            if completed % PROGRESS_EVERY == 0 or completed == expected:
                eta_text = "unknown" if last_eta is None else f"{last_eta / 60.0:.1f} min"
                print(
                    f"[mot_simple 27 mW] {completed}/{expected}; "
                    f"disc {sample.disc_index + 1}/{search.disc_count}, "
                    f"point {sample.point_index + 1}/{search.points_per_disc}; "
                    f"vc={sample.capture_velocity_m_per_s:.3f} m/s; ETA={eta_text}",
                    flush=True,
                )
            if completed % CHECKPOINT_EVERY == 0:
                _save_running_checkpoint(
                    paths,
                    results,
                    metadata,
                    elapsed_wall_time_s=prior_elapsed + segment_elapsed,
                    eta_s=last_eta,
                )
                print(f"[mot_simple 27 mW] checkpoint saved at {completed}/{expected}", flush=True)
    except BaseException as exc:
        execution_failed = True
        elapsed = prior_elapsed + (perf_counter() - segment_start)
        _save_running_checkpoint(
            paths,
            results,
            metadata,
            elapsed_wall_time_s=elapsed,
            eta_s=last_eta,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=execution_failed)

    if len(results) != expected:
        raise RuntimeError(f"sampling incomplete: obtained {len(results)} of {expected} samples")
    samples = _sorted_samples(results.values())
    save_samples_atomic(paths.partial_samples_csv, samples)
    save_samples_atomic(paths.final_samples_csv, samples)
    summary = analyze_completed_samples(
        samples, search, paths, signature=signature, geometry_hash=geometry_hash,
    )
    segment_elapsed = perf_counter() - segment_start
    metadata.update({
        "status": "completed",
        "updated_utc": _utc_now(),
        "completed_sample_count": expected,
        "completion_fraction": 1.0,
        "elapsed_wall_time_s": prior_elapsed + segment_elapsed,
        "eta_s": 0.0,
        "loading_rate": summary["loading_rate"],
    })
    _atomic_write_json(paths.metadata_json, metadata)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed 27 mW/beam, 50-disc x 50-point two-level MOT "
            "capture-cross-section and loading-rate study"
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_power_loading_study(
        worker_count=args.workers,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=args.resume,
        analyze_only=args.analyze_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_EVERY",
    "COOLING_POWER_W_PER_BEAM",
    "DEFAULT_WORKER_COUNT",
    "DISC_COUNT",
    "POINTS_PER_DISC",
    "RANDOM_SEED",
    "STUDY_NAME",
    "StudyPaths",
    "analyze_completed_samples",
    "build_27mw_apparatus",
    "build_argument_parser",
    "calculate_clustered_cross_section",
    "calculate_disc_clustered_loading",
    "configuration_invariance_audit",
    "default_study_paths",
    "default_study_search_config",
    "effective_saturation_metrics",
    "generate_study_geometry",
    "geometry_rows",
    "geometry_sha256",
    "run_power_loading_study",
    "study_signature",
    "study_signature_payload",
    "validate_checkpoint_samples",
]
