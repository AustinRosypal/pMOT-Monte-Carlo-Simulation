"""Resumable cooling-power/detuning loading study for the 24-state MOT.

The default remains the original 27 mW cooling, -15 MHz detuning, and 0.1 mW
repump production study.  The public runner also accepts an explicit red
cooling detuning so parameter campaigns can reuse the same signed, checkpointed
50-disc by 50-point workflow.  It uses the repumper-included, adiabatically
eliminated population-rate force.  Capture trajectories are deterministic
(recoil diffusion is disabled) so that the local monotonicity assumption used
by the capture-speed binary search remains meaningful.

Six cooling components are each assigned 27 mW, while the six co-propagating
repump components retain the authoritative 0.1 mW baseline.  At the default
-15 MHz cooling detuning, the cooling power gives an effective saturation near
unity.  Cooling and resonant-repump saturation are recorded separately.
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
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..configuration import MOTApparatusConfig, default_mot_apparatus_config
from ..fields import MOTBeam
from ..configuration import AntiHelmholtzCoilConfig
from ..magnetic_fields import (
    anti_helmholtz_axial_gradient_t_per_m,
    default_anti_helmholtz_config,
)
from ..loading import (
    LOADING_RATE_PREFACTOR,
    THERMAL_SCALE_M2_PER_S2,
    calculate_loading_rate_from_spectrum,
)
from ..capture_statistics import CaptureVelocitySample, load_capture_velocity_samples
from ..launch_geometry import (
    DiscSample,
    PointSample,
    sample_disc_points,
    sample_incident_disc,
    sample_incident_disc_full_sphere,
)
from .configuration import (
    MultilevelMOTConfig,
    default_multilevel_mot_config,
    multilevel_mot_paths,
)
from .loading_sweeps import find_multilevel_capture_velocity
from .rate_capture import RateCaptureSearchConfig
from .rate_equations import RateEquationModel, build_rate_equation_model
from .simulation import build_multilevel_mot_beams


STUDY_NAME = "loading_rate_cooling27mW_repump0p1mW_50_discs_50_points"
COOLING_POWER_W_PER_BEAM = 27.0e-3
COOLING_DETUNING_HZ = -15.0e6
REPUMP_POWER_W_PER_BEAM = 0.1e-3
DISC_COUNT = 50
POINTS_PER_DISC = 50
RANDOM_SEED = 0
DEFAULT_WORKER_COUNT = 20
CHECKPOINT_EVERY = 10
PROGRESS_EVERY = 1
METADATA_SCHEMA_VERSION = 1

TRAPPED_TERMINATION_REASONS = frozenset(
    {"two_core_entries", "bounded_core_residence"}
)
VALID_UNTRAPPED_TERMINATION_REASONS = frozenset({"escaped", "timeout"})

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

LOADING_BY_DISC_FIELDNAMES = [
    "disc_index",
    "point_count",
    "loading_integral_m6_per_s4",
    "loading_rate_atoms_per_s",
]


@dataclass(frozen=True, slots=True)
class StudyPaths:
    """Filesystem products for the dedicated multilevel power study."""

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
    """Return dedicated paths below the multilevel output architecture."""

    paths = multilevel_mot_paths(root)
    return StudyPaths(
        statistics=paths["statistics"] / STUDY_NAME,
        figures=paths["figures"] / STUDY_NAME,
    )


def default_study_search_config() -> RateCaptureSearchConfig:
    """Return the paired-comparison 50-disc by 50-point search design."""

    return replace(
        RateCaptureSearchConfig(),
        disc_count=DISC_COUNT,
        points_per_disc=POINTS_PER_DISC,
        seed=RANDOM_SEED,
        include_center_point=False,
    )


def build_27mw_multilevel_configuration(
    *,
    cooling_power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
    repump_power_w_per_beam: float = REPUMP_POWER_W_PER_BEAM,
    cooling_detuning_hz: float = COOLING_DETUNING_HZ,
) -> tuple[MultilevelMOTConfig, MOTApparatusConfig, list[MOTBeam]]:
    """Build the 24-state MOT, retaining the historical defaults.

    ``cooling_detuning_hz`` uses ordinary frequency units at this public
    apparatus boundary.  The multilevel solver receives the exactly matching
    angular detuning in rad/s.
    """

    if cooling_power_w_per_beam <= 0.0 or repump_power_w_per_beam <= 0.0:
        raise ValueError("cooling and repump powers must both be positive")
    if not np.isfinite(cooling_detuning_hz) or cooling_detuning_hz >= 0.0:
        raise ValueError("cooling_detuning_hz must be finite and negative")
    cooling_detuning_hz = float(cooling_detuning_hz)
    baseline_apparatus = default_mot_apparatus_config()
    apparatus = replace(
        baseline_apparatus,
        cooling=replace(
            baseline_apparatus.cooling,
            power_w_per_beam=float(cooling_power_w_per_beam),
            detuning_hz=cooling_detuning_hz,
        ),
        repump=replace(
            baseline_apparatus.repump,
            power_w_per_beam=float(repump_power_w_per_beam),
        ),
    )
    config = replace(
        default_multilevel_mot_config(),
        cooling_detuning_rad_per_s=2.0 * pi * cooling_detuning_hz,
        repumper_enabled=True,
        repump_power_w_per_beam=float(repump_power_w_per_beam),
    )
    beams = build_multilevel_mot_beams(apparatus_config=apparatus, config=config)
    _validate_built_model_inputs(
        config,
        beams,
        cooling_power_w_per_beam,
        repump_power_w_per_beam,
        cooling_detuning_hz,
    )
    return config, apparatus, beams


def _validate_built_model_inputs(
    config: MultilevelMOTConfig,
    beams: Sequence[MOTBeam],
    cooling_power_w_per_beam: float,
    repump_power_w_per_beam: float,
    cooling_detuning_hz: float,
) -> None:
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    if not config.repumper_enabled or len(cooling) != 6 or len(repump) != 6:
        raise RuntimeError("production study requires six cooling and six repump components")
    if any(
        not np.isclose(beam.power_w, cooling_power_w_per_beam, rtol=0.0, atol=1.0e-15)
        for beam in cooling
    ):
        raise RuntimeError("one or more cooling beams does not have the requested power")
    if any(
        not np.isclose(beam.power_w, repump_power_w_per_beam, rtol=0.0, atol=1.0e-15)
        for beam in repump
    ):
        raise RuntimeError("one or more repump beams does not have the requested power")
    if any(
        not np.isclose(beam.detuning_hz, cooling_detuning_hz, rtol=0.0, atol=1.0e-9)
        for beam in cooling
    ):
        raise RuntimeError("one or more cooling beams does not have the requested detuning")
    if not np.isclose(
        config.cooling_detuning_rad_per_s,
        2.0 * pi * cooling_detuning_hz,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError("solver and apparatus cooling detunings are inconsistent")


def effective_saturation_metrics(
    config: MultilevelMOTConfig | None = None,
    apparatus: MOTApparatusConfig | None = None,
) -> dict[str, object]:
    """Return distinct cooling and resonant-repump saturation metrics."""

    if config is None or apparatus is None:
        config, apparatus, _ = build_27mw_multilevel_configuration()

    def metrics(power_w: float, radius_m: float, detuning_rad_per_s: float) -> dict[str, float]:
        peak_intensity = 2.0 * power_w / (pi * radius_m**2)
        saturation = peak_intensity / config.saturation_intensity_w_per_m2
        denominator = 1.0 + (
            2.0 * detuning_rad_per_s / config.natural_linewidth_rad_per_s
        ) ** 2
        return {
            "beam_center_peak_intensity_w_per_m2": peak_intensity,
            "beam_center_on_resonance_saturation_parameter": saturation,
            "detuning_reduction_denominator": denominator,
            "beam_center_effective_saturation_parameter": saturation / denominator,
        }

    return {
        "definition": "s_eff = (I0/I_sat) / [1 + (2*Delta/Gamma)^2]",
        "cooling": metrics(
            apparatus.cooling.power_w_per_beam,
            apparatus.cooling.beam_radius_m,
            config.cooling_detuning_rad_per_s,
        ),
        "repump": metrics(
            config.repump_power_w_per_beam,
            apparatus.cooling.beam_radius_m,
            config.repump_detuning_rad_per_s,
        ),
        "interpretation": (
            f"The cooling value applies to the {1e3 * apparatus.cooling.power_w_per_beam:g} "
            f"mW, {apparatus.cooling.detuning_hz / 1e6:g} MHz cooling component. "
            "The resonant repumper's effective "
            "saturation equals its on-resonance saturation."
        ),
    }


def generate_study_geometry(
    search: RateCaptureSearchConfig,
) -> tuple[list[DiscSample], list[PointSample]]:
    """Generate the exact seeded geometry algorithm used by the simple study."""

    if search.disc_count <= 0 or search.points_per_disc <= 0:
        raise ValueError("disc_count and points_per_disc must be positive")
    if search.phase_space not in {"octant", "full_sphere"}:
        raise ValueError("phase_space must be 'octant' or 'full_sphere'")
    if search.include_center_point:
        raise ValueError("all production impact points must be random uniform-area draws")
    rng = np.random.default_rng(search.seed)
    discs: list[DiscSample] = []
    points: list[PointSample] = []
    direction_sampler = (
        sample_incident_disc
        if search.phase_space == "octant"
        else sample_incident_disc_full_sphere
    )
    for disc_index in range(search.disc_count):
        disc = direction_sampler(disc_index, search.radial_distance_m, rng)
        discs.append(disc)
        points.extend(
            sample_disc_points(
                disc,
                search.points_per_disc,
                search.disc_radius_m,
                search.include_center_point,
                rng,
            )
        )
    return discs, points


def geometry_rows(
    discs: Sequence[DiscSample],
    points: Sequence[PointSample],
) -> list[dict[str, int | float]]:
    """Return exact launch geometry in a stable CSV representation."""

    disc_map = {disc.disc_index: disc for disc in discs}
    rows: list[dict[str, int | float]] = []
    for point in sorted(points, key=lambda item: (item.disc_index, item.point_index)):
        disc = disc_map[point.disc_index]
        rows.append(
            {
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
            }
        )
    return rows


def _csv_text(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def geometry_csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    return _csv_text(rows, GEOMETRY_FIELDNAMES)


def geometry_sha256(rows: Sequence[Mapping[str, object]]) -> str:
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
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    _atomic_write_text(path, _csv_text(rows, fieldnames))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physics_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "power_loading_study.py",
        "loading_sweeps.py",
        "rate_equations.py",
        "coupling.py",
        "atomic_structure.py",
        "simulation.py",
        "configuration.py",
    )
    return {name: _sha256_file(directory / name) for name in names}


def _flatten_differences(baseline: object, candidate: object, prefix: str = "") -> list[str]:
    differences: list[str] = []
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(set(baseline) | set(candidate)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline or key not in candidate:
                differences.append(child)
            else:
                differences.extend(_flatten_differences(baseline[key], candidate[key], child))
        return differences
    if isinstance(baseline, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if len(baseline) != len(candidate):
            return [prefix]
        for index, (left, right) in enumerate(zip(baseline, candidate)):
            differences.extend(_flatten_differences(left, right, f"{prefix}[{index}]"))
        return differences
    if baseline != candidate:
        differences.append(prefix)
    return differences


def configuration_invariance_audit(
    config: MultilevelMOTConfig,
    apparatus: MOTApparatusConfig,
    coil: AntiHelmholtzCoilConfig,
    search: RateCaptureSearchConfig,
) -> dict[str, object]:
    """Record exactly which requested production fields differ from defaults."""

    apparatus_changes = _flatten_differences(
        asdict(default_mot_apparatus_config()), asdict(apparatus)
    )
    config_changes = _flatten_differences(
        asdict(default_multilevel_mot_config()), asdict(config)
    )
    coil_changes = _flatten_differences(
        asdict(default_anti_helmholtz_config()), asdict(coil)
    )
    search_changes = _flatten_differences(
        asdict(RateCaptureSearchConfig()), asdict(search)
    )
    sampling_and_execution_fields = {
        "disc_radius_m",
        "disc_count",
        "points_per_disc",
        "include_center_point",
        "seed",
        "worker_count",
        "phase_space",
    }
    unexpected_search_changes = sorted(
        set(search_changes) - sampling_and_execution_fields
    )
    return {
        "apparatus_changed_fields": apparatus_changes,
        "apparatus_only_cooling_and_repump_power_changed": apparatus_changes
        == ["cooling.power_w_per_beam", "repump.power_w_per_beam"],
        "apparatus_changes_within_requested_power_and_cooling_detuning_fields": (
            not (
                set(apparatus_changes)
                - {
                    "cooling.detuning_hz",
                    "cooling.power_w_per_beam",
                    "repump.power_w_per_beam",
                }
            )
        ),
        "multilevel_config_changed_fields": config_changes,
        "multilevel_only_repumper_enable_changed": config_changes
        == ["repumper_enabled"],
        "multilevel_changes_within_requested_repumper_and_cooling_detuning_fields": (
            not (
                set(config_changes)
                - {
                    "cooling_detuning_rad_per_s",
                    "repump_power_w_per_beam",
                    "repumper_enabled",
                }
            )
        ),
        "repump_power_matches_authoritative_multilevel_default": bool(
            np.isclose(
                config.repump_power_w_per_beam,
                default_multilevel_mot_config().repump_power_w_per_beam,
                rtol=0.0,
                atol=1.0e-15,
            )
        ),
        "coil_config_changed_fields": coil_changes,
        "coil_config_matches_default": not coil_changes,
        "capture_search_changed_fields": search_changes,
        "capture_search_allowed_sampling_and_execution_fields": sorted(
            sampling_and_execution_fields
        ),
        "capture_search_non_sampling_changed_fields": unexpected_search_changes,
        "capture_search_only_sampling_design_changed": not unexpected_search_changes,
    }


def study_signature_payload(
    config: MultilevelMOTConfig,
    apparatus: MOTApparatusConfig,
    coil: AntiHelmholtzCoilConfig,
    search: RateCaptureSearchConfig,
    geometry_hash: str,
    *,
    study_name: str = STUDY_NAME,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    search_payload = asdict(search)
    # The pool size is an execution choice, not part of the sampled geometry or
    # physics.  Excluding it lets a checkpoint resume safely with fewer workers.
    search_payload.pop("worker_count", None)
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "study_name": study_name,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "apparatus_config": asdict(apparatus),
        "multilevel_config": asdict(config),
        "coil_config": asdict(coil),
        "capture_search_config": search_payload,
        "geometry_sha256": geometry_hash,
        "physics_source_sha256": source_hashes or _physics_source_hashes(),
    }


def study_signature(payload: Mapping[str, object]) -> str:
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
    return sorted(samples, key=lambda item: (item.disc_index, item.point_index))


def save_samples_atomic(path: Path, samples: Iterable[CaptureVelocitySample]) -> None:
    _atomic_write_csv(path, [_sample_to_row(sample) for sample in _sorted_samples(samples)], SAMPLE_FIELDNAMES)


def _validate_sample_against_point(sample: CaptureVelocitySample, point: PointSample) -> None:
    key = (sample.disc_index, sample.point_index)
    if key != (point.disc_index, point.point_index):
        raise ValueError("checkpoint sample key does not match seeded geometry")
    scalar_pairs = (
        (sample.theta_rad, point.theta_rad),
        (sample.phi_rad, point.phi_rad),
        (sample.theta_prime_rad, point.theta_prime_rad),
        (sample.s_m, point.s_m),
        (sample.radial_distance_m, point.radial_distance_m),
    )
    if any(not np.isclose(a, b, rtol=0.0, atol=2.0e-14) for a, b in scalar_pairs):
        raise ValueError("checkpoint scalar geometry differs from seeded geometry")
    if not np.allclose(sample.initial_position_m, point.initial_position_m, rtol=0.0, atol=2.0e-14):
        raise ValueError("checkpoint initial position differs from seeded geometry")
    if not np.allclose(
        sample.incident_unit_vector,
        point.incident_unit_vector,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise ValueError("checkpoint incident direction differs from seeded geometry")
    numeric = np.asarray(
        [
            sample.capture_velocity_m_per_s,
            sample.velocity_resolution_m_per_s,
            sample.trapped_velocity_lower_m_per_s,
            sample.untrapped_velocity_upper_m_per_s,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
        raise ValueError("checkpoint contains a nonfinite or negative speed endpoint")
    if sample.untrapped_velocity_upper_m_per_s < sample.trapped_velocity_lower_m_per_s:
        raise ValueError("checkpoint capture-speed bracket is reversed")
    if sample.capture_velocity_m_per_s > 0.0 and sample.lower_classification not in TRAPPED_TERMINATION_REASONS:
        raise ValueError("positive capture threshold has an untrapped lower endpoint")
    if sample.upper_classification not in VALID_UNTRAPPED_TERMINATION_REASONS:
        raise ValueError("checkpoint upper endpoint is not escaped or timeout")


def validate_checkpoint_samples(
    samples: Sequence[CaptureVelocitySample],
    points: Sequence[PointSample],
) -> dict[tuple[int, int], CaptureVelocitySample]:
    point_map = {(point.disc_index, point.point_index): point for point in points}
    validated: dict[tuple[int, int], CaptureVelocitySample] = {}
    for sample in samples:
        key = (sample.disc_index, sample.point_index)
        if key in validated:
            raise ValueError(f"duplicate checkpoint sample {key}")
        if key not in point_map:
            raise ValueError(f"checkpoint contains unknown sample {key}")
        _validate_sample_against_point(sample, point_map[key])
        validated[key] = sample
    return validated


def _velocity_grid(samples: Sequence[CaptureVelocitySample], search: RateCaptureSearchConfig) -> np.ndarray:
    step = search.analysis_velocity_step_m_per_s
    largest_endpoint = max(
        (sample.untrapped_velocity_upper_m_per_s for sample in samples),
        default=search.analysis_velocity_max_m_per_s,
    )
    stop = step * np.ceil(max(search.analysis_velocity_max_m_per_s, largest_endpoint) / step)
    return np.arange(0.0, stop + 0.5 * step, step, dtype=float)


def _student_t_critical_95(sample_count: int) -> float:
    return 0.0 if sample_count <= 1 else float(student_t.ppf(0.975, df=sample_count - 1))


def calculate_clustered_cross_section(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    velocity_grid_m_per_s: np.ndarray | None = None,
) -> list[dict[str, int | float]]:
    """Calculate mean cross section and direction-clustered Student-t interval."""

    if not samples:
        raise ValueError("at least one capture sample is required")
    velocity = (
        _velocity_grid(samples, search)
        if velocity_grid_m_per_s is None
        else np.asarray(velocity_grid_m_per_s, dtype=float)
    )
    if velocity.ndim != 1 or len(velocity) < 2 or np.any(np.diff(velocity) <= 0.0):
        raise ValueError("velocity grid must be strictly increasing and one-dimensional")
    grouped: dict[int, list[CaptureVelocitySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample)
    disc_count = len(grouped)
    area = pi * search.disc_radius_m**2
    t_critical = _student_t_critical_95(disc_count)
    def captured_at(sample: CaptureVelocitySample, speed: float) -> bool:
        return bool(
            sample.lower_classification in TRAPPED_TERMINATION_REASONS
            and sample.capture_velocity_m_per_s >= speed - 1.0e-12
        )

    rows: list[dict[str, int | float]] = []
    for speed in velocity:
        disc_sigma = area * np.asarray(
            [
                np.mean([captured_at(sample, float(speed)) for sample in grouped[index]])
                for index in sorted(grouped)
            ]
        )
        mean = float(np.mean(disc_sigma))
        sample_std = float(np.std(disc_sigma, ddof=1)) if disc_count > 1 else 0.0
        sem = sample_std / np.sqrt(disc_count)
        half_width = t_critical * sem
        rows.append(
            {
                "velocity_m_per_s": float(speed),
                "captured_count": int(
                    sum(captured_at(sample, float(speed)) for sample in samples)
                ),
                "launched_count": len(samples),
                "capture_fraction": mean / area,
                "capture_cross_section_m2": mean,
                "capture_cross_section_sample_std_m2": sample_std,
                "capture_cross_section_disc_cluster_sem_m2": sem,
                "capture_cross_section_t95_lower_m2": max(0.0, mean - half_width),
                "capture_cross_section_t95_upper_m2": min(area, mean + half_width),
                "disc_count": disc_count,
                "student_t_critical_95": t_critical,
            }
        )
    return rows


def calculate_disc_clustered_loading(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    spectrum_rows: Sequence[Mapping[str, int | float]],
) -> tuple[list[dict[str, int | float]], dict[str, int | float | str]]:
    """Integrate each direction disc and summarize the clustered uncertainty."""

    velocity = np.asarray([row["velocity_m_per_s"] for row in spectrum_rows], dtype=float)
    if len(velocity) < 2:
        raise ValueError("capture spectrum must have at least two velocity points")
    grouped: dict[int, list[CaptureVelocitySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample)
    area = pi * search.disc_radius_m**2
    by_disc: list[dict[str, int | float]] = []
    for disc_index in sorted(grouped):
        disc_samples = grouped[disc_index]
        sigma = area * np.asarray(
            [
                np.mean(
                    [
                        sample.lower_classification in TRAPPED_TERMINATION_REASONS
                        and sample.capture_velocity_m_per_s >= speed - 1.0e-12
                        for sample in disc_samples
                    ]
                )
                for speed in velocity
            ],
            dtype=float,
        )
        result = calculate_loading_rate_from_spectrum(velocity, sigma)
        by_disc.append(
            {
                "disc_index": disc_index,
                "point_count": len(disc_samples),
                "loading_integral_m6_per_s4": result.integral_value_m5_per_s4,
                "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
            }
        )
    rates = np.asarray([row["loading_rate_atoms_per_s"] for row in by_disc], dtype=float)
    integrals = np.asarray([row["loading_integral_m6_per_s4"] for row in by_disc], dtype=float)
    mean_sigma = np.asarray([row["capture_cross_section_m2"] for row in spectrum_rows], dtype=float)
    mean_result = calculate_loading_rate_from_spectrum(velocity, mean_sigma)
    disc_count = len(by_disc)
    sample_std = float(np.std(rates, ddof=1)) if disc_count > 1 else 0.0
    sem = sample_std / np.sqrt(disc_count)
    t_critical = _student_t_critical_95(disc_count)
    mean = float(np.mean(rates))
    half_width = t_critical * sem
    return by_disc, {
        "loading_rate_mean_atoms_per_s": mean,
        "loading_rate_from_mean_spectrum_atoms_per_s": mean_result.loading_rate_atoms_per_s,
        "loading_rate_sample_std_atoms_per_s": sample_std,
        "loading_rate_disc_cluster_sem_atoms_per_s": sem,
        "loading_rate_t95_lower_atoms_per_s": max(0.0, mean - half_width),
        "loading_rate_t95_upper_atoms_per_s": mean + half_width,
        "student_t_critical_95": t_critical,
        "confidence_level": 0.95,
        "disc_count": disc_count,
        "point_count": len(samples),
        "loading_integral_mean_m6_per_s4": float(np.mean(integrals)),
        "loading_integral_from_mean_spectrum_m6_per_s4": mean_result.integral_value_m5_per_s4,
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


def plot_clustered_cross_section(
    rows: Sequence[Mapping[str, int | float]],
    path: Path,
    *,
    title: str = "24-State MOT Capture Cross Section (27 mW cooling; 0.1 mW repump)",
) -> Path:
    velocity = np.asarray([row["velocity_m_per_s"] for row in rows], dtype=float)
    mean = 1.0e6 * np.asarray([row["capture_cross_section_m2"] for row in rows], dtype=float)
    lower = 1.0e6 * np.asarray([row["capture_cross_section_t95_lower_m2"] for row in rows])
    upper = 1.0e6 * np.asarray([row["capture_cross_section_t95_upper_m2"] for row in rows])
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.fill_between(velocity, lower, upper, color="#99c8c2", alpha=0.5, linewidth=0.0, label="95% t interval across direction discs")
    axis.plot(velocity, mean, color="#0f766e", linewidth=2.2, label="Mean cross section")
    axis.set(
        title=title,
        xlabel="Launch speed [m/s]",
        ylabel=r"Capture cross section [mm$^2$]",
    )
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
    title: str = "24-State MOT Capture Velocity (27 mW cooling; 0.1 mW repump)",
) -> Path:
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
            if not len(values):
                continue
            centers.append(0.5 * (edges[index] + edges[index + 1]))
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        axis.errorbar(centers, means, yerr=sems, color="#9f4a13", marker="o", markersize=4, linewidth=1.7, capsize=2, label="Impact-parameter-bin mean ± SEM")
        axis.legend(frameon=False)
    axis.set(
        title=title,
        xlabel="Impact parameter [mm]",
        ylabel="Capture velocity [m/s]",
    )
    axis.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_loading_rate_by_disc(
    by_disc: Sequence[Mapping[str, int | float]],
    summary: Mapping[str, int | float | str],
    path: Path,
    *,
    title: str = "24-State MOT Loading Rate by Direction (27 mW cooling; 0.1 mW repump)",
) -> Path:
    disc = 1 + np.asarray([row["disc_index"] for row in by_disc], dtype=int)
    rate = np.asarray([row["loading_rate_atoms_per_s"] for row in by_disc], dtype=float)
    mean = float(summary["loading_rate_mean_atoms_per_s"])
    lower = float(summary["loading_rate_t95_lower_atoms_per_s"])
    upper = float(summary["loading_rate_t95_upper_atoms_per_s"])
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.axhspan(lower, upper, color="#99c8c2", alpha=0.5, linewidth=0.0, label="95% t interval for mean")
    axis.axhline(mean, color="#0f766e", linewidth=2.2, label="Mean loading rate")
    axis.scatter(disc, rate, color="#9f4a13", s=22, alpha=0.75, label="Direction-disc estimates")
    axis.set(
        title=title,
        xlabel="Random incident-direction disc",
        ylabel="Loading rate [atoms/s]",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_provenance(project_root: Path) -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": bool(status) if status is not None else None,
        "status_short": status.splitlines()[:200] if status else [],
    }


def _output_manifest(paths: StudyPaths) -> dict[str, str]:
    return {
        "statistics_directory": str(paths.statistics.resolve()),
        "figures_directory": str(paths.figures.resolve()),
        "launch_geometry_csv": str(paths.geometry_csv.resolve()),
        "partial_samples_csv": str(paths.partial_samples_csv.resolve()),
        "capture_velocity_samples_csv": str(paths.final_samples_csv.resolve()),
        "capture_velocity_summary_json": str(paths.capture_summary_json.resolve()),
        "capture_velocity_spectrum_csv": str(paths.spectrum_csv.resolve()),
        "loading_rate_by_disc_csv": str(paths.loading_by_disc_csv.resolve()),
        "loading_rate_result_json": str(paths.loading_json.resolve()),
        "capture_cross_section_plot": str(paths.cross_section_png.resolve()),
        "capture_velocity_impact_plot": str(paths.impact_parameter_png.resolve()),
        "loading_rate_by_disc_plot": str(paths.loading_by_disc_png.resolve()),
    }


def build_run_metadata(
    config: MultilevelMOTConfig,
    apparatus: MOTApparatusConfig,
    beams: Sequence[MOTBeam],
    coil: AntiHelmholtzCoilConfig,
    search: RateCaptureSearchConfig,
    signature_payload: Mapping[str, object],
    signature: str,
    paths: StudyPaths,
    *,
    worker_count: int,
    status: str,
    completed_sample_count: int,
    started_utc: str,
    elapsed_wall_time_s: float,
    study_name: str = STUDY_NAME,
    checkpoint_every: int = CHECKPOINT_EVERY,
    progress_every: int = PROGRESS_EVERY,
) -> dict[str, object]:
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    axial_gradient = anti_helmholtz_axial_gradient_t_per_m(
        coil.radius_m,
        coil.turns_per_coil,
        coil.current_a,
    )
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    expected = search.disc_count * search.points_per_disc
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "study_name": study_name,
        "status": status,
        "started_utc": started_utc,
        "updated_utc": _utc_now(),
        "completed_sample_count": completed_sample_count,
        "expected_sample_count": expected,
        "completion_fraction": completed_sample_count / expected,
        "elapsed_wall_time_s": elapsed_wall_time_s,
        "run_signature_sha256": signature,
        "run_signature_payload": signature_payload,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "indexed_state_count": model.state_count,
        "ground_state_count": model.ground_count,
        "excited_state_count": model.excited_count,
        "capture_dynamics": "deterministic multilevel mean force; recoil diffusion disabled",
        "capture_integrator": "semi-implicit Euler with fixed 5 microsecond external-motion timestep",
        "population_approximation": "adiabatically eliminated quasi-steady 24-state rate equations",
        "cooling_beam_component_count": len(cooling),
        "repump_beam_component_count": len(repump),
        "total_beam_component_count": len(beams),
        "cooling_power_w_per_beam": apparatus.cooling.power_w_per_beam,
        "repump_power_w_per_beam": config.repump_power_w_per_beam,
        "built_cooling_beam_powers_w": [beam.power_w for beam in cooling],
        "built_repump_beam_powers_w": [beam.power_w for beam in repump],
        "all_built_beams_match_requested_family_powers": (
            all(
                np.isclose(
                    beam.power_w,
                    apparatus.cooling.power_w_per_beam,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                for beam in cooling
            )
            and all(
                np.isclose(
                    beam.power_w,
                    config.repump_power_w_per_beam,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                for beam in repump
            )
        ),
        "effective_saturation": effective_saturation_metrics(config, apparatus),
        "cooling_detuning_hz": apparatus.cooling.detuning_hz,
        "cooling_detuning_rad_per_s": config.cooling_detuning_rad_per_s,
        "repump_detuning_hz": apparatus.repump.detuning_hz,
        "repump_detuning_rad_per_s": config.repump_detuning_rad_per_s,
        "internal_frequency_units": "angular frequency in rad/s",
        "anti_helmholtz_axial_gradient_t_per_m": axial_gradient,
        "anti_helmholtz_axial_gradient_g_per_cm": 100.0 * axial_gradient,
        "apparatus_config": asdict(apparatus),
        "multilevel_config": asdict(config),
        "coil_config": asdict(coil),
        "capture_search_config": asdict(search),
        "configuration_invariance_audit": configuration_invariance_audit(config, apparatus, coil, search),
        "geometry_sampler": (
            f"seed {search.seed}; directions uniform in solid angle over "
            + (
                "one symmetry octant; each "
                if search.phase_space == "octant"
                else "the complete 4 pi sphere; each "
            )
            + "perpendicular disc contains independent uniform-area points with s=R*sqrt(U) "
            "and uniform azimuth; all velocities are parallel to the inward disc normal"
        ),
        "paired_geometry_note": (
            "The fixed seed and signed launch-geometry CSV make the random geometry "
            "exactly reproducible."
        ),
        "trapped_criterion": (
            "continuous residence in the central 2 mm-radius core for at least 5 ms, "
            "or two core entries with an intervening exit"
        ),
        "cross_section_estimator": (
            "projected sampling-disc area times capture fraction at each speed, averaged "
            "over direction discs; uncertainty is clustered by direction disc"
        ),
        "loading_rate_estimator": (
            "integrate one cross-section spectrum per direction disc and report the mean, "
            "disc-cluster SEM, and Student-t 95% interval"
        ),
        "worker_count": worker_count,
        "checkpoint_every_completed_samples": checkpoint_every,
        "progress_every_completed_samples": progress_every,
        "output_manifest": _output_manifest(paths),
        "git": _git_provenance(multilevel_mot_paths()["root"]),
        "limitations": [
            "The population-rate approximation omits optical coherences and sub-Doppler physics.",
            "Capture-speed binary search assumes a locally monotonic trapped/untrapped boundary.",
            (
                "The one-octant convention is used although gravity breaks exact z-reflection symmetry."
                if search.phase_space == "octant"
                else "Full-sphere direction sampling avoids assuming octant reflection symmetry in the presence of gravity."
            ),
            f"The {1e3 * search.disc_radius_m:g} mm sampling-disc radius caps the projected capture cross section.",
            "Direction-cluster uncertainty does not include model-systematic or convergence uncertainty.",
            "Quantitative results remain provisional until timestep, timeout, and event-engine comparisons are documented.",
        ],
    }


def analyze_completed_samples(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    paths: StudyPaths,
    *,
    signature: str,
    geometry_hash: str,
    plot_context: str | None = None,
) -> dict[str, object]:
    expected = search.disc_count * search.points_per_disc
    if len(samples) != expected:
        raise ValueError(f"analysis requires {expected} samples, received {len(samples)}")
    counts = Counter(sample.disc_index for sample in samples)
    if set(counts) != set(range(search.disc_count)):
        raise ValueError("analysis is missing one or more direction discs")
    if any(count != search.points_per_disc for count in counts.values()):
        raise ValueError("analysis does not contain the requested points per disc")

    spectrum = calculate_clustered_cross_section(samples, search)
    by_disc, loading = calculate_disc_clustered_loading(samples, search, spectrum)
    _atomic_write_csv(paths.spectrum_csv, spectrum, SPECTRUM_FIELDNAMES)
    _atomic_write_csv(paths.loading_by_disc_csv, by_disc, LOADING_BY_DISC_FIELDNAMES)
    loading_payload = {
        **loading,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "run_signature_sha256": signature,
        "geometry_sha256": geometry_hash,
        "spectrum_csv": str(paths.spectrum_csv.resolve()),
        "samples_csv": str(paths.final_samples_csv.resolve()),
        "loading_rate_by_disc_csv": str(paths.loading_by_disc_csv.resolve()),
    }
    _atomic_write_json(paths.loading_json, loading_payload)
    context = plot_context or "27 mW cooling; 0.1 mW repump"
    plot_clustered_cross_section(
        spectrum,
        paths.cross_section_png,
        title=f"24-State MOT Capture Cross Section ({context})",
    )
    plot_capture_velocity_vs_impact_parameter(
        samples,
        paths.impact_parameter_png,
        title=f"24-State MOT Capture Velocity ({context})",
    )
    plot_loading_rate_by_disc(
        by_disc,
        loading,
        paths.loading_by_disc_png,
        title=f"24-State MOT Loading Rate by Direction ({context})",
    )

    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    resolution = np.asarray([sample.velocity_resolution_m_per_s for sample in samples], dtype=float)
    lower_counts = Counter(sample.lower_classification for sample in samples)
    upper_counts = Counter(sample.upper_classification for sample in samples)
    valid = sum(
        (sample.capture_velocity_m_per_s == 0.0 or sample.lower_classification in TRAPPED_TERMINATION_REASONS)
        and sample.upper_classification in VALID_UNTRAPPED_TERMINATION_REASONS
        for sample in samples
    )
    summary = {
        "model": "24-state repumper-included multilevel population-rate MOT",
        "run_signature_sha256": signature,
        "geometry_sha256": geometry_hash,
        "sample_count": len(samples),
        "expected_sample_count": expected,
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "capture_velocity_mean_m_per_s": float(np.mean(capture)),
        "capture_velocity_sample_std_m_per_s": float(np.std(capture, ddof=1)) if len(capture) > 1 else 0.0,
        "capture_velocity_min_m_per_s": float(np.min(capture)),
        "capture_velocity_max_m_per_s": float(np.max(capture)),
        "zero_capture_velocity_count": int(np.count_nonzero(capture == 0.0)),
        "velocity_resolution_mean_m_per_s": float(np.mean(resolution)),
        "velocity_resolution_max_m_per_s": float(np.max(resolution)),
        "valid_bracket_count": int(valid),
        "lower_classification_counts": dict(sorted(lower_counts.items())),
        "upper_classification_counts": dict(sorted(upper_counts.items())),
        "loading_rate": loading_payload,
        "search_config": asdict(search),
        "outputs": _output_manifest(paths),
    }
    _atomic_write_json(paths.capture_summary_json, summary)
    return summary


_WORKER_MODEL: RateEquationModel | None = None
_WORKER_BEAMS: list[MOTBeam] | None = None
_WORKER_COIL: AntiHelmholtzCoilConfig | None = None
_WORKER_CONFIG: MultilevelMOTConfig | None = None
_WORKER_SEARCH: RateCaptureSearchConfig | None = None


def _initialize_capture_worker(
    config: MultilevelMOTConfig,
    apparatus: MOTApparatusConfig,
    coil: AntiHelmholtzCoilConfig,
    search: RateCaptureSearchConfig,
) -> None:
    """Build and cache the immutable 24-state model once in each worker."""

    global _WORKER_MODEL, _WORKER_BEAMS, _WORKER_COIL, _WORKER_CONFIG, _WORKER_SEARCH
    _WORKER_CONFIG = config
    _WORKER_MODEL = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    _WORKER_BEAMS = build_multilevel_mot_beams(apparatus_config=apparatus, config=config)
    _validate_built_model_inputs(
        config,
        _WORKER_BEAMS,
        apparatus.cooling.power_w_per_beam,
        config.repump_power_w_per_beam,
        apparatus.cooling.detuning_hz,
    )
    if _WORKER_MODEL.state_count != 24 or _WORKER_MODEL.ground_count != 8 or _WORKER_MODEL.excited_count != 16:
        raise RuntimeError("worker did not initialize the authoritative 24-state model")
    _WORKER_COIL = coil
    _WORKER_SEARCH = search


def _capture_worker(point: PointSample) -> CaptureVelocitySample:
    if any(
        value is None
        for value in (_WORKER_MODEL, _WORKER_BEAMS, _WORKER_COIL, _WORKER_CONFIG, _WORKER_SEARCH)
    ):
        raise RuntimeError("capture worker was not initialized")
    return find_multilevel_capture_velocity(
        point,
        _WORKER_SEARCH,
        model=_WORKER_MODEL,
        beams=_WORKER_BEAMS,
        coil_config=_WORKER_COIL,
        config=_WORKER_CONFIG,
    )


def _validate_resume_state(paths: StudyPaths, signature: str, geometry_hash: str) -> dict[str, object]:
    if not paths.metadata_json.is_file() or not paths.geometry_csv.is_file():
        raise ValueError("resume requested but signed metadata or geometry is missing")
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("run_signature_sha256") != signature:
        raise ValueError("resume signature mismatch: physics, configuration, or geometry changed")
    if _sha256_file(paths.geometry_csv) != geometry_hash:
        raise ValueError("resume geometry mismatch: launch_geometry.csv changed or is corrupt")
    return metadata


def _load_existing_results(
    paths: StudyPaths, points: Sequence[PointSample]
) -> dict[tuple[int, int], CaptureVelocitySample]:
    candidate = paths.final_samples_csv if paths.final_samples_csv.is_file() else paths.partial_samples_csv
    if not candidate.is_file():
        return {}
    return validate_checkpoint_samples(load_capture_velocity_samples(candidate), points)


def _reset_known_study_outputs(paths: StudyPaths) -> None:
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
    results: Mapping[tuple[int, int], CaptureVelocitySample],
    metadata: dict[str, object],
    *,
    elapsed_wall_time_s: float,
    eta_s: float | None,
    status: str = "running",
    error: str | None = None,
) -> None:
    save_samples_atomic(paths.partial_samples_csv, results.values())
    updated = dict(metadata)
    updated.update(
        {
            "status": status,
            "updated_utc": _utc_now(),
            "completed_sample_count": len(results),
            "completion_fraction": len(results) / int(metadata["expected_sample_count"]),
            "elapsed_wall_time_s": elapsed_wall_time_s,
            "eta_s": eta_s,
        }
    )
    if error is not None:
        updated["last_error"] = error
    _atomic_write_json(paths.metadata_json, updated)
    metadata.clear()
    metadata.update(updated)


def run_power_loading_study(
    search_config: RateCaptureSearchConfig | None = None,
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
    analyze_only: bool = False,
    cooling_power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
    repump_power_w_per_beam: float = REPUMP_POWER_W_PER_BEAM,
    cooling_detuning_hz: float = COOLING_DETUNING_HZ,
    study_name: str = STUDY_NAME,
    progress_every: int = PROGRESS_EVERY,
    checkpoint_every: int = CHECKPOINT_EVERY,
    plot_context: str | None = None,
) -> dict[str, object]:
    """Run or resume a checkpointed 24-state capture/loading study."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if progress_every <= 0 or checkpoint_every <= 0:
        raise ValueError("progress_every and checkpoint_every must be positive")
    if not study_name.strip():
        raise ValueError("study_name must not be empty")
    if analyze_only and not resume:
        raise ValueError("--analyze-only requires resume mode")
    search = search_config or default_study_search_config()
    if search.required_core_entries != 2 or not np.isclose(search.bounded_core_residence_s, 5.0e-3):
        raise ValueError("production trapping requires two entries or 5 ms core residence")
    default_paths = default_study_paths()
    paths = StudyPaths(
        statistics=Path(output_directory) if output_directory is not None else default_paths.statistics,
        figures=Path(figure_directory) if figure_directory is not None else default_paths.figures,
    )
    paths.statistics.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)

    config, apparatus, beams = build_27mw_multilevel_configuration(
        cooling_power_w_per_beam=cooling_power_w_per_beam,
        repump_power_w_per_beam=repump_power_w_per_beam,
        cooling_detuning_hz=cooling_detuning_hz,
    )
    coil = default_anti_helmholtz_config()
    discs, points = generate_study_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    signature_payload = study_signature_payload(
        config,
        apparatus,
        coil,
        search,
        geometry_hash,
        study_name=study_name,
    )
    signature = study_signature(signature_payload)
    expected = search.disc_count * search.points_per_disc

    products = (
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
    )
    existing_state = any(path.exists() for path in products)
    if not resume and existing_state:
        _reset_known_study_outputs(paths)
        existing_state = False
    if resume and existing_state:
        prior_metadata = _validate_resume_state(paths, signature, geometry_hash)
        results = _load_existing_results(paths, points)
    else:
        prior_metadata = None
        _atomic_write_text(paths.geometry_csv, geometry_text)
        results = {}
    if len(results) > expected:
        raise ValueError("checkpoint contains more samples than requested geometry")

    prior_elapsed = float(prior_metadata.get("elapsed_wall_time_s", 0.0)) if prior_metadata else 0.0
    started_utc = str(prior_metadata.get("started_utc")) if prior_metadata else _utc_now()
    metadata = build_run_metadata(
        config,
        apparatus,
        beams,
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
        study_name=study_name,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
    )
    _atomic_write_json(paths.metadata_json, metadata)

    resolved_plot_context = plot_context or (
        f"{1e3 * cooling_power_w_per_beam:g} mW cooling; "
        f"{1e3 * repump_power_w_per_beam:g} mW repump; "
        f"cooling detuning {cooling_detuning_hz / 1e6:g} MHz"
    )

    if analyze_only:
        if len(results) != expected:
            raise ValueError(f"--analyze-only requires {expected} samples; found {len(results)}")
        samples = _sorted_samples(results.values())
        save_samples_atomic(paths.final_samples_csv, samples)
        summary = analyze_completed_samples(
            samples,
            search,
            paths,
            signature=signature,
            geometry_hash=geometry_hash,
            plot_context=resolved_plot_context,
        )
        metadata.update(
            {
                "status": "completed",
                "updated_utc": _utc_now(),
                "completed_sample_count": expected,
                "completion_fraction": 1.0,
                "analysis_only_invocation": True,
                "loading_rate": summary["loading_rate"],
            }
        )
        _atomic_write_json(paths.metadata_json, metadata)
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    point_map = {(point.disc_index, point.point_index): point for point in points}
    missing = [point for key, point in point_map.items() if key not in results]
    progress_label = (
        f"24-state MOT {1e3 * cooling_power_w_per_beam:g} mW cooling, "
        f"{1e3 * repump_power_w_per_beam:g} mW repump, "
        f"{cooling_detuning_hz / 1e6:g} MHz cooling detuning, "
        f"{1e3 * search.disc_radius_m:g} mm disc, "
        f"{search.phase_space.replace('_', ' ')}"
    )
    print(
        f"[{progress_label}] {len(results)}/{expected} complete; "
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
            _initialize_capture_worker(config, apparatus, coil, search)
            iterator = (_capture_worker(point) for point in missing)
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_capture_worker,
                initargs=(config, apparatus, coil, search),
            )
            futures = {executor.submit(_capture_worker, point): point for point in missing}
            iterator = (future.result() for future in as_completed(futures))
        for sample in iterator:
            key = (sample.disc_index, sample.point_index)
            _validate_sample_against_point(sample, point_map[key])
            results[key] = sample
            completed = len(results)
            segment_elapsed = perf_counter() - segment_start
            new_completed = completed - initial_completed
            rate = new_completed / segment_elapsed if segment_elapsed > 0.0 else 0.0
            remaining = expected - completed
            last_eta = remaining / rate if rate > 0.0 else None
            if completed % progress_every == 0 or completed == expected:
                eta_text = "unknown" if last_eta is None else f"{last_eta / 3600.0:.2f} h"
                print(
                    f"[{progress_label}] {completed}/{expected}; "
                    f"disc {sample.disc_index + 1}/{search.disc_count}, "
                    f"point {sample.point_index + 1}/{search.points_per_disc}; "
                    f"vc={sample.capture_velocity_m_per_s:.3f} m/s; ETA={eta_text}",
                    flush=True,
                )
            if completed % checkpoint_every == 0:
                _save_running_checkpoint(
                    paths,
                    results,
                    metadata,
                    elapsed_wall_time_s=prior_elapsed + segment_elapsed,
                    eta_s=last_eta,
                )
                print(
                    f"[{progress_label}] checkpoint saved at {completed}/{expected}",
                    flush=True,
                )
    except BaseException as exc:
        execution_failed = True
        _save_running_checkpoint(
            paths,
            results,
            metadata,
            elapsed_wall_time_s=prior_elapsed + (perf_counter() - segment_start),
            eta_s=last_eta,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=execution_failed)

    if len(results) != expected:
        raise RuntimeError(f"sampling incomplete: obtained {len(results)} of {expected}")
    samples = _sorted_samples(results.values())
    save_samples_atomic(paths.partial_samples_csv, samples)
    save_samples_atomic(paths.final_samples_csv, samples)
    summary = analyze_completed_samples(
        samples,
        search,
        paths,
        signature=signature,
        geometry_hash=geometry_hash,
        plot_context=resolved_plot_context,
    )
    metadata.update(
        {
            "status": "completed",
            "updated_utc": _utc_now(),
            "completed_sample_count": expected,
            "completion_fraction": 1.0,
            "elapsed_wall_time_s": prior_elapsed + (perf_counter() - segment_start),
            "eta_s": 0.0,
            "loading_rate": summary["loading_rate"],
        }
    )
    _atomic_write_json(paths.metadata_json, metadata)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the default 27 mW cooling + 0.1 mW repump, 50-disc x 50-point "
            "24-state MOT capture/loading study at a selectable red cooling detuning"
        )
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--cooling-detuning-mhz",
        type=float,
        default=COOLING_DETUNING_HZ / 1.0e6,
        help="ordinary-frequency cooling detuning in MHz; must be negative",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_power_loading_study(
        worker_count=args.workers,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=args.resume,
        analyze_only=args.analyze_only,
        cooling_detuning_hz=1.0e6 * args.cooling_detuning_mhz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_EVERY",
    "COOLING_DETUNING_HZ",
    "COOLING_POWER_W_PER_BEAM",
    "DEFAULT_WORKER_COUNT",
    "DISC_COUNT",
    "POINTS_PER_DISC",
    "RANDOM_SEED",
    "REPUMP_POWER_W_PER_BEAM",
    "STUDY_NAME",
    "StudyPaths",
    "analyze_completed_samples",
    "build_27mw_multilevel_configuration",
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
