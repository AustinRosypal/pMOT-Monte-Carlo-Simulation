"""Monte Carlo capture-velocity sampling for the simplified two-level MOT."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from math import cos
from math import pi
from math import sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..forces import AtomState
from ..mot.magnetic_fields import default_anti_helmholtz_config
from .configuration import default_simple_mot_apparatus
from .configuration import default_simple_mot_config
from .configuration import simple_mot_paths
from .plotting import draw_simple_mot_beam_volumes
from .simulation import build_simple_mot_beams
from .simulation import rk4_step
from .simulation import SimpleMOTBeam
from .simulation import SimpleMOTConfig


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CaptureSearchConfig:
    """Configuration for capture-velocity Monte Carlo sampling."""

    radial_distance_m: float = 15.0e-3
    disc_radius_m: float = 12.0e-3
    initial_velocity_guess_m_per_s: float = 20.0
    velocity_tolerance_m_per_s: float = 0.25
    disc_count: int = 100
    points_per_disc: int = 100
    max_simulation_time_s: float = 50.0e-3
    time_step_s: float = 5.0e-6
    trap_core_radius_m: float = 2.0e-3
    escape_radius_m: float = 30.0e-3
    required_core_entries: int = 2
    bounded_core_residence_s: float = 5.0e-3
    max_bracket_iterations: int = 24
    max_search_iterations: int = 24
    include_center_point: bool = False
    analysis_velocity_step_m_per_s: float = 0.25
    analysis_s_bin_count: int = 24
    analysis_velocity_min_m_per_s: float = 1.0
    analysis_velocity_max_m_per_s: float = 30.0
    seed: int = 0
    save_every: int = 25


@dataclass(frozen=True, slots=True)
class DiscSample:
    """One sampled incident disc."""

    disc_index: int
    theta_rad: float
    phi_rad: float
    outward_unit_vector: Vec3
    incident_unit_vector: Vec3
    center_position_m: Vec3
    basis_u: Vec3
    basis_v: Vec3


@dataclass(frozen=True, slots=True)
class PointSample:
    """One sampled launch point on one incident disc."""

    disc_index: int
    point_index: int
    theta_rad: float
    phi_rad: float
    theta_prime_rad: float
    s_m: float
    radial_distance_m: float
    initial_position_m: Vec3
    incident_unit_vector: Vec3
    launch_axis_unit_vector: Vec3


@dataclass(frozen=True, slots=True)
class VelocitySpectrumSample:
    """Capture statistics aggregated on a velocity grid."""

    velocity_m_per_s: float
    captured_count: int
    launched_count: int
    capture_fraction: float
    capture_cross_section_m2: float


@dataclass(frozen=True, slots=True)
class TrajectoryClassification:
    """Terminal classification for one launch trajectory."""

    trapped: bool
    termination_reason: str
    entered_trap_core: bool
    core_entry_count: int
    elapsed_time_s: float
    minimum_radius_m: float
    final_radius_m: float
    final_position_m: Vec3
    final_velocity_m_per_s: Vec3


@dataclass(frozen=True, slots=True)
class CaptureVelocitySample:
    """Capture-velocity result for one sampled launch point."""

    disc_index: int
    point_index: int
    theta_rad: float
    phi_rad: float
    theta_prime_rad: float
    s_m: float
    radial_distance_m: float
    initial_position_m: Vec3
    incident_unit_vector: Vec3
    capture_velocity_m_per_s: float
    velocity_resolution_m_per_s: float
    trapped_velocity_lower_m_per_s: float
    untrapped_velocity_upper_m_per_s: float
    lower_classification: str
    upper_classification: str
    lower_entered_trap_core: bool
    upper_entered_trap_core: bool
    lower_core_entry_count: int
    upper_core_entry_count: int


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vector: Vec3) -> float:
    return float(np.sqrt(dot(vector, vector)))


def normalize(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return scale(1.0 / magnitude, vector)


def choose_transverse_basis(direction: Vec3) -> tuple[Vec3, Vec3]:
    """Return orthonormal vectors spanning the plane perpendicular to direction."""

    direction_hat = normalize(direction)
    trial = (0.0, 0.0, 1.0)
    if abs(dot(direction_hat, trial)) > 0.95:
        trial = (0.0, 1.0, 0.0)
    basis_u = normalize(cross(direction_hat, trial))
    basis_v = normalize(cross(direction_hat, basis_u))
    return basis_u, basis_v


def spherical_octant_direction(theta_rad: float, phi_rad: float) -> Vec3:
    """Return a first-octant unit vector from polar and azimuthal angles."""

    return (
        sin(theta_rad) * cos(phi_rad),
        sin(theta_rad) * sin(phi_rad),
        cos(theta_rad),
    )


def sample_octant_direction(rng: np.random.Generator) -> tuple[float, float, Vec3]:
    """Sample one first-octant direction uniformly in solid angle."""

    cos_theta = float(rng.uniform(0.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    phi_rad = float(rng.uniform(0.0, 0.5 * pi))
    return theta_rad, phi_rad, spherical_octant_direction(theta_rad, phi_rad)


def sample_full_sphere_direction(rng: np.random.Generator) -> tuple[float, float, Vec3]:
    """Sample one direction uniformly over the complete solid angle."""

    cos_theta = float(rng.uniform(-1.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    phi_rad = float(rng.uniform(0.0, 2.0 * pi))
    return theta_rad, phi_rad, spherical_octant_direction(theta_rad, phi_rad)


def sample_incident_disc(
    disc_index: int,
    radial_distance_m: float,
    rng: np.random.Generator,
) -> DiscSample:
    """Sample one incident disc within the first octant."""

    theta_rad, phi_rad, outward_unit_vector = sample_octant_direction(rng)
    incident_unit_vector = scale(-1.0, outward_unit_vector)
    basis_u, basis_v = choose_transverse_basis(incident_unit_vector)
    center_position_m = scale(radial_distance_m, outward_unit_vector)
    return DiscSample(
        disc_index=disc_index,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        outward_unit_vector=outward_unit_vector,
        incident_unit_vector=incident_unit_vector,
        center_position_m=center_position_m,
        basis_u=basis_u,
        basis_v=basis_v,
    )


def sample_incident_disc_full_sphere(
    disc_index: int,
    radial_distance_m: float,
    rng: np.random.Generator,
) -> DiscSample:
    """Sample one incident disc with its normal uniform over 4 pi steradians."""

    theta_rad, phi_rad, outward_unit_vector = sample_full_sphere_direction(rng)
    incident_unit_vector = scale(-1.0, outward_unit_vector)
    basis_u, basis_v = choose_transverse_basis(incident_unit_vector)
    center_position_m = scale(radial_distance_m, outward_unit_vector)
    return DiscSample(
        disc_index=disc_index,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        outward_unit_vector=outward_unit_vector,
        incident_unit_vector=incident_unit_vector,
        center_position_m=center_position_m,
        basis_u=basis_u,
        basis_v=basis_v,
    )


def build_incident_disc_from_angles(
    disc_index: int,
    radial_distance_m: float,
    theta_rad: float,
    phi_rad: float,
) -> DiscSample:
    """Build one incident disc from user-specified spherical angles."""

    outward_unit_vector = spherical_octant_direction(theta_rad, phi_rad)
    incident_unit_vector = scale(-1.0, outward_unit_vector)
    basis_u, basis_v = choose_transverse_basis(incident_unit_vector)
    center_position_m = scale(radial_distance_m, outward_unit_vector)
    return DiscSample(
        disc_index=disc_index,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        outward_unit_vector=outward_unit_vector,
        incident_unit_vector=incident_unit_vector,
        center_position_m=center_position_m,
        basis_u=basis_u,
        basis_v=basis_v,
    )


def sample_disc_points(
    disc: DiscSample,
    points_per_disc: int,
    disc_radius_m: float,
    include_center_point: bool,
    rng: np.random.Generator,
) -> list[PointSample]:
    """Sample independent, uniformly distributed launch points over a disc.

    Production Monte Carlo uses ``include_center_point=False`` so every point
    has ``s = R * sqrt(U)`` and an independent uniform azimuth.  The explicit
    ``include_center_point=True`` opt-in is retained for legacy diagnostics;
    only that mode prepends the measure-zero center.  The rim is never forced.
    """

    points: list[PointSample] = []
    if points_per_disc <= 0:
        return points

    center_position_m = disc.center_position_m
    launch_axis_unit_vector = normalize(scale(-1.0, center_position_m))
    if include_center_point:
        points.append(
            PointSample(
                disc_index=disc.disc_index,
                point_index=0,
                theta_rad=disc.theta_rad,
                phi_rad=disc.phi_rad,
                theta_prime_rad=0.0,
                s_m=0.0,
                radial_distance_m=radial_distance_magnitude(disc.center_position_m),
                initial_position_m=center_position_m,
                incident_unit_vector=launch_axis_unit_vector,
                launch_axis_unit_vector=launch_axis_unit_vector,
            )
        )

    point_start_index = int(include_center_point)
    for point_index in range(point_start_index, points_per_disc):
        theta_prime_rad = float(rng.uniform(0.0, 2.0 * pi))
        area_fraction = float(rng.uniform(0.0, 1.0))
        s_m = disc_radius_m * float(np.sqrt(area_fraction))
        offset = add(
            scale(s_m * float(np.cos(theta_prime_rad)), disc.basis_u),
            scale(s_m * float(np.sin(theta_prime_rad)), disc.basis_v),
        )
        initial_position_m = add(disc.center_position_m, offset)
        points.append(
            PointSample(
                disc_index=disc.disc_index,
                point_index=point_index,
                theta_rad=disc.theta_rad,
                phi_rad=disc.phi_rad,
                theta_prime_rad=theta_prime_rad,
                s_m=s_m,
                radial_distance_m=radial_distance_magnitude(disc.center_position_m),
                initial_position_m=initial_position_m,
                incident_unit_vector=launch_axis_unit_vector,
                launch_axis_unit_vector=launch_axis_unit_vector,
            )
        )
    return points


def radial_distance_magnitude(position_m: Vec3) -> float:
    """Return the distance of one point from the origin."""

    return norm(position_m)


def classify_trajectory(
    beams: list[SimpleMOTBeam],
    point: PointSample,
    incident_speed_m_per_s: float,
    coil_config,
    simple_config: SimpleMOTConfig,
    search_config: CaptureSearchConfig,
) -> TrajectoryClassification:
    """Classify a launch trajectory as trapped, escaped, or timed out."""

    atom_state = AtomState(
        position_m=point.initial_position_m,
        velocity_m_per_s=scale(incident_speed_m_per_s, point.incident_unit_vector),
    )
    if search_config.time_step_s <= 0.0:
        raise ValueError("time_step_s must be positive")
    if search_config.max_simulation_time_s < 0.0:
        raise ValueError("max_simulation_time_s must be non-negative")
    if search_config.bounded_core_residence_s < 0.0:
        raise ValueError("bounded_core_residence_s must be non-negative")

    max_steps = int(np.ceil(search_config.max_simulation_time_s / search_config.time_step_s))
    minimum_radius_m = norm(atom_state.position_m)
    was_inside_trap_core = minimum_radius_m <= search_config.trap_core_radius_m
    entered_trap_core = was_inside_trap_core
    core_entry_count = int(was_inside_trap_core)
    inside_since_s: float | None = 0.0 if was_inside_trap_core else None
    elapsed_time_s = 0.0

    # Evaluate both the initial state and the state at the requested timeout.
    # Residence time is accumulated from sampled core entry until a sampled
    # exit resets it; this is conservative at a boundary crossed within a step.
    for _ in range(max_steps + 1):
        radius_m = norm(atom_state.position_m)
        minimum_radius_m = min(minimum_radius_m, radius_m)
        inside_trap_core = radius_m <= search_config.trap_core_radius_m
        entered_trap_core = entered_trap_core or inside_trap_core
        if inside_trap_core and not was_inside_trap_core:
            core_entry_count += 1
            inside_since_s = elapsed_time_s
        elif not inside_trap_core:
            inside_since_s = None
        was_inside_trap_core = inside_trap_core
        radial_velocity = dot(atom_state.position_m, atom_state.velocity_m_per_s) / max(radius_m, 1.0e-15)

        if core_entry_count >= search_config.required_core_entries:
            return TrajectoryClassification(
                trapped=True,
                termination_reason="two_core_entries",
                entered_trap_core=entered_trap_core,
                core_entry_count=core_entry_count,
                elapsed_time_s=elapsed_time_s,
                minimum_radius_m=minimum_radius_m,
                final_radius_m=radius_m,
                final_position_m=atom_state.position_m,
                final_velocity_m_per_s=atom_state.velocity_m_per_s,
            )

        if (
            inside_since_s is not None
            and elapsed_time_s - inside_since_s >= search_config.bounded_core_residence_s
        ):
            return TrajectoryClassification(
                trapped=True,
                termination_reason="bounded_core_residence",
                entered_trap_core=entered_trap_core,
                core_entry_count=core_entry_count,
                elapsed_time_s=elapsed_time_s,
                minimum_radius_m=minimum_radius_m,
                final_radius_m=radius_m,
                final_position_m=atom_state.position_m,
                final_velocity_m_per_s=atom_state.velocity_m_per_s,
            )

        if radius_m >= search_config.escape_radius_m and radial_velocity > 0.0:
            return TrajectoryClassification(
                trapped=False,
                termination_reason="escaped",
                entered_trap_core=entered_trap_core,
                core_entry_count=core_entry_count,
                elapsed_time_s=elapsed_time_s,
                minimum_radius_m=minimum_radius_m,
                final_radius_m=radius_m,
                final_position_m=atom_state.position_m,
                final_velocity_m_per_s=atom_state.velocity_m_per_s,
            )

        if elapsed_time_s >= search_config.max_simulation_time_s - 1.0e-15:
            break

        step_time_s = min(
            search_config.time_step_s,
            search_config.max_simulation_time_s - elapsed_time_s,
        )
        atom_state, _, _, _ = rk4_step(
            beams,
            atom_state,
            step_time_s,
            coil_config,
            simple_config=simple_config,
        )
        elapsed_time_s += step_time_s

        if not np.all(np.isfinite(np.asarray(atom_state.position_m))) or not np.all(
            np.isfinite(np.asarray(atom_state.velocity_m_per_s))
        ):
            return TrajectoryClassification(
                trapped=False,
                termination_reason="non_finite",
                entered_trap_core=entered_trap_core,
                core_entry_count=core_entry_count,
                elapsed_time_s=elapsed_time_s,
                minimum_radius_m=minimum_radius_m,
                final_radius_m=norm(atom_state.position_m),
                final_position_m=atom_state.position_m,
                final_velocity_m_per_s=atom_state.velocity_m_per_s,
            )

    return TrajectoryClassification(
        trapped=False,
        termination_reason="timeout",
        entered_trap_core=entered_trap_core,
        core_entry_count=core_entry_count,
        elapsed_time_s=elapsed_time_s,
        minimum_radius_m=minimum_radius_m,
        final_radius_m=norm(atom_state.position_m),
        final_position_m=atom_state.position_m,
        final_velocity_m_per_s=atom_state.velocity_m_per_s,
    )


def bracket_capture_velocity(
    beams: list[SimpleMOTBeam],
    point: PointSample,
    coil_config,
    simple_config: SimpleMOTConfig,
    search_config: CaptureSearchConfig,
) -> tuple[float | None, float, dict[float, TrajectoryClassification]]:
    """Find a trapped/untrapped velocity bracket for one launch point."""

    evaluations: dict[float, TrajectoryClassification] = {}

    def evaluate(speed_m_per_s: float) -> TrajectoryClassification:
        rounded = round(speed_m_per_s, 12)
        if rounded not in evaluations:
            evaluations[rounded] = classify_trajectory(
                beams,
                point,
                rounded,
                coil_config,
                simple_config,
                search_config,
            )
        return evaluations[rounded]

    trial = max(0.0, search_config.initial_velocity_guess_m_per_s)
    if evaluate(trial).trapped:
        lower = trial
        upper = max(1.0, trial)
        for _ in range(search_config.max_bracket_iterations):
            upper *= 2.0
            if not evaluate(upper).trapped:
                return lower, upper, evaluations
        raise RuntimeError("failed to find an untrapped upper bracket")

    upper = trial
    lower = upper
    for iteration in range(search_config.max_bracket_iterations):
        lower = 0.0 if iteration == search_config.max_bracket_iterations - 1 else 0.5 * lower
        if evaluate(lower).trapped:
            return lower, upper, evaluations
        if lower <= 1.0e-6:
            break
    if evaluate(0.0).trapped:
        return 0.0, upper, evaluations
    return None, upper, evaluations


def find_capture_velocity(
    beams: list[SimpleMOTBeam],
    point: PointSample,
    coil_config,
    simple_config: SimpleMOTConfig,
    search_config: CaptureSearchConfig,
) -> CaptureVelocitySample:
    """Find the capture velocity for one sampled launch point."""

    lower_speed, upper_speed, evaluations = bracket_capture_velocity(
        beams,
        point,
        coil_config,
        simple_config,
        search_config,
    )

    if lower_speed is None:
        zero_classification = evaluations[round(0.0, 12)]
        upper_classification = evaluations[round(upper_speed, 12)]
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
            capture_velocity_m_per_s=0.0,
            velocity_resolution_m_per_s=upper_speed,
            trapped_velocity_lower_m_per_s=0.0,
            untrapped_velocity_upper_m_per_s=upper_speed,
            lower_classification=zero_classification.termination_reason,
            upper_classification=upper_classification.termination_reason,
            lower_entered_trap_core=zero_classification.entered_trap_core,
            upper_entered_trap_core=upper_classification.entered_trap_core,
            lower_core_entry_count=zero_classification.core_entry_count,
            upper_core_entry_count=upper_classification.core_entry_count,
        )

    for _ in range(search_config.max_search_iterations):
        if upper_speed - lower_speed <= search_config.velocity_tolerance_m_per_s:
            break
        midpoint = round(0.5 * (lower_speed + upper_speed), 12)
        midpoint_classification = evaluations.get(midpoint)
        if midpoint_classification is None:
            midpoint_classification = classify_trajectory(
                beams,
                point,
                midpoint,
                coil_config,
                simple_config,
                search_config,
            )
            evaluations[midpoint] = midpoint_classification
        if midpoint_classification.trapped:
            lower_speed = midpoint
        else:
            upper_speed = midpoint

    lower_classification = evaluations[round(lower_speed, 12)]
    upper_classification = evaluations[round(upper_speed, 12)]
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
        capture_velocity_m_per_s=lower_speed,
        velocity_resolution_m_per_s=upper_speed - lower_speed,
        trapped_velocity_lower_m_per_s=lower_speed,
        untrapped_velocity_upper_m_per_s=upper_speed,
        lower_classification=lower_classification.termination_reason,
        upper_classification=upper_classification.termination_reason,
        lower_entered_trap_core=lower_classification.entered_trap_core,
        upper_entered_trap_core=upper_classification.entered_trap_core,
        lower_core_entry_count=lower_classification.core_entry_count,
        upper_core_entry_count=upper_classification.core_entry_count,
    )


def summarize_capture_velocity_samples(
    samples: list[CaptureVelocitySample],
    search_config: CaptureSearchConfig,
) -> dict[str, object]:
    """Return summary statistics for a completed sampling run."""

    capture_velocities = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    resolutions = np.asarray([sample.velocity_resolution_m_per_s for sample in samples], dtype=float)
    return {
        "sample_count": int(len(samples)),
        "disc_count": int(search_config.disc_count),
        "points_per_disc": int(search_config.points_per_disc),
        "capture_velocity_mean_m_per_s": float(np.mean(capture_velocities)) if len(samples) else None,
        "capture_velocity_std_m_per_s": float(np.std(capture_velocities)) if len(samples) else None,
        "capture_velocity_min_m_per_s": float(np.min(capture_velocities)) if len(samples) else None,
        "capture_velocity_max_m_per_s": float(np.max(capture_velocities)) if len(samples) else None,
        "resolution_mean_m_per_s": float(np.mean(resolutions)) if len(samples) else None,
        "search_config": asdict(search_config),
    }


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


def velocity_grid_from_samples(
    samples: list[CaptureVelocitySample],
    velocity_step_m_per_s: float,
    velocity_min_m_per_s: float,
    velocity_max_m_per_s: float,
) -> np.ndarray:
    """Build a velocity grid for post-processing from saved capture thresholds."""

    if velocity_step_m_per_s <= 0.0:
        raise ValueError("velocity_step_m_per_s must be positive")
    if velocity_min_m_per_s < 0.0:
        raise ValueError("velocity_min_m_per_s must be non-negative")
    if velocity_max_m_per_s <= velocity_min_m_per_s:
        raise ValueError("velocity_max_m_per_s must exceed velocity_min_m_per_s")
    sampled_upper_bound = max((sample.untrapped_velocity_upper_m_per_s for sample in samples), default=velocity_max_m_per_s)
    grid_start = velocity_step_m_per_s * np.floor(velocity_min_m_per_s / velocity_step_m_per_s)
    grid_stop = velocity_step_m_per_s * np.ceil(max(velocity_max_m_per_s, sampled_upper_bound) / velocity_step_m_per_s)
    return np.arange(grid_start, grid_stop + 0.5 * velocity_step_m_per_s, velocity_step_m_per_s, dtype=float)


def capture_spectrum_from_samples(
    samples: list[CaptureVelocitySample],
    disc_radius_m: float,
    velocity_step_m_per_s: float,
    velocity_min_m_per_s: float,
    velocity_max_m_per_s: float,
) -> list[VelocitySpectrumSample]:
    """Aggregate capture counts and capture cross section versus velocity."""

    velocity_grid = velocity_grid_from_samples(samples, velocity_step_m_per_s, velocity_min_m_per_s, velocity_max_m_per_s)
    launched_count = len(samples)
    disc_area_m2 = pi * disc_radius_m**2
    capture_velocities = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    spectrum: list[VelocitySpectrumSample] = []
    for velocity_m_per_s in velocity_grid:
        captured_count = int(np.count_nonzero(capture_velocities >= velocity_m_per_s - 1.0e-12))
        capture_fraction = captured_count / launched_count if launched_count else 0.0
        spectrum.append(
            VelocitySpectrumSample(
                velocity_m_per_s=float(velocity_m_per_s),
                captured_count=captured_count,
                launched_count=launched_count,
                capture_fraction=capture_fraction,
                capture_cross_section_m2=disc_area_m2 * capture_fraction,
            )
        )
    return spectrum


def save_capture_spectrum(
    spectrum: list[VelocitySpectrumSample],
    output_directory: Path,
    prefix: str = "capture_velocity",
) -> Path:
    """Save capture-cross-section data on the velocity grid."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{prefix}_spectrum.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "velocity_m_per_s",
                "captured_count",
                "launched_count",
                "capture_fraction",
                "capture_cross_section_m2",
            ],
        )
        writer.writeheader()
        for sample in spectrum:
            writer.writerow(
                {
                    "velocity_m_per_s": sample.velocity_m_per_s,
                    "captured_count": sample.captured_count,
                    "launched_count": sample.launched_count,
                    "capture_fraction": sample.capture_fraction,
                    "capture_cross_section_m2": sample.capture_cross_section_m2,
                }
            )
    return csv_path


def plot_capture_cross_section(
    spectrum: list[VelocitySpectrumSample],
    output_directory: Path,
    prefix: str = "capture_velocity",
) -> Path:
    """Plot the capture cross section versus launch speed."""

    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    velocities = np.asarray([sample.velocity_m_per_s for sample in spectrum], dtype=float)
    sigma_mm2 = 1.0e6 * np.asarray([sample.capture_cross_section_m2 for sample in spectrum], dtype=float)
    axis.plot(velocities, sigma_mm2, color="#0f766e", linewidth=2.2)
    axis.set_title(r"Capture Cross Section $\sigma_{\mathrm{captured}}(v)$")
    axis.set_xlabel("Launch speed v [m/s]")
    axis.set_ylabel(r"Capture cross section [mm$^2$]")
    axis.grid(True, alpha=0.25)
    path = output_directory / f"{prefix}_cross_section_vs_velocity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_capture_probability_heatmap(
    samples: list[CaptureVelocitySample],
    disc_radius_m: float,
    velocity_step_m_per_s: float,
    velocity_min_m_per_s: float,
    velocity_max_m_per_s: float,
    s_bin_count: int,
    output_directory: Path,
    prefix: str = "capture_velocity",
) -> Path:
    """Plot capture probability versus impact parameter and launch speed."""

    output_directory.mkdir(parents=True, exist_ok=True)
    if s_bin_count < 2:
        raise ValueError("s_bin_count must be at least 2")
    velocity_grid = velocity_grid_from_samples(samples, velocity_step_m_per_s, velocity_min_m_per_s, velocity_max_m_per_s)
    if len(velocity_grid) == 1:
        velocity_edges = np.asarray([velocity_grid[0] - 0.5 * velocity_step_m_per_s, velocity_grid[0] + 0.5 * velocity_step_m_per_s])
    else:
        velocity_edges = np.empty(len(velocity_grid) + 1, dtype=float)
        velocity_edges[1:-1] = 0.5 * (velocity_grid[:-1] + velocity_grid[1:])
        velocity_edges[0] = max(0.0, velocity_grid[0] - 0.5 * (velocity_grid[1] - velocity_grid[0]))
        velocity_edges[-1] = velocity_grid[-1] + 0.5 * (velocity_grid[-1] - velocity_grid[-2])
    s_edges_m = np.linspace(0.0, disc_radius_m, s_bin_count + 1)
    probability_grid = np.full((len(velocity_grid), s_bin_count), np.nan, dtype=float)
    sample_s = np.asarray([sample.s_m for sample in samples], dtype=float)
    sample_vc = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)

    for s_index in range(s_bin_count):
        left = s_edges_m[s_index]
        right = s_edges_m[s_index + 1]
        if s_index == s_bin_count - 1:
            in_bin = (sample_s >= left) & (sample_s <= right)
        else:
            in_bin = (sample_s >= left) & (sample_s < right)
        bin_vc = sample_vc[in_bin]
        if len(bin_vc) == 0:
            continue
        for velocity_index, velocity_m_per_s in enumerate(velocity_grid):
            probability_grid[velocity_index, s_index] = np.count_nonzero(bin_vc >= velocity_m_per_s - 1.0e-12) / len(bin_vc)

    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    mesh = axis.pcolormesh(
        s_edges_m * 1e3,
        velocity_edges,
        probability_grid,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(mesh, ax=axis, label="Capture probability")
    axis.set_title("Capture Probability vs Impact Parameter and Velocity")
    axis.set_xlabel("Impact parameter s [mm]")
    axis.set_ylabel("Launch speed v [m/s]")
    path = output_directory / f"{prefix}_capture_probability_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_capture_velocity_results(
    samples: list[CaptureVelocitySample],
    search_config: CaptureSearchConfig,
    output_directory: Path,
    prefix: str = "capture_velocity",
) -> tuple[Path, Path]:
    """Save sample-level results and summary metadata to disk."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / f"{prefix}_samples.csv"
    json_path = output_directory / f"{prefix}_summary.json"
    fieldnames = list(_sample_to_row(samples[0]).keys()) if samples else [
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
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(_sample_to_row(sample))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summarize_capture_velocity_samples(samples, search_config), handle, indent=2)
    return csv_path, json_path


def default_analysis_figure_directory(statistics_directory: Path) -> Path:
    """Return the paired figure directory for a statistics output directory."""

    statistics_directory = statistics_directory.resolve()
    parts = statistics_directory.parts
    if "outputs" in parts and "statistics" in parts:
        statistics_index = parts.index("statistics")
        prefix = Path(*parts[:statistics_index])
        suffix = Path(*parts[statistics_index + 1 :])
        return prefix / "figures" / suffix / "sampling_analysis"
    return statistics_directory.parent / f"{statistics_directory.name}_figures"


def load_capture_velocity_samples(csv_path: Path) -> list[CaptureVelocitySample]:
    """Load saved capture-velocity samples from CSV."""

    samples: list[CaptureVelocitySample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                CaptureVelocitySample(
                    disc_index=int(row["disc_index"]),
                    point_index=int(row["point_index"]),
                    theta_rad=float(row["theta_rad"]),
                    phi_rad=float(row["phi_rad"]),
                    theta_prime_rad=float(row["theta_prime_rad"]),
                    s_m=float(row["s_m"]),
                    radial_distance_m=float(row["radial_distance_m"]),
                    initial_position_m=(float(row["x0_m"]), float(row["y0_m"]), float(row["z0_m"])),
                    incident_unit_vector=(float(row["vx_hat"]), float(row["vy_hat"]), float(row["vz_hat"])),
                    capture_velocity_m_per_s=float(row["capture_velocity_m_per_s"]),
                    velocity_resolution_m_per_s=float(row["velocity_resolution_m_per_s"]),
                    trapped_velocity_lower_m_per_s=float(row["trapped_velocity_lower_m_per_s"]),
                    untrapped_velocity_upper_m_per_s=float(row["untrapped_velocity_upper_m_per_s"]),
                    lower_classification=row["lower_classification"],
                    upper_classification=row["upper_classification"],
                    lower_entered_trap_core=row["lower_entered_trap_core"].lower() == "true",
                    upper_entered_trap_core=row["upper_entered_trap_core"].lower() == "true",
                    lower_core_entry_count=int(row.get("lower_core_entry_count", row.get("lower_turning_point_count", 0))),
                    upper_core_entry_count=int(row.get("upper_core_entry_count", row.get("upper_turning_point_count", 0))),
                )
            )
    return samples


def run_capture_velocity_analysis(
    samples: list[CaptureVelocitySample],
    search_config: CaptureSearchConfig,
    output_directory: Path,
    figure_directory: Path | None = None,
) -> dict[str, Path]:
    """Generate analysis products from saved capture-velocity thresholds."""

    figures_directory = figure_directory or default_analysis_figure_directory(output_directory)
    spectrum = capture_spectrum_from_samples(
        samples,
        disc_radius_m=search_config.disc_radius_m,
        velocity_step_m_per_s=search_config.analysis_velocity_step_m_per_s,
        velocity_min_m_per_s=search_config.analysis_velocity_min_m_per_s,
        velocity_max_m_per_s=search_config.analysis_velocity_max_m_per_s,
    )
    spectrum_csv = save_capture_spectrum(spectrum, output_directory)
    cross_section_plot = plot_capture_cross_section(spectrum, figures_directory)
    heatmap_plot = plot_capture_probability_heatmap(
        samples,
        disc_radius_m=search_config.disc_radius_m,
        velocity_step_m_per_s=search_config.analysis_velocity_step_m_per_s,
        velocity_min_m_per_s=search_config.analysis_velocity_min_m_per_s,
        velocity_max_m_per_s=search_config.analysis_velocity_max_m_per_s,
        s_bin_count=search_config.analysis_s_bin_count,
        output_directory=figures_directory,
    )
    return {
        "spectrum_csv": spectrum_csv,
        "cross_section_plot": cross_section_plot,
        "heatmap_plot": heatmap_plot,
    }


def plot_capture_velocity_vs_radius(
    samples: list[CaptureVelocitySample],
    disc: DiscSample,
    output_directory: Path,
) -> Path:
    """Save a per-disc capture-velocity versus impact parameter plot."""

    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    s_mm = 1e3 * np.asarray([sample.s_m for sample in samples], dtype=float)
    capture_velocity = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    theta_prime = np.asarray([sample.theta_prime_rad for sample in samples], dtype=float)
    scatter = axis.scatter(s_mm, capture_velocity, c=theta_prime, cmap="viridis", s=26, alpha=0.9)
    figure.colorbar(scatter, ax=axis, label=r"$\theta'$ [rad]")
    axis.set_title(
        f"Disc {disc.disc_index}: capture velocity vs impact parameter\n"
        f"$\\theta$={disc.theta_rad:.3f} rad, $\\phi$={disc.phi_rad:.3f} rad"
    )
    axis.set_xlabel("Impact parameter s [mm]")
    axis.set_ylabel("Capture velocity [m/s]")
    axis.grid(True, alpha=0.25)
    path = output_directory / f"disc_{disc.disc_index:04d}_capture_velocity_vs_s.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_disc_geometry(
    disc: DiscSample,
    points: list[PointSample],
    beams: list[SimpleMOTBeam],
    output_directory: Path,
) -> Path:
    """Save a 3D geometry plot for one incident disc."""

    output_directory.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#fbfaf6")
    draw_simple_mot_beam_volumes(axis, beams)
    center_mm = 1e3 * np.asarray(disc.center_position_m, dtype=float)
    point_positions_mm = 1e3 * np.asarray([point.initial_position_m for point in points], dtype=float)
    axis.scatter([0.0], [0.0], [0.0], color="#111827", s=36, label="trap center")
    axis.scatter([center_mm[0]], [center_mm[1]], [center_mm[2]], color="#0f766e", s=46, label="disc center")
    axis.scatter(
        point_positions_mm[:, 0],
        point_positions_mm[:, 1],
        point_positions_mm[:, 2],
        color="#059669",
        s=12,
        alpha=0.8,
        label="sample points",
    )
    arrow = 12.0 * np.asarray(disc.incident_unit_vector, dtype=float)
    axis.quiver(
        center_mm[0],
        center_mm[1],
        center_mm[2],
        arrow[0],
        arrow[1],
        arrow[2],
        color="#7c3aed",
        linewidth=2.0,
        arrow_length_ratio=0.15,
    )
    axis.set_title(
        f"Disc {disc.disc_index} geometry\n"
        f"$\\theta$={disc.theta_rad:.3f} rad, $\\phi$={disc.phi_rad:.3f} rad"
    )
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_xlim(-25.0, 25.0)
    axis.set_ylim(-25.0, 25.0)
    axis.set_zlim(-25.0, 25.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend(loc="best")
    path = output_directory / f"disc_{disc.disc_index:04d}_geometry.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_disc_plane_view(
    disc: DiscSample,
    points: list[PointSample],
    beams: list[SimpleMOTBeam],
    output_directory: Path,
    radial_markers_mm: list[float] | None = None,
) -> Path:
    """Save a 2D view looking from the disc center toward the trap center.

    The plotted coordinates are the in-plane coordinates aligned with the
    disc's transverse basis vectors, so radial distance from the origin is the
    impact parameter s.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")

    radial_markers = radial_markers_mm or [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    center_position = np.asarray(disc.center_position_m, dtype=float)
    basis_u = np.asarray(disc.basis_u, dtype=float)
    basis_v = np.asarray(disc.basis_v, dtype=float)
    view_direction = np.asarray(disc.incident_unit_vector, dtype=float)
    point_positions = np.asarray([point.initial_position_m for point in points], dtype=float)
    offsets = point_positions - center_position[None, :]
    u_coordinates_mm = 1e3 * np.einsum("ij,j->i", offsets, basis_u)
    v_coordinates_mm = 1e3 * np.einsum("ij,j->i", offsets, basis_v)

    max_marker_mm = max(radial_markers) if radial_markers else 0.0
    max_sample_mm = float(np.max(np.sqrt(u_coordinates_mm**2 + v_coordinates_mm**2))) if len(points) else 0.0
    extent_mm = max(12.5, max_marker_mm, max_sample_mm) * 1.08

    def plane_coordinates(vector: np.ndarray) -> np.ndarray:
        return np.asarray([float(np.dot(vector, basis_u)), float(np.dot(vector, basis_v))], dtype=float)

    axis_color = {
        "horizontal_x": "#f9a8d4",
        "horizontal_y": "#93c5fd",
        "vertical_z": "#86efac",
    }
    propagation_style = {
        "incident": "-",
        "retro": "--",
    }
    beam_radius_mm = 1e3 * beams[0].intensity_beam.beam_radius_m if beams else 0.0
    for beam in beams:
        beam_direction = np.asarray(beam.direction, dtype=float)
        common_normal = np.cross(view_direction, beam_direction)
        normal_norm = float(np.linalg.norm(common_normal))
        color = axis_color.get(beam.axis_name, "#cbd5e1")
        if normal_norm < 1.0e-10:
            circle = plt.Circle(
                (0.0, 0.0),
                beam_radius_mm,
                edgecolor=color,
                facecolor=color,
                linewidth=1.5,
                linestyle=propagation_style[beam.propagation_sense],
                alpha=0.16,
            )
            axis.add_patch(circle)
            continue

        common_normal = common_normal / normal_norm
        strip_direction = np.cross(view_direction, common_normal)
        strip_direction = strip_direction / np.linalg.norm(strip_direction)
        normal_plane = plane_coordinates(common_normal)
        strip_plane = plane_coordinates(strip_direction)
        p1 = extent_mm * strip_plane + beam_radius_mm * normal_plane
        p2 = extent_mm * strip_plane - beam_radius_mm * normal_plane
        p3 = -extent_mm * strip_plane - beam_radius_mm * normal_plane
        p4 = -extent_mm * strip_plane + beam_radius_mm * normal_plane
        polygon = np.vstack([p1, p2, p3, p4])
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            linestyle=propagation_style[beam.propagation_sense],
            alpha=0.12,
        )

    for marker_mm in radial_markers:
        circle = plt.Circle(
            (0.0, 0.0),
            marker_mm,
            edgecolor="#94a3b8",
            facecolor="none",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
        axis.add_patch(circle)
        axis.text(
            marker_mm,
            0.35,
            f"{marker_mm:.0f} mm",
            color="#64748b",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    axis.scatter([0.0], [0.0], color="#0f766e", s=55, label="disc center")
    axis.scatter(
        u_coordinates_mm,
        v_coordinates_mm,
        color="#059669",
        s=18,
        alpha=0.85,
        label="sample points",
    )
    axis.axhline(0.0, color="#cbd5e1", linewidth=1.0)
    axis.axvline(0.0, color="#cbd5e1", linewidth=1.0)
    axis.set_title(
        f"Disc {disc.disc_index} plane view from disc center\n"
        f"$\\theta$={disc.theta_rad:.3f} rad, $\\phi$={disc.phi_rad:.3f} rad"
    )
    axis.set_xlabel("disc-plane coordinate u [mm]")
    axis.set_ylabel("disc-plane coordinate v [mm]")
    axis.set_xlim(-extent_mm, extent_mm)
    axis.set_ylim(-extent_mm, extent_mm)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.18)
    axis.legend(loc="upper right")

    path = output_directory / f"disc_{disc.disc_index:04d}_plane_view.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_capture_velocity_sampling(
    search_config: CaptureSearchConfig | None = None,
    apparatus_config=None,
    coil_config=None,
    simple_config: SimpleMOTConfig | None = None,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
) -> list[CaptureVelocitySample]:
    """Run the Monte Carlo capture-velocity study."""

    search = search_config or CaptureSearchConfig()
    apparatus = apparatus_config or default_simple_mot_apparatus()
    coils = coil_config or default_anti_helmholtz_config()
    simple = simple_config or default_simple_mot_config()
    paths = simple_mot_paths()
    statistics_directory = output_directory or paths["outputs_statistics_simple_mot"]
    base_figure_directory = figure_directory or default_analysis_figure_directory(statistics_directory).parent
    figures_directory = base_figure_directory / "sampling"
    beams = build_simple_mot_beams(apparatus, simple)
    rng = np.random.default_rng(search.seed)

    all_samples: list[CaptureVelocitySample] = []
    completed_samples = 0
    total_samples = search.disc_count * search.points_per_disc
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc(disc_index, search.radial_distance_m, rng)
        points = sample_disc_points(
            disc,
            search.points_per_disc,
            search.disc_radius_m,
            search.include_center_point,
            rng,
        )
        disc_results: list[CaptureVelocitySample] = []
        for point in points:
            result = find_capture_velocity(beams, point, coils, simple, search)
            disc_results.append(result)
            all_samples.append(result)
            completed_samples += 1
            if completed_samples % max(1, search.save_every) == 0:
                save_capture_velocity_results(all_samples, search, statistics_directory, prefix="capture_velocity_partial")
                print(f"[sampling] saved partial results at {completed_samples}/{total_samples} samples", flush=True)
        plot_capture_velocity_vs_radius(disc_results, disc, figures_directory)
        plot_disc_geometry(disc, points, beams, figures_directory)
        plot_disc_plane_view(disc, points, beams, figures_directory)
        print(f"[sampling] completed disc {disc_index + 1}/{search.disc_count}", flush=True)

    save_capture_velocity_results(all_samples, search, statistics_directory)
    run_capture_velocity_analysis(
        all_samples,
        search,
        statistics_directory,
        figure_directory=base_figure_directory / "sampling_analysis",
    )
    return all_samples


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the capture-velocity sampler."""

    defaults = CaptureSearchConfig()
    parser = argparse.ArgumentParser(description="Monte Carlo capture-velocity sampling for the simplified MOT")
    parser.add_argument("--disc-count", type=int, default=defaults.disc_count)
    parser.add_argument("--points-per-disc", type=int, default=defaults.points_per_disc)
    parser.add_argument("--radial-distance-mm", type=float, default=1e3 * defaults.radial_distance_m)
    parser.add_argument("--disc-radius-mm", type=float, default=1e3 * defaults.disc_radius_m)
    parser.add_argument("--initial-velocity-guess", type=float, default=defaults.initial_velocity_guess_m_per_s)
    parser.add_argument("--velocity-tolerance", type=float, default=defaults.velocity_tolerance_m_per_s)
    parser.add_argument("--max-simulation-time-ms", type=float, default=1e3 * defaults.max_simulation_time_s)
    parser.add_argument("--time-step-us", type=float, default=1e6 * defaults.time_step_s)
    parser.add_argument("--trap-core-radius-mm", type=float, default=1e3 * defaults.trap_core_radius_m)
    parser.add_argument("--escape-radius-mm", type=float, default=1e3 * defaults.escape_radius_m)
    parser.add_argument("--required-core-entries", type=int, default=defaults.required_core_entries)
    parser.add_argument(
        "--bounded-core-residence-ms",
        "--required-core-residence-ms",
        dest="bounded_core_residence_ms",
        type=float,
        default=1e3 * defaults.bounded_core_residence_s,
    )
    parser.add_argument("--include-center-point", action="store_true", default=defaults.include_center_point)
    parser.add_argument("--no-include-center-point", dest="include_center_point", action="store_false")
    parser.add_argument("--analysis-velocity-step", type=float, default=defaults.analysis_velocity_step_m_per_s)
    parser.add_argument("--analysis-s-bin-count", type=int, default=defaults.analysis_s_bin_count)
    parser.add_argument("--analysis-velocity-min", type=float, default=defaults.analysis_velocity_min_m_per_s)
    parser.add_argument("--analysis-velocity-max", type=float, default=defaults.analysis_velocity_max_m_per_s)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--analyze-csv", type=Path, default=None)
    return parser


def search_config_from_args(args: argparse.Namespace) -> CaptureSearchConfig:
    """Build a search config from CLI args."""

    return CaptureSearchConfig(
        radial_distance_m=1e-3 * args.radial_distance_mm,
        disc_radius_m=1e-3 * args.disc_radius_mm,
        initial_velocity_guess_m_per_s=args.initial_velocity_guess,
        velocity_tolerance_m_per_s=args.velocity_tolerance,
        disc_count=args.disc_count,
        points_per_disc=args.points_per_disc,
        max_simulation_time_s=1e-3 * args.max_simulation_time_ms,
        time_step_s=1e-6 * args.time_step_us,
        trap_core_radius_m=1e-3 * args.trap_core_radius_mm,
        escape_radius_m=1e-3 * args.escape_radius_mm,
        required_core_entries=args.required_core_entries,
        bounded_core_residence_s=1e-3 * args.bounded_core_residence_ms,
        include_center_point=args.include_center_point,
        analysis_velocity_step_m_per_s=args.analysis_velocity_step,
        analysis_s_bin_count=args.analysis_s_bin_count,
        analysis_velocity_min_m_per_s=args.analysis_velocity_min,
        analysis_velocity_max_m_per_s=args.analysis_velocity_max,
        seed=args.seed,
        save_every=args.save_every,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    search_config = search_config_from_args(args)
    if args.analyze_csv is not None:
        samples = load_capture_velocity_samples(args.analyze_csv)
        output_directory = args.output_dir or simple_mot_paths()["outputs_statistics_simple_mot"]
        analysis_outputs = run_capture_velocity_analysis(
            samples,
            search_config,
            output_directory,
            figure_directory=args.figures_dir,
        )
        print(json.dumps({key: str(value) for key, value in analysis_outputs.items()}, indent=2))
        return 0
    samples = run_capture_velocity_sampling(
        search_config=search_config,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
    )
    print(json.dumps(summarize_capture_velocity_samples(samples, search_config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
