"""Monte Carlo capture-velocity sampling for the simplified two-level MOT."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..capture_statistics import (
    CaptureVelocitySample,
    TrajectoryClassification,
    VelocitySpectrumSample,
    capture_spectrum_from_samples,
    default_analysis_figure_directory,
    load_capture_velocity_samples,
    plot_capture_cross_section,
    plot_capture_probability_heatmap,
    plot_capture_velocity_vs_radius,
    run_capture_velocity_analysis,
    save_capture_spectrum,
    save_capture_velocity_results,
    summarize_capture_velocity_samples,
    velocity_grid_from_samples,
)
from ..launch_geometry import (
    DiscSample,
    PointSample,
    Vec3,
    add,
    build_incident_disc_from_angles,
    choose_transverse_basis,
    cross,
    dot,
    norm,
    normalize,
    radial_distance_magnitude,
    sample_disc_points,
    sample_full_sphere_direction,
    sample_incident_disc,
    sample_incident_disc_full_sphere,
    sample_octant_direction,
    scale,
    spherical_direction,
    spherical_octant_direction,
)
from ..state import AtomState
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import default_simple_mot_apparatus
from .configuration import default_simple_mot_config
from .configuration import simple_mot_paths
from .plotting import draw_simple_mot_beam_volumes
from .simulation import build_simple_mot_beams
from .simulation import rk4_step
from .simulation import SimpleMOTBeam
from .simulation import SimpleMOTConfig


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
