"""Capture-velocity and loading-rate sampling for the efficient full MOT.

This reproduces the validated two-level disc/impact-parameter workflow using
the multilevel steady-state population-rate force.  Capture searches are
deterministic (mean force, no Langevin diffusion) so that the locally monotonic
binary-search assumption remains meaningful.  Individual diagnostic
trajectories can still include recoil diffusion.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2, RB87_MASS_KG
from ..magnetic_fields import default_anti_helmholtz_config
from ..capture_statistics import (
    CaptureVelocitySample,
    TrajectoryClassification,
    load_capture_velocity_samples,
    plot_capture_velocity_vs_radius,
    run_capture_velocity_analysis,
    save_capture_velocity_results,
)
from ..loading import (
    calculate_loading_rate_from_spectrum,
    load_capture_spectrum,
    save_loading_rate_result,
)
from ..launch_geometry import (
    DiscSample,
    PointSample,
    sample_disc_points,
    sample_incident_disc,
    sample_incident_disc_full_sphere,
    scale,
)
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config, multilevel_mot_paths
from .loading import calculate_sampling_uncertainty
from .rate_equations import build_rate_equation_model, rate_equation_observable
from .simulation import build_multilevel_mot_beams


@dataclass(frozen=True, slots=True)
class RateCaptureSearchConfig:
    """Numerical and sampling controls matching the two-level loading study."""

    radial_distance_m: float = 15.0e-3
    disc_radius_m: float = 12.0e-3
    initial_velocity_guess_m_per_s: float = 20.0
    velocity_tolerance_m_per_s: float = 0.25
    disc_count: int = 50
    points_per_disc: int = 25
    max_simulation_time_s: float = 50.0e-3
    time_step_s: float = 5.0e-6
    trap_core_radius_m: float = 2.0e-3
    escape_radius_m: float = 30.0e-3
    required_core_entries: int = 2
    bounded_core_residence_s: float = 5.0e-3
    max_bracket_iterations: int = 24
    max_search_iterations: int = 24
    include_center_point: bool = True
    analysis_velocity_step_m_per_s: float = 0.25
    analysis_s_bin_count: int = 24
    analysis_velocity_min_m_per_s: float = 1.0
    analysis_velocity_max_m_per_s: float = 30.0
    seed: int = 20260821
    save_every: int = 10
    worker_count: int = 8
    phase_space: str = "full_sphere"


def _validate_search(search: RateCaptureSearchConfig) -> None:
    positive = (
        search.radial_distance_m,
        search.disc_radius_m,
        search.velocity_tolerance_m_per_s,
        search.max_simulation_time_s,
        search.time_step_s,
        search.trap_core_radius_m,
        search.escape_radius_m,
        search.bounded_core_residence_s,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("all physical extents, times, and tolerances must be positive")
    if search.disc_count <= 0 or search.points_per_disc <= 0 or search.worker_count <= 0:
        raise ValueError("disc_count, points_per_disc, and worker_count must be positive")
    if search.required_core_entries <= 0:
        raise ValueError("required_core_entries must be positive")
    if search.phase_space not in {"octant", "full_sphere"}:
        raise ValueError("phase_space must be 'octant' or 'full_sphere'")


def generate_rate_capture_launches(
    search: RateCaptureSearchConfig,
) -> tuple[list[DiscSample], list[PointSample]]:
    """Generate reproducible direction discs and uniform-area impact points."""

    rng = np.random.default_rng(search.seed)
    discs: list[DiscSample] = []
    points: list[PointSample] = []
    for disc_index in range(search.disc_count):
        sampler = (
            sample_incident_disc if search.phase_space == "octant"
            else sample_incident_disc_full_sphere
        )
        disc = sampler(disc_index, search.radial_distance_m, rng)
        discs.append(disc)
        points.extend(sample_disc_points(
            disc,
            search.points_per_disc,
            search.disc_radius_m,
            search.include_center_point,
            rng,
        ))
    return discs, points


def plot_disc_plane_capture_outcomes(
    samples: list[CaptureVelocitySample],
    disc: DiscSample,
    output_directory: Path,
    *,
    reference_velocity_m_per_s: float,
) -> Path:
    """Plot impact points classified at one speed and label capture thresholds.

    A capture search produces a threshold rather than one binary launch.  A
    point is therefore green when an atom launched at the stated reference
    speed is trapped (v <= v_c), and red when it escapes.  The annotation is
    that point's numerically determined capture velocity v_c in m/s.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    center = np.asarray(disc.center_position_m)
    basis_u = np.asarray(disc.basis_u)
    basis_v = np.asarray(disc.basis_v)
    positions = np.asarray([sample.initial_position_m for sample in samples])
    offsets = positions - center
    u_mm = 1e3 * offsets @ basis_u
    v_mm = 1e3 * offsets @ basis_v
    thresholds = np.asarray([sample.capture_velocity_m_per_s for sample in samples])
    trapped = reference_velocity_m_per_s <= thresholds + 1e-12

    figure, axis = plt.subplots(figsize=(8.2, 7.8), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.scatter(u_mm[trapped], v_mm[trapped], color="#15803d", s=34, label="trapped")
    axis.scatter(u_mm[~trapped], v_mm[~trapped], color="#b91c1c", s=34, label="escaped")
    for u_value, v_value, threshold in zip(u_mm, v_mm, thresholds, strict=True):
        axis.annotate(
            f"{threshold:.1f}", (u_value, v_value), xytext=(3, 3),
            textcoords="offset points", fontsize=6, color="#111827",
        )
    radius_mm = 1e3 * max((sample.s_m for sample in samples), default=0.0)
    axis.add_patch(plt.Circle((0.0, 0.0), radius_mm, fill=False, color="#64748b", linestyle="--"))
    axis.axhline(0.0, color="#cbd5e1", linewidth=0.8)
    axis.axvline(0.0, color="#cbd5e1", linewidth=0.8)
    extent = max(1.0, 1.08 * radius_mm)
    axis.set(
        xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
        xlabel="disc-plane coordinate u [mm]",
        ylabel="disc-plane coordinate v [mm]",
        title=(f"Disc {disc.disc_index}: outcome at v={reference_velocity_m_per_s:g} m/s\n"
               "labels give capture velocity $v_c$ [m/s]"),
    )
    axis.grid(alpha=0.16)
    axis.legend(loc="upper right")
    path = output_directory / f"disc_{disc.disc_index:04d}_plane_outcomes.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def classify_rate_trajectory(
    point: PointSample,
    incident_speed_m_per_s: float,
    search: RateCaptureSearchConfig,
    *,
    coil_config=None,
    config: MultilevelMOTConfig | None = None,
) -> TrajectoryClassification:
    """Classify one deterministic mean-force trajectory without storing histories."""

    coil = coil_config or default_anti_helmholtz_config()
    cfg = replace(config or default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(cfg.natural_linewidth_rad_per_s)
    beams = build_multilevel_mot_beams(config=cfg)
    position = np.asarray(point.initial_position_m, dtype=float)
    velocity = np.asarray(scale(incident_speed_m_per_s, point.incident_unit_vector), dtype=float)
    quantization_axis = (0.0, 0.0, 1.0)
    minimum_radius = float(np.linalg.norm(position))
    was_inside = minimum_radius <= search.trap_core_radius_m
    entered_core = was_inside
    core_entries = int(was_inside)
    inside_since_s: float | None = 0.0 if was_inside else None
    elapsed = 0.0
    max_steps = int(np.ceil(search.max_simulation_time_s / search.time_step_s))

    for _ in range(max_steps + 1):
        radius = float(np.linalg.norm(position))
        minimum_radius = min(minimum_radius, radius)
        inside = radius <= search.trap_core_radius_m
        entered_core = entered_core or inside
        if inside and not was_inside:
            core_entries += 1
            inside_since_s = elapsed
        elif not inside:
            inside_since_s = None
        was_inside = inside
        radial_velocity = float(np.dot(position, velocity)) / max(radius, 1.0e-15)
        if core_entries >= search.required_core_entries:
            return TrajectoryClassification(
                True, "two_core_entries", entered_core, core_entries, elapsed,
                minimum_radius, radius, tuple(position), tuple(velocity),
            )
        if inside_since_s is not None and elapsed - inside_since_s >= search.bounded_core_residence_s:
            return TrajectoryClassification(
                True, "bounded_core_residence", entered_core, core_entries, elapsed,
                minimum_radius, radius, tuple(position), tuple(velocity),
            )
        if radius >= search.escape_radius_m and radial_velocity > 0.0:
            return TrajectoryClassification(
                False, "escaped", entered_core, core_entries, elapsed,
                minimum_radius, radius, tuple(position), tuple(velocity),
            )
        if elapsed >= search.max_simulation_time_s - 1.0e-15:
            break

        observable = rate_equation_observable(
            model,
            beams,
            tuple(position),
            tuple(velocity),
            coil,
            cfg,
            previous_axis=quantization_axis,
        )
        quantization_axis = observable.quantization_axis
        dt = min(search.time_step_s, search.max_simulation_time_s - elapsed)
        acceleration = np.asarray(observable.force_n) / RB87_MASS_KG
        if cfg.include_gravity:
            acceleration += np.asarray(GRAVITY_ACCELERATION_M_PER_S2)
        velocity = velocity + acceleration * dt
        position = position + velocity * dt
        elapsed += dt
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            return TrajectoryClassification(
                False, "non_finite", entered_core, core_entries, elapsed,
                minimum_radius, float(np.linalg.norm(position)), tuple(position), tuple(velocity),
            )

    return TrajectoryClassification(
        False, "timeout", entered_core, core_entries, elapsed,
        minimum_radius, float(np.linalg.norm(position)), tuple(position), tuple(velocity),
    )


def find_rate_capture_velocity(
    point: PointSample,
    search: RateCaptureSearchConfig,
    *,
    coil_config=None,
    config: MultilevelMOTConfig | None = None,
) -> CaptureVelocitySample:
    """Find a trapped lower and untrapped upper speed bracket for one point."""

    evaluations: dict[float, TrajectoryClassification] = {}

    def evaluate(speed: float) -> TrajectoryClassification:
        key = round(float(speed), 12)
        if key not in evaluations:
            evaluations[key] = classify_rate_trajectory(
                point, key, search, coil_config=coil_config, config=config,
            )
        return evaluations[key]

    trial = max(0.0, search.initial_velocity_guess_m_per_s)
    if evaluate(trial).trapped:
        lower, upper = trial, max(1.0, trial)
        for _ in range(search.max_bracket_iterations):
            upper *= 2.0
            if not evaluate(upper).trapped:
                break
        else:
            raise RuntimeError("failed to find an untrapped upper capture-speed bracket")
    else:
        upper = trial
        lower: float | None = upper
        for iteration in range(search.max_bracket_iterations):
            lower = 0.0 if iteration == search.max_bracket_iterations - 1 else 0.5 * lower
            if evaluate(lower).trapped:
                break
            if lower <= 1.0e-6:
                lower = 0.0 if evaluate(0.0).trapped else None
                break
        if lower is None:
            zero = evaluations[0.0]
            upper_result = evaluations[round(upper, 12)]
            return _capture_sample(point, 0.0, upper, zero, upper_result)

    assert lower is not None
    for _ in range(search.max_search_iterations):
        if upper - lower <= search.velocity_tolerance_m_per_s:
            break
        midpoint = round(0.5 * (lower + upper), 12)
        if evaluate(midpoint).trapped:
            lower = midpoint
        else:
            upper = midpoint
    return _capture_sample(
        point,
        lower,
        upper,
        evaluations[round(lower, 12)],
        evaluations[round(upper, 12)],
    )


def _capture_sample(
    point: PointSample,
    lower: float,
    upper: float,
    lower_result: TrajectoryClassification,
    upper_result: TrajectoryClassification,
) -> CaptureVelocitySample:
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
        capture_velocity_m_per_s=lower,
        velocity_resolution_m_per_s=upper - lower,
        trapped_velocity_lower_m_per_s=lower,
        untrapped_velocity_upper_m_per_s=upper,
        lower_classification=lower_result.termination_reason,
        upper_classification=upper_result.termination_reason,
        lower_entered_trap_core=lower_result.entered_trap_core,
        upper_entered_trap_core=upper_result.entered_trap_core,
        lower_core_entry_count=lower_result.core_entry_count,
        upper_core_entry_count=upper_result.core_entry_count,
    )


def _worker_find_capture_velocity(payload) -> CaptureVelocitySample:
    point, search, coil_config, config = payload
    return find_rate_capture_velocity(
        point, search, coil_config=coil_config, config=config,
    )


def _sort_samples(samples: list[CaptureVelocitySample]) -> list[CaptureVelocitySample]:
    return sorted(samples, key=lambda sample: (sample.disc_index, sample.point_index))


def summarize_rate_capture_run(
    samples: list[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    config: MultilevelMOTConfig,
    loading_result,
    loading_uncertainty,
    wall_time_s: float,
) -> dict[str, object]:
    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples])
    resolution = np.asarray([sample.velocity_resolution_m_per_s for sample in samples])
    return {
        "model": "full multilevel Rb-87 MOT; adiabatically eliminated population rate equations",
        "capture_dynamics": "deterministic mean force; recoil diffusion disabled for monotonic binary search",
        "classification": f"captured after {search.required_core_entries} entries into the {1e3*search.trap_core_radius_m:g} mm core or {1e3*search.bounded_core_residence_s:g} ms continuous residence there; escaped at {1e3*search.escape_radius_m:g} mm with outward radial velocity; otherwise timeout",
        "sample_count": len(samples),
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "phase_space": search.phase_space,
        "solid_angle_sr": float(0.5 * np.pi if search.phase_space == "octant" else 4.0 * np.pi),
        "cross_section_normalization": "direction-averaged projected area; no octant multiplicity factor",
        "capture_velocity_mean_m_per_s": float(np.mean(capture)),
        "capture_velocity_std_m_per_s": float(np.std(capture)),
        "capture_velocity_quantiles_m_per_s": {
            "q05": float(np.quantile(capture, .05)),
            "q25": float(np.quantile(capture, .25)),
            "median": float(np.quantile(capture, .5)),
            "q75": float(np.quantile(capture, .75)),
            "q95": float(np.quantile(capture, .95)),
        },
        "capture_velocity_min_m_per_s": float(np.min(capture)),
        "capture_velocity_max_m_per_s": float(np.max(capture)),
        "zero_capture_velocity_count": int(np.count_nonzero(capture == 0.0)),
        "velocity_resolution_mean_m_per_s": float(np.mean(resolution)),
        "velocity_resolution_max_m_per_s": float(np.max(resolution)),
        "valid_bracket_count": int(sum(
            sample.lower_classification in {"two_core_entries", "bounded_core_residence"}
            and sample.upper_classification not in {"two_core_entries", "bounded_core_residence"}
            for sample in samples
        )),
        "loading_rate_atoms_per_s": loading_result.loading_rate_atoms_per_s,
        "loading_rate_monte_carlo_point_standard_error_atoms_per_s": loading_uncertainty.point_standard_error_atoms_per_s,
        "loading_rate_monte_carlo_disc_cluster_standard_error_atoms_per_s": loading_uncertainty.disc_cluster_standard_error_atoms_per_s,
        "loading_rate_approximate_disc_cluster_95_percent_half_width_atoms_per_s": loading_uncertainty.disc_cluster_95_percent_half_width_atoms_per_s,
        "loading_integral_m5_per_s4": loading_result.integral_value_m5_per_s4,
        "speed_distribution": "f(v) = 2.80e-7 v^2 exp(-v^2/5.667e4)",
        "loading_formula": "R = 9.1196e5 integral sigma_cap(v) v^3 exp(-v^2/5.667e4) dv",
        "loading_quadrature": loading_result.quadrature_method,
        "wall_time_s": wall_time_s,
        "search_config": asdict(search),
        "multilevel_config": asdict(config),
    }


def run_rate_capture_sampling(
    search: RateCaptureSearchConfig | None = None,
    *,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Run, checkpoint, analyze, and integrate the complete loading study."""

    search = search or RateCaptureSearchConfig()
    _validate_search(search)
    paths = multilevel_mot_paths()
    output = output_directory or paths["statistics"] / "loading_rate_50_discs_25_points"
    figures = figure_directory or paths["figures"] / "loading_rate_50_discs_25_points"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    partial_csv = output / "capture_velocity_partial_samples.csv"
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    coil = default_anti_helmholtz_config()
    discs, points = generate_rate_capture_launches(search)
    existing = load_capture_velocity_samples(partial_csv) if resume and partial_csv.exists() else []
    results = {(sample.disc_index, sample.point_index): sample for sample in existing}
    missing = [point for point in points if (point.disc_index, point.point_index) not in results]
    total = len(points)
    start = perf_counter()
    print(
        f"[full MOT loading] {len(results)}/{total} samples already complete; "
        f"running {len(missing)} with {search.worker_count} workers",
        flush=True,
    )
    payloads = [(point, search, coil, config) for point in missing]
    if search.worker_count == 1:
        iterator = map(_worker_find_capture_velocity, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=search.worker_count)
        iterator = executor.map(_worker_find_capture_velocity, payloads, chunksize=1)
    try:
        for result in iterator:
            results[(result.disc_index, result.point_index)] = result
            completed = len(results)
            elapsed = perf_counter() - start
            newly_completed = completed - len(existing)
            rate = newly_completed / elapsed if elapsed > 0.0 else 0.0
            eta = (len(missing) - newly_completed) / rate if rate > 0.0 else float("inf")
            print(
                f"[full MOT loading] simulation {completed}/{total}; "
                f"disc {result.disc_index + 1}/{search.disc_count}, "
                f"point {result.point_index + 1}/{search.points_per_disc}, "
                f"vc={result.capture_velocity_m_per_s:.3f} m/s; ETA={eta/60:.1f} min",
                flush=True,
            )
            if completed % max(1, search.save_every) == 0:
                save_capture_velocity_results(
                    _sort_samples(list(results.values())), search, output,
                    prefix="capture_velocity_partial",
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    samples = _sort_samples(list(results.values()))
    if len(samples) != total:
        raise RuntimeError(f"sampling incomplete: obtained {len(samples)} of {total} points")
    save_capture_velocity_results(samples, search, output)
    analysis = run_capture_velocity_analysis(samples, search, output, figure_directory=figures)
    for disc in discs:
        disc_samples = [sample for sample in samples if sample.disc_index == disc.disc_index]
        plot_capture_velocity_vs_radius(disc_samples, disc, figures / "per_disc")
        plot_disc_plane_capture_outcomes(
            disc_samples,
            disc,
            figures / "per_disc_plane",
            reference_velocity_m_per_s=search.initial_velocity_guess_m_per_s,
        )
    velocity, cross_section = load_capture_spectrum(analysis["spectrum_csv"])
    loading_result = calculate_loading_rate_from_spectrum(velocity, cross_section)
    loading_json = save_loading_rate_result(loading_result, output / "loading_rate_result.json")
    uncertainty = calculate_sampling_uncertainty(
        np.asarray([sample.capture_velocity_m_per_s for sample in samples]),
        np.asarray([sample.disc_index for sample in samples]),
        velocity,
        search.disc_radius_m,
    )
    summary = summarize_rate_capture_run(
        samples,
        search,
        config,
        loading_result,
        uncertainty,
        perf_counter() - start,
    )
    summary["outputs"] = {
        "samples_csv": str(output / "capture_velocity_samples.csv"),
        "capture_spectrum_csv": str(analysis["spectrum_csv"]),
        "loading_rate_json": str(loading_json),
        "cross_section_plot": str(analysis["cross_section_plot"]),
        "capture_probability_heatmap": str(analysis["heatmap_plot"]),
        "per_disc_figure_directory": str(figures / "per_disc"),
        "per_disc_plane_figure_directory": str(figures / "per_disc_plane"),
    }
    summary_path = output / "full_mot_loading_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"]["summary_json"] = str(summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    defaults = RateCaptureSearchConfig()
    parser = argparse.ArgumentParser(description="Efficient full-MOT capture and loading-rate study")
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
    parser.add_argument("--bounded-core-residence-ms", type=float, default=1e3 * defaults.bounded_core_residence_s)
    parser.add_argument("--analysis-velocity-step", type=float, default=defaults.analysis_velocity_step_m_per_s)
    parser.add_argument("--analysis-velocity-min", type=float, default=defaults.analysis_velocity_min_m_per_s)
    parser.add_argument("--analysis-velocity-max", type=float, default=defaults.analysis_velocity_max_m_per_s)
    parser.add_argument("--analysis-s-bin-count", type=int, default=defaults.analysis_s_bin_count)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--workers", type=int, default=defaults.worker_count)
    parser.add_argument(
        "--phase-space", choices=("octant", "full-sphere"), default="full-sphere",
        help="sample incident directions over all 4 pi (default) or in one symmetry octant",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    search = RateCaptureSearchConfig(
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
        analysis_velocity_step_m_per_s=args.analysis_velocity_step,
        analysis_s_bin_count=args.analysis_s_bin_count,
        analysis_velocity_min_m_per_s=args.analysis_velocity_min,
        analysis_velocity_max_m_per_s=args.analysis_velocity_max,
        seed=args.seed,
        save_every=args.save_every,
        worker_count=args.workers,
        phase_space=args.phase_space.replace("-", "_"),
    )
    run_rate_capture_sampling(
        search,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
