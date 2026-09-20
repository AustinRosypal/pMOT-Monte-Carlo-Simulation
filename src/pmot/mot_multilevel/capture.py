"""Capture velocity, cross section, and loading-rate scaffolding.

All dynamics use the physical Section-12 population-rate force. Capture
trajectories are deterministic and recompute that force at every RK4 stage.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from ..capture_statistics import (
    CaptureVelocitySample,
    TrajectoryClassification,
    VelocitySpectrumSample,
    load_capture_velocity_samples,
    plot_capture_cross_section,
    save_capture_spectrum,
    save_capture_velocity_results,
    velocity_grid_from_samples,
)
from ..configuration import AntiHelmholtzCoilConfig, GRAVITY_ACCELERATION_M_PER_S2, RB87_MASS_KG
from ..launch_geometry import (
    DiscSample,
    PointSample,
    sample_disc_points,
    sample_incident_disc_full_sphere,
    scale,
)
from ..loading import LoadingRateResult, calculate_loading_rate_from_spectrum, save_loading_rate_result
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config, multilevel_mot_paths
from .polarization import quantization_axis
from .rate_equations import (
    RateEquationModel,
    build_rate_equation_model,
    local_magnetic_field_t,
    rate_equation_observable,
)
from .simulation import build_multilevel_mot_beams


@dataclass(frozen=True, slots=True)
class CaptureSearchConfig:
    """Sampling, integration, and capture-classification controls."""

    radial_distance_m: float = 15.0e-3
    disc_radius_m: float = 12.0e-3
    disc_count: int = 25
    points_per_disc: int = 25
    include_center_point: bool = False
    initial_velocity_guess_m_per_s: float = 10.0
    velocity_tolerance_m_per_s: float = 0.25
    maximum_bracket_speed_m_per_s: float = 80.0
    max_bracket_iterations: int = 16
    max_search_iterations: int = 16
    maximum_simulation_time_s: float = 50.0e-3
    time_step_s: float = 5.0e-6
    trap_core_radius_m: float = 2.0e-3
    bounded_core_residence_s: float = 5.0e-3
    required_core_entries: int = 2
    escape_radius_m: float = 30.0e-3
    analysis_velocity_step_m_per_s: float = 0.25
    analysis_velocity_min_m_per_s: float = 0.0
    analysis_velocity_max_m_per_s: float = 30.0
    analysis_s_bin_count: int = 24
    seed: int = 20260918
    worker_count: int = 1
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        positive = (
            self.radial_distance_m,
            self.disc_radius_m,
            self.velocity_tolerance_m_per_s,
            self.maximum_bracket_speed_m_per_s,
            self.maximum_simulation_time_s,
            self.time_step_s,
            self.trap_core_radius_m,
            self.bounded_core_residence_s,
            self.escape_radius_m,
            self.analysis_velocity_step_m_per_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("physical extents, times, and tolerances must be positive")
        if self.disc_count <= 0 or self.points_per_disc <= 0:
            raise ValueError("disc_count and points_per_disc must be positive")
        if self.required_core_entries <= 0 or self.worker_count <= 0:
            raise ValueError("core-entry and worker counts must be positive")
        if self.max_bracket_iterations <= 0 or self.max_search_iterations <= 0:
            raise ValueError("capture-search iteration counts must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.analysis_velocity_max_m_per_s <= self.analysis_velocity_min_m_per_s:
            raise ValueError("analysis velocity maximum must exceed its minimum")


@dataclass(frozen=True, slots=True)
class CaptureLoadingResult:
    samples: tuple[CaptureVelocitySample, ...]
    spectrum: tuple[VelocitySpectrumSample, ...]
    loading: LoadingRateResult
    output_paths: dict[str, Path]
    wall_time_s: float


class IndeterminateCaptureError(RuntimeError):
    """Raised when timeout/non-finite dynamics cannot establish a bracket."""


def generate_capture_launches(
    search: CaptureSearchConfig,
) -> tuple[list[DiscSample], list[PointSample]]:
    """Sample full-sphere direction discs and uniform-area impact points."""

    rng = np.random.default_rng(search.seed)
    discs: list[DiscSample] = []
    points: list[PointSample] = []
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc_full_sphere(
            disc_index,
            search.radial_distance_m,
            rng,
        )
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


def _rk4_step(
    position: np.ndarray,
    velocity: np.ndarray,
    time_step_s: float,
    previous_axis,
    model: RateEquationModel,
    beams,
    coil_config,
    config: MultilevelMOTConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    gravity = np.asarray(
        GRAVITY_ACCELERATION_M_PER_S2 if config.include_gravity else (0.0, 0.0, 0.0),
        dtype=float,
    )

    def derivative(local_position, local_velocity):
        observable = rate_equation_observable(
            model,
            beams,
            tuple(local_position),
            tuple(local_velocity),
            coil_config,
            config,
            previous_axis,
        )
        acceleration = np.asarray(observable.force_n) / RB87_MASS_KG + gravity
        return local_velocity, acceleration

    k1_r, k1_v = derivative(position, velocity)
    k2_r, k2_v = derivative(
        position + 0.5 * time_step_s * k1_r,
        velocity + 0.5 * time_step_s * k1_v,
    )
    k3_r, k3_v = derivative(
        position + 0.5 * time_step_s * k2_r,
        velocity + 0.5 * time_step_s * k2_v,
    )
    k4_r, k4_v = derivative(
        position + time_step_s * k3_r,
        velocity + time_step_s * k3_v,
    )
    next_position = position + time_step_s * (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6.0
    next_velocity = velocity + time_step_s * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
    field = local_magnetic_field_t(tuple(next_position), coil_config)
    next_axis = quantization_axis(
        field,
        previous_axis,
        config.magnetic_field_epsilon_t,
    )
    return next_position, next_velocity, next_axis


def classify_capture_trajectory(
    point: PointSample,
    incident_speed_m_per_s: float,
    search: CaptureSearchConfig,
    *,
    coil_config: AntiHelmholtzCoilConfig | None = None,
    config: MultilevelMOTConfig | None = None,
    model: RateEquationModel | None = None,
    beams=None,
) -> TrajectoryClassification:
    """Integrate one deterministic launch and apply the shared capture rule."""

    if incident_speed_m_per_s < 0.0:
        raise ValueError("incident speed must be non-negative")
    cfg = replace(config or default_multilevel_mot_config(), repumper_enabled=True)
    coil = coil_config or default_anti_helmholtz_config()
    rate_model = model or build_rate_equation_model()
    optical_beams = build_multilevel_mot_beams(config=cfg) if beams is None else beams
    position = np.asarray(point.initial_position_m, dtype=float)
    velocity = np.asarray(scale(incident_speed_m_per_s, point.incident_unit_vector))
    previous_axis = (0.0, 0.0, 1.0)
    elapsed = 0.0
    radius = float(np.linalg.norm(position))
    minimum_radius = radius
    was_inside = radius <= search.trap_core_radius_m
    entered_core = was_inside
    core_entries = int(was_inside)
    inside_since = 0.0 if was_inside else None
    maximum_steps = int(np.ceil(search.maximum_simulation_time_s / search.time_step_s))

    for _ in range(maximum_steps + 1):
        radius = float(np.linalg.norm(position))
        minimum_radius = min(minimum_radius, radius)
        inside = radius <= search.trap_core_radius_m
        entered_core = entered_core or inside
        if inside and not was_inside:
            core_entries += 1
            inside_since = elapsed
        elif not inside:
            inside_since = None
        was_inside = inside
        radial_velocity = float(np.dot(position, velocity)) / max(radius, 1.0e-15)
        if core_entries >= search.required_core_entries:
            return TrajectoryClassification(
                True,
                "two_core_entries",
                entered_core,
                core_entries,
                elapsed,
                minimum_radius,
                radius,
                tuple(position),
                tuple(velocity),
            )
        if inside_since is not None and elapsed - inside_since >= search.bounded_core_residence_s:
            return TrajectoryClassification(
                True,
                "bounded_core_residence",
                entered_core,
                core_entries,
                elapsed,
                minimum_radius,
                radius,
                tuple(position),
                tuple(velocity),
            )
        if radius >= search.escape_radius_m and radial_velocity > 0.0:
            return TrajectoryClassification(
                False,
                "escaped",
                entered_core,
                core_entries,
                elapsed,
                minimum_radius,
                radius,
                tuple(position),
                tuple(velocity),
            )
        if elapsed >= search.maximum_simulation_time_s - 1.0e-15:
            break
        time_step = min(search.time_step_s, search.maximum_simulation_time_s - elapsed)
        position, velocity, previous_axis = _rk4_step(
            position,
            velocity,
            time_step,
            previous_axis,
            rate_model,
            optical_beams,
            coil,
            cfg,
        )
        elapsed += time_step
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            return TrajectoryClassification(
                False,
                "non_finite",
                entered_core,
                core_entries,
                elapsed,
                minimum_radius,
                float(np.linalg.norm(position)),
                tuple(position),
                tuple(velocity),
            )

    return TrajectoryClassification(
        False,
        "timeout",
        entered_core,
        core_entries,
        elapsed,
        minimum_radius,
        float(np.linalg.norm(position)),
        tuple(position),
        tuple(velocity),
    )


def _is_definitive_untrapped(result: TrajectoryClassification) -> bool:
    return not result.trapped and result.termination_reason == "escaped"


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


def find_capture_velocity(
    point: PointSample,
    search: CaptureSearchConfig,
    *,
    coil_config: AntiHelmholtzCoilConfig | None = None,
    config: MultilevelMOTConfig | None = None,
    classifier: Callable[..., TrajectoryClassification] = classify_capture_trajectory,
) -> CaptureVelocitySample:
    """Bracket and bisect a locally monotone capture boundary, failing closed."""

    evaluations: dict[float, TrajectoryClassification] = {}

    def evaluate(speed: float) -> TrajectoryClassification:
        key = round(float(speed), 12)
        if key not in evaluations:
            result = classifier(
                point,
                key,
                search,
                coil_config=coil_config,
                config=config,
            )
            if not result.trapped and result.termination_reason != "escaped":
                raise IndeterminateCaptureError(
                    f"capture classification at {key:g} m/s is "
                    f"{result.termination_reason}; increase the trajectory duration"
                )
            evaluations[key] = result
        return evaluations[key]

    trial = max(0.0, search.initial_velocity_guess_m_per_s)
    trial_result = evaluate(trial)
    if trial_result.trapped:
        lower = trial
        upper = max(1.0, 2.0 * trial)
        for _ in range(search.max_bracket_iterations):
            if upper > search.maximum_bracket_speed_m_per_s:
                break
            if _is_definitive_untrapped(evaluate(upper)):
                break
            lower = upper
            upper *= 2.0
        else:
            raise RuntimeError("failed to find an escaped upper capture-speed bracket")
        if upper > search.maximum_bracket_speed_m_per_s:
            raise RuntimeError("capture boundary exceeds maximum_bracket_speed_m_per_s")
    else:
        upper = trial
        candidate = 0.5 * upper
        for _ in range(search.max_bracket_iterations):
            result = evaluate(candidate)
            if result.trapped:
                lower = candidate
                break
            if candidate == 0.0:
                return _capture_sample(
                    point,
                    0.0,
                    upper,
                    result,
                    evaluations[round(upper, 12)],
                )
            upper = candidate
            candidate = (
                0.0
                if candidate <= search.velocity_tolerance_m_per_s
                else 0.5 * candidate
            )
        else:
            raise RuntimeError("failed to find a trapped lower capture-speed bracket")

    lower_result = evaluate(lower)
    upper_result = evaluate(upper)
    if not lower_result.trapped or not _is_definitive_untrapped(upper_result):
        raise RuntimeError("capture search did not establish a valid trapped/escaped bracket")
    for _ in range(search.max_search_iterations):
        if upper - lower <= search.velocity_tolerance_m_per_s:
            break
        midpoint = round(0.5 * (lower + upper), 12)
        result = evaluate(midpoint)
        if result.trapped:
            lower, lower_result = midpoint, result
        else:
            upper, upper_result = midpoint, result
    return _capture_sample(point, lower, upper, lower_result, upper_result)


def _worker(payload):
    point, search, config, coil_config = payload
    return find_capture_velocity(point, search, config=config, coil_config=coil_config)


def capture_cross_section_spectrum(
    samples: list[CaptureVelocitySample],
    search: CaptureSearchConfig,
) -> list[VelocitySpectrumSample]:
    """Aggregate the direction-averaged cross section without a zero-speed bias.

    A compatibility threshold of zero can mean either that zero speed was
    trapped or that no trapped lower bracket exists. Only samples whose stored
    lower endpoint is actually trapped contribute at zero velocity.
    """

    velocity_grid = velocity_grid_from_samples(
        samples,
        search.analysis_velocity_step_m_per_s,
        search.analysis_velocity_min_m_per_s,
        search.analysis_velocity_max_m_per_s,
    )
    trapped_reasons = {"two_core_entries", "bounded_core_residence"}
    disc_area = pi * search.disc_radius_m**2
    launched_count = len(samples)
    output: list[VelocitySpectrumSample] = []
    for velocity in velocity_grid:
        captured_count = sum(
            sample.lower_classification in trapped_reasons
            and sample.capture_velocity_m_per_s >= velocity - 1.0e-12
            for sample in samples
        )
        fraction = captured_count / launched_count if launched_count else 0.0
        output.append(
            VelocitySpectrumSample(
                velocity_m_per_s=float(velocity),
                captured_count=int(captured_count),
                launched_count=launched_count,
                capture_fraction=float(fraction),
                capture_cross_section_m2=float(disc_area * fraction),
            )
        )
    return output


def plot_capture_probability_heatmap(
    samples: list[CaptureVelocitySample],
    search: CaptureSearchConfig,
    output_directory: Path,
) -> Path:
    """Plot the conservative threshold mask, excluding uncaptured zero nodes."""

    velocity = velocity_grid_from_samples(
        samples,
        search.analysis_velocity_step_m_per_s,
        search.analysis_velocity_min_m_per_s,
        search.analysis_velocity_max_m_per_s,
    )
    velocity_edges = np.empty(len(velocity) + 1)
    velocity_edges[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
    velocity_edges[0] = max(0.0, velocity[0] - 0.5 * (velocity[1] - velocity[0]))
    velocity_edges[-1] = velocity[-1] + 0.5 * (velocity[-1] - velocity[-2])
    radius_edges = np.linspace(0.0, search.disc_radius_m, search.analysis_s_bin_count + 1)
    impact = np.asarray([sample.s_m for sample in samples])
    threshold = np.asarray([sample.capture_velocity_m_per_s for sample in samples])
    valid_lower = np.asarray([
        sample.lower_classification in {"two_core_entries", "bounded_core_residence"}
        for sample in samples
    ])
    grid = np.full((len(velocity), search.analysis_s_bin_count), np.nan)
    for radius_index in range(search.analysis_s_bin_count):
        left = radius_edges[radius_index]
        right = radius_edges[radius_index + 1]
        in_bin = (
            (impact >= left)
            & ((impact <= right) if radius_index == search.analysis_s_bin_count - 1 else (impact < right))
        )
        if not np.any(in_bin):
            continue
        grid[:, radius_index] = np.mean(
            valid_lower[in_bin][None, :]
            & (threshold[in_bin][None, :] >= velocity[:, None] - 1.0e-12),
            axis=1,
        )
    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    mesh = axis.pcolormesh(
        1.0e3 * radius_edges,
        velocity_edges,
        grid,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(mesh, ax=axis, label="Captured fraction")
    axis.set(
        title="Capture fraction by impact parameter and speed",
        xlabel="Impact parameter [mm]",
        ylabel="Incident speed [m/s]",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "capture_velocity_capture_probability_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_capture_loading_study(
    search: CaptureSearchConfig | None = None,
    *,
    config: MultilevelMOTConfig | None = None,
    coil_config: AntiHelmholtzCoilConfig | None = None,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = False,
) -> CaptureLoadingResult:
    """Run a checkpointed capture study, cross section, and loading quadrature."""

    search = search or CaptureSearchConfig()
    cfg = replace(config or default_multilevel_mot_config(), repumper_enabled=True)
    coil = coil_config or default_anti_helmholtz_config()
    paths = multilevel_mot_paths()
    output = output_directory or paths["statistics"] / "capture_loading"
    figures = figure_directory or paths["figures"] / "capture_loading"
    manifest_path = output / "capture_run_config.json"
    partial_csv = output / "capture_velocity_partial_samples.csv"
    if output.exists() and not manifest_path.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"capture output directory contains data without a run manifest: {output}; "
            "choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "pmot.mot_multilevel.capture-run-config.v1",
        "search_config": asdict(search),
        "multilevel_config": asdict(cfg),
        "coil_config": asdict(coil),
    }
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"capture output directory already has a run: {output}; "
                "pass resume=True to continue matching inputs"
            )
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise ValueError("saved capture run configuration does not match")
    elif resume and partial_csv.exists():
        raise ValueError("cannot resume partial capture data without a config manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _, points = generate_capture_launches(search)
    start = perf_counter()
    existing = load_capture_velocity_samples(partial_csv) if resume and partial_csv.exists() else []
    results = {(sample.disc_index, sample.point_index): sample for sample in existing}
    expected_keys = {(point.disc_index, point.point_index) for point in points}
    if not set(results).issubset(expected_keys):
        raise ValueError("partial capture data contain launch points outside this run")
    missing = [
        point for point in points
        if (point.disc_index, point.point_index) not in results
    ]
    payloads = [(point, search, cfg, coil) for point in missing]
    if search.worker_count == 1:
        iterator = map(_worker, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=search.worker_count)
        iterator = executor.map(_worker, payloads, chunksize=1)
    try:
        for completed, sample in enumerate(iterator, start=1):
            results[(sample.disc_index, sample.point_index)] = sample
            if completed % search.checkpoint_every == 0:
                save_capture_velocity_results(
                    sorted(results.values(), key=lambda item: (item.disc_index, item.point_index)),
                    search,
                    output,
                    prefix="capture_velocity_partial",
                )
            print(
                f"[physical MOT capture] {len(results)}/{len(points)} "
                f"disc={sample.disc_index} point={sample.point_index} "
                f"vc={sample.capture_velocity_m_per_s:.3f} m/s",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    samples = sorted(results.values(), key=lambda item: (item.disc_index, item.point_index))
    if len(samples) != len(points):
        raise RuntimeError(f"capture sampling incomplete: {len(samples)}/{len(points)}")
    save_capture_velocity_results(
        samples,
        search,
        output,
        prefix="capture_velocity_partial",
    )
    samples_csv, samples_json = save_capture_velocity_results(samples, search, output)
    spectrum = capture_cross_section_spectrum(samples, search)
    spectrum_csv = save_capture_spectrum(spectrum, output)
    cross_section_plot = plot_capture_cross_section(spectrum, figures)
    heatmap_plot = plot_capture_probability_heatmap(samples, search, figures)
    velocity = np.asarray([item.velocity_m_per_s for item in spectrum])
    cross_section = np.asarray([item.capture_cross_section_m2 for item in spectrum])
    loading = calculate_loading_rate_from_spectrum(velocity, cross_section)
    loading_json = save_loading_rate_result(loading, output / "loading_rate_result.json")
    summary = {
        "schema": "pmot.mot_multilevel.capture-loading.v1",
        "model": "Section-12 24-state deterministic mean-force MOT",
        "capture_velocity_assumption": "local monotonicity; bracket endpoints are trapped and escaped",
        "timeout_policy": "fail closed; timeout is never relabeled escaped",
        "search_config": asdict(search),
        "multilevel_config": asdict(cfg),
        "coil_config": asdict(coil),
        "sample_count": len(samples),
        "mean_capture_velocity_m_per_s": float(
            np.mean([sample.capture_velocity_m_per_s for sample in samples])
        ),
        "loading_rate_atoms_per_s": loading.loading_rate_atoms_per_s,
        "wall_time_s": perf_counter() - start,
    }
    summary_path = output / "capture_loading_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return CaptureLoadingResult(
        samples=tuple(samples),
        spectrum=tuple(spectrum),
        loading=loading,
        output_paths={
            "config_manifest_json": manifest_path,
            "partial_samples_csv": partial_csv,
            "samples_csv": samples_csv,
            "samples_summary_json": samples_json,
            "spectrum_csv": spectrum_csv,
            "cross_section_plot": cross_section_plot,
            "heatmap_plot": heatmap_plot,
            "loading_json": loading_json,
            "summary_json": summary_path,
        },
        wall_time_s=perf_counter() - start,
    )


__all__ = [
    "CaptureLoadingResult",
    "CaptureSearchConfig",
    "IndeterminateCaptureError",
    "classify_capture_trajectory",
    "capture_cross_section_spectrum",
    "find_capture_velocity",
    "generate_capture_launches",
    "plot_capture_probability_heatmap",
    "run_capture_loading_study",
]
