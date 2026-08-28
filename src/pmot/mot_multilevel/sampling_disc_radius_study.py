"""Full-sphere sampling-disc-radius study for the production 24-state MOT.

Each radius uses 100 independently sampled incident directions and 100 random
uniform-area impact points per perpendicular disc.  Capture trajectories use
the deterministic repumper-enabled population-rate force.  Direction discs,
not individual impact points, are the independent clusters used for loading-
rate and capture-cross-section uncertainty estimates.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import ceil, log, pi
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi_squared_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .configuration import multilevel_mot_paths
from .power_loading_study import (
    REPUMP_POWER_W_PER_BEAM,
    StudyPaths,
    build_27mw_multilevel_configuration,
    default_study_search_config,
    generate_study_geometry,
    run_power_loading_study,
)
from .rate_capture import RateCaptureSearchConfig
from .screening import draw_multilevel_mot_beam_volumes


STUDY_NAME = "loading_vs_sampling_disc_radius_full_sphere_100_discs_100_points_cooling27mW"
SAMPLING_DISC_RADII_MM = (5.0, 12.0, 15.0, 20.0, 25.0, 30.0)
COOLING_POWER_W_PER_BEAM = 27.0e-3
DISC_COUNT = 100
POINTS_PER_DISC = 100
BASE_RANDOM_SEED = 20260827
CONFIRMATION_RANDOM_SEED = 20270827
DEFAULT_WORKER_COUNT = 24
PROGRESS_EVERY = 25
CHECKPOINT_EVERY = 100
CONVERGENCE_FRACTION = 0.95
FIT_GOODNESS_CONFIDENCE = 0.95
MAX_RELATIVE_ASYMPTOTE_SEM = 0.5
# Five adjacent-radius comparisons share a familywise one-sided alpha of 0.05.
MAX_ADJACENT_DOWNWARD_Z = 2.3263478740408408
CONVERGENCE_RADIUS_UPPER_Z = 1.96
MIN_LARGEST_RADIUS_ASYMPTOTE_FRACTION = 0.90
AGGREGATE_FIELDNAMES = (
    "sampling_disc_radius_mm",
    "random_seed",
    "disc_count",
    "points_per_disc",
    "sample_count",
    "phase_space",
    "cooling_power_w_per_beam",
    "repump_power_w_per_beam",
    "loading_rate_mean_atoms_per_s",
    "loading_rate_sample_std_atoms_per_s",
    "loading_rate_disc_cluster_sem_atoms_per_s",
    "loading_rate_t95_lower_atoms_per_s",
    "loading_rate_t95_upper_atoms_per_s",
    "loading_rate_t95_half_width_atoms_per_s",
    "student_t_critical_95",
    "capture_velocity_mean_m_per_s",
    "capture_velocity_min_m_per_s",
    "capture_velocity_max_m_per_s",
    "zero_capture_velocity_count",
    "valid_bracket_count",
    "initial_point_fraction_outside_escape_sphere",
    "statistics_directory",
    "figures_directory",
    "capture_cross_section_csv",
    "capture_cross_section_plot",
    "loading_rate_by_disc_csv",
    "geometry_plot",
)


@dataclass(frozen=True, slots=True)
class RadiusStudyPaths:
    """Output paths for the multi-radius study and its confirmation run."""

    statistics: Path
    figures: Path

    @property
    def radii_statistics(self) -> Path:
        return self.statistics / "radii"

    @property
    def radii_figures(self) -> Path:
        return self.figures / "radii"

    @property
    def confirmation_statistics(self) -> Path:
        return self.statistics / "confirmation"

    @property
    def confirmation_figures(self) -> Path:
        return self.figures / "confirmation"

    @property
    def aggregate_csv(self) -> Path:
        return self.statistics / "loading_rate_vs_sampling_disc_radius.csv"

    @property
    def metadata_json(self) -> Path:
        return self.statistics / "study_metadata.json"

    @property
    def convergence_json(self) -> Path:
        return self.statistics / "convergence_fit.json"

    @property
    def confirmation_json(self) -> Path:
        return self.statistics / "confirmation_result.json"

    @property
    def loading_plot(self) -> Path:
        return self.figures / "loading_rate_vs_sampling_disc_radius.png"

    @property
    def combined_cross_section_plot(self) -> Path:
        return self.figures / "capture_cross_section_vs_velocity_all_radii.png"


def default_radius_study_paths(root: Path | None = None) -> RadiusStudyPaths:
    paths = multilevel_mot_paths(root)
    return RadiusStudyPaths(
        statistics=paths["statistics"] / STUDY_NAME,
        figures=paths["figures"] / STUDY_NAME,
    )


def _radius_tag(radius_mm: float) -> str:
    value = f"{float(radius_mm):g}".replace(".", "p")
    return f"radius_{value}_mm"


def _radius_paths(
    paths: RadiusStudyPaths,
    radius_mm: float,
    *,
    confirmation: bool = False,
) -> StudyPaths:
    tag = _radius_tag(radius_mm)
    statistics_root = (
        paths.confirmation_statistics if confirmation else paths.radii_statistics
    )
    figures_root = paths.confirmation_figures if confirmation else paths.radii_figures
    return StudyPaths(statistics=statistics_root / tag, figures=figures_root / tag)


def search_config_for_radius(
    radius_mm: float,
    *,
    seed: int = BASE_RANDOM_SEED,
    disc_count: int = DISC_COUNT,
    points_per_disc: int = POINTS_PER_DISC,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> RateCaptureSearchConfig:
    """Return the default capture search with only the requested sampling changes."""

    if not np.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    if disc_count <= 0 or points_per_disc <= 0 or worker_count <= 0:
        raise ValueError("disc_count, points_per_disc, and worker_count must be positive")
    return replace(
        default_study_search_config(),
        disc_radius_m=1.0e-3 * float(radius_mm),
        disc_count=int(disc_count),
        points_per_disc=int(points_per_disc),
        seed=int(seed),
        include_center_point=False,
        worker_count=int(worker_count),
        phase_space="full_sphere",
    )


def _atomic_write_text(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8", newline="")
    temporary.replace(path)
    return path


def _atomic_write_json(path: Path, payload: object) -> Path:
    return _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return _atomic_write_text(path, buffer.getvalue())


def full_sphere_geometry_metrics(search: RateCaptureSearchConfig) -> dict[str, object]:
    """Quantify the signed-octant and moment coverage of the random directions."""

    discs, points = generate_study_geometry(search)
    centers = np.asarray([disc.center_position_m for disc in discs], dtype=float)
    unit = centers / np.linalg.norm(centers, axis=1)[:, None]
    octant_index = (
        (unit[:, 0] >= 0.0).astype(int)
        + 2 * (unit[:, 1] >= 0.0).astype(int)
        + 4 * (unit[:, 2] >= 0.0).astype(int)
    )
    octant_counts = np.bincount(octant_index, minlength=8)
    normalized_area = np.asarray(
        [(point.s_m / search.disc_radius_m) ** 2 for point in points],
        dtype=float,
    )
    second_moment = unit.T @ unit / len(unit)
    return {
        "direction_count": len(discs),
        "point_count": len(points),
        "phase_space": search.phase_space,
        "solid_angle_sr": 4.0 * pi,
        "mean_direction": np.mean(unit, axis=0).tolist(),
        "direction_second_moment": second_moment.tolist(),
        "direction_second_moment_eigenvalues": np.linalg.eigvalsh(second_moment).tolist(),
        "signed_octant_counts": octant_counts.tolist(),
        "positive_direction_counts": {
            "x": int(np.count_nonzero(unit[:, 0] >= 0.0)),
            "y": int(np.count_nonzero(unit[:, 1] >= 0.0)),
            "z": int(np.count_nonzero(unit[:, 2] >= 0.0)),
        },
        "normalized_disc_area_coordinate_mean": float(np.mean(normalized_area)),
        "normalized_disc_area_coordinate_quantiles": {
            "q05": float(np.quantile(normalized_area, 0.05)),
            "q50": float(np.quantile(normalized_area, 0.50)),
            "q95": float(np.quantile(normalized_area, 0.95)),
        },
    }


def plot_full_sphere_geometry_with_cooling_beams(
    search: RateCaptureSearchConfig,
    path: Path,
    *,
    cooling_power_w_per_beam: float = COOLING_POWER_W_PER_BEAM,
    repump_power_w_per_beam: float = REPUMP_POWER_W_PER_BEAM,
) -> Path:
    """Plot all direction-disc centers, inward normals, and cooling beam volumes."""

    discs, _ = generate_study_geometry(search)
    _, _, beams = build_27mw_multilevel_configuration(
        cooling_power_w_per_beam=cooling_power_w_per_beam,
        repump_power_w_per_beam=repump_power_w_per_beam,
    )
    cooling_beams = [beam for beam in beams if beam.family == "cooling"]
    centers_mm = 1.0e3 * np.asarray(
        [disc.center_position_m for disc in discs], dtype=float
    )
    inward = np.asarray([disc.incident_unit_vector for disc in discs], dtype=float)

    figure = plt.figure(figsize=(8.6, 7.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    draw_multilevel_mot_beam_volumes(axis, cooling_beams, length_m=40.0e-3)

    radius_mm = 1.0e3 * search.radial_distance_m
    theta = np.linspace(0.0, 2.0 * pi, 37)
    phi = np.linspace(0.0, pi, 19)
    sphere_x = radius_mm * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = radius_mm * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = radius_mm * np.outer(np.ones_like(theta), np.cos(phi))
    axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=4,
        cstride=3,
        color="#6b7280",
        alpha=0.12,
        linewidth=0.45,
    )
    axis.scatter(
        centers_mm[:, 0],
        centers_mm[:, 1],
        centers_mm[:, 2],
        c=centers_mm[:, 2],
        cmap="viridis",
        s=30,
        edgecolors="#111827",
        linewidths=0.25,
        depthshade=False,
    )
    axis.quiver(
        centers_mm[:, 0],
        centers_mm[:, 1],
        centers_mm[:, 2],
        inward[:, 0],
        inward[:, 1],
        inward[:, 2],
        length=2.2,
        normalize=True,
        color="#374151",
        linewidth=0.45,
        alpha=0.58,
        arrow_length_ratio=0.28,
    )
    axis.scatter([0.0], [0.0], [0.0], color="#111827", s=42, depthshade=False)
    extent = 21.0
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title=(
            f"Full-sphere launch directions: {search.disc_count} random disc centers\n"
            f"sampling-disc radius = {1e3 * search.disc_radius_m:g} mm; "
            f"27 mW cooling beams"
        ),
    )
    axis.view_init(elev=24.0, azim=36.0)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#10b981",
                markeredgecolor="#111827",
                label="random disc center",
            ),
            Line2D([0], [0], color="#374151", label="inward launch normal"),
            Patch(facecolor="#f9a8d4", alpha=0.35, label="x cooling-beam pair"),
            Patch(facecolor="#93c5fd", alpha=0.35, label="y cooling-beam pair"),
            Patch(facecolor="#86efac", alpha=0.35, label="z cooling-beam pair"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def initial_point_fraction_outside_escape_sphere(
    disc_radius_m: float,
    radial_distance_m: float,
    escape_radius_m: float,
) -> float:
    """Return the uniform-disc area fraction initially beyond the escape sphere."""

    if disc_radius_m <= 0.0 or radial_distance_m <= 0.0 or escape_radius_m <= 0.0:
        raise ValueError("all radii must be positive")
    if escape_radius_m <= radial_distance_m:
        return 1.0
    transverse_limit_squared = escape_radius_m**2 - radial_distance_m**2
    if disc_radius_m**2 <= transverse_limit_squared:
        return 0.0
    return 1.0 - transverse_limit_squared / disc_radius_m**2


def _summary_row(
    radius_mm: float,
    seed: int,
    search: RateCaptureSearchConfig,
    summary: Mapping[str, object],
    run_paths: StudyPaths,
    geometry_plot: Path,
) -> dict[str, object]:
    loading = summary["loading_rate"]
    if not isinstance(loading, Mapping):
        raise TypeError("loading_rate summary must be a mapping")
    mean = float(loading["loading_rate_mean_atoms_per_s"])
    lower = float(loading["loading_rate_t95_lower_atoms_per_s"])
    upper = float(loading["loading_rate_t95_upper_atoms_per_s"])
    return {
        "sampling_disc_radius_mm": float(radius_mm),
        "random_seed": int(seed),
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "sample_count": int(summary["sample_count"]),
        "phase_space": search.phase_space,
        "cooling_power_w_per_beam": COOLING_POWER_W_PER_BEAM,
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
        "loading_rate_mean_atoms_per_s": mean,
        "loading_rate_sample_std_atoms_per_s": float(
            loading["loading_rate_sample_std_atoms_per_s"]
        ),
        "loading_rate_disc_cluster_sem_atoms_per_s": float(
            loading["loading_rate_disc_cluster_sem_atoms_per_s"]
        ),
        "loading_rate_t95_lower_atoms_per_s": lower,
        "loading_rate_t95_upper_atoms_per_s": upper,
        "loading_rate_t95_half_width_atoms_per_s": 0.5 * (upper - lower),
        "student_t_critical_95": float(loading["student_t_critical_95"]),
        "capture_velocity_mean_m_per_s": float(
            summary["capture_velocity_mean_m_per_s"]
        ),
        "capture_velocity_min_m_per_s": float(
            summary["capture_velocity_min_m_per_s"]
        ),
        "capture_velocity_max_m_per_s": float(
            summary["capture_velocity_max_m_per_s"]
        ),
        "zero_capture_velocity_count": int(summary["zero_capture_velocity_count"]),
        "valid_bracket_count": int(summary["valid_bracket_count"]),
        "initial_point_fraction_outside_escape_sphere": (
            initial_point_fraction_outside_escape_sphere(
                search.disc_radius_m,
                search.radial_distance_m,
                search.escape_radius_m,
            )
        ),
        "statistics_directory": str(run_paths.statistics.resolve()),
        "figures_directory": str(run_paths.figures.resolve()),
        "capture_cross_section_csv": str(run_paths.spectrum_csv.resolve()),
        "capture_cross_section_plot": str(run_paths.cross_section_png.resolve()),
        "loading_rate_by_disc_csv": str(run_paths.loading_by_disc_csv.resolve()),
        "geometry_plot": str(geometry_plot.resolve()),
    }


def run_one_radius(
    radius_mm: float,
    paths: RadiusStudyPaths,
    *,
    workers: int = DEFAULT_WORKER_COUNT,
    seed: int = BASE_RANDOM_SEED,
    disc_count: int = DISC_COUNT,
    points_per_disc: int = POINTS_PER_DISC,
    resume: bool = True,
    analyze_only: bool = False,
    confirmation: bool = False,
) -> dict[str, object]:
    """Run, resume, or analyze one full-sphere radius configuration."""

    search = search_config_for_radius(
        radius_mm,
        seed=seed,
        disc_count=disc_count,
        points_per_disc=points_per_disc,
        worker_count=workers,
    )
    run_paths = _radius_paths(paths, radius_mm, confirmation=confirmation)
    run_paths.statistics.mkdir(parents=True, exist_ok=True)
    run_paths.figures.mkdir(parents=True, exist_ok=True)
    geometry_plot = run_paths.figures / "full_sphere_launch_geometry_with_cooling_beams.png"
    plot_full_sphere_geometry_with_cooling_beams(search, geometry_plot)
    _atomic_write_json(
        run_paths.statistics / "geometry_coverage.json",
        full_sphere_geometry_metrics(search),
    )
    context = (
        f"r = {radius_mm:g} mm)\n("
        "27 mW cooling; 0.1 mW repump; full-sphere directions"
    )
    summary = run_power_loading_study(
        search,
        worker_count=workers,
        output_directory=run_paths.statistics,
        figure_directory=run_paths.figures,
        resume=resume,
        analyze_only=analyze_only,
        cooling_power_w_per_beam=COOLING_POWER_W_PER_BEAM,
        repump_power_w_per_beam=REPUMP_POWER_W_PER_BEAM,
        study_name=(
            f"{STUDY_NAME}_confirmation_{_radius_tag(radius_mm)}"
            if confirmation
            else f"{STUDY_NAME}_{_radius_tag(radius_mm)}"
        ),
        progress_every=PROGRESS_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        plot_context=context,
    )
    return _summary_row(
        radius_mm,
        seed,
        search,
        summary,
        run_paths,
        geometry_plot,
    )


def _saturation_curve(
    radius_mm: np.ndarray | float,
    asymptote_atoms_per_s: float,
    scale_mm: float,
    shape: float,
) -> np.ndarray:
    radius = np.asarray(radius_mm, dtype=float)
    return asymptote_atoms_per_s * (
        1.0 - np.exp(-np.power(np.maximum(radius, 0.0) / scale_mm, shape))
    )


def _paired_disc_mean_covariance(
    rows: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, int, float] | None:
    """Estimate the covariance of radius means from paired direction discs."""

    if any(not row.get("loading_rate_by_disc_csv") for row in rows):
        return None
    disc_rate_maps: list[dict[int, float]] = []
    for row in rows:
        path = Path(str(row["loading_rate_by_disc_csv"]))
        if not path.is_file():
            raise FileNotFoundError(f"paired loading-rate file is missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        rate_map = {
            int(item["disc_index"]): float(item["loading_rate_atoms_per_s"])
            for item in csv_rows
        }
        if len(rate_map) != len(csv_rows):
            raise ValueError(f"duplicate disc index in paired loading-rate file: {path}")
        disc_rate_maps.append(rate_map)
    disc_indices = sorted(disc_rate_maps[0])
    if len(disc_indices) < 2 or any(
        sorted(rate_map) != disc_indices for rate_map in disc_rate_maps[1:]
    ):
        raise ValueError("paired radius runs do not contain the same direction-disc indices")
    matrix = np.asarray(
        [
            [rate_map[disc_index] for rate_map in disc_rate_maps]
            for disc_index in disc_indices
        ],
        dtype=float,
    )
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1)) / len(
        disc_indices
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(eigenvalues)), 1.0)
    eigenvalue_floor = scale * 1.0e-10
    regularization = max(0.0, eigenvalue_floor - float(np.min(eigenvalues)))
    if regularization > 0.0:
        covariance = covariance + regularization * np.eye(len(rows))
    return covariance, len(disc_indices), regularization


def fit_loading_rate_convergence(
    rows: Sequence[Mapping[str, object]],
    *,
    convergence_fraction: float = CONVERGENCE_FRACTION,
) -> dict[str, object]:
    """Fit a saturating curve and select its convergence radius conservatively."""

    if len(rows) < 4:
        raise ValueError("at least four radius points are required for a convergence fit")
    if not 0.5 < convergence_fraction < 1.0:
        raise ValueError("convergence_fraction must lie between 0.5 and 1")
    ordered = sorted(rows, key=lambda row: float(row["sampling_disc_radius_mm"]))
    radius = np.asarray(
        [float(row["sampling_disc_radius_mm"]) for row in ordered], dtype=float
    )
    rate = np.asarray(
        [float(row["loading_rate_mean_atoms_per_s"]) for row in ordered], dtype=float
    )
    sem = np.asarray(
        [
            float(row["loading_rate_disc_cluster_sem_atoms_per_s"])
            for row in ordered
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("fit inputs must be finite and loading rates non-negative")
    positive_sem = sem[np.isfinite(sem) & (sem > 0.0)]
    replacement_sem = (
        float(np.median(positive_sem)) if len(positive_sem) else max(np.max(rate) * 0.05, 1.0)
    )
    fit_sem = np.where(np.isfinite(sem) & (sem > 0.0), sem, replacement_sem)
    paired_covariance = _paired_disc_mean_covariance(ordered)
    if paired_covariance is None:
        fit_sigma: np.ndarray = fit_sem
        weighting_description = "disc-cluster standard error at each radius"
        paired_disc_count = None
        covariance_regularization = None
    else:
        fit_sigma, paired_disc_count, covariance_regularization = paired_covariance
        weighting_description = (
            "full covariance of the six means estimated from paired direction-disc "
            "loading rates"
        )
    maximum_rate = max(float(np.max(rate)), 1.0)
    initial = (1.10 * maximum_rate, max(float(np.median(radius)), 1.0), 2.0)
    lower = (maximum_rate, 0.05, 0.25)
    upper = (10.0 * maximum_rate, 10.0 * float(np.max(radius)), 8.0)
    try:
        parameters, covariance = curve_fit(
            _saturation_curve,
            radius,
            rate,
            p0=initial,
            sigma=fit_sigma,
            absolute_sigma=True,
            bounds=(lower, upper),
            maxfev=50_000,
        )
        if not (
            np.all(np.isfinite(parameters)) and np.all(np.isfinite(covariance))
        ):
            return {
                "model": "R(r)=R_inf*[1-exp(-(r/r_scale)^p)]",
                "weighted_by": weighting_description,
                "paired_disc_count": paired_disc_count,
                "covariance_regularization_atoms2_per_s2": covariance_regularization,
                "convergence_fraction": convergence_fraction,
                "convergence_ascertainable": False,
                "selected_confirmation_radius_mm": 12.0,
                "fallback_used": True,
                "fallback_reason": (
                    "fit returned a non-finite parameter covariance, so convergence "
                    "was not considered established"
                ),
            }
        parameter_sem = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        asymptote, scale, shape = (float(value) for value in parameters)
        convergence_radius = float(
            scale * (-log(1.0 - convergence_fraction)) ** (1.0 / shape)
        )
        convergence_log_factor = log(-log(1.0 - convergence_fraction))
        convergence_radius_gradient = np.asarray(
            [
                0.0,
                convergence_radius / scale,
                -convergence_radius * convergence_log_factor / shape**2,
            ],
            dtype=float,
        )
        convergence_radius_variance = float(
            convergence_radius_gradient @ covariance @ convergence_radius_gradient
        )
        convergence_radius_sem = float(
            np.sqrt(max(convergence_radius_variance, 0.0))
        )
        convergence_radius_upper_95 = float(
            convergence_radius
            + CONVERGENCE_RADIUS_UPPER_Z * convergence_radius_sem
        )
        relative_asymptote_sem = float(parameter_sem[0] / asymptote)
        fit_finite = bool(
            np.isfinite(convergence_radius)
            and np.isfinite(convergence_radius_sem)
            and np.isfinite(convergence_radius_upper_95)
        )
        if not fit_finite:
            return {
                "model": "R(r)=R_inf*[1-exp(-(r/r_scale)^p)]",
                "weighted_by": weighting_description,
                "paired_disc_count": paired_disc_count,
                "covariance_regularization_atoms2_per_s2": covariance_regularization,
                "convergence_fraction": convergence_fraction,
                "convergence_ascertainable": False,
                "selected_confirmation_radius_mm": 12.0,
                "fallback_used": True,
                "fallback_reason": (
                    "fit returned a non-finite parameter covariance, so convergence "
                    "was not considered established"
                ),
            }
        residual = rate - _saturation_curve(radius, *parameters)
        weighted_chi_squared = (
            float(residual @ np.linalg.solve(fit_sigma, residual))
            if fit_sigma.ndim == 2
            else float(np.sum((residual / fit_sigma) ** 2))
        )
        degrees_of_freedom = max(0, len(radius) - len(parameters))
        chi_squared_critical = (
            float(
                chi_squared_distribution.ppf(
                    FIT_GOODNESS_CONFIDENCE,
                    df=degrees_of_freedom,
                )
            )
            if degrees_of_freedom > 0
            else 0.0
        )
        goodness_of_fit_accepted = bool(
            degrees_of_freedom > 0
            and np.isfinite(weighted_chi_squared)
            and weighted_chi_squared <= chi_squared_critical
        )

        mean_covariance = (
            fit_sigma if fit_sigma.ndim == 2 else np.diag(np.square(fit_sigma))
        )
        adjacent_difference_variance = np.asarray(
            [
                mean_covariance[index, index]
                + mean_covariance[index + 1, index + 1]
                - 2.0 * mean_covariance[index, index + 1]
                for index in range(len(radius) - 1)
            ],
            dtype=float,
        )
        adjacent_difference_sem = np.sqrt(
            np.maximum(adjacent_difference_variance, 0.0)
        )
        adjacent_decrease = np.maximum(rate[:-1] - rate[1:], 0.0)
        adjacent_downward_z = np.divide(
            adjacent_decrease,
            adjacent_difference_sem,
            out=np.where(
                adjacent_decrease > 0.0,
                np.full_like(adjacent_decrease, np.inf),
                0.0,
            ),
            where=adjacent_difference_sem > 0.0,
        )
        maximum_adjacent_downward_z = float(np.max(adjacent_downward_z))
        no_significant_downward_step = bool(
            maximum_adjacent_downward_z <= MAX_ADJACENT_DOWNWARD_Z
        )

        largest_radius_upper = float(
            ordered[-1].get(
                "loading_rate_t95_upper_atoms_per_s",
                rate[-1] + 1.96 * fit_sem[-1],
            )
        )
        if not np.isfinite(largest_radius_upper):
            largest_radius_upper = float(rate[-1] + 1.96 * fit_sem[-1])
        largest_radius_asymptote_fraction = float(rate[-1] / asymptote)
        largest_radius_supports_plateau = bool(
            largest_radius_asymptote_fraction
            >= MIN_LARGEST_RADIUS_ASYMPTOTE_FRACTION
            and largest_radius_upper >= convergence_fraction * asymptote
        )

        failed_criteria: list[str] = []
        if convergence_radius_upper_95 > float(np.max(radius)):
            failed_criteria.append(
                "upper 95% uncertainty bound on the fitted 95%-of-asymptote "
                "radius exceeds 30 mm"
            )
        if relative_asymptote_sem > MAX_RELATIVE_ASYMPTOTE_SEM:
            failed_criteria.append("relative asymptote standard error exceeds 50%")
        if asymptote > 2.0 * maximum_rate:
            failed_criteria.append("fitted asymptote exceeds twice the largest observation")
        if not goodness_of_fit_accepted:
            failed_criteria.append("saturating-curve chi-squared goodness-of-fit test failed")
        if not no_significant_downward_step:
            failed_criteria.append("an adjacent loading-rate decrease is significant at about 95%")
        if not largest_radius_supports_plateau:
            failed_criteria.append("the 30 mm observation does not empirically support a plateau")
        ascertainable = not failed_criteria
        selected_radius = (
            float(
                np.clip(
                    ceil(convergence_radius_upper_95),
                    np.min(radius),
                    np.max(radius),
                )
            )
            if ascertainable
            else 12.0
        )
        return {
            "model": "R(r)=R_inf*[1-exp(-(r/r_scale)^p)]",
            "weighted_by": weighting_description,
            "paired_disc_count": paired_disc_count,
            "covariance_regularization_atoms2_per_s2": covariance_regularization,
            "parameters": {
                "asymptote_atoms_per_s": asymptote,
                "scale_mm": scale,
                "shape": shape,
            },
            "parameter_standard_errors": {
                "asymptote_atoms_per_s": float(parameter_sem[0]),
                "scale_mm": float(parameter_sem[1]),
                "shape": float(parameter_sem[2]),
            },
            "covariance": covariance.tolist(),
            "convergence_fraction": convergence_fraction,
            "convergence_radius_mm": convergence_radius,
            "convergence_radius_standard_error_mm": convergence_radius_sem,
            "convergence_radius_upper_95_mm": convergence_radius_upper_95,
            "convergence_radius_upper_z": CONVERGENCE_RADIUS_UPPER_Z,
            "relative_asymptote_standard_error": relative_asymptote_sem,
            "maximum_allowed_relative_asymptote_standard_error": MAX_RELATIVE_ASYMPTOTE_SEM,
            "weighted_chi_squared": weighted_chi_squared,
            "degrees_of_freedom": degrees_of_freedom,
            "chi_squared_goodness_confidence": FIT_GOODNESS_CONFIDENCE,
            "chi_squared_critical_value": chi_squared_critical,
            "goodness_of_fit_accepted": goodness_of_fit_accepted,
            "maximum_adjacent_downward_z": maximum_adjacent_downward_z,
            "maximum_allowed_adjacent_downward_z": MAX_ADJACENT_DOWNWARD_Z,
            "adjacent_decrease_familywise_confidence": 0.95,
            "no_significant_downward_step": no_significant_downward_step,
            "largest_radius_asymptote_fraction": largest_radius_asymptote_fraction,
            "minimum_largest_radius_asymptote_fraction": MIN_LARGEST_RADIUS_ASYMPTOTE_FRACTION,
            "largest_radius_t95_upper_atoms_per_s": largest_radius_upper,
            "largest_radius_supports_plateau": largest_radius_supports_plateau,
            "failed_convergence_criteria": failed_criteria,
            "convergence_ascertainable": ascertainable,
            "selected_confirmation_radius_mm": selected_radius,
            "fallback_used": not ascertainable,
            "fallback_reason": (
                None
                if ascertainable
                else "; ".join(failed_criteria)
            ),
        }
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {
            "model": "R(r)=R_inf*[1-exp(-(r/r_scale)^p)]",
            "weighted_by": (
                weighting_description
                if "weighting_description" in locals()
                else "disc-cluster standard error at each radius"
            ),
            "convergence_fraction": convergence_fraction,
            "convergence_ascertainable": False,
            "selected_confirmation_radius_mm": 12.0,
            "fallback_used": True,
            "fallback_reason": f"fit failed: {type(exc).__name__}: {exc}",
        }


def plot_loading_rate_vs_radius(
    rows: Sequence[Mapping[str, object]],
    fit: Mapping[str, object],
    path: Path,
) -> Path:
    """Plot six loading rates with direction-clustered 95% Student-t bars."""

    ordered = sorted(rows, key=lambda row: float(row["sampling_disc_radius_mm"]))
    radius = np.asarray(
        [float(row["sampling_disc_radius_mm"]) for row in ordered], dtype=float
    )
    rate = np.asarray(
        [float(row["loading_rate_mean_atoms_per_s"]) for row in ordered], dtype=float
    )
    lower = np.asarray(
        [float(row["loading_rate_t95_lower_atoms_per_s"]) for row in ordered],
        dtype=float,
    )
    upper = np.asarray(
        [float(row["loading_rate_t95_upper_atoms_per_s"]) for row in ordered],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8.7, 5.9), constrained_layout=True)
    axis.errorbar(
        radius,
        rate / 1.0e6,
        yerr=np.vstack(((rate - lower) / 1.0e6, (upper - rate) / 1.0e6)),
        fmt="o",
        color="#0f766e",
        ecolor="#0f766e",
        capsize=4,
        markersize=6,
        linewidth=1.5,
        label="mean loading rate; 95% direction-clustered t interval",
    )
    parameters = fit.get("parameters")
    if isinstance(parameters, Mapping):
        fit_radius = np.linspace(0.0, max(32.0, 1.05 * float(np.max(radius))), 500)
        fit_rate = _saturation_curve(
            fit_radius,
            float(parameters["asymptote_atoms_per_s"]),
            float(parameters["scale_mm"]),
            float(parameters["shape"]),
        )
        axis.plot(
            fit_radius,
            fit_rate / 1.0e6,
            color="#7c3aed",
            linewidth=2.0,
            label="weighted saturating fit",
        )
        if bool(fit.get("convergence_ascertainable")):
            convergence_radius = float(fit["convergence_radius_mm"])
            convergence_upper = float(fit["convergence_radius_upper_95_mm"])
            selected_radius = float(fit["selected_confirmation_radius_mm"])
            axis.axvline(
                convergence_radius,
                color="#9f4a13",
                linestyle="--",
                linewidth=1.4,
                label=f"fit 95%-of-asymptote radius = {convergence_radius:.2f} mm",
            )
            axis.axvline(
                selected_radius,
                color="#6b7280",
                linestyle=":",
                linewidth=1.4,
                label=(
                    f"confirmation = {selected_radius:g} mm "
                    f"(ceil of upper bound {convergence_upper:.2f} mm)"
                ),
            )
    if not bool(fit.get("convergence_ascertainable")):
        fallback_radius = float(fit["selected_confirmation_radius_mm"])
        axis.axvline(
            fallback_radius,
            color="#6b7280",
            linestyle=":",
            linewidth=1.4,
            label=f"conservative fallback confirmation = {fallback_radius:g} mm",
        )
    axis.set(
        xlabel="Sampling-disc radius [mm]",
        ylabel=r"Loading rate [$10^6$ atoms s$^{-1}$]",
        title=(
            "24-State MOT Loading Rate vs Sampling-Disc Radius\n"
            "27 mW per cooling beam; 0.1 mW per repump beam; full-sphere directions"
        ),
    )
    axis.set_xticks(radius)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.02,
        0.02,
        "Bars: t(0.975, 99) × SD of 100 disc-level rates / √100",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def _read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["velocity_m_per_s"]) for row in rows]),
        1.0e6 * np.asarray([float(row["capture_cross_section_m2"]) for row in rows]),
        1.0e6
        * np.asarray([float(row["capture_cross_section_t95_lower_m2"]) for row in rows]),
        1.0e6
        * np.asarray([float(row["capture_cross_section_t95_upper_m2"]) for row in rows]),
    )


def plot_combined_cross_sections(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Overlay the six mean capture-cross-section spectra."""

    figure, axis = plt.subplots(figsize=(9.0, 6.1), constrained_layout=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.92, len(rows))
    )
    for color, row in zip(
        colors,
        sorted(rows, key=lambda item: float(item["sampling_disc_radius_mm"])),
    ):
        velocity, mean, lower, upper = _read_spectrum(
            Path(str(row["capture_cross_section_csv"]))
        )
        radius = float(row["sampling_disc_radius_mm"])
        axis.plot(
            velocity,
            mean,
            color=color,
            linewidth=1.8,
            label=f"r = {radius:g} mm",
        )
        axis.fill_between(velocity, lower, upper, color=color, alpha=0.08, linewidth=0.0)
    axis.set(
        xlabel="Launch speed [m/s]",
        ylabel=r"Capture cross section [mm$^2$]",
        title=(
            "24-State MOT Capture Cross Section vs Velocity\n"
            "Shading: 95% direction-clustered Student-t interval"
        ),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return path


def run_sampling_disc_radius_study(
    *,
    paths: RadiusStudyPaths | None = None,
    radii_mm: Sequence[float] = SAMPLING_DISC_RADII_MM,
    workers: int = DEFAULT_WORKER_COUNT,
    disc_count: int = DISC_COUNT,
    points_per_disc: int = POINTS_PER_DISC,
    resume: bool = True,
    analyze_only: bool = False,
    run_confirmation: bool = True,
) -> dict[str, object]:
    """Run all radius points, fit convergence, then run an independent confirmation."""

    output = paths or default_radius_study_paths()
    output.statistics.mkdir(parents=True, exist_ok=True)
    output.figures.mkdir(parents=True, exist_ok=True)
    radii = tuple(float(value) for value in radii_mm)
    if len(radii) != len(set(radii)) or any(value <= 0.0 for value in radii):
        raise ValueError("radii_mm must contain distinct positive values")
    rows: list[dict[str, object]] = []
    print(
        f"[disc-radius study] phase 1: {len(radii)} paired radius configurations; "
        f"each {disc_count} full-sphere discs x {points_per_disc} random points; "
        f"27 mW cooling and 0.1 mW repump per beam",
        flush=True,
    )
    for index, radius_mm in enumerate(radii, start=1):
        print(
            f"[disc-radius study] radius {index}/{len(radii)}: r={radius_mm:g} mm",
            flush=True,
        )
        row = run_one_radius(
            radius_mm,
            output,
            workers=workers,
            seed=BASE_RANDOM_SEED,
            disc_count=disc_count,
            points_per_disc=points_per_disc,
            resume=resume,
            analyze_only=analyze_only,
        )
        rows.append(row)
        _atomic_write_csv(output.aggregate_csv, rows, AGGREGATE_FIELDNAMES)
        print(
            f"[disc-radius study] r={radius_mm:g} mm complete: "
            f"R={float(row['loading_rate_mean_atoms_per_s']):.6g} atoms/s; "
            f"95% t interval=[{float(row['loading_rate_t95_lower_atoms_per_s']):.6g}, "
            f"{float(row['loading_rate_t95_upper_atoms_per_s']):.6g}]",
            flush=True,
        )

    fit = fit_loading_rate_convergence(rows)
    _atomic_write_json(output.convergence_json, fit)
    plot_loading_rate_vs_radius(rows, fit, output.loading_plot)
    plot_combined_cross_sections(rows, output.combined_cross_section_plot)
    selected_radius = float(fit["selected_confirmation_radius_mm"])
    print(
        f"[disc-radius study] phase 1 complete; convergence ascertainable="
        f"{fit['convergence_ascertainable']}; selected confirmation radius="
        f"{selected_radius:g} mm",
        flush=True,
    )

    confirmation_row: dict[str, object] | None = None
    if run_confirmation:
        print(
            f"[disc-radius study] phase 2: independent full-sphere confirmation at "
            f"r={selected_radius:g} mm with seed {CONFIRMATION_RANDOM_SEED}",
            flush=True,
        )
        confirmation_row = run_one_radius(
            selected_radius,
            output,
            workers=workers,
            seed=CONFIRMATION_RANDOM_SEED,
            disc_count=disc_count,
            points_per_disc=points_per_disc,
            resume=resume,
            analyze_only=analyze_only,
            confirmation=True,
        )
        _atomic_write_json(output.confirmation_json, confirmation_row)
        print(
            f"[disc-radius study] confirmation complete: "
            f"R={float(confirmation_row['loading_rate_mean_atoms_per_s']):.6g} atoms/s; "
            f"95% t interval=[{float(confirmation_row['loading_rate_t95_lower_atoms_per_s']):.6g}, "
            f"{float(confirmation_row['loading_rate_t95_upper_atoms_per_s']):.6g}]",
            flush=True,
        )

    default_search = search_config_for_radius(
        radii[0],
        disc_count=disc_count,
        points_per_disc=points_per_disc,
        worker_count=workers,
    )
    metadata = {
        "schema_version": 1,
        "status": "completed" if run_confirmation else "phase_1_completed",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "24-state repumper-enabled adiabatic population-rate-equation MOT",
        "cooling_power_w_per_beam": COOLING_POWER_W_PER_BEAM,
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
        "sampling_disc_radii_mm": list(radii),
        "disc_count_per_radius": disc_count,
        "points_per_disc": points_per_disc,
        "capture_threshold_count_per_radius": disc_count * points_per_disc,
        "phase_space": "full_sphere",
        "direction_sampling": "cos(theta) uniform on [-1,1], phi uniform on [0,2pi)",
        "disc_point_sampling": "s=R*sqrt(U), theta_prime uniform on [0,2pi)",
        "common_random_numbers": (
            "The six phase-1 radii reuse one seeded normalized random geometry; "
            "disc offsets scale with radius. The confirmation run uses an independent seed."
        ),
        "cross_section_normalization": (
            "direction-averaged projected area; no 4pi or octant multiplicity factor"
        ),
        "error_bar_definition": (
            "For each radius, construct one loading-rate estimate per incident-direction "
            "disc from its 100 impact-point thresholds. The plotted 95% error bar is "
            "t_(0.975,99) times the sample standard deviation of those 100 disc-level "
            "rates divided by sqrt(100). Capture-spectrum bands use the same clustering "
            "at each velocity."
        ),
        "trapped_criterion": (
            "continuous residence inside the central 2 mm-radius core for at least 5 ms, "
            "or two core entries with an intervening exit"
        ),
        "capture_search_config_for_first_radius": asdict(default_search),
        "escape_sphere_note": (
            "The launch-disc center radius and 30 mm escape radius remain at defaults. "
            "For r=30 mm, 25% of uniform-area launch points begin outside the escape "
            "sphere but travel inward, so they are not immediately classified as escaped."
        ),
        "fit": fit,
        "outputs": {
            "aggregate_csv": str(output.aggregate_csv.resolve()),
            "loading_rate_plot": str(output.loading_plot.resolve()),
            "combined_cross_section_plot": str(
                output.combined_cross_section_plot.resolve()
            ),
            "convergence_fit_json": str(output.convergence_json.resolve()),
            "confirmation_result_json": (
                str(output.confirmation_json.resolve()) if run_confirmation else None
            ),
        },
    }
    _atomic_write_json(output.metadata_json, metadata)
    return {
        "radius_rows": rows,
        "fit": fit,
        "confirmation": confirmation_row,
        "metadata": metadata,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 27 mW, full-sphere, 100-disc x 100-point sampling-disc-radius "
            "study and its post-fit confirmation"
        )
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--statistics-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    paths = None
    if args.statistics_dir is not None or args.figures_dir is not None:
        defaults = default_radius_study_paths()
        paths = RadiusStudyPaths(
            statistics=(
                args.statistics_dir
                if args.statistics_dir is not None
                else defaults.statistics
            ),
            figures=(
                args.figures_dir if args.figures_dir is not None else defaults.figures
            ),
        )
    run_sampling_disc_radius_study(
        paths=paths,
        workers=args.workers,
        resume=args.resume,
        analyze_only=args.analyze_only,
        run_confirmation=not args.skip_confirmation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_FIELDNAMES",
    "BASE_RANDOM_SEED",
    "CHECKPOINT_EVERY",
    "CONFIRMATION_RANDOM_SEED",
    "CONVERGENCE_FRACTION",
    "COOLING_POWER_W_PER_BEAM",
    "DEFAULT_WORKER_COUNT",
    "DISC_COUNT",
    "POINTS_PER_DISC",
    "PROGRESS_EVERY",
    "RadiusStudyPaths",
    "SAMPLING_DISC_RADII_MM",
    "STUDY_NAME",
    "default_radius_study_paths",
    "fit_loading_rate_convergence",
    "full_sphere_geometry_metrics",
    "initial_point_fraction_outside_escape_sphere",
    "plot_combined_cross_sections",
    "plot_full_sphere_geometry_with_cooling_beams",
    "plot_loading_rate_vs_radius",
    "run_one_radius",
    "run_sampling_disc_radius_study",
    "search_config_for_radius",
]
