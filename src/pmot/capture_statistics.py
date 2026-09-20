"""Model-neutral capture-threshold statistics, persistence, and plots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .launch_geometry import DiscSample, Vec3


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


def summarize_capture_velocity_samples(
    samples: list[CaptureVelocitySample],
    search_config: Any,
) -> dict[str, object]:
    """Return summary statistics for a completed sampling run."""

    capture_velocities = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples], dtype=float
    )
    resolutions = np.asarray(
        [sample.velocity_resolution_m_per_s for sample in samples], dtype=float
    )
    return {
        "sample_count": int(len(samples)),
        "disc_count": int(search_config.disc_count),
        "points_per_disc": int(search_config.points_per_disc),
        "capture_velocity_mean_m_per_s": (
            float(np.mean(capture_velocities)) if len(samples) else None
        ),
        "capture_velocity_std_m_per_s": (
            float(np.std(capture_velocities)) if len(samples) else None
        ),
        "capture_velocity_min_m_per_s": (
            float(np.min(capture_velocities)) if len(samples) else None
        ),
        "capture_velocity_max_m_per_s": (
            float(np.max(capture_velocities)) if len(samples) else None
        ),
        "resolution_mean_m_per_s": (
            float(np.mean(resolutions)) if len(samples) else None
        ),
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
    sampled_upper_bound = max(
        (sample.untrapped_velocity_upper_m_per_s for sample in samples),
        default=velocity_max_m_per_s,
    )
    grid_start = velocity_step_m_per_s * np.floor(
        velocity_min_m_per_s / velocity_step_m_per_s
    )
    grid_stop = velocity_step_m_per_s * np.ceil(
        max(velocity_max_m_per_s, sampled_upper_bound) / velocity_step_m_per_s
    )
    return np.arange(
        grid_start,
        grid_stop + 0.5 * velocity_step_m_per_s,
        velocity_step_m_per_s,
        dtype=float,
    )


def capture_spectrum_from_samples(
    samples: list[CaptureVelocitySample],
    disc_radius_m: float,
    velocity_step_m_per_s: float,
    velocity_min_m_per_s: float,
    velocity_max_m_per_s: float,
) -> list[VelocitySpectrumSample]:
    """Aggregate capture counts and capture cross section versus velocity."""

    velocity_grid = velocity_grid_from_samples(
        samples,
        velocity_step_m_per_s,
        velocity_min_m_per_s,
        velocity_max_m_per_s,
    )
    launched_count = len(samples)
    disc_area_m2 = pi * disc_radius_m**2
    capture_velocities = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples], dtype=float
    )
    spectrum: list[VelocitySpectrumSample] = []
    for velocity_m_per_s in velocity_grid:
        captured_count = int(
            np.count_nonzero(capture_velocities >= velocity_m_per_s - 1.0e-12)
        )
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
            writer.writerow(asdict(sample))
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
    velocities = np.asarray(
        [sample.velocity_m_per_s for sample in spectrum], dtype=float
    )
    sigma_mm2 = 1.0e6 * np.asarray(
        [sample.capture_cross_section_m2 for sample in spectrum], dtype=float
    )
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
    velocity_grid = velocity_grid_from_samples(
        samples,
        velocity_step_m_per_s,
        velocity_min_m_per_s,
        velocity_max_m_per_s,
    )
    if len(velocity_grid) == 1:
        velocity_edges = np.asarray(
            [
                velocity_grid[0] - 0.5 * velocity_step_m_per_s,
                velocity_grid[0] + 0.5 * velocity_step_m_per_s,
            ]
        )
    else:
        velocity_edges = np.empty(len(velocity_grid) + 1, dtype=float)
        velocity_edges[1:-1] = 0.5 * (velocity_grid[:-1] + velocity_grid[1:])
        velocity_edges[0] = max(
            0.0, velocity_grid[0] - 0.5 * (velocity_grid[1] - velocity_grid[0])
        )
        velocity_edges[-1] = velocity_grid[-1] + 0.5 * (
            velocity_grid[-1] - velocity_grid[-2]
        )
    s_edges_m = np.linspace(0.0, disc_radius_m, s_bin_count + 1)
    probability_grid = np.full(
        (len(velocity_grid), s_bin_count), np.nan, dtype=float
    )
    sample_s = np.asarray([sample.s_m for sample in samples], dtype=float)
    sample_vc = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples], dtype=float
    )

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
            probability_grid[velocity_index, s_index] = (
                np.count_nonzero(bin_vc >= velocity_m_per_s - 1.0e-12)
                / len(bin_vc)
            )

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
    search_config: Any,
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
        json.dump(
            summarize_capture_velocity_samples(samples, search_config),
            handle,
            indent=2,
        )
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
                    initial_position_m=(
                        float(row["x0_m"]),
                        float(row["y0_m"]),
                        float(row["z0_m"]),
                    ),
                    incident_unit_vector=(
                        float(row["vx_hat"]),
                        float(row["vy_hat"]),
                        float(row["vz_hat"]),
                    ),
                    capture_velocity_m_per_s=float(row["capture_velocity_m_per_s"]),
                    velocity_resolution_m_per_s=float(
                        row["velocity_resolution_m_per_s"]
                    ),
                    trapped_velocity_lower_m_per_s=float(
                        row["trapped_velocity_lower_m_per_s"]
                    ),
                    untrapped_velocity_upper_m_per_s=float(
                        row["untrapped_velocity_upper_m_per_s"]
                    ),
                    lower_classification=row["lower_classification"],
                    upper_classification=row["upper_classification"],
                    lower_entered_trap_core=(
                        row["lower_entered_trap_core"].lower() == "true"
                    ),
                    upper_entered_trap_core=(
                        row["upper_entered_trap_core"].lower() == "true"
                    ),
                    lower_core_entry_count=int(
                        row.get(
                            "lower_core_entry_count",
                            row.get("lower_turning_point_count", 0),
                        )
                    ),
                    upper_core_entry_count=int(
                        row.get(
                            "upper_core_entry_count",
                            row.get("upper_turning_point_count", 0),
                        )
                    ),
                )
            )
    return samples


def run_capture_velocity_analysis(
    samples: list[CaptureVelocitySample],
    search_config: Any,
    output_directory: Path,
    figure_directory: Path | None = None,
) -> dict[str, Path]:
    """Generate analysis products from saved capture-velocity thresholds."""

    figures_directory = figure_directory or default_analysis_figure_directory(
        output_directory
    )
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
    capture_velocity = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples], dtype=float
    )
    theta_prime = np.asarray(
        [sample.theta_prime_rad for sample in samples], dtype=float
    )
    scatter = axis.scatter(
        s_mm,
        capture_velocity,
        c=theta_prime,
        cmap="viridis",
        s=26,
        alpha=0.9,
    )
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


__all__ = [
    "CaptureVelocitySample",
    "TrajectoryClassification",
    "VelocitySpectrumSample",
    "capture_spectrum_from_samples",
    "default_analysis_figure_directory",
    "load_capture_velocity_samples",
    "plot_capture_cross_section",
    "plot_capture_probability_heatmap",
    "plot_capture_velocity_vs_radius",
    "run_capture_velocity_analysis",
    "save_capture_spectrum",
    "save_capture_velocity_results",
    "summarize_capture_velocity_samples",
    "velocity_grid_from_samples",
]
