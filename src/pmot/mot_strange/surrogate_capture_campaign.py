"""Time-bounded full-sphere capture survey for the surrogate pMOT field.

This is deliberately a preliminary, aperture-limited diagnostic.  It uses the
physical Section-12 population-rate kernel, deterministic mean force, gravity,
and the effective magnetic field defined in ``surrogate_effective_field``.
It does not turn that surrogate into a state-resolved Stark Hamiltonian.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from ..launch_geometry import sample_disc_points, sample_incident_disc_full_sphere
from ..loading import LOADING_RATE_PREFACTOR, THERMAL_SCALE_M2_PER_S2


OUTCOME_ORIGIN = "origin_trapped"
OUTCOME_OFF_ORIGIN = "off_origin_trapped"
OUTCOME_ESCAPE = "wall_escape"
OUTCOME_INDETERMINATE = "indeterminate_timeout"


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    seed: int = 20260930
    direction_discs: int = 20
    points_per_disc: int = 10
    launch_radius_m: float = 10.0e-3
    disc_radius_m: float = 4.0e-3
    speeds_m_per_s: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0)
    duration_s: float = 20.0e-3
    time_step_s: float = 5.0e-6
    core_radius_m: float = 2.0e-3
    core_residence_s: float = 5.0e-3
    off_origin_window_s: float = 5.0e-3
    off_origin_centroid_min_radius_m: float = 2.0e-3
    off_origin_max_excursion_m: float = 0.75e-3
    off_origin_max_rms_speed_m_per_s: float = 0.10
    off_origin_max_net_displacement_m: float = 0.25e-3
    history_stride: int = 20
    workers: int = 4
    # Stop starting work at 4.9 h.  The remaining 36 minutes cover in-flight
    # trajectories, summaries, plots, and representative convergence audits.
    submission_budget_s: float = 4.9 * 3600.0
    absolute_wall_budget_s: float = 5.5 * 3600.0

    @property
    def sample_area_m2(self) -> float:
        return math.pi * self.disc_radius_m**2


@dataclass(frozen=True, slots=True)
class LaunchTask:
    disc_index: int
    point_index: int
    speed_m_per_s: float
    impact_parameter_m: float
    theta_rad: float
    phi_rad: float
    initial_position_m: tuple[float, float, float]
    incident_unit_vector: tuple[float, float, float]

    @property
    def key(self) -> str:
        return f"d{self.disc_index:03d}_p{self.point_index:03d}_v{self.speed_m_per_s:06.2f}"


_WORKER_CONTEXT = None


def _worker_context():
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        from ..mot_multilevel.configuration import default_multilevel_mot_config
        from ..mot_multilevel.rate_equations import build_rate_equation_model
        from ..mot_multilevel.simulation import build_multilevel_mot_beams

        config = default_multilevel_mot_config()
        _WORKER_CONTEXT = (
            build_rate_equation_model(),
            build_multilevel_mot_beams(config=config),
            config,
        )
    return _WORKER_CONTEXT


def build_launch_tasks(config: CampaignConfig) -> list[LaunchTask]:
    rng = np.random.default_rng(config.seed)
    tasks: list[LaunchTask] = []
    for disc_index in range(config.direction_discs):
        disc = sample_incident_disc_full_sphere(
            disc_index,
            config.launch_radius_m,
            rng,
        )
        points = sample_disc_points(
            disc,
            config.points_per_disc,
            config.disc_radius_m,
            False,
            rng,
        )
        for point in points:
            for speed in config.speeds_m_per_s:
                tasks.append(
                    LaunchTask(
                        disc_index=disc_index,
                        point_index=point.point_index,
                        speed_m_per_s=speed,
                        impact_parameter_m=point.s_m,
                        theta_rad=point.theta_rad,
                        phi_rad=point.phi_rad,
                        initial_position_m=point.initial_position_m,
                        incident_unit_vector=point.incident_unit_vector,
                    )
                )
    return tasks


def _origin_capture(times_s: np.ndarray, positions_m: np.ndarray, config: CampaignConfig):
    inside = np.linalg.norm(positions_m, axis=1) <= config.core_radius_m
    entries = int(np.count_nonzero(inside & ~np.r_[False, inside[:-1]]))
    longest = 0.0
    start: float | None = None
    for time_s, is_inside in zip(times_s, inside):
        if is_inside and start is None:
            start = float(time_s)
        elif not is_inside and start is not None:
            longest = max(longest, float(time_s) - start)
            start = None
    if start is not None:
        longest = max(longest, float(times_s[-1]) - start)
    captured = entries >= 2 or longest >= config.core_residence_s
    return captured, entries, longest


def classify_trajectory(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    termination_reason: str,
    config: CampaignConfig,
) -> dict[str, float | int | str | list[float]]:
    captured, entries, residence_s = _origin_capture(times_s, positions_m, config)
    radii = np.linalg.norm(positions_m, axis=1)
    if captured:
        outcome = OUTCOME_ORIGIN
    elif termination_reason in {"wall_loss", "escaped"}:
        outcome = OUTCOME_ESCAPE
    else:
        window = times_s >= max(0.0, times_s[-1] - config.off_origin_window_s)
        final_positions = positions_m[window]
        final_velocities = velocities_m_per_s[window]
        centroid = np.mean(final_positions, axis=0)
        centroid_radius = float(np.linalg.norm(centroid))
        excursion = float(np.max(np.linalg.norm(final_positions - centroid, axis=1)))
        rms_speed = float(
            np.sqrt(np.mean(np.sum(final_velocities * final_velocities, axis=1)))
        )
        net_displacement = float(np.linalg.norm(final_positions[-1] - final_positions[0]))
        full_window = float(times_s[-1] - times_s[window][0]) >= 0.99 * config.off_origin_window_s
        if (
            full_window
            and centroid_radius > config.off_origin_centroid_min_radius_m
            and excursion <= config.off_origin_max_excursion_m
            and rms_speed <= config.off_origin_max_rms_speed_m_per_s
            and net_displacement <= config.off_origin_max_net_displacement_m
        ):
            outcome = OUTCOME_OFF_ORIGIN
        else:
            outcome = OUTCOME_INDETERMINATE

    window = times_s >= max(0.0, times_s[-1] - config.off_origin_window_s)
    centroid = np.mean(positions_m[window], axis=0)
    excursion = float(np.max(np.linalg.norm(positions_m[window] - centroid, axis=1)))
    rms_speed = float(
        np.sqrt(np.mean(np.sum(velocities_m_per_s[window] ** 2, axis=1)))
    )
    net_displacement = float(np.linalg.norm(positions_m[window][-1] - positions_m[window][0]))
    return {
        "outcome": outcome,
        "core_entries": entries,
        "max_continuous_core_residence_s": residence_s,
        "minimum_radius_m": float(np.min(radii)),
        "final_radius_m": float(radii[-1]),
        "final_speed_m_per_s": float(np.linalg.norm(velocities_m_per_s[-1])),
        "final_centroid_m": [float(value) for value in centroid],
        "final_centroid_radius_m": float(np.linalg.norm(centroid)),
        "final_window_max_excursion_m": excursion,
        "final_window_rms_speed_m_per_s": rms_speed,
        "final_window_net_displacement_m": net_displacement,
    }


def _run_task(task: LaunchTask, config: CampaignConfig, history_dir: str) -> dict:
    from ..mot_multilevel.rate_equations import (
        RateEquationAtomState,
        RateEquationTrajectoryConfig,
        simulate_rate_equation_trajectory,
    )
    from .surrogate_effective_field import inside_surrogate_cell, pmot_effective_B

    model, beams, mot_config = _worker_context()
    initial_velocity = tuple(
        task.speed_m_per_s * value for value in task.incident_unit_vector
    )
    started = time.perf_counter()
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(task.initial_position_m, initial_velocity),
        config.duration_s,
        None,
        beams=beams,
        model=model,
        config=mot_config,
        trajectory_config=RateEquationTrajectoryConfig(time_step_s=config.time_step_s),
        magnetic_field_function=pmot_effective_B,
        spatial_domain_function=inside_surrogate_cell,
    )
    times = np.asarray(record.times_s)
    positions = np.asarray(record.positions_m)
    velocities = np.asarray(record.velocities_m_per_s)
    classification = classify_trajectory(
        times,
        positions,
        velocities,
        record.termination_reason,
        config,
    )
    stride = max(1, config.history_stride)
    selection = np.unique(np.r_[np.arange(0, len(times), stride), len(times) - 1])
    history_path = Path(history_dir) / f"{task.key}.npz"
    np.savez_compressed(
        history_path,
        times_s=times[selection],
        positions_m=positions[selection],
        velocities_m_per_s=velocities[selection],
    )
    return {
        "task_key": task.key,
        "disc_index": task.disc_index,
        "point_index": task.point_index,
        "speed_m_per_s": task.speed_m_per_s,
        "impact_parameter_m": task.impact_parameter_m,
        "theta_rad": task.theta_rad,
        "phi_rad": task.phi_rad,
        "initial_x_m": task.initial_position_m[0],
        "initial_y_m": task.initial_position_m[1],
        "initial_z_m": task.initial_position_m[2],
        "incident_x": task.incident_unit_vector[0],
        "incident_y": task.incident_unit_vector[1],
        "incident_z": task.incident_unit_vector[2],
        "termination_reason": record.termination_reason,
        "simulated_duration_s": float(times[-1]),
        "trajectory_steps": len(times) - 1,
        "runtime_s": time.perf_counter() - started,
        "history_file": history_path.name,
        **{
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in classification.items()
        },
    }


RESULT_COLUMNS = [
    "task_key", "disc_index", "point_index", "speed_m_per_s",
    "impact_parameter_m", "theta_rad", "phi_rad", "initial_x_m", "initial_y_m",
    "initial_z_m", "incident_x", "incident_y", "incident_z", "outcome",
    "termination_reason", "simulated_duration_s", "trajectory_steps", "runtime_s",
    "core_entries", "max_continuous_core_residence_s", "minimum_radius_m",
    "final_radius_m", "final_speed_m_per_s", "final_centroid_m",
    "final_centroid_radius_m", "final_window_max_excursion_m",
    "final_window_rms_speed_m_per_s", "history_file",
    "final_window_net_displacement_m",
]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(data) - 1) * np.std(data, ddof=1) / np.sqrt(len(data)))
    return mean, mean - half, mean + half


def _numeric_rows(rows: list[dict]) -> list[dict]:
    integer_keys = {"disc_index", "point_index", "trajectory_steps", "core_entries"}
    string_keys = {"task_key", "outcome", "termination_reason", "history_file", "final_centroid_m"}
    converted = []
    for row in rows:
        item = dict(row)
        for key, value in row.items():
            if key in string_keys:
                continue
            item[key] = int(value) if key in integer_keys else float(value)
        item["final_centroid_m"] = json.loads(row["final_centroid_m"])
        converted.append(item)
    return converted


def aggregate_results(rows: list[dict], config: CampaignConfig, output_dir: Path) -> dict:
    rows = _numeric_rows(rows)
    complete_discs = sorted(
        disc for disc in range(config.direction_discs)
        if all(
            sum(
                row["disc_index"] == disc and row["speed_m_per_s"] == speed
                for row in rows
            ) == config.points_per_disc
            for speed in config.speeds_m_per_s
        )
    )
    spectra: list[dict] = []
    category_tests = {
        "origin": lambda value: value == OUTCOME_ORIGIN,
        "off_origin": lambda value: value == OUTCOME_OFF_ORIGIN,
        "total_trapped": lambda value: value in {OUTCOME_ORIGIN, OUTCOME_OFF_ORIGIN},
        "escaped": lambda value: value == OUTCOME_ESCAPE,
        "indeterminate": lambda value: value == OUTCOME_INDETERMINATE,
    }
    disc_sigma: dict[tuple[int, float, str], float] = {}
    for speed in config.speeds_m_per_s:
        for category, test in category_tests.items():
            values = []
            for disc in complete_discs:
                subset = [
                    row for row in rows
                    if row["disc_index"] == disc and row["speed_m_per_s"] == speed
                ]
                sigma = config.sample_area_m2 * sum(test(row["outcome"]) for row in subset) / len(subset)
                disc_sigma[(disc, speed, category)] = sigma
                values.append(sigma)
            if values:
                mean, low, high = _mean_ci(values)
                spectra.append(
                    {
                        "speed_m_per_s": speed,
                        "category": category,
                        "independent_direction_discs": len(values),
                        "mean_cross_section_m2": mean,
                        "ci95_low_m2": max(0.0, low),
                        "ci95_high_m2": min(config.sample_area_m2, high),
                    }
                )
    spectrum_path = output_dir / "cross_section_spectrum.csv"
    if spectra:
        with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(spectra[0]))
            writer.writeheader()
            writer.writerows(spectra)

    loading = {}
    speeds = np.asarray(config.speeds_m_per_s)
    for category in ("origin", "off_origin", "total_trapped"):
        rates = []
        for disc in complete_discs:
            sigma = np.asarray([disc_sigma[(disc, speed, category)] for speed in speeds])
            integrand = sigma * speeds**3 * np.exp(-(speeds**2) / THERMAL_SCALE_M2_PER_S2)
            integral = float(np.trapezoid(np.r_[0.0, integrand], np.r_[0.0, speeds]))
            rates.append(LOADING_RATE_PREFACTOR * integral)
        if rates:
            mean, low, high = _mean_ci(rates)
            loading[category] = {
                "mean_rate_atoms_per_s": mean,
                "ci95_low_atoms_per_s": max(0.0, low),
                "ci95_high_atoms_per_s": high,
                "independent_direction_discs": len(rates),
                "velocity_range_m_per_s": [0.0, float(speeds[-1])],
                "exact_zero_speed_integrand_anchor": 0.0,
                "interpretation": "truncated preliminary aperture-limited loading integral",
            }

    counts = {name: sum(row["outcome"] == name for row in rows) for name in (
        OUTCOME_ORIGIN, OUTCOME_OFF_ORIGIN, OUTCOME_ESCAPE, OUTCOME_INDETERMINATE
    )}
    counts_by_speed = []
    for speed in config.speeds_m_per_s:
        subset = [row for row in rows if row["speed_m_per_s"] == speed]
        for outcome in (OUTCOME_ORIGIN, OUTCOME_OFF_ORIGIN, OUTCOME_ESCAPE, OUTCOME_INDETERMINATE):
            counts_by_speed.append(
                {
                    "speed_m_per_s": speed,
                    "outcome": outcome,
                    "count": sum(row["outcome"] == outcome for row in subset),
                    "fraction": (
                        sum(row["outcome"] == outcome for row in subset) / len(subset)
                        if subset else float("nan")
                    ),
                }
            )
    with (output_dir / "outcome_counts_by_speed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts_by_speed[0]))
        writer.writeheader(); writer.writerows(counts_by_speed)
    summary = {
        "requested_trajectories": config.direction_discs * config.points_per_disc * len(config.speeds_m_per_s),
        "completed_trajectories": len(rows),
        "complete_direction_discs": len(complete_discs),
        "outcome_counts": counts,
        "loading": loading,
        "maximum_sample_cross_section_m2": config.sample_area_m2,
        "statistics_note": "direction discs are the independent clusters; intervals are 95% Student-t",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(rows, spectra, config, output_dir)
    write_report(summary, config, output_dir)
    return summary


def make_plots(rows: list[dict], spectra: list[dict], config: CampaignConfig, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = {
        OUTCOME_ORIGIN: "#1b9e77", OUTCOME_OFF_ORIGIN: "#7570b3",
        OUTCOME_ESCAPE: "#d95f02", OUTCOME_INDETERMINATE: "#777777",
    }
    labels = {
        OUTCOME_ORIGIN: "origin trapped", OUTCOME_OFF_ORIGIN: "off-origin trapped",
        OUTCOME_ESCAPE: "wall escape", OUTCOME_INDETERMINATE: "indeterminate",
    }
    speeds = np.asarray(config.speeds_m_per_s)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(speeds))
    for outcome in colors:
        fractions = [
            np.mean([row["outcome"] == outcome for row in rows if row["speed_m_per_s"] == speed])
            for speed in speeds
        ]
        ax.bar(speeds, fractions, bottom=bottom, width=1.5, color=colors[outcome], label=labels[outcome])
        bottom += fractions
    ax.set(xlabel="incident speed (m/s)", ylabel="trajectory fraction", ylim=(0, 1), title="Full-sphere direct-mask outcomes")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(figure_dir / "outcome_fractions_vs_speed.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for category, color, label in (("origin", colors[OUTCOME_ORIGIN], "origin"), ("off_origin", colors[OUTCOME_OFF_ORIGIN], "off-origin"), ("total_trapped", "#1f78b4", "total trapped")):
        data = [item for item in spectra if item["category"] == category]
        if not data: continue
        x = np.asarray([item["speed_m_per_s"] for item in data])
        y = 1e6 * np.asarray([item["mean_cross_section_m2"] for item in data])
        lo = 1e6 * np.asarray([item["ci95_low_m2"] for item in data])
        hi = 1e6 * np.asarray([item["ci95_high_m2"] for item in data])
        ax.plot(x, y, "o-", color=color, label=label); ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set(xlabel="incident speed (m/s)", ylabel=r"capture cross section (mm$^2$)", title="Aperture-limited capture cross sections (95% cluster t intervals)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "capture_cross_sections.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for category, color, label in (("origin", colors[OUTCOME_ORIGIN], "origin"), ("off_origin", colors[OUTCOME_OFF_ORIGIN], "off-origin"), ("total_trapped", "#1f78b4", "total")):
        data = [item for item in spectra if item["category"] == category]
        if not data: continue
        x = np.asarray([item["speed_m_per_s"] for item in data])
        sigma = np.asarray([item["mean_cross_section_m2"] for item in data])
        g = sigma * x**3 * np.exp(-x**2 / THERMAL_SCALE_M2_PER_S2)
        ax.plot(np.r_[0.0, x], np.r_[0.0, g], "o-", color=color, label=label)
    ax.set(xlabel="speed (m/s)", ylabel=r"$\sigma(v)v^3 e^{-v^2/u^2}$ (m$^5$/s$^3$)", title="Preliminary loading integrand (exact g(0)=0 anchor)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "loading_integrand.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome in (OUTCOME_ORIGIN, OUTCOME_OFF_ORIGIN):
        values = 1e3 * np.asarray([row["final_centroid_radius_m"] for row in rows if row["outcome"] == outcome])
        if len(values): ax.hist(values, bins=20, alpha=0.65, color=colors[outcome], label=labels[outcome])
    ax.axvline(2.0, color="black", ls="--", lw=1, label="origin-core radius")
    ax.set(xlabel="final-window centroid radius (mm)", ylabel="trajectories", title="Locations of classified captures")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "capture_location_histogram.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome in colors:
        subset = [row for row in rows if row["outcome"] == outcome]
        if subset:
            ax.scatter([row["speed_m_per_s"] for row in subset], 1e3*np.asarray([row["impact_parameter_m"] for row in subset]), s=12, alpha=0.55, color=colors[outcome], label=labels[outcome])
    ax.set(xlabel="incident speed (m/s)", ylabel="impact parameter (mm)", title="Direct outcome mask")
    ax.legend(frameon=False, markerscale=1.5); fig.tight_layout(); fig.savefig(figure_dir / "impact_parameter_outcomes.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for outcome in colors:
        subset = [row for row in rows if row["outcome"] == outcome]
        if not subset: continue
        axes[0].hist(1e3*np.asarray([row["minimum_radius_m"] for row in subset]), bins=24, histtype="step", lw=1.8, color=colors[outcome], label=labels[outcome])
        axes[1].hist(np.asarray([row["final_speed_m_per_s"] for row in subset]), bins=24, histtype="step", lw=1.8, color=colors[outcome], label=labels[outcome])
    axes[0].set(xlabel="minimum radius reached (mm)", ylabel="trajectories")
    axes[1].set(xlabel="final speed (m/s)", ylabel="trajectories")
    axes[1].legend(frameon=False)
    fig.suptitle("Trajectory outcome histograms"); fig.tight_layout(); fig.savefig(figure_dir / "trajectory_outcome_histograms.png", dpi=180); plt.close(fig)

    off = [row for row in rows if row["outcome"] == OUTCOME_OFF_ORIGIN]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    if off:
        centroids = 1e3 * np.asarray([row["final_centroid_m"] for row in off])
        scatter = ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c=[row["speed_m_per_s"] for row in off], cmap="viridis", s=24)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title="Off-origin capture centroids")
        fig.colorbar(scatter, ax=ax, label="incident speed (m/s)")
    else:
        ax.text2D(0.5, 0.5, "No trajectory passed the conservative\noff-origin capture criteria", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title="Off-origin capture centroids")
    fig.tight_layout(); fig.savefig(figure_dir / "off_origin_centroids_3d.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for outcome in colors:
        candidates = [row for row in rows if row["outcome"] == outcome]
        if not candidates: continue
        row = candidates[0]
        history = np.load(output_dir / "histories" / row["history_file"])
        time_ms = 1e3 * history["times_s"]
        axes[0].plot(time_ms, 1e3*np.linalg.norm(history["positions_m"], axis=1), color=colors[outcome], label=labels[outcome])
        axes[1].plot(time_ms, np.linalg.norm(history["velocities_m_per_s"], axis=1), color=colors[outcome], label=labels[outcome])
    axes[0].axhline(2.0, color="black", ls="--", lw=1); axes[0].set(xlabel="time (ms)", ylabel="radius (mm)")
    axes[1].set(xlabel="time (ms)", ylabel="speed (m/s)"); axes[1].legend(frameon=False)
    fig.suptitle("Representative trajectory histories"); fig.tight_layout(); fig.savefig(figure_dir / "representative_trajectories.png", dpi=180); plt.close(fig)


def write_report(summary: dict, config: CampaignConfig, output_dir: Path) -> None:
    counts = summary["outcome_counts"]
    loading_lines = []
    for name, item in summary["loading"].items():
        loading_lines.append(
            f"- {name}: {item['mean_rate_atoms_per_s']:.6g} atoms/s "
            f"(95% cluster-t CI {item['ci95_low_atoms_per_s']:.6g} to {item['ci95_high_atoms_per_s']:.6g})"
        )
    text = f"""# Preliminary surrogate-field capture and loading survey

## Scope

This time-bounded diagnostic uses the rebuilt Section-12 24-state population-rate
solver, deterministic mean force, gravity, and the surrogate effective magnetic
field. It is not a validated physical pMOT prediction: the surrogate field is not
a state-resolved AC-Stark Hamiltonian and trap-light conservative forces,
scattering/heating, coherent interference, and nonadiabatic zero-field physics are
absent.

## Sampling and classification

- {config.direction_discs} full-sphere direction discs, sampled uniformly in solid angle.
- {config.points_per_disc} independent uniform-area points per disc; all velocities on a disc are parallel.
- Launch radius {1e3*config.launch_radius_m:.3f} mm; disc radius {1e3*config.disc_radius_m:.3f} mm; sampled area {1e6*config.sample_area_m2:.6f} mm^2.
- Direct speed mask: {', '.join(f'{value:g}' for value in config.speeds_m_per_s)} m/s; no scalar capture-boundary assumption.
- Origin capture uses the established 2 mm core rule: 5 ms continuous residence or two entries with an intervening exit.
- Off-origin capture is conservative: no origin capture or wall loss, plus a final 5 ms centroid beyond 2 mm, maximum excursion <= {1e3*config.off_origin_max_excursion_m:.3f} mm, RMS speed <= {config.off_origin_max_rms_speed_m_per_s:.3f} m/s, and net displacement <= {1e3*config.off_origin_max_net_displacement_m:.3f} mm.
- Anything retained but not meeting either capture definition remains indeterminate; it is not counted as trapped.
- Direction discs, not individual rays, are the independent clusters for 95% Student-t intervals.

## Completion and outcomes

- Completed {summary['completed_trajectories']} of {summary['requested_trajectories']} requested trajectories.
- Complete direction clusters: {summary['complete_direction_discs']} of {config.direction_discs}.
- Origin trapped: {counts[OUTCOME_ORIGIN]}.
- Off-origin trapped: {counts[OUTCOME_OFF_ORIGIN]}.
- Wall escaped: {counts[OUTCOME_ESCAPE]}.
- Indeterminate timeout: {counts[OUTCOME_INDETERMINATE]}.

## Preliminary loading integral (0--{config.speeds_m_per_s[-1]:g} m/s only)

{chr(10).join(loading_lines) if loading_lines else '- Not calculated: no complete direction clusters.'}

These are truncated, aperture-limited diagnostic integrals. The exact weighted
integrand anchor g(0)=0 is used; no zero-speed cross section is imputed. If the
cross section remains nonzero at the upper speed, the omitted high-speed tail is
unresolved.

## Files

- `trajectory_results.csv`: one row per completed trajectory.
- `cross_section_spectrum.csv`: clustered category cross sections and intervals.
- `outcome_counts_by_speed.csv`: direct-mask counts and fractions at each speed.
- `summary.json`: machine-readable counts and loading estimates.
- `figures/`: outcome, cross-section, loading-integrand, location, and representative-trajectory plots.
- `histories/`: downsampled trajectory histories for audit and plotting.
- `trajectory_results_initial_criteria.csv`: preserved first-pass labels before
  visual QA tightened the off-origin speed and displacement gates.
"""
    audit_path = output_dir / "representative_audits" / "audit_summary.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        half_step_duration_ms = 1.0e3 * float(audit["half_step_duration_s"])
        longer_duration_ms = 1.0e3 * float(audit["longer_duration_s"])
        text += f"""

## Representative numerical audit

One median-speed example from each populated outcome class was repeated at half
timestep for {half_step_duration_ms:g} ms and at the production timestep for a
longer {longer_duration_ms:g} ms duration. {audit['classification_agreements']} of
{audit['audit_runs']} repeated classifications agreed. See
`representative_audits/audit_results.csv` and `audit_summary.json`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_campaign(output_dir: Path, config: CampaignConfig | None = None) -> dict:
    config = config or CampaignConfig()
    from .surrogate_effective_field import DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "histories").mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    # Normalize tuples to their JSON list representation before comparing a
    # resumed run with the on-disk manifest.
    current_manifest = json.loads(json.dumps(asdict(config)))
    current_manifest["sample_area_m2"] = config.sample_area_m2
    current_manifest["scientific_status"] = "preliminary surrogate-field diagnostic"
    current_manifest["effective_field_configuration"] = (
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata()
    )
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: prior.get(key) for key in current_manifest}
        if comparable != current_manifest:
            prior_speeds = comparable.pop("speeds_m_per_s", None)
            current_speeds = current_manifest["speeds_m_per_s"]
            current_without_speeds = dict(current_manifest)
            current_without_speeds.pop("speeds_m_per_s")
            speed_extension = (
                comparable == current_without_speeds
                and prior_speeds is not None
                and set(prior_speeds).issubset(current_speeds)
            )
            if not speed_extension:
                raise RuntimeError("existing output manifest does not match this campaign")
            manifest_path.write_text(
                json.dumps(current_manifest, indent=2),
                encoding="utf-8",
            )
    else:
        manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")

    results_path = output_dir / "trajectory_results.csv"
    prior_rows = _read_rows(results_path)
    completed = {row["task_key"] for row in prior_rows}
    tasks = [task for task in build_launch_tasks(config) if task.key not in completed]
    geometry_digest = hashlib.sha256(
        json.dumps([asdict(task) for task in build_launch_tasks(config)], sort_keys=True).encode()
    ).hexdigest()
    (output_dir / "geometry_sha256.txt").write_text(geometry_digest + "\n", encoding="utf-8")

    started = time.perf_counter()
    print(f"Resuming with {len(completed)} complete; {len(tasks)} trajectories remain.", flush=True)
    submitted = 0
    finished_now = 0
    iterator = iter(tasks)
    pending = set()
    # ARC's immutable transition data are expensive to construct and its local
    # SQLite cache is not safe for several fresh processes to initialize at
    # exactly the same instant.  On the Linux production environment, build
    # once before ProcessPoolExecutor forks so every worker inherits the same
    # read-only model and beam configuration.
    if tasks:
        _worker_context()
    executor = ProcessPoolExecutor(max_workers=config.workers)
    try:
        while len(pending) < config.workers:
            try: task = next(iterator)
            except StopIteration: break
            pending.add(executor.submit(_run_task, task, config, str(output_dir / "histories")))
            submitted += 1
        last_report = time.perf_counter()
        while pending:
            elapsed = time.perf_counter() - started
            done, pending = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                _append_row(results_path, row)
                finished_now += 1
                if time.perf_counter() - started < config.submission_budget_s:
                    try: task = next(iterator)
                    except StopIteration: task = None
                    if task is not None:
                        pending.add(executor.submit(_run_task, task, config, str(output_dir / "histories")))
                        submitted += 1
            if time.perf_counter() - last_report >= 45.0 or not pending:
                total = len(completed) + finished_now
                elapsed_now = time.perf_counter() - started
                if finished_now:
                    throughput_per_s = finished_now / elapsed_now
                    remaining_now = max(0, len(tasks) - finished_now)
                    eta_s = remaining_now / throughput_per_s
                    eta_text = f"; rolling ETA {eta_s/3600:.2f} h"
                else:
                    eta_text = "; rolling ETA pending"
                print(
                    f"completed {total}/{len(completed)+len(tasks)}; "
                    f"elapsed {elapsed_now/60:.1f} min{eta_text}",
                    flush=True,
                )
                last_report = time.perf_counter()
            if elapsed >= config.submission_budget_s and not done:
                print("Submission budget reached; draining only already-running trajectories.", flush=True)
        executor.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    elapsed = time.perf_counter() - started
    if elapsed > config.absolute_wall_budget_s:
        raise RuntimeError(f"campaign exceeded its {config.absolute_wall_budget_s/3600:.2f} h absolute budget")
    rows = _read_rows(results_path)
    summary = aggregate_results(rows, config, output_dir)
    summary["runtime_this_invocation_s"] = elapsed
    summary["submitted_this_invocation"] = submitted
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Campaign complete in {elapsed/60:.1f} min: {summary['outcome_counts']}", flush=True)
    return summary


def run_representative_audits(
    output_dir: Path,
    config: CampaignConfig | None = None,
) -> dict:
    """Repeat one example per outcome at half-step and longer duration."""

    config = config or CampaignConfig()
    rows = _numeric_rows(_read_rows(output_dir / "trajectory_results.csv"))
    if not rows:
        raise RuntimeError("the primary campaign must exist before auditing")
    selected = []
    for outcome in (OUTCOME_ORIGIN, OUTCOME_OFF_ORIGIN, OUTCOME_ESCAPE, OUTCOME_INDETERMINATE):
        candidates = [row for row in rows if row["outcome"] == outcome]
        if candidates:
            # Prefer a median-speed example rather than an extreme edge case.
            candidates.sort(key=lambda row: (abs(row["speed_m_per_s"] - 7.0), row["task_key"]))
            selected.append(candidates[0])

    audit_root = output_dir / "representative_audits"
    audit_root.mkdir(exist_ok=True)
    longer_duration_s = 1.5 * config.duration_s
    duration_ms = 1.0e3 * config.duration_s
    longer_duration_ms = 1.0e3 * longer_duration_s
    cases = (
        (
            f"half_step_{duration_ms:g}ms",
            replace(config, time_step_s=0.5 * config.time_step_s),
        ),
        (
            f"longer_{longer_duration_ms:g}ms",
            replace(config, duration_s=longer_duration_s),
        ),
    )
    jobs = []
    for baseline in selected:
        task = LaunchTask(
            disc_index=baseline["disc_index"],
            point_index=baseline["point_index"],
            speed_m_per_s=baseline["speed_m_per_s"],
            impact_parameter_m=baseline["impact_parameter_m"],
            theta_rad=baseline["theta_rad"],
            phi_rad=baseline["phi_rad"],
            initial_position_m=(baseline["initial_x_m"], baseline["initial_y_m"], baseline["initial_z_m"]),
            incident_unit_vector=(baseline["incident_x"], baseline["incident_y"], baseline["incident_z"]),
        )
        for case_name, audit_config in cases:
            history_dir = audit_root / case_name
            history_dir.mkdir(exist_ok=True)
            jobs.append((baseline, task, case_name, audit_config, history_dir))

    audit_rows = []
    with ProcessPoolExecutor(max_workers=min(config.workers, len(jobs))) as executor:
        future_jobs = {
            executor.submit(_run_task, task, audit_config, str(history_dir)): (baseline, case_name, audit_config)
            for baseline, task, case_name, audit_config, history_dir in jobs
        }
        for future, metadata in future_jobs.items():
            baseline, case_name, audit_config = metadata
            row = future.result()
            row.update(
                audit_case=case_name,
                baseline_outcome=baseline["outcome"],
                classification_agrees=row["outcome"] == baseline["outcome"],
                audit_time_step_s=audit_config.time_step_s,
                audit_duration_s=audit_config.duration_s,
            )
            audit_rows.append(row)

    columns = [
        "audit_case", "task_key", "baseline_outcome", "outcome",
        "classification_agrees", "audit_time_step_s", "audit_duration_s",
        "termination_reason", "simulated_duration_s", "runtime_s", "core_entries",
        "max_continuous_core_residence_s", "minimum_radius_m", "final_radius_m",
        "final_speed_m_per_s", "final_centroid_m", "final_centroid_radius_m",
        "final_window_max_excursion_m", "final_window_rms_speed_m_per_s",
        "history_file",
    ]
    with (audit_root / "audit_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(audit_rows)
    report = {
        "audited_primary_trajectories": len(selected),
        "audit_runs": len(audit_rows),
        "classification_agreements": sum(bool(row["classification_agrees"]) for row in audit_rows),
        "half_step_duration_s": config.duration_s,
        "longer_duration_s": longer_duration_s,
        "rows": [
            {key: row[key] for key in (
                "audit_case", "task_key", "baseline_outcome", "outcome",
                "classification_agrees", "audit_time_step_s", "audit_duration_s",
            )}
            for row in audit_rows
        ],
    }
    (audit_root / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        write_report(
            json.loads(summary_path.read_text(encoding="utf-8")),
            config,
            output_dir,
        )
    return report


def reclassify_saved_results(
    output_dir: Path,
    config: CampaignConfig | None = None,
) -> dict:
    """Reapply classification rules to saved histories without rerunning dynamics."""

    config = config or CampaignConfig()
    from .surrogate_effective_field import DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    results_path = output_dir / "trajectory_results.csv"
    raw_rows = _read_rows(results_path)
    if not raw_rows:
        raise RuntimeError("no trajectory results are available to reclassify")
    backup_path = output_dir / "trajectory_results_initial_criteria.csv"
    if not backup_path.exists():
        shutil.copy2(results_path, backup_path)
    revised = []
    for raw in raw_rows:
        row = dict(raw)
        history = np.load(output_dir / "histories" / row["history_file"])
        result = classify_trajectory(
            np.asarray(history["times_s"]),
            np.asarray(history["positions_m"]),
            np.asarray(history["velocities_m_per_s"]),
            row["termination_reason"],
            config,
        )
        row.update(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in result.items()
            }
        )
        revised.append(row)
    temporary = results_path.with_suffix(".reclassified.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(revised)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(results_path)

    manifest = json.loads(json.dumps(asdict(config)))
    manifest["sample_area_m2"] = config.sample_area_m2
    manifest["scientific_status"] = "preliminary surrogate-field diagnostic"
    manifest["effective_field_configuration"] = (
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata()
    )
    manifest["classification_revision"] = (
        "conservative off-origin rule added final-window net-displacement and "
        "0.10 m/s RMS-speed gates after visual trajectory QA"
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = aggregate_results(revised, config, output_dir)
    summary["classification_revision_preserved_file"] = backup_path.name
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "CampaignConfig", "LaunchTask", "OUTCOME_ESCAPE", "OUTCOME_INDETERMINATE",
    "OUTCOME_OFF_ORIGIN", "OUTCOME_ORIGIN", "aggregate_results", "build_launch_tasks",
    "classify_trajectory", "reclassify_saved_results", "run_campaign",
    "run_representative_audits",
]
