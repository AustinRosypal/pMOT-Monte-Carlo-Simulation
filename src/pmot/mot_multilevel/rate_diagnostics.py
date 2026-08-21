"""Persistence and plots for efficient rate-equation MOT trajectories."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .rate_equations import (
    RateEquationModel,
    RateEquationTrajectoryRecord,
    rate_equation_observable,
)
from .screening import draw_multilevel_mot_beam_volumes


def summarize_rate_equation_trajectory(
    record: RateEquationTrajectoryRecord,
    *,
    core_radius_m: float = 2.0e-3,
    final_window_s: float = 1.0e-3,
    minimum_duration_s: float = 5.0e-3,
) -> dict[str, object]:
    """Return conservative capture and bounded-trajectory diagnostics."""

    times = np.asarray(record.times_s)
    positions = np.asarray(record.positions_m)
    velocities = np.asarray(record.velocities_m_per_s)
    radii = np.linalg.norm(positions, axis=1)
    inside = radii <= core_radius_m
    entries = int(np.count_nonzero(inside & np.concatenate(([True], ~inside[:-1]))))
    elapsed = float(times[-1] - times[0])
    window_start = times[-1] - min(final_window_s, elapsed)
    final_radii = radii[times >= window_start]
    bounded = bool(
        record.termination_reason == "duration"
        and elapsed >= minimum_duration_s
        and np.max(final_radii) <= core_radius_m
    )
    return {
        "elapsed_s": elapsed,
        "core_entry_count": entries,
        "two_core_entry_candidate": entries >= 2,
        "bounded_trapping_candidate": bounded,
        "minimum_radius_m": float(np.min(radii)),
        "final_radius_m": float(radii[-1]),
        "initial_speed_m_per_s": float(np.linalg.norm(velocities[0])),
        "final_speed_m_per_s": float(np.linalg.norm(velocities[-1])),
        "final_window_s": min(final_window_s, elapsed),
        "final_window_maximum_radius_m": float(np.max(final_radii)),
        "termination_reason": record.termination_reason,
    }


def save_rate_equation_trajectory(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    beams,
    output_stem: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> list[Path]:
    """Save uniform-time trajectory data in compressed and tabular formats."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(record.times_s)
    positions = np.asarray(record.positions_m)
    velocities = np.asarray(record.velocities_m_per_s)
    forces = np.asarray(record.forces_n)
    fields = np.asarray(record.magnetic_fields_t)
    populations = np.asarray(record.populations)
    beam_rates = np.asarray(record.beam_scattering_rates_per_s)
    state_labels = np.asarray([
        f"{state.manifold}:F={state.f},mF={state.m_f:+d}" for state in model.structure.states
    ])
    npz_path = output_stem.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        time_s=times,
        position_m=positions,
        velocity_m_per_s=velocities,
        force_n=forces,
        diffusion_kg2_m2_per_s3=np.asarray(record.diffusion_kg2_m2_per_s3),
        total_scattering_rate_per_s=np.asarray(record.total_scattering_rates_per_s),
        beam_scattering_rate_per_s=beam_rates,
        magnetic_field_t=fields,
        populations=populations,
        state_labels=state_labels,
        beam_labels=np.asarray([beam.label for beam in beams]),
    )
    csv_path = output_stem.with_suffix(".csv")
    population_columns = [f"population_state_{index}" for index in range(model.state_count)]
    beam_columns = [f"beam_{index}_scattering_rate_per_s" for index in range(len(beams))]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s",
            "Fx_N", "Fy_N", "Fz_N", "diffusion_kg2_m2_per_s3", "total_scattering_rate_per_s",
            *beam_columns, "Bx_T", "By_T", "Bz_T", *population_columns,
        ))
        for index in range(len(times)):
            writer.writerow((
                times[index], *positions[index], *velocities[index], *forces[index],
                record.diffusion_kg2_m2_per_s3[index], record.total_scattering_rates_per_s[index],
                *beam_rates[index], *fields[index], *populations[index],
            ))
    metadata_path = output_stem.with_name(f"{output_stem.name}_metadata.json")
    payload = {
        "schema": "pmot.multilevel.rate-equation-trajectory.v1",
        "method": "adiabatic-elimination population rate equations plus Langevin recoil diffusion",
        "state_count": model.state_count,
        "ground_state_count": model.ground_count,
        "excited_state_count": model.excited_count,
        "state_labels": state_labels.tolist(),
        "beam_labels": [beam.label for beam in beams],
        "termination_reason": record.termination_reason,
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return [npz_path, csv_path, metadata_path]


def plot_rate_equation_trajectory_3d(
    record: RateEquationTrajectoryRecord,
    beams,
    path: Path,
    *,
    title: str = "Full multilevel MOT rate-equation trajectory",
    show_beams: bool = True,
    spatial_extent_mm: float | None = None,
) -> Path:
    """Draw the complete 3D atom path inside the MOT beam volumes."""

    positions_mm = 1e3 * np.asarray(record.positions_m)
    time_ms = 1e3 * np.asarray(record.times_s)
    figure = plt.figure(figsize=(10.0, 8.2), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#fbfaf6")
    if show_beams:
        draw_multilevel_mot_beam_volumes(axis, beams)
    indices = np.unique(np.linspace(0, len(time_ms) - 1, min(10_000, len(time_ms)), dtype=int))
    scatter = axis.scatter(
        positions_mm[indices, 0], positions_mm[indices, 1], positions_mm[indices, 2],
        c=time_ms[indices], cmap="viridis", s=5, alpha=.78,
    )
    axis.plot(*positions_mm.T, color="#0f766e", linewidth=1.2, alpha=.72)
    axis.scatter(*positions_mm[0], color="#b91c1c", s=55, label="start")
    axis.scatter(*positions_mm[-1], color="#111827", s=55, label="end")
    axis.scatter(0.0, 0.0, 0.0, color="#7c3aed", s=40, label="trap center")
    if spatial_extent_mm is None:
        minimum_extent = 10.0 if show_beams else 0.5
        extent = max(minimum_extent, 1.15 * float(np.max(np.abs(positions_mm))))
        extent = min(extent, 32.0)
    else:
        extent = spatial_extent_mm
    axis.set(
        xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent),
        xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]",
        title=f"{title}\ntermination={record.termination_reason}; elapsed={time_ms[-1]:.3f} ms; steps={len(time_ms)-1:,}",
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, shrink=.65, pad=.08, label="Time [ms]")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_rate_equation_performance(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    path: Path,
) -> Path:
    """Plot Cartesian motion, speed/radius, scattering, and manifold populations."""

    time_ms = 1e3 * np.asarray(record.times_s)
    position_mm = 1e3 * np.asarray(record.positions_m)
    velocity = np.asarray(record.velocities_m_per_s)
    populations = np.asarray(record.populations)
    radius_mm = np.linalg.norm(position_mm, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    ground_f1 = [i for i, state in enumerate(model.structure.states) if state.is_ground and state.f == 1]
    ground_f2 = [i for i, state in enumerate(model.structure.states) if state.is_ground and state.f == 2]
    excited = list(range(model.ground_count, model.state_count))
    figure, panels = plt.subplots(3, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
    for component, (label, color) in enumerate(zip("xyz", ("#b91c1c", "#1d4ed8", "#15803d"))):
        panels[0, 0].plot(time_ms, position_mm[:, component], color=color, label=label)
        panels[0, 1].plot(time_ms, velocity[:, component], color=color, label=f"v{label}")
    panels[1, 0].plot(time_ms, radius_mm, color="#7c3aed", label="radius")
    panels[1, 0].axhline(2.0, color="#15803d", linestyle="--", label="2 mm core")
    panels[1, 1].plot(time_ms, speed, color="#0f766e")
    panels[2, 0].plot(time_ms, record.total_scattering_rates_per_s, color="#111827")
    panels[2, 1].plot(time_ms, populations[:, ground_f1].sum(axis=1), label="ground F=1")
    panels[2, 1].plot(time_ms, populations[:, ground_f2].sum(axis=1), label="ground F=2")
    panels[2, 1].plot(time_ms, populations[:, excited].sum(axis=1), label="excited")
    panels[0, 0].set(ylabel="Position [mm]", title="Cartesian position")
    panels[0, 1].set(ylabel="Velocity [m/s]", title="Cartesian velocity")
    panels[1, 0].set(ylabel="Radius [mm]", title="Distance from MOT center")
    panels[1, 1].set(ylabel="Speed [m/s]", title="Atomic speed")
    panels[2, 0].set(xlabel="Time [ms]", ylabel=r"Rate [s$^{-1}$]", title="Total steady-state scattering rate")
    panels[2, 1].set(xlabel="Time [ms]", ylabel="Population", title="Manifold populations")
    for panel in panels.flat:
        panel.grid(alpha=.22)
    for panel in (panels[0, 0], panels[0, 1], panels[1, 0], panels[2, 1]):
        panel.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def hyperfine_manifold_occupation_percent(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
) -> tuple[list[str], np.ndarray]:
    """Return time-averaged population percentages grouped by hyperfine level.

    The efficient solver has population probabilities rather than a single
    occupied state.  Consequently these values are the time integrals of the
    manifold populations, divided by the elapsed trajectory time.  All mF
    substates belonging to the same ``|g/e>, F`` manifold are combined.
    """

    times = np.asarray(record.times_s, dtype=float)
    populations = np.asarray(record.populations, dtype=float)
    if populations.ndim != 2 or populations.shape != (len(times), model.state_count):
        raise ValueError("trajectory populations must have shape (time, state_count)")
    if len(times) == 0:
        raise ValueError("trajectory must contain at least one time sample")

    local_states = [
        model.structure.states[index]
        for index in np.concatenate((model.ground_indices, model.excited_indices))
    ]
    groups = [
        (manifold, f)
        for manifold in ("ground", "excited")
        for f in sorted({state.f for state in local_states if state.manifold == manifold})
    ]
    elapsed = float(times[-1] - times[0])
    percentages = []
    for manifold, f in groups:
        indices = [
            index
            for index, state in enumerate(local_states)
            if state.manifold == manifold and state.f == f
        ]
        manifold_population = np.sum(populations[:, indices], axis=1)
        average = (
            float(np.trapezoid(manifold_population, times) / elapsed)
            if elapsed > 0.0
            else float(manifold_population[0])
        )
        percentages.append(100.0 * average)
    labels = [
        rf"$|{'g' if manifold == 'ground' else 'e'}\rangle\ F={f}$"
        for manifold, f in groups
    ]
    return labels, np.asarray(percentages)


def plot_hyperfine_manifold_occupation(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    path: Path | None = None,
):
    """Plot percentage of trajectory time assigned to each hyperfine level."""

    labels, percentages = hyperfine_manifold_occupation_percent(record, model)
    colors = ["#1d4ed8", "#0f766e", "#f59e0b", "#ea580c", "#dc2626", "#7c3aed"]
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    bars = axis.bar(labels, percentages, color=colors[: len(labels)], edgecolor="#111827", linewidth=.7)
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in percentages], padding=3, fontsize=9)
    axis.set(
        title="Time-averaged hyperfine-manifold occupation",
        xlabel="Hyperfine level (summed over all $m_F$ substates)",
        ylabel="Percentage of trajectory time [%]",
        ylim=(0.0, max(5.0, 1.12 * float(np.max(percentages)))),
    )
    axis.grid(axis="y", alpha=.25)
    axis.text(
        .01,
        .98,
        r"Rate-equation interpretation: $100 T^{-1}\int_0^T P_{\mathrm{manifold}}(t)\,dt$",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="#374151",
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def plot_rate_equation_time_diagnostics(
    record: RateEquationTrajectoryRecord,
    beams,
    path: Path | None = None,
):
    """Plot position, velocity, scattering, and optical force versus time."""

    times_ms = 1e3 * np.asarray(record.times_s, dtype=float)
    positions_mm = 1e3 * np.asarray(record.positions_m, dtype=float)
    velocities = np.asarray(record.velocities_m_per_s, dtype=float)
    forces = np.asarray(record.forces_n, dtype=float)
    beam_rates = np.asarray(record.beam_scattering_rates_per_s, dtype=float)
    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    figure, panels = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for component, (label, color) in enumerate(zip("xyz", colors)):
        panels[0, 0].plot(times_ms, positions_mm[:, component], color=color, label=label)
        panels[0, 1].plot(times_ms, velocities[:, component], color=color, label=rf"$v_{label}$")
        panels[1, 1].plot(times_ms, forces[:, component], color=color, label=rf"$F_{label}$")
    for beam_index, beam in enumerate(beams):
        panels[1, 0].plot(
            times_ms,
            beam_rates[:, beam_index],
            linewidth=1.0,
            alpha=.72,
            label=beam.label,
        )
    panels[1, 0].plot(
        times_ms,
        record.total_scattering_rates_per_s,
        color="#111827",
        linewidth=2.0,
        linestyle="--",
        label="total",
    )
    panels[0, 0].set(title="Cartesian position", ylabel="Position [mm]")
    panels[0, 1].set(title="Cartesian velocity", ylabel="Velocity [m/s]")
    panels[1, 0].set(title="Per-beam absorption/scattering rates", ylabel=r"Rate [s$^{-1}$]")
    panels[1, 1].set(
        title="Mean radiation-pressure force (gravity excluded)",
        ylabel="Force [N]",
    )
    for panel in panels.flat:
        panel.set_xlabel("Time [ms]")
        panel.grid(alpha=.25)
        panel.legend(loc="best", fontsize=8)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def plot_rate_equation_force_potential_curves(
    model: RateEquationModel,
    beams,
    coil_config,
    config,
    path: Path | None = None,
    *,
    position_extent_m: float = 8.0e-3,
    velocity_extent_m_per_s: float = 15.0,
    sample_count: int = 81,
):
    """Plot quasi-steady restoring, damping, and axis-integrated potential curves."""

    if sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer of at least 3")
    positions = np.linspace(-position_extent_m, position_extent_m, sample_count)
    velocities = np.linspace(-velocity_extent_m_per_s, velocity_extent_m_per_s, sample_count)
    figure, panels = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for component, label in enumerate("xyz"):
        restoring = []
        damping = []
        for coordinate in positions:
            position = np.zeros(3)
            position[component] = coordinate
            restoring.append(
                rate_equation_observable(
                    model, beams, tuple(position), (0.0, 0.0, 0.0), coil_config, config,
                ).force_n[component]
            )
        for speed in velocities:
            velocity = np.zeros(3)
            velocity[component] = speed
            damping.append(
                rate_equation_observable(
                    model, beams, (0.0, 0.0, 0.0), tuple(velocity), coil_config, config,
                ).force_n[component]
            )
        restoring = np.asarray(restoring)
        potential_j = np.zeros_like(restoring)
        potential_j[1:] = -np.cumsum(
            .5 * (restoring[1:] + restoring[:-1]) * np.diff(positions)
        )
        potential_j -= potential_j[sample_count // 2]
        panels[component, 0].plot(1e3 * positions, restoring, color="#7c3aed")
        panels[component, 1].plot(velocities, damping, color="#0f766e")
        panels[component, 2].plot(
            1e3 * positions,
            1e3 * potential_j / 1.380649e-23,
            color="#c2410c",
        )
        panels[component, 0].set(
            title=rf"Restoring: $F_{label}({label})$ at $v=0$",
            xlabel=f"{label} [mm]",
            ylabel=rf"$F_{label}$ [N]",
        )
        panels[component, 1].set(
            title=rf"Damping: $F_{label}(v_{label})$ at $r=0$",
            xlabel=rf"$v_{label}$ [m/s]",
            ylabel=rf"$F_{label}$ [N]",
        )
        panels[component, 2].set(
            title=rf"Axis potential $U_{label}({label})-U_{label}(0)$",
            xlabel=f"{label} [mm]",
            ylabel=r"Potential [$k_B$ mK]",
        )
        for panel in panels[component]:
            panel.axhline(0.0, color=".4", linewidth=.8)
            panel.axvline(0.0, color=".4", linewidth=.8)
            panel.grid(alpha=.25)
    figure.suptitle(
        "Full multilevel rate-equation MOT force and effective 1D potential curves\n"
        "Quasi-steady internal populations; radiation pressure only; gravity excluded",
        fontsize=14,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


__all__ = [name for name in globals() if not name.startswith("_")]
