"""Plots, persistence, tabular output, and animations for Section-12 trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ..beam_plotting import beam_surface_mesh_mm
from .rate_equations import RateEquationModel, RateEquationTrajectoryRecord


def trajectory_summary(
    record: RateEquationTrajectoryRecord,
    *,
    core_radius_m: float = 2.0e-3,
    final_window_s: float = 1.0e-3,
    minimum_duration_s: float = 5.0e-3,
) -> dict[str, object]:
    """Return geometric and kinematic trajectory diagnostics."""

    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    velocities = np.asarray(record.velocities_m_per_s, dtype=float)
    if len(times) == 0:
        raise ValueError("trajectory record is empty")
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
        "step_count": max(0, len(times) - 1),
        "termination_reason": record.termination_reason,
        "core_entry_count": entries,
        "two_core_entry_candidate": entries >= 2,
        "bounded_trapping_candidate": bounded,
        "minimum_radius_m": float(np.min(radii)),
        "final_radius_m": float(radii[-1]),
        "initial_speed_m_per_s": float(np.linalg.norm(velocities[0])),
        "final_speed_m_per_s": float(np.linalg.norm(velocities[-1])),
        "final_window_s": min(final_window_s, elapsed),
        "final_window_maximum_radius_m": float(np.max(final_radii)),
        "maximum_optical_force_n": float(
            np.max(np.linalg.norm(np.asarray(record.forces_n, dtype=float), axis=1))
        ),
        "mean_spontaneous_scattering_rate_per_s": float(
            np.mean(record.total_spontaneous_scattering_rates_per_s)
        ),
    }


def trajectory_table(record: RateEquationTrajectoryRecord):
    """Return a pandas table with the primary trajectory observables."""

    import pandas as pd

    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    velocities = np.asarray(record.velocities_m_per_s, dtype=float)
    forces = np.asarray(record.forces_n, dtype=float)
    fields = np.asarray(record.magnetic_fields_t, dtype=float)
    frame = pd.DataFrame(
        {
            "time_s": times,
            "x_m": positions[:, 0],
            "y_m": positions[:, 1],
            "z_m": positions[:, 2],
            "vx_m_per_s": velocities[:, 0],
            "vy_m_per_s": velocities[:, 1],
            "vz_m_per_s": velocities[:, 2],
            "Fx_N": forces[:, 0],
            "Fy_N": forces[:, 1],
            "Fz_N": forces[:, 2],
            "Bx_T": fields[:, 0],
            "By_T": fields[:, 1],
            "Bz_T": fields[:, 2],
            "spontaneous_scattering_rate_per_s": (
                record.total_spontaneous_scattering_rates_per_s
            ),
        }
    )
    for index in range(np.asarray(record.populations).shape[1]):
        frame[f"population_{index}"] = np.asarray(record.populations)[:, index]
    return frame


def save_trajectory(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    beams,
    output_stem: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> list[Path]:
    """Save trajectory arrays, a flat CSV, and self-describing metadata."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    state_labels = [
        f"{state.manifold}:F={state.f},mF={state.m_f:+d}"
        for state in model.structure.states
    ]
    arrays = {
        "time_s": np.asarray(record.times_s),
        "position_m": np.asarray(record.positions_m),
        "velocity_m_per_s": np.asarray(record.velocities_m_per_s),
        "force_n": np.asarray(record.forces_n),
        "beam_effective_scattering_rate_per_s": np.asarray(
            record.beam_effective_scattering_rates_per_s
        ),
        "total_spontaneous_scattering_rate_per_s": np.asarray(
            record.total_spontaneous_scattering_rates_per_s
        ),
        "magnetic_field_t": np.asarray(record.magnetic_fields_t),
        "quantization_axis": np.asarray(record.quantization_axes),
        "populations": np.asarray(record.populations),
        "state_labels": np.asarray(state_labels),
        "beam_labels": np.asarray([beam.label for beam in beams]),
    }
    npz_path = output_stem.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    csv_path = output_stem.with_suffix(".csv")
    trajectory_table(record).to_csv(csv_path, index=False)
    metadata_path = output_stem.with_name(f"{output_stem.name}_metadata.json")
    payload = {
        "schema": "pmot.mot_multilevel.population-rate-trajectory.v1",
        "model": "Section-12 24-state adiabatic population-rate equation",
        "force": "beam-resolved net absorption minus stimulated emission",
        "diffusion": "not implemented; deterministic mean force",
        "state_labels": state_labels,
        "beam_labels": [beam.label for beam in beams],
        "summary": trajectory_summary(record),
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return [npz_path, csv_path, metadata_path]


def draw_mot_beam_volumes(axis, beams, *, length_m: float = 60.0e-3) -> None:
    """Draw one translucent cylindrical envelope for each Cartesian path."""

    colors = {
        "horizontal_x": "#f9a8d4",
        "horizontal_y": "#93c5fd",
        "vertical_z": "#86efac",
    }
    drawn: set[str] = set()
    for beam in beams:
        if beam.axis_name in drawn:
            continue
        drawn.add(beam.axis_name)
        x, y, z = beam_surface_mesh_mm(
            beam.direction,
            beam.beam_radius_m,
            length_m,
            axial_samples=14,
            angular_samples=24,
        )
        axis.plot_surface(
            x,
            y,
            z,
            color=colors.get(beam.axis_name, "#cbd5e1"),
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.18,
        )


def plot_trajectory_3d(
    record: RateEquationTrajectoryRecord,
    beams,
    path: Path | None = None,
    *,
    title: str = "Physical multilevel MOT trajectory",
    show_beams: bool = True,
    spatial_extent_mm: float | None = None,
):
    """Plot the atom path and the three shared cooling/repump beam volumes."""

    positions_mm = 1.0e3 * np.asarray(record.positions_m, dtype=float)
    times_ms = 1.0e3 * np.asarray(record.times_s, dtype=float)
    figure = plt.figure(figsize=(9.5, 7.7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    if show_beams:
        draw_mot_beam_volumes(axis, beams)
    stride = max(1, len(times_ms) // 1500)
    indices = np.arange(0, len(times_ms), stride)
    if indices[-1] != len(times_ms) - 1:
        indices = np.append(indices, len(times_ms) - 1)
    scatter = axis.scatter(
        positions_mm[indices, 0],
        positions_mm[indices, 1],
        positions_mm[indices, 2],
        c=times_ms[indices],
        cmap="viridis",
        s=7,
    )
    axis.plot(*positions_mm.T, color="#0f766e", linewidth=1.2, alpha=0.72)
    axis.scatter(*positions_mm[0], color="#b91c1c", s=50, label="start")
    axis.scatter(*positions_mm[-1], color="#111827", s=50, label="end")
    axis.scatter(0.0, 0.0, 0.0, color="#7c3aed", s=38, label="trap center")
    extent = spatial_extent_mm
    if extent is None:
        extent = min(32.0, max(8.0, 1.15 * float(np.max(np.abs(positions_mm)))))
    axis.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        zlim=(-extent, extent),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title=(
            f"{title}\ntermination={record.termination_reason}; "
            f"elapsed={times_ms[-1]:.3f} ms"
        ),
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, shrink=0.65, pad=0.08, label="Time [ms]")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def manifold_populations(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
) -> dict[str, np.ndarray]:
    populations = np.asarray(record.populations, dtype=float)
    local_states = [
        model.structure.states[index]
        for index in np.concatenate((model.ground_indices, model.excited_indices))
    ]
    output: dict[str, np.ndarray] = {}
    for manifold in ("ground", "excited"):
        for f_value in sorted(
            {state.f for state in local_states if state.manifold == manifold}
        ):
            indices = [
                index
                for index, state in enumerate(local_states)
                if state.manifold == manifold and state.f == f_value
            ]
            output[f"{manifold} F={f_value}"] = np.sum(populations[:, indices], axis=1)
    return output


def plot_time_diagnostics(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    beams,
    path: Path | None = None,
):
    """Plot position, velocity, net beam rates, force, and manifolds."""

    time_ms = 1.0e3 * np.asarray(record.times_s, dtype=float)
    position_mm = 1.0e3 * np.asarray(record.positions_m, dtype=float)
    velocity = np.asarray(record.velocities_m_per_s, dtype=float)
    forces_zn = 1.0e21 * np.asarray(record.forces_n, dtype=float)
    beam_rates = np.asarray(record.beam_effective_scattering_rates_per_s, dtype=float)
    manifolds = manifold_populations(record, model)
    figure, panels = plt.subplots(3, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    for component, (label, color) in enumerate(zip("xyz", colors)):
        panels[0, 0].plot(time_ms, position_mm[:, component], color=color, label=label)
        panels[0, 1].plot(time_ms, velocity[:, component], color=color, label=f"v{label}")
        panels[1, 1].plot(time_ms, forces_zn[:, component], color=color, label=f"F{label}")
    for index, beam in enumerate(beams):
        panels[1, 0].plot(time_ms, beam_rates[:, index], linewidth=0.9, label=beam.label)
    panels[2, 0].plot(
        time_ms,
        record.total_spontaneous_scattering_rates_per_s,
        color="#111827",
    )
    for label, values in manifolds.items():
        panels[2, 1].plot(time_ms, values, label=label)
    panels[0, 0].set(title="Position", ylabel="Position [mm]")
    panels[0, 1].set(title="Velocity", ylabel="Velocity [m/s]")
    panels[1, 0].set(title="Net stimulated rate by beam", ylabel="Rate [s$^{-1}$]")
    panels[1, 1].set(title="Optical force", ylabel="Force [zN]")
    panels[2, 0].set(title="Total spontaneous scattering", ylabel="Rate [s$^{-1}$]")
    panels[2, 1].set(title="Hyperfine-manifold populations", ylabel="Population")
    for panel in panels.flat:
        panel.set_xlabel("Time [ms]")
        panel.grid(alpha=0.22)
    for panel in (panels[0, 0], panels[0, 1], panels[1, 0], panels[1, 1], panels[2, 1]):
        panel.legend(fontsize=7)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def create_trajectory_animation(
    record: RateEquationTrajectoryRecord,
    beams,
    path: Path | None = None,
    *,
    max_frames: int = 250,
    fps: int = 20,
    spatial_extent_mm: float | None = None,
):
    """Create an inspectable 3D trajectory animation and optionally save a GIF."""

    if max_frames < 2 or fps <= 0:
        raise ValueError("max_frames must be at least two and fps must be positive")
    positions = 1.0e3 * np.asarray(record.positions_m, dtype=float)
    times = 1.0e3 * np.asarray(record.times_s, dtype=float)
    frame_indices = np.unique(
        np.linspace(0, len(times) - 1, min(max_frames, len(times)), dtype=int)
    )
    extent = spatial_extent_mm
    if extent is None:
        extent = min(32.0, max(8.0, 1.15 * float(np.max(np.abs(positions)))))
    figure = plt.figure(figsize=(8.2, 7.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    draw_mot_beam_volumes(axis, beams)
    axis.plot(*positions.T, color="#94a3b8", linewidth=1.0, alpha=0.65)
    trace, = axis.plot([], [], [], color="#0f766e", linewidth=2.0)
    marker = axis.scatter([], [], [], color="#b91c1c", s=55)
    time_label = axis.text2D(0.03, 0.95, "", transform=axis.transAxes)
    axis.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        zlim=(-extent, extent),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title="Physical multilevel MOT trajectory",
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))

    def update(frame_number):
        index = int(frame_indices[frame_number])
        trace.set_data(positions[: index + 1, 0], positions[: index + 1, 1])
        trace.set_3d_properties(positions[: index + 1, 2])
        marker._offsets3d = (
            [positions[index, 0]],
            [positions[index, 1]],
            [positions[index, 2]],
        )
        time_label.set_text(f"t = {times[index]:.3f} ms")
        return trace, marker, time_label

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        repeat=True,
        blit=False,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    return movie


def create_population_histogram_animation(
    record: RateEquationTrajectoryRecord,
    model: RateEquationModel,
    path: Path | None = None,
    *,
    max_frames: int = 250,
    fps: int = 20,
):
    """Animate all 24 normalized state populations in atomic-basis order."""

    if max_frames < 2 or fps <= 0:
        raise ValueError("max_frames must be at least two and fps must be positive")
    populations = np.asarray(record.populations, dtype=float)
    times_ms = 1.0e3 * np.asarray(record.times_s, dtype=float)
    if populations.ndim != 2 or populations.shape[1] != 24:
        raise ValueError("a 24-state trajectory population record is required")
    if len(times_ms) != len(populations) or len(times_ms) < 2:
        raise ValueError("at least two synchronized trajectory records are required")
    if not np.all(np.isfinite(populations)) or np.any(populations < -1.0e-10):
        raise ValueError("trajectory populations must be finite and non-negative")
    if not np.allclose(np.sum(populations, axis=1), 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("trajectory populations must each sum to one")

    state_indices = np.concatenate((model.ground_indices, model.excited_indices))
    states = [model.structure.states[int(index)] for index in state_indices]
    labels = [
        f"{'g' if state.is_ground else 'e'} F={state.f}, m={state.m_f:+d}"
        for state in states
    ]
    colors = [
        "#2563eb" if state.is_ground and state.f == 1
        else "#0f766e" if state.is_ground
        else "#a855f7" if state.f in (0, 1)
        else "#d97706" if state.f == 2
        else "#dc2626"
        for state in states
    ]
    frame_indices = np.unique(
        np.linspace(0, len(times_ms) - 1, min(max_frames, len(times_ms)), dtype=int)
    )
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    bars = axis.bar(np.arange(24), populations[0], color=colors, width=0.82)
    axis.axvline(7.5, color="#334155", linewidth=1.2, linestyle="--")
    axis.set(
        xlim=(-0.8, 23.8),
        ylim=(0.0, max(0.02, 1.08 * float(np.max(populations)))),
        ylabel="Population probability",
        xlabel="Rb-87 D2 hyperfine-Zeeman state",
        title="24-state population-rate MOT: steady-state populations along trajectory",
        xticks=np.arange(24),
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=70, labelsize=8)
    axis.grid(axis="y", alpha=0.22)
    time_label = axis.text(0.02, 0.95, "", transform=axis.transAxes, va="top")

    def update(frame_number):
        index = int(frame_indices[frame_number])
        for bar, height in zip(bars, populations[index]):
            bar.set_height(float(height))
        time_label.set_text(
            f"t = {times_ms[index]:.3f} ms   "
            f"Σp = {np.sum(populations[index]):.9f}"
        )
        return (*bars, time_label)

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        repeat=True,
        blit=False,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    return movie


__all__ = [
    "create_population_histogram_animation",
    "create_trajectory_animation",
    "draw_mot_beam_volumes",
    "manifold_populations",
    "plot_time_diagnostics",
    "plot_trajectory_3d",
    "save_trajectory",
    "trajectory_summary",
    "trajectory_table",
]
