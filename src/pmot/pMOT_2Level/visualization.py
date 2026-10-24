"""Notebook-oriented plots and animation for effective two-level pMOT runs."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from ..mot_simple.simulation import SimpleMOTBeam
from ..mot_strange.trapping_beams import TrappingBeam
from .simulation import PMOT2LevelTrajectoryRecord


AXIS_COLORS = ("#d1495b", "#2a9d8f", "#3366cc")
COOLING_COLOR = "#4c78a8"
TRAPPING_INCIDENT_COLOR = "#f28e2b"
TRAPPING_RETRO_COLOR = "#b07aa1"


def trajectory_arrays(record: PMOT2LevelTrajectoryRecord) -> dict[str, np.ndarray]:
    """Return the trajectory histories as consistently shaped arrays."""

    arrays = {
        "time_s": np.asarray(record.times_s, dtype=float),
        "position_m": np.asarray(record.positions_m, dtype=float),
        "velocity_m_per_s": np.asarray(record.velocities_m_per_s, dtype=float),
        "force_n": np.asarray(record.forces_n, dtype=float),
    }
    sample_count = len(arrays["time_s"])
    if sample_count == 0:
        raise ValueError("record must contain at least one sample")
    for name in ("position_m", "velocity_m_per_s", "force_n"):
        if arrays[name].shape != (sample_count, 3):
            raise ValueError(f"record {name} must have shape (n, 3)")
    return arrays


def _transverse_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array((0.0, 0.0, 1.0))
    if abs(float(np.dot(axis, trial))) > 0.9:
        trial = np.array((0.0, 1.0, 0.0))
    first = np.cross(axis, trial)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _surface_coordinates(
    centers_m: np.ndarray,
    radii_m: np.ndarray,
    axis: np.ndarray,
    *,
    azimuth_samples: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = _transverse_basis(axis)
    angle = np.linspace(0.0, 2.0 * np.pi, azimuth_samples)
    ring = (
        np.cos(angle)[:, np.newaxis] * first
        + np.sin(angle)[:, np.newaxis] * second
    )
    points = centers_m[:, np.newaxis, :] + radii_m[:, np.newaxis, np.newaxis] * ring
    return points[..., 0], points[..., 1], points[..., 2]


def add_beam_geometry(
    axis,
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    *,
    extent_m: float,
    cooling_alpha: float = 0.10,
    trapping_alpha: float = 0.24,
) -> None:
    """Add true-scale translucent 1/e^2 beam envelopes to a 3D axis."""

    if extent_m <= 0.0:
        raise ValueError("extent_m must be positive")
    seen_cooling_axes: set[str] = set()
    axial = np.linspace(-extent_m, extent_m, 34)
    for beam in cooling_beams:
        direction = np.asarray(beam.direction, dtype=float)
        if beam.axis_name not in seen_cooling_axes:
            seen_cooling_axes.add(beam.axis_name)
            centers = axial[:, np.newaxis] * direction
            radii = np.full_like(axial, beam.intensity_beam.beam_radius_m)
            x, y, z = _surface_coordinates(centers, radii, direction)
            axis.plot_surface(
                1.0e3 * x,
                1.0e3 * y,
                1.0e3 * z,
                color=COOLING_COLOR,
                alpha=cooling_alpha,
                linewidth=0.0,
                shade=False,
            )
            axis.plot(
                1.0e3 * centers[:, 0],
                1.0e3 * centers[:, 1],
                1.0e3 * centers[:, 2],
                color=COOLING_COLOR,
                alpha=0.55,
                linewidth=0.9,
                linestyle="--",
            )
        arrow_start = -0.82 * extent_m * direction
        arrow_length = 0.24 * extent_m
        axis.quiver(
            *(1.0e3 * arrow_start),
            *(1.0e3 * arrow_length * direction),
            color=COOLING_COLOR,
            alpha=0.9,
            linewidth=1.1,
            arrow_length_ratio=0.28,
        )

    lab_axial = np.linspace(-extent_m, extent_m, 48)
    canonical_directions = {
        "horizontal_x": np.array((1.0, 0.0, 0.0)),
        "horizontal_y": np.array((0.0, 1.0, 0.0)),
        "vertical_z": np.array((0.0, 0.0, 1.0)),
    }
    for beam in trapping_beams:
        lab_direction = canonical_directions[beam.axis_name]
        centers = lab_axial[:, np.newaxis] * lab_direction
        waist_coordinate = float(
            np.dot(np.asarray(beam.waist_position_m, dtype=float), lab_direction)
        )
        radii = beam.waist_radius_m * np.sqrt(
            1.0 + ((lab_axial - waist_coordinate) / beam.rayleigh_range_m) ** 2
        )
        x, y, z = _surface_coordinates(centers, radii, lab_direction)
        color = (
            TRAPPING_INCIDENT_COLOR
            if beam.propagation_sense == "incident"
            else TRAPPING_RETRO_COLOR
        )
        axis.plot_surface(
            1.0e3 * x,
            1.0e3 * y,
            1.0e3 * z,
            color=color,
            alpha=trapping_alpha,
            edgecolor=color,
            linewidth=0.12,
            shade=False,
        )
        axis.plot(
            1.0e3 * centers[:, 0],
            1.0e3 * centers[:, 1],
            1.0e3 * centers[:, 2],
            color=color,
            alpha=0.78,
            linewidth=1.0,
            linestyle=":" if beam.propagation_sense == "retro" else "-.",
        )
        direction = np.asarray(beam.direction, dtype=float)
        arrow_start = -0.72 * extent_m * direction
        arrow_length = 0.22 * extent_m
        axis.quiver(
            *(1.0e3 * arrow_start),
            *(1.0e3 * arrow_length * direction),
            color=color,
            alpha=0.95,
            linewidth=1.2,
            arrow_length_ratio=0.30,
        )


def _format_spatial_axis(axis, extent_mm: float) -> None:
    axis.set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        xlim=(-extent_mm, extent_mm),
        ylim=(-extent_mm, extent_mm),
        zlim=(-extent_mm, extent_mm),
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.scatter((0.0,), (0.0,), (0.0,), color="black", marker="+", s=45)


def _geometry_legend() -> list:
    return [
        Patch(
            color=COOLING_COLOR,
            alpha=0.20,
            label="six 780 nm cooling components",
        ),
        Patch(
            color=TRAPPING_INCIDENT_COLOR,
            alpha=0.30,
            label="1529 nm incident envelope",
        ),
        Patch(
            color=TRAPPING_RETRO_COLOR,
            alpha=0.30,
            label="1529 nm retro envelope",
        ),
        Line2D((0,), (0,), color="black", linewidth=1.7, label="atom trajectory"),
    ]


def plot_trajectory_3d(
    record: PMOT2LevelTrajectoryRecord,
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    *,
    extent_mm: float | None = None,
):
    """Plot a speed-colored 3D trajectory inside both beam geometries."""

    arrays = trajectory_arrays(record)
    positions_mm = 1.0e3 * arrays["position_m"]
    speed = np.linalg.norm(arrays["velocity_m_per_s"], axis=1)
    inferred_extent = max(16.0, 1.08 * float(np.max(np.abs(positions_mm))))
    resolved_extent_mm = inferred_extent if extent_mm is None else float(extent_mm)
    figure = plt.figure(figsize=(10.0, 8.2))
    axis = figure.add_subplot(111, projection="3d")
    add_beam_geometry(
        axis,
        cooling_beams,
        trapping_beams,
        extent_m=1.0e-3 * resolved_extent_mm,
    )
    axis.plot(*positions_mm.T, color="black", linewidth=1.4, alpha=0.75)
    marker_indices = np.unique(
        np.linspace(0, len(positions_mm) - 1, min(180, len(positions_mm))).astype(int)
    )
    markers = axis.scatter(
        *positions_mm[marker_indices].T,
        c=speed[marker_indices],
        cmap="viridis",
        s=14,
        depthshade=False,
    )
    axis.scatter(*positions_mm[0], color="#2a9d8f", marker="o", s=50, label="start")
    axis.scatter(*positions_mm[-1], color="#d1495b", marker="X", s=60, label="end")
    _format_spatial_axis(axis, resolved_extent_mm)
    axis.set_title("Effective two-level pMOT trajectory and optical geometry")
    axis.legend(handles=_geometry_legend(), loc="upper left", fontsize=8)
    colorbar = figure.colorbar(markers, ax=axis, pad=0.08, shrink=0.72)
    colorbar.set_label("speed [m/s]")
    figure.tight_layout()
    return figure, axis


def plot_force_velocity_history(record: PMOT2LevelTrajectoryRecord):
    """Overlay x/y/z radiation forces and velocities on a shared time axis."""

    arrays = trajectory_arrays(record)
    time_ms = 1.0e3 * arrays["time_s"]
    force_z_n = 1.0e21 * arrays["force_n"]
    velocity = arrays["velocity_m_per_s"]
    figure, force_axis = plt.subplots(figsize=(11.5, 5.6))
    velocity_axis = force_axis.twinx()
    labels = ("x", "y", "z")
    handles = []
    for index, (label, color) in enumerate(zip(labels, AXIS_COLORS)):
        force_line, = force_axis.plot(
            time_ms,
            force_z_n[:, index],
            color=color,
            linewidth=1.5,
            label=f"F{label}",
        )
        velocity_line, = velocity_axis.plot(
            time_ms,
            velocity[:, index],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"v{label}",
        )
        handles.extend((force_line, velocity_line))
    force_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    force_axis.set(
        xlabel="time [ms]",
        ylabel="radiation force [zN]",
        title="Radiation force and velocity history",
    )
    velocity_axis.set_ylabel("velocity [m/s]")
    force_axis.grid(alpha=0.22)
    force_axis.legend(handles=handles, ncol=3, frameon=False, loc="best")
    figure.tight_layout()
    return figure, (force_axis, velocity_axis)


def animate_trajectory(
    record: PMOT2LevelTrajectoryRecord,
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    *,
    extent_mm: float | None = None,
    maximum_frames: int = 140,
    interval_ms: int = 55,
) -> FuncAnimation:
    """Animate atom location, its accumulated trail, and instantaneous speed."""

    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least two")
    arrays = trajectory_arrays(record)
    positions_mm = 1.0e3 * arrays["position_m"]
    time_ms = 1.0e3 * arrays["time_s"]
    speed = np.linalg.norm(arrays["velocity_m_per_s"], axis=1)
    frame_indices = np.unique(
        np.linspace(0, len(time_ms) - 1, min(maximum_frames, len(time_ms))).astype(int)
    )
    inferred_extent = max(16.0, 1.08 * float(np.max(np.abs(positions_mm))))
    resolved_extent_mm = inferred_extent if extent_mm is None else float(extent_mm)

    # Keep embedded JS animations comfortably below the notebook transport
    # limit even when the default ~140 display frames are requested.
    figure = plt.figure(figsize=(10.8, 5.2), dpi=85)
    spatial_axis = figure.add_subplot(121, projection="3d")
    speed_axis = figure.add_subplot(122)
    add_beam_geometry(
        spatial_axis,
        cooling_beams,
        trapping_beams,
        extent_m=1.0e-3 * resolved_extent_mm,
    )
    _format_spatial_axis(spatial_axis, resolved_extent_mm)
    spatial_axis.set_title("Atom location")
    spatial_axis.legend(handles=_geometry_legend()[:3], loc="upper left", fontsize=7)
    trail, = spatial_axis.plot([], [], [], color="black", linewidth=1.5)
    atom, = spatial_axis.plot([], [], [], marker="o", color="#d1495b", markersize=7)
    status = spatial_axis.text2D(0.03, 0.02, "", transform=spatial_axis.transAxes)

    speed_axis.plot(time_ms, speed, color="#4c78a8", linewidth=1.4, alpha=0.55)
    speed_marker, = speed_axis.plot([], [], marker="o", color="#d1495b", markersize=7)
    speed_axis.set(
        xlabel="time [ms]",
        ylabel="speed [m/s]",
        title="Speed evolution",
        xlim=(float(time_ms[0]), float(time_ms[-1]) if time_ms[-1] > 0.0 else 1.0),
    )
    speed_top = max(1.0, 1.08 * float(np.max(speed)))
    speed_axis.set_ylim(0.0, speed_top)
    speed_axis.grid(alpha=0.22)

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        current = positions_mm[index]
        trail.set_data(positions_mm[: index + 1, 0], positions_mm[: index + 1, 1])
        trail.set_3d_properties(positions_mm[: index + 1, 2])
        atom.set_data((current[0],), (current[1],))
        atom.set_3d_properties((current[2],))
        speed_marker.set_data((time_ms[index],), (speed[index],))
        status.set_text(
            f"t = {time_ms[index]:.3f} ms\n"
            f"r = ({current[0]:.3f}, {current[1]:.3f}, {current[2]:.3f}) mm\n"
            f"speed = {speed[index]:.4g} m/s"
        )
        return trail, atom, speed_marker, status

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    figure.tight_layout()
    return animation


__all__ = [
    "add_beam_geometry",
    "animate_trajectory",
    "plot_force_velocity_history",
    "plot_trajectory_3d",
    "trajectory_arrays",
]
