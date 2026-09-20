"""Initial-condition builders and trajectory diagnostics for pMOT studies."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import sin
from math import sqrt

import numpy as np

from ..fields import MOTBeam
from ..fields import beams_for_axis
from ..state import AtomState
from .preliminary_scattering import TrajectoryRecord
from .preliminary_scattering import total_scattering_rate_per_s


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrajectoryDiagnostics:
    """Derived scalar histories for one simulated trajectory."""

    times_s: list[float]
    x_mm: list[float]
    y_mm: list[float]
    z_mm: list[float]
    radius_mm: list[float]
    vx_m_per_s: list[float]
    vy_m_per_s: list[float]
    vz_m_per_s: list[float]
    speed_m_per_s: list[float]
    scattering_total_per_s: list[float]
    scattering_x_axis_per_s: list[float]
    scattering_y_axis_per_s: list[float]
    scattering_z_axis_per_s: list[float]


@dataclass(frozen=True, slots=True)
class AnimationSamples:
    """Downsampled trajectory arrays for responsive notebook animation."""

    frame_indices: list[int]
    times_ms: list[float]
    x_mm: list[float]
    y_mm: list[float]
    z_mm: list[float]
    vx_m_per_s: list[float]
    vy_m_per_s: list[float]
    vz_m_per_s: list[float]
    speed_m_per_s: list[float]
    static_x_mm: list[float]
    static_y_mm: list[float]
    static_z_mm: list[float]


def unit_vector_from_angles(azimuth_rad: float, polar_rad: float) -> Vec3:
    """Return a 3D unit vector from standard spherical angles.

    `polar_rad` is measured from +z.
    `azimuth_rad` is measured in the x-y plane from +x toward +y.
    """

    sin_polar = sin(polar_rad)
    return (
        sin_polar * cos(azimuth_rad),
        sin_polar * sin(azimuth_rad),
        cos(polar_rad),
    )


def inward_radial_atom_state(
    radial_distance_m: float,
    speed_m_per_s: float,
    azimuth_rad: float,
    polar_rad: float,
) -> AtomState:
    """Place the atom on a sphere and launch it inward toward the origin."""

    if radial_distance_m < 0.0:
        raise ValueError("radial_distance_m must be non-negative")
    if speed_m_per_s < 0.0:
        raise ValueError("speed_m_per_s must be non-negative")
    direction = unit_vector_from_angles(azimuth_rad, polar_rad)
    position = tuple(radial_distance_m * component for component in direction)
    velocity = tuple(-speed_m_per_s * component for component in direction)
    return AtomState(position_m=position, velocity_m_per_s=velocity)


def trajectory_diagnostics(
    beams: list[MOTBeam],
    trajectory: TrajectoryRecord,
    active_transition: str = "cooling",
) -> TrajectoryDiagnostics:
    """Compute axis-resolved positions, speeds, and scattering rates."""

    x_axis_beams = beams_for_axis(beams, "horizontal_x")
    y_axis_beams = beams_for_axis(beams, "horizontal_y")
    z_axis_beams = beams_for_axis(beams, "vertical_z")

    x_mm: list[float] = []
    y_mm: list[float] = []
    z_mm: list[float] = []
    radius_mm: list[float] = []
    vx_m_per_s: list[float] = []
    vy_m_per_s: list[float] = []
    vz_m_per_s: list[float] = []
    speed_m_per_s: list[float] = []
    scattering_total_per_s: list[float] = []
    scattering_x_axis_per_s: list[float] = []
    scattering_y_axis_per_s: list[float] = []
    scattering_z_axis_per_s: list[float] = []

    for position, velocity in zip(trajectory.positions_m, trajectory.velocities_m_per_s):
        atom_state = AtomState(position_m=position, velocity_m_per_s=velocity)
        x_mm.append(1e3 * position[0])
        y_mm.append(1e3 * position[1])
        z_mm.append(1e3 * position[2])
        radius_mm.append(1e3 * sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2))
        vx_m_per_s.append(velocity[0])
        vy_m_per_s.append(velocity[1])
        vz_m_per_s.append(velocity[2])
        speed_m_per_s.append(sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2))
        scattering_total_per_s.append(
            total_scattering_rate_per_s(beams, atom_state, active_transition=active_transition)
        )
        scattering_x_axis_per_s.append(
            total_scattering_rate_per_s(x_axis_beams, atom_state, active_transition=active_transition)
        )
        scattering_y_axis_per_s.append(
            total_scattering_rate_per_s(y_axis_beams, atom_state, active_transition=active_transition)
        )
        scattering_z_axis_per_s.append(
            total_scattering_rate_per_s(z_axis_beams, atom_state, active_transition=active_transition)
        )

    return TrajectoryDiagnostics(
        times_s=trajectory.times_s,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        radius_mm=radius_mm,
        vx_m_per_s=vx_m_per_s,
        vy_m_per_s=vy_m_per_s,
        vz_m_per_s=vz_m_per_s,
        speed_m_per_s=speed_m_per_s,
        scattering_total_per_s=scattering_total_per_s,
        scattering_x_axis_per_s=scattering_x_axis_per_s,
        scattering_y_axis_per_s=scattering_y_axis_per_s,
        scattering_z_axis_per_s=scattering_z_axis_per_s,
    )


def animation_samples(
    trajectory: TrajectoryRecord,
    diagnostics: TrajectoryDiagnostics,
    max_animation_frames: int = 250,
    max_static_path_points: int = 1500,
) -> AnimationSamples:
    """Return frame-thinned arrays for interactive trajectory animation."""

    if max_animation_frames < 2:
        raise ValueError("max_animation_frames must be at least 2")
    if max_static_path_points < 2:
        raise ValueError("max_static_path_points must be at least 2")

    x_full = np.asarray(diagnostics.x_mm, dtype=float)
    y_full = np.asarray(diagnostics.y_mm, dtype=float)
    z_full = np.asarray(diagnostics.z_mm, dtype=float)
    velocity_full = np.asarray(trajectory.velocities_m_per_s, dtype=float)
    speed_full = np.asarray(diagnostics.speed_m_per_s, dtype=float)
    times_full_ms = 1e3 * np.asarray(diagnostics.times_s, dtype=float)

    if len(times_full_ms) == 0:
        raise ValueError("trajectory contains no samples")

    frame_indices = np.linspace(
        0,
        len(times_full_ms) - 1,
        min(max_animation_frames, len(times_full_ms)),
        dtype=int,
    )
    frame_indices = np.unique(frame_indices)

    static_stride = max(1, int(np.ceil(len(x_full) / max_static_path_points)))
    static_x = x_full[::static_stride]
    static_y = y_full[::static_stride]
    static_z = z_full[::static_stride]
    if static_x[-1] != x_full[-1] or static_y[-1] != y_full[-1] or static_z[-1] != z_full[-1]:
        static_x = np.append(static_x, x_full[-1])
        static_y = np.append(static_y, y_full[-1])
        static_z = np.append(static_z, z_full[-1])

    return AnimationSamples(
        frame_indices=frame_indices.tolist(),
        times_ms=times_full_ms[frame_indices].tolist(),
        x_mm=x_full[frame_indices].tolist(),
        y_mm=y_full[frame_indices].tolist(),
        z_mm=z_full[frame_indices].tolist(),
        vx_m_per_s=velocity_full[frame_indices, 0].tolist(),
        vy_m_per_s=velocity_full[frame_indices, 1].tolist(),
        vz_m_per_s=velocity_full[frame_indices, 2].tolist(),
        speed_m_per_s=speed_full[frame_indices].tolist(),
        static_x_mm=static_x.tolist(),
        static_y_mm=static_y.tolist(),
        static_z_mm=static_z.tolist(),
    )
