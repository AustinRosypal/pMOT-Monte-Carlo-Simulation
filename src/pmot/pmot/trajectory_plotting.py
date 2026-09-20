r"""True-scale pMOT beam and single-trajectory visualizations.

The geometry renderer deliberately depends only on the public cooling/repump
and trapping-beam records.  It can therefore be reused by notebooks, batch
studies, and future trajectory solvers without coupling the geometry layer to
one provisional force model.

The collinear 780-nm cooling and repump beams are shown as the three overlapping
cylindrical paths formed by their counterpropagating components.  The 1529-nm
components are shown individually at their Gaussian :math:`1/e^2` intensity radii,

.. math::

   w(s) = w_0 \sqrt{1 + [(s-s_0)/z_R]^2}.

All mesh coordinates returned by this module are in millimetres for direct use
with Matplotlib axes.  Beam inputs and trajectory records remain in SI units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from ..beam_plotting import beam_surface_mesh_mm
from ..fields import MOTBeam
from ..fields import orthonormal_transverse_basis
from .trapping_beams import TrappingBeam


AXIS_COLORS = {
    "horizontal_x": "#f9a8d4",
    "horizontal_y": "#93c5fd",
    "vertical_z": "#86efac",
}
TRAPPING_SENSE_COLORS = {
    "incident": "#2563eb",
    "retro": "#ea580c",
}


@dataclass(frozen=True, slots=True)
class PMOTBeamArtists:
    """Artists added by :func:`draw_pmot_beam_volumes`."""

    cooling_surfaces: tuple[Any, ...]
    trapping_surfaces: tuple[Any, ...]
    waist_markers: tuple[Any, ...]
    direction_arrows: tuple[Any, ...]


def _validate_mesh_resolution(axial_samples: int, angular_samples: int) -> None:
    if axial_samples < 2:
        raise ValueError("axial_samples must be at least 2")
    if angular_samples < 4:
        raise ValueError("angular_samples must be at least 4")


def gaussian_trapping_surface_mesh_mm(
    beam: TrappingBeam,
    *,
    axial_extent_m: float = 20.0e-3,
    axial_samples: int = 81,
    angular_samples: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one trapping component's Gaussian ``1/e^2`` envelope mesh.

    ``axial_extent_m`` is a symmetric laboratory-coordinate extent about the
    trap origin, not a distance about the waist.  This convention makes the
    incident and retro envelopes cover the identical displayed trap volume
    while retaining their separate waist locations and Rayleigh ranges.
    """

    if axial_extent_m <= 0.0:
        raise ValueError("axial_extent_m must be positive")
    _validate_mesh_resolution(axial_samples, angular_samples)

    direction = np.asarray(beam.direction, dtype=float)
    direction /= np.linalg.norm(direction)
    basis_1, basis_2 = orthonormal_transverse_basis(tuple(direction))
    basis_1 = np.asarray(basis_1, dtype=float)
    basis_2 = np.asarray(basis_2, dtype=float)

    waist = np.asarray(beam.waist_position_m, dtype=float)
    waist_coordinate_m = float(np.dot(waist, direction))
    transverse_offset_m = waist - waist_coordinate_m * direction
    axial_coordinates_m = np.linspace(
        -axial_extent_m,
        axial_extent_m,
        axial_samples,
    )
    angles_rad = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=True)
    axial_grid_m, angle_grid_rad = np.meshgrid(
        axial_coordinates_m,
        angles_rad,
        indexing="ij",
    )
    local_radius_m = beam.waist_radius_m * np.sqrt(
        1.0
        + ((axial_grid_m - waist_coordinate_m) / beam.rayleigh_range_m) ** 2
    )
    centerline_m = (
        transverse_offset_m[None, None, :]
        + axial_grid_m[..., None] * direction[None, None, :]
    )
    radial_direction = (
        np.cos(angle_grid_rad)[..., None] * basis_1[None, None, :]
        + np.sin(angle_grid_rad)[..., None] * basis_2[None, None, :]
    )
    surface_mm = 1.0e3 * (
        centerline_m + local_radius_m[..., None] * radial_direction
    )
    return (
        surface_mm[..., 0],
        surface_mm[..., 1],
        surface_mm[..., 2],
    )


def _cooling_beams_by_axis(cooling_repump_beams: list[MOTBeam]) -> dict[str, MOTBeam]:
    selected: dict[str, MOTBeam] = {}
    for beam in cooling_repump_beams:
        if beam.family != "cooling":
            continue
        selected.setdefault(beam.axis_name, beam)
    if len(selected) != 3:
        raise ValueError("expected cooling beams on exactly three Cartesian axes")
    return selected


def draw_pmot_beam_volumes(
    axis,
    cooling_repump_beams: list[MOTBeam],
    trapping_beams: list[TrappingBeam],
    *,
    axial_extent_m: float = 20.0e-3,
    cooling_axial_samples: int = 25,
    trapping_axial_samples: int = 81,
    angular_samples: int = 40,
) -> PMOTBeamArtists:
    """Draw true-scale 780-nm cylinders and 1529-nm Gaussian envelopes.

    The common cooling/repump radius is read from each cooling ``MOTBeam``
    (6.35 mm for the current 12.7-mm-diameter default).  Cooling, repump, and
    their counterpropagating components are collinear and have the same default
    transverse envelope, so a single shared cylinder is drawn per Cartesian
    path.  All six focused trapping components are drawn separately.
    Blue/orange encodes incident/retro propagation, and each waist is marked
    with ``x``.
    """

    if axial_extent_m <= 0.0:
        raise ValueError("axial_extent_m must be positive")
    _validate_mesh_resolution(cooling_axial_samples, angular_samples)
    _validate_mesh_resolution(trapping_axial_samples, angular_samples)
    if len(trapping_beams) != 6:
        raise ValueError("expected six trapping components")

    cooling_surfaces: list[Any] = []
    for axis_name, beam in _cooling_beams_by_axis(cooling_repump_beams).items():
        x_mm, y_mm, z_mm = beam_surface_mesh_mm(
            direction=beam.direction,
            radius_m=beam.beam_radius_m,
            length_m=2.0 * axial_extent_m,
            axial_samples=cooling_axial_samples,
            angular_samples=angular_samples,
        )
        cooling_surfaces.append(
            axis.plot_surface(
                x_mm,
                y_mm,
                z_mm,
                color=AXIS_COLORS.get(axis_name, "#94a3b8"),
                linewidth=0.0,
                antialiased=True,
                shade=False,
                alpha=0.10,
            )
        )

    trapping_surfaces: list[Any] = []
    waist_markers: list[Any] = []
    direction_arrows: list[Any] = []
    arrow_length_m = min(3.0e-3, 0.18 * axial_extent_m)
    for beam in trapping_beams:
        x_mm, y_mm, z_mm = gaussian_trapping_surface_mesh_mm(
            beam,
            axial_extent_m=axial_extent_m,
            axial_samples=trapping_axial_samples,
            angular_samples=angular_samples,
        )
        color = TRAPPING_SENSE_COLORS[beam.propagation_sense]
        trapping_surfaces.append(
            axis.plot_surface(
                x_mm,
                y_mm,
                z_mm,
                color=color,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                alpha=0.12,
            )
        )
        waist_mm = 1.0e3 * np.asarray(beam.waist_position_m, dtype=float)
        waist_markers.append(
            axis.scatter(
                [waist_mm[0]],
                [waist_mm[1]],
                [waist_mm[2]],
                color=color,
                marker="x",
                s=34,
                linewidth=1.4,
                depthshade=False,
            )
        )
        direction = np.asarray(beam.direction, dtype=float)
        arrow_start_m = np.asarray(beam.waist_position_m, dtype=float) - 0.5 * arrow_length_m * direction
        direction_arrows.append(
            axis.quiver(
                *(1.0e3 * arrow_start_m),
                *(1.0e3 * arrow_length_m * direction),
                color=color,
                linewidth=1.2,
                arrow_length_ratio=0.28,
            )
        )

    return PMOTBeamArtists(
        cooling_surfaces=tuple(cooling_surfaces),
        trapping_surfaces=tuple(trapping_surfaces),
        waist_markers=tuple(waist_markers),
        direction_arrows=tuple(direction_arrows),
    )


def pmot_beam_legend_handles(
    cooling_diameter_mm: float = 12.7,
) -> list[Any]:
    """Return stable proxy artists for the true-scale geometry legend."""

    if cooling_diameter_mm <= 0.0:
        raise ValueError("cooling_diameter_mm must be positive")
    return [
        Patch(
            facecolor="#86efac",
            alpha=0.24,
            label=(
                "780 nm cooling/repump volume "
                f"({cooling_diameter_mm:g} mm diameter)"
            ),
        ),
        Patch(facecolor=TRAPPING_SENSE_COLORS["incident"], alpha=0.28, label="1529 nm incident 1/e² envelope"),
        Patch(facecolor=TRAPPING_SENSE_COLORS["retro"], alpha=0.28, label="1529 nm retro 1/e² envelope"),
        Line2D([], [], color="#334155", marker="x", linestyle="None", label="1529 nm waist center"),
    ]


def _rate_record(record):
    base = getattr(record, "rate_equation", record)
    required = (
        "times_s",
        "positions_m",
        "velocities_m_per_s",
        "forces_n",
        "total_scattering_rates_per_s",
        "beam_scattering_rates_per_s",
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise TypeError(f"trajectory record is missing fields: {', '.join(missing)}")
    return base


def plot_pmot_trajectory_diagnostics(
    record,
    cooling_repump_beams: list[MOTBeam],
    trapping_beams: list[TrappingBeam],
    *,
    path: Path | None = None,
    title: str = "Vector-only pMOT single-atom trajectory",
    axial_extent_m: float = 20.0e-3,
):
    """Plot a 3D trajectory with true-scale beams and four time histories.

    ``record`` may be a rate-equation trajectory directly or any pMOT record
    exposing it as ``record.rate_equation``.  This intentionally supports the
    vector-only pMOT record without importing its solver module.
    """

    base = _rate_record(record)
    times_ms = 1.0e3 * np.asarray(base.times_s, dtype=float)
    positions_mm = 1.0e3 * np.asarray(base.positions_m, dtype=float)
    velocities = np.asarray(base.velocities_m_per_s, dtype=float)
    forces_zeptonewton = 1.0e21 * np.asarray(base.forces_n, dtype=float)
    beam_rates = np.asarray(base.beam_scattering_rates_per_s, dtype=float)
    total_rates = np.asarray(base.total_scattering_rates_per_s, dtype=float)
    if times_ms.ndim != 1 or len(times_ms) == 0:
        raise ValueError("trajectory must contain at least one time sample")
    for name, values in (
        ("positions_m", positions_mm),
        ("velocities_m_per_s", velocities),
        ("forces_n", forces_zeptonewton),
    ):
        if values.shape != (len(times_ms), 3):
            raise ValueError(f"{name} must have shape (time, 3)")
    if total_rates.shape != (len(times_ms),):
        raise ValueError("total_scattering_rates_per_s must have shape (time,)")
    if beam_rates.ndim != 2 or beam_rates.shape[0] != len(times_ms):
        raise ValueError("beam_scattering_rates_per_s must have shape (time, beam)")

    figure = plt.figure(figsize=(16.0, 10.0), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    grid = figure.add_gridspec(2, 3, width_ratios=(1.25, 1.0, 1.0))
    trajectory_axis = figure.add_subplot(grid[:, 0], projection="3d")
    position_axis = figure.add_subplot(grid[0, 1])
    velocity_axis = figure.add_subplot(grid[0, 2])
    force_axis = figure.add_subplot(grid[1, 1])
    scattering_axis = figure.add_subplot(grid[1, 2])

    draw_pmot_beam_volumes(
        trajectory_axis,
        cooling_repump_beams,
        trapping_beams,
        axial_extent_m=axial_extent_m,
    )
    trajectory_axis.plot(
        positions_mm[:, 0],
        positions_mm[:, 1],
        positions_mm[:, 2],
        color="#111827",
        linewidth=3.0,
        label="atom trajectory",
        zorder=10,
    )
    marker_stride = max(1, len(positions_mm) // 180)
    trajectory_axis.scatter(
        positions_mm[::marker_stride, 0],
        positions_mm[::marker_stride, 1],
        positions_mm[::marker_stride, 2],
        color="#111827",
        s=4,
        alpha=0.72,
        depthshade=False,
        zorder=11,
    )
    trajectory_axis.scatter(
        *positions_mm[0], color="#b91c1c", s=42, label="start", depthshade=False
    )
    trajectory_axis.scatter(
        *positions_mm[-1], color="#111827", s=42, label="end", depthshade=False
    )
    trajectory_axis.scatter(0.0, 0.0, 0.0, color="#7c3aed", marker="+", s=55)
    plot_limit_mm = max(
        1.0e3 * axial_extent_m,
        1.05 * float(np.max(np.abs(positions_mm))),
    )
    trajectory_axis.set(
        title="Trajectory and optical geometry (true scale)",
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        xlim=(-plot_limit_mm, plot_limit_mm),
        ylim=(-plot_limit_mm, plot_limit_mm),
        zlim=(-plot_limit_mm, plot_limit_mm),
    )
    trajectory_axis.set_box_aspect((1.0, 1.0, 1.0))
    cooling_by_axis = _cooling_beams_by_axis(cooling_repump_beams)
    displayed_cooling_diameter_mm = 2.0e3 * next(iter(cooling_by_axis.values())).beam_radius_m
    trajectory_axis.legend(
        handles=pmot_beam_legend_handles(displayed_cooling_diameter_mm)
        + [
            Line2D([], [], color="#111827", linewidth=3.0, label="atom trajectory"),
            Line2D([], [], marker="o", color="#b91c1c", linestyle="None", label="launch"),
        ],
        loc="upper left",
        fontsize=7.5,
    )

    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    for component, (label, color) in enumerate(zip("xyz", colors)):
        position_axis.plot(times_ms, positions_mm[:, component], color=color, label=label)
        velocity_axis.plot(times_ms, velocities[:, component], color=color, label=f"v{label}")
        force_axis.plot(
            times_ms,
            forces_zeptonewton[:, component],
            color=color,
            label=f"F{label}",
        )

    cooling_indices = [
        index for index, beam in enumerate(cooling_repump_beams) if beam.family == "cooling"
    ]
    repump_indices = [
        index for index, beam in enumerate(cooling_repump_beams) if beam.family == "repump"
    ]
    if beam_rates.shape[1] != len(cooling_repump_beams):
        raise ValueError("beam scattering columns must match cooling_repump_beams")
    scattering_axis.plot(
        times_ms,
        np.sum(beam_rates[:, cooling_indices], axis=1),
        color="#7c3aed",
        label="cooling absorption",
    )
    scattering_axis.plot(
        times_ms,
        np.sum(beam_rates[:, repump_indices], axis=1),
        color="#c2410c",
        label="repump absorption",
    )
    scattering_axis.plot(
        times_ms,
        total_rates,
        color="#111827",
        linestyle="--",
        label="total absorption",
    )
    position_axis.set(title="Cartesian position", ylabel="Position [mm]")
    velocity_axis.set(title="Cartesian velocity", ylabel="Velocity [m/s]")
    force_axis.set(
        title="780-nm absorption-momentum force (gravity excluded)",
        ylabel=r"Force [$10^{-21}$ N]",
    )
    scattering_axis.set(
        title="Ground-weighted absorption rate",
        ylabel=r"Rate [s$^{-1}$]",
    )
    for panel in (position_axis, velocity_axis, force_axis, scattering_axis):
        panel.set_xlabel("Time [ms]")
        panel.axhline(0.0, color="0.45", linewidth=0.7)
        panel.grid(alpha=0.24)
        panel.legend(loc="best", fontsize=8)
    figure.suptitle(title)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=190, bbox_inches="tight")
    return figure, {
        "trajectory": trajectory_axis,
        "position": position_axis,
        "velocity": velocity_axis,
        "force": force_axis,
        "scattering": scattering_axis,
    }


__all__ = [
    "PMOTBeamArtists",
    "draw_pmot_beam_volumes",
    "gaussian_trapping_surface_mesh_mm",
    "plot_pmot_trajectory_diagnostics",
    "pmot_beam_legend_handles",
]
