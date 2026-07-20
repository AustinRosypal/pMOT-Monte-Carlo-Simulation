"""Cooling-beam geometry and field-sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import pi
from math import sin

from .beams import GaussianBeam
from .beams import Vec3
from .beams import add
from .beams import axis_direction_from_name
from .beams import beam_frame_coordinates
from .beams import dot
from .beams import focused_waist_radius
from .beams import normalize
from .beams import scale
from .configuration import PMOTSimulationConfig
from .configuration import default_simulation_config


@dataclass(frozen=True, slots=True)
class CoolingBeam:
    """One focused 780 nm cooling beam in the 3D apparatus."""

    label: str
    axis_name: str
    direction: Vec3
    waist_position_m: Vec3
    power_w: float
    wavelength_m: float
    waist_radius_m: float

    def __post_init__(self) -> None:
        if self.power_w < 0.0:
            raise ValueError("power_w must be non-negative")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        if self.waist_radius_m <= 0.0:
            raise ValueError("waist_radius_m must be positive")
        object.__setattr__(self, "direction", normalize(self.direction))

    def gaussian_beam(self) -> GaussianBeam:
        return GaussianBeam(
            power_w=self.power_w,
            wavelength_m=self.wavelength_m,
            waist_radius_m=self.waist_radius_m,
        )


def beam_intensity_w_per_m2(beam: CoolingBeam, position_m: Vec3) -> float:
    """Return the local Gaussian intensity from one cooling beam."""

    axial_m, radial_m = beam_frame_coordinates(position_m, beam.waist_position_m, beam.direction)
    return beam.gaussian_beam().intensity(radial_m=radial_m, axial_m=axial_m)


def build_cooling_beams(
    config: PMOTSimulationConfig | None = None,
) -> list[CoolingBeam]:
    """Build the six 780 nm cooling beams from the current apparatus configuration."""

    apparatus = config or default_simulation_config()
    waist_radius_m = focused_waist_radius(
        wavelength_m=apparatus.cooling.wavelength_m,
        focal_length_m=apparatus.lens.focal_length_m,
        input_radius_m=apparatus.trap_beam.input_radius_m,
    )

    beams: list[CoolingBeam] = []
    for axis in apparatus.axes:
        axis_direction = axis_direction_from_name(axis.name)
        incident_waist = scale(apparatus.trap_beam.incident_focus_offset_m, axis_direction)
        retro_waist = scale(apparatus.trap_beam.retro_focus_offset_m, axis_direction)
        beams.append(
            CoolingBeam(
                label=f"{axis.name}_incident_780",
                axis_name=axis.name,
                direction=axis_direction,
                waist_position_m=incident_waist,
                power_w=apparatus.cooling.power_w_per_beam,
                wavelength_m=apparatus.cooling.wavelength_m,
                waist_radius_m=waist_radius_m,
            )
        )
        beams.append(
            CoolingBeam(
                label=f"{axis.name}_retro_780",
                axis_name=axis.name,
                direction=scale(-1.0, axis_direction),
                waist_position_m=retro_waist,
                power_w=apparatus.cooling.power_w_per_beam,
                wavelength_m=apparatus.cooling.wavelength_m,
                waist_radius_m=waist_radius_m,
            )
        )
    return beams


def total_intensity_w_per_m2(beams: list[CoolingBeam], position_m: Vec3) -> float:
    return sum(beam_intensity_w_per_m2(beam, position_m) for beam in beams)


def sample_intensity_along_line(
    beams: list[CoolingBeam],
    start_m: Vec3,
    stop_m: Vec3,
    sample_count: int = 501,
) -> tuple[list[float], list[float]]:
    """Sample the total intensity along a Cartesian line segment.

    The returned coordinate is centered at zero and runs from ``-L/2`` to ``+L/2``.
    """

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    total_length_m = (
        (stop_m[0] - start_m[0]) ** 2
        + (stop_m[1] - start_m[1]) ** 2
        + (stop_m[2] - start_m[2]) ** 2
    ) ** 0.5
    coordinates_mm: list[float] = []
    intensities: list[float] = []
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        point = (
            start_m[0] + fraction * (stop_m[0] - start_m[0]),
            start_m[1] + fraction * (stop_m[1] - start_m[1]),
            start_m[2] + fraction * (stop_m[2] - start_m[2]),
        )
        coordinates_mm.append(1e3 * (fraction - 0.5) * total_length_m)
        intensities.append(total_intensity_w_per_m2(beams, point))
    return coordinates_mm, intensities


def sample_intensity_volume(
    beams: list[CoolingBeam],
    extent_m: float = 50e-3,
    samples_per_axis: int = 31,
) -> list[tuple[float, float, float, float]]:
    """Sample the total cooling intensity on a 3D Cartesian grid."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    points: list[tuple[float, float, float, float]] = []
    step = 2.0 * extent_m / (samples_per_axis - 1)
    for ix in range(samples_per_axis):
        x_value = -extent_m + ix * step
        for iy in range(samples_per_axis):
            y_value = -extent_m + iy * step
            for iz in range(samples_per_axis):
                z_value = -extent_m + iz * step
                points.append(
                    (
                        x_value,
                        y_value,
                        z_value,
                        total_intensity_w_per_m2(beams, (x_value, y_value, z_value)),
                    )
                )
    return points


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def orthonormal_transverse_basis(direction: Vec3) -> tuple[Vec3, Vec3]:
    trial = (0.0, 0.0, 1.0)
    if abs(dot(direction, trial)) > 0.9:
        trial = (0.0, 1.0, 0.0)
    basis_1 = normalize(cross(direction, trial))
    basis_2 = normalize(cross(direction, basis_1))
    return basis_1, basis_2


def sample_intensity_cloud(
    beams: list[CoolingBeam],
    axial_extent_m: float = 50e-3,
    axial_samples: int = 21,
    radial_rings: int = 4,
    angular_samples: int = 16,
    radial_waist_scale: float = 3.0,
) -> list[tuple[float, float, float, float]]:
    """Sample the cooling field around each beam axis for 3D visualizations."""

    if axial_samples < 2:
        raise ValueError("axial_samples must be at least 2")
    points: list[tuple[float, float, float, float]] = []
    for beam in beams:
        basis_1, basis_2 = orthonormal_transverse_basis(beam.direction)
        gaussian = beam.gaussian_beam()
        for axial_index in range(axial_samples):
            axial_m = -axial_extent_m + axial_index * (2.0 * axial_extent_m) / (axial_samples - 1)
            center = add(beam.waist_position_m, scale(axial_m, beam.direction))
            local_waist = gaussian.waist_at(axial_m)
            max_radius = radial_waist_scale * local_waist
            points.append((center[0], center[1], center[2], total_intensity_w_per_m2(beams, center)))
            for ring in range(1, radial_rings + 1):
                radius = max_radius * ring / radial_rings
                for angle_index in range(angular_samples):
                    angle = 2.0 * pi * angle_index / angular_samples
                    offset = add(
                        scale(radius * cos(angle), basis_1),
                        scale(radius * sin(angle), basis_2),
                    )
                    point = add(center, offset)
                    points.append((point[0], point[1], point[2], total_intensity_w_per_m2(beams, point)))
    return points


def sample_intensity_slice(
    beams: list[CoolingBeam],
    plane: str,
    extent_m: float = 50e-3,
    samples_per_axis: int = 201,
    fixed_coordinate_m: float = 0.0,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Sample the total cooling intensity on a 2D slice."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")
    axis_values = [
        -extent_m + index * (2.0 * extent_m) / (samples_per_axis - 1)
        for index in range(samples_per_axis)
    ]
    grid: list[list[float]] = []
    for second_value in axis_values:
        row: list[float] = []
        for first_value in axis_values:
            if plane == "xy":
                point = (first_value, second_value, fixed_coordinate_m)
            elif plane == "xz":
                point = (first_value, fixed_coordinate_m, second_value)
            else:
                point = (fixed_coordinate_m, first_value, second_value)
            row.append(total_intensity_w_per_m2(beams, point))
        grid.append(row)
    return axis_values, axis_values, grid
