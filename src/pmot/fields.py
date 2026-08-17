"""Cooling and repump beam geometry and field-sampling utilities."""

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
from .beams import normalize
from .beams import scale
from .configuration import MOTBeamConfig
from .configuration import PMOTSimulationConfig
from .configuration import default_simulation_config


@dataclass(frozen=True, slots=True)
class MOTBeam:
    """One collimated cooling or repump beam in the 3D apparatus."""

    label: str
    family: str
    axis_name: str
    propagation_sense: str
    circular_polarization: str
    direction: Vec3
    reference_position_m: Vec3
    power_w: float
    wavelength_m: float
    resonance_frequency_hz: float
    detuning_hz: float
    beam_radius_m: float

    def __post_init__(self) -> None:
        if self.power_w < 0.0:
            raise ValueError("power_w must be non-negative")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        if self.resonance_frequency_hz <= 0.0:
            raise ValueError("resonance_frequency_hz must be positive")
        if self.beam_radius_m <= 0.0:
            raise ValueError("beam_radius_m must be positive")
        if self.family not in {"cooling", "repump"}:
            raise ValueError("family must be 'cooling' or 'repump'")
        if self.propagation_sense not in {"incident", "retro"}:
            raise ValueError("propagation_sense must be 'incident' or 'retro'")
        if self.circular_polarization not in {"pi", "sigma+", "sigma-"}:
            raise ValueError("circular_polarization must be 'pi', 'sigma+', or 'sigma-'")
        object.__setattr__(self, "direction", normalize(self.direction))

    def gaussian_beam(self) -> GaussianBeam:
        return GaussianBeam(
            power_w=self.power_w,
            wavelength_m=self.wavelength_m,
            waist_radius_m=self.beam_radius_m,
        )

    @property
    def laser_frequency_hz(self) -> float:
        return self.resonance_frequency_hz + self.detuning_hz


def beam_intensity_w_per_m2(beam: MOTBeam, position_m: Vec3) -> float:
    """Return the local Gaussian intensity from one collimated MOT beam."""

    axial_m, radial_m = beam_frame_coordinates(position_m, beam.reference_position_m, beam.direction)
    return beam.gaussian_beam().intensity(radial_m=radial_m, axial_m=axial_m)


def _build_beam_family(
    family_config: MOTBeamConfig,
    family: str,
    axes,
) -> list[MOTBeam]:
    beams: list[MOTBeam] = []
    for axis in axes:
        axis_direction = axis_direction_from_name(axis.name)
        reference_position = (0.0, 0.0, 0.0)
        beams.append(
            MOTBeam(
                label=f"{axis.name}_incident_{family_config.name}",
                family=family,
                axis_name=axis.name,
                propagation_sense="incident",
                circular_polarization="sigma+",
                direction=axis_direction,
                reference_position_m=reference_position,
                power_w=family_config.power_w_per_beam,
                wavelength_m=family_config.wavelength_m,
                resonance_frequency_hz=family_config.resonance_frequency_hz,
                detuning_hz=family_config.detuning_hz,
                beam_radius_m=family_config.beam_radius_m,
            )
        )
        beams.append(
            MOTBeam(
                label=f"{axis.name}_retro_{family_config.name}",
                family=family,
                axis_name=axis.name,
                propagation_sense="retro",
                circular_polarization="sigma-",
                direction=scale(-1.0, axis_direction),
                reference_position_m=reference_position,
                power_w=family_config.power_w_per_beam,
                wavelength_m=family_config.wavelength_m,
                resonance_frequency_hz=family_config.resonance_frequency_hz,
                detuning_hz=family_config.detuning_hz,
                beam_radius_m=family_config.beam_radius_m,
            )
        )
    return beams


def build_mot_beams(
    config: PMOTSimulationConfig | None = None,
) -> list[MOTBeam]:
    """Build the 12 MOT beams: 6 cooling and 6 repump beams."""

    apparatus = config or default_simulation_config()
    return _build_beam_family(apparatus.cooling, "cooling", apparatus.axes) + _build_beam_family(
        apparatus.repump, "repump", apparatus.axes
    )


def total_intensity_w_per_m2(beams: list[MOTBeam], position_m: Vec3) -> float:
    return sum(beam_intensity_w_per_m2(beam, position_m) for beam in beams)


def beams_for_axis(beams: list[MOTBeam], axis_name: str) -> list[MOTBeam]:
    """Return the beams associated with one modeled apparatus axis."""

    selected = [beam for beam in beams if beam.axis_name == axis_name]
    if not selected:
        raise ValueError(f"no MOT beams found for axis '{axis_name}'")
    return selected


def beams_for_family(beams: list[MOTBeam], family: str) -> list[MOTBeam]:
    """Return the beams associated with one optical family."""

    selected = [beam for beam in beams if beam.family == family]
    if not selected:
        raise ValueError(f"no MOT beams found for family '{family}'")
    return selected


def beams_for_polarization(beams: list[MOTBeam], circular_polarization: str) -> list[MOTBeam]:
    """Return the beams associated with one circular-polarization handedness."""

    selected = [beam for beam in beams if beam.circular_polarization == circular_polarization]
    if not selected:
        raise ValueError(f"no MOT beams found for polarization '{circular_polarization}'")
    return selected


def filter_beams(
    beams: list[MOTBeam],
    axis_name: str | None = None,
    family: str | None = None,
    circular_polarization: str | None = None,
    propagation_sense: str | None = None,
) -> list[MOTBeam]:
    """Return beams matching the requested metadata filters."""

    selected = beams
    if axis_name is not None:
        selected = [beam for beam in selected if beam.axis_name == axis_name]
    if family is not None:
        selected = [beam for beam in selected if beam.family == family]
    if circular_polarization is not None:
        selected = [beam for beam in selected if beam.circular_polarization == circular_polarization]
    if propagation_sense is not None:
        selected = [beam for beam in selected if beam.propagation_sense == propagation_sense]
    if not selected:
        raise ValueError("no MOT beams matched the requested filter combination")
    return selected


def axis_pair_intensity_w_per_m2(
    beams: list[MOTBeam],
    axis_name: str,
    position_m: Vec3,
) -> float:
    """Return the summed intensity from all beams on one apparatus axis."""

    return total_intensity_w_per_m2(beams_for_axis(beams, axis_name), position_m)


def sample_intensity_along_line(
    beams: list[MOTBeam],
    start_m: Vec3,
    stop_m: Vec3,
    sample_count: int = 501,
) -> tuple[list[float], list[float]]:
    """Sample the total intensity along a Cartesian line segment."""

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


def sample_axis_pair_intensity_along_line(
    beams: list[MOTBeam],
    axis_name: str,
    start_m: Vec3,
    stop_m: Vec3,
    sample_count: int = 501,
) -> tuple[list[float], list[float]]:
    """Sample the intensity from one apparatus axis along a Cartesian line segment."""

    return sample_intensity_along_line(beams_for_axis(beams, axis_name), start_m, stop_m, sample_count)


def sample_family_intensity_along_line(
    beams: list[MOTBeam],
    family: str,
    start_m: Vec3,
    stop_m: Vec3,
    sample_count: int = 501,
) -> tuple[list[float], list[float]]:
    """Sample the intensity from one optical family along a Cartesian line segment."""

    return sample_intensity_along_line(beams_for_family(beams, family), start_m, stop_m, sample_count)


def sample_filtered_intensity_along_line(
    beams: list[MOTBeam],
    start_m: Vec3,
    stop_m: Vec3,
    sample_count: int = 501,
    axis_name: str | None = None,
    family: str | None = None,
    circular_polarization: str | None = None,
    propagation_sense: str | None = None,
) -> tuple[list[float], list[float]]:
    """Sample the intensity from a filtered beam subset along a Cartesian line segment."""

    selected = filter_beams(
        beams,
        axis_name=axis_name,
        family=family,
        circular_polarization=circular_polarization,
        propagation_sense=propagation_sense,
    )
    return sample_intensity_along_line(selected, start_m, stop_m, sample_count)


def sample_intensity_volume(
    beams: list[MOTBeam],
    extent_m: float = 50e-3,
    samples_per_axis: int = 31,
) -> list[tuple[float, float, float, float]]:
    """Sample the total MOT intensity on a 3D Cartesian grid."""

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
    beams: list[MOTBeam],
    axial_extent_m: float = 50e-3,
    axial_samples: int = 21,
    radial_rings: int = 4,
    angular_samples: int = 16,
    radial_waist_scale: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Sample the MOT field around each beam axis for 3D visualizations."""

    if axial_samples < 2:
        raise ValueError("axial_samples must be at least 2")
    points: list[tuple[float, float, float, float]] = []
    for beam in beams:
        basis_1, basis_2 = orthonormal_transverse_basis(beam.direction)
        gaussian = beam.gaussian_beam()
        for axial_index in range(axial_samples):
            axial_m = -axial_extent_m + axial_index * (2.0 * axial_extent_m) / (axial_samples - 1)
            center = add(beam.reference_position_m, scale(axial_m, beam.direction))
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


def sample_intensity_cloud_by_polarization(
    beams: list[MOTBeam],
    axial_extent_m: float = 50e-3,
    axial_samples: int = 21,
    radial_rings: int = 4,
    angular_samples: int = 16,
    radial_waist_scale: float = 1.0,
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Sample the MOT field around beam axes, grouped by beam handedness."""

    cloud_by_polarization: dict[str, list[tuple[float, float, float, float]]] = {}
    for circular_polarization in ("sigma+", "sigma-", "pi"):
        selected = filter_beams(beams, circular_polarization=circular_polarization)
        cloud_by_polarization[circular_polarization] = sample_intensity_cloud(
            selected,
            axial_extent_m=axial_extent_m,
            axial_samples=axial_samples,
            radial_rings=radial_rings,
            angular_samples=angular_samples,
            radial_waist_scale=radial_waist_scale,
        )
    return cloud_by_polarization


def sample_intensity_slice(
    beams: list[MOTBeam],
    plane: str,
    extent_m: float = 50e-3,
    samples_per_axis: int = 201,
    fixed_coordinate_m: float = 0.0,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Sample the total MOT intensity on a 2D slice."""

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
