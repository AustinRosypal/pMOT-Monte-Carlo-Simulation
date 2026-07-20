"""Beam primitives and geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from math import pi
from math import sqrt
from typing import TypeAlias


Vec3: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GaussianBeam:
    """Scalar Gaussian beam in SI units."""

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

    @property
    def rayleigh_range_m(self) -> float:
        return pi * self.waist_radius_m**2 / self.wavelength_m

    def waist_at(self, axial_m: float) -> float:
        return self.waist_radius_m * sqrt(1.0 + (axial_m / self.rayleigh_range_m) ** 2)

    def intensity(self, radial_m: float = 0.0, axial_m: float = 0.0) -> float:
        if radial_m < 0.0:
            raise ValueError("radial_m must be non-negative")
        local_waist = self.waist_at(axial_m)
        peak_intensity = 2.0 * self.power_w / (pi * local_waist**2)
        return peak_intensity * exp(-2.0 * radial_m**2 / local_waist**2)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def norm(vector: Vec3) -> float:
    return sqrt(dot(vector, vector))


def normalize(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude == 0.0:
        raise ValueError("vector must be non-zero")
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


def axis_direction_from_name(axis_name: str) -> Vec3:
    """Return the modeled 3D direction vector for a named apparatus axis."""

    mapping: dict[str, Vec3] = {
        "oblique_x": normalize((1.0, 0.0, 1.0)),
        "oblique_y": normalize((0.0, 1.0, 1.0)),
        "normal_z": (0.0, 0.0, 1.0),
    }
    if axis_name not in mapping:
        raise ValueError(f"unknown axis name: {axis_name}")
    return mapping[axis_name]


def focused_waist_radius(
    wavelength_m: float,
    focal_length_m: float,
    input_radius_m: float,
) -> float:
    """Return the diffraction-limited waist from a focused collimated Gaussian beam."""

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    if focal_length_m <= 0.0:
        raise ValueError("focal_length_m must be positive")
    if input_radius_m <= 0.0:
        raise ValueError("input_radius_m must be positive")
    return wavelength_m * focal_length_m / (pi * input_radius_m)


def beam_frame_coordinates(
    position_m: Vec3,
    waist_position_m: Vec3,
    direction: Vec3,
) -> tuple[float, float]:
    """Return axial and radial coordinates in the local beam frame."""

    relative = sub(position_m, waist_position_m)
    axial_m = dot(relative, direction)
    radial_vector = sub(relative, scale(axial_m, direction))
    radial_m = norm(radial_vector)
    return axial_m, radial_m
