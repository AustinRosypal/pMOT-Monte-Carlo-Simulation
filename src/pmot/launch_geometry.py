"""Model-neutral launch-disc geometry for MOT capture calculations.

This module defines only geometry and random sampling.  It contains no force
law or trajectory integrator, so both the simplified and multilevel MOT
packages can use it without depending on one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DiscSample:
    """One sampled incident disc."""

    disc_index: int
    theta_rad: float
    phi_rad: float
    outward_unit_vector: Vec3
    incident_unit_vector: Vec3
    center_position_m: Vec3
    basis_u: Vec3
    basis_v: Vec3


@dataclass(frozen=True, slots=True)
class PointSample:
    """One sampled launch point on one incident disc."""

    disc_index: int
    point_index: int
    theta_rad: float
    phi_rad: float
    theta_prime_rad: float
    s_m: float
    radial_distance_m: float
    initial_position_m: Vec3
    incident_unit_vector: Vec3
    launch_axis_unit_vector: Vec3


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(value: float, vector: Vec3) -> Vec3:
    return (value * vector[0], value * vector[1], value * vector[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(vector: Vec3) -> float:
    return float(np.sqrt(dot(vector, vector)))


def normalize(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return scale(1.0 / magnitude, vector)


def choose_transverse_basis(direction: Vec3) -> tuple[Vec3, Vec3]:
    """Return orthonormal vectors spanning the plane perpendicular to direction."""

    direction_hat = normalize(direction)
    trial = (0.0, 0.0, 1.0)
    if abs(dot(direction_hat, trial)) > 0.95:
        trial = (0.0, 1.0, 0.0)
    basis_u = normalize(cross(direction_hat, trial))
    basis_v = normalize(cross(direction_hat, basis_u))
    return basis_u, basis_v


def spherical_direction(theta_rad: float, phi_rad: float) -> Vec3:
    """Return a unit vector from polar and azimuthal angles."""

    return (
        sin(theta_rad) * cos(phi_rad),
        sin(theta_rad) * sin(phi_rad),
        cos(theta_rad),
    )


# Compatibility name retained for existing notebooks and APIs.  The formula
# itself is valid over the full angular domain.
spherical_octant_direction = spherical_direction


def sample_octant_direction(rng: np.random.Generator) -> tuple[float, float, Vec3]:
    """Sample one first-octant direction uniformly in solid angle."""

    cos_theta = float(rng.uniform(0.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    phi_rad = float(rng.uniform(0.0, 0.5 * pi))
    return theta_rad, phi_rad, spherical_direction(theta_rad, phi_rad)


def sample_full_sphere_direction(rng: np.random.Generator) -> tuple[float, float, Vec3]:
    """Sample one direction uniformly over the complete solid angle."""

    cos_theta = float(rng.uniform(-1.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    phi_rad = float(rng.uniform(0.0, 2.0 * pi))
    return theta_rad, phi_rad, spherical_direction(theta_rad, phi_rad)


def _disc_from_direction(
    disc_index: int,
    radial_distance_m: float,
    theta_rad: float,
    phi_rad: float,
    outward_unit_vector: Vec3,
) -> DiscSample:
    incident_unit_vector = scale(-1.0, outward_unit_vector)
    basis_u, basis_v = choose_transverse_basis(incident_unit_vector)
    center_position_m = scale(radial_distance_m, outward_unit_vector)
    return DiscSample(
        disc_index=disc_index,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        outward_unit_vector=outward_unit_vector,
        incident_unit_vector=incident_unit_vector,
        center_position_m=center_position_m,
        basis_u=basis_u,
        basis_v=basis_v,
    )


def sample_incident_disc(
    disc_index: int,
    radial_distance_m: float,
    rng: np.random.Generator,
) -> DiscSample:
    """Sample one incident disc within the first octant."""

    theta_rad, phi_rad, outward_unit_vector = sample_octant_direction(rng)
    return _disc_from_direction(
        disc_index, radial_distance_m, theta_rad, phi_rad, outward_unit_vector
    )


def sample_incident_disc_full_sphere(
    disc_index: int,
    radial_distance_m: float,
    rng: np.random.Generator,
) -> DiscSample:
    """Sample one incident disc with its normal uniform over 4 pi steradians."""

    theta_rad, phi_rad, outward_unit_vector = sample_full_sphere_direction(rng)
    return _disc_from_direction(
        disc_index, radial_distance_m, theta_rad, phi_rad, outward_unit_vector
    )


def build_incident_disc_from_angles(
    disc_index: int,
    radial_distance_m: float,
    theta_rad: float,
    phi_rad: float,
) -> DiscSample:
    """Build one incident disc from user-specified spherical angles."""

    outward_unit_vector = spherical_direction(theta_rad, phi_rad)
    return _disc_from_direction(
        disc_index, radial_distance_m, theta_rad, phi_rad, outward_unit_vector
    )


def sample_disc_points(
    disc: DiscSample,
    points_per_disc: int,
    disc_radius_m: float,
    include_center_point: bool,
    rng: np.random.Generator,
) -> list[PointSample]:
    """Sample independent, uniformly distributed launch points over a disc.

    Production Monte Carlo uses ``include_center_point=False`` so every point
    has ``s = R * sqrt(U)`` and an independent uniform azimuth.  The explicit
    ``include_center_point=True`` opt-in is retained for legacy diagnostics;
    only that mode prepends the measure-zero center.  The rim is never forced.
    """

    points: list[PointSample] = []
    if points_per_disc <= 0:
        return points

    center_position_m = disc.center_position_m
    launch_axis_unit_vector = normalize(scale(-1.0, center_position_m))
    if include_center_point:
        points.append(
            PointSample(
                disc_index=disc.disc_index,
                point_index=0,
                theta_rad=disc.theta_rad,
                phi_rad=disc.phi_rad,
                theta_prime_rad=0.0,
                s_m=0.0,
                radial_distance_m=radial_distance_magnitude(disc.center_position_m),
                initial_position_m=center_position_m,
                incident_unit_vector=launch_axis_unit_vector,
                launch_axis_unit_vector=launch_axis_unit_vector,
            )
        )

    point_start_index = int(include_center_point)
    for point_index in range(point_start_index, points_per_disc):
        theta_prime_rad = float(rng.uniform(0.0, 2.0 * pi))
        area_fraction = float(rng.uniform(0.0, 1.0))
        s_m = disc_radius_m * float(np.sqrt(area_fraction))
        offset = add(
            scale(s_m * float(np.cos(theta_prime_rad)), disc.basis_u),
            scale(s_m * float(np.sin(theta_prime_rad)), disc.basis_v),
        )
        initial_position_m = add(disc.center_position_m, offset)
        points.append(
            PointSample(
                disc_index=disc.disc_index,
                point_index=point_index,
                theta_rad=disc.theta_rad,
                phi_rad=disc.phi_rad,
                theta_prime_rad=theta_prime_rad,
                s_m=s_m,
                radial_distance_m=radial_distance_magnitude(disc.center_position_m),
                initial_position_m=initial_position_m,
                incident_unit_vector=launch_axis_unit_vector,
                launch_axis_unit_vector=launch_axis_unit_vector,
            )
        )
    return points


def radial_distance_magnitude(position_m: Vec3) -> float:
    """Return the distance of one point from the origin."""

    return norm(position_m)


__all__ = [
    "DiscSample",
    "PointSample",
    "Vec3",
    "add",
    "build_incident_disc_from_angles",
    "choose_transverse_basis",
    "cross",
    "dot",
    "norm",
    "normalize",
    "radial_distance_magnitude",
    "sample_disc_points",
    "sample_full_sphere_direction",
    "sample_incident_disc",
    "sample_incident_disc_full_sphere",
    "sample_octant_direction",
    "scale",
    "spherical_direction",
    "spherical_octant_direction",
]
