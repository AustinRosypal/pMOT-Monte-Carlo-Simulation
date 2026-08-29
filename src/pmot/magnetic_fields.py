"""Shared closed-form anti-Helmholtz magnetic fields for MOT validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.special import ellipe
from scipy.special import ellipk

from .configuration import AntiHelmholtzCoilConfig
from .configuration import GAUSS_PER_TESLA
from .configuration import TESLA_PER_METER_PER_GAUSS_PER_CM
from .configuration import VACUUM_PERMEABILITY_H_PER_M


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MagneticFieldSample:
    """One Cartesian magnetic-field sample."""

    x_m: float
    y_m: float
    z_m: float
    bx_t: float
    by_t: float
    bz_t: float

    @property
    def magnitude_t(self) -> float:
        return sqrt(self.bx_t**2 + self.by_t**2 + self.bz_t**2)


def anti_helmholtz_axial_gradient_t_per_m(
    radius_m: float,
    turns_per_coil: int,
    current_a: float,
) -> float:
    """Return the on-axis gradient magnitude at the anti-Helmholtz center."""

    prefactor = 48.0 / (5.0 ** 2.5)
    return prefactor * VACUUM_PERMEABILITY_H_PER_M * turns_per_coil * current_a / (radius_m**2)


def current_for_target_axial_gradient_a(
    target_gradient_t_per_m: float,
    radius_m: float,
    turns_per_coil: int,
) -> float:
    """Return the coil current needed to hit the target central axial gradient."""

    if target_gradient_t_per_m <= 0.0:
        raise ValueError("target_gradient_t_per_m must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if turns_per_coil <= 0:
        raise ValueError("turns_per_coil must be positive")
    prefactor = 48.0 / (5.0 ** 2.5)
    return target_gradient_t_per_m * radius_m**2 / (prefactor * VACUUM_PERMEABILITY_H_PER_M * turns_per_coil)


def default_anti_helmholtz_config(
    radius_m: float = 40.0e-3,
    turns_per_coil: int = 50,
    target_gradient_g_per_cm: float = 10.0,
) -> AntiHelmholtzCoilConfig:
    """Return a reasonable default coil pair close to 10 G/cm at the center."""

    current_a = current_for_target_axial_gradient_a(
        target_gradient_t_per_m=target_gradient_g_per_cm * TESLA_PER_METER_PER_GAUSS_PER_CM,
        radius_m=radius_m,
        turns_per_coil=turns_per_coil,
    )
    return AntiHelmholtzCoilConfig(
        radius_m=radius_m,
        turns_per_coil=turns_per_coil,
        current_a=current_a,
        center_separation_m=radius_m,
    )


def _coil_singularity_mask(rho_m, z_m, config: AntiHelmholtzCoilConfig, atol: float = 1.0e-15):
    """Return a mask for points exactly on the wire loops."""

    rho_array = np.asarray(rho_m, dtype=float)
    z_array = np.asarray(z_m, dtype=float)
    return np.isclose(rho_array, config.radius_m, atol=atol, rtol=0.0) & (
        np.isclose(z_array, -config.half_separation_m, atol=atol, rtol=0.0)
        | np.isclose(z_array, config.half_separation_m, atol=atol, rtol=0.0)
    )


def anti_helmholtz_cylindrical_field_t(
    rho_m,
    z_m,
    config: AntiHelmholtzCoilConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the anti-Helmholtz field in cylindrical coordinates."""

    rho_array, z_array = np.broadcast_arrays(
        np.asarray(rho_m, dtype=float),
        np.asarray(z_m, dtype=float),
    )
    brho = np.zeros_like(rho_array, dtype=float)
    bz = np.zeros_like(z_array, dtype=float)

    singular_mask = _coil_singularity_mask(rho_array, z_array, config)
    safe_mask = ~singular_mask
    if np.any(safe_mask):
        rho = rho_array[safe_mask]
        z = z_array[safe_mask]
        radius = config.radius_m
        half_separation = config.half_separation_m
        prefactor = VACUUM_PERMEABILITY_H_PER_M * config.turns_per_coil * config.current_a / (2.0 * np.pi)

        alpha_plus_sq = (radius - rho) ** 2 + (z + half_separation) ** 2
        alpha_minus_sq = (radius - rho) ** 2 + (z - half_separation) ** 2
        beta_plus_sq = (radius + rho) ** 2 + (z + half_separation) ** 2
        beta_minus_sq = (radius + rho) ** 2 + (z - half_separation) ** 2
        beta_plus = np.sqrt(beta_plus_sq)
        beta_minus = np.sqrt(beta_minus_sq)
        m_plus = 4.0 * radius * rho / beta_plus_sq
        m_minus = 4.0 * radius * rho / beta_minus_sq

        k_plus_term = ellipk(m_plus)
        k_minus_term = ellipk(m_minus)
        e_plus_term = ellipe(m_plus)
        e_minus_term = ellipe(m_minus)

        rho_axis_mask = np.isclose(rho, 0.0, atol=1.0e-18, rtol=0.0)
        off_axis_mask = ~rho_axis_mask

        bz_safe = prefactor * (
            (1.0 / beta_plus)
            * (k_plus_term + (radius**2 - rho**2 - (z + half_separation) ** 2) / alpha_plus_sq * e_plus_term)
            - (1.0 / beta_minus)
            * (k_minus_term + (radius**2 - rho**2 - (z - half_separation) ** 2) / alpha_minus_sq * e_minus_term)
        )
        bz[safe_mask] = bz_safe

        if np.any(off_axis_mask):
            rho_off = rho[off_axis_mask]
            z_off = z[off_axis_mask]
            brho_off = prefactor / rho_off * (
                ((z_off + half_separation) / beta_plus[off_axis_mask])
                * (
                    -k_plus_term[off_axis_mask]
                    + (
                        radius**2 + rho_off**2 + (z_off + half_separation) ** 2
                    )
                    / alpha_plus_sq[off_axis_mask]
                    * e_plus_term[off_axis_mask]
                )
                - ((z_off - half_separation) / beta_minus[off_axis_mask])
                * (
                    -k_minus_term[off_axis_mask]
                    + (
                        radius**2 + rho_off**2 + (z_off - half_separation) ** 2
                    )
                    / alpha_minus_sq[off_axis_mask]
                    * e_minus_term[off_axis_mask]
                )
            )
            brho_safe = np.zeros_like(rho, dtype=float)
            brho_safe[off_axis_mask] = brho_off
            brho[safe_mask] = brho_safe

        if np.any(rho_axis_mask):
            z_axis = z[rho_axis_mask]
            bz_axis = (
                VACUUM_PERMEABILITY_H_PER_M
                * config.turns_per_coil
                * config.current_a
                * radius**2
                / 2.0
                * (
                    1.0 / (radius**2 + (z_axis + half_separation) ** 2) ** 1.5
                    - 1.0 / (radius**2 + (z_axis - half_separation) ** 2) ** 1.5
                )
            )
            bz_safe[rho_axis_mask] = bz_axis
            bz[safe_mask] = bz_safe

    brho[singular_mask] = np.nan
    bz[singular_mask] = np.nan
    return brho, bz


def anti_helmholtz_field_t(
    x_m,
    y_m,
    z_m,
    config: AntiHelmholtzCoilConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the anti-Helmholtz field in Cartesian coordinates."""

    x_array, y_array, z_array = np.broadcast_arrays(
        np.asarray(x_m, dtype=float),
        np.asarray(y_m, dtype=float),
        np.asarray(z_m, dtype=float),
    )
    rho = np.sqrt(x_array**2 + y_array**2)
    brho, bz = anti_helmholtz_cylindrical_field_t(rho, z_array, config)

    bx = np.zeros_like(x_array, dtype=float)
    by = np.zeros_like(y_array, dtype=float)
    on_axis_mask = np.isclose(rho, 0.0, atol=1.0e-18, rtol=0.0)
    off_axis_mask = ~on_axis_mask
    bx[off_axis_mask] = brho[off_axis_mask] * x_array[off_axis_mask] / rho[off_axis_mask]
    by[off_axis_mask] = brho[off_axis_mask] * y_array[off_axis_mask] / rho[off_axis_mask]
    bx[on_axis_mask] = 0.0
    by[on_axis_mask] = 0.0
    return bx, by, bz


def field_sample(
    x_m: float,
    y_m: float,
    z_m: float,
    config: AntiHelmholtzCoilConfig,
) -> MagneticFieldSample:
    """Return one field sample at one Cartesian point."""

    bx, by, bz = anti_helmholtz_field_t(x_m, y_m, z_m, config)
    return MagneticFieldSample(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        bx_t=float(np.asarray(bx)),
        by_t=float(np.asarray(by)),
        bz_t=float(np.asarray(bz)),
    )


def line_sample(
    start_m: Vec3,
    stop_m: Vec3,
    config: AntiHelmholtzCoilConfig,
    sample_count: int = 401,
) -> dict[str, list[float]]:
    """Sample the field along a Cartesian line segment."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    fractions = np.linspace(0.0, 1.0, sample_count)
    x_values = start_m[0] + fractions * (stop_m[0] - start_m[0])
    y_values = start_m[1] + fractions * (stop_m[1] - start_m[1])
    z_values = start_m[2] + fractions * (stop_m[2] - start_m[2])
    bx, by, bz = anti_helmholtz_field_t(x_values, y_values, z_values, config)
    distances_mm = 1e3 * (
        fractions - 0.5
    ) * sqrt((stop_m[0] - start_m[0]) ** 2 + (stop_m[1] - start_m[1]) ** 2 + (stop_m[2] - start_m[2]) ** 2)
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    return {
        "distance_mm": distances_mm.tolist(),
        "bx_g": (GAUSS_PER_TESLA * bx).tolist(),
        "by_g": (GAUSS_PER_TESLA * by).tolist(),
        "bz_g": (GAUSS_PER_TESLA * bz).tolist(),
        "bmag_g": (GAUSS_PER_TESLA * magnitude).tolist(),
    }


def plane_sample(
    plane: str,
    extent_m: float,
    config: AntiHelmholtzCoilConfig,
    samples_per_axis: int = 201,
    fixed_coordinate_m: float = 0.0,
) -> dict[str, list[list[float]] | list[float]]:
    """Sample the magnetic field on a Cartesian plane."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")
    axis_values = np.linspace(-extent_m, extent_m, samples_per_axis)
    first_mesh, second_mesh = np.meshgrid(axis_values, axis_values)
    if plane == "xy":
        bx, by, bz = anti_helmholtz_field_t(first_mesh, second_mesh, fixed_coordinate_m, config)
        component_1 = bx
        component_2 = by
    elif plane == "xz":
        bx, by, bz = anti_helmholtz_field_t(first_mesh, fixed_coordinate_m, second_mesh, config)
        component_1 = bx
        component_2 = bz
    else:
        bx, by, bz = anti_helmholtz_field_t(fixed_coordinate_m, first_mesh, second_mesh, config)
        component_1 = by
        component_2 = bz
    magnitude = np.sqrt(bx**2 + by**2 + bz**2)
    return {
        "axis_values_m": axis_values.tolist(),
        "component_1_g": (GAUSS_PER_TESLA * component_1).tolist(),
        "component_2_g": (GAUSS_PER_TESLA * component_2).tolist(),
        "bz_g": (GAUSS_PER_TESLA * bz).tolist(),
        "bmag_g": (GAUSS_PER_TESLA * magnitude).tolist(),
    }


def vector_cloud_sample(
    extent_m: float,
    config: AntiHelmholtzCoilConfig,
    samples_per_axis: int = 9,
) -> list[tuple[float, float, float, float, float, float]]:
    """Sample a coarse 3D magnetic vector cloud."""

    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    axis_values = np.linspace(-extent_m, extent_m, samples_per_axis)
    cloud: list[tuple[float, float, float, float, float, float]] = []
    for x_value in axis_values:
        for y_value in axis_values:
            for z_value in axis_values:
                bx, by, bz = anti_helmholtz_field_t(x_value, y_value, z_value, config)
                if np.any(np.isnan([bx, by, bz])):
                    continue
                cloud.append(
                    (
                        x_value,
                        y_value,
                        z_value,
                        float(np.asarray(bx)),
                        float(np.asarray(by)),
                        float(np.asarray(bz)),
                    )
                )
    return cloud
