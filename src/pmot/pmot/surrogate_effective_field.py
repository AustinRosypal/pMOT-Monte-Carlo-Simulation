"""Six-beam surrogate effective magnetic field for the pMOT geometry.

This module implements ``PMOT_EFFECTIVE_FIELD_CODEX.md`` literally.  It maps
each 1530-nm traveling component's intensity to a stretched-transition-
equivalent magnetic-field vector and only then sums the six vectors.  The
result is a proof-of-principle field for use by the physical Section-12 MOT
solver; it is not a state-resolved AC-Stark Hamiltonian.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


PLANCK_CONSTANT_J_S = 6.62607015e-34
BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
DIFFERENTIAL_VECTOR_POLARIZABILITY_HZ_PER_W_PER_M2 = 1363.34


@dataclass(frozen=True, slots=True)
class SurrogateEffectiveFieldConfig:
    """Analytic beam and stretched-transition mapping parameters in SI units."""

    wavelength_m: float = 1.530e-6
    forward_waist_position_m: float = -1.00326971e-2
    # Center-balanced pair chosen on 2026-10-02.  The waist locations remain
    # those of the solved optical paths.  The pair preserves the geometric
    # mean of the former Rayleigh ranges while enforcing equal on-axis
    # intensity at the origin for equal forward/return powers.
    forward_rayleigh_range_m: float = 9.103072091279295e-6
    return_waist_position_m: float = 9.97601576e-3
    return_rayleigh_range_m: float = 9.000504018534303e-6
    x_path_power_w: float = 0.0108
    y_path_power_w: float = 0.0108
    z_path_power_w: float = 0.0216
    helicity_positive_x: float = 1.0
    helicity_negative_x: float = 1.0
    helicity_positive_y: float = 1.0
    helicity_negative_y: float = 1.0
    helicity_positive_z: float = -1.0
    helicity_negative_z: float = -1.0
    cell_half_length_m: float = 1.41421356e-2
    transition_zeeman_factor: float = 1.0
    differential_vector_polarizability_hz_per_w_per_m2: float = (
        DIFFERENTIAL_VECTOR_POLARIZABILITY_HZ_PER_W_PER_M2
    )

    def __post_init__(self) -> None:
        positive = (
            self.wavelength_m,
            self.forward_rayleigh_range_m,
            self.return_rayleigh_range_m,
            self.cell_half_length_m,
            self.transition_zeeman_factor,
            self.differential_vector_polarizability_hz_per_w_per_m2,
        )
        if any(value <= 0.0 or not np.isfinite(value) for value in positive):
            raise ValueError("wavelength, ranges, cell size, G, and polarizability must be positive")
        powers = (self.x_path_power_w, self.y_path_power_w, self.z_path_power_w)
        if any(value < 0.0 or not np.isfinite(value) for value in powers):
            raise ValueError("path powers must be finite and non-negative")

    @property
    def kappa_t_per_w_per_m2(self) -> float:
        return float(
            -2.0
            * PLANCK_CONSTANT_J_S
            * self.differential_vector_polarizability_hz_per_w_per_m2
            / (BOHR_MAGNETON_J_PER_T * self.transition_zeeman_factor)
        )

    def metadata(self) -> dict[str, float]:
        return {**asdict(self), "kappa_t_per_w_per_m2": self.kappa_t_per_w_per_m2}


DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG = SurrogateEffectiveFieldConfig()

# Retain the exact earlier specification so the September 2026 asymmetric
# diagnostic can still be reproduced explicitly after the centered pair became
# the default.
LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG = SurrogateEffectiveFieldConfig(
    forward_rayleigh_range_m=1.02540763e-5,
    return_rayleigh_range_m=7.99021136e-6,
)


def _position_array(positions_m) -> tuple[np.ndarray, bool]:
    positions = np.asarray(positions_m, dtype=float)
    scalar_input = positions.shape == (3,)
    if scalar_input:
        positions = positions[np.newaxis, :]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (3,) or (n, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    return positions, scalar_input


def inside_surrogate_cell(
    positions_m,
    config: SurrogateEffectiveFieldConfig | None = None,
):
    """Return whether each point lies in the closed cubic vapor domain."""

    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    positions, scalar_input = _position_array(positions_m)
    inside = np.all(np.abs(positions) <= cfg.cell_half_length_m, axis=1)
    return bool(inside[0]) if scalar_input else inside


def _gaussian_intensity_w_per_m2(
    power_w: float,
    longitudinal_coordinate_m: np.ndarray,
    transverse_radius_squared_m2: np.ndarray,
    waist_position_m: float,
    rayleigh_range_m: float,
    wavelength_m: float,
) -> np.ndarray:
    denominator_m2 = (
        longitudinal_coordinate_m - waist_position_m
    ) ** 2 + rayleigh_range_m**2
    return (
        2.0 * power_w * rayleigh_range_m / (wavelength_m * denominator_m2)
        * np.exp(
            -2.0
            * np.pi
            * rayleigh_range_m
            * transverse_radius_squared_m2
            / (wavelength_m * denominator_m2)
        )
    )


def surrogate_component_intensities_w_per_m2(
    positions_m,
    config: SurrogateEffectiveFieldConfig | None = None,
) -> np.ndarray:
    """Return ``(+x,-x,+y,-y,+z,-z)`` intensities in W/m^2.

    The returned shape is ``(6,)`` for one position and ``(n, 6)`` for a
    position array.  Every component is exactly zero outside the vapor cell.
    """

    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    positions, scalar_input = _position_array(positions_m)
    x, y, z = positions.T
    rho_x2 = y * y + z * z
    rho_y2 = x * x + z * z
    rho_z2 = x * x + y * y

    def forward(power_w: float, coordinate: np.ndarray, rho2: np.ndarray) -> np.ndarray:
        return _gaussian_intensity_w_per_m2(
            power_w,
            coordinate,
            rho2,
            cfg.forward_waist_position_m,
            cfg.forward_rayleigh_range_m,
            cfg.wavelength_m,
        )

    def returned(power_w: float, coordinate: np.ndarray, rho2: np.ndarray) -> np.ndarray:
        return _gaussian_intensity_w_per_m2(
            power_w,
            coordinate,
            rho2,
            cfg.return_waist_position_m,
            cfg.return_rayleigh_range_m,
            cfg.wavelength_m,
        )

    intensities = np.column_stack(
        (
            forward(cfg.x_path_power_w, x, rho_x2),
            returned(cfg.x_path_power_w, x, rho_x2),
            forward(cfg.y_path_power_w, y, rho_y2),
            returned(cfg.y_path_power_w, y, rho_y2),
            forward(cfg.z_path_power_w, z, rho_z2),
            returned(cfg.z_path_power_w, z, rho_z2),
        )
    )
    intensities *= inside_surrogate_cell(positions, cfg)[:, np.newaxis]
    return intensities[0] if scalar_input else intensities


def surrogate_effective_field_t(
    positions_m,
    config: SurrogateEffectiveFieldConfig | None = None,
) -> np.ndarray:
    """Return the vector-summed stretched-transition-equivalent field [T]."""

    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    positions, scalar_input = _position_array(positions_m)
    intensities = surrogate_component_intensities_w_per_m2(positions, cfg)
    beam_directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    helicities = np.asarray(
        (
            cfg.helicity_positive_x,
            cfg.helicity_negative_x,
            cfg.helicity_positive_y,
            cfg.helicity_negative_y,
            cfg.helicity_positive_z,
            cfg.helicity_negative_z,
        )
    )
    field = cfg.kappa_t_per_w_per_m2 * (
        intensities[:, :, np.newaxis]
        * helicities[np.newaxis, :, np.newaxis]
        * beam_directions[np.newaxis, :, :]
    ).sum(axis=1)
    return field[0] if scalar_input else field


def pmot_effective_B(position_m: np.ndarray) -> np.ndarray:
    """Specification-facing scalar callback returning ``[Bx, By, Bz]`` [T]."""

    return surrogate_effective_field_t(position_m)


def numerical_field_jacobian_t_per_m(
    position_m=(0.0, 0.0, 0.0),
    *,
    step_m: float = 1.0e-6,
    config: SurrogateEffectiveFieldConfig | None = None,
) -> np.ndarray:
    """Central-difference Jacobian ``dB_i/dx_j`` in T/m."""

    if step_m <= 0.0 or not np.isfinite(step_m):
        raise ValueError("step_m must be finite and positive")
    center = np.asarray(position_m, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("position_m must be a finite three-vector")
    return surrogate_field_jacobians_t_per_m(center, step_m=step_m, config=config)


def surrogate_field_jacobians_t_per_m(
    positions_m,
    *,
    step_m: float = 1.0e-6,
    config: SurrogateEffectiveFieldConfig | None = None,
) -> np.ndarray:
    """Return local Jacobians ``dB_i/dx_j`` for one or many positions."""

    if step_m <= 0.0 or not np.isfinite(step_m):
        raise ValueError("step_m must be finite and positive")
    positions, scalar_input = _position_array(positions_m)
    jacobians = np.empty((len(positions), 3, 3))
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = step_m
        jacobians[:, :, coordinate] = (
            surrogate_effective_field_t(positions + displacement, config)
            - surrogate_effective_field_t(positions - displacement, config)
        ) / (2.0 * step_m)
    return jacobians[0] if scalar_input else jacobians


__all__ = [
    "BOHR_MAGNETON_J_PER_T",
    "DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG",
    "DIFFERENTIAL_VECTOR_POLARIZABILITY_HZ_PER_W_PER_M2",
    "LEGACY_ASYMMETRIC_SURROGATE_EFFECTIVE_FIELD_CONFIG",
    "PLANCK_CONSTANT_J_S",
    "SurrogateEffectiveFieldConfig",
    "inside_surrogate_cell",
    "numerical_field_jacobian_t_per_m",
    "pmot_effective_B",
    "surrogate_component_intensities_w_per_m2",
    "surrogate_effective_field_t",
    "surrogate_field_jacobians_t_per_m",
]
