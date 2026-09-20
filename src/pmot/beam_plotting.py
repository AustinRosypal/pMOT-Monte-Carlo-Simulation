"""Shared beam-geometry helpers for MOT visualizations."""

from __future__ import annotations

import numpy as np


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, trial))) > 0.95:
        trial = np.asarray([0.0, 1.0, 0.0], dtype=float)
    basis_1 = np.cross(direction, trial)
    basis_1 = basis_1 / np.linalg.norm(basis_1)
    basis_2 = np.cross(direction, basis_1)
    basis_2 = basis_2 / np.linalg.norm(basis_2)
    return basis_1, basis_2


def beam_surface_mesh_mm(
    direction: tuple[float, float, float],
    radius_m: float,
    length_m: float,
    axial_samples: int = 25,
    angular_samples: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a cylindrical beam-surface mesh in millimetres."""

    direction_array = np.asarray(direction, dtype=float)
    direction_array = direction_array / np.linalg.norm(direction_array)
    basis_1, basis_2 = _orthonormal_basis(direction_array)
    axial_values_m = np.linspace(-0.5 * length_m, 0.5 * length_m, axial_samples)
    angular_values = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    axial_grid_m, angular_grid = np.meshgrid(
        axial_values_m,
        angular_values,
        indexing="ij",
    )
    radius_grid_m = (
        np.cos(angular_grid)[..., None] * basis_1[None, None, :]
        + np.sin(angular_grid)[..., None] * basis_2[None, None, :]
    ) * radius_m
    centerline_grid_m = axial_grid_m[..., None] * direction_array[None, None, :]
    surface_grid_mm = 1e3 * (centerline_grid_m + radius_grid_m)
    return (
        surface_grid_mm[..., 0],
        surface_grid_mm[..., 1],
        surface_grid_mm[..., 2],
    )


__all__ = ["beam_surface_mesh_mm"]
