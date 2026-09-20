"""Propagation-frame polarization and local spherical-basis projections."""

from __future__ import annotations

from math import sqrt

import numpy as np


Vec3 = tuple[float, float, float]
ComplexVec3 = tuple[complex, complex, complex]


def normalize(vector: Vec3) -> Vec3:
    array = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(array))
    if array.shape != (3,) or not np.all(np.isfinite(array)) or magnitude <= 0.0:
        raise ValueError("a finite nonzero three-vector is required")
    return tuple(float(value) for value in array / magnitude)


def transverse_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    direction = np.asarray(normalize(axis), dtype=float)
    reference = np.asarray(
        (0.0, 1.0, 0.0) if abs(direction[2]) > 0.9 else (0.0, 0.0, 1.0)
    )
    first = np.cross(reference, direction)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return tuple(first), tuple(second)


def spherical_basis(axis: Vec3) -> dict[int, ComplexVec3]:
    first, second = transverse_basis(axis)
    e_plus = tuple((-first[i] - 1j * second[i]) / sqrt(2.0) for i in range(3))
    e_minus = tuple((first[i] - 1j * second[i]) / sqrt(2.0) for i in range(3))
    return {
        -1: e_minus,
        0: tuple(complex(value, 0.0) for value in normalize(axis)),
        +1: e_plus,
    }


def propagation_frame_polarization(direction: Vec3, polarization: str) -> ComplexVec3:
    mapping = {"sigma+": +1, "pi": 0, "sigma-": -1}
    if polarization not in mapping:
        raise ValueError("polarization must be 'sigma+', 'pi', or 'sigma-'")
    return spherical_basis(direction)[mapping[polarization]]


def polarization_amplitudes(
    polarization_vector: ComplexVec3,
    quantization_axis: Vec3,
) -> dict[int, complex]:
    """Return normalized local amplitudes epsilon_q for q=-1,0,+1."""

    epsilon = np.asarray(polarization_vector, dtype=complex)
    norm = float(np.linalg.norm(epsilon))
    if epsilon.shape != (3,) or not np.isfinite(epsilon).all() or norm <= 0.0:
        raise ValueError("polarization vector must be a finite nonzero three-vector")
    epsilon /= norm
    basis = spherical_basis(quantization_axis)
    amplitudes = {
        q: complex(np.vdot(np.asarray(basis[q], dtype=complex), epsilon))
        for q in (-1, 0, +1)
    }
    normalization = sqrt(sum(abs(value) ** 2 for value in amplitudes.values()))
    return {q: value / normalization for q, value in amplitudes.items()}


def polarization_weights(
    polarization_vector: ComplexVec3,
    quantization_axis: Vec3,
) -> dict[int, float]:
    return {
        q: float(abs(value) ** 2)
        for q, value in polarization_amplitudes(
            polarization_vector,
            quantization_axis,
        ).items()
    }


def quantization_axis(field_t: Vec3, previous_axis: Vec3, epsilon_t: float) -> Vec3:
    if float(np.linalg.norm(np.asarray(field_t, dtype=float))) <= epsilon_t:
        return normalize(previous_axis)
    return normalize(field_t)


__all__ = [
    "ComplexVec3",
    "Vec3",
    "normalize",
    "polarization_amplitudes",
    "polarization_weights",
    "propagation_frame_polarization",
    "quantization_axis",
    "spherical_basis",
    "transverse_basis",
]
