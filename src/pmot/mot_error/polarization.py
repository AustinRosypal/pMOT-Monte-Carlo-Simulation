"""Complex polarization vectors and local spherical-basis projections."""

from __future__ import annotations

from math import sqrt

import numpy as np


Vec3 = tuple[float, float, float]
ComplexVec3 = tuple[complex, complex, complex]


def normalize(vector: Vec3) -> Vec3:
    array = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(array))
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple((array / magnitude).tolist())


def transverse_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    """Return a deterministic right-handed basis perpendicular to axis."""

    direction = np.asarray(normalize(axis), dtype=float)
    reference = np.asarray((0.0, 1.0, 0.0) if abs(direction[2]) > 0.9 else (0.0, 0.0, 1.0))
    first = np.cross(reference, direction)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return tuple(first.tolist()), tuple(second.tolist())


def spherical_basis(axis: Vec3) -> dict[int, ComplexVec3]:
    """Return e_q for q=-1,0,+1 relative to the supplied quantization axis."""

    first, second = transverse_basis(axis)
    e_plus = tuple((-first[i] - 1j * second[i]) / sqrt(2.0) for i in range(3))
    e_minus = tuple((first[i] - 1j * second[i]) / sqrt(2.0) for i in range(3))
    return {
        -1: e_minus,
        0: tuple(complex(value, 0.0) for value in normalize(axis)),
        +1: e_plus,
    }


def propagation_frame_polarization(direction: Vec3, polarization: str) -> ComplexVec3:
    """Construct pi/sigma polarization defined while looking along propagation."""

    basis = spherical_basis(direction)
    mapping = {"sigma+": +1, "pi": 0, "sigma-": -1}
    if polarization not in mapping:
        raise ValueError("polarization must be 'sigma+', 'pi', or 'sigma-'")
    return basis[mapping[polarization]]


def polarization_weights(polarization_vector: ComplexVec3, quantization_axis: Vec3) -> dict[int, float]:
    """Project a lab-frame polarization vector onto a local spherical basis."""

    basis = spherical_basis(quantization_axis)
    epsilon = np.asarray(polarization_vector, dtype=complex)
    weights = {q: float(abs(np.vdot(np.asarray(basis[q]), epsilon)) ** 2) for q in (-1, 0, +1)}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("polarization projection has zero norm")
    return {q: value / total for q, value in weights.items()}


def quantization_axis(field_t: Vec3, previous_axis: Vec3, epsilon_t: float) -> Vec3:
    """Return B-hat or retain the previous well-defined axis near B=0."""

    if float(np.linalg.norm(np.asarray(field_t, dtype=float))) <= epsilon_t:
        return normalize(previous_axis)
    return normalize(field_t)
