"""Zeeman-state and polarization utilities for the MOT validation model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

import numpy as np
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_6j

from ..configuration import HBAR_J_S
from ..configuration import PLANCK_CONSTANT_J_S
from ..beams import axis_direction_from_name
from ..fields import MOTBeam


Vec3 = tuple[float, float, float]
ComplexVec3 = tuple[complex, complex, complex]

BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
BOHR_MAGNETON_OVER_H_HZ_PER_T = BOHR_MAGNETON_J_PER_T / PLANCK_CONSTANT_J_S
BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T = BOHR_MAGNETON_J_PER_T / HBAR_J_S

RB87_NUCLEAR_SPIN = 1.5
GROUND_J = 0.5
EXCITED_J = 1.5
EXCITED_G_FACTOR_BY_F = {
    1: 2.0 / 3.0,
    2: 2.0 / 3.0,
    3: 2.0 / 3.0,
}
GROUND_G_FACTOR_BY_F = {
    1: -0.5,
    2: 0.5,
}
ZERO_FIELD_TRANSITION_FREQUENCIES_HZ = {
    (2, 3): 384.228115203271e12,
    (2, 2): 384.227848551271e12,
    (2, 1): 384.227691610771e12,
    (1, 1): 384.234526293382e12,
    (1, 2): 384.234683233882e12,
}


@dataclass(frozen=True, slots=True)
class GroundState:
    """One Rb-87 ground hyperfine Zeeman state."""

    f: int
    m_f: int

    def __post_init__(self) -> None:
        if self.f not in {1, 2}:
            raise ValueError("ground-state F must be 1 or 2")
        if self.m_f < -self.f or self.m_f > self.f:
            raise ValueError("m_f must lie within [-F, +F]")


@dataclass(frozen=True, slots=True)
class ExcitedState:
    """One addressed 5p3/2 excited hyperfine Zeeman state."""

    f_prime: int
    m_f_prime: int

    def __post_init__(self) -> None:
        if self.f_prime not in {1, 2, 3}:
            raise ValueError("f_prime must be one of 1, 2, 3 in the current model")
        if self.m_f_prime < -self.f_prime or self.m_f_prime > self.f_prime:
            raise ValueError("m_f_prime must lie within [-F', +F']")


def norm(vector: Vec3) -> float:
    return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(vector: Vec3) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


def choose_transverse_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    """Return a stable orthonormal basis transverse to one axis."""

    direction = normalize(axis)
    reference = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (1.0, 0.0, 0.0)
    first = normalize(cross(reference, direction))
    second = normalize(cross(direction, first))
    return first, second


def local_spherical_basis(axis: Vec3) -> dict[int, ComplexVec3]:
    """Return the q = +1, 0, -1 spherical basis vectors for one axis."""

    first, second = choose_transverse_basis(axis)
    sigma_plus = tuple((-first[index] - 1j * second[index]) / sqrt(2.0) for index in range(3))
    sigma_minus = tuple((first[index] - 1j * second[index]) / sqrt(2.0) for index in range(3))
    pi_component = tuple(complex(value, 0.0) for value in normalize(axis))
    return {
        +1: sigma_plus,
        0: pi_component,
        -1: sigma_minus,
    }


def beam_polarization_vector(beam: MOTBeam) -> ComplexVec3:
    """Return the beam polarization vector in laboratory Cartesian coordinates."""

    axis_reference_direction = axis_direction_from_name(beam.axis_name)
    basis = local_spherical_basis(axis_reference_direction)
    # The local spherical basis depends on the transverse lab-frame basis
    # chosen for each apparatus axis. With the current basis construction,
    # the horizontal axes and the vertical axis carry opposite effective
    # handedness relative to the user-facing right/left labels.
    if beam.axis_name in {"horizontal_x", "horizontal_y"}:
        if beam.circular_polarization == "right":
            return basis[+1]
        if beam.circular_polarization == "left":
            return basis[-1]
    else:
        if beam.circular_polarization == "right":
            return basis[-1]
        if beam.circular_polarization == "left":
            return basis[+1]
    raise ValueError(f"unsupported beam circular polarization: {beam.circular_polarization}")


def complex_dot_conjugate(left: ComplexVec3, right: ComplexVec3) -> complex:
    return sum(np.conjugate(left[index]) * right[index] for index in range(3))


def polarization_weights_for_quantization_axis(
    beam: MOTBeam,
    quantization_axis: Vec3,
) -> dict[int, float]:
    """Project one beam polarization onto the local spherical basis."""

    epsilon = beam_polarization_vector(beam)
    local_basis = local_spherical_basis(quantization_axis)
    weights: dict[int, float] = {}
    for q_value in (-1, 0, +1):
        overlap = complex_dot_conjugate(local_basis[q_value], epsilon)
        weights[q_value] = float(abs(overlap) ** 2)
    normalization = sum(weights.values())
    if normalization <= 0.0:
        raise ValueError("polarization projection produced zero total weight")
    return {q_value: weight / normalization for q_value, weight in weights.items()}


def zero_field_polarization_weights() -> dict[int, float]:
    """Return an isotropic surrogate polarization decomposition at B = 0.

    At exactly zero magnetic field there is no physical quantization axis.
    The present incoherent rate model therefore uses an orientation-averaged
    surrogate rather than privileging an arbitrary carried-over axis, which
    would otherwise create an artificial beam-helicity bias in field-free
    molasses tests.
    """

    return {
        -1: 1.0 / 3.0,
        0: 1.0 / 3.0,
        +1: 1.0 / 3.0,
    }


def allowed_excited_manifolds(ground_state: GroundState) -> tuple[int, ...]:
    """Return the addressed excited hyperfine manifolds for one ground F."""

    if ground_state.f == 2:
        return (1, 2, 3)
    return (1, 2)


def addressed_excited_manifolds_for_beam_family(
    ground_state: GroundState,
    beam_family: str,
) -> tuple[int, ...]:
    """Return the excited manifolds addressed by one MOT beam family.

    Current model:
    - cooling beams actively address only the bright F=2 manifold
    - repump beams actively address only F=1 -> F'=2
    """

    if beam_family == "cooling":
        return (1, 2, 3) if ground_state.f == 2 else ()
    if beam_family == "repump":
        return (2,) if ground_state.f == 1 else ()
    raise ValueError(f"unsupported beam family: {beam_family}")


def zero_field_transition_frequency_hz(ground_f: int, excited_f: int) -> float:
    """Return the precomputed zero-field transition frequency."""

    key = (ground_f, excited_f)
    if key not in ZERO_FIELD_TRANSITION_FREQUENCIES_HZ:
        raise KeyError(
            f"missing zero-field frequency for F={ground_f} -> F'={excited_f}; "
            "the current model intentionally leaves F=1 -> F'=0 disabled"
        )
    return ZERO_FIELD_TRANSITION_FREQUENCIES_HZ[key]


def ground_g_factor(ground_f: int) -> float:
    return GROUND_G_FACTOR_BY_F[ground_f]


def excited_g_factor(excited_f: int) -> float:
    if excited_f not in EXCITED_G_FACTOR_BY_F:
        raise KeyError(f"missing excited-state g-factor for F'={excited_f}")
    return EXCITED_G_FACTOR_BY_F[excited_f]


@lru_cache(maxsize=None)
def hyperfine_line_strength(ground_f: int, excited_f: int) -> float:
    """Return the hyperfine line-strength factor that is independent of m_F."""

    value = (
        (2 * excited_f + 1)
        * (2 * GROUND_J + 1)
        * float(wigner_6j(EXCITED_J, excited_f, RB87_NUCLEAR_SPIN, ground_f, GROUND_J, 1)) ** 2
    )
    return value


@lru_cache(maxsize=None)
def transition_strength(ground_f: int, m_f: int, excited_f: int, m_f_prime: int) -> float:
    """Return the relative dipole strength for one Zeeman transition."""

    if m_f < -ground_f or m_f > ground_f:
        return 0.0
    if m_f_prime < -excited_f or m_f_prime > excited_f:
        return 0.0
    q_value = m_f_prime - m_f
    if q_value not in (-1, 0, +1):
        return 0.0
    three_j = float(wigner_3j(excited_f, 1, ground_f, -m_f_prime, q_value, m_f))
    return hyperfine_line_strength(ground_f, excited_f) * (2 * ground_f + 1) * three_j**2


def allowed_excited_states(ground_state: GroundState) -> list[ExcitedState]:
    """Enumerate all addressed excited states for one ground state."""

    states: list[ExcitedState] = []
    for excited_f in allowed_excited_manifolds(ground_state):
        for q_value in (-1, 0, +1):
            m_f_prime = ground_state.m_f + q_value
            if -excited_f <= m_f_prime <= excited_f:
                strength = transition_strength(ground_state.f, ground_state.m_f, excited_f, m_f_prime)
                if strength > 0.0:
                    states.append(ExcitedState(f_prime=excited_f, m_f_prime=m_f_prime))
    return states


def addressed_excited_states_for_beam_family(
    ground_state: GroundState,
    beam_family: str,
) -> list[ExcitedState]:
    """Enumerate the excited states driven by one beam family."""

    states: list[ExcitedState] = []
    for excited_f in addressed_excited_manifolds_for_beam_family(ground_state, beam_family):
        for q_value in (-1, 0, +1):
            m_f_prime = ground_state.m_f + q_value
            if -excited_f <= m_f_prime <= excited_f:
                strength = transition_strength(ground_state.f, ground_state.m_f, excited_f, m_f_prime)
                if strength > 0.0:
                    states.append(ExcitedState(f_prime=excited_f, m_f_prime=m_f_prime))
    return states


def zeeman_shift_hz(
    field_magnitude_t: float,
    ground_state: GroundState,
    excited_state: ExcitedState,
) -> float:
    """Return the Zeeman shift in ordinary frequency units."""

    return BOHR_MAGNETON_OVER_H_HZ_PER_T * field_magnitude_t * (
        excited_g_factor(excited_state.f_prime) * excited_state.m_f_prime
        - ground_g_factor(ground_state.f) * ground_state.m_f
    )


def initial_ground_state(rng: np.random.Generator) -> GroundState:
    """Sample the specified initial hyperfine-state distribution."""

    if rng.uniform() < 3.0 / 8.0:
        ground_f = 1
    else:
        ground_f = 2
    m_f = int(rng.integers(-ground_f, ground_f + 1))
    return GroundState(f=ground_f, m_f=m_f)


def decay_branching_probabilities(excited_state: ExcitedState) -> list[tuple[GroundState, float]]:
    """Return normalized spontaneous-decay branching probabilities."""

    weights: list[tuple[GroundState, float]] = []
    for ground_f in (1, 2):
        for m_f in range(-ground_f, ground_f + 1):
            strength = transition_strength(ground_f, m_f, excited_state.f_prime, excited_state.m_f_prime)
            if strength > 0.0:
                weights.append((GroundState(f=ground_f, m_f=m_f), strength))
    normalization = sum(weight for _, weight in weights)
    if normalization <= 0.0:
        raise ValueError("excited state has no allowed spontaneous-decay channels")
    return [(state, weight / normalization) for state, weight in weights]
