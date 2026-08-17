"""Precomputed Rb-87 D2 hyperfine/Zeeman states and transition graphs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi

from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_6j


RB87_NUCLEAR_SPIN = 1.5
GROUND_J = 0.5
EXCITED_J = 1.5
GROUND_J_LANDE = 2.00233113
EXCITED_J_LANDE = 1.3341

BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
HBAR_J_S = 1.054571817e-34
BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T = BOHR_MAGNETON_J_PER_T / HBAR_J_S

EXCITED_HYPERFINE_OFFSET_RAD_PER_S = {
    1: -2.0 * pi * 423.597e6,
    2: -2.0 * pi * 266.650e6,
    3: 0.0,
}


@dataclass(frozen=True, slots=True)
class InternalState:
    """One indexed hyperfine/Zeeman state."""

    index: int
    manifold: str
    f: int
    m_f: int
    energy_offset_rad_per_s: float
    lande_g: float

    @property
    def is_ground(self) -> bool:
        return self.manifold == "ground"

    @property
    def is_excited(self) -> bool:
        return self.manifold == "excited"

    @property
    def is_dark(self) -> bool:
        return self.is_ground and self.f == 1


@dataclass(frozen=True, slots=True)
class DipoleTransition:
    """One directed ground-to-excited electric-dipole transition."""

    ground_state_index: int
    excited_state_index: int
    ground_f: int
    ground_m_f: int
    excited_f: int
    excited_m_f: int
    q: int
    c_squared: float
    hyperfine_offset_rad_per_s: float


@dataclass(frozen=True, slots=True)
class DecayChannel:
    """One normalized spontaneous-decay branch."""

    excited_state_index: int
    ground_state_index: int
    q: int
    branch_weight: float
    branch_probability: float


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    """Immutable state arrays and direct transition adjacency lists."""

    states: tuple[InternalState, ...]
    ground_state_indices: tuple[int, ...]
    excited_state_indices: tuple[int, ...]
    absorption_transitions: tuple[DipoleTransition, ...]
    decay_channels: tuple[DecayChannel, ...]
    absorption_by_ground: tuple[tuple[DipoleTransition, ...], ...]
    transitions_by_excited: tuple[tuple[DipoleTransition, ...], ...]
    decay_by_excited: tuple[tuple[DecayChannel, ...], ...]

    def state_index(self, manifold: str, f: int, m_f: int) -> int:
        for state in self.states:
            if state.manifold == manifold and state.f == f and state.m_f == m_f:
                return state.index
        raise KeyError((manifold, f, m_f))


def hyperfine_lande_g(f: int, j: float, electronic_g: float) -> float:
    """Return weak-field hyperfine g_F, neglecting the small nuclear term."""

    if f <= 0:
        return 0.0
    numerator = f * (f + 1.0) + j * (j + 1.0) - RB87_NUCLEAR_SPIN * (RB87_NUCLEAR_SPIN + 1.0)
    return electronic_g * numerator / (2.0 * f * (f + 1.0))


@lru_cache(maxsize=None)
def raw_dipole_strength(ground_f: int, ground_m_f: int, excited_f: int, excited_m_f: int) -> float:
    """Return the unnormalized state-resolved D2 dipole strength."""

    q = excited_m_f - ground_m_f
    if q not in (-1, 0, 1):
        return 0.0
    if abs(ground_m_f) > ground_f or abs(excited_m_f) > excited_f:
        return 0.0
    if abs(excited_f - ground_f) > 1 or (ground_f == 0 and excited_f == 0):
        return 0.0
    six_j = float(wigner_6j(EXCITED_J, excited_f, RB87_NUCLEAR_SPIN, ground_f, GROUND_J, 1))
    three_j = float(wigner_3j(excited_f, 1, ground_f, -excited_m_f, q, ground_m_f))
    return (2 * excited_f + 1) * (2 * GROUND_J + 1) * six_j**2 * (2 * ground_f + 1) * three_j**2


CYCLING_RAW_STRENGTH = raw_dipole_strength(2, 2, 3, 3)


def normalized_dipole_strength(ground_f: int, ground_m_f: int, excited_f: int, excited_m_f: int) -> float:
    """Return C^2 normalized to |2,+2> -> |3,+3>."""

    return raw_dipole_strength(ground_f, ground_m_f, excited_f, excited_m_f) / CYCLING_RAW_STRENGTH


@lru_cache(maxsize=1)
def build_atomic_structure() -> AtomicStructure:
    """Generate all 23 states and precompute allowed transition graphs."""

    states: list[InternalState] = []
    ground_indices: list[int] = []
    excited_indices: list[int] = []
    for f in (1, 2):
        for m_f in range(-f, f + 1):
            index = len(states)
            ground_indices.append(index)
            states.append(
                InternalState(index, "ground", f, m_f, 0.0, hyperfine_lande_g(f, GROUND_J, GROUND_J_LANDE))
            )
    for f in (1, 2, 3):
        for m_f in range(-f, f + 1):
            index = len(states)
            excited_indices.append(index)
            states.append(
                InternalState(
                    index,
                    "excited",
                    f,
                    m_f,
                    EXCITED_HYPERFINE_OFFSET_RAD_PER_S[f],
                    hyperfine_lande_g(f, EXCITED_J, EXCITED_J_LANDE),
                )
            )

    lookup = {(state.manifold, state.f, state.m_f): state.index for state in states}
    transitions: list[DipoleTransition] = []
    for ground_m_f in range(-2, 3):
        ground_index = lookup[("ground", 2, ground_m_f)]
        for excited_f in (1, 2, 3):
            for q in (-1, 0, 1):
                excited_m_f = ground_m_f + q
                if abs(excited_m_f) > excited_f:
                    continue
                strength = normalized_dipole_strength(2, ground_m_f, excited_f, excited_m_f)
                if strength <= 0.0:
                    continue
                transitions.append(
                    DipoleTransition(
                        ground_index,
                        lookup[("excited", excited_f, excited_m_f)],
                        2,
                        ground_m_f,
                        excited_f,
                        excited_m_f,
                        q,
                        strength,
                        EXCITED_HYPERFINE_OFFSET_RAD_PER_S[excited_f],
                    )
                )

    decay_channels: list[DecayChannel] = []
    for excited_index in excited_indices:
        excited = states[excited_index]
        raw_channels: list[tuple[int, int, float]] = []
        for ground_f in (1, 2):
            for ground_m_f in range(-ground_f, ground_f + 1):
                weight = raw_dipole_strength(ground_f, ground_m_f, excited.f, excited.m_f)
                if weight > 0.0:
                    raw_channels.append((lookup[("ground", ground_f, ground_m_f)], excited.m_f - ground_m_f, weight))
        normalization = sum(weight for _, _, weight in raw_channels)
        for ground_index, q, weight in raw_channels:
            decay_channels.append(DecayChannel(excited_index, ground_index, q, weight, weight / normalization))

    absorption_by_ground: list[list[DipoleTransition]] = [[] for _ in states]
    transitions_by_excited: list[list[DipoleTransition]] = [[] for _ in states]
    decay_by_excited: list[list[DecayChannel]] = [[] for _ in states]
    for transition in transitions:
        absorption_by_ground[transition.ground_state_index].append(transition)
        transitions_by_excited[transition.excited_state_index].append(transition)
    for channel in decay_channels:
        decay_by_excited[channel.excited_state_index].append(channel)

    return AtomicStructure(
        tuple(states),
        tuple(ground_indices),
        tuple(excited_indices),
        tuple(transitions),
        tuple(decay_channels),
        tuple(tuple(items) for items in absorption_by_ground),
        tuple(tuple(items) for items in transitions_by_excited),
        tuple(tuple(items) for items in decay_by_excited),
    )
