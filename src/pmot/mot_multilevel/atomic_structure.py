"""ARC-backed, position-independent atomic data for the Rb-87 D2 MOT.

ARC is invoked only while constructing the cached :class:`AtomicStructure`.
No trajectory or force evaluation calls ARC.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from math import pi

import numpy as np

from ..configuration import (
    HBAR_J_S,
    SPEED_OF_LIGHT_M_PER_S,
    VACUUM_PERMITTIVITY_F_PER_M,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19
BOHR_RADIUS_M = 5.29177210903e-11
BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T = BOHR_MAGNETON_J_PER_T / HBAR_J_S

GROUND_N = 5
GROUND_L = 0
GROUND_J = 0.5
EXCITED_N = 5
EXCITED_L = 1
EXCITED_J = 1.5


@dataclass(frozen=True, slots=True)
class InternalState:
    """One zero-field hyperfine-Zeeman basis state."""

    index: int
    manifold: str
    f: int
    m_f: int
    hyperfine_shift_rad_per_s: float
    lande_g: float

    @property
    def is_ground(self) -> bool:
        return self.manifold == "ground"

    @property
    def is_excited(self) -> bool:
        return self.manifold == "excited"


@dataclass(frozen=True, slots=True)
class DipoleTransition:
    """One allowed ground-to-excited electric-dipole transition."""

    ground_state_index: int
    excited_state_index: int
    ground_f: int
    ground_m_f: int
    excited_f: int
    excited_m_f: int
    q: int
    dipole_matrix_element_ea0: float
    dipole_matrix_element_c_m: float
    transition_angular_frequency_rad_per_s: float
    spontaneous_decay_rate_per_s: float


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    """Immutable ARC data and adjacency lists for all 24 states."""

    states: tuple[InternalState, ...]
    ground_state_indices: tuple[int, ...]
    excited_state_indices: tuple[int, ...]
    transitions: tuple[DipoleTransition, ...]
    transitions_by_ground: tuple[tuple[DipoleTransition, ...], ...]
    transitions_by_excited: tuple[tuple[DipoleTransition, ...], ...]
    cooling_reference_angular_frequency_rad_per_s: float
    repump_reference_angular_frequency_rad_per_s: float
    fine_structure_angular_frequency_rad_per_s: float
    arc_version: str

    def state_index(self, manifold: str, f: int, m_f: int) -> int:
        for state in self.states:
            if state.manifold == manifold and state.f == f and state.m_f == m_f:
                return state.index
        raise KeyError((manifold, f, m_f))


def _load_arc_rubidium87():
    """Load ARC, adapting its legacy SciPy spherical-harmonic import.

    ARC 3.8 imports ``scipy.special.sph_harm`` at module import time. Newer
    SciPy releases expose the replacement as ``sph_harm_y``. The dipole and
    hyperfine routines used here do not call spherical harmonics, but ARC still
    imports the symbol, so provide the argument-order-compatible wrapper when
    necessary.
    """

    import scipy.special as special

    if not hasattr(special, "sph_harm"):
        def sph_harm(m, n, theta, phi):
            return special.sph_harm_y(n, m, phi, theta)

        special.sph_harm = sph_harm

    from arc import Rubidium87

    return Rubidium87()


def _arc_version() -> str:
    try:
        return version("arc-alkali-rydberg-calculator")
    except PackageNotFoundError:
        return "unknown"


def _hyperfine_shift_hz(atom, *, l: int, j: float, f: int) -> float:
    a_hz, b_hz = atom.getHFSCoefficients(5, l, j)
    return float(atom.getHFSEnergyShift(j, f, a_hz, b_hz))


def _lande_g(atom, *, l: int, j: float, f: int) -> float:
    return 0.0 if f == 0 else float(atom.getLandegfExact(l, j, f))


def spontaneous_decay_rate_per_s(
    transition_angular_frequency_rad_per_s: float,
    dipole_matrix_element_c_m: float,
) -> float:
    """Return Eq. (40)'s pairwise Einstein-A coefficient in s^-1."""

    omega = float(transition_angular_frequency_rad_per_s)
    dipole = float(dipole_matrix_element_c_m)
    if omega <= 0.0:
        raise ValueError("transition angular frequency must be positive")
    return (
        omega**3
        * dipole**2
        / (
            3.0
            * pi
            * VACUUM_PERMITTIVITY_F_PER_M
            * HBAR_J_S
            * SPEED_OF_LIGHT_M_PER_S**3
        )
    )


@lru_cache(maxsize=1)
def build_atomic_structure() -> AtomicStructure:
    """Precompute the complete 8-ground/16-excited ARC transition graph."""

    atom = _load_arc_rubidium87()
    fine_frequency_hz = float(
        atom.getTransitionFrequency(
            GROUND_N,
            GROUND_L,
            GROUND_J,
            EXCITED_N,
            EXCITED_L,
            EXCITED_J,
        )
    )
    ground_shift_hz = {
        f: _hyperfine_shift_hz(atom, l=GROUND_L, j=GROUND_J, f=f)
        for f in (1, 2)
    }
    excited_shift_hz = {
        f: _hyperfine_shift_hz(atom, l=EXCITED_L, j=EXCITED_J, f=f)
        for f in (0, 1, 2, 3)
    }

    states: list[InternalState] = []
    ground_indices: list[int] = []
    excited_indices: list[int] = []
    for f in (1, 2):
        for m_f in range(-f, f + 1):
            index = len(states)
            ground_indices.append(index)
            states.append(
                InternalState(
                    index=index,
                    manifold="ground",
                    f=f,
                    m_f=m_f,
                    hyperfine_shift_rad_per_s=2.0 * pi * ground_shift_hz[f],
                    lande_g=_lande_g(atom, l=GROUND_L, j=GROUND_J, f=f),
                )
            )
    for f in (0, 1, 2, 3):
        for m_f in range(-f, f + 1):
            index = len(states)
            excited_indices.append(index)
            states.append(
                InternalState(
                    index=index,
                    manifold="excited",
                    f=f,
                    m_f=m_f,
                    hyperfine_shift_rad_per_s=2.0 * pi * excited_shift_hz[f],
                    lande_g=_lande_g(atom, l=EXCITED_L, j=EXCITED_J, f=f),
                )
            )

    lookup = {(state.manifold, state.f, state.m_f): state.index for state in states}
    transitions: list[DipoleTransition] = []
    for ground_f in (1, 2):
        for ground_m_f in range(-ground_f, ground_f + 1):
            for excited_f in (0, 1, 2, 3):
                if abs(excited_f - ground_f) > 1:
                    continue
                for q in (-1, 0, 1):
                    excited_m_f = ground_m_f + q
                    if abs(excited_m_f) > excited_f:
                        continue
                    dipole_ea0 = float(
                        atom.getDipoleMatrixElementHFS(
                            GROUND_N,
                            GROUND_L,
                            GROUND_J,
                            ground_f,
                            ground_m_f,
                            EXCITED_N,
                            EXCITED_L,
                            EXCITED_J,
                            excited_f,
                            excited_m_f,
                            q,
                        )
                    )
                    if abs(dipole_ea0) <= 1.0e-15:
                        continue
                    transition_frequency_hz = (
                        fine_frequency_hz
                        + excited_shift_hz[excited_f]
                        - ground_shift_hz[ground_f]
                    )
                    transition_omega = 2.0 * pi * transition_frequency_hz
                    dipole_c_m = dipole_ea0 * ELEMENTARY_CHARGE_C * BOHR_RADIUS_M
                    transitions.append(
                        DipoleTransition(
                            ground_state_index=lookup[("ground", ground_f, ground_m_f)],
                            excited_state_index=lookup[("excited", excited_f, excited_m_f)],
                            ground_f=ground_f,
                            ground_m_f=ground_m_f,
                            excited_f=excited_f,
                            excited_m_f=excited_m_f,
                            q=q,
                            dipole_matrix_element_ea0=dipole_ea0,
                            dipole_matrix_element_c_m=dipole_c_m,
                            transition_angular_frequency_rad_per_s=transition_omega,
                            spontaneous_decay_rate_per_s=spontaneous_decay_rate_per_s(
                                transition_omega,
                                dipole_c_m,
                            ),
                        )
                    )

    by_ground: list[list[DipoleTransition]] = [[] for _ in states]
    by_excited: list[list[DipoleTransition]] = [[] for _ in states]
    for transition in transitions:
        by_ground[transition.ground_state_index].append(transition)
        by_excited[transition.excited_state_index].append(transition)

    cooling_reference = 2.0 * pi * (
        fine_frequency_hz + excited_shift_hz[3] - ground_shift_hz[2]
    )
    repump_reference = 2.0 * pi * (
        fine_frequency_hz + excited_shift_hz[2] - ground_shift_hz[1]
    )
    return AtomicStructure(
        states=tuple(states),
        ground_state_indices=tuple(ground_indices),
        excited_state_indices=tuple(excited_indices),
        transitions=tuple(transitions),
        transitions_by_ground=tuple(tuple(items) for items in by_ground),
        transitions_by_excited=tuple(tuple(items) for items in by_excited),
        cooling_reference_angular_frequency_rad_per_s=cooling_reference,
        repump_reference_angular_frequency_rad_per_s=repump_reference,
        fine_structure_angular_frequency_rad_per_s=2.0 * pi * fine_frequency_hz,
        arc_version=_arc_version(),
    )


__all__ = [
    "AtomicStructure",
    "BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T",
    "BOHR_RADIUS_M",
    "DipoleTransition",
    "ELEMENTARY_CHARGE_C",
    "InternalState",
    "build_atomic_structure",
    "spontaneous_decay_rate_per_s",
]
