"""State and recoil primitives for later event-driven trajectory coupling."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from ..configuration import HBAR_J_S
from ..configuration import RB87_MASS_KG
from .atomic_structure import AtomicStructure
from .configuration import InitializationMode
from .polarization import Vec3


@dataclass(frozen=True, slots=True)
class MultilevelAtomState:
    """Classical phase-space state plus one definite indexed internal state."""

    position_m: Vec3
    velocity_m_per_s: Vec3
    internal_state_index: int
    last_quantization_axis: Vec3 = (0.0, 0.0, 1.0)
    dark: bool = False


@dataclass(slots=True)
class TrajectoryCounters:
    """Compact event diagnostics required by MULTILEVEL_MOT.md."""

    absorption_events: int = 0
    spontaneous_emissions: int = 0
    stimulated_emissions: int = 0
    photons_before_dark: int = 0
    dark_entry_time_s: float | None = None
    dark_parent_excited_f: int | None = None


def sample_initial_internal_state(
    structure: AtomicStructure,
    mode: InitializationMode,
    rng: np.random.Generator,
) -> int:
    """Sample validation or degeneracy-weighted vapor populations."""

    if mode == InitializationMode.VALIDATION:
        f = 2
    elif mode == InitializationMode.VAPOR:
        f = 2 if rng.random() < 5.0 / 8.0 else 1
    else:
        raise ValueError(f"unsupported initialization mode: {mode}")
    m_f = int(rng.integers(-f, f + 1))
    return structure.state_index("ground", f, m_f)


def recoil_speed_m_per_s(wavelength_m: float, atom_mass_kg: float = RB87_MASS_KG) -> float:
    return HBAR_J_S * (2.0 * pi / wavelength_m) / atom_mass_kg


def absorption_velocity_kick(direction: Vec3, wavelength_m: float) -> Vec3:
    speed = recoil_speed_m_per_s(wavelength_m)
    return tuple(speed * value for value in direction)


def stimulated_emission_velocity_kick(direction: Vec3, wavelength_m: float) -> Vec3:
    speed = recoil_speed_m_per_s(wavelength_m)
    return tuple(-speed * value for value in direction)


def isotropic_direction(rng: np.random.Generator) -> Vec3:
    z = float(rng.uniform(-1.0, 1.0))
    azimuth = float(rng.uniform(0.0, 2.0 * pi))
    radial = float(np.sqrt(max(0.0, 1.0 - z * z)))
    return radial * float(np.cos(azimuth)), radial * float(np.sin(azimuth)), z


def spontaneous_emission_velocity_kick(
    wavelength_m: float,
    rng: np.random.Generator,
) -> Vec3:
    """Return -hbar*k*n/m for isotropic emitted-photon direction n."""

    speed = recoil_speed_m_per_s(wavelength_m)
    direction = isotropic_direction(rng)
    return tuple(-speed * value for value in direction)
