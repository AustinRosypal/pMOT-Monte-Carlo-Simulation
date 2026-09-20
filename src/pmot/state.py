"""Shared external classical state used by model-specific MOT dynamics."""

from __future__ import annotations

from dataclasses import dataclass

from .beams import Vec3


@dataclass(frozen=True, slots=True)
class AtomState:
    """One atom's Cartesian position and velocity in SI units."""

    position_m: Vec3
    velocity_m_per_s: Vec3
