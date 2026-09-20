"""Shared compact Rb-87 transition data used by model-specific solvers."""

from __future__ import annotations

from dataclasses import dataclass

from .configuration import RB87_COOLING_RESONANCE_HZ
from .configuration import RB87_REPUMP_RESONANCE_HZ


@dataclass(frozen=True, slots=True)
class RB87CoolingTransition:
    """Basic Rb-87 cooling-transition parameters."""

    resonance_frequency_hz: float = RB87_COOLING_RESONANCE_HZ
    linewidth_hz: float = 6.065e6
    saturation_intensity_w_per_m2: float = 16.69


@dataclass(frozen=True, slots=True)
class RB87RepumpTransition:
    """Basic Rb-87 repump-transition parameters."""

    resonance_frequency_hz: float = RB87_REPUMP_RESONANCE_HZ
    linewidth_hz: float = 6.065e6
    saturation_intensity_w_per_m2: float = 16.69
