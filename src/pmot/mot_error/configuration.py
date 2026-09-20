"""Configuration and physical constants for the state-resolved Rb-87 D2 MOT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import pi
from pathlib import Path


class InitializationMode(StrEnum):
    """Supported initial ground-manifold population models."""

    VALIDATION = "validation"
    VAPOR = "vapor"


class DarkStateBehavior(StrEnum):
    """Action after an atom enters the uncoupled F=1 manifold."""

    TERMINATE = "terminate"
    BALLISTIC = "ballistic"


@dataclass(frozen=True, slots=True)
class MultilevelMOTConfig:
    """Numerical and physical configuration for the multilevel MOT model.

    All frequency-like values in this package are angular frequencies in rad/s.
    """

    cooling_detuning_rad_per_s: float = -2.0 * pi * 15.0e6
    repump_detuning_rad_per_s: float = 0.0
    repump_power_w_per_beam: float = 0.1e-3
    natural_linewidth_rad_per_s: float = 2.0 * pi * 6.07e6
    saturation_intensity_w_per_m2: float = 16.69
    wavelength_m: float = 780.0e-9
    repump_wavelength_m: float = 780.232684e-9
    magnetic_field_epsilon_t: float = 1.0e-12
    enabled_excited_manifolds: tuple[int, ...] = (1, 2, 3)
    enabled_repump_excited_manifolds: tuple[int, ...] = (0, 1, 2)
    repumper_enabled: bool = False
    initialization_mode: InitializationMode = InitializationMode.VALIDATION
    dark_state_behavior: DarkStateBehavior = DarkStateBehavior.TERMINATE
    include_gravity: bool = True
    diagnostics_enabled: bool = False


def default_multilevel_mot_config() -> MultilevelMOTConfig:
    """Return the authoritative initial multilevel configuration."""

    return MultilevelMOTConfig()


def multilevel_mot_paths(root: Path | None = None) -> dict[str, Path]:
    """Return output paths reserved for the multilevel MOT workflow."""

    project_root = root or Path(__file__).resolve().parents[3]
    base = project_root / "outputs"
    return {
        "root": project_root,
        "figures": base / "figures" / "mot_multilevel",
        "trajectories": base / "trajectories" / "mot_multilevel",
        "statistics": base / "statistics" / "mot_multilevel",
        "fields": base / "fields" / "mot_multilevel",
    }
