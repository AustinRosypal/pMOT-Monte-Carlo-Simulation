"""Configuration for the physical population-rate-equation Rb-87 MOT."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path


MODEL_OUTPUT_NAMESPACE = "mot_multilevel_population_rate_v1"


@dataclass(frozen=True, slots=True)
class MultilevelMOTConfig:
    """Physical and numerical controls, with angular frequencies in rad/s."""

    cooling_detuning_rad_per_s: float = -2.0 * pi * 15.0e6
    repump_detuning_rad_per_s: float = 0.0
    cooling_power_w_per_beam: float = 27.0e-3
    repump_power_w_per_beam: float = 0.1e-3
    cooling_wavelength_m: float = 780.0e-9
    repump_wavelength_m: float = 780.232684e-9
    enabled_cooling_excited_manifolds: tuple[int, ...] = (1, 2, 3)
    enabled_repump_excited_manifolds: tuple[int, ...] = (0, 1, 2)
    repumper_enabled: bool = True
    magnetic_field_epsilon_t: float = 1.0e-12
    include_gravity: bool = True
    conservation_absolute_tolerance_per_s: float = 1.0e-7
    steady_state_relative_tolerance: float = 1.0e-10
    population_absolute_tolerance: float = 1.0e-11

    def __post_init__(self) -> None:
        if self.cooling_power_w_per_beam < 0.0:
            raise ValueError("cooling_power_w_per_beam must be non-negative")
        if self.repump_power_w_per_beam < 0.0:
            raise ValueError("repump_power_w_per_beam must be non-negative")
        if self.cooling_wavelength_m <= 0.0 or self.repump_wavelength_m <= 0.0:
            raise ValueError("beam wavelengths must be positive")
        if self.magnetic_field_epsilon_t < 0.0:
            raise ValueError("magnetic_field_epsilon_t must be non-negative")


def default_multilevel_mot_config() -> MultilevelMOTConfig:
    return MultilevelMOTConfig()


def multilevel_mot_paths(root: Path | None = None) -> dict[str, Path]:
    project_root = root or Path(__file__).resolve().parents[3]
    base = project_root / "outputs"
    return {
        "root": project_root,
        "figures": base / "figures" / MODEL_OUTPUT_NAMESPACE,
        "trajectories": base / "trajectories" / MODEL_OUTPUT_NAMESPACE,
        "statistics": base / "statistics" / MODEL_OUTPUT_NAMESPACE,
        "fields": base / "fields" / MODEL_OUTPUT_NAMESPACE,
    }


__all__ = [
    "MODEL_OUTPUT_NAMESPACE",
    "MultilevelMOTConfig",
    "default_multilevel_mot_config",
    "multilevel_mot_paths",
]
