"""Configuration helpers for the validation MOT magnetic-field model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M


VACUUM_PERMEABILITY_H_PER_M = 1.0 / (VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S**2)
GAUSS_PER_TESLA = 1.0e4
TESLA_PER_GAUSS = 1.0e-4
TESLA_PER_METER_PER_GAUSS_PER_CM = 1.0e-2


@dataclass(frozen=True, slots=True)
class AntiHelmholtzCoilConfig:
    """One anti-Helmholtz coil pair in SI units."""

    radius_m: float
    turns_per_coil: int
    current_a: float
    center_separation_m: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.turns_per_coil <= 0:
            raise ValueError("turns_per_coil must be positive")
        if self.center_separation_m <= 0.0:
            raise ValueError("center_separation_m must be positive")

    @property
    def half_separation_m(self) -> float:
        return 0.5 * self.center_separation_m


def mot_project_paths(root: Path | None = None) -> dict[str, Path]:
    """Return dedicated output paths for MOT magnetic-field studies."""

    project_root = root or Path(__file__).resolve().parents[3]
    return {
        "root": project_root,
        "notebooks_mot": project_root / "notebooks" / "mot",
        "outputs_fields_mot": project_root / "outputs" / "fields" / "mot",
        "outputs_figures_mot": project_root / "outputs" / "figures" / "mot",
    }

