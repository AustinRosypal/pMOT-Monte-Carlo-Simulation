"""Configuration for the simplified two-level MOT model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2
from ..configuration import STANDARD_GRAVITY_M_PER_S2
from ..configuration import default_mot_apparatus_config
from ..configuration import MOTApparatusConfig


MU_B_OVER_H_HZ_PER_T = 1.3996245e10


@dataclass(frozen=True, slots=True)
class SimpleMOTConfig:
    """Configuration for the effective two-level MOT model.

    All frequency-like quantities are expressed in ordinary frequency units, Hz.
    """

    cooling_detuning_hz: float
    linewidth_hz: float
    saturation_intensity_w_per_m2: float
    effective_magnetic_moment_hz_per_t: float
    axis_polarization_sign: dict[str, float]
    include_gravity: bool
    gravity_acceleration_m_per_s2: tuple[float, float, float]
    sample_count_per_axis: int
    plot_extent_m: float


def default_simple_mot_config() -> SimpleMOTConfig:
    """Return the default simplified MOT configuration."""

    return SimpleMOTConfig(
        cooling_detuning_hz=-15.0e6,
        linewidth_hz=6.065e6,
        saturation_intensity_w_per_m2=16.69,
        effective_magnetic_moment_hz_per_t=MU_B_OVER_H_HZ_PER_T,
        axis_polarization_sign={
            "horizontal_x": +1.0,
            "horizontal_y": +1.0,
            "vertical_z": -1.0,
        },
        include_gravity=True,
        gravity_acceleration_m_per_s2=GRAVITY_ACCELERATION_M_PER_S2,
        sample_count_per_axis=181,
        plot_extent_m=20.0e-3,
    )


def default_simple_mot_apparatus() -> MOTApparatusConfig:
    """Return the shared optical apparatus config used by the simplified MOT."""

    return default_mot_apparatus_config()


def simple_mot_paths(root: Path | None = None) -> dict[str, Path]:
    """Return dedicated paths for the simplified MOT workflow."""

    project_root = root or Path(__file__).resolve().parents[3]
    return {
        "root": project_root,
        "notebooks_simple_mot": project_root / "notebooks" / "mot_simple",
        "outputs_fields_simple_mot": project_root / "outputs" / "fields" / "mot_simple",
        "outputs_trajectories_simple_mot": project_root / "outputs" / "trajectories" / "mot_simple",
        "outputs_figures_simple_mot": project_root / "outputs" / "figures" / "mot_simple",
        "outputs_statistics_simple_mot": project_root / "outputs" / "statistics" / "mot_simple",
    }
