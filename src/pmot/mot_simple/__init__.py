"""Simplified two-level MOT model."""

from .configuration import (
    default_simple_mot_apparatus,
    default_simple_mot_config,
    simple_mot_paths,
    SimpleMOTConfig,
    MU_B_OVER_H_HZ_PER_T,
)
from .plotting import (
    plot_magnetic_component_grid,
    plot_simple_mot_diagnostics,
    plot_simple_mot_geometry,
)
from .simulation import (
    SimpleMOTBeam,
    SimpleMOTRateSample,
    SimpleMOTTrajectoryRecord,
    acceleration_m_per_s2,
    beam_intensity,
    build_simple_mot_beams,
    doppler_shift_hz,
    local_magnetic_field_t,
    mean_force_n,
    rate_samples,
    rk4_step,
    simulate_simple_mot_trajectory,
    wavevector_magnitude_m_inv,
    zeeman_shift_hz,
)

__all__ = [
    "default_simple_mot_apparatus",
    "default_simple_mot_config",
    "simple_mot_paths",
    "SimpleMOTConfig",
    "MU_B_OVER_H_HZ_PER_T",
    "plot_magnetic_component_grid",
    "plot_simple_mot_diagnostics",
    "plot_simple_mot_geometry",
    "SimpleMOTBeam",
    "SimpleMOTRateSample",
    "SimpleMOTTrajectoryRecord",
    "acceleration_m_per_s2",
    "beam_intensity",
    "build_simple_mot_beams",
    "doppler_shift_hz",
    "local_magnetic_field_t",
    "mean_force_n",
    "rate_samples",
    "rk4_step",
    "simulate_simple_mot_trajectory",
    "wavevector_magnitude_m_inv",
    "zeeman_shift_hz",
]
