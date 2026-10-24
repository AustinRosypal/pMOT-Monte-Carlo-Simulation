"""Effective two-level pseudo-MOT for the Rb-87 F=2 -> F'=3 manifold.

The package combines the no-coil trapping-light geometry, the tabulated
differential AC-Stark shift, local cooling-polarization decomposition, and a
deterministic mean-force trajectory integrator.  It is an effective manifold
closure, not an m_F-resolved Stark Hamiltonian.
"""

from .trapping_geometry import (
    DEFAULT_FALLBACK_QUANTIZATION_AXIS,
    DEFAULT_INCIDENT_HELICITIES_BY_AXIS,
    DEFAULT_RETRO_HELICITIES_BY_AXIS,
    PMOT2LevelGeometrySample,
    build_pmot_2level_trapping_beams,
    component_intensities_w_per_m2,
    component_propagation_directions,
    evaluate_trapping_geometry,
    optical_spin_vector_w_per_m2,
    quantization_axis_at_point,
    total_intensity_w_per_m2,
)
from .ac_stark import (
    EFFECTIVE_TRANSITION_LABEL,
    ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION,
    PMOT2LevelStarkSample,
    atom_frame_trapping_wavelengths_nm,
    evaluate_differential_ac_stark_shift,
    load_pmot_2level_polarizability_table,
)
from .simulation import (
    DEFAULT_COOLING_HELICITIES_BY_AXIS,
    DEFAULT_PMOT_2LEVEL_COOLING_POWER_W_PER_BEAM,
    DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM,
    DEFAULT_PMOT_2LEVEL_REPUMP_WAVELENGTH_M,
    PMOT2LevelLightBundle,
    PMOT2LevelLocalObservable,
    PMOT2LevelRateSample,
    PMOT2LevelTrajectoryRecord,
    build_pmot_2level_cooling_beams,
    build_pmot_2level_mot_light,
    build_pmot_2level_repump_beams,
    evaluate_pmot_2level_local_observable,
    rk4_step_pmot_2level,
    simulate_pmot_2level_trajectory,
)
from .visualization import (
    add_beam_geometry,
    animate_trajectory,
    plot_force_velocity_history,
    plot_trajectory_3d,
    trajectory_arrays,
)

__all__ = [
    "DEFAULT_COOLING_HELICITIES_BY_AXIS",
    "DEFAULT_PMOT_2LEVEL_COOLING_POWER_W_PER_BEAM",
    "DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM",
    "DEFAULT_PMOT_2LEVEL_REPUMP_WAVELENGTH_M",
    "DEFAULT_FALLBACK_QUANTIZATION_AXIS",
    "DEFAULT_INCIDENT_HELICITIES_BY_AXIS",
    "DEFAULT_RETRO_HELICITIES_BY_AXIS",
    "EFFECTIVE_TRANSITION_LABEL",
    "ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION",
    "PMOT2LevelGeometrySample",
    "PMOT2LevelLocalObservable",
    "PMOT2LevelLightBundle",
    "PMOT2LevelRateSample",
    "PMOT2LevelStarkSample",
    "PMOT2LevelTrajectoryRecord",
    "atom_frame_trapping_wavelengths_nm",
    "add_beam_geometry",
    "animate_trajectory",
    "build_pmot_2level_cooling_beams",
    "build_pmot_2level_mot_light",
    "build_pmot_2level_repump_beams",
    "build_pmot_2level_trapping_beams",
    "component_intensities_w_per_m2",
    "component_propagation_directions",
    "evaluate_trapping_geometry",
    "evaluate_differential_ac_stark_shift",
    "evaluate_pmot_2level_local_observable",
    "load_pmot_2level_polarizability_table",
    "optical_spin_vector_w_per_m2",
    "plot_force_velocity_history",
    "plot_trajectory_3d",
    "quantization_axis_at_point",
    "rk4_step_pmot_2level",
    "simulate_pmot_2level_trajectory",
    "total_intensity_w_per_m2",
    "trajectory_arrays",
]
