"""Core package for pMOT beam geometry, fields, and atomic-data studies."""

from .atomic_data import (
    DifferentialPolarizabilitySample,
    DifferentialShiftCoefficients,
    RB87CoolingTransition,
    convert_differential_polarizability_to_mhz_per_intensity,
    default_polarizability_csv_path,
    differential_shift_coefficients_for_wavelength,
    load_differential_polarizability_csv,
    polarizability_dataframe,
)
from .beams import (
    GaussianBeam,
    Vec3,
    axis_direction_from_name,
    focused_waist_radius,
)
from .configuration import (
    PMOTSimulationConfig,
    default_simulation_config,
    describe_configuration,
    notebook_order,
    project_paths,
)
from .fields import (
    CoolingBeam,
    beam_intensity_w_per_m2,
    build_cooling_beams,
    sample_intensity_along_line,
    sample_intensity_cloud,
    sample_intensity_slice,
    sample_intensity_volume,
    total_intensity_w_per_m2,
)
from .plotting import (
    plot_apparatus_geometry_3d,
    plot_beam_crossing_zoom,
    plot_intensity_cloud_3d,
    plot_intensity_lineout,
    plot_polarizability_curves,
    plot_scalar_field_slice,
)

__all__ = [
    "DifferentialPolarizabilitySample",
    "DifferentialShiftCoefficients",
    "RB87CoolingTransition",
    "convert_differential_polarizability_to_mhz_per_intensity",
    "default_polarizability_csv_path",
    "differential_shift_coefficients_for_wavelength",
    "load_differential_polarizability_csv",
    "polarizability_dataframe",
    "GaussianBeam",
    "Vec3",
    "axis_direction_from_name",
    "focused_waist_radius",
    "PMOTSimulationConfig",
    "default_simulation_config",
    "describe_configuration",
    "notebook_order",
    "project_paths",
    "CoolingBeam",
    "beam_intensity_w_per_m2",
    "build_cooling_beams",
    "sample_intensity_along_line",
    "sample_intensity_cloud",
    "sample_intensity_slice",
    "sample_intensity_volume",
    "total_intensity_w_per_m2",
    "plot_apparatus_geometry_3d",
    "plot_beam_crossing_zoom",
    "plot_intensity_cloud_3d",
    "plot_intensity_lineout",
    "plot_polarizability_curves",
    "plot_scalar_field_slice",
]
