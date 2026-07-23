"""Validation MOT magnetic-field utilities."""

from .configuration import (
    AntiHelmholtzCoilConfig,
    GAUSS_PER_TESLA,
    TESLA_PER_GAUSS,
    TESLA_PER_METER_PER_GAUSS_PER_CM,
    VACUUM_PERMEABILITY_H_PER_M,
    mot_project_paths,
)
from .magnetic_fields import (
    MagneticFieldSample,
    anti_helmholtz_axial_gradient_t_per_m,
    anti_helmholtz_cylindrical_field_t,
    anti_helmholtz_field_t,
    current_for_target_axial_gradient_a,
    default_anti_helmholtz_config,
    field_sample,
    line_sample,
    plane_sample,
    vector_cloud_sample,
)
from .plotting import (
    plot_field_heatmap,
    plot_field_lineout,
    plot_plane_quiver,
    plot_vector_cloud_3d,
)

__all__ = [
    "AntiHelmholtzCoilConfig",
    "GAUSS_PER_TESLA",
    "TESLA_PER_GAUSS",
    "TESLA_PER_METER_PER_GAUSS_PER_CM",
    "VACUUM_PERMEABILITY_H_PER_M",
    "mot_project_paths",
    "MagneticFieldSample",
    "anti_helmholtz_axial_gradient_t_per_m",
    "anti_helmholtz_cylindrical_field_t",
    "anti_helmholtz_field_t",
    "current_for_target_axial_gradient_a",
    "default_anti_helmholtz_config",
    "field_sample",
    "line_sample",
    "plane_sample",
    "vector_cloud_sample",
    "plot_field_heatmap",
    "plot_field_lineout",
    "plot_plane_quiver",
    "plot_vector_cloud_3d",
]

