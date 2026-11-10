"""Controls for loading scans of the corrected strange-MOT surrogate."""

from dataclasses import dataclass, replace
from math import pi

import numpy as np

from ..configuration import default_mot_apparatus_config
from ..launch_geometry import sample_incident_disc_full_sphere
from ..mot_multilevel import build_multilevel_mot_beams, default_multilevel_mot_config
from . import accelerated_surrogate
from .cell_launch_sampling import sample_disc_points_inside_cell
from .polarizability import (
    default_polarizability_csv_path,
    differential_shift_coefficients_for_wavelength,
)
from .surrogate_effective_field import (
    SurrogateEffectiveFieldConfig,
    numerical_field_jacobian_t_per_m,
    surrogate_component_intensities_w_per_m2,
)
from .trapping_beams import TrappingLaserConfig, build_trapping_beams


COOLING_WAVELENGTH_M = 780.241209686e-9
REPUMP_WAVELENGTH_M = 780.232684e-9
REPUMP_DIAMETER_MM = 12.7
COOLING_POWER_W = .027
COOLING_REFERENCE_SATURATION_W_PER_M2 = 16.69
COOLING_REFERENCE_LINEWIDTH_MHZ = 6.065
TRAP_POWERS_MW = tuple(sorted((*range(1, 40, 2), 10, 40)))
COOLING_DETUNINGS_MHZ = tuple(-5 - 2.5 * index for index in range(15))
COOLING_DIAMETERS_MM = tuple(sorted((*(5 + 2.5 * index for index in range(11)), 12.7)))
TRAPPING_WAVELENGTH_NM = 1529.268881


def current_symmetric_surrogate() -> SurrogateEffectiveFieldConfig:
    """Map the current pMOT optical geometry into a new surrogate configuration."""
    optics = TrappingLaserConfig(wavelength_m=TRAPPING_WAVELENGTH_NM * 1e-9)
    forward, returned = build_trapping_beams(optics)[:2]
    coefficients = differential_shift_coefficients_for_wavelength(
        TRAPPING_WAVELENGTH_NM, default_polarizability_csv_path()
    )
    return SurrogateEffectiveFieldConfig(
        wavelength_m=optics.wavelength_m,
        forward_waist_position_m=forward.waist_position_m[0],
        return_waist_position_m=returned.waist_position_m[0],
        forward_rayleigh_range_m=forward.rayleigh_range_m,
        return_rayleigh_range_m=returned.rayleigh_range_m,
        differential_vector_polarizability_hz_per_w_per_m2=(
            coefficients.vector_mhz_per_intensity * 1e6 / 1e5
        ),
    )


@dataclass(frozen=True, slots=True)
class ScanPoint:
    """The trap power coordinate is the power of each z traveling component."""

    geometry: str = "axial"
    trap_power_mw: float = 10.0
    cooling_detuning_mhz: float = -15.0
    cooling_diameter_mm: float = 12.7


def scan_points():
    """Build independent sweeps with one shared axial reference point."""
    reference = ScanPoint()
    sweeps = {
        "trap_power_isotropic": tuple(
            replace(reference, geometry="isotropic", trap_power_mw=power)
            for power in TRAP_POWERS_MW
        ),
        "trap_power_axial": tuple(
            replace(reference, trap_power_mw=power) for power in TRAP_POWERS_MW
        ),
        "cooling_detuning": tuple(
            replace(reference, cooling_detuning_mhz=delta) for delta in COOLING_DETUNINGS_MHZ
        ),
        "cooling_diameter": tuple(
            replace(reference, cooling_diameter_mm=diameter)
            for diameter in COOLING_DIAMETERS_MM
        ),
    }
    points = tuple(dict.fromkeys((reference, *(p for sweep in sweeps.values() for p in sweep))))
    return points, sweeps


def field_config(
    point: ScanPoint, geometry: SurrogateEffectiveFieldConfig, *, isotropic_axial_helicity: int
) -> SurrogateEffectiveFieldConfig:
    """Use nonnegative optical powers; helicity sets the gradient sign."""
    if isotropic_axial_helicity not in (-1, 1):
        raise ValueError("isotropic axial helicity must be +1 or -1")
    if point.geometry == "isotropic":
        radial_power_w = point.trap_power_mw * 1e-3
        axial_helicity = isotropic_axial_helicity
    elif point.geometry == "axial":
        radial_power_w = point.trap_power_mw * 0.5e-3
        axial_helicity = -1
    else:
        raise ValueError(f"unknown trapping geometry: {point.geometry}")
    return replace(
        geometry,
        x_path_power_w=radial_power_w,
        y_path_power_w=radial_power_w,
        z_path_power_w=point.trap_power_mw * 1e-3,
        helicity_positive_x=1,
        helicity_negative_x=1,
        helicity_positive_y=1,
        helicity_negative_y=1,
        helicity_positive_z=axial_helicity,
        helicity_negative_z=axial_helicity,
    )


def cooling_configuration(point: ScanPoint):
    """Build the physical 24-state cooling and repump beams without a coil."""
    config = replace(
        default_multilevel_mot_config(),
        cooling_wavelength_m=COOLING_WAVELENGTH_M,
        cooling_power_w_per_beam=COOLING_POWER_W,
        cooling_detuning_rad_per_s=2 * pi * point.cooling_detuning_mhz * 1e6,
        repump_wavelength_m=REPUMP_WAVELENGTH_M,
        repump_power_w_per_beam=0.1e-3,
        repump_detuning_rad_per_s=0.0,
        repumper_enabled=True,
        include_gravity=True,
    )
    apparatus = default_mot_apparatus_config()
    apparatus = replace(
        apparatus,
        cooling=replace(apparatus.cooling, beam_diameter_m=point.cooling_diameter_mm * 1e-3),
        repump=replace(apparatus.repump, beam_diameter_m=REPUMP_DIAMETER_MM * 1e-3),
    )
    beams = tuple(
        replace(beam, beam_radius_m=(
            point.cooling_diameter_mm if beam.family == "cooling" else REPUMP_DIAMETER_MM
        ) * 0.5e-3)
        for beam in build_multilevel_mot_beams(apparatus, config)
    )
    return config, beams


def prepare(point: ScanPoint, field: SurrogateEffectiveFieldConfig):
    """Set both beam envelopes in the existing compiled surrogate solver."""
    config, beams = cooling_configuration(point)
    data = list(accelerated_surrogate.prepare(config, field))
    beam_data = data[1].copy()
    for row, beam in zip(beam_data, beams, strict=True):
        row[0] = beam.power_w
        row[1] = beam.beam_radius_m**2
        row[2] = (pi * beam.beam_radius_m**2 / beam.wavelength_m)**2
        row[3] = 2 * pi / beam.wavelength_m
    data[1] = beam_data
    return tuple(data)


def field_axes(field: SurrogateEffectiveFieldConfig) -> dict:
    """Document each axis at the common origin, away from the optical waists."""
    origin = np.zeros(3)
    intensity = surrogate_component_intensities_w_per_m2(origin, field).reshape(3, 2)
    gradient = numerical_field_jacobian_t_per_m(origin, step_m=1e-8, config=field)
    return {
        "trapping_power_mw_xyz": [
            1e3 * field.x_path_power_w,
            1e3 * field.y_path_power_w,
            1e3 * field.z_path_power_w,
        ],
        "origin_trapping_intensity_w_per_cm2_xyz_each_direction": (intensity * 1e-4).tolist(),
        "origin_surrogate_field_jacobian_g_per_cm": (gradient * 100).tolist(),
    }


def cooling_saturation(point: ScanPoint) -> dict:
    """Single-beam center saturation coordinates for the 780 nm cooling light."""
    radius_m = .5e-3 * point.cooling_diameter_mm
    intensity = 2 * COOLING_POWER_W / (pi * radius_m**2)
    raw = intensity / COOLING_REFERENCE_SATURATION_W_PER_M2
    effective = raw / (1 + (2 * point.cooling_detuning_mhz / COOLING_REFERENCE_LINEWIDTH_MHZ)**2)
    return {
        "cooling_center_intensity_w_per_m2": intensity,
        "cooling_s0": raw,
        "cooling_s_eff": effective,
    }


def sample_launch_geometry(
    disc_count: int,
    points_per_disc: int,
    seed: int,
    field: SurrogateEffectiveFieldConfig,
    *,
    radius_m: float = 0.015,
    plane_distance_m: float = 0.010,
):
    """Draw uniform full-sphere directions and area-weighted in-cell discs."""
    if disc_count < 1 or points_per_disc < 1:
        raise ValueError("disc and point counts must be positive")
    direction_seed, point_seed = np.random.SeedSequence(seed).spawn(2)
    directions_rng = np.random.default_rng(direction_seed)
    positions = np.empty((disc_count, points_per_disc, 3))
    directions = np.empty((disc_count, 3))
    areas = np.empty(disc_count)
    for index, child in enumerate(point_seed.spawn(disc_count)):
        disc = sample_incident_disc_full_sphere(index, plane_distance_m, directions_rng)
        launches, areas[index], _ = sample_disc_points_inside_cell(
            disc, points_per_disc, radius_m, np.random.default_rng(child), field
        )
        positions[index] = [launch.initial_position_m for launch in launches]
        directions[index] = launches[0].incident_unit_vector
    return positions, directions, areas
