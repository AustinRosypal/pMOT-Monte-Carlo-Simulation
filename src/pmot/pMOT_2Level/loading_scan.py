"""Controls and calibrated optical intensities for the two-level loading scan."""

from dataclasses import dataclass, replace
from math import pi, sqrt

import numpy as np

from ..configuration import (
    COOLING_WAVELENGTH_M, HBAR_J_S, PLANCK_CONSTANT_J_S, RB87_MASS_KG, STANDARD_GRAVITY_M_PER_S2,
    TESLA_PER_METER_PER_GAUSS_PER_CM,
)
from ..mot_simple.configuration import default_simple_mot_config
from ..mot_simple.simulation import beam_intensity
from ..mot_strange.trapping_beams import TrappingLaserConfig
from .ac_stark import evaluate_differential_ac_stark_shift, load_pmot_2level_component_polarizability_table
from .simulation import build_pmot_2level_mot_light
from .trapping_geometry import build_pmot_2level_trapping_beams, component_intensities_w_per_m2


WAVELENGTHS_NM = (1529.268881048958, 1529.273195181438)
RANKS = ('svt', 'sv', 'v')
GEOMETRIES = ('equal_power', 'doubled_axial')
GRADIENTS_G_CM = tuple(range(2, 41, 2))
COOLING_POWER_W = 0.027
COOLING_DIAMETER_M = 0.0127
COOLING_DETUNING_HZ = -17.5e6
HELICITIES = ('sigma+', 'sigma+', 'sigma-')
CELL_HALF_M = sqrt(2) * 0.010
DISC_RADIUS_M = 0.015
DISC_DISTANCE_M = 0.010
CORE_M = 0.002
FINAL_SPEED_M_S = 0.05


@dataclass(frozen=True)
class Setting:
    ranks: str
    wavelength_nm: float
    geometry: str
    axial_magnitude_g_cm: int


def settings():
    return tuple(Setting(r, w, g, b) for r in RANKS for w in WAVELENGTHS_NM
                 for g in GEOMETRIES for b in GRADIENTS_G_CM)


def simple_config():
    return replace(default_simple_mot_config(), cooling_detuning_hz=COOLING_DETUNING_HZ)


def cooling_components():
    # Use the existing builder's six cooling components only. The trajectory
    # inputs have no repump components and the repump power is explicitly zero.
    cooling = build_pmot_2level_mot_light(
        simple_config=simple_config(), cooling_power_w_per_beam=COOLING_POWER_W,
        repump_power_w_per_beam=0.0, cooling_helicities_by_axis=HELICITIES,
    ).cooling_beams
    for i, beam in enumerate(cooling):
        assert beam.wavelength_m == COOLING_WAVELENGTH_M
        assert beam.intensity_beam.beam_radius_m == COOLING_DIAMETER_M / 2
        assert beam.direction == tuple(np.eye(3)[i//2] * (1 if i % 2 == 0 else -1))
    return cooling


def prepare_reference(setting):
    """Calibrate signed origin slopes using the vector energy divided by mu_B.

    For either member of a symmetric pair at the origin, dI/dx is
    -2*x*I/(z_R**2+x**2), with x signed from its waist. The difference
    between derivatives is 4*d*I/(z_R**2+d**2) in magnitude.
    """
    laser = TrappingLaserConfig(wavelength_m=setting.wavelength_nm * 1e-9)
    trap = build_pmot_2level_trapping_beams(
        laser, incident_helicities_by_axis=HELICITIES, retro_helicities_by_axis=HELICITIES,
    )
    table = load_pmot_2level_component_polarizability_table(setting.ranks)
    simple = simple_config()
    ones = (1., 1., 1.)
    origin = np.zeros(3)
    sample = evaluate_differential_ac_stark_shift(
        trap, origin, origin, path_powers_w=ones, polarizability_table=table,
    )
    d = laser.focus_offset_m
    unit_slope = (
        -2 * d / (trap[0].rayleigh_range_m**2 + d*d)
        * np.asarray(sample.component_vector_energy_vectors_j)[::2].diagonal()
        * 2 / (simple.effective_magnetic_moment_hz_per_t * PLANCK_CONSTANT_J_S)
        / TESLA_PER_METER_PER_GAUSS_PER_CM
    )
    b = setting.axial_magnitude_g_cm
    radial = b if setting.geometry == 'equal_power' else b / 2
    signed = np.asarray((radial, radial, -b))
    powers = signed / unit_slope
    if not np.all(powers > 0):
        raise ValueError(f'Polarization does not realize target slopes: {unit_slope}')
    intensities = component_intensities_w_per_m2(origin, trap, path_powers_w=powers)
    return trap, table, powers, intensities, signed


def cooling_saturation():
    config = simple_config()
    raw = beam_intensity(cooling_components()[0], (0., 0., 0.)) / config.saturation_intensity_w_per_m2
    return raw, raw / (1 + (2 * config.cooling_detuning_hz / config.linewidth_hz)**2)


def speed_support_upper_m_s():
    """Above this speed no shot can slow to the retention limit in the cube.

    Sum of cooling rates is <= Gamma/2 since the q weights sum to one
    and every denominator is >= 1 + total saturation. Add gravity to the
    largest acceleration. Project displacement along the initial velocity;
    even maximal opposing acceleration cannot slow the atom before it
    has crossed more than the full cube diagonal.
    """
    accel = (HBAR_J_S * (2*pi / COOLING_WAVELENGTH_M)
             * simple_config().maximum_scattering_rate_per_s / RB87_MASS_KG
             + STANDARD_GRAVITY_M_PER_S2)
    return sqrt(FINAL_SPEED_M_S**2 + 2*accel*(2*sqrt(3)*CELL_HALF_M))
