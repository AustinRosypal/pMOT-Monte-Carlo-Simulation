"""Shared physical constants and apparatus parameters for all MOT models."""

from __future__ import annotations

from dataclasses import dataclass
PLANCK_CONSTANT_J_S = 6.62607015e-34
HBAR_J_S = 1.054571817e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
RB87_MASS_KG = 86.9091805310 * ATOMIC_MASS_UNIT_KG
STANDARD_GRAVITY_M_PER_S2 = 9.80665
GRAVITY_ACCELERATION_M_PER_S2 = (0.0, 0.0, -STANDARD_GRAVITY_M_PER_S2)
VACUUM_PERMEABILITY_H_PER_M = 1.0 / (
    VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S**2
)
GAUSS_PER_TESLA = 1.0e4
TESLA_PER_GAUSS = 1.0e-4
TESLA_PER_METER_PER_GAUSS_PER_CM = 1.0e-2

COOLING_WAVELENGTH_M = 780e-9
REPUMP_WAVELENGTH_M = 780e-9
RB87_COOLING_RESONANCE_HZ = 384.228115e12
RB87_REPUMP_RESONANCE_HZ = 384.234683e12


@dataclass(frozen=True, slots=True)
class LensSpec:
    """Mounted lens specification in SI units.

    Retained as apparatus metadata even though the present cooling/repump model
    treats the MOT beams as collimated throughout the cell.
    """

    name: str
    focal_length_m: float
    effective_focal_length_m: float
    back_focal_length_m: float
    center_thickness_m: float
    mount_length_m: float
    mount_overhang_m: float
    mount_half_height_m: float


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """Glass-cell geometry."""

    outer_diameter_m: float = 30e-3
    wall_thickness_m: float = 5e-3


@dataclass(frozen=True, slots=True)
class MOTBeamConfig:
    """Configuration for one collimated MOT-light family."""

    name: str
    role: str
    wavelength_m: float
    resonance_frequency_hz: float
    detuning_hz: float
    beam_diameter_m: float = 12.7e-3
    power_w_per_beam: float = 20.0e-3
    propagation_length_m: float = 120e-3

    @property
    def beam_radius_m(self) -> float:
        return 0.5 * self.beam_diameter_m


@dataclass(frozen=True, slots=True)
class AxisDefinition:
    """One physical beam axis in the shared MOT apparatus."""

    name: str
    cell_angle_of_incidence_deg: float
    description: str


@dataclass(frozen=True, slots=True)
class MOTApparatusConfig:
    """Top-level parameter set for the present apparatus model."""

    lens: LensSpec
    cell: CellGeometry
    cooling: MOTBeamConfig
    repump: MOTBeamConfig
    axes: tuple[AxisDefinition, AxisDefinition, AxisDefinition]
    samples_per_axis: int = 201
    volume_extent_m: float = 50e-3


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


def ac508_080_c_lens() -> LensSpec:
    """Return the horizontal-path achromat used in the current model."""

    return LensSpec(
        name="AC508-080-C",
        focal_length_m=80.3e-3,
        effective_focal_length_m=80.3e-3,
        back_focal_length_m=66.9e-3,
        center_thickness_m=20.5e-3,
        mount_length_m=27.7e-3,
        mount_overhang_m=3.9e-3,
        mount_half_height_m=33.0e-3,
    )


def default_mot_apparatus_config() -> MOTApparatusConfig:
    """Return the current default apparatus configuration."""

    return MOTApparatusConfig(
        lens=ac508_080_c_lens(),
        cell=CellGeometry(),
        cooling=MOTBeamConfig(
            name="cooling_780",
            role="780 nm cooling light",
            wavelength_m=COOLING_WAVELENGTH_M,
            resonance_frequency_hz=RB87_COOLING_RESONANCE_HZ,
            detuning_hz=-15.0e6,
            beam_diameter_m=12.7e-3,
            power_w_per_beam=20.0e-3,
        ),
        repump=MOTBeamConfig(
            name="repump_780",
            role="780 nm repump light",
            wavelength_m=REPUMP_WAVELENGTH_M,
            resonance_frequency_hz=RB87_REPUMP_RESONANCE_HZ,
            detuning_hz=0.0,
            beam_diameter_m=12.7e-3,
            power_w_per_beam=0.5e-3,
        ),
        axes=(
            AxisDefinition(
                name="horizontal_x",
                cell_angle_of_incidence_deg=45.0,
                description="first horizontal beam axis entering the cell at 45 degree AOI",
            ),
            AxisDefinition(
                name="horizontal_y",
                cell_angle_of_incidence_deg=45.0,
                description="second horizontal beam axis orthogonal to the first",
            ),
            AxisDefinition(
                name="vertical_z",
                cell_angle_of_incidence_deg=0.0,
                description="vertical beam axis entering from the top of the cell",
            ),
        ),
    )
