"""Project configuration and default apparatus parameters for the pMOT study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PLANCK_CONSTANT_J_S = 6.62607015e-34
HBAR_J_S = 1.054571817e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
RB87_MASS_KG = 86.9091805310 * ATOMIC_MASS_UNIT_KG
STANDARD_GRAVITY_M_PER_S2 = 9.80665
GRAVITY_ACCELERATION_M_PER_S2 = (0.0, 0.0, -STANDARD_GRAVITY_M_PER_S2)

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
    power_w_per_beam: float = 2.0e-3
    propagation_length_m: float = 120e-3

    @property
    def beam_radius_m(self) -> float:
        return 0.5 * self.beam_diameter_m


@dataclass(frozen=True, slots=True)
class AxisDefinition:
    """One physical beam axis in the pMOT apparatus."""

    name: str
    cell_angle_of_incidence_deg: float
    description: str


@dataclass(frozen=True, slots=True)
class PMOTSimulationConfig:
    """Top-level parameter set for the present apparatus model."""

    lens: LensSpec
    cell: CellGeometry
    cooling: MOTBeamConfig
    repump: MOTBeamConfig
    axes: tuple[AxisDefinition, AxisDefinition, AxisDefinition]
    samples_per_axis: int = 201
    volume_extent_m: float = 50e-3


def project_paths(root: Path | None = None) -> dict[str, Path]:
    """Return key project paths used by notebooks and scripts."""

    project_root = root or Path(__file__).resolve().parents[2]
    return {
        "root": project_root,
        "data_raw": project_root / "data" / "raw",
        "data_processed": project_root / "data" / "processed",
        "outputs_fields": project_root / "outputs" / "fields",
        "outputs_trajectories": project_root / "outputs" / "trajectories",
        "outputs_statistics": project_root / "outputs" / "statistics",
        "outputs_figures": project_root / "outputs" / "figures",
        "notebooks": project_root / "notebooks",
    }


def notebook_order() -> list[str]:
    """Return the intended notebook execution order."""

    return [
        "project_setup.ipynb",
        "atomic_and_laser_configuration.ipynb",
        "atomic_data_and_polarizability.ipynb",
        "beam_geometry_and_fields.ipynb",
        "force_model.ipynb",
        "trajectory_sampling.ipynb",
        "multilevel_atom_model.ipynb",
        "capture_statistics.ipynb",
        "loading_rate.ipynb",
        "validation_and_final_results.ipynb",
    ]


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


def default_simulation_config() -> PMOTSimulationConfig:
    """Return the current default apparatus configuration."""

    return PMOTSimulationConfig(
        lens=ac508_080_c_lens(),
        cell=CellGeometry(),
        cooling=MOTBeamConfig(
            name="cooling_780",
            role="780 nm cooling light",
            wavelength_m=COOLING_WAVELENGTH_M,
            resonance_frequency_hz=RB87_COOLING_RESONANCE_HZ,
            detuning_hz=-12.0e6,
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


def describe_configuration(config: PMOTSimulationConfig | None = None) -> dict[str, object]:
    """Return a notebook-friendly configuration summary."""

    apparatus = config or default_simulation_config()
    return {
        "lens": apparatus.lens.name,
        "cooling_beam_diameter_mm": 1e3 * apparatus.cooling.beam_diameter_m,
        "cooling_power_w_per_beam": apparatus.cooling.power_w_per_beam,
        "cooling_detuning_mhz": apparatus.cooling.detuning_hz / 1e6,
        "cooling_resonance_thz": apparatus.cooling.resonance_frequency_hz / 1e12,
        "repump_beam_diameter_mm": 1e3 * apparatus.repump.beam_diameter_m,
        "repump_power_w_per_beam": apparatus.repump.power_w_per_beam,
        "repump_detuning_mhz": apparatus.repump.detuning_hz / 1e6,
        "repump_resonance_thz": apparatus.repump.resonance_frequency_hz / 1e12,
        "axes": [axis.name for axis in apparatus.axes],
        "axis_aoi_deg": {axis.name: axis.cell_angle_of_incidence_deg for axis in apparatus.axes},
        "volume_extent_mm": 1e3 * apparatus.volume_extent_m,
        "samples_per_axis": apparatus.samples_per_axis,
    }
