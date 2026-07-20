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

TRAP_TONE_1_WAVELENGTH_M = 1529.376949e-9
TRAP_TONE_2_WAVELENGTH_M = 1529.358429e-9
COOLING_WAVELENGTH_M = 780e-9


@dataclass(frozen=True, slots=True)
class OpticalTone:
    """Named laser tone with a wavelength and relative power split."""

    name: str
    wavelength_m: float
    role: str
    relative_intensity: float = 0.5


@dataclass(frozen=True, slots=True)
class LensSpec:
    """Mounted lens specification in SI units."""

    name: str
    focal_length_m: float
    effective_focal_length_m: float
    back_focal_length_m: float
    center_thickness_m: float
    mount_length_m: float
    mount_overhang_m: float
    mount_half_height_m: float


@dataclass(frozen=True, slots=True)
class TrapBeamGeometry:
    """Shared geometry for the 1529 nm trapping axes."""

    beam_diameter_m: float = 35e-3
    incident_focus_offset_m: float = -10e-3
    retro_focus_offset_m: float = 10e-3
    mirror_gap_m: float = 40e-3
    total_power_w_per_beam_pair: float = 0.5

    @property
    def input_radius_m(self) -> float:
        return 0.5 * self.beam_diameter_m

    @property
    def focus_separation_m(self) -> float:
        return self.retro_focus_offset_m - self.incident_focus_offset_m


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """Glass-cell geometry."""

    outer_diameter_m: float = 30e-3
    wall_thickness_m: float = 5e-3


@dataclass(frozen=True, slots=True)
class CoolingBeamConfig:
    """Cooling-light overlay configuration."""

    wavelength_m: float = COOLING_WAVELENGTH_M
    beam_diameter_m: float = 12.7e-3
    power_w_per_beam: float = 2.0e-3
    extra_propagation_m: float = 120e-3
    detuning_hz: float = -12.0e6

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
    """Top-level parameter set for the current apparatus model."""

    lens: LensSpec
    trap_beam: TrapBeamGeometry
    cell: CellGeometry
    cooling: CoolingBeamConfig
    trap_tones: tuple[OpticalTone, OpticalTone]
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
        trap_beam=TrapBeamGeometry(),
        cell=CellGeometry(),
        cooling=CoolingBeamConfig(),
        trap_tones=(
            OpticalTone(
                name="trap_tone_1",
                wavelength_m=TRAP_TONE_1_WAVELENGTH_M,
                role="1529 nm trapping tone",
                relative_intensity=0.492762,
            ),
            OpticalTone(
                name="trap_tone_2",
                wavelength_m=TRAP_TONE_2_WAVELENGTH_M,
                role="1529 nm trapping tone",
                relative_intensity=0.507238,
            ),
        ),
        axes=(
            AxisDefinition(
                name="oblique_x",
                cell_angle_of_incidence_deg=45.0,
                description="first 45 degree axis through the cell",
            ),
            AxisDefinition(
                name="oblique_y",
                cell_angle_of_incidence_deg=45.0,
                description="second 45 degree axis through the cell",
            ),
            AxisDefinition(
                name="normal_z",
                cell_angle_of_incidence_deg=0.0,
                description="axis normal to the two oblique directions",
            ),
        ),
    )


def describe_configuration(config: PMOTSimulationConfig | None = None) -> dict[str, object]:
    """Return a notebook-friendly configuration summary."""

    apparatus = config or default_simulation_config()
    return {
        "lens": apparatus.lens.name,
        "trap_beam_diameter_mm": 1e3 * apparatus.trap_beam.beam_diameter_m,
        "trap_incident_focus_offset_mm": 1e3 * apparatus.trap_beam.incident_focus_offset_m,
        "trap_retro_focus_offset_mm": 1e3 * apparatus.trap_beam.retro_focus_offset_m,
        "trap_total_power_w_per_beam_pair": apparatus.trap_beam.total_power_w_per_beam_pair,
        "cooling_beam_diameter_mm": 1e3 * apparatus.cooling.beam_diameter_m,
        "cooling_power_w_per_beam": apparatus.cooling.power_w_per_beam,
        "cooling_detuning_mhz": apparatus.cooling.detuning_hz / 1e6,
        "trap_tone_wavelengths_nm": [1e9 * tone.wavelength_m for tone in apparatus.trap_tones],
        "trap_tone_relative_intensities": [tone.relative_intensity for tone in apparatus.trap_tones],
        "axes": [axis.name for axis in apparatus.axes],
        "volume_extent_mm": 1e3 * apparatus.volume_extent_m,
        "samples_per_axis": apparatus.samples_per_axis,
    }
