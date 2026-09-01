"""pMOT-specific apparatus configuration, paths, and workflow summaries."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from ..beams import Vec3
from ..configuration import MOTApparatusConfig
from ..configuration import default_mot_apparatus_config
from ..fields import MOTBeam
from ..mot_multilevel.configuration import MultilevelMOTConfig
from ..mot_multilevel.configuration import default_multilevel_mot_config
from ..mot_multilevel.simulation import build_multilevel_mot_beams
from .trapping_beams import DEFAULT_TRAPPING_WAVELENGTH_M
from .trapping_beams import TrappingLaserConfig
from .trapping_beams import build_trapping_beams


@dataclass(frozen=True, slots=True)
class PMOTApparatusConfig:
    """Geometry-only pMOT apparatus with no real magnetic-field hardware.

    The absence of a coil field is structural: this type has no coil
    configuration, and its external magnetic field is identically zero.
    """

    mot_light: MOTApparatusConfig
    trapping_laser: TrappingLaserConfig

    @property
    def external_magnetic_field_t(self) -> Vec3:
        return (0.0, 0.0, 0.0)

    @property
    def anti_helmholtz_coils_present(self) -> bool:
        return False


def default_pmot_apparatus_config(
    *,
    trapping_wavelength_m: float = DEFAULT_TRAPPING_WAVELENGTH_M,
    incident_trapping_helicity: str = "sigma+",
    retro_trapping_helicity: str | None = None,
    cooling_power_w_per_beam: float = 27.0e-3,
    repump_power_w_per_beam: float = 0.1e-3,
) -> PMOTApparatusConfig:
    """Return the first geometry-stage pMOT apparatus configuration.

    Cooling/repump powers retain the most recent full multilevel-MOT baseline.
    Trapping power is intentionally absent until its distribution among the
    three paths is specified; trapping intensities are therefore normalized per
    watt incident on a path.
    """

    mot = default_mot_apparatus_config()
    mot = replace(
        mot,
        cooling=replace(mot.cooling, power_w_per_beam=cooling_power_w_per_beam),
        repump=replace(mot.repump, power_w_per_beam=repump_power_w_per_beam),
    )
    return PMOTApparatusConfig(
        mot_light=mot,
        trapping_laser=TrappingLaserConfig(
            wavelength_m=trapping_wavelength_m,
            incident_helicity=incident_trapping_helicity,
            retro_helicity=(
                incident_trapping_helicity
                if retro_trapping_helicity is None
                else retro_trapping_helicity
            ),
        ),
    )


def build_pmot_cooling_and_repump_beams(
    config: PMOTApparatusConfig | None = None,
    multilevel_config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build the retained six cooling and six repump traveling components.

    This delegates to the authoritative multilevel-MOT builder so every beam
    property, including the 780.232684-nm repump wavelength, is reused exactly.
    """

    apparatus = config or default_pmot_apparatus_config()
    if multilevel_config is None:
        rate_config = replace(
            default_multilevel_mot_config(),
            repumper_enabled=True,
            repump_power_w_per_beam=apparatus.mot_light.repump.power_w_per_beam,
        )
    else:
        rate_config = multilevel_config
    if not rate_config.repumper_enabled or rate_config.repump_power_w_per_beam <= 0.0:
        raise ValueError(
            "pMOT retained-light geometry requires the multilevel repumper"
        )
    return build_multilevel_mot_beams(
        apparatus_config=apparatus.mot_light,
        config=rate_config,
    )


def pmot_paths(root: Path | None = None) -> dict[str, Path]:
    """Return paths owned by the future pMOT model branch."""

    project_root = root or Path(__file__).resolve().parents[3]
    return {
        "root": project_root,
        "data_raw": project_root / "data" / "raw" / "pmot",
        "data_processed": project_root / "data" / "processed" / "pmot",
        "outputs_fields": project_root / "outputs" / "fields" / "pmot",
        "outputs_trajectories": project_root / "outputs" / "trajectories" / "pmot",
        "outputs_statistics": project_root / "outputs" / "statistics" / "pmot",
        "outputs_figures": project_root / "outputs" / "figures" / "pmot",
        "notebooks": project_root / "notebooks" / "pmot",
    }


def pmot_notebook_order() -> list[str]:
    """Return the intended pMOT notebook execution order."""

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


def describe_pmot_configuration(
    config: PMOTApparatusConfig | MOTApparatusConfig | None = None,
) -> dict[str, object]:
    """Return a notebook-friendly summary of the current pMOT apparatus."""

    if config is None:
        pmot = default_pmot_apparatus_config()
    elif isinstance(config, PMOTApparatusConfig):
        pmot = config
    else:
        pmot = PMOTApparatusConfig(
            mot_light=config,
            trapping_laser=TrappingLaserConfig(),
        )
    apparatus = pmot.mot_light
    mot_beams = build_pmot_cooling_and_repump_beams(pmot)
    trapping_beams = build_trapping_beams(pmot.trapping_laser)
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
        "axis_aoi_deg": {
            axis.name: axis.cell_angle_of_incidence_deg for axis in apparatus.axes
        },
        "volume_extent_mm": 1e3 * apparatus.volume_extent_m,
        "samples_per_axis": apparatus.samples_per_axis,
        "cooling_component_count": sum(beam.family == "cooling" for beam in mot_beams),
        "repump_component_count": sum(beam.family == "repump" for beam in mot_beams),
        "trapping_laser_count": 1,
        "trapping_path_count": len(pmot.trapping_laser.axes),
        "trapping_component_count": len(trapping_beams),
        "trapping_wavelength_nm": 1e9 * pmot.trapping_laser.wavelength_m,
        "trapping_incident_helicity": pmot.trapping_laser.incident_helicity,
        "trapping_retro_helicity": pmot.trapping_laser.retro_helicity,
        "trapping_focus_positions_mm": [
            -1e3 * pmot.trapping_laser.focus_offset_m,
            1e3 * pmot.trapping_laser.focus_offset_m,
        ],
        "trapping_waist_radius_um": (
            1e6 * pmot.trapping_laser.resolved_incident_waist_radius_m
        ),
        "trapping_intensity_normalization": "per watt incident on each Cartesian path",
        "trapping_envelope_combination": pmot.trapping_laser.envelope_combination,
        "external_magnetic_field_t": list(pmot.external_magnetic_field_t),
        "anti_helmholtz_coils_present": pmot.anti_helmholtz_coils_present,
    }
