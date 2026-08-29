"""pMOT-specific paths, notebook workflow, and configuration summaries."""

from __future__ import annotations

from pathlib import Path

from ..configuration import MOTApparatusConfig
from ..configuration import default_mot_apparatus_config


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
    config: MOTApparatusConfig | None = None,
) -> dict[str, object]:
    """Return a notebook-friendly summary of the current optical apparatus."""

    apparatus = config or default_mot_apparatus_config()
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
    }
