"""MOT beam construction for the physical multilevel solver."""

from __future__ import annotations

from dataclasses import replace
from math import pi

from ..configuration import RB87_REPUMP_RESONANCE_HZ
from ..fields import MOTBeam, build_mot_beams
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config


_HELICITY_BY_AXIS = {
    "horizontal_x": "sigma+",
    "horizontal_y": "sigma+",
    "vertical_z": "sigma-",
}


def build_multilevel_cooling_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build the six cooling components with propagation-frame helicities."""

    cfg = config or default_multilevel_mot_config()
    return [
        replace(
            beam,
            circular_polarization=_HELICITY_BY_AXIS[beam.axis_name],
            power_w=cfg.cooling_power_w_per_beam,
            wavelength_m=cfg.cooling_wavelength_m,
            detuning_hz=cfg.cooling_detuning_rad_per_s / (2.0 * pi),
        )
        for beam in build_mot_beams(apparatus_config)
        if beam.family == "cooling"
    ]


def build_multilevel_repump_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build six co-propagating repump frequency components."""

    cfg = config or default_multilevel_mot_config()
    return [
        replace(
            beam,
            label=beam.label.replace("_cooling_", "_repump_"),
            family="repump",
            wavelength_m=cfg.repump_wavelength_m,
            resonance_frequency_hz=RB87_REPUMP_RESONANCE_HZ,
            detuning_hz=cfg.repump_detuning_rad_per_s / (2.0 * pi),
            power_w=cfg.repump_power_w_per_beam,
        )
        for beam in build_multilevel_cooling_beams(apparatus_config, cfg)
    ]


def build_multilevel_mot_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Return six cooling and, when enabled, six repump components."""

    cfg = config or default_multilevel_mot_config()
    beams = build_multilevel_cooling_beams(apparatus_config, cfg)
    if cfg.repumper_enabled and cfg.repump_power_w_per_beam > 0.0:
        beams.extend(build_multilevel_repump_beams(apparatus_config, cfg))
    return beams


__all__ = [
    "build_multilevel_cooling_beams",
    "build_multilevel_mot_beams",
    "build_multilevel_repump_beams",
]
