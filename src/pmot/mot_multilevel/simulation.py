"""MOT beam construction for the physical multilevel solver."""

from __future__ import annotations

from math import pi

from ..cartesian_light import CartesianLightParameters
from ..cartesian_light import build_cartesian_light_bundle
from ..configuration import RB87_COOLING_RESONANCE_HZ
from ..configuration import RB87_REPUMP_RESONANCE_HZ
from ..configuration import default_mot_apparatus_config
from ..fields import MOTBeam
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config


_HELICITY_BY_AXIS = {
    "horizontal_x": "sigma+",
    "horizontal_y": "sigma+",
    "vertical_z": "sigma-",
}


def _build_multilevel_light_bundle(apparatus_config, config: MultilevelMOTConfig):
    apparatus = apparatus_config or default_mot_apparatus_config()
    helicities = tuple(
        _HELICITY_BY_AXIS[name]
        for name in ("horizontal_x", "horizontal_y", "vertical_z")
    )
    return build_cartesian_light_bundle(
        apparatus,
        CartesianLightParameters(
            cooling_power_w_per_beam=config.cooling_power_w_per_beam,
            cooling_detuning_hz=config.cooling_detuning_rad_per_s / (2.0 * pi),
            cooling_wavelength_m=config.cooling_wavelength_m,
            cooling_resonance_frequency_hz=RB87_COOLING_RESONANCE_HZ,
            repump_power_w_per_beam=config.repump_power_w_per_beam,
            repump_detuning_hz=config.repump_detuning_rad_per_s / (2.0 * pi),
            repump_wavelength_m=config.repump_wavelength_m,
            repump_resonance_frequency_hz=RB87_REPUMP_RESONANCE_HZ,
            cooling_helicities_by_axis=helicities,
            repump_helicities_by_axis=helicities,
        ),
    )


def build_multilevel_cooling_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build the six cooling components with propagation-frame helicities."""

    cfg = config or default_multilevel_mot_config()
    return list(_build_multilevel_light_bundle(apparatus_config, cfg).cooling_beams)


def build_multilevel_repump_beams(
    apparatus_config=None,
    config: MultilevelMOTConfig | None = None,
) -> list[MOTBeam]:
    """Build six co-propagating repump frequency components."""

    cfg = config or default_multilevel_mot_config()
    return list(_build_multilevel_light_bundle(apparatus_config, cfg).repump_beams)


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
