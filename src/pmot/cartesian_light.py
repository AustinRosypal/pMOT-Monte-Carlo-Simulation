"""Shared Cartesian cooling/repump geometry with model-owned parameters.

This module centralizes only the construction of the six Cartesian traveling
directions and their cooling/repump metadata.  Each simulation supplies its
own powers, detunings, wavelengths, and helicities through an explicit
configuration object; sharing geometry must not silently make the models share
their physical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from .configuration import MOTApparatusConfig
from .fields import MOTBeam, build_mot_beams


_AXIS_NAMES = ("horizontal_x", "horizontal_y", "vertical_z")
_VALID_POLARIZATIONS = {"sigma+", "sigma-", "pi"}


@dataclass(frozen=True, slots=True)
class CartesianLightParameters:
    """Model-specific parameters for shared Cartesian beam geometry."""

    cooling_power_w_per_beam: float
    cooling_detuning_hz: float
    cooling_wavelength_m: float
    cooling_resonance_frequency_hz: float
    repump_power_w_per_beam: float
    repump_detuning_hz: float
    repump_wavelength_m: float
    repump_resonance_frequency_hz: float
    cooling_helicities_by_axis: tuple[str, str, str]
    repump_helicities_by_axis: tuple[str, str, str]

    def __post_init__(self) -> None:
        powers = (self.cooling_power_w_per_beam, self.repump_power_w_per_beam)
        if any(not isfinite(power) or power < 0.0 for power in powers):
            raise ValueError("cooling and repump powers must be finite and non-negative")
        detunings = (self.cooling_detuning_hz, self.repump_detuning_hz)
        if any(not isfinite(detuning) for detuning in detunings):
            raise ValueError("cooling and repump detunings must be finite")
        wavelengths = (self.cooling_wavelength_m, self.repump_wavelength_m)
        if any(not isfinite(wavelength) or wavelength <= 0.0 for wavelength in wavelengths):
            raise ValueError("cooling and repump wavelengths must be finite and positive")
        resonances = (
            self.cooling_resonance_frequency_hz,
            self.repump_resonance_frequency_hz,
        )
        if any(not isfinite(resonance) or resonance <= 0.0 for resonance in resonances):
            raise ValueError("cooling and repump resonances must be finite and positive")
        for name, values in (
            ("cooling_helicities_by_axis", self.cooling_helicities_by_axis),
            ("repump_helicities_by_axis", self.repump_helicities_by_axis),
        ):
            if len(values) != 3 or any(value not in _VALID_POLARIZATIONS for value in values):
                raise ValueError(
                    f"{name} must contain three 'sigma+', 'sigma-', or 'pi' labels"
                )


@dataclass(frozen=True, slots=True)
class CartesianLightBundle:
    """The six cooling and six repump traveling components."""

    cooling_beams: tuple[MOTBeam, ...]
    repump_beams: tuple[MOTBeam, ...]

    @property
    def all_beams(self) -> tuple[MOTBeam, ...]:
        return self.cooling_beams + self.repump_beams


def build_cartesian_light_bundle(
    apparatus_config: MOTApparatusConfig,
    parameters: CartesianLightParameters,
) -> CartesianLightBundle:
    """Build shared Cartesian geometry using explicitly model-owned settings."""

    cooling_helicity = dict(zip(_AXIS_NAMES, parameters.cooling_helicities_by_axis))
    repump_helicity = dict(zip(_AXIS_NAMES, parameters.repump_helicities_by_axis))
    cooling: list[MOTBeam] = []
    repump: list[MOTBeam] = []
    for beam in build_mot_beams(apparatus_config):
        if beam.family == "cooling":
            cooling.append(
                replace(
                    beam,
                    circular_polarization=cooling_helicity[beam.axis_name],
                    power_w=parameters.cooling_power_w_per_beam,
                    wavelength_m=parameters.cooling_wavelength_m,
                    resonance_frequency_hz=parameters.cooling_resonance_frequency_hz,
                    detuning_hz=parameters.cooling_detuning_hz,
                )
            )
        elif beam.family == "repump":
            repump.append(
                replace(
                    beam,
                    circular_polarization=repump_helicity[beam.axis_name],
                    power_w=parameters.repump_power_w_per_beam,
                    wavelength_m=parameters.repump_wavelength_m,
                    resonance_frequency_hz=parameters.repump_resonance_frequency_hz,
                    detuning_hz=parameters.repump_detuning_hz,
                )
            )
    if len(cooling) != 6 or len(repump) != 6:
        raise ValueError("Cartesian light geometry must contain six beams per family")
    return CartesianLightBundle(tuple(cooling), tuple(repump))


__all__ = [
    "CartesianLightBundle",
    "CartesianLightParameters",
    "build_cartesian_light_bundle",
]
