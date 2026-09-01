"""Focused trapping-light geometry for the pseudo-MOT.

This module describes only the optical envelope and the propagation/helicity
metadata.  It deliberately does not calculate AC Stark shifts, forces, or
trajectories.  The single configured trapping-laser frequency is routed onto
three Cartesian round-trip paths, giving an incident and retroreflected
traveling-wave component on each path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from ..beams import Vec3
from ..beams import axis_direction_from_name
from ..beams import focused_waist_radius
from ..beams import normalize
from ..beams import scale


DEFAULT_TRAPPING_WAVELENGTH_M = 1529.268881e-9
DEFAULT_TRAPPING_AXES = ("horizontal_x", "horizontal_y", "vertical_z")
SUPPORTED_HELICITIES = ("sigma+", "sigma-", "pi")


def helicity_sign(helicity: str) -> float:
    """Return the sign of ``i E* x E`` along the component wavevector.

    This matches ``mot_multilevel.polarization``: propagation-frame ``sigma+``
    has ``i E* x E = -I k_hat`` and ``sigma-`` has the opposite sign.
    """

    if helicity == "sigma+":
        return -1.0
    if helicity == "sigma-":
        return 1.0
    if helicity == "pi":
        return 0.0
    raise ValueError("helicity must be 'sigma+', 'sigma-', or 'pi'")


@dataclass(frozen=True, slots=True)
class TrappingLaserConfig:
    """One trapping-laser frequency and its symmetric in-trap geometry.

    The default waist is the ideal diffraction-limited waist obtained from a
    35 mm diameter collimated input and an 80.3 mm focal-length lens.  Absolute
    power has intentionally not been assigned.  Intensities from this module
    are responses per watt incident on one Cartesian path; a later physics
    configuration must provide the laser power and its three-path split.

    ``incident_helicity`` and ``retro_helicity`` use the repository convention:
    each label is defined while looking along that component's own propagation
    direction.  Equal labels therefore produce opposite lab-frame vector
    directions for the counterpropagating components.
    """

    wavelength_m: float = DEFAULT_TRAPPING_WAVELENGTH_M
    incident_helicity: str = "sigma+"
    retro_helicity: str = "sigma+"
    focus_offset_m: float = 10.0e-3
    input_beam_diameter_m: float = 35.0e-3
    focal_length_m: float = 80.3e-3
    incident_waist_radius_m: float | None = None
    retro_waist_radius_m: float | None = None
    retro_power_fraction: float = 1.0
    axes: tuple[str, ...] = DEFAULT_TRAPPING_AXES
    envelope_combination: str = "incoherent"

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        helicity_sign(self.incident_helicity)
        helicity_sign(self.retro_helicity)
        if self.focus_offset_m <= 0.0:
            raise ValueError("focus_offset_m must be positive")
        if self.input_beam_diameter_m <= 0.0:
            raise ValueError("input_beam_diameter_m must be positive")
        if self.focal_length_m <= 0.0:
            raise ValueError("focal_length_m must be positive")
        if self.incident_waist_radius_m is not None and self.incident_waist_radius_m <= 0.0:
            raise ValueError("incident_waist_radius_m must be positive when supplied")
        if self.retro_waist_radius_m is not None and self.retro_waist_radius_m <= 0.0:
            raise ValueError("retro_waist_radius_m must be positive when supplied")
        if self.retro_power_fraction < 0.0:
            raise ValueError("retro_power_fraction must be non-negative")
        if len(self.axes) != 3 or len(set(self.axes)) != 3:
            raise ValueError("axes must contain three distinct Cartesian paths")
        for axis_name in self.axes:
            axis_direction_from_name(axis_name)
        if self.envelope_combination != "incoherent":
            raise ValueError(
                "only the standing-wave-averaged 'incoherent' envelope is implemented"
            )

    @property
    def input_beam_radius_m(self) -> float:
        return 0.5 * self.input_beam_diameter_m

    @property
    def resolved_incident_waist_radius_m(self) -> float:
        if self.incident_waist_radius_m is not None:
            return self.incident_waist_radius_m
        return focused_waist_radius(
            self.wavelength_m,
            self.focal_length_m,
            self.input_beam_radius_m,
        )

    @property
    def resolved_retro_waist_radius_m(self) -> float:
        if self.retro_waist_radius_m is not None:
            return self.retro_waist_radius_m
        return self.resolved_incident_waist_radius_m


@dataclass(frozen=True, slots=True)
class TrappingBeam:
    """One traveling component of the focused trapping-light envelope."""

    label: str
    axis_name: str
    propagation_sense: str
    helicity: str
    direction: Vec3
    waist_position_m: Vec3
    wavelength_m: float
    waist_radius_m: float
    power_per_incident_path_watt: float

    def __post_init__(self) -> None:
        if self.propagation_sense not in {"incident", "retro"}:
            raise ValueError("propagation_sense must be 'incident' or 'retro'")
        helicity_sign(self.helicity)
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")
        if self.waist_radius_m <= 0.0:
            raise ValueError("waist_radius_m must be positive")
        if self.power_per_incident_path_watt < 0.0:
            raise ValueError("power_per_incident_path_watt must be non-negative")
        object.__setattr__(self, "direction", normalize(self.direction))

    @property
    def rayleigh_range_m(self) -> float:
        return pi * self.waist_radius_m**2 / self.wavelength_m


def build_trapping_beams(
    config: TrappingLaserConfig | None = None,
) -> list[TrappingBeam]:
    """Build the six traveling components from one trapping-laser config."""

    cfg = config or TrappingLaserConfig()
    beams: list[TrappingBeam] = []
    for axis_name in cfg.axes:
        direction = axis_direction_from_name(axis_name)
        beams.append(
            TrappingBeam(
                label=f"{axis_name}_incident_trapping",
                axis_name=axis_name,
                propagation_sense="incident",
                helicity=cfg.incident_helicity,
                direction=direction,
                waist_position_m=scale(-cfg.focus_offset_m, direction),
                wavelength_m=cfg.wavelength_m,
                waist_radius_m=cfg.resolved_incident_waist_radius_m,
                power_per_incident_path_watt=1.0,
            )
        )
        beams.append(
            TrappingBeam(
                label=f"{axis_name}_retro_trapping",
                axis_name=axis_name,
                propagation_sense="retro",
                helicity=cfg.retro_helicity,
                direction=scale(-1.0, direction),
                waist_position_m=scale(cfg.focus_offset_m, direction),
                wavelength_m=cfg.wavelength_m,
                waist_radius_m=cfg.resolved_retro_waist_radius_m,
                power_per_incident_path_watt=cfg.retro_power_fraction,
            )
        )
    return beams


def _positions_array(positions_m) -> tuple[np.ndarray, bool]:
    array = np.asarray(positions_m, dtype=float)
    scalar_input = array.shape == (3,)
    if scalar_input:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions_m must have shape (3,) or (n, 3)")
    return array, scalar_input


def trapping_beam_intensity_response_m_inv2(
    beam: TrappingBeam,
    positions_m,
):
    """Return intensity divided by incident path power, in inverse metres squared."""

    positions, scalar_input = _positions_array(positions_m)
    waist = np.asarray(beam.waist_position_m, dtype=float)
    direction = np.asarray(beam.direction, dtype=float)
    relative = positions - waist
    axial_m = relative @ direction
    radial_vector = relative - axial_m[:, np.newaxis] * direction
    radial_squared_m2 = np.einsum("ij,ij->i", radial_vector, radial_vector)
    local_waist_m = beam.waist_radius_m * np.sqrt(
        1.0 + (axial_m / beam.rayleigh_range_m) ** 2
    )
    response = (
        2.0
        * beam.power_per_incident_path_watt
        / (pi * local_waist_m**2)
        * np.exp(-2.0 * radial_squared_m2 / local_waist_m**2)
    )
    return float(response[0]) if scalar_input else response


def total_trapping_intensity_response_m_inv2(
    beams: list[TrappingBeam],
    positions_m,
):
    """Return the incoherent scalar-intensity response of all components."""

    positions, scalar_input = _positions_array(positions_m)
    response = np.zeros(len(positions), dtype=float)
    for beam in beams:
        response += trapping_beam_intensity_response_m_inv2(beam, positions)
    return float(response[0]) if scalar_input else response


def vector_intensity_response_m_inv2(
    beams: list[TrappingBeam],
    positions_m,
):
    """Return ``i E* x E``'s intensity factor per incident-path watt.

    In intensity units this is ``sum(s_helicity * I * k_hat)``, with
    ``s_helicity=-1`` for propagation-frame ``sigma+`` and ``+1`` for
    ``sigma-``.  It is the purely optical geometry factor entering the vector
    AC Stark shift, not itself an effective magnetic field; the state-dependent
    polarizability and ``F g_F mu_B`` factors are deferred to the later physics
    model.
    """

    positions, scalar_input = _positions_array(positions_m)
    response = np.zeros((len(positions), 3), dtype=float)
    for beam in beams:
        scalar_response = trapping_beam_intensity_response_m_inv2(beam, positions)
        response += (
            helicity_sign(beam.helicity)
            * scalar_response[:, np.newaxis]
            * np.asarray(beam.direction, dtype=float)
        )
    return tuple(float(value) for value in response[0]) if scalar_input else response


def beams_for_trapping_axis(
    beams: list[TrappingBeam],
    axis_name: str,
) -> list[TrappingBeam]:
    selected = [beam for beam in beams if beam.axis_name == axis_name]
    if len(selected) != 2:
        raise ValueError(f"expected one incident/retro pair for axis '{axis_name}'")
    return selected
