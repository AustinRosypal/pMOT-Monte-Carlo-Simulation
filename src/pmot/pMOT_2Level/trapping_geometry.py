"""Pointwise trapping-light geometry for the effective two-level pMOT.

The implementation reuses the validated ideal Gaussian envelopes in
``pmot.mot_strange.trapping_beams``.  It adds the physical power scaling and a compact
pointwise interface needed by the future two-level AC-Stark model.

No polarizability data are read here.  The optical-spin vector returned by this
module is the geometry factor

    sum_i s_i I_i(r) k_hat_i,

not an effective magnetic field.  The later AC-Stark layer must supply the
polarizability and atomic-state factors that map this vector to an energy shift
or fictitious magnetic field.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Sequence

import numpy as np

from ..mot_strange.trapping_beams import DEFAULT_TRAPPING_AXES
from ..mot_strange.trapping_beams import TrappingBeam
from ..mot_strange.trapping_beams import TrappingLaserConfig
from ..mot_strange.trapping_beams import build_trapping_beams
from ..mot_strange.trapping_beams import helicity_sign
from ..mot_strange.trapping_beams import trapping_beam_intensity_response_m_inv2


DEFAULT_INCIDENT_HELICITIES_BY_AXIS = ("sigma+", "sigma+", "sigma-")
DEFAULT_RETRO_HELICITIES_BY_AXIS = ("sigma+", "sigma+", "sigma-")
DEFAULT_FALLBACK_QUANTIZATION_AXIS = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class PMOT2LevelGeometrySample:
    """All trapping-light geometry evaluated at one position.

    ``component_k_hat`` is the beamwise propagation direction and therefore
    provides each component's natural propagation-frame quantization axis.
    ``quantization_axis`` is instead the local axis of the summed optical-spin
    vector.  At a symmetry zero that vector has no physical direction, so the
    configured fallback is returned and
    ``quantization_axis_from_optical_spin`` is false.
    """

    position_m: tuple[float, float, float]
    component_labels: tuple[str, ...]
    component_intensities_w_per_m2: tuple[float, ...]
    component_k_hat: tuple[tuple[float, float, float], ...]
    component_helicities: tuple[str, ...]
    total_intensity_w_per_m2: float
    optical_spin_vector_w_per_m2: tuple[float, float, float]
    optical_spin_magnitude_w_per_m2: float
    quantization_axis: tuple[float, float, float]
    quantization_axis_from_optical_spin: bool


def _finite_vector3(value, *, name: str, nonzero: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if nonzero and float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector


def _path_powers_by_axis(path_powers_w: Sequence[float]) -> dict[str, float]:
    powers = np.asarray(path_powers_w, dtype=float)
    if powers.shape != (3,) or not np.all(np.isfinite(powers)):
        raise ValueError("path_powers_w must contain three finite x, y, z powers")
    if np.any(powers < 0.0):
        raise ValueError("path_powers_w must be non-negative")
    return dict(zip(DEFAULT_TRAPPING_AXES, (float(value) for value in powers)))


def _validate_helicities(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    helicities = tuple(values)
    if len(helicities) != 3:
        raise ValueError(f"{name} must contain x, y, and z helicities")
    for helicity in helicities:
        helicity_sign(helicity)
    return helicities


def build_pmot_2level_trapping_beams(
    laser_config: TrappingLaserConfig | None = None,
    *,
    incident_helicities_by_axis: Sequence[str] = DEFAULT_INCIDENT_HELICITIES_BY_AXIS,
    retro_helicities_by_axis: Sequence[str] = DEFAULT_RETRO_HELICITIES_BY_AXIS,
) -> tuple[TrappingBeam, ...]:
    """Build the six pMOT trapping components without multilevel dependencies.

    The beam envelopes, 1529.268881-nm default wavelength, +/-10-mm waist
    locations, and propagation directions come directly from the existing pMOT
    geometry.  The default x/y/z propagation-frame helicities are the
    provisional centered matched-path ``(sigma+, sigma+, sigma-)`` tuple.
    """

    config = laser_config or TrappingLaserConfig()
    if tuple(config.axes) != DEFAULT_TRAPPING_AXES:
        raise ValueError(
            "the pMOT_2Level map currently requires the Cartesian x, y, z path order"
        )
    incident_values = _validate_helicities(
        incident_helicities_by_axis,
        name="incident_helicities_by_axis",
    )
    retro_values = _validate_helicities(
        retro_helicities_by_axis,
        name="retro_helicities_by_axis",
    )
    incident = dict(zip(config.axes, incident_values))
    retro = dict(zip(config.axes, retro_values))
    return tuple(
        replace(
            beam,
            helicity=(
                incident[beam.axis_name]
                if beam.propagation_sense == "incident"
                else retro[beam.axis_name]
            ),
        )
        for beam in build_trapping_beams(config)
    )


def component_propagation_directions(
    beams: Sequence[TrappingBeam],
) -> np.ndarray:
    """Return the beamwise propagation axes ``k_hat_i`` with shape ``(n, 3)``."""

    directions = np.asarray([beam.direction for beam in beams], dtype=float)
    if directions.ndim != 2 or directions.shape[1:] != (3,):
        raise ValueError("beams must contain at least one valid trapping component")
    return directions


def component_intensities_w_per_m2(
    position_m,
    beams: Sequence[TrappingBeam],
    *,
    path_powers_w: Sequence[float],
) -> np.ndarray:
    """Evaluate all component intensities at one 3D point.

    ``path_powers_w`` gives the incident powers launched on the x, y, and z
    paths.  Each retro component is additionally scaled by the
    ``retro_power_fraction`` stored in its shared geometry configuration.
    """

    point = _finite_vector3(position_m, name="position_m")
    powers = _path_powers_by_axis(path_powers_w)
    return np.asarray(
        [
            powers[beam.axis_name]
            * trapping_beam_intensity_response_m_inv2(beam, point)
            for beam in beams
        ],
        dtype=float,
    )


def total_intensity_w_per_m2(
    position_m,
    beams: Sequence[TrappingBeam],
    *,
    path_powers_w: Sequence[float],
) -> float:
    """Return the standing-wave-averaged unsigned intensity at one 3D point."""

    return float(
        np.sum(
            component_intensities_w_per_m2(
                position_m,
                beams,
                path_powers_w=path_powers_w,
            )
        )
    )


def optical_spin_vector_w_per_m2(
    position_m,
    beams: Sequence[TrappingBeam],
    *,
    path_powers_w: Sequence[float],
) -> np.ndarray:
    """Return ``sum_i s_i I_i k_hat_i``, the vector-shift geometry factor."""

    intensities = component_intensities_w_per_m2(
        position_m,
        beams,
        path_powers_w=path_powers_w,
    )
    directions = component_propagation_directions(beams)
    signs = np.asarray([helicity_sign(beam.helicity) for beam in beams], dtype=float)
    return np.sum((signs * intensities)[:, np.newaxis] * directions, axis=0)


def quantization_axis_at_point(
    position_m,
    beams: Sequence[TrappingBeam],
    *,
    path_powers_w: Sequence[float],
    fallback_axis=DEFAULT_FALLBACK_QUANTIZATION_AXIS,
    relative_zero_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, bool]:
    """Return the local optical-spin axis and whether the field defines it.

    The boolean is false at a vector-field zero, where no physical local axis
    exists and the normalized ``fallback_axis`` is returned.  This makes the
    numerical basis available everywhere while preserving the physical
    degeneracy for future nonadiabatic or basis-zero handling.
    """

    if not np.isfinite(relative_zero_tolerance) or relative_zero_tolerance < 0.0:
        raise ValueError("relative_zero_tolerance must be finite and non-negative")
    fallback = _finite_vector3(
        fallback_axis,
        name="fallback_axis",
        nonzero=True,
    )
    vector = optical_spin_vector_w_per_m2(
        position_m,
        beams,
        path_powers_w=path_powers_w,
    )
    magnitude = float(np.linalg.norm(vector))
    total = total_intensity_w_per_m2(
        position_m,
        beams,
        path_powers_w=path_powers_w,
    )
    if magnitude <= relative_zero_tolerance * total:
        return fallback / np.linalg.norm(fallback), False
    return vector / magnitude, True


def evaluate_trapping_geometry(
    position_m,
    beams: Sequence[TrappingBeam] | None = None,
    *,
    path_powers_w: Sequence[float],
    fallback_axis=DEFAULT_FALLBACK_QUANTIZATION_AXIS,
    relative_zero_tolerance: float = 1.0e-12,
) -> PMOT2LevelGeometrySample:
    """Evaluate intensity, every ``k_hat_i``, and the local axis at one point."""

    resolved_beams = tuple(beams or build_pmot_2level_trapping_beams())
    if not resolved_beams:
        raise ValueError("beams must contain at least one trapping component")
    point = _finite_vector3(position_m, name="position_m")
    intensities = component_intensities_w_per_m2(
        point,
        resolved_beams,
        path_powers_w=path_powers_w,
    )
    directions = component_propagation_directions(resolved_beams)
    spin = optical_spin_vector_w_per_m2(
        point,
        resolved_beams,
        path_powers_w=path_powers_w,
    )
    axis, defined = quantization_axis_at_point(
        point,
        resolved_beams,
        path_powers_w=path_powers_w,
        fallback_axis=fallback_axis,
        relative_zero_tolerance=relative_zero_tolerance,
    )
    return PMOT2LevelGeometrySample(
        position_m=tuple(float(value) for value in point),
        component_labels=tuple(beam.label for beam in resolved_beams),
        component_intensities_w_per_m2=tuple(float(value) for value in intensities),
        component_k_hat=tuple(
            tuple(float(value) for value in direction) for direction in directions
        ),
        component_helicities=tuple(beam.helicity for beam in resolved_beams),
        total_intensity_w_per_m2=float(np.sum(intensities)),
        optical_spin_vector_w_per_m2=tuple(float(value) for value in spin),
        optical_spin_magnitude_w_per_m2=float(np.linalg.norm(spin)),
        quantization_axis=tuple(float(value) for value in axis),
        quantization_axis_from_optical_spin=defined,
    )
