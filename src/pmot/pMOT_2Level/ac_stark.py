"""Differential AC-Stark shift for the effective F=2 -> F'=3 pMOT.

The Arora CCSD CSV contains effective excited-minus-ground
(``5P3/2 - 5S1/2``) *differential* scalar, vector, and tensor polarizabilities
for the cooling-transition reference. It does not
contain separate level-resolved polarizabilities or a complete m_F-resolved
Hamiltonian.  This module therefore implements an explicitly effective
two-manifold closure and keeps all frequency-like outputs in ordinary hertz,
matching :mod:`pmot.mot_simple`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M
from ..mot_multilevel.polarization import propagation_frame_polarization
from ..mot_strange.polarizability import DifferentialPolarizabilityTable
from ..mot_strange.polarizability import default_polarizability_csv_path
from ..mot_strange.polarizability import interpolate_differential_polarizability_arrays
from ..mot_strange.polarizability import load_differential_polarizability_table
from ..mot_strange.trapping_beams import TrappingBeam
from ..mot_strange.trapping_beams import helicity_sign
from .trapping_geometry import DEFAULT_FALLBACK_QUANTIZATION_AXIS
from .trapping_geometry import component_intensities_w_per_m2


EFFECTIVE_TRANSITION_LABEL = "5S1/2 F=2 -> 5P3/2 F'=3"
ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION = (
    "delta_nu_eff[b,q] = delta_nu_0[b] - dot(k_b,v)/(2*pi) "
    "- DeltaE_AC[q]/h"
)


@dataclass(frozen=True, slots=True)
class PMOT2LevelStarkSample:
    """Differential Stark observables at one external phase-space point."""

    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    trapping_component_labels: tuple[str, ...]
    lab_wavelengths_nm: tuple[float, ...]
    seen_wavelengths_nm: tuple[float, ...]
    component_intensities_w_per_m2: tuple[float, ...]
    component_field_squared_v2_per_m2: tuple[float, ...]
    scalar_polarizabilities_si: tuple[float, ...]
    vector_polarizabilities_si: tuple[float, ...]
    tensor_polarizabilities_si: tuple[float, ...]
    component_vector_energy_vectors_j: tuple[tuple[float, float, float], ...]
    net_vector_energy_vector_j: tuple[float, float, float]
    quantization_axis: tuple[float, float, float]
    quantization_axis_from_vector_shift: bool
    component_vector_angular_factors: tuple[float, ...]
    component_tensor_angular_factors: tuple[float, ...]
    component_scalar_energies_j: tuple[float, ...]
    component_vector_energies_j: tuple[float, ...]
    component_tensor_energies_j: tuple[float, ...]
    total_scalar_energy_j: float
    total_vector_energy_j: float
    total_tensor_energy_j: float
    transition_shifts_hz_by_q: tuple[float, float, float]

    @property
    def total_transition_energy_j_for_sigma_plus(self) -> float:
        return (
            self.total_scalar_energy_j
            + self.total_vector_energy_j
            + self.total_tensor_energy_j
        )

    @property
    def total_transition_shift_hz_for_sigma_plus(self) -> float:
        return self.transition_shifts_hz_by_q[2]

    def transition_shift_hz(self, q: int) -> float:
        """Return the effective q=-1,0,+1 transition shift in hertz."""

        if q not in (-1, 0, +1):
            raise ValueError("q must be -1, 0, or +1")
        return self.transition_shifts_hz_by_q[q + 1]


def _finite_vector3(value, *, name: str, nonzero: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if nonzero and float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector


def load_pmot_2level_polarizability_table() -> DifferentialPolarizabilityTable:
    """Preload the required narrow Arora CCSD differential table."""

    return load_differential_polarizability_table(default_polarizability_csv_path())


def atom_frame_trapping_wavelengths_nm(
    trapping_beams: Sequence[TrappingBeam],
    velocity_m_per_s,
) -> np.ndarray:
    """Return each trapping component's nonrelativistic seen wavelength.

    ``lambda_seen = lambda_lab / (1 - dot(k_hat, v)/c)`` is evaluated
    separately for every traveling component.
    """

    velocity = _finite_vector3(velocity_m_per_s, name="velocity_m_per_s")
    directions = np.asarray([beam.direction for beam in trapping_beams], dtype=float)
    wavelengths_m = np.asarray(
        [beam.wavelength_m for beam in trapping_beams],
        dtype=float,
    )
    if directions.ndim != 2 or directions.shape != (len(trapping_beams), 3):
        raise ValueError("trapping_beams must contain valid propagation directions")
    denominator = 1.0 - directions @ velocity / SPEED_OF_LIGHT_M_PER_S
    if np.any(denominator <= 0.0):
        raise ValueError("atom-frame trapping wavelengths must remain positive")
    return 1.0e9 * wavelengths_m / denominator


def evaluate_differential_ac_stark_shift(
    trapping_beams: Sequence[TrappingBeam],
    position_m,
    velocity_m_per_s,
    *,
    path_powers_w: Sequence[float],
    previous_axis=DEFAULT_FALLBACK_QUANTIZATION_AXIS,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
    relative_axis_zero_tolerance: float = 1.0e-12,
) -> PMOT2LevelStarkSample:
    """Evaluate the effective F=2 -> F'=3 differential Stark shift.

    For trapping component ``j`` the implemented equations are

    ``E_j^2 = 2 I_j/(c epsilon_0)``,

    ``dU0_j = -dalpha0_j E_j^2``,

    ``dU1_j = -dalpha1_j E_j^2 s_j (khat_j dot n_hat)``, and

    ``dU2_j = -(1/2) dalpha2_j E_j^2
              (3 |epsilon_j^* dot n_hat|^2 - 1)``.

    The vector quantization axis is obtained from the vector-energy sum before
    the scalar/vector/tensor projections are evaluated.  At a vector zero the
    normalized previous axis is used only as a numerical basis convention.
    """

    beams = tuple(trapping_beams)
    if not beams:
        raise ValueError("at least one trapping component is required")
    if len(beams) > 6:
        raise ValueError("the pMOT geometry supports at most six trapping components")
    if not np.isfinite(relative_axis_zero_tolerance) or relative_axis_zero_tolerance < 0.0:
        raise ValueError("relative_axis_zero_tolerance must be finite and non-negative")
    position = _finite_vector3(position_m, name="position_m")
    velocity = _finite_vector3(velocity_m_per_s, name="velocity_m_per_s")
    fallback = _finite_vector3(previous_axis, name="previous_axis", nonzero=True)
    fallback /= np.linalg.norm(fallback)

    table = polarizability_table or load_pmot_2level_polarizability_table()
    seen_wavelengths_nm = atom_frame_trapping_wavelengths_nm(beams, velocity)
    alpha_scalar, alpha_vector, alpha_tensor = (
        interpolate_differential_polarizability_arrays(
            seen_wavelengths_nm,
            table,
        )
    )
    intensities = component_intensities_w_per_m2(
        position,
        beams,
        path_powers_w=path_powers_w,
    )
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )
    directions = np.asarray([beam.direction for beam in beams], dtype=float)
    spin_signs = np.asarray([helicity_sign(beam.helicity) for beam in beams])

    vector_energy_vectors = (
        -alpha_vector[:, np.newaxis]
        * field_squared[:, np.newaxis]
        * spin_signs[:, np.newaxis]
        * directions
    )
    net_vector_energy = np.sum(vector_energy_vectors, axis=0)
    vector_magnitude = float(np.linalg.norm(net_vector_energy))
    vector_scale = float(np.sum(np.linalg.norm(vector_energy_vectors, axis=1)))
    if vector_magnitude <= relative_axis_zero_tolerance * vector_scale:
        axis = fallback
        axis_from_vector_shift = False
    else:
        axis = net_vector_energy / vector_magnitude
        axis_from_vector_shift = True

    vector_factors = spin_signs * (directions @ axis)
    tensor_factors = np.empty(len(beams), dtype=float)
    for index, beam in enumerate(beams):
        epsilon = np.asarray(
            propagation_frame_polarization(beam.direction, beam.helicity),
            dtype=complex,
        )
        tensor_factors[index] = 3.0 * abs(np.vdot(epsilon, axis)) ** 2 - 1.0

    scalar_energies = -alpha_scalar * field_squared
    vector_energies = -alpha_vector * field_squared * vector_factors
    tensor_energies = -0.5 * alpha_tensor * field_squared * tensor_factors
    total_scalar = float(np.sum(scalar_energies))
    total_vector = float(np.sum(vector_energies))
    total_tensor = float(np.sum(tensor_energies))

    # The table's vector entry is the +stretched reference.  The effective
    # F-manifold closure mirrors that term for sigma- and sets it to zero for
    # pi, while retaining the same scalar/tensor reference contribution.
    common_energy = total_scalar + total_tensor
    shifts_hz = tuple(
        float((common_energy + q * total_vector) / PLANCK_CONSTANT_J_S)
        for q in (-1, 0, +1)
    )
    return PMOT2LevelStarkSample(
        position_m=tuple(float(value) for value in position),
        velocity_m_per_s=tuple(float(value) for value in velocity),
        trapping_component_labels=tuple(beam.label for beam in beams),
        lab_wavelengths_nm=tuple(1.0e9 * beam.wavelength_m for beam in beams),
        seen_wavelengths_nm=tuple(float(value) for value in seen_wavelengths_nm),
        component_intensities_w_per_m2=tuple(float(value) for value in intensities),
        component_field_squared_v2_per_m2=tuple(float(value) for value in field_squared),
        scalar_polarizabilities_si=tuple(float(value) for value in alpha_scalar),
        vector_polarizabilities_si=tuple(float(value) for value in alpha_vector),
        tensor_polarizabilities_si=tuple(float(value) for value in alpha_tensor),
        component_vector_energy_vectors_j=tuple(
            tuple(float(value) for value in vector) for vector in vector_energy_vectors
        ),
        net_vector_energy_vector_j=tuple(float(value) for value in net_vector_energy),
        quantization_axis=tuple(float(value) for value in axis),
        quantization_axis_from_vector_shift=axis_from_vector_shift,
        component_vector_angular_factors=tuple(float(value) for value in vector_factors),
        component_tensor_angular_factors=tuple(float(value) for value in tensor_factors),
        component_scalar_energies_j=tuple(float(value) for value in scalar_energies),
        component_vector_energies_j=tuple(float(value) for value in vector_energies),
        component_tensor_energies_j=tuple(float(value) for value in tensor_energies),
        total_scalar_energy_j=total_scalar,
        total_vector_energy_j=total_vector,
        total_tensor_energy_j=total_tensor,
        transition_shifts_hz_by_q=shifts_hz,
    )


__all__ = [
    "EFFECTIVE_TRANSITION_LABEL",
    "ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION",
    "PMOT2LevelStarkSample",
    "atom_frame_trapping_wavelengths_nm",
    "evaluate_differential_ac_stark_shift",
    "load_pmot_2level_polarizability_table",
]
