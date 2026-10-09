"""Fixed-basis stretched-transition AC-Stark diagnostic.

This temporary model evaluates only the differential shift of

    5S1/2 F=2, mF=+2 -> 5P3/2 F'=3, mF'=+3.

The Arora CCSD table supplies differential scalar, vector, and tensor values
for that reference transition.  It does *not* supply separate level-resolved
polarizabilities, so this module intentionally does not synthesize shifts for
the other 53 absorption transitions in the 24-state model.

Unlike the older local-adiabatic provisional proxy, the quantization axis here
is an explicit fixed input.  Consequently the vector term is the signed
projection Q.n, not |Q|, and a fixed-name +mF transition remains the same state
on opposite sides of the optical-spin zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...configuration import PLANCK_CONSTANT_J_S
from ...configuration import SPEED_OF_LIGHT_M_PER_S
from ...configuration import VACUUM_PERMITTIVITY_F_PER_M
from ...fields import MOTBeam
from ...mot_multilevel.polarization import propagation_frame_polarization
from ..ac_stark import ProvisionalStarkConfig
from ..ac_stark import atom_frame_trapping_frequencies_hz
from ..ac_stark import trapping_component_intensities_w_per_m2
from ..polarizability import DifferentialPolarizabilityTable
from ..polarizability import interpolate_differential_polarizability_arrays
from ..trapping_beams import TrappingBeam
from ..trapping_beams import TrappingLaserConfig
from ..trapping_beams import helicity_sign


STRETCHED_TRANSITION_LABEL = (
    "5S1/2 F=2, mF=+2 -> 5P3/2 F'=3, mF'=+3"
)

ORDINARY_FREQUENCY_EFFECTIVE_DETUNING_EQUATION = (
    "delta_nu_eff[b] = delta_nu_L[b] - delta_nu_HFS "
    "- dot(k_hat_b, v)/lambda_b - DeltaE_AC/h"
)

ANGULAR_FREQUENCY_EFFECTIVE_DETUNING_EQUATION = (
    "Delta_eff[b] = Delta_L[b] - delta_HFS "
    "- k_b dot v - DeltaE_AC/hbar"
)


def _unit_vector(vector, *, name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return values / norm


@dataclass(frozen=True, slots=True)
class FixedStretchedShiftObservable:
    """All beamwise inputs and differential-shift terms at one phase point."""

    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    quantization_axis: tuple[float, float, float]
    trapping_component_labels: tuple[str, ...]
    trapping_component_directions: tuple[tuple[float, float, float], ...]
    trapping_component_helicities: tuple[str, ...]
    projected_speeds_m_per_s: np.ndarray
    lab_frequencies_hz: np.ndarray
    atom_frame_frequencies_hz: np.ndarray
    trapping_doppler_shifts_hz: np.ndarray
    atom_frame_wavelengths_nm: np.ndarray
    component_intensities_w_per_m2: np.ndarray
    component_field_squared_v2_per_m2: np.ndarray
    scalar_polarizability_assumed_si: np.ndarray
    vector_polarizability_assumed_si: np.ndarray
    tensor_polarizability_assumed_si: np.ndarray
    component_vector_angular_factor: np.ndarray
    component_tensor_angular_factor: np.ndarray
    component_scalar_energy_j: np.ndarray
    component_vector_energy_j: np.ndarray
    component_tensor_energy_j: np.ndarray
    component_total_energy_j: np.ndarray
    component_scalar_shift_hz: np.ndarray
    component_vector_shift_hz: np.ndarray
    component_tensor_shift_hz: np.ndarray
    component_total_shift_hz: np.ndarray
    total_scalar_energy_j: float
    total_vector_energy_j: float
    total_tensor_energy_j: float
    total_energy_j: float
    total_scalar_shift_hz: float
    total_vector_shift_hz: float
    total_tensor_shift_hz: float
    total_frequency_shift_hz: float

    @property
    def total_intensity_w_per_m2(self) -> float:
        return float(np.sum(self.component_intensities_w_per_m2))

    def shifted_resonance_frequency_hz(
        self,
        bare_resonance_frequency_hz: float,
    ) -> float:
        """Return nu_res = nu_bare + DeltaE_AC/h for the named transition."""

        if not np.isfinite(bare_resonance_frequency_hz) or bare_resonance_frequency_hz <= 0.0:
            raise ValueError("bare_resonance_frequency_hz must be finite and positive")
        return float(bare_resonance_frequency_hz + self.total_frequency_shift_hz)


def evaluate_fixed_stretched_transition_shift(
    trapping_beams: list[TrappingBeam],
    position_m,
    velocity_m_per_s,
    quantization_axis,
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
    polarizability_table: DifferentialPolarizabilityTable,
) -> FixedStretchedShiftObservable:
    """Evaluate the signed differential shift of the fixed +mF reference.

    The raw table values are assumed to be SI polarizabilities.  For each
    trapping component j the calculation is

        E_j^2 = 2 I_j / (c epsilon_0)
        U0_j = -alpha0_j E_j^2
        U1_j = -alpha1_j E_j^2 s_j (k_hat_j . n_hat)
        U2_j = -alpha2_j E_j^2 [3|epsilon_j.n_hat|^2-1] / 2

    where s=-1 for propagation-frame sigma+ and +1 for sigma-.  The factor
    1/2 is the differential tensor angular factor for the named stretched
    transition under the repository's Arora-table convention.
    """

    if len(trapping_beams) != 6:
        raise ValueError("the current pMOT geometry requires six trapping components")
    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_m_per_s, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_m must be a finite 3-vector")
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_m_per_s must be a finite 3-vector")
    axis = _unit_vector(quantization_axis, name="quantization_axis")

    directions = np.asarray([beam.direction for beam in trapping_beams], dtype=float)
    lab_frequencies_hz = np.asarray(
        [SPEED_OF_LIGHT_M_PER_S / beam.wavelength_m for beam in trapping_beams],
        dtype=float,
    )
    atom_frame_frequencies_hz = atom_frame_trapping_frequencies_hz(
        trapping_beams,
        velocity,
    )
    atom_frame_wavelengths_nm = (
        1.0e9 * SPEED_OF_LIGHT_M_PER_S / atom_frame_frequencies_hz
    )
    alpha_scalar, alpha_vector, alpha_tensor = (
        interpolate_differential_polarizability_arrays(
            atom_frame_wavelengths_nm,
            polarizability_table,
        )
    )
    intensities = trapping_component_intensities_w_per_m2(
        trapping_beams,
        position,
        laser_config,
        stark_config,
    )
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )

    vector_factor = np.asarray(
        [
            helicity_sign(beam.helicity)
            * float(np.dot(np.asarray(beam.direction, dtype=float), axis))
            for beam in trapping_beams
        ],
        dtype=float,
    )
    tensor_factor = np.empty(len(trapping_beams), dtype=float)
    for index, beam in enumerate(trapping_beams):
        epsilon = np.asarray(
            propagation_frame_polarization(beam.direction, beam.helicity),
            dtype=complex,
        )
        tensor_factor[index] = 3.0 * abs(np.vdot(axis, epsilon)) ** 2 - 1.0

    scalar_energy = -alpha_scalar * field_squared
    vector_energy = -alpha_vector * field_squared * vector_factor
    tensor_energy = -0.5 * alpha_tensor * field_squared * tensor_factor
    total_energy = scalar_energy + vector_energy + tensor_energy

    scalar_shift_hz = scalar_energy / PLANCK_CONSTANT_J_S
    vector_shift_hz = vector_energy / PLANCK_CONSTANT_J_S
    tensor_shift_hz = tensor_energy / PLANCK_CONSTANT_J_S
    total_shift_hz = total_energy / PLANCK_CONSTANT_J_S

    total_scalar_energy = float(np.sum(scalar_energy))
    total_vector_energy = float(np.sum(vector_energy))
    total_tensor_energy = float(np.sum(tensor_energy))
    total_energy_j = float(np.sum(total_energy))
    return FixedStretchedShiftObservable(
        position_m=tuple(float(value) for value in position),
        velocity_m_per_s=tuple(float(value) for value in velocity),
        quantization_axis=tuple(float(value) for value in axis),
        trapping_component_labels=tuple(beam.label for beam in trapping_beams),
        trapping_component_directions=tuple(beam.direction for beam in trapping_beams),
        trapping_component_helicities=tuple(beam.helicity for beam in trapping_beams),
        projected_speeds_m_per_s=directions @ velocity,
        lab_frequencies_hz=lab_frequencies_hz,
        atom_frame_frequencies_hz=atom_frame_frequencies_hz,
        trapping_doppler_shifts_hz=atom_frame_frequencies_hz - lab_frequencies_hz,
        atom_frame_wavelengths_nm=atom_frame_wavelengths_nm,
        component_intensities_w_per_m2=intensities,
        component_field_squared_v2_per_m2=field_squared,
        scalar_polarizability_assumed_si=alpha_scalar,
        vector_polarizability_assumed_si=alpha_vector,
        tensor_polarizability_assumed_si=alpha_tensor,
        component_vector_angular_factor=vector_factor,
        component_tensor_angular_factor=tensor_factor,
        component_scalar_energy_j=scalar_energy,
        component_vector_energy_j=vector_energy,
        component_tensor_energy_j=tensor_energy,
        component_total_energy_j=total_energy,
        component_scalar_shift_hz=scalar_shift_hz,
        component_vector_shift_hz=vector_shift_hz,
        component_tensor_shift_hz=tensor_shift_hz,
        component_total_shift_hz=total_shift_hz,
        total_scalar_energy_j=total_scalar_energy,
        total_vector_energy_j=total_vector_energy,
        total_tensor_energy_j=total_tensor_energy,
        total_energy_j=total_energy_j,
        total_scalar_shift_hz=total_scalar_energy / PLANCK_CONSTANT_J_S,
        total_vector_shift_hz=total_vector_energy / PLANCK_CONSTANT_J_S,
        total_tensor_shift_hz=total_tensor_energy / PLANCK_CONSTANT_J_S,
        total_frequency_shift_hz=total_energy_j / PLANCK_CONSTANT_J_S,
    )


def cooling_effective_detunings_hz(
    cooling_beams: list[MOTBeam],
    velocity_m_per_s,
    transition_shift_hz: float,
    *,
    hyperfine_offset_hz: float = 0.0,
    cooling_carrier_offset_hz: float = 0.0,
) -> np.ndarray:
    """Return the six ordinary-frequency detunings for the named transition.

    ``cooling_carrier_offset_hz`` shifts the laboratory cooling carrier from
    its configured value.  Setting it equal to the central Stark shift keeps
    the carrier at the configured red detuning relative to the shifted central
    resonance.
    """

    velocity = np.asarray(velocity_m_per_s, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_m_per_s must be a finite 3-vector")
    if not np.isfinite(transition_shift_hz):
        raise ValueError("transition_shift_hz must be finite")
    selected = [beam for beam in cooling_beams if beam.family == "cooling"]
    if len(selected) != 6:
        raise ValueError("cooling_beams must contain the six cooling components")
    return np.asarray(
        [
            beam.detuning_hz
            + cooling_carrier_offset_hz
            - hyperfine_offset_hz
            - float(np.dot(np.asarray(beam.direction), velocity)) / beam.wavelength_m
            - transition_shift_hz
            for beam in selected
        ],
        dtype=float,
    )


__all__ = [name for name in globals() if not name.startswith("_")]
