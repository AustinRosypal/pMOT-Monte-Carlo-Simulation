"""Provisional differential AC-Stark layer for the no-coil pMOT.

The supplied Arora CSV contains one differential scalar/vector/tensor triplet
per wavelength.  It does not contain separate polarizabilities for the 24
hyperfine-Zeeman levels, so this module deliberately exposes a named
*provisional* transition model rather than a production Stark Hamiltonian.

The model is exact for the tabulated stretched cycling-transition reference.
Outside that channel it applies a Zeeman-like vector rescaling and an
excited-state tensor proxy, both recorded explicitly in output metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from math import pi

import numpy as np

from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M
from ..mot_multilevel.polarization import propagation_frame_polarization
from ..mot_multilevel.rate_equations import RateEquationModel
from .polarizability import DifferentialPolarizabilityTable
from .polarizability import interpolate_differential_polarizability_arrays
from .polarizability import load_differential_polarizability_table
from .trapping_beams import DEFAULT_TRAPPING_AXES
from .trapping_beams import TrappingBeam
from .trapping_beams import TrappingLaserConfig
from .trapping_beams import build_trapping_beams
from .trapping_beams import helicity_sign
from .trapping_beams import trapping_beam_intensity_response_m_inv2


PROVISIONAL_MODEL_NAME = "stretched_reference_differential_transition_proxy"
EFFECTIVE_DETUNING_EQUATION = (
    "Delta_eff[b,g->e] = Delta_L[b] - delta_HFS[g->e] - k_b dot v "
    "- (DeltaE_AC[e]-DeltaE_AC[g])/hbar; B_external = 0"
)


@dataclass(frozen=True, slots=True)
class ProvisionalStarkConfig:
    """Absolute powers and path polarizations for the provisional Stark layer."""

    incident_path_powers_w: tuple[float, float, float]
    incident_helicities_by_axis: tuple[str, str, str] = (
        "sigma+",
        "sigma+",
        "sigma-",
    )
    retro_helicities_by_axis: tuple[str, str, str] = (
        "sigma+",
        "sigma+",
        "sigma-",
    )
    optical_spin_energy_epsilon_j: float = 1.0e-40

    def __post_init__(self) -> None:
        if len(self.incident_path_powers_w) != 3:
            raise ValueError("incident_path_powers_w must contain x, y, and z path powers")
        if not all(np.isfinite(power) and power >= 0.0 for power in self.incident_path_powers_w):
            raise ValueError("incident path powers must be finite and non-negative")
        for collection in (
            self.incident_helicities_by_axis,
            self.retro_helicities_by_axis,
        ):
            if len(collection) != 3:
                raise ValueError("each helicity tuple must contain x, y, and z values")
            for helicity in collection:
                helicity_sign(helicity)
                if helicity == "pi":
                    raise ValueError(
                        "the provisional tensor model requires sigma+ or sigma-; "
                        "a pi beam needs an explicit transverse linear-polarization axis"
                    )
        if self.optical_spin_energy_epsilon_j < 0.0:
            raise ValueError("optical_spin_energy_epsilon_j must be non-negative")

    @classmethod
    def uniform_power(
        cls,
        power_w_per_path: float,
        **kwargs,
    ) -> "ProvisionalStarkConfig":
        """Build a three-path configuration with equal launched powers."""

        return cls(
            incident_path_powers_w=(
                power_w_per_path,
                power_w_per_path,
                power_w_per_path,
            ),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class ProvisionalStarkObservable:
    """Beamwise inputs and transition shifts at one phase-space point."""

    atom_frame_frequencies_hz: tuple[float, ...]
    atom_frame_wavelengths_nm: tuple[float, ...]
    component_intensities_w_per_m2: tuple[float, ...]
    component_scalar_energy_j: tuple[float, ...]
    vector_reference_energy_vector_j: tuple[float, float, float]
    component_tensor_reference_energy_j: tuple[float, ...]
    scalar_transition_energy_j: np.ndarray
    vector_transition_energy_j: np.ndarray
    tensor_transition_energy_j: np.ndarray
    total_transition_energy_j: np.ndarray
    transition_frequency_shift_hz: np.ndarray
    transition_angular_frequency_shift_rad_per_s: np.ndarray
    effective_field_t: tuple[float, float, float]
    quantization_axis: tuple[float, float, float]

    @property
    def total_intensity_w_per_m2(self) -> float:
        return float(sum(self.component_intensities_w_per_m2))


def build_physics_trapping_beams(
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
) -> list[TrappingBeam]:
    """Build trapping components with independently configurable path helicities."""

    beams = build_trapping_beams(laser_config)
    incident = dict(zip(laser_config.axes, stark_config.incident_helicities_by_axis))
    retro = dict(zip(laser_config.axes, stark_config.retro_helicities_by_axis))
    return [
        replace(
            beam,
            helicity=(
                incident[beam.axis_name]
                if beam.propagation_sense == "incident"
                else retro[beam.axis_name]
            ),
        )
        for beam in beams
    ]


def atom_frame_trapping_frequencies_hz(
    beams: list[TrappingBeam],
    velocity_m_per_s,
) -> np.ndarray:
    """Return first-order 3D Doppler frequencies for every trapping component.

    The existing multilevel MOT is nonrelativistic, so this uses
    ``nu' = nu * (1 - k_hat dot v / c)`` consistently.  The dot product retains
    all three velocity components; no one-dimensional speed shortcut is used.
    """

    velocity = np.asarray(velocity_m_per_s, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_m_per_s must be a finite 3-vector")
    directions = np.asarray([beam.direction for beam in beams], dtype=float)
    wavelengths = np.asarray([beam.wavelength_m for beam in beams], dtype=float)
    denominators = 1.0 - directions @ velocity / SPEED_OF_LIGHT_M_PER_S
    if np.any(denominators <= 0.0):
        raise ValueError("atom-frame trapping frequency must remain positive")
    return (SPEED_OF_LIGHT_M_PER_S / wavelengths) * denominators


def atom_frame_trapping_wavelengths_m(
    beams: list[TrappingBeam],
    velocity_m_per_s,
) -> np.ndarray:
    """Return wavelengths corresponding to the beamwise atom-frame frequencies."""

    frequencies_hz = atom_frame_trapping_frequencies_hz(
        beams,
        velocity_m_per_s,
    )
    return SPEED_OF_LIGHT_M_PER_S / frequencies_hz


def _axis_path_powers(
    beams: list[TrappingBeam],
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
) -> np.ndarray:
    powers = dict(zip(laser_config.axes, stark_config.incident_path_powers_w))
    return np.asarray([powers[beam.axis_name] for beam in beams], dtype=float)


def trapping_component_intensities_w_per_m2(
    beams: list[TrappingBeam],
    position_m,
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
) -> np.ndarray:
    """Return physical component intensities from per-path-watt responses."""

    point = np.asarray(position_m, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("position_m must be a finite 3-vector")
    path_powers = _axis_path_powers(beams, laser_config, stark_config)
    responses = np.asarray(
        [trapping_beam_intensity_response_m_inv2(beam, point) for beam in beams],
        dtype=float,
    )
    return path_powers * responses


def _reference_cycling_transition_index(model: RateEquationModel) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        ground = model.structure.states[transition.ground_state_index]
        excited = model.structure.states[transition.excited_state_index]
        if (ground.f, ground.m_f, excited.f, excited.m_f) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the 24-state graph is missing the stretched cycling transition")


def _excited_tensor_factors(model: RateEquationModel) -> np.ndarray:
    factors = np.zeros(len(model.transition_ground), dtype=float)
    for index, transition in enumerate(model.structure.absorption_transitions):
        excited = model.structure.states[transition.excited_state_index]
        if excited.f <= 0:
            continue
        factors[index] = (
            3.0 * excited.m_f**2 - excited.f * (excited.f + 1.0)
        ) / (2.0 * excited.f * (2.0 * excited.f - 1.0))
    return factors


def provisional_transition_stark_shifts(
    model: RateEquationModel,
    trapping_beams: list[TrappingBeam],
    position_m,
    velocity_m_per_s,
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
    previous_axis=(0.0, 0.0, 1.0),
    *,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
) -> ProvisionalStarkObservable:
    """Calculate provisional scalar, vector, and tensor transition shifts.

    Scalar differential shifts are applied to every channel.  Vector shifts
    are rescaled from the stretched cycling reference by the channel's
    Zeeman coefficient.  Tensor shifts use the excited-state angular factor
    while neglecting the unavailable separate ground-state tensor response.
    These two extensions are ansatzes, not a recovered 24-state Hamiltonian.
    """

    if len(trapping_beams) != 6:
        raise ValueError("the current pMOT physics layer requires six trapping components")
    table = polarizability_table or load_differential_polarizability_table()
    frequencies_hz = atom_frame_trapping_frequencies_hz(
        trapping_beams,
        velocity_m_per_s,
    )
    wavelengths_m = SPEED_OF_LIGHT_M_PER_S / frequencies_hz
    wavelengths_nm = 1.0e9 * wavelengths_m
    alpha_scalar, alpha_vector, alpha_tensor = (
        interpolate_differential_polarizability_arrays(wavelengths_nm, table)
    )
    intensities = trapping_component_intensities_w_per_m2(
        trapping_beams,
        position_m,
        laser_config,
        stark_config,
    )
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )

    component_scalar = -alpha_scalar * field_squared
    vector_energy = np.zeros(3, dtype=float)
    for beam, alpha, amplitude_squared in zip(
        trapping_beams,
        alpha_vector,
        field_squared,
    ):
        vector_energy += (
            -alpha
            * amplitude_squared
            * helicity_sign(beam.helicity)
            * np.asarray(beam.direction, dtype=float)
        )

    previous = np.asarray(previous_axis, dtype=float)
    previous_norm = float(np.linalg.norm(previous))
    if previous.shape != (3,) or not np.all(np.isfinite(previous)) or previous_norm <= 0.0:
        raise ValueError("previous_axis must be a finite nonzero 3-vector")
    vector_norm = float(np.linalg.norm(vector_energy))
    if vector_norm <= stark_config.optical_spin_energy_epsilon_j:
        axis = previous / previous_norm
        vector_reference_energy = 0.0
    else:
        axis = vector_energy / vector_norm
        vector_reference_energy = float(np.dot(vector_energy, axis))

    component_tensor_by_transition = np.empty(
        (len(trapping_beams), len(model.transition_ground)),
        dtype=float,
    )
    tensor_factors = _excited_tensor_factors(model)
    component_tensor_reference = []
    for beam_index, (beam, alpha, amplitude_squared) in enumerate(
        zip(trapping_beams, alpha_tensor, field_squared)
    ):
        epsilon = np.asarray(
            propagation_frame_polarization(beam.direction, beam.helicity),
            dtype=complex,
        )
        tensor_geometry = 3.0 * abs(np.vdot(axis, epsilon)) ** 2 - 1.0
        component_tensor_by_transition[beam_index] = (
            -alpha * amplitude_squared * tensor_geometry * tensor_factors
        )
        component_tensor_reference.append(
            float(-alpha * amplitude_squared * tensor_geometry * 0.5)
        )

    transition_count = len(model.transition_ground)
    scalar_transition = np.full(
        transition_count,
        float(np.sum(component_scalar)),
        dtype=float,
    )
    reference_index = _reference_cycling_transition_index(model)
    reference_coefficient = model.transition_zeeman_coefficient[reference_index]
    if abs(reference_coefficient) <= 0.0:
        raise RuntimeError("stretched cycling transition has zero Zeeman coefficient")
    vector_transition = (
        vector_reference_energy
        * model.transition_zeeman_coefficient
        / reference_coefficient
    )
    tensor_transition = np.sum(component_tensor_by_transition, axis=0)
    total_transition = scalar_transition + vector_transition + tensor_transition
    shift_hz = total_transition / PLANCK_CONSTANT_J_S

    # B_eff is only a diagnostic representation of the vector ansatz.  It is
    # not passed to the solver as an external magnetic field.
    effective_field = vector_energy / (
        reference_coefficient * (PLANCK_CONSTANT_J_S / (2.0 * pi))
    )
    return ProvisionalStarkObservable(
        atom_frame_frequencies_hz=tuple(float(value) for value in frequencies_hz),
        atom_frame_wavelengths_nm=tuple(float(value) for value in wavelengths_nm),
        component_intensities_w_per_m2=tuple(float(value) for value in intensities),
        component_scalar_energy_j=tuple(float(value) for value in component_scalar),
        vector_reference_energy_vector_j=tuple(float(value) for value in vector_energy),
        component_tensor_reference_energy_j=tuple(component_tensor_reference),
        scalar_transition_energy_j=scalar_transition,
        vector_transition_energy_j=vector_transition,
        tensor_transition_energy_j=tensor_transition,
        total_transition_energy_j=total_transition,
        transition_frequency_shift_hz=shift_hz,
        transition_angular_frequency_shift_rad_per_s=2.0 * pi * shift_hz,
        effective_field_t=tuple(float(value) for value in effective_field),
        quantization_axis=tuple(float(value) for value in axis),
    )


def provisional_power_for_target_gradient_w_per_path(
    model: RateEquationModel,
    laser_config: TrappingLaserConfig,
    *,
    target_gradient_g_per_cm: float = 20.0,
    displacement_m: float = 1.0e-6,
    axis_name: str = "horizontal_x",
    polarizability_table: DifferentialPolarizabilityTable | None = None,
) -> float:
    """Return the stretched-reference power scale for a target vector gradient.

    This is a calibration of the provisional effective-field proxy, not a
    physical power recommendation.
    """

    if target_gradient_g_per_cm <= 0.0 or displacement_m <= 0.0:
        raise ValueError("target gradient and displacement must be positive")
    if axis_name not in DEFAULT_TRAPPING_AXES:
        raise ValueError(f"axis_name must be one of {DEFAULT_TRAPPING_AXES}")
    unit_config = ProvisionalStarkConfig.uniform_power(1.0)
    beams = build_physics_trapping_beams(laser_config, unit_config)
    component = DEFAULT_TRAPPING_AXES.index(axis_name)
    plus = np.zeros(3)
    minus = np.zeros(3)
    plus[component] = displacement_m
    minus[component] = -displacement_m
    table = polarizability_table or load_differential_polarizability_table()
    plus_observable = provisional_transition_stark_shifts(
        model,
        beams,
        plus,
        (0.0, 0.0, 0.0),
        laser_config,
        unit_config,
        polarizability_table=table,
    )
    minus_observable = provisional_transition_stark_shifts(
        model,
        beams,
        minus,
        (0.0, 0.0, 0.0),
        laser_config,
        unit_config,
        polarizability_table=table,
    )
    slope_t_per_m_per_w = (
        plus_observable.effective_field_t[component]
        - minus_observable.effective_field_t[component]
    ) / (2.0 * displacement_m)
    if abs(slope_t_per_m_per_w) <= 0.0:
        raise RuntimeError("provisional effective-field gradient is zero")
    target_t_per_m = 1.0e-2 * target_gradient_g_per_cm
    return target_t_per_m / abs(slope_t_per_m_per_w)


__all__ = [name for name in globals() if not name.startswith("_")]
