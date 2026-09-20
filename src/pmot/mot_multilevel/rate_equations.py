"""Physical 24-state population-rate-equation MOT from Section 12.

The elementary laser coefficient is

    W = Gamma_e |Omega|^2 / (Gamma_e^2 + 4 Delta^2),

with no saturated two-level ``1+s`` denominator. Saturation and optical
pumping arise only from the coupled steady-state populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2, HBAR_J_S, RB87_MASS_KG
from ..fields import MOTBeam, beam_intensity_w_per_m2
from ..magnetic_fields import anti_helmholtz_field_t
from .atomic_structure import (
    AtomicStructure,
    BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T,
    build_atomic_structure,
)
from .configuration import MultilevelMOTConfig, default_multilevel_mot_config
from .coupling import (
    beam_polarization_vector,
    electric_field_amplitude_v_per_m,
    wavevector_rad_per_m,
)
from .polarization import Vec3, polarization_amplitudes, quantization_axis
from .simulation import build_multilevel_mot_beams


@dataclass(frozen=True, slots=True)
class RateEquationModel:
    """All position-independent ARC data in vectorized solver form."""

    structure: AtomicStructure
    ground_indices: np.ndarray
    excited_indices: np.ndarray
    global_to_ground: np.ndarray
    global_to_excited: np.ndarray
    transition_ground: np.ndarray
    transition_excited: np.ndarray
    transition_q: np.ndarray
    transition_ground_f: np.ndarray
    transition_excited_f: np.ndarray
    transition_dipole_c_m: np.ndarray
    transition_dipole_ea0: np.ndarray
    transition_angular_frequency_rad_per_s: np.ndarray
    transition_spontaneous_decay_rate_per_s: np.ndarray
    transition_zeeman_coefficient_rad_per_s_per_t: np.ndarray
    spontaneous_decay_matrix_per_s: np.ndarray
    excited_decay_rates_per_s: np.ndarray

    @property
    def ground_count(self) -> int:
        return len(self.ground_indices)

    @property
    def excited_count(self) -> int:
        return len(self.excited_indices)

    @property
    def state_count(self) -> int:
        return self.ground_count + self.excited_count

    @property
    def transition_count(self) -> int:
        return len(self.transition_ground)


@dataclass(frozen=True, slots=True)
class BeamTransitionQuantities:
    """Beam/transition-resolved Section-12 quantities at one phase point."""

    rabi_frequencies_rad_per_s: np.ndarray
    effective_detunings_rad_per_s: np.ndarray
    stimulated_coefficients_per_s: np.ndarray


@dataclass(frozen=True, slots=True)
class RateEquationObservable:
    """Steady populations and radiation-pressure observables at one point."""

    populations: np.ndarray
    force_n: Vec3
    beam_effective_scattering_rates_per_s: tuple[float, ...]
    beam_absorption_rates_per_s: tuple[float, ...]
    beam_stimulated_emission_rates_per_s: tuple[float, ...]
    total_spontaneous_scattering_rate_per_s: float
    magnetic_field_t: Vec3
    quantization_axis: Vec3
    rate_matrix_per_s: np.ndarray | None = None
    beam_transition_quantities: BeamTransitionQuantities | None = None

    @property
    def total_effective_scattering_rate_per_s(self) -> float:
        return float(sum(self.beam_effective_scattering_rates_per_s))


@dataclass(frozen=True, slots=True)
class RateEquationAtomState:
    position_m: Vec3
    velocity_m_per_s: Vec3
    last_quantization_axis: Vec3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class RateEquationTrajectoryConfig:
    """Deterministic RK4 controls for external motion."""

    time_step_s: float = 5.0e-6
    escape_radius_m: float = 30.0e-3
    store_rate_matrices: bool = False
    store_beam_transition_quantities: bool = False

    def __post_init__(self) -> None:
        if self.time_step_s <= 0.0 or self.escape_radius_m <= 0.0:
            raise ValueError("time step and escape radius must be positive")


@dataclass(slots=True)
class RateEquationTrajectoryRecord:
    times_s: list[float] = field(default_factory=list)
    positions_m: list[Vec3] = field(default_factory=list)
    velocities_m_per_s: list[Vec3] = field(default_factory=list)
    forces_n: list[Vec3] = field(default_factory=list)
    beam_effective_scattering_rates_per_s: list[tuple[float, ...]] = field(
        default_factory=list
    )
    total_spontaneous_scattering_rates_per_s: list[float] = field(
        default_factory=list
    )
    magnetic_fields_t: list[Vec3] = field(default_factory=list)
    quantization_axes: list[Vec3] = field(default_factory=list)
    populations: list[np.ndarray] = field(default_factory=list)
    rate_matrices_per_s: list[np.ndarray] = field(default_factory=list)
    beam_transition_quantities: list[BeamTransitionQuantities] = field(
        default_factory=list
    )
    termination_reason: str = "duration"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@lru_cache(maxsize=1)
def build_rate_equation_model() -> RateEquationModel:
    """Build and cache ARC dipoles, pairwise decays, and sparse index arrays."""

    structure = build_atomic_structure()
    ground_indices = np.asarray(structure.ground_state_indices, dtype=int)
    excited_indices = np.asarray(structure.excited_state_indices, dtype=int)
    global_to_ground = np.full(len(structure.states), -1, dtype=int)
    global_to_excited = np.full(len(structure.states), -1, dtype=int)
    global_to_ground[ground_indices] = np.arange(len(ground_indices))
    global_to_excited[excited_indices] = np.arange(len(excited_indices))

    transitions = structure.transitions
    transition_ground = np.asarray(
        [global_to_ground[item.ground_state_index] for item in transitions],
        dtype=int,
    )
    transition_excited = np.asarray(
        [global_to_excited[item.excited_state_index] for item in transitions],
        dtype=int,
    )
    spontaneous_decay = np.zeros((len(ground_indices), len(excited_indices)))
    np.add.at(
        spontaneous_decay,
        (transition_ground, transition_excited),
        np.asarray(
            [item.spontaneous_decay_rate_per_s for item in transitions],
            dtype=float,
        ),
    )
    excited_decay = np.sum(spontaneous_decay, axis=0)
    if np.any(excited_decay <= 0.0):
        raise RuntimeError("every excited state must have a positive total decay rate")

    ground_states = [structure.states[item.ground_state_index] for item in transitions]
    excited_states = [structure.states[item.excited_state_index] for item in transitions]
    zeeman_coefficients = BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T * np.asarray(
        [
            excited.lande_g * excited.m_f - ground.lande_g * ground.m_f
            for ground, excited in zip(ground_states, excited_states)
        ],
        dtype=float,
    )
    return RateEquationModel(
        structure=structure,
        ground_indices=_readonly(ground_indices),
        excited_indices=_readonly(excited_indices),
        global_to_ground=_readonly(global_to_ground),
        global_to_excited=_readonly(global_to_excited),
        transition_ground=_readonly(transition_ground),
        transition_excited=_readonly(transition_excited),
        transition_q=_readonly(np.asarray([item.q for item in transitions], dtype=int)),
        transition_ground_f=_readonly(
            np.asarray([item.ground_f for item in transitions], dtype=int)
        ),
        transition_excited_f=_readonly(
            np.asarray([item.excited_f for item in transitions], dtype=int)
        ),
        transition_dipole_c_m=_readonly(
            np.asarray([item.dipole_matrix_element_c_m for item in transitions])
        ),
        transition_dipole_ea0=_readonly(
            np.asarray([item.dipole_matrix_element_ea0 for item in transitions])
        ),
        transition_angular_frequency_rad_per_s=_readonly(
            np.asarray(
                [item.transition_angular_frequency_rad_per_s for item in transitions]
            )
        ),
        transition_spontaneous_decay_rate_per_s=_readonly(
            np.asarray([item.spontaneous_decay_rate_per_s for item in transitions])
        ),
        transition_zeeman_coefficient_rad_per_s_per_t=_readonly(
            zeeman_coefficients
        ),
        spontaneous_decay_matrix_per_s=_readonly(spontaneous_decay),
        excited_decay_rates_per_s=_readonly(excited_decay),
    )


def local_magnetic_field_t(position_m: Vec3, coil_config) -> Vec3:
    values = anti_helmholtz_field_t(*position_m, coil_config)
    field = tuple(float(np.asarray(value)) for value in values)
    if not np.all(np.isfinite(field)):
        raise ValueError("magnetic field is non-finite at the requested position")
    return field


def _transition_shift_array(
    model: RateEquationModel,
    transition_resonance_shift_rad_per_s: np.ndarray | None,
) -> np.ndarray:
    if transition_resonance_shift_rad_per_s is None:
        return np.zeros(model.transition_count)
    shift = np.asarray(transition_resonance_shift_rad_per_s, dtype=float)
    if shift.shape != (model.transition_count,) or not np.all(np.isfinite(shift)):
        raise ValueError(
            "transition_resonance_shift_rad_per_s must contain one finite "
            "value per allowed transition"
        )
    return shift


def build_beam_transition_quantities(
    model: RateEquationModel,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig | None = None,
    *,
    transition_resonance_shift_rad_per_s: np.ndarray | None = None,
) -> BeamTransitionQuantities:
    """Evaluate Section 12 Eqs. (38), (39), and (41) for every beam/pair."""

    cfg = config or default_multilevel_mot_config()
    if field_magnitude_t < 0.0 or not np.isfinite(field_magnitude_t):
        raise ValueError("field magnitude must be finite and non-negative")
    velocity = np.asarray(velocity_m_per_s, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be a finite three-vector")
    transition_shift = _transition_shift_array(
        model,
        transition_resonance_shift_rad_per_s,
    )

    beam_count = len(beams)
    rabi = np.zeros((beam_count, model.transition_count), dtype=complex)
    detuning = np.full((beam_count, model.transition_count), np.nan)
    stimulated = np.zeros((beam_count, model.excited_count, model.ground_count))
    gamma_by_transition = model.excited_decay_rates_per_s[model.transition_excited]
    zeeman = (
        model.transition_zeeman_coefficient_rad_per_s_per_t
        * float(field_magnitude_t)
    )
    q_columns = model.transition_q + 1

    for beam_index, beam in enumerate(beams):
        if beam.family == "cooling":
            addressed = model.transition_ground_f == 2
            enabled = np.isin(
                model.transition_excited_f,
                cfg.enabled_cooling_excited_manifolds,
            )
            laser_omega = (
                model.structure.cooling_reference_angular_frequency_rad_per_s
                + cfg.cooling_detuning_rad_per_s
            )
        elif beam.family == "repump":
            if not cfg.repumper_enabled or beam.power_w <= 0.0:
                continue
            addressed = model.transition_ground_f == 1
            enabled = np.isin(
                model.transition_excited_f,
                cfg.enabled_repump_excited_manifolds,
            )
            laser_omega = (
                model.structure.repump_reference_angular_frequency_rad_per_s
                + cfg.repump_detuning_rad_per_s
            )
        else:
            raise ValueError(f"unsupported beam family: {beam.family}")

        mask = addressed & enabled
        intensity = beam_intensity_w_per_m2(beam, position_m)
        electric_field = electric_field_amplitude_v_per_m(intensity)
        amplitudes = polarization_amplitudes(
            beam_polarization_vector(beam),
            quantization_axis_vector,
        )
        epsilon = np.asarray(
            [amplitudes[-1], amplitudes[0], amplitudes[+1]],
            dtype=complex,
        )[q_columns]
        beam_rabi = (
            -electric_field
            * epsilon
            * model.transition_dipole_c_m
            / HBAR_J_S
        )
        wavevector = np.asarray(wavevector_rad_per_m(beam), dtype=float)
        beam_detuning = (
            laser_omega
            - model.transition_angular_frequency_rad_per_s
            - float(np.dot(wavevector, velocity))
            - zeeman
            - transition_shift
        )
        coefficients = (
            gamma_by_transition
            * np.abs(beam_rabi) ** 2
            / (gamma_by_transition**2 + 4.0 * beam_detuning**2)
        )
        rabi[beam_index, mask] = beam_rabi[mask]
        detuning[beam_index, mask] = beam_detuning[mask]
        stimulated[
            beam_index,
            model.transition_excited[mask],
            model.transition_ground[mask],
        ] = coefficients[mask]

    return BeamTransitionQuantities(
        rabi_frequencies_rad_per_s=rabi,
        effective_detunings_rad_per_s=detuning,
        stimulated_coefficients_per_s=stimulated,
    )


def build_beam_stimulated_rate_matrices(*args, **kwargs) -> np.ndarray:
    """Return W[b,e,g] for callers that need only the Section-12 coefficients."""

    return build_beam_transition_quantities(
        *args,
        **kwargs,
    ).stimulated_coefficients_per_s


def assemble_rate_matrix(
    stimulated_rate_matrix_per_s: np.ndarray,
    spontaneous_decay_matrix_per_s: np.ndarray,
) -> np.ndarray:
    """Build Eq. (43), choosing diagonals for exact column conservation."""

    stimulated = np.asarray(stimulated_rate_matrix_per_s, dtype=float)
    decay = np.asarray(spontaneous_decay_matrix_per_s, dtype=float)
    if stimulated.ndim != 2:
        raise ValueError("stimulated rate matrix must be two-dimensional")
    excited_count, ground_count = stimulated.shape
    if decay.shape != (ground_count, excited_count):
        raise ValueError("spontaneous decay matrix has incompatible shape")
    if (
        not np.all(np.isfinite(stimulated))
        or not np.all(np.isfinite(decay))
        or np.any(stimulated < 0.0)
        or np.any(decay < 0.0)
    ):
        raise ValueError("off-diagonal rates must be finite and non-negative")

    matrix = np.zeros((ground_count + excited_count,) * 2)
    matrix[ground_count:, :ground_count] = stimulated
    matrix[:ground_count, ground_count:] = stimulated.T + decay
    diagonal = np.diag_indices_from(matrix)
    matrix[diagonal] = -np.sum(matrix, axis=0)
    return matrix


def assert_rate_matrix_conserves_probability(
    rate_matrix_per_s: np.ndarray,
    absolute_tolerance_per_s: float = 1.0e-7,
) -> None:
    matrix = np.asarray(rate_matrix_per_s, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rate matrix must be square")
    column_sums = np.sum(matrix, axis=0)
    # Summing the positive and negative O(10^7 s^-1) links loses a few low
    # bits. Bound that cancellation by each column's absolute rate sum rather
    # than rejecting physically conservative matrices at a fixed threshold.
    roundoff = 32.0 * np.finfo(float).eps * np.sum(np.abs(matrix), axis=0)
    tolerance = np.maximum(absolute_tolerance_per_s, roundoff)
    if np.any(np.abs(column_sums) > tolerance):
        raise RuntimeError(
            "rate matrix does not conserve probability; maximum column-sum "
            f"residual is {np.max(np.abs(column_sums)):.6e} s^-1"
        )


def steady_state_populations(
    rate_matrix_per_s: np.ndarray,
    *,
    conservation_absolute_tolerance_per_s: float = 1.0e-7,
    residual_relative_tolerance: float = 1.0e-10,
    population_absolute_tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Solve M p=0 with sum(p)=1 and verify the physical solution."""

    matrix = np.asarray(rate_matrix_per_s, dtype=float)
    assert_rate_matrix_conserves_probability(
        matrix,
        conservation_absolute_tolerance_per_s,
    )
    scale = float(np.max(np.abs(matrix)))
    if scale <= 0.0:
        raise RuntimeError("zero rate matrix has no unique steady state")
    system = matrix / scale
    rhs = np.zeros(matrix.shape[0])
    system = system.copy()
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    try:
        populations = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "population rate matrix has no unique normalized steady state"
        ) from error

    minimum = float(np.min(populations))
    if minimum < -population_absolute_tolerance:
        raise RuntimeError(
            f"steady-state solve produced a negative population ({minimum:.6e})"
        )
    # A weakly driven excited state can have a small *positive* population
    # whose decay flow is still measurable. Discarding it breaks the optical
    # flow balance even though the population is below the negativity tolerance.
    populations = np.clip(populations, 0.0, None)
    populations /= np.sum(populations)
    residual = float(np.max(np.abs(matrix @ populations)))
    if residual > residual_relative_tolerance * scale:
        raise RuntimeError(
            "steady-state residual exceeds tolerance: "
            f"{residual:.6e} s^-1"
        )
    if not np.isclose(np.sum(populations), 1.0, rtol=0.0, atol=1.0e-13):
        raise RuntimeError("steady-state populations are not normalized")
    return populations


def rate_equation_observable(
    model: RateEquationModel,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    coil_config,
    config: MultilevelMOTConfig | None = None,
    previous_axis: Vec3 = (0.0, 0.0, 1.0),
    *,
    store_rate_matrix: bool = False,
    store_beam_transition_quantities: bool = False,
) -> RateEquationObservable:
    cfg = config or default_multilevel_mot_config()
    field_t = local_magnetic_field_t(position_m, coil_config)
    axis = quantization_axis(field_t, previous_axis, cfg.magnetic_field_epsilon_t)
    return rate_equation_observable_from_local_environment(
        model,
        beams,
        position_m,
        velocity_m_per_s,
        field_t,
        axis,
        cfg,
        store_rate_matrix=store_rate_matrix,
        store_beam_transition_quantities=store_beam_transition_quantities,
    )


def rate_equation_observable_from_local_environment(
    model: RateEquationModel,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    magnetic_field_t: Vec3,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig | None = None,
    *,
    transition_resonance_shift_rad_per_s: np.ndarray | None = None,
    store_rate_matrix: bool = False,
    store_beam_transition_quantities: bool = False,
) -> RateEquationObservable:
    """Evaluate the complete Section-12 chain at one local environment."""

    cfg = config or default_multilevel_mot_config()
    field = np.asarray(magnetic_field_t, dtype=float)
    if field.shape != (3,) or not np.all(np.isfinite(field)):
        raise ValueError("magnetic_field_t must be a finite three-vector")
    axis_array = np.asarray(quantization_axis_vector, dtype=float)
    axis_norm = float(np.linalg.norm(axis_array))
    if axis_array.shape != (3,) or not np.all(np.isfinite(axis_array)) or axis_norm <= 0.0:
        raise ValueError("quantization axis must be a finite nonzero three-vector")
    axis = tuple(float(value) for value in axis_array / axis_norm)

    quantities = build_beam_transition_quantities(
        model,
        beams,
        position_m,
        velocity_m_per_s,
        float(np.linalg.norm(field)),
        axis,
        cfg,
        transition_resonance_shift_rad_per_s=transition_resonance_shift_rad_per_s,
    )
    total_stimulated = np.sum(quantities.stimulated_coefficients_per_s, axis=0)
    rate_matrix = assemble_rate_matrix(
        total_stimulated,
        model.spontaneous_decay_matrix_per_s,
    )
    populations = steady_state_populations(
        rate_matrix,
        conservation_absolute_tolerance_per_s=cfg.conservation_absolute_tolerance_per_s,
        residual_relative_tolerance=cfg.steady_state_relative_tolerance,
        population_absolute_tolerance=cfg.population_absolute_tolerance,
    )
    ground_populations = populations[: model.ground_count]
    excited_populations = populations[model.ground_count :]
    matrices = quantities.stimulated_coefficients_per_s
    absorption = np.sum(matrices * ground_populations[None, None, :], axis=(1, 2))
    stimulated_emission = np.sum(
        matrices * excited_populations[None, :, None],
        axis=(1, 2),
    )
    effective_scattering = absorption - stimulated_emission
    spontaneous_scattering = float(
        np.dot(model.excited_decay_rates_per_s, excited_populations)
    )
    balance_tolerance = max(
        cfg.conservation_absolute_tolerance_per_s,
        cfg.steady_state_relative_tolerance
        * max(spontaneous_scattering, 1.0),
    )
    if not np.isclose(
        np.sum(effective_scattering),
        spontaneous_scattering,
        rtol=cfg.steady_state_relative_tolerance * 10.0,
        atol=balance_tolerance,
    ):
        raise RuntimeError(
            "steady-state optical flow does not balance spontaneous decay: "
            f"net={np.sum(effective_scattering):.9e} s^-1, "
            f"spontaneous={spontaneous_scattering:.9e} s^-1, "
            f"tolerance={balance_tolerance:.9e} s^-1"
        )

    force = np.zeros(3)
    for beam, rate in zip(beams, effective_scattering):
        force += HBAR_J_S * np.asarray(wavevector_rad_per_m(beam)) * rate
    return RateEquationObservable(
        populations=populations,
        force_n=tuple(float(value) for value in force),
        beam_effective_scattering_rates_per_s=tuple(
            float(value) for value in effective_scattering
        ),
        beam_absorption_rates_per_s=tuple(float(value) for value in absorption),
        beam_stimulated_emission_rates_per_s=tuple(
            float(value) for value in stimulated_emission
        ),
        total_spontaneous_scattering_rate_per_s=spontaneous_scattering,
        magnetic_field_t=tuple(float(value) for value in field),
        quantization_axis=axis,
        rate_matrix_per_s=rate_matrix if store_rate_matrix else None,
        beam_transition_quantities=(
            quantities if store_beam_transition_quantities else None
        ),
    )


def _append_record(
    record: RateEquationTrajectoryRecord,
    time_s: float,
    state: RateEquationAtomState,
    observable: RateEquationObservable,
) -> None:
    record.times_s.append(float(time_s))
    record.positions_m.append(state.position_m)
    record.velocities_m_per_s.append(state.velocity_m_per_s)
    record.forces_n.append(observable.force_n)
    record.beam_effective_scattering_rates_per_s.append(
        observable.beam_effective_scattering_rates_per_s
    )
    record.total_spontaneous_scattering_rates_per_s.append(
        observable.total_spontaneous_scattering_rate_per_s
    )
    record.magnetic_fields_t.append(observable.magnetic_field_t)
    record.quantization_axes.append(observable.quantization_axis)
    record.populations.append(observable.populations.copy())
    if observable.rate_matrix_per_s is not None:
        record.rate_matrices_per_s.append(observable.rate_matrix_per_s.copy())
    if observable.beam_transition_quantities is not None:
        quantities = observable.beam_transition_quantities
        record.beam_transition_quantities.append(
            BeamTransitionQuantities(
                rabi_frequencies_rad_per_s=quantities.rabi_frequencies_rad_per_s.copy(),
                effective_detunings_rad_per_s=(
                    quantities.effective_detunings_rad_per_s.copy()
                ),
                stimulated_coefficients_per_s=(
                    quantities.stimulated_coefficients_per_s.copy()
                ),
            )
        )


def _escaped(position_m: np.ndarray, velocity_m_per_s: np.ndarray, radius_m: float) -> bool:
    radius = float(np.linalg.norm(position_m))
    return radius >= radius_m and float(np.dot(position_m, velocity_m_per_s)) > 0.0


def simulate_rate_equation_trajectory(
    initial_state: RateEquationAtomState,
    duration_s: float,
    coil_config,
    *,
    beams: list[MOTBeam] | None = None,
    model: RateEquationModel | None = None,
    config: MultilevelMOTConfig | None = None,
    trajectory_config: RateEquationTrajectoryConfig | None = None,
) -> RateEquationTrajectoryRecord:
    """Integrate classical motion with RK4 and a fresh Section-12 solve per stage."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    cfg = config or default_multilevel_mot_config()
    numerical = trajectory_config or RateEquationTrajectoryConfig()
    rate_model = model or build_rate_equation_model()
    optical_beams = build_multilevel_mot_beams(config=cfg) if beams is None else beams
    gravity = np.asarray(
        GRAVITY_ACCELERATION_M_PER_S2 if cfg.include_gravity else (0.0, 0.0, 0.0)
    )
    previous_axis = initial_state.last_quantization_axis

    def derivative(position: np.ndarray, velocity: np.ndarray):
        observable = rate_equation_observable(
            rate_model,
            optical_beams,
            tuple(position),
            tuple(velocity),
            coil_config,
            cfg,
            previous_axis,
        )
        acceleration = np.asarray(observable.force_n) / RB87_MASS_KG + gravity
        return velocity, acceleration

    position = np.asarray(initial_state.position_m, dtype=float)
    velocity = np.asarray(initial_state.velocity_m_per_s, dtype=float)
    record = RateEquationTrajectoryRecord()
    time_s = 0.0

    def record_current() -> RateEquationObservable:
        state = RateEquationAtomState(tuple(position), tuple(velocity), previous_axis)
        observable = rate_equation_observable(
            rate_model,
            optical_beams,
            state.position_m,
            state.velocity_m_per_s,
            coil_config,
            cfg,
            previous_axis,
            store_rate_matrix=numerical.store_rate_matrices,
            store_beam_transition_quantities=numerical.store_beam_transition_quantities,
        )
        _append_record(record, time_s, state, observable)
        return observable

    observable = record_current()
    previous_axis = observable.quantization_axis
    while time_s < duration_s:
        dt_s = min(numerical.time_step_s, duration_s - time_s)
        k1_r, k1_v = derivative(position, velocity)
        k2_r, k2_v = derivative(
            position + 0.5 * dt_s * k1_r,
            velocity + 0.5 * dt_s * k1_v,
        )
        k3_r, k3_v = derivative(
            position + 0.5 * dt_s * k2_r,
            velocity + 0.5 * dt_s * k2_v,
        )
        k4_r, k4_v = derivative(
            position + dt_s * k3_r,
            velocity + dt_s * k3_v,
        )
        position = position + dt_s * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0
        velocity = velocity + dt_s * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
        time_s += dt_s
        observable = record_current()
        previous_axis = observable.quantization_axis
        if _escaped(position, velocity, numerical.escape_radius_m):
            record.termination_reason = "escaped"
            break
    return record


__all__ = [
    "BeamTransitionQuantities",
    "RateEquationAtomState",
    "RateEquationModel",
    "RateEquationObservable",
    "RateEquationTrajectoryConfig",
    "RateEquationTrajectoryRecord",
    "assemble_rate_matrix",
    "assert_rate_matrix_conserves_probability",
    "build_beam_stimulated_rate_matrices",
    "build_beam_transition_quantities",
    "build_rate_equation_model",
    "local_magnetic_field_t",
    "rate_equation_observable",
    "rate_equation_observable_from_local_environment",
    "simulate_rate_equation_trajectory",
    "steady_state_populations",
]
