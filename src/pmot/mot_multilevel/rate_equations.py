"""Efficient adiabatic-elimination rate-equation model for the full Rb-87 MOT.

The main solver propagates steady-state populations rather than individual
hyperfine quantum jumps.  Optical coherences and sub-Doppler physics are
intentionally outside this approximation; see EFFICIENT_MOT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import pi

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
from .coupling import beam_polarization_vector, wavevector_rad_per_m
from .polarization import Vec3, polarization_weights, quantization_axis
from .simulation import build_multilevel_mot_beams


@dataclass(frozen=True, slots=True)
class RateEquationModel:
    """Precomputed state, dipole, decay, and sparse-transition arrays."""

    structure: AtomicStructure
    ground_indices: np.ndarray
    excited_indices: np.ndarray
    global_to_ground: np.ndarray
    global_to_excited: np.ndarray
    transition_ground: np.ndarray
    transition_excited: np.ndarray
    transition_q: np.ndarray
    transition_strength: np.ndarray
    transition_ground_f: np.ndarray
    transition_excited_f: np.ndarray
    transition_hyperfine_offset_rad_per_s: np.ndarray
    transition_zeeman_coefficient: np.ndarray
    dipole_tensor: np.ndarray
    decay_rate_matrix_per_s: np.ndarray

    @property
    def ground_count(self) -> int:
        return len(self.ground_indices)

    @property
    def excited_count(self) -> int:
        return len(self.excited_indices)

    @property
    def state_count(self) -> int:
        return self.ground_count + self.excited_count


@dataclass(frozen=True, slots=True)
class RateEquationObservable:
    """Steady-state populations and mechanical coefficients at one phase point."""

    populations: np.ndarray
    force_n: Vec3
    diffusion_kg2_m2_per_s3: float
    total_scattering_rate_per_s: float
    beam_scattering_rates_per_s: tuple[float, ...]
    magnetic_field_t: Vec3
    quantization_axis: Vec3
    rate_matrix_per_s: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class RateEquationAtomState:
    position_m: Vec3
    velocity_m_per_s: Vec3
    last_quantization_axis: Vec3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class RateEquationTrajectoryConfig:
    """Fixed-timestep external-motion controls for the efficient solver."""

    time_step_s: float = 5.0e-6
    include_diffusion: bool = True
    seed: int = 20260819
    escape_radius_m: float = 30.0e-3
    store_rate_matrices: bool = False


@dataclass(slots=True)
class RateEquationTrajectoryRecord:
    times_s: list[float] = field(default_factory=list)
    positions_m: list[Vec3] = field(default_factory=list)
    velocities_m_per_s: list[Vec3] = field(default_factory=list)
    forces_n: list[Vec3] = field(default_factory=list)
    diffusion_kg2_m2_per_s3: list[float] = field(default_factory=list)
    total_scattering_rates_per_s: list[float] = field(default_factory=list)
    beam_scattering_rates_per_s: list[tuple[float, ...]] = field(default_factory=list)
    magnetic_fields_t: list[Vec3] = field(default_factory=list)
    populations: list[np.ndarray] = field(default_factory=list)
    termination_reason: str = "duration"


def precompute_dipole_tensor(structure: AtomicStructure) -> np.ndarray:
    """Return D[q,e,g], with squared amplitude normalized to cycling strength."""

    ground_lookup = {index: local for local, index in enumerate(structure.ground_state_indices)}
    excited_lookup = {index: local for local, index in enumerate(structure.excited_state_indices)}
    tensor = np.zeros((3, len(structure.excited_state_indices), len(structure.ground_state_indices)))
    for transition in structure.absorption_transitions:
        tensor[transition.q + 1, excited_lookup[transition.excited_state_index], ground_lookup[transition.ground_state_index]] = np.sqrt(transition.c_squared)
    return tensor


def precompute_decay_rate_matrix(
    structure: AtomicStructure,
    gamma_per_s: float,
) -> np.ndarray:
    """Return spontaneous rates decay[g,e], whose columns sum to Gamma."""

    ground_lookup = {index: local for local, index in enumerate(structure.ground_state_indices)}
    excited_lookup = {index: local for local, index in enumerate(structure.excited_state_indices)}
    decay = np.zeros((len(ground_lookup), len(excited_lookup)))
    for channel in structure.decay_channels:
        decay[ground_lookup[channel.ground_state_index], excited_lookup[channel.excited_state_index]] += gamma_per_s * channel.branch_probability
    return decay


@lru_cache(maxsize=4)
def build_rate_equation_model(
    gamma_per_s: float | None = None,
) -> RateEquationModel:
    """Build and cache all position-independent rate-equation quantities."""

    structure = build_atomic_structure()
    gamma = gamma_per_s or default_multilevel_mot_config().natural_linewidth_rad_per_s
    ground_indices = np.asarray(structure.ground_state_indices, dtype=int)
    excited_indices = np.asarray(structure.excited_state_indices, dtype=int)
    global_to_ground = np.full(len(structure.states), -1, dtype=int)
    global_to_excited = np.full(len(structure.states), -1, dtype=int)
    global_to_ground[ground_indices] = np.arange(len(ground_indices))
    global_to_excited[excited_indices] = np.arange(len(excited_indices))
    transitions = structure.absorption_transitions
    transition_ground = np.asarray([global_to_ground[t.ground_state_index] for t in transitions], dtype=int)
    transition_excited = np.asarray([global_to_excited[t.excited_state_index] for t in transitions], dtype=int)
    ground_states = [structure.states[t.ground_state_index] for t in transitions]
    excited_states = [structure.states[t.excited_state_index] for t in transitions]
    zeeman_coefficient = BOHR_MAGNETON_OVER_HBAR_RAD_PER_S_PER_T * np.asarray([
        excited.lande_g * excited.m_f - ground.lande_g * ground.m_f
        for ground, excited in zip(ground_states, excited_states)
    ])
    return RateEquationModel(
        structure=structure,
        ground_indices=ground_indices,
        excited_indices=excited_indices,
        global_to_ground=global_to_ground,
        global_to_excited=global_to_excited,
        transition_ground=transition_ground,
        transition_excited=transition_excited,
        transition_q=np.asarray([t.q for t in transitions], dtype=int),
        transition_strength=np.asarray([t.c_squared for t in transitions], dtype=float),
        transition_ground_f=np.asarray([t.ground_f for t in transitions], dtype=int),
        transition_excited_f=np.asarray([t.excited_f for t in transitions], dtype=int),
        transition_hyperfine_offset_rad_per_s=np.asarray([t.hyperfine_offset_rad_per_s for t in transitions]),
        transition_zeeman_coefficient=zeeman_coefficient,
        dipole_tensor=precompute_dipole_tensor(structure),
        decay_rate_matrix_per_s=precompute_decay_rate_matrix(structure, gamma),
    )


def _local_field(position_m: Vec3, coil_config) -> Vec3:
    values = anti_helmholtz_field_t(*position_m, coil_config)
    return tuple(float(np.asarray(value)) for value in values)


def build_beam_stimulated_rate_matrices(
    model: RateEquationModel,
    beams: list[MOTBeam],
    position_m: Vec3,
    velocity_m_per_s: Vec3,
    field_magnitude_t: float,
    quantization_axis_vector: Vec3,
    config: MultilevelMOTConfig,
    *,
    transition_resonance_shift_rad_per_s: np.ndarray | None = None,
) -> np.ndarray:
    """Return incoherently summed beam rates W[b,e,g] in s^-1.

    Different MOT beams are treated as mutually incoherent. Their rates, not
    complex Rabi amplitudes, are therefore added in the total rate matrix.
    """

    gamma = config.natural_linewidth_rad_per_s
    if transition_resonance_shift_rad_per_s is None:
        transition_shift = np.zeros(len(model.transition_ground), dtype=float)
    else:
        transition_shift = np.asarray(
            transition_resonance_shift_rad_per_s,
            dtype=float,
        )
        if transition_shift.shape != model.transition_ground.shape:
            raise ValueError(
                "transition_resonance_shift_rad_per_s must have one value "
                "per absorption transition"
            )
        if not np.all(np.isfinite(transition_shift)):
            raise ValueError(
                "transition_resonance_shift_rad_per_s must contain only finite values"
            )
    beam_rates = np.zeros((len(beams), model.excited_count, model.ground_count))
    q_columns = model.transition_q + 1
    f2_reference_offset = model.structure.states[
        model.structure.state_index("excited", 2, 0)
    ].energy_offset_rad_per_s
    velocity = np.asarray(velocity_m_per_s)
    for beam_index, beam in enumerate(beams):
        if beam.family == "cooling":
            enabled = np.isin(model.transition_excited_f, config.enabled_excited_manifolds)
            addressed = model.transition_ground_f == 2
            laser_minus_hyperfine = config.cooling_detuning_rad_per_s - model.transition_hyperfine_offset_rad_per_s
        elif beam.family == "repump":
            if not config.repumper_enabled or beam.power_w <= 0.0:
                continue
            enabled = np.isin(model.transition_excited_f, config.enabled_repump_excited_manifolds)
            addressed = model.transition_ground_f == 1
            laser_minus_hyperfine = config.repump_detuning_rad_per_s - (
                model.transition_hyperfine_offset_rad_per_s - f2_reference_offset
            )
        else:
            continue
        mask = enabled & addressed
        if not np.any(mask):
            continue
        intensity_ratio = beam_intensity_w_per_m2(beam, position_m) / config.saturation_intensity_w_per_m2
        weights = polarization_weights(beam_polarization_vector(beam), quantization_axis_vector)
        polarization = np.asarray([weights[-1], weights[0], weights[1]])[q_columns]
        saturation = intensity_ratio * model.transition_strength * polarization
        doppler = float(np.dot(np.asarray(wavevector_rad_per_m(beam)), velocity))
        detuning = (
            laser_minus_hyperfine
            - doppler
            - model.transition_zeeman_coefficient * field_magnitude_t
            - transition_shift
        )
        # EFFICIENT_MOT.md transition-specific saturated Lorentzian.
        rates = 0.5 * gamma * saturation / (
            1.0 + saturation + 4.0 * detuning**2 / gamma**2
        )
        np.add.at(
            beam_rates[beam_index],
            (model.transition_excited[mask], model.transition_ground[mask]),
            rates[mask],
        )
    return beam_rates


def assemble_rate_matrix(
    stimulated_rate_matrix_per_s: np.ndarray,
    decay_rate_matrix_per_s: np.ndarray,
) -> np.ndarray:
    """Assemble the real population Liouvillian with column-wise outflow."""

    excited_count, ground_count = stimulated_rate_matrix_per_s.shape
    total_count = ground_count + excited_count
    matrix = np.zeros((total_count, total_count))
    matrix[ground_count:, :ground_count] = stimulated_rate_matrix_per_s
    # The same laser-driven rate acts in reverse for stimulated emission;
    # spontaneous decay is added independently.
    matrix[:ground_count, ground_count:] = stimulated_rate_matrix_per_s.T + decay_rate_matrix_per_s
    diagonal = np.diag_indices(total_count)
    matrix[diagonal] = -np.sum(matrix, axis=0)
    return matrix


def steady_state_populations(
    rate_matrix_per_s: np.ndarray,
    gamma_scale_per_s: float,
) -> np.ndarray:
    """Solve R p=0, sum(p)=1, and clean roundoff-scale negativity."""

    matrix = np.asarray(rate_matrix_per_s, dtype=float)
    count = matrix.shape[0]
    if matrix.shape != (count, count):
        raise ValueError("rate matrix must be square")
    system = matrix / gamma_scale_per_s
    system = system.copy()
    rhs = np.zeros(count)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    try:
        populations = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        populations = np.linalg.lstsq(system, rhs, rcond=None)[0]
    populations[np.abs(populations) < 1.0e-13] = 0.0
    populations = np.clip(populations, 0.0, None)
    normalization = float(np.sum(populations))
    if normalization <= 0.0:
        raise RuntimeError("steady-state population solve returned zero population")
    return populations / normalization


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
) -> RateEquationObservable:
    """Solve quasi-steady populations and return force and recoil diffusion."""

    cfg = config or default_multilevel_mot_config()
    field_t = _local_field(position_m, coil_config)
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
) -> RateEquationObservable:
    """Solve the 24-state rate equations for an explicit local environment.

    The conventional MOT wrapper supplies its anti-Helmholtz field and Zeeman
    quantization axis.  Other apparatus models may supply a different local
    axis and a transition-resonance shift while retaining the identical
    cooling/repump population and radiation-pressure calculation.  A positive
    transition-resonance shift is subtracted from the laser detuning.
    """

    cfg = config or default_multilevel_mot_config()
    field_t = tuple(float(value) for value in magnetic_field_t)
    if len(field_t) != 3 or not np.all(np.isfinite(field_t)):
        raise ValueError("magnetic_field_t must contain three finite values")
    axis_array = np.asarray(quantization_axis_vector, dtype=float)
    axis_norm = float(np.linalg.norm(axis_array))
    if axis_array.shape != (3,) or not np.all(np.isfinite(axis_array)) or axis_norm <= 0.0:
        raise ValueError("quantization_axis_vector must be a finite nonzero 3-vector")
    axis = tuple(float(value) for value in axis_array / axis_norm)
    beam_matrices = build_beam_stimulated_rate_matrices(
        model,
        beams,
        position_m,
        velocity_m_per_s,
        float(np.linalg.norm(field_t)),
        axis,
        cfg,
        transition_resonance_shift_rad_per_s=transition_resonance_shift_rad_per_s,
    )
    total_stimulated = np.sum(beam_matrices, axis=0)
    rate_matrix = assemble_rate_matrix(total_stimulated, model.decay_rate_matrix_per_s)
    populations = steady_state_populations(rate_matrix, cfg.natural_linewidth_rad_per_s)
    ground_populations = populations[: model.ground_count]
    beam_scattering = np.asarray([
        float(np.sum(matrix * ground_populations[None, :])) for matrix in beam_matrices
    ])
    force = np.zeros(3)
    diffusion = 0.0
    for beam, scattering_rate in zip(beams, beam_scattering):
        photon_momentum = HBAR_J_S * np.asarray(wavevector_rad_per_m(beam))
        force += photon_momentum * scattering_rate
        diffusion += 0.5 * float(np.dot(photon_momentum, photon_momentum)) * scattering_rate
    return RateEquationObservable(
        populations=populations,
        force_n=tuple(force),
        diffusion_kg2_m2_per_s3=diffusion,
        total_scattering_rate_per_s=float(np.sum(beam_scattering)),
        beam_scattering_rates_per_s=tuple(beam_scattering),
        magnetic_field_t=field_t,
        quantization_axis=axis,
        rate_matrix_per_s=rate_matrix if store_rate_matrix else None,
    )


def _append_record(
    record: RateEquationTrajectoryRecord,
    time_s: float,
    state: RateEquationAtomState,
    observable: RateEquationObservable,
) -> None:
    record.times_s.append(time_s)
    record.positions_m.append(state.position_m)
    record.velocities_m_per_s.append(state.velocity_m_per_s)
    record.forces_n.append(observable.force_n)
    record.diffusion_kg2_m2_per_s3.append(observable.diffusion_kg2_m2_per_s3)
    record.total_scattering_rates_per_s.append(observable.total_scattering_rate_per_s)
    record.beam_scattering_rates_per_s.append(observable.beam_scattering_rates_per_s)
    record.magnetic_fields_t.append(observable.magnetic_field_t)
    record.populations.append(observable.populations.copy())


def _escaped(position_m: Vec3, velocity_m_per_s: Vec3, radius_m: float) -> bool:
    position = np.asarray(position_m)
    radius = float(np.linalg.norm(position))
    return radius >= radius_m and float(np.dot(position, velocity_m_per_s)) / max(radius, 1e-15) > 0.0


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
    """Integrate the external Langevin trajectory at fixed microsecond-scale dt."""

    cfg = config or default_multilevel_mot_config()
    numerical = trajectory_config or RateEquationTrajectoryConfig()
    if duration_s <= 0.0 or numerical.time_step_s <= 0.0 or numerical.escape_radius_m <= 0.0:
        raise ValueError("duration, timestep, and escape radius must be positive")
    rate_model = model or build_rate_equation_model(cfg.natural_linewidth_rad_per_s)
    optical_beams = build_multilevel_mot_beams(config=cfg) if beams is None else beams
    rng = np.random.default_rng(numerical.seed)
    record = RateEquationTrajectoryRecord()
    state = initial_state
    time_s = 0.0
    observable = rate_equation_observable(
        rate_model, optical_beams, state.position_m, state.velocity_m_per_s,
        coil_config, cfg, state.last_quantization_axis,
        store_rate_matrix=numerical.store_rate_matrices,
    )
    _append_record(record, time_s, state, observable)
    gravity = np.asarray(GRAVITY_ACCELERATION_M_PER_S2 if cfg.include_gravity else (0.0, 0.0, 0.0))
    while time_s < duration_s:
        dt_s = min(numerical.time_step_s, duration_s - time_s)
        momentum = RB87_MASS_KG * np.asarray(state.velocity_m_per_s)
        momentum += (np.asarray(observable.force_n) + RB87_MASS_KG * gravity) * dt_s
        if numerical.include_diffusion and observable.diffusion_kg2_m2_per_s3 > 0.0:
            momentum += np.sqrt(2.0 * observable.diffusion_kg2_m2_per_s3 * dt_s) * rng.normal(size=3)
        velocity = momentum / RB87_MASS_KG
        position = np.asarray(state.position_m) + velocity * dt_s
        time_s += dt_s
        state = RateEquationAtomState(tuple(position), tuple(velocity), observable.quantization_axis)
        observable = rate_equation_observable(
            rate_model, optical_beams, state.position_m, state.velocity_m_per_s,
            coil_config, cfg, state.last_quantization_axis,
            store_rate_matrix=numerical.store_rate_matrices,
        )
        _append_record(record, time_s, state, observable)
        if _escaped(state.position_m, state.velocity_m_per_s, numerical.escape_radius_m):
            record.termination_reason = "escaped"
            break
    return record


__all__ = [name for name in globals() if not name.startswith("_")]
