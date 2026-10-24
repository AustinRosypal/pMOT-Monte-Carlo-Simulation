"""Short diagnostic trajectories for the provisional pMOT Stark model."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2
from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import RB87_MASS_KG
from ..fields import MOTBeam
from ..mot_multilevel.configuration import MultilevelMOTConfig
from ..mot_multilevel.rate_equations import RateEquationAtomState
from ..mot_multilevel.rate_equations import RateEquationModel
from ..mot_multilevel.rate_equations import RateEquationObservable
from ..mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from ..mot_multilevel.rate_equations import RateEquationTrajectoryRecord
from ..mot_multilevel.rate_equations import rate_equation_observable_from_local_environment
from .ac_stark import ProvisionalStarkConfig
from .ac_stark import ProvisionalStarkObservable
from .ac_stark import provisional_transition_stark_shifts
from .polarizability import DifferentialPolarizabilityTable
from .polarizability import load_differential_polarizability_table
from .trapping_beams import TrappingBeam
from .trapping_beams import TrappingLaserConfig


@dataclass(frozen=True, slots=True)
class ProvisionalPMOTObservable:
    """The unchanged 24-state light observable plus its Stark environment."""

    rate_equation: RateEquationObservable
    stark: ProvisionalStarkObservable


@dataclass(slots=True)
class ProvisionalPMOTTrajectoryRecord:
    """Rate-equation trajectory plus pMOT-specific diagnostic histories."""

    rate_equation: RateEquationTrajectoryRecord = field(
        default_factory=RateEquationTrajectoryRecord
    )
    atom_frame_frequencies_hz: list[tuple[float, ...]] = field(default_factory=list)
    atom_frame_wavelengths_nm: list[tuple[float, ...]] = field(default_factory=list)
    trapping_component_intensities_w_per_m2: list[tuple[float, ...]] = field(
        default_factory=list
    )
    effective_fields_t: list[tuple[float, float, float]] = field(default_factory=list)
    quantization_axes: list[tuple[float, float, float]] = field(default_factory=list)
    reference_scalar_shift_hz: list[float] = field(default_factory=list)
    reference_vector_shift_hz: list[float] = field(default_factory=list)
    reference_tensor_shift_hz: list[float] = field(default_factory=list)
    reference_total_shift_hz: list[float] = field(default_factory=list)


def _cycling_transition_index(model: RateEquationModel) -> int:
    for index, transition in enumerate(model.structure.transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the rate model is missing the stretched cycling transition")


def provisional_pmot_observable(
    model: RateEquationModel,
    cooling_repump_beams: list[MOTBeam],
    trapping_beams: list[TrappingBeam],
    position_m,
    velocity_m_per_s,
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
    multilevel_config: MultilevelMOTConfig,
    previous_axis=(0.0, 0.0, 1.0),
    *,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
    store_rate_matrix: bool = False,
    store_beam_transition_quantities: bool = False,
) -> ProvisionalPMOTObservable:
    """Evaluate one no-coil pMOT phase point.

    The external field passed to the 24-state kernel is exactly zero.  The
    trapping light supplies a local basis and a transition-resonance shift;
    it is never represented as a real magnetic field inside the force solver.
    """

    stark = provisional_transition_stark_shifts(
        model,
        trapping_beams,
        position_m,
        velocity_m_per_s,
        laser_config,
        stark_config,
        previous_axis,
        polarizability_table=polarizability_table,
    )
    rate = rate_equation_observable_from_local_environment(
        model,
        cooling_repump_beams,
        tuple(float(value) for value in position_m),
        tuple(float(value) for value in velocity_m_per_s),
        (0.0, 0.0, 0.0),
        stark.quantization_axis,
        multilevel_config,
        transition_resonance_shift_rad_per_s=(
            stark.transition_angular_frequency_shift_rad_per_s
        ),
        store_rate_matrix=store_rate_matrix,
        store_beam_transition_quantities=store_beam_transition_quantities,
    )
    return ProvisionalPMOTObservable(rate, stark)


def _append_record(
    record: ProvisionalPMOTTrajectoryRecord,
    time_s: float,
    state: RateEquationAtomState,
    observable: ProvisionalPMOTObservable,
    reference_index: int,
) -> None:
    rate = observable.rate_equation
    stark = observable.stark
    base = record.rate_equation
    base.times_s.append(float(time_s))
    base.positions_m.append(state.position_m)
    base.velocities_m_per_s.append(state.velocity_m_per_s)
    base.forces_n.append(rate.force_n)
    base.total_spontaneous_scattering_rates_per_s.append(
        rate.total_spontaneous_scattering_rate_per_s
    )
    base.beam_effective_scattering_rates_per_s.append(
        rate.beam_effective_scattering_rates_per_s
    )
    base.magnetic_fields_t.append(rate.magnetic_field_t)
    base.quantization_axes.append(rate.quantization_axis)
    base.populations.append(rate.populations.copy())
    if rate.rate_matrix_per_s is not None:
        base.rate_matrices_per_s.append(rate.rate_matrix_per_s.copy())
    if rate.beam_transition_quantities is not None:
        base.beam_transition_quantities.append(rate.beam_transition_quantities)
    record.atom_frame_frequencies_hz.append(stark.atom_frame_frequencies_hz)
    record.atom_frame_wavelengths_nm.append(stark.atom_frame_wavelengths_nm)
    record.trapping_component_intensities_w_per_m2.append(
        stark.component_intensities_w_per_m2
    )
    record.effective_fields_t.append(stark.effective_field_t)
    record.quantization_axes.append(stark.quantization_axis)
    record.reference_scalar_shift_hz.append(
        float(stark.scalar_transition_energy_j[reference_index] / PLANCK_CONSTANT_J_S)
    )
    record.reference_vector_shift_hz.append(
        float(stark.vector_transition_energy_j[reference_index] / PLANCK_CONSTANT_J_S)
    )
    record.reference_tensor_shift_hz.append(
        float(stark.tensor_transition_energy_j[reference_index] / PLANCK_CONSTANT_J_S)
    )
    record.reference_total_shift_hz.append(
        float(stark.transition_frequency_shift_hz[reference_index])
    )


def _escaped(position_m, velocity_m_per_s, radius_m: float) -> bool:
    position = np.asarray(position_m, dtype=float)
    radius = float(np.linalg.norm(position))
    if radius < radius_m:
        return False
    return float(np.dot(position, velocity_m_per_s)) / max(radius, 1.0e-15) > 0.0


def simulate_provisional_pmot_trajectory(
    initial_state: RateEquationAtomState,
    duration_s: float,
    model: RateEquationModel,
    cooling_repump_beams: list[MOTBeam],
    trapping_beams: list[TrappingBeam],
    laser_config: TrappingLaserConfig,
    stark_config: ProvisionalStarkConfig,
    multilevel_config: MultilevelMOTConfig,
    *,
    trajectory_config: RateEquationTrajectoryConfig | None = None,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
    progress_callback=None,
) -> ProvisionalPMOTTrajectoryRecord:
    """Integrate deterministic motion with a fresh Section-12 solve at every RK4 stage."""

    numerical = trajectory_config or RateEquationTrajectoryConfig()
    if duration_s <= 0.0 or numerical.time_step_s <= 0.0:
        raise ValueError("duration and timestep must be positive")
    if numerical.escape_radius_m <= 0.0:
        raise ValueError("escape radius must be positive")
    table = polarizability_table or load_differential_polarizability_table()
    record = ProvisionalPMOTTrajectoryRecord()
    position = np.asarray(initial_state.position_m, dtype=float)
    velocity = np.asarray(initial_state.velocity_m_per_s, dtype=float)
    previous_axis = initial_state.last_quantization_axis
    time_s = 0.0
    reference_index = _cycling_transition_index(model)
    gravity = np.asarray(
        GRAVITY_ACCELERATION_M_PER_S2
        if multilevel_config.include_gravity
        else (0.0, 0.0, 0.0),
        dtype=float,
    )

    def evaluate(stage_position: np.ndarray, stage_velocity: np.ndarray, *, store=False):
        return provisional_pmot_observable(
            model,
            cooling_repump_beams,
            trapping_beams,
            tuple(float(value) for value in stage_position),
            tuple(float(value) for value in stage_velocity),
            laser_config,
            stark_config,
            multilevel_config,
            previous_axis,
            polarizability_table=table,
            store_rate_matrix=store and numerical.store_rate_matrices,
            store_beam_transition_quantities=(
                store and numerical.store_beam_transition_quantities
            ),
        )

    def derivative(stage_position: np.ndarray, stage_velocity: np.ndarray):
        stage = evaluate(stage_position, stage_velocity)
        acceleration = (
            np.asarray(stage.rate_equation.force_n, dtype=float) / RB87_MASS_KG
            + gravity
        )
        return stage_velocity, acceleration

    def record_current() -> ProvisionalPMOTObservable:
        state = RateEquationAtomState(
            tuple(float(value) for value in position),
            tuple(float(value) for value in velocity),
            previous_axis,
        )
        current = evaluate(position, velocity, store=True)
        _append_record(record, time_s, state, current, reference_index)
        return current

    observable = record_current()
    previous_axis = observable.stark.quantization_axis
    step_ratio = duration_s / numerical.time_step_s
    rounded_steps = int(round(step_ratio))
    expected_steps = (
        rounded_steps
        if abs(step_ratio - rounded_steps) <= 1.0e-10 * max(1.0, abs(step_ratio))
        else int(np.ceil(step_ratio))
    )
    next_progress_step = max(1, expected_steps // 5)
    for completed_steps in range(1, expected_steps + 1):
        dt_s = min(numerical.time_step_s, duration_s - time_s)
        if dt_s <= 0.0:
            break
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
        position = position + dt_s * (
            k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r
        ) / 6.0
        velocity = velocity + dt_s * (
            k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
        ) / 6.0
        time_s = min(duration_s, time_s + dt_s)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            record.rate_equation.termination_reason = "non_finite"
            break
        observable = record_current()
        previous_axis = observable.stark.quantization_axis
        if progress_callback is not None and (
            completed_steps % next_progress_step == 0
            or completed_steps == expected_steps
        ):
            progress_callback(completed_steps, expected_steps, time_s)
        if _escaped(
            position,
            velocity,
            numerical.escape_radius_m,
        ):
            record.rate_equation.termination_reason = "escaped"
            break
    return record


__all__ = [name for name in globals() if not name.startswith("_")]
