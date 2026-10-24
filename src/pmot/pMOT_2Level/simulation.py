"""Deterministic effective two-level pMOT mean-force simulation.

This retains the saturated scattering closure of :mod:`pmot.mot_simple`, but
replaces the conventional Zeeman term by a differential AC-Stark transition
shift from the six trapping-light components.  No external magnetic field is
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..configuration import HBAR_J_S
from ..configuration import RB87_COOLING_RESONANCE_HZ
from ..configuration import RB87_MASS_KG
from ..configuration import RB87_REPUMP_RESONANCE_HZ
from ..cartesian_light import CartesianLightParameters
from ..cartesian_light import build_cartesian_light_bundle
from ..fields import MOTBeam
from ..mot_multilevel.polarization import polarization_weights
from ..mot_multilevel.polarization import propagation_frame_polarization
from ..mot_simple.configuration import SimpleMOTConfig
from ..mot_simple.configuration import default_simple_mot_apparatus
from ..mot_simple.configuration import default_simple_mot_config
from ..mot_simple.simulation import SimpleMOTBeam
from ..mot_simple.simulation import beam_intensity
from ..mot_simple.simulation import doppler_shift_hz
from ..mot_simple.simulation import wavevector_magnitude_m_inv
from ..mot_strange.polarizability import DifferentialPolarizabilityTable
from ..mot_strange.trapping_beams import TrappingBeam
from ..state import AtomState
from .ac_stark import PMOT2LevelStarkSample
from .ac_stark import evaluate_differential_ac_stark_shift
from .ac_stark import load_pmot_2level_polarizability_table


DEFAULT_COOLING_HELICITIES_BY_AXIS = ("sigma+", "sigma+", "sigma-")
DEFAULT_PMOT_2LEVEL_COOLING_POWER_W_PER_BEAM = 27.0e-3
DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM = 0.1e-3
DEFAULT_PMOT_2LEVEL_REPUMP_WAVELENGTH_M = 780.232684e-9


@dataclass(frozen=True, slots=True)
class PMOT2LevelLightBundle:
    """Effective two-level cooling beams plus passive repump scaffolding.

    The strict F=2 -> F'=3 closure has no F=1 population, so the repump beams
    are represented with the correct geometry and resonance metadata but do
    not enter the two-level force equation.  They become dynamically active
    only in a state-resolved multilevel population model.
    """

    cooling_beams: tuple[SimpleMOTBeam, ...]
    repump_beams: tuple[MOTBeam, ...]


def build_pmot_2level_mot_light(
    apparatus_config=None,
    simple_config: SimpleMOTConfig | None = None,
    *,
    cooling_power_w_per_beam: float | None = None,
    repump_power_w_per_beam: float = DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM,
    repump_detuning_hz: float = 0.0,
    cooling_helicities_by_axis: Sequence[str] = DEFAULT_COOLING_HELICITIES_BY_AXIS,
    repump_helicities_by_axis: Sequence[str] = DEFAULT_COOLING_HELICITIES_BY_AXIS,
) -> PMOT2LevelLightBundle:
    """Build six cooling and six resonance-tuned repump components.

    Powers and detunings are explicit model parameters even though the two
    families share the central Cartesian geometry builder.
    """

    apparatus = apparatus_config or default_simple_mot_apparatus()
    simple = simple_config or default_simple_mot_config()
    cooling_power = (
        float(cooling_power_w_per_beam)
        if cooling_power_w_per_beam is not None
        else (
            apparatus.cooling.power_w_per_beam
            if apparatus_config is not None
            else DEFAULT_PMOT_2LEVEL_COOLING_POWER_W_PER_BEAM
        )
    )
    parameters = CartesianLightParameters(
        cooling_power_w_per_beam=cooling_power,
        cooling_detuning_hz=simple.cooling_detuning_hz,
        cooling_wavelength_m=apparatus.cooling.wavelength_m,
        cooling_resonance_frequency_hz=RB87_COOLING_RESONANCE_HZ,
        repump_power_w_per_beam=repump_power_w_per_beam,
        repump_detuning_hz=repump_detuning_hz,
        repump_wavelength_m=DEFAULT_PMOT_2LEVEL_REPUMP_WAVELENGTH_M,
        repump_resonance_frequency_hz=RB87_REPUMP_RESONANCE_HZ,
        cooling_helicities_by_axis=tuple(cooling_helicities_by_axis),
        repump_helicities_by_axis=tuple(repump_helicities_by_axis),
    )
    shared = build_cartesian_light_bundle(apparatus, parameters)
    cooling = tuple(
        SimpleMOTBeam(
            label=beam.label,
            axis_name=beam.axis_name,
            propagation_sense=beam.propagation_sense,
            circular_polarization=beam.circular_polarization,
            direction=beam.direction,
            wavelength_m=beam.wavelength_m,
            intensity_beam=beam,
            polarization_sign=simple.axis_polarization_sign[beam.axis_name],
            detuning_hz=beam.detuning_hz,
        )
        for beam in shared.cooling_beams
    )
    return PMOT2LevelLightBundle(cooling, shared.repump_beams)


def build_pmot_2level_cooling_beams(
    apparatus_config=None,
    simple_config: SimpleMOTConfig | None = None,
    *,
    helicities_by_axis: Sequence[str] = DEFAULT_COOLING_HELICITIES_BY_AXIS,
) -> list[SimpleMOTBeam]:
    """Build six cooling components with matched pMOT path helicities.

    ``helicities_by_axis`` is ordered x, y, z. Each label is applied to both
    propagation directions on its round-trip path because these are
    propagation-frame labels at the atom. The default ``(sigma+, sigma+,
    sigma-)`` is the current provisional no-coil pMOT candidate, not a
    laboratory waveplate prescription or a validated design conclusion.
    """

    return list(
        build_pmot_2level_mot_light(
            apparatus_config,
            simple_config,
            cooling_helicities_by_axis=helicities_by_axis,
        ).cooling_beams
    )


def build_pmot_2level_repump_beams(
    apparatus_config=None,
    simple_config: SimpleMOTConfig | None = None,
    *,
    power_w_per_beam: float = DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM,
    detuning_hz: float = 0.0,
    helicities_by_axis: Sequence[str] = DEFAULT_COOLING_HELICITIES_BY_AXIS,
) -> list[MOTBeam]:
    """Return six passive repump components at the MOT repump resonance."""

    return list(
        build_pmot_2level_mot_light(
            apparatus_config,
            simple_config,
            repump_power_w_per_beam=power_w_per_beam,
            repump_detuning_hz=detuning_hz,
            repump_helicities_by_axis=helicities_by_axis,
        ).repump_beams
    )


@dataclass(frozen=True, slots=True)
class PMOT2LevelRateSample:
    """One cooling beam's locally decomposed effective scattering response."""

    beam_label: str
    intensity_w_per_m2: float
    saturation_parameter: float
    total_saturation_parameter: float
    doppler_shift_hz: float
    polarization_weights_q_minus_pi_q_plus: tuple[float, float, float]
    ac_stark_shifts_hz_q_minus_pi_q_plus: tuple[float, float, float]
    effective_detunings_hz_q_minus_pi_q_plus: tuple[float, float, float]
    scattering_rates_per_s_q_minus_pi_q_plus: tuple[float, float, float]
    total_scattering_rate_per_s: float


@dataclass(frozen=True, slots=True)
class PMOT2LevelLocalObservable:
    """Complete effective two-level pMOT result at one phase-space point."""

    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    stark: PMOT2LevelStarkSample
    rate_samples: tuple[PMOT2LevelRateSample, ...]
    force_n: tuple[float, float, float]
    acceleration_m_per_s2: tuple[float, float, float]

    @property
    def total_scattering_rate_per_s(self) -> float:
        return float(sum(sample.total_scattering_rate_per_s for sample in self.rate_samples))


@dataclass(frozen=True, slots=True)
class PMOT2LevelTrajectoryRecord:
    """Saved deterministic two-level pMOT trajectory and local observables."""

    times_s: tuple[float, ...]
    positions_m: tuple[tuple[float, float, float], ...]
    velocities_m_per_s: tuple[tuple[float, float, float], ...]
    forces_n: tuple[tuple[float, float, float], ...]
    quantization_axes: tuple[tuple[float, float, float], ...]
    axis_defined_by_vector_shift: tuple[bool, ...]
    total_ac_stark_shifts_hz_for_sigma_plus: tuple[float, ...]
    total_scattering_rates_per_s: tuple[float, ...]


def _finite_vector3(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


def evaluate_pmot_2level_local_observable(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    position_m,
    velocity_m_per_s,
    *,
    path_powers_w: Sequence[float],
    previous_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
) -> PMOT2LevelLocalObservable:
    """Evaluate Stark shifts, six cooling rates, and the 3D mean force.

    The effective F=2 -> F'=3 closure splits each cooling beam into local
    q=-1,0,+1 fractions.  For channel ``q`` the ordinary-frequency detuning is

    ``delta_eff[b,q] = delta_0[b] - k_b.v/(2*pi) - DeltaE_AC[q]/h``.

    Rates from all three fractions and all six beams are summed
    deterministically.  The model does not select individual photons or evolve
    m_F populations. Since ``linewidth_hz = Gamma/(2*pi)``, the ordinary-
    frequency scattering prefactor is ``Gamma/2 = pi*linewidth_hz``.
    """

    cooling = tuple(cooling_beams)
    if len(cooling) != 6:
        raise ValueError("cooling_beams must contain six Cartesian components")
    position = _finite_vector3(position_m, name="position_m")
    velocity = _finite_vector3(velocity_m_per_s, name="velocity_m_per_s")
    simple = simple_config or default_simple_mot_config()
    table = polarizability_table or load_pmot_2level_polarizability_table()
    stark = evaluate_differential_ac_stark_shift(
        trapping_beams,
        position,
        velocity,
        path_powers_w=path_powers_w,
        previous_axis=previous_axis,
        polarizability_table=table,
    )

    intensities = np.asarray([beam_intensity(beam, tuple(position)) for beam in cooling])
    saturations = intensities / simple.saturation_intensity_w_per_m2
    total_saturation = float(np.sum(saturations))
    ac_shifts = np.asarray(stark.transition_shifts_hz_by_q, dtype=float)
    axis = stark.quantization_axis

    force = np.zeros(3, dtype=float)
    samples: list[PMOT2LevelRateSample] = []
    for beam, intensity, saturation in zip(cooling, intensities, saturations):
        weights_by_q = polarization_weights(
            propagation_frame_polarization(
                beam.direction,
                beam.circular_polarization,
            ),
            axis,
        )
        weights = np.asarray([weights_by_q[q] for q in (-1, 0, +1)])
        doppler = doppler_shift_hz(beam, tuple(velocity))
        effective_detunings = beam.detuning_hz - doppler - ac_shifts
        rates = (
            simple.maximum_scattering_rate_per_s
            * saturation
            * weights
            / (
                1.0
                + total_saturation
                + (2.0 * effective_detunings / simple.linewidth_hz) ** 2
            )
        )
        total_rate = float(np.sum(rates))
        force += (
            HBAR_J_S
            * wavevector_magnitude_m_inv(beam)
            * total_rate
            * np.asarray(beam.direction, dtype=float)
        )
        samples.append(
            PMOT2LevelRateSample(
                beam_label=beam.label,
                intensity_w_per_m2=float(intensity),
                saturation_parameter=float(saturation),
                total_saturation_parameter=total_saturation,
                doppler_shift_hz=float(doppler),
                polarization_weights_q_minus_pi_q_plus=tuple(
                    float(value) for value in weights
                ),
                ac_stark_shifts_hz_q_minus_pi_q_plus=tuple(
                    float(value) for value in ac_shifts
                ),
                effective_detunings_hz_q_minus_pi_q_plus=tuple(
                    float(value) for value in effective_detunings
                ),
                scattering_rates_per_s_q_minus_pi_q_plus=tuple(
                    float(value) for value in rates
                ),
                total_scattering_rate_per_s=total_rate,
            )
        )

    acceleration = force / RB87_MASS_KG
    if simple.include_gravity:
        acceleration += np.asarray(simple.gravity_acceleration_m_per_s2, dtype=float)
    return PMOT2LevelLocalObservable(
        position_m=tuple(float(value) for value in position),
        velocity_m_per_s=tuple(float(value) for value in velocity),
        stark=stark,
        rate_samples=tuple(samples),
        force_n=tuple(float(value) for value in force),
        acceleration_m_per_s2=tuple(float(value) for value in acceleration),
    )


def rk4_step_pmot_2level(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    atom_state: AtomState,
    time_step_s: float,
    *,
    path_powers_w: Sequence[float],
    previous_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
) -> tuple[AtomState, PMOT2LevelLocalObservable]:
    """Advance one RK4 step, recomputing the complete local problem per stage."""

    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    simple = simple_config or default_simple_mot_config()
    table = polarizability_table or load_pmot_2level_polarizability_table()
    position = np.asarray(atom_state.position_m, dtype=float)
    velocity = np.asarray(atom_state.velocity_m_per_s, dtype=float)

    def local(stage_position, stage_velocity, stage_previous_axis):
        return evaluate_pmot_2level_local_observable(
            cooling_beams,
            trapping_beams,
            stage_position,
            stage_velocity,
            path_powers_w=path_powers_w,
            previous_axis=stage_previous_axis,
            simple_config=simple,
            polarizability_table=table,
        )

    observable_1 = local(position, velocity, previous_axis)
    k1_r = velocity
    k1_v = np.asarray(observable_1.acceleration_m_per_s2)
    observable_2 = local(
        position + 0.5 * time_step_s * k1_r,
        velocity + 0.5 * time_step_s * k1_v,
        observable_1.stark.quantization_axis,
    )
    k2_r = velocity + 0.5 * time_step_s * k1_v
    k2_v = np.asarray(observable_2.acceleration_m_per_s2)
    observable_3 = local(
        position + 0.5 * time_step_s * k2_r,
        velocity + 0.5 * time_step_s * k2_v,
        observable_2.stark.quantization_axis,
    )
    k3_r = velocity + 0.5 * time_step_s * k2_v
    k3_v = np.asarray(observable_3.acceleration_m_per_s2)
    observable_4 = local(
        position + time_step_s * k3_r,
        velocity + time_step_s * k3_v,
        observable_3.stark.quantization_axis,
    )
    k4_r = velocity + time_step_s * k3_v
    k4_v = np.asarray(observable_4.acceleration_m_per_s2)

    next_position = position + (time_step_s / 6.0) * (
        k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r
    )
    next_velocity = velocity + (time_step_s / 6.0) * (
        k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
    )
    endpoint = local(
        next_position,
        next_velocity,
        observable_4.stark.quantization_axis,
    )
    return (
        AtomState(
            position_m=tuple(float(value) for value in next_position),
            velocity_m_per_s=tuple(float(value) for value in next_velocity),
        ),
        endpoint,
    )


def simulate_pmot_2level_trajectory(
    cooling_beams: Sequence[SimpleMOTBeam],
    trapping_beams: Sequence[TrappingBeam],
    initial_state: AtomState,
    duration_s: float,
    time_step_s: float,
    *,
    path_powers_w: Sequence[float],
    initial_quantization_axis=(0.0, 0.0, 1.0),
    simple_config: SimpleMOTConfig | None = None,
    polarizability_table: DifferentialPolarizabilityTable | None = None,
) -> PMOT2LevelTrajectoryRecord:
    """Integrate a deterministic no-coil effective two-level pMOT trajectory."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    simple = simple_config or default_simple_mot_config()
    table = polarizability_table or load_pmot_2level_polarizability_table()
    state = initial_state
    observable = evaluate_pmot_2level_local_observable(
        cooling_beams,
        trapping_beams,
        state.position_m,
        state.velocity_m_per_s,
        path_powers_w=path_powers_w,
        previous_axis=initial_quantization_axis,
        simple_config=simple,
        polarizability_table=table,
    )

    times = [0.0]
    positions = [state.position_m]
    velocities = [state.velocity_m_per_s]
    observables = [observable]
    time_s = 0.0
    maximum_step_count = int(np.ceil(duration_s / time_step_s))
    endpoint_tolerance_s = 1.0e-9 * time_step_s
    for step_index in range(maximum_step_count):
        target_time_s = min((step_index + 1) * time_step_s, duration_s)
        dt = target_time_s - time_s
        if dt <= endpoint_tolerance_s:
            continue
        state, observable = rk4_step_pmot_2level(
            cooling_beams,
            trapping_beams,
            state,
            dt,
            path_powers_w=path_powers_w,
            previous_axis=observable.stark.quantization_axis,
            simple_config=simple,
            polarizability_table=table,
        )
        time_s = target_time_s
        times.append(time_s)
        positions.append(state.position_m)
        velocities.append(state.velocity_m_per_s)
        observables.append(observable)

    return PMOT2LevelTrajectoryRecord(
        times_s=tuple(float(value) for value in times),
        positions_m=tuple(positions),
        velocities_m_per_s=tuple(velocities),
        forces_n=tuple(observable.force_n for observable in observables),
        quantization_axes=tuple(
            observable.stark.quantization_axis for observable in observables
        ),
        axis_defined_by_vector_shift=tuple(
            observable.stark.quantization_axis_from_vector_shift
            for observable in observables
        ),
        total_ac_stark_shifts_hz_for_sigma_plus=tuple(
            observable.stark.total_transition_shift_hz_for_sigma_plus
            for observable in observables
        ),
        total_scattering_rates_per_s=tuple(
            observable.total_scattering_rate_per_s for observable in observables
        ),
    )


__all__ = [
    "DEFAULT_COOLING_HELICITIES_BY_AXIS",
    "DEFAULT_PMOT_2LEVEL_COOLING_POWER_W_PER_BEAM",
    "DEFAULT_PMOT_2LEVEL_REPUMP_POWER_W_PER_BEAM",
    "DEFAULT_PMOT_2LEVEL_REPUMP_WAVELENGTH_M",
    "PMOT2LevelLightBundle",
    "PMOT2LevelLocalObservable",
    "PMOT2LevelRateSample",
    "PMOT2LevelTrajectoryRecord",
    "build_pmot_2level_cooling_beams",
    "build_pmot_2level_mot_light",
    "build_pmot_2level_repump_beams",
    "evaluate_pmot_2level_local_observable",
    "rk4_step_pmot_2level",
    "simulate_pmot_2level_trajectory",
]
