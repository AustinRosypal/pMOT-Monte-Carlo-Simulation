"""Configurable diagnostic trajectories for the ideal-magic vector-only pMOT.

This module deliberately sits on the provisional side of the pMOT validation
boundary.  It reuses the unchanged 24-state cooling/repump population-rate
kernel, sets the external magnetic field to exactly zero, and applies only the
1529-nm vector differential-transition shift.  Scalar and tensor shifts are
suppressed by the imposed ideal-magic assumption.  The 1529-nm light does not
contribute a direct mechanical force in this model; it changes the local
transition resonances and the retained 780-nm beams provide the recorded
radiation-pressure force.

The default propagation-frame helicities are the locally validated design:
``(sigma+, sigma+, sigma-)`` on x, y, and z for both propagation directions of
the 780-nm and 1529-nm fields.  Every traveling component remains independently
configurable for notebook experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from math import cos
from math import pi
from math import sin
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2
from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import RB87_MASS_KG
from ..fields import MOTBeam
from ..launch_geometry import build_incident_disc_from_angles
from ..mot_multilevel.configuration import MultilevelMOTConfig
from ..mot_multilevel.configuration import default_multilevel_mot_config
from ..mot_multilevel.rate_equations import RateEquationAtomState
from ..mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from ..mot_multilevel.rate_equations import RateEquationTrajectoryRecord
from .ac_stark import ProvisionalStarkConfig
from .ac_stark import build_physics_trapping_beams
from .ac_stark import provisional_power_for_target_gradient_w_per_path
from .configuration import PMOTApparatusConfig
from .configuration import build_pmot_cooling_and_repump_beams
from .configuration import default_pmot_apparatus_config
from .polarizability import load_differential_polarizability_table
from .trapping_beams import DEFAULT_TRAPPING_AXES
from .trapping_beams import TrappingBeam
from .trapping_beams import TrappingLaserConfig
from .vector_only import VectorOnlyContext
from .vector_only import VectorOnlyObservable
from .vector_only import vector_only_pmot_observable


DEFAULT_PATH_HELICITIES_XYZ = ("sigma+", "sigma+", "sigma-")
SUPPORTED_PROPAGATION_FRAME_POLARIZATIONS = ("sigma+", "sigma-", "pi")
SUPPORTED_TRAPPING_HELICITIES = ("sigma+", "sigma-")
STANDARD_LAUNCH_DISTANCE_M = 15.0e-3
STANDARD_LAUNCH_SPEED_M_PER_S = 17.0
STANDARD_POLAR_ANGLE_DEG = 90.0
STANDARD_AZIMUTH_ANGLE_DEG = 0.0
VECTOR_ONLY_TRAJECTORY_MODEL_METADATA = {
    "model": "ideal_magic_scalar_tensor_cancelled_vector_transition_proxy",
    "external_magnetic_field_t": [0.0, 0.0, 0.0],
    "scalar_transition_shift_included": False,
    "tensor_transition_shift_included": False,
    "vector_transition_shift_included": True,
    "direct_1529nm_mechanical_force_included": False,
    "conservative_stark_gradient_force_included": False,
    "trap_light_scattering_heating_loss_included": False,
    "force_source": "inherited 780-nm ground-population-weighted absorption proxy",
}


def _validate_three_labels(
    name: str,
    values: tuple[str, str, str],
    supported: tuple[str, ...],
) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain x, y, and z values")
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise ValueError(f"{name} contains unsupported values: {unsupported}")


@dataclass(frozen=True, slots=True)
class PMOTBeamHelicities:
    """Propagation-frame polarization of every pMOT traveling component.

    Each tuple is ordered ``(x, y, z)``.  Incident and retro tuples are
    independent, as are the cooling and repump families.  ``pi`` is accepted
    for 780-nm exploratory controls; the current vector-only trapping ansatz
    requires circular ``sigma+`` or ``sigma-`` components.
    """

    cooling_incident_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ
    cooling_retro_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ
    repump_incident_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ
    repump_retro_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ
    trapping_incident_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ
    trapping_retro_xyz: tuple[str, str, str] = DEFAULT_PATH_HELICITIES_XYZ

    def __post_init__(self) -> None:
        for name in (
            "cooling_incident_xyz",
            "cooling_retro_xyz",
            "repump_incident_xyz",
            "repump_retro_xyz",
        ):
            _validate_three_labels(
                name,
                getattr(self, name),
                SUPPORTED_PROPAGATION_FRAME_POLARIZATIONS,
            )
        for name in ("trapping_incident_xyz", "trapping_retro_xyz"):
            _validate_three_labels(
                name,
                getattr(self, name),
                SUPPORTED_TRAPPING_HELICITIES,
            )


@dataclass(frozen=True, slots=True)
class VectorOnlyPMOTTrajectoryContext:
    """Resolved optical, atomic, and vector-only inputs for one trajectory."""

    apparatus: PMOTApparatusConfig
    multilevel_config: MultilevelMOTConfig
    model: Any
    polarizability_table: Any
    cooling_repump_beams: tuple[MOTBeam, ...]
    trapping_beams: tuple[TrappingBeam, ...]
    stark_config: ProvisionalStarkConfig
    trapping_power_w_per_path: float
    trapping_power_source: str
    target_gradient_g_per_cm: float | None
    helicities: PMOTBeamHelicities
    _observable_context: VectorOnlyContext
    trapping_code: str


@dataclass(frozen=True, slots=True)
class VectorOnlyCaptureStatus:
    """Early-exit capture criterion evaluated over a stored trajectory."""

    trapped: bool
    classification: str
    entered_core: bool
    core_entry_count: int
    maximum_continuous_core_residence_s: float
    required_continuous_core_residence_s: float
    required_core_entries: int
    core_radius_m: float
    minimum_radius_m: float
    final_radius_m: float


@dataclass(slots=True)
class VectorOnlyPMOTTrajectoryRecord:
    """24-state trajectory and pMOT vector-shift diagnostic histories."""

    rate_equation: RateEquationTrajectoryRecord = field(
        default_factory=RateEquationTrajectoryRecord
    )
    atom_frame_wavelengths_nm: list[tuple[float, ...]] = field(default_factory=list)
    atom_frame_frequencies_hz: list[tuple[float, ...]] = field(default_factory=list)
    trapping_component_intensities_w_per_m2: list[tuple[float, ...]] = field(
        default_factory=list
    )
    effective_fields_t: list[tuple[float, float, float]] = field(default_factory=list)
    quantization_axes: list[tuple[float, float, float]] = field(default_factory=list)
    reference_vector_shift_hz: list[float] = field(default_factory=list)
    applied_reference_transition_shift_hz: list[float] = field(default_factory=list)
    capture: VectorOnlyCaptureStatus | None = None
    model_metadata: dict[str, Any] = field(
        default_factory=lambda: dict(VECTOR_ONLY_TRAJECTORY_MODEL_METADATA)
    )


def _path_symbol(label: str) -> str:
    if label == "sigma+":
        return "+"
    if label == "sigma-":
        return "-"
    raise ValueError("trapping components must use sigma+ or sigma-")


def _trapping_code(helicities: PMOTBeamHelicities) -> str:
    return "".join(
        _path_symbol(value)
        for values in (
            helicities.trapping_incident_xyz,
            helicities.trapping_retro_xyz,
        )
        for value in values
    )


def _apply_780_helicities(
    beams: list[MOTBeam],
    helicities: PMOTBeamHelicities,
) -> tuple[MOTBeam, ...]:
    axis_index = {axis_name: index for index, axis_name in enumerate(DEFAULT_TRAPPING_AXES)}
    output: list[MOTBeam] = []
    for beam in beams:
        family_values = (
            helicities.cooling_incident_xyz
            if beam.family == "cooling" and beam.propagation_sense == "incident"
            else helicities.cooling_retro_xyz
            if beam.family == "cooling"
            else helicities.repump_incident_xyz
            if beam.propagation_sense == "incident"
            else helicities.repump_retro_xyz
        )
        output.append(
            replace(
                beam,
                circular_polarization=family_values[axis_index[beam.axis_name]],
            )
        )
    return tuple(output)


def build_vector_only_apparatus(
    *,
    cooling_power_w_per_beam: float = 27.0e-3,
    repump_power_w_per_beam: float = 0.1e-3,
    cooling_detuning_hz: float = -15.0e6,
    repump_detuning_hz: float = 0.0,
    beam_diameter_m: float = 12.7e-3,
    trapping_wavelength_m: float = 1529.268881e-9,
    trapping_focus_offset_m: float = 10.0e-3,
    trapping_input_beam_diameter_m: float = 35.0e-3,
    trapping_focal_length_m: float = 80.3e-3,
    trapping_incident_waist_radius_m: float | None = None,
    trapping_retro_waist_radius_m: float | None = None,
    trapping_retro_power_fraction: float = 1.0,
) -> PMOTApparatusConfig:
    """Build a notebook-friendly pMOT apparatus from physical controls."""

    base = default_pmot_apparatus_config(
        trapping_wavelength_m=trapping_wavelength_m,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
        repump_power_w_per_beam=repump_power_w_per_beam,
    )
    mot_light = replace(
        base.mot_light,
        cooling=replace(
            base.mot_light.cooling,
            power_w_per_beam=cooling_power_w_per_beam,
            detuning_hz=cooling_detuning_hz,
            beam_diameter_m=beam_diameter_m,
        ),
        repump=replace(
            base.mot_light.repump,
            power_w_per_beam=repump_power_w_per_beam,
            detuning_hz=repump_detuning_hz,
            beam_diameter_m=beam_diameter_m,
        ),
    )
    trapping_laser = TrappingLaserConfig(
        wavelength_m=trapping_wavelength_m,
        focus_offset_m=trapping_focus_offset_m,
        input_beam_diameter_m=trapping_input_beam_diameter_m,
        focal_length_m=trapping_focal_length_m,
        incident_waist_radius_m=trapping_incident_waist_radius_m,
        retro_waist_radius_m=trapping_retro_waist_radius_m,
        retro_power_fraction=trapping_retro_power_fraction,
    )
    return replace(base, mot_light=mot_light, trapping_laser=trapping_laser)


def build_vector_only_trajectory_context(
    *,
    apparatus: PMOTApparatusConfig | None = None,
    multilevel_config: MultilevelMOTConfig | None = None,
    helicities: PMOTBeamHelicities | None = None,
    trapping_power_w_per_path: float | None = None,
    target_gradient_g_per_cm: float = 20.0,
) -> VectorOnlyPMOTTrajectoryContext:
    """Resolve a no-coil vector-only pMOT trajectory environment.

    If no trapping power is given, the historical diagnostic power scale for
    the requested stretched-reference vector-gradient proxy is calculated.
    It remains a provisional scale, not an apparatus power recommendation.
    """

    optical = apparatus or default_pmot_apparatus_config()
    selected_helicities = helicities or PMOTBeamHelicities()
    if multilevel_config is None:
        config = replace(
            default_multilevel_mot_config(),
            cooling_detuning_rad_per_s=2.0 * pi * optical.mot_light.cooling.detuning_hz,
            repump_detuning_rad_per_s=2.0 * pi * optical.mot_light.repump.detuning_hz,
            repumper_enabled=True,
            repump_power_w_per_beam=optical.mot_light.repump.power_w_per_beam,
        )
    else:
        config = replace(multilevel_config, repumper_enabled=True)
    # Local import avoids constructing the cached 24-state graph during module import.
    from ..mot_multilevel.rate_equations import build_rate_equation_model

    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    table = load_differential_polarizability_table()
    if trapping_power_w_per_path is None:
        power = provisional_power_for_target_gradient_w_per_path(
            model,
            optical.trapping_laser,
            target_gradient_g_per_cm=target_gradient_g_per_cm,
            polarizability_table=table,
        )
        power_source = "derived provisional stretched-reference vector-gradient proxy"
        recorded_gradient: float | None = float(target_gradient_g_per_cm)
    else:
        if not np.isfinite(trapping_power_w_per_path) or trapping_power_w_per_path < 0.0:
            raise ValueError("trapping_power_w_per_path must be finite and non-negative")
        power = float(trapping_power_w_per_path)
        power_source = "explicit value"
        recorded_gradient = None
    cooling_repump = _apply_780_helicities(
        build_pmot_cooling_and_repump_beams(optical, config),
        selected_helicities,
    )
    stark_config = ProvisionalStarkConfig(
        incident_path_powers_w=(power, power, power),
        incident_helicities_by_axis=selected_helicities.trapping_incident_xyz,
        retro_helicities_by_axis=selected_helicities.trapping_retro_xyz,
    )
    trapping_beams = tuple(
        build_physics_trapping_beams(optical.trapping_laser, stark_config)
    )
    observable_context = VectorOnlyContext(
        apparatus=optical,
        multilevel_config=config,
        model=model,
        polarizability_table=table,
        cooling_repump_beams=cooling_repump,
        power_w_per_path=power,
    )
    return VectorOnlyPMOTTrajectoryContext(
        apparatus=optical,
        multilevel_config=config,
        model=model,
        polarizability_table=table,
        cooling_repump_beams=cooling_repump,
        trapping_beams=trapping_beams,
        stark_config=stark_config,
        trapping_power_w_per_path=power,
        trapping_power_source=power_source,
        target_gradient_g_per_cm=recorded_gradient,
        helicities=selected_helicities,
        _observable_context=observable_context,
        trapping_code=_trapping_code(selected_helicities),
    )


def vector_only_trajectory_observable(
    context: VectorOnlyPMOTTrajectoryContext,
    position_m,
    velocity_m_per_s,
    previous_axis=(0.0, 0.0, 1.0),
) -> VectorOnlyObservable:
    """Evaluate the corrected vector-only observable with external B exactly zero."""

    return vector_only_pmot_observable(
        context._observable_context,
        context.trapping_code,
        position_m,
        velocity_m_per_s,
        previous_axis=previous_axis,
        prepared=(context.stark_config, list(context.trapping_beams)),
    )


def inward_launch_state(
    *,
    radial_distance_m: float = STANDARD_LAUNCH_DISTANCE_M,
    speed_m_per_s: float = STANDARD_LAUNCH_SPEED_M_PER_S,
    polar_angle_deg: float = STANDARD_POLAR_ANGLE_DEG,
    azimuth_angle_deg: float = STANDARD_AZIMUTH_ANGLE_DEG,
    impact_parameter_m: float = 0.0,
    impact_azimuth_deg: float = 0.0,
) -> RateEquationAtomState:
    """Build the established incident-disc launch, directed parallel inward.

    The standard launch is a clean axial shot from ``(+15, 0, 0) mm`` with
    velocity ``(-17, 0, 0) m/s`` and zero impact parameter.  A nonzero impact
    parameter moves the starting point within the perpendicular launch plane
    without aiming that offset point separately at the origin.
    """

    numeric = (
        radial_distance_m,
        speed_m_per_s,
        polar_angle_deg,
        azimuth_angle_deg,
        impact_parameter_m,
        impact_azimuth_deg,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("launch parameters must be finite")
    if radial_distance_m <= 0.0 or speed_m_per_s < 0.0 or impact_parameter_m < 0.0:
        raise ValueError("launch distance must be positive; speed and impact must be non-negative")
    disc = build_incident_disc_from_angles(
        0,
        radial_distance_m,
        np.deg2rad(polar_angle_deg),
        np.deg2rad(azimuth_angle_deg),
    )
    angle = np.deg2rad(impact_azimuth_deg)
    offset = impact_parameter_m * (
        cos(angle) * np.asarray(disc.basis_u)
        + sin(angle) * np.asarray(disc.basis_v)
    )
    position = np.asarray(disc.center_position_m) + offset
    velocity = speed_m_per_s * np.asarray(disc.incident_unit_vector)
    return RateEquationAtomState(
        tuple(float(value) for value in position),
        tuple(float(value) for value in velocity),
    )


def classify_vector_only_trajectory(
    record: RateEquationTrajectoryRecord,
    *,
    core_radius_m: float = 2.0e-3,
    required_core_entries: int = 2,
    required_continuous_core_residence_s: float = 5.0e-3,
) -> VectorOnlyCaptureStatus:
    """Apply the project's two-entry-or-five-ms diagnostic trapped criterion."""

    if core_radius_m <= 0.0 or required_continuous_core_residence_s <= 0.0:
        raise ValueError("core radius and required residence must be positive")
    if required_core_entries <= 0:
        raise ValueError("required_core_entries must be positive")
    if not record.times_s or len(record.times_s) != len(record.positions_m):
        raise ValueError("trajectory record must contain aligned time and position samples")
    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("trajectory positions must have shape (n, 3)")
    radii = np.linalg.norm(positions, axis=1)
    inside = radii <= core_radius_m
    entries = int(inside[0]) + int(np.count_nonzero(inside[1:] & ~inside[:-1]))
    maximum_residence = 0.0
    entry_time: float | None = float(times[0]) if inside[0] else None
    for index in range(1, len(times)):
        if inside[index] and not inside[index - 1]:
            entry_time = float(times[index])
        if not inside[index] and inside[index - 1] and entry_time is not None:
            # The boundary crossing lies between samples; only the last stored
            # inside time is established by the numerical record.
            maximum_residence = max(
                maximum_residence,
                float(times[index - 1] - entry_time),
            )
            entry_time = None
    if inside[-1] and entry_time is not None:
        maximum_residence = max(maximum_residence, float(times[-1] - entry_time))
    two_entries = entries >= required_core_entries
    residence = maximum_residence >= required_continuous_core_residence_s - 1.0e-15
    if two_entries:
        classification = "two_core_entries"
    elif residence:
        classification = "bounded_core_residence"
    elif record.termination_reason == "escaped":
        classification = "escaped"
    elif record.termination_reason == "non_finite":
        classification = "non_finite"
    else:
        classification = "duration_not_trapped"
    return VectorOnlyCaptureStatus(
        trapped=bool(two_entries or residence),
        classification=classification,
        entered_core=bool(np.any(inside)),
        core_entry_count=entries,
        maximum_continuous_core_residence_s=maximum_residence,
        required_continuous_core_residence_s=required_continuous_core_residence_s,
        required_core_entries=required_core_entries,
        core_radius_m=core_radius_m,
        minimum_radius_m=float(np.min(radii)),
        final_radius_m=float(radii[-1]),
    )


def _cycling_transition_index(context: VectorOnlyPMOTTrajectoryContext) -> int:
    for index, transition in enumerate(context.model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the rate model is missing the stretched cycling transition")


def _append_sample(
    record: VectorOnlyPMOTTrajectoryRecord,
    time_s: float,
    state: RateEquationAtomState,
    observable: VectorOnlyObservable,
    reference_index: int,
) -> None:
    rate = observable.rate_equation
    stark = observable.stark_diagnostic
    base = record.rate_equation
    base.times_s.append(float(time_s))
    base.positions_m.append(state.position_m)
    base.velocities_m_per_s.append(state.velocity_m_per_s)
    base.forces_n.append(rate.force_n)
    base.diffusion_kg2_m2_per_s3.append(rate.diffusion_kg2_m2_per_s3)
    base.total_scattering_rates_per_s.append(rate.total_scattering_rate_per_s)
    base.beam_scattering_rates_per_s.append(rate.beam_scattering_rates_per_s)
    base.magnetic_fields_t.append(rate.magnetic_field_t)
    base.populations.append(rate.populations.copy())
    record.atom_frame_wavelengths_nm.append(stark.atom_frame_wavelengths_nm)
    record.atom_frame_frequencies_hz.append(stark.atom_frame_frequencies_hz)
    record.trapping_component_intensities_w_per_m2.append(
        stark.component_intensities_w_per_m2
    )
    record.effective_fields_t.append(stark.effective_field_t)
    record.quantization_axes.append(stark.quantization_axis)
    vector_shift_hz = float(
        stark.vector_transition_energy_j[reference_index] / PLANCK_CONSTANT_J_S
    )
    record.reference_vector_shift_hz.append(vector_shift_hz)
    record.applied_reference_transition_shift_hz.append(
        float(observable.applied_transition_shift_rad_per_s[reference_index] / (2.0 * pi))
    )


def _escaped(position_m, velocity_m_per_s, escape_radius_m: float) -> bool:
    position = np.asarray(position_m, dtype=float)
    radius = float(np.linalg.norm(position))
    radial_velocity = float(np.dot(position, velocity_m_per_s)) / max(radius, 1.0e-15)
    return radius >= escape_radius_m and radial_velocity > 0.0


def simulate_vector_only_pmot_trajectory(
    initial_state: RateEquationAtomState | None = None,
    *,
    duration_s: float = 25.0e-3,
    context: VectorOnlyPMOTTrajectoryContext | None = None,
    trajectory_config: RateEquationTrajectoryConfig | None = None,
    core_radius_m: float = 2.0e-3,
    required_core_entries: int = 2,
    required_continuous_core_residence_s: float = 5.0e-3,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> VectorOnlyPMOTTrajectoryRecord:
    """Integrate a vector-only pMOT diagnostic trajectory.

    The default run is deterministic (mean force, no recoil diffusion).  Set
    ``RateEquationTrajectoryConfig.include_diffusion=True`` for an optional
    Langevin recoil realization.  Gravity follows ``context.multilevel_config``
    and is enabled by the repository default.
    """

    environment = context or build_vector_only_trajectory_context()
    numerical = trajectory_config or RateEquationTrajectoryConfig(
        include_diffusion=False
    )
    state = initial_state or inward_launch_state()
    if duration_s <= 0.0 or numerical.time_step_s <= 0.0:
        raise ValueError("duration and timestep must be positive")
    if numerical.escape_radius_m <= 0.0:
        raise ValueError("escape radius must be positive")
    rng = np.random.default_rng(numerical.seed)
    record = VectorOnlyPMOTTrajectoryRecord()
    reference_index = _cycling_transition_index(environment)
    time_s = 0.0
    observable = vector_only_trajectory_observable(
        environment,
        state.position_m,
        state.velocity_m_per_s,
        state.last_quantization_axis,
    )
    _append_sample(record, time_s, state, observable, reference_index)
    gravity = np.asarray(
        GRAVITY_ACCELERATION_M_PER_S2
        if environment.multilevel_config.include_gravity
        else (0.0, 0.0, 0.0),
        dtype=float,
    )
    expected_steps = int(np.ceil(duration_s / numerical.time_step_s))
    progress_interval = max(1, expected_steps // 20)
    for completed_steps in range(1, expected_steps + 1):
        dt_s = min(numerical.time_step_s, duration_s - time_s)
        if dt_s <= 0.0:
            break
        momentum = RB87_MASS_KG * np.asarray(state.velocity_m_per_s, dtype=float)
        momentum += (
            np.asarray(observable.rate_equation.force_n, dtype=float)
            + RB87_MASS_KG * gravity
        ) * dt_s
        if numerical.include_diffusion and observable.rate_equation.diffusion_kg2_m2_per_s3 > 0.0:
            momentum += np.sqrt(
                2.0 * observable.rate_equation.diffusion_kg2_m2_per_s3 * dt_s
            ) * rng.normal(size=3)
        velocity = momentum / RB87_MASS_KG
        position = np.asarray(state.position_m, dtype=float) + velocity * dt_s
        time_s = min(duration_s, time_s + dt_s)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            record.rate_equation.termination_reason = "non_finite"
            break
        state = RateEquationAtomState(
            tuple(float(value) for value in position),
            tuple(float(value) for value in velocity),
            observable.stark_diagnostic.quantization_axis,
        )
        observable = vector_only_trajectory_observable(
            environment,
            state.position_m,
            state.velocity_m_per_s,
            state.last_quantization_axis,
        )
        _append_sample(record, time_s, state, observable, reference_index)
        if progress_callback is not None and (
            completed_steps % progress_interval == 0
            or completed_steps == expected_steps
        ):
            progress_callback(completed_steps, expected_steps, time_s)
        if _escaped(state.position_m, state.velocity_m_per_s, numerical.escape_radius_m):
            record.rate_equation.termination_reason = "escaped"
            break
    record.capture = classify_vector_only_trajectory(
        record.rate_equation,
        core_radius_m=core_radius_m,
        required_core_entries=required_core_entries,
        required_continuous_core_residence_s=required_continuous_core_residence_s,
    )
    return record


def vector_only_trajectory_dataframe(
    record: VectorOnlyPMOTTrajectoryRecord,
) -> pd.DataFrame:
    """Return the principal trajectory and vector diagnostics as one table."""

    base = record.rate_equation
    positions = np.asarray(base.positions_m, dtype=float)
    velocities = np.asarray(base.velocities_m_per_s, dtype=float)
    forces = np.asarray(base.forces_n, dtype=float)
    fields = np.asarray(record.effective_fields_t, dtype=float)
    axes = np.asarray(record.quantization_axes, dtype=float)
    intensities = np.asarray(record.trapping_component_intensities_w_per_m2, dtype=float)
    data: dict[str, Any] = {"time_s": np.asarray(base.times_s, dtype=float)}
    for index, axis_name in enumerate("xyz"):
        data[f"{axis_name}_m"] = positions[:, index]
        data[f"v{axis_name}_m_per_s"] = velocities[:, index]
        data[f"F{axis_name}_n"] = forces[:, index]
        data[f"effective_field_proxy_{axis_name}_t"] = fields[:, index]
        data[f"quantization_axis_{axis_name}"] = axes[:, index]
    data["radius_m"] = np.linalg.norm(positions, axis=1)
    data["speed_m_per_s"] = np.linalg.norm(velocities, axis=1)
    data["total_780_scattering_rate_per_s"] = np.asarray(
        base.total_scattering_rates_per_s,
        dtype=float,
    )
    data["reference_vector_shift_hz"] = np.asarray(
        record.reference_vector_shift_hz,
        dtype=float,
    )
    data["applied_reference_transition_shift_hz"] = np.asarray(
        record.applied_reference_transition_shift_hz,
        dtype=float,
    )
    data["total_trapping_intensity_w_per_m2"] = np.sum(intensities, axis=1)
    return pd.DataFrame(data)


def save_vector_only_trajectory_csv(
    record: VectorOnlyPMOTTrajectoryRecord,
    path: str | Path,
) -> Path:
    """Save the notebook-facing trajectory table to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vector_only_trajectory_dataframe(record).to_csv(output, index=False)
    return output


__all__ = [name for name in globals() if not name.startswith("_")]
