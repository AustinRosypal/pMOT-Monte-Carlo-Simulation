"""Checkpointed August 22 loading sweeps for the multilevel Rb-87 MOT.

The optical force in this module is always evaluated with the population-rate
model in :mod:`pmot.mot_multilevel.rate_equations`.  The reusable launch-disc,
capture-threshold, and loading-integral data structures are shared with the
two-level workflow, but no two-level force or trajectory routine is used.

Capture searches deliberately disable recoil diffusion.  A deterministic
mean force is required for the local monotonicity assumption behind the
trapped/untrapped speed bracket.  The model still solves the complete
multilevel steady-state population problem at every external-motion step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from math import pi
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..configuration import (
    GRAVITY_ACCELERATION_M_PER_S2,
    MOTApparatusConfig,
    RB87_MASS_KG,
    default_mot_apparatus_config,
)
from ..fields import MOTBeam
from ..magnetic_fields import default_anti_helmholtz_config
from ..capture_statistics import (
    CaptureVelocitySample,
    TrajectoryClassification,
    VelocitySpectrumSample,
    load_capture_velocity_samples,
)
from ..launch_geometry import (
    DiscSample,
    PointSample,
    sample_incident_disc,
)
from .configuration import (
    MultilevelMOTConfig,
    default_multilevel_mot_config,
    multilevel_mot_paths,
)
from .loading import calculate_loading_rate_from_spectrum, calculate_sampling_uncertainty
from .rate_capture import (
    RateCaptureSearchConfig,
)
from .rate_equations import (
    RateEquationModel,
    build_rate_equation_model,
    rate_equation_observable,
)
from .simulation import (
    build_multilevel_cooling_beams,
    build_multilevel_mot_beams,
    build_multilevel_repump_beams,
)


SATURATION_N_VALUES: tuple[float, ...] = tuple(float(value) for value in range(1, 36))
"""Requested peak cooling-beam intensity ratios, ``I0 / I_sat = 1, ..., 35``."""


_BEAM_DIAMETER_STEP_MM = (30.0 - 12.7) / 17.0
BEAM_DIAMETER_MM_VALUES: tuple[float, ...] = tuple(
    12.7 + (index - 7) * _BEAM_DIAMETER_STEP_MM for index in range(25)
)
"""Twenty-five even diameters with 12.7 mm at index 7 and 30 mm last."""


DEFAULT_CAPTURE_DISC_RADIUS_M = 12.0e-3
DEFAULT_BEAM_DIAMETER_M = 12.7e-3
DEFAULT_COOLING_POWER_W_PER_BEAM = 20.0e-3
USER_SATURATION_INTENSITY_W_PER_M2 = 16.7
DEFAULT_PARAMETER_WORKER_COUNT = max(1, min(24, os.cpu_count() or 1))
PLOT_POINT_DISC_COUNT = 10
POINTS_PER_DISC = 25
# Backward-compatible names retained for callers of the preliminary sweep
# module.  A replicate is now one randomly oriented disc containing 25 launch
# points, not one launch point.
PLOT_POINT_REPLICATE_COUNT = PLOT_POINT_DISC_COUNT
POINTS_PER_REPLICATE = POINTS_PER_DISC
SAMPLE_SHARD_SIZE = 5
"""Maximum same-disc launch points evaluated by one worker task."""

_STUDENT_T_95_DF9 = 2.2621571627409915

_FORMAT_VERSION = 4
_MODEL_NAME = "multilevel steady-state population-rate MOT"
_CAPTURE_DYNAMICS = (
    "deterministic multilevel mean force; 24 indexed states (the 23-state "
    "cooling specification plus repumper-required F'=0); recoil diffusion disabled"
)
_TRAPPED_TERMINATION_REASONS = frozenset(
    {"two_core_entries", "bounded_core_residence"}
)
_VALID_UNTRAPPED_TERMINATION_REASONS = frozenset({"escaped", "timeout"})
_BEAM_SIZE_APERTURE_LIMITATION = (
    "The study is aperture-limited by the fixed 12 mm-radius sampling disc, which caps "
    "the reported capture cross section at "
    "pi*(12 mm)^2. Because the random design inserts no exact boundary point, any "
    "high-diameter aperture limitation must be assessed from the saved near-rim samples."
)
_CAPTURE_FIELDS = (
    "disc_index",
    "point_index",
    "theta_rad",
    "phi_rad",
    "theta_prime_rad",
    "s_m",
    "radial_distance_m",
    "capture_velocity_m_per_s",
    "velocity_resolution_m_per_s",
    "trapped_velocity_lower_m_per_s",
    "untrapped_velocity_upper_m_per_s",
    "lower_classification",
    "upper_classification",
    "lower_entered_trap_core",
    "upper_entered_trap_core",
    "lower_core_entry_count",
    "upper_core_entry_count",
    "x0_m",
    "y0_m",
    "z0_m",
    "vx_hat",
    "vy_hat",
    "vz_hat",
)
_SPECTRUM_FIELDS = (
    "velocity_m_per_s",
    "captured_count",
    "launched_count",
    "capture_fraction",
    "capture_cross_section_m2",
)
_REPLICATE_LOADING_FIELDS = (
    "replicate_index",
    "disc_index",
    "theta_rad",
    "phi_rad",
    "point_count",
    "capture_velocity_mean_m_per_s",
    "capture_velocity_sample_std_m_per_s",
    "loading_rate_atoms_per_s",
)
_COMMON_AGGREGATE_FIELDS = (
    "parameter_index",
    "parameter_key",
    "run_signature",
    "model",
    "indexed_state_count",
    "cooling_specification_state_count",
    "repumper_fprime0_extension",
    "cooling_power_w_per_beam",
    "repump_power_w_per_beam",
    "beam_diameter_mm",
    "cooling_peak_intensity_w_per_m2",
    "cooling_peak_intensity_ratio",
    "repump_peak_intensity_w_per_m2",
    "sampling_disc_radius_mm",
    "spectrum_velocity_min_m_per_s",
    "spectrum_velocity_max_m_per_s",
    "spectrum_velocity_step_m_per_s",
    "maximum_capture_threshold_m_per_s",
    "loading_rate_atoms_per_s",
    "simulation_replicate_count",
    "launch_disc_count",
    "points_per_disc",
    "capture_threshold_simulation_count",
    "replicate_loading_rate_mean_atoms_per_s",
    "replicate_loading_rate_sample_std_atoms_per_s",
    "replicate_loading_rate_standard_error_atoms_per_s",
    "replicate_loading_rate_95_percent_half_width_atoms_per_s",
    "loading_rate_disc_cluster_standard_error_atoms_per_s",
    "loading_integral_m6_per_s4",
    "capture_velocity_mean_m_per_s",
    "capture_velocity_std_m_per_s",
    "valid_bracket_count",
    "zero_capture_no_bracket_count",
    "upper_escaped_count",
    "upper_timeout_count",
    "upper_other_count",
    "sample_count",
    "geometry_sha256",
    "elapsed_s",
)
_SATURATION_AGGREGATE_FIELDS = (
    *_COMMON_AGGREGATE_FIELDS[:3],
    "n",
    *_COMMON_AGGREGATE_FIELDS[3:],
)
_BEAM_SIZE_AGGREGATE_FIELDS = _COMMON_AGGREGATE_FIELDS


def saturation_power_w_per_beam(
    n_value: float,
    *,
    saturation_intensity_w_per_m2: float | None = None,
    beam_diameter_m: float = DEFAULT_BEAM_DIAMETER_M,
) -> float:
    """Return cooling power for ``I0 / I_sat = n`` at the Gaussian waist.

    The user's August 22 value, 1.67 mW/cm^2 = 16.7 W/m^2, is used unless an
    explicit value is supplied.  The multilevel package stores all optical
    detunings and linewidths in angular-frequency units; this intensity-to-
    power conversion does not alter those internal frequency conventions.
    """

    if n_value <= 0.0 or beam_diameter_m <= 0.0:
        raise ValueError("n_value and beam_diameter_m must be positive")
    intensity = (
        USER_SATURATION_INTENSITY_W_PER_M2
        if saturation_intensity_w_per_m2 is None
        else float(saturation_intensity_w_per_m2)
    )
    if intensity <= 0.0:
        raise ValueError("saturation_intensity_w_per_m2 must be positive")
    return pi * beam_diameter_m**2 * n_value * intensity / 8.0


def beam_size_power_w_per_beam(
    beam_diameter_mm: float,
    *,
    reference_power_w: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
    reference_diameter_mm: float = 12.7,
) -> float:
    """Return power scaled as diameter squared to preserve peak intensity."""

    if beam_diameter_mm <= 0.0 or reference_diameter_mm <= 0.0:
        raise ValueError("beam diameters must be positive")
    if reference_power_w < 0.0:
        raise ValueError("reference_power_w must be non-negative")
    return reference_power_w * (beam_diameter_mm / reference_diameter_mm) ** 2


def build_multilevel_loading_configuration(
    sweep_kind: str,
    parameter_value: float,
) -> tuple[MultilevelMOTConfig, MOTApparatusConfig, list[MOTBeam]]:
    """Build a mutually consistent multilevel config, apparatus, and beam set.

    For the saturation study only cooling power changes.  For the diameter
    study only cooling diameter and cooling power change; every plot-point
    simulation rebuilds the repump beams at their baseline power and diameter.
    Thus no optical state carries over from a preceding plot point.
    """

    if parameter_value <= 0.0:
        raise ValueError("parameter_value must be positive")
    config = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        saturation_intensity_w_per_m2=USER_SATURATION_INTENSITY_W_PER_M2,
    )
    apparatus = default_mot_apparatus_config()
    if sweep_kind == "saturation":
        power_w = saturation_power_w_per_beam(
            parameter_value,
            saturation_intensity_w_per_m2=config.saturation_intensity_w_per_m2,
            beam_diameter_m=apparatus.cooling.beam_diameter_m,
        )
        apparatus = replace(
            apparatus,
            cooling=replace(apparatus.cooling, power_w_per_beam=power_w),
        )
    elif sweep_kind == "beam_size":
        diameter_m = 1.0e-3 * parameter_value
        scale_factor = (parameter_value / (1.0e3 * apparatus.cooling.beam_diameter_m)) ** 2
        apparatus = replace(
            apparatus,
            cooling=replace(
                apparatus.cooling,
                beam_diameter_m=diameter_m,
                power_w_per_beam=apparatus.cooling.power_w_per_beam * scale_factor,
            ),
        )
    else:
        raise ValueError("sweep_kind must be 'saturation' or 'beam_size'")
    if sweep_kind == "beam_size":
        # The varied apparatus defines cooling beams only.  Repump components
        # are rebuilt from an untouched default apparatus and the baseline
        # repump configuration, preserving both their power and diameter.
        beams = build_multilevel_cooling_beams(apparatus) + build_multilevel_repump_beams(
            default_mot_apparatus_config(),
            config,
        )
    else:
        beams = build_multilevel_mot_beams(apparatus_config=apparatus, config=config)
    return config, apparatus, beams


def _validate_search(search: RateCaptureSearchConfig) -> None:
    if search.disc_count <= 0 or search.points_per_disc <= 0:
        raise ValueError("disc_count and points_per_disc must be positive")
    if search.time_step_s <= 0.0 or search.max_simulation_time_s <= 0.0:
        raise ValueError("trajectory time and timestep must be positive")
    if search.velocity_tolerance_m_per_s <= 0.0:
        raise ValueError("velocity tolerance must be positive")
    if search.max_bracket_iterations <= 0 or search.max_search_iterations <= 0:
        raise ValueError("capture bracket and binary-search iteration limits must be positive")
    if search.required_core_entries != 2:
        raise ValueError("August 22 capture requires a two-entry trapping route")
    if not np.isclose(
        search.bounded_core_residence_s,
        5.0e-3,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("August 22 capture requires 5 ms continuous core residence")
    if search.trap_core_radius_m <= 0.0 or search.escape_radius_m <= 0.0:
        raise ValueError("trap-core and escape radii must be positive")
    if search.analysis_velocity_step_m_per_s <= 0.0:
        raise ValueError("analysis velocity step must be positive")
    if search.analysis_velocity_min_m_per_s < 0.0:
        raise ValueError("analysis velocity minimum must be non-negative")
    if search.analysis_velocity_max_m_per_s <= search.analysis_velocity_min_m_per_s:
        raise ValueError("analysis velocity maximum must exceed its minimum")
    if search.phase_space != "octant":
        raise ValueError("August 22 loading sweeps require one-octant solid-angle sampling")
    if not np.isclose(
        search.disc_radius_m,
        DEFAULT_CAPTURE_DISC_RADIUS_M,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("multilevel loading sweeps require a fixed 12 mm sampling-disc radius")


def _validate_plot_point_replicate_design(search: RateCaptureSearchConfig) -> None:
    """Enforce ten random launch discs with 25 random points per disc."""

    _validate_search(search)
    if search.disc_count != PLOT_POINT_DISC_COUNT:
        raise ValueError(
            "each loading plot point requires exactly 10 launch-geometry discs"
        )
    if search.points_per_disc != POINTS_PER_DISC:
        raise ValueError(
            "each launch-geometry disc requires exactly 25 sampled impact points"
        )


def _sample_independent_uniform_area_points(
    disc: DiscSample,
    points_per_disc: int,
    disc_radius_m: float,
    rng: np.random.Generator,
) -> list[PointSample]:
    """Sample independent random points uniformly in disc area.

    Every point uses an independent uniform area fraction and azimuth.  Thus
    ``s = R sqrt(U)`` has the required ``s ds dtheta`` measure.  No center or
    boundary point is inserted deterministically.
    """

    if points_per_disc <= 0 or disc_radius_m <= 0.0:
        raise ValueError("points_per_disc and disc_radius_m must be positive")
    center = np.asarray(disc.center_position_m, dtype=float)
    basis_u = np.asarray(disc.basis_u, dtype=float)
    basis_v = np.asarray(disc.basis_v, dtype=float)
    incident = np.asarray(disc.incident_unit_vector, dtype=float)
    points: list[PointSample] = []
    for point_index in range(points_per_disc):
        area_fraction = float(rng.random())
        theta_prime = float(rng.uniform(0.0, 2.0 * pi))
        s_m = disc_radius_m * float(np.sqrt(area_fraction))
        offset = s_m * (
            np.cos(theta_prime) * basis_u + np.sin(theta_prime) * basis_v
        )
        initial_position = center + offset
        points.append(
            PointSample(
                disc_index=disc.disc_index,
                point_index=point_index,
                theta_rad=disc.theta_rad,
                phi_rad=disc.phi_rad,
                theta_prime_rad=theta_prime,
                s_m=s_m,
                radial_distance_m=float(np.linalg.norm(center)),
                initial_position_m=tuple(float(value) for value in initial_position),
                incident_unit_vector=tuple(float(value) for value in incident),
                launch_axis_unit_vector=tuple(float(value) for value in incident),
            )
        )
    return points


def generate_common_capture_points(search: RateCaptureSearchConfig) -> list[PointSample]:
    """Return reproducible octant discs with independent uniform-area points."""

    _validate_search(search)
    rng = np.random.default_rng(search.seed)
    points: list[PointSample] = []
    for disc_index in range(search.disc_count):
        disc = sample_incident_disc(disc_index, search.radial_distance_m, rng)
        points.extend(
            _sample_independent_uniform_area_points(
                disc,
                search.points_per_disc,
                search.disc_radius_m,
                rng,
            )
        )
    return points


def _geometry_sha256(points: Sequence[PointSample]) -> str:
    digest = hashlib.sha256()
    for point in points:
        values = (
            point.disc_index,
            point.point_index,
            point.theta_rad,
            point.phi_rad,
            point.theta_prime_rad,
            point.s_m,
            *point.initial_position_m,
            *point.incident_unit_vector,
        )
        digest.update(
            (",".join(format(float(value), ".17g") for value in values) + "\n").encode("ascii")
        )
    return digest.hexdigest()


def _validate_multilevel_inputs(
    model: RateEquationModel,
    beams: Sequence[MOTBeam],
    config: MultilevelMOTConfig,
) -> None:
    if model.ground_count != 8 or model.excited_count != 16 or model.state_count != 24:
        raise RuntimeError(
            "loading sweeps require the current multilevel repumper graph: "
            "8 ground plus 16 excited indexed states"
        )
    if not config.repumper_enabled:
        raise RuntimeError("loading sweeps require the multilevel repumper")
    families = {beam.family for beam in beams}
    if families != {"cooling", "repump"}:
        raise RuntimeError("multilevel loading requires cooling and repump beam components")


def _validate_trajectory_endpoint(
    result: TrajectoryClassification,
    point: PointSample,
    speed_m_per_s: float,
) -> None:
    """Reject numerically or semantically invalid capture-search evaluations."""

    reason = result.termination_reason
    location = (
        f"disc={point.disc_index}, point={point.point_index}, "
        f"speed={speed_m_per_s:.12g} m/s"
    )
    if reason == "non_finite":
        raise RuntimeError(
            f"fatal non_finite multilevel trajectory during capture search ({location})"
        )
    if result.trapped:
        if reason not in _TRAPPED_TERMINATION_REASONS:
            raise RuntimeError(
                f"invalid trapped termination reason {reason!r} during capture search "
                f"({location})"
            )
    elif reason not in _VALID_UNTRAPPED_TERMINATION_REASONS:
        raise RuntimeError(
            f"invalid untrapped termination reason {reason!r} during capture search "
            f"({location})"
        )


def _validate_capture_sample_endpoints(sample: CaptureVelocitySample) -> None:
    """Validate new or checkpointed sample endpoint data before aggregation."""

    endpoint_values = (
        sample.capture_velocity_m_per_s,
        sample.velocity_resolution_m_per_s,
        sample.trapped_velocity_lower_m_per_s,
        sample.untrapped_velocity_upper_m_per_s,
    )
    if not np.all(np.isfinite(endpoint_values)):
        raise RuntimeError(
            "capture sample contains non-finite endpoint speeds "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )
    if min(endpoint_values) < 0.0:
        raise RuntimeError(
            "capture sample contains a negative endpoint speed "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )
    if sample.lower_classification == "non_finite" or sample.upper_classification == "non_finite":
        raise RuntimeError(
            "capture sample contains a fatal non_finite endpoint "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )
    valid_lower_reasons = (
        _TRAPPED_TERMINATION_REASONS | _VALID_UNTRAPPED_TERMINATION_REASONS
    )
    if sample.lower_classification not in valid_lower_reasons:
        raise RuntimeError(
            f"capture sample has invalid lower termination reason "
            f"{sample.lower_classification!r} "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )
    if sample.upper_classification not in _VALID_UNTRAPPED_TERMINATION_REASONS:
        raise RuntimeError(
            f"capture sample has invalid upper termination reason "
            f"{sample.upper_classification!r} "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )
    if sample.untrapped_velocity_upper_m_per_s < sample.trapped_velocity_lower_m_per_s:
        raise RuntimeError(
            "capture sample has reversed speed endpoints "
            f"(disc={sample.disc_index}, point={sample.point_index})"
        )


def classify_multilevel_loading_trajectory(
    point: PointSample,
    incident_speed_m_per_s: float,
    search: RateCaptureSearchConfig,
    *,
    model: RateEquationModel,
    beams: list[MOTBeam],
    coil_config,
    config: MultilevelMOTConfig,
) -> TrajectoryClassification:
    """Classify one deterministic 24-index multilevel rate-equation trajectory."""

    _validate_multilevel_inputs(model, beams, config)
    position = np.asarray(point.initial_position_m, dtype=float)
    velocity = incident_speed_m_per_s * np.asarray(point.incident_unit_vector, dtype=float)
    previous_axis = (0.0, 0.0, 1.0)
    minimum_radius = float(np.linalg.norm(position))
    was_inside = minimum_radius <= search.trap_core_radius_m
    entered_core = was_inside
    core_entries = int(was_inside)
    inside_since_s: float | None = 0.0 if was_inside else None
    elapsed_s = 0.0
    max_steps = int(np.ceil(search.max_simulation_time_s / search.time_step_s))

    for _ in range(max_steps + 1):
        radius_m = float(np.linalg.norm(position))
        minimum_radius = min(minimum_radius, radius_m)
        inside = radius_m <= search.trap_core_radius_m
        entered_core = entered_core or inside
        if inside and not was_inside:
            core_entries += 1
            inside_since_s = elapsed_s
        elif not inside:
            inside_since_s = None
        was_inside = inside
        radial_velocity = float(np.dot(position, velocity)) / max(radius_m, 1.0e-15)

        if core_entries >= search.required_core_entries:
            return TrajectoryClassification(
                True,
                "two_core_entries",
                entered_core,
                core_entries,
                elapsed_s,
                minimum_radius,
                radius_m,
                tuple(position),
                tuple(velocity),
            )
        if (
            inside_since_s is not None
            and elapsed_s - inside_since_s >= search.bounded_core_residence_s
        ):
            return TrajectoryClassification(
                True,
                "bounded_core_residence",
                entered_core,
                core_entries,
                elapsed_s,
                minimum_radius,
                radius_m,
                tuple(position),
                tuple(velocity),
            )
        if radius_m >= search.escape_radius_m and radial_velocity > 0.0:
            return TrajectoryClassification(
                False,
                "escaped",
                entered_core,
                core_entries,
                elapsed_s,
                minimum_radius,
                radius_m,
                tuple(position),
                tuple(velocity),
            )
        if elapsed_s >= search.max_simulation_time_s - 1.0e-15:
            break

        observable = rate_equation_observable(
            model,
            beams,
            tuple(position),
            tuple(velocity),
            coil_config,
            config,
            previous_axis=previous_axis,
        )
        previous_axis = observable.quantization_axis
        dt_s = min(search.time_step_s, search.max_simulation_time_s - elapsed_s)
        acceleration = np.asarray(observable.force_n, dtype=float) / RB87_MASS_KG
        if config.include_gravity:
            acceleration += np.asarray(GRAVITY_ACCELERATION_M_PER_S2)
        # Match the production multilevel rate-capture integrator: velocity
        # update followed by the updated-velocity position drift.
        velocity = velocity + acceleration * dt_s
        position = position + velocity * dt_s
        elapsed_s += dt_s
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            return TrajectoryClassification(
                False,
                "non_finite",
                entered_core,
                core_entries,
                elapsed_s,
                minimum_radius,
                float(np.linalg.norm(position)),
                tuple(position),
                tuple(velocity),
            )

    return TrajectoryClassification(
        False,
        "timeout",
        entered_core,
        core_entries,
        elapsed_s,
        minimum_radius,
        float(np.linalg.norm(position)),
        tuple(position),
        tuple(velocity),
    )


def _capture_sample(
    point: PointSample,
    lower: float,
    upper: float,
    lower_result: TrajectoryClassification,
    upper_result: TrajectoryClassification,
) -> CaptureVelocitySample:
    return CaptureVelocitySample(
        disc_index=point.disc_index,
        point_index=point.point_index,
        theta_rad=point.theta_rad,
        phi_rad=point.phi_rad,
        theta_prime_rad=point.theta_prime_rad,
        s_m=point.s_m,
        radial_distance_m=point.radial_distance_m,
        initial_position_m=point.initial_position_m,
        incident_unit_vector=point.incident_unit_vector,
        capture_velocity_m_per_s=lower,
        velocity_resolution_m_per_s=upper - lower,
        trapped_velocity_lower_m_per_s=lower,
        untrapped_velocity_upper_m_per_s=upper,
        lower_classification=lower_result.termination_reason,
        upper_classification=upper_result.termination_reason,
        lower_entered_trap_core=lower_result.entered_trap_core,
        upper_entered_trap_core=upper_result.entered_trap_core,
        lower_core_entry_count=lower_result.core_entry_count,
        upper_core_entry_count=upper_result.core_entry_count,
    )


def find_multilevel_capture_velocity(
    point: PointSample,
    search: RateCaptureSearchConfig,
    *,
    model: RateEquationModel,
    beams: list[MOTBeam],
    coil_config,
    config: MultilevelMOTConfig,
) -> CaptureVelocitySample:
    """Find an explicitly trapped lower and untrapped upper speed bound."""

    evaluations: dict[float, TrajectoryClassification] = {}

    def evaluate(speed_m_per_s: float) -> TrajectoryClassification:
        key = round(float(speed_m_per_s), 12)
        if key not in evaluations:
            result = classify_multilevel_loading_trajectory(
                point,
                key,
                search,
                model=model,
                beams=beams,
                coil_config=coil_config,
                config=config,
            )
            _validate_trajectory_endpoint(result, point, key)
            evaluations[key] = result
        return evaluations[key]

    trial = max(0.0, search.initial_velocity_guess_m_per_s)
    if evaluate(trial).trapped:
        lower, upper = trial, max(1.0, trial)
        for _ in range(search.max_bracket_iterations):
            upper *= 2.0
            if not evaluate(upper).trapped:
                break
        else:
            raise RuntimeError("failed to find an untrapped upper multilevel capture bracket")
    else:
        upper = trial
        lower: float | None = upper
        for iteration in range(search.max_bracket_iterations):
            lower = 0.0 if iteration == search.max_bracket_iterations - 1 else 0.5 * lower
            if evaluate(lower).trapped:
                break
            if lower <= 1.0e-6:
                lower = 0.0 if evaluate(0.0).trapped else None
                break
        if lower is not None and not evaluate(lower).trapped:
            lower = 0.0 if evaluate(0.0).trapped else None
        if lower is None:
            return _capture_sample(
                point,
                0.0,
                upper,
                evaluations[0.0],
                evaluations[round(upper, 12)],
            )

    assert lower is not None
    for _ in range(search.max_search_iterations):
        if upper - lower <= search.velocity_tolerance_m_per_s:
            break
        midpoint = round(0.5 * (lower + upper), 12)
        if evaluate(midpoint).trapped:
            lower = midpoint
        else:
            upper = midpoint
    return _capture_sample(
        point,
        lower,
        upper,
        evaluations[round(lower, 12)],
        evaluations[round(upper, 12)],
    )


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Replace a checkpoint despite short-lived Windows sync-client locks."""

    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_write_json(path: Path, payload: Mapping | Sequence) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
    _replace_with_retry(temporary, path)
    return path


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping],
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_ready(row.get(field, "")) for field in fieldnames})
    _replace_with_retry(temporary, path)
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sample_row(sample: CaptureVelocitySample) -> dict[str, object]:
    return {
        "disc_index": sample.disc_index,
        "point_index": sample.point_index,
        "theta_rad": sample.theta_rad,
        "phi_rad": sample.phi_rad,
        "theta_prime_rad": sample.theta_prime_rad,
        "s_m": sample.s_m,
        "radial_distance_m": sample.radial_distance_m,
        "capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "velocity_resolution_m_per_s": sample.velocity_resolution_m_per_s,
        "trapped_velocity_lower_m_per_s": sample.trapped_velocity_lower_m_per_s,
        "untrapped_velocity_upper_m_per_s": sample.untrapped_velocity_upper_m_per_s,
        "lower_classification": sample.lower_classification,
        "upper_classification": sample.upper_classification,
        "lower_entered_trap_core": sample.lower_entered_trap_core,
        "upper_entered_trap_core": sample.upper_entered_trap_core,
        "lower_core_entry_count": sample.lower_core_entry_count,
        "upper_core_entry_count": sample.upper_core_entry_count,
        "x0_m": sample.initial_position_m[0],
        "y0_m": sample.initial_position_m[1],
        "z0_m": sample.initial_position_m[2],
        "vx_hat": sample.incident_unit_vector[0],
        "vy_hat": sample.incident_unit_vector[1],
        "vz_hat": sample.incident_unit_vector[2],
    }


def _write_capture_samples(path: Path, samples: Sequence[CaptureVelocitySample]) -> Path:
    ordered = sorted(samples, key=lambda sample: (sample.disc_index, sample.point_index))
    return _atomic_write_csv(path, [_sample_row(sample) for sample in ordered], _CAPTURE_FIELDS)


def _write_spectrum(path: Path, spectrum) -> Path:
    rows = [
        {
            "velocity_m_per_s": sample.velocity_m_per_s,
            "captured_count": sample.captured_count,
            "launched_count": sample.launched_count,
            "capture_fraction": sample.capture_fraction,
            "capture_cross_section_m2": sample.capture_cross_section_m2,
        }
        for sample in spectrum
    ]
    return _atomic_write_csv(path, rows, _SPECTRUM_FIELDS)


def _build_replicate_loading_rows(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Return ten disc-level loading rates and clustered statistics."""

    expected_sample_count = PLOT_POINT_DISC_COUNT * POINTS_PER_DISC
    if len(samples) != expected_sample_count:
        raise RuntimeError(
            "disc-level loading analysis requires exactly 10 discs x 25 points"
        )
    ordered = sorted(samples, key=lambda sample: (sample.disc_index, sample.point_index))
    disc_ids = sorted({sample.disc_index for sample in ordered})
    if len(disc_ids) != PLOT_POINT_DISC_COUNT:
        raise RuntimeError("disc-level analysis requires 10 sampled incident directions")

    rows: list[dict[str, object]] = []
    rates: list[float] = []
    for replicate_index, disc_index in enumerate(disc_ids, start=1):
        disc_samples = [sample for sample in ordered if sample.disc_index == disc_index]
        if len(disc_samples) != POINTS_PER_DISC:
            raise RuntimeError(
                f"disc {disc_index} does not contain exactly {POINTS_PER_DISC} points"
            )
        if {sample.point_index for sample in disc_samples} != set(range(POINTS_PER_DISC)):
            raise RuntimeError(f"disc {disc_index} has missing or duplicate point indices")
        theta_values = {sample.theta_rad for sample in disc_samples}
        phi_values = {sample.phi_rad for sample in disc_samples}
        if len(theta_values) != 1 or len(phi_values) != 1:
            raise RuntimeError(f"disc {disc_index} does not share one incident direction")
        spectrum = build_dynamic_capture_spectrum(disc_samples, search)
        loading = calculate_loading_rate_from_spectrum(
            np.asarray([item.velocity_m_per_s for item in spectrum], dtype=float),
            np.asarray([item.capture_cross_section_m2 for item in spectrum], dtype=float),
        )
        rate = float(loading.loading_rate_atoms_per_s)
        rates.append(rate)
        capture_velocity = np.asarray(
            [sample.capture_velocity_m_per_s for sample in disc_samples],
            dtype=float,
        )
        rows.append(
            {
                "replicate_index": replicate_index,
                "disc_index": disc_index,
                "theta_rad": disc_samples[0].theta_rad,
                "phi_rad": disc_samples[0].phi_rad,
                "point_count": len(disc_samples),
                "capture_velocity_mean_m_per_s": float(np.mean(capture_velocity)),
                "capture_velocity_sample_std_m_per_s": float(
                    np.std(capture_velocity, ddof=1)
                ),
                "loading_rate_atoms_per_s": rate,
            }
        )
    values = np.asarray(rates, dtype=float)
    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    standard_error = sample_std / np.sqrt(PLOT_POINT_DISC_COUNT)
    return rows, {
        "mean_atoms_per_s": mean,
        "sample_std_atoms_per_s": sample_std,
        "standard_error_atoms_per_s": standard_error,
        "nominal_student_t_95_percent_half_width_atoms_per_s": (
            _STUDENT_T_95_DF9 * standard_error
        ),
    }


def build_dynamic_capture_spectrum(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
) -> list[VelocitySpectrumSample]:
    """Build a zero-based spectrum with a data-dependent upper bound.

    The last velocity is always at least one complete analysis step above the
    largest captured threshold and never below the configured analysis maximum.
    Consequently the saved spectrum contains an explicit zero-cross-section
    tail sample even when capture velocities exceed the former 30 m/s default.
    """

    if not samples:
        raise ValueError("at least one capture sample is required")
    step = float(search.analysis_velocity_step_m_per_s)
    if step <= 0.0:
        raise ValueError("analysis velocity step must be positive")
    maximum_capture = max(float(sample.capture_velocity_m_per_s) for sample in samples)
    capture_ceiling_index = int(np.ceil(maximum_capture / step)) + 1
    configured_ceiling_index = int(np.ceil(search.analysis_velocity_max_m_per_s / step))
    ceiling_index = max(capture_ceiling_index, configured_ceiling_index)
    velocity_grid = step * np.arange(ceiling_index + 1, dtype=float)
    capture_velocity = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples],
        dtype=float,
    )
    launched_count = len(samples)
    disc_area_m2 = pi * search.disc_radius_m**2
    spectrum: list[VelocitySpectrumSample] = []
    for velocity_m_per_s in velocity_grid:
        captured_count = int(
            np.count_nonzero(capture_velocity >= velocity_m_per_s - 1.0e-12)
        )
        capture_fraction = captured_count / launched_count
        spectrum.append(
            VelocitySpectrumSample(
                velocity_m_per_s=float(velocity_m_per_s),
                captured_count=captured_count,
                launched_count=launched_count,
                capture_fraction=capture_fraction,
                capture_cross_section_m2=disc_area_m2 * capture_fraction,
            )
        )
    return spectrum


def _parameter_key(index: int, label: str, value: float) -> str:
    value_text = format(value, ".10g").replace("-", "minus_").replace(".", "p")
    return f"{index:03d}_{label}_{value_text}"


def _parameter_directory(output_directory: Path, parameter_key: str) -> Path:
    return output_directory / "parameters" / parameter_key


def _effective_signature_payload(
    sweep_kind: str,
    values: Sequence[float],
    search: RateCaptureSearchConfig,
) -> dict[str, object]:
    base_config = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        saturation_intensity_w_per_m2=USER_SATURATION_INTENSITY_W_PER_M2,
    )
    base_apparatus = default_mot_apparatus_config()
    coil = default_anti_helmholtz_config()
    model = build_rate_equation_model(base_config.natural_linewidth_rad_per_s)
    payload: dict[str, object] = {
        "format_version": _FORMAT_VERSION,
        "sweep_kind": sweep_kind,
        "values": list(values),
        "search_config": asdict(search),
        "base_multilevel_config": asdict(base_config),
        "base_apparatus_config": asdict(base_apparatus),
        "coil_config": asdict(coil),
        "indexed_state_count": model.state_count,
        "ground_state_count": model.ground_count,
        "excited_state_count": model.excited_count,
        "study_execution_mode": "one separately presented parameter study",
        "plot_point_replicate_count": PLOT_POINT_DISC_COUNT,
        "launch_disc_count": PLOT_POINT_DISC_COUNT,
        "points_per_disc": POINTS_PER_DISC,
        "capture_threshold_simulation_count": (
            PLOT_POINT_DISC_COUNT * POINTS_PER_DISC
        ),
        "replicate_definition": (
            "one independently sampled incident direction and its 25 independent "
            "uniform-area impact-point capture-threshold realizations"
        ),
        "points_per_replicate": POINTS_PER_DISC,
        "plot_value_estimator": (
            "loading rate from all 250 capture thresholds, algebraically equal to "
            "the arithmetic mean of the 10 disc-level loading rates"
        ),
        "plot_uncertainty": (
            "direction-clustered sample standard error across 10 disc-level loading "
            "rates (25 points per disc); nominal "
            "Student-t 95 percent half-width with 9 degrees of freedom also saved"
        ),
        "capture_integrator": "semi-implicit Euler (velocity update, then position)",
        "capture_dynamics": _CAPTURE_DYNAMICS,
        "capture_endpoint_policy": (
            "trapped lower endpoint requires either 5 ms continuous residence in the "
            "2 mm core or two core entries with an intervening exit; escaped or "
            "configured-timeout upper endpoint; non_finite and all other termination "
            "reasons are fatal"
        ),
        "geometry_sampler": (
            "10 directions uniform in solid angle in one symmetry octant; per direction, "
            "25 independent uniform-area disc points with independent uniform azimuth; "
            "no forced center or boundary points; all launch velocities on a disc are "
            "parallel to its inward normal"
        ),
        "user_saturation_intensity_w_per_m2": USER_SATURATION_INTENSITY_W_PER_M2,
        "saturation_power_equation": "P = pi d^2 n I_sat / 8",
        "beam_size_power_equation": "P(d) = P(12.7 mm) (d/12.7 mm)^2",
    }
    if sweep_kind == "beam_size":
        # This sweep-specific rule is part of the scientific signature.  It
        # intentionally invalidates preliminary beam checkpoints that scaled
        # the repump along with the cooling beams, without invalidating valid
        # saturation-study checkpoints.
        payload["beam_size_power_equation"] = (
            "P_cooling(d) = P_cooling(12.7 mm) (d/12.7 mm)^2"
        )
        payload["beam_size_configuration_policy_version"] = 2
        payload["beam_size_varied_components"] = [
            "cooling beam diameter",
            "cooling power per beam",
        ]
        payload["beam_size_fixed_components"] = [
            "repump beam diameter at default baseline",
            "repump power per beam at default baseline",
        ]
        payload["beam_size_repump_rule"] = (
            "repump power and diameter remain at independent default baseline values"
        )
    return payload


def _run_signature(
    sweep_kind: str,
    values: Sequence[float],
    search: RateCaptureSearchConfig,
) -> tuple[str, dict[str, object]]:
    payload = _effective_signature_payload(sweep_kind, values, search)
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), payload


@dataclass(frozen=True, slots=True)
class _ParameterWorkerSpec:
    sweep_kind: str
    parameter_index: int
    parameter_count: int
    parameter_value: float
    parameter_key: str
    run_signature: str
    search: RateCaptureSearchConfig
    output_directory: Path
    resume: bool


@dataclass(frozen=True, slots=True)
class _PreparedLoadingSweep:
    sweep_kind: str
    values: tuple[float, ...]
    output_directory: Path
    figure_directory: Path
    search: RateCaptureSearchConfig
    run_signature: str
    specs: tuple[_ParameterWorkerSpec, ...]
    existing_rows: Mapping[str, Mapping]
    aggregate_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SampleShardSpec:
    """A deterministic subset of one parameter's missing launch points."""

    parameter: _ParameterWorkerSpec
    shard_index: int
    shard_count: int
    points: tuple[PointSample, ...]


@dataclass(frozen=True, slots=True)
class _SampleShardResult:
    """Capture results returned to the sole parent-process checkpoint writer."""

    sweep_kind: str
    parameter_key: str
    run_signature: str
    shard_index: int
    samples: tuple[CaptureVelocitySample, ...]


@dataclass(slots=True)
class _ParameterShardState:
    prepared: _PreparedLoadingSweep
    spec: _ParameterWorkerSpec
    points: tuple[PointSample, ...]
    results: dict[tuple[int, int], CaptureVelocitySample]
    initial_result_count: int
    started_monotonic_s: float


def _valid_checkpoint(
    parameter_directory: Path,
    run_signature: str,
    parameter_key: str,
) -> bool:
    marker = parameter_directory / "checkpoint.json"
    partial = parameter_directory / "capture_velocity_partial_samples.csv"
    if not marker.is_file() or not partial.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("run_signature") == run_signature
        and payload.get("parameter_key") == parameter_key
    )


def _save_checkpoint(
    parameter_directory: Path,
    samples: Sequence[CaptureVelocitySample],
    spec: _ParameterWorkerSpec,
) -> None:
    _write_capture_samples(
        parameter_directory / "capture_velocity_partial_samples.csv",
        samples,
    )
    _atomic_write_json(
        parameter_directory / "checkpoint.json",
        {
            "run_signature": spec.run_signature,
            "parameter_key": spec.parameter_key,
            "completed_sample_count": len(samples),
            "expected_sample_count": spec.search.disc_count * spec.search.points_per_disc,
        },
    )


def _beam_peak_intensity_w_per_m2(beam: MOTBeam) -> float:
    return 2.0 * beam.power_w / (pi * beam.beam_radius_m**2)


def _checkpointed_parameter_results(
    spec: _ParameterWorkerSpec,
    points: Sequence[PointSample],
) -> dict[tuple[int, int], CaptureVelocitySample]:
    """Load a compatible signed partial checkpoint and discard foreign point keys."""

    parameter_directory = _parameter_directory(spec.output_directory, spec.parameter_key)
    existing: list[CaptureVelocitySample] = []
    if spec.resume and _valid_checkpoint(
        parameter_directory,
        spec.run_signature,
        spec.parameter_key,
    ):
        try:
            existing = load_capture_velocity_samples(
                parameter_directory / "capture_velocity_partial_samples.csv"
            )
        except (OSError, KeyError, TypeError, ValueError):
            existing = []
    valid_point_keys = {(point.disc_index, point.point_index) for point in points}
    results = {
        (sample.disc_index, sample.point_index): sample
        for sample in existing
        if (sample.disc_index, sample.point_index) in valid_point_keys
    }
    for sample in results.values():
        _validate_capture_sample_endpoints(sample)
    return results


def _run_sample_shard_worker(shard: _SampleShardSpec) -> _SampleShardResult:
    """Evaluate one independent capture shard without writing shared files."""

    spec = shard.parameter
    config, _apparatus, beams = build_multilevel_loading_configuration(
        spec.sweep_kind,
        spec.parameter_value,
    )
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    coil = default_anti_helmholtz_config()
    _validate_multilevel_inputs(model, beams, config)
    samples: list[CaptureVelocitySample] = []
    for point in shard.points:
        sample = find_multilevel_capture_velocity(
            point,
            spec.search,
            model=model,
            beams=beams,
            coil_config=coil,
            config=config,
        )
        _validate_capture_sample_endpoints(sample)
        samples.append(sample)
    return _SampleShardResult(
        sweep_kind=spec.sweep_kind,
        parameter_key=spec.parameter_key,
        run_signature=spec.run_signature,
        shard_index=shard.shard_index,
        samples=tuple(samples),
    )


def _validate_sample_shard_result(
    shard: _SampleShardSpec,
    result: _SampleShardResult,
) -> None:
    """Reject a malformed, duplicated, or misrouted worker result."""

    spec = shard.parameter
    if (
        result.sweep_kind != spec.sweep_kind
        or result.parameter_key != spec.parameter_key
        or result.run_signature != spec.run_signature
        or result.shard_index != shard.shard_index
    ):
        raise RuntimeError(
            f"misrouted sample shard result for {spec.sweep_kind}/{spec.parameter_key}"
        )
    expected_keys = {(point.disc_index, point.point_index) for point in shard.points}
    observed_keys = {
        (sample.disc_index, sample.point_index) for sample in result.samples
    }
    if len(result.samples) != len(expected_keys) or observed_keys != expected_keys:
        raise RuntimeError(
            f"incomplete sample shard result for {spec.sweep_kind}/{spec.parameter_key}"
        )
    for sample in result.samples:
        _validate_capture_sample_endpoints(sample)


def _run_parameter_worker(spec: _ParameterWorkerSpec) -> dict[str, object]:
    started = time.monotonic()
    config, apparatus, beams = build_multilevel_loading_configuration(
        spec.sweep_kind,
        spec.parameter_value,
    )
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    coil = default_anti_helmholtz_config()
    _validate_multilevel_inputs(model, beams, config)
    points = generate_common_capture_points(spec.search)
    geometry_sha256 = _geometry_sha256(points)
    parameter_directory = _parameter_directory(spec.output_directory, spec.parameter_key)
    parameter_directory.mkdir(parents=True, exist_ok=True)

    results = _checkpointed_parameter_results(spec, points)
    missing = [
        point
        for point in points
        if (point.disc_index, point.point_index) not in results
    ]
    label = "n" if spec.sweep_kind == "saturation" else "d_mm"
    print(
        f"[multilevel {spec.sweep_kind} study] plot point "
        f"{spec.parameter_index + 1}/{spec.parameter_count} "
        f"({label}={spec.parameter_value:g}); resumed "
        f"{len(results)}/{PLOT_POINT_DISC_COUNT * POINTS_PER_DISC} capture-threshold "
        "simulations (10 discs x 25 points)",
        flush=True,
    )

    newly_completed = 0
    for point in missing:
        result = find_multilevel_capture_velocity(
            point,
            spec.search,
            model=model,
            beams=beams,
            coil_config=coil,
            config=config,
        )
        _validate_capture_sample_endpoints(result)
        results[(result.disc_index, result.point_index)] = result
        newly_completed += 1
        elapsed_s = time.monotonic() - started
        throughput = newly_completed / elapsed_s if elapsed_s > 0.0 else 0.0
        eta_s = (len(missing) - newly_completed) / throughput if throughput > 0.0 else float("inf")
        print(
            f"[multilevel {spec.sweep_kind} study] plot point "
            f"{spec.parameter_index + 1}/{spec.parameter_count} "
            f"({label}={spec.parameter_value:g}); disc {point.disc_index + 1}/"
            f"{PLOT_POINT_DISC_COUNT}, point {point.point_index + 1}/{POINTS_PER_DISC}; "
            f"samples {len(results)}/{PLOT_POINT_DISC_COUNT * POINTS_PER_DISC} complete; "
            f"vc={result.capture_velocity_m_per_s:.3f} m/s; "
            f"ETA={eta_s / 60.0:.1f} min",
            flush=True,
        )
        if newly_completed % max(1, spec.search.save_every) == 0 or newly_completed == len(missing):
            _save_checkpoint(parameter_directory, list(results.values()), spec)

    samples = sorted(results.values(), key=lambda sample: (sample.disc_index, sample.point_index))
    if len(samples) != len(points):
        raise RuntimeError(f"parameter sampling incomplete: {len(samples)} of {len(points)}")
    capture_velocity = np.asarray(
        [sample.capture_velocity_m_per_s for sample in samples],
        dtype=float,
    )
    valid_bracket_count = sum(
        sample.lower_classification in _TRAPPED_TERMINATION_REASONS
        and sample.upper_classification in _VALID_UNTRAPPED_TERMINATION_REASONS
        for sample in samples
    )
    zero_capture_no_bracket_count = sum(
        sample.capture_velocity_m_per_s == 0.0
        and sample.lower_classification in _VALID_UNTRAPPED_TERMINATION_REASONS
        and sample.upper_classification in _VALID_UNTRAPPED_TERMINATION_REASONS
        for sample in samples
    )
    upper_escaped_count = sum(
        sample.upper_classification == "escaped" for sample in samples
    )
    upper_timeout_count = sum(
        sample.upper_classification == "timeout" for sample in samples
    )
    upper_other_reasons = sorted(
        {
            sample.upper_classification
            for sample in samples
            if sample.upper_classification not in _VALID_UNTRAPPED_TERMINATION_REASONS
        }
    )
    upper_other_count = len(samples) - upper_escaped_count - upper_timeout_count
    _write_capture_samples(parameter_directory / "capture_velocity_samples.csv", samples)
    spectrum = build_dynamic_capture_spectrum(samples, spec.search)
    _write_spectrum(parameter_directory / "capture_velocity_spectrum.csv", spectrum)
    spectrum_velocity = np.asarray([sample.velocity_m_per_s for sample in spectrum], dtype=float)
    spectrum_cross_section = np.asarray(
        [sample.capture_cross_section_m2 for sample in spectrum],
        dtype=float,
    )
    loading = calculate_loading_rate_from_spectrum(spectrum_velocity, spectrum_cross_section)
    uncertainty = calculate_sampling_uncertainty(
        np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float),
        np.asarray([sample.disc_index for sample in samples], dtype=int),
        spectrum_velocity,
        spec.search.disc_radius_m,
    )
    replicate_rows, replicate_statistics = _build_replicate_loading_rows(
        samples,
        spec.search,
    )
    if not np.isclose(
        loading.loading_rate_atoms_per_s,
        replicate_statistics["mean_atoms_per_s"],
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "aggregate loading rate is not the mean of the 10 disc-level rates"
        )
    if not np.isclose(
        uncertainty.disc_cluster_standard_error_atoms_per_s,
        replicate_statistics["standard_error_atoms_per_s"],
        rtol=1.0e-12,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "disc-cluster uncertainty is not the 10-disc standard error"
        )
    _atomic_write_csv(
        parameter_directory / "replicate_loading_rates.csv",
        replicate_rows,
        _REPLICATE_LOADING_FIELDS,
    )
    _atomic_write_json(
        parameter_directory / "loading_rate.json",
        {
            "loading_rate_atoms_per_s": loading.loading_rate_atoms_per_s,
            "simulation_replicate_count": PLOT_POINT_DISC_COUNT,
            "launch_disc_count": PLOT_POINT_DISC_COUNT,
            "points_per_disc": POINTS_PER_DISC,
            "capture_threshold_simulation_count": (
                PLOT_POINT_DISC_COUNT * POINTS_PER_DISC
            ),
            "replicate_loading_rate_statistics": replicate_statistics,
            # The shared result object's legacy attribute says m5/s4.  The
            # integral sigma(v) v^3 dv is dimensionally m6/s4, so new outputs
            # use the corrected public label without changing its value.
            "loading_integral_m6_per_s4": loading.integral_value_m5_per_s4,
            "quadrature_method": loading.quadrature_method,
            "velocity_min_m_per_s": loading.velocity_min_m_per_s,
            "velocity_max_m_per_s": loading.velocity_max_m_per_s,
            "velocity_step_m_per_s": spec.search.analysis_velocity_step_m_per_s,
            "maximum_capture_threshold_m_per_s": float(
                max(sample.capture_velocity_m_per_s for sample in samples)
            ),
            "sampling_uncertainty": asdict(uncertainty),
            "primary_uncertainty": (
                "direction-clustered standard error across 10 independently oriented "
                "disc-level rates, each estimated from 25 random impact points"
            ),
            "upper_termination_counts": {
                "escaped": upper_escaped_count,
                "timeout": upper_timeout_count,
                "other": upper_other_count,
                "other_reasons": upper_other_reasons,
            },
            "formula": "R = 9.1196e5 integral sigma_cap(v) v^3 exp(-v^2/5.667e4) dv",
        },
    )

    cooling_beams = [beam for beam in beams if beam.family == "cooling"]
    repump_beams = [beam for beam in beams if beam.family == "repump"]
    cooling_intensity = _beam_peak_intensity_w_per_m2(cooling_beams[0])
    repump_intensity = _beam_peak_intensity_w_per_m2(repump_beams[0])
    elapsed_s = time.monotonic() - started
    row: dict[str, object] = {
        "parameter_index": spec.parameter_index,
        "parameter_key": spec.parameter_key,
        "run_signature": spec.run_signature,
        "model": _MODEL_NAME,
        "indexed_state_count": model.state_count,
        "cooling_specification_state_count": 23,
        "repumper_fprime0_extension": True,
        "cooling_power_w_per_beam": cooling_beams[0].power_w,
        "repump_power_w_per_beam": repump_beams[0].power_w,
        "beam_diameter_mm": 2.0e3 * cooling_beams[0].beam_radius_m,
        "cooling_peak_intensity_w_per_m2": cooling_intensity,
        "cooling_peak_intensity_ratio": (
            cooling_intensity / config.saturation_intensity_w_per_m2
        ),
        "repump_peak_intensity_w_per_m2": repump_intensity,
        "sampling_disc_radius_mm": 1.0e3 * spec.search.disc_radius_m,
        "spectrum_velocity_min_m_per_s": float(spectrum_velocity[0]),
        "spectrum_velocity_max_m_per_s": float(spectrum_velocity[-1]),
        "spectrum_velocity_step_m_per_s": spec.search.analysis_velocity_step_m_per_s,
        "maximum_capture_threshold_m_per_s": float(np.max(capture_velocity)),
        "loading_rate_atoms_per_s": loading.loading_rate_atoms_per_s,
        "simulation_replicate_count": PLOT_POINT_DISC_COUNT,
        "launch_disc_count": PLOT_POINT_DISC_COUNT,
        "points_per_disc": POINTS_PER_DISC,
        "capture_threshold_simulation_count": (
            PLOT_POINT_DISC_COUNT * POINTS_PER_DISC
        ),
        "replicate_loading_rate_mean_atoms_per_s": replicate_statistics[
            "mean_atoms_per_s"
        ],
        "replicate_loading_rate_sample_std_atoms_per_s": replicate_statistics[
            "sample_std_atoms_per_s"
        ],
        "replicate_loading_rate_standard_error_atoms_per_s": replicate_statistics[
            "standard_error_atoms_per_s"
        ],
        "replicate_loading_rate_95_percent_half_width_atoms_per_s": (
            replicate_statistics[
                "nominal_student_t_95_percent_half_width_atoms_per_s"
            ]
        ),
        "loading_rate_disc_cluster_standard_error_atoms_per_s": (
            uncertainty.disc_cluster_standard_error_atoms_per_s
        ),
        "loading_integral_m6_per_s4": loading.integral_value_m5_per_s4,
        "capture_velocity_mean_m_per_s": float(np.mean(capture_velocity)),
        "capture_velocity_std_m_per_s": float(np.std(capture_velocity)),
        "valid_bracket_count": valid_bracket_count,
        "zero_capture_no_bracket_count": zero_capture_no_bracket_count,
        "upper_escaped_count": upper_escaped_count,
        "upper_timeout_count": upper_timeout_count,
        "upper_other_count": upper_other_count,
        "sample_count": len(samples),
        "geometry_sha256": geometry_sha256,
        "elapsed_s": elapsed_s,
    }
    if spec.sweep_kind == "saturation":
        row["n"] = spec.parameter_value

    _atomic_write_json(
        parameter_directory / "metadata.json",
        {
            "sweep_kind": spec.sweep_kind,
            "parameter_value": spec.parameter_value,
            "parameter": row,
            "model": _MODEL_NAME,
            "capture_dynamics": _CAPTURE_DYNAMICS,
            "rate_equation_approximation": (
                "quasi-steady populations; no optical coherences or sub-Doppler physics"
            ),
            "plot_point_estimator": (
                "loading rate calculated from all 250 capture thresholds; because the "
                "estimator is linear, this equals the arithmetic mean of 10 disc-level "
                "rates, each calculated from 25 random impact points"
            ),
            "plot_point_error_bar": (
                "direction-clustered sample standard error across the 10 disc-level "
                "loading rates; a Student-t 95 percent half-width (9 degrees of "
                "freedom) is also saved"
            ),
            "replicate_loading_rate_statistics": replicate_statistics,
            "replicate_loading_rates_file": "replicate_loading_rates.csv",
            "common_random_geometry_note": (
                "the same seeded 10 disc directions and 25 impact points per disc are "
                "reused across plot points, improving paired comparisons and "
                "correlating neighboring plotted estimates"
            ),
            "internal_frequency_units": "angular frequency in rad/s",
            "capture_integrator": "semi-implicit Euler; fixed external-motion timestep",
            "capture_endpoint_policy": (
                "two_core_entries and bounded_core_residence are trapped: respectively, "
                "two entries into the 2 mm core with an intervening exit, or at least "
                "5 ms of continuous residence in that core. Escaped and timeout are "
                "explicit untrapped endpoints; non_finite and all other reasons are fatal"
            ),
            "upper_termination_counts": {
                "escaped": upper_escaped_count,
                "timeout": upper_timeout_count,
                "other": upper_other_count,
                "other_reasons": upper_other_reasons,
            },
            "timeout_interpretation": (
                "untrapped under search_config.max_simulation_time_s; compare with a "
                "longer timeout during capture convergence validation"
            ),
            "scientific_limitations": (
                [_BEAM_SIZE_APERTURE_LIMITATION]
                if spec.sweep_kind == "beam_size"
                else []
            ),
            "multilevel_config": asdict(config),
            "apparatus_config": asdict(apparatus),
            "coil_config": asdict(coil),
            "search_config": asdict(spec.search),
            "geometry_sampler": (
                "10 directions uniform in solid angle in one symmetry octant; 25 "
                "independent uniform-area random points per disc with uniform azimuth; "
                "no forced center or boundary points; all velocities are parallel to "
                "the disc's inward normal rather than aimed point-by-point at the origin"
            ),
            "spectrum_bounds": {
                "velocity_min_m_per_s": float(spectrum_velocity[0]),
                "velocity_max_m_per_s": float(spectrum_velocity[-1]),
                "velocity_step_m_per_s": spec.search.analysis_velocity_step_m_per_s,
                "maximum_capture_threshold_m_per_s": float(np.max(capture_velocity)),
                "upper_bound_rule": (
                    "max(configured ceiling, one full analysis step above maximum capture threshold)"
                ),
            },
            "beam_components": [asdict(beam) for beam in beams],
            "geometry_sha256": geometry_sha256,
        },
    )
    # The aggregate row is the completion marker and is written last.
    _atomic_write_json(parameter_directory / "aggregate_row.json", row)
    print(
        f"[multilevel {spec.sweep_kind} study] plot point "
        f"{spec.parameter_index + 1}/{spec.parameter_count} "
        f"({label}={spec.parameter_value:g}) complete; 10 discs x 25 points, "
        f"R={loading.loading_rate_atoms_per_s:.6g} atoms/s; "
        f"elapsed={elapsed_s / 60.0:.1f} min",
        flush=True,
    )
    return row


def _completion_row_valid(
    row: Mapping,
    parameter_directory: Path,
    run_signature: str,
    expected_sample_count: int,
) -> bool:
    required_files = (
        "capture_velocity_samples.csv",
        "capture_velocity_spectrum.csv",
        "replicate_loading_rates.csv",
        "loading_rate.json",
        "metadata.json",
    )
    try:
        return (
            row.get("run_signature") == run_signature
            and bool(row.get("parameter_key"))
            and parameter_directory.name == str(row["parameter_key"])
            and int(float(row["sample_count"])) == expected_sample_count
            and int(float(row["simulation_replicate_count"]))
            == PLOT_POINT_DISC_COUNT
            and int(float(row["launch_disc_count"])) == PLOT_POINT_DISC_COUNT
            and int(float(row["points_per_disc"])) == POINTS_PER_DISC
            and int(float(row["capture_threshold_simulation_count"]))
            == expected_sample_count
            and (
                int(float(row["upper_escaped_count"]))
                + int(float(row["upper_timeout_count"]))
                == expected_sample_count
            )
            and int(float(row["upper_other_count"])) == 0
            and np.isfinite(float(row["loading_rate_atoms_per_s"]))
            and float(row["loading_rate_atoms_per_s"]) >= 0.0
            and all((parameter_directory / name).is_file() for name in required_files)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _completed_rows(
    output_directory: Path,
    run_signature: str,
    expected_sample_count: int,
) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    parameters_directory = output_directory / "parameters"
    if not parameters_directory.is_dir():
        return rows
    for row_path in parameters_directory.glob("*/aggregate_row.json"):
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _completion_row_valid(
            row,
            row_path.parent,
            run_signature,
            expected_sample_count,
        ):
            rows[str(row["parameter_key"])] = row
    return rows


def _validate_parameter_worker_result(
    row: Mapping,
    spec: _ParameterWorkerSpec,
) -> None:
    """Ensure a worker result belongs to its submitted parameter job."""

    try:
        matches_spec = (
            str(row["parameter_key"]) == spec.parameter_key
            and str(row["run_signature"]) == spec.run_signature
            and int(float(row["parameter_index"])) == spec.parameter_index
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"malformed worker result for {spec.sweep_kind}/{spec.parameter_key}"
        ) from error
    parameter_directory = _parameter_directory(spec.output_directory, spec.parameter_key)
    expected_sample_count = spec.search.disc_count * spec.search.points_per_disc
    if not matches_spec or not _completion_row_valid(
        row,
        parameter_directory,
        spec.run_signature,
        expected_sample_count,
    ):
        raise RuntimeError(
            f"invalid worker result for {spec.sweep_kind}/{spec.parameter_key}"
        )


def _execute_sample_shard_workers(
    prepared_sweeps: Sequence[_PreparedLoadingSweep],
    *,
    worker_count: int,
) -> dict[str, list[dict]]:
    """Run one study's internal samples through a work-stealing process pool.

    Capture workers are deliberately file-system silent.  The parent merges
    each returned shard into the signed partial CSV and writes the
    matching checkpoint marker atomically.  Consequently old partial runs are
    directly resumable, completion order cannot lose rows, and multiple
    workers may safely help the same slow plot-point simulation.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if len(prepared_sweeps) != 1:
        raise ValueError(
            "sample-shard execution accepts exactly one separately presented study"
        )
    sweep_kinds = [prepared.sweep_kind for prepared in prepared_sweeps]
    if len(set(sweep_kinds)) != len(sweep_kinds):
        raise ValueError("sample-shard execution requires unique sweep kinds")

    rows_by_sweep: dict[str, dict[str, dict]] = {
        prepared.sweep_kind: {
            key: dict(value) for key, value in prepared.existing_rows.items()
        }
        for prepared in prepared_sweeps
    }
    for prepared in prepared_sweeps:
        ordered_existing = sorted(
            rows_by_sweep[prepared.sweep_kind].values(),
            key=lambda row: int(float(row["parameter_index"])),
        )
        _atomic_write_csv(
            prepared.output_directory / "aggregate.csv",
            ordered_existing,
            prepared.aggregate_fields,
        )

    pending_by_sweep: list[
        list[tuple[_PreparedLoadingSweep, _ParameterWorkerSpec]]
    ] = []
    for prepared in prepared_sweeps:
        rows = rows_by_sweep[prepared.sweep_kind]
        pending_by_sweep.append(
            [
                (prepared, spec)
                for spec in prepared.specs
                if spec.parameter_key not in rows
            ]
        )
    pending_parameters: list[tuple[_PreparedLoadingSweep, _ParameterWorkerSpec]] = []
    longest_grid = max((len(items) for items in pending_by_sweep), default=0)
    for parameter_index in range(longest_grid):
        for items in pending_by_sweep:
            if parameter_index < len(items):
                pending_parameters.append(items[parameter_index])

    if not pending_parameters:
        prepared = prepared_sweeps[0]
        total = len(prepared.specs)
        print(
            f"[multilevel {prepared.sweep_kind} study] all {total} plot points "
            "already complete (10 launch discs x 25 points per plot point)",
            flush=True,
        )
        return {
            prepared.sweep_kind: sorted(
                rows_by_sweep[prepared.sweep_kind].values(),
                key=lambda row: int(float(row["parameter_index"])),
            )
            for prepared in prepared_sweeps
        }

    scheduler_started = time.monotonic()
    def accept_parameter_row(
        prepared: _PreparedLoadingSweep,
        spec: _ParameterWorkerSpec,
        row: Mapping,
    ) -> None:
        _validate_parameter_worker_result(row, spec)
        rows = rows_by_sweep[prepared.sweep_kind]
        rows[str(row["parameter_key"])] = dict(row)
        ordered_rows = sorted(
            rows.values(),
            key=lambda item: int(float(item["parameter_index"])),
        )
        _atomic_write_csv(
            prepared.output_directory / "aggregate.csv",
            ordered_rows,
            prepared.aggregate_fields,
        )
        elapsed_s = time.monotonic() - scheduler_started
        print(
            f"[multilevel {prepared.sweep_kind} study] plot points complete "
            f"{len(rows)}/{len(prepared.specs)} (10 discs x 25 points each); "
            f"elapsed={elapsed_s / 60.0:.1f} min",
            flush=True,
        )

    states: list[_ParameterShardState] = []
    shards_by_state: list[tuple[_ParameterShardState, list[_SampleShardSpec]]] = []
    for prepared, spec in pending_parameters:
        points = tuple(generate_common_capture_points(spec.search))
        parameter_directory = _parameter_directory(
            spec.output_directory,
            spec.parameter_key,
        )
        parameter_directory.mkdir(parents=True, exist_ok=True)
        results = _checkpointed_parameter_results(spec, points)
        state = _ParameterShardState(
            prepared=prepared,
            spec=spec,
            points=points,
            results=results,
            initial_result_count=len(results),
            started_monotonic_s=time.monotonic(),
        )
        states.append(state)
        missing = [
            point
            for point in points
            if (point.disc_index, point.point_index) not in results
        ]
        label = "n" if spec.sweep_kind == "saturation" else "d_mm"
        print(
            f"[multilevel {spec.sweep_kind} study] plot point "
            f"{spec.parameter_index + 1}/{spec.parameter_count} "
            f"({label}={spec.parameter_value:g}); resumed "
            f"{len(results)}/{PLOT_POINT_DISC_COUNT * POINTS_PER_DISC} "
            "capture-threshold simulations",
            flush=True,
        )
        if not missing:
            row = _run_parameter_worker(replace(spec, resume=True))
            accept_parameter_row(prepared, spec, row)
            continue
        # Keep each shard within one disc so a worker reuses one rebuilt
        # 24-state model without confusing the direction-cluster progress or
        # checkpoint semantics.  Small five-point shards retain work stealing.
        chunks: list[tuple[PointSample, ...]] = []
        for disc_index in range(PLOT_POINT_DISC_COUNT):
            disc_missing = [
                point for point in missing if point.disc_index == disc_index
            ]
            chunks.extend(
                tuple(disc_missing[start : start + SAMPLE_SHARD_SIZE])
                for start in range(0, len(disc_missing), SAMPLE_SHARD_SIZE)
            )
        shards_by_state.append(
            (
                state,
                [
                    _SampleShardSpec(
                        parameter=spec,
                        shard_index=index,
                        shard_count=len(chunks),
                        points=chunk,
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )
        )

    pending_shards: list[tuple[_ParameterShardState, _SampleShardSpec]] = []
    longest_shard_list = max(
        (len(shards) for _state, shards in shards_by_state),
        default=0,
    )
    # Round-robin submission keeps every plot point in this one study moving.
    for shard_index in range(longest_shard_list):
        for state, shards in shards_by_state:
            if shard_index < len(shards):
                pending_shards.append((state, shards[shard_index]))

    finalized_parameters = {
        (state.spec.sweep_kind, state.spec.parameter_key)
        for state in states
        if len(state.results) == len(state.points)
    }

    def accept_shard(
        state: _ParameterShardState,
        shard: _SampleShardSpec,
        result: _SampleShardResult,
    ) -> None:
        _validate_sample_shard_result(shard, result)
        result_keys = {
            (sample.disc_index, sample.point_index) for sample in result.samples
        }
        duplicate_keys = result_keys.intersection(state.results)
        if duplicate_keys:
            raise RuntimeError(
                f"duplicate sample shard result for {state.spec.sweep_kind}/"
                f"{state.spec.parameter_key}: {sorted(duplicate_keys)!r}"
            )
        state.results.update(
            {
                (sample.disc_index, sample.point_index): sample
                for sample in result.samples
            }
        )
        parameter_directory = _parameter_directory(
            state.spec.output_directory,
            state.spec.parameter_key,
        )
        # The parent is the only writer for this plot point.  Each accepted
        # returned shard is checkpointed immediately.
        _save_checkpoint(
            parameter_directory,
            list(state.results.values()),
            state.spec,
        )
        label = "n" if state.spec.sweep_kind == "saturation" else "d_mm"
        elapsed_s = time.monotonic() - state.started_monotonic_s
        newly_completed = max(
            1,
            len(state.results) - state.initial_result_count,
        )
        throughput = newly_completed / elapsed_s if elapsed_s > 0.0 else 0.0
        remaining = len(state.points) - len(state.results)
        eta_s = remaining / throughput if throughput > 0.0 else float("inf")
        disc_index = result.samples[0].disc_index
        if any(sample.disc_index != disc_index for sample in result.samples):
            raise RuntimeError("a capture shard crossed launch-disc boundaries")
        completed_on_disc = sum(
            sample.disc_index == disc_index for sample in state.results.values()
        )
        print(
            f"[multilevel {state.spec.sweep_kind} study] plot point "
            f"{state.spec.parameter_index + 1}/{state.spec.parameter_count} "
            f"({label}={state.spec.parameter_value:g}); disc {disc_index + 1}/"
            f"{PLOT_POINT_DISC_COUNT} points {completed_on_disc}/{POINTS_PER_DISC}; "
            f"all samples {len(state.results)}/"
            f"{PLOT_POINT_DISC_COUNT * POINTS_PER_DISC}; "
            f"ETA={eta_s / 60.0:.1f} min",
            flush=True,
        )
        parameter_identity = (
            state.spec.sweep_kind,
            state.spec.parameter_key,
        )
        if len(state.results) == len(state.points):
            if parameter_identity in finalized_parameters:
                raise RuntimeError(
                    f"parameter finalized twice: {state.spec.sweep_kind}/"
                    f"{state.spec.parameter_key}"
                )
            finalized_parameters.add(parameter_identity)
            # Reuse the established aggregation and output path.  With a full
            # parent-written checkpoint this performs no trajectory work.
            row = _run_parameter_worker(replace(state.spec, resume=True))
            accept_parameter_row(state.prepared, state.spec, row)

    if worker_count == 1:
        for state, shard in pending_shards:
            try:
                shard_result = _run_sample_shard_worker(shard)
            except Exception as error:
                raise RuntimeError(
                    f"sample shard worker failed for {state.spec.sweep_kind}/"
                    f"{state.spec.parameter_key}, shard {shard.shard_index + 1}/"
                    f"{shard.shard_count}"
                ) from error
            accept_shard(state, shard, shard_result)
    elif pending_shards:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_sample_shard_worker, shard): (state, shard)
                for state, shard in pending_shards
            }
            try:
                for future in as_completed(futures):
                    state, shard = futures[future]
                    try:
                        shard_result = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"sample shard worker failed for {state.spec.sweep_kind}/"
                            f"{state.spec.parameter_key}, shard {shard.shard_index + 1}/"
                            f"{shard.shard_count}"
                        ) from error
                    accept_shard(state, shard, shard_result)
            except BaseException:
                # In particular, do not let a KeyboardInterrupt drain every
                # queued shard before the executor context can exit.  Running
                # shards finish or receive the console interrupt; queued work
                # is canceled and every accepted shard is already checkpointed.
                for remaining_future in futures:
                    remaining_future.cancel()
                raise

    return {
        prepared.sweep_kind: sorted(
            rows_by_sweep[prepared.sweep_kind].values(),
            key=lambda row: int(float(row["parameter_index"])),
        )
        for prepared in prepared_sweeps
    }


def _validate_aggregate_rows(
    rows: Sequence[Mapping],
    search: RateCaptureSearchConfig,
) -> None:
    expected = search.disc_count * search.points_per_disc
    if expected != PLOT_POINT_DISC_COUNT * POINTS_PER_DISC:
        raise RuntimeError("loading study must contain exactly 10 discs x 25 points")
    if len({str(row["geometry_sha256"]) for row in rows}) != 1:
        raise RuntimeError("parameter values did not use identical launch geometry")
    for row in rows:
        if int(float(row["sample_count"])) != expected:
            raise RuntimeError("aggregate sample count does not match configured geometry")
        valid_brackets = int(float(row["valid_bracket_count"]))
        zero_capture = int(float(row["zero_capture_no_bracket_count"]))
        if valid_brackets + zero_capture != expected:
            raise RuntimeError(
                "one or more capture searches is neither bracketed nor an explicit zero-capture result"
            )
        upper_escaped = int(float(row["upper_escaped_count"]))
        upper_timeout = int(float(row["upper_timeout_count"]))
        upper_other = int(float(row["upper_other_count"]))
        if upper_other != 0:
            raise RuntimeError(
                "one or more capture searches has a nonfinite or otherwise invalid upper endpoint"
            )
        if upper_escaped + upper_timeout != expected:
            raise RuntimeError(
                "upper endpoint termination counts do not match the configured sample count"
            )
        loading_rate = float(row["loading_rate_atoms_per_s"])
        if not np.isfinite(loading_rate) or loading_rate < 0.0:
            raise RuntimeError("non-finite or negative loading rate")
        if int(float(row["simulation_replicate_count"])) != PLOT_POINT_DISC_COUNT:
            raise RuntimeError("aggregate disc-level replicate count is not exactly 10")
        replicate_mean = float(row["replicate_loading_rate_mean_atoms_per_s"])
        replicate_std = float(row["replicate_loading_rate_sample_std_atoms_per_s"])
        replicate_se = float(row["replicate_loading_rate_standard_error_atoms_per_s"])
        replicate_half_width = float(
            row["replicate_loading_rate_95_percent_half_width_atoms_per_s"]
        )
        if not np.isclose(loading_rate, replicate_mean, rtol=1.0e-12, atol=1.0e-9):
            raise RuntimeError("plotted loading rate is not the 10-disc mean")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (replicate_std, replicate_se, replicate_half_width)
        ):
            raise RuntimeError("invalid 10-disc uncertainty statistic")
        if not np.isclose(
            float(row["sampling_disc_radius_mm"]),
            12.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("sampling-disc radius changed during sweep")
        spectrum_minimum = float(row["spectrum_velocity_min_m_per_s"])
        spectrum_maximum = float(row["spectrum_velocity_max_m_per_s"])
        spectrum_step = float(row["spectrum_velocity_step_m_per_s"])
        maximum_capture = float(row["maximum_capture_threshold_m_per_s"])
        if spectrum_minimum != 0.0:
            raise RuntimeError("capture spectrum must begin at 0 m/s")
        if spectrum_maximum + 1.0e-12 < maximum_capture + spectrum_step:
            raise RuntimeError("capture spectrum does not extend one step beyond its largest threshold")


def _prepare_loading_sweep(
    sweep_kind: str,
    values: Sequence[float],
    *,
    output_directory: Path,
    figure_directory: Path,
    search: RateCaptureSearchConfig,
    worker_count: int,
    resume: bool,
    execution_mode: str,
) -> _PreparedLoadingSweep:
    _validate_plot_point_replicate_design(search)
    normalized_values = tuple(float(value) for value in values)
    if not normalized_values or any(value <= 0.0 for value in normalized_values):
        raise ValueError("sweep values must be nonempty and positive")
    signature, signature_payload = _run_signature(sweep_kind, normalized_values, search)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    label = "n" if sweep_kind == "saturation" else "diameter_mm"
    specs = [
        _ParameterWorkerSpec(
            sweep_kind=sweep_kind,
            parameter_index=index,
            parameter_count=len(normalized_values),
            parameter_value=value,
            parameter_key=_parameter_key(index, label, value),
            run_signature=signature,
            search=search,
            output_directory=output_directory,
            resume=resume,
        )
        for index, value in enumerate(normalized_values)
    ]
    discovered_existing = (
        _completed_rows(
            output_directory,
            signature,
            search.disc_count * search.points_per_disc,
        )
        if resume
        else {}
    )
    expected_parameter_keys = {spec.parameter_key for spec in specs}
    existing = {
        key: row
        for key, row in discovered_existing.items()
        if key in expected_parameter_keys
    }
    _atomic_write_json(
        output_directory / "sweep_metadata.json",
        {
            "sweep_kind": sweep_kind,
            "run_signature": signature,
            "signature_payload": signature_payload,
            "parameter_values": normalized_values,
            "parameter_count": len(normalized_values),
            "plot_point_count": len(normalized_values),
            "launch_geometry_discs_per_plot_point": PLOT_POINT_DISC_COUNT,
            "points_per_launch_disc": POINTS_PER_DISC,
            "simulations_per_plot_point": PLOT_POINT_DISC_COUNT * POINTS_PER_DISC,
            "simulation_definition": (
                "one deterministic capture-threshold search for one random impact point; "
                "250 searches are arranged as 10 random-direction discs x 25 points"
            ),
            "simulation_independence": (
                "each capture trajectory has independent external state; workers rebuild "
                "multilevel, apparatus, beam, and field inputs from defaults, and no "
                "dynamical or optical state carries between trajectories"
            ),
            "plot_point_estimator": (
                "loading rate from all 250 capture thresholds; equivalently the mean of "
                "10 disc-level rates, each estimated from 25 impact points"
            ),
            "common_random_geometry_note": (
                "the same seeded 10 directions and 25 points per direction are reused "
                "across plot points for paired curve comparisons; this correlates "
                "neighboring plotted estimates but does not carry simulation state"
            ),
            # Retained for compatibility with the initial output schema; the
            # workers now execute sample shards rather than whole parameters.
            "parameter_worker_count": worker_count,
            "sample_shard_size": SAMPLE_SHARD_SIZE,
            "checkpoint_writer": "parent process only",
            "execution_mode": execution_mode,
            "statistics_output_directory": output_directory,
            "figure_output_directory": figure_directory,
            "model": _MODEL_NAME,
            "capture_dynamics": _CAPTURE_DYNAMICS,
            "parameter_grid_definition": (
                "n = 1, 2, ..., 35"
                if sweep_kind == "saturation"
                else (
                    "25 evenly spaced diameters; d[7] = 12.7 mm, d[24] = 30 mm; "
                    f"step = {_BEAM_DIAMETER_STEP_MM:.16g} mm; "
                    f"start = {BEAM_DIAMETER_MM_VALUES[0]:.16g} mm"
                )
            ),
            "scientific_status": (
                "requires timestep/timeout convergence and comparison with short event-driven "
                "trajectories before trusted absolute loading claims; n up to 35 also probes "
                "the high-saturation edge of the rate-equation approximation"
            ),
            "scientific_limitations": (
                [_BEAM_SIZE_APERTURE_LIMITATION]
                if sweep_kind == "beam_size"
                else []
            ),
        },
    )
    fields = (
        _SATURATION_AGGREGATE_FIELDS
        if sweep_kind == "saturation"
        else _BEAM_SIZE_AGGREGATE_FIELDS
    )
    return _PreparedLoadingSweep(
        sweep_kind=sweep_kind,
        values=normalized_values,
        output_directory=output_directory,
        figure_directory=figure_directory,
        search=search,
        run_signature=signature,
        specs=tuple(specs),
        existing_rows=existing,
        aggregate_fields=tuple(fields),
    )


def _finalize_loading_sweep(
    prepared: _PreparedLoadingSweep,
    rows: Sequence[Mapping],
) -> list[dict]:
    finalized_rows = [dict(row) for row in rows]
    expected_keys = {spec.parameter_key for spec in prepared.specs}
    observed_keys = {str(row.get("parameter_key", "")) for row in finalized_rows}
    if len(finalized_rows) != len(prepared.specs) or observed_keys != expected_keys:
        raise RuntimeError(
            f"{prepared.sweep_kind} aggregate does not match its requested parameter grid"
        )
    _validate_aggregate_rows(finalized_rows, prepared.search)
    if prepared.sweep_kind == "saturation":
        plot_multilevel_loading_rate_vs_saturation(
            prepared.output_directory / "aggregate.csv",
            prepared.figure_directory / "loading_rate_vs_saturation.png",
        )
    else:
        plot_multilevel_loading_rate_vs_beam_size(
            prepared.output_directory / "aggregate.csv",
            prepared.figure_directory / "loading_rate_vs_beam_size.png",
        )
    return finalized_rows


def _run_loading_sweep(
    sweep_kind: str,
    values: Sequence[float],
    *,
    output_directory: Path,
    figure_directory: Path,
    search: RateCaptureSearchConfig,
    worker_count: int,
    resume: bool,
) -> list[dict]:
    study_label = "saturation" if sweep_kind == "saturation" else "beam-size"
    print(
        f"[multilevel {study_label} study] starting {len(values)} plot points; "
        f"exactly {PLOT_POINT_DISC_COUNT} random launch discs x {POINTS_PER_DISC} "
        "random points per disc (250 capture thresholds per point); each point "
        "rebuilds the 24-state MOT from defaults",
        flush=True,
    )
    prepared = _prepare_loading_sweep(
        sweep_kind,
        values,
        output_directory=output_directory,
        figure_directory=figure_directory,
        search=search,
        worker_count=worker_count,
        resume=resume,
        execution_mode="single_sweep_sample_shard_pool",
    )
    rows_by_sweep = _execute_sample_shard_workers(
        (prepared,),
        worker_count=worker_count,
    )
    return _finalize_loading_sweep(
        prepared,
        rows_by_sweep[prepared.sweep_kind],
    )


def run_multilevel_saturation_loading_sweep(
    *,
    output_directory: Path,
    figure_directory: Path,
    search: RateCaptureSearchConfig | None = None,
    worker_count: int = DEFAULT_PARAMETER_WORKER_COUNT,
    resume: bool = True,
    n_values: Sequence[float] = SATURATION_N_VALUES,
) -> list[dict]:
    """Run loading rate versus ``n = I0 / I_sat`` with multilevel forces only."""

    normalized_search = replace(
        search or RateCaptureSearchConfig(),
        disc_count=PLOT_POINT_DISC_COUNT,
        points_per_disc=POINTS_PER_DISC,
        include_center_point=False,
        worker_count=1,
    )
    return _run_loading_sweep(
        "saturation",
        n_values,
        output_directory=output_directory,
        figure_directory=figure_directory,
        search=normalized_search,
        worker_count=worker_count,
        resume=resume,
    )


def run_multilevel_beam_size_loading_sweep(
    *,
    output_directory: Path,
    figure_directory: Path,
    search: RateCaptureSearchConfig | None = None,
    worker_count: int = DEFAULT_PARAMETER_WORKER_COUNT,
    resume: bool = True,
    diameter_mm_values: Sequence[float] = BEAM_DIAMETER_MM_VALUES,
) -> list[dict]:
    """Run loading rate versus diameter with fixed cooling/repump peak intensity."""

    normalized_search = replace(
        search or RateCaptureSearchConfig(),
        disc_count=PLOT_POINT_DISC_COUNT,
        points_per_disc=POINTS_PER_DISC,
        include_center_point=False,
        worker_count=1,
    )
    return _run_loading_sweep(
        "beam_size",
        diameter_mm_values,
        output_directory=output_directory,
        figure_directory=figure_directory,
        search=normalized_search,
        worker_count=worker_count,
        resume=resume,
    )


def plot_multilevel_loading_rate_vs_saturation(
    aggregate_csv: Path,
    output_path: Path,
) -> Path:
    """Plot saved multilevel loading results without fitting a model."""

    rows = _read_csv_rows(aggregate_csv)
    if not rows:
        raise ValueError(f"no saturation rows found in {aggregate_csv}")
    rows.sort(key=lambda row: float(row["n"]))
    values = np.asarray([float(row["n"]) for row in rows])
    loading = np.asarray([float(row["loading_rate_atoms_per_s"]) for row in rows])
    errors = np.asarray(
        [float(row["replicate_loading_rate_standard_error_atoms_per_s"]) for row in rows]
    )
    saturation_intensity = USER_SATURATION_INTENSITY_W_PER_M2
    figure, axis = plt.subplots(figsize=(8.5, 5.9), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.errorbar(values, loading, yerr=errors, marker="o", linewidth=1.7, capsize=2.5)
    axis.set_title("Mean Multilevel MOT Loading Rate vs Peak Cooling-Beam Saturation")
    axis.set_xlabel(r"Peak cooling-beam intensity ratio $n=I_0/I_{\rm sat}$")
    axis.set_ylabel(r"Loading rate $R$ [atoms s$^{-1}$]")
    axis.set_xlim(0.0, 35.0)
    axis.set_xticks(np.arange(0.0, 36.0, 5.0))
    axis.grid(True, alpha=0.25)
    axis.text(
        0.03,
        0.96,
        (
            r"$I_0=nI_{\rm sat},\quad s_0=I_0/I_{\rm sat}=n$"
            + "\n"
            + rf"Multilevel $I_{{\rm sat}}={saturation_intensity:g}\ \rm W\,m^{{-2}}$"
            + "\n10 random directions $\\times$ 25 random disc points"
            + "\nError bars: direction-clustered SEM across 10 disc rates"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_multilevel_loading_rate_vs_beam_size(
    aggregate_csv: Path,
    output_path: Path,
) -> Path:
    """Plot saved multilevel loading results versus Gaussian diameter."""

    rows = _read_csv_rows(aggregate_csv)
    if not rows:
        raise ValueError(f"no beam-size rows found in {aggregate_csv}")
    rows.sort(key=lambda row: float(row["beam_diameter_mm"]))
    diameter = np.asarray([float(row["beam_diameter_mm"]) for row in rows])
    loading = np.asarray([float(row["loading_rate_atoms_per_s"]) for row in rows])
    errors = np.asarray(
        [float(row["replicate_loading_rate_standard_error_atoms_per_s"]) for row in rows]
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.9), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.errorbar(
        diameter,
        loading,
        yerr=errors,
        marker="o",
        linewidth=1.7,
        capsize=2.5,
        color="#7c3aed",
    )
    axis.axvline(12.7, color="#64748b", linestyle="--", linewidth=1.0, label="default 12.7 mm")
    axis.set_title("Mean Multilevel MOT Loading Rate vs Gaussian Beam Diameter")
    axis.set_xlabel(r"Gaussian beam diameter $d$ [mm]")
    axis.set_ylabel(r"Loading rate $R$ [atoms s$^{-1}$]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.text(
        0.03,
        0.96,
        (
            r"Cooling power scales as $(d/12.7\ \mathrm{mm})^2$"
            + "\nRepump power and diameter remain at their default baseline values"
            + "\nCooling peak intensity fixed; sampling-disc radius fixed at 12 mm"
            + "\n10 random directions $\\times$ 25 random disc points"
            + "\nError bars: direction-clustered SEM across 10 disc rates"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.text(
        0.03,
        0.04,
        (
            r"APERTURE LIMIT: $\sigma_{\rm cap}\leq\pi(12\ \mathrm{mm})^2$"
            + "\nNo exact boundary point was sampled; assess high-d truncation from saved near-rim points"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#9a3412",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#fff7ed",
            "edgecolor": "#ea580c",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _default_output_directories(sweep_kind: str) -> tuple[Path, Path]:
    paths = multilevel_mot_paths()
    name = (
        "loading_vs_saturation_rate_equation_10_discs_25_points"
        if sweep_kind == "saturation"
        else "loading_vs_beam_size_rate_equation_10_discs_25_points"
    )
    return paths["statistics"] / "august_22" / name, paths["figures"] / "august_22" / name


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_single_sweep_paths: bool = True,
) -> None:
    defaults = RateCaptureSearchConfig()
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARAMETER_WORKER_COUNT,
        help=(
            "parallel capture-threshold worker processes "
            f"(default: {DEFAULT_PARAMETER_WORKER_COUNT})"
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    if include_single_sweep_paths:
        parser.add_argument("--output-dir", type=Path)
        parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multilevel-only August 22 loading sweeps"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_common_arguments(
        subparsers.add_parser("saturation", help="n=1..35 cooling-saturation sweep")
    )
    _add_common_arguments(
        subparsers.add_parser("beam-size", help="25 beam diameters through 30 mm")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    sweep_kind = "saturation" if args.command == "saturation" else "beam_size"
    default_output, default_figures = _default_output_directories(sweep_kind)
    output = args.output_dir or default_output
    figures = args.figures_dir or default_figures
    if args.plot_only:
        if sweep_kind == "saturation":
            plot_multilevel_loading_rate_vs_saturation(
                output / "aggregate.csv",
                figures / "loading_rate_vs_saturation.png",
            )
        else:
            plot_multilevel_loading_rate_vs_beam_size(
                output / "aggregate.csv",
                figures / "loading_rate_vs_beam_size.png",
            )
        return 0

    search = replace(
        RateCaptureSearchConfig(),
        disc_count=PLOT_POINT_DISC_COUNT,
        points_per_disc=POINTS_PER_DISC,
        seed=args.seed,
        worker_count=1,
    )
    if sweep_kind == "saturation":
        run_multilevel_saturation_loading_sweep(
            output_directory=output,
            figure_directory=figures,
            search=search,
            worker_count=args.workers,
            resume=not args.no_resume,
        )
    else:
        run_multilevel_beam_size_loading_sweep(
            output_directory=output,
            figure_directory=figures,
            search=search,
            worker_count=args.workers,
            resume=not args.no_resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEAM_DIAMETER_MM_VALUES",
    "PLOT_POINT_DISC_COUNT",
    "PLOT_POINT_REPLICATE_COUNT",
    "POINTS_PER_DISC",
    "POINTS_PER_REPLICATE",
    "SATURATION_N_VALUES",
    "USER_SATURATION_INTENSITY_W_PER_M2",
    "beam_size_power_w_per_beam",
    "build_dynamic_capture_spectrum",
    "build_argument_parser",
    "build_multilevel_loading_configuration",
    "classify_multilevel_loading_trajectory",
    "find_multilevel_capture_velocity",
    "generate_common_capture_points",
    "plot_multilevel_loading_rate_vs_beam_size",
    "plot_multilevel_loading_rate_vs_saturation",
    "run_multilevel_beam_size_loading_sweep",
    "run_multilevel_saturation_loading_sweep",
    "saturation_power_w_per_beam",
]
