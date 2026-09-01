"""Production multilevel-MOT ensemble temperature versus cooling detuning.

Temperature is an ensemble property, so this study evolves independently
sampled, preloaded atom clouds at every detuning.  The external dynamics use
the 24-state, repumper-enabled adiabatic population-rate model and its Langevin
recoil-diffusion term.  Only the cooling detuning changes between sweep points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter, sleep
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from ..configuration import MOTApparatusConfig, RB87_MASS_KG, default_mot_apparatus_config
from ..fields import MOTBeam
from ..magnetic_fields import default_anti_helmholtz_config
from .configuration import (
    MultilevelMOTConfig,
    default_multilevel_mot_config,
    multilevel_mot_paths,
)
from .rate_equations import (
    RateEquationAtomState,
    RateEquationModel,
    RateEquationTrajectoryConfig,
    build_rate_equation_model,
    simulate_rate_equation_trajectory,
)
from .simulation import build_multilevel_mot_beams
from .temperature import (
    BOLTZMANN_CONSTANT_J_PER_K,
    doppler_temperature_k,
    effective_saturation_parameter,
)


# Broad far-red sampling is sufficient where the response is smooth; the grid
# is progressively refined around the expected cooling optimum and resonance.
DETUNING_N_VALUES: tuple[float, ...] = (
    -10.0,
    -9.0,
    -8.0,
    -7.0,
    -6.0,
    -5.0,
    -4.0,
    -3.0,
    -2.5,
    -2.25,
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.25,
    -0.2,
    -0.15,
    -0.1,
)

REQUIRED_ENSEMBLE_REALIZATION_COUNT = 10
REQUIRED_ATOMS_PER_ENSEMBLE = 25
REQUIRED_TRAJECTORY_COUNT = (
    REQUIRED_ENSEMBLE_REALIZATION_COUNT * REQUIRED_ATOMS_PER_ENSEMBLE
)
DEFAULT_COOLING_POWER_W_PER_BEAM = (
    default_mot_apparatus_config().cooling.power_w_per_beam
)
DEFAULT_DURATION_S = 25.0e-3
DEFAULT_TIME_STEP_S = 5.0e-6
DEFAULT_WORKER_COUNT = 8
DEFAULT_INITIAL_TEMPERATURE_K = 2.0e-3
DEFAULT_INITIAL_POSITION_SIGMA_M = 0.25e-3
DEFAULT_PLATEAU_WINDOW_S = 5.0e-3
DEFAULT_RECORD_INTERVAL_S = 0.1e-3
DEFAULT_MINIMUM_SURVIVORS_PER_ENSEMBLE = 5
DEFAULT_MINIMUM_SURVIVOR_FRACTION = (
    DEFAULT_MINIMUM_SURVIVORS_PER_ENSEMBLE / REQUIRED_ATOMS_PER_ENSEMBLE
)
DEFAULT_STATIONARITY_LIMIT = 0.15
DEFAULT_SEED = 20260822
TRAPPED_CORE_RADIUS_M = 2.0e-3
ESCAPE_RADIUS_M = 30.0e-3
SCHEMA_VERSION = 5
DOPPLER_REFERENCE_VERSION = 2

PHYSICAL_MODEL_STATEMENT = (
    "At each cooling detuning, ten independently seeded realizations of a "
    "preloaded 25-atom Rb-87 cloud are evolved in the fixed default six-beam "
    "MOT using the repumper-enabled 24-state adiabatic population-rate equations "
    "and Langevin recoil diffusion.  Each realization's reported final "
    "temperature is the time average over the final 5 ms of the unbiased, "
    "instantaneous-center-of-mass-subtracted three-dimensional velocity variance; "
    "the plotted point is the arithmetic mean of the ten realization temperatures. "
    "Only the cooling detuning Delta=n Gamma changes between points."
)


def _physical_model_statement(
    ensemble_realization_count: int,
    atoms_per_ensemble: int,
    plateau_window_s: float,
) -> str:
    """Describe the sampled cloud design while retaining the legacy wording."""

    if (
        ensemble_realization_count == REQUIRED_ENSEMBLE_REALIZATION_COUNT
        and atoms_per_ensemble == REQUIRED_ATOMS_PER_ENSEMBLE
        and np.isclose(plateau_window_s, DEFAULT_PLATEAU_WINDOW_S)
    ):
        return PHYSICAL_MODEL_STATEMENT
    return (
        f"At each cooling detuning, {ensemble_realization_count} independently "
        f"seeded realizations of a preloaded {atoms_per_ensemble}-atom Rb-87 cloud "
        "are evolved in the fixed six-beam MOT using the repumper-enabled 24-state "
        "adiabatic population-rate equations and Langevin recoil diffusion. Each "
        "realization's reported final temperature is the time average over the final "
        f"{1e3 * plateau_window_s:.6g} ms of the unbiased, instantaneous-center-of-"
        "mass-subtracted three-dimensional velocity variance; the plotted point is "
        f"the arithmetic mean of the {ensemble_realization_count} realization "
        "temperatures. Only the cooling detuning Delta=n Gamma changes between points."
    )


SUMMARY_CSV_FIELDNAMES = (
    "point_index",
    "detuning_n",
    "detuning_rad_per_s",
    "detuning_hz",
    "detuning_mhz",
    "cooling_power_w_per_beam",
    "requested_ensemble_count",
    "temperature_ensemble_count",
    "valid_temperature_ensemble_count",
    "requested_atom_count",
    "complete_atom_count",
    "trapped_atom_count",
    "complete_outside_core_count",
    "escaped_atom_count",
    "other_incomplete_atom_count",
    "failed_atom_count",
    "duration_complete_fraction",
    "trapped_fraction",
    "trapped_fraction_sem",
    "trapped_fraction_ci_low",
    "trapped_fraction_ci_high",
    "trapped_fraction_pooled_wilson_low",
    "trapped_fraction_pooled_wilson_high",
    "minimum_survivors_per_ensemble",
    "median_survivors_per_ensemble",
    "maximum_survivors_per_ensemble",
    "initial_temperature_mean_k",
    "plateau_temperature_x_k",
    "plateau_temperature_y_k",
    "plateau_temperature_z_k",
    "plateau_temperature_mean_k",
    "temperature_sem_k",
    "temperature_ci_low_k",
    "temperature_ci_high_k",
    "final_temperature_mean_k",
    "final_temperature_sem_k",
    "cooling_beam_center_on_resonance_saturation_parameter",
    "cooling_beam_center_effective_saturation_parameter",
    "doppler_temperature_k",
    "plateau_over_doppler",
    "stationarity_pass_count",
    "statistics_pass_count",
    "plateau_max_radius_median_m",
    "plateau_max_radius_p90_m",
    "plateau_max_radius_max_m",
    "valid",
    "quality_status",
    "termination_counts_json",
    "boundedness_counts_json",
    "point_wall_time_s",
)


ENSEMBLE_CSV_FIELDNAMES = (
    "point_index",
    "detuning_n",
    "ensemble_index",
    "requested_atom_count",
    "successful_atom_count",
    "complete_atom_count",
    "trapped_atom_count",
    "aligned_trapped_atom_count",
    "complete_outside_core_count",
    "escaped_atom_count",
    "other_incomplete_atom_count",
    "failed_atom_count",
    "duration_complete_fraction",
    "trapped_fraction",
    "initial_temperature_x_k",
    "initial_temperature_y_k",
    "initial_temperature_z_k",
    "initial_temperature_mean_k",
    "plateau_temperature_x_k",
    "plateau_temperature_y_k",
    "plateau_temperature_z_k",
    "plateau_temperature_mean_k",
    "final_temperature_x_k",
    "final_temperature_y_k",
    "final_temperature_z_k",
    "final_temperature_mean_k",
    "plateau_window_1_mean_k",
    "plateau_window_2_mean_k",
    "plateau_window_3_mean_k",
    "plateau_window_4_mean_k",
    "stationarity_metric",
    "relative_drift",
    "stationarity_pass",
    "statistics_pass",
    "survivor_conditioning_warning",
    "valid",
    "quality_status",
    "plateau_max_radius_median_m",
    "plateau_max_radius_p90_m",
    "plateau_max_radius_max_m",
    "termination_counts_json",
    "boundedness_counts_json",
)


@dataclass(frozen=True, slots=True)
class TemperatureSweepWorkerPayload:
    """Pickle-safe input for one atom in one ensemble and detuning point."""

    point_index: int
    ensemble_index: int
    atom_index: int
    detuning_n: float
    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    recoil_seed: int
    duration_s: float
    time_step_s: float
    record_stride: int
    plateau_window_s: float
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM


@dataclass(slots=True)
class TemperatureSweepWorkerResult:
    """Subsampled trajectory data returned by a worker process."""

    ensemble_index: int
    atom_index: int
    complete: bool
    trapped: bool
    termination_reason: str
    boundedness_reason: str
    final_radius_m: float
    maximum_radius_m: float
    plateau_max_radius_m: float
    plateau_mean_radius_m: float
    times_s: np.ndarray
    velocities_m_per_s: np.ndarray


def detuning_n_grid(
    detuning_n_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Return a validated mutable ``Delta/Gamma`` grid.

    Omitting ``detuning_n_values`` preserves the established 25-point grid.
    Custom campaigns may provide any nonempty sequence of unique, finite red
    detunings.  Input order is retained because it defines checkpoint indices
    and recoil-noise streams.
    """

    values = np.asarray(
        DETUNING_N_VALUES if detuning_n_values is None else detuning_n_values,
        dtype=float,
    )
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("detuning_n_values must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values >= 0.0):
        raise ValueError("every detuning multiplier must be finite and negative")
    keys = [_detuning_key(value) for value in values]
    if len(set(keys)) != len(keys):
        raise ValueError("detuning_n_values must not contain duplicate points")
    return values.copy()


def _detuning_key(value: object) -> str:
    return format(float(value), ".12g")


def build_temperature_sweep_configuration(
    detuning_n: float,
    *,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> tuple[MultilevelMOTConfig, MOTApparatusConfig, list[MOTBeam]]:
    """Build consistent solver, apparatus, and beams for one red detuning."""

    if not np.isfinite(detuning_n) or detuning_n >= 0.0:
        raise ValueError("detuning_n must be a finite negative number")
    if (
        not np.isfinite(cooling_power_w_per_beam)
        or cooling_power_w_per_beam <= 0.0
    ):
        raise ValueError("cooling_power_w_per_beam must be finite and positive")
    base = default_multilevel_mot_config()
    detuning_rad_per_s = float(detuning_n * base.natural_linewidth_rad_per_s)
    config = replace(
        base,
        cooling_detuning_rad_per_s=detuning_rad_per_s,
        repumper_enabled=True,
    )
    apparatus_base = default_mot_apparatus_config()
    apparatus = replace(
        apparatus_base,
        cooling=replace(
            apparatus_base.cooling,
            detuning_hz=detuning_rad_per_s / (2.0 * np.pi),
            power_w_per_beam=float(cooling_power_w_per_beam),
        ),
        repump=replace(
            apparatus_base.repump,
            power_w_per_beam=float(config.repump_power_w_per_beam),
        ),
    )
    beams = build_multilevel_mot_beams(apparatus_config=apparatus, config=config)
    return config, apparatus, beams


def cooling_doppler_reference(
    detuning_n: float,
    *,
    linewidth_rad_per_s: float | None = None,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, float]:
    """Return the per-beam saturation metrics and Doppler reference at ``n``.

    The saturation is evaluated at the center of one cooling beam.  All six
    cooling components share the same fixed power and radius in this sweep.
    ``s_eff`` is detuning-reduced before it is inserted into the requested
    multilevel Doppler-temperature expression.
    """

    config, _, beams = build_temperature_sweep_configuration(
        detuning_n,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    cooling_beams = [beam for beam in beams if beam.family == "cooling"]
    if len(cooling_beams) != 6:
        raise RuntimeError("temperature sweep requires exactly six cooling beams")
    reference_beam = cooling_beams[0]
    if any(
        not np.isclose(beam.power_w, reference_beam.power_w, rtol=0.0, atol=0.0)
        or not np.isclose(
            beam.beam_radius_m,
            reference_beam.beam_radius_m,
            rtol=0.0,
            atol=0.0,
        )
        for beam in cooling_beams[1:]
    ):
        raise RuntimeError("cooling beams do not share one saturation reference")

    peak_intensity = (
        2.0
        * reference_beam.power_w
        / (np.pi * reference_beam.beam_radius_m**2)
    )
    on_resonance_saturation = (
        peak_intensity / config.saturation_intensity_w_per_m2
    )
    gamma = (
        config.natural_linewidth_rad_per_s
        if linewidth_rad_per_s is None
        else float(linewidth_rad_per_s)
    )
    detuning_rad_per_s = float(detuning_n) * gamma
    effective_saturation = effective_saturation_parameter(
        on_resonance_saturation,
        detuning_rad_per_s,
        gamma,
    )
    doppler = doppler_temperature_k(
        gamma,
        detuning_rad_per_s,
        effective_saturation,
    )
    return {
        "detuning_rad_per_s": detuning_rad_per_s,
        "cooling_beam_center_peak_intensity_w_per_m2": peak_intensity,
        "cooling_beam_center_on_resonance_saturation_parameter": (
            on_resonance_saturation
        ),
        "cooling_beam_center_effective_saturation_parameter": effective_saturation,
        "doppler_temperature_k": doppler,
    }


def _json_compatible(value: object) -> object:
    return json.loads(json.dumps(value))


def _detuning_invariant_payload(
    detuning_n: float,
    *,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, object]:
    config, apparatus, beams = build_temperature_sweep_configuration(
        detuning_n,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    config_payload = asdict(config)
    config_payload["cooling_detuning_rad_per_s"] = "<varied cooling detuning>"
    apparatus_payload = asdict(apparatus)
    apparatus_payload["cooling"]["detuning_hz"] = "<varied cooling detuning>"
    beam_payloads = []
    for beam in beams:
        payload = asdict(beam)
        if beam.family == "cooling":
            payload["detuning_hz"] = "<varied cooling detuning>"
        beam_payloads.append(payload)
    return _json_compatible(
        {
            "multilevel_config": config_payload,
            "apparatus_config": apparatus_payload,
            "beams": beam_payloads,
            "coil_config": asdict(default_anti_helmholtz_config()),
        }
    )


def verify_only_cooling_detuning_changes(
    detuning_values: Sequence[float] = DETUNING_N_VALUES,
    *,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, object]:
    """Assert that every physical configuration differs only in cooling detuning."""

    values = detuning_n_grid(detuning_values)
    reference_payload = _detuning_invariant_payload(
        float(values[0]),
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    reference_json = json.dumps(reference_payload, sort_keys=True, separators=(",", ":"))
    base = default_multilevel_mot_config()
    for value in values:
        config, apparatus, beams = build_temperature_sweep_configuration(
            float(value),
            cooling_power_w_per_beam=cooling_power_w_per_beam,
        )
        expected_rad_per_s = float(value) * base.natural_linewidth_rad_per_s
        if not np.isclose(config.cooling_detuning_rad_per_s, expected_rad_per_s):
            raise RuntimeError("solver cooling detuning is inconsistent with n Gamma")
        if not np.isclose(
            apparatus.cooling.detuning_hz,
            expected_rad_per_s / (2.0 * np.pi),
        ):
            raise RuntimeError("apparatus cooling detuning is inconsistent with n Gamma")
        if not config.repumper_enabled:
            raise RuntimeError("temperature sweep unexpectedly disabled the repumper")
        if not all(
            np.isclose(beam.detuning_hz, expected_rad_per_s / (2.0 * np.pi))
            for beam in beams
            if beam.family == "cooling"
        ):
            raise RuntimeError("cooling-beam detuning metadata is inconsistent")
        candidate_json = json.dumps(
            _detuning_invariant_payload(
                float(value),
                cooling_power_w_per_beam=cooling_power_w_per_beam,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        if candidate_json != reference_json:
            raise RuntimeError(
                "a physical parameter other than cooling detuning varies across the sweep"
            )
    return {
        "passed": True,
        "point_count": len(values),
        "cooling_power_w_per_beam": float(cooling_power_w_per_beam),
        "allowed_varied_fields": [
            "multilevel_config.cooling_detuning_rad_per_s",
            "apparatus_config.cooling.detuning_hz",
            "cooling_beams[*].detuning_hz",
        ],
        "fixed_configuration_sha256": hashlib.sha256(
            reference_json.encode("utf-8")
        ).hexdigest(),
    }


def generate_common_initial_ensembles(
    ensemble_count: int,
    atoms_per_ensemble: int,
    *,
    initial_temperature_k: float = DEFAULT_INITIAL_TEMPERATURE_K,
    initial_position_sigma_m: float = DEFAULT_INITIAL_POSITION_SIGMA_M,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent clouds once so the same clouds are reused at every n."""

    if ensemble_count <= 0 or atoms_per_ensemble < 2:
        raise ValueError("ensemble_count must be positive and atoms_per_ensemble at least two")
    if initial_temperature_k <= 0.0 or initial_position_sigma_m < 0.0 or seed < 0:
        raise ValueError("initial distribution parameters and seed are invalid")
    root = np.random.SeedSequence([seed, 0])
    positions = np.empty((ensemble_count, atoms_per_ensemble, 3), dtype=float)
    velocities = np.empty_like(positions)
    velocity_sigma = np.sqrt(
        BOLTZMANN_CONSTANT_J_PER_K * initial_temperature_k / RB87_MASS_KG
    )
    for ensemble_index, sequence in enumerate(root.spawn(ensemble_count)):
        rng = np.random.default_rng(sequence)
        positions[ensemble_index] = rng.normal(
            scale=initial_position_sigma_m,
            size=(atoms_per_ensemble, 3),
        )
        velocities[ensemble_index] = rng.normal(
            scale=velocity_sigma,
            size=(atoms_per_ensemble, 3),
        )
    return positions, velocities


def generate_common_initial_ensemble(
    atom_count: int,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible one-cloud wrapper used by older analysis notebooks."""

    positions, velocities = generate_common_initial_ensembles(1, atom_count, **kwargs)
    return positions[0], velocities[0]


def recoil_seed(
    master_seed: int,
    point_index: int,
    ensemble_index: int,
    atom_index: int,
) -> int:
    """Return an independent deterministic Langevin stream seed."""

    if min(master_seed, point_index, ensemble_index, atom_index) < 0:
        raise ValueError("seed and indices must be non-negative")
    sequence = np.random.SeedSequence(
        [master_seed, 1, point_index, ensemble_index, atom_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@lru_cache(maxsize=32)
def _worker_context(
    detuning_n: float,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> tuple[MultilevelMOTConfig, tuple[MOTBeam, ...], RateEquationModel]:
    config, _, beams = build_temperature_sweep_configuration(
        detuning_n,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    return config, tuple(beams), model


def _temperature_sweep_worker(
    payload: TemperatureSweepWorkerPayload,
) -> TemperatureSweepWorkerResult:
    config, beam_tuple, model = _worker_context(
        payload.detuning_n,
        payload.cooling_power_w_per_beam,
    )
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(payload.position_m, payload.velocity_m_per_s),
        payload.duration_s,
        default_anti_helmholtz_config(),
        beams=list(beam_tuple),
        model=model,
        config=config,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=payload.time_step_s,
            include_diffusion=True,
            seed=payload.recoil_seed,
            escape_radius_m=ESCAPE_RADIUS_M,
        ),
    )
    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    velocities = np.asarray(record.velocities_m_per_s, dtype=float)
    complete = bool(
        record.termination_reason == "duration"
        and times[-1] >= payload.duration_s - 1.0e-12
    )
    radii = np.linalg.norm(positions, axis=1)
    plateau_start_s = max(0.0, payload.duration_s - payload.plateau_window_s)
    plateau_radii = radii[times >= plateau_start_s - 1.0e-15]
    plateau_max_radius_m = (
        float(np.max(plateau_radii)) if len(plateau_radii) else float("nan")
    )
    plateau_mean_radius_m = (
        float(np.mean(plateau_radii)) if len(plateau_radii) else float("nan")
    )
    trapped = bool(
        complete
        and len(plateau_radii)
        and plateau_max_radius_m <= TRAPPED_CORE_RADIUS_M
    )
    if trapped:
        boundedness_reason = "complete_final_plateau_core"
    elif complete:
        boundedness_reason = "complete_outside_final_plateau_core"
    elif record.termination_reason == "escaped":
        boundedness_reason = "escaped"
    else:
        boundedness_reason = f"incomplete_{record.termination_reason}"
    indices = np.arange(0, len(times), payload.record_stride, dtype=int)
    if indices[-1] != len(times) - 1:
        indices = np.append(indices, len(times) - 1)
    return TemperatureSweepWorkerResult(
        ensemble_index=payload.ensemble_index,
        atom_index=payload.atom_index,
        complete=complete,
        trapped=trapped,
        termination_reason=record.termination_reason,
        boundedness_reason=boundedness_reason,
        final_radius_m=float(radii[-1]),
        maximum_radius_m=float(np.max(radii)),
        plateau_max_radius_m=plateau_max_radius_m,
        plateau_mean_radius_m=plateau_mean_radius_m,
        times_s=times[indices],
        velocities_m_per_s=velocities[indices],
    )


def _temperature_history_components_k(
    velocity_history_m_per_s: np.ndarray,
) -> np.ndarray:
    """Return unbiased component temperatures at each recorded time."""

    velocities = np.asarray(velocity_history_m_per_s, dtype=float)
    if velocities.ndim != 3 or velocities.shape[0] < 2 or velocities.shape[2] != 3:
        raise ValueError(
            "velocity history must have shape (N_atoms, N_times, 3), N_atoms >= 2"
        )
    return (
        RB87_MASS_KG
        * np.var(velocities, axis=0, ddof=1)
        / BOLTZMANN_CONSTANT_J_PER_K
    )


def _snapshot_temperature_components_k(velocities_m_per_s: np.ndarray) -> np.ndarray:
    velocities = np.asarray(velocities_m_per_s, dtype=float)
    if velocities.ndim != 2 or velocities.shape[0] < 2 or velocities.shape[1] != 3:
        raise ValueError("velocities must have shape (N_atoms, 3), N_atoms >= 2")
    return (
        RB87_MASS_KG
        * np.var(velocities, axis=0, ddof=1)
        / BOLTZMANN_CONSTANT_J_PER_K
    )


def ensemble_temperature_metrics(
    velocity_history_m_per_s: np.ndarray,
    times_s: np.ndarray,
    *,
    plateau_window_s: float = DEFAULT_PLATEAU_WINDOW_S,
    stationarity_limit: float = DEFAULT_STATIONARITY_LIMIT,
) -> dict[str, object]:
    """Estimate one cloud's final plateau temperature and stationarity.

    At every time, the center-of-mass velocity is removed by the unbiased
    sample variance.  The reported value is then averaged over atoms,
    Cartesian components, and the final plateau interval.
    """

    velocities = np.asarray(velocity_history_m_per_s, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if velocities.ndim != 3 or velocities.shape[0] < 2 or velocities.shape[2] != 3:
        raise ValueError(
            "velocity history must have shape (N_atoms, N_times, 3), N_atoms >= 2"
        )
    if times.ndim != 1 or velocities.shape[1] != len(times) or len(times) < 4:
        raise ValueError("times must match at least four velocity-history samples")
    if plateau_window_s <= 0.0 or stationarity_limit <= 0.0:
        raise ValueError("plateau window and stationarity limit must be positive")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")

    plateau_start_s = max(float(times[0]), float(times[-1] - plateau_window_s))
    plateau_indices = np.flatnonzero(times >= plateau_start_s - 1.0e-15)
    if len(plateau_indices) < 4:
        raise ValueError("plateau window must include at least four recorded samples")

    component_history = _temperature_history_components_k(velocities)
    scalar_history = np.mean(component_history, axis=1)
    plateau_components = np.mean(component_history[plateau_indices], axis=0)
    plateau_temperature = float(np.mean(plateau_components))
    final_components = component_history[-1]
    final_temperature = float(np.mean(final_components))
    window_groups = np.array_split(plateau_indices, 4)
    window_means = np.asarray(
        [float(np.mean(scalar_history[group])) for group in window_groups],
        dtype=float,
    )
    denominator = max(abs(plateau_temperature), np.finfo(float).tiny)
    stationarity_metric = float(
        (np.max(window_means) - np.min(window_means)) / denominator
    )
    plateau_times = times[plateau_indices]
    centered_times = plateau_times - np.mean(plateau_times)
    slope_k_per_s = float(
        np.polyfit(centered_times, scalar_history[plateau_indices], 1)[0]
    )
    relative_drift = float(
        slope_k_per_s * (plateau_times[-1] - plateau_times[0]) / denominator
    )
    stationarity_pass = bool(
        stationarity_metric < stationarity_limit
        and abs(relative_drift) < stationarity_limit
    )
    positive_components = plateau_components[plateau_components > 0.0]
    anisotropy_ratio = (
        float(np.max(plateau_components) / np.min(positive_components))
        if len(positive_components) == 3
        else None
    )
    return {
        "plateau_temperature_components_k": plateau_components,
        "plateau_temperature_mean_k": plateau_temperature,
        "final_temperature_components_k": final_components,
        "final_temperature_mean_k": final_temperature,
        "plateau_four_window_means_k": window_means,
        "stationarity_metric": stationarity_metric,
        "relative_drift": relative_drift,
        "stationarity_pass": stationarity_pass,
        "anisotropy_ratio": anisotropy_ratio,
        "plateau_start_s": plateau_start_s,
        "plateau_window_s": float(times[-1] - plateau_start_s),
    }


def _quantile_or_none(values: Sequence[float], quantile: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if len(finite) else None


def _blank_ensemble_temperature_fields() -> dict[str, object]:
    return {
        name: None
        for name in (
            "initial_temperature_x_k",
            "initial_temperature_y_k",
            "initial_temperature_z_k",
            "initial_temperature_mean_k",
            "plateau_temperature_x_k",
            "plateau_temperature_y_k",
            "plateau_temperature_z_k",
            "plateau_temperature_mean_k",
            "final_temperature_x_k",
            "final_temperature_y_k",
            "final_temperature_z_k",
            "final_temperature_mean_k",
            "plateau_window_1_mean_k",
            "plateau_window_2_mean_k",
            "plateau_window_3_mean_k",
            "plateau_window_4_mean_k",
            "stationarity_metric",
            "relative_drift",
        )
    }


def _summarize_temperature_realization(
    *,
    point_index: int,
    detuning_n: float,
    ensemble_index: int,
    results: Sequence[TemperatureSweepWorkerResult],
    initial_velocities_m_per_s: np.ndarray,
    requested_atom_count: int,
    failed_atom_count: int,
    plateau_window_s: float,
    minimum_survivor_count: int,
    stationarity_limit: float,
) -> dict[str, object]:
    """Reduce 25 trajectories to one statistically independent cloud estimate."""

    complete_count = sum(result.complete for result in results)
    trapped_results = [result for result in results if result.complete and result.trapped]
    complete_outside_count = sum(
        result.complete and not result.trapped for result in results
    )
    escaped_count = sum(result.termination_reason == "escaped" for result in results)
    other_incomplete_count = sum(
        not result.complete and result.termination_reason != "escaped"
        for result in results
    )
    termination_counts = Counter(result.termination_reason for result in results)
    boundedness_counts = Counter(result.boundedness_reason for result in results)
    if failed_atom_count:
        termination_counts["worker_error"] += failed_atom_count
        boundedness_counts["worker_error"] += failed_atom_count

    aligned: list[TemperatureSweepWorkerResult] = []
    if trapped_results:
        reference_times = trapped_results[0].times_s
        aligned = [
            result
            for result in trapped_results
            if len(result.times_s) == len(reference_times)
            and np.allclose(result.times_s, reference_times, rtol=0.0, atol=1.0e-12)
        ]
    plateau_max_radii = [
        result.plateau_max_radius_m
        for result in results
        if result.complete and np.isfinite(result.plateau_max_radius_m)
    ]
    statistics_pass = len(aligned) >= minimum_survivor_count
    survivor_conditioning_warning = len(aligned) < requested_atom_count
    row: dict[str, object] = {
        "point_index": point_index,
        "detuning_n": float(detuning_n),
        "ensemble_index": ensemble_index,
        "requested_atom_count": requested_atom_count,
        "successful_atom_count": len(results),
        "complete_atom_count": complete_count,
        "trapped_atom_count": len(trapped_results),
        "aligned_trapped_atom_count": len(aligned),
        "complete_outside_core_count": complete_outside_count,
        "escaped_atom_count": escaped_count,
        "other_incomplete_atom_count": other_incomplete_count,
        "failed_atom_count": failed_atom_count,
        "duration_complete_fraction": complete_count / requested_atom_count,
        "trapped_fraction": len(trapped_results) / requested_atom_count,
        "stationarity_pass": False,
        "statistics_pass": statistics_pass,
        "survivor_conditioning_warning": survivor_conditioning_warning,
        "valid": False,
        "quality_status": (
            "worker_errors" if failed_atom_count else "insufficient_survivors"
        ),
        "plateau_max_radius_median_m": _quantile_or_none(plateau_max_radii, 0.5),
        "plateau_max_radius_p90_m": _quantile_or_none(plateau_max_radii, 0.9),
        "plateau_max_radius_max_m": _quantile_or_none(plateau_max_radii, 1.0),
        "termination_counts_json": json.dumps(dict(sorted(termination_counts.items()))),
        "boundedness_counts_json": json.dumps(dict(sorted(boundedness_counts.items()))),
        **_blank_ensemble_temperature_fields(),
    }
    if len(aligned) < 2:
        return row

    velocity_history = np.stack([result.velocities_m_per_s for result in aligned])
    survivor_indices = np.asarray([result.atom_index for result in aligned], dtype=int)
    initial_components = _snapshot_temperature_components_k(
        np.asarray(initial_velocities_m_per_s)[survivor_indices]
    )
    metrics = ensemble_temperature_metrics(
        velocity_history,
        aligned[0].times_s,
        plateau_window_s=plateau_window_s,
        stationarity_limit=stationarity_limit,
    )
    plateau_components = np.asarray(metrics["plateau_temperature_components_k"])
    final_components = np.asarray(metrics["final_temperature_components_k"])
    window_means = np.asarray(metrics["plateau_four_window_means_k"])
    stationarity_pass = bool(metrics["stationarity_pass"])
    valid = bool(failed_atom_count == 0 and statistics_pass and stationarity_pass)
    if failed_atom_count:
        quality_status = "worker_errors"
    elif not statistics_pass:
        quality_status = "insufficient_survivors"
    elif not stationarity_pass:
        quality_status = "nonstationary"
    elif survivor_conditioning_warning:
        quality_status = "valid_survivor_conditioned"
    else:
        quality_status = "valid"
    row.update(
        {
            "initial_temperature_x_k": float(initial_components[0]),
            "initial_temperature_y_k": float(initial_components[1]),
            "initial_temperature_z_k": float(initial_components[2]),
            "initial_temperature_mean_k": float(np.mean(initial_components)),
            "plateau_temperature_x_k": float(plateau_components[0]),
            "plateau_temperature_y_k": float(plateau_components[1]),
            "plateau_temperature_z_k": float(plateau_components[2]),
            "plateau_temperature_mean_k": float(metrics["plateau_temperature_mean_k"]),
            "final_temperature_x_k": float(final_components[0]),
            "final_temperature_y_k": float(final_components[1]),
            "final_temperature_z_k": float(final_components[2]),
            "final_temperature_mean_k": float(metrics["final_temperature_mean_k"]),
            "plateau_window_1_mean_k": float(window_means[0]),
            "plateau_window_2_mean_k": float(window_means[1]),
            "plateau_window_3_mean_k": float(window_means[2]),
            "plateau_window_4_mean_k": float(window_means[3]),
            "stationarity_metric": float(metrics["stationarity_metric"]),
            "relative_drift": float(metrics["relative_drift"]),
            "stationarity_pass": stationarity_pass,
            "valid": valid,
            "quality_status": quality_status,
        }
    )
    return row


def _mean_sem_t_interval(
    values: Sequence[float],
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[int, float | None, float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0, None, None, None, None
    mean = float(np.mean(finite))
    if len(finite) == 1:
        return 1, mean, None, mean, mean
    sem = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
    critical = float(student_t.ppf(0.975, df=len(finite) - 1))
    low = mean - critical * sem
    high = mean + critical * sem
    if bounds is not None:
        low = max(bounds[0], low)
        high = min(bounds[1], high)
    return len(finite), mean, sem, float(low), float(high)


def _wilson_interval(success_count: int, trial_count: int) -> tuple[float, float]:
    if trial_count <= 0 or not 0 <= success_count <= trial_count:
        raise ValueError("Wilson interval counts are invalid")
    z = 1.959963984540054
    proportion = success_count / trial_count
    denominator = 1.0 + z**2 / trial_count
    center = (proportion + z**2 / (2.0 * trial_count)) / denominator
    half_width = (
        z
        / denominator
        * np.sqrt(
            proportion * (1.0 - proportion) / trial_count
            + z**2 / (4.0 * trial_count**2)
        )
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _counter_from_rows(rows: Sequence[Mapping[str, object]], field: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for row in rows:
        raw = row.get(field, "{}")
        payload = json.loads(str(raw)) if raw else {}
        total.update({str(key): int(value) for key, value in payload.items()})
    return total


def _optional_mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "") and np.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else None


def _summarize_temperature_point(
    *,
    point_index: int,
    detuning_n: float,
    ensemble_rows: Sequence[Mapping[str, object]],
    requested_ensemble_count: int,
    atoms_per_ensemble: int,
    point_wall_time_s: float,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, object]:
    """Average the independently sampled cloud temperatures."""

    gamma = default_multilevel_mot_config().natural_linewidth_rad_per_s
    doppler_reference = cooling_doppler_reference(
        detuning_n,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    requested_atom_count = requested_ensemble_count * atoms_per_ensemble
    temperatures = [
        float(row["plateau_temperature_mean_k"])
        for row in ensemble_rows
        if row.get("plateau_temperature_mean_k") not in (None, "")
    ]
    temperature_count, temperature_mean, temperature_sem, temperature_low, temperature_high = (
        _mean_sem_t_interval(temperatures)
    )
    final_temperatures = [
        float(row["final_temperature_mean_k"])
        for row in ensemble_rows
        if row.get("final_temperature_mean_k") not in (None, "")
    ]
    _, final_mean, final_sem, _, _ = _mean_sem_t_interval(final_temperatures)
    trapped_fractions = [float(row["trapped_fraction"]) for row in ensemble_rows]
    _, trapped_mean, trapped_sem, trapped_low, trapped_high = _mean_sem_t_interval(
        trapped_fractions,
        bounds=(0.0, 1.0),
    )
    counts = {
        field: sum(int(row[field]) for row in ensemble_rows)
        for field in (
            "complete_atom_count",
            "trapped_atom_count",
            "complete_outside_core_count",
            "escaped_atom_count",
            "other_incomplete_atom_count",
            "failed_atom_count",
        )
    }
    wilson_low, wilson_high = _wilson_interval(
        counts["trapped_atom_count"], requested_atom_count
    )
    survivor_counts = [int(row["trapped_atom_count"]) for row in ensemble_rows]
    valid_temperature_count = sum(bool(row["valid"]) for row in ensemble_rows)
    stationarity_pass_count = sum(bool(row["stationarity_pass"]) for row in ensemble_rows)
    statistics_pass_count = sum(bool(row["statistics_pass"]) for row in ensemble_rows)
    all_requested_ensembles_present = len(ensemble_rows) == requested_ensemble_count
    valid = bool(
        all_requested_ensembles_present
        and temperature_count == requested_ensemble_count
        and valid_temperature_count == requested_ensemble_count
        and counts["failed_atom_count"] == 0
    )
    survivor_conditioned = counts["trapped_atom_count"] < requested_atom_count
    if counts["failed_atom_count"]:
        quality_status = "worker_errors"
    elif not all_requested_ensembles_present or temperature_count < requested_ensemble_count:
        quality_status = "incomplete_ensemble_statistics"
    elif statistics_pass_count < requested_ensemble_count:
        quality_status = "insufficient_survivors"
    elif stationarity_pass_count < requested_ensemble_count:
        quality_status = "nonstationary"
    elif survivor_conditioned:
        quality_status = "valid_survivor_conditioned"
    else:
        quality_status = "valid"
    plateau_radii = [
        float(row["plateau_max_radius_median_m"])
        for row in ensemble_rows
        if row.get("plateau_max_radius_median_m") not in (None, "")
    ]
    plateau_p90 = [
        float(row["plateau_max_radius_p90_m"])
        for row in ensemble_rows
        if row.get("plateau_max_radius_p90_m") not in (None, "")
    ]
    plateau_max = [
        float(row["plateau_max_radius_max_m"])
        for row in ensemble_rows
        if row.get("plateau_max_radius_max_m") not in (None, "")
    ]
    doppler = doppler_reference["doppler_temperature_k"]
    return {
        "point_index": point_index,
        "detuning_n": float(detuning_n),
        "detuning_rad_per_s": float(detuning_n * gamma),
        "detuning_hz": float(detuning_n * gamma / (2.0 * np.pi)),
        "detuning_mhz": float(detuning_n * gamma / (2.0 * np.pi * 1.0e6)),
        "cooling_power_w_per_beam": float(cooling_power_w_per_beam),
        "requested_ensemble_count": requested_ensemble_count,
        "temperature_ensemble_count": temperature_count,
        "valid_temperature_ensemble_count": valid_temperature_count,
        "requested_atom_count": requested_atom_count,
        **counts,
        "duration_complete_fraction": counts["complete_atom_count"] / requested_atom_count,
        "trapped_fraction": trapped_mean,
        "trapped_fraction_sem": trapped_sem,
        "trapped_fraction_ci_low": trapped_low,
        "trapped_fraction_ci_high": trapped_high,
        "trapped_fraction_pooled_wilson_low": wilson_low,
        "trapped_fraction_pooled_wilson_high": wilson_high,
        "minimum_survivors_per_ensemble": min(survivor_counts) if survivor_counts else None,
        "median_survivors_per_ensemble": (
            float(np.median(survivor_counts)) if survivor_counts else None
        ),
        "maximum_survivors_per_ensemble": max(survivor_counts) if survivor_counts else None,
        "initial_temperature_mean_k": _optional_mean(
            ensemble_rows, "initial_temperature_mean_k"
        ),
        "plateau_temperature_x_k": _optional_mean(
            ensemble_rows, "plateau_temperature_x_k"
        ),
        "plateau_temperature_y_k": _optional_mean(
            ensemble_rows, "plateau_temperature_y_k"
        ),
        "plateau_temperature_z_k": _optional_mean(
            ensemble_rows, "plateau_temperature_z_k"
        ),
        "plateau_temperature_mean_k": temperature_mean,
        "temperature_sem_k": temperature_sem,
        "temperature_ci_low_k": temperature_low,
        "temperature_ci_high_k": temperature_high,
        "final_temperature_mean_k": final_mean,
        "final_temperature_sem_k": final_sem,
        "cooling_beam_center_on_resonance_saturation_parameter": (
            doppler_reference[
                "cooling_beam_center_on_resonance_saturation_parameter"
            ]
        ),
        "cooling_beam_center_effective_saturation_parameter": (
            doppler_reference[
                "cooling_beam_center_effective_saturation_parameter"
            ]
        ),
        "doppler_temperature_k": doppler,
        "plateau_over_doppler": (
            temperature_mean / doppler if temperature_mean is not None else None
        ),
        "stationarity_pass_count": stationarity_pass_count,
        "statistics_pass_count": statistics_pass_count,
        "plateau_max_radius_median_m": _quantile_or_none(plateau_radii, 0.5),
        "plateau_max_radius_p90_m": _quantile_or_none(plateau_p90, 0.9),
        "plateau_max_radius_max_m": _quantile_or_none(plateau_max, 1.0),
        "valid": valid,
        "quality_status": quality_status,
        "termination_counts_json": json.dumps(
            dict(sorted(_counter_from_rows(ensemble_rows, "termination_counts_json").items()))
        ),
        "boundedness_counts_json": json.dumps(
            dict(sorted(_counter_from_rows(ensemble_rows, "boundedness_counts_json").items()))
        ),
        "point_wall_time_s": point_wall_time_s,
    }


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _refresh_doppler_reference_fields(
    row: Mapping[str, object],
    *,
    cooling_power_w_per_beam: float | None = None,
) -> dict[str, object]:
    """Recompute analytical Doppler fields without rerunning trajectories."""

    refreshed = dict(row)
    power = (
        float(cooling_power_w_per_beam)
        if cooling_power_w_per_beam is not None
        else float(
            refreshed.get(
                "cooling_power_w_per_beam",
                DEFAULT_COOLING_POWER_W_PER_BEAM,
            )
            or DEFAULT_COOLING_POWER_W_PER_BEAM
        )
    )
    reference = cooling_doppler_reference(
        float(refreshed["detuning_n"]),
        cooling_power_w_per_beam=power,
    )
    doppler = reference["doppler_temperature_k"]
    temperature = _float_or_none(refreshed.get("plateau_temperature_mean_k"))
    refreshed.update(
        {
            "cooling_power_w_per_beam": power,
            "cooling_beam_center_on_resonance_saturation_parameter": reference[
                "cooling_beam_center_on_resonance_saturation_parameter"
            ],
            "cooling_beam_center_effective_saturation_parameter": reference[
                "cooling_beam_center_effective_saturation_parameter"
            ],
            "doppler_temperature_k": doppler,
            "plateau_over_doppler": (
                None if temperature is None else temperature / doppler
            ),
        }
    )
    return refreshed


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.05 * (attempt + 1))


def _write_csv_checkpoint(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] = SUMMARY_CSV_FIELDNAMES,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    if "ensemble_index" in fieldnames:
        row_list.sort(
            key=lambda row: (int(row["point_index"]), int(row["ensemble_index"]))
        )
    else:
        row_list.sort(key=lambda row: int(row["point_index"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow({field: row.get(field) for field in fieldnames})
    _replace_with_retry(temporary, path)


def _read_csv_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    _replace_with_retry(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )


def _resume_signature(
    *,
    ensemble_realization_count: int,
    atoms_per_ensemble: int,
    duration_s: float,
    time_step_s: float,
    initial_temperature_k: float,
    initial_position_sigma_m: float,
    seed: int,
    plateau_window_s: float,
    record_interval_s: float,
    minimum_survivor_count: int,
    stationarity_limit: float,
    configuration_audit: Mapping[str, object],
    detuning_n_values: Sequence[float] = DETUNING_N_VALUES,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, object]:
    values = detuning_n_grid(detuning_n_values)
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    apparatus_base = default_mot_apparatus_config()
    apparatus = replace(
        apparatus_base,
        cooling=replace(
            apparatus_base.cooling,
            power_w_per_beam=float(cooling_power_w_per_beam),
        ),
        repump=replace(
            apparatus_base.repump,
            power_w_per_beam=float(config.repump_power_w_per_beam),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "doppler_reference_version": DOPPLER_REFERENCE_VERSION,
        "solver": "24_state_repumper_adiabatic_population_rate_equation_langevin",
        "physical_model_statement": _physical_model_statement(
            ensemble_realization_count,
            atoms_per_ensemble,
            plateau_window_s,
        ),
        "detuning_n_values": values.tolist(),
        "cooling_power_w_per_beam": float(cooling_power_w_per_beam),
        "ensemble_realization_count": ensemble_realization_count,
        "atoms_per_ensemble": atoms_per_ensemble,
        "trajectory_count_per_point": ensemble_realization_count * atoms_per_ensemble,
        "duration_s": duration_s,
        "time_step_s": time_step_s,
        "initial_temperature_k": initial_temperature_k,
        "initial_position_sigma_m": initial_position_sigma_m,
        "seed": seed,
        "plateau_window_s": plateau_window_s,
        "record_interval_s": record_interval_s,
        "minimum_survivor_count_per_ensemble": minimum_survivor_count,
        "stationarity_limit": stationarity_limit,
        "temperature_core_radius_m": TRAPPED_CORE_RADIUS_M,
        "escape_radius_m": ESCAPE_RADIUS_M,
        "configuration_invariance_audit": dict(configuration_audit),
        "multilevel_config": _json_compatible(asdict(config)),
        "apparatus_config": _json_compatible(asdict(apparatus)),
        "coil_config": _json_compatible(asdict(default_anti_helmholtz_config())),
    }


def _resume_signature_is_compatible(
    prior_signature: Mapping[str, object],
    current_signature: Mapping[str, object],
) -> bool:
    """Allow analytical upgrades and resume legacy default-power checkpoints."""

    prior = dict(prior_signature)
    current = dict(current_signature)
    prior_reference_version = int(prior.pop("doppler_reference_version", 1))
    current_reference_version = int(current.pop("doppler_reference_version"))
    if "cooling_power_w_per_beam" not in prior:
        prior["cooling_power_w_per_beam"] = DEFAULT_COOLING_POWER_W_PER_BEAM
    prior_audit = prior.get("configuration_invariance_audit")
    current_audit = current.get("configuration_invariance_audit")
    if isinstance(prior_audit, Mapping) and isinstance(current_audit, Mapping):
        prior_audit = dict(prior_audit)
        if "cooling_power_w_per_beam" not in prior_audit:
            prior_audit["cooling_power_w_per_beam"] = (
                DEFAULT_COOLING_POWER_W_PER_BEAM
            )
        prior["configuration_invariance_audit"] = prior_audit
    prior_apparatus = prior.get("apparatus_config")
    current_apparatus = current.get("apparatus_config")
    if isinstance(prior_apparatus, Mapping) and isinstance(current_apparatus, Mapping):
        prior_apparatus = dict(prior_apparatus)
        prior_repump = prior_apparatus.get("repump")
        current_repump = current_apparatus.get("repump")
        if isinstance(prior_repump, Mapping) and isinstance(current_repump, Mapping):
            prior_repump = dict(prior_repump)
            old_generic_power = default_mot_apparatus_config().repump.power_w_per_beam
            simulated_power = default_multilevel_mot_config().repump_power_w_per_beam
            if (
                np.isclose(
                    float(prior_repump.get("power_w_per_beam", np.nan)),
                    old_generic_power,
                    rtol=0.0,
                    atol=0.0,
                )
                and np.isclose(
                    float(current_repump.get("power_w_per_beam", np.nan)),
                    simulated_power,
                    rtol=0.0,
                    atol=0.0,
                )
            ):
                prior_repump["power_w_per_beam"] = simulated_power
                prior_apparatus["repump"] = prior_repump
                prior["apparatus_config"] = prior_apparatus
                if isinstance(prior_audit, dict) and isinstance(current_audit, Mapping):
                    prior_audit["fixed_configuration_sha256"] = current_audit.get(
                        "fixed_configuration_sha256"
                    )
                    prior["configuration_invariance_audit"] = prior_audit
    return (
        prior == current
        and 1 <= prior_reference_version <= current_reference_version
    )


def physical_model_markdown(signature: Mapping[str, object]) -> str:
    """Return the human-readable scientific definition saved with the data."""

    detuning_values = detuning_n_grid(signature["detuning_n_values"])
    ensemble_count = int(signature["ensemble_realization_count"])
    atoms_per_ensemble = int(signature["atoms_per_ensemble"])
    trajectory_count = int(signature["trajectory_count_per_point"])
    cooling_power_w_per_beam = float(
        signature.get(
            "cooling_power_w_per_beam",
            DEFAULT_COOLING_POWER_W_PER_BEAM,
        )
    )
    config, apparatus, _ = build_temperature_sweep_configuration(
        float(detuning_values[0]),
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    doppler_references = [
        cooling_doppler_reference(
            value,
            cooling_power_w_per_beam=cooling_power_w_per_beam,
        )
        for value in detuning_values
    ]
    doppler_temperatures = [
        reference["doppler_temperature_k"] for reference in doppler_references
    ]
    on_resonance_saturation = doppler_references[0][
        "cooling_beam_center_on_resonance_saturation_parameter"
    ]
    physical_statement = str(
        signature.get(
            "physical_model_statement",
            _physical_model_statement(
                ensemble_count,
                atoms_per_ensemble,
                float(signature["plateau_window_s"]),
            ),
        )
    )
    return f"""# Physical model: temperature versus cooling detuning

## What was modeled

{physical_statement}

This is a trapped-cloud equilibration/temperature study, not a capture/loading
study. No incident launch disk is used: a disk would require choosing an
incident launch speed and would therefore add another physical variable.
Consequently, full-sphere versus octant launch-direction sampling does not apply.
Instead, each of the {ensemble_count} independent realizations is a randomly
sampled {atoms_per_ensemble}-atom preloaded cloud. A plotted estimate is
interpreted as an equilibrium temperature only when it passes the survivor and
stationarity checks below.

## Fixed physical and numerical parameters

- Atomic/optical model: repumper-enabled 24-state Rb-87 D2 adiabatic population-rate equations (8 ground states and 16 excited states).
- Cooling beams: six default Gaussian beams, {1e3 * apparatus.cooling.power_w_per_beam:.6g} mW per beam and {1e3 * apparatus.cooling.beam_diameter_m:.6g} mm diameter.
- Repumper: enabled at {1e3 * config.repump_power_w_per_beam:.6g} mW per beam with the fixed baseline diameter and detuning; all dipole-allowed F=1 channels including F'=0 are retained.
- Magnetic field, gravity, beam geometry, polarizations, linewidth, and every configuration value other than cooling detuning: fixed at the production baseline recorded in the metadata. The repumper is explicitly enabled for this baseline even though the generic configuration factory defaults to disabled.
- Initial cloud: independent normal position draws with sigma={1e3 * float(signature['initial_position_sigma_m']):.6g} mm per coordinate and Maxwell-Gaussian velocity draws at {1e3 * float(signature['initial_temperature_k']):.6g} mK.
- Evolution: {1e3 * float(signature['duration_s']):.6g} ms with {1e6 * float(signature['time_step_s']):.6g} microsecond steps and Langevin recoil diffusion enabled.
- Final plateau: last {1e3 * float(signature['plateau_window_s']):.6g} ms; a temperature survivor reaches the requested duration and stays within the 2 mm core throughout this final interval.
- Detuning grid: {len(detuning_values)} specified values spanning n={min(detuning_values):.6g} to n={max(detuning_values):.6g}, with Delta=n Gamma and Gamma/(2 pi)={config.natural_linewidth_rad_per_s / (2*np.pi*1e6):.6g} MHz.
- Detuning-dependent Doppler reference: T_D=-hbar Gamma^2/[8 k_B Delta] * [1+s_eff+(2 Delta/Gamma)^2], evaluated independently at every detuning using angular-frequency units and red Delta<0.
- Saturation convention: s_eff=s_0/[1+(2 Delta/Gamma)^2], where s_0=I_0/I_sat and I_0=2P/(pi w^2) is the Gaussian peak intensity at the center of one cooling beam. The fixed {1e3 * cooling_power_w_per_beam:.6g} mW, {1e3 * apparatus.cooling.beam_diameter_m:.6g} mm-diameter cooling beams give s_0={on_resonance_saturation:.8g}.
- Across the saved detuning grid, this reference spans {1e6*min(doppler_temperatures):.6g} to {1e6*max(doppler_temperatures):.6g} microkelvin and is plotted as a curve, not a constant line.

The {ensemble_count} random initial phase-space clouds are drawn once and reused at every
detuning (common random initial conditions). Langevin recoil streams are
independent between atoms, realizations, and detunings. A configuration audit
verifies that only the solver, apparatus, and cooling-beam detuning fields vary.

## Temperature and uncertainty estimator

For each realization and recorded time, the instantaneous center-of-mass
velocity is removed and the unbiased sample variance is calculated separately
for x, y, and z: T_i=m Var(v_i)/k_B. The scalar temperature is
(T_x+T_y+T_z)/3 and is averaged over the final plateau. The plotted point is
the arithmetic mean of the {ensemble_count} realization temperatures. Explicit
asymmetric error bars and the shaded band show the two-sided 95% Student-t
interval across the {ensemble_count} independent cloud realizations; one SEM is
also saved in the CSV.

For a cloud estimate to pass, at least
{int(signature['minimum_survivor_count_per_ensemble'])} atoms must survive in
the final core and both its four-subwindow spread and fitted relative drift over
the plateau must be below {float(signature['stationarity_limit']):.3g}. A plot
point is marked valid only if all {ensemble_count} cloud estimates pass. Hollow red circles
are nonstationary diagnostics and the hollow orange triangle marks an
insufficient-survivor diagnostic; neither is claimed as an equilibrium
temperature.

The survivor fraction is reported independently from temperature. Its primary
uncertainty interval is a 95% Student-t interval across the {ensemble_count} realization
fractions, which preserves the ensemble clustering. A pooled 95% Wilson
interval across all {trajectory_count} trajectories is also saved as a secondary diagnostic.

## Why an earlier trapped fraction could equal one

These atoms start as a compact preloaded cloud (position sigma 0.25 mm, well
inside the 2 mm core), rather than arriving from a capture disk. With only ten
previous trajectories, all ten could readily survive for the short simulated
duration, producing a displayed fraction of 1. That value described survival
of that small preloaded sample; it did not mean every incident atom would be
captured. This run uses {trajectory_count} trajectories per detuning and reports an interval.

## Limitations

- A survivor-only temperature is conditional on remaining in the final core and can exhibit survivor bias; survivor fraction is therefore shown separately.
- The adiabatic rate equations omit optical coherences and sub-Doppler polarization-gradient cooling.
- Atoms are noninteracting; density-dependent reabsorption and collisions are absent.
- The detuning-dependent Doppler curve is an analytical reference using the explicitly defined per-beam s_eff; it is not a claim that the full 24-state magnetic MOT must attain it.
- The {1e3 * float(signature['duration_s']):.6g} ms evolution is finite; quality-flagged values diagnose where this duration or the final-core sample does not establish equilibrium.
- Quantitative claims remain provisional until timestep and duration convergence are checked independently.
"""


def _metadata_payload(
    *,
    signature: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
    status: str,
    worker_count: int,
    resume: bool,
    resumed_point_count: int,
    created_utc: str,
    wall_time_s: float,
    summary_csv_path: Path,
    ensemble_csv_path: Path,
    plot_path: Path,
    temperature_only_plot_path: Path,
    metadata_path: Path,
    physical_model_path: Path,
) -> dict[str, object]:
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    detuning_values = detuning_n_grid(signature["detuning_n_values"])
    ensemble_count = int(signature["ensemble_realization_count"])
    atoms_per_ensemble = int(signature["atoms_per_ensemble"])
    trajectory_count = int(signature["trajectory_count_per_point"])
    cooling_power_w_per_beam = float(
        signature.get(
            "cooling_power_w_per_beam",
            DEFAULT_COOLING_POWER_W_PER_BEAM,
        )
    )
    physical_statement = str(signature["physical_model_statement"])
    doppler_reference_points = []
    for detuning_n in detuning_values:
        reference = cooling_doppler_reference(
            detuning_n,
            cooling_power_w_per_beam=cooling_power_w_per_beam,
        )
        doppler_reference_points.append(
            {
                "detuning_n": float(detuning_n),
                "detuning_rad_per_s": reference["detuning_rad_per_s"],
                "cooling_beam_center_on_resonance_saturation_parameter": reference[
                    "cooling_beam_center_on_resonance_saturation_parameter"
                ],
                "cooling_beam_center_effective_saturation_parameter": reference[
                    "cooling_beam_center_effective_saturation_parameter"
                ],
                "doppler_temperature_k": reference["doppler_temperature_k"],
            }
        )
    on_resonance_saturation = doppler_reference_points[0][
        "cooling_beam_center_on_resonance_saturation_parameter"
    ]
    doppler_temperatures = [
        point["doppler_temperature_k"] for point in doppler_reference_points
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "doppler_reference_version": DOPPLER_REFERENCE_VERSION,
        "status": status,
        "created_utc": created_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "resume_enabled": resume,
        "resumed_point_count": resumed_point_count,
        "completed_point_count": len(rows),
        "completed_ensemble_row_count": len(ensemble_rows),
        "total_point_count": len(detuning_values),
        "ensemble_realization_count_per_point": ensemble_count,
        "atoms_per_ensemble": atoms_per_ensemble,
        "trajectory_count_per_point": trajectory_count,
        "cooling_power_w_per_beam": cooling_power_w_per_beam,
        "wall_time_s_current_invocation": wall_time_s,
        "worker_count_current_invocation": worker_count,
        "resume_signature": dict(signature),
        "physical_model_statement": physical_statement,
        "study_design_rationale": (
            f"Temperature is estimated from {ensemble_count} independently sampled "
            f"preloaded {atoms_per_ensemble}-atom clouds. Incident capture disks are "
            "not used because their required launch speed would add a second varied "
            "physical input; full-sphere versus octant launch sampling is therefore "
            "not applicable."
        ),
        "temperature_definition": (
            "For every time and realization: T_i=m*s_i^2/k_B where s_i^2 is the "
            "unbiased sample variance about that realization's instantaneous COM "
            "velocity; average x/y/z and then average the final plateau in time."
        ),
        "reported_point_definition": (
            f"arithmetic mean of the {ensemble_count} independently calculated "
            "realization plateau temperatures"
        ),
        "equilibrium_quality_definition": (
            f"valid only when all {ensemble_count} clouds meet the minimum final-core survivor "
            "count and both stationarity metrics; quality-flagged values are saved "
            "and plotted as diagnostics, not claimed equilibrium temperatures"
        ),
        "point_quality_status_counts": dict(
            sorted(Counter(str(row.get("quality_status", "")) for row in rows).items())
        ),
        "temperature_uncertainty_definition": (
            f"one SEM and a two-sided 95% Student-t interval across the {ensemble_count} "
            "independent ensemble-realization temperatures"
        ),
        "temperature_plot_uncertainty_definition": (
            "explicit asymmetric error bars and shading show the two-sided 95% "
            "Student-t interval across independent cloud realizations; clouds, not "
            "individual atoms, are the statistical clusters"
        ),
        "survivor_fraction_uncertainty_definition": (
            f"primary: 95% Student-t interval across {ensemble_count} "
            f"{atoms_per_ensemble}-atom realization fractions; secondary: pooled "
            f"two-sided 95% Wilson score interval over {trajectory_count} trajectories"
        ),
        "temperature_survivor_definition": (
            "reaches requested duration, remains within 2 mm throughout the final "
            "temperature plateau, and has the common time grid needed for an ensemble "
            "variance; this is a temperature-analysis survivor definition, not a "
            "replacement for the hybrid loading/capture criterion"
        ),
        "only_varied_physical_parameter": "cooling detuning Delta=n Gamma",
        "configuration_invariance_audit": signature["configuration_invariance_audit"],
        "common_initial_condition_strategy": (
            f"{ensemble_count} independent initial phase-space clouds drawn once and reused exactly at "
            "every detuning"
        ),
        "recoil_seed_strategy": (
            "SeedSequence([seed,1,point_index,ensemble_index,atom_index]); streams are "
            "independent across atoms, realizations, and detunings"
        ),
        "detuning_grid_rationale": (
            "established 25-point grid: one-linewidth spacing on the far-red wing, "
            "finer spacing through the likely cooling minimum, and the finest spacing "
            "near resonance"
            if tuple(float(value) for value in detuning_values) == DETUNING_N_VALUES
            else "explicit user-supplied red-detuning grid; input order is retained"
        ),
        "doppler_equation": (
            "T_D=-hbar Gamma^2/[8 k_B Delta] "
            "* [1+s_eff+(2 Delta/Gamma)^2]"
        ),
        "effective_saturation_equation": (
            "s_eff=s_0/[1+(2 Delta/Gamma)^2]"
        ),
        "doppler_saturation_convention": (
            "s_0=I_0/I_sat for one cooling beam at its Gaussian center, "
            "with I_0=2P/(pi w^2); s_eff is recomputed at every detuning"
        ),
        "doppler_reference_is_detuning_dependent": True,
        "frequency_convention": "Gamma and Delta are angular frequencies in rad/s",
        "linewidth_rad_per_s": config.natural_linewidth_rad_per_s,
        "cooling_beam_center_on_resonance_saturation_parameter": (
            on_resonance_saturation
        ),
        "doppler_temperature_min_k": min(doppler_temperatures),
        "doppler_temperature_max_k": max(doppler_temperatures),
        "doppler_reference_points": doppler_reference_points,
        "diffusion_enabled": True,
        "repumper_enabled": True,
        "effective_repump_power_w_per_beam": config.repump_power_w_per_beam,
        "apparatus_config_repump_note": (
            "The apparatus, multilevel configuration, and built repump beams are "
            "synchronized to the simulated 0.1 mW per-beam repumper power. Older "
            "temperature checkpoints recorded the generic 0.5 mW apparatus default "
            "even though their built beams already used 0.1 mW."
        ),
        "model_state_count": len(model.structure.states),
        "ground_state_count": len(model.structure.ground_state_indices),
        "excited_state_count": len(model.structure.excited_state_indices),
        "limitations": [
            "Temperature is conditioned on final-core survival and can have survivor bias.",
            "The population-rate approximation omits optical coherences and sub-Doppler cooling.",
            "The model omits density-dependent collisions and photon reabsorption.",
            "The detuning-dependent Doppler curve is an analytical reference using the recorded per-beam s_eff convention, not a guaranteed full-MOT limit.",
            f"Quality-flagged values do not establish equilibrium within the {1e3 * float(signature['duration_s']):.6g} ms run.",
            "Timestep and duration convergence remain required for quantitative claims.",
        ],
        "outputs": {
            "temperature_summary_csv": str(summary_csv_path),
            "temperature_ensemble_csv": str(ensemble_csv_path),
            "temperature_plot": str(plot_path),
            "temperature_only_plot": str(temperature_only_plot_path),
            "metadata_json": str(metadata_path),
            "physical_model_markdown": str(physical_model_path),
        },
    }


def plot_temperature_vs_detuning(
    rows_or_csv: Sequence[Mapping[str, object]] | Path | str,
    output_path: Path | str,
    *,
    linewidth_rad_per_s: float | None = None,
    cooling_power_w_per_beam: float | None = None,
    ensemble_realization_count: int | None = None,
    atoms_per_ensemble: int | None = None,
    include_survivor_panel: bool = True,
) -> Path:
    """Plot cloud-mean temperature, optionally with the survivor-fraction panel."""

    if isinstance(rows_or_csv, (str, Path)):
        rows = _read_csv_checkpoint(Path(rows_or_csv))
    else:
        rows = list(rows_or_csv)
    if not rows:
        raise ValueError("at least one completed detuning row is required")
    recorded_powers = {
        float(row["cooling_power_w_per_beam"])
        for row in rows
        if _float_or_none(row.get("cooling_power_w_per_beam")) is not None
    }
    if cooling_power_w_per_beam is None:
        if len(recorded_powers) > 1:
            raise ValueError("temperature rows contain inconsistent cooling powers")
        cooling_power_w_per_beam = (
            next(iter(recorded_powers))
            if recorded_powers
            else DEFAULT_COOLING_POWER_W_PER_BEAM
        )
    if ensemble_realization_count is None:
        recorded_counts = {
            int(row["requested_ensemble_count"])
            for row in rows
            if row.get("requested_ensemble_count") not in (None, "")
        }
        ensemble_realization_count = (
            next(iter(recorded_counts))
            if len(recorded_counts) == 1
            else REQUIRED_ENSEMBLE_REALIZATION_COUNT
        )
    if atoms_per_ensemble is None:
        first_requested_atoms = _float_or_none(rows[0].get("requested_atom_count"))
        atoms_per_ensemble = (
            int(first_requested_atoms) // ensemble_realization_count
            if first_requested_atoms is not None
            else REQUIRED_ATOMS_PER_ENSEMBLE
        )
    # Production runs are non-interactive and may import another plotting
    # module before this one, so enforce a headless backend at render time.
    plt.switch_backend("Agg")
    rows.sort(key=lambda row: float(row["detuning_n"]))
    gamma = (
        linewidth_rad_per_s
        if linewidth_rad_per_s is not None
        else default_multilevel_mot_config().natural_linewidth_rad_per_s
    )
    n_values = np.asarray([float(row["detuning_n"]) for row in rows])
    doppler = np.asarray(
        [
            cooling_doppler_reference(
                detuning_n,
                linewidth_rad_per_s=gamma,
                cooling_power_w_per_beam=cooling_power_w_per_beam,
            )["doppler_temperature_k"]
            for detuning_n in n_values
        ]
    )
    temperatures = np.asarray(
        [
            np.nan
            if _float_or_none(row.get("plateau_temperature_mean_k")) is None
            else float(row["plateau_temperature_mean_k"])
            for row in rows
        ]
    )
    temperature_sem = np.asarray(
        [
            0.0
            if _float_or_none(row.get("temperature_sem_k")) is None
            else float(row["temperature_sem_k"])
            for row in rows
        ]
    )
    temperature_ci_low = np.asarray(
        [
            temperatures[index]
            if _float_or_none(row.get("temperature_ci_low_k")) is None
            else float(row["temperature_ci_low_k"])
            for index, row in enumerate(rows)
        ]
    )
    temperature_ci_high = np.asarray(
        [
            temperatures[index]
            if _float_or_none(row.get("temperature_ci_high_k")) is None
            else float(row["temperature_ci_high_k"])
            for index, row in enumerate(rows)
        ]
    )
    temperature_ci_yerr = np.vstack(
        (
            np.maximum(0.0, temperatures - temperature_ci_low),
            np.maximum(0.0, temperature_ci_high - temperatures),
        )
    )
    trapped_fraction = np.asarray([float(row["trapped_fraction"]) for row in rows])
    trapped_ci_low = np.asarray(
        [
            trapped_fraction[index]
            if _float_or_none(row.get("trapped_fraction_ci_low")) is None
            else float(row["trapped_fraction_ci_low"])
            for index, row in enumerate(rows)
        ]
    )
    trapped_ci_high = np.asarray(
        [
            trapped_fraction[index]
            if _float_or_none(row.get("trapped_fraction_ci_high")) is None
            else float(row["trapped_fraction_ci_high"])
            for index, row in enumerate(rows)
        ]
    )
    valid = np.asarray([_bool_value(row.get("valid", False)) for row in rows])
    quality_status = np.asarray([str(row.get("quality_status", "")) for row in rows])
    finite = np.isfinite(temperatures)

    if include_survivor_panel:
        figure, (temperature_axis, survivor_axis) = plt.subplots(
            2,
            1,
            figsize=(9.8, 7.8),
            sharex=True,
            gridspec_kw={"height_ratios": (3.0, 1.45)},
            constrained_layout=True,
        )
    else:
        figure, temperature_axis = plt.subplots(
            figsize=(9.8, 6.2),
            constrained_layout=True,
        )
        survivor_axis = None
    if np.any(finite):
        temperature_axis.fill_between(
            n_values[finite],
            1.0e6 * temperature_ci_low[finite],
            1.0e6 * temperature_ci_high[finite],
            color="#93c5fd",
            alpha=0.24,
            label="95% Student-t interval across clouds",
        )
        temperature_axis.plot(
            n_values[finite],
            1.0e6 * temperatures[finite],
            color="#334155",
            linewidth=1.2,
            linestyle=":",
            alpha=0.7,
        )
        accepted = finite & valid
        nonstationary = finite & (quality_status == "nonstationary")
        insufficient = finite & ~valid & ~nonstationary
        if np.any(accepted):
            temperature_axis.errorbar(
                n_values[accepted],
                1.0e6 * temperatures[accepted],
                yerr=1.0e6 * temperature_ci_yerr[:, accepted],
                fmt="o",
                color="#2563eb",
                capsize=3,
                label=(
                    f"mean of {ensemble_realization_count} clouds; "
                    "95% Student-t CI"
                ),
            )
        if np.any(nonstationary):
            temperature_axis.errorbar(
                n_values[nonstationary],
                1.0e6 * temperatures[nonstationary],
                yerr=1.0e6 * temperature_ci_yerr[:, nonstationary],
                fmt="o",
                markerfacecolor="white",
                markeredgecolor="#dc2626",
                ecolor="#dc2626",
                capsize=3,
                label="nonstationary estimate; 95% Student-t CI",
            )
        if np.any(insufficient):
            temperature_axis.errorbar(
                n_values[insufficient],
                1.0e6 * temperatures[insufficient],
                yerr=1.0e6 * temperature_ci_yerr[:, insufficient],
                fmt="^",
                markerfacecolor="white",
                markeredgecolor="#d97706",
                ecolor="#d97706",
                capsize=3,
                label="insufficient-survivor estimate; 95% Student-t CI",
            )
    temperature_axis.plot(
        n_values,
        1.0e6 * doppler,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.5,
        label=r"detuning-dependent Doppler reference $T_D(\Delta)$",
    )
    temperature_axis.text(
        0.34,
        0.97,
        rf"$\Delta=n\Gamma$; red detuning $n<0$; "
        rf"$\Gamma/(2\pi)={gamma/(2*np.pi*1e6):.2f}$ MHz"
        f"\n{ensemble_realization_count} independent preloaded clouds × "
        f"{atoms_per_ensemble} atoms; fixed {1e3 * cooling_power_w_per_beam:.6g} mW/beam"
        "\n"
        r"Doppler curve uses per-beam, detuning-reduced $s_{\mathrm{eff}}$",
        transform=temperature_axis.transAxes,
        ha="left",
        va="top",
    )
    temperature_axis.set(
        ylabel="final plateau temperature [µK]",
        title=(
            "24-state repumper-enabled multilevel MOT: ensemble temperature "
            "versus cooling detuning"
        ),
    )
    if not include_survivor_panel:
        temperature_axis.set_xlabel(
            r"detuning multiplier $n$ in $\Delta=n\Gamma$"
        )
    temperature_axis.grid(alpha=0.22)
    temperature_axis.legend(loc="best", fontsize=8.5)

    if survivor_axis is not None:
        survivor_axis.fill_between(
            n_values,
            trapped_ci_low,
            trapped_ci_high,
            color="#86efac",
            alpha=0.28,
            label="95% Student-t interval across clouds",
        )
        survivor_axis.plot(
            n_values,
            trapped_fraction,
            "o-",
            color="#047857",
            markersize=4,
            linewidth=1.4,
            label="final-core survivor fraction",
        )
        survivor_axis.set(
            xlabel=r"detuning multiplier $n$ in $\Delta=n\Gamma$",
            ylabel="survivor fraction",
            ylim=(-0.03, 1.03),
        )
        survivor_axis.grid(alpha=0.22)
        survivor_axis.legend(loc="best", fontsize=8.5)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)
    return destination


def _temperature_sweep_name(
    *,
    ensemble_realization_count: int,
    atoms_per_ensemble: int,
    cooling_power_w_per_beam: float,
    detuning_n_values: Sequence[float],
) -> str:
    values = tuple(float(value) for value in detuning_n_values)
    if (
        ensemble_realization_count == REQUIRED_ENSEMBLE_REALIZATION_COUNT
        and atoms_per_ensemble == REQUIRED_ATOMS_PER_ENSEMBLE
        and np.isclose(
            cooling_power_w_per_beam,
            DEFAULT_COOLING_POWER_W_PER_BEAM,
            rtol=0.0,
            atol=0.0,
        )
        and values == DETUNING_N_VALUES
    ):
        return "temperature_vs_detuning_10x25_preloaded_ensembles"
    power_label = format(1.0e3 * cooling_power_w_per_beam, ".6g").replace(".", "p")
    return (
        f"temperature_vs_detuning_{ensemble_realization_count}x{atoms_per_ensemble}_"
        f"{power_label}mW_{len(values)}points_preloaded_ensembles"
    )


def run_temperature_detuning_sweep(
    *,
    ensemble_realization_count: int = REQUIRED_ENSEMBLE_REALIZATION_COUNT,
    atoms_per_ensemble: int = REQUIRED_ATOMS_PER_ENSEMBLE,
    duration_s: float = DEFAULT_DURATION_S,
    time_step_s: float = DEFAULT_TIME_STEP_S,
    worker_count: int = DEFAULT_WORKER_COUNT,
    seed: int = DEFAULT_SEED,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
    initial_temperature_k: float = DEFAULT_INITIAL_TEMPERATURE_K,
    initial_position_sigma_m: float = DEFAULT_INITIAL_POSITION_SIGMA_M,
    plateau_window_s: float = DEFAULT_PLATEAU_WINDOW_S,
    record_interval_s: float = DEFAULT_RECORD_INTERVAL_S,
    minimum_survivor_count: int | None = None,
    stationarity_limit: float = DEFAULT_STATIONARITY_LIMIT,
    detuning_n_values: Sequence[float] = DETUNING_N_VALUES,
    cooling_power_w_per_beam: float = DEFAULT_COOLING_POWER_W_PER_BEAM,
) -> dict[str, object]:
    """Run, checkpoint, resume, and plot a preloaded-cloud temperature study."""

    detuning_values = tuple(float(value) for value in detuning_n_grid(detuning_n_values))
    if ensemble_realization_count <= 0:
        raise ValueError("ensemble_realization_count must be positive")
    if atoms_per_ensemble < 2:
        raise ValueError("atoms_per_ensemble must be at least two")
    if minimum_survivor_count is None:
        minimum_survivor_count = max(
            2,
            int(np.ceil(DEFAULT_MINIMUM_SURVIVOR_FRACTION * atoms_per_ensemble)),
        )
    if duration_s <= 0.0 or time_step_s <= 0.0 or worker_count <= 0:
        raise ValueError("duration, timestep, and worker count must be positive")
    if seed < 0 or plateau_window_s <= 0.0 or record_interval_s <= 0.0:
        raise ValueError("seed must be non-negative and intervals must be positive")
    if minimum_survivor_count < 2 or minimum_survivor_count > atoms_per_ensemble:
        raise ValueError(
            f"minimum survivor count must be between 2 and {atoms_per_ensemble}"
        )
    if stationarity_limit <= 0.0:
        raise ValueError("stationarity limit must be positive")
    continuous_parameters = (
        duration_s,
        time_step_s,
        initial_temperature_k,
        initial_position_sigma_m,
        plateau_window_s,
        record_interval_s,
        stationarity_limit,
        cooling_power_w_per_beam,
    )
    if not np.all(np.isfinite(continuous_parameters)):
        raise ValueError("all continuous sweep parameters must be finite")
    if plateau_window_s > duration_s or time_step_s > duration_s:
        raise ValueError("plateau window and timestep must not exceed duration")
    record_stride = max(1, int(round(record_interval_s / time_step_s)))
    plateau_sample_count = int(
        np.floor(plateau_window_s / (record_stride * time_step_s))
    ) + 1
    if plateau_sample_count < 4:
        raise ValueError("plateau window must include at least four recorded samples")

    configuration_audit = verify_only_cooling_detuning_changes(
        detuning_values,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )
    paths = multilevel_mot_paths()
    sweep_name = _temperature_sweep_name(
        ensemble_realization_count=ensemble_realization_count,
        atoms_per_ensemble=atoms_per_ensemble,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
        detuning_n_values=detuning_values,
    )
    output = output_directory or paths["statistics"] / sweep_name
    figures = figure_directory or paths["figures"] / sweep_name
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output / "temperature_vs_detuning.csv"
    ensemble_csv_path = output / "temperature_vs_detuning_ensembles.csv"
    metadata_path = output / "temperature_vs_detuning_metadata.json"
    physical_model_path = output / "physical_model.md"
    plot_path = figures / "temperature_vs_detuning.png"
    temperature_only_plot_path = figures / "temperature_only_vs_detuning.png"
    signature = _resume_signature(
        ensemble_realization_count=ensemble_realization_count,
        atoms_per_ensemble=atoms_per_ensemble,
        duration_s=duration_s,
        time_step_s=time_step_s,
        initial_temperature_k=initial_temperature_k,
        initial_position_sigma_m=initial_position_sigma_m,
        seed=seed,
        plateau_window_s=plateau_window_s,
        record_interval_s=record_interval_s,
        minimum_survivor_count=minimum_survivor_count,
        stationarity_limit=stationarity_limit,
        configuration_audit=configuration_audit,
        detuning_n_values=detuning_values,
        cooling_power_w_per_beam=cooling_power_w_per_beam,
    )

    created_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    checkpoint_paths = (summary_csv_path, ensemble_csv_path, metadata_path)
    if resume and any(path.exists() for path in checkpoint_paths):
        if not all(path.exists() for path in checkpoint_paths):
            raise RuntimeError(
                "resume requires summary CSV, ensemble CSV, and metadata JSON"
            )
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prior_signature = prior_metadata.get("resume_signature")
        if not isinstance(prior_signature, Mapping) or not _resume_signature_is_compatible(
            prior_signature,
            signature,
        ):
            raise ValueError("existing temperature checkpoint has incompatible parameters")
        created_utc = str(prior_metadata.get("created_utc", created_utc))
        rows = [
            _refresh_doppler_reference_fields(
                row,
                cooling_power_w_per_beam=cooling_power_w_per_beam,
            )
            for row in _read_csv_checkpoint(summary_csv_path)
        ]
        ensemble_rows = _read_csv_checkpoint(ensemble_csv_path)

    expected_index = {
        _detuning_key(value): index for index, value in enumerate(detuning_values)
    }
    completed_by_key: dict[str, dict[str, object]] = {}
    for row in rows:
        key = _detuning_key(row["detuning_n"])
        if key not in expected_index or int(row["point_index"]) != expected_index[key]:
            raise ValueError("checkpoint contains an unexpected detuning/index pair")
        matching_ensembles = [
            candidate
            for candidate in ensemble_rows
            if int(candidate["point_index"]) == int(row["point_index"])
        ]
        ensemble_indices = {int(candidate["ensemble_index"]) for candidate in matching_ensembles}
        if (
            len(matching_ensembles) == ensemble_realization_count
            and ensemble_indices == set(range(ensemble_realization_count))
            and int(row.get("failed_atom_count", 0)) == 0
        ):
            completed_by_key[key] = row
    rows = list(completed_by_key.values())
    completed_indices = {int(row["point_index"]) for row in rows}
    ensemble_rows = [
        row for row in ensemble_rows if int(row["point_index"]) in completed_indices
    ]
    resumed_point_count = len(rows)
    remaining = [
        (index, value)
        for index, value in enumerate(detuning_values)
        if _detuning_key(value) not in completed_by_key
    ]

    _atomic_write_text(physical_model_path, physical_model_markdown(signature))
    positions, velocities = generate_common_initial_ensembles(
        ensemble_realization_count,
        atoms_per_ensemble,
        initial_temperature_k=initial_temperature_k,
        initial_position_sigma_m=initial_position_sigma_m,
        seed=seed,
    )
    started = perf_counter()
    trajectory_count_per_point = ensemble_realization_count * atoms_per_ensemble
    total_new_trajectories = len(remaining) * trajectory_count_per_point
    completed_new_trajectories = 0
    print(
        f"[temperature-sweep] points={len(detuning_values)}; "
        f"resumed={resumed_point_count}; remaining={len(remaining)}; "
        f"ensembles/point={ensemble_realization_count}; "
        f"atoms/ensemble={atoms_per_ensemble}; trajectories/point={trajectory_count_per_point}; "
        f"cooling={1e3 * cooling_power_w_per_beam:.6g} mW/beam; workers={worker_count}",
        flush=True,
    )
    print(
        "[temperature-sweep] physical model recorded: preloaded clouds, 24-state "
        "repumper rate equations, Langevin recoil, COM-subtracted final-plateau "
        "temperature; only cooling detuning varies",
        flush=True,
    )
    initial_metadata = _metadata_payload(
        signature=signature,
        rows=rows,
        ensemble_rows=ensemble_rows,
        status="running" if remaining else "completed",
        worker_count=worker_count,
        resume=resume,
        resumed_point_count=resumed_point_count,
        created_utc=created_utc,
        wall_time_s=0.0,
        summary_csv_path=summary_csv_path,
        ensemble_csv_path=ensemble_csv_path,
        plot_path=plot_path,
        temperature_only_plot_path=temperature_only_plot_path,
        metadata_path=metadata_path,
        physical_model_path=physical_model_path,
    )
    _write_csv_checkpoint(summary_csv_path, rows)
    _write_csv_checkpoint(
        ensemble_csv_path,
        ensemble_rows,
        fieldnames=ENSEMBLE_CSV_FIELDNAMES,
    )
    _atomic_write_json(metadata_path, initial_metadata)

    if remaining:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for point_index, detuning_n in remaining:
                point_started = perf_counter()
                gamma_hz = (
                    default_multilevel_mot_config().natural_linewidth_rad_per_s
                    / (2.0 * np.pi)
                )
                print(
                    f"[temperature-sweep] point {point_index+1}/{len(detuning_values)} "
                    f"start: n={detuning_n:+.2f}, "
                    f"Delta/(2pi)={detuning_n*gamma_hz/1e6:+.3f} MHz; "
                    f"submitting {ensemble_realization_count} x {atoms_per_ensemble} trajectories",
                    flush=True,
                )
                payloads = [
                    TemperatureSweepWorkerPayload(
                        point_index=point_index,
                        ensemble_index=ensemble_index,
                        atom_index=atom_index,
                        detuning_n=detuning_n,
                        cooling_power_w_per_beam=cooling_power_w_per_beam,
                        position_m=tuple(
                            float(value) for value in positions[ensemble_index, atom_index]
                        ),
                        velocity_m_per_s=tuple(
                            float(value) for value in velocities[ensemble_index, atom_index]
                        ),
                        recoil_seed=recoil_seed(
                            seed, point_index, ensemble_index, atom_index
                        ),
                        duration_s=duration_s,
                        time_step_s=time_step_s,
                        record_stride=record_stride,
                        plateau_window_s=plateau_window_s,
                    )
                    for ensemble_index in range(ensemble_realization_count)
                    for atom_index in range(atoms_per_ensemble)
                ]
                future_to_identity = {
                    executor.submit(_temperature_sweep_worker, payload): (
                        payload.ensemble_index,
                        payload.atom_index,
                    )
                    for payload in payloads
                }
                results_by_ensemble: list[list[TemperatureSweepWorkerResult]] = [
                    [] for _ in range(ensemble_realization_count)
                ]
                failed_by_ensemble = np.zeros(ensemble_realization_count, dtype=int)
                finished_by_ensemble = np.zeros(ensemble_realization_count, dtype=int)
                completed_point_trajectories = 0
                progress_interval = max(10, trajectory_count_per_point // 100)
                for future in as_completed(future_to_identity):
                    ensemble_index, atom_index = future_to_identity[future]
                    completed_point_trajectories += 1
                    completed_new_trajectories += 1
                    finished_by_ensemble[ensemble_index] += 1
                    try:
                        results_by_ensemble[ensemble_index].append(future.result())
                        outcome = "ok"
                    except Exception as error:
                        failed_by_ensemble[ensemble_index] += 1
                        outcome = f"ERROR {type(error).__name__}: {error}"
                    elapsed = perf_counter() - started
                    eta_s = (
                        (total_new_trajectories - completed_new_trajectories)
                        * elapsed
                        / completed_new_trajectories
                        if completed_new_trajectories
                        else 0.0
                    )
                    if (
                        finished_by_ensemble[ensemble_index] == atoms_per_ensemble
                        or completed_point_trajectories % progress_interval == 0
                        or outcome != "ok"
                    ):
                        print(
                            f"[temperature-sweep] point {point_index+1}/{len(detuning_values)} "
                            f"n={detuning_n:+.2f}; trajectories "
                            f"{completed_point_trajectories}/{trajectory_count_per_point}; "
                            f"ensemble {ensemble_index+1}/{ensemble_realization_count} "
                            f"atom {atom_index+1}/{atoms_per_ensemble}; "
                            f"{outcome}; global {completed_new_trajectories}/"
                            f"{total_new_trajectories}; ETA={eta_s/60.0:.1f} min",
                            flush=True,
                        )

                point_ensemble_rows: list[dict[str, object]] = []
                for ensemble_index in range(ensemble_realization_count):
                    results_by_ensemble[ensemble_index].sort(
                        key=lambda result: result.atom_index
                    )
                    point_ensemble_rows.append(
                        _summarize_temperature_realization(
                            point_index=point_index,
                            detuning_n=detuning_n,
                            ensemble_index=ensemble_index,
                            results=results_by_ensemble[ensemble_index],
                            initial_velocities_m_per_s=velocities[ensemble_index],
                            requested_atom_count=atoms_per_ensemble,
                            failed_atom_count=int(failed_by_ensemble[ensemble_index]),
                            plateau_window_s=plateau_window_s,
                            minimum_survivor_count=minimum_survivor_count,
                            stationarity_limit=stationarity_limit,
                        )
                    )
                ensemble_rows.extend(point_ensemble_rows)
                row = _summarize_temperature_point(
                    point_index=point_index,
                    detuning_n=detuning_n,
                    ensemble_rows=point_ensemble_rows,
                    requested_ensemble_count=ensemble_realization_count,
                    atoms_per_ensemble=atoms_per_ensemble,
                    point_wall_time_s=perf_counter() - point_started,
                    cooling_power_w_per_beam=cooling_power_w_per_beam,
                )
                rows.append(row)
                rows.sort(key=lambda item: int(item["point_index"]))
                ensemble_rows.sort(
                    key=lambda item: (
                        int(item["point_index"]),
                        int(item["ensemble_index"]),
                    )
                )
                _write_csv_checkpoint(summary_csv_path, rows)
                _write_csv_checkpoint(
                    ensemble_csv_path,
                    ensemble_rows,
                    fieldnames=ENSEMBLE_CSV_FIELDNAMES,
                )
                plot_temperature_vs_detuning(
                    rows,
                    plot_path,
                    cooling_power_w_per_beam=cooling_power_w_per_beam,
                    ensemble_realization_count=ensemble_realization_count,
                    atoms_per_ensemble=atoms_per_ensemble,
                )
                plot_temperature_vs_detuning(
                    rows,
                    temperature_only_plot_path,
                    cooling_power_w_per_beam=cooling_power_w_per_beam,
                    ensemble_realization_count=ensemble_realization_count,
                    atoms_per_ensemble=atoms_per_ensemble,
                    include_survivor_panel=False,
                )
                metadata = _metadata_payload(
                    signature=signature,
                    rows=rows,
                    ensemble_rows=ensemble_rows,
                    status="running",
                    worker_count=worker_count,
                    resume=resume,
                    resumed_point_count=resumed_point_count,
                    created_utc=created_utc,
                    wall_time_s=perf_counter() - started,
                    summary_csv_path=summary_csv_path,
                    ensemble_csv_path=ensemble_csv_path,
                    plot_path=plot_path,
                    temperature_only_plot_path=temperature_only_plot_path,
                    metadata_path=metadata_path,
                    physical_model_path=physical_model_path,
                )
                _atomic_write_json(metadata_path, metadata)
                temperature = _float_or_none(row["plateau_temperature_mean_k"])
                temperature_text = (
                "unavailable" if temperature is None else f"{1e6*temperature:.1f} µK"
                )
                print(
                    f"[temperature-sweep] point {point_index+1}/{len(detuning_values)} "
                    f"checkpointed: n={detuning_n:+.2f}; "
                    f"cloud temperatures={row['temperature_ensemble_count']}/"
                    f"{ensemble_realization_count}; survivors={row['trapped_atom_count']}/"
                    f"{trajectory_count_per_point}; "
                    f"T_plateau={temperature_text}; quality={row['quality_status']}",
                    flush=True,
                )

    if rows:
        plot_temperature_vs_detuning(
            rows,
            plot_path,
            cooling_power_w_per_beam=cooling_power_w_per_beam,
            ensemble_realization_count=ensemble_realization_count,
            atoms_per_ensemble=atoms_per_ensemble,
        )
        plot_temperature_vs_detuning(
            rows,
            temperature_only_plot_path,
            cooling_power_w_per_beam=cooling_power_w_per_beam,
            ensemble_realization_count=ensemble_realization_count,
            atoms_per_ensemble=atoms_per_ensemble,
            include_survivor_panel=False,
        )
    final_status = (
        "completed"
        if len(rows) == len(detuning_values)
        and all(int(row.get("failed_atom_count", 0)) == 0 for row in rows)
        else "incomplete"
    )
    final_metadata = _metadata_payload(
        signature=signature,
        rows=rows,
        ensemble_rows=ensemble_rows,
        status=final_status,
        worker_count=worker_count,
        resume=resume,
        resumed_point_count=resumed_point_count,
        created_utc=created_utc,
        wall_time_s=perf_counter() - started,
        summary_csv_path=summary_csv_path,
        ensemble_csv_path=ensemble_csv_path,
        plot_path=plot_path,
        temperature_only_plot_path=temperature_only_plot_path,
        metadata_path=metadata_path,
        physical_model_path=physical_model_path,
    )
    _write_csv_checkpoint(summary_csv_path, rows)
    _write_csv_checkpoint(
        ensemble_csv_path,
        ensemble_rows,
        fieldnames=ENSEMBLE_CSV_FIELDNAMES,
    )
    _atomic_write_json(metadata_path, final_metadata)
    print(
        f"[temperature-sweep] {final_status}: {len(rows)}/{len(detuning_values)} "
        f"points; {len(ensemble_rows)} ensemble rows; "
        f"wall={perf_counter()-started:.1f} s",
        flush=True,
    )
    return {
        "status": final_status,
        "completed_point_count": len(rows),
        "total_point_count": len(detuning_values),
        "rows": rows,
        "ensemble_rows": ensemble_rows,
        "outputs": final_metadata["outputs"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 24-state multilevel MOT preloaded-cloud temperature sweep"
        )
    )
    parser.add_argument(
        "--ensemble-realizations",
        type=int,
        default=REQUIRED_ENSEMBLE_REALIZATION_COUNT,
        help="independent preloaded clouds per detuning (default: 10)",
    )
    parser.add_argument(
        "--atoms-per-ensemble",
        type=int,
        default=REQUIRED_ATOMS_PER_ENSEMBLE,
        help="atoms in each preloaded cloud (default: 25)",
    )
    parser.add_argument(
        "--detuning-n",
        type=float,
        nargs="+",
        default=None,
        metavar="N",
        help="custom negative Delta/Gamma values; default is the established 25-point grid",
    )
    parser.add_argument(
        "--cooling-power-mw",
        type=float,
        default=1.0e3 * DEFAULT_COOLING_POWER_W_PER_BEAM,
        help="fixed power in each of the six cooling beams (default: 20 mW)",
    )
    parser.add_argument(
        "--minimum-survivors",
        type=int,
        default=None,
        help="minimum final-core atoms per cloud; default scales as 20%% of cloud size",
    )
    parser.add_argument("--duration-ms", type=float, default=1.0e3 * DEFAULT_DURATION_S)
    parser.add_argument("--dt-us", type=float, default=1.0e6 * DEFAULT_TIME_STEP_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_temperature_detuning_sweep(
        ensemble_realization_count=args.ensemble_realizations,
        atoms_per_ensemble=args.atoms_per_ensemble,
        duration_s=1.0e-3 * args.duration_ms,
        time_step_s=1.0e-6 * args.dt_us,
        worker_count=args.workers,
        seed=args.seed,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=args.resume,
        minimum_survivor_count=args.minimum_survivors,
        detuning_n_values=(
            DETUNING_N_VALUES if args.detuning_n is None else args.detuning_n
        ),
        cooling_power_w_per_beam=1.0e-3 * args.cooling_power_mw,
    )
    print(json.dumps(result["outputs"], indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COOLING_POWER_W_PER_BEAM",
    "DEFAULT_MINIMUM_SURVIVOR_FRACTION",
    "DETUNING_N_VALUES",
    "DOPPLER_REFERENCE_VERSION",
    "PHYSICAL_MODEL_STATEMENT",
    "REQUIRED_ATOMS_PER_ENSEMBLE",
    "REQUIRED_ENSEMBLE_REALIZATION_COUNT",
    "REQUIRED_TRAJECTORY_COUNT",
    "TemperatureSweepWorkerPayload",
    "TemperatureSweepWorkerResult",
    "build_argument_parser",
    "build_temperature_sweep_configuration",
    "cooling_doppler_reference",
    "detuning_n_grid",
    "ensemble_temperature_metrics",
    "generate_common_initial_ensemble",
    "generate_common_initial_ensembles",
    "main",
    "physical_model_markdown",
    "plot_temperature_vs_detuning",
    "recoil_seed",
    "run_temperature_detuning_sweep",
    "verify_only_cooling_detuning_changes",
]
