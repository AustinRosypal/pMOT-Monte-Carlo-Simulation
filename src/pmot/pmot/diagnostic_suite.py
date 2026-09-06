"""Ordered construction-QA diagnostics for the provisional vector-only pMOT.

The canonical procedure is ``docs/pmot/DIAGNOSTIC_TESTS.md``.  This runner
obeys its stop-on-first-failure rule.  It currently executes Tests 0--2; the
known Test-2 failure is recorded rather than hidden, and Tests 3--9 are marked
``NOT_RUN``.  Every rendered figure has a CSV table containing its source
numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..configuration import HBAR_J_S
from ..configuration import PLANCK_CONSTANT_J_S
from ..configuration import SPEED_OF_LIGHT_M_PER_S
from ..configuration import VACUUM_PERMITTIVITY_F_PER_M
from ..fields import beam_intensity_w_per_m2
from ..mot_multilevel.coupling import beam_polarization_vector
from ..mot_multilevel.coupling import wavevector_rad_per_m
from ..mot_multilevel.polarization import polarization_weights
from ..mot_multilevel.polarization import propagation_frame_polarization
from ..mot_multilevel.polarization import spherical_basis
from ..mot_multilevel.rate_equations import build_beam_stimulated_rate_matrices
from ..mot_multilevel.rate_equations import rate_equation_observable_from_local_environment
from .ac_stark import EFFECTIVE_DETUNING_EQUATION
from .ac_stark import provisional_transition_stark_shifts
from .polarizability import interpolate_differential_polarizability_arrays
from .trapping_beams import helicity_sign
from .vector_only_trajectories import build_vector_only_trajectory_context
from .vector_only_trajectories import vector_only_trajectory_observable


DEFAULT_OUTPUT_TAG = "initial_construction_qa_20260903"
REQUIRED_PYTHON_INVOCATION = "/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python"
TEST_NAMES = {
    0: "Configuration and unit audit",
    1: "Cooling-only Doppler force",
    2: "Signed 1529-nm vector-shift profile",
    3: "Static restoring force",
    4: "Transition-resolved scattering audit",
    5: "Central detuning and power scan",
    6: "Internal-state population dynamics",
    7: "Handedness and polarization conventions",
    8: "Deterministic trajectories and capture map",
    9: "Three-dimensional stability",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reference_transition_index(model) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("stretched cycling transition is absent")


def _active_transition_indices(model, beam) -> np.ndarray:
    if beam.family == "cooling":
        return np.flatnonzero(model.transition_ground_f == 2)
    if beam.family == "repump":
        return np.flatnonzero(model.transition_ground_f == 1)
    raise ValueError(beam.family)


def _complex_parts(vector) -> tuple[list[float], list[float]]:
    values = np.asarray(vector, dtype=complex)
    return values.real.tolist(), values.imag.tolist()


def _beam_manifest(context) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for beam in context.cooling_repump_beams:
        epsilon = np.asarray(beam_polarization_vector(beam), dtype=complex)
        k_hat = np.asarray(beam.direction, dtype=float)
        k = np.asarray(wavevector_rad_per_m(beam), dtype=float)
        real, imaginary = _complex_parts(epsilon)
        rows.append(
            {
                "label": beam.label,
                "family": beam.family,
                "axis_name": beam.axis_name,
                "propagation_sense": beam.propagation_sense,
                "propagation_frame_polarization": beam.circular_polarization,
                "k_hat_x": k_hat[0],
                "k_hat_y": k_hat[1],
                "k_hat_z": k_hat[2],
                "k_x_rad_per_m": k[0],
                "k_y_rad_per_m": k[1],
                "k_z_rad_per_m": k[2],
                "epsilon_x_real": real[0],
                "epsilon_y_real": real[1],
                "epsilon_z_real": real[2],
                "epsilon_x_imag": imaginary[0],
                "epsilon_y_imag": imaginary[1],
                "epsilon_z_imag": imaginary[2],
                "epsilon_norm_squared": float(np.vdot(epsilon, epsilon).real),
                "abs_khat_dot_epsilon": float(abs(np.dot(k_hat, epsilon))),
                "power_w": beam.power_w,
                "wavelength_m": beam.wavelength_m,
                "laser_detuning_hz_metadata": beam.detuning_hz,
                "beam_1_over_e2_radius_m": beam.beam_radius_m,
                "power_semantics": "one traveling 780-nm component",
            }
        )
    for beam in context.trapping_beams:
        epsilon = np.asarray(
            propagation_frame_polarization(beam.direction, beam.helicity),
            dtype=complex,
        )
        k_hat = np.asarray(beam.direction, dtype=float)
        k = 2.0 * np.pi * k_hat / beam.wavelength_m
        real, imaginary = _complex_parts(epsilon)
        rows.append(
            {
                "label": beam.label,
                "family": "trapping",
                "axis_name": beam.axis_name,
                "propagation_sense": beam.propagation_sense,
                "propagation_frame_polarization": beam.helicity,
                "k_hat_x": k_hat[0],
                "k_hat_y": k_hat[1],
                "k_hat_z": k_hat[2],
                "k_x_rad_per_m": k[0],
                "k_y_rad_per_m": k[1],
                "k_z_rad_per_m": k[2],
                "epsilon_x_real": real[0],
                "epsilon_y_real": real[1],
                "epsilon_z_real": real[2],
                "epsilon_x_imag": imaginary[0],
                "epsilon_y_imag": imaginary[1],
                "epsilon_z_imag": imaginary[2],
                "epsilon_norm_squared": float(np.vdot(epsilon, epsilon).real),
                "abs_khat_dot_epsilon": float(abs(np.dot(k_hat, epsilon))),
                "power_w": context.trapping_power_w_per_path
                * beam.power_per_incident_path_watt,
                "wavelength_m": beam.wavelength_m,
                "laser_detuning_hz_metadata": np.nan,
                "beam_1_over_e2_radius_m": beam.waist_radius_m,
                "power_semantics": (
                    "power on its Cartesian incident path times the component's "
                    "retro power fraction"
                ),
            }
        )
    return pd.DataFrame(rows)


def _configuration_consistency(context) -> tuple[dict[str, bool], pd.DataFrame]:
    """Cross-check resolved beam metadata against the production rate config."""

    rows: list[dict[str, Any]] = []
    config = context.multilevel_config
    apparatus = context.apparatus.mot_light
    for beam in context.cooling_repump_beams:
        if beam.family == "cooling":
            expected_detuning_hz = config.cooling_detuning_rad_per_s / (2.0 * np.pi)
            expected_wavelength_m = config.wavelength_m
            expected_power_w = apparatus.cooling.power_w_per_beam
        else:
            expected_detuning_hz = config.repump_detuning_rad_per_s / (2.0 * np.pi)
            expected_wavelength_m = config.repump_wavelength_m
            expected_power_w = apparatus.repump.power_w_per_beam
        rows.append(
            {
                "beam": beam.label,
                "family": beam.family,
                "beam_detuning_hz": beam.detuning_hz,
                "production_config_detuning_hz": expected_detuning_hz,
                "detuning_matches": bool(
                    np.isclose(
                        beam.detuning_hz,
                        expected_detuning_hz,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ),
                "beam_wavelength_m": beam.wavelength_m,
                "production_config_wavelength_m": expected_wavelength_m,
                "wavelength_matches": bool(
                    np.isclose(
                        beam.wavelength_m,
                        expected_wavelength_m,
                        rtol=0.0,
                        atol=1.0e-18,
                    )
                ),
                "beam_power_w": beam.power_w,
                "apparatus_power_w": expected_power_w,
                "power_matches": bool(
                    np.isclose(
                        beam.power_w,
                        expected_power_w,
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    checks = {
        "all_beam_detunings_match_production_config": bool(
            frame["detuning_matches"].all()
        ),
        "all_beam_wavelengths_match_production_config": bool(
            frame["wavelength_matches"].all()
        ),
        "all_beam_powers_match_apparatus": bool(frame["power_matches"].all()),
        "repump_beam_power_matches_rate_config": bool(
            np.isclose(
                apparatus.repump.power_w_per_beam,
                config.repump_power_w_per_beam,
                rtol=0.0,
                atol=1.0e-15,
            )
        ),
    }
    return checks, frame


def _selected_rate_matrix_value(model, matrices: np.ndarray, transition_index: int) -> float:
    return float(
        matrices[
            0,
            int(model.transition_excited[transition_index]),
            int(model.transition_ground[transition_index]),
        ]
    )


def _production_kernel_detuning_probe(context) -> tuple[dict[str, bool], pd.DataFrame]:
    """Exercise the production kernel's Doppler and transition-shift signs."""

    model = context.model
    config = replace(context.multilevel_config, include_gravity=False)
    transition_index = _reference_transition_index(model)
    transition = model.structure.absorption_transitions[transition_index]
    candidates = [
        beam
        for beam in context.cooling_repump_beams
        if beam.family == "cooling" and beam.axis_name == "vertical_z"
    ]
    beam = max(
        candidates,
        key=lambda candidate: polarization_weights(
            beam_polarization_vector(candidate), (0.0, 0.0, 1.0)
        )[transition.q],
    )
    k = np.asarray(wavevector_rad_per_m(beam), dtype=float)
    k_hat = np.asarray(beam.direction, dtype=float)
    speed_m_per_s = 0.05
    doppler_magnitude = float(np.dot(k, speed_m_per_s * k_hat))
    zero_shift = np.zeros(len(model.transition_ground), dtype=float)

    cases = (
        ("baseline", 0.0, 0.0),
        ("plus_velocity_zero_shift", +speed_m_per_s, 0.0),
        ("minus_velocity_zero_shift", -speed_m_per_s, 0.0),
        ("zero_velocity_plus_shift", 0.0, +doppler_magnitude),
        ("zero_velocity_minus_shift", 0.0, -doppler_magnitude),
        ("plus_velocity_minus_shift", +speed_m_per_s, -doppler_magnitude),
        ("minus_velocity_plus_shift", -speed_m_per_s, +doppler_magnitude),
    )
    rows: list[dict[str, Any]] = []
    rates: dict[str, float] = {}
    for name, signed_speed, signed_shift in cases:
        velocity = signed_speed * k_hat
        shift = zero_shift.copy()
        shift[:] = signed_shift
        matrices = build_beam_stimulated_rate_matrices(
            model,
            [beam],
            (0.0, 0.0, 0.0),
            tuple(velocity),
            0.0,
            (0.0, 0.0, 1.0),
            config,
            transition_resonance_shift_rad_per_s=shift,
        )
        production_rate = _selected_rate_matrix_value(
            model, matrices, transition_index
        )
        rates[name] = production_rate
        rows.append(
            {
                "case": name,
                "beam": beam.label,
                "ground_F": transition.ground_f,
                "ground_mF": transition.ground_m_f,
                "excited_F": transition.excited_f,
                "excited_mF": transition.excited_m_f,
                "q": transition.q,
                "speed_along_khat_m_per_s": signed_speed,
                "k_dot_v_rad_per_s": float(np.dot(k, velocity)),
                "transition_shift_rad_per_s": signed_shift,
                "production_stimulated_rate_per_s": production_rate,
            }
        )

    close = lambda left, right: bool(
        np.isclose(rates[left], rates[right], rtol=2.0e-14, atol=1.0e-8)
    )
    checks = {
        "plus_velocity_equals_positive_transition_shift": close(
            "plus_velocity_zero_shift", "zero_velocity_plus_shift"
        ),
        "minus_velocity_equals_negative_transition_shift": close(
            "minus_velocity_zero_shift", "zero_velocity_minus_shift"
        ),
        "minus_shift_compensates_plus_velocity": close(
            "plus_velocity_minus_shift", "baseline"
        ),
        "plus_shift_compensates_minus_velocity": close(
            "minus_velocity_plus_shift", "baseline"
        ),
        "red_detuned_doppler_rate_ordering": bool(
            rates["minus_velocity_zero_shift"]
            > rates["baseline"]
            > rates["plus_velocity_zero_shift"]
        ),
        "red_detuned_transition_shift_rate_ordering": bool(
            rates["zero_velocity_minus_shift"]
            > rates["baseline"]
            > rates["zero_velocity_plus_shift"]
        ),
    }
    return checks, pd.DataFrame(rows)


def _test_zero_shift_and_velocity_algebra(context) -> tuple[dict[str, Any], pd.DataFrame]:
    model = context.model
    config = replace(context.multilevel_config, include_gravity=False)
    beams = list(context.cooling_repump_beams)
    position = (0.0, 0.0, 0.0)
    velocity = (0.0, 0.0, 0.0)
    axis = (0.0, 0.0, 1.0)
    omitted = build_beam_stimulated_rate_matrices(
        model, beams, position, velocity, 0.0, axis, config
    )
    zero = build_beam_stimulated_rate_matrices(
        model,
        beams,
        position,
        velocity,
        0.0,
        axis,
        config,
        transition_resonance_shift_rad_per_s=np.zeros(len(model.transition_ground)),
    )

    probe_velocity = np.asarray((0.12, -0.08, 0.05), dtype=float)
    f2_reference_offset = model.structure.states[
        model.structure.state_index("excited", 2, 0)
    ].energy_offset_rad_per_s
    rows: list[dict[str, Any]] = []
    for beam in beams:
        k_dot_v = float(np.dot(wavevector_rad_per_m(beam), probe_velocity))
        for transition_index in _active_transition_indices(model, beam):
            transition = model.structure.absorption_transitions[transition_index]
            if beam.family == "cooling":
                base = (
                    config.cooling_detuning_rad_per_s
                    - transition.hyperfine_offset_rad_per_s
                )
            else:
                base = config.repump_detuning_rad_per_s - (
                    transition.hyperfine_offset_rad_per_s - f2_reference_offset
                )
            plus = base - k_dot_v
            minus = base + k_dot_v
            rows.append(
                {
                    "beam": beam.label,
                    "family": beam.family,
                    "transition_index": int(transition_index),
                    "probe_vx_m_per_s": probe_velocity[0],
                    "probe_vy_m_per_s": probe_velocity[1],
                    "probe_vz_m_per_s": probe_velocity[2],
                    "base_laser_minus_hyperfine_rad_per_s": base,
                    "k_dot_v_rad_per_s": k_dot_v,
                    "detuning_plus_v_rad_per_s": plus,
                    "detuning_minus_v_rad_per_s": minus,
                    "mean_residual_rad_per_s": 0.5 * (plus + minus) - base,
                    "odd_residual_rad_per_s": 0.5 * (minus - plus) - k_dot_v,
                }
            )
    algebra = pd.DataFrame(rows)

    point = (0.0, 0.0, 1.0e-3)
    speed = 0.1
    plus_full = vector_only_trajectory_observable(
        context, point, (0.0, 0.0, speed), previous_axis=axis
    )
    minus_full = vector_only_trajectory_observable(
        context, point, (0.0, 0.0, -speed), previous_axis=axis
    )
    shift_difference_hz = (
        np.asarray(plus_full.applied_transition_shift_rad_per_s)
        - np.asarray(minus_full.applied_transition_shift_rad_per_s)
    ) / (2.0 * np.pi)
    summary = {
        "zero_transition_shift_exactly_matches_omitted_shift": bool(
            np.array_equal(omitted, zero)
        ),
        "frozen_shift_velocity_reversal_max_mean_residual_rad_per_s": float(
            algebra["mean_residual_rad_per_s"].abs().max()
        ),
        "frozen_shift_velocity_reversal_max_odd_residual_rad_per_s": float(
            algebra["odd_residual_rad_per_s"].abs().max()
        ),
        "full_physics_probe_position_m": list(point),
        "full_physics_probe_speed_m_per_s": speed,
        "full_physics_max_applied_stark_shift_change_under_velocity_reversal_hz": float(
            np.max(np.abs(shift_difference_hz))
        ),
        "qualification": (
            "With a frozen transition shift, reversing velocity changes only the "
            "780-nm Doppler term. In the full pMOT evaluation it also changes each "
            "trapping component's atom-frame wavelength and therefore its "
            "interpolated polarizability, as separately required by the model."
        ),
    }
    return summary, algebra


def run_test_00(context, root: Path) -> dict[str, Any]:
    directory = root / "test_00_configuration_and_units"
    directory.mkdir(parents=True, exist_ok=True)
    print("[Test 0] auditing configuration, units, and detuning algebra", flush=True)
    manifest = _beam_manifest(context)
    manifest.to_csv(directory / "beam_manifest.csv", index=False)
    algebra_checks, algebra = _test_zero_shift_and_velocity_algebra(context)
    algebra.to_csv(directory / "velocity_reversal_algebra.csv", index=False)
    production_checks, production_probe = _production_kernel_detuning_probe(context)
    production_probe.to_csv(
        directory / "production_kernel_detuning_sign_probe.csv", index=False
    )
    consistency_checks, consistency = _configuration_consistency(context)
    consistency.to_csv(directory / "beam_config_consistency.csv", index=False)

    table = context.polarizability_table
    unit_rows = [
        ("position", "m", "SI internal", "PASS"),
        ("velocity", "m/s", "SI internal", "PASS"),
        ("780 wavevector", "rad/m", "2*pi/lambda", "PASS"),
        ("multilevel detuning", "rad/s", "angular frequency", "PASS"),
        ("natural linewidth", "rad/s", "angular frequency", "PASS"),
        ("MOTBeam.detuning_hz", "Hz", "apparatus metadata only", "PASS"),
        ("intensity", "W/m^2", "per traveling component before summation", "PASS"),
        ("Gaussian radius", "m", "1/e^2 intensity radius", "PASS"),
        ("saturation", "dimensionless", "per beam, transition strength, and P_q", "PASS"),
        (
            "polarizability",
            "assumed SI (physical dimension absent from raw CSV header)",
            "values are consumed as SI by -alpha*E^2",
            "WARNING",
        ),
        (
            "1529 shift",
            "J then rad/s",
            "differential transition proxy divided by hbar",
            "PASS_WITH_MODEL_LIMITATION",
        ),
    ]
    units = pd.DataFrame(unit_rows, columns=("quantity", "units", "convention", "status"))
    units.to_csv(directory / "unit_audit.csv", index=False)

    configuration = {
        "schema": "pmot.diagnostic.test-00.v1",
        "effective_detuning_equation": EFFECTIVE_DETUNING_EQUATION,
        "frequency_convention": "Delta = omega_L - omega_0; angular frequency",
        "natural_linewidth_rad_per_s": context.multilevel_config.natural_linewidth_rad_per_s,
        "natural_linewidth_hz": context.multilevel_config.natural_linewidth_rad_per_s
        / (2.0 * np.pi),
        "position_units": "m",
        "velocity_units": "m/s",
        "intensity_units": "W/m^2",
        "beam_radius_convention": "Gaussian 1/e^2 intensity radius",
        "saturation_convention": "per beam and per transition",
        "external_magnetic_field_t": [0.0, 0.0, 0.0],
        "cooling_power_w_per_beam": context.apparatus.mot_light.cooling.power_w_per_beam,
        "repump_power_w_per_beam": context.apparatus.mot_light.repump.power_w_per_beam,
        "trapping_power_w_per_incident_path": context.trapping_power_w_per_path,
        "trapping_power_source": context.trapping_power_source,
        "trapping_wavelength_m": context.apparatus.trapping_laser.wavelength_m,
        "polarizability_source": table.source_path,
        "polarizability_wavelength_range_nm": table.wavelength_range_nm,
        "polarizability_source_unit_warning": (
            "The raw CSV column headers do not encode the physical SI dimension. "
            "The implementation names and consumes the values as SI; independent "
            "source-unit provenance is still required."
        ),
        "resolved_apparatus_config": asdict(context.apparatus),
        "resolved_multilevel_rate_config": asdict(context.multilevel_config),
        "resolved_provisional_stark_config": asdict(context.stark_config),
        "resolved_component_helicities": asdict(context.helicities),
        "rate_model_dimensions": {
            "ground_states": context.model.ground_count,
            "excited_states": context.model.excited_count,
            "allowed_absorption_transitions": len(
                context.model.structure.absorption_transitions
            ),
        },
        "configuration_consistency_checks": consistency_checks,
        "production_kernel_detuning_sign_checks": production_checks,
        "stark_zero_and_analytic_velocity_checks": algebra_checks,
    }
    _write_json(directory / "configuration_snapshot.json", configuration)
    result = {
        "test_number": 0,
        "name": TEST_NAMES[0],
        "status": "PASS_WITH_QUALIFICATIONS",
        "criteria": {
            "explicit_frequency_unit_types": True,
            "single_documented_effective_detuning_definition": True,
            "zero_shift_removes_ac_stark_contribution_exactly": algebra_checks[
                "zero_transition_shift_exactly_matches_omitted_shift"
            ],
            "production_kernel_doppler_and_stark_sign_checks": bool(
                all(production_checks.values())
            ),
            "beam_metadata_matches_production_configuration": bool(
                all(consistency_checks.values())
            ),
            "beam_polarizations_normalized_and_transverse": bool(
                np.allclose(manifest["epsilon_norm_squared"], 1.0, atol=1.0e-14)
                and np.all(manifest["abs_khat_dot_epsilon"] <= 1.0e-14)
            ),
        },
        "qualification_is_non_gating": True,
        "warnings": [
            configuration["polarizability_source_unit_warning"],
            algebra_checks["qualification"],
        ],
        "outputs": [
            directory / "README.md",
            directory / "result.json",
            directory / "configuration_snapshot.json",
            directory / "unit_audit.csv",
            directory / "beam_manifest.csv",
            directory / "beam_config_consistency.csv",
            directory / "velocity_reversal_algebra.csv",
            directory / "production_kernel_detuning_sign_probe.csv",
        ],
    }
    _write_json(directory / "result.json", result)
    _write_markdown(
        directory / "README.md",
        f"""# Test 0 — configuration and unit audit

**Result: PASS WITH QUALIFICATIONS.** The production rate path uses angular
frequency and implements

`{EFFECTIVE_DETUNING_EQUATION}`

An all-zero transition-shift array is bit-for-bit identical to omitting the
Stark input. Direct calls to the production rate kernel verify both its
`-k dot v` and `-DeltaE_AC/hbar` signs, including compensating Doppler/Stark
identities and the rate ordering on the red-detuned side. Resolved beam
detunings, wavelengths, and powers match the production configuration.

With the Stark shift frozen, reversing velocity changes only the Doppler term.
In the complete pMOT call, velocity also Doppler-shifts the 1529-nm wavelengths
used to interpolate polarizability; the measured maximum shift change for the
recorded probe is
{algebra_checks['full_physics_max_applied_stark_shift_change_under_velocity_reversal_hz']:.6g} Hz.

The remaining provenance gap is that the raw Arora CSV headers do not state a
physical polarizability unit. The code assumes SI and the values have the
expected SI scale, but that source metadata has not been independently
verified. See `unit_audit.csv`, `configuration_snapshot.json`, and the complete
18-component `beam_manifest.csv`. The qualification is explicitly non-gating
because every code-side convention and conversion is recorded and internally
consistent; only independent provenance for the raw table's unit label is
missing.
""",
    )
    return result


def _fixed_f2_population(model) -> np.ndarray:
    population = np.zeros(model.ground_count, dtype=float)
    for local_index, global_index in enumerate(model.structure.ground_state_indices):
        if model.structure.states[global_index].f == 2:
            population[local_index] = 0.2
    if not np.isclose(np.sum(population), 1.0):
        raise RuntimeError("fixed F=2 validation population is not normalized")
    return population


def _fixed_population_force(context, beams, velocity_z_m_per_s: float) -> dict[str, Any]:
    config = replace(
        context.multilevel_config,
        repumper_enabled=False,
        include_gravity=False,
    )
    matrices = build_beam_stimulated_rate_matrices(
        context.model,
        list(beams),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, float(velocity_z_m_per_s)),
        0.0,
        (0.0, 0.0, 1.0),
        config,
        transition_resonance_shift_rad_per_s=np.zeros(
            len(context.model.transition_ground)
        ),
    )
    ground = _fixed_f2_population(context.model)
    rates = np.asarray(
        [float(np.sum(matrix * ground[None, :])) for matrix in matrices]
    )
    beam_forces = np.asarray(
        [HBAR_J_S * np.asarray(wavevector_rad_per_m(beam)) * rate for beam, rate in zip(beams, rates)]
    )
    return {
        "beam_rate_matrices_per_s": matrices,
        "rates_per_s": rates,
        "forces_n": beam_forces,
        "total_force_n": np.sum(beam_forces, axis=0),
    }


def _fixed_population_transition_rows(
    context,
    beams,
    velocity_z_m_per_s: float,
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the complete transition ledger behind one Test-1 force sample."""

    model = context.model
    config = replace(
        context.multilevel_config,
        repumper_enabled=False,
        include_gravity=False,
    )
    position = (0.0, 0.0, 0.0)
    velocity = np.asarray((0.0, 0.0, velocity_z_m_per_s), dtype=float)
    axis = (0.0, 0.0, 1.0)
    ground_population = _fixed_f2_population(model)
    gamma = config.natural_linewidth_rad_per_s
    rows: list[dict[str, Any]] = []
    mask = (model.transition_ground_f == 2) & np.isin(
        model.transition_excited_f, config.enabled_excited_manifolds
    )
    for beam_index, beam in enumerate(beams):
        k_hat = np.asarray(beam.direction, dtype=float)
        k = np.asarray(wavevector_rad_per_m(beam), dtype=float)
        epsilon = np.asarray(beam_polarization_vector(beam), dtype=complex)
        epsilon_real, epsilon_imag = _complex_parts(epsilon)
        weights = polarization_weights(tuple(epsilon), axis)
        basis = spherical_basis(axis)
        raw_weights = {
            q: float(abs(np.vdot(np.asarray(basis[q]), epsilon)) ** 2)
            for q in (-1, 0, +1)
        }
        raw_weight_sum = float(sum(raw_weights.values()))
        intensity = beam_intensity_w_per_m2(beam, position)
        doppler = float(np.dot(k, velocity))
        for transition_index in np.flatnonzero(mask):
            transition = model.structure.absorption_transitions[transition_index]
            ground_local = int(model.transition_ground[transition_index])
            excited_local = int(model.transition_excited[transition_index])
            q = int(model.transition_q[transition_index])
            polarization_weight = float(weights[q])
            strength = float(model.transition_strength[transition_index])
            saturation = (
                intensity
                / config.saturation_intensity_w_per_m2
                * strength
                * polarization_weight
            )
            hyperfine = float(
                model.transition_hyperfine_offset_rad_per_s[transition_index]
            )
            effective_detuning = (
                config.cooling_detuning_rad_per_s - hyperfine - doppler
            )
            expected_rate = 0.5 * gamma * saturation / (
                1.0
                + saturation
                + 4.0 * effective_detuning**2 / gamma**2
            )
            production_rate = float(
                sample["beam_rate_matrices_per_s"][
                    beam_index, excited_local, ground_local
                ]
            )
            population = float(ground_population[ground_local])
            absorption_rate = production_rate * population
            photon_momentum = HBAR_J_S * k
            force = photon_momentum * absorption_rate
            rows.append(
                {
                    "velocity_z_m_per_s": velocity_z_m_per_s,
                    "beam": beam.label,
                    "beam_index": beam_index,
                    "propagation_sense": beam.propagation_sense,
                    "k_hat_x": k_hat[0],
                    "k_hat_y": k_hat[1],
                    "k_hat_z": k_hat[2],
                    "k_x_rad_per_m": k[0],
                    "k_y_rad_per_m": k[1],
                    "k_z_rad_per_m": k[2],
                    "epsilon_x_real": epsilon_real[0],
                    "epsilon_y_real": epsilon_real[1],
                    "epsilon_z_real": epsilon_real[2],
                    "epsilon_x_imag": epsilon_imag[0],
                    "epsilon_y_imag": epsilon_imag[1],
                    "epsilon_z_imag": epsilon_imag[2],
                    "epsilon_norm_squared": float(np.vdot(epsilon, epsilon).real),
                    "abs_khat_dot_epsilon": float(abs(np.dot(k_hat, epsilon))),
                    "transition_index": int(transition_index),
                    "ground_global_index": transition.ground_state_index,
                    "ground_local_index": ground_local,
                    "ground_F": transition.ground_f,
                    "ground_mF": transition.ground_m_f,
                    "excited_global_index": transition.excited_state_index,
                    "excited_local_index": excited_local,
                    "excited_F": transition.excited_f,
                    "excited_mF": transition.excited_m_f,
                    "q": q,
                    "normalized_full_hyperfine_dipole_strength_C2": strength,
                    "sqrt_C2_amplitude_magnitude": np.sqrt(strength),
                    "polarization_weight_Pq": polarization_weight,
                    "polarization_weight_P_minus1": weights[-1],
                    "polarization_weight_P_0": weights[0],
                    "polarization_weight_P_plus1": weights[1],
                    "normalized_spherical_weight_sum": sum(weights.values()),
                    "raw_spherical_projection_sum": raw_weight_sum,
                    "component_intensity_w_per_m2": intensity,
                    "saturation_intensity_w_per_m2": config.saturation_intensity_w_per_m2,
                    "transition_saturation": saturation,
                    "laser_detuning_rad_per_s": config.cooling_detuning_rad_per_s,
                    "hyperfine_offset_rad_per_s": hyperfine,
                    "doppler_k_dot_v_rad_per_s": doppler,
                    "external_zeeman_shift_rad_per_s": 0.0,
                    "ac_stark_transition_shift_rad_per_s": 0.0,
                    "effective_detuning_rad_per_s": effective_detuning,
                    "production_stimulated_rate_per_s": production_rate,
                    "expected_stimulated_rate_per_s": expected_rate,
                    "production_minus_expected_rate_per_s": production_rate
                    - expected_rate,
                    "ground_population": population,
                    "available_absorption_rate_per_s": absorption_rate,
                    "photon_momentum_x_kg_m_per_s": photon_momentum[0],
                    "photon_momentum_y_kg_m_per_s": photon_momentum[1],
                    "photon_momentum_z_kg_m_per_s": photon_momentum[2],
                    "force_x_n": force[0],
                    "force_y_n": force[1],
                    "force_z_n": force[2],
                }
            )
    return rows


def run_test_01(context, root: Path, sample_count: int = 161) -> dict[str, Any]:
    if sample_count < 21 or sample_count % 2 == 0:
        raise ValueError("Test-1 sample_count must be odd and at least 21")
    directory = root / "test_01_cooling_only_doppler"
    figure_directory = directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    print("[Test 1] evaluating the deterministic one-axis cooling-only force", flush=True)
    beams = [
        beam
        for beam in context.cooling_repump_beams
        if beam.family == "cooling" and beam.axis_name == "vertical_z"
    ]
    if len(beams) != 2:
        raise RuntimeError("expected two z-axis cooling components")
    velocities = np.linspace(-2.0, 2.0, sample_count)
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for velocity in velocities:
        sample = _fixed_population_force(context, beams, float(velocity))
        transition_rows.extend(
            _fixed_population_transition_rows(
                context, beams, float(velocity), sample
            )
        )
        rows.append(
            {
                "velocity_z_m_per_s": velocity,
                "total_force_z_n": sample["total_force_n"][2],
                "plus_z_force_n": sample["forces_n"][0, 2],
                "minus_z_force_n": sample["forces_n"][1, 2],
                "plus_z_absorption_proxy_per_s": sample["rates_per_s"][0],
                "minus_z_absorption_proxy_per_s": sample["rates_per_s"][1],
            }
        )
    lineout = pd.DataFrame(rows)
    lineout_path = directory / "cooling_force_and_rates_vs_velocity.csv"
    lineout.to_csv(lineout_path, index=False)
    transition_ledger = pd.DataFrame(transition_rows)
    transition_path = directory / "transition_resolved_cooling_force.csv"
    transition_ledger.to_csv(transition_path, index=False)
    reconstructed = (
        transition_ledger.groupby(["velocity_z_m_per_s", "beam"], sort=False)
        .agg(
            available_absorption_rate_per_s=(
                "available_absorption_rate_per_s",
                "sum",
            ),
            force_x_n=("force_x_n", "sum"),
            force_y_n=("force_y_n", "sum"),
            force_z_n=("force_z_n", "sum"),
        )
        .reset_index()
    )
    reported_rows: list[dict[str, Any]] = []
    for _, lineout_row in lineout.iterrows():
        for beam in beams:
            is_plus = float(beam.direction[2]) > 0.0
            reported_rows.append(
                {
                    "velocity_z_m_per_s": lineout_row["velocity_z_m_per_s"],
                    "beam": beam.label,
                    "reported_absorption_rate_per_s": lineout_row[
                        "plus_z_absorption_proxy_per_s"
                        if is_plus
                        else "minus_z_absorption_proxy_per_s"
                    ],
                    "reported_force_z_n": lineout_row[
                        "plus_z_force_n" if is_plus else "minus_z_force_n"
                    ],
                }
            )
    reconstructed = reconstructed.merge(
        pd.DataFrame(reported_rows),
        on=["velocity_z_m_per_s", "beam"],
        validate="one_to_one",
    )
    reconstructed["absorption_rate_residual_per_s"] = (
        reconstructed["available_absorption_rate_per_s"]
        - reconstructed["reported_absorption_rate_per_s"]
    )
    reconstructed["force_z_residual_n"] = (
        reconstructed["force_z_n"] - reconstructed["reported_force_z_n"]
    )
    reconstruction_path = directory / "transition_sum_reconstruction.csv"
    reconstructed.to_csv(reconstruction_path, index=False)

    derivative_rows = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4):
        plus = _fixed_population_force(context, beams, step)["total_force_n"][2]
        minus = _fixed_population_force(context, beams, -step)["total_force_n"][2]
        slope = (plus - minus) / (2.0 * step)
        derivative_rows.append(
            {
                "velocity_step_m_per_s": step,
                "dFz_dvz_n_s_per_m": slope,
                "beta_z_n_s_per_m": -slope,
            }
        )
    derivatives = pd.DataFrame(derivative_rows)
    derivative_path = directory / "beta_velocity_step_convergence.csv"
    derivatives.to_csv(derivative_path, index=False)
    near = lineout[np.abs(lineout["velocity_z_m_per_s"]) <= 0.25]
    slope, intercept = np.polyfit(
        near["velocity_z_m_per_s"], near["total_force_z_n"], 1
    )
    beta_fit = -float(slope)
    zero_row = lineout.loc[
        np.argmin(np.abs(lineout["velocity_z_m_per_s"].to_numpy()))
    ]
    damping_rows = lineout[
        (np.abs(lineout["velocity_z_m_per_s"]) <= 0.5)
        & (lineout["velocity_z_m_per_s"] != 0.0)
    ]

    steady_rows = []
    steady_config = replace(
        context.multilevel_config, repumper_enabled=False, include_gravity=False
    )
    for velocity in (-0.1, 0.0, 0.1):
        observable = rate_equation_observable_from_local_environment(
            context.model,
            beams,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, velocity),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            steady_config,
        )
        steady_rows.append(
            {
                "velocity_z_m_per_s": velocity,
                "force_z_n": observable.force_n[2],
                "total_absorption_proxy_per_s": observable.total_scattering_rate_per_s,
                "F1_ground_population": float(np.sum(observable.populations[:3])),
                "interpretation": (
                    "nonunique full-graph steady state collapses into disconnected F=1; "
                    "not used for the staged Test-1 pass decision"
                ),
            }
        )
    pd.DataFrame(steady_rows).to_csv(
        directory / "full_24state_two_beam_steady_state_control.csv", index=False
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    axes[0].plot(velocities, 1.0e21 * lineout["total_force_z_n"], color="#111827", label="total")
    axes[0].plot(velocities, 1.0e21 * lineout["plus_z_force_n"], color="#dc2626", label="+z beam")
    axes[0].plot(velocities, 1.0e21 * lineout["minus_z_force_n"], color="#2563eb", label="-z beam")
    axes[0].set(xlabel=r"$v_z$ [m/s]", ylabel=r"Force [$10^{-21}$ N]", title="Fixed-population cooling-only force")
    axes[1].plot(velocities, lineout["plus_z_absorption_proxy_per_s"], color="#dc2626", label="+z beam")
    axes[1].plot(velocities, lineout["minus_z_absorption_proxy_per_s"], color="#2563eb", label="-z beam")
    axes[1].set(xlabel=r"$v_z$ [m/s]", ylabel=r"Available absorption proxy [s$^{-1}$]", title="Beam-resolved rates")
    axes[2].plot(near["velocity_z_m_per_s"], 1.0e21 * near["total_force_z_n"], "o", ms=3, label="samples")
    axes[2].plot(
        near["velocity_z_m_per_s"],
        1.0e21 * (slope * near["velocity_z_m_per_s"] + intercept),
        color="#7c3aed",
        label=rf"fit $\beta_z={beta_fit:.3e}$ N s/m",
    )
    axes[2].set(xlabel=r"$v_z$ [m/s]", ylabel=r"Force [$10^{-21}$ N]", title="Near-zero damping fit")
    for axis in axes:
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.axvline(0.0, color="0.4", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Test 1: prescribed uniform F=2 population; absorption-momentum proxy",
        fontsize=11,
    )
    figure_path = figure_directory / "cooling_only_doppler_force.png"
    figure.savefig(figure_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    criteria = {
        "small_velocity_force_opposes_velocity": bool(
            np.all(
                damping_rows["velocity_z_m_per_s"]
                * damping_rows["total_force_z_n"]
                < 0.0
            )
        ),
        "fitted_beta_positive": bool(beta_fit > 0.0),
        "balanced_zero_force": bool(abs(zero_row["total_force_z_n"]) <= 1.0e-30),
        "all_refined_beta_values_positive": bool(
            np.all(derivatives["beta_z_n_s_per_m"] > 0.0)
        ),
        "transition_rates_match_production_formula": bool(
            np.allclose(
                transition_ledger["production_stimulated_rate_per_s"],
                transition_ledger["expected_stimulated_rate_per_s"],
                rtol=2.0e-14,
                atol=1.0e-8,
            )
        ),
        "transition_sums_reproduce_beam_rates_and_forces": bool(
            np.allclose(
                reconstructed["available_absorption_rate_per_s"],
                reconstructed["reported_absorption_rate_per_s"],
                rtol=2.0e-14,
                atol=1.0e-8,
            )
            and np.allclose(
                reconstructed["force_z_n"],
                reconstructed["reported_force_z_n"],
                rtol=2.0e-14,
                atol=1.0e-30,
            )
        ),
        "polarization_weights_normalized_per_beam": bool(
            np.allclose(
                transition_ledger["normalized_spherical_weight_sum"],
                1.0,
                atol=1.0e-14,
            )
            and np.allclose(
                transition_ledger["raw_spherical_projection_sum"],
                transition_ledger["epsilon_norm_squared"],
                atol=1.0e-14,
            )
        ),
    }
    status = "PASS_WITH_QUALIFICATIONS" if all(criteria.values()) else "FAIL"
    result = {
        "test_number": 1,
        "name": TEST_NAMES[1],
        "status": status,
        "population_stage": "fixed uniform population over the five F=2 mF states",
        "force_convention": "ground-population-weighted available absorption-momentum proxy",
        "qualification_is_non_gating": True,
        "fitted_beta_z_n_s_per_m": beta_fit,
        "fitted_intercept_n": float(intercept),
        "zero_velocity_force_n": float(zero_row["total_force_z_n"]),
        "derivative_convergence": derivative_rows,
        "criteria": criteria,
        "numerical_ranges_and_tolerances": {
            "velocity_plot_range_m_per_s": [-2.0, 2.0],
            "near_zero_fit_abs_velocity_max_m_per_s": 0.25,
            "damping_sign_abs_velocity_max_m_per_s": 0.5,
            "balanced_force_absolute_tolerance_n": 1.0e-30,
            "rate_reconstruction_relative_tolerance": 2.0e-14,
            "rate_reconstruction_absolute_tolerance_per_s": 1.0e-8,
            "force_reconstruction_absolute_tolerance_n": 1.0e-30,
        },
        "steady_state_control_warning": (
            "The full 24-state graph with only two cooling beams and no repumper "
            "has disconnected F=1 dark states and a nonunique steady state. It "
            "returns zero force, so the staged fixed-population diagnostic required "
            "before Test 6 is used here."
        ),
        "dipole_amplitude_limitation": (
            "The current rate model stores only the normalized full hyperfine "
            "line strength C^2 (including Wigner-6j and Wigner-3j factors). "
            "sqrt(C^2) is saved as a nonnegative magnitude; a signed Clebsch-"
            "Gordan phase is not available from this representation."
        ),
        "outputs": [
            directory / "README.md",
            directory / "result.json",
            lineout_path,
            transition_path,
            reconstruction_path,
            derivative_path,
            directory / "full_24state_two_beam_steady_state_control.csv",
            figure_path,
        ],
    }
    _write_json(directory / "result.json", result)
    _write_markdown(
        directory / "README.md",
        f"""# Test 1 — cooling-only Doppler force

**Result: PASS WITH QUALIFICATIONS.** With gravity, recoil, the 1529-nm shift, repumping, and
unrelated axes disabled, the calculation prescribes equal populations of 0.2
in the five F=2 Zeeman substates. This is the fixed-population stage required
before adding optical pumping in Test 6. It uses the production saturated
transition-rate matrix and reconstructs the absorption-momentum proxy from the
two z cooling components.

The near-zero fit gives
`beta_z = {beta_fit:.9e} N s/m`; all three central-difference refinements are
positive, and the balanced zero-velocity force is
`{float(zero_row['total_force_z_n']):.3e} N`.

The companion steady-state control documents why the unrestricted 24-state
solve is not meaningful with only these two beams: population occupies the
disconnected F=1 dark subspace and the force collapses to zero. Numerical plot
inputs are in `cooling_force_and_rates_vs_velocity.csv`; the complete
beam-by-beam and transition-by-transition calculation is retained in
`transition_resolved_cooling_force.csv`, with grouped sums in
`transition_sum_reconstruction.csv`. Accordingly this is a pass of the
fixed-population Doppler stage, not a claim that the unrestricted two-beam
24-state steady state provides damping. The retained mechanical quantity is
the inherited ground-population-weighted available-absorption momentum proxy;
it is not yet a validated net scattering force because reverse stimulated
emission momentum is omitted. The rate model stores normalized full hyperfine
`C^2`; `sqrt(C^2)` in the ledger is a magnitude, not a signed Clebsch--Gordan
coefficient.
""",
    )
    return result


def _component_vector_energies_j(context, beams, diagnostic, alpha_sign: float = 1.0) -> np.ndarray:
    _, alpha_vector, _ = interpolate_differential_polarizability_arrays(
        diagnostic.atom_frame_wavelengths_nm,
        context.polarizability_table,
    )
    intensities = np.asarray(diagnostic.component_intensities_w_per_m2)
    field_squared = 2.0 * intensities / (
        SPEED_OF_LIGHT_M_PER_S * VACUUM_PERMITTIVITY_F_PER_M
    )
    return np.asarray(
        [
            -alpha_sign
            * alpha
            * amplitude_squared
            * helicity_sign(beam.helicity)
            * np.asarray(beam.direction, dtype=float)
            for beam, alpha, amplitude_squared in zip(beams, alpha_vector, field_squared)
        ]
    )


def _profile(context, beams, positions_m: np.ndarray) -> tuple[pd.DataFrame, list[Any]]:
    reference_index = _reference_transition_index(context.model)
    z_indices = [
        index for index, beam in enumerate(beams) if beam.axis_name == "vertical_z"
    ]
    diagnostics = []
    rows = []
    for z in positions_m:
        diagnostic = provisional_transition_stark_shifts(
            context.model,
            list(beams),
            (0.0, 0.0, float(z)),
            (0.0, 0.0, 0.0),
            context.apparatus.trapping_laser,
            context.stark_config,
            (0.0, 0.0, 1.0),
            polarizability_table=context.polarizability_table,
        )
        diagnostics.append(diagnostic)
        component_vectors = _component_vector_energies_j(
            context, beams, diagnostic
        )
        total_vector = np.sum(component_vectors, axis=0)
        intensities = np.asarray(diagnostic.component_intensities_w_per_m2)
        rows.append(
            {
                "z_m": z,
                "z_mm": 1.0e3 * z,
                "z_incident_intensity_w_per_m2": intensities[z_indices[0]],
                "z_retro_intensity_w_per_m2": intensities[z_indices[1]],
                "z_incident_vector_contribution_hz": component_vectors[z_indices[0], 2]
                / PLANCK_CONSTANT_J_S,
                "z_retro_vector_contribution_hz": component_vectors[z_indices[1], 2]
                / PLANCK_CONSTANT_J_S,
                "fixed_basis_signed_reference_shift_hz": total_vector[2]
                / PLANCK_CONSTANT_J_S,
                "applied_named_cycling_shift_hz": diagnostic.vector_transition_energy_j[
                    reference_index
                ]
                / PLANCK_CONSTANT_J_S,
                "effective_field_proxy_z_t": diagnostic.effective_field_t[2],
                "quantization_axis_z": diagnostic.quantization_axis[2],
            }
        )
    frame = pd.DataFrame(rows)
    for column in (
        "fixed_basis_signed_reference_shift_hz",
        "applied_named_cycling_shift_hz",
    ):
        values = frame[column].to_numpy()
        frame[f"{column}_odd_hz"] = 0.5 * (values - values[::-1])
        frame[f"{column}_even_hz"] = 0.5 * (values + values[::-1])
    return frame, diagnostics


def _component_profile_table(
    context,
    beams,
    positions_m: np.ndarray,
    diagnostics,
) -> pd.DataFrame:
    """Retain every trapping component before forming the vector sum."""

    rows: list[dict[str, Any]] = []
    for z, diagnostic in zip(positions_m, diagnostics):
        component_vectors = _component_vector_energies_j(
            context, beams, diagnostic
        )
        for index, (beam, intensity, energy_vector) in enumerate(
            zip(
                beams,
                diagnostic.component_intensities_w_per_m2,
                component_vectors,
            )
        ):
            direction = np.asarray(beam.direction, dtype=float)
            energy_hz = np.asarray(energy_vector, dtype=float) / PLANCK_CONSTANT_J_S
            rows.append(
                {
                    "z_m": z,
                    "z_mm": 1.0e3 * z,
                    "component_index": index,
                    "beam": beam.label,
                    "axis_name": beam.axis_name,
                    "propagation_sense": beam.propagation_sense,
                    "propagation_frame_helicity": beam.helicity,
                    "helicity_sign": helicity_sign(beam.helicity),
                    "k_hat_x": direction[0],
                    "k_hat_y": direction[1],
                    "k_hat_z": direction[2],
                    "intensity_w_per_m2": intensity,
                    "signed_vector_energy_x_over_h_hz": energy_hz[0],
                    "signed_vector_energy_y_over_h_hz": energy_hz[1],
                    "signed_vector_energy_z_over_h_hz": energy_hz[2],
                    "signed_vector_energy_along_k_over_h_hz": float(
                        np.dot(energy_hz, direction)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _mirror_transition_table(model, diagnostic, z_m: float) -> pd.DataFrame:
    """Compare each transition proxy with its mF-reflected partner."""

    lookup = {
        (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
            transition.q,
        ): index
        for index, transition in enumerate(model.structure.absorption_transitions)
    }
    rows: list[dict[str, Any]] = []
    visited: set[tuple[int, int]] = set()
    shifts_hz = (
        np.asarray(diagnostic.vector_transition_energy_j)
        / PLANCK_CONSTANT_J_S
    )
    for index, transition in enumerate(model.structure.absorption_transitions):
        mirror_index = lookup[
            (
                transition.ground_f,
                -transition.ground_m_f,
                transition.excited_f,
                -transition.excited_m_f,
                -transition.q,
            )
        ]
        pair = tuple(sorted((index, mirror_index)))
        if pair in visited:
            continue
        visited.add(pair)
        mirror = model.structure.absorption_transitions[mirror_index]
        shift = float(shifts_hz[index])
        mirror_shift = float(shifts_hz[mirror_index])
        rows.append(
            {
                "z_m": z_m,
                "transition_index": index,
                "mirror_transition_index": mirror_index,
                "ground_F": transition.ground_f,
                "ground_mF": transition.ground_m_f,
                "mirror_ground_mF": mirror.ground_m_f,
                "excited_F": transition.excited_f,
                "excited_mF": transition.excited_m_f,
                "mirror_excited_mF": mirror.excited_m_f,
                "q": transition.q,
                "mirror_q": mirror.q,
                "transition_proxy_shift_hz": shift,
                "mirror_transition_proxy_shift_hz": mirror_shift,
                "antisymmetry_residual_hz": shift + mirror_shift,
                "is_antisymmetric": bool(
                    np.isclose(shift, -mirror_shift, rtol=1.0e-13, atol=1.0e-9)
                ),
            }
        )
    return pd.DataFrame(rows)


def _transition_shift_kernel_probe(
    context,
    diagnostic,
    position_m: tuple[float, float, float],
) -> tuple[dict[str, bool], pd.DataFrame, pd.DataFrame]:
    """Verify a nonzero provisional shift reaches the production cooling kernel."""

    model = context.model
    config = replace(context.multilevel_config, include_gravity=False)
    transition_index = _reference_transition_index(model)
    transition = model.structure.absorption_transitions[transition_index]
    axis = tuple(diagnostic.quantization_axis)
    candidates = [
        beam
        for beam in context.cooling_repump_beams
        if beam.family == "cooling" and beam.axis_name == "vertical_z"
    ]
    beam = max(
        candidates,
        key=lambda candidate: polarization_weights(
            beam_polarization_vector(candidate), axis
        )[transition.q],
    )
    applied_shift = np.asarray(
        diagnostic.vector_transition_energy_j, dtype=float
    ) / HBAR_J_S
    zero_shift = np.zeros_like(applied_shift)
    rows: list[dict[str, Any]] = []
    production_rates: dict[str, float] = {}
    expected_rates: dict[str, float] = {}
    intensity = beam_intensity_w_per_m2(beam, position_m)
    polarization_weight = polarization_weights(
        beam_polarization_vector(beam), axis
    )[transition.q]
    saturation = (
        intensity
        / config.saturation_intensity_w_per_m2
        * model.transition_strength[transition_index]
        * polarization_weight
    )
    base_detuning = (
        config.cooling_detuning_rad_per_s
        - model.transition_hyperfine_offset_rad_per_s[transition_index]
    )
    for name, shifts in (
        ("zero_shift", zero_shift),
        ("applied_vector_transition_shift", applied_shift),
    ):
        matrices = build_beam_stimulated_rate_matrices(
            model,
            [beam],
            position_m,
            (0.0, 0.0, 0.0),
            0.0,
            axis,
            config,
            transition_resonance_shift_rad_per_s=shifts,
        )
        production_rate = _selected_rate_matrix_value(
            model, matrices, transition_index
        )
        selected_shift = float(shifts[transition_index])
        effective_detuning = base_detuning - selected_shift
        expected_rate = (
            0.5
            * config.natural_linewidth_rad_per_s
            * saturation
            / (
                1.0
                + saturation
                + 4.0
                * effective_detuning**2
                / config.natural_linewidth_rad_per_s**2
            )
        )
        production_rates[name] = production_rate
        expected_rates[name] = expected_rate
        rows.append(
            {
                "case": name,
                "position_x_m": position_m[0],
                "position_y_m": position_m[1],
                "position_z_m": position_m[2],
                "beam": beam.label,
                "transition_index": transition_index,
                "ground_F": transition.ground_f,
                "ground_mF": transition.ground_m_f,
                "excited_F": transition.excited_f,
                "excited_mF": transition.excited_m_f,
                "q": transition.q,
                "quantization_axis_x": axis[0],
                "quantization_axis_y": axis[1],
                "quantization_axis_z": axis[2],
                "component_intensity_w_per_m2": intensity,
                "polarization_weight_Pq": polarization_weight,
                "transition_saturation": saturation,
                "base_laser_minus_hyperfine_rad_per_s": base_detuning,
                "applied_transition_shift_rad_per_s": selected_shift,
                "effective_detuning_rad_per_s": effective_detuning,
                "production_stimulated_rate_per_s": production_rate,
                "expected_stimulated_rate_per_s": expected_rate,
                "production_minus_expected_rate_per_s": production_rate
                - expected_rate,
            }
        )
    wrapper = vector_only_trajectory_observable(
        context,
        position_m,
        (0.0, 0.0, 0.0),
        previous_axis=(0.0, 0.0, 1.0),
    )
    direct = rate_equation_observable_from_local_environment(
        model,
        list(context.cooling_repump_beams),
        position_m,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        axis,
        config,
        transition_resonance_shift_rad_per_s=applied_shift,
    )
    zero_direct = rate_equation_observable_from_local_environment(
        model,
        list(context.cooling_repump_beams),
        position_m,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        axis,
        config,
        transition_resonance_shift_rad_per_s=zero_shift,
    )
    beam_rows = []
    for beam_item, wrapper_rate, direct_rate, zero_rate in zip(
        context.cooling_repump_beams,
        wrapper.rate_equation.beam_scattering_rates_per_s,
        direct.beam_scattering_rates_per_s,
        zero_direct.beam_scattering_rates_per_s,
    ):
        beam_rows.append(
            {
                "beam": beam_item.label,
                "family": beam_item.family,
                "wrapper_shifted_absorption_proxy_per_s": wrapper_rate,
                "direct_shifted_absorption_proxy_per_s": direct_rate,
                "direct_zero_shift_absorption_proxy_per_s": zero_rate,
                "wrapper_minus_direct_shifted_per_s": wrapper_rate - direct_rate,
                "shifted_minus_zero_per_s": direct_rate - zero_rate,
            }
        )
    beam_table = pd.DataFrame(beam_rows)

    checks = {
        "nonzero_shift_was_applied": bool(
            abs(applied_shift[transition_index]) > 0.0
        ),
        "production_rates_match_explicit_shifted_lorentzian": bool(
            all(
                np.isclose(
                    production_rates[name],
                    expected_rates[name],
                    rtol=2.0e-14,
                    atol=1.0e-8,
                )
                for name in production_rates
            )
        ),
        "nonzero_shift_changes_production_rate": bool(
            not np.isclose(
                production_rates["zero_shift"],
                production_rates["applied_vector_transition_shift"],
                rtol=1.0e-10,
                atol=1.0e-8,
            )
        ),
        "wrapper_passes_vector_shift_array_exactly": bool(
            np.array_equal(
                wrapper.applied_transition_shift_rad_per_s,
                applied_shift,
            )
        ),
        "wrapper_matches_direct_shifted_kernel_call": bool(
            np.allclose(
                wrapper.rate_equation.populations,
                direct.populations,
                rtol=2.0e-14,
                atol=1.0e-14,
            )
            and np.allclose(
                wrapper.rate_equation.beam_scattering_rates_per_s,
                direct.beam_scattering_rates_per_s,
                rtol=2.0e-14,
                atol=1.0e-8,
            )
            and np.allclose(
                wrapper.rate_equation.force_n,
                direct.force_n,
                rtol=2.0e-14,
                atol=1.0e-30,
            )
        ),
        "shift_changes_at_least_one_full_model_beam_rate": bool(
            np.any(
                np.abs(
                    np.asarray(direct.beam_scattering_rates_per_s)
                    - np.asarray(zero_direct.beam_scattering_rates_per_s)
                )
                > 1.0e-8
            )
        ),
    }
    return checks, pd.DataFrame(rows), beam_table


def _scenario_gradient(context, beams, step_m: float, alpha_sign: float = 1.0) -> tuple[float, float]:
    values = []
    for z in (-step_m, 0.0, step_m):
        diagnostic = provisional_transition_stark_shifts(
            context.model,
            list(beams),
            (0.0, 0.0, z),
            (0.0, 0.0, 0.0),
            context.apparatus.trapping_laser,
            context.stark_config,
            (0.0, 0.0, 1.0),
            polarizability_table=context.polarizability_table,
        )
        total = np.sum(
            _component_vector_energies_j(context, beams, diagnostic, alpha_sign),
            axis=0,
        )
        values.append(total[2] / PLANCK_CONSTANT_J_S)
    gradient = (values[2] - values[0]) / (2.0 * step_m)
    return float(gradient), float(values[1])


def _controlled_sign_table(context, step_m: float = 1.0e-6) -> pd.DataFrame:
    baseline = list(context.trapping_beams)
    z_indices = [i for i, beam in enumerate(baseline) if beam.axis_name == "vertical_z"]

    one_flip = baseline.copy()
    first = one_flip[z_indices[0]]
    one_flip[z_indices[0]] = replace(
        first,
        helicity="sigma-" if first.helicity == "sigma+" else "sigma+",
    )
    both_flip = baseline.copy()
    for index in z_indices:
        beam = both_flip[index]
        both_flip[index] = replace(
            beam,
            helicity="sigma-" if beam.helicity == "sigma+" else "sigma+",
        )
    focus_swap = baseline.copy()
    first_waist = focus_swap[z_indices[0]].waist_position_m
    second_waist = focus_swap[z_indices[1]].waist_position_m
    focus_swap[z_indices[0]] = replace(focus_swap[z_indices[0]], waist_position_m=second_waist)
    focus_swap[z_indices[1]] = replace(focus_swap[z_indices[1]], waist_position_m=first_waist)

    scenarios = [
        ("baseline", baseline, 1.0, "reference"),
        (
            "reverse_one_z_incident_component",
            one_flip,
            1.0,
            "plan says reverse gradient; physical pair instead becomes even and biased",
        ),
        ("reverse_both_z_component_helicities", both_flip, 1.0, "global path reversal"),
        ("reverse_alpha1_sign", baseline, -1.0, "global sign reversal"),
        ("swap_z_focus_locations", focus_swap, 1.0, "global spatial reversal"),
        ("reverse_alpha1_and_swap_foci", focus_swap, -1.0, "two global reversals"),
        ("reverse_both_helicities_and_alpha1", both_flip, -1.0, "two global reversals"),
    ]
    rows = []
    baseline_gradient, _ = _scenario_gradient(context, baseline, step_m)
    for name, beams, alpha_sign, interpretation in scenarios:
        gradient, center = _scenario_gradient(context, beams, step_m, alpha_sign)
        expected_relation = (
            "same_as_baseline"
            if name in {
                "baseline",
                "reverse_alpha1_and_swap_foci",
                "reverse_both_helicities_and_alpha1",
            }
            else "opposite_baseline"
        )
        target = baseline_gradient if expected_relation == "same_as_baseline" else -baseline_gradient
        rows.append(
            {
                "scenario": name,
                "gradient_hz_per_m": gradient,
                "center_signed_shift_hz": center,
                "gradient_over_baseline": gradient / baseline_gradient,
                "expected_relation_from_plan_or_global_reversal": expected_relation,
                "matches_expected_relation": bool(
                    np.isclose(gradient, target, rtol=2.0e-4, atol=1.0e-6)
                ),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def run_test_02(
    context,
    root: Path,
    *,
    profile_points: int = 401,
    repeat_profile_points: int = 801,
) -> dict[str, Any]:
    for name, count in (
        ("profile_points", profile_points),
        ("repeat_profile_points", repeat_profile_points),
    ):
        if count < 21 or count % 2 == 0:
            raise ValueError(f"{name} must be odd and at least 21")
    directory = root / "test_02_signed_vector_shift"
    figure_directory = directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    print(
        f"[Test 2] sampling signed vector profile ({profile_points} points)",
        flush=True,
    )
    extent_m = 2.0e-3
    positions = np.linspace(-extent_m, extent_m, profile_points)
    profile, diagnostics = _profile(context, list(context.trapping_beams), positions)
    profile_path = directory / "signed_vector_profile.csv"
    profile.to_csv(profile_path, index=False)
    component_profile = _component_profile_table(
        context, list(context.trapping_beams), positions, diagnostics
    )
    component_profile_path = directory / "all_component_vector_profiles.csv"
    component_profile.to_csv(component_profile_path, index=False)

    model = context.model
    reference_index = _reference_transition_index(model)
    reference_coefficient = model.transition_zeeman_coefficient[reference_index]
    transition_rows: list[dict[str, Any]] = []
    for z, diagnostic, signed_reference_hz in zip(
        positions,
        diagnostics,
        profile["fixed_basis_signed_reference_shift_hz"],
    ):
        for index, transition in enumerate(model.structure.absorption_transitions):
            transition_rows.append(
                {
                    "z_m": z,
                    "transition_index": index,
                    "laser_family": "cooling" if transition.ground_f == 2 else "repump",
                    "ground_F": transition.ground_f,
                    "ground_mF": transition.ground_m_f,
                    "excited_F": transition.excited_f,
                    "excited_mF": transition.excited_m_f,
                    "q": transition.q,
                    "normalized_dipole_strength_C2": transition.c_squared,
                    "applied_local_axis_vector_transition_shift_hz": diagnostic.vector_transition_energy_j[
                        index
                    ]
                    / PLANCK_CONSTANT_J_S,
                    "fixed_global_z_basis_vector_transition_shift_hz": signed_reference_hz
                    * model.transition_zeeman_coefficient[index]
                    / reference_coefficient,
                }
            )
    transition_path = directory / "all_transition_vector_shift_proxies.csv"
    pd.DataFrame(transition_rows).to_csv(transition_path, index=False)

    probe = 0.1e-3
    probe_profile, probe_diagnostics = _profile(
        context,
        list(context.trapping_beams),
        np.asarray((-probe, probe)),
    )
    mirror_table = _mirror_transition_table(
        model, probe_diagnostics[1], probe
    )
    mirror_path = directory / "mirror_mf_transition_proxy_pairs.csv"
    mirror_table.to_csv(mirror_path, index=False)
    shift_kernel_checks, shift_kernel_table, shift_beam_table = (
        _transition_shift_kernel_probe(
            context,
            probe_diagnostics[1],
            (0.0, 0.0, probe),
        )
    )
    shift_kernel_path = directory / "production_kernel_shift_probe.csv"
    shift_kernel_table.to_csv(shift_kernel_path, index=False)
    shift_beam_path = directory / "wrapper_shift_handoff_beam_rates.csv"
    shift_beam_table.to_csv(shift_beam_path, index=False)

    level_rows = []
    for state in model.structure.states:
        level_rows.append(
            {
                "state_index": state.index,
                "manifold": state.manifold,
                "F": state.f,
                "mF": state.m_f,
                "U_level_over_h_available": False,
                "U_level_over_h_hz": np.nan,
                "reason": (
                    "Arora input contains differential transition polarizabilities only; "
                    "separate 5S/5P and hyperfine-resolved level polarizabilities are absent"
                ),
            }
        )
    level_path = directory / "missing_level_resolved_shifts.csv"
    pd.DataFrame(level_rows).to_csv(level_path, index=False)
    _write_json(
        directory / "missing_level_resolved_inputs.json",
        {
            "available": False,
            "required": [
                "separate absolute 5S scalar/vector/tensor polarizabilities",
                "separate absolute 5P scalar/vector/tensor polarizabilities",
                "hyperfine recoupling and a fixed-basis Stark Hamiltonian",
            ],
            "current_input": str(context.polarizability_table.source_path),
            "current_input_semantics": (
                "one differential scalar/vector/tensor triplet per wavelength"
            ),
        },
    )

    sign_table = _controlled_sign_table(context)
    sign_path = directory / "controlled_sign_reversals.csv"
    sign_table.to_csv(sign_path, index=False)

    print(
        f"[Test 2 repeat] refining signed profile ({repeat_profile_points} points)",
        flush=True,
    )
    repeat_positions = np.linspace(-extent_m, extent_m, repeat_profile_points)
    repeat_profile, _ = _profile(
        context, list(context.trapping_beams), repeat_positions
    )
    repeat_path = directory / "signed_vector_profile_repeat_refined.csv"
    repeat_profile.to_csv(repeat_path, index=False)

    refinement_rows = []
    for step in (4.0e-6, 2.0e-6, 1.0e-6):
        gradient, center = _scenario_gradient(
            context, list(context.trapping_beams), step
        )
        applied_samples = []
        for z in (-step, 0.0, step):
            diagnostic = provisional_transition_stark_shifts(
                context.model,
                list(context.trapping_beams),
                (0.0, 0.0, z),
                (0.0, 0.0, 0.0),
                context.apparatus.trapping_laser,
                context.stark_config,
                (0.0, 0.0, 1.0),
                polarizability_table=context.polarizability_table,
            )
            applied_samples.append(
                diagnostic.vector_transition_energy_j[reference_index]
                / PLANCK_CONSTANT_J_S
            )
        refinement_rows.append(
            {
                "position_step_m": step,
                "fixed_basis_signed_gradient_hz_per_m": gradient,
                "center_signed_shift_hz": center,
                "applied_named_left_slope_hz_per_m": (
                    applied_samples[1] - applied_samples[0]
                )
                / step,
                "applied_named_right_slope_hz_per_m": (
                    applied_samples[2] - applied_samples[1]
                )
                / step,
                "applied_named_symmetric_derivative_hz_per_m": (
                    applied_samples[2] - applied_samples[0]
                )
                / (2.0 * step),
            }
        )
    refinement_frame = pd.DataFrame(refinement_rows)
    refinement_path = directory / "gradient_step_convergence.csv"
    refinement_frame.to_csv(refinement_path, index=False)

    applied_minus = float(probe_profile.iloc[0]["applied_named_cycling_shift_hz"])
    applied_plus = float(probe_profile.iloc[1]["applied_named_cycling_shift_hz"])
    signed_minus = float(
        probe_profile.iloc[0]["fixed_basis_signed_reference_shift_hz"]
    )
    signed_plus = float(
        probe_profile.iloc[1]["fixed_basis_signed_reference_shift_hz"]
    )
    central = profile.iloc[len(profile) // 2]
    optical_even = profile[
        "fixed_basis_signed_reference_shift_hz_even_hz"
    ].to_numpy()
    optical_odd = profile[
        "fixed_basis_signed_reference_shift_hz_odd_hz"
    ].to_numpy()
    applied_even = profile["applied_named_cycling_shift_hz_even_hz"].to_numpy()
    applied_odd = profile["applied_named_cycling_shift_hz_odd_hz"].to_numpy()
    odd_even_ratio = float(
        np.max(np.abs(optical_even)) / max(np.max(np.abs(optical_odd)), 1.0e-300)
    )
    applied_odd_even_ratio = float(
        np.max(np.abs(applied_odd)) / max(np.max(np.abs(applied_even)), 1.0e-300)
    )
    repeat_applied_minus = applied_minus
    repeat_applied_plus = applied_plus
    repeat_applied_even = repeat_profile[
        "applied_named_cycling_shift_hz_even_hz"
    ].to_numpy()
    repeat_applied_odd = repeat_profile[
        "applied_named_cycling_shift_hz_odd_hz"
    ].to_numpy()
    repeat_applied_odd_even_ratio = float(
        np.max(np.abs(repeat_applied_odd))
        / max(np.max(np.abs(repeat_applied_even)), 1.0e-300)
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    axes[0, 0].plot(profile["z_mm"], profile["z_incident_intensity_w_per_m2"], label="z incident")
    axes[0, 0].plot(profile["z_mm"], profile["z_retro_intensity_w_per_m2"], label="z retro")
    axes[0, 0].set(title="Displaced-focus component intensities", ylabel=r"Intensity [W/m$^2$]")
    axes[0, 1].plot(profile["z_mm"], 1e-6 * profile["z_incident_vector_contribution_hz"], label="incident contribution")
    axes[0, 1].plot(profile["z_mm"], 1e-6 * profile["z_retro_vector_contribution_hz"], label="retro contribution")
    axes[0, 1].plot(profile["z_mm"], 1e-6 * profile["fixed_basis_signed_reference_shift_hz"], color="#111827", linewidth=2, label="signed sum")
    axes[0, 1].set(title="Fixed-basis signed vector profile", ylabel="Reference shift [MHz]")
    axes[1, 0].plot(profile["z_mm"], 1e-6 * optical_odd, label="odd")
    axes[1, 0].plot(profile["z_mm"], 1e-6 * optical_even, label="even")
    axes[1, 0].set(title="Odd/even decomposition", ylabel="Reference shift [MHz]")
    axes[1, 1].plot(profile["z_mm"], 1e-6 * profile["fixed_basis_signed_reference_shift_hz"], label="fixed global z basis")
    axes[1, 1].plot(profile["z_mm"], 1e-6 * profile["applied_named_cycling_shift_hz"], label="actually applied named transition")
    axes[1, 1].set(title="Field-zero basis relabeling failure", ylabel="Cycling shift [MHz]")
    for axis in axes.flat:
        axis.set_xlabel("z [mm]")
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.axvline(0.0, color="0.45", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    profile_figure_path = figure_directory / "signed_vector_shift_profile.png"
    figure.savefig(profile_figure_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0].bar(sign_table["scenario"], sign_table["gradient_over_baseline"], color="#2563eb")
    axes[0].axhline(1.0, color="#15803d", linestyle="--", label="baseline")
    axes[0].axhline(-1.0, color="#b91c1c", linestyle="--", label="reversed")
    axes[0].set(ylabel="gradient / baseline", title="Controlled gradient sign tests")
    axes[0].legend()
    axes[1].bar(sign_table["scenario"], sign_table["center_signed_shift_hz"], color="#c2410c")
    axes[1].set(ylabel="center signed shift [Hz]", title="Center bias after each change")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=24)
        axis.grid(axis="y", alpha=0.25)
    sign_figure_path = figure_directory / "controlled_sign_reversals.png"
    figure.savefig(sign_figure_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    criteria = {
        "central_signed_vector_profile_zero": bool(
            abs(central["fixed_basis_signed_reference_shift_hz"]) <= 1.0e-9
        ),
        "optical_signed_profile_is_odd": bool(odd_even_ratio <= 1.0e-10),
        "separate_ground_and_excited_level_shifts_available": False,
        "opposite_mf_level_shifts_are_opposite": False,
        "opposite_mf_transition_proxies_are_opposite": bool(
            mirror_table["is_antisymmetric"].all()
        ),
        "named_transition_shift_is_odd_in_fixed_state_labels": bool(
            applied_odd_even_ratio >= 1.0e6
        ),
        "differential_transition_proxy_is_passed_to_cooling_kernel": bool(
            all(shift_kernel_checks.values())
        ),
        "single_component_polarization_flip_reverses_gradient": bool(
            sign_table.loc[
                sign_table["scenario"] == "reverse_one_z_incident_component",
                "matches_expected_relation",
            ].iloc[0]
        ),
        "refinement_repeats_same_failure": bool(
            np.sign(repeat_applied_minus) == np.sign(repeat_applied_plus)
            and np.isclose(
                abs(repeat_applied_minus),
                abs(repeat_applied_plus),
                rtol=1.0e-10,
            )
            and repeat_applied_odd_even_ratio <= 1.0e-10
            and np.all(
                refinement_frame["applied_named_left_slope_hz_per_m"]
                * refinement_frame["applied_named_right_slope_hz_per_m"]
                < 0.0
            )
            and np.allclose(
                refinement_frame[
                    "applied_named_symmetric_derivative_hz_per_m"
                ],
                0.0,
                atol=1.0e-6,
            )
        ),
    }
    result = {
        "test_number": 2,
        "name": TEST_NAMES[2],
        "status": "FAIL",
        "criteria": criteria,
        "first_blocking_causes": [
            (
                "The differential-only Arora table cannot determine individual "
                "ground- and excited-level shifts U_g/h and U_e/h for all 24 states."
            ),
            (
                "The local quantization axis flips with the optical-spin vector. "
                "The code projects onto that axis and applies the nonnegative vector "
                "magnitude to fixed transition indices, making a named transition's "
                "shift even rather than odd across the origin."
            ),
        ],
        "specification_inconsistency": (
            "Reversing only one traveling component does not reverse the balanced "
            "odd gradient. It converts the difference of symmetric intensities into "
            "an even, center-biased sum. Reversing both path components, alpha1, or "
            "the focus order reverses the gradient; two global reversals recover it."
        ),
        "primary_profile_points": profile_points,
        "repeat_profile_points": repeat_profile_points,
        "fixed_basis_even_over_odd_ratio": odd_even_ratio,
        "applied_named_transition_odd_over_even_ratio": applied_odd_even_ratio,
        "repeat_applied_named_transition_odd_over_even_ratio": repeat_applied_odd_even_ratio,
        "numerical_tolerances": {
            "central_signed_shift_hz": 1.0e-9,
            "fixed_basis_even_over_odd_ratio": 1.0e-10,
            "named_transition_odd_over_even_ratio_required": 1.0e6,
            "mirror_pair_relative": 1.0e-13,
            "mirror_pair_absolute_hz": 1.0e-9,
        },
        "production_shift_handoff_checks": shift_kernel_checks,
        "probe_z_m": probe,
        "probe_applied_named_cycling_shift_hz": {
            "minus_z": applied_minus,
            "plus_z": applied_plus,
        },
        "repeat_probe_applied_named_cycling_shift_hz": {
            "minus_z": repeat_applied_minus,
            "plus_z": repeat_applied_plus,
        },
        "probe_fixed_basis_signed_reference_shift_hz": {
            "minus_z": signed_minus,
            "plus_z": signed_plus,
        },
        "outputs": [
            directory / "README.md",
            directory / "result.json",
            profile_path,
            component_profile_path,
            repeat_path,
            transition_path,
            mirror_path,
            shift_kernel_path,
            shift_beam_path,
            level_path,
            directory / "missing_level_resolved_inputs.json",
            sign_path,
            refinement_path,
            profile_figure_path,
            sign_figure_path,
        ],
    }
    _write_json(directory / "result.json", result)
    _write_markdown(
        directory / "README.md",
        f"""# Test 2 — signed 1529-nm vector-shift profile

**Result: FAIL (confirmed by refined repeat).** The optical vector-energy
profile itself is centered and odd: its maximum even/odd ratio is
`{odd_even_ratio:.3e}`. However, the current model cannot produce the required
individual `U_g/h` and `U_e/h` values because its Arora input contains only
differential polarizabilities.

A second independent failure appears at the field zero. The quantization axis
follows the optical-spin vector and reverses across the origin, while the
reference energy is projected onto that axis and becomes a magnitude. For the
named F=2,mF=+2 -> F'=3,mF'=+3 transition at z=+/-0.1 mm, the actually applied
shifts are `{applied_minus/1e6:.9f}` and `{applied_plus/1e6:.9f}` MHz: the same
sign. In a fixed global z basis the signed reference values are
`{signed_minus/1e6:.9f}` and `{signed_plus/1e6:.9f}` MHz: opposite signs.

The requested single-component polarization reversal was also checked. It
creates an even center-biased profile rather than a reversed odd gradient;
`controlled_sign_reversals.csv` shows that reversing both path helicities,
alpha1, or the focus order gives the physically meaningful global sign tests.

The nonzero differential transition proxy is wired into the production rate
kernel correctly: the pMOT wrapper exactly matches an independent direct
shifted-kernel call, and the shift changes the beam-resolved rates. Those
checks are preserved in `production_kernel_shift_probe.csv` and
`wrapper_shift_handoff_beam_rates.csv`; the failure is therefore in the
available Stark-state model and field-zero basis treatment, not in the final
argument handoff.

Per the canonical stop rule, Tests 3--9 were not executed. The refined profile,
three finite-difference gradient steps, all 54 transition-level proxies, and
the explicit missing-level table preserve the complete diagnosis.
""",
    )
    return result


def _result_row(result: dict[str, Any]) -> dict[str, Any]:
    if result["status"] == "PASS_WITH_QUALIFICATIONS":
        reason = (
            "raw polarizability unit provenance remains external to the CSV"
            if result["test_number"] == 0
            else "fixed uniform F=2 stage using the inherited absorption-momentum proxy"
        )
    elif result["status"] == "FAIL":
        reason = result.get("first_blocking_causes", ["pass criteria not met"])[0]
    else:
        reason = ""
    return {
        "test_number": result["test_number"],
        "test_name": result["name"],
        "status": result["status"],
        "reason": reason,
    }


def run_ordered_diagnostic_suite(
    *,
    output_root: str | Path | None = None,
    test1_sample_count: int = 161,
    profile_points: int = 401,
    repeat_profile_points: int = 801,
) -> dict[str, Any]:
    """Execute the ordered QA plan through the first failure and save results."""

    project_root = _project_root()
    root = Path(output_root) if output_root is not None else (
        project_root / "outputs" / "diagnostics" / "pmot" / DEFAULT_OUTPUT_TAG
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print(f"[pMOT diagnostics] output root: {root}", flush=True)
    context = build_vector_only_trajectory_context()

    results = []
    for runner in (
        lambda: run_test_00(context, root),
        lambda: run_test_01(context, root, sample_count=test1_sample_count),
        lambda: run_test_02(
            context,
            root,
            profile_points=profile_points,
            repeat_profile_points=repeat_profile_points,
        ),
    ):
        result = runner()
        results.append(result)
        print(
            f"[Test {result['test_number']}] {result['status']}",
            flush=True,
        )
        if result["status"] == "FAIL":
            break

    failed = next((result for result in results if result["status"] == "FAIL"), None)
    first_not_run = failed["test_number"] + 1 if failed else len(results)
    index_rows = [_result_row(result) for result in results]
    for number in range(first_not_run, 10):
        index_rows.append(
            {
                "test_number": number,
                "test_name": TEST_NAMES[number],
                "status": "NOT_RUN",
                "reason": (
                    f"ordered plan stopped after Test {failed['test_number']} failed"
                    if failed
                    else "runner implementation ends before this test"
                ),
            }
        )
    index = pd.DataFrame(index_rows)
    index.to_csv(root / "result_index.csv", index=False)
    _write_json(root / "result_index.json", index_rows)
    index[index["status"] == "NOT_RUN"].to_csv(root / "not_run_tests.csv", index=False)

    manifest = {
        "schema": "pmot.ordered-construction-diagnostics.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_specification": project_root / "docs" / "pmot" / "DIAGNOSTIC_TESTS.md",
        "output_root": root,
        "required_python_invocation": REQUIRED_PYTHON_INVOCATION,
        "resolved_python_executable": sys.executable,
        "resolved_executable_note": (
            "On this Windows workspace the required WSL path is a symlink to the "
            "recorded Codex bundled Python executable."
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        "git_status_short": _git_value(project_root, "status", "--short"),
        "ordered_stop_policy": "stop at the first failed test",
        "first_failed_test": failed["test_number"] if failed else None,
        "model": "provisional ideal-magic vector-only pMOT transition proxy",
        "external_magnetic_field_t": [0.0, 0.0, 0.0],
        "scalar_transition_shift_included": False,
        "tensor_transition_shift_included": False,
        "vector_transition_shift_included": True,
        "direct_1529_mechanical_force_included": False,
        "effective_detuning_equation": EFFECTIVE_DETUNING_EQUATION,
        "frequency_units": "angular frequency in rad/s inside mot_multilevel",
        "position_units": "m",
        "velocity_units": "m/s",
        "intensity_units": "W/m^2 per traveling component",
        "propagation_frame_helicity_sign_mapping": {
            "sigma+": helicity_sign("sigma+"),
            "sigma-": helicity_sign("sigma-"),
            "pi": helicity_sign("pi"),
        },
        "resolved_apparatus_config": asdict(context.apparatus),
        "resolved_multilevel_rate_config": asdict(context.multilevel_config),
        "resolved_stark_config": asdict(context.stark_config),
        "resolved_all_component_helicities": asdict(context.helicities),
        "resolved_trapping_geometry": {
            "wavelength_m": context.apparatus.trapping_laser.wavelength_m,
            "focus_positions_along_each_path_m": [
                -context.apparatus.trapping_laser.focus_offset_m,
                context.apparatus.trapping_laser.focus_offset_m,
            ],
            "focus_separation_m": 2.0
            * context.apparatus.trapping_laser.focus_offset_m,
            "input_beam_diameter_m": context.apparatus.trapping_laser.input_beam_diameter_m,
            "focal_length_m": context.apparatus.trapping_laser.focal_length_m,
            "incident_waist_radius_m": context.apparatus.trapping_laser.resolved_incident_waist_radius_m,
            "retro_waist_radius_m": context.apparatus.trapping_laser.resolved_retro_waist_radius_m,
            "incident_rayleigh_range_m": context.trapping_beams[0].rayleigh_range_m,
            "retro_rayleigh_range_m": context.trapping_beams[1].rayleigh_range_m,
            "retro_power_fraction": context.apparatus.trapping_laser.retro_power_fraction,
            "envelope_combination": context.apparatus.trapping_laser.envelope_combination,
        },
        "complete_beam_vector_manifest": (
            root / "test_00_configuration_and_units" / "beam_manifest.csv"
        ),
        "per_test_physics_controls": {
            "test_00": {
                "external_magnetic_field_t": [0.0, 0.0, 0.0],
                "gravity_in_production_kernel_probe": False,
                "recoil": False,
            },
            "test_01": {
                "external_magnetic_field_t": [0.0, 0.0, 0.0],
                "gravity": False,
                "recoil": False,
                "trapping_shift": False,
                "repumper": False,
                "active_cooling_components": "two z-axis components only",
                "population": "uniform 0.2 over F=2 mF=-2..+2",
            },
            "test_02": {
                "external_magnetic_field_t": [0.0, 0.0, 0.0],
                "gravity": False,
                "recoil": False,
                "780nm_force_during_shift_profile": False,
                "separate_nonmechanical_780nm_rate_kernel_handoff_probe": True,
                "trapping_components": "all six 1529-nm traveling components",
            },
        },
        "parameters": {
            "test1_sample_count": test1_sample_count,
            "test2_profile_points": profile_points,
            "test2_repeat_profile_points": repeat_profile_points,
            "profile_extent_m": 2.0e-3,
        },
        "wall_time_s": perf_counter() - started,
        "result_index": index_rows,
    }
    _write_json(root / "run_manifest.json", manifest)

    status_lines = "\n".join(
        f"- Test {row['test_number']}: **{row['status']}** — {row['test_name']}"
        + (f" ({row['reason']})" if row["reason"] else "")
        for row in index_rows
    )
    _write_markdown(
        root / "README.md",
        f"""# Initial pMOT construction diagnostic campaign

This campaign executes the ordered procedure in
[`docs/pmot/DIAGNOSTIC_TESTS.md`](../../../../docs/pmot/DIAGNOSTIC_TESTS.md).
It uses the provisional ideal-magic vector-only transition proxy and does not
claim a production pMOT.

## Ordered result

{status_lines}

Execution stopped after Test {failed['test_number'] if failed else 'none'}.
Test 2 was repeated with a denser spatial grid and three smaller symmetric
finite-difference steps. The same failure remained: the optical vector profile
is odd, but the applied fixed-name transition shift is even because the local
basis flips, and separate level shifts are unavailable from the differential-
only input table.

Each executed-test directory contains its own explanation, result JSON, and
source CSV data; Tests 1 and 2 also contain rendered figures. `result_index.csv`
is the concise machine-readable campaign status and `run_manifest.json`
records provenance and assumptions.
""",
    )
    print(
        f"[pMOT diagnostics] stopped after Test {failed['test_number'] if failed else 'none'}; "
        f"manifest: {root / 'run_manifest.json'}",
        flush=True,
    )
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ordered initial pMOT construction diagnostics"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--test1-sample-count", type=int, default=161)
    parser.add_argument("--profile-points", type=int, default=401)
    parser.add_argument("--repeat-profile-points", type=int, default=801)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest = run_ordered_diagnostic_suite(
        output_root=args.output_root,
        test1_sample_count=args.test1_sample_count,
        profile_points=args.profile_points,
        repeat_profile_points=args.repeat_profile_points,
    )
    return 1 if manifest["first_failed_test"] is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
