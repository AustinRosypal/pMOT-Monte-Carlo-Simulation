"""Run and save short provisional no-coil pMOT Stark diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..configuration import RB87_MASS_KG
from ..mot_multilevel.configuration import default_multilevel_mot_config
from ..mot_multilevel.rate_equations import RateEquationAtomState
from ..mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from ..mot_multilevel.rate_equations import build_rate_equation_model
from .ac_stark import EFFECTIVE_DETUNING_EQUATION
from .ac_stark import PROVISIONAL_MODEL_NAME
from .ac_stark import ProvisionalStarkConfig
from .ac_stark import build_physics_trapping_beams
from .ac_stark import provisional_power_for_target_gradient_w_per_path
from .configuration import build_pmot_cooling_and_repump_beams
from .configuration import default_pmot_apparatus_config
from .configuration import pmot_paths
from .polarizability import load_differential_polarizability_table
from .stark_trajectories import ProvisionalPMOTTrajectoryRecord
from .stark_trajectories import provisional_pmot_observable
from .stark_trajectories import simulate_provisional_pmot_trajectory


ATOM_FRAME_DOPPLER_EQUATION = (
    "nu'_trap,j = nu_trap,j * (1 - k_hat_j dot v / c); "
    "lambda'_trap,j = lambda_trap,j / (1 - k_hat_j dot v / c)"
)
STARK_DECOMPOSITION_EQUATION = (
    "DeltaE_AC = DeltaE_scalar + DeltaE_vector + DeltaE_tensor; "
    "delta_nu_AC = DeltaE_AC / h; delta_omega_AC = DeltaE_AC / hbar"
)
PROVISIONAL_LIMITATION = (
    "The CSV contains only differential fine-structure-rank data. Scalar shifts "
    "are applied directly; vector shifts use a stretched-channel Zeeman-like "
    "rescaling; tensor shifts use an excited-state angular proxy. This is not a "
    "unique 24-state Stark Hamiltonian."
)
OMITTED_PHYSICS = (
    "conservative Stark-gradient force",
    "1529-nm scattering, heating, and loss",
    "coherent standing-wave interference",
    "window and mirror polarization transformations",
    "full local Stark-operator diagonalization and nonadiabatic zero crossing",
)
INHERITED_RATE_KERNEL_LIMITATION = (
    "The authoritative multilevel kernel uses saturated per-transition rates, "
    "explicit bidirectional stimulated population links, and a ground-population-"
    "weighted absorption rate for radiation pressure and diffusion. That inherited "
    "closure has not yet been validated against a consistent two-level limit or the "
    "event engine, so its force magnitude and sign cannot support a quantitative "
    "pMOT trapping claim."
)


def _cycling_transition_index(model) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("missing stretched cycling transition")


def summarize_trajectory(record: ProvisionalPMOTTrajectoryRecord) -> dict[str, object]:
    """Return short-run boundedness and diagnostic extrema."""

    base = record.rate_equation
    positions = np.asarray(base.positions_m, dtype=float)
    velocities = np.asarray(base.velocities_m_per_s, dtype=float)
    forces = np.asarray(base.forces_n, dtype=float)
    radii = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    effective_fields = np.linalg.norm(
        np.asarray(record.effective_fields_t, dtype=float),
        axis=1,
    )
    return {
        "termination_reason": base.termination_reason,
        "elapsed_s": float(base.times_s[-1]),
        "sample_count": len(base.times_s),
        "initial_position_m": list(base.positions_m[0]),
        "initial_velocity_m_per_s": list(base.velocities_m_per_s[0]),
        "final_position_m": list(base.positions_m[-1]),
        "final_velocity_m_per_s": list(base.velocities_m_per_s[-1]),
        "final_radius_m": float(radii[-1]),
        "maximum_radius_m": float(np.max(radii)),
        "final_speed_m_per_s": float(speeds[-1]),
        "maximum_speed_m_per_s": float(np.max(speeds)),
        "remained_inside_2mm_during_short_run": bool(np.all(radii <= 2.0e-3)),
        "mean_kernel_ground_weighted_absorption_rate_per_s": float(
            np.mean(base.total_scattering_rates_per_s)
        ),
        "maximum_force_n": float(np.max(np.linalg.norm(forces, axis=1))),
        "maximum_effective_field_proxy_g": float(np.max(effective_fields) * 1.0e4),
        "reference_shift_range_mhz": [
            float(np.min(record.reference_total_shift_hz) / 1.0e6),
            float(np.max(record.reference_total_shift_hz) / 1.0e6),
        ],
    }


def save_trajectory_csv(
    record: ProvisionalPMOTTrajectoryRecord,
    path: Path,
) -> Path:
    """Save the requested position/velocity/force/scattering histories."""

    base = record.rate_equation
    position = np.asarray(base.positions_m, dtype=float)
    velocity = np.asarray(base.velocities_m_per_s, dtype=float)
    force = np.asarray(base.forces_n, dtype=float)
    effective_field = np.asarray(record.effective_fields_t, dtype=float)
    axis = np.asarray(record.quantization_axes, dtype=float)
    frequencies = np.asarray(record.atom_frame_frequencies_hz, dtype=float)
    wavelengths = np.asarray(record.atom_frame_wavelengths_nm, dtype=float)
    dataframe = pd.DataFrame(
        {
            "time_s": base.times_s,
            "x_m": position[:, 0],
            "y_m": position[:, 1],
            "z_m": position[:, 2],
            "vx_m_per_s": velocity[:, 0],
            "vy_m_per_s": velocity[:, 1],
            "vz_m_per_s": velocity[:, 2],
            "fx_n": force[:, 0],
            "fy_n": force[:, 1],
            "fz_n": force[:, 2],
            "kernel_ground_weighted_absorption_rate_per_s": (
                base.total_scattering_rates_per_s
            ),
            "beff_x_t": effective_field[:, 0],
            "beff_y_t": effective_field[:, 1],
            "beff_z_t": effective_field[:, 2],
            "quantization_x": axis[:, 0],
            "quantization_y": axis[:, 1],
            "quantization_z": axis[:, 2],
            "reference_scalar_shift_hz": record.reference_scalar_shift_hz,
            "reference_vector_shift_hz": record.reference_vector_shift_hz,
            "reference_tensor_shift_hz": record.reference_tensor_shift_hz,
            "reference_total_shift_hz": record.reference_total_shift_hz,
        }
    )
    for component in range(wavelengths.shape[1]):
        dataframe[f"trap_component_{component:02d}_atom_frame_frequency_hz"] = (
            frequencies[:, component]
        )
        dataframe[f"trap_component_{component:02d}_atom_frame_wavelength_nm"] = (
            wavelengths[:, component]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def plot_trajectory_time_diagnostics(
    record: ProvisionalPMOTTrajectoryRecord,
    cooling_repump_beams,
    title: str,
    path: Path,
) -> Path:
    """Plot position, velocity, radiation force, and scattering versus time."""

    base = record.rate_equation
    times_ms = 1.0e3 * np.asarray(base.times_s, dtype=float)
    positions_mm = 1.0e3 * np.asarray(base.positions_m, dtype=float)
    velocities = np.asarray(base.velocities_m_per_s, dtype=float)
    forces_scaled = 1.0e21 * np.asarray(base.forces_n, dtype=float)
    beam_rates = np.asarray(base.beam_scattering_rates_per_s, dtype=float)
    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    figure, panels = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for component, (label, color) in enumerate(zip("xyz", colors)):
        panels[0, 0].plot(times_ms, positions_mm[:, component], color=color, label=label)
        panels[0, 1].plot(
            times_ms,
            velocities[:, component],
            color=color,
            label=f"v{label}",
        )
        panels[1, 0].plot(
            times_ms,
            forces_scaled[:, component],
            color=color,
            label=f"F{label}",
        )
    cooling_indices = [
        index for index, beam in enumerate(cooling_repump_beams) if beam.family == "cooling"
    ]
    repump_indices = [
        index for index, beam in enumerate(cooling_repump_beams) if beam.family == "repump"
    ]
    panels[1, 1].plot(
        times_ms,
        np.sum(beam_rates[:, cooling_indices], axis=1),
        label="cooling kernel absorption",
        color="#7c3aed",
    )
    panels[1, 1].plot(
        times_ms,
        np.sum(beam_rates[:, repump_indices], axis=1),
        label="repump kernel absorption",
        color="#c2410c",
    )
    panels[1, 1].plot(
        times_ms,
        base.total_scattering_rates_per_s,
        label="total kernel absorption",
        color="#111827",
        linestyle="--",
        linewidth=1.8,
    )
    panels[0, 0].set(title="Cartesian position", ylabel="Position [mm]")
    panels[0, 1].set(title="Cartesian velocity", ylabel="Velocity [m/s]")
    panels[1, 0].set(
        title="Inherited absorption-momentum force (gravity excluded)",
        ylabel="Force [10^-21 N]",
    )
    panels[1, 1].set(
        title="Kernel ground-weighted absorption rate",
        ylabel="Rate [s^-1]",
    )
    for panel in panels.flat:
        panel.set_xlabel("Time [ms]")
        panel.axhline(0.0, color="0.45", linewidth=0.7)
        panel.grid(alpha=0.24)
        panel.legend(loc="best", fontsize=8)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_trajectory_stark_diagnostics(
    record: ProvisionalPMOTTrajectoryRecord,
    title: str,
    path: Path,
) -> Path:
    """Plot the explicitly provisional shift decomposition and field proxy."""

    times_ms = 1.0e3 * np.asarray(record.rate_equation.times_s, dtype=float)
    shifts_mhz = 1.0e-6 * np.column_stack(
        (
            record.reference_scalar_shift_hz,
            record.reference_vector_shift_hz,
            record.reference_tensor_shift_hz,
            record.reference_total_shift_hz,
        )
    )
    fields_g = 1.0e4 * np.asarray(record.effective_fields_t, dtype=float)
    figure, panels = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for column, (label, color, style) in enumerate(
        (
            ("scalar", "#1d4ed8", "-"),
            ("vector", "#b91c1c", "-"),
            ("tensor proxy", "#15803d", "-"),
            ("total", "#111827", "--"),
        )
    ):
        panels[0].plot(
            times_ms,
            shifts_mhz[:, column],
            label=label,
            color=color,
            linestyle=style,
        )
    for component, (label, color) in enumerate(
        zip(("Bx proxy", "By proxy", "Bz proxy"), ("#b91c1c", "#1d4ed8", "#15803d"))
    ):
        panels[1].plot(times_ms, fields_g[:, component], label=label, color=color)
    panels[0].set(
        title="Stretched cycling-reference differential shift",
        xlabel="Time [ms]",
        ylabel="Shift [MHz]",
    )
    panels[1].set(
        title="Vector effective-field proxy (not external B)",
        xlabel="Time [ms]",
        ylabel="Field proxy [G]",
    )
    for panel in panels:
        panel.axhline(0.0, color="0.45", linewidth=0.7)
        panel.grid(alpha=0.24)
        panel.legend(loc="best", fontsize=8)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def sample_force_lineouts(
    model,
    cooling_repump_beams,
    trapping_beams,
    laser_config,
    stark_config,
    multilevel_config,
    polarizability_table,
    *,
    extent_m: float = 2.0e-3,
    sample_count: int = 81,
) -> dict[str, np.ndarray]:
    """Sample static Cartesian force curves for the provisional model."""

    coordinates = np.linspace(-extent_m, extent_m, sample_count)
    output: dict[str, np.ndarray] = {"coordinate_m": coordinates}
    for component, label in enumerate("xyz"):
        forces = []
        shifts = []
        for coordinate in coordinates:
            position = np.zeros(3)
            position[component] = coordinate
            observable = provisional_pmot_observable(
                model,
                cooling_repump_beams,
                trapping_beams,
                position,
                (0.0, 0.0, 0.0),
                laser_config,
                stark_config,
                multilevel_config,
                polarizability_table=polarizability_table,
            )
            forces.append(observable.rate_equation.force_n[component])
            shifts.append(
                observable.stark.transition_frequency_shift_hz[
                    _cycling_transition_index(model)
                ]
            )
        output[f"force_{label}_n"] = np.asarray(forces)
        output[f"reference_shift_{label}_hz"] = np.asarray(shifts)
    return output


def plot_force_lineouts(data: dict[str, np.ndarray], path: Path) -> Path:
    """Plot restoring-force and reference-shift curves with explicit origins."""

    coordinate_mm = 1.0e3 * data["coordinate_m"]
    zero_index = int(np.argmin(np.abs(coordinate_mm)))
    figure, panels = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    styles = (
        ("x", "#b91c1c", "-", "o", 0),
        ("y", "#1d4ed8", "--", "s", 4),
        ("z", "#15803d", ":", "^", 8),
    )
    for label, color, line_style, marker, marker_offset in styles:
        force = 1.0e21 * data[f"force_{label}_n"]
        shift = 1.0e-6 * data[f"reference_shift_{label}_hz"]
        panels[0].plot(
            coordinate_mm,
            force,
            label=label,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=(marker_offset, 18),
            markersize=4.5,
            markerfacecolor="white",
            linewidth=1.8,
        )
        panels[1].plot(
            coordinate_mm,
            shift,
            label=label,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=(marker_offset, 18),
            markersize=4.5,
            markerfacecolor="white",
            linewidth=1.8,
        )
        panels[0].scatter(
            [coordinate_mm[zero_index]],
            [force[zero_index]],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        panels[1].scatter(
            [coordinate_mm[zero_index]],
            [shift[zero_index]],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    panels[0].set(
        title="Stationary-atom same-axis 780-nm force proxy",
        xlabel="Axis coordinate [mm]",
        ylabel="Same-axis absorption-force proxy [10^-21 N]",
    )
    panels[1].set(
        title="Stretched cycling-transition resonance shift",
        xlabel="Axis coordinate [mm]",
        ylabel=r"AC transition shift $\delta\nu_{\rm AC}$ [MHz]",
    )
    for panel in panels:
        panel.axhline(0.0, color="0.4", linewidth=0.8)
        panel.axvline(0.0, color="0.4", linewidth=0.8)
        panel.grid(alpha=0.24)
        panel.legend(loc="best")
    figure.suptitle(
        "Provisional no-coil pMOT: AC-shifted 24-state absorption-force proxy\n"
        "1D zeros require a full-Jacobian check; gravity and 1529-nm forces excluded\n"
        "x, y, and z overlap in the symmetric model; right panel is a resonance shift, not intensity/potential"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _lineout_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    coordinates = data["coordinate_m"]
    center = int(np.argmin(np.abs(coordinates)))
    positive = center + 1
    negative = center - 1
    summary: dict[str, object] = {}
    for label in "xyz":
        forces = data[f"force_{label}_n"]
        slope = (forces[positive] - forces[negative]) / (
            coordinates[positive] - coordinates[negative]
        )
        summary[f"force_slope_{label}_n_per_m"] = float(slope)
        summary[f"linear_frequency_{label}_hz"] = (
            float(np.sqrt(-slope / RB87_MASS_KG) / (2.0 * np.pi))
            if slope < 0.0
            else None
        )
        roots = []
        for index in range(len(coordinates) - 1):
            left_force = forces[index]
            right_force = forces[index + 1]
            if left_force == 0.0:
                root = coordinates[index]
            elif left_force * right_force < 0.0:
                root = coordinates[index] - left_force * (
                    coordinates[index + 1] - coordinates[index]
                ) / (right_force - left_force)
            else:
                continue
            if not roots or abs(root - roots[-1]) > 1.0e-12:
                roots.append(float(root))
        summary[f"force_zero_crossings_{label}_m"] = roots
        summary[f"nonzero_axis_crossing_radius_{label}_m"] = (
            float(np.mean([abs(root) for root in roots if abs(root) > 1.0e-9]))
            if any(abs(root) > 1.0e-9 for root in roots)
            else None
        )
    summary["all_axes_locally_restoring"] = all(
        summary[f"force_slope_{label}_n_per_m"] < 0.0 for label in "xyz"
    )
    summary["axis_lineouts_establish_3d_stability"] = False
    return summary


def _static_force_vector_n(
    position_m,
    model,
    cooling_repump_beams,
    trapping_beams,
    laser_config,
    stark_config,
    multilevel_config,
    polarizability_table,
) -> np.ndarray:
    observable = provisional_pmot_observable(
        model,
        cooling_repump_beams,
        trapping_beams,
        position_m,
        (0.0, 0.0, 0.0),
        laser_config,
        stark_config,
        multilevel_config,
        polarizability_table=polarizability_table,
    )
    return np.asarray(observable.rate_equation.force_n, dtype=float)


def _positive_ray_equilibrium_coordinates_m(
    ray,
    force_function,
    *,
    minimum_coordinate_m: float = 0.05e-3,
    maximum_coordinate_m: float = 1.5e-3,
    scan_count: int = 81,
) -> list[float]:
    """Find all sampled nonzero force roots along ``position = coordinate * ray``."""

    ray_array = np.asarray(ray, dtype=float)
    ray_norm_squared = float(np.dot(ray_array, ray_array))
    if ray_array.shape != (3,) or ray_norm_squared <= 0.0:
        raise ValueError("ray must be a nonzero three-vector")

    def projected_force(coordinate_m: float) -> float:
        return float(
            np.dot(force_function(coordinate_m * ray_array), ray_array)
            / ray_norm_squared
        )

    coordinates = np.linspace(
        minimum_coordinate_m,
        maximum_coordinate_m,
        scan_count,
    )
    left_coordinate = float(coordinates[0])
    left_force = projected_force(left_coordinate)
    roots: list[float] = []
    for right_coordinate in coordinates[1:]:
        right_coordinate = float(right_coordinate)
        right_force = projected_force(right_coordinate)
        if left_force == 0.0:
            if not roots or abs(left_coordinate - roots[-1]) > 1.0e-10:
                roots.append(left_coordinate)
        if left_force * right_force < 0.0:
            lower = left_coordinate
            upper = right_coordinate
            lower_force = left_force
            for _ in range(48):
                midpoint = 0.5 * (lower + upper)
                midpoint_force = projected_force(midpoint)
                if lower_force * midpoint_force <= 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
                    lower_force = midpoint_force
            root = 0.5 * (lower + upper)
            if not roots or abs(root - roots[-1]) > 1.0e-10:
                roots.append(root)
        left_coordinate, left_force = right_coordinate, right_force
    if left_force == 0.0 and (
        not roots or abs(left_coordinate - roots[-1]) > 1.0e-10
    ):
        roots.append(left_coordinate)
    return roots


def _force_jacobian_n_per_m(
    position_m,
    force_function,
    *,
    step_m: float = 2.0e-6,
) -> np.ndarray:
    """Return a centered finite-difference Jacobian of the static force field."""

    position = np.asarray(position_m, dtype=float)
    jacobian = np.empty((3, 3), dtype=float)
    for component in range(3):
        displacement = np.zeros(3)
        displacement[component] = step_m
        jacobian[:, component] = (
            force_function(position + displacement)
            - force_function(position - displacement)
        ) / (2.0 * step_m)
    return jacobian


def _position_equilibrium_record(position_m, force_function) -> dict[str, object]:
    position = np.asarray(position_m, dtype=float)
    force = force_function(position)
    jacobian = _force_jacobian_n_per_m(position, force_function)
    eigenvalues = np.linalg.eigvals(jacobian)
    real_parts = np.real(eigenvalues)
    if np.all(real_parts < 0.0):
        classification = "position-restoring in all three linearized directions"
    elif np.all(real_parts > 0.0):
        classification = "position-anti-restoring in all three linearized directions"
    else:
        classification = "saddle in the linearized static force field"
    return {
        "position_m": [float(value) for value in position],
        "force_n": [float(value) for value in force],
        "force_norm_n": float(np.linalg.norm(force)),
        "force_jacobian_n_per_m": jacobian.tolist(),
        "jacobian_eigenvalues_n_per_m": [
            {
                "real": float(np.real(value)),
                "imaginary": float(np.imag(value)),
            }
            for value in eigenvalues
        ],
        "classification": classification,
        "dynamic_stability_established": False,
    }


def analyze_static_force_equilibria(
    model,
    cooling_repump_beams,
    trapping_beams,
    laser_config,
    stark_config,
    multilevel_config,
    polarizability_table,
) -> dict[str, object]:
    """Classify origin, Cartesian-axis, and body-diagonal static-force zeros."""

    def force_function(position_m):
        return _static_force_vector_n(
            position_m,
            model,
            cooling_repump_beams,
            trapping_beams,
            laser_config,
            stark_config,
            multilevel_config,
            polarizability_table,
        )

    result: dict[str, object] = {
        "method": (
            "nonzero ray roots followed by a 2-um centered finite-difference "
            "Jacobian of the complete three-dimensional static force"
        ),
        "origin": _position_equilibrium_record((0.0, 0.0, 0.0), force_function),
        "positive_cartesian_axis_equilibria": {},
        "body_diagonal_equilibria": [],
    }
    for component, label in enumerate("xyz"):
        ray = np.zeros(3)
        ray[component] = 1.0
        coordinates = _positive_ray_equilibrium_coordinates_m(ray, force_function)
        result["positive_cartesian_axis_equilibria"][label] = [
            _position_equilibrium_record(coordinate * ray, force_function)
            for coordinate in coordinates
        ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                ray = np.asarray((sx, sy, sz))
                coordinates = _positive_ray_equilibrium_coordinates_m(
                    ray,
                    force_function,
                )
                for coordinate in coordinates:
                    result["body_diagonal_equilibria"].append(
                        _position_equilibrium_record(
                            coordinate * ray,
                            force_function,
                        )
                    )
    return result


def run_provisional_stark_trajectory_study(
    *,
    power_w_per_path: float | None = None,
    target_gradient_g_per_cm: float = 20.0,
    duration_s: float = 5.0e-3,
    time_step_s: float = 5.0e-6,
    include_diffusion: bool = False,
    output_tag: str = "provisional_stark_20Gpcm",
) -> dict[str, object]:
    """Run force checks and four short trajectories with explicit caveats."""

    apparatus = default_pmot_apparatus_config()
    multilevel = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        repump_power_w_per_beam=apparatus.mot_light.repump.power_w_per_beam,
    )
    model = build_rate_equation_model(multilevel.natural_linewidth_rad_per_s)
    table = load_differential_polarizability_table()
    if power_w_per_path is None:
        selected_power = provisional_power_for_target_gradient_w_per_path(
            model,
            apparatus.trapping_laser,
            target_gradient_g_per_cm=target_gradient_g_per_cm,
            polarizability_table=table,
        )
        power_source = "derived from provisional stretched-reference vector-gradient proxy"
    else:
        if power_w_per_path < 0.0:
            raise ValueError("power_w_per_path must be non-negative")
        selected_power = power_w_per_path
        power_source = "explicit command-line value"
    stark_config = ProvisionalStarkConfig.uniform_power(selected_power)
    trapping_beams = build_physics_trapping_beams(
        apparatus.trapping_laser,
        stark_config,
    )
    cooling_repump_beams = build_pmot_cooling_and_repump_beams(
        apparatus,
        multilevel,
    )
    paths = pmot_paths()
    data_root = paths["outputs_trajectories"] / output_tag
    figure_root = paths["outputs_figures"] / output_tag
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    print("[pMOT Stark diagnostic] effective detuning equation:", flush=True)
    print(f"  {EFFECTIVE_DETUNING_EQUATION}", flush=True)
    print(f"  {ATOM_FRAME_DOPPLER_EQUATION}", flush=True)
    print(f"  {STARK_DECOMPOSITION_EQUATION}", flush=True)
    print(
        "[pMOT Stark diagnostic] no external field; "
        f"cooling={1e3*apparatus.mot_light.cooling.power_w_per_beam:.3f} mW/beam; "
        f"repump={1e3*apparatus.mot_light.repump.power_w_per_beam:.3f} mW/beam",
        flush=True,
    )
    print(
        "[pMOT Stark diagnostic] trapping path power="
        f"{1e3*selected_power:.6f} mW/path ({power_source})",
        flush=True,
    )
    print(f"[pMOT Stark diagnostic] LIMITATION: {PROVISIONAL_LIMITATION}", flush=True)
    print(
        "[pMOT Stark diagnostic] INHERITED KERNEL LIMITATION: "
        f"{INHERITED_RATE_KERNEL_LIMITATION}",
        flush=True,
    )

    origin = provisional_pmot_observable(
        model,
        cooling_repump_beams,
        trapping_beams,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        apparatus.trapping_laser,
        stark_config,
        multilevel,
        polarizability_table=table,
    )
    reference_index = _cycling_transition_index(model)
    origin_shift_mhz = (
        origin.stark.transition_frequency_shift_hz[reference_index] / 1.0e6
    )
    origin_effective_cooling_detuning_mhz = (
        multilevel.cooling_detuning_rad_per_s / (2.0 * np.pi) / 1.0e6
        - origin_shift_mhz
    )
    print(
        "[pMOT Stark diagnostic] origin: "
        f"I_trap={origin.stark.total_intensity_w_per_m2:.6g} W/m^2; "
        f"cycling-reference shift={origin_shift_mhz:+.6f} MHz; "
        f"effective F=2,m=2->F'=3,m'=3 cooling detuning="
        f"{origin_effective_cooling_detuning_mhz:+.6f} MHz",
        flush=True,
    )

    print("[pMOT Stark diagnostic] sampling static force lineouts (3 x 81 points)", flush=True)
    lineout_start = perf_counter()
    lineouts = sample_force_lineouts(
        model,
        cooling_repump_beams,
        trapping_beams,
        apparatus.trapping_laser,
        stark_config,
        multilevel,
        table,
    )
    lineout_csv = data_root / "force_lineouts.csv"
    pd.DataFrame(lineouts).to_csv(lineout_csv, index=False)
    force_plot = plot_force_lineouts(
        lineouts,
        figure_root / "force_and_stark_shift_lineouts.png",
    )
    lineout_summary = _lineout_summary(lineouts)
    print(
        "[pMOT Stark diagnostic] force lineouts complete in "
        f"{perf_counter()-lineout_start:.2f} s; "
        f"locally_restoring={lineout_summary['all_axes_locally_restoring']}",
        flush=True,
    )
    print(
        "[pMOT Stark diagnostic] classifying static force zeros with full 3D Jacobians",
        flush=True,
    )
    equilibrium_summary = analyze_static_force_equilibria(
        model,
        cooling_repump_beams,
        trapping_beams,
        apparatus.trapping_laser,
        stark_config,
        multilevel,
        table,
    )
    axis_classifications = {
        label: [record["classification"] for record in records]
        for label, records in equilibrium_summary[
            "positive_cartesian_axis_equilibria"
        ].items()
    }
    body_diagonal_records = equilibrium_summary["body_diagonal_equilibria"]
    body_diagonal_restoring_count = sum(
        record["classification"]
        == "position-restoring in all three linearized directions"
        for record in body_diagonal_records
    )
    print(
        "[pMOT Stark diagnostic] 3D equilibrium check: "
        f"axis={axis_classifications}; "
        f"body_diagonal_count={len(body_diagonal_records)}, "
        f"position_restoring_body_diagonal_count={body_diagonal_restoring_count}",
        flush=True,
    )

    cases = (
        ("x_offset", (1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("y_offset", (0.0, 1.0e-3, 0.0), (0.0, 0.0, 0.0)),
        ("z_offset", (0.0, 0.0, 1.0e-3), (0.0, 0.0, 0.0)),
        ("diagonal", (0.8e-3, -0.7e-3, 0.9e-3), (-0.15, 0.10, -0.05)),
    )
    outputs = [str(lineout_csv), str(force_plot)]
    trajectory_summaries: dict[str, object] = {}
    for case_index, (name, position, velocity) in enumerate(cases, start=1):
        step_ratio = duration_s / time_step_s
        rounded_steps = int(round(step_ratio))
        steps = (
            rounded_steps
            if abs(step_ratio - rounded_steps) <= 1.0e-10 * max(1.0, abs(step_ratio))
            else int(np.ceil(step_ratio))
        )
        print(
            f"[pMOT Stark diagnostic] trajectory {case_index}/{len(cases)} {name}: "
            f"T={1e3*duration_s:.3f} ms, dt={1e6*time_step_s:.3f} us, "
            f"steps={steps:,}",
            flush=True,
        )
        started = perf_counter()

        def progress(completed, expected, current_time, *, case=name):
            percent = min(100.0, 100.0 * completed / max(expected, 1))
            print(
                f"[pMOT Stark diagnostic] {case}: {percent:5.1f}% "
                f"(t={1e3*current_time:.3f} ms)",
                flush=True,
            )

        record = simulate_provisional_pmot_trajectory(
            RateEquationAtomState(position, velocity),
            duration_s,
            model,
            cooling_repump_beams,
            trapping_beams,
            apparatus.trapping_laser,
            stark_config,
            multilevel,
            trajectory_config=RateEquationTrajectoryConfig(
                time_step_s=time_step_s,
                include_diffusion=include_diffusion,
                seed=20260831 + case_index,
                escape_radius_m=30.0e-3,
            ),
            polarizability_table=table,
            progress_callback=progress,
        )
        summary = summarize_trajectory(record)
        summary["wall_time_s"] = perf_counter() - started
        trajectory_summaries[name] = summary
        csv_path = save_trajectory_csv(record, data_root / f"{name}.csv")
        title = (
            f"Provisional pMOT trajectory: {name} | "
            f"{1e3*selected_power:.3f} mW/path, no external B"
        )
        time_plot = plot_trajectory_time_diagnostics(
            record,
            cooling_repump_beams,
            title,
            figure_root / f"{name}_time_diagnostics.png",
        )
        stark_plot = plot_trajectory_stark_diagnostics(
            record,
            title,
            figure_root / f"{name}_stark_diagnostics.png",
        )
        outputs.extend((str(csv_path), str(time_plot), str(stark_plot)))
        print(
            f"[pMOT Stark diagnostic] trajectory {case_index}/{len(cases)} {name} complete: "
            f"termination={summary['termination_reason']}, "
            f"max_radius={1e3*summary['maximum_radius_m']:.4f} mm, "
            f"final_speed={summary['final_speed_m_per_s']:.5f} m/s, "
            f"wall={summary['wall_time_s']:.2f} s",
            flush=True,
        )

    print(
        "[pMOT Stark diagnostic] timestep check: repeating x_offset at "
        f"{1e6*0.5*time_step_s:.3f} us and {1e6*0.25*time_step_s:.3f} us",
        flush=True,
    )
    if include_diffusion:
        convergence_baseline = simulate_provisional_pmot_trajectory(
            RateEquationAtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
            duration_s,
            model,
            cooling_repump_beams,
            trapping_beams,
            apparatus.trapping_laser,
            stark_config,
            multilevel,
            trajectory_config=RateEquationTrajectoryConfig(
                time_step_s=time_step_s,
                include_diffusion=False,
                seed=20260832,
                escape_radius_m=30.0e-3,
            ),
            polarizability_table=table,
        )
        convergence_baseline_summary = summarize_trajectory(convergence_baseline)
    else:
        convergence_baseline_summary = trajectory_summaries["x_offset"]
    convergence_rows = [
        {
            "time_step_s": time_step_s,
            "diffusion_enabled": False,
            "final_position_m": convergence_baseline_summary["final_position_m"],
            "final_velocity_m_per_s": convergence_baseline_summary["final_velocity_m_per_s"],
            "maximum_radius_m": convergence_baseline_summary["maximum_radius_m"],
            "maximum_speed_m_per_s": convergence_baseline_summary["maximum_speed_m_per_s"],
        }
    ]
    for refined_dt_s in (0.5 * time_step_s, 0.25 * time_step_s):
        refined = simulate_provisional_pmot_trajectory(
            RateEquationAtomState((1.0e-3, 0.0, 0.0), (0.0, 0.0, 0.0)),
            duration_s,
            model,
            cooling_repump_beams,
            trapping_beams,
            apparatus.trapping_laser,
            stark_config,
            multilevel,
            trajectory_config=RateEquationTrajectoryConfig(
                time_step_s=refined_dt_s,
                include_diffusion=False,
                seed=20260832,
                escape_radius_m=30.0e-3,
            ),
            polarizability_table=table,
        )
        refined_summary = summarize_trajectory(refined)
        convergence_rows.append(
            {
                "time_step_s": refined_dt_s,
                "diffusion_enabled": False,
                "final_position_m": refined_summary["final_position_m"],
                "final_velocity_m_per_s": refined_summary["final_velocity_m_per_s"],
                "maximum_radius_m": refined_summary["maximum_radius_m"],
                "maximum_speed_m_per_s": refined_summary["maximum_speed_m_per_s"],
            }
        )
        print(
            "[pMOT Stark diagnostic] timestep check complete: "
            f"dt={1e6*refined_dt_s:.3f} us, "
            f"max_radius={1e3*refined_summary['maximum_radius_m']:.6f} mm",
            flush=True,
        )
    finest_position = np.asarray(convergence_rows[-1]["final_position_m"], dtype=float)
    finest_velocity = np.asarray(convergence_rows[-1]["final_velocity_m_per_s"], dtype=float)
    for row in convergence_rows:
        row["final_position_error_vs_finest_m"] = float(
            np.linalg.norm(np.asarray(row["final_position_m"]) - finest_position)
        )
        row["final_velocity_error_vs_finest_m_per_s"] = float(
            np.linalg.norm(np.asarray(row["final_velocity_m_per_s"]) - finest_velocity)
        )
    convergence_path = data_root / "timestep_convergence.json"
    convergence_path.write_text(
        json.dumps(convergence_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(str(convergence_path))

    metadata = {
        "status": "provisional diagnostic; not a validated pMOT trapping prediction",
        "model": PROVISIONAL_MODEL_NAME,
        "effective_detuning_equation": EFFECTIVE_DETUNING_EQUATION,
        "atom_frame_doppler_equation": ATOM_FRAME_DOPPLER_EQUATION,
        "stark_decomposition_equation": STARK_DECOMPOSITION_EQUATION,
        "provisional_limitation": PROVISIONAL_LIMITATION,
        "inherited_rate_kernel_limitation": INHERITED_RATE_KERNEL_LIMITATION,
        "omitted_physics": list(OMITTED_PHYSICS),
        "external_magnetic_field_t": [0.0, 0.0, 0.0],
        "cooling_power_w_per_beam": apparatus.mot_light.cooling.power_w_per_beam,
        "repump_power_w_per_beam": apparatus.mot_light.repump.power_w_per_beam,
        "cooling_detuning_hz": apparatus.mot_light.cooling.detuning_hz,
        "trapping_wavelength_m": apparatus.trapping_laser.wavelength_m,
        "trapping_power_w_per_path": selected_power,
        "trapping_power_source": power_source,
        "target_gradient_g_per_cm": (
            target_gradient_g_per_cm if power_w_per_path is None else None
        ),
        "stark_config": asdict(stark_config),
        "multilevel_config": asdict(multilevel),
        "duration_s": duration_s,
        "time_step_s": time_step_s,
        "include_diffusion": include_diffusion,
        "polarizability_csv": str(table.source_path),
        "polarizability_wavelength_range_nm": list(table.wavelength_range_nm),
        "origin_total_trapping_intensity_w_per_m2": origin.stark.total_intensity_w_per_m2,
        "origin_reference_shift_mhz": origin_shift_mhz,
        "origin_reference_effective_cooling_detuning_mhz": (
            origin_effective_cooling_detuning_mhz
        ),
        "lineout_summary": lineout_summary,
        "static_force_equilibrium_summary": equilibrium_summary,
        "trajectory_summaries": trajectory_summaries,
        "timestep_convergence": convergence_rows,
        "outputs": outputs,
    }
    metadata_path = data_root / "study_summary.json"
    outputs.append(str(metadata_path))
    metadata["outputs"] = outputs
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[pMOT Stark diagnostic] complete; summary={metadata_path}", flush=True)
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explicitly provisional no-coil pMOT Stark diagnostics"
    )
    parser.add_argument(
        "--power-mw-per-path",
        type=float,
        default=None,
        help="absolute launched path power; omit to use the 20 G/cm proxy scale",
    )
    parser.add_argument("--target-gradient-g-per-cm", type=float, default=20.0)
    parser.add_argument("--duration-ms", type=float, default=5.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--include-diffusion", action="store_true")
    parser.add_argument("--output-tag", default="provisional_stark_20Gpcm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_provisional_stark_trajectory_study(
        power_w_per_path=(
            None
            if args.power_mw_per_path is None
            else 1.0e-3 * args.power_mw_per_path
        ),
        target_gradient_g_per_cm=args.target_gradient_g_per_cm,
        duration_s=1.0e-3 * args.duration_ms,
        time_step_s=1.0e-6 * args.dt_us,
        include_diffusion=args.include_diffusion,
        output_tag=args.output_tag,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
